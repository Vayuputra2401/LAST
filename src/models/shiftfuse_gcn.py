"""
ShiftFuse-GCN (LAST-Lite): Lightweight Skeleton Action Recognition.

Architecture summary
--------------------
  4 Streams (joint, velocity, bone, bone_velocity): each (B, 3, T, V)
        |
  StreamFusionConcat — per-stream BN + Concat(12ch) + Conv1×1 → C₀
        |
  Stage 1: [ShiftFuseBlock × num_blocks₁]  stride=1  → CTRLightGCN (shared)
        |
  Stage 2: [ShiftFuseBlock × num_blocks₂]  stride=2  → CTRLightGCN (shared)
        |
  Stage 3: [ShiftFuseBlock × num_blocks₃]  stride=2  → CTRLightGCN (shared)
        |
  Gated GAP+GMP Pool → BN1d → Dropout → FC(C₃ → num_classes)

ShiftFuseBlock (per block)
--------------------------
  BodyRegionShift   (BRASP — 0 params, anatomical channel permutation)
  Conv2d(C,C,1×1) + BN + Hardswish  (pointwise channel mixing)
  JointEmbedding    (additive per-joint semantic bias, V×C params)
  BSE               (Bilateral Symmetry Encoding — 2C+1 params)
  FrozenDCTGate     (FDCR — learnable frequency mask, C×T params)
  EpSepTCN / MultiScaleEpSepTCN  (temporal convolution)
  DropPath          (stochastic depth, linear rate schedule)
  Outer residual    (Conv1×1+BN if channel/stride mismatch, else Identity)
  Backbone Dropout  (intermediate feature regularization)
  CTRLightGCN       (shared across stage — channel-group topology refinement)

AdaptiveCTRGCN (one per stage, shared weight) — v5 upgrade
-----------------------------------------------------------
  For each group g:
    Q_g, K_g = learned projections of x_group_g, pooled over T
    A_adaptive_g = softmax(Q_g^T @ K_g / √d)   (per-sample topology)
    A_g = A_physical + α_g * A_adaptive_g
    h_g = GroupConv_g(A_g @ x_group_g)
  out = x + BN(concat([h_0, ..., h_{G-1}]))

Novel contributions
-------------------
  BRASP:             Anatomically-partitioned channel shift, 0 params.
  BSE:               Bilateral symmetry encoding — L-R joint diff + dynamics.
  FDCR:              Fixed-compute frequency-domain channel specialisation.
  AdaptiveCTRGCN:    Per-sample channel-group topology refinement (v5 upgrade).
  TemporalAttention: Lightweight global temporal context via self-attention.
  IB Loss:           Information bottleneck auxiliary loss at the head.
  Training:          DropPath + Mixup/CutMix for ultra-compact skeleton GCNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.body_region_shift import BodyRegionShift
from .blocks.frozen_dct_gate import FrozenDCTGate
from .blocks.joint_embedding import JointEmbedding
from .blocks.frame_dynamics_gate import FrameDynamicsGate
from .blocks.bilateral_symmetry import BilateralSymmetryEncoding
from .blocks.ep_sep_tcn import EpSepTCN, MultiScaleEpSepTCN
from .blocks.stream_fusion_concat import StreamFusionConcat
from .blocks.adaptive_ctr_gcn import AdaptiveCTRGCN
from .blocks.temporal_attention import LightweightTemporalAttention
from .blocks.drop_path import DropPath
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
MODEL_VARIANTS_SHIFTFUSE = {
    'nano': {
        'stem_channels':      24,
        'channels':           [32, 48, 64],
        'num_blocks':         [1, 1, 1],
        'strides':            [1, 2, 2],
        'expand_ratio':       2,
        'max_hop':            1,
        'use_dct_gate':       True,
        'use_joint_embed':    True,
        'use_frame_gate':     False,   # Removed: redundant with FrozenDCTGate
        'use_bilateral':      True,
        'use_multiscale_tcn': False,   # EpSepTCN for nano (param budget)
        'num_tcn_branches':   2,       # unused when use_multiscale_tcn=False
        'num_gcn_groups':     2,       # CTRLightGCN groups (nano budget)
        'drop_path_rate':     0.10,    # max stochastic depth rate
        'dropout':            0.10,    # classifier head dropout (v5: reduced from 0.15)
    },
    'small': {
        'stem_channels':      32,
        'channels':           [48, 72, 96],
        'num_blocks':         [1, 2, 2],
        'strides':            [1, 2, 2],
        'expand_ratio':       2,
        'max_hop':            2,
        'use_dct_gate':       True,
        'use_joint_embed':    True,
        'use_frame_gate':     False,   # Removed: redundant with FrozenDCTGate
        'use_bilateral':      True,
        'use_multiscale_tcn': True,    # Multi-scale k=3/k=5/MaxPool for small
        'num_tcn_branches':   3,
        'num_gcn_groups':     4,       # CTRLightGCN groups (full)
        'drop_path_rate':     0.15,    # max stochastic depth rate
        'dropout':            0.20,    # classifier head dropout (v5: reduced from 0.30)
    },
}


# ---------------------------------------------------------------------------
# ShiftFuseBlock
# ---------------------------------------------------------------------------
class ShiftFuseBlock(nn.Module):
    """
    One block of ShiftFuse-GCN.

    Pipeline:
        BodyRegionShift → Conv1×1+BN+Hardswish → JointEmbedding → BSE
        → FrozenDCTGate → EpSepTCN (or MultiScaleEpSepTCN)
        → DropPath → Outer residual
        → AdaptiveCTRGCN (shared, optional)
        → LightweightTemporalAttention (optional)

    AdaptiveCTRGCN is passed in by LAST_Lite — one instance shared across all
    blocks in the same stage, so graph weights are not duplicated.
    LightweightTemporalAttention is created per-block (lightweight, not shared).

    Args:
        in_channels:        Input channels.
        out_channels:       Output channels.
        A_flat:             (V, V) flat adjacency for BodyRegionShift init.
        T:                  Temporal length (for FrozenDCTGate).
        stride:             Temporal stride applied by temporal conv (default 1).
        expand_ratio:       EpSepTCN expansion ratio (default 2).
        num_joints:         Number of skeleton joints (default 25).
        use_dct_gate:       Include FrozenDCTGate (default True).
        use_joint_embed:    Include JointEmbedding (default True).
        use_frame_gate:     Include FrameDynamicsGate (default False).
        use_bilateral:      Include BilateralSymmetryEncoding (default True).
        use_multiscale_tcn: Use MultiScaleEpSepTCN instead of EpSepTCN.
        num_tcn_branches:   Number of TCN branches (2 or 3, for multiscale only).
        drop_path_prob:     DropPath probability for this block (default 0.0).
        gcn:                Optional AdaptiveCTRGCN (shared reference from LAST_Lite).
        use_temporal_attn:  Include LightweightTemporalAttention (default True).
        temporal_attn_r:    Reduce ratio for temporal attention Q/K/V.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_flat: torch.Tensor,
        T: int,
        stride: int = 1,
        expand_ratio: int = 2,
        num_joints: int = 25,
        use_dct_gate: bool = True,
        use_joint_embed: bool = True,
        use_frame_gate: bool = False,
        use_bilateral: bool = True,
        use_multiscale_tcn: bool = False,
        num_tcn_branches: int = 3,
        drop_path_prob: float = 0.0,
        gcn: nn.Module = None,
        use_temporal_attn: bool = True,
        temporal_attn_r: int = 4,
    ):
        super().__init__()

        # 1. Body-Region-Aware Spatial Shift (BRASP) — 0 params
        self.shift = BodyRegionShift(in_channels, A_flat)

        # 2. Pointwise channel mixing
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

        # 3. Joint semantic embedding (SGN-style per-joint additive bias)
        self.joint_embed = (
            JointEmbedding(out_channels, num_joints) if use_joint_embed
            else nn.Identity()
        )

        # 3b. Bilateral Symmetry Encoding (BSE — novel, 2C+1 params)
        self.bilateral = (
            BilateralSymmetryEncoding(out_channels) if use_bilateral
            else nn.Identity()
        )

        # 4. Frozen DCT frequency gate (FDCR — C×T params, residual)
        self.dct_gate = (
            FrozenDCTGate(out_channels, T) if use_dct_gate
            else nn.Identity()
        )

        # 5. Temporal convolution
        if use_multiscale_tcn:
            self.tcn = MultiScaleEpSepTCN(
                out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                num_branches=num_tcn_branches,
            )
        else:
            self.tcn = EpSepTCN(
                out_channels,
                kernel_size=5,
                stride=stride,
                expand_ratio=expand_ratio,
            )

        # 6. Frame dynamics gate (disabled by default — redundant with FrozenDCTGate)
        T_out = T // stride
        self.frame_gate = (
            FrameDynamicsGate(out_channels, T_out) if use_frame_gate
            else nn.Identity()
        )

        # 7. DropPath (stochastic depth) — applied to main path before residual
        self.drop_path = DropPath(drop_path_prob)

        # 8. Outer residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # 9. Shared AdaptiveCTRGCN — stored as plain attribute (not registered as
        #     a submodule) so that LAST_Lite owns and registers it once per stage.
        object.__setattr__(self, 'gcn', gcn)

        # 10. Lightweight temporal attention (global temporal context)
        T_out = T // stride
        self.temporal_attn = (
            LightweightTemporalAttention(out_channels, reduce_ratio=temporal_attn_r)
            if use_temporal_attn else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T, V)
        Returns:
            out: (B, out_channels, T', V)  where T' = T // stride
        """
        res = self.residual(x)

        out = self.shift(x)
        out = self.pw_conv(out)
        out = self.joint_embed(out)
        out = self.bilateral(out)
        out = self.dct_gate(out)
        out = self.tcn(out)
        out = self.frame_gate(out)

        out = self.drop_path(out)        # stochastic depth on main path
        out = res + out                  # block outer residual — identity untouched

        if self.gcn is not None:
            out = self.gcn(out)          # spatial refinement (AdaptiveCTRGCN has own residual)

        out = self.temporal_attn(out)    # global temporal context

        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LAST_Lite(nn.Module):
    """
    LAST-Lite / ShiftFuse-GCN: fixed-compute edge skeleton action recogniser.

    Args:
        num_classes:    Number of output classes (60 or 120).
        variant:        'nano' | 'small'.
        in_channels:    Input channels per stream (default 3 for x,y,z).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy: Adjacency strategy (default 'spatial').
        T:              Temporal length after preprocessing (default 64).
        num_joints:     Number of skeleton joints (default 25).
        dropout:        Override classifier dropout (None = variant default).
    """

    def __init__(
        self,
        num_classes: int = 60,
        variant: str = 'small',
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        T: int = 64,
        num_joints: int = 25,
        dropout: float = None,
    ):
        super().__init__()

        if variant not in MODEL_VARIANTS_SHIFTFUSE:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(MODEL_VARIANTS_SHIFTFUSE.keys())}"
            )

        cfg = MODEL_VARIANTS_SHIFTFUSE[variant]
        stem_ch           = cfg['stem_channels']
        channels          = cfg['channels']
        num_blocks        = cfg['num_blocks']
        strides           = cfg['strides']
        expand_ratio      = cfg['expand_ratio']
        max_hop           = cfg['max_hop']
        use_dct_gate      = cfg['use_dct_gate']
        use_joint_embed   = cfg['use_joint_embed']
        use_frame_gate    = cfg['use_frame_gate']
        use_bilateral     = cfg.get('use_bilateral', True)
        use_multiscale    = cfg.get('use_multiscale_tcn', False)
        num_tcn_branches  = cfg.get('num_tcn_branches', 3)
        num_gcn_groups    = cfg.get('num_gcn_groups', 4)
        drop_path_rate    = cfg.get('drop_path_rate', 0.0)
        _dropout = dropout if dropout is not None else cfg['dropout']

        self.variant = variant
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']

        # ── 1. Graph adjacency ───────────────────────────────────────────
        self.graph = Graph(
            layout=graph_layout,
            strategy=graph_strategy,
            max_hop=max_hop,
            raw_partitions=True,
        )
        A_raw = self.graph.A                        # (K, V, V) — raw 0/1 partitions
        A_sym = normalize_symdigraph_full(A_raw)    # (K, V, V) — D^{-1/2}AD^{-1/2}
        A = torch.tensor(A_sym, dtype=torch.float32)
        self.register_buffer('A', A)

        # Flat (V, V) adjacency for BodyRegionShift neighbour lookup
        A_flat = torch.tensor(
            (A_raw.sum(0) > 0).astype('float32')
        )

        # ── 2. Stream fusion (EfficientGCN-exact concat) ─────────────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ──────────────────────────────────────────────
        # DropPath rate linearly increases from 0 → drop_path_rate across all blocks
        total_blocks = sum(num_blocks)
        block_idx_global = 0

        # One AdaptiveCTRGCN per stage, shared by all blocks within that stage.
        self.stages     = nn.ModuleList()
        self.stage_gcns = nn.ModuleList()
        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            # Create the shared GCN for this stage
            stage_gcn = AdaptiveCTRGCN(
                channels=stage_ch,
                A=A,
                num_joints=num_joints,
                num_groups=num_gcn_groups,
            )
            self.stage_gcns.append(stage_gcn)

            stage_blocks = nn.ModuleList()
            for blk_idx in range(num_blocks[stage_idx]):
                blk_in     = prev_ch   if blk_idx == 0 else stage_ch
                blk_out    = stage_ch
                blk_stride = strides[stage_idx] if blk_idx == 0 else 1

                # Linear DropPath schedule: 0.0 → drop_path_rate
                dp_rate = (
                    drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                    if total_blocks > 1 else 0.0
                )
                block_idx_global += 1

                stage_blocks.append(ShiftFuseBlock(
                    in_channels=blk_in,
                    out_channels=blk_out,
                    A_flat=A_flat,
                    T=T_cur,
                    stride=blk_stride,
                    expand_ratio=expand_ratio,
                    num_joints=num_joints,
                    use_dct_gate=use_dct_gate,
                    use_joint_embed=use_joint_embed,
                    use_frame_gate=use_frame_gate,
                    use_bilateral=use_bilateral,
                    use_multiscale_tcn=use_multiscale,
                    num_tcn_branches=num_tcn_branches,
                    drop_path_prob=dp_rate,
                    gcn=stage_gcn,
                ))

                if blk_idx == 0:
                    T_cur = T_cur // blk_stride

            self.stages.append(stage_blocks)
            prev_ch = stage_ch

        # ── 4. Head: Gated GAP+GMP + classifier + IB loss ─────────────────
        last_ch = channels[-1]
        self.pool_gate = nn.Parameter(torch.zeros(1, last_ch, 1, 1))
        self.head_bn   = nn.BatchNorm1d(last_ch)
        self.drop      = nn.Dropout(_dropout)
        self.fc        = nn.Linear(last_ch, num_classes)

        # IB loss: class-conditional prototypes (InfoGCN-inspired)
        # Each class has a prototype vector. IB loss = mean distance from
        # features to nearest prototype → features cluster, better generalization.
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, last_ch) * 0.01
        )

        # ── Weight initialisation ────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: Dict with keys 'joint', 'velocity', 'bone', 'bone_velocity'
               Each value shape: (B, C, T, V) or (B, C, T, V, M).
               OR a single Tensor (B, C, T, V) — used for all 4 streams.
        Returns:
            During training:  (logits, ib_loss)
            During eval:      logits  (B, num_classes)
        """
        # ── Input handling ──────────────────────────────────────────────
        if isinstance(x, dict):
            streams = []
            for name in self.stream_names:
                if name in x:
                    s = x[name]
                    if s.dim() == 5:
                        s = s[..., 0]       # take primary body (M=0)
                    streams.append(s)
            while len(streams) < 4:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x, x, x, x]

        # ── Fuse streams → (B, stem_ch, T, V) ───────────────────────────
        out = self.fusion(streams)

        # ── Backbone stages ──────────────────────────────────────────────
        for stage_blocks in self.stages:
            for block in stage_blocks:
                out = block(out)

        # ── Gated GAP+GMP pool ───────────────────────────────────────────
        gate   = torch.sigmoid(self.pool_gate)
        gap    = F.adaptive_avg_pool2d(out, (1, 1))
        gmp    = F.adaptive_max_pool2d(out, (1, 1))
        pooled = gap * gate + gmp * (1 - gate)
        pooled = pooled.view(pooled.size(0), -1)   # (B, C)

        # ── Classifier ───────────────────────────────────────────────────
        features = self.head_bn(pooled)
        logits = self.fc(self.drop(features))

        if self.training:
            # IB loss: mean distance from features to nearest class prototype
            # Encourages compact, class-discriminative feature clusters.
            # (B, num_classes) pairwise L2 distances
            proto_dists = torch.cdist(
                features.unsqueeze(0),
                self.class_prototypes.unsqueeze(0),
                p=2,
            ).squeeze(0)
            # Mean of min distance per sample → pull features toward nearest prototype
            ib_loss = proto_dists.min(dim=-1).values.mean()
            return logits, ib_loss

        return logits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_shiftfuse_nano(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='nano', **kwargs)

def create_shiftfuse_small(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='small', **kwargs)
