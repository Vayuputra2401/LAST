"""
ShiftFuse-GCN (LAST-Lite v7): Lightweight Skeleton Action Recognition.

Architecture summary (v7 — 4-stream late fusion + spatial-first blocks)
------------------------------------------------------------------------
  4 Streams: joint / velocity / bone / bone_velocity  each (B, 3, T, V)
        |
  MultiStreamStem — 4 independent BN+Conv1×1 stems (one per stream)
                    streams stacked along batch dim → (4B, C0, T, V)
        |
  Shared Backbone:
    Stage 1: [ShiftFuseBlock × n₁]  stride=1  → MultiScaleAdaptiveGCN (shared)
    Stage 2: [ShiftFuseBlock × n₂]  stride=2  → MultiScaleAdaptiveGCN (shared)
    Stage 3: [ShiftFuseBlock × n₃]  stride=2  → MultiScaleAdaptiveGCN (shared)
        |
  Split batch back → 4 × (B, C₃, T', V')
        |
  4 × ClassificationHead (Gated GAP+GMP → BN → Dropout → FC)
        |
  Ensemble: softmax_weighted average of 4 logit vectors → (B, num_classes)

ShiftFuseBlock (per block) — v7: SPATIAL-FIRST order (ST-GCN paradigm)
-----------------------------------------------------------------------
  BodyRegionShift   (BRASP — 0 params)
  Conv2d(C,C,1×1) + BN + Hardswish   (channel mixing)
  MultiScaleAdaptiveGCN (shared)      ← SPATIAL FIRST (moved from after residual)
  ChannelSE                           ← channel recalibration after GCN (EfficientGCN)
  JointEmbedding    (V×C params)
  BSE               (2C+1 params)
  FrozenDCTGate     (C×T params)
  EpSepTCN / MultiScaleEpSepTCN      ← TEMPORAL SECOND
  DropPath
  Outer residual (clean single residual, no double-compound)
  LightweightTemporalAttention

MultiScaleAdaptiveGCN (one per stage, shared across blocks in stage)
--------------------------------------------------------------------
  K subsets kept SEPARATE (ST-GCN paradigm) — not summed.
  Each subset: its own group conv. Shared per-sample adaptive topology (Q/K).
  K=3 subsets × G group convs each → K×G total spatial operations.

v7 changes over v5/v6
---------------------
  1. StreamFusionConcat → MultiStreamStem (4-stream late fusion, +3–4%)
  2. GCN moved before TCN (spatial-first, clean gradient path, +0.5–1%)
  3. ChannelSE after GCN (EfficientGCN channel recalibration, +0.5–1%)
  4. AdaptiveCTRGCN → MultiScaleAdaptiveGCN (K separate subsets, +0.5–1%)
  5. small blocks [1,2,2] → [1,3,3] (within 300K budget, +0.5%)
  6. nano G: 2 → 4 (more adaptive topology, within 100K budget)
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
from .blocks.stream_fusion_concat import MultiStreamStem
from .blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from .blocks.temporal_attention import LightweightTemporalAttention
from .blocks.drop_path import DropPath
from .blocks.channel_se import ChannelSE
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configurations (v7)
# ---------------------------------------------------------------------------
MODEL_VARIANTS_SHIFTFUSE = {
    'nano': {
        'stem_channels':      24,
        'channels':           [32, 64, 128],  # v9: 1:2:4 ratio (EfficientGCN-B0 scale), no param limit
        'num_blocks':         [1, 1, 1],
        'strides':            [1, 2, 2],
        'expand_ratio':       2,
        'max_hop':            1,
        'use_dct_gate':       False,           # v9: removed — TemporalAttn covers global temporal
        'use_joint_embed':    True,
        'use_frame_gate':     False,
        'use_bilateral':      False,           # v9: removed — zero-init no-op; bone stream covers L-R
        'use_multiscale_tcn': True,
        'num_tcn_branches':   4,               # TSM + d=2 + d=4 + MaxPool
        'num_gcn_groups':     4,
        'drop_path_rate':     0.10,
        'dropout':            0.10,
    },
    'small': {
        'stem_channels':      32,
        'channels':           [64, 128, 256],  # v9: 1:2:4 ratio matching EfficientGCN-B0, no param limit
        'num_blocks':         [1, 2, 3],
        'strides':            [1, 2, 2],
        'expand_ratio':       2,
        'max_hop':            2,
        'use_dct_gate':       False,            # v9: removed
        'use_joint_embed':    True,
        'use_frame_gate':     False,
        'use_bilateral':      False,            # v9: removed
        'use_multiscale_tcn': True,
        'num_tcn_branches':   4,
        'num_gcn_groups':     4,
        'drop_path_rate':     0.15,
        'dropout':            0.20,
    },
}


# ---------------------------------------------------------------------------
# Classification head (one per stream)
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """Gated GAP+GMP → BN → Dropout → FC.

    Args:
        channels:    Input feature channels (last backbone stage).
        num_classes: Number of output classes.
        dropout:     Dropout probability.
    """

    def __init__(self, channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pool_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bn        = nn.BatchNorm1d(channels)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T', V') — backbone output for one stream.
        Returns:
            logits:   (B, num_classes)
            features: (B, C)  — pooled features (used for IB loss)
        """
        gate   = torch.sigmoid(self.pool_gate)
        gap    = F.adaptive_avg_pool2d(x, (1, 1))
        gmp    = F.adaptive_max_pool2d(x, (1, 1))
        pooled = (gap * gate + gmp * (1 - gate)).view(x.size(0), -1)
        features = self.bn(pooled)
        logits   = self.fc(self.drop(features))
        return logits, features


# ---------------------------------------------------------------------------
# ShiftFuseBlock (v7: spatial-first order)
# ---------------------------------------------------------------------------
class ShiftFuseBlock(nn.Module):
    """
    One block of ShiftFuse-GCN v7.

    Forward order (SPATIAL-FIRST — ST-GCN / EfficientGCN paradigm):
        BodyRegionShift → Conv1×1+BN+Hardswish
        → GCN (spatial)       ← moved before TCN; clean gradient path
        → ChannelSE            ← channel recalibration after graph propagation
        → JointEmbedding → BSE → FrozenDCTGate
        → EpSepTCN (temporal) ← temporal second
        → DropPath → Outer residual (single, no double-compound)
        → LightweightTemporalAttention

    Args:
        in_channels:        Input channels.
        out_channels:       Output channels.
        A_flat:             (V, V) flat adjacency for BodyRegionShift.
        T:                  Temporal length (for FrozenDCTGate).
        stride:             Temporal stride (default 1).
        expand_ratio:       EpSepTCN expansion ratio (default 2).
        num_joints:         Skeleton joints (default 25).
        use_dct_gate:       Include FrozenDCTGate (default True).
        use_joint_embed:    Include JointEmbedding (default True).
        use_frame_gate:     Include FrameDynamicsGate (default False).
        use_bilateral:      Include BilateralSymmetryEncoding (default True).
        use_multiscale_tcn: Use MultiScaleEpSepTCN instead of EpSepTCN.
        num_tcn_branches:   Number of TCN branches (multiscale only).
        drop_path_prob:     DropPath probability for this block.
        gcn:                Optional MultiScaleAdaptiveGCN (shared by stage).
        use_temporal_attn:  Include LightweightTemporalAttention (default True).
        temporal_attn_r:    Reduce ratio for temporal attention.
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
        brasp_after_pw: bool = False,
        register_gcn: bool = False,
        je: nn.Module = None,        # optional shared JointEmbedding (parent owns it)
    ):
        super().__init__()
        self.brasp_after_pw = brasp_after_pw

        # 1. Body-Region-Aware Spatial Shift (BRASP) — 0 params
        # brasp_after_pw=True: BRASP uses out_channels (runs after pw_conv,
        #   so channel-region mapping matches the GCN's operating channels).
        brasp_ch = out_channels if brasp_after_pw else in_channels
        self.shift = BodyRegionShift(brasp_ch, A_flat)

        # 2. Pointwise channel projection — only when dimensions change.
        # Within-stage blocks (in==out, stride==1) skip this: GCN directly receives BRASP output,
        # avoiding a redundant C² mix before graph aggregation (EfficientGCN/ST-GCN don't have it).
        if in_channels != out_channels or stride != 1:
            self.pw_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(inplace=True),
            )
        else:
            self.pw_conv = nn.Identity()

        # 3. GCN — either shared (plain attribute, parent owns) or per-block (submodule)
        if register_gcn:
            self.gcn = gcn          # block owns this GCN (per-block mode)
        else:
            object.__setattr__(self, 'gcn', gcn)   # shared reference, NOT registered

        # 4. Channel SE — recalibrates after graph propagation (EfficientGCN-style)
        self.se = ChannelSE(out_channels)

        # 5. Joint semantic embedding (SGN-style per-joint additive bias)
        # If a shared `je` is passed in, use it unregistered (parent stage owns it).
        if je is not None:
            object.__setattr__(self, 'joint_embed', je)
        else:
            self.joint_embed = (
                JointEmbedding(out_channels, num_joints) if use_joint_embed
                else nn.Identity()
            )

        # 6. Bilateral Symmetry Encoding (BSE)
        self.bilateral = (
            BilateralSymmetryEncoding(out_channels) if use_bilateral
            else nn.Identity()
        )

        # 7. Frozen DCT frequency gate (FDCR)
        self.dct_gate = (
            FrozenDCTGate(out_channels, T) if use_dct_gate
            else nn.Identity()
        )

        # 8. Temporal convolution — TEMPORAL SECOND
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

        # 9. Frame dynamics gate (disabled — redundant with FrozenDCTGate)
        T_out = T // stride
        self.frame_gate = (
            FrameDynamicsGate(out_channels, T_out) if use_frame_gate
            else nn.Identity()
        )

        # 10. DropPath (stochastic depth on main path before residual)
        self.drop_path = DropPath(drop_path_prob)

        # 11. Outer residual (single — no double-compound with GCN's own residual)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # 12. Lightweight temporal attention (global temporal context, per-block)
        T_out = T // stride
        self.temporal_attn = (
            LightweightTemporalAttention(out_channels, reduce_ratio=temporal_attn_r)
            if use_temporal_attn else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, in_channels, T, V)
        Returns:
            out: (B, out_channels, T', V)  T' = T // stride

        v9 block order (principled, non-redundant):
            BRASP → [pw_conv if in≠out] → JointEmbed → GCN → SE → TCN → DropPath → res+out → TemporalAttn
        """
        res = self.residual(x)

        # ── Spatial path ────────────────────────────────────────────────
        if self.brasp_after_pw:
            out = self.pw_conv(x)        # project first (in→out channels)
            out = self.shift(out)        # BRASP on out_channels (correct mapping)
        else:
            out = self.shift(x)          # BRASP on in_channels (v7 default)
            out = self.pw_conv(out)      # dim expansion (Identity when in==out)
        out = self.joint_embed(out)      # joint semantic identity BEFORE GCN aggregation
        if self.gcn is not None:
            out = self.gcn(out)          # K-subset GCN on joint-aware features
        out = self.se(out)               # channel recalibration after graph prop

        # ── Temporal path ───────────────────────────────────────────────
        out = self.tcn(out)              # 4-branch local temporal (TSM/d=2/d=4/MaxPool)

        out = self.drop_path(out)        # stochastic depth on main path
        out = res + out                  # single clean outer residual

        out = self.temporal_attn(out)    # global temporal context (gate=0.5, active from ep1)

        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LAST_Lite(nn.Module):
    """
    LAST-Lite / ShiftFuse-GCN v7: 4-stream late fusion skeleton recogniser.

    Args:
        num_classes:    Output classes (60 or 120).
        variant:        'nano' | 'small'.
        in_channels:    Input channels per stream (default 3).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy: Adjacency strategy (default 'spatial').
        T:              Temporal length (default 64).
        num_joints:     Skeleton joints (default 25).
        dropout:        Override classifier head dropout (None = variant default).
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

        self.variant      = variant
        self.num_streams  = 4
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

        # ── 2. Late-fusion stems (4 independent BN+Conv1×1) ──────────────
        self.fusion = MultiStreamStem(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ──────────────────────────────────────────────
        total_blocks = sum(num_blocks)
        block_idx_global = 0

        self.stages     = nn.ModuleList()
        self.stage_gcns = nn.ModuleList()
        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            stage_gcn = MultiScaleAdaptiveGCN(
                channels=stage_ch,
                A=A,
                num_joints=num_joints,
                num_groups=num_gcn_groups,
            )
            self.stage_gcns.append(stage_gcn)

            stage_blocks = nn.ModuleList()
            for blk_idx in range(num_blocks[stage_idx]):
                blk_in     = prev_ch  if blk_idx == 0 else stage_ch
                blk_out    = stage_ch
                blk_stride = strides[stage_idx] if blk_idx == 0 else 1

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

        # ── 4. 4 × Classification head (one per stream) ──────────────────
        last_ch = channels[-1]
        self.stream_heads = nn.ModuleList([
            ClassificationHead(last_ch, num_classes, _dropout)
            for _ in range(self.num_streams)
        ])

        # Learned stream ensemble weights (softmax-normalised at inference)
        # Init zeros → equal weighting from epoch 1.
        self.stream_weights = nn.Parameter(torch.zeros(self.num_streams))

        # IB loss: class-conditional prototypes (InfoGCN-inspired)
        # Applied to the mean feature across all 4 streams.
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, last_ch) * 0.01
        )

        # ── Weight initialisation ────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # better variance propagation for attention projections
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Override: classifier FC heads use small normal init so logits are near-zero at epoch 0.
        # Xavier on Linear(C, num_classes) gives logit std ≈ 1.0+ → softmax saturates to 1-2 classes.
        for head in self.stream_heads:
            nn.init.normal_(head.fc.weight, 0, 0.01)
            nn.init.constant_(head.fc.bias, 0)

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------

    def _run_backbone(self, streams: list) -> list:
        """Stack 4 streams, run shared backbone once, split back.

        Args:
            streams: list of 4 tensors each (B, 3, T, V)
        Returns:
            list of 4 tensors each (B, last_ch, T', V')
        """
        # MultiStreamStem: projects each stream independently, cats along batch
        x_stacked = self.fusion(streams)            # (4B, C0, T, V)

        # Shared backbone (one pass)
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x_stacked = block(x_stacked)        # (4B, C, T', V')

        # Split back into 4 per-stream feature maps
        return list(x_stacked.chunk(self.num_streams, dim=0))   # 4 × (B, C, T', V')

    # -----------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: Dict with keys 'joint', 'velocity', 'bone', 'bone_velocity'
               Each value: (B, 3, T, V) or (B, 3, T, V, M).
               OR a single Tensor — broadcast to all 4 streams.
        Returns:
            Training: ([logits_j, logits_v, logits_b, logits_bv], ib_loss)
            Eval:     logits (B, num_classes) — softmax-weighted ensemble
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
            while len(streams) < self.num_streams:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x] * self.num_streams

        # ── Shared backbone (stacked batch) ─────────────────────────────
        feats = self._run_backbone(streams)   # 4 × (B, last_ch, T', V')

        # ── 4 × stream classification heads ─────────────────────────────
        all_logits   = []
        all_features = []
        for i, head in enumerate(self.stream_heads):
            logits_i, features_i = head(feats[i])
            all_logits.append(logits_i)
            all_features.append(features_i)

        if self.training:
            # IB loss on mean feature across all 4 streams
            mean_feat = torch.stack(all_features, dim=0).mean(dim=0)  # (B, C)
            proto_dists = torch.cdist(
                mean_feat.unsqueeze(0),
                self.class_prototypes.unsqueeze(0),
                p=2,
            ).squeeze(0)
            ib_loss = proto_dists.min(dim=-1).values.mean()
            return all_logits, ib_loss

        # ── Ensemble (softmax-weighted average) ─────────────────────────
        w = F.softmax(self.stream_weights, dim=0)   # (4,)
        ensemble = sum(w[i] * all_logits[i] for i in range(self.num_streams))
        return ensemble


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_shiftfuse_nano(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='nano', **kwargs)

def create_shiftfuse_small(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='small', **kwargs)
