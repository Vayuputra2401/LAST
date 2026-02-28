"""
ShiftFuse-GCN (LAST-Lite): Lightweight Skeleton Action Recognition.

Architecture summary
--------------------
  4 Streams (joint, velocity, bone, bone_velocity): each (B, 3, T, V)
        |
  StreamFusionConcat — per-stream BN + Concat(12ch) + Conv1×1 → C₀
        |
  Stage 1: [ShiftFuseBlock × num_blocks₁]  stride=1  → StaticGCN (shared)
        |
  Stage 2: [ShiftFuseBlock × num_blocks₂]  stride=2  → StaticGCN (shared)
        |
  Stage 3: [ShiftFuseBlock × num_blocks₃]  stride=2  → StaticGCN (shared)
        |
  Gated GAP+GMP Pool → BN1d → Dropout → FC(C₃ → num_classes)

ShiftFuseBlock (per block)
--------------------------
  BodyRegionShift   (Idea F — 0 params, anatomical channel permutation)
  Conv2d(C,C,1×1) + BN + Hardswish  (pointwise channel mixing)
  JointEmbedding    (additive per-joint semantic bias, V×C params)
  FrozenDCTGate     (Idea G — learnable frequency mask, C×T params)
  EpSepTCN          (depthwise-sep temporal conv, reused from EfficientGCN)
  FrameDynamicsGate (learnable per-frame temporal gate, C×T params)
  Outer residual    (Conv1×1+BN if channel/stride mismatch, else Identity)
  StaticGCN         (shared across stage — C² + 2C + 625 params per stage)

StaticGCN (one per stage, shared weight)
-----------------------------------------
  x_agg = Σ_k A_k @ x  +  A_learned_norm @ x   (static + trainable graph)
  out   = x + BN(Conv1×1(x_agg))                (projection + residual)

Novel contributions
-------------------
  BRASP (Idea F): Anatomically-partitioned channel shift, 0 params.
  FDCR  (Idea G): Fixed-compute frequency-domain channel specialisation.
  StaticGCN: Stage-shared graph conv with learnable topology correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.body_region_shift import BodyRegionShift
from .blocks.frozen_dct_gate import FrozenDCTGate
from .blocks.joint_embedding import JointEmbedding
from .blocks.frame_dynamics_gate import FrameDynamicsGate
from .blocks.ep_sep_tcn import EpSepTCN
from .blocks.stream_fusion_concat import StreamFusionConcat
from .blocks.static_gcn import StaticGCN
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
MODEL_VARIANTS_SHIFTFUSE = {
    'nano': {
        'stem_channels': 24,
        'channels':      [32, 48, 64],
        'num_blocks':    [1, 1, 1],
        'strides':       [1, 2, 2],
        'expand_ratio':  2,
        'max_hop':       1,
        'use_dct_gate':    True,
        'use_joint_embed': True,
        'use_frame_gate':  True,
        'dropout':         0.1,
    },
    'small': {
        'stem_channels': 32,
        'channels':      [48, 72, 96],
        'num_blocks':    [1, 2, 2],
        'strides':       [1, 2, 2],
        'expand_ratio':  2,
        'max_hop':       2,
        'use_dct_gate':    True,
        'use_joint_embed': True,
        'use_frame_gate':  True,
        'dropout':         0.15,
    },
}


# ---------------------------------------------------------------------------
# ShiftFuseBlock
# ---------------------------------------------------------------------------
class ShiftFuseBlock(nn.Module):
    """
    One block of ShiftFuse-GCN.

    Pipeline:
        BodyRegionShift → Conv1×1+BN+Hardswish → JointEmbedding
        → FrozenDCTGate → EpSepTCN → FrameDynamicsGate
        → Outer residual → StaticGCN (shared, optional)

    StaticGCN is passed in by LAST_Lite — one instance shared across all
    blocks in the same stage, so graph weights are not duplicated.

    Args:
        in_channels:      Input channels.
        out_channels:     Output channels.
        A_flat:           (V, V) flat adjacency for BodyRegionShift init.
        T:                Temporal length (for FrozenDCTGate / FrameDynamicsGate).
        stride:           Temporal stride applied by EpSepTCN (default 1).
        expand_ratio:     EpSepTCN expansion ratio (default 2).
        num_joints:       Number of skeleton joints (default 25).
        use_dct_gate:     Include FrozenDCTGate (default True).
        use_joint_embed:  Include JointEmbedding (default True).
        use_frame_gate:   Include FrameDynamicsGate (default True).
        gcn:              Optional StaticGCN (shared reference from LAST_Lite).
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
        use_frame_gate: bool = True,
        gcn: nn.Module = None,
    ):
        super().__init__()

        # 1. Body-Region-Aware Spatial Shift (Idea F) — 0 params
        self.shift = BodyRegionShift(in_channels, A_flat)

        # 2. Pointwise channel mixing
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

        # 3. Joint semantic embedding
        self.joint_embed = JointEmbedding(out_channels, num_joints) if use_joint_embed \
            else nn.Identity()

        # 4. Frozen DCT frequency gate (Idea G)
        self.dct_gate = FrozenDCTGate(out_channels, T) if use_dct_gate \
            else nn.Identity()

        # 5. Temporal convolution (EfficientGCN EpSepTCN — reused as-is)
        self.tcn = EpSepTCN(
            out_channels,
            kernel_size=5,
            stride=stride,
            expand_ratio=expand_ratio,
        )

        # 6. Frame dynamics gate  (T_out = T // stride after EpSepTCN)
        T_out = T // stride
        self.frame_gate = FrameDynamicsGate(out_channels, T_out) if use_frame_gate \
            else nn.Identity()

        # 7. Outer residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # 8. Shared StaticGCN — stored as plain attribute (not registered as
        #    a submodule) so that LAST_Lite owns and registers it once per stage.
        #    Using object.__setattr__ prevents nn.Module from re-registering it.
        object.__setattr__(self, 'gcn', gcn)

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
        out = self.dct_gate(out)
        out = self.tcn(out)
        out = self.frame_gate(out)

        out = res + out                    # block outer residual

        if self.gcn is not None:
            out = self.gcn(out)            # spatial refinement (StaticGCN has own residual)

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
        stem_ch      = cfg['stem_channels']
        channels     = cfg['channels']
        num_blocks   = cfg['num_blocks']
        strides      = cfg['strides']
        expand_ratio = cfg['expand_ratio']
        max_hop      = cfg['max_hop']
        use_dct_gate    = cfg['use_dct_gate']
        use_joint_embed = cfg['use_joint_embed']
        use_frame_gate  = cfg['use_frame_gate']
        _dropout = dropout if dropout is not None else cfg['dropout']

        self.variant = variant
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']

        # ── 1. Graph adjacency ───────────────────────────────────────────
        # Build K-subset adjacency (same convention as LAST-E v3).
        # raw_partitions=True → clean 0/1 subsets, no double normalisation.
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

        # Flat (V, V) adjacency for BodyRegionShift neighbour lookup:
        # union of all K subsets — A[v,w]>0 iff v,w are graph-connected.
        A_flat = torch.tensor(
            (A_raw.sum(0) > 0).astype('float32')
        )
        # A_flat is only used at __init__ time (compute_shift_indices), not stored.

        # ── 2. Stream fusion (EfficientGCN-exact concat) ─────────────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ──────────────────────────────────────────────
        # One StaticGCN per stage, shared by all blocks within that stage.
        # Registered in self.stage_gcns so optimizer / state_dict sees them.
        self.stages     = nn.ModuleList()
        self.stage_gcns = nn.ModuleList()   # index matches stage index
        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            # Create the shared GCN for this stage (operates at stage_ch)
            stage_gcn = StaticGCN(
                channels=stage_ch,
                A=A,                  # (K, V, V) normalised adjacency
                num_joints=num_joints,
            )
            self.stage_gcns.append(stage_gcn)

            stage_blocks = nn.ModuleList()
            for blk_idx in range(num_blocks[stage_idx]):
                blk_in     = prev_ch  if blk_idx == 0 else stage_ch
                blk_out    = stage_ch
                blk_stride = strides[stage_idx] if blk_idx == 0 else 1

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
                    gcn=stage_gcn,    # shared reference — not re-registered
                ))

                # Update T after the first (stride) block
                if blk_idx == 0:
                    T_cur = T_cur // blk_stride

            self.stages.append(stage_blocks)
            prev_ch = stage_ch

        # ── 4. Head: Gated GAP+GMP + classifier ─────────────────────────
        last_ch = channels[-1]
        self.pool_gate = nn.Parameter(torch.zeros(1, last_ch, 1, 1))
        self.head_bn   = nn.BatchNorm1d(last_ch)
        self.drop      = nn.Dropout(_dropout)
        self.fc        = nn.Linear(last_ch, num_classes)

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
            logits: (B, num_classes)
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
            # Pad with zeros if some streams are missing (e.g. 3-stream dicts)
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
        pooled = pooled.view(pooled.size(0), -1)

        # ── Classifier ───────────────────────────────────────────────────
        pooled = self.head_bn(pooled)
        pooled = self.drop(pooled)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_shiftfuse_nano(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='nano', **kwargs)

def create_shiftfuse_small(num_classes: int = 60, **kwargs) -> LAST_Lite:
    return LAST_Lite(num_classes=num_classes, variant='small', **kwargs)
