"""
LAST-E v3: EfficientGCN-Based Architecture with MotionGate.

Architecture summary (after audit fixes)
-----------------------------------------
  3 Streams (joint, velocity, bone): each (B, 3, T, V)
        |
  StreamFusionConcat — per-stream BN + Concat(9ch) + Conv1×1 → C₀
        |
  Stage 1: [V3Block × num_blocks₁]   (each block: SpatialGCN → EpSepTCN×d → Gate → ST_JointAtt)
        |
  Stage 2: [V3Block × num_blocks₂]   (stride=2 on first block)
        |
  Stage 3: [V3Block × num_blocks₃]   (stride=2 on first block, HybridGate for large)
        |
  Gated GAP+GMP Pool → BN → Dropout → FC(C₃ → num_classes)
        |  [+ InfoGCN IB loss during training for base/large]

Audit fixes applied
--------------------
- P2: raw_partitions=True → single correct D^{-1/2}AD^{-1/2} normalization
- P5: num_blocks per stage → spatial receptive field 10-16 hops (was 6)
- P6: Wider channels → 3-4 dims/class (was 2.13)
- P4: StreamFusionConcat replaces softmax-coupled StreamFusionV2
- P3: Post-aggregation subset attention in SpatialGCN
- P0/P1: edge + stream_weights excluded from WD (in trainer.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks.spatial_gcn import SpatialGCN
from .blocks.ep_sep_tcn import EpSepTCN
from .blocks.motion_gate import MotionGate, HybridGate
from .blocks.st_joint_att import ST_JointAtt
from .blocks.stream_fusion_concat import StreamFusionConcat
from .graph import Graph, normalize_symdigraph


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Drop entire residual paths during training (stochastic depth)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ---------------------------------------------------------------------------
# V3 Block: SpatialGCN → EpSepTCN(×depth) → Gate → ST_JointAtt
# ---------------------------------------------------------------------------
class V3Block(nn.Module):
    """One block of LAST-E v3.

    SpatialGCN → [EpSepTCN × depth] → MotionGate/HybridGate → ST_JointAtt
    with DropPath on the residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        depth: int = 1,
        stride: int = 1,
        max_hop: int = 2,
        expand_ratio: int = 2,
        gate_type: str = 'motion',
        use_st_att: bool = True,
        use_subset_att: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Spatial graph convolution
        self.gcn = SpatialGCN(
            in_channels, out_channels, A,
            max_hop=max_hop,
            use_subset_att=use_subset_att,
        )
        self.act = nn.Hardswish(inplace=True)

        # Temporal convolution stack
        tcns = []
        for i in range(depth):
            s = stride if i == 0 else 1
            tcns.append(EpSepTCN(
                out_channels,
                kernel_size=5,
                stride=s,
                expand_ratio=expand_ratio,
            ))
        self.tcns = nn.Sequential(*tcns)

        # Channel gate (our novelty)
        if gate_type == 'hybrid':
            self.gate = HybridGate(out_channels)
        else:
            self.gate = MotionGate(out_channels)

        # Spatial-Temporal Joint Attention
        self.st_att = ST_JointAtt(out_channels) if use_st_att else nn.Identity()

        # Residual connection (handles channel/stride mismatch)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act(self.gcn(x))
        out = self.tcns(out)
        out = self.gate(out)
        out = self.st_att(out)
        return res + self.drop_path(out)


# ---------------------------------------------------------------------------
# Variant configurations
#
# Param budgets (EfficientGCN single-stream):
#   B0 ≈ 320K, B1 ≈ 420K, B2 ≈ 540K, B3 ≈ 740K, B4 ≈ 940K
#
# Targets: nano < B0, small < B1, base < B3, large < B4
# ---------------------------------------------------------------------------
MODEL_VARIANTS_E_V3 = {
    'nano': {
        'stem_channels': 24,
        'channels': [32, 48, 64],
        'num_blocks': [1, 1, 1],       # 3 SpatialGCN → 3×1=3 hops (ok for nano)
        'depths': [1, 1, 1],
        'strides': [1, 2, 2],
        'expand_ratio': 2,
        'max_hop': 1,
        'gate_type': 'motion',
        'use_st_att': [True, True, True],
        'use_subset_att': False,
        'use_ib_loss': False,
        'dropout': 0.2,
        'drop_path_rate': 0.0,
    },
    'small': {
        'stem_channels': 32,
        'channels': [48, 64, 96],
        'num_blocks': [1, 2, 2],       # 5 SpatialGCN → 5×2=10 hops ✅
        'depths': [1, 1, 1],
        'strides': [1, 2, 2],
        'expand_ratio': 2,
        'max_hop': 2,
        'gate_type': 'motion',
        'use_st_att': [True, True, True],
        'use_subset_att': True,
        'use_ib_loss': False,
        'dropout': 0.25,
        'drop_path_rate': 0.0,
    },
    'base': {
        'stem_channels': 48,
        'channels': [64, 96, 128],
        'num_blocks': [2, 2, 2],       # 6 SpatialGCN → 6×2=12 hops ✅
        'depths': [1, 1, 1],
        'strides': [1, 2, 2],
        'expand_ratio': 2,
        'max_hop': 2,
        'gate_type': 'motion',
        'use_st_att': [True, True, True],
        'use_subset_att': True,
        'use_ib_loss': True,
        'dropout': 0.3,
        'drop_path_rate': 0.05,
    },
    'large': {
        'stem_channels': 48,
        'channels': [80, 112, 160],
        'num_blocks': [2, 2, 2],       # 6 SpatialGCN → 6×2=12 hops ✅
        'depths': [1, 1, 1],
        'strides': [1, 2, 2],
        'expand_ratio': 2,
        'max_hop': 2,
        'gate_type': 'hybrid',
        'use_st_att': [True, True, True],
        'use_subset_att': True,
        'use_ib_loss': True,
        'dropout': 0.3,
        'drop_path_rate': 0.1,
    },
}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LAST_E_v3(nn.Module):
    """LAST-E v3: EfficientGCN-Based Lightweight Action Skeleton Transformer.

    Args:
        num_classes:     Number of output classes (60 or 120).
        variant:         'nano' | 'small' | 'base' | 'large'.
        in_channels:     Input channels per stream (default 3 for x,y,z).
        graph_layout:    Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy:  Adjacency strategy (default 'spatial').
        dropout:         Override classifier dropout (None = variant default).
        drop_path_rate:  Override max DropPath rate (None = variant default).
    """

    def __init__(
        self,
        num_classes: int = 60,
        variant: str = 'base',
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        dropout: float = None,
        drop_path_rate: float = None,
    ):
        super().__init__()

        if variant not in MODEL_VARIANTS_E_V3:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(MODEL_VARIANTS_E_V3.keys())}"
            )

        cfg = MODEL_VARIANTS_E_V3[variant]
        stem_ch = cfg['stem_channels']
        channels = cfg['channels']
        num_blocks = cfg['num_blocks']
        depths = cfg['depths']
        strides = cfg['strides']
        expand_ratio = cfg['expand_ratio']
        max_hop = cfg['max_hop']
        gate_type = cfg['gate_type']
        use_st_att = cfg['use_st_att']
        use_subset_att = cfg['use_subset_att']
        use_ib_loss = cfg['use_ib_loss']

        _dropout = dropout if dropout is not None else cfg['dropout']
        _drop_path = drop_path_rate if drop_path_rate is not None else cfg['drop_path_rate']

        self.variant = variant
        self.use_ib_loss = use_ib_loss
        self.stream_names = ['joint', 'velocity', 'bone']

        # ── 1. Graph adjacency ───────────────────────────────────────────
        # P2 FIX: raw_partitions=True → subsets have clean 0/1 values,
        # normalize_symdigraph applied exactly once here.
        self.graph = Graph(
            layout=graph_layout,
            strategy=graph_strategy,
            max_hop=max_hop,
            raw_partitions=True,
        )
        A_raw = self.graph.A  # (K, V, V) — raw 0/1 partitions
        A_sym = np.stack([
            normalize_symdigraph(A_raw[k]) for k in range(A_raw.shape[0])
        ])
        A = torch.tensor(A_sym, dtype=torch.float32)
        self.register_buffer('A', A)

        # ── 2. Stream fusion (P4 FIX: concat instead of softmax) ────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=3,
        )

        # ── 3. Stages (P5 FIX: multiple V3Blocks per stage) ────────────
        # Total blocks across all stages for DropPath linear ramp
        total_blocks = sum(num_blocks)
        block_count = 0

        self.stages = nn.ModuleList()
        prev_ch = stem_ch
        for stage_idx in range(len(channels)):
            stage_blocks = nn.ModuleList()
            for blk_idx in range(num_blocks[stage_idx]):
                # First block of each stage has stride + channel change
                if blk_idx == 0:
                    blk_in = prev_ch
                    blk_out = channels[stage_idx]
                    blk_stride = strides[stage_idx]
                else:
                    # Subsequent blocks: same channels, stride=1
                    blk_in = channels[stage_idx]
                    blk_out = channels[stage_idx]
                    blk_stride = 1

                dp_rate = _drop_path * (block_count / max(total_blocks - 1, 1))
                block_count += 1

                # HybridGate only on last stage's blocks for large variant
                blk_gate = gate_type if stage_idx == len(channels) - 1 else 'motion'

                stage_blocks.append(V3Block(
                    in_channels=blk_in,
                    out_channels=blk_out,
                    A=A,
                    depth=depths[stage_idx],
                    stride=blk_stride,
                    max_hop=max_hop,
                    expand_ratio=expand_ratio,
                    gate_type=blk_gate,
                    use_st_att=use_st_att[stage_idx],
                    use_subset_att=use_subset_att,
                    drop_path_rate=dp_rate,
                ))

            self.stages.append(stage_blocks)
            prev_ch = channels[stage_idx]

        # ── 4. Head: Gated GAP+GMP + classifier ─────────────────────────
        last_ch = channels[-1]
        self.pool_gate = nn.Parameter(torch.zeros(1, last_ch, 1, 1))
        self.head_bn = nn.BatchNorm1d(last_ch)
        self.drop = nn.Dropout(_dropout)
        self.fc = nn.Linear(last_ch, num_classes)

        # ── 5. InfoGCN IB loss heads (training only) ────────────────────
        if use_ib_loss:
            latent_dim = min(128, last_ch)
            self.fc_mu = nn.Linear(last_ch, latent_dim)
            self.fc_logvar = nn.Linear(last_ch, latent_dim)
        else:
            self.fc_mu = None
            self.fc_logvar = None

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
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Dict {'joint': Tensor, 'velocity': Tensor, 'bone': Tensor}
               Each tensor shape: (B, C, T, V) or (B, C, T, V, M).
               OR a single Tensor (B, C, T, V) or (B, C, T, V, M).

        Returns:
            logits: (B, num_classes)
            ib_loss: scalar (only during training if use_ib_loss is True)
        """
        # ── Input handling ──
        if isinstance(x, dict):
            streams = []
            for name in self.stream_names:
                if name in x:
                    s = x[name]
                    if s.dim() == 5:
                        s = s[..., 0]
                    streams.append(s)
            while len(streams) < 3:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x, x, x]

        # ── Fuse streams → (B, stem_ch, T, V) ──
        out = self.fusion(streams)

        # ── Backbone: multi-block stages ──
        for stage_blocks in self.stages:
            for block in stage_blocks:
                out = block(out)

        # ── Gated GAP+GMP pool ──
        gate = torch.sigmoid(self.pool_gate)
        gap = F.adaptive_avg_pool2d(out, (1, 1))
        gmp = F.adaptive_max_pool2d(out, (1, 1))
        pooled = gap * gate + gmp * (1 - gate)
        pooled = pooled.view(pooled.size(0), -1)

        # ── InfoGCN IB loss (training only) ──
        ib_loss = None
        if self.training and self.use_ib_loss and self.fc_mu is not None:
            mu = self.fc_mu(pooled)
            log_var = self.fc_logvar(pooled)
            ib_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean()

        # ── Classifier ──
        pooled = self.head_bn(pooled)
        pooled = self.drop(pooled)
        logits = self.fc(pooled)

        if ib_loss is not None:
            return logits, ib_loss
        return logits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_last_e_v3_nano(num_classes: int = 60, **kwargs) -> LAST_E_v3:
    return LAST_E_v3(num_classes=num_classes, variant='nano', **kwargs)

def create_last_e_v3_small(num_classes: int = 60, **kwargs) -> LAST_E_v3:
    return LAST_E_v3(num_classes=num_classes, variant='small', **kwargs)

def create_last_e_v3_base(num_classes: int = 60, **kwargs) -> LAST_E_v3:
    return LAST_E_v3(num_classes=num_classes, variant='base', **kwargs)

def create_last_e_v3_large(num_classes: int = 60, **kwargs) -> LAST_E_v3:
    return LAST_E_v3(num_classes=num_classes, variant='large', **kwargs)
