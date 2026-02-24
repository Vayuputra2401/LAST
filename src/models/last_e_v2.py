"""
LAST-E v2: CTR-GCN-Aligned Efficient 3-Stream Model.

Architecture summary
--------------------
  3 Streams (joint, velocity, bone): each (B, 3, T, V)
        |
  StreamFusionV2 — per-stream BN + shared stem + per-stream stem BN + learned α
        |
  Stage 1 [C0, S1 blocks, stride=1]
        |
  Stage 2 [C0→C1, S2 blocks, stride=2 on block-0]
        |
  Stage 3 [C1→C2, S3 blocks, stride=2 on block-0]
        |
  Gated GAP+GMP Pool → BN → Dropout → FC(C2 → num_classes)

Key differences vs LAST-E v1
-----------------------------
- CTRLightGCNConv: per-group (G=4) channel-topology refinement via Q/K projections
  (CTR-GCN core innovation). Replaces single-conv DirectionalGCNConv.
- MultiScaleTCN4: 4-branch temporal (conv, dilated conv, maxpool, 1x1).
  Replaces 2-branch MultiScaleTCN.
- FreqTemporalGate: FFT-based frequency-domain channel attention (novel).
  Configurable via use_freq_gate flag (default True).
- DropPath: stochastic depth with linear ramp per block.
- Gated GAP+GMP head: learnable per-channel blend of avg and max pooling.
- StreamFusionV2: per-stream stem_bn fixes BN contamination bug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.ctr_gcn_block import CTRGCNBlock
from .blocks.stream_fusion_v2 import StreamFusionV2
from .graph import Graph


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
MODEL_VARIANTS_E_V2 = {
    'nano': {
        'channels': [24, 48, 96],
        'blocks': [2, 2, 3],
        'use_st_att': [False, False, True],
        'num_groups': 4,
        'drop_path_rate': 0.1,
        'use_freq_gate': True,
    },
    'small': {
        'channels': [32, 64, 128],
        'blocks': [2, 3, 3],
        'use_st_att': [False, True, True],
        'num_groups': 4,
        'drop_path_rate': 0.1,
        'use_freq_gate': True,
    },
    'base': {
        'channels': [40, 80, 160],
        'blocks': [3, 4, 4],
        'use_st_att': [True, True, True],
        'num_groups': 4,
        'drop_path_rate': 0.15,
        'use_freq_gate': True,
    },
    'large': {
        'channels': [48, 96, 192],
        'blocks': [4, 5, 5],
        'use_st_att': [True, True, True],
        'num_groups': 4,
        'drop_path_rate': 0.2,
        'use_freq_gate': True,
    },
}


class LAST_E_v2(nn.Module):
    """
    LAST-E v2: CTR-GCN-Aligned Efficient 3-Stream Model.

    Args:
        num_classes:     Number of output classes (60 or 120).
        variant:         'nano' | 'small' | 'base' | 'large'.
        in_channels:     Input channels per stream (default 3 for x,y,z).
        graph_layout:    Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy:  Adjacency strategy (default 'spatial').
        dropout:         Classifier dropout probability (default 0.3).
        use_freq_gate:   Override for FreqTemporalGate (None = use variant default).
        num_groups:      Override for CTR groups (None = use variant default).
        drop_path_rate:  Override for DropPath max rate (None = use variant default).
    """

    def __init__(
        self,
        num_classes: int = 60,
        variant: str = 'base',
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        dropout: float = 0.3,
        use_freq_gate: bool = None,
        num_groups: int = None,
        drop_path_rate: float = None,
    ):
        super().__init__()

        if variant not in MODEL_VARIANTS_E_V2:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(MODEL_VARIANTS_E_V2.keys())}"
            )

        cfg = MODEL_VARIANTS_E_V2[variant]
        channels   = cfg['channels']     # [C0, C1, C2]
        n_blocks   = cfg['blocks']       # [S1, S2, S3]
        use_st_att = cfg['use_st_att']   # [bool, bool, bool] per stage

        # Allow overrides from config/CLI, fall back to variant defaults
        self._num_groups   = num_groups   if num_groups   is not None else cfg['num_groups']
        self._drop_path    = drop_path_rate if drop_path_rate is not None else cfg['drop_path_rate']
        self._use_freq_gate = use_freq_gate if use_freq_gate is not None else cfg['use_freq_gate']

        self.variant = variant
        self.stream_names = ['joint', 'velocity', 'bone']

        # Total number of blocks across all stages (for DropPath linear ramp)
        self._total_blocks = sum(n_blocks)
        self._block_idx = 0  # running counter for DropPath rate assignment

        # ── 1. Graph adjacency ──────────────────────────────────────────────
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)   # (K, V, V)

        # ── 2. Stream fusion (V2: per-stream stem_bn) ──────────────────────
        self.fusion = StreamFusionV2(
            in_channels=in_channels,
            out_channels=channels[0],
            num_streams=3,
        )

        # ── 3. Stages ──────────────────────────────────────────────────────
        self._block_idx = 0  # reset counter
        self.stage1 = self._make_stage(
            in_c=channels[0], out_c=channels[0],
            num_blocks=n_blocks[0], stride=1,
            use_st_att=use_st_att[0],
        )
        self.stage2 = self._make_stage(
            in_c=channels[0], out_c=channels[1],
            num_blocks=n_blocks[1], stride=2,
            use_st_att=use_st_att[1],
        )
        self.stage3 = self._make_stage(
            in_c=channels[1], out_c=channels[2],
            num_blocks=n_blocks[2], stride=2,
            use_st_att=use_st_att[2],
        )

        # ── 4. Gated GAP+GMP classification head ──────────────────────────
        # Learnable per-channel blend of avg-pool and max-pool.
        # Init at 0 → sigmoid(0) = 0.5 → equal blend at start.
        # Max-pool backpropagates strong gradients to peak activations,
        # bypassing the 1/400 averaging that attenuates GAP gradients.
        self.pool_gate = nn.Parameter(torch.zeros(1, channels[2], 1, 1))
        self.head_bn = nn.BatchNorm1d(channels[2])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[2], num_classes)

        # ── Weight initialisation ───────────────────────────────────────────
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

    # -----------------------------------------------------------------------

    def _make_stage(
        self,
        in_c: int,
        out_c: int,
        num_blocks: int,
        stride: int,
        use_st_att: bool,
    ) -> nn.Sequential:
        """
        Build one stage of CTRGCNBlocks with linearly-ramped DropPath.

        Args:
            in_c:       Input channels.
            out_c:      Output channels.
            num_blocks: Total blocks in this stage.
            stride:     Temporal stride applied to block 0 only.
            use_st_att: Whether blocks use ST_JointAtt.
        """
        layers = []
        for i in range(num_blocks):
            # Linear ramp: 0 for first block overall → drop_path_rate for last
            dp_rate = self._drop_path * (self._block_idx / max(self._total_blocks - 1, 1))
            self._block_idx += 1

            layers.append(
                CTRGCNBlock(
                    in_channels=in_c if i == 0 else out_c,
                    out_channels=out_c,
                    A_physical=self.A,
                    stride=stride if i == 0 else 1,
                    residual=True,
                    use_st_att=use_st_att,
                    use_freq_gate=self._use_freq_gate,
                    num_groups=self._num_groups,
                    drop_path_rate=dp_rate,
                )
            )
        return nn.Sequential(*layers)

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Dict {'joint': Tensor, 'velocity': Tensor, 'bone': Tensor}
               Each tensor shape: (B, C, T, V) or (B, C, T, V, M).
               OR a single Tensor (B, C, T, V) or (B, C, T, V, M).

        Returns:
            logits: (B, num_classes)
        """
        if isinstance(x, dict):
            streams = []
            for name in self.stream_names:
                if name in x:
                    s = x[name]
                    if s.dim() == 5:      # (B, C, T, V, M) — take primary body
                        s = s[..., 0]
                    streams.append(s)
            while len(streams) < 3:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x, x, x]

        # Fuse streams → (B, C0, T, V)
        out = self.fusion(streams)

        # Backbone
        out = self.stage1(out)   # (B, C0, T,   V)
        out = self.stage2(out)   # (B, C1, T/2, V)
        out = self.stage3(out)   # (B, C2, T/4, V)

        # Gated GAP+GMP pool over (T, V) → (B, C2)
        gate = torch.sigmoid(self.pool_gate)  # (1, C2, 1, 1)
        gap = F.adaptive_avg_pool2d(out, (1, 1))  # (B, C2, 1, 1)
        gmp = F.adaptive_max_pool2d(out, (1, 1))  # (B, C2, 1, 1)
        pooled = gap * gate + gmp * (1 - gate)     # (B, C2, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)   # (B, C2)

        # Classifier
        pooled = self.head_bn(pooled)
        pooled = self.drop(pooled)
        logits = self.fc(pooled)   # (B, num_classes)

        return logits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_last_e_v2_nano(num_classes: int = 60, **kwargs) -> LAST_E_v2:
    """Create LAST-E-v2-Nano."""
    return LAST_E_v2(num_classes=num_classes, variant='nano', **kwargs)


def create_last_e_v2_small(num_classes: int = 60, **kwargs) -> LAST_E_v2:
    """Create LAST-E-v2-Small."""
    return LAST_E_v2(num_classes=num_classes, variant='small', **kwargs)


def create_last_e_v2_base(num_classes: int = 60, **kwargs) -> LAST_E_v2:
    """Create LAST-E-v2-Base."""
    return LAST_E_v2(num_classes=num_classes, variant='base', **kwargs)


def create_last_e_v2_large(num_classes: int = 60, **kwargs) -> LAST_E_v2:
    """Create LAST-E-v2-Large."""
    return LAST_E_v2(num_classes=num_classes, variant='large', **kwargs)
