"""
LAST-E: Lightweight Efficient 3-Stream Shared Backbone.

Architecture summary
--------------------
  3 Streams (joint, velocity, bone): each (B, 3, T, V)
        |
  StreamFusion — per-stream BN + shared stem + learned α weights
        |
  Stage 1 [C0, S1 blocks, stride=1, optional ST_JointAtt]
        |
  Stage 2 [C1, S2 blocks, stride=2 on block-0, optional ST_JointAtt]
        |
  Stage 3 [C2, S3 blocks, stride=2 on block-0, optional ST_JointAtt]
        |
  Global Average Pool → Dropout → FC(C2 → num_classes)

Key differences vs LAST-v2
---------------------------
- v2 runs the full backbone 3× (once per stream) and sums logits at the end.
  LAST-E fuses streams at the *input* → backbone runs exactly 1× → 3× cheaper.
- LightGCNBlock replaces EffGCNBlock: 1-conv fixed graph vs 5-conv adaptive graph.
- ST_JointAtt is optional per stage (use_st_att flag), saving params in early stages.
- No LinearAttention — student uses TCN only; distillation handles the gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.light_gcn import LightGCNBlock
from .blocks.stream_fusion import StreamFusion
from .graph import Graph


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
MODEL_VARIANTS_E = {
    'nano':  {'channels': [24, 48, 96],   'blocks': [2, 2, 3], 'use_st_att': [False, False, True]},
    'small': {'channels': [32, 64, 128],  'blocks': [2, 3, 3], 'use_st_att': [False, True, True]},
    'base':  {'channels': [40, 80, 160],  'blocks': [3, 4, 4], 'use_st_att': [True, True, True]},
    'large': {'channels': [48, 96, 192],  'blocks': [4, 5, 5], 'use_st_att': [True, True, True]},
}


class LAST_E(nn.Module):
    """
    LAST-E: Lightweight Efficient 3-Stream Shared Backbone for skeleton
    action recognition.

    Args:
        num_classes:    Number of output classes (60 or 120).
        variant:        'nano' | 'small' | 'base' | 'large'.
        in_channels:    Input channels per stream (default 3 for x,y,z).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy: Adjacency strategy (default 'spatial').
        dropout:        Classifier dropout probability (default 0.3).
    """

    def __init__(
        self,
        num_classes: int = 60,
        variant: str = 'base',
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        dropout: float = 0.3,
    ):
        super().__init__()

        if variant not in MODEL_VARIANTS_E:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(MODEL_VARIANTS_E.keys())}"
            )

        cfg = MODEL_VARIANTS_E[variant]
        channels   = cfg['channels']    # [C0, C1, C2]
        n_blocks   = cfg['blocks']      # [S1, S2, S3]
        use_st_att = cfg['use_st_att']  # [bool, bool, bool] per stage

        self.variant = variant
        self.stream_names = ['joint', 'velocity', 'bone']

        # ── 1. Graph adjacency ──────────────────────────────────────────────
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)   # (K, V, V)

        # ── 2. Stream fusion ────────────────────────────────────────────────
        self.fusion = StreamFusion(
            in_channels=in_channels,
            out_channels=channels[0],
            num_streams=3,
        )

        # ── 3. Stage 1: C0, stride=1 ────────────────────────────────────────
        self.stage1 = self._make_stage(
            in_c=channels[0],
            out_c=channels[0],
            num_blocks=n_blocks[0],
            stride=1,
            use_st_att=use_st_att[0],
        )

        # ── 4. Stage 2: C0→C1, stride=2 on first block ──────────────────────
        self.stage2 = self._make_stage(
            in_c=channels[0],
            out_c=channels[1],
            num_blocks=n_blocks[1],
            stride=2,
            use_st_att=use_st_att[1],
        )

        # ── 5. Stage 3: C1→C2, stride=2 on first block ──────────────────────
        self.stage3 = self._make_stage(
            in_c=channels[1],
            out_c=channels[2],
            num_blocks=n_blocks[2],
            stride=2,
            use_st_att=use_st_att[2],
        )

        # ── 6. Classification head ──────────────────────────────────────────
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(channels[2], num_classes)

        # ── Weight initialisation ───────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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
        Build one stage of LightGCNBlocks.

        Args:
            in_c:       Input channels.
            out_c:      Output channels.
            num_blocks: Total blocks in this stage.
            stride:     Temporal stride applied to block 0 only.
            use_st_att: Whether all blocks in this stage use ST_JointAtt.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                LightGCNBlock(
                    in_channels=in_c if i == 0 else out_c,
                    out_channels=out_c,
                    A_physical=self.A,
                    stride=stride if i == 0 else 1,
                    residual=True,
                    use_st_att=use_st_att,
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
               When M > 1 (multi-person), only the primary body (M=0) is used.

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
            if x.dim() == 5:             # (B, C, T, V, M) — take primary body
                x = x[..., 0]
            streams = [x, x, x]

        # Fuse streams → (B, C0, T, V)
        out = self.fusion(streams)

        # Backbone
        out = self.stage1(out)   # (B, C0, T,   V)
        out = self.stage2(out)   # (B, C1, T/2, V)
        out = self.stage3(out)   # (B, C2, T/4, V)

        # Global average pool over (T, V) → (B, C2)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Classifier
        out = self.drop(out)
        out = self.fc(out)       # (B, num_classes)

        return out


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_last_e_nano(num_classes: int = 60) -> LAST_E:
    """Create LAST-E-Nano (~108K params, beats EfficientGCN-B0 at 150K)."""
    return LAST_E(num_classes=num_classes, variant='nano')


def create_last_e_small(num_classes: int = 60) -> LAST_E:
    """Create LAST-E-Small (~207K params, beats EfficientGCN-B1 at 300K)."""
    return LAST_E(num_classes=num_classes, variant='small')


def create_last_e_base(num_classes: int = 60) -> LAST_E:
    """Create LAST-E-Base (~427K params, beats EfficientGCN-B4 at 2M)."""
    return LAST_E(num_classes=num_classes, variant='base')


def create_last_e_large(num_classes: int = 60) -> LAST_E:
    """Create LAST-E-Large (~759K params, beats EfficientGCN-B4 at 2M)."""
    return LAST_E(num_classes=num_classes, variant='large')
