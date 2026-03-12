"""
ShiftFuse-Zero — Accuracy-per-parameter skeleton action recognition.

Design goal: beat EfficientGCN-B0 (290K params, 90.2%) on accuracy-per-
parameter using < 180K params.

Key ideas:
  1. Early fusion (StreamFusionConcat): 4 streams → single BN domain → single
     CE loss → clean gradient from epoch 1 (same trick as EfficientGCN).
  2. Zero-param graph processing: BRASP + SGPShift route information along
     anatomically-meaningful paths at 0 parameter cost.
  3. Single learnable graph correction per stage (A_learned, V×V=625 params)
     shared across all blocks in the stage — fine-tunes the structural prior.
  4. Wide channels [40, 80, 160] — param budget freed by removing the GCN
     kernel (was 127K of 225K in V10 nano) is redistributed into channel width.
  5. No IB loss, no class_prototypes — clean CrossEntropyLoss, converges fast.

Block pipeline (ZeroGCNBlock):
    Input (B, C, T, V)
        │
        ├─ BRASP        — anatomical channel routing (0 params)
        │
        ├─ SGPShift     — semantic-typed graph shift (0 params)
        │
        ├─ A_learned correction (shared per stage, V×V, no WD)
        │   Conv1×1(C→C) + BN + ReLU
        │
        ├─ MultiScaleTCN (d=1, d=2, k=9)  — temporal modelling
        │   + DropPath
        │
        ├─ JointEmbedding (shared per stage, V×C, no WD)
        │
        └─ TemporalLandmarkAttention (stage 3 only, O(T×K))
           Residual + output

Model pipeline (ShiftFuseZero):
    4 streams  →  StreamFusionConcat (C0=24)
    →  Stage 1 (C=40, 2 blocks)
    →  Stage 2 (C=80, 3 blocks)
    →  Stage 3 (C=160, 2 blocks)
    →  Gated GAP+GMP pool
    →  Dropout → Linear(160 → num_classes)

Variant:
  nano  channels=[40,80,160]  blocks=[2,3,2]  ~163–170K params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.stream_fusion_concat import StreamFusionConcat
from .blocks.body_region_shift import BodyRegionShift
from .blocks.sgp_shift import SGPShift
from .blocks.light_gcn import MultiScaleTCN
from .blocks.joint_embedding import JointEmbedding
from .blocks.temporal_landmark_attn import TemporalLandmarkAttention
from .blocks.drop_path import DropPath
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
ZERO_VARIANTS = {
    'nano': {
        'stem_channels':   24,
        'channels':        [40, 80, 160],
        'num_blocks':      [2, 3, 2],
        'strides':         [1, 2, 2],       # stride-2 at stage boundary
        'drop_path_rate':  0.10,
        'dropout':         0.10,
        'tla_landmarks':   8,
        'tla_reduce_ratio': 8,
    },
}


# ---------------------------------------------------------------------------
# ZeroGCNBlock
# ---------------------------------------------------------------------------
class ZeroGCNBlock(nn.Module):
    """Single ShiftFuse-Zero block.

    Graph propagation is entirely zero-parameter (BRASP + SGPShift).
    Learnable components: Conv1×1 graph-mixing, MultiScaleTCN, JE (optional),
    TLA (optional).

    Args:
        in_channels:    Input channels C_in.
        out_channels:   Output channels C_out.
        stride:         Temporal stride (1 = same T; 2 = halve T).
        A_flat:         (V, V) flat adjacency used by BRASP cross-body group.
        A_intra:        (V, V) intra-part adjacency for SGPShift group 0.
        A_inter:        (V, V) inter-part adjacency for SGPShift group 1.
        A_learned:      Shared nn.Parameter (V, V) for learned correction.
                        Passed from parent stage; normalized per-forward.
        je:             Shared JointEmbedding (or None = no JE for this block).
        tla:            TemporalLandmarkAttention (or None).
        drop_path_rate: Stochastic depth drop probability.
        num_joints:     V (default 25).
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        stride:         int,
        A_flat:         torch.Tensor,
        A_intra:        torch.Tensor,
        A_inter:        torch.Tensor,
        A_learned:      nn.Parameter,
        je:             nn.Module = None,
        tla:            nn.Module = None,
        drop_path_rate: float = 0.0,
        num_joints:     int   = 25,
    ):
        super().__init__()

        self.stride = stride

        # ── Zero-param spatial routing ────────────────────────────────────
        # BRASP: anatomical channel routing (arm/leg/torso/cross-body groups)
        self.brasp = BodyRegionShift(
            channels=in_channels,
            A=A_flat,
        )

        # SGPShift: semantic-typed graph shift (intra/inter/identity groups)
        self.sgp_shift = SGPShift(
            channels=in_channels,
            A_intra=A_intra,
            A_inter=A_inter,
            num_joints=num_joints,
        )

        # ── Learnable graph-mixing conv ───────────────────────────────────
        # A_learned is a shared Parameter (V, V) stored in parent model.
        # We hold a reference only — gradients flow back to the shared param.
        self._A_learned = A_learned

        # Conv1×1 + BN + ReLU: mixes channels after graph-shifted features.
        # Channel expansion handled here if in_channels ≠ out_channels.
        self.graph_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── Temporal modelling ────────────────────────────────────────────
        self.tcn = MultiScaleTCN(out_channels, stride=stride)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # ── Optional addons ───────────────────────────────────────────────
        self.je  = je    # shared JointEmbedding (or None)
        self.tla = tla   # TemporalLandmarkAttention (or None)

        # ── Residual ──────────────────────────────────────────────────────
        if (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x  # save for residual

        # ── Zero-param spatial routing ────────────────────────────────────
        x = self.brasp(x)       # anatomical channel routing
        x = self.sgp_shift(x)   # semantic-typed joint shift

        # ── Apply A_learned correction (aggregate then mix) ───────────────
        # Normalize shared A_learned per-forward: D^{-1/2}|A|D^{-1/2}
        A_l = self._A_learned.abs()
        d   = A_l.sum(dim=1).clamp(min=1e-6).pow(-0.5)
        A_l_norm = d.unsqueeze(1) * A_l * d.unsqueeze(0)  # (V, V)

        B, C, T, V = x.shape
        x_flat = x.reshape(B, C * T, V)
        x_agg  = torch.matmul(x_flat, A_l_norm).reshape(B, C, T, V)

        # Mix channels with learned 1×1 conv (also handles channel expansion)
        x = self.graph_conv(x + x_agg)

        # ── Joint embedding (shared per stage) ───────────────────────────
        if self.je is not None:
            x = self.je(x)

        # ── Temporal modelling ────────────────────────────────────────────
        x = self.drop_path(self.tcn(x))

        # ── Residual ──────────────────────────────────────────────────────
        x = self.out_relu(x + self.residual(res))

        # ── Optional TLA (stage 3 only) ───────────────────────────────────
        if self.tla is not None:
            x = self.tla(x)

        return x


# ---------------------------------------------------------------------------
# ShiftFuseZero
# ---------------------------------------------------------------------------
class ShiftFuseZero(nn.Module):
    """ShiftFuse-Zero: early-fusion zero-GCN skeleton action recogniser.

    Args:
        num_classes:    Output classes (default 60).
        variant:        Model size variant (default 'nano').
        in_channels:    Channels per input stream (default 3: x, y, z).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        num_joints:     Number of joints (default 25).
        dropout:        Classifier head dropout override (None = variant default).
    """

    def __init__(
        self,
        num_classes:  int  = 60,
        variant:      str  = 'nano',
        in_channels:  int  = 3,
        graph_layout: str  = 'ntu-rgb+d',
        num_joints:   int  = 25,
        dropout:      float = None,
    ):
        super().__init__()

        if variant not in ZERO_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(ZERO_VARIANTS.keys())}"
            )

        cfg             = ZERO_VARIANTS[variant]
        stem_ch         = cfg['stem_channels']
        channels        = cfg['channels']
        num_blocks      = cfg['num_blocks']
        strides         = cfg['strides']
        drop_path_rate  = cfg['drop_path_rate']
        tla_landmarks   = cfg['tla_landmarks']
        tla_reduce      = cfg['tla_reduce_ratio']
        _dropout        = dropout if dropout is not None else cfg['dropout']

        self.variant      = variant
        self.num_classes  = num_classes
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']

        # ── 1. Semantic body-part graph ───────────────────────────────────
        # max_hop=2 to populate A_cross; raw_partitions=True so that
        # normalize_symdigraph_full applies normalization exactly once.
        graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',
            max_hop=2,
            raw_partitions=True,
        )
        A_raw  = graph.A                             # (3, V, V) raw 0/1
        A_sym  = normalize_symdigraph_full(A_raw)    # (3, V, V) normalized

        A_intra = torch.tensor(A_sym[0], dtype=torch.float32)  # within-part
        A_inter = torch.tensor(A_sym[1], dtype=torch.float32)  # cross-part edges
        # A_cross = A_sym[2]  (multi-hop; not used directly in shift groups)

        # Flat adjacency for BRASP cross-body group: union of all edges
        A_flat = torch.tensor(
            (A_raw.sum(0) > 0).astype('float32')
        )  # (V, V)

        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # ── 2. Early fusion stem ──────────────────────────────────────────
        # 4 streams → per-stream BN → concat → Conv1×1 → BN → ReLU
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ───────────────────────────────────────────────
        total_blocks     = sum(num_blocks)
        block_idx_global = 0

        self.stages = nn.ModuleList()

        # Per-stage shared A_learned correction parameters.
        # Named 'stage{i}_A_learned' so trainer's 'A_learned' no_decay rule
        # matches them automatically (no trainer.py change required).
        for i in range(len(channels)):
            param = nn.Parameter(torch.zeros(num_joints, num_joints))
            setattr(self, f'stage{i}_A_learned', param)

        prev_ch = stem_ch

        for stage_idx in range(len(channels)):
            stage_ch     = channels[stage_idx]
            stage_stride = strides[stage_idx]
            n_blocks     = num_blocks[stage_idx]
            is_last_stage = (stage_idx == len(channels) - 1)

            # Shared JointEmbedding for this stage
            stage_je = JointEmbedding(stage_ch, num_joints)

            # TLA only on last stage
            stage_tla = (
                TemporalLandmarkAttention(stage_ch, tla_landmarks, tla_reduce)
                if is_last_stage else None
            )

            # Shared A_learned for this stage
            A_learned_param = getattr(self, f'stage{stage_idx}_A_learned')

            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                # Linear drop-path rate: 0 at first block → drop_path_rate at last
                dp_rate = drop_path_rate * block_idx_global / max(total_blocks - 1, 1)

                # Stride only on the first block of non-first stages
                stride = stage_stride if (block_idx == 0 and stage_idx > 0) else 1
                c_in   = prev_ch if block_idx == 0 else stage_ch

                block = ZeroGCNBlock(
                    in_channels    = c_in,
                    out_channels   = stage_ch,
                    stride         = stride,
                    A_flat         = A_flat,
                    A_intra        = A_intra,
                    A_inter        = A_inter,
                    A_learned      = A_learned_param,
                    je             = stage_je,
                    tla            = stage_tla if (block_idx == n_blocks - 1) else None,
                    drop_path_rate = dp_rate,
                    num_joints     = num_joints,
                )
                stage_blocks.append(block)
                block_idx_global += 1

            self.stages.append(stage_blocks)
            prev_ch = stage_ch

        # ── 4. Classifier head ────────────────────────────────────────────
        final_ch = channels[-1]

        # Gated GAP+GMP blend: pool_gate scalar (sigmoid gate, no WD)
        self.pool_gate = nn.Parameter(torch.zeros(1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(final_ch, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, stream_dict: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            stream_dict: Dict with keys 'joint', 'velocity', 'bone',
                         'bone_velocity', each (B, 3, T, V) or (B, 3, T, V, M).

        Returns:
            logits: (B, num_classes)
        """
        # ── Unpack streams (handle M=2 multi-body input by taking body 0) ──
        streams = []
        for name in self.stream_names:
            s = stream_dict[name]
            if s.dim() == 5:
                s = s[..., 0]   # (B, 3, T, V, M) → (B, 3, T, V)
            streams.append(s)

        # ── Early fusion ──────────────────────────────────────────────────
        x = self.fusion(streams)   # (B, stem_ch, T, V)

        # ── Backbone stages ───────────────────────────────────────────────
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)

        # ── Pooling ───────────────────────────────────────────────────────
        # Temporal average-pool first (T → 1)
        x = x.mean(dim=2)                          # (B, C, V)
        # Then spatial: gated blend of GAP and GMP over joints
        gap = x.mean(dim=-1)                       # (B, C)
        gmp = x.max(dim=-1).values                 # (B, C)
        g   = torch.sigmoid(self.pool_gate)        # scalar gate
        x   = g * gap + (1 - g) * gmp             # (B, C)

        # ── Classifier ────────────────────────────────────────────────────
        return self.classifier(x)                  # (B, num_classes)


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------
def build_shiftfuse_zero(variant: str = 'nano', num_classes: int = 60, **kwargs) -> ShiftFuseZero:
    """Convenience factory.

    Args:
        variant:     'nano' (only option currently).
        num_classes: Output classes (60 for NTU-60, 120 for NTU-120).
        **kwargs:    Forwarded to ShiftFuseZero (e.g. dropout override).

    Returns:
        ShiftFuseZero model.
    """
    return ShiftFuseZero(num_classes=num_classes, variant=variant, **kwargs)
