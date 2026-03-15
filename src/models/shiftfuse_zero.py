"""
ShiftFuse-Zero — Accuracy-per-parameter skeleton action recognition.

Variants:
  nano       — early fusion, K=3 shared A_learned, BRASP+SGPShift (0-param routing)
  small      — same + wider channels + SE + deeper stages
  small_late — TWO-BACKBONE late fusion (joint+velocity | bone+bone_velocity),
               each backbone uses proper K=3 per-partition GCN (separate W per
               structural partition), no SE.  ~260-310K params, targeting 87-89%.

Block pipeline (ZeroGCNBlock):
    Input (B, C, T, V)
        │
        ├─ BRASP        — anatomical channel routing (0 params)
        │
        ├─ SGPShift     — semantic-typed graph shift (0 params)
        │
        ├─ [multi_gcn=False] A_learned correction (shared per stage, K×V×V)
        │   + block_adj (per-block V×V) → graph_conv: Conv1×1(C→C) + BN + act
        │
        ├─ [multi_gcn=True ] K=3 per-partition GCN:
        │   sum_k W_k(A_structural_k @ x)  — separate Conv1×1 per structural partition
        │
        ├─ MultiScaleTCN (d=1, d=2, k=9)
        │   + DropPath
        │
        ├─ JointEmbedding (shared per stage)
        │
        └─ TemporalLandmarkAttention (last stage only)
           Residual + output
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
from .blocks.channel_se import ChannelSE
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
ZERO_VARIANTS = {
    'nano': {
        'stem_channels':    48,
        'channels':         [40, 80, 160],
        'num_blocks':       [2, 3, 2],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.10,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           False,
        'use_k3_adj':       True,    # K=3 shared adjacency per stage (+3,771p)
        'multi_gcn':        False,
        'use_adyn':         False,
    },
    'small': {
        'stem_channels':    64,
        'channels':         [48, 96, 192],
        'num_blocks':       [3, 3, 3],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.10,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           True,
        'use_k3_adj':       True,
        'multi_gcn':        False,
        'use_adyn':         False,
    },
    # Sub-backbone config for ShiftFuseZeroLate (two of these form small_late)
    'small_late': {
        'stem_channels':    32,
        'channels':         [32, 64, 128],
        'num_blocks':       [2, 3, 2],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.10,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           False,
        'use_k3_adj':       False,
        'multi_gcn':        True,
        'use_adyn':         False,
    },
    # ── New efficient variants ─────────────────────────────────────────────
    # nano_multi: same param budget as nano, K=3 per-partition GCN (proper W per partition)
    # Channels shrunk [40→32] to absorb 3× GCN conv cost. Expected: 85–86%.
    'nano_multi': {
        'stem_channels':    48,
        'channels':         [32, 64, 128],
        'num_blocks':       [2, 3, 2],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.10,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           False,
        'use_k3_adj':       False,   # replaced by multi_gcn structural partitions
        'multi_gcn':        True,    # K=3 per-partition: W_intra + W_inter + W_cross
        'use_adyn':         False,
    },
    # small_multi: same param budget as small but proper per-partition GCN, no SE.
    # Channels shrunk [48→40] to absorb 3× GCN conv cost. Expected: 86–87%.
    'small_multi': {
        'stem_channels':    48,
        'channels':         [40, 80, 160],
        'num_blocks':       [2, 3, 2],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.10,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           False,
        'use_k3_adj':       False,
        'multi_gcn':        True,
        'use_adyn':         False,
    },
    # large: full stack — per-partition GCN + A_dynamic per-sample graph + SE.
    # Targets 88–90%+ on NTU-60 xsub.
    'large': {
        'stem_channels':    64,
        'channels':         [48, 96, 192],
        'num_blocks':       [3, 4, 3],
        'strides':          [1, 2, 2],
        'drop_path_rate':   0.15,
        'dropout':          0.10,
        'tla_landmarks':    8,
        'tla_reduce_ratio': 8,
        'use_se':           True,    # SE recalibration — worth it at this scale
        'use_k3_adj':       False,
        'multi_gcn':        True,    # per-partition W
        'use_adyn':         True,    # per-sample cosine adjacency, gated residual
    },
}


# ---------------------------------------------------------------------------
# ZeroGCNBlock
# ---------------------------------------------------------------------------
class ZeroGCNBlock(nn.Module):
    """Single ShiftFuse-Zero block.

    Supports two spatial mixing modes:
      multi_gcn=False (default): BRASP+SGPShift → A_learned+block_adj → Conv1×1
      multi_gcn=True  (late):   BRASP+SGPShift → sum_k W_k(A_k @ x), K=3 structural

    Args:
        in_channels:        Input channels.
        out_channels:       Output channels.
        stride:             Temporal stride.
        A_flat:             (V,V) flat adjacency for BRASP.
        A_intra:            (V,V) intra-part adjacency for SGPShift.
        A_inter:            (V,V) inter-part adjacency for SGPShift.
        A_learned:          Shared nn.Parameter (K,V,V) or (V,V); None if multi_gcn.
        je:                 Shared JointEmbedding (or None).
        tla:                TemporalLandmarkAttention (or None).
        drop_path_rate:     Stochastic depth probability.
        dropout:            TCN dropout rate.
        num_joints:         V (default 25).
        use_se:             Enable ChannelSE recalibration.
        K_adj:              K for A_learned blending (used when multi_gcn=False).
        multi_gcn:          Use per-partition GCN instead of shared graph_conv.
        A_gcn_partitions:   List of K (V,V) normalized structural adjacency tensors;
                            required when multi_gcn=True.
    """

    def __init__(
        self,
        in_channels:       int,
        out_channels:      int,
        stride:            int,
        A_flat:            torch.Tensor,
        A_intra:           torch.Tensor,
        A_inter:           torch.Tensor,
        A_learned:         nn.Parameter,
        je:                nn.Module   = None,
        tla:               nn.Module   = None,
        drop_path_rate:    float       = 0.0,
        dropout:           float       = 0.1,
        num_joints:        int         = 25,
        use_se:            bool        = False,
        K_adj:             int         = 1,
        multi_gcn:         bool        = False,
        A_gcn_partitions:  list        = None,
        use_adyn:          bool        = False,
    ):
        super().__init__()

        self.stride    = stride
        self.multi_gcn = multi_gcn
        self.K_adj     = K_adj

        # ── Zero-param spatial routing (always present) ───────────────────
        self.brasp = BodyRegionShift(channels=in_channels, A=A_flat)
        self.sgp_shift = SGPShift(
            channels=in_channels, A_intra=A_intra, A_inter=A_inter,
            num_joints=num_joints,
        )

        # ── Spatial mixing ────────────────────────────────────────────────
        if multi_gcn:
            # K=3 per-partition GCN: separate Conv1×1 per structural group.
            # Skips A_learned and block_adj — W_k handles per-partition learning.
            assert A_gcn_partitions is not None, "A_gcn_partitions required for multi_gcn"
            self.K_gcn = len(A_gcn_partitions)
            for k, A_k in enumerate(A_gcn_partitions):
                self.register_buffer(f'_gcn_A_{k}', A_k)
            self.gcn_convs = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
                for _ in range(self.K_gcn)
            ])
            self.gcn_bn  = nn.BatchNorm2d(out_channels)
            self.gcn_act = nn.Hardswish(inplace=True)

            # A_dynamic: per-sample cosine similarity adjacency (gated residual).
            # Gate init -4.0 → sigmoid(-4) ≈ 0.018, near-zero at epoch 1.
            # Activates gradually as training progresses.
            self.use_adyn = use_adyn
            if use_adyn:
                embed_dim = max(in_channels // 4, 8)
                self.adyn_proj = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
                self.adyn_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.adyn_gate = nn.Parameter(torch.tensor(-4.0))
            else:
                self.use_adyn = False
        else:
            # Standard path: per-block adaptive adjacency + single graph_conv
            self.block_adj = nn.Parameter(
                torch.eye(num_joints, dtype=torch.float32)
                + 0.01 * torch.randn(num_joints, num_joints)
            )
            self._A_learned = A_learned
            if K_adj > 1:
                # Per-block blend weights for K-matrix A_learned
                self.adj_alpha = nn.Parameter(torch.zeros(K_adj))
            self.graph_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(inplace=True),
            )

        # ── Channel recalibration ─────────────────────────────────────────
        self.se = ChannelSE(out_channels) if use_se else nn.Identity()

        # ── Temporal modelling ────────────────────────────────────────────
        self.tcn       = MultiScaleTCN(out_channels, stride=stride, dropout=dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # ── Optional addons ───────────────────────────────────────────────
        self.je  = je
        self.tla = tla

        # ── Residual ──────────────────────────────────────────────────────
        if (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.out_relu = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # ── Zero-param spatial routing ────────────────────────────────────
        x = self.brasp(x)
        x = self.sgp_shift(x)

        B, C, T, V = x.shape

        if self.multi_gcn:
            # K=3 per-partition: y = (1/K) * sum_k W_k(A_k @ x)
            # Divide by K_gcn to normalise activation variance before BN.
            x_flat = x.reshape(B, C * T, V)
            y = None
            for k in range(self.K_gcn):
                A_k = getattr(self, f'_gcn_A_{k}')
                x_k = torch.matmul(x_flat, A_k).reshape(B, C, T, V)
                out_k = self.gcn_convs[k](x_k)
                y = out_k if y is None else y + out_k
            y = y / self.K_gcn   # variance normalisation

            # A_dynamic: per-sample cosine similarity graph (gated residual).
            if self.use_adyn:
                embed   = self.adyn_proj(x).mean(dim=2)          # (B, embed_dim, V)
                embed_n = F.normalize(embed, dim=1)               # unit-norm per joint
                A_dyn   = torch.bmm(embed_n.transpose(1, 2), embed_n)  # (B, V, V)
                x_dyn   = torch.matmul(x_flat, A_dyn).reshape(B, C, T, V)
                y = y + torch.sigmoid(self.adyn_gate) * self.adyn_conv(x_dyn)

            x = self.gcn_act(self.gcn_bn(y))
        else:
            # A_learned correction (K=1 or K=3 blended)
            x_flat = x.reshape(B, C * T, V)
            if self.K_adj > 1:
                weights = torch.softmax(self.adj_alpha, dim=0)
                x_agg = torch.zeros_like(x)
                for k in range(self.K_adj):
                    A_l = self._A_learned[k].abs()
                    d   = A_l.sum(dim=1).clamp(min=1e-6).pow(-0.5)
                    A_l_norm = d.unsqueeze(1) * A_l * d.unsqueeze(0)
                    x_agg = x_agg + weights[k] * torch.matmul(x_flat, A_l_norm).reshape(B, C, T, V)
            else:
                A_l = self._A_learned.abs()
                d   = A_l.sum(dim=1).clamp(min=1e-6).pow(-0.5)
                A_l_norm = d.unsqueeze(1) * A_l * d.unsqueeze(0)
                x_agg = torch.matmul(x_flat, A_l_norm).reshape(B, C, T, V)

            x_spatial = x + x_agg
            x_flat2   = x_spatial.reshape(B, C * T, V)
            x_scaled  = torch.matmul(x_flat2, self.block_adj).reshape(B, C, T, V)
            x = self.graph_conv(x_scaled)

        # ── Channel recalibration ─────────────────────────────────────────
        x = self.se(x)

        # ── Joint embedding (shared per stage) ────────────────────────────
        if self.je is not None:
            x = self.je(x)

        # ── Temporal modelling ────────────────────────────────────────────
        x = self.drop_path(self.tcn(x))

        # ── Residual ──────────────────────────────────────────────────────
        x = self.out_relu(x + self.residual(res))

        # ── Optional TLA ──────────────────────────────────────────────────
        if self.tla is not None:
            x = self.tla(x)

        return x


# ---------------------------------------------------------------------------
# ShiftFuseZero  (nano / small, and sub-backbone for small_late)
# ---------------------------------------------------------------------------
class ShiftFuseZero(nn.Module):
    """ShiftFuse-Zero: early-fusion skeleton action recogniser.

    Also serves as the per-stream-pair sub-backbone inside ShiftFuseZeroLate.

    Args:
        num_classes:    Output classes.
        variant:        'nano', 'small', or 'small_late' (sub-backbone).
        in_channels:    Channels per stream (default 3).
        graph_layout:   Skeleton layout (default 'ntu-rgb+d').
        num_joints:     V (default 25).
        dropout:        Override variant default dropout.
        use_se:         Override variant default SE flag.
        use_k3_adj:     Override variant default K=3 adjacency flag.
        num_streams:    Number of input streams (4 for nano/small, 2 for sub-backbone).
        stream_names:   Which stream keys to read from stream_dict.
        multi_gcn:      Override variant multi_gcn flag.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'nano',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        use_se:       bool  = None,
        use_k3_adj:   bool  = None,
        num_streams:  int   = 4,
        stream_names: list  = None,
        multi_gcn:    bool  = None,
        use_adyn:     bool  = None,
    ):
        super().__init__()

        if variant not in ZERO_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(ZERO_VARIANTS.keys())}")

        cfg             = ZERO_VARIANTS[variant]
        stem_ch         = cfg['stem_channels']
        channels        = cfg['channels']
        num_blocks      = cfg['num_blocks']
        strides         = cfg['strides']
        drop_path_rate  = cfg['drop_path_rate']
        tla_landmarks   = cfg['tla_landmarks']
        tla_reduce      = cfg['tla_reduce_ratio']
        use_se          = use_se       if use_se       is not None else cfg.get('use_se', False)
        _dropout        = dropout      if dropout      is not None else cfg['dropout']
        use_k3_adj      = use_k3_adj   if use_k3_adj   is not None else cfg.get('use_k3_adj', False)
        multi_gcn       = multi_gcn    if multi_gcn    is not None else cfg.get('multi_gcn', False)
        _use_adyn       = use_adyn     if use_adyn     is not None else cfg.get('use_adyn', False)
        K_adj           = 3 if use_k3_adj else 1

        self.variant      = variant
        self.num_classes  = num_classes
        # Stream names: defaults to first num_streams of the 4 standard streams
        _default_names = ['joint', 'velocity', 'bone', 'bone_velocity']
        self.stream_names = stream_names if stream_names is not None else _default_names[:num_streams]

        # ── 1. Semantic body-part graph ───────────────────────────────────
        graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',
            max_hop=2,
            raw_partitions=True,
        )
        A_raw = graph.A                          # (3, V, V) raw 0/1
        A_sym = normalize_symdigraph_full(A_raw) # (3, V, V) normalized

        A_intra = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat  = torch.tensor((A_raw.sum(0) > 0).astype('float32'))

        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # Pre-normalised structural partitions for multi_gcn (K=3)
        A_gcn_partitions = None
        if multi_gcn:
            A_gcn_partitions = [
                torch.tensor(A_sym[k], dtype=torch.float32)
                for k in range(A_sym.shape[0])   # K=3
            ]

        # ── 2. Early fusion stem ──────────────────────────────────────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=num_streams,
        )

        # ── 3. Build stages ───────────────────────────────────────────────
        total_blocks     = sum(num_blocks)
        block_idx_global = 0

        self.stages = nn.ModuleList()

        # Per-stage shared A_learned (only for multi_gcn=False)
        if not multi_gcn:
            for i in range(len(channels)):
                if K_adj > 1:
                    param = nn.Parameter(torch.full((K_adj, num_joints, num_joints), 0.01))
                else:
                    param = nn.Parameter(torch.full((num_joints, num_joints), 0.01))
                setattr(self, f'stage{i}_A_learned', param)

        prev_ch = stem_ch

        for stage_idx in range(len(channels)):
            stage_ch      = channels[stage_idx]
            stage_stride  = strides[stage_idx]
            n_blocks      = num_blocks[stage_idx]
            is_last_stage = (stage_idx == len(channels) - 1)

            stage_je  = JointEmbedding(stage_ch, num_joints)
            stage_tla = (
                TemporalLandmarkAttention(stage_ch, tla_landmarks, tla_reduce)
                if is_last_stage else None
            )

            A_learned_param = (
                getattr(self, f'stage{stage_idx}_A_learned') if not multi_gcn else None
            )

            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                dp_rate = drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                stride  = stage_stride if (block_idx == 0 and stage_idx > 0) else 1
                c_in    = prev_ch if block_idx == 0 else stage_ch

                block = ZeroGCNBlock(
                    in_channels      = c_in,
                    out_channels     = stage_ch,
                    stride           = stride,
                    A_flat           = A_flat,
                    A_intra          = A_intra,
                    A_inter          = A_inter,
                    A_learned        = A_learned_param,
                    je               = stage_je,
                    tla              = stage_tla if (block_idx == n_blocks - 1) else None,
                    drop_path_rate   = dp_rate,
                    dropout          = _dropout,
                    num_joints       = num_joints,
                    use_se           = use_se,
                    K_adj            = K_adj,
                    multi_gcn        = multi_gcn,
                    A_gcn_partitions = A_gcn_partitions,
                    use_adyn         = _use_adyn,
                )
                stage_blocks.append(block)
                block_idx_global += 1

            self.stages.append(stage_blocks)
            prev_ch = stage_ch

        # ── 4. Classifier head ────────────────────────────────────────────
        final_ch = channels[-1]
        self.pool_gate  = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(final_ch, num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        # Unpack streams (handle M=2 multi-body input)
        streams = []
        for name in self.stream_names:
            s = stream_dict[name]
            if s.dim() == 5:
                s = s[..., 0]
            streams.append(s)

        # Early fusion
        x = self.fusion(streams)

        # Backbone
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)

        # Gated GAP+GMP pooling
        x   = x.mean(dim=2)                   # (B, C, V)
        gap = x.mean(dim=-1)                   # (B, C)
        gmp = x.max(dim=-1).values             # (B, C)
        g   = torch.sigmoid(self.pool_gate)
        x   = g * gap + (1 - g) * gmp         # (B, C)

        return self.classifier(x)


# ---------------------------------------------------------------------------
# ShiftFuseZeroLate  (2-backbone late fusion)
# ---------------------------------------------------------------------------
class ShiftFuseZeroLate(nn.Module):
    """Late-fusion ShiftFuse-Zero.

    Two independent sub-backbones with proper K=3 per-partition GCN:
      Backbone A: joint + velocity  → logits_A
      Backbone B: bone  + bone_vel  → logits_B
      Final: (logits_A + logits_B) / 2

    The per-partition GCN (multi_gcn=True) gives each backbone a separate
    learned spatial transform W_k for each structural adjacency partition
    (intra-part, inter-part, multi-hop), replacing the shared A_learned +
    block_adj approach.  Targeting ~87–89% with ~260–310K params.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'small_late',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        use_se:       bool  = None,
    ):
        super().__init__()

        shared_kwargs = dict(
            num_classes  = num_classes,
            variant      = variant,
            in_channels  = in_channels,
            graph_layout = graph_layout,
            num_joints   = num_joints,
            dropout      = dropout,
            use_se       = use_se if use_se is not None else False,
            use_k3_adj   = False,    # replaced by multi_gcn structural partitions
            num_streams  = 2,
            multi_gcn    = True,
        )

        self.backbone_a = ShiftFuseZero(
            **shared_kwargs,
            stream_names=['joint', 'velocity'],
        )
        self.backbone_b = ShiftFuseZero(
            **shared_kwargs,
            stream_names=['bone', 'bone_velocity'],
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        logits_a = self.backbone_a(stream_dict)
        logits_b = self.backbone_b(stream_dict)
        return (logits_a + logits_b) / 2


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_shiftfuse_zero(variant: str = 'nano', num_classes: int = 60, **kwargs) -> ShiftFuseZero:
    """Build nano or small ShiftFuse-Zero."""
    return ShiftFuseZero(num_classes=num_classes, variant=variant, **kwargs)


def build_shiftfuse_zero_late(variant: str = 'small_late', num_classes: int = 60, **kwargs) -> ShiftFuseZeroLate:
    """Build small_late (2-backbone late fusion) ShiftFuse-Zero."""
    return ShiftFuseZeroLate(num_classes=num_classes, variant=variant, **kwargs)
