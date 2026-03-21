"""
ShiftFuse-Zero — Three-model family for NTU-60/120 skeleton action recognition.

Models:
  nano_tiny_efficient   — single backbone, 4-stream early fusion, ~97K params
  small_late_efficient  — 2-backbone late fusion (joint+vel / bone+bone_vel), ~240K
  large_late_efficient  — B4-style mid-network fusion, 3 streams, ~1.09M params

Block pipeline (EfficientZeroBlock):
    BRASP → SGPShift → JE(in_ch) → STCAttention(in_ch)
    → GCN: (1/K) Σ_k W_k(normalize(Ã_k + A_k_learned[k]) @ x)
    → DepthwiseSepTCN → DropPath → residual
    → [TLA at last block of last stage when use_tla=True]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.stream_fusion_concat import StreamFusionConcat
from .blocks.cross_stream_fusion import CrossStreamFusion
from .blocks.body_region_shift import BodyRegionShift
from .blocks.sgp_shift import SGPShift
from .blocks.joint_embedding import JointEmbedding
from .blocks.temporal_landmark_attn import TemporalLandmarkAttention
from .blocks.drop_path import DropPath
from .blocks.stc_attention import STCAttention
from .blocks.dw_sep_tcn import DepthwiseSepTCN
from .graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
ZERO_VARIANTS = {
    # nano_tiny_efficient: single backbone, 4-stream early fusion. ~97K params.
    # TLA enabled with reduce_ratio=16 (C_r=8) to stay sub-100K.
    # drop_path=0.10 for better regularisation vs previous 0.05.
    'nano_tiny_efficient': {
        'stem_channels':       24,
        'channels':            [32, 64, 128],
        'num_blocks':          [1, 1, 1],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.10,
        'dropout':             0.10,
        'tla_landmarks':       12,
        'tla_reduce_ratio':    16,   # C_r=8 — keeps TLA cost ~4K params
        'use_efficient_block': True,
        'use_tla':             True,
    },
    # small_late_efficient_bb: per-backbone config for 2-backbone late fusion.
    # Backbone A: joint+velocity. Backbone B: bone+bone_velocity.
    # Two backbones + TLA + CrossStreamFusion ≈ 248K total.
    # TLA enabled (gate init -4.0) — adds ~8K params per backbone (C=128, d_k=16).
    'small_late_efficient_bb': {
        'stem_channels':       24,
        'channels':            [32, 64, 128],
        'num_blocks':          [1, 2, 1],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.10,
        'dropout':             0.10,
        'tla_landmarks':       8,
        'tla_reduce_ratio':    8,
        'use_efficient_block': True,
        'use_tla':             True,
    },
    # large_efficient_3s: single backbone, 3-stream early fusion, B4-scale (~1.1M).
    # Streams: joint + bone + velocity (bone_velocity dropped — noisy standalone).
    # channels=[64,128,192], blocks=[3,4,4] — matched analytically to B4's 1.1M.
    # All EfficientZeroBlock novelties: BRASP, SGPShift, STC-Attn, K=3 GCN,
    # per-block A_k_learned, DS-TCN, TLA with learnable K=14 temporal anchors.
    # Target: >92% NTU-60 xsub (B4=92.1%).
    'large_efficient_3s': {
        'stem_channels':       64,
        'channels':            [64, 128, 192],
        'num_blocks':          [3, 4, 4],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.20,
        'dropout':             0.10,
        'tla_landmarks':       14,
        'tla_reduce_ratio':    8,
        'use_efficient_block': True,
        'use_tla':             True,
        'num_streams':         3,
        'stream_names':        ['joint', 'bone', 'velocity'],
    },
}


# ---------------------------------------------------------------------------
# EfficientZeroBlock
# ---------------------------------------------------------------------------
class EfficientZeroBlock(nn.Module):
    """ShiftFuse-Zero block with EfficientGCN-exact graph + DS-TCN + STC-Attention.

    Pipeline:
        BRASP → SGPShift → JE(in_ch) → STCAttention(in_ch)
        → GCN: (1/K) Σ_k W_k(normalize(Ã_k + A_k_learned[k]) @ x)
        → DepthwiseSepTCN → DropPath → residual → Hardswish

    Ordering rationale:
      1. JE(in_ch) before STC-Attn: spatial attention (softmax over V joints) runs on
         identity-enriched features → attention map knows which joint is which.
      2. STC-Attn before GCN: gates the GCN input so aggregation only propagates
         relevant (attended) signal across edges.
      3. Each block owns its own JE sized to in_channels (not shared) to handle
         in_ch ≠ out_ch at stage boundaries without a channel mismatch.

    Key differences from earlier ZeroGCNBlock:
      - A_k = normalize(Ã_k_fixed + A_k_learned[k]) — EfficientGCN-exact learnable graph
      - STC-Attention (spatial+temporal+channel) with residual gate replaces ChannelSE
      - DepthwiseSepTCN (~8× cheaper than MultiScaleTCN) enables K=3 W_k in budget
      - JE per-block on in_channels (not shared stage JE on out_channels)

    Args:
        in_channels:       Input feature channels.
        out_channels:      Output feature channels.
        stride:            Temporal stride (1 or 2).
        A_flat:            (V,V) flat adjacency for BRASP.
        A_intra:           (V,V) intra-part adjacency for SGPShift.
        A_inter:           (V,V) inter-part adjacency for SGPShift.
        A_gcn_partitions:  List of K (V,V) normalized fixed structural tensors.
        drop_path_rate:    Stochastic depth probability.
        dropout:           DepthwiseSepTCN dropout rate.
        num_joints:        V (default 25).
        stc_reduce_ratio:  Channel reduction ratio for SE part of STC-Attention.
        tla:               Optional TemporalLandmarkAttention (last block of last stage).
    """

    def __init__(
        self,
        in_channels:       int,
        out_channels:      int,
        stride:            int,
        A_flat:            torch.Tensor,
        A_intra:           torch.Tensor,
        A_inter:           torch.Tensor,
        A_gcn_partitions:  list,
        drop_path_rate:    float       = 0.0,
        dropout:           float       = 0.1,
        num_joints:        int         = 25,
        stc_reduce_ratio:  int         = 4,
        tla:               nn.Module   = None,
    ):
        super().__init__()
        self.K_gcn = len(A_gcn_partitions)
        # Per-block learnable adjacency residuals (EfficientGCN-exact)
        # Each block learns its own graph topology corrections, zero-initialized
        self.A_k_learned = nn.ParameterList([
            nn.Parameter(torch.zeros(num_joints, num_joints))
            for _ in range(self.K_gcn)
        ])

        # ── 0-param spatial routing ──────────────────────────────────────────
        self.brasp     = BodyRegionShift(channels=in_channels, A=A_flat)
        self.sgp_shift = SGPShift(
            channels=in_channels, A_intra=A_intra, A_inter=A_inter,
            num_joints=num_joints,
        )

        # ── Joint identity embedding (in_channels — before STC and GCN) ─────
        self.je = JointEmbedding(in_channels, num_joints)

        # ── Fixed structural adjacency (K=3 anatomical partitions) ───────────
        for k, A_k in enumerate(A_gcn_partitions):
            self.register_buffer(f'_A_fixed_{k}', A_k)

        # ── Per-partition GCN convolutions (K separate W_k) ──────────────────
        self.gcn_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.K_gcn)
        ])
        self.gcn_bn  = nn.BatchNorm2d(out_channels)
        self.gcn_act = nn.Hardswish(inplace=True)

        # ── STC-Attention on in_channels (after JE, before GCN) ─────────────
        self.stc_attn = STCAttention(in_channels, num_joints, stc_reduce_ratio)

        # ── Depthwise-separable multi-scale TCN ──────────────────────────────
        self.tcn       = DepthwiseSepTCN(out_channels, out_channels,
                                         stride=stride, dropout=dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # ── Residual ─────────────────────────────────────────────────────────
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.out_act = nn.Hardswish(inplace=True)

        # ── Optional TLA (last block of last stage) ───────────────────────────
        self.tla = tla

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # ── 0-param spatial routing ──────────────────────────────────────────
        x = self.brasp(x)
        x = self.sgp_shift(x)

        # ── JE: inject joint identity before attention ────────────────────────
        x = self.je(x)

        # ── STC-Attention gates the GCN input ────────────────────────────────
        x = self.stc_attn(x)

        # ── GCN: y = (1/K) Σ_k W_k(normalize(Ã_k + A_k_learned[k]) @ x) ────
        B, C, T, V = x.shape
        x_flat = x.reshape(B, C * T, V)
        y = None
        for k in range(self.K_gcn):
            A_fixed  = getattr(self, f'_A_fixed_{k}')
            A_learnt = self.A_k_learned[k]
            A_comb = (A_fixed + A_learnt).abs()
            d      = A_comb.sum(dim=1).clamp(min=1e-6).pow(-0.5)
            A_norm = d.unsqueeze(1) * A_comb * d.unsqueeze(0)
            x_k    = torch.matmul(x_flat, A_norm).reshape(B, C, T, V)
            out_k  = self.gcn_convs[k](x_k)
            y = out_k if y is None else y + out_k
        y = y / self.K_gcn
        x = self.gcn_act(self.gcn_bn(y))

        # ── DS-TCN + DropPath ────────────────────────────────────────────────
        x = self.drop_path(self.tcn(x))

        # ── Residual ─────────────────────────────────────────────────────────
        x = self.out_act(x + self.residual(res))

        # ── TLA ──────────────────────────────────────────────────────────────
        if self.tla is not None:
            x = self.tla(x)

        return x


# ---------------------------------------------------------------------------
# ShiftFuseZero  (backbone — used by all three models)
# ---------------------------------------------------------------------------
class ShiftFuseZero(nn.Module):
    """ShiftFuse-Zero backbone. Supports early fusion (4 streams) and sub-backbone
    roles inside ShiftFuseZeroLate / ShiftFuseZeroLate3 (1–2 streams).

    Args:
        num_classes:    Output classes.
        variant:        One of the keys in ZERO_VARIANTS.
        in_channels:    Channels per input stream (default 3).
        graph_layout:   Skeleton layout (default 'ntu-rgb+d').
        num_joints:     V (default 25).
        dropout:        Override variant default dropout.
        num_streams:    Number of input streams (4 for standalone, 1-2 for sub-backbone).
        stream_names:   Which stream keys to read from the input dict.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'nano_tiny_efficient',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        num_streams:  int   = 4,
        stream_names: list  = None,
    ):
        super().__init__()

        if variant not in ZERO_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(ZERO_VARIANTS.keys())}")

        cfg            = ZERO_VARIANTS[variant]
        stem_ch        = cfg['stem_channels']
        channels       = cfg['channels']
        num_blocks     = cfg['num_blocks']
        strides        = cfg['strides']
        drop_path_rate = cfg['drop_path_rate']
        tla_landmarks  = cfg['tla_landmarks']
        tla_reduce     = cfg['tla_reduce_ratio']
        _dropout       = dropout if dropout is not None else cfg['dropout']
        use_tla        = cfg.get('use_tla', False)
        # Variant can specify stream layout; constructor args override if provided
        num_streams    = num_streams if num_streams != 4 else cfg.get('num_streams', num_streams)

        self.variant     = variant
        self.num_classes = num_classes
        _default_names   = ['joint', 'velocity', 'bone', 'bone_velocity']
        _var_names       = cfg.get('stream_names', None)
        self.stream_names = (
            stream_names if stream_names is not None
            else _var_names if _var_names is not None
            else _default_names[:num_streams]
        )

        # ── 1. Semantic body-part graph ───────────────────────────────────
        graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',
            max_hop=2,
            raw_partitions=True,
        )
        A_raw = graph.A
        A_sym = normalize_symdigraph_full(A_raw)

        A_intra = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat  = torch.tensor((A_raw.sum(0) > 0).astype('float32'))

        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # Pass RAW (unnormalized) partitions to each block.
        # EfficientZeroBlock.forward() normalizes (A_raw_fixed + A_k_learned) once inline.
        # Passing pre-normalized A_sym here would cause double normalization, shrinking
        # the effective adjacency spectrum at every block across 9+ blocks.
        A_gcn_partitions = [
            torch.tensor(A_raw[k], dtype=torch.float32)
            for k in range(A_raw.shape[0])
        ]

        # ── 2. Early-fusion stem ──────────────────────────────────────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=num_streams,
        )

        # ── 3. Build stages ───────────────────────────────────────────────
        total_blocks     = sum(num_blocks)
        block_idx_global = 0
        self.stages      = nn.ModuleList()
        prev_ch          = stem_ch

        for stage_idx in range(len(channels)):
            stage_ch     = channels[stage_idx]
            stage_stride = strides[stage_idx]
            n_blocks     = num_blocks[stage_idx]
            is_last_stage = (stage_idx == len(channels) - 1)

            stage_tla = (
                TemporalLandmarkAttention(stage_ch, tla_landmarks, tla_reduce)
                if (is_last_stage and use_tla) else None
            )

            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                dp_rate = drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                stride  = stage_stride if (block_idx == 0 and stage_idx > 0) else 1
                c_in    = prev_ch if block_idx == 0 else stage_ch

                block = EfficientZeroBlock(
                    in_channels      = c_in,
                    out_channels     = stage_ch,
                    stride           = stride,
                    A_flat           = A_flat,
                    A_intra          = A_intra,
                    A_inter          = A_inter,
                    A_gcn_partitions = A_gcn_partitions,
                    drop_path_rate   = dp_rate,
                    dropout          = _dropout,
                    num_joints       = num_joints,
                    tla              = stage_tla if (block_idx == n_blocks - 1) else None,
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

    def extract_features(self, stream_dict: dict):
        """Run stream fusion + all backbone stages.

        Returns:
            feat:       (2B or B, C, T', V) pre-pooling feature map.
            B:          Original batch size.
            multi_body: Whether M=2 body doubling was applied.
        """
        streams    = []
        B          = None
        multi_body = False
        for name in self.stream_names:
            s = stream_dict[name]
            if s.dim() == 5:
                if B is None:
                    B = s.shape[0]
                multi_body = True
                s = torch.cat([s[..., 0], s[..., 1]], dim=0)
            else:
                if B is None:
                    B = s.shape[0]
            streams.append(s)

        x = self.fusion(streams)
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)
        return x, B, multi_body

    def classify(self, x: torch.Tensor, B: int, multi_body: bool) -> torch.Tensor:
        """Gated GAP+GMP pool → classifier → body average."""
        x   = x.mean(dim=2)
        gap = x.mean(dim=-1)
        gmp = x.max(dim=-1).values
        g   = torch.sigmoid(self.pool_gate)
        x   = g * gap + (1 - g) * gmp
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out

    def forward(self, stream_dict: dict) -> torch.Tensor:
        x, B, multi_body = self.extract_features(stream_dict)
        return self.classify(x, B, multi_body)


# ---------------------------------------------------------------------------
# ShiftFuseZeroLate  (2-backbone late fusion — small_late_efficient)
# ---------------------------------------------------------------------------
class ShiftFuseZeroLate(nn.Module):
    """2-backbone late-fusion ShiftFuse-Zero.

    Backbone A: joint + velocity  → logits_A
    Backbone B: bone  + bone_vel  → logits_B
    Final: (logits_A + logits_B) / 2

    cross_stream=True: after backbone stages, each backbone's features receive a
    gated bottleneck contribution from the other stream before pooling.
    Gate init -4.0 → sigmoid ≈ 0.018, fades in gradually from epoch 0.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'small_late_efficient_bb',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        cross_stream: bool  = False,
    ):
        super().__init__()

        shared_kwargs = dict(
            num_classes  = num_classes,
            variant      = variant,
            in_channels  = in_channels,
            graph_layout = graph_layout,
            num_joints   = num_joints,
            dropout      = dropout,
            num_streams  = 2,
        )

        self.backbone_a = ShiftFuseZero(**shared_kwargs, stream_names=['joint', 'velocity'])
        self.backbone_b = ShiftFuseZero(**shared_kwargs, stream_names=['bone', 'bone_velocity'])

        self.cross_fusion = None
        if cross_stream:
            final_ch = ZERO_VARIANTS[variant]['channels'][-1]
            self.cross_fusion = CrossStreamFusion(final_ch)

    def forward(self, stream_dict: dict) -> torch.Tensor:
        if self.cross_fusion is not None:
            feat_a, B_a, mb_a = self.backbone_a.extract_features(stream_dict)
            feat_b, B_b, mb_b = self.backbone_b.extract_features(stream_dict)
            feat_a, feat_b    = self.cross_fusion(feat_a, feat_b)
            logits_a = self.backbone_a.classify(feat_a, B_a, mb_a)
            logits_b = self.backbone_b.classify(feat_b, B_b, mb_b)
        else:
            logits_a = self.backbone_a(stream_dict)
            logits_b = self.backbone_b(stream_dict)
        return (logits_a + logits_b) / 2


# ---------------------------------------------------------------------------
# ShiftFuseZeroMidFusion  (B4-style mid-network fusion — large_late_efficient)
# ---------------------------------------------------------------------------
class ShiftFuseZeroMidFusion(nn.Module):
    """B4-matched mid-network fusion: 3 streams processed independently in
    stages 1–2, then concatenated and processed by a single shared stage 3.

    Fusion topology (mirrors EfficientGCN-B4):
        joint    → [stem] → [stage1] → [stage2] ─┐
        bone     → [stem] → [stage1] → [stage2] ──┤ concat → FusionConv → [stage3] → TLA → cls
        velocity → [stem] → [stage1] → [stage2] ─┘

    Architecture (tuned to ~1.09M ≈ B4's 1.1M):
        Per-stream: stem=32, channels=[48,96], blocks=[1,2], strides=[1,2]
        Fusion:     Conv2d(288→192) + BN + Hardswish
        Shared:     channels=192, blocks=4, stride=2 at first block
        TLA:        K=14 learnable anchors, reduce_ratio=8 (last shared block)

    Novel additions vs B4:
        BRASP (0-param anatomical shift), SGPShift (0-param semantic grouping),
        STC-Attention, per-block A_k_learned residuals, TLA learnable anchors.
    """

    STREAM_NAMES   = ['joint', 'bone', 'velocity']
    STREAM_STEM    = 32
    STREAM_CH      = [48, 96]
    STREAM_BLOCKS  = [1, 2]
    SHARED_CH      = 192
    SHARED_BLOCKS  = 4
    DROP_PATH_RATE = 0.20
    TLA_K          = 14
    TLA_REDUCE     = 8

    def __init__(
        self,
        num_classes:  int   = 60,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        in_channels:  int   = 3,
        dropout:      float = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        _dropout = dropout if dropout is not None else 0.10

        # ── Graph ─────────────────────────────────────────────────────────
        graph    = Graph(layout=graph_layout, strategy='semantic_bodypart',
                         max_hop=2, raw_partitions=True)
        A_raw    = graph.A
        A_sym    = normalize_symdigraph_full(A_raw)
        A_intra  = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter  = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat   = torch.tensor((A_raw.sum(0) > 0).astype('float32'))
        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)
        A_gcn_parts = [torch.tensor(A_raw[k], dtype=torch.float32)
                       for k in range(A_raw.shape[0])]

        # ── Drop-path schedule ─────────────────────────────────────────────
        n_stream_blk = sum(self.STREAM_BLOCKS)
        total_blocks = 3 * n_stream_blk + self.SHARED_BLOCKS

        # ── Per-stream stems (one per stream, 1-stream input) ──────────────
        self.stream_stems = nn.ModuleList([
            StreamFusionConcat(in_channels=in_channels,
                               out_channels=self.STREAM_STEM, num_streams=1)
            for _ in range(3)
        ])

        # ── Per-stream backbones (stages 1–2, independent weights) ─────────
        # Layout: stream_backbone[stream_idx][stage_idx] = nn.ModuleList of blocks
        self.stream_backbone = nn.ModuleList()
        blk_global = 0
        for _ in range(3):
            prev_ch    = self.STREAM_STEM
            stream_mods = nn.ModuleList()
            for si, (ch, nb) in enumerate(zip(self.STREAM_CH, self.STREAM_BLOCKS)):
                stage_mods = nn.ModuleList()
                for b in range(nb):
                    dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
                    stride = 2 if (b == 0 and si > 0) else 1
                    c_in   = prev_ch if b == 0 else ch
                    stage_mods.append(EfficientZeroBlock(
                        in_channels=c_in, out_channels=ch, stride=stride,
                        A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                        A_gcn_partitions=A_gcn_parts,
                        drop_path_rate=dp, dropout=_dropout,
                        num_joints=num_joints,
                    ))
                    blk_global += 1
                stream_mods.append(stage_mods)
                prev_ch = ch
            self.stream_backbone.append(stream_mods)

        # ── Fusion: concat(3×96=288) → 192 ────────────────────────────────
        fusion_in = 3 * self.STREAM_CH[-1]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in, self.SHARED_CH, 1, bias=False),
            nn.BatchNorm2d(self.SHARED_CH),
            nn.Hardswish(inplace=True),
        )

        # ── Shared stage 3 ─────────────────────────────────────────────────
        tla = TemporalLandmarkAttention(self.SHARED_CH, self.TLA_K, self.TLA_REDUCE)
        self.shared_stage = nn.ModuleList()
        for b in range(self.SHARED_BLOCKS):
            dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
            stride = 2 if b == 0 else 1
            self.shared_stage.append(EfficientZeroBlock(
                in_channels=self.SHARED_CH, out_channels=self.SHARED_CH,
                stride=stride,
                A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                A_gcn_partitions=A_gcn_parts,
                drop_path_rate=dp, dropout=_dropout,
                num_joints=num_joints,
                tla=tla if (b == self.SHARED_BLOCKS - 1) else None,
            ))
            blk_global += 1

        # ── Classifier ─────────────────────────────────────────────────────
        self.pool_gate  = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(self.SHARED_CH, num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        # Collect stream inputs, handle M=2 bodies
        raw = [stream_dict[n] for n in self.STREAM_NAMES]
        B          = raw[0].shape[0]
        multi_body = raw[0].dim() == 5
        if multi_body:
            raw = [torch.cat([s[..., 0], s[..., 1]], dim=0) for s in raw]

        # Per-stream: stem → stage1 → stage2
        feats = []
        for s, stem, stages in zip(raw, self.stream_stems, self.stream_backbone):
            x = stem([s])
            for stage_blocks in stages:
                for block in stage_blocks:
                    x = block(x)
            feats.append(x)

        # Fusion: concat along channel dim → projection
        x = self.fusion_conv(torch.cat(feats, dim=1))

        # Shared stage 3
        for block in self.shared_stage:
            x = block(x)

        # Gated GAP+GMP → classifier
        x   = x.mean(dim=2)
        g   = torch.sigmoid(self.pool_gate)
        x   = g * x.mean(dim=-1) + (1 - g) * x.max(dim=-1).values
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out


def build_shiftfuse_zero_midfusion(num_classes: int = 60, **kwargs) -> ShiftFuseZeroMidFusion:
    """Build large_late_efficient (B4-style mid-network fusion, ~1.09M)."""
    return ShiftFuseZeroMidFusion(num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# ShiftFuseZeroB4  (EfficientGCN-B4-exact Large — our novelties on top)
# ---------------------------------------------------------------------------
class ShiftFuseZeroB4(nn.Module):
    """EfficientGCN-B4-exact Large model with our zero-param novelties on top.

    Architecture mirrors B4-SG (1.1M) exactly:
        Per-stream (J / V / B):
            stem(3 → 64) → stage1[64→96, ×2 blocks] → stage2[96→48, ×2 blocks, stride=2]
        Fusion: concat(3×48 = 144) — no projection conv; first shared block handles dim change
        Shared:
            stage1[144→128, ×3 blocks] → stage2[128→272, ×3 blocks, stride=2, +TLA at last]
        Classifier: Dropout(0.25) → Linear(272 → num_classes)

    Our additions on top of B4:
        BRASP (0-param anatomical shift), SGPShift (0-param semantic grouping),
        per-block A_k_learned adjacency residuals, TLA (K=14 learnable anchors).

    Expected params: ~1.1M (B4 base) + ~20K novelties ≈ 1.12M total.
    """

    STREAM_NAMES   = ['joint', 'velocity', 'bone']
    STREAM_STEM    = 64
    # (in_ch, out_ch, n_blocks, stride_first)
    STREAM_STAGES  = [(64, 96, 2, 1), (96, 48, 2, 2)]
    SHARED_STAGES  = [(144, 128, 3, 1), (128, 272, 3, 2)]
    DROP_PATH_RATE = 0.0   # B4-exact: no stochastic depth
    DROPOUT        = 0.25
    TLA_K          = 14
    TLA_REDUCE     = 8

    def __init__(
        self,
        num_classes:  int   = 60,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        in_channels:  int   = 3,
        dropout:      float = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        _dropout = dropout if dropout is not None else self.DROPOUT

        # ── Graph ─────────────────────────────────────────────────────────
        graph   = Graph(layout=graph_layout, strategy='semantic_bodypart',
                        max_hop=2, raw_partitions=True)
        A_raw   = graph.A
        A_sym   = normalize_symdigraph_full(A_raw)
        A_intra = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat  = torch.tensor((A_raw.sum(0) > 0).astype('float32'))
        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)
        A_gcn_parts = [torch.tensor(A_raw[k], dtype=torch.float32)
                       for k in range(A_raw.shape[0])]

        # ── Drop-path schedule ─────────────────────────────────────────────
        n_stream = sum(nb for _, _, nb, _ in self.STREAM_STAGES)
        n_shared = sum(nb for _, _, nb, _ in self.SHARED_STAGES)
        total_blocks = 3 * n_stream + n_shared

        def _make_block(c_in, c_out, stride, dp, tla=None):
            return EfficientZeroBlock(
                in_channels=c_in, out_channels=c_out, stride=stride,
                A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                A_gcn_partitions=A_gcn_parts,
                drop_path_rate=dp, dropout=_dropout,
                num_joints=num_joints,
                tla=tla,
            )

        blk_global = 0

        # ── Per-stream stems (one per stream, 1-stream input) ──────────────
        self.stream_stems = nn.ModuleList([
            StreamFusionConcat(in_channels=in_channels,
                               out_channels=self.STREAM_STEM, num_streams=1)
            for _ in range(3)
        ])

        # ── Per-stream stages (independent weights per stream) ─────────────
        # stream_stages[stream_idx][stage_idx] = nn.ModuleList of blocks
        # blk_global resets per stream — parallel streams at the same depth must
        # receive identical drop_path rates (not accumulate across 3 streams).
        stream_blocks = sum(nb for (_, _, nb, _) in self.STREAM_STAGES)
        self.stream_stages = nn.ModuleList()
        for _ in range(3):
            blk_global = 0  # reset for each stream — symmetric DP across parallel streams
            prev_ch = self.STREAM_STEM
            per_stream = nn.ModuleList()
            for (c_in, c_out, nb, stride_first) in self.STREAM_STAGES:
                stage_mods = nn.ModuleList()
                for b in range(nb):
                    dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
                    stride = stride_first if b == 0 else 1
                    c_b    = prev_ch if b == 0 else c_out
                    stage_mods.append(_make_block(c_b, c_out, stride, dp))
                    blk_global += 1
                per_stream.append(stage_mods)
                prev_ch = c_out
            self.stream_stages.append(per_stream)
        blk_global = stream_blocks  # resume from end of one stream's block count

        # ── Shared stages (after concat of 3×48=144) ──────────────────────
        self.shared_stages = nn.ModuleList()
        for si, (c_in, c_out, nb, stride_first) in enumerate(self.SHARED_STAGES):
            stage_mods = nn.ModuleList()
            for b in range(nb):
                dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
                stride = stride_first if b == 0 else 1
                c_b    = c_in if b == 0 else c_out
                stage_mods.append(_make_block(c_b, c_out, stride, dp))
                blk_global += 1
            self.shared_stages.append(stage_mods)

        # ── Classifier ─────────────────────────────────────────────────────
        final_ch = self.SHARED_STAGES[-1][1]
        self.pool_gate  = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(final_ch, num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        raw = [stream_dict[n] for n in self.STREAM_NAMES]
        B   = raw[0].shape[0]
        multi_body = raw[0].dim() == 5
        if multi_body:
            raw = [torch.cat([s[..., 0], s[..., 1]], dim=0) for s in raw]

        # Per-stream: stem → stage1 → stage2
        feats = []
        for s, stem, stages in zip(raw, self.stream_stems, self.stream_stages):
            x = stem([s])
            for stage_blocks in stages:
                for block in stage_blocks:
                    x = block(x)
            feats.append(x)

        # Fusion: direct concat (3×48=144, no projection)
        x = torch.cat(feats, dim=1)

        # Shared stages
        for stage_blocks in self.shared_stages:
            for block in stage_blocks:
                x = block(x)

        # Gated GAP+GMP → classifier
        x   = x.mean(dim=2)
        g   = torch.sigmoid(self.pool_gate)
        x   = g * x.mean(dim=-1) + (1 - g) * x.max(dim=-1).values
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out


def build_shiftfuse_zero_b4(num_classes: int = 60, **kwargs) -> ShiftFuseZeroB4:
    """Build large_b4_efficient (B4-exact mid-fusion, ~1.12M)."""
    return ShiftFuseZeroB4(num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_shiftfuse_zero(
    variant: str = 'nano_tiny_efficient', num_classes: int = 60, **kwargs
) -> ShiftFuseZero:
    """Build a ShiftFuseZero model. Stream layout is read from variant config."""
    return ShiftFuseZero(num_classes=num_classes, variant=variant, **kwargs)


def build_shiftfuse_zero_late(
    variant: str = 'small_late_efficient_bb', num_classes: int = 60, **kwargs
) -> ShiftFuseZeroLate:
    """Build small_late_efficient (2-backbone late fusion)."""
    return ShiftFuseZeroLate(num_classes=num_classes, variant=variant, **kwargs)
