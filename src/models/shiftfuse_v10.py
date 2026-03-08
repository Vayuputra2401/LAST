"""
ShiftFuse-V10 — SOTA-targeting skeleton action recognition model.

Architecture (V10.2 — efficiency + accuracy fixes):
  1. Semantic Body-Part Graph (SGP): A_intra / A_inter / A_cross
  2. Group Conv GCN (G=4): cross-channel mixing in spatial core
  3. Temporal Landmark Attention (TLA): global temporal via K landmarks
  4. Standard BatchNorm: full-batch stats (nano/small/large all use regular BN)
  5. BRASP after pw_conv: channel-region mapping matches GCN channels
  6. Shared GCN per stage (nano): 3 GCNs instead of 6 → saves ~40K params
     Per-block GCN (small/large): max accuracy at higher param budget

V10.2 nano changes vs V10.1:
  - use_stream_bn=False: regular BN with full batch=64 (was 16/stream) → +2–3pp
  - use_tla=True: TLA enabled (was disabled) → +1–1.5pp
  - share_gcn=True: 1 GCN per stage shared across blocks → saves ~40K params
  - num_blocks=[2,3,2]: 7 blocks (was 6) → +0.5pp depth in stage 2

Variants:
  nano   channels=[32,64,128]   blocks=[2,3,2]  ~160–180K  target 84–87% (→ 88–90% with KD)
  small  channels=[64,128,256]  blocks=[2,3,4]  ~1.5M      target 90–92%
  large  channels=[96,192,384]  blocks=[2,3,4]  ~3.2M      target 92–93%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.temporal_landmark_attn import TemporalLandmarkAttention
from .blocks.stream_fusion_concat import MultiStreamStem
from .blocks.stream_batch_norm import StreamBatchNorm2d
from .blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from .graph import Graph, normalize_symdigraph_full
from .shiftfuse_gcn import ShiftFuseBlock, ClassificationHead


# ---------------------------------------------------------------------------
# Per-stream BN replacement utility
# ---------------------------------------------------------------------------
def _replace_bn2d_with_stream_bn(module: nn.Module, num_streams: int = 4):
    """Recursively replace all nn.BatchNorm2d with StreamBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name,
                    StreamBatchNorm2d(child.num_features, num_streams))
        else:
            _replace_bn2d_with_stream_bn(child, num_streams)


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
V10_VARIANTS = {
    'nano': {
        'stem_channels':   24,
        'channels':        [32, 64, 128],
        'num_blocks':      [2, 3, 2],       # V10.2: 7 blocks (was 6), +1 in stage 2
        'strides':         [1, 2, 2],
        'expand_ratio':    2,
        'max_hop':         2,
        'drop_path_rate':  0.10,
        'dropout':         0.10,
        'tla_landmarks':   8,
        'tla_reduce':      8,
        # V10.2 flags
        'use_tla':         True,     # enabled: global temporal context (+1–1.5pp)
        'share_gcn':       False,    # reverted: per-block GCN (share caused 3× grad → overfit)
        'use_stream_bn':   False,    # regular BN: full batch=64 stats (was 16/stream)
        'share_je':        False,    # reverted: per-block JE (share_je coupled with share_gcn)
        'single_head':     False,    # reverted: 4 heads + logit ensemble (avg features → train-val gap)
    },
    'small': {
        'stem_channels':   32,
        'channels':        [64, 128, 256],
        'num_blocks':      [2, 3, 4],
        'strides':         [1, 2, 2],
        'expand_ratio':    2,
        'max_hop':         2,
        'drop_path_rate':  0.20,
        'dropout':         0.20,
        'tla_landmarks':   8,
        'tla_reduce':      8,
        'use_tla':         True,
        'share_gcn':       False,    # per-block GCN for max accuracy at higher budget
        'use_stream_bn':   False,    # regular BN: full batch stats
        'share_je':        False,
        'single_head':     False,
    },
    'large': {
        'stem_channels':   48,
        'channels':        [96, 192, 384],
        'num_blocks':      [2, 3, 4],
        'strides':         [1, 2, 2],
        'expand_ratio':    2,
        'max_hop':         2,
        'drop_path_rate':  0.30,
        'dropout':         0.25,
        'tla_landmarks':   8,
        'tla_reduce':      8,
        'use_tla':         True,
        'share_gcn':       False,    # per-block GCN for max accuracy
        'use_stream_bn':   False,    # regular BN: full batch stats
        'share_je':        False,
        'single_head':     False,
    },
}


# ---------------------------------------------------------------------------
# ShiftFuseV10
# ---------------------------------------------------------------------------
class ShiftFuseV10(nn.Module):
    """
    ShiftFuse-V10: anatomically-grounded 4-stream skeleton action recogniser.

    Args:
        num_classes:    Output classes (default 60).
        variant:        'nano' | 'small' | 'large' (default 'nano').
        in_channels:    Input channels per stream (default 3).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        T:              Temporal length (default 64).
        num_joints:     Number of skeleton joints (default 25).
        dropout:        Override classifier dropout (None = variant default).
    """

    def __init__(
        self,
        num_classes: int = 60,
        variant: str = 'nano',
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        T: int = 64,
        num_joints: int = 25,
        dropout: float = None,
    ):
        super().__init__()

        if variant not in V10_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(V10_VARIANTS.keys())}"
            )

        cfg            = V10_VARIANTS[variant]
        stem_ch        = cfg['stem_channels']
        channels       = cfg['channels']
        num_blocks     = cfg['num_blocks']
        strides        = cfg['strides']
        expand_ratio   = cfg['expand_ratio']
        drop_path_rate = cfg['drop_path_rate']
        tla_landmarks  = cfg['tla_landmarks']
        tla_reduce     = cfg['tla_reduce']
        _dropout       = dropout if dropout is not None else cfg['dropout']

        # V10.2 flags
        use_tla       = cfg.get('use_tla', True)
        share_gcn     = cfg.get('share_gcn', True)
        use_stream_bn = cfg.get('use_stream_bn', False)
        share_je      = cfg.get('share_je', False)
        single_head   = cfg.get('single_head', False)

        self.variant      = variant
        self.num_streams  = 4
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']
        self.use_tla      = use_tla
        self.single_head  = single_head

        # ── 1. Semantic body-part graph adjacency ─────────────────────────
        self.graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',
            max_hop=cfg['max_hop'],
            raw_partitions=True,
        )
        A_raw = self.graph.A
        A_sym = normalize_symdigraph_full(A_raw)
        A     = torch.tensor(A_sym, dtype=torch.float32)
        self.register_buffer('A', A)

        A_flat = torch.tensor(
            (A_raw.sum(0) > 0).astype('float32')
        )

        # ── 2. MultiStreamStem ───────────────────────────────────────────
        self.fusion = MultiStreamStem(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ──────────────────────────────────────────────
        total_blocks     = sum(num_blocks)
        block_idx_global = 0

        self.stages     = nn.ModuleList()
        self.stage_gcns = nn.ModuleList()   # only used when share_gcn=True
        self.stage_jes  = nn.ModuleList()   # only used when share_je=True
        if use_tla:
            self.stage_tlas = nn.ModuleList()

        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            # Per-stage shared GCN (or None if per-block)
            if share_gcn:
                stage_gcn = MultiScaleAdaptiveGCN(
                    channels=stage_ch, A=A, num_joints=num_joints,
                    num_groups=4, depthwise=False,
                )
                self.stage_gcns.append(stage_gcn)

            # Per-stage shared JointEmbedding (or None if per-block)
            # Only valid at stage_ch dim (all blocks in stage share same C)
            if share_je:
                from .blocks.joint_embedding import JointEmbedding as _JE
                stage_je = _JE(stage_ch, num_joints)
                self.stage_jes.append(stage_je)

            stage_blocks = nn.ModuleList()
            stage_tlas   = nn.ModuleList() if use_tla else None

            for blk_idx in range(num_blocks[stage_idx]):
                blk_in     = prev_ch  if blk_idx == 0 else stage_ch
                blk_out    = stage_ch
                blk_stride = strides[stage_idx] if blk_idx == 0 else 1

                dp_rate = (
                    drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                    if total_blocks > 1 else 0.0
                )
                block_idx_global += 1

                # GCN: shared reference or fresh per-block instance
                if share_gcn:
                    gcn_ref  = stage_gcn
                    reg_flag = False
                else:
                    gcn_ref = MultiScaleAdaptiveGCN(
                        channels=stage_ch, A=A, num_joints=num_joints,
                        num_groups=4, depthwise=False,
                    )
                    reg_flag = True

                # JE: shared per-stage or per-block (default)
                # Note: first block may have blk_in != stage_ch; JE always on blk_out.
                # share_je only applies when blk_in == blk_out (within-stage blocks).
                je_ref = stage_je if (share_je and blk_in == blk_out) else None

                stage_blocks.append(ShiftFuseBlock(
                    in_channels=blk_in,
                    out_channels=blk_out,
                    A_flat=A_flat,
                    T=T_cur,
                    stride=blk_stride,
                    expand_ratio=expand_ratio,
                    num_joints=num_joints,
                    use_dct_gate=False,
                    use_joint_embed=True,
                    use_frame_gate=False,
                    use_bilateral=False,
                    use_multiscale_tcn=True,
                    num_tcn_branches=4,
                    drop_path_prob=dp_rate,
                    gcn=gcn_ref,
                    use_temporal_attn=False,    # TLA handled externally
                    brasp_after_pw=True,        # BRASP on out_channels
                    register_gcn=reg_flag,
                    je=je_ref,                  # shared JE (None = block creates its own)
                ))

                if use_tla:
                    stage_tlas.append(TemporalLandmarkAttention(
                        channels=blk_out,
                        num_landmarks=tla_landmarks,
                        reduce_ratio=tla_reduce,
                    ))

                if blk_idx == 0:
                    T_cur = T_cur // blk_stride

            self.stages.append(stage_blocks)
            if use_tla:
                self.stage_tlas.append(stage_tlas)
            prev_ch = stage_ch

        # ── 4. Per-stream BN replacement ─────────────────────────────────
        # Replace all BN2d in the backbone with StreamBatchNorm2d BEFORE
        # weight init so the init loop finds the sub-BN instances.
        if use_stream_bn:
            _replace_bn2d_with_stream_bn(self.stages, num_streams=4)
            if share_gcn:
                _replace_bn2d_with_stream_bn(self.stage_gcns, num_streams=4)
            if use_tla:
                _replace_bn2d_with_stream_bn(self.stage_tlas, num_streams=4)

        # ── 5. Classification heads ──────────────────────────────────────
        last_ch = channels[-1]
        if single_head:
            # One head on the mean of all stream features — saves ~3×head params.
            # Streams still run independently through backbone (late fusion preserved).
            self.stream_heads = nn.ModuleList([
                ClassificationHead(last_ch, num_classes, _dropout)
            ])
        else:
            self.stream_heads = nn.ModuleList([
                ClassificationHead(last_ch, num_classes, _dropout)
                for _ in range(self.num_streams)
            ])

        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, last_ch) * 0.01
        )

        # ── 6. Weight initialisation ─────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for head in self.stream_heads:
            nn.init.normal_(head.fc.weight, 0, 0.01)
            nn.init.constant_(head.fc.bias, 0)
        self.single_head = single_head

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------

    def _run_backbone(self, streams: list) -> list:
        x = self.fusion(streams)   # (4B, stem_ch, T, V)

        if self.use_tla:
            for stage_blocks, stage_tlas in zip(self.stages, self.stage_tlas):
                for block, tla in zip(stage_blocks, stage_tlas):
                    x = block(x)
                    x = tla(x)
        else:
            for stage_blocks in self.stages:
                for block in stage_blocks:
                    x = block(x)

        return list(x.chunk(self.num_streams, dim=0))

    # -----------------------------------------------------------------------

    def forward(self, x, labels=None):
        """
        Args:
            x:      Dict{'joint','velocity','bone','bone_velocity'} each (B,3,T,V[,M])
                    or a single Tensor broadcast to all 4 streams.
            labels: (B,) integer class labels — used for class-conditional IB loss
                    during training. If None, falls back to nearest-prototype.
        Returns:
            Training: ([logits_j, logits_v, logits_b, logits_bv], ib_loss)
            Eval:     logits (B, num_classes)
        """
        if isinstance(x, dict):
            streams = []
            for name in self.stream_names:
                if name in x:
                    s = x[name]
                    if s.dim() == 5:
                        s = s[..., 0]
                    streams.append(s)
            while len(streams) < self.num_streams:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x] * self.num_streams

        feats = self._run_backbone(streams)

        if self.single_head:
            # Average stream features → run through single head
            head = self.stream_heads[0]
            mean_feat_raw = torch.stack(feats, dim=0).mean(dim=0)  # (B, C, T', V')
            logits, features = head(mean_feat_raw)
            if self.training:
                proto_dists = torch.cdist(
                    features.unsqueeze(0),
                    self.class_prototypes.unsqueeze(0),
                    p=2,
                ).squeeze(0)
                if labels is not None:
                    ib_loss = proto_dists[torch.arange(features.size(0), device=features.device), labels].mean()
                else:
                    ib_loss = proto_dists.min(dim=-1).values.mean()
                return [logits], ib_loss
            return logits

        all_logits, all_features = [], []
        for i, head in enumerate(self.stream_heads):
            logits_i, features_i = head(feats[i])
            all_logits.append(logits_i)
            all_features.append(features_i)

        if self.training:
            mean_feat = torch.stack(all_features, dim=0).mean(dim=0)
            proto_dists = torch.cdist(
                mean_feat.unsqueeze(0),
                self.class_prototypes.unsqueeze(0),
                p=2,
            ).squeeze(0)
            if labels is not None:
                ib_loss = proto_dists[torch.arange(mean_feat.size(0), device=mean_feat.device), labels].mean()
            else:
                ib_loss = proto_dists.min(dim=-1).values.mean()
            return all_logits, ib_loss

        # Eval: uniform average
        ensemble = torch.stack(all_logits, dim=0).mean(dim=0)
        return ensemble


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def create_v10_nano(num_classes: int = 60, **kwargs) -> ShiftFuseV10:
    return ShiftFuseV10(num_classes=num_classes, variant='nano', **kwargs)

def create_v10_small(num_classes: int = 60, **kwargs) -> ShiftFuseV10:
    return ShiftFuseV10(num_classes=num_classes, variant='small', **kwargs)

def create_v10_large(num_classes: int = 60, **kwargs) -> ShiftFuseV10:
    return ShiftFuseV10(num_classes=num_classes, variant='large', **kwargs)
