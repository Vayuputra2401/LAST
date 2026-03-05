"""
ShiftFuse-V10 — SOTA-targeting skeleton action recognition model.

Architecture advances over V9 / Experimental:
  1. Semantic Body-Part Graph (SGP):
       Replaces spine-direction K=3 with body-part semantic K=3:
       A_intra (same part) | A_inter (adjacent parts) | A_cross (cross-body)
       Zero extra params. More meaningful spatial prior than hop direction.

  2. Depthwise Spatial GCN (DW-GCN):
       Replaces Conv2d(C,C,1,groups=G) with Conv2d(C,C,1,groups=C) in K-subset convs.
       32× reduction in GCN conv params at C=128 (C²/G → C per subset × K).
       Channel mixing flows through adaptive Q/K correction + SE + TCN mix_conv.

  3. Temporal Landmark Attention (TLA):
       Replaces MultiScaleTSM (fixed shifts, zero learning) and full T×T attention.
       O(T×K) with K=8: 8× cheaper compute than T×T, still global temporal reach.
       Learned: each frame attends to K=8 uniformly-spaced landmark frames.

  4. Increased depth (more blocks, same param budget):
       DW-GCN savings fund extra blocks: nano [1,2,2], small [2,3,4], large [2,3,4]
       vs experimental [1,1,1] / [1,2,3].

  5. max_hop=2 for all variants (was max_hop=1 for nano):
       2-hop connections give A_cross non-empty → cross-body edges populated.

Three-level anatomical grounding (novel coherent contribution):
  BRASP:        body-region channel routing (0 params)
  SGP:          body-region graph partition (0 params)
  JointEmbed:   joint identity node features before GCN (V×C params)

Block pipeline (identical to v9 / experimental, with DW-GCN and TLA):
  BRASP → [pw_conv if dim change] → JointEmbed → DW-SGP-GCN → SE
       → 4-branch TCN → DropPath → res+out → TLA

Variants:
  nano   channels=[32,64,128]   blocks=[1,2,2]  ~130–150K  target 87–90%
  small  channels=[64,128,256]  blocks=[2,3,4]  ~450–600K  target 90–92%
  large  channels=[96,192,384]  blocks=[2,3,4]  ~1.2–1.5M  target 92–93%

All with: semantic_bodypart graph, depthwise GCN, TLA, 4-stream late fusion, IB loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.temporal_landmark_attn import TemporalLandmarkAttention
from .blocks.stream_fusion_concat import MultiStreamStem
from .blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from .graph import Graph, normalize_symdigraph_full
from .shiftfuse_gcn import ShiftFuseBlock, ClassificationHead


# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
V10_VARIANTS = {
    'nano': {
        'stem_channels':   24,
        'channels':        [32, 64, 128],
        'num_blocks':      [1, 2, 2],
        'strides':         [1, 2, 2],
        'expand_ratio':    2,
        'max_hop':         2,
        'drop_path_rate':  0.10,
        'dropout':         0.10,
        'tla_landmarks':   8,
        'tla_reduce':      8,
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
    },
}


# ---------------------------------------------------------------------------
# ShiftFuseV10
# ---------------------------------------------------------------------------
class ShiftFuseV10(nn.Module):
    """
    ShiftFuse-V10: anatomically-grounded 4-stream skeleton action recogniser.

    Combines semantic graph partitioning (SGP), depthwise spatial GCN (DW-GCN),
    and temporal landmark attention (TLA) for SOTA accuracy at minimal params.

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

        self.variant      = variant
        self.num_streams  = 4
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']

        # ── 1. Semantic body-part graph adjacency ─────────────────────────
        self.graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',   # SGP: intra / inter / cross-body
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
        self.stage_gcns = nn.ModuleList()
        self.stage_tlas = nn.ModuleList()

        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            # Depthwise GCN shared across all blocks in this stage
            stage_gcn = MultiScaleAdaptiveGCN(
                channels=stage_ch,
                A=A,
                num_joints=num_joints,
                num_groups=4,
                depthwise=True,   # DW-GCN: true depthwise, 32× fewer GCN conv params
            )
            self.stage_gcns.append(stage_gcn)

            stage_blocks = nn.ModuleList()
            stage_tlas   = nn.ModuleList()

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
                    use_dct_gate=False,
                    use_joint_embed=True,
                    use_frame_gate=False,
                    use_bilateral=False,
                    use_multiscale_tcn=True,
                    num_tcn_branches=4,
                    drop_path_prob=dp_rate,
                    gcn=stage_gcn,
                    use_temporal_attn=False,   # TLA is appended separately below
                ))

                stage_tlas.append(TemporalLandmarkAttention(
                    channels=blk_out,
                    num_landmarks=tla_landmarks,
                    reduce_ratio=tla_reduce,
                ))

                if blk_idx == 0:
                    T_cur = T_cur // blk_stride

            self.stages.append(stage_blocks)
            self.stage_tlas.append(stage_tlas)
            prev_ch = stage_ch

        # ── 4. Classification heads ──────────────────────────────────────
        last_ch = channels[-1]
        self.stream_heads = nn.ModuleList([
            ClassificationHead(last_ch, num_classes, _dropout)
            for _ in range(self.num_streams)
        ])

        self.stream_weights   = nn.Parameter(torch.zeros(self.num_streams))
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, last_ch) * 0.01
        )

        # ── 5. Weight initialisation ─────────────────────────────────────
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

        # Classifier FC: small normal init to prevent logit saturation at epoch 0
        for head in self.stream_heads:
            nn.init.normal_(head.fc.weight, 0, 0.01)
            nn.init.constant_(head.fc.bias, 0)

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------

    def _run_backbone(self, streams: list) -> list:
        x = self.fusion(streams)   # (4B, stem_ch, T, V)

        for stage_blocks, stage_tlas in zip(self.stages, self.stage_tlas):
            for block, tla in zip(stage_blocks, stage_tlas):
                x = block(x)
                x = tla(x)

        return list(x.chunk(self.num_streams, dim=0))

    # -----------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: Dict{'joint','velocity','bone','bone_velocity'} each (B,3,T,V[,M])
               or a single Tensor broadcast to all 4 streams.
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
            ib_loss = proto_dists.min(dim=-1).values.mean()
            return all_logits, ib_loss

        w = F.softmax(self.stream_weights, dim=0)
        ensemble = sum(w[i] * all_logits[i] for i in range(self.num_streams))
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
