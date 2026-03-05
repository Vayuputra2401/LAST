"""
ShiftFuseExperimental — Option 3 lean variant of ShiftFuse-GCN.

Architecture differences from v9 (shiftfuse_gcn.py 'small'):
  REMOVED:  LightweightTemporalAttention after each block  (~233K params)
  ADDED:    MultiScaleTSM after each block                  (0 params)

Everything else is identical to v9 small:
  BRASP → [pw_conv if in≠out] → JointEmbed → AdaptiveGCN → SE
       → 4-branch TCN → DropPath → res+out → MultiScaleTSM

Channels: [64, 128, 256]  (1:2:4 ratio, same as v9 small)
Blocks:   [1, 2, 3]       (6 total, same as v9 small)
Params:   ~820K           (vs 1.06M for v9 small)

All novel components preserved:
  ✓ BRASP (anatomical channel routing, 0 params)
  ✓ JointEmbed placed BEFORE GCN (novel placement)
  ✓ K=3 + per-sample adaptive topology (CTR-GCN style)
  ✓ SE channel recalibration
  ✓ 4-stream late fusion + IB prototype loss
  ✗ Global T×T temporal attention (replaced by MultiScaleTSM ±2/±4 frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.multiscale_tsm import MultiScaleTSM
from .blocks.stream_fusion_concat import MultiStreamStem
from .blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from .graph import Graph, normalize_symdigraph_full
from .shiftfuse_gcn import ShiftFuseBlock, ClassificationHead


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
EXPERIMENTAL_CFG = {
    'stem_channels':   32,
    'channels':        [64, 128, 256],   # 1:2:4 ratio, same as v9 small
    'num_blocks':      [1, 2, 3],
    'strides':         [1, 2, 2],
    'expand_ratio':    2,
    'max_hop':         2,                # same as v9 small
    'num_gcn_groups':  4,
    'drop_path_rate':  0.15,
    'dropout':         0.20,
    'tsm_shift_s':     2,               # short temporal offset (frames)
    'tsm_shift_l':     4,               # long temporal offset (frames)
}


# ---------------------------------------------------------------------------
# ShiftFuseExperimental
# ---------------------------------------------------------------------------
class ShiftFuseExperimental(nn.Module):
    """
    ShiftFuse-Experimental: 4-stream late-fusion skeleton recogniser.

    Replaces TemporalAttention in every block with zero-param MultiScaleTSM.
    All other novel components identical to v9 small.

    Args:
        num_classes:    Output classes (default 60).
        in_channels:    Input channels per stream (default 3).
        graph_layout:   Skeleton graph layout (default 'ntu-rgb+d').
        graph_strategy: Adjacency strategy (default 'spatial').
        T:              Temporal length (default 64).
        num_joints:     Number of skeleton joints (default 25).
        dropout:        Override classifier dropout (None = 0.20).
    """

    def __init__(
        self,
        num_classes: int = 60,
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        T: int = 64,
        num_joints: int = 25,
        dropout: float = None,
    ):
        super().__init__()

        cfg            = EXPERIMENTAL_CFG
        stem_ch        = cfg['stem_channels']
        channels       = cfg['channels']
        num_blocks     = cfg['num_blocks']
        strides        = cfg['strides']
        expand_ratio   = cfg['expand_ratio']
        num_gcn_groups = cfg['num_gcn_groups']
        drop_path_rate = cfg['drop_path_rate']
        tsm_shift_s    = cfg['tsm_shift_s']
        tsm_shift_l    = cfg['tsm_shift_l']
        _dropout       = dropout if dropout is not None else cfg['dropout']

        self.num_streams  = 4
        self.stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']

        # ── 1. Graph adjacency ───────────────────────────────────────────
        self.graph = Graph(
            layout=graph_layout,
            strategy=graph_strategy,
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

        # ── 2. MultiStreamStem (4 independent BN+Conv1×1) ────────────────
        self.fusion = MultiStreamStem(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=4,
        )

        # ── 3. Build stages ──────────────────────────────────────────────
        # Each stage: one shared GCN + N blocks, each block followed by a TSM.
        total_blocks     = sum(num_blocks)
        block_idx_global = 0

        self.stages     = nn.ModuleList()
        self.stage_gcns = nn.ModuleList()
        self.stage_tsms = nn.ModuleList()   # parallel list: one TSM per block

        prev_ch = stem_ch
        T_cur   = T

        for stage_idx in range(len(channels)):
            stage_ch = channels[stage_idx]

            # One GCN per stage, shared by reference across all blocks in stage
            stage_gcn = MultiScaleAdaptiveGCN(
                channels=stage_ch,
                A=A,
                num_joints=num_joints,
                num_groups=num_gcn_groups,
            )
            self.stage_gcns.append(stage_gcn)

            stage_blocks = nn.ModuleList()
            stage_tsms   = nn.ModuleList()

            for blk_idx in range(num_blocks[stage_idx]):
                blk_in     = prev_ch   if blk_idx == 0 else stage_ch
                blk_out    = stage_ch
                blk_stride = strides[stage_idx] if blk_idx == 0 else 1

                # Linear stochastic depth schedule: 0 → drop_path_rate
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
                    use_temporal_attn=False,   # replaced by MultiScaleTSM
                ))

                # One MultiScaleTSM per block (zero params)
                stage_tsms.append(MultiScaleTSM(
                    channels=blk_out,
                    shift_s=tsm_shift_s,
                    shift_l=tsm_shift_l,
                ))

                if blk_idx == 0:
                    T_cur = T_cur // blk_stride

            self.stages.append(stage_blocks)
            self.stage_tsms.append(stage_tsms)
            prev_ch = stage_ch

        # ── 4. Classification heads (one per stream) ─────────────────────
        last_ch = channels[-1]
        self.stream_heads = nn.ModuleList([
            ClassificationHead(last_ch, num_classes, _dropout)
            for _ in range(self.num_streams)
        ])

        # Learned stream ensemble weights (softmax at inference)
        self.stream_weights = nn.Parameter(torch.zeros(self.num_streams))

        # IB prototype loss (InfoGCN-style)
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

        # Small normal init for classifier heads (prevent logit saturation at epoch 0)
        for head in self.stream_heads:
            nn.init.normal_(head.fc.weight, 0, 0.01)
            nn.init.constant_(head.fc.bias, 0)

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------

    def _run_backbone(self, streams: list) -> list:
        """Stack 4 streams, run shared backbone + TSMs, split back.

        Args:
            streams: list of 4 tensors each (B, 3, T, V)
        Returns:
            list of 4 tensors each (B, last_ch, T', V')
        """
        x = self.fusion(streams)   # (4B, stem_ch, T, V)

        for stage_blocks, stage_tsms in zip(self.stages, self.stage_tsms):
            for block, tsm in zip(stage_blocks, stage_tsms):
                x = block(x)       # ShiftFuseBlock (no TemporalAttention)
                x = tsm(x)         # MultiScaleTSM  (0-param ±2/±4 shifts)

        return list(x.chunk(self.num_streams, dim=0))   # 4 × (B, last_ch, T', V')

    # -----------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: Dict{'joint','velocity','bone','bone_velocity'} each (B,3,T,V[,M])
               or a single Tensor broadcast to all 4 streams.
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
                        s = s[..., 0]   # take primary body (M=0)
                    streams.append(s)
            while len(streams) < self.num_streams:
                streams.append(torch.zeros_like(streams[0]))
        else:
            if x.dim() == 5:
                x = x[..., 0]
            streams = [x] * self.num_streams

        # ── Shared backbone ─────────────────────────────────────────────
        feats = self._run_backbone(streams)   # 4 × (B, last_ch, T', V')

        # ── Classification heads ─────────────────────────────────────────
        all_logits, all_features = [], []
        for i, head in enumerate(self.stream_heads):
            logits_i, features_i = head(feats[i])
            all_logits.append(logits_i)
            all_features.append(features_i)

        if self.training:
            # IB loss: mean distance to nearest class prototype
            mean_feat = torch.stack(all_features, dim=0).mean(dim=0)   # (B, C)
            proto_dists = torch.cdist(
                mean_feat.unsqueeze(0),
                self.class_prototypes.unsqueeze(0),
                p=2,
            ).squeeze(0)                                                # (B, num_classes)
            ib_loss = proto_dists.min(dim=-1).values.mean()
            return all_logits, ib_loss

        # ── Eval: softmax-weighted ensemble of raw logits ────────────────
        w = F.softmax(self.stream_weights, dim=0)   # (4,)
        ensemble = sum(w[i] * all_logits[i] for i in range(self.num_streams))
        return ensemble


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_shiftfuse_experimental(num_classes: int = 60, **kwargs) -> ShiftFuseExperimental:
    return ShiftFuseExperimental(num_classes=num_classes, **kwargs)
