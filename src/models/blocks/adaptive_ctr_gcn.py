"""
AdaptiveCTRGCN — Per-Sample Adaptive Channel-Topology-Refinement GCN.

Upgrade from CTRLightGCN: adds per-sample Q/K attention-based topology
refinement so each input gets a dynamically adjusted adjacency matrix.

Design
------
  For each group g ∈ {0, ..., G-1}:
    Q_g = Conv1×1(x_g).mean(T)       (B, C//G, V) — joint queries
    K_g = Conv1×1(x_g).mean(T)       (B, C//G, V) — joint keys
    A_adaptive_g = softmax(Q_g^T @ K_g / sqrt(d))  (B, V, V) — per-sample
    A_g = A_physical + α_g * A_adaptive_g           data-dependent topology
    x_agg_g = A_g @ x_group_g
    h_g = GroupConv_g(x_agg_g)

  out = x + BN(concat([h_0, ..., h_{G-1}]))

Key difference from CTRLightGCN:
  - CTRLightGCN: A_group[g] is a fixed (V,V) parameter — same for every sample
  - AdaptiveCTRGCN: A_adaptive_g is computed per sample via Q/K attention
  - Both share A_physical as the static backbone

Param count per stage (channels C, joints V=25, groups G, reduce_ratio r=4):
  Group convs:   G × (C//G)² = C²/G  params
  Q/K projections: G × 2 × Conv1×1(C//G, C//G//r) = 2C²/(G·r)  params
  α gates:       G  params
  BN:            2C  params
  ───────────────────────────────────────────
  Total ≈ C²/G + 2C²/(G·r) + 2C + G

  Example — small, G=4, r=4:
    C=48:  576 + 288 + 96 + 4 = 964
    C=72:  1296 + 648 + 144 + 4 = 2,092
    C=96:  2304 + 1152 + 192 + 4 = 3,652
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCTRGCN(nn.Module):
    """Per-sample adaptive channel-topology-refinement GCN.

    One instance is created per stage and shared by reference across all blocks
    in that stage — same weight-sharing as CTRLightGCN / StaticGCN.

    Args:
        channels:      C — number of input/output feature channels.
        A:             (K, V, V) float tensor — pre-normalised static adjacency.
        num_joints:    V — skeleton joints (default 25).
        num_groups:    G — channel groups (default 4; use 2 for nano).
        reduce_ratio:  r — channel reduction for Q/K (default 4).
    """

    def __init__(
        self,
        channels: int,
        A: torch.Tensor,
        num_joints: int = 25,
        num_groups: int = 4,
        reduce_ratio: int = 4,
    ):
        super().__init__()
        assert channels % num_groups == 0, (
            f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        )
        self.G = num_groups
        self.C_g = channels // num_groups
        V = num_joints
        self.d_k = max(self.C_g // reduce_ratio, 1)  # attention dim

        # ── Static adjacency (shared, row-normalised) ─────────────────────
        A_sum = A.sum(dim=0)
        row_sum = A_sum.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer('A_physical', A_sum / row_sum)

        # ── Per-group Q/K projections for sample-dependent topology ───────
        self.query_convs = nn.ModuleList([
            nn.Conv2d(self.C_g, self.d_k, 1, bias=False)
            for _ in range(num_groups)
        ])
        self.key_convs = nn.ModuleList([
            nn.Conv2d(self.C_g, self.d_k, 1, bias=False)
            for _ in range(num_groups)
        ])

        # ── Per-group gate: blends physical + adaptive topology ───────────
        # Init at 0.1: tanh(0.1)≈0.10 → query/key convs get ~10% gradient from
        # epoch 1. Zero-init (tanh(0)=0) gives zero gradient to query_convs and
        # key_convs for the entire warm-up, making them dead weights initially.
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.1)) for _ in range(num_groups)
        ])

        # ── Per-group spatial projection ──────────────────────────────────
        self.group_convs = nn.ModuleList([
            nn.Conv2d(self.C_g, self.C_g, 1, bias=False)
            for _ in range(num_groups)
        ])

        # ── BN over full channel dim ──────────────────────────────────────
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V)
        """
        B, C, T, V = x.shape
        x_groups = x.chunk(self.G, dim=1)

        outs = []
        for g in range(self.G):
            xg = x_groups[g]  # (B, C_g, T, V)

            # Per-sample adaptive adjacency via Q/K attention
            # Pool over T to get per-joint features
            q = self.query_convs[g](xg).mean(dim=2)  # (B, d_k, V)
            k = self.key_convs[g](xg).mean(dim=2)    # (B, d_k, V)

            # (B, V, V) attention — each sample gets its own topology
            A_adaptive = torch.bmm(
                q.permute(0, 2, 1),   # (B, V, d_k)
                k,                     # (B, d_k, V)
            ) / (self.d_k ** 0.5)
            A_adaptive = F.softmax(A_adaptive, dim=-1)  # row-normalised

            # Blend: static + gated adaptive
            alpha_g = torch.tanh(self.alpha[g])  # [-1, 1], starts at 0
            A_g = self.A_physical.unsqueeze(0) + alpha_g * A_adaptive  # (B, V, V)

            # Aggregate neighbours: A_g[b,v,w] * xg[b,c,t,w] → (B, C_g, T, V)
            x_agg = torch.einsum('bvw,bctw->bctv', A_g, xg)

            outs.append(self.group_convs[g](x_agg))

        x_agg_all = torch.cat(outs, dim=1)
        return x + self.bn(x_agg_all)
