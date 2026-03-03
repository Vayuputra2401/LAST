"""
MultiScaleAdaptiveGCN — K-Subset Multi-Scale Spatial GCN.

Combines the ST-GCN paradigm (K separate adjacency subsets, one weight matrix
per subset) with CTR-GCN-style per-sample adaptive topology (Q/K attention).

Key upgrade over the previous AdaptiveCTRGCN:
  OLD: K=3 subsets summed into ONE adjacency → single weight matrix for all hops
  NEW: K=3 subsets kept SEPARATE → K independent weight matrices
       "self-loop / centripetal / centrifugal" each learn distinct spatial ops

Design (per forward):
  1. Shared Q/K projections over x → per-sample V×V attention map A_dyn
  2. For each subset k ∈ {0, ..., K-1}:
       A_k = A_physical_k + alpha * A_dyn   (static + adaptive blend)
       x_agg_k = A_k @ x                   (graph aggregation)
       h_k = GroupConv_k(x_agg_k)          (per-subset group conv)
  3. out = x + BN(sum_k h_k)               (residual + BN)

Param count per stage (channels C, joints V=25, groups G, K subsets):
  Q/K projections:   2 × C × (d_k × G)  where d_k = C // (G × 4)
  K group convs:     K × G × (C//G)²   = K × C²/G
  BN:                2C
  alpha:             1  (scalar, shared across all K and G)
  ────────────────────────────────────────────────────────
  Example — small, C=96, G=4, K=3:
    Q/K:    2 × 96 × (6×4) = 4,608
    convs:  3 × 4 × 576    = 6,912
    BN:     192
    alpha:  1
    total:  11,713
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAdaptiveGCN(nn.Module):
    """K-subset spatial GCN + shared per-sample adaptive topology.

    One instance is created per stage and shared by reference across all blocks
    in that stage (same weight-sharing pattern as before).

    Args:
        channels:     C — input/output feature channels.
        A:            (K, V, V) float tensor — pre-computed raw adjacency subsets.
                      Each subset is row-normalised inside __init__.
        num_joints:   V (default 25).
        num_groups:   G — channel groups per subset conv (default 4; use 2 for nano).
        reduce_ratio: r — channel reduction for Q/K attention (default 4).
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
        K, V, _ = A.shape
        self.K = K
        self.G = num_groups

        # ── K separate row-normalised adjacency buffers ───────────────────────
        for k in range(K):
            Ak = A[k].clone().float()
            row_sum = Ak.sum(-1, keepdim=True).clamp(min=1e-6)
            self.register_buffer(f'A_{k}', Ak / row_sum)   # row sums = 1.0

        # ── K independent group convolutions (one per subset) ────────────────
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, groups=num_groups, bias=False)
            for _ in range(K)
        ])

        # ── Shared Q/K for per-sample adaptive topology ───────────────────────
        # d_k = attention dim per group; shared across all K subsets
        self.d_k = max(1, channels // (num_groups * reduce_ratio))
        self.query = nn.Conv2d(channels, self.d_k * num_groups, 1, bias=False)
        self.key   = nn.Conv2d(channels, self.d_k * num_groups, 1, bias=False)

        # ── Scalar gate: blends physical + adaptive adjacency ─────────────────
        # Init 0.1 → tanh(0.1)≈0.10 → Q/K convs get ~10% gradient from epoch 1.
        # Zero-init causes dead Q/K weights throughout warmup.
        self.alpha = nn.Parameter(torch.full((1,), 0.1))   # no_decay in trainer

        # ── BN over full channel dim ──────────────────────────────────────────
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V)
        """
        B, C, T, V = x.shape

        # ── Shared per-sample adaptive topology ──────────────────────────────
        q = self.query(x).mean(dim=2)           # (B, d_k*G, V)  — pool over T
        k = self.key(x).mean(dim=2)             # (B, d_k*G, V)
        q = q.view(B, self.G, self.d_k, V)      # (B, G, d_k, V)
        k = k.view(B, self.G, self.d_k, V)
        A_dyn = torch.einsum('bgdv,bgdw->bgvw', q, k) / (self.d_k ** 0.5)
        A_dyn = F.softmax(A_dyn, dim=-1).mean(dim=1)   # (B, V, V) avg over groups

        alpha = torch.tanh(self.alpha)          # scalar in (-1, 1)

        # ── K-subset aggregation ──────────────────────────────────────────────
        out = torch.zeros_like(x)
        for k_idx in range(self.K):
            A_k = getattr(self, f'A_{k_idx}').unsqueeze(0) + alpha * A_dyn   # (B, V, V)
            x_agg = torch.einsum('bvw,bctw->bctv', A_k, x)                   # (B, C, T, V)
            out = out + self.convs[k_idx](x_agg)

        return x + self.bn(out)   # residual


# ── Backwards-compatibility alias ────────────────────────────────────────────
AdaptiveCTRGCN = MultiScaleAdaptiveGCN
