"""
TemporalLandmarkAttention (TLA) — Efficient O(T×K) temporal attention.

Replaces both MultiScaleTSM (zero learning) and full T×T LightweightTemporalAttention
(O(T²) compute) with a learned, O(T×K) attention over K uniformly-spaced landmark frames.

Design rationale:
  - TSM shifts channels by fixed ±2, ±4 — no learning, identical for all actions.
  - Full T×T attention: 64×64=4096 dot-products per head per block — expensive.
  - T×K with K=8: 64×8=512 dot-products — 8× cheaper, still global temporal reach.
    Each of T frames attends to 8 landmark frames spread across the sequence.
    Forces the model to summarise via temporal landmarks (phase structure of actions).

Forward:
  x (B, C, T, V)
    → pool over V → (B, T, C)
    → extract K landmark frames: x_l (B, K, C) [uniformly spaced]
    → Q = Linear(C → d_k)(all T frames)     (B, T, d_k)
    → K = Linear(C → d_k)(landmarks)        (B, K, d_k)
    → V = Linear(C → d_k)(landmarks)        (B, K, d_k)
    → attn = softmax(Q @ K^T / √d_k)        (B, T, K)  — float32 for AMP safety
    → out  = attn @ V → Linear(d_k → C)    (B, T, C)
    → gate × out.unsqueeze(-1) added back to x (B, C, T, V)

Params (d_k = max(4, C//reduce_ratio), default reduce_ratio=8):
  4 × C × d_k + 1  (Q/K/V/proj Linears + gate scalar)
  Example C=128, d_k=16: 4×128×16 + 1 = 8,193 params
  vs full attention at same d_k: same params, 8× less compute

Reference: Longformer (Beltagy et al., 2020) — local+global attention;
  adapted here as landmark-only global attention for skeleton sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLandmarkAttention(nn.Module):
    """O(T×K) temporal attention over K uniformly-spaced landmark frames.

    Args:
        channels:       C — input/output channels.
        num_landmarks:  K — number of landmark frames (default 8).
        reduce_ratio:   r — channel reduction for Q/K/V (default 8 → d_k=C//8).
    """

    def __init__(self, channels: int, num_landmarks: int = 8, reduce_ratio: int = 8):
        super().__init__()
        self.K   = num_landmarks
        self.d_k = max(4, channels // reduce_ratio)

        self.q_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.k_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.v_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.out_proj = nn.Linear(self.d_k, channels, bias=False)

        # Gate: sigmoid(0) = 0.5 → attention active at 50% from epoch 1.
        # Excluded from weight decay via '.gate' in name (trainer.py no_decay).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x + gate × temporal-landmark-attended features
        """
        B, C, T, V = x.shape

        # Pool spatial dim → temporal sequence
        x_t = x.mean(dim=-1).permute(0, 2, 1)   # (B, T, C)

        # Uniformly-spaced landmark indices
        stride   = max(1, T // self.K)
        indices  = list(range(0, T, stride))[: self.K]
        x_land   = x_t[:, indices, :]            # (B, K, C)

        # Projections — float32 for softmax stability under AMP float16
        q  = self.q_proj(x_t.float())            # (B, T, d_k)
        k  = self.k_proj(x_land.float())         # (B, K, d_k)
        v  = self.v_proj(x_land.float())         # (B, K, d_k)

        # T × K attention
        scale = self.d_k ** 0.5
        attn  = torch.bmm(q, k.transpose(1, 2)) / scale   # (B, T, K)
        attn  = F.softmax(attn, dim=-1).to(x.dtype)        # back to input dtype

        # Weighted combination of landmark features
        out = torch.bmm(attn, v.to(x.dtype))               # (B, T, d_k)
        out = self.out_proj(out.float()).to(x.dtype)        # (B, T, C)

        # Gated residual: broadcasts (B, C, T, 1) over V
        out = out.permute(0, 2, 1).unsqueeze(-1)           # (B, C, T, 1)
        return x + torch.sigmoid(self.gate) * out
