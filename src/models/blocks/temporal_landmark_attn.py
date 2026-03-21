"""
TemporalLandmarkAttention (TLA) — Efficient O(T×K) temporal attention.

Replaces both MultiScaleTSM (zero learning) and full T×T LightweightTemporalAttention
(O(T²) compute) with a learned, O(T×K) attention over K landmark frames.

Two anchor modes:
  fixed (default):    K uniformly-spaced frame indices — fast, no extra params.
  learnable:          K scalars (sigmoid → [0,1] → ×(T-1)) learned via bilinear
                      interpolation — model discovers discriminative frame positions.
                      Only K extra no-decay params (anchor_logits).

Design rationale:
  - TSM shifts channels by fixed ±2, ±4 — no learning, identical for all actions.
  - Full T×T attention: 64×64=4096 dot-products per head per block — expensive.
  - T×K with K=14: 64×14=896 dot-products — 4.6× cheaper than T², still global reach.

Forward:
  x (B, C, T, V)
    → pool over V → (B, T, C)
    → extract/interpolate K landmark frames: x_l (B, K, C)
    → Q = Linear(C → d_k)(all T frames)     (B, T, d_k)
    → K = Linear(C → d_k)(landmarks)        (B, K, d_k)
    → V = Linear(C → d_k)(landmarks)        (B, K, d_k)
    → attn = softmax(Q @ K^T / √d_k)        (B, T, K)  — float32 for AMP safety
    → out  = attn @ V → Linear(d_k → C)    (B, T, C)
    → gate × out.unsqueeze(-1) added back to x (B, C, T, V)

Params (d_k = max(4, C//reduce_ratio), default reduce_ratio=8):
  4 × C × d_k + 1          (Q/K/V/proj + gate)         — always
  + K                       (anchor_logits, no-decay)   — learnable mode only
  Example C=192, d_k=24: 4×192×24 + 1 = 18,433 params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLandmarkAttention(nn.Module):
    """O(T×K) temporal attention over K landmark frames.

    Args:
        channels:          C — input/output channels.
        num_landmarks:     K — number of landmark frames (default 14).
        reduce_ratio:      r — channel reduction for Q/K/V (default 8 → d_k=C//8).
        learnable_anchors: If True, K anchor positions are learned via bilinear
                           interpolation instead of fixed uniform spacing.
    """

    def __init__(
        self,
        channels:          int  = 128,
        num_landmarks:     int  = 14,
        reduce_ratio:      int  = 8,
        learnable_anchors: bool = True,
    ):
        super().__init__()
        self.K                 = num_landmarks
        self.d_k               = max(4, channels // reduce_ratio)
        self.learnable_anchors = learnable_anchors

        self.q_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.k_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.v_proj   = nn.Linear(channels, self.d_k, bias=False)
        self.out_proj = nn.Linear(self.d_k, channels, bias=False)

        # Gate: sigmoid(-4) ≈ 0.018 at init → near-identity pass-through.
        # Required: TLA injects attended features from epoch 1; without this gate
        # the attention output (random at init) adds noise to every frame.
        # Gate fades in gradually as projections learn. Added to no_decay via '.gate'.
        self.gate = nn.Parameter(torch.full((1,), -4.0))

        if learnable_anchors:
            # K scalars initialised to uniform spacing in logit space.
            # sigmoid(logits) × (T-1) → frame positions learned end-to-end.
            # Excluded from weight decay via 'anchor_logits' in name.
            uniform = torch.linspace(0.05, 0.95, num_landmarks)
            # inverse sigmoid so initial positions ≈ uniform after sigmoid
            self.anchor_logits = nn.Parameter(torch.log(uniform / (1 - uniform)))

    def _get_landmarks(self, x_t: torch.Tensor, T: int) -> torch.Tensor:
        """Extract K landmark frames from (B, T, C) sequence.

        Fixed mode:     integer index slicing — O(1), no grad through positions.
        Learnable mode: bilinear interpolation — differentiable anchor positions.
        """
        if not self.learnable_anchors:
            stride  = max(1, T // self.K)
            indices = list(range(0, T, stride))[: self.K]
            return x_t[:, indices, :]                        # (B, K, C)

        # Learnable: sigmoid → [0,1] → continuous positions in [0, T-1]
        pos       = torch.sigmoid(self.anchor_logits) * (T - 1)   # (K,)
        floor_idx = pos.long().clamp(0, T - 2)                     # (K,)
        ceil_idx  = (floor_idx + 1).clamp(0, T - 1)               # (K,)
        frac      = (pos - floor_idx.float()).unsqueeze(0)         # (1, K)

        # Bilinear interpolation: differentiable, gradients flow to anchor_logits
        x_floor = x_t[:, floor_idx, :]    # (B, K, C)
        x_ceil  = x_t[:, ceil_idx,  :]    # (B, K, C)
        return x_floor + frac.unsqueeze(-1) * (x_ceil - x_floor)  # (B, K, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x + gate × temporal-landmark-attended features
        """
        B, C, T, V = x.shape

        # Pool spatial dim → temporal sequence
        x_t  = x.mean(dim=-1).permute(0, 2, 1)     # (B, T, C)

        # Landmark frames
        x_land = self._get_landmarks(x_t, T)        # (B, K, C)

        # Projections — float32 for softmax stability under AMP bfloat16/float16
        q = self.q_proj(x_t.float())                # (B, T, d_k)
        k = self.k_proj(x_land.float())             # (B, K, d_k)
        v = self.v_proj(x_land.float())             # (B, K, d_k)

        # T × K attention
        scale = self.d_k ** 0.5
        attn  = torch.bmm(q, k.transpose(1, 2)) / scale   # (B, T, K) float32
        attn  = F.softmax(attn, dim=-1)

        # Weighted combination of landmark features
        out = torch.bmm(attn, v)                           # (B, T, d_k)
        out = self.out_proj(out).to(x.dtype)               # (B, T, C)

        # Gated residual: broadcasts over V
        out = out.permute(0, 2, 1).unsqueeze(-1)           # (B, C, T, 1)
        return x + torch.sigmoid(self.gate) * out
