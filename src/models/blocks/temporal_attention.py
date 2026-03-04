"""
LightweightTemporalAttention — Global temporal context via self-attention.

Adds global temporal receptive field after local TCN convolutions.
Pools over joints (V) → temporal self-attention → broadcast back.

Design
------
  x: (B, C, T, V)
  x_t = x.mean(dim=V)              → (B, C, T)
  x_t = x_t.permute(0, 2, 1)      → (B, T, C)
  q = Linear(x_t)                  → (B, T, d_k)
  k = Linear(x_t)                  → (B, T, d_k)
  v = Linear(x_t)                  → (B, T, d_v)
  attn = softmax(q @ k^T / √d_k)  → (B, T, T)
  x_att = attn @ v                 → (B, T, d_v)
  x_att = Proj(x_att)              → (B, T, C)
  out = x + x_att.permute(0,2,1).unsqueeze(-1)  → (B, C, T, V) residual

Parameters: 3 × C × d_k + C × d_v = 4C²/r  (with d_k = d_v = C//r)

Examples (r=4):
  C=32: 4×32×8 = 1024 params
  C=48: 4×48×12 = 2304 params
  C=96: 4×96×24 = 9216 params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightTemporalAttention(nn.Module):
    """Global temporal self-attention pooled over joints.

    Gives the model a global temporal receptive field (full sequence)
    complementing the local TCN convolutions (kernel=5 window).

    Args:
        channels:      C — feature channels.
        reduce_ratio:  r — channel reduction for Q/K/V (default 4).
        dropout:       attention dropout (default 0.0).
    """

    def __init__(
        self,
        channels: int,
        reduce_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_k = max(channels // reduce_ratio, 1)

        self.query = nn.Linear(channels, self.d_k, bias=False)
        self.key   = nn.Linear(channels, self.d_k, bias=False)
        self.value = nn.Linear(channels, self.d_k, bias=False)
        self.proj  = nn.Linear(self.d_k, channels, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Gate: sigmoid(0) = 0.5 → attention at 50% strength from epoch 1.
        # Model can push gate toward 1.0 (if attention helps) or 0 (if not).
        # With xavier init for Q/K/V/proj, attention logits are informative from ep1.
        # Must be in no_decay (trainer 'temporal_attn.' keyword) so WD doesn't suppress it.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V)
        """
        B, C, T, V = x.shape

        # Pool over joints → (B, T, C)
        x_t = x.mean(dim=-1).permute(0, 2, 1)   # (B, T, C)

        # Self-attention over temporal dimension
        q = self.query(x_t)    # (B, T, d_k)
        k = self.key(x_t)      # (B, T, d_k)
        v = self.value(x_t)    # (B, T, d_k)

        # (B, T, T) attention weights
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attend → project back to C
        x_att = torch.bmm(attn, v)   # (B, T, d_k)
        x_att = self.proj(x_att)      # (B, T, C)

        # Broadcast back to (B, C, T, V) and add as gated residual
        x_att = x_att.permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1)
        return x + torch.sigmoid(self.gate) * x_att
