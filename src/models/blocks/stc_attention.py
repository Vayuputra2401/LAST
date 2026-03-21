"""
STCAttention — Spatial-Temporal-Channel Attention (EfficientGCN), residual-gated.

Three attention maps computed from block input and applied multiplicatively:
  A_spatial  (B, 1, 1, V): which joints matter       — softmax over V
  A_temporal (B, 1, T, 1): which frames matter       — sigmoid, lightweight conv
  A_channel  (B, C, 1, 1): channel recalibration     — SE-style (FC → ReLU → FC)

Output: x*(1-scale) + x*A_s*A_t*A_c*scale
  where scale = sigmoid(gate), gate init -4.0 → scale ≈ 0.018 at epoch 0.

The residual gate prevents signal collapse at init (softmax over 25 joints gives
~1/25 per joint; combined with two sigmoid≈0.5 gates → x×0.01 without the fix).
Attention fades in gradually as the gate learns, so GCN/DS-TCN receive full signal
from epoch 1 and all components co-learn from the start.

Reference: EfficientGCN: Constructing Stronger and Faster Baselines for
Skeleton-based Action Recognition, Song et al. 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STCAttention(nn.Module):
    """Spatial-Temporal-Channel attention with residual gate.

    Args:
        channels:     Input/output channel count C.
        num_joints:   V (joint dimension, default 25).
        reduce_ratio: Channel reduction ratio for SE part (default 4).
    """

    def __init__(self, channels: int, num_joints: int = 25, reduce_ratio: int = 4):
        super().__init__()
        C_r = max(channels // reduce_ratio, 4)

        # Spatial: avg(C, T) → (B, V) → Linear(V, V) → softmax → (B, 1, 1, V)
        self.spatial_fc = nn.Linear(num_joints, num_joints, bias=False)

        # Temporal: avg(C, V) → (B, 1, T) → Conv1d(1,1,k=3) → sigmoid → (B, 1, T, 1)
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Channel: avg(T, V) → (B, C) → FC(C→C_r) → ReLU → FC(C_r→C) → sigmoid → (B, C, 1, 1)
        self.channel_fc1 = nn.Linear(channels, C_r, bias=True)
        self.channel_fc2 = nn.Linear(C_r, channels, bias=True)

        # Residual gate: sigmoid(-4) ≈ 0.018 at init → near-identity pass-through.
        # Required because spatial softmax over 25 joints gives ~0.04/joint, combined
        # with temporal(~0.5) and channel(~0.5) → x*0.01. Without this gate that would
        # halve the signal from epoch 1. Gate fades in attention gradually as it learns.
        # Added to no_decay in trainer (name contains 'gate').
        self.gate = nn.Parameter(torch.full((1,), -4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — lerp(x, x*A_s*A_t*A_c, scale)
        """
        B, C, T, V = x.shape

        # Spatial attention — scale by V so mean weight = 1.0 (not 1/V).
        # Without scaling: softmax over 25 joints gives mean 0.04/joint →
        # attention branch contributes <1% even at full gate. ×V restores
        # the spatial map to unit-mean so attention is meaningful when gate opens.
        x_s = x.mean(dim=(1, 2))                              # (B, V)
        A_s = torch.softmax(self.spatial_fc(x_s), dim=-1) * V # (B, V), mean=1.0
        A_s = A_s.view(B, 1, 1, V)

        # Temporal attention
        x_t = x.mean(dim=(1, 3)).unsqueeze(1)                 # (B, 1, T)
        A_t = torch.sigmoid(self.temporal_conv(x_t))          # (B, 1, T)
        A_t = A_t.view(B, 1, T, 1)

        # Channel attention (SE-style)
        x_c = x.mean(dim=(2, 3))                              # (B, C)
        A_c = F.relu(self.channel_fc1(x_c), inplace=True)     # (B, C_r)
        A_c = torch.sigmoid(self.channel_fc2(A_c))            # (B, C)
        A_c = A_c.view(B, C, 1, 1)

        # Residual gate: lerp from identity to full attention
        scale = torch.sigmoid(self.gate)                       # scalar in (0, 1)
        return x * (1.0 - scale) + x * A_s * A_t * A_c * scale
