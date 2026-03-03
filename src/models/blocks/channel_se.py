"""
ChannelSE — Squeeze-Excitation channel attention block.

Used after the spatial GCN in ShiftFuseBlock to recalibrate which channels
are discriminative after graph propagation (EfficientGCN-style).

Architecture:
  x (B, C, T, V)
  → AdaptiveAvgPool2d(1, 1)    # (B, C, 1, 1)
  → Flatten                    # (B, C)
  → Linear(C → C//r)           # squeeze
  → ReLU
  → Linear(C//r → C)           # excite
  → Sigmoid                    # gate in [0, 1]
  → reshape + multiply         # scale channels of x

Parameters: 2 × C × (C // reduce_ratio)  [no bias]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSE(nn.Module):
    """
    Squeeze-Excitation channel recalibration.

    Args:
        channels (int): number of input channels C
        reduce_ratio (int): bottleneck reduction factor r (default 4)
    """

    def __init__(self, channels: int, reduce_ratio: int = 4):
        super().__init__()
        mid = max(4, channels // reduce_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            x * channel_scale: (B, C, T, V)
        """
        # Squeeze: (B, C, T, V) → (B, C)
        s = self.avg_pool(x).view(x.size(0), -1)
        # Excite: (B, C) → (B, C)
        scale = torch.sigmoid(self.fc(s))
        # Reshape and scale: (B, C) → (B, C, 1, 1)
        return x * scale.view(x.size(0), x.size(1), 1, 1)
