"""
MotionGate & HybridGate — Novel channel attention for LAST-E v3.

MotionGate:
    Temporal-difference-based channel gating.  Computes per-channel motion
    energy (|x[t+1] − x[t]|) then passes it through an MLP to produce a
    per-channel gate.  Channels carrying dynamic/motion features get
    upweighted; static channels get downweighted.

    No published skeleton GCN uses this mechanism.  Velocity streams provide
    raw joint-level temporal differences at the INPUT; MotionGate operates on
    INTERMEDIATE spatial-temporal features — a higher-level concept.

HybridGate (large variant only):
    Extends MotionGate with a 3-band FFT spectral descriptor (low/mid/high
    frequency energy) for richer temporal characterisation.  Uses float32 FFT
    for AMP safety.
"""

import torch
import torch.nn as nn


class MotionGate(nn.Module):
    """Motion-Aware Channel Gate (LAST-E v3 novel contribution).

    Args:
        channels:  Number of input/output channels.
        reduction: MLP bottleneck reduction ratio (default: 4).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )
        # Zero-init residual scalar — at startup the block is pure identity.
        self.gate_alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — same shape, gated by motion energy.
        """
        # 1. Temporal difference → per-channel motion energy
        motion = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()  # (B, C, T-1, V)
        energy = motion.mean(dim=(2, 3))                     # (B, C)

        # 2. MLP → channel gate
        gate = self.mlp(energy)                              # (B, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1)              # (B, C, 1, 1)

        # 3. Zero-init residual blend
        return x + self.gate_alpha * (x * gate - x)


class HybridGate(nn.Module):
    """Motion + 3-Band FFT Hybrid Gate (for large variant).

    Combines temporal-diff motion energy (1 × C) with three-band FFT spectral
    descriptor (3 × C) → MLP(4C → C) → per-channel gate.

    Args:
        channels:  Number of input/output channels.
        reduction: MLP bottleneck reduction ratio (default: 4).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )
        self.gate_alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V)
        """
        # Motion descriptor (C)
        motion = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(2, 3))

        # Frequency descriptor (3C) — float32 for AMP safety
        x_f = torch.fft.rfft(x.float(), dim=2)
        power = (x_f.real ** 2 + x_f.imag ** 2).to(x.dtype)
        F = power.shape[2]
        f3 = max(F // 3, 1)
        low = power[:, :, :f3, :].mean(dim=(2, 3))
        mid = power[:, :, f3:2 * f3, :].mean(dim=(2, 3))
        high = power[:, :, 2 * f3:, :].mean(dim=(2, 3))

        # Combine: (B, 4C)
        desc = torch.cat([motion, low, mid, high], dim=1)
        gate = self.mlp(desc).unsqueeze(-1).unsqueeze(-1)

        return x + self.gate_alpha * (x * gate - x)
