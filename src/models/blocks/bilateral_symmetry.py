"""
BilateralSymmetryEncoding (BSE) — Bilateral Symmetry Encoding for skeleton GCN.

Exploits the bilateral symmetry of the human skeleton (left arm <-> right arm,
left leg <-> right leg) to compute discriminative features.

Computes left-right differences for 10 symmetric joint pairs, weights them
per-channel, and injects antisymmetrically back to both sides. Also captures
symmetry DYNAMICS via temporal differentiation of the bilateral signal.

Discriminative signal:
  - Symmetric actions (clapping): L-R diff ~ 0, in-phase
  - Anti-phase actions (walking): L-R diff alternates sign over time
  - Asymmetric actions (drinking, throwing): L-R diff is large and persistent

Cost: 2C + 1 parameters per instance (two per-channel weights + gate scalar).
"""

import torch
import torch.nn as nn


class BilateralSymmetryEncoding(nn.Module):
    """
    Encodes bilateral symmetry patterns of the human skeleton.

    For each of 10 symmetric joint pairs (left <-> right), computes:
      1. Bilateral difference (left - right) — static symmetry signal
      2. Temporal diff of bilateral difference — symmetry dynamics

    These are weighted per-channel and injected antisymmetrically:
      left joints  += gate * bilateral_signal
      right joints -= gate * bilateral_signal

    The model learns per-channel whether to SEPARATE (w > 0) or BLEND (w < 0)
    symmetric joints, and how much weight to give symmetry dynamics vs static
    symmetry.

    Args:
        channels: Number of input/output channels.

    Params: 2 * channels + 1
    """

    # 10 symmetric joint pairs for NTU RGB+D 25-joint skeleton.
    # Left arm (6 joints) + Left leg (4 joints) = 10 left joints
    # Right arm (6 joints) + Right leg (4 joints) = 10 right joints
    # Torso [0,1,2,3,20] lies on the midline — no symmetric pair.
    LEFT_JOINTS = [4, 5, 6, 7, 21, 22, 12, 13, 14, 15]
    RIGHT_JOINTS = [8, 9, 10, 11, 23, 24, 16, 17, 18, 19]

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Per-channel weight for static bilateral difference.
        # Zero-init: starts as no-op.
        self.sym_weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Per-channel weight for bilateral velocity (temporal derivative of L-R diff).
        # Captures symmetry dynamics: in-phase vs anti-phase vs static.
        self.sym_vel_weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Residual gate: sigmoid(-2.0) ≈ 0.12, so output ≈ identity at init.
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V) where V = 25 (NTU RGB+D joints).
        Returns:
            (B, C, T, V) with bilateral symmetry features injected.
        """
        left = x[:, :, :, self.LEFT_JOINTS]    # (B, C, T, 10)
        right = x[:, :, :, self.RIGHT_JOINTS]  # (B, C, T, 10)

        # Static bilateral difference
        diff = left - right  # (B, C, T, 10)

        # Bilateral velocity: how symmetry changes over time
        # prepend first frame to maintain T dimension
        diff_vel = torch.diff(diff, dim=2, prepend=diff[:, :, :1, :])  # (B, C, T, 10)

        # Per-channel weighted combination
        bilateral = self.sym_weight * diff + self.sym_vel_weight * diff_vel  # (B, C, T, 10)

        g = torch.sigmoid(self.gate)

        # Antisymmetric injection: left += signal, right -= signal
        mod = torch.zeros_like(x)
        mod[:, :, :, self.LEFT_JOINTS] = g * bilateral
        mod[:, :, :, self.RIGHT_JOINTS] = -g * bilateral

        return x + mod
