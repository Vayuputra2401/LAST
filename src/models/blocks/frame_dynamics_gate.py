"""
FrameDynamicsGate — Temporal frame position awareness (SGN-style).

Each temporal position gets its own per-channel gate logit, allowing the
model to weight frames differently based on their position in the sequence
(e.g., motion onset frames vs. static hold frames).

This is the temporal analog of JointEmbedding: both add positional/identity
awareness at very low parameter cost.

Unlike MotionGate (LAST-E v3) which computes gates from data (temporal
differences), FrameDynamicsGate uses a fixed learned gate per position —
data-independent and edge-friendly.

Implementation: a single (1, C, T, 1) parameter, directly used as the gate
logit.  sigmoid(0) = 0.5 at init, so all frames contribute equally at start.

Cost: C × T params (e.g., 48 × 64 = 3072 for Stage 1 of ShiftFuse-small).
Excluded from weight decay (gate/mask parameter, not a convolution weight).

Reference: SGN (Semantic Graph Neural Networks), Liu et al. 2020.
"""

import torch
import torch.nn as nn


class FrameDynamicsGate(nn.Module):
    """
    Temporal position gate: each frame position has a learned per-channel
    gating coefficient.

    Args:
        channels: Feature channel dimension (C).
        T:        Temporal length.  Must match the T of the input tensor.
    """

    def __init__(self, channels: int, T: int):
        super().__init__()
        # Shape (1, C, T, 1): broadcasts over B and V.
        # Zero-init → sigmoid(0) = 0.5 → all frames equally weighted at init.
        # Excluded from weight decay (see trainer.py no_decay list).
        self.gate_logit = nn.Parameter(torch.zeros(1, channels, T, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x * per-frame gate (broadcast over B, V)
        """
        return x * torch.sigmoid(self.gate_logit)
