"""
MultiScaleTSM — Multi-Scale Temporal Shift Module.

Zero parameters. Shifts different channel groups by different temporal offsets,
providing multi-scale temporal context without learned weights.

Used in ShiftFuseExperimental as a zero-cost replacement for
LightweightTemporalAttention. Maintains temporal diversity across blocks
by composing short (±2) and long (±4) frame shifts through network depth.

Design (bidirectional shifts per group):
  Channels   0 : C//3          → half +shift_s, half -shift_s  (short-range)
  Channels C//3 : 2*(C//3)     → half +shift_l, half -shift_l  (medium-range)
  Channels 2*(C//3) : C        → unchanged  (anchor / reference features)

Coverage across 6 blocks (shifts compose through depth):
  Block 1–3: ±2 and ±4 frames visible
  Stacked: effective temporal context grows multiplicatively

Cost: 0 parameters. torch.roll is a view operation in PyTorch ≥ 1.11.

Reference: TSM (Lin et al., 2019) extended to multi-scale offsets.
"""

import torch
import torch.nn as nn


class MultiScaleTSM(nn.Module):
    """Zero-parameter multi-scale temporal shift.

    Splits channels into 3 groups and shifts the first two bidirectionally:
      Group 1 (C//3): half channels +shift_s, half channels -shift_s
      Group 2 (C//3): half channels +shift_l, half channels -shift_l
      Group 3 (rest): unchanged — preserves anchor features at current frame

    Args:
        channels: Number of feature channels C.
        shift_s:  Short shift amount in frames (default 2).
        shift_l:  Long shift amount in frames (default 4).
    """

    def __init__(self, channels: int, shift_s: int = 2, shift_l: int = 4):
        super().__init__()
        self.c_s     = channels // 3
        self.c_l     = channels // 3
        self.shift_s = shift_s
        self.shift_l = shift_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V) — same shape, channel groups shifted in time
        """
        c_s, c_l = self.c_s, self.c_l
        half_s   = c_s // 2
        half_l   = c_l // 2

        x_s  = x[:, :c_s]              # short-shift group  (B, c_s, T, V)
        x_l  = x[:, c_s:c_s + c_l]    # long-shift group   (B, c_l, T, V)
        x_id = x[:, c_s + c_l:]        # anchor group       (B, rest, T, V)

        xs_fwd = torch.roll(x_s[:, :half_s],   self.shift_s, dims=2)
        xs_bwd = torch.roll(x_s[:, half_s:],  -self.shift_s, dims=2)
        xl_fwd = torch.roll(x_l[:, :half_l],   self.shift_l, dims=2)
        xl_bwd = torch.roll(x_l[:, half_l:],  -self.shift_l, dims=2)

        return torch.cat([xs_fwd, xs_bwd, xl_fwd, xl_bwd, x_id], dim=1)
