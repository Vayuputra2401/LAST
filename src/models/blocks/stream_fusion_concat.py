"""
Stream fusion utilities.

StreamFusionConcat — EfficientGCN-style input-fusion (kept for reference / LAST-E).
MultiStreamStem    — Late-fusion stems: 4 independent BN+Conv1×1 projections,
                     one per stream (joint / velocity / bone / bone_velocity).
                     Used by ShiftFuse-GCN v7+ for 4-stream late fusion.

Late fusion rationale
─────────────────────
  Input fusion (StreamFusionConcat): all streams merged before backbone.
    Backbone operates on a blended representation from frame 1 → cannot
    specialise per stream → ceiling at ~83%.

  Late fusion (MultiStreamStem): each stream processed by independent stem,
    then the SAME backbone weights are applied to each stream separately
    (stacked along batch dim for efficiency).  4 independent heads produce
    4 logit vectors that are averaged at inference.  Each head specialises
    in one stream's discriminative features → SOTA gap closed (+3-4%).
"""

import torch
import torch.nn as nn


class StreamFusionConcat(nn.Module):
    """Concatenation-based stream fusion (input fusion).

    Per-stream BN → Concat along C → Conv1×1 → BN → ReLU.

    Args:
        in_channels:  Input channels per stream (3 for x,y,z).
        out_channels: Output channels (C₀ of Stage 1).
        num_streams:  Number of input streams (default 3).
    """

    def __init__(self, in_channels: int, out_channels: int, num_streams: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_streams = num_streams

        # Independent per-stream batch normalization on raw input
        self.stream_bn = nn.ModuleList([
            nn.BatchNorm2d(in_channels) for _ in range(num_streams)
        ])

        # Concatenate all streams → project to out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * num_streams, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, streams: list) -> torch.Tensor:
        """
        Args:
            streams: List of tensors, each (B, in_channels, T, V).

        Returns:
            fused: (B, out_channels, T, V)
        """
        assert len(streams) == self.num_streams, (
            f"Expected {self.num_streams} streams, got {len(streams)}"
        )

        normed = [self.stream_bn[i](s) for i, s in enumerate(streams)]
        return self.proj(torch.cat(normed, dim=1))


class MultiStreamStem(nn.Module):
    """4 independent BN+Conv1×1 stems for late-fusion training.

    Each of the 4 streams (joint / velocity / bone / bone_velocity) has its
    own BN(3) + Conv1×1(3→C0) + BN(C0) + Hardswish stem.  There is NO
    cross-stream concatenation here — the backbone is shared and receives
    each stream individually (stacked along the batch dimension).

    Args:
        in_channels:   Channels per stream (3: x, y, z).
        out_channels:  C₀ — backbone stem width.
        num_streams:   Number of streams (default 4).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32,
                 num_streams: int = 4):
        super().__init__()
        self.num_streams = num_streams
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardswish(inplace=True),
            )
            for _ in range(num_streams)
        ])

    def forward(self, streams: list) -> torch.Tensor:
        """Project each stream independently, then stack along batch dim.

        Args:
            streams: List of num_streams tensors, each (B, 3, T, V).

        Returns:
            stacked: (num_streams × B, C0, T, V) — ready for one backbone pass.
        """
        assert len(streams) == self.num_streams, (
            f"Expected {self.num_streams} streams, got {len(streams)}"
        )
        projected = [self.stems[i](s) for i, s in enumerate(streams)]
        return torch.cat(projected, dim=0)   # (4B, C0, T, V)
