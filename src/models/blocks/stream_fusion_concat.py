"""
StreamFusionConcat — EfficientGCN-style concatenation fusion for LAST-E v3.

Replaces softmax-coupled StreamFusionV2. Instead of competing per-channel
softmax weights (zero-sum gradient interference), concatenates all streams
along the channel dimension and learns an arbitrary linear combination via
a 1×1 convolution.

This is strictly more expressive: any weighted average that softmax could
learn, Conv1×1 can also learn — plus arbitrary linear mixtures.

EfficientGCN does exactly this: Concatenate joint+velocity+bone along
channel dim → single Conv → one unified representation.
"""

import torch
import torch.nn as nn


class StreamFusionConcat(nn.Module):
    """Concatenation-based stream fusion.

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
