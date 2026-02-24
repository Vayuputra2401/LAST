"""
StreamFusion v2 — Per-stream stem BatchNorm fix.

Identical to StreamFusion except stem_bn is a ModuleList with one BN per stream
instead of a single shared BN. This fixes the running-stat contamination bug
where joint/velocity/bone features (three very different distributions) were
normalized through one BN, producing incorrect eval-time normalization.

EfficientGCN avoids this by concatenating along the channel dimension (9 input
channels), giving ONE BN a fixed distribution. Our 3-pass approach requires
per-stream BN to achieve the same effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamFusionV2(nn.Module):
    """
    Fuses N input streams (joint, velocity, bone) into a single feature tensor.

    Same logic as StreamFusion, but with per-stream stem_bn to prevent
    BN running-stat contamination across streams.

    Args:
        in_channels:  Input channels per stream (3 for x,y,z)
        out_channels: Output channels (C0 of Stage 1)
        num_streams:  Number of input streams (default 3)
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

        # Shared 1×1 conv stem
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # FIX: Per-stream stem BN instead of shared
        # Each stream's post-stem features get their own clean BN stats.
        # Joint projected features have smooth, centered, moderate variance.
        # Velocity projected features are near-zero for static, spiky for fast.
        # Bone projected features are directionally biased, structure-dependent.
        # A shared BN mixes these distributions, producing incorrect eval stats.
        self.stem_bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in range(num_streams)
        ])

        self.stem_relu = nn.ReLU(inplace=True)

        # Per-channel per-stream blend weights
        self.stream_weights = nn.Parameter(torch.zeros(num_streams, out_channels))

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

        weights = F.softmax(self.stream_weights, dim=0)  # (num_streams, C0)

        fused = None
        for i, x in enumerate(streams):
            x = self.stream_bn[i](x)       # Per-stream input BN
            x = self.stem(x)                # Shared stem projection
            x = self.stem_bn[i](x)          # Per-stream STEM BN (FIX)
            x = self.stem_relu(x)
            w = weights[i].view(1, self.out_channels, 1, 1)
            fused = w * x if fused is None else fused + w * x

        return fused
