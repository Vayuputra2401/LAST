import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamFusion(nn.Module):
    """
    Fuses N input streams (joint, velocity, bone) into a single feature tensor.

    Each stream is independently normalized via its own BN2d, then projected
    through a single shared 1×1 conv stem. Per-channel softmax weights blend
    the per-stream projections into one fused tensor.

    This is 3× cheaper than running a separate backbone per stream (v2 MIB)
    because the backbone only runs once after fusion.

    v2 change — channel-wise stream weights:
    Old: 3 scalar softmax weights shared across all C0 channels.
    New: (3, C0) weight matrix — each of the C0 output channels independently
         learns which stream to prefer. Channel 7 can prefer velocity
         (motion dynamics); channel 23 can prefer bone (structural angles).
    Extra cost: 3×C0 − 3 params (e.g. +117 for base). Negligible.

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

        # Independent per-stream batch normalization — each stream has its own
        # scale (joint ≈ 1, velocity ≈ 0.02–0.05, bone ≈ 0.3–1).
        self.stream_bn = nn.ModuleList([
            nn.BatchNorm2d(in_channels) for _ in range(num_streams)
        ])

        # Shared 1×1 conv stem: same weights project every stream to C0.
        # No bias — BN above already handles centering.
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(out_channels)
        self.stem_relu = nn.ReLU(inplace=True)

        # Per-channel per-stream blend weights.
        # Shape: (num_streams, out_channels) — softmax over num_streams dim.
        # Zeros → softmax = 1/num_streams (equal blend) at init for every channel.
        # During training each channel's weights diverge independently.
        self.stream_weights = nn.Parameter(torch.zeros(num_streams, out_channels))

    def forward(self, streams: list) -> torch.Tensor:
        """
        Args:
            streams: List of tensors, each (B, in_channels, T, V).
                     Order must match the BN registration order
                     (e.g., [joint, velocity, bone]).

        Returns:
            fused: (B, out_channels, T, V)
        """
        assert len(streams) == self.num_streams, (
            f"Expected {self.num_streams} streams, got {len(streams)}"
        )

        # Per-channel softmax across streams → (num_streams, C0).
        # Each channel gets its own blend ratio; sums to 1 per channel.
        weights = F.softmax(self.stream_weights, dim=0)  # (num_streams, C0)

        fused = None
        for i, x in enumerate(streams):
            # Per-stream BN (handles differing input scales)
            x = self.stream_bn[i](x)
            # Shared stem projection
            x = self.stem(x)
            x = self.stem_bn(x)
            x = self.stem_relu(x)
            # Channel-wise weighted accumulation — weight shape (C0,) broadcast to (1,C0,1,1)
            w = weights[i].view(1, self.out_channels, 1, 1)
            fused = w * x if fused is None else fused + w * x

        return fused
