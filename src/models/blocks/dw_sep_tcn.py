"""
DepthwiseSepTCN — Depthwise-separable multi-scale temporal convolution.

Replaces MultiScaleTCN with ~8× fewer parameters for the same receptive
field, making room for K=3 per-partition GCN convolutions at the same
total parameter budget.

Two branches (dilation=1, dilation=2), channels split equally, concatenated.

Param comparison (C=128, k=9, out=128):
  MultiScaleTCN (full):  2 × (C//2)² × k = 2 × 64 × 64 × 9 = 73,728
  DepthwiseSepTCN:       2 × (C×k + C×C//2) = 2 × (1152 + 8192) = 18,688  (~4× less)

Dilation padding rule: pad = dilation × (kernel - 1) // 2
  dilation=1, kernel=9 → pad=4
  dilation=2, kernel=9 → pad=8
"""

import torch
import torch.nn as nn


class DepthwiseSepTCN(nn.Module):
    """Depthwise-separable multi-scale temporal convolution.

    Two branches (dilation=1, dilation=2). Each branch:
      DWConv2d(C, C, (k,1), groups=C)  — per-channel temporal mixing
      PWConv2d(C, out//2, 1)           — cross-channel projection

    Branches concatenated along channel dim → out_channels. BN + Hardswish.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels (must be even — split equally per branch).
        stride:       Temporal stride applied on DW conv.
        kernel_size:  Temporal kernel size (default 9).
        dropout:      Dropout probability after activation.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int   = 1,
        kernel_size:  int   = 9,
        dropout:      float = 0.0,
    ):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels must be even for 2-branch split"
        mid  = out_channels // 2
        pad1 = (kernel_size - 1) // 2   # dilation=1
        pad2 = kernel_size - 1           # dilation=2: 2×(k-1)//2 = k-1

        # Branch 1: dilation=1
        self.dw1 = nn.Conv2d(
            in_channels, in_channels, (kernel_size, 1),
            stride=(stride, 1), padding=(pad1, 0),
            groups=in_channels, bias=False,
        )
        self.pw1 = nn.Conv2d(in_channels, mid, 1, bias=False)

        # Branch 2: dilation=2
        self.dw2 = nn.Conv2d(
            in_channels, in_channels, (kernel_size, 1),
            stride=(stride, 1), padding=(pad2, 0),
            dilation=(2, 1), groups=in_channels, bias=False,
        )
        self.pw2 = nn.Conv2d(in_channels, mid, 1, bias=False)

        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.Hardswish(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T, V)
        Returns:
            out: (B, C_out, T', V)  where T' = T // stride
        """
        b1  = self.pw1(self.dw1(x))            # (B, mid, T', V)
        b2  = self.pw2(self.dw2(x))            # (B, mid, T', V)
        out = torch.cat([b1, b2], dim=1)       # (B, out_channels, T', V)
        return self.drop(self.act(self.bn(out)))
