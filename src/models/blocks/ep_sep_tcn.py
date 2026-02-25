"""
EpSepTCN — Expanded Separable Temporal Convolution for LAST-E v3.

Ported from EfficientGCN's Temporal_Sep_Layer.  MobileNetV2-style inverted
bottleneck:
    1. Expand:    Conv2d(C → rC, 1×1) + BN + act
    2. Depthwise: Conv2d(rC → rC, (k,1), groups=rC) + BN + act
    3. Pointwise: Conv2d(rC → C, 1×1) + BN
    4. Residual:  identity or strided 1×1 conv

This replaces the 4-branch MultiScaleTCN — single clean layer, fewer params,
multi-scale receptive field achieved via stacking multiple EpSepTCN layers.
"""

import torch
import torch.nn as nn


class EpSepTCN(nn.Module):
    """Expanded-Separable Temporal Convolution.

    Args:
        channels:    Input/output channels (must be equal for residual).
        kernel_size: Temporal kernel size (default: 5).
        stride:      Temporal stride (default: 1).
        expand_ratio: Channel expansion ratio (default: 2).
        act:         Activation function (default: nn.Hardswish).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        expand_ratio: int = 2,
        act: nn.Module = None,
    ):
        super().__init__()
        inner = channels * expand_ratio
        padding = (kernel_size - 1) // 2
        self.act = act if act is not None else nn.Hardswish(inplace=True)

        # Expand
        self.expand_conv = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # Depthwise temporal
        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                inner, inner, (kernel_size, 1),
                stride=(stride, 1), padding=(padding, 0),
                groups=inner, bias=False,
            ),
            nn.BatchNorm2d(inner),
        )

        # Pointwise project back
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Residual path
        if stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T', V) where T' = T // stride
        """
        res = self.residual(x)
        out = self.act(self.expand_conv(x))
        out = self.act(self.depth_conv(out))
        out = self.point_conv(out)
        return out + res
