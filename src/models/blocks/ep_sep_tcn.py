"""
Temporal convolution blocks for ShiftFuse-GCN.

EpSepTCN
--------
Ported from EfficientGCN's Temporal_Sep_Layer.  MobileNetV2-style inverted
bottleneck:
    1. Expand:    Conv2d(C → rC, 1×1) + BN + act
    2. Depthwise: Conv2d(rC → rC, (k,1), groups=rC) + BN + act
    3. Pointwise: Conv2d(rC → C, 1×1) + BN
    4. Residual:  identity or strided 1×1 conv

MultiScaleEpSepTCN
------------------
Parallel multi-branch variant for the 'small' model:
    Branch 0: k=3  (local gestures, 3-frame receptive field)
    Branch 1: k=5  (mid-range motion, 5-frame receptive field)
    Branch 2: MaxPool(k=3) (sharp temporal transitions)
Each branch processes C // num_branches channels (depthwise, near-zero cost).
Concat → project back to C via pointwise conv.
Effective receptive field: up to 5 frames vs 5 for single EpSepTCN,
but covers 3 temporal scales simultaneously.
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


class MultiScaleEpSepTCN(nn.Module):
    """Multi-scale parallel temporal convolution (small variant).

    Three parallel branches at different temporal scales, each operating on
    a channel split. Captures local gestures (k=3), mid-range motion (k=5),
    and sharp transitions (MaxPool) simultaneously.

    All branches share the same expand/project outer structure; only the
    depthwise kernel differs. Total params ≈ EpSepTCN (same channel budget).

    Args:
        channels:     Input/output channels.
        stride:       Temporal stride (applied to branches 0 and 1).
        expand_ratio: EpSepTCN-style channel expansion (default: 2).
        num_branches: 2 = (k=3, k=5) for nano; 3 = (k=3, k=5, MaxPool) for small.
        act:          Activation function (default: nn.Hardswish).
    """

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
        num_branches: int = 3,
        act: nn.Module = None,
    ):
        super().__init__()
        assert channels % num_branches == 0, (
            f"channels ({channels}) must be divisible by num_branches ({num_branches})"
        )
        self.num_branches = num_branches
        self.act = act if act is not None else nn.Hardswish(inplace=True)
        C_b = channels // num_branches   # channels per branch
        inner = C_b * expand_ratio
        kernels = [3, 5][:num_branches]  # (k=3), (k=5), MaxPool uses branch 2 slot

        # ── Shared expand (per branch, operates on C_b channels) ─────────
        self.expand_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C_b, inner, 1, bias=False),
                nn.BatchNorm2d(inner),
            ) for _ in range(num_branches - (1 if num_branches == 3 else 0))
        ])

        # ── Depthwise temporal per kernel ─────────────────────────────────
        self.depth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    inner, inner, (k, 1),
                    stride=(stride, 1), padding=((k - 1) // 2, 0),
                    groups=inner, bias=False,
                ),
                nn.BatchNorm2d(inner),
            ) for k in kernels
        ])

        # ── MaxPool branch (branch 2 for num_branches=3) ──────────────────
        if num_branches == 3:
            self.maxpool_branch = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(C_b),
            )

        # ── Shared pointwise project per branch ───────────────────────────
        self.point_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner, C_b, 1, bias=False),
                nn.BatchNorm2d(C_b),
            ) for _ in range(len(kernels))
        ])

        # ── Final mix → C ─────────────────────────────────────────────────
        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # ── Outer residual ────────────────────────────────────────────────
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
        # Split channels into branches
        x_splits = x.chunk(self.num_branches, dim=1)   # each (B, C//nb, T, V)

        branch_outs = []
        # Conv branches (0, 1 for nb=2; 0, 1 for nb=3)
        num_conv = len(self.depth_convs)
        for i in range(num_conv):
            h = self.act(self.expand_convs[i](x_splits[i]))
            h = self.act(self.depth_convs[i](h))
            h = self.point_convs[i](h)
            branch_outs.append(h)

        # MaxPool branch (only when num_branches=3)
        if self.num_branches == 3:
            branch_outs.append(self.maxpool_branch(x_splits[2]))

        out = torch.cat(branch_outs, dim=1)    # (B, C, T', V)
        out = self.mix_conv(out)
        return out + res
