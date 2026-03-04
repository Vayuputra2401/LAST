"""
Temporal convolution blocks for ShiftFuse-GCN.

EpSepTCN
--------
Ported from EfficientGCN's Temporal_Sep_Layer.  MobileNetV2-style inverted
bottleneck:
    1. Expand:    Conv2d(C → rC, 1×1) + BN + act
    2. Depthwise: Conv2d(rC → rC, (k,1), groups=rC) + BN + act
    3. Pointwise: Conv2d(rC → C, 1×1) + BN
    4. Residual:  handled by outer ShiftFuseBlock (no inner residual)

MultiScaleEpSepTCN
------------------
Parallel multi-branch temporal convolution.  Three modes:

  num_branches=2  (nano legacy):
    Branch 0: EpSep k=3
    Branch 1: EpSep k=5

  num_branches=3  (small legacy):
    Branch 0: EpSep k=3
    Branch 1: EpSep k=5
    Branch 2: MaxPool(k=3)

  num_branches=4  (v8 — EfficientGCN+Shift-GCN style, recommended):
    Branch 0: TSM — 0-param temporal shift (±1 frame)  ← replaces d=1 conv
    Branch 1: EpSep k=3, dilation=2  (5-frame receptive field)
    Branch 2: EpSep k=3, dilation=4  (9-frame receptive field)
    Branch 3: MaxPool(k=3)
    → Receptive fields: [1-frame shift, 5f, 9f, 3f pooling]
    → Param saving: TSM has 0 params vs d=1 branch; net cost ≈ old 3-branch

All branches split C channels equally (C // num_branches per branch).
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

        # No internal residual — ShiftFuseBlock's outer residual handles skip.
        # Stride is applied by depth_conv when stride > 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T', V) where T' = T // stride
        """
        out = self.act(self.expand_conv(x))
        out = self.act(self.depth_conv(out))
        out = self.point_conv(out)
        return out


class MultiScaleEpSepTCN(nn.Module):
    """Multi-scale parallel temporal convolution.

    Supports three branch configurations; see module docstring for details.

    Args:
        channels:     Input/output channels.
        stride:       Temporal stride (applied to all branches).
        expand_ratio: EpSepTCN-style channel expansion (default: 2).
        num_branches: 2 / 3 (legacy) or 4 (v8 dilated+TSM, recommended).
        act:          Activation function (default: nn.Hardswish).
    """

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
        num_branches: int = 4,
        act: nn.Module = None,
    ):
        super().__init__()
        assert channels % num_branches == 0, (
            f"channels ({channels}) must be divisible by num_branches ({num_branches})"
        )
        self.num_branches = num_branches
        self.act = act if act is not None else nn.Hardswish(inplace=True)

        if num_branches == 4:
            self._init_4branch(channels, stride, expand_ratio)
        else:
            self._init_legacy(channels, stride, expand_ratio, num_branches)

        # Final pointwise mix — shared across all modes
        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    # ------------------------------------------------------------------ #
    #  4-branch init  (TSM + d=2 + d=4 + MaxPool)                        #
    # ------------------------------------------------------------------ #
    def _init_4branch(self, channels: int, stride: int, expand_ratio: int):
        C_b   = channels // 4
        inner = C_b * expand_ratio

        # Branch 0 — TSM: 0-param temporal shift; AvgPool to handle stride
        if stride > 1:
            self.tsm_downsample = nn.Sequential(
                nn.AvgPool2d((stride, 1), stride=(stride, 1)),
                nn.BatchNorm2d(C_b),
            )
        else:
            self.tsm_downsample = nn.BatchNorm2d(C_b)

        # Branch 1 — EpSep k=3, dilation=2  (pad = d*(k-1)//2 = 2)
        self.d2_expand  = nn.Sequential(nn.Conv2d(C_b, inner, 1, bias=False), nn.BatchNorm2d(inner))
        self.d2_depth   = nn.Sequential(
            nn.Conv2d(inner, inner, (3, 1), stride=(stride, 1),
                      padding=(2, 0), dilation=(2, 1), groups=inner, bias=False),
            nn.BatchNorm2d(inner),
        )
        self.d2_project = nn.Sequential(nn.Conv2d(inner, C_b, 1, bias=False), nn.BatchNorm2d(C_b))

        # Branch 2 — EpSep k=3, dilation=4  (pad = 4)
        self.d4_expand  = nn.Sequential(nn.Conv2d(C_b, inner, 1, bias=False), nn.BatchNorm2d(inner))
        self.d4_depth   = nn.Sequential(
            nn.Conv2d(inner, inner, (3, 1), stride=(stride, 1),
                      padding=(4, 0), dilation=(4, 1), groups=inner, bias=False),
            nn.BatchNorm2d(inner),
        )
        self.d4_project = nn.Sequential(nn.Conv2d(inner, C_b, 1, bias=False), nn.BatchNorm2d(C_b))

        # Branch 3 — MaxPool(k=3)
        self.maxpool_4 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(C_b),
        )

    # ------------------------------------------------------------------ #
    #  Legacy init  (num_branches = 2 or 3)                               #
    # ------------------------------------------------------------------ #
    def _init_legacy(self, channels: int, stride: int, expand_ratio: int, num_branches: int):
        C_b    = channels // num_branches
        inner  = C_b * expand_ratio
        kernels = [3, 5][:num_branches]   # k=3 for nb=2; k=3,k=5 for nb=3 (MaxPool takes slot 2)

        self.expand_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C_b, inner, 1, bias=False),
                nn.BatchNorm2d(inner),
            ) for _ in range(num_branches - (1 if num_branches == 3 else 0))
        ])

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

        if num_branches == 3:
            self.maxpool_branch = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(C_b),
            )

        self.point_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner, C_b, 1, bias=False),
                nn.BatchNorm2d(C_b),
            ) for _ in range(len(kernels))
        ])

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T', V) where T' = T // stride
        """
        if self.num_branches == 4:
            return self._forward_4branch(x)

        # Legacy (num_branches = 2 or 3)
        x_splits    = x.chunk(self.num_branches, dim=1)
        branch_outs = []
        for i in range(len(self.depth_convs)):
            h = self.act(self.expand_convs[i](x_splits[i]))
            h = self.act(self.depth_convs[i](h))
            h = self.point_convs[i](h)
            branch_outs.append(h)

        if self.num_branches == 3:
            branch_outs.append(self.maxpool_branch(x_splits[2]))

        out = torch.cat(branch_outs, dim=1)
        return self.mix_conv(out)

    def _forward_4branch(self, x: torch.Tensor) -> torch.Tensor:
        x_splits = x.chunk(4, dim=1)   # 4 × (B, C//4, T, V)

        # Branch 0: TSM — shift half channels +1, half channels -1 in time
        C_b  = x_splits[0].shape[1]
        half = C_b // 2
        h0   = torch.cat([
            torch.roll(x_splits[0][:, :half],  1, dims=2),   # shift +1 (future)
            torch.roll(x_splits[0][:, half:], -1, dims=2),   # shift -1 (past)
        ], dim=1)
        h0 = self.tsm_downsample(h0)   # BN (+ AvgPool when stride > 1)

        # Branch 1: k=3, d=2
        h1 = self.act(self.d2_expand(x_splits[1]))
        h1 = self.act(self.d2_depth(h1))
        h1 = self.d2_project(h1)

        # Branch 2: k=3, d=4
        h2 = self.act(self.d4_expand(x_splits[2]))
        h2 = self.act(self.d4_depth(h2))
        h2 = self.d4_project(h2)

        # Branch 3: MaxPool
        h3 = self.maxpool_4(x_splits[3])

        out = torch.cat([h0, h1, h2, h3], dim=1)
        return self.mix_conv(out)
