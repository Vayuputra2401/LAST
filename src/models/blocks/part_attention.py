"""
PartAttention — Body-part attention for skeleton action recognition.

EfficientGCN Part_Att implementation (att_type='pa'): computes per-channel
importance weights for each of the 5 NTU body parts using a global GAP →
bottleneck FCN → softmax over parts. Each joint in a part inherits its part's
weight vector (C-dimensional, not scalar).

Body parts (NTU-RGB+D, 25 joints):
  0: torso+spine  [0,1,20]
  1: left arm     [4,5,6,7,21,22]
  2: right arm    [8,9,10,11,23,24]
  3: left leg     [12,13,14,15]
  4: right leg    [16,17,18,19]

Pipeline:
  GAP(x) → (B,C,1,1) → Conv(C→C//r) → BN → ReLU →
  Conv(C//r → C×num_parts) → (B,C,1,num_parts) → softmax(dim=3) →
  index_select joints → (B,C,1,V) → expand × x

EfficientGCN design: output is (B,C,T,V), each channel gets a different
part weight assignment (richer than scalar-per-part).

Param count at C=128, r=16: Conv(128→8)+BN(8)+Conv(8→640) = 1024+16+5120 = 6,160 ✓

Reference: EfficientGCN: Constructing Stronger and Faster Baselines for
Skeleton-based Action Recognition, Song et al. 2022. (att_type='pa')
"""

import torch
import torch.nn as nn


# NTU-RGB+D 25-joint body part definitions (0-indexed)
NTU25_BODY_PARTS = [
    [0, 1, 20],             # torso + spine
    [4, 5, 6, 7, 21, 22],   # left arm
    [8, 9, 10, 11, 23, 24], # right arm
    [12, 13, 14, 15],       # left leg
    [16, 17, 18, 19],       # right leg
]


class PartAttention(nn.Module):
    """Body-part attention (EfficientGCN Part_Att exact).

    Args:
        channels:     Feature channels C.
        reduce_ratio: Bottleneck reduction ratio (default 4).
        num_joints:   Total joints V (default 25).
        body_parts:   List of joint-index lists (default NTU25_BODY_PARTS).
    """

    def __init__(
        self,
        channels:     int,
        reduce_ratio: int  = 4,
        num_joints:   int  = 25,
        body_parts:   list = None,
    ):
        super().__init__()
        self.parts     = body_parts if body_parts is not None else NTU25_BODY_PARTS
        self.num_parts = len(self.parts)
        inner          = max(channels // reduce_ratio, 4)

        # Build joint → part index mapping buffer
        joints_buf = self._get_corr_joints(num_joints)
        self.register_buffer('joints', joints_buf)

        # GAP → bottleneck → C×num_parts (EfficientGCN-exact)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, channels * self.num_parts, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=3)

    def _get_corr_joints(self, num_joints: int) -> torch.Tensor:
        """Build (V,) tensor mapping each joint index to its part index."""
        joints = [
            p_idx
            for j in range(num_joints)
            for p_idx, part in enumerate(self.parts)
            if j in part
        ]
        # Joints not in any defined part default to part 0
        if len(joints) < num_joints:
            joints = joints + [0] * (num_joints - len(joints))
        return torch.LongTensor(joints[:num_joints])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x scaled by per-channel part weights
        """
        B, C, T, V = x.shape
        # (B, C×num_parts, 1, 1) → (B, C, 1, num_parts) → softmax
        x_att = self.softmax(
            self.fcn(x).view(B, C, 1, self.num_parts)
        )   # (B, C, 1, num_parts)
        # Broadcast part weights to all joints via joint→part map
        x_att = x_att.index_select(3, self.joints).expand_as(x)  # (B, C, T, V)
        return x * x_att
