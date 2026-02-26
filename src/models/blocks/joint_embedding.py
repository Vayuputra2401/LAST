"""
JointEmbedding — Semantic joint identity embedding (SGN-style).

Adds a learned per-joint bias to the feature map so each joint type
(hand, elbow, knee, etc.) has a distinct identity signal.  Unlike spatial
graph convolution which propagates topology, JointEmbedding encodes the
absolute semantic meaning of each joint.

The embedding is additive and broadcasts over B and T:
    x = x + embed.weight.T                 # embed.weight: (V, C)
                                           # reshaped to (1, C, 1, V) for broadcast

Cost: V × C params (e.g., 25 × 48 = 1200 for Stage 1 of ShiftFuse-small).
Each ShiftFuseBlock has its own JointEmbedding (not shared across blocks).

Reference: SGN (Semantic Graph Neural Networks), Liu et al. 2020.
"""

import torch
import torch.nn as nn


class JointEmbedding(nn.Module):
    """
    Additive per-joint semantic embedding.

    Args:
        channels:   Feature channel dimension (C).
        num_joints: Number of skeleton joints (default 25 for NTU RGB+D).
    """

    def __init__(self, channels: int, num_joints: int = 25):
        super().__init__()
        # nn.Embedding(num_joints, channels) stores a (num_joints, channels) table.
        # Zero-init: no effect at start, lets other components establish signal first.
        self.embed = nn.Embedding(num_joints, channels)
        nn.init.zeros_(self.embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x + per-joint embedding (broadcast over B, T)
        """
        # embed.weight: (V, C) → transpose → (C, V) → (1, C, 1, V)
        joint_bias = self.embed.weight.T.unsqueeze(0).unsqueeze(2)  # (1, C, 1, V)
        return x + joint_bias
