"""
STCAttention — Spatial-Temporal-Channel Attention (EfficientGCN), residual-gated.

Three attention maps computed from block input and applied multiplicatively:
  A_spatial  (B, 1, 1, V): which joints matter       — softmax over V
  A_temporal (B, 1, T, 1): which frames matter       — Frame_Att style (avg+max, k=9)
  A_channel  (B, C, 1, 1): channel recalibration     — SE-style (FC → ReLU → FC)
                            optional (use_channel_se=True)

Output: x*(1-scale) + x*A_s*A_t[*A_c]*scale
  where scale = sigmoid(gate), gate init -4.0 → scale ≈ 0.018 at epoch 0.

The residual gate prevents signal collapse at init (softmax over 25 joints gives
~1/25 per joint; combined with two sigmoid≈0.5 gates → x×0.01 without the fix).
Attention fades in gradually as the gate learns, so GCN/DS-TCN receive full signal
from epoch 1 and all components co-learn from the start.

Temporal branch (Frame_Att style):
  avg_pool(C,V) + max_pool(C,V) → cat → Conv1d(2→1, k=9, pad=4) → sigmoid
  17-frame receptive field vs old k=3 (5 frames). Better captures motion dynamics.

Reference: EfficientGCN: Constructing Stronger and Faster Baselines for
Skeleton-based Action Recognition, Song et al. 2022.
CBAM: Convolutional Block Attention Module, Woo et al. 2018 (Frame_Att style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STCAttention(nn.Module):
    """Spatial-Temporal-Channel attention with residual gate.

    Args:
        channels:        Input/output channel count C.
        num_joints:      V (joint dimension, default 25).
        reduce_ratio:    Channel reduction ratio for SE part (default 4).
        use_channel_se:  Include channel SE branch (default True).
                         Set False for nano/small to save params with minimal accuracy loss.
        use_spatial:     Include spatial attention branch (default True).
                         Set False when PartAttention is active on the same block — both
                         operate on the joint axis (V) around the same GCN and produce
                         competing spatial-importance signals. Disabling the spatial branch
                         leaves the temporal gate intact (18 params) so the block still
                         weights frames before GCN aggregation, while PartAttention handles
                         the spatial role after GCN.
    """

    def __init__(
        self,
        channels:       int,
        num_joints:     int  = 25,
        reduce_ratio:   int  = 4,
        use_channel_se: bool = True,
        use_spatial:    bool = True,
    ):
        super().__init__()
        self.use_channel_se = use_channel_se
        self.use_spatial    = use_spatial

        # Spatial: avg(C, T) → (B, V) → Linear(V, V) → softmax → (B, 1, 1, V)
        if use_spatial:
            self.spatial_fc = nn.Linear(num_joints, num_joints, bias=False)

        # Temporal (Frame_Att style): avg(C,V) + max(C,V) → cat(2,T) →
        # Conv1d(2→1, k=9, pad=4) → sigmoid → (B, 1, T, 1)
        # 17-frame receptive field (dilation=1, k=9): captures ~0.5s at 30fps.
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=9, padding=4, bias=False)

        # Channel SE (optional): avg(T, V) → FC(C→C_r) → ReLU → FC(C_r→C) → sigmoid
        if use_channel_se:
            C_r = max(channels // reduce_ratio, 4)
            self.channel_fc1 = nn.Linear(channels, C_r, bias=True)
            self.channel_fc2 = nn.Linear(C_r, channels, bias=True)

        # Residual gate: sigmoid(-4) ≈ 0.018 at init → near-identity pass-through.
        # Required because spatial softmax over 25 joints gives ~0.04/joint, combined
        # with temporal(~0.5) and channel(~0.5) → x*0.01. Without this gate that would
        # halve the signal from epoch 1. Gate fades in attention gradually as it learns.
        # Added to no_decay in trainer (name contains 'gate').
        self.gate = nn.Parameter(torch.full((1,), -4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — lerp(x, x*A_s*A_t[*A_c], scale)
        """
        B, C, T, V = x.shape

        # Temporal attention (Frame_Att style: avg + max over C and V)
        x_avg = x.mean(dim=(1, 3))                            # (B, T)
        x_max = x.amax(dim=(1, 3))                            # (B, T)
        x_t   = torch.stack([x_avg, x_max], dim=1)            # (B, 2, T)
        A_t   = torch.sigmoid(self.temporal_conv(x_t))        # (B, 1, T)
        A_t   = A_t.view(B, 1, T, 1)

        if self.use_spatial:
            # Spatial attention — scale by V so mean weight = 1.0 (not 1/V).
            x_s = x.mean(dim=(1, 2))                              # (B, V)
            A_s = torch.softmax(self.spatial_fc(x_s), dim=-1) * V # (B, V), mean=1.0
            A_s = A_s.view(B, 1, 1, V)
            attn = A_s * A_t
        else:
            attn = A_t

        if self.use_channel_se:
            # Channel attention (SE-style)
            x_c = x.mean(dim=(2, 3))                          # (B, C)
            A_c = F.relu(self.channel_fc1(x_c), inplace=True) # (B, C_r)
            A_c = torch.sigmoid(self.channel_fc2(A_c))        # (B, C)
            A_c = A_c.view(B, C, 1, 1)
            attn = attn * A_c

        # Residual gate: lerp from identity to full attention
        scale = torch.sigmoid(self.gate)                       # scalar in (0, 1)
        return x * (1.0 - scale) + x * attn * scale
