import torch
import torch.nn as nn
import torch.nn.functional as F


class ST_JointAtt(nn.Module):
    """
    Spatial-Temporal Joint Attention Module — fixed sigmoid saturation.

    Factorized attention that refines features based on:
      1. Frame importance  (Temporal branch)
      2. Joint importance  (Spatial branch)

    Problem with original design:
    Both branches used raw sigmoid gates: out = x * att_t * att_s.
    In early training, sigmoid outputs are initialized near 0.5 on average,
    so the product att_t * att_s ≈ 0.25, attenuating all features by 75%.
    As training progresses, gates can saturate to 0, killing gradients through
    that path entirely — explaining the plateau at 3-5% reported experimentally.

    Fix — learnable residual gating (SE-style):
    Instead of multiplicative-only gating, the attention output is blended with
    the identity via a learnable scalar α per channel, initialized to 0:

        out = x + α * (x * att_t * att_s - x)
            = x * (1 + α * (att_t * att_s - 1))

    At init: α=0 → pure identity, gradients flow freely.
    During training: α grows to blend in attention, preventing dead-gate collapse.
    This is equivalent to the zero-initialization trick used in ResNet-v2 and
    the SE-Net skip connection design.

    Args:
        channel:   Input channels
        reduction: Bottleneck reduction ratio for MLP (default: 4)
    """

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.channel = channel
        reduced_channel = max(4, channel // reduction)

        # Temporal Attention: pool joints → conv bottleneck → sigmoid gate
        # Output shape: (N, C, T, 1) — channel-wise frame importance
        self.temporal_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),                       # (N, C, T, 1)
            nn.Conv2d(channel, reduced_channel, 1, bias=False),
            nn.BatchNorm2d(reduced_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention: pool frames → conv bottleneck → sigmoid gate
        # Output shape: (N, C, 1, V) — channel-wise joint importance
        self.spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),                       # (N, C, 1, V)
            nn.Conv2d(channel, reduced_channel, 1, bias=False),
            nn.BatchNorm2d(reduced_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # Learnable gate scalar α per channel, initialized to 0.
        # At init the block is a perfect identity (no attention applied).
        # Gradients flow through x directly, attention branches learn gradually.
        # Shape (1, C, 1, 1) → broadcasts over (N, C, T, V).
        self.alpha = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            out: (N, C, T, V) — attention-refined, with residual gate
        """
        att_t = self.temporal_att(x)   # (N, C, T, 1)
        att_s = self.spatial_att(x)    # (N, C, 1, V)

        # Combined attention-modulated features
        x_att = x * att_t * att_s     # (N, C, T, V)

        # Residual blend: α=0 at init → identity; grows to add attention signal
        out = x + self.alpha * (x_att - x)

        return out
