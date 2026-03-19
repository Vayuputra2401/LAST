"""
CrossStreamFusion — Gated cross-stream feature exchange for late-fusion models.

Each stream's final backbone features receive a gated bottleneck contribution
from the other stream. Applied after backbone stages, before pooling+classifier.

Gate init -4.0 → sigmoid(-4) ≈ 0.018, near-zero at epoch 0, grows gradually.
Bottleneck (C → C//4 → C) keeps added params small (~2 × 2 × C × C//4).

Example C=192: 2 × (192×48 + 48×192) × 2 = ~37K params.
"""

import torch
import torch.nn as nn


class CrossStreamFusion(nn.Module):
    """Gated cross-stream feature exchange between two backbone feature maps.

    Each stream gets a residual contribution from the other stream via a
    bottleneck projection. Both streams use the original (pre-update) features
    for projection — symmetric, no sequential dependency.

    Args:
        channels: Feature channels C (must match both backbone output channels).
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(8, channels // 4)

        def _proj(c_in: int, c_mid: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in,  c_mid, 1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.Hardswish(inplace=True),
                nn.Conv2d(c_mid, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )

        self.proj_b_to_a = _proj(channels, mid, channels)
        self.proj_a_to_b = _proj(channels, mid, channels)

        # Near-zero gate at init — cross-stream info fades in gradually.
        # Caught by '.gate' in trainer no_decay.
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x_a: (B, C, T, V) — stream-A backbone features.
            x_b: (B, C, T, V) — stream-B backbone features.
        Returns:
            Tuple (x_a_fused, x_b_fused) — each enriched with cross-stream context.
        """
        g = torch.sigmoid(self.gate)
        # Project original features — symmetric exchange, no sequential bias
        return (
            x_a + g * self.proj_b_to_a(x_b),
            x_b + g * self.proj_a_to_b(x_a),
        )
