"""
StreamBatchNorm2d — Per-stream batch normalization for late-fusion models.

In 4-stream late-fusion architectures, the shared backbone receives all
streams concatenated along the batch dimension: (4B, C, T, V).  Standard
BatchNorm2d computes a single mean/variance over all 4 streams, but each
stream (joint, velocity, bone, bone_velocity) has a fundamentally different
feature distribution — velocity values are ~10× smaller than joint positions.

StreamBatchNorm2d maintains N independent BN instances (one per stream).
During forward, the input is chunked along dim=0, each piece normalized
by its own BN with its own running statistics, then re-concatenated.

Drop-in replacement for nn.BatchNorm2d in shared backbone modules.
"""

import torch
import torch.nn as nn


class StreamBatchNorm2d(nn.Module):
    """Per-stream batch normalization.

    Args:
        num_features: Number of channels (same as nn.BatchNorm2d).
        num_streams:  Number of streams concatenated along batch dim (default 4).
        **bn_kwargs:  Extra kwargs forwarded to each nn.BatchNorm2d instance.
    """

    def __init__(self, num_features: int, num_streams: int = 4, **bn_kwargs):
        super().__init__()
        self.num_features = num_features
        self.num_streams = num_streams
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, **bn_kwargs)
            for _ in range(num_streams)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_total = x.size(0)
        if B_total % self.num_streams != 0:
            # Single-stream fallback (e.g., FLOPs estimation with odd B)
            return self.bns[0](x)
        chunks = x.chunk(self.num_streams, dim=0)
        out = [self.bns[i](c) for i, c in enumerate(chunks)]
        return torch.cat(out, dim=0)

    def extra_repr(self) -> str:
        return f"{self.num_features}, num_streams={self.num_streams}"
