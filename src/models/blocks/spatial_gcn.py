"""
SpatialGCN — Multi-hop spatial graph convolution for LAST-E v3.

Ported from EfficientGCN with two additions:
    1. Symmetric D^{-1/2}AD^{-1/2} normalised adjacency (fixes P4 gradient audit).
    2. HD-GCN-inspired lightweight subset attention: instead of equally summing
       K-hop outputs, learns per-sample softmax weights over K subsets.

The layer uses a single 1×1 Conv2d to split channels into K groups (one per
hop distance 0..max_hop), then aggregates each group with the corresponding
adjacency subset via einsum.
"""

import torch
import torch.nn as nn


class SpatialGCN(nn.Module):
    """Multi-hop spatial graph convolution with learnable edge importance.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        A:            Pre-computed adjacency subsets, shape (K, V, V).
        max_hop:      Maximum graph distance (K = max_hop + 1).
        use_subset_att: Enable HD-GCN-style subset attention (default True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        max_hop: int = 2,
        use_subset_att: bool = True,
    ):
        super().__init__()
        K = max_hop + 1
        self.K = K
        self.out_channels = out_channels

        # 1×1 conv: in → out*K channels, then reshaped into K groups
        self.gcn = nn.Conv2d(in_channels, out_channels * K, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Adjacency subsets — non-learnable, set by Graph
        # A may have more subsets than K (spatial strategy → 3 for 1-hop);
        # we take the first K.
        self.register_buffer('A', A[:K].clone())  # (K, V, V)

        # Learnable edge importance multiplier (per-entry)
        self.edge = nn.Parameter(torch.ones_like(self.A))

        # HD-GCN inspired: lightweight attention over K subsets
        self.use_subset_att = use_subset_att
        if use_subset_att:
            self.subset_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                          # (B, C*K, 1, 1)
                nn.Flatten(),                                      # (B, C*K)
                nn.Linear(out_channels * K, K, bias=False),        # (B, K)
                nn.Softmax(dim=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T, V)
        Returns:
            out: (B, C_out, T, V)
        """
        x = self.gcn(x)                         # (B, C*K, T, V)
        B, CK, T, V = x.shape
        C = self.out_channels

        A_eff = self.A * self.edge               # (K, V, V)

        if self.use_subset_att:
            # Per-sample attention weights over subsets
            att = self.subset_att(x)             # (B, K)
            x = x.view(B, self.K, C, T, V)
            x = torch.einsum('bkctv,kvw->bkctw', x, A_eff)
            x = (x * att[:, :, None, None, None]).sum(dim=1)  # (B, C, T, V)
        else:
            x = x.view(B, self.K, C, T, V)
            x = torch.einsum('bkctv,kvw->bctw', x, A_eff)    # (B, C, T, V)

        return self.bn(x)
