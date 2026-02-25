"""
SpatialGCN — Multi-hop spatial graph convolution for LAST-E v3.

Ported from EfficientGCN with key improvements:
    1. Symmetric D^{-1/2}AD^{-1/2} normalised adjacency (via full-graph degree).
    2. HD-GCN-inspired lightweight subset attention: learns per-sample softmax
       weights over K subsets, computed from POST-aggregation features.
    3. N2 Fix: Uses ALL available subsets (K = A.shape[0]), not max_hop+1.
       For max_hop=2 with spatial strategy: K=5 (self, centrip×2, centrifug×2).

The layer uses a single 1×1 Conv2d to split channels into K groups (one per
adjacency subset), then aggregates each group with the corresponding
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
                      K is determined by graph strategy + max_hop.
        use_subset_att: Enable HD-GCN-style subset attention (default True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        use_subset_att: bool = True,
    ):
        super().__init__()
        # N2 FIX: Use ALL available subsets, not max_hop + 1
        K = A.shape[0]
        self.K = K
        self.out_channels = out_channels

        # 1×1 conv: in → out*K channels, then reshaped into K groups
        self.gcn = nn.Conv2d(in_channels, out_channels * K, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Adjacency subsets — all of them
        self.register_buffer('A', A.clone())  # (K, V, V)

        # Learnable edge importance multiplier (per-entry)
        self.edge = nn.Parameter(torch.ones_like(self.A))

        # HD-GCN inspired: lightweight attention over K subsets
        # P3 Fix: attention computed on POST-aggregation features
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
            # 1. Split into K groups and aggregate with graph
            x = x.view(B, self.K, C, T, V)
            x = torch.einsum('bkctv,kvw->bkctw', x, A_eff)  # (B, K, C, T, V)

            # 2. P3 Fix: Compute attention on POST-aggregation features
            att = self.subset_att(x.reshape(B, self.K * C, T, V))  # (B, K)

            # 3. Weighted sum over subsets
            x = (x * att[:, :, None, None, None]).sum(dim=1)  # (B, C, T, V)
        else:
            x = x.view(B, self.K, C, T, V)
            x = torch.einsum('bkctv,kvw->bctw', x, A_eff)    # (B, C, T, V)

        return self.bn(x)
