"""
DropPath — Stochastic Depth regularization.

Randomly drops entire residual branches during training (per-sample).
At inference, all branches are active and the output is not scaled.

Applied to the *main path* in ShiftFuseBlock before adding the skip
connection, so the residual (identity/downsampled input) is always
preserved:

    out = self.drop_path(out)   # stochastic depth on main branch
    out = res + out             # residual always passes through

Rate is set linearly from 0.0 at the first block to `drop_path_rate`
at the last block (computed in LAST_Lite.__init__).

Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: drops the entire residual branch per sample.

    Args:
        drop_prob: Probability of dropping a sample's contribution.
                   0.0 = identity (no drop). Default: 0.0.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Binary mask: shape (B, 1, 1, 1) — per-sample, broadcast over C,T,V
        mask = torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device)
        mask = torch.floor(mask + keep_prob)      # Bernoulli(keep_prob)
        return x * mask / keep_prob               # scale to preserve expectation

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"
