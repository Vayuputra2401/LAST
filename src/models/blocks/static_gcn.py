"""
StaticGCN — Lightweight static graph convolution with per-stage shared weights
and a trainable adjacency correction (A_learned).

Design
------
One StaticGCN instance is created **per stage** and shared across all blocks
in that stage.  This means blocks that duplicate channel width (the 2nd, 3rd …
block inside a stage, where in_ch == out_ch == stage_ch) pay zero extra params
for graph weights — the same W matrix is reused.

Forward
-------
  x_agg = Σ_k  A_k @ x            (static K-subset aggregation, no params)
         + A_learned_norm @ x      (trainable topology correction, V² params)
  out   = x + BN(Conv1×1(x_agg))  (projection + residual)

Param count per stage (channels C, joints V=25, K=3 static subsets):
  Conv2d(C, C, 1, bias=False)  →  C²      params
  BN2d(C)                      →  2C      params
  A_learned (V×V)              →  625     params  (zero-init, no weight decay)
  ─────────────────────────────────────────────────
  Total per stage              →  C² + 2C + 625

Trainer note: A_learned matches keyword 'A_learned' in no_decay list —
it is automatically excluded from weight decay by the trainer.
"""

import torch
import torch.nn as nn


class StaticGCN(nn.Module):
    """
    Lightweight static GCN with learnable adjacency correction.

    Intended to be created once per stage and shared (by reference) across
    every block within that stage, so graph weights are not duplicated.

    Args:
        channels:   C — number of input/output feature channels.
        A:          (K, V, V) float tensor — pre-normalised static adjacency
                    subsets (D^{-1/2}AD^{-1/2} from normalize_symdigraph_full).
        num_joints: V — number of skeleton joints (default 25).
    """

    def __init__(self, channels: int, A: torch.Tensor, num_joints: int = 25):
        super().__init__()

        # ── Static adjacency ────────────────────────────────────────────
        # (K, V, V) buffer — not a parameter, same as LAST-E convention.
        self.register_buffer('A', A)

        # ── Trainable topology correction ────────────────────────────────
        # Zero-init → A_learned_norm = 0 at start → model starts as pure
        # static GCN, learns residual corrections during training.
        # Excluded from weight decay via 'A_learned' keyword in trainer.
        self.A_learned = nn.Parameter(torch.zeros(num_joints, num_joints))

        # ── Spatial projection ───────────────────────────────────────────
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)

    # -------------------------------------------------------------------

    def _norm_A_learned(self) -> torch.Tensor:
        """
        Return symmetric D^{-1/2} |A_learned| D^{-1/2}.

        abs  → non-negative edge weights
        D^{-1/2} … D^{-1/2} → symmetric normalisation (same as LAST-E v3)
        """
        A = torch.abs(self.A_learned)                                # (V, V)
        D = A.sum(dim=1).clamp(min=1e-6)                             # (V,)
        D_inv = D.pow(-0.5)                                          # (V,)
        return D_inv.unsqueeze(1) * A * D_inv.unsqueeze(0)           # (V, V)

    # -------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V)  — same shape, spatially aggregated + projected
        """
        # 1. Static aggregation over K subsets
        #    einsum contracts the V (joint) dimension:
        #    A[k,v,w] * x[b,c,t,w]  →  summed over k and w  →  (B,C,T,V)
        x_agg = torch.einsum('kvw,bctw->bctv', self.A, x)

        # 2. Learned adjacency correction
        A_l   = self._norm_A_learned()                               # (V, V)
        x_agg = x_agg + torch.einsum('vw,bctw->bctv', A_l, x)

        # 3. Project + BN + residual
        return x + self.bn(self.conv(x_agg))
