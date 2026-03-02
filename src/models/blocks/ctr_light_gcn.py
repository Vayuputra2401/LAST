"""
CTRLightGCN — Channel-Topology-Refinement Lightweight GCN.

Replaces StaticGCN as the stage-shared spatial aggregator in LAST-Lite.
Splits channels into G groups; each group gets its own learnable adjacency
correction on top of the shared physical graph — per-channel spatial
topology specialisation without per-sample compute.

Design
------
  For each group g ∈ {0, ..., G-1}:
    A_g     = A_physical_sum  +  A_group[g]        (V×V, learned correction)
    x_agg_g = A_g @ x_group_g                      (aggregate neighbours)
    h_g     = GroupConv_g(x_agg_g)                 (per-group projection)

  out = x  +  BN(concat([h_0, ..., h_{G-1}]))      (residual)

This gives each channel-group its own spatial receptive pattern:
  - Group 0 might specialise in centripetal edges (parent→child joints)
  - Group 1 might specialise in cross-body connections
  - etc.

Compared to StaticGCN (one shared W matrix):
  - Same total conv params (C² total, split C²/G per group × G = C²)
  - Extra: G × V² learnable adjacency params (zero-init, no weight decay)
  - At high C (e.g. C=96, G=4): C²/G ≈ 2304 vs C²=9216 per group — cheaper!

Param count per stage (channels C, joints V=25, groups G):
  Group convs:  G × (C//G)² × 1              =  C²/G   (= C² when aggregated)

  Wait — actually: G × Conv2d(C//G, C//G, 1) = G × (C//G)²  = C²/G total

  No, that's wrong. Let me restate:
  G × Conv2d(C//G, C//G, 1, bias=False)  →  G × (C//G)²  = C²/G  params
  BN2d(C)                                →  2C             params
  G × A_group[g] (V×V)                  →  G × 625        params
  ─────────────────────────────────────────────────────────────────────────
  Total = C²/G + 2C + G×625

  Example — small, G=4:
    C=48: 576 + 96 + 2500 = 3,172   (StaticGCN: 2304+96+625 = 3,025 → +147)
    C=72: 1296 + 144 + 2500 = 3,940 (StaticGCN: 5184+144+625 = 5,953 → -2013)
    C=96: 2304 + 192 + 2500 = 4,996 (StaticGCN: 9216+192+625 = 10,033 → -5037)
  → CTRLightGCN is *cheaper* than StaticGCN at large channel widths!

Trainer note: A_group matches keyword 'A_group' in no_decay list — excluded
from weight decay automatically.
"""

import torch
import torch.nn as nn


class CTRLightGCN(nn.Module):
    """Channel-Topology-Refinement Lightweight GCN.

    One instance is created **per stage** and shared (by reference) across
    every block within that stage — same weight-sharing strategy as StaticGCN.

    Args:
        channels:   C — number of input/output feature channels.
        A:          (K, V, V) float tensor — pre-normalised static adjacency
                    subsets (D^{-1/2}AD^{-1/2}).
        num_joints: V — number of skeleton joints (default 25).
        num_groups: G — number of channel groups (default 4).
                    Use G=2 for nano (param budget), G=4 for small.
    """

    def __init__(
        self,
        channels: int,
        A: torch.Tensor,
        num_joints: int = 25,
        num_groups: int = 4,
    ):
        super().__init__()
        assert channels % num_groups == 0, (
            f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        )
        self.G = num_groups
        self.C_g = channels // num_groups  # channels per group
        V = num_joints

        # ── Static adjacency (shared across all groups) ──────────────────
        # Sum K subsets → single (V, V) reference adjacency, then row-normalise
        # once at init so A_physical has row sums = 1.0.
        # Doing this at init (not per-forward) avoids the normalization Jacobian
        # ∂(A_g/row_sum)/∂A_group = (I·row_sum - A_g⊗1)/row_sum² which becomes
        # large/unstable when A_group corrections partially cancel A_physical rows.
        A_sum = A.sum(dim=0)                   # (V, V) — sum over K subsets
        row_sum = A_sum.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer('A_physical', A_sum / row_sum)   # row sums = 1.0

        # ── Per-group learned adjacency corrections ───────────────────────
        # Zero-init → starts as pure static GCN.
        # Excluded from weight decay via 'A_group' keyword in trainer.
        self.A_group = nn.ParameterList([
            nn.Parameter(torch.zeros(V, V)) for _ in range(num_groups)
        ])

        # ── Per-group spatial projection ─────────────────────────────────
        # Each group projects its own aggregated features.
        # Total conv params = G × (C//G)² = C²/G
        self.group_convs = nn.ModuleList([
            nn.Conv2d(self.C_g, self.C_g, 1, bias=False)
            for _ in range(num_groups)
        ])

        # ── BN over full channel dim ──────────────────────────────────────
        self.bn = nn.BatchNorm2d(channels)

    # -------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T, V)
        Returns:
            out: (B, C, T, V)  — spatially aggregated + projected, residual added
        """
        # Split channels into G groups: each (B, C//G, T, V)
        x_groups = x.chunk(self.G, dim=1)

        outs = []
        for g in range(self.G):
            # Per-group adjacency = physical (shared, pre-normalised) + learnable correction
            # A_physical was row-normalised at init (row sums = 1.0).
            # A_group starts at zero → initial A_g is already unit-scale.
            A_g = self.A_physical + self.A_group[g]   # (V, V)

            # Aggregate neighbours for this group
            # einsum: A_g[v,w] * x_groups[g][b,c,t,w] → (B, C//G, T, V)
            x_agg = torch.einsum('vw,bctw->bctv', A_g, x_groups[g])

            # Per-group projection
            outs.append(self.group_convs[g](x_agg))

        # Concat groups → (B, C, T, V), BN, residual
        x_agg_all = torch.cat(outs, dim=1)
        return x + self.bn(x_agg_all)
