"""
SGPShift — Zero-parameter Semantic Graph-Part Shift.

Extends Shift-GCN's channel-partition shift to semantically-typed edges
from the SGP (Semantic Body-Part) graph.

Shift-GCN uses hop-distance neighborhoods (1-hop, 2-hop) — structurally
correct but semantically agnostic.  SGPShift uses:
  - A_intra edges: within the same anatomical body-part (fine motor control)
  - A_inter edges: crossing part boundaries (joint coordination)
  - Identity:      anchor channels (no shift)

Channel split (C//3 per group):
  Group 0 (channels 0 : C//3)      → shift along A_intra nearest neighbor
  Group 1 (channels C//3 : 2*C//3) → shift along A_inter nearest neighbor
  Group 2 (channels 2*C//3 : C)    → identity (no shift)

Forward: single torch.gather call — 0 learnable parameters, 0 extra FLOPs
beyond the index lookup.  Shift indices are precomputed at __init__ and
stored as a non-gradient buffer.

Usage:
    graph = Graph('ntu-rgb+d', 'semantic_bodypart', max_hop=2, raw_partitions=True)
    A_intra = torch.tensor(graph.A[0], dtype=torch.float32)
    A_inter = torch.tensor(graph.A[1], dtype=torch.float32)
    sgp_shift = SGPShift(channels=40, A_intra=A_intra, A_inter=A_inter)
    out = sgp_shift(x)   # (B, 40, T, 25) → (B, 40, T, 25)
"""

import torch
import torch.nn as nn


class SGPShift(nn.Module):
    """Zero-parameter graph shift using SGP semantic typed edges.

    Args:
        channels:   Number of input/output channels.
        A_intra:    (V, V) intra-part adjacency (float, from SGP graph A[0]).
        A_inter:    (V, V) inter-part adjacency (float, from SGP graph A[1]).
        num_joints: Number of skeleton joints (default 25 for NTU RGB+D).
    """

    def __init__(
        self,
        channels: int,
        A_intra: torch.Tensor,
        A_inter: torch.Tensor,
        num_joints: int = 25,
    ):
        super().__init__()
        self.channels   = channels
        self.num_joints = num_joints

        g0 = channels // 3          # intra-part group
        g1 = channels // 3          # inter-part group
        # g2 = channels - g0 - g1   # identity group

        # Build per-joint, per-channel target-joint indices for each group.
        # Neighbor cycling: each channel c shifts to a different ranked
        # neighbor (c % num_neighbors) for increased spatial diversity.
        intra_idx = self._build_shift_idx_cycling(A_intra, num_joints, g0)  # (g0, V)
        inter_idx = self._build_shift_idx_cycling(A_inter, num_joints, g1)  # (g1, V)
        identity  = torch.arange(num_joints).unsqueeze(0).expand(channels - g0 - g1, -1)

        # Assemble full (C, V) index tensor:
        #   rows 0..g0-1       → intra_idx (cycling neighbors)
        #   rows g0..g0+g1-1   → inter_idx (cycling neighbors)
        #   rows g0+g1..C-1    → identity
        idx = torch.cat([intra_idx, inter_idx, identity], dim=0)  # (C, V)

        self.register_buffer('shift_indices', idx)  # LongTensor (C, V)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_shift_idx_cycling(A: torch.Tensor, num_joints: int, num_channels: int) -> torch.Tensor:
        """Per-channel neighbor cycling: channel c shifts to the (c % K)-th ranked neighbor.

        For each joint v, all neighbors with positive edge weight are ranked
        by descending weight. Channel c in this group shifts joint v to
        neighbor ranked (c % num_neighbors). Joints with no neighbors fall
        back to identity (v → v).

        This provides more spatial diversity than the original single-neighbor
        broadcast at zero parameter cost.

        Args:
            A:            (V, V) adjacency matrix.
            num_joints:   V.
            num_channels: Number of channels in this group.

        Returns:
            idx: LongTensor (num_channels, V) — per-channel, per-joint target.
        """
        # Start with identity for all channels
        idx = torch.arange(num_joints).unsqueeze(0).expand(num_channels, -1).clone()  # (num_channels, V)
        A_f = A.float() if not A.is_floating_point() else A.clone()
        for v in range(num_joints):
            row = A_f[v].clone()
            row[v] = -1.0  # exclude self-loop
            # Find all positive neighbors, sorted by descending weight
            positive_mask = row > 0
            if not positive_mask.any():
                continue  # all channels keep identity for isolated joints
            # Get sorted neighbor indices (descending by weight)
            weights = row[positive_mask]
            joint_indices = torch.where(positive_mask)[0]
            sorted_order = weights.argsort(descending=True)
            ranked_neighbors = joint_indices[sorted_order]  # best → worst
            n_nbrs = len(ranked_neighbors)
            for c in range(num_channels):
                idx[c, v] = ranked_neighbors[c % n_nbrs]
        return idx  # (num_channels, V)

    @staticmethod
    def _build_shift_idx(A: torch.Tensor, num_joints: int) -> torch.Tensor:
        """For each joint v, select the neighbor with the highest edge weight.

        Falls back to identity (v → v) for joints with no outgoing edges
        in this adjacency (e.g., root joint in A_inter may be isolated).

        Args:
            A:          (V, V) adjacency matrix (row = source, col = target).
            num_joints: V.

        Returns:
            idx: LongTensor (V,) — target joint index per source joint.
        """
        idx = torch.arange(num_joints)
        A_f = A.float() if not A.is_floating_point() else A.clone()
        for v in range(num_joints):
            row = A_f[v].clone()
            row[v] = -1.0    # exclude self-loop
            best = row.argmax().item()
            if row[best] > 0:
                idx[v] = best
            # else: fallback to identity (idx[v] = v, already set)
        return idx  # (V,)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute joints per channel group via precomputed shift indices.

        Args:
            x: (B, C, T, V)

        Returns:
            out: (B, C, T, V) — same shape, joint dimension permuted per group.
        """
        B, C, T, V = x.shape
        # Expand (C, V) → (B, C, T, V) for torch.gather along dim=3 (V)
        idx = (
            self.shift_indices           # (C, V)
            .unsqueeze(0)                # (1, C, V)
            .unsqueeze(2)                # (1, C, 1, V)
            .expand(B, C, T, V)
        )
        return torch.gather(x, 3, idx)
