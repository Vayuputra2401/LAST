"""
BodyRegionShift — Body-Region-Aware Spatial Shift (Idea F / BRASP).

Anatomically-partitioned channel shift for skeleton action recognition.
Channels are divided into four groups; each group's channels are shifted
only within the joints that belong to that anatomical region.

Cost: 0 learnable parameters, 0 FLOPs beyond a single torch.gather call.
The shift indices are precomputed at __init__ and stored as a buffer.

Reference: Experiment-LAST-Lite.md — Sections 2, 3.
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# NTU RGB+D 25-joint body region definitions (0-based indices)
# ---------------------------------------------------------------------------
BODY_REGIONS = {
    'left_arm':  [4, 5, 6, 7, 21, 22],     # shoulder → elbow → wrist → hand tip, thumb
    'right_arm': [8, 9, 10, 11, 23, 24],    # shoulder → elbow → wrist → hand tip, thumb
    'left_leg':  [12, 13, 14, 15],           # hip → knee → ankle → foot
    'right_leg': [16, 17, 18, 19],           # hip → knee → ankle → foot
    'torso':     [0, 1, 2, 3, 20],           # spine base → spine mid → spine shoulder → neck → head
}


def get_channel_groups(C: int) -> dict:
    """
    Return channel slice for each anatomical group.

    Allocation (proportional to group importance for action recognition):
      Arms   (left + right): C//4   = 25%  — fine manipulation
      Legs   (left + right): C//4   = 25%  — locomotion
      Torso:                 C//8   = 12.5% — posture
      Cross-body:            rest   = 37.5% — inter-limb coordination
    """
    arm_end   = C // 4          # 0  : arm_end   → arms
    leg_end   = C // 2          # arm_end : leg_end → legs
    torso_end = 5 * C // 8      # leg_end : torso_end → torso
    # torso_end : C → cross-body (37.5%)
    return {
        'arm':   slice(0, arm_end),
        'leg':   slice(arm_end, leg_end),
        'torso': slice(leg_end, torso_end),
        'cross': slice(torso_end, C),
    }


def compute_shift_indices(A: torch.Tensor, C: int) -> torch.Tensor:
    """
    Precompute per-channel joint-permutation indices for BodyRegionShift.

    For each channel c and joint v, shift_indices[c, v] holds the source
    joint that channel c at position v should read from.

    Arms/legs/torso channels: permuted within the joints of their region.
    Cross-body channels:      permuted via graph-neighbour cycling.

    Args:
        A:  (V, V) float adjacency matrix.  A[v, w] > 0 ↔ v,w are connected.
        C:  Number of channels.

    Returns:
        shift_indices: LongTensor (C, V)
    """
    V = A.shape[0]
    shift_indices = torch.zeros(C, V, dtype=torch.long)
    channel_groups = get_channel_groups(C)

    # Precompute graph neighbours for cross-body group
    A_np = A.numpy() if isinstance(A, torch.Tensor) else np.array(A)
    neighbors = [list(np.where(A_np[v] > 0)[0]) for v in range(V)]

    for group_name, ch_slice in channel_groups.items():
        ch_list = list(range(ch_slice.start, ch_slice.stop))

        if group_name == 'cross':
            # Shift across all joints: each channel cycles through graph neighbours
            for i, c in enumerate(ch_list):
                for v in range(V):
                    nbrs = neighbors[v]
                    if len(nbrs) == 0:
                        shift_indices[c, v] = v         # isolated joint → identity
                    else:
                        shift_indices[c, v] = nbrs[i % len(nbrs)]

        else:
            # Build the joint set for this group (arms = left + right combined)
            if group_name == 'arm':
                region_joints = BODY_REGIONS['left_arm'] + BODY_REGIONS['right_arm']
            elif group_name == 'leg':
                region_joints = BODY_REGIONS['left_leg'] + BODY_REGIONS['right_leg']
            else:  # torso
                region_joints = BODY_REGIONS['torso']

            region_set = set(region_joints)

            for i, c in enumerate(ch_list):
                for v in range(V):
                    if v in region_set:
                        # Shift within the region: cycle by offset i+1
                        idx = region_joints.index(v)
                        target = region_joints[(idx + i + 1) % len(region_joints)]
                        shift_indices[c, v] = target
                    else:
                        # Joint not in this region → identity (read from self)
                        shift_indices[c, v] = v

    return shift_indices  # (C, V)


# ---------------------------------------------------------------------------
# BodyRegionShift module
# ---------------------------------------------------------------------------
class BodyRegionShift(nn.Module):
    """
    Zero-parameter spatial channel shift structured by body anatomy.

    Splits channels into four groups (arm, leg, torso, cross-body) and
    permutes each group's channels only within the joints of that region.
    This injects structural skeleton bias at zero cost.

    Args:
        channels:     Number of input/output channels (must equal C).
        A:            (V, V) adjacency tensor used for cross-body neighbour lookup.
    """

    def __init__(self, channels: int, A: torch.Tensor):
        super().__init__()
        shift_indices = compute_shift_indices(A, channels)
        self.register_buffer('shift_indices', shift_indices)  # (C, V), no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — same shape, channels spatially permuted by region
        """
        B, C, T, V = x.shape
        # Expand (C, V) → (B, C, T, V) for torch.gather
        idx = self.shift_indices.unsqueeze(0).unsqueeze(2).expand(B, C, T, V)
        return torch.gather(x, 3, idx)
