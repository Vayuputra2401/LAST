"""
CTR-GCN-aligned building blocks for LAST-E v2.

This module implements the core architectural components that align LAST-E
with SOTA skeleton action recognition models while preserving edge-deployment
efficiency.

Key components:
  - DropPath:          Stochastic depth regularization (linear ramp)
  - CTRLightGCNConv:   Per-group channel-topology refinement (CTR-GCN core)
  - MultiScaleTCN4:    4-branch temporal module (EfficientGCN-aligned)
  - FreqTemporalGate:  FFT-based frequency-domain attention (novel)
  - CTRGCNBlock:       Full block assembly

Reference implementations:
  CTR-GCN:        https://github.com/Uason-Chen/CTR-GCN  (ICCV 2021)
  EfficientGCN:   https://github.com/yfsong0709/EfficientGCNv1  (TPAMI 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_joint_att import ST_JointAtt


# ---------------------------------------------------------------------------
# DropPath — Stochastic Depth
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Stochastic depth (drop path) regularization.

    During training, drops the entire residual branch with probability `p`.
    During eval, acts as identity. This is the standard implementation from
    timm / DeiT, adapted for skeleton GCN blocks.

    Applied to the MAIN branch output, NOT the skip connection.
    When dropped, only the skip connection survives, creating an implicit
    ensemble of sub-networks of different depths.

    Args:
        drop_prob: Probability of dropping the path (0.0 = no drop).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, 1) — drop entire sample's branch, not individual elements
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        # Scale by 1/keep_prob to maintain expected value
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# CTRLightGCNConv — Channel-Topology Refinement GCN
# ---------------------------------------------------------------------------

class CTRLightGCNConv(nn.Module):
    """
    Lightweight Channel-Topology Refinement GCN (CTR-GCN inspired).

    Core innovation from CTR-GCN (ICCV 2021): instead of all channels sharing
    one adjacency topology, split channels into G groups and let each group
    learn its own topology refinement via input-dependent Q/K projections.

    This fixes the P1 gradient-flow bottleneck identified in the LAST-E audit:
    the original DirectionalGCNConv sums 5 adjacency components through a
    single 1x1 conv, giving each component only ~C/5 effective channels.
    CTRLightGCNConv gives each group its own refined topology, then concatenates.

    Design choices for efficiency (vs full CTR-GCN):
      - G=4 groups (CTR-GCN uses 8, but we have smaller channels)
      - Shared Q/K projection across groups (reduces params by G×)
      - Physical topology: softmax-weighted blend of K=3 subsets per group
      - Refinement: low-rank Q^T K with sigmoid-gated addition

    Args:
        in_channels:   Input feature channels.
        out_channels:  Output feature channels.
        A_physical:    Pre-computed adjacency (K, V, V) from Graph().
        num_groups:    Number of channel groups for topology refinement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_physical: torch.Tensor,
        num_groups: int = 4,
    ):
        super().__init__()

        K = A_physical.shape[0] if A_physical.dim() == 3 else 1
        V = A_physical.shape[-1]
        self.K = K
        self.V = V
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # ── Physical adjacency subsets (K=3: self, centripetal, centrifugal) ──
        for k in range(K):
            A_k = A_physical[k] if A_physical.dim() == 3 else A_physical
            self.register_buffer(f'A_{k}', A_k.clone())

        # Per-group per-subset blend weights: (G, K)
        # Zeros → softmax = uniform 1/K at init; each group diverges during training.
        self.alpha = nn.Parameter(torch.zeros(num_groups, K))

        # ── Input-dependent topology refinement (CTR-GCN core) ──────────────
        # Shared Q/K projection: project temporal-mean features to embed space.
        # embed_dim = max(C_in // 4, 8) — same as original DirectionalGCNConv
        # but now used for per-group refinement instead of a global dynamic adj.
        self.embed_dim = max(in_channels // 4, 8)
        self.proj_q = nn.Conv2d(in_channels, self.embed_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(in_channels, self.embed_dim, 1, bias=False)

        # Per-group gate for refinement contribution.
        # Init at 0 → sigmoid(0) = 0.5, but multiplied by the near-zero
        # random init of Q/K projections, so effective contribution starts small.
        # Unlike alpha_dyn's -4 init, this doesn't suppress learning.
        self.refine_gate = nn.Parameter(torch.zeros(num_groups))

        # ── Output projection: per-group aggregation → concat → 1x1 conv ────
        # This avoids the single-conv bottleneck: each group has its own
        # topology, and the output conv mixes all groups' outputs.
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        G = self.num_groups
        Cg = C // G  # channels per group

        x_flat = x.reshape(N, C * T, V)  # (N, C*T, V)

        # ── 1. Physical topology blend per group ──────────────────────────────
        # alpha: (G, K) → softmax over K dimension
        alpha_w = F.softmax(self.alpha, dim=1)  # (G, K)

        # Build per-group blended adjacency: A_blend[g] = sum_k alpha[g,k] * A_k
        A_stack = torch.stack([getattr(self, f'A_{k}') for k in range(self.K)])  # (K, V, V)
        # (G, K) @ (K, V*V) → (G, V*V) → (G, V, V)
        A_blend = torch.matmul(alpha_w, A_stack.reshape(self.K, -1)).reshape(G, V, V)

        # ── 2. Input-dependent refinement (CTR-GCN Q/K) ──────────────────────
        # Temporal mean pool → project to embed space
        x_mean = x.mean(dim=2, keepdim=True)  # (N, C, 1, V)
        Q = self.proj_q(x_mean).squeeze(2)     # (N, embed_dim, V)
        K_feat = self.proj_k(x_mean).squeeze(2)         # (N, embed_dim, V)

        # Compute refinement matrix: softmax(Q^T K / sqrt(d))
        scale = self.embed_dim ** 0.5
        # (N, V, embed_dim) @ (N, embed_dim, V) = (N, V, V)
        M = torch.bmm(Q.transpose(1, 2), K_feat) / scale
        M = F.softmax(M, dim=-1)  # (N, V, V) — row-normalized

        # ── 3. Per-group aggregation with refined topology ────────────────────
        gate = torch.sigmoid(self.refine_gate)  # (G,)
        x_grouped = x_flat.reshape(N, G, Cg * T, V)  # (N, G, Cg*T, V)

        out_groups = []
        for g in range(G):
            x_g = x_grouped[:, g]  # (N, Cg*T, V)
            A_g = A_blend[g]       # (V, V) — physical blend for this group

            # Graph aggregation with physical topology
            x_agg = torch.matmul(x_g, A_g)  # (N, Cg*T, V)

            # Add input-dependent refinement (gated)
            x_refine = torch.bmm(x_g, M)  # (N, Cg*T, V)
            x_agg = x_agg + gate[g] * x_refine

            out_groups.append(x_agg)

        # Concat groups → reshape back to (N, C, T, V)
        x_out = torch.stack(out_groups, dim=1)  # (N, G, Cg*T, V)
        x_out = x_out.reshape(N, C * T, V).reshape(N, C, T, V)

        # Output projection
        return self.conv_out(x_out)  # (N, out_channels, T, V)


# ---------------------------------------------------------------------------
# MultiScaleTCN4 — 4-Branch Temporal Convolution
# ---------------------------------------------------------------------------

class MultiScaleTCN4(nn.Module):
    """
    4-branch parallel temporal convolution module (EfficientGCN-aligned).

    Branch 1: DW-Sep Conv k=9, dilation=1 → 9-frame receptive field
    Branch 2: DW-Sep Conv k=9, dilation=2 → 17-frame receptive field
    Branch 3: MaxPool k=3 + 1x1 conv    → temporal peak detection
    Branch 4: Conv 1x1                   → identity-like pathway

    Channels are split 4-way with C//4 per branch. All branches produce
    identical T_out for any stride, so concat is always safe.

    vs MultiScaleTCN (2-branch):
      - 2 additional branches (maxpool + 1x1) add temporal diversity
      - C//4 per branch vs C//2 → quadratic pointwise savings (~40%)
      - MaxPool branch captures temporal extremes (critical for fast actions)

    Args:
        channels: Total channel count (must be divisible by 4).
        stride:   Temporal stride (1 or 2).
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        assert channels % 4 == 0, f"channels must be divisible by 4, got {channels}"
        quarter = channels // 4

        pad1 = (9 - 1) // 2            # = 4, dilation=1
        pad2 = 2 * (9 - 1) // 2        # = 8, dilation=2

        # Branch 1: standard 9×1 depthwise-separable
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(quarter, quarter, (9, 1), stride=(stride, 1),
                      padding=(pad1, 0), groups=quarter, bias=False),
            nn.Conv2d(quarter, quarter, 1, bias=False),
            nn.BatchNorm2d(quarter),
        )

        # Branch 2: dilated 9×1 depthwise-separable (dilation=2)
        self.branch2 = nn.Sequential(
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.Conv2d(quarter, quarter, (9, 1), stride=(stride, 1),
                      padding=(pad2, 0), dilation=(2, 1), groups=quarter, bias=False),
            nn.Conv2d(quarter, quarter, 1, bias=False),
            nn.BatchNorm2d(quarter),
        )

        # Branch 3: MaxPool k=3 + 1x1 conv (temporal peak detection)
        # MaxPool captures extreme activations that avg-pool smooths away.
        # Crucial for fast/explosive actions (kick, punch, throw).
        maxpool_pad = (3 - 1) // 2  # = 1
        self.branch3 = nn.Sequential(
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1),
                         padding=(maxpool_pad, 0)),
            nn.Conv2d(quarter, quarter, 1, bias=False),
            nn.BatchNorm2d(quarter),
        )

        # Branch 4: 1x1 conv (identity-like, preserves high-freq detail)
        if stride == 1:
            self.branch4 = nn.Sequential(
                nn.BatchNorm2d(quarter),
                nn.ReLU(inplace=True),
                nn.Conv2d(quarter, quarter, 1, bias=False),
                nn.BatchNorm2d(quarter),
            )
        else:
            # When stride > 1, need explicit temporal downsampling
            self.branch4 = nn.Sequential(
                nn.BatchNorm2d(quarter),
                nn.ReLU(inplace=True),
                nn.Conv2d(quarter, quarter, 1, bias=False),
                nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1)),
                nn.BatchNorm2d(quarter),
            )

        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along channel dim into 4 equal parts
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        out = torch.cat([
            self.branch1(x1),
            self.branch2(x2),
            self.branch3(x3),
            self.branch4(x4),
        ], dim=1)
        return self.drop(out)


# ---------------------------------------------------------------------------
# FreqTemporalGate — Frequency-Domain Attention (Novel)
# ---------------------------------------------------------------------------

class FreqTemporalGate(nn.Module):
    """
    Frequency-Aware Temporal Gating — our novel contribution.

    Uses FFT along the temporal dimension to compute spectral energy per
    frequency bin, then applies a lightweight MLP to produce per-channel
    attention weights. This allows the network to explicitly reason about
    the frequency content of each channel's temporal signal.

    Motivation: Skeleton actions have distinct frequency signatures.
    Static poses (sitting) are dominated by DC/low-freq components.
    Fast/explosive actions (punch, throw) have strong high-freq energy.
    Standard temporal convolutions can only implicitly learn these patterns
    via large kernels. FFT makes frequency information explicit.

    Design:
      1. rfft(x, dim=T) → spectral magnitudes per channel
      2. Pool over (V, freq) → per-channel spectral descriptor
      3. MLP (C → C//r → C) with sigmoid → channel gate
      4. Zero-init residual: out = x + α * (x * gate - x)

    Novel aspect: No published skeleton GCN uses frequency-domain attention.
    Most frequency-domain work in action recognition focuses on video (RGB).
    This adaptation to skeleton data is structurally different because skeleton
    time series are 1D per joint-coordinate (not 2D spatial).

    Args:
        channels:  Number of input/output channels.
        reduction: Bottleneck reduction ratio for MLP (default 4).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        reduced = max(4, channels // reduction)

        # MLP: spectral descriptor → channel gate
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

        # Zero-init residual gate (same pattern as ST_JointAtt)
        # At init: α=0 → pure identity, gate has no effect
        # During training: α grows, frequency attention is gradually introduced
        self.freq_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V)
        Returns:
            out: (N, C, T, V)
        """
        N, C, T, V = x.shape

        # 1. Compute spectral energy via FFT along temporal dim
        # rfft returns complex tensor of shape (N, C, T//2+1, V)
        x_freq = torch.fft.rfft(x, dim=2)
        # Spectral magnitude: |FFT|^2 (power spectrum)
        spectral_power = x_freq.real ** 2 + x_freq.imag ** 2  # (N, C, T//2+1, V)

        # 2. Pool over frequency bins and joints → per-channel descriptor
        # Mean over freq bins and joints → (N, C)
        spectral_desc = spectral_power.mean(dim=(2, 3))  # (N, C)

        # 3. MLP → channel gate
        gate = self.mlp(spectral_desc)  # (N, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)

        # 4. Zero-init residual blend
        x_gated = x * gate  # (N, C, T, V)
        out = x + self.freq_gate * (x_gated - x)

        return out


# ---------------------------------------------------------------------------
# CTRGCNBlock — Full Block Assembly
# ---------------------------------------------------------------------------

class CTRGCNBlock(nn.Module):
    """
    CTR-GCN-aligned block for LAST-E v2.

    Structure:
        CTRLightGCNConv → BN+ReLU → [FreqTemporalGate] → [ST_JointAtt]
        → MultiScaleTCN4 → DropPath(main) + skip → ReLU

    This replaces LightGCNBlock from LAST-E v1.

    Args:
        in_channels:    Input channels.
        out_channels:   Output channels.
        A_physical:     Physical adjacency (K, V, V) from Graph().
        stride:         Temporal stride (1 or 2).
        residual:       Whether to use residual connection.
        use_st_att:     Whether to apply ST_JointAtt after GCN.
        use_freq_gate:  Whether to apply FreqTemporalGate (novel addition).
        num_groups:     Number of groups for CTR topology refinement.
        drop_path_rate: DropPath probability for this block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_physical: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        use_st_att: bool = False,
        use_freq_gate: bool = True,
        num_groups: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.stride = stride

        # 1. CTR-GCN spatial graph convolution
        self.gcn = CTRLightGCNConv(
            in_channels, out_channels, A_physical,
            num_groups=num_groups,
        )

        # 2. Post-GCN normalization
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.gcn_relu = nn.ReLU(inplace=True)

        # 3. Optional Frequency-Domain Temporal Gate (novel)
        self.freq_gate = FreqTemporalGate(out_channels) if use_freq_gate else None

        # 4. Optional ST-Joint Attention
        self.st_att = ST_JointAtt(out_channels, reduction=4) if use_st_att else None

        # 5. 4-branch Multi-scale TCN (EfficientGCN-aligned)
        self.tcn = MultiScaleTCN4(out_channels, stride=stride)

        # 6. DropPath (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # 7. Residual connection
        self.residual = residual
        if not residual:
            self.residual_path = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual_path = nn.Identity()
        else:
            self.residual_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # Spatial: CTR-GCN per-group graph aggregation + projection
        x = self.gcn(x)
        x = self.gcn_bn(x)
        x = self.gcn_relu(x)

        # Optional frequency-domain attention (novel)
        if self.freq_gate is not None:
            x = self.freq_gate(x)

        # Optional spatial-temporal attention
        if self.st_att is not None:
            x = self.st_att(x)

        # Temporal: 4-branch multi-scale TCN
        x = self.tcn(x)

        # DropPath on main branch
        x = self.drop_path(x)

        # Residual
        if self.residual and self.residual_path is not None:
            x = x + self.residual_path(res)

        return self.relu(x)
