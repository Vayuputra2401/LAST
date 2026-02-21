import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_joint_att import ST_JointAtt


# ---------------------------------------------------------------------------
# LightGCNConv (kept for backward compatibility)
# ---------------------------------------------------------------------------

class LightGCNConv(nn.Module):
    """
    Original lightweight GCN: sums K subsets into one undirected A_norm.
    Kept for reference. LightGCNBlock now uses DirectionalGCNConv instead.
    """

    def __init__(self, in_channels: int, out_channels: int, A_physical: torch.Tensor):
        super().__init__()
        A_sum = A_physical.sum(0) if A_physical.dim() == 3 else A_physical.clone()
        row_sum = A_sum.sum(dim=1).clamp(min=1e-6)
        D_inv_sqrt = row_sum.pow(-0.5)
        A_norm = D_inv_sqrt.unsqueeze(1) * A_sum * D_inv_sqrt.unsqueeze(0)
        self.register_buffer('A_norm', A_norm)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x_flat = x.view(N, C * T, V)
        x_agg = torch.matmul(x_flat, self.A_norm).view(N, C, T, V)
        return self.conv(x_agg)


# ---------------------------------------------------------------------------
# DirectionalGCNConv
# ---------------------------------------------------------------------------

class DirectionalGCNConv(nn.Module):
    """
    Directional GCN convolution preserving K physical adjacency subsets.

    Why this improves on LightGCNConv:
    LightGCNConv sums the K=3 spatial subsets into one undirected graph, losing
    the directional distinction between:
      A[0]: self-connections
      A[1]: centripetal edges  (joint → parent, toward body center)
      A[2]: centrifugal edges  (joint → child, away from center)

    This module keeps K normalized adjacency buffers and learns per-channel
    importance weights for each subset (alpha), so channels that capture
    centripetal signals can suppress centrifugal aggregation and vice versa.

    Extra cost vs LightGCNConv: K × C_in params per block (e.g. 3×40=120 for
    base Stage 1). The pointwise conv cost remains identical (1 × C_in × C_out).

    Pitfalls handled:
    - K separate register_buffer calls (not a list attribute) — GPU move safe
    - alpha initialized to zeros → softmax gives uniform 1/K start (neutral)
    - F.softmax over dim=0 (across K, per channel) — correct axis
    """

    def __init__(self, in_channels: int, out_channels: int, A_physical: torch.Tensor):
        super().__init__()

        K = A_physical.shape[0] if A_physical.dim() == 3 else 1
        self.K = K
        self.in_channels = in_channels

        # Register K separate D^{-1/2} A_k D^{-1/2} buffers — one per subset.
        # Named A_0, A_1, A_2 so they appear correctly in state_dict and move
        # to GPU with .to(device) automatically.
        for k in range(K):
            A_k = A_physical[k] if A_physical.dim() == 3 else A_physical
            row_sum = A_k.sum(dim=1).clamp(min=1e-6)
            D_inv_sqrt = row_sum.pow(-0.5)
            A_k_norm = D_inv_sqrt.unsqueeze(1) * A_k * D_inv_sqrt.unsqueeze(0)
            self.register_buffer(f'A_{k}', A_k_norm)

        # Per-channel per-subset attention weights.
        # Shape (K, C_in): softmax over K for each of the C_in channels.
        # Zeros → softmax = [1/K, …, 1/K] — equal weighting at init.
        # Gradient flows freely; weights diverge during training as channels
        # specialise (e.g. channel 7 learns centripetal emphasis).
        self.alpha = nn.Parameter(torch.zeros(K, in_channels))

        # Single shared 1×1 pointwise conv — identical cost to LightGCNConv.
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x_flat = x.reshape(N, C * T, V)   # (N, C*T, V) — reshape handles non-contiguous input

        # Per-channel softmax across K subsets — shape (K, C_in).
        alpha_w = F.softmax(self.alpha, dim=0)

        x_agg = None
        for k in range(self.K):
            A_k = getattr(self, f'A_{k}')                      # (V, V)
            x_k = torch.matmul(x_flat, A_k).view(N, C, T, V)  # (N, C, T, V)
            # Broadcast weight (C_in,) over (N, C, T, V)
            x_k = x_k * alpha_w[k].view(1, C, 1, 1)
            x_agg = x_k if x_agg is None else x_agg + x_k

        return self.conv(x_agg)   # (N, out_channels, T, V)


# ---------------------------------------------------------------------------
# MultiScaleTCN
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module):
    """
    Two-branch parallel depthwise-separable TCN.

    Branch 1  (channels C//2):  kernel 9×1, dilation=1  →  9-frame field
    Branch 2  (channels C//2):  kernel 9×1, dilation=2  → 17-frame field

    Channels are split with x.chunk(2, dim=1) — no extra projection needed
    since both branches output C//2 and concat gives C exactly.

    Param cost vs single 9×1 DSepTCN:
      Single:   C×9 + C²   (depthwise + pointwise)
      2-branch: C×9 + C²/2  (depthwise same, pointwise halved)
    Savings: C²/2 per block. For C=160 and 4 blocks: 51,200 params.

    PITFALL — dilation padding:
      Branch 1: pad = (9-1)//2              = 4
      Branch 2: pad = 2 × (9-1)//2         = 8   ← not 4!
      Formula:  pad = dilation × (k-1) // 2
    Both produce identical T_out for any stride s:
      T_out = floor((T + 2×pad - dilation×(k-1) - 1) / s + 1) = T // s
    → concat is always safe.

    PITFALL — stride placement:
      Stride goes on the depthwise conv, pointwise always has stride=1.
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        assert channels % 2 == 0, f"channels must be even, got {channels}"
        half = channels // 2

        pad1 = (9 - 1) // 2            # = 4, dilation=1
        pad2 = 2 * (9 - 1) // 2        # = 8, dilation=2

        # Branch 1: standard 9×1 depthwise-separable
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            # Depthwise: stride on this conv
            nn.Conv2d(half, half, (9, 1), stride=(stride, 1),
                      padding=(pad1, 0), groups=half, bias=False),
            # Pointwise: always stride 1
            nn.Conv2d(half, half, 1, bias=False),
            nn.BatchNorm2d(half),
        )

        # Branch 2: dilated 9×1 depthwise-separable (dilation=2)
        self.branch2 = nn.Sequential(
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            # Dilated depthwise: dilation=(2,1) for temporal dilation only
            nn.Conv2d(half, half, (9, 1), stride=(stride, 1),
                      padding=(pad2, 0), dilation=(2, 1), groups=half, bias=False),
            # Pointwise: always stride 1
            nn.Conv2d(half, half, 1, bias=False),
            nn.BatchNorm2d(half),
        )

        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along channel dim — chunk handles even C cleanly
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        return self.drop(out)


# ---------------------------------------------------------------------------
# LightGCNBlock (updated: DirectionalGCNConv + MultiScaleTCN)
# ---------------------------------------------------------------------------

class LightGCNBlock(nn.Module):
    """
    Lightweight GCN Block for LAST-E student model — v2.

    Structure:
        DirectionalGCNConv → BN+ReLU → [optional ST_JointAtt]
        → MultiScaleTCN (2-branch, dilation 1+2) → Residual → ReLU

    vs LightGCNBlock v1:
    - DirectionalGCNConv restores centripetal/centrifugal distinction
      (K=3 separate adjacency buffers + per-channel K-subset weights)
    - MultiScaleTCN covers both 9-frame and 17-frame receptive fields
      while using ~50% fewer pointwise params than single-branch TCN

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        A_physical:   Physical adjacency (K, V, V) from Graph().
        stride:       Temporal stride (1 or 2).
        residual:     Whether to use residual connection.
        use_st_att:   Whether to apply ST_JointAtt after GCN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_physical: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        use_st_att: bool = False,
    ):
        super().__init__()

        self.stride = stride

        # 1. Directional graph convolution (spatial)
        self.gcn = DirectionalGCNConv(in_channels, out_channels, A_physical)

        # 2. Post-GCN normalization
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.gcn_relu = nn.ReLU(inplace=True)

        # 3. Optional ST-Joint Attention (None → zero param overhead)
        self.st_att = ST_JointAtt(out_channels, reduction=4) if use_st_att else None

        # 4. Multi-scale parallel TCN (2-branch, dilation 1+2)
        self.tcn = MultiScaleTCN(out_channels, stride=stride)

        # 5. Residual connection
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

        # Spatial: directional graph aggregation + projection
        x = self.gcn(x)
        x = self.gcn_bn(x)
        x = self.gcn_relu(x)

        # Optional spatial-temporal attention
        if self.st_att is not None:
            x = self.st_att(x)

        # Temporal: multi-scale (dilation 1 + dilation 2)
        x = self.tcn(x)

        # Residual
        if self.residual and self.residual_path is not None:
            x = x + self.residual_path(res)

        return self.relu(x)
