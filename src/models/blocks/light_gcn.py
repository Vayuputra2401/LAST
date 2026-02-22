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
    Directional GCN convolution with three complementary adjacency components.

    Component 1 — Physical (K=3 static subsets, fixed buffers):
      A[0]: self-connections
      A[1]: centripetal edges  (joint → parent, toward body center)
      A[2]: centrifugal edges  (joint → child, away from center)
      Per-channel softmax weights (K, C_in) let each channel specialise.

    Component 2 — Learned global adjacency A_learned (V×V, trainable):
      Global 25×25 matrix shared across ALL samples.
      Captures consistent cross-class long-range correlations (e.g. wrist↔mouth
      for all face-touching actions). Initialized to zero so training starts
      from the physical prior alone. abs() + D^{-1/2}AD^{-1/2} per-forward.
      Excluded from weight decay (already in trainer no_decay group).

    Component 3 — Dynamic adjacency A_dynamic (B×V×V, per-sample):
      Cosine similarity of temporal-mean node embeddings — different for
      every sample. Discovers which joints co-activate in this specific clip
      (e.g. right foot ↔ left arm for a specific jumping sequence).
      Gated by per-channel sigmoid scalar (alpha_dyn), initialized to -4.0
      so sigmoid(-4)≈0.018: near-zero initial contribution prevents dynamic
      term from dominating before training begins.

    FIX (E0) — Removed double normalization:
      graph.py normalize_digraph already applies D^{-1} column normalization.
      Prior version additionally applied D^{-1/2}AD^{-1/2} inside this module,
      producing O(1/d²) edge weights instead of O(1/d). End-node joints
      (fingertips, feet — critical for kicking/waving/writing) had
      excessively small aggregation weights. Now A_physical is registered
      as-is, consistent with LAST-v2's AdaptiveGraphConv.

    Param cost for base (C=[40,80,160], 3+4+4 blocks):
      Physical alpha: K × C_in per block (e.g. 3×40=120 at stage 1)
      A_learned:      V×V = 625 per block (11 blocks × 625 = 6,875 total)
      node_proj:      C_in × embed_dim per block (40×10=400 at stage 1)
      alpha_dyn:      C_in per block
      Total new (E1+E2): ~41,155 for base  →  ~405K total
    """

    def __init__(self, in_channels: int, out_channels: int, A_physical: torch.Tensor):
        super().__init__()

        K = A_physical.shape[0] if A_physical.dim() == 3 else 1
        V = A_physical.shape[-1]
        self.K = K
        self.in_channels = in_channels

        # ── Physical subsets (FIX E0: no re-normalization) ────────────────────
        # graph.py normalize_digraph already applied D^{-1} column normalization.
        # Registering as-is avoids the O(1/d²) double-normalization bug that
        # suppressed end-node joints (fingertips, feet).
        for k in range(K):
            A_k = A_physical[k] if A_physical.dim() == 3 else A_physical
            self.register_buffer(f'A_{k}', A_k.clone())

        # Per-channel per-subset softmax weights (K, C_in).
        # Zeros → uniform 1/K start; diverges during training per channel.
        self.alpha = nn.Parameter(torch.zeros(K, in_channels))

        # ── Learned global adjacency (E2) ─────────────────────────────────────
        # 25×25 trainable matrix: captures consistent cross-class long-range
        # joint correlations. Zero init → training starts from physical prior.
        # abs() + symmetric normalization applied per-forward for stability.
        # Excluded from weight decay (trainer no_decay group matches 'A_learned').
        self.A_learned = nn.Parameter(torch.zeros(V, V))

        # ── Dynamic adjacency projection (E1) ─────────────────────────────────
        # node_proj: C_in → embed_dim (embed_dim = max(C_in//4, 8))
        # Excluded from weight decay (add 'node_proj' to trainer no_decay list).
        self.embed_dim = max(in_channels // 4, 8)
        self.node_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, bias=False)

        # Per-channel gate for dynamic contribution.
        # Init at -4.0 → sigmoid(-4) ≈ 0.018: near-zero start, grows during training.
        # Excluded from weight decay (trainer no_decay group matches 'alpha').
        self.alpha_dyn = nn.Parameter(torch.full((in_channels,), -4.0))

        # Single shared 1×1 pointwise conv — unchanged from original.
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x_flat = x.reshape(N, C * T, V)   # (N, C*T, V)

        # ── 1. Physical K=3 static aggregation ───────────────────────────────
        alpha_w = F.softmax(self.alpha, dim=0)   # (K, C_in)
        x_agg = None
        for k in range(self.K):
            A_k = getattr(self, f'A_{k}')                      # (V, V)
            x_k = torch.matmul(x_flat, A_k).view(N, C, T, V)
            x_k = x_k * alpha_w[k].view(1, C, 1, 1)
            x_agg = x_k if x_agg is None else x_agg + x_k

        # ── 2. Learned global adjacency (E2) ──────────────────────────────────
        # D^{-1/2}|A_learned|D^{-1/2} normalization per-forward.
        A_l = self.A_learned.abs()
        row_sum_l = A_l.sum(dim=1).clamp(min=1e-6)
        D_inv_sqrt_l = row_sum_l.pow(-0.5)
        A_l_norm = D_inv_sqrt_l.unsqueeze(1) * A_l * D_inv_sqrt_l.unsqueeze(0)
        x_learn = torch.matmul(x_flat, A_l_norm).view(N, C, T, V)
        x_agg = x_agg + x_learn

        # ── 3. Dynamic per-sample adjacency (E1) ──────────────────────────────
        # Temporal-mean pool → project to embed_dim → cosine similarity → (N,V,V)
        x_mean = x.mean(dim=2, keepdim=True)              # (N, C, 1, V)
        emb    = self.node_proj(x_mean).squeeze(2)         # (N, embed_dim, V)
        emb    = emb.permute(0, 2, 1)                      # (N, V, embed_dim)
        emb    = F.normalize(emb, p=2, dim=-1)
        A_dyn  = torch.bmm(emb, emb.transpose(1, 2))      # (N, V, V) cosine sim
        A_dyn  = F.softmax(A_dyn, dim=-1)                 # row-normalize
        x_dyn  = torch.bmm(x_flat, A_dyn).view(N, C, T, V)
        gate   = torch.sigmoid(self.alpha_dyn).view(1, C, 1, 1)
        x_agg  = x_agg + gate * x_dyn

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
