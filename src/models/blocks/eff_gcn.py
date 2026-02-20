import torch
import torch.nn as nn
import torch.nn.functional as F
from .st_joint_att import ST_JointAtt
from .linear_attn import LinearAttention


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AdaptiveGraphConv(nn.Module):
    """
    Adaptive Graph Convolution with three complementary adjacency components:

    1. A_physical (K, V, V) — fixed normalized skeleton topology from Graph().
       Encodes the biomechanical prior (parent→child edges, 3 spatial subsets).
       Registered as a buffer: no gradients, moves to device automatically.

    2. A_learned (V, V) — global trainable matrix shared across all samples.
       Learns action-class-agnostic long-range joint correlations that are absent
       from the skeleton graph (e.g., hand↔foot co-activation during kicks).
       Initialized as zeros so training starts from the physical prior alone.
       Normalized per-forward to stay well-conditioned.

    3. A_dynamic (B, V, V) — sample-dependent correlation computed from current
       input features. Captures instance-specific pose context (e.g., which
       joints are moving together in this particular clip). Uses a lightweight
       channel-projection (embed_dim = C//8) for efficiency.

    Per-subset projection weights W_k:
    Standard ST-GCN applies a separate learnable W_k per subset AFTER graph
    propagation, giving each partition its own channel mixing. This block uses
    K+2 projection convolutions (K physical subsets + 1 for A_learned + 1 for
    A_dynamic), preserving the directional decoupling introduced in the fix for
    the original adjacency-sum bug.

    Parameter budget for 64→64 block (V=25, K=3, embed_dim=8):
      A_learned:   25×25 = 625 params
      node_proj:   64×8 = 512 params (1×1 conv)
      W_k (K+2):   5 × (64×1×1→64) = 5 × 4096 = 20,480 params
      Total new:   ~21,617 params per block — negligible vs backbone total
    """

    def __init__(self, in_channels, out_channels, A_physical, embed_dim_ratio=8):
        """
        Args:
            in_channels:   Input feature channels.
            out_channels:  Output feature channels.
            A_physical:    Pre-computed adjacency tensor (K, V, V) from Graph().
            embed_dim_ratio: C // embed_dim_ratio for dynamic graph projection.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── Physical adjacency ─────────────────────────────────────────────
        # (K, V, V): K directional subsets from spatial partitioning strategy.
        self.register_buffer('A_physical', A_physical)
        K = A_physical.shape[0] if A_physical.dim() == 3 else 1
        V = A_physical.shape[-1]
        self.K = K
        self.V = V

        # ── Learnable global adjacency ─────────────────────────────────────
        # Initialized to zeros: training starts purely from physical prior.
        # Gradient flows freely — the model learns which extra edges matter.
        self.A_learned = nn.Parameter(torch.zeros(V, V))

        # ── Dynamic adjacency projection ───────────────────────────────────
        embed_dim = max(4, in_channels // embed_dim_ratio)
        self.embed_dim = embed_dim
        # Lightweight 1×1 conv to project features to embedding space
        self.node_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # ── Per-subset projection weights W_k ─────────────────────────────
        # One conv per adjacency component: K physical + 1 learned + 1 dynamic
        num_subsets = K + 2  # K physical subsets, 1 learned, 1 dynamic
        self.subset_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for _ in range(num_subsets)
        ])
        self.num_subsets = num_subsets

    def _compute_dynamic_A(self, x):
        """
        Compute sample-dependent adjacency from input features.

        Args:
            x: (N, C, T, V)
        Returns:
            A_dynamic: (N, V, V) — normalized pairwise joint correlations
        """
        N, C, T, V = x.shape
        # Pool time → (N, C, 1, V), then project to embed_dim
        x_pool = x.mean(dim=2, keepdim=True)          # (N, C, 1, V)
        emb = self.node_proj(x_pool).squeeze(2)       # (N, embed_dim, V)
        emb = emb.permute(0, 2, 1)                    # (N, V, embed_dim)
        # L2-normalize embeddings along embed_dim
        emb = F.normalize(emb, p=2, dim=-1)
        # Cosine similarity matrix → softmax for non-negative weights
        A_dyn = torch.bmm(emb, emb.transpose(1, 2))   # (N, V, V)
        A_dyn = torch.softmax(A_dyn, dim=-1)
        return A_dyn

    def _normalize_learned_A(self):
        """
        Symmetric degree normalization of A_learned: D^{-1/2} A D^{-1/2}.
        Applied per-forward to keep the matrix well-conditioned regardless of
        gradient updates. Uses abs() to handle negative learned weights safely.
        """
        A = self.A_learned.abs()  # ensure non-negative before degree norm
        # Clamp small values away from zero for numerical stability
        row_sum = A.sum(dim=1).clamp(min=1e-6)
        D_inv_sqrt = row_sum.pow(-0.5)
        return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            out: (N, out_channels, T, V)
        """
        N, C, T, V = x.shape
        x_flat = x.view(N, C * T, V)   # (N, C*T, V) for batched matmul

        out = 0

        # ── Physical subsets ───────────────────────────────────────────────
        if self.A_physical.dim() == 3:
            for k in range(self.K):
                # x_flat @ A[k]: (N, C*T, V) × (V, V) → (N, C*T, V)
                x_agg = torch.matmul(x_flat, self.A_physical[k])
                x_agg = x_agg.view(N, C, T, V)
                out = out + self.subset_convs[k](x_agg)
        else:
            x_agg = torch.matmul(x_flat, self.A_physical)
            x_agg = x_agg.view(N, C, T, V)
            out = out + self.subset_convs[0](x_agg)

        # ── Learned global adjacency ───────────────────────────────────────
        A_l = self._normalize_learned_A()          # (V, V)
        x_agg_l = torch.matmul(x_flat, A_l)       # (N, C*T, V)
        x_agg_l = x_agg_l.view(N, C, T, V)
        out = out + self.subset_convs[self.K](x_agg_l)

        # ── Dynamic adjacency ──────────────────────────────────────────────
        A_dyn = self._compute_dynamic_A(x)         # (N, V, V)
        # Batched matmul: (N, C*T, V) × (N, V, V) → (N, C*T, V)
        x_agg_d = torch.bmm(x_flat, A_dyn)
        x_agg_d = x_agg_d.view(N, C, T, V)
        out = out + self.subset_convs[self.K + 1](x_agg_d)

        return out


class EffGCNBlock(nn.Module):
    """
    Efficient GCN Block (LAST v2 Core) — fully upgraded.

    Structure per block:
    1. AdaptiveGraphConv  — spatial modeling with physical + learned + dynamic A,
                            per-subset projection weights W_k
    2. BatchNorm + ReLU   — post-GCN normalization
    3. ST_JointAtt        — factorized spatial-temporal attention refinement
    4. TCN or LinearAttn  — temporal modeling (local TCN in early stages,
                            global LinearAttn in deep stages)
    5. Residual           — skip connection with optional channel projection

    Args:
        in_channels:      Input channels
        out_channels:     Output channels
        A:                Physical adjacency (K, V, V) from Graph()
        stride:           Temporal stride (1 or 2)
        residual:         Whether to use residual connection
        use_linear_attn:  Use LinearAttention instead of TCN for temporal modeling
    """
    def __init__(self, in_channels, out_channels, A, stride=1,
                 residual=True, use_linear_attn=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # Note: A_physical is registered as a buffer inside AdaptiveGraphConv;
        # no need to duplicate it here.

        # 1. Adaptive Graph Convolution (spatial)
        self.gcn = AdaptiveGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            A_physical=A,
        )

        # 2. Post-GCN normalization
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.gcn_relu = nn.ReLU(inplace=True)

        # 3. ST-Joint Attention
        self.st_att = ST_JointAtt(out_channels, reduction=4)

        # 4. Temporal modeling
        self.use_linear_attn = use_linear_attn
        if use_linear_attn:
            # Global temporal context via linear attention
            self.tcn = LinearAttention(embed_dim=out_channels, num_heads=4)
        else:
            # Local temporal context via depthwise-separable TCN (kernel 9×1)
            pad = (9 - 1) // 2
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_channels, out_channels, (9, 1),
                                       (stride, 1), (pad, 0)),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.1, inplace=True)
            )

        # 5. Residual connection
        self.residual = residual
        if not residual:
            self.residual_path = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual_path = nn.Identity()
        else:
            self.residual_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        # 1. Adaptive spatial GCN
        x = self.gcn(x)                 # (N, out_channels, T, V)
        x = self.gcn_bn(x)
        x = self.gcn_relu(x)

        # 2. ST-Joint attention refinement
        x = self.st_att(x)

        # 3. Temporal modeling
        if self.use_linear_attn:
            x = self.tcn(x)
            # Explicit temporal downsample when stride > 1
            if self.stride > 1:
                x = F.avg_pool2d(x, kernel_size=(3, 1), stride=(self.stride, 1),
                                 padding=(1, 0))
        else:
            x = self.tcn(x)

        # 4. Residual
        if self.residual and self.residual_path is not None:
            x = x + self.residual_path(res)

        x = self.relu(x)
        return x
