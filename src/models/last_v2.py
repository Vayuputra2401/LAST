import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.eff_gcn import EffGCNBlock
from .graph import Graph

# Configuration for variants (Channels per stage, Blocks per stage)
MODEL_VARIANTS = {
    'small': {
        'channels': [64, 128, 256],
        'blocks': [3, 3, 4],     # Total 10 blocks
        'strides': [1, 2, 2],    # Downsample time at stage 2 and 3 start
        'use_attn': [False, False, True] # Use LinearAttn only in last stage
    },
    'base': {
        'channels': [96, 192, 384],
        'blocks': [4, 5, 5],     # Total 14 blocks
        'strides': [1, 2, 2],
        'use_attn': [False, True, True] # Use LinearAttn in stage 2 and 3
    },
    'large': {
        'channels': [128, 256, 512],
        'blocks': [6, 6, 6],     # Total 18 blocks
        'strides': [1, 2, 2],
        'use_attn': [False, True, True]
    }
}

class LAST_v2(nn.Module):
    """
    LAST v2: Latent Action-Space Transformer (Efficient Version).

    Features:
    - Shared Backbone for Multi-Input Streams (Joint, Velocity, Bone).
    - EfficientGCN Blocks with Hybrid Temporal Modeling.
    - Scalable Architecture (Small, Base, Large).

    Args:
        num_classes: Number of action classes.
        variant: 'small', 'base', 'large'.
        in_channels: Input channels (3).
        graph_layout: 'ntu-rgb+d'.
        graph_strategy: 'spatial'.
    """
    def __init__(self, num_classes=60, variant='base', in_channels=3,
                 graph_layout='ntu-rgb+d', graph_strategy='spatial'):
        super().__init__()

        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {list(MODEL_VARIANTS.keys())}")

        config = MODEL_VARIANTS[variant]
        self.variant = variant

        # 1. Graph & Adjacency
        # FIX (Bug 1): Preserve all K subsets as (K, V, V). Do NOT sum them.
        # The 'spatial' strategy produces 3 directionally distinct matrices:
        #   A[0]: self-connections (same hop-distance joints)
        #   A[1]: centripetal edges (joint->parent, toward body center joint 20)
        #   A[2]: centrifugal edges (joint->child, away from center)
        # Summing collapses these into one undirected matrix, destroying the
        # directional inductive bias that gives ST-GCN its spatial power.
        # EffGCNBlock now sums per-subset weighted messages internally.
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # A.shape: (K, V, V) for 'spatial', (1, V, V) for 'uniform'
        self.register_buffer('A', A)

        # 2. Per-stream Input BatchNorm.
        # FIX: A single shared data_bn was used across all 3 MIB streams.
        # Joint (XYZ positions, std≈1), velocity (finite diffs, std≈0.02-0.05),
        # and bone (child-parent vectors, std≈0.3-1) have very different scales.
        # A shared BN's running mean/variance is the mixture of all 3 modalities,
        # meaning no individual stream is correctly normalized — velocity in
        # particular was severely under-normalized, explaining the plateau at 3-5%.
        # Fix: 3 independent BN modules indexed by MIB stream order
        # (0=joint, 1=velocity, 2=bone). Each learns its own mean/variance.
        # For single-stream (non-MIB) inference, stream_idx=0 (joint) is used.
        # Parameter cost: 3 × 2 × in_channels = 3 × 6 = 18 scalars — negligible.
        self.stream_names = ['joint', 'velocity', 'bone']
        self.data_bn = nn.ModuleList([
            nn.BatchNorm2d(in_channels) for _ in range(3)
        ])

        # 3. Backbone Construction
        layers = []
        c_in = in_channels

        for stage_idx, (c_out, num_blocks, stride, use_attn) in enumerate(zip(
            config['channels'], config['blocks'], config['strides'], config['use_attn']
        )):
            for i in range(num_blocks):
                # Stride only on first block of stage
                s = stride if i == 0 else 1

                layers.append(EffGCNBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    A=self.A,       # Pass full (K, V, V) tensor
                    stride=s,
                    residual=True,
                    use_linear_attn=use_attn
                ))
                c_in = c_out

        self.backbone = nn.Sequential(*layers)

        # 4. Classification Head
        # Dropout before FC: 9.2M params on 40K samples needs head regularization.
        # Consistent with LAST-E (dropout=0.3) and SOTA (EfficientGCN 0.25, CTR-GCN 0.5).
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c_in, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (N, C, T, V, M) or Dictionary of streams.
               If Dict: {'joint': ..., 'velocity': ..., 'bone': ...}

        Returns:
            log-probabilities: (N, num_classes)
              For multi-stream input: log(mean_softmax) across streams.
              For single-stream input: raw logits (CrossEntropyLoss compatible).

        FIX (V0) — Softmax probability averaging replaces raw logit summation.

        Prior design: return sum(logit_joint + logit_velocity + logit_bone).
        Problem: raw logit sum is scale-dependent. At validation, BN runs in
        eval mode using running stats from AUGMENTED training data. Velocity
        is near-zero for slow actions at eval; BN amplifies this using wrong
        stats, producing a corrupted logit vector that can overrule correct
        joint+bone predictions. Training hides this via per-batch BN + augmentation.
        Result: wild per-epoch val accuracy oscillation correlated with which
        action classes (slow vs. fast) appear in each batch.

        Fix: apply softmax per-stream before combining. Softmax output is
        invariant to constant offsets (exactly what BN running-stat mismatch
        produces). A degenerate velocity stream contributes ~1/60 per class
        (uniform noise) instead of a corrupted spike that corrupts the argmax.

        Returns log(mean_softmax) = log-probabilities → use NLLLoss in trainer.
        """
        if isinstance(x, dict):
            # Multi-stream: average softmax probabilities (scale-invariant fusion).
            # Returns log of mean probability → compatible with nn.NLLLoss.
            prob_sum = None
            n_streams = 0
            for stream_name, stream_input in x.items():
                stream_idx = self.stream_names.index(stream_name) \
                    if stream_name in self.stream_names else 0
                logit = self._forward_single_stream(stream_input, stream_idx=stream_idx)
                p = F.softmax(logit, dim=-1)                       # (N, num_classes)
                prob_sum = p if prob_sum is None else prob_sum + p
                n_streams += 1
            # log of mean probability — NLLLoss-compatible log-probabilities
            return torch.log(prob_sum / n_streams + 1e-8)
        else:
            # Single-stream inference: raw logits (CrossEntropyLoss compatible)
            return self._forward_single_stream(x, stream_idx=0)

    def _forward_single_stream(self, x, stream_idx: int = 0):
        """
        Process a single skeleton stream.

        Args:
            x:          Tensor of shape (N, C, T, V, M) or (N, C, T, V).
            stream_idx: Index into self.data_bn for per-stream normalization.
                        0=joint, 1=velocity, 2=bone.

        Returns:
            logits: (N, num_classes)
        """
        # Accept both 4D and 5D inputs.
        if x.dim() == 4:
            x = x.unsqueeze(-1)  # (N, C, T, V) -> (N, C, T, V, 1)

        N, C, T, V, M = x.shape

        # Rearrange: (N, C, T, V, M) -> (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # Per-stream BatchNorm: each modality (joint/velocity/bone) normalizes
        # independently so the backbone sees correctly scaled inputs for all 3.
        x = self.data_bn[stream_idx](x)

        # Backbone
        x = self.backbone(x)

        # Global Average Pooling: (N*M, C_out, T_out, V) -> (N*M, C_out)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(N * M, -1)

        # Classifier (with head dropout for regularization)
        x = self.drop(x)
        x = self.fc(x)  # (N*M, num_classes)

        # Merge body dimension: mean over M bodies -> (N, num_classes)
        x = x.view(N, M, -1).mean(dim=1)

        return x
