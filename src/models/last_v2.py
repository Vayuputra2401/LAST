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
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        
        # EffGCNBlock expects (V, V). If strategy returns (K, V, V), sum them.
        if A.dim() == 3:
            A = A.sum(dim=0)
            # Clamp to binary or normalize? 
            # Usually ST-GCN uses normalized A. Summing might break normalization.
            # But for "Efficient" message passing, we just want connectivity.
            # Let's trust that the learned weights will handle the scale.
            
        self.register_buffer('A', A)
        
        # 2. Input BN (Data Normalization Layer)
        # Input: (N, C, T, V) -> (N, C, T, V) normalized
        self.data_bn = nn.BatchNorm1d(in_channels * 25) # Standard trick: BN over V*C?
        # Actually standard SOTA (CTR-GCN) does BN on (N, C, T, V).
        # Let's use BatchNorm2d on (N, C, T, V) or manually.
        # "CTR-GCN uses scalar BN on (N, C*V, T) then reshape". 
        # Simpler: nn.BatchNorm2d(C) treating T*V as spatial?
        # We will use BN on 'C' dimension.
        self.data_bn = nn.BatchNorm2d(in_channels)
        
        # 3. Backbone Construction
        layers = []
        c_in = in_channels
        
        for stage_idx, (c_out, num_blocks, stride, use_attn) in enumerate(zip(
            config['channels'], config['blocks'], config['strides'], config['use_attn']
        )):
            for i in range(num_blocks):
                # Stride only on first block of stage
                s = stride if i == 0 else 1
                
                # Residual connection logic handled inside block
                # but we need to ensure channel match for identity
                
                layers.append(EffGCNBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    A=self.A,
                    stride=s,
                    residual=True,
                    use_linear_attn=use_attn
                ))
                c_in = c_out
                
        self.backbone = nn.Sequential(*layers)
        
        # 4. Classification Head
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
            logits: (N, num_classes)
        """
        # Handle MIB (Multi-Input Branch)
        if isinstance(x, dict):
            # Sum scores from all available streams
            score_sum = 0
            for stream_name, stream_input in x.items():
                score_sum += self._forward_single_stream(stream_input)
            return score_sum
        else:
            # Single stream
            return self._forward_single_stream(x)

    def _forward_single_stream(self, x):
        N, C, T, V, M = x.shape
        
        # Permute to (N*M, C, T, V) to process bodies together
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Data BN
        x = self.data_bn(x)
        
        # Backbone
        x = self.backbone(x)
        
        # Global Pooling
        # x shape: (NM, C_out, T_out, V)
        # Pool over T, V
        x = F.adaptive_avg_pool2d(x, (1, 1)) # (NM, C, 1, 1)
        x = x.view(N * M, -1) # (NM, C)
        
        # FC
        x = self.fc(x) # (NM, num_classes)
        
        # Reshape back to (N, M, num_classes) and mean over bodies
        x = x.view(N, M, -1)
        x = x.mean(dim=1) # (N, num_classes)
        
        return x
