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

class EffGCNBlock(nn.Module):
    """
    Efficient GCN Block (Last v2 Core).
    
    Structure:
    1. Spatial GCN (Separable Conv on V)
    2. ST-Joint Attention (Refinement)
    3. Temporal Modeling (Hybrid: TCN or Linear Attn)
    4. Residual Connection
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        A: Adjacency matrix (V, V)
        stride: Temporal stride
        residual: Boolean
        use_linear_attn: Use Attention instead of TCN (for deep layers)
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, use_linear_attn=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # Register A as buffer so it moves to device automatically
        # A should be (V, V)
        self.register_buffer('A', A)
        
        # 1. Spatial GCN (Separable)
        self.gcn_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Attention
        self.st_att = ST_JointAtt(out_channels, reduction=4)
        
        # 3. Temporal
        self.use_linear_attn = use_linear_attn
        if use_linear_attn:
            # Global Temporal Context
            # LinearAttention signature: (embed_dim, num_heads, ...)
            self.tcn = LinearAttention(embed_dim=out_channels, num_heads=4)
        else:
            # Local Temporal Context (Separable TCN)
            # Kernel (9, 1) means 9 temporal frames, 1 joint
            pad = (9 - 1) // 2
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_channels, out_channels, (9, 1), (stride, 1), (pad, 0)),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.1, inplace=True)
            )

        # 4. Residual
        self.residual = residual
        if not residual:
            self.residual_path = nn.Identity() # Placeholder, logic handled in forward
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
        
        # 1. Spatial GCN (Graph Conv)
        # X' = A * X * W  =>  X = W(X); X = A(X)
        x = self.gcn_conv(x) # W(X)
        
        # Apply A: (N, C, T, V) @ (V, V) -> (N, C, T, V)
        N, C, T, V = x.shape
        x = x.view(N, C * T, V)
        x = torch.matmul(x, self.A)
        x = x.view(N, C, T, V)
        
        # 2. Attention
        x = self.st_att(x)
        
        # 3. Temporal
        if self.use_linear_attn:
            # Linear Attention (Global)
            x = self.tcn(x)
            # If stride > 1, explicit downsample
            if self.stride > 1:
                x = F.avg_pool2d(x, kernel_size=(3, 1), stride=(self.stride, 1), padding=(1, 0))
        else:
            # TCN (Local + Stride handled inside)
            x = self.tcn(x)
            
        # 4. Residual
        if self.residual:
            x = x + self.residual_path(res)
            
        x = self.relu(x)
        return x
