import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_JointAtt(nn.Module):
    """
    Spatial-Temporal Joint Attention Module.
    
    Factorized attention mechanism to refine features based on:
    1. Joint importance ( Spatial)
    2. Frame importance (Temporal)
    
    Args:
        channel: Input channels
        reduction: Reduction ratio for MLP (default: 8)
    """
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel = channel
        reduced_channel = max(4, channel // reduction)
        
        # Temporal Attention (Focus on key frames)
        # Input: (N, C, T, V) -> AvgPool(V) -> (N, C, T, 1) -> Conv -> (N, 1, T, 1)
        self.temporal_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)), # Pool Joints: (N, C, T, 1)
            nn.Conv2d(channel, reduced_channel, 1, bias=False),
            nn.BatchNorm2d(reduced_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention (Focus on key joints)
        # Input: (N, C, T, V) -> AvgPool(T) -> (N, C, 1, V) -> Conv -> (N, 1, 1, V)
        self.spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), # Pool Time: (N, C, 1, V)
            nn.Conv2d(channel, reduced_channel, 1, bias=False),
            nn.BatchNorm2d(reduced_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Channel Attention (Squeeze & Excitation style - Optional, included in some variants)
        # For simplicity and speed, we stick to Factorized S-T Attention as per design.

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        """
        N, C, T, V = x.shape
        
        # Temporal Attention Branch
        # Uses C channels to compute importance of T frames
        # Output: (N, C, T, 1) - channel-wise temporal weights? 
        # Standard implementation: (N, 1, T, 1) or (N, C, T, 1)
        # We use Channel-wise attention over time: (N, C, T, 1)
        att_t = self.temporal_att(x) # (N, C, T, 1)
        
        # Spatial Attention Branch
        # Output: (N, C, 1, V)
        att_s = self.spatial_att(x)  # (N, C, 1, V)
        
        # Apply Attention
        # x = x * att_t * att_s
        out = x * att_t
        out = out * att_s
        
        return out
