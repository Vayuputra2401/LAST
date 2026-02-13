"""
LAST - Lightweight Adaptive-Shift Transformer

Complete LAST model for skeleton-based action recognition.

Architecture:
    Input (B, 3, T, V, M) →
    Select primary body (B, 3, T, V) →
    Stem Conv (3 → 64) → 
    LAST Block 1 (64 → 64) →
    LAST Block 2 (64 → 128) →
    LAST Block 3 (128 → 128) →
    LAST Block 4 (128 → 256) →
    Global Average Pooling (B, 256, T, V → B, 256) →
    Dropout(0.5) →
    FC (256 → num_classes) →
    Output (B, num_classes)
    
Reference: LAST framework for efficient skeleton-based HAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LASTBlock, LASTBlockStack


class LAST(nn.Module):
    """
    LAST: Lightweight Adaptive-Shift Transformer.
    
    Complete model for skeleton-based action recognition combining:
    - Adaptive GCN for spatial modeling
    - Temporal Shift Module for zero-param temporal mixing
    - Linear Attention for global temporal context
    
    Args:
        num_classes: Number of action classes (120 for NTU RGB+D)
        num_joints: Number of skeleton joints (25 for NTU RGB+D)
        in_channels: Input channels (3 for x,y,z coordinates)
        channels: List of channel dimensions for each block
        num_heads: Number of attention heads
        tsm_ratio: TSM shift ratio
        dropout: Dropout rate
        fc_dropout: Dropout before final classifier
    """
    
    def __init__(
        self,
        num_classes: int = 120,
        num_joints: int = 25,
        in_channels: int = 3,
        channels: list = None,
        num_heads: int = 8,
        tsm_ratio: float = 0.125,
        dropout: float = 0.1,
        fc_dropout: float = 0.5
    ):
        super().__init__()
        
        if channels is None:
            # Default LAST-Base configuration
            channels = [64, 64, 128, 128, 256]
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.channels = channels
        
        # Stem: Project input features to first hidden dimension
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # LAST blocks
        self.blocks = LASTBlockStack(
            channels=channels,
            num_joints=num_joints,
            num_heads=num_heads,
            tsm_ratio=tsm_ratio,
            dropout=dropout
        )
        
        # Global average pooling
        # (B, C, T, V) → (B, C)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(channels[-1], num_classes)
    
    def _select_primary_body(self, x):
        """
        Select primary body from multi-body input.
        
        Args:
            x: Input (B, C, T, V, M) where M=max_bodies
            
        Returns:
            Primary body (B, C, T, V)
        """
        B, C, T, V, M = x.shape
        
        if M == 1:
            return x.squeeze(-1)  # (B, C, T, V)
        
        # Select body with most non-zero frames
        # Sum over channels and joints to get activity per body per frame
        activity = x.abs().sum(dim=(1, 3))  # (B, T, M)
        body_scores = (activity > 0).sum(dim=1)  # (B, M) - count non-zero frames
        
        # Select most active body for each sample in batch
        primary_indices = body_scores.argmax(dim=1)  # (B,)
        
        # Gather primary body
        # Create index tensor for gathering
        indices = primary_indices.view(B, 1, 1, 1, 1).expand(-1, C, T, V, -1)
        primary = torch.gather(x, dim=4, index=indices)  # (B, C, T, V, 1)
        primary = primary.squeeze(-1)  # (B, C, T, V)
        
        return primary
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V, M) or (B, C, T, V)
               B = batch size
               C = channels (3 for x,y,z)
               T = time frames
               V = joints
               M = max bodies (optional, usually 2)
               
        Returns:
            Logits (B, num_classes)
        """
        # Handle multi-body input
        if x.dim() == 5:
            x = self._select_primary_body(x)  # (B, C, T, V)
        
        # Verify shape
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
        B, C, T, V = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} input channels, got {C}"
        assert V == self.num_joints, f"Expected {self.num_joints} joints, got {V}"
        
        # Stem: Project to first hidden dimension
        x = self.stem(x)  # (B, channels[0], T, V)
        
        # LAST blocks
        x = self.blocks(x)  # (B, channels[-1], T, V)
        
        # Global pooling
        x = self.pool(x)  # (B, channels[-1], 1, 1)
        x = x.view(B, -1)  # (B, channels[-1])
        
        # Classification
        x = self.fc_dropout(x)
        logits = self.fc(x)  # (B, num_classes)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self):
        """Get model configuration."""
        return {
            'num_classes': self.num_classes,
            'num_joints': self.num_joints,
            'in_channels': self.in_channels,
            'channels': self.channels,
            'num_heads': 8,  # From blocks
            'tsm_ratio': 0.125,
            'dropout': 0.1,
            'fc_dropout': 0.5
        }


def create_last_base(num_classes=120, num_joints=25):
    """
    Create LAST-Base model variant.
    
    Configuration:
    - 4 blocks: 64→64, 64→128, 128→128, 128→256
    - 8 attention heads
    - 0.125 TSM ratio
    
    Target: <1M parameters, <1 GFLOP
    """
    return LAST(
        num_classes=num_classes,
        num_joints=num_joints,
        channels=[64, 64, 128, 128, 256],
        num_heads=8,
        tsm_ratio=0.125,
        dropout=0.1,
        fc_dropout=0.5
    )


def create_last_small(num_classes=120, num_joints=25):
    """
    Create LAST-Small model variant.
    
    Smaller version for even faster inference.
    """
    return LAST(
        num_classes=num_classes,
        num_joints=num_joints,
        channels=[32, 32, 64, 64, 128],
        num_heads=4,
        tsm_ratio=0.125,
        dropout=0.1,
        fc_dropout=0.5
    )


def create_last_large(num_classes=120, num_joints=25):
    """
    Create LAST-Large model variant.
    
    Larger version for better accuracy.
    """
    return LAST(
        num_classes=num_classes,
        num_joints=num_joints,
        channels=[128, 128, 256, 256, 512],
        num_heads=8,
        tsm_ratio=0.125,
        dropout=0.1,
        fc_dropout=0.5
    )


if __name__ == '__main__':
    # Test LAST model
    print("="*60)
    print("Testing LAST Model")
    print("="*60)
    
    # Test 1: LAST-Base
    print("\nTest 1: LAST-Base model")
    model = create_last_base(num_classes=120, num_joints=25)
    
    # Single body input
    x = torch.randn(2, 3, 64, 25)  # B=2, C=3, T=64, V=25
    logits = model(x)
    
    print(f"Input shape: {x.shape}") 
    print(f"Output shape: {logits.shape}")
    print(f"Expected: (2, 120)")
    assert logits.shape == (2, 120), "Output shape mismatch!"
    print("✓ Single body test passed")
    
    # Multi-body input
    x_multi = torch.randn(2, 3, 64, 25, 2)  # B=2, C=3, T=64, V=25, M=2
    logits_multi = model(x_multi)
    assert logits_multi.shape == (2, 120)
    print("✓ Multi-body test passed")
    
    # Test 2: Parameter count
    print("\nTest 2: Parameter count")
    num_params = model.count_parameters()
    print(f"Total parameters: {num_params:,}")
    print(f"Target: < 1,000,000 (1M)")
    
    if num_params < 1_000_000:
        print("✓ Lightweight model (<1M params)")
    else:
        print(f"⚠ Model has {num_params/1e6:.2f}M parameters")
    
    # Test 3: Model variants
    print("\nTest 3: Model variants")
    
    model_small = create_last_small(120, 25)
    x_test = torch.randn(1, 3, 64, 25)
    out_small = model_small(x_test)
    assert out_small.shape == (1, 120)
    params_small = model_small.count_parameters()
    print(f"LAST-Small: {params_small:,} params")
    
    params_base = model.count_parameters()
    print(f"LAST-Base:  {params_base:,} params")
    
    model_large = create_last_large(120, 25)
    out_large = model_large(x_test)
    assert out_large.shape == (1, 120)
    params_large = model_large.count_parameters()
    print(f"LAST-Large: {params_large:,} params")
    
    print("✓ All variants work")
    
    # Test 4: No NaN/Inf
    print("\nTest 4: No NaN/Inf in output")
    assert not torch.isnan(logits).any(), "Found NaN in output!"
    assert not torch.isinf(logits).any(), "Found Inf in output!"
    print("✓ Output is clean")
    
    # Test 5: Configuration
    print("\nTest 5: Get configuration")
    config = model.get_config()
    print("Model config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("✓ ALL LAST MODEL TESTS PASSED!")
    print("="*60)
    print(f"\nLAST-Base is ready for training!")
    print(f"  - Parameters: {params_base:,}")
    print(f"  - Input: (B, 3, T, 25)")
    print(f"  - Output: (B, 120)")
