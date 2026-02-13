"""
LAST Block - Combining A-GCN, TSM, and Linear Attention

This is the core building block of the LAST architecture, combining:
1. Adaptive GCN - Spatial joint relationships
2. Temporal Shift Module - Local temporal mixing
3. Linear Attention - Global temporal context

Order: A-GCN → TSM → Linear Attention
Rationale: Spatial → Local Temporal → Global Temporal
"""

import torch
import torch.nn as nn

from .agcn import AdaptiveGCN
from .tsm import TSMBlock, TemporalShiftModule
from .linear_attn import LinearAttention, LinearAttentionBlock


class LASTBlock(nn.Module):
    """
    LAST Block: Adaptive GCN + TSM + Linear Attention.
    
    This is one complete block in the LAST architecture.
    Multiple blocks are stacked with increasing channels.
    
    Processing order:
        Input → A-GCN (spatial) → TSM (local temporal) → Linear Attn (global temporal) → Output
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_joints: Number of skeleton joints (25 for NTU RGB+D)
        num_heads: Number of attention heads (default: 8)
        tsm_ratio: TSM shift ratio (default: 0.125)
        dropout: Dropout rate (default: 0.1)
        use_learned_adj: Use learned adjacency in A-GCN (default: True)
        use_dynamic_adj: Use dynamic adjacency in A-GCN (default: True)
        use_tsm: Whether to use TSM (default: True)
        use_attention: Whether to use attention (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 25,
        num_heads: int = 8,
        tsm_ratio: float = 0.125,
        dropout: float = 0.1,
        use_learned_adj: bool = True,
        use_dynamic_adj: bool = True,
        use_tsm: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_tsm = use_tsm
        self.use_attention = use_attention
        
        # 1. Adaptive GCN for spatial modeling
        self.agcn = AdaptiveGCN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_joints=num_joints,
            num_subsets=3,
            use_learned=use_learned_adj,
            use_dynamic=use_dynamic_adj,
            residual=True
        )
        
        # 2. Temporal Shift Module (zero parameters!)
        if self.use_tsm:
            self.tsm = TemporalShiftModule(
                num_channels=out_channels,
                shift_ratio=tsm_ratio
            )
        
        # 3. Linear Attention for global temporal context
        if self.use_attention:
            self.attention = LinearAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                kernel_fn='elu'
            )
            
            # Layer norm after attention
            self.norm = nn.LayerNorm(out_channels)
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through LAST block.
        
        Args:
            x: Input tensor (B, C_in, T, V)
               B = batch size
               C_in = input channels
               T = time frames  
               V = joints/vertices
               
        Returns:
            Output tensor (B, C_out, T, V)
        """
        # 1. Spatial modeling with A-GCN
        x = self.agcn(x)  # (B, C_out, T, V)
        
        # 2. Local temporal mixing with TSM
        if self.use_tsm:
            x = self.tsm(x)  # (B, C_out, T, V) - zero params!
        
        # 3. Global temporal context with Linear Attention
        if self.use_attention:
            # Store for residual
            residual = x
            
            # Apply attention
            x = self.attention(x)  # (B, C_out, T, V)
            
            # Add residual and normalize
            # Need to reshape for LayerNorm
            B, C, T, V = x.shape
            x = x.permute(0, 3, 2, 1).contiguous()  # (B, V, T, C)
            residual = residual.permute(0, 3, 2, 1).contiguous()
            
            x = x + residual
            x = x.view(B * V, T, C)
            x = self.norm(x)
            x = x.view(B, V, T, C)
            x = x.permute(0, 3, 2, 1).contiguous()  # (B, C, T, V)
        
        # Final dropout
        x = self.dropout(x)
        
        return x


class LASTBlockStack(nn.Module):
    """
    Stack of LAST blocks with configurable channels.
    
    This is useful for building the full LAST architecture.
    
    Args:
        channels: List of channel dimensions [in, hidden1, hidden2, ..., out]
        num_joints: Number of skeleton joints
        num_heads: Number of attention heads per block
        tsm_ratio: TSM shift ratio
        dropout: Dropout rate
    
    Example:
        channels = [64, 64, 128, 128, 256]
        Creates 4 blocks: 64→64, 64→128, 128→128, 128→256
    """
    
    def __init__(
        self,
        channels: list,
        num_joints: int = 25,
        num_heads: int = 8,
        tsm_ratio: float = 0.125,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_blocks = len(channels) - 1
        
        # Create blocks
        self.blocks = nn.ModuleList([
            LASTBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_joints=num_joints,
                num_heads=num_heads,
                tsm_ratio=tsm_ratio,
                dropout=dropout
            )
            for i in range(self.num_blocks)
        ])
    
    def forward(self, x):
        """
        Forward through all blocks.
        
        Args:
            x: Input tensor (B, C_in, T, V)
            
        Returns:
            Output tensor (B, C_out, T, V)
        """
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    # Test LAST Block
    print("Testing LAST Block...")
    
    # Test 1: Single LAST block
    print("\nTest 1: Single LAST Block")
    block = LASTBlock(
        in_channels=64,
        out_channels=128,
        num_joints=25,
        num_heads=8,
        tsm_ratio=0.125,
        dropout=0.1
    )
    
    x = torch.randn(2, 64, 50, 25)  # B=2, C=64, T=50, V=25
    out = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 128, 50, 25), "Output shape mismatch!"
    print("✓ Shape test passed")
    
    # Test 2: Parameter count
    print("\nTest 2: Parameter count")
    num_params = sum(p.numel() for p in block.parameters())
    print(f"LAST Block parameters: {num_params:,}")
    
    # Breakdown
    agcn_params = sum(p.numel() for p in block.agcn.parameters())
    tsm_params = sum(p.numel() for p in block.tsm.parameters()) if block.use_tsm else 0
    attn_params = sum(p.numel() for p in block.attention.parameters()) if block.use_attention else 0
    
    print(f"  A-GCN: {agcn_params:,} params")
    print(f"  TSM: {tsm_params:,} params (should be 0!)")
    print(f"  Attention: {attn_params:,} params")
    
    assert tsm_params == 0, "TSM should have zero parameters!"
    print("✓ TSM has zero parameters")
    
    # Test 3: LAST Block Stack
    print("\nTest 3: LAST Block Stack")
    channels = [64, 64, 128, 128, 256]
    stack = LASTBlockStack(
        channels=channels,
        num_joints=25,
        num_heads=8
    )
    
    x = torch.randn(2, 64, 50, 25)
    out = stack(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: (2, 256, 50, 25)")
    assert out.shape == (2, 256, 50, 25), "Stack output shape mismatch!"
    print("✓ Stack test passed")
    
    print(f"\nStack has {stack.num_blocks} blocks:")
    for i in range(stack.num_blocks):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        print(f"  Block {i+1}: {in_ch} → {out_ch}")
    
    # Test 4: Ablation options
    print("\nTest 4: Ablation options (disable components)")
    
    # A-GCN only
    block_agcn_only = LASTBlock(
        in_channels=64, out_channels=128, num_joints=25,
        use_tsm=False, use_attention=False
    )
    out = block_agcn_only(x)
    assert out.shape == (2, 128, 50, 25)
    print("✓ A-GCN only: works")
    
    # A-GCN + TSM only
    block_no_attn = LASTBlock(
        in_channels=64, out_channels=128, num_joints=25,
        use_tsm=True, use_attention=False
    )
    out = block_no_attn(x)
    assert out.shape == (2, 128, 50, 25)
    print("✓ A-GCN + TSM: works")
    
    # Full block
    out = block(x)
    assert out.shape == (2, 128, 50, 25)
    print("✓ Full block (A-GCN + TSM + Attn): works")
    
    print("\n" + "="*60)
    print("✓ ALL LAST BLOCK TESTS PASSED!")
    print("="*60)
    print("\nThe LAST block successfully combines:")
    print("  1. A-GCN for spatial modeling")
    print("  2. TSM for zero-param temporal mixing")
    print("  3. Linear Attention for global context")
