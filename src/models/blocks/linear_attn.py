"""
Linear Attention Module

Implements O(T) complexity attention using kernel trick, reducing quadratic O(T^2) 
complexity of standard attention to linear O(T).

Reference: Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Applied to skeleton-based action recognition in LAST framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearAttention(nn.Module):
    """
    Linear Attention with O(T) complexity instead of O(T^2).
    
    Standard attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    Complexity: O(T^2 * d) due to QK^T matrix multiplication
    
    Linear attention: Attention(Q, K, V) = φ(Q) (φ(K)^T V) / (φ(Q) φ(K)^T 1)
    Complexity: O(T * d^2) by reordering operations
    
    Where φ is a kernel function, we use φ(x) = elu(x) + 1 (non-negative)
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.1)
        kernel_fn: Kernel function ('elu' or 'relu', default: 'elu')
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        kernel_fn: str = 'elu'
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kernel_fn_name = kernel_fn
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _kernel_function(self, x):
        """
        Apply kernel function φ(x) to make values non-negative.
        
        Args:
            x: Input tensor
            
        Returns:
            φ(x): Non-negative transformed tensor
        """
        if self.kernel_fn_name == 'elu':
            return F.elu(x) + 1.0  # elu(x) + 1 is always positive
        elif self.kernel_fn_name == 'relu':
            return F.relu(x) + 1e-6  # Small epsilon for numerical stability
        else:
            raise ValueError(f"Unknown kernel function: {self.kernel_fn_name}")
    
    def forward(self, x, mask=None):
        """
        Forward pass with linear attention.
        
        Args:
            x: Input tensor (B, C, T, V) or (B, T, C)
               For skeleton: (B, C, T, V) will be reshaped to (B*V, T, C)
            mask: Optional attention mask (B, T) or (B*V, T)
            
        Returns:
            Output tensor (same shape as input)
            Attention weights (optional, for visualization)
        """
        # Determine input format
        if x.dim() == 4:
            # Skeleton format: (B, C, T, V)
            B, C, T, V = x.shape
            # Reshape to treat each joint independently
            x = x.permute(0, 3, 2, 1).contiguous()  # (B, V, T, C)
            x = x.view(B * V, T, C)  # (B*V, T, C)
            original_shape = 'skeleton'
        else:
            # Sequence format: (B, T, C)
            B, T, C = x.shape
            V = 1
            original_shape = 'sequence'
        
        BV = x.shape[0]  # B or B*V
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # (BV, T, 3*C)
        qkv = qkv.reshape(BV, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BV, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (BV, H, T, D)
        
        # Apply kernel function to Q and K
        q = self._kernel_function(q)  # (BV, H, T, D)
        k = self._kernel_function(k)  # (BV, H, T, D)
        
        # Linear attention computation
        # Standard: softmax(QK^T)V requires O(T^2 * D)
        # Linear: φ(Q)(φ(K)^T V) requires O(T * D^2)
        
        # Compute K^T V: (BV, H, D, T) @ (BV, H, T, D) -> (BV, H, D, D)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (BV, H, D, D)
        
        # Compute normalizer: K^T 1 summed over time
        k_sum = k.sum(dim=2, keepdim=True)  # (BV, H, 1, D)
        
        # Compute attention output: Q (K^T V)
        out = torch.einsum('bhnd,bhdm->bhnm', q, kv)  # (BV, H, T, D)
        
        # Normalize: Q (K^T V) / (Q K^T 1)
        normalizer = torch.einsum('bhnd,bhmd->bhnm', q, k_sum)  # (BV, H, T, 1)
        normalizer = normalizer + 1e-6  # Avoid division by zero
        out = out / normalizer
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous()  # (BV, T, H, D)
        out = out.reshape(BV, T, C)  # (BV, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape back to original format
        if original_shape == 'skeleton':
            out = out.view(B, V, T, C)  # (B, V, T, C)
            out = out.permute(0, 3, 2, 1).contiguous()  # (B, C, T, V)
        
        return out


class LinearAttentionBlock(nn.Module):
    """
    Complete Linear Attention Block with Feed-Forward Network and residual connections.
    
    This is the full Transformer-style block suitable for integration into LAST.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension (default: 4*embed_dim)
        dropout: Dropout rate
        kernel_fn: Kernel function for linear attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        kernel_fn: str = 'elu'
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        
        self.embed_dim = embed_dim
        
        # Linear attention
        self.attn = LinearAttention(embed_dim, num_heads, dropout, kernel_fn)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V) or (B, T, C)
            
        Returns:
            Output tensor (same shape as input)
        """
        # Determine format
        skeleton_format = x.dim() == 4
        
        if skeleton_format:
            # (B, C, T, V) -> (B, V, T, C) for LayerNorm
            B, C, T, V = x.shape
            x = x.permute(0, 3, 2, 1).contiguous()  # (B, V, T, C)
            x = x.view(B * V, T, C)  # (BV, T, C)
        
        # Attention with residual
        x_norm = self.norm1(x)
        if skeleton_format:
            x_norm = x_norm.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()  # (B, C, T, V)
        attn_out = self.attn(x_norm)
        if skeleton_format:
            attn_out = attn_out.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)  # (BV, T, C)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        # Reshape back if needed
        if skeleton_format:
            x = x.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()  # (B, C, T, V)
        
        return x


if __name__ == '__main__':
    # Test Linear Attention
    print("Testing Linear Attention...")
    
    # Test 1: Basic forward pass with skeleton format
    print("\nTest 1: Skeleton format (B, C, T, V)")
    attn = LinearAttention(embed_dim=128, num_heads=8, dropout=0.1)
    x = torch.randn(2, 128, 50, 25)  # B=2, C=128, T=50, V=25
    out = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input!"
    print("✓ Shape test passed")
    
    # Test 2: Sequence format
    print("\nTest 2: Sequence format (B, T, C)")
    x_seq = torch.randn(4, 100, 128)  # B=4, T=100, C=128
    out_seq = attn(x_seq)
    assert out_seq.shape == x_seq.shape
    print(f"Input shape: {x_seq.shape}")
    print(f"Output shape: {out_seq.shape}")
    print("✓ Sequence format test passed")
    
    # Test 3: Kernel function
    print("\nTest 3: Kernel function non-negativity")
    test_input = torch.randn(10, 20)
    phi_elu = attn._kernel_function(test_input)
    assert (phi_elu >= 0).all(), "ELU+1 kernel should be non-negative!"
    print(f"Min value: {phi_elu.min().item():.6f}")
    print(f"Max value: {phi_elu.max().item():.6f}")
    print("✓ Kernel function test passed")
    
    # Test 4: Different sequence lengths (scalability)
    print("\nTest 4: Scalability with different sequence lengths")
    for T in [30, 60, 120, 300]:
        x_test = torch.randn(2, 128, T, 25)
        out_test = attn(x_test)
        assert out_test.shape == x_test.shape
        print(f"T={T:3d}: ✓")
    print("✓ Scalability test passed")
    
    # Test 5: Linear Attention Block
    print("\nTest 5: Linear Attention Block with FFN")
    attn_block = LinearAttentionBlock(embed_dim=128, num_heads=8)
    x = torch.randn(2, 128, 50, 25)
    out = attn_block(x)
    assert out.shape == x.shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Attention block test passed")
    
    # Test 6: Parameter count
    print("\nTest 6: Parameter count")
    num_params = sum(p.numel() for p in attn.parameters())
    print(f"Linear Attention parameters: {num_params:,}")
    num_params_block = sum(p.numel() for p in attn_block.parameters())
    print(f"Linear Attention Block parameters: {num_params_block:,}")
    print("✓ Parameter count test passed")
    
    print("\n" + "="*60)
    print("✓ ALL LINEAR ATTENTION TESTS PASSED!")
    print("="*60)
