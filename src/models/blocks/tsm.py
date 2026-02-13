"""
Temporal Shift Module (TSM)

Zero-parameter temporal modeling by shifting feature channels along time dimension.
This enables temporal information exchange without any FLOPs or parameters.

Reference: Lin et al. "TSM: Temporal Shift Module for Efficient Video Understanding" (ICCV 2019)
Applied to skeleton-based action recognition in LAST framework.
"""

import torch
import torch.nn as nn


class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module for efficient temporal modeling.
    
    Shifts a portion of channels forward/backward in time:
    - shift_ratio/2 channels → shift forward (t → t+1)
    - shift_ratio/2 channels → shift backward (t → t-1)
    - remaining channels → no shift
    
    This creates temporal receptive field WITHOUT any parameters or FLOPs!
    
    Args:
        num_channels: Number of input/output channels
        shift_ratio: Fraction of channels to shift (default: 0.125 = 1/8)
        
    Example:
        With 128 channels and shift_ratio=0.125:
        - 8 channels shift forward
        - 8 channels shift backward
        - 112 channels static
    """
    
    def __init__(self, num_channels: int, shift_ratio: float = 0.125):
        super().__init__()
        
        self.num_channels = num_channels
        self.shift_ratio = shift_ratio
        
        # Calculate number of channels for each shift direction
        self.num_shift = int(num_channels * shift_ratio // 2)  # Half for each direction
        self.num_forward = self.num_shift
        self.num_backward = self.num_shift
        self.num_static = num_channels - self.num_forward - self.num_backward
        
        assert self.num_forward + self.num_backward + self.num_static == num_channels, \
            f"Channel split error: {self.num_forward} + {self.num_backward} + {self.num_static} != {num_channels}"
    
    def forward(self, x):
        """
        Forward pass with temporal shifting.
        
        Args:
            x: Input tensor (B, C, T, V)
               B = batch size
               C = channels
               T = time frames
               V = spatial dimension (joints)
               
        Returns:
            Shifted tensor (B, C, T, V) - same shape!
        """
        B, C, T, V = x.shape
        
        # Split channels into three groups
        x_forward = x[:, :self.num_forward]                                    # First group
        x_backward = x[:, self.num_forward:self.num_forward + self.num_backward]  # Second group
        x_static = x[:, self.num_forward + self.num_backward:]                    # Remaining
        
        # Create shifted tensors
        # Forward shift: t → t+1 (pad at beginning)
        x_forward_shifted = torch.zeros_like(x_forward)
        x_forward_shifted[:, :, 1:, :] = x_forward[:, :, :-1, :]  # Shift right
        x_forward_shifted[:, :, 0, :] = x_forward[:, :, 0, :]     # Duplicate first frame
        
        # Backward shift: t → t-1 (pad at end)
        x_backward_shifted = torch.zeros_like(x_backward)
        x_backward_shifted[:, :, :-1, :] = x_backward[:, :, 1:, :]  # Shift left
        x_backward_shifted[:, :, -1, :] = x_backward[:, :, -1, :]   # Duplicate last frame
        
        # Concatenate back
        out = torch.cat([x_forward_shifted, x_backward_shifted, x_static], dim=1)
        
        return out
    
    def extra_repr(self):
        """String representation for printing."""
        return f'num_channels={self.num_channels}, shift_ratio={self.shift_ratio}, ' \
               f'forward={self.num_forward}, backward={self.num_backward}, static={self.num_static}'


class TSMBlock(nn.Module):
    """
    TSM Block with optional residual connection.
    
    This is a drop-in replacement for temporal convolution but with 0 parameters/FLOPs.
    Can be inserted before/after spatial convolutions in the network.
    
    Args:
        num_channels: Number of channels
        shift_ratio: Fraction of channels to shift (default: 0.125)
        use_residual: Whether to add residual connection (default: True)
    """
    
    def __init__(self, num_channels: int, shift_ratio: float = 0.125, use_residual: bool = True):
        super().__init__()
        
        self.shift = TemporalShiftModule(num_channels, shift_ratio)
        self.use_residual = use_residual
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V)
            
        Returns:
            Output tensor (B, C, T, V)
        """
        out = self.shift(x)
        
        if self.use_residual:
            out = out + x  # Residual connection
        
        return out


if __name__ == '__main__':
    # Test TSM module
    print("Testing Temporal Shift Module...")
    
    # Test 1: Basic forward pass
    print("\nTest 1: Basic forward pass")
    tsm = TemporalShiftModule(num_channels=128, shift_ratio=0.125)
    x = torch.randn(2, 128, 50, 25)  # B=2, C=128, T=50, V=25
    out = tsm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input!"
    print("✓ Shape test passed")
    
    # Test 2: Zero parameters
    print("\nTest 2: Zero parameters check")
    num_params = sum(p.numel() for p in tsm.parameters())
    print(f"Number of parameters: {num_params}")
    assert num_params == 0, "TSM should have ZERO parameters!"
    print("✓ Zero parameter test passed")
    
    # Test 3: Channel splitting
    print("\nTest 3: Channel splitting")
    print(f"Total channels: {tsm.num_channels}")
    print(f"Forward shift: {tsm.num_forward} channels")
    print(f"Backward shift: {tsm.num_backward} channels")
    print(f"Static: {tsm.num_static} channels")
    assert tsm.num_forward + tsm.num_backward + tsm.num_static == 128
    print("✓ Channel split test passed")
    
    # Test 4: Shifting verification
    print("\nTest 4: Shifting verification")
    # Create a distinctive pattern
    x_test = torch.zeros(1, 16, 10, 1)
    # Mark specific frames in forward/backward channels
    x_test[0, 0, 5, 0] = 1.0  # Forward channel, frame 5
    x_test[0, 1, 5, 0] = 2.0  # Backward channel, frame 5
    
    tsm_test = TemporalShiftModule(num_channels=16, shift_ratio=0.125)
    out_test = tsm_test(x_test)
    
    # Forward shift: frame 5 should appear at frame 6
    print(f"Forward channel - Original frame 5: {x_test[0, 0, 5, 0].item()}")
    print(f"Forward channel - Shifted frame 6: {out_test[0, 0, 6, 0].item()}")
    assert abs(out_test[0, 0, 6, 0].item() - 1.0) < 1e-6, "Forward shift failed!"
    
    # Backward shift: frame 5 should appear at frame 4
    print(f"Backward channel - Original frame 5: {x_test[0, 1, 5, 0].item()}")
    print(f"Backward channel - Shifted frame 4: {out_test[0, 1, 4, 0].item()}")
    assert abs(out_test[0, 1, 4, 0].item() - 2.0) < 1e-6, "Backward shift failed!"
    print("✓ Shifting verification passed")
    
    # Test 5: TSM Block with residual
    print("\nTest 5: TSM Block with residual")
    tsm_block = TSMBlock(num_channels=128, shift_ratio=0.125, use_residual=True)
    x = torch.randn(2, 128, 50, 25)
    out = tsm_block(x)
    assert out.shape == x.shape
    # With residual, output should be different from input but same shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ TSM Block test passed")
    
    print("\n" + "="*60)
    print("✓ ALL TSM TESTS PASSED!")
    print("="*60)
