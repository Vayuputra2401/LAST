import sys
import os
import torch
import torch.nn as nn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.last_v2 import LAST_v2, MODEL_VARIANTS

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_last_v2():
    print("="*60)
    print("TESTING LAST v2 ARCHITECTURE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Test Variants & Parameters
    print("\n[Parameter Counts]")
    for variant in ['small', 'base', 'large']:
        model = LAST_v2(num_classes=60, variant=variant).to(device)
        params = count_params(model)
        print(f"  {variant.upper()}: {params/1e6:.2f}M parameters")
        
        # Check against targets
        if variant == 'small':
            assert params < 1.0e6, "Small model should be < 1M params"
            # Our target was ~0.35M with sharing, but that's "effective" params.
            # The actual model instance has 1 backbone. MIB sharing reuse the same instance.
            # So the physical parameter count is 1 backbone.
            
    # 2. Test Forward Pass (Single Stream)
    print("\n[Forward Pass - Single Stream]")
    model = LAST_v2(num_classes=60, variant='small').to(device)
    model.eval()
    
    # Input: (N, C, T, V, M)
    N, C, T, V, M = 2, 3, 64, 25, 2
    x = torch.randn(N, C, T, V, M).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (N, 60)
    print("  ✓ Success")
    
    # 3. Test Forward Pass (MIB Dict)
    print("\n[Forward Pass - MIB (Multi-Stream)]")
    # Dictionary of streams
    x_mib = {
        'joint': torch.randn(N, C, T, V, M).to(device),
        'velocity': torch.randn(N, C, T, V, M).to(device),
        'bone': torch.randn(N, C, T, V, M).to(device)
    }
    
    with torch.no_grad():
        out_mib = model(x_mib)
        
    print(f"  Input: Dict with 3 streams of {x.shape}")
    print(f"  Output: {out_mib.shape}")
    assert out_mib.shape == (N, 60)
    print("  ✓ Success")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_last_v2()
