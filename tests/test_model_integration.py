"""
Integration test for LAST model with real skeleton data.

This script tests the complete pipeline:
1. Load real skeleton data from NTU RGB+D dataset
2. Create LAST model
3. Forward pass
4. Verify output shapes and values
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np

from src.utils.config import load_config
from src.data.dataset import SkeletonDataset
from src.models import create_last_base, create_last_small, create_last_large


def test_model_with_real_data():
    """Test LAST model with actual NTU RGB+D skeleton data."""
    
    print("="*60)
    print("LAST MODEL INTEGRATION TEST")
    print("="*60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    try:
        config = load_config(env='local', dataset='ntu120')
        print(f"✓ Config loaded: {config['environment']['environment']['name']}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Create dataset
    print("\n2. Loading dataset...")
    try:
        dataset = SkeletonDataset(
            data_path=config['environment']['paths']['data_root'],
            data_type='skeleton',
            split='train',
            split_type='xsub',
            max_frames=300,
            split_config=config['data']['dataset']['splits']
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # Load one sample
    print("\n3. Loading sample skeleton...")
    try:
        data, label = dataset[0]
        print(f"✓ Sample loaded")
        print(f"  Shape: {data.shape}")
        print(f"  Label: {label} (class {label})")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        return False
    
    # Test all model variants
    variants = {
        'LAST-Small': create_last_small,
        'LAST-Base': create_last_base,
        'LAST-Large': create_last_large
    }
    
    for variant_name, create_fn in variants.items():
        print(f"\n4. Testing {variant_name}...")
        try:
            # Create model
            model = create_fn(num_classes=120, num_joints=25)
            model.eval()
            
            # Count parameters
            num_params = model.count_parameters()
            print(f"✓ Model created: {num_params:,} parameters")
            
            # Prepare input
            batch_data = data.unsqueeze(0)  # Add batch dimension
            print(f"  Input shape: {batch_data.shape}")
            
            # Forward pass
            with torch.no_grad():
                logits = model(batch_data)
            
            print(f"  Output shape: {logits.shape}")
            print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
            
            # Verify output
            assert logits.shape == (1, 120), f"Expected (1, 120), got {logits.shape}"
            assert not torch.isnan(logits).any(), "Found NaN in output"
            assert not torch.isinf(logits).any(), "Found Inf in output"
            
            # Get prediction
            pred_class = logits.argmax(dim=1).item()
            print(f"  Predicted class: {pred_class}")
            print(f"  Ground truth: {label}")
            
            print(f"✓ {variant_name} test PASSED")
            
        except Exception as e:
            print(f"✗ {variant_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test multi-body input
    print("\n5. Testing multi-body input...")
    try:
        model = create_last_base()
        model.eval()
        
        # Create multi-body input
        multi_body_data = torch.randn(2, 3, 100, 25, 2)  # B=2, C=3, T=100, V=25, M=2
        print(f"  Multi-body input: {multi_body_data.shape}")
        
        with torch.no_grad():
            logits = model(multi_body_data)
        
        print(f"  Output shape: {logits.shape}")
        assert logits.shape == (2, 120)
        print("✓ Multi-body test PASSED")
        
    except Exception as e:
        print(f"✗ Multi-body test failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*60)
    print("\nModel is ready for training!")
    print(f"  - Variants tested: {len(variants)}")
    print(f"  - Real data tested: ✓")
    print(f"  - Multi-body tested: ✓")
    print(f"  - Output verified: ✓")
    
    return True


if __name__ == '__main__':
    success = test_model_with_real_data()
    sys.exit(0 if success else 1)
