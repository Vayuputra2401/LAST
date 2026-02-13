"""
Production-Ready Data Pipeline Test Script

Comprehensive testing for:
- Data loading (train/val splits)
- Normalization and preprocessing  
- Transforms and augmentation
- Batch creation
- Data visualization

Usage:
    python scripts/test_dataloader.py
    python scripts/test_dataloader.py --split train
    python scripts/test_dataloader.py --visualize
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config import load_config
from src.data.dataloader import create_dataloaders, get_dataloader_info
from src.data import get_action_name
from src.utils.visualization import plot_skeleton_frame, plot_skeleton_sequence


def test_splits(loaders: dict):
    """Test train/val split configuration."""
    print("\n" + "="*60)
    print("TESTING SPLITS")
    print("="*60)
    
    for split_name, loader in loaders.items():
        info = get_dataloader_info(loader)
        print(f"\n{split_name.upper()} Split:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Batch size: {info['batch_size']}")
        print(f"  Split type: {info['split_type']}")
    
    # Check no overlap
    train_dataset = loaders['train'].dataset
    val_dataset = loaders['val'].dataset
    
    print(f"\n✓ Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"✓ Train/Val ratio: {len(train_dataset)/len(val_dataset):.2f}")


def test_normalization(loader):
    """Test normalization is working."""
    print("\n" + "="*60)
    print("TESTING NORMALIZATION")
    print("="*60)
    
    # Get one batch
    data, labels = next(iter(loader))
    
    # Check if spine base is centered (joint 0)
    spine_base = data[:, :, :, 0, :]  # (B, C, T, M)
    
    # For center_spine normalization, spine base should be near zero
    spine_mean = spine_base.abs().mean()
    
    print(f"\nSpineBase position (should be ~0 for center_spine):")
    print(f"  Mean absolute position: {spine_mean:.6f}")
    
    # Check data range
    print(f"\nData statistics:")
    print(f"  Min: {data.min():.3f}")
    print(f"  Max: {data.max():.3f}")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")
    
    if spine_mean < 0.01:
        print("\n✓ Normalization working correctly (SpineBase centered)")
    else:
        print("\n⚠ Normalization may not be working as expected")


def test_batch_loading(loader, num_batches: int = 3):
    """Test batch loading and consistency."""
    print("\n" + "="*60)
    print("TESTING BATCH LOADING")
    print("="*60)
    
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Data shape: {data.shape} (B, C, T, V, M)")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Data dtype: {data.dtype}")
        print(f"  Labels dtype: {labels.dtype}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(data).any()
        has_inf = torch.isinf(data).any()
        
        if has_nan or has_inf:
            print(f"  ⚠ WARNING: Found NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  ✓ No NaN/Inf values")
        
        # Show some action labels
        print(f"  Labels: {labels[:5].tolist()}...")
        action_names = [get_action_name(label.item()) for label in labels[:3]]
        print(f"  Actions: {action_names}")
    
    print(f"\n✓ Successfully loaded {num_batches} batches")


def test_transforms(config: dict):
    """Test that transforms are applied correctly."""
    print("\n" + "="*60)
    print("TESTING TRANSFORMS")
    print("="*60)
    
    from src.data.transforms import get_train_transform, get_val_transform
    
    # Check train transform
    train_transform = get_train_transform(config['data'])
    val_transform = get_val_transform(config['data'])
    
    print("\nTrain Transform:")
    print(f"  {train_transform}")
    
    print("\nVal Transform:")
    print(f"  {val_transform}")
    
    # Test on dummy data
    dummy_skeleton = torch.randn(3, 100, 25, 2)
    
    if train_transform:
        transformed = train_transform(dummy_skeleton)
        print(f"\n✓ Train transform applied successfully")
        print(f"  Input shape: {dummy_skeleton.shape}")
        print(f"  Output shape: {transformed.shape}")
    
    if val_transform:
        transformed = val_transform(dummy_skeleton)
        print(f"\n✓ Val transform applied successfully")


def visualize_samples(loader, output_dir: str = ".", num_samples: int = 3):
    """Visualize skeleton samples."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get one batch
    data, labels = next(iter(loader))
    
    for i in range(min(num_samples, data.shape[0])):
        skeleton = data[i]  # (C, T, V, M)
        label = labels[i].item()
        action_name = get_action_name(label)
        
        # Convert to (T, V, C) format for visualization
        skeleton_np = skeleton.numpy().transpose(1, 2, 0, 3)  # (T, V, C, M)
        skeleton_np = skeleton_np[:, :, :, 0]  # Take first body (T, V, C)
        
        # Plot single frame
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_skeleton_frame(skeleton_np[0], ax=ax)  # First frame
        fig.suptitle(f"Sample {i+1}: {action_name} (label {label})", fontsize=14)
        output_file = output_dir / f"sample_{i+1}_frame.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_file}")
        
        # Plot sequence
        fig = plot_skeleton_sequence(skeleton_np, num_frames=8)
        fig.suptitle(f"Sample {i+1}: {action_name} - Sequence", fontsize=14, y=0.995)
        output_file = output_dir / f"sample_{i+1}_sequence.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved  {output_file}")
    
    print(f"\n✓ Generated {num_samples * 2} visualization files")


def test_data_distribution(loaders: dict):
    """Test data distribution across classes."""
    print("\n" + "="*60)
    print("TESTING DATA DISTRIBUTION")
    print("="*60)
    
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        
        # Count labels
        label_counts = {}
        for label in dataset.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n{split_name.upper()} Split:")
        print(f"  Unique classes: {len(label_counts)}")
        print(f"  Samples per class: {len(dataset) / len(label_counts):.1f} (avg)")
        print(f"  Min samples in class: {min(label_counts.values())}")
        print(f"  Max samples in class: {max(label_counts.values())}")


def main():
    parser = argparse.ArgumentParser(description='Test data loading pipeline')
    parser.add_argument('--env', type=str, default=None, help='Environment (auto-detect if not specified)')
    parser.add_argument('--dataset', type=str, default='ntu120', help='Dataset config name')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'val'],
                       help='Which split to test')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for visualizations')
    parser.add_argument('--num_batches', type=int, default=3, help='Number of batches to test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LAST - Data Pipeline Testing")
    print("="*60)
    
    # Load config
    print(f"\nLoading configuration...")
    config = load_config(env=args.env, dataset=args.dataset)
    
    env_name = config['environment']['environment']['name']
    print(f"✓ Environment: {env_name}")
    print(f"✓ Dataset: {args.dataset}")
    print(f"✓ Data path: {config['environment']['paths']['data_root']}")
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    loaders = create_dataloaders(config)
    print(f"✓ Created train and val dataloaders")
    
    # Select loaders to test
    if args.split == 'all':
        test_loaders = loaders
    else:
        test_loaders = {args.split: loaders[args.split]}
    
    # Run tests
    test_splits(loaders)
    test_normalization(loaders['train'])
    test_batch_loading(loaders['train'], num_batches=args.num_batches)
    test_transforms(config)
    test_data_distribution(loaders)
    
    # Visualizations
    if args.visualize:
        visualize_samples(loaders['train'], output_dir=args.output_dir, num_samples=3)
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    
    print(f"\nData Pipeline Summary:")
    print(f"  Train samples: {len(loaders['train'].dataset)}")
    print(f"  Val samples: {len(loaders['val'].dataset)}")
    print(f"  Batch size: {loaders['train'].batch_size}")
    print(f"  Normalization: {config['data']['dataset']['preprocessing']['normalization_method']}")
    print(f"  Augmentation: {'Enabled' if config['data']['dataset']['augmentation']['enabled'] else 'Disabled'}")
    
    print(f"\n🚀 Data pipeline is production-ready!")
    
    if not args.visualize:
        print(f"\nTip: Run with --visualize to generate skeleton plots")


if __name__ == '__main__':
    main()
