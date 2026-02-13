"""
Main data loading script - Config-driven implementation.

Uses YAML configs for environment and dataset parameters.

Usage:
    python scripts/load_data.py --env local --split train
    python scripts/load_data.py --env kaggle --split val
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.dataset import SkeletonDataset


def create_dataloader(config: dict, split: str = 'train') -> DataLoader:
    """
    Create PyTorch DataLoader from config.
    
    Args:
        config: Loaded configuration dict
        split: 'train' or 'val'
        
    Returns:
        DataLoader instance
    """
    # Extract configs
    env_config = config['environment']
    data_config = config['data']
    
    # Get paths
    data_path = env_config['paths']['data_root']
    
    # Dataset parameters
    dataset_params = data_config['dataset']
    
    # Create dataset
    dataset = SkeletonDataset(
        data_path=data_path,
        data_type=dataset_params['data_type'],
        max_frames=dataset_params['max_frames'],
        num_joints=dataset_params['num_joints'],
        max_bodies=dataset_params['max_bodies'],
        split=split,
        split_type=dataset_params['split_type']
    )
    
    # DataLoader parameters
    dataloader_params = data_config['dataloader']
    hardware_params = env_config['hardware']
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_params['batch_size'],
        shuffle=dataloader_params['shuffle'] if split == 'train' else False,
        num_workers=hardware_params['num_workers'],
        pin_memory=hardware_params['pin_memory'],
        drop_last=dataloader_params['drop_last'] if split == 'train' else False
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Load skeleton data using configs')
    parser.add_argument('--env', type=str, default=None,
                       help='Environment: local, kaggle (auto-detect if not specified)')
    parser.add_argument('--dataset', type=str, default='ntu120',
                       help='Dataset config name')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val'],
                       help='Dataset split')
    parser.add_argument('--show_samples', type=int, default=3,
                       help='Number of sample batches to display')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LAST - Config-Driven Data Loading")
    print("="*60)
    
    # Load config
    print(f"\nLoading configuration...")
    config = load_config(env=args.env, dataset=args.dataset)
    
    env_name = config['environment']['environment']['name']
    print(f"Environment: {env_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    
    # Show config details
    print(f"\nData path: {config['environment']['paths']['data_root']}")
    print(f"Output path: {config['environment']['paths']['output_root']}")
    print(f"Device: {config['environment']['hardware']['device']}")
    print(f"Batch size: {config['data']['dataloader']['batch_size']}")
    
    # Create dataloader
    print(f"\nCreating DataLoader...")
    dataloader = create_dataloader(config, split=args.split)
    
    print(f"✓ DataLoader created")
    print(f"  Total samples: {len(dataloader.dataset)}")
    print(f"  Total batches: {len(dataloader)}")
    
    # Load and display sample batches
    print(f"\nLoading {args.show_samples} sample batches...")
    
    for i, (data, labels) in enumerate(dataloader):
        if i >= args.show_samples:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Data shape: {data.shape} (B, C, T, V, M)")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Labels: {labels.tolist()[:5]}..." if len(labels) > 5 else f"  Labels: {labels.tolist()}")
    
    print("\n" + "="*60)
    print("✅ Data loading successful!")
    print("="*60)
    
    # Summary
    print(f"\nConfig-driven architecture:")
    print(f"  ✓ Environment config: configs/environment/{env_name}.yaml")
    print(f"  ✓ Data config: configs/data/{args.dataset}.yaml")
    print(f"  ✓ Auto-detection: {'Enabled' if args.env is None else 'Disabled'}")
    print(f"\nReady for training pipeline!")


if __name__ == '__main__':
    main()
