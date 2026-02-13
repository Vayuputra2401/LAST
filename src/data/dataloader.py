"""
DataLoader Utilities

Unified functions for creating dataloaders with proper configuration.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict

from .dataset import SkeletonDataset
from .transforms import get_train_transform, get_val_transform


def create_dataloader(
    config: dict,
    split: str = 'train',
    transform=None
) -> DataLoader:
    """
    Create a single DataLoader for specified split.
    
    Args:
        config: Full configuration dict (from load_config)
        split: 'train' or 'val'
        transform: Optional transform override
        
    Returns:
        DataLoader instance
        
    Example:
        >>> config = load_config(env='local', dataset='ntu120')
        >>> train_loader = create_dataloader(config, split='train')
        >>> for data, labels in train_loader:
        ...     # Training loop
    """
    env_config = config['environment']
    data_config = config['data']
    
    # Get paths
    data_path = env_config['paths']['data_root']
    
    # Get dataset parameters
    dataset_params = data_config['dataset']
    
    # Get or create transform
    if transform is None:
        if split == 'train':
            transform = get_train_transform(data_config)
        else:
            transform = get_val_transform(data_config)
    
    # Create dataset
    dataset = SkeletonDataset(
        data_path=data_path,
        data_type=dataset_params['data_type'],
        max_frames=dataset_params['max_frames'],
        num_joints=dataset_params['num_joints'],
        max_bodies=dataset_params['max_bodies'],
        transform=transform,
        split=split,
        split_type=dataset_params['split_type'],
        split_config=dataset_params.get('splits')  # Pass splits config
    )
    
    # DataLoader parameters
    dataloader_params = data_config['dataloader']
    hardware_params = env_config['hardware']
    
    # Create DataLoader with split-specific settings
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_params['batch_size'],
        shuffle=dataloader_params['shuffle'] if split == 'train' else False,
        num_workers=hardware_params['num_workers'],
        pin_memory=hardware_params['pin_memory'],
        drop_last=dataloader_params['drop_last'] if split == 'train' else False
    )
    
    return dataloader


def create_dataloaders(config: dict) -> Dict[str, DataLoader]:
    """
    Create all dataloaders (train and val) at once.
    
    Args:
        config: Full configuration dict
        
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
        
    Example:
        >>> config = load_config(env='local', dataset='ntu120')
        >>> loaders = create_dataloaders(config)
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
        >>> print(f"Train: {len(train_loader.dataset)} samples")
        >>> print(f"Val: {len(val_loader.dataset)} samples")
    """
    loaders = {}
    
    for split in ['train', 'val']:
        loaders[split] = create_dataloader(config, split=split)
    
    return loaders


def get_dataloader_info(dataloader: DataLoader) -> dict:
    """
    Get information about a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Info dictionary
        
    Example:
        >>> info = get_dataloader_info(train_loader)
        >>> print(f"Batches: {info['num_batches']}, Batch size: {info['batch_size']}")
    """
    dataset = dataloader.dataset
    
    return {
        'num_samples': len(dataset),
        'num_batches': len(dataloader),
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'split': dataset.split if hasattr(dataset, 'split') else 'unknown',
        'split_type': dataset.split_type if hasattr(dataset, 'split_type') else 'unknown',
    }
