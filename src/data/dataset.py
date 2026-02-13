"""
Skeleton Dataset for PyTorch DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Callable
import os
import glob

from .skeleton_loader import SkeletonFileParser
from .ntu120_actions import get_action_name, validate_label


class SkeletonDataset(Dataset):
    """
    PyTorch Dataset for skeleton-based action recognition.
    
    Supports both .skeleton files (raw) and .npy files (preprocessed).
    """
    
    def __init__(
        self,
        data_path: str,
        data_type: str = 'skeleton',  # 'skeleton' or 'npy'
        max_frames: int = 300,
        num_joints: int = 25,
        max_bodies: int = 2,
        transform: Optional[Callable] = None,
        split: str = 'train',  # 'train' or 'val'
        split_type: str = 'xsub',  # 'xsub' or 'xset'
        split_config: Optional[dict] = None,  # Split configuration from YAML
    ):
        """
        Args:
            data_path: Path to data directory
            data_type: 'skeleton' for raw .skeleton files, 'npy' for preprocessed
            max_frames: Maximum sequence length
            num_joints: Number of joints (25 for NTU RGB+D)
            max_bodies: Maximum bodies per frame
            transform: Optional transform to apply
            split: 'train' or 'val'
            split_type: 'xsub' (cross-subject) or 'xset' (cross-setup)
            split_config: Optional dict with split configuration from YAML
        """
        self.data_path = data_path
        self.data_type = data_type
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.max_bodies = max_bodies
        self.transform = transform
        self.split = split
        self.split_type = split_type
        self._split_config = split_config  # Store for _should_include_sample
        
        # Initialize parser for .skeleton files
        if data_type == 'skeleton':
            self.parser = SkeletonFileParser(num_joints=num_joints, max_bodies=max_bodies)
        
        # Load data
        self.samples = []
        self.labels = []
        self._load_data()
        
        print(f"Loaded {len(self.samples)} samples for {split} split ({split_type})")
    
    def _load_data(self):
        """Load dataset based on data_type."""
        if self.data_type == 'skeleton':
            self._load_skeleton_files()
        elif self.data_type == 'npy':
            self._load_npy_files()
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")
    
    def _load_skeleton_files(self):
        """Load raw .skeleton files."""
        # Find all skeleton files
        skeleton_files = glob.glob(os.path.join(self.data_path, '*.skeleton'))
        
        if len(skeleton_files) == 0:
            # Try nested directory
            skeleton_files = glob.glob(os.path.join(self.data_path, '**', '*.skeleton'), recursive=True)
        
        if len(skeleton_files) == 0:
            raise FileNotFoundError(f"No .skeleton files found in {self.data_path}")
        
        # Filter based on split
        for file_path in skeleton_files:
            metadata = self.parser.extract_metadata_from_filename(file_path)
            
            if self._should_include_sample(metadata):
                self.samples.append(file_path)
                self.labels.append(metadata['action'])
    
    def _load_npy_files(self):
        """Load preprocessed .npy files."""
        # Expected structure: data_path/xsub/train_data.npy, train_label.pkl
        import pickle
        
        data_file = os.path.join(self.data_path, self.split_type, f'{self.split}_data.npy')
        label_file = os.path.join(self.data_path, self.split_type, f'{self.split}_label.pkl')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data (memory-mapped for efficiency)
        self.data = np.load(data_file, mmap_mode='r')
        
        # Load labels
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)
        
        self.samples = list(range(len(self.labels)))
    
    def _should_include_sample(self, metadata: dict) -> bool:
        """Determine if sample should be included based on split."""
        # Load from config if available (passed during init)
        if hasattr(self, '_split_config'):
            split_config = self._split_config
        else:
            # Fallback to hardcoded (for backward compatibility)
            split_config = None
        
        if self.split_type == 'xsub':
            # Cross-subject split
            if split_config and 'xsub' in split_config:
                train_subjects = set(split_config['xsub']['train_subjects'])
                val_subjects = set(split_config['xsub']['val_subjects'])
            else:
                # Fallback hardcoded
                train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                                45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81,
                                82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103}
                val_subjects = {3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40,
                              41, 42, 43, 44, 48, 51, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73,
                              75, 76, 77, 79, 87, 88, 90, 96, 99, 101, 102, 104, 105, 106}
            
            person = metadata['person']
            if self.split == 'train':
                return person in train_subjects
            elif self.split == 'val':
                return person in val_subjects
        
        elif self.split_type == 'xset':
            # Cross-setup split
            if split_config and 'xset' in split_config:
                train_setups = set(split_config['xset']['train_setups'])
                val_setups = set(split_config['xset']['val_setups'])
            else:
                # Fallback: even setups for train, odd for val
                train_setups = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
                val_setups = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
            
            setup = metadata['setup']
            if self.split == 'train':
                return setup in train_setups
            elif self.split == 'val':
                return setup in val_setups
        
        return False
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """
        Get a single sample.
        
        Returns:
            data: Tensor of shape (C, T, V, M)
            label: Action class label
        """
        label = self.labels[idx]
        
        if self.data_type == 'skeleton':
            # Parse skeleton file
            file_path = self.samples[idx]
            skeleton_data, _ = self.parser.parse_file(file_path)
            # skeleton_data shape: (T, V, C, M)
            
            # Pad or sample to max_frames
            skeleton_data = self._temporal_transform(skeleton_data)
            
            # Transpose to (C, T, V, M)
            data = skeleton_data.transpose(2, 0, 1, 3)
            
        else:  # npy
            # Load from memory-mapped array
            data = self.data[idx]  # Already shape (C, T, V, M)
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform is not None:
            data = self.transform(data)
        
        return data, label
    
    def _temporal_transform(self, skeleton_data: np.ndarray) -> np.ndarray:
        """
        Pad or sample sequence to max_frames.
        
        Args:
            skeleton_data: Shape (T, V, C, M)
            
        Returns:
            transformed_data: Shape (max_frames, V, C, M)
        """
        T, V, C, M = skeleton_data.shape
        
        if T == self.max_frames:
            return skeleton_data
        
        elif T < self.max_frames:
            # Zero-pad
            padded = np.zeros((self.max_frames, V, C, M), dtype=np.float32)
            padded[:T] = skeleton_data
            return padded
        
        else:
            # Uniformly sample
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            return skeleton_data[indices]
    
    def get_sample_info(self, idx: int) -> dict:
        """Get metadata about a sample."""
        label = self.labels[idx]
        
        info = {
            'label': label,
            'action_name': get_action_name(label)
        }
        
        if self.data_type == 'skeleton':
            file_path = self.samples[idx]
            metadata = self.parser.extract_metadata_from_filename(file_path)
            info.update(metadata)
            info['file_path'] = file_path
        else:
            info['index'] = idx
            
        return info
