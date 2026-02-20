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
        self._split_config = split_config
        
        # MIB Support
        # If 'mib', we load 3 streams: joint, velocity, bone
        self.streams = None
        
        # Initialize parser
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
        elif self.data_type == 'mib':
            self._load_mib_files()
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
        
        # Load labels - handle both formats:
        # Format 1 (our preprocess_data.py): plain list [0, 1, 2, ...]
        # Format 2 (CTR-GCN/standard):      tuple (sample_names, labels)
        with open(label_file, 'rb') as f:
            raw_labels = pickle.load(f)
        
        if isinstance(raw_labels, tuple) or isinstance(raw_labels, list) and len(raw_labels) == 2 and isinstance(raw_labels[0], (list, np.ndarray)):
            # Tuple format: (sample_names, labels)
            if isinstance(raw_labels, tuple):
                sample_names, label_values = raw_labels
            else:
                sample_names, label_values = raw_labels[0], raw_labels[1]
            self.labels = list(label_values)
            print(f"  [{self.split}] Label format: tuple (sample_names, labels)")
        else:
            # Plain list format
            self.labels = list(raw_labels)
            print(f"  [{self.split}] Label format: plain list")
        
        # Diagnostic prints
        print(f"  [{self.split}] Data shape: {self.data.shape}")
        print(f"  [{self.split}] Num labels: {len(self.labels)}")
        print(f"  [{self.split}] Label range: [{min(self.labels)}, {max(self.labels)}]")
        print(f"  [{self.split}] Label dtype: {type(self.labels[0])}")
        print(f"  [{self.split}] First 10 labels: {self.labels[:10]}")
        
        # Sanity check: data and labels must align
        assert len(self.labels) == self.data.shape[0], \
            f"Data-label mismatch: {self.data.shape[0]} samples but {len(self.labels)} labels"
        
        # Check a sample's stats
        sample0 = np.array(self.data[0])
        print(f"  [{self.split}] Sample[0] stats: mean={sample0.mean():.4f}, std={sample0.std():.4f}, "
              f"min={sample0.min():.4f}, max={sample0.max():.4f}, zeros={np.sum(sample0 == 0)/sample0.size*100:.1f}%")
        
        # Filter out corrupted/empty skeletons (302 known bad samples in NTU RGB+D)
        valid_indices = []
        num_filtered = 0
        for i in range(len(self.labels)):
            # Check if sample is all zeros (missing skeleton data)
            sample_sum = np.abs(self.data[i]).sum()
            if sample_sum > 0:
                valid_indices.append(i)
            else:
                num_filtered += 1
        
        if num_filtered > 0:
            print(f"  Filtered {num_filtered} empty/corrupted samples from {self.split} split")
        
        # Remap to valid indices only
        self._valid_indices = valid_indices
        self.labels = [self.labels[i] for i in valid_indices]
        self.samples = list(range(len(self.labels)))

    def _load_mib_files(self):
        """Load 3-stream .npy files."""
        # Expected: split_joint.npy, split_velocity.npy, split_bone.npy
        import pickle
        
        # Folder for v2 processed data
        # data_path/xsub/
        # Check if folder exists, if not maybe it's in data_path directly
        
        self.streams = {}
        stream_names = ['joint', 'velocity', 'bone']
        
        # Load Labels first (from joint stream logic usually or shared label file)
        label_file = os.path.join(self.data_path, self.split_type, f'{self.split}_label.pkl')
        if not os.path.exists(label_file):
             # Try without split_type subfolder (if user put them together)
             label_file = os.path.join(self.data_path, f'{self.split}_label.pkl')
             
        if not os.path.exists(label_file):
             raise FileNotFoundError(f"Label file not found: {label_file}")
             
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)
            if isinstance(self.labels, tuple): self.labels = list(self.labels[1])
            
        print(f"  [{self.split}] Found {len(self.labels)} labels")

        # Load Streams
        for s in stream_names:
            fpath = os.path.join(self.data_path, self.split_type, f'{self.split}_{s}.npy')
            if not os.path.exists(fpath):
                 # Try without subfolder
                 fpath = os.path.join(self.data_path, f'{self.split}_{s}.npy')
            
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Stream file not found: {fpath}")
                
            print(f"  [{self.split}] Loading {s} stream...")
            self.streams[s] = np.load(fpath, mmap_mode='r')
            
        # Basic validation
        N = len(self.labels)
        for s in stream_names:
            assert self.streams[s].shape[0] == N, f"Stream {s} has {self.streams[s].shape[0]} samples, labels have {N}"
            
        # Filter empty samples (check Joint stream)
        valid_indices = []
        for i in range(N):
            if np.abs(self.streams['joint'][i]).sum() > 0:
                valid_indices.append(i)
                
        self._valid_indices = valid_indices
        self.labels = [self.labels[i] for i in valid_indices]
        self.samples = list(range(len(self.labels)))
        print(f"  [{self.split}] Valid samples: {len(self.labels)} (Filtered {N - len(self.labels)})")
    
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
                # FIX (Bug High): Fallback now uses correct NTU RGB+D 60 subject IDs.
                # NTU-60 has 40 subjects (IDs 1-40). The previous fallback included
                # IDs up to 106 (NTU-120 split), which would contaminate the val set
                # with training subjects when split_config is not provided.
                # Official NTU-60 cross-subject split (from Liu et al. CVPR 2016):
                train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
                val_subjects = {3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40}
            
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
        
        if self.data_type == 'mib':
            # Data is dict: {'joint': ..., 'velocity': ..., 'bone': ...}
            # Map valid idx
            data_idx = self._valid_indices[idx] if hasattr(self, '_valid_indices') else idx

            # FIX (Observation B): Collect ALL streams into a dict FIRST, then apply
            # the transform ONCE. MIBTransform samples one set of shared random
            # parameters (rotation angle, scale, temporal crop start, flip decision)
            # and applies them identically to every stream, ensuring geometric
            # consistency across joint/velocity/bone. Previously, self.transform(d)
            # was called once per stream in a loop, generating a different random
            # seed each time and producing physically inconsistent augmentations.
            out_data = {
                name: torch.from_numpy(np.array(stream_data[data_idx])).float()
                for name, stream_data in self.streams.items()
            }
            if self.transform is not None:
                out_data = self.transform(out_data)  # MIBTransform(dict) -> dict

            return out_data, label
            
        elif self.data_type == 'npy':
            # Map through valid indices (filtered during init)
            data_idx = self._valid_indices[idx] if hasattr(self, '_valid_indices') else idx
            data = self.data[data_idx]  # Already shape (C, T, V, M)
            
            # Convert to tensor
            data = torch.from_numpy(np.array(data)).float()
            
            # Apply transforms if provided
            if self.transform is not None:
                data = self.transform(data)
                
            return data, label
            
        else:  # skeleton
            # Parse skeleton file
            file_path = self.samples[idx]
            skeleton_data, _ = self.parser.parse_file(file_path)
            # skeleton_data shape: (T, V, C, M)
            
            # Pad or sample to max_frames
            skeleton_data = self._temporal_transform(skeleton_data)
            
            # Transpose to (C, T, V, M)
            data = skeleton_data.transpose(2, 0, 1, 3)
            data = torch.from_numpy(np.array(data)).float()
            
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
