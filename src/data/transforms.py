"""
Data Transforms and Augmentation

Composable transforms for skeleton data augmentation during training.
"""

import numpy as np
import torch
from typing import Callable, List


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of transform functions/objects
        """
        self.transforms = transforms
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms in sequence.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Transformed skeleton
        """
        for transform in self.transforms:
            skeleton = transform(skeleton)
        return skeleton
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class Normalize:
    """Normalize skeleton data."""
    
    def __init__(self, method: str = 'center_spine', center_joint: int = 0, scale_by_torso: bool = True):
        """
        Args:
            method: 'center_spine', 'first_frame', or 'none'
            center_joint: Joint index to center on (for 'center_spine' method)
            scale_by_torso: Whether to scale by torso length
        """
        self.method = method
        self.center_joint = center_joint
        self.scale_by_torso = scale_by_torso
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Normalized skeleton
        """
        from .preprocessing import normalize_skeleton
        
        # Convert to numpy, normalize, convert back
        skeleton_np = skeleton.numpy() if isinstance(skeleton, torch.Tensor) else skeleton
        normalized = normalize_skeleton(
            skeleton_np,
            method=self.method,
            center_joint=self.center_joint,
            scale_by_torso=self.scale_by_torso
        )
        return torch.from_numpy(normalized).float() if isinstance(skeleton, torch.Tensor) else normalized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method='{self.method}', center_joint={self.center_joint})"


class RandomRotation:
    """Random rotation around Y-axis (vertical)."""
    
    def __init__(self, angle_range: tuple = (-15, 15)):
        """
        Args:
            angle_range: (min_deg, max_deg) rotation range
        """
        self.angle_range = angle_range
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Rotated skeleton
        """
        # Random angle in radians
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = np.deg2rad(angle)
        
        # Rotation matrix around Y-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        R = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=skeleton.dtype)
        
        # Apply rotation: skeleton (C, T, V, M) @ R -> (C, T, V, M)
        # Reshape to (C×T×V×M, 3) -> rotate -> reshape back
        C, T, V, M = skeleton.shape
        skeleton_flat = skeleton.permute(1, 2, 3, 0).reshape(-1, 3)  # (T×V×M, C)
        rotated = skeleton_flat @ R.T
        skeleton = rotated.reshape(T, V, M, C).permute(3, 0, 1, 2)  # (C, T, V, M)
        
        return skeleton
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle_range={self.angle_range})"


class RandomScale:
    """Random scaling."""
    
    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        """
        Args:
            scale_range: (min_scale, max_scale)
        """
        self.scale_range = scale_range
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Apply random scaling.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Scaled skeleton
        """
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return skeleton * scale
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"


class RandomShear:
    """Random shearing."""
    
    def __init__(self, shear_range: tuple = (-0.1, 0.1)):
        """
        Args:
            shear_range: (min_shear, max_shear)
        """
        self.shear_range = shear_range
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Apply random shearing.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Sheared skeleton
        """
        shear = np.random.uniform(self.shear_range[0], self.shear_range[1])
        
        # Shear matrix
        S = torch.tensor([
            [1, shear, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=skeleton.dtype)
        
        # Apply shear
        C, T, V, M = skeleton.shape
        skeleton_flat = skeleton.permute(1, 2, 3, 0).reshape(-1, 3)
        sheared = skeleton_flat @ S.T
        skeleton = sheared.reshape(T, V, M, C).permute(3, 0, 1, 2)
        
        return skeleton
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shear_range={self.shear_range})"


class GaussianNoise:
    """Add Gaussian noise."""
    
    def __init__(self, std: float = 0.001):
        """
        Args:
            std: Standard deviation of noise
        """
        self.std = std
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Noisy skeleton
        """
        noise = torch.randn_like(skeleton) * self.std
        return skeleton + noise
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class TemporalCrop:
    """Random temporal crop and resize to target number of frames."""
    
    def __init__(self, target_frames: int = 64):
        """
        Args:
            target_frames: Number of frames to crop/resize to
        """
        self.target_frames = target_frames
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Random temporal crop.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Cropped skeleton of shape (C, target_frames, V, M)
        """
        C, T, V, M = skeleton.shape
        
        if T <= self.target_frames:
            # Pad with zeros if too short
            padded = torch.zeros(C, self.target_frames, V, M, dtype=skeleton.dtype)
            padded[:, :T, :, :] = skeleton
            return padded
        
        # Random start point
        start = np.random.randint(0, T - self.target_frames)
        return skeleton[:, start:start + self.target_frames, :, :]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_frames={self.target_frames})"


class RandomTemporalFlip:
    """Reverse frame order with given probability."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Randomly reverse temporal order.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Possibly flipped skeleton
        """
        if np.random.random() < self.p:
            return skeleton.flip(dims=[1])  # Flip along T dimension
        return skeleton
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def get_train_transform(config: dict):
    """
    Create training transform pipeline from config.
    
    Args:
        config: Data configuration dict
        
    Returns:
        Compose transform
    """
    transforms = []
    
    # Always normalize (if not using preprocessed npy which is already normalized)
    preproc = config.get('dataset', {}).get('preprocessing', {})
    if preproc.get('normalize', False):
        transforms.append(Normalize(
            method=preproc.get('normalization_method', 'center_spine'),
            center_joint=preproc.get('center_joint', 0),
            scale_by_torso=preproc.get('scale_by_torso', True)
        ))
    
    # Temporal crop
    training = config.get('training', {})
    input_frames = training.get('input_frames', 64)
    transforms.append(TemporalCrop(target_frames=input_frames))
    
    # Spatial augmentations
    aug = config.get('dataset', {}).get('augmentation', {})
    if aug.get('enabled', True):
        transforms.extend([
            RandomRotation(aug.get('rotation_range', (-15, 15))),
            RandomScale(aug.get('scale_range', (0.9, 1.1))),
            RandomShear(aug.get('shear_range', (-0.1, 0.1))),
            GaussianNoise(aug.get('noise_std', 0.01)),
            RandomTemporalFlip(p=aug.get('temporal_flip_p', 0.5)),
        ])
    
    return Compose(transforms) if transforms else None


class UniformTemporalSample:
    """
    Uniformly sample frames across the full sequence.
    
    Used during validation/testing to cover the entire temporal span
    at lower resolution, ensuring no part of the action is missed.
    This is the standard evaluation protocol used by EfficientGCN,
    CTR-GCN, and InfoGCN.
    """
    
    def __init__(self, target_frames: int = 64):
        """
        Args:
            target_frames: Number of frames to sample
        """
        self.target_frames = target_frames
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sample frames.
        
        Args:
            skeleton: Tensor of shape (C, T, V, M)
            
        Returns:
            Sampled skeleton of shape (C, target_frames, V, M)
        """
        C, T, V, M = skeleton.shape
        
        if T <= self.target_frames:
            # Pad with zeros if too short
            padded = torch.zeros(C, self.target_frames, V, M, dtype=skeleton.dtype)
            padded[:, :T, :, :] = skeleton
            return padded
        
        # Uniformly spaced indices across the full sequence
        indices = np.linspace(0, T - 1, self.target_frames, dtype=int)
        return skeleton[:, indices, :, :]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_frames={self.target_frames})"


def get_val_transform(config: dict):
    """
    Create validation transform pipeline from config.
    
    Args:
        config: Data configuration dict
        
    Returns:
        Compose transform (normalization + uniform temporal sampling, no augmentation)
    """
    transforms = []
    
    # Only normalize for validation (if not already preprocessed)
    preproc = config.get('dataset', {}).get('preprocessing', {})
    if preproc.get('normalize', False):
        transforms.append(Normalize(
            method=preproc.get('normalization_method', 'center_spine'),
            center_joint=preproc.get('center_joint', 0),
            scale_by_torso=preproc.get('scale_by_torso', True)
        ))
    
    # Uniform temporal sampling for validation (covers full action)
    training = config.get('training', {})
    input_frames = training.get('input_frames', 64)
    transforms.append(UniformTemporalSample(target_frames=input_frames))
    
    return Compose(transforms) if transforms else None

