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


def get_train_transform(config: dict):
    """
    Create training transform pipeline from config.
    
    Args:
        config: Data configuration dict
        
    Returns:
        Compose transform
    """
    transforms = []
    
    # Always normalize
    if config['dataset']['preprocessing']['normalize']:
        transforms.append(Normalize(
            method=config['dataset']['preprocessing']['normalization_method'],
            center_joint=config['dataset']['preprocessing']['center_joint'],
            scale_by_torso=config['dataset']['preprocessing']['scale_by_torso']
        ))
    
    # Add augmentation if enabled
    if config['dataset']['augmentation']['enabled']:
        transforms.extend([
            RandomRotation(config['dataset']['augmentation']['rotation_range']),
            RandomScale(config['dataset']['augmentation']['scale_range']),
            RandomShear(config['dataset']['augmentation']['shear_range']),
            GaussianNoise(config['dataset']['augmentation']['noise_std'])
        ])
    
    return Compose(transforms) if transforms else None


def get_val_transform(config: dict):
    """
    Create validation transform pipeline from config.
    
    Args:
        config: Data configuration dict
        
    Returns:
        Compose transform (only normalization, no augmentation)
    """
    transforms = []
    
    # Only normalize for validation
    if config['dataset']['preprocessing']['normalize']:
        transforms.append(Normalize(
            method=config['dataset']['preprocessing']['normalization_method'],
            center_joint=config['dataset']['preprocessing']['center_joint'],
            scale_by_torso=config['dataset']['preprocessing']['scale_by_torso']
        ))
    
    return Compose(transforms) if transforms else None
