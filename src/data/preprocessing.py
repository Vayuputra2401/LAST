"""
Skeleton Data Preprocessing and Normalization

Utilities for normalizing and preprocessing skeleton data.
"""

import numpy as np
import torch


def normalize_skeleton_by_center(
    skeleton: np.ndarray,
    center_joint: int = 0,
    scale_by_torso: bool = True
) -> np.ndarray:
    """
    Normalize skeleton by centering on a specific joint and optionally scaling by torso length.
    
    Args:
        skeleton: Skeleton data in shape (C, T, V, M) or (T, V, C, M)
        center_joint: Joint index to use as center (default: 0 = SpineBase for NTU RGB+D)
        scale_by_torso: Whether to scale by torso length (SpineBase to Neck distance)
        
    Returns:
        Normalized skeleton with same shape as input
        
    Example:
        >>> skeleton = np.random.randn(3, 100, 25, 2)
        >>> normalized = normalize_skeleton_by_center(skeleton, center_joint=0)
        >>> # SpineBase is now at origin for all frames
    """
    skeleton = skeleton.copy()
    
    # Detect shape format
    if skeleton.shape[0] == 3:  # (C, T, V, M) format
        C, T, V, M = skeleton.shape
        
        # Get center joint position: (C, T, 1, M)
        center_pos = skeleton[:, :, center_joint:center_joint+1, :]
        
        # Center skeleton
        skeleton = skeleton - center_pos
        
        # Scale by torso length if requested
        if scale_by_torso and V >= 3:
            # Torso = distance from SpineBase (0) to Neck (2)
            spine_to_neck = skeleton[:, :, 2, :] - skeleton[:, :, 0, :]  # (C, T, M)
            torso_length = np.linalg.norm(spine_to_neck, axis=0, keepdims=True)  # (1, T, M)

            # FIX (Observation C): Use MEAN torso length across all frames, not per-frame.
            # Per-frame torso length is unstable: during bending/crouching poses, the
            # measured spine-to-neck distance shrinks, causing that frame to be scaled up
            # anomalously. Mean-frame torso length provides a stable global reference,
            # consistent with normalization.py's normalize_skeleton_scale approach.
            torso_length = np.mean(torso_length, axis=1, keepdims=True)  # (1, 1, M)

            # Avoid division by zero
            torso_length = np.maximum(torso_length, 1e-6)

            # Scale: (C, T, V, M) / (1, 1, 1, M) — broadcast over T dimension
            skeleton = skeleton / torso_length[:, :, np.newaxis, :]
            
    else:  # (T, V, C, M) format
        T, V, C, M = skeleton.shape
        
        # Get center joint position: (T, 1, C, M)
        center_pos = skeleton[:, center_joint:center_joint+1, :, :]
        
        # Center skeleton
        skeleton = skeleton - center_pos
        
        # Scale by torso length if requested
        if scale_by_torso and V >= 3:
            # Torso = distance from SpineBase (0) to Neck (2)
            spine_to_neck = skeleton[:, 2, :, :] - skeleton[:, 0, :, :]  # (T, C, M)
            torso_length = np.linalg.norm(spine_to_neck, axis=1, keepdims=True)  # (T, 1, M)

            # FIX (Observation C): Use MEAN torso length across all frames (same as C,T,V,M branch).
            torso_length = np.mean(torso_length, axis=0, keepdims=True)  # (1, 1, M)

            # Avoid division by zero
            torso_length = np.maximum(torso_length, 1e-6)

            # Scale: (T, V, C, M) / (1, 1, 1, M) — broadcast over T dimension
            skeleton = skeleton / torso_length[:, np.newaxis, :, :]
    
    return skeleton


def normalize_skeleton_by_first_frame(skeleton: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton by subtracting the center position of the first frame.
    
    This removes global translation while preserving relative motion.
    
    Args:
        skeleton: Skeleton data in shape (C, T, V, M) or (T, V, C, M)
        
    Returns:
        Normalized skeleton with same shape as input
        
    Example:
        >>> skeleton = np.random.randn(3, 100, 25, 2)
        >>> normalized = normalize_skeleton_by_first_frame(skeleton)
        >>> # First frame is centered at origin
    """
    skeleton = skeleton.copy()
    
    # Detect shape format
    if skeleton.shape[0] == 3:  # (C, T, V, M) format
        # Compute center of first frame: (C, 1, 1, M)
        first_frame_center = skeleton[:, 0:1, :, :].mean(axis=2, keepdims=True)
        skeleton = skeleton - first_frame_center
        
    else:  # (T, V, C, M) format
        # Compute center of first frame: (1, 1, C, M)
        first_frame_center = skeleton[0:1, :, :, :].mean(axis=1, keepdims=True)
        skeleton = skeleton - first_frame_center
    
    return skeleton


def normalize_skeleton(
    skeleton: np.ndarray,
    method: str = 'center_spine',
    **kwargs
) -> np.ndarray:
    """
    Normalize skeleton using specified method.
    
    Args:
        skeleton: Skeleton data
        method: Normalization method ('center_spine', 'first_frame', or 'none')
        **kwargs: Additional arguments for specific normalization methods
        
    Returns:
        Normalized skeleton
        
    Example:
        >>> skeleton = np.random.randn(3, 100, 25, 2)
        >>> normalized = normalize_skeleton(skeleton, method='center_spine')
    """
    if method == 'center_spine':
        return normalize_skeleton_by_center(
            skeleton,
            center_joint=kwargs.get('center_joint', 0),
            scale_by_torso=kwargs.get('scale_by_torso', True)
        )
    elif method == 'first_frame':
        return normalize_skeleton_by_first_frame(skeleton)
    elif method == 'none':
        return skeleton
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def temporal_crop(skeleton: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Crop or pad skeleton sequence to target number of frames.
    
    Args:
        skeleton: Skeleton in (C, T, V, M) or (T, V, C, M) format
        max_frames: Target number of frames
        
    Returns:
        Skeleton with T=max_frames
    """
    # Detect format
    if skeleton.shape[0] == 3:  # (C, T, V, M)
        C, T, V, M = skeleton.shape
        
        if T == max_frames:
            return skeleton
        elif T < max_frames:
            # Zero-pad
            padded = np.zeros((C, max_frames, V, M), dtype=skeleton.dtype)
            padded[:, :T, :, :] = skeleton
            return padded
        else:
            # Uniformly sample
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            return skeleton[:, indices, :, :]
    
    else:  # (T, V, C, M)
        T, V, C, M = skeleton.shape
        
        if T == max_frames:
            return skeleton
        elif T < max_frames:
            # Zero-pad
            padded = np.zeros((max_frames, V, C, M), dtype=skeleton.dtype)
            padded[:T, :, :, :] = skeleton
            return padded
        else:
            # Uniformly sample
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            return skeleton[indices, :, :, :]


def select_primary_body(skeleton: np.ndarray) -> np.ndarray:
    """
    Select the primary body from multi-body skeleton data.
    
    For M=2 bodies, selects the body with more non-zero frames.
    
    Args:
        skeleton: Skeleton in shape (C, T, V, M)
        
    Returns:
        Skeleton with M=1: (C, T, V, 1)
    """
    if skeleton.shape[-1] == 1:
        return skeleton
    
    C, T, V, M = skeleton.shape
    
    # Count non-zero frames for each body
    body_scores = []
    for m in range(M):
        body_data = skeleton[:, :, :, m]
        # Count frames where body is present (any non-zero coordinate)
        non_zero_frames = np.any(body_data != 0, axis=(0, 2)).sum()
        body_scores.append(non_zero_frames)
    
    # Select body with highest score
    primary_idx = np.argmax(body_scores)
    
    return skeleton[:, :, :, primary_idx:primary_idx+1]
