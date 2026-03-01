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
            # FIX (Observation D): Repeat-pad last frame instead of zero-padding.
            # Zero-pad creates an artificial 'zero pose' that confuses linear attention
            # (φ(elu(0)+1) ≠ 0, so zero frames pollute the context sum).
            # Repeating the last valid frame is semantically neutral.
            last_frame = skeleton[:, -1:, :, :]
            repeat_count = self.target_frames - T
            pad = last_frame.expand(-1, repeat_count, -1, -1)
            return torch.cat([skeleton, pad], dim=1)

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


class RandomTemporalSpeed:
    """
    Random temporal speed augmentation — applied identically across MIB streams.

    Simulates a faster action by selecting a random contiguous window of
    T_sub = int(T × r) frames (r ∈ speed_range), then repeat-padding the
    last frame back to T. Only r < 1 is effective; r ≥ 1 is a no-op since
    pre-subsampled data cannot produce extra frames.

    Must be used inside MIBTransform (shared params across streams).
    """

    def __init__(self, speed_range: tuple = (0.8, 1.2), p: float = 1.0):
        self.speed_range = speed_range
        self.p = p

    def sample_params(self, T: int):
        """
        Sample (T_sub, start_idx) for a random crop.
        Returns (T, 0) when augmentation is skipped (no-op).
        """
        if np.random.random() >= self.p:
            return T, 0
        r = float(np.random.uniform(self.speed_range[0], self.speed_range[1]))
        r = min(r, 1.0)  # r > 1 → no extra frames → identity
        T_sub = max(1, int(T * r))
        start = np.random.randint(0, T - T_sub + 1) if T_sub < T else 0
        return T_sub, start

    @staticmethod
    def apply(x: torch.Tensor, T_sub: int, start: int) -> torch.Tensor:
        """Crop x[:, start:start+T_sub] and repeat-pad back to original T."""
        T = x.shape[1]
        if T_sub >= T:
            return x
        x_crop = x[:, start:start + T_sub]                          # (C, T_sub, V, M)
        pad    = x_crop[:, -1:].expand(-1, T - T_sub, -1, -1)      # repeat last frame
        return torch.cat([x_crop, pad], dim=1)

    def __repr__(self) -> str:
        return f"RandomTemporalSpeed(speed_range={self.speed_range}, p={self.p})"


class MIBTransform:
    """
    Geometrically consistent augmentation for Multi-Input Branch (MIB) data.

    Problem: When a single-stream Compose transform is applied independently to
    joint, velocity, and bone streams inside dataset.__getitem__, each call
    generates its OWN random parameters (different rotation angle, different scale,
    different temporal start, different flip decision). The three streams then
    describe physically inconsistent views of the same action, corrupting the
    multi-stream fusion signal.

    Fix: MIBTransform accepts a dict of streams, samples ONE set of shared random
    parameters, and applies the SAME parameters identically to every stream.

    The velocity stream (finite differences) and bone stream (child-parent vectors)
    are both linear functions of the joint positions, so all linear spatial
    transforms (rotation, scale, shear) are equivariant:
        velocity_rotated == rotate(velocity)
        bone_rotated     == rotate(bone)
    Temporal crop and flip must also be shared (same time window for all streams).
    Gaussian noise is added INDEPENDENTLY per stream on purpose, since each stream
    has a different noise scale (position noise vs. velocity noise vs. bone noise).

    Args:
        target_frames:  Output temporal length (same as TemporalCrop.target_frames).
        rotation_range: (min_deg, max_deg), default (-15, 15).
        scale_range:    (min_scale, max_scale), default (0.9, 1.1).
        shear_range:    (min_shear, max_shear), default (-0.1, 0.1).
        noise_std:      Per-stream Gaussian noise std. Can be a float (same for
                         all streams) or a dict mapping stream names to std values,
                         e.g. {'joint': 0.01, 'velocity': 0.002, 'bone': 0.005}.
                         Per-stream scaling prevents velocity-stream drowning:
                         velocity std ≈ 0.02, so noise_std=0.01 → 50% SNR.
        temporal_flip_p: Probability of reversing frame order, default 0.0.
                         Keep at 0.0 for NTU60/120 — most actions are directional.
        is_training:    If False, uses deterministic center-crop (no augmentation).
    """

    # Default per-stream noise scaling factors (relative to base noise_std).
    # Calibrated to ~6-8% SNR across all streams:
    #   joint  std ≈ 0.15 → 0.01 / 0.15 ≈ 6.7%
    #   velocity std ≈ 0.02 → 0.002 / 0.02 ≈ 10%
    #   bone   std ≈ 0.12 → 0.005 / 0.12 ≈ 4.2%
    DEFAULT_NOISE_STD = {'joint': 0.01, 'velocity': 0.002, 'bone': 0.005, 'bone_velocity': 0.005}

    def __init__(
        self,
        target_frames: int = 64,
        rotation_range: tuple = (-15, 15),
        scale_range: tuple = (0.9, 1.1),
        shear_range: tuple = (-0.1, 0.1),
        noise_std = 0.01,
        temporal_flip_p: float = 0.0,
        temporal_speed_p: float = 0.0,
        temporal_speed_range: tuple = (0.8, 1.2),
        is_training: bool = True,
    ):
        self.target_frames = target_frames
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        # Accept float (backward-compat) or dict (per-stream)
        if isinstance(noise_std, dict):
            self.noise_std = noise_std
        else:
            # Legacy scalar → uniform across all streams
            self.noise_std = {'joint': float(noise_std),
                              'velocity': float(noise_std),
                              'bone': float(noise_std),
                              'bone_velocity': float(noise_std)}
        self.temporal_flip_p = temporal_flip_p
        self.temporal_speed_p = temporal_speed_p
        self.temporal_speed_range = tuple(temporal_speed_range)
        self.is_training = is_training
        self._speed_aug = RandomTemporalSpeed(temporal_speed_range, p=temporal_speed_p)

    # ------------------------------------------------------------------
    # Helpers that accept pre-sampled scalars so we can reuse parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_temporal_crop(x: torch.Tensor, start: int, target: int) -> torch.Tensor:
        """Crop x[:, start:start+target] or repeat-pad last frame if x is too short.

        FIX (Observation D): Use repeat-padding (replicate last real frame) instead of
        zero-padding for short sequences. Zero-padding creates an artificial 'zero pose'
        that does not exist in any real action and confuses linear attention — φ(elu(0)+1)
        is non-zero, so zero frames actively participate in the context sum. Repeating the
        last valid frame is semantically neutral (zero velocity, no articulation change)
        and keeps the attention context meaningful throughout the padded region.
        """
        C, T, V, M = x.shape
        if T <= target:
            # Repeat-pad last frame to fill remaining length
            last_frame = x[:, -1:, :, :]  # (C, 1, V, M)
            repeat_count = target - T
            pad = last_frame.expand(-1, repeat_count, -1, -1)  # (C, repeat_count, V, M)
            return torch.cat([x, pad], dim=1)
        return x[:, start:start + target]

    @staticmethod
    def _apply_center_crop(x: torch.Tensor, target: int) -> torch.Tensor:
        """Deterministic center crop for validation, repeat-pad if too short.

        Same repeat-padding strategy as _apply_temporal_crop (see Observation D fix).
        """
        C, T, V, M = x.shape
        if T <= target:
            last_frame = x[:, -1:, :, :]
            repeat_count = target - T
            pad = last_frame.expand(-1, repeat_count, -1, -1)
            return torch.cat([x, pad], dim=1)
        start = (T - target) // 2
        return x[:, start:start + target]

    @staticmethod
    def _apply_rotation(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate around Y-axis by angle_deg (shared across streams)."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = torch.tensor(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]],
            dtype=x.dtype
        )
        C, T, V, M = x.shape
        flat = x.permute(1, 2, 3, 0).reshape(-1, 3)   # (T*V*M, C)
        rotated = flat @ R.T
        return rotated.reshape(T, V, M, C).permute(3, 0, 1, 2)

    @staticmethod
    def _apply_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
        return x * scale

    @staticmethod
    def _apply_shear(x: torch.Tensor, shear: float) -> torch.Tensor:
        S = torch.tensor(
            [[1, shear, 0], [0, 1, 0], [0, 0, 1]],
            dtype=x.dtype
        )
        C, T, V, M = x.shape
        flat = x.permute(1, 2, 3, 0).reshape(-1, 3)
        sheared = flat @ S.T
        return sheared.reshape(T, V, M, C).permute(3, 0, 1, 2)

    @staticmethod
    def _apply_flip(x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=[1])

    # ------------------------------------------------------------------

    def __call__(self, streams: dict) -> dict:
        """
        Apply geometrically consistent transforms to all MIB streams.

        Args:
            streams: dict with keys 'joint', 'velocity', 'bone',
                     each value is a Tensor of shape (C, T, V, M).

        Returns:
            Transformed dict with same keys.
        """
        # ---- Sample shared random parameters ONCE ----
        if self.is_training:
            angle   = float(np.random.uniform(self.rotation_range[0], self.rotation_range[1]))
            scale   = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
            shear   = float(np.random.uniform(self.shear_range[0], self.shear_range[1]))
            do_flip = np.random.random() < self.temporal_flip_p
            # Temporal speed: sample crop params once for all streams
            _T_ref  = next(iter(streams.values())).shape[1]
            speed_T_sub, speed_start = self._speed_aug.sample_params(_T_ref)
            do_speed = speed_T_sub < _T_ref

        # ---- Apply identical geometric transforms to every stream ----
        out = {}
        for name, x in streams.items():
            if self.is_training:
                x = self._apply_rotation(x, angle)
                x = self._apply_scale(x, scale)
                x = self._apply_shear(x, shear)
                if do_flip:
                    x = self._apply_flip(x)
                if do_speed:
                    x = RandomTemporalSpeed.apply(x, speed_T_sub, speed_start)
                # Independent per-stream noise, scaled per-stream to maintain
                # consistent SNR (~6-8%) across joint/velocity/bone.
                stream_noise_std = self.noise_std.get(name, 0.01)
                if stream_noise_std > 0:
                    x = x + torch.randn_like(x) * stream_noise_std

            # Subsampling/padding is already done precisely in preprocess_data.py
            # So no crop needed here during validation or training.
            out[name] = x
        return out

    def __repr__(self) -> str:
        noise_repr = ', '.join(f'{k}={v}' for k, v in self.noise_std.items())
        return (
            f"MIBTransform(target_frames={self.target_frames}, "
            f"rotation={self.rotation_range}, scale={self.scale_range}, "
            f"shear={self.shear_range}, noise_std={{{noise_repr}}}, "
            f"flip_p={self.temporal_flip_p}, speed_p={self.temporal_speed_p}, "
            f"speed_range={self.temporal_speed_range}, training={self.is_training})"
        )


def get_train_transform(config: dict):
    """
    Create training transform pipeline from config.

    For MIB data_type, returns a MIBTransform (shared-seed, dict-in/dict-out).
    For single-stream npy/skeleton data, returns a Compose pipeline.

    Args:
        config: Data configuration dict

    Returns:
        MIBTransform (for MIB) or Compose (for single-stream), or None
    """
    data_type = config.get('dataset', {}).get('data_type', 'npy')

    training = config.get('training', {})
    input_frames = training.get('input_frames', 64)
    aug = config.get('dataset', {}).get('augmentation', {})

    # MIB mode: shared-seed augmentation across joint/velocity/bone streams
    if data_type == 'mib':
        if aug.get('enabled', True):
            # Build per-stream noise dict from config.
            # Accepts either a scalar noise_std (legacy, uniform) or
            # per-stream keys noise_std_joint / noise_std_velocity / noise_std_bone.
            raw_noise = aug.get('noise_std', 0.01)
            if isinstance(raw_noise, (int, float)):
                # Check for per-stream overrides in the config
                noise_joint = aug.get('noise_std_joint', None)
                noise_vel   = aug.get('noise_std_velocity', None)
                noise_bone  = aug.get('noise_std_bone', None)
                noise_bv = aug.get('noise_std_bone_velocity', None)
                if any(v is not None for v in (noise_joint, noise_vel, noise_bone)):
                    # At least one per-stream override → build dict,
                    # falling back to the global noise_std for unspecified streams
                    noise_std = {
                        'joint':         noise_joint if noise_joint is not None else raw_noise,
                        'velocity':      noise_vel   if noise_vel   is not None else raw_noise,
                        'bone':          noise_bone  if noise_bone  is not None else raw_noise,
                        'bone_velocity': noise_bv    if noise_bv    is not None else
                                         (noise_bone if noise_bone is not None else raw_noise),
                    }
                else:
                    # Pure scalar → use defaults calibrated for ~6-8% SNR
                    noise_std = MIBTransform.DEFAULT_NOISE_STD
            else:
                noise_std = dict(raw_noise)  # Already a dict from YAML

            return MIBTransform(
                target_frames=input_frames,
                rotation_range=tuple(aug.get('rotation_range', (-15, 15))),
                scale_range=tuple(aug.get('scale_range', (0.9, 1.1))),
                shear_range=tuple(aug.get('shear_range', (-0.1, 0.1))),
                noise_std=noise_std,
                temporal_flip_p=aug.get('temporal_flip_p', 0.0),
                temporal_speed_p=aug.get('temporal_speed_p', 0.0),
                temporal_speed_range=tuple(aug.get('temporal_speed_range', (0.8, 1.2))),
                is_training=True,
            )
        else:
            # Augmentation disabled: only temporal crop (still needs MIBTransform for dict)
            return MIBTransform(
                target_frames=input_frames,
                rotation_range=(0, 0),
                scale_range=(1, 1),
                shear_range=(0, 0),
                noise_std=0.0,
                temporal_flip_p=0.0,
                is_training=True,
            )

    # Single-stream mode (npy / skeleton)
    transforms = []

    # Normalize if data is NOT pre-normalized
    preproc = config.get('dataset', {}).get('preprocessing', {})
    if preproc.get('normalize', False):
        transforms.append(Normalize(
            method=preproc.get('normalization_method', 'center_spine'),
            center_joint=preproc.get('center_joint', 0),
            scale_by_torso=preproc.get('scale_by_torso', True)
        ))

    # Note: TemporalCrop removed as data is already subsampled precisely to target frames.

    if aug.get('enabled', True):
        transforms.extend([
            RandomRotation(aug.get('rotation_range', (-15, 15))),
            RandomScale(aug.get('scale_range', (0.9, 1.1))),
            RandomShear(aug.get('shear_range', (-0.1, 0.1))),
            GaussianNoise(aug.get('noise_std', 0.01)),
            RandomTemporalFlip(p=aug.get('temporal_flip_p', 0.0)),
        ])

    return Compose(transforms) if transforms else None


class UniformTemporalSample:
    """
    Deterministic center temporal crop for validation/testing.
    
    Takes a contiguous window of target_frames from the center of the sequence.
    This matches the distribution of TemporalCrop (training) but is deterministic,
    ensuring consistent, reproducible validation results.
    
    Note: We take contiguous frames (not subsampled) because the model learns
    temporal patterns from contiguous motion sequences during training.
    Subsampling every Nth frame destroys these patterns.
    """
    
    def __init__(self, target_frames: int = 64):
        """
        Args:
            target_frames: Number of frames to sample
        """
        self.target_frames = target_frames
    
    def __call__(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Center crop frames.

        Args:
            skeleton: Tensor of shape (C, T, V, M)

        Returns:
            Cropped skeleton of shape (C, target_frames, V, M)
        """
        C, T, V, M = skeleton.shape

        if T <= self.target_frames:
            # FIX (Observation D): Repeat-pad last frame instead of zero-padding
            # (same rationale as TemporalCrop fix above).
            last_frame = skeleton[:, -1:, :, :]
            repeat_count = self.target_frames - T
            pad = last_frame.expand(-1, repeat_count, -1, -1)
            return torch.cat([skeleton, pad], dim=1)

        # Deterministic center crop
        start = (T - self.target_frames) // 2
        return skeleton[:, start:start + self.target_frames, :, :]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_frames={self.target_frames})"


def get_val_transform(config: dict):
    """
    Create validation transform pipeline from config.

    For MIB data_type, returns MIBTransform(is_training=False) which applies
    deterministic center-crop only — no augmentation, dict-in/dict-out.
    For single-stream npy/skeleton data, returns a Compose pipeline.

    Args:
        config: Data configuration dict

    Returns:
        MIBTransform (for MIB) or Compose (for single-stream), or None
    """
    data_type = config.get('dataset', {}).get('data_type', 'npy')

    training = config.get('training', {})
    input_frames = training.get('input_frames', 64)

    # MIB mode: deterministic center-crop only, no augmentation
    if data_type == 'mib':
        return MIBTransform(
            target_frames=input_frames,
            is_training=False,
        )

    # Single-stream mode (npy / skeleton)
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
    transforms.append(UniformTemporalSample(target_frames=input_frames))

    return Compose(transforms) if transforms else None

