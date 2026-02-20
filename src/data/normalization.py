import numpy as np

def align_to_spine_base(skeleton):
    """
    Subtract the first frame's SpineBase coordinates from all joints.
    Preserves trajectory information (relative motion).
    
    Args:
        skeleton: (C, T, V, M)
        
    Returns:
        Aligned skeleton: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    spine_base_idx = 0
    
    # Extract spine base from the FIRST frame: (C, 1, 1, M)
    # We broadcast this across T to preserve global motion relative to initial position
    spine_base = skeleton[:, 0:1, spine_base_idx:spine_base_idx+1, :]
    
    # Subtract
    skeleton = skeleton - spine_base
    return skeleton

def rotate_to_front(skeleton):
    """
    Rotate the skeleton around the Y-axis (UP) so the vector from 
    Left Shoulder to Right Shoulder is parallel to the X-axis.
    
    NTU Coords: Y is UP. Rotation happens in X-Z plane.
    
    Args:
        skeleton: (C, T, V, M)
        
    Returns:
        Rotated skeleton: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    idx_l_shoulder = 4
    idx_r_shoulder = 8
    
    # Get shoulders from the FIRST frame to determine rotation angle
    # ensuring the whole sequence rotates together rigidly
    l_shoulder = skeleton[:, 0, idx_l_shoulder, :] # (C, M)
    r_shoulder = skeleton[:, 0, idx_r_shoulder, :] # (C, M)
    
    # Vector form Left to Right in X-Z plane
    # C=0: X, C=1: Y, C=2: Z
    x_diff = r_shoulder[0] - l_shoulder[0] # (M,)
    z_diff = r_shoulder[2] - l_shoulder[2] # (M,)
    
    # Angle to X-axis in X-Z plane
    theta = np.arctan2(z_diff, x_diff) # (M,)
    
    # We want to rotate by -theta
    rotation_angle = -theta
    
    cos_t = np.cos(rotation_angle) # (M,)
    sin_t = np.sin(rotation_angle) # (M,)
    
    # Rotation Matrix around Y-axis
    # [ cos  0  sin ]
    # [  0   1   0  ]
    # [ -sin 0  cos ]
    
    x = skeleton[0] # (T, V, M)
    y = skeleton[1] # (T, V, M)
    z = skeleton[2] # (T, V, M)
    
    # Expand dims for broadcasting: (M,) -> (1, 1, M) -> (T, V, M)
    cos_t = cos_t[None, None, :]
    sin_t = sin_t[None, None, :]
    
    # x' = x cos + z sin
    # z' = -x sin + z cos
    # y' = y
    x_new = x * cos_t + z * sin_t
    y_new = y
    z_new = z * cos_t - x * sin_t
    
    return np.stack([x_new, y_new, z_new], axis=0)

def normalize_skeleton_scale(skeleton):
    """
    Scale the skeleton so that the average torso length (SpineBase to Neck) is 1.0.
    This provides scale invariance across different subjects (e.g., child vs adult).
    
    Args:
        skeleton: (C, T, V, M)
        
    Returns:
        Scaled skeleton: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    idx_spine = 0
    idx_neck = 2
    
    # Calculate torso vector for each frame: (C, T, 1, M)
    # Using all frames to get a stable scale factor for the whole clip
    spine = skeleton[:, :, idx_spine:idx_spine+1, :]
    neck = skeleton[:, :, idx_neck:idx_neck+1, :]
    
    torso_vector = neck - spine # (C, T, 1, M)
    torso_len = np.linalg.norm(torso_vector, axis=0, keepdims=True) # (1, T, 1, M)
    
    # Average torso length across frames for each body: (1, 1, 1, M)
    # We want one scale factor per body per clip to preserve temporal dynamics (breathing/bending)
    mean_torso_len = np.mean(torso_len, axis=1, keepdims=True)
    
    # Avoid division by zero
    mean_torso_len = np.maximum(mean_torso_len, 1e-6)
    
    # Scale
    skeleton = skeleton / mean_torso_len
    
    return skeleton
