import numpy as np

def align_to_spine_base(skeleton):
    """
    Subtract the SpineBase coordinates from all joints for each frame.
    
    Args:
        skeleton: (C, T, V, M)
        
    Returns:
        Aligned skeleton: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    # SpineBase is usually Joint 0 in NTU (verify if using 0-based index)
    # NTU-RGB+D: 1 is base of spine. 0-index -> 0.
    spine_base_idx = 0
    
    # Extract spine base: (C, T, 1, M)
    spine_base = skeleton[:, :, spine_base_idx:spine_base_idx+1, :]
    
    # Subtract
    skeleton = skeleton - spine_base
    return skeleton

def rotate_to_front(skeleton):
    """
    Rotate the skeleton so the vector from Left Shoulder to Right Shoulder 
    is parallel to the X-axis.
    
    Args:
        skeleton: (C, T, V, M)
        
    Returns:
        Rotated skeleton: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    # NTU Joints (0-based):
    # 4: Left Shoulder
    # 8: Right Shoulder
    # (Check verify_labels or documentation for exact indices if standard NTU ordering)
    # Assuming Standard NTU: #4=LeftShoulder, #8=RightShoulder
    # Use config or hardcode for now, but ensure consistency.
    
    # standard ntu: 
    # 1 base of spine -> 0
    # 5 left shoulder -> 4
    # 9 right shoulder -> 8
    
    idx_l_shoulder = 4
    idx_r_shoulder = 8
    
    # Get shoulders for the first frame (or average across frames? Usually first frame is enough or frame-wise)
    # Standard practice: Rotate based on the first frame of the clip to maintain temporal consistency,
    # OR rotate frame-by-frame. Frame-by-frame removes global rotation augmentation but fixes view. 
    # EfficientGCN usually does it frame-by-frame in preprocessing or once per clip.
    # Let's do it per frame for maximum invariance.
    
    # Left (C, T, 1, M), Right (C, T, 1, M)
    l_shoulder = skeleton[:, :, idx_l_shoulder:idx_l_shoulder+1, :]
    r_shoulder = skeleton[:, :, idx_r_shoulder:idx_r_shoulder+1, :]
    
    # Vector form Left to Right
    # Project to XY plane (ignore Z for rotation angle calc around Z-axis)
    # Vector: (X, Y)
    x_diff = r_shoulder[0] - l_shoulder[0] # (T, 1, M)
    y_diff = r_shoulder[1] - l_shoulder[1] # (T, 1, M)
    
    # Angle to X-axis
    theta = np.arctan2(y_diff, x_diff) # (T, 1, M)
    
    # We want to rotate by -theta to align with X-axis
    rotation_angle = -theta
    
    cos_t = np.cos(rotation_angle)
    sin_t = np.sin(rotation_angle)
    
    # Rotation Matrix around Z-axis
    # [ cos -sin  0 ]
    # [ sin  cos  0 ]
    # [  0    0   1 ]
    
    # Apply rotation
    x = skeleton[0] # (T, V, M)
    y = skeleton[1] # (T, V, M)
    z = skeleton[2] # (T, V, M)
    
    # Expand dims for broadcasting
    # cos_t is (T, 1, M) -> broadcast to (T, V, M)
    cos_t = np.tile(cos_t, (1, V, 1))
    sin_t = np.tile(sin_t, (1, V, 1))
    
    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    z_new = z
    
    return np.stack([x_new, y_new, z_new], axis=0)

def normalize_skeleton_scale(skeleton):
    """
    Scale skeleton so the channel-wise std across all frames/joints is 1 (or other metric).
    However, standard SOTA usually just normalizes to mean 0, std 1 globally later.
    EfficientGCN preprocessing often includes centering and view invariance.
    Let's stick to View Invariance -> Mean/Std Norm later.
    
    Alternatively, scale by average bone length (as per design doc).
    """
    return skeleton
