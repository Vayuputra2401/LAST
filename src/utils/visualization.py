"""
Visualization utilities for skeleton data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# NTU RGB+D 25-joint skeleton connections (bones)
NTU_SKELETON_BONES = [
    (0, 1),   # Spine Base -> Spine Mid
    (1, 20),  # Spine Mid -> Spine
    (20, 2),  # Spine -> Neck
    (2, 3),   # Neck -> Head
    (20, 4),  # Spine -> Left Shoulder
    (4, 5),   # Left Shoulder -> Left Elbow
    (5, 6),   # Left Elbow -> Left Wrist
    (6, 7),   # Left Wrist -> Left Hand
    (7, 21),  # Left Hand -> Left Hand Tip
    (7, 22),  # Left Hand -> Left Thumb
    (20, 8),  # Spine -> Right Shoulder
    (8, 9),   # Right Shoulder -> Right Elbow
    (9, 10),  # Right Elbow -> Right Wrist
    (10, 11), # Right Wrist -> Right Hand
    (11, 23), # Right Hand -> Right Hand Tip
    (11, 24), # Right Hand -> Right Thumb
    (0, 12),  # Spine Base -> Left Hip
    (12, 13), # Left Hip -> Left Knee
    (13, 14), # Left Knee -> Left Ankle
    (14, 15), # Left Ankle -> Left Foot
    (0, 16),  # Spine Base -> Right Hip
    (16, 17), # Right Hip -> Right Knee
    (17, 18), # Right Knee -> Right Ankle
    (18, 19), # Right Ankle -> Right Foot
]


def plot_skeleton_frame(
    skeleton: np.ndarray,
    joint_idx: int = None,
    ax: plt.Axes = None,
    title: str = None
):
    """
    Plot a single skeleton frame in 3D.
    
    Args:
        skeleton: np.ndarray of shape (V, C) or (V, C, M)
            V = joints (25), C = coordinates (3), M = bodies (optional)
        joint_idx: Optional specific joint to highlight
        ax: Matplotlib 3D axis (created if None)
        title: Plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Handle multi-body case
    if skeleton.ndim == 3:
        skeleton = skeleton[:, :, 0]  # Take first body
    
    # Extract coordinates
    x = skeleton[:, 0]
    y = skeleton[:, 1]
    z = skeleton[:, 2]
    
    # Plot joints
    ax.scatter(x, y, z, c='blue', marker='o', s=50, alpha=0.8)
    
    # Highlight specific joint if provided
    if joint_idx is not None:
        ax.scatter(x[joint_idx], y[joint_idx], z[joint_idx], 
                  c='red', marker='o', s=100, label=f'Joint {joint_idx}')
    
    # Plot bones
    for bone in NTU_SKELETON_BONES:
        joint1, joint2 = bone
        ax.plot([x[joint1], x[joint2]], 
               [y[joint1], y[joint2]], 
               [z[joint1], z[joint2]], 
               'g-', linewidth=2, alpha=0.6)
    
    # Set labels and limits
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def plot_skeleton_sequence(
    skeleton_seq: np.ndarray,
    num_frames: int = 8,
    figsize: tuple = (16, 8)
):
    """
    Plot multiple frames from a skeleton sequence.
    
    Args:
        skeleton_seq: np.ndarray of shape (T, V, C) or (T, V, C, M)
            T = frames, V = joints, C = coordinates
        num_frames: Number of frames to visualize
        figsize: Figure size
    """
    T = skeleton_seq.shape[0]
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    rows = 2
    cols = num_frames // rows
    
    fig = plt.figure(figsize=figsize)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        plot_skeleton_frame(skeleton_seq[frame_idx], ax=ax, title=f'Frame {frame_idx}')
    
    plt.tight_layout()
    return fig


def plot_data_distribution(
    dataset,
    max_samples: int = 1000
):
    """
    Plot data distribution statistics.
    
    Args:
        dataset: SkeletonDataset instance
        max_samples: Maximum samples to analyze
    """
    num_samples = min(len(dataset), max_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Label distribution
    labels = [dataset.labels[i] for i in range(num_samples)]
    axes[0, 0].hist(labels, bins=50, edgecolor='black')
    axes[0, 0].set_title('Action Label Distribution')
    axes[0, 0].set_xlabel('Action Class')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Sequence length distribution (for skeleton files)
    if dataset.data_type == 'skeleton':
        seq_lengths = []
        for i in range(min(100, num_samples)):  # Sample subset
            data, _ = dataset[i]
            # Find actual sequence length (non-zero frames)
            non_zero_frames = np.where(data.sum(dim=(0, 2, 3)).numpy() > 0)[0]
            if len(non_zero_frames) > 0:
                seq_lengths.append(len(non_zero_frames))
        
        axes[0, 1].hist(seq_lengths, bins=30, edgecolor='black')
        axes[0, 1].set_title('Sequence Length Distribution')
        axes[0, 1].set_xlabel('Frames')
        axes[0, 1].set_ylabel('Count')
    
    # 3. Sample visualization
    sample_data, sample_label = dataset[0]
    # sample_data shape: (C, T, V, M)
    sample_np = sample_data[:, :, :, 0].numpy()  # (C, T, V)
    sample_np = sample_np.transpose(1, 2, 0)  # (T, V, C)
    
    # Plot first frame
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    plot_skeleton_frame(sample_np[0], ax=ax, title=f'Sample Frame (Label: {sample_label})')
    
    # 4. Coordinate distribution
    coords = sample_np.reshape(-1, 3)  # Flatten all frames and joints
    axes[1, 1].scatter(coords[:, 0], coords[:, 1], alpha=0.3, s=1)
    axes[1, 1].set_title('XY Coordinate Distribution')
    axes[1, 1].set_xlabel('X (meters)')
    axes[1, 1].set_ylabel('Y (meters)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def print_data_stats(data: np.ndarray, label: int = None):
    """
    Print statistics about skeleton data.
    
    Args:
        data: Skeleton data of shape (C, T, V, M) or (T, V, C)
        label: Optional action label
    """
    print("=" * 60)
    print("Skeleton Data Statistics")
    print("=" * 60)
    
    if label is not None:
        print(f"Action Label: {label}")
    
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {data.min():.4f}")
    print(f"Max value: {data.max():.4f}")
    print(f"Mean value: {data.mean():.4f}")
    print(f"Std value: {data.std():.4f}")
    
    # Check for non-zero content
    non_zero = np.count_nonzero(data)
    total = data.size
    print(f"Non-zero elements: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
    
    print("=" * 60)
