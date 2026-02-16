"""
Skeleton Visualization Test

Visualizes a raw .skeleton file and its normalized numpy version as GIF animations.
Outputs are saved to tests/data_test/.

Usage:
    python tests/test_visualize_skeleton.py
    python tests/test_visualize_skeleton.py --file S001C001P001R001A001.skeleton
    python tests/test_visualize_skeleton.py --max_frames 60
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.data.skeleton_loader import SkeletonFileParser
from src.data.preprocessing import normalize_skeleton
from src.utils.config import load_config

# NTU RGB+D 25-joint skeleton connections
# Each tuple is (parent_joint, child_joint)
NTU_SKELETON_EDGES = [
    # Spine
    (0, 1),    # SpineBase -> SpineMid
    (1, 20),   # SpineMid -> SpineShoulder
    (20, 2),   # SpineShoulder -> Neck
    (2, 3),    # Neck -> Head
    # Left arm
    (20, 4),   # SpineShoulder -> ShoulderLeft
    (4, 5),    # ShoulderLeft -> ElbowLeft
    (5, 6),    # ElbowLeft -> WristLeft
    (6, 7),    # WristLeft -> HandLeft
    (7, 21),   # HandLeft -> HandTipLeft
    (6, 22),   # WristLeft -> ThumbLeft
    # Right arm
    (20, 8),   # SpineShoulder -> ShoulderRight
    (8, 9),    # ShoulderRight -> ElbowRight
    (9, 10),   # ElbowRight -> WristRight
    (10, 11),  # WristRight -> HandRight
    (11, 23),  # HandRight -> HandTipRight
    (10, 24),  # WristRight -> ThumbRight
    # Left leg
    (0, 12),   # SpineBase -> HipLeft
    (12, 13),  # HipLeft -> KneeLeft
    (13, 14),  # KneeLeft -> AnkleLeft
    (14, 15),  # AnkleLeft -> FootLeft
    # Right leg
    (0, 16),   # SpineBase -> HipRight
    (16, 17),  # HipRight -> KneeRight
    (17, 18),  # KneeRight -> AnkleRight
    (18, 19),  # AnkleRight -> FootRight
]

# Joint colors by body region
JOINT_COLORS = {
    'spine': [0, 1, 2, 3, 20],
    'left_arm': [4, 5, 6, 7, 21, 22],
    'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg': [12, 13, 14, 15],
    'right_leg': [16, 17, 18, 19],
}

REGION_COLORS = {
    'spine': '#FFD700',
    'left_arm': '#00BFFF',
    'right_arm': '#FF6347',
    'left_leg': '#32CD32',
    'right_leg': '#FF69B4',
}


def get_joint_color(joint_idx):
    """Get color for a joint based on body region."""
    for region, joints in JOINT_COLORS.items():
        if joint_idx in joints:
            return REGION_COLORS[region]
    return '#FFFFFF'


def get_edge_color(j1, j2):
    """Get color for an edge based on the region of its joints."""
    for region, joints in JOINT_COLORS.items():
        if j1 in joints and j2 in joints:
            return REGION_COLORS[region]
    return '#AAAAAA'


def draw_skeleton_frame(ax, joints_xyz, title="", view='front'):
    """
    Draw a single skeleton frame on a matplotlib 3D axis.

    Args:
        ax: matplotlib 3D axis
        joints_xyz: (V, 3) array of joint positions [x, y, z]
        title: Plot title
        view: 'front' or 'side'
    """
    ax.cla()

    # NTU RGB+D coordinate system: X=horizontal, Y=vertical(up), Z=depth
    # Matplotlib 3D: X=horizontal, Y=depth, Z=vertical(up)
    # So we map: NTU_X -> plot_X, NTU_Z -> plot_Y (depth), NTU_Y -> plot_Z (up)
    px = joints_xyz[:, 0]   # X stays horizontal
    py = joints_xyz[:, 2]   # NTU Z -> plot depth
    pz = joints_xyz[:, 1]   # NTU Y -> plot vertical (up)

    # Draw edges
    for j1, j2 in NTU_SKELETON_EDGES:
        color = get_edge_color(j1, j2)
        ax.plot([px[j1], px[j2]], [py[j1], py[j2]], [pz[j1], pz[j2]],
                color=color, linewidth=2, alpha=0.8)

    # Draw joints
    for i in range(len(px)):
        color = get_joint_color(i)
        ax.scatter(px[i], py[i], pz[i], c=color, s=30, edgecolors='black',
                   linewidths=0.5, zorder=5)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (left/right)')
    ax.set_ylabel('Z (depth)')
    ax.set_zlabel('Y (up)')

    if view == 'front':
        ax.view_init(elev=15, azim=-70)
    else:
        ax.view_init(elev=15, azim=20)

    ax.set_facecolor('#1a1a2e')
    ax.grid(True, alpha=0.2)


def create_skeleton_gif(skeleton_data, output_path, title_prefix="",
                        max_frames=60, fps=15):
    """
    Create a GIF animation from skeleton sequence.

    Args:
        skeleton_data: (T, V, C) array — T frames, V joints, C=3 coords
        output_path: Path to save GIF
        title_prefix: Title prefix for frames
        max_frames: Max frames to render
        fps: Frames per second for GIF
    """
    T = skeleton_data.shape[0]

    # Skip zero-padded trailing frames
    active_frames = T
    for t in range(T - 1, -1, -1):
        if np.any(skeleton_data[t] != 0):
            active_frames = t + 1
            break

    num_frames = min(active_frames, max_frames)
    if num_frames < active_frames:
        indices = np.linspace(0, active_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(num_frames)

    # Compute global axis limits from active frames
    # After Y/Z swap: plot_X=NTU_X, plot_Y=NTU_Z, plot_Z=NTU_Y
    active_data = skeleton_data[:active_frames]
    nonzero_mask = np.any(active_data != 0, axis=-1)  # (T, V)
    if nonzero_mask.any():
        valid = active_data[nonzero_mask]  # (N, 3) with columns [x, y, z]
        # Compute limits in plot space (after swap)
        x_vals = valid[:, 0]  # NTU X
        y_vals = valid[:, 2]  # NTU Z -> plot Y
        z_vals = valid[:, 1]  # NTU Y -> plot Z
        mins = np.array([x_vals.min(), y_vals.min(), z_vals.min()])
        maxs = np.array([x_vals.max(), y_vals.max(), z_vals.max()])
    else:
        mins = np.array([-1, -1, -1])
        maxs = np.array([1, 1, 1])

    padding = 0.15 * (maxs - mins + 1e-6)
    mins -= padding
    maxs += padding

    fig = plt.figure(figsize=(8, 6), facecolor='#0f0f23')
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_num):
        t = indices[frame_num]
        joints = skeleton_data[t]  # (V, 3)
        title = f"{title_prefix} — Frame {t+1}/{active_frames}"
        draw_skeleton_frame(ax, joints, title=title)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        return []

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 // fps)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved: {output_path} ({len(indices)} frames)")


def main():
    parser = argparse.ArgumentParser(description='Visualize skeleton data as GIF')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific .skeleton filename (default: picks first file)')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Max frames to render in GIF (default: 60)')
    parser.add_argument('--fps', type=int, default=15,
                       help='GIF frames per second (default: 15)')
    args = parser.parse_args()

    print("=" * 60)
    print("SKELETON VISUALIZATION TEST")
    print("=" * 60)

    # Load config to get data path
    config = load_config(dataset='ntu60')
    data_root = config['environment']['paths']['data_root']

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'data_test')
    os.makedirs(output_dir, exist_ok=True)

    # Find a skeleton file
    if args.file:
        skeleton_path = os.path.join(data_root, args.file)
    else:
        files = glob.glob(os.path.join(data_root, '*.skeleton'))
        if not files:
            print(f"ERROR: No .skeleton files found in {data_root}")
            return
        skeleton_path = files[0]

    filename = os.path.basename(skeleton_path)
    sample_name = filename.replace('.skeleton', '')
    print(f"\nFile: {filename}")

    # Parse metadata
    skel_parser = SkeletonFileParser(num_joints=25, max_bodies=2)
    metadata = skel_parser.extract_metadata_from_filename(skeleton_path)
    print(f"  Setup: S{metadata['setup']:03d}  Camera: C{metadata['camera']:03d}")
    print(f"  Person: P{metadata['person']:03d}  Action: A{metadata['action']+1:03d}")

    # ---- 1. Raw skeleton ----
    print(f"\n1. Parsing raw skeleton...")
    raw_data, parse_meta = skel_parser.parse_file(skeleton_path)
    # raw_data shape: (T, V, C, M)
    T, V, C, M = raw_data.shape
    print(f"  Shape: (T={T}, V={V}, C={C}, M={M})")

    # Select primary body: (T, V, C)
    body0 = raw_data[:, :, :, 0]
    print(f"  Body 0 shape: {body0.shape}")

    print(f"\n  Generating raw skeleton GIF...")
    raw_gif_path = os.path.join(output_dir, f'{sample_name}_raw.gif')
    create_skeleton_gif(body0, raw_gif_path,
                       title_prefix=f"Raw — {sample_name}",
                       max_frames=args.max_frames, fps=args.fps)

    # ---- 2. Normalized skeleton ----
    print(f"\n2. Normalizing skeleton...")

    # Transpose to (C, T, V, M) for normalization
    raw_ctvm = raw_data.transpose(2, 0, 1, 3)
    normalized = normalize_skeleton(raw_ctvm, method='center_spine',
                                     center_joint=0, scale_by_torso=True)
    print(f"  Normalized shape (C,T,V,M): {normalized.shape}")

    # Back to (T, V, C) for body 0
    norm_body0 = normalized[:, :, :, 0].transpose(1, 2, 0)  # (C,T,V) -> (T,V,C)
    print(f"  Body 0 normalized shape: {norm_body0.shape}")

    print(f"\n  Generating normalized skeleton GIF...")
    norm_gif_path = os.path.join(output_dir, f'{sample_name}_normalized.gif')
    create_skeleton_gif(norm_body0, norm_gif_path,
                       title_prefix=f"Normalized — {sample_name}",
                       max_frames=args.max_frames, fps=args.fps)

    # ---- 3. Side-by-side comparison frame ----
    print(f"\n3. Generating comparison frame...")
    mid_frame = T // 2
    fig = plt.figure(figsize=(14, 6), facecolor='#0f0f23')

    ax1 = fig.add_subplot(121, projection='3d')
    draw_skeleton_frame(ax1, body0[mid_frame],
                       title=f"Raw — Frame {mid_frame+1}")

    ax2 = fig.add_subplot(122, projection='3d')
    draw_skeleton_frame(ax2, norm_body0[mid_frame],
                       title=f"Normalized — Frame {mid_frame+1}")

    plt.tight_layout()
    compare_path = os.path.join(output_dir, f'{sample_name}_comparison.png')
    fig.savefig(compare_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {compare_path}")

    # ---- 4. Stats ----
    print(f"\n4. Data Statistics:")
    print(f"  {'':20s} {'Raw':>12s}  {'Normalized':>12s}")
    print(f"  {'Mean':20s} {body0.mean():12.4f}  {norm_body0.mean():12.4f}")
    print(f"  {'Std':20s} {body0.std():12.4f}  {norm_body0.std():12.4f}")
    print(f"  {'Min':20s} {body0.min():12.4f}  {norm_body0.min():12.4f}")
    print(f"  {'Max':20s} {body0.max():12.4f}  {norm_body0.max():12.4f}")

    print(f"\n{'='*60}")
    print(f"✓ All outputs saved to: {output_dir}/")
    print(f"  • {sample_name}_raw.gif")
    print(f"  • {sample_name}_normalized.gif")
    print(f"  • {sample_name}_comparison.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
