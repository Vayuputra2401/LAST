"""
Teacher-LAST: NTU RGB+D 60 Annotation Generator
=================================================
Scans the NTU RGB+D video folders and generates train/val CSV files
using the official cross-subject split.

Usage:
    python scripts/prepare_annotations.py \
        --video_root "E:\nturgbd-videos" \
        --output_dir "E:\teacher-last\annotations"

Output:
    train.csv  — Training videos with labels
    val.csv    — Validation videos with labels

CSV Format:
    <absolute_path_to_video> <label_id>
    E:\nturgbd-videos\nturgbd_rgb_s001\nturgb+d_rgb\S001C001P001R001A001_rgb.avi 0
    E:\nturgbd-videos\nturgbd_rgb_s001\nturgb+d_rgb\S001C001P001R001A002_rgb.avi 1

NTU RGB+D 60 Filename Format:
    SsssCcccPpppRrrrAaaa_rgb.avi
    - S: Setup number (001-017)
    - C: Camera ID (001-003)
    - P: Performer/Subject ID (001-040)
    - R: Replication number (001-002)
    - A: Action class (001-060)

Cross-Subject Split (NTU-60):
    Train subjects: {1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38}
    Val subjects:   {3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40}
"""

import os
import glob
import argparse


# Official NTU RGB+D 60 cross-subject split
# Reference: "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"
TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
VAL_SUBJECTS = {3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40}


def parse_ntu_filename(filename):
    """
    Parse NTU RGB+D filename to extract metadata.
    
    Filename format: SsssCcccPpppRrrrAaaa_rgb.avi
    
    Args:
        filename: Video filename (e.g., 'S001C001P001R001A001_rgb.avi')
    
    Returns:
        dict: {setup, camera, subject, replication, action} or None if invalid
    """
    # Get just the filename without extension
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # Remove '_rgb' suffix if present
    if basename.endswith('_rgb'):
        basename = basename[:-4]

    try:
        # Parse: SsssCcccPpppRrrrAaaa
        setup = int(basename[1:4])           # S001 → 1
        camera = int(basename[5:8])          # C001 → 1
        subject = int(basename[9:12])        # P001 → 1
        replication = int(basename[13:16])   # R001 → 1
        action = int(basename[17:20])        # A001 → 1

        return {
            'setup': setup,
            'camera': camera,
            'subject': subject,
            'replication': replication,
            'action': action,
        }
    except (ValueError, IndexError):
        return None


def discover_videos(video_root):
    """
    Discover all NTU RGB+D 60 video files.
    
    Expected directory structure:
        E:\nturgbd-videos\
        ├── nturgbd_rgb_s001\nturgb+d_rgb\*.avi
        ├── nturgbd_rgb_s002\nturgb+d_rgb\*.avi
        ...
        └── nturgbd_rgb_s017\nturgb+d_rgb\*.avi
    
    Args:
        video_root: Root directory containing setup folders
    
    Returns:
        list: List of (video_path, metadata_dict) tuples
    """
    videos = []

    # Search for .avi files in all setup folders
    patterns = [
        os.path.join(video_root, '**', '*.avi'),
        os.path.join(video_root, '**', '*.mp4'),
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    print(f"[Annotations] Found {len(all_files)} video files in {video_root}")

    for video_path in sorted(all_files):
        metadata = parse_ntu_filename(video_path)
        if metadata is not None:
            # Only include NTU-60 actions (A001-A060)
            if 1 <= metadata['action'] <= 60:
                videos.append((video_path, metadata))

    print(f"[Annotations] Parsed {len(videos)} valid NTU-60 videos")

    return videos


def split_train_val(videos):
    """
    Split videos into train/val using the official cross-subject protocol.
    
    Args:
        videos: List of (video_path, metadata) tuples
    
    Returns:
        tuple: (train_list, val_list) — each is a list of (path, label_id)
    """
    train_list = []
    val_list = []

    for video_path, meta in videos:
        # Label ID: 0-indexed (action 1 → label 0)
        label_id = meta['action'] - 1
        subject = meta['subject']

        if subject in TRAIN_SUBJECTS:
            train_list.append((video_path, label_id))
        elif subject in VAL_SUBJECTS:
            val_list.append((video_path, label_id))
        else:
            print(f"  WARNING: Unknown subject {subject} in {video_path}")

    return train_list, val_list


def write_csv(data_list, output_path):
    """
    Write annotation CSV file.
    
    Format: <absolute_video_path> <label_id>
    
    Args:
        data_list: List of (video_path, label_id) tuples
        output_path: Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for video_path, label_id in data_list:
            f.write(f"{video_path} {label_id}\n")

    print(f"[Annotations] Wrote {len(data_list)} entries → {output_path}")


def print_statistics(train_list, val_list):
    """Print dataset split statistics."""
    print(f"\n{'='*50}")
    print(f"NTU RGB+D 60 — Cross-Subject Split")
    print(f"{'='*50}")
    print(f"  Train videos: {len(train_list)}")
    print(f"  Val videos:   {len(val_list)}")
    print(f"  Total:        {len(train_list) + len(val_list)}")

    # Count classes
    train_classes = set(label for _, label in train_list)
    val_classes = set(label for _, label in val_list)

    print(f"  Train classes: {len(train_classes)}")
    print(f"  Val classes:   {len(val_classes)}")

    # Per-class distribution
    print(f"\n  Per-class counts (first 10):")
    from collections import Counter
    train_counts = Counter(label for _, label in train_list)
    val_counts = Counter(label for _, label in val_list)

    for action_id in sorted(train_counts.keys())[:10]:
        print(f"    Action {action_id:02d}: train={train_counts[action_id]:4d}, "
              f"val={val_counts.get(action_id, 0):4d}")
    print(f"    ...")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate NTU RGB+D 60 train/val annotation CSV files'
    )
    parser.add_argument(
        '--video_root', type=str, required=True,
        help='Root directory containing NTU video folders (e.g., E:\\nturgbd-videos)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save train.csv and val.csv (e.g., E:\\teacher-last\\annotations)'
    )
    args = parser.parse_args()

    # 1. Discover all videos
    videos = discover_videos(args.video_root)

    if len(videos) == 0:
        print("ERROR: No valid NTU-60 videos found!")
        print(f"  Searched in: {args.video_root}")
        print(f"  Expected structure: <video_root>/nturgbd_rgb_s001/nturgb+d_rgb/*.avi")
        return

    # 2. Split into train/val
    train_list, val_list = split_train_val(videos)

    # 3. Write CSV files
    write_csv(train_list, os.path.join(args.output_dir, 'train.csv'))
    write_csv(val_list, os.path.join(args.output_dir, 'val.csv'))

    # 4. Print statistics
    print_statistics(train_list, val_list)

    print("Done! Annotations ready for training.")


if __name__ == '__main__':
    main()
