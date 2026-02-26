"""
Offline Data Preprocessing Script

Converts .skeleton files to normalized .npy format for faster training.

This script:
1. Loads all .skeleton files
2. Applies normalization (center spine, scale by torso)
3. Saves as memory-mapped .npy files
4. Saves labels as pickle files

Run once, train fast forever!

Usage:
    python scripts/preprocess_data.py --dataset ntu60
    python scripts/preprocess_data.py --dataset ntu60 --split_type xview
    python scripts/preprocess_data.py --dataset ntu120 --split_type xset
    python scripts/preprocess_data.py --max_samples 1000  # For testing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from src.utils.config import load_config
from src.data.skeleton_loader import SkeletonFileParser
from src.data.preprocessing import normalize_skeleton

# Pairs (Child, Parent) -- used for bone vector generation:
ntu_pairs_0_based = [
    (1, 0), (20, 1), (2, 20), (3, 2), # Spine
    (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6), # Left Arm
    (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10), # Right Arm
    (12, 0), (13, 12), (14, 13), (15, 14), # Left Leg
    (16, 0), (17, 16), (18, 17), (19, 18)  # Right Leg
]


def gen_bone_data(joint_data):
    """
    Generate bone data from joint data.
    Args:
        joint_data: (N, C, T, V, M)
    Returns:
        bone_data: (N, C, T, V, M)
    """
    N, C, T, V, M = joint_data.shape
    bone_data = np.zeros_like(joint_data)
    
    for v1, v2 in ntu_pairs_0_based:
        # Vector from v2 (Parent) to v1 (Child)
        if v1 < V and v2 < V:
             bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]
             
    # For root (0), bone is 0 (or connect to itself)
    return bone_data

def gen_velocity_data(joint_data):
    """
    Generate velocity data.
    V_t = J_{t+1} - J_t
    """
    velocity_data = np.zeros_like(joint_data)
    velocity_data[:, :, :-1, :, :] = joint_data[:, :, 1:, :, :] - joint_data[:, :, :-1, :, :]
    
    return velocity_data



def process_single_file(file_info, parser, config, max_frames=300):
    """
    Process a single skeleton file.
    
    Args:
        file_info: (file_path, label) tuple
        parser: SkeletonFileParser instance
        config: Configuration dict
        max_frames: Maximum number of frames
        
    Returns:
        Normalized skeleton array (C, T, V, M) or None if error
    """
    file_path, label = file_info
    
    try:
        # Parse skeleton file
        skeleton_data, metadata = parser.parse_file(file_path)
        # skeleton_data shape: (T, V, C, M)
        
        T, V, C, M = skeleton_data.shape
        
        # Temporal repeat-pad / uniform subsample to max_frames (EfficientGCN style)
        if T < max_frames:
            # Repeat-pad (copy last frame) to avoid velocity spikes
            padded = np.zeros((max_frames, V, C, M), dtype=np.float32)
            padded[:T] = skeleton_data
            for t_idx in range(T, max_frames):
                padded[t_idx] = skeleton_data[-1]  # copy last frame
            skeleton_data = padded
        elif T > max_frames:
            # Uniformly subsample
            indices = np.linspace(0, T - 1, max_frames, dtype=int)
            skeleton_data = skeleton_data[indices]
        
        # Transpose to (C, T, V, M)
        skeleton_data = skeleton_data.transpose(2, 0, 1, 3)
        
        # Normalize if enabled
        if config['dataset']['preprocessing']['normalize']:
            skeleton_data = normalize_skeleton(
                skeleton_data,
                method=config['dataset']['preprocessing']['normalization_method'],
                center_joint=config['dataset']['preprocessing']['center_joint'],
                scale_by_torso=config['dataset']['preprocessing']['scale_by_torso']
            )
        
        return skeleton_data.astype(np.float32)
        
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return None


def collect_files_for_split(data_path, split, split_type, split_config, parser):
    """
    Collect all files for a given split.
    
    Returns:
        List of (file_path, label) tuples
    """
    import glob
    
    # Find all skeleton files
    skeleton_files = glob.glob(os.path.join(data_path, '*.skeleton'))
    if len(skeleton_files) == 0:
        skeleton_files = glob.glob(os.path.join(data_path, '**', '*.skeleton'), recursive=True)
    
    file_label_pairs = []
    
    for file_path in skeleton_files:
        metadata = parser.extract_metadata_from_filename(file_path)
        
        # Check if file belongs to this split
        should_include = False
        
        if split_type == 'xsub':
            train_subjects = set(split_config['xsub']['train_subjects'])
            val_subjects = set(split_config['xsub']['val_subjects'])
            person = metadata['person']
            
            if split == 'train' and person in train_subjects:
                should_include = True
            elif split == 'val' and person in val_subjects:
                should_include = True
        
        elif split_type == 'xset':
            train_setups = set(split_config['xset']['train_setups'])
            val_setups = set(split_config['xset']['val_setups'])
            setup = metadata['setup']
            
            if split == 'train' and setup in train_setups:
                should_include = True
            elif split == 'val' and setup in val_setups:
                should_include = True
        
        elif split_type == 'xview':
            train_cameras = set(split_config['xview']['train_cameras'])
            val_cameras = set(split_config['xview']['val_cameras'])
            camera = metadata['camera']
            
            if split == 'train' and camera in train_cameras:
                should_include = True
            elif split == 'val' and camera in val_cameras:
                should_include = True
        
        if should_include:
            label = metadata['action']
            file_label_pairs.append((file_path, label))
    
    return file_label_pairs


def preprocess_split(config, split='train', split_type='xsub', max_samples=None):
    """
    Preprocess a single split (train or val).
    
    Args:
        config: Configuration dict
        split: 'train' or 'val'
        split_type: 'xsub' or 'xset'
        max_samples: Maximum samples to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing {split.upper()} split ({split_type})")
    print(f"{'='*60}")
    
    # Create parser
    parser = SkeletonFileParser(
        num_joints=config['data']['dataset']['num_joints'],
        max_bodies=config['data']['dataset']['max_bodies']
    )
    
    # Get data path
    data_path = config['environment']['paths']['data_root']
    
    # Collect files
    print(f"Scanning files in {data_path}...")
    file_label_pairs = collect_files_for_split(
        data_path,
        split,
        split_type,
        config['data']['dataset']['splits'],
        parser
    )
    
    if max_samples:
        file_label_pairs = file_label_pairs[:max_samples]
    
    num_samples = len(file_label_pairs)
    print(f"Found {num_samples} samples for {split} split")
    
    if num_samples == 0:
        print(f"Warning: No samples found for {split} split!")
        return
    
    # Prepare output
    max_frames = config['data']['dataset']['max_frames']
    num_joints = config['data']['dataset']['num_joints']
    max_bodies = config['data']['dataset']['max_bodies']
    
    # Preallocate array
    print(f"Allocating array: ({num_samples}, 3, {max_frames}, {num_joints}, {max_bodies})")
    data_array = np.zeros((num_samples, 3, max_frames, num_joints, max_bodies), dtype=np.float32)
    labels = []
    
    # Process files with progress bar
    print(f"Processing {num_samples} files...")
    
    successful = 0
    failed = 0
    
    for i, (file_path, label) in enumerate(tqdm(file_label_pairs, desc=f"{split} split")):
        skeleton = process_single_file((file_path, label), parser, config['data'], max_frames)
        
        if skeleton is not None:
            data_array[i] = skeleton
            labels.append(label)
            successful += 1
        else:
            # Keep zero-filled for failed samples
            labels.append(-1)  # Invalid label
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Remove failed samples
    if failed > 0:
        valid_mask = np.array(labels) != -1
        data_array = data_array[valid_mask]
        labels = [l for l in labels if l != -1]
        print(f"  After filtering: {len(labels)} valid samples")
    
    # Generate streams (HI-GCN / EfficientGCN style)
    print("Generating Velocity Stream...")
    velocity_array = gen_velocity_data(data_array)
    
    print("Generating Bone Stream...")
    bone_array = gen_bone_data(data_array)
    
    print("Generating Bone Velocity Stream...")
    bone_velocity_array = gen_velocity_data(bone_array)
    
    # Save to disk
    output_dir = Path(config['environment']['paths']['processed_data']) / split_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving data to {output_dir}...")
    np.save(output_dir / f"{split}_joint.npy", data_array)
    np.save(output_dir / f"{split}_velocity.npy", velocity_array)
    np.save(output_dir / f"{split}_bone.npy", bone_array)
    np.save(output_dir / f"{split}_bone_velocity.npy", bone_velocity_array)
    
    label_file = output_dir / f"{split}_label.pkl"
    print(f"Saving labels to {label_file}...")
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)
    
    # Print statistics
    joint_file = output_dir / f"{split}_joint.npy"
    file_size_mb = joint_file.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"{split.upper()} SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"  Samples: {len(labels)}")
    print(f"  Shape: {data_array.shape}")
    print(f"  Size (Joint): {file_size_mb:.1f} MB")
    print(f"  Output Directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess skeleton data for faster training')
    parser.add_argument('--env', type=str, default=None, help='Environment (auto-detect if not specified)')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                       help='Dataset config name (default: ntu60)')
    parser.add_argument('--split_type', type=str, default='xsub', choices=['xsub', 'xset', 'xview'],
                       help='Split type (xsub, xset, or xview)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                       choices=['train', 'val'],
                       help='Which splits to preprocess')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SKELETON DATA PREPROCESSING")
    print("="*60)
    print("\nThis script will:")
    print("  1. Load raw .skeleton files")
    print("  2. Apply normalization (center spine, scale torso)")
    print("  3. Save as .npy files for fast loading")
    print("\n" + "="*60 + "\n")
    
    # Load config
    print(f"Loading configuration...")
    config = load_config(env=args.env, dataset=args.dataset)
    
    # Override split_type if specified
    config['data']['dataset']['split_type'] = args.split_type
    
    env_name = config['environment']['environment']['name']
    data_path = config['environment']['paths']['data_root']
    
    # Build output path dynamically: LAST-60 for ntu60, LAST-120 for ntu120
    data_base = config['environment']['paths']['data_base']
    folder_name = "LAST-60" if args.dataset == 'ntu60' else "LAST-120"
    output_path = os.path.join(data_base, folder_name, "data", "processed")
    
    # Override in config so preprocess_split uses correct path
    config['environment']['paths']['processed_data'] = output_path
    
    print(f"✓ Environment: {env_name}")
    print(f"✓ Dataset: {args.dataset}")
    print(f"✓ Split type: {args.split_type}")
    print(f"✓ Source: {data_path}")
    print(f"✓ Destination: {output_path}/{args.split_type}/")
    
    if args.max_samples:
        print(f"✓ Testing mode: {args.max_samples} samples per split")
    
    # Process each split
    for split in args.splits:
        preprocess_split(config, split=split, split_type=args.split_type, max_samples=args.max_samples)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nTo use preprocessed data, update configs/data/{args.dataset}.yaml:")
    print("  dataset:")
    print("    data_type: 'npy'  # Change from 'skeleton'")
    print("\nThen training will be 10x faster!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
