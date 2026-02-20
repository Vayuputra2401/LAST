import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from src.utils.config import load_config
from src.data.skeleton_loader import SkeletonFileParser
from src.data.official_loader import read_skeleton_official, convert_official_to_numpy
from src.data.normalization import align_to_spine_base, rotate_to_front, normalize_skeleton_scale

# NTU RGB+D Joint Labels (0-based):
# 0: base of spine   1: mid spine     2: neck           3: head
# 4: left shoulder   5: left elbow    6: left wrist     7: left hand
# 8: right shoulder  9: right elbow  10: right wrist   11: right hand
# 12: left hip      13: left knee    14: left ankle    15: left foot
# 16: right hip     17: right knee   18: right ankle   19: right foot
# 20: spine shoulder 21: left hand tip 22: left thumb  23: right hand tip  24: right thumb

# Pairs (Child, Parent) -- used for bone vector generation:
ntu_pairs_0_based = [
    (1, 0), (20, 1), (2, 20), (3, 2), # Spine
    (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 6), # Left Arm
    (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 10), # Right Arm
    (12, 0), (13, 12), (14, 13), (15, 14), # Left Leg
    (16, 0), (17, 16), (18, 17), (19, 18)  # Right Leg
]


def data_generator(file_list, parser, config, max_frames=300):
    for file_path, label in file_list:
        try:
            # Parse with OFFICIAL logic
            # data, _ = parser.parse_file(file_path) # Old way
            bodymat = read_skeleton_official(file_path, max_body=4, njoints=25)
            data = convert_official_to_numpy(bodymat, max_frames=max_frames, max_bodies=2)
            
            if data is None:
                yield None, None
                continue
            
            # Data is already (C, T, V, M) from converter
            
            # Apply Normalization
            data = align_to_spine_base(data)
            data = rotate_to_front(data)
            data = normalize_skeleton_scale(data) 
            
            yield data, label
        except Exception as e:
            print(f"Error {file_path}: {e}")
            yield None, None

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
    N, C, T, V, M = joint_data.shape
    velocity_data = np.zeros_like(joint_data)
    
    # V[:, :, 0:-1] = J[:, :, 1:] - J[:, :, 0:-1]
    # Keep output shape same
    velocity_data[:, :, :-1, :, :] = joint_data[:, :, 1:, :, :] - joint_data[:, :, :-1, :, :]
    
    return velocity_data

def collect_files(data_path, split_config, split_type, split):
    # Reuse logic from preprocess_data.py but simplified
    # Assuming the user runs it correctly
    # For now, let's just copy the logic or import it?
    # Better to copy to keep this standalone V2 script.
    import glob
    files = glob.glob(os.path.join(data_path, '**/*.skeleton'), recursive=True)
    
    pairs = []
    # Simplified filtering (assuming split_config is passed correctly)
    # ... (Logic from preprocess_data.py) ...
    # This part is verbose. Let's trust preprocess_data.py logic and maybe import `collect_files_for_split`?
    # Yes, from scripts.preprocess_data import collect_files_for_split
    # FIX: Use absolute import path instead of relative scripts.* import
    # which fails when the script is run from a different working directory.
    import importlib, sys, os as _os
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from scripts.preprocess_data import collect_files_for_split
    return collect_files_for_split(data_path, split, split_type, split_config, SkeletonFileParser(25, 2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ntu60')
    parser.add_argument('--split_type', default='xsub')
    args = parser.parse_args()
    
    config = load_config(dataset=args.dataset)
    split_config = config['data']['dataset']['splits']
    data_root = config['environment']['paths']['data_root']
    
    # Output Dir
    # User requested change to LAST-60-v2
    out_dir = Path(config['environment']['paths']['data_base']) / "LAST-60-v2" / "data" / "processed_v2" / args.split_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val']
    
    for split in splits:
        print(f"Processing {split}...")
        
        # 1. Collect Files
        parser_obj = SkeletonFileParser(25, 2)
        pairs = collect_files(data_root, split_config, args.split_type, split)
        # pairs = pairs[:100] # DEBUG
        
        # 2. Generate Joint Stream
        print("Generating Joint Stream (using Official Parser)...")
        joints = []
        labels = []
        
        # Generator
        gen = data_generator(pairs, parser_obj, config)
        
        for data, label in tqdm(gen, total=len(pairs)):
            if data is not None:
                joints.append(data)
                labels.append(label)
        
        joints = np.array(joints, dtype=np.float32) # (N, C, T, V, M)
        print(f"Joints Shape: {joints.shape}")
        
        # 3. Generate Other Streams
        print("Generating Velocity Stream...")
        velocity = gen_velocity_data(joints)
        
        print("Generating Bone Stream...")
        bone = gen_bone_data(joints)
        
        # 4. Save
        print(f"Saving {split} to {out_dir}...")
        np.save(out_dir / f"{split}_joint.npy", joints)
        np.save(out_dir / f"{split}_velocity.npy", velocity)
        np.save(out_dir / f"{split}_bone.npy", bone)
        with open(out_dir / f"{split}_label.pkl", 'wb') as f:
            pickle.dump(labels, f)
            
    print("Done.")

if __name__ == '__main__':
    main()
