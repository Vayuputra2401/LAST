import sys
import os
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import SkeletonDataset
from src.data.official_loader import read_skeleton_official, convert_official_to_numpy

def test_official_loader():
    print("Testing Official Loader...")
    # Create valid dummy skeleton file
    content = """5
    1
    72057594037927936 2 25 0.222 0.333 0.444 0.555 0.666 0.777 0.888 1.0 1.0 1.0"""
    # This is too complex to mock easily as text string due to the loop structure. 
    # Let's mock the file object or just trust the logic if we can't easily write a valid file.
    # Actually, let's write a minimal valid file.
    
    with open("test_skeleton.skeleton", "w") as f:
        f.write("2\n") # 2 frames
        
        # Frame 0
        f.write("1\n") # 1 body
        f.write("72057594037927936 0 0 0 0 0 0 0 0 0\n") # Body info
        f.write("2\n") # 2 joints (simplified)
        f.write("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0 0\n") # Joint 1
        f.write("1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 0 0\n") # Joint 2
        
        # Frame 1
        f.write("1\n") # 1 body
        f.write("72057594037927936 0 0 0 0 0 0 0 0 0\n") # Body info
        f.write("2\n") # 2 joints
        f.write("0.15 0.25 0.35 0.4 0.5 0.6 0.7 0.8 0.9 0 0\n") # Joint 1
        f.write("1.15 1.25 1.35 1.4 1.5 1.6 1.7 1.8 1.9 0 0\n") # Joint 2

    try:
        data = read_skeleton_official("test_skeleton.skeleton", njoints=2)
        print("✓ read_skeleton_official passed")
        
        numpy_data = convert_official_to_numpy(data, max_frames=5, max_bodies=1)
        # Shape: (C, T, V, M) -> (3, 5, 2, 1)
        print(f"  Shape: {numpy_data.shape}")
        assert numpy_data.shape == (3, 5, 2, 1)
        print("✓ convert_official_to_numpy passed")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists("test_skeleton.skeleton"):
            os.remove("test_skeleton.skeleton")

def test_dataset_mib():
    print("\nTesting Dataset (MIB Mode)...")
    # Mock .npy files
    data_path = "test_data_mib"
    os.makedirs(data_path, exist_ok=True)
    
    # Create dummy streams
    N = 10
    C, T, V, M = 3, 64, 25, 2
    joint = np.random.randn(N, C, T, V, M).astype(np.float32)
    velocity = np.random.randn(N, C, T, V, M).astype(np.float32)
    bone = np.random.randn(N, C, T, V, M).astype(np.float32)
    labels = [0] * N
    
    np.save(os.path.join(data_path, "train_joint.npy"), joint)
    np.save(os.path.join(data_path, "train_velocity.npy"), velocity)
    np.save(os.path.join(data_path, "train_bone.npy"), bone)
    import pickle
    with open(os.path.join(data_path, "train_label.pkl"), "wb") as f:
        pickle.dump(labels, f)
        
    try:
        from src.data.dataset import SkeletonDataset
        ds = SkeletonDataset(
            data_path=data_path,
            data_type='mib',
            split='train',
            split_type='.', # Current dir
            # transform=None
        )
        
        print(f"✓ Dataset loaded {len(ds)} samples")
        
        sample, label = ds[0]
        # Expect dict
        assert isinstance(sample, dict), "Sample should be dict"
        assert 'joint' in sample
        assert 'velocity' in sample
        assert 'bone' in sample
        
        print(f"  Joint Shape: {sample['joint'].shape}")
        assert sample['joint'].shape == (C, T, V, M)
        print("✓ MIB items correct")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        import shutil
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

if __name__ == "__main__":
    test_official_loader()
    test_dataset_mib()
