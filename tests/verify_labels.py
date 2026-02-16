
import os
import sys

# Add project root to path (assuming script is run from project root or tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from tqdm import tqdm
from src.data.dataset import SkeletonDataset
from src.utils.config import load_config

def verify_labels():
    # Load config
    # Use load_config to get full environment + dataset config
    try:
        config = load_config(dataset='ntu60')
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    data_cfg = config['data']['dataset']
    env_cfg = config['environment']
    
    # Construct data path (same logic as train.py)
    data_base = env_cfg['paths']['data_base']
    # Assuming NTU-60 for this verification script
    folder_name = "LAST-60"
    processed_data_path = os.path.join(data_base, folder_name, "data", "processed")
    
    print(f"Verifying Dataset Label Range...")
    print(f"Data Path: {processed_data_path}")
    
    # Train Split
    try:
        print("\nLoading Train Dataset...")
        train_set = SkeletonDataset(
            data_path=processed_data_path,
            data_type='npy', # Force npy for verification
            split='train',
            split_type=data_cfg['split_type'],
            split_config=data_cfg.get('splits', None)
        )
        
        # Iterate to show progress and verify integrity
        print("Iterating through Train samples to verify integrity...")
        train_iter = iter(train_set)
        for _ in tqdm(range(len(train_set)), desc="Checking Train"):
            # inspect the first one
            pass
            
        # Check first sample stats
        sample0, _ = train_set[0]
        print(f"\n[Train] Sample 0 Stats:")
        print(f"  Shape: {sample0.shape}")
        print(f"  Mean: {sample0.mean():.4f}")
        print(f"  Min:  {sample0.min():.4f}")
        print(f"  Max:  {sample0.max():.4f}")
        
        if abs(sample0.mean()) < 0.5:
             print("✅ Train Data appears NORMALIZED (centered near 0).")
        else:
             print("⚠️ Train Data appears RAW (large offset).")

        
        labels = train_set.labels
        min_label = min(labels)
        max_label = max(labels)
        
        print(f"[Train] Min Label: {min_label}")
        print(f"[Train] Max Label: {max_label}")
        
        if min_label == 1:
            print("🚨 ALERT: Labels are 1-based (1-60)! They MUST be 0-based (0-59).")
        elif min_label == 0:
            print("✅ Labels appear to be 0-based (0-59).")
        else:
            print(f"❓ Unusual label range: {min_label}-{max_label}")
            
    except Exception as e:
        print(f"Failed to load Train set: {e}")

    # Val Split
    try:
        print("\nLoading Val Dataset...")
        val_set = SkeletonDataset(
            data_path=processed_data_path,
            data_type='npy',
            split='val',
            split_type=data_cfg['split_type'],
            split_config=data_cfg.get('splits', None)
        )
        
        # Iterate to show progress
        print("Iterating through Val samples to verify integrity...")
        for _ in tqdm(val_set, desc="Checking Val"):
            pass
        
        labels = val_set.labels
        min_label = min(labels)
        max_label = max(labels)
        
        print(f"[Val] Min Label: {min_label}")
        print(f"[Val] Max Label: {max_label}")
        
        if min_label == 1:
            print("🚨 ALERT: Validation Labels are 1-based!")
        elif min_label == 0:
            print("✅ Validation Labels are 0-based.")
            
    except Exception as e:
        print(f"Failed to load Val set: {e}")

if __name__ == "__main__":
    # Add project root to path
    sys.path.append(os.getcwd())
    verify_labels()
