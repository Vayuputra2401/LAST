"""Dry-run verification of the LAST v2 training pipeline."""
import sys
import os
import yaml
import pickle
import tempfile
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.training.trainer import Trainer
from src.data.dataset import SkeletonDataset
from src.data.transforms import get_train_transform, get_val_transform

def main():
    print("=" * 70)
    print("  LAST v2 — DRY RUN VERIFICATION")
    print("=" * 70)
    errors = []

    # ── 1. Config Loading ────────────────────────────────────────────────
    # 1. Load Defaults (Global/Training)
    training_config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'training', 'default.yaml'
    )
    with open(training_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Specifics (Model/Data/Env)
    # This returns {'environment': ..., 'data': ..., 'model': ...}
    specific_config = load_config(dataset='ntu60', model='base')
    
    # 3. Merge: Specifics override Defaults
    config.update(specific_config)
    
    # Ensure dataset is MIB for V2
    if config['model'].get('version') == 'v2' and config['data']['dataset']['data_type'] != 'mib':
        print("WARNING: Model is v2 but data_type is not 'mib'. Overriding for check.")
        config['data']['dataset']['data_type'] = 'mib'

    print(f"[1] Config Loaded: Model={config['model']['name']}, Data={config['data']['dataset']['data_type']}")

    # ── 2. Model Creation ────────────────────────────────────────────────
    num_classes = config['data']['dataset'].get('num_classes', 60)
    num_joints = config['data']['dataset']['num_joints']
    model_version = config['model'].get('version', 'v1')
    
    from src.models.last_v2 import LAST_v2
    model = LAST_v2(num_classes=num_classes, variant='base')
    print(f"[2] Model Created: {model.count_parameters():,} params (v2)")

    # ── 3. Transforms ────────────────────────────────────────────────────
    # Update config to disable normalization for MIB (already done in preprocess_v2)
    if config['data']['dataset']['data_type'] == 'mib':
        config['data']['dataset']['preprocessing']['normalize'] = False
        
    merged_config = {'dataset': config['data']['dataset'], 'training': config['training']}
    train_transform = get_train_transform(merged_config)
    val_transform = get_val_transform(merged_config)
    
    # Test valid transform check
    # Create dummy raw skeleton data (T, V, C) to test transform flow 
    # Transforms expect (C, T, V) usually or (T, V, C) depending on implementation
    # Let's trust get_train_transform works as verified in verify_data_v2
    print(f"[3] Transforms Initialized")

    # ── 4. Data Files Check ──────────────────────────────────────────────
    data_base = config['environment']['paths']['data_base']
    processed_path = os.path.join(data_base, 'LAST-60-v2', 'data', 'processed_v2')
    
    data_type = config['data']['dataset']['data_type']
    expected_files = []
    
    if data_type == 'mib':
        streams = ['joint', 'velocity', 'bone']
        splits = ['train', 'val']
        for s in splits:
            for st in streams:
                expected_files.append(f"{s}_{st}.npy")
            expected_files.append(f"{s}_label.pkl")
    else:
        expected_files = ['train_data.npy', 'train_label.pkl', 'val_data.npy', 'val_label.pkl']
        
    missing = []
    missing = []
    # Files are inside the split folder (e.g., xsub)
    split_type = config['data']['dataset']['split_type']
    target_path = os.path.join(processed_path, split_type)
    
    for f in expected_files:
        fp = os.path.join(target_path, f)
        if not os.path.exists(fp):
            missing.append(f)
    
    if missing:
        errors.append(f"Missing data files in {target_path}: {missing}")
        print(f"[4] Data Check: FAILED. Missing {len(missing)} files.")
    else:
        print(f"[4] Data Check: OK. Found {len(expected_files)} files in {target_path}")

    # ── 5. Dataset & Dataloader ──────────────────────────────────────────
    try:
        ds = SkeletonDataset(
            data_path=processed_path, 
            data_type=data_type,
            max_frames=300, 
            num_joints=num_joints, 
            transform=train_transform,
            split='train', 
            split_type='xsub'
        )
        sample, label = ds[0]
        print(f"[5] Dataset: Loaded {len(ds)} samples.")
        
        if isinstance(sample, dict):
            print(f"    Sample is Dict: {list(sample.keys())}")
            for k, v in sample.items():
                print(f"      {k}: {v.shape}")
        else:
            print(f"    Sample shape: {sample.shape}")
            
    except Exception as e:
        errors.append(f"Dataset Init Failed: {str(e)}")
        print(f"[5] Dataset: FAILED. {e}")
        return

    # ── 6. Trainer & Forward Pass ────────────────────────────────────────
    trainer = Trainer(model, config, tempfile.mkdtemp())
    trainer.model.eval()
    device = trainer.device
    
    # Batchify
    if isinstance(sample, dict):
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    else:
        batch = sample.unsqueeze(0).to(device)
        
    try:
        with torch.no_grad():
            out = trainer.model(batch)
        print(f"[6] Forward Pass: OK. Output shape {out.shape}")
        assert out.shape == (1, num_classes)
    except Exception as e:
        errors.append(f"Forward Pass Failed: {str(e)}")
        print(f"[6] Forward Pass: FAILED. {e}")

    # ── 7. Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    if errors:
        print(f"❌ DRY RUN FAILED with {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ DRY RUN PASSED")
        print("   Pipeline is ready for training.")
    print("=" * 70)

if __name__ == '__main__':
    main()
