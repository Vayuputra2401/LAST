"""
Load and inspect a LAST model variant.

Usage:
    python scripts/load_model.py --model base --dataset ntu60
    python scripts/load_model.py --model small --dataset ntu60
    python scripts/load_model.py --model large --dataset ntu120
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch

from src.models.last import create_last_base, create_last_small, create_last_large
from src.utils.config import load_config


MODEL_CREATORS = {
    'base': create_last_base,
    'small': create_last_small,
    'large': create_last_large,
}


def main():
    parser = argparse.ArgumentParser(description='Load and inspect a LAST model')
    parser.add_argument('--model', type=str, default='base', choices=['base', 'small', 'large'],
                       help='Model variant (default: base)')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                       help='Dataset to configure num_classes (default: ntu60)')
    args = parser.parse_args()

    # Load full config (including model config)
    config = load_config(dataset=args.dataset, model=args.model)
    
    # Model Params
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)
    num_joints = config['data']['dataset']['num_joints']
    
    # Create model based on version
    model_version = config.get('model', {}).get('version', 'v1')
    
    if model_version == 'v2':
        print(f"Creating LAST v2 model (Variant: {args.model})...")
        from src.models.last_v2 import LAST_v2
        model = LAST_v2(num_classes=num_classes, variant=args.model)
    else:
        print(f"Creating LAST v1 model (Variant: {args.model})...")
        create_fn = MODEL_CREATORS[args.model]
        model = create_fn(num_classes=num_classes, num_joints=num_joints)

    num_params = model.count_parameters()

    print("=" * 60)
    print(f"LAST-{args.model.capitalize()} ({model_version.upper()}) | {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Classes:    {num_classes}")
    print(f"  Joints:     {num_joints}")
    if hasattr(model, 'channels'):
        print(f"  Channels:   {model.channels}")
    print(f"  Parameters: {num_params:,}")
    print()

    # Dummy forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Prepare input based on version
    if model_version == 'v2':
        # MIB Input (Dict)
        N, C, T, V, M = 2, 3, 64, 25, 2
        x = {
            'joint': torch.randn(N, C, T, V, M).to(device),
            'velocity': torch.randn(N, C, T, V, M).to(device),
            'bone': torch.randn(N, C, T, V, M).to(device)
        }
        print(f"  Input:      MIB Dict (Joint, Vel, Bone) -> Each {tuple(x['joint'].shape)}")
    else:
        # Standard Input
        x = torch.randn(2, 3, 64, num_joints, 2).to(device)
        print(f"  Input:      {tuple(x.shape)}  (B, C, T, V, M)")

    with torch.no_grad():
        out = model(x)
        
    print(f"  Output:     {tuple(out.shape)}  (B, num_classes)")

    assert out.shape == (2, num_classes), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"

    print()
    print("✓ Forward pass OK")
    print("=" * 60)


if __name__ == '__main__':
    main()
