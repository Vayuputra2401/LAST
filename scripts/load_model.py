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

    # Load dataset config to get num_classes
    config = load_config(dataset=args.dataset)
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)
    num_joints = config['data']['dataset']['num_joints']

    # Create model
    create_fn = MODEL_CREATORS[args.model]
    model = create_fn(num_classes=num_classes, num_joints=num_joints)

    num_params = model.count_parameters()

    print("=" * 60)
    print(f"LAST-{args.model.capitalize()} | {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Classes:    {num_classes}")
    print(f"  Joints:     {num_joints}")
    print(f"  Channels:   {model.channels}")
    print(f"  Parameters: {num_params:,}")
    print()

    # Dummy forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    x = torch.randn(2, 3, 64, num_joints, 2).to(device)
    with torch.no_grad():
        out = model(x)

    print(f"  Input:      {tuple(x.shape)}  (B, C, T, V, M)")
    print(f"  Output:     {tuple(out.shape)}  (B, num_classes)")

    assert out.shape == (2, num_classes), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"

    print()
    print("✓ Forward pass OK")
    print("=" * 60)


if __name__ == '__main__':
    main()
