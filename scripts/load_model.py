"""
Load and inspect a LAST model variant.

Usage:
    python scripts/load_model.py --model base --dataset ntu60
    python scripts/load_model.py --model small --dataset ntu60
    python scripts/load_model.py --model base_e --dataset ntu60
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Load and inspect a LAST model')
    parser.add_argument('--model', type=str, default='base_e_v3',
                       choices=['nano_e_v3', 'small_e_v3', 'base_e_v3', 'large_e_v3'],
                       help='Model variant (default: base_e_v3)')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                       help='Dataset to configure num_classes (default: ntu60)')
    args = parser.parse_args()

    # Load full config (including model config)
    config = load_config(dataset=args.dataset, model=args.model)

    # Model Params
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)

    # Create model
    if args.model.endswith('_e_v3'):
        variant = args.model.replace('_e_v3', '')
        print(f"Creating LAST-E v3 model (Variant: {variant})...")
        from src.models.last_e_v3 import LAST_E_v3
        model = LAST_E_v3(
            num_classes=num_classes,
            variant=variant,
            dropout=config['model'].get('dropout', 0.3),
            drop_path_rate=config['model'].get('drop_path_rate'),
            use_st_att=config['model'].get('use_st_att'),
        )
        model_label = f"LAST-E-v3-{variant.capitalize()}"
    else:
        raise ValueError(f"Unsupported model variant: {args.model}")

    num_params = model.count_parameters()

    print("=" * 60)
    print(f"{model_label} | {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Classes:    {num_classes}")
    print(f"  Parameters: {num_params:,}")
    print()

    # Dummy forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # MIB input (dict)
    N, C, T, V = 2, 3, 64, 25
    x = {
        'joint':    torch.randn(N, C, T, V).to(device),
        'velocity': torch.randn(N, C, T, V).to(device),
        'bone':     torch.randn(N, C, T, V).to(device),
    }
    print(f"  Input:  MIB Dict (joint, velocity, bone) — each {tuple(x['joint'].shape)}")

    with torch.no_grad():
        out = model(x)

    print(f"  Output: {tuple(out.shape)}  (B, num_classes)")

    assert out.shape == (N, num_classes), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"

    print()
    print("Forward pass OK")
    print("=" * 60)


if __name__ == '__main__':
    main()
