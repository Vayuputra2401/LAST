"""
Load and inspect a LAST-Lite (ShiftFuse-GCN) model variant.

Usage:
    python scripts/load_model.py --model shiftfuse_small --dataset ntu60
    python scripts/load_model.py --model shiftfuse_nano --dataset ntu60
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Load and inspect a LAST-Lite model')
    parser.add_argument('--model', type=str, default='shiftfuse_small',
                       choices=['shiftfuse_nano', 'shiftfuse_small'],
                       help='Model variant (default: shiftfuse_small)')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                       help='Dataset to configure num_classes (default: ntu60)')
    args = parser.parse_args()

    # Load full config (including model config)
    config = load_config(dataset=args.dataset, model=args.model)

    # Model Params
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)
    num_joints = config['data']['dataset']['num_joints']
    T = config['data']['dataset']['max_frames']

    # Create model
    variant = args.model.replace('shiftfuse_', '')
    print(f"Creating LAST-Lite / ShiftFuse-GCN (Variant: {variant}, T={T})...")
    from src.models.shiftfuse_gcn import LAST_Lite
    model = LAST_Lite(
        num_classes=num_classes,
        variant=variant,
        T=T,
        num_joints=num_joints,
        dropout=config['model'].get('dropout'),
    )
    model_label = f"LAST-Lite-{variant.capitalize()}"

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

    # 4-stream input (dict)
    N, C, T_in, V = 2, 3, T, 25
    x = {
        'joint':         torch.randn(N, C, T_in, V).to(device),
        'velocity':      torch.randn(N, C, T_in, V).to(device),
        'bone':          torch.randn(N, C, T_in, V).to(device),
        'bone_velocity': torch.randn(N, C, T_in, V).to(device),
    }
    print(f"  Input:  4-stream Dict (joint, velocity, bone, bone_velocity) — each {tuple(x['joint'].shape)}")

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
