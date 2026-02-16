"""
Teacher-LAST: Fine-tuning Entry Point
=======================================
Main script for fine-tuning VideoMAE-2 Large on NTU RGB+D 60.

Usage:
    # Default (1 GPU, standard LR scaling)
    python scripts/finetune.py --config configs/ntu60_finetune.yaml

    # 2 GPUs
    python scripts/finetune.py --config configs/ntu60_finetune.yaml --num_gpus 2

    # Dry run (2 epochs, small batch)
    python scripts/finetune.py --config configs/ntu60_finetune.yaml --epochs 2 --batch_size 2 --debug

    # Resume from checkpoint
    python scripts/finetune.py --config configs/ntu60_finetune.yaml
    # (auto-resume enabled by default in config)

LR Scaling:
    actual_lr = base_lr * (batch_size * num_gpus * update_freq) / 256
    
    With default config (base_lr=0.001, batch=8, update_freq=4):
      --num_gpus 1 → actual_lr = 1.25e-4
      --num_gpus 2 → actual_lr = 2.5e-4
      --num_gpus 4 → actual_lr = 5e-4
"""

import os
import sys
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config, set_seed, setup_logging, compute_actual_lr
from src.dataset import build_dataloader
from src.model import build_model
from src.trainer import train


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune VideoMAE-2 Large on NTU RGB+D 60',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/finetune.py --config configs/ntu60_finetune.yaml
  python scripts/finetune.py --config configs/ntu60_finetune.yaml --num_gpus 2
  python scripts/finetune.py --config configs/ntu60_finetune.yaml --epochs 2 --debug
        """
    )

    # Required
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file (e.g., configs/ntu60_finetune.yaml)'
    )

    # GPU & LR
    parser.add_argument(
        '--num_gpus', type=int, default=1,
        help='Number of GPUs. LR scales via: base_lr * (batch*gpus*accum)/256 (default: 1)'
    )

    # Training overrides
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override per-GPU batch size'
    )
    parser.add_argument(
        '--base_lr', type=float, default=None,
        help='Override base learning rate'
    )
    parser.add_argument(
        '--update_freq', type=int, default=None,
        help='Override gradient accumulation steps'
    )

    # Debugging
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode: reduce workers, enable extra logging'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    return parser.parse_args()


def apply_overrides(config, args):
    """
    Apply CLI argument overrides to the configuration.
    
    CLI flags take precedence over YAML config values.
    
    Args:
        config: Configuration dictionary loaded from YAML
        args: Parsed command-line arguments
    """
    # Store num_gpus in config for the trainer
    config['num_gpus'] = args.num_gpus

    # Training overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"[Override] epochs = {args.epochs}")

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"[Override] batch_size = {args.batch_size}")

    if args.base_lr is not None:
        config['training']['base_lr'] = args.base_lr
        print(f"[Override] base_lr = {args.base_lr}")

    if args.update_freq is not None:
        config['training']['update_freq'] = args.update_freq
        print(f"[Override] update_freq = {args.update_freq}")

    # Debug mode: reduce data loading overhead
    if args.debug:
        config['data']['num_workers'] = 0
        config['data']['pin_memory'] = False
        print(f"[Override] Debug mode: num_workers=0, pin_memory=False")


def main():
    """Main entry point for fine-tuning."""

    # =========================================================================
    # 1. Parse Arguments & Load Config
    # =========================================================================
    args = parse_args()

    print("=" * 60)
    print("Teacher-LAST: VideoMAE-2 Fine-tuning on NTU RGB+D 60")
    print("=" * 60)

    config = load_config(args.config)
    apply_overrides(config, args)
    set_seed(args.seed)

    # =========================================================================
    # 2. Print Configuration Summary
    # =========================================================================
    training_config = config['training']
    actual_lr = compute_actual_lr(
        base_lr=training_config['base_lr'],
        batch_size=training_config['batch_size'],
        num_gpus=args.num_gpus,
        update_freq=training_config.get('update_freq', 1),
    )

    print(f"\n--- Configuration Summary ---")
    print(f"  Dataset:      {config['data']['dataset_name']}")
    print(f"  Classes:      {config['data']['num_classes']}")
    print(f"  Model:        {config['model']['name']}")
    print(f"  Pretrained:   {config['model']['pretrained_path']}")
    print(f"  Epochs:       {training_config['epochs']}")
    print(f"  Batch size:   {training_config['batch_size']}")
    print(f"  Num GPUs:     {args.num_gpus}")
    print(f"  Update freq:  {training_config.get('update_freq', 1)}")
    print(f"  Actual LR:    {actual_lr:.6f}")
    print(f"  AMP:          {config['hardware'].get('mixed_precision', True)}")
    print(f"  Checkpoints:  {config['checkpoint']['output_dir']}")
    print(f"---\n")

    # =========================================================================
    # 3. Build Data Loaders
    # =========================================================================
    print("[Data] Building training dataloader...")
    train_loader, train_dataset = build_dataloader(config, mode='train')

    print("[Data] Building validation dataloader...")
    val_loader, val_dataset = build_dataloader(config, mode='validation')

    # =========================================================================
    # 4. Build Model
    # =========================================================================
    print("\n[Model] Building VideoMAE-2 Large...")
    model = build_model(config)

    # =========================================================================
    # 5. Setup Logging
    # =========================================================================
    writer = setup_logging(
        log_dir=config['checkpoint']['log_dir'],
        experiment_name="teacher-last"
    )

    # =========================================================================
    # 6. Train
    # =========================================================================
    try:
        best_acc = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            writer=writer,
        )
        print(f"\nFinal Best Accuracy: {best_acc:.2f}%")

    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Checkpoints saved.")

    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
