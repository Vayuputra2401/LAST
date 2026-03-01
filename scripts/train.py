"""
LAST Training Script

Usage:
    python scripts/train.py --model base_e_v3 --dataset ntu60
    python scripts/train.py --model small_e_v3 --dataset ntu60 --epochs 5 --batch_size 8
    python scripts/train.py --model base --dataset ntu60 --resume E:\\LAST-runs\\run-...\\checkpoints\\best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from src.data.dataset import SkeletonDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.training.trainer import Trainer
from src.utils.config import load_config, set_nested_key


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions for speed


def count_flops(model, input_shape, device):
    """
    Estimate FLOPs using a forward pass with profiling.
    
    Args:
        model: LAST model
        input_shape: tuple (C, T, V, M)
        device: torch device
    
    Returns:
        dict with flops, params, memory info
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(1, *input_shape).to(device)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size in MB
    param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size_mb = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    
    # FLOPs estimation using torch profiler
    flops = 0
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            model(x)
        events = prof.key_averages()
        for evt in events:
            if evt.flops:
                flops += evt.flops
    except Exception:
        # Fallback: rough estimate based on model size
        flops = total_params * 2  # Very rough approximation
    
    # GPU memory usage (if CUDA)
    gpu_memory_mb = 0
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': round(param_size_mb, 2),
        'buffer_size_mb': round(buffer_size_mb, 2),
        'total_model_size_mb': round(param_size_mb + buffer_size_mb, 2),
        'flops': flops,
        'gflops': round(flops / 1e9, 3),
        'gpu_memory_mb': round(gpu_memory_mb, 2),
    }


def _run_checkpoint_averaging(model, val_loader, run_dir, n_checkpoints, device):
    """
    Average the last N epoch checkpoints and evaluate on val_loader.
    Prints the averaged checkpoint top-1/top-5 accuracy.
    """
    import glob as _glob
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    # Find epoch checkpoints (exclude best_model.pth and final_model.pth)
    ckpt_paths = sorted(
        _glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*.pth')),
        key=lambda p: int(os.path.basename(p).replace('checkpoint_ep', '').replace('.pth', ''))
    )
    if len(ckpt_paths) == 0:
        print('\n[avg_checkpoints] No epoch checkpoints found — skipping.')
        return
    selected = ckpt_paths[-n_checkpoints:]
    print(f'\n[avg_checkpoints] Averaging {len(selected)} checkpoints:')
    for p in selected:
        print(f'  {os.path.basename(p)}')

    # Load and average state dicts
    avg_state = None
    for path in selected:
        ckpt = torch.load(path, map_location=device)
        sd   = ckpt['model_state_dict']
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg_state:
                avg_state[k] += sd[k].float()
    for k in avg_state:
        avg_state[k] /= len(selected)
        avg_state[k] = avg_state[k].to(next(iter(torch.load(selected[0], map_location=device)['model_state_dict'].values())).dtype)

    model.load_state_dict(avg_state)
    model.eval()

    # Evaluate
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                labels = batch.pop('label')
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            _, pred_top5 = out.topk(5, dim=1)
            correct1 += pred_top5[:, 0].eq(labels).sum().item()
            correct5 += pred_top5.eq(labels.unsqueeze(1).expand_as(pred_top5)).any(dim=1).sum().item()
            total    += labels.size(0)

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    print(f'[avg_checkpoints] Averaged model — Val Top-1: {top1:.2f}%  Top-5: {top5:.2f}%')

    # Save averaged checkpoint
    avg_path = os.path.join(checkpoint_dir, f'averaged_last{len(selected)}.pth')
    torch.save({'model_state_dict': avg_state, 'val_top1': top1}, avg_path)
    print(f'[avg_checkpoints] Saved to {avg_path}')


def main():
    parser = argparse.ArgumentParser(description='Train LAST model')
    parser.add_argument('--model', type=str, default='base',
                       choices=['base', 'small', 'large', 'nano_e', 'base_e', 'small_e', 'large_e',
                                'nano_e_v2', 'small_e_v2', 'base_e_v2', 'large_e_v2',
                                'nano_e_v3', 'small_e_v3', 'base_e_v3', 'large_e_v3',
                                'shiftfuse_nano', 'shiftfuse_small'],
                       help='Model variant (default: base)')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                       help='Dataset (default: ntu60)')
    parser.add_argument('--split_type', type=str, default='xsub', choices=['xsub', 'xview', 'xset'],
                       help='Split type (default: xsub)')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--workers', type=int, default=None, help='DataLoader workers')
    parser.add_argument('--env', type=str, default=None, choices=['local', 'kaggle', 'gcp', 'lambda', 'a100'],
                       help='Environment (default: auto-detect)')
    # ── Specific hyperparam shortcuts (avoid editing YAML between runs) ─────
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Override training.weight_decay')
    parser.add_argument('--dropout',      type=float, default=None,
                       help='Override model.dropout')
    parser.add_argument('--scheduler',    type=str,   default=None,
                       help='Override training.scheduler (cosine_warmup | multistep_warmup | multistep)')
    parser.add_argument('--min_lr',       type=float, default=None,
                       help='Override training.min_lr (cosine floor)')
    parser.add_argument('--milestones',   type=str,   default=None,
                       help='Override training.milestones as comma-separated: --milestones 50,65')
    parser.add_argument('--data_root',    type=str,   default=None,
                       help='Override environment.paths.data_base (root dir containing LAST-60-v2/)')
    # ── General key=value override (highest priority, applied last) ──────────
    parser.add_argument('--set', nargs='*', metavar='KEY=VALUE', default=None,
                       help='Override any config key via dot-notation: '
                            '--set training.lr=0.1 training.epochs=90 model.dropout=0.5')
    parser.add_argument('--avg_checkpoints', type=int, default=0, metavar='N',
                       help='After training, average the last N epoch checkpoints '
                            'and evaluate on val set (0 = disabled)')
    args = parser.parse_args()

    # ── 1. Load & merge config ──────────────────────────────────────────
    # Load training config baseline.
    # ShiftFuse-GCN uses its own calibrated defaults (EfficientGCN-aligned schedule,
    # no label smoothing, relaxed gradient clip). All other models use default.yaml.
    _training_cfg_name = 'shiftfuse' if args.model.startswith('shiftfuse_') else 'default'
    training_config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'training', f'{_training_cfg_name}.yaml'
    )
    with open(training_config_path, 'r') as f:
        default_cfg = yaml.safe_load(f)

    # Load specifics (Env, Data, Model)
    specific_config = load_config(env=args.env, dataset=args.dataset, model=args.model)
    
    # Merge: Specifics override Defaults
    config = default_cfg
    config.update(specific_config)

    # Apply environment hardware settings to training config.
    # Fixes: hardware.num_workers / pin_memory in env YAMLs were loaded but never
    # propagated to config['training'] where DataLoader reads them from.
    # CLI --workers still takes priority (args.workers is None check).
    _hw = config.get('environment', {}).get('hardware', {})
    if _hw.get('num_workers') is not None and args.workers is None:
        config['training']['num_workers'] = _hw['num_workers']
    if 'pin_memory' in _hw:
        config['training']['pin_memory'] = _hw['pin_memory']

    # Apply environment-level training overrides (batch_size, lr, warmup_start_lr, etc.)
    # Runs BEFORE CLI arg overrides below, so CLI still wins.
    # Only a100.yaml uses this; lambda.yaml/kaggle.yaml have no training_overrides → no-op.
    for _k, _v in config.get('environment', {}).get('training_overrides', {}).items():
        config['training'][_k] = _v

    # CLI overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.amp:
        config['training']['use_amp'] = True
    if args.workers is not None:
        config['training']['num_workers'] = args.workers
    # Specific hyperparam shortcuts
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.scheduler is not None:
        config['training']['scheduler'] = args.scheduler
    if args.min_lr is not None:
        config['training']['min_lr'] = args.min_lr
    if args.milestones is not None:
        config['training']['milestones'] = [int(x) for x in args.milestones.split(',')]
    # --data_root override
    if args.data_root is not None:
        config['environment']['paths']['data_base'] = args.data_root
    # General --set (applied last — highest priority)
    if args.set:
        for kv in args.set:
            key_path, raw_val = kv.split('=', 1)
            set_nested_key(config, key_path, raw_val)

    # ── 2. Setup ────────────────────────────────────────────────────────
    seed = config['training']['seed']
    set_seed(seed)

    # Create run directory: prioritize environment config over output config
    if 'environment' in config and 'paths' in config['environment']:
        runs_root = config['environment']['paths'].get('output_root')
    else:
        runs_root = config.get('output', {}).get('runs_root', './LAST-runs')

    run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    
    # Save run config
    run_config = {
        'model_variant': args.model,
        'dataset': args.dataset,
        'split_type': args.split_type,
        'training': config['training'],
        'seed': seed,
        'start_time': datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    print("=" * 70)
    print("  LAST — Training Pipeline")
    print("=" * 70)
    print(f"  Model:     LAST-{args.model.capitalize()}")
    print(f"  Dataset:   {args.dataset.upper()} ({args.split_type})")
    print(f"  Epochs:    {config['training']['epochs']}")
    print(f"  Batch:     {config['training']['batch_size']}")
    print(f"  LR:        {config['training']['lr']}")
    print(f"  Scheduler: {config['training']['scheduler']}")
    print(f"  Seed:      {seed}")
    print(f"  Run Dir:   {run_dir}")
    print("=" * 70)
    
    # ── 3. Model ────────────────────────────────────────────────────────
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)
    num_joints = config['data']['dataset']['num_joints']
    
    if args.model.endswith('_e_v3'):
        variant = args.model.replace('_e_v3', '')
        print(f"\n  Creating LAST-E v3 model (Variant: {variant})...")
        from src.models.last_e_v3 import LAST_E_v3
        model = LAST_E_v3(
            num_classes=num_classes,
            variant=variant,
            dropout=config['model'].get('dropout'),
            drop_path_rate=config['model'].get('drop_path_rate'),
            use_st_att=config['model'].get('use_st_att'),
        )
        # Store IB loss weight for trainer
        config['training']['ib_loss_weight'] = config.get('model', {}).get(
            'ib_loss_weight', config['training'].get('ib_loss_weight', 0.01)
        )
    elif args.model.startswith('shiftfuse_'):
        variant = args.model.replace('shiftfuse_', '')
        T = config['data']['dataset']['max_frames']
        print(f"\n  Creating ShiftFuse-GCN (Variant: {variant}, T={T})...")
        from src.models.shiftfuse_gcn import LAST_Lite
        model = LAST_Lite(
            num_classes=num_classes,
            variant=variant,
            T=T,
            num_joints=num_joints,
            dropout=config['model'].get('dropout'),
        )
        config['training']['ib_loss_weight'] = 0.0
    else:
        raise ValueError(f"Unsupported model variant: {args.model}")
    
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ── 4. Measure FLOPs & memory ───────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_frames = config['training']['input_frames']
    
    # Check if MIB or standard input
    data_type = config['data']['dataset'].get('data_type', 'npy')
    
    if data_type == 'mib':
        # MIB input is a dict of 3 streams, each (N, C, T, V, M)
        # FLOPs counter might need adjustment or just test with one stream x 3?
        # Let's mock a single stream for FLOPs first or update count_flops
        # For simplicity, we just use one stream shape to check backbone size * 3? 
        # ActuallyLAST-E v3 shares backbone. But inference runs backbone 3 times.
        # So FLOPs = 3 * Backbone + Head.
        # We'll rely on the simple FLOPs estimator handling a standard input, 
        # ensuring the model can handle a single tensor if passed.
        input_shape = (3, input_frames, num_joints, 2)
    else:
        input_shape = (3, input_frames, num_joints, 2)
    
    print(f"  Measuring FLOPs on input {input_shape}...")
    model_stats = count_flops(model, input_shape, device)
    
    print(f"  Parameters:  {model_stats['total_params']:,}")
    print(f"  Model Size:  {model_stats['total_model_size_mb']} MB")
    print(f"  FLOPs:       {model_stats['gflops']} GFLOPs (Single Stream Estimate)")
    if model_stats['gpu_memory_mb'] > 0:
        print(f"  GPU Memory:  {model_stats['gpu_memory_mb']} MB (inference)")
    
    # Save model stats
    with open(os.path.join(run_dir, 'model_stats.json'), 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    # ── 5. Data ─────────────────────────────────────────────────────────
    # Build data path — prefer explicit 'data_folder' env key; fall back to dataset defaults
    data_base = config['environment']['paths']['data_base']
    _env_folder = config.get('environment', {}).get('paths', {}).get('data_folder')
    if _env_folder:
        folder_name = _env_folder
    else:
        folder_name = "LAST-60" if args.dataset == 'ntu60' else "LAST-120"
    processed_data_path = os.path.join(data_base, folder_name, "data", "processed")
    
    print(f"\n  Loading data from: {processed_data_path}")
    print(f"  Data Type: {data_type}")
    
    # Transforms (config needs training.input_frames for TemporalCrop)
    merged_transform_config = {
        'dataset': config['data']['dataset'],
        'training': config['training'],
    }
    
    # IMPORTANT: Preprocessed .npy files are ALREADY normalized by preprocess_data.py
    # (center_spine + scale_by_torso). Applying Normalize again would double-normalize.
    merged_transform_config['dataset']['preprocessing']['normalize'] = False
    
    train_transform = get_train_transform(merged_transform_config)
    val_transform = get_val_transform(merged_transform_config)
    
    print(f"  Train transforms: {train_transform}")
    print(f"  Val transforms: {val_transform}")
    
    # Datasets
    split_config = config['data']['dataset'].get('splits', None)
    
    train_dataset = SkeletonDataset(
        data_path=processed_data_path,
        data_type=data_type,
        max_frames=config['data']['dataset']['max_frames'],
        num_joints=num_joints,
        transform=train_transform,
        split='train',
        split_type=args.split_type,
        split_config=split_config,
    )
    
    val_dataset = SkeletonDataset(
        data_path=processed_data_path,
        data_type=data_type,
        max_frames=config['data']['dataset']['max_frames'],
        num_joints=num_joints,
        transform=val_transform,
        split='val',
        split_type=args.split_type,
        split_config=split_config,
    )
    
    # DataLoaders
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    _pf = config.get('environment', {}).get('hardware', {}).get('prefetch_factor', 2)
    prefetch_factor = _pf if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # ── 6. Train ────────────────────────────────────────────────────────
    trainer = Trainer(model, config, run_dir)
    
    # ── Pre-training diagnostics ─────────────────────────────────────
    print(f"\n  --- Pre-training diagnostics ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Check one train batch
    train_batch_data, train_batch_labels = next(iter(train_loader))
    if isinstance(train_batch_data, dict):
        print(f"  Train batch (Dict): keys={list(train_batch_data.keys())}, labels={train_batch_labels.shape}")
        for k, v in train_batch_data.items():
            print(f"    {k}: {v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")
    else:
        print(f"  Train batch: data={train_batch_data.shape}, labels={train_batch_labels.shape}")
        print(f"  Train data stats: mean={train_batch_data.mean():.4f}, std={train_batch_data.std():.4f}")
    print(f"  Train labels: {train_batch_labels[:10].tolist()}")
    
    # Check one val batch
    val_batch_data, val_batch_labels = next(iter(val_loader))
    if isinstance(val_batch_data, dict):
        print(f"  Val   batch (Dict): keys={list(val_batch_data.keys())}, labels={val_batch_labels.shape}")
        for k, v in val_batch_data.items():
            print(f"    {k}: {v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")
        # Move dict to device for forward pass
        val_input = {k: v.to(device) for k, v in val_batch_data.items()}
    else:
        print(f"  Val   batch: data={val_batch_data.shape}, labels={val_batch_labels.shape}")
        print(f"  Val   data stats: mean={val_batch_data.mean():.4f}, std={val_batch_data.std():.4f}")
        val_input = val_batch_data.to(device)

    print(f"  Val   labels: {val_batch_labels[:10].tolist()}")
    
    # Check model output on val batch
    with torch.no_grad():
        val_out = model(val_input)
        _, val_preds = val_out.max(1)
        unique_preds = val_preds.unique()
        print(f"  Val predictions (1 batch): {val_preds[:10].tolist()}")
        print(f"  Val unique predictions: {unique_preds.tolist()} ({len(unique_preds)} classes)")
        val_correct = val_preds.eq(val_batch_labels.to(device)).sum().item()
        print(f"  Val batch top-1: {val_correct}/{len(val_batch_labels)} = {100*val_correct/len(val_batch_labels):.1f}%")
        # Sanity: random chance = 1/num_classes
        num_classes_diag = config['data']['dataset'].get('num_classes', 60)
        print(f"  (Random chance baseline: {100.0/num_classes_diag:.2f}%)")
    
    model.train()
    print(f"  --- End diagnostics ---\n")
    
    # Resume if specified
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(train_loader, val_loader)

    # ── Checkpoint averaging (optional) ─────────────────────────────────
    if args.avg_checkpoints > 0:
        _run_checkpoint_averaging(
            model, val_loader, run_dir, args.avg_checkpoints, device
        )


if __name__ == '__main__':
    main()
