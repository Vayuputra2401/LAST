"""
Teacher-LAST: Training Engine
===============================
Training and validation loops for fine-tuning VideoMAE-2 on NTU RGB+D 60.

Based on official VideoMAE engine_for_finetuning.py with:
  - Mixed precision training (PyTorch AMP)
  - Gradient accumulation (update_freq)
  - Gradient clipping (clip_grad)
  - Mixup / CutMix regularization
  - Label smoothing
  - Cosine LR scheduling with warmup
  - TensorBoard logging
  - Top-1 and Top-5 accuracy tracking

References:
    - Official engine: engine_for_finetuning.py in MCG-NJU/VideoMAE
    - Mixup: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    - CutMix: "CutMix: Regularization Strategy" (Yun et al., 2019)
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from src.utils import MetricLogger, SmoothedValue, cosine_scheduler, save_checkpoint


# =============================================================================
# Accuracy Computation
# =============================================================================

def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy for the specified values of k.
    
    Args:
        output: Model predictions (B, num_classes)
        target: Ground truth labels (B,)
        topk: Tuple of k values to compute
    
    Returns:
        list: Top-k accuracy values (as percentages)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, epoch, config,
                    scaler=None, lr_schedule=None, writer=None,
                    mixup_fn=None, start_steps=0):
    """
    Train the model for one epoch.
    
    Handles:
      - Mixed precision (AMP) training
      - Gradient accumulation (update_freq steps before optimizer.step())
      - Mixup / CutMix augmentation at the batch level
      - Label smoothing
      - Gradient clipping
      - Learning rate scheduling (per-iteration cosine schedule)
    
    Args:
        model: The model to train
        dataloader: Training DataLoader
        optimizer: The optimizer
        epoch: Current epoch number
        config: Training configuration dictionary
        scaler: AMP GradScaler (None to disable AMP)
        lr_schedule: Pre-computed LR schedule array (from cosine_scheduler)
        writer: TensorBoard SummaryWriter
        mixup_fn: Mixup/CutMix function (from timm)
        start_steps: Global step offset for LR schedule indexing
    
    Returns:
        dict: Training metrics {loss, lr, top1, top5}
    """
    model.train()

    update_freq = config.get('update_freq', 1)
    clip_grad = config.get('clip_grad', None)
    use_amp = config.get('mixed_precision', True) if isinstance(config, dict) else True

    # Loss function: depends on whether mixup is active
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        smoothing = config.get('label_smoothing', 0.1)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 20
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _) in enumerate(
        metric_logger.log_every(dataloader, print_freq, header)
    ):
        # Update LR for this iteration
        step = start_steps + data_iter_step
        if lr_schedule is not None and step < len(lr_schedule):
            for param_group in optimizer.param_groups:
                # Scale by the group's lr_scale (from layer-wise decay)
                lr_scale = param_group.get('lr_scale', 1.0)
                param_group['lr'] = lr_schedule[step] * lr_scale

        # Move data to GPU
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # Apply mixup/cutmix
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass with AMP
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            raise ValueError(f"Loss is {loss_value}")

        # Scale loss for gradient accumulation
        loss = loss / update_freq

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()

            # Optimizer step every update_freq iterations
            if (data_iter_step + 1) % update_freq == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()

            if (data_iter_step + 1) % update_freq == 0:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=current_lr)

    # Gather stats
    print(f"Averaged stats: {metric_logger}")

    train_stats = {
        'loss': metric_logger.meters['loss'].global_avg,
        'lr': current_lr,
    }

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/loss', train_stats['loss'], epoch)
        writer.add_scalar('train/lr', train_stats['lr'], epoch)

    return train_stats


# =============================================================================
# Validation Loop
# =============================================================================

@torch.no_grad()
def validate(model, dataloader, epoch, config, writer=None):
    """
    Validate the model on the validation set.
    
    Computes Top-1 and Top-5 accuracy using uniform temporal sampling.
    No mixup/cutmix during validation.
    
    Args:
        model: The model to evaluate
        dataloader: Validation DataLoader
        epoch: Current epoch number
        config: Configuration dictionary
        writer: TensorBoard SummaryWriter
    
    Returns:
        dict: Validation metrics {loss, top1, top5}
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Val: [{epoch}]'

    for samples, targets, _ in metric_logger.log_every(dataloader, 20, header):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # Forward pass (no AMP needed for validation)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        # Compute accuracy
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['top1'].update(top1, n=batch_size)
        metric_logger.meters['top5'].update(top5, n=batch_size)

    print(f"* Acc@1 {metric_logger.meters['top1'].global_avg:.2f}  "
          f"Acc@5 {metric_logger.meters['top5'].global_avg:.2f}  "
          f"Loss {metric_logger.meters['loss'].global_avg:.4f}")

    val_stats = {
        'loss': metric_logger.meters['loss'].global_avg,
        'top1': metric_logger.meters['top1'].global_avg,
        'top5': metric_logger.meters['top5'].global_avg,
    }

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', val_stats['loss'], epoch)
        writer.add_scalar('val/top1', val_stats['top1'], epoch)
        writer.add_scalar('val/top5', val_stats['top5'], epoch)

    return val_stats


# =============================================================================
# Mixup/CutMix Setup
# =============================================================================

def build_mixup_fn(config):
    """
    Build Mixup/CutMix function from config.
    
    Mixup and CutMix are regularization techniques that blend training
    samples and their labels. This forces the model to learn more robust
    features instead of memorizing individual samples.
    
    Official VideoMAE uses:
      - mixup_alpha=0.8, cutmix_alpha=1.0
      - mixup_prob=1.0, switch_prob=0.5
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        Mixup function or None if both mixup and cutmix are 0
    """
    mixup_alpha = config.get('mixup', 0.8)
    cutmix_alpha = config.get('cutmix', 1.0)
    mixup_prob = config.get('mixup_prob', 1.0)
    switch_prob = config.get('mixup_switch_prob', 0.5)
    label_smoothing = config.get('label_smoothing', 0.1)
    num_classes = config.get('num_classes', 60)

    if mixup_alpha > 0 or cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=switch_prob,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )
        print(f"[Mixup] Enabled: mixup={mixup_alpha}, cutmix={cutmix_alpha}, "
              f"prob={mixup_prob}, switch={switch_prob}")
        return mixup_fn
    else:
        print("[Mixup] Disabled")
        return None


# =============================================================================
# Full Training Pipeline
# =============================================================================

def train(model, train_loader, val_loader, config, writer=None):
    """
    Full training pipeline: train for all epochs with validation.
    
    Orchestrates:
      1. Optimizer setup with layer-wise LR decay
      2. Cosine LR schedule with warmup
      3. AMP GradScaler
      4. Mixup/CutMix
      5. Training + validation loop
      6. Checkpoint saving (periodic + best)
      7. Auto-resume from latest checkpoint
    
    Args:
        model: VideoMAE-2 model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Full configuration dictionary
        writer: TensorBoard SummaryWriter
    """
    training_config = config['training']
    checkpoint_config = config['checkpoint']
    hardware_config = config['hardware']

    # Move model to GPU
    device = torch.device(hardware_config.get('device', 'cuda'))
    model = model.to(device)

    # Compute actual LR using standard linear scaling rule
    from src.utils import compute_actual_lr
    actual_lr = compute_actual_lr(
        base_lr=training_config['base_lr'],
        batch_size=training_config['batch_size'],
        num_gpus=config.get('num_gpus', 1),
        update_freq=training_config.get('update_freq', 1),
    )

    # Build optimizer with layer-wise LR decay
    param_groups = model.get_parameter_groups(
        base_lr=actual_lr,
        weight_decay=training_config.get('weight_decay', 0.05),
        layer_decay=training_config.get('layer_decay', 0.75),
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(training_config.get('opt_betas', [0.9, 0.999])),
        eps=training_config.get('opt_eps', 1e-8),
    )

    # AMP GradScaler
    use_amp = hardware_config.get('mixed_precision', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Build LR schedule (cosine annealing with warmup)
    num_epochs = training_config['epochs']
    niter_per_ep = len(train_loader) // training_config.get('update_freq', 1)
    lr_schedule = cosine_scheduler(
        base_value=actual_lr,
        final_value=training_config.get('min_lr', 1e-6),
        epochs=num_epochs,
        niter_per_ep=niter_per_ep,
        warmup_epochs=training_config.get('warmup_epochs', 5),
        start_warmup_value=training_config.get('warmup_lr', 1e-6),
    )

    # Mixup / CutMix
    mixup_config = {
        'mixup': training_config.get('mixup', 0.8),
        'cutmix': training_config.get('cutmix', 1.0),
        'mixup_prob': training_config.get('mixup_prob', 1.0),
        'mixup_switch_prob': training_config.get('mixup_switch_prob', 0.5),
        'label_smoothing': training_config.get('label_smoothing', 0.1),
        'num_classes': config['data']['num_classes'],
    }
    mixup_fn = build_mixup_fn(mixup_config)

    # Auto-resume from checkpoint
    from src.utils import auto_resume as try_auto_resume
    start_epoch = 0
    best_val_acc = 0.0

    if checkpoint_config.get('auto_resume', True):
        start_epoch = try_auto_resume(
            checkpoint_config['output_dir'], model, optimizer, scaler
        )
        if start_epoch > 0:
            print(f"[Training] Resuming from epoch {start_epoch}")

    # =========================================================================
    # Training Loop
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"Starting training: epochs={num_epochs}, start_epoch={start_epoch}")
    print(f"Effective LR: {actual_lr:.6f}")
    print(f"AMP: {use_amp}, Mixup: {mixup_fn is not None}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # --- Train ---
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            config={**training_config, 'mixed_precision': use_amp},
            scaler=scaler,
            lr_schedule=lr_schedule,
            writer=writer,
            mixup_fn=mixup_fn,
            start_steps=epoch * niter_per_ep,
        )

        # --- Validate ---
        val_stats = validate(
            model=model,
            dataloader=val_loader,
            epoch=epoch,
            config=config,
            writer=writer,
        )

        # --- Checkpoint ---
        is_best = val_stats['top1'] > best_val_acc
        if is_best:
            best_val_acc = val_stats['top1']

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            val_acc=val_stats['top1'],
            config=config,
            output_dir=checkpoint_config['output_dir'],
            is_best=is_best,
        )

        # --- Epoch Summary ---
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} complete in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Val Acc@1: {val_stats['top1']:.2f}%  "
              f"Acc@5: {val_stats['top5']:.2f}%  "
              f"Loss: {val_stats['loss']:.4f}")
        print(f"  Best Acc@1: {best_val_acc:.2f}%")
        print(f"  LR: {train_stats['lr']:.6f}")
        print()

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc@1: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    return best_val_acc
