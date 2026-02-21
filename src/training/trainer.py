"""
LAST Training Engine

Core trainer class handling the training loop, validation, checkpointing,
and metric logging.

Optimizations vs original:
  - zero_grad(set_to_none=True): frees gradient memory instead of zeroing
  - NaN/Inf check BEFORE backward: prevents corrupt gradients
  - Loss and accuracy accumulated as tensors, .item() called ONCE per epoch
  - LinearLR + SequentialLR replaces manual warmup LR override: resumes
    correctly from checkpoints regardless of which phase was interrupted
  - Gradient accumulation: effective batch = batch_size × accum_steps
  - torch.compile opt-in: 10-40% throughput gain on PyTorch 2.0+
  - Top-1 and Top-5 accuracy tracked and logged
  - alpha and A_learned excluded from weight decay
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, MultiStepLR, LinearLR, SequentialLR
)
from tqdm import tqdm
from datetime import datetime


def _accuracy_topk(output, target, topk=(1, 5)):
    """
    Compute top-k accuracy as integer correct counts (not %).
    Returns a tensor on the same device as output — no .item() call here
    so there is no GPU sync until the caller explicitly needs the value.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                           # (maxk, B)
        correct = pred.eq(target.unsqueeze(0))   # (maxk, B) bool
        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum()  # scalar tensor
            res.append(correct_k)
        return res  # list of scalar tensors, still on GPU


class Trainer:
    """
    Training engine for LAST v2 model.

    Handles training loop, validation, LR scheduling (warmup + cosine/step),
    gradient accumulation, optional torch.compile, checkpointing, and
    JSON-based metric logging.
    """

    def __init__(self, model, config, run_dir):
        """
        Args:
            model:    LAST model instance (not yet .to(device) — done here)
            config:   Full merged config dict
            run_dir:  Path to this run's output folder
        """
        self.config = config
        self.run_dir = run_dir
        self.train_cfg = config['training']

        # ── Device ──────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── torch.compile (opt-in, PyTorch 2.0+) ───────────────────────────
        # Fuses ops, eliminates Python overhead in the forward pass.
        # ~10-40% throughput gain on repeated GCN matmuls and convolutions.
        # Disabled by default: set use_compile: true in training config.
        if self.train_cfg.get('use_compile', False):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("  torch.compile: enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"  torch.compile: skipped ({e})")
        self.model = model.to(self.device)

        # ── Gradient Accumulation ───────────────────────────────────────────
        # Effective batch = batch_size × accum_steps.
        # Allows larger effective batches on VRAM-limited GPUs.
        # For Kaggle 16GB: Base can use batch=32 directly; Large may need
        # batch=16 + accum_steps=2 for effective batch=32.
        self.accum_steps = self.train_cfg.get('gradient_accumulation_steps', 1)

        # ── Optimizer ───────────────────────────────────────────────────────
        # Three param groups:
        #   decay:    conv/linear weights → weight_decay applied
        #   no_decay: bias, BN, LayerNorm, alpha gates, A_learned matrices
        #             → weight_decay=0 (these should not be shrunk toward 0)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude from weight decay:
            #   - bias terms
            #   - BN/LayerNorm scale+bias (bn, norm)
            #   - ST_JointAtt alpha gates (zero-init, WD fights their growth)
            #   - AdaptiveGraphConv A_learned matrices (WD fights edge learning)
            if (
                'bias' in name
                or 'bn' in name
                or 'norm' in name
                or 'alpha' in name          # ST_JointAtt zero-init gate scalars
                or 'A_learned' in name      # adaptive graph learnable edges
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        optim_params = [
            {'params': decay,    'weight_decay': self.train_cfg['weight_decay']},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

        opt_name = self.train_cfg.get('optimizer', 'sgd').lower()
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                optim_params, lr=self.train_cfg['lr']
            )
        elif opt_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                optim_params,
                lr=self.train_cfg['lr'],
                momentum=self.train_cfg['momentum'],
                nesterov=self.train_cfg['nesterov'],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # ── LR Scheduler: LinearLR warmup + CosineAnnealingLR ───────────────
        # Replaces the manual _warmup_lr() override which did not serialize
        # into checkpoint state and caused incorrect LR on resume.
        # SequentialLR chains both schedulers and is fully checkpoint-safe.
        warmup_epochs = self.train_cfg['warmup_epochs']
        total_epochs  = self.train_cfg['epochs']

        if self.train_cfg['scheduler'] == 'cosine_warmup':
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=self.train_cfg['warmup_start_lr'] / self.train_cfg['lr'],
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=self.train_cfg['min_lr'],
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            # Step decay fallback
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.train_cfg.get('milestones', [35, 55]),
                gamma=self.train_cfg.get('gamma', 0.1),
            )

        # ── Loss ────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.train_cfg['label_smoothing']
        )

        # ── Mixed Precision ─────────────────────────────────────────────────
        self.use_amp = self.train_cfg.get('use_amp', False)
        self.scaler  = GradScaler() if self.use_amp else None

        # ── Gradient Clipping ───────────────────────────────────────────────
        self.grad_clip = self.train_cfg['gradient_clip']

        # ── State Tracking ──────────────────────────────────────────────────
        self.best_val_acc  = 0.0
        self.start_epoch   = 0
        self.metrics_log   = {}

        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    # ────────────────────────────────────────────────────────────────────────

    def train_one_epoch(self, train_loader, epoch):
        """
        Train for one epoch with gradient accumulation.

        Returns:
            dict: {'loss', 'top1_acc', 'top5_acc'}
        """
        self.model.train()

        # Accumulate on GPU — no .item() per batch (avoids repeated GPU sync)
        running_loss   = torch.tensor(0.0, device=self.device)
        correct_top1   = torch.tensor(0,   device=self.device)
        correct_top5   = torch.tensor(0,   device=self.device)
        total_samples  = 0

        num_classes = self.config['data']['dataset'].get('num_classes', 60)
        topk = (1, 5) if num_classes >= 5 else (1,)

        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1:3d} [Train]",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_data, batch_labels) in enumerate(pbar):
            # ── Move to device ───────────────────────────────────────────
            if isinstance(batch_data, dict):
                batch_data = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch_data.items()
                }
                B = next(iter(batch_data.values())).size(0)
            else:
                batch_data   = batch_data.to(self.device, non_blocking=True)
                B = batch_data.size(0)
            batch_labels = batch_labels.to(self.device, non_blocking=True)

            # ── Forward ──────────────────────────────────────────────────
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_data)
                    loss    = self.criterion(outputs, batch_labels)
            else:
                outputs = self.model(batch_data)
                loss    = self.criterion(outputs, batch_labels)

            # ── NaN/Inf guard BEFORE backward ────────────────────────────
            # Checking after backward (original) lets corrupt gradients
            # propagate into optimizer state. Check here and skip the step.
            if not torch.isfinite(loss):
                print(f"\n  WARNING: non-finite loss at epoch {epoch+1} "
                      f"batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # ── Backward (scaled for accumulation) ───────────────────────
            # Divide loss by accum_steps so gradients average over the
            # accumulation window rather than summing.
            scaled_loss = loss / self.accum_steps
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # ── Optimizer step every accum_steps batches ─────────────────
            is_last_batch      = (batch_idx + 1 == len(train_loader))
            is_accum_step      = ((batch_idx + 1) % self.accum_steps == 0)

            if is_accum_step or is_last_batch:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # ── Accumulate metrics (stay on GPU) ─────────────────────────
            with torch.no_grad():
                running_loss  += loss.detach() * B
                accs = _accuracy_topk(outputs.detach(), batch_labels, topk=topk)
                correct_top1  += accs[0]
                if len(accs) > 1:
                    correct_top5 += accs[1]
            total_samples += B

        # ── Sync once at epoch end ────────────────────────────────────────
        avg_loss   = (running_loss / max(total_samples, 1)).item()
        top1_acc   = 100.0 * correct_top1.item() / max(total_samples, 1)
        top5_acc   = 100.0 * correct_top5.item() / max(total_samples, 1)

        pbar.close()
        return {'loss': avg_loss, 'top1_acc': top1_acc, 'top5_acc': top5_acc}

    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on validation set.

        Returns:
            dict: {'loss', 'top1_acc', 'top5_acc'}
        """
        self.model.eval()

        running_loss  = torch.tensor(0.0, device=self.device)
        correct_top1  = torch.tensor(0,   device=self.device)
        correct_top5  = torch.tensor(0,   device=self.device)
        total_samples = 0

        num_classes = self.config['data']['dataset'].get('num_classes', 60)
        topk = (1, 5) if num_classes >= 5 else (1,)

        pbar = tqdm(
            val_loader,
            desc=f"              [Val  ]",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            leave=False,
        )

        for batch_data, batch_labels in pbar:
            if isinstance(batch_data, dict):
                batch_data = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch_data.items()
                }
                B = next(iter(batch_data.values())).size(0)
            else:
                batch_data   = batch_data.to(self.device, non_blocking=True)
                B = batch_data.size(0)
            batch_labels = batch_labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_data)
                    loss    = self.criterion(outputs, batch_labels)
            else:
                outputs = self.model(batch_data)
                loss    = self.criterion(outputs, batch_labels)

            if not torch.isfinite(loss):
                continue

            running_loss += loss * B
            accs = _accuracy_topk(outputs, batch_labels, topk=topk)
            correct_top1 += accs[0]
            if len(accs) > 1:
                correct_top5 += accs[1]
            total_samples += B

        pbar.close()
        avg_loss = (running_loss / max(total_samples, 1)).item()
        top1_acc = 100.0 * correct_top1.item() / max(total_samples, 1)
        top5_acc = 100.0 * correct_top5.item() / max(total_samples, 1)
        return {'loss': avg_loss, 'top1_acc': top1_acc, 'top5_acc': top5_acc}

    # ────────────────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        """Full training loop."""
        total_epochs = self.train_cfg['epochs']

        print(f"\n{'='*70}")
        model_cls = type(self.model).__name__
        print(f"  Training {model_cls} | Device: {self.device} | Epochs: {total_epochs}")
        print(f"  Optimizer: {self.train_cfg.get('optimizer','sgd').upper()} | "
              f"LR: {self.train_cfg['lr']} | "
              f"Batch: {self.train_cfg['batch_size']} × accum {self.accum_steps} "
              f"= effective {self.train_cfg['batch_size'] * self.accum_steps}")
        print(f"  Scheduler: {self.train_cfg['scheduler']} | "
              f"Warmup: {self.train_cfg['warmup_epochs']} epochs | "
              f"AMP: {self.use_amp}")
        print(f"  Output: {self.run_dir}")
        print(f"{'='*70}\n")

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # Step scheduler every epoch (SequentialLR handles warmup internally)
            self.scheduler.step()

            # Validate
            val_metrics = self.validate(val_loader)

            current_lr  = self._get_lr()
            epoch_time  = time.time() - epoch_start

            # Log
            epoch_data = {
                'train_loss':    round(train_metrics['loss'],     4),
                'train_top1':    round(train_metrics['top1_acc'], 2),
                'train_top5':    round(train_metrics['top5_acc'], 2),
                'val_loss':      round(val_metrics['loss'],       4),
                'val_top1':      round(val_metrics['top1_acc'],   2),
                'val_top5':      round(val_metrics['top5_acc'],   2),
                'lr':            round(current_lr,                6),
                'epoch_time':    round(epoch_time,                1),
                'timestamp':     datetime.now().isoformat(),
            }
            self.metrics_log[str(epoch + 1)] = epoch_data
            self._save_metrics()

            is_best     = val_metrics['top1_acc'] > self.best_val_acc
            best_marker = " ★" if is_best else ""

            print(
                f"  Epoch {epoch+1:3d}/{total_epochs} │ "
                f"Train: {epoch_data['train_loss']:.4f} / "
                f"{epoch_data['train_top1']:.2f}% (t5:{epoch_data['train_top5']:.1f}%) │ "
                f"Val: {epoch_data['val_loss']:.4f} / "
                f"{epoch_data['val_top1']:.2f}% (t5:{epoch_data['val_top5']:.1f}%) │ "
                f"LR: {current_lr:.6f} │ "
                f"{epoch_time:.1f}s{best_marker}"
            )

            if is_best:
                self.best_val_acc = val_metrics['top1_acc']
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, 'best_model.pth'),
                    epoch, val_metrics
                )

            if (epoch + 1) % self.train_cfg['save_interval'] == 0:
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, f'checkpoint_ep{epoch+1}.pth'),
                    epoch, val_metrics
                )

        self._save_checkpoint(
            os.path.join(self.checkpoint_dir, 'final_model.pth'),
            total_epochs - 1, val_metrics
        )

        print(f"\n{'='*70}")
        print(f"  Training Complete!")
        print(f"  Best Val Top-1: {self.best_val_acc:.2f}%")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Metrics: {os.path.join(self.run_dir, 'metrics.json')}")
        print(f"{'='*70}\n")

    # ────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, path, epoch, val_metrics):
        """Save model + optimizer + scheduler state."""
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict() if self.scaler else None,
            'best_val_acc':         self.best_val_acc,
            'val_top1':             val_metrics['top1_acc'],
            'val_top5':             val_metrics['top5_acc'],
            'val_loss':             val_metrics['loss'],
            'config':               self.config,
        }, path)

    def load_checkpoint(self, path):
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.start_epoch  = checkpoint['epoch'] + 1

        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics_log = json.load(f)

        print(f"  Resumed from epoch {self.start_epoch} "
              f"(best val top-1: {self.best_val_acc:.2f}%)")

    def _save_metrics(self):
        """Append epoch metrics to JSON log."""
        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
