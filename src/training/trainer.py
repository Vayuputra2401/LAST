"""
LAST Training Engine

Core trainer class handling the training loop, validation, checkpointing,
and metric logging.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from datetime import datetime


class Trainer:
    """
    Training engine for LAST model.
    
    Handles training loop, validation, LR scheduling (with warmup),
    checkpointing, and JSON-based metric logging.
    """
    
    def __init__(self, model, config, run_dir):
        """
        Args:
            model: LAST model instance
            config: Full merged config dict
            run_dir: Path to this run's output folder
        """
        self.config = config
        self.run_dir = run_dir
        self.train_cfg = config['training']
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Optimizer: SGD + Nesterov
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.train_cfg['lr'],
            momentum=self.train_cfg['momentum'],
            nesterov=self.train_cfg['nesterov'],
            weight_decay=self.train_cfg['weight_decay']
        )
        
        # LR Scheduler
        warmup_epochs = self.train_cfg['warmup_epochs']
        total_epochs = self.train_cfg['epochs']
        
        if self.train_cfg['scheduler'] == 'cosine_warmup':
            # Cosine annealing after warmup
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=self.train_cfg['min_lr']
            )
        else:
            # Step decay
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.train_cfg.get('milestones', [35, 55]),
                gamma=self.train_cfg.get('gamma', 0.1)
            )
        
        # Loss: Cross-entropy + label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.train_cfg['label_smoothing']
        )
        
        # Mixed precision
        self.use_amp = self.train_cfg.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.grad_clip = self.train_cfg['gradient_clip']
        
        # Tracking
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.metrics_log = {}  # epoch -> metrics dict
        
        # Create output dirs
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def _warmup_lr(self, epoch):
        """Apply linear warmup to learning rate."""
        warmup_epochs = self.train_cfg['warmup_epochs']
        if epoch < warmup_epochs:
            start_lr = self.train_cfg['warmup_start_lr']
            target_lr = self.train_cfg['lr']
            lr = start_lr + (target_lr - start_lr) * (epoch / warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_one_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Returns:
            dict: {'loss': float, 'accuracy': float}
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item() * batch_data.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on validation set.
        
        Returns:
            dict: {'loss': float, 'accuracy': float}
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
            else:
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
            
            total_loss += loss.item() * batch_data.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_loader, val_loader):
        """
        Full training loop with epoch-level tqdm progress bar.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        """
        total_epochs = self.train_cfg['epochs']
        
        print(f"\n{'='*70}")
        print(f"  Training LAST | Device: {self.device} | Epochs: {total_epochs}")
        print(f"  LR: {self.train_cfg['lr']} | Batch: {self.train_cfg['batch_size']} | "
              f"Warmup: {self.train_cfg['warmup_epochs']} epochs")
        print(f"  Output: {self.run_dir}")
        print(f"{'='*70}\n")
        
        # Epoch-level progress bar
        epoch_bar = tqdm(
            range(self.start_epoch, total_epochs),
            desc="Training",
            unit="epoch",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for epoch in epoch_bar:
            epoch_start = time.time()
            
            # Warmup LR
            self._warmup_lr(epoch)
            
            # Train
            train_metrics = self.train_one_epoch(train_loader, epoch)
            
            # Step scheduler (after warmup phase)
            if epoch >= self.train_cfg['warmup_epochs']:
                self.scheduler.step()
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Current LR
            current_lr = self._get_lr()
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            epoch_data = {
                'train_loss': round(train_metrics['loss'], 4),
                'train_acc': round(train_metrics['accuracy'], 2),
                'val_loss': round(val_metrics['loss'], 4),
                'val_acc': round(val_metrics['accuracy'], 2),
                'lr': round(current_lr, 6),
                'epoch_time': round(epoch_time, 1),
                'timestamp': datetime.now().isoformat()
            }
            self.metrics_log[str(epoch + 1)] = epoch_data
            
            # Save metrics JSON after every epoch
            self._save_metrics()
            
            # Terminal output: compact epoch summary
            is_best = val_metrics['accuracy'] > self.best_val_acc
            best_marker = " ★" if is_best else ""
            
            epoch_bar.set_postfix_str(
                f"TrL={epoch_data['train_loss']:.3f} "
                f"TrA={epoch_data['train_acc']:.1f}% "
                f"VaL={epoch_data['val_loss']:.3f} "
                f"VaA={epoch_data['val_acc']:.1f}% "
                f"LR={current_lr:.5f}"
            )
            
            # Print detailed line below progress bar
            tqdm.write(
                f"  Epoch {epoch+1:3d}/{total_epochs} │ "
                f"Train: {epoch_data['train_loss']:.4f} / {epoch_data['train_acc']:.2f}% │ "
                f"Val: {epoch_data['val_loss']:.4f} / {epoch_data['val_acc']:.2f}% │ "
                f"LR: {current_lr:.6f} │ "
                f"{epoch_time:.1f}s{best_marker}"
            )
            
            # Save best model
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, 'best_model.pth'),
                    epoch, val_metrics
                )
            
            # Periodic checkpoint
            if (epoch + 1) % self.train_cfg['save_interval'] == 0:
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, f'checkpoint_ep{epoch+1}.pth'),
                    epoch, val_metrics
                )
        
        # Save final checkpoint
        self._save_checkpoint(
            os.path.join(self.checkpoint_dir, 'final_model.pth'),
            total_epochs - 1, val_metrics
        )
        
        print(f"\n{'='*70}")
        print(f"  Training Complete!")
        print(f"  Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Metrics: {os.path.join(self.run_dir, 'metrics.json')}")
        print(f"{'='*70}\n")
    
    def _save_checkpoint(self, path, epoch, val_metrics):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path):
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.start_epoch = checkpoint['epoch'] + 1
        
        # Load existing metrics if present
        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics_log = json.load(f)
        
        print(f"  Resumed from epoch {self.start_epoch} (best val acc: {self.best_val_acc:.2f}%)")
    
    def _save_metrics(self):
        """Save metrics log to JSON."""
        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
