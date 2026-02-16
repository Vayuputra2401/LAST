"""
Teacher-LAST: Utility Functions
================================
Provides:
  - compute_actual_lr: Standard VideoMAE linear LR scaling rule
  - save_checkpoint / load_checkpoint: Model checkpoint management
  - setup_logging: TensorBoard + console logging
  - MetricTracker: Running average for loss and accuracy
  - cosine_scheduler: Cosine annealing with warmup

Based on official MCG-NJU/VideoMAE utils.py
"""

import os
import math
import time
import datetime
import logging
import yaml
import torch
import numpy as np
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Learning Rate Scaling
# =============================================================================

def compute_actual_lr(base_lr: float, batch_size: int, num_gpus: int, update_freq: int) -> float:
    """
    Standard VideoMAE linear scaling rule. Always applied.
    
    Formula:
        actual_lr = base_lr * (batch_size * num_gpus * update_freq) / 256
    
    Reference: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
    The official VideoMAE repo uses 256 as the reference batch size.
    
    Args:
        base_lr: Base learning rate from config (default: 0.001)
        batch_size: Per-GPU batch size (default: 8)
        num_gpus: Number of GPUs (from --num_gpus CLI flag)
        update_freq: Gradient accumulation steps (default: 4)
    
    Returns:
        Scaled learning rate
    
    Example:
        1 GPU:  0.001 * (8 * 1 * 4) / 256 = 1.25e-4
        4 GPUs: 0.001 * (8 * 4 * 4) / 256 = 5e-4
    """
    effective_batch = batch_size * num_gpus * update_freq
    actual_lr = base_lr * effective_batch / 256

    print(f"[LR] base_lr={base_lr}, "
          f"batch={batch_size} × {num_gpus} GPUs × {update_freq} accum "
          f"= {effective_batch} effective batch "
          f"→ actual_lr = {actual_lr:.6f}")

    return actual_lr


# =============================================================================
# Cosine Annealing Scheduler with Warmup
# =============================================================================

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """
    Create a cosine annealing schedule with linear warmup.
    
    Used for both learning rate and weight decay scheduling.
    Returns a numpy array of values, one per training iteration.
    
    Schedule:
        1. Warmup phase (linear):  start_warmup_value → base_value
        2. Cosine phase:           base_value → final_value
    
    Args:
        base_value: Peak value after warmup
        final_value: Minimum value at end of training
        epochs: Total training epochs
        niter_per_ep: Number of iterations per epoch
        warmup_epochs: Number of warmup epochs (default: 0)
        start_warmup_value: Starting value for warmup (default: 0)
    
    Returns:
        np.ndarray: Schedule values, shape (epochs * niter_per_ep,)
    """
    # Warmup phase: linear interpolation
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    # Cosine phase: smooth decay from base_value to final_value
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model, optimizer, scaler, epoch, val_acc, config, output_dir,
                    is_best=False):
    """
    Save training checkpoint.
    
    Saves two files:
      - checkpoint-{epoch}.pth: Periodic checkpoint (every save_freq epochs)
      - checkpoint-latest.pth: Always updated (for auto-resume)
      - checkpoint-best.pth: Best validation accuracy (if is_best=True)
    
    Checkpoint contains:
      - epoch: Current epoch number
      - model: Model state dict
      - optimizer: Optimizer state dict
      - scaler: AMP GradScaler state dict
      - val_acc: Validation accuracy at this epoch
      - config: Full training config (for reproducibility)
    
    Args:
        model: The model (nn.Module)
        optimizer: The optimizer
        scaler: AMP GradScaler
        epoch: Current epoch number
        val_acc: Validation top-1 accuracy
        config: Training configuration dict
        output_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'val_acc': val_acc,
        'config': config,
    }

    # Always save latest (for auto-resume)
    latest_path = os.path.join(output_dir, 'checkpoint-latest.pth')
    torch.save(checkpoint, latest_path)

    # Save periodic checkpoint
    save_freq = config.get('checkpoint', {}).get('save_freq', 10)
    if (epoch + 1) % save_freq == 0:
        epoch_path = os.path.join(output_dir, f'checkpoint-{epoch:04d}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"[Checkpoint] Saved epoch {epoch} → {epoch_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint-best.pth')
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] New best model (acc={val_acc:.2f}%) → {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to .pth file
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scaler: AMP GradScaler to restore state (optional)
    
    Returns:
        dict: Checkpoint data (epoch, val_acc, config)
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_path}")
        return None

    print(f"[Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model'])
    print(f"[Checkpoint] Model weights loaded")

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[Checkpoint] Optimizer state loaded")

    if scaler is not None and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"[Checkpoint] Scaler state loaded")

    print(f"[Checkpoint] Resuming from epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint.get('val_acc', 'N/A')}")

    return checkpoint


def auto_resume(output_dir, model, optimizer=None, scaler=None):
    """
    Automatically resume from the latest checkpoint if it exists.
    
    Args:
        output_dir: Checkpoint directory
        model: Model to load weights into
        optimizer: Optimizer (optional)
        scaler: GradScaler (optional)
    
    Returns:
        int: Start epoch (0 if no checkpoint found)
    """
    latest_path = os.path.join(output_dir, 'checkpoint-latest.pth')
    if os.path.exists(latest_path):
        checkpoint = load_checkpoint(latest_path, model, optimizer, scaler)
        if checkpoint is not None:
            return checkpoint['epoch'] + 1  # Resume from next epoch
    return 0


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir, experiment_name="teacher-last"):
    """
    Set up TensorBoard writer and console logging.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name prefix for the run
    
    Returns:
        SummaryWriter: TensorBoard writer
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    # Console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print(f"[Logging] TensorBoard logs → {run_dir}")
    print(f"[Logging] Run: tensorboard --logdir={log_dir}")

    return writer


# =============================================================================
# Metric Tracking
# =============================================================================

class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values
    over a window or the global series average.
    
    From official VideoMAE utils.py.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg, value=self.value
        )


class MetricLogger:
    """
    Log and display training metrics.
    
    From official VideoMAE utils.py.
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log metrics every print_freq iterations.
        Yields items from iterable while printing progress.
        """
        i = 0
        if header is None:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"{header}  [{i}/{len(iterable)}]  "
                      f"eta: {eta_string}  "
                      f"{str(self)}  "
                      f"time: {str(iter_time)}  "
                      f"data: {str(data_time)}")

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} "
              f"({total_time / len(iterable):.4f} s/it)")


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to .yaml config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[Config] Loaded from {config_path}")
    return config


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"[Seed] Set to {seed}")
