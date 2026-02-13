# Kaggle Support & CLI Overrides - Addition to Code Architecture

**This extends the GCP Support additions with Kaggle environment and flexible command-line argument parsing**

---

## 2.5.4 Kaggle Environment (`configs/environment/kaggle.yaml`) - NEW

```yaml
# Kaggle Notebook/Kernel Environment Configuration
environment:
  name: "kaggle"
  type: "cloud"
  platform: "kaggle"
  
  # Kaggle-specific settings
  kaggle:
    kernel_type: "notebook"        # or "script"
    accelerator: "gpu"             # GPU P100 or T4
    internet: true                 # Enable internet for downloads
    
  # Path Configuration (Kaggle standard paths)
  paths:
    # Kaggle read-only input datasets
    kaggle_input: "/kaggle/input"
    
    # Kaggle working directory (read-write)
    data_root: "/kaggle/working/data"
    checkpoint_root: "/kaggle/working/checkpoints"
    log_root: "/kaggle/working/logs"
    results_root: "/kaggle/working/results"
    temp_workspace: "/kaggle/tmp"
    
    # Cached datasets (if using Kaggle Datasets)
    kaggle_datasets:
      ntu120_skeletons: "/kaggle/input/ntu-rgbd-120-skeletons"
      teacher_logits: "/kaggle/input/ntu120-teacher-logits"
    
  # Cloud Storage (optional - for saving results)
  cloud:
    enabled: false               # No direct GCS in Kaggle
    save_to_kaggle_datasets: true  # Create output dataset
    
    # Alternative: Save outputs as Kaggle Dataset
    output_dataset:
      enabled: true
      name: "last-training-results"
      title: "LAST Training Results - NTU-120"
      
  # Hardware Configuration (Kaggle limits)
  hardware:
    device: "cuda"                 # Kaggle provides GPU
    num_workers: 2                 # Max 2 due to CPU limits (2 cores)
    pin_memory: true
    persistent_workers: false      # Limited memory
    
  # Kaggle-specific constraints
  constraints:
    max_runtime_hours: 12          # Kaggle max session time (9-12h)
    max_disk_space_gb: 20          # Working directory limit (73GB total but 20GB usable)
    ram_gb: 13                     # Available RAM (out of 16GB)
    gpu_memory_gb: 16              # P100 (16GB) or T4 (16GB)
    cpu_cores: 2                   # Limited CPUs
    
  # Training adjustments for Kaggle
  training_overrides:
    # Reduce batch size for memory constraints
    batch_size: 32                 # vs 64 on GCP (less RAM)
    num_workers: 2                 # Only 2 CPU cores
    save_frequency: 10             # Save less often (disk limit)
    checkpoint_cleanup: true       # Auto-delete old checkpoints
    keep_best_k: 2                 # Keep only 2 best models (disk space)
    
    # Faster logging (shorter sessions)
    log_frequency: 20              # vs 50 on other platforms
    eval_frequency: 500            # More frequent validation
```

---

## 3.9.1 Updated EnvironmentDetector (with Kaggle support)

```python
class EnvironmentDetector:
    """
    Automatically detect execution environment: local, GCP, or Kaggle.
    
    Class Methods:
        detect_environment() -> str
            Returns: "local", "gcp", or "kaggle"
            Implementation:
                1. Check for Kaggle: os.path.exists('/kaggle')
                2. Check for GCP: metadata server
                3. Default: local
            
        is_kaggle() -> bool
            Returns: True if running on Kaggle
            Check: os.path.exists('/kaggle/working')
            
        is_gcp() -> bool
            Returns: True if running on GCP instance
            Check: GCP metadata server
            
        is_local() -> bool
            Returns: True if running on local machine
            
        get_kaggle_metadata() -> dict
            Returns: {
                'kernel_id': str,
                'kernel_type': str,  # 'notebook' or 'script'
                'gpu_available': bool,
                'internet_enabled': bool
            } or None if not on Kaggle
            Source: Environment variables KAGGLE_*
            
        get_instance_metadata() -> dict
            Returns: GCP instance metadata or None
            
        load_environment_config(override_env: Optional[str] = None) -> dict
            Inputs: Optional environment name to force
            Returns: Merged config from auto-detected or specified environment
            Auto-selects:
                - configs/environment/kaggle.yaml (if on Kaggle)
                - configs/environment/gcp.yaml (if on GCP)
                - configs/environment/local.yaml (default)
            Override: If override_env specified, loads that instead
            
        resolve_paths(config: dict) -> dict
            Inputs: Config dictionary
            Returns: Config with resolved absolute paths
            Handles: Windows, Linux, Kaggle paths automatically
    """
```

---

## 4.1 Updated Training Script with CLI Overrides (`scripts/train.py`)

**Purpose:** Main training script with flexible command-line argument parsing.

```python
"""
LAST Training Entry Point with CLI Overrides

Usage Examples:

    # Auto-detect environment (local/GCP/Kaggle)
    python scripts/train.py

    # Force specific environment
    python scripts/train.py --env local
    python scripts/train.py --env gcp
    python scripts/train.py --env kaggle
    
    # Override specific configs
    python scripts/train.py --env kaggle --batch_size 16 --epochs 50
    
    # Override data paths
    python scripts/train.py --data_path /custom/data --checkpoint_dir ./checkpoints
    
    # Multiple overrides
    python scripts/train.py \
        --env gcp \
        --data_config configs/data/ntu120_xset.yaml \
        --model_config configs/model/last_large.yaml \
        --batch_size 32 \
        --learning_rate 0.0005 \
        --epochs 120 \
        --device cuda:0
"""

import argparse
import os
from pathlib import Path

from src.cloud.environment import EnvironmentDetector
from src.utils.config import ConfigLoader
from src.training.trainer import Trainer
from src.data.dataset import SkeletonDataset
from src.models.registry import ModelRegistry


def parse_arguments():
    """
    Parse command-line arguments with config override support.
    
    Returns:
        argparse.Namespace with all arguments
    """
    parser = argparse.ArgumentParser(
        description="LAST Training - Lightweight Adaptive-Shift Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================
    # Environment Selection
    # ========================
    parser.add_argument(
        '--env', '--environment',
        type=str,
        choices=['local', 'gcp', 'kaggle', 'auto'],
        default='auto',
        help='Execution environment (auto-detect by default)'
    )
    
    # ========================
    # Config File Paths
    # ========================
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data/ntu120_xsub.yaml',
        help='Path to data configuration YAML'
    )
    
    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model/last_base.yaml',
        help='Path to model configuration YAML'
    )
    
    parser.add_argument(
        '--train_config',
        type=str,
        default='configs/train/baseline.yaml',
        help='Path to training configuration YAML'
    )
    
    # ========================
    # Path Overrides
    # ========================
    parser.add_argument(
        '--data_path', '--data_root',
        type=str,
        default=None,
        help='Override data root path'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Override checkpoint directory'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Override log directory'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Override results directory'
    )
    
    # ========================
    # Training Hyperparameter Overrides
    # ========================
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--learning_rate', '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Override number of dataloader workers'
    )
    
    # ========================
    # Hardware Overrides
    # ========================
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device (cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        default=None,
        help='Enable mixed precision training'
    )
    
    parser.add_argument(
        '--no_mixed_precision',
        action='store_true',
        help='Disable mixed precision training'
    )
    
    # ========================
    # Experiment Tracking
    # ========================
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for tracking'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # ========================
    # Debugging & Testing
    # ========================
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (smaller dataset, more logging)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode (setup only, no training)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def merge_configs_with_overrides(
    config_loader: ConfigLoader,
    args: argparse.Namespace
) -> dict:
    """
    Load all configs and merge with command-line overrides.
    
    Inputs:
        config_loader: ConfigLoader instance
        args: Parsed command-line arguments
        
    Returns:
        dict: Merged configuration with overrides applied
    """
    # 1. Detect or load environment config
    if args.env == 'auto':
        env_name = EnvironmentDetector.detect_environment()
        print(f"Auto-detected environment: {env_name}")
    else:
        env_name = args.env
        print(f"Using specified environment: {env_name}")
    
    env_config = config_loader.load(f'configs/environment/{env_name}.yaml')
    
    # 2. Load data, model, and training configs
    data_config = config_loader.load(args.data_config)
    model_config = config_loader.load(args.model_config)
    train_config = config_loader.load(args.train_config)
    
    # 3. Merge all configs
    merged_config = {
        'environment': env_config,
        'data': data_config,
        'model': model_config,
        'training': train_config
    }
    
    # 4. Apply command-line overrides (Priority: CLI > Config)
    overrides = {}
    
    # Path overrides
    if args.data_path:
        overrides['data_root'] = args.data_path
    if args.checkpoint_dir:
        overrides['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        overrides['log_dir'] = args.log_dir
    if args.results_dir:
        overrides['results_dir'] = args.results_dir
    
    # Training hyperparameter overrides
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.learning_rate:
        overrides['learning_rate'] = args.learning_rate
    if args.num_workers is not None:
        overrides['num_workers'] = args.num_workers
    
    # Hardware overrides
    if args.device:
        overrides['device'] = args.device
    if args.mixed_precision is not None:
        overrides['mixed_precision'] = True
    if args.no_mixed_precision:
        overrides['mixed_precision'] = False
    
    # Experiment tracking
    if args.experiment_name:
        overrides['experiment_name'] = args.experiment_name
    if args.resume:
        overrides['resume_checkpoint'] = args.resume
    
    # Debug mode adjustments
    if args.debug:
        overrides['debug'] = True
        overrides['log_frequency'] = 10
        overrides['save_frequency'] = 1
    
    # Apply overrides to merged config
    config_loader.apply_overrides(merged_config, overrides)
    
    return merged_config


def main():
    """Main training execution."""
    # Parse arguments
    args = parse_arguments()
    
    # Load and merge configs
    config_loader = ConfigLoader()
    config = merge_configs_with_overrides(config_loader, args)
    
    # Set random seed
    from src.utils.seed import set_seed
    set_seed(args.seed)
    
    # Print configuration summary
    print("\n" + "=" * 60)
    print("LAST Training Configuration")
    print("=" * 60)
    print(f"Environment: {config['environment']['name']}")
    print(f"Data Root: {config['environment']['paths']['data_root']}")
    print(f"Model: {config['model']['name']} ({config['model']['variant']})")
    print(f"Training Mode: {config['training']['mode']}")
    print(f"Batch Size: {config['data']['dataloader']['batch_size']}")
    print(f"Epochs: {config['training']['max_epochs']}")
    print(f"Device: {config['environment']['hardware']['device']}")
    print("=" * 60 + "\n")
    
    # Dry run mode
    if args.dry_run:
        print("Dry run mode - configuration validated successfully!")
        return
    
    # Initialize components
    print("Initializing model...")
    model = ModelRegistry.create(config['model'])
    
    print("Loading datasets...")
    train_dataset = SkeletonDataset(config['data'], split='train')
    val_dataset = SkeletonDataset(config['data'], split='val')
    
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\nStarting training...\n")
    trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
```

---

## 4.7 Kaggle-Specific Workflow

### Running on Kaggle Notebooks

**Option 1: Auto-detect (Recommended)**
```python
# In Kaggle notebook cell
!python scripts/train.py
# Automatically detects Kaggle environment and uses kaggle.yaml
```

**Option 2: Explicit Kaggle mode**
```python
!python scripts/train.py --env kaggle
```

**Option 3: Override for testing**
```python
# Quick test run with reduced settings
!python scripts/train.py \
    --env kaggle \
    --batch_size 16 \
    --epochs 5 \
    --num_workers 2 \
    --debug
```

**Option 4: Custom dataset path**
```python
# Use Kaggle dataset as input
!python scripts/train.py \
    --env kaggle \
    --data_path /kaggle/input/ntu-rgbd-120-skeletons \
    --checkpoint_dir /kaggle/working/checkpoints
```

### Kaggle Notebook Setup Example

```python
# Cell 1: Install dependencies
!pip install -q -r requirements.txt

# Cell 2: Verify environment
from src.cloud.environment import EnvironmentDetector
print(f"Environment: {EnvironmentDetector.detect_environment()}")
print(f"Kaggle metadata: {EnvironmentDetector.get_kaggle_metadata()}")

# Cell 3: Run training
!python scripts/train.py \
    --env kaggle \
    --data_config configs/data/ntu120_xsub.yaml \
    --model_config configs/model/last_tiny.yaml \
    --epochs 20 \
    --batch_size 24

# Cell 4: Save results as Kaggle Dataset
from src.cloud.gcs_manager import GCSManager
# (Optionally package results for download or create output dataset)
!zip -r last_results.zip /kaggle/working/checkpoints /kaggle/working/logs
```

---

## 12. Multi-Environment Comparison Table (Updated)

| Aspect | Local | GCP | Kaggle |
|--------|-------|-----|--------|
| **Config File** | `local.yaml` | `gcp.yaml` | `kaggle.yaml` |
| **Auto-Detection** | Default | Metadata API | `/kaggle` path |
| **Data Path** | `C:/Users/...` | `/mnt/local-ssd` | `/kaggle/input` |
| **Checkpoint Path** | Local folder | `/home/pathi/last` | `/kaggle/working` |
| **GPU** | Optional | T4 (paid) | P100/T4 (free!) |
| **Max Runtime** | Unlimited | Until stopped | 9-12 hours |
| **Disk Space** | Unlimited | 100GB + SSD | 73GB (20GB usable) |
| **Num Workers** | 4-8 | 8 | 2 (CPU limit) |
| **Batch Size** | 64 | 64 | 24-32 (RAM limit) |
| **Sync to Cloud** | No | GCS auto-sync | Kaggle Datasets |
| **Auto-Shutdown** | No | Yes | Session timeout |
| **Cost** | Free | ~$0.83/hr | Free |
| **Command** | `python train.py --env local` | `python train.py --env gcp` | `python train.py --env kaggle` |

---

## 13. CLI Override Priority System

**Priority Order (Highest to Lowest):**

1. **Command-line arguments** (`--batch_size 32`)
2. **Training config YAML** (`configs/train/baseline.yaml`)
3. **Environment config YAML** (`configs/environment/kaggle.yaml`)
4. **Default values** (in code)

**Example:**
```yaml
# configs/environment/kaggle.yaml
training_overrides:
  batch_size: 32

# configs/train/baseline.yaml
training:
  batch_size: 64  # This overrides kaggle.yaml

# Command line:
python train.py --batch_size 16  # This overrides everything!
# Final batch_size: 16
```

---

## 14. Common CLI Override Patterns

### Quick Testing
```bash
# Fast test run
python train.py --debug --epochs 2 --batch_size 8

# Dry run to validate config
python train.py --dry_run --env kaggle
```

### Hyperparameter Sweeps
```bash
# Different learning rates
python train.py --lr 0.001 --experiment_name lr_001
python train.py --lr 0.0005 --experiment_name lr_0005
python train.py --lr 0.0001 --experiment_name lr_0001
```

### Environment-Specific Adjustments
```bash
# Kaggle with memory optimization
python train.py --env kaggle --batch_size 16 --num_workers 2

# GCP with full resources
python train.py --env gcp --batch_size 64 --num_workers 8

# Local CPU testing
python train.py --env local --device cpu --batch_size 4 --epochs 1
```

### Resume Training
```bash
# Resume from checkpoint
python train.py --resume checkpoints/epoch_50.pth

# Resume and change hparams
python train.py --resume checkpoints/epoch_50.pth --lr 0.0001
```

---

**This addition ensures:** Seamless execution across local, GCP, and Kaggle with zero code changes - only config/CLI flags!
