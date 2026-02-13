# LAST - Code Architecture Design

This document outlines the complete code architecture for the LAST (Lightweight Adaptive-Shift Transformer) project following clean code principles: **YAGNI**, **KISS**, **DRY**, and **SOLID**.

---

## 1. Project Directory Structure

```
LAST/
├── configs/                          # All configuration files (YAML)
│   ├── data/
│   │   ├── ntu120_xsub.yaml         # NTU-120 Cross-Subject split config
│   │   ├── ntu120_xset.yaml         # NTU-120 Cross-Setup split config
│   │   └── kinetics_skeleton.yaml   # Kinetics-Skeleton config
│   ├── model/
│   │   ├── last_base.yaml           # LAST base architecture (4 blocks)
│   │   ├── last_large.yaml          # LAST large architecture (8 blocks)
│   │   └── last_tiny.yaml           # LAST tiny for extreme edge
│   ├── train/
│   │   ├── baseline.yaml            # Phase 1: skeleton-only training
│   │   ├── distillation.yaml        # Phase 2: with teacher
│   │   └── ablation_*.yaml          # Ablation study configs
│   ├── eval/
│   │   └── evaluation.yaml          # Evaluation settings
│   ├── inference/
│   │   └── inference.yaml           # Deployment inference config
│   └── export/
│       ├── onnx.yaml                # ONNX export settings
│       └── quantization.yaml        # INT8 quantization config
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Dataset classes
│   │   ├── transforms.py            # Data augmentation
│   │   ├── preprocessing.py         # Skeleton preprocessing
│   │   └── skeleton_loader.py       # Raw .skeleton file parser
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── last.py                  # Main LAST model
│   │   ├── blocks/
│   │   │   ├── __init__.py
│   │   │   ├── agcn.py              # Adaptive GCN block
│   │   │   ├── tsm.py               # Temporal Shift Module
│   │   │   └── linear_attn.py       # Linear Attention
│   │   ├── teacher.py               # VideoMAE V2 wrapper
│   │   └── registry.py              # Model factory/registry
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training orchestrator
│   │   ├── losses.py                # Loss functions (CE, KD)
│   │   ├── optimizer.py             # Optimizer factory
│   │   └── scheduler.py             # LR scheduler factory
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Evaluation orchestrator
│   │   └── metrics.py               # Accuracy, FLOPs, latency
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py             # Real-time inference
│   │   └── video_processor.py       # Video → skeleton extraction
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   ├── onnx_exporter.py         # ONNX conversion
│   │   └── quantizer.py             # Model quantization
│   │
│   ├── cloud/                       # GCP integration (NEW)
│   │   ├── __init__.py
│   │   ├── gcs_manager.py           # Google Cloud Storage operations
│   │   ├── environment.py           # Auto-detect local vs GCP
│   │   └── instance_manager.py      # GCP instance lifecycle
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Config loader/parser
│       ├── logger.py                # Logging utilities
│       ├── checkpoint.py            # Save/load checkpoints
│       ├── visualization.py         # CAM, t-SNE plots
│       └── seed.py                  # Reproducibility
│
├── scripts/
│   ├── train.py                     # Training entry point
│   ├── eval.py                      # Evaluation entry point
│   ├── inference.py                 # Inference entry point
│   ├── export_model.py              # Export entry point
│   ├── precompute_teacher.py        # Pre-compute teacher logits
│   ├── preprocess_data.py           # Convert .skeleton → .npy
│   │
│   └── gcp/                         # GCP-specific scripts (NEW)
│       ├── upload_to_gcp.py         # Upload code to GCP instance
│       ├── download_results.py      # Download results from GCS
│       ├── setup_environment.sh     # Setup GCP VM environment
│       └── run_training.sh          # GCP training orchestrator
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_inference.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 2. Configuration Schema

### 2.1 Data Config (`configs/data/ntu120_xsub.yaml`)

```yaml
# Dataset Configuration
dataset:
  name: "NTU-RGBD-120"
  protocol: "cross_subject"  # or "cross_setup"
  
  paths:
    skeleton_dir: "/data/ntu120/skeletons"
    processed_dir: "/data/ntu120/processed"
    teacher_logits_dir: "/data/ntu120/teacher_logits"  # Optional for Phase 2
  
  splits:
    train_subjects: [1, 2, 4, 5, 8, ...]  # 53 subjects
    val_subjects: [3, 6, 7, 10, ...]      # 53 subjects
  
  preprocessing:
    num_joints: 25
    num_frames: 300              # Max temporal frames
    num_persons: 2               # Max actors per frame
    center_joint_idx: 0          # SpineBase
    normalize: true
    
  augmentation:
    enabled: true
    rotation_range: [-15, 15]   # degrees
    scale_range: [0.9, 1.1]
    shear_range: [-0.1, 0.1]
    noise_std: 0.001
    temporal_crop: true
    
  dataloader:
    batch_size: 64
    num_workers: 8
    pin_memory: true
    shuffle_train: true
```

### 2.2 Model Config (`configs/model/last_base.yaml`)

```yaml
# LAST Model Architecture
model:
  name: "LAST"
  variant: "base"
  
  input:
    num_channels: 3              # x, y, z
    num_joints: 25
    num_frames: 64               # After temporal sampling
    
  stem:
    out_channels: 64
    kernel_size: 1
    
  blocks:
    num_blocks: 4
    channels: [64, 128, 128, 256]
    
    agcn:
      num_subsets: 3             # Physical, learned, dynamic
      use_residual: true
      
    tsm:
      shift_ratio: 0.125         # 1/8 channels forward, 1/8 backward
      
    linear_attention:
      num_heads: 8
      qkv_bias: true
      dropout: 0.1
      kernel_fn: "elu"           # Feature map φ
      
  head:
    global_pool: "mean"          # or "max"
    dropout: 0.5
    num_classes: 120
    
  initialization:
    method: "kaiming_normal"
```

### 2.3 Training Config (`configs/train/distillation.yaml`)

```yaml
# Training Configuration
training:
  mode: "distillation"           # or "baseline"
  max_epochs: 100
  early_stopping:
    enabled: true
    patience: 15
    monitor: "val_acc"
    
  loss:
    classification:
      type: "CrossEntropyLoss"
      weight: 0.5                # α
      label_smoothing: 0.1
      
    distillation:
      enabled: true
      type: "KLDivergence"
      weight: 0.5                # β
      temperature: 4.0           # τ
      
  optimizer:
    type: "AdamW"
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 2
    eta_min: 1e-6
    warmup_epochs: 5
    
  gradient:
    clip_norm: 1.0
    accumulation_steps: 1
    
  mixed_precision:
    enabled: true
    
  checkpoint:
    save_dir: "checkpoints/"
    save_frequency: 5            # epochs
    keep_best_k: 3
    monitor: "val_acc"
    
  logging:
    log_dir: "logs/"
    log_frequency: 50            # iterations
    tensorboard: true
    wandb:
      enabled: false
      project: "LAST"
```

### 2.4 Evaluation Config (`configs/eval/evaluation.yaml`)

```yaml
# Evaluation Configuration
evaluation:
  metrics:
    - "top1_accuracy"
    - "top5_accuracy"
    - "per_class_accuracy"
    
  efficiency:
    measure_flops: true
    measure_params: true
    measure_latency: true
    latency_device: "cpu"        # or "cuda"
    num_runs: 100                # for latency averaging
    
  visualization:
    enabled: true
    cam_samples: 10              # Generate CAM for N samples
    tsne_classes: 20             # t-SNE for N classes
    
  output:
    save_predictions: true
    save_dir: "results/"
    confusion_matrix: true
```

---

## 3. Core Module Design

### 3.1 Data Module (`src/data/`)

#### **Class: SkeletonDataset**

**Purpose:** Main dataset class for loading skeleton data.

**Design Pattern:** Template Method (defines data loading pipeline)

```python
class SkeletonDataset:
    """
    PyTorch Dataset for skeleton-based action recognition.
    
    Inputs (constructor):
        - config: dict - Data configuration
        - split: str - 'train' or 'val'
        - transform: Optional[Callable] - Data augmentation pipeline
        
    Methods:
        __len__() -> int
            Returns: Total number of samples
            
        __getitem__(idx: int) -> dict
            Inputs: Sample index
            Returns: {
                'skeleton': torch.Tensor (C, T, V, M),
                'label': int,
                'teacher_logits': Optional[torch.Tensor] (num_classes,),
                'sample_name': str
            }
            
        _load_skeleton(file_path: str) -> np.ndarray
            Inputs: Path to .skeleton or .npy file
            Returns: Skeleton data (T, V, 3)
            
        _normalize_skeleton(skeleton: np.ndarray) -> np.ndarray
            Inputs: Raw skeleton (T, V, 3)
            Returns: Normalized skeleton (T, V, 3)
            
        _temporal_sample(skeleton: np.ndarray, target_frames: int) -> np.ndarray
            Inputs: Skeleton (T_orig, V, 3), target frame count
            Returns: Resampled skeleton (target_frames, V, 3)
    """
```

#### **Class: SkeletonFileParser**

**Purpose:** Parse raw NTU .skeleton files.

```python
class SkeletonFileParser:
    """
    Parse NTU RGB+D .skeleton text files.
    
    Methods:
        parse(file_path: str) -> dict
            Inputs: Path to .skeleton file
            Returns: {
                'num_frames': int,
                'skeletons': np.ndarray (T, max_bodies, V, 3),
                'body_info': list of dicts
            }
            
        extract_joints(skeleton_data: dict) -> np.ndarray
            Inputs: Parsed skeleton dictionary
            Returns: Clean joint coordinates (T, V, 3)
    """
```

#### **Class: SkeletonTransform**

**Purpose:** Composition of data augmentations.

**Design Pattern:** Composite (chain multiple transforms)

```python
class SkeletonTransform:
    """
    Composable transforms for skeleton data augmentation.
    
    Inputs (constructor):
        - transforms: list of callables
        
    Methods:
        __call__(skeleton: np.ndarray) -> np.ndarray
            Inputs: Skeleton (T, V, 3)
            Returns: Transformed skeleton (T, V, 3)
            
        add_transform(transform: Callable) -> None
            Inputs: Transform function
            Returns: None (modifies self.transforms)
    """
```

#### **Individual Transform Functions**

```python
class RandomRotation:
    """
    Inputs: angle_range: tuple (min_deg, max_deg)
    __call__(skeleton: np.ndarray) -> np.ndarray
    """

class RandomScale:
    """
    Inputs: scale_range: tuple (min_scale, max_scale)
    __call__(skeleton: np.ndarray) -> np.ndarray
    """

class RandomShear:
    """
    Inputs: shear_range: tuple (min_shear, max_shear)
    __call__(skeleton: np.ndarray) -> np.ndarray
    """

class GaussianNoise:
    """
    Inputs: std: float
    __call__(skeleton: np.ndarray) -> np.ndarray
    """
```

---

### 3.2 Model Module (`src/models/`)

#### **Class: LAST**

**Purpose:** Main LAST model orchestrating all components.

**Design Pattern:** Composite + Builder

```python
class LAST(nn.Module):
    """
    Main LAST model for skeleton-based action recognition.
    
    Inputs (constructor):
        - config: dict - Model configuration
        
    Attributes:
        - stem: nn.Module - Input embedding
        - blocks: nn.ModuleList - Stack of LAST blocks
        - head: ClassificationHead - Final prediction layer
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C, T, V, M) skeleton tensor
            Returns: (B, num_classes) logits
            
        extract_features(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C, T, V, M)
            Returns: (B, embed_dim) features before classification
            
        get_attention_weights() -> dict
            Returns: Dictionary of attention weights from each block
            
        count_parameters() -> int
            Returns: Total trainable parameters
            
        compute_flops(input_shape: tuple) -> int
            Inputs: Input tensor shape
            Returns: FLOPs count
    """
```

#### **Class: LASTBlock**

**Purpose:** Single LAST block containing A-GCN + TSM + Linear Attention.

```python
class LASTBlock(nn.Module):
    """
    Single LAST block: A-GCN → TSM → Linear Attention → Residual.
    
    Inputs (constructor):
        - in_channels: int
        - out_channels: int
        - config: dict - Block configuration
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C_in, T, V)
            Returns: (B, C_out, T, V)
            
        get_attention_map() -> torch.Tensor
            Returns: (B, T, V) attention weights
    """
```

#### **Class: AdaptiveGCN**

**Purpose:** Adaptive Graph Convolution with learnable adjacency.

**Design Pattern:** Strategy (different adjacency computation strategies)

```python
class AdaptiveGCN(nn.Module):
    """
    Adaptive Graph Convolution Network.
    
    Inputs (constructor):
        - in_channels: int
        - out_channels: int
        - num_joints: int (V)
        - num_subsets: int (typically 3)
        - use_residual: bool
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C_in, T, V)
            Returns: (B, C_out, T, V)
            
        _compute_physical_adjacency() -> torch.Tensor
            Returns: (V, V) fixed skeleton topology
            
        _compute_learned_adjacency() -> torch.Tensor
            Returns: (V, V) global learned matrix
            
        _compute_dynamic_adjacency(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C, T, V)
            Returns: (B, V, V) sample-dependent matrix
    """
```

#### **Class: TemporalShiftModule**

**Purpose:** Zero-parameter temporal feature exchange.

```python
class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module (TSM) for efficient temporal modeling.
    
    Inputs (constructor):
        - num_channels: int
        - shift_ratio: float (default 0.125)
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, C, T, V)
            Returns: (B, C, T, V) with shifted channels
            
        _shift_forward(x: torch.Tensor, n_channels: int) -> torch.Tensor
            Shifts n_channels forward in time
            
        _shift_backward(x: torch.Tensor, n_channels: int) -> torch.Tensor
            Shifts n_channels backward in time
    """
```

#### **Class: LinearAttention**

**Purpose:** Efficient O(T) attention mechanism.

**Design Pattern:** Strategy (different kernel functions)

```python
class LinearAttention(nn.Module):
    """
    Linear complexity attention mechanism.
    
    Inputs (constructor):
        - embed_dim: int
        - num_heads: int
        - kernel_fn: str ('elu', 'relu', etc.)
        - dropout: float
        
    Methods:
        forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
            Inputs: (B, C, T, V)
            Returns: 
                - output: (B, C, T, V)
                - attention_weights: (B, T, V)
                
        _kernel_function(x: torch.Tensor) -> torch.Tensor
            Inputs: (B, T, d)
            Returns: (B, T, d) - φ(x) = elu(x) + 1
            
        _compute_linear_attention(Q, K, V) -> torch.Tensor
            Inputs: Q, K, V tensors (B, H, T, d_k)
            Returns: (B, H, T, d_v)
    """
```

#### **Class: TeacherModel**

**Purpose:** Wrapper for VideoMAE V2 teacher.

```python
class TeacherModel:
    """
    Wrapper for VideoMAE V2 teacher model.
    
    Inputs (constructor):
        - checkpoint_path: str
        - config: dict
        - device: str
        
    Methods:
        load_model() -> None
            Loads pretrained weights and freezes parameters
            
        forward(rgb_frames: torch.Tensor) -> torch.Tensor
            Inputs: (B, C, T, H, W) RGB video
            Returns: (B, num_classes) logits
            
        precompute_logits(video_paths: list, output_dir: str) -> None
            Inputs: List of video file paths, output directory
            Returns: None (saves .npy files to disk)
            
        is_frozen() -> bool
            Returns: True if all parameters require_grad=False
    """
```

#### **Class: ModelRegistry**

**Purpose:** Factory for creating models from config.

**Design Pattern:** Factory + Registry

```python
class ModelRegistry:
    """
    Factory for model instantiation from config.
    
    Class Methods:
        register(name: str, model_class: type) -> None
            Inputs: Model name, model class
            Returns: None (registers in internal dict)
            
        create(config: dict) -> nn.Module
            Inputs: Model configuration
            Returns: Instantiated model
            
        list_models() -> list[str]
            Returns: List of available model names
    """
```

---

### 3.3 Training Module (`src/training/`)

#### **Class: Trainer**

**Purpose:** Main training orchestrator.

**Design Pattern:** Template Method + Observer (for logging)

```python
class Trainer:
    """
    Main training loop orchestrator.
    
    Inputs (constructor):
        - model: nn.Module
        - train_loader: DataLoader
        - val_loader: DataLoader
        - config: dict - Training configuration
        
    Attributes:
        - optimizer: torch.optim.Optimizer
        - scheduler: torch.optim.lr_scheduler
        - loss_fn: LossFunction
        - logger: Logger
        - checkpointer: CheckpointManager
        
    Methods:
        train() -> dict
            Returns: Training history dict
            
        _train_epoch(epoch: int) -> dict
            Inputs: Epoch number
            Returns: {
                'loss': float,
                'acc': float,
                'lr': float
            }
            
        _validate_epoch(epoch: int) -> dict
            Inputs: Epoch number
            Returns: {
                'val_loss': float,
                'val_acc': float,
                'val_top5_acc': float
            }
            
        _should_stop() -> bool
            Returns: True if early stopping criteria met
            
        save_checkpoint(epoch: int, metrics: dict) -> None
            
        load_checkpoint(checkpoint_path: str) -> None
    """
```

#### **Class: LossFunction**

**Purpose:** Unified loss computation.

**Design Pattern:** Composite (combines multiple losses)

```python
class LossFunction:
    """
    Composite loss: Classification + Distillation.
    
    Inputs (constructor):
        - config: dict - Loss configuration
        
    Methods:
        __call__(
            student_logits: torch.Tensor,
            labels: torch.Tensor,
            teacher_logits: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, dict]
            Inputs:
                - student_logits: (B, num_classes)
                - labels: (B,)
                - teacher_logits: Optional (B, num_classes)
            Returns:
                - total_loss: torch.Tensor (scalar)
                - loss_dict: {
                    'loss_ce': float,
                    'loss_kd': float,
                    'loss_total': float
                }
                
        compute_ce_loss(logits, labels) -> torch.Tensor
            
        compute_kd_loss(student_logits, teacher_logits) -> torch.Tensor
    """
```

#### **Class: OptimizerFactory**

**Design Pattern:** Factory

```python
class OptimizerFactory:
    """
    Factory for creating optimizers from config.
    
    Methods:
        create(
            model_params: Iterator,
            config: dict
        ) -> torch.optim.Optimizer
            Inputs: Model parameters, optimizer config
            Returns: Configured optimizer instance
    """
```

#### **Class: SchedulerFactory**

```python
class SchedulerFactory:
    """
    Factory for creating LR schedulers from config.
    
    Methods:
        create(
            optimizer: torch.optim.Optimizer,
            config: dict
        ) -> torch.optim.lr_scheduler._LRScheduler
            Inputs: Optimizer, scheduler config
            Returns: Configured scheduler instance
    """
```

---

### 3.4 Evaluation Module (`src/evaluation/`)

#### **Class: Evaluator**

**Purpose:** Model evaluation and metric computation.

```python
class Evaluator:
    """
    Model evaluation orchestrator.
    
    Inputs (constructor):
        - model: nn.Module
        - test_loader: DataLoader
        - config: dict - Evaluation configuration
        
    Methods:
        evaluate() -> dict
            Returns: {
                'top1_acc': float,
                'top5_acc': float,
                'per_class_acc': dict,
                'flops': int,
                'params': int,
                'latency_ms': float,
                'confusion_matrix': np.ndarray
            }
            
        _compute_accuracy(
            predictions: torch.Tensor,
            labels: torch.Tensor,
            topk: tuple = (1, 5)
        ) -> dict
            
        _measure_efficiency() -> dict
            Returns: {
                'flops': int,
                'params': int,
                'model_size_mb': float
            }
            
        _measure_latency(num_runs: int = 100) -> float
            Returns: Average latency in milliseconds
            
        save_results(output_dir: str) -> None
    """
```

#### **Class: MetricCalculator**

```python
class MetricCalculator:
    """
    Utility for computing various metrics.
    
    Static Methods:
        accuracy(
            predictions: np.ndarray,
            labels: np.ndarray,
            topk: int = 1
        ) -> float
            
        confusion_matrix(
            predictions: np.ndarray,
            labels: np.ndarray,
            num_classes: int
        ) -> np.ndarray
            
        per_class_accuracy(
            predictions: np.ndarray,
            labels: np.ndarray,
            num_classes: int
        ) -> dict
            
        count_flops(
            model: nn.Module,
            input_shape: tuple
        ) -> int
            
        count_parameters(model: nn.Module) -> int
    """
```

---

### 3.5 Inference Module (`src/inference/`)

#### **Class: Predictor**

**Purpose:** Real-time inference pipeline.

```python
class Predictor:
    """
    Real-time action prediction from skeleton data.
    
    Inputs (constructor):
        - model_path: str - Path to trained checkpoint
        - config: dict - Inference configuration
        - device: str - 'cpu' or 'cuda'
        
    Methods:
        predict(skeleton: np.ndarray) -> dict
            Inputs: Skeleton (T, V, 3)
            Returns: {
                'action': str,
                'class_id': int,
                'confidence': float,
                'all_probs': np.ndarray (num_classes,)
            }
            
        predict_batch(skeletons: list) -> list[dict]
            Inputs: List of skeletons
            Returns: List of predictions
            
        preprocess(skeleton: np.ndarray) -> torch.Tensor
            Inputs: Raw skeleton (T, V, 3)
            Returns: Preprocessed tensor (1, C, T, V, M)
            
        postprocess(logits: torch.Tensor) -> dict
            Inputs: Model output (1, num_classes)
            Returns: Prediction dictionary
    """
```

#### **Class: VideoProcessor**

**Purpose:** Extract skeletons from video using MediaPipe.

```python
class VideoProcessor:
    """
    Extract skeleton sequences from video files.
    
    Inputs (constructor):
        - pose_model: str - 'mediapipe' or 'openpose'
        
    Methods:
        process_video(video_path: str) -> np.ndarray
            Inputs: Path to video file
            Returns: Skeleton sequence (T, V, 3)
            
        process_frame(frame: np.ndarray) -> np.ndarray
            Inputs: Single RGB frame (H, W, 3)
            Returns: Skeleton (V, 3) or None if no detection
            
        map_joints_to_ntu(joints: np.ndarray) -> np.ndarray
            Inputs: MediaPipe joints (33, 3)
            Returns: NTU format joints (25, 3)
    """
```

---

### 3.6 Export Module (`src/export/`)

#### **Class: ONNXExporter**

**Purpose:** Export model to ONNX format.

```python
class ONNXExporter:
    """
    Export PyTorch model to ONNX.
    
    Methods:
        export(
            model: nn.Module,
            output_path: str,
            config: dict
        ) -> None
            Inputs: Model, output file path, export config
            Returns: None (saves .onnx file)
            
        validate(onnx_path: str, pytorch_model: nn.Module) -> bool
            Inputs: ONNX file path, original PyTorch model
            Returns: True if outputs match
    """
```

#### **Class: ModelQuantizer**

**Purpose:** Quantize model to INT8.

```python
class ModelQuantizer:
    """
    Quantize model for deployment.
    
    Methods:
        quantize_dynamic(
            model: nn.Module,
            config: dict
        ) -> nn.Module
            Inputs: FP32 model, quantization config
            Returns: INT8 quantized model
            
        quantize_static(
            model: nn.Module,
            calibration_loader: DataLoader,
            config: dict
        ) -> nn.Module
            Inputs: Model, calibration data, config
            Returns: Statically quantized model
            
        measure_size_reduction(
            original_model: nn.Module,
            quantized_model: nn.Module
        ) -> dict
            Returns: {
                'original_size_mb': float,
                'quantized_size_mb': float,
                'compression_ratio': float
            }
    """
```

---

### 3.7 Utility Module (`src/utils/`)

#### **Class: ConfigLoader**

**Purpose:** Load and validate YAML configs.

**Design Pattern:** Singleton

```python
class ConfigLoader:
    """
    Configuration loader and validator.
    
    Methods:
        load(config_path: str) -> dict
            Inputs: Path to YAML config file
            Returns: Parsed config dictionary
            
        merge_configs(base: dict, override: dict) -> dict
            Inputs: Base config, override config
            Returns: Merged configuration
            
        validate(config: dict, schema: dict) -> bool
            Inputs: Config dict, validation schema
            Returns: True if valid, raises ValueError otherwise
            
        get(key_path: str, default: Any = None) -> Any
            Inputs: Dot-notation key path (e.g., 'model.blocks.num_blocks')
            Returns: Config value or default
    """
```

#### **Class: Logger**

**Purpose:** Unified logging interface.

**Design Pattern:** Adapter (wraps multiple logging backends)

```python
class Logger:
    """
    Multi-backend logger (console + file + TensorBoard + W&B).
    
    Inputs (constructor):
        - config: dict - Logging configuration
        
    Methods:
        log_scalar(tag: str, value: float, step: int) -> None
            
        log_dict(metrics: dict, step: int) -> None
            
        log_image(tag: str, image: np.ndarray, step: int) -> None
            
        log_text(message: str, level: str = 'info') -> None
            
        close() -> None
    """
```

#### **Class: CheckpointManager**

**Purpose:** Save and load model checkpoints.

```python
class CheckpointManager:
    """
    Manage model checkpoints.
    
    Inputs (constructor):
        - save_dir: str
        - keep_best_k: int
        - monitor: str - Metric to monitor
        
    Methods:
        save(
            epoch: int,
            model: nn.Module,
            optimizer: Optimizer,
            metrics: dict
        ) -> str
            Inputs: Epoch, model, optimizer, metrics
            Returns: Path to saved checkpoint
            
        load(
            checkpoint_path: str,
            model: nn.Module,
            optimizer: Optional[Optimizer] = None
        ) -> dict
            Inputs: Checkpoint path, model, optional optimizer
            Returns: {
                'epoch': int,
                'metrics': dict
            }
            
        get_best_checkpoint() -> str
            Returns: Path to best checkpoint
            
        cleanup_old_checkpoints() -> None
            Removes checkpoints beyond keep_best_k
    """
```

#### **Class: Visualizer**

**Purpose:** Generate CAM, t-SNE, and other visualizations.

```python
class Visualizer:
    """
    Visualization utilities.
    
    Methods:
        plot_cam(
            attention_weights: np.ndarray,
            skeleton: np.ndarray,
            action_name: str,
            output_path: str
        ) -> None
            Inputs: Attention weights (T, V), skeleton (T, V, 3), action name, save path
            Returns: None (saves image)
            
        plot_tsne(
            features: np.ndarray,
            labels: np.ndarray,
            output_path: str
        ) -> None
            Inputs: Feature vectors (N, D), labels (N,), save path
            Returns: None (saves image)
            
        plot_confusion_matrix(
            cm: np.ndarray,
            class_names: list,
            output_path: str
        ) -> None
            
        plot_training_curves(
            history: dict,
            output_path: str
        ) -> None
    """
```

---

## 4. Entry Point Scripts

### 4.1 Training Script (`scripts/train.py`)

```python
"""
Training entry point.

Usage:
    python scripts/train.py --config configs/train/distillation.yaml \
                            --data_config configs/data/ntu120_xsub.yaml \
                            --model_config configs/model/last_base.yaml

Arguments:
    --config: Training configuration YAML
    --data_config: Dataset configuration YAML
    --model_config: Model architecture YAML
    --resume: Optional checkpoint path to resume training
    --device: 'cpu' or 'cuda'
    --seed: Random seed for reproducibility

Functions:
    main(args) -> None
    setup_training(config) -> tuple[Trainer, DataLoader, DataLoader]
"""
```

### 4.2 Evaluation Script (`scripts/eval.py`)

```python
"""
Evaluation entry point.

Usage:
    python scripts/eval.py --checkpoint checkpoints/best.pth \
                           --config configs/eval/evaluation.yaml \
                           --data_config configs/data/ntu120_xsub.yaml

Arguments:
    --checkpoint: Path to model checkpoint
    --config: Evaluation configuration
    --data_config: Dataset configuration
    --output_dir: Where to save results

Functions:
    main(args) -> None
    run_evaluation(evaluator, config) -> dict
"""
```

### 4.3 Inference Script (`scripts/inference.py`)

```python
"""
Real-time inference entry point.

Usage:
    python scripts/inference.py --checkpoint checkpoints/best.pth \
                                --video input_video.mp4 \
                                --config configs/inference/inference.yaml

Arguments:
    --checkpoint: Model checkpoint path
    --video: Input video file or camera ID (0 for webcam)
    --config: Inference configuration
    --visualize: Show skeleton and predictions in real-time

Functions:
    main(args) -> None
    process_video(predictor, video_processor, video_path) -> None
"""
```

### 4.4 Export Script (`scripts/export_model.py`)

```python
"""
Model export entry point.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best.pth \
                                   --format onnx \
                                   --config configs/export/onnx.yaml

Arguments:
    --checkpoint: PyTorch checkpoint
    --format: 'onnx' or 'quantized'
    --config: Export configuration
    --output: Output file path

Functions:
    main(args) -> None
    export_onnx(model, config, output_path) -> None
    export_quantized(model, config, output_path) -> None
```

### 4.5 Teacher Pre-computation Script (`scripts/precompute_teacher.py`)

```python
"""
Pre-compute teacher logits for distillation.

Usage:
    python scripts/precompute_teacher.py \
           --teacher_checkpoint teacher_videomae.pth \
           --rgb_video_dir /data/ntu120/rgb_videos \
           --output_dir /data/ntu120/teacher_logits

Arguments:
    --teacher_checkpoint: VideoMAE checkpoint path
    --rgb_video_dir: Directory with RGB .avi files
    --output_dir: Where to save logit .npy files
    --batch_size: Processing batch size

Functions:
    main(args) -> None
    precompute_batch(teacher, video_paths, output_dir) -> None
"""
```

---

## 5. Design Patterns Summary

| Pattern | Usage | Location |
|---------|-------|----------|
| **Factory** | Model/optimizer/scheduler creation | `models/registry.py`, `training/optimizer.py` |
| **Template Method** | Training/evaluation loops | `training/trainer.py`, `evaluation/evaluator.py` |
| **Strategy** | Different attention kernels, adjacency types | `models/blocks/linear_attn.py`, `models/blocks/agcn.py` |
| **Composite** | Loss combination, transform chaining | `training/losses.py`, `data/transforms.py` |
| **Singleton** | Config loader | `utils/config.py` |
| **Adapter** | Logging backends | `utils/logger.py` |
| **Observer** | Training event logging | `training/trainer.py` |
| **Builder** | Model construction | `models/last.py` |

---

## 6. Key Principles Applied

### 6.1 YAGNI (You Aren't Gonna Need It)
- No over-engineered abstractions
- Features added only when required by config
- No unused parameters or methods

### 6.2 KISS (Keep It Simple, Stupid)
- Each class has a single, clear responsibility
- Function signatures are explicit about inputs/outputs
- No deep inheritance hierarchies (max 2 levels)

### 6.3 DRY (Don't Repeat Yourself)
- Common preprocessing logic in `data/preprocessing.py`
- Metric computation centralized in `evaluation/metrics.py`
- Config loading reused across all scripts

### 6.4 SOLID Principles
- **S**ingle Responsibility: Each class does one thing well
- **O**pen/Closed: Extend via config, not code modification
- **L**iskov Substitution: All transforms/losses are swappable
- **I**nterface Segregation: Minimal required methods per interface
- **D**ependency Inversion: Depend on abstractions (config), not concrete implementations

---

## 7. Config-Driven Development Benefits

### ✅ Change Training Strategy:
```yaml
# Switch from baseline to distillation
training.mode: "distillation"  # Just one line!
```

### ✅ Try Different Model Sizes:
```yaml
# Use last_tiny.yaml instead of last_base.yaml
python train.py --model_config configs/model/last_tiny.yaml
```

### ✅ Ablation Studies:
```yaml
# Disable TSM
model.blocks.tsm.enabled: false
```

### ✅ Different Datasets:
```yaml
# Switch to Kinetics
python train.py --data_config configs/data/kinetics_skeleton.yaml
```

---

## 8. Debugging Strategy

### 8.1 Modular Testing
```python
# Test each component independently
pytest tests/test_data.py::test_skeleton_normalization
pytest tests/test_models.py::test_agcn_forward
```

### 8.2 Config Validation
```python
# Validate config before training
config_loader.validate(config, schema)
# Raises clear error if misconfigured
```

### 8.3 Logging Hooks
```python
# Every module logs to same logger
logger.log_dict({
    'module': 'AdaptiveGCN',
    'forward_time_ms': 1.23
}, step=iteration)
```

### 8.4 Checkpoint Inspection
```python
# Load and inspect any checkpoint
checkpoint = torch.load('checkpoints/epoch_50.pth')
print(checkpoint.keys())  # ['epoch', 'model', 'optimizer', 'metrics']
```

---

## 9. Next Steps

1. **Implement configs first** - Define all YAML files before code
2. **Build data pipeline** - Start with `SkeletonFileParser` and `SkeletonDataset`
3. **Implement model blocks** - A-GCN, TSM, Linear Attention incrementally
4. **Assemble LAST model** - Combine blocks with config
5. **Build trainer** - Implement training loop with Phase 1 (baseline)
6. **Test end-to-end** - Train on small subset, verify convergence
7. **Add Phase 2** - Distillation after baseline works
8. **Implement evaluation** - Metrics, visualization
9. **Deploy** - Export, quantize, real-time inference

---

**This architecture ensures:** Maintainability, Extensibility, Debuggability, and Config-Driven Flexibility. Every change happens in YAML configs, not in code.
