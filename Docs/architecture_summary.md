# LAST Code Architecture - Complete Summary

This document provides an overview of the complete LAST (Lightweight Adaptive-Shift Transformer) code architecture with multi-environment support.

---

## 📁 Complete File Structure

```
LAST/
├── configs/
│   ├── environment/              # Environment-specific configs (NEW)
│   │   ├── local.yaml           # Local machine settings
│   │   ├── gcp.yaml             # GCP instance settings
│   │   ├── gcp_instance.yaml   # GCP VM specifications
│   │   └── kaggle.yaml          # Kaggle environment settings
│   ├── data/
│   │   ├── ntu120_xsub.yaml
│   │   ├── ntu120_xset.yaml
│   │   └── kinetics_skeleton.yaml
│   ├── model/
│   │   ├── last_base.yaml
│   │   ├── last_large.yaml
│   │   └── last_tiny.yaml
│   ├── train/
│   │   ├── baseline.yaml
│   │   ├── distillation.yaml
│   │   └── ablation_*.yaml
│   ├── eval/
│   │   └── evaluation.yaml
│   ├── inference/
│   │   └── inference.yaml
│   └── export/
│       ├── onnx.yaml
│       └── quantization.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── preprocessing.py
│   │   └── skeleton_loader.py
│   ├── models/
│   │   ├── last.py
│   │   ├── blocks/
│   │   │   ├── agcn.py
│   │   │   ├── tsm.py
│   │   │   └── linear_attn.py
│   │   ├── teacher.py
│   │   └── registry.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── video_processor.py
│   ├── export/
│   │   ├── onnx_exporter.py
│   │   └── quantizer.py
│   ├── cloud/                    # Multi-cloud support (NEW)
│   │   ├── environment.py       # Auto-detect local/GCP/Kaggle
│   │   ├── gcs_manager.py       # Google Cloud Storage ops
│   │   └── instance_manager.py  # GCP instance lifecycle
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── checkpoint.py
│       ├── visualization.py
│       └── seed.py
│
├── scripts/
│   ├── train.py                 # Main training with CLI overrides
│   ├── eval.py
│   ├── inference.py
│   ├── export_model.py
│   ├── precompute_teacher.py
│   ├── preprocess_data.py
│   └── gcp/                     # GCP-specific scripts (NEW)
│       ├── upload_to_gcp.py
│       ├── download_results.py
│       ├── setup_environment.sh
│       └── run_training.sh
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

## 🌍 Multi-Environment Support Matrix

| Feature | Local | GCP | Kaggle |
|---------|-------|-----|--------|
| **Environment Detection** | Default | Metadata API | `/kaggle` path |
| **Config File** | `local.yaml` | `gcp.yaml` | `kaggle.yaml` |
| **Data Storage** | Local disk | GCS + Local SSD | Kaggle Datasets |
| **GPU** | Optional | T4 ($0.35/hr) | P100/T4 (Free) |
| **RAM** | Variable | 30 GB | 13 GB |
| **CPU Cores** | Variable | 8 vCPUs | 2 vCPUs |
| **Max Runtime** | Unlimited | Until stopped | 9-12 hours |
| **Disk Space** | Unlimited | 100GB + 375GB SSD | 73GB (20GB usable) |
| **Auto-Sync** | No | GCS bucket | Kaggle Datasets |
| **Auto-Shutdown** | No | Yes (config) | Session timeout |
| **Cost** | Free (hardware) | ~$0.83/hr | Free |
| **Best For** | Development | Full training | Quick experiments |

---

## 🚀 Quick Start Commands

### Auto-Detect Environment
```bash
# Automatically detects and uses appropriate config
python scripts/train.py
```

### Specify Environment
```bash
# Local machine
python scripts/train.py --env local

# GCP instance
python scripts/train.py --env gcp

# Kaggle kernel
python scripts/train.py --env kaggle
```

### With Overrides
```bash
# Override batch size and learning rate
python scripts/train.py --env kaggle --batch_size 16 --lr 0.0005

# Override data path
python scripts/train.py --data_path /custom/path --checkpoint_dir ./ckpts

# Quick debug run
python scripts/train.py --debug --epochs 2 --batch_size 8

# Dry run (validate config only)
python scripts/train.py --dry_run --env gcp
```

---

## 📋 Typical Workflows

### 1. Local Development & Testing
```bash
# On your laptop (Windows)
cd C:\Users\pathi\OneDrive\Desktop\LAST

# Quick test with small batch
python scripts/train.py --env local --batch_size 8 --epochs 2 --debug

# Full baseline training (CPU, slower)
python scripts/train.py --env local --train_config configs/train/baseline.yaml
```

### 2. Kaggle Experimentation
```python
# In Kaggle notebook

# Cell 1: Setup
!pip install -q -r requirements.txt

# Cell 2: Quick experiment
!python scripts/train.py \
    --env kaggle \
    --model_config configs/model/last_tiny.yaml \
    --epochs 10 \
    --batch_size 24

# Cell 3: Save results
!zip -r results.zip /kaggle/working/checkpoints /kaggle/working/logs
```

### 3. GCP Full Training
```bash
# Step 1: Upload code from local machine
python scripts/gcp/upload_to_gcp.py

# Step 2: SSH to GCP instance
gcloud compute ssh last-training-gpu --zone=asia-east1-c

# Step 3: Setup environment (on GCP)
cd ~/last
bash scripts/gcp/setup_environment.sh

# Step 4: Run training (on GCP)
bash scripts/gcp/run_training.sh
# OR with overrides:
python scripts/train.py --env gcp --epochs 150 --batch_size 64

# Step 5: Download results (back on local)
python scripts/gcp/download_results.py --experiment last_baseline_001
```

---

## 🎯 Configuration Override Priority

**Highest → Lowest Priority:**

1. **CLI Arguments** (`--batch_size 32`)
2. **Training Config** (`configs/train/baseline.yaml`)
3. **Environment Config** (`configs/environment/kaggle.yaml`)
4. **Code Defaults**

**Example:**
```bash
# Environment says batch_size=32
# Training config says batch_size=64
# CLI says --batch_size 16

python scripts/train.py --env kaggle --batch_size 16
# Final batch_size = 16 (CLI wins!)
```

---

## 🔧 Key Design Principles

### 1. YAGNI (You Aren't Gonna Need It)
- No over-engineering
- Features added only when needed
- Clean, minimal abstractions

### 2. KISS (Keep It Simple, Stupid)
- Single responsibility per class
- Explicit function signatures
- Shallow inheritance (max 2 levels)

### 3. DRY (Don't Repeat Yourself)
- Centralized config loading
- Shared preprocessing logic
- Reusable metric computation

### 4. Config-Driven Development
- **Zero code changes** between environments
- All settings in YAML files
- CLI overrides for flexibility

### 5. Environment Agnostic
- Automatic environment detection
- Path resolution (Windows/Linux/Kaggle)
- Conditional cloud integrations

---

## 📦 Main Components

### Data Pipeline
- `SkeletonDataset`: Main dataset class
- `SkeletonFileParser`: Parse NTU .skeleton files
- `SkeletonTransform`: Composable augmentations
- Auto-detect `.skeleton` or `.npy` format

### Model Architecture
- `LAST`: Main model (composite pattern)
- `LASTBlock`: A-GCN + TSM + Linear Attention
- `AdaptiveGCN`: Learnable graph convolution
- `TemporalShiftModule`: Zero-param temporal modeling
- `LinearAttention`: O(T) efficient attention
- `TeacherModel`: VideoMAE V2 wrapper

### Training System
- `Trainer`: Orchestrates full training loop
- `LossFunction`: Classification + Distillation
- `OptimizerFactory`: Creates optimizers from config
- `SchedulerFactory`: Creates LR schedulers from config

### Cloud Integration
- `EnvironmentDetector`: Auto-detect local/GCP/Kaggle
- `GCSManager`: Upload/download to Google Cloud Storage
- `InstanceManager`: GCP lifecycle (start/stop/delete)

### Evaluation & Export
- `Evaluator`: Compute metrics (accuracy, FLOPs, latency)
- `MetricCalculator`: Top-k accuracy, confusion matrix
- `ONNXExporter`: Export to ONNX format
- `ModelQuantizer`: INT8 quantization

---

## 📚 Documentation Files

1. **`code_architecture.md`** - Core architecture (original)
   - Project structure
   - Class/function signatures
   - Design patterns
   - Clean code principles

2. **`gcp_support_addition.md`** - GCP integration
   - GCP environment config
   - GCS storage integration
   - Instance management
   - Upload/download scripts

3. **`kaggle_cli_addition.md`** - Kaggle & CLI
   - Kaggle environment config
   - CLI argument parsing
   - Override system
   - Multi-environment workflows

4. **`architecture_summary.md`** - This file
   - Complete overview
   - Quick reference
   - Common workflows

---

## 🎓 Next Steps

### Phase 1: Setup (Week 1)
1. Create all config files (`configs/environment/*.yaml`)
2. Implement `EnvironmentDetector` class
3. Test auto-detection on local/GCP/Kaggle
4. Implement `ConfigLoader` with override support

### Phase 2: Data Pipeline (Week 2)
1. Implement `SkeletonFileParser`
2. Build `SkeletonDataset`
3. Create transforms (augmentations)
4. Test on small NTU subset

### Phase 3: Model (Week 3-4)
1. Implement `AdaptiveGCN`
2. Implement `TemporalShiftModule`
3. Implement `LinearAttention`
4. Assemble `LASTBlock`
5. Build full `LAST` model

### Phase 4: Training (Week 5)
1. Implement `Trainer` class
2. Build `LossFunction` (CE + KD)
3. Test Phase 1 training (baseline, skeleton-only)
4. Verify convergence on small dataset

### Phase 5: Cloud Integration (Week 6)
1. Implement `GCSManager`
2. Implement `InstanceManager`
3. Test GCP upload/training/download workflow
4. Test Kaggle kernel execution

### Phase 6: Evaluation & Export (Week 7)
1. Implement `Evaluator`
2. Build metric computation
3. Implement ONNX export
4. Add quantization support

### Phase 7: Full Training (Week 8+)
1. Download NTU RGB videos
2. Pre-compute teacher logits
3. Phase 2 training (distillation)
4. Ablation studies
5. Paper writing

---

## ✅ Key Benefits

✅ **Multi-Environment**: Runs on local, GCP, Kaggle without code changes  
✅ **Config-Driven**: Everything controlled via YAML files  
✅ **CLI Overrides**: Flexible command-line argument system  
✅ **Auto-Detection**: Automatically detects execution environment  
✅ **Cloud Integration**: Automatic GCS sync, auto-shutdown  
✅ **Modular**: Clean separation of concerns (SOLID principles)  
✅ **Extensible**: Easy to add new models, losses, datasets  
✅ **Debuggable**: Clear interfaces, comprehensive logging  
✅ **Cost-Effective**: Free Kaggle for experiments, paid GCP for full runs  

---

## 🔥 Most Common Commands

```bash
# Development (local)
python scripts/train.py --debug --epochs 1

# Testing (Kaggle)
python scripts/train.py --env kaggle --model_config configs/model/last_tiny.yaml

# Production (GCP)
python scripts/train.py --env gcp --train_config configs/train/distillation.yaml

# Hyperparameter sweep
python scripts/train.py --lr 0.001 --experiment_name lr_001
python scripts/train.py --lr 0.0005 --experiment_name lr_0005

# Resume training
python scripts/train.py --resume checkpoints/epoch_50.pth

# Validation only
python scripts/eval.py --checkpoint checkpoints/best.pth

# Export model
python scripts/export_model.py --checkpoint checkpoints/best.pth --format onnx
```

---

**Architecture Designed For:** Maximum flexibility, minimal code changes, seamless multi-environment execution!
