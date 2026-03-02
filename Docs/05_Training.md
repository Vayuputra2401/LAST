# 05 — Training

## Optimiser

**SGD** with momentum and Nesterov acceleration:

```yaml
optimizer: sgd
lr: 0.1
momentum: 0.9
nesterov: true
weight_decay: 0.0005
```

### Weight Decay Exclusion

The following parameter groups are excluded from weight decay (WD=0) to prevent regularisation from suppressing learned structural parameters that should be free to diverge from zero initialisation:

| Parameter pattern | Component | Rationale |
|-------------------|-----------|-----------|
| `bias` | All layers | Standard practice |
| `bn`, `norm` | BatchNorm weights/biases | BN affine params should not be decayed |
| `alpha`, `alpha_dyn` | Graph blending weights | Structural balance parameters |
| `A_learned` | StaticGCN topology correction | Trainable adjacency should be unconstrained |
| `pool_gate` | Gated head blend | Should be free to learn optimal GAP/GMP mix |
| `freq_mask` | FDCR frequency mask | Frequency preferences should be unconstrained |
| `gate_logit` | FrameDynamicsGate | Temporal gating should not be regularised |
| `joint_embed` | JointEmbedding | Per-joint bias should be unconstrained |
| `sym_weight`, `sym_vel_weight` | BSE bilateral weights | Symmetry encoding should not be regularised |
| `node_proj` | Dynamic graph projection | Embedding projection |

---

## Learning Rate Schedule

### `multistep_warmup` (LAST-Lite default)

```
Phase 1: LinearLR warmup
  epochs 0-4 (5 epochs)
  lr: 0.01 → 0.1

Phase 2: MultiStepLR
  milestones: [60, 90]  (global epochs, adjusted internally)
  gamma: 0.1
  epoch 5-59:  lr = 0.1
  epoch 60-89: lr = 0.01
  epoch 90-119: lr = 0.001
```

Implemented as `SequentialLR(LinearLR, MultiStepLR)`. The milestones in the config are **global epochs**; the trainer internally adjusts them to MultiStepLR-relative by subtracting warmup_epochs.

The 5-epoch warmup is essential for LAST-Lite's gate stability: FDCR freq_mask and BSE gate are initialised near zero, and a warmup period allows the gates to stabilise before the main learning rate kicks in.

### `cosine_warmup` (alternative)

```
Phase 1: LinearLR warmup (5 epochs)
Phase 2: CosineAnnealingLR (remaining epochs)
  lr: 0.1 → min_lr (0.0001)
```

Smooth decay without LR cliffs. Useful for longer training runs.

---

## LAST-Lite Training Hyperparameters

### Round 3 Configuration (current)

| Parameter | Value | Config key | Notes |
|-----------|-------|------------|-------|
| Epochs | 120 | `epochs` | Extended from 90 (Round 2) |
| Batch size | 64 | `batch_size` | Single GPU, no accumulation |
| Input frames | 64 | `input_frames` | |
| Label smoothing | 0.05 | `label_smoothing` | Reduced from 0.1 (Round 2) |
| Weight decay | 0.0005 | `weight_decay` | Reduced from 0.001 (Round 2) |
| Gradient clip | 5.0 | `gradient_clip` | ShiftFuse has no adaptive modules needing tight clip |
| Scheduler | multistep_warmup | `scheduler` | 5-epoch warmup + milestones [60, 90] |
| Warmup LR | 0.01 | `warmup_start_lr` | |
| AMP | enabled | `use_amp` | Mixed precision |
| Seed | 42 | `seed` | Reproducibility |

### Regularisation Stack

| Technique | LAST-Lite nano | LAST-Lite small | Rationale |
|-----------|---------------|----------------|-----------|
| Dropout (head) | 0.1 | 0.2 | Light — small models need capacity |
| Label smoothing | 0.05 | 0.05 | Prevents overconfident logits |
| Weight decay | 0.0005 | 0.0005 | L2 on conv weights only |
| Gradient clip | 5.0 | 5.0 | Safety net, rarely triggered |

**No DropPath**: LAST-Lite has no adaptive modules to co-adapt, making stochastic depth unnecessary.

**No gradient accumulation**: Models are small enough to fit in batch 64 on a single T4 GPU.

### Round 2 → Round 3 Corrections

Round 2 revealed that nano (79.75%) beat small (78.84%) because small was **over-regularised**:

| Parameter | Round 2 | Round 3 | Diagnosis |
|-----------|---------|---------|-----------|
| Label smoothing | 0.1 | 0.05 | Too aggressive for small model |
| Weight decay | 0.001 | 0.0005 | Over-regularised small's extra capacity |
| Small dropout | 0.3 | 0.2 | Triple-whammy with LS + WD |
| Epochs | 90 | 120 | Small was still improving at epoch 90 |
| Milestones | [60, 80] | [60, 90] | Third phase 90-120 at LR=0.001 for fine-tuning |

---

## AMP (Automatic Mixed Precision)

Enabled with `use_amp: true` in config. Uses `torch.amp.GradScaler` for FP16 forward/backward.

- ~40% VRAM reduction on T4
- No observed accuracy degradation
- **Critical**: FDCR matmul is explicitly cast to fp32 before execution to prevent AMP fp16 accumulation errors (~1-2% accuracy impact if not handled)

---

## Training Commands

### LAST-Lite small (uses YAML defaults directly)

```bash
python scripts/train.py --model shiftfuse_small --dataset ntu60 --split_type xsub \
    --env kaggle --amp --avg_checkpoints 5
```

### LAST-Lite nano (CLI overrides for lighter regularisation)

```bash
python scripts/train.py --model shiftfuse_nano --dataset ntu60 --split_type xsub \
    --env kaggle --amp --avg_checkpoints 5 \
    --weight_decay 0.0003 --set training.label_smoothing=0.03
```

### Local smoke test

```bash
python scripts/train.py --model shiftfuse_nano --dataset ntu60 --epochs 2 --batch_size 4
```

### All --model choices

| Flag | Model | Params |
|------|-------|--------|
| `shiftfuse_nano` | LAST-Lite nano | 80,234 |
| `shiftfuse_small` | LAST-Lite small | 247,548 |

---

## Trainer Implementation

**File**: `src/training/trainer.py`

Key features:
- **Two-level NaN guard**: pre-forward input check + post-forward loss check with BN snapshot/restore
- **AMP GradScaler**: automatically skips optimiser step on gradient overflow
- **GPU-side metric accumulation**: `.item()` called once at epoch end to minimise CPU-GPU sync
- **Validation clamping**: val loss clamped to +/-30 to prevent NaN propagation
- **NaN counter**: tracks consecutive NaN batches, raises if threshold exceeded
- **Checkpoint averaging**: averages top-N checkpoints at end of training for better generalisation

### Config Auto-Selection

The training script automatically selects the correct training config based on model name:
- `shiftfuse_*` models → `configs/training/shiftfuse.yaml`
- All others → `configs/training/default.yaml`

### CLI Override System

**File**: `src/utils/config.py`

Supports dot-notation overrides with auto-casting:

```bash
--set training.label_smoothing=0.05      # → float
--set model.use_bilateral=true           # → bool
--set training.milestones=60,90          # → [60, 90]
```

Direct CLI flags: `--lr`, `--weight_decay`, `--dropout`, `--scheduler`, `--milestones`, `--min_lr`, `--epochs`, `--batch_size`.

---

## Checkpoint Averaging

At the end of training, the top-N checkpoints (by validation accuracy) are averaged for improved generalisation:

```bash
--avg_checkpoints 5   # Average best 5 checkpoints
```

The averaged model is evaluated on the validation set and saved as the final model. This typically provides a 0.3-0.5% accuracy boost over the single best checkpoint.

---

## Training Pipeline

| Step | Model | Description | Status |
|------|-------|-------------|--------|
| 1 | LAST-Lite nano (Round 1) | Initial training, 90 epochs, baseline config | Done (80.77%) |
| 2 | LAST-Lite nano+small (Round 2) | Regularisation + augmentation | Done (nano 79.75%, small 78.84%) |
| 3 | LAST-Lite nano+small (Round 3) | Corrected hyperparameters + BSE | **Next** |
| 4 | LAST-Base | Teacher training | Planned |
| 5 | LAST-Base → LAST-Lite | Knowledge distillation | Planned |
| 6 | LAST-Lite | MaskCLR pretraining (if gap exists) | Planned |
| 7 | LAST-Lite | INT8 quantisation + edge deployment | Planned |
