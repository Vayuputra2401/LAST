# 05 — Training

## Optimizer

**SGD** with momentum and Nesterov acceleration:
```yaml
optimizer: sgd
momentum: 0.9
nesterov: true
weight_decay: 0.0004
```

**Weight decay exclusion** — the following parameter groups have WD=0:
- Bias parameters (`bias`)
- Batch normalization weights and biases
- `alpha` parameters (ST_JointAtt gate weights + SpatialGCN subset blending weights)
- `A_learned` (adaptive adjacency in LAST-v2)
- `pool_gate` (gated head blend parameter)
- `edge` (learnable edge weights in SpatialGCN)

This prevents regularization from suppressing learned structural parameters that should be
free to diverge from zero initialization.

---

## Learning Rate Schedules

Three scheduler options, configurable via `--scheduler`:

### Option 1: `cosine_warmup` (Phase A default)

```
Phase 1: LinearLR warmup
  epochs 0–4 (5 epochs)
  lr: 0.01 → 0.1

Phase 2: CosineAnnealingLR
  epochs 5–89 (85 epochs)
  lr: 0.1 → 0.0001
```

Smooth decay, no LR cliffs. Implemented as `SequentialLR(LinearLR, CosineAnnealingLR)`.

### Option 2: `cosine_warmup_restart` (Phase D — SGDR)

```
Phase 1: LinearLR warmup
  epochs 0–4 (5 epochs)
  lr: 0.01 → 0.1

Phase 2: CosineAnnealingWarmRestarts (SGDR)
  T_0 = 30 (first cycle period)
  T_mult = 1 (constant cycle length)
  eta_min = 0.0001

  Cycle 1: epochs 5–35, lr: 0.1 → 0.0001
  Cycle 2: epochs 35–65, lr: 0.1 → 0.0001 (restart)
  Cycle 3: epochs 65–95, lr: 0.1 → 0.0001 (restart)
```

SGDR acts as **implicit regularization** — periodic warm restarts shake the model out of
sharp minima, encouraging convergence to flatter regions of the loss landscape. Used by
EfficientGCN. Configurable via:
```bash
--set training.restart_period=30 training.restart_mult=1
```

### Option 3: `multistep_warmup` (legacy, not recommended)

```
Phase 1: LinearLR warmup
Phase 2: MultiStepLR at milestones [50, 65], gamma=0.1
```

Creates aggressive 10× LR cliffs that shock BN stats and cause premature convergence.
**Not used for v3 training.**

---

## Training Hyperparameters

### Phase A+B+D Combined (Current Run)

| Parameter | Value | Config key |
|-----------|-------|------------|
| Epochs | 90 | `epochs` |
| Batch size | 32 | `batch_size` |
| Effective batch | 64 | `batch_size × gradient_accumulation_steps` |
| Input frames | 64 | `input_frames` |
| Label smoothing | 0.1 | `label_smoothing` |
| Gradient clip | 1.0 | `gradient_clip` |
| Drop path rate | 0.15 | `model.drop_path_rate` |
| use_st_att | [F, F, T] | `model.use_st_att` |
| Scheduler | SGDR | `cosine_warmup_restart` |
| AMP | enabled | `--amp` |

### Regularization Stack

| Technique | Config | Effect |
|-----------|--------|--------|
| Dropout (head) | 0.3 | Prevents classifier overfitting |
| DropPath | 0.15 (linear ramp) | Forces gradient flow through skip connections |
| Label smoothing | 0.1 | Prevents overconfident logits |
| Weight decay | 0.0004 | L2 penalty on conv weights |
| IB loss | weight=0.01 | Information bottleneck regularization (base/large) |
| Gradient accumulation | 2× | Effective batch 64 for better gradient estimates |

---

## AMP (Automatic Mixed Precision)

Enable with `--amp` flag. Uses `torch.cuda.amp.GradScaler` for FP16 forward + backward.
- ~40% VRAM reduction on all GPUs
- No observed accuracy degradation in preliminary tests
- Required on Kaggle T4 (16GB) for base variant at batch_size=32

---

## Training Commands

```bash
# Phase A+B+D (current — Kaggle T4)
python scripts/train.py \
  --model base_e_v3 --dataset ntu60 --split_type xsub \
  --epochs 90 --batch_size 32 --lr 0.1 \
  --weight_decay 0.0004 --dropout 0.3 \
  --scheduler cosine_warmup_restart --min_lr 0.0001 \
  --amp --workers 2 --seed 42 --env kaggle \
  --set training.gradient_clip=1.0 \
       training.gradient_accumulation_steps=2 \
       training.warmup_epochs=5 \
       training.warmup_start_lr=0.01 \
       training.label_smoothing=0.1 \
       training.ib_loss_weight=0.01 \
       training.save_interval=10 \
       training.nesterov=true \
       training.momentum=0.9 \
       training.restart_period=30 \
       training.restart_mult=1 \
       model.drop_path_rate=0.15 \
       model.use_st_att=false,false,true

# Local — smoke test (2 epochs)
python scripts/train.py --model nano_e_v3 --dataset ntu60 --epochs 2 --batch_size 4

# Override data root (e.g., for different dataset location)
python scripts/train.py --model base_e_v3 --dataset ntu60 --env kaggle \
  --data_root /kaggle/input/my-custom-dataset
```

### All --model choices

| Flag | Model | Params |
|------|-------|--------|
| `nano_e_v3` | LAST-E v3 nano | 83K |
| `small_e_v3` | LAST-E v3 small | 345K |
| `base_e_v3` | LAST-E v3 base | 720K |
| `large_e_v3` | LAST-E v3 large | 1.08M |
| `base` | LAST-v2 base | ~9.2M |

---

## Trainer Implementation

**File:** `src/training/trainer.py`

Key features:
- **Two-level NaN guard**: pre-forward input check + post-forward loss check with BN reset
- **AMP GradScaler** — automatically skips step on gradient overflow
- **GPU-side metric accumulation** — `.item()` called once at epoch end
- **Gradient accumulation** — configurable via `gradient_accumulation_steps`
- **Checkpoint save** — model, optimizer, scheduler, scaler, epoch, best_acc to `.pth`
- **Three scheduler options** — `cosine_warmup`, `cosine_warmup_restart` (SGDR), `multistep_warmup`

---

## Config System

**File:** `src/utils/config.py`

Environment auto-detection:
- `--env` flag provided → use that environment
- `/kaggle` exists in filesystem → uses `configs/environment/kaggle.yaml`
- Otherwise → uses `configs/environment/local.yaml`

CLI overrides via `--set KEY=VALUE` (dot-notation, auto-cast):
```bash
--set training.lr=0.1           # → float 0.1
--set model.use_st_att=false,false,true  # → [False, False, True]
--set training.nesterov=true    # → bool True
```

---

## Training Pipeline

| Step | Model | Phase | Environment | Status |
|------|-------|-------|-------------|--------|
| 1 | LAST-E v3 base | A+B+D (regularization + ablation + SGDR) | Kaggle T4 | **Running** |
| 2 | LAST-E v3 base | Evaluate convergence | Kaggle T4 | Pending |
| 3 | LAST-E v3 all variants | Full variant sweep | Kaggle T4 | Pending |
| 4 | LAST-v2 base | Teacher training | Kaggle T4 | Pending |
| 5 | LAST-E v3 → LAST-Lite | Knowledge distillation | Kaggle T4 | Planned |
| 6 | LAST-Lite | MaskCLR pretraining (if gap exists) | Kaggle T4 | Planned |
| 7 | LAST-Lite | INT8 quantization + edge deployment | Local | Planned |
