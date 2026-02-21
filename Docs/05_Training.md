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
- `alpha` parameters (ST_JointAtt gate weights + DirectionalGCNConv subset blending weights)
- `A_learned` (adaptive adjacency in LAST-v2)

This prevents regularization from suppressing learned structural parameters that should be
free to diverge from zero initialization.

---

## Learning Rate Schedule

Two-phase schedule implemented as `torch.optim.lr_scheduler.SequentialLR`:

```
Phase 1: LinearLR warmup
  epochs 0–9 (10 epochs)
  lr: 0.001 → 0.01

Phase 2: CosineAnnealingLR
  epochs 10–69 (60 epochs)
  lr: 0.01 → 1e-5
```

`SequentialLR` is fully checkpoint-safe: state is saved and restored correctly on resume
without requiring epoch offset arithmetic.

---

## Training Hyperparameters

| Parameter        | Value   | Config key            |
|------------------|---------|-----------------------|
| Epochs           | 70      | `epochs`              |
| Batch size       | 16      | `batch_size`          |
| Input frames     | 64      | `input_frames`        |
| Label smoothing  | 0.1     | `label_smoothing`     |
| Gradient clipping| 1.0     | `grad_clip`           |
| AMP              | optional| `--amp` flag          |

**Config file:** `configs/training/default.yaml`

---

## AMP (Automatic Mixed Precision)

Enable with `--amp` flag. Uses `torch.cuda.amp.GradScaler` for FP16 forward + backward.
- ~40% VRAM reduction on T4/P100
- No observed accuracy degradation in preliminary tests
- Required on Kaggle (16GB T4) for batch_size=16 with base_e

---

## Training Commands

```bash
# Kaggle (T4 16GB) — LAST-E baseline
python scripts/train.py --model base_e --dataset ntu60 --env kaggle --amp

# GCP P100 — LAST-v2 teacher
python scripts/train.py --model base --dataset ntu60 --env gcp --amp

# Local — smoke test (2 epochs, small batch)
python scripts/train.py --model nano_e --dataset ntu60 --epochs 2 --batch_size 4

# Resume from checkpoint
python scripts/train.py --model base_e --dataset ntu60 --env kaggle --amp \
  --resume /path/to/best_model.pth
```

### All --model choices

| Flag       | Model            | Params  |
|------------|------------------|---------|
| `nano_e`   | LAST-E nano      | 92K     |
| `small_e`  | LAST-E small     | 178K    |
| `base_e`   | LAST-E base      | 364K    |
| `large_e`  | LAST-E large     | 644K    |
| `small`    | LAST-v2 small    | ~4.8M   |
| `base`     | LAST-v2 base     | ~9.2M   |
| `large`    | LAST-v2 large    | ~14M    |

---

## Trainer Implementation

**File:** `src/training/trainer.py`

Key features:
- **NaN guard**: checks loss for NaN before backward; logs and skips batch if NaN detected
- **GPU-side metric accumulation**: confusion matrix and running accuracy accumulated on GPU,
  transferred to CPU only at epoch end (reduces PCIe overhead)
- **Gradient accumulation**: configurable via `grad_accum_steps` (default=1)
- **Banner**: displays `type(self.model).__name__` at training start (not hardcoded string)
- **Checkpoint save**: saves model state, optimizer, scheduler, epoch, best_acc to `.pth`

---

## Config System

**File:** `src/utils/config.py`

Environment auto-detection:
- `/kaggle` exists in filesystem → uses `configs/environment/kaggle.yaml`
- Otherwise → uses `configs/environment/local.yaml`
- `--env` flag overrides auto-detection

Model name mapping:
```
base_e   → configs/model/last_e_base.yaml
small_e  → configs/model/last_e_small.yaml
large_e  → configs/model/last_e_large.yaml
nano_e   → configs/model/last_e_nano.yaml
base     → configs/model/last_v2_base.yaml
...
```

---

## Training Sequence (Planned)

1. **Kaggle baseline** — `LAST-E-base`, NTU60 xsub, standalone (no distillation)
2. **GCP teacher** — `LAST-v2-base`, NTU60 xsub, standalone
3. **Distillation runs** — LAST-v2-base → LAST-E (all 4 variants)
4. **NTU120** — repeat on larger dataset after NTU60 results confirmed
