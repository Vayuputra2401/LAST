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
| Batch size       | 16*     | `batch_size`          |
| Input frames     | 64      | `input_frames`        |
| Label smoothing  | 0.1     | `label_smoothing`     |
| Gradient clipping| 1.0     | `grad_clip`           |
| AMP              | optional| `--amp` flag          |

\* Default. Override per environment: Lambda A10 uses `--batch_size 32` (LAST-v2) or `--batch_size 128` (LAST-E).

**Config file:** `configs/training/default.yaml`

---

## AMP (Automatic Mixed Precision)

Enable with `--amp` flag. Uses `torch.cuda.amp.GradScaler` for FP16 forward + backward.
- ~40% VRAM reduction on all GPUs
- No observed accuracy degradation in preliminary tests
- Required on Kaggle T4 (16GB) for base_e at batch_size=16
- Recommended on Lambda A10 even though VRAM is larger — faster throughput

---

## Training Commands

```bash
# Lambda A10 (primary) — LAST-v2 teacher
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py --model base --dataset ntu60 --env lambda --amp --batch_size 32

# Lambda A10 — LAST-E student
python scripts/train.py --model base_e --dataset ntu60 --env lambda --amp --batch_size 128

# Kaggle T4 — LAST-E baseline
python scripts/train.py --model base_e --dataset ntu60 --env kaggle --amp

# Local — smoke test (2 epochs, small batch)
python scripts/train.py --model nano_e --dataset ntu60 --epochs 2 --batch_size 4

# Resume from checkpoint
python scripts/train.py --model base --dataset ntu60 --env lambda --amp --batch_size 32 \
  --resume /lambda/nfs/research-last/LAST-runs/run-YYYY-MM-DD/checkpoints/best_model.pth
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
- **Two-level NaN guard**:
  1. Pre-forward input check — if any MIB stream contains non-finite values, skip the batch
     entirely (BN running stats are never touched by garbage data)
  2. Post-forward loss check — if loss is NaN/Inf, reset all BN running stats to neutral
     (mean=0, var=1) and skip the backward + optimizer step
- **AMP GradScaler** — automatically skips optimizer step when gradients overflow; independent
  layer of protection on top of the explicit NaN checks
- **GPU-side metric accumulation** — loss and accuracy accumulated as tensors on GPU;
  `.item()` called once at epoch end to avoid repeated PCIe syncs
- **Gradient accumulation** — configurable via `gradient_accumulation_steps` (default=1)
- **Banner** — displays `type(self.model).__name__` at training start (not hardcoded string)
- **Checkpoint save** — saves model state, optimizer, scheduler, scaler, epoch, best_acc to `.pth`

---

## Config System

**File:** `src/utils/config.py`

Environment auto-detection:
- `--env` flag provided → use that environment
- `/kaggle` exists in filesystem → uses `configs/environment/kaggle.yaml`
- Otherwise → uses `configs/environment/local.yaml`
- Lambda requires explicit `--env lambda`

Hardware settings (num_workers, pin_memory, prefetch_factor) in environment YAMLs are
automatically applied to the DataLoader — no CLI flags needed for these.

Model name mapping:
```
base_e   → configs/model/last_e_base.yaml
small_e  → configs/model/last_e_small.yaml
large_e  → configs/model/last_e_large.yaml
nano_e   → configs/model/last_e_nano.yaml
base     → configs/model/last_base.yaml
...
```

---

## Training Sequence

| Step | Model          | Environment | Status   | Target                      |
|------|----------------|-------------|----------|-----------------------------|
| 1    | LAST-v2 base   | Lambda A10  | running  | Establish teacher accuracy  |
| 2    | LAST-E base    | Lambda A10  | pending  | Standalone student baseline |
| 3    | LAST-E (all 4) | Lambda A10  | pending  | Full variant sweep          |
| 4    | Distillation   | Lambda A10  | planned  | +2–4% over standalone       |
| 5    | NTU120         | Lambda A10  | planned  | Generalisation benchmark    |
