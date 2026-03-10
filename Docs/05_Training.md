# 05 — Training

## Overview

ShiftFuse V10 uses a standard SGD + cosine-warmup training protocol, calibrated for stability with the IB triplet loss, AMP float16, and the shared-GCN architecture. All V10.3 changes to the training configuration are documented with explicit rationale.

**Config file:** `configs/training/shiftfuse_v10.yaml`
**Model config:** `configs/model/shiftfuse_v10_nano.yaml`

---

## Optimiser

**SGD with Nesterov momentum:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | 0.1 | Standard for SGD + cosine on NTU-60; matches InfoGCN, CTR-GCN |
| `momentum` | 0.9 | Standard |
| `nesterov` | True | Lookahead gradient update; faster convergence |
| `weight_decay` | 0.0004 | L2 regularisation on conv weights; lower than typical (0.0005) to avoid over-regularising the narrow nano channels |

**Why SGD over Adam?** SGD with cosine LR generalises better for CNNs/GCNs on classification benchmarks. Adam adapts per-parameter LR, which can over-fit to small gradients in shared modules (GCN, JE). Momentum-SGD with a large initial LR and long cosine decay is empirically stronger for this task.

---

## Weight Decay Exclusion (no_decay groups)

Parameters excluded from weight decay (set to WD=0) because L2 regularisation would incorrectly penalise structural parameters that should diverge from zero:

| Pattern | Component | Rationale |
|---------|-----------|-----------|
| `bias` | All biases | Standard practice |
| `bn`, `norm` | BatchNorm γ/β | Affine params should not be decayed |
| `class_prototypes` | IB prototype embeddings | Should freely move to class centroids |
| `stream_weights` | Learned ensemble weights | Should freely specialise per-stream |
| `alpha`, `alpha_dyn` | Dynamic graph blending weights | Structural balance; should be unconstrained |
| `gate` | TLA gate scalar | Attention gate should not be regularised |
| `joint_embed` | JointEmbedding bias table | Per-joint identity; should be unconstrained |
| `gcn_scale` | Per-block GCN output scalar | Scale guard; should be unconstrained |
| `pool_gate` | Gated head GAP/GMP blend | Pooling preference; should be unconstrained |
| `A_learned`, `A_group` | Learnable adjacency corrections | Topology corrections; should be unconstrained |

**Implementation:** `src/training/trainer.py` — param group split via name-based keyword matching.

---

## Learning Rate Schedule

### `cosine_warmup` (V10 default)

```
Phase 1: Linear Warmup
  Epochs 0 – warmup_epochs (10)
  LR: warmup_start_lr (0.005) → lr (0.1)
  Purpose: Prevent large gradients from disrupting TLA gates and
           class_prototypes (zero-init) in the first epochs.

Phase 2: Cosine Annealing
  Epochs 10 – 240
  LR: 0.1 → min_lr (0.0001)
  Smooth decay: no sharp LR cliffs that cause re-memorisation.
```

**V10.3 change — warmup_epochs 5 → 10:**
With the triplet IB loss active from epoch 1, the proto_dists_wrong computation produces noisy `d_wrong` estimates in early epochs (prototypes are zero-init, all classes equidistant). A 10-epoch warmup allows prototypes to separate before the IB gradient carries significant weight. 5 epochs was insufficient — IB loss caused gradient spikes in epochs 6–9 that destabilised BN running statistics.

**LR schedule plot:**
```
LR
0.100 ┤                     ╭─────────────────────────────────╮
      │                   ╭─╯                                   ╲
      │                 ╭─╯                                       ╲
0.050 ┤               ╭─╯                                           ╲
      │             ╭─╯                                               ╲
      │           ╭─╯                                                   ╲
0.010 ┤         ╭─╯                                                       ╲
0.005 ┤ ────────╯                                                           ╲
0.000 ┤─────────────────────────────────────────────────────────────────────── epoch
      0   10   30   50   70   90   110  130  150  170  190  210  230  240
      ↑ warmup end ↑
```

---

## Full Hyperparameter Table (ShiftFuse V10.3 nano)

| Parameter | Config key | Value | Change from V10.2 |
|-----------|-----------|-------|-------------------|
| **Optimiser** | | | |
| Optimiser | `optimizer` | SGD | — |
| Learning rate | `lr` | 0.1 | — |
| Momentum | `momentum` | 0.9 | — |
| Nesterov | `nesterov` | True | — |
| Weight decay | `weight_decay` | 0.0004 | — |
| **Scheduler** | | | |
| Scheduler | `scheduler` | cosine_warmup | — |
| Warmup epochs | `warmup_epochs` | **10** | **5 → 10 (V10.3)** |
| Warmup start LR | `warmup_start_lr` | 0.005 | — |
| Min LR | `min_lr` | 0.0001 | — |
| **Training duration** | | | |
| Epochs | `epochs` | 240 | 180 → 240 |
| Batch size | `batch_size` | 24 | 64 → 24 (gradient accum.) |
| Gradient accumulation | `gradient_accumulation_steps` | 3 | Added |
| Effective batch | (computed) | **72** | Similar to prior 64 |
| **Regularisation** | | | |
| Label smoothing | `label_smoothing` | 0.1 | — |
| Gradient clip | `gradient_clip` | 5.0 | — |
| Dropout (head) | `dropout` | 0.10 | — |
| Mixup alpha | `mixup_alpha` | 0.0 (disabled) | — |
| CutMix prob | `cutmix_prob` | 0.0 (disabled) | — |
| **Loss** | | | |
| IB loss weight | `ib_loss_weight` | **0.01** | **0.001 → 0.01 (V10.3)** |
| **Precision** | | | |
| AMP | `use_amp` | True | — |
| torch.compile | `use_compile` | False | — |
| **Input** | | | |
| Input frames | `input_frames` | 64 | — |
| **DataLoader** | | | |
| Num workers | `num_workers` | 4 (local) / 2 (Kaggle) | — |
| Pin memory | `pin_memory` | True | — |
| **Logging** | | | |
| Log interval | `log_interval` | 20 batches | — |
| Eval interval | `eval_interval` | 1 epoch | — |
| Save interval | `save_interval` | 5 epochs | — |
| **Reproducibility** | | | |
| Seed | `seed` | 42 | — |

---

## IB Loss Weight Calibration

**V10.3 change — ib_loss_weight 0.001 → 0.01:**

The IB loss gradient relative to CE gradient is:

```
|∇_ib|   ib_loss_weight × ∂ib/∂θ
───── = ─────────────────────────
|∇_CE|         ∂CE/∂θ
```

At weight 0.001, the IB triplet gradient was below the gradient noise floor for the first ~50 epochs — effectively a no-op. At weight 0.01, the IB gradient contributes ~1–2% of the CE gradient (calibrated to be informative without dominating). InfoGCN uses 0.0001 for an attraction-only IB loss; our triplet version produces larger gradient magnitudes per sample, justifying the 100× increase.

**Expected training signature:**
- Epochs 1–10 (warmup): IB loss high (~0.5 = margin), decreasing slowly
- Epochs 10–50: IB loss decreasing rapidly as prototypes separate
- Epochs 50+: IB loss near 0 for easy samples, active on boundary samples

---

## Regularisation Stack

| Technique | Value | Applied to | Effect |
|-----------|-------|-----------|--------|
| DropPath | linear 0→0.10 | All blocks, ramp over depth | Stochastic depth; forces multi-path gradients |
| Dropout | 0.10 | Classification head | Prevents head over-fit |
| Label smoothing | 0.10 | CE loss | Softens targets; pairs with soft IB prototypes |
| Weight decay | 0.0004 | Conv/Linear weights | L2 regularisation |
| Gradient clip | 5.0 | All gradients | Prevents explosions in shared GCN + IB |

**No Mixup/CutMix:** The combination of AMP float16 + IB triplet loss on mixed batches + the 10-epoch warmup period creates NaN risks. The triplet IB loss requires `argmin` over prototype distances, which is undefined for linearly mixed labels (soft labels give no clear nearest-wrong-prototype). Mixup re-evaluation is planned for V10.4.

---

## AMP (Automatic Mixed Precision)

ShiftFuse V10 uses AMP with special float32 guards:

| Module | Float32 guard | Reason |
|--------|--------------|--------|
| TLA softmax | `q = q_proj(x_t.float())` | BMM accumulation in float16 overflows for T=64, K=14 |
| IB proto distances | `cdist(mean_feat.float(), ...)` | Euclidean distance is sensitive to float16 rounding |
| GradScaler | `torch.amp.GradScaler` | Skips optimiser step on gradient overflow |

All other layers (conv, BN, linear) run in float16 for ~40% VRAM reduction on T4.

---

## Trainer Features

**File:** `src/training/trainer.py`

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| IB loss routing | `inspect.signature` checks `labels` arg | Only passes labels to V10 models; prevents TypeError on other models |
| BN snapshot/restore | Copy running stats before forward; restore on NaN | Prevents BN corruption on bad batches |
| NaN counter | Consecutive NaN batch limit | Raises if training degenerates |
| Val loss clamping | `clamp(±30)` | Prevents NaN propagation from val |
| GPU metric accumulation | `.item()` once at epoch end | Minimises CPU-GPU sync overhead |
| Checkpoint saving | Top-K by val acc | Saves best N models for averaging |
| KD integration | Teacher forward in `no_grad` | Adds soft-label KD loss if `use_kd=True` |

---

## Training Commands

### V10 nano (primary)

```bash
# Kaggle
python scripts/train.py \
    --model shiftfuse_v10_nano \
    --dataset ntu60 \
    --split_type xsub \
    --env kaggle \
    --amp

# Local smoke test (2 epochs)
python scripts/train.py \
    --model shiftfuse_v10_nano \
    --dataset ntu60 \
    --epochs 2 \
    --batch_size 4 \
    --gradient_accumulation_steps 1
```

### V10 large → nano KD (after large is trained)

```bash
python scripts/train.py \
    --model shiftfuse_v10_nano \
    --dataset ntu60 \
    --split_type xsub \
    --env kaggle \
    --amp \
    --teacher_checkpoint /path/to/large_best.pt \
    --kd_weight 0.5 \
    --kd_temp 4.0
```

---

## Training Timeline

| Round | Model | Key changes | Status | Best val |
|-------|-------|-------------|--------|----------|
| V10.1 | nano | Initial V10 architecture, batch=64, 180ep | Done | **81.17%** (ep177) |
| V10.2 | nano | BN fix (no StreamBN), TLA enabled, blocks=[2,3,2] | Done | — |
| **V10.3** | **nano** | **Set A+B: BRASP fix, triplet IB, TLA K=14, warmup 10ep, IB weight 0.01, share_gcn+scale, share_je, learned ensemble** | **Pending** | **TBD** |
| V10.4 | nano | (Planned) Mixup re-evaluation, longer training | Planned | — |
| V10-KD | nano | KD from V10.3 large (or V10.3 small) | Planned | — |

---

## Expected V10.3 Training Dynamics

Based on component analysis and prior V10.1 baseline (81.17%):

| Epoch range | Expected behaviour |
|-------------|-------------------|
| 0–10 (warmup) | LR 0.005→0.1; IB loss high; TLA gates at 50% activity |
| 10–50 | Rapid accuracy gain; IB prototypes separating; gcn_scale adapting per-block |
| 50–150 | Steady improvement; stream_weights diverging from uniform |
| 150–240 | Fine-tuning; LR at cosine tail (~0.005); checkpoint averaging window |

**Projected gains over V10.1 (81.17%):**

| Change | Projected gain |
|--------|---------------|
| BRASP before pw_conv (A1) | +1.0–2.0pp |
| Triplet IB loss (A2) | +0.5–1.5pp |
| TLA K=8→14 (A3) | +0.3–0.7pp |
| Warmup 5→10ep (A4) | +0.2–0.5pp |
| IB weight 0.001→0.01 (A5) | +0.3–0.8pp |
| share_gcn+gcn_scale (B1) | ±0.2pp (neutral to slight gain) |
| share_je (B2) | ±0.1pp (neutral) |
| Learned ensemble (B3) | +0.2–0.5pp |
| **Total projected** | **+2.5–6.0pp → 83–87%** |
