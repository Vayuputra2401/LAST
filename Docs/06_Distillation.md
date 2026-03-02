# 06 — Distillation and Pretraining

## Overview

The training pipeline has three paths, applied selectively based on standalone accuracy:

```
Path A: Supervised Training Only
  LAST-Lite → train with labels → standalone accuracy

Path B: Knowledge Distillation (KD)
  LAST-Base (teacher, frozen) → distill → LAST-Lite (student)

Path C: Self-Supervised Pretraining + KD
  MaskCLR pretrain LAST-Lite → then Path B
```

**Decision tree**: Path A first. If accuracy is sufficient (>85% val), publish standalone results. If Path A leaves a gap to EfficientGCN-B0 (90.2%), add Path B. If Path B still falls short, add Path C.

---

## Part 1: Knowledge Distillation (LAST-Base → LAST-Lite)

### Strategy

The teacher (LAST-Base) and student (LAST-Lite) operate on the same 4-stream input and share the same graph structure (K-subset adjacency with symmetric normalisation). This structural alignment makes feature-level mimicry highly effective — spatial features have compatible geometric semantics at each stage boundary.

### Loss Function

```
L = alpha * CE(student_logits, labels)                              # hard label loss
  + beta  * tau^2 * KL(softmax(student/tau), softmax(teacher/tau))  # soft label loss
  + gamma * Sum_i MSE(proj_i(student_feat[i]), teacher_feat[i])     # feature mimicry
```

Where:
- `alpha` — hard label weight (standard cross-entropy)
- `beta = 1 - alpha` — soft label weight (teacher knowledge transfer)
- `tau` — temperature (softens probability distributions; higher = softer)
- `gamma` — feature mimicry weight
- `proj_i` — 1x1 conv aligning student channel dimensions to teacher per stage

### Recommended Hyperparameters

| Student | alpha | beta | tau | gamma | Rationale |
|---------|-------|------|-----|-------|-----------|
| LAST-Lite nano (80K) | 0.3 | 0.7 | 4.0 | 0.1 | Large capacity gap — rely heavily on teacher |
| LAST-Lite small (248K) | 0.5 | 0.5 | 4.0 | 0.1 | Balanced — student has enough capacity to learn independently |

Temperature tau=4.0 is robust across model sizes. Feature mimicry gamma=0.1 (light) because the architectural families share the same graph structure and stage boundaries; heavy mimicry (gamma >= 0.5) risks over-constraining the student.

### Teacher Configuration

- **Model**: LAST-Base (planned, ~4.2M params per stream)
- **Mode**: `eval()` + `torch.no_grad()` — frozen during distillation
- **Feature extraction points**: output of each stage (3 feature maps per stream)
- **Live computation**: teacher forward pass adds ~30% batch overhead — acceptable since LAST-Base is much smaller than transformer-based teachers

### Feature Mimicry Alignment

| Stage | LAST-Lite small | LAST-Base | Projection |
|-------|----------------|-----------|------------|
| 1 | (B, 48, 64, 25) | (B, 128, 64, 25) | Conv1x1(48 -> 128) |
| 2 | (B, 72, 32, 25) | (B, 256, 32, 25) | Conv1x1(72 -> 256) |
| 3 | (B, 96, 16, 25) | (B, 384, 16, 25) | Conv1x1(96 -> 384) |

Temporal and spatial dimensions align naturally (same strides, same V=25). Only channel dimensions need projection.

---

## Part 2: MaskCLR Self-Supervised Pretraining

### Motivation

Self-supervised pretraining benefits **small models more** than large ones. A 248K-parameter model cannot memorise all 40K training samples, but pretrained features give it a better initialisation point:

| Aspect | Large Model (>1M) | Small Model (<250K) |
|--------|-------------------|---------------------|
| Labelled data capacity | Can memorise NTU-60 | Underfits — insufficient capacity |
| Self-supervised gain | Marginal (+0.5%) | **Significant (+2-4%)** |

### MaskCLR Design

Combines **masked autoencoding** (MAE) with **contrastive learning** (CLR):

1. **Masked reconstruction**: Mask 50-75% of joints across temporal frames. Train encoder to reconstruct masked joints. Uses **graph-aware masking** — masks entire body regions (arm, leg) rather than random joints, forcing the GCN to infer missing parts via adjacency structure.

2. **Contrastive learning**: Create augmented views of the same sequence. Pull same-sequence embeddings close, push different-sequence embeddings apart (InfoNCE loss).

3. **Temporal block masking**: Mask contiguous frame blocks rather than random frames, forcing temporal reasoning about motion continuity.

### When to Apply

| Condition | Action |
|-----------|--------|
| LAST-Lite small standalone >= 88% | Skip MaskCLR |
| LAST-Lite small + KD >= 90% | Skip MaskCLR |
| LAST-Lite small + KD < 88% | Add MaskCLR pretraining |

MaskCLR pretraining takes ~4 hours on T4 for one variant — acceptable cost to try.

---

## Part 3: Expected Accuracy Gains

### LAST-Lite small (247,548 params)

| Training Path | Est. Accuracy | vs EfficientGCN-B0 (90.2%) |
|--------------|--------------|---------------------------|
| Standalone (Round 3, corrected reg) | ~84-87% | -3 to -6% |
| + Knowledge distillation from LAST-Base | ~88-90% | -2 to 0% |
| + MaskCLR pretrain + KD | ~90-92% | **0 to +2%** |

### LAST-Lite nano (80,234 params)

| Training Path | Est. Accuracy | Notes |
|--------------|--------------|-------|
| Standalone (Round 3) | ~82-84% | Limited capacity |
| + KD from LAST-Base | ~85-87% | Teacher soft labels help most here |
| + MaskCLR + KD | ~87-89% | Significant gain from pretraining |

---

## Part 4: Post-Distillation Edge Pipeline

```
LAST-Lite small (248K, FP32, ~88-92% acc)
  → INT8 Post-Training Quantisation (~62KB, <1% acc drop)
  → ONNX export → TensorRT (Jetson) / TFLite (mobile) / CoreML (iOS)

  No pruning needed — models are already tiny.
  INT8 provides 4x size compression.
```

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Distillation trainer | `src/training/distill_trainer.py` | Planned |
| Distillation training script | `scripts/train_distill.py` | Planned |
| MaskCLR pretrainer | `src/training/maskclr.py` | Planned |
| ONNX export script | `scripts/export_onnx.py` | Planned |
| LAST-Base teacher model | `src/models/last_base.py` | Planned |

---

## Full Training Sequence

| Step | Action | Prerequisite | Status |
|------|--------|-------------|--------|
| 1 | Train LAST-Lite standalone (Round 3) | BSE + corrected hyperparameters | **Next** |
| 2 | Evaluate Round 3 accuracy | Step 1 | Pending |
| 3 | Implement LAST-Base model | Step 2 analysis | Planned |
| 4 | Train LAST-Base teacher | Step 3 | Planned |
| 5 | Distill LAST-Base → LAST-Lite small/nano | Step 4 | Planned |
| 6 | Evaluate: is LAST-Lite small >= 90%? | Step 5 | Planned |
| 7 | (If needed) MaskCLR pretrain → re-distill | Step 6 | Planned |
| 8 | INT8 quantisation + ONNX export | Best Lite checkpoint | Planned |
