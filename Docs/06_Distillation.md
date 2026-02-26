# 06 — Distillation & Pretraining

## Overview

The deployment pipeline has three paths, used selectively:

```
Path A: Supervised Training Only
  LAST-E v3 base → train → standalone accuracy

Path B: Knowledge Distillation (KD)
  LAST-E v3 base (teacher, frozen) → distill → LAST-Lite nano/small (student)

Path C: MaskCLR Pretraining + KD
  MaskCLR self-supervised pretrain LAST-Lite → then Path B (distill from teacher)
```

**Decision tree:** Path A first. If accuracy is sufficient (>75% val), proceed to Path B.
If Path B leaves a >3% gap to EfficientGCN at same params, add Path C (MaskCLR).

---

## Part 1: Knowledge Distillation (LAST-E v3 → LAST-Lite)

### Strategy

Teacher and student share the same graph structure (SpatialGCN with D⁻½AD⁻½ normalization),
so feature-level mimicry is highly effective — spatial features have compatible semantics at
each stage.

### Loss Function

```
L = α × CE(student_logits, labels)                         # hard label loss
  + β × τ² × KL(softmax(student/τ), softmax(teacher/τ))    # soft label loss
  + γ × Σ_i MSE(proj_i(student_feat[i]), teacher_feat[i])  # feature mimicry (per stage)
```

Where:
- `α` — hard label weight (standard CE)
- `β = 1 - α` — soft label weight (teacher knowledge)
- `τ` — temperature (softens distributions; higher = softer)
- `γ` — feature mimicry weight (0 = logit-only distillation)
- `proj_i` — 1×1 conv aligning student→teacher channel dims per stage

### Recommended Hyperparameters

| Student | α | β | τ | γ | Rationale |
|---------|---|---|---|---|-----------|
| nano_lite (60K) | 0.3 | 0.7 | 4.0 | 0.1 | Big capacity gap — rely heavily on teacher |
| small_lite (180K) | 0.5 | 0.5 | 4.0 | 0.1 | Balanced — student can learn independently |

Temperature τ=4.0 is robust across model sizes. Feature mimicry γ=0.1 (light) because
same-family architectures already align well; heavy mimicry (γ=1.0) can over-constrain.

### Teacher Configuration

- **Model:** LAST-E v3 base (720K params, best checkpoint)
- **Mode:** `eval()` + `torch.no_grad()` — frozen
- **Run live:** Teacher forward-only on T4 adds ~30% batch overhead, acceptable
  (720K model is lightweight). Pre-computing logits requires ~1.4GB storage and
  complicates the pipeline.

### Why Feature Mimicry Works Here

Teacher (LAST-E v3 base) and student (LAST-Lite) share:
- Same N1-normalized adjacency structure
- Same graph partitioning (K=5 or K=3 subsets)
- Same EpSepTCN temporal module
- Same 3-stage progression

The only difference is the student REMOVES adaptive modules (gates, attention). Feature
dimensions align at each stage — only a 1×1 channel projection is needed.

---

## Part 2: MaskCLR Self-Supervised Pretraining

### What is MaskCLR?

Combines **masked autoencoding** (MAE for skeletons) with **contrastive learning** (CLR):

```
Phase 1: Self-supervised pretrain (no labels)
  ├── Mask 50–75% of joints across temporal frames
  ├── Train encoder to reconstruct masked joints (MAE objective)
  ├── Create augmented views of same sequence
  └── Pull same-sequence embeddings close, push different apart (InfoNCE)

Phase 2: Supervised fine-tune (with labels)
  ├── Take pretrained encoder
  ├── Add classification head
  └── Fine-tune (or distill from teacher with pretrained init)
```

### Why MaskCLR for LAST-Lite

Self-supervised pretraining benefits **small models MORE** than large ones:

| Aspect | Large Model (720K) | Small/Lite Model (60–180K) |
|--------|-------------------|---------------------------|
| Labeled data capacity | Can memorize NTU-60's 40K samples | Underfits — not enough capacity |
| Self-supervised benefit | Marginal (+0.5%) | **Significant (+2–4%)** |
| Why | Already has enough params for all patterns | Pretrained features give better starting point |

### LAST-Specific MaskCLR Design

1. **Graph-aware masking** — mask by body region (entire arm/leg) instead of random joints
   → forces GCN to infer missing parts via adjacency structure
2. **Temporal block masking** — mask contiguous frame blocks (not random)
   → forces temporal reasoning about motion continuity
3. **N1-normalized encoder** — uses LAST's D⁻½AD⁻½ adjacency
   → pretrained representations are LAST-specific, not transferable to EfficientGCN

### Implementation Sketch

```python
class MaskCLRPretrainer:
    def __init__(self, encoder, mask_ratio=0.5, temperature=0.07):
        self.encoder = encoder          # LAST-Lite (no classification head)
        self.decoder = nn.Linear(C_last, 3 * V)  # reconstruct masked joints
        self.projector = nn.Linear(C_last, 128)   # contrastive projection
        self.mask_ratio = mask_ratio
        self.temp = temperature

    def forward(self, x):
        # x: (B, 3, T, V)
        view1 = augment(x)     # random rotate, scale, noise
        view2 = augment(x)     # different augmentation

        view1_masked, mask1 = graph_aware_mask(view1, self.mask_ratio)
        view2_masked, mask2 = graph_aware_mask(view2, self.mask_ratio)

        z1 = self.encoder(view1_masked)  # (B, C, T', V)
        z2 = self.encoder(view2_masked)

        # Reconstruction loss (MAE)
        recon = self.decoder(z1.mean(dim=2))
        L_recon = MSE(recon[mask1], x[mask1])

        # Contrastive loss (InfoNCE)
        h1 = self.projector(z1.mean(dim=[2,3]))  # (B, 128)
        h2 = self.projector(z2.mean(dim=[2,3]))
        L_clr = NT_Xent(h1, h2, self.temp)

        return L_recon + 0.5 * L_clr
```

### When to Use MaskCLR

**Decision criteria:**
- If distillation alone achieves LAST-Lite small ≥ 88% (≥ EfficientGCN-B0): **skip MaskCLR**
- If LAST-Lite small ≤ 85% after distillation: **add MaskCLR pretraining**
- MaskCLR pretraining takes ~4 hours on T4 for one variant — acceptable cost to try

---

## Part 3: Expected Accuracy Gains

### LAST-Lite small (~180K params)

| Training Path | Est. Accuracy | vs EfficientGCN-B0 (88.3%) |
|---------------|--------------|---------------------------|
| Standalone (no pretraining, no KD) | ~84–86% | -2 to -4% |
| + Knowledge distillation from v3 teacher | ~87–89% | -1 to +1% |
| + MaskCLR pretrain + KD | ~89–91% | **+1 to +3%** |

### LAST-Lite nano (~60K params)

| Training Path | Est. Accuracy | Notes |
|---------------|--------------|-------|
| Standalone | ~80–83% | Very limited capacity |
| + KD | ~84–86% | Teacher soft labels help most here |
| + MaskCLR + KD | ~86–88% | Significant gain from pretraining |

---

## Part 4: Post-Distillation Edge Pipeline

```
LAST-Lite small (180K, FP32, ~87–91% acc)
  → INT8 Post-Training Quantization (~45KB, <1% acc drop)
  → ONNX export → TensorRT (Jetson) / TFLite (mobile/Coral) / CoreML (iOS)

  No pruning needed — models are already tiny.
  INT8 provides 4× size compression: 720KB → 180KB (enough for any edge device).
```

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Distillation trainer | `src/training/distill_trainer.py` | Planned |
| Distillation training script | `scripts/train_distill.py` | Planned |
| MaskCLR pretrainer | `src/training/maskclr.py` | Planned |
| MaskCLR pretraining script | `scripts/pretrain_maskclr.py` | Planned |
| Distillation config | `configs/distillation/default.yaml` | Planned |
| ONNX export script | `scripts/export_onnx.py` | Planned |

---

## Full Training Sequence

| Step | Action | Prerequisite | Status |
|------|--------|-------------|--------|
| 1 | Train LAST-E v3 base (Phase A+B+D) | — | **Running** |
| 2 | Evaluate val accuracy, check overfitting gap | Step 1 converges | Pending |
| 3 | Train all v3 variants (nano/small/large) | Step 2 validates approach | Pending |
| 4 | Implement LAST-Lite variant configs | Step 2 | Planned |
| 5 | Distill LAST-E v3 base → LAST-Lite small/nano | Step 3 provides teacher | Planned |
| 6 | Evaluate: is LAST-Lite small ≥ 88%? | Step 5 | Planned |
| 7 | (If needed) MaskCLR pretrain → re-distill | Step 6 shows gap | Planned |
| 8 | INT8 quantization + ONNX export | Best Lite checkpoint | Planned |
| 9 | Edge deployment demo (Jetson/mobile) | Step 8 | Planned |
