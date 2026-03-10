# 06 — Knowledge Distillation

## Overview

ShiftFuse V10 includes a built-in knowledge distillation (KD) pipeline that enables a larger, more accurate variant (teacher) to transfer its knowledge to the nano variant (student). KD is the planned second-stage training after standalone V10.3 results are obtained.

**Implementation status:** KD is **fully implemented** in `src/training/trainer.py`. It is disabled by default (`use_kd: false`) and activated by setting `use_kd: true` and providing a `teacher_checkpoint` path.

---

## Motivation

| Property | Nano alone | Nano + KD from Large |
|----------|-----------|---------------------|
| Params | 225,533 | 225,533 (student unchanged) |
| Knowledge source | Hard labels (one-hot CE) | Hard labels + soft teacher distributions |
| Class confusion signal | None (CE treats all wrong classes equally) | Rich (teacher encodes inter-class similarity) |
| Expected accuracy | TBD (baseline V10.2: 81.17%) | TBD (+2–5pp projected) |

**Why soft labels help small models:**
- Hard-label CE treats every wrong class as equally wrong. A nano model learning to classify "drinking water" has no gradient signal distinguishing it from "eating" (visually similar) vs "jumping" (visually dissimilar).
- A teacher's soft output (e.g., P(drinking)=0.85, P(eating)=0.10, P(pouring)=0.04) encodes the inter-class similarity structure. The student learns to match these probability distributions, acquiring implicit knowledge about which classes are confusable — the "dark knowledge" (Hinton et al., 2015).
- This effect is **stronger for smaller models** that cannot learn fine-grained discriminative features independently.

---

## Loss Function

```
L_total = (1 − α) × L_CE(student_logits, labels)
        + α × τ² × KL(σ(student_logits / τ) ∥ σ(teacher_logits / τ))
```

| Symbol | Name | Value | Rationale |
|--------|------|-------|-----------|
| α | `kd_weight` | 0.5 | Balance hard labels (student-driven) vs soft labels (teacher-driven) |
| τ | `kd_temperature` | 4.0 | Softer teacher distributions → richer class similarity signal |
| KL | KL divergence | — | Measures difference between student and teacher probability distributions |
| σ(·/τ) | Softmax at temperature τ | — | Higher τ → flatter distribution, more inter-class info |
| τ² | Temperature scaling | — | Re-scales KL to same magnitude as CE (Hinton et al., 2015) |

**Temperature effect on target distribution (τ):**

| τ | P(correct) | P(2nd class) | Information content |
|---|-----------|-------------|-------------------|
| 1 | 0.95 | 0.04 | Near one-hot; little class-similarity signal |
| 2 | 0.75 | 0.15 | Some soft information |
| **4** | **0.55** | **0.30** | **Rich inter-class similarity; recommended** |
| 6 | 0.42 | 0.38 | Very flat; can destabilise training |

---

## Teacher and Student Variants

| Role | Model | Params | Training |
|------|-------|--------|---------|
| Teacher (planned primary) | ShiftFuse V10 **large** | 3,100,506 | Trained first (standalone, no KD) |
| Teacher (alternative) | ShiftFuse V10 **small** | 1,425,050 | If large training is impractical |
| Student | ShiftFuse V10 **nano** | 225,533 | Distilled from teacher |

**Teacher training:** The large variant is trained standalone (no KD) for 240 epochs using the same V10.3 hyperparameters with adjusted regularisation (drop_path=0.20, dropout=0.25). The best checkpoint is used as the frozen teacher.

**Architectural compatibility:** All V10 variants share identical graph structure, temporal stride patterns, and number of stages (3). This means teacher and student feature maps are spatially and structurally aligned at each stage boundary — enabling optional feature-level distillation without projection layers.

---

## Configuration

```yaml
# configs/training/shiftfuse_v10.yaml
use_kd: true
teacher_checkpoint: "/kaggle/working/v10_large_best.pt"
teacher_variant: "large"
teacher_num_classes: 60
kd_weight: 0.5
kd_temperature: 4.0
```

Or via CLI:
```bash
python scripts/train.py \
    --model shiftfuse_v10_nano \
    --dataset ntu60 \
    --env kaggle \
    --amp \
    --teacher_checkpoint /path/to/large_best.pt \
    --kd_weight 0.5 \
    --kd_temp 4.0
```

---

## Implementation Details

**File:** `src/training/trainer.py`

The teacher model is loaded in eval mode with all gradients frozen:
```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
```

Per training batch:
```python
with torch.no_grad():
    teacher_logits = teacher(batch)          # (B, num_classes), float32

student_logits = student(batch, labels)      # (B, num_classes) + ib_loss

# Soft label KL divergence
kl_loss = tau**2 * F.kl_div(
    F.log_softmax(student_logits / tau, dim=-1),
    F.softmax(teacher_logits / tau, dim=-1),
    reduction='batchmean'
)

# Combined loss
loss = (1 - kd_weight) * ce_loss + kd_weight * kl_loss + ib_loss_weight * ib_loss
```

**Note:** The IB triplet loss is retained during KD — class prototypes continue to be updated with the student's feature centroids, providing complementary discriminative signal.

---

## Expected KD Training Dynamics

| Phase | Epoch range | Expected behaviour |
|-------|------------|-------------------|
| Warmup | 0–10 | LR ramps; teacher logits noisy relative to student; KL loss high |
| Rapid convergence | 10–80 | Student rapidly tracks teacher soft targets; accuracy climb |
| Refinement | 80–240 | Student fine-tunes on hard label CE; gap to teacher narrows |

**Checkpoint**: The best student checkpoint (by val acc) is saved separately from the KD run.

---

## Projected Accuracy (NTU-60 xsub nano)

| Training path | Expected top-1 | Notes |
|--------------|----------------|-------|
| V10.2 standalone (confirmed) | **81.17%** | ep177, batch=64 |
| V10.3 standalone | **TBD** (proj. 83–87%) | All Set A + Set B improvements |
| V10.3 + KD from large | **TBD** (proj. 86–90%) | +2–5pp over standalone |

*Projections based on observed KD gains in InfoGCN (+1.5pp), CTR-GCN (+2.0pp), and general KD literature for models with >3× capacity ratio.*

---

## Hyperparameter Sensitivity

### α (kd_weight) sweep

| α | Effect | Recommended for |
|---|--------|----------------|
| 0.3 | Student-led; uses teacher as regulariser | Student has sufficient capacity; teacher is weak |
| **0.5** | **Balanced; default** | **Standard recommendation** |
| 0.7 | Teacher-led; student closely mimics teacher distributions | Large capacity gap (teacher 10× larger) |

### τ (kd_temperature) sweep

| τ | Effect | Risk |
|---|--------|------|
| 2.0 | Near one-hot soft targets; small gain | Low information transfer |
| **4.0** | **Soft targets with rich class similarity; recommended** | — |
| 6.0 | Very flat distributions; maximal similarity info | Can destabilise early training |

---

## Decision Flowchart

```
V10.3 standalone training complete
        │
        ├─ val acc ≥ 88%? → Publish standalone; KD optional
        │
        └─ val acc < 88%?
                │
                ├─ Train V10.3 large (or small) as teacher
                │
                ├─ Distill large → nano (α=0.5, τ=4.0)
                │
                ├─ val acc ≥ 88%? → Publish KD results
                │
                └─ val acc < 86%?
                        │
                        └─ Consider: MaskCLR pretraining →
                           Pretrain nano backbone self-supervised →
                           Distill again (planned)
```

---

## Feature-Level Distillation (Optional Extension)

If soft-label KD alone does not close the accuracy gap, feature-level mimicry can be added at stage boundaries:

```
L_feat = Σ_s γ_s × MSE(proj_s(student_feat_s), teacher_feat_s)
```

Projections (1×1 conv) align channel dimensions at each stage:

| Stage | Student channels | Large channels | Projection |
|-------|-----------------|----------------|------------|
| 1 | 32 | 96 | Conv1×1(32 → 96) |
| 2 | 64 | 192 | Conv1×1(64 → 192) |
| 3 | 128 | 384 | Conv1×1(128 → 384) |

T and V dimensions align naturally (same strides, same V=25).

Recommended γ = 0.05–0.10 to avoid over-constraining the student's internal representations.

*Feature mimicry is not yet implemented; added as a planned extension if soft-label KD proves insufficient.*

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Soft-label KD (logit-level) | `src/training/trainer.py` | **Implemented** |
| Teacher model loading | `src/training/trainer.py` | **Implemented** |
| KD CLI args (`--teacher_checkpoint`, `--kd_weight`, `--kd_temp`) | `scripts/train.py` | **Implemented** |
| Feature-level mimicry | `src/training/trainer.py` | Planned |
| Teacher (V10 large) training | — | Pending V10.3 nano validation |
