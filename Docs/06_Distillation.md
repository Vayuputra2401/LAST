# 06 — Distillation

## Strategy: LAST-v2 → LAST-E (Skeleton → Skeleton KD)

This is straightforward soft-label knowledge distillation. Teacher and student share the same
data format (MIB dict of skeleton streams), so no cross-modal adaptation is needed.

---

## Loss Function

```
L = α × CE(student_logits, labels)
  + (1 − α) × τ² × KL(
      softmax(student_logits / τ),
      softmax(teacher_logits / τ)
    )
```

Where:
- `α` — task loss weight (controls CE vs. KD balance)
- `τ` — temperature (softens probability distributions; higher τ = softer targets)
- `τ²` — scaling factor to maintain gradient magnitude despite softening

---

## Recommended Hyperparameters Per Variant

| Variant | α    | τ   | Rationale                                           |
|---------|------|-----|-----------------------------------------------------|
| nano    | 0.7  | 4.0 | Larger KD signal needed — big capacity gap          |
| small   | 0.6  | 4.0 | Moderate balance                                    |
| base    | 0.5  | 4.0 | Equal task/distillation weight — similar-ish size  |
| large   | 0.5  | 4.0 | Closest capacity to teacher; less gap to bridge    |

Temperature τ=4.0 is used uniformly because it has been found to be robust across model sizes
in prior GCN distillation work. If standalone results show underfitting (nano/small), try τ=6.0.

---

## Teacher Configuration

- **Model:** LAST-v2-base (9.2M params, `configs/model/last_v2_base.yaml`)
- **Mode during distillation:** `eval()` + `torch.no_grad()` — teacher is frozen
- **Run live vs. pre-compute:** Teacher runs live during each student training step.
  LAST-v2-base is small enough (~9.2M) that forward-only on T4 adds ~30% overhead to a
  batch of 16, which is acceptable. Pre-computing logits would require ~1.4GB extra storage
  for NTU60 and complicates the training pipeline.

---

## Expected Accuracy Gains

Estimated improvement over standalone student training:

| Variant      | Standalone est. | Post-distill est. | Delta  |
|--------------|-----------------|-------------------|--------|
| LAST-E-nano  | 85–87%          | 87–89%            | +2–4%  |
| LAST-E-small | 87–89%          | 89–91%            | +2–3%  |
| LAST-E-base  | 89–91%          | 91–92%            | +1–2%  |
| LAST-E-large | 91–93%          | 92–93%            | +0–1%  |

Smaller variants benefit more because the teacher's soft labels provide richer gradient
signal across more classes than one-hot labels alone.

---

## What We Are NOT Doing

### Cross-Modal Distillation (VideoMAE → Skeleton)
Early planning considered using a VideoMAE RGB pretrained model as teacher to transfer visual
semantics to skeleton features. This approach was **abandoned** because:
- Requires RGB video alongside every skeleton sequence (not always available)
- Cross-modal alignment is non-trivial and adds a separate pretraining phase
- LAST-v2 itself is a strong skeleton teacher with no cross-modal complexity

### MaskCLR Pretraining
Self-supervised skeleton pretraining via masked autoencoding was considered but **deferred**.
It adds significant implementation complexity for an unclear gain over direct distillation.
Decision: run baseline + distillation first; evaluate whether pretraining headroom exists
before committing to implementation.

---

## Implementation Status

Not yet built. Required components:

| Component                          | File                               | Status  |
|------------------------------------|------------------------------------|---------|
| Distillation trainer wrapper       | `src/training/distill_trainer.py`  | Planned |
| Distillation training script       | `scripts/train_distill.py`         | Planned |
| Config: α, τ, teacher_checkpoint   | `configs/distillation/default.yaml`| Planned |

---

## Training Sequence

1. Train LAST-v2-base standalone on NTU60 xsub → save best checkpoint
2. Train LAST-E-base standalone on NTU60 xsub → establish baseline (Kaggle)
3. Distill LAST-v2-base → LAST-E-nano, small, base, large (each run independently)
4. Evaluate all variants on NTU60 xsub + NTU120 xsub/xset
5. Report standalone vs. distilled accuracy in `07_Experiments.md`
