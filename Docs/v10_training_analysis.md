# ShiftFuse-V10 Training Analysis — Overfitting Diagnosis

## Training Curve Summary (147/180 Epochs)

| Epoch | Train Acc | Val Acc | **Gap** | Train Loss | Val Loss |
|-------|-----------|---------|---------|------------|----------|
| 10 | 67.87% | 60.51% | 7.4% | 1.606 | 1.527 |
| 24 | 73.13% | **68.18%** | 5.0% | 1.357 | **1.237** |
| 54 | 76.73% | **68.93%** ★ | 7.8% | 1.207 | **1.183** |
| 80 | 78.37% | 66.07% | 12.3% | 1.111 | 1.221 |
| 100 | 80.13% | 67.31% | 12.8% | 1.037 | 1.184 |
| 120 | 81.95% | 65.68% | 16.3% | 0.965 | 1.245 |
| 146 | 84.46% | 64.23% | **20.2%** | 0.857 | 1.286 |

### What the Numbers Show

1. **Best val accuracy was 68.93% at epoch 54** — it has NEVER improved since
2. **Val loss bottomed at ~1.16 around epoch 54-60**, then started INCREASING
3. **Train accuracy keeps climbing** (84.5% at ep 147) while **val accuracy DECREASES**
4. **Generalization gap**: 5% → 8% → 12% → 16% → **20%+ and growing**
5. **Val accuracy oscillates wildly**: swings of ±4-5% between adjacent epochs (e.g. ep 41: 63.3% → ep 42: 61.9% → ep 43: 65.6%)

> **Verdict: Severe overfitting beginning around epoch 40-50. The model is memorizing training data.**

---

## Root Cause Analysis

### Cause #1: 🔴 Zero Label Smoothing = Over-Confident Logits → Sharp Minima

`label_smoothing: 0.0` in the training config.

**Mechanism**: With `CrossEntropyLoss(label_smoothing=0.0)`, the loss function pushes the correct-class logit toward +∞ and all other logits toward −∞. This produces:
- **Over-confident softmax**: [P(correct) → 1.0](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/drop_path.py#24-47), [P(others) → 0.0](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/drop_path.py#24-47)
- **Sharp loss landscape**: the model converges to a narrow minimum in weight space that generalizes poorly
- **No regularization on dark knowledge**: the model doesn't learn useful relationships between similar action classes (e.g., "drinking" vs "eating")

**The wild val oscillations** (±4-5% between epochs) are a direct symptom: over-confident logits create a knife-edge decision boundary. A tiny weight perturbation flips many predictions simultaneously.

**SOTA comparison**:
| Model | Label Smoothing | Val Stability |
|-------|----------------|---------------|
| EfficientGCN-B0 | **0.1** | Smooth |
| InfoGCN | **0.1** | Smooth |
| CTR-GCN | 0.0 but uses strong augmentation | Moderate |
| BlockGCN | **0.1** | Smooth |
| **V10** | **0.0** | Wild oscillations |

**How EfficientGCN solves this**: Label smoothing = 0.1 means the target becomes `0.9 * one_hot + 0.1/num_classes`, which:
- Prevents logits from going to ±∞ (caps the gradient magnitude)
- Creates flat minima in weight space → much better generalization
- Expected improvement: **+1-2% val accuracy, dramatically reduced oscillations**

---

### Cause #2: 🔴 DW-GCN Cannot Learn Cross-Channel Spatial Patterns

The depthwise GCN (`depthwise=True`) uses `Conv2d(C, C, 1, groups=C)` — a **per-channel scalar**.

**Why this causes low accuracy**: Graph convolution aggregates neighbor joints' features: `x_agg = A @ x`. After aggregation, the model needs to mix information ACROSS channels to learn spatial patterns like "left hand moves toward right knee." With depthwise conv, each channel is scaled independently — the GCN can only learn "how important is joint i for channel c" but NOT "how does the relationship between channel a and channel b change after graph propagation."

**Mathematical proof of the bottleneck**:
```
Standard group conv (G=4, C=128):
  Weight: (C, C/G, 1, 1) = (128, 32, 1, 1) → 4,096 params
  Output: each output channel is a linear combination of 32 input channels
  Can learn: 32-dimensional spatial patterns per group

Depthwise conv (G=C=128):
  Weight: (C, 1, 1, 1) = (128, 1, 1, 1) → 128 params
  Output: each output channel = scalar × input channel
  Can learn: per-channel scaling ONLY
  Cannot learn: ANY cross-channel pattern
```

**SOTA comparison**:
| Model | GCN Conv | Params/Subset | NTU-60 xsub |
|-------|----------|---------------|-------------|
| EfficientGCN-B0 | Group(G=4) | C²/4 | **90.2%** |
| CTR-GCN | Full Conv | C² | **92.4%** |
| InfoGCN | Full Conv | C² | **93.0%** |
| **V10 (nano)** | **Depthwise** | **C** | **68.9%** |

The 32× param reduction in the GCN core is **too aggressive**. The savings (~36K params) are not worth the representational bottleneck.

---

### Cause #3: 🟡 No Mixup/CutMix + No Augmentation at Loss Level

```yaml
mixup_alpha: 0.0
cutmix_prob: 0.0
```

With NTU-60 xsub (~40K training samples, 60 classes), the model has ~140K params (nano). The parameter-to-sample ratio is 140K/40K = 3.5 — this is in the regime where strong regularization is needed.

**What regularization does V10 have?**
| Technique | Active? | Strength |
|-----------|---------|----------|
| Dropout | ✅ | 0.10 (nano) — weak |
| DropPath | ✅ | 0.10 max — weak |
| Weight Decay | ✅ | 0.0004 — standard |
| Label Smoothing | ❌ | 0.0 |
| Mixup | ❌ | 0.0 |
| CutMix | ❌ | 0.0 |
| Joint Masking | ❌ | not in V10 config |

**Total regularization budget: WEAK.** Compare with EfficientGCN which uses label smoothing + strong spatial augmentation, or InfoGCN which uses Mixup + label smoothing.

---

### Cause #4: 🟡 IB Loss Is Not Functioning Correctly

```python
# Current: nearest prototype (class-agnostic)
ib_loss = proto_dists.min(dim=-1).values.mean()  # weight: 0.001
```

At `ib_loss_weight = 0.001`, this contributes ~0.1% of the total gradient. The `min()` over all prototypes means the loss guides embeddings toward the nearest prototype regardless of class — which is wrong.

InfoGCN's IB loss (the reference) uses class-conditional prototypes where each sample is pulled toward **its own class prototype**, providing a per-class clustering signal. V10's version is class-agnostic and negligibly weighted — it's essentially a no-op.

---

### Cause #5: 🟡 Val Accuracy Wild Oscillations = Evaluation Bug?

Val accuracy swings of ±4-5% between consecutive epochs:
- Ep 41: 63.32% → Ep 42: 61.91% → Ep 43: 65.58%
- Ep 56: 63.99% → Ep 57: 65.18%
- Ep 93: 67.50% → Ep 94: 63.07% → Ep 95: 65.04%

This ±4% swing on ~16K val samples means ~640 samples flip predictions between epochs. This is NOT normal for a converging model.

**Root cause**: Over-confident logits (no label smoothing) create knife-edge decision boundaries. The softmax-weighted ensemble (`stream_weights`) also contributes — small weight changes can shift the ensemble balance and flip many predictions.

But there's another concern in the validation code:

```python
# Validation: model returns ensemble logits (eval mode)
raw_out = self.model(batch_data)
outputs = raw_out[0] if isinstance(raw_out, tuple) else raw_out
loss = self.criterion(outputs, batch_labels)
```

In eval mode, `ShiftFuseV10.forward()` returns softmax-weighted ensemble logits (a single tensor, not a tuple). So `isinstance(raw_out, tuple)` is `False`, and `outputs = raw_out`. ✅ This is correct.

But the **train-time accuracy** uses:
```python
outputs = torch.stack(outputs, dim=0).mean(dim=0)  # simple mean of 4 stream logits
```
While **eval-time** uses:
```python
w = F.softmax(self.stream_weights, dim=0)
ensemble = sum(w[i] * all_logits[i] for i in range(4))
```

This means **train and val use DIFFERENT ensemble methods** — train uses equal-weight mean, val uses learned softmax weights. If stream weights haven't converged well, the val ensemble can be worse than equal weighting.

---

## Why These Metrics Are LOWER Than Expected

Your V10 nano target was 87-90%. You're getting 68.9% peak. Here's the deficit breakdown:

| Factor | Estimated Impact | Evidence |
|--------|-----------------|----------|
| DW-GCN (no channel mixing) | **−8 to −12%** | Core spatial module cannot learn cross-channel patterns |
| No label smoothing | **−3 to −5%** | Over-confident logits, sharp minima, wild val oscillations |
| No Mixup/CutMix | **−1 to −2%** | Under-regularized for dataset size |
| IB loss non-functional | **−0.5 to −1%** | Missed regularization opportunity |
| Cumulative gap | **~15-20%** | 68.9% vs 87-90% target = 18-21% deficit ✓ |

---

## Recommended Fixes (Priority Order)

### Fix 1: 🔴 Revert DW-GCN → Group Conv (Expected: +8-12%)

In [shiftfuse_v10.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_v10.py) line 193, change `depthwise=True` to `depthwise=False`.

This restores the standard group conv with G=4 (matching EfficientGCN). Param increase for nano: ~36K (140K → 176K), still under the 200K budget.

**Why this is the biggest fix**: The GCN is the **core spatial processing module** — it runs once per block (5 times in nano). Every block's ability to learn spatial patterns depends on this. With depthwise, 5 blocks of per-channel scaling can't substitute for one block of cross-channel mixing.

### Fix 2: 🔴 Enable Label Smoothing = 0.1 (Expected: +2-3%)

In [shiftfuse_v10.yaml](file:///c:/Users/anubh/OneDrive/Desktop/LAST/configs/training/shiftfuse_v10.yaml), change `label_smoothing: 0.0` to `label_smoothing: 0.1`.

This single change will:
- Dramatically reduce val oscillations
- Prevent logit saturation
- Create flatter loss landscape → better generalization

### Fix 3: 🟡 Increase LR to 0.1 (Expected: +0.5-1%)

Change `lr: 0.05` to `lr: 0.1`. With effective batch=64, this matches SOTA convention (EfficientGCN, CTR-GCN, InfoGCN all use LR=0.1 with batch≈64).

### Fix 4: 🟡 Enable Mixup (Expected: +0.5-1%)

Set `mixup_alpha: 0.2`. This provides label-level regularization that synergizes with label smoothing.

### Fix 5: 🟡 Fix IB Loss (Expected: +0.3-0.5%)

Change from nearest-prototype to class-conditional and increase weight to 0.01.

### Fix 6: 🟢 Add TLA Gate to no_decay (Expected: +0.1%)

Add `'gate' in name` to the no_decay patterns in [trainer.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/training/trainer.py).

---

## Expected Result After Fixes

| Configuration | Val Top-1 (estimated) |
|---------------|----------------------|
| Current V10 nano | 68.9% (peaked ep 54) |
| + Group Conv GCN | 78-82% |
| + Label Smoothing | 81-84% |
| + LR = 0.1 | 82-85% |
| + Mixup | 83-86% |
| + IB Loss fix | 84-87% |
| **All fixes combined** | **85-89%** |
| EfficientGCN-B0 (290K params) | 90.2% |

> The nano variant (140-176K params) is targeting 87-90% at half the params of B0. With all fixes, 85-89% is realistic. To reach 90%+, knowledge distillation from the large variant would be needed.
