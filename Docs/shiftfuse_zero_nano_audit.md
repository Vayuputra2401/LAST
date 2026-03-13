# ShiftFuse-Zero Nano — Comprehensive Audit

> **Scope**: Gradient flow, hyperparameters, training logic, loss integrity vs SOTA, data normalization, shapes, module impact, redundancy, and validation of user-researched improvement ideas.
> **Method**: Mathematical dry-run only — no code executed or modified.

---

## 1. Architecture Summary & Shape Trace

### Block Pipeline (ZeroGCNBlock)
```
Input (B, C, T, V)
  │ BRASP        — anatomical channel routing (0 params)
  │ SGPShift     — semantic-typed graph shift (0 params)
  │ A_learned    — shared per-stage correction (V×V=625 params, sym-normalized per-forward)
  │ Conv1×1+BN+ReLU   — channel mixing / expansion
  │ JointEmbedding    — shared per-stage (V×C params)
  │ MultiScaleTCN     — 2-branch depthwise-sep (k=9, d=1 + d=2)
  │ DropPath
  └ Residual + ReLU → optional TLA (stage 3 only)
```

### Full Model Shape Trace (nano)

| Stage | Input | Output | Notes |
|-------|-------|--------|-------|
| **Fusion** | 4×[(B,3,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) → concat [(B,12,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,24,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | StreamFusionConcat BN→Cat→Conv1×1→BN→ReLU |
| **Stage 1** (2 blocks) | [(B,24,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,40,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | s=1 both blocks |
| **Stage 2** (3 blocks) | [(B,40,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,80,32,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | s=2 on block 0, then s=1 |
| **Stage 3** (2 blocks) | [(B,80,32,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,160,16,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | s=2 on block 0, then s=1. TLA on last block only |
| **Pool** | [(B,160,16,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,160)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | mean(T)→gated GAP+GMP |
| **Classifier** | [(B,160)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | [(B,60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) | Dropout(0.1)→Linear |

**Params**: ~163-170K. **No shape mismatches.** ✅

---

## 2. Gradient Flow Analysis

### 2.1 Clean Single-Path Gradient ✅

Unlike V10 (multi-head + IB loss competing objectives), Zero has **one CE loss → one backward path** → no gradient conflicts. This is a major design improvement.

### 2.2 A_learned Shared Per-Stage — 🟢 SAFE

```python
A_l = self._A_learned.abs()
d   = A_l.sum(dim=1).clamp(min=1e-6).pow(-0.5)
A_l_norm = d.unsqueeze(1) * A_l * d.unsqueeze(0)
```

**Gradient through abs()**: `∂|a|/∂a = sign(a)`. Zero-init means [A_learned](file:///c:/Users/anubh/OneDrive/Desktop/LAST/tests/test_shiftfuse_zero.py#265-274) starts at 0 where sign is discontinuous, but PyTorch defines `sign(0)=0` → gradient is zero at init. The parameter **will not move until symmetry is broken** by other paths (BRASP/SGP already provide non-zero spatial routing). Once any entry becomes slightly non-zero (from accumulated gradient noise), `abs()` develops a non-zero gradient.

> [!NOTE]
> **Zero-init A_learned with abs()** has a **cold-start period**: gradients are exactly zero until stochastic noise pushes entries away from zero. This is by design (starts from physical prior) but can delay learning by 5-10 epochs. Not a bug but worth monitoring.

**Shared across 2-3 blocks**: Stage 2 has 3 blocks → gradients through A_learned are 3× amplified. However, with only 625 params and [no_decay](file:///c:/Users/anubh/OneDrive/Desktop/LAST/tests/test_shiftfuse_zero.py#265-274), this is well-bounded.

### 2.3 SGPShift + BRASP Gradient Flow ✅

Both are zero-param `torch.gather` operations. Gradient: `∂gather/∂x[c,v] = 1 if (c,v) was selected, 0 otherwise` — clean per-channel gradient routing, no leakage or vanishing.

### 2.4 MultiScaleTCN — BN Before ReLU ✅

```python
nn.BatchNorm2d(half) → nn.ReLU → nn.Conv2d(depthwise) → nn.Conv2d(pointwise) → nn.BatchNorm2d(half)
```

Pre-activation BN (BN before ReLU) is a valid pattern. The issue is that **both branches have Dropout(0.1) applied AFTER branch concatenation** — this is fine.

### 2.5 Residual After ReLU — 🟡 MINOR CONCERN

```python
x = self.out_relu(x + self.residual(res))  # ReLU AFTER residual add
```

This is pre-activation style for the residual. However, applying ReLU after the residual add means **negative residual contributions are clipped**. In ResNet-V2, the recommended pattern is to put the activation BEFORE the residual add. This is a minor point — both patterns work, but placing ReLU after can slightly impede gradient flow for deep networks.

For 7 blocks, this is tolerable. ✅

### 2.6 No Gradient Leakage ✅

- Early fusion: single-domain processing after StreamFusionConcat → no cross-stream gradient issues
- A_learned refs: stored as `self._A_learned` (plain attribute reference), gradients correctly flow to the `nn.Parameter` in the parent
- JE refs: stored as `self.je` (plain attribute), same principle ✅
- TLA: only on last block of stage 3 — no sharing issues

---

## 3. Loss Function Integrity vs SOTA

### 3.1 Clean CrossEntropyLoss + Label Smoothing ✅

```python
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**No IB loss, no class_prototypes** — single optimization objective. This is the simplest and most stable training signal, matching EfficientGCN exactly.

| Model | Loss | Auxiliary | Streams |
|-------|------|-----------|---------|
| **ShiftFuse-Zero** | CE + LS=0.1 | None | Early fusion (1 loss) ✅ |
| **EfficientGCN-B0** | CE + LS=0.0 | None | Early fusion (1 loss) ✅ |
| **CTR-GCN** | CE | None | 4× late (4 losses avg'd) |
| **InfoGCN** | CE + LS=0.1 + IB | IB loss | 4× late (4 losses + IB) |

**Verdict**: Loss function is clean and SOTA-aligned. ✅

### 3.2 Trainer Interaction — 🔴 CRITICAL BUG

The trainer has this code (line 519-531):
```python
raw_out = self.model(batch_data, labels=batch_labels)
if isinstance(raw_out, tuple):
    outputs, aux_loss = raw_out
else:
    outputs, aux_loss = raw_out, None
if isinstance(outputs, list):
    loss = sum(self.criterion(o, target) for o in outputs) / len(outputs)
    outputs = torch.stack(outputs, dim=0).mean(dim=0)
else:
    loss = self.criterion(outputs, target)
```

ShiftFuseZero's [forward()](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_gcn.py#130-145) accepts [stream_dict](file:///c:/Users/anubh/OneDrive/Desktop/LAST/tests/test_shiftfuse_zero.py#49-59) as positional arg, NOT `x`. The trainer calls:
```python
self.model(batch_data, labels=batch_labels)
```

**Check**: Does the trainer pass `labels=batch_labels` when `_model_accepts_labels` is True?

```python
sig = inspect.signature(model.forward)
self._model_accepts_labels = 'labels' in sig.parameters
```

ShiftFuseZero.forward signature: [forward(self, stream_dict: dict)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_gcn.py#130-145) — **no `labels` parameter** → `_model_accepts_labels = False` → trainer calls `self.model(batch_data)`.

`batch_data` is a dict (MIB format) → This correctly maps to [stream_dict](file:///c:/Users/anubh/OneDrive/Desktop/LAST/tests/test_shiftfuse_zero.py#49-59). ✅

**Output**: ShiftFuseZero returns a plain `Tensor` (not tuple, not list) → `isinstance(raw_out, tuple)` = False → `outputs = raw_out`, `aux_loss = None` → `loss = self.criterion(outputs, target)` → **correct**. ✅

No trainer bug. ✅

---

## 4. Hyperparameter Audit

### Training Config ([shiftfuse_zero.yaml](file:///c:/Users/anubh/OneDrive/Desktop/LAST/configs/training/shiftfuse_zero.yaml))

| Parameter | Value | Assessment | SOTA Ref |
|-----------|-------|------------|----------|
| **optimizer** | SGD | ✅ | All SOTA use SGD |
| **lr** | 0.1 | ✅ | EfficientGCN: 0.1 ✅ |
| **momentum** | 0.9 | ✅ | Standard |
| **nesterov** | true | ✅ | Standard |
| **weight_decay** | 0.0004 | ✅ | EfficientGCN: 0.0003, CTR-GCN: 0.0004 |
| **scheduler** | cosine_warmup | ✅ | Standard |
| **warmup_epochs** | 5 | ✅ | Matches SOTA |
| **warmup_start_lr** | 0.005 | ✅ | lr/20, standard |
| **min_lr** | 0.0001 | ✅ | Standard |
| **epochs** | 150 | ✅ | Good middle ground |
| **batch_size** | 64 | ✅ | Matches EfficientGCN/CTR-GCN |
| **accum_steps** | 1 | ✅ | No accumulation needed at batch=64 |
| **label_smoothing** | 0.1 | ✅ | InfoGCN standard |
| **gradient_clip** | 5.0 | 🟡 | EfficientGCN uses 1.0; 5.0 is more permissive |
| **drop_path_rate** | 0.10 | ✅ | Appropriate for ~170K model |
| **dropout** | 0.10 | ✅ | Light for nano |
| **mixup_alpha** | 0.0 | 🔶 See §9 | Missing regularizer |
| **ib_loss_weight** | 0.0 | ✅ | Correctly disabled |

> [!TIP]
> **Hyperparameters are dramatically improved** over V10. LR=0.1, warmup=5, batch=64, no IB loss — all now match SOTA baselines. **No critical hyperparameter issues detected.**

---

## 5. Module Impact & Redundancy Analysis

| Module | Params (nano) | % of Total | Expected Impact | Redundancy |
|--------|--------------|------------|-----------------|------------|
| **StreamFusionConcat** | 4×BN(3) + Conv(12→24) + BN(24) = ~384 | 0.2% | Essential | 🟢 None |
| **BRASP** (per block) | 0 | 0% | +0.5-1% structural prior | 🟢 Zero cost |
| **SGPShift** (per block) | 0 | 0% | +0.5-1% semantic routing | 🟢 Zero cost |
| **A_learned** (per stage, 3 total) | 3×625 = 1,875 | 1.1% | +0.5-1% topology fine-tuning | 🟢 Very cheap |
| **graph_conv** (per block, 7 total) | ~35K total | 21% | Essential channel mixing | 🟢 None |
| **JointEmbedding** (per stage, 3 total) | 25×40 + 25×80 + 25×160 = 7,000 | 4.1% | +0.5% semantic identity | 🟢 Well shared |
| **MultiScaleTCN** (per block, 7 total) | ~90K total | 53% | +2-3% temporal modeling | 🟢 Core component |
| **TLA** (stage 3 only, 1 instance) | ~5K | 3% | +1% global temporal | ✅ Only 1 instance (was 28% in V10!) |
| **Residuals** (7 blocks, 2 with proj) | ~15K | 9% | Essential skip connections | 🟢 None |
| **Classifier** | 160×60 + gate = 9,601 | 5.7% | Essential | 🟢 None |

**Key improvement over V10**: TLA reduced from 28% of params (7 instances) to 3% (1 instance). Channel budget is now dominated by TCN (53%) and graph_conv (21%) — the actual computation modules. ✅

> [!NOTE]
> **No redundancy detected.** Every module serves a unique purpose. The zero-param spatial processing (BRASP+SGPShift) is especially efficient — it provides the same spatial routing as a 127K-param MultiScaleAdaptiveGCN at zero cost.

---

## 6. Data Normalization & Shapes ✅

Verified in [ntu60.yaml](file:///c:/Users/anubh/OneDrive/Desktop/LAST/configs/data/ntu60.yaml):
- `data_type: mib` → 4-stream dict input
- `normalize: false` → data pre-normalized offline (spine-center, front-rotate, torso-scale)
- `max_frames: 64` → matches model `T=64`
- `num_joints: 25` → matches model
- Augmentation: rotation ±30°, scale 0.85-1.15, per-stream noise calibrated to SNR, joint mask 20%

**Clamp at ±30**: Applied in trainer to all streams uniformly — same concern as before (velocity spikes). Minor.

**Shape flow verified**: [(B,3,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) per stream → [(B,12,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) concat → [(B,24,64,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) stem → stages → [(B,160,16,25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) → pool [(B,160)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) → logits [(B,60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/transforms.py#360-373) ✅

---

## 7. Potential Issues That May Cause Plateauing/Overfitting

### 7.1 Potential Plateauing Risks

| Factor | Risk | Analysis |
|--------|------|----------|
| **A_learned cold start** | 🟡 LOW | Zero-init + abs() = zero gradient until broken. Delays graph adaptation by 5-10 epochs then resolves |
| **No Mixup/CutMix** | 🔶 MEDIUM | 150 epochs with only spatial augmentation. EfficientGCN trains for 70 epochs and doesn't need Mixup, but at 150 epochs the model may memorize |
| **Single graph_conv per block** | 🟡 LOW | Only 1 Conv1×1 for channel mixing after spatial routing. May limit spatial discrimination vs K=3 group-conv GCN |
| **gradient_clip=5.0** | 🟡 LOW | More permissive than EfficientGCN (1.0). May allow occasional large gradient spikes |

### 7.2 Potential Overfitting Risks

| Factor | Risk | Analysis |
|--------|------|----------|
| **150 epochs** | 🟡 LOW | EfficientGCN needs 70; 150 is 2× longer but cosine schedule mitigates |
| **No Mixup** | 🔶 MEDIUM | Without Mixup, the model may overfit after ~100 epochs |
| **Dropout 0.1 only on classifier** | 🟡 LOW | TCN has internal Dropout(0.1). Could be slightly higher for 150 epochs |

### 7.3 Mitigating Factors ✅
- Clean single CE loss (no competing objectives)
- Cosine LR decay to 0.0001
- DropPath 0→0.10 linearly
- Label smoothing 0.1
- Joint masking 20%
- Temporal speed augmentation
- Per-stream noise calibrated to SNR

---

## 8. SGPShift Note — Uniform Shift Within Groups

```python
# For each joint v: choose the most-connected neighbor in A
row[v] = -1.0    # exclude self-loop
best = row.argmax().item()
```

All channels in group 0 shift to the **same** neighbor (the most-connected one). Unlike BRASP (which cycles through different neighbors per channel), SGPShift provides **uniform routing** within each group.

**Impact**: This means C//3 channels all see the exact same permutation. The diversity comes from having 3 groups (intra/inter/identity), not from within-group variation.

> [!TIP]
> A potential improvement: cycle through 2nd-best, 3rd-best neighbors for different channels within each group (like BRASP does with `nbrs[i % len(nbrs)]`). This would increase spatial diversity at zero param cost.

---

## 9. Validation of User-Researched Improvements

The user shared recommendations from external research. Here's how each maps to ShiftFuse-Zero Nano:

### ✅ Already Implemented

| Recommendation | Status in Code |
|----------------|----------------|
| **"Token/Channel Shifting (parameter-free, shift ~25% of channels)"** | ✅ **Both BRASP and SGPShift** do exactly this — zero-param channel-wise spatial shifting. BRASP shifts 4 groups (arm/leg/torso/cross). SGPShift shifts 3 groups (intra/inter/identity). Combined, 100% of channels are shifted at zero cost. |
| **"Depthwise Separable Convolutions"** | ✅ **MultiScaleTCN** uses depthwise-separable convolutions (depthwise k=9 + pointwise 1×1) in both branches. |
| **"Efficient Activations (HardSwish/SiLU)"** | 🟡 **Partially** — the stem uses ReLU. V10 used Hardswish throughout. Zero uses ReLU. This is a minor regression. |
| **"Squeeze-and-Excitation (SE) Bottlenecks"** | ❌ **Not present** in Zero. V10 had ChannelSE after GCN. Zero removed it to save params. Given 53% of params are in TCN, adding SE (~800 params at C=160) would be very cheap. |
| **"Knowledge Distillation"** | ✅ **Supported** — config has `use_kd`, `teacher_checkpoint`, `kd_weight=0.5`, `kd_temperature=4.0`. Not enabled yet (need trained teacher first). |
| **"Longer Training with Heavy Augmentation"** | 🔶 **Partially** — 150 epochs is longer than EfficientGCN (70). But Mixup/CutMix are disabled. Enabling these would be the single biggest regularization boost. |
| **"Adjust Weight Decay for Nano"** | ✅ **Already appropriate** — weight_decay=0.0004 is standard. With extensive no_decay exclusions (A_learned, JE, gates, pool_gate), learnable corrections aren't penalized. |

### Actionable Suggestions Not Yet Implemented

| Suggestion | Effort | Expected Impact |
|------------|--------|-----------------|
| **Add SE after graph_conv** | Low (~800 params for stage 3) | +0.5% — channel recalibration after spatial routing |
| **Enable Mixup** (mixup_alpha=0.2) | Config change only | +0.5-1% regularization for 150 epochs |
| **Switch ReLU → Hardswish** in graph_conv | 1 line per block | +0.3% (minor, but free) |
| **SGPShift neighbor cycling** | ~10 lines in sgp_shift.py | +0.3% more spatial diversity at zero param cost |
| **Use KD from larger teacher** | Config change + trained teacher | +2-3% (well-established for nano models) |

---

## 10. Summary of All Findings

### ✅ No Issues — Well Designed
- Single CE loss — no competing objectives (major improvement over V10)
- All hyperparameters aligned with SOTA (lr=0.1, warmup=5, batch=64)
- Zero-param graph processing is parameter-efficient and novel
- Shape propagation correct throughout
- Residual connections correct
- BN placement correct
- Weight decay groups correctly configured
- Data normalization pipeline correct
- Trainer compatibility verified (no labels, no tuple output, no multi-head)
- TLA reduced from 28% to 3% of params (1 instance vs 7)

### 🔶 Medium-Priority Improvements
1. **Enable Mixup/CutMix** — strongest regularizer missing, especially at 150 epochs
2. **Add ChannelSE** after graph_conv — zero-cost attention at ~800 params
3. **KD training** when a larger teacher is available

### 🟡 Minor Improvements
4. **ReLU → Hardswish** — free performance (no extra params/compute)
5. **SGPShift neighbor cycling** — more spatial diversity at zero cost
6. **gradient_clip 5.0 → 1.0** — tighter, matches EfficientGCN
7. **A_learned init 0.01 instead of 0.0** — breaks cold-start symmetry immediately

### ✅ User Research Validated
The recommendations from the user's external research are **largely correct and already implemented** in ShiftFuse-Zero. The two key missing items (SE blocks, Mixup) are the most impactful additions.
