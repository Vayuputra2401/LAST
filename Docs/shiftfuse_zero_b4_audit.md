# ShiftFuse-Zero Large B4 — Comprehensive Audit

**Target**: 92.5% Top-1 on NTU-60 xsub (B4 baseline: 92.1%)

---

## 1. Architecture Dry Run — Shape Trace

### Input Shape
3 streams (J/V/B), each [(B, 3, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) for single-body, [(B, 3, 64, 25, 2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) for dual-body.
Dual-body: cat along batch → [(2B, 3, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40).

### Per-Stream Path (independent weights ×3)
| Layer | Out Shape | Notes |
|-------|-----------|-------|
| [StreamFusionConcat(3→64, 1-stream)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/stream_fusion_concat.py#26-69) | [(2B, 64, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) | BN(3)→concat→Conv1×1(3→64)→BN→HSwish |
| Stage 1: [EfficientZeroBlock(64→96, ×2, stride=1)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 96, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) | Block 0: in=64→out=96. Block 1: in=96→out=96 |
| Stage 2: [EfficientZeroBlock(96→48, ×2, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 48, 32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) | Block 0: stride=2 halves T. Block 1: stride=1 |

### Fusion
`torch.cat([J_feat, V_feat, B_feat], dim=1)` → [(2B, 144, 32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) — **no projection conv, direct concat**.

### Shared Path
| Layer | Out Shape | Notes |
|-------|-----------|-------|
| Shared Stage 1: [EfficientZeroBlock(144→128, ×3, stride=1)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 128, 32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) | Block 0: 144→128, blocks 1-2: 128→128 |
| Shared Stage 2: [EfficientZeroBlock(128→272, ×3, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 272, 16, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) | Block 0: stride=2 halves T. Last block has TLA |

### Head
- `x.mean(dim=2)` → [(2B, 272, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40) — temporal GAP
- Gated GAP+GMP over joints: `g·mean(V) + (1-g)·max(V)` → [(2B, 272)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40)
- `Dropout(0.25)→Linear(272→60)` → [(2B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40)
- If multi-body: `(out[:B] + out[B:]) / 2` → [(B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/cross_stream_fusion.py#32-40)

> **Shape Verdict**: ✅ All shapes trace correctly. No dimension mismatches.

---

## 2. Gradient Flow Analysis — Mathematical Dry Run

### 2.1 GCN: Double Normalization Problem

> [!CAUTION]
> **CRITICAL: The GCN applies normalization TWICE on A_fixed.**

The graph construction in [graph.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/graph.py) with `raw_partitions=True` feeds raw 0/1 adjacency to [normalize_symdigraph_full()](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/graph.py#176-206) → output A_sym is already D^{-1/2} A D^{-1/2} normalized.

Then in `EfficientZeroBlock.forward()` (lines 217-219):
```python
A_comb = (A_fixed + A_learnt).abs()           # A_fixed is ALREADY normalized
d      = A_comb.sum(dim=1).clamp(min=1e-6).pow(-0.5)
A_norm = d.unsqueeze(1) * A_comb * d.unsqueeze(0)   # Second normalization!
```

**Impact**: A_fixed values are ~0.1-0.3 (pre-normalized). After `.abs()` and re-normalizing, the row sums shrink further → **signal attenuation at every block**. Over 3+6=9 shared blocks, this can compound to severe gradient vanishing.

**EfficientGCN-B4 comparison**: B4 uses `A_fixed` (pre-normalized) + `A_learned` additively with only ONE normalization step. The learnable part starts at zero, so the initial forward pass uses the correctly-normalized fixed adjacency directly.

**Severity**: HIGH. This is likely a root cause of plateauing. The double-norm compresses the adjacency spectrum, reducing the effective receptive field of each GCN layer.

### 2.2 TLA gate init = 0.0 → sigmoid(0) = 0.5

The TLA gate is initialized to `torch.zeros(1)`, so `sigmoid(0) = 0.5`. This means TLA output is added with 50% weight from epoch 1.

**Problem**: At epoch 1, the Q/K/V projections are random. The TLA produces random attention-weighted features that immediately influence the final shared block output at 50% strength. This injects noise into the classifier's input features during early training.

**EfficientGCN-B4**: No temporal attention → cleaner signal flow. Models that do use attention (InfoGCN, BlockGCN) typically use a smaller init (~0.01) or a progressive warmup.

**Fix**: Should initialize gate to -4.0 like STC-Attention does (sigmoid(-4)≈0.018).

**Severity**: MEDIUM. Won't prevent convergence but adds noise during early training, potentially slowing convergence by 5-10 epochs.

### 2.3 STC-Attention placed BEFORE GCN → spatial softmax over 25 joints

STC spatial attention: `softmax(FC(mean(x, dim=(C,T))))` over V=25 → values ~1/25 = 0.04.

Combined with temporal sigmoid (~0.5) and channel sigmoid (~0.5):
```
x * A_s * A_t * A_c ≈ x * 0.04 * 0.5 * 0.5 = x * 0.01
```

**With the residual gate** (init -4.0 → scale ≈ 0.018):
```
output = x*(1-0.018) + x*0.01*0.018 ≈ 0.982·x + 0.0002·x ≈ x
```

**Verdict**: ✅ The residual gate correctly handles this. STC starts as near-identity and gradually activates. This is correctly implemented.

### 2.4 Hardswish activation after GCN + residual

```python
x = self.out_act(x + self.residual(res))   # Hardswish(inplace=True)
```

Hardswish: `x * relu6(x+3)/6`. For small x (near zero), gradient ≈ x/3 + 0.5. For large x, gradient ≈ 1. For x < -3, gradient = 0.

**Risk**: After double-normalized GCN (attenuated output) + residual (potentially still small due to channel projection), the sum could land in the low-gradient zone of Hardswish.

**EfficientGCN-B4**: Uses ReLU, which has gradient = 1 for x > 0. No low-gradient zone for positive values.

**Severity**: LOW-MEDIUM. Hardswish is standard in efficient models. But combined with the signal attenuation from double normalization, it could compound.

### 2.5 DropPath at 0.25 max rate

Linear schedule from 0.0 (first block) to 0.25 (last block). With total_blocks = 3×4 + 6 = 18:
- Last block: drop_prob = 0.25 → during training, the main branch is zeroed 25% of the time.
- With 6 shared blocks, the last 3 have drop rates: 0.194, 0.222, 0.25.

**B4 comparison**: EfficientGCN-B4 uses 0% drop path (no stochastic depth). The 0.25 rate here is aggressive for 150 epochs, especially when combined with other regularization (WD=1e-4, dropout=0.25, label_smoothing=0.1).

**Severity**: MEDIUM. Over-regularization risk. The model capacity (~1.12M) is similar to B4 (~1.1M), but B4 achieves 92.1% with zero drop path and only WD + simple augmentation in 70 epochs.

---

## 3. Training Configuration Audit

### 3.1 Optimizer & LR Schedule

| Param | B4 Config | EfficientGCN-B4 (original) | Verdict |
|-------|-----------|--------------------------|---------|
| Optimizer | SGD+Nesterov | SGD+Nesterov | ✅ |
| LR | 0.1 | 0.1 | ✅ |
| Momentum | 0.9 | 0.9 | ✅ |
| Weight Decay | 1e-4 | 1e-4 | ✅ |
| Scheduler | Cosine+warmup(10ep) | Cosine | ⚠️ Warmup is fine but 10 epochs at LR=0.005 is long for 150 total |
| Min LR | 1e-5 | 0 | ✅ Having a floor is fine |
| Epochs | 150 | 70 | ⚠️ 2x+ longer, see below |

### 3.2 Regularization Stack Analysis

| Technique | Value | B4 | Effect |
|-----------|-------|-----|--------|
| Weight Decay | 1e-4 | 1e-4 | ✅ |
| Dropout (head) | 0.25 | 0.25 | ✅ |
| Drop Path | 0.25 | 0.0 | ⚠️ Novel addition |
| Label Smoothing | 0.1 | 0.0 | ⚠️ Novel addition |
| Mixup | 0.0 | 0.0 | ✅ |
| CutMix | 0.0 | 0.0 | ✅ |

**Cumulative regularization**: WD + Dropout(0.25) + DropPath(0.25) + LabelSmoothing(0.1).

This is **substantially more regularization than B4**, which only uses WD + Dropout. The extra regularization may be suppressing the higher-capacity novel components (STC, TLA, A_k_learned) from learning, causing the model to plateau at a lower accuracy than B4.

> [!WARNING]
> **Over-regularization is the #1 training-side cause of plateauing.** The model has ~1.12M params (similar to B4's 1.1M), but 3 additional regularization mechanisms. The novel components (STC, A_k_learned, TLA) need room to learn — excessive regularization fights their parameter updates.

### 3.3 Batch Size Conflict

Training config YAML: `batch_size: 64`
Data config YAML: `batch_size: 32`

The dataloader config says 32, but the training config says 64. Which one is actually used depends on how the config is merged in [scripts/train.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py).

**Severity**: MEDIUM. If the actual effective batch size is 32 with LR=0.1, it may be too high an LR for that batch. If 64, it matches B4's setup well.

### 3.4 Epochs: 150 vs B4's 70

150 epochs with cosine-warmup is 2.14× longer than B4's 70. With the cosine schedule, LR reaches near-zero around epoch 130-140. The last 20 epochs contribute almost nothing to learning.

With the added regularization, the model may converge to a flatter (lower-accuracy) minimum by epoch 80-90 and then stagnate for the remaining 60+ epochs.

---

## 4. Loss Function Integrity

### 4.1 CrossEntropyLoss with Label Smoothing

```python
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

EfficientGCN-B4: plain CrossEntropyLoss (no smoothing).

Label smoothing with ε=0.1 converts hard targets `y=k` to `y_soft[k] = 0.9, y_soft[j≠k] = 0.1/59`. This lowers the theoretical maximum training accuracy to ~90% (since the model converges toward the soft distribution, not 100%).

**Impact on metrics**: Training accuracy will plateau around 88-90% by design. If you're monitoring train acc as a plateau signal, this is expected behavior, not a bug. Val accuracy should still be able to exceed 92% if the model generalizes well.

**SOTA comparison**: InfoGCN uses label smoothing 0.1 and achieves 93.0%. CTR-GCN doesn't use it and achieves 92.4%. It's fine for val accuracy but makes train accuracy a misleading metric.

### 4.2 Accuracy computed AFTER mixup/interpolation

```python
accs = _accuracy_topk(outputs.detach(), batch_labels, topk=topk)
```

**Note**: When mixup is enabled, accuracy is computed against the ORIGINAL integer labels, not the soft targets. This is standard practice. But currently mixup is disabled (alpha=0.0), so this is a non-issue.

---

## 5. Data Normalization Audit

### 5.1 Preprocessing

Config says `normalize: false` because data is MIB format from [preprocess_data.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/preprocess_data.py), which already applies:
1. Spine-centering (subtract spine base from first frame)
2. Front-rotation (align shoulders to X-axis)
3. Mean-torso scaling (normalize by average torso length)

→ Joint positions are ~[-3, 3], velocity ~[-2, 2], bone ~[-2, 2].

**Verdict**: ✅ Correct. No double-normalization issue.

### 5.2 Per-Stream BN in Stem

Each stream stem has `BN(3)` → normalizes raw (x,y,z) channels independently. This handles any remaining scale differences between streams.

**Verdict**: ✅ Correct.

### 5.3 Input Clamping

Training: `clamp(-30, 30)`. Validation: `clamp(-10, 10)`.

**Problem**: Val clamp at ±10 is tighter than training ±30. This means the model sees values up to ±30 during training but never above ±10 during validation. If there are legitimate val samples with values between 10-30, they're silently clipped, potentially losing discriminative info.

However, with spine-centered, torso-normalized data, values beyond ±10 are extremely rare (would require a body part to be 10× the torso length). The asymmetric clamping is a minor inconsistency but unlikely to affect accuracy.

---

## 6. Module Impact & Redundancy Analysis

### Estimated Parameter Count by Module

| Component | Per-Block | Total (all 18 blocks) | % of 1.12M |
|-----------|-----------|----------------------|------------|
| GCN convs (K=3 W_k) | 3×c_in×c_out | ~350K | 31% |
| DS-TCN (DW+PW ×2) | ~(C×9 + C×C/2)×2 | ~280K | 25% |
| A_k_learned (K=3) | 3×25×25 = 1875 | 33.8K | 3% |
| STC-Attention | ~C²/4 + C + 25+1 | ~30K | 2.7% |
| JointEmbedding | C×25 | ~42K | 3.7% |
| BRASP | **0** | **0** | 0% |
| SGPShift | **0** | **0** | 0% |
| TLA (last block only) | 4×272×34 + 15 = ~37K | 37K | 3.3% |
| StreamFusion stems ×3 | 3×(BN+3×64 conv+BN) | ~2K | 0.2% |
| Fusion: none (direct concat) | 0 | 0 | 0% |
| Classifier | 272×60 + 60 | ~16K | 1.4% |
| BN layers | ~2×C per layer | ~30K | 2.7% |

### Module-by-Module Impact Assessment

| Module | Expected Impact | Risk |
|--------|----------------|------|
| **BRASP** | +0.1-0.3% (spatial diversity) | None — 0 params, pure routing |
| **SGPShift** | +0.1-0.3% (semantic clustering) | None — 0 params, pure routing |
| **STC-Attention** | +0.5-1.0% (learnable spatial/temporal/channel focus) | Over-regularized by gate init + drop_path |
| **A_k_learned** | +0.3-0.5% (topology adaptation) | **Crippled by double normalization** |
| **TLA** | +0.2-0.5% (temporal landmark focus) | Gate too aggressive (0.5 from epoch 1) |
| **JointEmbedding** | +0.1-0.2% (joint identity signal) | Zero-init → slow to activate |
| **DS-TCN** | Equivalent to MultiScaleTCN at 1/4 cost | ✅ Good tradeoff |
| **Gated GAP+GMP** | Marginal over pure GAP | ✅ Cheap learnable blend |

### Redundancy
- **BRASP + SGPShift**: Both are zero-param spatial routing before the GCN. BRASP does region-based shift, SGP does graph-neighbor shift. Some joint channels get shifted twice (different routing). This is intentional diversity, not redundancy. ✅
- **STC-Attention + TLA**: STC is per-block spatial/temporal/channel attention. TLA is final-block temporal landmark attention. Different granularity, no redundancy. ✅
- **3 independent stream backbones**: Each processes J/V/B independently for 4 blocks, then merges. This IS B4's design. No redundancy. ✅

---

## 7. SOTA Comparison — How They Avoid Plateau/Overfit

| Model | Accuracy (NTU-60 xsub) | Params | Key Design Choices |
|-------|--------------------------|--------|---------------------|
| **EfficientGCN-B4** | 92.1% | 1.1M | Minimal regularization (WD only), 70 epochs, STC-Attention, multi-scale TCN, input fusion |
| **CTR-GCN** | 92.4% | 1.5M | Channel topology refinement, step-LR [35,55], 65 epochs, NO drop_path |
| **InfoGCN** | 93.0% | 1.3M | Info-NCE + label smooth 0.1, mutual-info objective, 110 epochs |
| **BlockGCN** | 92.8% | 1.6M | Block-wise attention, progressive depth, cosine LR |
| **HD-GCN** | 93.4% | 1.6M | Hierarchical decomposition, multi-scale |

**Key SOTA patterns**:
1. **Minimal regularization**: B4, CTR-GCN use ONLY WD + dropout. No drop_path, no label smoothing.
2. **Shorter training**: 65-110 epochs is standard. 150 with aggressive regularization is overkill.
3. **Step LR is dominant**: CTR-GCN ([35,55]), MS-G3D ([50,65]) use step decay, not cosine. Step LR gives a sharp accuracy jump at each milestone.
4. **Single normalization**: All SOTAs normalize the adjacency ONCE, then add learnable residuals directly.

---

## 8. Critical Findings Summary

### ❌ CRITICAL — Will Cause Plateau

| # | Issue | Impact | Fix Direction |
|---|-------|--------|---------------|
| **C1** | **GCN double normalization** — A_fixed is pre-normalized by [normalize_symdigraph_full()](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/graph.py#176-206), then re-normalized in `EfficientZeroBlock.forward()` | Signal attenuation compounds across 18 blocks. Gradients shrink exponentially. A_k_learned updates are fighting against the re-normalization. | Remove inline normalization OR pass raw adjacency and normalize once inline |
| **C2** | **Over-regularization stack** — DropPath(0.25) + LabelSmooth(0.1) + Dropout(0.25) + WD(1e-4) is 3 more regularizers than B4 | Model can't learn enough before capacity is exhausted. Plateau below B4's 92.1%. | Match B4: drop_path=0.0, label_smoothing=0.0. Add them back ONE AT A TIME only after matching B4 baseline. |

### ⚠️ MEDIUM — Contributes to Slower Convergence

| # | Issue | Impact |
|---|-------|--------|
| **M1** | TLA gate init 0.0 → sigmoid=0.5. Injects random attention at 50% weight from epoch 1 | Delays convergence by ~5-10 epochs |
| **M2** | 10-epoch warmup at LR=0.005 for 150 epochs is 6.7% of training spent at very low LR | Slow start compounds with other issues |
| **M3** | Batch size conflict (32 in data config vs 64 in training config) | Ambiguous effective batch |
| **M4** | Cosine LR may be suboptimal vs step-LR for GCN models based on SOTA evidence | Step-LR gives sharper accuracy jumps at milestones |

### ℹ️ LOW — Minor/Cosmetic

| # | Issue |
|---|-------|
| **L1** | Val clamp ±10 vs train ±30 — asymmetric but practically harmless |
| **L2** | JointEmbedding zero-init — correct but slow to activate |
| **L3** | Hardswish vs ReLU — minor gradient factor difference |

---

## 9. Path to 92.5% — Concrete Recommendations

### Phase 1: Fix Critical Issues → Match B4 Baseline (~92%)
1. **Fix double normalization**: In `EfficientZeroBlock.__init__`, store raw (unnormalized) partitions as `_A_fixed_{k}`, and let the inline code be the single normalization point. OR skip the inline normalization and use `A_comb = A_fixed + A_learnt` directly (since A_fixed is already normalized).
2. **Strip regularization to B4 level**: `drop_path_rate=0.0`, `label_smoothing=0.0`, keep `dropout=0.25` and `WD=1e-4`.
3. **Fix TLA gate init**: Change from `torch.zeros(1)` to `torch.full((1,), -4.0)`.
4. **Reduce epochs to 70-90**: Match B4's training horizon.
5. **Clarify batch size**: Ensure it's 64.

### Phase 2: Add Back Novelties One-at-a-Time → Push Past 92%
6. Add label smoothing 0.1 — if val acc improves, keep.
7. Add drop_path 0.1 (not 0.25) — run ablation.
8. Try step-LR with milestones [35, 55] as alternative to cosine.
9. Enable spatial_flip TTA at eval time (+0.1-0.3% free).

### Phase 3: Push to 92.5%+
10. Enable Mixup (alpha=0.2) or CutMix (prob=0.3) — one, not both.
11. Consider knowledge distillation from a trained B4 teacher.
12. Increase num_blocks in shared stage: `[3,4]` → `[4,4]` (if param budget allows).

---

## 10. Gradient Leak Check — Weight Decay on No-Decay Params

The trainer correctly identifies no-decay params via:
- [id(param) in _bn_param_ids](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/training/trainer.py#618-704) — catches all BN/LN
- `'A_k_learned' in name` — catches learnable adjacency ✅
- `'anchor_logits' in name` — catches TLA anchors ✅
- `'.gate' in name` — catches STC, TLA, pool gates ✅
- `'joint_embed' in name` — catches JointEmbedding ✅
- `'pool_gate' in name` — catches gated pooling ✅

**Missing**: The `gcn_convs` weights have `bias=False` ✅. The `spatial_fc`, `temporal_conv`, `channel_fc1/fc2` in STC-Attention are NOT in the no-decay list → they receive WD. This is **correct** — these are standard conv/linear weights that should have WD applied.

**Verdict**: ✅ No gradient leakage via incorrect weight decay assignment.

---

## Final Verdict

The model has **two critical issues** that together explain why accuracy plateaus below B4's 92.1%:

1. **Double normalization** shrinks the GCN's effective adjacency at every block, compounding signal attenuation across 18 blocks. This is a mathematical bug — the gradients for `A_k_learned` are fighting against a normalization step that undoes their updates.

2. **Over-regularization** (3 extra regularizers vs B4) suppresses the learning capacity of the novel components, preventing them from contributing their expected accuracy gains.

Fix C1 + C2 first. This should bring the model to ~92% (matching B4). Then incrementally add regularization and tune hyperparameters to push toward 92.5%.
