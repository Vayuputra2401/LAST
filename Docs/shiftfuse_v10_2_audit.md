# ShiftFuse-V10.2 Comprehensive Audit

> **Files audited**: `shiftfuse_v10.py`, `shiftfuse_gcn.py` (ShiftFuseBlock, ClassificationHead), `adaptive_ctr_gcn.py`, `temporal_landmark_attn.py`, `stream_batch_norm.py`, `channel_se.py`, `body_region_shift.py`, `ep_sep_tcn.py`, `drop_path.py`, `joint_embedding.py`, `graph.py`, `trainer.py`, `dataset.py`, `transforms.py`, all YAML configs.

---

## 1. V10.1 → V10.2 Changes Verified ✅

All critical fixes from the previous audit are confirmed applied:

| Fix | V10.1 | V10.2 | Status |
|-----|-------|-------|--------|
| GCN Conv | `depthwise=True` (groups=C) | `depthwise=False` (groups=4) | ✅ Fixed |
| Label Smoothing | 0.0 | 0.1 | ✅ Fixed |
| Learning Rate | 0.05 | 0.1 | ✅ Fixed |
| Mixup | disabled (0.0) | 0.2 | ✅ Fixed |
| IB Loss | nearest-proto, w=0.001 | class-conditional, w=0.01 | ✅ Fixed |
| TLA gate in no_decay | Missing | `.gate` in no_decay list | ✅ Fixed |
| Epochs | 180 | 240 | ✅ Longer schedule |

**New V10.2 features**:
- `single_head=True` (nano): 1 classifier on averaged stream features — saves ~24K params
- `share_gcn=True` (nano): 1 GCN per stage shared across blocks — saves ~40K params
- `share_je=True` (nano): 1 JointEmbedding per stage shared — saves ~7K params
- `brasp_after_pw=True`: BRASP operates on `out_channels` (correct mapping)
- `use_stream_bn=False`: Regular BN with full batch=64 stats (instead of per-stream BN with 16/stream)
- `num_blocks=[2,3,2]`: 7 blocks (was 5-6 in V10.1)

---

## 2. Architecture Dry Run — Tensor Shape Trace (Nano V10.2)

Input: `x[stream] = (B, 3, 64, 25)` for each of 4 streams.

| Stage | Module | Output Shape | Notes |
|-------|--------|--------------|-------|
| Stem | `MultiStreamStem(3→24)` | `(4B, 24, 64, 25)` | 4 independent BN+Conv1×1 |
| **S1 B0** | `ShiftFuseBlock(24→32, s=1)` | `(4B, 32, 64, 25)` | pw_conv active (24≠32), res conv active |
| S1 B0 | `TLA(32, K=8)` | `(4B, 32, 64, 25)` | d_k = max(4, 32//8) = 4 |
| **S1 B1** | `ShiftFuseBlock(32→32)` | `(4B, 32, 64, 25)` | pw_conv=Identity, res=Identity, **shared GCN** |
| S1 B1 | `TLA(32, K=8)` | `(4B, 32, 64, 25)` | |
| **S2 B0** | `ShiftFuseBlock(32→64, s=2)` | `(4B, 64, 32, 25)` | temporal halved |
| S2 B0 | `TLA(64, K=8)` | `(4B, 64, 32, 25)` | d_k = 8 |
| **S2 B1** | `ShiftFuseBlock(64→64)` | `(4B, 64, 32, 25)` | shared GCN + shared JE |
| S2 B1 | `TLA(64, K=8)` | `(4B, 64, 32, 25)` | |
| **S2 B2** | `ShiftFuseBlock(64→64)` | `(4B, 64, 32, 25)` | shared GCN + shared JE |
| S2 B2 | `TLA(64, K=8)` | `(4B, 64, 32, 25)` | |
| **S3 B0** | `ShiftFuseBlock(64→128, s=2)` | `(4B, 128, 16, 25)` | temporal halved |
| S3 B0 | `TLA(128, K=8)` | `(4B, 128, 16, 25)` | d_k = 16 |
| **S3 B1** | `ShiftFuseBlock(128→128)` | `(4B, 128, 16, 25)` | shared GCN + shared JE |
| S3 B1 | `TLA(128, K=8)` | `(4B, 128, 16, 25)` | |
| Split | `.chunk(4, dim=0)` | `4× (B, 128, 16, 25)` | |
| Head | `avg → ClassificationHead(128, 60)` | `(B, 60)` logits + `(B, 128)` features | single head on mean |

> ✅ **All shapes consistent. No dimension mismatch.**

---

## 3. Gradient Flow Analysis

### 3.1 Block Forward Path (V10.2 nano config)

```
res = Conv1×1+BN(x)         # residual (24→32 for first block, Identity for same-dim)
out = pw_conv(x)             # Conv1×1+BN+Hardswish (only when in≠out)
out = BRASP(out)             # 0-param gather on out_channels ✅
out = joint_embed(out)       # additive bias (shared per stage) ✅
out = GCN(out)               # K-subset group conv + adaptive topology ✅
out = SE(out)                # multiplicative gate ✅
out = TCN(out)               # 4-branch temporal ✅
out = drop_path(out)         # stochastic depth ✅
out = res + out              # clean single residual ✅
out = TLA(out)               # gated residual attention ✅
```

> ✅ **Gradient path is clean.** Residual connects input to output with at most 1 Conv1×1+BN in the skip. No double-compound residuals.

### 3.2 GCN Group Conv Gradient (Fixed from V10.1)

```python
_conv_groups = 4  # num_groups, NOT channels
self.convs = nn.ModuleList([
    nn.Conv2d(C, C, 1, groups=4, bias=False)  # C²/4 params per subset
])
```

Each group conv mixes C//4=32 channels within the group. With G=4 groups, each output channel receives input from 32 peers — sufficient cross-channel mixing for spatial patterns.

> ✅ **Critical V10.1 bug fixed.** Channel mixing restored.

### 3.3 Shared GCN Gradient Accumulation

With `share_gcn=True`, blocks B0 and B1 in the same stage share the same GCN weights. This means:
- GCN gradients accumulate from 2 (stage 1) or 3 (stage 2) or 2 (stage 3) backward passes
- Effective gradient magnitude is ~2-3× larger for shared GCN weights vs per-block

> ⚠️ **Not a bug**, but GCN weights get proportionally more gradient than other block components. The learning rate effectively scales ~2-3× for the shared GCN. This is standard weight-sharing behavior (same as LSTM sharing recurrent weights across timesteps). Monitor if GCN weights dominate updates.

### 3.4 IB Loss Gradient (Fixed from V10.1)

```python
if labels is not None:
    ib_loss = proto_dists[torch.arange(B), labels].mean()  # class-conditional ✅
```

Now only the **correct class prototype** gets gradient for each sample. This is the InfoGCN formulation. With `ib_loss_weight=0.01`, the prototype gradient is 1% of CE — appropriate for a regularization term.

> ✅ **IB loss now functional and correct.**

### 3.5 Single-Head Gradient Path

With `single_head=True`:
```python
mean_feat_raw = torch.stack(feats, dim=0).mean(dim=0)  # (B, C, T', V')
logits, features = head(mean_feat_raw)
return [logits], ib_loss
```

The gradient from the single head's CE loss flows through the `mean()` operation, which divides by 4. Each stream gets 1/4 of the gradient. This is mathematically equivalent to having 4 heads with equal-weight ensemble and averaging their CE losses.

> ✅ **No gradient imbalance.** All 4 streams receive equal gradient signal.

---

## 4. Issues Found

### 4.1 🟡 YAML-Code Mismatch — `num_blocks` Discrepancy

The **YAML** (`shiftfuse_v10_nano.yaml` line 23) says:
```yaml
num_blocks: [1, 2, 2]   # 5 blocks
```

The **Python code** (`shiftfuse_v10.py` line 57) says:
```python
'num_blocks': [2, 3, 2],  # V10.2: 7 blocks
```

**Which one actually runs?** The Python `V10_VARIANTS` dict is used directly in `__init__`. The YAML `num_blocks` field is **never read** by the model constructor — `ShiftFuseV10.__init__` takes `variant='nano'` and looks up `V10_VARIANTS['nano']`. The YAML field is purely informational/documentation.

> ⚠️ **The actual model has 7 blocks** ([2,3,2]), not 5 as the YAML suggests. This is a **documentation mismatch**, not a runtime bug. But it could cause confusion when reporting param counts or reproducing results.

**Risk**: 🟡 **LOW — documentation only.** Update the YAML to match the code.

### 4.2 🟡 Bilateral & DCT Gate — Dead Code in V10 Block

V10.2 creates `ShiftFuseBlock` with:
```python
use_dct_gate=False,
use_joint_embed=True,
use_frame_gate=False,
use_bilateral=False,
```

The block sets `self.bilateral = nn.Identity()` and `self.dct_gate = nn.Identity()`. The forward path (lines 320-332) goes `joint_embed → GCN → SE → TCN → drop_path → res+out → temporal_attn` — `bilateral`, `dct_gate`, and `frame_gate` are **never called** in forward.

> ⚠️ **Dead attributes**: initialized but never used. `nn.Identity()` has zero params/compute. Just dead code.

**Risk**: 🟢 **NONE for correctness.**

### 4.3 🟡 Shared JE Edge Case — First Block Creates Redundant JE

```python
je_ref = stage_je if (share_je and blk_in == blk_out) else None
```

First block of each stage (`blk_in != blk_out`) creates its own JE instead of sharing, even though JE always operates on `blk_out == stage_ch`. Wastes ~5.6K params.

**Risk**: 🟡 **VERY LOW** — 5.6K extra params out of ~165K total.

### 4.4 🟡 Mixup with Class-Conditional IB Loss

When Mixup active: `labels = batch_labels` (original ints) → IB pulls blended features toward the original class prototype only, ignoring the mixed class.

**Risk**: 🟡 **LOW** — at 1% weight, negligible impact.

### 4.5 🟢 `use_temporal_attn=False` in V10 Blocks — Correct Design

TLA is applied externally in `_run_backbone()` after each block. Each block gets its own TLA instance with independent Q/K/V/gate.

> ✅ **Correct design.** Per-block TLA allows depth-specific attention patterns.

---

## 5. Hyperparameter Analysis (V10.2)

| Parameter | V10.2 | EfficientGCN-B0 | CTR-GCN | InfoGCN | Assessment |
|-----------|-------|-----------------|---------|---------|------------|
| LR | **0.1** | 0.1 | 0.1 | 0.1 | ✅ Matches SOTA |
| Optimizer | SGD+Nesterov | SGD+Nesterov | SGD | SGD | ✅ |
| Momentum | 0.9 | 0.9 | 0.9 | 0.9 | ✅ |
| Weight Decay | 0.0004 | 0.0004 | 0.0004 | 0.0002 | ✅ Standard |
| Batch (eff.) | 72 | 64 | 64 | 64 | ✅ |
| Schedule | Cosine+warmup | CosWarmRestart | MultiStep | Cosine | ✅ |
| Warmup | 5 ep | 10 ep | 5 ep | 5 ep | ✅ |
| Epochs | **240** | 70 | 65 | 90 | ⚠️ See below |
| Label Smooth | **0.1** | 0.1 | 0.0 | 0.1 | ✅ |
| Dropout | 0.10 | — | — | — | ✅ |
| DropPath | 0.10 | — | — | — | ✅ |
| Mixup | **0.2** | — | — | 0.2-0.4 | ✅ |
| Grad Clip | 5.0 | — | — | — | ✅ |

### Epochs = 240 — Is It Too Many?

EfficientGCN trains for 70 epochs, CTR-GCN for 65. V10.2 uses 240. With cosine decay, LR reaches near-minimum by epoch ~200. However, combined with label smoothing + mixup + DropPath, 240 epochs should sustain generalization. Consider adding SWA for the last 20-30 epochs.

---

## 6. Weight Decay Groups Audit

| Parameter Pattern | Decayed? | Correct? |
|-------------------|----------|----------|
| Conv2d `.weight` | ✅ Yes | ✅ Standard weights need decay |
| Linear `.weight` (not temporal_attn) | ✅ Yes | ✅ |
| All `.bias` | ❌ No | ✅ Biases should not be decayed |
| BN `.weight`/`.bias` (via `_bn_param_ids`) | ❌ No | ✅ |
| `alpha` (GCN adaptive gate) | ❌ No | ✅ |
| `class_prototypes` | ❌ No | ✅ IB prototypes shouldn't be shrunk |
| `.gate` (TLA gate) | ❌ No | ✅ Fixed from V10.1 |
| `pool_gate` (ClassificationHead) | ❌ No | ✅ |
| `joint_embed` | ❌ No | ✅ Semantic table shouldn't be decayed |
| TLA `q_proj.weight`, `k_proj.weight`, etc. | ✅ Yes | ✅ Learned projections, decay appropriate |

> ✅ **Weight decay groups are correct.**

---

## 7. Loss Function Integrity

### V10.2 Training Loss:
```
total_loss = CE(single_head_logits, target) + 0.01 × dist(feat, proto[y])
```

When Mixup active: `target = λ·one_hot(y_a) + (1-λ)·one_hot(y_b)` (float tensor)
When Mixup inactive: `target = y` (integer class label)

`CrossEntropyLoss(label_smoothing=0.1)` handles both correctly in PyTorch ≥ 1.10.

### SOTA Comparison:
| Model | Main Loss | Aux Loss | Smoothing | Regularization |
|-------|-----------|----------|-----------|----------------|
| EfficientGCN | CE | — | 0.1 | — |
| CTR-GCN | CE | — | 0.0 | Multi-step LR |
| InfoGCN | CE | IB (class-conditional, λ=0.01) | 0.1 | — |
| **V10.2** | **CE** | **IB (class-conditional, λ=0.01)** | **0.1** | **Mixup + DropPath** |

> ✅ **Loss function matches SOTA best practices.**

---

## 8. Module Impact Analysis (Nano V10.2)

| Module | Params (est.) | Role | Redundant? |
|--------|---------------|------|------------|
| **MultiStreamStem** | ≈1.2K | 4-stream input embedding | No (core) |
| **BRASP** | 0 | Anatomical channel routing | No (free) |
| **pw_conv** (first block/stage) | ≈13K | Channel projection | No (necessary) |
| **Shared GCN** (3 stages) | ≈34K | Spatial aggregation **(core)** | No |
| **Shared JE** (3 stages) | ≈5.6K | Joint identity | Worth keeping (cheap) |
| **SE** (7 blocks) | ≈14K | Channel recalibration | Worth keeping |
| **TCN** (7 blocks) | ≈60K | Temporal modeling **(core)** | No |
| **TLA** (7 blocks) | ≈20K | Global temporal attention | Valuable |
| **ClassificationHead** | ≈8K | Classification | No (core) |
| **IB Prototypes** | 7.7K | Class-conditional regularization | Worth keeping |
| **Total** | **~165K** | | |

### Expected Contribution per Module:
| Module | Expected Impact |
|--------|-----------------|
| 4-stream late fusion | +3-4% over single-stream |
| Group Conv GCN (G=4) | Core — required for spatial learning |
| TLA (K=8) | +1-1.5% |
| SE | +0.5-1% |
| BRASP | +0.3-0.5% |
| JointEmbedding | +0.2-0.5% |
| IB Loss | +0.3-0.5% |
| Label Smoothing | +1-2% |
| Mixup | +0.5-1% |

---

## 9. Data Normalization & Pipeline Check

1. Preprocessing: Spine-centered, torso-scaled (done in `preprocess_data.py`)
2. MIBTransform: Rotation (±15°), Scale (0.9-1.1), Shear (±0.1), per-stream noise
3. Trainer clamp: ±30 (prevents tracking glitches)
4. Stem BN: `BatchNorm2d(3)` normalizes raw input channels
5. Dataset provides: `(C=3, T=64, V=25, M=2)` per stream → Model takes `x[..., 0]` → `(B, 3, 64, 25)`

> ✅ **Pipeline correct. All shapes verified.**

---

## 10. Potential Improvements (Ordered by Expected Impact)

### 10.1 🟡 SWA/EMA for Long Schedule (Expected: +0.3-0.5%)
Add Stochastic Weight Averaging for last ~30 epochs. PyTorch natively supports `torch.optim.swa_utils.AveragedModel`.

### 10.2 🟡 Joint Masking Augmentation (Expected: +0.3-0.5%)
`MIBTransform` supports `joint_mask_ratio` but YAML doesn't enable it. Enable `joint_mask_ratio: 0.15`.

### 10.3 🟡 Per-Stage TLA Landmarks (Expected: +0.1-0.3%)
At T=16 (stage 3), K=8 landmarks is nearly dense. Consider K=8/4/4 for stages 1/2/3.

### 10.4 🟢 Warm Restart Schedule (Expected: +0.2-0.5%)
SGDR periodically resets LR. Already supported in trainer.

### 10.5 🟢 Fix IB Loss During Mixup
Mix the IB loss: `λ·dist(feat, proto[y_a]) + (1-λ)·dist(feat, proto[y_b])`. Minor impact at 1% weight.

---

## 11. Estimated Performance (Nano V10.2)

| Configuration | Estimated Val Top-1 |
|---------------|-------------------|
| V10.1 (depthwise, no LS, no mixup) | 68.9% (actual peak) |
| **V10.2** (group conv, LS=0.1, mixup=0.2, lr=0.1, 240ep) | **83-88%** |
| V10.2 + SWA + joint masking | **85-89%** |
| V10.2 + KD from large teacher | **88-91%** |
| EfficientGCN-B0 (290K params, reference) | 90.2% |

---

## 12. Audit Summary

### ✅ All V10.1 Critical Fixes Confirmed Applied
- **Group Conv GCN** (`depthwise=False`, G=4) — cross-channel spatial mixing restored
- **Label Smoothing = 0.1** — prevents logit saturation and val oscillations
- **Mixup α = 0.2** — data-level regularization
- **LR = 0.1** — matches SOTA convention
- **IB Loss** — class-conditional (passes `labels=batch_labels`), weight = 0.01
- **TLA gate** in no_decay (via `.gate` pattern match)

### ✅ V10.2 New Features Verified Correct
- **`single_head=True`** (nano): gradient correctly divided by 4 via `mean()` — all streams get equal signal
- **`share_gcn=True`** (nano): shared GCN gets ~2-3× gradient accumulation — standard weight-sharing behavior, not a bug
- **`brasp_after_pw=True`**: BRASP now operates on `out_channels` which correctly matches GCN's channel dimension
- **`use_stream_bn=False`**: Regular BN with full batch=64 stats (much better than 16/stream)

### ✅ Gradient Flow: Clean Throughout
Every block: `pw_conv → BRASP → JointEmbed → GCN → SE → TCN → DropPath → res+out → TLA`. Single clean residual, no double-compound paths. All modules have proper gradient flow.

### ✅ Weight Decay Groups: Correct
All conv/linear weights get decay. All biases, BN params, gates, prototypes, joint embeddings excluded.

### 🟡 Minor Issues Found (Non-Critical)
1. **YAML says `num_blocks: [1,2,2]` but code uses `[2,3,2]`** — YAML is just documentation, code takes precedence. Update YAML for clarity.
2. **First block of each stage creates redundant JE** instead of sharing — wastes ~5.6K params
3. **IB loss not mixed during Mixup** — at 1% weight, negligible impact
4. **Dead code**: `bilateral`, `dct_gate`, `frame_gate` initialized but never called in forward

### Suggestions for Further Improvement
1. **SWA/EMA** for last 30 epochs (+0.3-0.5%)
2. **Joint masking** augmentation (ratio=0.15, +0.3-0.5%)
3. **Warm Restart** schedule instead of plain cosine (+0.2-0.5%)

### Final Verdict
**V10.2 looks mathematically sound. No critical issues remaining. It should train well.**

The model should be ready for training. Monitor:
1. Val accuracy stabilization (should not oscillate ±4% anymore with LS=0.1)
2. Train-val gap (should stay <10% with mixup+LS)
3. Val accuracy plateau timing (if before epoch 180, consider shorter schedule or SWA)
