# ShiftFuse-Zero Nano & Small — Comprehensive Audit

**Nano**: `nano_tiny_efficient` — single backbone, 4-stream early fusion, ~97K params, TLA enabled
**Small**: `small_late_efficient` — 2-backbone late fusion (J+V / B+BV), ~240K params, CrossStreamFusion

**Target**: 92.5% Top-1 on NTU-60 xsub

---

## 1. Architecture Dry Run — Shape Traces

### 1.1 Nano (`nano_tiny_efficient`)

Config: `stem=24, channels=[32,64,128], blocks=[1,1,1], strides=[1,2,2], 4-stream early fusion`

Dropout override: model YAML says `dropout: 0.0` (overrides variant default 0.10).

| Layer | Out Shape | Notes |
|-------|-----------|-------|
| Input: 4 streams | [(B, 3, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) ×4 | joint, velocity, bone, bone_velocity |
| Multi-body handling | [(2B, 3, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) ×4 | cat M=2 bodies along batch |
| [StreamFusionConcat(3→24, 4-stream)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/stream_fusion_concat.py#26-69) | [(2B, 24, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | 4×BN(3)→concat(12ch)→Conv1×1(12→24)→BN→HSwish |
| Stage 1: [EfficientZeroBlock(24→32, ×1, stride=1)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 32, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | c_in=24→c_out=32 (residual: Conv1×1 + BN) |
| Stage 2: [EfficientZeroBlock(32→64, ×1, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 64, 32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | T halved. residual: Conv+BN with stride |
| Stage 3: [EfficientZeroBlock(64→128, ×1, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) + **TLA** | [(2B, 128, 16, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | T halved. TLA(K=12, d_k=8) at this block |
| `x.mean(dim=2)` | [(2B, 128, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | Temporal GAP |
| Gated GAP+GMP over joints | [(2B, 128)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | `g·mean(V) + (1-g)·max(V)` |
| `Dropout(0.0)→Linear(128→60)` | [(2B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | **⚠️ 0% dropout — no regularization at head** |
| Multi-body average | [(B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | `(out[:B]+out[B:])/2` |

**Total blocks**: 3. Drop-path schedule: 0.0, 0.05, 0.10.
**Shape Verdict**: ✅ All shapes trace correctly.

#### Nano Parameter Breakdown (~97K)

| Component | Params |
|-----------|--------|
| StreamFusionConcat: BN(3)×4 + Conv(12→24) + BN(24) | ~384 |
| Block 0 (24→32): BRASP(0) + SGP(0) + JE(24×25=600) + STC(24,r=4) + 3×Conv(24→32) + BN(32) + DS-TCN(32) + residual(24→32) | ~8.5K |
| Block 1 (32→64): same pattern | ~18K |
| Block 2 (64→128): same + TLA(128, K=12, d_k=8) | ~42K + 4.1K TLA |
| A_k_learned: 3 blocks × 3 × 25×25 = 5625 | 5.6K |
| Classifier: Linear(128→60) + bias | 7.7K |
| Pool gate | 1 |
| **Total** | **~97K** |

### 1.2 Small Late (`small_late_efficient`)

Config: `stem=24, channels=[32,64,128], blocks=[1,2,1], strides=[1,2,2], TLA=False`
Two independent backbones, each with 2-stream early fusion. CrossStreamFusion enabled.

Dropout: model YAML says `dropout: 0.10` (matches variant default).

**Per backbone** (Backbone A: J+V, Backbone B: B+BV):

| Layer | Out Shape | Notes |
|-------|-----------|-------|
| Input: 2 streams | [(2B, 3, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) ×2 | e.g. joint + velocity |
| [StreamFusionConcat(3→24, 2-stream)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/stream_fusion_concat.py#26-69) | [(2B, 24, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | 2×BN(3)→concat(6ch)→Conv1×1(6→24)→BN→HSwish |
| Stage 1: [EfficientZeroBlock(24→32, ×1, stride=1)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 32, 64, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | |
| Stage 2: [EfficientZeroBlock(32→64, ×2, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 64, 32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | Block 0: 32→64,stride=2. Block 1: 64→64,stride=1 |
| Stage 3: [EfficientZeroBlock(64→128, ×1, stride=2)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) | [(2B, 128, 16, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | No TLA (use_tla=False) |
| **CrossStreamFusion** | [(2B, 128, 16, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | gate(init -4.0) × bottleneck exchange |
| Gated GAP+GMP | [(2B, 128)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | |
| `Dropout(0.10)→Linear(128→60)` | [(2B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | |
| Multi-body average | [(B, 60)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) | |

**Final output**: [(logits_A + logits_B) / 2](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464)

**Total blocks per backbone**: 4. Two backbones = 8 blocks total.
Drop-path schedule per backbone: {0.0, 0.033, 0.067, 0.10} (blocks 0-3 out of total=4).

**Per-backbone params**: ~110K. Two backbones = ~220K. CrossStreamFusion(128): ~37K.
**Total ~257K** (documented as ~240K — close enough given approximation).

**Shape Verdict**: ✅ All shapes trace correctly.

---

## 2. Gradient Flow Analysis — Mathematical Dry Run

### 2.1 GCN Double Normalization (SAME AS B4 — CRITICAL)

> [!CAUTION]
> **CRITICAL: Both nano and small share the same [EfficientZeroBlock](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/shiftfuse_zero.py#92-237) which double-normalizes the adjacency.**

[normalize_symdigraph_full()](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/graph.py#176-206) produces D^{-1/2}·A·D^{-1/2} → stored as `_A_fixed_{k}`.
Then [forward()](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/models/blocks/drop_path.py#36-44) re-normalizes with another D^{-1/2} step (lines 217-219).

**Impact on nano (3 blocks)**: Signal attenuation compounds over 3 blocks. Less severe than B4's 18 blocks, but still measurable. With channels only 24→32→64→128, every block matters.

**Impact on small (4 blocks per backbone)**: 4 blocks of compound attenuation × 2 backbones. The backbones are independent so attenuation doesn't cross-compound, but each backbone's features arrive at the CrossStreamFusion already attenuated.

**A_k_learned impact**: Zero-initialized. The inline normalization re-normalizes `A_fixed + A_learned` each forward pass. As A_learned deviates from zero, the normalization partially undoes the update by redistributing row sums. This makes A_learned updates **less effective per gradient step** than they should be.

### 2.2 TLA Gate Init = 0.0 → sigmoid(0) = 0.5 (NANO ONLY)

Nano has `use_tla=True`. TLA is attached to the LAST block (stage 3, block 0).

TLA gate: `nn.Parameter(torch.zeros(1))` → sigmoid(0) = 0.5.

At epoch 1, Q/K/V projections are random noise. TLA output is essentially random attention-weighted features scaled by 0.5 and added to the backbone output. This adds **50% noise to the final feature map** that feeds the classifier.

For a 97K-param model with only 3 blocks, this is proportionally MORE damaging than for B4 (18 blocks). The last block IS the classifier's only source of features — there's no filtering after TLA.

**Small has `use_tla=False`** → not affected. ✅

### 2.3 Nano: Dropout = 0.0 in Classifier Head

Model YAML: `dropout: 0.0`. This overrides the variant's default 0.10.

In `ShiftFuseZero.__init__()`:
```python
_dropout = dropout if dropout is not None else cfg['dropout']
# ...
self.classifier = nn.Sequential(
    nn.Dropout(p=_dropout),   # p=0.0 → no-op
    nn.Linear(final_ch, num_classes),
)
```

**At 97K params**, the model needs SOME regularization at the head. The only regularization is:
- DropPath: max 0.10 (only last block gets 0.10)
- WD: 1e-4
- Mixup: α=0.1

No label smoothing, no head dropout, only mild drop_path. This is **under-regularized** for a model that will be trained for 300 epochs.

> [!WARNING]
> **Nano has ZERO head dropout.** At 97K params, this is fine for the first ~100 epochs. But by epoch 200-300, the model will overfit unless the augmentation + mixup is sufficient. Monitor train/val gap closely.

### 2.4 Small: CrossStreamFusion Gate = -4.0 at Init

CrossStreamFusion gate: `nn.Parameter(torch.tensor(-4.0))` → sigmoid(-4) = 0.018.

This starts at 1.8% cross-stream contribution. The gate must learn to grow during training. At 300 epochs, this is fine — it has ~50+ epochs to warm up.

**But**: The `.gate` parameter is in the no-decay list (matched by `'.gate' in name`). This means it has WD=0, which is correct — WD would fight the gate's growth.

**Verdict**: ✅ Gate init is appropriate for CrossStreamFusion.

### 2.5 Logit Averaging in ShiftFuseZeroLate

```python
return (logits_a + logits_b) / 2
```

**Problem**: This averages raw logits, not probabilities. If backbone A is confident (logits concentrated) and backbone B is uncertain (logits near-uniform), the average dilutes A's confidence. Better: average softmax probabilities or use a learnable blend.

**Impact**: LOW. Logit averaging is standard in multi-branch GCN models (EfficientGCN, CTR-GCN ensembles). The streams have similar capacity so logit magnitudes should be comparable.

### 2.6 Residual Path at Stage Boundaries

At each stage boundary (e.g., 24→32, 32→64, 64→128):
```python
self.residual = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
    nn.BatchNorm2d(out_channels),
)
```

The residual has **no activation** — it's a linear projection. The main path has Hardswish.
```python
x = self.out_act(x + self.residual(res))  # Hardswish(main + residual)
```

Hardswish applied to the SUM. If `self.residual(res)` dominates (which it does at init, since the main path passes through double-normalized GCN → attenuated signal), the sum is approximately `self.residual(res)`, and Hardswish acts on the linear residual. This is fine — gradient flows through the residual path cleanly via Conv+BN.

**Verdict**: ✅ Residual gradient flow is correct.

### 2.7 BRASP + SGP Channel Mismatch at Stage Boundaries

BRASP and SGP are initialized with `channels=in_channels`.

At block 0 of stage 2 (nano): `in_channels=32, out_channels=64`.
BRASP(32) and SGP(32) operate on the INPUT channels (32), then the GCN projects 32→64. This is correct.

But the shift indices are precomputed at init for `C=in_channels`. If the block's forward receives `x` with shape [(B, 32, T, V)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464), BRASP index buffer is [(32, 25)](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py#167-464) → shape matches. ✅

**Verdict**: ✅ No channel mismatch.

---

## 3. Training Configuration Audit

### 3.1 Both Nano and Small Use the SAME Training Config

[train.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py) line 214-217:
```python
if args.model in ('shiftfuse_zero_large_late_efficient', 'shiftfuse_zero_large_b4_efficient'):
    _training_cfg_name = 'shiftfuse_zero_large_efficient'
else:
    _training_cfg_name = 'shiftfuse_zero_nano_efficient'
```

So **both nano (97K) and small (240K) share identical training hyperparameters**:

| Param | Value | Issue? |
|-------|-------|--------|
| Optimizer | SGD+Nesterov | ✅ |
| LR | 0.1 | ✅ for 97K, potentially high for 240K with 2 backbones |
| Momentum | 0.9 | ✅ |
| WD | 1e-4 | ✅ |
| Scheduler | cosine_warmup, 10ep warmup | ✅ |
| Warmup start LR | 0.005 | ✅ |
| Min LR | 1e-4 | ⚠️ High floor for 300 epochs |
| Epochs | 300 | ✅ for nano, ⚠️ for small (240K converges faster) |
| Batch Size | 64 | ✅ |
| Label Smoothing | 0.0 | ✅ |
| Mixup α | 0.1 | ✅ |
| CutMix | 0.0 | ✅ |
| Gradient Clip | 1.0 | ✅ |

### 3.2 Min LR = 1e-4 is Too High

With cosine annealing over 290 post-warmup epochs (300 - 10), the LR reaches `min_lr = 1e-4` around epoch 290.

**Problem**: 1e-4 is **10× higher** than typical SOTA settings (1e-5 to 0). This means the model never enters the fine-tuning phase where small gradient updates polish decision boundaries. EfficientGCN uses `eta_min=0`. CTR-GCN step-LR decays to `0.1 × 0.1 × 0.1 = 0.001` at epoch 65 and keeps it flat.

For a 97K model that peaks around epoch 280-300, the final 50 epochs are essentially wasted because the LR is too high to fine-tune.

**Severity**: MEDIUM. Reducing `min_lr` from 1e-4 to 1e-5 would likely gain +0.2-0.5% in the last 50 epochs.

### 3.3 Same Epochs for Different Model Sizes

Nano (97K) and small (240K) both train for 300 epochs. However:
- Nano peaking at epoch 288 (per config comments) → 300 epochs is appropriate.
- Small (~240K, 2 backbones) has 2.5× more params → should converge ~30% faster. 200-250 epochs would be sufficient.

Training too long with mixup=0.1 can cause the model to overfit to the interpolated distribution rather than the true data distribution. This manifests as train acc continuing to rise while val acc plateaus.

### 3.4 Batch Size 32 vs 64 Conflict (SAME AS B4)

Data config says `batch_size: 32`, training config says `batch_size: 64`. The [train.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py) uses `config['training']['batch_size']` → **64 is used**. But the config merger might install the data config's 32 depending on merge order.

Looking at [train.py](file:///c:/Users/anubh/OneDrive/Desktop/LAST/scripts/train.py) line 227-228:
```python
config = default_cfg        # training config → batch_size=64
config.update(specific_config)  # data+model+env configs → may overwrite
```

The `specific_config` from `load_config()` contains [data](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/dataset.py#72-82), `model`, `environment` keys. The training config has a `training` key. Since `config.update(specific_config)` is a shallow merge (top-level keys), and `specific_config` contains [data](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/data/dataset.py#72-82) (with `dataloader.batch_size=32`), the `training.batch_size=64` should survive because `training` key is not overwritten by the specific_config.

**BUT**: If `config.get('environment', {}).get('training_overrides', {})` contains a batch_size key, it would override the training config. This depends on the environment config.

**Verdict**: Likely 64, but ambiguous. Should be verified at runtime.

---

## 4. Loss Function Integrity

### 4.1 CrossEntropyLoss without Label Smoothing + Mixup

```python
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
```

With `mixup_alpha=0.1`:
- ~50% of batches: λ ∈ [0.9, 1.0] (Beta(0.1, 0.1) is U-shaped, concentrates at extremes)
- Target: `lam_real * y_a_onehot + (1-lam_real) * y_b_onehot` → float tensor

**CrossEntropyLoss with soft float targets**: PyTorch's `nn.CrossEntropyLoss` supports float targets since 1.10. The loss becomes the cross-entropy between the predicted distribution and the soft target distribution. This works correctly. ✅

**Accuracy measurement**: Measured against ORIGINAL integer labels, not soft targets. This means during mixup epochs, top-1 accuracy is slightly underreported (model predicts the interpolated class, which may match neither label). This is standard practice.

### 4.2 Nano: No Label Smoothing + No Head Dropout

The only regularization at the loss level is Mixup(α=0.1). Since Beta(0.1, 0.1) concentrates at the extremes, most samples are barely mixed (λ ≈ 0.95). This provides mild regularization — weaker than label smoothing.

For a 97K model, this is borderline. The model is small enough that overfitting risk is lower, but at 300 epochs, there's still risk.

---

## 5. Data Pipeline Audit

### 5.1 Input Streams for Each Model

**Nano (4-stream fusion)**:
- `stream_names = ['joint', 'velocity', 'bone', 'bone_velocity']`
- All 4 streams concatenated: 4×BN(3) → concat(12ch) → Conv(12→24) → BN → HSwish

**Small (2-stream per backbone)**:
- Backbone A: `['joint', 'velocity']` → concat(6ch) → Conv(6→24)
- Backbone B: `['bone', 'bone_velocity']` → concat(6ch) → Conv(6→24)

**Data loading**: MIB format loads all 4 streams. Each model only indexes the streams it needs via `self.stream_names`. Extra streams in the dict are ignored. ✅

### 5.2 Augmentation

Shared MIBTransform (geometrically consistent across all streams):
- Rotation: [-30, 30]°
- Scale: [0.85, 1.15]
- Shear: [-0.1, 0.1]
- Noise: per-stream (J=0.01, V=0.002, B=0.005)
- Spatial flip: p=0.5
- Temporal speed: p=1.0, range [0.8, 1.2]
- Joint mask: 20% (5/25 joints)
- Temporal flip: disabled ✅

**Augmentation strength**: Rotation ±30° is very aggressive. Standard is ±15°. For skeleton data, 30° rotation means extreme viewpoint change — fine for xsub (cross-subject has viewpoint variation) but could be destabilizing for small models.

### 5.3 Normalization

Same as B4: MIB data is pre-normalized (spine-centered, front-rotated, torso-scaled). Online `normalize: false`. ✅

### 5.4 Input Clamping

Training: ±30. Validation: ±10.
Same asymmetric clamping as B4. Low severity. ✅

---

## 6. Module Impact Analysis

### 6.1 Nano — Every Module Counts at 97K

| Module | Params | % of 97K | Expected Impact |
|--------|--------|----------|-----------------|
| BRASP (0-param) | 0 | 0% | +0.1-0.3% (spatial diversity) |
| SGPShift (0-param) | 0 | 0% | +0.1-0.3% (semantic routing) |
| JE per-block | ~2.5K | 2.6% | +0.1-0.2% (joint identity signal) |
| STC-Attention | ~1.2K | 1.2% | +0.3-0.5% (attention gating) |
| A_k_learned ×3 blocks | 5.6K | 5.8% | ⚠️ **Crippled by double normalization** |
| DS-TCN | ~40K | 41% | Core temporal learning |
| GCN (K=3 W_k) | ~25K | 26% | Core spatial learning |
| TLA (K=12, d_k=8) | 4.1K | 4.2% | ⚠️ **50% noise injection from epoch 1** |
| Classifier | 7.7K | 7.9% | Final decision |
| Stem + BN | ~2K | 2% | Input projection |

**Critical**: At 97K, A_k_learned (5.8%) and TLA (4.2%) together are 10% of params. If both are crippled (double-norm + wrong gate init), that's 10% dead weight.

### 6.2 Small — CrossStreamFusion Impact

| Component | Params | Expected Impact |
|-----------|--------|-----------------|
| Each backbone | ~110K | Backbone accuracy |
| CrossStreamFusion(128) | ~37K | +0.3-0.5% (inter-backbone info exchange) |
| A_k_learned ×8 blocks | 15K | ⚠️ Double-norm crippled |

CrossStreamFusion at 37K (~15% of total) is a significant module. Its gate at -4.0 means it starts virtually disabled. With cosine LR going to min_lr=1e-4, the gate gradient might not be strong enough to push it open quickly.

### 6.3 Redundancy Check

- **BRASP + SGPShift**: Both at 0 params. Different routing strategies. Not redundant. ✅
- **Nano: 4 streams for 97K params**: The stem compresses 12→24 channels (50% compression). This is aggressive — 4 streams of 3ch each = 12 features compressed to 24. Compare EfficientGCN-B0 (similar size) which also does 4→concat, so this is standard. ✅
- **Small: 2 backbones × 4 blocks with SEPARATE weights**: No weight sharing → 2× params for 2× data views. This is the correct late-fusion design (identical to EfficientGCN 2-branch setup). ✅
- **Small: TLA disabled**: Correct — TLA on top of 2-backbone late fusion + CrossStreamFusion would be over-parameterized for 240K. ✅

---

## 7. SOTA Comparison — Lightweight Models

| Model | Accuracy (NTU-60 xsub) | Params | Key Choices |
|-------|--------------------------|--------|-------------|
| **EfficientGCN-B0** | 90.2% | 86K | Minimal arch, STC-Attn, 70ep, WD only |
| **EfficientGCN-B2** | 91.4% | 340K | 2-branch, stronger backbone, 70ep |
| **ShiftGCN** | 90.7% | 180K | Shift-based spatial mixing + soft edges |
| **Ta-CNN** | 90.7% | 700K | Temporal attention CNN |
| **Nano target** | 92.5% | 97K | Ambitious: B0 is 90.2% at 86K |

> [!IMPORTANT]
> **92.5% at 97K is extremely ambitious.** B0 at 86K achieves 90.2%. B2 at 340K achieves 91.4%. Hitting 92.5% at 97K would be SOTA-breaking for this param range. Realistic ceiling: **90.5-91.0%** (B0 baseline + novelties). The 92.5% target is more appropriate for the Small model at 240K.

---

## 8. Critical Findings Summary

### ❌ CRITICAL — Will Cause Plateau or Suboptimal Convergence

| # | Issue | Affects | Impact |
|---|-------|---------|--------|
| **C1** | **GCN double normalization** — A_fixed pre-normalized, then re-normalized inline | Both | Signal attenuation, A_k_learned updates partially undone each forward pass |
| **C2** | **TLA gate init 0.0 → 50% random noise injection from epoch 1** | Nano | TLA output is random at start but weighted 50%. Directly corrupts final features for the classifier |

### ⚠️ MEDIUM — Contributes to Slower/Suboptimal Convergence

| # | Issue | Affects | Impact |
|---|-------|---------|--------|
| **M1** | **Nano dropout=0.0 at classifier head** — model YAML overrides variant default | Nano | Under-regularized at 300 epochs, risk of overfitting in later epochs |
| **M2** | **min_lr=1e-4 too high** — cosine schedule never enters fine-tuning zone | Both | Last 50-100 epochs wasted. 1e-5 or 0 would allow polishing |
| **M3** | **Same 300 epochs for both** — small (240K) converges faster than nano (97K) | Small | Over-training with mixup → overfits to interpolated distribution |
| **M4** | **Rotation ±30° too aggressive** — standard is ±15° | Both | Small models can't learn robust rotation invariance at 97K-240K |
| **M5** | **joint_mask_ratio=0.20 (5/25 joints)** — aggressive for nano's 3-block depth | Nano | Zeroing 20% of joints forces the 3-block model to hallucinate global context with very limited capacity |

### ℹ️ LOW

| # | Issue | Affects |
|---|-------|---------|
| L1 | Val clamp ±10 vs train ±30 asymmetry | Both |
| L2 | Batch size ambiguity (32 vs 64) | Both |
| L3 | JointEmbedding zero-init — slow to activate | Both |

---

## 9. Path to Improving Metrics

### For Nano (97K → realistic 90.5-91.0%)

1. **Fix double normalization** — remove inline re-normalization OR pass raw adjacency
2. **Fix TLA gate init** — change `torch.zeros(1)` to `torch.full((1,), -4.0)` in TLA
3. **Set dropout=0.10** at head (match variant default, not model YAML override of 0.0)
4. **Lower min_lr** to 1e-5 or 0
5. **Reduce rotation** from ±30° to ±15°
6. **Reduce joint_mask** from 0.20 to 0.10 (3/25 joints) for a 3-block backbone
7. Keep mixup=0.1 and 300 epochs (appropriate for 97K)

### For Small (240K → realistic 91.5-92.0%)

1. **Fix double normalization** — same as nano
2. **Lower min_lr** to 1e-5
3. **Reduce epochs** from 300 to 200-250
4. **Reduce rotation** from ±30° to ±20° (small has more capacity than nano)
5. Consider **enabling TLA** on last shared block (use_tla=True with gate init -4.0)
6. Consider lowering CrossStreamFusion gate init from -4.0 to -2.0 (sigmoid≈0.12) for faster cross-stream learning in early epochs

### For Either Model to Push Toward 92.5%

7. **Knowledge distillation** from a trained B4 teacher → most impactful single change
8. **Checkpoint averaging** of last 5-10 epochs → free +0.1-0.3%
9. **TTA spatial flip** at eval time → free +0.1-0.3%
10. Enable **CutMix** (prob=0.3) instead of or alongside Mixup

---

## 10. Weight Decay / Gradient Leak Check

### Nano (ShiftFuseZero)

Parameters that should be no-decay:
- `A_k_learned` → matched by `'A_k_learned' in name` ✅
- `pool_gate` → matched by `'pool_gate' in name` and `'.gate' in name` ✅
- `stc_attn.gate` → matched by `'.gate' in name` ✅
- `joint_embed` (JE) → matched by `'joint_embed' in name` ✅
- TLA `anchor_logits` → matched by `'anchor_logits' in name` ✅
- TLA `gate` → matched by `'.gate' in name` ✅
- All BN params → matched by [id(param) in _bn_param_ids](file:///c:/Users/anubh/OneDrive/Desktop/LAST/src/training/trainer.py#618-704) ✅
- All bias terms → matched by `'bias' in name` ✅

**Verdict**: ✅ No gradient leakage via incorrect WD.

### Small (ShiftFuseZeroLate)

Additional parameters:
- `cross_fusion.gate` → matched by `'.gate' in name` ✅
- `cross_fusion.proj_*.*.weight` (Conv2d) → correctly in decay group ✅
- Backbone A + B each have their own complete parameter set → no sharing issues ✅

**Verdict**: ✅ No gradient leakage.

---

## 11. DS-TCN Padding Verification

DS-TCN branch 2 (dilation=2):
```python
pad2 = kernel_size - 1  # 9 - 1 = 8
self.dw2 = nn.Conv2d(
    ..., (kernel_size, 1), stride=(stride, 1), padding=(pad2, 0), dilation=(2, 1), ...
)
```

For causal TCN with dilation=2, kernel=9: effective kernel length = 1 + (9-1)×2 = 17.
Same-padding required: (17-1)/2 = 8. `pad2 = 8`. ✅

With stride=2: output temporal length = floor((T + 2×8 - 17) / 2) + 1 = floor((T-1)/2) + 1 = T/2 (for even T). At T=64 → 32. ✅

Branch 1 (dilation=1): effective kernel = 9. Same-padding = 4. `pad1 = (9-1)//2 = 4`. ✅
With stride=2: floor((64 + 8 - 9) / 2) + 1 = 32. ✅

Both branches produce same temporal dimension. `torch.cat` along channel dim works. ✅

---

## Final Verdict

**Nano (97K)**: Has 3 issues that together explain underperformance — double normalization (C1), TLA noise injection (C2), and zero head dropout (M1). The double normalization is the same architectural bug as B4 but less severe with only 3 blocks. The TLA issue is MORE severe in nano because TLA is the final transform before the classifier, and with only 3 blocks there's no downstream filtering.

**Small (240K)**: Cleaner design — TLA disabled, dropout=0.10. Main issues are double normalization (C1) and suboptimal training schedule (M2 min_lr, M3 epochs). CrossStreamFusion is well-designed with appropriate gate init.

**92.5% is unrealistic for nano (97K)** — SOTA at this param range is ~90.2% (B0). Realistic target: 90.5-91.0% after fixes.

**92.5% is ambitious but possible for small (240K)** — sits between B0 (86K→90.2%) and B2 (340K→91.4%). With novelties + KD, 91.5-92.0% is realistic. 92.5% would require KD from a strong teacher.
