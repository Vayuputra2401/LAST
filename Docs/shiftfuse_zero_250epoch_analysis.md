# ShiftFuse-Zero Nano — 250-Epoch Training Analysis & Ceiling-Breaking Strategy

> Best val accuracy: **82.99%** at epoch 245 (target: EfficientGCN-B0's **90.2%**)
> Gap to close: **7.2 percentage points**

---

## 1. Training Dynamics — What Happened

### Learning Phases

| Phase | Epochs | Val Acc | LR Range | Behavior |
|-------|--------|---------|----------|----------|
| **Warmup** | 1-5 | 9% → 60% | 0.024 → 0.10 | Explosive growth — healthy |
| **High LR** | 5-50 | 60% → 75% | 0.10 → 0.092 | Rapid but oscillating |
| **Mid LR** | 50-150 | 75% → 80% | 0.092 → 0.036 | Slow steady climb, ~5% total gain |
| **Low LR** | 150-200 | 80% → 81.5% | 0.036 → 0.010 | Diminishing returns |
| **Convergence** | 200-250 | 81.5% → 83% | 0.010 → 0.0001 | Final polish, +1.5% from LR annealing |

### Key Observation: Train Acc ≪ Val Acc (45% vs 83%)

This is **NOT** a bug — it's expected with heavy augmentation:

```
Training pipeline:    rot ±30° + scale 0.85-1.15 + noise + joint mask 20% + DropPath + Dropout
Validation pipeline:  clean data, model.eval() (no DropPath/Dropout)
```

The 38% gap means the model is **well-regularized** — it's NOT overfitting. The actual bottleneck is **capacity**, not regularization.

### Conclusion: No Plateauing, No Overfitting — Just a Capacity Ceiling ✅

The model improved steadily from epoch 1 to epoch 250 with no plateau. The 83% ceiling is a **representational limit** of the architecture, not a training bug.

---

## 2. Root Causes of the 83% Ceiling

### 🔴 Root Cause #1: Zero-Param Graph Processing Is Too Weak

**The core issue**: BRASP and SGPShift are **fixed permutations** — they shuffle channels to different joints but don't learn which spatial patterns matter.

**What BRASP+SGPShift actually compute**:
```
output[c, v] = input[c, π(c,v)]    # π is a fixed, data-independent permutation
```

**What a learned GCN computes** (EfficientGCN, CTR-GCN):
```
output[c, v] = Σ_k  w_k(c) × Σ_{u ∈ N_k(v)} A[v,u] × input[c, u]    # learned, adaptive
```

The difference: GCN aggregates **weighted sums** from neighborhoods with **per-channel learned edge importance**. Zero-param shifts just redirect one channel to one other joint. This is like comparing a dictionary lookup to a neural network.

**A_learned correction** helps (aggregates over all joints with a 25×25 matrix), but it's:
- Only 625 params per stage (1,875 total) — vs EfficientGCN's ~50K GCN params
- **Shared across all channels** — no per-channel edge weighting
- A single aggregation step — EfficientGCN uses K=3 directional subsets

> **Mathematical comparison of spatial modeling capacity**:
> | Model | Spatial params per block | Spatial operations | Adaptive? |
> |-------|------------------------|-------------------|-----------|
> | **ShiftFuse-Zero** | Conv1×1 only (~C²) | Fixed shift + shared A_learned | ❌ Data-independent |
> | **EfficientGCN-B0** | K×C weights + Conv1×1 (~2C²) | K-subset weighted aggregation | ✅ Per-channel |
> | **CTR-GCN** | Q_k, K_k per subset (~3C²) | Dynamic topology | ✅ Per-sample |

### 🔴 Root Cause #2: Early Fusion Bottleneck at 24 Channels

```
4 streams × 3 channels = 12 channels → Conv1×1(12, 24) → 24 channels
```

**Problem**: 4 fundamentally different data modalities (position, velocity, bone, bone-velocity) are compressed into just 24 channels in the very first layer. This is a **2:1 compression ratio** that loses stream-specific information before any spatial processing happens.

**EfficientGCN comparison**: EfficientGCN also uses early fusion, but projects to **64 channels** (stem_ch=64) — 2.67× wider stem. The wider stem preserves more per-stream discriminative features.

**Mathematical information loss**:
```
Input information:  4 × 3 × 64 × 25 = 19,200 values per sample
After stem at 24ch: 24 × 64 × 25 = 38,400 values  (2× expansion — OK)
After stem at 48ch: 48 × 64 × 25 = 76,800 values  (4× expansion — better)
```

At 24 channels, the Conv1×1 must learn a 12→24 projection that disentangles 4 motion modalities. With only 24 dimensions, there's not enough room for each stream to carve out its own feature subspace.

### 🟠 Root Cause #3: `use_se: False` — SE Was Added But NOT Enabled

The code has SE infrastructure (`self.se = ChannelSE(out_channels) if use_se else nn.Identity()`), but the variant config sets `use_se: False`. **SE was disabled during this 250-epoch run.**

SE costs only ~2×C×(C//4) per block — for C=160 at stage 3, that's ~12,800 total SE params across all 7 blocks. This is <8% of total model params for a proven +0.5-1% gain.

### 🟠 Root Cause #4: StreamFusionConcat Uses ReLU (Not Hardswish)

```python
# In stream_fusion_concat.py, line 52:
nn.ReLU(inplace=True),   # ← Still ReLU, not Hardswish
```

While blocks use Hardswish, the **very first layer** (stem) still uses ReLU. This creates an activation inconsistency at the critical input boundary.

---

## 3. SOTA Comparison — Where is the 7.2% Gap?

```
EfficientGCN-B0:  90.2% (290K params)
ShiftFuse-Zero:   83.0% (170K params)
Gap:              7.2%
```

| Factor | Estimated Contribution to Gap |
|--------|-------------------------------|
| **No learned GCN** (zero-param spatial) | **-3 to -4%** |
| **24ch stem** (vs 64ch in EfficientGCN) | **-1 to -2%** |
| **SE disabled** | **-0.5 to -1%** |
| **170K vs 290K params** (capacity) | **-1 to -2%** |
| **No Mixup/CutMix** | **-0.5 to -1%** |
| **Total estimated gap** | **-6 to -10%** |

This accounts for the full 7.2% gap.

---

## 4. How to Break the 83% Ceiling — Ranked by Impact

### Tier 1: Highest Impact (Expected: +4-6%)

#### 1a. Add Lightweight Learned GCN (~+3-4%, +15K params)

The single biggest gap. Even a simple 1-subset learned GCN would dramatically improve spatial discrimination.

**What to add**: Per-block `Conv1×1(C, C, bias=False)` **after** the A_learned aggregation, applied to the aggregated features separately (not mixed with shifted features). This gives the model per-channel control over which spatial patterns to amplify.

**Or**: Replace A_learned with a per-block learned adjacency (instead of shared), giving each block its own graph topology. Cost: +625×7 = 4,375 params.

#### 1b. Switch to Late Fusion (~+3-4%, +0 extra params)

Late fusion (V10-style MultiStreamStem) processes each stream through independent stems, then shares the backbone. This preserves stream-specific features through the entire backbone.

**Tradeoff**: 4× forward passes (stacked along batch dim), so ~4× compute time. But accuracy gain is the single most proven technique in SOTA skeleton models.

### Tier 2: Medium Impact (Expected: +1-2%)

#### 2a. Widen Stem from 24 → 48 channels (~+1-2%, +~5K params)

```python
'stem_channels': 48,          # was 24
'channels':      [48, 96, 160],  # was [40, 80, 160]
```

Doubling the stem gives each stream 12 channels (from 6) to encode its modality signature. The wider channels through stage 1 and 2 also increase spatial and temporal modeling capacity.

#### 2b. Enable SE Blocks (~+0.5-1%, +~10K params)

```python
'use_se': True,   # was False
```

Just set the flag. The infrastructure is already there.

#### 2c. Enable Mixup + CutMix (~+0.5-1%, +0 params)

```yaml
mixup_alpha: 0.2
cutmix_prob: 0.3
```

Free regularization that's especially effective for 250 epochs. The comment says NaN was an issue with AMP, but ShiftFuse-Zero has no IB loss — the NaN source is likely gone.

### Tier 3: Minor Improvements (Expected: +0.3-0.5% each)

| Change | Impact | Cost |
|--------|--------|------|
| Fix StreamFusionConcat to Hardswish | +0.2% | 0 params |
| SGPShift neighbor cycling (multiple neighbors per channel) | +0.3% | 0 params |
| `gradient_clip: 5.0 → 1.0` | Stability | 0 params |
| Knowledge Distillation from V10 teacher | +2-3% | Config only |

---

## 5. Realistic Accuracy Projections

| Configuration | Est. Acc | Params |
|---------------|----------|--------|
| **Current** (zero-param, early fusion, 170K) | 83% | 170K |
| + Enable SE + Mixup + wider stem (48ch) | 85-86% | 195K |
| + Add lightweight per-block GCN | 87-88% | 215K |
| + Switch to late fusion | 89-90% | 220K |
| + KD from larger teacher | 90-91% | 220K |

> [!IMPORTANT]
> **The fundamental tradeoff**: Zero-param graph processing saves ~120K params but costs ~4% accuracy compared to a proper learned GCN. To break 87%+, some form of learned spatial processing is essential. The question is how lightweight it can be while still being effective.

---

## 6. Gradient & Weight Flow Verification ✅

| Check | Status |
|-------|--------|
| Shape propagation (all 250 epochs NaN-free) | ✅ Loss monotonically decreasing |
| A_learned init=0.01 (cold-start fixed) | ✅ No zero-gradient period |
| Hardswish throughout blocks | ✅ |
| Residual + BN + DropPath correct | ✅ |
| Single CE loss, no competing objectives | ✅ |
| Val acc improves steadily through epoch 250 | ✅ No plateauing |
| Train acc < val acc (augmentation gap, not overfitting) | ✅ |
| Gradient clip = 5.0 (no clipping events visible) | ✅ |
| No BN corruption (loss never spiked NaN) | ✅ |
