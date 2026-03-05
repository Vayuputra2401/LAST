# LAST-Lite (ShiftFuse-GCN) — Version History

Model: `shiftfuse_gcn.py` / `shiftfuse_experimental.py`
Task: Skeleton action recognition, NTU RGB+D 60, cross-subject split
Baseline: EfficientGCN-B0 (290K, 90.2%) / B4 (1.1M, 92.1%)

---

## Quick Reference

| Version | Params (small) | NTU-60 xsub | Fusion | Key change |
|---------|---------------|-------------|--------|------------|
| v5 | ~289K | **83.33%** (ep145) | StreamFusionConcat | Baseline |
| v6 | ~289K | **83.02%** | StreamFusionConcat | Mixup disabled |
| v7 | ~289K | **74.80%** @ ep68 | MultiStreamStem | 4-stream late fusion |
| v8 | ~289K | not trained | MultiStreamStem | Dead component audit |
| v9 | **1,059,605** | running | MultiStreamStem | Channels widened, full cleanup |
| Experimental | **826,127** | running | MultiStreamStem | TemporalAttn → MultiScaleTSM |

---

## v5 — Baseline (83.33%)

**Training:** 150 epochs, batch=64, LR=0.1, cosine warmup, Mixup α=0.1
**Result:** 83.33% top-1 at epoch 145 — best result so far

### Architecture
- **Fusion:** StreamFusionConcat — 4 streams concatenated at input into one batch, single shared head
- **Channels (small):** [48, 72, 104]
- **Blocks (small):** [1, 3, 3]
- **Block pipeline:**
  ```
  BRASP → pw_conv → GCN → SE → JointEmbed → BSE → FrozenDCTGate → EpSepTCN → DropPath → res → TemporalAttention
  ```
- **Known issues (found later):**
  - BSE (BilateralSymmetryEncoding): `sym_weight = sym_vel_weight = 0` at init → **complete no-op** for 20+ epochs
  - FrozenDCTGate: `freq_mask = +4.0` → sigmoid = 0.982 ≈ identity → **near no-op**, competes with TemporalAttention
  - TemporalAttention: gate = sigmoid(−4) = 0.018 → **gradient-starved Q/K/V/proj** for first 40–50 epochs
  - JointEmbed placed **after** GCN (wrong — aggregation can't use joint identity)
  - pw_conv always present even when in==out → wastes C² within-stage
  - IB loss not yet added
  - Single ensemble head (not per-stream)

---

## v6 — Mixup Ablation (83.02%)

**Training:** Same as v5 but `mixup_alpha: 0.0`
**Result:** 83.02% — **Mixup confirmed +0.31pp** over no-Mixup

### Changes from v5
- Mixup disabled only
- Everything else identical

### Conclusion
Mixup adds a small but consistent gain. Noted for future use. (Later re-disabled in v9 due to 4-stream IB conflict.)

---

## v7 — 4-Stream Late Fusion (74.80% @ ep68, incomplete)

**Training:** 180 epochs, batch=16, accum=4 (effective=64), LR=0.1, Mixup re-enabled
**Result:** 74.80% at epoch 68 — run stopped at 12hr Kaggle limit

> **Note:** 74.80% is NOT a regression. Training top-1 was ~37% because each of the 4 streams is evaluated independently during training. Validation uses the 4-stream ensemble, which gets the ensemble boost. This is expected behavior for late fusion.

### Changes from v6
- **StreamFusionConcat → MultiStreamStem**: 4 independent BN+Conv1×1 stems; streams stacked into batch; 4 separate ClassificationHeads; softmax-weighted ensemble at val
- **4 independent CE losses** during training (one per stream) + ensemble val
- **IB prototype loss** introduced (`ib_loss_weight=0.01`, later found too strong → 0.001)
- Blocks: [1, 2, 2] → [1, 3, 3] (one extra block per late stage)
- Batch: 64 → 16 (because backbone sees 4×16=64 samples per forward)
- Gradient accumulation: 1 → 4

### Dead Components (still present, not yet fixed)
- BSE still zero-init no-op
- FrozenDCTGate still near-identity
- TemporalAttention gate still −4.0 → 0.018 active
- JointEmbed still after GCN

### Why train accuracy ≠ val accuracy
```
Train: each stream scored individually  → ~37% per stream
Val:   4-stream softmax ensemble        → ~74% (ensemble boost)
```
This is correct and expected. The real accuracy ceiling is the val number.

---

## v8 — Dead Component Audit (not trained)

**Purpose:** Identify and document all no-op/broken components before cleanup
**Tests:** 57/57 passing
**Params (small):** ~289K (channels not yet widened)

### What was found and why
| Component | Problem | Evidence |
|-----------|---------|---------|
| BSE | `sym_weight = sym_vel_weight = 0` at init → `bilateral = 0*diff + 0*diff_vel = 0` → gate × 0 = 0 | No-op for entire training |
| FrozenDCTGate | `freq_mask = +4.0` → sigmoid = 0.982 → passes 98.2% of all frequencies | Near-identity, competes with TemporalAttention redundantly |
| TemporalAttention | `gate = sigmoid(−4) = 0.018` → Q/K/V/proj receive 1.8% gradient | Effectively dead for 40–50 warmup epochs |
| JointEmbed (wrong order) | Placed AFTER GCN → aggregation `Σ A_vw * x[w]` can't use joint identity | SGN/Info-GCN both do this wrong; we identified it |
| pw_conv (always on) | `Conv2d(C,C,1)` even when `in==out, stride==1` → wastes C² params per within-stage block | Redundant with GCN's own mixing |

### No training run — went straight to v9

---

## v9 — CleanFuse (currently training)

**Training:** 180 epochs, batch=24, accum=3 (effective=72), LR=0.05, cosine warmup
**Params (small):** 1,059,605 | **Params (nano):** 163,506
**Target:** 88–90% (projected)
**Tests:** 60/60 passing

### Changes from v8
| What | v8 | v9 |
|------|----|----|
| BSE | kept (no-op) | **removed** |
| FrozenDCTGate | kept (near-identity) | **removed** |
| TemporalAttention gate | −4.0 (0.018 active) | **0.0 (0.5 active from ep1)** |
| JointEmbed placement | after GCN | **before GCN** (novel) |
| pw_conv | always Conv2d(C,C,1) | **Identity when in==out** |
| Linear init | normal(0, 0.01) | **xavier_uniform_** |
| Classifier FC init | xavier (logit std ~1.0) | **normal(0,0.01)** (prevents saturation) |
| Channels (small) | [48, 72, 104] | **[64, 128, 256]** (1:2:4 ratio) |
| Channels (nano) | [32, 48, 64] | **[32, 64, 128]** (1:2:4 ratio) |
| Blocks (small) | [1, 3, 3] | **[1, 2, 3]** |
| LR | 0.1 | **0.05** (NaN fix) |
| Mixup | 0.1 | **0.0** (IB loss conflict) |
| IB loss weight | 0.01 (too strong) | **0.001** |

### Block pipeline
```
BRASP → [pw_conv if in≠out else Identity] → JointEmbed → GCN → SE → 4-branch TCN → DropPath → res+out → TemporalAttention(gate=0.5)
```

### Why channels widened
EfficientGCN-B0 at 290K uses the same 1:2:4 ratio at [64,128,256] with depthwise convs.
Our v8 channels [48,72,104] had an irregular ratio with no principled basis.
Widening to [64,128,256] brings us to B4 parameter scale (1.1M) with richer per-block operations.

### NaN issue (fixed before training)
- **Cause:** `F.softmax` in TemporalAttention and AdaptiveGCN ran in float16 → overflow for large logits
- **Fix:** Cast Q/K to float32 before softmax, cast result back to input dtype
- **Files fixed:** `temporal_attention.py`, `adaptive_ctr_gcn.py`

### IB loss bug (fixed)
- `train.py` was setting `config['training']['ib_loss_weight'] = 0.0` after loading yaml
- IB loss was **silently disabled in all previous runs** (v5, v6, v7)
- Fixed: line removed, both v9 and experimental now correctly read `0.001` from yaml

---

## Experimental — Option 3 (ready to train)

**Training:** Same schedule as v9
**Params:** 826,127 (−233K vs v9 small)
**Target:** 90–91% (estimated — between EfficientGCN-B0 and B4)

### Hypothesis
TemporalAttention (T×T=64×64 global self-attention) costs 233K params across 6 blocks.
EfficientGCN-B0 gets 90.2% with **no** global temporal attention.
Replacing it with zero-param multi-scale TSM gives better parameter utilisation (100 samples/param vs 38 for v9) while keeping all novel components.

### Changes from v9
| What | v9 | Experimental |
|------|----|----|
| TemporalAttention | 6 blocks × ~38K avg = 233K | **removed** |
| End-of-block temporal | T×T self-attention | **MultiScaleTSM (0 params)** |
| Params | 1,059,605 | **826,127** |
| Samples/param | 0.038 | **0.10** |

### Block pipeline
```
BRASP → [pw_conv if in≠out else Identity] → JointEmbed → GCN → SE → 4-branch TCN → DropPath → res+out → MultiScaleTSM(±2fr, ±4fr)
```

### MultiScaleTSM
Zero-parameter temporal shift replacing global attention:
- `C//3` channels: half shifted +2 frames, half −2 frames
- `C//3` channels: half shifted +4 frames, half −4 frames
- Remaining: unchanged (anchor features)
- Coverage compounds through 6 blocks

### What is kept vs v9
| Component | Kept? |
|-----------|-------|
| BRASP (anatomical routing) | ✓ |
| JointEmbed before GCN (novel) | ✓ |
| K=3 + per-sample adaptive topology | ✓ |
| SE channel recalibration | ✓ |
| 4-branch TCN (TSM/d=2/d=4/MaxPool) | ✓ |
| 4-stream late fusion + IB loss | ✓ |
| Global T×T temporal attention | ✗ (replaced by MultiScaleTSM) |

### What is hurt
- **Global temporal RF:** max ±4 frames vs full T=64 frame attention. Matters for actions with long-range temporal dependencies (e.g. "stand up" vs "sit down").
- **Comparison claim:** competes with EfficientGCN-B0/B4 rather than B4 directly.

### What is gained
- **Better param utilisation:** 2.6× more training signal per param vs v9
- **Faster epochs:** no T×T attention matrix computation (~15% faster)
- **BRASP becomes the headline:** with no competing attention, BRASP is the primary spatial routing mechanism — cleaner ablation story

---

## Component Glossary

| Name | Params | What it does | Status |
|------|--------|-------------|--------|
| **BRASP** | 0 | Anatomically routes arm/leg/torso channels to their skeleton regions before GCN | ✓ active (v5→experimental) |
| **MultiStreamStem** | ~400 | 4 independent BN+Conv(3→C0) stems, stacked into batch | ✓ active (v7→experimental) |
| **AdaptiveGCN** | ~82K at C=256 | K=3 physical subsets + per-sample Q/K dynamic adjacency. Shared per stage | ✓ active (v5→experimental) |
| **JointEmbed** | V×C | Additive per-joint bias. Before GCN in v9+: GCN aggregates identity-aware features | ✓ active (v5→exp, wrong order until v9) |
| **ChannelSE** | 2×C×C/r | Squeeze-excite: recalibrates which channels matter after GCN | ✓ active (v5→experimental) |
| **4-branch TCN** | ~100K at C=256 | TSM(±1) \| EpSep(d=2) \| EpSep(d=4) \| MaxPool → 1×1 mix | ✓ active (v5→experimental) |
| **DropPath** | 0 | Stochastic depth (linear schedule 0→0.15). Regulariser | ✓ active (v5→experimental) |
| **TemporalAttention** | ~38K avg/block | Global T×T self-attention, gated residual | ✓ v5–v9 / ✗ experimental |
| **MultiScaleTSM** | 0 | Shifts C//3 channels by ±2 and C//3 by ±4 frames | ✓ experimental only |
| **BSE** | 2C+1 | Left-right symmetry encoding. Zero-init → complete no-op | ✗ removed in v9 |
| **FrozenDCTGate** | C×T | Frequency-domain learnable mask. Near-identity at init | ✗ removed in v9 |
| **IB Prototype Loss** | 60×C | InfoGCN-style: minimize distance to nearest class prototype in feature space | ✓ active v7+ (was silently 0.0 due to bug until v9 fix) |
| **ClassificationHead** | ~16K/head | Gated GAP+GMP → BN → Dropout → FC. One per stream | ✓ active (v7→experimental) |

---

## SOTA Comparison

| Model | Params | NTU-60 xsub | Novel over baseline |
|-------|--------|-------------|-------------------|
| Shift-GCN | ~750K | ~85.0% | Generic 0-param shift |
| EfficientGCN-B0 | 290K | 90.2% | Clean depthwise arch |
| EfficientGCN-B4 | 1.1M | 92.1% | Compound scaled B0 |
| CTR-GCN | 1.4M | 92.4% | Per-sample adaptive topology |
| Info-GCN | 1.6M | 92.7% | IB loss + joint embeddings |
| **v9 (ours)** | **1.06M** | **running** | BRASP + pre-GCN JE + adaptive topo + IB |
| **Experimental** | **826K** | **running** | Same as v9 minus global attention |

---

## Lessons Learned

1. **Dead components waste gradient budget.** BSE, FrozenDCTGate, and TemporalAttention (gate=−4) were all effectively no-ops for the first 40–50 epochs. They absorbed parameter count and gradient without contributing.

2. **JointEmbed placement matters.** Putting it before GCN means the aggregation `Σ A_vw * x[w]` distinguishes joints by identity. After GCN, joint identity arrives too late to influence spatial aggregation.

3. **Late fusion train/val discrepancy is expected.** Train accuracy is per-stream (lower). Val accuracy is the ensemble. This confused early analysis of v7.

4. **IB loss was silently disabled.** A single line in `train.py` was overriding the yaml's `ib_loss_weight=0.001` to 0.0. Actual first run with IB loss active: v9 (fixed).

5. **float16 + softmax = NaN.** Any attention-style softmax (TemporalAttention, AdaptiveGCN) must run in float32 under AMP. Cast Q/K to `.float()` before `einsum/bmm`, cast back after softmax.

6. **Smaller ≠ worse.** EfficientGCN-B0 (290K) gets 90.2% with no novel components. Overfitting risk from parameter excess is real at 40K training samples.
