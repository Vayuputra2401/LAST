# 03 — Architecture

## A. LAST-Lite (ShiftFuse-GCN) — Main Model

**Source:** `src/models/shiftfuse_gcn.py`
**Current version:** v8 (4-stream late fusion, spatial-first blocks, dilated TCN with TSM)

### Overview

LAST-Lite is a parameter-efficient skeleton GCN built around complementary inductive biases. Instead of one expensive mechanism, it stacks many cheap structural priors (0-param shifts, anatomy-aware routing, frequency gating) so the backbone can specialise with far fewer learnable parameters than generic GCNs.

**v8 design principles:**
- **4-stream late fusion** — each stream (joint/velocity/bone/bone_velocity) has its own stem and classification head; shared backbone processes all 4 with a stacked batch; ensemble at eval
- **Spatial-first block order** (ST-GCN paradigm) — GCN before TCN, single clean outer residual
- **K=3 separate adjacency subsets** — self / centripetal / centrifugal maintain distinct weight matrices
- **4-branch dilated TCN** — TSM (0-param) + d=2 + d=4 + MaxPool for receptive fields {1, 5, 9, 3}

### Full Model Flow (v8)

```
Input: Dict{'joint': (B,3,T,V), 'velocity': ..., 'bone': ..., 'bone_velocity': ...}
  │
  ├── MultiStreamStem — 4 independent stems (one per stream)
  │     Each: BN(3) → Conv1×1(3→C₀) → BN(C₀)
  │     Stack along batch dim → (4B, C₀, T, V)
  │
  ├── Shared Backbone (processes all 4 streams in one forward pass)
  │   ├── Stage 1: [ShiftFuseBlock × N₁], stride=1  → MultiScaleAdaptiveGCN₁ (shared)
  │   ├── Stage 2: [ShiftFuseBlock × N₂], stride=2  → MultiScaleAdaptiveGCN₂ (shared)
  │   └── Stage 3: [ShiftFuseBlock × N₃], stride=2  → MultiScaleAdaptiveGCN₃ (shared)
  │
  ├── Split → 4 × (B, C₃, T/4, V)
  │
  ├── 4 × ClassificationHead (independent per stream)
  │     Gated GAP+GMP → BN1d → Dropout → FC(C₃, num_classes)
  │     Returns (logits, features) per head
  │
  │   Training:  returns [logit_j, logit_v, logit_b, logit_bv], ib_loss
  │              loss = mean(CE(logit_i, target)) + ib_loss_weight * ib_loss
  │
  └── Eval:      softmax-weighted ensemble → (B, num_classes)
```

### ShiftFuseBlock — Detailed Pipeline (v8, spatial-first)

```
input (B, C_in, T, V)
  │
  ├── res = residual(x)                  — outer residual branch
  │
  ├── 1. BRASP ★ NOVEL (0 params)
  │     Body-Region-Aware Spatial Shift
  │     Channels routed within anatomical regions (arms/legs/torso/cross-body)
  │     Pre-computed index buffer — pure gather, zero runtime overhead
  │
  ├── 2. Conv2d(C_in, C_out, 1×1) + BN + Hardswish
  │     Pointwise channel mixing after spatial shift
  │
  ├── 3. MultiScaleAdaptiveGCN (shared, SPATIAL FIRST ← moved before TCN in v7)
  │     K=3 separate adjacency buffers: A_0 (self), A_1 (centripetal), A_2 (centrifugal)
  │     Per-sample topology: A_k_eff = A_k + tanh(α) × A_dynamic(Q/K)
  │     K independent group convolutions → sum → BN → residual
  │     One instance shared across all blocks in the stage
  │
  ├── 4. ChannelSE (EfficientGCN-style, reduce_ratio=4)
  │     GAP → FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → scale
  │     Recalibrates which channels are discriminative after graph propagation
  │
  ├── 5. JointEmbedding (from SGN, V×C params)
  │     x = x + embed[joint_id]  — additive per-joint semantic bias
  │
  ├── 6. BSE ★ NOVEL (2C + 1 params)
  │     Bilateral Symmetry Encoding — L-R joint differences + symmetry velocity
  │     Antisymmetric injection into left/right joint pairs
  │
  ├── 7. FDCR ★ NOVEL (C×T params, no internal residual in v7+)
  │     Frozen DCT frequency routing
  │     x_freq = x @ DCT_basis (frozen) → x_gated = x_freq * σ(mask) → IDCT
  │     output = x_back  (outer block residual handles identity path)
  │     mask init = +4.0 → σ ≈ 0.982 (near-identity at init, critical path)
  │
  ├── 8. MultiScaleEpSepTCN — 4 branches (TEMPORAL SECOND)
  │     Channels split into 4 equal groups (C/4 each):
  │     Branch 0: TSM ★ (0-param temporal shift, ±1 frame) — replaces d=1 conv
  │     Branch 1: EpSep(k=3, d=2) — 5-frame receptive field
  │     Branch 2: EpSep(k=3, d=4) — 9-frame receptive field
  │     Branch 3: MaxPool(k=3) — sharp temporal transitions
  │     → Concat → mix Conv1×1
  │
  ├── 9. DropPath (stochastic depth, 0 params)
  │
  ├── 10. res + out  — single clean outer residual (no double-compound gradient)
  │
  └── 11. LightweightTemporalAttention ★ NOVEL
        Joint-pool → T×T self-attention → gated residual
        gate init = sigmoid(-4) ≈ 0.018 (activates gradually)
        Provides long-range temporal context (complementary to TCN's local patterns)
```

### Novel Component Details

#### BRASP (Body-Region-Aware Spatial Shift)

**Source:** `src/models/blocks/body_region_shift.py`

Shift-GCN introduced zero-parameter shifts (Yan et al., 2020) but uses random channel-to-joint assignments. BRASP is the first to structure shifts by anatomical body region:

```
Arms group   (25% of channels): shifts within {shoulder, elbow, wrist, hand} joints
Legs group   (25%):             shifts within {hip, knee, ankle, foot} joints
Torso group  (12.5%):           shifts within {spine, neck, head} joints
Cross-body   (37.5%):           shifts across ALL joints (global coordination)
```

Zero parameters, zero runtime overhead vs standard shift. Encodes the prior that arm channels should aggregate arm-local context before GCN processes cross-body patterns.

#### MultiScaleAdaptiveGCN (K-subset + per-sample adaptive)

**Source:** `src/models/blocks/adaptive_ctr_gcn.py`

Combines ST-GCN's K-subset paradigm (separate weight matrices per hop) with CTR-GCN's per-sample topology adaptation:

1. **K=3 separate buffers**: `A_0` (self-connections), `A_1` (centripetal), `A_2` (centrifugal) — each independently row-normalised at init
2. **Per-sample Q/K**: `A_dynamic = softmax(Q·K^T / √d_k)` computed per sample, gated by `tanh(α)` (α init=0.1)
3. **K independent group convolutions**: each subset has its own weight matrix (C²/G × K params total)
4. **Effective adjacency**: `A_k_eff = A_k + tanh(α) × A_dynamic`

v5 bug fix: old code summed K subsets into one matrix before the convolution (`A_sum = A.sum(dim=0)`), destroying the per-hop distinction. v7 keeps them separate.

#### ChannelSE

**Source:** `src/models/blocks/channel_se.py`

Squeeze-Excitation (Hu et al., 2018) applied after GCN — exactly as in EfficientGCN-B series. After graph propagation changes which channels carry discriminative information, SE recalibrates the channel importance:

```
x → GAP → FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → scale × x
```

Cost: `2 × C × (C//4)` params per block. No bias → no WD concern.

#### BSE (Bilateral Symmetry Encoding)

**Source:** `src/models/blocks/bilateral_symmetry.py`

No prior skeleton model explicitly exploits bilateral symmetry. BSE encodes L-R joint differences as a discriminative signal:

- **10 symmetric pairs**: (shoulder-L/R), (elbow-L/R), (wrist-L/R), (hand-L/R), (knee-L/R), (ankle-L/R), (foot-L/R), (hip-L/R), (handtip-L/R), (thumb-L/R)
- `diff = feat[LEFT] - feat[RIGHT]` — asymmetry signal
- `diff_vel = temporal_diff(diff)` — symmetry dynamics
- Antisymmetric injection: `LEFT += gate × bilateral;  RIGHT -= gate × bilateral`

Discriminates: clapping (diff≈0), drinking (persistent asymmetry), walking (alternating anti-phase).

#### FDCR (Frozen DCT Frequency Routing)

**Source:** `src/models/blocks/frozen_dct_gate.py`

Actions have characteristic temporal frequencies (walking ~2Hz periodic; punching impulsive broadband). FDCR lets each channel specialise for a frequency range at zero per-sample adaptive cost:

1. `x_freq = x @ DCT` — transform to frequency domain (DCT matrix frozen at init)
2. `x_gated = x_freq × σ(freq_mask)` — per-channel frequency gate (C×T learned params)
3. `x_back = x_gated @ IDCT` — back to time domain
4. No internal residual (added in v7): outer block residual handles identity path; DCT is on critical gradient path

**v7 init fix**: `freq_mask` initialised at +4.0 (σ≈0.982 = near-identity). Old init of -2.0 had the DCT output adding only 12% of input (acted as a noise injection early in training).

#### LightweightTemporalAttention

**Source:** `src/models/blocks/temporal_attention.py`

TCN (even with d=4) sees at most 9 frames. For a 64-frame sequence, "stand up" requires comparing t=0 (sitting) to t=64 (standing). Full T×T self-attention is too expensive at budget; this variant pools joints first:

```
x.mean(V) → Q, K, V via Linear(C, d_k)   # joint-pooled: (B, d_k, T)
attn = softmax(Q^T K / √d_k)             # (B, T, T)
out = attn @ V                            # (B, d_k, T)
x = x + gate × proj(out).unsqueeze(V)    # gated residual, broadcast back
```

Cost: 4 × C × d_k + 1 params. gate init = sigmoid(-4) ≈ 0.018 → activates gradually (avoids disrupting backbone training early).

#### IB Prototype Loss

Each `ClassificationHead` also returns pooled features. During training, an information-bottleneck loss pulls per-class features toward learned class prototypes:

```python
ib_loss = distance(features, class_prototypes[target])
total_loss = CE_loss + 0.001 × ib_loss
```

`class_prototypes` shape: `(num_classes, C)`, zero-init, excluded from weight decay. Weight 0.001 (not 0.01) — calibrated to keep prototype gradient ≤10% of CE gradient.

---

### Variant Configurations (v8)

| Property | nano | small |
|----------|------|-------|
| stem_channels | 24 | 32 |
| channels | [32, 48, 64] | [48, 72, 104] |
| num_blocks | [1, 1, 1] | [1, 2, 3] |
| strides | [1, 2, 2] | [1, 2, 2] |
| temporal conv | MultiScaleEpSepTCN 4-branch | MultiScaleEpSepTCN 4-branch |
| TCN branches | TSM + d=2 + d=4 + MaxPool | TSM + d=2 + d=4 + MaxPool |
| GCN groups G | 4 | 4 |
| drop_path_rate | 0.10 | 0.15 |
| head dropout | 0.10 | 0.20 |
| streams | 4 (joint/vel/bone/bone_vel) | 4 |
| heads | 4 (one per stream) | 4 |
| **Total params** | **76,533** | **289,531** |

### Version History

| Version | Key Change | Val acc (NTU-60 xsub small) |
|---------|-----------|---------------------------|
| v3 | BRASP + BSE + FDCR + DirectionalGCN | 81.67% |
| v4 | Mixup, longer training | 82.47% |
| v5 | AdaptiveCTRGCN + TemporalAttn + DropPath | 83.33% |
| v6 | Gradient flow fixes (BN WD, GCN norm) | 83.02% (Mixup disabled) |
| v7 | 4-stream late fusion, spatial-first GCN, K-subset GCN, ChannelSE | training... |
| **v8** | **4-branch dilated TCN (TSM+d2+d4+MaxPool), channels [48,72,104]** | **training...** |

### Shape Trace (small, v8, B=2, T=64, V=25)

```
Input:        4 × (2, 3, 64, 25)
MultiStemStem: stack → (8, 32, 64, 25)
Stage 1:      (8, 48, 64, 25)  →  MultiScaleAdaptiveGCN₁
Stage 2:      (8, 72, 32, 25)  →  MultiScaleAdaptiveGCN₂
Stage 3:      (8, 104, 16, 25) →  MultiScaleAdaptiveGCN₃
Split:        4 × (2, 104, 16, 25)
4 Heads:      4 × (2, 60)
Ensemble:     (2, 60)
```

---

## B. LAST-Base — High-Accuracy Research Model (Planned)

**Source:** `Docs/Experiment-LAST-Base.md` (design document)

### Overview

LAST-Base is designed to achieve >93% NTU-60 xsub accuracy (beating HI-GCN at 93.3%) with no parameter budget constraint. It integrates the strongest ideas from Generation 2--4 SOTA models alongside two original contributions.

### Block Design

```
LAST-Base Block:
  input (B, C, T, V)
    │
    ├── 1. CrossTemporalPrototypeGCN ★
    │     - Temporal context gathering at scales s ∈ {1, 3}
    │     - Channel-topology refinement (CTR-GCN style, G=4 groups)
    │     - Action-Prototype Graph: K=15 learnable prototype adjacencies,
    │       per-sample softmax blending → class-conditioned topology
    │     - Final: A = A_physical + ΔA_temporal + A_group + A_proto
    │
    ├── 2. FreqTemporalGate ★ (full adaptive version of FDCR)
    │     - DCT transform → per-sample MLP frequency attention → IDCT
    │     - Unlike FDCR, this version is data-dependent (per-sample mask)
    │
    ├── 3. PartitionedTemporalAttention (adapted from SkateFormer)
    │     - 4 attention heads, one per partition type:
    │       Head 1: Near-Joint × Near-Time  (local articulation)
    │       Head 2: Near-Joint × Far-Time   (joint trajectory)
    │       Head 3: Far-Joint  × Near-Time  (body coordination)
    │       Head 4: Far-Joint  × Far-Time   (global action shape)
    │
    ├── 4. HierarchicalBodyRegion (adapted from HD-GCN)
    │     - 5 body regions → intra-region attention → region summary tokens
    │     → inter-region attention → broadcast back to joints
    │
    ├── DropPath (stochastic depth, linear ramp)
    └── Residual connection
```

### Full Model Architecture

```
4-stream ensemble (each stream trained independently):

  StreamFusion: Per-stream DataBN → Stem Conv2d(3, C₀, 1)

  Stage 1: LAST-Base Block × 3, C=128, stride=1
  Stage 2: LAST-Base Block × 4, C=256, stride=2
  Stage 3: LAST-Base Block × 3, C=384, stride=2

  Gated Head: GAP+GMP blend → BN → Dropout(0.3) → FC(384, num_classes)
  IB Loss: Information bottleneck (from InfoGCN) at stage 3

  Inference ensemble: final = mean(softmax(logits_i)) across 4 streams
```

### Expected Parameter Count

```
Single stream:  ~4.2M
4-stream:       ~16.8M
```

### Implementation Status

| Component | Status |
|-----------|--------|
| CrossTemporalPrototypeGCN | Designed, not implemented |
| FreqTemporalGate (adaptive) | Designed, not implemented |
| PartitionedTemporalAttention | Designed, not implemented |
| HierarchicalBodyRegion | Designed, not implemented |
| Full LAST-Base model | Planned |

---

## C. Design Trade-offs: LAST-Lite vs LAST-Base

| Property | LAST-Lite | LAST-Base |
|----------|-----------|-----------|
| Backbone runs | 1 (fused 4-stream input) | 4 (independent per-stream) |
| Stream interaction | Early fusion (StreamFusionConcat) | None (late ensemble) |
| GCN style | CTRLightGCN (G-group topology refinement) | CrossTemporalPrototypeGCN (adaptive) |
| Spatial processing | BRASP (zero-param shift) | Body-region attention |
| Temporal processing | EpSepTCN/MultiScaleTCN + FDCR | Partitioned attention + FreqTemporalGate |
| Symmetry modelling | BSE (explicit bilateral encoding) | Implicit via attention |
| Per-sample adaptive ops | **0** | Many (attention, dynamic graph, prototype blend) |
| Regularization | DropPath + Mixup/CutMix + block dropout | DropPath |
| Quantisable (INT8) | Yes | No |
| Parameters | **74K–162K** | ~4.2M per stream |
| Use case | Edge deployment, real-time | Research, teacher for KD |
