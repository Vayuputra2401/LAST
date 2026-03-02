# 03 — Architecture

## A. LAST-Lite (ShiftFuse-GCN) — Main Model

**Source:** `src/models/shiftfuse_gcn.py`

### Overview

LAST-Lite is a fixed-computation skeleton GCN with four novel components. The design principle is to replace per-sample adaptive mechanisms (attention, dynamic graphs, gating) with structural inductive biases that encode human skeletal properties directly into the architecture.

### Full Model Flow

```
Input: Dict{'joint': (B,3,T,V,M), 'velocity': ..., 'bone': ..., 'bone_velocity': ...}
  │
  ├── Strip M dimension: s = s[..., 0]  (primary body only)
  │
  ├── StreamFusionConcat (EfficientGCN-style)
  │     Per-stream BN(4) → Concat(4×3=12ch) → Conv1×1(12→C₀) → BN → Hardswish
  │     Output: (B, C₀, T, V)
  │
  ├── Stage 1: [ShiftFuseBlock × N₁], stride=1  → CTRLightGCN₁ (shared)
  │     Output: (B, C₁, T, V)
  │
  ├── Stage 2: [ShiftFuseBlock × N₂], stride=2  → CTRLightGCN₂ (shared)
  │     Output: (B, C₂, T/2, V)
  │
  ├── Stage 3: [ShiftFuseBlock × N₃], stride=2  → CTRLightGCN₃ (shared)
  │     Output: (B, C₃, T/4, V)
  │
  ├── Gated Head
  │     GAP + GMP → σ(gate) blend → BN1d → Dropout → FC(C₃, num_classes)
  │
  └── Output: logits (B, num_classes)
```

### ShiftFuseBlock — Detailed Pipeline

Each block follows a fixed 9-component pipeline. The block is the fundamental unit of LAST-Lite:

```
input (B, C_in, T, V)
  │
  ├── 1. BRASP ★ NOVEL (0 params)
  │     Body-Region-Aware Spatial Shift
  │     Channels 0–C/4:       shift among arm joints
  │     Channels C/4–C/2:     shift among leg joints
  │     Channels C/2–5C/8:    shift among torso/head joints
  │     Channels 5C/8–C:      shift across ALL joints (cross-body)
  │     Pre-computed shift_indices buffer — pure index gather at runtime
  │
  ├── 2. Conv2d(C_in, C_out, 1×1) + BN + Hardswish
  │     Pointwise channel mixing after spatial shift
  │     Params: C_in × C_out + 2 × C_out
  │
  ├── 3. JointEmbedding (from SGN)
  │     x = x + embed[joint_id]   (additive per-joint semantic bias)
  │     Params: V × C_out = 25 × C_out
  │
  ├── 4. BSE ★ NOVEL (2C_out + 1 params)
  │     Bilateral Symmetry Encoding
  │     Computes L-R joint differences for 10 symmetric pairs
  │     + temporal derivative of bilateral signal
  │     Antisymmetric injection: left += g*signal, right -= g*signal
  │     Gate initialised at sigmoid(-2.0) ≈ 0.12 (near-identity at init)
  │
  ├── 5. FDCR ★ NOVEL (C_out × T params)
  │     Frozen DCT Frequency Routing
  │     x_freq = x @ DCT_basis        (frozen, no grad)
  │     x_gated = x_freq * σ(mask)    (learnable C×T mask)
  │     x_back = x_gated @ IDCT_basis (frozen)
  │     output = x + x_back           (residual)
  │     mask init = -2.0 → σ ≈ 0.12, output ≈ 1.12x (near-identity)
  │     Matmul cast to fp32 to avoid AMP fp16 accumulation error
  │
  ├── 6. EpSepTCN / MultiScaleEpSepTCN (temporal convolution)
  │
  │   nano: EpSepTCN (from EfficientGCN)
  │     MobileNetV2-style: Expand(1×1, r=2) → DW(k=5) → PW(1×1) + residual
  │     Applies temporal stride at this stage
  │
  │   small: MultiScaleEpSepTCN ★ NOVEL (parallel multi-scale branches)
  │     Channels split into 3 equal groups:
  │     Branch 0 (C/3): Expand→DW(k=3)→PW — local gestures (3-frame RF)
  │     Branch 1 (C/3): Expand→DW(k=5)→PW — mid-range motion (5-frame RF)
  │     Branch 2 (C/3): MaxPool(k=3)  — sharp temporal transitions
  │     Concat → mix Conv1×1 → residual
  │     Covers 3 temporal scales simultaneously at same channel cost
  │
  ├── 7. DropPath (stochastic depth regularization, 0 params)
  │     Randomly drops entire main branch per sample during training
  │     Rate increases linearly: 0.0 at block 0 → drop_path_rate at last block
  │     nano max rate: 0.10   |   small max rate: 0.15
  │     At inference: identity (no effect on forward pass)
  │
  ├── 8. Outer Residual + Backbone Dropout
  │     out = residual(input) + drop_path(pipeline_output)
  │     backbone_dropout(out)   — intermediate feature regularization
  │     nano: 0.05   |   small: 0.10
  │
  └── 9. CTRLightGCN ★ NOVEL (shared per stage)
        For each group g ∈ {0..G-1}:
          A_g = A_physical_sum + A_group[g]   (physical + learned per-group correction)
          h_g = GroupConv_g(A_g @ x_group_g) (group-specific projection)
        out = x + BN(concat(h_0, ..., h_{G-1}))
        One instance shared across all blocks in the stage
        nano G=2: Params per stage = C²/2 + 2C + 2×625
        small G=4: Params per stage = C²/4 + 2C + 4×625
```

### CTRLightGCN — Stage-Shared Channel-Group Topology Refinement

**Source:** `src/models/blocks/ctr_light_gcn.py`

CTRLightGCN performs spatial graph aggregation after the block residual. Unlike a single-matrix GCN, it splits channels into G groups and assigns each group its own learnable adjacency correction — enabling channel-specific spatial topology specialisation.

**Graph adjacency** per group consists of two components:
1. **A_physical** (V×V): Sum of all K static subsets, registered as a buffer. Shared across all groups.
2. **A_group[g]** (V×V = 625 params per group): Per-group learnable topology correction, zero-initialised. Excluded from weight decay (matched by `'A_group'` keyword in trainer).

**Channel-group spatial specialisation**: Different channel groups can learn different spatial attention patterns. For example, group 0 might strengthen centripetal (parent→child) connections for arm joints, while group 3 focuses on cross-body coordination between arm and leg joints.

**Weight sharing**: Creating one CTRLightGCN per stage and passing it by reference to each block eliminates params per additional block in the stage.

**Parameter efficiency**: At large channel widths, CTRLightGCN with G=4 is *cheaper* than StaticGCN because group convolutions scale as C²/G instead of C²:
- C=96, StaticGCN: 96²+2×96+625 = **10,033** params
- C=96, CTRLightGCN G=4: (24²×4)+2×96+4×625 = **4,996** params (2× cheaper!)

### Novel Component Details

#### BRASP (Body-Region-Aware Spatial Shift)

**Source:** `src/models/blocks/body_region_shift.py`

The human skeleton has five anatomically distinct regions with different kinematic roles:

```python
BODY_REGIONS = {
    'left_arm':  [4, 5, 6, 7, 21, 22],     # shoulder → hand tip, thumb (6 joints)
    'right_arm': [8, 9, 10, 11, 23, 24],    # shoulder → hand tip, thumb (6 joints)
    'left_leg':  [12, 13, 14, 15],           # hip → foot (4 joints)
    'right_leg': [16, 17, 18, 19],           # hip → foot (4 joints)
    'torso':     [0, 1, 2, 3, 20],           # spine base → spine shoulder (5 joints)
}
```

BRASP assigns channel groups to body regions and shifts each channel only among the graph neighbours within its assigned region:
- **Arms group** (25% of channels): shifts within arm joints only
- **Legs group** (25%): shifts within leg joints only
- **Torso group** (12.5%): shifts within torso/head joints only
- **Cross-body group** (37.5%): shifts across ALL joints (global coordination)

The shift indices are pre-computed at init and registered as a buffer. At runtime, BRASP is a single `gather` operation with **zero parameters and zero FLOPs overhead** compared to a standard shift.

**Why this is novel**: Shift-GCN (2020) introduced zero-parameter shifts for skeleton GCN but uses random channel-to-joint assignments. BRASP is the first to use anatomical body-region structure to guide the shift pattern, providing a meaningful inductive bias.

#### BSE (Bilateral Symmetry Encoding)

**Source:** `src/models/blocks/bilateral_symmetry.py`

The human skeleton is bilaterally symmetric: left arm mirrors right arm, left leg mirrors right leg. BSE explicitly encodes this symmetry as a discriminative feature.

**10 symmetric joint pairs** (NTU RGB+D 25-joint):
- Left arm ↔ Right arm: (4,8), (5,9), (6,10), (7,11), (21,23), (22,24)
- Left leg ↔ Right leg: (12,16), (13,17), (14,18), (15,19)
- Torso joints [0,1,2,3,20] lie on the midline and are unaffected.

**Mechanism**:
1. Compute `diff = x[:,:,:,LEFT] - x[:,:,:,RIGHT]` — bilateral difference (B,C,T,10)
2. Compute `diff_vel = temporal_diff(diff)` — symmetry dynamics (B,C,T,10)
3. Weighted combination: `bilateral = w_sym * diff + w_vel * diff_vel`
4. Antisymmetric injection: left joints += g * bilateral, right joints -= g * bilateral

The antisymmetric injection ensures that the bilateral signal acts as a **separation force** (w > 0) or **blending force** (w < 0) between left and right joints, learned independently per channel.

**Discriminative power**:
- Clapping: diff ≈ 0 (symmetric) → bilateral signal near zero
- Walking: diff alternates sign (anti-phase) → bilateral velocity is large
- Drinking: diff is large, persistent (asymmetric) → bilateral difference is large

#### FDCR (Frozen DCT Frequency Routing)

**Source:** `src/models/blocks/frozen_dct_gate.py`

Different action categories occupy different temporal frequency bands: walking is periodic (~2Hz), while punching is impulsive (broadband). FDCR allows each channel to specialise for a particular frequency range.

**Mechanism**:
1. Forward DCT: `x_freq = x @ DCT_basis` (frozen orthonormal matrix, T×T)
2. Frequency masking: `x_gated = x_freq * σ(freq_mask)` (learnable C×T mask)
3. Inverse DCT: `x_back = x_gated @ IDCT_basis` (DCT^T)
4. Residual: `output = x + x_back`

The DCT basis is computed once from `scipy.fft.dct(eye(T), type=2, norm='ortho')` and frozen. Only the frequency mask is learned (C × T parameters per block). The residual connection ensures that even with mask ≈ 0, the original signal passes through.

**Implementation detail**: The matmul is cast to fp32 before execution to prevent AMP fp16 accumulation errors that would otherwise cause ~1-2% accuracy loss.

---

### Variant Configurations

| Property | nano | small |
|----------|------|-------|
| stem_channels | 24 | 32 |
| channels | [32, 48, 64] | [48, 72, 96] |
| num_blocks | [1, 1, 1] | [1, 2, 2] |
| strides | [1, 2, 2] | [1, 2, 2] |
| expand_ratio | 2 | 2 |
| max_hop (K subsets) | 1 (K=3) | 2 (K=5) |
| temporal conv | EpSepTCN(k=5) | MultiScaleEpSepTCN(k=3/5/MaxPool) |
| CTRLightGCN groups G | 2 | 4 |
| drop_path_rate (max) | 0.10 | 0.15 |
| block_dropout | 0.05 | 0.10 |
| head dropout | 0.15 | 0.30 |
| **Total params** | **73,789** | **162,181** |

### Parameter Budget Breakdown (small variant, Round 4)

```
StreamFusionConcat:                              ~2.0K
  4 × BN2d(3) + Conv2d(12→32, 1) + BN(32)

Stage 1 (1 block, C=48):
  BRASP:                  0
  Conv2d(32→48) + BN:     1,632
  JointEmbed:             25 × 48 = 1,200
  BSE:                    2 × 48 + 1 = 97
  FrozenDCTGate:          48 × 64 = 3,072
  MultiScaleEpSepTCN:     ~4,100   (3 branches, C/3=16 each)
  DropPath:               0
  Residual(32→48):        32 × 48 + 96 = 1,632
  block_dropout:          0
  CTRLightGCN₁ (G=4):    576 + 96 + 2,500 = 3,172
  Subtotal:               ~15.0K

Stage 2 (2 blocks, C=72):
  Block 1 (48→72, stride=2): ~15.5K
  Block 2 (72→72, stride=1): ~13.5K
  CTRLightGCN₂ (shared):    1,296 + 144 + 2,500 = 3,940
  Subtotal:                  ~33K

Stage 3 (2 blocks, C=96):
  Block 1 (72→96, stride=2): ~21K
  Block 2 (96→96, stride=1): ~19K
  CTRLightGCN₃ (shared):     2,304 + 192 + 2,500 = 4,996
  Subtotal:                   ~45K

Gated Head:
  pool_gate(96) + BN1d(96) + FC(96→60):          ~6.1K

TOTAL (confirmed):                                162,181
```

### Shape Trace (small variant, B=2, T=64, V=25)

```
Input:       4 × (2, 3, 64, 25)
Fusion:      (2, 32, 64, 25)
Stage 1:     (2, 48, 64, 25)  →  CTRLightGCN₁  →  (2, 48, 64, 25)
Stage 2:     (2, 72, 32, 25)  →  CTRLightGCN₂  →  (2, 72, 32, 25)
Stage 3:     (2, 96, 16, 25)  →  CTRLightGCN₃  →  (2, 96, 16, 25)
Pool:        (2, 96)
Logits:      (2, 60)
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
