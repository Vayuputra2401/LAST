# 03 — Architecture: ShiftFuse V10

**Source files:**
- `src/models/shiftfuse_v10.py` — main model, V10_VARIANTS, fusion, heads
- `src/models/shiftfuse_gcn.py` — ShiftFuseBlock base class, gcn_scale
- `src/models/blocks/temporal_landmark_attn.py` — TLA module
- `src/models/blocks/body_region_shift.py` — BRASP
- `src/models/blocks/joint_embedding.py` — JointEmbedding
- `src/models/graph.py` — SGP adjacency (strategy='semantic_bodypart')

---

## Overview

ShiftFuse V10 is a four-stream, late-fusion skeleton GCN that achieves competitive accuracy at sub-250K parameters through six novel, domain-motivated architectural components. A single shared backbone processes all four kinematic streams simultaneously (stacked along the batch dimension), producing four per-stream logit vectors that are combined at evaluation via a learned softmax-weighted ensemble.

**Design priorities (in order):**
1. Structural inductive biases over generic learned modules
2. O(T×K) temporal coverage without O(T²) cost
3. Shared parameter pools with per-consumer gain control
4. Information-theoretic class separation at training time

---

## Full Model Flow

```
Input: Dict{
    'joint':         (B, 3, T, V)   — 3D coordinates, spine-base centred
    'velocity':      (B, 3, T, V)   — frame differences of joint
    'bone':          (B, 3, T, V)   — child-parent differences
    'bone_velocity': (B, 3, T, V)   — frame differences of bone
}
    │
    ├── 4 × StreamStem (independent per stream)
    │     BN(3) → Conv2d(3 → C_stem, 1×1) → BN(C_stem)
    │     Stack along batch dim → (4B, C_stem, T, V)
    │
    ├── Shared Backbone (4 streams processed together)
    │   ├── Stage 1: ShiftFuseBlock × N₁ (stride=1)
    │   │     Shared GCN: MultiScaleAdaptiveGCN₁
    │   │     Per-block: JE₁, gcn_scale (when share_gcn=True)
    │   │
    │   ├── Stage 2: ShiftFuseBlock × N₂ (stride=2 at block 0)
    │   │     Shared GCN: MultiScaleAdaptiveGCN₂
    │   │
    │   └── Stage 3: ShiftFuseBlock × N₃ (stride=2 at block 0)
    │         Shared GCN: MultiScaleAdaptiveGCN₃
    │
    ├── Split → 4 × (B, C₃, T/4, V)
    │
    ├── 4 × ClassificationHead (independent per stream)
    │     GAP → BN1d → Dropout → FC(C₃, num_classes)
    │     Returns (logits_i, features_i)
    │
    │   Training path:
    │     per-head loss = CE(logits_i, labels)
    │     total_ce = mean(per_head_ce)
    │     ib_loss = ReLU(0.5 + d_same − d_wrong).mean()  ← triplet margin
    │     total = total_ce + ib_loss_weight × ib_loss
    │
    └── Eval path:
          w = softmax(stream_weights)           ← learned (4,) vector
          ensemble = Σ_i w[i] × logits_i        ← weighted sum
          return argmax(ensemble)
```

---

## ShiftFuseBlock — Detailed Pipeline (V10.3)

```
input x: (4B, C_in, T, V)
    │
    ├── res = residual_branch(x)           — 1×1 conv if C_in≠C_out or stride≠1
    │
    ├── 1. BRASP ★ NOVEL (0 params)
    │     Zero-parameter anatomical channel permutation.
    │     Applied BEFORE pw_conv so BN sees anatomically structured features.
    │     Channels partitioned by body region; permuted within region neighbourhood.
    │
    ├── 2. pw_conv: Conv2d(C_in, C_out, 1×1) + BN + ReLU6
    │     Channel mixing after anatomical shift.
    │     At stage transitions: also performs temporal striding (stride=2).
    │
    ├── 3. MultiScaleAdaptiveGCN (SHARED per stage)
    │     See Section "MultiScaleAdaptiveGCN" below.
    │     When share_gcn=True: one instance per stage, passed as external module.
    │
    ├──   gcn_scale ★ NOVEL (1 param per block, when share_gcn=True)
    │     out = gcn_scale × gcn(x)
    │     Scalar, init=1.0, no weight decay.
    │     Guards gradient magnitude through shared GCN.
    │
    ├── 4. JointEmbedding (SHARED per stage when share_je=True)
    │     Additive per-joint bias: x = x + embed[joint_id]
    │     (V × C_out params per stage, zero-init)
    │
    ├── 5. MultiScaleTCN — 4-branch temporal convolution
    │     Channels split into 4 groups (C_out//4 each):
    │     Branch 0: k=3, dilation=1 (local motion)
    │     Branch 1: k=3, dilation=2 (5-frame receptive field)
    │     Branch 2: k=3, dilation=4 (9-frame receptive field)
    │     Branch 3: MaxPool k=3 (sharp temporal transitions)
    │     → Concat → Conv1×1 mix → BN
    │
    ├── 6. DropPath (stochastic depth, linear ramp 0→max_rate)
    │
    ├── 7. x = res + DropPath(out)        — clean outer residual
    │
    └── 8. TLA ★ NOVEL (per-block, not shared)
          TemporalLandmarkAttention(channels=C_out, K=14)
          See Section "Temporal Landmark Attention" below.
          Returns x + sigmoid(gate) × attention_output
```

---

## Novel Component Details

### 1. Semantic Body-Part Graph (SGP) ★

**Source:** `src/models/graph.py`, `strategy='semantic_bodypart'`

SGP replaces the generic hop-distance partitioning of ST-GCN with three anatomically-typed adjacency subsets:

| Subset | Edges | Semantic Role |
|--------|-------|---------------|
| **A_intra** | Within body regions (arm-arm, leg-leg, torso-torso) | Regional articulation |
| **A_inter** | Adjacent region boundaries (shoulder↔spine, hip↔spine) | Coordination at region interfaces |
| **A_cross** | Long-range cross-body pairs (left hand↔right knee, etc.) | Bilateral and cross-body coordination |

All subsets normalised with symmetric D^{−1/2}AD^{1/2} normalisation. max_hop=2 extends A_cross to two-hop cross-body connections, enabling pairs like (left elbow ↔ right hip) that are critical for asymmetric actions.

**Comparison to ST-GCN K=3:** ST-GCN partitions by graph distance (self-link / inward / outward). SGP partitions by anatomical membership, encoding body-part kinematic roles as an architectural prior.

```
NTU-25 joint groups:
  Torso:     spine (0), spine-shoulder (1), neck (2), head (3), hip (12)
  Left arm:  left shoulder (4), elbow (5), wrist (6), hand (7), handtip (21), thumb (22)
  Right arm: right shoulder (8), elbow (9), wrist (10), hand (11), handtip (23), thumb (24)
  Left leg:  left hip (16), knee (17), ankle (18), foot (19)
  Right leg: right hip (20), knee (21), ankle (22), foot (23)
```

---

### 2. Temporal Landmark Attention (TLA) ★

**Source:** `src/models/blocks/temporal_landmark_attn.py`

**Complexity:** O(T×K) vs O(T²) for full self-attention.

| Method | Dot-products (T=64, K=14) | Relative cost |
|--------|--------------------------|---------------|
| Full T×T attention | 64×64 = 4,096 | 1.0× (reference) |
| TLA (K=14 landmarks) | 64×14 = 896 | **0.22× (4.6× cheaper)** |
| TLA (K=8 landmarks) | 64×8 = 512 | 0.125× (8× cheaper) |

**Design:**
```
x: (B, C, T, V)
  → pool over V → x_t: (B, T, C)
  → uniformly-spaced landmark indices: stride = T//K, indices[0:K]
  → x_land: (B, K, C)         — landmark frame features

  Q = Linear(C → d_k)(x_t)    (B, T, d_k)   — all frames attend
  K = Linear(C → d_k)(x_land) (B, K, d_k)   — to landmark frames
  V = Linear(C → d_k)(x_land) (B, K, d_k)

  # Float32 throughout for AMP stability
  attn = softmax(Q @ K^T / √d_k, dim=-1)     (B, T, K)
  out  = attn @ V                             (B, T, d_k)
  out  = Linear(d_k → C)(out)                (B, T, C)

  # Gated residual
  gate = sigmoid(self.gate)   # scalar, init=0.0 → 0.5 active at start
  return x + gate × out.permute(0,2,1).unsqueeze(-1)
```

**Parameters** (d_k = max(4, C//8)):
- Q/K/V/proj: 4 × C × d_k params
- gate: 1 scalar
- Example C=128, d_k=16: 4×128×16 + 1 = **8,193 params**

**Rationale for K=14 over K=8:**
At T=64, K=14 landmarks are spaced every ~4.5 frames, giving sub-action-phase resolution. K=8 (8-frame gaps) loses fine temporal structure. K=14 adds 448 extra dot-products per block (negligible vs 4.6× cheaper than full attention).

**Placement after residual:** TLA is applied after the residual sum so it sees fully integrated spatial+temporal features. The gate starts at 0.5 (sigmoid(0)) to provide meaningful attention from epoch 1, avoiding the cold-start problem of the old gate=-4.0 initialisation.

---

### 3. BRASP — Body-Region-Aware Spatial Shift (Corrected) ★

**Source:** `src/models/blocks/body_region_shift.py`

BRASP is a zero-parameter operation that permutes channel features within anatomically-relevant joint neighbourhoods before any learned transformation:

```
Channel partitioning (25 joints, 4 groups):
  Arms group   (25% of channels): joints in {left arm ∪ right arm}
  Legs group   (25% of channels): joints in {left leg ∪ right leg}
  Torso group  (12.5% of channels): joints in {torso joints}
  Cross-body   (37.5% of channels): all joints (global coordination)

Within each group: channels are cyclically shifted along the joint dimension
  arm_channels at joint j → arm_channels at next-arm-group joint
```

**V10.3 correction — `brasp_after_pw=False`:**

In V10.1/2, BRASP was placed after pw_conv (`brasp_after_pw=True`). This was incorrect: pw_conv contains BatchNorm, whose running statistics are computed on joint-shuffled features, causing the BN to learn statistics for permuted activations. GCN then receives BN-normalised joint-shuffled features and learns to *ignore* the permutation structure — nullifying BRASP's anatomical routing benefit.

With `brasp_after_pw=False`, BRASP runs on the raw input before channel mixing. The pointwise conv and its BN then see anatomically-routed features, and the GCN aggregates pre-structured information.

---

### 4. MultiScaleAdaptiveGCN ★

**Source:** `src/models/shiftfuse_v10.py` (`_build_gcn` method)

Combines SGP static adjacency with per-sample Q/K dynamic topology:

```
Input: x (B, C_in, T, V)
  Static branch: K=3 subsets → K × GroupConv(C//G, C//G) → BN → sum
  Dynamic branch (per-sample):
    Q = Linear(C, d_k)(x.mean(T,V))  (B, V, d_k)
    K = Linear(C, d_k)(x.mean(T,V))  (B, V, d_k)
    A_dynamic = softmax(Q @ K^T / √d_k)  (B, V, V) per-sample
    Blended: A_eff = A_static[k] + tanh(alpha) × A_dynamic
  Output: BN(sum over K subsets) + residual
```

**Stage sharing (share_gcn=True):** One `MultiScaleAdaptiveGCN` instance per stage. All N blocks in the stage share its weights. The `gcn_scale` parameter (per block) compensates by providing independent amplitude control:

```
block_0_output = gcn_scale_0 × GCN(x_0)
block_1_output = gcn_scale_1 × GCN(x_1)   # same GCN, different scale
```

**Group conv (G=4):** Depthwise GCN (G=C) lacks cross-channel mixing, which is critical for multi-stream fusion where different channels encode different kinematic aspects. G=4 groups provide mixing while remaining cheap: (C/G)² × K × G = C²/G × K params vs C² × K for dense conv.

---

### 5. JointEmbedding (Stage-Shared)

**Source:** `src/models/blocks/joint_embedding.py`

Additive per-joint semantic bias (SGN, CVPR 2020):

```
x = x + embed[joint_id]   # embed shape: (V, C)
```

The embedding provides each joint with a distinct learned offset that persists across all frames and all samples. This encodes joint identity (wrist vs ankle) as a constant feature component, complementary to the data-dependent GCN output.

**Stage sharing (share_je=True):** One embedding table per stage (3 tables total). All blocks in a stage share the same embedding. Parameter savings: (N_blocks − 1) × V × C_stage per stage. For nano (blocks=[2,3,2], C=[32,64,128], V=25): saves ~6.1K params.

---

### 6. Classification Head and Triplet IB Loss

**Source:** `src/models/shiftfuse_v10.py` (`ClassificationHead`, `forward()`)

**Head structure** (4 independent heads, one per stream):
```
x: (B, C₃, T/4, V)
  → GAP: x.mean([2,3]) → (B, C₃)
  → BN1d(C₃)
  → Dropout(p)
  → Linear(C₃, num_classes)
  → (logits: (B, num_classes), features: (B, C₃))
```

**Triplet-Margin IB Loss:**
```python
# Per training step, over the mean feature across all 4 stream heads:
mean_feat   shape: (B, C₃)
class_prototypes: (num_classes, C₃)   # learned, zero-init, no WD

proto_dists = cdist(mean_feat, class_prototypes)   # (B, num_classes)

d_same = proto_dists[arange, labels]               # pull toward correct prototype

proto_dists_wrong = proto_dists.clone()
proto_dists_wrong[arange, labels] = inf            # mask correct class
d_wrong = proto_dists_wrong.min(dim=1).values      # nearest incorrect prototype

ib_loss = F.relu(0.5 + d_same - d_wrong).mean()   # margin=0.5
```

**Why triplet margin improves over attraction-only:**
- Attraction-only (`d_same.mean()`) pulls samples to prototypes but gives no gradient once the sample is near its prototype — even if it is equally near a wrong prototype.
- The ReLU margin `relu(0.5 + d_same - d_wrong)` gives zero loss only when `d_wrong > d_same + 0.5`, i.e., the sample is at least 0.5 units closer to the correct prototype than the nearest wrong one. This enforces an explicit decision boundary gap.

---

### 7. Learned Stream-Weight Ensemble ★

**Source:** `src/models/shiftfuse_v10.py` (`ShiftFuseV10.forward()`)

At evaluation:
```python
w = F.softmax(self.stream_weights, dim=0)   # (4,) → sums to 1
ensemble = sum(w[i] * logits[i] for i in range(4))
```

`stream_weights` is an `nn.Parameter(torch.zeros(4))`, giving uniform weights at initialisation. During training, gradients flow from the four head losses back through the ensemble, causing stream_weights to diverge toward the streams that most consistently distinguish classes. Expected behavior: bone and joint streams acquire higher weight for pose-dominant actions; velocity streams acquire higher weight for motion-dominant actions.

`stream_weights` is excluded from weight decay (no_decay in trainer.py).

---

## Variant Configurations

| Property | nano | small | large |
|----------|------|-------|-------|
| **Params** | **225,533** | 1,425,050 | 3,100,506 |
| stem_channels | 24 | 32 | 48 |
| channels | [32, 64, 128] | [64, 128, 256] | [96, 192, 384] |
| num_blocks | [2, 3, 2] | [2, 3, 4] | [2, 3, 4] |
| strides | [1, 2, 2] | [1, 2, 2] | [1, 2, 2] |
| expand_ratio | 2 | 2 | 2 |
| max_hop | 2 | 2 | 2 |
| drop_path_rate | 0.10 | 0.15 | 0.20 |
| dropout (head) | 0.10 | 0.20 | 0.25 |
| share_gcn | True | False | False |
| share_je | True | False | False |
| TLA (use_tla) | True | True | True |
| tla_landmarks K | 14 | 14 | 14 |
| tla_reduce_ratio | 8 | 8 | 8 |
| use_stream_bn | False | False | False |
| single_head | False | False | False |

---

## Parameter Breakdown: nano (225,533 total)

| Component | Count | Params | Notes |
|-----------|-------|--------|-------|
| 4 × StreamStem | 4 | ~4.2K | BN(3) + Conv(3→24) + BN(24) |
| Stage 1 blocks (N=2) | 2 | ~22K | pw_conv + TCN per block |
| Stage 1 GCN (shared) | 1 | ~8K | K=3 group conv C=32, G=4 |
| Stage 1 JE (shared) | 1 | ~0.8K | 25×32 |
| TLA stage 1 (N=2) | 2 | ~1.0K | C=32, d_k=4 |
| Stage 2 blocks (N=3) | 3 | ~65K | pw_conv + TCN per block |
| Stage 2 GCN (shared) | 1 | ~28K | K=3 group conv C=64, G=4 |
| Stage 2 JE (shared) | 1 | ~1.6K | 25×64 |
| TLA stage 2 (N=3) | 3 | ~6.1K | C=64, d_k=8 |
| Stage 3 blocks (N=2) | 2 | ~70K | pw_conv + TCN per block |
| Stage 3 GCN (shared) | 1 | ~88K | K=3 group conv C=128, G=4 |
| Stage 3 JE (shared) | 1 | ~3.2K | 25×128 |
| TLA stage 3 (N=2) | 2 | ~16.5K | C=128, d_k=16 |
| 4 × ClassificationHead | 4 | ~31K | BN1d + FC(128→60) |
| class_prototypes | 1 | ~7.7K | 60×128, zero-init, no WD |
| gcn_scale (7 blocks total) | 7 | 7 | 1 scalar per block |
| stream_weights | 1 | 4 | 4 scalars, no WD |

---

## Shape Trace: nano (B=2, T=64, V=25)

```
Input:          4 × (2, 3, 64, 25)
MultiStreamStem: stack → (8, 24, 64, 25)

Stage 1, Block 0 (stride=1):
  BRASP → pw_conv(24→32) → GCN → JE → TCN → DropPath
  TLA → (8, 32, 64, 25)

Stage 1, Block 1:
  (8, 32, 64, 25) → (8, 32, 64, 25)

Stage 2, Block 0 (stride=2):
  BRASP → pw_conv(32→64, stride=2) → GCN → JE → TCN → DropPath
  TLA → (8, 64, 32, 25)

Stage 2, Block 1: (8, 64, 32, 25) → (8, 64, 32, 25)
Stage 2, Block 2: (8, 64, 32, 25) → (8, 64, 32, 25)

Stage 3, Block 0 (stride=2):
  BRASP → pw_conv(64→128, stride=2) → GCN → JE → TCN → DropPath
  TLA → (8, 128, 16, 25)

Stage 3, Block 1: (8, 128, 16, 25) → (8, 128, 16, 25)

Split: 4 × (2, 128, 16, 25)

4 × ClassificationHead:
  GAP(T,V) → (2, 128) → BN1d → Dropout → FC → (2, 60)

Eval ensemble:
  softmax(stream_weights) × 4 logit tensors → (2, 60)
```

---

## Comparison: ShiftFuse V10 vs Prior LAST-Lite (v9)

| Property | LAST-Lite v9 (CleanFuse) | ShiftFuse V10.3 (nano) |
|----------|--------------------------|------------------------|
| **Params** | 163,506 | **225,533** |
| **Graph** | K=3 hop-distance (ST-GCN) | **K=3 typed SGP (A_intra/A_inter/A_cross)** |
| **Temporal** | Full T×T LightweightTemporalAttn | **O(T×K) TLA (K=14 landmarks)** |
| **Spatial routing** | BRASP (after pw_conv, broken) | **BRASP (before pw_conv, correct)** |
| **IB loss** | Attraction-only | **Triplet margin (d_same − d_wrong)** |
| **GCN sharing** | Per-block | **Stage-shared + gcn_scale guard** |
| **JE sharing** | Per-block | **Stage-shared** |
| **Ensemble** | Uniform mean | **Learned softmax stream_weights** |
| **Architecture** | 1 GCN per block, BSE, FDCR removed | **6 novel components, no BSE/FDCR** |
