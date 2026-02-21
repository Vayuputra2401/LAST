# 03 — Architecture

## A. LAST-v2 (Teacher, 9.2M params)

**Source:** `src/models/last_v2.py`, `src/models/blocks/eff_gcn.py`

### Input Format
```
{
  'joint':    (B, C, T, V, M=2),
  'velocity': (B, C, T, V, M=2),
  'bone':     (B, C, T, V, M=2)
}
```
B=batch, C=3 coords, T=64 frames, V=25 joints, M=2 bodies.

### Forward Pass

Each of the 3 streams processes its input independently through an identical backbone:
```
stream_i:
  DataBN → [EffGCNBlock × N_stages] → GlobalAvgPool → FC(num_classes)
```
The three logit vectors are **summed** before softmax (late fusion). No cross-stream
interaction occurs during backbone processing — each stream can be run independently at
inference time (useful for ablation).

### EffGCNBlock Internals

```
input (B, C_in, T, V)
  │
  ├─ AdaptiveGraphConv
  │    A_total = A_physical + A_learned + A_dynamic(x)
  │    out = Conv2d(A_total @ x)   # (B, C_out, T, V)
  │
  ├─ BN + ReLU
  │
  ├─ ST_JointAtt (optional, some blocks)
  │    spatial attention gate over V dimension
  │
  └─ LinearAttention or TCN (temporal modeling)
       LinearAttention: O(T) complexity via kernel trick
       TCN: standard dilated causal conv (fallback/early blocks)
```

### LAST-v2 Variants

| Variant | Stage channels     | Blocks per stage | Params  |
|---------|--------------------|------------------|---------|
| small   | [64, 128, 256]     | [2, 2, 2]        | ~4.8M   |
| base    | [64, 128, 256]     | [3, 4, 4]        | ~9.2M   |
| large   | [64, 128, 256]     | [4, 6, 6]        | ~14M    |

---

## B. LAST-E (Student, 92K–644K params)

**Source:** `src/models/last_e.py`, `src/models/blocks/light_gcn.py`, `src/models/blocks/stream_fusion.py`

### Input Format
Same dict format as LAST-v2. The forward method strips the M dimension:
```python
s = data[key][..., 0]   # (B, C, T, V)  — takes primary body only
```

### StreamFusion (`src/models/blocks/stream_fusion.py`)

Fuses 3 streams into a single feature tensor before the backbone:

```
stream_j → BN_j → stem_conv(1×1, C=3→C0) → feat_j  (B, C0, T, V)
stream_v → BN_v → stem_conv(1×1, C=3→C0) → feat_v
stream_b → BN_b → stem_conv(1×1, C=3→C0) → feat_b

weights: (3, C0) learnable, softmax over dim=0
output = sum_i  softmax(weights)[i, :] * feat_i    # (B, C0, T, V)
```

Per-channel softmax means each channel can prefer a different stream. This is strictly more
expressive than a scalar blend while using only 3×C0 extra parameters.

### Backbone: LightGCNBlock (`src/models/blocks/light_gcn.py`)

The fused feature (B, C0, T, V) passes through a single shared backbone — one copy, not three:

```
LightGCNBlock:
  input (B, C_in, T, V)
    │
    ├─ DirectionalGCNConv
    │    alpha: (K=3, C_in), zero-initialized
    │    feat_k = alpha[k] * (A_k @ x)   for k in {centripetal, centrifugal, self}
    │    out = Conv2d(sum_k feat_k)        # single shared weight matrix
    │
    ├─ BN + ReLU
    │
    ├─ ST_JointAtt (optional — see variant table)
    │    spatial gate over V, lightweight MLP
    │
    ├─ MultiScaleTCN
    │    branch1: Conv2d(C_in, C//2, 9×1, dilation=1, pad=4)
    │    branch2: Conv2d(C_in, C//2, 9×1, dilation=2, pad=8)
    │    out = cat([branch1, branch2], dim=1)   → (B, C, T, V)
    │    Dropout(0.1) applied after concat
    │
    └─ Residual + ReLU
         skip: Conv2d(C_in, C_out, 1×1) if C_in ≠ C_out, else identity
```

**Dilation padding rule:** `pad = dilation × (kernel − 1) // 2`
- dilation=1, kernel=9 → pad=4
- dilation=2, kernel=9 → pad=8

Both branches produce the same T_out → concatenation is always dimension-safe.

### DirectionalGCNConv Details

Three directed adjacency subsets (pre-computed, registered as buffers):
- `A_0`: centripetal — edges directed toward the root joint
- `A_1`: centrifugal — edges directed away from root
- `A_2`: self-loops — identity connections

Per-channel alpha (K=3, C_in) is **zero-initialized**, so at start of training the model
effectively averages the three subsets equally before learning to specialize. This acts as
a stable initialization that doesn't break training.

### MultiScaleTCN Parameter Savings

A single C×C temporal branch costs C² params per block.
Two parallel C//2 × C branches cost 2 × (C//2 × C) = C² — same count but with two
different receptive fields (dilation 1 + dilation 2), effectively doubling temporal coverage
at zero extra cost.

### LAST-E Variants (Confirmed Parameter Counts)

| Variant | Channels          | Blocks    | ST_Att stages  | Params  | EfficientGCN target |
|---------|-------------------|-----------|----------------|---------|---------------------|
| nano    | [24, 48, 96]      | [2, 2, 3] | [F, F, T]      | 92,358  | <150K (B0) ✓        |
| small   | [32, 64, 128]     | [2, 3, 3] | [F, T, T]      | 177,646 | <300K (B1) ✓        |
| base    | [40, 80, 160]     | [3, 4, 4] | [T, T, T]      | 363,958 | <2M (B4) ✓          |
| large   | [48, 96, 192]     | [4, 5, 5] | [T, T, T]      | 644,094 | <2M (B4) ✓          |

All four variants beat EfficientGCN at every parameter tier.
All param counts verified by integration tests (`tests/test_model_integration.py`).

---

## C. LAST-v2 vs. LAST-E: Design Trade-offs

| Property              | LAST-v2               | LAST-E                          |
|-----------------------|-----------------------|---------------------------------|
| Backbone runs         | 3 (one per stream)    | 1 (fused input)                 |
| Stream interaction    | None (late logit sum) | Early (StreamFusion at input)   |
| GCN style             | AdaptiveGraphConv     | DirectionalGCNConv              |
| Temporal module       | LinearAttention / TCN | MultiScaleTCN (dual dilation)   |
| Spatial attention     | ST_JointAtt (all)     | ST_JointAtt (selective)         |
| FLOPs ratio           | 3× LAST-E-base        | 1×                              |
| Use case              | Teacher / high accuracy | Edge deployment / distillation |
