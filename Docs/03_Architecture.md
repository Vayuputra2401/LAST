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
interaction occurs during backbone processing.

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

## B. LAST-E v3 (Student, 83K–1.08M params)

**Source:** `src/models/last_e_v3.py`, `src/models/blocks/spatial_gcn.py`,
`src/models/blocks/ep_sep_tcn.py`, `src/models/blocks/motion_gate.py`,
`src/models/blocks/st_joint_att.py`, `src/models/blocks/stream_fusion.py`

### Input Format
Same dict format as LAST-v2. The forward method strips the M dimension:
```python
s = data[key][..., 0]   # (B, C, T, V) — takes primary body only
```

### Full Model Flow

```
Input: Dict{'joint': (B,3,T,V), 'velocity': (B,3,T,V), 'bone': (B,3,T,V)}

  StreamFusion
    |- Per-stream BN (3 independent BN2d)
    |- Shared stem Conv2d(3, C0, 1)
    |- Per-channel softmax blend weights (3, C0)
    Output: (B, C0, T, V)
        |
  Stage 1: V3Block x N1, stride=1, C0 → C0   (local patterns)
        |
  Stage 2: V3Block x N2, stride=2, C0 → C1   (mid-level dynamics)
        |
  Stage 3: V3Block x N3, stride=2, C1 → C2   (high-level semantics)
        |
  Gated Head
    |- GAP: AdaptiveAvgPool2d(1,1)
    |- GMP: AdaptiveMaxPool2d(1,1)
    |- Gated fusion: gap * σ(gate) + gmp * (1 - σ(gate))
    |- BN1d(C2) → Dropout → FC(C2, num_classes)
    Output: logits (B, num_classes) [+ ib_loss if use_ib_loss]
```

### V3Block Internals

```
V3Block: SpatialGCN → Hardswish → EpSepTCN(×depth) → MotionGate/HybridGate → ST_JointAtt → DropPath + residual

input (B, C_in, T, V)
  │
  ├─ SpatialGCN(C_in, C_out, A)
  │    K subsets (K=3 for 1-hop, K=5 for 2-hop)
  │    Full-graph D⁻½AD⁻½ normalization (N1 fix)
  │    Learnable edge weights + optional subset attention
  │    out = Conv2d(sum_k feat_k)
  │
  ├─ Hardswish activation
  │
  ├─ EpSepTCN (×depth, MobileNetV2-style)
  │    Expand(1×1, r=2) → DepthwiseTemporal(k=5) → Pointwise(1×1) + residual
  │    Multi-scale receptive field via stacking (not branching)
  │
  ├─ MotionGate or HybridGate
  │    MotionGate: sigmoid(BN(W · (x_t - x_{t-1}))) ⊙ x  (temporal-diff gating)
  │    HybridGate: MotionGate + SE-style channel attention blend
  │    ★ Novel — no prior skeleton work uses temporal-difference channel gating
  │
  ├─ ST_JointAtt (optional per stage)
  │    Factorized: temporal attention (global avg over V → sigmoid gate)
  │              + spatial attention (global avg over T → sigmoid gate)
  │    Zero-init α residual gate: output = x + α × att(x), α=0 at init
  │
  └─ DropPath + Residual
       Stochastic depth (random drop entire branch, scale by 1/keep)
       Residual: Conv2d(1×1) + BN if channel/stride mismatch, else Identity
```

### LAST-E v3 Variant Configurations

```python
MODEL_VARIANTS_E_V3 = {
    'nano': {
        'stem_channels': 24, 'channels': [32, 48, 64],
        'num_blocks': [1, 1, 1], 'depths': [1, 1, 1],
        'strides': [1, 2, 2], 'expand_ratio': 2,
        'max_hop': 1,  # K=3 subsets
        'gate_type': 'motion',
        'use_st_att': [True, True, True],
        'use_subset_att': False, 'use_ib_loss': False,
        'dropout': 0.2, 'drop_path_rate': 0.0,
    },
    'small': {
        'stem_channels': 32, 'channels': [48, 64, 96],
        'num_blocks': [1, 2, 2], 'depths': [1, 1, 1],
        'strides': [1, 2, 2], 'expand_ratio': 2,
        'max_hop': 2,  # K=5 subsets
        'gate_type': 'motion',
        'use_st_att': [True, True, True],
        'use_subset_att': True, 'use_ib_loss': False,
        'dropout': 0.25, 'drop_path_rate': 0.0,
    },
    'base': {
        'stem_channels': 48, 'channels': [64, 96, 128],
        'num_blocks': [2, 2, 2], 'depths': [1, 1, 1],
        'strides': [1, 2, 2], 'expand_ratio': 2,
        'max_hop': 2,  # K=5 subsets
        'gate_type': 'motion',
        'use_st_att': [True, True, True],  # overridable via CLI
        'use_subset_att': True, 'use_ib_loss': True,
        'dropout': 0.3, 'drop_path_rate': 0.05,
    },
    'large': {
        'stem_channels': 48, 'channels': [80, 112, 160],
        'num_blocks': [2, 2, 2], 'depths': [1, 1, 1],
        'strides': [1, 2, 2], 'expand_ratio': 2,
        'max_hop': 2,  # K=5 subsets
        'gate_type': 'hybrid',  # HybridGate = MotionGate + SE blend
        'use_st_att': [True, True, True],
        'use_subset_att': True, 'use_ib_loss': True,
        'dropout': 0.3, 'drop_path_rate': 0.1,
    },
}
```

### LAST-E v3 Confirmed Parameter Counts

| Variant | Params | EfficientGCN target |
|---------|--------|---------------------|
| LAST-E v3 nano | **82,847** | < B0 (150K) ✓ |
| LAST-E v3 small | **344,727** | < B2 (540K) ✓ |
| LAST-E v3 base | **720,028** | < B4 (2M) ✓ |
| LAST-E v3 large | **1,080,668** | < B4 (2M) ✓ |

### Configurable Phase B (ST_JointAtt Ablation)

The `use_st_att` parameter can be overridden via CLI:
```bash
--set model.use_st_att=false,false,true   # attention only in stage 3
```
Phase B config `[False, False, True]` saves ~27K params (3.8%) from base variant:
- Stages 1–2: ST_JointAtt → Identity (no attention)
- Stage 3: ST_JointAtt active (where it matters most with high channels)

---

## C. LAST-Lite (Edge Deployment, ~60K–180K params)

**Status:** Planned (post-training)

LAST-Lite variants are **fixed-computation** models designed for edge deployment. They share
the LAST graph structure but remove all per-sample adaptive modules.

### What Changes from LAST-E v3

| Component | LAST-E v3 | LAST-Lite |
|-----------|-----------|-----------|
| MotionGate / HybridGate | Per-sample sigmoid gating | **Removed** → Identity |
| ST_JointAtt | Per-sample attention | **Removed** → Identity |
| Subset attention | Softmax over K subsets | **Removed** → simple sum |
| Learnable edge | Per-edge learned weight | **Removed** → fixed adjacency |
| SpatialGCN (fixed) | ✓ | ✓ (kept) |
| EpSepTCN | ✓ | ✓ (kept) |
| Gated GAP+GMP head | ✓ | ✓ (kept) |
| DropPath | ✓ | ✗ (not needed) |

### LAST-Lite Variant Configs (Planned)

```python
'nano_lite': {
    'channels': [32, 48, 64],
    'num_blocks': [1, 1, 1],
    'max_hop': 1,
    'gate_type': 'none',
    'use_st_att': [False, False, False],
    'use_subset_att': False,
    'use_learnable_edge': False,
}
# Estimated: ~60K params, ~0.5ms CPU, INT8: ~15KB

'small_lite': {
    'channels': [48, 72, 96],
    'num_blocks': [1, 2, 2],
    'max_hop': 2,
    'gate_type': 'none',
    'use_st_att': [False, False, False],
    'use_subset_att': False,
    'use_learnable_edge': False,
}
# Estimated: ~180K params, ~3ms CPU, INT8: ~45KB
```

### Why LAST-Lite ≠ EfficientGCN

Even after removing all adaptive modules, 4 structural differences remain:

1. **Full-graph D⁻½AD⁻½ normalization** (vs per-subset D⁻¹A)
2. **Late 3-stream concat fusion** (vs early MIB concat at input)
3. **Gated GAP+GMP pooling head** (vs GAP only)
4. **Hardswish activation** (vs ReLU)

---

## D. Novel Ideas — Allocation to Base vs Edge

Five novel ideas from `Docs/novel_solutions.md`. Which go into base vs lite:

| Idea | Description | For Base? | For Lite? | Params |
|------|-------------|-----------|-----------|--------|
| **A. FreqTemporalGate** | DCT frequency-domain channel attention | ✅ Yes | ❌ No (FFT not edge-friendly) | ~5K/block |
| **B. Action-Prototype Graph** | K=15 class-conditioned adjacency via prototypes | ✅ Yes | ❌ No (per-sample ops) | ~10K total |
| **C. Progressive Re-fusion** | Stream re-injection at different backbone depths | ✅ Yes | ❌ No (complexity) | ~10K/stage |
| **D. Hierarchical Body-Region Att** | Anatomical partition: intra-region + inter-region | ✅ Yes | ⚠️ Maybe (if fixed partition) | ~20K/stage |
| **E. Causal Training** | 50% causal masking during training | ✅ Yes | ✅ Yes (training only, 0 params) | 0 |

**Recommendation for base:** Start with A (FreqTemporalGate) — orthogonal to existing components, only ~5K params/block, addresses unexplored frequency axis. Add B (APG) if frequency gating gains are validated.

**Recommendation for lite:** Only E (Causal Training) applies — zero inference-time overhead, purely a training regime change. All other ideas involve per-sample operations or FFT which defeat the purpose of fixed-computation edge models.

---

## E. LAST-v2 vs LAST-E v3 vs LAST-Lite: Design Trade-offs

| Property | LAST-v2 | LAST-E v3 | LAST-Lite |
|----------|---------|-----------|-----------|
| Backbone runs | 3 (one per stream) | 1 (fused input) | 1 (fused input) |
| Stream interaction | None (late logit sum) | Early (StreamFusion) | Late concat |
| GCN style | AdaptiveGraphConv | SpatialGCN (N1 norm) | SpatialGCN (fixed, no edge/att) |
| Temporal module | LinearAttention + TCN | EpSepTCN (stacked) | EpSepTCN (stacked) |
| Channel gating | None | MotionGate / HybridGate | None |
| Spatial attention | ST_JointAtt (all) | ST_JointAtt (selective) | None |
| Per-sample adaptive ops | Many | 14 (gates + attention) | **0** |
| Quantizable (INT8) | ❌ | ❌ | ✅ |
| Params | 4.8M–14M | 83K–1.08M | ~60K–180K |
| Use case | Teacher / high accuracy | Standalone / distillation | Edge / real-time |
