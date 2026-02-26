# Experiment: LAST-Lite (ShiftFuse-GCN)

A novel fixed-compute architecture for edge skeleton action recognition, combining the best
principles from Shift-GCN, SGN, and EfficientGCN with two original contributions:
**Body-Region-Aware Shift (Idea F)** and **Frozen DCT Frequency Routing (Idea G)**.

---

## 1. Architecture

### Block: ShiftFuseBlock

```
input (B, C, T, V)
  │
  ├── BodyRegionShift ★ NOVEL (Idea F)
  │     Channels 0–C/4:       shift among arm joints only
  │     Channels C/4–C/2:     shift among leg joints only
  │     Channels C/2–5C/8:    shift among torso/head joints only
  │     Channels 5C/8–C:      shift across ALL joints (cross-body)
  │     → 0 params, 0 FLOPs overhead vs standard shift
  │
  ├── Conv2d(C, C, 1×1) + BN + Hardswish
  │     Pointwise mixing of shifted channels
  │     → C² + 4C params
  │
  ├── JointEmbedding (from SGN)
  │     x = x + embed[joint_id]   (additive, shared across all blocks in stage)
  │     → 25 × C params (shared)
  │
  ├── FrozenDCTGate ★ NOVEL (Idea G)
  │     x_freq = x @ DCT_matrix              (frozen, no grad)
  │     x_gated = x_freq * σ(freq_mask)      (C×T learnable, data-independent)
  │     x_back = x_gated @ DCT_matrix.T      (frozen)
  │     → C × T params, zero per-sample adaptive compute
  │
  ├── EpSepTCN (from EfficientGCN)
  │     Expand(1×1, ratio=2) → DepthwiseConv(k=5) → Pointwise(1×1) + residual
  │     → ~(2C × 5 + 4C² + 8C) params
  │
  ├── FrameDynamicsGate (from SGN)
  │     gate = σ(frame_embed[t] @ W)    (T position → per-channel gate)
  │     x = x * gate
  │     → T × C + C params
  │
  └── Residual (Conv1×1 + BN if channel mismatch, else Identity)
```

### Full Model

```
Input: Dict{'joint': (B,3,T,V), 'velocity': (B,3,T,V), 'bone': (B,3,T,V)}

  StreamFusion (early input-level stream fusion)
    |- Per-stream BN (3 independent BN2d)
    |- Shared stem Conv2d(3, C0, 1)
    |- Per-channel softmax blend weights (3, C0)
    Output: (B, C0, T, V)
        |
  JointEmbedding init: embed = nn.Embedding(25, C0)
        |
  Stage 1: ShiftFuseBlock × N1, stride=1, C0 → C0
        |
  JointEmbedding update: embed = nn.Embedding(25, C1)  (new for wider channels)
        |
  Stage 2: ShiftFuseBlock × N2, stride=2, C0 → C1
        |
  Stage 3: ShiftFuseBlock × N3, stride=2, C1 → C2
        |
  Gated Head
    |- GAP + GMP + learnable gate blend
    |- BN1d → Dropout → FC(C2, 60)
    Output: (B, 60)
```

---

## 2. Body Region Definitions (NTU-25)

```python
BODY_REGIONS = {
    'left_arm':  [4, 5, 6, 7, 21, 22],     # shoulder → hand tip, thumb
    'right_arm': [8, 9, 10, 11, 23, 24],    # shoulder → hand tip, thumb
    'left_leg':  [12, 13, 14, 15],           # hip → foot
    'right_leg': [16, 17, 18, 19],           # hip → foot
    'torso':     [0, 1, 2, 3, 20],           # spine base → spine shoulder
}

# Channel group allocation (proportional to region importance):
# Arms (24%): fine manipulation (writing, eating, phone)
# Legs (16%): locomotion (walking, kicking)
# Torso (17%): posture (standing, bowing)
# Cross-body (43%): coordination (throwing, hugging)

def get_channel_groups(C):
    arm_end   = C // 4          # 25% for arms
    leg_end   = C // 2          # 25% for legs
    torso_end = 5 * C // 8      # 12.5% for torso
    # remaining 37.5% for cross-body
    return {
        'arm':   slice(0, arm_end),
        'leg':   slice(arm_end, leg_end),
        'torso': slice(leg_end, torso_end),
        'cross': slice(torso_end, C),
    }
```

---

## 3. Shift Index Computation

```python
def compute_shift_indices(A, body_regions, channel_groups, C):
    """
    Precompute shift indices at init. For each channel, determine
    which joint it reads from based on its group assignment.

    Args:
        A: adjacency matrix (V, V) — physical skeleton graph
        body_regions: dict mapping region_name → list of joint indices
        channel_groups: dict mapping group_name → slice
        C: number of channels

    Returns:
        shift_indices: (C, V) — for channel c at joint v, read from joint shift_indices[c, v]
    """
    V = A.shape[0]
    shift_indices = torch.zeros(C, V, dtype=torch.long)

    for group_name, ch_slice in channel_groups.items():
        ch_range = range(ch_slice.start, ch_slice.stop)

        if group_name == 'cross':
            # Cross-body: shift across all joints using graph neighbors
            neighbors = [torch.where(A[v] > 0)[0] for v in range(V)]
            for i, c in enumerate(ch_range):
                for v in range(V):
                    nbrs = neighbors[v]
                    shift_indices[c, v] = nbrs[i % len(nbrs)]
        else:
            # Region-specific: shift only within the region's joints
            if group_name == 'arm':
                region_joints = body_regions['left_arm'] + body_regions['right_arm']
            elif group_name == 'leg':
                region_joints = body_regions['left_leg'] + body_regions['right_leg']
            else:
                region_joints = body_regions['torso']

            for i, c in enumerate(ch_range):
                for v in range(V):
                    if v in region_joints:
                        # Shift within region
                        idx = region_joints.index(v)
                        target = region_joints[(idx + i + 1) % len(region_joints)]
                        shift_indices[c, v] = target
                    else:
                        # Joint not in this region → identity (read from self)
                        shift_indices[c, v] = v

    return shift_indices  # Registered as buffer — no gradient
```

---

## 4. Frozen DCT Gate Implementation

```python
class FrozenDCTGate(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        # Frozen DCT basis (never trained)
        dct_matrix = torch.tensor(
            scipy.fft.dct(np.eye(T), type=2, norm='ortho'),
            dtype=torch.float32
        )
        self.register_buffer('dct', dct_matrix)       # (T, T)
        self.register_buffer('idct', dct_matrix.T)     # (T, T) — DCT-II inverse

        # Learnable frequency mask — data-independent
        # Zero-init → sigmoid(0) = 0.5 → passes all frequencies initially
        self.freq_mask = nn.Parameter(torch.zeros(1, channels, T, 1))

    def forward(self, x):
        # x: (B, C, T, V)
        x_freq = torch.matmul(x.transpose(2, 3), self.dct).transpose(2, 3)
        # x_freq: (B, C, T_freq, V) — in DCT domain

        mask = torch.sigmoid(self.freq_mask)  # (1, C, T, 1) — same for all samples
        x_gated = x_freq * mask

        x_back = torch.matmul(x_gated.transpose(2, 3), self.idct).transpose(2, 3)
        return x_back
```

---

## 5. Variant Configurations

### ShiftFuse-GCN nano (~60K target)

```python
'shiftfuse_nano': {
    'stem_channels': 24,
    'channels': [32, 48, 64],
    'num_blocks': [1, 1, 1],
    'strides': [1, 2, 2],
    'expand_ratio': 2,
    'max_hop': 1,
    'use_dct_gate': True,
    'use_joint_embed': True,
    'use_frame_gate': True,
    'dropout': 0.1,
}
```

### ShiftFuse-GCN small (~120K target)

```python
'shiftfuse_small': {
    'stem_channels': 32,
    'channels': [48, 72, 96],
    'num_blocks': [1, 2, 2],
    'strides': [1, 2, 2],
    'expand_ratio': 2,
    'max_hop': 2,
    'use_dct_gate': True,
    'use_joint_embed': True,
    'use_frame_gate': True,
    'dropout': 0.15,
}
```

### Param Budget Breakdown (small variant)

```
StreamFusion:       3×32×3 + 3×32 = 384          ~0.4K
Stage 1 (1 block):
  Shift:            0
  Pointwise:        48² + 4×48 = 2496             ~2.5K
  JointEmbed:       25 × 48 = 1200                ~1.2K
  DCT Gate:         48 × 64 = 3072                ~3.1K
  EpSepTCN:         48×96 + 96×5 + 96×48 = 9696   ~9.7K
  FrameGate:        64 × 48 + 48 = 3120           ~3.1K
  BN layers:        ~400                           ~0.4K
  Subtotal:                                        ~20.4K

Stage 2 (2 blocks):
  Per block: 72² + 25×72 + 72×64 + EpSepTCN(72)   ~32K
  × 2 blocks + residual conv:                      ~66K

Stage 3 (2 blocks):
  Per block: 96² + 25×96 + 96×64 + EpSepTCN(96)   ~48K
  × 2 blocks + residual conv:                      ~98K

Head: 96×60 + BN(96) + gate = 5952                ~6K

TOTAL:                                             ~191K
```

Slightly over 120K target — can reduce by:
- Narrowing stage 3 to 80 channels: saves ~30K → ~161K
- Or removing DCT gate from stage 1 (low C, less benefit): saves ~3K

---

## 6. Training Plan

### Phase 1: Standalone Training

```bash
python scripts/train.py \
  --model shiftfuse_small --dataset ntu60 --split_type xsub \
  --epochs 120 --batch_size 64 --lr 0.1 \
  --scheduler cosine_warmup --min_lr 0.0001 \
  --amp --env kaggle \
  --set training.warmup_epochs=5 \
       training.warmup_start_lr=0.01 \
       training.label_smoothing=0.1 \
       training.gradient_clip=1.0
```

**No DropPath needed** — no adaptive modules to co-adapt.
**No gradient accumulation needed** — small model fits easily in batch 64.

### Phase 2: Knowledge Distillation (Existing Model → ShiftFuse)

```bash
python scripts/train_distill.py \
  --student shiftfuse_small --teacher existing_model \
  --teacher_ckpt path/to/best_existing_model.pth \
  --alpha 0.5 --tau 4.0 --gamma 0.1 \
  --epochs 120
```

Feature mimicry projections needed at each stage boundary (1×1 conv to match channels).

### Phase 3: MaskCLR Pretraining (if gap exists)

Only if standalone + distillation < 88% (EfficientGCN-B0 threshold).

### Phase 4: Causal Training (Idea E)

```bash
# Add causal training as augmentation flag:
--set training.causal_mask_ratio=0.5
```

50% of batches use left-only padding in EpSepTCN. Zero architecture change.

---

## 7. Ablation Plan

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| Baseline | Standard shift (random) + EpSepTCN only | Pure Shift-GCN + EfficientGCN baseline |
| + JointEmbed | Add joint semantic embedding | Measure SGN contribution |
| + FrameGate | Add frame dynamics gate | Measure temporal position awareness |
| + BodyRegionShift | Replace random shift with region-aware | Measure Idea F contribution |
| + FrozenDCTGate | Add frozen DCT frequency routing | Measure Idea G contribution |
| + CausalTraining | 50% causal masking | Measure Idea E contribution |
| Full ShiftFuse | All components | Full model |
| - DCT Gate | Full minus DCT gate | Is frequency routing worth the params? |
| - JointEmbed | Full minus joint embedding | How much does joint identity help? |

---

## 8. Expected Results

| Model | Params | Standalone | + Distill | + MaskCLR | Edge (INT8) |
|-------|--------|-----------|----------|----------|-------------|
| ShiftFuse nano | ~60K | ~84% | ~87% | ~89% | ~15KB |
| ShiftFuse small | ~120–190K | ~87% | ~89% | ~91% | ~45KB |
| EfficientGCN-B0 | 150K | 88.3% | — | — | — |

**Target:** ShiftFuse small ≥ 89% at ≤ 190K params after distillation.
This beats EfficientGCN-B0 (88.3% at 150K) with a completely different architecture.

---

## 9. Novelty Claims for Paper

1. **Body-Region-Aware Shift (BRASP):** First anatomically-partitioned channel shift for skeleton
   action recognition. Zero-param spatial mixing structured by body semantics.

2. **Frozen DCT Frequency Routing (FDCR):** First fixed-compute frequency-domain channel
   specialization. Learns global frequency preferences per channel without per-sample compute.

3. **ShiftFuse-GCN architecture:** First combination of spatial shifting, semantic joint embedding,
   frequency routing, and depthwise-separable temporal convolution for skeleton recognition.

4. **Finding:** Adaptive modules (gates, attention) require knowledge distillation to generalize
   at small scale. Pure-conv architectures with structural novelty (BRASP, FDCR) achieve
   comparable accuracy without distillation dependency.

---

## 10. Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `src/models/shiftfuse_gcn.py` | ShiftFuse-GCN model | TODO |
| `src/models/blocks/body_region_shift.py` | BRASP shift module | TODO |
| `src/models/blocks/frozen_dct_gate.py` | FDCR frequency gate | TODO |
| `src/models/blocks/joint_embedding.py` | SGN joint embedding | TODO |
| `src/models/blocks/frame_dynamics_gate.py` | SGN frame gate | TODO |
| `tests/test_shiftfuse.py` | Integration tests | TODO |
