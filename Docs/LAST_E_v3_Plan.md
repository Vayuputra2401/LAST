# LAST-E v3 Architecture Plan

Full redesign of LAST-E to align with SOTA (CTR-GCN + EfficientGCN) while incorporating
a novel Frequency-Aware Temporal Gate. Targets all four variants (nano/small/base/large).

---

## 1. Diagnosis: Why LAST-E Plateaus

### 1.1 GCN Expressiveness Bottleneck

**Current** (`src/models/blocks/light_gcn.py:38-154`, `DirectionalGCNConv`):
All channels share a single combined topology. The K=3 physical subsets are weighted
per-channel via alpha softmax, but after summation with A_learned and A_dynamic, a
single `Conv2d(C_in, C_out, 1)` projects everything. Every channel sees the same
aggregated graph signal before projection.

**SOTA reference** (CTR-GCN, ICCV 2021, 92.4% NTU60 xsub):
Channel-topology refinement -- each channel group gets its own input-dependent adjacency
refinement matrix. Different features attend to different body-part relationships.
This is the single biggest architectural gap between LAST-E and SOTA.

### 1.2 Temporal Module Limitations

**Current** (`src/models/blocks/light_gcn.py:161-226`, `MultiScaleTCN`):
Only 2 branches (dilation 1 + dilation 2, kernel 9). Missing:
- Max-pool branch for detecting temporal peaks/extremes (crucial for impulsive actions like punching)
- Identity/1x1 branch for preserving high-frequency detail through the residual path

**SOTA reference** (EfficientGCN, TPAMI 2022, 91.7% NTU60 xsub):
4-branch temporal module: {conv, dilated conv, max-pool, 1x1}. The max-pool branch
captures the frame with maximum activation per channel -- critical for actions with
sharp temporal events.

### 1.3 Training Recipe

**Current** (`configs/training/default.yaml`):
MultiStepLR with milestones [50, 65] and gamma 0.1 over 90 epochs. This creates
two aggressive 10x LR cliffs:
- Epoch 50: LR drops 0.1 -> 0.01 (shock to BN stats, premature convergence)
- Epoch 65: LR drops 0.01 -> 0.001 (25 epochs at very low LR, insufficient fine-tuning)
- Dynamic adjacency gate `sigmoid(-4)=0.018` never grows because gradients are killed by the step drops

### 1.4 Weak Regularization

Only `Dropout(0.1)` in TCN and `Dropout(0.3)` at head. No block-level stochastic
regularization (DropPath/stochastic depth), which is standard in all modern architectures
(ConvNeXt, Swin, DeiT, HDGCN).

### 1.5 StreamFusion BN Contamination

**Current** (`src/models/blocks/stream_fusion.py:45`):
Single shared `stem_bn` processes all 3 streams sequentially. Its `running_mean/var`
becomes a contaminated mixture of joint, velocity, and bone statistics after projection.
At inference time, this mismatched BN produces suboptimal normalization for every stream.

### 1.6 Simple Classification Head

**Current** (`src/models/last_e.py:220`):
`AdaptiveAvgPool2d` only. Average pooling smooths out peak activations. SOTA uses
both average and max pooling to capture complementary information -- average for
general feature strength, max for the most discriminative activation per channel.

---

## 2. Architecture Overview: LAST-E v3

### 2.1 Full Model Flow

```
Input: Dict{'joint': (B,3,T,V), 'velocity': (B,3,T,V), 'bone': (B,3,T,V)}

  StreamFusion (FIXED: per-stream stem_bn)
    |- Per-stream BN (3 independent BN2d)
    |- Shared stem Conv2d(3, C0, 1)
    |- Per-stream stem_bn (3 independent BN2d)  <-- FIX: was 1 shared
    |- Per-channel softmax blend weights (3, C0)
    Output: (B, C0, T, V)
        |
  Stage 1: CTRGCNBlock x N1, stride=1, C0 -> C0
        |
  Stage 2: CTRGCNBlock x N2, stride=2 on block-0, C0 -> C1
        |
  Stage 3: CTRGCNBlock x N3, stride=2 on block-0, C1 -> C2
        |
  Enhanced Head
    |- GAP: AdaptiveAvgPool2d(1,1)  -> (B, C2, 1, 1)
    |- GMP: AdaptiveMaxPool2d(1,1)  -> (B, C2, 1, 1)
    |- Gated fusion: gap * sigma(gate) + gmp * (1 - sigma(gate))
    |- BN1d(C2) -> Dropout(0.3) -> FC(C2, num_classes)
    Output: logits (B, num_classes)
```

### 2.2 CTRGCNBlock (Replaces LightGCNBlock)

```
Input (N, C_in, T, V)
    |
CTRLightGCNConv(C_in, C_out, G=4)       ---- CTR-GCN per-group topology refinement
    |
BatchNorm2d(C_out) + ReLU
    |
[Optional] ST_JointAtt(C_out, r=4)      ---- Factorized T+S attention (kept from v2)
    |
FreqTemporalGate(C_out)                 ---- NOVEL: frequency-domain attention
    |
MultiScaleTCN4(C_out, stride, 4 branches) -- EfficientGCN 4-branch temporal
    |
DropPath(main) + Residual(skip)          ---- Stochastic depth regularization
    |
ReLU
    |
Output (N, C_out, T/stride, V)
```

### 2.3 Variant Configurations

```
LAST-E v3 Nano:
  channels:    [24,  48,  96]
  blocks:      [2,   2,   3]
  use_st_att:  [F,   F,   T]     (attention only in stage 3 for param budget)
  num_groups:  4
  drop_path:   0.1

LAST-E v3 Small:
  channels:    [32,  64,  128]
  blocks:      [2,   3,   3]
  use_st_att:  [F,   T,   T]
  num_groups:  4
  drop_path:   0.1

LAST-E v3 Base:
  channels:    [40,  80,  160]
  blocks:      [3,   4,   4]
  use_st_att:  [T,   T,   T]
  num_groups:  4
  drop_path:   0.15

LAST-E v3 Large:
  channels:    [48,  96,  192]
  blocks:      [4,   5,   5]
  use_st_att:  [T,   T,   T]
  num_groups:  4
  drop_path:   0.2
```

---

## 3. Detailed Block Designs

### 3.1 CTRLightGCNConv — Channel-Topology Refinement GCN

**Replaces**: `DirectionalGCNConv` (`src/models/blocks/light_gcn.py:38-154`)

**Core idea** (from CTR-GCN): Instead of all channels sharing one combined adjacency,
split channels into G groups. Each group gets:
1. Its own softmax-weighted blend of K=3 physical subsets
2. Its own input-dependent topology refinement matrix (CTR-GCN innovation)
3. All groups share a single full 1x1 projection for cross-group channel mixing

**What is removed** (vs DirectionalGCNConv):
- `A_learned` (V x V = 625 static params per block, 11 blocks = 6,875 total)
- `node_proj` (C_in x embed_dim per block)
- `alpha_dyn` (C_in per block)
- Per-channel alpha over K subsets (K x C_in) -- replaced by per-group alpha (G x K)

**What is added**:
- Per-group alpha: (G, K) = (4, 3) = 12 params (vs K x C_in = 3 x 40 = 120 for base)
- Grouped Q/K projections for topology refinement: 2 x C_in x d via grouped Conv1x1
- Per-group refinement gate: G params

```python
class CTRLightGCNConv(nn.Module):
    """
    CTR-GCN-inspired GCN with per-group topology refinement.

    G groups of channels, each with:
      - Its own blend of K=3 physical subsets (alpha softmax)
      - Input-dependent topology refinement via Q/K projections
      - Gated residual addition of refinement to physical adjacency
    """
    def __init__(self, in_channels, out_channels, A_physical, num_groups=4):
        super().__init__()
        K = A_physical.shape[0]   # 3 for spatial strategy
        V = A_physical.shape[-1]  # 25 for NTU
        G = num_groups
        self.K, self.V, self.G = K, V, G
        self.in_channels = in_channels

        # K=3 physical adjacency buffers (no re-normalization, same as current)
        for k in range(K):
            A_k = A_physical[k] if A_physical.dim() == 3 else A_physical
            self.register_buffer(f'A_{k}', A_k.clone())

        # Per-group per-subset blend: (G, K). Zero init -> uniform 1/K
        self.alpha = nn.Parameter(torch.zeros(G, K))

        # Topology refinement projections (CTR-GCN core innovation)
        # d = embedding dim per group; small for efficiency
        d = max(4, in_channels // (G * 2))
        self.d = d
        # Grouped 1x1 conv: each group projects C_in/G channels to d dims
        # Params: C_in * d (due to grouping: G groups x (C_in/G) x d = C_in * d)
        self.refine_q = nn.Conv2d(in_channels, d * G, 1, groups=G, bias=False)
        self.refine_k = nn.Conv2d(in_channels, d * G, 1, groups=G, bias=False)

        # Per-group gate for refinement magnitude
        # sigmoid(0) = 0.5 -> moderate initial contribution
        self.refine_gate = nn.Parameter(torch.zeros(G))

        # Full 1x1 projection (NOT grouped) for cross-group channel mixing
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        N, C, T, V = x.shape
        G, K, d = self.G, self.K, self.d
        group_c = C // G

        # 1. Per-group blend of K physical subsets
        alpha_w = F.softmax(self.alpha, dim=1)          # (G, K)
        A_stack = torch.stack(
            [getattr(self, f'A_{k}') for k in range(K)]
        )                                                # (K, V, V)
        A_combined = torch.einsum('gk,kvw->gvw', alpha_w, A_stack)  # (G, V, V)

        # 2. Input-dependent topology refinement (CTR-GCN core)
        x_pool = x.mean(dim=2, keepdim=True)            # (N, C, 1, V)
        q = self.refine_q(x_pool).squeeze(2)             # (N, d*G, V)
        k_feat = self.refine_k(x_pool).squeeze(2)        # (N, d*G, V)
        q = F.normalize(q.view(N, G, d, V), p=2, dim=2)
        k_feat = F.normalize(k_feat.view(N, G, d, V), p=2, dim=2)
        M = torch.einsum('ngdi,ngdj->ngij', q, k_feat)  # (N, G, V, V)
        M = F.softmax(M, dim=-1)                         # row-normalize

        # 3. Final adjacency = physical + gated refinement
        gate = torch.sigmoid(self.refine_gate).view(1, G, 1, 1)  # (1, G, 1, 1)
        A_refined = A_combined.unsqueeze(0) + gate * M    # (N, G, V, V)

        # 4. Per-group graph aggregation
        x_grouped = x.view(N, G, group_c, T, V)
        x_flat = x_grouped.reshape(N * G, group_c * T, V)
        A_flat = A_refined.reshape(N * G, V, V)
        x_agg = torch.bmm(x_flat, A_flat)                # (N*G, group_c*T, V)
        x_agg = x_agg.view(N, C, T, V)

        # 5. Cross-group channel mixing via full 1x1 conv
        return self.proj(x_agg)
```

**Parameter counts per block (G=4):**

| Block type | C_in | C_out | d  | alpha | Q proj | K proj | gate | proj     | Total  | OLD Total | Delta  |
|------------|------|-------|----|-------|--------|--------|------|----------|--------|-----------|--------|
| 40 -> 40   | 40   | 40    | 5  | 12    | 200    | 200    | 4    | 1,600    | 2,016  | 2,785     | -769   |
| 40 -> 80   | 40   | 80    | 5  | 12    | 200    | 200    | 4    | 3,200    | 3,616  | 4,385     | -769   |
| 80 -> 80   | 80   | 80    | 10 | 12    | 800    | 800    | 4    | 6,400    | 8,016  | 8,945     | -929   |
| 80 -> 160  | 80   | 160   | 10 | 12    | 800    | 800    | 4    | 12,800   | 14,416 | 15,345    | -929   |
| 160 -> 160 | 160  | 160   | 20 | 12    | 3,200  | 3,200  | 4    | 25,600   | 32,016 | 33,265    | -1,249 |

**Advantages over DirectionalGCNConv:**
- G=4 independent topologies per block (vs 1 shared)
- Input-dependent refinement (like CTR-GCN) instead of static A_learned
- Fewer params per block while more expressive
- `refine_gate` at sigmoid(0)=0.5 gives meaningful initial contribution
  (vs current `alpha_dyn` at sigmoid(-4)=0.018 that never grows under MultiStepLR)

---

### 3.2 MultiScaleTCN4 — EfficientGCN-style 4-Branch Temporal

**Replaces**: `MultiScaleTCN` (`src/models/blocks/light_gcn.py:161-226`)

**Core idea** (from EfficientGCN): Split channels into C//4 groups with 4 complementary
temporal processing branches. The max-pool branch detects temporal extremes (peaks of
velocity, sudden direction changes) with zero conv params. The 1x1 branch preserves
high-frequency detail through a lightweight identity-like pathway.

```python
class MultiScaleTCN4(nn.Module):
    """
    4-branch parallel depthwise-separable TCN (EfficientGCN-style).

    Branch 1: conv k=9, dilation=1  -> 9-frame receptive field   (local)
    Branch 2: conv k=9, dilation=2  -> 17-frame receptive field  (wide)
    Branch 3: MaxPool k=3           -> temporal extremes          (peaks)
    Branch 4: Conv 1x1              -> identity-like pathway      (detail)

    Each branch operates on C//4 channels, concatenated at output.
    """
    def __init__(self, channels, stride=1):
        super().__init__()
        assert channels % 4 == 0, f"channels must be divisible by 4, got {channels}"
        q = channels // 4

        pad1 = (9 - 1) // 2       # 4, for dilation=1
        pad2 = 2 * (9 - 1) // 2   # 8, for dilation=2
        mp_pad = (3 - 1) // 2     # 1, for maxpool k=3

        # Branch 1: standard k=9 depthwise-separable
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(q), nn.ReLU(inplace=True),
            nn.Conv2d(q, q, (9,1), (stride,1), (pad1,0),
                      groups=q, bias=False),                # depthwise
            nn.Conv2d(q, q, 1, bias=False),                 # pointwise
            nn.BatchNorm2d(q),
        )

        # Branch 2: dilated k=9 depthwise-separable (d=2)
        self.branch2 = nn.Sequential(
            nn.BatchNorm2d(q), nn.ReLU(inplace=True),
            nn.Conv2d(q, q, (9,1), (stride,1), (pad2,0),
                      dilation=(2,1), groups=q, bias=False), # dilated DW
            nn.Conv2d(q, q, 1, bias=False),                  # pointwise
            nn.BatchNorm2d(q),
        )

        # Branch 3: temporal max-pool (0 conv params, captures extremes)
        self.branch3 = nn.Sequential(
            nn.BatchNorm2d(q), nn.ReLU(inplace=True),
            nn.MaxPool2d((3,1), (stride,1), (mp_pad,0)),
            nn.BatchNorm2d(q),
        )

        # Branch 4: 1x1 conv (identity-like, handles stride)
        self.branch4 = nn.Sequential(
            nn.BatchNorm2d(q), nn.ReLU(inplace=True),
            nn.Conv2d(q, q, 1, (stride,1), bias=False),
            nn.BatchNorm2d(q),
        )

        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        out = torch.cat([
            self.branch1(x1), self.branch2(x2),
            self.branch3(x3), self.branch4(x4),
        ], dim=1)
        return self.drop(out)
```

**T_out verification** (all branches produce T_out = T // stride):
- Branch 1: `floor((T + 2*4 - 1*(9-1) - 1)/s + 1) = T/s`
- Branch 2: `floor((T + 2*8 - 2*(9-1) - 1)/s + 1) = T/s`
- Branch 3: `floor((T + 2*1 - 1*(3-1) - 1)/s + 1) = T/s`
- Branch 4: `floor(T/s)` (1x1 with stride)

**Parameter counts per block (q = C//4):**

| C   | q  | Br1 (DW+PW+BN) | Br2 (DW+PW+BN) | Br3 (BN+MaxPool+BN) | Br4 (BN+1x1+BN) | New Total | OLD (2-branch) | Savings |
|-----|----|----|----|----|----|----|----|----|
| 24  | 6  | 174  | 174  | 24  | 84  | 456    | 780    | -324    |
| 32  | 8  | 264  | 264  | 32  | 128 | 688    | 1,136  | -448    |
| 40  | 10 | 370  | 370  | 40  | 180 | 960    | 1,320  | -360    |

Wait, let me recompute these more carefully.

Branch 1 for q channels:
- BN(q): 2q
- ReLU: 0
- DW Conv(q, k=9): 9q
- PW Conv(q, q): q^2
- BN(q): 2q
Total: q^2 + 13q

Branch 2: same = q^2 + 13q
Branch 3: BN(q) + MaxPool + BN(q) = 4q
Branch 4: BN(q) + Conv(q,q,1) + BN(q) = q^2 + 4q
Total: 3*q^2 + 34*q

| C   | q  | Total new     | OLD 2-br (half=C/2) | Savings |
|-----|----|----|----|----|
| 24  | 6  | 312    | 780    | -468    |
| 32  | 8  | 464    | 1,136  | -672    |
| 40  | 10 | 640    | 1,320  | -680    |
| 48  | 12 | 840    | 1,536  | -696    |
| 64  | 16 | 1,312  | 2,400  | -1,088  |
| 80  | 20 | 1,880  | 4,240  | -2,360  |
| 96  | 24 | 2,544  | 5,520  | -2,976  |
| 128 | 32 | 4,160  | 9,344  | -5,184  |
| 160 | 40 | 6,160  | 14,880 | -8,720  |
| 192 | 48 | 8,544  | 20,640 | -12,096 |

The savings are massive, especially at high C. The quadratic term dominates:
- old q^2 = (C/2)^2 x 2 = C^2/2
- new q^2 = (C/4)^2 x 3 = 3*C^2/16
- savings = C^2/2 - 3*C^2/16 = 5*C^2/16

---

### 3.3 FreqTemporalGate — Novel Frequency-Aware Temporal Attention

**Source**: Idea A from `Docs/novel_solutions.md:70-95`

**Motivation**: Human actions have distinct frequency signatures. Walking is periodic
(0.5-2 Hz), punching is impulsive (high-frequency burst), writing is fine-grained
high-frequency wrist motion. Current TCNs treat all frequencies equally -- they are
just spatial convolutions over time. No skeleton recognition paper has explored
frequency-domain gating.

**Design**: Apply `torch.fft.rfft` along the temporal dimension, learn a per-channel
frequency mask via sigmoid gating, apply mask in frequency domain, then `irfft` back.
Residual connection with zero-init gate ensures identity at initialization.

```python
class FreqTemporalGate(nn.Module):
    """
    Frequency-Aware Temporal Gate.

    Operates in frequency domain via real FFT. Each channel learns which
    frequency bands are action-discriminative. A learnable sigmoid mask
    suppresses irrelevant frequencies (e.g., high-freq noise for slow
    actions, low-freq baseline for impulsive actions).

    Params: C x max_freq_bins + 1 per block.
    For C=160, T=64 (max_freq_bins=33): 5,281 params. Zero conv ops.
    """
    def __init__(self, channels, max_freq_bins=33):
        super().__init__()
        # Learnable frequency mask: (1, C, max_freq_bins, 1)
        # Zero init -> sigmoid(0) = 0.5 (pass all frequencies equally at start)
        self.freq_mask = nn.Parameter(
            torch.zeros(1, channels, max_freq_bins, 1)
        )
        # Residual gate: zero init -> pure identity at start
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape

        # 1. Transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=2)         # (N, C, T//2+1, V) complex
        n_freq = x_freq.shape[2]                    # T//2+1

        # 2. Apply learnable frequency mask
        mask = torch.sigmoid(
            self.freq_mask[:, :, :n_freq, :]        # truncate if T varies
        )                                            # (1, C, n_freq, 1)
        x_filtered = x_freq * mask                  # element-wise scale

        # 3. Transform back to time domain
        x_back = torch.fft.irfft(x_filtered, n=T, dim=2)  # (N, C, T, V)

        # 4. Residual blend: gate=0 at init -> identity
        return x + self.gate * (x_back - x)
```

**Placement in block**: After ST_JointAtt, before MultiScaleTCN4. Rationale: frequency
filtering refines the spatially-aggregated features before temporal convolution, allowing
the TCN to focus on the frequency bands that matter per channel.

**Params per block:**

| C   | Bins (T=64) | freq_mask | gate | Total |
|-----|-------------|-----------|------|-------|
| 24  | 33          | 792       | 1    | 793   |
| 32  | 33          | 1,056     | 1    | 1,057 |
| 40  | 33          | 1,320     | 1    | 1,321 |
| 48  | 33          | 1,584     | 1    | 1,585 |
| 64  | 33          | 2,112     | 1    | 2,113 |
| 80  | 33          | 2,640     | 1    | 2,641 |
| 96  | 33          | 3,168     | 1    | 3,169 |
| 128 | 33          | 4,224     | 1    | 4,225 |
| 160 | 33          | 5,280     | 1    | 5,281 |
| 192 | 33          | 6,336     | 1    | 6,337 |

**Why this is novel**: No skeleton recognition paper (ST-GCN, CTR-GCN, EfficientGCN,
InfoGCN, HD-GCN, SkateFormer, BlockGCN) operates in the frequency domain. DCT/FFT is
standard in signal processing and image compression but unexplored for temporal skeleton
features. The frequency signature of an action is a fundamentally different axis of
discriminative information than spatial topology or temporal convolution.

---

### 3.4 DropPath — Stochastic Depth Regularization

Standard module used in ConvNeXt, Swin Transformer, DeiT. During training, randomly
drops the entire main branch output for a block, forcing gradient flow through the
skip connection. At eval, identity.

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().add_(keep)
        return x * mask / keep
```

**Applied on the main branch** (not the skip connection):
```python
# In CTRGCNBlock.forward():
x = self.drop_path(x) + self.residual_path(res)
```

**Drop rates** (linearly ramped from 0 at first block to max at last block):
- Nano/Small: max 0.1
- Base: max 0.15
- Large: max 0.2

---

### 3.5 Enhanced Classification Head

**Replaces**: `GAP -> Dropout -> FC` in `src/models/last_e.py:219-226`

```python
# In LAST_E.__init__:
self.pool_gate = nn.Parameter(torch.zeros(1, C2, 1, 1))
self.head_bn   = nn.BatchNorm1d(C2)
self.drop      = nn.Dropout(dropout)
self.fc        = nn.Linear(C2, num_classes)

# In LAST_E.forward:
gap = F.adaptive_avg_pool2d(out, (1, 1))    # (B, C2, 1, 1)
gmp = F.adaptive_max_pool2d(out, (1, 1))    # (B, C2, 1, 1)
gate = torch.sigmoid(self.pool_gate)         # (1, C2, 1, 1)
pooled = gap * gate + gmp * (1.0 - gate)    # per-channel blend
pooled = pooled.view(pooled.size(0), -1)     # (B, C2)
pooled = self.head_bn(pooled)
pooled = self.drop(pooled)
logits = self.fc(pooled)
```

At init: sigmoid(0)=0.5, so equal blend of avg and max pool. Each channel learns during
training whether average or max is more informative for it.

Extra params: C2 (pool_gate) + 2*C2 (head_bn) = 3*C2. Negligible (~480 for base).

---

### 3.6 StreamFusion Fix

**File**: `src/models/blocks/stream_fusion.py:45,78`

Replace single shared `stem_bn` with per-stream ModuleList:

```python
# OLD (line 45):
self.stem_bn = nn.BatchNorm2d(out_channels)

# NEW:
self.stem_bn = nn.ModuleList([
    nn.BatchNorm2d(out_channels) for _ in range(num_streams)
])

# OLD forward (line 78):
x = self.stem_bn(x)

# NEW forward:
x = self.stem_bn[i](x)   # use stream-specific BN
```

Extra params: 2 more BN modules = 4 * C0. For base: +160 params.

---

## 4. Full Parameter Budget

### 4.1 Per-Block Breakdown (Base Variant: [40, 80, 160])

**Stage 1 block (40 -> 40, stride=1, with ST_JointAtt):**

| Component          | Params  |
|--------------------|---------|
| CTRLightGCNConv    | 2,016   |
| gcn_bn             | 80      |
| ST_JointAtt(40)    | 1,680   |
| FreqTemporalGate   | 1,321   |
| MultiScaleTCN4(40) | 640     |
| Residual (Identity)| 0       |
| **Block total**    | **5,737** |

**Stage 2 block 0 (40 -> 80, stride=2, with ST_JointAtt):**

| Component          | Params  |
|--------------------|---------|
| CTRLightGCNConv    | 3,616   |
| gcn_bn             | 160     |
| ST_JointAtt(80)    | 6,560   |
| FreqTemporalGate   | 2,641   |
| MultiScaleTCN4(80) | 1,880   |
| Residual (conv+bn) | 3,360   |
| **Block total**    | **18,217** |

**Stage 2 blocks 1-3 (80 -> 80, stride=1):**

| Component          | Params  |
|--------------------|---------|
| CTRLightGCNConv    | 8,016   |
| gcn_bn             | 160     |
| ST_JointAtt(80)    | 6,560   |
| FreqTemporalGate   | 2,641   |
| MultiScaleTCN4(80) | 1,880   |
| **Block total**    | **19,257** |

**Stage 3 block 0 (80 -> 160, stride=2):**

| Component           | Params  |
|---------------------|---------|
| CTRLightGCNConv     | 14,416  |
| gcn_bn              | 320     |
| ST_JointAtt(160)    | 25,920  |
| FreqTemporalGate    | 5,281   |
| MultiScaleTCN4(160) | 6,160   |
| Residual (conv+bn)  | 13,120  |
| **Block total**     | **65,217** |

**Stage 3 blocks 1-3 (160 -> 160, stride=1):**

| Component           | Params  |
|---------------------|---------|
| CTRLightGCNConv     | 32,016  |
| gcn_bn              | 320     |
| ST_JointAtt(160)    | 25,920  |
| FreqTemporalGate    | 5,281   |
| MultiScaleTCN4(160) | 6,160   |
| **Block total**     | **69,697** |

### 4.2 Full Model Totals (All Variants)

| Section    | Nano   | Small  | Base    | Large   |
|------------|--------|--------|---------|---------|
| Fusion     | 306    | 402    | 498     | 594     |
| Stage 1    | 3,874  | 5,762  | 17,211  | 31,268  |
| Stage 2    | 10,546 | 38,099 | 75,988  | 132,709 |
| Stage 3    | 79,011 | 134,515| 274,308 | 486,517 |
| Head       | 6,108  | 8,124  | 10,140  | 12,156  |
| **Total**  |**99,845**|**186,902**|**378,145**|**663,244**|

### 4.3 Comparison vs Current LAST-E

| Variant | Current (v2) | New (v3) | Delta    | Under 1M? |
|---------|--------------|----------|----------|-----------|
| Nano    | ~99K         | ~100K    | +1K (+1%)| Yes (100K)|
| Small   | ~187K        | ~187K    | +0K (0%) | Yes (187K)|
| Base    | ~399K        | ~378K    | -21K (-5%)| Yes (378K)|
| Large   | ~663K        | ~663K    | +0K (0%) | Yes (663K)|

The architecture redesign is **roughly parameter-neutral**. The savings from
removing A_learned + node_proj + alpha_dyn (GCN simplification) and from the 4-branch
TCN (C//4 quadratic savings) are reinvested into:
- CTR-GCN-style per-group topology refinement
- Frequency-Aware Temporal Gate
- Enhanced head

### 4.4 Without FreqTemporalGate (Ablation Baseline)

| Variant | With FreqGate | Without | FreqGate cost |
|---------|---------------|---------|---------------|
| Nano    | ~100K         | ~86K    | ~14K          |
| Small   | ~187K         | ~166K   | ~21K          |
| Base    | ~378K         | ~342K   | ~36K          |
| Large   | ~663K         | ~609K   | ~54K          |

Without FreqGate, all variants are ~14% SMALLER than current. This provides
headroom for future novel additions (e.g., Action-Prototype Graph from Idea B).

---

## 5. Training Recipe

### 5.1 Config Changes

**File**: `configs/training/default.yaml`

| Parameter       | Current (v2)       | New (v3)            | Rationale |
|-----------------|--------------------|---------------------|-----------|
| scheduler       | multistep_warmup   | **cosine_warmup**   | Smoother LR decay, no 10x cliff at ep50. Already implemented in trainer.py:178-194. |
| epochs          | 90                 | **140**             | 55% longer. Cosine needs room to decay. CTR topology refinement needs time to converge. |
| warmup_epochs   | 5                  | **10**              | Longer warmup stabilizes grouped topology init. |
| warmup_start_lr | 0.01               | **0.001**           | Gentler start prevents early instability in K-group attention. |
| min_lr          | 0.0001             | **0.00001**         | Lower floor for extended fine-tuning tail. |
| gradient_clip   | 2.0                | **1.5**             | Tighter clipping for cosine stability. |

**Unchanged**: lr=0.1, SGD, momentum=0.9, nesterov=true, weight_decay=0.0004,
label_smoothing=0.1, batch_size=64, input_frames=64, use_amp=true.

### 5.2 LR Curve Comparison

```
Current (MultiStepLR):
  ep 0-5:   warmup  0.01 -> 0.10
  ep 5-50:  flat    0.10
  ep 50:    drop    0.10 -> 0.01  (10x cliff)
  ep 65:    drop    0.01 -> 0.001 (10x cliff)
  ep 65-90: flat    0.001

  Problem: 45 epochs at peak LR, then 25 at very low LR. Premature convergence.

New (Cosine Annealing):
  ep 0-10:   warmup   0.001 -> 0.10
  ep 10-140: cosine   0.10  -> 0.00001

  Benefit: LR stays above 0.01 until ~epoch 100, giving topology refinement and
  frequency gates ample time to train. Smooth decay avoids BN stat shock.
```

### 5.3 Cosine Annealing Implementation

The `cosine_warmup` scheduler is **already fully implemented** in
`src/training/trainer.py:178-194`. It uses `LinearLR` warmup chained with
`CosineAnnealingLR` via `SequentialLR`. **No trainer code changes needed** for the
scheduler -- just the YAML config change.

### 5.4 Weight Decay Exclusions Update

**File**: `src/training/trainer.py:139-146`

Add to the no_decay condition:
```python
if (
    'bias' in name
    or 'bn' in name
    or 'norm' in name
    or 'alpha' in name          # CTRLightGCNConv alpha + ST_JointAtt alpha
    or 'A_learned' in name      # kept for LAST-v2 backward compat
    or 'node_proj' in name      # kept for LAST-v2 backward compat
    or 'refine_gate' in name    # NEW: CTRLightGCNConv group gate
    or 'pool_gate' in name      # NEW: head gated pooling
    or 'freq_mask' in name      # NEW: FreqTemporalGate mask
    or 'gate' in name           # NEW: FreqTemporalGate residual gate
):
    no_decay.append(param)
```

---

## 6. Files to Create / Modify

### 6.1 NEW: `src/models/blocks/ctr_gcn_block.py` (~250 lines)

Contains all new modules:
- `DropPath` (~15 lines)
- `CTRLightGCNConv` (~70 lines)
- `FreqTemporalGate` (~35 lines)
- `MultiScaleTCN4` (~60 lines)
- `CTRGCNBlock` (~55 lines)

### 6.2 MODIFY: `src/models/last_e.py`

| Lines   | Change |
|---------|--------|
| 31      | Import `CTRGCNBlock` from `.blocks.ctr_gcn_block` instead of `LightGCNBlock` |
| 39-44   | Update `MODEL_VARIANTS_E` dict: add `num_groups`, `drop_path` keys |
| 77-80   | Extract `num_groups` and `drop_path` from cfg |
| 97-122  | Pass `num_groups` to `_make_stage`, compute per-block drop rates |
| 124-126 | Replace head: add `pool_gate`, `head_bn`, keep `drop` and `fc` |
| 143-173 | Update `_make_stage`: use `CTRGCNBlock`, pass `num_groups`, `drop_path_rate` per block |
| 219-226 | Update forward: GAP+GMP gated fusion, `head_bn`, dropout, fc |

### 6.3 MODIFY: `src/models/blocks/stream_fusion.py`

| Lines | Change |
|-------|--------|
| 45    | `self.stem_bn = nn.BatchNorm2d(...)` -> `nn.ModuleList([...] for _ in range(num_streams))` |
| 78    | `self.stem_bn(x)` -> `self.stem_bn[i](x)` |

### 6.4 MODIFY: `src/training/trainer.py`

| Lines   | Change |
|---------|--------|
| 139-146 | Add `'refine_gate'`, `'pool_gate'`, `'freq_mask'` to no_decay conditions |

### 6.5 MODIFY: `configs/training/default.yaml`

| Lines | Change |
|-------|--------|
| 22    | `scheduler: "cosine_warmup"` |
| 23    | `warmup_epochs: 10` |
| 24    | `warmup_start_lr: 0.001` |
| 27    | `min_lr: 0.00001` |
| 30    | `epochs: 140` |
| 32    | `gradient_clip: 1.5` |

### 6.6 MODIFY: `configs/model/last_e_*.yaml` (4 files)

Add to each:
```yaml
  num_groups: 4
  drop_path_rate: 0.1   # nano/small: 0.1, base: 0.15, large: 0.2
```

### 6.7 MODIFY: `tests/test_model_integration.py`

- Update param count assertion bounds for all 4 variants
- Add test for CTRLightGCNConv (G groups produce different topologies)
- Add test for MultiScaleTCN4 (all 4 branches produce correct T_out)
- Add test for FreqTemporalGate (identity at init, correct FFT shapes)
- Add test for DropPath (stochastic in train, identity in eval)
- Add test for enhanced head (pool_gate, head_bn)
- Add test for StreamFusion per-stream stem_bn

---

## 7. Verification Plan

### Phase 1: Unit Tests (before any training)

1. Forward pass shape check for all 4 variants with MIB dict input -> (B, 60)
2. Param count verification against Section 4 estimates (within 5%)
3. Backward pass gradient flow: no NaN gradients
4. CTRLightGCNConv:
   - G=4 groups produce 4 different A_refined matrices at init
   - alpha softmax gives uniform 1/K per group at init
   - refine_gate sigmoid gives 0.5 at init
5. MultiScaleTCN4:
   - Test all channel widths [24,32,40,48,64,80,96,128,160,192] x stride [1,2]
   - Verify T_out = T // stride for all 4 branches
6. FreqTemporalGate:
   - gate=0 at init -> output == input (identity)
   - Works for variable T (T=64, T=32, T=16)
   - Output shape matches input shape
7. DropPath: stochastic in train mode, identity in eval mode
8. StreamFusion: 3 separate stem_bn modules, different running stats after forward
9. Head: pool_gate shape correct, gated blend of GAP and GMP

### Phase 2: Smoke Training (5 epochs, nano variant)

1. Loss decreases monotonically
2. No NaN/Inf in outputs or loss
3. LR curve: warmup 0.001 -> 0.1 over 10 epochs, then cosine
4. Checkpoint save/load round-trip with new model state_dict

### Phase 3: Ablation Training (base variant, NTU60 xsub, full 140 epochs)

| Run | Description | Purpose |
|-----|-------------|---------|
| A   | Current arch + current recipe | Baseline |
| B   | Current arch + new recipe (cosine 140ep) | Isolate training recipe gain |
| C   | New arch (no FreqGate) + current recipe | Isolate architecture gain |
| D   | New arch (no FreqGate) + new recipe | Combined arch + recipe |
| E   | New arch (with FreqGate) + new recipe | Full v3 |

### Phase 4: Full Training (all 4 variants, 140 epochs)

Train all variants, compare against:
- Current LAST-E v2 results (baseline)
- EfficientGCN-B0 (150K, 88.3%), B2 (1.1M, 90.2%), B4 (2M, 91.7%)
- CTR-GCN (5M, 92.4%), InfoGCN (5M, 92.7%)

### Expected Accuracy Targets (NTU60 xsub)

| Variant | Current est. | v3 Target | vs EfficientGCN |
|---------|-------------|-----------|-----------------|
| Nano    | 85-87%      | 88-90%    | >B0 (88.3%) at 100K vs 150K |
| Small   | 87-89%      | 90-91%    | >B2 (90.2%) at 187K vs 1.1M |
| Base    | 89-91%      | 91-93%    | >B4 (91.7%) at 378K vs 2M |
| Large   | 91-92%      | 92-94%    | Near CTR-GCN (92.4%) at 663K vs 5M |

---

## 8. SOTA Alignment Summary

| SOTA Technique | Source | Adopted in LAST-E v3 |
|----------------|--------|----------------------|
| Channel-topology refinement | CTR-GCN (ICCV 2021) | CTRLightGCNConv with G=4 groups |
| 4-branch temporal (max-pool + 1x1) | EfficientGCN (TPAMI 2022) | MultiScaleTCN4 |
| Stochastic depth / DropPath | ConvNeXt, Swin, DeiT | DropPath with linear ramp |
| GAP + GMP fusion | EfficientNet, ResNeSt | Gated pool_gate blend |
| Cosine annealing LR | Universal modern training | cosine_warmup (already in codebase) |
| Frequency-domain attention | **Novel (unexplored in skeleton)** | FreqTemporalGate |

| LAST-E v2 Innovation | Kept in v3? |
|----------------------|-------------|
| Per-stream BN | Yes (stream_fusion.py) |
| Per-channel stream weights | Yes (stream_fusion.py) |
| Early StreamFusion (backbone 1x) | Yes |
| K=3 physical subsets | Yes (in CTRLightGCNConv) |
| ST_JointAtt zero-init gate | Yes (unchanged) |
| Variant-progressive attention | Yes (nano: stage3 only) |
| Per-stream stem_bn fix | **Improved** (3 separate BNs) |
