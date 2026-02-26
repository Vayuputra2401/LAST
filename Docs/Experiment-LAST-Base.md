# Experiment: LAST-Base (High-Accuracy Research Model)

No param budget. Goal: beat HI-GCN (93.3% NTU-60 xsub) with a novel combination of
SOTA techniques + our original contributions (Ideas A, B, D from `novel_solutions.md`).

---

## 1. SOTA Landscape (What We're Beating)

| Model | Year | NTU-60 xsub | NTU-120 xsub | Params | Key Innovation |
|-------|------|-------------|-------------|--------|----------------|
| HI-GCN | 2025 | **93.3%** | **90.3%** | ~3-4M | Intertwined context graph + shifted window temporal transformer |
| HD-GCN | 2023 | 93.4% (6-ens) | 90.1% | ~3M | Hierarchical body decomposition |
| SkateFormer | 2024 | ~93.0% | ~89.8% | ~4M | 4-type partition attention (near/far × local/global) |
| InfoGCN | 2022 | 93.0% | 89.8% | 1.5M | Information bottleneck + attention graph |
| BlockGCN | 2024 | ~92.8% | ~90.0% | <2M | Persistent homology topology |
| CTR-GCN | 2021 | 92.4% | 88.9% | 1.7M | Channel-topology refinement |

---

## 2. Architecture: LAST-Base Block

### What We Take From Each SOTA

| Source | What We Take | Their Limitation We Solve |
|--------|-------------|--------------------------|
| HI-GCN | Cross-temporal topology refinement | No class conditioning → we add Action-Prototype Graph (Idea B) |
| SkateFormer | 4-type partition attention | No body-region hierarchy → we add HBRA (Idea D) |
| HD-GCN | Hierarchical body decomposition | Fixed regions + expensive GCN within → we use learnable regions |
| InfoGCN | Information bottleneck loss | No frequency processing → we add FreqTemporalGate (Idea A) |
| CTR-GCN | Channel-topology refinement | No temporal awareness in topology → we add cross-temporal context |

### Block Design

```
LAST-Base Block:
  input (B, C, T, V)
    │
    ├── 1. CrossTemporalPrototypeGCN ★ NOVEL
    │     │
    │     ├─ Temporal context: gather x_{t-s}, x_t, x_{t+s} at scales s ∈ {1, 3}
    │     │  → context = Concat([x_{t-1}, x_t, x_{t+1}, x_{t-3}, x_{t+3}])
    │     │  → ΔA = MLP(GlobalPool(context)) reshaped to (V, V)
    │     │
    │     ├─ Channel-topology refinement (CTR-GCN style):
    │     │  → Split channels into G=4 groups
    │     │  → Each group gets its own adjacency: A_g = A_shared + ΔA + A_group_specific
    │     │
    │     ├─ Action-Prototype Graph (Idea B):
    │     │  → K=15 learnable prototype adjacencies: {A_proto_1, ..., A_proto_15}
    │     │  → Per-sample blend: w = softmax(GlobalPool(x) @ P.T)    (B, K)
    │     │  → A_proto = Σ_k w_k × A_proto_k                        (B, V, V)
    │     │
    │     ├─ Final adjacency per sample per group:
    │     │  → A_final_g = A_physical + ΔA_temporal + A_group + A_proto
    │     │  → h = Σ_g Conv1x1_g(A_final_g @ x)   (CTR-GCN aggregation)
    │     │
    │     → This is the strongest spatial GCN in the literature:
    │       temporal-aware + channel-specific + class-conditioned
    │
    ├── BN + GELU
    │
    ├── 2. FreqTemporalGate ★ NOVEL (Idea A — full adaptive version)
    │     │
    │     ├─ DCT along T: x_freq = DCT(x)                     (B, C, F, V)
    │     ├─ Learnable frequency attention: mask = σ(MLP(x_freq.mean(V)))
    │     ├─ x_gated = x_freq * mask                           (per-sample gating)
    │     ├─ IDCT: x_back = IDCT(x_gated)                     (B, C, T, V)
    │     └─ Residual: output = x + x_back
    │
    │     → Completely unexplored in ALL skeleton SOTA models
    │     → Separates periodic (walking) from impulsive (punching) actions
    │
    ├── 3. PartitionedTemporalAttention (from SkateFormer, adapted)
    │     │
    │     ├─ Define partitions:
    │     │  Near joints:  1-hop neighbors in skeleton graph
    │     │  Far joints:   ≥2-hop neighbors
    │     │  Near frames:  |Δt| ≤ 3
    │     │  Far frames:   |Δt| > 3
    │     │
    │     ├─ 4 attention heads, one per partition type:
    │     │  Head 1: Near-J × Near-T  → local articulation (finger wiggle, eye blink)
    │     │  Head 2: Near-J × Far-T   → joint trajectory (wrist path across time)
    │     │  Head 3: Far-J  × Near-T  → body coordination (arm↔leg at same moment)
    │     │  Head 4: Far-J  × Far-T   → global action (start pose vs end pose)
    │     │
    │     ├─ Each head: Q,K,V projections + scaled dot-product attention
    │     │  Dimension per head: C // 4
    │     │
    │     └─ Concat all heads → Linear projection → residual
    │
    ├── 4. HierarchicalBodyRegion (adapted from Idea D / HD-GCN)
    │     │
    │     ├─ 5 body regions (fixed, anatomical):
    │     │  Left arm [4,5,6,7,21,22], Right arm [8,9,10,11,23,24]
    │     │  Left leg [12,13,14,15], Right leg [16,17,18,19]
    │     │  Torso [0,1,2,3,20]
    │     │
    │     ├─ Step 1 — Intra-region attention:
    │     │  5 regions × ~5² = 125 attention pairs (vs 625 full V²)
    │     │  Lightweight: Q,K,V with C_head = C // 8
    │     │
    │     ├─ Step 2 — Region summary tokens:
    │     │  Per-region learnable query attends to all joints in region → 1 token per region
    │     │  → 5 summary tokens of dim C
    │     │
    │     ├─ Step 3 — Inter-region attention:
    │     │  5 × 5 = 25 pairs — negligible cost
    │     │  Learns "arm↔leg" for kicking, "hand↔torso" for hugging
    │     │
    │     └─ Step 4 — Broadcast + residual:
    │        Region tokens → expand back to per-joint via learned weights → add residually
    │
    ├── DropPath (stochastic depth, linearly increasing across stages)
    │
    └── Residual connection
```

### Full Model Architecture

```
Input: Dict{'joint': (B,3,T,V,M), 'velocity': (B,3,T,V,M),
            'bone': (B,3,T,V,M), 'bone_velocity': (B,3,T,V,M)}  — 4 streams

For EACH stream (trained independently, ensembled at inference):

  StreamFusion (from LAST-E):
    |- Per-stream DataBN
    |- Shared stem Conv2d(3, C0, 1)
    Output: (B, C0, T, V)

  Stage 1: LAST-Base Block × 3
    |- C = 128, stride = 1
    |- DropPath rate: 0.0 → 0.05

  Stage 2: LAST-Base Block × 4
    |- C = 256, stride = 2 (temporal downsampled 2×)
    |- DropPath rate: 0.05 → 0.15

  Stage 3: LAST-Base Block × 3
    |- C = 384, stride = 2 (temporal downsampled 2×)
    |- DropPath rate: 0.15 → 0.25

  Gated Head:
    |- GAP(T,V) + GMP(T,V) → learnable gate blend
    |- BN1d(C) → Dropout(0.3) → FC(384, num_classes)

  IB Loss (from InfoGCN):
    |- Applied to stage 3 output before head
    |- Maximizes I(z; y) while minimizing I(z; x)

  Output: logits (B, num_classes)

Ensemble (inference only):
  final_logits = mean([softmax(logits_joint), softmax(logits_bone),
                       softmax(logits_joint_vel), softmax(logits_bone_vel)])
```

---

## 3. Detailed Module Specifications

### 3A. CrossTemporalPrototypeGCN

```python
class CrossTemporalPrototypeGCN(nn.Module):
    def __init__(self, C_in, C_out, V=25, K_proto=15, G=4, temporal_scales=[1, 3]):
        # Physical adjacency (fixed)
        self.A_physical = ...  # (V, V) from skeleton graph, D^{-1/2} A D^{-1/2}

        # Temporal context MLP → topology correction
        # Input: concat of x at multiple temporal offsets → (B, C*(2*len(scales)+1), V)
        context_dim = C_in * (2 * len(temporal_scales) + 1)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(context_dim, V * V // 4),
            nn.ReLU(),
            nn.Linear(V * V // 4, V * V),
        )  # Outputs ΔA (V, V) per sample

        # Channel-topology refinement (CTR-GCN style)
        self.group_convs = nn.ModuleList([
            nn.Conv2d(C_in, C_out // G, 1) for _ in range(G)
        ])
        self.A_group = nn.ParameterList([
            nn.Parameter(torch.zeros(V, V)) for _ in range(G)
        ])

        # Action-Prototype Graph (Idea B)
        self.prototypes = nn.Parameter(torch.randn(K_proto, C_in))  # prototype features
        self.A_proto = nn.Parameter(torch.zeros(K_proto, V, V))     # prototype adjacencies
        # Init: A_proto slightly different per prototype (uniform + small noise)

    def forward(self, x):
        B, C, T, V = x.shape

        # 1. Gather temporal context
        contexts = [x]
        for s in self.temporal_scales:
            contexts.append(torch.roll(x, shifts=s, dims=2))    # x_{t-s}
            contexts.append(torch.roll(x, shifts=-s, dims=2))   # x_{t+s}
        ctx = torch.cat(contexts, dim=1)  # (B, C*(2S+1), T, V)

        # 2. Temporal topology correction
        ctx_pool = ctx.mean(dim=2)  # (B, C*(2S+1), V) → pool over T
        delta_A = self.temporal_mlp(ctx_pool.transpose(1,2).reshape(B, V, -1))
        delta_A = delta_A.reshape(B, V, V)  # per-sample topology correction

        # 3. Action-Prototype blending
        x_pool = x.mean(dim=[2, 3])  # (B, C)
        w = F.softmax(x_pool @ self.prototypes.T, dim=1)  # (B, K)
        A_proto = torch.einsum('bk,kvw->bvw', w, self.A_proto)  # (B, V, V)

        # 4. Channel-group GCN with combined adjacency
        outs = []
        for g in range(self.G):
            A_g = self.A_physical + delta_A + self.A_group[g] + A_proto  # (B, V, V)
            A_g = F.softmax(A_g, dim=-1)  # row-normalize
            h = torch.einsum('bvw,bctw->bctv', A_g, x)
            outs.append(self.group_convs[g](h))
        return torch.cat(outs, dim=1)  # (B, C_out, T, V)
```

### 3B. FreqTemporalGate (Idea A — full adaptive)

```python
class FreqTemporalGate(nn.Module):
    def __init__(self, channels, T):
        # DCT basis (frozen)
        dct = scipy.fft.dct(np.eye(T), type=2, norm='ortho')
        self.register_buffer('dct_matrix', torch.tensor(dct, dtype=torch.float32))

        # Per-sample frequency attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, T),  # per-frequency-bin weight
        )  # Input: pooled features → output: frequency mask

    def forward(self, x):
        B, C, T, V = x.shape
        x_freq = torch.matmul(x.permute(0,1,3,2), self.dct_matrix).permute(0,1,3,2)
        # x_freq: (B, C, T_freq, V)

        # Per-sample frequency attention
        pooled = x_freq.mean(dim=[2, 3])  # (B, C)
        mask = torch.sigmoid(self.mlp(pooled))  # (B, T)
        mask = mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, T, 1)

        x_gated = x_freq * mask
        x_back = torch.matmul(x_gated.permute(0,1,3,2), self.dct_matrix.T).permute(0,1,3,2)
        return x + x_back  # residual
```

### 3C. PartitionedTemporalAttention

```python
class PartitionedTemporalAttention(nn.Module):
    def __init__(self, C, V=25, A_physical=None, near_hop=1, near_frames=3, num_heads=4):
        self.head_dim = C // num_heads
        self.num_heads = num_heads

        # Pre-compute partition masks
        # near_joint_mask[v1, v2] = 1 if hop_distance(v1, v2) <= near_hop
        # far_joint_mask = 1 - near_joint_mask
        # near_frame_mask[t1, t2] = 1 if |t1 - t2| <= near_frames
        # far_frame_mask = 1 - near_frame_mask

        # QKV projections — one per head (each head = one partition type)
        self.qkv = nn.ModuleList([
            nn.Linear(self.head_dim, 3 * self.head_dim) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        B, C, T, V = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, T * V, C)  # (B, TV, C)

        heads = []
        for h, (qkv_layer, mask) in enumerate(zip(self.qkv, self.partition_masks)):
            x_h = x_flat[:, :, h*self.head_dim:(h+1)*self.head_dim]
            q, k, v = qkv_layer(x_h).chunk(3, dim=-1)
            attn = (q @ k.transpose(-1, -2)) / sqrt(self.head_dim)
            attn = attn.masked_fill(~mask, float('-inf'))  # apply partition mask
            attn = F.softmax(attn, dim=-1)
            heads.append(attn @ v)

        out = torch.cat(heads, dim=-1)  # (B, TV, C)
        out = self.proj(out)
        return out.reshape(B, T, V, C).permute(0, 3, 1, 2) + x  # residual
```

---

## 4. Param Budget

| Component | Per block (C=256, V=25) | Notes |
|-----------|------------------------|-------|
| CrossTemporalPrototypeGCN | ~210K | temporal MLP(~40K) + 4 group convs(~65K each) + prototypes(~10K) |
| FreqTemporalGate | ~18K | MLP(256→64→64) + frozen DCT (0 params) |
| PartitionedTemporalAttention | ~133K | 4 heads × QKV(64→192) + output proj(256→256) |
| HierarchicalBodyRegion | ~55K | intra(~30K) + region tokens(~5K) + inter(~5K) + broadcast(~15K) |
| BN + DropPath | ~2K | |
| **Per block total** | **~418K** | |

```
Stage 1: 3 blocks × 130K (C=128)   =   390K
Stage 2: 4 blocks × 418K (C=256)   = 1,672K
Stage 3: 3 blocks × 700K (C=384)   = 2,100K
StreamFusion + Stem:                ≈     5K
Head:                               ≈    25K
Prototype graphs (shared):         ≈    10K
Joint/frame embeddings:            ≈     3K
─────────────────────────────────────────────
Single stream total:                ≈  4.2M
4-stream ensemble:                  ≈ 16.8M
```

---

## 5. Target Accuracy

| Config | NTU-60 xsub | NTU-120 xsub |
|--------|-------------|-------------|
| LAST-Base single stream | 91.5–92.5% | 88–89% |
| LAST-Base 4-stream ensemble | **93.5–94%** | **90–91%** |
| Current SOTA (HI-GCN) | 93.3% | 90.3% |

---

## 6. Why This Beats Current SOTA

| Technique | HI-GCN | HD-GCN | SkateFormer | InfoGCN | **LAST-Base** |
|-----------|--------|--------|-------------|---------|---------------|
| Cross-temporal topology | ✅ | ❌ | ❌ | ❌ | ✅ |
| Channel-topology groups | ❌ | ❌ | ❌ | ❌ | ✅ (CTR-GCN) |
| Class-conditioned graph | ❌ | ❌ | ❌ | ❌ | ✅ **(Idea B)** |
| Frequency-domain gate | ❌ | ❌ | ❌ | ❌ | ✅ **(Idea A)** |
| Partitioned attention | ❌ | ❌ | ✅ | ❌ | ✅ |
| Body-region hierarchy | ❌ | ✅ | ❌ | ❌ | ✅ **(Idea D)** |
| Info bottleneck loss | ❌ | ❌ | ❌ | ✅ | ✅ |
| Multi-scale temporal | ✅ (shifted window) | ❌ | ✅ (partitions) | ❌ | ✅ (both) |

**LAST-Base is the ONLY model with all 8 techniques.** Each prior model has 1–2 of these.

---

## 7. 4-Stream Ensemble Details

| Stream | Input | Computation |
|--------|-------|-------------|
| Joint | (B, 3, T, V, M) raw coordinates | Primary spatial information |
| Bone | child_joint - parent_joint per bone pair | Limb direction + length |
| Joint velocity | J[t+1] - J[t] | Motion dynamics |
| Bone velocity | B[t+1] - B[t] | Limb angular velocity |

Each stream trained independently (same architecture, different weights).
At inference: `final = mean(softmax(stream_i_logits))` across 4 streams.

HD-GCN achieved 93.4% with 6-stream ensemble (adding joint-motion and bone-motion).
We use 4 streams (standard) for cleaner comparison.

---

## 8. Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `src/models/last_base.py` | Full LAST-Base model | TODO |
| `src/models/blocks/cross_temporal_gcn.py` | CrossTemporalPrototypeGCN | TODO |
| `src/models/blocks/freq_temporal_gate.py` | FreqTemporalGate (Idea A) | TODO |
| `src/models/blocks/partitioned_attention.py` | PartitionedTemporalAttention | TODO |
| `src/models/blocks/hierarchical_body_region.py` | HBRA (Idea D) | TODO |
| `src/models/blocks/action_prototype_graph.py` | APG (Idea B) | TODO |
| `tests/test_last_base.py` | Integration tests | TODO |
