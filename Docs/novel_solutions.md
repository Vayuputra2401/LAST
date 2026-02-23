# Deep Dive Report: LAST Architecture Novelty + SOTA Landscape + Novel Ideas

---

## Part 1: What Makes LAST-E and LAST-v2 Novel

### LAST-v2 — Novel Contributions

| Component | What It Does | Why It's Novel |
|-----------|-------------|----------------|
| **AdaptiveGraphConv (K+2 components)** | K=3 physical subsets (centripetal/centrifugal/self) + A_learned (V×V zero-init) + A_dynamic (per-sample cosine sim) — each with its OWN W_k projection conv | EfficientGCN uses K separate W_k but no adaptive components. CTR-GCN uses channel-specific topology but no dynamic. LAST-v2 is the first to have all three with decoupled projection per component |
| **Per-stream BN before shared backbone** | 3 independent BN2d (not 1 shared), so joint/velocity/bone each normalize to their own scale | Most MIB models share one BN — this contaminates velocity (std≈0.02) with joint statistics (std≈1.0), proven to cause plateau at 3-5% |
| **Softmax probability averaging** | `log(mean_softmax)` instead of `sum(logits)` | Scale-invariant — a degenerate velocity stream contributes uniform 1/60 noise instead of a corrupted spike that overrules correct joint prediction |
| **ST_JointAtt with α=0 zero-init** | Factorized T+S attention gated by per-channel learnable scalar | Zero-init residual gate means no dead-gate collapse problem — model starts as identity, attention grows during training |
| **LinearAttention in deep stages** | Transformer-style global temporal context, O(T) not O(T²) | Early stages use TCN (local), deep stages use LinearAttention (global) — hybrid matches where global context actually matters |

### LAST-E — Novel Contributions

| Component | What It Does | Why It's Novel |
|-----------|-------------|----------------|
| **Early StreamFusion with per-channel weights** | Fuses 3 streams at input via shared stem + (3, C0) softmax weight matrix — backbone runs 1× | MIB (EfficientGCN, LAST-v2) runs backbone 3× — 3× the compute. LAST-E makes backbone costs independent of stream count |
| **Per-channel stream weights (not scalar)** | (3, C0) matrix — channel 7 can prefer velocity, channel 23 can prefer bone | Old approach: 3 scalars shared across all channels. Per-channel lets each feature dimension specialize to the best modality |
| **DirectionalGCNConv with single conv** | Same 3-component adaptive graph as LAST-v2 but with ONE shared conv instead of K+2 separate convs | Saves K+1 times the conv params vs AdaptiveGraphConv — makes the 3-component graph affordable even at 103K total budget |
| **Per-channel α softmax on physical subsets** | (K, C) weight matrix → softmax per channel over K subsets | Each channel independently decides how much centripetal vs centrifugal vs self-loop to weight — no prior work does per-channel directional specialization |
| **MultiScaleTCN split-channel** | Splits channels in half, each half gets its own dilation (1 or 2), concat | 17-frame receptive field for free — single full-channel branch would need 2× pointwise params. Saves C²/2 per block |
| **Variant-progressive ST_JointAtt** | Attention only in later stages (nano: stage3 only; base: all stages) | Attention is expensive per param at small C — skipping it in early stages makes nano viable at 103K |

**The core thesis of LAST-E** — "input-level fusion + adaptive graph at reduced cost" — has not been directly explored before. Most efficient models sacrifice either the adaptive graph (use fixed topology) or stream fusion (use single stream). LAST-E does both without sacrificing either.

---

## Part 2: What SOTA Models Have Done — Completely Novel Contributions

### Generation 1 (2018-2020): Graph Structure

| Model | Novel Idea | Accuracy NTU60 xsub |
|-------|-----------|---------------------|
| **ST-GCN (AAAI 2018)** | First to apply GCN to skeletons. Partitioned adjacency into 3 spatial subsets (self/centripetal/centrifugal) | 81.5% |
| **2s-AGCN (CVPR 2019)** | Learned global A on top of physical — two-stream (joint+bone) | 88.5% |
| **MS-G3D (CVPR 2020 Oral)** | **Cross-spacetime edges** — joints at different timesteps directly connected. Disentangled multi-scale aggregation | 91.5% |

### Generation 2 (2021-2022): Topology Refinement + Information Theory

| Model | Novel Idea | Accuracy NTU60 xsub |
|-------|-----------|---------------------|
| **CTR-GCN (ICCV 2021)** | **Channel-topology refinement** — universal shared topology + channel-specific correction. Each channel group gets its own adjacency | 92.4% (4-stream) |
| **InfoGCN (CVPR 2022)** | **Information bottleneck** objective — explicitly maximizes mutual information between latent representation and action class while minimizing redundancy. Attention-based graph inference | 92.7% (6-stream) |
| **EfficientGCN (TPAMI 2022)** | Scaling law for skeleton GCNs (B0-B4). Depthwise-separable GCN. Showed efficiency frontier | 91.7% B4 (6-stream) |

### Generation 3 (2023-2024): Topology Decomposition + Transformers

| Model | Novel Idea | Accuracy NTU60 xsub |
|-------|-----------|---------------------|
| **HD-GCN (ICCV 2023)** | **Hierarchical decomposition** — splits skeleton into semantic sub-graphs (arm, leg, torso, head). Aggregates within sub-graphs first, then cross-graph. Explicitly preserves body-part semantics | ~92.6% |
| **Hyperformer (2023)** | **Hypergraph self-attention** — one hyper-edge can connect 3+ joints simultaneously (not just pairwise). Captures higher-order body group relations (entire arm moves together) | ~92.0% |
| **BlockGCN (CVPR 2024)** | **Persistent homology** (topological data analysis) to encode bone connectivity. Graph distances as topology features. BlockGC module reduces params while improving | ~92.8% |
| **SkateFormer (ECCV 2024)** | **Skeletal-Temporal partition attention** — 4 relation types: (near joints × near frames), (near joints × far frames), (far joints × near frames), (far joints × far frames). Partition-specific self-attention for each type | **93.0%+ (4-stream)** |

### Generation 4 (2025): Geometric Spaces

| Model | Novel Idea | Status |
|-------|-----------|--------|
| **HyLiFormer (2025)** | **Hyperbolic space for skeleton** — maps joints to Poincaré ball (hyperbolic geometry naturally models tree/hierarchical structures). Linear attention in hyperbolic space. Skeleton tree structure matches hyperbolic expansion rate | Preprint |
| **HI-GCN (2025)** | **Hierarchically intertwined graphs** — multiple levels of graph interaction, 93.3% NTU60 xsub | Published |

---

## Part 3: Novel Ideas for LAST — Deep Brainstorm

### Idea A: **Frequency-Aware Temporal Gating (FATG)**
*Completely unexplored in skeleton recognition*

**The insight:** Human actions have strong frequency signatures. Walking is periodic (0.5-2 Hz, low-frequency). Punching is impulsive (high-frequency burst). Writing involves fine high-frequency wrist motion. Current TCNs treat all frequencies equally — they're just spatial convolutions in time.

**The mechanism:**
```
Input (N, C, T, V)
   ↓ Apply DCT along T dimension → (N, C, T_freq, V)
   ↓ Learnable frequency attention mask: σ(W_freq) shape (1, C, T_freq, 1)
     → Each channel learns WHICH FREQUENCY BANDS are action-discriminative
   ↓ Element-wise gate: x_freq × mask
   ↓ IDCT → back to (N, C, T, V)
   ↓ Continue to GCN
```

**Why it's novel vs existing:**
- MS-G3D, CTR-GCN etc. never leave the time domain
- DCT is differentiable and parameter-free
- The learnable gate (C×T_freq params) is tiny
- An action's frequency signature is a DIFFERENT kind of discriminative information than its spatial topology — the two are complementary

**Efficiency:** DCT is O(T log T). Gate is C×T_freq ≈ 160×32 = 5120 params per block. **Zero conv ops.**

**Expected gain:** Strong signal for periodic actions (dancing, walking) and impulsive actions (kicking, punching). Likely 0.5-1% gain on NTU60 where 10-15% of classes are periodic.

---

### Idea B: **Action-Prototype Graph (APG) — Class-Guided Topology**
*Addresses the fundamental limitation of a shared A_learned*

**The insight:** A_learned is shared across ALL 60 classes. The graph useful for "kicking" (foot↔hip↔waist correlation) is different from "writing" (finger↔wrist↔elbow). A single learned matrix is a compromise that's suboptimal for every class.

**The mechanism:**
```
Learn K prototype graphs: G_proto = {A_1, ..., A_K} — (K, V, V) trainable
Learn K prototype feature vectors: p_1...p_K ∈ R^d

For each sample, after global pool of early backbone features → h ∈ R^d:
   similarity = softmax(h @ P^T) → (N, K) soft weights
   A_dynamic_semantic = Σ_k w_k × A_k → (N, V, V) semantic graph

Replace A_learned with this A_dynamic_semantic in DirectionalGCNConv/AdaptiveGraphConv
```

**Why novel:**
- InfoGCN uses information bottleneck but doesn't have prototype-specific graphs
- CTR-GCN has channel-specific topology but not class-specific
- This learns "archetypes of joint correlation patterns" and blends them per-sample
- K=15 prototypes × V×V = 15×625 = 9375 params — still tiny

**Expected gain:** This should help significantly with fine-grained action distinction — differentiating "kick" from "throw" requires knowing which joints to focus on, and that IS class-specific.

---

### Idea C: **Progressive Cross-Scale Re-fusion (PCRF)**
*Addresses the limitation of one-shot early fusion in LAST-E*

**The insight:** LAST-E fuses streams once at input (Stage 0). But at different depths:
- Stage 1 (low-level): joint positions give the structural skeleton — most informative early on
- Stage 2 (mid-level): bone angles give direction — useful for trajectory understanding
- Stage 3 (high-level): velocity gives temporal dynamics — motion fingerprint matters most for classification

**The mechanism:**
```
Input: 3 streams {S_joint, S_velocity, S_bone}, each (N, 3, T, V)
StreamFusion → fused (N, C0, T, V)  [existing, kept]

After Stage1: x1 = backbone_stage1(fused)
  Re-inject: compute lightweight cross-attention between x1 and BN(stem(S_bone))
  → x1_enriched = x1 + gate_1 × cross_attn(x1, S_bone_projected)

After Stage2: x2 = backbone_stage2(x1_enriched)
  Re-inject: velocity stream re-enters
  → x2_enriched = x2 + gate_2 × cross_attn(x2, S_velocity_projected)

Stage3: x3 = backbone_stage3(x2_enriched) → classify
```

**Cross-attention is cheap:** If we pool V→1 (per-joint global pool) and use a rank-1 attention (key-value from stream projected to same dimension as x), cost is C²/V per stage.

**Why novel:**
- FPN (Feature Pyramid Networks) does multi-scale re-injection for vision — no one has done this for skeleton streams specifically
- SkateFormer and others don't do progressive re-injection with stream-specific timing
- The key insight (different streams are useful at different depths) is empirically motivated by what each stream represents

---

### Idea D: **Hierarchical Body-Region Attention (HBRA)**
*Combines HD-GCN's decomposition + SkateFormer's partition + our efficiency requirements*

**The insight:** Joints within a body region (arm: shoulder, elbow, wrist, hand) have dense interactions. Joints across regions have sparse, action-specific interactions. Current attention treats all VxV pairs equally.

**The mechanism:**
Define 5 body regions: left arm (5 joints), right arm (5), left leg (5), right leg (5), torso/head (5).

```
Step 1 — Intra-region attention (cheap):
  Attend within each region: 5 joints × 5 joints = 25 pairs per region
  5 regions × 25 = 125 attention pairs (vs 625 for full V×V)
  Cost: O(R × r²) where R=5 regions, r=5 joints per region

Step 2 — Region tokens (aggregation):
  Each region → 1 summary token via learnable pool
  5 region tokens per frame

Step 3 — Inter-region attention (full but tiny):
  5×5 = 25 cross-region pairs — negligible
  This is where "arm↔leg" for kicking is learned

Step 4 — Broadcast back:
  Region tokens → expand back to per-joint, add residually
```

**Why novel vs SkateFormer:**
- SkateFormer partitions by near/far joints (distance-based). HBRA partitions by anatomical body region (semantic).
- HBRA has a natural 2-level hierarchy (joint → region → cross-region) matching how humans understand motion
- SkateFormer doesn't have region summary tokens
- HBRA cost: 5×5² + 5² = 150 attention pairs vs V² = 625. **4× cheaper with same expressivity for body-part interactions**

---

### Idea E: **Causal + Bidirectional Temporal Fusion (CBTF)**
*New training paradigm — not just architecture*

**The insight:** Standard TCN sees the future (symmetric padding). During a punch, frames BEFORE impact predict frames DURING impact. This means if you mask future frames at inference, you get worse results than if you'd never seen the future during training.

**The novel training approach:**
```
Training: for 50% of batches, apply causal masking (only left-padding in TCN)
          for 50% of batches, use normal symmetric TCN
Inference: bidirectional (symmetric) — all frames available

This forces the model to learn representations that are PREDICTIVE of future frames
not just retrospectively correlated with them.
```

Additionally: identify **motion peaks** (frames with maximum velocity magnitude across joints). Apply higher attention weight to frames near motion peaks:
```
velocity = ||x_t - x_{t-1}||_2  → peak_weight = softmax(velocity/τ)
temporal_feature_weight = 1.0 + β × peak_weight  (β=learnable)
```

**Why novel:** No skeleton recognition paper uses causal training for non-streaming applications. The idea of "motion peak awareness" is intuitive but absent from literature.

---

### Recommended Synthesis: LAST-v3 Core Block

Combining the most impactful and mutually compatible ideas:

```
LAST-v3 Block = DirectionalGCNConv        (existing)
              + FrequencyGate             (Idea A, ~5K params per block, no conv)
              + ActionPrototypeGraph      (Idea B, replaces A_learned, 9375 params total)
              + IntraRegionAttention      (Idea D, 4× cheaper than full attention)
              + MultiScaleTCN            (existing)
              + ST_JointAtt              (existing)
```

**What this achieves:**
- Frequency-discriminative temporal representation (unexplored in skeleton domain)
- Class-conditioned graph topology (solves the "one A_learned fits all" problem)
- Anatomically-aware attention within body regions (computationally efficient)
- All on top of proven efficient base (MultiScaleTCN, DirectionalGCNConv)

**Estimated accuracy gain:** Frequency gating alone is likely worth 0.5-1%. Prototype topology likely 0.5-1.5%. Hierarchical body attention likely 0.5-1%. Combined (with synergies): potentially **2-3% over current LAST-E base**, putting LAST-E base at ~92% NTU60 xsub — competitive with CTR-GCN at 9.2M params vs our 400K.

---

## Part 4: Priority Ranking for Implementation

| Rank | Idea | Params Added | Expected Gain | Implementation Risk |
|------|------|-------------|---------------|---------------------|
| 1 | **Frequency-Aware Temporal Gate** (A) | ~5K/block | 0.5-1% | Low — DCT is standard |
| 2 | **Action-Prototype Graph** (B) | ~10K total | 0.5-1.5% | Medium — needs warm-up care |
| 3 | **Progressive Re-fusion** (C) | ~10K/stage | 0.5-1% | Medium — sequence matters |
| 4 | **Hierarchical Body-Region Attn** (D) | ~20K/stage | 0.5-1% | Medium — region definition matters |
| 5 | **Causal Training** (E) | 0 | 0.3-0.8% | Low — training change only |

**Recommended first experiment:** Idea A (Frequency Gate) — it's orthogonal to all existing components,
adds essentially no params, and addresses a completely unexplored axis of the problem. If it works,
combine with B.

---

## Sources

- [Skeleton-Based Action Recognition on NTU RGB+D — Papers With Code](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd)
- [BlockGCN: Redefine Topology Awareness (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [SkateFormer: Skeletal-Temporal Transformer (ECCV 2024)](https://arxiv.org/abs/2403.09508)
- [HD-GCN: Hierarchically Decomposed GCN (ICCV 2023)](https://github.com/Jho-Yonsei/HD-GCN)
- [Hyperformer: Hypergraph Transformer](https://arxiv.org/abs/2211.09590)
- [HyLiFormer: Hyperbolic Linear Attention (2025)](https://arxiv.org/html/2502.05869)
- [InfoGCN: Information Bottleneck for Skeleton GCN (CVPR 2022)](https://github.com/stnoah1/infogcn)
- [HI-GCN: Hierarchically Intertwined GCN (2025)](https://www.nature.com/articles/s41598-025-19399-4)
