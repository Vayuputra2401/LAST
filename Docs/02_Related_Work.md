# 02 — Related Work

## Generational Landscape of Skeleton-based Action Recognition

We organise the evolution of skeleton GCNs into four generations, each defined by a paradigm shift in how spatial and temporal structure is modelled.

### Generation 1 (2018--2020): Graph Structure Discovery

The foundational generation established that treating skeleton sequences as spatio-temporal graphs — rather than coordinate vectors — unlocks powerful relational reasoning.

| Method | Venue | Params | NTU-60 xsub | Key Innovation |
|--------|-------|--------|-------------|----------------|
| ST-GCN | AAAI 2018 | ~3M | 81.5% | First GCN on skeletons; 3-subset spatial partitioning |
| 2s-AGCN | CVPR 2019 | ~3.5M | 88.5% | Learnable global adjacency; two-stream (joint + bone) |
| MS-G3D | CVPR 2020 | ~3.2M | 91.5% | Cross-spacetime edges; disentangled multi-scale aggregation |

**Limitation**: Fixed or globally-learned adjacency matrices cannot capture the sample-specific, action-dependent topology variations that distinguish fine-grained actions.

### Generation 2 (2021--2022): Topology Refinement and Information Theory

This generation refined graph topology at finer granularity — per-channel, per-sample, or per-subset — and introduced information-theoretic training objectives.

| Method | Venue | Params | NTU-60 xsub | Key Innovation |
|--------|-------|--------|-------------|----------------|
| CTR-GCN | ICCV 2021 | 1.7M | 92.4% | Channel-topology refinement; per-channel-group adjacency |
| EfficientGCN | TPAMI 2022 | 290K--2M | 90.2--91.7% | Compound scaling (B0--B4); depthwise-separable GCN |
| InfoGCN | CVPR 2022 | 1.5M | 93.0% | Information bottleneck loss; attention-based graph inference |

**Limitation**: Topology refinement is applied uniformly across all body parts, ignoring the distinct kinematic roles of arms, legs, and torso. No frequency-domain processing is considered.

### Generation 3 (2023--2024): Decomposition and Transformers

Methods in this generation decompose the skeleton into body regions or apply transformer-style attention to capture long-range spatio-temporal dependencies.

| Method | Venue | Params | NTU-60 xsub | Key Innovation |
|--------|-------|--------|-------------|----------------|
| HD-GCN | ICCV 2023 | 3.3M | 93.6% | Hierarchical decomposition into body-region sub-graphs |
| BlockGCN | CVPR 2024 | ~1.8M | 92.8% | Persistent homology for topology encoding |
| SkateFormer | ECCV 2024 | ~2.5M | 93.0%+ | 4-type skeletal-temporal partition attention |

**Limitation**: Region decomposition (HD-GCN) treats left and right body halves as independent groups, missing the bilateral symmetry signal. Transformer-based methods (SkateFormer) incur quadratic complexity, limiting deployment on edge hardware.

### Generation 4 (2025): Multi-level and Geometric Approaches

The most recent generation explores hierarchical graph intertwining and non-Euclidean embedding spaces.

| Method | Venue | Params | NTU-60 xsub | Key Innovation |
|--------|-------|--------|-------------|----------------|
| HyLiFormer | 2025 | ~3M | ~92.5% | Hyperbolic space for skeleton hierarchy |
| HI-GCN | 2025 | ~2.5M | 93.3% | Hierarchically intertwined multi-level graphs |

---

## Lightweight Skeleton Recognition

The lightweight regime (sub-500K parameters) has received far less attention than the accuracy-maximising regime. Two notable works define this space:

**EfficientGCN** (TPAMI 2022) is the primary baseline. Its B0 variant (290K params, 90.2% NTU-60 xsub) uses depthwise-separable GCN and compound scaling but introduces no novel architectural primitives — it is a scaled-down version of a standard GCN pipeline.

**Shift-GCN** (CVPR 2020, 2.8M params, 90.7%) replaces graph convolution with zero-parameter shift operations along the joint dimension. While the shift idea is parameter-free, the original implementation uses random channel-to-joint assignments with no anatomical structure, and the full model still requires 2.8M parameters.

### Gap in the Literature

| Property | EfficientGCN-B0 | Shift-GCN | **LAST-Lite (Ours)** |
|----------|----------------|-----------|---------------------|
| Params | 290K | 2.8M | **80K--248K** |
| Anatomical spatial routing | No | No (random shift) | **Yes (BRASP)** |
| Bilateral symmetry modelling | No | No | **Yes (BSE)** |
| Frequency-domain processing | No | No | **Yes (FDCR)** |
| Learnable graph correction | Yes (learnable edge) | No | **Yes (StaticGCN + A_learned)** |
| Novel contributions | 0 | 1 (shift) | **4 (BRASP, BSE, FDCR, StaticGCN)** |

LAST-Lite is the first sub-250K parameter model to introduce multiple novel, theoretically motivated architectural primitives for skeleton action recognition.

---

## What LAST-Lite Borrows from Prior Work

We acknowledge the components adapted from existing methods:

| Component | Source | Our Adaptation |
|-----------|--------|---------------|
| EpSepTCN | EfficientGCN | MobileNetV2-style expand-depthwise-project temporal conv; used as-is |
| StreamFusionConcat | EfficientGCN | Per-stream BN + concat + Conv1x1 for early 4-stream fusion |
| JointEmbedding | SGN (2020) | Additive per-joint semantic bias (V x C learnable embedding) |
| FrameDynamicsGate | SGN (2020) | Temporal position gate: per-frame learnable sigmoid mask |
| Gated GAP+GMP head | Various | Learnable per-channel blend of average and max pooling |
| K-subset adjacency | ST-GCN | Multi-hop spatial partitioning with D^{-1/2}AD^{-1/2} normalisation |

The distinction is clear: borrowed components are **standard building blocks** (temporal convolution, stream fusion, pooling). Our **four novel contributions** (BRASP, BSE, FDCR, StaticGCN) address structural gaps that no prior work has explored.

---

## What LAST-Base Adds Beyond LAST-Lite

LAST-Base (planned) incorporates ideas from Generation 2--4 SOTA models alongside original contributions:

| Source | What LAST-Base Takes | What We Add |
|--------|---------------------|-------------|
| CTR-GCN | Channel-topology refinement | Per-channel-group adjacency with temporal context |
| InfoGCN | Information bottleneck loss | Applied to stage 3 output |
| SkateFormer | Partitioned attention concept | 4-type spatio-temporal partition heads |
| HD-GCN | Body-region decomposition idea | Hierarchical intra- and inter-region attention |
| HI-GCN | Cross-temporal topology | Temporal context MLP for dynamic adjacency correction |
| **Novel** | — | Action-Prototype Graph: K=15 class-conditioned adjacency via prototype blending |
| **Novel** | — | FreqTemporalGate: full adaptive DCT frequency attention (per-sample) |

---

## Sources

- [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455)
- [2s-AGCN (CVPR 2019)](https://arxiv.org/abs/1912.06971)
- [MS-G3D (CVPR 2020)](https://arxiv.org/abs/2003.14111)
- [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213)
- [EfficientGCN (TPAMI 2022)](https://arxiv.org/abs/2106.15125)
- [InfoGCN (CVPR 2022)](https://github.com/stnoah1/infogcn)
- [SGN (CVPR 2020)](https://arxiv.org/abs/1904.01189)
- [HD-GCN (ICCV 2023)](https://github.com/Jho-Yonsei/HD-GCN)
- [BlockGCN (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [SkateFormer (ECCV 2024)](https://arxiv.org/abs/2403.09508)
- [HyLiFormer (2025)](https://arxiv.org/html/2502.05869)
- [HI-GCN (2025)](https://www.nature.com/articles/s41598-025-19399-4)
- [Shift-GCN (CVPR 2020)](https://arxiv.org/abs/2003.07466)
- [Papers With Code — NTU RGB+D](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd)
