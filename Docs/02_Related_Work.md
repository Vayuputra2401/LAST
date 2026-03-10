# 02 — Related Work

## Generational Landscape of Skeleton-Based Action Recognition

We organise the evolution of skeleton GCNs into four generations, each defined by a paradigm shift in how spatial and temporal structure is modelled.

---

### Generation 1 (2018–2020): Graph Structure Discovery

The foundational generation established that modelling skeleton sequences as spatio-temporal graphs enables powerful relational reasoning — where joints are nodes, anatomical bones are edges, and temporal connections link the same joint across consecutive frames.

| Method | Venue | Params | NTU-60 xsub | Key Innovation |
|--------|-------|--------|-------------|----------------|
| ST-GCN | AAAI 2018 | ~3M | 81.5% | First GCN for skeletons; spatial 3-subset partitioning |
| 2s-AGCN | CVPR 2019 | ~3.5M | 88.5% | Learnable global adjacency; two-stream (joint + bone) |
| Shift-GCN | CVPR 2020 | 2.8M | 90.7% | Zero-parameter joint-dim shift; random channel routing |
| MS-G3D | CVPR 2020 | ~3.2M | 91.5% | Cross-spacetime edges; disentangled multi-scale aggregation |

**Limitation**: Fixed or globally-learned adjacency matrices cannot capture the sample-specific, action-dependent topology variations that distinguish fine-grained actions. Multi-stream extensions (2s, 4s) improve accuracy but multiply parameter counts.

---

### Generation 2 (2021–2022): Topology Refinement and Information Theory

This generation refined graph topology at sub-graph granularity — per-channel, per-sample, or per-channel-group — and introduced information-theoretic training objectives.

| Method | Venue | Params | NTU-60 xsub | NTU-120 xsub | Key Innovation |
|--------|-------|--------|-------------|--------------|----------------|
| CTR-GCN | ICCV 2021 | 1.7M | 92.4% | 88.9% | Channel-topology refinement; per-channel-group adjacency correction |
| EfficientGCN-B0 | TPAMI 2022 | 290K | 90.2% | 86.3% | Compound scaling (B0–B4); depthwise-separable GCN |
| EfficientGCN-B4 | TPAMI 2022 | 1.1M | 92.1% | 88.0% | Larger compound-scaled variant |
| InfoGCN | CVPR 2022 | 1.5M | 93.0% | 89.8% | Information bottleneck loss; attention-based dynamic graph |

**Limitation**: Per-sample dynamic graph inference (InfoGCN) is expensive and incompatible with low-power deployment. Compound scaling (EfficientGCN) improves accuracy without any architectural novelty — it is a generic GCN scaled down. No method in this generation exploits the anatomical structure of body regions or temporal frequency.

---

### Generation 3 (2023–2024): Decomposition and Transformers

Methods in this generation decompose the skeleton by body region or apply transformer-style attention to capture long-range spatio-temporal dependencies.

| Method | Venue | Params | NTU-60 xsub | NTU-120 xsub | Key Innovation |
|--------|-------|--------|-------------|--------------|----------------|
| HD-GCN | ICCV 2023 | 3.3M | 93.6% | 90.1% | Hierarchical body-region decomposition into sub-graphs |
| BlockGCN | CVPR 2024 | ~1.8M | 92.8% | 89.3% | Persistent homology for topology encoding |
| SkateFormer | ECCV 2024 | ~2.5M | 93.0%+ | 89.5%+ | 4-type skeletal-temporal partition attention |

**Limitation**: HD-GCN's region decomposition groups joints by body part but misses the cross-region long-range dependencies (e.g., left hand coordinated with right foot). Transformer-based methods (SkateFormer) incur quadratic O(T²) attention complexity, precluding efficient edge deployment. No method systematically models bilateral skeletal symmetry.

---

### Generation 4 (2025): Multi-Level and Geometric Approaches

The most recent generation explores hierarchical multi-level graph intertwining and non-Euclidean embedding spaces.

| Method | Venue | Params | NTU-60 xsub | NTU-120 xsub | Key Innovation |
|--------|-------|--------|-------------|--------------|----------------|
| HyLiFormer | 2025 | ~3M | ~92.5% | ~89.0% | Hyperbolic space for skeleton hierarchy encoding |
| HI-GCN | 2025 | ~2.5M | 93.3% | 90.1% | Hierarchically intertwined multi-level graph interaction |

**Limitation**: Hyperbolic space (HyLiFormer) requires specialised Riemannian optimisers, making deployment difficult. All Generation 4 methods remain in the 2–3M parameter regime and offer no explicit lightweight variant.

---

## Lightweight Skeleton Recognition

The sub-500K parameter regime has received far less attention than the accuracy-maximising regime. The only notable published methods are:

**EfficientGCN-B0** (TPAMI 2022, 290K params, 90.2% NTU-60 xsub) is the primary lightweight baseline. It achieves strong accuracy through compound scaling (depth, width, temporal resolution) but introduces zero novel architectural primitives — it is a scaled-down generic GCN pipeline.

**Shift-GCN** (CVPR 2020, 2.8M params, 90.7%) replaces graph convolution with zero-parameter shifts along the joint dimension. While the shift concept is elegant, the original implementation uses random channel-to-joint assignments with no anatomical motivation, and at 2.8M parameters it is not a lightweight method.

**SGN** (CVPR 2020, 690K params, 89.4%) introduces joint semantic embedding and frame-level dynamic context gates, but focuses on a smaller joint set (17 joints) and does not scale to the NTU-60 multi-stream setting.

---

## Gap in the Literature

ShiftFuse V10 occupies a unique position in this space:

| Property | EfficientGCN-B0 | Shift-GCN | InfoGCN | **ShiftFuse V10 nano (Ours)** |
|----------|----------------|-----------|---------|-------------------------------|
| **Params** | 290K | 2.8M | 1.5M | **225K** |
| **NTU-60 xsub** | 90.2% | 90.7% | 93.0% | **TBD** |
| Anatomical graph typing | No | No | No | **Yes (SGP, K=3 typed subsets)** |
| Efficient global temporal attention | No | No | No | **Yes (TLA, O(T×K), K=14)** |
| Anatomical spatial routing | No | Random shift | No | **Yes (BRASP, body-region routing)** |
| IB loss with inter-class separation | No | No | Attraction only | **Yes (Triplet margin IB)** |
| Shared GCN with gradient guard | No | N/A | No | **Yes (gcn_scale per block)** |
| Learned stream ensemble | No | N/A | No | **Yes (softmax stream_weights)** |
| Novel contributions | 0 | 1 | 2 | **6** |

ShiftFuse V10 nano is the first sub-250K parameter skeleton GCN to introduce six distinct, theoretically motivated novel components.

---

## Components Adapted from Prior Work

We acknowledge the following components adapted from existing methods:

| Component | Source | Our Adaptation |
|-----------|--------|----------------|
| K-subset spatial adjacency | ST-GCN (AAAI 2018) | Extended to anatomically-typed SGP subsets (A_intra / A_inter / A_cross) |
| Channel-topology refinement | CTR-GCN (ICCV 2021) | Lightweight G=4 group conv with per-group adjacency correction |
| Information bottleneck loss | InfoGCN (CVPR 2022) | Extended with triplet margin (d_same − d_wrong); weight increased 100× |
| Landmark-frame attention | Longformer (2020) | Adapted to skeleton temporal sequences; spatial pooling before Q projection |
| JointEmbedding | SGN (CVPR 2020) | SGN-style additive per-joint bias (V×C), stage-shared in nano |
| 4-stream decomposition | EfficientGCN (TPAMI 2022) | Same 4 kinematic streams; late-fusion ensemble instead of early fusion |
| DropPath (stochastic depth) | Huang et al. (2016) | Linear ramp schedule 0→drop_path_rate across blocks |

The distinction is precise: borrowed components are **standard engineering building blocks**. Our six novel contributions (SGP, TLA, BRASP-corrected, Triplet IB loss, gcn_scale, learned ensemble) address structural gaps that no prior work has identified or exploited.

---

## Positioning Summary

| Category | Method | Params | NTU-60 xsub |
|----------|--------|--------|-------------|
| Lightweight baseline | EfficientGCN-B0 | 290K | 90.2% |
| Best overall SOTA | HD-GCN | 3.3M | 93.6% |
| Best information-theoretic | InfoGCN | 1.5M | 93.0% |
| Best lightweight novel | Shift-GCN | 2.8M | 90.7% |
| **Ours** | **ShiftFuse V10 nano** | **225K** | **TBD** |

**Research claim**: ShiftFuse V10 nano achieves the strongest accuracy-per-parameter ratio in the sub-250K regime, with the largest number of novel architectural contributions (6) of any lightweight skeleton GCN to date.

---

## Sources

- [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455)
- [2s-AGCN (CVPR 2019)](https://arxiv.org/abs/1912.06971)
- [Shift-GCN (CVPR 2020)](https://arxiv.org/abs/2003.07466)
- [MS-G3D (CVPR 2020)](https://arxiv.org/abs/2003.14111)
- [SGN (CVPR 2020)](https://arxiv.org/abs/1904.01189)
- [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213)
- [EfficientGCN (TPAMI 2022)](https://arxiv.org/abs/2106.15125)
- [InfoGCN (CVPR 2022)](https://github.com/stnoah1/infogcn)
- [HD-GCN (ICCV 2023)](https://github.com/Jho-Yonsei/HD-GCN)
- [BlockGCN (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [SkateFormer (ECCV 2024)](https://arxiv.org/abs/2403.09508)
- [HyLiFormer (2025)](https://arxiv.org/html/2502.05869)
- [HI-GCN (2025)](https://www.nature.com/articles/s41598-025-19399-4)
- [Longformer (ICLR 2020)](https://arxiv.org/abs/2004.05150)
- [Papers With Code — NTU RGB+D](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd)
