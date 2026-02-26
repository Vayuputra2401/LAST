# 02 — Related Work

## SOTA Benchmark (NTU RGB+D 60, X-Sub protocol)

### Generation 1 (2018–2020): Graph Structure

| Method | Params | Top-1 (%) | Novel Contribution |
|--------|--------|-----------|-------------------|
| ST-GCN (AAAI 2018) | ~3M | 81.5 | First GCN on skeletons; 3-subset spatial partitioning |
| 2s-AGCN (CVPR 2019) | ~3.5M | 88.5 | Learned global adjacency; two-stream (joint+bone) |
| MS-G3D (CVPR 2020) | ~3.2M | 91.5 | Cross-spacetime edges; disentangled multi-scale |

### Generation 2 (2021–2022): Topology Refinement + Information Theory

| Method | Params | Top-1 (%) | Novel Contribution |
|--------|--------|-----------|-------------------|
| CTR-GCN (ICCV 2021) | 1.7M | 92.4 | Channel-topology refinement; per-channel-group adjacency |
| EfficientGCN (TPAMI 2022) | 150K–2M | 88.3–91.7 | Scaling law (B0–B4); depthwise-separable GCN |
| InfoGCN (CVPR 2022) | 1.5M | 93.0 | Information bottleneck; attention-based graph inference |

### Generation 3 (2023–2024): Topology Decomposition + Transformers

| Method | Params | Top-1 (%) | Novel Contribution |
|--------|--------|-----------|-------------------|
| HD-GCN (ICCV 2023) | 3.3M | 93.6 | Hierarchical decomposition into body-region sub-graphs |
| BlockGCN (CVPR 2024) | ~1.8M | 92.8 | Persistent homology for topology encoding |
| SkateFormer (ECCV 2024) | ~2.5M | 93.0+ | 4-type skeletal-temporal partition attention |

### Generation 4 (2025): Geometric Spaces

| Method | Params | Top-1 (%) | Novel Contribution |
|--------|--------|-----------|-------------------|
| HyLiFormer (2025) | ~3M | ~92.5 | Hyperbolic space for skeleton hierarchy |
| HI-GCN (2025) | ~2.5M | 93.3 | Hierarchically intertwined multi-level graphs |

### LAST Family (Ours)

| Model | Params | Top-1 | Status |
|-------|--------|-------|--------|
| LAST-v2-base (teacher) | 9.2M | *TBD* | Training |
| LAST-E v3 nano | 83K | *TBD* | Configurable |
| LAST-E v3 small | 345K | *TBD* | Configurable |
| **LAST-E v3 base** | **720K** | *TBD* | **Phase A+B+D running** |
| LAST-E v3 large | 1.08M | *TBD* | Configurable |
| LAST-Lite nano (edge) | ~60K | *TBD* | Planned |
| LAST-Lite small (edge) | ~180K | *TBD* | Planned |

---

## What LAST-E v3 Borrows from EfficientGCN

1. **SpatialGCN graph convolution** — Multi-subset adjacency partitioning with learnable edge and
   subset attention. LAST-E v3 improves this with full-graph D⁻½AD⁻½ normalization (N1 fix)
   instead of EfficientGCN's per-subset D⁻¹A.

2. **EpSepTCN (Expanded Separable TCN)** — MobileNetV2-style inverted bottleneck for temporal
   convolution. Expand(1×1) → Depthwise(k×1) → Pointwise(1×1) with residual. Directly ported
   from EfficientGCN's `Temporal_Sep_Layer`.

3. **Variant scaling principle** — nano/small/base/large scaling by width and depth, similar
   to EfficientGCN's B0–B4 compound scaling.

---

## What LAST-E v3 Adds Beyond EfficientGCN

| Feature | EfficientGCN | LAST-E v3 | Novelty |
|---------|-------------|-----------|---------|
| **Graph normalization** | Per-subset D⁻¹A | Full-graph D⁻½AD⁻½ | Prevents subset norm imbalance |
| **MotionGate** | None | Temporal-difference channel gating | Novel — no prior work |
| **HybridGate** | None | SE-style + MotionGate blend | Novel combination |
| **ST_JointAtt** | None | Factorized S+T attention, zero-init gate | From LAST-v2, improved init |
| **StreamFusion** | MIB early concat | Per-channel (3,C₀) softmax blend | Per-channel specialization |
| **Gated pooling head** | GAP only | Learnable GAP+GMP blend | Per-channel avg/max selection |
| **DropPath** | None | Stochastic depth, linear ramp | Standard modern technique |
| **Activation** | ReLU | Hardswish | Better for quantization |

---

## LAST-Lite: Architectural Distinction from EfficientGCN

LAST-Lite (edge variants) removes all adaptive modules but retains 4 structural differences:

1. **Full-graph D⁻½AD⁻½ normalization** — mathematically prevents self-loop dominance
2. **Late 3-stream concat fusion** — each stream learns independently before head (vs EfficientGCN's early MIB concat)
3. **Gated GAP+GMP pooling head** — learnable average vs max blend per channel
4. **Hardswish activation** — better gradient flow and quantization-friendly

---

## Sources

- [Papers With Code — NTU RGB+D](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd)
- [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213)
- [EfficientGCN (TPAMI 2022)](https://arxiv.org/abs/2106.15125)
- [InfoGCN (CVPR 2022)](https://github.com/stnoah1/infogcn)
- [HD-GCN (ICCV 2023)](https://github.com/Jho-Yonsei/HD-GCN)
- [BlockGCN (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [SkateFormer (ECCV 2024)](https://arxiv.org/abs/2403.09508)
- [HyLiFormer (2025)](https://arxiv.org/html/2502.05869)
- [HI-GCN (2025)](https://www.nature.com/articles/s41598-025-19399-4)
