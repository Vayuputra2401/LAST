# 01 — Introduction

## Problem Statement

Human action recognition from 3D skeleton sequences is a core computer vision task with applications
in human-computer interaction, surveillance, sports analysis, and healthcare monitoring. Unlike RGB
video, skeleton data is compact, viewpoint-invariant, and robust to illumination and background noise.

The key challenge is the **efficiency vs. accuracy tension**: high-accuracy GCN models (CTR-GCN,
InfoGCN, HD-GCN) require 1.5–10M parameters and are impractical on edge hardware. Lightweight models
(EfficientGCN-B0) sacrifice significant accuracy. There is no existing work that achieves both
sub-1M parameter count *and* competitive accuracy with rich novel architectural contributions.

---

## Why 3 Streams (MIB)?

The Motion-Interaction-Bone (MIB) decomposition captures complementary kinematic signals:

| Stream   | Input                          | Signal captured                         |
|----------|--------------------------------|-----------------------------------------|
| Joint (J)| Coords relative to spine base  | Absolute pose configuration             |
| Velocity (V)| Frame-diff J_{t+1} − J_t   | Temporal dynamics, motion speed         |
| Bone (B) | J_child − J_parent             | Limb orientations, body structure       |

Concatenating or fusing all three streams consistently outperforms single-stream models by 1–3%
on NTU60/120. LAST-v2 processes all three independently; LAST-E fuses them at input to run a
single backbone.

---

## Contributions

1. **LAST-v2 (Teacher)** — A high-accuracy 9.2M-parameter model with three independent per-stream
   backbones, AdaptiveGraphConv (physical + learned + dynamic adjacency), ST_JointAtt spatial
   attention, and LinearAttention for temporal modeling. Designed for maximum accuracy as the
   teacher in knowledge distillation.

2. **LAST-E v3 (Student) family** — Four efficiency variants (nano/small/base/large) spanning
   83K–1.08M parameters. Built on EfficientGCN-derived SpatialGCN with full-graph D⁻½AD⁻½
   normalization and multi-subset partitioning. Key innovations:
   - **StreamFusion**: per-channel (3, C₀) softmax-weighted blend replaces 3× backbone cost
   - **SpatialGCN**: full-graph degree-normalized, multi-hop K=5 subset partitioning (N1 fix)
   - **EpSepTCN**: MobileNetV2-style inverted-bottleneck separable temporal convolution
   - **MotionGate / HybridGate**: temporal-difference channel gating (novel — no prior work)
   - **ST_JointAtt**: factorized spatial-temporal attention with zero-init residual gate
   - **Gated GAP+GMP head**: learnable per-channel blend of average and max pooling
   - **DropPath**: stochastic depth regularization with linear ramp

3. **LAST-Lite (Edge) family** — Two fixed-computation variants (nano_lite/small_lite) designed
   for edge deployment. Same graph structure as LAST-E v3 but with all adaptive modules removed
   (no MotionGate, ST_JointAtt, subset_att, learnable edge). Pure convolution, trivially
   quantizable. Target: ~60K–180K params, INT8 deployable at <5ms on Jetson Nano.

4. **Novel architectural ideas** for future research:
   - **Frequency-Aware Temporal Gate (FATG)**: DCT-domain per-channel frequency attention
   - **Action-Prototype Graph (APG)**: class-conditioned topology via prototype blending
   - **Progressive Cross-Scale Re-fusion (PCRF)**: stream re-injection at different backbone depths
   - **Hierarchical Body-Region Attention (HBRA)**: anatomical partition → 4× cheaper attention
   - **Causal + Bidirectional Training (CBTF)**: 50% causal masking for predictive representations

5. **Training pipeline** — Knowledge distillation (LAST-E v3 → LAST-Lite), optional MaskCLR
   self-supervised pretraining, INT8 quantization, and ONNX/TensorRT edge deployment.

---

## Target Venue

**ECCV 2026**
- Abstract deadline: February 26, 2026
- Paper deadline: March 5, 2026

---

## Document Map

| Section | File | Content |
|---------|------|---------|
| 02 | [Related Work](02_Related_Work.md) | SOTA comparison, generational landscape |
| 03 | [Architecture](03_Architecture.md) | LAST-v2 + LAST-E v3 + LAST-Lite design |
| 04 | [Data Pipeline](04_Data_Pipeline.md) | MIB streams, NTU60/120, preprocessing |
| 05 | [Training](05_Training.md) | Optimizer, schedulers, SGDR, commands |
| 06 | [Distillation](06_Distillation.md) | KD + MaskCLR pretraining plan |
| 07 | [Experiments](07_Experiments.md) | Param counts, results, ablation plan |
| 08 | [Environment Setup](08_Environment_Setup.md) | Local / Kaggle / GCP setup |
