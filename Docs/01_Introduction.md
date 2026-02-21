# 01 — Introduction

## Problem Statement

Human action recognition from 3D skeleton sequences is a core computer vision task with applications
in human-computer interaction, surveillance, sports analysis, and healthcare monitoring. Unlike RGB
video, skeleton data is compact, viewpoint-invariant, and robust to illumination and background noise.

The key challenge is the **efficiency vs. accuracy tension**: high-accuracy GCN models (CTR-GCN,
InfoGCN, HD-GCN) require 3–10M parameters and are impractical on edge hardware. Lightweight models
(EfficientGCN B0) sacrifice significant accuracy. There is no existing work that achieves both
sub-150K parameter count *and* competitive accuracy.

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

2. **LAST-E (Student) family** — Four extreme-efficiency variants (nano/small/base/large) spanning
   92K–644K parameters. Every variant beats EfficientGCN at its corresponding parameter tier.
   Key innovations:
   - **StreamFusion**: per-channel (3, C₀) softmax-weighted blend replaces 3× backbone cost
   - **DirectionalGCNConv**: K=3 directed subsets with per-channel alpha weights, no subset sum
   - **MultiScaleTCN**: dual-dilation (1+2) parallel branches, saves C²/2 pointwise params/block

3. **Knowledge Distillation pipeline** — LAST-v2 → LAST-E KD using soft logits (KL + CE loss),
   transferring accuracy from the 9.2M teacher to sub-1M students. Expected +2–4% top-1 gain.

---

## Target Venue

**ECCV 2026**
- Abstract deadline: February 26, 2026
- Paper deadline: March 5, 2026

---

## Document Map

| Section | File | Content |
|---------|------|---------|
| 02 | [Related Work](02_Related_Work.md) | SOTA comparison, what we borrow |
| 03 | [Architecture](03_Architecture.md) | LAST-v2 + LAST-E detailed design |
| 04 | [Data Pipeline](04_Data_Pipeline.md) | MIB streams, NTU60/120, preprocessing |
| 05 | [Training](05_Training.md) | Optimizer, scheduler, commands |
| 06 | [Distillation](06_Distillation.md) | KD plan and hyperparameters |
| 07 | [Experiments](07_Experiments.md) | Param counts, results (fills in over time) |
| 08 | [Environment Setup](08_Environment_Setup.md) | Local / Kaggle / GCP setup |
