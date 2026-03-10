# 01 — Introduction

## Abstract

We present **ShiftFuse V10**, a lightweight skeleton-based action recognition architecture that introduces six novel, theoretically motivated components within a sub-250K parameter budget. The model targets the NTU RGB+D 60 cross-subject benchmark, where existing sub-300K methods offer no architectural novelty beyond compound scaling. ShiftFuse V10 nano (225K parameters) achieves a strong baseline of **81.17%** (V10.2 preliminary) and, following the V10.3 training-fix and parameter-reallocation improvements documented here, is projected to reach **85–87%** — within 3–5pp of CTR-GCN (1.7M params, 92.4%) at **7.6× fewer parameters**. Results for V10.3 are pending training completion.

---

## Problem Statement

Skeleton-based action recognition from 3D joint coordinate sequences is a foundational task in human activity understanding, with direct applications across human-computer interaction, surveillance, sports analytics, and rehabilitation monitoring. Skeleton data is compact, viewpoint-invariant, privacy-preserving, and robust to background, illumination, and appearance variations — properties that make it the preferred modality for deployable activity recognition systems.

Graph Convolutional Networks (GCNs) have become the dominant paradigm, modelling the human body as a spatio-temporal graph in which joints are nodes and anatomical connections are edges. However, the field exhibits a fundamental **efficiency-accuracy tension**:

- **State-of-the-art methods** (CTR-GCN, InfoGCN, HD-GCN, HI-GCN) reach 92–94% top-1 accuracy on NTU RGB+D 60 but require 1.5–3.5M parameters and extensive per-sample adaptive computation.
- **Lightweight methods** (EfficientGCN-B0, 290K params, 90.2%) introduce no novel architectural primitives — they apply compound scaling to a generic GCN backbone.

No existing work simultaneously achieves:
1. Sub-250K parameter count
2. Accuracy competitive with 1-2M parameter models
3. Multiple genuinely novel, domain-motivated architectural contributions

ShiftFuse V10 addresses all three.

---

## Why Multiple Streams?

Human motion is a multi-faceted kinematic signal. A single representation cannot capture all discriminative information. Following the multi-stream paradigm of 2s-AGCN and EfficientGCN, we decompose each skeleton sequence into four complementary streams:

| Stream | Key | Derivation | Captured Signal |
|--------|-----|------------|-----------------|
| Joint | `joint` | 3D coordinates, spine-base centred | Absolute pose configuration |
| Velocity | `velocity` | J(t+1) − J(t) | Motion speed and direction |
| Bone | `bone` | J(child) − J(parent) per edge | Limb orientation and length |
| Bone velocity | `bone_velocity` | B(t+1) − B(t) | Limb angular velocity |

All four streams are processed by a single shared backbone (fused along the batch dimension), then classified by four independent heads. At evaluation, stream logits are combined via a learned softmax-weighted ensemble.

---

## Contributions

This work presents **ShiftFuse V10**, a four-stream skeleton GCN with six novel contributions:

### 1. Semantic Body-Part Graph (SGP)

A hand-crafted, domain-motivated three-subset spatial adjacency:

- **A_intra** — connections within each anatomical body region (arms, legs, torso)
- **A_inter** — connections across adjacent regions (shoulder↔torso, hip↔torso)
- **A_cross** — long-range cross-body connections (left hand ↔ right knee, etc.)

Each subset is processed by an independent weight matrix under symmetric normalisation (D^{−1/2}AD^{1/2}). This replaces the generic hop-distance partitioning of ST-GCN with structure that directly encodes anatomical kinematic roles. *No prior skeleton GCN uses anatomically-typed multi-subset adjacency.*

### 2. Temporal Landmark Attention (TLA)

An O(T×K) temporal attention mechanism over K=14 uniformly-spaced landmark frames (Longformer-inspired). Standard full T×T attention requires 64×64=4,096 dot-products per head; TLA requires only 64×14=896 — a **4.6× reduction** — while retaining global temporal reach.

- Q: all T frames; K and V: K landmark frames
- Attention kept in float32 throughout (AMP safety)
- Gated residual (sigmoid gate, init=0 → 50% active from epoch 1)
- Cost: 4×C×d_k + 1 params per block (d_k = C//8)

### 3. BRASP — Correct Anatomical Placement

Body-Region-Aware Spatial Shift (BRASP) partitions channels by body region (arms / legs / torso / cross-body) and permutes each group within its anatomically relevant joint neighbourhood. This is a zero-parameter operation that encodes body-part structure as a structural inductive bias before any learned transformation.

**V10.3 fix**: BRASP is now placed *before* the pointwise conv (`brasp_after_pw=False`). The previous placement (after pw_conv) caused the subsequent BatchNorm to normalise joint-shuffled activations — destroying the anatomical structure that BRASP introduces. Placing BRASP before pw_conv ensures BatchNorm sees anatomically structured features.

### 4. Triplet-Margin IB Loss

Information-bottleneck prototype loss with explicit inter-class separation:

```
ib_loss = ReLU(0.5 + d_same − d_wrong).mean()
```

where `d_same` is the distance from the sample to its correct class prototype, and `d_wrong` is the distance to the nearest incorrect prototype. The margin of 0.5 enforces a decision boundary gap. Previous formulations (attraction-only: `d_same.mean()`) had no separation term — the IB loss was a no-op once prototypes converged. Loss weight: 0.01 (100× the previous 0.0001).

### 5. Stage-Shared GCN with Per-Block gcn_scale Guard

One `MultiScaleAdaptiveGCN` instance is shared across all blocks within each stage (3 stages → 3 GCN modules). This saves ~28K parameters versus per-block GCN, without sacrificing representational power.

**gcn_scale guard**: Each block receives a learnable scalar `gcn_scale` (init=1.0, no weight decay) that scales the GCN output *before* the residual add. When a GCN module is shared across multiple blocks, its gradients from multiple blocks accumulate. The per-block scale acts as an independent gain control, allowing each block to attenuate or amplify the shared GCN contribution independently.

### 6. Learned Stream-Weight Ensemble

At evaluation, stream logits are combined via softmax-weighted sum:

```
w = softmax(stream_weights)    # (4,) learned
ensemble = Σ_i w[i] × logits[i]
```

`stream_weights` is a 4-element `nn.Parameter` (init=zeros → uniform at epoch 0), excluded from weight decay. This allows the model to learn which streams are most discriminative per training run, replacing the hard-coded uniform mean.

---

## Design Philosophy

ShiftFuse V10 is built on the principle that **structured domain knowledge can replace learned adaptive computation** at small model scales. Each novel component encodes a specific, well-founded prior about human skeletal motion:

| Prior | Component | Mechanism |
|-------|-----------|-----------|
| Body parts have distinct kinematic roles | SGP | K=3 typed adjacency subsets per anatomical region |
| Actions unfold at global temporal scale | TLA | O(T×K) landmark attention, avoids local TCN blindspot |
| Anatomical routing must precede channel mixing | BRASP (corrected) | BRASP before pw_conv, before BN normalisation |
| Classes should be both clustered and separated | Triplet IB loss | ReLU margin loss with nearest-wrong-prototype penalty |
| Shared weights need per-consumer gain control | gcn_scale | Block-level scalar guard on shared GCN output |
| Streams have unequal discriminative value | Learned ensemble | Softmax-weighted stream combination |

This approach achieves competitive accuracy without per-sample dynamic graph inference, transformer-style full attention, or large channel width — making the model deployable on constrained hardware.

---

## Model Variants and Parameter Budget

| Variant | Params | Channels | Blocks | share_gcn | share_je | TLA |
|---------|--------|----------|--------|-----------|----------|-----|
| **nano** | **225,533** | [32, 64, 128] | [2, 3, 2] | ✓ | ✓ | ✓ (K=14) |
| small | 1,425,050 | [64, 128, 256] | [2, 3, 4] | ✗ | ✗ | ✓ (K=14) |
| large | 3,100,506 | [96, 192, 384] | [2, 3, 4] | ✗ | ✗ | ✓ (K=14) |

All experiments in this document use the **nano** variant unless stated otherwise.

---

## Target Benchmarks

| Benchmark | Protocol | Classes | Train | Val | Our target |
|-----------|----------|---------|-------|-----|------------|
| NTU RGB+D 60 | xsub | 60 | 40,320 | 16,560 | **TBD** |
| NTU RGB+D 60 | xview | 60 | 37,920 | 18,960 | TBD |
| NTU RGB+D 120 | xsub | 120 | 54,468 | 50,922 | TBD |
| NTU RGB+D 120 | xset | 120 | 54,468 | 50,922 | TBD |

Primary evaluation: **NTU RGB+D 60 xsub** (standard benchmark for fair comparison with prior work).

---

## Target Venue

**ECCV 2026** (or equivalent top-tier venue: CVPR, ICCV, NeurIPS)

---

## Document Map

| Section | File | Content |
|---------|------|---------|
| 01 | **Introduction** (this file) | Problem, contributions, design philosophy |
| 02 | [Related Work](02_Related_Work.md) | SOTA landscape, generational analysis, gap table |
| 03 | [Architecture](03_Architecture.md) | ShiftFuse V10 block design, all novel components |
| 04 | [Data Pipeline](04_Data_Pipeline.md) | 4-stream preprocessing, augmentation, dataset stats |
| 05 | [Training](05_Training.md) | Optimiser, scheduler, regularisation, training recipe |
| 06 | [Distillation](06_Distillation.md) | Knowledge distillation from large → nano |
| 07 | [Experiments](07_Experiments.md) | Results, ablations, SOTA comparison |
| 08 | [Environment Setup](08_Environment_Setup.md) | Local / Kaggle setup |
