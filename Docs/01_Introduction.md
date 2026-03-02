# 01 — Introduction

## Problem Statement

Skeleton-based action recognition from 3D joint sequences is a core task in human activity understanding, with direct applications in human-computer interaction, surveillance, rehabilitation monitoring, and sports analytics. Skeleton data offers compelling advantages over RGB video: it is compact, viewpoint-invariant, privacy-preserving, and robust to variations in illumination, background clutter, and appearance.

Graph Convolutional Networks (GCNs) have emerged as the dominant paradigm for skeleton-based recognition, modelling the human body as a spatio-temporal graph where joints are nodes and bones are edges. However, the field faces a fundamental **efficiency-accuracy tension**: state-of-the-art methods (CTR-GCN, InfoGCN, HD-GCN, HI-GCN) achieve 92--93% top-1 accuracy on NTU RGB+D 60 but require 1.5--3.5M parameters and substantial per-sample adaptive computation (learned adjacencies, attention mechanisms, dynamic graph inference). At the other extreme, EfficientGCN-B0 achieves 90.2% at 290K parameters but provides no novel architectural contributions beyond compound scaling of a standard GCN.

No existing work simultaneously achieves (i) sub-250K parameter count, (ii) competitive accuracy approaching the 90% mark, and (iii) introduces genuinely novel architectural primitives with theoretical motivation grounded in human skeletal structure.

---

## Why Multiple Streams?

Human motion is a multi-faceted signal. A single coordinate representation cannot capture all discriminative information. Following the multi-stream paradigm established by 2s-AGCN and refined by EfficientGCN, we decompose each skeleton sequence into four complementary kinematic streams:

| Stream | Key | Derivation | Signal |
|--------|-----|------------|--------|
| Joint | `joint` | 3D coordinates centred at spine base | Absolute pose configuration |
| Velocity | `velocity` | J(t+1) - J(t) (temporal difference) | Motion speed and direction |
| Bone | `bone` | J(child) - J(parent) per skeleton edge | Limb orientation and length |
| Bone velocity | `bone_velocity` | B(t+1) - B(t) (temporal diff of bone) | Limb angular velocity |

All four streams are fused at the input level via a learned channel-wise concatenation, enabling a single lightweight backbone to jointly reason over complementary kinematic cues.

---

## Contributions

This work presents **LAST** (Lightweight Action recognition via Shift-based Topology), a two-model framework comprising a high-accuracy research model and a novel edge-deployable architecture:

### 1. LAST-Lite (ShiftFuse-GCN) — Main Contribution

A fixed-computation skeleton GCN family achieving competitive accuracy at sub-250K parameters. LAST-Lite introduces **four novel architectural primitives**, each addressing a distinct modelling gap in prior skeleton GCNs:

- **BRASP (Body-Region-Aware Spatial Shift)**: Anatomically-partitioned zero-parameter spatial mixing. Channels are grouped by body region (arms, legs, torso, cross-body) and shifted only among graph neighbours within their assigned region. This encodes body-part semantics as a structural inductive bias at zero computational cost. *No prior skeleton GCN uses anatomy-guided channel routing.*

- **BSE (Bilateral Symmetry Encoding)**: Explicit modelling of left-right skeletal symmetry. For each of 10 symmetric joint pairs, BSE computes the bilateral difference and its temporal derivative, weights them per-channel, and injects the signal antisymmetrically (left joints receive +signal, right joints receive -signal). This enables the network to distinguish symmetric (clapping), anti-phase (walking), and asymmetric (drinking) actions. Cost: 2C + 1 parameters per block. *No prior work explicitly models bilateral symmetry as a learned feature.*

- **FDCR (Frozen DCT Frequency Routing)**: Data-independent frequency-domain channel specialisation. Each channel learns a sigmoid mask over DCT frequency bins, allowing different channels to specialise for low-frequency (slow, periodic) vs. high-frequency (fast, impulsive) temporal patterns. The DCT basis is frozen (zero parameters), and the learnable mask costs C x T parameters per block. *Frequency-domain processing is unexplored in skeleton GCNs.*

- **StaticGCN with A_learned**: A lightweight stage-shared graph convolution combining K fixed adjacency subsets with a trainable topology correction matrix (V x V = 625 parameters). One StaticGCN instance is shared across all blocks within a stage, eliminating redundant graph parameters. The learned adjacency uses symmetric normalisation (D^{-1/2}|A|D^{-1/2}) and is zero-initialised for stable training.

Two model variants are provided:

| Variant | Parameters | Architecture |
|---------|-----------|--------------|
| LAST-Lite nano | 80,234 | channels=[32,48,64], blocks=[1,1,1] |
| LAST-Lite small | 247,548 | channels=[48,72,96], blocks=[1,2,2] |

### 2. LAST-Base — High-Accuracy Research Model (Planned)

A large-capacity model (~4.2M params per stream, 4-stream ensemble) designed to beat the current SOTA (HI-GCN, 93.3% NTU-60 xsub). LAST-Base integrates cross-temporal topology refinement, action-prototype graph conditioning, frequency-domain gating, and partitioned spatio-temporal attention into a unified block. It also serves as the teacher for knowledge distillation into LAST-Lite.

### 3. Knowledge Distillation Pipeline (Planned)

A structured distillation path from LAST-Base to LAST-Lite, with optional MaskCLR self-supervised pretraining for additional representation learning at small scale.

---

## Design Philosophy

LAST-Lite is built on the principle that **structural inductive biases can replace learned adaptive mechanisms** at small model scales. Where prior lightweight GCNs (EfficientGCN, Shift-GCN) use generic modules scaled down from larger models, LAST-Lite introduces domain-specific primitives that encode knowledge about human skeletal structure directly into the architecture:

1. **Body regions matter** (BRASP): Arms, legs, and torso have distinct kinematic roles. Routing information flow by body part is more informative than random channel mixing.

2. **Bilateral symmetry is discriminative** (BSE): The human body is bilaterally symmetric, and the degree and dynamics of left-right symmetry directly distinguish action categories.

3. **Temporal frequency separates action types** (FDCR): Periodic actions (walking, waving) and impulsive actions (punching, throwing) occupy different frequency bands. Channel-wise frequency specialisation captures this without per-sample computation.

4. **Graph topology can be shared and corrected** (StaticGCN): A single graph convolution per stage, shared across blocks, is sufficient when augmented with a small learnable correction matrix.

These four principles yield a model family that achieves ~80% top-1 accuracy on NTU-60 xsub in early standalone training (Round 2), with substantial headroom expected from corrected hyperparameters (Round 3) and knowledge distillation.

---

## Target Venue

**ECCV 2026** (or equivalent top-tier venue)

---

## Document Map

| Section | File | Content |
|---------|------|---------|
| 02 | [Related Work](02_Related_Work.md) | SOTA landscape, generational analysis |
| 03 | [Architecture](03_Architecture.md) | LAST-Lite block design, LAST-Base overview |
| 04 | [Data Pipeline](04_Data_Pipeline.md) | 4-stream preprocessing, augmentation |
| 05 | [Training](05_Training.md) | Optimiser, scheduler, regularisation |
| 06 | [Distillation](06_Distillation.md) | KD + MaskCLR pretraining plan |
| 07 | [Experiments](07_Experiments.md) | Results, ablations, SOTA comparison |
| 08 | [Environment Setup](08_Environment_Setup.md) | Local / Kaggle setup |
| A1 | [Experiment-LAST-Lite](Experiment-LAST-Lite.md) | Detailed LAST-Lite experiment log |
| A2 | [Experiment-LAST-Base](Experiment-LAST-Base.md) | Detailed LAST-Base experiment log |
