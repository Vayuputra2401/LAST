# LAST v2: Next-Generation Architecture

**Goal:** Surpass EfficientGCN in accuracy while matching its efficiency (Lightweight).
**Core Philosophy:** "Efficiency via Separable Convs (EffGCN) + Robustness via Masked Modeling (MaskCLR)."

---

## 1. High-Level Architecture
The LAST v2 model is a hybrid **Graph-Convolutional** and **Attention-Based** network. It discards the pure "Transformer" label for a more efficient **CNN-GCN-Attention** hybrid, proven to be SOTA for skeletons.

### Key Components
1.  **Multi-Input Branches (MIB):** We do not just feed "Joints". We feed 3 streams:
    *   **J-Stream:** Relative Joint Positions.
    *   **V-Stream:** Joint Velocities ($X_{t+1} - X_t$).
    *   **B-Stream:** Bone Vectors ($X_{child} - X_{parent}$).
2.  **Backbone:** `EffGCN-Block`.
    *   **Spatial:** Depthwise Separable GCN (DS-GCN).
    *   **Temporal:** Depthwise Separable TCN (DS-TCN) with dilation.
    *   **Attention:** **ST-JointAtt** (Spatial-Temporal Joint Attention) - highlights critical joints dynamically.
    *   **Squeeze-and-Excitation (SE):** Channel attention.
3.  **Head:**
    *   Global Average Pooling.
    *   FC Classification Head.

### Model Variants (Compound Scaling)
We scale depth (blocks) and width (channels) to create 3 variants.
**Crucial:** We use **Weight Sharing** across the 3 input streams (Joint, Velocity, Bone) to keep parameters low.

| Model Variant | Blocks (Stages) | Channels (C1, C2, C3) | Est. Params (Shared) | Target Hardware | EfficientGCN Equiv |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **LAST-v2-Small** | 3 (10 blocks) | [64, 128, 256] | **~0.35M** | Mobile / Edge | EffGCN-B0 (0.3M) |
| **LAST-v2-Base** | 4 (14 blocks) | [96, 192, 384] | **~1.0M** | Server / GPU | EffGCN-B2 (1.0M) |
| **LAST-v2-Large** | 6 (18 blocks) | [128, 256, 512] | **~2.5M** | Research SOTA | EffGCN-B4 (2.5M) |

**Layer Configuration (Base Example):**
*   **Stage 1 (Local Motion):** 4 Blocks (TCN-only). $C=96$.
*   **Stage 2 (Mid-Level):** 5 Blocks (Hybrid). $C=192$.
*   **Stage 3 (Global Context):** 5 Blocks (Attention-heavy). $C=384$.
*   **Head:** Global Average Pooling -> FC.
*   **MIB Strategy:** 3 Streams (J, V, B) pass through the *same* backbone weights. We sum the scores at the end. Parameters = 1x Backbone. Inference = 3x Compute (parallelizable).

---

## 4. Implementation Strategy (Phased)

### Phase 1: Pure Supervised (Baseline)
Train all 3 variants on NTU-60 using standard Cross-Entropy loss. This establishes our benchmark against EfficientGCN.

### Phase 2: Self-Supervised (MaskCLR)
Pretrain the best-performing variant using Masked Modeling to boost robustness and accuracy further.
