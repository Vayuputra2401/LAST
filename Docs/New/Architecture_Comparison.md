# LAST v2 vs. EfficientGCN: A Detailed Comparison

EfficientGCN is the current State-of-the-Art (SOTA) for efficiency. LAST v2 adopts its core strengths but introduces critical upgrades to solve its weaknesses (robustness and global context).

---

## 1. High-Level Difference

| Feature | EfficientGCN (Baseline) | LAST v2 (Ours) |
| :--- | :--- | :--- |
| **Philosophy** | **Pure CNN** (Separable Convs) | **Hybrid** (CNN + Linear Attention) |
| **Temporal Modeling** | Local TCN + Dilation (Limited Receptive Field) | **Hybrid Strategy** (Local TCN + Global Linear Attn) |
| **Training Paradigm** | Supervised Only (Labels) | **Self-Supervised (MaskCLR)** + Supervised |
| **Context Awareness** | Blind (Skeleton Coordinates Only) | **RGB-Aware** (via VideoMAE Distillation) |
| **Data Augmentation** | Standard Affine Transforms | **Attention-Guided Masking** + Part-Drop |

---

## 2. Architectural Upgrades

### A. The Temporal Bottleneck fixed
*   **EfficientGCN:** Uses Multi-Scale TCN with dilation. While efficient, convolution is inherently *local*. It takes many deep layers to "see" the relationship between Frame 1 and Frame 300.
*   **LAST v2:** Uses a **Hybrid Block**.
    *   **Early Layers:** Uses TCN (like EffGCN) to extract local motion (e.g., "fast hand wave").
    *   **Deep Layers:** Uses **Linear Attention** (O(T) complexity). Attention is *global*—it instantly relates Frame 1 to Frame 300.
    *   **Result:** Better long-term temporal reasoning without the O(T^2) cost of Transformers.

### B. The Context Blindness fixed
*   **EfficientGCN:** Sees only X,Y,Z points. It cannot distinguish "Holding a Cup" from "Holding a Ball" if the hand pose is identical.
*   **LAST v2:** Uses **Knowledge Distillation**.
    *   **Teacher:** VideoMAE V2 (sees pixels/objects).
    *   **Student:** LAST v2 (sees skeleton).
    *   **Result:** The LAST v2 feature space is forced to align with VideoMAE. The model "hallucinates" the object context.

---

## 3. Training Upgrades (Phase 1 & 3)

### A. MaskCLR Pretraining
*   **EfficientGCN:** Starts from random weights. If labeled data is noisy or scarce, it overfits.
*   **LAST v2:** Pretrains on **masked skeletons**.
    *   Task: "Reconstruct the missing arm from the rest of the body."
    *   Result: The model learns the *grammar* of human motion before it ever sees a label.

### B. MIB Optimization
*   **EfficientGCN:** Uses Multi-Input Branches (Joint, Vel, Bone) but often trains them as separate models and ensembles (3x inference cost).
*   **LAST v2:** We aim for **Early Fusion** or **Parameter Sharing** to keep inference cost lower while retaining the multi-view benefit.
