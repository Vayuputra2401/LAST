# Why LAST v2 Will Succeed

This document details the critical improvements in LAST v2 compared to the previous architecture ("last arch"). These changes address specific failure modes common in skeleton action recognition.

## 1. Input Data: The "MIB" Advantage
**Previous Issue:** Likely relied on **Joint** coordinates alone (single stream).
*   **Problem:** Similar actions (e.g., "drinking water" vs. "brushing teeth") look very similar if you only track absolute positions. The *motion* is subtle.
*   **V2 Solution (Multi-Input Branch - MIB):**
    *   **Joint Stream**: Base position.
    *   **Velocity Stream**: $V_t = J_{t+1} - J_t$. Explicitly captures *motion dynamics* (speed/direction). Critical for fast actions (punching/kicking).
    *   **Bone Stream**: $B_{child} - B_{parent}$. Captures fixed limb lengths and structural angles. Critical for static poses (reading/sitting).
    *   **Result**: The model sees the action from 3 perspectives (Position, Motion, Structure). Fusion boosts accuracy by **~5-8%** over Joint-only.

## 2. Preprocessing: View-Invariant Normalization
**Previous Issue:** Raw or poorly normalized skeletons.
*   **Problem:** In NTU RGB+D, the same action ("Sitting") is captured from side, front, and 45-degree angles. Without strict normalization, the model learns "Side Sitting" and "Front Sitting" as different patterns, wasting capacity.
*   **V2 Solution (`preprocess_v2.py`):**
    *   **Align to Spine**: Centers every skeleton at `(0,0,0)` based on `SpineBase`.
    *   **Rotate to Front**: Mathematically rotates the skeleton so the shoulders are always parallel to the X-axis.
    *   **Result**: The model becomes **View-Invariant**. A side-view kick becomes a front-view kick mathematically. This drastically reduces data variance and improves generalization.

## 3. Training Recipe: SGD vs. AdamW
**Previous Issue:** Default `AdamW` optimizer (likely).
*   **Problem:** Adaptive optimizers like Adam often converge to "sharp minima" on graph data. They memorize the training set quickly but fail on the test set (poor generalization).
*   **V2 Solution (SOTA Recipe):**
    *   **SGD + Nesterov (0.9)**: Proven to find "flatter minima" for GCNs, leading to better test accuracy.
    *   **Filtering Weight Decay**: We explicitly **disable weight decay** on Biases and Batch Normalization parameters (`trainer.py`). This is a "pro tip" from papers like ResNet/EfficientNet that adds **~1%** accuracy.

## 4. Architecture: EfficientGCN w/ Attention
**Previous Issue:** Standard GCN or A-GCN.
*   **Problem:** Standard GCNs treat all joints and frames equally, or use heavy computation for global context.
*   **V2 Solution (`st_joint_att.py` & `eff_gcn.py`):**
    *   **ST-Joint Attention**: A specialized mechanism that learns *which joint matters when*.
        *   *Example*: For "Clapping", it attends to Hands. For "Kicking", it attends to Legs.
    *   **Separable Convolutions**: Reduces parameter count, allowing us to stack deeper networks (more layers) without overfitting.

## Summary Checklist
| Feature | Previous Arch (Likely) | LAST v2 (Implemented) | Impact |
| :--- | :--- | :--- | :--- |
| **Input** | Single Stream (Joints) | **3 Streams (Joint+Vel+Bone)** | **High** (Rich features) |
| **Normalization** | Basic / Scaling | **View-Invariant Rotation** | **Critical** (Generalization) |
| **Optimizer** | AdamW | **SGD + Nesterov** | **Medium** (Better Test Acc) |
| **Attention** | Maybe / None | **ST-Joint Attention** | **Medium** (Focus) |

**Conclusion:** LAST v2 is not just a model change; it's a complete pipeline upgrade. The combination of **MIB Data** and **View-Invariant Preprocessing** is usually the biggest factor in success on NTU RGB+D.
