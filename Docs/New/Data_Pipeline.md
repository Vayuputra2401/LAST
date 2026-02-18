# LAST v2 Data Pipeline: Multi-Input Branch (MIB)

This document details the data preprocessing and loading strategy for LAST v2.

---

## 1. MIB Concept (Multi-Input Branch)

Instead of feeding raw joint coordinates ($X, Y, Z$) directly, we decompose the skeleton into three distinct "views" or "streams" to capture different aspects of motion. This is standard in EfficientGCN.

### Stream 1: Joint (J-Stream)
*   **Definition:** Relative positions of joints.
*   **Formula:** $J_t = P_t - P_{center}$, where $P_{center}$ is "SpineBase" (Joint 0).
*   **Shape:** $(3, T, V, M)$.

### Stream 2: Velocity (V-Stream)
*   **Definition:** Temporal difference between frames. Captures speed and direction of movement.
*   **Formula:** $V_t = P_{t+1} - P_t$. Pad last frame with 0.
*   **Shape:** $(3, T, V, M)$.

### Stream 3: Bone (B-Stream)
*   **Definition:** Vector connecting two physical joints. Captures limb length and orientation.
*   **Formula:** $B_t = P_{child} - P_{parent}$.
*   **Shape:** $(3, T, V, M)$.

---

## 2. Normalization Strategy (Standard SOTA)

We apply strict normalization during preprocessing to ensure invariance to camera angle and subject size.

### Step 1: Pre-Normalization (View Invariance)
For *every frame* $t$:
1.  **Translate to Origin:** Subtract $P_{SpineBase}$ from all joints.
2.  **Rotate to Front View:**
    *   Calculate vector $V_{shoulder} = P_{ShoulderLeft} - P_{ShoulderRight}$.
    *   Calculate projection of $V_{shoulder}$ onto XY plane.
    *   Rotate entire skeleton around Z-axis so $V_{shoulder}$ aligns with X-axis.
3.  **Scale Normalization:**
    *   Calculate average length of "Spine" bone across dataset.
    *   Scale every skeleton so its average spine length equals 1.0.

### Step 2: Input Normalization (Mean/Std)
*   Calculate global Mean ($\mu$) and Std ($\sigma$) for each channel (X, Y, Z) across the entire training set (after Step 1).
*   During training: $Input = (Input - \mu) / \sigma$.

---

## 3. Implementation Plan

1.  **Update `scripts/preprocess_data.py`:**
    *   Add MIB generation logic.
    *   Save 3 separate `.npy` files: `train_joint.npy`, `train_velocity.npy`, `train_bone.npy`.
2.  **Create `src/data/normalization.py`:**
    *   Implement `align_to_spine_base()`.
    *   Implement `rotate_to_front()`.
3.  **Update `src/data/dataset.py`:**
    *   Load all 3 streams if `data_type='mib'`.
    *   Return dictionary: `{'joint': ..., 'velocity': ..., 'bone': ...}`.
