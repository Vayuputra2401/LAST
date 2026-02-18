# LAST v2 Distillation Strategy: VideoMAE -> Skeleton

This document outlines the **Cross-Modal Knowledge Distillation (CMKD)** strategy to transfer the rich, context-aware features of **VideoMAE V2 (RGB)** into our efficient **LAST v2 (Skeleton)** model.

---

## 1. Concept: Why Distill?

*   **Teacher (VideoMAE V2):**
    *   **Input:** RGB Video (Pixels).
    *   **Pros:** Sees context (clothing, objects, environment). Very robust.
    *   **Cons:** Heavy (ViT-B/L), slow inference (GFLOPs is huge).
*   **Student (LAST v2):**
    *   **Input:** Skeleton (Coordinates).
    *   **Pros:** Extremely light, fast inference.
    *   **Cons:** Blind to context (can't see objects).

**Goal:** Force the Skeleton model to hallucinate/approximate the "contextual features" that the RGB model sees, improving its discriminative power without increasing inference cost.

---

## 2. Distillation Pipeline

### A. Pre-requisites
1.  **Pretrained Teacher:** Download `VideoMAE_V2_ViT_B_1600e.pth` (SOTA on NTU-60/120).
2.  **Paired Data:** We need (RGB Video, Skeleton Sequence) pairs. NTU RGB+D provides both.

### B. Architecture
1.  **Teacher Branch (Frozen):**
    *   Input: Raw Video (sampled, resized to 224x224).
    *   Model: `VideoMAE_V2_ViT_B`.
    *   Output: `Feature_T` (CLS Token, 768-dim).
2.  **Student Branch (Trainable):**
    *   Input: Skeleton (MIB Streams).
    *   Model: `LAST_v2`.
    *   Output: `Feature_S` (before Classifier, 256-dim).
3.  **Projector (Adaptation):**
    *   Since dim(Student) != dim(Teacher), we add a lightweight MLP to project Student features.
    *   `Projected_S = MLP(Feature_S)` -> 768-dim.

### C. Distillation Loss
We minimize the difference between the Teacher's rich representation and the Student's approximation.

$$L_{total} = (1 - \alpha) L_{CE} + \alpha L_{Distill}$$

Where:
*   $L_{CE}$: Standard Cross-Entropy Loss (Student vs Ground Truth).
*   $L_{Distill}$: **MSE Loss** or **Cosine Similarity Loss**.
    *   $L_{MSE} = || Projected_S - Feature_T ||^2$
    *   $L_{Cos} = 1 - CosineSim(Projected_S, Feature_T)$

**Hyperparameters:**
*   $\alpha$: Distillation weight. Start 0.5, decay to 0.

---

## 3. Implementation Steps

1.  **Teacher Setup:**
    *   Install `videomae` library or copy model definition.
    *   Create `scripts/load_teacher.py`.
2.  **Data Loader Update:**
    *   Modify `SkeletonDataset` to optionally load RGB frames (slow I/O!).
    *   *Optimization:* Pre-compute Teacher Features. Run VideoMAE *once* on dataset, save `teacher_features_train.npy` (Dataset Size x 768). Then load this during training. **This saves massive compute.**
3.  **Training Update:**
    *   Add `distillation_loss` to `Trainer`.

---

## 4. Expected Gains
*   **Accuracy:** +1-2% on difficult classes (interactions with objects).
*   **Robustness:** +3-5% on "noisy skeleton" samples (where pose estimation fails).
*   **Cost:** Only increases *Training* time. Inference cost is ZERO.
