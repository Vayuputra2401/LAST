# LAST v2 Training Pipeline: Phased Approach

We follow a strict 2-Phase Strategy to ensure solid baselines before adding complexity.

---

## Phase 1: Pure Supervised (The Benchmark)

**Goal:** Establish the accuracy of LAST-v2 Small/Base/Large on NTU-60 using standard labels.
**Why:** This proves the architecture works *before* we add pretraining.

### Hyperparameters
*   **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-2).
*   **Scheduler:** Cosine Annealing (Warmup 5 epochs).
*   **Loss:** CrossEntropy + Label Smoothing (0.1).
*   **Epochs:** 70-100.

### Augmentations (SOTA Standard)
1.  **Random Rotation:** $\pm 15^\circ$.
2.  **Random Scaling:** $\pm 10\%$.
3.  **Shear:** $\pm 15^\circ$.
4.  **Skeleton Part-Drop:** Randomly zero out one limb (e.g., Left Arm) in 5% of batches.

---

## Phase 2: Self-Supervised Pretraining (MaskCLR)

**Goal:** Boost the best Phase 1 model by pretraining on *masked* data.
**Data:** NTU-60 (or NTU-120 if available).

### Concept
We use **Attention-Guided Probabilistic Masking**.
1.  **Forward 1:** Pass original skeleton. Get Attention Map.
2.  **Masking:** Mask the *Top 30%* activated joints.
3.  **Forward 2:** Pass masked skeleton.
4.  **Loss:** Minimize distance between features of Original and Masked.

### Fine-Tuning
After pretraining, we load the weights and run **Phase 1** training again (with lower LR) to adapt to the class labels.

