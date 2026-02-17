# Pretraining Strategy Analysis

## Proposed Strategy
The user proposed a 3-step pretraining strategy:
1.  **Proxy Tasks:** Temporal shuffling (order prediction) and Motion prediction (future frame).
2.  **Contrastive Pretraining (InfoNCE):** Augmenting views ($A_1, A_2$) and maximizing agreement.
3.  **Fine-Tuning:** Linear probe (frozen backbone) followed by full fine-tuning.

## Analysis

### 1. Proxy Tasks (Temporal Shuffle / Prediction)
*   **Assessment:** Older method (circa 2017-2019).
*   **Why It Works:** Forces the model to understand causality and temporal coherence.
*   **Relevance to LAST:**
    *   LAST uses **TSM** (Temporal Shift Module), which is explicitly designed to mix past/future frames. Temporal shuffling might confuse TSM or force it to learn "unnatural" features.
    *   **Motion Prediction:** Very strong. Matches modern "Masked Modeling" trends.
*   **Verdict:** Skip strict shuffling. Focus on **Motion Prediction** (generating future frames from past) or **Masking** (filling in gaps).

### 2. Contrastive Pretraining (InfoNCE)
*   **Assessment:** The Gold Standard (SimCLR, MoCo, SkeletonGCL).
*   **Why It Works:** Directly tackles the main problem in skeletal action recognition: **View Invariance**. "Drinking water" looks different from the side vs. front, but the semantic meaning is identical.
*   **Relevance to LAST:**
    *   NTU-120 has 106 subjects and 32 setups. Cross-subject variance is huge.
    *   Contrastive learning is the **best way** to make the model robust to new subjects.
*   **Verdict:** **Highly Recommended.** This is the single most effective way to boost performance if you have the compute time.

### 3. Fine-Tuning Strategy
*   **Assessment:** Standard Transfer Learning.
*   **Why It Works:** Prevents "catastrophic forgetting" of the pretrained features. If you update all weights immediately with a high LR, the model might "forget" the rich features it learned in pretraining and just overfit to the labels.
*   **Verdict:** **Essential.** Always freeze backbone for ~5 epochs when transferring.

## Feasibility for LAST
*   **NTU-120 Availability:** `configs/data/ntu120.yaml` exists. You can use this larger dataset for pretraining as suggested.
*   **Architecture:** LAST is modular. We can easily swap the classification head for a projection head (for InfoNCE) or a decoder (for prediction).

## Implementation Roadmap
This is a significant undertaking (approx. 2-3 days of coding & debugging).

1.  **Create `pretrain.py`:** A training loop that ignores labels and uses pairs of augmented inputs.
2.  **Augmentation Pipeline:** Need strong augmentations (Shear, Crop, Jitter) to make $A_1$ and $A_2$ sufficiently different.
3.  **Loss Function:** Implement `InfoNCELoss`.
4.  **Workflow:**
    *   Train `LAST-Base` on NTU-120 (Unsupervised).
    *   Save weights.
    *   Load weights into `train.py` for NTU-60 (Supervised).

## Recommended Action
**Start with Supervised Training first.**
If we can get ~85% accuracy with standard Supervisor learning (which is typical for A-GCN models), we might not *need* this complexity.
**Only implement this if:**
1.  Accuracy plateaus below 80%.
2.  We see massive overfitting (Train 99%, Val 60%).
3.  We want to publish/beat SOTA.
