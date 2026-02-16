# Architecture Deep Dive & SOTA Comparison

## 1. Are the Blocks Implemented Properly?

**Yes, the components are mathematically sound and correctly implemented.**

### 1.1 TSM (Temporal Shift Module)
*   **Code Check:** `x[:, :n_f]` (Group 1), `shift(x)` along `dim=2` (Time).
*   **Correctness:** Valid. It correctly performs the "Shift" operation described in the ICCV 2019 paper. It moves features in time without adding parameters.
*   **Comparison:** SOTA models like **Shift-GCN** use similar shift operations but often combined with adaptive receptive fields. Your implementation is the standard, efficient version.

### 1.2 Linear Attention
*   **Code Check:** `ELU() + 1` kernel, `Q(K^T V)` ordering.
*   **Correctness:** Valid. This matches the "Transformers are RNNs" (2020) specification for $O(T)$ complexity.
*   **Comparison:** Most SOTA skeleton models (EfficientGCN, MS-G3D) do **not** use Global Attention because standard attention is $O(T^2)$ (too slow). Your Linear Attention is a **unique advantage**, allowing global context with efficiency.

### 1.3 A-GCN (Adaptive GCN)
*   **Code Check:** `Sum(Physical, Learned, Dynamic)`.
*   **Correctness:** Valid. This follows the **2s-AGCN** design pattern (Physical + Learned). Adding "Dynamic" (sample-dependent) makes it more expressive, similar to **CTR-GCN**.
*   **Flaw detected?** No logic flaw, but it is computationally heavier than EfficientGCN's separable convolutions.

---

## 2. Is it Too Shallow?

**Yes, compared to SOTA, LAST-Base is shallow.**

| Model | Layers / Blocks | Channels |
| :--- | :--- | :--- |
| **ST-GCN** (Baseline) | **10 Blocks** | 64 (x4) → 128 (x3) → 256 (x3) |
| **EfficientGCN-B0** | **10 Blocks** | Scaling width/depth |
| **MS-G3D** | **Multiple Scales** | Deep multi-branch |
| **LAST-Base** (Yours) | **4 Blocks** | 64 → 128 → 128 → 256 |

**Analysis:**
*   4 Batches of graph convolutions might be **insufficient** to capture complex high-level semantics (like "giving something to other person").
*   However, GCNs suffer from "Over-smoothing" (features becoming identical) if too deep.
*   **Recommendation:** If accuracy is the priority over speed, **increase depth to 9-10 blocks** (e.g., `[64, 64, 64, 128, 128, 128, 256, 256, 256]`).

---

## 3. What is MaskGL?

**MaskGL (Masked Graph Learning)** is a technique to handle **Novel Motion Patterns**.

*   **Core Idea:** In real world, test data has noise/motions never seen in training.
*   **Mechanism:** It learns a binary **Mask** on the Adjacency Matrix.
    *   It identifies "Action-Agnostic" (noise) joints and masks them out.
    *   It focuses only on "Action-Specific" joints.
*   **Relevance:** It's great for **Robustness** (cleaner inputs), but not necessarily a replacement for the backbone (GCN/Transformer). It's an enhancement.

---

## 4. Multi-Stream Parameters (Joints + Bones + Velocity)

Multi-stream fusion is the standard way to boost accuracy by **+5-8%**.

**Parameter Calculation:**
*   **Stream 1 (Joints):** `P` parameters (e.g., 0.7M).
*   **Stream 2 (Bones):** Identical network structure. `P` parameters.
*   **Stream 3 (Velocity):** Identical network structure. `P` parameters.
*   **Fusion:** Simple average (0 params) or small MLP (<0.01M).

**Total Parameters:** ~`3 * P`.

| Model | Single Stream | 3-Stream (J+B+V) |
| :--- | :--- | :--- |
| **EfficientGCN-B0** | 0.29M | **0.87M** |
| **LAST-Base** | 0.69M | **2.07M** |
| **MS-G3D** | 3.2M | **9.6M** |

**Conclusion:** Even with 3 streams, LAST (~2M) is still lighter than ST-GCN (3.1M) and MS-G3D (9.6M). It serves its purpose as a "Lightweight" model capable of SOTA accuracy via fusion.

---

## 5. EfficientGCN vs. LAST: Flaws & Comparison

### EfficientGCN Architecture (The "Competitor")
*   **Philosophy:** "Efficiency via Separable Convs".
*   **Blocks:** Alternating **Spatial GCN** and **Temporal GCN** (Separated).
*   **Conv Type:** **Depthwise Separable Convolution** (Standard in MobileNets). drastically reduces parameters.
*   **Optimization:** Uses SGD. Works because it's purely CNN-based.

### LAST Architecture (Ours)
*   **Philosophy:** "Efficiency via Linear Attention + TSM".
*   **Blocks:** Unified Spatial (GCN) + Temporal (TSM/Attn).
*   **Conv Type:** Standard Convolution (in AGCN). **Heavier** than EfficientGCN.
*   **Optimization:** **Requires AdamW** (Transformer nature).

### Critical Flaws in LAST (vs EfficientGCN)
1.  **Optimizer Mismatch (FIXED):** We tried to train a Transformer (LAST) with a CNN optimizer (SGD). EfficientGCN uses SGD because it IS a CNN. This was the root cause of "Same results nothing".
2.  **Conv Efficiency:** Our AGCN uses standard convolutions (`nn.Conv2d`). EfficientGCN uses Separable Convs.
    *   *Improvement:* We could replace `nn.Conv2d` in `agcn.py` with Depthwise-Separable blocks to cut param count by ~4x.
3.  **Depth:** Shallow (4 blocks) vs Deep (10 blocks).

### Review Report Summary
*   **Blocks:** Valid and correct.
*   **Depth:** Shallow. Recommended to increase if accuracy plateau is reached.
*   **Optimizer:** Was the critical failure point. Switched to AdamW.
*   **Potential Upgrade:** Replace standard Convs with Separable Convs for parameter reduction.
