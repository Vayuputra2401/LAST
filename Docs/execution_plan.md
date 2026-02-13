# Research Execution Plan: Cloud & Schedule

## 1. Cloud Infrastructure Plan (GCP)

To meet the efficiency requirements and deadline, we need a reliable but cost-effective setup. Since "LAST" is a lightweight model, we do **not** need massive A100 clusters.

### Recommended VM Specification
*   **Machine Machine:** `n1-standard-8` (8 vCPUs, 30GB RAM).
    *   *Why:* Preprocessing and dataloading for skeleton data can be CPU bound.
*   **GPU:** **1 x NVIDIA T4** (16GB VRAM).
    *   *Why:* T4 is optimized for INT8/FP16 inference and is very cheap (~$0.35/hr spot). It is more than powerful enough for skeleton-based training (batch sizes usually 64-128).
    *   *Alternative:* If training speed is too slow during Week 2, switch to **V100**.
*   **Disk:**
    *   **Boot Disk:** 100GB (Standard PD) for OS/Packages.
    *   **Data Disk:** **375GB Local SSD (NVMe)**.
    *   *Critical:* You **must** mount the Local SSD and store the `.npy` dataset files there. Reading from standard Persistent Disk (PD) will bottleneck your GPU.

### Environment Setup
*   **OS Image:** Deep Learning VM Image (latest PyTorch version).
*   **Key Libraries:**
    *   `torch`, `torchvision`
    *   `torch-geometric` (for GCN ops)
    *   `thop` (for FLOPs counting)
    *   `tensorboard` (for monitoring)

---

## 2. Research Schedule (Feb 2026)

**Goal:** Complete research, evaluation, and documentation by **Feb 28, 2026**.

### Week 1: Infrastructure & Data Pipeline (Feb 3 - Feb 8)
*   **Objective:** Get a working training loop with a baseline model.
*   **Tasks:**
    1.  Provision GCP VM with T4 GPU.
    2.  Download NTU RGB+D 120 and run preprocessing scripts (Generates `.npy`).
    3.  Implement the **Dataloader** with augmentations (Rotate, Shear).
    4.  **Milestone:** Train a simple baseline (e.g., standard ResNet18-1D or Plain GCN) to verify the pipeline works and loss decreases.

### Week 2: Model Implementation & Distillation (Feb 9 - Feb 15)
*   **Objective:** Implement LAST and existing SOTA checks.
*   **Tasks:**
    1.  Implement **LAST Block**:
        *   Adaptive Graph Conv (A-GCN).
        *   Temporal Shift (TSM).
        *   Linear Attention.
    2.  Implement **VideoMAE Teacher**:
        *   Load pretrained weights.
        *   Setup Distillation Loss (KL Div).
    3.  **Milestone:** Complete first full training run of LAST on NTU-120 (X-Sub split). Target: >85% accuracy.

### Week 3: Optimization & Ablation (Feb 16 - Feb 22)
*   **Objective:** Refine accuracy and prove efficiency.
*   **Tasks:**
    1.  **Ablation Studies:**
        *   Train version w/o TSM (Compare params).
        *   Train version w/o Linear Attention (Compare long-term dependency).
    2.  **Hyperparameter Tuning:** Adjust learning rate, weight decay, and distillation alpha/beta.
    3.  **Secondary Datasets:** Start training on Kinetics-Skeleton if time permits.
    4.  **Milestone:** Achieve target accuracy (~89-90%) and stable convergence.

### Week 4: Final Evaluation & Reporting (Feb 23 - Feb 28)
*   **Objective:** Generate final metrics and artifacts.
*   **Tasks:**
    1.  **Efficiency Benchmarking:**
        *   Measure FPS/Latency on CPU (simulate Edge).
        *   Calculate exact GFLOPs using `thop`.
    2.  **Visualization:** Generate CAMs (Class Activation Maps) for report.
    3.  **Documentation:** Write final experiment report, format results for paper/submission.
    4.  **Wrap-up:** Finalize code repository.

---

## 3. Immediate Next Steps (Today)
1.  **Codebase Init:** Set up the Python project structure (git repo).
2.  **Dataset Request:** Ensure you have access/links to download NTU RGB+D 120.
3.  **GCP Config:** Confirm quota for T4 GPUs in your desired region.
