This report outlines the submission strategy, theoretical validation, and competitive landscape for your framework, **LAST (Lightweight Adaptive-Shift Transformer)**, targeted for **ECCV 2026**.

## **1. ECCV 2026 Submission Timeline**

As of today, January 29, 2026, you are in the prime window for submission.

* **Abstract Registration:** February 26, 2026.
* **Full Paper Submission:** **March 5, 2026**.
* **Supplementary Materials:** March 12, 2026.
* **Conference Dates:** September 8–13, 2026 (Malmö, Sweden).

---

## **2. Theoretical Validation: Why LAST Works**

The core innovation of LAST is its **hybrid efficiency triad**. Based on current research, here is the realistic mathematical and theoretical basis for your proposed modules:

### **A. Adaptive Spatial Modeling (A-GCN)**

* **The Problem:** Traditional GCNs (like ST-GCN) use a fixed adjacency matrix  based on physical bone connections. This fails to capture "semantic" connections (e.g., the relationship between "hand" and "head" during a phone call).
* **The LAST Solution:** By using a learned, dynamic matrix , the model captures global context without the overhead of computing a new graph for every sample, which is a hallmark of recent SOTA models like **MSA-GCN**.

### **B. Zero-Parameter Temporal Modeling (TSM)**

* **The Calculation:** 3D Convolutions or even 1D temporal convolutions have a computational cost of , where  is kernel size.
* **Efficiency:** The **Temporal Shift Module (TSM)** achieves temporal information exchange by simply shifting a subset of channels along the time dimension. This effectively costs **0 FLOPs** and **0 parameters**, making it superior to the 12.2 GFLOPs typically consumed by the temporal branch of an ST-GCN.

### **C. Linearized Global Context (Linear Transformer)**

* **Complexity Shift:** Standard Transformers have a quadratic complexity of **** relative to sequence length .
* **Real-World Impact:** For a 30-second action at 30 FPS (), a standard transformer requires ~810,000 attention operations per head. LAST’s **Linear Attention** reduces this to ****, or 900 operations, enabling the "Real-Time" claim on edge devices with  MB of RAM.

---

## **3. Benchmarks & Top Models to Compare Against**

To be competitive for ECCV, you must beat or closely match the efficiency-accuracy trade-off of these models:

### **Models to Compare (Baseline & SOTA)**

| Model Type | Key Competitors | Rationale for Comparison |
| --- | --- | --- |
| **High Performance** | **InfoGCN / HD-GCN** | Current SOTA on NTU-120. You need to show your accuracy is within 1–3% of these while being 10x faster. |
| **Efficiency SOTA** | **Shift-GCN** | The benchmark for "Shift" operations in GCNs. It consumes ~0.7 GFLOPs. |
| **Hybrid SOTA** | **MSA-GCN** | A 2025/2026 leader that also combines adaptive graphs with multi-scale temporal modeling. |
| **Edge Baselines** | **MobileNetV3 + TSM** | A classic "lightweight" video baseline to prove your skeleton-based approach is superior for edge. |

### **Target Accuracy Benchmarks**

You should target the **NTU RGB+D 120 (Cross-Subject)** benchmark.

* **Current SOTA:** .
* **Your Target:**  with  GFLOP complexity.

---

## **4. Dataset Targeting Strategy**

For an A* conference like ECCV, using a single dataset is insufficient. You should target:

1. **NTU RGB+D 120 (Primary):** The gold standard for skeleton-based HAR.
2. **Kinetics-Skeleton (Secondary):** To prove generalization on large-scale, "in-the-wild" data.
3. **Northwestern-UCLA:** A smaller dataset often used to prove robustness in viewpoint variations.

---

## **5. Detailed Model Roadmap**

### **The "Look" of the Model (Architecture)**

Your LAST block should follow a **Bottleneck Design**:

1. **Skeleton Input:** Use **MediaPipe** for real-time extraction to ensure the end-to-end latency is measured.
2. **Adaptive GCN Layer:** Implement the "Student" base using **PyTorch Geometric**.
3. **Shift-Attention Sandwich:** Place the TSM layer *before* the Linear Attention to "mix" temporal features locally before the Transformer captures global dependencies.
4. **Knowledge Distillation (KD):** Use **VideoMAE V2** as your teacher. The student (LAST) should minimize the KL-Divergence between its logits and the teacher's heavy RGB-based features.

### **Calculated Evaluation Metrics**

* **Latency:** Must be **** on a mobile CPU (Snapdragon) or Raspberry Pi 4 to claim "Real-Time".
* **Model Size:** Target ** MB** (standard for mobile deployment).
* **Complexity:** Target ** GFLOP** per action sample.

Would you like me to help you draft the **"Ablation Study"** plan to prove which of your three pillars—A-GCN, TSM, or Linear Attention—contributes most to the accuracy gains?


The technical pipeline for **LAST** is designed as a streamlined, high-efficiency engine that moves from sparse spatial data to global temporal reasoning. To meet the requirements for **ECCV**, the architecture must be modular, allowing for clear ablation studies of its three core components.

### **Stage 1: Front-End Skeleton Extraction**

The pipeline begins with raw video frames, but to avoid the computational cost of 3D-CNNs, it immediately discards the RGB data.

* **Extraction:** MediaPipe is used to extract a skeletal representation  where  is the number of joints (e.g., 25 for NTU RGB+D 120) and  represents the 3D coordinates.
* **Data Format:** The input tensor for the model is shaped as , representing Batch, Channels (3), Time frames, and Joints.

### **Stage 2: The LAST Block (Spatial-Temporal Fusion)**

The core of the model consists of several stacked **LAST Blocks**. Each block performs a three-step operation to process movement:

#### **1. Adaptive Spatial Modeling (A-GCN)**

Instead of relying on a static physical skeleton, the model uses a dynamic adjacency matrix.

* **The Operation:** It combines a fixed physical matrix (), a globally learned matrix (), and a sample-dependent matrix ().
* **Result:** This allows the model to learn "virtual edges," such as the relationship between a hand and the head during a "phone call" action, which a standard fixed-graph model would miss.

#### **2. Zero-Parameter Temporal Modeling (TSM)**

Immediately following the spatial convolution, the features are passed through a Temporal Shift Module.

* **The Operation:** A portion of the feature channels is shifted forward and backward along the time dimension ().
* **Result:** This enables information exchange between neighboring frames without adding any new parameters or Floating Point Operations (FLOPs), maintaining a  computational cost for this step.

#### **3. Linearized Global Context (Linear Attention)**

To capture long-range dependencies across the entire video sequence (e.g., actions lasting 30+ seconds), a Transformer head is applied.

* **The Operation:** Traditional quadratic attention  is replaced with **Linear Attention**, where the complexity is reduced to .
* **Result:** This ensures the model remains lightweight enough for edge devices with limited RAM, such as a Raspberry Pi or mobile CPU.

### **Stage 3: The Teacher-Student Training Pipeline**

To ensure the lightweight student model (LAST) performs as well as heavy RGB-based models, you will use **Knowledge Distillation**.

* **The Teacher:** A heavy-duty RGB model like **VideoMAE V2** processes the full video to extract rich semantic features.
* **The Student:** LAST (the student) is trained to match the output features of the teacher using only skeletal data.
* **The Goal:** This transfers complex visual understanding into the lightweight architecture, allowing it to reach an expected accuracy of **** on the NTU-120 dataset.

### **Stage 4: Deployment & Inference**

The finalized model is exported via **ONNX** or **TFLite** to target edge hardware.

* **Target Latency:** The goal is an inference time of **** per action sample.
* **Resource Footprint:** By focusing on memory-efficient operations and skeletal data, the model avoids the "massive GPU memory" requirements of traditional 3D-CNNs.

This pipeline ensures that every operation—from graph convolution to temporal shifting—is mathematically optimized for linear time complexity.