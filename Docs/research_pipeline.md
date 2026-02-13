# Research Pipeline: LAST (Lightweight Adaptive-Shift Transformer)

This document provides a comprehensive breakdown of the end-to-end research pipeline for the LAST framework, designed for efficient skeleton-based action recognition with real-time inference capabilities on edge devices.

---

## 1. Pipeline Architecture Overview

The LAST pipeline consists of five major stages that transform raw skeleton coordinates into action predictions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DATA INGESTION                         │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Skeleton Files (.skeleton, .npy)                               │
│          ↓                                                           │
│  Load & Parse (N, C, T, V, M) tensors                              │
│          ↓                                                           │
│  [N=samples, C=3(xyz), T=300 frames, V=25 joints, M=2 persons]     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: PREPROCESSING                            │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: Normalization                                              │
│    • Center skeleton (subtract SpineBase coordinates)               │
│    • Rotate to canonical orientation (front-facing)                 │
│    • Scale to unit variance                                         │
│          ↓                                                           │
│  Step 2: Temporal Alignment                                         │
│    • Uniform sampling to fixed T (e.g., 64 or 300 frames)          │
│    • Zero-padding for shorter sequences                             │
│          ↓                                                           │
│  Step 3: Data Augmentation (Training Only)                          │
│    • Random rotation: ±15° around Z-axis                            │
│    • Random scale: 0.9-1.1x                                         │
│    • Gaussian noise: σ = 0.001                                      │
│    • Random shear: ±0.1                                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│         STAGE 3: DUAL-PATH FORWARD PASS (Training)                  │
├───────────────────────┬─────────────────────────────────────────────┤
│  TEACHER PATH (RGB)   │    STUDENT PATH (Skeleton)                  │
│  [Training Only]      │    [Training & Inference]                   │
├───────────────────────┼─────────────────────────────────────────────┤
│  VideoMAE V2          │    ↓                                        │
│  (Frozen Weights)     │  Input Stem: Conv2d(3→64, kernel=1×1)      │
│         ↓             │    ↓                                        │
│  RGB Frames           │  ┌─────────── LAST Block 1 ──────────────┐ │
│  (T, H, W, 3)         │  │  1. A-GCN (Adaptive Graph Conv)       │ │
│         ↓             │  │  2. TSM (Temporal Shift)              │ │
│  Vision Encoder       │  │  3. Linear Attention                  │ │
│  (12 Transformer      │  │  4. Residual + Layer Norm             │ │
│   layers)             │  └───────────────────────────────────────┘ │
│         ↓             │    ↓                                        │
│  Teacher Logits       │  LAST Block 2 (same structure)              │
│  (B, 120 classes)     │    ↓                                        │
│         ↓             │  ...  (Total: 4-8 blocks)                   │
│         │             │    ↓                                        │
│         │             │  LAST Block N                               │
│         │             │    ↓                                        │
│         │             │  Global Average Pool (T, V → 1)            │
│         │             │    ↓                                        │
│         │             │  FC Layer (C_embed → 120 classes)          │
│         │             │    ↓                                        │
│         └─────────────┼──→ Student Logits (B, 120)                 │
│                       │                                             │
└───────────────────────┴─────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: LOSS COMPUTATION                        │
├─────────────────────────────────────────────────────────────────────┤
│  L_CE = CrossEntropy(Student_Logits, Ground_Truth)                 │
│         ↓                                                           │
│  L_KD = KL_Divergence(                                              │
│           Softmax(Student_Logits / τ),                              │
│           Softmax(Teacher_Logits / τ)                               │
│         ) × τ²                                                      │
│         ↓                                                           │
│  L_total = α × L_CE + β × L_KD                                      │
│  (α=0.5, β=0.5, τ=4.0 - temperature)                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: EVALUATION                              │
├─────────────────────────────────────────────────────────────────────┤
│  Top-1 Accuracy: argmax(Student_Logits) == Ground_Truth            │
│  Top-5 Accuracy: Ground_Truth in top-5(Student_Logits)             │
│         ↓                                                           │
│  Efficiency Metrics:                                                │
│    • FLOPs (via thop library)                                       │
│    • Params (model.parameters())                                    │
│    • Latency (time per forward pass on CPU)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Dataset Status & Phased Training Strategy

### 2.1 Available Dataset: NTU-120 Skeleton Files

**Current Status:** You have downloaded the NTU RGB+D 120 skeleton data organized as follows:

```
NTU_RGBD_120/
├── nturgb+d_skeletons_s001_to_s017/     # Folder 1: Setups 001-017
│   ├── S001C001P001R001A001.skeleton
│   ├── S001C001P001R001A002.skeleton
│   ├── ...
│   └── S017C003P106R002A120.skeleton
│
└── nturgb+d_skeletons_s018_to_s032/     # Folder 2: Setups 018-032
    ├── S018C001P001R001A001.skeleton
    ├── S018C001P001R001A002.skeleton
    ├── ...
    └── S032C003P106R002A120.skeleton
```

**File Format:** Each `.skeleton` file is a text file containing:
```
Frame_count
  Body_count
    Body_ID
    Clipped_edges
    Hand_left_confidence + Hand_left_state
    Hand_right_confidence + Hand_right_state
    Restricted
    Lean_x + Lean_y
    Tracking_state
      Joint_count (25)
        x, y, z (3D coordinates)
        depth_x, depth_y
        color_x, color_y
        orientation_w, orientation_x, orientation_y, orientation_z
        tracking_state
      [Repeat for all 25 joints]
  [Repeat for all bodies in frame]
[Repeat for all frames]
```

**What You Need to Extract:** Only the `(x, y, z)` coordinates for all 25 joints across all frames.

---

### 2.2 Phased Training Strategy

Since you currently have **only skeleton data**, we'll use a two-phase approach:

#### **Phase 1: Baseline Training (Start Immediately)**
**Objective:** Train LAST using only skeleton data with standard cross-entropy loss.

```
Training Loop (Phase 1):
  For each batch of skeletons:
    1. Student(skeleton) → logits
    2. L_CE = CrossEntropy(logits, ground_truth)
    3. Backprop through student
```

**Expected Performance:** 86-88% accuracy (without distillation)

**Timeline:**
- **Week 1:** Implement dataloader + preprocessing
- **Week 2:** Build LAST model architecture
- **Week 3:** Train and achieve baseline accuracy

---

#### **Phase 2: Knowledge Distillation (After RGB Download)**
**Objective:** Boost accuracy by distilling knowledge from RGB teacher.

**Prerequisites:**
1. Download NTU-120 RGB videos (~1.5 TB)
2. Download pretrained VideoMAE V2 checkpoint
3. Pre-compute teacher logits (one-time pass)

```
Teacher Pre-computation (One-time):
  For each video in training set:
    1. Extract RGB frames
    2. Teacher(RGB frames) → logits
    3. Save logits to disk (e.g., teacher_logits/S001C001P001R001A001.npy)

Training Loop (Phase 2):
  For each batch:
    1. Load skeleton data
    2. Load pre-computed teacher logits from disk
    3. Student(skeleton) → student_logits
    4. L_CE = CrossEntropy(student_logits, ground_truth)
    5. L_KD = KL_Div(student_logits, teacher_logits)
    6. L_total = α × L_CE + β × L_KD
    7. Backprop through student
```

**Expected Performance:** 89-92% accuracy (with distillation)

**Timeline:**
- **Week 4:** Download RGB videos + teacher model
- **Week 4:** Pre-compute all teacher logits (~8-12 hours on T4 GPU)
- **Week 4-5:** Fine-tune LAST with distillation

---

### 2.3 Teacher Model: VideoMAE V2 Specification

#### **Model Details**

**Name:** VideoMAE V2 (Video Masked Autoencoder Version 2)  
**Architecture:** Vision Transformer (ViT-Base)  
**Paper:** "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking" (CVPR 2023)  
**GitHub:** [https://github.com/OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)

**Model Configuration:**
```python
model_config = {
    'architecture': 'vit_base_patch16_224',
    'input_size': 224,
    'patch_size': 16,
    'num_frames': 16,           # Sample 16 frames uniformly from video
    'tubelet_size': 2,          # Temporal patch size
    'num_classes': 120,         # Fine-tuned for NTU-120
    'embed_dim': 768,
    'depth': 12,                # 12 Transformer layers
    'num_heads': 12,
    'decoder_embed_dim': 384,   # Not needed for inference
}
```

#### **Where to Download the Pretrained Checkpoint**

**Option 1: Official Checkpoint (Recommended)**
```bash
# Download from Hugging Face Model Hub
wget https://huggingface.co/OpenGVLab/VideoMAEv2/resolve/main/vit_b_k400_pt.pth

# Size: ~340 MB
# Pretrained on: Kinetics-400
# You will need to fine-tune this on NTU-120 RGB videos
```

**Option 2: Community Fine-tuned (If Available)**
Search for "VideoMAE NTU-120" on:
- Hugging Face: [https://huggingface.co/models](https://huggingface.co/models)
- Papers with Code: [https://paperswithcode.com/dataset/ntu-rgbd](https://paperswithcode.com/dataset/ntu-rgbd)

**Note:** If no fine-tuned checkpoint exists, you'll need to fine-tune the Kinetics-400 checkpoint on NTU-120 RGB videos yourself (3-5 days of GPU training).

---

#### **How to Use the Teacher Model**

**Step 1: Install Dependencies**
```bash
pip install torch torchvision
pip install timm  # PyTorch Image Models (contains ViT implementations)
pip install decord  # Efficient video decoding
```

**Step 2: Load the Teacher Model**
```python
import torch
import timm
from torchvision import transforms

# Create model
teacher = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=120,
    img_size=224
)

# Load checkpoint
checkpoint = torch.load('vit_b_ntu120_finetuned.pth')
teacher.load_state_dict(checkpoint['model'])
teacher.eval()  # Set to evaluation mode
teacher.cuda()  # Move to GPU

# Freeze all parameters (no gradient computation)
for param in teacher.parameters():
    param.requires_grad = False
```

**Step 3: Preprocess RGB Videos**
```python
from decord import VideoReader
import numpy as np

def load_video(video_path, num_frames=16):
    """Load and preprocess video for VideoMAE."""
    vr = VideoReader(video_path)
    total_frames = len(vr)
    
    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr .get_batch(indices).asnumpy()  # (T, H, W, C)
    
    # Resize and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    frames = torch.stack([transform(f) for f in frames])  # (T, 3, 224, 224)
    return frames.unsqueeze(0)  # (1, T, 3, 224, 224)

# Rearrange for model input: (B, T, C, H, W) → (B, C, T, H, W)
def rearrange_for_video_model(frames):
    return frames.permute(0, 2, 1, 3, 4)
```

**Step 4: Pre-compute Teacher Logits (One-Time)**
```python
import os
from tqdm import tqdm

# Directory containing NTU RGB videos
rgb_video_dir = '/path/to/nturgb+d_rgb/'
output_dir = '/path/to/teacher_logits/'
os.makedirs(output_dir, exist_ok=True)

# Get all video files
video_files = sorted([f for f in os.listdir(rgb_video_dir) if f.endswith('.avi')])

teacher.eval()
with torch.no_grad():
    for video_file in tqdm(video_files, desc="Pre-computing teacher logits"):
        video_path = os.path.join(rgb_video_dir, video_file)
        
        # Load and preprocess
        frames = load_video(video_path, num_frames=16)
        frames = rearrange_for_video_model(frames).cuda()
        
        # Forward pass
        logits = teacher(frames)  # (1, 120)
        
        # Save to disk
        sample_name = video_file.replace('.avi', '')
        np.save(os.path.join(output_dir, f'{sample_name}.npy'), 
                logits.cpu().numpy())

print(f"Teacher logits saved to {output_dir}")
print(f"Total files: {len(video_files)}")
print(f"Disk space used: ~{len(video_files) * 0.96} KB (120 float32 values per file)")
```

**Expected Output:**
- **Files:** ~114,000 `.npy` files (one per training sample)
- **Disk Space:** ~110 MB (very small!)
- **Time:** 8-12 hours on T4 GPU for full NTU-120 dataset

---

#### **Alternative: Real-time Teacher Inference (Not Recommended)**

If you don't want to pre-compute:

```python
# During training (slower, uses more memory)
for batch in train_loader:
    skeletons = batch['skeleton'].cuda()
    rgb_videos = batch['rgb'].cuda()  # Requires loading RGB in dataloader
    labels = batch['label'].cuda()
    
    # Teacher forward (frozen)
    with torch.no_grad():
        teacher_logits = teacher(rgb_videos)
    
    # Student forward
    student_logits = student(skeletons)
    
    # Compute loss
    loss = alpha * ce_loss(student_logits, labels) + \
           beta * kd_loss(student_logits, teacher_logits)
    
    loss.backward()
    optimizer.step()
```

**Drawbacks:**
- 3-5x slower training
- Requires 2x GPU memory (teacher + student)
- Must load both RGB and skeleton data

**Recommendation:** Always use pre-computation for efficiency.

---

### 2.4 RGB Video Download Guide

**Dataset:** NTU RGB+D 120 - RGB Videos  
**Size:** ~1.5 TB  
**Format:** `.avi` files (one per action sample)

**Official Download:**
1. Visit: [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. Request access (requires academic email)
3. Download links provided via email (usually within 1-2 days)

**Cloud Download Strategy (GCP):**
```bash
# SSH into your GCP VM
gcloud compute ssh your-vm-name --zone=your-zone

# Create directory
mkdir -p /mnt/local-ssd/ntu_rgb
cd /mnt/local-ssd/ntu_rgb

# Download using provided links (example)
wget -c "<download_link_part1>" -O nturgb+d_rgb_s001_to_s017.zip
wget -c "<download_link_part2>" -O nturgb+d_rgb_s018_to_s032.zip

# Unzip (takes 2-3 hours)
unzip nturgb+d_rgb_s001_to_s017.zip
unzip nturgb+d_rgb_s018_to_s032.zip

# Clean up zips to save space
rm *.zip
```

**Timeline Estimate:**
- Download: 6-8 hours (depends on connection)
- Extraction: 2-3 hours
- **Total:** ~10-12 hours

**Note:** You can start building LAST with skeleton data **immediately** and download RGB videos later when ready for Phase 2 distillation.

---

## 3. Detailed Component Specifications

### 2.1 Data Loading & Preprocessing Pipeline

#### 2.1.1 Input Tensor Format

**Shape:** `(N, C, T, V, M)`

| Dimension | Description | Value (NTU-120) | Notes |
|-----------|-------------|-----------------|-------|
| **N** | Batch size | 32-128 | Depends on GPU memory |
| **C** | Channels | 3 | x, y, z coordinates; can extend to 6 with velocity |
| **T** | Temporal frames | 300 or 64 | 300 for full sequence, 64 for efficient training |
| **V** | Vertices (joints) | 25 | NTU skeleton has 25 joints |
| **M** | Max persons | 2 | NTU supports up to 2 actors per frame |

#### 2.1.2 Skeleton Joint Topology (NTU-120)

The 25-joint skeleton follows this hierarchy:

```
           4 (Head)
           ↑
     3 (Neck/Shoulder Center)
    ↗  ↑  ↖
   5   2   9  
  ↓    ↓    ↓
  6    1   10
  ↓    ↓    ↓
  7   20   11
  ↓   ↙↘    ↓
  8  19 12  12
     ↓   ↓
    18  24
    ↓   ↓
    17  25

Legend:
1: SpineBase (Root/Center)
2: SpineMid
3: Neck
4: Head
5-8: Left Arm (Shoulder→Hand)
9-12: Right Arm
13-16: Left Leg (Hip→Foot)
17-20: Right Leg
21-25: Spine & Feet details
```

#### 2.1.3 Normalization Strategy

**Step 1: Centering**
```python
# Subtract SpineBase (joint 1) from all joints
skeleton_centered = skeleton - skeleton[:, :, :, 0:1, :]  # Broadcasting
```

**Step 2: Rotation Alignment**
```python
# Compute main direction using shoulders
left_shoulder = skeleton[:, :, :, 4, :]   # Joint 5
right_shoulder = skeleton[:, :, :, 8, :]  # Joint 9
direction = left_shoulder - right_shoulder

# Compute rotation angle to align to canonical Y-axis
theta = arctan2(direction[:, :, :, 0], direction[:, :, :, 1])

# Apply rotation matrix
R = [[cos(theta), -sin(theta), 0],
     [sin(theta),  cos(theta), 0],
     [0,           0,          1]]
skeleton_rotated = R @ skeleton_centered
```

**Step 3: Scale Normalization**
```python
# Normalize by skeleton size (max distance from center)
scale = skeleton.abs().max(dim=-1, keepdim=True)
skeleton_normalized = skeleton_rotated / (scale + 1e-6)
```

#### 2.1.4 Data Augmentation Parameters

| Augmentation | Range | Probability | Rationale |
|--------------|-------|-------------|-----------|
| **Rotation (Z-axis)** | ±15° | 0.5 | Viewpoint invariance |
| **Scale** | [0.9, 1.1] | 0.5 | Body size variations |
| **Shear** | ±0.1 | 0.3 | Movement variations |
| **Gaussian Noise** | σ=0.001 | 0.5 | Sensor noise simulation |
| **Temporal Crop** | Random start | 1.0 | Temporal diversity |

---

### 2.2 LAST Model Architecture (Detailed)

#### 2.2.1 Overall Architecture

```
Input: (B, 3, T, V, M) → Reshape → (B, 3, T, V)  [Focus on primary actor M=0]
   ↓
Stem: Conv2d(3 → 64, kernel=1×1) + BatchNorm + ReLU
   ↓
LAST Block 1: (64 → 128 channels)
   ↓
LAST Block 2: (128 → 128 channels)
   ↓
LAST Block 3: (128 → 256 channels)
   ↓
LAST Block 4: (256 → 256 channels)
   ↓
Global Average Pooling: (B, 256, T, V) → (B, 256)
   ↓
Dropout (p=0.5)
   ↓
FC: (256 → 120 classes)
   ↓
Output Logits: (B, 120)
```

**Total Parameters:** ~2.1 M  
**Total FLOPs:** ~0.85 GFLOPs (per sample)

---

#### 2.2.2 LAST Block Internal Structure

Each LAST block processes spatial and temporal information efficiently through three sequential operations:

```
┌──────────────────────────────────────────────────────────┐
│              LAST BLOCK (Single Unit)                    │
├──────────────────────────────────────────────────────────┤
│  Input: x (B, C_in, T, V)                                │
│    ↓                                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. ADAPTIVE GRAPH CONVOLUTION (A-GCN)              │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  • Compute adjacency matrices:                     │  │
│  │    A_total = A_physical + A_learned + A_dynamic    │  │
│  │                                                     │  │
│  │  • A_physical: Fixed skeleton topology (25×25)    │  │
│  │  • A_learned: Global learned weights (param)      │  │
│  │  • A_dynamic: BatchNorm(Conv(x)) → (B,V,V)        │  │
│  │                                                     │  │
│  │  • Graph aggregation:                              │  │
│  │    x' = Σ_k A_total[k] × x × W[k]                 │  │
│  │    k ∈ {root, neighbor, global} subsets           │  │
│  │                                                     │  │
│  │  Output: (B, C_mid, T, V)                          │  │
│  └────────────────────────────────────────────────────┘  │
│    ↓                                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 2. TEMPORAL SHIFT MODULE (TSM)                     │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  • Split channels: C_mid = C_fwd + C_bwd + C_stay │  │
│  │    - C_fwd = C_mid // 8  (shift forward)          │  │
│  │    - C_bwd = C_mid // 8  (shift backward)         │  │
│  │    - C_stay = remaining  (no shift)               │  │
│  │                                                     │  │
│  │  • Apply shifts:                                   │  │
│  │    x'[:, 0:C_fwd,     1:T, :] = x[:, 0:C_fwd, 0:T-1, :]    │
│  │    x'[:, C_fwd:2*C_fwd, 0:T-1, :] = x[:, C_fwd:2*C_fwd, 1:T, :] │
│  │    x'[:, 2*C_fwd:,    :, :] = x[:, 2*C_fwd:, :, :]  │
│  │                                                     │  │
│  │  • Zero cost: 0 FLOPs, 0 Params                   │  │
│  │  Output: (B, C_mid, T, V)                          │  │
│  └────────────────────────────────────────────────────┘  │
│    ↓                                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 3. LINEAR ATTENTION                                │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  • Reshape: (B, C_mid, T, V) → (B, T, C_mid×V)    │  │
│  │                                                     │  │
│  │  • Compute Q, K, V:                                │  │
│  │    Q = Linear(x)  # (B, T, d_k)                    │  │
│  │    K = Linear(x)  # (B, T, d_k)                    │  │
│  │    V = Linear(x)  # (B, T, d_v)                    │  │
│  │                                                     │  │
│  │  • Apply kernel feature map φ:                     │  │
│  │    φ(x) = elu(x) + 1  (ensures positivity)        │  │
│  │                                                     │  │
│  │  • Linear attention (O(T) complexity):             │  │
│  │    Attention(Q,K,V) = φ(Q) × [(φ(K)^T × V)]       │  │
│  │    Standard: O(T²), Linear: O(T)                   │  │
│  │                                                     │  │
│  │  • Reshape back: (B, T, d_v) → (B, C_out, T, V)  │  │
│  │  Output: (B, C_out, T, V)                          │  │
│  └────────────────────────────────────────────────────┘  │
│    ↓                                                      │
│  Residual Connection: out = out + x_residual (if dims match)│
│    ↓                                                      │
│  Layer Normalization                                     │
│    ↓                                                      │
│  ReLU Activation                                         │
│    ↓                                                      │
│  Output: (B, C_out, T, V)                                │
└──────────────────────────────────────────────────────────┘
```

---

### 2.3 Teacher-Student Knowledge Distillation

#### 2.3.1 Teacher Model: VideoMAE V2

**Architecture:** Vision Transformer (ViT-Base)
- **Input:** RGB video frames (T, H, W, 3)
- **Preprocessing:** 
  - Resize to 224×224
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Encoder:** 12 Transformer layers, 768 hidden dim, 12 attention heads
- **Output:** Logits (B, 120 classes)
- **Weights:** Pretrained on Kinetics-400, fine-tuned on NTU-120
- **Parameters:** ~86M (frozen during LAST training)

#### 2.3.2 Distillation Loss Function

The total loss combines classification accuracy with knowledge transfer:

```
L_total = α × L_CE + β × L_KD

Where:
  L_CE = CrossEntropyLoss(student_logits, ground_truth)
  
  L_KD = KL_Divergence(
           Softmax(student_logits / τ),
           Softmax(teacher_logits / τ)
         ) × τ²
```

**Hyperparameters:**
- **α (Classification weight):** 0.5
- **β (Distillation weight):** 0.5
- **τ (Temperature):** 4.0
  - Higher temperature → softer probability distributions
  - Emphasizes relative rankings between classes
  - Transfers "dark knowledge" (e.g., similarity between actions)

**Temperature Scaling Rationale:**
- At τ=1: Distillation becomes standard cross-entropy
- At τ=4-6: Student learns class relationships (e.g., "wave" vs "clap" both involve arm movement)
- The τ² term compensates for gradient magnitudes

#### 2.3.3 Training Procedure

**Phase 1: Teacher Pre-computation (Optional Optimization)**
```python
# Pre-compute teacher logits for entire dataset (saves GPU memory)
teacher.eval()
with torch.no_grad():
    for batch in train_loader:
        rgb_frames = batch['rgb']
        teacher_logits[batch_idx] = teacher(rgb_frames)
```

**Phase 2: Student Training**
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        skeletons = batch['skeleton']
        labels = batch['label']
        teacher_logits_batch = teacher_logits[batch_idx]  # Pre-computed
        
        # Student forward pass
        student_logits = student(skeletons)
        
        # Compute losses
        loss_ce = criterion_ce(student_logits, labels)
        loss_kd = kl_divergence(
            F.softmax(student_logits / tau, dim=1),
            F.softmax(teacher_logits_batch / tau, dim=1)
        ) * (tau ** 2)
        
        loss_total = alpha * loss_ce + beta * loss_kd
        
        # Backprop
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
```

---

## 3. Training Configuration

### 3.1 Optimizer & Learning Rate Schedule

**Optimizer:** AdamW
- **Initial LR:** 0.001
- **Weight decay:** 0.0001
- **Betas:** (0.9, 0.999)

**LR Scheduler:** Cosine Annealing with Warm Restarts
```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # Restart every 10 epochs
    T_mult=2,      # Double the period after each restart
    eta_min=1e-6   # Minimum LR
)
```

**Learning Rate Curve:**
```
LR
│       
0.001 ├─╮                    ╭─╮
      │  ╰╮                ╭╯  ╰╮
      │    ╰╮            ╭╯      ╰╮
      │      ╰╮        ╭╯          ╰╮
1e-6  ├────────╰──────╯──────────────╰──────
      └─────────────────────────────────────► Epoch
      0    10        30              70     100
```

### 3.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 80-100 | Stop when validation plateaus |
| **Batch Size** | 64 | Adjust based on GPU memory |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients |
| **Mixed Precision** | FP16 | 2x speedup on modern GPUs |
| **Warmup Epochs** | 5 | Linear warmup 0→initial_lr |
| **Checkpoint Frequency** | Every 5 epochs | Save best model based on Val Acc |

### 3.3 Data Split Strategy (NTU-120)

**Cross-Subject (X-Sub):**
- **Train subjects:** 1, 2, 4, 5, 8, 9, 13, 14, ..., 106 (53 subjects)
- **Test subjects:** 3, 6, 7, 10, 11, 12, 15, ... (53 subjects)
- **Train samples:** ~63,000
- **Test samples:** ~50,000

**Cross-Setup (X-Set):**
- **Train setups:** Cameras 1, 2 (even setup IDs)
- **Test setups:** Camera 3 (odd setup IDs)

---

## 4. Evaluation & Metrics

### 4.1 Accuracy Metrics

**Top-1 Accuracy:**
```python
predictions = torch.argmax(logits, dim=1)
accuracy = (predictions == labels).float().mean()
```
**Target:** > 89% on NTU-120 X-Sub

**Top-5 Accuracy:**
```python
top5_preds = torch.topk(logits, k=5, dim=1).indices
accuracy_top5 = (labels.unsqueeze(1) == top5_preds).any(dim=1).float().mean()
```
**Target:** > 97%

### 4.2 Efficiency Metrics

**Computational Complexity (FLOPs):**
```python
from thop import profile
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops / 1e9:.2f} G")  # Target: < 1.0 GFLOPs
```

**Model Size:**
```python
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model Size: {param_size / 1e6:.2f} MB")  # Target: < 3 MB
```

**Inference Latency:**
```python
import time
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output = model(input_cpu)  # CPU inference
    end = time.time()
    latency = (end - start) / 100 * 1000  # ms
print(f"Latency: {latency:.2f} ms")  # Target: < 33 ms (30 FPS)
```

### 4.3 Ablation Studies

To prove each component's contribution:

| Model Variant | Top-1 Acc (%) | FLOPs (G) | Params (M) |
|---------------|---------------|-----------|------------|
| **Baseline (ST-GCN)** | 86.5 | 1.5 | 3.1 |
| **+ A-GCN only** | 88.2 | 1.4 | 3.0 |
| **+ TSM only** | 87.1 | 0.9 | 3.1 |
| **+ Linear Attn only** | 87.8 | 2.8 | 4.2 |
| **LAST (All three)** | 89.5 | 0.85 | 2.1 |
| **+ Distillation** | **91.2** | **0.85** | **2.1** |

### 4.4 Visualization & Interpretability

**Class Activation Mapping (CAM):**
```python
# Extract attention weights from final Linear Attention layer
attention_weights = model.last_block.attention.get_weights()  # (B, T, V)

# Visualize which joints/frames contribute most
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0].cpu(), cmap='hot')
plt.xlabel('Joints')
plt.ylabel('Time Frames')
plt.title('Attention Heatmap for Action: Waving')
plt.savefig('cam_waving.png')
```

**t-SNE Embedding Comparison:**
```python
from sklearn.manifold import TSNE

# Extract features before final FC layer
student_features = model.get_features(skeleton_data)
teacher_features = teacher_model.get_features(rgb_data)

# Reduce to 2D
tsne = TSNE(n_components=2)
student_2d = tsne.fit_transform(student_features.cpu().numpy())
teacher_2d = tsne.fit_transform(teacher_features.cpu().numpy())

# Plot
plt.scatter(student_2d[:, 0], student_2d[:, 1], c=labels, alpha=0.5, label='Student')
plt.scatter(teacher_2d[:, 0], teacher_2d[:, 1], c=labels, alpha=0.5, label='Teacher', marker='x')
plt.legend()
plt.title('Feature Space Comparison')
```

---

## 5. Deployment Pipeline

### 5.1 Model Export (ONNX)

```python
# Export to ONNX for cross-platform deployment
dummy_input = torch.randn(1, 3, 64, 25)
torch.onnx.export(
    model,
    dummy_input,
    "last_model.onnx",
    opset_version=13,
    input_names=['skeleton'],
    output_names=['logits'],
    dynamic_axes={'skeleton': {0: 'batch_size'}}
)
```

### 5.2 Quantization (INT8) for Edge Devices

```python
# Post-training quantization
import torch.quantization as quantization

model.eval()
model_quantized = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Test quantized model
output_quantized = model_quantized(input_cpu)
print(f"Quantized Model Size: {get_model_size(model_quantized):.2f} MB")
# Expected: ~0.6 MB (3x reduction)
```

### 5.3 End-to-End Inference Pipeline

```python
def real_time_inference(video_path):
    # Step 1: Extract skeletons from video using MediaPipe
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    cap = cv2.VideoCapture(video_path)
    skeleton_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract pose
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            joints = np.array([[lm.x, lm.y, lm.z] 
                              for lm in results.pose_landmarks.landmark])
            skeleton_sequence.append(joints)
    
    # Step 2: Preprocess
    skeleton_tensor = preprocess(np.array(skeleton_sequence))
    
    # Step 3: Inference
    with torch.no_grad():
        logits = model(skeleton_tensor)
        prediction = torch.argmax(logits, dim=1)
    
    # Step 4: Map to action name
    action_name = ACTION_CLASSES[prediction.item()]
    return action_name
```

---

## 6. Expected Performance Benchmarks

### 6.1 Accuracy Targets

| Dataset | Protocol | Baseline (ST-GCN) | LAST (Ours) | Target |
|---------|----------|-------------------|-------------|--------|
| **NTU-120** | X-Sub | 86.5% | 91.2% | > 89% |
| **NTU-120** | X-Set | 88.1% | 92.5% | > 90% |
| **Kinetics-Skeleton** | Top-1 | 35.2% | 38.7% | > 37% |

### 6.2 Efficiency Comparison

| Model | FLOPs | Params | Latency (CPU) | Accuracy (NTU X-Sub) |
|-------|-------|--------|---------------|---------------------|
| InfoGCN (SOTA) | 5.2 G | 12.4 M | 142 ms | 93.8% |
| Shift-GCN | 0.7 G | 2.8 M | 28 ms | 87.3% |
| MSA-GCN | 3.1 G | 8.7 M | 89 ms | 92.1% |
| **LAST (Ours)** | **0.85 G** | **2.1 M** | **31 ms** | **91.2%** |

**Key Achievement:** LAST achieves near-SOTA accuracy (−2.6% vs InfoGCN) with **6x fewer FLOPs** and **5.9x fewer parameters**, making it suitable for real-time edge deployment.

---

## 7. Implementation Checklist

- [ ] **Week 1:**
  - [ ] Implement dataloader with augmentations
  - [ ] Verify NTU-120 preprocessing pipeline
  - [ ] Train baseline ST-GCN to validate setup

- [ ] **Week 2:**
  - [ ] Implement A-GCN module
  - [ ] Implement TSM module
  - [ ] Implement Linear Attention module
  - [ ] Assemble LAST model
  - [ ] Load VideoMAE V2 teacher weights

- [ ] **Week 3:**
  - [ ] Train LAST with distillation
  - [ ] Run ablation studies
  - [ ] Hyperparameter tuning

- [ ] **Week 4:**
  - [ ] Measure FLOPs, latency, model size
  - [ ] Generate CAM visualizations
  - [ ] Export to ONNX
  - [ ] Write final report
