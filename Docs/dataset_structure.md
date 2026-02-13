# Dataset Structures & Management

This document outlines the file structures for the primary and secondary datasets required for the LAST research project, with detailed parsing instructions for NTU RGB+D skeleton files.

---

## 1. Primary Dataset: NTU RGB+D 120

**Description:** The largest scale dataset for skeleton-based action recognition.  
**Classes:** 120 Action Classes.  
**Subjects:** 106 distinct subjects.  
**Samples:** ~114,480 skeleton sequences

### 1.1 Raw Data Structure

The raw dataset is organized as individual `.skeleton` text files (one per action sample).

```text
NTU_RGBD_120/
└── nturgb+d_skeletons/
    ├── S001C001P001R001A001.skeleton
    ├── S001C001P001R001A002.skeleton
    ├── S001C001P001R001A003.skeleton
    ├── ...
    └── S032C016P106R002A120.skeleton
```

**Filename Convention:**
```
S<Setup>C<Camera>P<Person>R<Replication>A<Action>.skeleton
```

- **S (Setup ID):** 1-32 (Camera setup configuration)
- **C (Camera ID):** 1-3 (Which camera captured this view)
- **P (Performer/Subject ID):** 1-106 (Person performing action)
- **R (Replication ID):** 1-2 (First or second take)
- **A (Action Class ID):** 1-120 (Action label)

**Example:** `S001C001P001R001A013.skeleton`
- Setup 1, Camera 1, Person 1, Take 1, Action 13 (drink water)

---

### 1.2 Skeleton File Format (`.skeleton`)

Each `.skeleton` file is a **plaintext file** containing temporal skeleton data in a structured binary-like ASCII format.

#### File Structure Overview

```
<total_frames>              # Line 1: Total number of frames in this sequence
<frame_info>                # Frame-level metadata (repeated per frame)
  <num_bodies>              # Number of bodies detected in this frame
  <body_data>               # Body-level data (repeated per body)
    <body_id>               # Unique body tracking ID
    <cliped_edges>          # Clipping indicator
    <hand_left_confidence>  # Left hand tracking confidence (0, 1, or 2)
    <hand_left_state>       # Left hand state
    <hand_right_confidence> # Right hand tracking confidence
    <hand_right_state>      # Right hand state
    <is_restricted>         # Visibility restriction flag
    <lean_x> <lean_y>       # Body lean (tilt) vector
    <tracking_state>        # Kinect tracking quality (0=not tracked, 2=tracked)
    <num_joints>            # Number of joints (always 25 for NTU RGB+D)
    <joint_data>            # 25 joints × (3D pos + orientation + tracking)
      <x> <y> <z>           # 3D position (camera coordinates, meters)
      <depth_x> <depth_y>   # 2D position in depth image (pixels)
      <color_x> <color_y>   # 2D position in color image (pixels)
      <orientation_w> <orientation_x> <orientation_y> <orientation_z>  # Quaternion
      <tracking_state>      # Joint tracking state (0=not tracked, 1=inferred, 2=tracked)
```

#### Detailed File Format (Actual Example)

From `S001C001P001R001A001.skeleton`:

```
103                          # Total frames in sequence
1                            # Frame 0: Number of bodies (1 person)
72057594037931101 0 1 1 1 1 0 0.02764709 0.05745083 2  # Body metadata
25                           # Number of joints (always 25)
# Joint 0: Spine Base
0.2181153 0.1725972 3.785547  # (x, y, z) in meters, camera coordinates
277.419 191.8218              # Depth image coordinates (pixels)
1036.233 519.1677             # Color image coordinates (pixels)
-0.2059419 0.05349901 0.9692109 -0.1239193  # Orientation quaternion (w, x, y, z)
2                             # Tracking state (2 = tracked)

# Joint 1: Spine Mid
0.2323292 0.4326636 3.714767
279.2439 165.8569
1041.918 444.3235
-0.2272637 0.05621852 0.964434 -0.1227094
2

# ... (23 more joints)

# Next frame (Frame 1)
1                            # Number of bodies
72057594037931101 0 1 1 1 1 0 0.03022554 0.04645365 2
25
# ... (25 joints)
```

---

### 1.3 NTU RGB+D Joint Definitions (25 Joints)

```
Joint Index  |  Joint Name
-------------|------------------
    0        |  Spine Base
    1        |  Spine Mid
    2        |  Neck
    3        |  Head
    4        |  Left Shoulder
    5        |  Left Elbow
    6        |  Left Wrist
    7        |  Left Hand
    8        |  Right Shoulder
    9        |  Right Elbow
   10        |  Right Wrist
   11        |  Right Hand
   12        |  Left Hip
   13        |  Left Knee
   14        |  Left Ankle
   15        |  Left Foot
   16        |  Right Hip
   17        |  Right Knee
   18        |  Right Ankle
   19        |  Right Foot
   20        |  Spine (between Base and Mid)
   21        |  Left Hand Tip
   22        |  Left Thumb
   23        |  Right Hand Tip
   24        |  Right Thumb
```

**Skeletal Topology (Bone Connections):**
```
   3 (Head)
   |
   2 (Neck)
   |
   1 (Spine Mid) --- 4 (L Shoulder) --- 5 (L Elbow) --- 6 (L Wrist) --- 7 (L Hand)
   |                                                                         |
  20 (Spine)                                                            21,22 (Tips)
   |
   0 (Spine Base)
  / \
12   16 (Hips)
|     |
13   17 (Knees)
|     |
14   18 (Ankles)
|     |
15   19 (Feet)
```

---

### 1.4 Coordinate System

**3D Coordinates (x, y, z):**
- **Units:** Meters
- **Origin:** Kinect camera position
- **X-axis:** Right (from camera's perspective)
- **Y-axis:** Up
- **Z-axis:** Forward (depth, away from camera)

**2D Depth Image Coordinates:**
- Resolution: 512 × 424 pixels (Kinect V2)
- Origin: Top-left corner

**2D Color Image Coordinates:**
- Resolution: 1920 × 1080 pixels (Kinect V2)
- Origin: Top-left corner

---

### 1.5 Skeleton File Parsing Logic

#### Python Pseudocode

```python
def parse_skeleton_file(file_path):
    """
    Parse NTU RGB+D .skeleton file.
    
    Returns:
        skeleton_data: np.ndarray of shape (T, 25, 3)
            T = number of frames
            25 = number of joints
            3 = (x, y, z) coordinates
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    line_idx = 0
    
    # Read total frames
    num_frames = int(lines[line_idx].strip())
    line_idx += 1
    
    skeleton_sequence = []
    
    for frame_idx in range(num_frames):
        # Read number of bodies in this frame
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1
        
        # For simplicity, take the first body (most datasets have 1-2 bodies)
        # Multi-body handling: store up to M=2 bodies, zero-pad if fewer
        
        frame_skeletons = []
        
        for body_idx in range(num_bodies):
            # Read body metadata (1 line)
            body_info = lines[line_idx].strip().split()
            # body_id, clipped_edges, hand_confidence, ... , tracking_state
            body_tracking_state = int(body_info[-1])
            line_idx += 1
            
            # Read number of joints (should be 25)
            num_joints = int(lines[line_idx].strip())
            line_idx += 1
            
            joints_3d = np.zeros((num_joints, 3))
            
            for joint_idx in range(num_joints):
                joint_data = lines[line_idx].strip().split()
                line_idx += 1
                
                # Parse 3D coordinates
                x = float(joint_data[0])
                y = float(joint_data[1])
                z = float(joint_data[2])
                
                # Depth image coords: joint_data[3], joint_data[4]
                # Color image coords: joint_data[5], joint_data[6]
                # Orientation quaternion: joint_data[7:11]
                # Tracking state: joint_data[11]
                
                joints_3d[joint_idx] = [x, y, z]
            
            frame_skeletons.append(joints_3d)
        
        # Store first body (or handle multi-body with padding)
        if len(frame_skeletons) > 0:
            skeleton_sequence.append(frame_skeletons[0])
        else:
            # No body detected, use zero skeleton
            skeleton_sequence.append(np.zeros((25, 3)))
    
    return np.array(skeleton_sequence)  # Shape: (T, 25, 3)
```

---

### 1.6 Data Preprocessing Pipeline

#### Step 1: Parse Raw `.skeleton` Files
- Extract 3D joint coordinates (x, y, z)
- Handle missing frames (interpolation or zero-padding)
- Normalize coordinates (center on spine base, scale)

#### Step 2: Temporal Padding/Sampling
- **Max Length:** 300 frames (configurable)
- **Padding:** Zero-pad shorter sequences
- **Sampling:** Uniformly sample or interpolate longer sequences

#### Step 3: Multi-Body Handling
- **Max Bodies (M):** 2 (configurable)
- If only 1 body: Second body is all zeros
- If > 2 bodies: Select top 2 by activity (max joint variance)

#### Step 4: Data Augmentation (Optional, During Training)
- **Spatial:** Rotation, scaling, translation, shearing
- **Temporal:** Speed perturbation, random cropping
- **Joint-level:** Gaussian noise, joint dropout

#### Step 5: Save as `.npy` Files

```python
# Shape: (N, C, T, V, M)
# N = number of samples
# C = 3 (x, y, z)
# T = 300 (max frames)
# V = 25 (joints)
# M = 2 (max bodies)

train_data = np.zeros((N_train, 3, 300, 25, 2), dtype=np.float32)
train_labels = []  # List of action labels (0-119)

for idx, skeleton_file in enumerate(train_files):
    skeleton_seq = parse_skeleton_file(skeleton_file)  # (T, 25, 3)
    
    # Pad or sample to 300 frames
    skeleton_seq = temporal_transform(skeleton_seq, target_length=300)
    
    # Transpose to (C, T, V, M) and store
    train_data[idx, :, :, :, 0] = skeleton_seq.transpose(2, 0, 1)  # (3, T, 25)
    
    # Extract label from filename
    label = extract_action_label(skeleton_file)
    train_labels.append(label)

# Save
np.save('train_data.npy', train_data)
with open('train_label.pkl', 'wb') as f:
    pickle.dump(train_labels, f)
```

---

### 1.7 Processed Data Structure (Ready for Dataloader)

To speed up training, data is preprocessed into memory-mapped numpy arrays or pickle files.

```text
NTU_RGBD_120/
├── raw_skeletons/               # Original .skeleton files
│   └── nturgb+d_skeletons/
│       ├── S001C001P001R001A001.skeleton
│       └── ...
├── processed/                   # Preprocessed .npy files
│   ├── xsub/  (Cross-Subject Split)
│   │   ├── train_data.npy       # Shape: (N_train, 3, 300, 25, 2)
│   │   ├── train_label.pkl      # List of labels (length N_train)
│   │   ├── val_data.npy         # Shape: (N_val, 3, 300, 25, 2)
│   │   └── val_label.pkl        # List of labels (length N_val)
│   └── xset/  (Cross-Setup Split)
│       ├── train_data.npy
│       ├── train_label.pkl
│       ├── val_data.npy
│       └── val_label.pkl
```

**Tensor Shape Details `(N, C, T, V, M)`:**
- **N:** Number of samples (e.g., ~63k for X-Sub train, ~50k for X-Sub val)
- **C:** 3 (x, y, z coordinates)
- **T:** 300 (Max frames, zero-padded if shorter)
- **V:** 25 (Joints per skeleton)
- **M:** 2 (Max actors in frame; if 1 actor, 2nd index is zeroed)

**Cross-Subject Split (X-Sub):**
- **Train:** Subjects {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103}
- **Val:** Remaining subjects

**Cross-Setup Split (X-Set):**
- **Train:** Even setup IDs (2, 4, 6, ..., 32)
- **Val:** Odd setup IDs (1, 3, 5, ..., 31)

---

## 2. Secondary Datasets

### A. Kinetics-Skeleton 400

**Description:** Large-scale dataset derived from YouTube videos.  
**Use Case:** Pretraining or generalization testing ("in-the-wild").

**Structure:**
Since Kinetics does not have native skeleton files, they are usually extracted using OpenPose or MediaPipe and stored as JSONs, then converted to `.npy`.

```text
Kinetics-Skeleton/
├── raw_json/
│   ├── train/
│   │   ├── video_id_1.json
│   │   └── ...
│   └── val/
│       ├── video_id_2.json
│       └── ...
├── processed/
│   ├── train_data.npy      # Shape: (240k, 3, 300, 18 or 25, 2)
│   ├── train_label.pkl
│   ├── val_data.npy
│   └── val_label.pkl
```

*Note: Joint count depends on the extractor (OpenPose=18/25, MediaPipe=33).*

---

### B. Northwestern-UCLA Multiview

**Description:** Smaller dataset focusing on viewpoint invariance.  
**Classes:** 10 Actions.  
**Data:** 3 Kinect V1 cameras.

```text
NW_UCLA/
├── multiview_action/
│   ├── view_1/
│   │   ├── action_1/
│   │   └── ...
│   ├── view_2/
│   └── view_3/
├── processed/
│   ├── train_data.npy   # Shape: (N, 3, T, 20, 1) - Kinect V1 has 20 joints
│   └── val_data.npy
```

---

## 3. Data Storage Strategy (Multi-Environment)

### Local Development
- Store raw `.skeleton` files and processed `.npy` files on local disk
- Path: `C:/Users/pathi/OneDrive/Desktop/LAST/data/ntu120/`

### GCP Cloud Execution
For cloud execution, direct file I/O for 100k+ small files is slow.

**Recommendation:**
1. **Upload to GCS:** Store raw `.skeleton` files in Google Cloud Storage bucket
   - `gs://last-research-bucket/datasets/ntu120/raw_skeletons/`
2. **Preprocessing on VM:** Download raw files once to the VM
3. **Conversion:** Run preprocessing script to generate `.npy` files
4. **Optimization:** Store `.npy` files on VM's **Local SSD** (formatted as ext4) for extreme I/O speed during training
   - Mount point: `/mnt/local-ssd/data/ntu120/`
5. **Backup:** Upload processed `.npy` files back to GCS for reuse

**Note:** Using GCS buckets (gcsfuse) for live training data is often too high latency for dataloaders unless cached extensively.

### Kaggle Environment
- Upload processed `.npy` files as Kaggle Dataset
- Kaggle path: `/kaggle/input/ntu-rgbd-120-skeletons/`
- Fast access from `/kaggle/input/` (read-only)
- Working directory: `/kaggle/working/` (read-write)

---

## 4. Data Statistics (NTU RGB+D 120)

| Metric | Value |
|--------|-------|
| Total Samples | ~114,480 |
| Action Classes | 120 |
| Subjects | 106 |
| Camera Views | 3 |
| Setups | 32 |
| Avg. Sequence Length | ~70 frames |
| Max Sequence Length | ~300 frames |
| Joints per Body | 25 |
| Max Bodies per Frame | 2 |

**Training/Validation Split:**
- **X-Sub Train:** ~63,000 samples
- **X-Sub Val:** ~50,000 samples
- **X-Set Train:** ~54,000 samples
- **X-Set Val:** ~60,000 samples

---

**This structure ensures:** Fast data loading, efficient preprocessing, and seamless integration with LAST training pipeline across local, GCP, and Kaggle environments!
