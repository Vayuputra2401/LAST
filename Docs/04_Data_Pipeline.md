# 04 — Data Pipeline

## Datasets

### NTU RGB+D 60

- **Classes**: 60 action categories (daily actions, health-related, mutual actions)
- **Subjects**: 40, up to M=2 bodies per sequence
- **Joints**: 25 per body (Kinect v2 skeleton)
- **Split protocols**:
  - `xsub`: train/test by subject ID (20 training subjects, 20 test) — primary benchmark
  - `xview`: train/test by camera viewpoint

### NTU RGB+D 120

- **Classes**: 120 action categories (superset of NTU-60)
- **Subjects**: 106, up to M=2 bodies
- **Split protocols**:
  - `xsub`: subject-based split
  - `xset`: camera setup-based split

Both datasets provide 25-joint 3D skeleton coordinates per body per frame.

---

## 4-Stream Decomposition

Following the multi-stream paradigm established by 2s-AGCN and extended to 4 streams by EfficientGCN, we decompose each skeleton sequence into four complementary kinematic representations:

| Stream | Key | Formula | Signal |
|--------|-----|---------|--------|
| Joint | `joint` | Coordinates centred at spine base (joint 0) | Absolute spatial configuration |
| Velocity | `velocity` | J(t+1) - J(t) (frame difference) | Motion speed and direction |
| Bone | `bone` | J(child) - J(parent) per skeleton edge | Limb orientation and length |
| Bone velocity | `bone_velocity` | B(t+1) - B(t) (frame diff of bone) | Limb angular dynamics |

Each stream has shape (C=3, T, V=25, M=2) where C=3 corresponds to (x, y, z) coordinates.

---

## Preprocessing

**Script**: `scripts/preprocess_v2.py`

Applied offline to raw NTU skeleton files:

1. **Centre at spine base** — translate all joints so joint 0 (spine base) is at the origin per frame
2. **Scale by torso length** — divide by the distance from spine base to spine shoulder for scale invariance
3. **Stream extraction**:
   - Velocity: exact frame difference `J[t+1] - J[t]`, last frame duplicated
   - Bone: exact parent-child difference using the NTU skeleton tree
   - Bone velocity: frame difference of bone stream
4. **Save per-split**: `{joint,velocity,bone,bone_velocity}.npy` + `label.pkl` per split

### Preprocessed Data Statistics (NTU-60 xsub, verified)

| Split | Samples | Shape | Classes | Balance |
|-------|---------|-------|---------|---------|
| train | 40,320 | (40320, 3, 64, 25, 2) | 60 | 672 per class (perfectly balanced) |
| val | 16,560 | (16560, 3, 64, 25, 2) | 60 | 276 per class (perfectly balanced) |

- All 60 classes present in both splits, no NaN/Inf values
- Joint coordinate range: [-2.849, 4.881] (after spine-base centring and torso scaling)
- Velocity stream: exact match to frame diff of joint (max_abs_err = 0.0)
- Bone stream: exact match to child-parent diff (max_abs_err = 0.0)

---

## Temporal Sampling

All sequences are uniformly resized to **T=64 frames** during preprocessing. At training time:

| Transform | Train | Val |
|-----------|-------|-----|
| Random temporal crop | Yes (64 frames from padded sequence) | No |
| Centre crop | No | Yes (64 frames) |

The `max_frames` parameter is read from the config for both train and val datasets.

---

## Data Augmentation

**File**: `src/data/transforms.py`

Applied online during training:

| Augmentation | Range | Applied to | Purpose |
|-------------|-------|-----------|---------|
| RandomRotation | +/-15 deg per axis | Joint stream | Viewpoint invariance |
| RandomScale | +/-10% | Joint stream | Scale robustness |
| RandomTemporalSpeed | 0.9--1.1x | All streams | Temporal speed variation |

Augmentation is applied to the joint stream; velocity, bone, and bone_velocity are derived from augmented joints where applicable.

---

## M-Dimension Handling

Both models receive (B, C, T, V, M=2) tensors but handle the multi-body dimension differently:

**LAST-Lite**: Strips M in forward: `s = s[..., 0]` — takes primary body only. This is a deliberate simplification: most NTU sequences are single-person, and for two-person actions the primary body carries sufficient discriminative signal for a lightweight model.

**LAST-Base** (planned): Will process M internally per-stream, potentially with cross-body interaction.

---

## Dataset Class

**File**: `src/data/dataset.py`

```python
dataset = SkeletonDataset(
    data_path='data/processed/xsub/',
    split='train',
    max_frames=64,
)
# Returns: (dict_of_4_stream_tensors, label_int)
# Each stream: Tensor(C=3, T=64, V=25, M=2)
```

The DataLoader collates into `(dict_of_batched_tensors, labels)` where each stream tensor has shape (B, 3, 64, 25, 2).

---

## DataLoader Configuration

Workers and prefetch settings are set per environment:

| Environment | Workers | pin_memory | Notes |
|-------------|---------|-----------|-------|
| Local | 4 | True | Full CPU cores |
| Kaggle (T4) | 2 | True | Limited CPU on Kaggle |

Batch size is set via config/CLI (`batch_size: 64` for LAST-Lite by default).

---

## Data Paths

The training script auto-resolves data paths from environment config:

```yaml
# configs/environment/kaggle.yaml
data_folder: '/kaggle/input/LAST-60'

# configs/environment/local.yaml
data_folder: 'E:/LAST-60'
```

The `data_folder` key is read at runtime. The script constructs the full path as `{data_folder}/data/processed/{split_type}/`.
