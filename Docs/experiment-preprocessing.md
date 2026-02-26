# Experiment: Preprocessing Pipeline

How to preprocess NTU RGB+D 60/120 skeleton data for both LAST-Lite and LAST-Base.
Preprocessing matches EfficientGCN's strategy: uniform temporal subsampling to a fixed T,
normalized coordinates, with multi-stream generation (joint, bone, velocity, bone-velocity).

---

## 1. Pipeline Overview

```
Raw .skeleton files (NTU RGB+D)
  │
  ├── Step 1: Parse skeleton → (T_raw, V=25, C=3, M=2)
  ├── Step 2: Body selection → pick primary body per frame
  ├── Step 3: Normalization → spine-center, rotate-to-front, torso-scale
  ├── Step 4: Temporal resampling → uniformly subsample to T_target
  ├── Step 5: Stream generation → joint, bone, velocity, bone_velocity
  └── Step 6: Save as .npy → (N, C=3, T_target, V=25, M)
```

---

## 2. Step-by-Step

### Step 1: Parse Skeleton

Use the official NTU parser (`read_skeleton_official`). Handles:
- Multi-body detection (up to 4 bodies, we keep 2)
- Missing frame interpolation
- Invalid frame filtering (all-zero coordinates)

```python
bodymat = read_skeleton_official(file_path, max_body=4, njoints=25)
# Returns dict with 'skel_body0', 'skel_body1', each (T, V, 3)
```

### Step 2: Body Selection

```python
data = convert_official_to_numpy(bodymat, max_frames=T_target, max_bodies=2)
# Internally:
#   - If T_raw < T_target: repeat-pad last frame (NOT zero-pad)
#   - If T_raw > T_target: uniform subsample via np.linspace
#   - Shape: (C=3, T_target, V=25, M=2)
```

**Why repeat-pad, not zero-pad:**
Zero frames create artificial velocity spikes (v[t] = 0 - J[t-1] = -J[t-1]) and confuse
BN statistics. Repeat-padding keeps velocity ≈ 0 in padded region (semantically: frozen subject).

**Why uniform subsample, not crop:**
EfficientGCN subsamples uniformly so the model ALWAYS sees 100% of the action, just at
lower temporal resolution. Contiguous cropping risks missing the discriminative part.

### Step 3: Normalization

Applied BEFORE temporal resampling (on full-resolution data):

```python
# 1. Translate spine-base (joint 0) to origin
data = align_to_spine_base(data)       # x -= x[:, :, 0:1, :]

# 2. Rotate so torso points along Z-axis
data = rotate_to_front(data)           # R = rotation from spine→neck to (0,0,1)

# 3. Scale by torso length (neck-to-spine distance)
data = normalize_skeleton_scale(data)  # x /= ||joint[20] - joint[0]||
```

These are identical to what CTR-GCN, InfoGCN, EfficientGCN use.

### Step 4: Temporal Resampling

This is the **critical difference** from our current pipeline:

```python
# CURRENT (WRONG for model accuracy):
max_frames = 300    # Store 300 frames, crop 64 at train time
# Many frames are repeat-padded → random crop can land on padding

# NEW (EfficientGCN-style):
T_target = 64       # For LAST-Lite
T_target = 64       # For LAST-Base (same — model handles internally)
```

| Model | T_target | Rationale |
|-------|----------|-----------|
| LAST-Lite nano | 64 | Match EfficientGCN-B0 (uses T=20-64) |
| LAST-Lite small | 64 | Same — keep comparable |
| LAST-Base (single stream) | 64 | Standard for GCN models (CTR-GCN, InfoGCN) |
| LAST-Base (if needed) | 120 | Match EfficientGCN-B4 for maximum temporal coverage |

**At training time:** NO temporal crop needed. Data is already T_target frames.
**Augmentation:** Random rotation, scale, shear, noise (spatial only). No temporal crop.

```python
# NEW transform pipeline:
def get_train_transform(config):
    # NO TemporalCrop — data is already resampled to T_target
    transforms = []
    if aug_enabled:
        transforms.extend([
            RandomRotation((-15, 15)),
            RandomScale((0.9, 1.1)),
            RandomShear((-0.1, 0.1)),
            GaussianNoise(0.01),
        ])
    return Compose(transforms)

def get_val_transform(config):
    return None  # No transform needed — data is already the right shape
```

### Step 5: Stream Generation

Generate 4 streams from joint coordinates:

```python
def gen_bone_data(joint_data):
    """Bone = child_joint - parent_joint."""
    # 24 bone pairs from NTU skeleton topology
    bone_data = np.zeros_like(joint_data)
    for child, parent in NTU_BONE_PAIRS:
        bone_data[:, :, :, child, :] = joint_data[:, :, :, child, :] - joint_data[:, :, :, parent, :]
    return bone_data

def gen_velocity_data(data):
    """Velocity = frame[t+1] - frame[t]."""
    vel = np.zeros_like(data)
    vel[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
    return vel

# Generate all 4 streams:
joints = ...                              # (N, C, T, V, M)
bones = gen_bone_data(joints)             # bone vectors
joint_velocity = gen_velocity_data(joints) # joint motion
bone_velocity = gen_velocity_data(bones)   # bone angular motion
```

| Stream | Physical meaning | Captures |
|--------|-----------------|----------|
| Joint | Raw 3D coordinates | Spatial structure, pose |
| Bone | Limb direction vectors | Body proportions, limb angles |
| Joint velocity | Frame-to-frame position delta | Speed, acceleration of each joint |
| Bone velocity | Frame-to-frame bone delta | Angular velocity of limbs |

### Step 6: Save

For MIB (multi-input-branch) used by LAST-Lite:
```python
# 3 streams (joint, bone, velocity) — standard MIB
np.save(f'{split}_joint.npy', joints)
np.save(f'{split}_velocity.npy', joint_velocity)
np.save(f'{split}_bone.npy', bones)
pickle.dump(labels, f'{split}_label.pkl')
```

For LAST-Base (4 independent streams):
```python
# 4 streams — each trained as separate model
np.save(f'{split}_joint.npy', joints)
np.save(f'{split}_bone.npy', bones)
np.save(f'{split}_joint_velocity.npy', joint_velocity)
np.save(f'{split}_bone_velocity.npy', bone_velocity)
pickle.dump(labels, f'{split}_label.pkl')
```

---

## 3. Preprocessing for LAST-Lite vs LAST-Base

| Aspect | LAST-Lite | LAST-Base |
|--------|-----------|-----------|
| T_target | 64 | 64 (or 120 for high-res variant) |
| Streams saved | 3 (joint, bone, velocity) | 4 (+ bone_velocity) |
| Stream handling | MIB dict → StreamFusion early | Independent — each stream = separate model |
| Body handling | M=1 (primary body only) | M=2 (both bodies, concat or process separately) |
| Normalization | spine-center + rotate + torso-scale | Same |
| Temporal sampling | np.linspace uniform | Same |

### Body Handling for LAST-Base

For multi-person actions (15 NTU-60 classes), LAST-Base should use BOTH bodies:

```python
# Option A: Concat along temporal axis (double T)
# (C, T, V, M=2) → (C, T, V, 1) for body 0 and (C, T, V, 1) for body 1
# → concat → (C, 2T, V, 1)
# Pro: simple, model sees both bodies
# Con: doubles temporal dimension → 2× compute

# Option B: Process independently + aggregate (HD-GCN style)
# body0 = model(x[:, :, :, :, 0])  # (B, C_out)
# body1 = model(x[:, :, :, :, 1])  # (B, C_out)
# out = max(body0, body1)           # or mean, or learned gate
# Pro: same compute per body
# Con: no cross-body interaction

# Option C: Inter-body edges in graph (SkateFormer style)
# Create 50-joint graph: 25 joints × 2 bodies
# Add edges between corresponding joints across bodies
# → captures handshake = hand0 ↔ hand1
# Pro: captures inter-person interaction
# Con: V=50 → 4× attention cost
```

**Recommendation:** Option B for initial experiments (simple). Option C for max accuracy on
multi-person classes if there's a gap.

---

## 4. Implementation Changes Needed

### Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `scripts/preprocess_v2.py` | `max_frames=300` → `max_frames=64` | **P0** |
| `src/data/transforms.py` | Remove `TemporalCrop` from MIB train transform | **P0** |
| `src/data/transforms.py` | Remove center crop from MIB val transform | **P0** |
| `scripts/train.py` | `max_frames=300` → `max_frames=64` (lines 361, 372) | **P0** |
| `scripts/preprocess_v2.py` | Add `bone_velocity` stream generation + save | P1 |
| `src/data/dataset.py` | Support loading 4-stream data | P1 |

### New Files Needed

| File | Purpose | Priority |
|------|---------|----------|
| `scripts/preprocess_v3.py` | New preprocessor with configurable T_target, 4-stream output | P1 |

### Verification After Preprocessing Change

```bash
# After re-preprocessing with T_target=64:
python -c "
import numpy as np
data = np.load('path/to/train_joint.npy', mmap_mode='r')
print(f'Shape: {data.shape}')  # should be (N, 3, 64, 25, M)
print(f'Non-zero frames per sample (mean): {(np.abs(data).sum(axis=(1,3,4)) > 0).sum(axis=1).mean():.1f}')
# Should be ~64.0 (all frames are real — no padding for uniform subsample)
"
```

---

## 5. NTU RGB+D Skeleton Statistics

Useful for setting T_target:

| Statistic | NTU-60 | NTU-120 |
|-----------|--------|---------|
| Total samples | 56,880 | 114,480 |
| Min frames | 10 | 8 |
| Max frames | 300 | 300 |
| Mean frames | ~72 | ~70 |
| Median frames | ~64 | ~62 |
| 25th percentile | ~40 | ~38 |
| 75th percentile | ~95 | ~90 |
| Joints | 25 | 25 |
| Bodies | 1-2 | 1-2 |
| Classes | 60 | 120 |

**T_target=64** is optimal because it matches the median sequence length. Sequences shorter
than 64 are upsampled (np.linspace repeats frames). Sequences longer than 64 are downsampled
(np.linspace skips frames). In both cases, the FULL action is represented.

---

## 6. Quality Checks Post-Preprocessing

Run these BEFORE training to catch preprocessing bugs:

```bash
# 1. Shape check
python -c "import numpy as np; d=np.load('train_joint.npy',mmap_mode='r'); print(d.shape, d.dtype)"
# Expected: (N_train, 3, 64, 25, 2) float32

# 2. All-zero sample check
python -c "
import numpy as np
d = np.load('train_joint.npy', mmap_mode='r')
zero_samples = np.where(np.abs(d).sum(axis=(1,2,3,4)) == 0)[0]
print(f'All-zero samples: {len(zero_samples)} / {d.shape[0]}')
# Should be 0 (filtered during preprocessing)
"

# 3. Normalization check — spine base should be at origin
python -c "
import numpy as np
d = np.load('train_joint.npy', mmap_mode='r')
spine_base = d[:, :, :, 0, 0]  # joint 0, body 0
print(f'Spine-base mean: {np.abs(spine_base).mean():.6f}')
# Should be < 0.001 (centered at origin)
"

# 4. Velocity spike check — no large spikes from padding
python -c "
import numpy as np
d = np.load('train_velocity.npy', mmap_mode='r')
max_vel = np.abs(d).max()
print(f'Max velocity: {max_vel:.4f}')
# Should be < 0.5 (no padding → no velocity spikes)
"

# 5. Label distribution check
python -c "
import pickle, numpy as np
with open('train_label.pkl', 'rb') as f: labels = pickle.load(f)
unique, counts = np.unique(labels, return_counts=True)
print(f'Classes: {len(unique)}, Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.0f}')
"
```
