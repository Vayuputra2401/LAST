# 04 — Data Pipeline

## Datasets

### NTU RGB+D 60

The primary benchmark for this work.

| Property | Value |
|----------|-------|
| Action classes | 60 |
| Subjects | 40 actors |
| Bodies per sequence | up to M=2 |
| Joints per body | 25 (Kinect v2 skeleton) |
| Primary protocol | **xsub** — train/test split by subject ID |
| Secondary protocol | xview — train/test split by camera viewpoint |

**xsub split**: 20 training subjects, 20 test subjects. All experiments use xsub unless stated otherwise, as it is the standard protocol for fair comparison with all prior work.

### NTU RGB+D 120

Extension of NTU-60 with 60 additional action categories.

| Property | Value |
|----------|-------|
| Action classes | 120 |
| Subjects | 106 actors |
| Bodies per sequence | up to M=2 |
| Protocols | xsub (subject split), xset (camera-setup split) |

*NTU-120 experiments are planned post NTU-60 validation.*

---

## 4-Stream Kinematic Decomposition

Following the multi-stream paradigm of 2s-AGCN and EfficientGCN, we decompose each skeleton sequence into four complementary kinematic representations. Each stream captures a distinct and non-redundant aspect of human motion:

| Stream | Key | Formula | Shape | Captured Information |
|--------|-----|---------|-------|---------------------|
| Joint | `joint` | Coordinates centred at joint 0 (spine base) | (3, T, V, M) | Absolute body pose configuration |
| Velocity | `velocity` | J[t+1] − J[t]; last frame duplicated | (3, T, V, M) | Instantaneous motion speed and direction |
| Bone | `bone` | J[child] − J[parent] per NTU skeleton edge | (3, T, V, M) | Limb orientation, length, and posture |
| Bone velocity | `bone_velocity` | B[t+1] − B[t]; last frame duplicated | (3, T, V, M) | Limb angular velocity and dynamics |

**Complementarity analysis:**

| Action | Joint signal | Velocity signal | Bone signal | Bone vel. signal |
|--------|-------------|----------------|-------------|-----------------|
| Standing still | Stable, distinctive pose | Near-zero | Stable orientations | Near-zero |
| Walking | Periodic trajectory | Periodic motion | Periodic limb angles | Periodic angular vel. |
| Throwing | Global arm trajectory | Strong arm velocity | Arm extension pattern | High angular acceleration |
| Clapping | Symmetric joint positions | Symmetric velocity | Symmetric limb alignment | High frequency dynamics |

The four streams are processed by a single shared backbone (stacked along the batch dimension). No explicit cross-stream fusion occurs within the backbone — streams interact implicitly through the shared BatchNorm statistics.

---

## Preprocessing

**Script**: `scripts/preprocess_v2.py`

Applied offline to raw NTU skeleton files before training:

### Steps

1. **Centre at spine base** — subtract joint 0 (spine base) coordinates from all joints per frame. Removes global position.

2. **Scale by torso length** — divide by the Euclidean distance from joint 0 (spine base) to joint 1 (spine shoulder). Removes height-dependent scale differences. Applied consistently across all M bodies.

3. **Temporal resampling** — sequences are uniformly resampled to T=64 frames using bilinear interpolation. Sequences shorter than 64 frames are padded with the last frame.

4. **Stream extraction**:
   - **Velocity**: `velocity[t] = joint[t+1] − joint[t]` for t < T−1; `velocity[T−1] = velocity[T−2]` (last frame duplicated)
   - **Bone**: `bone[v] = joint[child(v)] − joint[parent(v)]` using the official NTU kinematic tree (24 edges for 25 joints; root joint bone is zero-padded)
   - **Bone velocity**: `bone_velocity[t] = bone[t+1] − bone[t]` (same convention as velocity)

5. **Save per-split**: separate `.npy` files for each stream + `label.pkl` per split.

### Verified Data Statistics (NTU-60 xsub)

| Split | Samples | Shape | Classes | Per-class balance |
|-------|---------|-------|---------|-------------------|
| train | 40,320 | (40320, 3, 64, 25, 2) | 60 | **672 per class (perfectly balanced)** |
| val | 16,560 | (16560, 3, 64, 25, 2) | 60 | **276 per class (perfectly balanced)** |

- All 60 classes present in both splits; no NaN or Inf values confirmed
- Joint coordinate range after normalisation: [−2.849, 4.881]
- Velocity stream accuracy: `max_abs_err = 0.0` vs frame-diff of joint stream
- Bone stream accuracy: `max_abs_err = 0.0` vs child-parent diff of joint stream

---

## Multi-Body Handling (M=2)

Raw NTU data can contain M=2 bodies per sequence. ShiftFuse V10 takes the primary body (M=0 index):

```python
x = x[..., 0]   # (B, 3, T, V, M=2) → (B, 3, T, V)
```

**Rationale**: For the NTU-60 action categories, the primary actor (body 0) provides sufficient discriminative signal. Most actions are single-person; for two-person interactions, body 0 is the action initiator. A full M=2 interaction model would add substantial complexity without proportional accuracy gain at the nano scale.

*Multi-body interaction modelling is left for the large variant or future work.*

---

## Training-Time Data Augmentation

**File**: `src/data/transforms.py`

Applied per-batch, online, to the joint stream only. Velocity, bone, and bone_velocity are not augmented (they are pre-computed offsets; augmenting them independently would create inconsistencies).

| Augmentation | Parameter | Applied to | Purpose |
|-------------|-----------|-----------|---------|
| Random rotation | ±15° per axis (roll/pitch/yaw) | Joint stream | Viewpoint robustness |
| Random scale | ±10% uniform | Joint stream | Scale robustness |
| Random temporal speed | 0.9–1.1× frame sampling rate | All streams | Temporal speed variation |

**Mixup/CutMix: disabled** (`mixup_alpha=0.0`, `cutmix_prob=0.0`).
Rationale: AMP float16 + IB triplet loss + 10-epoch warmup → NaN losses on mixed batches during the first 10 epochs. The triplet IB loss computes nearest-wrong-prototype distances which become undefined for mixed-label samples. Mixup is re-evaluated post V10.3 validation.

---

## Dataset Class

**File**: `src/data/dataset.py`

```python
dataset = SkeletonDataset(
    data_path='data/processed/xsub/',
    split='train',
    max_frames=64,
)
# Returns: (stream_dict, label_int)
# stream_dict = {
#     'joint':         Tensor(3, 64, 25),   # M=0 selected
#     'velocity':      Tensor(3, 64, 25),
#     'bone':          Tensor(3, 64, 25),
#     'bone_velocity': Tensor(3, 64, 25),
# }
```

The DataLoader collates into `(stream_dict_batched, labels)` where each stream tensor is `(B, 3, 64, 25)`.

---

## DataLoader Configuration

| Environment | Batch size | Workers | pin_memory | Gradient accum. | Effective batch |
|-------------|-----------|---------|-----------|-----------------|-----------------|
| Kaggle T4 | 24 | 2 | True | 3 steps | **72** |
| Local | 24 | 4 | True | 3 steps | **72** |

The effective batch of 72 matches the training batch size used by InfoGCN (64) and CTR-GCN (64), ensuring the SGD+momentum optimiser sees comparable gradient estimates.

---

## Data Paths

Resolved from environment config at runtime:

```yaml
# configs/environment/kaggle.yaml
data_folder: '/kaggle/input/LAST-60'

# configs/environment/local.yaml
data_folder: 'E:/LAST-60'
```

Full path: `{data_folder}/data/processed/xsub/{split}/{stream}.npy`

---

## NTU Skeleton Tree (25 joints, for reference)

```
                    Head (3)
                      │
                    Neck (2)
                      │
              ┌── SpineShoulder (1) ──┐
              │                        │
        ShoulderL (4)           ShoulderR (8)
              │                        │
         ElbowL (5)              ElbowR (9)
              │                        │
         WristL (6)             WristR (10)
              │                        │
         HandL (7)              HandR (11)
         HandTipL (21)          HandTipR (23)
         ThumbL (22)            ThumbR (24)
              │
           SpineBase (0) ──── HipL (16)  HipR (20)
                                 │           │
                              KneeL (17) KneeR (21)
                                 │           │
                            AnkleL (18) AnkleR (22)
                                 │           │
                             FootL (19) FootR (23)
               Also: SpineMid (12), SpineBase2 (13), Neck2 (14) ... (dataset variation)
```

*Note: exact joint numbering follows the NTU RGB+D 25-joint convention as defined in the preprocessing script.*
