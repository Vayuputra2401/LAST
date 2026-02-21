# 04 — Data Pipeline

## Datasets

### NTU RGB+D 60
- **Classes:** 60 action categories (daily actions, health-related, mutual actions)
- **Subjects:** 40, up to **M=2 bodies** per sequence
- **Split protocols:**
  - `xsub`: train/test split by subject ID (primary benchmark)
  - `xview`: split by camera viewpoint

### NTU RGB+D 120
- **Classes:** 120 action categories (superset of NTU60)
- **Subjects:** 106, up to M=2 bodies
- **Split protocols:**
  - `xsub`: subject-based split
  - `xset`: camera setup-based split

Both datasets provide 25-joint 3D skeleton coordinates per body.

---

## MIB Stream Definitions

| Stream   | Key        | Formula                           | Shape returned |
|----------|------------|-----------------------------------|----------------|
| Joint    | `'joint'`  | coords relative to spine base     | (C, T, V, M)   |
| Velocity | `'velocity'`| J_{t+1} − J_t (frame-diff)      | (C, T, V, M)   |
| Bone     | `'bone'`   | J_child − J_parent                | (C, T, V, M)   |

C=3 (x,y,z), T=frames, V=25 joints, M=2 bodies.

---

## Preprocessing

**Script:** `scripts/preprocess_v2.py`

Steps applied to raw NTU skeleton files:
1. **center_spine** — translate all joints so spine base (joint 0) is at origin
2. **scale_by_torso** — divide by torso length for scale invariance
3. **MIB stream extraction** — compute velocity (frame diff) and bone vectors
4. **Per-stream .npy save** — saves `joint.npy`, `velocity.npy`, `bone.npy` per split
5. **label.pkl** — class labels for each sequence

Preprocessed data is stored under `data/processed_v2/{xsub,xview,xset}/`.

---

## Dataset Class

**File:** `src/data/dataset.py`

```python
dataset = SkeletonDataset(
    data_path=...,
    data_type='mib',      # returns dict with 3 streams
    split='xsub',
    phase='train'
)

sample = dataset[i]
# Returns:
# {
#   'joint':    Tensor(C, T, V, M),
#   'velocity': Tensor(C, T, V, M),
#   'bone':     Tensor(C, T, V, M),
# }, label: int
```

The dataloader collates these into `(B, C, T, V, M)` tensors per key.

---

## Transforms

**File:** `src/data/transforms.py`

| Transform              | Train | Val/Test | Notes                                   |
|------------------------|-------|----------|-----------------------------------------|
| TemporalCrop(64)       | ✓     | ✓        | Random crop (train), center crop (test) |
| RandomRotation(±15°)   | ✓     | ✗        | Applied to J-stream only                |
| RandomScale(±10%)      | ✓     | ✗        | Uniform scale augmentation              |
| Normalize              | ✗     | ✗        | Done in preprocessing; skipped here     |

Transform config sets `normalize=False` — normalization is applied offline at preprocessing
time, not online during training.

---

## M-Dimension Handling

Both models receive `(B, C, T, V, M=2)` tensors but handle M differently:

**LAST-v2:** processes M internally per-stream, handles multi-body sequences.

**LAST-E:** strips M dimension in forward:
```python
s = data[key][..., 0]   # (B, C, T, V) — primary body only
```
This is a deliberate design choice to keep LAST-E's single backbone simple and efficient.
Secondary body information is not used, which is acceptable because most NTU sequences are
single-person and for two-person actions the primary body carries sufficient signal.

---

## DataLoader Configuration

Batch size and workers are set per environment (see `configs/environment/`):

| Environment  | Batch size (LAST-v2) | Batch size (LAST-E) | Workers | prefetch_factor |
|--------------|---------------------|---------------------|---------|-----------------|
| local        | 16                  | 16                  | 4       | 2               |
| kaggle (T4)  | 16                  | 16                  | 2       | 2               |
| gcp (P100)   | 16                  | 32                  | 8       | 2               |
| lambda (A10) | 32                  | 128                 | 12      | 4               |

Batch sizes are set via CLI `--batch_size`; workers and prefetch_factor are read automatically
from each environment's `hardware:` config block.
