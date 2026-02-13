# Data Pipeline Architecture & File Explanations

## 📂 Directory Structure

```
LAST/
├── src/data/                    # Core data pipeline modules
│   ├── __init__.py             # Package exports
│   ├── skeleton_loader.py      # Raw .skeleton file parser
│   ├── dataset.py              # PyTorch Dataset class
│   ├── ntu120_actions.py       # Action class mapping (120 classes)
│   ├── ntu120_actions.txt      # Human-readable action reference
│   ├── preprocessing.py        # Normalization utilities
│   ├── transforms.py           # Data augmentation pipeline
│   └── dataloader.py           # DataLoader creation utilities
│
├── configs/
│   ├── environment/            # Environment-specific configs
│   │   ├── local.yaml         # Local development (Windows, E: drive)
│   │   └── kaggle.yaml        # Kaggle notebooks
│   └── data/
│       └── ntu120.yaml         # Dataset configuration & splits
│
├── scripts/                    # Production scripts
│   └── load_data.py           # Main data loading (production use)
│
└── tests/                      # Testing scripts
    └── test_dataloader.py     # Comprehensive data pipeline tests
```

---

## 🔧 Core Data Pipeline Files

### 1. `src/data/skeleton_loader.py`

**Purpose:** Low-level parser for raw `.skeleton` files (NTU RGB+D format)

**What it does:**
- Reads text-based `.skeleton` files from disk
- Parses frame-by-frame skeleton joint positions
- Extracts metadata from filename (setup, camera, person, action, etc.)
- Handles multi-body scenarios (up to 2 people per frame)

**Key Functions:**
```python
class SkeletonFileParser:
    def parse_file(filename) -> (skeleton_data, metadata)
        # Returns: (T, V, C, M) array + metadata dict
        # T = frames, V = 25 joints, C = 3 coords (x,y,z), M = 2 bodies
    
    def extract_metadata_from_filename(filename) -> dict
        # Example: S001C001P001R001A013.skeleton
        # Returns: {setup: 1, camera: 1, person: 1, action: 12}
```

**When it's used:**
- During dataset initialization when `data_type='skeleton'`
- Called by `SkeletonDataset.__getitem__()` to load individual samples

**Input:** `S001C001P001R001A013.skeleton` (text file)
**Output:** NumPy array (T, V, C, M) + metadata dict

---

### 2. `src/data/dataset.py`

**Purpose:** PyTorch `Dataset` wrapper for skeleton data

**What it does:**
- Scans directory for `.skeleton` files
- Filters files based on train/val split (from config)
- Loads individual samples on-demand via `__getitem__()`
- Applies temporal transformations (pad/sample to max_frames)
- Applies user-defined transforms (normalization, augmentation)
- Returns batches in PyTorch tensor format

**Key Class:**
```python
class SkeletonDataset(Dataset):
    def __init__(data_path, split='train', split_type='xsub', split_config=None)
        # Scans files, filters by split
    
    def __getitem__(idx) -> (data, label)
        # Load skeleton file -> normalize -> transform
        # Returns: (C, T, V, M) tensor, int label
    
    def _should_include_sample(metadata) -> bool
        # Checks if file belongs to current split (train/val)
```

**When it's used:**
- Created by `create_dataloader()` function
- PyTorch's `DataLoader` calls `__getitem__()` to fetch batches

**Input:** Directory path + split config
**Output:** PyTorch Dataset (iterable)

---

### 3. `src/data/ntu120_actions.py`

**Purpose:** Action class label ↔ name mapping

**What it does:**
- Stores dictionary mapping: `{0: "drink water", 1: "eat meal/snack", ..., 119: "rock-paper-scissors"}`
- Provides conversion functions between labels and names
- Validates action labels

**Key Functions:**
```python
NTU120_ACTIONS = {0: "drink water", ..., 119: "rock-paper-scissors"}

def get_action_name(label: int) -> str
    # Example: get_action_name(12) → "tear up paper"

def get_action_label(name: str) -> int
    # Example: get_action_label("tear up paper") → 12

def validate_label(label: int) -> bool
    # Check if label is in [0, 119]
```

**When it's used:**
- During logging/visualization to show action names
- In test scripts to display readable results
- Optional during training for interpretable outputs

**Companion File:** `ntu120_actions.txt` (human-readable reference)

---

### 4. `src/data/preprocessing.py`

**Purpose:** Skeleton normalization and preprocessing utilities

**What it does:**
- Centers skeleton on a reference joint (SpineBase)
- Scales skeleton by torso length (for size invariance)
- Provides multiple normalization methods
- Temporal cropping (pad/sample sequences)
- Body selection (primary body from multi-body frames)

**Key Functions:**
```python
def normalize_skeleton(skeleton, method='center_spine')
    # Center on joint 0, scale by torso length
    # Input: (C, T, V, M) or (T, V, C, M)
    # Output: Normalized skeleton (same shape)

def normalize_skeleton_by_center(skeleton, center_joint=0, scale_by_torso=True)
    # Detailed normalization:
    # 1. SpineBase (joint 0) → origin
    # 2. Scale by distance(SpineBase, Neck)

def temporal_crop(skeleton, max_frames)
    # Pad (if T < max_frames) or sample (if T > max_frames)

def select_primary_body(skeleton)
    # For multi-body: select body with most non-zero frames
```

**When it's used:**
- Called by `transforms.Normalize` during data loading
- Can be used offline in preprocessing scripts

**Example:**
```python
skeleton = np.random.randn(3, 100, 25, 2)  # Raw skeleton
normalized = normalize_skeleton(skeleton, method='center_spine')
# Now SpineBase is at (0,0,0) for all frames
```

---

### 5. `src/data/transforms.py`

**Purpose:** Composable transform pipeline for augmentation

**What it does:**
- Provides transform classes (similar to `torchvision.transforms`)
- Normalization transform (wraps `preprocessing.py`)
- Data augmentation: rotation, scaling, shear, noise
- Separate train/val transform factories from config

**Key Classes:**
```python
class Compose:
    # Chain multiple transforms: Compose([Normalize(), RandomRotation()])

class Normalize:
    # Apply skeleton normalization
    
class RandomRotation:
    # Rotate around Y-axis (vertical) by random angle
    
class RandomScale:
    # Scale skeleton by random factor
    
class RandomShear:
    # Apply shear transformation
    
class GaussianNoise:
    # Add random noise to coordinates
```

**Key Functions:**
```python
def get_train_transform(config) -> Compose
    # Returns: Normalize + Augmentation (if enabled)

def get_val_transform(config) -> Compose
    # Returns: Normalize only (no augmentation)
```

**When it's used:**
- Created by `create_dataloader()` based on config
- Applied by `SkeletonDataset.__getitem__()` before returning data

**Example:**
```python
transform = Compose([
    Normalize(method='center_spine'),
    RandomRotation(angle_range=(-15, 15)),
    RandomScale(scale_range=(0.9, 1.1))
])

# Applied to each sample during training
data = transform(skeleton)
```

---

### 6. `src/data/dataloader.py`

**Purpose:** High-level utilities for creating PyTorch DataLoaders

**What it does:**
- Unified API for creating train/val dataloaders
- Reads configuration from YAML
- Creates datasets with proper transforms
- Sets split-specific parameters (shuffle, drop_last)
- Returns ready-to-use DataLoader objects

**Key Functions:**
```python
def create_dataloader(config, split='train') -> DataLoader
    # Single dataloader for specified split
    # Handles: dataset creation, transform selection, dataloader params
    
def create_dataloaders(config) -> dict
    # Create both train and val loaders at once
    # Returns: {'train': DataLoader, 'val': DataLoader}
    
def get_dataloader_info(dataloader) -> dict
    # Returns metadata: num_samples, num_batches, batch_size, etc.
```

**When it's used:**
- In training scripts to create data loaders
- In test scripts to verify pipeline

**Example:**
```python
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders

config = load_config(env='local', dataset='ntu120')
loaders = create_dataloaders(config)

# Ready to use!
for batch_data, batch_labels in loaders['train']:
    # batch_data: (B, 3, 300, 25, 2)
    # batch_labels: (B,)
    train_step(batch_data, batch_labels)
```

---

## 📋 Configuration Files

### 7. `configs/environment/local.yaml`

**Purpose:** Environment-specific paths and hardware settings

**Contains:**
```yaml
environment:
  name: "local"
  type: "local"

paths:
  data_root: "E:\\nturgbd_skeletons_s001_to_s017\\nturgb+d_skeletons"
  processed_data: "E:\\LAST\\data\\processed"
  output_root: "E:\\LAST\\outputs"
  checkpoints: "E:\\LAST\\outputs\\checkpoints"
  logs: "E:\\LAST\\outputs\\logs"

hardware:
  device: "cuda"
  num_workers: 4
  pin_memory: true
```

**When it's used:**
- Auto-detected by `load_config()` based on environment
- Provides data paths, output directories, hardware settings

---

### 8. `configs/data/ntu120.yaml`

**Purpose:** Dataset parameters and split configuration

**Contains:**
```yaml
dataset:
  # Basic parameters
  data_type: "skeleton"
  num_joints: 25
  max_frames: 300
  max_bodies: 2
  
  # FIXED train/val splits (reproducible)
  splits:
    xsub:
      train_subjects: [1, 2, 4, 5, ..., 103]  # 53 subjects
      val_subjects: [3, 6, 7, 10, ..., 106]   # 53 subjects
    xset:
      train_setups: [2, 4, 6, ..., 32]        # Even setups
      val_setups: [1, 3, 5, ..., 31]          # Odd setups
  
  split_type: "xsub"
  
  # Normalization
  preprocessing:
    normalize: true
    normalization_method: "center_spine"
    center_joint: 0
    scale_by_torso: true
  
  # Augmentation
  augmentation:
    enabled: false  # Disabled by default
    rotation_range: [-15, 15]
    scale_range: [0.9, 1.1]

dataloader:
  batch_size: 32
  shuffle: true
  drop_last: true
  num_workers: 4
```

**When it's used:**
- Loaded by `create_dataloader()` to configure dataset
- Determines which subjects go to train vs val
- Sets normalization and augmentation parameters

---

## 🚀 Production Scripts

### 9. `scripts/load_data.py`

**Purpose:** Main production script for data loading

**What it does:**
- Loads configuration
- Creates dataloader for specified split
- Iterates through batches
- Displays progress and statistics
- Can be used as template for training scripts

**Usage:**
```bash
# Load training data
python scripts/load_data.py --split train

# Load validation data
python scripts/load_data.py --split val --env local
```

**When it's used:**
- Quick data loading check
- Template for training loop
- Debugging data pipeline

---

## 🧪 Test Scripts

### 10. `tests/test_dataloader.py`

**Purpose:** Comprehensive data pipeline testing

**What it does:**
- Tests train/val split correctness
- Verifies normalization is working
- Checks batch loading (no NaN/Inf)
- Tests transforms
- Analyzes data distribution
- Generates visualizations (optional)

**Test Functions:**
```python
def test_splits(loaders)
    # Verify train/val sizes, split type

def test_normalization(loader)
    # Check SpineBase is centered, data range

def test_batch_loading(loader, num_batches=3)
    # Load batches, check shapes, detect NaN/Inf

def test_transforms(config)
    # Verify transforms are applied

def test_data_distribution(loaders)
    # Check all 120 classes in both splits

def visualize_samples(loader, output_dir='.')
    # Generate skeleton plots
```

**Usage:**
```bash
# Run all tests
python tests/test_dataloader.py

# Run with visualizations
python tests/test_dataloader.py --visualize

# Test only train split
python tests/test_dataloader.py --split train --num_batches 5
```

**Expected Output:**
```
============================================================
LAST - Data Pipeline Testing
============================================================
✓ Environment: local
✓ Dataset: ntu120
✓ Data path: E:\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons

TESTING SPLITS
============================================================
TRAIN Split:
  Samples: 20,160
  Batches: 630
✓ Total samples: 40,320
✓ Train/Val ratio: 1.00

TESTING NORMALIZATION
============================================================
SpineBase position: 0.000023 (centered ✓)

TESTING BATCH LOADING
============================================================
Batch 1: (32, 3, 300, 25, 2) ✓ No NaN/Inf

✅ ALL TESTS PASSED
```

---

## 🔄 Complete Data Pipeline Workflow

### Training Scenario:

```
1. USER runs training script
   ↓
2. load_config(env='local', dataset='ntu120')
   ↓ Reads: configs/environment/local.yaml
   ↓       configs/data/ntu120.yaml
   ↓
3. create_dataloaders(config)
   ↓
4. For train split:
   ├─ Create SkeletonDataset(split='train', split_config=...)
   │  ├─ Scan E:\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons\
   │  ├─ Filter files by train_subjects [1,2,4,5,...]
   │  └─ Store ~20,160 file paths
   │
   ├─ Create train transform: Normalize + Augmentation
   │
   └─ Create DataLoader(dataset, batch_size=32, shuffle=True)
   
5. For val split: (same process, but val_subjects, no augmentation)

6. Training loop:
   for epoch in range(num_epochs):
       for batch_data, batch_labels in train_loader:
           # batch_data shape: (32, 3, 300, 25, 2)
           
           # Behind the scenes for each sample:
           # SkeletonDataset.__getitem__(idx)
           # ├─ SkeletonFileParser.parse_file() → (T, V, 3, 2)
           # ├─ Temporal crop/pad → (300, 25, 3, 2)
           # ├─ Transpose → (3, 300, 25, 2)
           # ├─ Apply transforms:
           # │  ├─ Normalize (center spine, scale torso)
           # │  ├─ RandomRotation (-15° to +15°)
           # │  └─ RandomScale (0.9x to 1.1x)
           # └─ Return tensor + label
           
           # Forward pass
           predictions = model(batch_data)
           loss = criterion(predictions, batch_labels)
           
           # Backward pass
           loss.backward()
           optimizer.step()
```

---

## 📊 Data Flow Diagram

```
Raw Data (.skeleton files)
         ↓
   [SkeletonFileParser]
         ↓
   (T, V, C, M) NumPy array
         ↓
   [SkeletonDataset]
   • Filter by split
   • Temporal transform
   • Transpose to (C, T, V, M)
         ↓
   [Transforms Pipeline]
   • Normalize
   • Augment (train only)
         ↓
   PyTorch Tensor (C, T, V, M)
         ↓
   [DataLoader]
   • Batch collation
   • Shuffle (train)
   • Multi-worker loading
         ↓
   Batch: (B, C, T, V, M)
         ↓
   [Model Training]
```

---

## 🎯 Key Design Principles

1. **Config-Driven:** All parameters in YAML (reproducible)
2. **Modular:** Each file has single responsibility
3. **Composable:** Transforms can be mixed and matched
4. **Efficient:** Memory-mapped for .npy, on-demand for .skeleton
5. **Standard:** Follows PyTorch Dataset/DataLoader conventions
6. **Tested:** Comprehensive test suite

---

## 💡 Usage Tips

### Quick Start:
```python
from src.utils.config import load_config
from src.data import create_dataloaders

config = load_config()
loaders = create_dataloaders(config)

# Start training!
for data, labels in loaders['train']:
    train_step(data, labels)
```

### Modify Split:
Edit `configs/data/ntu120.yaml`:
```yaml
split_type: "xset"  # Change from xsub to xset
```

### Enable Augmentation:
Edit `configs/data/ntu120.yaml`:
```yaml
augmentation:
  enabled: true
```

### Change Batch Size:
Edit `configs/data/ntu120.yaml`:
```yaml
dataloader:
  batch_size: 64  # Change from 32
```

---

## 📚 File Dependencies

```
test_dataloader.py
├── depends on: dataloader.py
│   └── depends on: dataset.py, transforms.py
│       ├── depends on: skeleton_loader.py
│       ├── depends on: ntu120_actions.py
│       ├── depends on: preprocessing.py
│       └── depends on: config.py (in src/utils)

load_data.py
└── depends on: dataloader.py (same tree as above)
```

All components work together to provide a clean, config-driven data pipeline! 🚀
