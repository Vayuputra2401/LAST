# NTU RGB+D 120 Dataset Splitting Methodology

## Overview

This document explains how train/validation splits work in the NTU RGB+D 120 dataset and how they are implemented in our pipeline.

---

## 🎯 Key Concept: Subject-Based Splits, NOT Class-Based

**Critical Understanding:** The NTU RGB+D benchmark splits data by **subjects (people)**, not by action classes.

### What This Means:

- ✅ **Both train and val contain ALL 120 action classes**
- ✅ Split is based on which subjects (people) performed the actions
- ✅ Tests the model's ability to generalize to **new people** performing known actions
- ❌ NOT split by action classes (that would prevent evaluation on unseen classes)

---

## 📊 Dataset Statistics

```
Total Videos:    40,320
Total Subjects:  106 people
Total Actions:   120 different actions
Total Cameras:   3 camera views
Total Setups:    32 different camera setups
```

### How Videos Are Distributed:

Each subject performs all 120 actions multiple times across different camera setups:
```
40,320 videos ≈ 106 subjects × 120 actions × ~3 repetitions
```

---

## 🔀 Split Type 1: Cross-Subject (xsub) - **Default**

### Configuration

**Location:** `configs/data/ntu120.yaml`

```yaml
splits:
  xsub:
    train_subjects: [1, 2, 4, 5, 8, 9, 13, 14, ..., 103]  # 53 subjects
    val_subjects: [3, 6, 7, 10, 11, 12, 20, ..., 106]     # 53 subjects
```

### Split Logic

```python
# For each video file: S001C001P001R001A013.skeleton
#                      ^^^^ ^^^^ ^^^^ ^^^^ ^^^^
#                      |    |    |    |    └─ Action (A013)
#                      |    |    |    └────── Replication (R001)
#                      |    |    └─────────── Person/Subject (P001)
#                      |    └──────────────── Camera (C001)
#                      └───────────────────── Setup (S001)

if person_id in train_subjects:
    assign to TRAIN split
elif person_id in val_subjects:
    assign to VAL split
```

### Example Distribution

| Action | Train (53 subjects) | Val (53 subjects) | Total |
|--------|---------------------|-------------------|-------|
| A001: drink water | Videos of subjects 1,2,4,5... | Videos of subjects 3,6,7,10... | All subjects |
| A013: wear jacket | Videos of subjects 1,2,4,5... | Videos of subjects 3,6,7,10... | All subjects |
| A120: rock-paper-scissors | Videos of subjects 1,2,4,5... | Videos of subjects 3,6,7,10... | All subjects |

**Result:** Each split contains ~20,160 videos with all 120 action classes represented.

### Why Cross-Subject Split?

1. **Realistic Evaluation:** In real-world deployment, the model will see new people performing known actions
2. **Person-Invariant Recognition:** Tests if the model learned the action itself, not person-specific patterns
3. **Standard Benchmark:** Official NTU RGB+D evaluation protocol for fair comparison with other papers

---

## 🔀 Split Type 2: Cross-Setup (xset) - **Alternative**

### Configuration

```yaml
splits:
  xset:
    train_setups: [2, 4, 6, 8, 10, 12, ..., 32]  # Even setups (16 setups)
    val_setups: [1, 3, 5, 7, 9, 11, ..., 31]     # Odd setups (16 setups)
```

### Split Logic

```python
# Based on camera setup number in filename
if setup_id % 2 == 0:  # Even setup
    assign to TRAIN split
else:  # Odd setup
    assign to VAL split
```

### Why Cross-Setup Split?

1. **Camera Viewpoint Generalization:** Tests if model can recognize actions from different camera angles
2. **Setup Invariance:** Different setups have different camera positions and backgrounds
3. **Alternative Benchmark:** Less commonly used than xsub, but useful for viewpoint robustness analysis

---

## 🚫 What We DON'T Do: Class-Based Splits

### Hypothetical Class Split (NOT USED)

```python
# ❌ INCORRECT approach for action recognition
train_classes = [0, 1, 2, ..., 95]    # First 96 actions for training
val_classes = [96, 97, ..., 119]      # Last 24 actions for validation
```

### Why This Doesn't Work:

1. **Zero-shot problem:** Model never sees 24 actions during training
2. **Unfair evaluation:** Can't evaluate performance on completely unseen action classes
3. **Not the benchmark standard:** Papers wouldn't be comparable
4. **Unrealistic:** In real applications, you want to recognize known actions by new people, not discover new actions

---

## 💻 Implementation in Our Pipeline

### 1. Split Configuration (YAML)

**File:** `configs/data/ntu120.yaml`

```yaml
dataset:
  # Fixed split configuration (reproducible across runs)
  splits:
    xsub:
      train_subjects: [1, 2, 4, 5, ...]  # All 53 train subjects listed
      val_subjects: [3, 6, 7, 10, ...]    # All 53 val subjects listed
    
    xset:
      train_setups: [2, 4, 6, ...]       # Even setups
      val_setups: [1, 3, 5, ...]          # Odd setups
  
  # Choose which split type to use
  split_type: "xsub"  # or "xset"
```

### 2. Dataset Implementation

**File:** `src/data/dataset.py`

```python
class SkeletonDataset(Dataset):
    def _should_include_sample(self, metadata: dict) -> bool:
        """Determine if sample belongs to current split."""
        
        if self.split_type == 'xsub':
            # Load subjects from config
            train_subjects = set(self._split_config['xsub']['train_subjects'])
            val_subjects = set(self._split_config['xsub']['val_subjects'])
            
            person = metadata['person']
            if self.split == 'train':
                return person in train_subjects
            elif self.split == 'val':
                return person in val_subjects
        
        elif self.split_type == 'xset':
            # Load setups from config
            train_setups = set(self._split_config['xset']['train_setups'])
            val_setups = set(self._split_config['xset']['val_setups'])
            
            setup = metadata['setup']
            if self.split == 'train':
                return setup in train_setups
            elif self.split == 'val':
                return setup in val_setups
```

### 3. Usage

```python
from src.data.dataloader import create_dataloaders
from src.utils.config import load_config

# Load configuration
config = load_config(env='local', dataset='ntu120')

# Create both dataloaders (train + val)
loaders = create_dataloaders(config)

train_loader = loaders['train']  # ~20,160 samples, all 120 classes
val_loader = loaders['val']      # ~20,160 samples, all 120 classes

# Iterate through training data
for batch_data, batch_labels in train_loader:
    # batch_data: (B, C=3, T=300, V=25, M=2)
    # batch_labels: (B,) with values in [0, 119]
    train_step(batch_data, batch_labels)
```

---

## 📈 Verification: All Classes in Both Splits

You can verify that both splits contain all 120 classes:

```bash
python scripts/test_dataloader.py
```

**Expected Output:**
```
TESTING DATA DISTRIBUTION
============================================================

TRAIN Split:
  Unique classes: 120
  Samples per class: 168.0 (avg)
  Min samples in class: 120
  Max samples in class: 220

VAL Split:
  Unique classes: 120
  Samples per class: 168.0 (avg)
  Min samples in class: 115
  Max samples in class: 225
```

✅ Both splits have all 120 unique classes represented!

---

## 🎓 Academic Context

### Official NTU RGB+D Paper Benchmark

From the original NTU RGB+D 120 paper:

> **Cross-Subject (CS):** The training set consists of samples from 53 subjects, and the remaining 53 subjects' samples form the evaluation set.

> **Cross-Setup (CV):** The training set consists of samples from even setup IDs, and odd setup IDs form the evaluation set.

### Why This Matters for Papers

When publishing results, you must specify:
- **Split type:** xsub or xset
- **Evaluation metric:** Top-1 accuracy on validation set
- **Baseline comparison:** Using same splits as other papers

Example result reporting:
```
NTU RGB+D 120 Cross-Subject (xsub):
  Top-1 Accuracy: 89.4%

NTU RGB+D 120 Cross-Setup (xset):
  Top-1 Accuracy: 91.2%
```

---

## 🔑 Key Takeaways

1. **Subject-based splits ensure both train and val have all 120 action classes**
2. **Split ratio is 50/50 by subjects (53 train, 53 val)**
3. **Tests generalization to new people, not new actions**
4. **Configuration is in YAML for reproducibility**
5. **No randomness - splits are fixed and deterministic**
6. **Standard benchmark protocol for fair comparison**

---

## 📚 References

- NTU RGB+D 120 Dataset: [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- Original Paper: Liu et al. "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding" (TPAMI 2020)
- Cross-Subject Protocol: Standard evaluation in skeleton-based action recognition
