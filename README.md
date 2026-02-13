# LAST: Lightweight Adaptive-Shift Transformer

Skeleton-based action recognition with efficient temporal modeling.

## 📁 Project Structure

```
LAST/
├── src/                          # Source code
│   ├── data/                     # Data loading ✅
│   │   ├── skeleton_loader.py    # .skeleton file parser
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/                   # Model architectures (TODO)
│   ├── training/                 # Training logic (TODO)
│   └── utils/                    # Utilities ✅
│       ├── config.py             # Config loader ✅
│       └── visualization.py      # Skeleton visualization
├── configs/                      # Configuration files ✅
│   ├── environment/              # Environment configs
│   │   ├── local.yaml            # Local development ✅
│   │   └── kaggle.yaml           # Kaggle execution ✅
│   └── data/                     # Dataset configs
│       └── ntu120.yaml           # NTU RGB+D 120 ✅
├── scripts/                      # Execution scripts
│   ├── load_data.py              # Config-driven data loading ✅
│   ├── test_dataloader.py        # Test/validation ✅
│   └── quick_test.py             # Quick validation ✅
├── tests/                        # Unit tests (TODO)
├── environment_setup.txt         # Python dependencies ✅
└── activate_ai.bat               # Environment activation ✅
```

## 🚀 Quick Start - Config-Driven Data Loading

### 1. Install Dependencies

```bash
# Activate environment
activate_ai.bat

# Install if not done yet
pip install -r environment_setup.txt
```

### 2. Configuration Files

The project uses **YAML configs** for all parameters:

**Environment configs** (`configs/environment/`):
- `local.yaml` - Local Windows development
- `kaggle.yaml` - Kaggle notebook execution

**Data configs** (`configs/data/`):
- `ntu120.yaml` - NTU RGB+D 120 dataset parameters

### 3. Load Data (Production Way)

```bash
# Local environment (auto-detected)
python scripts/load_data.py --split train

# Explicitly specify environment
python scripts/load_data.py --env local --split train

# Kaggle environment
python scripts/load_data.py --env kaggle --split val
```

### 4. Test Data Loader (Validation Only)

For testing/debugging only (not production):
```bash
python scripts/quick_test.py
```

## 📊 Data Format

**Input:** `.skeleton` files from NTU RGB+D 120
- 103 frames (example)
- 25 joints per frame
- 3D coordinates (x, y, z) in meters

**Output:** PyTorch tensors
- Shape: `(C, T, V, M)` = `(3, 300, 25, 2)`
- C = coordinates, T = frames, V = joints, M = max bodies

## 🎯 Next Steps

1. ✅ Data loading - **COMPLETED**
2. ⏳ Data preprocessing (.skeleton → .npy)
3. ⏳ Model implementation
4. ⏳ Training pipeline
5. ⏳ Evaluation

## 📝 Current Status

**Phase 1: Data Pipeline** - ✅ Core Implementation Done
- Skeleton file parser
- PyTorch Dataset with cross-subject/cross-setup splits
- 3D visualization utilities
- Test script for validation

Ready to test with your NTU RGB+D data!
