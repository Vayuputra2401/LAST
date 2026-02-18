# LAST v2 Pipeline Walkthrough

This document explains the new files and the updated flow for the LAST v2 framework.

## 1. New File Structure

### Data Pipeline (Phase 1)
*   **`scripts/preprocess_v2.py`**: The new preprocessing script.
    *   **Role**: Generates 3 streams (Joint, Velocity, Bone) from raw `.skeleton` files.
    *   **Key Logic**: Uses `official_loader` to parse, then applies `normalization` (View-Invariant), and computes Velocity/Bone differences.
    *   **Output**: `train_joint.npy`, `train_velocity.npy`, `train_bone.npy` (and same for val).

*   **`src/data/official_loader.py`**:
    *   **Role**: Exact implementation of the NTU RGB+D "Official" parser you provided.
    *   **Logic**: Reads `.skeleton` text files, handles missing bodies, and prunes abundant bodies exactly as the original logic.

*   **`src/data/dataset.py` (Updated)**:
    *   **Change**: Added `data_type='mib'` support.
    *   **Logic**: When `mib` is selected, `__getitem__` returns a **Dictionary** `{'joint': ..., 'velocity': ..., 'bone': ...}` instead of a single tensor.

### Architecture (Phase 2)
*   **`src/models/last_v2.py`**:
    *   **Role**: The main model class for v2.
    *   **Logic**: Instantiates the backbone (EffGCN). In `forward`, if input is a dict (MIB), it runs the backbone 3 times (with shared weights) and sums the scores (Late Fusion).
    *   **Variants**: Configurable `small`, `base`, `large` (defined in `MODEL_VARIANTS`).

*   **`src/models/blocks/eff_gcn.py`**:
    *   **Role**: The core building block.
    *   **Logic**: Combines Spatial GCN (Separable), ST-Joint Attention, and Temporal Modeling (TCN or Linear Attention) with Residual connections.

*   **`src/models/blocks/st_joint_att.py`**:
    *   **Role**: New "Spatial-Temporal Joint Attention" module.
    *   **Logic**: Factorized attention to refine features based on Joint importance and Key Frames.

*   **`src/models/blocks/linear_attn.py`**:
    *   **Role**: O(T) complexity attention for efficient long-sequence modeling in deeper layers.

## 2. Updated Flow

### Configuration
The system allows full configuration via YAML files.
1.  **`configs/training/default.yaml`**: Global training settings (Epochs, LR, Batch Size).
2.  **`configs/data/{dataset}.yaml`**: Data settings (e.g., `ntu60.yaml`).
    *   Set `data_type: "mib"` to enable V2 multi-stream loading.
3.  **`configs/model/last_{variant}.yaml`**: Model specific settings.
    *   `last_base.yaml`, `last_small.yaml`, `last_large.yaml` control the model version and variant.
    *   `model.version: "v2"` triggers the new architecture.

### Training (`scripts/train.py`)
1.  **Load Config**: 
    *   Loads `default.yaml`.
    *   Loads `configs/data/{dataset}.yaml`.
    *   Loads `configs/model/last_{args.model}.yaml`.
    *   Merges them into a single `config` dictionary.
2.  **Initialize Data**: 
    *   Checks `config['data']['dataset']['data_type']` (set to `mib`).
    *   Creates `SkeletonDataset` loading 3 streams.
3.  **Initialize Model**: 
    *   Checks `config['model']['version']` (set to `v2`).
    *   Instantiates `LAST_v2(variant=...)`.

## 3. How to Run

1.  **Preprocess Data** (Once):
    ```bash
    python scripts/preprocess_v2.py
    ```
    This uses `official_loader` and creates the `_joint.npy`, `_velocity.npy`, `_bone.npy` files.

2.  **Train**:
    ```bash
    python scripts/train.py --model small --dataset ntu60
    ```
    (No need to specify v2 in CLI if it's default in config, or we can add CLI support later).

## 4. Verification
Run the verification scripts to confirm everything is connected:
```bash
python tests/verify_data_v2.py
python tests/verify_model_v2.py
```
