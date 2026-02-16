# Teacher-LAST: VideoMAE-2 Fine-tuning on NTU RGB+D 60

Fine-tune **VideoMAE-2 Large** (ViT-L/16) on **NTU RGB+D 60** for action recognition.  
Teacher model for knowledge distillation into the LAST skeleton-based system.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate annotations (cross-subject split)
python scripts/prepare_annotations.py \
    --video_root "E:\nturgbd-videos" \
    --output_dir "E:\teacher-last\annotations"

# 3. Fine-tune (1 GPU, actual_lr = 1.25e-4)
python scripts/finetune.py --config configs/ntu60_finetune.yaml

# 4. Fine-tune with 2 GPUs (actual_lr = 2.5e-4)
python scripts/finetune.py --config configs/ntu60_finetune.yaml --num_gpus 2

# 5. Dry run (2 epochs, debug mode)
python scripts/finetune.py --config configs/ntu60_finetune.yaml --epochs 2 --batch_size 2 --debug

# 6. Monitor training
tensorboard --logdir=E:\teacher-last\logs
```

## LR Scaling

```
actual_lr = base_lr × (batch_size × num_gpus × update_freq) / 256
```

| `--num_gpus` | Effective Batch | actual_lr |
|---|---|---|
| 1 (default) | 32 | 1.25e-4 |
| 2 | 64 | 2.5e-4 |
| 4 | 128 | 5e-4 |

## Frame Sampling (Official VideoMAE Behavior)

All sampling strategies follow the [official MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) implementation (`kinetics.py`).

| | Temporal | Spatial |
|--|--|--|
| **Training** | Random Temporal Crop → 16 frames | RandomCrop(224) + flip |
| **Validation** | Random Temporal Crop → 16 frames (same as train) | CenterCrop(224) |
| **Testing** | Multi-Clip Evaluation (10 temporal × 3 spatial) | 3-position crop |

### Long Videos (≥ 64 frames) — Random Temporal Crop

A random 64-frame window is selected, then 16 frames are sampled uniformly within it.
Each epoch sees a different temporal slice — free data augmentation.

### Short Videos (< 64 frames) — Temporal Repeat Padding (Last-Frame Repeat)

Many NTU RGB+D 60 actions (clapping, nodding, etc.) produce videos with < 64 frames.
These are handled using **temporal repeat padding** (official VideoMAE method):
- Sample as many real frames as possible at the sampling rate
- Pad remaining slots by **repeating the last frame**
- The video "freezes" on the final pose — natural for NTU actions where subjects stop after performing

This is the same for both training and validation. No temporal augmentation for short videos;
spatial augmentation (RandomCrop, flip) and batch-level Mixup/CutMix provide regularization.

### Testing — Multi-Clip Evaluation

All frames are loaded at the sampling rate (every 4th frame), then divided into
10 evenly-spaced temporal clips × 3 spatial crops = **30 forward passes per video**.
Predictions are averaged for the final deterministic result.

## Project Structure

```
Teacher-LAST/
├── configs/ntu60_finetune.yaml    # All hyperparameters
├── src/
│   ├── dataset.py                 # Video dataset + frame sampling
│   ├── model.py                   # VideoMAE-2 wrapper + layer decay
│   ├── trainer.py                 # Training engine (AMP, mixup, etc.)
│   └── utils.py                   # LR scaling, checkpointing, logging
├── scripts/
│   ├── prepare_annotations.py     # Generate train/val CSVs
│   └── finetune.py                # Main entry point
├── Docs/README.md
└── requirements.txt
```

## References

- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [VideoMAE V2](https://arxiv.org/abs/2303.16727)
- [NTU RGB+D 60](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
