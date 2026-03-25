# ShiftFuse-Zero: Anatomy is Free

**ShiftFuse-Zero (SFZ)** is a family of efficient GCNs for skeleton-based human action recognition on NTU RGB+D 60/120. The core insight — *"Anatomy is Free"* — is that anatomical spatial structure can be injected with **zero parameters** via two novel modules:

- **BRASP** (Body-Region Anatomical Shift): routes joint features through anatomically meaningful groups via pure tensor indexing.
- **SGPShift** (Semantic Graph-Partitioned Shift): aggregates intra-part and inter-part neighbours through pre-defined graph index buffers.

Both modules run at 0 parameters, 0 FLOPs overhead, and require no changes to training.

---

## Model Family

| Model | CLI name | Params | Fusion | NTU-60 xsub |
|-------|----------|--------|--------|-------------|
| **Nano** | `shiftfuse_zero_nano_tiny_efficient` | ~94K | 3-stream early | 86.24% (88.5% +KD) |
| **Small** | `shiftfuse_zero_small_late_efficient` | ~284K | 2-backbone late | 88.0% (target 90.5%+KD) |
| **Medium** | `shiftfuse_zero_medium_late_efficient` | ~594K | 2-backbone late | *(next to train)* |
| **Large (B4)** | `shiftfuse_zero_large_b4_efficient` | ~1.18M | 3-stream mid-fusion | *(next to train)* |
| **X** | `shiftfuse_zero_x_efficient` | ~2.0M | 3-stream mid-fusion | *(next to train)* |

Nano and Small are confirmed results (NTU-60 xsub, 300 epochs). Medium and X are implemented and ready to train.

---

## Architecture

### EfficientZero Block (Nano / Small / Medium)
```
BRASP → SGPShift → JE → STC-Attention → GCN(K=3, A_learned) → DS-TCN → DropPath → Residual
[TLA at last block of last stage]
```

- **BRASP + SGPShift**: zero-parameter anatomical routing (our novel contribution)
- **JE**: per-joint identity embedding (V×C params)
- **STC-Attention**: spatial + temporal gates
- **GCN**: K=3 partitions + learnable additive residual A_learned
- **DS-TCN**: dual-dilation depthwise-separable TCN (dil 1 + dil 2, ~17-frame receptive field)
- **TLA**: Temporal Landmark Attention O(T·K) — last block only

### Late Fusion (Small / Medium)
Two-backbone split: Backbone A (joint + velocity) and Backbone B (bone), fused via CrossStreamFusion after stage 2 (early) and stage 3 (late).

### Mid-Fusion (Large / X)
Three separate per-stream backbones → concat at mid-network → shared backbone.

---

## Training

**Train Nano:**
```bash
python scripts/train.py --model shiftfuse_zero_nano_tiny_efficient --dataset ntu60 --amp
```

**Train Small:**
```bash
python scripts/train.py --model shiftfuse_zero_small_late_efficient --dataset ntu60 --amp
```

**Train with Knowledge Distillation (Nano from Large):**
```bash
python scripts/train.py \
  --model shiftfuse_zero_nano_tiny_efficient \
  --dataset ntu60 --amp \
  --teacher_checkpoint /path/to/large_best.pth \
  --kd_weight 0.5 --kd_temp 4.0
```

**Train Large (B4-style):**
```bash
python scripts/train.py --model shiftfuse_zero_large_b4_efficient --dataset ntu60 --amp
```

**Environment override:**
```bash
python scripts/train.py --model shiftfuse_zero_nano_tiny_efficient --env kaggle --dataset ntu60
```

**Arbitrary config override:**
```bash
python scripts/train.py --model shiftfuse_zero_nano_tiny_efficient \
  --set training.lr=0.05 training.epochs=200 model.dropout=0.15
```

---

## Project Structure

```
LAST/
├── src/
│   ├── models/
│   │   ├── shiftfuse_zero.py           # Full model family (Nano → X)
│   │   ├── graph.py                    # Skeleton graph (K=3 semantic body-part)
│   │   └── blocks/
│   │       ├── body_region_shift.py    # BRASP (0-param anatomical shift)
│   │       ├── sgp_shift.py            # SGPShift (0-param semantic aggregation)
│   │       ├── stc_attention.py        # STC-Attention (spatial + temporal gates)
│   │       ├── joint_embedding.py      # JE (per-joint identity bias)
│   │       ├── temporal_landmark_attn.py  # TLA (O(T·K) temporal anchors)
│   │       ├── part_attention.py       # PartAttention (5-group anatomy gating)
│   │       ├── dw_sep_tcn.py           # DS-TCN (dual-dilation depthwise-sep TCN)
│   │       ├── drop_path.py            # DropPath stochastic depth
│   │       ├── stream_fusion_concat.py # Multi-stream BN + concat stem
│   │       └── cross_stream_fusion.py  # CrossStreamFusion (late fusion gate)
│   ├── training/
│   │   └── trainer.py                  # Trainer (SGD/AdamW, KD, Mixup, TTA, AMP)
│   └── data/
│       ├── dataset.py                  # SkeletonDataset (joint/bone/velocity)
│       └── transforms.py              # TemporalCrop, spatial flip, augmentations
├── configs/
│   ├── model/                          # Per-variant model YAML configs
│   ├── training/                       # Per-variant training schedules
│   └── environment/                   # local / kaggle / gcp / a100 YAMLs
├── scripts/
│   └── train.py                        # Training entry point
└── tests/
    └── test_shiftfuse.py               # Model build + forward pass tests
```

---

## Knowledge Distillation

KD is available for all models. Set in the training config or via CLI:

```yaml
# In configs/training/shiftfuse_zero_nano_efficient.yaml
use_kd: true
teacher_checkpoint: "/path/to/large_best.pth"
teacher_variant: "large_b4_efficient"   # or small_late_efficient_bb, etc.
teacher_num_classes: 60
kd_weight: 0.5        # blend: (1-w)·CE + w·T²·KL
kd_temperature: 4.0
```

Loss: `(1 − α)·CrossEntropy + α·T²·KL(student ‖ teacher)`

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `numpy`, `pyyaml`, `tqdm`.

Data: NTU RGB+D preprocessed `.npy` files at `<data_base>/LAST-60/data/processed/`.
