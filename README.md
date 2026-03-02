# LAST: Lightweight Action recognition via Shift-based Topology

Skeleton-based human action recognition with two model families: **LAST-Lite** (ShiftFuse-GCN),
a novel edge-deployable architecture with 4 original contributions at sub-250K parameters, and
**LAST-Base** (planned), a high-accuracy research model for SOTA competition and knowledge distillation.

Target venue: **ECCV 2026**.

---

## Status

| Model | Params | Architecture | Training | Top-1 (NTU-60 xsub) |
|-------|--------|-------------|----------|---------------------|
| LAST-Lite nano | 80,234 | stem=24, C=[32,48,64], blocks=[1,1,1] | Round 2 done | 79.75% |
| LAST-Lite small | 247,548 | stem=32, C=[48,72,96], blocks=[1,2,2] | Round 2 done | 78.84% |
| LAST-Base (planned) | ~4.2M | Cross-temporal + prototype + attention | Planned | — |

Round 3 (corrected hyperparameters + BSE) pending.

---

## Quick Start

**Kaggle (T4 16GB) — LAST-Lite small:**
```bash
python scripts/train.py --model shiftfuse_small --dataset ntu60 --env kaggle --amp --avg_checkpoints 5
```

**Local — smoke test (CPU or any GPU):**
```bash
python scripts/train.py --model shiftfuse_nano --dataset ntu60 --epochs 2 --batch_size 4
```

**Verify models and param counts:**
```bash
python -m pytest tests/test_shiftfuse.py -v
```

---

## Architecture Summary

**LAST-Lite (ShiftFuse-GCN):** 4-stream input fusion → single backbone with ShiftFuseBlocks.
Each block: BRASP (anatomical shift) → Conv1x1 → JointEmbed → BSE (bilateral symmetry) →
FDCR (frequency gate) → EpSepTCN → FrameGate → Residual → StaticGCN (shared per stage).

**LAST-Base (Planned):** 4 independent per-stream backbones with CrossTemporalPrototypeGCN,
FreqTemporalGate, PartitionedAttention, and HierarchicalBodyRegion. Late ensemble at inference.

### Novel Contributions (LAST-Lite)

1. **BRASP** — Body-Region-Aware Spatial Shift (0 params, anatomy-guided channel routing)
2. **BSE** — Bilateral Symmetry Encoding (2C+1 params, L-R symmetry as feature)
3. **FDCR** — Frozen DCT Frequency Routing (CxT params, frequency-domain specialisation)
4. **StaticGCN** — Stage-shared graph conv with learnable topology correction (C^2 + 2C + 625)

---

## Documentation

| Section | File | Description |
|---------|------|-------------|
| 01 | [Introduction](Docs/01_Introduction.md) | Problem, motivation, contributions |
| 02 | [Related Work](Docs/02_Related_Work.md) | SOTA landscape, generational analysis |
| 03 | [Architecture](Docs/03_Architecture.md) | LAST-Lite block design, LAST-Base overview |
| 04 | [Data Pipeline](Docs/04_Data_Pipeline.md) | 4-stream preprocessing, augmentation |
| 05 | [Training](Docs/05_Training.md) | Optimiser, scheduler, regularisation |
| 06 | [Distillation](Docs/06_Distillation.md) | KD + MaskCLR pretraining plan |
| 07 | [Experiments](Docs/07_Experiments.md) | Results, ablations, SOTA comparison |
| 08 | [Environment Setup](Docs/08_Environment_Setup.md) | Local / Kaggle setup |

---

## Project Structure

```
LAST/
├── src/
│   ├── models/
│   │   ├── shiftfuse_gcn.py          # LAST-Lite model (nano + small)
│   │   ├── graph.py                   # Adjacency matrices (K-subset spatial)
│   │   └── blocks/
│   │       ├── body_region_shift.py   # BRASP (novel, 0 params)
│   │       ├── bilateral_symmetry.py  # BSE (novel, 2C+1 params)
│   │       ├── frozen_dct_gate.py     # FDCR (novel, CxT params)
│   │       ├── static_gcn.py          # StaticGCN (novel, stage-shared)
│   │       ├── joint_embedding.py     # SGN-style joint identity
│   │       ├── frame_dynamics_gate.py # SGN-style temporal gate
│   │       ├── ep_sep_tcn.py          # EfficientGCN temporal conv
│   │       └── stream_fusion_concat.py # 4-stream concatenation fusion
│   ├── data/
│   │   ├── dataset.py                 # SkeletonDataset (4-stream dict output)
│   │   └── transforms.py             # TemporalCrop, RandomRotation, RandomScale
│   ├── training/
│   │   └── trainer.py                 # Trainer (AMP, schedulers, NaN guard)
│   └── utils/
│       └── config.py                  # Config loader + env auto-detection
├── configs/
│   ├── model/                         # Per-model YAML configs
│   ├── training/shiftfuse.yaml        # LAST-Lite training hyperparameters
│   └── environment/                   # local / kaggle YAMLs
├── scripts/
│   ├── train.py                       # Training entry point
│   └── preprocess_v2.py               # 4-stream preprocessing
├── tests/
│   └── test_shiftfuse.py             # Integration tests (all PASS)
└── Docs/                              # Research paper documentation
```
