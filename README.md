# LAST: Lightweight Action Skeleton Transformer

Skeleton-based human action recognition with two complementary model families: a high-accuracy
teacher (LAST-v2, 9.2M params) and an extreme-efficiency student family (LAST-E, 92K–644K params).
LAST-E beats EfficientGCN at every parameter tier. Target venue: **ECCV 2026**.

---

## Status

| Model          | Params    | EfficientGCN target | Code | Training  | Top-1     |
|----------------|-----------|---------------------|------|-----------|-----------|
| LAST-E-nano    | 92,358    | <150K (B0) ✓        | PASS | Planned   | —         |
| LAST-E-small   | 177,646   | <300K (B1) ✓        | PASS | Planned   | —         |
| LAST-E-base    | 363,958   | <2M (B4) ✓          | PASS | Kaggle    | —         |
| LAST-E-large   | 644,094   | <2M (B4) ✓          | PASS | Planned   | —         |
| LAST-v2-base   | 9,217,256 | — (teacher)         | PASS | GCP       | —         |

All integration tests pass. GPU training runs pending.

---

## Quick Start

**Kaggle (T4 16GB) — LAST-E baseline:**
```bash
python scripts/train.py --model base_e --dataset ntu60 --env kaggle --amp
```

**Local — smoke test (CPU or any GPU):**
```bash
python scripts/train.py --model nano_e --dataset ntu60 --epochs 2 --batch_size 4
```

**GCP P100 — LAST-v2 teacher:**
```bash
python scripts/train.py --model base --dataset ntu60 --env gcp --amp
```

**Verify models and param counts:**
```bash
python tests/test_model_integration.py
```

---

## Architecture Summary

**LAST-v2 (Teacher):** 3 independent per-stream backbones (joint / velocity / bone), each with
AdaptiveGraphConv + ST_JointAtt + LinearAttention. Logits summed at end. Maximum accuracy.

**LAST-E (Student):** StreamFusion blends all 3 streams at input → single shared backbone with
LightGCNBlocks (DirectionalGCNConv + MultiScaleTCN). 3× fewer FLOPs than LAST-v2.

---

## Documentation

| Section | File | Description |
|---------|------|-------------|
| 01 | [Introduction](Docs/01_Introduction.md) | Problem, motivation, contributions |
| 02 | [Related Work](Docs/02_Related_Work.md) | SOTA table, EfficientGCN comparison |
| 03 | [Architecture](Docs/03_Architecture.md) | LAST-v2 + LAST-E full design |
| 04 | [Data Pipeline](Docs/04_Data_Pipeline.md) | MIB streams, NTU60/120, preprocessing |
| 05 | [Training](Docs/05_Training.md) | Optimizer, scheduler, commands |
| 06 | [Distillation](Docs/06_Distillation.md) | LAST-v2 → LAST-E KD plan |
| 07 | [Experiments](Docs/07_Experiments.md) | Param counts + results (live) |
| 08 | [Environment Setup](Docs/08_Environment_Setup.md) | Local / Kaggle / GCP setup |

---

## Project Structure

```
LAST/
├── src/
│   ├── models/
│   │   ├── last_v2.py              # Teacher model
│   │   ├── last_e.py               # Student model
│   │   ├── graph.py                # Adjacency matrices (K=3 subsets)
│   │   └── blocks/
│   │       ├── eff_gcn.py          # EffGCNBlock (LAST-v2)
│   │       ├── light_gcn.py        # LightGCNBlock (LAST-E)
│   │       └── stream_fusion.py    # StreamFusion (LAST-E input)
│   ├── data/
│   │   ├── dataset.py              # SkeletonDataset (MIB dict output)
│   │   └── transforms.py           # TemporalCrop, RandomRotation, RandomScale
│   ├── training/
│   │   └── trainer.py              # Trainer (AMP, grad accum, SequentialLR)
│   └── utils/
│       └── config.py               # Config loader + env auto-detection
├── configs/
│   ├── model/                      # Per-model YAML configs
│   ├── training/default.yaml       # Training hyperparameters
│   └── environment/                # local / kaggle / gcp YAMLs
├── scripts/
│   ├── train.py                    # Training entry point
│   └── preprocess_v2.py            # MIB stream preprocessing
├── tests/
│   └── test_model_integration.py   # Integration tests (all PASS)
└── Docs/                           # Research paper documentation
```
