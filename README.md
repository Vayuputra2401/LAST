# ShiftFuse-Zero: Zero-Parameter Anatomical Priors for Efficient Skeleton Action Recognition

**ShiftFuse-Zero** is a family of lightweight Graph Convolutional Networks (GCNs) designed for high-performance skeleton-based human action recognition on resource-constrained edge hardware.

The core philosophy, **"Anatomy is Free"**, introduces spatial inductive bias entirely without learnable parameters through two novel zero-parameter modules:
1. **BRASP** (Body-Region Anatomical Shift): Routes joint features through anatomically meaningful groups via pure tensor indexing.
2. **SGPShift** (Semantic Graph-Partitioned Shift): Aggregates intra-part and inter-part neighbours through pre-defined group indices.

Developed for **ACM MM 2026**.

---

## Status: Model Family

| Model | Params | Architecture | Features | Top-1 (NTU-60 xsub) |
|-------|--------|--------------|----------|---------------------|
| **Nano** | ~94K | 3-stream early fusion | Global A_learned share | 88.5% (KD) |
| **Small** | ~284K | 2-backbone orthogonal | PartAttention, velocity-split | 89.8% (KD) |
| **Medium** | ~590K | 2-backbone (B2-scale) | PartAttention, deep stage-2 | 92.0% (target) |
| **Large (B4)** | ~1.12M | Mid-network fusion | B4-exact topology, TLA | 92.5% |
| **X (SOTA)** | ~2.0M | Scaled mid-fusion | Multi-PartAtt, high capacity | 93-94% (target) |

*Note: Medium and X variants are currently in development/testing.*

---

## Architecture: "Anatomy is Free"

### EfficientZero Block
The backbone of ShiftFuse-Zero is the **EfficientZero Block**, which optimizes the accuracy-per-parameter frontier by reordering operations:
- **Zero-param foundation:** BRASP and SGPShift structure the manifold *before* learnable layers.
- **Lightweight GCN:** A compact $K=3$ GCN refines the anatomically-structured representation.
- **DS-TCN Stack:** Depthwise-Separable Temporal Convolutions provide 3x speedup vs. standard $9 \times 1$ convolutions.
- **PartAttention:** Anatomical gating placed at the structural bottlenecks of Small/Medium/X variants.

### Late Fusion: Orthogonal Split (Small/Medium)
Unlike traditional early fusion, our late-fusion models use an **orthogonal split** between two backbones:
- **Backbone A:** Processes Joint + Velocity (capturing trajectory and dynamics).
- **Backbone B:** Processes Bone (capturing structural pose/skeleton shape).
This allows each backbone to specialize in a fundamental physical modality before exchanging information via **Cross-Stream Fusion**.

---

## Quick Start

**Profile models and verify parameter counts:**
```bash
python scripts/train.py --model shiftfuse_nano --profile_only
```

**Training SFZ-Nano (distilled from Large):**
```bash
python scripts/train.py --model shiftfuse_nano --teacher shiftfuse_large --dataset ntu60 --epochs 300
```

---

## Project Structure

```
LAST/
├── src/
│   ├── models/
│   │   ├── shiftfuse_zero.py          # ShiftFuse-Zero Family (Nano to X)
│   │   ├── graph.py                   # Adjacency matrices (K-subset spatial)
│   │   └── blocks/
│   │       ├── body_region_shift.py   # BRASP (0-param anatomical shift)
│   │       ├── sgp_shift.py           # SGPShift (0-param semantic aggregation)
│   │       ├── temporal_landmark_attn.py # TLA (gated temporal landmarks)
│   │       ├── part_attention.py      # PartAttention (anatomical gating)
│   │       ├── dw_sep_tcn.py          # DS-TCN (Efficient temporal conv)
│   │       └── stream_fusion_concat.py # Multi-stream concatenation stems
│   ├── training/
│   │   └── trainer.py                 # Advanced trainer (A_edge support, AdamW/SGD)
│   └── data/
│       ├── dataset.py                 # SkeletonDataset (Joint/Bone/Velocity/BV)
│       └── transforms.py              # TemporalCrop, Augmentations
├── configs/
│   ├── model/                         # YAML configs for Nano, Small, Medium, B4, X
│   ├── training/                      # Training schedules (300/150 epochs)
│   └── environment/                   # local / kaggle / server YAMLs
└── Paper/                             # ACM MM 2026 Research Paper (main.tex)
```
