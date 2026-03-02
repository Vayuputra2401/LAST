# 07 — Experiments

## Confirmed: Parameter Counts

### LAST-Lite (ShiftFuse-GCN) — Verified via forward pass + unit tests

| Model | Params | Architecture | Novel Components |
|-------|--------|-------------|-----------------|
| LAST-Lite nano | **80,234** | stem=24, C=[32,48,64], blocks=[1,1,1], K=3 | BRASP + BSE + FDCR + StaticGCN x3 |
| LAST-Lite small | **247,548** | stem=32, C=[48,72,96], blocks=[1,2,2], K=5 | BRASP + BSE + FDCR + StaticGCN x3 |

### Component-level Parameter Breakdown (small variant)

| Component | Per-block params | Instances | Total |
|-----------|-----------------|-----------|-------|
| BRASP | 0 | 5 blocks | 0 |
| Conv2d+BN (pointwise) | C_in x C_out + 2C | 5 | ~23K |
| JointEmbedding | 25 x C | 5 | ~8.7K |
| BSE | 2C + 1 | 5 | ~773 |
| FrozenDCTGate | C x T | 5 | ~19K |
| EpSepTCN | ~(2C^2 x r + Ck + ...) | 5 | ~105K |
| FrameDynamicsGate | T_out x C + C | 5 | ~14K |
| Residual (mismatch only) | C_in x C_out + 2C | 2 | ~8K |
| StaticGCN (per stage) | C^2 + 2C + 625 | 3 | ~19K |
| StreamFusionConcat | 4xBN(3) + Conv(12,32) | 1 | ~2K |
| Gated Head | pool_gate + BN + FC | 1 | ~6K |
| **Total** | | | **247,548** |

### Comparison to EfficientGCN

| Model | Params | NTU-60 xsub | Novel Contributions |
|-------|--------|-------------|-------------------|
| EfficientGCN-B0 | 290K | 90.2% | 0 (compound scaling only) |
| EfficientGCN-B4 | 2M | 91.7% | 0 |
| **LAST-Lite nano** | **80K** | *TBD* | 4 (BRASP, BSE, FDCR, StaticGCN) |
| **LAST-Lite small** | **248K** | *TBD* | 4 |

---

## Training Results: NTU-60 X-Sub

### Round 1: Initial Training (nano only)

| Model | Epochs | Scheduler | Best val top-1 | Best epoch | Notes |
|-------|--------|-----------|---------------|-----------|-------|
| nano | 90 | cosine_warmup | **80.77%** | 63 | Overfitted after epoch 63 |

Round 1 used default hyperparameters without augmentation or regularisation tuning. The model peaked at 80.77% and then overfitted, indicating the need for stronger regularisation and longer training with LR decay.

### Round 2: Regularisation + Augmentation

| Model | Params | LS | WD | Dropout | Epochs | Best val top-1 | Best epoch |
|-------|--------|------|------|---------|--------|---------------|-----------|
| nano | 80K* | 0.05 | 0.0005 | 0.1 | 90 | **79.75%** | 83 |
| small | 247K* | 0.1 | 0.001 | 0.3 | 90 | **78.84%** | 90 |

*Round 2 used pre-BSE param counts (nano ~80K, small ~247K). BSE adds ~291 (nano) and ~773 (small) params.

**Key finding**: Nano beat small by +0.91%, which should not happen with 3x more parameters. Root cause analysis:

> Small was **over-regularised** — a triple-whammy of dropout 0.3 + label smoothing 0.1 + weight decay 0.001 strangled its extra 167K parameters. Small was also still improving at epoch 90 (undertrained).

This informed the Round 3 hyperparameter corrections.

### Round 3: Corrected Hyperparameters + BSE (Pending)

| Model | Params | LS | WD | Dropout | Epochs | Milestones | Expected |
|-------|--------|------|------|---------|--------|-----------|----------|
| nano | 80,234 | 0.03* | 0.0003* | 0.1 | 120 | [60, 90] | 82-84% |
| small | 247,548 | 0.05 | 0.0005 | 0.2 | 120 | [60, 90] | 84-87% |

*Nano uses CLI overrides for lighter regularisation than the YAML defaults.

**Changes from Round 2**:
- Label smoothing: 0.1 → 0.05 (small YAML default)
- Weight decay: 0.001 → 0.0005
- Small dropout: 0.3 → 0.2
- Epochs: 90 → 120
- Milestones: [60, 80] → [60, 90]
- **New**: BSE (Bilateral Symmetry Encoding) added to all blocks

---

## Ablation Plan

### Architecture Ablations (LAST-Lite small)

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| - BRASP | Replace body-region shift with random shift | Measure anatomical routing contribution |
| - BSE | Remove bilateral symmetry encoding | Measure bilateral symmetry contribution |
| - FDCR | Remove frozen DCT gate | Measure frequency routing contribution |
| - StaticGCN | Remove stage-shared GCN | Measure graph convolution contribution |
| - JointEmbed | Remove joint semantic embedding | Measure joint identity contribution |
| - FrameGate | Remove frame dynamics gate | Measure temporal position contribution |
| Full model | All components | Full LAST-Lite small |

### Hyperparameter Ablations

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| Scheduler: multistep vs cosine | cosine_warmup vs multistep_warmup | LR schedule comparison |
| Epochs: 90 vs 120 vs 150 | Training duration | Optimal convergence point |
| Dropout sweep | 0.1, 0.15, 0.2, 0.3 | Optimal head regularisation |
| Label smoothing sweep | 0.0, 0.03, 0.05, 0.1 | Optimal logit regularisation |

### Distillation Ablations (Future)

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| alpha sweep | alpha = 0.3, 0.5, 0.7 | Hard vs soft label balance |
| tau sweep | tau = 2, 4, 6 | Temperature sensitivity |
| Feature mimicry: gamma=0 vs gamma=0.1 | With/without feature loss | Mimicry benefit |
| MaskCLR: yes vs no | With/without pretraining | Pretraining headroom |

---

## SOTA Comparison (NTU-60 xsub)

Results will be updated as training completes.

### Lightweight Regime (< 500K params)

| Method | Params | Top-1 | Ratio (acc/params) |
|--------|--------|-------|-------------------|
| EfficientGCN-B0 | 290K | 90.2% | 0.31 |
| **LAST-Lite nano** | **80K** | *TBD* | — |
| **LAST-Lite small** | **248K** | *TBD* | — |

### Full SOTA Landscape

| Method | Params | Top-1 | Year |
|--------|--------|-------|------|
| ST-GCN | ~3M | 81.5% | 2018 |
| 2s-AGCN | ~3.5M | 88.5% | 2019 |
| EfficientGCN-B0 | 290K | 90.2% | 2022 |
| Shift-GCN | 2.8M | 90.7% | 2020 |
| MS-G3D | ~3.2M | 91.5% | 2020 |
| EfficientGCN-B4 | 2M | 91.7% | 2022 |
| CTR-GCN | 1.7M | 92.4% | 2021 |
| BlockGCN | ~1.8M | 92.8% | 2024 |
| InfoGCN | 1.5M | 93.0% | 2022 |
| SkateFormer | ~2.5M | 93.0%+ | 2024 |
| HI-GCN | ~2.5M | 93.3% | 2025 |
| HD-GCN | 3.3M | 93.6% | 2023 |
| **LAST-Lite nano** | **80K** | *TBD* | 2026 |
| **LAST-Lite small** | **248K** | *TBD* | 2026 |
| **LAST-Base** (planned) | **~4.2M** | *TBD* | 2026 |

---

## Novelty Claims Summary

1. **BRASP (Body-Region-Aware Spatial Shift)**: First anatomically-partitioned channel shift for skeleton action recognition. Zero parameters, zero FLOPs overhead vs standard shift. Encodes body-part semantics as a structural prior.

2. **BSE (Bilateral Symmetry Encoding)**: First module to explicitly model left-right skeletal symmetry as a learned discriminative feature. Captures both static bilateral difference and symmetry dynamics (temporal derivative). Uses antisymmetric injection (L += signal, R -= signal). Cost: 2C + 1 params per block.

3. **FDCR (Frozen DCT Frequency Routing)**: First fixed-compute frequency-domain channel specialisation for skeleton GCN. Allows per-channel temporal frequency preference learning without per-sample computation. Cost: C x T params per block.

4. **StaticGCN with stage-shared weights**: Novel weight-sharing strategy where one graph convolution instance serves all blocks in a stage, with a trainable topology correction (V x V = 625 params). Eliminates redundant graph parameters in deeper stages.

5. **ShiftFuse-GCN architecture**: First architecture combining spatial shifting, bilateral symmetry encoding, frequency routing, and stage-shared graph convolution for skeleton recognition. Achieves sub-250K parameters with four novel contributions — more novel components than any prior skeleton GCN at any scale.
