# 07 — Experiments

## Status: Phase A+B+D Running on Kaggle T4

LAST-E v3 base training is in progress. This file is the living results table.

---

## Confirmed: Parameter Counts

### LAST-E v3 (Verified via forward pass)

| Model | Params | EfficientGCN target | Key features |
|-------|--------|---------------------|-------------|
| LAST-E v3 nano | **82,847** | < B0 (150K) ✓ | 1-hop, MotionGate, 3 blocks |
| LAST-E v3 small | **344,727** | < B2 (540K) ✓ | 2-hop, MotionGate, subset_att, 5 blocks |
| LAST-E v3 base | **720,028** | < B4 (2M) ✓ | 2-hop, MotionGate, subset_att, IB loss, 6 blocks |
| LAST-E v3 large | **1,080,668** | < B4 (2M) ✓ | 2-hop, HybridGate, subset_att, IB loss, 6 blocks |

### Phase B Ablation: use_st_att=[F, F, T]

| Model | Default params | Phase B params | Saved | % |
|-------|---------------|---------------|-------|--|
| LAST-E v3 base | 720,028 | 692,764 | 27,264 | 3.8% |

### LAST-Lite (Planned — Edge Deployment)

| Model | Est. Params | vs EfficientGCN | Key change |
|-------|------------|-----------------|-----------|
| LAST-Lite nano | ~60K | 60% smaller than B0 | No gates, no attention, fixed graph |
| LAST-Lite small | ~180K | 44% smaller than B0 | No gates, no attention, fixed graph |

---

## Pending: NTU60 X-Sub Accuracy

### LAST-E v3 Standalone Training

| Model | Phase | Scheduler | drop_path | use_st_att | Top-1 | Top-5 | Epochs | Status |
|-------|-------|-----------|-----------|-----------|-------|-------|--------|--------|
| v3 base | A+B+D | SGDR (T₀=30) | 0.15 | [F,F,T] | — | — | 90 | **Running** |
| v3 base | A only | cosine | 0.05 | [T,T,T] | — | — | 120 | Pending |
| v3 nano | — | — | — | — | — | — | — | Pending |
| v3 small | — | — | — | — | — | — | — | Pending |
| v3 large | — | — | — | — | — | — | — | Pending |

### Post-Distillation (LAST-E v3 → LAST-Lite)

| Student | Teacher | Pretrain | Top-1 | Δ vs standalone | α | τ | γ |
|---------|---------|----------|-------|----------------|---|---|---|
| Lite nano | v3 base | None | — | — | 0.3 | 4.0 | 0.1 |
| Lite nano | v3 base | MaskCLR | — | — | 0.3 | 4.0 | 0.1 |
| Lite small | v3 base | None | — | — | 0.5 | 4.0 | 0.1 |
| Lite small | v3 base | MaskCLR | — | — | 0.5 | 4.0 | 0.1 |

### Edge Deployment (INT8 Quantized)

| Model | FP32 acc | INT8 acc | Model size | Inference (CPU) |
|-------|----------|---------|-----------|----------------|
| Lite nano INT8 | — | — | ~15 KB | — |
| Lite small INT8 | — | — | ~45 KB | — |

---

## Expected Accuracy Targets (NTU60 xsub)

| Model | Standalone | + KD | + MaskCLR + KD | EfficientGCN ref |
|-------|-----------|------|---------------|-----------------|
| LAST-E v3 nano (83K) | 85–87% | — | — | B0: 88.3% (150K) |
| LAST-E v3 small (345K) | 87–89% | — | — | B2: 90.2% (540K) |
| LAST-E v3 base (720K) | 89–91% | — | — | B4: 91.7% (2M) |
| LAST-E v3 large (1.08M) | 91–93% | — | — | B4: 91.7% (2M) |
| LAST-Lite nano (60K) | ~83% | ~86% | ~88% | B0: 88.3% (150K) |
| LAST-Lite small (180K) | ~86% | ~89% | ~91% | B0: 88.3% (150K) |

---

## Ablation Plan

### Architecture Ablations (LAST-E v3 base)

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| ST_JointAtt ablation | use_st_att=[F,F,T] vs [T,T,T] | Measure attention contribution per stage |
| MotionGate ablation | gate_type=none vs motion | Quantify MotionGate contribution |
| HybridGate vs MotionGate | gate_type=hybrid vs motion | SE-style benefit at base budget |
| Subset attention ablation | use_subset_att=False | Impact of per-sample subset weighting |
| IB loss ablation | use_ib_loss=False | Information bottleneck contribution |
| Drop path sweep | 0.0, 0.05, 0.10, 0.15, 0.20 | Optimal stochastic depth |

### Training Recipe Ablations

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| Scheduler: cosine vs SGDR | cosine_warmup vs cosine_warmup_restart | SGDR implicit regularization |
| SGDR T₀ sweep | T₀ = 15, 30, 45 | Optimal restart period |
| Epochs: 90 vs 120 vs 140 | Longer training | Cosine needs room to decay |
| Effective batch: 32 vs 64 vs 128 | grad_accum 1 vs 2 vs 4 | Gradient quality vs speed |

### Distillation Ablations

| Experiment | What changes | Purpose |
|-----------|-------------|---------|
| α sweep (Lite small) | α = 0.3, 0.5, 0.7 | Hard vs soft label balance |
| τ sweep (Lite small) | τ = 2, 4, 6 | Temperature sensitivity |
| Feature mimicry: γ=0 vs γ=0.1 | With/without feature loss | Mimicry benefit |
| MaskCLR pretrain: yes vs no | With/without pretraining | Pretraining headroom |

### Novel Ideas Ablations (Future)

| Experiment | Model | Purpose |
|-----------|-------|---------|
| FreqTemporalGate (Idea A) | v3 base | Frequency-domain channel attention |
| Action-Prototype Graph (Idea B) | v3 base | Class-conditioned topology |
| Causal Training (Idea E) | v3 base + Lite | 50% causal masking benefit |

---

## SOTA Comparison (NTU60 xsub — update as results arrive)

| Method | Params | Top-1 | Efficiency ratio |
|--------|--------|-------|-----------------|
| EfficientGCN-B0 | ~150K | 88.3% | baseline |
| EfficientGCN-B2 | ~540K | 90.2% | — |
| EfficientGCN-B4 | ~2M | 91.7% | — |
| Shift-GCN | 2.8M | 90.7% | — |
| CTR-GCN | 1.7M | 92.4% | — |
| InfoGCN | 1.5M | 93.0% | — |
| HD-GCN | 3.3M | 93.6% | — |
| BlockGCN | ~1.8M | 92.8% | — |
| SkateFormer | ~2.5M | 93.0%+ | — |
| HI-GCN | ~2.5M | 93.3% | — |
| **LAST-E v3 nano** | **83K** | *TBD* | — |
| **LAST-E v3 small** | **345K** | *TBD* | — |
| **LAST-E v3 base** | **720K** | *TBD* | — |
| **LAST-E v3 large** | **1.08M** | *TBD* | — |
| **LAST-Lite nano** | **~60K** | *TBD* | — |
| **LAST-Lite small** | **~180K** | *TBD* | — |
| **LAST-v2 base** | **9.2M** | *TBD* | — |
