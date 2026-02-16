# LAST Training Pipeline — Competitor Analysis & Strategy

## 1. Competitor Training Pipelines Summary

| Model | Optimizer | LR | LR Schedule | Epochs | Batch | Weight Decay | Loss |
|-------|-----------|-----|-------------|--------|-------|--------------|------|
| **EfficientGCN** | SGD+Nesterov | 0.1 | Warmup 10ep + SGDR cosine | 70 | 128 | 1e-4 | CE |
| **SGN** | SGD+Nesterov | 0.05 | Step decay ×0.1 at ep 110, 120 | 120+ | 64 | 4e-4 | CE |
| **Shift-GCN** | SGD+Nesterov | 0.1 | Step decay ×0.1 at ep 60, 80, 100 | 140 | 64 | 1e-4 | CE |
| **MS-G3D** | SGD+Nesterov | 0.1 | Step decay ×0.1 at ep 30, 40 | 50 | 64 | 1e-4 | CE |
| **CTR-GCN** | SGD+Nesterov | 0.1 | Step decay ×0.1 at ep 35, 55 | 80 | 64 | 4e-4 | CE |
| **InfoGCN** | SGD+Nesterov | 0.1 | Step decay | 80 | 64 | 4e-4 | CE + IB + LS(0.1) |

> [!IMPORTANT]
> **Universal patterns:** Every SOTA model uses SGD with Nesterov momentum at 0.9. Most use lr=0.1 with some form of step decay. Batch sizes are 64–128. Label smoothing (0.1) is emerging as standard.

---

## 2. Data Augmentation Strategies

### Spatial Augmentations
| Technique | Description | Used By |
|-----------|-------------|---------|
| **Random Rotation** | Rotate skeleton ±15° around Y-axis (vertical) | EfficientGCN, CTR-GCN, InfoGCN |
| **Random Shear** | Apply random shear transformation (±0.3) | EfficientGCN, Shift-GCN |
| **Gaussian Noise** | Add N(0, 0.01) noise to joint coords | SGN, InfoGCN |
| **Joint Masking** | Zero out random subset of joints | CTR-GCN |
| **Random Scale** | Scale skeleton by factor [0.9, 1.1] | EfficientGCN |

### Temporal Augmentations
| Technique | Description | Used By |
|-----------|-------------|---------|
| **Random Crop** | Sample contiguous subsequence, resize to T | Most models |
| **Temporal Reverse** | Flip frame order with p=0.5 | EfficientGCN |
| **Temporal Shift** | Randomly shift start frame | CTR-GCN |
| **Gaussian Blur** | Smooth temporal trajectory per joint | EfficientGCN |

---

## 3. LAST Training Strategy

Based on the competitor analysis and our lightweight architecture constraints:

### 3.1 Optimizer: SGD with Nesterov Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4
)
```

**Rationale:** Every top-performing skeleton model uses SGD+Nesterov. Adam converges faster initially but generalizes worse for this domain.

### 3.2 Learning Rate Schedule: Cosine Annealing with Warm Restart

```python
# Warmup: linear ramp from 0.001 to 0.1 over 5 epochs
# Cosine: decay from 0.1 to 1e-5 over remaining epochs
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=65, T_mult=1)
```

**Rationale:** EfficientGCN uses SGDR warmup and achieves top results. Cosine is smoother than step decay and avoids the need to hand-tune milestone epochs. We use 5-epoch warmup to stabilize early GCN training.

| Phase | Epochs | LR Range |
|-------|--------|----------|
| Warmup | 1–5 | 0.001 → 0.1 |
| Cosine | 6–70 | 0.1 → 1e-5 |

### 3.3 Loss Function: Cross-Entropy + Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Rationale:** InfoGCN showed label smoothing (0.1) prevents overconfidence and improves generalization. Especially important for our lightweight model which is more prone to overfitting given fewer parameters.

### 3.4 Data Augmentation Pipeline

```python
train_transforms = Compose([
    RandomRotation(degrees=15, axis='y'),     # ±15° around vertical
    RandomShear(magnitude=0.3),                # Random shear
    GaussianNoise(std=0.01),                   # Joint noise
    TemporalCrop(target_frames=64),            # Random temporal crop
    # p=0.5 probability transforms:
    RandomTemporalReverse(p=0.5),              # Flip time
])
```

### 3.5 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 70 | EfficientGCN range, sufficient for <1M param model |
| Batch Size | 64 | Standard for NTU; fits single GPU |
| Gradient Clipping | max_norm=5.0 | Prevents GCN gradient explosion |
| Input Frames | 64 | Temporal crop during training |
| Weight Decay | 1e-4 | Standard L2 regularization |
| FC Dropout | 0.5 | Before classifier head |
| Block Dropout | 0.1 | Within LAST blocks |

### 3.6 Evaluation

- **Metric:** Top-1 accuracy (standard for NTU RGB+D)
- **Validation:** Evaluate every epoch, save best model by val accuracy
- **Test:** Use best checkpoint for final evaluation

---

## 4. Performance Targets

| Dataset | Split | LAST-Base Target | SOTA Reference |
|---------|-------|-----------------|----------------|
| NTU 60 | X-Sub | **88.0%+** | EfficientGCN: 91.7%, Shift-GCN: 90.7% |
| NTU 60 | X-View | **93.0%+** | EfficientGCN: 96.1%, Shift-GCN: 96.5% |

> [!NOTE]
> LAST-Base has <1M params vs EfficientGCN's 2.0M+. Our target prioritizes the efficiency-accuracy tradeoff — competitive accuracy with significantly fewer parameters and FLOPs.

---

## 5. Training Script Requirements

The training script (`scripts/train.py`) must support:

1. **Config-driven** — All hyperparameters from YAML files
2. **Resume training** — Load from checkpoint, restore optimizer/scheduler state
3. **Logging** — TensorBoard or CSV-based metrics (loss, accuracy, lr)
4. **Checkpointing** — Save best model + periodic checkpoints
5. **Mixed precision** — Optional AMP for faster training
6. **Reproducibility** — Seed setting for deterministic results
7. **Multi-variant** — `--model base|small|large`, `--dataset ntu60|ntu120`
