# Experiment: Training Pipeline

Complete training pipeline for LAST-Lite and LAST-Base: standalone training, distillation,
pretraining, hyperparameters, scheduling, and ablation experiments.

---

## 1. Full Pipeline Overview

> **Note:** Our existing trained model (LAST-E v3, 720K params) serves as a mid-size teacher
> for distillation if LAST-Base is not yet ready. It is NOT part of the new experiments.

```
Phase 0: Preprocessing
  ├── Re-preprocess NTU-60/120 with T_target=64 (uniform subsample)
  └── Generate 4 streams: joint, bone, velocity, bone_velocity

Phase 1: Train LAST-Base (teacher — new architecture)
  ├── Train 4 streams independently
  └── Ensemble → best checkpoint (LAST-Base-teacher)

Phase 2: Distillation (LAST-Base → LAST-Lite)
  ├── Teacher: frozen LAST-Base best checkpoint
  ├── (Fallback teacher: our existing trained model if Base not ready)
  ├── Student: LAST-Lite small / nano
  └── Loss: CE + KD + feature mimicry

Phase 3: MaskCLR Pretraining (conditional — only if Phase 2 < 88%)
  ├── Self-supervised pretrain LAST-Lite encoder
  └── Re-run Phase 2 with pretrained init

Phase 4: Quantization + Export
  ├── INT8 PTQ with calibration
  └── ONNX / TFLite export
```

---

## 2. Phase 1: LAST-Base Training

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Transformers (PartitionedAttention, HBRA) converge better with Adam |
| **LR** | 1e-3 | AdamW standard — lower than SGD because adaptive |
| **Weight decay** | 0.05 | AdamW standard — acts as L2 on decoupled weights |
| **β1, β2** | 0.9, 0.999 | Default Adam |
| **Scheduler** | cosine_warmup | Plain cosine, no restarts |
| **Warmup epochs** | 10 | Longer warmup for transformer attention to stabilize |
| **Warmup start LR** | 1e-5 | lr/100 |
| **Min LR** | 1e-6 | Lower floor for Adam (vs 1e-4 for SGD) |
| **Epochs** | 200 | Transformers need longer training than pure GCN |
| **Batch size** | 32 | GPU memory — 4M params × 4 bytes × batch features |
| **Gradient accumulation** | 4 | Effective batch = 128 |
| **Gradient clip** | 1.0 | Prevent attention gradient explosions |
| **DropPath** | [0.0, 0.1, 0.2] | Linearly increasing across stages |
| **Dropout (head)** | 0.3 | Before final FC |
| **Label smoothing** | 0.1 | Standard |
| **IB loss weight** | 0.01 | From InfoGCN |
| **AMP** | True | FP16 mixed precision |
| **SWA** | Start at epoch 150 | Average weights in final 25% |

### Training Commands (4 streams)

```bash
# Stream 1: Joint
python scripts/train.py \
  --model last_base --dataset ntu60 --split_type xsub \
  --epochs 200 --batch_size 32 --lr 0.001 --optimizer adamw \
  --scheduler cosine_warmup --min_lr 0.000001 \
  --amp --workers 4 --seed 42 \
  --set training.weight_decay=0.05 \
       training.warmup_epochs=10 \
       training.warmup_start_lr=0.00001 \
       training.gradient_clip=1.0 \
       training.gradient_accumulation_steps=4 \
       training.label_smoothing=0.1 \
       training.ib_loss_weight=0.01 \
       model.drop_path_rate=0.2 \
       data.dataset.stream=joint

# Stream 2: Bone
# Same as above but: data.dataset.stream=bone

# Stream 3: Joint velocity
# Same as above but: data.dataset.stream=joint_velocity

# Stream 4: Bone velocity
# Same as above but: data.dataset.stream=bone_velocity
```

### Training Duration Estimate

| Hardware | Single stream | 4 streams total |
|----------|--------------|-----------------|
| A10 (24GB) | ~15h | ~60h |
| A100 (40GB) | ~8h | ~32h |
| T4 (16GB) | ~30h | ~120h (5 days) |

---

## 3. Phase 2: Knowledge Distillation

### 3A. LAST-Base → LAST-Lite (Primary)

**Teacher:** LAST-Base 4-stream ensemble (frozen, eval mode, no_grad)
**Student:** LAST-Lite small (ShiftFuse-GCN, ~120-190K params)

### Loss Function

```
L_total = α × L_CE + β × L_KD + γ × L_feat

Where:
  L_CE   = CrossEntropy(student_logits, hard_labels)
  L_KD   = τ² × KL(softmax(student/τ), softmax(teacher/τ))
  L_feat = Σ_s MSE(proj_s(student_feat[s]), teacher_feat[s])   per stage
```

### Distillation Hyperparameters

| Parameter | LAST-Lite nano | LAST-Lite small | Rationale |
|-----------|---------------|-----------------|-----------|
| α (CE weight) | 0.3 | 0.5 | Smaller student → more reliance on teacher |
| β (KD weight) | 0.7 | 0.5 | Complementary to α |
| γ (feature weight) | 0.1 | 0.1 | Feature mimicry regularization |
| τ (temperature) | 4.0 | 4.0 | Standard for skeleton (softer than vision's τ=2) |
| Optimizer | SGD + Nesterov | SGD + Nesterov | Student is pure conv |
| LR | 0.1 | 0.1 | |
| Scheduler | cosine_warmup | cosine_warmup | |
| Warmup | 5 epochs | 5 epochs | |
| Epochs | 120 | 120 | |
| Batch size | 64 | 64 | Small student → large batch fits easily |
| Label smoothing | 0.0 | 0.0 | Teacher provides soft targets — no smoothing needed |
| DropPath | 0.0 | 0.0 | Student has no adaptive modules |
| Dropout (head) | 0.1 | 0.15 | |
| Gradient clip | 1.0 | 1.0 | |

### Feature Mimicry Details

Feature mimicry requires matching spatial dimensions. Since teacher (LAST-Base) and student
(LAST-Lite) have different channel widths, add 1×1 projection convs:

```python
# At each stage boundary:
proj_s = nn.Conv2d(student_C[s], teacher_C[s], kernel_size=1)
L_feat_s = MSE(proj_s(student_output[s]), teacher_output[s].detach())
```

| Stage | Student C (Lite small) | Teacher C (Base) | Proj params |
|-------|----------------------|------------------|-------------|
| 1 | 48 | 128 | 48 × 128 = 6,144 |
| 2 | 72 | 256 | 72 × 256 = 18,432 |
| 3 | 96 | 384 | 96 × 384 = 36,864 |
| **Total proj** | | | **~61K** (discarded after training) |

### Distillation Command

```bash
python scripts/train_distill.py \
  --student shiftfuse_small --teacher last_base \
  --teacher_ckpt runs/last_base_joint/best.pth \
  --dataset ntu60 --split_type xsub \
  --epochs 120 --batch_size 64 --lr 0.1 \
  --scheduler cosine_warmup \
  --set distill.alpha=0.5 \
       distill.tau=4.0 \
       distill.gamma=0.1 \
       training.warmup_epochs=5 \
       training.label_smoothing=0.0
```

### 3B. Existing Model → LAST-Lite (Fallback, if Base not ready)

Same loss function but:
- Teacher: our existing trained model (720K, MIB mode)
- Student: LAST-Lite small
- α = 0.5, β = 0.5 (smaller teacher → less reliance on KD)

---

## 4. Phase 3: MaskCLR Self-Supervised Pretraining

### When to Activate

**Decision rule:** If Phase 2 distillation gives LAST-Lite small < 88% (below EfficientGCN-B0),
add MaskCLR pretraining before distillation.

### Pretraining Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Data | NTU-60 train split (NO labels used) | Self-supervised |
| Encoder | LAST-Lite small backbone (no classification head) | Same arch as student |
| Mask ratio | 0.6 | Mask 60% of joints — force reconstruction |
| Mask strategy | Body-region masking (mask entire arm/leg) | Graph-aware |
| Contrastive temp | 0.07 | InfoNCE standard |
| Decoder | Linear(C_last, 3 × V) | Reconstruct masked joints |
| Projector | Linear(C_last, 128) | Contrastive projection head |
| Optimizer | AdamW | Standard for SSL |
| LR | 1e-3 | |
| Epochs | 100 | |
| Batch size | 128 | Larger batch helps contrastive |

### Loss

```
L_pretrain = L_recon + 0.5 × L_contrastive

L_recon       = MSE(decoder(encoder(masked_x)), original_x)    at masked positions only
L_contrastive = InfoNCE(projector(encoder(view1)), projector(encoder(view2)))
```

### Augmentations for Contrastive Views

```python
view1 = RandomRotation(15) + RandomScale(0.9, 1.1) + RegionMask(0.6)
view2 = RandomRotation(15) + RandomScale(0.9, 1.1) + RegionMask(0.6)  # different mask
```

### After Pretraining

1. Discard decoder and projector heads
2. Initialize student with pretrained encoder weights
3. Add classification head
4. Run Phase 2 (distillation) from pretrained init

### Pretraining Command

```bash
python scripts/pretrain_maskclr.py \
  --model shiftfuse_small --dataset ntu60 \
  --epochs 100 --batch_size 128 --lr 0.001 \
  --optimizer adamw --scheduler cosine_warmup \
  --set pretrain.mask_ratio=0.6 \
       pretrain.temperature=0.07 \
       pretrain.mask_strategy=body_region
```

---

## 5. Phase 4: Quantization + Export

### INT8 Post-Training Quantization

```python
# 1. Collect calibration data (1000 samples from train set)
calibration_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
calibration_data = [batch for batch, _ in itertools.islice(calibration_loader, 32)]

# 2. Quantize
model.eval()
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# 3. Verify accuracy drop < 1%
val_acc_fp32 = evaluate(model, val_loader)
val_acc_int8 = evaluate(quantized, val_loader)
assert val_acc_fp32 - val_acc_int8 < 1.0, f"Quantization dropped {val_acc_fp32 - val_acc_int8}%"
```

### ONNX Export

```bash
python scripts/export_onnx.py \
  --model shiftfuse_small \
  --checkpoint runs/distill_lite_small/best.pth \
  --output exports/last_lite_small.onnx \
  --opset 13 --dynamic_batch
```

---

## 7. SWA (Stochastic Weight Averaging)

Apply to ALL training phases for free +0.5-1.5% accuracy:

```python
from torch.optim.swa_utils import AveragedModel, update_bn

swa_model = AveragedModel(model)
swa_start = int(0.75 * total_epochs)

# In training loop:
for epoch in range(total_epochs):
    train_one_epoch(model, ...)
    if epoch >= swa_start:
        swa_model.update_parameters(model)

# After training:
update_bn(train_loader, swa_model)
torch.save(swa_model.module.state_dict(), 'best_swa.pth')
```

---

## 8. Causal Training (Idea E)

Apply to Phase 1 and Phase 2 for free +0.3-0.8%:

```python
# In training loop, 50% of batches use causal temporal masking:
if random.random() < 0.5:
    # Replace symmetric padding with left-only padding in all TCN layers
    for module in model.modules():
        if isinstance(module, nn.Conv1d) and module.padding[0] > 0:
            # Apply causal mask: zero out future positions
            # This forces the model to learn predictive representations
            module._causal_mode = True
```

**When:** Apply from epoch 1. No warmup needed — it's a form of augmentation.
**Cost:** Zero additional params. ~5% training throughput loss (masking overhead).

---

## 9. Complete Ablation Study Plan

### 9A. LAST-Base Ablations

| # | Experiment | What changes | Purpose | GPU hours |
|---|-----------|-------------|---------|-----------|
| B0 | Baseline | Standard GCN (SpatialGCN + TCN + residual, no novel modules) | Pure conv baseline | 8h |
| B1 | + CrossTemporalGCN | Add temporal context to adjacency | Measure HI-GCN contribution | 10h |
| B2 | + ActionPrototypeGraph | Add K=15 class-conditioned graphs (Idea B) | Measure Idea B gain | 10h |
| B3 | + FreqTemporalGate | Add frequency-domain gating (Idea A) | Measure frequency contribution | 10h |
| B4 | + PartitionedAttention | Add SkateFormer 4-type attention | Measure partition attention gain | 12h |
| B5 | + HBRA | Add hierarchical body-region (Idea D) | Measure body region gain | 12h |
| B6 | + IB loss | Add information bottleneck loss | Measure IB contribution | 10h |
| B7 | Full LAST-Base | All components | Full model | 15h |
| B8 | − FreqGate | Full minus Idea A | Ablate frequency | 15h |
| B9 | − PrototypeGraph | Full minus Idea B | Ablate prototypes | 15h |
| B10 | − HBRA | Full minus Idea D | Ablate body regions | 15h |
| B11 | − PartitionedAttn (use factorized S+T) | Replace SkateFormer with ST_JointAtt | Partition vs factorized | 15h |
| B12 | + CausalTraining | Full + Idea E | Measure causal training | 15h |
| B13 | + SWA | Full + weight averaging | Measure SWA gain | 15h |
| B14 | 4-stream ensemble | 4 × B7 | Final ensemble accuracy | 60h |

**Total GPU budget for Base ablations:** ~217h ≈ 9 A100-days

### 9B. LAST-Lite (ShiftFuse-GCN) Ablations

| # | Experiment | What changes | Purpose | GPU hours |
|---|-----------|-------------|---------|-----------|
| L0 | Baseline | Random shift + EpSepTCN only | Shift-GCN + EfficientGCN baseline | 3h |
| L1 | + JointEmbed | Add joint semantic embedding (SGN) | Measure embedding gain | 3h |
| L2 | + FrameGate | Add frame dynamics gate (SGN) | Measure temporal awareness | 3h |
| L3 | + BodyRegionShift | Replace random shift with BRASP (Idea F) | Measure region shift gain | 3h |
| L4 | + FrozenDCTGate | Add frozen DCT frequency routing (Idea G) | Measure frequency gain | 3h |
| L5 | Full ShiftFuse | All components | Full Lite model | 3h |
| L6 | − DCT Gate | Full minus Idea G | Ablate frequency routing | 3h |
| L7 | − JointEmbed | Full minus embeddings | Ablate identity encoding | 3h |
| L8 | − BodyRegionShift (use random) | Full but random shift | Ablate body-aware shift | 3h |
| L9 | + CausalTraining | Full + Idea E | Measure causal training (Lite) | 3h |
| L10 | + Distillation (from existing model) | Full + KD from existing trained model | Distillation from mid-size teacher | 4h |
| L11 | + Distillation (from Base) | Full + KD from LAST-Base | Distillation from best teacher | 4h |
| L12 | + MaskCLR → Distill | Pretrained init + KD | Full pipeline | 8h |

**Total GPU budget for Lite ablations:** ~46h ≈ 2 A100-days

### 9C. Preprocessing Ablations

| # | Experiment | What changes | Purpose | GPU hours |
|---|-----------|-------------|---------|-----------|
| P0 | T=64 uniform subsample (new) | Recommended preprocessing | Baseline comparison | 6h |
| P1 | T=300 store + crop 64 (current) | Current pipeline | Quantify preprocessing impact | 6h |
| P2 | T=120 uniform subsample | Higher temporal resolution | Measure T resolution gain | 8h |
| P3 | T=64 + M=2 (both bodies) | Include second body | Multi-person action impact | 6h |

**Total GPU budget for preprocessing:** ~26h ≈ 1 A100-day

---

## 10. Results Table Template

### LAST-Base Results

| Model | NTU-60 xsub | NTU-60 xview | NTU-120 xsub | NTU-120 xset | Params |
|-------|-------------|-------------|-------------|-------------|--------|
| EfficientGCN-B0 | 88.3 | — | — | — | 150K |
| CTR-GCN (4s) | 92.4 | 96.8 | 88.9 | 90.6 | 1.7M |
| InfoGCN (6s) | 93.0 | 97.1 | 89.8 | 91.2 | 1.5M |
| HD-GCN (6s) | 93.4 | — | 90.1 | — | ~3M |
| HI-GCN | 93.3 | — | 90.3 | — | ~3-4M |
| **LAST-Base (1s)** | — | — | — | — | ~4.2M |
| **LAST-Base (4s)** | — | — | — | — | ~16.8M |

### LAST-Lite Results

| Model | NTU-60 xsub | Params | FLOPs | INT8 size |
|-------|-------------|--------|-------|-----------|
| EfficientGCN-B0 | 88.3 | 150K | — | — |
| LAST-Lite nano (standalone) | — | ~60K | — | ~15KB |
| LAST-Lite small (standalone) | — | ~190K | — | ~48KB |
| LAST-Lite small + distill (existing model) | — | ~190K | — | ~48KB |
| LAST-Lite small + distill (Base) | — | ~190K | — | ~48KB |
| LAST-Lite small + MaskCLR + distill | — | ~190K | — | ~48KB |

---

## 11. Experiment Execution Order

```
Week 1: Preprocessing
  ├── Day 1: Re-preprocess NTU-60 with T=64
  ├── Day 2: Run quality checks, verify shapes
  └── Day 3: P0 vs P1 comparison (current vs new preprocessing)

Week 2-3: Retrain existing model with new preprocessing
  ├── Train with T=64 uniform subsample → confirm improvement over T=300+crop
  └── This becomes the fallback teacher for Lite distillation (if Base not ready)

Week 3-4: LAST-Lite Ablations (L0-L5)
  ├── Day 1: L0 baseline
  ├── Day 2: L1, L2 (add embeddings, frame gate)
  ├── Day 3: L3, L4 (add body-region shift, DCT gate)
  └── Day 4: L5 full model + L6-L9 removal ablations

Week 4-5: LAST-Lite Distillation (L10-L12)
  ├── L10: Distill from existing trained model
  ├── L11: Distill from LAST-Base (if ready)
  └── L12: MaskCLR + distill (if L10/L11 < 88%)

Week 5-8: LAST-Base Implementation + Training
  ├── Week 5: Implement CrossTemporalPrototypeGCN
  ├── Week 6: Implement FreqTemporalGate + PartitionedAttention
  ├── Week 7: Implement HBRA + integration tests
  └── Week 8: Train B0-B7 ablations

Week 9-10: LAST-Base Full Training + Ensemble
  ├── Train 4 streams × B7 full model
  ├── Ensemble evaluation
  └── Final distillation: Base → Lite

Week 11: Paper Writing
  ├── Run final missing ablations
  └── Generate all tables and figures
```
