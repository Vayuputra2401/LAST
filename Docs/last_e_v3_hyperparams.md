# LAST-E v3 Hyperparameters & Training Reference

## Kaggle T4 Training Command

```bash
python scripts/train.py \
  --model base_e_v3 \
  --dataset ntu60 \
  --split_type xsub \
  --epochs 120 \
  --batch_size 32 \
  --lr 0.1 \
  --weight_decay 0.0004 \
  --dropout 0.3 \
  --scheduler cosine_warmup \
  --min_lr 0.0001 \
  --amp \
  --workers 2 \
  --seed 42 \
  --env kaggle \
  --set training.gradient_clip=1.0 \
       training.gradient_accumulation_steps=1 \
       training.warmup_epochs=5 \
       training.warmup_start_lr=0.01 \
       training.label_smoothing=0.1 \
       training.ib_loss_weight=0.01 \
       training.save_interval=10 \
       training.nesterov=true \
       training.momentum=0.9
```

### One-liner (copy-paste for Kaggle notebook cell)

```bash
!python scripts/train.py --model base_e_v3 --dataset ntu60 --split_type xsub --epochs 120 --batch_size 32 --lr 0.1 --weight_decay 0.0004 --dropout 0.3 --scheduler cosine_warmup --min_lr 0.0001 --amp --workers 2 --seed 42 --env kaggle --set training.gradient_clip=1.0 training.gradient_accumulation_steps=1 training.warmup_epochs=5 training.warmup_start_lr=0.01 training.label_smoothing=0.1 training.ib_loss_weight=0.01 training.save_interval=10 training.nesterov=true training.momentum=0.9
```

### Other Variants

```bash
# Nano (~75K params, fastest)
!python scripts/train.py --model nano_e_v3 --dataset ntu60 --epochs 120 --batch_size 64 --lr 0.1 --amp --workers 2 --env kaggle --set training.gradient_clip=1.0 training.warmup_epochs=5 training.warmup_start_lr=0.01

# Small (~284K params)
!python scripts/train.py --model small_e_v3 --dataset ntu60 --epochs 120 --batch_size 48 --lr 0.1 --amp --workers 2 --env kaggle --set training.gradient_clip=1.0 training.warmup_epochs=5 training.warmup_start_lr=0.01

# Large (~911K params, best accuracy)
!python scripts/train.py --model large_e_v3 --dataset ntu60 --epochs 120 --batch_size 24 --lr 0.1 --dropout 0.3 --amp --workers 2 --env kaggle --set training.gradient_clip=1.0 training.warmup_epochs=5 training.warmup_start_lr=0.01 training.ib_loss_weight=0.01
```

---

## Default Hyperparameters

Source: `configs/training/cosine_v3.yaml` + model variant configs.

### Optimizer

| Parameter | Value | Notes |
|-----------|-------|-------|
| optimizer | `sgd` | Stochastic Gradient Descent |
| lr | `0.1` | Peak learning rate |
| momentum | `0.9` | SGD momentum |
| nesterov | `true` | Nesterov accelerated gradient |
| weight_decay | `0.0004` | L2 regularization; `edge`, `stream_weights`, `alpha`, `bn`, `bias` excluded |

### Scheduler

| Parameter | Value | Notes |
|-----------|-------|-------|
| scheduler | `cosine_warmup` | Cosine annealing with linear warmup |
| warmup_epochs | `5` | Linear warmup duration |
| warmup_start_lr | `0.01` | LR at epoch 0 |
| min_lr | `0.0001` | Cosine floor (lowest LR at end) |

### Training Duration & Batching

| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | `120` | Total training epochs |
| batch_size | `32` | Per-GPU batch size |
| gradient_accumulation_steps | `1` | Effective batch = batch_size × this |
| input_frames | `64` | Temporal frames per sample |

### Regularization

| Parameter | Value | Notes |
|-----------|-------|-------|
| label_smoothing | `0.1` | Cross-entropy label smoothing |
| gradient_clip | `1.0` | Max gradient norm (clip above) |
| dropout | Variant-dependent | See table below |
| drop_path_rate | Variant-dependent | Stochastic depth (DropPath) |
| ib_loss_weight | `0.01` | InfoGCN IB auxiliary loss weight (base/large only) |

### Precision & Hardware

| Parameter | Value | Notes |
|-----------|-------|-------|
| use_amp | `true` | Automatic Mixed Precision (fp16) |
| num_workers | `4` (local) / `2` (Kaggle) | DataLoader workers |
| pin_memory | `true` | Pinned CPU memory for faster GPU transfer |

### Checkpointing

| Parameter | Value | Notes |
|-----------|-------|-------|
| save_interval | `10` | Save checkpoint every N epochs |
| seed | `42` | Random seed for reproducibility |

---

## Model Variant Configs

| | Nano | Small | Base | Large |
|---|---|---|---|---|
| **Params** | 75,596 | 284,461 | 600,912 | 911,376 |
| **Budget** | < B0 (320K) | < B1 (420K) | < B3 (740K) | < B4 (940K) |
| stem_channels | 24 | 32 | 48 | 48 |
| channels | [32, 48, 64] | [48, 64, 96] | [64, 96, 128] | [80, 112, 160] |
| num_blocks | [1, 1, 1] | [1, 2, 2] | [2, 2, 2] | [2, 2, 2] |
| depths | [1, 1, 1] | [1, 1, 1] | [1, 1, 1] | [1, 1, 1] |
| strides | [1, 2, 2] | [1, 2, 2] | [1, 2, 2] | [1, 2, 2] |
| expand_ratio | 2 | 2 | 2 | 2 |
| max_hop | 1 | 2 | 2 | 2 |
| gate_type | motion | motion | motion | hybrid |
| use_subset_att | false | true | true | true |
| use_ib_loss | false | false | true | true |
| dropout | 0.2 | 0.25 | 0.3 | 0.3 |
| drop_path_rate | 0.0 | 0.0 | 0.05 | 0.1 |
| **GCN layers** | 3 | 5 | 6 | 6 |
| **Receptive field** | 3 hops | 10 hops | 12 hops | 12 hops |

---

## CLI Flags Reference

### Direct Override Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `base` | Model variant |
| `--dataset` | str | `ntu60` | Dataset name |
| `--split_type` | str | `xsub` | Train/val split |
| `--epochs` | int | from YAML | Number of epochs |
| `--batch_size` | int | from YAML | Batch size |
| `--lr` | float | from YAML | Peak learning rate |
| `--weight_decay` | float | from YAML | L2 weight decay |
| `--dropout` | float | from YAML | Classifier dropout |
| `--scheduler` | str | from YAML | LR scheduler type |
| `--min_lr` | float | from YAML | Cosine floor LR |
| `--milestones` | str | from YAML | MultiStep milestones (comma-sep) |
| `--amp` | flag | from YAML | Enable mixed precision |
| `--workers` | int | from YAML | DataLoader workers |
| `--seed` | int | from YAML | Random seed |
| `--resume` | str | None | Checkpoint path to resume |
| `--env` | str | auto | Environment: `local`, `kaggle`, `gcp`, `lambda`, `a100` |

### Generic Override (Highest Priority)

```
--set KEY=VALUE [KEY=VALUE ...]
```

Override **any** config key using dot-notation. Applied after all other flags.

**Examples:**
```bash
--set training.gradient_clip=2.0
--set training.gradient_accumulation_steps=4
--set training.warmup_epochs=3
--set training.warmup_start_lr=0.005
--set training.ib_loss_weight=0.05
--set training.label_smoothing=0.2
--set training.save_interval=5
--set training.nesterov=false
--set model.dropout=0.4
```

---

## Weight Decay Exclusion List

The following parameter patterns are excluded from weight decay (WD = 0.0):

| Pattern | Matches | Reason |
|---------|---------|--------|
| `bias` | All bias terms | Standard practice |
| `bn` | BatchNorm weight/bias | BN params shouldn't decay |
| `norm` | LayerNorm params | Same as BN |
| `alpha` | ST_JointAtt alpha gates | Zero-init, WD fights growth |
| `A_learned` | Adaptive graph matrices | Graph structure params |
| `node_proj` | Dynamic adj projection | Embedding projection |
| `refine_gate` | CTR topology gate (v2) | Gate scalar |
| `pool_gate` | Gated GAP+GMP blend | Gate scalar |
| `freq_gate` | FreqTemporalGate (v2) | Gate scalar |
| `edge` | SpatialGCN edge importance | Graph topology param |
| `stream_weights` | StreamFusion blend logits | Softmax pre-weights |

Everything else (conv weights, linear weights, fc_mu, fc_logvar) gets the full weight decay.
