# 08 — Environment Setup

## Requirements

- Python 3.11
- CUDA 12.x
- PyTorch 2.x

```bash
pip install -r requirements.txt
```

---

## A. Local Setup (Windows)

**Virtual environment:**
```bash
# Activate (Git Bash / WSL)
source /c/Users/pathi/envs/ai_research/Scripts/activate
```

**Data root:** configured in `configs/environment/local.yaml`

**Smoke test (no GPU required):**
```bash
python scripts/train.py --model nano_e --dataset ntu60 --epochs 2 --batch_size 4
```

---

## B. Kaggle Setup (T4 16GB, 9h session limit)

### Dataset

Upload preprocessed data as a Kaggle dataset:
- Dataset name: `pathikreet/last-research-preprocessed-v2`
- Mount path: `/kaggle/input/last-research-preprocessed-v2/`

### Symlink

Kaggle notebooks mount datasets at read-only paths. Create a writable symlink:
```bash
mkdir -p /tmp/LAST-60-v2/data/processed_v2
ln -sf /kaggle/input/last-research-preprocessed-v2/xsub \
       /tmp/LAST-60-v2/data/processed_v2/xsub
```
(Adjust for xview/xset splits as needed.)

### Config

`configs/environment/kaggle.yaml` is auto-detected when `/kaggle` exists in the filesystem.
Override manually with `--env kaggle`.

### Training Command

```bash
python scripts/train.py --model base_e --dataset ntu60 --env kaggle --amp
```

### Notes
- Workers: 2 (Kaggle limit; configured in kaggle.yaml)
- AMP is required — 16GB is tight for base_e at batch=16 without it
- Kaggle sessions are 9h; 70 epochs of LAST-E-base takes ~6–7h on T4
- Save checkpoints to `/kaggle/working/` — this directory persists after session end

---

## C. Lambda AI Setup (A10 23GB) — Primary Training Environment

### Instance Specs

| Field       | Value                               |
|-------------|-------------------------------------|
| GPU         | NVIDIA A10 (23GB VRAM)              |
| CPU         | Intel Xeon Platinum 8358, 30 cores  |
| RAM         | 222 GB                              |
| Storage     | 1.4 TB local + NFS persistent share |

### Data Mount

Code and preprocessed data are mounted via NFS:
```
~/research-last/LAST/                                       → project root
~/research-last/data/LAST-60-v2/data/processed_v2/xsub/   → MIB data
```

Both paths resolve to `/lambda/nfs/research-last/...` which **persists across instance restarts**.

### Config

`configs/environment/lambda.yaml` — set explicitly via `--env lambda`:
- `data_base: /lambda/nfs/research-last/data` (train.py appends `LAST-60-v2/data/processed_v2`)
- `output_root: /lambda/nfs/research-last/LAST-runs` (persistent NFS)
- `num_workers: 12`, `prefetch_factor: 4`, `pin_memory: true`

### Minimal pip install

Lambda instances ship with PyTorch + CUDA pre-installed. Only install:
```bash
pip install tqdm pyyaml
```

Verify before training:
```bash
python -c "import torch, yaml, tqdm; print(torch.cuda.get_device_name(0))"
```

### Training with tmux (SSH-independent)

Always run inside tmux so training survives SSH disconnects:
```bash
# Start session
tmux new -s train

cd ~/research-last/LAST
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py --model base --dataset ntu60 --env lambda --amp --batch_size 32

# Detach (leave running): Ctrl+B  then  D
# Reattach later:         tmux attach -t train
```

### Training Commands

```bash
# LAST-v2 base (teacher) — primary target
python scripts/train.py --model base --dataset ntu60 --env lambda --amp --batch_size 32

# LAST-E base (student) — efficient model
python scripts/train.py --model base_e --dataset ntu60 --env lambda --amp --batch_size 128

# Resume from checkpoint
python scripts/train.py --model base --dataset ntu60 --env lambda --amp --batch_size 32 \
  --resume /lambda/nfs/research-last/LAST-runs/run-YYYY-MM-DD/checkpoints/best_model.pth
```

### Batch Size Reference (A10 23GB with AMP)

| Model         | Batch size | Note                                       |
|---------------|------------|--------------------------------------------|
| LAST-v2 base  | 32         | 3 streams × 9.2M params; 64 causes OOM    |
| LAST-E (any)  | 128        | 92K–644K params; A10 has ample headroom    |

### Notes
- Outputs to NFS: checkpoints and metrics survive instance stop/restart
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents A10 memory fragmentation
- Workers (12) and prefetch_factor (4) auto-applied from `lambda.yaml` — no `--workers` flag needed
- Monitor GPU: `watch -n 1 nvidia-smi` (target >80% utilisation during training)

---

## D. GCP Setup (P100 16GB)

### One-Time Instance Setup

```bash
bash scripts/gcp_setup.sh
```

This script installs CUDA, PyTorch, and project dependencies on the instance.

### Training

```bash
bash scripts/gcp_train_base.sh
```

Runs LAST-v2-base on NTU60 xsub with GCP P100 config.

### Config

`configs/environment/gcp.yaml` — set via `--env gcp`.

### Recommended Instance

| Field        | Value                              |
|--------------|------------------------------------|
| Machine type | `n1-standard-8`                    |
| Accelerator  | NVIDIA Tesla P100 (1× 16GB)        |
| Zone         | `us-central1-c`                    |
| Boot disk    | 100GB SSD (50GB+ for NTU120)       |
| Preemptible  | Yes — use checkpoint resume        |

### Shutdown After Training

```bash
# From local machine (replace INSTANCE_NAME and ZONE)
gcloud compute instances stop INSTANCE_NAME --zone=ZONE
```

Always stop the instance manually — auto-shutdown is not configured by default.

---

## E. Config Auto-Detection Logic

`src/utils/config.py` determines the environment at startup:

1. If `--env` flag is provided → use that environment
2. Else if `/kaggle` exists in filesystem → use `kaggle`
3. Else → use `local`

Lambda requires `--env lambda` explicitly (no auto-detection trigger).

This means no code changes are needed when moving between environments — just pass the correct
`--env` flag and the right paths/workers/etc. are loaded automatically.

---

## F. Verifying Setup

```bash
# Check model loads and param counts match expected
python tests/test_model_integration.py

# Check data pipeline loads correctly
python -c "
from src.data.dataset import SkeletonDataset
ds = SkeletonDataset(split='xsub', phase='train')
print('Dataset length:', len(ds))
x, y = ds[0]
print('Sample keys:', list(x.keys()))
print('Joint shape:', x['joint'].shape)
"
```
