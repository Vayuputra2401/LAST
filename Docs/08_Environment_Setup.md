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
python scripts/train.py --model shiftfuse_nano --dataset ntu60 --epochs 2 --batch_size 4
```

---

## B. Kaggle Setup (T4 16GB, 9h session limit)

### Dataset

Upload preprocessed data as a Kaggle dataset:
- Dataset name: `pathikreet/last-research-preprocessed-v2`
- Mount path: `/kaggle/input/last-research-preprocessed-v2/`

### Config

`configs/environment/kaggle.yaml` is auto-detected when `/kaggle` exists in the filesystem.
Override manually with `--env kaggle`.

### Training Commands

```bash
# LAST-Lite small (uses YAML defaults)
python scripts/train.py --model shiftfuse_small --dataset ntu60 --env kaggle --amp --avg_checkpoints 5

# LAST-Lite nano (with CLI overrides for lighter regularisation)
python scripts/train.py --model shiftfuse_nano --dataset ntu60 --env kaggle --amp --avg_checkpoints 5 \
    --weight_decay 0.0003 --set training.label_smoothing=0.03
```

### Notes
- Workers: 2 (Kaggle limit; configured in kaggle.yaml)
- AMP is recommended for faster training
- Kaggle sessions are 9h; 120 epochs of LAST-Lite takes ~4-5h on T4
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
~/research-last/LAST/                                       -> project root
~/research-last/data/LAST-60/data/processed/xsub/          -> 4-stream data
```

Both paths resolve to `/lambda/nfs/research-last/...` which **persists across instance restarts**.

### Config

`configs/environment/lambda.yaml` — set explicitly via `--env lambda`:
- `data_base: /lambda/nfs/research-last/data`
- `output_root: /lambda/nfs/research-last/LAST-runs` (persistent NFS)
- `num_workers: 12`, `prefetch_factor: 4`, `pin_memory: true`

### Minimal pip install

Lambda instances ship with PyTorch + CUDA pre-installed. Only install:
```bash
pip install tqdm pyyaml
```

### Training with tmux (SSH-independent)

Always run inside tmux so training survives SSH disconnects:
```bash
tmux new -s train
cd ~/research-last/LAST
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# LAST-Lite small
python scripts/train.py --model shiftfuse_small --dataset ntu60 --env lambda --amp --avg_checkpoints 5

# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
```

### Batch Size Reference (A10 23GB with AMP)

| Model              | Batch size | Note                                    |
|--------------------|------------|-----------------------------------------|
| LAST-Lite (any)    | 128        | 80K-248K params; A10 has ample headroom |
| LAST-Base (future) | 32         | ~4.2M params per stream                 |

### Notes
- Outputs to NFS: checkpoints and metrics survive instance stop/restart
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents A10 memory fragmentation
- Workers (12) and prefetch_factor (4) auto-applied from `lambda.yaml`
- Monitor GPU: `watch -n 1 nvidia-smi` (target >80% utilisation during training)

---

## D. Config Auto-Detection Logic

`src/utils/config.py` determines the environment at startup:

1. If `--env` flag is provided -> use that environment
2. Else if `/kaggle` exists in filesystem -> use `kaggle`
3. Else -> use `local`

Lambda requires `--env lambda` explicitly (no auto-detection trigger).

This means no code changes are needed when moving between environments — just pass the correct
`--env` flag and the right paths/workers/etc. are loaded automatically.

---

## E. Verifying Setup

```bash
# Check LAST-Lite model loads and param counts match expected
python scripts/load_model.py --model shiftfuse_small --dataset ntu60

# Run ShiftFuse integration tests
python -m pytest tests/test_shiftfuse.py -v
```
