# GCP Training Guide - LAST v2 Base

Complete step-by-step guide for training LAST v2 Base model on Google Cloud Platform with P100 GPU.

**Two Methods Available:**
- **Method A**: Using gcloud SDK on local machine (recommended for frequent use)
- **Method B**: Using browser-based SSH from GCP Console (easiest, no local setup)

---

## Prerequisites

- ✅ GCP instance launched: `research-last-v2-base-run1`
  - **Machine type**: n1-standard-8 (8 vCPUs, 30GB RAM)
  - **GPU**: NVIDIA Tesla P100 (16GB HBM2)
  - **Zone**: asia-east1-c
  - **Boot disk**: 100GB (Deep Learning image with CUDA 12.4)

- ✅ Dataset uploaded to GCS bucket:
  - **Path**: `gs://research-last/LAST-60-v2/data/processed_v2/xsub/`
  - **Files**: `train_joint.npy`, `train_velocity.npy`, `train_bone.npy`, `train_label.pkl`, `val_joint.npy`, `val_velocity.npy`, `val_bone.npy`, `val_label.pkl`

- ✅ gcloud SDK installed and authenticated on local machine (Method A only)

---

## Method Comparison

| Feature | Method A: Local gcloud SDK | Method B: Browser SSH |
|---------|---------------------------|----------------------|
| **Setup Required** | Install gcloud SDK, authenticate | None - just login to GCP Console |
| **File Upload** | Direct `gcloud compute scp` | Via GCS bucket (2-step) |
| **SSH Access** | Terminal/PowerShell on local machine | Browser tab |
| **Copy/Paste** | Native OS clipboard | Browser context menu |
| **Multi-window** | Easy (multiple terminals) | Multiple browser tabs |
| **Works Offline** | No (needs internet) | No (needs internet) |
| **File Download** | Direct `gcloud compute scp` | Via GCS bucket |
| **Best For** | Frequent GCP users, automation | One-time runs, quick access |

**Recommendation:** Use Method B (Browser SSH) if this is your first time or you don't have gcloud SDK installed.

---

## Training Workflow

# METHOD A: Using Local gcloud SDK

## Step 1A: Upload LAST Repository to GCP Instance

**Run this command on your local Windows machine (PowerShell or CMD):**

```bash
gcloud compute scp --recurse C:\Users\pathi\OneDrive\Desktop\LAST research-last-v2-base-run1:~/ --zone=asia-east1-c --project=research-454007
```

**What this does:**
- Uploads your entire LAST project folder directly to the instance's home directory (`~/LAST/`)
- Includes all code, configs, scripts, and the latest fixes (LR=0.05, warmup=10, clip=1.0)
- Takes ~2-5 minutes depending on network speed
- **One-step process** - files go directly from your machine to GCP instance

**Expected output:**
```
Uploading [C:\Users\pathi\OneDrive\Desktop\LAST] to [research-last-v2-base-run1:~/]
... files uploaded ...
```

---

## Step 2A: SSH into the GCP Instance

**Run this command on your local machine:**

```bash
gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
```

**What this does:**
- Opens a secure shell connection to your running instance
- Uses your local terminal (PowerShell/CMD/Terminal)
- You'll see a Linux terminal prompt like: `yourname@research-last-v2-base-run1:~$`

**Verify you're connected:**
```bash
pwd  # Should show: /home/yourname
ls   # Should show: LAST/
```

**Advantages:**
- ✅ Native terminal experience
- ✅ Works with Windows Terminal, PowerShell, CMD
- ✅ Easy copy/paste with Ctrl+C/V (or right-click)
- ✅ Can open multiple SSH sessions in separate windows

---

# METHOD B: Using Browser SSH (GCP Console)

## Step 1B: Upload LAST Repository to GCS Bucket

**Run this command on your local Windows machine (PowerShell or CMD):**

```bash
gsutil -m cp -r C:\Users\pathi\OneDrive\Desktop\LAST gs://research-last/code/LAST/
```

**What this does:**
- Uploads your LAST project folder to Google Cloud Storage bucket
- **Two-step process**: Local → GCS → Instance (necessary because browser SSH can't receive direct file transfers)
- Takes ~3-5 minutes depending on upload speed
- Files are stored temporarily in GCS bucket

**Expected output:**
```
Copying file://C:\Users\pathi\OneDrive\Desktop\LAST\... to gs://research-last/code/LAST/...
...
Operation completed over X objects/Y MB
```

---

## Step 2B: Open Browser SSH from GCP Console

**Steps:**
1. Open browser and go to: https://console.cloud.google.com/compute/instances?project=research-454007
2. Find your instance: `research-last-v2-base-run1`
3. Click the **SSH** button in the row (rightmost column)
4. A new browser tab/window opens with a terminal

**What this does:**
- Opens an SSH session directly in your browser
- No local software installation needed
- Works from any computer with a browser
- Connects using Google's IAP (Identity-Aware Proxy)

**You'll see:**
```
Linux research-last-v2-base-run1 5.10.0-... #1 SMP Debian ...
...
yourname@research-last-v2-base-run1:~$
```

**Browser SSH Tips:**
- **Copy**: Select text, then right-click → Copy (or browser's Edit menu)
- **Paste**: Right-click → Paste (Ctrl+V may not work in browser terminal)
- **Full screen**: Click the maximize icon in top-right
- **Multiple sessions**: Click SSH button again for a new tab

---

## Step 3B: Download LAST Repository from GCS to Instance

**Run this in the browser SSH terminal:**

```bash
gsutil -m cp -r gs://research-last/code/LAST ~/
```

**What this does:**
- Downloads your LAST project from GCS bucket to the instance
- Completes the two-step upload process (Local → GCS → Instance)
- Takes ~1-2 minutes (faster than upload, GCP internal network)
- Files are now on the instance at `~/LAST/`

**Verify download:**
```bash
ls ~/LAST/
# Should show: configs/ docs/ scripts/ src/ requirements.txt README.md etc.
```

---

## Step 4B: Fix Windows Line Endings (Important for Browser SSH Method)

**Why needed:**
- Files created on Windows use `\r\n` line endings (CRLF)
- Linux expects `\n` line endings (LF)
- Without fixing, bash scripts will fail with `$'\r': command not found`

**Run these commands:**

```bash
# Fix line endings in all scripts
sed -i 's/\r$//' ~/LAST/scripts/gcp_setup.sh
sed -i 's/\r$//' ~/LAST/scripts/gcp_train_base.sh

# Make scripts executable
chmod +x ~/LAST/scripts/gcp_setup.sh ~/LAST/scripts/gcp_train_base.sh
```

**What this does:**
- `sed -i 's/\r$//'` - Removes carriage return characters from end of each line
- `chmod +x` - Makes scripts executable
- Must be done before running any `.sh` files

**Note:** Method A (local gcloud SDK) may also need this if you created scripts on Windows.

---

## Common Steps for Both Methods

**⚠️ You are now on the GCP instance terminal. All remaining steps are identical for both methods.**

---

### Run One-Time Setup Script

**For Method A:** Already executable after upload
**For Method B:** Line endings already fixed in Step 4B

```bash
bash ~/LAST/scripts/gcp_setup.sh
```

**What this does (automatically):**

1. **System dependencies** - Installs git, vim, htop, tmux, screen
2. **GPU verification** - Runs `nvidia-smi` to confirm P100 is detected
3. **Repository setup** - Ensures LAST repo is present (already uploaded in Step 1)
4. **Python environment** - Creates conda environment named `last` with Python 3.10
5. **PyTorch installation** - Installs PyTorch 2.x with CUDA 12.4 support
6. **Dependencies** - Installs all packages from `requirements.txt` (numpy, pyyaml, tqdm, etc.)
7. **Dataset download** - Syncs preprocessed data from GCS bucket to `~/data/LAST-60-v2/data/processed_v2/xsub/`
8. **Verification** - Confirms PyTorch can detect the GPU

**Expected duration:** 10-15 minutes

**Expected final output:**
```
========================================
Setup Complete!
========================================
Next steps:
  1. Activate environment: conda activate last
  2. Start training: bash ~/LAST/scripts/gcp_train_base.sh
========================================
```

**Troubleshooting:**
- If conda command not found: `source ~/miniconda3/etc/profile.d/conda.sh`
- If gsutil fails: Run `gcloud auth login` and authenticate (Method A) or authenticate in browser (Method B)
- If GPU not detected: Check `nvidia-smi` shows P100
- If backports error appears: Ignore it (non-critical), setup will continue

---

### Start a Tmux Session

**Why tmux?**
- Training takes ~15-20 hours for 70 epochs
- Tmux keeps the process running even if your SSH connection drops (works for both Method A and B)
- You can close your terminal/browser tab and training continues
- You can detach and reattach anytime without interrupting training

```bash
tmux new -s training
```

**What this does:**
- Creates a persistent terminal session named "training"
- You'll see a green status bar at the bottom of the terminal (same in browser SSH and local terminal)

**Tmux cheat sheet (identical for both methods):**
- **Detach** (leave training running): `Ctrl+B`, then press `D`
- **Reattach** (reconnect to running session): `tmux attach -t training`
- **List sessions**: `tmux ls`
- **Kill session**: `tmux kill-session -t training`

**Browser SSH specific:**
- After detaching, you can safely close the browser tab
- Training continues on the server
- Open a new browser SSH tab anytime and run `tmux attach -t training` to reconnect

---

### Activate Conda Environment

```bash
conda activate last
```

**What this does:**
- Activates the Python environment with PyTorch and all dependencies
- Your prompt will change to show `(last)` at the beginning

**Verify activation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.5.x+cu124
CUDA: True
```

---

### Start Training

```bash
bash ~/LAST/scripts/gcp_train_base.sh
```

**What this does:**

1. Sets environment variables:
   - `CUDA_VISIBLE_DEVICES=0` - Use first GPU
   - `OMP_NUM_THREADS=8` - Optimize for 8 CPU cores

2. Creates timestamped run directory:
   - Example: `~/LAST-runs/base-xsub-20250220-143052/`
   - Stores logs, checkpoints, and final model

3. Launches training with:
   - **Model**: LAST-v2-Base (5.8M params)
   - **Dataset**: NTU RGB+D 60 classes, cross-subject split
   - **Batch size**: 4 per GPU step
   - **Gradient accumulation**: 4 steps (effective batch = 16)
   - **Learning rate**: 0.05 with 10-epoch warmup
   - **Epochs**: 70 (~18-20 hours on P100)

4. Runs in background with `nohup` - survives SSH disconnects

**Expected initial output:**
```
=========================================
LAST v2 Base Training on GCP P100
=========================================

Run directory: /home/yourname/LAST-runs/base-xsub-20250220-143052
Model: LAST-v2-Base
Dataset: NTU RGB+D 60
Split: xsub
Effective batch size: 16 (batch=4 × accum=4)
GPU: Tesla P100-PCIE-16GB
=========================================

Training started!
  PID: 12345
  Log: /home/yourname/LAST-runs/base-xsub-20250220-143052/train.log

Monitor progress:
  tail -f /home/yourname/LAST-runs/base-xsub-20250220-143052/train.log
```

**Training will show:**
- Model architecture summary
- Parameter count (~5.8M)
- Dataset loading (train: 40,320 samples, val: 16,560 samples)
- Epoch progress with tqdm bars
- Loss, top-1 accuracy, top-5 accuracy per epoch
- Validation metrics every epoch
- Checkpoint saves every 10 epochs

---

### Detach from Tmux

**Once training starts successfully, detach to let it run in background:**

1. Press `Ctrl+B` (release both keys)
2. Then press `D`

You'll see: `[detached (from session training)]`

**What this means (identical for both methods):**
- Training continues running on the server
- **Method A**: You can safely close your PowerShell/CMD window
- **Method B**: You can safely close your browser tab
- SSH connection can drop without affecting training
- Instance will keep running until you stop it or training completes

---

### Monitor Training Progress

#### Option A: Reattach to tmux session

**Method A (local gcloud SDK):**
```bash
# SSH back into instance from your local machine
gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007

# Reattach to running session
tmux attach -t training
```

**Method B (browser SSH):**
```bash
# 1. Go to GCP Console: https://console.cloud.google.com/compute/instances?project=research-454007
# 2. Click SSH button again (opens new tab)
# 3. In the new terminal, run:
tmux attach -t training
```

You'll see the live training output (identical for both methods).

---

#### Option B: Tail the log file

**Same for both methods (after SSH reconnection):**
```bash
# Find the run directory (with timestamp)
ls ~/LAST-runs/

# Tail the log
tail -f ~/LAST-runs/base-xsub-*/train.log
```

**Press `Ctrl+C` to stop tailing** (doesn't stop training)

**Note:** For Method B users, reconnect via browser SSH first (see Option A above)

---

#### Option C: Check GPU utilization

**Same command for both methods:**
```bash
watch -n 1 nvidia-smi
```

**What to look for (identical output for both methods):**
- **GPU Utilization**: Should be 80-100% during training
- **Memory Usage**: Should be ~12-14GB / 16GB (P100)
- **Temperature**: Should be 60-80°C
- **Power**: Should be 150-250W / 250W max

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.x.x       Driver Version: 525.x.x       CUDA Version: 12.4 |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   68C    P0   180W / 250W |  13824MiB / 16384MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**Press `Ctrl+C` to exit**

---

### Check Training Progress (Metrics to Watch)

**Training behavior is identical regardless of how you connected (Method A or B):**

**Expected training behavior:**

#### Epochs 1-10 (Warmup Phase)
- **Loss**: Should decrease from ~4.0 to ~2.5
- **Train top-1 acc**: Should increase from ~5% to ~15-25%
- **Val top-1 acc**: Should increase from ~3% to ~10-20%
- **Learning rate**: Gradually increases from 0.005 → 0.05

**🚨 Red flags:**
- Loss becomes NaN/Inf → restart with lower LR
- Val acc stays at ~1.67% (random chance) → data loading issue
- GPU util < 50% → dataloader bottleneck (increase num_workers)

---

#### Epochs 11-40 (High LR Phase)
- **Loss**: Should decrease to ~1.5-2.0
- **Train top-1 acc**: Should reach 40-60%
- **Val top-1 acc**: Should reach 30-50%
- **Learning rate**: Decreases from 0.05 → 0.01 (cosine)

**What to look for:**
- Smooth loss curves (no sudden spikes)
- Val acc consistently improving
- Top-5 acc should be 15-20% higher than top-1

---

#### Epochs 41-70 (Fine-tuning Phase)
- **Loss**: Should decrease to ~1.0-1.5
- **Train top-1 acc**: Should reach 70-85%
- **Val top-1 acc**: Should reach 60-75% (target: 70%+ for SOTA)
- **Learning rate**: Decreases from 0.01 → 0.0001 (cosine floor)

**SOTA benchmarks for NTU-60 xsub:**
- CTR-GCN: 89.9% (heavyweight, ~3.5M params)
- AGCN: 88.5% (original baseline)
- **Target for LAST-v2-Base**: 70-75% (efficiency-focused, 5.8M params)

---

### Handle Long Training Runs

**Estimated training time:** 18-20 hours for 70 epochs on P100

**Best practices:**

1. **Set up email notifications** (optional):
   ```bash
   # At end of training script, send email
   echo "Training complete" | mail -s "LAST Training Done" your.email@gmail.com
   ```

2. **Enable auto-shutdown after training** (save costs):
   ```bash
   # Add to end of gcp_train_base.sh
   sudo shutdown -h +10  # Shutdown 10 min after training ends
   ```

3. **Check training status remotely**:

   **Method A (local gcloud SDK):**
   ```bash
   # From local machine, check if training process is running
   gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007 --command="ps aux | grep train.py"
   ```

   **Method B (browser SSH):**
   ```bash
   # Open browser SSH and run:
   ps aux | grep train.py
   ```

4. **Estimate completion time**:
   - Each epoch takes ~15-18 minutes on P100
   - 70 epochs = 1050-1260 minutes = 17.5-21 hours
   - Check log for epoch timestamps to calculate remaining time

---

### After Training Completes

#### Verify training finished successfully

**Reconnect to instance (choose your method):**

**Method A:**
```bash
gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
```

**Method B:**
- Go to GCP Console → Click SSH button

**Then check logs (same for both methods):**
```bash
# Check last 100 lines of log
tail -n 100 ~/LAST-runs/base-xsub-*/train.log
```

**Look for:**
```
Epoch 70/70 - Loss: 1.234 - Top-1: 72.34% - Top-5: 88.56%
Validation - Loss: 1.456 - Top-1: 68.90% - Top-5: 85.23%
Best validation accuracy: 70.12% at epoch 67
Training completed successfully!
```

---

#### Check generated files

```bash
ls -lh ~/LAST-runs/base-xsub-*/
```

**Expected files:**
```
checkpoints/
  ├── best_model.pth          # Best validation accuracy checkpoint
  ├── checkpoint_epoch_10.pth # Periodic checkpoint
  ├── checkpoint_epoch_20.pth
  ├── ...
  └── checkpoint_epoch_70.pth # Final epoch

train.log                      # Full training log
train.pid                      # Process ID (for monitoring)
config_snapshot.yaml           # Config used for this run
```

---

#### Sync results to GCS bucket (backup + access from anywhere)

**Run from the GCP instance (works same for both Method A and B):**

```bash
# Sync entire run directory to GCS
gsutil -m rsync -r ~/LAST-runs/base-xsub-* gs://research-last/runs/base-xsub-run1/

# Verify upload
gsutil ls gs://research-last/runs/base-xsub-run1/
```

**What this preserves:**
- All checkpoints (best model + periodic saves)
- Full training log
- Config snapshot
- Can download later for inference/analysis

---

#### Download best model to local machine (optional)

**Method A (direct download via gcloud SDK):**
```bash
# On your local Windows machine
gcloud compute scp research-last-v2-base-run1:~/LAST-runs/base-xsub-*/checkpoints/best_model.pth C:\Users\pathi\OneDrive\Desktop\LAST\models\ --zone=asia-east1-c --project=research-454007
```

**Method B (via GCS bucket - works for both methods):**
```bash
# On your local machine (requires gsutil)
gsutil cp gs://research-last/runs/base-xsub-run1/checkpoints/best_model.pth C:\Users\pathi\OneDrive\Desktop\LAST\models\
```

**Alternative for Method B users without gsutil:**
1. Go to GCS Console: https://console.cloud.google.com/storage/browser/research-last/runs/base-xsub-run1/checkpoints
2. Find `best_model.pth`
3. Click the three dots → Download

---

### Clean Up Resources (Important!)

**These commands work from your local machine for both methods:**

#### Stop the instance (keeps disk, can restart later)

**Method A (from local machine):**
```bash
gcloud compute instances stop research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
```

**Method B (from GCP Console):**
1. Go to: https://console.cloud.google.com/compute/instances?project=research-454007
2. Find `research-last-v2-base-run1`
3. Click the three dots → Stop
4. Confirm

**Cost:** ~$0.40/day for stopped instance (disk storage only)

---

#### Delete the instance (removes everything, save ~$10/day)

**Method A (from local machine):**
```bash
# ONLY after syncing results to GCS!
gcloud compute instances delete research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
```

**Method B (from GCP Console):**
1. Go to: https://console.cloud.google.com/compute/instances?project=research-454007
2. Find `research-last-v2-base-run1`
3. Click the three dots → Delete
4. Confirm (type instance name)

**⚠️ Warning:** This is irreversible. Make sure you've backed up:
- ✅ Training logs to GCS
- ✅ All checkpoints to GCS
- ✅ Any analysis results

---

## Training Configuration Summary

**Model:** LAST-v2-Base
- Architecture: Multi-Input Branch (MIB) with adaptive graph
- Channels: [96, 192, 384]
- Blocks: [4, 5, 5]
- Parameters: ~5.8M
- Input streams: joint + velocity + bone

**Training Hyperparameters:**
- Optimizer: SGD + Nesterov momentum (0.9)
- Learning rate: 0.05 (reduced from 0.1 for stability)
- LR schedule: Linear warmup (10 epochs) → Cosine decay
- Warmup range: 0.005 → 0.05
- Min LR: 0.0001
- Weight decay: 0.0004 (excludes bias/BN/alpha/A_learned)
- Gradient clipping: 1.0 (tighter for adaptive graph)
- Label smoothing: 0.1
- Batch size: 4 per step
- Gradient accumulation: 4 steps (effective batch = 16)
- Epochs: 70
- Mixed precision: Enabled (AMP)

**Data:**
- Dataset: NTU RGB+D 60 classes
- Split: Cross-subject (xsub)
- Train samples: 40,320
- Val samples: 16,560
- Input frames: 64 (temporal crop)
- Augmentation: Temporal jitter, spatial rotation

**Hardware:**
- GPU: NVIDIA Tesla P100 (16GB HBM2)
- CPU: 8 vCPUs (n1-standard-8)
- Memory: 30GB RAM
- Disk: 100GB SSD

---

## Troubleshooting

### Training won't start

**Symptom:** Script exits immediately after running

**Solutions:**
1. Check conda environment is activated: `conda activate last`
2. Verify PyTorch installed: `python -c "import torch; print(torch.__version__)"`
3. Check GPU detected: `nvidia-smi`
4. Check dataset files exist: `ls ~/data/LAST-60-v2/data/processed_v2/xsub/`

---

### Loss becomes NaN

**Symptom:** Training log shows "WARNING: non-finite loss" and val acc drops to 0%

**Solutions:**
1. **Reduce learning rate further**: Edit `configs/training/default.yaml`, change `lr: 0.05` → `lr: 0.02`
2. **Extend warmup**: Change `warmup_epochs: 10` → `warmup_epochs: 15`
3. **Tighten gradient clip**: Change `gradient_clip: 1.0` → `gradient_clip: 0.5`
4. **Restart training** (current run is corrupted after NaN)

---

### GPU out of memory

**Symptom:** `CUDA out of memory` error

**Solutions:**
1. Reduce batch size: `batch_size: 4` → `batch_size: 2`
2. Increase accumulation: `gradient_accumulation_steps: 4` → `gradient_accumulation_steps: 8`
3. Keep effective batch same: 2 × 8 = 16

---

### Training very slow

**Symptom:** Each epoch takes > 30 minutes

**Solutions:**
1. Check GPU utilization: `nvidia-smi` should show 80-100%
2. Increase dataloader workers: Edit `configs/environment/gcp.yaml`, change `num_workers: 8` → `num_workers: 12`
3. Enable torch.compile: Edit `configs/training/default.yaml`, change `use_compile: false` → `use_compile: true`

---

### Val accuracy stuck at ~1.67%

**Symptom:** Validation accuracy doesn't improve from random chance

**Possible causes:**
1. **Data loading issue** - Check dataset files are not corrupted
2. **Label mismatch** - Verify `train_label.pkl` and `val_label.pkl` are correct
3. **Model bug** - Check model forward pass doesn't have bugs

**Debug:**
```bash
# Test data loading
python -c "
import pickle
with open('~/data/LAST-60-v2/data/processed_v2/xsub/train_label.pkl', 'rb') as f:
    labels = pickle.load(f)
    print(f'Train samples: {len(labels)}')
    print(f'Label range: {min(labels)} to {max(labels)}')
"
```

---

### SSH connection lost

**Symptom:** Terminal shows "Connection closed" or "Network error"

**Impact:** None if running in tmux!

**Solution:**
1. Reconnect: `gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007`
2. Reattach: `tmux attach -t training`
3. Training continues unaffected

---

### Cannot reattach to tmux

**Symptom:** `tmux attach -t training` shows "no sessions"

**Cause:** Training crashed or tmux session was killed

**Debug:**
1. Check if training process still running: `ps aux | grep train.py`
2. Check log for errors: `tail -n 200 ~/LAST-runs/base-xsub-*/train.log`
3. If crashed, review error and restart training

---

## Cost Estimation

**P100 instance (n1-standard-8 + 1x P100):**
- Running cost: ~$1.46/hour
- 20 hours training: ~$29.20
- Storage (100GB): ~$0.40/day when stopped

**Total estimated cost for one training run:** ~$30-35

**Cost optimization tips:**
1. Stop instance immediately after training completes
2. Use preemptible instances (70% cheaper, may get interrupted)
3. Delete instance after syncing to GCS
4. Use smaller batch size to test code before full run

---

## Next Steps After Training

1. **Analyze results**: Plot loss/accuracy curves, compare to baseline
2. **Test inference**: Load best checkpoint and run on test set
3. **Train LAST-Large**: Repeat process with `configs/model/last_large.yaml`
4. **Knowledge distillation**: Train small variant using Base/Large as teachers
5. **Hyperparameter tuning**: Try different LR, warmup, weight decay
6. **Ablation studies**: Disable components (adaptive graph, ST-attention) to measure impact

---

## Quick Reference Commands

### Method A: Local gcloud SDK

```bash
# 1. Upload repo to GCP
gcloud compute scp --recurse C:\Users\pathi\OneDrive\Desktop\LAST research-last-v2-base-run1:~/ --zone=asia-east1-c --project=research-454007

# 2. SSH into instance
gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007

# 3. Run setup (one-time)
bash ~/LAST/scripts/gcp_setup.sh

# 4. Start training (in tmux)
tmux new -s training
conda activate last
bash ~/LAST/scripts/gcp_train_base.sh
# Detach: Ctrl+B, D

# 5. Monitor (reconnect first)
gcloud compute ssh research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
tmux attach -t training           # Reattach to session
tail -f ~/LAST-runs/base-xsub-*/train.log  # View log
watch -n 1 nvidia-smi             # GPU usage

# 6. After training (on instance)
gsutil -m rsync -r ~/LAST-runs/base-xsub-* gs://research-last/runs/base-xsub-run1/

# 7. Stop instance (from local machine)
gcloud compute instances stop research-last-v2-base-run1 --zone=asia-east1-c --project=research-454007
```

---

### Method B: Browser SSH

```bash
# 1. Upload repo to GCS (from local machine)
gsutil -m cp -r C:\Users\pathi\OneDrive\Desktop\LAST gs://research-last/code/LAST/

# 2. Open browser SSH
# Go to: https://console.cloud.google.com/compute/instances?project=research-454007
# Click SSH button on research-last-v2-base-run1

# 3. Download repo from GCS
gsutil -m cp -r gs://research-last/code/LAST ~/

# 4. Fix line endings and run setup
sed -i 's/\r$//' ~/LAST/scripts/*.sh
chmod +x ~/LAST/scripts/*.sh
bash ~/LAST/scripts/gcp_setup.sh

# 5. Start training (in tmux)
tmux new -s training
conda activate last
bash ~/LAST/scripts/gcp_train_base.sh
# Detach: Ctrl+B, D

# 6. Monitor (open new browser SSH tab)
# Click SSH button again, then:
tmux attach -t training           # Reattach to session
tail -f ~/LAST-runs/base-xsub-*/train.log  # View log
watch -n 1 nvidia-smi             # GPU usage

# 7. After training (on instance)
gsutil -m rsync -r ~/LAST-runs/base-xsub-* gs://research-last/runs/base-xsub-run1/

# 8. Stop instance
# Go to GCP Console → Instances → Three dots → Stop
```

---

### Common Commands (Same for Both Methods)

**On the GCP instance (after SSH connection):**
```bash
# Check if training is running
ps aux | grep train.py
tmux ls

# View recent log output
tail -n 50 ~/LAST-runs/base-xsub-*/train.log

# Check GPU memory
nvidia-smi

# Check disk space
df -h

# Find training run directory
ls -lth ~/LAST-runs/
```

---

## Contact & Support

- GitHub Issues: https://github.com/Vayuputra2401/LAST/issues
- Training logs: Check `~/LAST-runs/*/train.log`
- GCP Console: https://console.cloud.google.com/compute/instances?project=research-454007

---

**Last Updated:** 2026-02-20
**Author:** Pathikreet (LAST v2 Research)
**Instance:** research-last-v2-base-run1 (P100, asia-east1-c)
