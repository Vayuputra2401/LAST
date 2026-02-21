#!/bin/bash
# GCP Instance Setup Script for LAST v2 Base Training
# Instance: research-last-v2-base-run1 (P100, n1-standard-8)

set -e  # Exit on error

echo "========================================="
echo "LAST v2 Base - GCP Training Setup"
echo "========================================="

# ── 1. System Setup ──────────────────────────────────────────────────────────
echo -e "\n[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git vim htop tmux screen

# ── 2. Verify GPU ────────────────────────────────────────────────────────────
echo -e "\n[2/6] Verifying NVIDIA GPU..."
nvidia-smi

# ── 3. Clone Repository ──────────────────────────────────────────────────────
echo -e "\n[3/6] Cloning LAST repository..."
cd ~
if [ -d "LAST" ]; then
    echo "  Repository already exists, pulling latest..."
    cd LAST && git pull && cd ~
else
    git clone https://github.com/Vayuputra2401/LAST.git
fi

# ── 4. Python Environment ────────────────────────────────────────────────────
echo -e "\n[4/6] Setting up Python environment..."
# GCP Deep Learning image comes with conda pre-installed
conda create -n last python=3.10 -y || true
source activate last

# Install PyTorch (CUDA 12.4 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
cd ~/LAST
pip install -r requirements.txt

# ── 5. Download Dataset from GCS ─────────────────────────────────────────────
echo -e "\n[5/6] Syncing preprocessed dataset from GCS..."
mkdir -p ~/data/LAST-60-v2/data/processed_v2/xsub
gsutil -m cp -r gs://research-last/LAST-60-v2/data/processed_v2/xsub/* ~/data/LAST-60-v2/data/processed_v2/xsub/

echo "  Dataset files:"
ls -lh ~/data/LAST-60-v2/data/processed_v2/xsub/

# ── 6. Verify Installation ───────────────────────────────────────────────────
echo -e "\n[6/6] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -e "\n========================================="
echo "Setup Complete!"
echo "========================================="
echo "Next steps:"
echo "  1. Activate environment: conda activate last"
echo "  2. Start training: bash ~/LAST/scripts/gcp_train_base.sh"
echo "========================================="
