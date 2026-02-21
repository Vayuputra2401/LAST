#!/bin/bash
# GCP Training Script for LAST v2 Base
# Run this inside a tmux/screen session for long training runs

set -e

echo "========================================="
echo "LAST v2 Base Training on GCP P100"
echo "========================================="

# Activate conda environment (if not already active)
# Note: Run this script with 'last' conda environment already activated
# The script will work regardless of conda path differences

# Navigate to project
cd ~/LAST

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Training configuration
MODEL_VARIANT="base"
DATASET="ntu60"  # NTU RGB+D 60 classes (ntu60 or ntu120)
SPLIT="xsub"     # Cross-subject split (xsub, xview, xset)

# Create run directory with timestamp
RUN_NAME="base-xsub-$(date +%Y%m%d-%H%M%S)"
RUN_DIR=~/LAST-runs/${RUN_NAME}
mkdir -p ${RUN_DIR}

echo -e "\nRun directory: ${RUN_DIR}"
echo "Model: LAST-v2-Base"
echo "Dataset: NTU RGB+D ${DATASET}"
echo "Split: ${SPLIT}"
echo "Batch size: 16 | LR: 0.01 (safe config)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Start training with nohup (continues even if SSH disconnects)
nohup python scripts/train.py \
    --model ${MODEL_VARIANT} \
    --dataset ${DATASET} \
    --split_type ${SPLIT} \
    --env gcp \
    --amp \
    > ${RUN_DIR}/train.log 2>&1 &

TRAIN_PID=$!
echo ${TRAIN_PID} > ${RUN_DIR}/train.pid

echo -e "\nTraining started!"
echo "  PID: ${TRAIN_PID}"
echo "  Log: ${RUN_DIR}/train.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${RUN_DIR}/train.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill training:"
echo "  kill ${TRAIN_PID}"
echo "========================================="

# Show initial log output
sleep 3
tail -n 50 ${RUN_DIR}/train.log
