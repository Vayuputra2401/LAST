#!/usr/bin/env bash
# ============================================================
#  ShiftFuse-Zero — Linux/macOS environment setup
#  Run once after cloning:  bash setup_env.sh
#  Requires: Python 3.9+
# ============================================================

set -e

echo ""
echo "=== Creating virtual environment ==="
python3 -m venv venv

echo ""
echo "=== Activating environment ==="
source venv/bin/activate

echo ""
echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Done! To activate in future sessions:"
echo "   source venv/bin/activate"
echo "============================================================"
