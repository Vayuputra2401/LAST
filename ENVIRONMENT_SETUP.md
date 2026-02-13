# AI/ML Research Environment Setup Guide (Python venv)

## Quick Setup with Python Virtual Environment

### Step 1: Check Python Version
```bash
python --version
```
Required: Python 3.10 or higher (recommended: 3.11)

If you need to update Python, download from: https://www.python.org/downloads/

---

### Step 2: Create Virtual Environment
```bash
# Navigate to a location for your global environments
# (e.g., C:\Users\pathi\envs or wherever you prefer)
mkdir C:\Users\pathi\envs
cd C:\Users\pathi\envs

# Create virtual environment
python -m venv ai_research
```

---

### Step 3: Activate Environment
```bash
# On Windows
C:\Users\pathi\envs\ai_research\Scripts\activate

# You should see (ai_research) in your prompt
```

**To make activation easier, create a shortcut:**
Create a file `activate_ai.bat` with:
```batch
@echo off
call C:\Users\pathi\envs\ai_research\Scripts\activate
```

Then just run `activate_ai.bat` from any location.

---

### Step 4: Upgrade pip
```bash
python -m pip install --upgrade pip
```

---

### Step 5: Install PyTorch with CUDA (Latest Stable)

**Check your GPU first:**
```bash
nvidia-smi
```
This will show your CUDA version.

**Install PyTorch 2.10.0 with CUDA 12.6 (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**OR for CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**OR for CUDA 11.8 (older GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**OR CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
```

---

### Step 6: Install Additional Libraries
```bash
pip install -r environment_setup.txt
```

---

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.10.0+cu126
CUDA Available: True
CUDA Version: 12.6
```

---

## Daily Usage

### Activate environment
```bash
# Navigate to your project
cd C:\Users\pathi\OneDrive\Desktop\LAST

# Activate environment
C:\Users\pathi\envs\ai_research\Scripts\activate

# Run your code
python scripts/quick_test.py
```

### Deactivate environment
```bash
deactivate
```

---

## Using Across Multiple Projects

The `ai_research` environment is **global** and can be used anywhere:

```bash
# Project 1: LAST
cd C:\Users\pathi\OneDrive\Desktop\LAST
C:\Users\pathi\envs\ai_research\Scripts\activate
python scripts/train.py

# Project 2: Another project
cd C:\Users\pathi\Documents\MyProject
C:\Users\pathi\envs\ai_research\Scripts\activate
python main.py
```

---

## Environment Management

### Check installed packages
```bash
pip list
```

### Install new package
```bash
pip install package_name
```

### Export environment
```bash
pip freeze > requirements_full.txt
```

### Delete environment (if needed)
```bash
# Deactivate first
deactivate

# Delete folder
rmdir /s C:\Users\pathi\envs\ai_research
```

---

## Troubleshooting

### Issue: "python not found"
- Make sure Python is in your PATH
- Try `py` instead of `python`
- Reinstall Python and check "Add to PATH"

### Issue: CUDA not available in PyTorch
1. Check GPU: `nvidia-smi`
2. Verify CUDA version matches PyTorch installation
3. Reinstall PyTorch with correct CUDA version

### Issue: Permission denied
- Run command prompt as Administrator
- Or use `--user` flag: `pip install --user package_name`

---

## Quick Reference

| Action | Command |
|--------|---------|
| Create env | `python -m venv ai_research` |
| Activate | `ai_research\Scripts\activate` |
| Deactivate | `deactivate` |
| Install package | `pip install package` |
| List packages | `pip list` |
| Upgrade pip | `python -m pip install --upgrade pip` |

---

## Included Libraries (after full install)

✅ PyTorch 2.10.0 (CUDA 12.6)  
✅ TorchVision, TorchAudio  
✅ NumPy, Pandas, SciPy, Scikit-learn  
✅ OpenCV, Pillow, Albumentations  
✅ Matplotlib, Seaborn, Plotly  
✅ TensorBoard, Weights & Biases  
✅ Video processing (decord, av)  
✅ Jupyter Notebook  
✅ Black, Pytest (dev tools)  

**Ready to use across all your ML/AI projects!** 🚀
