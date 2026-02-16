@echo off
set "ENV_DIR=C:\Users\anubh\envs\ml_global"

if not exist "%ENV_DIR%" (
    echo Creating venv at %ENV_DIR%...
    python -m venv "%ENV_DIR%"
)

echo Activating...
call "%ENV_DIR%\Scripts\activate"

echo Installing PyTorch CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing other requirements...
pip install -r Teacher-LAST\requirements.txt

echo Setup Done.
