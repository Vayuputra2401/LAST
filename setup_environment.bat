@echo off
REM Complete setup script for AI research environment
REM Run this once to create and setup the environment

echo ========================================
echo AI/ML Research Environment Setup
echo ========================================
echo.

REM Check Python version
python --version
echo.

REM Create environments directory
echo Creating environments directory...
if not exist "C:\Users\pathi\envs" mkdir "C:\Users\pathi\envs"

REM Create virtual environment
echo Creating virtual environment: ai_research
python -m venv C:\Users\pathi\envs\ai_research

REM Activate environment
echo Activating environment...
call C:\Users\pathi\envs\ai_research\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA
echo.
echo ========================================
echo Installing PyTorch with CUDA 12.6...
echo (This may take a few minutes)
echo ========================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Install other packages
echo.
echo ========================================
echo Installing additional libraries...
echo ========================================
pip install -r environment_setup.txt

REM Verify installation
echo.
echo ========================================
echo Verifying installation...
echo ========================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate this environment in the future, run:
echo   C:\Users\pathi\envs\ai_research\Scripts\activate
echo.
echo Or use the shortcut:
echo   activate_ai.bat
echo.

pause
