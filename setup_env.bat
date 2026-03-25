@echo off
REM ============================================================
REM  ShiftFuse-Zero — Windows environment setup
REM  Run once after cloning:  setup_env.bat
REM  Requires: Python 3.9+ on PATH
REM ============================================================

echo.
echo === Creating virtual environment ===
python -m venv venv
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.9+ and add to PATH.
    pause & exit /b 1
)

echo.
echo === Activating environment ===
call venv\Scripts\activate

echo.
echo === Installing dependencies ===
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================================
echo  Done! To activate in future sessions:
echo    venv\Scripts\activate
echo ============================================================
cmd /k
