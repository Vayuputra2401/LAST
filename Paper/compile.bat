@echo off
REM ============================================================
REM  ShiftFuse-Zero paper — local Windows compile script
REM  Run from the Paper\ directory:  cd Paper && compile.bat
REM  Requires: MiKTeX (pdflatex, bibtex) on PATH
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo === Checking for pdflatex ===
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo ERROR: pdflatex not found. Make sure MiKTeX is installed and on PATH.
    echo   Install: winget install MiKTeX.MiKTeX
    echo   Then restart this terminal.
    pause
    exit /b 1
)
pdflatex --version | findstr "MiKTeX\|pdfTeX"

echo.
echo === Pass 1: pdflatex ===
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :error

echo.
echo === Pass 2: bibtex ===
bibtex main
REM bibtex may warn about missing fields — that's OK, continue

echo.
echo === Pass 3: pdflatex ===
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :error

echo.
echo === Pass 4: pdflatex (final) ===
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  SUCCESS! Output: Paper\main.pdf
echo ============================================================
start "" "main.pdf"
goto :end

:error
echo.
echo ============================================================
echo  COMPILATION FAILED — check main.log for details
echo ============================================================
echo Last 30 lines of main.log:
powershell -Command "Get-Content main.log -Tail 30"
pause
exit /b 1

:end
endlocal
