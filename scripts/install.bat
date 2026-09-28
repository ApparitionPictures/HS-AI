@echo off
setlocal EnableDelayedExpansion
title HS-AI installer
cd /d "%~dp0\.."
echo ============================================================
echo  HS-AI installer  (Half Sword AI)
echo  This creates a private Python environment in .venv and
echo  installs PyTorch with CUDA for your NVIDIA GPU.
echo ============================================================

set PY=
for %%v in (3.12 3.11 3.13) do (
  if "!PY!"=="" (
    py -%%v -c "import sys" >nul 2>nul && set PY=py -%%v
  )
)
if "!PY!"=="" (
  python -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12),(3,13))" >nul 2>nul && set PY=python
)
if "!PY!"=="" (
  echo Python 3.11/3.12 was not found. Installing Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  set PY=py -3.12
)
echo Using: !PY!
if not exist .venv (
  !PY! -m venv .venv || goto :fail
)
set VPY=.venv\Scripts\python.exe
"%VPY%" -m pip install --upgrade pip wheel setuptools || goto :fail

echo.
echo Installing PyTorch (CUDA 12.8 build; falls back to CUDA 12.6)...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :fail
)
echo.
echo Installing the rest...
"%VPY%" -m pip install -r requirements.txt || goto :fail
"%VPY%" -m pip install -e . || echo (optional editable install skipped; hsai.bat still works)
if /I "%~1"=="--trt" (
  echo Installing TensorRT (optional fastest inference backend)...
  "%VPY%" -m pip install tensorrt onnx
)
echo.
echo ============================================================
echo  Installing UE4SS + the HS-AI mod into the game folder ...
echo ============================================================
"%VPY%" -m hsai mod install
if errorlevel 1 (
  echo.
  echo The mod install FAILED. Read the message above. Common fixes:
  echo   - permission problem: run scripts\install_mod_admin.bat
  echo   - antivirus removed UE4SS: add the Half Sword folder to the exclusions, run scripts\install_mod.bat
  echo   - no internet: download the UE4SS zip in a browser and run: hsai mod install --zip "path\to\UE4SS.zip"
)
echo.
echo Running the system check...
"%VPY%" -m hsai doctor --quick --no-bench
echo.
echo Done. Next: start Half Sword, enter a fight, then run scripts\doctor.bat
echo Then record your menu flow with scripts\record_macro.bat and train with scripts\train.bat
pause
exit /b 0
:fail
echo.
echo Installation failed. Scroll up for the error, fix it and run this script again.
pause
exit /b 1
