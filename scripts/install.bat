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
"%VPY%" -m pip install -e . || goto :fail
if /I "%~1"=="--trt" (
  echo Installing TensorRT (optional fastest inference backend)...
  "%VPY%" -m pip install tensorrt onnx
)
echo.
echo Running setup (installs the game mod and checks the system)...
"%VPY%" -m hsai setup
echo.
echo Done. Use hsai.bat for everything, e.g.:   hsai doctor     hsai train --hours 2
pause
exit /b 0
:fail
echo.
echo Installation failed. Scroll up for the error, fix it and run this script again.
pause
exit /b 1
