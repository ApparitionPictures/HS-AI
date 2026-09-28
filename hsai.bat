@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo HS-AI is not installed yet. Run scripts\install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m hsai %*
