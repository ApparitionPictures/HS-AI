@echo off
cd /d "%~dp0\.."
echo Disabling only the HSAI Lua mod (UE4SS stays on). Use this to test whether UE4SS alone runs.
".venv\Scripts\python.exe" -m hsai mod disable
pause
