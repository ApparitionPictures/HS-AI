@echo off
cd /d "%~dp0\.."
echo Installing UE4SS and the HS-AI telemetry mod into the Half Sword folder...
echo (If this fails with "Permission denied", right-click this file and choose "Run as administrator".)
".venv\Scripts\python.exe" -m hsai mod install %*
echo.
".venv\Scripts\python.exe" -m hsai mod status
pause
