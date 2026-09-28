@echo off
cd /d "%~dp0\.."
set HOURS=%1
if "%HOURS%"=="" set HOURS=2
echo Starting training for %HOURS% hours. Half Sword must be running and in the foreground.
echo F11 = pause, F12 = stop.
".venv\Scripts\python.exe" -m hsai train --hours %HOURS%
pause
