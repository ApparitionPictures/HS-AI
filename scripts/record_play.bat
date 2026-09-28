@echo off
cd /d "%~dp0\.."
echo Recording your own fights for imitation learning. Press F12 to stop.
".venv\Scripts\python.exe" -m hsai record
pause
