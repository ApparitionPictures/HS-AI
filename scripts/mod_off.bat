@echo off
cd /d "%~dp0\.."
echo Turning UE4SS OFF: the game will run completely unmodified.
".venv\Scripts\python.exe" -m hsai mod off
pause
