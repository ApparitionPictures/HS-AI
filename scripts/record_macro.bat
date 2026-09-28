@echo off
cd /d "%~dp0\.."
echo This records what YOU do after a fight ends to start the next fight.
echo Recording starts 5 seconds after you press a key. Press F8 when the next fight has started.
pause
".venv\Scripts\python.exe" -m hsai macro record after_fight
pause
