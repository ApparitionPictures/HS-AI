@echo off
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m hsai mod diag > mod_diag.txt 2>&1
type mod_diag.txt
echo.
echo (The same text was saved to mod_diag.txt in the HS-AI folder. Paste it to Claude.)
pause
