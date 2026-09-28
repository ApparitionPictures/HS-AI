@echo off
rem Runs the mod installer with administrator rights (needed only if the game folder is write-protected).
powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~dp0install_mod.bat'"
