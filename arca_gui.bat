@echo off
title ARCA Panorama GUI

REM Fix UNC path issue - use a safe working directory
cd /d "%USERPROFILE%" 2>nul || cd /d C:\

echo.
echo   ARCA Panorama GUI
echo   ==================
echo.

REM Open browser first (before blocking on WSL)
echo   Abriendo navegador en http://localhost:5800 ...
start "" "http://localhost:5800"

echo   Iniciando servidor en WSL...
echo   Cierra esta ventana para detener el servidor.
echo.

REM Launch the Flask server in WSL with --no-browser flag
wsl -e bash -c "cd /home/erik90mx/.openclaw/workspace/ARCA && gui_venv/bin/python3 arca_gui.py --no-browser"

pause
