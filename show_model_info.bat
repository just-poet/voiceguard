@echo off
setlocal
cd /d "%~dp0"
if not defined VOICEGUARD_DATA set "VOICEGUARD_DATA=D:\VoiceGuardData"
set "VOICEGUARD_PYTHON="
if exist "%VOICEGUARD_DATA%\venv\Scripts\python.exe" (
    set "VOICEGUARD_PYTHON=%VOICEGUARD_DATA%\venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "VOICEGUARD_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "VOICEGUARD_PYTHON=python"
)
"%VOICEGUARD_PYTHON%" tools/model_report.py
pause
