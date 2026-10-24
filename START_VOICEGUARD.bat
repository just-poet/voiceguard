@echo off
setlocal enabledelayedexpansion
title VoiceGuard - AI Voice Detection

cd /d "%~dp0"

echo ===================================================
echo               STARTING VOICEGUARD
echo ===================================================

if not defined VOICEGUARD_DATA set "VOICEGUARD_DATA=D:\VoiceGuardData"
if not defined VOICEGUARD_THREADS set "VOICEGUARD_THREADS=6"
set "TEMP=%VOICEGUARD_DATA%\temp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"

:: Determine Python executable
set "VOICEGUARD_PYTHON="
if exist "%VOICEGUARD_DATA%\venv\Scripts\python.exe" (
    set "VOICEGUARD_PYTHON=%VOICEGUARD_DATA%\venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "VOICEGUARD_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 set "VOICEGUARD_PYTHON=python"
)

if "%VOICEGUARD_PYTHON%"=="" (
    echo Python environment not found. Running setup...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-voiceguard.ps1"
    if errorlevel 1 goto fail
    set "VOICEGUARD_PYTHON=%VOICEGUARD_DATA%\venv\Scripts\python.exe"
)

echo Using Python: %VOICEGUARD_PYTHON%

:: Check if port 8765 is already running
netstat -ano | findstr /R /C:":8765 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo VoiceGuard server is already running on port 8765!
    echo Opening browser to http://127.0.0.1:8765 ...
    start http://127.0.0.1:8765
    echo.
    echo Press any key to exit this launcher window...
    pause >nul
    exit /b 0
)

:: Launch browser in background after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"

echo.
echo ===================================================
echo Server starting at http://127.0.0.1:8765
echo Opening your browser automatically...
echo Keep this terminal window open while using VoiceGuard.
echo ===================================================
echo.

"%VOICEGUARD_PYTHON%" -m uvicorn voiceguard.server:app --host 127.0.0.1 --port 8765 --ws-max-size 32012 --ws-max-queue 2
if errorlevel 1 goto fail

exit /b 0

:fail
echo.
echo ===================================================
echo Startup failed! Please check the error above.
echo ===================================================
pause
exit /b 1
