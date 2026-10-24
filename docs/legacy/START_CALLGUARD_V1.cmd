@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import numpy, scipy, sklearn, soundfile, fastapi, uvicorn, websockets" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
  if errorlevel 1 goto fail
)
echo.
echo CallGuard is starting. Open http://127.0.0.1:8765 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m uvicorn callguard.server:app --host 127.0.0.1 --port 8765 --ws-max-size 32768
if errorlevel 1 goto fail
exit /b 0
:fail
echo.
echo Startup failed. Install Python 3.11 or later and check the error above.
echo If port 8765 is already in use, the app may already be running.
pause
exit /b 1
