$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
}
& $taskPython -c "import numpy, scipy, sklearn, soundfile, fastapi, uvicorn, websockets" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection.' }
}
Write-Host 'Open http://127.0.0.1:8765 — press Ctrl+C to stop.'
& $taskPython -m uvicorn callguard.server:app --host 127.0.0.1 --port 8765 --ws-max-size 32768
