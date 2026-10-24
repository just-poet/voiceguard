$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:VOICEGUARD_DATA) { $env:VOICEGUARD_DATA = 'D:\VoiceGuardData' }
New-Item -ItemType Directory -Force -Path $env:VOICEGUARD_DATA | Out-Null
$voiceguardPython = Join-Path $env:VOICEGUARD_DATA 'venv\Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path $env:VOICEGUARD_DATA 'pip-cache'
$env:npm_config_cache = Join-Path $env:VOICEGUARD_DATA 'npm-cache'
$env:TEMP = Join-Path $env:VOICEGUARD_DATA 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
if (-not (Test-Path -LiteralPath $voiceguardPython)) {
    python -m venv (Join-Path $env:VOICEGUARD_DATA 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
}
& $voiceguardPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $voiceguardPython -m pip install -c requirements-voiceguard-transitive.txt -r requirements-voiceguard.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Push-Location frontend
try {
    npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependencies failed.' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
& $voiceguardPython tools/check_voiceguard_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Model asset check failed.' }
Write-Host 'Setup complete. Run START_VOICEGUARD.cmd.'
