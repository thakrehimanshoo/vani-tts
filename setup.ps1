# One-shot installer for Vani TTS on Windows (PowerShell).
#
# Creates a Python venv, installs PyTorch with CUDA 12.1 wheels, installs
# Coqui TTS + audio deps, and pre-downloads XTTS v2 weights so the first
# synthesis call does not stall on a ~2 GB download.
#
# Run from the repo root in PowerShell:
#     .\setup.ps1
#
# If Windows blocks the script with "running scripts is disabled on this
# system", unblock it for this session only:
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = 'Stop'

# --- locate python -----------------------------------------------------------
$PythonBin = $env:PYTHON_BIN
if (-not $PythonBin) {
    foreach ($candidate in @('py -3.11', 'py -3.10', 'python', 'python3')) {
        $exe, $rest = $candidate -split ' ', 2
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            $PythonBin = $candidate
            break
        }
    }
}
if (-not $PythonBin) {
    Write-Error "No Python interpreter found. Install Python 3.10 or 3.11 from python.org (check 'Add to PATH')."
    exit 1
}
Write-Host "Using Python launcher: $PythonBin"

# --- python version check ----------------------------------------------------
$PyVer = & cmd /c "$PythonBin -c ""import sys;print('%d.%d'%sys.version_info[:2])"""
$PyVer = $PyVer.Trim()
if ($PyVer -ne '3.10' -and $PyVer -ne '3.11') {
    Write-Error "Python $PyVer is not supported. Install Python 3.10 or 3.11."
    exit 1
}

# --- ffmpeg check ------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found on PATH. Install with: winget install Gyan.FFmpeg"
    Write-Warning "  (or: choco install ffmpeg  — or download from https://ffmpeg.org and add to PATH)"
}

# --- create venv -------------------------------------------------------------
$VenvDir = '.venv'
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating venv at $VenvDir ..."
    & cmd /c "$PythonBin -m venv $VenvDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed"; exit 1 }
}

$PipExe    = Join-Path (Resolve-Path $VenvDir) 'Scripts\pip.exe'
$PythonExe = Join-Path (Resolve-Path $VenvDir) 'Scripts\python.exe'

# --- install dependencies ----------------------------------------------------
Write-Host "Upgrading pip, wheel, setuptools ..."
& $PythonExe -m pip install --upgrade pip wheel setuptools

Write-Host "Installing torch + torchaudio (CUDA 12.1 wheels) ..."
& $PipExe install --index-url https://download.pytorch.org/whl/cu121 `
    torch==2.1.2 torchaudio==2.1.2
if ($LASTEXITCODE -ne 0) { Write-Error "torch install failed"; exit 1 }

Write-Host "Installing requirements.txt ..."
& $PipExe install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "requirements install failed"; exit 1 }

# --- prefetch XTTS v2 weights and NLTK data ----------------------------------
$env:COQUI_TOS_AGREED = '1'

Write-Host "Pre-downloading XTTS v2 weights ..."
& $PythonExe -c @"
from TTS.api import TTS
print('Downloading XTTS v2 weights ...')
TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True)
print('Done.')
"@

Write-Host "Downloading NLTK sentence tokenizer data ..."
& $PythonExe -c @"
import nltk
for pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
"@

# --- done --------------------------------------------------------------------
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the venv with:   .\.venv\Scripts\Activate.ps1"
Write-Host "Smoke test:               python tts.py --input examples\sample_input.txt --output out.wav"
