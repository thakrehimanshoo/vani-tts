# Quick end-to-end smoke test on Windows (PowerShell).
# Requires .\setup.ps1 to have been run first.
#
# Run from the repo root:
#     .\examples\run_example.ps1

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path .venv)) {
    Write-Error "No .venv found. Run: .\setup.ps1"
    exit 1
}

$PythonExe = Join-Path (Resolve-Path .venv) 'Scripts\python.exe'

if (-not (Test-Path outputs)) {
    New-Item -ItemType Directory -Path outputs | Out-Null
}

& $PythonExe tts.py `
    --input examples\sample_input.txt `
    --output outputs\sample.wav `
    --emotion storytelling `
    --rate 0.97 `
    --pause-scale 1.1 `
    --subtitles outputs\sample.srt

Write-Host ""
Write-Host "Wrote outputs\sample.wav and outputs\sample.srt" -ForegroundColor Green
