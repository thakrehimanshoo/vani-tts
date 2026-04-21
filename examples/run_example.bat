@echo off
REM Quick end-to-end smoke test on Windows (CMD).
REM Requires setup.bat to have been run first.
REM
REM Run from the repo root:
REM     examples\run_example.bat

pushd "%~dp0\.."

if not exist .venv (
    echo No .venv found. Run: setup.bat
    popd
    exit /b 1
)

if not exist outputs mkdir outputs

".venv\Scripts\python.exe" tts.py ^
    --input examples\sample_input.txt ^
    --output outputs\sample.wav ^
    --emotion storytelling ^
    --rate 0.97 ^
    --pause-scale 1.1 ^
    --subtitles outputs\sample.srt

if errorlevel 1 (
    popd
    exit /b 1
)

echo.
echo Wrote outputs\sample.wav and outputs\sample.srt
popd
