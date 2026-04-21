@echo off
REM One-shot installer for Vani TTS on Windows (CMD).
REM
REM Simpler fallback if PowerShell is locked down. Does the same thing as
REM setup.ps1: creates a venv, installs PyTorch with CUDA 12.1, installs
REM the rest of requirements.txt, and pre-downloads XTTS v2 weights.
REM
REM Run from the repo root:
REM     setup.bat

setlocal enabledelayedexpansion

REM --- locate python -----------------------------------------------------
set "PYTHON_BIN=%PYTHON_BIN%"
if "%PYTHON_BIN%"=="" (
    where py >nul 2>&1 && set "PYTHON_BIN=py -3.11"
)
if "%PYTHON_BIN%"=="" (
    where python >nul 2>&1 && set "PYTHON_BIN=python"
)
if "%PYTHON_BIN%"=="" (
    echo Error: No Python interpreter found. Install Python 3.10 or 3.11 from python.org.
    exit /b 1
)
echo Using Python launcher: %PYTHON_BIN%

REM --- python version check ---------------------------------------------
for /f "delims=" %%V in ('%PYTHON_BIN% -c "import sys;print(\"%%d.%%d\"%%sys.version_info[:2])"') do set "PYVER=%%V"
if not "%PYVER%"=="3.10" if not "%PYVER%"=="3.11" (
    echo Error: Python %PYVER% is not supported. Use 3.10 or 3.11.
    exit /b 1
)

REM --- ffmpeg check ------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo Warning: ffmpeg not found on PATH.
    echo          Install with: winget install Gyan.FFmpeg
)

REM --- create venv -------------------------------------------------------
if not exist .venv (
    echo Creating venv at .venv ...
    %PYTHON_BIN% -m venv .venv
    if errorlevel 1 exit /b 1
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "VENV_PIP=%CD%\.venv\Scripts\pip.exe"

REM --- install dependencies ---------------------------------------------
echo Upgrading pip, wheel, setuptools ...
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 exit /b 1

echo Installing torch + torchaudio (CUDA 12.1 wheels) ...
"%VENV_PIP%" install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 torchaudio==2.1.2
if errorlevel 1 exit /b 1

echo Installing requirements.txt ...
"%VENV_PIP%" install -r requirements.txt
if errorlevel 1 exit /b 1

REM --- prefetch XTTS v2 weights -----------------------------------------
set COQUI_TOS_AGREED=1
echo Pre-downloading XTTS v2 weights ...
"%VENV_PY%" -c "from TTS.api import TTS; print('Downloading XTTS v2 weights ...'); TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True); print('Done.')"

echo Downloading NLTK data ...
"%VENV_PY%" -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab')]"

echo.
echo Setup complete.
echo Activate the venv with:   .venv\Scripts\activate.bat
echo Smoke test:               python tts.py --input examples\sample_input.txt --output out.wav
endlocal
