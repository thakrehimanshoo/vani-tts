#!/usr/bin/env bash
# One-shot installer for Vani TTS.
#
# Creates a Python venv, installs PyTorch with CUDA 12.1 wheels, installs
# Coqui TTS + audio deps, and pre-downloads XTTS v2 weights so the first
# synthesis call does not stall on a ~2 GB download.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found. Install Python 3.10 or 3.11." >&2
    exit 1
fi

PY_VER=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PY_VER" in
    3.10|3.11) ;;
    *) echo "Error: Python $PY_VER is not supported. Use 3.10 or 3.11." >&2; exit 1 ;;
esac

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Warning: ffmpeg not found. Install it (sudo apt install ffmpeg)."
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

# Install Torch with CUDA 12.1 first (so the generic requirements.txt doesn't
# pull a CPU-only wheel).
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchaudio==2.1.2

pip install -r requirements.txt

# Prefetch the XTTS v2 weights. The first call always accepts the Coqui license;
# we set COQUI_TOS_AGREED=1 so it does not block the installer.
export COQUI_TOS_AGREED=1
python - <<'PY'
from TTS.api import TTS
print("Downloading XTTS v2 weights ...")
TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("Done.")
PY

# NLTK data for the sentence splitter.
python - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

echo
echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
echo "Smoke test:       python tts.py --input examples/sample_input.txt --output out.wav"
