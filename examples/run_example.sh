#!/usr/bin/env bash
# Quick end-to-end smoke test. Requires `bash setup.sh` to have been run first.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
    echo "No .venv found. Run: bash setup.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p outputs

python tts.py \
    --input examples/sample_input.txt \
    --output outputs/sample.wav \
    --emotion storytelling \
    --rate 0.97 \
    --pause-scale 1.1 \
    --subtitles outputs/sample.srt

echo
echo "Wrote outputs/sample.wav and outputs/sample.srt"
