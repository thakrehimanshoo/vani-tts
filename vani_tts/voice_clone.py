"""Voice-cloning utilities.

XTTS v2 accepts a short reference waveform and computes speaker + GPT
conditioning latents from it. The model is robust, but quality is very
sensitive to the reference audio itself: background noise, music, or clipping
all leak into the synthesized voice.

This module:

  1. Loads a user-supplied reference.
  2. Converts to mono, resamples to 22 kHz (XTTS's native reference rate).
  3. Trims leading/trailing silence.
  4. Caps to 30 s — anything longer hurts rather than helps.
  5. Light peak-normalizes so a quiet recording isn't underrepresented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vani_tts.utils import get_logger

_LOG = get_logger(__name__)

REFERENCE_SAMPLE_RATE = 22_050
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0
SILENCE_DB_THRESHOLD = -40.0


def _trim_silence(wav: np.ndarray, sr: int, top_db: float = -SILENCE_DB_THRESHOLD) -> np.ndarray:
    """Trim leading and trailing silence using librosa."""
    import librosa  # local import keeps module import cheap

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed


def preprocess_reference(
    path: Path,
    out_path: Optional[Path] = None,
) -> Path:
    """Clean a user-supplied reference clip and return the path to the cleaned
    version. If `out_path` is None the cleaned file is written next to the
    input with a `.cleaned.wav` suffix.
    """
    import librosa
    import soundfile as sf

    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    wav, sr = librosa.load(path, sr=REFERENCE_SAMPLE_RATE, mono=True)
    if wav.size == 0:
        raise ValueError(f"Reference audio is empty: {path}")

    duration = len(wav) / sr
    _LOG.info("Loaded reference '%s' (%.2f s)", path.name, duration)

    wav = _trim_silence(wav, sr)
    duration = len(wav) / sr

    if duration < MIN_DURATION_S:
        raise ValueError(
            f"Reference clip is only {duration:.2f} s after silence trim; "
            f"need at least {MIN_DURATION_S} s of clean speech."
        )

    if duration > MAX_DURATION_S:
        wav = wav[: int(MAX_DURATION_S * sr)]
        _LOG.info("Truncated reference to %.1f s", MAX_DURATION_S)

    # Peak-normalize to -1 dBFS, avoiding division by zero on silent input.
    peak = float(np.max(np.abs(wav))) or 1.0
    wav = (wav / peak) * 0.89

    if out_path is None:
        out_path = path.with_suffix(".cleaned.wav")
    sf.write(out_path, wav, sr, subtype="PCM_16")
    _LOG.info("Wrote cleaned reference to %s", out_path)
    return out_path
