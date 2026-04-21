"""Stitch rendered chunks into a single continuous WAV.

Two join strategies:

* **equal-power crossfade** between consecutive chunks (default 30 ms) to kill
  clicks at the seam without audibly smearing consonants.
* **silence insertion** for the per-chunk pause budget (commas, sentence ends,
  paragraph breaks). Silence goes *after* the crossfade, so the fade is
  against actual speech tails, not against inserted silence.

Loudness is LUFS-normalized at the end (ITU-R BS.1770) rather than peak
normalized — peak normalization on speech over-amplifies the quietest
breath-laden chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from vani_tts.synthesizer import SynthesizedChunk
from vani_tts.utils import equal_power_crossfade, get_logger, silence

_LOG = get_logger(__name__)

_TARGET_LUFS = -20.0   # typical audiobook / podcast target


def _load_wav(path: Path, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav.astype(np.float32, copy=False)


def _lufs_normalize(wav: np.ndarray, sample_rate: int, target: float) -> np.ndarray:
    """Normalize integrated loudness to `target` LUFS.

    Falls back to peak normalization if pyloudnorm is unavailable or the input
    is too short for a valid LUFS measurement (<0.4 s).
    """
    if len(wav) < int(0.4 * sample_rate):
        peak = float(np.max(np.abs(wav))) or 1.0
        return (wav / peak) * 0.9

    try:
        import pyloudnorm as pyln
    except ImportError:  # pragma: no cover
        _LOG.warning("pyloudnorm not installed — falling back to peak norm")
        peak = float(np.max(np.abs(wav))) or 1.0
        return (wav / peak) * 0.9

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(wav)
    if not np.isfinite(loudness):
        peak = float(np.max(np.abs(wav))) or 1.0
        return (wav / peak) * 0.9
    adjusted = pyln.normalize.loudness(wav, loudness, target)
    # Hard-limit to prevent clipping from the LUFS boost.
    peak = float(np.max(np.abs(adjusted)))
    if peak > 0.99:
        adjusted = adjusted * (0.99 / peak)
    return adjusted.astype(np.float32, copy=False)


def _mix_background(
    speech: np.ndarray,
    background_path: Path,
    sample_rate: int,
    gain_db: float,
) -> np.ndarray:
    """Loop-mix a background track under the speech at `gain_db`."""
    bg = _load_wav(background_path, sample_rate)
    if len(bg) == 0:
        return speech

    # Tile/truncate the background to the speech length.
    reps = int(np.ceil(len(speech) / len(bg)))
    bg = np.tile(bg, reps)[: len(speech)]

    # Short fade-in/out on the background so it doesn't start/end abruptly.
    fade = int(min(len(bg), sample_rate * 1.5))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        bg[:fade] *= ramp
        bg[-fade:] *= ramp[::-1]

    gain = float(10.0 ** (gain_db / 20.0))
    return (speech + bg * gain).astype(np.float32, copy=False)


def merge(
    chunks: list[SynthesizedChunk],
    output_path: Path,
    sample_rate: int,
    crossfade_ms: int = 30,
    normalize: bool = True,
    background_path: Optional[Path] = None,
    background_gain_db: float = -24.0,
) -> Path:
    """Merge rendered chunks and write the final WAV.

    Works for both in-memory (`chunk.audio is not None`) and streaming
    (`chunk.path is not None`) modes.
    """
    if not chunks:
        raise ValueError("No chunks to merge.")

    fade_samples = max(0, int(sample_rate * crossfade_ms / 1000))
    result: np.ndarray = np.zeros(0, dtype=np.float32)

    for i, ch in enumerate(chunks):
        if ch.audio is not None:
            wav = ch.audio.astype(np.float32, copy=False)
        elif ch.path is not None:
            wav = _load_wav(ch.path, sample_rate)
        else:
            raise RuntimeError(f"Chunk {ch.index} has neither audio nor path.")

        if i == 0:
            result = wav
        else:
            result = equal_power_crossfade(result, wav, fade_samples)

        # Append the pause requested after this chunk (skip after the last one
        # to avoid trailing silence).
        if i < len(chunks) - 1 and ch.pause_after_s > 0:
            result = np.concatenate(
                [result, silence(ch.pause_after_s, sample_rate)]
            )

    if normalize:
        result = _lufs_normalize(result, sample_rate, _TARGET_LUFS)

    if background_path is not None:
        result = _mix_background(
            result, background_path, sample_rate, background_gain_db
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, result, sample_rate, subtype="PCM_16")
    _LOG.info(
        "Wrote %.2f s of audio to %s",
        len(result) / sample_rate,
        output_path,
    )
    return output_path
