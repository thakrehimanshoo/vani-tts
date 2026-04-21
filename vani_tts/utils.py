"""Shared helpers."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


def get_logger(name: str = "vani_tts") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("VANI_LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


@contextmanager
def temp_dir(prefix: str = "vani_tts_") -> Iterator[Path]:
    """Auto-cleaning temp directory."""
    with tempfile.TemporaryDirectory(prefix=prefix) as d:
        yield Path(d)


def silence(duration_s: float, sample_rate: int) -> np.ndarray:
    """Generate a float32 silence buffer."""
    n = max(0, int(round(duration_s * sample_rate)))
    return np.zeros(n, dtype=np.float32)


def equal_power_crossfade(
    a: np.ndarray, b: np.ndarray, fade_samples: int
) -> np.ndarray:
    """Cosine (equal-power) crossfade between two mono float32 buffers.

    If either buffer is shorter than the fade, we shrink the fade to fit
    instead of throwing.
    """
    if fade_samples <= 0 or len(a) == 0 or len(b) == 0:
        return np.concatenate([a, b])

    fade = min(fade_samples, len(a), len(b))
    t = np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    head = a[:-fade]
    tail_a = a[-fade:] * fade_out
    head_b = b[:fade] * fade_in
    tail = b[fade:]
    overlap = tail_a + head_b
    return np.concatenate([head, overlap, tail])


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    # Strip UTF-8 BOM if present.
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("utf-8", errors="replace")
