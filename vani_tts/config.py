"""Defaults, presets, and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS v2 emits 24 kHz audio natively. We stitch at this rate and resample only
# if the caller asks for something else.
NATIVE_SAMPLE_RATE = 24_000

# Emotion presets. `style_prompt` is prepended to the input text as a soft
# conditioning hint; XTTS does not have a formal emotion token, but the
# decoder responds measurably to framing phrases. `temperature` and
# `repetition_penalty` are passed straight through to the XTTS inference call.
EMOTIONS: dict[str, dict] = {
    "neutral": {
        "temperature": 0.65,
        "repetition_penalty": 2.0,
        "length_penalty": 1.0,
        "style_prompt": None,
    },
    "happy": {
        "temperature": 0.80,
        "repetition_penalty": 2.0,
        "length_penalty": 1.0,
        "style_prompt": "spoken cheerfully and warmly",
    },
    "serious": {
        "temperature": 0.55,
        "repetition_penalty": 2.5,
        "length_penalty": 1.0,
        "style_prompt": "spoken in a calm, measured tone",
    },
    "storytelling": {
        "temperature": 0.75,
        "repetition_penalty": 2.0,
        "length_penalty": 1.0,
        "style_prompt": "narrated like a captivating story",
    },
    "dramatic": {
        "temperature": 0.90,
        "repetition_penalty": 1.8,
        "length_penalty": 1.1,
        "style_prompt": "spoken with dramatic emphasis",
    },
}


@dataclass
class Config:
    """Runtime configuration for a single synthesis job."""

    input_path: Path
    output_path: Path

    # Voice
    speaker_wav: Optional[Path] = None
    language: str = "en"

    # Expression
    emotion: str = "neutral"
    rate: float = 1.0
    pause_scale: float = 1.0
    style_prompt: Optional[str] = None  # overrides emotion preset if given

    # Chunking / performance
    max_chars: int = 250
    stream: bool = False
    batch_size: int = 1
    fp16: bool = True
    device: str = "cuda"  # falls back to cpu in model_loader if unavailable

    # Output
    sample_rate: int = NATIVE_SAMPLE_RATE
    normalize: bool = True
    crossfade_ms: int = 30
    subtitles_path: Optional[Path] = None
    background_path: Optional[Path] = None
    background_gain_db: float = -24.0

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path(".tts_cache"))

    def emotion_preset(self) -> dict:
        if self.emotion not in EMOTIONS:
            raise ValueError(
                f"Unknown emotion '{self.emotion}'. "
                f"Choose from: {sorted(EMOTIONS)}"
            )
        preset = dict(EMOTIONS[self.emotion])
        if self.style_prompt is not None:
            preset["style_prompt"] = self.style_prompt
        return preset

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.speaker_wav is not None and not self.speaker_wav.exists():
            raise FileNotFoundError(
                f"Speaker reference not found: {self.speaker_wav}"
            )
        if not 0.5 <= self.rate <= 1.5:
            raise ValueError(f"rate must be in [0.5, 1.5], got {self.rate}")
        if not 0.5 <= self.pause_scale <= 3.0:
            raise ValueError(
                f"pause_scale must be in [0.5, 3.0], got {self.pause_scale}"
            )
        if self.max_chars < 60:
            raise ValueError("max_chars < 60 is too small for XTTS")
        if self.max_chars > 400:
            # XTTS degrades past ~400 chars per inference.
            raise ValueError("max_chars > 400 exceeds XTTS safe window")
