"""End-to-end chunk → audio synthesis orchestrator.

Two execution modes:

* **in-memory** (default): rendered chunks are collected as numpy arrays and
  crossfaded at the end. Fast, uses ~(chunks * duration) RAM.
* **streaming** (`Config.stream=True`): each chunk is written to a temp WAV
  immediately and the merger concatenates from disk. Peak RAM stays flat,
  which is what makes 1-hour renders safe on a 3050 / 8 GB system.

Both modes also return a list of `SynthesizedChunk` records with start/end
timestamps so `subtitles.py` can build an SRT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from vani_tts.chunker import chunk_text
from vani_tts.config import Config
from vani_tts.emotion import StyledChunk, style_chunks
from vani_tts.model_loader import XTTSEngine, load_engine
from vani_tts.utils import get_logger, read_text_file, silence

_LOG = get_logger(__name__)


@dataclass
class SynthesizedChunk:
    """Metadata about one rendered chunk."""

    index: int
    original_text: str
    start_s: float
    end_s: float
    pause_after_s: float
    audio: Optional[np.ndarray] = None   # populated in in-memory mode
    path: Optional[Path] = None          # populated in streaming mode


class Synthesizer:
    def __init__(self, engine: XTTSEngine, config: Config) -> None:
        self.engine = engine
        self.config = config

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _plan(self, text: str) -> list[StyledChunk]:
        raw_chunks = chunk_text(text, max_chars=self.config.max_chars)
        if not raw_chunks:
            raise ValueError("Input text produced zero chunks.")
        _LOG.info(
            "Split text into %d chunks (avg %.0f chars)",
            len(raw_chunks),
            np.mean([len(c.text) for c in raw_chunks]),
        )
        return style_chunks(
            raw_chunks,
            emotion=self.config.emotion,
            rate=self.config.rate,
            pause_scale=self.config.pause_scale,
            style_prompt_override=self.config.style_prompt,
        )

    # ------------------------------------------------------------------
    # Per-chunk render
    # ------------------------------------------------------------------
    def _render_one(self, styled: StyledChunk, cond) -> np.ndarray:
        gpt_latent, speaker_emb = cond
        wav = self.engine.synthesize(
            text=styled.text,
            language=self.config.language,
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_emb,
            temperature=styled.temperature,
            repetition_penalty=styled.repetition_penalty,
            length_penalty=styled.length_penalty,
            speed=self.config.rate,
        )
        return wav.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------
    def run(self, work_dir: Optional[Path] = None) -> list[SynthesizedChunk]:
        """Render every chunk. Dispatches to streaming or in-memory mode."""
        text = read_text_file(self.config.input_path)
        styled = self._plan(text)

        _LOG.info(
            "Computing speaker conditioning (speaker_wav=%s)",
            self.config.speaker_wav,
        )
        cond = self.engine.compute_conditioning(self.config.speaker_wav)

        if self.config.stream:
            if work_dir is None:
                raise ValueError("streaming mode requires a work_dir")
            return list(self._run_streaming(styled, cond, work_dir))
        return list(self._run_in_memory(styled, cond))

    def _run_in_memory(
        self, styled: list[StyledChunk], cond
    ) -> Iterator[SynthesizedChunk]:
        sr = self.engine.sample_rate
        cursor = 0.0
        for idx, sc in enumerate(tqdm(styled, desc="synth", unit="chunk")):
            wav = self._render_one(sc, cond)
            duration = len(wav) / sr
            yield SynthesizedChunk(
                index=idx,
                original_text=sc.original_text,
                start_s=cursor,
                end_s=cursor + duration,
                pause_after_s=sc.pause_after_s,
                audio=wav,
            )
            # Account for the silence the merger will insert after this chunk.
            cursor += duration + sc.pause_after_s

    def _run_streaming(
        self, styled: list[StyledChunk], cond, work_dir: Path
    ) -> Iterator[SynthesizedChunk]:
        sr = self.engine.sample_rate
        work_dir.mkdir(parents=True, exist_ok=True)
        cursor = 0.0
        for idx, sc in enumerate(tqdm(styled, desc="synth", unit="chunk")):
            wav = self._render_one(sc, cond)
            duration = len(wav) / sr
            path = work_dir / f"chunk_{idx:06d}.wav"
            sf.write(path, wav, sr, subtype="PCM_16")
            yield SynthesizedChunk(
                index=idx,
                original_text=sc.original_text,
                start_s=cursor,
                end_s=cursor + duration,
                pause_after_s=sc.pause_after_s,
                path=path,
            )
            cursor += duration + sc.pause_after_s


def build_synthesizer(config: Config) -> Synthesizer:
    engine = load_engine(device=config.device, fp16=config.fp16)
    return Synthesizer(engine=engine, config=config)
