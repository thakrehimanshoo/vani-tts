"""XTTS v2 loader.

Wraps the Coqui `TTS` class with:
  - lazy, cached instantiation (model load is expensive)
  - fp16 on CUDA when possible
  - one-time speaker-conditioning extraction so a reference wav is encoded once
    per job instead of once per chunk
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

from vani_tts.config import XTTS_MODEL_ID
from vani_tts.utils import get_logger

_LOG = get_logger(__name__)

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[tuple, "XTTSEngine"] = {}


class XTTSEngine:
    """Thin wrapper around `TTS.api.TTS` exposing the bits we need.

    We go through the high-level TTS class during load (so we get correct
    config + weights), but for inference we reach into `.synthesizer.tts_model`
    to call `inference()` directly. That gives us access to temperature,
    repetition_penalty, and the precomputed speaker latents — none of which
    the high-level `.tts()` method exposes.
    """

    def __init__(self, device: str = "cuda", fp16: bool = True) -> None:
        # Accept the Coqui TOS once, non-interactively.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        from TTS.api import TTS  # local import, heavy

        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device != "cuda":
            _LOG.warning("CUDA requested but unavailable — falling back to CPU.")
        self.fp16 = fp16 and self.device == "cuda"

        _LOG.info("Loading XTTS v2 on %s (fp16=%s)", self.device, self.fp16)
        tts = TTS(XTTS_MODEL_ID, progress_bar=False).to(self.device)

        if self.fp16:
            # XTTS's GPT decoder is the VRAM hotspot; cast it to half.
            try:
                tts.synthesizer.tts_model.half()
            except Exception as e:  # pragma: no cover - hardware dependent
                _LOG.warning("fp16 cast failed, staying in fp32: %s", e)
                self.fp16 = False

        self.tts = tts
        self.model = tts.synthesizer.tts_model
        self.sample_rate = tts.synthesizer.output_sample_rate
        _LOG.info("XTTS ready. Output SR=%d Hz", self.sample_rate)

    # ------------------------------------------------------------------
    # Speaker conditioning
    # ------------------------------------------------------------------
    def compute_conditioning(
        self, speaker_wav: Optional[Path]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GPT and speaker latents once per job.

        Returns `(gpt_cond_latent, speaker_embedding)`. When `speaker_wav` is
        None, we use the default XTTS studio speaker ("Claribel Dervla"); it
        is consistent across runs and a reasonable default narrator voice.
        """
        if speaker_wav is not None:
            gpt_cond_latent, speaker_embedding = (
                self.model.get_conditioning_latents(
                    audio_path=[str(speaker_wav)],
                    gpt_cond_len=self.model.config.gpt_cond_len,
                    max_ref_length=self.model.config.max_ref_len,
                    sound_norm_refs=self.model.config.sound_norm_refs,
                )
            )
            return gpt_cond_latent, speaker_embedding

        # No reference — use a built-in studio speaker.
        default = "Claribel Dervla"
        speaker_manager = self.model.speaker_manager
        if speaker_manager is None or default not in speaker_manager.speakers:
            raise RuntimeError(
                "XTTS default speaker not available; pass --speaker-wav."
            )
        entry = speaker_manager.speakers[default]
        return entry["gpt_cond_latent"], entry["speaker_embedding"]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        language: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor,
        temperature: float = 0.65,
        repetition_penalty: float = 2.0,
        length_penalty: float = 1.0,
        speed: float = 1.0,
    ):
        """Run one synthesis pass and return a 1-D float32 numpy array."""
        import numpy as np

        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            speed=float(speed),
            enable_text_splitting=False,  # we already did smart chunking
        )
        wav = out["wav"]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        return wav


def load_engine(device: str = "cuda", fp16: bool = True) -> XTTSEngine:
    """Load (or reuse) an XTTS engine. Safe to call multiple times."""
    key = (device, fp16)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = XTTSEngine(device=device, fp16=fp16)
        return _MODEL_CACHE[key]
