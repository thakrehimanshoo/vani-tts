"""Emotion, pause, and speaking-rate controls.

XTTS v2 does not expose formal emotion tokens, but two levers give us
measurable expressive control:

1. **Style prompt framing.** Prepending a short framing phrase (e.g. "spoken
   cheerfully and warmly") consistently shifts prosody without hurting
   intelligibility. We prepend it once per chunk, with an em-dash separator.
2. **Sampling knobs.** `temperature`, `repetition_penalty`, and `length_penalty`
   flow straight through to the XTTS inference call.

Pause scaling is handled by rewriting punctuation into post-chunk silence
durations (see `pause_after_chunk`) rather than by mutating the text that goes
into the model — inserting literal silence tokens tends to confuse XTTS's
duration predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vani_tts.chunker import Chunk
from vani_tts.config import EMOTIONS


@dataclass(frozen=True)
class StyledChunk:
    """A chunk of text plus the conditioning that should accompany it."""

    text: str                       # what we actually send to XTTS
    original_text: str              # pre-styling (for subtitles)
    temperature: float
    repetition_penalty: float
    length_penalty: float
    pause_after_s: float            # silence to append after this chunk
    paragraph_index: int
    is_paragraph_end: bool


# Base pause durations (seconds) before scaling. Calibrated against natural
# narration — a mid-sentence chunk break gets almost nothing, a paragraph end
# gets roughly a breath.
_BASE_PAUSE_SENTENCE_END = 0.25
_BASE_PAUSE_PARAGRAPH_END = 0.70
_BASE_PAUSE_QUESTION = 0.35
_BASE_PAUSE_EXCLAIM = 0.30
_BASE_PAUSE_ELLIPSIS = 0.45


def _base_pause_for(chunk: Chunk) -> float:
    if chunk.is_paragraph_end:
        return _BASE_PAUSE_PARAGRAPH_END
    last = chunk.text.rstrip().rstrip("\"')]")[-1:] if chunk.text else ""
    if last == "?":
        return _BASE_PAUSE_QUESTION
    if last == "!":
        return _BASE_PAUSE_EXCLAIM
    if chunk.text.rstrip().endswith("…") or chunk.text.rstrip().endswith("..."):
        return _BASE_PAUSE_ELLIPSIS
    return _BASE_PAUSE_SENTENCE_END


def _apply_rate_to_style(style_prompt: Optional[str], rate: float) -> Optional[str]:
    """Fold speaking-rate hints into the style prompt.

    XTTS does not expose a direct rate knob, but the decoder is sensitive to
    pacing phrases. We bias only on clearly-slow or clearly-fast requests.
    """
    if rate <= 0.85:
        hint = "spoken slowly and deliberately"
    elif rate >= 1.15:
        hint = "spoken at a brisk pace"
    else:
        return style_prompt
    return hint if not style_prompt else f"{style_prompt}, {hint}"


def style_chunks(
    chunks: list[Chunk],
    emotion: str = "neutral",
    rate: float = 1.0,
    pause_scale: float = 1.0,
    style_prompt_override: Optional[str] = None,
) -> list[StyledChunk]:
    """Apply emotion/rate/pause conditioning to a list of text chunks."""
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion: {emotion}")

    preset = EMOTIONS[emotion]
    style_prompt = (
        style_prompt_override if style_prompt_override is not None
        else preset["style_prompt"]
    )
    style_prompt = _apply_rate_to_style(style_prompt, rate)

    styled: list[StyledChunk] = []
    for chunk in chunks:
        text = chunk.text
        if style_prompt:
            # The em-dash separator is deliberately punctuated so the decoder
            # treats the framing phrase as a brief pre-clause and doesn't read
            # it aloud as part of the main sentence.
            text = f"{style_prompt} — {text}"

        pause = _base_pause_for(chunk) * pause_scale
        styled.append(
            StyledChunk(
                text=text,
                original_text=chunk.text,
                temperature=preset["temperature"],
                repetition_penalty=preset["repetition_penalty"],
                length_penalty=preset["length_penalty"],
                pause_after_s=pause,
                paragraph_index=chunk.paragraph_index,
                is_paragraph_end=chunk.is_paragraph_end,
            )
        )
    return styled
