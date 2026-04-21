"""Smoke tests for emotion/pause/rate conditioning."""

from __future__ import annotations

import pytest

from vani_tts.chunker import Chunk
from vani_tts.emotion import style_chunks


def _chunk(text: str, idx: int = 0, end: bool = True) -> Chunk:
    return Chunk(text=text, paragraph_index=idx, is_paragraph_end=end)


def test_style_prompt_prepended_for_non_neutral():
    chunks = [_chunk("Hello world.")]
    styled = style_chunks(chunks, emotion="happy")
    assert styled[0].text.startswith("spoken cheerfully and warmly")
    assert styled[0].original_text == "Hello world."


def test_neutral_does_not_modify_text():
    chunks = [_chunk("Hello world.")]
    styled = style_chunks(chunks, emotion="neutral")
    assert styled[0].text == "Hello world."


def test_pause_scale_applies():
    chunks = [_chunk("End of paragraph.", end=True)]
    base = style_chunks(chunks, emotion="neutral", pause_scale=1.0)
    doubled = style_chunks(chunks, emotion="neutral", pause_scale=2.0)
    assert doubled[0].pause_after_s == pytest.approx(2.0 * base[0].pause_after_s)


def test_paragraph_end_gets_longer_pause():
    mid = _chunk("Mid sentence.", end=False)
    end = _chunk("End of para.", end=True)
    styled = style_chunks([mid, end], emotion="neutral")
    assert styled[1].pause_after_s > styled[0].pause_after_s


def test_rate_hint_appears_in_style_prompt():
    chunks = [_chunk("Hello.")]
    slow = style_chunks(chunks, emotion="neutral", rate=0.7)
    # With no base prompt and a slow rate, the rate hint becomes the prompt.
    assert "slowly" in slow[0].text


def test_unknown_emotion_raises():
    with pytest.raises(ValueError):
        style_chunks([_chunk("Hi.")], emotion="angstful")
