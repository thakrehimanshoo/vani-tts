"""Smoke tests for the text chunker.

These run without Torch / TTS installed — they only exercise pure-Python code.
Run with: python -m pytest tests/test_chunker.py
"""

from __future__ import annotations

from vani_tts.chunker import chunk_text, split_sentences


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n  \n  ") == []


def test_respects_sentence_boundaries():
    # With a max_chars wide enough to hold every sentence whole, chunks must
    # always end in sentence-final punctuation.
    text = "Hello world. This is a test. Another sentence here."
    chunks = chunk_text(text, max_chars=30)
    for c in chunks:
        assert c.text.rstrip()[-1] in ".!?…", c.text
    assert "".join(c.text for c in chunks).replace(" ", "") == \
        text.replace(" ", "")


def test_paragraph_boundaries_marked():
    text = "First paragraph. It has one sentence.\n\nSecond paragraph here."
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) == 2
    assert chunks[0].paragraph_index == 0
    assert chunks[1].paragraph_index == 1
    assert chunks[0].is_paragraph_end is True
    assert chunks[1].is_paragraph_end is True


def test_abbreviations_do_not_split():
    text = "Dr. Smith went to St. Louis. He was late."
    sentences = split_sentences(text)
    # "Dr." and "St." should not end sentences.
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")


def test_oversized_sentence_split_on_clauses():
    sentence = (
        "This is a very long sentence, one with many clauses, "
        "separated by commas, and designed to exceed the limit."
    )
    chunks = chunk_text(sentence, max_chars=40)
    assert len(chunks) > 1
    # No chunk should wildly exceed the soft cap — clause-level splitting can
    # produce slightly longer lines because clauses themselves may exceed the
    # bound, but none should double it.
    for c in chunks:
        assert len(c.text) <= 80


def test_question_and_exclaim_preserved():
    text = "Why? Because! That's why."
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].text == "Why? Because! That's why."
