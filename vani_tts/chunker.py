"""Paragraph- and sentence-aware chunker.

XTTS v2 is most stable on inputs of ~150–300 characters. We want to:

  1. Never split mid-sentence (that causes prosody glitches).
  2. Keep paragraph boundaries intact (they map to longer pauses downstream).
  3. Handle abbreviations, ellipses, quoted speech, and numbered lists.
  4. Split a single oversized sentence on clause boundaries as a last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    text: str
    paragraph_index: int
    is_paragraph_end: bool  # last chunk of its paragraph → longer pause after


_PARAGRAPH_RE = re.compile(r"\n\s*\n+")

# Sentence splitter: cut after ., !, ?, … and the sentence-ending variants in
# CJK/Devanagari. Keeps the punctuation with the sentence.
_SENTENCE_RE = re.compile(
    r"""(?<=[.!?।。！？…])["')\]]*\s+(?=[A-Z0-9"'(\[À-￿])""",
    re.UNICODE,
)

# Clause splitter (used only when a single sentence exceeds max_chars).
_CLAUSE_RE = re.compile(r"(?<=[,;:—–])\s+")

# Common English abbreviations that would otherwise fool the sentence regex.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr",
    "vs", "etc", "e.g", "i.e", "fig", "no",
}


def _merge_abbrev_splits(sentences: list[str]) -> list[str]:
    """Re-join sentences accidentally split after an abbreviation."""
    merged: list[str] = []
    for s in sentences:
        if merged:
            prev = merged[-1]
            last_word = re.split(r"\s+", prev.strip())[-1].rstrip(".").lower()
            if last_word in _ABBREVIATIONS:
                merged[-1] = f"{prev} {s}"
                continue
        merged.append(s)
    return merged


def split_sentences(paragraph: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(paragraph) if p.strip()]
    return _merge_abbrev_splits(parts)


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break a single long sentence along clause boundaries, then on spaces."""
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [c.strip() for c in _CLAUSE_RE.split(sentence) if c.strip()]
    out: list[str] = []
    buf = ""
    for clause in clauses:
        if len(clause) > max_chars:
            # Still too long — fall back to hard-wrap on whitespace.
            if buf:
                out.append(buf)
                buf = ""
            words = clause.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) > max_chars and current:
                    out.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                out.append(current)
            continue

        candidate = f"{buf} {clause}".strip() if buf else clause
        if len(candidate) > max_chars and buf:
            out.append(buf)
            buf = clause
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, max_chars: int = 250) -> list[Chunk]:
    """Split `text` into XTTS-friendly chunks.

    Args:
        text: Full input text.
        max_chars: Soft upper bound per chunk. Actual chunks may be shorter
            because we always stop at a sentence boundary.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = _PARAGRAPH_RE.split(text)
    chunks: list[Chunk] = []

    for p_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        sentences = split_sentences(para)
        if not sentences:
            continue

        # Pre-expand oversized sentences.
        expanded: list[str] = []
        for sent in sentences:
            expanded.extend(_split_long_sentence(sent, max_chars))

        # Greedy pack sentences into chunks under `max_chars`.
        packed: list[str] = []
        buf = ""
        for sent in expanded:
            if not buf:
                buf = sent
                continue
            candidate = f"{buf} {sent}"
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                packed.append(buf)
                buf = sent
        if buf:
            packed.append(buf)

        for i, piece in enumerate(packed):
            chunks.append(
                Chunk(
                    text=piece,
                    paragraph_index=p_idx,
                    is_paragraph_end=(i == len(packed) - 1),
                )
            )

    return chunks


def iter_chunks(text: str, max_chars: int = 250) -> Iterable[Chunk]:
    """Streaming wrapper — useful if you want to render as you go."""
    yield from chunk_text(text, max_chars=max_chars)
