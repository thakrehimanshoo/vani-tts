"""SRT subtitle generation from synthesized chunk timings."""

from __future__ import annotations

from pathlib import Path

from vani_tts.synthesizer import SynthesizedChunk


def _fmt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    # Carry over when ms rounds up to 1000.
    if ms == 1000:
        ms = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(chunks: list[SynthesizedChunk], path: Path) -> Path:
    """Write an SRT file using the start/end timestamps recorded during synthesis.

    Each chunk becomes one cue, showing the *original* text (before emotion
    framing was prepended).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_timestamp(ch.start_s)} --> {_fmt_timestamp(ch.end_s)}")
        # SRT convention: wrap long cues at ~42 chars per line, max 2 lines.
        lines.append(_wrap_caption(ch.original_text))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _wrap_caption(text: str, max_line: int = 42) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_line:
        return text
    words = text.split()
    line1: list[str] = []
    line2: list[str] = []
    current = line1
    for w in words:
        candidate = (" ".join(current + [w])).strip()
        if len(candidate) > max_line and current is line1:
            current = line2
            current.append(w)
        else:
            current.append(w)
    return " ".join(line1) + "\n" + " ".join(line2)
