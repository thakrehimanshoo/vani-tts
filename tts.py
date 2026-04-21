#!/usr/bin/env python3
"""Vani TTS command-line interface.

Usage:
    python tts.py --input FILE --output FILE [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vani_tts.audio_merger import merge
from vani_tts.config import Config, EMOTIONS
from vani_tts.subtitles import write_srt
from vani_tts.synthesizer import build_synthesizer
from vani_tts.utils import get_logger, temp_dir
from vani_tts.voice_clone import preprocess_reference

_LOG = get_logger("vani_tts.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local expressive TTS built on XTTS v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Input .txt file")
    p.add_argument("--output", required=True, type=Path, help="Output .wav file")

    # Voice
    p.add_argument("--speaker-wav", type=Path, default=None,
                   help="Reference clip (10–30 s) for voice cloning")
    p.add_argument("--language", default="en",
                   help="ISO language code (XTTS supports 17)")

    # Expression
    p.add_argument("--emotion", default="neutral", choices=sorted(EMOTIONS),
                   help="Expressive preset")
    p.add_argument("--rate", type=float, default=1.0,
                   help="Speaking rate multiplier [0.5, 1.5]")
    p.add_argument("--pause-scale", type=float, default=1.0,
                   help="Pause length multiplier [0.5, 3.0]")
    p.add_argument("--style-prompt", default=None,
                   help="Free-form style prompt; overrides the emotion preset")

    # Chunking / performance
    p.add_argument("--max-chars", type=int, default=250,
                   help="Soft upper bound per chunk")
    p.add_argument("--stream", action="store_true",
                   help="Write chunks to disk as they render (low RAM)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Reserved for future batched inference")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable fp16 inference (fp16 is on by default on CUDA)")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                   help="Inference device")

    # Output
    p.add_argument("--sample-rate", type=int, default=24_000,
                   help="Output sample rate")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip LUFS normalization")
    p.add_argument("--crossfade-ms", type=int, default=30,
                   help="Crossfade between chunks in milliseconds")
    p.add_argument("--subtitles", type=Path, default=None,
                   help="Write an .srt alongside the audio")
    p.add_argument("--background", type=Path, default=None,
                   help="Optional background music (WAV/MP3)")
    p.add_argument("--background-gain-db", type=float, default=-24.0,
                   help="Background gain relative to speech")

    # Voice cleanup
    p.add_argument("--skip-reference-cleanup", action="store_true",
                   help="Use --speaker-wav as-is without trim/resample")

    return p


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        input_path=args.input,
        output_path=args.output,
        speaker_wav=args.speaker_wav,
        language=args.language,
        emotion=args.emotion,
        rate=args.rate,
        pause_scale=args.pause_scale,
        style_prompt=args.style_prompt,
        max_chars=args.max_chars,
        stream=args.stream,
        batch_size=args.batch_size,
        fp16=(not args.no_fp16),
        device=args.device,
        sample_rate=args.sample_rate,
        normalize=(not args.no_normalize),
        crossfade_ms=args.crossfade_ms,
        subtitles_path=args.subtitles,
        background_path=args.background,
        background_gain_db=args.background_gain_db,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    config.validate()

    # Clean the speaker reference if present.
    if config.speaker_wav is not None and not args.skip_reference_cleanup:
        try:
            config.speaker_wav = preprocess_reference(config.speaker_wav)
        except Exception as e:
            _LOG.error("Reference audio preprocessing failed: %s", e)
            return 2

    synth = build_synthesizer(config)

    with temp_dir() as work_dir:
        chunks = synth.run(work_dir=work_dir if config.stream else None)
        merge(
            chunks=chunks,
            output_path=config.output_path,
            sample_rate=config.sample_rate,
            crossfade_ms=config.crossfade_ms,
            normalize=config.normalize,
            background_path=config.background_path,
            background_gain_db=config.background_gain_db,
        )

        if config.subtitles_path is not None:
            write_srt(chunks, config.subtitles_path)
            _LOG.info("Wrote subtitles to %s", config.subtitles_path)

    _LOG.info("Done: %s", config.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
