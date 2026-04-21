"""FastAPI wrapper around the Vani TTS pipeline.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

The model is loaded lazily on the first request and kept resident in memory
thereafter — there is only ever one XTTS engine per process.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from vani_tts.audio_merger import merge
from vani_tts.config import Config, EMOTIONS
from vani_tts.synthesizer import build_synthesizer
from vani_tts.utils import get_logger
from vani_tts.voice_clone import preprocess_reference

_LOG = get_logger("vani_tts.api")

app = FastAPI(
    title="Vani TTS",
    version="0.1.0",
    description="Local expressive TTS (XTTS v2).",
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/emotions")
def list_emotions() -> dict:
    return {"emotions": sorted(EMOTIONS)}


@app.post("/synthesize")
async def synthesize(
    text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    speaker_wav: Optional[UploadFile] = File(None),
    emotion: str = Form("neutral"),
    rate: float = Form(1.0),
    pause_scale: float = Form(1.0),
    language: str = Form("en"),
    max_chars: int = Form(250),
    stream: bool = Form(True),
):
    """Render text to a WAV and return it.

    Either `text` or `text_file` must be supplied. `speaker_wav` is optional —
    when omitted the default XTTS studio voice is used.
    """
    if not text and not text_file:
        raise HTTPException(400, "Provide `text` or `text_file`.")
    if emotion not in EMOTIONS:
        raise HTTPException(400, f"Unknown emotion. Choices: {sorted(EMOTIONS)}")

    job_id = uuid.uuid4().hex[:12]
    work_dir = Path(tempfile.mkdtemp(prefix=f"vani_{job_id}_"))

    try:
        # Materialize inputs on disk — keeps the pipeline the same shape as CLI.
        if text_file is not None:
            input_path = work_dir / "input.txt"
            with input_path.open("wb") as fh:
                shutil.copyfileobj(text_file.file, fh)
        else:
            input_path = work_dir / "input.txt"
            input_path.write_text(text or "", encoding="utf-8")

        speaker_path: Optional[Path] = None
        if speaker_wav is not None:
            speaker_path = work_dir / "speaker.wav"
            with speaker_path.open("wb") as fh:
                shutil.copyfileobj(speaker_wav.file, fh)
            try:
                speaker_path = preprocess_reference(speaker_path)
            except Exception as e:
                raise HTTPException(400, f"Bad speaker_wav: {e}") from e

        output_path = work_dir / "output.wav"
        config = Config(
            input_path=input_path,
            output_path=output_path,
            speaker_wav=speaker_path,
            language=language,
            emotion=emotion,
            rate=rate,
            pause_scale=pause_scale,
            max_chars=max_chars,
            stream=stream,
        )
        config.validate()

        synth = build_synthesizer(config)
        chunks = synth.run(work_dir=work_dir if config.stream else None)
        merge(
            chunks=chunks,
            output_path=output_path,
            sample_rate=config.sample_rate,
            crossfade_ms=config.crossfade_ms,
            normalize=config.normalize,
        )

        # FileResponse streams the file, then a background task can clean up.
        # We detach the temp dir by copying to a tmpfile FastAPI owns.
        final = Path(tempfile.NamedTemporaryFile(
            prefix=f"vani_{job_id}_", suffix=".wav", delete=False
        ).name)
        shutil.copyfile(output_path, final)
        return FileResponse(
            final,
            media_type="audio/wav",
            filename=f"{job_id}.wav",
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
