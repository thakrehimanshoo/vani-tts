# Vani TTS

Local, expressive, long-form Text-to-Speech built around **XTTS v2** (Coqui TTS)
with optional voice cloning, emotion / style prompting, intelligent chunking,
seamless audio stitching, subtitle export, and an optional FastAPI wrapper.

Targeted at a single workstation with a consumer GPU (RTX 3050, 4 GB VRAM) and
designed to handle inputs from a few sentences up to a full hour of audio
(50k+ words) without OOM crashes or audible seams between chunks.

---

## Why XTTS v2?

| Model            | Quality | Speed (RTX 3050) | Voice clone | Multilingual | Long-form stability | Notes |
|------------------|---------|------------------|-------------|--------------|---------------------|-------|
| **XTTS v2**      | High    | ~0.3–0.5 x RT    | 6 s sample  | 17 languages | Excellent with chunking | Best overall tradeoff |
| Tortoise TTS     | Highest | ~5–10 x RT (slow)| Yes         | English only | Poor — drifts on long text | Too slow for 1 h audio |
| Bark             | High    | ~1–2 x RT        | Limited     | Multilingual | Poor — 14 s hard cap per gen | Needs aggressive stitching |
| Piper / VITS     | Medium  | ~0.05 x RT       | No          | Many         | Good                | Robotic prosody |

**XTTS v2 wins for this use case** because it is the only open model that
combines (a) fast enough inference to render an hour of audio in ~20–30 min on
a 3050, (b) genuine zero-shot voice cloning from a short sample, (c) expressive
prosody with pauses and intonation, and (d) deterministic conditioning — the
same speaker embedding can be reused across every chunk, which is what makes
seamless hour-long output possible.

Tortoise produces marginally better single-clip audio but is 10–20× slower.
Bark cannot sustain tone across chunks. Piper is fast but flat.

---

## Folder structure

```
vani-tts/
├── README.md
├── requirements.txt
├── setup.sh                   # one-shot installer (CUDA + deps)
├── tts.py                     # CLI entry point
├── .gitignore
├── vani_tts/
│   ├── __init__.py
│   ├── config.py              # defaults, presets, paths
│   ├── model_loader.py        # XTTS v2 load, fp16, CUDA
│   ├── chunker.py             # paragraph/sentence-aware splitter
│   ├── emotion.py             # emotion + pause + rate controls
│   ├── voice_clone.py         # reference-audio preprocessing
│   ├── synthesizer.py         # chunk → wav, streaming & batched
│   ├── audio_merger.py        # crossfade, normalize, export
│   ├── subtitles.py           # SRT/VTT generation
│   └── utils.py
├── api/
│   └── server.py              # FastAPI wrapper
├── examples/
│   ├── sample_input.txt
│   └── run_example.sh
└── tests/
    ├── test_chunker.py
    └── test_emotion.py
```

---

## Setup guide

### 1. System prerequisites

- NVIDIA driver ≥ 535
- CUDA 12.1 runtime (the PyTorch wheel bundles it — you only need the driver)
- Python 3.10 or 3.11 (Coqui TTS is not yet packaged for 3.12)
- ffmpeg (`sudo apt install ffmpeg`)
- ~6 GB free disk for model weights

### 2. Install

```bash
git clone https://github.com/thakrehimanshoo/vani-tts.git
cd vani-tts
bash setup.sh
```

`setup.sh` creates a venv, installs PyTorch with CUDA 12.1, installs Coqui TTS
and the rest of `requirements.txt`, and pre-downloads XTTS v2 weights.

### 3. First run (smoke test)

```bash
source .venv/bin/activate
python tts.py --input examples/sample_input.txt --output out.wav
```

### 4. Voice cloning

Record 10–30 s of clean speech (mono, 16 kHz+, no background music) and pass it:

```bash
python tts.py \
    --input book.txt \
    --output book.wav \
    --speaker-wav my_voice.wav \
    --emotion storytelling \
    --rate 0.95
```

### 5. Long-form (1 hour)

```bash
python tts.py \
    --input novel_chapter.txt \
    --output chapter.wav \
    --speaker-wav my_voice.wav \
    --emotion dramatic \
    --subtitles chapter.srt \
    --stream
```

`--stream` writes each chunk to disk as it is generated and only crossfades at
the end, so VRAM stays flat regardless of input length.

---

## CLI reference

```
python tts.py --input FILE --output FILE [options]

Required
  --input PATH             Path to a .txt file (UTF-8)
  --output PATH            Output .wav path

Voice
  --speaker-wav PATH       Reference clip (10–30 s) for voice cloning
  --language CODE          ISO code, default "en" (17 supported)

Expression
  --emotion NAME           neutral | happy | serious | storytelling | dramatic
  --rate FLOAT             Speaking rate multiplier (0.7–1.3, default 1.0)
  --pause-scale FLOAT      Pause length multiplier (default 1.0)
  --style-prompt TEXT      Free-form style conditioning (experimental)

Chunking / performance
  --max-chars INT          Chunk size in chars (default 250)
  --stream                 Write chunks to disk instead of keeping in RAM
  --batch-size INT         Number of chunks per GPU batch (default 1)
  --fp16                   Force fp16 inference (auto-enabled on 3050)

Output
  --sample-rate INT        Output SR (default 24000, XTTS native)
  --normalize              Apply LUFS loudness normalization (default on)
  --crossfade-ms INT       Crossfade between chunks (default 30)
  --subtitles PATH         Write .srt alongside wav
  --background PATH        Optional background music, mixed at -24 dB
```

---

## Architecture & dataflow

```
text file
   │
   ▼
┌───────────┐   paragraph + sentence aware
│  chunker  │   (keeps < max_chars, never splits mid-sentence)
└───────────┘
   │ List[str]
   ▼
┌────────────┐   emotion/rate/pause injected as SSML-like markers
│  emotion   │   and XTTS style tokens
└────────────┘
   │ List[StyledChunk]
   ▼
┌──────────────┐   XTTS v2 (fp16, CUDA)
│ synthesizer  │   - single speaker embedding reused for every chunk
│              │   - streaming or batched
└──────────────┘
   │ List[np.ndarray] | files on disk
   ▼
┌──────────────┐   equal-power crossfade, LUFS normalize,
│ audio_merger │   optional background ducking
└──────────────┘
   │
   ▼
   output.wav  (+ output.srt)
```

### Key design choices

1. **One speaker embedding per job.** The reference waveform (or the default
   speaker) is encoded *once* and reused for every chunk. This is what keeps
   tone consistent across a 1-hour render — the common failure mode in
   Bark/Tortoise is re-sampling the voice per chunk and drifting.
2. **Sentence-safe chunking.** We never cut mid-sentence. `--max-chars` is a
   soft upper bound; the chunker backs up to the nearest sentence boundary.
3. **Equal-power crossfade.** 30 ms cosine crossfade at each join kills clicks
   without audibly smearing consonants.
4. **Streaming mode.** For inputs >5k words, `--stream` writes each chunk as a
   temp WAV and merges with ffmpeg at the end, so peak RAM is ~1 chunk worth.
5. **fp16 on 3050.** XTTS v2 runs comfortably in ~2.5 GB VRAM at fp16, leaving
   headroom for the OS/desktop.

---

## Expressiveness controls

`--emotion` maps to a preset bundle of (temperature, repetition penalty,
length penalty, and a style prompt prepended to the text). Presets in
`vani_tts/emotion.py`:

| Emotion       | Temp | Rep. penalty | Style prompt                          |
|---------------|------|--------------|---------------------------------------|
| neutral       | 0.65 | 2.0          | —                                     |
| happy         | 0.80 | 2.0          | "spoken cheerfully and warmly"        |
| serious       | 0.55 | 2.5          | "spoken in a calm, measured tone"     |
| storytelling  | 0.75 | 2.0          | "narrated like a captivating story"   |
| dramatic      | 0.90 | 1.8          | "spoken with dramatic emphasis"       |

`--pause-scale` rewrites commas, periods, em-dashes and paragraph breaks into
explicit silence insertions (see `emotion.py::insert_pauses`).

---

## API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```
POST /synthesize
  form-data:
    text: string           (required if no text_file)
    text_file: upload      (required if no text)
    speaker_wav: upload    (optional)
    emotion: string        (optional, default "neutral")
    rate: float            (optional)
    language: string       (optional, default "en")
  → audio/wav stream
```

See `api/server.py` for the full schema.

---

## Troubleshooting

- **CUDA OOM** — lower `--max-chars` to 180 and use `--stream`.
- **Robotic prosody** — raise `--emotion storytelling` or `--rate 0.95`.
- **Clicks between chunks** — raise `--crossfade-ms` to 50.
- **Voice drift** — ensure your `--speaker-wav` is mono, 22 kHz+, 10–30 s,
  speech only.
- **License** — XTTS v2 is released under Coqui's non-commercial license.
  Check `https://coqui.ai/cpml` before commercial use.
