# EchoBeak

Transcribe one audio or video file locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The pipeline checks the MIME type, converts the file to 16 kHz mono 16-bit WAV, splits audio longer than twenty minutes, optionally applies stationary or non-stationary noise reduction, and writes a timestamped Markdown transcript to `~/EchoBeak-Transcription` by default. Use `-o DIR` or `--output-dir DIR` to choose another transcript directory.

## Pipeline

```
input path + transcript name
  → MIME check
  → ffmpeg (WAV 16 kHz mono s16), or the file itself when it already matches
  → twenty-minute split when the audio is longer than 1200 seconds
  → optional noise reduction (--noise-reduce)
  → faster-whisper
  → <output-dir>/<name>.md (default: ~/EchoBeak-Transcription)
```

1. **Input** — the command takes one audio or video path and a transcript file name. Before loading the model or processing the file, it checks that `ffmpeg` is available on `PATH` and stops with an installation message if it is missing.
2. **Conversion** — `python-magic` requires an `audio/*` or `video/*` MIME type. A WAV that is already 16 kHz, mono, and 16-bit skips conversion. Every other accepted file is converted with `ffmpeg` (`-vn -ac 1 -ar 16000 -sample_fmt s16`).
3. **Splitting** — audio longer than twenty minutes is cut into consecutive twenty-minute WAV parts. Timestamps in the transcript stay continuous across parts.
4. **Noise reduction** — disabled by default. Use `--noise-reduce stationary` or `--noise-reduce non-stationary` to choose the mode for every part. `--noise-reduce` without a value uses non-stationary reduction. The result is written to a temporary WAV.
5. **Transcription** — Whisper runs with beam size 5. Use `--language es` to specify the audio language for every chunk. When omitted, the model detects the language (`language=None`). `transcribe_file()` also accepts a language code when called from Python. The model is loaded before conversion to validate explicit codes against `model.supported_languages`. Unsupported codes stop the run with status 1 and a message listing the supported codes, before conversion or noise reduction. Language rejections from the library are also reported without a traceback.
6. **Publish** — segments are written to a temporary draft and copied to the selected output directory (default: `~/EchoBeak-Transcription`) only after the whole file succeeds. Each segment is one line: `[start -> end] text`.

By default, each processed temporary WAV is deleted with `Path.unlink()`, and remaining temporary files are removed when the run ends. With `--keep-wave`, generated WAVs (conversion, split parts, and optional noise reduction) are saved in `~/EchoBeak-Transcription` with unique names and kept even if transcription fails. Incomplete files from failed conversion or noise reduction are still removed. The temporary Markdown draft is always removed. The source file stays in place.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- the system `libmagic` library used by `python-magic`

## Setup

```bash
uv sync
```

For a CUDA run, expose the cuBLAS and cuDNN libraries shipped in the virtual environment:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

## Usage

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting
```

To specify Spanish instead of automatic language detection:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --language es
```

To choose the transcript directory:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --output-dir ~/Documents/transcripts
```

To choose the noise reduction mode:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --noise-reduce stationary
uv run src/main.py ~/recordings/meeting.mp4 meeting --noise-reduce non-stationary
```

`--noise-reduce` without a value still enables non-stationary reduction. Omit the
option to disable reduction. Place this option after the input path and transcript
name when using it without a value.

For stationary noise reduction from Python, use `reduce_noise_stationary()` from
`processor.cleaner` (with `src` on `PYTHONPATH`):

```python
from pathlib import Path
from processor.cleaner import reduce_noise_stationary

cleaned_path = reduce_noise_stationary(Path("recording.wav"), Path("/tmp"))
```

The function returns the path of a generated mono WAV and preserves the source
file. `transcribe_file()` and `transcript_lines()` also accept
`noise_reduce="stationary"` or `noise_reduce="non-stationary"`; `True` still selects
non-stationary reduction and `False` disables it.

To keep generated WAV files:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --keep-wave
```

Arguments:

| Argument | Meaning |
|---|---|
| `input_path` | Path to one audio or video file |
| `-o DIR`, `--output-dir DIR` | Transcript directory (default: `~/EchoBeak-Transcription`). Accepts relative paths and `~`; missing directories are created. This option does not change where `--keep-wave` saves WAV files. |
| `--keep-wave` | Keep generated WAV files in `~/EchoBeak-Transcription` (deleted by default). Can be combined with `--noise-reduce`. |
| `--noise-reduce [MODE]` | Choose `stationary` or `non-stationary`. Without a value, use `non-stationary`; omitted, disable reduction. |
| `--language CODE` | Audio language code, such as `es` (Spanish) or `en` (English). Omit for automatic detection. This selects the transcription language; it does not translate the audio. |
| `name` | Transcript file name. It must be a single file name: empty names, `.`, `..`, and names containing `/` or `\` are rejected. `meeting` and `meeting.md` are both saved as `meeting.md` |

The transcript is saved as `<output-dir>/meeting.md`, using `~/EchoBeak-Transcription` when no directory is specified. The directory is created when needed. When that name is already taken, the new file is `meeting-(1).md`, then `meeting-(2).md`, and so on. A destination that is a file or cannot be created stops the run before loading the model.

Logs go to stderr. The Markdown file holds the transcript:

```
2026-10-20 10:00:00,000 [INFO] Loading model in CPU with INT8
2026-10-20 10:00:01,000 [INFO] Model loaded successfully: small (cpu, int8)
2026-10-20 10:00:01,100 [INFO] Transcribing: temp-3f2a1b9c-4d5e-6789-abcd-ef0123456789.wav
2026-10-20 10:00:12,000 [INFO] Transcription saved to /home/user/EchoBeak-Transcription/meeting.md
```

```
[0.00s -> 3.20s] Hola, esto es una prueba de transcripción.
[3.20s -> 6.50s] El sistema funciona correctamente.
```

A failed conversion, noise-reduction, or transcription run logs the exception and exits with status 1. A previous transcript at the destination is left as it was.

Run the command once per file to transcribe several recordings.

## Project structure

```
EchoBeak/
├── src/
│   ├── cli/                    # empty
│   ├── errors.py               # Conversion, noise-reduction, transcription, and language errors
│   ├── main.py                 # CLI and pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py            # Model lifetime, model selection, GPU loading fallback
│   │   └── runtime.py          # CUDA detection and memory cleanup
│   ├── processor/
│   │   ├── chunks.py           # Audio chunks and cumulative timestamp offsets
│   │   ├── cleaner.py          # Stationary and non-stationary noise reduction
│   │   ├── converter.py        # MIME check, ffmpeg conversion, twenty-minute splitting
│   │   └── transcriber.py      # Chunk transcription, timestamps, processed audio cleanup
│   └── utils/
│       ├── paths.py            # Generated WAV paths and shared output directory
│       └── transcripts.py      # Transcript names, collisions, and publication
├── tests/
│   ├── conftest.py
│   ├── test_chunks.py
│   ├── test_cleaner.py
│   ├── test_converter.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_runtime.py
│   ├── test_transcriber.py
│   └── test_transcripts.py
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## Model selection

`load_whisper_model()` in `src/models/model.py` chooses the model at runtime:

- When CTranslate2 reports a CUDA device, it loads `medium` on `cuda` with `float16`.
- When that load fails, when no CUDA device is present, or when the device check itself fails, it loads `small` on `cpu` with `int8`.

An error while transcribing stops the run. The CPU model is the fallback for loading, and the GPU model is unloaded with a CUDA cache clear when it was loaded on `cuda`.

## Tests

`pytest` is included in the project dependencies and configured in `pyproject.toml` (`pythonpath = ["src"]`, `testpaths = ["tests"]`):

```bash
uv run pytest
uv run python -m compileall src
```

The automated suite mocks model loading and does not require a GPU or downloaded Whisper models. Syntax compilation alone does not verify an end-to-end transcription.

## Dependencies

| Package | Purpose |
|---|---|
| `faster-whisper` | CTranslate2-based Whisper inference |
| `ctranslate2` | CUDA device detection |
| `noisereduce` | Stationary and non-stationary noise reduction |
| `soundfile` | WAV read/write for noise reduction |
| `python-magic` | MIME type detection |
| `pytest` | Automated tests |
| `torch` | CUDA memory cleanup after a GPU model unloads |
| `nvidia-cublas-cu12` | Bundled cuBLAS for CUDA runs |
| `nvidia-cudnn-cu12` | Bundled cuDNN for CUDA runs |
