# Repository Guidelines

## Project Overview

EchoBeak is a Python 3.13 local audio transcription application managed with `uv`.

The pipeline converts audio to WAV, reduces noise, transcribes it using `faster-whisper`, generates Markdown transcripts, and cleans up temporary files.

Work from the repository root containing `pyproject.toml` and `uv.lock`.

## Repository Structure

- `src/main.py`: CLI arguments and pipeline orchestration.
- `src/utils/paths.py`: Temporary audio paths and the default output directory.
- `src/utils/transcripts.py`: Transcript naming, destination directories, collisions, and publication.
- `src/processor/converter.py`: MIME validation, WAV conversion, and twenty-minute segmentation.
- `src/processor/chunks.py`: Audio chunks and cumulative timestamp offsets.
- `src/processor/cleaner.py`: Non-stationary (`reduce_noise`) and stationary (`reduce_noise_stationary`) noise reduction using `soundfile` and `noisereduce`.
- `src/processor/transcriber.py`: Chunk transcription and processed audio cleanup.
- `src/models/model.py` and `runtime.py`: Model lifetime, GPU detection, and CPU fallback.
- `tests/`: Pytest coverage with mocked model loading.

Inspect the actual source before assuming function signatures or module responsibilities.

## Development Environment

Use `uv` for dependencies and execution.

```bash
uv sync
uv run src/main.py /path/to/audio.wav transcript-name
```

`ffmpeg` must be available on `PATH`.

For CUDA/cuDNN execution, verify the required NVIDIA libraries and their compatibility with CTranslate2. Do not assume GPU detection guarantees successful model initialization.

Preserve CPU fallback behavior.

## Coding Conventions

- Follow PEP 8 and existing repository conventions.
- Use four-space indentation and `snake_case` naming.
- Prefer `pathlib.Path` for filesystem operations.
- Use type hints and explicit return types for public functions.
- Use `logging` for diagnostics.
- Keep functions focused and avoid unnecessary abstractions.
- Do not introduce dependencies without justification.
- Preserve existing module boundaries unless refactoring is explicitly required.

## Pipeline Constraints

- `convert_to_wav_16_mono(selected_audio_file, temp_dir, *, keep_wave=False)` returns a list of paths.
- WAV files already matching the required format may be returned unchanged.
- Preserve the existing 16 kHz mono 16-bit WAV format and twenty-minute segmentation behavior.
- Preserve transcription segment order and text when generating Markdown.
- `--noise-reduce [stationary|non-stationary]` selects the reduction mode; without a value it uses non-stationary reduction, and omission disables reduction. `transcribe_file()` and `transcript_lines()` accept the same strings via `noise_reduce`, retaining `True` for non-stationary and `False` for disabled.
- Default transcript directory: `~/EchoBeak-Transcription`. The CLI accepts `-o DIR` / `--output-dir DIR` for Markdown output; retained WAVs still use the default directory.
- `create_name_trans_path(name=None, *, output_dir=None)` creates the directory, expands `~`, and preserves default dated names and collision handling.
- Keep model initialization and CPU fallback in `src/models/model.py`, and GPU detection and cleanup in `src/models/runtime.py`.

### File Safety

- Never delete or overwrite user-selected source files.
- Distinguish application-generated temporary files from original input files.
- Only clean up files owned by the application.
- Never assume that a path returned by a conversion helper is temporary.
- Do not overwrite existing transcripts without explicit authorization.

## Known Issues

Input is provided through the CLI; there is no file picker. Do not assume a real model transcription works end-to-end until verified.

## Testing and Validation

Install dependencies:

```bash
uv sync
```

Check syntax:

```bash
uv run python -m compileall src
```

Run the configured automated suite with `uv run pytest`. No linter is currently configured.

When adding tests:

- Use `pytest` and `tests/test_*.py`.
- Mock external processes and model loading when appropriate.
- Avoid loading Whisper models or requiring GPU hardware in unit tests.
- Prioritize conversion, segmentation, cleanup, error handling, and filename collision tests.

After changes:

1. Run relevant available checks.
2. Verify modified function signatures and call sites.
3. Inspect the final diff.
4. Report failed or skipped validation honestly.

Syntax compilation does not establish end-to-end correctness.

## Agent Workflow

1. Read relevant files before making changes.
2. Understand affected functions and dependencies.
3. Plan significant changes before implementation.
4. Implement the smallest maintainable solution.
5. Avoid unrelated modifications or speculative refactoring.
6. Validate affected behavior when possible.
7. Review the final diff.
8. Summarize changes, verification results, and unresolved issues.

### Restrictions

- Preserve existing user changes.
- Do not modify unrelated files.
- Do not silently change public APIs.
- Do not remove functionality without authorization.
- Do not perform destructive operations without approval.
- Do not create Git commits unless requested.
- Do not modify Git history without authorization.
- Never claim successful validation without executing the relevant checks.

## Documentation Maintenance

Keep `AGENTS.md` synchronized with the actual repository.

Update it when changes affect:

- Repository structure or module responsibilities.
- Important function signatures or architectural boundaries.
- Development tools and validation commands.
- Pipeline constraints or known issues.

When updating:

- Verify statements against current source code.
- Remove obsolete information and resolved issues.
- Keep instructions concise and actionable.
- Avoid duplicating `README.md` or implementation details.
- Do not update this file for trivial changes.

Review `README.md` and `CLAUDE.md` when changes affect their documented behavior.

## Git Conventions

Follow existing commit prefixes such as `feat:`, `fix:`, `docs:`, and `refactor:`.

Keep changes scoped.

Do not commit generated caches, virtual environments, temporary audio files, or transcripts.

Create commits only when explicitly requested.
