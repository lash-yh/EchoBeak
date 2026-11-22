from pathlib import Path

import noisereduce as nr
import soundfile as sf

from errors import NoiseReductionError
from utils.paths import create_temp_audio_path


def reduce_noise(path_file: Path, temp_dir: Path | str) -> Path:
    """Reduce non-stationary noise and return a generated WAV path."""
    return _reduce_noise(path_file, temp_dir, stationary=False)


def reduce_noise_stationary(path_file: Path, temp_dir: Path | str) -> Path:
    """Reduce stationary noise and return a generated WAV path."""
    return _reduce_noise(path_file, temp_dir, stationary=True)


def _reduce_noise(
    path_file: Path, temp_dir: Path | str, *, stationary: bool,
) -> Path:
    output_path = create_temp_audio_path(temp_dir)
    try:
        data, rate = sf.read(path_file, dtype="float32")
        if data.size == 0:
            raise ValueError("Audio contains no samples")
        if data.ndim >= 2:
            data = data.mean(axis=1)
        reduced_noise = nr.reduce_noise(
            y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=stationary
        )
        sf.write(str(output_path), reduced_noise, rate)
    except (sf.SoundFileError, OSError, ValueError, RuntimeError, MemoryError) as error:
        try:
            output_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"Cannot remove partial audio {output_path}: {cleanup_error}")
        raise NoiseReductionError(f"Cannot reduce noise in {path_file}: {error}") from error
    return output_path
