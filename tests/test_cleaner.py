from unittest.mock import Mock

import numpy as np
import pytest

from errors import NoiseReductionError
from processor import cleaner


@pytest.mark.parametrize("reducer", [cleaner.reduce_noise, cleaner.reduce_noise_stationary])
def test_noise_write_failure_preserves_cause_and_removes_partial(tmp_path, monkeypatch, reducer):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    output = tmp_path / "cleaned.wav"
    failure = OSError("disk full")
    monkeypatch.setattr(cleaner, "create_temp_audio_path", lambda directory: output)
    monkeypatch.setattr(cleaner.sf, "read", lambda *a, **k: (np.zeros(160), 16000))
    monkeypatch.setattr(cleaner.nr, "reduce_noise", lambda **kwargs: kwargs["y"])

    def write(*args):
        output.write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(cleaner.sf, "write", write)
    with pytest.raises(NoiseReductionError) as caught:
        reducer(source, tmp_path)
    assert caught.value.__cause__ is failure
    assert not output.exists()
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("reducer", [cleaner.reduce_noise, cleaner.reduce_noise_stationary])
def test_noise_read_failure_preserves_cause(tmp_path, monkeypatch, reducer):
    failure = OSError("cannot read")
    monkeypatch.setattr(cleaner.sf, "read", Mock(side_effect=failure))
    with pytest.raises(NoiseReductionError) as caught:
        reducer(tmp_path / "source.wav", tmp_path)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("reducer,stationary", [
    (cleaner.reduce_noise, False),
    (cleaner.reduce_noise_stationary, True),
])
@pytest.mark.parametrize("stereo", [False, True])
def test_noise_modes_write_mono_wav_and_preserve_source(
    tmp_path, monkeypatch, reducer, stationary, stereo,
):
    source = tmp_path / "source.wav"
    data = np.full((160, 2) if stereo else 160, 0.25, dtype=np.float32)
    cleaner.sf.write(source, data, 16000, subtype="PCM_16")
    original = source.read_bytes()
    expected = data.mean(axis=1) if stereo else data
    reduce = Mock(return_value=expected / 2)
    monkeypatch.setattr(cleaner.nr, "reduce_noise", reduce)

    output = reducer(source, tmp_path)

    reduce.assert_called_once()
    np.testing.assert_array_equal(reduce.call_args.kwargs["y"], expected)
    assert reduce.call_args.kwargs["sr"] == 16000
    assert reduce.call_args.kwargs["stationary"] is stationary
    if not stationary:
        assert reduce.call_args.kwargs["thresh_n_mult_nonstationary"] == 2
    result, rate = cleaner.sf.read(output)
    np.testing.assert_allclose(result, expected / 2)
    assert rate == 16000
    assert result.ndim == 1
    assert cleaner.sf.info(output).subtype == "PCM_16"
    assert output != source
    assert source.read_bytes() == original


@pytest.mark.parametrize("reducer", [cleaner.reduce_noise, cleaner.reduce_noise_stationary])
def test_empty_audio_is_rejected_before_noise_reduction(tmp_path, monkeypatch, reducer):
    reduce = Mock()
    monkeypatch.setattr(cleaner.sf, "read", lambda *a, **k: (np.array([]), 16000))
    monkeypatch.setattr(cleaner.nr, "reduce_noise", reduce)

    with pytest.raises(NoiseReductionError, match="Audio contains no samples"):
        reducer(tmp_path / "source.wav", tmp_path)

    reduce.assert_not_called()
