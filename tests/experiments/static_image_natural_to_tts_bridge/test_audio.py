import numpy as np

from scripts.experiments.static_image_bridge.audio import (
    delayed_audio,
    quarter_swap,
    stft_roundtrip_alpha0,
)


def test_quarter_swap_is_the_registered_a_c_b_d_permutation() -> None:
    values = np.arange(4096, dtype=np.int16)
    result, meta = quarter_swap(values)
    assert meta["order"] == [0, 2, 1, 3]
    quarter = values.size // 4
    expected = np.concatenate((values[:quarter], values[2 * quarter:3 * quarter], values[quarter:2 * quarter], values[3 * quarter:])).tolist()
    assert result.tolist() == expected


def test_delayed_audio_preserves_length_and_moves_the_boundary() -> None:
    values = np.arange(5000, dtype=np.int16)
    result, meta = delayed_audio(values, delay_samples=3200)
    assert result.shape == values.shape
    assert result[:3200].tolist() == [0] * 3200
    assert result[3200:].tolist() == values[:-3200].tolist()
    assert meta["delay_frames"] == 5


def test_alpha_zero_stft_roundtrip_is_pcm_identity_on_a_bounded_signal() -> None:
    t = np.arange(16_000, dtype=np.float64) / 16_000.0
    values = np.rint(0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    result, meta = stft_roundtrip_alpha0(values)
    assert result.shape == values.shape
    assert int(np.max(np.abs(result.astype(np.int32) - values.astype(np.int32)))) <= 1
    assert meta["max_abs_pcm_error"] <= 1
