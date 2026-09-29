import numpy as np

from scripts.experiments.tts_native_gain_attribution.audio import (
    istft,
    noise_realization,
    periodic_hann,
    quantize_pcm16,
    stft,
)


def test_explicit_stft_istft_is_an_identity_on_the_registered_clock() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal(16_001).astype(np.float64)
    spectra, starts = stft(values)
    restored = istft(spectra, starts, values.size)
    assert np.max(np.abs(restored - values)) < 1e-8
    assert periodic_hann()[0] == 0.0


def test_pcm_quantization_uses_ties_to_even_without_clipping() -> None:
    values = np.asarray([-32768.5, -32767.5, 32766.5, 32767.0]) / 32768.0
    assert quantize_pcm16(values).tolist() == [-32768, -32768, 32766, 32767]


def test_colored_noise_seed_is_reproducible() -> None:
    power = np.ones(257, dtype=np.float64)
    first, first_meta = noise_realization(16_000, power, sample_id=151, source="N")
    second, second_meta = noise_realization(16_000, power, sample_id=151, source="N")
    assert np.array_equal(first, second)
    assert first_meta["seed_uint64_little_endian"] == second_meta["seed_uint64_little_endian"]
