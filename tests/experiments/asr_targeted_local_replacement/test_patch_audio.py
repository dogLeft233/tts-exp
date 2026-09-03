from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.asr_targeted_local_replacement.patch_audio import (
    apply_patches,
    build_condition_audio,
    float_to_pcm16,
    write_pcm16,
)


def _patch(start: int, end: int, donor_start: int = 0, donor_end: int = 8) -> dict[str, int | str]:
    return {
        "block_id": f"block_{start}",
        "destination_start_sample": start,
        "destination_end_sample": end,
        "donor_start_sample": donor_start,
        "donor_end_sample": donor_end,
        "destination_samples": end - start,
        "donor_samples": donor_end - donor_start,
    }


def test_exact_length_and_outside_identity():
    natural = np.zeros(1000, dtype=np.int16)
    tts = np.full(1000, 1000, dtype=np.int16)
    result, meta = apply_patches(natural, tts, [_patch(100, 500)])
    assert result.shape == (1000,)
    assert np.array_equal(result[:100], natural[:100] / 32768.0)
    assert np.array_equal(result[500:], natural[500:] / 32768.0)
    assert meta["patches"][0]["ramp_samples"] == 100


def test_multiple_patches_must_be_disjoint():
    natural = np.zeros(100, dtype=np.int16)
    tts = np.ones(100, dtype=np.int16)
    with pytest.raises(ValueError, match="overlap"):
        apply_patches(natural, tts, [_patch(10, 30), _patch(29, 40)])


def test_clipping_and_nonfinite_fail_without_fallback():
    natural = np.zeros(20, dtype=np.float64)
    tts = np.full(20, 2.0)
    with pytest.raises(ValueError, match="clips"):
        apply_patches(natural, tts, [_patch(2, 10)])
    with pytest.raises(ValueError, match="finite"):
        apply_patches(natural, np.full(20, np.nan), [_patch(2, 10)])


def test_natural_condition_is_identity():
    natural = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    result, meta = build_condition_audio(natural, natural, "natural", [])
    assert meta["identity"] is True
    assert np.array_equal(float_to_pcm16(result), natural)


def test_writer_preserves_pcm16_identity(tmp_path):
    natural = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    path = tmp_path / "natural.wav"
    write_pcm16(path, natural)
    import soundfile as sf
    decoded, _ = sf.read(path, dtype="int16", always_2d=True)
    assert decoded[:, 0].tobytes() == natural.tobytes()
