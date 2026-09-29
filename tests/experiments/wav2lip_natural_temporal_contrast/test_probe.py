from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_natural_temporal_contrast.transform import chunk_mels, temporal_contrast


def test_constant_mel_is_identity_and_uses_time_axis() -> None:
    mel = np.ones((80, 308), dtype=np.float32)
    result = temporal_contrast(mel)
    assert result["amplitude"] == 0.0
    assert np.array_equal(result["N"], result["SMOOTH"])
    assert np.array_equal(result["N"], result["SHARP"])


def test_temporal_pulse_does_not_move_or_convolve_frequency() -> None:
    mel = np.zeros((80, 308), dtype=np.float32)
    mel[17, 100] = 1.0
    result = temporal_contrast(mel)
    assert np.allclose(result["SMOOTH"][17, 100], result["N"][17, 100], atol=0.5)
    assert np.all(result["SMOOTH"][16, 100] == 0.0)
    assert np.all(result["SMOOTH"][18, 100] == 0.0)
    assert np.array_equal(result["N"][:, :2], result["SMOOTH"][:, :2])


def test_reflect_edges_and_official_chunk_rule() -> None:
    mel = np.linspace(-1.0, 1.0, 80 * 308, dtype=np.float32).reshape(80, 308)
    result = temporal_contrast(mel)
    assert np.array_equal(result["N"][:, :2], result["SMOOTH"][:, :2])
    assert np.array_equal(result["N"][:, -2:], result["SHARP"][:, -2:])
    chunks = chunk_mels(mel)
    assert len(chunks) == 93
    assert np.array_equal(chunks[-1], mel[:, -16:])


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(Exception):
        temporal_contrast(np.zeros((308, 80), dtype=np.float32))
