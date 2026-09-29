from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_global_shift_response import config
from scripts.experiments.wav2lip_global_shift_response.audio import assert_shift_contract, shift_pcm
from scripts.experiments.wav2lip_global_shift_response.metrics import endpoint


def _pcm(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


def test_exact_shifts_preserve_length_and_do_not_wrap() -> None:
    natural = _pcm(np.arange(20, dtype=np.int16))
    delay = np.frombuffer(shift_pcm(natural, config.AUDIO_DELAY, 4), dtype="<i2")
    advance = np.frombuffer(shift_pcm(natural, config.AUDIO_ADVANCE, 4), dtype="<i2")
    assert delay.tolist() == [0, 0, 0, 0, *range(16)]
    assert advance.tolist() == [*range(4, 20), 0, 0, 0, 0]
    assert len(delay) == len(advance) == 20
    assert_shift_contract(natural, _pcm(delay), _pcm(advance), 4)


def test_shift_rejects_unknown_direction() -> None:
    with pytest.raises(Exception):
        shift_pcm(_pcm(np.arange(10, dtype=np.int16)), "ROLL", 2)


def test_endpoint_averages_rows_before_c_and_d() -> None:
    matrix = np.ones((2, config.MATRIX_COLUMNS), dtype=np.float32)
    matrix[0, 15] = 0.0
    matrix[1, 16] = 0.0
    result = endpoint(matrix, [0, 1])
    assert result["min_index"] == 15
    assert result["offset"] == 0
    assert result["D"] == pytest.approx(0.5)


def test_registered_counts_are_fixed() -> None:
    assert config.EXPECTED_RECORD_COUNT == 12
    assert config.EXPECTED_VIDEO_COUNT == 96
    assert config.EXPECTED_SCORE_COUNT == 192
