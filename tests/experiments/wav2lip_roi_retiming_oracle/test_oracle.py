from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_roi_retiming_oracle import config
from scripts.experiments.wav2lip_roi_retiming_oracle.analysis import peak, summarize
from scripts.experiments.wav2lip_roi_retiming_oracle.common import OracleError
from scripts.experiments.wav2lip_roi_retiming_oracle.media import (
    build_oracle_indices,
    nearest_half_up,
)
from scripts.experiments.wav2lip_roi_retiming_oracle.validate import (
    _distance_from_embeddings,
    validate_stored_arrays,
)


def test_nearest_half_up_and_oracle_indices_are_monotone() -> None:
    assert nearest_half_up(np.array([0.0, 0.49, 0.5, 1.5, 2.500001])).tolist() == [0, 0, 1, 2, 3]
    indices = build_oracle_indices(96_256, 147)
    q = np.asarray(indices["q_oracle"], dtype=np.int64)
    assert np.all(np.diff(q) >= 0)
    assert int(q.min()) >= 0
    assert int(q.max()) < 147
    assert indices["max_abs_quantization_error"] <= 0.5 + 1e-9


def test_half_up_rejects_negative_or_nonfinite_coordinates() -> None:
    with pytest.raises(OracleError, match="finite"):
        nearest_half_up(np.array([-0.1]))
    with pytest.raises(OracleError, match="finite"):
        nearest_half_up(np.array([np.nan]))


def test_peak_gap_and_boundary_are_strict() -> None:
    curve = np.ones(31, dtype=np.float64)
    curve[15] = 0.0
    curve[14] = config.PEAK_GAP_THRESHOLD
    assert peak(curve)["clear"] is False
    curve[14] = config.PEAK_GAP_THRESHOLD + 1e-6
    assert peak(curve)["clear"] is True
    curve[0] = -1.0
    assert peak(curve)["offset"] == config.VSHIFT
    assert peak(curve)["clear"] is False


def test_offset_signs_match_oracle_protocol() -> None:
    def matrix(min_index: int) -> np.ndarray:
        value = np.ones((8, 31), dtype=np.float64)
        value[:, min_index] = 0.0
        return value

    masks = {"common_window_rows": list(range(8)), "plus_rows": [1, 2], "minus_rows": [5, 6]}
    natural = summarize(matrix(15), masks["plus_rows"])
    warped = summarize(matrix(12), masks["plus_rows"])
    oracle_natural = summarize(matrix(18), masks["plus_rows"])
    oracle_warped = summarize(matrix(15), masks["plus_rows"])
    assert warped["offset"] - natural["offset"] == 3
    assert oracle_natural["offset"] - natural["offset"] == -3
    assert oracle_warped["offset"] - natural["offset"] == 0


def test_validator_recomputes_distance_and_rejects_tampering() -> None:
    visual = np.zeros((2, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    visual[1, 0] = 1.0
    expected = _distance_from_embeddings(visual, audio)
    validate_stored_arrays(visual, audio, expected)
    tampered = expected.copy()
    tampered[0, 0] += 0.01
    with pytest.raises(OracleError, match="stored matrix"):
        validate_stored_arrays(visual, audio, tampered)
