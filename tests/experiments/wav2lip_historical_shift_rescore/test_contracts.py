from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_historical_shift_rescore import analysis
from scripts.experiments.wav2lip_historical_shift_rescore.common import RescoreError


def test_endpoint_averages_rows_before_finding_curve_statistics() -> None:
    matrix = np.full((2, 31), 10.0, dtype=np.float32)
    matrix[0, 15] = 0.0
    matrix[1, 16] = 0.0
    result = analysis.endpoint(matrix, [0, 1])
    assert result["min_index"] == 15
    assert result["D"] == pytest.approx(5.0)
    assert result["C"] == pytest.approx(5.0)


def test_benefit_d_direction_is_natural_minus_shifted() -> None:
    natural = {"C": 2.0, "D": 5.0, "D_anchor": 5.5, "M": 7.0, "offset": 0}
    shifted = {"C": 3.0, "D": 4.0, "D_anchor": 4.5, "M": 7.0, "offset": 0}
    value = analysis.benefits(natural, shifted)
    assert value["benefit_C"] == pytest.approx(1.0)
    assert value["benefit_D"] == pytest.approx(1.0)
    assert value["benefit_anchor"] == pytest.approx(1.0)


def test_shared_anchor_does_not_follow_shifted_free_offset() -> None:
    matrix = np.full((1, 31), 10.0, dtype=np.float32)
    matrix[0, 15] = 1.0
    natural = analysis.endpoint(matrix, [0])
    shifted_matrix = matrix.copy()
    shifted_matrix[0, 10] = 0.0
    shifted = analysis.endpoint(shifted_matrix, [0], int(natural["offset"]))
    assert shifted["offset"] == 5
    assert shifted["D_anchor"] == pytest.approx(1.0)


def test_endpoint_rejects_empty_or_out_of_range_support() -> None:
    matrix = np.zeros((3, 31), dtype=np.float32)
    with pytest.raises(RescoreError):
        analysis.endpoint(matrix, [])
    with pytest.raises(RescoreError):
        analysis.endpoint(matrix, [3])
