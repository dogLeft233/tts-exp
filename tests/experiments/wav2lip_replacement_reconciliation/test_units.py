import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.wav2lip_replacement_reconciliation import config
from scripts.experiments.wav2lip_replacement_reconciliation.analysis import (
    _decomposition,
    bootstrap_summary,
    make_bootstrap_indices,
    summarize_matrix,
)
from scripts.experiments.wav2lip_replacement_reconciliation.common import (
    ReconciliationError,
)
from scripts.experiments.wav2lip_replacement_reconciliation.protocol import (
    _cell_index,
    _diff_pcm,
)


def test_summarize_matrix_means_rows_before_metrics_and_breaks_ties_by_column() -> None:
    matrix = np.full((2, config.MATRIX_COLUMNS), 5.0, dtype=np.float32)
    matrix[:, 0] = [1.0, 3.0]
    matrix[:, 1] = [2.0, 2.0]

    summary = summarize_matrix(matrix, [0, 1])

    assert summary["curve"][0] == pytest.approx(2.0)
    assert summary["curve"][1] == pytest.approx(2.0)
    assert summary["min_index"] == 0
    assert summary["sync_d"] == pytest.approx(2.0)
    assert summary["sync_c"] == pytest.approx(3.0)
    assert summary["offset"] == 15


def test_pcm_diff_uses_int32_before_subtraction() -> None:
    left = np.asarray([-32768, 32767], dtype="<i2").tobytes()
    right = np.asarray([32767, -32768], dtype="<i2").tobytes()

    diff = _diff_pcm(left, right)

    assert diff["pcm_equal"] is False
    assert diff["max_abs_diff_lsb"] == 65535
    assert diff["rms_diff_lsb"] == pytest.approx(65535.0)


def test_keyed_cell_index_is_order_independent_and_rejects_duplicates() -> None:
    rows = [
        {"sample_id": "b", "cell": "V_N/A_N"},
        {"sample_id": "a", "cell": "V_N/A_N"},
    ]
    indexed = _cell_index(rows, ("sample_id", "cell"))
    assert indexed[("a", "V_N/A_N")]["sample_id"] == "a"
    assert indexed[("b", "V_N/A_N")]["sample_id"] == "b"

    with pytest.raises(ReconciliationError, match="duplicate or incomplete cell"):
        _cell_index([rows[0], rows[0]], ("sample_id", "cell"))


def test_bootstrap_indices_are_shared_and_deterministic() -> None:
    protocol = {"records": [{"source_group": f"g{i:02d}"} for i in range(22)]}

    labels_a, indices_a = make_bootstrap_indices(protocol)
    labels_b, indices_b = make_bootstrap_indices(protocol)

    assert labels_a == labels_b == [f"g{i:02d}" for i in range(22)]
    assert indices_a.shape == (config.BOOTSTRAP_DRAWS, 22)
    np.testing.assert_array_equal(indices_a, indices_b)
    result = bootstrap_summary(np.arange(22, dtype=float), labels_a, labels_a, indices_a)
    assert result["mean"] == pytest.approx(10.5)
    assert result["draws"] == config.BOOTSTRAP_DRAWS


def test_decomposition_identity_is_exact_for_synthetic_endpoint_rows() -> None:
    sid = "sample"
    record = {"sample_id": sid, "source_group": "group"}
    endpoint_index = {}

    def add(origin: str, arm: str, endpoint: str, sync_c: float) -> None:
        endpoint_index[(sid, origin, arm, "N", endpoint)] = {
            "sync_c": sync_c,
            "sync_d": 0.0,
            "offset": 0,
        }

    for origin, arm in (("H", "N"), ("H", "BRIDGE_075"), ("S", "N"), ("S", "MAG")):
        for endpoint in ("FULL", "COMMON_INTERIOR", "U"):
            add(origin, arm, endpoint, 1.0)
    add("H", "BRIDGE_075", "FULL", 3.0)
    add("H", "BRIDGE_075", "COMMON_INTERIOR", 4.0)
    add("H", "BRIDGE_075", "U", 5.0)
    add("S", "MAG", "COMMON_INTERIOR", 6.0)
    add("S", "MAG", "U", 7.0)

    indices = np.zeros((config.BOOTSTRAP_DRAWS, 1), dtype=np.int64)
    result = _decomposition(endpoint_index, [record], ["group"], indices, "C")

    assert result["terms"]["identity_check"]["passes"] is True
    assert result["per_record"][0]["total_gap"] == pytest.approx(4.0)
    assert result["per_record"][0]["support_term"] == pytest.approx(1.0)
    assert result["per_record"][0]["window_term"] == pytest.approx(1.0)
    assert result["per_record"][0]["pipeline_term"] == pytest.approx(2.0)
