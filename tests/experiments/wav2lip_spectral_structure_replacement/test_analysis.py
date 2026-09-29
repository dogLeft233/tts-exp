from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_spectral_structure_replacement import analysis, config


def test_summarize_averages_rows_before_peak() -> None:
    matrix = np.full((4, config.MATRIX_COLUMNS), 10.0, dtype=np.float64)
    matrix[0, 15] = 0.0
    matrix[1, 15] = 2.0
    result = analysis.summarize(matrix, [0, 1])
    expected = analysis.peak(np.mean(matrix[[0, 1]], axis=0))
    assert result["offset"] == expected["offset"]
    assert result["sync_c"] == expected["sync_c"]
    assert result["sync_d"] == expected["sync_d"]


def test_bootstrap_uses_fixed_shared_draws() -> None:
    groups = [f"g{i}" for i in range(config.EXPECTED_SOURCE_GROUP_COUNT)]
    labels, indices = analysis._shared_indices(groups)
    left = analysis.bootstrap(np.arange(22, dtype=np.float64), groups, level=0.95, labels=labels, indices=indices)
    right = analysis.bootstrap(np.arange(22, dtype=np.float64) + 1.0, groups, level=0.95, labels=labels, indices=indices)
    assert left["draw_indices_sha256"] == right["draw_indices_sha256"]
    assert right["mean"] - left["mean"] == 1.0


def test_input_loader_performs_sample_id_join() -> None:
    from scripts.experiments.wav2lip_spectral_structure_replacement import protocol

    inputs = protocol.load_frozen_inputs()
    assert len(inputs["rows"]) == config.EXPECTED_RECORD_COUNT
    assert len({row["source_group"] for row in inputs["rows"]}) == config.EXPECTED_SOURCE_GROUP_COUNT
    assert all(len(row["u"]) == len(row["support"]) for row in inputs["rows"])
    assert all(all(item["support_legal"] for item in row["support"]) for row in inputs["rows"])
