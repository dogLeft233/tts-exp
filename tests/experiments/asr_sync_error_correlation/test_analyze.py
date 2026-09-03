import numpy as np
import pytest

from scripts.experiments.asr_sync_error_correlation.analyze import (
    aggregate_records,
    bootstrap_interval,
    build_grid,
    compute_metrics,
    decisions,
)


def test_grid_boundaries_and_metrics():
    grid = build_grid([0.0, 0.1, 0.2, 0.3], [0.0, -3.0, -2.0, 0.0], [{"start_s": 0.1, "end_s": 0.3}])
    assert grid["error_mask"].tolist() == [False, True, True, False]
    assert grid["low_sync_mask"].tolist() == [False, True, False, False]
    metrics = compute_metrics(grid, word_error_rate=0.5)
    assert metrics["intersection_cells"] == 1
    assert metrics["error_recall"] == pytest.approx(0.5)
    assert metrics["low_sync_precision"] == pytest.approx(1.0)
    assert metrics["spearman_error_vs_negative_local"] > 0


def test_constant_confidence_nulls_metrics():
    grid = build_grid([0.0, 0.1], [1.0, 1.0], [{"start_s": 0.0, "end_s": 0.1}])
    metrics = compute_metrics(grid)
    assert metrics["low_sync_cells"] == 0
    assert metrics["low_sync_precision_null_reason"] == "no_low_sync_cells"


def test_bootstrap_determinism_and_sample_pairing():
    first = bootstrap_interval([1.0, 2.0, 3.0], draws=100)
    second = bootstrap_interval([1.0, 2.0, 3.0], draws=100)
    assert first == second
    rows = [
        {"sample_id": "a", "source_group": "ga", "arm": "natural", "spearman_error_vs_negative_local": 0.1, "badness_contrast": 0.2, "intersection_cells": 1, "union_cells": 2, "error_cells": 2, "low_sync_cells": 1, "iou": 0.5, "error_recall": 0.5, "low_sync_precision": 1.0, "word_error_rate": 0.1},
        {"sample_id": "a", "source_group": "ga", "arm": "tts", "spearman_error_vs_negative_local": 0.2, "badness_contrast": 0.4, "intersection_cells": 1, "union_cells": 2, "error_cells": 2, "low_sync_cells": 1, "iou": 0.5, "error_recall": 0.5, "low_sync_precision": 1.0, "word_error_rate": 0.1},
        {"sample_id": "b", "source_group": "gb", "arm": "natural", "spearman_error_vs_negative_local": 0.1, "badness_contrast": 0.2, "intersection_cells": 1, "union_cells": 2, "error_cells": 2, "low_sync_cells": 1, "iou": 0.5, "error_recall": 0.5, "low_sync_precision": 1.0, "word_error_rate": 0.1},
    ]
    aggregate = aggregate_records(rows, draws=100)
    assert aggregate["paired"]["defined_pair_count"] == 1


def test_engineering_failure_prevents_science():
    result = decisions({"sample_count": 24, "successful_arm_count": 47}, engineering_checks={"plots": False})
    assert result["engineering"] == "NO_GO"
    assert result["scientific"]["natural"]["status"] == "NOT_EVALUATED"
