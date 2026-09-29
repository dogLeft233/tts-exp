"""Contract tests for the independent saved-output auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "experiments" / "check_tts_acoustic_reorganization_independent.py"
SPEC = importlib.util.spec_from_file_location("check_tts_acoustic_reorganization_independent", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_independent_curve_metrics_and_cell_cardinality() -> None:
    lags = list(range(-15, 16))
    values = [1.0] * 31
    values[17] = 0.25
    metrics = MODULE._curve_metrics({"lags": lags, "values": values})
    assert metrics["D"] == pytest.approx(0.25)
    assert metrics["B"] == pytest.approx(1.0)
    assert metrics["C"] == pytest.approx(0.75)
    assert metrics["k_star"] == 2
    assert len(MODULE._expected_cells()) == 210
