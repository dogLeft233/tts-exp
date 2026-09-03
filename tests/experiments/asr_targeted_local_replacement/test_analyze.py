from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.asr_targeted_local_replacement.analyze import (
    bootstrap_medians,
    compute_estimands,
    decide,
    summarize_rows,
)


def _scores(target_c=3.0, target_d=7.0):
    return {
        "natural": {"sync_c": 2.0, "sync_d": 8.0},
        "asr_targeted": {"sync_c": target_c, "sync_d": target_d},
        "target_control_0": {"sync_c": 2.5, "sync_d": 7.5},
        "target_control_1": {"sync_c": 2.0, "sync_d": 7.0},
    }


def test_estimands_have_positive_is_better_signs():
    result = compute_estimands(_scores())
    assert result == {"baseline_C_gain": 1.0, "baseline_D_gain": 1.0, "control_C_adv": 0.75, "control_D_adv": 0.25}


def test_bootstrap_resamples_complete_source_group_rows():
    rows = [{"unit_id": f"g{i}", **{name: float(i + 1) for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}} for i in range(12)]
    first = bootstrap_medians(rows, draws=100, seed=20260901)
    second = bootstrap_medians(rows, draws=100, seed=20260901)
    assert first == second
    assert all(value["unit"] == "source_group" and value["percentile_method"] == "linear" for value in first.values())


def test_duplicate_source_group_is_rejected():
    rows = [{"unit_id": "same", **{name: 1.0 for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}}, {"unit_id": "same", **{name: 2.0 for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}}]
    with pytest.raises(ValueError, match="source_group"):
        summarize_rows(rows)


def test_decision_boundary_and_interval_interpretation():
    rows = [{"unit_id": str(i), **{name: 1.0 for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}} for i in range(12)]
    bootstrap = {name: {"lower": 0.0, "upper": 1.0} for name in ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")}
    result = decide(engineering_status="GO", rows=rows, bootstrap=bootstrap)
    assert result["science"] == "NO_PROTOTYPE_SUPPORT"
    assert set(result["interval_interpretation"].values()) == {"crosses_zero"}
    assert decide(engineering_status="NO_GO", rows=rows, bootstrap=bootstrap)["science"] == "NOT_EVALUATED"
