from __future__ import annotations

from scripts.experiments.lrs3_mfa_dtw_replacement.statistics import (
    analyze_diagonal,
    cluster_bootstrap,
)


def _rows(c_values=(1.0, 1.0), d_values=(1.0, 1.0)):
    return [
        {
            "sample_id": f"sample_{index}",
            "source_group": group,
            "dtw_sync_c": float(c),
            "linear_sync_c": 0.0,
            "dtw_sync_d": 0.0,
            "linear_sync_d": float(d),
        }
        for index, (group, c, d) in enumerate(zip(("a", "a", "b", "b"), c_values * 2, d_values * 2, strict=True))
    ]


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [{"source_group": "a", "value": 1.0}, {"source_group": "a", "value": 3.0}, {"source_group": "b", "value": 5.0}]
    assert cluster_bootstrap(rows, "value", draws=100, seed=9) == cluster_bootstrap(rows, "value", draws=100, seed=9)


def test_diagonal_gate_requires_both_positive_lower_bounds() -> None:
    result = analyze_diagonal(_rows(), engineering_complete=True, expected_count=4, draws=100, seed=9)
    assert result["decision"] == "DTW_TFG_ADVANTAGE"
    assert result["replacement_authorized"] is True
    blocked = analyze_diagonal(_rows(c_values=(1.0, -1.0)), engineering_complete=True, expected_count=4, draws=100, seed=9)
    assert blocked["decision"] == "NO_DTW_TFG_ADVANTAGE"
    assert blocked["replacement_authorized"] is False
