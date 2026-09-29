from __future__ import annotations

from scripts.experiments.tts_evidence_temporal_patch.analysis import (
    assert_c_decomposition,
    cluster_bootstrap,
    holm_adjust,
    paired_sign_flip,
    summarize_patch,
)


def test_group_bootstrap_is_deterministic_and_equal_weighted() -> None:
    first = cluster_bootstrap({"a": 1.0, "b": 3.0, "c": 5.0}, draws=100, seed=7)
    second = cluster_bootstrap({"a": 1.0, "b": 3.0, "c": 5.0}, draws=100, seed=7)
    assert first == second
    assert first["mean"] == 3.0


def test_exact_sign_flip_and_holm() -> None:
    result = paired_sign_flip([1.0, 1.0, 1.0])
    assert result["enumerated"] == 8
    assert result["p"] == 0.25
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_c_decomposition_is_local_to_one_support() -> None:
    assert_c_decomposition(4.0, 1.5, 2.5)


def test_patch_summary_does_not_promote_pending_human() -> None:
    rows = []
    for group in ("a", "b", "c"):
        for condition, metric in (("BASE", 1.0), ("COHERENT", 2.0), ("SCRAMBLED", 0.5), ("ERASE", 0.0)):
            rows.append({"source_group": group, "direction": "natural_from_tts", "condition": condition, "metric_family": "rank", "metric": metric})
    result = summarize_patch(rows)
    assert result["human"]["status"] == "PENDING_HUMAN"
    assert result["mediation_percentage"] is None
