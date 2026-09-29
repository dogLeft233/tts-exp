from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.experiments.lrs3_phone_rules_metrics import (
    decide_stage_a,
    derive_frame_times,
    fit_reference_centroids,
    normalize_phone,
    paired_bootstrap,
    pool_phone_tokens,
    score_pair,
)


def _record(label: str, condition: str, group: str, vector: tuple[float, float], *, valid: bool = True) -> dict:
    return {
        "label": label,
        "condition": condition,
        "source_group": group,
        "embedding": list(vector),
        "valid": valid,
        "speech": True,
    }


def test_frontend_centers_use_receptive_field_not_half_stride() -> None:
    kernels = [10, 3, 3, 3, 3, 2, 2]
    strides = [5, 2, 2, 2, 2, 2, 2]
    times = derive_frame_times(2, 16_000, kernels, strides)
    assert times[0] == pytest.approx(199.5 / 16_000)
    assert times[1] - times[0] == pytest.approx(320 / 16_000)


def test_normalize_phone_preserves_ipa_marks_and_only_nfc_strips() -> None:
    assert normalize_phone("  iː  ") == "iː"
    assert normalize_phone("e\u0301") == "é"
    assert normalize_phone("tʰ") == "tʰ"


def test_pooling_is_half_open_and_retains_no_frame_token() -> None:
    features = np.eye(4, dtype=np.float64)
    times = np.asarray([0.0, 0.1, 0.2, 0.3])
    tokens = [
        {"label": "a", "start_s": 0.0, "end_s": 0.2, "speech": True},
        {"label": "sil", "start_s": 0.2, "end_s": 0.25, "silence": True},
        {"label": "b", "start_s": 0.25, "end_s": 0.26, "speech": True},
    ]
    pooled = pool_phone_tokens(features, times, tokens, sample_id="x")
    assert pooled[0]["frame_indices"] == [0, 1]
    assert pooled[1]["reason"] == "non_speech"
    assert pooled[2]["valid"] is False
    assert pooled[2]["reason"] == "no_frame"


def test_reference_fit_equalizes_groups_and_conditions() -> None:
    records = []
    for condition in ("natural", "tts"):
        for group in ("g1", "g2", "g3"):
            for _ in range(2):
                records.append(_record("a", condition, group, (1.0, 0.0)))
                records.append(_record("b", condition, group, (0.0, 1.0)))
    # A duplicate clip in g1 must not change the group-equally-weighted center.
    records.extend(_record("a", "natural", "g1", (1.0, 0.0)) for _ in range(20))
    reference = fit_reference_centroids(records, min_tokens=3, min_groups=3)
    assert reference["labels"] == ["a", "b"]
    assert reference["support"]["a"]["natural"]["group_count"] == 3
    assert np.asarray(reference["centroids"]["a"])[0] > 0.9


def test_score_pair_uses_common_labels_and_reports_coverage() -> None:
    reference = {
        "labels": ["a", "b"],
        "centroids": {"a": [1.0, 0.0], "b": [0.0, 1.0]},
    }
    records = [
        {**_record("a", "natural", "g", (1.0, 0.0)), "token_id": "a1"},
        {**_record("b", "natural", "g", (0.0, 1.0)), "token_id": "b1"},
        {**_record("c", "natural", "g", (1.0, 0.0)), "token_id": "c1"},
    ]
    result = score_pair(records, reference, condition="natural", pair_id="x", min_labels=2, min_tokens=2, min_coverage=0.5)
    assert result["eligible"] is True
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["coverage"] == pytest.approx(2 / 3)


def test_score_pair_uses_json_null_for_undefined_accuracy() -> None:
    reference = {
        "labels": ["a"],
        "centroids": {"a": [1.0, 0.0]},
    }
    records = [{**_record("unsupported", "natural", "g", (1.0, 0.0)), "token_id": "u1"}]

    result = score_pair(records, reference, condition="natural", pair_id="x")

    assert result["eligible"] is False
    assert result["accuracy"] is None
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_bootstrap_is_seed_stable_and_swap_changes_direction() -> None:
    positive = paired_bootstrap({"a": 0.1, "b": 0.2, "c": 0.3}, seed=20260920, draws=1000)
    negative = paired_bootstrap({key: -value for key, value in {"a": 0.1, "b": 0.2, "c": 0.3}.items()}, seed=20260920, draws=1000)
    assert positive == paired_bootstrap({"c": 0.3, "a": 0.1, "b": 0.2}, seed=20260920, draws=1000)
    assert positive["estimate"] == pytest.approx(0.2)
    assert negative["estimate"] == pytest.approx(-0.2)
    assert negative["ci_low"] == pytest.approx(-positive["ci_high"])


def test_stage_a_gate_has_explicit_boundary_and_directional_guard() -> None:
    good = {"estimate": 0.020, "ci_low": 0.001, "ci_high": 0.04}
    assert decide_stage_a(good, {"estimate": 0.001}, support_ok=True)["science_decision"] == "ADVANTAGE_SUPPORTED"
    bad = decide_stage_a({"estimate": 0.0199, "ci_low": 0.001}, {"estimate": 0.001}, support_ok=True)
    assert bad["science_decision"] == "NO_CLEAR_ADVANTAGE"
    assert "HUBERT_EFFECT_BELOW_THRESHOLD" in bad["reason_codes"]
    assert decide_stage_a(good, {"estimate": -0.001}, support_ok=True)["science_decision"] == "NO_CLEAR_ADVANTAGE"
    assert decide_stage_a(good, good, support_ok=False)["science_decision"] == "INSUFFICIENT_SUPPORT"
