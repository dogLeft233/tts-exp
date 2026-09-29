from __future__ import annotations

import numpy as np

from scripts.experiments.phone_separability_enhancement.metrics import (
    benjamini_hochberg,
    build_abx_triplets,
    score_fixed_support,
    signed_phone_scores,
)


def test_signed_margin_is_correct_label_not_top1_gap() -> None:
    result = signed_phone_scores([1.0, 0.0], "b", {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert result["prediction"] == "a"
    assert result["signed_margin"] < 0.0
    assert result["legacy_top1_gap"] > 0.0


def test_fixed_support_keeps_missing_token_in_denominator() -> None:
    expected = [
        {"support_key": "g1-a", "source_group": "g1", "label": "a"},
        {"support_key": "g1-b", "source_group": "g1", "label": "b"},
    ]
    rows = [{"support_key": "g1-a", "source_group": "g1", "label": "a", "embedding": [1.0, 0.0], "valid": True}]
    result = score_fixed_support(rows, {"a": [1.0, 0.0], "b": [0.0, 1.0]}, expected=expected)
    assert result["support_count"] == 2
    assert result["missing_count"] == 1
    assert result["accuracy"] == 0.5


def test_abx_never_reuses_a_as_x_and_ties_are_half() -> None:
    rows = [
        {"source_group": "g", "label": "a", "occurrence_key": "a1", "embedding": [1.0, 0.0], "valid": True},
        {"source_group": "g", "label": "a", "occurrence_key": "a2", "embedding": [1.0, 0.0], "valid": True},
        {"source_group": "g", "label": "b", "occurrence_key": "b1", "embedding": [0.0, 1.0], "valid": True},
    ]
    triplets = build_abx_triplets(rows)
    assert triplets
    assert all(row["a"] != row["x"] for row in triplets)


def test_bh_monotone_adjustment() -> None:
    values = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5})
    assert values["a"] <= values["b"] <= values["c"]
