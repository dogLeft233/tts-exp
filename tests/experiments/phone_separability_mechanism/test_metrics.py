from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.experiments.phone_separability_mechanism.metrics import (
    contrast_scores,
    fit_probe_bundle,
    paired_group_bootstrap,
    score_frozen_support,
)


def _row(label: str, condition: str, group: str, vector: tuple[float, float], split: str = "fit") -> dict:
    return {"label": label, "condition": condition, "source_group": group, "analysis_split": split, "view": "full", "speech": True, "valid": True, "embedding": list(vector), "token_id": f"{group}-{condition}-{label}"}


def test_probe_is_group_equal_and_returns_null_for_empty_support() -> None:
    rows = []
    for condition in ("natural", "tts"):
        for group in ("g1", "g2", "g3"):
            rows.extend([_row("a", condition, group, (1.0, 0.0)), _row("b", condition, group, (0.0, 1.0))])
    reference = fit_probe_bundle(rows, min_tokens=1, min_groups=3)
    assert reference["labels"] == ["a", "b"]
    scored = score_frozen_support(rows, reference, condition="natural", min_labels=2, min_tokens=2, min_coverage=1.0)
    assert scored["accuracy"] == 1.0
    assert paired_group_bootstrap({})["estimate"] is None
    json.dumps(scored, allow_nan=False)


def test_contrast_uses_group_effects_not_token_count() -> None:
    rows = {
        "natural": {"eligible_groups": ["g1", "g2"], "group_metrics": [{"source_group": "g1", "accuracy": 0.5, "margin": 0.1}, {"source_group": "g2", "accuracy": 0.5, "margin": 0.1}]},
        "tts": {"eligible_groups": ["g1", "g2"], "group_metrics": [{"source_group": "g1", "accuracy": 0.6, "margin": 0.2}, {"source_group": "g2", "accuracy": 0.7, "margin": 0.3}]},
    }
    result = contrast_scores(rows, "tts", "natural", draws=100)
    assert result["accuracy"]["estimate"] == pytest.approx(0.15)
    assert result["accuracy"]["n_groups"] == 2


def test_empty_reference_is_a_structured_no_support_result() -> None:
    scored = score_frozen_support([_row("a", "natural", "g1", (1.0, 0.0))], {"labels": [], "centroids": {}}, condition="natural")
    assert scored["support_ok"] is False
    assert scored["reason"] == "EMPTY_REFERENCE"
    assert scored["accuracy"] is None
