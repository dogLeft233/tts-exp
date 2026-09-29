from __future__ import annotations

import numpy as np

from scripts.experiments.phone_separability_mechanism.features import match_occurrences, pool_views


def _token(label: str, start: float, end: float) -> dict:
    return {"label": label, "start_s": start, "end_s": end, "speech": True}


def test_match_occurrences_handles_insertion_without_zip_shift() -> None:
    source = [_token("a", 0.0, 0.1), _token("b", 0.1, 0.2)]
    target = [_token("a", 0.0, 0.1), _token("x", 0.1, 0.15), _token("b", 0.15, 0.2)]
    result = match_occurrences(source, target)
    assert result["cost"] == 1
    assert [(row["source_index"], row["target_index"]) for row in result["matches"]] == [(0, 0), (1, 2)]


def test_match_occurrences_marks_duplicate_alignment_ambiguous() -> None:
    source = [_token("a", 0.0, 0.1), _token("a", 0.1, 0.2)]
    target = [_token("a", 0.0, 0.2)]
    result = match_occurrences(source, target)
    assert result["ambiguous"] is True
    assert result["matched_count"] == 0


def test_pool_views_does_not_invent_short_phone_frames() -> None:
    features = np.eye(4, dtype=np.float64)
    times = np.asarray([0.0, 0.1, 0.2, 0.3])
    tokens = [_token("a", 0.0, 0.2), _token("b", 0.25, 0.26)]
    views = pool_views(features, times, tokens)
    short = [row for row in views["matched_1frame"] if row["label"] == "b"][0]
    assert short["valid"] is False
    assert short["embedding"] is None
