from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.portable_source_axis_20260930.independent_consumer import (
    ConsumerError,
    _effect_summary,
    gather_temporal_null_matrix,
    make_temporal_permutations,
)


def test_frozen_pcg64_permutation_domain_and_hashes_are_deterministic() -> None:
    first, first_hashes = make_temporal_permutations(64, ordered_eval_index=17, count=5)
    second, second_hashes = make_temporal_permutations(64, ordered_eval_index=17, count=5)
    assert first_hashes == second_hashes
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert len(set(first_hashes)) == 5
    for permutation in first:
        assert np.array_equal(np.sort(permutation), np.arange(64))


def test_temporal_null_gathers_fixed_guard_queries_without_dropping_rows() -> None:
    cross = np.arange(80 * 80, dtype=np.float32).reshape(80, 80)
    zero = np.full(80, -5.0, dtype=np.float32)
    queries = np.arange(20, 60, dtype=np.int64)
    permutation = np.arange(80, dtype=np.int64)[::-1]
    lags = np.arange(-15, 16, dtype=np.int64)
    matrix = gather_temporal_null_matrix(cross, zero, queries, permutation, lags)
    assert matrix.shape == (len(queries), 31)
    for row, query in enumerate(queries):
        for column, lag in enumerate(lags):
            assert matrix[row, column] == cross[query, permutation[query + lag]]


def test_temporal_null_rejects_guard_query_that_would_drop_lag_support() -> None:
    cross = np.zeros((40, 40), dtype=np.float32)
    zero = np.zeros(40, dtype=np.float32)
    permutation = np.arange(40, dtype=np.int64)
    with pytest.raises(ConsumerError, match="invalid source index"):
        gather_temporal_null_matrix(cross, zero, np.asarray([14]), permutation)


def test_effect_summary_preserves_intended_denominator_when_clip_is_missing() -> None:
    speaker_by_sample = {
        "a1": "s1", "a2": "s1", "b1": "s2", "b2": "s2",
    }
    complete = _effect_summary(
        {"a1": 1.0, "a2": 3.0, "b1": 5.0, "b2": 7.0},
        speaker_by_sample, sorted(speaker_by_sample), draws=500,
    )
    assert complete["eligible_for_intended_cohort"] is True
    assert complete["equal_speaker_mean"] == pytest.approx(4.0)
    assert complete["complete_two_clip_speaker_count"] == 2
    assert len(complete["speaker_bootstrap_99_ci"]) == 2

    incomplete = _effect_summary(
        {"a1": 1.0, "a2": 3.0, "b1": 5.0},
        speaker_by_sample, sorted(speaker_by_sample), draws=500,
    )
    assert incomplete["eligible_for_intended_cohort"] is False
    assert incomplete["expected_clip_count"] == 4
    assert incomplete["complete_paired_clip_count"] == 3
    assert incomplete["missing_clip_ids"] == ["b2"]
    assert incomplete["complete_two_clip_speaker_count"] == 1
    assert incomplete["speaker_bootstrap_99_ci"] is None
