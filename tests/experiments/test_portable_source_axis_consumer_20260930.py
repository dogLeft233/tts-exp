from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.portable_source_axis_20260930.independent_consumer import (
    ConsumerError,
    _check_producer_metrics,
    _effect_summary,
    _expected_support,
    _file_sha256,
    _metric_payload,
    _primary_scope_status,
    _read_score_artifact,
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


def test_40_frame_row_keeps_full_and_guard15_but_marks_guard20_too_short() -> None:
    n = 40
    matrix = np.arange(n * 31, dtype=np.float32).reshape(n, 31)
    supports = {
        "full": _expected_support(n, 0),
        "valid": _expected_support(n, 0),
        "guard20": _expected_support(n, 20),
        "guard15": _expected_support(n, 15),
    }
    metrics = _metric_payload(matrix, supports, anchor_column=18)
    assert metrics["full"]["support_n"] == 40
    assert metrics["valid"]["support_n"] == 40
    assert metrics["guard15"]["support_n"] == 10
    assert metrics["guard20"] == {"status": "too_short", "support_n": 0}


def test_score_reader_accepts_scored_40_frame_row_with_empty_guard20(tmp_path) -> None:
    n = 40
    lag_k = np.arange(-15, 16, dtype=np.int32)
    cross_raw = np.arange(n * n, dtype=np.float32).reshape(n, n) / 100.0
    cross_unit = cross_raw / 2.0
    zero_raw = np.full(n, 2.0, dtype=np.float32)
    zero_unit = np.full(n, 1.0, dtype=np.float32)

    def gather(cross: np.ndarray, zero: np.ndarray) -> np.ndarray:
        out = np.empty((n, 31), dtype=np.float32)
        for query in range(n):
            for column, lag in enumerate(lag_k):
                source = query + int(lag)
                out[query, column] = cross[query, source] if 0 <= source < n else zero[query]
        return out

    raw_matrix = gather(cross_raw, zero_raw)
    unit_matrix = gather(cross_unit, zero_unit)
    support = np.arange(n, dtype=np.int32)
    guard15 = np.arange(15, 25, dtype=np.int32)
    guard20 = np.asarray([], dtype=np.int32)
    score_path = tmp_path / "scores" / "short.npz"
    score_path.parent.mkdir(parents=True)
    np.savez_compressed(
        score_path,
        distance_matrix_raw_float32=raw_matrix,
        distance_matrix_unit_float32=unit_matrix,
        cross_distance_raw_float32=cross_raw,
        cross_distance_unit_float32=cross_unit,
        cross_distance_to_zero_raw_float32=zero_raw,
        cross_distance_to_zero_unit_float32=zero_unit,
        common_support_indices_int32=support,
        full_support_indices_int32=support,
        valid_support_indices_int32=support,
        guard20_indices_int32=guard20,
        guard15_indices_int32=guard15,
        lag_columns_int32=np.arange(31, dtype=np.int32),
        lag_k_int32=lag_k,
    )

    supports = {
        "full": support,
        "valid": support,
        "guard20": guard20,
        "guard15": guard15,
    }

    def producer_metrics(matrix: np.ndarray) -> dict[str, dict[str, float | int | str]]:
        values = _metric_payload(matrix, supports, anchor_column=18)
        result = {}
        for name, summary in values.items():
            if summary.get("status") == "too_short":
                result[name] = {"status": "too_short", "support_n": 0}
            else:
                result[name] = {
                    "D_min": summary["D"], "B_median": summary["B"],
                    "C_best": summary["C"], "C_fixed_anchor": summary["C_anchor"],
                    "D_fixed_anchor": summary["D_anchor"],
                }
        return result

    plan = {"row_id": "s1__primary__fourcell__tts__q00", "score_npz": "scores/short.npz"}
    result = {
        "score_npz_sha256": _file_sha256(score_path),
        "common_support_n": n,
        "scoring_audio_domain_n": n,
        "audio_row_domain_n": n,
        "visual_query_domain_n": n,
        "raw_geometry": producer_metrics(raw_matrix),
        "row_unit_geometry": producer_metrics(unit_matrix),
    }
    metrics, arrays = _read_score_artifact(tmp_path, plan, result, anchor_column=18)
    assert arrays["guard20_indices_int32"].size == 0
    assert metrics["raw"]["guard20"] == {"status": "too_short", "support_n": 0}
    assert metrics["raw"]["guard15"]["support_n"] == 10


def test_primary_eligibility_ignores_secondary_t_fourcell_short_support() -> None:
    samples = [f"sample{i:02d}" for i in range(80)]
    speakers = {sample: f"speaker{i // 2:02d}" for i, sample in enumerate(samples)}
    metric_rows = {}
    score_supports = {}
    for sample in samples:
        group = (sample, "primary", "source_axis", "natural")
        score_supports[group] = {
            condition: ("common-query-sha", "guard20-sha")
            for condition in ("baseline", "orth_plus", "orth_minus")
        }
        for condition in ("baseline", "orth_plus", "orth_minus"):
            metric_rows[("primary", sample, "source_axis", "natural", condition)] = {
                "raw": {"C": 1.0, "C_anchor": 0.5, "B": 2.0, "D": 1.0}
            }
        # A T four-cell group may have full support but no guard20 rows. It is
        # outside the predeclared source-axis primary estimand.
        score_supports[(sample, "primary", "fourcell", "tts")] = {"q00": ("n40", "empty")}

    status = _primary_scope_status(
        samples, speakers, metric_rows, score_supports, run_rows_sealed=True,
    )
    assert status["eligible"] is True
    assert status["validated_required_score_rows"] == 240
    assert status["common_support_groups_valid"] == 80

    del metric_rows[("primary", samples[0], "source_axis", "natural", "orth_minus")]
    incomplete = _primary_scope_status(
        samples, speakers, metric_rows, score_supports, run_rows_sealed=True,
    )
    assert incomplete["eligible"] is False
    assert incomplete["validated_required_score_rows"] == 239


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
