"""Pure CPU checks for the MFA-linear trajectory decomposition protocol."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/experiments/mfa_linear_trajectory_decomposition.py"
SPEC = importlib.util.spec_from_file_location("mfa_linear_trajectory_decomposition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _curve(default: float = 10.0) -> np.ndarray:
    values = np.full(31, default, dtype=np.float64)
    values[15] = 5.0
    return values


def test_curve_metrics_and_four_synthetic_decompositions() -> None:
    baseline = _curve()
    center_down = baseline.copy()
    center_down[15] = 4.0
    background_up = baseline.copy()
    background_up[[index for index in range(31) if index != 15]] = 11.0
    shifted_up = baseline + 3.0
    scaled = baseline * 2.0

    base_metrics = MODULE._curve_metrics(MODULE.LAGS, baseline)
    assert base_metrics["C"] == pytest.approx(5.0)
    assert MODULE._decompose(MODULE._curve_metrics(MODULE.LAGS, center_down), base_metrics) == pytest.approx(
        {"delta_C": 1.0, "match_gain": 1.0, "background_gain": 0.0, "dominance": -1.0}
    )
    assert MODULE._decompose(MODULE._curve_metrics(MODULE.LAGS, background_up), base_metrics) == pytest.approx(
        {"delta_C": 1.0, "match_gain": 0.0, "background_gain": 1.0, "dominance": 1.0}
    )
    assert MODULE._decompose(MODULE._curve_metrics(MODULE.LAGS, shifted_up), base_metrics) == pytest.approx(
        {"delta_C": 0.0, "match_gain": -3.0, "background_gain": 3.0, "dominance": 6.0}
    )
    scaled_metrics = MODULE._curve_metrics(MODULE.LAGS, scaled)
    assert scaled_metrics["C"] == pytest.approx(2.0 * base_metrics["C"])
    base_lag = MODULE._lag_decomposition(baseline, MODULE.LAGS, 0)
    scaled_lag = MODULE._lag_decomposition(scaled, MODULE.LAGS, 0)
    assert scaled_lag["relative_margin"] == pytest.approx(base_lag["relative_margin"])


def test_ties_use_first_lag_and_anchor_uses_lag_value() -> None:
    values = np.full(31, 9.0, dtype=np.float64)
    values[15] = 4.0
    values[16] = 4.0
    metrics = MODULE._curve_metrics(MODULE.LAGS, values)
    assert metrics["k_star"] == 0

    natural = np.arange(31, dtype=np.float64) + 10.0
    natural[18] = 1.0  # k_N=+3 is index 18, not array index 3.
    natural_metrics = MODULE._curve_metrics(MODULE.LAGS, natural)
    assert natural_metrics["k_star"] == 3
    candidate = natural.copy()
    candidate[17] = 0.5
    extra = MODULE._lag_decomposition(candidate, MODULE.LAGS, natural_metrics["k_star"])
    assert extra["D_anchor"] == pytest.approx(candidate[18])
    assert extra["search_bonus"] == pytest.approx(candidate[18] - candidate.min())
    assert extra["C_anchor"] + extra["search_bonus"] == pytest.approx(
        MODULE._curve_metrics(MODULE.LAGS, candidate)["C"]
    )


def test_lag_mask_is_valid_at_boundary() -> None:
    values = _curve()
    result = MODULE._lag_decomposition(values, MODULE.LAGS, -15)
    assert result["O"] == pytest.approx(np.median(values[np.asarray(MODULE.LAGS) > -13]))


def test_optional_relative_margin_stays_null_and_reports_valid_count() -> None:
    tiny = np.full(31, 1e-13, dtype=np.float64)
    assert MODULE._lag_decomposition(tiny, MODULE.LAGS, 0)["relative_margin"] is None
    indices = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    summary = MODULE._optional_summary([1.0, None, 3.0], indices)
    assert summary["mean"] is None
    assert summary["ci95"] == [None, None]
    assert summary["valid_n"] == 2
    assert summary["missing_n"] == 1

    base = {
        "C": np.asarray([5.0, 5.0, 5.0]),
        "D": np.asarray([5.0, 6.0, 7.0]),
        "B": np.asarray([10.0, 11.0, 12.0]),
        "D_anchor": np.asarray([5.0, 6.0, 7.0]),
        "C_anchor": np.asarray([5.0, 5.0, 5.0]),
        "search_bonus": np.asarray([0.0, 0.0, 0.0]),
        "O": np.asarray([12.0, 12.0, 12.0]),
        "relative_margin": [0.0, 0.0, 1.0],
        "lag_delta": np.asarray([0.0, 0.0, 0.0]),
    }
    candidate = {
        **base,
        "C": np.asarray([7.0, 7.0, 7.0]),
        "D": np.asarray([4.0, 5.0, 6.0]),
        "B": np.asarray([11.0, 12.0, 13.0]),
        "relative_margin": [1.0, None, 3.0],
    }
    effect = MODULE._effect_from_condition_vectors(candidate, base)
    assert effect["relative_margin"] == [1.0, None, 2.0]


def test_interpretation_reports_negative_component_as_cancellation() -> None:
    def stat(interval: list[float]) -> dict[str, list[float]]:
        return {"ci_bonferroni": interval, "ci95": interval}

    def comparison(delta: list[float], match: list[float], background: list[float], dominance: list[float]) -> dict[str, dict[str, list[float]]]:
        return {
            "delta_C": stat(delta),
            "match_gain": stat(match),
            "background_gain": stat(background),
            "dominance": stat(dominance),
        }

    analysis = {
        "primary": {
            "comparisons": {
                "N_dose": comparison([1.0, 2.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
                "T_dose": comparison([1.0, 2.0], [-2.0, -1.0], [2.0, 3.0], [-1.0, 1.0]),
                "source_interaction": comparison([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]),
            }
        },
        "descriptive": {"source_differences": {"own": {"T_100_minus_N_RAW": {"delta_C": {"ci95": [-1.0, 1.0]}}}}},
        "dose": {"natural": {"T": {"segments": {"100_minus_000": {"effects": {"delta_C": {"ci95": [-1.0, 1.0]}}}}}}},
    }
    lines = MODULE._interpretation(analysis)
    assert any("match_gain 的校正区间上界低于 0" in line for line in lines)


def test_bootstrap_uses_shared_indices_and_exact_corrected_quantile() -> None:
    indices = MODULE._bootstrap_indices(draws=128, n=3, seed=7)
    values = np.asarray([1.0, 2.0, 8.0])
    summary = MODULE._bootstrap_summary(values, indices, corrected=True)
    means = values[indices].mean(axis=1)
    expected = np.quantile(means, [MODULE.BONFERRONI_ALPHA / 2, 1 - MODULE.BONFERRONI_ALPHA / 2], method="linear")
    assert summary["ci_bonferroni"] == pytest.approx(expected.tolist())
    assert summary["n"] == 3
    assert summary["positive_count"] == 3
    second = MODULE._bootstrap_summary(values * 2, indices, corrected=True)
    assert second["ci_bonferroni"] == pytest.approx((expected * 2).tolist())
    assert summary["ci_bonferroni_coverage"] == pytest.approx(1 - 0.05 / 12)


def test_source_interaction_is_paired_before_bootstrap_and_two_by_two_identity() -> None:
    natural = np.asarray([1.0, 4.0, 10.0])
    tts = np.asarray([4.0, 7.0, 13.0])
    n_effect = natural - np.asarray([0.0, 1.0, 8.0])
    t_effect = tts - np.asarray([0.0, 2.0, 9.0])
    interaction = t_effect - n_effect
    assert np.mean(interaction) == pytest.approx(np.mean(t_effect) - np.mean(n_effect))
    q00, q10, q01, q11 = (np.asarray(x) for x in ([1, 2, 3], [2, 4, 4], [3, 3, 6], [8, 9, 12]))
    video = q10 - q00
    audio = q01 - q00
    interaction_2x2 = q11 - q10 - q01 + q00
    total = q11 - q00
    assert np.allclose(total, video + audio + interaction_2x2)


def test_aggregate_after_per_sample_metrics_not_mean_curve() -> None:
    first = _curve()
    second = _curve()
    first[15] = 1.0
    second[16] = 2.0
    per_sample = [MODULE._curve_metrics(MODULE.LAGS, values) for values in (first, second)]
    aggregate_c = float(np.mean([row["C"] for row in per_sample]))
    mean_curve_c = MODULE._curve_metrics(MODULE.LAGS, np.mean([first, second], axis=0))["C"]
    assert aggregate_c != pytest.approx(mean_curve_c)
    assert aggregate_c == pytest.approx(np.mean([9.0, 8.0]))


def test_expected_cell_matrix_is_identity_keyed_and_support_thresholds() -> None:
    records = [{"sample_id": 1, "paired_key": "one", "speaker_id": "S0765"}]
    cells = MODULE.expected_cells(records)
    assert len(cells) == 15
    assert (1, "one", "T_RAW", "T_RAW") in cells
    assert (1, "one", "N_RAW", "T_100") in cells
    assert (1, "one", "T_100", "N_RAW") in cells
    assert (1, "wrong", "N_RAW", "N_RAW") not in cells


def test_source_hash_mismatch_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in MODULE.SOURCE_EXPECTED_HASHES:
        (source / name).write_text(name, encoding="utf-8")
    with pytest.raises(MODULE.ProtocolError, match="frozen source hash mismatch"):
        MODULE._verify_source_hashes(source)


def _write_loader_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, str]], dict[str, str], dict[str, object]]:
    records = [{"sample_id": sid, "paired_key": f"pair-{sid}", "speaker_id": "S0765"} for sid in range(1, 16)]
    model_hash = "model-hash"
    manifest = {
        "status": "complete",
        "syncnet_model_sha256": model_hash,
        "sample_count": 15,
        "expected_cells": 225,
        "score_rows": 225,
        "curve_rows": 225,
        "failures": [],
        "scoring_protocol": {"vshift": 15, "min_common_windows": 30},
    }
    curves: list[dict[str, object]] = []
    scores: list[dict[str, str]] = []
    for record in records:
        sample_id = record["sample_id"]
        paired_key = record["paired_key"]
        for _, _, video_arm, audio_arm in sorted(MODULE.expected_cells([{"sample_id": sample_id, "paired_key": paired_key}])):
            values = np.full(31, 10.0 + sample_id / 100.0, dtype=np.float64)
            values[15] = 5.0 + sample_id / 100.0
            support = 31 if (video_arm, audio_arm) == MODULE.REFERENCE_CELL else 30
            curves.append(
                {
                    "sample_id": sample_id,
                    "paired_key": paired_key,
                    "video_arm": video_arm,
                    "audio_arm": audio_arm,
                    "lags": list(MODULE.LAGS),
                    "values": values.tolist(),
                    "common_support_count": support,
                    "k_n": None,  # legal: the analyzer derives k_N from N_RAW/N_RAW.
                }
            )
            score = {
                "status": "complete",
                "sample_id": str(sample_id),
                "paired_key": paired_key,
                "video_arm": video_arm,
                "audio_arm": audio_arm,
                "C": "5.0",
                "D": str(5.0 + sample_id / 100.0),
                "curve_median": str(10.0 + sample_id / 100.0),
                "k_star": "0",
                "d_zero": str(5.0 + sample_id / 100.0),
                "common_support_count": str(support),
                "fixed_k_n": "0" if audio_arm == "N_RAW" else "",
                "d_fixed_natural_lag": str(5.0 + sample_id / 100.0) if audio_arm == "N_RAW" else "",
                "score_box_sha256": f"box-{sample_id}",
                "syncnet_model_sha256": model_hash,
            }
            scores.append(score)
    score_path = tmp_path / "scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(reversed(scores))
    curve_path = tmp_path / "curves.json"
    curve_path.write_text(json.dumps({"schema_version": 1, "lags": list(MODULE.LAGS), "curves": list(reversed(curves))}), encoding="utf-8")
    return score_path, curve_path, records, manifest, {"curves": curves, "scores": scores}


def test_loaders_associate_shuffled_rows_allow_optional_kn_and_t_raw_support(tmp_path: Path) -> None:
    score_path, curve_path, records, manifest, _raw = _write_loader_fixture(tmp_path)
    scores = MODULE._load_scores(score_path, records, manifest)
    curves = MODULE._load_curves(curve_path, records, scores)
    assert len(scores) == len(curves) == 225
    assert curves[(1, "pair-1", "N_RAW", "N_RAW")]["k_n"] is None
    assert curves[(1, "pair-1", "T_RAW", "T_RAW")]["common_support_count"] == 31
    rows, by_sample, _ = MODULE._build_cell_metrics(records, scores, curves)
    assert len(rows) == 225
    assert len(by_sample[1]) == 14
    assert by_sample[1][("N_100", "N_RAW")]["k_N"] == 0


def test_sensitivity_uses_complete_mechanism_support_only(tmp_path: Path) -> None:
    score_path, curve_path, records, manifest, _raw = _write_loader_fixture(tmp_path)
    scores = MODULE._load_scores(score_path, records, manifest)
    curves = MODULE._load_curves(curve_path, records, scores)
    _, by_sample, identity_rows = MODULE._build_cell_metrics(records, scores, curves)
    assert all(row["identity_error"] <= MODULE.ABS_TOL for row in identity_rows)
    support_by_sid = {
        sid: (50 if sid <= 8 else 45 if sid <= 10 else 40 if sid <= 12 else 35 if sid == 13 else 30)
        for sid in range(1, 16)
    }
    for sid, entries in by_sample.items():
        for row in entries.values():
            row["common_support_count"] = support_by_sid[sid]
    table = MODULE._sensitivity(by_sample, np.zeros((1, 15), dtype=np.int64))
    assert [(row["threshold"], row["n"]) for row in table] == [(30, 15), (35, 13), (40, 12), (45, 10), (50, 8)]
    assert table[0]["sample_ids"] == list(range(1, 16))
    assert table[-1]["sample_ids"] == list(range(1, 9))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "lag_order", "nan", "support", "paired_key"])
def test_curve_input_contract_rejects_corruption(tmp_path: Path, mutation: str) -> None:
    score_path, curve_path, records, manifest, raw = _write_loader_fixture(tmp_path)
    scores = MODULE._load_scores(score_path, records, manifest)
    curves = raw["curves"]
    if mutation == "duplicate":
        curves.append(dict(curves[0]))
    elif mutation == "missing":
        curves.pop()
    elif mutation == "lag_order":
        curves[0]["lags"] = list(reversed(MODULE.LAGS))
    elif mutation == "nan":
        curves[0]["values"][0] = float("nan")
    elif mutation == "support":
        curves[0]["common_support_count"] = 29
    elif mutation == "paired_key":
        curves[0]["paired_key"] = "wrong-pair"
    curve_path.write_text(
        json.dumps(
            {"schema_version": 1, "lags": list(MODULE.LAGS), "curves": curves},
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.ProtocolError):
        MODULE._load_curves(curve_path, records, scores)
