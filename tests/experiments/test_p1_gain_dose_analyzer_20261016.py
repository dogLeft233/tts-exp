"""Synthetic unit tests for P1 analysis contracts; no experiment data or models."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/experiments/p1_gain_dose_response_20261016/analyze_p1_results.py"
SPEC = importlib.util.spec_from_file_location("p1_analyzer", SOURCE)
assert SPEC and SPEC.loader
p1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p1)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_manifest(n: int = 4) -> dict:
    speaker_ids = [f"SYN{i:02d}" for i in range(n)]
    dose_db = {"RAW_R1": 0.0, "RAW_R2": 0.0, "FIXED_R1": None, "FIXED_R2": None,
               "DOSE_M6": -6.0, "DOSE_M3": -3.0, "DOSE_P3": 3.0, "DOSE_P6": 6.0}
    plan = [{"condition_id": cid, "relative_dose_db": dose_db[cid]} for cid in p1.CONDITION_ORDER]
    lags = list(range(-15, 16))
    speakers = []
    offsets = {
        "RAW_R1": 0.0,
        "RAW_R2": 0.05,
        "FIXED_R1": 0.25,
        "FIXED_R2": 0.45,
        "DOSE_M6": -0.10,
        "DOSE_M3": 0.05,
        "DOSE_P3": 0.35,
        "DOSE_P6": 0.60,
    }
    for i, speaker_id in enumerate(speaker_ids):
        base = 1.0 + 0.1 * i
        fixed_relative = -2.0 + 0.2 * i
        conditions = []
        for cid in p1.CONDITION_ORDER:
            scale = base + offsets[cid]
            curve = [10.0 + scale * abs(lag) for lag in lags]
            relative = fixed_relative if cid.startswith("FIXED") else dose_db[cid]
            audio_level = -25.0 + (fixed_relative if cid.startswith("FIXED") else dose_db[cid])
            b = float(np.median(curve))
            dmin = float(min(curve))
            d0 = float(curve[lags.index(0)])
            anchor_d = float(curve[lags.index(0)])
            conditions.append(
                {
                    "condition_id": cid,
                    "status": "PASS",
                    "relative_dose_db": relative,
                    "audio_rms_dbfs": audio_level,
                    "q_support": list(range(10, 30)),
                    "lags": lags,
                    "curve31": curve,
                    "B": b,
                    "Dmin": dmin,
                    "C": b - dmin,
                    "D0": d0,
                    "anchor": {
                        "D": anchor_d,
                        "C": b - anchor_d,
                        "bestlag": 0.0,
                        "search_gain": d0 - dmin,
                    },
                    "motion": {"mean_aperture": 0.1 + i / 100.0, "mean_speed": 0.02},
                    "encoder": {"direction_cosine": 0.9, "norm_ratio": 1.0},
                    "video_sha256": _h(f"video-{speaker_id}-{cid}"),
                    "visual_sha256": _h(f"visual-{speaker_id}-{cid}"),
                    "audio_variant_sha256": _h(f"audio-{speaker_id}-{cid}"),
                    "distance_path": f"distances/{speaker_id}/{cid}.npy",
                    "distance_sha256": _h(f"distance-{speaker_id}-{cid}"),
                    "distance_shape": [20, 31],
                    "engine_metadata": {"fixture": True},
                    "lag_frame_convention": "synthetic frame offsets",
                }
            )
        speakers.append(
            {
                "speaker_id": speaker_id,
                "source_sha256": _h(f"source-{speaker_id}"),
                "support": {
                    "status": "PASS",
                    "primary_guard20_q_indices": list(range(10, 30)),
                    "full_q_indices": list(range(5, 35)),
                },
                "conditions": conditions,
            }
        )
    return {"schema_version": "synthetic-test-only", "protocol_sha256": _h("synthetic-protocol"), "planned_speakers": speaker_ids,
            "planned_conditions": plan, "speakers": speakers}


def test_primary_averages_repeats_per_speaker_before_pairing() -> None:
    doc = synthetic_manifest(4)
    data = p1.normalize_manifest(doc, require_locked_cohort=False)
    s = data["speakers"][0]
    assert s["raw_mean_C"] == (s["conditions"]["RAW_R1"]["C"] + s["conditions"]["RAW_R2"]["C"]) / 2
    assert s["fixed_mean_C"] == (s["conditions"]["FIXED_R1"]["C"] + s["conditions"]["FIXED_R2"]["C"]) / 2
    assert np.isclose(s["primary_delta_C"], 2.6)
    summary, _, _ = p1.analyze(data, "synthetic-input", "synthetic-source")
    assert summary["primary_fixed_minus_raw"]["status"] == "COMPUTABLE"
    assert summary["primary_fixed_minus_raw"]["n"] == 4


def test_dose_family_is_centered_max_t_with_degenerate_zero_variance_policy() -> None:
    x = np.zeros((4, 4), dtype=float)
    indices = p1.bootstrap_indices(4, draws=256, seed=p1.SEED)
    result = p1.max_t_family_stats(x, list(p1.DOSE_ORDER), indices)
    assert result["critical_value"] == 0.0
    for row in result["contrasts"].values():
        assert row["sample_se"] == 0.0
        assert row["simultaneous_ci95"] == [0.0, 0.0]
        assert row["zero_variance"] is True


def test_bootstrap_is_deterministic_and_family_ci_is_simultaneous() -> None:
    idx1 = p1.bootstrap_indices(5, draws=400)
    idx2 = p1.bootstrap_indices(5, draws=400)
    assert np.array_equal(idx1, idx2)
    x = np.asarray([[1, 0, -1, 2], [2, 1, 0, 2], [1, -1, -2, 0], [3, 2, 0, 4], [0, 0, -1, 1]], dtype=float)
    result = p1.max_t_family_stats(x, list(p1.DOSE_ORDER), idx1)
    crit = result["critical_value"]
    assert crit >= 0.0
    for j, key in enumerate(p1.DOSE_ORDER):
        row = result["contrasts"][key]
        assert row["simultaneous_ci95"][0] <= row["mean"] <= row["simultaneous_ci95"][1]
        assert np.isclose(row["simultaneous_ci95"][1] - row["mean"], crit * row["sample_se"])


def test_missing_planned_dose_is_retained_and_withholds_family_inference() -> None:
    doc = synthetic_manifest(4)
    doc["speakers"][1]["conditions"] = [
        row for row in doc["speakers"][1]["conditions"] if row["condition_id"] != "DOSE_P6"
    ]
    data = p1.normalize_manifest(doc, require_locked_cohort=False)
    summary, _, _ = p1.analyze(data, "synthetic-input", "synthetic-source")
    assert len(data["condition_rows"]) == 4 * 8
    missing = [r for r in data["condition_rows"] if r["speaker_id"] == "SYN01" and r["condition_id"] == "DOSE_P6"]
    assert len(missing) == 1 and missing[0]["status"] == "MISSING_CONDITION_ROW"
    assert summary["primary_fixed_minus_raw"]["status"] == "COMPUTABLE"
    assert summary["dose_family_vs_raw_r1"]["status"] == "WITHHELD_INCOMPLETE_PLANNED_DENOMINATOR"


def test_locked_cohort_is_verified_in_production_mode() -> None:
    doc = synthetic_manifest(4)
    try:
        p1.normalize_manifest(doc)
    except ValueError as exc:
        assert "frozen 20-speaker order" in str(exc)
    else:
        raise AssertionError("non-science fixture must not pass production cohort lock")


def test_svg_renderers_keep_fixed_rms_on_its_own_axis() -> None:
    data = p1.normalize_manifest(synthetic_manifest(4), require_locked_cohort=False)
    summary, _, _ = p1.analyze(data, "synthetic-input", "synthetic-source")
    with TemporaryDirectory(prefix="p1_synthetic_plot_test_") as temp:
        root = Path(temp)
        p1._render_effects_svg(summary, data["speakers"], root / "effects.svg")
        p1._render_dose_family_svg(summary, root / "family.svg")
        p1._render_levels_svg(data, root / "levels.svg")
        p1._render_curves_svg(summary, root / "curves.svg")
        effect_svg = (root / "effects.svg").read_text(encoding="utf-8")
        family_svg = (root / "family.svg").read_text(encoding="utf-8")
        levels_svg = (root / "levels.svg").read_text(encoding="utf-8")
        assert "RAW_R1 is zero" in effect_svg
        assert "Relative dose from RAW (dB)" in effect_svg
        assert "actual RMS dBFS" in levels_svg
        assert "simultaneous 95% max-|t|" in family_svg and "ΔC = C(dose) − C(RAW_R1)" in family_svg
        assert all((root / name).stat().st_size > 100 for name in ("effects.svg", "family.svg", "levels.svg", "curves.svg"))


class _AnalyzerTests(unittest.TestCase):
    pass


for _test_name in [name for name in globals() if name.startswith("test_")]:
    _test_function = globals()[_test_name]
    setattr(_AnalyzerTests, _test_name, lambda self, fn=_test_function: fn())


if __name__ == "__main__":
    unittest.main()
