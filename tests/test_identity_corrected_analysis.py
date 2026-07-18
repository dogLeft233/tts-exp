"""Unit tests for scripts/24_identity_corrected_analysis.py.

These tests cover the residual computation, t-statistic classification,
file-input handling, and the per-intervention residual analysis against the
identity control baseline. End-to-end CLI execution is exercised against
synthetic inputs in tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "script_24", _REPO / "scripts" / "24_identity_corrected_analysis.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_record(delta_c: float, baseline_c: float = 6.0, intervention_c: float | None = None) -> dict:
    ic = baseline_c + delta_c if intervention_c is None else intervention_c
    return {
        "intervention_sync_c": ic,
        "baseline_sync_c": baseline_c,
        "delta_c": delta_c,
    }


def _intervention_record(delta_c: float, baseline_c: float = 6.0) -> dict:
    return {
        "intervention_sync_c": baseline_c + delta_c,
        "baseline_sync_c": baseline_c,
        "delta_c": delta_c,
    }


def _make_synthetic_files(
    tmp_path: Path,
    *,
    identity_deltas: dict[int, float],
    intervention_deltas: dict[str, dict[int, float]],
    natural_deltas: dict[str, dict[int, float]] | None = None,
) -> tuple[Path, Path]:
    """Build tiny dose_response.json + identity_control.json files."""
    dose = {
        "run_id": "synthetic",
        "interventions": {
            name: {
                "source": "tts",
                "baseline_cond": "tts_raw",
                "description": name,
                "expected_sync_direction": "unknown",
                "n_samples": len(per_sample),
                "mean_delta_c": float(np.mean(list(per_sample.values()))),
                "std_delta_c": float(np.std(list(per_sample.values()), ddof=1)) if len(per_sample) > 1 else 0.0,
                "per_sample": {str(sid): _intervention_record(dc) for sid, dc in per_sample.items()},
            }
            for name, per_sample in intervention_deltas.items()
        },
    }
    if natural_deltas:
        for name, per_sample in natural_deltas.items():
            dose["interventions"][name] = {
                "source": "natural",
                "baseline_cond": "natural_raw",
                "description": name,
                "expected_sync_direction": "unknown",
                "n_samples": len(per_sample),
                "mean_delta_c": float(np.mean(list(per_sample.values()))),
                "std_delta_c": float(np.std(list(per_sample.values()), ddof=1)) if len(per_sample) > 1 else 0.0,
                "per_sample": {str(sid): _intervention_record(dc) for sid, dc in per_sample.items()},
            }
    ident = {
        "run_id": "synthetic",
        "interventions": {
            "identity_tts": {
                "source": "tts",
                "baseline_cond": "tts_raw",
                "description": "no-op",
                "expected_sync_direction": "no change",
                "n_samples": len(identity_deltas),
                "mean_delta_c": float(np.mean(list(identity_deltas.values()))),
                "std_delta_c": float(np.std(list(identity_deltas.values()), ddof=1)) if len(identity_deltas) > 1 else 0.0,
                "per_sample": {str(sid): _identity_record(dc) for sid, dc in identity_deltas.items()},
            },
        },
        "verdict": "PIPELINE_CONFOUNDS_DOSE_RESPONSE",
        "verdict_message": "synthetic test verdict",
    }
    dose_p = tmp_path / "dose_response.json"
    ident_p = tmp_path / "identity_control.json"
    dose_p.write_text(json.dumps(dose, indent=2))
    ident_p.write_text(json.dumps(ident, indent=2))
    return dose_p, ident_p


# ---------------------------------------------------------------------------
# Load inputs & identity extraction
# ---------------------------------------------------------------------------


class TestLoadAndExtract:
    def test_load_inputs_returns_both_dicts(self, tmp_path):
        dose_p, ident_p = _make_synthetic_files(
            tmp_path,
            identity_deltas={1: -1.0, 2: -1.2},
            intervention_deltas={"LUFS_match_tts_to_nat": {1: -0.5, 2: -0.7}},
        )
        dose, ident = _MOD.load_inputs(dose_p, ident_p)
        assert "interventions" in dose
        assert "interventions" in ident

    def test_load_inputs_raises_on_missing_dose(self, tmp_path):
        ident_p = tmp_path / "identity_control.json"
        ident_p.write_text(json.dumps({"interventions": {"identity_tts": {"per_sample": {}}}}))
        with pytest.raises(FileNotFoundError):
            _MOD.load_inputs(tmp_path / "missing.json", ident_p)

    def test_load_inputs_raises_on_missing_identity(self, tmp_path):
        dose_p = tmp_path / "dose_response.json"
        dose_p.write_text(json.dumps({"interventions": {}}))
        with pytest.raises(FileNotFoundError):
            _MOD.load_inputs(dose_p, tmp_path / "missing.json")

    def test_get_identity_per_sample_returns_int_keys(self, tmp_path):
        _, ident_p = _make_synthetic_files(
            tmp_path,
            identity_deltas={1: -1.0, 2: -1.2, 3: -0.5},
            intervention_deltas={"LUFS_match_tts_to_nat": {1: -0.5}},
        )
        _, ident = _MOD.load_inputs(dose_p := (tmp_path / "dose_response.json"), ident_p)
        out = _MOD.get_identity_per_sample(ident)
        assert set(out.keys()) == {1, 2, 3}
        assert out[1]["delta_c"] == -1.0
        assert isinstance(list(out.keys())[0], int)

    def test_get_identity_raises_when_identity_tts_missing(self):
        with pytest.raises(ValueError):
            _MOD.get_identity_per_sample({"interventions": {}})


# ---------------------------------------------------------------------------
# compute_residuals
# ---------------------------------------------------------------------------


class TestComputeResiduals:
    def test_residual_matches_manual_subtraction(self):
        intervention = {
            "1": _intervention_record(-1.2),
            "2": _intervention_record(-1.0),
            "3": _intervention_record(-1.5),
        }
        identity = {
            1: _identity_record(-1.0),
            2: _identity_record(-1.0),
            3: _identity_record(-1.0),
        }
        out = _MOD.compute_residuals(intervention, identity)
        assert out["n_samples"] == 3
        # Residuals: -0.2, 0.0, -0.5 → mean -0.2333
        assert out["mean_residual_c"] == pytest.approx(-0.23333, abs=1e-4)
        # Cohen's d = mean / std
        assert out["cohens_d"] is not None
        assert out["t_stat"] is not None
        assert out["p_value"] is not None

    def test_zero_residuals_when_intervention_equals_identity(self):
        intervention = {str(s): _intervention_record(-1.2) for s in (1, 2, 3, 4, 5)}
        identity = {s: _identity_record(-1.2) for s in (1, 2, 3, 4, 5)}
        out = _MOD.compute_residuals(intervention, identity)
        assert out["mean_residual_c"] == pytest.approx(0.0, abs=1e-9)
        # std = 0 → no t-test; t and p should be NaN, d None
        assert out["t_stat"] != out["t_stat"]  # NaN check
        assert out["cohens_d"] is None

    def test_skips_samples_missing_from_identity(self):
        # Sample 9 is missing from identity (face det fails)
        intervention = {
            "1": _intervention_record(-1.0),
            "9": _intervention_record(-1.0),
        }
        identity = {1: _identity_record(-1.0)}  # 9 absent
        out = _MOD.compute_residuals(intervention, identity)
        assert out["n_samples"] == 1
        assert 1 in out["per_sample"]
        assert 9 not in out["per_sample"]

    def test_empty_when_no_overlap(self):
        intervention = {"1": _intervention_record(-1.0)}
        identity = {2: _identity_record(-1.0)}
        out = _MOD.compute_residuals(intervention, identity)
        assert out["n_samples"] == 0
        assert out["mean_residual_c"] is None
        assert out["t_stat"] is None
        assert out["p_value"] is None

    def test_skips_records_without_delta_c(self):
        intervention = {
            "1": {"sync_c": 5.0},  # no delta_c
            "2": _intervention_record(-1.0),
        }
        identity = {
            1: _identity_record(-1.0),
            2: _identity_record(-1.0),
        }
        out = _MOD.compute_residuals(intervention, identity)
        assert out["n_samples"] == 1
        assert 2 in out["per_sample"]


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_significant_below_alpha(self):
        assert _MOD.classify(0.001) == "significant"
        assert _MOD.classify(0.0499) == "significant"

    def test_not_significant_at_or_above_alpha(self):
        assert _MOD.classify(0.05) == "not_significant"
        assert _MOD.classify(0.5) == "not_significant"
        assert _MOD.classify(0.9) == "not_significant"

    def test_none_returns_no_data(self):
        assert _MOD.classify(None) == "no_data"

    def test_custom_alpha(self):
        assert _MOD.classify(0.04, alpha=0.05) == "significant"
        assert _MOD.classify(0.04, alpha=0.01) == "not_significant"


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_cli_writes_summary_and_classifies(self, tmp_path, monkeypatch):
        # Identity: ~-1.16 mean (matches real data).
        # Tilt residual: small per-sample noise with mean +0.17 → not sig at n=9
        # (This mirrors the actual analysis output.)
        rng = np.random.default_rng(42)
        identity_deltas = {1: -1.2, 2: -1.0, 3: -1.4, 4: -1.1, 5: -1.0, 6: -1.0, 7: -1.3, 8: -1.2, 10: -1.2}
        # Per-sample tilt residual with realistic std-0.37 noise
        tilt_residuals = {s: 0.17 + float(rng.standard_normal() * 0.37) for s in identity_deltas}
        intervention_deltas = {
            "LUFS_match_tts_to_nat": identity_deltas,  # exactly identity → residual 0
            "spectral_tilt_match_tts_to_nat": {s: identity_deltas[s] + tilt_residuals[s] for s in identity_deltas},
            "dynamic_compression_tts_compress": {s: d - 0.04 for s, d in identity_deltas.items()},
            "dynamic_expansion_tts_expand": {s: d - 0.07 for s, d in identity_deltas.items()},
        }
        dose_p, ident_p = _make_synthetic_files(tmp_path, identity_deltas=identity_deltas, intervention_deltas=intervention_deltas)
        out_p = tmp_path / "summary.json"

        rc = 0
        try:
            monkeypatch.setattr(
                sys, "argv",
                ["24", "--dose", str(dose_p), "--identity", str(ident_p), "--output", str(out_p)],
            )
            _MOD.main()
        except SystemExit as e:
            rc = e.code or 0
        assert rc == 0
        assert out_p.exists()
        s = json.loads(out_p.read_text())
        assert "identity_baseline" in s
        assert "tts_source_residuals" in s
        assert set(s["tts_source_residuals"].keys()) == set(intervention_deltas.keys())
        # LUFS residual should be ~0, classified not_significant
        lufs_r = s["tts_source_residuals"]["LUFS_match_tts_to_nat"]
        assert lufs_r["mean_residual_c"] == pytest.approx(0.0, abs=1e-9)
        assert lufs_r["significance_class"] == "not_significant"
        # Tilt residual should be small (mean ~0.17 by construction, but
        # per-sample noise lifts/lower it), classified not_significant at n=9
        tilt_r = s["tts_source_residuals"]["spectral_tilt_match_tts_to_nat"]
        assert -1.0 < tilt_r["mean_residual_c"] < 1.0
        assert tilt_r["significance_class"] == "not_significant"

    def test_cli_handles_sample_9_missing_from_identity(self, tmp_path, monkeypatch):
        # Identity missing sample 9 (face det); intervention has 9 — should skip
        identity_deltas = {1: -1.2, 2: -1.0, 3: -1.4}
        intervention_deltas = {
            "LUFS_match_tts_to_nat": {1: -1.2, 2: -1.0, 3: -1.4, 9: -1.5},  # 9 missing from identity
        }
        dose_p, ident_p = _make_synthetic_files(tmp_path, identity_deltas=identity_deltas, intervention_deltas=intervention_deltas)
        out_p = tmp_path / "summary.json"
        monkeypatch.setattr(
            sys, "argv",
            ["24", "--dose", str(dose_p), "--identity", str(ident_p), "--output", str(out_p)],
        )
        _MOD.main()
        s = json.loads(out_p.read_text())
        lufs_r = s["tts_source_residuals"]["LUFS_match_tts_to_nat"]
        assert lufs_r["n_samples"] == 3  # 9 dropped
        assert 9 not in lufs_r["per_sample"]

    def test_cli_emits_natural_source_raw_section(self, tmp_path, monkeypatch):
        identity_deltas = {1: -1.0, 2: -1.2}
        intervention_deltas = {"LUFS_match_tts_to_nat": {1: -1.0, 2: -1.2}}
        natural_deltas = {"LUFS_match_nat_to_tts": {1: -0.5, 2: -0.3}}
        dose_p, ident_p = _make_synthetic_files(
            tmp_path,
            identity_deltas=identity_deltas,
            intervention_deltas=intervention_deltas,
            natural_deltas=natural_deltas,
        )
        out_p = tmp_path / "summary.json"
        monkeypatch.setattr(
            sys, "argv",
            ["24", "--dose", str(dose_p), "--identity", str(ident_p), "--output", str(out_p)],
        )
        _MOD.main()
        s = json.loads(out_p.read_text())
        assert "natural_source_raw" in s
        assert "LUFS_match_nat_to_tts" in s["natural_source_raw"]
        rec = s["natural_source_raw"]["LUFS_match_nat_to_tts"]
        assert rec["source"] == "natural"
        assert rec["raw_mean_delta_c"] == pytest.approx(-0.4, abs=1e-6)
        assert "identity subtraction" in rec["note"]