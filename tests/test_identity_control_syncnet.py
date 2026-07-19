"""Unit tests for scripts/23_identity_control_syncnet.py.

These tests cover the identity intervention transform, the verdict
classification thresholds, and the registry plumbing. End-to-end
Ditto + SyncNet execution is exercised only on the server.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "script_23", _REPO / "scripts" / "23_identity_control_syncnet.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


# ---------------------------------------------------------------------------
# Identity intervention
# ---------------------------------------------------------------------------


class TestIdentityIntervention:
    def test_builds_bilateral_identity_interventions(self):
        ivs = _MOD._build_identity_interventions()
        assert len(ivs) == 2
        assert [iv.name for iv in ivs] == ["identity_natural", "identity_tts"]
        assert [iv.source for iv in ivs] == ["natural", "tts"]
        assert [iv.baseline_cond for iv in ivs] == ["natural_raw", "tts_raw"]
        assert all(iv.expected_sync_direction == "no change" for iv in ivs)

    def test_identity_transform_returns_input_unchanged(self):
        rng = np.random.default_rng(7)
        y_tts = rng.standard_normal(16000).astype(np.float32) * 0.1
        y_nat = rng.standard_normal(16000).astype(np.float32) * 0.1
        iv = next(
            item for item in _MOD._build_identity_interventions()
            if item.name == "identity_tts"
        )
        y_out = iv.transform(y_tts, y_nat, sr=16000, sid=1)
        # Must be bit-identical to the TTS input.
        assert y_out is y_tts or np.array_equal(y_out, y_tts)
        assert np.all(np.isfinite(y_out))

    def test_identity_transform_preserves_dtype_and_shape(self):
        rng = np.random.default_rng(2)
        y_tts = rng.standard_normal(8000).astype(np.float32)
        iv = next(
            item for item in _MOD._build_identity_interventions()
            if item.name == "identity_tts"
        )
        y_out = iv.transform(y_tts, np.zeros_like(y_tts), sr=16000, sid=3)
        assert y_out.shape == y_tts.shape
        assert y_out.dtype == y_tts.dtype

    def test_identity_transform_ignores_natural_audio(self):
        """Identity must not peek at y_nat to decide the output."""
        rng = np.random.default_rng(11)
        y_tts = rng.standard_normal(4000).astype(np.float32)
        y_nat_a = np.zeros(4000, dtype=np.float32)
        y_nat_b = rng.standard_normal(4000).astype(np.float32) * 0.5
        iv = next(
            item for item in _MOD._build_identity_interventions()
            if item.name == "identity_tts"
        )
        out_a = iv.transform(y_tts, y_nat_a, sr=16000, sid=1)
        out_b = iv.transform(y_tts, y_nat_b, sr=16000, sid=1)
        assert np.array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Reuse of script-22 machinery
# ---------------------------------------------------------------------------


class TestScript22MachineryImport:
    def test_intervention_class_reused(self):
        """Script 23 must reuse Intervention from script 22, not redefine it."""
        assert _MOD.Intervention is _MOD._s22.Intervention

    def test_pipeline_functions_imported(self):
        for name in ("run_intervention_pipeline", "load_baseline_results", "aggregate_deltas"):
            assert getattr(_MOD, name) is getattr(_MOD._s22, name)

    def test_identity_intervention_is_compatible_with_pipeline(self):
        """Identity intervention must be a valid Intervention instance."""
        from dataclasses import is_dataclass
        iv = next(
            item for item in _MOD._build_identity_interventions()
            if item.name == "identity_tts"
        )
        assert is_dataclass(iv)
        # The transform must accept the (y_tts, y_nat, sr, sid) signature.
        rng = np.random.default_rng(0)
        y_tts = rng.standard_normal(1000).astype(np.float32)
        y_nat = rng.standard_normal(1000).astype(np.float32)
        result = iv.transform(y_tts, y_nat, 16000, 1)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Verdict thresholds (interpreted here without running the full script)
# ---------------------------------------------------------------------------


class TestVerdictThresholds:
    """Sanity-check the interpretation thresholds encoded in main().

    The thresholds themselves are embedded in main(); these tests document
    the boundaries we expect the script to enforce. If we ever refactor the
    thresholds into a standalone function, these tests should move to that
    function.
    """

    @staticmethod
    def _classify(mean_dc):
        if mean_dc is None:
            return "NO_DATA"
        if abs(mean_dc) < 0.2:
            return "PIPELINE_CLEAN"
        if mean_dc <= -0.5:
            return "PIPELINE_CONFOUNDS_DOSE_RESPONSE"
        return "AMBIGUOUS"

    def test_clean_when_within_0_2(self):
        assert self._classify(0.0) == "PIPELINE_CLEAN"
        assert self._classify(0.19) == "PIPELINE_CLEAN"
        assert self._classify(-0.19) == "PIPELINE_CLEAN"

    def test_ambiguous_in_middle_band(self):
        assert self._classify(0.25) == "AMBIGUOUS"
        assert self._classify(-0.3) == "AMBIGUOUS"
        assert self._classify(-0.49) == "AMBIGUOUS"

    def test_pipeline_confounds_when_drop_above_0_5(self):
        assert self._classify(-0.5) == "PIPELINE_CONFOUNDS_DOSE_RESPONSE"
        assert self._classify(-1.05) == "PIPELINE_CONFOUNDS_DOSE_RESPONSE"
        assert self._classify(-2.0) == "PIPELINE_CONFOUNDS_DOSE_RESPONSE"

    def test_no_data_for_none(self):
        assert self._classify(None) == "NO_DATA"


# ---------------------------------------------------------------------------
# Aggregation integration (uses real script-22 aggregate_deltas)
# ---------------------------------------------------------------------------


class TestAggregationWithIdentity:
    def _identity_iv(self):
        return next(
            item for item in _MOD._build_identity_interventions()
            if item.name == "identity_tts"
        )

    def test_aggregate_identity_zero_deltas_when_results_match_baseline(self):
        iv = self._identity_iv()
        results = {iv.name: {1: {"sync_c": 6.0, "sync_d": 7.0}}}
        baselines = {"tts_raw": {1: {"sync_c": 6.0, "sync_d": 7.0}}}
        out = _MOD.aggregate_deltas(results, baselines, [iv])
        s = out[iv.name]
        assert s["n_samples"] == 1
        assert s["mean_delta_c"] == pytest.approx(0.0)
        assert s["mean_delta_d"] == pytest.approx(0.0)

    def test_aggregate_identity_negative_delta_when_pipeline_drops(self):
        iv = self._identity_iv()
        results = {iv.name: {1: {"sync_c": 5.0, "sync_d": 8.0}}}
        baselines = {"tts_raw": {1: {"sync_c": 6.0, "sync_d": 7.0}}}
        out = _MOD.aggregate_deltas(results, baselines, [iv])
        s = out[iv.name]
        assert s["mean_delta_c"] == pytest.approx(-1.0)
        assert s["mean_delta_d"] == pytest.approx(1.0)

    def test_aggregate_excludes_samples_without_baseline(self):
        iv = self._identity_iv()
        results = {iv.name: {
            1: {"sync_c": 5.0, "sync_d": 8.0},
            9: {"sync_c": 4.5, "sync_d": 9.0},
        }}
        baselines = {"tts_raw": {1: {"sync_c": 6.0, "sync_d": 7.0}}}  # 9 missing
        out = _MOD.aggregate_deltas(results, baselines, [iv])
        assert out[iv.name]["n_samples"] == 1
        assert "9" not in out[iv.name]["per_sample"]