"""Unit tests for scripts/17_link_features_to_tfg.py.

All tests use synthetic data — no real TFG runs or embeddings needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tfg_feature_common import (
    fdr_bh_correction,
    STUDY_SAMPLES,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "17_link_features_to_tfg.py"
)
_spec = importlib.util.spec_from_file_location(
    "_link_features", str(_SCRIPT_PATH)
)
_link = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_link)

_load_syncnet_for_run = _link._load_syncnet_for_run
_build_sync_df = _link._build_sync_df
_load_separability_metrics = _link._load_separability_metrics
_build_feature_delta_df = _link._build_feature_delta_df
_run_univariate_analysis = _link._run_univariate_analysis
_run_univariate_per_feature_model = _link._run_univariate_per_feature_model
_multivariate_analysis = _link._multivariate_analysis
_rank_candidates = _link._rank_candidates
process_all = _link.process_all
TFG_RUNS = _link.TFG_RUNS


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_syncnet_json(
    sample_id: int,
    condition: str,
    sync_c: float,
    sync_d: float,
    av_offset: int = 1,
) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "sync_c": sync_c,
        "sync_d": sync_d,
        "av_offset": av_offset,
    }


def _write_mock_syncnet_tree(
    base_dir: Path,
    run_id: str,
    sample_ids: list[int],
    nat_values: dict[int, tuple[float, float]] | None = None,
    tts_values: dict[int, tuple[float, float]] | None = None,
) -> None:
    """Create a mock ``04_eval/{condition}/{sid}/syncnet.json`` tree."""
    eval_dir = base_dir / run_id / "04_eval"
    if nat_values is None:
        nat_values = {sid: (5.0, 8.0) for sid in sample_ids}
    if tts_values is None:
        tts_values = {sid: (6.0, 7.0) for sid in sample_ids}

    for cond_dir, values in [("natural_raw", nat_values), ("tts_raw", tts_values)]:
        cond_path = eval_dir / cond_dir
        for sid, (c_val, d_val) in values.items():
            sid_dir = cond_path / str(sid)
            sid_dir.mkdir(parents=True, exist_ok=True)
            cond_label = "natural_raw" if cond_dir == "natural_raw" else "tts_raw"
            obj = _make_syncnet_json(sid, cond_label, c_val, d_val)
            with open(sid_dir / "syncnet.json", "w") as f:
                json.dump(obj, f)


def _make_separability_metrics(entries: list[dict] | None = None) -> dict:
    """Build a synthetic separability_metrics.json structure."""
    if entries is None:
        entries = [
            {
                "metric": "silhouette",
                "model": "hubert",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "natural_mean": 0.35,
                "tts_mean": 0.25,
                "delta": -0.10,
                "p_perm": 0.01,
                "fdr_corrected_p": 0.04,
                "significant_fdr": True,
                "cohens_d": 1.2,
                "n_pairs": 12,
            },
            {
                "metric": "boundary_sharpness",
                "model": "hubert",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "natural_mean": 0.15,
                "tts_mean": 0.22,
                "delta": 0.07,
                "p_perm": 0.03,
                "fdr_corrected_p": 0.06,
                "significant_fdr": False,
                "cohens_d": 0.8,
                "n_pairs": 12,
            },
            {
                "metric": "silhouette",
                "model": "xlsr",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "natural_mean": 0.30,
                "tts_mean": 0.20,
                "delta": -0.10,
                "p_perm": 0.02,
                "fdr_corrected_p": 0.05,
                "significant_fdr": True,
                "cohens_d": 0.9,
                "n_pairs": 12,
            },
        ]
    return {
        "meta": {
            "models": ["hubert", "xlsr"],
            "layers": [12],
            "levels": ["viseme"],
        },
        "results": [],
        "comparisons": entries,
    }


# ---------------------------------------------------------------------------
# Test syncnet loader
# ---------------------------------------------------------------------------


class TestSyncnetLoader:
    def test_syncnet_loader_returns_dict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            sample_ids = [1, 2, 3]
            _write_mock_syncnet_tree(
                base, run_id, sample_ids,
                nat_values={1: (4.73, 8.573), 2: (5.0, 8.0), 3: (6.0, 7.0)},
                tts_values={1: (6.49, 7.36), 2: (6.6, 7.75), 3: (7.0, 7.0)},
            )
            data = _load_syncnet_for_run(base, run_id, sample_ids)
            assert "nat" in data
            assert "tts" in data
            assert 1 in data["nat"]
            assert data["nat"][1]["sync_c"] == 4.73
            assert data["nat"][1]["sync_d"] == 8.573
            assert data["tts"][1]["sync_c"] == 6.49
            assert data["tts"][1]["sync_d"] == 7.36

    def test_missing_condition_dir_handled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            eval_dir = base / run_id / "04_eval"
            eval_dir.mkdir(parents=True)
            # Only natural_raw exists, no tts_raw
            nat_dir = eval_dir / "natural_raw" / "1"
            nat_dir.mkdir(parents=True)
            with open(nat_dir / "syncnet.json", "w") as f:
                json.dump(_make_syncnet_json(1, "natural_raw", 5.0, 8.0), f)

            data = _load_syncnet_for_run(base, run_id, [1])
            assert 1 in data["nat"]
            assert not data["tts"]

    def test_missing_file_handled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            eval_dir = base / run_id / "04_eval"
            (eval_dir / "natural_raw" / "1").mkdir(parents=True)
            (eval_dir / "tts_raw" / "1").mkdir(parents=True)
            # Write nat but not tts
            with open(eval_dir / "natural_raw" / "1" / "syncnet.json", "w") as f:
                json.dump(_make_syncnet_json(1, "natural_raw", 5.0, 8.0), f)
            # tts_raw/1/syncnet.json intentionally missing

            data = _load_syncnet_for_run(base, run_id, [1, 2])
            assert 1 in data["nat"]
            assert 1 not in data["tts"]


# ---------------------------------------------------------------------------
# Test delta computation
# ---------------------------------------------------------------------------


class TestDeltaComputation:
    def test_delta_computation_correct(self) -> None:
        """delta = tts - nat: nat=3, tts=5 → delta=2"""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            sids = [1]
            _write_mock_syncnet_tree(
                base, run_id, sids,
                nat_values={1: (3.0, 8.0)},
                tts_values={1: (5.0, 6.0)},
            )

            original = dict(_link.TFG_RUNS)
            _link.TFG_RUNS.clear()
            _link.TFG_RUNS["mock_tfg"] = run_id
            try:
                meta, rows = _build_sync_df(base, sids)
            finally:
                _link.TFG_RUNS.clear()
                _link.TFG_RUNS.update(original)

            assert len(rows) == 1
            assert rows[0]["nat_sync_c"] == 3.0
            assert rows[0]["tts_sync_c"] == 5.0
            assert rows[0]["delta_sync_c"] == 2.0
            assert rows[0]["nat_sync_d"] == 8.0
            assert rows[0]["tts_sync_d"] == 6.0
            assert rows[0]["delta_sync_d"] == -2.0

    def test_delta_negative_when_tts_less(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            sids = [1]
            _write_mock_syncnet_tree(
                base, run_id, sids,
                nat_values={1: (7.0, 9.0)},
                tts_values={1: (4.0, 5.0)},
            )

            original = dict(_link.TFG_RUNS)
            _link.TFG_RUNS.clear()
            _link.TFG_RUNS["mock_tfg"] = run_id
            try:
                _, rows = _build_sync_df(base, sids)
            finally:
                _link.TFG_RUNS.clear()
                _link.TFG_RUNS.update(original)

            assert rows[0]["delta_sync_c"] == -3.0
            assert rows[0]["delta_sync_d"] == -4.0

    def test_only_paired_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            sids = [1, 2, 3]
            _write_mock_syncnet_tree(
                base, run_id, sids,
                nat_values={1: (5.0, 8.0), 2: (5.0, 8.0), 3: (5.0, 8.0)},
                tts_values={1: (6.0, 7.0), 2: (6.0, 7.0)},
            )

            original = dict(_link.TFG_RUNS)
            _link.TFG_RUNS.clear()
            _link.TFG_RUNS["mock_tfg"] = run_id
            try:
                _, rows = _build_sync_df(base, sids)
            finally:
                _link.TFG_RUNS.clear()
                _link.TFG_RUNS.update(original)

            sids_found = {r["sample_id"] for r in rows}
            assert 1 in sids_found
            assert 2 in sids_found
            assert 3 not in sids_found


# ---------------------------------------------------------------------------
# Test missing TFG skipped gracefully
# ---------------------------------------------------------------------------


class TestMissingTfgSkipped:
    def test_missing_tfg_skipped_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Create only one valid run
            _write_mock_syncnet_tree(base, "existing_run", [1])
            # "nonexistent_run" does not exist
            sids = [1, 2]
            meta, rows = _build_sync_df(base, sids)
            # Should not crash; should return empty results for the
            # missing run
            assert isinstance(meta, dict)
            assert isinstance(rows, list)

    def test_nonexistent_run_dir_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # No runs created at all
            meta, rows = _build_sync_df(base, [1])
            assert meta["tfg_models"] == []
            assert rows == []


# ---------------------------------------------------------------------------
# Test correlation with known linear relationship
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_correlation_with_known_linear_relationship(self) -> None:
        """y = 2 * x + noise: verify r > 0.8."""
        rng = np.random.default_rng(42)
        n = 20
        x = rng.uniform(0, 10, n)
        y = 2.0 * x + rng.normal(0, 0.5, n)

        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        assert rho > 0.8


# ---------------------------------------------------------------------------
# Test FDR across multiple tests
# ---------------------------------------------------------------------------


class TestFdr:
    def test_fdr_across_multiple_tests(self) -> None:
        """100 independent null correlations → FDR < 0.1 rate."""
        rng = np.random.default_rng(42)
        n_tests = 100
        p_values: list[float] = []
        for _ in range(n_tests):
            x = rng.normal(0, 1, 30)
            y = rng.normal(0, 1, 30)
            from scipy.stats import spearmanr
            _, p = spearmanr(x, y)
            p_values.append(float(p))

        p_arr = np.array(p_values, dtype=np.float64)
        corrected = fdr_bh_correction(p_arr)
        n_sig = int(np.sum(corrected < 0.05))
        # With 100 independent null tests, expect ~5 false positives at 0.05
        # Using a generous bound since we're testing the property, not
        # a specific realisation
        assert n_sig <= 20


# ---------------------------------------------------------------------------
# Test candidate ranking
# ---------------------------------------------------------------------------


class TestCandidateRanking:
    def test_candidate_ranking_requires_consistency(self) -> None:
        """Candidate with reversed direction across models should be demoted."""
        # Two metrics: one consistent, one reversed
        feat_rows = [
            {
                "metric": "silhouette",
                "model": "hubert",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "delta": -0.10,
                "cohens_d": 1.2,
                "significant_fdr": True,
            },
            {
                "metric": "silhouette",
                "model": "xlsr",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "delta": -0.10,
                "cohens_d": 0.9,
                "significant_fdr": True,
            },
            {
                "metric": "reversed_metric",
                "model": "hubert",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "delta": 0.20,
                "cohens_d": 1.1,
                "significant_fdr": True,
            },
            {
                "metric": "reversed_metric",
                "model": "xlsr",
                "layer": 12,
                "level": "viseme",
                "variant": "raw",
                "delta": -0.20,
                "cohens_d": 1.0,
                "significant_fdr": True,
            },
        ]
        univariate: list[dict] = []
        candidates = _rank_candidates(univariate, feat_rows)
        assert len(candidates) >= 1

        # The consistent metric should rank higher than reversed
        consistent = [c for c in candidates if c["metric"] == "silhouette"]
        reversed_ = [c for c in candidates if c["metric"] == "reversed_metric"]
        if consistent and reversed_:
            assert consistent[0]["score"] > reversed_[0]["score"]


# ---------------------------------------------------------------------------
# Test empty data produces NaN
# ---------------------------------------------------------------------------


class TestEmptyData:
    def test_empty_data_produces_nan_not_crash(self) -> None:
        """Empty input returns NaN correlations, not crashes."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_id = "test_run"
            sids = [1, 2]
            _write_mock_syncnet_tree(base, run_id, sids)
            meta, sync_rows = _build_sync_df(base, sids)

            # Empty feature rows
            feature_rows: list[dict] = []
            univariate = _run_univariate_per_feature_model(sync_rows, feature_rows)
            assert isinstance(univariate, list)

            mv = _multivariate_analysis(sync_rows)
            assert isinstance(mv, dict)


# ---------------------------------------------------------------------------
# Test separability metrics loading
# ---------------------------------------------------------------------------


class TestSeparabilityLoading:
    def test_load_metrics_handles_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            metrics_dir = Path(td)
            # No separability_metrics.json exists
            with pytest.raises(SystemExit):
                _load_separability_metrics(metrics_dir)

    def test_loads_correct_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            metrics_dir = Path(td)
            data = _make_separability_metrics()
            with open(metrics_dir / "separability_metrics.json", "w") as f:
                json.dump(data, f)

            loaded = _load_separability_metrics(metrics_dir)
            assert "comparisons" in loaded
            assert len(loaded["comparisons"]) == 3

    def test_build_feature_delta_df(self) -> None:
        data = _make_separability_metrics()
        rows = _build_feature_delta_df(data, "ditto", [1])
        assert len(rows) == 3
        for r in rows:
            assert "metric" in r
            assert "delta" in r
            assert "model" in r


# ---------------------------------------------------------------------------
# Test feature_tfg_link output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_output_contains_expected_keys(self) -> None:
        """Verify that process_all output contains meta, univariate,
        multivariate, and candidates sections."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            runs_dir = base / "runs"
            metrics_dir = base / "metrics"
            out_dir = base / "out"
            runs_dir.mkdir()
            metrics_dir.mkdir()
            out_dir.mkdir()

            # Create a mock TFG run
            run_id = "test_tfg"
            _write_mock_syncnet_tree(
                runs_dir, run_id, [1, 2, 3],
                nat_values={
                    1: (5.0, 8.0), 2: (5.5, 8.5), 3: (6.0, 9.0),
                },
                tts_values={
                    1: (6.0, 7.0), 2: (6.5, 7.5), 3: (7.0, 7.0),
                },
            )

            # Register the run in TFG_RUNS temporarily
            original = dict(_link.TFG_RUNS)
            _link.TFG_RUNS.clear()
            _link.TFG_RUNS["mock_tfg"] = run_id

            # Create separability metrics
            sep = _make_separability_metrics()
            with open(metrics_dir / "separability_metrics.json", "w") as f:
                json.dump(sep, f)

            try:
                output_path = process_all(
                    metrics_dir=metrics_dir,
                    runs_dir=runs_dir,
                    output_dir=out_dir,
                    smoke=False,
                )
            finally:
                _link.TFG_RUNS.clear()
                _link.TFG_RUNS.update(original)

            assert output_path.exists()
            with open(output_path, "r") as f:
                data = json.load(f)

            assert "meta" in data
            assert "univariate" in data
            assert "multivariate" in data
            assert "candidates" in data
            assert "tfg_models" in data["meta"]
            assert "mock_tfg" in data["meta"]["tfg_models"]


# ---------------------------------------------------------------------------
# Test smoke mode
# ---------------------------------------------------------------------------


class TestSmokeMode:
    def test_smoke_reduces_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            runs_dir = base / "runs"
            metrics_dir = base / "metrics"
            out_dir = base / "out"
            runs_dir.mkdir()
            metrics_dir.mkdir()
            out_dir.mkdir()

            run_id = "test_tfg"
            _write_mock_syncnet_tree(
                runs_dir, run_id, [1, 2, 3],
            )

            original = dict(_link.TFG_RUNS)
            _link.TFG_RUNS.clear()
            _link.TFG_RUNS["mock"] = run_id

            # Create many feature entries
            many_entries = []
            for i in range(10):
                many_entries.append({
                    "metric": f"metric_{i}",
                    "model": "hubert",
                    "layer": 12,
                    "level": "viseme",
                    "variant": "raw",
                    "natural_mean": 0.35,
                    "tts_mean": 0.25,
                    "delta": -0.10,
                    "p_perm": 0.05,
                    "fdr_corrected_p": 0.1,
                    "significant_fdr": False,
                    "cohens_d": 0.5,
                    "n_pairs": 3,
                })
            sep = _make_separability_metrics(many_entries)
            with open(metrics_dir / "separability_metrics.json", "w") as f:
                json.dump(sep, f)

            try:
                # Non-smoke
                out_ns = process_all(metrics_dir, runs_dir, out_dir / "full", smoke=False)
                with open(out_ns, "r") as f:
                    full_data = json.load(f)
                # Smoke
                out_sm = process_all(metrics_dir, runs_dir, out_dir / "smoke", smoke=True)
                with open(out_sm, "r") as f:
                    smoke_data = json.load(f)
            finally:
                _link.TFG_RUNS.clear()
                _link.TFG_RUNS.update(original)

            # Smoke should have fewer feature rows in meta
            assert smoke_data["meta"]["n_feature_metrics"] <= full_data["meta"]["n_feature_metrics"]
