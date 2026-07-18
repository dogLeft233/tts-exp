"""Tests for 30B/C/D/E Wav2Sem analysis modules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "30_analyze_english_wav2sem.py"
_spec = importlib.util.spec_from_file_location("_english_wav2sem", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def _stub_separability_payload():
    return [
        {
            "model": "hubert", "layer": 11, "condition": "natural", "level": "viseme",
            "metric": "segment_stability", "value": 0.30,
        },
        {
            "model": "hubert", "layer": 11, "condition": "tts", "level": "viseme",
            "metric": "segment_stability", "value": 0.31,
        },
        {
            "model": "hubert", "layer": 11, "condition": "natural", "level": "viseme",
            "metric": "segment_stability", "value": 0.32,
        },
        {
            "model": "hubert", "layer": 11, "condition": "tts", "level": "viseme",
            "metric": "segment_stability", "value": 0.33,
        },
        {
            "model": "hubert", "layer": 11, "condition": "natural", "level": "viseme",
            "metric": "segment_stability", "value": 0.29,
        },
        {
            "model": "hubert", "layer": 11, "condition": "tts", "level": "viseme",
            "metric": "segment_stability", "value": 0.34,
        },
    ]


def _stub_fd_payload():
    return [
        {
            "sample_id": sid, "condition": condition,
            "fp_metrics": {"silhouette": 0.05, "intra_class_dist": 0.80,
                            "boundary_sharpness": 0.30, "segment_stability": 0.31,
                            "fisher": 1.10},
            "fd_zero_metrics": {"silhouette": 0.06, "intra_class_dist": 0.78,
                                  "boundary_sharpness": 0.31, "segment_stability": 0.32,
                                  "fisher": 1.12},
            "fd_random_metrics": {"silhouette": 0.07, "intra_class_dist": 0.76,
                                    "boundary_sharpness": 0.33, "segment_stability": 0.33,
                                    "fisher": 1.15},
        }
        for sid in range(1, 4)
        for condition in ("natural", "tts")
    ]


def test_analyze_separability_runs():
    assert hasattr(_mod, "analyze_separability"), "analyze_separability not implemented"
    payload = _stub_separability_payload()
    summary = _mod.analyze_separability(payload)
    assert "h1_verdict" in summary
    assert summary["h1_verdict"] in ("PASS", "FAIL", "TREND", "INSUFFICIENT_DATA")


def test_analyze_fd_gain_compute_h2_verdict():
    assert hasattr(_mod, "analyze_fd_gain"), "analyze_fd_gain not implemented"
    payload = _stub_fd_payload()
    summary = _mod.analyze_fd_gain(payload)
    assert "h2_verdict" in summary
    assert summary["h2_verdict"] in (
        "PASS", "FAIL", "TREND", "M3_FC_ARTIFACT", "INSUFFICIENT_DATA",
    )


def test_analyze_tfg_correlation_returns_spearman():
    assert hasattr(_mod, "analyze_tfg_correlation"), "analyze_tfg_correlation not implemented"
    separability_values = np.array([0.30, 0.31, 0.32, 0.33])
    sync_c_deltas = np.array([+0.5, +0.4, +0.6, +0.3])
    syncnet_path = Path("/dev/null")
    summary = _mod.analyze_tfg_correlation(
        separability_values=separability_values,
        sync_c_deltas=sync_c_deltas,
        syncnet_results_path=syncnet_path,
    )
    assert "h3_verdict" in summary
    assert "spearman_rho" in summary


def test_render_full_report_present():
    """render_full_report must exist and produce non-empty markdown."""
    assert hasattr(_mod, "render_full_report"), "render_full_report not implemented"
    output_path = Path("/tmp/_test_render_full_report.md")
    _mod.render_full_report(
        gate1_summary={"verdict": "FAIL_EVALUATOR_OR_INTERACTION_DRIVEN",
                        "primary": {"sync_c": {"mean": 0.16, "holm_p": 0.571}}},
        separability_summary={"h1_verdict": "FAIL", "cohens_d": -0.2},
        oracle_fs_summary={"n_pairs": 13, "mean_cosine_cls_nt": 0.97, "n_degenerate": 0},
        fd_gain_summary={"h2_verdict": "FAIL", "favorable_counts": {}, "fc_artifact_suspected": False},
        tfg_correlation_summary={"h3_verdict": "FAIL", "spearman_rho": 0.1, "permutation_p": 0.5},
        output_path=output_path,
    )
    assert output_path.exists()
    md = output_path.read_text()
    assert "English Wav2Sem Analysis Report" in md
    assert "H3" in md
    output_path.unlink()
