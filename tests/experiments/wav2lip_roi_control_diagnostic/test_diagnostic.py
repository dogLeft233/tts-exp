from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_roi_control_diagnostic import config
from scripts.experiments.wav2lip_roi_control_diagnostic.analysis import (
    _c_diagnostic,
    reconstruct_timing,
    summarize_curve,
)
from scripts.experiments.wav2lip_roi_control_diagnostic.common import (
    DiagnosticError,
    file_sha256,
)
from scripts.experiments.wav2lip_roi_control_diagnostic.validate import (
    _check_final_contract,
    _check_pair,
    _ci_gate,
    _curve,
)


def _matrix(min_index: int = 16, gap: float = 0.02, rows: int = 5) -> np.ndarray:
    values = np.full(31, 3.0, dtype=np.float64)
    values[min_index] = 1.0
    values[15 if min_index != 15 else 14] = 1.0 + gap
    return np.repeat(values[None, :], rows, axis=0)


def test_synthetic_matrix_has_hand_checkable_offset_and_sync_metrics() -> None:
    matrix = _matrix()
    result = summarize_curve(matrix)
    assert result["min_index"] == 16
    assert result["offset"] == -1
    assert result["peak_gap"] == pytest.approx(0.02)
    assert result["sync_d"] == pytest.approx(1.0)
    assert result["sync_c"] == pytest.approx(2.0)
    assert result["clear"] is True
    assert _curve(matrix)["offset"] == result["offset"]


def test_peak_gap_and_boundary_are_strict() -> None:
    boundary = _matrix()
    boundary[:, 15] = np.nextafter(1.01, 1.0)
    assert summarize_curve(boundary)["clear"] is False
    assert summarize_curve(_matrix(gap=0.0101))["clear"] is True
    assert summarize_curve(_matrix(min_index=0))["clear"] is False
    assert summarize_curve(_matrix(min_index=30))["clear"] is False


def test_noninferiority_boundary_and_zero_crossing() -> None:
    assert _ci_gate({"ci95": [-0.100, 0.2]}, lower=-0.10, strict_lower=True) is False
    assert _ci_gate({"ci95": [-0.099, 0.2]}, lower=-0.10, strict_lower=True) is True
    assert _ci_gate({"ci95": [-0.05, 0.20]}, lower=-0.10, strict_lower=True) is True


def test_timing_sign_and_one_frame_offset_boundary() -> None:
    timing = reconstruct_timing(96_256, {arm: 147 for arm in config.VIDEO_ARMS})
    assert timing["plus_expected_video_response"] < 0
    assert timing["minus_expected_video_response"] > 0
    assert abs(1.0) <= config.OFFSET_TOLERANCE_FRAMES
    assert abs(1.0001) > config.OFFSET_TOLERANCE_FRAMES


def test_one_frame_pair_passes_but_swapped_sign_fails() -> None:
    curve = {"offset": 2, "clear": True}
    other = {"offset": 1, "clear": True}
    evidence = {"local": {"PLUS": curve, "MINUS": curve}}
    right = {"local": {"PLUS": other, "MINUS": other}}
    assert _check_pair(evidence, right, {"plus_expected_offset": 1.0, "minus_expected_offset": 1.0}, ("plus_expected_offset", "minus_expected_offset"))["passes"] is True
    assert _check_pair(evidence, right, {"plus_expected_offset": -1.0, "minus_expected_offset": -1.0}, ("plus_expected_offset", "minus_expected_offset"))["passes"] is False


def test_c_failure_flags_can_overlap() -> None:
    boundary = _matrix(min_index=0)
    boundary[:, 15] = np.nextafter(1.01, 1.0)
    bad = summarize_curve(boundary)
    evidence = {"local": {"PLUS": bad, "MINUS": bad}}
    timing = {"plus_expected_video_response": -2.8, "minus_expected_video_response": 2.8}
    result = _c_diagnostic(evidence, evidence, timing, {"passes": False})
    assert result["passes"] is False
    assert result["flags"] == {
        "baseline_invalid": True,
        "boundary_peak": True,
        "unclear_peak": True,
        "offset_error": True,
    }


def test_invalid_timing_input_is_rejected() -> None:
    with pytest.raises(DiagnosticError):
        reconstruct_timing(1, {arm: 1 for arm in config.VIDEO_ARMS})


def test_final_binding_rejects_tamper(tmp_path) -> None:
    for name in ("protocol.json", "audit.json", "diagnostics.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "result.md").write_text("result\n", encoding="utf-8")
    final = {
        "protocol_sha256": file_sha256(tmp_path / "protocol.json"),
        "audit_sha256": file_sha256(tmp_path / "audit.json"),
        "diagnostics_sha256": file_sha256(tmp_path / "diagnostics.json"),
        "result_sha256": file_sha256(tmp_path / "result.md"),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "bridge_executed": False,
        "training_authorized": False,
        "cross_model_spec_eligible": False,
        "generalization_established": False,
    }
    _check_final_contract(tmp_path, final)
    (tmp_path / "result.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DiagnosticError):
        _check_final_contract(tmp_path, final)
