from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.wav2lip_roi_peak_recheck import config
from scripts.experiments.wav2lip_roi_peak_recheck.analysis import (
    _c_pass,
    _peak,
    analyze,
)
from scripts.experiments.wav2lip_roi_peak_recheck.common import (
    RecheckError,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.wav2lip_roi_peak_recheck.runner import _result_markdown
from scripts.experiments.wav2lip_roi_peak_recheck.validate import (
    _comparison,
    check_cell_keys,
    validate_stored_arrays,
)
from scripts.experiments.wav2lip_roi_peak_recheck.worker import pairwise_distance


def test_distance_direction_epsilon_and_zero_padding() -> None:
    visual = np.zeros((2, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    visual[0, 0] = 1.0
    audio[0, 0] = 1.0
    visual[1, 0] = 2.0
    audio[1, 0] = 7.0
    matrix = pairwise_distance(visual, audio)

    assert matrix.shape == (2, 31)
    assert int(np.argmin(matrix[0])) == 15
    assert int(np.argmin(matrix[1])) == 14
    zero_expected = np.sqrt((1.0 + config.EPSILON) ** 2 + (config.EMBEDDING_DIM - 1) * config.EPSILON**2)
    assert matrix[0, 0] == pytest.approx(zero_expected, rel=1e-6)
    same_expected = np.sqrt(config.EMBEDDING_DIM * config.EPSILON**2)
    assert matrix[0, 15] == pytest.approx(same_expected, rel=1e-6)


def test_peak_gap_is_strict_and_boundary_is_not_clear() -> None:
    exact = np.ones(31, dtype=np.float64)
    exact[15] = 0.0
    exact[14] = config.PEAK_GAP_THRESHOLD
    assert _peak(exact)["clear"] is False

    clear = exact.copy()
    clear[14] = config.PEAK_GAP_THRESHOLD + 1e-6
    assert _peak(clear)["clear"] is True

    boundary = clear.copy()
    boundary[0] = -1.0
    assert _peak(boundary)["offset"] == config.VSHIFT
    assert _peak(boundary)["clear"] is False


def test_c_pass_includes_inherited_baseline() -> None:
    evidence = {"clear": True, "residual": 0.0}
    pair = {"gn": evidence, "gw": evidence}
    passed, reasons = _c_pass(pair, pair, baseline_pass=False)
    assert passed is False
    assert reasons == ["baseline_invalid"]


def test_analyze_fixture_covers_whole_c_call_chain(tmp_path: Path) -> None:
    records = []
    score_index = {}
    matrix_cache = {}
    score_rows = []
    protocol_records = []
    for sample_id in config.FAIL_IDS + config.PASS_IDS:
        source_group = sample_id.split("_")[1]
        masks = {"plus_rows": [0], "minus_rows": [1], "d_by_row": {"0": 0.0, "1": 0.0}}
        records.append({"sample_id": sample_id, "source_group": source_group, "masks": masks})
        protocol_records.append({"sample_id": sample_id, "source_group": source_group, "masks": masks, "baseline_g_n_n": {"passes": True}})
        for video_arm in config.VIDEO_ARMS:
            old = np.ones((2, 31), dtype=np.float64)
            old[0, 15] = 0.0
            old[1, 15] = 0.0
            if sample_id in config.FAIL_IDS:
                new_offset_index = 13
                old[0, new_offset_index] = 0.0
                old[1, new_offset_index] = 0.0
            else:
                new_offset_index = 15
            new = old.astype(np.float32)
            new[0, new_offset_index] = 0.0
            new[1, new_offset_index] = 0.0
            matrix = new.astype(np.float64)
            matrix_hash = f"{sample_id}-{video_arm}"
            matrix_cache[matrix_hash] = old
            score_index[(sample_id, video_arm, config.AUDIO_ARM, config.REPEAT)] = {
                "matrix_sha256": matrix_hash,
                "matrix": str(tmp_path / f"old-{sample_id}-{video_arm}.npy"),
            }
            new_path = tmp_path / f"new-{sample_id}-{video_arm}.npy"
            np.save(new_path, matrix.astype(np.float32), allow_pickle=False)
            score_rows.append({"sample_id": sample_id, "video_arm": video_arm, "audio_arm": "N", "repeat": False, "matrix": str(new_path)})

    parent = {"records": records, "score_index": score_index, "matrix_cache": matrix_cache}
    protocol = {"selected_records": protocol_records}
    result = analyze(parent, protocol, score_rows)
    assert result["decision"] == "PEAKS_REPRODUCED"
    assert result["historical_failure_reproduced"] == 8
    assert result["historical_pass_reproduced"] == 2
    assert all(row["c_decision_match"] for row in result["per_record"])


def test_missing_or_duplicate_cell_is_rejected() -> None:
    rows = [{"sample_id": config.FAIL_IDS[0], "video_arm": "G_N", "audio_arm": "N", "repeat": False}]
    with pytest.raises(RecheckError, match="score cell set mismatch"):
        check_cell_keys(rows, [config.FAIL_IDS[0]])
    duplicate = rows * 2
    with pytest.raises(RecheckError, match="duplicate"):
        check_cell_keys(duplicate, [config.FAIL_IDS[0]])


def test_validator_rejects_tampered_matrix() -> None:
    visual = np.zeros((2, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    matrix = pairwise_distance(visual, audio)
    matrix[0, 0] += 0.01
    with pytest.raises(RecheckError, match="stored matrix differs"):
        validate_stored_arrays(visual, audio, matrix)


def test_comparison_reports_location_and_both_values() -> None:
    old = np.zeros((2, 31), dtype=np.float64)
    new = old.copy()
    old[1, 4] = 1.0
    new[1, 4] = 1.25
    comparison = _comparison(old, new, 0.001, "G_N")
    assert comparison["argmax"] == {"row": 1, "column": 4}
    assert comparison["old_value"] == 1.0
    assert comparison["new_value"] == 1.25
    assert comparison["pass"] is False


def test_result_markdown_uses_nested_segment_offsets() -> None:
    evidence = {
        "gn": {"offset": -4, "actual": -4.0, "expected": -2.85, "residual": -1.15},
        "gw": {"offset": 0, "actual": -4.0, "expected": -2.85, "residual": -1.15},
    }
    analysis = {
        "decision": "PEAKS_REPRODUCED",
        "record_count": 1,
        "expected_record_count": 10,
        "score_cell_count": 2,
        "expected_score_cell_count": 20,
        "historical_failure_reproduced": 1,
        "historical_failure_count": 8,
        "historical_pass_reproduced": 0,
        "historical_pass_count": 2,
        "matrix_max_abs_difference": 0.0,
        "curve_max_abs_difference": 0.0,
        "per_record": [
            {
                "sample_id": "sample",
                "source_group": "group",
                "historical_label": "historical_fail",
                "historical": {"PLUS": {"gn": {"offset": -4}, "gw": {"offset": 0}}, "MINUS": {"gn": {"offset": -4}, "gw": {"offset": 0}}},
                "new": {"PLUS": evidence, "MINUS": evidence, "c_pass": False},
            }
        ],
    }
    report = _result_markdown(analysis)
    assert "-4.000 / -2.850 / -1.150" in report


def test_self_hashed_final_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "final.json"
    write_self_hashed_json(path, {"status": "complete", "diagnostic_decision": "PEAKS_REPRODUCED"})
    path.write_text(path.read_text(encoding="utf-8").replace("PEAKS_REPRODUCED", "SCORER_MISMATCH"), encoding="utf-8")
    with pytest.raises(RecheckError, match="self-hash mismatch"):
        verify_self_hashed_json(path)
