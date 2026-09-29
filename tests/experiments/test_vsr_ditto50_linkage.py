"""CPU tests for the Ditto-50 linkage protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import vsr_ditto50_linkage as runner
from scripts.experiments import vsr_ditto50_metrics as metrics


def test_decoys_are_unique_and_stable_before_qc() -> None:
    rows = [
        {"id": 1, "normalized_text": "abcd", "target_token_ids": [1, 2, 3, 4]},
        {"id": 2, "normalized_text": "xy", "target_token_ids": [1, 2]},
        {"id": 3, "normalized_text": "abcd", "target_token_ids": [1, 2, 3, 4]},
        {"id": 4, "normalized_text": "abcde", "target_token_ids": [1, 2, 3, 4, 5]},
        {"id": 5, "normalized_text": "abcdef", "target_token_ids": [1, 2, 3, 4, 5, 6]},
        {"id": 6, "normalized_text": "q", "target_token_ids": None},
    ]
    decoys = metrics.freeze_decoys(rows, count=3)
    assert [item["id"] for item in decoys["1"]] == [4, 2, 5]
    assert all(item["id"] != 3 for item in decoys["1"])
    assert decoys["6"] == []


def test_syncnet_parser_requires_one_complete_signed_group() -> None:
    parsed = runner.parse_syncnet_output("Confidence: 1.25\nMin dist: 0.75\nAV offset: -3\n")
    assert parsed == {"sync_c": 1.25, "sync_d": 0.75, "av_offset": -3}
    with pytest.raises(ValueError):
        runner.parse_syncnet_output("Confidence: 1\nMin dist: 2\n")
    with pytest.raises(ValueError):
        runner.parse_syncnet_output("Confidence: 1\nMin dist: 2\nAV offset: 1\nConfidence: 2\n")


def test_longest_track_ties_choose_lowest_index() -> None:
    tracks = [{"frame": np.arange(3)}, {"frame": np.arange(3) + 2}, {"frame": np.arange(2)}]
    index, selected = runner.select_longest_track(tracks)
    assert index == 0
    assert np.array_equal(selected["frame"], np.arange(3))


def test_linkage_statistics_and_contingency_use_zero_as_nonpositive() -> None:
    rows = [
        {"id": 1, "g": 1.0, "b": 1.0, "gmatched": 1.0, "r_natural": 0.1, "r_tts": 0.2, "delta_c": 1.0, "delta_d": 0.2, "delta_cer": 0.1},
        {"id": 2, "g": 0.0, "b": 1.0, "gmatched": 1.0, "r_natural": 0.1, "r_tts": 0.2, "delta_c": 0.0, "delta_d": 0.2, "delta_cer": 0.1},
        {"id": 3, "g": -1.0, "b": 1.0, "gmatched": 1.0, "r_natural": 0.1, "r_tts": 0.2, "delta_c": 1.0, "delta_d": 0.2, "delta_cer": 0.1},
    ]
    result = metrics.linkage_statistics(rows, draws=100)
    table = result["contingency"]
    assert table["rows"]["G>0"]["deltaC>0"] == 1
    assert table["rows"]["G<=0"]["deltaC>0"] == 1
    assert table["rows"]["G<=0"]["deltaC<=0"] == 1
    assert table["both_positive_count"] == 1


def test_visual_and_association_gates_keep_smoke_non_scientific() -> None:
    summary = {"mean": 1.0, "ci95": [0.5, 1.5]}
    assert metrics.visual_status(engineering_status="PASS", calibration_status="CALIBRATED_ON_JOINT_COHORT", joint_count=3, g_summary=summary, b_summary=summary, matched_summary=summary) == "INSUFFICIENT_JOINT_PAIRS"
    assert metrics.association_status("PASS", 50, "CALIBRATED_ON_JOINT_COHORT", {"rho": 0.8, "p_two_sided": 0.001, "ci95": [0.3, 0.95]}) == "EXPLORATORY_POSITIVE_ASSOCIATION"
    assert metrics.sync_status({"status": "COMPLETE", "ci95": [0.1, 0.9]}) == "EXPLORATORY_SYNC_GAIN"
    assert metrics.sync_status({"status": "COMPLETE", "ci95": [-0.1, 0.9]}) == "NO_CLEAR_SYNC_GAIN"
    assert metrics.sync_status({"status": "COMPLETE", "ci95": [-0.9, -0.1]}) == "EXPLORATORY_REVERSE_SYNC_GAIN"
    assert metrics.sync_status({"status": "NOT_ESTIMABLE", "ci95": None}) == "NOT_ESTIMABLE"


def test_frozen_input_verification_detects_hash_change(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"before")
    manifest = {"records": [{"id": 1, "natural_video": str(source), "sha256": {"natural_video": runner.sha256_file(source)}}], "source_documents": {}}
    runner.verify_inputs(tmp_path, manifest)
    source.write_bytes(b"after")
    with pytest.raises(ValueError, match="frozen input changed"):
        runner.verify_inputs(tmp_path, manifest)
