from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/experiments/lrs3_tts_visual_advantage/run_metric_parity.py"
SPEC = importlib.util.spec_from_file_location("lrs3_tts_visual_advantage_metric_parity", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _metrics(natural_distance: float = 0.2, tts_distance: float = 0.1) -> dict:
    return {
        "exact_time": {
            "natural": {"distance": natural_distance, "coverage": 1.0},
            "tts": {"distance": tts_distance, "coverage": 1.0},
            "benefit": natural_distance - tts_distance,
        },
        "visual_dtw": {
            "natural": {"distance": natural_distance, "coverage": 1.0, "path_length": 10, "warp_burden": 0.0},
            "tts": {"distance": tts_distance, "coverage": 1.0, "path_length": 10, "warp_burden": 0.0},
            "benefit": natural_distance - tts_distance,
        },
    }


def test_json_number_replaces_nonfinite_values():
    result = module._json_number({"nan": math.nan, "nested": [math.inf, 2.0]})
    assert result == {"nan": None, "nested": [None, 2.0]}


def test_sequence_gate_requires_valid_coverage_and_finite_metrics():
    sequence = SimpleNamespace(valid=np.ones(10, dtype=bool))
    gate = module._sequence_gate(sequence, sequence, sequence, _metrics())
    assert gate["eligible"]
    blocked = module._sequence_gate(sequence, sequence, sequence, _metrics(float("nan"), 0.1))
    assert not blocked["eligible"]
    assert not blocked["checks"]["finite_primary_metrics"]
    incomplete_dtw = _metrics()
    incomplete_dtw["visual_dtw"]["tts"]["distance"] = None
    blocked_dtw = module._sequence_gate(sequence, sequence, sequence, incomplete_dtw)
    assert not blocked_dtw["eligible"]
    assert not blocked_dtw["checks"]["finite_secondary_metrics"]


def test_sequence_gate_requires_complete_finite_dtw_result():
    sequence = SimpleNamespace(valid=np.ones(10, dtype=bool))
    metrics = _metrics()
    metrics["visual_dtw"]["tts"]["warp_burden"] = None
    blocked = module._sequence_gate(sequence, sequence, sequence, metrics)
    assert not blocked["eligible"]
    assert not blocked["checks"]["finite_secondary_metrics"]


def test_compare_old_direction_reports_paired_sign_agreement():
    rows = [
        {"sample_id": "a", "exact_time_benefit": -0.2},
        {"sample_id": "b", "exact_time_benefit": 0.1},
        {"sample_id": "c", "exact_time_benefit": None},
    ]
    result = module._compare_old_direction(rows, {"a": -0.3, "b": -0.1, "c": 0.2})
    assert result["overlap_count"] == 2
    assert result["sign_agreement_fraction"] == pytest.approx(0.5)


def test_pilot_cohort_requires_two_unique_fit_records_per_group():
    groups = [f"group{index}" for index in range(24)]
    records = [
        {
            "sample_id": f"lrs3_{group}_{offset:05d}",
            "source_group": group,
            "protocol_split": "fit",
        }
        for group in groups
        for offset in (2, 3)
    ]
    manifest = {
        "cohorts": {
            "split": {"fit_groups": groups},
            "pilot": {"record_count": 48, "source_group_count": 24, "records": records},
        }
    }
    assert len(module._validate_pilot_cohort(manifest)) == 48
    duplicate = {**manifest, "cohorts": {**manifest["cohorts"], "pilot": {**manifest["cohorts"]["pilot"], "records": records[:-1] + [records[0]]}}}
    with pytest.raises(ValueError, match="sample ids must be unique"):
        module._validate_pilot_cohort(duplicate)


def test_teacher_record_requires_distinct_semantic_arms():
    protocol = module.load_json(module.FROZEN_PROTOCOL_MANIFEST_PATH)
    teacher_manifest = module.load_json(
        module.REPO
        / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teacher_manifest.json"
    )
    record = protocol["cohorts"]["pilot"]["records"][0]
    teacher = next(row for row in teacher_manifest["records"] if row["sample_id"] == record["sample_id"])
    bad_teacher = {
        **teacher,
        "tts_teacher": {
            **teacher["tts_teacher"],
            "video": teacher["natural_teacher"]["video"],
            "video_sha256": teacher["natural_teacher"]["video_sha256"],
        },
    }
    with pytest.raises(ValueError, match="teacher video provenance mismatch"):
        module._validate_teacher_record(
            record,
            bad_teacher,
            module.REPO / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teachers",
        )


def test_frozen_protocol_lock_binds_canonical_manifest_and_test_lock():
    manifest = module.load_json(module.FROZEN_PROTOCOL_MANIFEST_PATH)
    hashes = module._assert_frozen_protocol_lock(
        module.FROZEN_PROTOCOL_MANIFEST_PATH,
        manifest,
        module.FROZEN_PROTOCOL_MANIFEST_SHA256,
    )
    assert hashes == {
        "protocol_manifest_sha256": module.FROZEN_PROTOCOL_MANIFEST_SHA256,
        "test_lock_sha256": module.FROZEN_TEST_LOCK_SHA256,
    }


def test_snapshot_file_verifies_copied_bytes(tmp_path):
    source = tmp_path / "source.bin"
    target = tmp_path / "snapshots" / "copy.bin"
    source.write_bytes(b"stable bytes")
    expected = module.file_sha256(source)
    assert module._snapshot_file(source, expected, target, "fixture") == target
    assert target.read_bytes() == b"stable bytes"


def test_run_success_path_enforces_all_boundaries_without_media(monkeypatch, tmp_path):
    protocol_path = module.FROZEN_PROTOCOL_MANIFEST_PATH
    teacher_path = module.REPO / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teacher_manifest.json"
    audit_path = module.REPO / "runs/lrs3_motion_teacher_20260822/02_landmark_teacher_audit_retry2/landmark_audit.json"
    asset_path = module.REPO / "checkpoints/mediapipe/face_landmarker.task"
    protocol = module.load_json(protocol_path)
    teacher_manifest = module.load_json(teacher_path)
    teacher_index = {row["sample_id"]: row for row in teacher_manifest["records"]}
    video_hashes = {asset_path.resolve(): module._assert_bound_asset(protocol, "mediapipe_landmarker", asset_path)}
    for record in protocol["cohorts"]["pilot"]["records"]:
        teacher = teacher_index[record["sample_id"]]
        video_hashes[Path(record["face"]).resolve()] = record["face_sha256"]
        video_hashes[Path(teacher["natural_teacher"]["video"]).resolve()] = teacher["natural_teacher"]["video_sha256"]
        video_hashes[Path(teacher["tts_teacher"]["video"]).resolve()] = teacher["tts_teacher"]["video_sha256"]

    def fake_file_sha256(path):
        resolved = Path(path).resolve()
        if resolved not in video_hashes:
            raise AssertionError(f"unexpected hash request: {resolved}")
        return video_hashes[resolved]

    def fake_snapshot(source, expected, target, _label):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic snapshot")
        video_hashes[target.resolve()] = expected
        return target

    def fake_extract(video_path, asset_path):
        assert str(video_path).startswith(str(tmp_path / "run" / "snapshots"))
        assert str(asset_path).startswith(str(tmp_path / "run" / "snapshots"))
        return SimpleNamespace(valid=np.ones(10, dtype=bool))

    def fake_save(path, _sequence):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic feature")

    def fake_pair_metrics(*_args, **_kwargs):
        arm = lambda distance: {"distance": distance, "coverage": 1.0, "path_length": 10, "warp_burden": 0.0}
        return {
            "exact_time": {"natural": arm(0.2), "tts": arm(0.1), "benefit": 0.1},
            "visual_dtw": {"natural": arm(0.2), "tts": arm(0.1), "benefit": 0.1, "band_fraction": 0.2, "max_run": 2},
        }

    monkeypatch.setattr(module, "file_sha256", fake_file_sha256)
    monkeypatch.setattr(module, "_snapshot_file", fake_snapshot)
    monkeypatch.setattr(module, "_verify_video", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "extract_video_features", fake_extract)
    monkeypatch.setattr(module, "save_visual_sequence", fake_save)
    monkeypatch.setattr(module, "pair_metrics", fake_pair_metrics)
    args = SimpleNamespace(
        protocol_manifest=protocol_path,
        teacher_manifest=teacher_path,
        old_landmark_audit=audit_path,
        landmarker_asset=asset_path,
        outdir=tmp_path / "run",
    )
    decision = module.run(args)
    assert decision["decision"] == "GO"
    summary = module.load_json(args.outdir / "summary.json")
    assert summary["coverage_gate"]["eligible_record_count"] == 48
    assert summary["media_access"]["internal_dev_media_opened"] is False
    assert summary["media_access"]["validation_media_opened"] is False
    assert summary["media_access"]["test_media_opened"] is False


def test_frozen_protocol_lock_rejects_noncanonical_manifest(tmp_path):
    replacement = tmp_path / "manifest.json"
    replacement.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the frozen Stage 00 manifest"):
        module._assert_frozen_protocol_lock(
            replacement,
            {},
            "0" * 64,
        )


def test_frozen_protocol_lock_rejects_overlapping_sealed_splits():
    manifest = module.load_json(module.FROZEN_PROTOCOL_MANIFEST_PATH)
    split = manifest["cohorts"]["split"]
    split["fit_groups"][0] = split["internal_dev_groups"][0]
    with pytest.raises(ValueError, match="groups overlap"):
        module._assert_frozen_protocol_lock(
            module.FROZEN_PROTOCOL_MANIFEST_PATH,
            manifest,
            module.FROZEN_PROTOCOL_MANIFEST_SHA256,
        )
