from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/experiments/lrs3_tts_visual_advantage/protocol.py"
SPEC = importlib.util.spec_from_file_location("lrs3_tts_visual_advantage_protocol", MODULE_PATH)
assert SPEC and SPEC.loader
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)

PARENT_DIR = REPO / "runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1"
TEACHER_MANIFEST = (
    REPO / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teacher_manifest.json"
)


def load_parents():
    return (
        protocol.load_json(PARENT_DIR / "manifest.json"),
        protocol.load_json(TEACHER_MANIFEST),
        protocol.load_json(PARENT_DIR / "test_lock.json"),
        protocol.load_json(REPO / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"),
        protocol.load_json(REPO / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"),
    )


def test_actual_cohorts_are_fit_only_disjoint_and_deterministic():
    parent, teacher, lock, source, tts = load_parents()
    first = protocol.build_cohorts(parent, teacher, lock, source, tts)
    second = protocol.build_cohorts(parent, teacher, lock, source, tts)

    assert first == second
    assert first["pilot"]["record_count"] == 48
    assert first["pilot"]["source_group_count"] == 24
    assert first["fresh_confirmation"]["record_count"] == 24
    assert first["fresh_confirmation"]["source_group_count"] == 24
    assert all(row["protocol_split"] == "fit" for row in first["pilot"]["records"])
    assert all(row["protocol_split"] == "fit" for row in first["fresh_confirmation"]["records"])

    pilot_ids = {row["sample_id"] for row in first["pilot"]["records"]}
    fresh_ids = {row["sample_id"] for row in first["fresh_confirmation"]["records"]}
    assert not pilot_ids & fresh_ids

    teacher_ids = {row["sample_id"] for row in teacher["records"]}
    working_by_group = {}
    for row in parent["protocol"]["working_records"]:
        if row["protocol_split"] == "fit" and row["sample_id"] not in teacher_ids:
            working_by_group.setdefault(row["source_group"], []).append(row["sample_id"])
    for row in first["fresh_confirmation"]["records"]:
        assert row["sample_id"] == min(working_by_group[row["source_group"]])


def test_test_lock_rejects_any_access_flag():
    _, _, lock, source, _ = load_parents()
    for key in ("media_opened", "derived_features_created", "scores_created"):
        changed = deepcopy(lock)
        changed[key] = True
        with pytest.raises(ValueError, match=key):
            protocol.validate_test_lock(changed)


def test_test_lock_rejects_membership_mismatch():
    _, _, lock, source, _ = load_parents()
    changed = deepcopy(lock)
    changed["test_sample_ids"][0] = "lrs3_6XNrzmh0EVs_99999"
    with pytest.raises(ValueError, match="exact canonical"):
        protocol.validate_test_lock(changed, source["records"])


def test_working_records_must_not_intersect_test_ids():
    parent, teacher, lock, source, tts = load_parents()
    changed = deepcopy(parent)
    changed["protocol"]["working_records"][0]["sample_id"] = lock["test_sample_ids"][0]
    with pytest.raises(ValueError, match="sealed test"):
        protocol.build_cohorts(changed, teacher, lock, source, tts)


def test_fresh_record_is_required_for_every_fit_group():
    parent, teacher, lock, source, tts = load_parents()
    changed = deepcopy(parent)
    first_group = changed["protocol"]["split"]["fit_groups"][0]
    teacher_ids = {row["sample_id"] for row in teacher["records"]}
    changed["protocol"]["working_records"] = [
        row
        for row in changed["protocol"]["working_records"]
        if row["source_group"] != first_group or row["sample_id"] in teacher_ids
    ]
    while len(changed["protocol"]["working_records"]) < protocol.EXPECTED_COUNTS["working_records"]:
        clone = deepcopy(changed["protocol"]["working_records"][-1])
        clone["sample_id"] += "_clone"
        changed["protocol"]["working_records"].append(clone)
    with pytest.raises(ValueError, match="absent from canonical"):
        protocol.build_cohorts(changed, teacher, lock, source, tts)


def test_teacher_selection_must_not_use_syncnet():
    parent, teacher, lock, source, tts = load_parents()
    changed = deepcopy(teacher)
    changed["syncnet_used_for_selection"] = True
    with pytest.raises(ValueError, match="score-independent"):
        protocol.build_cohorts(parent, changed, lock, source, tts)


def test_stage_test_lock_scopes_all_non_fit_splits_as_unopened():
    parent, teacher, lock, source, tts = load_parents()
    cohorts = protocol.build_cohorts(parent, teacher, lock, source, tts)
    result = protocol.stage_test_lock(lock, cohorts)

    assert result["status"] == "sealed_unvisited"
    assert result["test_group_count"] == 7
    assert len(result["test_sample_ids"]) == 84
    for split in ("internal_dev", "validation", "test"):
        assert result[f"{split}_media_opened"] is False
        assert result[f"{split}_derived_features_created"] is False
        assert result[f"{split}_scores_created"] is False


def test_generated_lock_round_trips_and_rejects_split_access_flags():
    parent, teacher, lock, source, tts = load_parents()
    cohorts = protocol.build_cohorts(parent, teacher, lock, source, tts)
    generated = protocol.stage_test_lock(lock, cohorts)
    assert protocol.validate_test_lock(generated, source["records"])["sample_ids"] == generated["test_sample_ids"]

    changed = deepcopy(generated)
    changed["validation_media_opened"] = True
    with pytest.raises(ValueError, match="validation_media_opened"):
        protocol.validate_test_lock(changed, source["records"])


def test_missing_access_flags_are_rejected():
    _, _, lock, source, _ = load_parents()
    for key in ("media_opened", "derived_features_created", "scores_created"):
        changed = deepcopy(lock)
        del changed[key]
        with pytest.raises(ValueError, match="missing required access flag"):
            protocol.validate_test_lock(changed, source["records"])


def test_missing_split_access_flag_is_rejected_for_generated_schema():
    parent, teacher, lock, source, tts = load_parents()
    cohorts = protocol.build_cohorts(parent, teacher, lock, source, tts)
    generated = protocol.stage_test_lock(lock, cohorts)
    del generated["validation_scores_created"]
    with pytest.raises(ValueError, match="missing split access flag"):
        protocol.validate_test_lock(generated, source["records"])

    parent, teacher, lock, source, tts = load_parents()
    changed = deepcopy(parent)
    changed["protocol"]["working_records"][0]["natural_audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="natural audio hash mismatch"):
        protocol.build_cohorts(changed, teacher, lock, source, tts)


def test_teacher_must_be_subset_of_parent_working_records():
    parent, teacher, lock, source, tts = load_parents()
    changed = deepcopy(teacher)
    changed["records"][0]["sample_id"] = "lrs3_6ORDQFh0Byw_00099"
    with pytest.raises(ValueError, match="absent from parent working"):
        protocol.build_cohorts(parent, changed, lock, source, tts)


def test_code_review_is_bound_to_exact_sources(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    rows = [{"path": "source.py", "sha256": protocol.file_sha256(source)}]
    review_path = tmp_path / "review.json"
    review = {
        "stage_id": protocol.STAGE_ID,
        "status": "PASS",
        "test_status": "PASS",
        "source_manifest_sha256": protocol.canonical_sha256(rows),
    }
    review_path.write_text(json.dumps(review), encoding="utf-8")
    assert protocol.validate_code_review(review_path, rows)["status"] == "PASS"

    source.write_text("x = 2\n", encoding="utf-8")
    changed = [{"path": "source.py", "sha256": protocol.file_sha256(source)}]
    with pytest.raises(ValueError, match="source hash mismatch"):
        protocol.validate_code_review(review_path, changed)


def test_non_finite_json_is_rejected(tmp_path: Path):
    target = tmp_path / "bad.json"
    target.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        protocol.load_json(target)
