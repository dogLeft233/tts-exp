from __future__ import annotations

from pathlib import Path

import importlib.util

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/experiments/lrs3_mfa_linear_replacement_mfa3_exploratory/run_visual_teacher_audit_face_ready.py"
SPEC = importlib.util.spec_from_file_location("lrs3_visual_teacher_face_ready", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_frozen_face_ready_join_preserves_order_and_group_coverage():
    stage00 = runner._validate_stage00(runner.STAGE00_PATH)
    alignment, alignment_records = runner._validate_alignment(runner.ALIGNMENT_PATH, runner.STAGE00_PATH, stage00)
    candidate, candidate_results = runner._validate_candidate(runner.CANDIDATE_PATH, runner.STAGE00_PATH, runner.ALIGNMENT_PATH, alignment)
    preflight, ready_ids = runner._validate_preflight(
        runner.FACE_PREFLIGHT_PATH,
        runner.STAGE00_PATH,
        runner.ALIGNMENT_PATH,
        runner.CANDIDATE_PATH,
    )
    render_protocol, render = runner._validate_render(
        runner.RENDER_PATH,
        runner.RENDER_PROTOCOL_PATH,
        runner.STAGE00_PATH,
        runner.ALIGNMENT_PATH,
        runner.CANDIDATE_PATH,
        runner.FACE_PREFLIGHT_PATH,
        ready_ids,
    )
    records = runner._build_records(
        stage00,
        alignment_records,
        candidate_results,
        render_protocol,
        render,
        ready_ids,
    )

    assert candidate["record_count"] == 62
    assert preflight["face_ready_record_count"] == 59
    assert [row["sample_id"] for row in records] == ready_ids
    assert len(records) == 59
    assert len({row["source_group"] for row in records}) == 23
    assert all(row["protocol_split"] == "fit" for row in records)
    assert all(Path(row["real_video"]).is_file() for row in records)
    assert all(Path(row["natural_video"]).is_file() for row in records)
    assert all(Path(row["candidate_video"]).is_file() for row in records)


def test_face_ready_denominator_is_fixed_before_visual_metrics():
    assert runner.INPUT_CLEAN_RECORD_COUNT == 62
    assert runner.INPUT_FACE_READY_RECORD_COUNT == 59
    assert runner.MINIMUM_ELIGIBLE_RECORD_COUNT == 57
    assert runner.EXPECTED_GROUP_COUNT == 23


def test_render_retry2_is_score_free_and_complete():
    render = runner.load_json(runner.RENDER_PATH)
    assert render["render_count"] == 118
    assert render["record_count"] == 59
    assert render["face_ready_sample_ids"] == runner.load_json(runner.FACE_PREFLIGHT_PATH)["face_ready_sample_ids"]
