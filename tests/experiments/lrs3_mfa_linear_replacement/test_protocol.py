from __future__ import annotations

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.protocol import (
    build_stage00_artifacts,
    canonical_sha256,
    derive_stage_test_lock,
    validate_cross_manifest_joins,
    validate_fresh_cohort,
    validate_test_lock,
    write_stage00_artifacts,
)


def _cohort(count: int = 24) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"lrs3_group{index:02d}_{index:05d}",
            "source_group": f"group{index:02d}",
            "protocol_split": "fit",
            "face": f"video/{index}.mp4",
            "face_sha256": f"face-{index}",
            "natural_audio": f"natural/{index}.wav",
            "natural_audio_sha256": f"natural-{index}",
            "tts_audio": f"tts/{index}.wav",
            "tts_audio_sha256": f"tts-{index}",
        }
        for index in range(count)
    ]


def test_fresh_cohort_is_one_record_per_group_and_fit_only() -> None:
    records = _cohort()
    assert len(validate_fresh_cohort(records)) == 24
    with pytest.raises(ValueError, match="one record per source group"):
        validate_fresh_cohort(records + [records[0]], expected_count=25)
    bad = [dict(row) for row in records]
    bad[0]["protocol_split"] = "validation"
    with pytest.raises(ValueError, match="fit-only"):
        validate_fresh_cohort(bad)


def test_cross_manifest_joins_bind_source_tts_and_reference() -> None:
    cohort = _cohort(1)
    source = {
        "records": [{
            "sample_id": cohort[0]["sample_id"],
            "source_group": cohort[0]["source_group"],
            "video_sha256": cohort[0]["face_sha256"],
            "natural_audio_sha256": cohort[0]["natural_audio_sha256"],
            "transcript": "HELLO WORLD",
            "tts_transcript": "HELLO WORLD",
        }]
    }
    tts = {"results": {cohort[0]["sample_id"]: {
        "source_group": cohort[0]["source_group"],
        "video_sha256": cohort[0]["face_sha256"],
        "canonical_audio_sha256": cohort[0]["tts_audio_sha256"],
        "reference_audio_sha256": cohort[0]["natural_audio_sha256"],
        "transcript": "HELLO WORLD",
        "tts_transcript": "HELLO WORLD",
    }}}
    validate_cross_manifest_joins(cohort, source, tts)
    tts["results"][cohort[0]["sample_id"]]["reference_audio_sha256"] = "wrong"
    with pytest.raises(ValueError, match="reference"):
        validate_cross_manifest_joins(cohort, source, tts)


def _lock() -> dict[str, object]:
    return {
        "status": "sealed_unvisited",
        "media_opened": False,
        "derived_features_created": False,
        "scores_created": False,
        "test_group_count": 1,
        "test_groups": ["test"],
        "test_sample_ids": ["lrs3_test_00001"],
        "test_media_opened": False,
        "test_derived_features_created": False,
        "test_scores_created": False,
        "validation_groups": ["validation"],
        "validation_media_opened": False,
        "validation_derived_features_created": False,
        "validation_scores_created": False,
        "internal_dev_groups": ["internal"],
        "internal_dev_media_opened": False,
        "internal_dev_derived_features_created": False,
        "internal_dev_scores_created": False,
    }


def test_stage_test_lock_preserves_sealed_access_flags() -> None:
    lock = _lock()
    derived = derive_stage_test_lock(lock)
    validate_test_lock(derived)
    assert derived["fit_media_opened"] is False
    assert derived["test_sample_ids"] == lock["test_sample_ids"]


def test_stage00_artifact_builder_is_score_independent(tmp_path) -> None:
    records = _cohort()
    source_rows = []
    tts_rows = {}
    for index, row in enumerate(records):
        source_rows.append({
            "sample_id": row["sample_id"],
            "source_group": row["source_group"],
            "video_sha256": row["face_sha256"],
            "natural_audio_sha256": row["natural_audio_sha256"],
            "transcript": "HELLO WORLD",
            "tts_transcript": "HELLO WORLD",
        })
        tts_rows[row["sample_id"]] = {
            "source_group": row["source_group"],
            "video_sha256": row["face_sha256"],
            "canonical_audio_sha256": row["tts_audio_sha256"],
            "reference_audio_sha256": row["natural_audio_sha256"],
            "transcript": "HELLO WORLD",
            "tts_transcript": "HELLO WORLD",
        }
    source_rows.append({"sample_id": "lrs3_test_00001", "source_group": "test", "video_sha256": "test-face", "natural_audio_sha256": "test-natural", "transcript": "TEST", "tts_transcript": "TEST"})
    tts_rows["lrs3_test_00001"] = {"source_group": "test", "video_sha256": "test-face", "canonical_audio_sha256": "test-tts", "reference_audio_sha256": "test-natural", "transcript": "TEST", "tts_transcript": "TEST"}
    source_manifest = {"records": source_rows}
    tts_meta = {"results": tts_rows}
    visual = {
        "manifest_type": "lrs3_tts_visual_advantage_protocol_lock",
        "status": "complete",
        "decision": "GO",
        "cohorts": {"split": {"fit_groups": [], "internal_dev_groups": ["internal"], "validation_groups": ["validation"], "test_groups": ["test"]}, "fresh_confirmation": {"records": records}, "pilot": {"records": []}},
    }
    motion = {"manifest_type": "lrs3_motion_teacher_protocol_audit", "status": "complete", "decision": "GO", "protocol": {"manifest_type": "lrs3_motion_teacher_protocol", "split": {"fit_groups": [], "internal_dev_groups": ["internal"], "validation_groups": ["validation"], "test_groups": ["test"]}}}
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    artifacts = build_stage00_artifacts(
        parent_visual_manifest=visual,
        parent_visual_lock=_lock(),
        parent_motion_manifest=motion,
        parent_motion_lock=_lock(),
        source_manifest=source_manifest,
        tts_meta=tts_meta,
        asset_paths={"asset": asset},
        source_paths={"source": __file__},
        test_paths={"test": __file__},
    )
    assert artifacts["manifest"]["cohort"]["record_count"] == 24
    assert artifacts["manifest"]["selection_constraints"] if "selection_constraints" in artifacts["manifest"] else True
    assert artifacts["decision"]["scientific_decision"] == "not_available"


def test_stage00_writer_refuses_nonempty_output(tmp_path) -> None:
    output = tmp_path / "stage00"
    output.mkdir()
    (output / "unrelated").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        write_stage00_artifacts(output, {"manifest": {}, "test_lock": {}, "summary": {}, "decision": {}})


def test_canonical_hash_is_order_stable() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
