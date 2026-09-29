from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.experiments.lrs3_real_video_local_timing import config, validate
from scripts.experiments.lrs3_real_video_local_timing.common import (
    DiagnosticError,
    file_sha256,
    write_self_hashed_json,
)


def _fake_timeline(_path: Path) -> dict[str, object]:
    pts = [index * 512 for index in range(150)]
    return {
        "frame_count": 150,
        "frame_rate": 25.0,
        "width": 224,
        "height": 224,
        "start_time": 0.0,
        "duration": 6.0,
        "time_base": "1/12800",
        "codec_name": "fake",
        "first_pts": 0,
        "first_pts_time": 0.0,
        "pts": pts,
        "pts_time": [index / 25 for index in range(150)],
        "pts_strictly_increasing": True,
        "pts_max_deviation_ms": 0.0,
    }


def _fake_audio(path: Path) -> dict[str, object]:
    return {
        "sample_count": 150 * config.SAMPLES_PER_FRAME,
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.PCM_CHANNELS,
        "sample_width": config.PCM_SAMPLE_WIDTH,
        "container_sha256": file_sha256(path),
        "pcm_sha256": "pcm-sha256",
    }


def _fake_pairing(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {"start_difference_ms": 0.0}


@pytest.fixture
def audit_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[config.RunPaths, dict[str, object], dict[str, object]]:
    monkeypatch.setattr(validate, "video_timeline", _fake_timeline)
    monkeypatch.setattr(validate, "pcm_wav_metadata", _fake_audio)
    monkeypatch.setattr(validate, "_independent_pairing", _fake_pairing)

    sources: list[dict[str, object]] = []
    for index in range(config.EXPECTED_RECORD_COUNT):
        video = tmp_path / f"video-{index}.mp4"
        audio = tmp_path / f"audio-{index}.wav"
        video.write_bytes(f"video-{index}".encode())
        audio.write_bytes(f"audio-{index}".encode())
        sources.append(
            {
                "sample_id": f"sample-{index}",
                "source_group": f"group-{index}",
                "face_video": {"path": str(video), "sha256": file_sha256(video)},
                "natural_audio": {"path": str(audio), "sha256": file_sha256(audio)},
            }
        )
    cohort: dict[str, object] = {"records": sources}
    records = [validate._independent_audit_record(source, index) for index, source in enumerate(sources)]
    audit: dict[str, object] = {
        "schema_version": 2,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "cohort": {"sha256": config.HISTORY_COHORT_SHA256},
        "source_manifest": {
            "path": str(config.HISTORY_SOURCE_MANIFEST.resolve()),
            "sha256": config.HISTORY_SOURCE_MANIFEST_SHA256,
        },
        "record_count": config.EXPECTED_RECORD_COUNT,
        "passed_count": len(records),
        "blocked_count": 0,
        "status": "complete",
        "records": records,
    }
    paths = config.RunPaths(tmp_path / "run")
    write_self_hashed_json(paths.input_audit, audit)
    return paths, cohort, audit


def _assert_audit_rejected(paths: config.RunPaths, cohort: dict[str, object], audit: dict[str, object]) -> None:
    write_self_hashed_json(paths.input_audit, audit)
    with pytest.raises(DiagnosticError):
        validate.validate_input_audit(paths, cohort)


def test_validator_recomputes_a_complete_input_audit(audit_fixture) -> None:
    paths, cohort, _audit = audit_fixture
    audit, rows = validate.validate_input_audit(paths, cohort)
    assert audit["status"] == "complete"
    assert len(rows) == config.EXPECTED_RECORD_COUNT


def test_validator_rejects_tampered_sample_count(audit_fixture) -> None:
    paths, cohort, audit = audit_fixture
    tampered = deepcopy(audit)
    tampered["records"][0]["sample_count"] += 1
    _assert_audit_rejected(paths, cohort, tampered)


def test_validator_rejects_tampered_pts(audit_fixture) -> None:
    paths, cohort, audit = audit_fixture
    tampered = deepcopy(audit)
    tampered["records"][0]["source_video_timeline"]["pts"][1] += 1
    _assert_audit_rejected(paths, cohort, tampered)


def test_validator_rejects_tampered_source_hash(audit_fixture) -> None:
    paths, cohort, audit = audit_fixture
    tampered = deepcopy(audit)
    tampered["records"][0]["face_video"]["expected_sha256"] = "0" * 64
    _assert_audit_rejected(paths, cohort, tampered)


def test_validator_rejects_missing_audit_record(audit_fixture) -> None:
    paths, cohort, audit = audit_fixture
    tampered = deepcopy(audit)
    tampered["records"].pop()
    _assert_audit_rejected(paths, cohort, tampered)


def test_bounded_blocked_final_validates_against_the_complete_audit(audit_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, cohort, audit = audit_fixture
    result_text = "bounded-tail blocked\n"
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(result_text, encoding="utf-8")
    monkeypatch.setattr(validate, "load_history_cohort", lambda: cohort)
    write_self_hashed_json(
        paths.final,
        {
            "schema_version": 2,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": None,
            "eligibility": False,
            "result_sha256": file_sha256(paths.result),
            "input_audit": {
                "path": str(paths.input_audit.resolve()),
                "sha256": file_sha256(paths.input_audit),
            },
            "input_audit_sha256": file_sha256(paths.input_audit),
            "spec_bindings": {
                "parent_spec": {
                    "path": str(config.PARENT_SPEC.resolve()),
                    "sha256": config.PARENT_SPEC_SHA256,
                },
                "amendment_spec": {
                    "path": str(config.AMENDMENT_SPEC.resolve()),
                    "sha256": config.AMENDMENT_SPEC_SHA256,
                },
            },
            "parent_blocked_run": {
                "run_id": "lrs3_real_video_local_timing_20260905_v2",
                "path": str(config.PARENT_BLOCKED_FINAL.resolve()),
                "sha256": config.PARENT_BLOCKED_FINAL_SHA256,
                "result_sha256": config.PARENT_BLOCKED_RESULT_SHA256,
            },
        },
    )
    validated = validate.validate_blocked(paths)
    assert validated["status"] == "valid"
    assert validated["schema"] == "bounded_tail_v2"
    assert validated["new_contract"] == audit["status"]
    assert validated["audit_record_count"] == config.EXPECTED_RECORD_COUNT


def test_legacy_blocked_final_is_validated_only_as_legacy(tmp_path: Path) -> None:
    paths = config.RunPaths(tmp_path / "legacy")
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text("legacy blocked\n", encoding="utf-8")
    write_self_hashed_json(
        paths.final,
        {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": None,
            "eligibility": False,
            "result_sha256": file_sha256(paths.result),
        },
    )
    result = validate.validate_run(paths.root)
    assert result["stages"][0]["schema"] == "legacy"
    assert result["stages"][0]["new_contract"] == "not_checked"
