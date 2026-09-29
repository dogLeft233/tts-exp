from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.fresh_source_inputs.protocol import (
    ProtocolError,
    delay_pcm,
    inspect_clip,
    select_cohort,
    source_group_from_sample,
    validate_visual_audit,
    write_json,
)


def test_video_only_zero_pts_is_valid_for_normalized_real_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "real.mkv"
    clip.write_bytes(b"video-only")
    monkeypatch.setattr(
        "scripts.experiments.fresh_source_inputs.protocol._ffprobe",
        lambda _path: {
            "format": {"duration": "5.6", "start_time": "0"},
            "streams": [{"codec_type": "video", "r_frame_rate": "25/1", "width": 512, "height": 512, "nb_frames": "140", "start_time": "0"}],
        },
    )
    monkeypatch.setattr("scripts.experiments.fresh_source_inputs.protocol._first_stream_pts", lambda _path, _selector: 0.0)
    result = inspect_clip(clip)
    assert result["has_audio"] is False
    assert result["pts_ok"] is True


def test_source_group_requires_original_id_shape() -> None:
    assert source_group_from_sample("lrs3_ABCDEFGHIJK_00001") == "ABCDEFGHIJK"
    assert source_group_from_sample("renamed-new-sample") is None
    assert source_group_from_sample("lrs3_ABCDEFGHIJK_00001_extra/evil") is None
    assert source_group_from_sample("lrs3_replacement_source_pool") is None


def test_visual_audit_requires_clip_binding_and_model(tmp_path: Path) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"frozen-model")
    with pytest.raises(ProtocolError):
        validate_visual_audit({"schema_version": 1, "method": "mediapipe_face_mesh_v1", "groups": {}})
    audit = validate_visual_audit({
        "schema_version": 1,
        "method": "mediapipe_face_mesh_v1",
        "model_path": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "thresholds": {"min_valid_fraction": 0.95, "min_eye_distance_px": 40, "min_frames": 140},
        "groups": {
            "ABCDEFGHIJK": {
                "status": "PASS",
                "clip_sha256": "b" * 64,
                "valid_fraction": 0.99,
                "eye_distance_px": 40.0,
                "frames_covered": 140,
                "first_frame_face": True,
                "mouth_visible": True,
                "face_box": [10, 10, 100, 100],
            }
        },
    })
    assert audit["ABCDEFGHIJK"]["audit_verified"] is True


def test_visual_fail_is_a_valid_non_admitting_record(tmp_path: Path) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"frozen-model")
    audit = validate_visual_audit({
        "schema_version": 1,
        "method": "mediapipe_face_mesh_v1",
        "model_path": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "thresholds": {"min_valid_fraction": 0.95, "min_eye_distance_px": 40, "min_frames": 140},
        "groups": {"ABCDEFGHIJK": {"status": "FAIL", "clip_sha256": "c" * 64, "valid_fraction": 0.2, "eye_distance_px": 12, "frames_covered": 28, "first_frame_face": False, "mouth_visible": False, "reason": "valid_fraction;first_frame_face"}},
    })
    assert audit["ABCDEFGHIJK"]["audit_verified"] is False


def test_strict_visual_array_rejects_resigned_numeric_tamper(tmp_path: Path) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"frozen-model")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip")
    clip_sha = hashlib.sha256(clip_path.read_bytes()).hexdigest()
    array_path = tmp_path / "landmarks.npz"
    points = np.zeros((140, 478, 2), dtype=np.float64)
    points[:, 33] = [0.25, 0.3]
    points[:, 263] = [0.75, 0.3]
    valid = np.ones((140,), dtype=bool)
    metadata = {"schema_version": 1, "source_group": "ABCDEFGHIJK", "clip_sha256": clip_sha, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "width": 100, "height": 100}
    np.savez_compressed(array_path, landmarks=points, valid=valid, timestamps_s=np.arange(140, dtype=np.float64) / 25.0, metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_))
    evidence = tmp_path / "preview.jpg"
    evidence.write_bytes(b"preview")
    write_json(tmp_path / "review.json", {"schema_version": 1, "clip_sha256": clip_sha, "status": "PASS", "reviewer": "agent", "mouth_visible": True, "evidence_path": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()})
    base = {
        "schema_version": 1, "method": "mediapipe_face_mesh_v1", "audit_mode": "unblock_v1",
        "model_path": str(model), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "thresholds": {"min_valid_fraction": 0.95, "min_eye_distance_px": 40, "min_frames": 140},
        "groups": {"ABCDEFGHIJK": {"status": "PASS", "clip_path": str(clip_path), "clip_sha256": clip_sha, "valid_fraction": 1.0, "eye_distance_px": 50.0, "frames_covered": 140, "first_frame_face": True, "mouth_visible": True, "face_box": [0, 0, 75, 30], "landmark_array_path": str(array_path), "landmark_array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(), "valid_array_sha256": hashlib.sha256(valid.tobytes()).hexdigest(), "input_review_path": str(tmp_path / "review.json"), "input_review_sha256": hashlib.sha256((tmp_path / "review.json").read_bytes()).hexdigest()}},
    }
    assert validate_visual_audit(base)["ABCDEFGHIJK"]["audit_verified"] is True
    tampered = json.loads(json.dumps(base))
    tampered["groups"]["ABCDEFGHIJK"]["eye_distance_px"] = 51.0
    with pytest.raises(ProtocolError, match="does not match raw array"):
        validate_visual_audit(tampered)


def test_select_cohort_blocks_without_fourteen_groups() -> None:
    rows = [{"source_group": f"g{i:02d}", "accepted_clip": {"clip_id": "1"}} for i in range(13)]
    cohort = select_cohort(rows)
    assert cohort["status"] == "BLOCKED_NEW_SOURCE"
    assert cohort["accepted_count"] == 13


def test_delay_pcm_keeps_exact_source_map() -> None:
    delayed, source_index = delay_pcm(np.arange(10, dtype=np.float32), samples=3)
    assert delayed.tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert source_index.tolist() == [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6]


def test_delay_pcm_zero_is_identity() -> None:
    delayed, source_index = delay_pcm(np.arange(4, dtype=np.float32), samples=0)
    assert delayed.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert source_index.tolist() == [0, 1, 2, 3]
