"""Protocol tests for the TTS-only fixed-coordinate experiment."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch

from scripts.experiments.mfa_linear_real_video_sync.config import CURVE_SIZE, SEGMENT_SAMPLES
from scripts.experiments.mfa_linear_real_video_sync.protocol import (
    ProtocolError,
    calibrate_target_offset,
    load_json_resume_cell,
    materialize_ffv1_once,
    model_inputs,
    read_fixed_video_frames,
    p1_metadata_template,
    validate_protocol_metadata,
    validate_target_offset_artifact,
    validate_training_record_fields,
    validate_waveform,
    write_json_once,
    sha256_file,
)


def valid_record() -> dict[str, object]:
    return {
        "mfa_linear_tts_waveform": torch.zeros(1, 1, SEGMENT_SAMPLES),
        "visual_embedding": torch.zeros(91, 1024),
        "target_offset": 0,
        "pristine_curve": torch.ones(CURVE_SIZE),
    }


def test_training_record_exposes_one_model_input() -> None:
    record = valid_record()
    validate_training_record_fields(record)
    assert set(model_inputs(record)) == {"mfa_linear_tts_waveform"}


@pytest.mark.parametrize(
    "field",
    [
        "natural_waveform",
        "natural_features",
        "video_conditioning",
        "identity_embedding",
        "speaker_id",
        "transcript_features",
        "second_audio",
        "cross_attention_key_value",
    ],
)
def test_training_record_rejects_leakage_fields(field: str) -> None:
    record = valid_record()
    record[field] = object()
    with pytest.raises(ProtocolError, match="isolation|forbidden"):
        validate_training_record_fields(record)


def test_training_record_rejects_missing_or_unknown_fields() -> None:
    record = valid_record()
    record.pop("target_offset")
    with pytest.raises(ProtocolError, match="missing"):
        validate_training_record_fields(record)
    record = valid_record()
    record["debug"] = None
    with pytest.raises(ProtocolError, match="unknown"):
        validate_training_record_fields(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", "other"),
        ("source_group", "other"),
        ("protocol_split", "test"),
        ("fps", 24),
        ("frame_start", 1),
        ("frame_end", 95),
        ("sample_start", 1),
        ("sample_end", SEGMENT_SAMPLES - 1),
        ("sample_rate", 48_000),
        ("channels", 2),
    ],
)
def test_protocol_metadata_rejects_each_mismatch(field: str, value: object) -> None:
    metadata = p1_metadata_template()
    metadata[field] = value
    with pytest.raises(ProtocolError, match="metadata mismatch"):
        validate_protocol_metadata(
            metadata,
            sample_id="lrs3_6ORDQFh0Byw_00004",
            source_group="6ORDQFh0Byw",
        )


def test_waveform_is_never_repaired() -> None:
    validate_waveform(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))
    for malformed in (
        np.zeros(SEGMENT_SAMPLES - 1, dtype=np.float32),
        np.zeros((2, SEGMENT_SAMPLES), dtype=np.float32),
        np.full(SEGMENT_SAMPLES, np.nan, dtype=np.float32),
    ):
        with pytest.raises(ProtocolError):
            validate_waveform(malformed)
    with pytest.raises(ProtocolError, match="sample rate"):
        validate_waveform(np.zeros(SEGMENT_SAMPLES, dtype=np.float32), sample_rate=8_000)


@pytest.mark.parametrize("offset", [-15, 0, 15])
def test_target_offset_sign_contract(offset: int) -> None:
    curve = np.full(CURVE_SIZE, 2.0, dtype=np.float64)
    curve[offset + 15] = 0.5
    artifact = calibrate_target_offset(curve)
    assert artifact["target_index"] == offset + 15
    assert artifact["target_offset"] == offset
    assert artifact["official_av_offset"] == -offset
    validate_target_offset_artifact(artifact)


def test_ffv1_round_trip_preserves_decoded_bgr_frames(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    rng = np.random.default_rng(9)
    frames = rng.integers(0, 256, (96, 16, 16, 3), dtype=np.uint8)
    path = tmp_path / "fixed.avi"
    lock = materialize_ffv1_once(path, frames)
    assert lock["codec"] == "ffv1"
    np.testing.assert_array_equal(read_fixed_video_frames(path), frames)
    with pytest.raises(FileExistsError, match="create-once"):
        materialize_ffv1_once(path, frames)


def test_create_once_and_hash_valid_resume_fail_closed(tmp_path) -> None:
    path = tmp_path / "cell.json"
    write_json_once(path, {"status": "complete", "value": 7})
    digest = sha256_file(path)
    assert load_json_resume_cell(path, expected_sha256=digest)["value"] == 7
    with pytest.raises(FileExistsError, match="create-once"):
        write_json_once(path, {"status": "complete", "value": 8})
    path.write_text('{"status":"complete","value":8}', encoding="utf-8")
    with pytest.raises(ProtocolError, match="hash mismatch"):
        load_json_resume_cell(path, expected_sha256=digest)


def test_partial_resume_cell_is_rejected(tmp_path) -> None:
    path = tmp_path / "partial.json"
    write_json_once(path, {"status": "in_progress"})
    with pytest.raises(ProtocolError, match="partial or stale"):
        load_json_resume_cell(path, expected_sha256=sha256_file(path))


def test_ambiguous_target_offset_is_rejected() -> None:
    curve = np.arange(CURVE_SIZE, dtype=np.float64) + 1.0
    curve[0] = 0.5
    curve[1] = 0.502
    with pytest.raises(ProtocolError, match="ambiguous"):
        calibrate_target_offset(curve)
