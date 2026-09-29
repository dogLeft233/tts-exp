"""Frozen configuration for the post-hoc audio-tolerance diagnostic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..mfa_linear_real_video_sync.config import Q, SAMPLE_RATE, SEGMENT_FRAMES, SEGMENT_SAMPLES, SYNCNET_MODEL_SHA256, VSHIFT, VIDEO_FPS

EXPERIMENT = "mfa-linear-existing-data-audio-tolerance"
SCHEMA_VERSION = 1
ORIGINAL_MEL_LIMIT = 0.10
DIAGNOSTIC_MEL_LIMIT = 0.11
TRAIN_RECORD_COUNT = 94
EVAL_RECORD_COUNT = 8
MIN_SUCCESS_RECORDS = 6
GAIN_MIN = 3 * Q
TARGET_GAP = 2 * Q
RESIDUAL_BOUND = 0.05
SATURATION_FRACTION_MAX = 1e-4
EXPECTED_MATRIX_CELLS = EVAL_RECORD_COUNT * 3 * 3
PARENT_RUN_NAME = "lrs3_mfa_linear_sync_existing_data_20260903_cublas_fix"
PARENT_MANIFEST_RELATIVE = "01_data_lock/manifest.json"
PARENT_CHECKPOINT_RELATIVE = "02_training/step100/model.pt"
PARENT_HISTORY_RELATIVE = "02_training/history.json"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_CHECKPOINT_SHA256 = SYNCNET_MODEL_SHA256

DIAGNOSTIC_PARENT_INVALID = "DIAGNOSTIC_PARENT_INVALID"
DIAGNOSTIC_NOT_EVALUATED = "DIAGNOSTIC_NOT_EVALUATED"
DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_REAL_VIDEO_COMPLETE"
REPLACEMENT_NOT_RUN = "DIAGNOSTIC_REPLACEMENT_NOT_RUN"
REPLACEMENT_OBSERVED = "DIAGNOSTIC_REPLACEMENT_OBSERVED"
REPLACEMENT_NOT_EVALUATED = "DIAGNOSTIC_REPLACEMENT_NOT_EVALUATED"


@dataclass(frozen=True)
class DiagnosticConfig:
    schema_version: int = SCHEMA_VERSION
    experiment: str = EXPERIMENT
    original_mel_limit: float = ORIGINAL_MEL_LIMIT
    diagnostic_mel_limit: float = DIAGNOSTIC_MEL_LIMIT
    train_record_count: int = TRAIN_RECORD_COUNT
    evaluation_record_count: int = EVAL_RECORD_COUNT
    minimum_success_records: int = MIN_SUCCESS_RECORDS
    q: float = Q
    gain_min: float = GAIN_MIN
    target_gap: float = TARGET_GAP
    sample_rate: int = SAMPLE_RATE
    video_fps: int = VIDEO_FPS
    segment_frames: int = SEGMENT_FRAMES
    segment_samples: int = SEGMENT_SAMPLES
    vshift: int = VSHIFT
    residual_bound: float = RESIDUAL_BOUND
    saturation_fraction_max: float = SATURATION_FRACTION_MAX
    expected_matrix_cells: int = EXPECTED_MATRIX_CELLS
    parent_run_name: str = PARENT_RUN_NAME
    syncnet_checkpoint_sha256: str = SYNCNET_CHECKPOINT_SHA256
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    claim_scope: str = "post-hoc exploratory diagnostic; record-heldout IDs with source-group overlap"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["forbidden_controls"] = [
            "retrain", "warm_start", "extra_seed", "checkpoint_search", "retry",
            "record_substitution", "threshold_tuning", "natural_audio_input",
            "scientific_promotion",
        ]
        return payload

    def validate(self) -> None:
        if self != DiagnosticConfig():
            raise ValueError("diagnostic configuration is frozen")


def validate_serialized_config(payload: Mapping[str, Any]) -> None:
    if dict(payload) != DiagnosticConfig().to_dict():
        raise ValueError("serialized diagnostic configuration is frozen")


def default_parent(repo: str | Path) -> Path:
    return Path(repo).resolve() / "runs" / PARENT_RUN_NAME


def default_output(repo: str | Path) -> Path:
    return Path(repo).resolve() / "runs" / "lrs3_mfa_linear_existing_data_audio_tolerance_20260903"
