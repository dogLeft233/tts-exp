"""Frozen configuration for the existing-data record-heldout run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..mfa_linear_real_video_sync.config import (
    ADAM_BETAS,
    ADAM_EPS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MEL_TRUST_LIMIT,
    MODEL_CHANNELS,
    MODEL_DILATIONS,
    MODEL_RESIDUAL_SCALE,
    P2_RECORDS,
    Q,
    SAMPLE_RATE,
    SATURATION_FRACTION_MAX,
    SEED,
    SEGMENT_FRAMES,
    SEGMENT_SAMPLES,
    VSHIFT,
    VIDEO_FPS,
    WEIGHT_DECAY,
    SYNCNET_MODEL_SHA256,
)

EXPERIMENT = "mfa-linear-sync-existing-data"
SCHEMA_VERSION = 1
SELECTION_SALT = "mfa-linear-sync-existing-data-v1"
TRAIN_RECORD_COUNT = 94
EVAL_RECORD_COUNT = 8
MIN_SUCCESS_RECORDS = 6
TRAIN_STEPS = 100
GAIN_MIN = 3 * Q
TARGET_GAP = 2 * Q
RESIDUAL_BOUND = MODEL_RESIDUAL_SCALE
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
ARMS = ("N", "B", "C")
EVAL_ARMS = ARMS
EXPECTED_MATRIX_CELLS = EVAL_RECORD_COUNT * len(ARMS) * len(EVAL_ARMS)
P2_SOURCE_GROUPS = tuple(source_group for _, source_group in P2_RECORDS)

BLOCKED_EXISTING_DATA = "BLOCKED_EXISTING_DATA_INVENTORY"
NATURAL_LEAKAGE = "NATURAL_OR_SIDE_CHANNEL_LEAKAGE"
ADAPTER_MUTATION = "FROZEN_ADAPTER_MUTATION"
OUTCOME_SELECTION_VIOLATION = "OUTCOME_SELECTION_VIOLATION"
REAL_VIDEO_TRANSFER = "RECORD_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED"
NO_REAL_VIDEO_TRANSFER = "NO_RECORD_HELDOUT_REAL_VIDEO_TRANSFER"
REAL_VIDEO_NOT_EVALUATED = "RECORD_HELDOUT_REAL_VIDEO_NOT_EVALUATED"
REPLACEMENT_GATE_FAILED = "REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED"
REPLACEMENT_OBSERVED = "FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED"
NO_REPLACEMENT = "NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER"
REPLACEMENT_NOT_EVALUATED = "REPLACEMENT_NOT_EVALUATED"
RENDER_INCOMPLETE = "RENDER_MATRIX_INCOMPLETE"


@dataclass(frozen=True)
class ExistingDataConfig:
    schema_version: int = SCHEMA_VERSION
    experiment: str = EXPERIMENT
    selection_salt: str = SELECTION_SALT
    train_record_count: int = TRAIN_RECORD_COUNT
    evaluation_record_count: int = EVAL_RECORD_COUNT
    minimum_success_records: int = MIN_SUCCESS_RECORDS
    train_steps: int = TRAIN_STEPS
    seed: int = SEED
    q: float = Q
    gain_min: float = GAIN_MIN
    target_gap: float = TARGET_GAP
    sample_rate: int = SAMPLE_RATE
    video_fps: int = VIDEO_FPS
    segment_frames: int = SEGMENT_FRAMES
    segment_samples: int = SEGMENT_SAMPLES
    vshift: int = VSHIFT
    model_channels: int = MODEL_CHANNELS
    model_dilations: tuple[int, ...] = MODEL_DILATIONS
    residual_scale: float = RESIDUAL_BOUND
    learning_rate: float = LEARNING_RATE
    betas: tuple[float, float] = ADAM_BETAS
    epsilon: float = ADAM_EPS
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRAD_CLIP_NORM
    mel_trust_limit: float = MEL_TRUST_LIMIT
    saturation_fraction_max: float = SATURATION_FRACTION_MAX
    p2_source_groups: tuple[str, ...] = P2_SOURCE_GROUPS
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    syncnet_checkpoint_sha256: str = SYNCNET_MODEL_SHA256
    arms: tuple[str, ...] = ARMS
    evaluation_arms: tuple[str, ...] = EVAL_ARMS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("model_dilations", "betas", "p2_source_groups", "arms", "evaluation_arms"):
            payload[key] = list(payload[key])
        payload["expected_matrix_cells"] = EXPECTED_MATRIX_CELLS
        payload["claim_scope"] = "record-heldout within fit-only LRS3; source-group overlap allowed"
        payload["forbidden_controls"] = [
            "extra_seed", "checkpoint_search", "threshold_tuning", "retry", "ensemble",
            "warm_start", "early_stopping", "record_substitution", "sealed_split_access",
            "natural_audio_input", "natural_audio_loss", "side_channel_input",
            "source_group_generalization_claim",
        ]
        return payload

    def validate(self) -> None:
        if self != ExistingDataConfig():
            raise ValueError("scientific existing-data configuration is frozen")


def validate_serialized_config(payload: Mapping[str, Any]) -> None:
    if dict(payload) != ExistingDataConfig().to_dict():
        raise ValueError("serialized scientific existing-data configuration is frozen")


def default_paths(repo: str | Path) -> dict[str, Path]:
    root = Path(repo).resolve()
    return {
        "repo": root,
        "policy_records": root / ".claude/worktrees/lrs3-wavlm-resynthesis-50/tmp/lrs3_policy_a1_200_20260828/policy_cohort/records",
        "mfa_linear": root / "runs/lrs3_cem_fixed_video_20260820/12_policy_train_mfa_linear",
        "syncnet_model": root / "third_party/syncnet_python/data/syncnet_v2.model",
        "wav2lip_checkpoint": root / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth",
    }


def run_name() -> str:
    return "lrs3_mfa_linear_sync_existing_data_20260903"
