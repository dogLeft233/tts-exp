"""Frozen protocol constants for shared-adapter transfer evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..mfa_linear_real_video_sync.config import (
    P2_RECORDS,
    PrototypeConfig,
    Q,
    SAMPLE_RATE,
    SEGMENT_FRAMES,
    SEGMENT_SAMPLES,
    SYNCNET_MODEL_SHA256,
    TARGET_GAP_MIN,
    VSHIFT,
    VIDEO_FPS,
)

EXPERIMENT = "mfa-linear-sync-transfer-and-replacement-evaluation"
SCHEMA_VERSION = 1
SELECTION_SALT = "mfa-linear-sync-transfer-v1"
COHORT_SIZE = 8
MIN_SUCCESS_RECORDS = 6
GAIN_MIN = 3 * Q
TARGET_GAP = 2 * Q
MEL_TRUST_LIMIT = 0.10
RESIDUAL_BOUND = 0.05
SATURATION_FRACTION_MAX = 1e-4

P2_STAGE = "P2_SHARED_FOUR"
P2_STATUS = "FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED"
P2_STEPS = 100
P2_RECORD_ORDER = tuple(sample_id for sample_id, _ in P2_RECORDS)
P2_SOURCE_GROUPS = tuple(source_group for _, source_group in P2_RECORDS)

WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_CHECKPOINT_SHA256 = SYNCNET_MODEL_SHA256
ARMS = ("N", "B", "C")
EVAL_ARMS = ARMS
EXPECTED_MATRIX_CELLS = COHORT_SIZE * len(ARMS) * len(EVAL_ARMS)

BLOCKED_P2 = "BLOCKED_P2_NOT_PASSED"
BLOCKED_COHORT = "BLOCKED_COHORT_LOCK"
SELECTION_VIOLATION = "OUTCOME_SELECTION_VIOLATION"
NATURAL_LEAKAGE = "NATURAL_OR_SIDE_CHANNEL_LEAKAGE"
ADAPTER_MUTATION = "FROZEN_ADAPTER_MUTATION"
REAL_VIDEO_TRANSFER = "ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER_OBSERVED"
NO_REAL_VIDEO_TRANSFER = "NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER"
REAL_VIDEO_NOT_EVALUATED = "REAL_VIDEO_TRANSFER_NOT_EVALUATED"
REPLACEMENT_GATE_FAILED = "REPLACEMENT_NOT_RUN_REAL_VIDEO_GATE_FAILED"
RENDER_INCOMPLETE = "RENDER_MATRIX_INCOMPLETE"
REPLACEMENT_OBSERVED = "FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED"
NO_REPLACEMENT = "NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER"
REPLACEMENT_NOT_EVALUATED = "REPLACEMENT_NOT_EVALUATED"

@dataclass(frozen=True)
class TransferConfig:
    schema_version: int = SCHEMA_VERSION
    experiment: str = EXPERIMENT
    selection_salt: str = SELECTION_SALT
    cohort_size: int = COHORT_SIZE
    minimum_success_records: int = MIN_SUCCESS_RECORDS
    q: float = Q
    gain_min: float = GAIN_MIN
    target_gap: float = TARGET_GAP
    sample_rate: int = SAMPLE_RATE
    video_fps: int = VIDEO_FPS
    segment_frames: int = SEGMENT_FRAMES
    segment_samples: int = SEGMENT_SAMPLES
    vshift: int = VSHIFT
    p2_stage: str = P2_STAGE
    p2_status: str = P2_STATUS
    p2_steps: int = P2_STEPS
    p2_record_order: tuple[str, ...] = P2_RECORD_ORDER
    p2_source_groups: tuple[str, ...] = P2_SOURCE_GROUPS
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    syncnet_checkpoint_sha256: str = SYNCNET_CHECKPOINT_SHA256
    arms: tuple[str, ...] = ARMS
    evaluation_arms: tuple[str, ...] = EVAL_ARMS
    residual_bound: float = RESIDUAL_BOUND
    mel_trust_limit: float = MEL_TRUST_LIMIT
    saturation_fraction_max: float = SATURATION_FRACTION_MAX

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("p2_record_order", "p2_source_groups", "arms", "evaluation_arms"):
            payload[key] = list(payload[key])
        payload["expected_matrix_cells"] = EXPECTED_MATRIX_CELLS
        payload["predecessor_config"] = PrototypeConfig().to_dict()
        payload["forbidden_controls"] = [
            "extra_seed", "checkpoint_search", "threshold_tuning", "retry", "ensemble",
            "optimizer", "backward", "adapter_update", "record_substitution",
        ]
        return payload

    def validate(self) -> None:
        if self != TransferConfig():
            raise ValueError("scientific transfer configuration is frozen")


def validate_serialized_config(payload: Mapping[str, Any]) -> None:
    if dict(payload) != TransferConfig().to_dict():
        raise ValueError("serialized scientific configuration is frozen")


def default_paths(repo: str | Path) -> dict[str, Path]:
    root = Path(repo).resolve()
    return {
        "repo": root,
        "policy_records": root / ".claude/worktrees/lrs3-wavlm-resynthesis-50/tmp/lrs3_policy_a1_200_20260828/policy_cohort/records",
        "mfa_linear": root / "runs/lrs3_cem_fixed_video_20260820/12_policy_train_mfa_linear",
        "syncnet_model": root / "third_party/syncnet_python/data/syncnet_v2.model",
        "wav2lip_root": root / "third_party/Wav2Lip",
        "wav2lip_checkpoint": root / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth",
    }


def p2_parent_run_name() -> str:
    return "lrs3_mfa_linear_sync_transfer_20260903_p2"
