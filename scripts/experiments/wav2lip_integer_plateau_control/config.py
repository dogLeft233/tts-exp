from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/test-wav2lip-integer-plateau-control"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-integer-plateau-control/spec.md"

ORACLE_ROOT = REPO / "runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4"
ORACLE_FINAL = ORACLE_ROOT / "final.json"
ORACLE_VALIDATION = ORACLE_ROOT / "validation.json"
ORACLE_PROTOCOL = ORACLE_ROOT / "protocol.json"
ORACLE_INPUT_AUDIT = ORACLE_ROOT / "input_audit.json"
ORACLE_FRAMES = ORACLE_ROOT / "frames/manifest.json"
ORACLE_MEDIA = ORACLE_ROOT / "media/manifest.json"
ORACLE_SCORES = ORACLE_ROOT / "scores/manifest.json"

ROI_FINAL = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix5/final.json"
ROI_VALIDATION = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix5/validation.json"
# host_fix5 is the frozen final/control result; its protocol assets are the
# immutable host_fix1 inputs referenced by that final artifact.
ROI_PROTOCOL = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix1/protocol.json"
ROI_ROI_MANIFEST = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix1/roi/manifest.json"

LINEAR_ROOT = REPO / "runs/wav2lip_oracle_frame_interpolation_20260907_linear_v1"
LINEAR_FINAL = LINEAR_ROOT / "final.json"
LINEAR_VALIDATION = LINEAR_ROOT / "validation.json"
LINEAR_PROTOCOL = LINEAR_ROOT / "protocol.json"

SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
GENERATION_WORKER = REPO / "scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

ORACLE_HASHES = {
    "final": "919a5af95e361cbe218fc2d88003aa74d2e061d68d3543fe8fcbc7466cc0cff1",
    "validation": "a8e4719af2aa1e15477871caadea16f2c7b6fc69b4404c71f19bca2d33546724",
    "protocol": "6bec09cb8abdc7d8de9b50dfb388f1395924b24c9b130b808a9c5387378fa079",
}
ROI_HASHES = {
    "final": "76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563",
    "validation": "328e562cc8dacfb960fb2319fb15f1cfa90eecf593f04434382c9a5e13bd2577",
    "protocol": "835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4",
}
LINEAR_HASHES = {
    "final": "3c34ed447326eb164cc5549cd902fc55f64f4d4b5ce8c3e8b5ce09489bd3a542",
    "validation": "e5f82d26b3c7bd7f45bfeca9c0fe6589a5983ba6147348fc04e11aac46ea5820",
    "protocol": "7666cab5d483383155284a43f43e48b9068d0312c24d1ecfc3b2e00881b56473",
}

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
VSHIFT = 15
WINDOW_FRAMES = 5
MFCC_STRIDE = 4
MFCC_AUDIO_FRAMES = 20
MFCC_FRAME_LENGTH = 400
MFCC_FRAME_STEP = 160
MFCC_PREEMPHASIS_PREDECESSOR = 1
EMBEDDING_DIM = 1024
EPSILON = 1e-6
BATCH_SIZE = 20
TORCH_THREADS = 4
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
MATRIX_DIFF_TOLERANCE = 0.001
VALIDATOR_MATRIX_TOLERANCE = 0.0001
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905
FRAME_SHIFT = 5
AUDIO_SHIFT = FRAME_SHIFT * SAMPLES_PER_FRAME

VIDEO_ID = "V_ID"
VIDEO_P = "V_P"
AUDIO_N = "N"
AUDIO_P = "P"
FRESH_CELL_SPECS = ((VIDEO_ID, AUDIO_P), (VIDEO_P, AUDIO_N), (VIDEO_P, AUDIO_P))
CACHED_CELL_SPEC = (VIDEO_ID, AUDIO_N)

EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
ORDERED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_integer_plateau_control_{validate_run_id(run_id)}"


def cell_key(sample_id: str, video_arm: str, audio_arm: str) -> str:
    return f"{sample_id}__{video_arm}__{audio_arm}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def input_audit(self) -> Path:
        return self.root / "input_audit.json"

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def frames_manifest(self) -> Path:
        return self.root / "frames/manifest.json"

    @property
    def audio_manifest(self) -> Path:
        return self.root / "audio/manifest.json"

    @property
    def media_manifest(self) -> Path:
        return self.root / "media/manifest.json"

    @property
    def score_manifest(self) -> Path:
        return self.root / "scores/manifest.json"

    @property
    def oracle_analysis(self) -> Path:
        return self.root / "oracle_analysis.json"

    @property
    def oracle_validation(self) -> Path:
        return self.root / "oracle_validation.json"

    @property
    def generated_analysis(self) -> Path:
        return self.root / "generated_analysis.json"

    @property
    def generated_validation(self) -> Path:
        return self.root / "generated_validation.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def videos(self) -> Path:
        return self.root / "videos"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = "wav2lip_integer_plateau_control"
    protocol_revision: str = "integer_plateau_v1"
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    frame_width: int = FRAME_WIDTH
    frame_height: int = FRAME_HEIGHT
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    mfcc_stride: int = MFCC_STRIDE
    mfcc_audio_frames: int = MFCC_AUDIO_FRAMES
    mfcc_frame_length: int = MFCC_FRAME_LENGTH
    mfcc_frame_step: int = MFCC_FRAME_STEP
    mfcc_preemphasis_predecessor: int = MFCC_PREEMPHASIS_PREDECESSOR
    embedding_dim: int = EMBEDDING_DIM
    epsilon: float = EPSILON
    batch_size: int = BATCH_SIZE
    torch_threads: int = TORCH_THREADS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    matrix_diff_tolerance: float = MATRIX_DIFF_TOLERANCE
    validator_matrix_tolerance: float = VALIDATOR_MATRIX_TOLERANCE
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    frame_shift: int = FRAME_SHIFT
    audio_shift: int = AUDIO_SHIFT
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    ordered_sample_id_sha256: str = ORDERED_SAMPLE_ID_SHA256

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "video_arms": [VIDEO_ID, VIDEO_P],
                "audio_arms": [AUDIO_N, AUDIO_P],
                "fresh_cell_specs": [list(item) for item in FRESH_CELL_SPECS],
                "cached_cell_spec": list(CACHED_CELL_SPEC),
                "video_shift_formula": "s_v[i]=+5 if i<floor(F/2) else -5; q[i]=i+s_v[i]",
                "audio_shift_formula": "s_a[n]=+3200 if n<640*floor(F/2) else -3200; P[n]=N[n+s_a[n]]",
                "row_sets": "PLUS=arange(25,m-25), MINUS=arange(m+25,F-25), U=concatenate(PLUS,MINUS), Q=U+s_v[U]",
                "distance_formula": "sqrt(sum((visual-audio+1e-6)^2)) in float32",
                "offset_column_formula": "offset=15-column_index",
                "bootstrap_policy": "sorted source groups; PCG64/default_rng seed reset per metric; 10000 draws; linear quantiles",
                "forbidden_operations": [
                    "audio_resampling",
                    "audio_truncation",
                    "video_padding_or_looping",
                    "mixing_or_interpolation",
                    "fade_or_normalization",
                    "score_based_retry",
                    "record_filtering",
                    "training",
                    "mfa_or_dtw",
                    "bridge_execution",
                    "sealed_data_access",
                ],
            }
        )
        return value


def spec_bindings(file_sha256: Any) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("proposal", PROPOSAL), ("design", DESIGN), ("spec", SPEC))
    }


def environment() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "syncnet_python": str(SYNCNET_PYTHON),
        "wav2lip_python": str(WAV2LIP_PYTHON),
        "syncnet_model": str(SYNCNET_MODEL.resolve()),
        "syncnet_model_sha256": SYNCNET_MODEL_SHA256,
        "wav2lip_checkpoint": str(WAV2LIP_CHECKPOINT.resolve()),
        "wav2lip_checkpoint_sha256": WAV2LIP_CHECKPOINT_SHA256,
        "ffmpeg": str(FFMPEG),
        "ffprobe": str(FFPROBE),
    }
