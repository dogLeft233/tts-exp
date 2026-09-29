from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

PARENT_ROOT = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix5"
PARENT_HASHES = {
    "final": "76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563",
    "protocol": "835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4",
    "control": "6944cc299f30f29a2c6f6a4f41db024014ec09aa583ca2bbc15c12133c8abce5",
    "media_manifest": "977b2f48afe03380c710daa57515068c62d345245495bf37c6d45ada509061cf",
    "score_manifest": "60950f9a26abcb37d9113541e23acd038feb8f21b827f07fb3c2f07570aa3799",
}
RECHECK_ROOT = REPO / "runs/wav2lip_roi_peak_recheck_20260906_review2"
RECHECK_HASHES = {
    "final": "478873a531db61823159b499368cc31ef9ec9f72a6fc7b80b857e42d4b37e12c",
    "validation": "883a7c6fe4596f422030dbdc996c95a7f5bb8a66c5f4b4827c68a4d9db1c86f0",
}

CHANGE_ROOT = REPO / "openspec/changes/diagnose-wav2lip-roi-retiming-oracle"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-roi-retiming-oracle/spec.md"

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

PROTOCOL_ID = "wav2lip_roi_retiming_oracle"
PROTOCOL_REVISION = "pixel_oracle_v1"
SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
VSHIFT = 15
WINDOW_FRAMES = 5
MFCC_AUDIO_FRAMES = 20
MFCC_STRIDE = 4
EMBEDDING_DIM = 1024
EPSILON = 1e-6
BATCH_SIZE = 20
TORCH_THREADS = 4
WARP_AMPLITUDE_SAMPLES = 1_920
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1.0
MATRIX_DIFF_TOLERANCE = 0.001
VALIDATOR_MATRIX_TOLERANCE = 0.0001
MIN_LOCAL_ROWS = 5
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905

VIDEO_ARMS = ("V_ID", "V_ORACLE")
AUDIO_ARMS = ("N", "W")
CELL_SPECS = tuple((video, audio) for video in VIDEO_ARMS for audio in AUDIO_ARMS)
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_STREAM_COUNT = EXPECTED_RECORD_COUNT * len(VIDEO_ARMS)
EXPECTED_MEDIA_COUNT = EXPECTED_RECORD_COUNT * len(CELL_SPECS)
EXPECTED_SCORE_CELL_COUNT = EXPECTED_MEDIA_COUNT
ORDERED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_roi_retiming_oracle_{validate_run_id(run_id)}"


def cell_key(sample_id: str, video_arm: str, audio_arm: str) -> str:
    return f"{sample_id}__{video_arm}__{audio_arm}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def input_audit(self) -> Path:
        return self.root / "input_audit.json"

    @property
    def frame_manifest(self) -> Path:
        return self.root / "frames/manifest.json"

    @property
    def audio_manifest(self) -> Path:
        return self.root / "audio/manifest.json"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def media_manifest(self) -> Path:
        return self.media / "manifest.json"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def score_manifest(self) -> Path:
        return self.scores / "manifest.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = PROTOCOL_ID
    protocol_revision: str = PROTOCOL_REVISION
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    frame_width: int = FRAME_WIDTH
    frame_height: int = FRAME_HEIGHT
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    warp_amplitude_samples: int = WARP_AMPLITUDE_SAMPLES
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: float = OFFSET_TOLERANCE_FRAMES
    matrix_diff_tolerance: float = MATRIX_DIFF_TOLERANCE
    validator_matrix_tolerance: float = VALIDATOR_MATRIX_TOLERANCE
    min_local_rows: int = MIN_LOCAL_ROWS
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_stream_count: int = EXPECTED_STREAM_COUNT
    expected_media_count: int = EXPECTED_MEDIA_COUNT
    expected_score_cell_count: int = EXPECTED_SCORE_CELL_COUNT
    ordered_sample_id_sha256: str = ORDERED_SAMPLE_ID_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "video_arms": list(VIDEO_ARMS),
                "audio_arms": list(AUDIO_ARMS),
                "cell_specs": [list(item) for item in CELL_SPECS],
                "warp_formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced",
                "oracle_index_formula": "floor(interp(s, t_i)/640 + 0.5), t_i=640*i",
                "rounding": "nearest-half-up for frame indices; np.rint nearest-even for PCM16",
                "video_format": "FFV1/Matroska, bgr0, color_range=pc, color_space=gbr",
                "offset_column_formula": "offset=15-column_index",
                "bootstrap_policy": "sorted source groups; PCG64 seed reset per metric; 10000 draws; linear quantiles",
                "forbidden_operations": [
                    "tts_generation",
                    "wav2lip_generation",
                    "audio_resampling",
                    "audio_truncation",
                    "video_padding_or_looping",
                    "score_based_retry",
                    "record_filtering",
                    "training",
                    "mfa_or_dtw",
                    "bridge_execution",
                    "sealed_data_access",
                ],
            }
        )
        return payload


def spec_bindings() -> dict[str, dict[str, str]]:
    from .common import file_sha256

    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("proposal", PROPOSAL), ("design", DESIGN), ("spec", SPEC))
    }
