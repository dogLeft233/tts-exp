from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PARENT_ROOT = REPO / "runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4"
PARENT_FILES = {
    "final": PARENT_ROOT / "final.json",
    "validation": PARENT_ROOT / "validation.json",
    "protocol": PARENT_ROOT / "protocol.json",
    "input_audit": PARENT_ROOT / "input_audit.json",
    "analysis": PARENT_ROOT / "analysis.json",
    "frame_manifest": PARENT_ROOT / "frames/manifest.json",
    "audio_manifest": PARENT_ROOT / "audio/manifest.json",
    "media_manifest": PARENT_ROOT / "media/manifest.json",
    "score_manifest": PARENT_ROOT / "scores/manifest.json",
}
PARENT_HASHES = {
    "final": "919a5af95e361cbe218fc2d88003aa74d2e061d68d3543fe8fcbc7466cc0cff1",
    "validation": "a8e4719af2aa1e15477871caadea16f2c7b6fc69b4404c71f19bca2d33546724",
    "protocol": "6bec09cb8abdc7d8de9b50dfb388f1395924b24c9b130b808a9c5387378fa079",
    "input_audit": "f2b6603d4999db60ecd44b27a7396f25b6d582b43ff93abd2bac3b4ee6efc5c6",
    "analysis": "74ab3eb0b3fb0f8d3ced5be53117c2b9ea90fc2f7214b5aa1f8d7cec2d3d365e",
    "frame_manifest": "3a713cf922dbe9c9849715d93ab0a7a219fe69502b9df5117ba50e4401a051e3",
    "audio_manifest": "4ea22a2fa2838891661d4cc5d18ccc3e3d4857d489afadd2ee42dcee5c39d3dd",
    "media_manifest": "609cded610f1807f532fd80964c02e837521f7493c8214ed3c75eb6c2fe2a1da",
    "score_manifest": "d25b08e26534c8a407eba594413ff5eb19cad9d4fe4160f0b9714a46e9148842",
}

CHANGE_ROOT = REPO / "openspec/changes/diagnose-wav2lip-oracle-frame-interpolation"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-oracle-frame-interpolation/spec.md"

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

PROTOCOL_ID = "wav2lip_oracle_frame_interpolation"
PROTOCOL_REVISION = "linear_pixel_v1"
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
VALIDATOR_MATRIX_TOLERANCE = 0.0001
STAT_TOLERANCE = 1e-6
MIN_LOCAL_ROWS = 5
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905

VIDEO_ARM = "V_LINEAR"
AUDIO_ARMS = ("N", "W")
FRESH_CELL_SPECS = tuple((VIDEO_ARM, audio) for audio in AUDIO_ARMS)
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_FRESH_STREAM_COUNT = EXPECTED_RECORD_COUNT
EXPECTED_FRESH_MEDIA_COUNT = EXPECTED_RECORD_COUNT * len(FRESH_CELL_SPECS)
EXPECTED_FRESH_SCORE_CELL_COUNT = EXPECTED_FRESH_MEDIA_COUNT
EXPECTED_CACHED_SCORE_CELL_COUNT = 88
ORDERED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_oracle_frame_interpolation_{validate_run_id(run_id)}"


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
    def frames_manifest(self) -> Path:
        return self.root / "frames/manifest.json"

    @property
    def media_manifest(self) -> Path:
        return self.root / "media/manifest.json"

    @property
    def scores_manifest(self) -> Path:
        return self.root / "scores/manifest.json"

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
    validator_matrix_tolerance: float = VALIDATOR_MATRIX_TOLERANCE
    stat_tolerance: float = STAT_TOLERANCE
    min_local_rows: int = MIN_LOCAL_ROWS
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_fresh_stream_count: int = EXPECTED_FRESH_STREAM_COUNT
    expected_fresh_media_count: int = EXPECTED_FRESH_MEDIA_COUNT
    expected_fresh_score_cell_count: int = EXPECTED_FRESH_SCORE_CELL_COUNT
    expected_cached_score_cell_count: int = EXPECTED_CACHED_SCORE_CELL_COUNT
    ordered_sample_id_sha256: str = ORDERED_SAMPLE_ID_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "video_arm": VIDEO_ARM,
                "audio_arms": list(AUDIO_ARMS),
                "fresh_cell_specs": [list(item) for item in FRESH_CELL_SPECS],
                "linear_formula": "u=interp(640*i, arange(L), s)/640; Y=(1-w)X[floor(u)]+wX[ceil(u)]; floor(Y+0.5)",
                "warp_formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced",
                "rounding": "float64 pixel arithmetic; per-channel nearest-half-up to uint8",
                "video_format": "FFV1/Matroska, bgr0, color_range=pc, color_space=gbr",
                "offset_column_formula": "offset=15-column_index",
                "bootstrap_policy": "sorted source groups; PCG64 seed reset per metric; 10000 draws; linear quantiles",
                "forbidden_operations": [
                    "using_V_ORACLE_as_interpolation_input",
                    "embedding_interpolation",
                    "tts_generation",
                    "wav2lip_generation",
                    "audio_resampling",
                    "audio_truncation",
                    "score_based_retry",
                    "record_filtering",
                    "training",
                    "bridge_execution",
                    "mfa_or_dtw",
                    "sealed_data_access",
                ],
            }
        )
        return payload


def spec_bindings() -> dict[str, dict[str, str]]:
    from scripts.experiments.wav2lip_roi_retiming_oracle.common import file_sha256

    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("proposal", PROPOSAL), ("design", DESIGN), ("spec", SPEC))
    }
