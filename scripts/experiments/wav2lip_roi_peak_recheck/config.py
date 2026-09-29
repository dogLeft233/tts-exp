from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

PARENT_ROOT = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix5"
DIAGNOSTIC_ROOT = REPO / "runs/wav2lip_roi_control_diagnostic_20260906_audit4"
CHANGE_ROOT = REPO / "openspec/changes/recheck-wav2lip-roi-local-peaks"

PARENT_HASHES = {
    "final": "76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563",
    "validation": "328e562cc8dacfb960fb2319fb15f1cfa90eecf593f04434382c9a5e13bd2577",
    "control": "6944cc299f30f29a2c6f6a4f41db024014ec09aa583ca2bbc15c12133c8abce5",
    "protocol": "835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4",
    "score_manifest": "60950f9a26abcb37d9113541e23acd038feb8f21b827f07fb3c2f07570aa3799",
}
DIAGNOSTIC_FINAL_SHA256 = "f575515b90da58c348565fbbf2980cf2f7e355ef5b308c3d77ffccb9c7b12d05"
DIAGNOSTIC_DIAGNOSTICS_SHA256 = "24f8c52440499f5f86985439ec87e8921f479fbeba13357463ae1e05405ed608"
DIAGNOSTIC_VALIDATION = DIAGNOSTIC_ROOT / "validation.json"

PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-roi-local-peak-recheck/spec.md"

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")

PROTOCOL_ID = "wav2lip_roi_local_peak_recheck"
PROTOCOL_REVISION = "independent_forward_v1"
SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
VSHIFT = 15
WINDOW_FRAMES = 5
MFCC_AUDIO_FRAMES = 20
MFCC_STRIDE = 4
EMBEDDING_DIM = 1024
EPSILON = 1e-6
BATCH_SIZE = 20
TORCH_THREADS = 4
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1.0
MATRIX_DIFF_TOLERANCE = 0.001
CURVE_DIFF_TOLERANCE = 0.001
VALIDATOR_MATRIX_TOLERANCE = 0.0001

VIDEO_ARMS = ("G_N", "G_W")
AUDIO_ARM = "N"
REPEAT = False

FAIL_IDS = (
    "lrs3_6wk4dkYSrV0_00006",
    "lrs3_6qqqVwM6bMM_00007",
    "lrs3_73cTNHEQhkQ_00007",
    "lrs3_796LfXwzIUk_00007",
    "lrs3_7CIq4mtiamY_00007",
    "lrs3_7DCofMA9eQA_00007",
    "lrs3_6ydYeyNSQVY_00008",
    "lrs3_79tRTivyMSM_00014",
)
PASS_IDS = (
    "lrs3_6WeS1bXRBOk_00006",
    "lrs3_6ul2TSvUDog_00007",
)
EXPECTED_RECORD_COUNT = 10
EXPECTED_CELL_COUNT = 20


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_roi_peak_recheck_{validate_run_id(run_id)}"


def cell_key(sample_id: str, video_arm: str, audio_arm: str = AUDIO_ARM) -> str:
    return f"{sample_id}__{video_arm}__{audio_arm}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def audit(self) -> Path:
        return self.root / "audit.json"

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
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    mfcc_audio_frames: int = MFCC_AUDIO_FRAMES
    mfcc_stride: int = MFCC_STRIDE
    embedding_dim: int = EMBEDDING_DIM
    epsilon: float = EPSILON
    batch_size: int = BATCH_SIZE
    torch_threads: int = TORCH_THREADS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: float = OFFSET_TOLERANCE_FRAMES
    matrix_diff_tolerance: float = MATRIX_DIFF_TOLERANCE
    curve_diff_tolerance: float = CURVE_DIFF_TOLERANCE
    validator_matrix_tolerance: float = VALIDATOR_MATRIX_TOLERANCE
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_cell_count: int = EXPECTED_CELL_COUNT

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "video_arms": list(VIDEO_ARMS),
                "audio_arm": AUDIO_ARM,
                "repeat": REPEAT,
                "offset_column_formula": "offset=15-column_index",
                "distance_formula": "sqrt(sum((visual-audio_or_zero+1e-6)^2)) in float32",
                "visual_window": "frames[r:r+5]",
                "audio_window": "mfcc[:,4*r:4*r+20]",
                "mfcc_contract": "python_speech_features.mfcc default parameters on int16 PCM",
                "frame_contract": "ffmpeg image2 threads=1; cv2 BGR; float32 0..255; no resize",
                "audio_contract": "ffmpeg async=1 mono PCM16 16000 Hz",
                "forbidden_operations": [
                    "historical_embedding_or_matrix_as_new_input",
                    "SyncNetInstance.evaluate",
                    "SyncNetInstance.calc_pdist",
                    "score_based_retry",
                    "record_replacement",
                    "threshold_adjustment",
                    "bridge_execution",
                    "training",
                ],
            }
        )
        return value
