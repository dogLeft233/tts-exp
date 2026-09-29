from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/diagnose-wav2lip-global-shift-response"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-global-shift-response/spec.md"

H_ROOT = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904"
H_COHORT = H_ROOT / "00_protocol/cohort.json"
H_AUDIO = H_ROOT / "01_candidates/audio_manifest.json"
H_VIDEOS = H_ROOT / "03_videos/videos_manifest.json"
H_SCORES = H_ROOT / "04_scores/scores_manifest.json"
H_FINAL = H_ROOT / "05_final/final.json"

S_ROOT = REPO / "runs/wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3"
S_PROTOCOL = S_ROOT / "protocol.json"

H_HASHES = {
    "cohort": "850c224856bc7acb95aa709a18f4c0b3369ca724d8759603d1c4bf3aac74d72a",
    "audio": "628e9e16ec4708421c9db57f636ad2871123bb0fddd5db3e33f611c109ad00cb",
    "videos": "0008acd3da7a7ec0066ce181b9c7204d34569d27e0aa4f30a92326b4561ac2bc",
    "scores": "da7bac4f19903954573752abf029a65265029fe5462c9434a21659112abcf788",
    "final": "9bab2a853edda0cdb16e79e2b9a8c6d32ce3e36d721e4a1c1140306d9ffac8c1",
    "s_protocol": "284dea1b07fbccc07a309512b164265d215075af41b8009b38d163362d502cfa",
}

EXPECTED_IDS = (
    "lrs3_6ORDQFh0Byw_00008",
    "lrs3_6SdtkXAQq3k_00009",
    "lrs3_6VnKV1sr5VQ_00008",
    "lrs3_6WeS1bXRBOk_00006",
    "lrs3_6qqqVwM6bMM_00007",
    "lrs3_6tSlMoMNSlY_00009",
    "lrs3_6ul2TSvUDog_00007",
    "lrs3_6weGCM3sWKc_00014",
    "lrs3_6wk4dkYSrV0_00006",
    "lrs3_6xtmm0MnaS0_00010",
    "lrs3_6xy5pWOgeBY_00010",
    "lrs3_6yR5OUVb2gY_00008",
)

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
VSHIFT = 15
WINDOW_FRAMES = 5
MATRIX_COLUMNS = 2 * VSHIFT + 1
MFCC_STRIDE = 4
MFCC_AUDIO_FRAMES = 20
MFCC_FRAME_LENGTH = 400
MFCC_FRAME_STEP = 160
EMBEDDING_DIM = 1024
EPSILON = 1e-6
GENERATION_BATCH_SIZE = 4
SYNCNET_BATCH_SIZE = 20
TORCH_THREADS = 4
FRAME_SHIFT = 5
AUDIO_SHIFT = FRAME_SHIFT * SAMPLES_PER_FRAME
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260908
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
DISK_FREE_MIN_BYTES = 15 * 1024**3

FACE_DYNAMIC = "DYNAMIC"
FACE_STATIC = "STATIC"
FACE_MODES = (FACE_DYNAMIC, FACE_STATIC)
VIDEO_N = "V_N"
VIDEO_N_REPEAT = "V_N_REPEAT"
VIDEO_DELAY = "V_DELAY_200"
VIDEO_ADVANCE = "V_ADVANCE_200"
VIDEO_ARMS = (VIDEO_N, VIDEO_N_REPEAT, VIDEO_DELAY, VIDEO_ADVANCE)
AUDIO_N = "N"
AUDIO_N_REPEAT = "N_REPEAT"
AUDIO_DELAY = "DELAY_200"
AUDIO_ADVANCE = "ADVANCE_200"
AUDIO_ARMS = (AUDIO_N, AUDIO_N_REPEAT, AUDIO_DELAY, AUDIO_ADVANCE)

SCORE_CELLS = (
    (VIDEO_N, AUDIO_N),
    (VIDEO_N, AUDIO_DELAY),
    (VIDEO_N, AUDIO_ADVANCE),
    (VIDEO_N_REPEAT, AUDIO_N),
    (VIDEO_DELAY, AUDIO_N),
    (VIDEO_DELAY, AUDIO_DELAY),
    (VIDEO_ADVANCE, AUDIO_N),
    (VIDEO_ADVANCE, AUDIO_ADVANCE),
)
EXPECTED_RECORD_COUNT = len(EXPECTED_IDS)
EXPECTED_VIDEO_COUNT = EXPECTED_RECORD_COUNT * len(FACE_MODES) * len(VIDEO_ARMS)
EXPECTED_SCORE_COUNT = EXPECTED_RECORD_COUNT * len(FACE_MODES) * len(SCORE_CELLS)

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


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_global_shift_response_{validate_run_id(run_id)}"


def cell_key(sample_id: str, face_mode: str, video_arm: str, audio_arm: str) -> str:
    return f"{sample_id}__{face_mode}__{video_arm}__{audio_arm}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def input_audit(self) -> Path: return self.root / "input_audit.json"
    @property
    def protocol(self) -> Path: return self.root / "protocol.json"
    @property
    def audio_manifest(self) -> Path: return self.root / "audio/manifest.json"
    @property
    def faces_manifest(self) -> Path: return self.root / "faces/manifest.json"
    @property
    def history(self) -> Path: return self.root / "history.json"
    @property
    def videos_manifest(self) -> Path: return self.root / "videos/manifest.json"
    @property
    def scores_manifest(self) -> Path: return self.root / "scores/manifest.json"
    @property
    def bootstrap_indices(self) -> Path: return self.root / "bootstrap_indices.npy"
    @property
    def history_bootstrap_indices(self) -> Path: return self.root / "history_bootstrap_indices.npy"
    @property
    def visual_response(self) -> Path: return self.root / "visual_response.json"
    @property
    def endpoints(self) -> Path: return self.root / "endpoints.json"
    @property
    def analysis(self) -> Path: return self.root / "analysis.json"
    @property
    def validation(self) -> Path: return self.root / "validation.json"
    @property
    def review(self) -> Path: return self.root / "review.json"
    @property
    def final(self) -> Path: return self.root / "final.json"
    @property
    def result(self) -> Path: return self.root / "result.md"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = "wav2lip_global_shift_response"
    protocol_revision: str = "global_shift_response_v1"
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    matrix_columns: int = MATRIX_COLUMNS
    mfcc_stride: int = MFCC_STRIDE
    mfcc_audio_frames: int = MFCC_AUDIO_FRAMES
    mfcc_frame_length: int = MFCC_FRAME_LENGTH
    mfcc_frame_step: int = MFCC_FRAME_STEP
    embedding_dim: int = EMBEDDING_DIM
    epsilon: float = EPSILON
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    frame_shift: int = FRAME_SHIFT
    audio_shift: int = AUDIO_SHIFT
    generation_batch_size: int = GENERATION_BATCH_SIZE
    syncnet_batch_size: int = SYNCNET_BATCH_SIZE
    torch_threads: int = TORCH_THREADS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_video_count: int = EXPECTED_VIDEO_COUNT
    expected_score_count: int = EXPECTED_SCORE_COUNT

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "face_modes": list(FACE_MODES),
                "video_arms": list(VIDEO_ARMS),
                "audio_arms": list(AUDIO_ARMS),
                "score_cells": [list(item) for item in SCORE_CELLS],
                "delay_formula": "DELAY_200[n]=N[n-3200], out-of-range zero",
                "advance_formula": "ADVANCE_200[n]=N[n+3200], out-of-range zero",
                "distance_formula": "sqrt(sum((visual-audio+1e-6)^2)) in float32",
                "endpoint_formula": "mean rows first; D=min(z); C=median(z)-D; offset=15-argmin(z)",
                "fixed_flags": {
                    "training_authorized": False,
                    "generalization_established": False,
                    "historical_gate_repaired": False,
                },
            }
        )
        return payload


def spec_paths() -> dict[str, Path]:
    return {"proposal": PROPOSAL, "design": DESIGN, "spec": SPEC}
