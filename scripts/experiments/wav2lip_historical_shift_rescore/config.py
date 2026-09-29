from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
H_ROOT = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904"
G_ROOT = REPO / "runs/wav2lip_global_shift_response_20260908_v4"
H_COHORT = H_ROOT / "00_protocol/cohort.json"
H_AUDIO = H_ROOT / "01_candidates/audio_manifest.json"
H_VIDEOS = H_ROOT / "03_videos/videos_manifest.json"
H_SCORES = H_ROOT / "04_scores/scores_manifest.json"
G_HISTORY = G_ROOT / "history.json"
G_PROTOCOL = G_ROOT / "protocol.json"
G_FINAL = G_ROOT / "final.json"
G_SCORES = G_ROOT / "scores/manifest.json"

INPUT_HASHES = {
    "cohort": "850c224856bc7acb95aa709a18f4c0b3369ca724d8759603d1c4bf3aac74d72a",
    "audio": "628e9e16ec4708421c9db57f636ad2871123bb0fddd5db3e33f611c109ad00cb",
    "videos": "0008acd3da7a7ec0066ce181b9c7204d34569d27e0aa4f30a92326b4561ac2bc",
    "scores": "da7bac4f19903954573752abf029a65265029fe5462c9434a21659112abcf788",
    "history": "0108fe32b7aadb748708e14e4b21483eb8658c9c7c84a1f3fb0332efba6fd61c",
    "protocol": "4c06a58f88942415b9cdb5f0ff54024eb1d403c132d19169c16d126b8919e140",
    "final": "19de6d86d3411c7da35673edd6620d417924d4fce2dec2bc291dcde97144ea6e",
    "score_manifest": "19dc5e4ee7644a08e34e23d0b8bf16d2f7b8c13b8477208fb8e1dde20791dbad",
}

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
WINDOW_FRAMES = 5
VSHIFT = 15
MATRIX_COLUMNS = 31
MFCC_STRIDE = 4
MFCC_AUDIO_FRAMES = 20
EMBEDDING_DIM = 1024
EPSILON = 1e-6
SYNCNET_BATCH_SIZE = 20
TORCH_THREADS = 4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260908
SHIFT_SAMPLES = 3_200
INNER_MARGIN = 30

VIDEO_N = "V_N"
VIDEO_SHIFT = "V_SHIFT_200"
VIDEO_ARMS = (VIDEO_N, VIDEO_SHIFT)
AUDIO_N = "N"
AUDIO_SHIFT = "SHIFT_200"

SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_historical_shift_rescore_{validate_run_id(run_id)}"


def cell_key(sample_id: str, video_arm: str) -> str:
    return f"{sample_id}__{video_arm}__N__FULL_FRAME_V4"


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
    def history(self) -> Path:
        return self.root / "history.json"

    @property
    def parity(self) -> Path:
        return self.root / "parity.json"

    @property
    def media_manifest(self) -> Path:
        return self.root / "media_manifest.json"

    @property
    def scores_manifest(self) -> Path:
        return self.root / "scores_manifest.json"

    @property
    def endpoints(self) -> Path:
        return self.root / "endpoints.json"

    @property
    def bootstrap_indices(self) -> Path:
        return self.root / "bootstrap_indices.npy"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"
