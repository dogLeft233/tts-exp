from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
Q = REPO / "runs/wav2lip_natural_content_residual_continuation_20260909_v1"
DRIVERS = PARENT / "drivers/manifest.json"
CONTROL_SCORES = PARENT / "control_scores/manifest.json"
STATIC_FACES = PARENT / "static_faces/manifest.json"
Q_CANDIDATES = Q / "candidate_scores/manifest.json"
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")

FIXED_HASHES = {
    "drivers": "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf",
    "control_scores": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2",
    "static_faces": "e57ed01f78f4870998f69ee5ee140e1831b0975bb8195bec2cbb1c7972a4f56e",
    "q_candidates": "1ee531b15b78c21ae29c1482f12511ad403fa3fbdc9d8b9d86cd79dfd869c4de",
    "checkpoint": "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8",
    "syncnet": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
}

SAMPLE_RATE = 16_000
FPS = 25
SUPPORT_SAMPLES = 61_440
MEL_BINS = 80
MEL_FRAMES = 308
FRAME_COUNT = 93
EMBEDDING_ROWS = 88
MATRIX_COLUMNS = 31
U_ROWS = tuple(range(30, 58))
VSHIFT = 15
WAV2LIP_BATCH_SIZE = 4
SYNCNET_BATCH_SIZE = 20
SYNCNET_THREADS = 2
GENERATION_SEED = 20260909
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 20_000
GPU_LOCK = Path("/tmp/tts-exp-wav2lip-gpu0.lock")
RECORD_COUNT = 16
GROUP_COUNT = 8
REPLAY_COUNT = 2
PARITY_COUNT = 2
CANDIDATE_COUNT = 32
MAX_VIDEOS = 34
MAX_SCORES = 36


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return value


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_natural_temporal_contrast_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def p(self, name: str) -> Path:
        return self.root / name

    @property
    def protocol(self) -> Path: return self.p("protocol.json")
    @property
    def input_audit(self) -> Path: return self.p("input_audit.json")
    @property
    def drivers(self) -> Path: return self.p("drivers/manifest.json")
    @property
    def control_analysis(self) -> Path: return self.p("control_analysis.json")
    @property
    def control_scores(self) -> Path: return self.p("control_scores/manifest.json")
    @property
    def candidate_scores(self) -> Path: return self.p("candidate_scores/manifest.json")
    @property
    def analysis(self) -> Path: return self.p("analysis.json")
    @property
    def validation(self) -> Path: return self.p("validation.json")
    @property
    def review(self) -> Path: return self.p("review.json")
    @property
    def final(self) -> Path: return self.p("final.json")
    @property
    def result(self) -> Path: return self.p("result.md")


def configuration() -> dict:
    return {
        "protocol_id": "wav2lip_natural_temporal_contrast",
        "mel_shape": [MEL_BINS, MEL_FRAMES],
        "fps": FPS,
        "u_rows": list(U_ROWS),
        "kernel": [1, 4, 6, 4, 1],
        "kernel_denominator": 16,
        "edge_mode": "reflect",
        "amplitude_rule": "min(0.25, 0.5/max_abs_residual)",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "source_group", "quantile": "linear"},
        "budget": {"fresh_videos": MAX_VIDEOS, "fresh_scores": MAX_SCORES, "training": 0, "tts": 0, "vocoder": 0},
        "fixed_flags": {"replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False},
    }
