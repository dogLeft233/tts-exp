from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]

PARENT = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
DIAGNOSTIC = REPO / "runs/wav2lip_delay_search_support_20260909_v1"

PARENT_FILES = {
    "protocol.json": "e0ad38110883bc7079e66866b5187f68636ee2ec8882ddba94879b203a5a2485",
    "drivers/manifest.json": "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf",
    "control_scores/manifest.json": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2",
    "control_analysis.json": "068bf6ec3d43e432032ad6bfde8ce7cc284669b0c020575eb4ccf77bb30f9041",
    "control_validation.json": "9ae69db31e1f4b636a0b09e2793a86b373b58f183d8589f8a10f60f11d9138c6",
    "final.json": "a9a423d96cddd774a3b4ac570f1d1cdceddd6e6eb700dd0b668c652faeb473aa",
}
DIAGNOSTIC_FILES = {
    "protocol.json": "4b8fbd105c38edc09a03b9adbfeb01b90812adc44c150a583dfa9d5da4f87968",
    "analysis.json": "c6db175908c7be7964d0b9d2cf71d870f48a1466b57d3b729d286492ba675a41",
    "validation.json": "7a3e46012d15fa22cb25ccfb21d0e5b3ab3ffc7d5d9400432c61816dce761c2e",
    "final.json": "8e7fb08dcbf10c688d87a14f5e131ca1d8e96fb8259f399a9e6810c30a178f26",
}

PARENT_PROTOCOL = PARENT / "protocol.json"
PARENT_DRIVERS = PARENT / "drivers/manifest.json"
PARENT_SCORES = PARENT / "control_scores/manifest.json"
PARENT_CONTROL = PARENT / "control_analysis.json"
PARENT_VALIDATION = PARENT / "control_validation.json"
PARENT_FINAL = PARENT / "final.json"

DIAGNOSTIC_PROTOCOL = DIAGNOSTIC / "protocol.json"
DIAGNOSTIC_ANALYSIS = DIAGNOSTIC / "analysis.json"
DIAGNOSTIC_VALIDATION = DIAGNOSTIC / "validation.json"
DIAGNOSTIC_FINAL = DIAGNOSTIC / "final.json"

CHANGE_ROOT = REPO / "openspec/changes/resume-wav2lip-natural-content-residual"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-natural-content-residual-continuation/spec.md"

ORIGINAL_CHANGE = REPO / "openspec/changes/probe-wav2lip-natural-content-residual"
ORIGINAL_DESIGN = ORIGINAL_CHANGE / "design.md"
ORIGINAL_SPEC = ORIGINAL_CHANGE / "specs/wav2lip-natural-content-residual/spec.md"

ORIGINAL_DESIGN_SHA256 = "1b06fc44d5ecc96a28eeb907d3692517e8bacd3cafb2ed3d9f08fb96465a3286"
ORIGINAL_SPEC_SHA256 = "d512c5ed4ef4153d536b16d1ca74ff367df02f5bead97f2014d713a763039cce"

PARITY_MANIFEST = REPO / "runs/wav2lip_historical_shift_rescore_20260908_v7/parity.json"
PARITY_MANIFEST_SHA256 = "56faf1727b6091e1dec7c2c8f9e37b330007e9fd195fdf151bfaa0e4a10057a5"

PROTOCOL_ID = "wav2lip_natural_content_residual_continuation"
PROTOCOL_REVISION = "matched_delay_continuation_v1"
RUN_PREFIX = "wav2lip_natural_content_residual_continuation"

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
DELAY_FRAMES = 5
DELAY_SAMPLES = DELAY_FRAMES * SAMPLES_PER_FRAME
MEL_BINS = 80
MEL_FRAMES = 308
CROP_SIZE = 224
WAV2LIP_FRAMES = 93
EMBEDDING_ROWS = 88
EMBEDDING_DIM = 1024
MATRIX_COLUMNS = 31
U_ROWS = tuple(range(30, 58))
SUPPORT_SAMPLES = 61_440
VSHIFT = 15
NATURAL_LAG_START = -15
DELAY_LAG_START = -10
OFFSET_LOW = -6
OFFSET_HIGH = -4
MATRIX_TOLERANCE = 1e-4
METRIC_TOLERANCE = 1e-6

SEEDS = (20260901, 20260902, 20260903)
ARMS = ("N", "CORRECT", "WRONG", "SHUFFLE")
CANDIDATE_ARMS = ("CORRECT", "WRONG", "SHUFFLE")
SHUFFLE_SALT = b"natural-content-residual-v1\0"
GENERATION_SEED = 20260909
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_DRAWS = 10_000
EXPECTED_RECORD_COUNT = 16
EXPECTED_GROUP_COUNT = 8
EXPECTED_REPLAY_COUNT = 2
EXPECTED_PARITY_COUNT = 2
EXPECTED_CANDIDATE_VIDEO_COUNT = 48
EXPECTED_CANDIDATE_SCORE_COUNT = 48
MAX_NEW_VIDEO_COUNT = 50
MAX_NEW_SCORE_COUNT = 52

WAV2LIP_BATCH_SIZE = 4
SYNCNET_BATCH_SIZE = 20
SYNCNET_THREADS = 4
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"{RUN_PREFIX}_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path: return self.root / "protocol.json"
    @property
    def input_audit(self) -> Path: return self.root / "input_audit.json"
    @property
    def reuse_manifest(self) -> Path: return self.root / "reuse_manifest.json"
    @property
    def drivers(self) -> Path: return self.root / "drivers/manifest.json"
    @property
    def control_analysis(self) -> Path: return self.root / "control_analysis.json"
    @property
    def control_scores(self) -> Path: return self.root / "control_scores/manifest.json"
    @property
    def control_validation(self) -> Path: return self.root / "control_validation.json"
    @property
    def candidate_scores(self) -> Path: return self.root / "candidate_scores/manifest.json"
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
    @property
    def media(self) -> Path: return self.root / "media"
    @property
    def scores(self) -> Path: return self.root / "scores"


def spec_paths() -> dict[str, Path]:
    return {
        "proposal": PROPOSAL,
        "design": DESIGN,
        "spec": SPEC,
        "original_design": ORIGINAL_DESIGN,
        "original_spec": ORIGINAL_SPEC,
    }


def configuration() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "sample_rate": SAMPLE_RATE,
        "fps": FPS,
        "delay_frames": DELAY_FRAMES,
        "natural_lags": list(range(NATURAL_LAG_START, NATURAL_LAG_START + MATRIX_COLUMNS)),
        "delay_lags": list(range(DELAY_LAG_START, DELAY_LAG_START + MATRIX_COLUMNS)),
        "natural_offset": "15-j",
        "delay_offset": "10-j",
        "u_rows": list(U_ROWS),
        "matrix_tolerance": MATRIX_TOLERANCE,
        "metric_tolerance": METRIC_TOLERANCE,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "rng": "numpy.default_rng/PCG64", "quantile": "linear", "unit": "source_group"},
        "budget": {"new_videos": MAX_NEW_VIDEO_COUNT, "new_scores": MAX_NEW_SCORE_COUNT, "training": 0, "new_tts": 0, "new_vocoder": 0},
        "fixed_flags": {"stage_b_authorized": False, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False},
    }
