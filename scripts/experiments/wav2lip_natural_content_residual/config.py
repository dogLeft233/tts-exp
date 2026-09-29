from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
COHORT = PARENT / "00_cohort/manifest.json"
NATURAL_MELS = PARENT / "03_data/natural_mels.npz"
MASK_MANIFEST = PARENT / "03_data/mask_manifest.json"
RECONSTRUCTION = PARENT / "04_reconstruction/reconstruction.json"
DRIVERS = PARENT / "05_drivers/drivers.json"
BOX_MANIFEST = PARENT / "06_renders/box_manifest.json"

PARENT_HASHES = {
    "cohort": "8ec7a2ec0d37c6abf51075d57f1467658fc0b3de9574799a1b8f08f9a2bf21b5",
    "natural_mels": "b48628ee3f09c0e46abc3d121d717bc72af10384cf7f20ea8caa2f5a7aaff44f",
    "mask_manifest": "956d7eb5bd7fe6b12a658191ac0ab80a9a8a6af77a0fce2aebcec30be561deec",
    "reconstruction": "79f79c0e1a5a439324dce1d35c0e2861ce428ece7f87f6c78eb195871f757df6",
    "drivers": "ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410",
    "box_manifest": "1a1304633e2eae0b525e5b7f4a06391cf8c28a52668d7ea6e1598fc1f68a17be",
}

PARITY = REPO / "runs/wav2lip_historical_shift_rescore_20260908_v7/parity.json"
PARITY_HASH = "56faf1727b6091e1dec7c2c8f9e37b330007e9fd195fdf151bfaa0e4a10057a5"

SPEC = REPO / "openspec/changes/probe-wav2lip-natural-content-residual/specs/wav2lip-natural-content-residual/spec.md"
DESIGN = REPO / "openspec/changes/probe-wav2lip-natural-content-residual/design.md"
PROPOSAL = REPO / "openspec/changes/probe-wav2lip-natural-content-residual/proposal.md"

WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
CROP_SIZE = 224
SUPPORT_SAMPLES = 61_440
MEL_FRAMES = 308
MEL_BINS = 80
WAV2LIP_FRAMES = 93
MATRIX_ROWS = 88
U_START = 30
U_STOP = 58
VSHIFT = 15
WINDOW_FRAMES = 5
WAV2LIP_BATCH_SIZE = 4
SYNCNET_BATCH_SIZE = 20
SYNCNET_THREADS = 4

SEEDS = (20260901, 20260902, 20260903)
DRIVER_CONDITIONS = ("PAIRED_TTS", "SAME_PHONE_WRONG_INSTANCE", "NAT_ONLY")
ARMS = ("N", "CORRECT", "WRONG", "SHUFFLE")
CONTROL_ARMS = ("N", "N_REPEAT", "A_DELAY")
CANDIDATE_ARMS = ("CORRECT", "WRONG", "SHUFFLE")
SHUFFLE_SALT = b"natural-content-residual-v1\0"
RUN_PREFIX = "wav2lip_natural_content_residual"
GENERATION_SEED = 20260909
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_DRAWS = 10_000
EXPECTED_RECORD_COUNT = 16
EXPECTED_GROUP_COUNT = 8
EXPECTED_CANDIDATE_VIDEO_COUNT = 48
EXPECTED_CONTROL_SCORE_COUNT = 36
EXPECTED_CANDIDATE_SCORE_COUNT = 48
MAX_VIDEO_COUNT = 66
MAX_SCORE_COUNT = 84


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
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def input_audit(self) -> Path:
        return self.root / "input_audit.json"

    @property
    def driver_manifest(self) -> Path:
        return self.root / "drivers/manifest.json"

    @property
    def static_face_manifest(self) -> Path:
        return self.root / "static_faces/manifest.json"

    @property
    def controls(self) -> Path:
        return self.root / "control_analysis.json"

    @property
    def control_scores(self) -> Path:
        return self.root / "control_scores/manifest.json"

    @property
    def control_validation(self) -> Path:
        return self.root / "control_validation.json"

    @property
    def candidates(self) -> Path:
        return self.root / "candidate_scores/manifest.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"


def spec_paths() -> dict[str, Path]:
    return {"spec": SPEC, "design": DESIGN, "proposal": PROPOSAL}
