from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUN_PREFIX = "static_image_bridge"
PROTOCOL_ID = "static_image_natural_to_tts_bridge"
PROTOCOL_REVISION = "static_reference_v1"

PARENT_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
PARENT_COHORT = PARENT_ROOT / "00_protocol/cohort.json"
PARENT_AUDIO = PARENT_ROOT / "01_audio/audio_manifest.json"
PARENT_ANALYSIS = PARENT_ROOT / "04_final/analysis.json"
DISCOVERY_ANALYSIS = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904/05_final/analysis.json"

PARENT_COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
PARENT_AUDIO_SHA256 = "2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288"
PARENT_ANALYSIS_SHA256 = "0841a0e95efb8747516f866427bafa601c1fa5270ca4bcaaf6cfd396fc77556f"
DISCOVERY_ANALYSIS_SHA256 = "965e4bbe34ab1538b0e806225faaedc42d67f2243e9c84df4266bd471461228f"

EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_SAMPLE_IDS_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
MIN_AUDIO_SAMPLES = 1024
SEED = 42

WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
FFMPEG_SHA256 = "019d92fd5839dfeabf5c67cbdf8408e1d7bdf0f1528cece52a19e5fa89593fcd"
FFPROBE_SHA256 = "14b31fde8eb1e12e6f53fb8de706d9475604f8b74548757f978c7bedbfd3c3f2"
PYTHON_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"

WAV2LIP_BATCH_SIZE = 4
FACE_DET_THRESHOLD = 0.9
GENERATION_BOTTOM_PAD = 10
SYNCNET_VSHIFT = 15
SYNCNET_BATCH_SIZE = 20
MIN_COMMON_WINDOWS = 20
MIN_SWAP_WINDOWS = 5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260913
GAIN_THRESHOLD = 0.05
DELAY_D_ANCHOR_THRESHOLD = 0.10
DELAY_OFFSET_TOLERANCE = 1
DELAY_MIN_RECORDS = 20
DELAY_POSITIVE_MIN_RECORDS = 18
REPEAT_DIFF_BOUND = 0.05
REPEAT_RECORD_ABS_BOUND = 0.10
REPEAT_OFFSET_BOUND = 1
SWAP_MARGIN = 0.10
SWAP_MIN_POSITIVE_RECORDS = 18

ARMS = ("N", "N_REPEAT", "RT", "B", "S")
SCORE_AUDIO_ARMS = ("N", "ND", "B", "S")
STAGE_A_VIDEOS = ("N", "N_REPEAT")
STAGE_B_VIDEOS = ("RT", "B", "S")
STAGE_A_CELLS = (
    ("N", "N"),
    ("N_REPEAT", "N"),
    ("N", "ND"),
)
STAGE_B_CELLS = (
    ("RT", "N"),
    ("B", "N"),
    ("B", "B"),
    ("N", "S"),
    ("S", "N"),
    ("S", "S"),
)
ALL_CELLS = STAGE_A_CELLS + STAGE_B_CELLS

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"{RUN_PREFIX}_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.json"

    @property
    def audio_manifest(self) -> Path:
        return self.root / "audio_manifest.json"

    @property
    def video_manifest(self) -> Path:
        return self.root / "video_manifest.json"

    @property
    def scores_manifest(self) -> Path:
        return self.root / "scores_manifest.json"

    @property
    def stage_a(self) -> Path:
        return self.root / "stage_a.json"

    @property
    def support(self) -> Path:
        return self.root / "support"

    @property
    def matrices(self) -> Path:
        return self.root / "matrices"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def image_dir(self) -> Path:
        return self.root / "images"

    @property
    def video_dir(self) -> Path:
        return self.root / "videos"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def playback(self) -> Path:
        return self.root / "playback"

    @property
    def per_record_csv(self) -> Path:
        return self.root / "per_record.csv"

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
    def report(self) -> Path:
        return self.root / "report.md"


def cell_key(video_arm: str, audio_arm: str) -> str:
    return f"V_{video_arm}/A_{audio_arm}"


def cell_id(sample_id: str, video_arm: str, audio_arm: str) -> str:
    return f"{sample_id}__{video_arm}_{audio_arm}"


def expected_video_count(stage: str | None = None) -> int:
    if stage == "A":
        return EXPECTED_RECORD_COUNT * len(STAGE_A_VIDEOS)
    if stage == "B":
        return EXPECTED_RECORD_COUNT * len(STAGE_B_VIDEOS)
    return EXPECTED_RECORD_COUNT * len(ARMS)


def expected_cell_count(stage: str | None = None) -> int:
    if stage == "A":
        return EXPECTED_RECORD_COUNT * len(STAGE_A_CELLS)
    if stage == "B":
        return EXPECTED_RECORD_COUNT * len(STAGE_B_CELLS)
    return EXPECTED_RECORD_COUNT * len(ALL_CELLS)


def runtime_bindings() -> dict[str, Any]:
    return {
        "wav2lip_root": str(WAV2LIP_ROOT.resolve()),
        "wav2lip_python": str(WAV2LIP_PYTHON.resolve()),
        "wav2lip_checkpoint": str(WAV2LIP_CHECKPOINT.resolve()),
        "syncnet_root": str(SYNCNET_ROOT.resolve()),
        "syncnet_python": str(SYNCNET_PYTHON.resolve()),
        "syncnet_model": str(SYNCNET_MODEL.resolve()),
        "ffmpeg": str(FFMPEG.resolve()),
        "ffprobe": str(FFPROBE.resolve()),
        "seed": SEED,
        "wav2lip_batch_size": WAV2LIP_BATCH_SIZE,
        "syncnet_batch_size": SYNCNET_BATCH_SIZE,
    }
