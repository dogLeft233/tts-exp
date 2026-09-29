from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HISTORY_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
HISTORY_COHORT = HISTORY_ROOT / "00_protocol/cohort.json"
HISTORY_AUDIO = HISTORY_ROOT / "01_audio/audio_manifest.json"
HISTORY_VIDEOS = HISTORY_ROOT / "02_videos/videos_manifest.json"
HISTORY_SCORES = HISTORY_ROOT / "03_scores/scores_manifest.json"
HISTORY_FINAL = HISTORY_ROOT / "04_final/final.json"

PROTOCOL_ID = "local_swap_minimal_replay"
PROTOCOL_REVISION = "known_pairing_2x2_v1"
RUN_PREFIX = "local_swap_minimal_replay"

SAMPLE_IDS = (
    "lrs3_6WeS1bXRBOk_00006",
    "lrs3_6ul2TSvUDog_00007",
    "lrs3_6wk4dkYSrV0_00006",
)
EXPECTED_HISTORY_COUNT = 22
EXPECTED_NEW_CELL_COUNT = 24
SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
WINDOW_FRAMES = 5
VSHIFT = 15
MIN_K = 26
NUMERIC_REPLAY_TOLERANCE = 0.100
OFFSET_REPLAY_TOLERANCE = 1
FLOAT_TIE_TOLERANCE = 1e-6
MIN_LOCAL_WINDOWS = 10
LOCAL_MARGIN_FRAMES = 5

FAMILIES = ("A", "B", "C")
CELL_SPECS = (
    ("A", "G_N", "G", "N", "official_pipeline"),
    ("A", "G_S", "G", "S", "official_pipeline"),
    ("B", "R_N0", "R", "N0", "fixed_forward"),
    ("B", "R_S0", "R", "S0", "fixed_forward"),
    ("B", "Rs_N0", "Rs", "N0", "fixed_forward"),
    ("B", "Rs_S0", "Rs", "S0", "fixed_forward"),
    ("C", "Gc_Nc", "Gc", "Nc", "fixed_forward"),
    ("C", "Gc_Sc", "Gc", "Sc", "fixed_forward"),
)

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_WORKER = REPO / "scripts/experiments/lrs3_real_video_local_timing/syncnet_worker.py"
SYNCNET_PIPELINE = SYNCNET_ROOT / "run_pipeline.py"
SYNCNET_SCORE = SYNCNET_ROOT / "run_syncnet.py"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

# Hash the experiment implementation itself in inputs.json.  The historical
# artifacts and SyncNet runtime are bound separately; this list makes the
# diagnostic code that produced a run auditable as well.
EXPERIMENT_CODE_FILES = tuple(
    Path(__file__).with_name(name)
    for name in (
        "__init__.py",
        "common.py",
        "config.py",
        "audio.py",
        "protocol.py",
        "audit.py",
        "media.py",
        "prepare.py",
        "scoring.py",
        "analysis.py",
        "review.py",
        "validate.py",
        "runner.py",
    )
)


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
    def inputs(self) -> Path:
        return self.root / "inputs.json"

    @property
    def historical_audit(self) -> Path:
        return self.root / "historical_audit.json"

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def support(self) -> Path:
        return self.root / "support.json"

    @property
    def scores_csv(self) -> Path:
        return self.root / "scores.csv"

    @property
    def playback(self) -> Path:
        return self.root / "playback"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def final(self) -> Path:
        return self.root / "final.json"


def cell_id(sample_id: str, family: str, cell: str) -> str:
    return f"{sample_id}__{family}_{cell}"
