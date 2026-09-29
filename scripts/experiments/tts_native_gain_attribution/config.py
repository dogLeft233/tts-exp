from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/disentangle-tts-native-gain"
INPUT_BINDINGS = CHANGE_ROOT / "input-bindings.json"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
PROTOCOL = CHANGE_ROOT / "protocol.md"
SPEC = CHANGE_ROOT / "specs/tts-native-gain-attribution/spec.md"
TASKS = CHANGE_ROOT / "tasks.md"

PARENT_RUN = REPO / "runs/lrs3_tts_gain_mechanism_review_v15"
PARENT_FINAL = PARENT_RUN / "final.json"
PARENT_VALIDATION = PARENT_RUN / "validation.json"
PARENT_COHORT = PARENT_RUN / "00_audit/cohort.json"
PARENT_INPUTS = PARENT_RUN / "00_audit/inputs.json"

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_INSTANCE = SYNCNET_ROOT / "SyncNetInstance.py"
SYNCNET_DEFINITION = SYNCNET_ROOT / "SyncNetModel.py"
SYNCNET_PIPELINE = SYNCNET_ROOT / "run_pipeline.py"
SYNCNET_PYTHON = Path(os.environ.get("SYNCNET_PYTHON", "/home/wjj/.venvs/syncnet/bin/python"))
FFMPEG = Path(os.environ.get("FFMPEG", "/home/wjj/miniconda3/bin/ffmpeg"))
FFPROBE = Path(os.environ.get("FFPROBE", "/home/wjj/miniconda3/bin/ffprobe"))
# This lock is deliberately shared by every tts-exp GPU experiment.  An
# experiment-specific lock would only serialize cooperating invocations of
# this one module and would still allow two repository experiments to collide.
GPU_LOCK = Path(os.environ.get("TTS_EXP_GPU_LOCK", "/tmp/tts-exp-gpu.lock"))

PROTOCOL_ID = "tts_native_gain_attribution_v1"
SAMPLE_IDS = tuple(range(151, 163))
SOURCES = ("N", "T", "R")
SOURCE_NAMES = {"N": "natural", "T": "tts", "R": "real"}
VIDEO_TYPES = ("V_N", "V_T", "R")
AUDIO_CONDITIONS = ("ORIGINAL", "A0", "GAIN", "NOISE", "DENOISE")
B_DRIVERS = ("A0", "NOISE", "DENOISE")
SEEDS = (42, 43)
CONTROL_IDS = (151, 152)
CONTROL_SHIFTS = ("IDENTITY", "PLUS_200MS", "MINUS_200MS")

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
NFFT = 512
HOP = 160
WINDOW = 512
PAD = 256
LAG_COUNT = 31
VSHIFT = 15
EMBEDDING_DIM = 1024
SYNCNET_BATCH_SIZE = 20
# The historical SyncNet worker invokes ffmpeg's image2 MJPEG encoder and
# then OpenCV-decodes the JPEGs before the lip forward.  Keep this explicit so
# cached features cannot silently mix that contract with direct container
# decoding.
SYNCNET_VISUAL_DECODE_MODE = "ffmpeg_image2pipe_mjpeg_v1"
MIN_INTERIOR_ROWS = 25
MIN_ACTIVE_SAMPLES = 16_000
NOISE_SNR_DB = 20.0
GAIN_DB = -6.0
RESERVE_BYTES = 1 << 30
PERSISTENT_BUDGET_BYTES = 512 << 20
CELL_TEMP_BUDGET_BYTES = 384 << 20
GPU_PEAK_BUDGET_BYTES = 2 << 30
DISK_RESERVE_BYTES = 1 << 30
# SyncNet reads one crop through a small frame ring and writes no decoder
# frames to disk; its measured transient disk footprint is much lower than a
# face-tracking pipeline's.  Keep this separate so the gate reflects the
# actual stage rather than an unnecessarily pessimistic shared default.
SYNCNET_CELL_TEMP_BUDGET_BYTES = 64 << 20
BOOTSTRAP_SEED = 20_260_915
BOOTSTRAP_DRAWS = 20_000
PRIMARY_FAMILY_SIZE = 6
PRACTICAL_THRESHOLD = 0.200
EPSILON = 1e-12

EXPECTED_A_SCIENCE = len(SAMPLE_IDS) * len(VIDEO_TYPES) * len(AUDIO_CONDITIONS)
EXPECTED_A_CONTROLS = len(CONTROL_IDS) * len(VIDEO_TYPES) * len(CONTROL_SHIFTS)
EXPECTED_WRONG_CONTENT = len(VIDEO_TYPES) * len(SAMPLE_IDS) * (len(SAMPLE_IDS) - 1)
EXPECTED_B_VIDEOS = len(SAMPLE_IDS) * 2 * len(B_DRIVERS) * len(SEEDS)
EXPECTED_B_REPEATS = len(CONTROL_IDS) * 2
EXPECTED_B_SCIENCE = len(SAMPLE_IDS) * 2 * len(SEEDS) * 7
EXPECTED_B_CONTROLS = len(CONTROL_IDS) * 2 * 2
EXPECTED_SYNC_PAIRS = len(SAMPLE_IDS) * 2 * 2 * 2
EXPECTED_QUALITY_PAIRS = len(SAMPLE_IDS) * 2 * 2

OFFICIAL_PIPELINE = {
    "facedet_scale": 0.25,
    "crop_scale": 0.40,
    "min_track": 25,
    "frame_rate": FPS,
    "num_failed_det": 25,
    "min_face_size": 100,
}


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"tts_native_gain_attribution_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def audit(self) -> Path:
        return self.root / "00_audit"

    @property
    def audio(self) -> Path:
        return self.root / "01_audio"

    @property
    def fixed_video(self) -> Path:
        return self.root / "02_fixed_video"

    @property
    def generation(self) -> Path:
        return self.root / "03_generation"

    @property
    def crossed(self) -> Path:
        return self.root / "04_crossed_scores"

    @property
    def analysis(self) -> Path:
        return self.root / "05_analysis"

    @property
    def perception(self) -> Path:
        return self.root / "06_perception"

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.json"

    @property
    def claims(self) -> Path:
        return self.root / "claim_registry.json"

    @property
    def provenance(self) -> Path:
        return self.root / "model_provenance.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def parent_evidence(self) -> Path:
        return self.root / "parent_evidence.json"

    @property
    def reuse_manifest(self) -> Path:
        return self.root / "reuse_manifest.json"

    @property
    def preflight(self) -> Path:
        return self.root / "preflight.json"


def package_files() -> tuple[Path, ...]:
    return tuple(sorted(Path(__file__).parent.glob("*.py")))


def runtime_files() -> tuple[Path, ...]:
    return (SYNCNET_INSTANCE, SYNCNET_DEFINITION, SYNCNET_PIPELINE, SYNCNET_MODEL, FFMPEG, FFPROBE)


def leaptalk_candidates() -> tuple[Path, ...]:
    values = []
    explicit = os.environ.get("LEAPTALK_ROOT")
    if explicit:
        values.append(Path(explicit))
    values.extend(
        (
            REPO / "third_party/LeapTalk",
            REPO.parent / "LeapTalk",
            Path("/home/wjj/LeapTalk"),
            Path("/home/wjj/tts-audio/LeapTalk"),
        )
    )
    return tuple(dict.fromkeys(value.resolve() for value in values))
