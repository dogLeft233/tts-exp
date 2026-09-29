from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/explain-lrs3-tts-syncnet-gain"
INPUT_BINDINGS = CHANGE_ROOT / "input-bindings.json"
SPEC = CHANGE_ROOT / "specs/lrs3-tts-gain-mechanism/spec.md"
DESIGN = CHANGE_ROOT / "design.md"

MANIFEST = REPO / "data/dataset_samples/video_manifest_250.json"
DITTO_SCORES = REPO / "runs/multiset_pipeline/04_eval/scores_250.json"
DITTO_META = REPO / "runs/multiset_pipeline/04_eval/eval_meta.json"
LEAPTALK_SCORES = REPO / "runs/leaptalk_eval/04_eval/scores_leaptalk.json"
DITTO_CELL_ROOT = REPO / "runs/multiset_pipeline/04_eval"
LEAPTALK_CELL_ROOT = REPO / "runs/leaptalk_eval/04_eval"
LEAPTALK_MEDIA_ROOT = REPO / "runs/leaptalk_multiset"

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
GPU_LOCK = Path("/tmp/tts-exp-lrs3-tts-gain-mechanism-gpu.lock")

PROTOCOL_ID = "lrs3_tts_syncnet_gain_mechanism_v1"
SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
VSHIFT = 15
LAG_COUNT = 31
MIN_TRACK = 25
BATCH_SIZE = 20
CURVE_SAMPLE_IDS = tuple(range(151, 163))
HISTORICAL_SAMPLE_IDS = tuple(range(151, 201))
CONDITIONS = ("natural_raw", "tts_raw")
MODELS = ("Ditto", "LeapTalk")
MAX_NEW_CELLS = 28
MAIN_CURVE_CELLS = 24
REPEAT_CELLS = 2
DELAY_CELLS = 2
BOOTSTRAP_SEED = 20_260_915
BOOTSTRAP_DRAWS = 10_000
RESERVE_BYTES = 1 << 30
MATRIX_TOLERANCE = 1e-4
SAVED_SCORE_TOLERANCE = 0.000501

OFFICIAL_PIPELINE_CONFIG = {
    "facedet_scale": 0.25,
    "crop_scale": 0.40,
    "min_track": MIN_TRACK,
    "frame_rate": FPS,
    "num_failed_det": 25,
    "min_face_size": 100,
}


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"lrs3_tts_gain_mechanism_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def audit(self) -> Path:
        return self.root / "00_audit"

    @property
    def decomposition(self) -> Path:
        return self.root / "01_decomposition"

    @property
    def curves(self) -> Path:
        return self.root / "02_curves"

    @property
    def analysis(self) -> Path:
        return self.root / "03_analysis"

    @property
    def perception(self) -> Path:
        return self.root / "04_perception"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"
