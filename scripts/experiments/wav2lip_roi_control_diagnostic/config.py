from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PARENT_ROOT = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix5"
CHANGE_ROOT = REPO / "openspec/changes/diagnose-wav2lip-roi-control-failure"
PARENT_HASHES = {
    "final": "76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563",
    "validation": "328e562cc8dacfb960fb2319fb15f1cfa90eecf593f04434382c9a5e13bd2577",
    "control": "6944cc299f30f29a2c6f6a4f41db024014ec09aa583ca2bbc15c12133c8abce5",
    "protocol": "835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4",
    "score_manifest": "60950f9a26abcb37d9113541e23acd038feb8f21b827f07fb3c2f07570aa3799",
}

PROTOCOL_ID = "wav2lip_roi_control_failure_diagnostic"
PROTOCOL_REVISION = "independent_recompute_v1"
SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
VSHIFT = 15
WINDOW_FRAMES = 5
WARP_AMPLITUDE_SAMPLES = 1_920
MIN_LOCAL_ROWS = 5
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

VIDEO_ARMS = ("R", "G_N", "G_NR", "G_W")
MAIN_CELL_SPECS = (
    ("R", "N", False),
    ("R", "W", False),
    ("G_N", "N", False),
    ("G_N", "W", False),
    ("G_W", "N", False),
    ("G_W", "W", False),
    ("G_NR", "N", False),
)
REPEAT_CELL_SPECS = (("R", "N", True), ("G_N", "N", True))
ALL_CELL_SPECS = MAIN_CELL_SPECS + REPEAT_CELL_SPECS
EXPECTED_MAIN_CELL_COUNT = EXPECTED_RECORD_COUNT * len(MAIN_CELL_SPECS)
EXPECTED_REPEAT_CELL_COUNT = EXPECTED_RECORD_COUNT * len(REPEAT_CELL_SPECS)
EXPECTED_CELL_COUNT = EXPECTED_MAIN_CELL_COUNT + EXPECTED_REPEAT_CELL_COUNT

SPEC_PATHS = {
    "proposal": CHANGE_ROOT / "proposal.md",
    "design": CHANGE_ROOT / "design.md",
    "spec": CHANGE_ROOT / "specs/wav2lip-roi-control-failure-diagnostic/spec.md",
}


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    validate_run_id(run_id)
    return REPO / "runs" / f"wav2lip_roi_control_diagnostic_{run_id}"


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
    def diagnostics(self) -> Path:
        return self.root / "diagnostics.json"

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
    warp_amplitude_samples: int = WARP_AMPLITUDE_SAMPLES
    min_local_rows: int = MIN_LOCAL_ROWS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_main_cell_count: int = EXPECTED_MAIN_CELL_COUNT
    expected_repeat_cell_count: int = EXPECTED_REPEAT_CELL_COUNT
    expected_cell_count: int = EXPECTED_CELL_COUNT
    ordered_sample_id_sha256: str = EXPECTED_SAMPLE_ID_SHA256

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "video_arms": list(VIDEO_ARMS),
                "main_cell_specs": [list(item) for item in MAIN_CELL_SPECS],
                "repeat_cell_specs": [list(item) for item in REPEAT_CELL_SPECS],
                "warp_formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced; np.interp inverse",
                "offset_column_formula": "offset=15-column_index",
                "bootstrap_policy": "sorted source groups; NumPy default_rng/PCG64; 10000 draws; percentile 2.5/97.5; seed reset per metric",
                "forbidden_operations": [
                    "video_generation",
                    "score_generation",
                    "audio_modification",
                    "record_filtering",
                    "training",
                    "bridge_execution",
                ],
            }
        )
        return value
