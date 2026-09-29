from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
PARENT_PROTOCOL = PARENT / "protocol.json"
PARENT_SCORE_MANIFEST = PARENT / "control_scores/manifest.json"
PARENT_CONTROL_ANALYSIS = PARENT / "control_analysis.json"
PARENT_CONTROL_VALIDATION = PARENT / "control_validation.json"
PARENT_FINAL = PARENT / "final.json"

CHANGE_ROOT = REPO / "openspec/changes/diagnose-wav2lip-delay-search-support"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-delay-search-support/spec.md"

PARENT_HASHES = {
    "protocol.json": "e0ad38110883bc7079e66866b5187f68636ee2ec8882ddba94879b203a5a2485",
    "control_scores/manifest.json": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2",
    "control_analysis.json": "068bf6ec3d43e432032ad6bfde8ce7cc284669b0c020575eb4ccf77bb30f9041",
    "control_validation.json": "9ae69db31e1f4b636a0b09e2793a86b373b58f183d8589f8a10f60f11d9138c6",
    "final.json": "a9a423d96cddd774a3b4ac570f1d1cdceddd6e6eb700dd0b668c652faeb473aa",
}

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
DELAY_FRAMES = 5
DELAY_SAMPLES = DELAY_FRAMES * SAMPLES_PER_FRAME
FRAME_COUNT = 93
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
EMBEDDING_ROWS = 88
EMBEDDING_DIM = 1024
MATRIX_COLUMNS = 31
VSHIFT = 15
U_ROWS = tuple(range(30, 58))
NATURAL_LAG_START = -15
DELAY_LAG_START = -10
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_DRAWS = 10_000
MATRIX_TOLERANCE = 1e-4
METRIC_TOLERANCE = 1e-6
OFFSET_LOW = -6
OFFSET_HIGH = -4
EXPECTED_RECORD_COUNT = 16
EXPECTED_GROUP_COUNT = 8
EXPECTED_CELL_COUNT = 32

EXPECTED_ANOMALIES = {
    "lrs3_7JVTirBEfho_00039": {"k_N": 27, "expected_column": 32, "old_offset_difference": 9},
    "lrs3_7PwvGfs6Pok_00003": {"k_N": 27, "expected_column": 32, "old_offset_difference": 1},
    "lrs3_7c5t6FkvUG0_00001": {"k_N": 28, "expected_column": 33, "old_offset_difference": -2},
}


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_delay_search_support_{validate_run_id(run_id)}"


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
    def matrix_manifest(self) -> Path:
        return self.root / "matrices/manifest.json"

    @property
    def per_record(self) -> Path:
        return self.root / "per_record.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def bootstrap_indices(self) -> Path:
        return self.root / "bootstrap_indices.npy"


def config_payload() -> dict[str, Any]:
    return {
        "sample_rate": SAMPLE_RATE,
        "fps": FPS,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "delay_frames": DELAY_FRAMES,
        "delay_samples": DELAY_SAMPLES,
        "frame_count": FRAME_COUNT,
        "frame_shape": [FRAME_HEIGHT, FRAME_WIDTH, 3],
        "embedding_shape": [EMBEDDING_ROWS, EMBEDDING_DIM],
        "legacy_lags": list(range(-15, 16)),
        "natural_lags": list(range(NATURAL_LAG_START, NATURAL_LAG_START + MATRIX_COLUMNS)),
        "delay_matched_lags": list(range(DELAY_LAG_START, DELAY_LAG_START + MATRIX_COLUMNS)),
        "u_rows": list(U_ROWS),
        "distance_formula": "sqrt(sum((float32(V-A)+float32(1e-6))**2, axis=1, dtype=float32))",
        "matrix_tolerance": MATRIX_TOLERANCE,
        "metric_tolerance": METRIC_TOLERANCE,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "rng": "numpy.default_rng/PCG64",
            "quantile": "linear",
            "unit": "source_group",
        },
        "budget": {
            "cached_cells": EXPECTED_CELL_COUNT,
            "derived_u_matrices": EXPECTED_CELL_COUNT,
            "new_videos": 0,
            "new_forward": 0,
            "training": 0,
            "gpu_calls": 0,
        },
        "fixed_flags": {
            "stage_b_authorized": False,
            "replacement_confirmed": False,
            "historical_shift_gate_repaired": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
        },
    }
