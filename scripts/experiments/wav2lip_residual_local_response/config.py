from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
Q = REPO / "runs/wav2lip_natural_content_residual_continuation_20260909_v1"
D = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
DRIVERS = PARENT / "drivers/manifest.json"
PARENT_SCORES = PARENT / "control_scores/manifest.json"
Q_SCORES = Q / "candidate_scores/manifest.json"
Q_ANALYSIS = Q / "analysis.json"
D_DRIVERS = D / "05_drivers/drivers.json"
FIXED_HASHES = {"drivers": "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf", "parent_scores": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2", "q_scores": "1ee531b15b78c21ae29c1482f12511ad403fa3fbdc9d8b9d86cd79dfd869c4de", "q_analysis": "83a3c27798a31b8a8d2ee1350dc9850fe8a5710a421cf51bae4c18e88be084cc", "d_drivers": "ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410"}
U_ROWS = tuple(range(30, 58))
MATRIX_ROWS = 88
MATRIX_COLUMNS = 31
EMBEDDING_DIM = 1024
VSHIFT = 15
FPS = 25
MEL_FRAMES = 308
FRAME_COUNT = 93
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 20_000


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None: raise ValueError("invalid run-id")
    return value


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_residual_local_response_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    def p(self, name: str) -> Path: return self.root / name
    @property
    def protocol(self) -> Path: return self.p("protocol.json")
    @property
    def input_audit(self) -> Path: return self.p("input_audit.json")
    @property
    def reused_manifest(self) -> Path: return self.p("reused_manifest.json")
    @property
    def support(self) -> Path: return self.p("support_indices.npz")
    @property
    def exposure(self) -> Path: return self.p("exposure.json")
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
    return {"protocol_id": "wav2lip_residual_local_response", "u_rows": list(U_ROWS), "fps": FPS, "chunk_rule": "official int(t*80/25), final last 16", "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "source_group", "quantile": "linear"}, "budget": {"fresh_videos": 0, "fresh_scores": 0, "models": 0, "gpu": 0}, "fixed_flags": {"replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
