from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
A = REPO / "runs/wav2lip_natural_temporal_contrast_20260909_v1"
B = REPO / "runs/wav2lip_reference_conditioning_interaction_20260909_v1"
C = REPO / "runs/wav2lip_residual_local_response_20260909_v1"
P = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
Q = REPO / "runs/wav2lip_natural_content_residual_continuation_20260909_v1"
D = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
FIXED_FILES = {
    "A/protocol.json": "516b3c52c1267e587540395cbf122e867fb1b02732369df50988bb985612a69a",
    "A/drivers/manifest.json": "9af82f121940613441a59b5c13d0e87e3ac01200f93bfd4eb5274c2267f73949",
    "A/control_analysis.json": "8607989e557a1a7ff5a8e2c32f8d0b20113f1743130f5d26896ccfbdefdbd7ef",
    "A/candidate_scores/manifest.json": "c92b60b67da5a4f301fb8ba1e9b09b462d67153bd66d198387aacabc2960ff6e",
    "A/analysis.json": "c63ca572b3fdcaf93e4a63f63da66e6aca7efd7968973c6fd179540c6cf448ff",
    "B/protocol.json": "51231bd182b8555db3c0b81a57a1c5a8cf495f0a186b3584b9f9f8bf99abe6cc",
    "B/reference_manifest.json": "bbd7f121e8b383c308f20f1eb5300c760ae2214a154ff81ec6e1dda654a2c89e",
    "B/control_scores/manifest.json": "9e5a724d3634054d4328c0d35816f06db0065ccac41368e22bdb71291adc6264",
    "B/control_analysis.json": "dc79b2f53bdc5a31395e596b2ddb26824e592fb096ca2de0806d8d577b51cda0",
    "B/final.json": "5cc7bdaee6d7e21142bf949aef0250a4a3be411ebe19e4890f5d770e6b696f28",
    "C/protocol.json": "3a20ff1f4a7a5693212655332b064a586830d4872224c6e6a9a115bc9d8b4870",
    "C/exposure.json": "666e19efec2daa811088f4e72a35e7872d8fe124006f9b9b58dcb4a1c13f4e3b",
    "C/reused_manifest.json": "99d8867d41f46c062ff529d1cb42993af2cfe297f07ad5dabaf977963bac8110",
    "C/analysis.json": "283f0c2605ab291a86ec6eae79b152fef33e36ac8f08299634e9f02b3f25b076",
}
P_FILES = {
    "drivers": P / "drivers/manifest.json",
    "control_scores": P / "control_scores/manifest.json",
}
Q_FILES = {"candidate_scores": Q / "candidate_scores/manifest.json"}
D_FILES = {"drivers": D / "05_drivers/drivers.json", "mask_manifest": D / "03_data/mask_manifest.json"}
P_HASHES = {"drivers": "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf", "control_scores": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2"}
Q_HASHES = {"candidate_scores": "1ee531b15b78c21ae29c1482f12511ad403fa3fbdc9d8b9d86cd79dfd869c4de"}
D_HASHES = {"drivers": "ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410", "mask_manifest": "956d7eb5bd7fe6b12a658191ac0ab80a9a8a6af77a0fce2aebcec30be561deec"}
U_ROWS = tuple(range(30, 58))
MEL_FRAMES = 308
FPS = 25
MATRIX_COLUMNS = 31
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 20_000


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None: raise ValueError("invalid run-id")
    return value


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_parallel_evidence_contract_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    def p(self, name: str) -> Path: return self.root / name
    @property
    def protocol(self) -> Path: return self.p("protocol.json")
    @property
    def input_audit(self) -> Path: return self.p("input_audit.json")
    @property
    def per_record(self) -> Path: return self.p("per_record.json")
    @property
    def recomputed(self) -> Path: return self.p("recomputed_analysis.json")
    @property
    def discrepancies(self) -> Path: return self.p("discrepancies.json")
    @property
    def validation(self) -> Path: return self.p("validation.json")
    @property
    def review(self) -> Path: return self.p("review.json")
    @property
    def final(self) -> Path: return self.p("final.json")
    @property
    def result(self) -> Path: return self.p("result.md")
