from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
D = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
P_DRIVERS = P / "drivers/manifest.json"
P_CONTROL = P / "control_scores/manifest.json"
D_DRIVERS = D / "05_drivers/drivers.json"
D_MASKS = D / "03_data/mask_manifest.json"
P_DRIVERS_SHA256 = "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf"
P_CONTROL_SHA256 = "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2"
D_DRIVERS_SHA256 = "ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410"
D_MASKS_SHA256 = "956d7eb5bd7fe6b12a658191ac0ab80a9a8a6af77a0fce2aebcec30be561deec"
OLD_RUN = REPO / "runs/wav2lip_phone_core_shrinkage_20260910_v1"
OLD_PROTOCOL = OLD_RUN / "protocol.json"
OLD_INPUT_AUDIT = OLD_RUN / "input_audit.json"
OLD_DRIVERS = OLD_RUN / "drivers/manifest.json"
OLD_PROTOCOL_SHA256 = "dfa69b9c1310d68fdadd0a04ab4e597ae6cbeca42cfa3b010e4b45e43fe2a67a"
OLD_INPUT_AUDIT_SHA256 = (
    "219a433a3da9b95f5738cb65804fc3528b593e8eb11e444e8e6cc786cf0e9413"
)
OLD_DRIVERS_SHA256 = "a7e7ed3401a3f48d33026e9ca97c6d1f91939905d184844bc73343196932c982"
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = (
    "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
)
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = (
    "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
)
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
PROPOSAL = (
    REPO / "openspec/changes/complete-wav2lip-phone-core-group-support/proposal.md"
)
DESIGN = REPO / "openspec/changes/complete-wav2lip-phone-core-group-support/design.md"
SPEC = (
    REPO
    / "openspec/changes/complete-wav2lip-phone-core-group-support/specs/wav2lip-phone-core-group-support/spec.md"
)
HELPER_PATHS = (
    REPO / "scripts/experiments/wav2lip_probe_runtime.py",
    REPO / "scripts/experiments/wav2lip_phone_core_shrinkage/validator_math.py",
    REPO / "scripts/experiments/wav2lip_probe_gpu.py",
    REPO / "scripts/experiments/wav2lip_probe_score.py",
    REPO / "scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py",
    REPO / "scripts/experiments/wav2lip_roi_peak_recheck/worker.py",
)
EXCLUDED_LABELS = {"sil", "sp", "spn", "<eps>"}
U_ROWS = tuple(range(30, 58))
ARMS = ("N", "PHONE_CORE", "GENERIC_CORE")


def run_root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid run id")
    return REPO / "runs" / f"wav2lip_phone_core_shrinkage_{run_id}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def p(self, name: str) -> Path:
        return self.root / name

    @property
    def protocol(self) -> Path:
        return self.p("protocol.json")

    @property
    def input_audit(self) -> Path:
        return self.p("input_audit.json")

    @property
    def drivers(self) -> Path:
        return self.p("drivers/manifest.json")

    @property
    def controls(self) -> Path:
        return self.p("control_analysis.json")

    @property
    def control_validation(self) -> Path:
        return self.p("control_validation.json")

    @property
    def plan(self) -> Path:
        return self.p("plans/candidates.json")

    @property
    def generation(self) -> Path:
        return self.p("generation/candidates.json")

    @property
    def scores(self) -> Path:
        return self.p("scores/candidates/manifest.json")

    @property
    def control_scores(self) -> Path:
        return self.p("scores/controls/manifest.json")

    @property
    def analysis(self) -> Path:
        return self.p("analysis.json")

    @property
    def validation(self) -> Path:
        return self.p("validation.json")

    @property
    def review(self) -> Path:
        return self.p("review.json")

    @property
    def final(self) -> Path:
        return self.p("final.json")

    @property
    def result(self) -> Path:
        return self.p("result.md")
