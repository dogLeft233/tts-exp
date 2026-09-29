from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
P = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
P_DRIVERS = P / "drivers/manifest.json"
P_CONTROL = P / "control_scores/manifest.json"
P_DRIVERS_SHA256 = "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf"
P_CONTROL_SHA256 = "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2"
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
PROPOSAL = REPO / "openspec/changes/probe-wav2lip-waveform-gain/proposal.md"
DESIGN = REPO / "openspec/changes/probe-wav2lip-waveform-gain/design.md"
SPEC = REPO / "openspec/changes/probe-wav2lip-waveform-gain/specs/wav2lip-waveform-gain/spec.md"
SAMPLE_RATE = 16_000
SUPPORT_SAMPLES = 61_440
U_ROWS = tuple(range(30, 58))
ARMS = ("N", "GAIN_PLUS", "GAIN_MINUS")


def run_root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid run id")
    return REPO / "runs" / f"wav2lip_waveform_gain_{run_id}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def p(self, name: str) -> Path:
        return self.root / name

    @property
    def protocol(self) -> Path: return self.p("protocol.json")
    @property
    def input_audit(self) -> Path: return self.p("input_audit.json")
    @property
    def drivers(self) -> Path: return self.p("drivers/manifest.json")
    @property
    def audio_manifest(self) -> Path: return self.p("audio/manifest.json")
    @property
    def controls(self) -> Path: return self.p("control_analysis.json")
    @property
    def control_validation(self) -> Path: return self.p("control_validation.json")
    @property
    def plan(self) -> Path: return self.p("plans/candidates.json")
    @property
    def generation(self) -> Path: return self.p("generation/candidates.json")
    @property
    def scores(self) -> Path: return self.p("scores/candidates/manifest.json")
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
