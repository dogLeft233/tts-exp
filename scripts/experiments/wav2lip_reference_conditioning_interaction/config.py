from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/wav2lip_natural_content_residual_20260908_v2"
Q = REPO / "runs/wav2lip_natural_content_residual_continuation_20260909_v1"
DRIVERS = PARENT / "drivers/manifest.json"
CONTROL_SCORES = PARENT / "control_scores/manifest.json"
Q_CANDIDATES = Q / "candidate_scores/manifest.json"
OLD_RUN = REPO / "runs/wav2lip_reference_conditioning_interaction_20260909_v1"
OLD_CONTROL_SCORES = OLD_RUN / "control_scores/manifest.json"
OLD_PROTOCOL = OLD_RUN / "protocol.json"
OLD_DRIVERS = OLD_RUN / "drivers/manifest.json"
OLD_REFERENCE = OLD_RUN / "reference_manifest.json"
INPUT_SNAPSHOT = REPO / "openspec/next-experiments-20260910-inputs.json"
HELPER_PATHS = (
    REPO / "scripts/experiments/lrs3_real_video_local_timing/media.py",
    REPO / "scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py",
    REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py",
    REPO / "scripts/experiments/wav2lip_roi_peak_recheck/worker.py",
)
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FIXED_HASHES = {
    "drivers": "4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf",
    "control_scores": "143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2",
    "q_candidates": "1ee531b15b78c21ae29c1482f12511ad403fa3fbdc9d8b9d86cd79dfd869c4de",
    "checkpoint": "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8",
    "syncnet": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
    "old_control_scores": "9e5a724d3634054d4328c0d35816f06db0065ccac41368e22bdb71291adc6264",
    "old_protocol": "51231bd182b8555db3c0b81a57a1c5a8cf495f0a186b3584b9f9f8bf99abe6cc",
    "old_drivers": "f8d8c4184a76848e553558f811db1935a9efbc18d4181c78efe1fae70209a129",
    "old_reference": "bbd7f121e8b383c308f20f1eb5300c760ae2214a154ff81ec6e1dda654a2c89e",
}
SAMPLE_RATE = 16_000
FPS = 25
FRAME_COUNT = 93
MEL_SHAPE = (80, 308)
MATRIX_COLUMNS = 31
EMBEDDING_ROWS = 88
U_ROWS = tuple(range(30, 58))
NATURAL_LAG_START = -15
DELAY_LAG_START = -10
NATURAL_OFFSET_RANGE = (-15, 15)
DELAY_OFFSET_RANGE = (-10, 20)
WAV2LIP_BATCH_SIZE = 4
SYNCNET_BATCH_SIZE = 20
SYNCNET_THREADS = 2
GENERATION_SEED = 20260909
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 20_000
RECORD_COUNT = 16
GROUP_COUNT = 8
GPU_LOCK = Path("/tmp/tts-exp-wav2lip-gpu0.lock")


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("invalid run-id")
    return value


def run_root_for(run_id: str) -> Path:
    return (
        REPO
        / "runs"
        / f"wav2lip_reference_conditioning_interaction_{validate_run_id(run_id)}"
    )


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
    def reference(self) -> Path:
        return self.p("reference_manifest.json")

    @property
    def control_analysis(self) -> Path:
        return self.p("control_analysis.json")

    @property
    def historical_control(self) -> Path:
        return self.p("historical_control.json")

    @property
    def control_scores(self) -> Path:
        return self.p("control_scores/manifest.json")

    @property
    def candidate_scores(self) -> Path:
        return self.p("candidate_scores/manifest.json")

    @property
    def analysis(self) -> Path:
        return self.p("analysis.json")

    @property
    def validation(self) -> Path:
        return self.p("validation.json")

    @property
    def control_validation(self) -> Path:
        return self.p("control_validation.json")

    @property
    def review(self) -> Path:
        return self.p("review.json")

    @property
    def final(self) -> Path:
        return self.p("final.json")

    @property
    def result(self) -> Path:
        return self.p("result.md")


def configuration() -> dict:
    return {
        "protocol_id": "wav2lip_reference_conditioning_interaction",
        "frame_index": 46,
        "fps": FPS,
        "u_rows": list(U_ROWS),
        "lag_domains": {
            "natural": {
                "q": "r+j-15",
                "offset": "15-j",
                "range": list(NATURAL_OFFSET_RANGE),
            },
            "delayed_matched": {
                "q": "r+j-10",
                "offset": "10-j",
                "range": list(DELAY_OFFSET_RANGE),
            },
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "unit": "source_group",
            "quantile": "linear",
        },
        "budget": {
            "fresh_videos": 36,
            "fresh_scores": 54,
            "training": 0,
            "tts": 0,
            "vocoder": 0,
        },
        "fixed_flags": {
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    }
