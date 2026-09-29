from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO: Final[Path] = Path(__file__).resolve().parents[3]
CHANGE_ROOT: Final[Path] = REPO / "openspec/changes/reconcile-wav2lip-replacement-endpoints"
PROPOSAL: Final[Path] = CHANGE_ROOT / "proposal.md"
DESIGN: Final[Path] = CHANGE_ROOT / "design.md"
SPEC: Final[Path] = CHANGE_ROOT / "specs/wav2lip-replacement-endpoint-reconciliation/spec.md"
TASKS: Final[Path] = CHANGE_ROOT / "tasks.md"

H_ROOT: Final[Path] = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
S_ROOT: Final[Path] = REPO / "runs/wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3"
H_ROOT_ID: Final[str] = "lrs3_natural_to_tts_bridge_confirmation_20260904"
S_ROOT_ID: Final[str] = "wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3"

ROOT_INPUTS: Final[dict[str, tuple[Path, str]]] = {
    "H/cohort": (H_ROOT / "00_protocol/cohort.json", "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"),
    "H/audio_manifest": (H_ROOT / "01_audio/audio_manifest.json", "2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288"),
    "H/videos_manifest": (H_ROOT / "02_videos/videos_manifest.json", "680053f3ca851cea6a2572654a228671bbe616f369b00d84a552b22b305543c2"),
    "H/scores_manifest": (H_ROOT / "03_scores/scores_manifest.json", "0022c07d68f859a0189833c58fd7f8daf639787d8d4ffc99e48e6ccaa1a99e5b"),
    "H/analysis": (H_ROOT / "04_final/analysis.json", "0841a0e95efb8747516f866427bafa601c1fa5270ca4bcaaf6cfd396fc77556f"),
    "H/final": (H_ROOT / "04_final/final.json", "df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e"),
    "S/protocol": (S_ROOT / "protocol.json", "284dea1b07fbccc07a309512b164265d215075af41b8009b38d163362d502cfa"),
    "S/audio_manifest": (S_ROOT / "audio/manifest.json", "2086e55df0e116e60656577f0ce255aad32c87a52ae02e279c2d8e0b96206ae3"),
    "S/videos_manifest": (S_ROOT / "videos/manifest.json", "92b091c6bc77a1ad0487af3d701173d7fd969c62db940f3c930c02e69981aeba"),
    "S/scores_manifest": (S_ROOT / "scores/manifest.json", "a35c09c0a1d5ed2ae9f6ea6a363fa1e0a6e190ff3d69dcb380b186d2ecec1270"),
    "S/analysis": (S_ROOT / "analysis.json", "957c8cb83d59f9fc5bb4ec3ec7c321449fa87a6a582c8d8fe4e0b9f7769e2316"),
    "S/validation": (S_ROOT / "validation.json", "82a85878f9dbe50d7a406f1ab32457b30b2f4e94cd1953e7836e32ee895e0bd0"),
    "S/final": (S_ROOT / "final.json", "60454d9d5a70494db42f8f5b8bf5e0106db2cc92aee70dbe6c4f0057653ff701"),
}

PROTOCOL_ID: Final[str] = "wav2lip_replacement_endpoint_reconciliation_20260908"
MATRIX_COLUMNS: Final[int] = 31
VSHIFT: Final[int] = 15
EXPECTED_RECORD_COUNT: Final[int] = 22
EXPECTED_SOURCE_GROUP_COUNT: Final[int] = 22
EXPECTED_H_MATRIX_COUNT: Final[int] = 66
EXPECTED_S_MATRIX_COUNT: Final[int] = 110
EXPECTED_MATRIX_COUNT: Final[int] = 176
EXPECTED_ENDPOINT_COUNT: Final[int] = 528
BOOTSTRAP_SEED: Final[int] = 20260908
BOOTSTRAP_DRAWS: Final[int] = 10_000
HISTORICAL_BOOTSTRAP_SEED: Final[int] = 20260904
HISTORICAL_BOOTSTRAP_DRAWS: Final[int] = 10_000
SYNCNET_MODEL_SHA256: Final[str] = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
ORDERED_SAMPLE_ID_SHA256: Final[str] = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

H_VIDEO_ARMS: Final[tuple[str, ...]] = ("N", "N_REPEAT", "BRIDGE_075")
S_VIDEO_ARMS: Final[tuple[str, ...]] = ("N", "N_REPEAT", "RT", "MAG", "ENV")
ENDPOINTS: Final[tuple[str, ...]] = ("FULL", "COMMON_INTERIOR", "U")
NO_SEALED_MEDIA_TOKENS: Final[tuple[str, ...]] = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")
CODE_FILES: Final[tuple[Path, ...]] = (
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/__init__.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/config.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/common.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/protocol.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/analysis.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/runner.py",
    REPO / "scripts/experiments/wav2lip_replacement_reconciliation/validate.py",
)
SPEC_FILES: Final[tuple[Path, ...]] = (PROPOSAL, DESIGN, SPEC)


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_replacement_reconciliation_{validate_run_id(run_id)}"


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
    def parity(self) -> Path:
        return self.root / "parity.json"

    @property
    def discrepancy(self) -> Path:
        return self.root / "discrepancy.json"

    @property
    def endpoints(self) -> Path:
        return self.root / "endpoints.json"

    @property
    def bootstrap_indices(self) -> Path:
        return self.root / "bootstrap_indices.npy"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"


def frozen_config() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "matrix_columns": MATRIX_COLUMNS,
        "vshift": VSHIFT,
        "record_count": EXPECTED_RECORD_COUNT,
        "source_group_count": EXPECTED_SOURCE_GROUP_COUNT,
        "h_matrix_count": EXPECTED_H_MATRIX_COUNT,
        "s_matrix_count": EXPECTED_S_MATRIX_COUNT,
        "matrix_count": EXPECTED_MATRIX_COUNT,
        "endpoint_count": EXPECTED_ENDPOINT_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "h_video_arms": list(H_VIDEO_ARMS),
        "s_video_arms": list(S_VIDEO_ARMS),
        "endpoints": list(ENDPOINTS),
        "support_rule": "I=range(15,min(T_8cells)-15); U=plus_rows+minus_rows in protocol order",
        "aggregation": "mean distance curve first; D=min(z); C=median(z)-min(z); offset=15-argmin(z)",
        "scope": "retrospective_cache_only_diagnostic",
        "new_media_count": 0,
        "model_forward_count": 0,
        "training_authorized": False,
        "generalization_established": False,
        "historical_gate_repaired": False,
    }
