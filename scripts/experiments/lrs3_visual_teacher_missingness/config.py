from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARENT = REPO / "runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4"
# H is the completed missingness audit.  It is the only source for this
# retrospective diagnostic; its media and feature files remain read-only.
H = REPO / "runs/lrs3_visual_teacher_missingness_20260909_v1"
LOCK = REPO / "runs/lrs3_tts_visual_control_20260825/00_protocol_lock_retry2"
REVIEW = (
    REPO
    / "_reviews/lrs3_tts_visual_control/01_visual_teacher_audit/artifact_review_retry4.json"
)
PARENT_HASHES = {
    "manifest.json": "139ab35e89f837de6f32b4f4934c49d94577b0216a622b416cc6523bda6d23be",
    "summary.json": "0d080bfb7bc49467d19296aa674ebc2e79a9541ff86c896dcb6acc43c5fef8ff",
    "decision.json": "3e82b583937ee655b714f1e19ebca1327d8828c70dfcd229df5129583fd07ba9",
}
LOCK_HASHES = {
    "manifest.json": "e1c970542eb90a2dac787463ebe4e8708ad3f2c4b4f1c4379f22e8382a29ec38",
    "test_lock.json": "295ccba71fb50d873a3ddf1b5cfd895418fd5bd95d242ec6441c57ef7e9660d4",
}
REVIEW_HASH = "8525d2959888bd72ed296ebbce107254a7e1da58c5dbc7b2cc37899e2a489349"
TREE_HASH = "da8faef0c9432d81dee9556f2c7435230e75b6a01aa0aa0192ab0100ddeba57c"
H_HASHES = {
    "protocol.json": "522c8f565c7e0cc6e18543b474506501e6dd66121c4e4a74c9bb852f85a7e4cc",
    "input_audit.json": "20b90cfc7da3dc937c943d1fd1aaba31b5434c35ccd1d7dabc829140d3ed8ae0",
    "records.json": "ab2314835b46a9234965dfad1d77b81308ab9f21c9013476aced7974ab413f9d",
    "analysis.json": "830233ae8e5fa8dc3a58fc7a18d966d50b2ad5cf1a726d353d3335fa1637c638",
    "support_indices.npz": "c1bda6c377d43c16385df32b8249922a1e1247fc3b302f44d1501ac9fd4d68a1",
}
RECORD_COUNT = 133
GROUP_COUNT = 23
MINIMUM_ELIGIBLE = 122
VALID_FRACTION_MIN = 0.90
COVERAGE_MIN = 0.85


def validate_run_id(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("invalid run-id")
    return value


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"lrs3_visual_teacher_missingness_{validate_run_id(run_id)}"


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
    def records(self) -> Path:
        return self.p("records.json")

    @property
    def support_indices(self) -> Path:
        return self.p("support_indices.npz")

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
