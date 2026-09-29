from __future__ import annotations

from scripts.experiments.mfa_linear_sync_existing_data.config import (
    NO_REAL_VIDEO_TRANSFER,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
)
from scripts.experiments.mfa_linear_sync_existing_data.real_video import decide_real_video
from scripts.experiments.mfa_linear_sync_existing_data.replacement import decide_replacement


def _real_rows(success_count: int, *, invalid: bool = False) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"row-{index}",
            "engineering_valid": not invalid or index != 0,
            "scientific_success": index < success_count,
            "d_gain": 0.004,
            "c_gain": 0.004,
        }
        for index in range(8)
    ]


def _replacement_rows(success_count: int, *, invalid: bool = False) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"row-{index}",
            "engineering_valid": not invalid or index != 0,
            "scientific_success": index < success_count,
            "replacement_d_gain": 0.004,
            "replacement_c_gain": 0.004,
        }
        for index in range(8)
    ]


def test_real_video_uses_six_of_eight_gate() -> None:
    assert decide_real_video(_real_rows(6))["status"] == REAL_VIDEO_TRANSFER
    assert decide_real_video(_real_rows(5))["status"] == NO_REAL_VIDEO_TRANSFER
    invalid = decide_real_video(_real_rows(8, invalid=True))
    assert invalid["status"] == REAL_VIDEO_NOT_EVALUATED


def test_replacement_requires_all_72_cells() -> None:
    passed = decide_replacement(_replacement_rows(6), cell_count=72)
    assert passed["status"] == REPLACEMENT_OBSERVED
    incomplete = decide_replacement(_replacement_rows(8), cell_count=71)
    assert incomplete["status"] == REPLACEMENT_NOT_EVALUATED
    invalid = decide_replacement(_replacement_rows(8, invalid=True), cell_count=72)
    assert invalid["status"] == REPLACEMENT_NOT_EVALUATED
