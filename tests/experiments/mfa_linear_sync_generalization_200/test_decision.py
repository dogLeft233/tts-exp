from __future__ import annotations

from scripts.experiments.mfa_linear_sync_generalization_200.config import (
    NO_REAL_VIDEO_TRANSFER,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
)
from scripts.experiments.mfa_linear_sync_generalization_200.real_video import decide_real_video
from scripts.experiments.mfa_linear_sync_generalization_200.replacement import decide_replacement


def _real_rows(success_count: int, *, engineering_invalid: bool = False) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"row-{index}",
            "engineering_valid": not engineering_invalid or index != 0,
            "scientific_success": index < success_count,
            "d_gain": 0.004,
            "c_gain": 0.004,
        }
        for index in range(40)
    ]


def _replacement_rows(success_count: int, *, engineering_invalid: bool = False) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"row-{index}",
            "engineering_valid": not engineering_invalid or index != 0,
            "scientific_success": index < success_count,
            "replacement_d_gain": 0.004,
            "replacement_c_gain": 0.004,
        }
        for index in range(40)
    ]


def test_real_video_requires_complete_30_of_40_gate() -> None:
    passed = decide_real_video(_real_rows(30))
    assert passed["status"] == REAL_VIDEO_TRANSFER
    assert passed["success_count"] == 30
    assert passed["record_count"] == 40

    miss = decide_real_video(_real_rows(29))
    assert miss["status"] == NO_REAL_VIDEO_TRANSFER
    assert miss["pass"] is False

    invalid = decide_real_video(_real_rows(40, engineering_invalid=True))
    assert invalid["status"] == REAL_VIDEO_NOT_EVALUATED
    assert invalid["engineering_status"] == "invalid_or_incomplete"


def test_replacement_requires_full_360_cell_gate() -> None:
    passed = decide_replacement(_replacement_rows(30), cell_count=360)
    assert passed["status"] == REPLACEMENT_OBSERVED
    assert passed["cell_count"] == 360

    incomplete = decide_replacement(_replacement_rows(40), cell_count=359)
    assert incomplete["status"] == REPLACEMENT_NOT_EVALUATED

    invalid = decide_replacement(_replacement_rows(40, engineering_invalid=True), cell_count=360)
    assert invalid["status"] == REPLACEMENT_NOT_EVALUATED
