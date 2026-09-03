from __future__ import annotations

from typing import Any

from scripts.experiments.mfa_linear_sync_transfer.real_video import decide_real_video
from scripts.experiments.mfa_linear_sync_transfer.replacement import (
    decide_replacement,
    replacement_record_evidence,
)


def _metrics(d: float, c: float, shift: int = -1, gap: float = 0.01) -> dict[str, float | int]:
    return {"sync_d": d, "sync_c": c, "best_signed_shift": shift, "best_second_gap": gap}


def _real_row(success: bool) -> dict[str, Any]:
    gain = 0.01 if success else 0.0
    return {
        "engineering_valid": True,
        "scientific_success": success,
        "d_gain": gain,
        "c_gain": gain,
    }


def _cells(sample_id: str, *, strict: bool = True) -> list[dict[str, Any]]:
    cells = []
    for driver in "NBC":
        for evaluation in "NBC":
            is_candidate_nat = driver == "C" and evaluation == "N"
            d = 4.99 if is_candidate_nat and strict else 5.0
            c = 1.01 if is_candidate_nat and strict else 1.0
            if driver == "C" and evaluation == "C" and not strict:
                d, c = 4.0, 2.0
            cells.append({
                "status": "complete",
                "sample_id": sample_id,
                "key": f"G_{driver}_E_{evaluation}",
                "score": {"window_count": 91},
                "metrics": _metrics(d, c),
            })
    return cells


def test_real_video_gate_requires_six_and_both_medians() -> None:
    decision = decide_real_video([_real_row(i < 6) for i in range(8)])
    assert decision["pass"] is True
    assert decision["success_count"] == 6
    miss = decide_real_video([_real_row(i < 5) for i in range(8)])
    assert miss["pass"] is False
    assert miss["status"] == "NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER"


def test_replacement_uses_natural_column_not_diagonal() -> None:
    rows = [replacement_record_evidence(_cells(f"s{i}"), f"s{i}", -1) for i in range(8)]
    decision = decide_replacement(rows)
    assert decision["pass"] is True
    assert decision["status"] == "FROZEN_WAV2LIP_REPLACEMENT_TRANSFER_OBSERVED"

    rows = [replacement_record_evidence(_cells(f"s{i}", strict=False), f"s{i}", -1) for i in range(8)]
    decision = decide_replacement(rows)
    assert decision["pass"] is False
    assert decision["status"] == "NO_FROZEN_WAV2LIP_REPLACEMENT_TRANSFER"
    assert all(row["scientific_success"] is False for row in rows)


def test_replacement_missing_cell_is_not_evaluated() -> None:
    cells = _cells("s0")[:-1]
    row = replacement_record_evidence(cells, "s0", -1)
    assert row["engineering_valid"] is False
    decision = decide_replacement([row] * 8)
    assert decision["status"] == "REPLACEMENT_NOT_EVALUATED"
