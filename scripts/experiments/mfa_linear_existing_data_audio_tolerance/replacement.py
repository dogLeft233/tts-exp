"""Conditional replacement helpers for the exploratory continuation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..mfa_linear_sync_transfer.replacement import (
    materialize_matrix,
    render_arm_once,
    render_p0_seam,
    replacement_record_evidence,
)
from .config import (
    EVAL_RECORD_COUNT,
    EXPECTED_MATRIX_CELLS,
    GAIN_MIN,
    MIN_SUCCESS_RECORDS,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_NOT_RUN,
    REPLACEMENT_OBSERVED,
)


def decide_replacement(record_rows: Sequence[Mapping[str, Any]], *, cell_count: int) -> dict[str, Any]:
    if len(record_rows) != EVAL_RECORD_COUNT or int(cell_count) != EXPECTED_MATRIX_CELLS:
        return {
            "stage": "DIAGNOSTIC_STRICT_REPLACEMENT",
            "status": REPLACEMENT_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "records": list(record_rows),
        }
    if not all(bool(row.get("engineering_valid")) for row in record_rows):
        return {
            "stage": "DIAGNOSTIC_STRICT_REPLACEMENT",
            "status": REPLACEMENT_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "records": list(record_rows),
        }
    d_gains = [float(row["replacement_d_gain"]) for row in record_rows]
    c_gains = [float(row["replacement_c_gain"]) for row in record_rows]
    successes = sum(bool(row.get("scientific_success")) for row in record_rows)
    passed = successes >= MIN_SUCCESS_RECORDS and float(np.median(d_gains)) >= GAIN_MIN and float(np.median(c_gains)) >= GAIN_MIN
    return {
        "stage": "DIAGNOSTIC_STRICT_REPLACEMENT",
        "status": REPLACEMENT_OBSERVED if passed else REPLACEMENT_NOT_RUN,
        "pass": False,
        "engineering_status": "complete",
        "record_count": EVAL_RECORD_COUNT,
        "cell_count": EXPECTED_MATRIX_CELLS,
        "success_count_observed": successes,
        "median_replacement_d_gain": float(np.median(d_gains)),
        "median_replacement_c_gain": float(np.median(c_gains)),
        "records": list(record_rows),
        "scientific_claim_available": False,
        "claim_scope": "post-hoc exploratory replacement diagnostic only",
    }
