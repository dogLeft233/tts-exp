"""Strict natural-audio replacement for the record-heldout cohort."""
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
    NO_REPLACEMENT,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
)


def decide_replacement(
    record_rows: Sequence[Mapping[str, Any]],
    *,
    cell_count: int | None = None,
) -> dict[str, Any]:
    actual_cells = EXPECTED_MATRIX_CELLS if cell_count is None else int(cell_count)
    if len(record_rows) != EVAL_RECORD_COUNT or actual_cells != EXPECTED_MATRIX_CELLS:
        return {
            "stage": "RECORD_HELDOUT_STRICT_NATURAL_REPLACEMENT",
            "status": REPLACEMENT_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "incomplete",
            "records": list(record_rows),
        }
    if not all(bool(row.get("engineering_valid")) for row in record_rows):
        return {
            "stage": "RECORD_HELDOUT_STRICT_NATURAL_REPLACEMENT",
            "status": REPLACEMENT_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "record_count": EVAL_RECORD_COUNT,
            "cell_count": EXPECTED_MATRIX_CELLS,
            "records": list(record_rows),
        }
    d_gains = [float(row["replacement_d_gain"]) for row in record_rows]
    c_gains = [float(row["replacement_c_gain"]) for row in record_rows]
    successes = sum(bool(row.get("scientific_success")) for row in record_rows)
    median_d = float(np.median(d_gains))
    median_c = float(np.median(c_gains))
    passed = successes >= MIN_SUCCESS_RECORDS and median_d >= GAIN_MIN and median_c >= GAIN_MIN
    return {
        "stage": "RECORD_HELDOUT_STRICT_NATURAL_REPLACEMENT",
        "status": REPLACEMENT_OBSERVED if passed else NO_REPLACEMENT,
        "pass": passed,
        "engineering_status": "complete",
        "record_count": EVAL_RECORD_COUNT,
        "cell_count": EXPECTED_MATRIX_CELLS,
        "success_count": successes,
        "median_replacement_d_gain": median_d,
        "median_replacement_c_gain": median_c,
        "records": list(record_rows),
        "claim_scope": (
            "empirical strict natural-audio replacement transfer through one frozen Wav2Lip/SyncNet fixed-crop protocol"
            if passed else "complete strict replacement evidence did not pass the preregistered gate"
        ),
    }
