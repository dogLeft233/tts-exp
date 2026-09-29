"""Official paired real-video scoring for record-heldout evaluation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..mfa_linear_sync_transfer.real_video import evaluate_real_video_record as _evaluate_record
from .config import (
    EVAL_RECORD_COUNT,
    GAIN_MIN,
    MIN_SUCCESS_RECORDS,
    NO_REAL_VIDEO_TRANSFER,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
)


def evaluate_real_video_record(
    row: Mapping[str, Any],
    candidate_waveform: Tensor,
    syncnet: nn.Module,
    output_root: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    return _evaluate_record(row, candidate_waveform, syncnet, output_root, device=device)


def decide_real_video(evidence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(evidence_rows) != EVAL_RECORD_COUNT:
        return {
            "stage": "RECORD_HELDOUT_REAL_VIDEO",
            "status": REAL_VIDEO_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "incomplete",
            "records": list(evidence_rows),
            "claim_scope": "no scientific result",
        }
    if not all(bool(row.get("engineering_valid")) for row in evidence_rows):
        return {
            "stage": "RECORD_HELDOUT_REAL_VIDEO",
            "status": REAL_VIDEO_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "record_count": EVAL_RECORD_COUNT,
            "records": list(evidence_rows),
            "claim_scope": "no scientific result; evidence is incomplete or invalid",
        }
    d_gains = [float(row["d_gain"]) for row in evidence_rows]
    c_gains = [float(row["c_gain"]) for row in evidence_rows]
    successes = sum(bool(row.get("scientific_success")) for row in evidence_rows)
    median_d = float(np.median(d_gains))
    median_c = float(np.median(c_gains))
    passed = successes >= MIN_SUCCESS_RECORDS and median_d >= GAIN_MIN and median_c >= GAIN_MIN
    return {
        "stage": "RECORD_HELDOUT_REAL_VIDEO",
        "status": REAL_VIDEO_TRANSFER if passed else NO_REAL_VIDEO_TRANSFER,
        "pass": passed,
        "engineering_status": "complete",
        "record_count": EVAL_RECORD_COUNT,
        "success_count": successes,
        "median_d_gain": median_d,
        "median_c_gain": median_c,
        "records": list(evidence_rows),
        "claim_scope": (
            "empirical record-heldout transfer within the fit-only LRS3 inventory under fixed real-video SyncNet scoring"
            if passed else "complete record-heldout evidence did not pass the preregistered gate"
        ),
    }
