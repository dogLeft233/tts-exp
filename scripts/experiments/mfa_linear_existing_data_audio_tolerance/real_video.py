"""Official real-video scoring under the explicitly post-hoc audio tolerance."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..mfa_linear_real_video_sync.evaluate import curve_metrics, mux_condition_once, official_file_curve, write_pcm16_wav_once
from ..mfa_linear_real_video_sync.protocol import load_mfa_linear_waveform, sha256_file
from ..mfa_linear_real_video_sync.syncnet_loss import log_mel_trust_components, waveform_qc
from .config import (
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_MEL_LIMIT,
    DIAGNOSTIC_NOT_EVALUATED,
    EVAL_RECORD_COUNT,
    GAIN_MIN,
    MIN_SUCCESS_RECORDS,
    TARGET_GAP,
)
from .protocol import read_object


def _finite_curve(curve: Sequence[float]) -> bool:
    values = np.asarray(curve, dtype=np.float64)
    return values.shape == (31,) and bool(np.isfinite(values).all())


def evaluate_record(
    row: Mapping[str, Any],
    candidate: Tensor,
    syncnet: nn.Module,
    output_root: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    sid = str(row["sample_id"])
    root = Path(output_root).resolve() / "03_real_video" / sid
    baseline_np, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
    baseline = torch.from_numpy(baseline_np)[None, None]
    if tuple(candidate.shape) != tuple(baseline.shape):
        raise ValueError(f"candidate waveform shape mismatch for {sid}")
    baseline_wav = write_pcm16_wav_once(root / "baseline.wav", baseline)
    candidate_wav = write_pcm16_wav_once(root / "candidate.wav", candidate.detach().cpu())
    canonical = Path(str(row["visual"]["path"]))
    baseline_condition = root / "baseline.avi"
    candidate_condition = root / "candidate.avi"
    mux_condition_once(canonical, baseline_wav["path"], baseline_condition)
    mux_condition_once(canonical, candidate_wav["path"], candidate_condition)
    baseline_curve = official_file_curve(
        syncnet,
        baseline_condition,
        expected_frame_hashes=row["visual"]["decoded_bgr_frame_sha256"],
        expected_scoring_frame_hashes=row["visual"].get("scoring_bgr_frame_sha256"),
        expected_pcm_sha256=baseline_wav["pcm_sha256"],
        device=device,
    )
    candidate_curve = official_file_curve(
        syncnet,
        candidate_condition,
        expected_frame_hashes=row["visual"]["decoded_bgr_frame_sha256"],
        expected_scoring_frame_hashes=row["visual"].get("scoring_bgr_frame_sha256"),
        expected_pcm_sha256=candidate_wav["pcm_sha256"],
        device=device,
    )
    baseline_metrics = curve_metrics(baseline_curve["curve"])
    candidate_metrics = curve_metrics(candidate_curve["curve"])
    d_gain = float(baseline_metrics["sync_d"] - candidate_metrics["sync_d"])
    c_gain = float(candidate_metrics["sync_c"] - baseline_metrics["sync_c"])
    target = row["natural_target"]
    target_match = (
        candidate_metrics["best_signed_shift"] == int(target["target_offset"])
        and candidate_metrics["best_second_gap"] > TARGET_GAP
    )
    qc = waveform_qc(candidate, baseline, residual_bound=0.05)
    with torch.no_grad():
        qc["mel_distance"] = float(log_mel_trust_components(candidate, baseline)["distance"].cpu())
    qc["original_mel_limit"] = 0.10
    qc["diagnostic_mel_limit"] = DIAGNOSTIC_MEL_LIMIT
    qc["diagnostic_tolerance_pass"] = qc["mel_distance"] <= DIAGNOSTIC_MEL_LIMIT
    engineering_valid = bool(
        _finite_curve(baseline_curve["curve"])
        and _finite_curve(candidate_curve["curve"])
        and baseline_curve["window_count"] == 91
        and candidate_curve["window_count"] == 91
        and baseline_curve["decoded_bgr_frame_sha256"] == candidate_curve["decoded_bgr_frame_sha256"]
        and qc.get("finite") is True
        and qc.get("exact_shape") is True
        and qc.get("residual_bound_pass") is True
        and qc.get("pcm_saturation_pass") is True
        and qc["diagnostic_tolerance_pass"]
    )
    scientific_success = bool(
        engineering_valid and d_gain >= GAIN_MIN and c_gain >= GAIN_MIN and target_match
    )
    evidence = {
        "status": "complete",
        "sample_id": sid,
        "source_group": row["source_group"],
        "protocol_split": row["protocol_split"],
        "mfa_audio_sha256": row["mfa_audio_sha256"],
        "candidate_audio_sha256": candidate_wav["pcm_sha256"],
        "visual_sha256": row["visual"]["sha256"],
        "natural_target": target,
        "baseline_curve": baseline_curve,
        "candidate_curve": candidate_curve,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "d_gain": d_gain,
        "c_gain": c_gain,
        "target_match": target_match,
        "candidate_best_second_gap": candidate_metrics["best_second_gap"],
        "qc": qc,
        "engineering_valid": engineering_valid,
        "scientific_success_observed": scientific_success,
        "official_only_decision": True,
        "post_hoc_audio_tolerance": True,
    }
    from ..mfa_linear_real_video_sync.protocol import write_json_once
    write_json_once(root / "evidence.json", evidence)
    return evidence


def decide_diagnostic(evidence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(evidence_rows) != EVAL_RECORD_COUNT or not all(bool(row.get("engineering_valid")) for row in evidence_rows):
        return {
            "stage": "DIAGNOSTIC_REAL_VIDEO",
            "status": DIAGNOSTIC_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "record_count": len(evidence_rows),
            "records": list(evidence_rows),
            "scientific_claim_available": False,
        }
    d_gains = [float(row["d_gain"]) for row in evidence_rows]
    c_gains = [float(row["c_gain"]) for row in evidence_rows]
    successes = sum(bool(row.get("scientific_success_observed")) for row in evidence_rows)
    real_gate = successes >= MIN_SUCCESS_RECORDS and float(np.median(d_gains)) >= GAIN_MIN and float(np.median(c_gains)) >= GAIN_MIN
    return {
        "stage": "DIAGNOSTIC_REAL_VIDEO",
        "status": DIAGNOSTIC_COMPLETE,
        "pass": False,
        "engineering_status": "complete",
        "record_count": EVAL_RECORD_COUNT,
        "success_count_observed": successes,
        "median_d_gain": float(np.median(d_gains)),
        "median_c_gain": float(np.median(c_gains)),
        "unchanged_real_video_gate_pass": real_gate,
        "records": list(evidence_rows),
        "scientific_claim_available": False,
        "claim_scope": "post-hoc descriptive diagnostic only; no pre-registered scientific transfer claim",
    }
