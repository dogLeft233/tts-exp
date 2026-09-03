"""Paired official SyncNet scoring on fixed real-video coordinates."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..mfa_linear_real_video_sync.evaluate import curve_metrics, mux_condition_once, official_file_curve, write_pcm16_wav_once
from ..mfa_linear_real_video_sync.protocol import load_mfa_linear_waveform, sha256_file, write_json_once
from ..mfa_linear_real_video_sync.syncnet_loss import log_mel_trust_components, waveform_qc
from .config import (
    GAIN_MIN,
    NO_REAL_VIDEO_TRANSFER,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    SATURATION_FRACTION_MAX,
    TARGET_GAP,
)
from .protocol import TransferProtocolError


def _finite_curve(curve: Sequence[float]) -> bool:
    values = np.asarray(curve, dtype=np.float64)
    return values.shape == (31,) and bool(np.isfinite(values).all())


def _score_condition(
    syncnet: nn.Module,
    condition: Path,
    row: Mapping[str, Any],
    wav_lock: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    visual = row["visual"]
    return official_file_curve(
        syncnet,
        condition,
        expected_frame_hashes=visual["decoded_bgr_frame_sha256"],
        expected_scoring_frame_hashes=visual.get("scoring_bgr_frame_sha256"),
        expected_pcm_sha256=str(wav_lock["pcm_sha256"]),
        device=device,
    )


def evaluate_real_video_record(
    row: Mapping[str, Any],
    candidate_waveform: Tensor,
    syncnet: nn.Module,
    output_root: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    sid = str(row["sample_id"])
    root = Path(output_root).resolve() / "03_real_video" / sid
    mfa_waveform, mfa_lock = load_mfa_linear_waveform(
        row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"])
    )
    baseline = torch.from_numpy(mfa_waveform)[None, None]
    if candidate_waveform.ndim != 3 or tuple(candidate_waveform.shape) != tuple(baseline.shape):
        raise TransferProtocolError("candidate waveform shape does not match baseline")
    baseline_wav = write_pcm16_wav_once(root / "baseline.wav", baseline)
    candidate_wav = write_pcm16_wav_once(root / "candidate.wav", candidate_waveform.detach().cpu())
    canonical = Path(str(row["visual"]["path"]))
    baseline_condition = root / "baseline.avi"
    candidate_condition = root / "candidate.avi"
    mux_condition_once(canonical, baseline_wav["path"], baseline_condition)
    mux_condition_once(canonical, candidate_wav["path"], candidate_condition)
    base_curve = _score_condition(syncnet, baseline_condition, row, baseline_wav, device=device)
    cand_curve = _score_condition(syncnet, candidate_condition, row, candidate_wav, device=device)
    base_metrics = curve_metrics(base_curve["curve"])
    cand_metrics = curve_metrics(cand_curve["curve"])
    target = row["natural_target"]
    d_gain = base_metrics["sync_d"] - cand_metrics["sync_d"]
    c_gain = cand_metrics["sync_c"] - base_metrics["sync_c"]
    qc = waveform_qc(candidate_waveform, baseline, residual_bound=0.05)
    with torch.no_grad():
        mel_distance = float(log_mel_trust_components(candidate_waveform, baseline)["distance"].cpu())
    qc["mel_distance"] = mel_distance
    target_match = (
        cand_metrics["best_signed_shift"] == int(target["target_offset"])
        and cand_metrics["best_second_gap"] > TARGET_GAP
    )
    engineering_valid = (
        _finite_curve(base_curve["curve"]) and _finite_curve(cand_curve["curve"])
        and base_curve["window_count"] == 91 and cand_curve["window_count"] == 91
        and base_curve["decoded_bgr_frame_sha256"] == cand_curve["decoded_bgr_frame_sha256"]
        and qc.get("finite") is True and qc.get("exact_shape") is True
        and qc.get("residual_bound_pass") is True
        and qc.get("pcm_saturation_pass") is True
        and float(qc.get("mel_distance", math.inf)) <= 0.10
        and float(qc.get("pcm_saturation_fraction", math.inf)) <= SATURATION_FRACTION_MAX
    )
    success = bool(
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
        "baseline_curve": base_curve,
        "candidate_curve": cand_curve,
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "d_gain": float(d_gain),
        "c_gain": float(c_gain),
        "target_match": target_match,
        "candidate_best_second_gap": cand_metrics["best_second_gap"],
        "qc": qc,
        "engineering_valid": engineering_valid,
        "scientific_success": success,
        "official_only_decision": True,
    }
    write_json_once(root / "evidence.json", evidence)
    return evidence


def decide_real_video(evidence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(evidence_rows) != 8:
        return {
            "stage": "P3_ADAPTER_HELDOUT_REAL_VIDEO",
            "status": REAL_VIDEO_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "incomplete",
            "claim_scope": "no scientific result",
            "records": list(evidence_rows),
        }
    engineering_valid = [bool(row.get("engineering_valid")) for row in evidence_rows]
    if not all(engineering_valid):
        return {
            "stage": "P3_ADAPTER_HELDOUT_REAL_VIDEO",
            "status": REAL_VIDEO_NOT_EVALUATED,
            "pass": False,
            "engineering_status": "invalid_or_incomplete",
            "records": list(evidence_rows),
            "claim_scope": "no scientific result; real-video evidence is incomplete or invalid",
        }
    d_gains = [float(row["d_gain"]) for row in evidence_rows]
    c_gains = [float(row["c_gain"]) for row in evidence_rows]
    successes = sum(bool(row.get("scientific_success")) for row in evidence_rows)
    passed = successes >= 6 and float(np.median(d_gains)) >= GAIN_MIN and float(np.median(c_gains)) >= GAIN_MIN
    return {
        "stage": "P3_ADAPTER_HELDOUT_REAL_VIDEO",
        "status": REAL_VIDEO_TRANSFER if passed else NO_REAL_VIDEO_TRANSFER,
        "pass": passed,
        "engineering_status": "complete",
        "record_count": 8,
        "success_count": successes,
        "median_d_gain": float(np.median(d_gains)),
        "median_c_gain": float(np.median(c_gains)),
        "records": list(evidence_rows),
        "claim_scope": "empirical transfer on the preregistered eight adapter-heldout source groups under fixed real-video SyncNet scoring",
        "disallowed_claims": ["population generalization", "sealed-test generalization", "perceptual quality", "content/speaker preservation", "multi-TFG replacement safety", "production readiness"],
    }
