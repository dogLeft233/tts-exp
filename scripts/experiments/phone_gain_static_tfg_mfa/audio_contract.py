"""PCM-domain export contract for the repaired gain-head experiment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scripts.experiments.phone_separability_enhancement.audio import export_pcm


def _measure(source: np.ndarray, candidate: np.ndarray, protected: np.ndarray) -> dict[str, Any]:
    editable = ~protected
    source_f = source.astype(np.float64)
    candidate_f = candidate.astype(np.float64)
    residual = candidate_f - source_f
    residual_energy = float(np.sum(residual[editable] ** 2)) if np.any(editable) else 0.0
    signal_energy = float(np.sum(source_f[editable] ** 2)) if np.any(editable) else 0.0
    ratio = residual_energy / max(signal_energy, 1e-8)
    snr = 10.0 * math.log10(max(signal_energy, 1e-8) / max(residual_energy, 1e-8)) if np.any(editable) and residual_energy else None
    source_rms = float(np.sqrt(np.mean(source_f[editable] ** 2))) if np.any(editable) else 0.0
    candidate_rms = float(np.sqrt(np.mean(candidate_f[editable] ** 2))) if np.any(editable) else 0.0
    rms_change = 20.0 * math.log10(max(candidate_rms, 1e-12) / max(source_rms, 1e-12)) if np.any(editable) else 0.0
    source_sat = np.abs(source.astype(np.int32)) >= 32767
    candidate_sat = np.abs(candidate.astype(np.int32)) >= 32767
    return {
        "residual_energy_ratio": float(ratio),
        "snr_db": snr,
        "rms_change_db": float(rms_change),
        "new_saturated_samples": int(np.count_nonzero(candidate_sat & ~source_sat)),
        "protected_pcm_equal": bool(np.array_equal(source[protected], candidate[protected])),
    }


def export_safe_pcm(
    waveform: Any,
    source_pcm: np.ndarray,
    protected: np.ndarray,
    *,
    max_residual_ratio: float = 0.01,
    min_snr_db: float = 20.0,
    max_rms_change_db: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize and re-check the actual PCM, using a fixed residual grid.

    The grid is independent of phone/TFG scores.  ``alpha=0`` is an explicit
    identity result, never a silently substituted zero-loss training target.
    """
    raw = np.asarray(waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else waveform, dtype=np.float64).reshape(-1)
    source = np.asarray(source_pcm, dtype=np.int16).reshape(-1)
    hard = np.asarray(protected, dtype=bool).reshape(-1)
    if raw.size != source.size or hard.size != source.size:
        raise ValueError("PCM export arrays have different lengths")
    if not np.isfinite(raw).all():
        raise ValueError("PCM export waveform contains non-finite values")
    editable = ~hard
    if not np.any(editable):
        return source.copy(), {"alpha": 0.0, "reason": "NO_EDITABLE_SUPPORT", "identity": True, **_measure(source, source, hard)}
    source_float = source.astype(np.float64) / 32768.0
    residual = raw - source_float
    for alpha in [1.0, *[value / 100.0 for value in range(99, 0, -1)], 0.0]:
        candidate, quant_meta = export_pcm(source_float + float(alpha) * residual, source, protected=hard)
        metrics = _measure(source, candidate, hard)
        passed = bool(
            metrics["protected_pcm_equal"]
            and metrics["residual_energy_ratio"] <= float(max_residual_ratio)
            and (alpha == 0.0 or (metrics["snr_db"] is not None and float(metrics["snr_db"]) >= float(min_snr_db)))
            and abs(float(metrics["rms_change_db"])) <= float(max_rms_change_db)
            and metrics["new_saturated_samples"] == 0
        )
        if passed:
            return candidate, {
                "alpha": float(alpha),
                "identity": bool(alpha == 0.0),
                "reason": "IDENTITY" if alpha == 0.0 else "PCM_CONTRACT_PASS",
                "projection_before_quantization": {"max_abs_float": float(np.max(np.abs(residual[editable]))) if np.any(editable) else 0.0},
                "quantization": quant_meta,
                **metrics,
            }
    raise ValueError("no fixed PCM residual scale satisfies the audio contract")


__all__ = ["export_safe_pcm"]
