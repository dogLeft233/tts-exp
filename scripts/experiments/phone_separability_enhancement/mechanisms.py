"""Post-lock static/dynamic and codec mechanism contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .audio import BandGainRenderer, export_pcm, gain_field_from_parameters, make_protected_mask, validate_distortion


def _torch():
    import torch

    return torch


def _dose(pcm: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    edit = np.asarray(mask, dtype=bool)
    residual = pcm.astype(np.float64) - source.astype(np.float64)
    return float(np.sum(residual[edit] ** 2) / max(np.sum(source.astype(np.float64)[edit] ** 2), 1e-8)) if np.any(edit) else 0.0


def dynamic_ablation_fields(gain_db: np.ndarray, frame_times: np.ndarray, tokens: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    field = np.asarray(gain_db, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("gain_db must have shape [bands, frames]")
    mean = field.copy()
    reverse = field.copy()
    for token in tokens:
        if not bool(token.get("speech", not token.get("silence", False))):
            continue
        indices = np.flatnonzero((frame_times >= float(token["start_s"])) & (frame_times < float(token["end_s"])))
        if indices.size:
            mean[:, indices] = field[:, indices].mean(axis=1, keepdims=True)
            reverse[:, indices] = field[:, indices][:, ::-1]
    return {"PHONE_TIME_MEAN": mean, "PHONE_TIME_REVERSE": reverse}


def dose_match_scale(source_pcm: np.ndarray, original_pcm: np.ndarray, ablation_field: np.ndarray, renderer: BandGainRenderer, mask: np.ndarray, *, scale_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0), max_gain_db: float = 6.0) -> dict[str, Any]:
    torch = _torch()
    source = np.asarray(source_pcm, dtype=np.int16)
    target_dose = _dose(np.asarray(original_pcm, dtype=np.int16), source, np.asarray(mask) > 0)
    best = None
    x = torch.as_tensor(source.astype(np.float32) / 32768.0, device=next(renderer_device(renderer), torch.device("cpu"))) if False else torch.as_tensor(source.astype(np.float32) / 32768.0)
    for scale in scale_grid:
        field = torch.as_tensor(np.clip(np.asarray(ablation_field, dtype=np.float32) * float(scale), -max_gain_db, max_gain_db)).unsqueeze(0)
        output, meta = renderer.render(x, field, torch.as_tensor(mask, dtype=torch.float32))
        pcm, pcm_meta = export_pcm(output, source, protected=np.asarray(mask) <= 0)
        dose = _dose(pcm, source, np.asarray(mask) > 0)
        candidate = {"scale": float(scale), "pcm": pcm, "dose": dose, "relative_error": abs(dose - target_dose) / max(target_dose, 1e-8), "render": meta, "pcm_meta": pcm_meta}
        if dose <= target_dose * 1.05 + 1e-12 and (best is None or candidate["relative_error"] < best["relative_error"] or (candidate["relative_error"] == best["relative_error"] and scale < best["scale"])):
            best = candidate
    return {"target_dose": target_dose, "selected": best, "status": "DOSE_MATCHED" if best is not None and best["relative_error"] <= 0.05 else "DOSE_UNMATCHED"}


def renderer_device(renderer: BandGainRenderer):
    return []


def codec_status(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the fixed DAC binding without silently substituting another codec."""

    source = Path(repo_root) / "scripts/experiments/lrs3_dac16k_codec_identity"
    if not source.is_dir():
        return {"status": "DEPENDENCY_UNAVAILABLE", "reason": "DAC_SOURCE_MISSING"}
    checkpoint = config.get("mechanisms", {}).get("dac_checkpoint_sha256")
    return {"status": "REGISTERED", "source_dir": str(source), "tag": config.get("mechanisms", {}).get("dac_tag"), "source_commit": config.get("mechanisms", {}).get("dac_source_commit"), "checkpoint_sha256": checkpoint, "note": "actual codec execution is optional and must verify current input/output hashes"}


__all__ = ["codec_status", "dose_match_scale", "dynamic_ablation_fields"]
