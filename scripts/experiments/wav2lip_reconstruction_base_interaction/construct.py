from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config


def _k_for(sample_id: str, rows: list[dict[str, Any]], masks: dict[str, dict[str, Any]]) -> set[int]:
    sets = []
    for row in rows:
        if str(row["sample_id"]) != sample_id: continue
        current: set[int] = set()
        for used in row.get("used_masks", []):
            item = masks.get(str(used["mask_sha256"]))
            if item is None: raise rt.ProtocolError(f"unknown mask: {sample_id}")
            s, e = int(used["global_start_frame"]), int(used["global_end_frame"])
            if (s, e) != (int(item["natural_core_start_frame"]), int(item["natural_core_end_frame"])): raise rt.ProtocolError(f"mask identity mismatch: {sample_id}")
            current.update(range(s, e))
        sets.append(current)
    if not sets or any(value != sets[0] for value in sets[1:]): raise rt.ProtocolError(f"K differs across seed/condition: {sample_id}")
    return sets[0]


def construct_factorial(natural: np.ndarray, d_rows: list[dict[str, Any]], masks: dict[str, dict[str, Any]], sample_id: str) -> dict[str, Any]:
    m = np.asarray(natural, dtype=np.float64)
    if m.shape != (80, 308) or not np.isfinite(m).all(): raise rt.ProtocolError("invalid natural mel")
    rows = [row for row in d_rows if str(row["sample_id"]) == sample_id and str(row.get("condition")) in config.CONDITIONS]
    if len(rows) != 9: raise rt.ProtocolError(f"expected 9 D rows: {sample_id}")
    by = {(int(row["seed"]), str(row["condition"])): np.asarray(np.load(Path(str(row["path"])), allow_pickle=False), dtype=np.float64) for row in rows}
    if any(value.shape != (80, 308) or not np.isfinite(value).all() for value in by.values()): raise rt.ProtocolError("invalid D driver shape")
    k = _k_for(sample_id, rows, masks); outside = np.ones(308, dtype=bool); outside[list(k)] = False
    z = np.mean(np.stack([by[(seed, "NAT_ONLY")] for seed in (20260901, 20260902, 20260903)], axis=0), axis=0, dtype=np.float64)
    c_raw = np.mean(np.stack([by[(seed, "PAIRED_TTS")] - by[(seed, "NAT_ONLY")] for seed in (20260901, 20260902, 20260903)], axis=0), axis=0, dtype=np.float64)
    w_raw = np.mean(np.stack([by[(seed, "SAME_PHONE_WRONG_INSTANCE")] - by[(seed, "NAT_ONLY")] for seed in (20260901, 20260902, 20260903)], axis=0), axis=0, dtype=np.float64)
    if np.max(np.abs(c_raw[:, outside])) > 1e-6 or np.max(np.abs(w_raw[:, outside])) > 1e-6: raise rt.ProtocolError("D residual is nonzero outside K")
    c_norm = float(np.linalg.norm(c_raw[:, list(k)])); w_norm = float(np.linalg.norm(w_raw[:, list(k)]))
    if c_norm <= 1e-12 or w_norm <= 1e-12: raise rt.ProtocolError("INPUT_DEGENERATE: content residual")
    w_raw *= c_norm / w_norm; maximum = max(float(np.max(np.abs(c_raw))), float(np.max(np.abs(w_raw)))); a = min(0.25, 0.5 / maximum); c = a * c_raw; w = a * w_raw
    r_base = z - m
    if np.max(np.abs(r_base[:, outside])) > 1e-6: raise rt.ProtocolError("NAT_ONLY base residual is nonzero outside K")
    base_norm = float(np.linalg.norm(r_base))
    if base_norm <= 1e-12: raise rt.ProtocolError("INPUT_DEGENERATE: base residual")
    b = min(0.25, 0.5 / float(np.max(np.abs(r_base))))
    base = np.clip(m + b * r_base, -4, 4)  # deliberately float64 until the four output casts
    n_content = np.clip(m + c, -4, 4).astype(np.float32)
    base_content = np.clip(base + c, -4, 4).astype(np.float32)
    base_wrong = np.clip(base + w, -4, 4).astype(np.float32)
    natural32 = m.astype(np.float32); base32 = base.astype(np.float32)
    n_delta = float(np.linalg.norm(n_content.astype(np.float64) - natural32.astype(np.float64))); bc_delta = float(np.linalg.norm(base_content.astype(np.float64) - base32.astype(np.float64))); bw_delta = float(np.linalg.norm(base_wrong.astype(np.float64) - base32.astype(np.float64)))
    ratios = {"base_content_over_n_content": bc_delta / n_delta if n_delta > 0 else 0.0, "base_wrong_over_base_content": bw_delta / bc_delta if bc_delta > 0 else 0.0}
    valid = all(0.95 <= value <= 1.05 for value in ratios.values()) and n_delta > 0 and bc_delta > 0 and bw_delta > 0
    if not valid: raise rt.ProtocolError("interaction norm control invalid")
    return {"N": natural32, "N_CONTENT": n_content, "BASE": base32, "BASE_CONTENT": base_content, "BASE_WRONG": base_wrong, "metadata": {"K": sorted(k), "seed_count": 3, "a": float(a), "b": float(b), "content_norm": c_norm, "wrong_norm_after_scale": float(np.linalg.norm(w_raw)), "base_norm": base_norm, "ratios": ratios, "interaction_norm_valid": valid}}
