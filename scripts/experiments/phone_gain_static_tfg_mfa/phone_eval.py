"""Frozen HuBERT/XLSR phone scoring and pre-registered group summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.phone_separability_enhancement.metrics import score_fixed_support
from scripts.experiments.phone_separability_enhancement.teacher import FrozenPhoneTeacher, pool_hidden

from .assets import read_pcm16


def score_waveform(teacher: FrozenPhoneTeacher, pcm: np.ndarray, tokens: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]], centroids: Mapping[str, Sequence[float]], *, layer: int, condition: str) -> dict[str, Any]:
    import torch

    values = torch.as_tensor(np.asarray(pcm, dtype=np.float32).reshape(-1) / 32768.0, device=teacher.device)
    hidden, frame_times, _ = teacher.encode(values, layer=int(layer))
    pooled = pool_hidden(hidden, frame_times, tokens, view="core")
    token_by_id = {str(token.get("token_id", f":{index}")): token for index, token in enumerate(tokens)}
    vectors: dict[str, np.ndarray] = {}
    expected = []
    rows = []
    for entry in entries:
        token_id = str(entry.get("natural_token_id") if condition != "tts" else entry.get("tts_token_id"))
        if not token_id or token_id == "None":
            token_id = f":{entry.get('natural_token_index' if condition != 'tts' else 'tts_token_index')}"
        key = str(entry.get("support_key"))
        expected.append({"support_key": key, "label": str(entry.get("label", "")), "source_group": str(entry.get("source_group", ""))})
        pooled_row = pooled.get(token_id)
        if pooled_row is not None:
            vectors[key] = pooled_row["embedding"].detach().cpu().numpy()
            token = token_by_id.get(token_id)
            if token is not None:
                start = float(token["start_s"])
                end = float(token["end_s"])
                duration = end - start
                indices = torch.nonzero((frame_times >= start + 0.20 * duration) & (frame_times < end - 0.20 * duration), as_tuple=False).reshape(-1)
                frame_indices = [int(value) for value in indices.detach().cpu().tolist()]
            else:
                frame_indices = []
            rows.append({"support_key": key, "label": str(entry.get("label", "")), "source_group": str(entry.get("source_group", "")), "embedding": vectors[key], "frame_indices": frame_indices})
    scored = score_fixed_support(rows, centroids, expected=expected, vectors_by_key=vectors)
    scored.update({"condition": condition, "layer": int(layer), "sample_count": int(np.asarray(pcm).size)})
    return scored


def score_candidate_set(teacher: FrozenPhoneTeacher, row: Mapping[str, Any], support: Mapping[str, Any], arms: Mapping[str, np.ndarray], *, encoder: str, view: str = "natural_primary", split: str = "e_seen") -> dict[str, Any]:
    if view == "natural_primary" and "T" in arms:
        raise ValueError("TTS arm is not defined on natural_primary; use matched_nt with TTS timing")
    entries = [item for item in support[view][encoder] if str(item.get("analysis_split")) == str(split) and str(item.get("pair_id")) == str(row["pair_id"]) and bool(item.get("pair_eligible", True))]
    centroids = support["mixed_centroids"][encoder]
    layer = 6 if encoder == "hubert" else 10
    result = {}
    for arm, pcm in arms.items():
        tokens = row["tts"]["tokens"] if arm == "T" and view == "matched_nt" else row["natural"]["tokens"]
        result[arm] = score_waveform(teacher, pcm, tokens, entries, centroids, layer=layer, condition="tts" if arm == "T" else "natural")
    return {"pair_id": row["pair_id"], "source_group": row["source_group"], "encoder": encoder, "view": view, "scores": result}


def group_effects(rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str = "accuracy") -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scores = row.get("scores", {})
        if left not in scores or right not in scores:
            continue
        left_value = scores[left].get(metric)
        right_value = scores[right].get(metric)
        if left_value is not None and right_value is not None:
            grouped[str(row["source_group"])].append(float(left_value) - float(right_value))
    return {group: float(np.mean(values)) for group, values in grouped.items() if values}


def _bootstrap_effects(effects: Mapping[str, float], *, draws: int, seed: int, confidence: float) -> dict[str, Any]:
    values = np.asarray([float(value) for value in effects.values()], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "draws": int(draws), "seed": int(seed), "confidence": float(confidence)}
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    sampled = values[rng.integers(0, values.size, size=(int(draws), values.size))].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(sampled, tail)), "ci_high": float(np.quantile(sampled, 1.0 - tail)), "n_groups": int(values.size), "draws": int(draws), "seed": int(seed), "confidence": float(confidence), "group_effects": {str(key): float(value) for key, value in effects.items()}}


def summarize_pairwise(rows: Sequence[Mapping[str, Any]], comparisons: Sequence[tuple[str, str]], *, metric: str = "accuracy", draws: int = 20000, seed: int = 20260922, confidence: float = 0.95) -> dict[str, Any]:
    result = {}
    for left, right in comparisons:
        effects = group_effects(rows, left, right, metric=metric)
        result[f"{left}-{right}"] = _bootstrap_effects(effects, draws=draws, seed=seed, confidence=confidence)
    return result


def classify_noninferiority(summary: Mapping[str, Any], *, epsilon: float = 0.02) -> str:
    estimate = summary.get("estimate")
    lower = summary.get("ci_low")
    upper = summary.get("ci_high")
    if estimate is None or lower is None or upper is None:
        return "INCONCLUSIVE_MISSING"
    if float(lower) >= -float(epsilon) and float(upper) <= float(epsilon):
        return "EQUIVALENT"
    if float(lower) > float(epsilon):
        return "HIGHER"
    if float(lower) >= -float(epsilon):
        return "NONINFERIOR"
    return "INCONCLUSIVE"


__all__ = ["classify_noninferiority", "group_effects", "score_candidate_set", "score_waveform", "summarize_pairwise"]
