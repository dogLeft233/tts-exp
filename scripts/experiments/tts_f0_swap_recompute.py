#!/usr/bin/env python3
"""Independently recompute the F0-swap score statistics.

This checker intentionally does not import ``tts_f0_swap.py``.  It reads the
frozen score and input manifests, reconstructs the published contrasts, and
repeats the speaker-cluster bootstrap with the protocol's fixed RNG.  Its only
purpose is to catch a statistics or row-selection error in the production
runner; it is not another scoring implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SEED = 20260918
DRAW_COUNT = 20_000
BONFERRONI_ALPHA = 0.05 / 4.0
METRICS = (
    "g_N", "l_T", "h_N", "h_T", "G_raw", "G_id", "G_id_minus_G_raw",
    "own_gain_N", "own_gain_T", "evaluator_N", "evaluator_T",
    "raw_replace_N", "raw_replace_T", "interaction_N", "interaction_T",
)


class RecomputeError(RuntimeError):
    """Raised when a frozen output cannot be independently recomputed."""


REPAIR_PROTOCOL_ID = "tts_f0_swap_repair_v2"
REPAIR_REQUIRED_KEYS = ("N", "T")
REPAIR_ARMS = ("RAW", "ID", "LEVEL", "CONTOUR")
REPAIR_MEASUREMENTS = ("H_source", "H_RAW", "H_ID", "H_LEVEL", "H_CONTOUR", "D_RAW", "D_ID", "D_LEVEL", "D_CONTOUR")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(payload, dict):
        raise RecomputeError(f"expected JSON object: {path}")
    return payload


def _verify_self_hash(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recorded = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != hashlib.sha256(_canonical(body)).hexdigest():
        raise RecomputeError(f"self-hash mismatch: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RecomputeError(f"non-finite {name}")
    return result


def _row_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            raise RecomputeError("score manifest contains a non-complete row")
        key = (
            str(row.get("paired_key")), str(row.get("receiver")),
            str(row.get("category")), str(row.get("video_arm")),
            str(row.get("audio_arm")),
        )
        if key in result:
            raise RecomputeError(f"duplicate score cell: {key}")
        for field in ("C", "D", "B", "d_k0"):
            if row.get(field) is not None:
                _finite(row[field], field)
        result[key] = row
    return result


def _cell(index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]], pair: str, receiver: str, category: str, video_arm: str, audio_arm: str) -> float:
    key = (pair, receiver, category, video_arm, audio_arm)
    try:
        return _finite(index[key]["C"], f"C/{key}")
    except KeyError as exc:
        raise RecomputeError(f"missing score cell: {key}") from exc


def _record_contrasts(pair: str, speaker: str, index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    n_raw = _cell(index, pair, "N", "own", "N_RAW", "N_RAW")
    n_id = _cell(index, pair, "N", "own", "N_ID", "N_ID")
    n_contour = _cell(index, pair, "N", "own", "N_CONTOUR", "N_CONTOUR")
    t_raw = _cell(index, pair, "T", "own", "T_RAW", "T_RAW")
    t_id = _cell(index, pair, "T", "own", "T_ID", "T_ID")
    t_contour = _cell(index, pair, "T", "own", "T_CONTOUR", "T_CONTOUR")

    fa_n_contour = _cell(index, pair, "N", "fixed_audio", "N_CONTOUR", "N_ID")
    fa_n_level = _cell(index, pair, "N", "fixed_audio", "N_LEVEL", "N_ID")
    fa_t_contour = _cell(index, pair, "T", "fixed_audio", "T_CONTOUR", "T_ID")
    fa_t_level = _cell(index, pair, "T", "fixed_audio", "T_LEVEL", "T_ID")
    fv_n_contour = _cell(index, pair, "N", "fixed_video", "N_ID", "N_CONTOUR")
    fv_t_contour = _cell(index, pair, "T", "fixed_video", "T_ID", "T_CONTOUR")
    rr_n_contour = _cell(index, pair, "N", "raw_replacement", "N_CONTOUR", "N_RAW")
    rr_n_id = _cell(index, pair, "N", "raw_replacement", "N_ID", "N_RAW")
    rr_t_contour = _cell(index, pair, "T", "raw_replacement", "T_CONTOUR", "T_RAW")
    rr_t_id = _cell(index, pair, "T", "raw_replacement", "T_ID", "T_RAW")

    values = {
        "g_N": fa_n_contour - n_id,
        "l_T": t_id - fa_t_contour,
        "h_N": fa_n_contour - fa_n_level,
        "h_T": fa_t_level - fa_t_contour,
        "G_raw": t_raw - n_raw,
        "G_id": t_id - n_id,
        "own_gain_N": n_contour - n_id,
        "own_gain_T": t_contour - t_id,
        "evaluator_N": fv_n_contour - n_id,
        "evaluator_T": fv_t_contour - t_id,
        "raw_replace_N": rr_n_contour - rr_n_id,
        "raw_replace_T": rr_t_contour - rr_t_id,
        "interaction_N": n_contour - fa_n_contour - fv_n_contour + n_id,
        "interaction_T": t_contour - fa_t_contour - fv_t_contour + t_id,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise RecomputeError(f"non-finite contrast for {pair}")
    return {"paired_key": pair, "speaker_id": speaker, **{name: float(value) for name, value in values.items()}}


def _indices(group_count: int) -> np.ndarray:
    if group_count < 1:
        raise RecomputeError("speaker bootstrap requires at least one group")
    rng = np.random.Generator(np.random.PCG64(SEED))
    return rng.integers(0, group_count, size=(DRAW_COUNT, group_count), dtype=np.int64)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _bootstrap(values: Sequence[float], speakers: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    if len(values) != len(speakers) or len(values) == 0:
        return {"status": "NOT_ESTIMABLE", "reason": "empty or mismatched values"}
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, speaker in zip(values, speakers, strict=True):
        grouped[str(speaker)].append(_finite(value, "bootstrap value"))
    labels = sorted(grouped)
    if indices.shape != (DRAW_COUNT, len(labels)):
        raise RecomputeError("bootstrap index shape differs from speaker labels")
    group_values = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    draws = group_values[indices].mean(axis=1, dtype=np.float64)
    corrected_probability = BONFERRONI_ALPHA / 2.0
    corrected = [_quantile(draws, corrected_probability), _quantile(draws, 1.0 - corrected_probability)]
    return {
        "status": "COMPLETE",
        "mean": float(group_values.mean()),
        "ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
        "ci98_75_bonferroni": corrected,
        "speaker_count": len(labels),
        "speaker_labels": labels,
        "speaker_values": {label: float(value) for label, value in zip(labels, group_values, strict=True)},
        "pair_count": len(values),
        "positive_speaker_count": int(np.sum(group_values > 0.0)),
        "draws": DRAW_COUNT,
        "seed": SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": "linear",
        "status_by_corrected_ci": "POSITIVE" if corrected[0] > 0.0 else "NEGATIVE" if corrected[1] < 0.0 else "INCONCLUSIVE",
    }


def _same_number(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _audio_binding_check(run_dir: Path) -> dict[str, Any]:
    """Inspect one real pair without importing the production audio code."""

    manifest_path = run_dir / "audio_manifest.json"
    if not manifest_path.is_file():
        return {"status": "NOT_AVAILABLE", "reason": "formal audio manifest is absent"}
    try:
        manifest = _verify_self_hash(manifest_path)
        # The v3 paired-Harvest runner has a different audio QC reference and
        # is validated by its own stage contract.  This score recomputer is
        # intentionally limited to the v2 repair audio binding; do not apply
        # v2's DIO gate to a v3 formal run while recomputing score statistics.
        if manifest.get("protocol_id") != REPAIR_PROTOCOL_ID:
            return {"status": "NOT_APPLICABLE", "protocol_id": manifest.get("protocol_id"), "reason": "audio binding is validated by the protocol-specific runner"}
        entries = [item for item in manifest.get("manifests", []) if isinstance(item, Mapping)]
        if not entries:
            return {"status": "NOT_AVAILABLE", "reason": "audio manifest has no completed pair"}
        entry = entries[0]
        parameter_path = Path(str(entry.get("parameter_path", "")))
        if not parameter_path.is_file() or _sha256(parameter_path) != str(entry.get("parameter_sha256", "")):
            raise RecomputeError("parameter binding hash is invalid")
        parameter_meta_path = parameter_path.with_suffix(".json")
        parameter_meta = _verify_self_hash(parameter_meta_path) if parameter_meta_path.is_file() else None
        with np.load(parameter_path, allow_pickle=False) as arrays:
            checks: dict[str, Any] = {}
            for receiver in ("N", "T"):
                prefix = receiver
                f0 = np.asarray(arrays[f"{prefix}_f0"], dtype=np.float64)
                sp = np.asarray(arrays[f"{prefix}_sp"], dtype=np.float64)
                ap = np.asarray(arrays[f"{prefix}_ap"], dtype=np.float64)
                weight = np.asarray(arrays[f"{prefix}_weight"], dtype=np.float64)
                valid = np.asarray(arrays[f"{prefix}_valid"], dtype=bool)
                donor_time = np.asarray(arrays[f"{prefix}_donor_time"], dtype=np.float64)
                phone_index = np.asarray(arrays[f"{prefix}_phone_index"], dtype=np.int64)
                donor_left = np.asarray(arrays[f"{prefix}_donor_left"], dtype=np.int64)
                donor_right = np.asarray(arrays[f"{prefix}_donor_right"], dtype=np.int64)
                donor_alpha = np.asarray(arrays[f"{prefix}_donor_alpha"], dtype=np.float64)
                target = np.asarray(arrays[f"{prefix}_target_CONTOUR"], dtype=np.float64)
                if f0.ndim != 1 or sp.ndim != 2 or ap.shape != sp.shape or sp.shape[0] != f0.size or any(value.shape != f0.shape for value in (weight, valid, donor_time, phone_index, donor_left, donor_right, donor_alpha, target)):
                    raise RecomputeError(f"{receiver} parameter shapes are inconsistent")
                if not np.isfinite(f0).all() or not np.isfinite(sp).all() or not np.isfinite(ap).all() or not np.isfinite(weight).all() or not np.isfinite(target).all():
                    raise RecomputeError(f"{receiver} parameter arrays contain non-finite values")
                support = valid & (weight > 0.0) & (f0 > 0.0) & (target > 0.0)
                if not np.any(support):
                    raise RecomputeError(f"{receiver} has no valid F0 support")
                if np.any(valid & (~np.isfinite(donor_time) | (phone_index < 0) | (donor_left < 0) | (donor_right < 0) | (donor_alpha < 0.0) | (donor_alpha > 1.0))):
                    raise RecomputeError(f"{receiver} mapping arrays violate valid-frame bounds")
                if parameter_meta is not None:
                    meta_receiver = parameter_meta.get("receivers", {}).get(receiver, {})
                    reasons = meta_receiver.get("mapping_invalid_reasons", [])
                    if not isinstance(reasons, list) or len(reasons) != f0.size:
                        raise RecomputeError(f"{receiver} mapping invalid-reason ledger is incomplete")
                delta = 12.0 * np.log2(target[support]) - 12.0 * np.log2(f0[support])
                identity_error = float(np.sum(weight[support] * delta) / np.sum(weight[support]))
                checks[receiver] = {"frame_count": int(f0.size), "sp_shape": list(sp.shape), "ap_shape": list(ap.shape), "valid_frame_count": int(np.sum(valid)), "mapping_ledger_frame_count": int(len(parameter_meta.get("receivers", {}).get(receiver, {}).get("mapping_invalid_reasons", []))) if parameter_meta is not None else None, "weighted_contour_identity_error_st": identity_error, "identity_pass": abs(identity_error) <= 1e-8}
                if abs(identity_error) > 1e-8:
                    raise RecomputeError(f"{receiver} contour identity failed in parameter file")
            return {"status": "PASS", "paired_key": entry.get("paired_key"), "parameter_path": str(parameter_path.resolve()), "receivers": checks}
    except (OSError, ValueError, KeyError, RecomputeError) as exc:
        return {"status": "FAIL", "error": str(exc)}


def _repair_read_pcm16(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16_000:
            raise RecomputeError(f"repair WAV contract mismatch: {path}")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _repair_envelope_delta(candidate: np.ndarray, identity: np.ndarray) -> tuple[float | None, float | None, bool]:
    frame, hop = round(0.020 * 16_000), round(0.010 * 16_000)
    count = max(0, 1 + (min(candidate.size, identity.size) - frame) // hop)
    if count <= 0:
        return None, None, False
    cand = np.asarray([np.sqrt(np.mean(np.square(candidate[i * hop:i * hop + frame]))) for i in range(count)], dtype=np.float64)
    ident = np.asarray([np.sqrt(np.mean(np.square(identity[i * hop:i * hop + frame]))) for i in range(count)], dtype=np.float64)
    support = ident > np.max(ident) * 10.0 ** (-40.0 / 20.0)
    if not np.any(support):
        return None, None, False
    delta = np.abs(20.0 * np.log10(np.maximum(cand[support], 1e-12) / np.maximum(ident[support], 1e-12)))
    median, p90 = float(np.median(delta)), float(np.quantile(delta, 0.90, method="linear"))
    return median, p90, bool(median <= 1.0 and p90 <= 3.0)


def _repair_measurement(arrays: Mapping[str, np.ndarray], prefix: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    f0_key, time_key = f"{prefix}_{name}_f0", f"{prefix}_{name}_time"
    if f0_key not in arrays or time_key not in arrays:
        raise RecomputeError(f"missing repair measurement arrays: {prefix}/{name}")
    f0, time = np.asarray(arrays[f0_key], dtype=np.float64), np.asarray(arrays[time_key], dtype=np.float64)
    if f0.ndim != 1 or time.shape != f0.shape or not np.isfinite(f0).all() or not np.isfinite(time).all() or np.any(f0 < 0.0):
        raise RecomputeError(f"invalid repair measurement arrays: {prefix}/{name}")
    return f0, time


def _repair_pair_receiver(run_dir: Path, result: Mapping[str, Any], receiver: str) -> dict[str, Any]:
    pair = str(result.get("paired_key"))
    parameter_path = Path(str(result.get("parameter_path", "")))
    if not parameter_path.is_file() or _sha256(parameter_path) != str(result.get("parameter_sha256", "")):
        raise RecomputeError(f"repair parameter hash mismatch: {pair}/{receiver}")
    with np.load(parameter_path, allow_pickle=False) as arrays:
        prefix = receiver
        source = np.asarray(arrays[f"{prefix}_f0"], dtype=np.float64)
        source_time = np.asarray(arrays[f"{prefix}_time"], dtype=np.float64)
        weight = np.asarray(arrays[f"{prefix}_weight"], dtype=np.float64)
        valid = np.asarray(arrays[f"{prefix}_valid"], dtype=bool)
        if source.ndim != 1 or source_time.shape != source.shape or weight.shape != source.shape or valid.shape != source.shape:
            raise RecomputeError(f"repair source/mapping shape mismatch: {pair}/{receiver}")
        if not np.isfinite(source).all() or not np.isfinite(source_time).all() or not np.isfinite(weight).all():
            raise RecomputeError(f"repair source/mapping contains non-finite values: {pair}/{receiver}")
        voiced = source > 0.0
        support = valid & (weight > 0.0)
        strong = valid & (weight >= 0.5)
        if not np.any(support) or not np.any(strong):
            raise RecomputeError(f"repair support is empty: {pair}/{receiver}")
        target_level = np.asarray(arrays[f"{prefix}_target_LEVEL"], dtype=np.float64)
        target_contour = np.asarray(arrays[f"{prefix}_target_CONTOUR"], dtype=np.float64)
        if target_level.shape != source.shape or target_contour.shape != source.shape or not np.isfinite(target_level).all() or not np.isfinite(target_contour).all():
            raise RecomputeError(f"repair target arrays invalid: {pair}/{receiver}")
        contour_delta = np.zeros_like(source)
        level_delta = np.zeros_like(source)
        contour_delta[support] = 12.0 * np.log2(target_contour[support] / source[support])
        level_delta[support] = 12.0 * np.log2(target_level[support] / source[support])
        ordinary_error = float(np.mean(contour_delta[voiced])) if np.any(voiced) else float("nan")
        weighted_error = float(np.sum(weight[support] * contour_delta[support]) / np.sum(weight[support]))
        dose = float(np.sqrt(np.mean(np.square(contour_delta[strong]))))
        measurements = {name: _repair_measurement(arrays, prefix, name) for name in REPAIR_MEASUREMENTS}
        for name, (f0, time) in measurements.items():
            if time.shape != source_time.shape or not np.allclose(time, source_time, atol=1e-8, rtol=0.0):
                raise RecomputeError(f"repair TIME_GRID_MISMATCH: {pair}/{receiver}/{name}")
        h_raw, h_id = measurements["H_RAW"][0], measurements["H_ID"][0]
        d_raw, d_id = measurements["D_RAW"][0], measurements["D_ID"][0]
        if not np.array_equal(target_contour[~support], source[~support]) or not np.array_equal(target_level[~support], source[~support]):
            raise RecomputeError(f"repair M-outside target changed: {pair}/{receiver}")
        original_mask = source > 0.0
        d_id_mask = d_id > 0.0
        id_overlap = d_id_mask & original_mask
        id_stats = np.abs(12.0 * np.log2(np.maximum(d_id[id_overlap], 1e-12)) - 12.0 * np.log2(np.maximum(source[id_overlap], 1e-12))) if np.any(id_overlap) else np.asarray([], dtype=np.float64)
        candidate_rows: dict[str, Any] = {}
        for arm in ("LEVEL", "CONTOUR"):
            candidate = measurements[f"D_{arm}"][0]
            candidate_mask = candidate > 0.0
            compared = strong & candidate_mask & d_id_mask
            expected_f0 = target_level if arm == "LEVEL" else target_contour
            expected_delta = 12.0 * np.log2(np.maximum(expected_f0, 1e-12) / np.maximum(source, 1e-12))
            err = np.abs((12.0 * np.log2(np.maximum(candidate[compared], 1e-12)) - 12.0 * np.log2(np.maximum(d_id[compared], 1e-12))) - expected_delta[compared]) if np.any(compared) else np.asarray([], dtype=np.float64)
            candidate_rows[arm] = {"mask_mismatch": float(np.mean(candidate_mask != d_id_mask)), "support_coverage": float(np.sum(compared) / np.sum(strong)), "error_median_st": float(np.median(err)) if err.size else None, "error_p90_st": float(np.quantile(err, 0.90, method="linear")) if err.size else None}
        four = {"both_raw_and_id_voiced": int(np.sum((h_raw > 0.0) & (d_raw > 0.0) & (d_id > 0.0))), "only_raw_voiced": int(np.sum((h_raw > 0.0) & (d_raw > 0.0) & ~(d_id > 0.0))), "only_id_voiced": int(np.sum((h_raw > 0.0) & ~(d_raw > 0.0) & (d_id > 0.0))), "both_unvoiced": int(np.sum((h_raw > 0.0) & ~(d_raw > 0.0) & ~(d_id > 0.0)))}
        wav_stats: dict[str, Any] = {}
        for arm in REPAIR_ARMS:
            meta = result.get("audio", {}).get(receiver, {}).get(arm, {})
            path = Path(str(meta.get("path", "")))
            values = _repair_read_pcm16(path)
            wav_stats[arm] = {"sample_count": int(values.size), "peak": float(np.max(np.abs(values), initial=0.0)), "rms": float(np.sqrt(np.mean(np.square(values)))), "values": values}
        reasons: list[str] = []
        effective_coverage = float(np.sum(weight) / np.sum(voiced)) if np.any(voiced) else 0.0
        phone_index = np.asarray(arrays[f"{prefix}_phone_index"], dtype=np.int64)
        effective_phone_count = int(len(set(int(value) for value in phone_index[support])))
        if effective_coverage < 0.40 or effective_phone_count < 5 or np.sum(voiced) * 0.005 < 1.0:
            reasons.append("INSUFFICIENT_SUPPORT")
        if not np.all(target_level[~support] == source[~support]) or not np.all(target_contour[~support] == source[~support]):
            reasons.append("M_OUTSIDE_CHANGED")
        if not np.array_equal(target_level > 0.0, original_mask) or not np.array_equal(target_contour > 0.0, original_mask):
            reasons.append("TARGET_VOICED_MASK_CHANGED")
        if abs(ordinary_error) > 1e-8:
            reasons.append("CONTOUR_MEAN_IDENTITY_FAIL")
        id_mismatch = float(np.mean(d_id_mask != original_mask))
        id_coverage = float(np.sum(id_overlap) / np.sum(original_mask)) if np.any(original_mask) else 0.0
        if id_mismatch > 0.10:
            reasons.append("ID_VOICED_MASK_MISMATCH")
        if id_coverage < 0.80:
            reasons.append("ID_ORIGINAL_VOICED_COVERAGE")
        if not id_stats.size or float(np.median(id_stats)) > 1.0 or float(np.quantile(id_stats, 0.90, method="linear")) > 3.0:
            reasons.append("ID_F0_ERROR")
        for arm, row in candidate_rows.items():
            if row["mask_mismatch"] > 0.05:
                reasons.append(f"{arm}_VOICED_MASK_MISMATCH")
            if row["support_coverage"] < 0.80:
                reasons.append(f"{arm}_F0_MEASURE_COVERAGE")
            if row["error_median_st"] is None or row["error_median_st"] > 1.0 or row["error_p90_st"] > 3.0:
                reasons.append(f"{arm}_F0_ERROR")
            median_env, p90_env, env_pass = _repair_envelope_delta(wav_stats[arm]["values"], wav_stats["ID"]["values"])
            row["envelope_median_db"], row["envelope_p90_db"], row["envelope_pass"] = median_env, p90_env, env_pass
            if not env_pass:
                reasons.append("ENVELOPE_CONFOUND")
        raw_rms, id_rms = wav_stats["RAW"]["rms"], wav_stats["ID"]["rms"]
        for arm in ("ID", "LEVEL", "CONTOUR"):
            for baseline_name, baseline in (("RAW", raw_rms), ("ID", id_rms)):
                if baseline <= 0.0 or wav_stats[arm]["rms"] <= 0.0 or abs(20.0 * np.log10(wav_stats[arm]["rms"] / baseline)) > 0.1 + 1e-9:
                    reasons.append(f"RMS_MISMATCH_{baseline_name}")
        for arm in REPAIR_ARMS:
            if wav_stats[arm]["sample_count"] < 16_000 or wav_stats[arm]["peak"] >= 1.0:
                reasons.append(f"{arm}_WAV_INVALID")
        return {"paired_key": pair, "receiver": receiver, "status": "PASS" if not reasons else "QC_FAIL", "reasons": sorted(set(reasons)), "effective_coverage": effective_coverage, "effective_phone_count": effective_phone_count, "ordinary_identity_error_st": ordinary_error, "weighted_identity_error_st": weighted_error, "dose_contour_rms_st": dose, "id_mask_mismatch": id_mismatch, "id_coverage": id_coverage, "id_error_median_st": float(np.median(id_stats)) if id_stats.size else None, "id_error_p90_st": float(np.quantile(id_stats, 0.90, method="linear")) if id_stats.size else None, "candidate": candidate_rows, "four_cells": four, "wav": {arm: {key: value for key, value in stats.items() if key != "values"} for arm, stats in wav_stats.items()}}


def recompute_repair(run_dir: Path) -> dict[str, Any]:
    protocol = _verify_self_hash(run_dir / "protocol.json")
    inputs = _verify_self_hash(run_dir / "inputs.json")
    audio = _verify_self_hash(run_dir / "audio_manifest.json")
    if protocol.get("protocol_id") != REPAIR_PROTOCOL_ID or inputs.get("protocol_id") != REPAIR_PROTOCOL_ID or audio.get("protocol_id") != REPAIR_PROTOCOL_ID:
        raise RecomputeError("repair protocol id mismatch")
    keys = [str(value) for value in inputs.get("fixed_keys", [])]
    if len(keys) != 4 or len(set(keys)) != 4:
        raise RecomputeError("repair fixed key denominator is not four")
    results = {str(item.get("paired_key")): item for item in audio.get("manifests", []) if isinstance(item, Mapping)}
    receiver_rows: list[dict[str, Any]] = []
    for key in keys:
        if key not in results:
            for receiver in REPAIR_REQUIRED_KEYS:
                receiver_rows.append({"paired_key": key, "receiver": receiver, "status": "QC_FAIL", "reasons": ["PAIR_MANIFEST_MISSING"], "ordinary_identity_error_st": None, "weighted_identity_error_st": None, "dose_contour_rms_st": None, "id_mask_mismatch": None, "id_coverage": None, "id_error_median_st": None, "id_error_p90_st": None, "candidate": {}, "four_cells": {}})
            continue
        for receiver in REPAIR_REQUIRED_KEYS:
            receiver_rows.append(_repair_pair_receiver(run_dir, results[key], receiver))
    if len(receiver_rows) != 8:
        raise RecomputeError("repair receiver denominator is incomplete")
    ordinary = [row["ordinary_identity_error_st"] for row in receiver_rows if row["ordinary_identity_error_st"] is not None]
    pair_status: dict[str, bool] = {}
    independent_qc = {(str(row.get("paired_key")), str(row.get("receiver"))): row for row in receiver_rows}
    for key in keys:
        pair_status[key] = all(independent_qc.get((key, receiver), {}).get("status") == "PASS" for receiver in REPAIR_REQUIRED_KEYS)
    passed = int(sum(pair_status.values()))
    gate_status = "READY_FOR_FORMAL_AUDIO" if passed >= 3 else "MANIPULATION_NOT_VALIDATED"
    return {"schema_version": 1, "status": "PASS", "protocol_id": REPAIR_PROTOCOL_ID, "run_dir": str(run_dir.resolve()), "fixed_pair_count": 4, "completed_pair_count": len(results), "receiver_count": len(receiver_rows), "ordinary_identity_max_abs_st": float(max(abs(value) for value in ordinary)) if ordinary else None, "ordinary_identity_all_pass": len(ordinary) == 8 and max(abs(value) for value in ordinary) <= 1e-8, "pair_status": pair_status, "passed_pairs": passed, "required_pairs": 3, "gate_status": gate_status, "receiver_rows": receiver_rows, "independent_implementation": "stdlib+numpy; no import of tts_f0_swap.py"}


def recompute(run_dir: Path) -> dict[str, Any]:
    inputs_path = run_dir / "inputs.json"
    score_path = run_dir / "scores_manifest.json"
    analysis_path = run_dir / "analysis.json"
    for path in (inputs_path, score_path, analysis_path):
        if not path.is_file():
            raise RecomputeError(f"missing required artifact: {path}")
    inputs = _verify_self_hash(inputs_path)
    scores = _verify_self_hash(score_path)
    analysis = _verify_self_hash(analysis_path)
    if scores.get("status") != "complete":
        raise RecomputeError(f"score manifest is not complete: {scores.get('status')}")
    if analysis.get("status") != "complete":
        raise RecomputeError(f"analysis is not complete: {analysis.get('status')}")
    rows = scores.get("rows")
    if not isinstance(rows, list) or int(scores.get("row_count", -1)) != len(rows):
        raise RecomputeError("score row count is inconsistent")
    index = _row_index(rows)
    speakers_by_pair = {
        str(row["paired_key"]): str(row.get("speaker_id", ""))
        for row in inputs.get("records", []) if row.get("role") == "formal"
    }
    if len(speakers_by_pair) != int(inputs.get("formal_pair_count", len(speakers_by_pair))):
        raise RecomputeError("formal input pair count is inconsistent")
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pair in sorted(speakers_by_pair):
        try:
            records.append(_record_contrasts(pair, speakers_by_pair[pair], index))
        except RecomputeError as exc:
            missing.append({"paired_key": pair, "speaker_id": speakers_by_pair[pair], "missing_reason": str(exc)})
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[str(row["speaker_id"])] += 1
    main = [row for row in records if counts[str(row["speaker_id"])] >= 2]
    labels = sorted({str(row["speaker_id"]) for row in main})
    bootstrap_indices = _indices(len(labels)) if labels else np.empty((0, 0), dtype=np.int64)
    statistics: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row["G_id"] - row["G_raw"]) if metric == "G_id_minus_G_raw" else float(row[metric]) for row in main]
        statistics[metric] = _bootstrap(values, [str(row["speaker_id"]) for row in main], bootstrap_indices) if labels else {"status": "NOT_ESTIMABLE", "reason": "no included speakers"}

    production_stats = analysis.get("statistics", {})
    comparisons: dict[str, Any] = {}
    mismatch = False
    for metric in METRICS:
        expected = production_stats.get(metric, {})
        actual = statistics[metric]
        fields = ("mean", "ci95", "ci98_75_bonferroni", "speaker_count", "pair_count", "positive_speaker_count", "status_by_corrected_ci")
        checks: dict[str, bool] = {}
        for field in fields:
            left, right = actual.get(field), expected.get(field)
            if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
                checks[field] = all(_same_number(a, b) for a, b in zip(left, right, strict=True))
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                checks[field] = _same_number(left, right)
            else:
                checks[field] = left == right
            mismatch = mismatch or not checks[field]
        comparisons[metric] = {"match": all(checks.values()), "checks": checks}

    prod_records = {str(row.get("paired_key")): row for row in analysis.get("records", []) if isinstance(row, Mapping) and "g_N" in row}
    record_checks: dict[str, bool] = {}
    for row in main:
        expected = prod_records.get(str(row["paired_key"]))
        ok = expected is not None and all(_same_number(row[name], expected.get(name)) for name in METRICS if name != "G_id_minus_G_raw")
        record_checks[str(row["paired_key"])] = ok
        mismatch = mismatch or not ok
    production_indices = analysis.get("bootstrap", {}).get("indices")
    index_hash_match = False
    if isinstance(production_indices, str) and Path(production_indices).is_file() and labels:
        loaded = np.asarray(np.load(production_indices, allow_pickle=False), dtype=np.int64)
        index_hash_match = loaded.shape == bootstrap_indices.shape and np.array_equal(loaded, bootstrap_indices)
        mismatch = mismatch or not index_hash_match

    audio_binding = _audio_binding_check(run_dir)
    if (run_dir / "audio_manifest.json").is_file():
        audio_manifest = _read_json(run_dir / "audio_manifest.json")
        if audio_manifest.get("status") == "complete" and audio_binding.get("status") not in {"PASS", "NOT_APPLICABLE"}:
            mismatch = True

    payload = {
        "schema_version": 1,
        "status": "PASS" if not mismatch else "FAIL",
        "protocol_id": str(analysis.get("protocol_id", "")),
        "run_dir": str(run_dir.resolve()),
        "production_analysis_sha256": _sha256(analysis_path),
        "production_score_manifest_sha256": _sha256(score_path),
        "formal_pair_count": len(speakers_by_pair),
        "recomputed_pair_count": len(records),
        "missing_pairs": missing,
        "main_pair_count": len(main),
        "included_speakers": labels,
        "statistics": statistics,
        "comparisons": comparisons,
        "record_checks": record_checks,
        "bootstrap_indices_match": index_hash_match,
        "audio_binding_check": audio_binding,
        "independent_implementation": "stdlib+numpy; no import of tts_f0_swap.py",
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repair-run-id")
    parser.add_argument("--repair-run-dir", type=Path)
    args = parser.parse_args(argv)
    repair_mode = bool(args.repair_run_id) or bool(args.repair_run_dir)
    if repair_mode:
        if bool(args.repair_run_id) == bool(args.repair_run_dir) or args.run_id or args.run_dir:
            parser.error("provide exactly one of --repair-run-id/--repair-run-dir")
    elif bool(args.run_id) == bool(args.run_dir):
        parser.error("provide exactly one of --run-id or --run-dir")
    repo = Path(__file__).resolve().parents[2]
    if repair_mode:
        run_dir = args.repair_run_dir.resolve() if args.repair_run_dir else (repo / "runs" / f"tts_f0_swap_repair_{args.repair_run_id}").resolve()
    else:
        run_dir = args.run_dir.resolve() if args.run_dir else (repo / "runs" / f"tts_f0_swap_{args.run_id}").resolve()
    try:
        output = recompute_repair(run_dir) if repair_mode else recompute(run_dir)
    except RecomputeError as exc:
        message = str(exc)
        unavailable = message.startswith("missing required artifact") or "is not complete" in message
        output = {"schema_version": 1, "status": "NOT_AVAILABLE" if unavailable else "FAIL", "protocol_id": REPAIR_PROTOCOL_ID if repair_mode else "tts_f0_swap_v1", "run_dir": str(run_dir), "reason": message, "independent_implementation": "stdlib+numpy; no import of tts_f0_swap.py"}
    _write_json(run_dir / "recompute.json", output)
    print(json.dumps({"status": output["status"], "output": str((run_dir / "recompute.json").resolve())}, ensure_ascii=False))
    return 0 if output["status"] in {"PASS", "NOT_AVAILABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
