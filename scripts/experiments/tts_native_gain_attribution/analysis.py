"""Pre-registered analysis for A, B, geometry, content, and rhythm evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    canonical_sha256,
    csv_write,
    file_sha256,
    read_self_hashed_json,
    write_json,
    write_self_hashed_json,
)


def _quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="linear"))
    except TypeError:  # pragma: no cover - compatibility with old NumPy
        return float(np.quantile(values, probability, interpolation="linear"))


def bootstrap_indices(group_count: int = len(config.SAMPLE_IDS)) -> np.ndarray:
    if group_count != len(config.SAMPLE_IDS):
        raise ProtocolError("the registered attribution bootstrap has exactly 12 source-group columns")
    return np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED)).integers(0, group_count, size=(config.BOOTSTRAP_DRAWS, group_count), dtype=np.int64)


def curve_metrics(matrix: np.ndarray, support: Sequence[int], *, label: str = "matrix") -> dict[str, Any]:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[1] != config.LAG_COUNT or value.shape[0] < 1 or not np.isfinite(value).all():
        raise ProtocolError(f"{label} must be finite [T,31], got {value.shape}")
    rows = np.asarray(list(support), dtype=np.int64)
    if rows.ndim != 1 or rows.size < 1 or np.any(rows < 0) or np.any(rows >= value.shape[0]) or np.any(np.diff(rows) <= 0):
        raise ProtocolError(f"{label} support is invalid")
    # Official arithmetic: float32 distances, time mean first, then lag
    # statistics.  Saving as float64 later does not change this reduction.
    curve = value[rows].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    min_index = int(np.argmin(curve))
    d = float(curve[min_index])
    b = float(np.median(curve))
    c = b - d
    return {
        "support_count": int(rows.size),
        "support_rows": [int(item) for item in rows],
        "curve": [float(item) for item in curve],
        "offsets": [int(config.VSHIFT - index) for index in range(config.LAG_COUNT)],
        "min_index": min_index,
        "official_offset": int(config.VSHIFT - min_index),
        "sync_d": d,
        "background_b": b,
        "sync_c": c,
        "d0": float(curve[config.VSHIFT]),
        "search_gain": float(curve[config.VSHIFT] - d),
        "boundary_best": bool(min_index in (0, config.LAG_COUNT - 1)),
        "ties": [int(index) for index, item in enumerate(curve) if item == d],
    }


def common_support(matrices: Sequence[np.ndarray], *, trim: int = config.VSHIFT) -> tuple[list[int], str]:
    values = [np.asarray(item) for item in matrices]
    if not values or any(item.ndim != 2 or item.shape[1] != config.LAG_COUNT for item in values):
        raise ProtocolError("cannot construct common support from invalid matrices")
    count = min(int(item.shape[0]) for item in values)
    rows = list(range(trim, count - trim))
    return rows, canonical_sha256({"rows": rows, "trim": trim, "rule": "common valid time-index intersection"})


def four_cell(q00: float, q01: float, q10: float, q11: float, *, metric: str = "C") -> dict[str, float]:
    values = [float(q00), float(q01), float(q10), float(q11)]
    if not np.isfinite(values).all():
        raise ProtocolError("four-cell values must be finite")
    generation = float(q10 - q00)
    evaluation = float(q01 - q00)
    interaction = float(q11 - q10 - q01 + q00)
    total = float(q11 - q00)
    if abs(generation + evaluation + interaction - total) > 1e-10:
        raise ProtocolError("four-cell decomposition identity failed")
    return {"q00": float(q00), "q01": float(q01), "q10": float(q10), "q11": float(q11), "metric": metric, "generation": generation, "evaluation": evaluation, "interaction": interaction, "total": total, "generation_symmetric": float(0.5 * ((q10 - q00) + (q11 - q01))), "evaluation_symmetric": float(0.5 * ((q01 - q00) + (q11 - q10)))}


def _summary(values_by_group: Mapping[str, float], indices: np.ndarray, *, metric: str) -> dict[str, Any]:
    labels = sorted((str(label) for label in values_by_group), key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    if len(labels) != len(config.SAMPLE_IDS):
        return {"status": "INCOMPLETE", "metric": metric, "group_count": len(labels), "expected_group_count": len(config.SAMPLE_IDS)}
    values = np.asarray([float(values_by_group[label]) for label in labels], dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.shape != (config.BOOTSTRAP_DRAWS, len(config.SAMPLE_IDS)):
        raise ProtocolError(f"bootstrap index shape mismatch for {metric}: {idx.shape}")
    sampled = values[idx].mean(axis=1, dtype=np.float64)
    tail = 0.05 / (2.0 * config.PRIMARY_FAMILY_SIZE)
    return {
        "status": "COMPLETE",
        "metric": metric,
        "mean": float(values.mean(dtype=np.float64)),
        "ci95": [_quantile(sampled, 0.025), _quantile(sampled, 0.975)],
        "ci99_166667_bonferroni": [_quantile(sampled, tail), _quantile(sampled, 1.0 - tail)],
        "ci_bonferroni": [_quantile(sampled, tail), _quantile(sampled, 1.0 - tail)],
        "bonferroni_comparisons": config.PRIMARY_FAMILY_SIZE,
        "bonferroni_tail_probability": tail,
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, values, strict=True)},
        "group_count": len(labels),
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": "linear",
        "positive_evidence": bool(_quantile(sampled, tail) > 0.0),
        "reverse_evidence": bool(_quantile(sampled, 1.0 - tail) < 0.0),
        "practical_threshold_diagnostic": bool(float(values.mean()) >= config.PRACTICAL_THRESHOLD),
        "equivalent_within_0.2": bool(_quantile(sampled, tail) >= -config.PRACTICAL_THRESHOLD and _quantile(sampled, 1.0 - tail) <= config.PRACTICAL_THRESHOLD),
    }


def _effect_status(summary: Mapping[str, Any]) -> str:
    if summary.get("status") != "COMPLETE":
        return "INCOMPLETE"
    lower, upper = [float(item) for item in summary["ci99_166667_bonferroni"]]
    if lower > 0.0:
        return "POSITIVE_EVIDENCE"
    if upper < 0.0:
        return "NEGATIVE_DIRECTION_EVIDENCE"
    if lower >= -config.PRACTICAL_THRESHOLD and upper <= config.PRACTICAL_THRESHOLD:
        return "SMALL_WITHIN_PREDECLARED_RANGE"
    return "INCONCLUSIVE"


def _load_matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix_path", "")))
    if not path.is_file():
        raise ProtocolError(f"matrix artifact is missing: {path}")
    actual = file_sha256(path)
    if str(row.get("matrix_hash")) != actual:
        raise ProtocolError(f"matrix hash mismatch: {path}")
    matrix = np.load(path, allow_pickle=False)
    shape = row.get("matrix_shape")
    if isinstance(shape, list) and [int(item) for item in matrix.shape] != [int(item) for item in shape]:
        raise ProtocolError(f"matrix shape metadata mismatch: {path}")
    return matrix


def _group_values(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(float(row[metric]))
    return {group: float(np.mean(values, dtype=np.float64)) for group, values in grouped.items()}


def _dz(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return None
    sd = float(np.std(array, ddof=1))
    return None if sd == 0.0 else float(array.mean() / sd)


def _read_feature(path: str | Path) -> np.ndarray:
    target = Path(path)
    if not target.is_file():
        raise ProtocolError(f"feature artifact is missing: {target}")
    value = np.load(target, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != config.EMBEDDING_DIM or not np.isfinite(value).all():
        raise ProtocolError(f"feature artifact is invalid: {target}")
    return np.asarray(value, dtype=np.float32)


def _feature_diagnostics(feature: np.ndarray) -> dict[str, Any]:
    value = np.asarray(feature, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= 0.0):
        raise ProtocolError("zero feature norm in geometry diagnostic")
    adjacent = np.linalg.norm(np.diff(value, axis=0), axis=1) if value.shape[0] > 1 else np.asarray([], dtype=np.float32)
    normalized = value / norms[:, None]
    normalized_adjacent = np.linalg.norm(np.diff(normalized, axis=0), axis=1) if value.shape[0] > 1 else np.asarray([], dtype=np.float32)
    return {
        "count": int(value.shape[0]),
        "dimension": int(value.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_sd": float(norms.std(ddof=1)) if norms.size > 1 else None,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "adjacent_l2_mean": float(adjacent.mean()) if adjacent.size else None,
        "adjacent_unit_l2_mean": float(normalized_adjacent.mean()) if normalized_adjacent.size else None,
        "finite": True,
    }


def _pair_geometry(video: np.ndarray, audio: np.ndarray, support: Sequence[int]) -> dict[str, Any]:
    v = np.asarray(video, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    rows = np.asarray(list(support), dtype=np.int64)
    count = min(v.shape[0], a.shape[0])
    rows = rows[rows < count]
    if rows.size < 1:
        raise ProtocolError("geometry support has no valid rows")
    v = v[rows]
    a = a[rows]
    diff = v - a
    dot = np.sum(v * a, axis=1, dtype=np.float64)
    squared = np.sum(diff * diff, axis=1, dtype=np.float64)
    vn = np.linalg.norm(v, axis=1)
    an = np.linalg.norm(a, axis=1)
    unit = v / vn[:, None], a / an[:, None]
    unit_squared = np.sum((unit[0] - unit[1]) ** 2, axis=1, dtype=np.float64)
    return {"support_count": int(rows.size), "video_norm_mean": float(vn.mean()), "audio_norm_mean": float(an.mean()), "dot_mean": float(dot.mean()), "squared_distance_mean": float(squared.mean()), "unit_squared_distance_mean": float(unit_squared.mean()), "unit_norms_checked": True}


def _primary_record(rows: Sequence[Mapping[str, Any]], *, sample_id: int, video_type: str, first: str, second: str, direction: int, name: str) -> dict[str, Any] | None:
    selected = [row for row in rows if int(row["id"]) == sample_id and str(row["video_type"]) == video_type]
    by_condition = {str(row["eval_condition"]): row for row in selected}
    if first not in by_condition or second not in by_condition:
        return None
    left, right = by_condition[first], by_condition[second]
    left_metrics, right_metrics = left["metrics"], right["metrics"]
    return {"id": sample_id, "source_group": str(left["source_group"]), "metric": name, "value": float(direction * (float(right_metrics["sync_c"]) - float(left_metrics["sync_c"]))), "left_condition": first, "right_condition": second}


def _fixed_video_analysis(a_manifest: Mapping[str, Any], feature_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    by_family: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for raw in a_manifest.get("cells", []):
        row = dict(raw)
        matrix = _load_matrix(row)
        support = [int(item) for item in row.get("support_rows", [])]
        metrics = curve_metrics(matrix, support, label=f"A/{row.get('id')}/{row.get('video_type')}/{row.get('eval_condition')}")
        row["metrics"] = metrics
        raw_rows.append(row)
        by_family[(int(row["id"]), str(row["video_type"]))].append(matrix)
    # Independent support check: every condition in a family must use the
    # same registered support hash and row indices.
    support_failures = []
    for key in {(int(row["id"]), str(row["video_type"])) for row in raw_rows}:
        family = [row for row in raw_rows if (int(row["id"]), str(row["video_type"])) == key]
        hashes = {str(row.get("support_hash")) for row in family}
        supports = {tuple(int(item) for item in row.get("support_rows", [])) for row in family}
        if len(hashes) != 1 or len(supports) != 1:
            support_failures.append({"family": list(key), "support_hashes": sorted(hashes), "support_sets": [list(item) for item in supports]})
    return {"status": "COMPLETE" if len(raw_rows) == config.EXPECTED_A_SCIENCE and not support_failures else "INCOMPLETE", "cell_count": len(raw_rows), "expected_cell_count": config.EXPECTED_A_SCIENCE, "support_failures": support_failures, "cells": raw_rows}, raw_rows


def _primary_analysis(a_rows: Sequence[Mapping[str, Any]], crossed: Mapping[str, Any] | None, indices: np.ndarray) -> dict[str, Any]:
    specifications = (
        ("A_N_DENOISE_E", "V_N", "A0", "DENOISE", 1, "A fixed V_N: C(Ad)-C(A0)"),
        ("A_T_NOISE_E_HARM", "V_T", "NOISE", "A0", 1, "A fixed V_T: C(A0)-C(An)"),
        ("A_R_DENOISE_E", "R", "A0", "DENOISE", 1, "A fixed R: C(Ad)-C(A0)"),
    )
    records: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for name, video_type, first, second, direction, label in specifications:
        values = []
        for sample_id in config.SAMPLE_IDS:
            row = _primary_record(a_rows, sample_id=sample_id, video_type=video_type, first=first, second=second, direction=direction, name=name)
            if row is not None:
                values.append(row)
        records[name] = values
        grouped = _group_values(values, "value") if values else {}
        summaries[name] = _summary(grouped, indices, metric=name) if len(grouped) == len(config.SAMPLE_IDS) else {"status": "INCOMPLETE", "metric": name, "group_count": len(grouped), "expected_group_count": len(config.SAMPLE_IDS)}
        summaries[name]["label"] = label
        summaries[name]["evidence_status"] = _effect_status(summaries[name])
    b_specs = (("B_N_DENOISE_G", "N", "DENOISE", 1, "B natural generation: q10_d-q00"), ("B_T_NOISE_G_HARM", "T", "NOISE", -1, "B TTS generation: q00-q10_n"), ("B_NATIVE_FRESH", "", "NATIVE", 1, "B fresh native baseline"))
    if crossed is None or crossed.get("status") != "COMPLETE":
        for name, source, driver, direction, label in b_specs:
            summaries[name] = {"status": "INCOMPLETE", "metric": name, "evidence_status": "INCOMPLETE", "label": label, "reason": "B crossed scoring is unavailable"}
            records[name] = []
    else:
        b_records = crossed.get("primary_records", {})
        for name, source, driver, direction, label in b_specs:
            values = []
            for sample_id in config.SAMPLE_IDS:
                if name == "B_NATIVE_FRESH":
                    item = b_records.get(f"{sample_id}:NATIVE")
                else:
                    item = b_records.get(f"{sample_id}:{source}:{driver}")
                if item is None:
                    continue
                value = float(item["value"]) * direction
                values.append({"id": sample_id, "source_group": str(item["source_group"]), "metric": name, "value": value})
            records[name] = values
            grouped = _group_values(values, "value") if values else {}
            summaries[name] = _summary(grouped, indices, metric=name) if len(grouped) == len(config.SAMPLE_IDS) else {"status": "INCOMPLETE", "metric": name, "group_count": len(grouped), "expected_group_count": len(config.SAMPLE_IDS)}
            summaries[name]["evidence_status"] = _effect_status(summaries[name])
            summaries[name]["label"] = label
    return {"specifications": [item[0] for item in specifications] + [item[0] for item in b_specs], "records": records, "summaries": summaries, "bonferroni_family_fixed_six": True, "positive_rule": "corrected lower bound > 0", "practical_threshold": config.PRACTICAL_THRESHOLD}


def _unit_effects(a_rows: Sequence[Mapping[str, Any]], feature_root: Path, indices: np.ndarray) -> dict[str, Any]:
    from .syncnet import SyncNetEngine

    unit_rows: list[dict[str, Any]] = []
    feature_diagnostics: list[dict[str, Any]] = []
    feature_cache: dict[str, np.ndarray] = {}
    for row in a_rows:
        video_type = str(row["video_type"])
        sample_id = int(row["id"])
        source = str(row["source"])
        condition = str(row["eval_condition"])
        visual_path = feature_root / "visual" / video_type / f"{sample_id}.npy"
        audio_path = feature_root / "audio" / source / str(sample_id) / f"{condition}.npy"
        # ORIGINAL is a logical cell that is intentionally allowed to reuse
        # A0's byte-identical audio feature.  The scorer records that reuse in
        # the A manifest instead of writing a second 1024-D array.
        if not audio_path.is_file() and condition == "ORIGINAL":
            audio_path = feature_root / "audio" / source / str(sample_id) / "A0.npy"
        if not visual_path.is_file() or not audio_path.is_file():
            continue
        v = feature_cache.setdefault(str(visual_path), _read_feature(visual_path))
        a = feature_cache.setdefault(str(audio_path), _read_feature(audio_path))
        support = [int(item) for item in row.get("support_rows", [])]
        matrix = SyncNetEngine.distance_matrix(SyncNetEngine.unit_features(v), SyncNetEngine.unit_features(a))
        metrics = curve_metrics(matrix, support, label=f"unit/{sample_id}/{video_type}/{condition}")
        unit_rows.append({**row, "unit_metrics": metrics})
    for path, feature in feature_cache.items():
        item = _feature_diagnostics(feature)
        item["path"] = path
        feature_diagnostics.append(item)
    specs = (("A_N_DENOISE_E", "V_N", "A0", "DENOISE", 1), ("A_T_NOISE_E_HARM", "V_T", "NOISE", "A0", 1), ("A_R_DENOISE_E", "R", "A0", "DENOISE", 1))
    summaries: dict[str, Any] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for name, video_type, left, right, direction in specs:
        rows = []
        for sample_id in config.SAMPLE_IDS:
            group = [row for row in unit_rows if int(row["id"]) == sample_id and row["video_type"] == video_type]
            by_condition = {row["eval_condition"]: row for row in group}
            if left in by_condition and right in by_condition:
                rows.append({"id": sample_id, "source_group": str(by_condition[left]["source_group"]), "value": direction * (by_condition[right]["unit_metrics"]["sync_c"] - by_condition[left]["unit_metrics"]["sync_c"]), "metric": name})
        records[name] = rows
        grouped = _group_values(rows, "value")
        summaries[name] = _summary(grouped, indices, metric=f"{name}.C_unit") if len(grouped) == len(config.SAMPLE_IDS) else {"status": "INCOMPLETE", "metric": f"{name}.C_unit"}
        summaries[name]["evidence_status"] = _effect_status(summaries[name])
    dz_comparison: dict[str, Any] = {}
    for name in ("A_N_DENOISE_E", "A_T_NOISE_E_HARM", "A_R_DENOISE_E"):
        raw = {int(row["id"]): float(row["value"]) for row in _primary_analysis(a_rows, None, indices)["records"].get(name, [])}
        unit = {int(row["id"]): float(row["value"]) for row in records.get(name, [])}
        common = sorted(set(raw) & set(unit))
        dz_raw = _dz([raw[item] for item in common])
        dz_unit = _dz([unit[item] for item in common])
        dz_comparison[name] = {"n": len(common), "dz_raw": dz_raw, "dz_unit": dz_unit, "dz_raw_minus_unit": None if dz_raw is None or dz_unit is None else dz_raw - dz_unit, "raw_positive_count": sum(raw[item] > 0 for item in common), "unit_positive_count": sum(unit[item] > 0 for item in common)}
    return {"status": "COMPLETE" if len(unit_rows) == len(a_rows) else "PARTIAL", "rows": unit_rows, "summaries": summaries, "feature_diagnostics": feature_diagnostics, "standardized_effects": dz_comparison, "metric_scale_warning": "C and C_unit are never subtracted as raw scores; only within-metric dz and direction descriptions are compared"}


def _content_retrieval(a_rows: Sequence[Mapping[str, Any]], feature_root: Path, assets: Mapping[str, Any]) -> dict[str, Any]:
    from .syncnet import SyncNetEngine

    video_types = config.VIDEO_TYPES
    source_for = {"V_N": "N", "V_T": "T", "R": "R"}
    records: list[dict[str, Any]] = []
    transcript_by_group = {str(row["source_group"]): ((row.get("transcript") or {}).get("text")) for row in assets.get("records", [])}
    transcript_status = {str(row["source_group"]): ((row.get("transcript") or {}).get("status")) for row in assets.get("records", [])}
    for video_type in video_types:
        source = source_for[video_type]
        feature_paths = []
        for sample_id in config.SAMPLE_IDS:
            feature_paths.append((sample_id, feature_root / "audio" / source / str(sample_id) / "A0.npy"))
        visual_cache = {sample_id: _read_feature(feature_root / "visual" / video_type / f"{sample_id}.npy") for sample_id in config.SAMPLE_IDS}
        audio_cache = {sample_id: _read_feature(path) for sample_id, path in feature_paths}
        row_counts = {sample_id: min(visual_cache[sample_id].shape[0], audio_cache[sample_id].shape[0]) for sample_id in config.SAMPLE_IDS}
        interior_counts = {sample_id: max(0, count - 2 * config.VSHIFT) for sample_id, count in row_counts.items()}
        k = min(interior_counts.values())
        if k < config.MIN_INTERIOR_ROWS:
            return {"status": "INCOMPLETE", "reason": f"content support below {config.MIN_INTERIOR_ROWS}", "pair_count": 0}
        rows_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for target_id in config.SAMPLE_IDS:
            target_v = visual_cache[target_id]
            for donor_id in config.SAMPLE_IDS:
                if donor_id == target_id:
                    continue
                donor_a = audio_cache[donor_id]
                matrix = SyncNetEngine.distance_matrix(target_v, donor_a)
                support = list(range(config.VSHIFT, matrix.shape[0] - config.VSHIFT))
                if len(support) < k:
                    continue
                indices = np.floor(np.linspace(0, len(support) - 1, k)).astype(np.int64)
                if len(set(indices.tolist())) != k:
                    raise ProtocolError("content quantile indices contain duplicates")
                selected_rows = [support[int(index)] for index in indices]
                distance = float(np.mean(matrix[np.asarray(selected_rows), config.VSHIFT], dtype=np.float64))
                target_group = next(str(item["source_group"]) for item in assets["records"] if int(item["sample_id"]) == target_id)
                donor_group = next(str(item["source_group"]) for item in assets["records"] if int(item["sample_id"]) == donor_id)
                text_equal = transcript_by_group.get(target_group) is not None and transcript_by_group.get(target_group) == transcript_by_group.get(donor_group)
                rows_by_video[target_id].append({"video_type": video_type, "target_id": target_id, "target_group": target_group, "donor_id": donor_id, "donor_group": donor_group, "distance_raw": distance, "text_equal": text_equal, "transcript_status_target": transcript_status.get(target_group), "transcript_status_donor": transcript_status.get(donor_group), "support_count": k})
            correct_matrix = SyncNetEngine.distance_matrix(target_v, audio_cache[target_id])
            correct_support = list(range(config.VSHIFT, correct_matrix.shape[0] - config.VSHIFT))
            indices = np.floor(np.linspace(0, len(correct_support) - 1, k)).astype(np.int64)
            selected = [correct_support[int(index)] for index in indices]
            correct_distance = float(np.mean(correct_matrix[np.asarray(selected), config.VSHIFT], dtype=np.float64))
            candidates = [row for row in rows_by_video[target_id] if not row["text_equal"]]
            if any(row["transcript_status_target"] != "TRANSCRIPT_VERIFIED" or row["transcript_status_donor"] != "TRANSCRIPT_VERIFIED" for row in rows_by_video[target_id]):
                content_status = "CONTENT_UNVERIFIED"
            else:
                content_status = "COMPLETE"
            distances = [float(row["distance_raw"]) for row in candidates]
            rank = 1 + sum(distance < correct_distance for distance in distances) + 0.5 * sum(distance == correct_distance for distance in distances)
            raw_beats = [1.0 if correct_distance < distance else 0.5 if correct_distance == distance else 0.0 for distance in distances]
            records.append({"video_type": video_type, "target_id": target_id, "target_group": next(str(item["source_group"]) for item in assets["records"] if int(item["sample_id"]) == target_id), "correct_distance_raw": correct_distance, "wrong_donor_count": len(candidates), "wrong_median_raw": float(np.median(distances)) if distances else None, "correct_minus_wrong_median_raw": None if not distances else correct_distance - float(np.median(distances)), "correct_rank_raw": float(rank), "correct_beats_wrong_rate_raw": None if not distances else float(np.mean(raw_beats)), "content_status": content_status, "excluded_same_transcript_count": len(rows_by_video[target_id]) - len(candidates), "k": k, "wrong_pairs": rows_by_video[target_id]})
        # Add unit norm descriptors using the same zero-lag and quantile rows.
        for item in records:
            if item["video_type"] != video_type:
                continue
            target_id = int(item["target_id"])
            v_unit = SyncNetEngine.unit_features(visual_cache[target_id])
            correct_a = SyncNetEngine.unit_features(audio_cache[target_id])
            k_local = int(item["k"])
            correct_matrix = SyncNetEngine.distance_matrix(v_unit, correct_a)
            support = list(range(config.VSHIFT, correct_matrix.shape[0] - config.VSHIFT))
            selected = [support[int(index)] for index in np.floor(np.linspace(0, len(support) - 1, k_local)).astype(np.int64)]
            correct = float(np.mean(correct_matrix[np.asarray(selected), config.VSHIFT], dtype=np.float64))
            wrong = []
            for donor in item["wrong_pairs"]:
                a_unit = SyncNetEngine.unit_features(audio_cache[int(donor["donor_id"])])
                matrix = SyncNetEngine.distance_matrix(v_unit, a_unit)
                donor_support = list(range(config.VSHIFT, matrix.shape[0] - config.VSHIFT))
                donor_selected = [donor_support[int(index)] for index in np.floor(np.linspace(0, len(donor_support) - 1, k_local)).astype(np.int64)]
                wrong.append(float(np.mean(matrix[np.asarray(donor_selected), config.VSHIFT], dtype=np.float64)))
            wrong_valid = [value for donor, value in zip(item["wrong_pairs"], wrong, strict=True) if not donor["text_equal"]]
            unit_beats = [1.0 if correct < value else 0.5 if correct == value else 0.0 for value in wrong_valid]
            item.update({"correct_distance_unit": correct, "wrong_median_unit": float(np.median(wrong_valid)) if wrong_valid else None, "correct_minus_wrong_median_unit": None if not wrong_valid else correct - float(np.median(wrong_valid)), "correct_rank_unit": None if not wrong_valid else float(1 + sum(value < correct for value in wrong_valid) + 0.5 * sum(value == correct for value in wrong_valid)), "correct_beats_wrong_rate_unit": None if not wrong_valid else float(np.mean(unit_beats))})
    group_rows = []
    for video_type in video_types:
        current = [row for row in records if row["video_type"] == video_type]
        group_rows.append({"video_type": video_type, "record_count": len(current), "correct_minus_wrong_median_raw_mean": float(np.mean([row["correct_minus_wrong_median_raw"] for row in current])), "correct_minus_wrong_median_unit_mean": float(np.mean([row["correct_minus_wrong_median_unit"] for row in current]))})
    return {"status": "COMPLETE" if len(records) == len(config.VIDEO_TYPES) * len(config.SAMPLE_IDS) else "INCOMPLETE", "pair_count": sum(len(row["wrong_pairs"]) for row in records), "expected_pair_count": config.EXPECTED_WRONG_CONTENT, "records": records, "video_type_summaries": group_rows, "same_transcript_rule": "same-text donor excluded from denominator and reported", "transcript_status": "CONTENT_UNVERIFIED" if any(row["content_status"] != "COMPLETE" for row in records) else "VERIFIED", "no_new_model_forwards": True}


def _rms_segments(mask: np.ndarray, *, minimum: int) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    lengths: list[int] = []
    start: int | None = None
    for index, flag in enumerate(np.r_[values, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            length = index - start
            if length >= minimum:
                lengths.append(length)
            start = None
    return lengths


def _rhythm_row(values: np.ndarray, mask: np.ndarray, *, sample_id: int, source: str, source_group: str) -> dict[str, Any]:
    from .audio import envelope_10ms, stft

    x = np.asarray(values, dtype=np.float64) / 32768.0
    activity = np.asarray(mask, dtype=bool)
    envelope = envelope_10ms(x)
    pause_lengths = _rms_segments(~activity, minimum=3200)
    active_lengths = _rms_segments(activity, minimum=1600)
    env_centered = envelope - envelope.mean() if envelope.size else envelope
    spectrum = np.abs(np.fft.rfft(env_centered)) ** 2 if env_centered.size else np.asarray([])
    freq = np.fft.rfftfreq(env_centered.size, d=0.01) if env_centered.size else np.asarray([])
    band = lambda low, high: float(np.sum(spectrum[(freq >= low) & (freq <= high)], dtype=np.float64))
    broad = band(0.5, 20.0)
    spectra, starts = stft(x)
    magnitude = np.asarray(np.abs(spectra), dtype=np.float64)
    normalized = magnitude / np.maximum(np.linalg.norm(magnitude, axis=1, keepdims=True), config.EPSILON)
    active_frames = np.asarray([
        bool(np.any(activity[max(0, int(start) - config.PAD) : min(activity.size, int(start) - config.PAD + config.WINDOW)]))
        for start in starts
    ], dtype=bool)
    adjacent = np.linalg.norm(np.diff(normalized, axis=0), axis=1) if normalized.shape[0] > 1 else np.asarray([])
    valid_adjacent = adjacent[active_frames[:-1] & active_frames[1:]] if adjacent.size and active_frames.size == normalized.shape[0] else adjacent
    return {"sample_id": sample_id, "source": source, "source_group": source_group, "activity_ratio": float(activity.mean()), "pause_count": len(pause_lengths), "pause_median_ms": None if not pause_lengths else float(np.median(pause_lengths) * 1000.0 / config.SAMPLE_RATE), "pause_iqr_ms": None if not pause_lengths else float((np.quantile(pause_lengths, 0.75) - np.quantile(pause_lengths, 0.25)) * 1000.0 / config.SAMPLE_RATE), "active_segment_count": len(active_lengths), "active_duration_median_ms": None if not active_lengths else float(np.median(active_lengths) * 1000.0 / config.SAMPLE_RATE), "active_duration_cv": None if len(active_lengths) < 2 or np.mean(active_lengths) == 0 else float(np.std(active_lengths, ddof=1) / np.mean(active_lengths)), "envelope_modulation_ratio_0.5_8_over_0.5_20": None if broad <= 0.0 else float(band(0.5, 8.0) / broad), "adjacent_activity_stft_normalized_l2_mean": None if valid_adjacent.size == 0 else float(valid_adjacent.mean()), "rhythm_status": "DESCRIPTIVE_ONLY"}


def _rhythm_analysis(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], native_values: Mapping[int, float] | None) -> dict[str, Any]:
    from .audio import read_pcm16_wav

    rows = []
    for raw in assets.get("records", []):
        sample_id = int(raw["sample_id"])
        group = str(raw["source_group"])
        for source in ("N", "T"):
            record = next(row for row in audio_manifest["records"] if int(row["sample_id"]) == sample_id and row["source"] == source and row["condition"] == "A0")
            values = read_pcm16_wav(record["path"])
            mask = np.load(paths.audio / str(sample_id) / source / "activity_mask.npy", allow_pickle=False).astype(bool)
            rows.append(_rhythm_row(values, mask, sample_id=sample_id, source=source, source_group=group))
    paired = []
    for sample_id in config.SAMPLE_IDS:
        n = next(row for row in rows if row["sample_id"] == sample_id and row["source"] == "N")
        t = next(row for row in rows if row["sample_id"] == sample_id and row["source"] == "T")
        row = {"sample_id": sample_id, "source_group": n["source_group"]}
        for key in ("activity_ratio", "pause_median_ms", "pause_iqr_ms", "active_duration_cv", "envelope_modulation_ratio_0.5_8_over_0.5_20", "adjacent_activity_stft_normalized_l2_mean"):
            row[key] = None if n[key] is None or t[key] is None else float(t[key] - n[key])
        if native_values is not None and sample_id in native_values:
            row["native_delta"] = float(native_values[sample_id])
        paired.append(row)
    correlations: dict[str, Any] = {}
    if native_values is None:
        for key in ("activity_ratio", "pause_median_ms", "pause_iqr_ms", "active_duration_cv", "envelope_modulation_ratio_0.5_8_over_0.5_20", "adjacent_activity_stft_normalized_l2_mean"):
            correlations[key] = {"status": "NOT_ASSESSED", "reason": "fresh native B baseline unavailable", "method": "Spearman prespecified descriptor only"}
    else:
        from scipy.stats import spearmanr

        for key in ("activity_ratio", "pause_median_ms", "pause_iqr_ms", "active_duration_cv", "envelope_modulation_ratio_0.5_8_over_0.5_20", "adjacent_activity_stft_normalized_l2_mean"):
            valid = [row for row in paired if row.get("native_delta") is not None and row.get(key) is not None]
            if len(valid) < 3:
                correlations[key] = {"status": "NOT_ESTIMABLE", "n": len(valid)}
            else:
                rho, p = spearmanr([row[key] for row in valid], [row["native_delta"] for row in valid])
                correlations[key] = {"status": "DESCRIPTIVE_ONLY", "n": len(valid), "rho": float(rho), "p_value_descriptive": float(p)}
    return {"status": "COMPLETE", "rows": rows, "paired_N_to_T": paired, "correlations": correlations, "causal_status": "DESCRIPTIVE_ONLY", "no_mediator_claim": True}


def _plots(root: Path, primary: Mapping[str, Any], unit: Mapping[str, Any], content: Mapping[str, Any], crossed: Mapping[str, Any] | None) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError, RuntimeError):  # pragma: no cover
        write_json(root / "figures_status.json", {"status": "UNAVAILABLE"})
        return []
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    names = list(primary.get("summaries", {}).keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(names))
    means, lows, highs = [], [], []
    labels = []
    for name in names:
        summary = primary["summaries"][name]
        labels.append(name)
        if summary.get("status") == "COMPLETE":
            means.append(float(summary["mean"]))
            lows.append(float(summary["ci95"][0]))
            highs.append(float(summary["ci95"][1]))
        else:
            means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
    ax.errorbar(xs, means, yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs, labels, rotation=45, ha="right")
    ax.set_ylabel("effect in official Sync-C")
    ax.set_title("Six preregistered contrasts (95% descriptive intervals)")
    fig.tight_layout()
    path = figure_dir / "primary_effects.png"; fig.savefig(path, dpi=160); outputs.append(str(path)); plt.close(fig)
    if unit.get("status") in {"COMPLETE", "PARTIAL"}:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = []
        labels = []
        for name, summary in unit.get("summaries", {}).items():
            labels.append(name + " unit")
            values.append(float(summary["mean"]) if summary.get("status") == "COMPLETE" else np.nan)
        ax.bar(np.arange(len(values)), values); ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right"); ax.set_ylabel("C_unit effect"); fig.tight_layout(); path = figure_dir / "unit_geometry.png"; fig.savefig(path, dpi=160); outputs.append(str(path)); plt.close(fig)
    if content.get("status") == "COMPLETE":
        fig, ax = plt.subplots(figsize=(8, 4)); current = content.get("records", []); labels = [f"{row['video_type']}:{row['target_id']}" for row in current]; values = [row["correct_minus_wrong_median_raw"] for row in current]; ax.axhline(0, color="black", linewidth=0.8); ax.bar(np.arange(len(values)), values); ax.set_xticks(np.arange(len(values)), labels, rotation=90); ax.set_ylabel("correct − wrong median distance"); fig.tight_layout(); path = figure_dir / "content_retrieval.png"; fig.savefig(path, dpi=160); outputs.append(str(path)); plt.close(fig)
    return outputs


def analyze_stage(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], fixed_manifest: Mapping[str, Any], crossed_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = paths.analysis
    root.mkdir(parents=True, exist_ok=True)
    cached_path = root / "summary.json"
    if cached_path.is_file():
        try:
            cached = read_self_hashed_json(cached_path)
            current_a_hash = file_sha256(paths.fixed_video / "a_manifest.json")
            current_crossed_hash = file_sha256(paths.crossed / "manifest.json") if paths.crossed.joinpath("manifest.json").is_file() else None
            current_audio_hash = file_sha256(paths.audio / "manifest.json")
            current_assets_hash = file_sha256(paths.audit / "assets.json")
            if cached.get("protocol_id") == config.PROTOCOL_ID and cached.get("analysis_code_hash") == _analysis_code_hash() and cached.get("a_manifest_sha256") == current_a_hash and cached.get("crossed_manifest_sha256") == current_crossed_hash and cached.get("audio_manifest_sha256") == current_audio_hash and cached.get("assets_sha256") == current_assets_hash:
                return cached
        except ProtocolError:
            pass
    a_manifest = read_self_hashed_json(paths.fixed_video / "a_manifest.json")
    a_analysis, a_rows = _fixed_video_analysis(a_manifest, paths.fixed_video / "features")
    indices = bootstrap_indices()
    bootstrap_path = root / "bootstrap_indices.npy"
    temporary = bootstrap_path.with_name(f".{bootstrap_path.name}.tmp.npy")
    np.save(temporary, indices, allow_pickle=False); temporary.replace(bootstrap_path)
    primary = _primary_analysis(a_rows, crossed_manifest, indices)
    unit = _unit_effects(a_rows, paths.fixed_video / "features", indices)
    content = _content_retrieval(a_rows, paths.fixed_video / "features", assets)
    native_values = None
    if crossed_manifest and crossed_manifest.get("primary_records"):
        native_values = {int(key.split(":", 1)[0]): float(value["value"]) for key, value in crossed_manifest["primary_records"].items() if key.endswith(":NATIVE")}
    rhythm = _rhythm_analysis(paths, assets, audio_manifest, native_values)
    plots = _plots(root, primary, unit, content, crossed_manifest)
    csv_write(root / "a_endpoints.csv", ("id", "source_group", "source", "video_type", "eval_condition", "support_count", "sync_c", "sync_d", "background_b", "d0", "official_offset", "matrix_hash"), [{**{key: row.get(key) for key in ("id", "source_group", "source", "video_type", "eval_condition", "matrix_hash")}, **{key: row["metrics"].get(key) for key in ("support_count", "sync_c", "sync_d", "background_b", "d0", "official_offset")}} for row in a_rows])
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "05_analysis",
        "status": "COMPLETE" if a_analysis.get("status") == "COMPLETE" else "PARTIAL",
        "a": a_analysis,
        "primary": primary,
        "unit_geometry": unit,
        "content_retrieval": content,
        "rhythm": rhythm,
        "bootstrap_indices": {"path": str(bootstrap_path), "sha256": file_sha256(bootstrap_path), "shape": [int(item) for item in indices.shape], "seed": config.BOOTSTRAP_SEED, "draws": config.BOOTSTRAP_DRAWS, "shared_by_all_six": True},
        "figures": plots,
        "human_status": {"sync": "PERCEPTION_NOT_ASSESSED", "quality": "QUALITY_NOT_ASSESSED"},
        "limitations": ["A effects are evaluator-input effects on fixed pixels", "B dependency/resource status is separate", "H3 is descriptive only", "H4 is not causally identified", "Sync-C is not human synchronization quality"],
        "analysis_code_hash": _analysis_code_hash(),
        "a_manifest_sha256": file_sha256(paths.fixed_video / "a_manifest.json"),
        "crossed_manifest_sha256": file_sha256(paths.crossed / "manifest.json") if paths.crossed.joinpath("manifest.json").is_file() else None,
        "audio_manifest_sha256": file_sha256(paths.audio / "manifest.json"),
        "assets_sha256": file_sha256(paths.audit / "assets.json"),
    }
    return write_self_hashed_json(cached_path, result)


def _analysis_code_hash() -> str:
    return file_sha256(Path(__file__))


def _npy_atomic(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    return file_sha256(path)
