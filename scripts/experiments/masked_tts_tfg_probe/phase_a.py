"""Read-only diagnosis for the exploratory masked-TTS parent run."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VELOCITY_WEIGHT = 0.25
SEEDS = (20260901, 20260902, 20260903)
CONDITIONS = ("PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY")
NEGATIVE_GROUPS = ("6ORDQFh0Byw", "6yR5OUVb2gY")
EPSILON = 1e-6
WAVLM_DIM = 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite {name}")
    return result


def _median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("median requires at least one value")
    result = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite median")
    return result


def _mean(values: np.ndarray) -> float:
    result = float(np.asarray(values, dtype=np.float32).mean(dtype=np.float32))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite mean")
    return result


def _stats_array(stats: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    values = np.asarray(stats[key], dtype=np.float32)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"invalid normalization statistic {key}")
    return values


def _standardize_mel(mel: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(mel, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 80:
        raise ValueError(f"natural mel must have shape [80,N], got {values.shape}")
    mean = _stats_array(stats, "mel_mean", 80)
    std = _stats_array(stats, "mel_std", 80)
    return (values - mean[:, None]) / std[:, None]


def _standardize_tts(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    features = np.asarray(values, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != WAVLM_DIM:
        raise ValueError(f"TTS feature must have shape [N,1024], got {features.shape}")
    mean = _stats_array(stats, "tts_mean", WAVLM_DIM)
    std = _stats_array(stats, "tts_std", WAVLM_DIM)
    return (features - mean[None, :]) / std[None, :]


def _phone_phase_linear(source: np.ndarray, source_start: int, source_end: int, destination_length: int) -> np.ndarray:
    values = np.asarray(source)
    if values.ndim != 2 or values.shape[1] != WAVLM_DIM:
        raise ValueError("source trajectory must have shape [N,1024]")
    if source_start < 0 or source_end > values.shape[0] or source_end - source_start < 2:
        raise ValueError("source phone does not satisfy frozen WavLM support")
    if destination_length < 1:
        raise ValueError("destination phone must contain at least one frame")
    selected = np.asarray(values[source_start:source_end], dtype=np.float32)
    n = selected.shape[0]
    result = np.empty((destination_length, WAVLM_DIM), dtype=np.float32)
    for index in range(destination_length):
        u = (index + 0.5) / destination_length
        coordinate = float(np.clip(u * n - 0.5, 0.0, n - 1.0))
        left = int(np.floor(coordinate))
        right = min(left + 1, n - 1)
        alpha = coordinate - left
        result[index] = (1.0 - alpha) * selected[left] + alpha * selected[right]
    if not np.isfinite(result).all():
        raise FloatingPointError("phone phase interpolation produced non-finite data")
    return result


def _load_natural_mels(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if not result:
        raise ValueError("natural mel archive is empty")
    for sample_id, values in result.items():
        if values.ndim != 2 or values.shape[0] != 80 or not np.isfinite(values).all():
            raise ValueError(f"invalid natural mel for {sample_id}")
    return result


def _load_centroids(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        labels = [str(value) for value in archive["labels"].tolist()]
        values = np.asarray(archive["centroids"], dtype=np.float32)
    if values.shape != (len(labels), WAVLM_DIM) or len(set(labels)) != len(labels):
        raise ValueError("invalid phone centroid archive")
    if not np.isfinite(values).all():
        raise FloatingPointError("phone centroid archive contains non-finite data")
    return {label: values[index] for index, label in enumerate(labels)}


def _load_tts_features(lock: Mapping[str, Any], *, verify_all: bool = True) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in lock.get("records", []):
        sample_id = str(row["sample_id"])
        path = Path(row["paths"]["tts_feature"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(row["sha256"]["tts_feature"])
        if verify_all and file_sha256(path) != expected:
            raise ValueError(f"TTS feature hash mismatch for {sample_id}")
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != WAVLM_DIM or not np.isfinite(values).all():
            raise ValueError(f"invalid TTS feature for {sample_id}")
        if sample_id in result and not np.array_equal(result[sample_id], values):
            raise ValueError(f"duplicate TTS feature identity for {sample_id}")
        result[sample_id] = values
    return result


def _validate_parent_artifacts(parent: Path) -> dict[str, Any]:
    required = [
        "00_lock/lock.json",
        "01_masks/mask_manifest.json",
        "02_features/natural_mels.npz",
        "02_features/normalization.json",
        "02_features/phone_centroids.npz",
        "02_features/phone_centroids.json",
        "04_eval/evaluation.json",
    ]
    for relative in required:
        path = parent / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = read_json(parent / "00_lock/lock.json")
    masks = read_json(parent / "01_masks/mask_manifest.json")
    evaluation = read_json(parent / "04_eval/evaluation.json")
    if lock.get("sealed_splits_accessed") is not False or evaluation.get("sealed_splits_accessed") is not False:
        raise ValueError("parent accessed sealed splits")
    if evaluation.get("status") != "GO" or evaluation.get("conditions") != list(CONDITIONS):
        raise ValueError("parent evaluation status or conditions are invalid")
    if tuple(int(seed) for seed in evaluation.get("seeds", [])) != SEEDS:
        raise ValueError("parent seed order is invalid")
    eval_masks = [row for row in masks.get("masks", []) if row.get("prototype_split") == "evaluation"]
    if int(evaluation.get("evaluation_masks", -1)) != len(eval_masks):
        raise ValueError("evaluation mask count mismatch")
    mask_by_hash = {str(row["mask_sha256"]): row for row in eval_masks}
    if len(mask_by_hash) != len(eval_masks):
        raise ValueError("duplicate evaluation mask hash")
    if tuple(str(group) for group in masks.get("evaluation_groups", [])) != tuple(str(group) for group in lock["groups"]["evaluation"]):
        raise ValueError("evaluation group binding mismatch")
    records = evaluation.get("records", [])
    expected = len(eval_masks) * len(SEEDS) * len(CONDITIONS)
    if len(records) != expected or int(evaluation.get("required_cells", -1)) != expected:
        raise ValueError(f"parent evaluation cell count mismatch: {len(records)} != {expected}")
    cells: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in records:
        if row.get("status") != "complete" or str(row.get("condition")) not in CONDITIONS:
            raise ValueError("parent required cell is not complete")
        key = (int(row["seed"]), str(row["mask_sha256"]))
        if key in cells and str(row["condition"]) in cells[key]:
            raise ValueError(f"duplicate parent cell {key} {row['condition']}")
        if key[1] not in mask_by_hash:
            raise ValueError(f"parent cell references unknown mask {key[1]}")
        cells[key].add(str(row["condition"]))
    expected_keys = {(seed, mask_hash) for seed in SEEDS for mask_hash in mask_by_hash}
    if set(cells) != expected_keys or any(value != set(CONDITIONS) for value in cells.values()):
        raise ValueError("parent evaluation matrix is incomplete")
    return {"lock": lock, "masks": masks, "evaluation": evaluation, "mask_by_hash": mask_by_hash}


def _record_index(evaluation: Mapping[str, Any]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in evaluation["records"]:
        key = (int(row["seed"]), str(row["mask_sha256"]), str(row["condition"]))
        if key in result:
            raise ValueError(f"duplicate record {key}")
        result[key] = row
    return result


def _recompute_losses(
    prediction: np.ndarray,
    natural_mel: np.ndarray,
    stats: Mapping[str, Any],
    mask: Mapping[str, Any],
) -> dict[str, float]:
    if prediction.shape != (96, 80) or not np.isfinite(prediction).all():
        raise ValueError(f"invalid prediction shape {prediction.shape}")
    target = _standardize_mel(natural_mel, stats)[:, int(mask["window_start_frame"]):int(mask["window_start_frame"]) + 96].T
    core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
    if not (0 <= core_start < core_end <= 96):
        raise ValueError("invalid target core")
    core = np.zeros(96, dtype=bool)
    core[core_start:core_end] = True
    patch = _mean(np.abs(prediction[core] - target[core]))
    endpoint = core[1:] & core[:-1]
    if endpoint.any():
        velocity = _mean(np.abs((prediction[1:] - prediction[:-1])[endpoint] - (target[1:] - target[:-1])[endpoint]))
    else:
        velocity = 0.0
    return {"patch": patch, "velocity": velocity, "total": patch + VELOCITY_WEIGHT * velocity}


def _contrast_row(
    mask: Mapping[str, Any],
    seed: int,
    by_condition: Mapping[str, Mapping[str, Any]],
    natural_mels: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
    tts_features: Mapping[str, np.ndarray],
    centroids: Mapping[str, np.ndarray],
    parent: Path,
) -> dict[str, Any]:
    sample_id = str(mask["sample_id"])
    natural_mel = natural_mels[sample_id]
    recomputed: dict[str, dict[str, float]] = {}
    arithmetic_errors: list[float] = []
    component_errors: list[float] = []
    predictions: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        record = by_condition[condition]
        prediction_path = parent / "04_eval" / str(seed) / str(mask["mask_sha256"]) / condition / "prediction.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as archive:
            if archive.files != ["prediction"]:
                raise ValueError(f"unexpected prediction archive for {prediction_path}")
            prediction = np.asarray(archive["prediction"], dtype=np.float32)
        expected_hash = str(record["prediction_sha256"])
        actual_hash = _array_hash(prediction)
        if actual_hash != expected_hash:
            raise ValueError(f"prediction hash mismatch for {prediction_path}")
        predictions[condition] = prediction
        recomputed[condition] = _recompute_losses(prediction, natural_mel, stats, mask)
        for component in ("patch", "velocity", "total"):
            component_errors.append(abs(recomputed[condition][component] - float(record["loss"][component])))
    nat, paired, centroid = (recomputed["NAT_ONLY"], recomputed["PAIRED_TTS"], recomputed["PHONE_CENTROID"])
    modality_patch = nat["patch"] - paired["patch"]
    modality_velocity = nat["velocity"] - paired["velocity"]
    modality_total = modality_patch + VELOCITY_WEIGHT * modality_velocity
    token_patch = centroid["patch"] - paired["patch"]
    token_velocity = centroid["velocity"] - paired["velocity"]
    token_total = token_patch + VELOCITY_WEIGHT * token_velocity
    for contrast, left, right in (("modality", "NAT_ONLY", "PAIRED_TTS"), ("token", "PHONE_CENTROID", "PAIRED_TTS")):
        calculated = recomputed[left]["total"] - recomputed[right]["total"]
        recorded = float(by_condition[left]["loss"]["total"]) - float(by_condition[right]["loss"]["total"])
        arithmetic_errors.append(abs(calculated - recorded))
    trajectory = _phone_phase_linear(
        _standardize_tts(tts_features[sample_id], stats),
        int(mask["tts_frame_start"]),
        int(mask["tts_frame_end"]),
        int(mask["core_end"]) - int(mask["core_start"]),
    )
    label = str(mask["label"])
    if label not in centroids:
        raise ValueError(f"missing centroid for label {label}")
    centroid_trajectory = np.broadcast_to(centroids[label][None, :], trajectory.shape)
    distance_frame = np.sqrt(np.mean((trajectory - centroid_trajectory) ** 2, axis=1, dtype=np.float32))
    temporal_delta = np.diff(trajectory, axis=0)
    temporal_rms = 0.0 if not len(temporal_delta) else _mean(np.sqrt(np.mean(temporal_delta**2, axis=1, dtype=np.float32)))
    paired_prediction = predictions["PAIRED_TTS"][int(mask["core_start"]):int(mask["core_end"])]
    centroid_prediction = predictions["PHONE_CENTROID"][int(mask["core_start"]):int(mask["core_end"])]
    natural_duration = float(mask["natural_end_s"]) - float(mask["natural_start_s"])
    tts_duration = float(mask["tts_duration_s"])
    if natural_duration <= 0 or tts_duration <= 0:
        raise ValueError("invalid mask duration")
    return {
        "seed": int(seed),
        "mask_sha256": str(mask["mask_sha256"]),
        "sample_id": sample_id,
        "source_group": str(mask["source_group"]),
        "label": label,
        "canonical_index": int(mask["canonical_index"]),
        "core_start": int(mask["core_start"]),
        "core_end": int(mask["core_end"]),
        "core_length": int(mask["core_end"]) - int(mask["core_start"]),
        "natural_frame_count": int(mask["natural_frame_count"]),
        "tts_source_frame_length": int(mask["tts_frame_end"]) - int(mask["tts_frame_start"]),
        "tts_duration_s": tts_duration,
        "natural_duration_s": natural_duration,
        "tts_to_natural_duration_ratio": tts_duration / natural_duration,
        "paired_tts_patch": paired["patch"],
        "paired_tts_velocity": paired["velocity"],
        "paired_tts_total": paired["total"],
        "phone_centroid_patch": centroid["patch"],
        "phone_centroid_velocity": centroid["velocity"],
        "phone_centroid_total": centroid["total"],
        "nat_only_patch": nat["patch"],
        "nat_only_velocity": nat["velocity"],
        "nat_only_total": nat["total"],
        "modality_patch_gain": modality_patch,
        "modality_velocity_gain": modality_velocity,
        "modality_weighted_velocity_gain": VELOCITY_WEIGHT * modality_velocity,
        "modality_total_gain": modality_total,
        "token_patch_gain": token_patch,
        "token_velocity_gain": token_velocity,
        "token_weighted_velocity_gain": VELOCITY_WEIGHT * token_velocity,
        "token_total_gain": token_total,
        "paired_to_centroid_feature_distance_rms": _mean(distance_frame),
        "paired_within_phone_temporal_variation_rms": temporal_rms,
        "paired_prediction_core_rms": _mean(np.sqrt(np.mean(paired_prediction**2, axis=1, dtype=np.float32))),
        "centroid_prediction_core_rms": _mean(np.sqrt(np.mean(centroid_prediction**2, axis=1, dtype=np.float32))),
        "paired_centroid_prediction_core_rms": _mean(np.sqrt(np.mean((paired_prediction - centroid_prediction)**2, axis=1, dtype=np.float32))),
        "max_component_recompute_error": max(component_errors),
        "max_contrast_arithmetic_error": max(arithmetic_errors),
    }


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], count_key: str | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    result = {key: rows[0][key] for key in keys}
    numeric_keys = [
        key for key in rows[0]
        if key not in keys and isinstance(rows[0][key], (int, float)) and not isinstance(rows[0][key], bool)
    ]
    for key in numeric_keys:
        result[key] = _median(float(row[key]) for row in rows)
    if count_key is not None:
        result[count_key] = len(rows)
    return result


def _aggregate_hierarchy(rows: Sequence[Mapping[str, Any]], masks: Mapping[str, Mapping[str, Any]], groups: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (int(row["canonical_index"]), int(row["seed"])))
    record_rows: list[dict[str, Any]] = []
    by_record: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_record[(str(row["sample_id"]), int(row["seed"]))].append(row)
    for (sample_id, seed), local in sorted(by_record.items(), key=lambda item: (item[0][0].encode("utf-8"), item[0][1])):
        record = _aggregate(local, ("source_group", "seed", "sample_id"), "mask_count")
        record_rows.append(record)
    group_seed_rows: list[dict[str, Any]] = []
    by_group_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in record_rows:
        by_group_seed[(str(row["source_group"]), int(row["seed"]))].append(row)
    for (group, seed), local in sorted(by_group_seed.items(), key=lambda item: (groups.index(item[0][0]), item[0][1])):
        group_seed_rows.append(_aggregate(local, ("source_group", "seed"), "record_count"))
        group_seed_rows[-1]["mask_count"] = int(sum(int(row["mask_count"]) for row in local))
    final_group_rows: list[dict[str, Any]] = []
    for group in groups:
        local = [row for row in group_seed_rows if str(row["source_group"]) == group]
        if len(local) != len(SEEDS):
            raise ValueError(f"group lacks three seed rows: {group}")
        row = _aggregate(local, ("source_group",), "seed_count")
        final_group_rows.append(row)
    return {"record_rows": record_rows, "group_seed_rows": group_seed_rows, "final_group_rows": final_group_rows}


def _phone_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_group"]), str(row["label"]))].append(row)
    result = []
    for (group, label), local in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        result.append(_aggregate(local, ("source_group", "label"), "mask_count"))
    return result


def _pooled_positive_metrics(final_group_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positives = [row for row in final_group_rows if str(row["source_group"]) not in NEGATIVE_GROUPS]
    return {
        "group_count": len(positives),
        **{key: _median(float(row[key]) for row in positives) for key in (
            "token_patch_gain", "token_weighted_velocity_gain", "token_total_gain",
            "core_length", "tts_source_frame_length", "tts_to_natural_duration_ratio",
            "paired_to_centroid_feature_distance_rms", "paired_within_phone_temporal_variation_rms",
            "paired_centroid_prediction_core_rms",
        )},
    }


def _dominant_pattern(group: Mapping[str, Any], pooled: Mapping[str, Any], group_seed_rows: Sequence[Mapping[str, Any]], phone_rows: Sequence[Mapping[str, Any]]) -> str:
    patch = abs(float(group["token_patch_gain"]))
    velocity = abs(float(group["token_weighted_velocity_gain"]))
    if patch > 1.5 * max(velocity, EPSILON):
        return "patch"
    if velocity > 1.5 * max(patch, EPSILON):
        return "velocity"
    seeds = [row for row in group_seed_rows if row["source_group"] == group["source_group"]]
    if any(float(row["token_total_gain"]) > 0 for row in seeds):
        return "seed_instability"
    if abs(float(group["tts_to_natural_duration_ratio"]) - float(pooled["tts_to_natural_duration_ratio"])) > max(0.25 * abs(float(pooled["tts_to_natural_duration_ratio"])), 0.05):
        return "duration_alignment"
    local_phone = [row for row in phone_rows if row["source_group"] == group["source_group"]]
    if local_phone and max(int(row["mask_count"]) for row in local_phone) >= 0.2 * sum(int(row["mask_count"]) for row in local_phone):
        return "phone_mix"
    return "diffuse"


def _bootstrap(values: np.ndarray, draws: int = 10_000, seed: int = 20260902) -> dict[str, Any]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid bootstrap values")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(medians, 0.025, method="linear")), float(np.quantile(medians, 0.975, method="linear"))],
        "draws": int(draws),
        "seed": int(seed),
        "unit": "whole_source_group",
        "method": "numpy_quantile_linear",
    }


def diagnose(parent: Path, output_dir: Path) -> dict[str, Any]:
    parent = parent.resolve()
    output_dir = output_dir.resolve()
    artifacts = _validate_parent_artifacts(parent)
    lock, masks, evaluation = artifacts["lock"], artifacts["masks"], artifacts["evaluation"]
    stats = read_json(parent / "02_features/normalization.json")
    natural_mels = _load_natural_mels(parent / "02_features/natural_mels.npz")
    centroids = _load_centroids(parent / "02_features/phone_centroids.npz")
    tts_features = _load_tts_features(lock)
    expected_ids = {str(row["sample_id"]) for row in lock["records"]}
    if set(natural_mels) != expected_ids or set(tts_features) != expected_ids:
        raise ValueError("parent feature IDs do not match locked records")
    mask_by_hash = artifacts["mask_by_hash"]
    records = _record_index(evaluation)
    rows: list[dict[str, Any]] = []
    max_component_error = 0.0
    max_arithmetic_error = 0.0
    for mask in sorted(mask_by_hash.values(), key=lambda row: int(row["canonical_index"])):
        mask_hash = str(mask["mask_sha256"])
        for seed in SEEDS:
            by_condition = {condition: records[(seed, mask_hash, condition)] for condition in CONDITIONS}
            row = _contrast_row(mask, seed, by_condition, natural_mels, stats, tts_features, centroids, parent)
            rows.append(row)
            max_component_error = max(max_component_error, float(row["max_component_recompute_error"]))
            max_arithmetic_error = max(max_arithmetic_error, float(row["max_contrast_arithmetic_error"]))
    if max_arithmetic_error > EPSILON:
        raise ValueError(f"loss contrast arithmetic mismatch: {max_arithmetic_error}")
    groups = tuple(str(group) for group in masks["evaluation_groups"])
    hierarchy = _aggregate_hierarchy(rows, mask_by_hash, groups)
    phone_rows = _phone_rows(rows)
    final_group_rows = hierarchy["final_group_rows"]
    pooled = _pooled_positive_metrics(final_group_rows)
    comparisons = []
    for group_row in final_group_rows:
        group = str(group_row["source_group"])
        comparisons.append({
            "source_group": group,
            "negative_group": group in NEGATIVE_GROUPS,
            "dominant_observed_pattern": _dominant_pattern(group_row, pooled, hierarchy["group_seed_rows"], phone_rows) if group in NEGATIVE_GROUPS else None,
            "token_gain": float(group_row["token_total_gain"]),
            "token_patch_gain": float(group_row["token_patch_gain"]),
            "token_weighted_velocity_gain": float(group_row["token_weighted_velocity_gain"]),
            "token_gain_minus_pooled_positive": float(group_row["token_total_gain"]) - float(pooled["token_total_gain"]),
            "core_length": float(group_row["core_length"]),
            "tts_source_frame_length": float(group_row["tts_source_frame_length"]),
            "tts_to_natural_duration_ratio": float(group_row["tts_to_natural_duration_ratio"]),
            "paired_to_centroid_feature_distance_rms": float(group_row["paired_to_centroid_feature_distance_rms"]),
            "paired_within_phone_temporal_variation_rms": float(group_row["paired_within_phone_temporal_variation_rms"]),
            "paired_centroid_prediction_core_rms": float(group_row["paired_centroid_prediction_core_rms"]),
        })
    seed_overall = {}
    for seed in SEEDS:
        local = [row for row in hierarchy["group_seed_rows"] if int(row["seed"]) == seed]
        seed_overall[str(seed)] = {
            "seed": seed,
            "group_count": len(local),
            "modality_patch_gain": _median(float(row["modality_patch_gain"]) for row in local),
            "modality_weighted_velocity_gain": _median(float(row["modality_weighted_velocity_gain"]) for row in local),
            "modality_total_gain": _median(float(row["modality_total_gain"]) for row in local),
            "token_patch_gain": _median(float(row["token_patch_gain"]) for row in local),
            "token_weighted_velocity_gain": _median(float(row["token_weighted_velocity_gain"]) for row in local),
            "token_total_gain": _median(float(row["token_total_gain"]) for row in local),
        }
    modality = np.asarray([float(row["modality_total_gain"]) for row in final_group_rows], dtype=np.float64)
    token = np.asarray([float(row["token_total_gain"]) for row in final_group_rows], dtype=np.float64)
    source_artifacts = {
        "parent_run": str(parent),
        "lock_sha256": file_sha256(parent / "00_lock/lock.json"),
        "mask_manifest_sha256": file_sha256(parent / "01_masks/mask_manifest.json"),
        "evaluation_sha256": file_sha256(parent / "04_eval/evaluation.json"),
        "natural_mels_sha256": file_sha256(parent / "02_features/natural_mels.npz"),
        "normalization_sha256": file_sha256(parent / "02_features/normalization.json"),
        "phone_centroids_sha256": file_sha256(parent / "02_features/phone_centroids.npz"),
    }
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "phase": "A",
        "read_only": True,
        "sealed_splits_accessed": False,
        "parent_execution_mode": lock.get("execution_mode"),
        "parent_science": read_json(parent / "05_analysis/decision.json").get("science"),
        "source_artifacts": source_artifacts,
        "counts": {
            "evaluation_masks": len(mask_by_hash),
            "evaluation_records": len({str(row["sample_id"]) for row in mask_by_hash.values()}),
            "evaluation_groups": len(groups),
            "seeds": len(SEEDS),
            "required_cells": len(rows),
            "negative_groups": list(NEGATIVE_GROUPS),
            "positive_groups": [group for group in groups if group not in NEGATIVE_GROUPS],
        },
        "loss_definition": {"total": "patch + 0.25 * velocity", "velocity_weight": VELOCITY_WEIGHT, "arithmetic_tolerance": EPSILON},
        "loss_validation": {"max_component_recompute_error": max_component_error, "max_contrast_arithmetic_error": max_arithmetic_error, "passed": max_arithmetic_error <= EPSILON},
        "mask_rows": rows,
        "record_rows": hierarchy["record_rows"],
        "group_seed_rows": hierarchy["group_seed_rows"],
        "final_group_rows": final_group_rows,
        "seed_overall": seed_overall,
        "phone_rows": phone_rows,
        "negative_group_comparison": comparisons,
        "pooled_six_positive_groups": pooled,
        "descriptive_bootstrap": {
            "modality_total_gain": _bootstrap(modality),
            "token_total_gain": _bootstrap(token),
        },
        "claim_boundary": [
            "post-hoc descriptive diagnosis, not causal attribution",
            "not perceptible TTS-feature retention",
            "not waveform reachability",
            "not a TFG or SyncNet result",
        ],
    }
    write_json(output_dir / "analysis.json", analysis)
    report = render_report(analysis)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return analysis


def render_report(analysis: Mapping[str, Any]) -> str:
    counts = analysis["counts"]
    validation = analysis["loss_validation"]
    lines = [
        "# Phase A masked-TTS diagnosis",
        "",
        "This is a read-only, descriptive analysis of the completed exploratory parent run.",
        "",
        "## Completeness and arithmetic",
        "",
        f"- Evaluation cells: {counts['required_cells']} ({counts['evaluation_masks']} masks × {counts['seeds']} seeds).",
        f"- Source groups: {counts['evaluation_groups']}; negative groups: `{', '.join(counts['negative_groups'])}`.",
        f"- Maximum recomputed component error: `{validation['max_component_recompute_error']:.3g}`.",
        f"- Maximum contrast arithmetic error: `{validation['max_contrast_arithmetic_error']:.3g}` (tolerance `1e-6`).",
        f"- Arithmetic check: **{'PASS' if validation['passed'] else 'FAIL'}**.",
        "",
        "## Group-level contrasts",
        "",
        "| Group | Token patch | Token weighted velocity | Token total | Pattern |",
        "|---|---:|---:|---:|---|",
    ]
    for row in analysis["negative_group_comparison"]:
        pattern = row["dominant_observed_pattern"] or "positive reference"
        lines.append(f"| `{row['source_group']}` | {row['token_patch_gain']:.5f} | {row['token_weighted_velocity_gain']:.5f} | {row['token_gain']:.5f} | {pattern} |")
    lines.extend([
        "",
        "The six non-negative groups are the descriptive reference pool; their pooled median token total gain is "
        f"`{analysis['pooled_six_positive_groups']['token_total_gain']:.5f}`.",
        "",
        "## Interpretation boundary",
        "",
        "The pattern labels identify observed concentrations in the recorded data only. They are not causal explanations, new gates, or evidence that a particular phone, duration, or trajectory property caused the negative contrast.",
        "",
        "Phase B may start because the parent artifacts and Phase-A arithmetic are complete. The downstream result remains a direct-mel frozen-Wav2Lip sensitivity probe, not a waveform or perceptual TTS-retention test.",
        "",
    ])
    return "\n".join(lines)


__all__ = ["diagnose", "render_report", "VELOCITY_WEIGHT", "NEGATIVE_GROUPS", "SEEDS"]
