"""Build full-record mel drivers from frozen parent predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phase_a import CONDITIONS, SEEDS, _array_hash, _load_natural_mels, _standardize_mel, file_sha256, read_json, write_json

MEL_MIN = -4.0
MEL_MAX = 4.0
DRIVER_CONDITIONS = ("NATURAL_MEL", "PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY")


def _stats(stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(stats["mel_mean"], dtype=np.float32)
    std = np.asarray(stats["mel_std"], dtype=np.float32)
    if mean.shape != (80,) or std.shape != (80,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid mel normalization")
    return mean, std, mean[:, None], std[:, None]


def patch_normalized_mel(
    natural_standardized: np.ndarray,
    contributions: Sequence[tuple[int, int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    natural = np.asarray(natural_standardized, dtype=np.float32)
    if natural.ndim != 2 or natural.shape[0] != 80 or not np.isfinite(natural).all():
        raise ValueError("natural standardized mel must have shape [80,N]")
    total = np.zeros_like(natural, dtype=np.float32)
    counts = np.zeros(natural.shape[1], dtype=np.int32)
    for start, end, prediction_core in contributions:
        start, end = int(start), int(end)
        core = np.asarray(prediction_core, dtype=np.float32)
        if core.shape != (end - start, 80) or start < 0 or end > natural.shape[1] or start >= end:
            raise ValueError("invalid mel core contribution")
        total[:, start:end] += core.T
        counts[start:end] += 1
    result = natural.copy()
    covered = counts > 0
    result[:, covered] = total[:, covered] / counts[covered][None, :]
    return result, counts


def _prediction(parent: Path, seed: int, mask_hash: str, condition: str, expected_hash: str) -> np.ndarray:
    path = parent / "04_eval" / str(seed) / mask_hash / condition / "prediction.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["prediction"], dtype=np.float32)
    if prediction.shape != (96, 80) or _array_hash(prediction) != expected_hash or not np.isfinite(prediction).all():
        raise ValueError(f"invalid or changed parent prediction: {path}")
    return prediction


def _driver_id(sample_id: str, condition: str, seed: int | None) -> str:
    return f"{sample_id}__{condition}" if seed is None else f"{sample_id}__{condition}__{seed}"


def _build_one(
    parent: Path,
    record: Mapping[str, Any],
    masks: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
    natural: np.ndarray,
    seed: int | None,
    condition: str,
    output_dir: Path,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    natural_standardized = _standardize_mel(natural, stats)
    contributions: list[tuple[int, int, np.ndarray]] = []
    used_masks = []
    if condition != "NATURAL_MEL":
        if seed is None or condition not in CONDITIONS:
            raise ValueError("model driver requires a seed and frozen condition")
        for mask_hash in record["evaluation_mask_sha256"]:
            mask = masks[str(mask_hash)]
            eval_record_path = parent / "04_eval" / str(seed) / str(mask_hash) / condition / "record.json"
            eval_record = read_json(eval_record_path)
            prediction = _prediction(parent, seed, str(mask_hash), condition, str(eval_record["prediction_sha256"]))
            core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
            global_start = int(mask["window_start_frame"]) + core_start
            global_end = int(mask["window_start_frame"]) + core_end
            if global_start < 0 or global_end > natural.shape[1] or global_start >= global_end:
                raise ValueError(f"prediction core outside natural mel for {mask_hash}")
            contributions.append((global_start, global_end, prediction[core_start:core_end]))
            used_masks.append({
                "mask_sha256": str(mask_hash),
                "global_start_frame": global_start,
                "global_end_frame": global_end,
                "core_start": core_start,
                "core_end": core_end,
                "prediction_sha256": str(eval_record["prediction_sha256"]),
            })
    candidate_standardized, contribution_count = patch_normalized_mel(natural_standardized, contributions)
    mean, std, mean_2d, std_2d = _stats(stats)
    denormalized = candidate_standardized * std_2d + mean_2d
    output = natural.copy()
    covered = contribution_count > 0
    if condition != "NATURAL_MEL":
        output[:, covered] = denormalized[:, covered]
    unclamped = output.copy()
    clamp_mask = (output < MEL_MIN) | (output > MEL_MAX)
    output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise FloatingPointError(f"non-finite mel driver for {sample_id}")
    if condition == "NATURAL_MEL" and not np.array_equal(output, natural):
        raise ValueError("natural mel driver changed the natural source")
    if condition != "NATURAL_MEL" and np.any(output[:, ~covered] != natural[:, ~covered]):
        raise ValueError("mel driver changed frames outside evaluated target cores")
    driver_id = _driver_id(sample_id, condition, seed)
    path = output_dir / f"{driver_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, output, allow_pickle=False)
    return {
        "driver_id": driver_id,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "condition": condition,
        "seed": None if seed is None else int(seed),
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "shape": list(output.shape),
        "natural_mel_sha256": _array_hash(natural),
        "clamp_fraction": float(np.mean(clamp_mask)),
        "clamped_values": int(np.count_nonzero(clamp_mask)),
        "total_values": int(output.size),
        "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())],
        "mel_range_after_clamp": [float(output.min()), float(output.max())],
        "covered_frame_count": int(np.count_nonzero(covered)),
        "overlap_frame_count": int(np.count_nonzero(contribution_count > 1)),
        "used_masks": used_masks,
    }


def build_drivers(parent: Path, cohort_path: Path, output_dir: Path) -> dict[str, Any]:
    parent = parent.resolve()
    cohort = read_json(cohort_path)
    record_count = int(cohort.get("record_count", -1))
    records = cohort.get("records", [])
    if cohort.get("status") != "complete" or record_count != len(records) or record_count <= 0:
        raise ValueError("cohort is not complete")
    masks_manifest = read_json(parent / "01_masks/mask_manifest.json")
    masks = {str(row["mask_sha256"]): row for row in masks_manifest["masks"] if row.get("prototype_split") == "evaluation"}
    stats = read_json(parent / "02_features/normalization.json")
    natural_mels = _load_natural_mels(parent / "02_features/natural_mels.npz")
    if len(records) != record_count or len({str(row["sample_id"]) for row in records}) != record_count:
        raise ValueError("cohort records are incomplete or duplicated")
    rows: list[dict[str, Any]] = []
    output_dir = output_dir.resolve()
    for record in records:
        sample_id = str(record["sample_id"])
        natural = natural_mels.get(sample_id)
        if natural is None:
            raise ValueError(f"missing natural mel for {sample_id}")
        mask_hashes = [str(value) for value in record["evaluation_mask_sha256"]]
        if not mask_hashes or not set(mask_hashes).issubset(masks):
            raise ValueError(f"cohort mask binding is incomplete for {sample_id}")
        rows.append(_build_one(parent, record, masks, stats, natural, None, "NATURAL_MEL", output_dir))
        for condition in ("PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY"):
            for seed in SEEDS:
                rows.append(_build_one(parent, record, masks, stats, natural, seed, condition, output_dir))
    expected = record_count * 10
    if len(rows) != expected or len({str(row["driver_id"]) for row in rows}) != expected:
        raise ValueError("driver matrix is incomplete")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_direct_mel_drivers",
        "status": "complete",
        "parent_run": str(parent),
        "cohort_manifest_sha256": file_sha256(cohort_path),
        "natural_mel_source_sha256": file_sha256(parent / "02_features/natural_mels.npz"),
        "normalization_sha256": file_sha256(parent / "02_features/normalization.json"),
        "conditions": list(DRIVER_CONDITIONS),
        "seeds": list(SEEDS),
        "record_count": record_count,
        "drivers_per_record": 10,
        "driver_count": len(rows),
        "overlap_policy": "arithmetic_mean_of_contributing_prediction_cores",
        "outside_core_policy": "exact_natural_mel_source",
        "valid_mel_range": [MEL_MIN, MEL_MAX],
        "drivers": rows,
        "sealed_splits_accessed": False,
    }
    write_json(output_dir / "drivers.json", manifest)
    return manifest


__all__ = ["build_drivers", "patch_normalized_mel", "DRIVER_CONDITIONS", "SEEDS"]
