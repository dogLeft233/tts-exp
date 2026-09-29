"""Independent numerical checks for the phone-core validator.

The producer and the validator intentionally keep separate copies of the
distance, endpoint, and grouped-bootstrap calculations.  Only the runtime
module is used for bounded artifact I/O and worker-array loading.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

U_ROWS = tuple(range(30, 58))
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_DRAWS = 20_000


def matrix_from_embeddings(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Recompute the SyncNet distance matrix from the raw embeddings."""
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    if (
        visual32.ndim != 2
        or audio32.ndim != 2
        or visual32.shape != audio32.shape
        or visual32.shape[0] < 88
        or visual32.shape[1] != 1024
    ):
        raise rt.ProtocolError(
            f"invalid embedding shape: {visual32.shape}/{audio32.shape}"
        )
    padded = np.pad(audio32, ((15, 15), (0, 0)), mode="constant")
    result = np.empty((visual32.shape[0], 31), dtype=np.float32)
    for row in range(visual32.shape[0]):
        difference = visual32[row : row + 1] - padded[row : row + 31]
        result[row] = np.sqrt(
            np.sum(np.square(difference + np.float32(1e-6), dtype=np.float32), axis=1),
            dtype=np.float32,
        )
    return result


def score_metrics(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != 31
        or value.shape[0] < 58
        or not np.isfinite(value).all()
    ):
        raise rt.ProtocolError(f"invalid SyncNet matrix: {value.shape}")
    curve = np.mean(value[np.asarray(U_ROWS)], axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {
        "curve": [float(item) for item in curve],
        "min_index": index,
        "offset": 15 - index,
        "D": float(curve[index]),
        "C": float(np.median(curve) - curve[index]),
    }


def _gain(candidate: dict[str, Any], natural: dict[str, Any]) -> dict[str, float]:
    anchor = int(natural["min_index"])
    return {
        "C": float(candidate["C"] - natural["C"]),
        "D": float(natural["D"] - candidate["D"]),
        "A": float(natural["curve"][anchor] - candidate["curve"][anchor]),
    }


def bootstrap_indices(labels: list[str]) -> np.ndarray:
    if len(labels) != 8:
        raise rt.ProtocolError("group bootstrap requires eight labels")
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels))
    )


def grouped_stats(
    values: Iterable[float], groups: Iterable[str], indices: np.ndarray
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels):
        raise rt.ProtocolError("grouped statistic requires eight groups x two records")
    means = np.asarray(
        [np.mean(grouped[label], dtype=np.float64) for label in labels],
        dtype=np.float64,
    )
    sampled = np.asarray(indices, dtype=np.int64)
    estimates = means[sampled].mean(axis=1, dtype=np.float64)
    return {
        "mean": float(means.mean()),
        "ci99": [
            float(np.quantile(estimates, 0.005, method="linear")),
            float(np.quantile(estimates, 0.995, method="linear")),
        ],
        "group_labels": labels,
        "group_means": {
            label: float(value) for label, value in zip(labels, means, strict=True)
        },
        "group_positive_count": int(np.sum(means > 0.0)),
        "draws": int(sampled.shape[0]),
        "seed": BOOTSTRAP_SEED,
        "indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
        "indices": sampled.tolist(),
    }


def contrast_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    gains = [_gain(row[arm], row["N"]) for row in records]
    groups = [str(row["source_group"]) for row in records]
    labels = sorted(set(groups))
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for group, value in zip(groups, gains, strict=True):
        grouped[group].append(value)
    group_means = {
        label: {
            metric: float(
                np.mean([item[metric] for item in grouped[label]], dtype=np.float64)
            )
            for metric in ("C", "D", "A")
        }
        for label in labels
    }
    indices = bootstrap_indices(labels)
    metrics = {
        metric: grouped_stats([item[metric] for item in gains], groups, indices)
        for metric in ("C", "D", "A")
    }
    joint = int(
        sum(
            all(group_means[label][metric] > 0.0 for metric in ("C", "D", "A"))
            for label in labels
        )
    )
    return {
        "records": [
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "gain": value,
            }
            for row, value in zip(records, gains, strict=True)
        ],
        "metrics": metrics,
        "joint_positive_count": joint,
        "mean_C": metrics["C"]["mean"],
    }


def gain_pass(summary: dict[str, Any]) -> bool:
    return bool(
        summary["mean_C"] > 0.05
        and summary["joint_positive_count"] >= 7
        and all(
            summary["metrics"][metric]["ci99"][0] > 0.0 for metric in ("C", "D", "A")
        )
    )


def matched_curve(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    result = []
    for row in U_ROWS:
        indices = row + np.arange(lag_start, lag_start + 31, dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= audio32.shape[0]:
            raise rt.ProtocolError("matched lag domain is outside the audio embedding")
        difference = visual32[row : row + 1] - audio32[indices]
        result.append(
            np.mean(
                np.sqrt(
                    np.sum(
                        np.square(difference + np.float32(1e-6), dtype=np.float32),
                        axis=1,
                    ),
                    dtype=np.float64,
                )
            )
        )
    return np.asarray(result, dtype=np.float64)


def parent_control_gate(parent_scores: Path, parent_drivers: Path) -> dict[str, Any]:
    """Recompute the frozen parent control without producer statistics."""
    manifest = rt.load_self(parent_scores)
    driver_rows = rt.load_self(parent_drivers).get("rows", [])
    groups = {str(row["sample_id"]): str(row["source_group"]) for row in driver_rows}
    rows = manifest.get("rows", [])
    if len(rows) != 36:
        raise rt.ProtocolError(f"expected 36 parent control rows, got {len(rows)}")
    lookup = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row
        for row in rows
    }
    records = []
    for sample_id, group in sorted(groups.items(), key=lambda item: (item[1], item[0])):
        natural = lookup[(sample_id, "N", "N")]
        delayed = lookup[(sample_id, "N", "A_DELAY")]
        visual_n, audio_n, cached_n = rt.load_worker_arrays(natural)
        visual_d, audio_d, cached_d = rt.load_worker_arrays(delayed)
        if not np.array_equal(visual_n, visual_d):
            raise rt.ProtocolError(f"delayed visual embedding differs: {sample_id}")
        if (
            np.max(np.abs(matrix_from_embeddings(visual_n, audio_n) - cached_n)) > 1e-4
            or np.max(np.abs(matrix_from_embeddings(visual_d, audio_d) - cached_d))
            > 1e-4
        ):
            raise rt.ProtocolError(f"parent matrix cannot be rebuilt: {sample_id}")
        n_curve = matched_curve(visual_n, audio_n, -15)
        d_curve = matched_curve(visual_n, audio_d, -10)
        n_index = int(np.argmin(n_curve))
        d_index = int(np.argmin(d_curve))
        old_n = score_metrics(cached_n)
        old_d = score_metrics(cached_d)
        anchor = float(
            np.mean(cached_d[np.asarray(U_ROWS), old_n["min_index"]])
            - np.mean(cached_n[np.asarray(U_ROWS), old_n["min_index"]])
        )
        records.append(
            {
                "sample_id": sample_id,
                "source_group": group,
                "old_difference": int(old_d["offset"] - old_n["offset"]),
                "matched_difference": int((10 - d_index) - (15 - n_index)),
                "anchor_damage": anchor,
            }
        )
    old_count = sum(-6 <= row["old_difference"] <= -4 for row in records)
    matched_count = sum(-6 <= row["matched_difference"] <= -4 for row in records)
    damage_values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        damage_values[row["source_group"]].append(float(row["anchor_damage"]))
    damage_labels = sorted(damage_values)
    damage_means = np.asarray(
        [np.mean(damage_values[label], dtype=np.float64) for label in damage_labels],
        dtype=np.float64,
    )
    damage_indices = np.random.Generator(np.random.PCG64(20260909)).integers(
        0, 8, size=(10_000, 8)
    )
    damage_estimates = damage_means[damage_indices].mean(axis=1)
    damage = {
        "mean": float(damage_means.mean()),
        "ci95": [
            float(np.quantile(damage_estimates, 0.025, method="linear")),
            float(np.quantile(damage_estimates, 0.975, method="linear")),
        ],
        "group_labels": damage_labels,
        "group_means": {
            label: float(value)
            for label, value in zip(damage_labels, damage_means, strict=True)
        },
        "group_positive_count": int(np.sum(damage_means > 0)),
        "draws": 10_000,
        "seed": 20260909,
        "indices_sha256": hashlib.sha256(damage_indices.tobytes()).hexdigest(),
        "indices": damage_indices.tolist(),
    }
    parity = [
        row for row in rows if str(row.get("video_arm", "")).startswith("PARITY_")
    ]
    parity_ok = len(parity) == 2 and all(
        bool(row.get("parity", {}).get("passes")) for row in parity
    )
    return {
        "status": "PASS"
        if matched_count >= 14
        and damage["ci95"][0] > 0.0
        and damage["group_positive_count"] >= 7
        and parity_ok
        else "CONTROL_FAILED",
        "old_offset_pass_count": old_count,
        "matched_offset_pass_count": matched_count,
        "anchor_damage": damage,
        "parity_ok": parity_ok,
        "records": records,
    }
