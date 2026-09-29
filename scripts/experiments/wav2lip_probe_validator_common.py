"""Numerical primitives used only by the independent probe validators.

This module deliberately does not import producer construction, support, or
decision functions.  It reads arrays supplied by a validator and recomputes
the fixed SyncNet endpoint and grouped bootstrap rules from scratch.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_probe_runtime import ProtocolError

U_ROWS = tuple(range(30, 58))
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_DRAWS = 20_000


def matrix_from_embeddings(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Rebuild the cached SyncNet distance matrix independently.

    The producer emits float32 embeddings and performs the distance reduction
    in float32.  Keeping that arithmetic here makes the validator check the
    actual cache contents rather than trusting the producer's matrix file.
    """
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    if (
        visual32.ndim != 2
        or audio32.ndim != 2
        or visual32.shape != audio32.shape
        or visual32.shape[0] < 88
        or visual32.shape[1] != 1024
    ):
        raise ProtocolError(
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
        raise ProtocolError(f"invalid SyncNet matrix: {value.shape}")
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
        raise ProtocolError("group bootstrap requires eight labels")
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
        raise ProtocolError("grouped statistic requires eight groups x two records")
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


def validate_fresh_controls(
    rows: list[dict[str, Any]], parent_rows: list[dict[str, Any]]
) -> None:
    expected_parent = {}
    for row in parent_rows:
        arm = str(row.get("video_arm"))
        if arm == "N_REPEAT" or arm.startswith("PARITY_"):
            expected_parent[(str(row["sample_id"]), f"FRESH_{arm}")] = row
    if len(rows) != 4 or {
        (str(row["sample_id"]), str(row["video_arm"])) for row in rows
    } != set(expected_parent):
        raise ProtocolError("fresh control identity mismatch")
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        parent = expected_parent[key]
        parent_score = parent.get("score", parent)
        reference = Path(
            str(parent.get("parity", {}).get("actual_matrix", parent_score["matrix"]))
        ).resolve()
        actual_path = Path(str(row["score"]["matrix"]))
        actual = np.asarray(np.load(actual_path, allow_pickle=False), dtype=np.float64)
        expected = np.asarray(np.load(reference, allow_pickle=False), dtype=np.float64)
        if (
            actual.shape != expected.shape
            or float(np.max(np.abs(actual - expected))) > 1e-4
        ):
            raise ProtocolError(f"fresh control matrix mismatch: {key}")
        if hashlib.sha256(actual_path.read_bytes()).hexdigest() != str(
            row["score"].get("matrix_sha256")
        ):
            raise ProtocolError(f"fresh control matrix hash mismatch: {key}")
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        if (
            str(parent.get("parity", {}).get("actual_matrix_sha256", reference_hash))
            != reference_hash
        ):
            raise ProtocolError(f"fresh control reference hash mismatch: {key}")
        actual_metrics = score_metrics(actual)
        expected_metrics = score_metrics(expected)
        endpoint = max(
            abs(left - right)
            for left, right in zip(
                actual_metrics["curve"], expected_metrics["curve"], strict=True
            )
        )
        if endpoint > 1e-6 or actual_metrics["offset"] != expected_metrics["offset"]:
            raise ProtocolError(f"fresh control endpoint mismatch: {key}")
