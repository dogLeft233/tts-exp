from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    SILENCE_LABELS,
    FrameMapping,
    build_frame_mapping,
    frame_owners,
    validate_tokens,
)


class DTWError(ValueError):
    pass


CENTER_EXCLUSIVE_FRAME_OWNERSHIP = "center_exclusive_v1"
NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP = "nominal_half_stride_overlap_shared_v1"


def phone_frame_indices(
    tokens: Sequence[Mapping[str, Any]],
    frame_count: int,
    *,
    frame_stride_samples: int = 320,
    sample_rate: int = 16_000,
    policy: str = CENTER_EXCLUSIVE_FRAME_OWNERSHIP,
) -> list[list[int]]:
    if frame_count <= 0 or frame_stride_samples <= 0 or sample_rate <= 0:
        raise DTWError("frame count, stride, and sample rate must be positive")
    normalized = validate_tokens(tokens)
    if policy == CENTER_EXCLUSIVE_FRAME_OWNERSHIP:
        owners = frame_owners(
            frame_count,
            tokens,
            frame_stride_samples=frame_stride_samples,
            sample_rate=sample_rate,
        )
        return [
            [owner.frame_index for owner in owners if owner.token_index == token_index]
            for token_index in range(len(normalized))
        ]
    if policy != NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP:
        raise DTWError(f"unknown frame ownership policy: {policy}")

    half_stride_s = frame_stride_samples / (2.0 * sample_rate)
    indices_by_token: list[list[int]] = [[] for _ in normalized]
    for frame_index in range(frame_count):
        centre_s = (frame_index + 0.5) * frame_stride_samples / sample_rate
        support_start = centre_s - half_stride_s
        support_end = centre_s + half_stride_s
        for token_index, token in enumerate(normalized):
            overlap = min(token.end_s, support_end) - max(token.start_s, support_start)
            if overlap > 1e-12:
                indices_by_token[token_index].append(frame_index)
    return indices_by_token


@dataclass(frozen=True)
class HardDTWResult:
    path: tuple[tuple[int, int], ...]
    total_cost: float
    mean_cost: float
    vertical_steps: int
    horizontal_steps: int
    diagonal_steps: int


def cosine_cost_matrix(natural: np.ndarray, tts: np.ndarray) -> np.ndarray:
    left = np.asarray(natural, dtype=np.float64)
    right = np.asarray(tts, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or not left.shape[0] or not right.shape[0]:
        raise DTWError("feature arrays must be non-empty [frames,dim] with equal dimensions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise DTWError("feature arrays must be finite")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    if np.any(left_norm == 0.0) or np.any(right_norm == 0.0):
        raise DTWError("feature arrays contain a zero vector")
    similarity = (left / left_norm[:, None]) @ (right / right_norm[:, None]).T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _in_band(i: int, j: int, n: int, m: int, band_ratio: float) -> bool:
    if n == 1 or m == 1:
        return True
    return abs(i / (n - 1) - j / (m - 1)) <= band_ratio


def hard_dtw_path(cost: np.ndarray, *, band_ratio: float = 0.5) -> HardDTWResult:
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise DTWError("cost must have non-empty [N,M] shape")
    if not np.isfinite(matrix).all():
        raise DTWError("cost must be finite")
    if not 0.0 <= band_ratio <= 1.0:
        raise DTWError("band_ratio must be in [0,1]")
    n, m = matrix.shape
    dp = np.full((n, m), np.inf, dtype=np.float64)
    back = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if not _in_band(i, j, n, m, band_ratio):
                continue
            if i == 0 and j == 0:
                dp[i, j] = matrix[i, j]
                continue
            candidates: list[tuple[float, int]] = []
            if i > 0 and math.isfinite(float(dp[i - 1, j])):
                candidates.append((float(dp[i - 1, j]), 0))
            if j > 0 and math.isfinite(float(dp[i, j - 1])):
                candidates.append((float(dp[i, j - 1]), 1))
            if i > 0 and j > 0 and math.isfinite(float(dp[i - 1, j - 1])):
                candidates.append((float(dp[i - 1, j - 1]), 2))
            if candidates:
                best_value, best_step = min(candidates, key=lambda item: item[0])
                dp[i, j] = matrix[i, j] + best_value
                back[i, j] = best_step
    if not math.isfinite(float(dp[-1, -1])):
        raise DTWError("band constraint admits no complete DTW path")
    reversed_path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while True:
        reversed_path.append((i, j))
        if i == 0 and j == 0:
            break
        step = int(back[i, j])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        elif step == 2:
            i -= 1
            j -= 1
        else:
            raise DTWError("invalid DTW backpointer")
    path = tuple(reversed(reversed_path))
    steps = [(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]) for index in range(1, len(path))]
    vertical = sum(step == (1, 0) for step in steps)
    horizontal = sum(step == (0, 1) for step in steps)
    diagonal = sum(step == (1, 1) for step in steps)
    path_costs = [float(matrix[i, j]) for i, j in path]
    return HardDTWResult(path, float(dp[-1, -1]), float(np.mean(path_costs)), vertical, horizontal, diagonal)


def _coordinate_row(natural_frame_index: int, natural_token_index: int, natural_label: str, natural_silence: bool, tts_token_index: int | None, coordinate: float, tts_frame_count: int, mapping_type: str) -> FrameMapping:
    if not math.isfinite(coordinate) or coordinate < 0.0 or coordinate > max(0, tts_frame_count - 1):
        raise DTWError("DTW source coordinate is outside the TTS feature sequence")
    left = math.floor(coordinate)
    right = min(tts_frame_count - 1, left + 1)
    return FrameMapping(
        natural_frame_index=natural_frame_index,
        natural_token_index=natural_token_index,
        natural_label=natural_label,
        natural_silence=natural_silence,
        mapping_type=mapping_type,
        tts_token_index=tts_token_index,
        left_frame_index=left,
        right_frame_index=right,
        interpolation_alpha=float(coordinate - left),
        fallback_reason=None,
    )


def _path_to_coordinates(path: Sequence[tuple[int, int]], natural_indices: Sequence[int], tts_indices: Sequence[int]) -> dict[int, float]:
    visited: dict[int, list[int]] = {index: [] for index in range(len(natural_indices))}
    for local_natural, local_tts in path:
        visited[local_natural].append(local_tts)
    if any(not values for values in visited.values()):
        raise DTWError("DTW path does not cover every natural frame")
    coordinates = {
        natural_indices[local_natural]: float(np.mean([tts_indices[index] for index in local_tts]))
        for local_natural, local_tts in visited.items()
    }
    values = [coordinates[index] for index in natural_indices]
    if any(right < left for left, right in pairwise(values)):
        raise DTWError("DTW source coordinates are not monotone")
    return coordinates


def build_hard_dtw_mapping(
    natural_features: np.ndarray,
    tts_features: np.ndarray,
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    *,
    band_ratio: float = 0.5,
    frame_ownership_policy: str = CENTER_EXCLUSIVE_FRAME_OWNERSHIP,
) -> tuple[list[FrameMapping], dict[str, Any], dict[str, Any]]:
    natural = np.asarray(natural_features, dtype=np.float32)
    tts = np.asarray(tts_features, dtype=np.float32)
    linear, linear_stats = build_frame_mapping(natural.shape[0], tts.shape[0], natural_tokens, tts_tokens)
    natural_owners = frame_owners(natural.shape[0], natural_tokens)
    tts_frames_by_token = phone_frame_indices(
        tts_tokens,
        tts.shape[0],
        policy=frame_ownership_policy,
    )
    tts_frame_use_count = [0] * tts.shape[0]
    for indices in tts_frames_by_token:
        for frame_index in indices:
            tts_frame_use_count[frame_index] += 1
    rows_by_natural = {row.natural_frame_index: row for row in linear}
    output: list[FrameMapping] = []
    phones: list[dict[str, Any]] = []
    for natural_token_index, natural_token in enumerate(natural_tokens):
        token_natural_indices = [owner.frame_index for owner in natural_owners if owner.token_index == natural_token_index]
        linear_rows = [rows_by_natural[index] for index in token_natural_indices]
        if not token_natural_indices:
            continue
        if bool(natural_token.get("silence", str(natural_token.get("label", natural_token.get("token", ""))).lower() in SILENCE_LABELS)):
            output.extend(linear_rows)
            continue
        tts_token_indices = {row.tts_token_index for row in linear_rows}
        if len(tts_token_indices) != 1 or None in tts_token_indices:
            raise DTWError(f"speech phone {natural_token_index} has no unique matched TTS instance")
        tts_token_index = next(iter(tts_token_indices))
        token_tts_indices = tts_frames_by_token[tts_token_index]
        if not token_tts_indices:
            raise DTWError(
                f"speech phone {natural_token_index} has no TTS feature support under {frame_ownership_policy}"
            )
        local_natural = natural[token_natural_indices]
        local_tts = tts[token_tts_indices]
        result = hard_dtw_path(cosine_cost_matrix(local_natural, local_tts), band_ratio=band_ratio)
        coordinates = _path_to_coordinates(result.path, token_natural_indices, token_tts_indices)
        label = str(natural_token.get("label", natural_token.get("token", ""))).lower()
        for frame_index in token_natural_indices:
            output.append(_coordinate_row(frame_index, natural_token_index, label, False, tts_token_index, coordinates[frame_index], tts.shape[0], "hard_dtw"))
        phones.append({
            "natural_token_index": natural_token_index,
            "tts_token_index": tts_token_index,
            "label": label,
            "natural_frame_indices": list(token_natural_indices),
            "tts_frame_indices": list(token_tts_indices),
            "tts_frame_support_policy": frame_ownership_policy,
            "tts_frame_support_shared": bool(any(tts_frame_use_count[index] > 1 for index in token_tts_indices)),
            "path": [[int(i), int(j)] for i, j in result.path],
            "path_cost_total": result.total_cost,
            "path_cost_mean": result.mean_cost,
            "vertical_steps": result.vertical_steps,
            "horizontal_steps": result.horizontal_steps,
            "diagonal_steps": result.diagonal_steps,
            "source_coordinates": [coordinates[index] for index in token_natural_indices],
            "linear_coordinates": [
                float(row.left_frame_index + row.interpolation_alpha) for row in linear_rows
            ],
        })
    output.sort(key=lambda row: row.natural_frame_index)
    if [row.natural_frame_index for row in output] != list(range(natural.shape[0])):
        raise DTWError("DTW mapping does not cover every natural frame")
    if any(row.natural_silence and row.mapping_type != linear[row.natural_frame_index].mapping_type for row in output):
        raise DTWError("silence mapping policy changed")
    diagnostics = {
        "natural_frame_count": int(natural.shape[0]),
        "tts_frame_count": int(tts.shape[0]),
        "matched_frames": int(sum(row.mapping_type == "hard_dtw" for row in output)),
        "silence_frames": int(sum(row.natural_silence for row in output)),
        "speech_phone_count": len(phones),
        "trivial_phone_count": int(sum(len(phone["path"]) == 1 for phone in phones)),
        "total_path_steps": int(sum(len(phone["path"]) for phone in phones)),
        "frame_ownership_policy": frame_ownership_policy,
        "tts_shared_frame_count": int(sum(count > 1 for count in tts_frame_use_count)),
        "tts_unassigned_frame_count": int(sum(count == 0 for count in tts_frame_use_count)),
        "mean_path_cost": float(np.mean([phone["path_cost_mean"] for phone in phones])) if phones else 0.0,
        "max_coordinate_displacement": float(max(
            abs(float(row.left_frame_index + row.interpolation_alpha) - float(linear[row.natural_frame_index].left_frame_index + linear[row.natural_frame_index].interpolation_alpha))
            for row in output
        )),
        "linear_mapping": linear_stats,
    }
    trace = {
        "schema_version": 1,
        "band_ratio": float(band_ratio),
        "frame_ownership_policy": frame_ownership_policy,
        "frames": [
            {
                "natural_frame_index": row.natural_frame_index,
                "natural_token_index": row.natural_token_index,
                "natural_label": row.natural_label,
                "natural_silence": row.natural_silence,
                "mapping_type": row.mapping_type,
                "tts_token_index": row.tts_token_index,
                "left_frame_index": row.left_frame_index,
                "right_frame_index": row.right_frame_index,
                "interpolation_alpha": row.interpolation_alpha,
                "fallback_reason": row.fallback_reason,
            }
            for row in output
        ],
        "phones": phones,
        "diagnostics": diagnostics,
    }
    return output, diagnostics, trace
