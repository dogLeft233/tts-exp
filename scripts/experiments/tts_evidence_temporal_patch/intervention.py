from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol import (
    FRAME_RATE,
    MEL_HOP_SAMPLES,
    MEL_RATE,
    MEL_WIDTH,
    ProtocolError,
)


@dataclass(frozen=True)
class MelFrame:
    index: int
    start_column: int
    center_s: float
    support_start_samples: int
    support_end_samples: int
    tail_reused: bool


@dataclass
class DynamicBlock:
    """One independently smoothed and norm-matched continuous support block."""

    block_index: int
    recipient_indices: np.ndarray
    donor_indices: np.ndarray
    donor_index_float: np.ndarray
    u: np.ndarray
    v: np.ndarray
    v_scaled: np.ndarray
    delta: np.ndarray
    recipient_norm: float
    donor_norm: float
    scale: float
    mean_delta: np.ndarray

    @property
    def length(self) -> int:
        return int(self.recipient_indices.size)


def mel_chunk_starts(mel_columns: int, frame_count: int | None = None) -> list[int]:
    if mel_columns < MEL_WIDTH:
        raise ProtocolError(f"mel has fewer than {MEL_WIDTH} columns: {mel_columns}")
    if frame_count is None:
        count = 0
        starts: list[int] = []
        while True:
            nominal = int(math.floor(count * MEL_RATE / FRAME_RATE))
            actual = nominal
            if actual + MEL_WIDTH > mel_columns:
                actual = mel_columns - MEL_WIDTH
            starts.append(actual)
            count += 1
            if nominal + MEL_WIDTH > mel_columns:
                break
    else:
        if frame_count < 1:
            raise ProtocolError("frame_count must be positive")
        starts = []
        for index in range(frame_count):
            nominal = int(math.floor(index * MEL_RATE / FRAME_RATE))
            starts.append(min(nominal, mel_columns - MEL_WIDTH))
    return starts


def mel_frame_clock(mel_columns: int, frame_count: int | None = None) -> list[MelFrame]:
    starts = mel_chunk_starts(int(mel_columns), frame_count)
    result: list[MelFrame] = []
    for index, start in enumerate(starts):
        result.append(
            MelFrame(
                index=index,
                start_column=int(start),
                center_s=float((start + MEL_WIDTH / 2.0) / MEL_RATE),
                support_start_samples=int(start * MEL_HOP_SAMPLES),
                support_end_samples=int((start + MEL_WIDTH) * MEL_HOP_SAMPLES),
                tail_reused=bool(index > 0 and start == starts[index - 1]),
            )
        )
    return result


def mel_clock_from_starts(starts: Sequence[int]) -> list[MelFrame]:
    if not starts:
        raise ProtocolError("mel start sequence is empty")
    result: list[MelFrame] = []
    previous = -1
    for index, raw in enumerate(starts):
        start = int(raw)
        if start < 0 or start < previous:
            raise ProtocolError("mel starts must be non-negative and monotone")
        result.append(
            MelFrame(
                index=index,
                start_column=start,
                center_s=float((start + MEL_WIDTH / 2.0) / MEL_RATE),
                support_start_samples=int(start * MEL_HOP_SAMPLES),
                support_end_samples=int((start + MEL_WIDTH) * MEL_HOP_SAMPLES),
                tail_reused=bool(index > 0 and start == previous),
            )
        )
        previous = start
    return result


def _map_one(value: float, segments: Sequence[Mapping[str, Any]], direction: str) -> tuple[float, int] | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    if direction not in {"n_to_t", "t_to_n"}:
        raise ValueError("direction must be n_to_t or t_to_n")
    source_start = "n_start_s" if direction == "n_to_t" else "t_start_s"
    source_end = "n_end_s" if direction == "n_to_t" else "t_end_s"
    target_start = "t_start_s" if direction == "n_to_t" else "n_start_s"
    target_end = "t_end_s" if direction == "n_to_t" else "n_end_s"
    for index, segment in enumerate(segments):
        left = float(segment[source_start])
        right = float(segment[source_end])
        last = index == len(segments) - 1
        if left <= value < right or (last and math.isclose(value, right, abs_tol=1e-9)):
            ratio = (value - left) / (right - left)
            return float(float(segment[target_start]) + ratio * (float(segment[target_end]) - float(segment[target_start]))), index
    return None


def _block_for_segment(segment_index: int, support_blocks: Sequence[Mapping[str, Any]] | None) -> int | None:
    if support_blocks is None:
        return segment_index
    for index, block in enumerate(support_blocks):
        if segment_index in {int(item) for item in block.get("segment_indices", [])}:
            return index
    return None


def _interpolate_center(value: float, centers: np.ndarray) -> tuple[int, int, float] | None:
    if centers.ndim != 1 or centers.size == 0 or not np.isfinite(centers).all():
        return None
    if value < float(centers[0]) - 1e-9 or value > float(centers[-1]) + 1e-9:
        return None
    right = int(np.searchsorted(centers, value, side="left"))
    if right == 0:
        return 0, 0, 0.0
    if right >= centers.size:
        if math.isclose(value, float(centers[-1]), abs_tol=1e-9):
            last = int(centers.size - 1)
            return last, last, 0.0
        return None
    left = right - 1
    denominator = float(centers[right] - centers[left])
    if denominator <= 0:
        raise ProtocolError("frame centers are not strictly increasing")
    alpha = float((value - float(centers[left])) / denominator)
    return left, right, alpha


def build_frame_map(
    recipient_centers: Sequence[float],
    donor_centers: Sequence[float],
    segments: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    support_blocks: Sequence[Mapping[str, Any]] | None = None,
    recipient_tail: Sequence[bool] | None = None,
    donor_tail: Sequence[bool] | None = None,
) -> list[dict[str, Any]]:
    """Map actual mel-window centers without gap interpolation or clamping."""

    if not segments:
        return []
    recipient = np.asarray(recipient_centers, dtype=np.float64)
    donor = np.asarray(donor_centers, dtype=np.float64)
    if recipient.ndim != 1 or donor.ndim != 1 or recipient.size == 0 or donor.size == 0:
        raise ProtocolError("recipient and donor frame centers must be non-empty vectors")
    if np.any(np.diff(recipient) < 0) or np.any(np.diff(donor) < 0):
        raise ProtocolError("frame centers must be monotone")
    result: list[dict[str, Any]] = []
    for index, center in enumerate(recipient):
        row: dict[str, Any] = {"recipient_index": int(index), "recipient_center_s": float(center), "status": "UNSUPPORTED"}
        if recipient_tail is not None and bool(recipient_tail[index]):
            row["reason"] = "RECIPIENT_TAIL"
            result.append(row)
            continue
        mapped = _map_one(float(center), segments, direction)
        if mapped is None:
            row["reason"] = "MAPPING_GAP_OR_BOUNDARY"
            result.append(row)
            continue
        donor_time, segment_index = mapped
        block_index = _block_for_segment(segment_index, support_blocks)
        if block_index is None:
            row["reason"] = "NO_CONTINUOUS_BLOCK"
            result.append(row)
            continue
        interpolated = _interpolate_center(donor_time, donor)
        if interpolated is None:
            row["reason"] = "DONOR_CENTER_OUT_OF_SUPPORT"
            result.append(row)
            continue
        left, right, alpha = interpolated
        if donor_tail is not None and (bool(donor_tail[left]) or bool(donor_tail[right])):
            row["reason"] = "DONOR_TAIL"
            result.append(row)
            continue
        row.update(
            {
                "status": "VALID",
                "donor_time_s": float(donor_time),
                "donor_index_float": float(left + alpha * (right - left)),
                "donor_left_index": int(left),
                "donor_right_index": int(right),
                "donor_alpha": float(alpha),
                "segment_index": int(segment_index),
                "block_index": int(block_index),
            }
        )
        result.append(row)
    return result


def continuous_core_blocks(frame_map: Sequence[Mapping[str, Any]], *, minimum_frames: int = 25) -> list[list[dict[str, Any]]]:
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for raw in frame_map:
        row = dict(raw)
        valid = row.get("status") == "VALID"
        adjacent = bool(
            previous is not None
            and valid
            and previous.get("status") == "VALID"
            and int(row["recipient_index"]) == int(previous["recipient_index"]) + 1
            and int(row["block_index"]) == int(previous["block_index"])
        )
        if not valid or (current and not adjacent):
            if len(current) >= minimum_frames:
                blocks.append(current)
            current = []
        if valid:
            current.append(row)
        previous = row
    if len(current) >= minimum_frames:
        blocks.append(current)
    return blocks


def _boxcar(values: np.ndarray, width: int) -> np.ndarray:
    if width < 1 or width % 2 == 0:
        raise ValueError("smoothing width must be a positive odd number")
    kernel = np.ones(width, dtype=np.float32) / np.float32(width)
    return np.stack([np.convolve(values[:, column], kernel, mode="same") for column in range(values.shape[1])], axis=1).astype(np.float32)


def _mapped_donor(values: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result: list[np.ndarray] = []
    for row in rows:
        left = int(row["donor_left_index"])
        right = int(row["donor_right_index"])
        alpha = float(row["donor_alpha"])
        result.append((1.0 - alpha) * values[left] + alpha * values[right])
    return np.asarray(result, dtype=np.float32)


def decompose_dynamic(
    recipient: np.ndarray,
    donor: np.ndarray,
    frame_map: Sequence[Mapping[str, Any]],
    *,
    smoothing_frames: int = 5,
    minimum_core_frames: int = 25,
) -> list[DynamicBlock]:
    """Build aligned short-time residuals while preserving the recipient base."""

    recipient_value = np.asarray(recipient, dtype=np.float32)
    donor_value = np.asarray(donor, dtype=np.float32)
    if recipient_value.ndim != 2 or donor_value.ndim != 2 or recipient_value.shape[1] != 512 or donor_value.shape[1] != 512:
        raise ProtocolError("bottleneck arrays must have shape (frames, 512)")
    if not np.isfinite(recipient_value).all() or not np.isfinite(donor_value).all():
        raise ProtocolError("bottleneck arrays contain non-finite values")
    if smoothing_frames % 2 == 0 or smoothing_frames < 1:
        raise ValueError("smoothing_frames must be a positive odd number")
    blocks = continuous_core_blocks(frame_map, minimum_frames=minimum_core_frames)
    output: list[DynamicBlock] = []
    half = smoothing_frames // 2
    for block_index, full_rows in enumerate(blocks):
        if len(full_rows) < minimum_core_frames + 2 * half:
            continue
        # Smooth over the full continuous block first; trimming the edges is
        # what prevents the convolution kernel from crossing a support gap.
        r_indices = np.asarray([int(row["recipient_index"]) for row in full_rows], dtype=np.int64)
        r_values = recipient_value[r_indices]
        d_values = _mapped_donor(donor_value, full_rows)
        r_smooth = _boxcar(r_values, smoothing_frames)
        d_smooth = _boxcar(d_values, smoothing_frames)
        core_rows = full_rows[half:-half] if half else full_rows
        core_start = half
        core_end = len(full_rows) - half if half else len(full_rows)
        r_core = r_values[core_start:core_end]
        r_low = r_smooth[core_start:core_end]
        d_low = d_smooth[core_start:core_end]
        u = r_core - r_low
        v = d_values[core_start:core_end] - d_low
        u = u - np.mean(u, axis=0, keepdims=True, dtype=np.float32)
        v = v - np.mean(v, axis=0, keepdims=True, dtype=np.float32)
        u_norm = float(np.linalg.norm(u.astype(np.float64)))
        v_norm = float(np.linalg.norm(v.astype(np.float64)))
        if u_norm <= 1e-8 or v_norm <= 1e-8:
            continue
        scale = u_norm / v_norm
        v_scaled = (v * np.float32(scale)).astype(np.float32)
        delta = (v_scaled - u).astype(np.float32)
        output.append(
            DynamicBlock(
                block_index=int(block_index),
                recipient_indices=r_indices[core_start:core_end],
                donor_indices=np.asarray([int(row["donor_left_index"]) for row in core_rows], dtype=np.int64),
                donor_index_float=np.asarray([float(row["donor_index_float"]) for row in core_rows], dtype=np.float32),
                u=u.astype(np.float32),
                v=v.astype(np.float32),
                v_scaled=v_scaled,
                delta=delta,
                recipient_norm=u_norm,
                donor_norm=v_norm,
                scale=float(scale),
                mean_delta=np.mean(delta, axis=0, dtype=np.float64).astype(np.float32),
            )
        )
    return output


def _block_array(block: DynamicBlock | Mapping[str, Any], name: str) -> np.ndarray:
    value = getattr(block, name, None) if isinstance(block, DynamicBlock) else block.get(name)
    if value is None:
        raise ProtocolError(f"dynamic block lacks {name}")
    return np.asarray(value, dtype=np.float32)


def _block_indices(block: DynamicBlock | Mapping[str, Any]) -> np.ndarray:
    value = getattr(block, "recipient_indices", None) if isinstance(block, DynamicBlock) else block.get("recipient_indices")
    if value is None:
        raise ProtocolError("dynamic block lacks recipient_indices")
    return np.asarray(value, dtype=np.int64)


def build_patch(
    recipient: np.ndarray,
    blocks: Sequence[DynamicBlock | Mapping[str, Any]],
    condition: str,
    *,
    lam: float = 0.5,
) -> np.ndarray:
    """Return one condition while leaving every non-core frame bitwise equal."""

    base = np.asarray(recipient, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 512:
        raise ProtocolError("recipient bottleneck must have shape (frames, 512)")
    condition = str(condition).upper()
    if condition not in {"BASE", "COHERENT", "SCRAMBLED", "ERASE"}:
        raise ValueError(f"unknown patch condition: {condition}")
    if not math.isfinite(float(lam)) or lam < 0:
        raise ValueError("lambda must be finite and non-negative")
    result = base.copy()
    if condition == "BASE":
        return result
    for block in blocks:
        indices = _block_indices(block)
        u = _block_array(block, "u")
        delta = _block_array(block, "delta")
        if len(indices) != len(u) or len(indices) != len(delta):
            raise ProtocolError("dynamic block arrays have inconsistent lengths")
        if np.any(indices < 0) or np.any(indices >= len(result)):
            raise ProtocolError("dynamic block index is outside recipient bottleneck")
        if condition == "COHERENT":
            adjustment = delta
        elif condition == "SCRAMBLED":
            shift = len(delta) // 2
            adjustment = np.roll(delta, shift=shift, axis=0)
        else:
            adjustment = -u
        result[indices] = (result[indices] + np.float32(lam) * adjustment).astype(np.float32)
    if not np.isfinite(result).all():
        raise ProtocolError("patch produced non-finite bottleneck values")
    return result


def patch_metadata(
    blocks: Sequence[DynamicBlock | Mapping[str, Any]],
    *,
    condition: str,
    lam: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        indices = _block_indices(block)
        delta = _block_array(block, "delta")
        u = _block_array(block, "u")
        if isinstance(block, DynamicBlock):
            block_index = int(block.block_index)
        else:
            block_index = int(block.get("block_index", len(rows)))
        if condition.upper() == "SCRAMBLED":
            permutation = np.roll(np.arange(len(delta), dtype=np.int64), len(delta) // 2)
        else:
            permutation = np.arange(len(delta), dtype=np.int64)
        rows.append(
            {
                "block_index": block_index,
                "recipient_indices": indices.tolist(),
                "permutation": permutation.tolist(),
                "length": int(len(indices)),
                "u_norm": float(np.linalg.norm(u.astype(np.float64))),
                "delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
                "delta_mean_abs_max": float(np.max(np.abs(np.mean(delta, axis=0)))) if len(delta) else 0.0,
                "lambda": float(lam),
            }
        )
    return {"condition": condition.upper(), "lambda": float(lam), "blocks": rows}


def validate_patch(
    base: np.ndarray,
    patched: np.ndarray,
    blocks: Sequence[DynamicBlock | Mapping[str, Any]],
    condition: str,
    *,
    lam: float = 0.5,
    atol: float = 1e-5,
) -> dict[str, Any]:
    base_value = np.asarray(base, dtype=np.float32)
    patch_value = np.asarray(patched, dtype=np.float32)
    if base_value.shape != patch_value.shape:
        raise ProtocolError("base/patched bottleneck shape mismatch")
    expected = build_patch(base_value, blocks, condition, lam=lam)
    if not np.allclose(expected, patch_value, rtol=0.0, atol=atol):
        raise ProtocolError("patched bottleneck does not match the registered transform")
    core = {int(index) for block in blocks for index in _block_indices(block)}
    outside = [index for index in range(len(base_value)) if index not in core]
    if outside and not np.array_equal(base_value[outside], patch_value[outside]):
        raise ProtocolError("patch changed frames outside the registered core")
    return {
        "status": "PASS",
        "condition": str(condition).upper(),
        "core_frame_count": len(core),
        "outside_bitwise_equal": True,
        "artifact_sha256": None,
    }


def shift_frame_map(frame_map: Sequence[Mapping[str, Any]], shift: int, *, donor_count: int | None = None) -> list[dict[str, Any]]:
    """Shift donor indices for calibration controls without wrapping/filling."""

    result: list[dict[str, Any]] = []
    for row in frame_map:
        value = dict(row)
        if value.get("status") == "VALID":
            left = int(value["donor_left_index"]) + int(shift)
            right = int(value["donor_right_index"]) + int(shift)
            if left < 0 or right < 0 or (donor_count is not None and right >= int(donor_count)):
                value.update(status="UNSUPPORTED", reason="SHIFT_OUT_OF_RANGE")
            else:
                value["donor_left_index"] = left
                value["donor_right_index"] = right
                value["donor_index_float"] = float(value["donor_index_float"]) + float(shift)
        result.append(value)
    return result
