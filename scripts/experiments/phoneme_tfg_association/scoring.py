"""Dynamic-length SyncNet scoring and C/D curve summaries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import CONDITIONS, PROTOCOL_ID, ProtocolError, file_sha256, read_json, write_json, write_jsonl


def equal_count_rows(interior_count: int, target_count: int) -> list[int]:
    if interior_count < 1 or target_count < 1 or target_count > interior_count:
        raise ProtocolError("equal-count target is invalid")
    if target_count == interior_count:
        return list(range(interior_count))
    result = np.floor(np.linspace(0, interior_count - 1, target_count)).astype(np.int64).tolist()
    if len(set(result)) != target_count:
        raise ProtocolError("equal-count rows contain duplicates")
    return [int(item) for item in result]


def support_indices(length: int, support: str, *, vshift: int = 15, minimum_interior_windows: int = 25) -> list[int]:
    if length < 1:
        raise ProtocolError("empty SyncNet matrix")
    if support == "FULL":
        rows = list(range(length))
    elif support == "INTERIOR":
        rows = list(range(vshift, length - vshift))
    else:
        raise ProtocolError(f"unknown single-matrix support: {support}")
    if support == "INTERIOR" and len(rows) < minimum_interior_windows:
        raise ProtocolError(f"INTERIOR support has fewer than {minimum_interior_windows} rows: {length}")
    return rows


def _finite_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise ProtocolError(f"{label} must be finite [T,31], got {value.shape}")
    return value


def summarize_curve(matrix: np.ndarray, rows: Sequence[int], *, vshift: int = 15, label: str = "curve") -> dict[str, Any]:
    value = _finite_matrix(matrix, label)
    selected = np.asarray(list(rows), dtype=np.int64)
    if selected.size < 1 or int(selected.min()) < 0 or int(selected.max()) >= value.shape[0]:
        raise ProtocolError(f"{label} support is out of range")
    curve = np.asarray(value[selected].astype(np.float32).mean(axis=0, dtype=np.float32), dtype=np.float64)
    min_index = int(np.argmin(curve))
    d_value = float(curve[min_index])
    b_value = float(np.median(curve))
    c_value = b_value - d_value
    d0 = float(curve[vshift])
    half = d_value + c_value / 2.0
    if c_value <= 1e-6:
        width_ms = None
        width_status = "flat_curve"
    else:
        left = min_index
        right = min_index
        while left > 0 and float(curve[left - 1]) <= half:
            left -= 1
        while right + 1 < len(curve) and float(curve[right + 1]) <= half:
            right += 1
        width_ms = float((right - left + 1) * 40.0)
        width_status = "censored" if left == 0 or right == len(curve) - 1 else "complete"
    c5 = float(np.median(curve[10:21]) - np.min(curve[10:21]))
    return {
        "support_rows": [int(item) for item in selected],
        "support_count": int(selected.size),
        "curve": [float(item) for item in curve],
        "offsets": [vshift - index for index in range(31)],
        "min_index": min_index,
        "official_offset": vshift - min_index,
        "sync_d": d_value,
        "sync_c": c_value,
        "background_b": b_value,
        "d0": d0,
        "search_gain_s": d0 - d_value,
        "c5": c5,
        "boundary_best": bool(min_index in (0, 30)),
        "trough_width_ms": width_ms,
        "trough_width_status": width_status,
        "trough_ties": [int(index) for index, item in enumerate(curve) if float(item) == d_value],
    }


def summarize_pair_matrices(natural: np.ndarray, tts: Mapping[str, np.ndarray], *, vshift: int = 15, minimum_interior_windows: int = 25) -> list[dict[str, Any]]:
    """Return arm/support rows using only real matrix rows; no padding."""
    n_value = _finite_matrix(natural, "natural")
    t_values = {str(arm): _finite_matrix(matrix, f"tts/{arm}") for arm, matrix in tts.items()}
    if not t_values:
        raise ProtocolError("no TTS matrices")
    n_interior = support_indices(len(n_value), "INTERIOR", vshift=vshift, minimum_interior_windows=minimum_interior_windows)
    t_interior = {arm: support_indices(len(matrix), "INTERIOR", vshift=vshift, minimum_interior_windows=minimum_interior_windows) for arm, matrix in t_values.items()}
    rows_by_support: dict[str, dict[str, list[int]]] = {
        "FULL": {"natural": list(range(len(n_value))), **{arm: list(range(len(matrix))) for arm, matrix in t_values.items()}},
        "INTERIOR": {"natural": n_interior, **t_interior},
    }
    target = min([len(n_interior), *[len(rows) for rows in t_interior.values()]])
    rows_by_support["EQUAL_COUNT"] = {
        "natural": [n_interior[index] for index in equal_count_rows(len(n_interior), target)],
        **{arm: [rows[index] for index in equal_count_rows(len(rows), target)] for arm, rows in t_interior.items()},
    }
    output: list[dict[str, Any]] = []
    for support, row_map in rows_by_support.items():
        n_metrics = summarize_curve(n_value, row_map["natural"], vshift=vshift, label=f"natural/{support}")
        output.append({"condition": "natural", "support": support, **n_metrics})
        for arm, matrix in t_values.items():
            output.append({"condition": arm, "support": support, **summarize_curve(matrix, row_map[arm], vshift=vshift, label=f"{arm}/{support}")})
    return output


def distance_from_embeddings(visual: np.ndarray, audio: np.ndarray, *, vshift: int = 15) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.ndim != 2 or audio_value.ndim != 2 or visual_value.shape != audio_value.shape:
        raise ProtocolError(f"embedding shape mismatch: {visual_value.shape}/{audio_value.shape}")
    if not np.isfinite(visual_value).all() or not np.isfinite(audio_value).all():
        raise ProtocolError("non-finite SyncNet embeddings")
    padded = np.pad(audio_value, ((vshift, vshift), (0, 0)), mode="constant")
    result = np.empty((len(visual_value), 2 * vshift + 1), dtype=np.float32)
    for row in range(len(visual_value)):
        difference = visual_value[row : row + 1] - padded[row : row + 2 * vshift + 1]
        result[row] = np.sqrt(np.sum((difference + 1e-6) ** 2, axis=1)).astype(np.float32)
    return result


def score_native(
    cell: Mapping[str, Any],
    media: Path,
    source_audio: Path,
    output_root: Path,
    cfg: Mapping[str, Any],
    *,
    scorer: Any | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one independent SyncNet forward and persist its arrays/receipt."""
    if not media.is_file() or not source_audio.is_file():
        raise ProtocolError(f"missing media/audio: {media}/{source_audio}")
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

    if scorer is None:
        scorer = SyncNetScorer(
            Path(str(cfg["syncnet_model"])).resolve(),
            device=device,
            batch_size=int(cfg.get("syncnet_batch_size", 20)),
            threads=int(cfg.get("syncnet_threads", 4)),
        )
    out_dir = output_root / "04_syncnet" / str(cell["tfg"]) / str(cell["sample_id"]) / str(cell["arm"])
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_media_sha = file_sha256(media)
    reference = cell.get("reference")
    visual_box = None
    if isinstance(reference, Mapping) and reference.get("box_top_bottom_left_right") is not None:
        values = reference["box_top_bottom_left_right"]
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ProtocolError(f"invalid frozen visual box for {cell['cell_key']}: {values}")
        visual_box = tuple(int(value) for value in values)
    result = scorer.score(media, source_audio, out_dir, expected_media_sha, _pcm_hash(source_audio), visual_box)
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cell_key": cell["cell_key"],
        "sample_id": cell["sample_id"],
        "source_group": cell["source_group"],
        "tfg": cell["tfg"],
        "arm": cell["arm"],
        "status": "complete",
        "media": str(media.resolve()),
        "media_sha256": expected_media_sha,
        "source_audio": str(source_audio.resolve()),
        "source_audio_pcm_sha256": _pcm_hash(source_audio),
        "visual_box": list(visual_box) if visual_box is not None else None,
        "worker": result,
    }
    write_json(out_dir / "receipt.json", receipt)
    return receipt


def _pcm_hash(path: Path) -> str:
    import wave

    with wave.open(str(path), "rb") as handle:
        params = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        if params != (1, 2, 16000):
            raise ProtocolError(f"canonical audio is not PCM16/16k mono: {path}: {params}")
        pcm = handle.readframes(handle.getnframes())
    import hashlib

    return hashlib.sha256(pcm).hexdigest()


def score_static_control(visual: np.ndarray | Mapping[str, np.ndarray], audio_by_arm: Mapping[str, np.ndarray], *, vshift: int = 15, minimum_interior_windows: int = 25) -> list[dict[str, Any]]:
    """Summarize static-reference distances from already computed embeddings."""
    if isinstance(visual, Mapping):
        matrices = {arm: distance_from_embeddings(np.asarray(visual[arm]), audio, vshift=vshift) for arm, audio in audio_by_arm.items()}
    else:
        matrices = {arm: distance_from_embeddings(visual, audio, vshift=vshift) for arm, audio in audio_by_arm.items()}
    natural = matrices.pop("natural")
    return summarize_pair_matrices(natural, matrices, vshift=vshift, minimum_interior_windows=minimum_interior_windows)


def static_visual_embeddings(scorer: Any, crop_path: Path, audio_path: Path, target_window_count: int, *, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute visual embeddings for a repeated reference crop.

    The audio embedding returned by the low-level forward is discarded by the
    static-control caller; native audio embeddings remain the bound audio
    input.  This creates no TFG video and therefore cannot alter generation.
    """
    import wave
    import cv2

    from scripts.experiments.wav2lip_roi_peak_recheck import worker as sync_worker

    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[:2] != (224, 224):
        raise ProtocolError(f"static crop must decode as 224x224 BGR: {crop_path}")
    with wave.open(str(audio_path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ProtocolError(f"static-control audio is not PCM16/16k mono: {audio_path}")
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
    if target_window_count < 1:
        raise ProtocolError("static target window count must be positive")
    frames = [np.ascontiguousarray(image) for _ in range(int(target_window_count) + 5)]
    visual, audio_embedding = sync_worker._forward(
        scorer.model,
        frames,
        audio,
        scorer.device,
        int(batch_size or scorer.batch_size),
    )
    if visual.shape[0] != target_window_count:
        raise ProtocolError(f"static visual length changed: {visual.shape[0]} != {target_window_count}")
    return visual, audio_embedding


def endpoint_rows(sample_id: str, source_group: str, rows: Sequence[Mapping[str, Any]], *, tfg: str = "wav2lip") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({"schema_version": 1, "protocol_id": PROTOCOL_ID, "sample_id": sample_id, "source_group": source_group, "tfg": tfg, **dict(row)})
    return result


def write_score_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(path, list(rows))


__all__ = ["equal_count_rows", "support_indices", "summarize_curve", "summarize_pair_matrices", "distance_from_embeddings", "score_native", "score_static_control", "static_visual_embeddings", "endpoint_rows"]
