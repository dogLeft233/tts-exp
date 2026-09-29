from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .analysis import cluster_bootstrap
from .protocol import (
    DEFAULT_SEED,
    MODELS,
    NATURAL,
    ProtocolError,
    RunPaths,
    VSHIFT,
    bytes_sha256,
    canonical_hash,
    file_sha256,
    read_json,
    read_jsonl,
    resolve_path,
    write_csv,
    write_json,
    write_jsonl,
)


def import_official(parent_root: Path, *, destination: Path | None = None) -> dict[str, Any]:
    """Import the already-completed official full-track result, never rerun it."""

    summary_path = parent_root / "07_official_syncnet/full/summary.json"
    summary = read_json(summary_path)
    records = summary.get("records", [])
    if int(summary.get("completed", -1)) != 108 or int(summary.get("failed", -1)) != 0 or len(records) != 108:
        raise ProtocolError("official parent summary is not complete 108/108")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("cell_key", ""))
        if not key or key in seen:
            raise ProtocolError(f"official summary has duplicate/missing cell key: {key}")
        seen.add(key)
        confidence = record.get("sync_c")
        if confidence is None:
            raise ProtocolError(f"official record has no Confidence value: {key}")
        receipt_path = Path(str(record.get("receipt", "")))
        receipt_value: Mapping[str, Any] = read_json(receipt_path) if receipt_path.is_file() else {}
        support_hash = canonical_hash({"metric_family": "official_fulltrack", "min_track": record.get("min_track"), "scorer": record.get("scorer")})
        rows.append(
            {
                "protocol_id": "tts_evidence_temporal_patch_v1",
                "revision": "20260922",
                "metric_family": "official_fulltrack",
                "endpoint": "confidence",
                "source_group": str(record["source_group"]),
                "sample_id": str(record["sample_id"]),
                "tts_arm": str(record["arm"]),
                "condition": NATURAL if str(record["arm"]) == NATURAL else str(record["arm"]),
                "cell_key": key,
                "value": float(confidence),
                "min_dist": float(record["sync_d"]) if record.get("sync_d") is not None else None,
                "av_offset": record.get("av_offset"),
                "min_track": record.get("min_track"),
                "scorer": str(record.get("scorer", "official_syncnet_v2")),
                "video": str(record.get("video", "")),
                "video_sha256": record.get("video_sha256"),
                "audio_pcm_sha256": receipt_value.get("audio_pcm_sha256"),
                "model_sha256": receipt_value.get("checkpoint_sha256"),
                "support_hash": support_hash,
                "calibration_hash": None,
                "receipt": str(record.get("receipt", "")),
                "receipt_sha256": record.get("receipt_sha256"),
                "source_summary_sha256": file_sha256(summary_path),
                "status": "IMPORTED",
            }
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "endpoint": "official_fulltrack_confidence",
        "summary": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "count": len(rows),
        "rows": rows,
    }
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    if destination is not None:
        write_json(destination, result)
    return result


def _distance_from_embeddings(visual: np.ndarray, audio: np.ndarray, *, vshift: int = VSHIFT) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.ndim != 2 or audio_value.ndim != 2 or visual_value.shape != audio_value.shape:
        raise ProtocolError("SyncNet embedding shapes do not match")
    if not np.isfinite(visual_value).all() or not np.isfinite(audio_value).all():
        raise ProtocolError("SyncNet embeddings contain non-finite values")
    padded = np.pad(audio_value, ((vshift, vshift), (0, 0)), mode="constant")
    output = np.empty((len(visual_value), 2 * vshift + 1), dtype=np.float32)
    for row in range(len(visual_value)):
        difference = visual_value[row : row + 1] - padded[row : row + 2 * vshift + 1]
        output[row] = np.sqrt(np.sum((difference + 1e-6) ** 2, axis=1)).astype(np.float32)
    return output


def _unit_norm(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value.astype(np.float64), axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ProtocolError("cannot unit-normalize a zero SyncNet embedding")
    return (value / norm).astype(np.float32)


def _lag_column(lag: int, vshift: int = VSHIFT) -> int:
    column = int(vshift - int(lag))
    if column < 0 or column > 2 * vshift:
        raise ValueError(f"lag outside registered range: {lag}")
    return column


def score_temporal_rank(
    matrix: np.ndarray,
    *,
    normalized_matrix: np.ndarray | None = None,
    candidate_rows: Sequence[int] | None = None,
    calibration_rows: Sequence[int] | None = None,
    evaluation_rows: Sequence[int] | None = None,
    fixed_lag: int | None = None,
    deltas: Sequence[int] = (-5, -3, -2, 2, 3, 5),
    vshift: int = VSHIFT,
    guard_rows: int = 20,
    min_calibration_rows: int = 10,
    min_evaluation_rows: int = 25,
    tail_rows: Sequence[int] = (),
    query_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Calibrate one lag on an early support and evaluate rank on disjoint rows."""

    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2 * vshift + 1:
        raise ProtocolError(f"distance matrix must have shape (F,{2 * vshift + 1}), got {value.shape}")
    if not np.isfinite(value).all():
        raise ProtocolError("distance matrix contains non-finite values")
    if normalized_matrix is not None:
        normalized = np.asarray(normalized_matrix, dtype=np.float32)
        if normalized.shape != value.shape or not np.isfinite(normalized).all():
            raise ProtocolError("normalized distance matrix does not match raw matrix")
    else:
        normalized = None
    if calibration_rows is not None:
        calibration = np.asarray(sorted({int(row) for row in calibration_rows if 0 <= int(row) < len(value)}), dtype=np.int64)
    else:
        if candidate_rows is None:
            candidate = np.arange(len(value), dtype=np.int64)
        else:
            candidate = np.asarray(sorted({int(row) for row in candidate_rows if 0 <= int(row) < len(value)}), dtype=np.int64)
        calibration_end = int(len(candidate) // 3)
        calibration = candidate[:calibration_end]
    if evaluation_rows is not None:
        evaluation = np.asarray(sorted({int(row) for row in evaluation_rows if 0 <= int(row) < len(value)}), dtype=np.int64)
    else:
        if candidate_rows is None:
            candidate = np.arange(len(value), dtype=np.int64)
        elif "candidate" not in locals():
            candidate = np.asarray(sorted({int(row) for row in candidate_rows if 0 <= int(row) < len(value)}), dtype=np.int64)
        calibration_end = int(len(candidate) // 3)
        evaluation_start = (int(candidate[calibration_end]) + int(guard_rows)) if calibration_end < len(candidate) else int(guard_rows)
        evaluation = np.asarray([row for row in candidate[calibration_end:] if int(row) >= evaluation_start], dtype=np.int64)
    excluded_tail = {int(row) for row in tail_rows}
    calibration = np.asarray([row for row in calibration if int(row) not in excluded_tail], dtype=np.int64)
    evaluation = np.asarray([row for row in evaluation if int(row) not in excluded_tail], dtype=np.int64)
    if len(calibration) < min_calibration_rows or len(evaluation) < min_evaluation_rows:
        return {
            "status": "UNMEASURABLE",
            "reason": "INSUFFICIENT_DISJOINT_SUPPORT",
            "calibration_rows": calibration.tolist(),
            "evaluation_rows": evaluation.tolist(),
            "lag": None,
        }
    means = np.mean(value[calibration], axis=0, dtype=np.float64)
    if fixed_lag is None:
        choices = [(float(means[_lag_column(lag, vshift)]), abs(lag), lag) for lag in range(-vshift, vshift + 1)]
        _best_value, _abs_lag, best_lag = min(choices)
    else:
        best_lag = int(fixed_lag)
        if best_lag < -vshift or best_lag > vshift:
            raise ProtocolError("fixed lag is outside the registered range")
    base_column = _lag_column(best_lag, vshift)
    available_deltas = [int(delta) for delta in deltas if -vshift <= best_lag + int(delta) <= vshift]
    rank_rows = evaluation
    ranks: dict[str, float] = {}
    for delta in available_deltas:
        candidate_column = _lag_column(best_lag + delta, vshift)
        left = value[rank_rows, candidate_column]
        right = value[rank_rows, base_column]
        ranks[str(delta)] = float(np.mean((left > right).astype(np.float64) + 0.5 * np.isclose(left, right, rtol=0.0, atol=1e-8), dtype=np.float64))
    usable = np.ones(len(rank_rows), dtype=bool)
    for delta in available_deltas:
        candidate = value[rank_rows, _lag_column(best_lag + delta, vshift)]
        base = value[rank_rows, base_column]
        usable &= np.isfinite(candidate) & np.isfinite(base)
    if not all(usable):
        rank_rows = rank_rows[usable]
    event_ranks: dict[str, dict[str, float]] = {}
    if query_labels is not None:
        for label in sorted({str(query_labels.get(int(row), "UNKNOWN")) for row in rank_rows}):
            selected = np.asarray([int(row) for row in rank_rows if str(query_labels.get(int(row), "UNKNOWN")) == label], dtype=np.int64)
            if selected.size == 0:
                continue
            event_ranks[label] = {}
            for delta in available_deltas:
                candidate = value[selected, _lag_column(best_lag + delta, vshift)]
                base = value[selected, base_column]
                event_ranks[label][str(delta)] = float(np.mean((candidate > base).astype(np.float64) + 0.5 * np.isclose(candidate, base, rtol=0.0, atol=1e-8)))
    curve = np.mean(value[rank_rows], axis=0, dtype=np.float64)
    minimum = float(np.min(curve))
    background = float(np.median(curve))
    output: dict[str, Any] = {
        "status": "MEASURABLE",
        "reason": None,
        "calibration_rows": calibration.tolist(),
        "evaluation_rows": rank_rows.tolist(),
        "lag": int(best_lag),
        "deltas": available_deltas,
        "rank": ranks,
        "rank_mean": float(np.mean(list(ranks.values()), dtype=np.float64)) if ranks else None,
        "event_rank": event_ranks,
        "curve": [float(item) for item in curve],
        "background_b": background,
        "minimum_d": minimum,
        "c_b_minus_d": background - minimum,
        "trough_width_frames": int(np.sum(curve <= minimum + (background - minimum) / 2.0)) if background > minimum else 0,
        "tail_rows_excluded": sorted(excluded_tail),
        "support_hash": canonical_hash({"calibration_rows": calibration.tolist(), "evaluation_rows": rank_rows.tolist(), "lag": int(best_lag), "deltas": available_deltas}),
    }
    if normalized is not None:
        output["normalized"] = score_temporal_rank(
            normalized,
            deltas=deltas,
            vshift=vshift,
            guard_rows=guard_rows,
            min_calibration_rows=min_calibration_rows,
            min_evaluation_rows=min_evaluation_rows,
            candidate_rows=candidate_rows,
            tail_rows=tail_rows,
            query_labels=query_labels,
        )
    return output


def _load_score_rows(parent_root: Path) -> list[dict[str, Any]]:
    path = parent_root / "04_syncnet/scores.jsonl"
    if not path.is_file():
        raise ProtocolError(f"custom score cache is missing: {path}")
    return read_jsonl(path)


def score_existing_temporal(
    parent_root: Path,
    inventory: Mapping[str, Any],
    *,
    destination: Path | None = None,
    deltas: Sequence[int] = (-5, -3, -2, 2, 3, 5),
) -> dict[str, Any]:
    """Score only cached embeddings/matrices; this function never renders TFG."""

    score_rows = _load_score_rows(parent_root)
    cells = {(str(row["sample_id"]), str(row["arm"])): row for row in inventory.get("cells", [])}
    output_rows: list[dict[str, Any]] = []
    for sample in score_rows:
        sample_id = str(sample.get("sample_id", ""))
        source_group = str(sample.get("source_group", ""))
        matrix_paths = sample.get("matrix_paths", {})
        if not isinstance(matrix_paths, Mapping):
            continue
        endpoint_rows = sample.get("endpoints", [])
        endpoint_by_arm = {
            str(row.get("condition")): row
            for row in endpoint_rows
            if isinstance(row, Mapping) and str(row.get("support")) == "INTERIOR"
        }
        for arm in sorted(matrix_paths):
            cell = cells.get((sample_id, arm))
            if cell is None:
                continue
            matrix_path = resolve_path(str(matrix_paths[arm]))
            visual_path = matrix_path.with_name("visual.npy")
            audio_path = matrix_path.with_name("audio.npy")
            if not matrix_path.is_file() or not visual_path.is_file() or not audio_path.is_file():
                output_rows.append({"source_group": source_group, "sample_id": sample_id, "tts_arm": arm, "status": "MISSING_CACHE", "reason": str(matrix_path)})
                continue
            matrix = np.load(matrix_path, allow_pickle=False).astype(np.float32)
            visual = np.load(visual_path, allow_pickle=False).astype(np.float32)
            audio = np.load(audio_path, allow_pickle=False).astype(np.float32)
            normalized_matrix = _distance_from_embeddings(_unit_norm(visual), _unit_norm(audio))
            endpoint = endpoint_by_arm.get(str(arm), {})
            candidate_rows = endpoint.get("support_rows") if isinstance(endpoint, Mapping) else None
            result = score_temporal_rank(matrix, normalized_matrix=normalized_matrix, candidate_rows=candidate_rows, deltas=deltas)
            value = result.get("rank_mean")
            calibration_hash = canonical_hash({"calibration_rows": result.get("calibration_rows", []), "lag": result.get("lag")})
            output_rows.append(
                {
                    "protocol_id": "tts_evidence_temporal_patch_v1",
                    "metric_family": "custom_syncnet_temporal_rank",
                    "endpoint": "rank_mean",
                    "source_group": source_group,
                    "sample_id": sample_id,
                    "tts_arm": arm,
                    "condition": NATURAL if arm == NATURAL else arm,
                    "value": value,
                    "delta_r": None,
                    "status": result["status"],
                    "reason": result.get("reason"),
                    "raw": result,
                    "matrix_sha256": file_sha256(matrix_path),
                    "visual_sha256": file_sha256(visual_path),
                    "audio_embedding_sha256": file_sha256(audio_path),
                    "support_hash": result.get("support_hash"),
                    "calibration_hash": calibration_hash,
                    "video_sha256": cell.get("video_sha256"),
                    "audio_pcm_sha256": cell.get("audio_pcm_sha256"),
                    "model_sha256": cell.get("checkpoint_sha256"),
                    "source": "parent_04_syncnet_cache",
                }
            )
    # Convert each TTS-vs-natural pair only after each arm has been scored.
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in output_rows:
        if row.get("status") == "MEASURABLE":
            paired[(str(row["source_group"]), str(row["sample_id"]))][str(row["tts_arm"])] = row
    for pair in paired.values():
        natural = pair.get(NATURAL)
        if natural is None:
            continue
        for arm, row in pair.items():
            if arm == NATURAL:
                continue
            if row.get("value") is not None and natural.get("value") is not None:
                row["delta_r"] = float(row["value"] - natural["value"])
    result = {"schema_version": 1, "status": "complete", "rows": output_rows, "count": len(output_rows), "source": "custom_fixed_roi_not_official"}
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    if destination is not None:
        write_json(destination, result)
    return result


def extract_visual(
    video: Path,
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run/reuse the isolated landmark worker and propagate its validity state."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npz"
    metadata_path = output_dir / "metadata.json"
    cached = metadata_path.is_file() and feature_path.is_file()
    if cached:
        metadata = read_json(metadata_path)
        if metadata.get("video_sha256") == file_sha256(video):
            with np.load(feature_path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in ("landmarks", "valid", "timestamps_s")}
            return {"metadata": metadata, "arrays": arrays, "status": "CACHED"}
    worker = resolve_path(str(config.get("visual_worker", "scripts/experiments/tts_visual_timing_worker.py")))
    python = str(config.get("visual_python", os.environ.get("PYTHON", os.sys.executable)))
    landmarker = resolve_path(str(config.get("landmarker", ""))) if config.get("landmarker") else None
    if landmarker is None or not landmarker.is_file():
        return {"status": "UNAVAILABLE", "reason": "LANDMARKER_MISSING", "video": str(video)}
    command = [python, str(worker), "--mode", "extract", "--video", str(video), "--landmarker", str(landmarker), "--output", str(feature_path), "--metadata", str(metadata_path), "--preview-dir", str(output_dir / "previews")]
    completed = runner(command, cwd=str(resolve_path(".")), capture_output=True, text=True, check=False, timeout=3600)
    if completed.returncode != 0 or not feature_path.is_file() or not metadata_path.is_file():
        return {"status": "FAILED", "reason": (completed.stderr or "")[-1000:], "command": command}
    metadata = read_json(metadata_path)
    with np.load(feature_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in ("landmarks", "valid", "timestamps_s")}
    if metadata.get("video_sha256") != file_sha256(video):
        raise ProtocolError("landmark worker output video hash does not match input")
    return {"metadata": metadata, "arrays": arrays, "status": "COMPUTED", "command": command}


def score_visual_timing(extracted: Mapping[str, Any], *, width: float, height: float) -> dict[str, Any]:
    if extracted.get("status") not in {"CACHED", "COMPUTED"}:
        return {"status": "UNMEASURABLE", "reason": extracted.get("reason", extracted.get("status"))}
    from scripts.experiments.tts_visual_timing_metrics import aperture_events

    arrays = extracted["arrays"]
    return aperture_events(arrays["landmarks"], arrays["valid"], arrays["timestamps_s"], width=width, height=height)


def _reindex_visual_arrays(arrays: Mapping[str, Any], source_indices: Sequence[int]) -> dict[str, np.ndarray]:
    landmarks = np.asarray(arrays["landmarks"], dtype=np.float64)
    valid = np.asarray(arrays["valid"], dtype=bool)
    timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
    if landmarks.ndim != 3 or len(source_indices) != len(landmarks) or valid.shape != (len(landmarks),) or timestamps.shape != (len(landmarks),):
        raise ProtocolError("visual control arrays have inconsistent shapes")
    output = np.full_like(landmarks, np.nan, dtype=np.float64)
    output_valid = np.zeros(len(landmarks), dtype=bool)
    for target, source in enumerate(source_indices):
        source_index = int(source)
        if 0 <= source_index < len(landmarks) and valid[source_index] and np.isfinite(landmarks[source_index]).all():
            output[target] = landmarks[source_index]
            output_valid[target] = True
    return {"landmarks": output, "valid": output_valid, "timestamps_s": timestamps.copy()}


def _triangular_shift_indices(frame_count: int, *, sign: int, amplitude: int = 2) -> list[int]:
    if frame_count < 1:
        return []
    center = (frame_count - 1) / 2.0
    half_width = max(2.0, min(20.0, frame_count / 4.0))
    shifts: list[int] = []
    for index in range(frame_count):
        distance = abs(float(index) - center)
        weight = max(0.0, 1.0 - distance / half_width)
        displacement = int(round(float(sign) * float(amplitude) * weight))
        shifts.append(index - displacement)
    return shifts


def score_visual_controls(extracted: Mapping[str, Any], *, width: float, height: float) -> dict[str, Any]:
    """Apply deterministic visual-only index controls without re-running TFG."""

    if extracted.get("status") not in {"CACHED", "COMPUTED"}:
        return {"status": "UNMEASURABLE", "reason": extracted.get("reason", extracted.get("status")), "controls": {}}
    arrays = extracted["arrays"]
    frame_count = len(np.asarray(arrays["landmarks"]))
    controls: dict[str, Any] = {}
    for shift in (-3, 3):
        indices = [index - shift for index in range(frame_count)]
        control_arrays = _reindex_visual_arrays(arrays, indices)
        controls[f"GLOBAL_{shift:+d}"] = {
            "shift_frames": int(shift),
            "source_indices": indices,
            "result": aperture_events(control_arrays["landmarks"], control_arrays["valid"], control_arrays["timestamps_s"], width=width, height=height),
        }
    for sign in (-1, 1):
        indices = _triangular_shift_indices(frame_count, sign=sign, amplitude=2)
        control_arrays = _reindex_visual_arrays(arrays, indices)
        controls[f"TRIANGLE_{sign:+d}"] = {
            "amplitude_frames": int(sign * 2),
            "source_indices": indices,
            "result": aperture_events(control_arrays["landmarks"], control_arrays["valid"], control_arrays["timestamps_s"], width=width, height=height),
        }
    return {"status": "COMPLETE", "base": score_visual_timing(extracted, width=width, height=height), "controls": controls, "control_contract": "no_wrap_or_interpolation;timestamps_fixed;visual_only"}


def build_native_pairs(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in inventory.get("cells", []):
        groups[str(row["source_group"])][str(row["arm"])] = row
    output: list[dict[str, Any]] = []
    for group in sorted(groups):
        arms = groups[group]
        natural = arms.get(NATURAL)
        if natural is None:
            continue
        for arm in MODELS:
            tts = arms.get(arm)
            if tts is None:
                continue
            output.append(
                {
                    "pair_id": f"{group}::{tts['sample_id']}::{arm}",
                    "source_group": group,
                    "sample_id": tts["sample_id"],
                    "tts_arm": arm,
                    "natural_video": natural["video"],
                    "tts_video": tts["video"],
                    "natural_video_sha256": natural["video_sha256"],
                    "tts_video_sha256": tts["video_sha256"],
                }
            )
    return output


def export_blind_pack(
    pairs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    raters_per_pair: int = 3,
    ffmpeg: Path | None = None,
    require_qc_media: bool = False,
) -> dict[str, Any]:
    """Export an opaque offline pack with primary trials and QC trials.

    The subject manifest deliberately contains no condition labels.  The
    administrator key is the only place where the experimental assignment,
    control answer, and repeat relationship are retained.  QC media is
    materialized only when an ffmpeg binary is supplied; the default remains
    useful for unit fixtures that do not have real videos.
    """
    if raters_per_pair < 1:
        raise ValueError("raters_per_pair must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    subject_rows: list[dict[str, Any]] = []
    admin_rows: list[dict[str, Any]] = []
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    def link_opaque(source: str | Path, target: Path) -> bool:
        if target.exists() or target.is_symlink():
            target.unlink()
        source_path = Path(str(source))
        if not source_path.is_file():
            return False
        # A relative symlink keeps the pack small while exposing only an
        # opaque trial filename to the subject-facing manifest.
        target.symlink_to(source_path.resolve())
        return True

    def make_offset_media(source: str | Path, target: Path, offset_ms: int) -> bool:
        if ffmpeg is None or not Path(ffmpeg).is_file():
            return False
        source_path = Path(str(source))
        if not source_path.is_file():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
        temporary.unlink(missing_ok=True)
        command = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-itsoffset",
            f"{float(offset_ms) / 1000.0:.3f}",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-shortest",
            "-f",
            "matroska",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            if require_qc_media:
                raise ProtocolError(f"failed to create QC offset media: {(completed.stderr or '')[-1000:]}")
            return False
        temporary.replace(target)
        return True

    def add_subject_trial(
        *,
        trial_id: str,
        left: str | Path,
        right: str | Path,
        trial_kind: str,
    ) -> bool:
        left_subject = media_dir / f"{trial_id}_left.mkv"
        right_subject = media_dir / f"{trial_id}_right.mkv"
        left_ok = link_opaque(left, left_subject)
        right_ok = link_opaque(right, right_subject)
        if not (left_ok and right_ok):
            for path in (left_subject, right_subject):
                if path.exists() or path.is_symlink():
                    path.unlink()
            if require_qc_media:
                raise ProtocolError(f"blind media is missing for {trial_id}")
            # Fixture mode still emits the deterministic manifest/key so the
            # protocol can be unit-tested without shipping video files.
            subject_rows.append(
                {
                    "trial_id": trial_id,
                    "left_media": str(left_subject.relative_to(output_dir)),
                    "right_media": str(right_subject.relative_to(output_dir)),
                    "question": "哪段声音与嘴部动作更同步？",
                    "choices": "left|right|tie|unjudgeable",
                    "quality_question": "是否有明显脸部/嘴部破坏导致无法公平判断？",
                    "quality_choices": "yes|no",
                    "trial_kind": trial_kind,
                }
            )
            return True
        subject_rows.append(
            {
                "trial_id": trial_id,
                "left_media": str(left_subject.relative_to(output_dir)),
                "right_media": str(right_subject.relative_to(output_dir)),
                "question": "哪段声音与嘴部动作更同步？",
                "choices": "left|right|tie|unjudgeable",
                "quality_question": "是否有明显脸部/嘴部破坏导致无法公平判断？",
                "quality_choices": "yes|no",
                "trial_kind": trial_kind,
            }
        )
        return True

    for pair in pairs:
        for rater_slot in range(raters_per_pair):
            trial_id = hashlib.sha256(f"{seed}|primary|{pair['pair_id']}|{rater_slot}".encode()).hexdigest()[:20]
            left_tts = bool(rng.getrandbits(1))
            left = pair["tts_video"] if left_tts else pair["natural_video"]
            right = pair["natural_video"] if left_tts else pair["tts_video"]
            if not add_subject_trial(trial_id=trial_id, left=left, right=right, trial_kind="primary"):
                continue
            admin_rows.append({"trial_id": trial_id, "pair_id": pair["pair_id"], "source_group": pair["source_group"], "sample_id": pair["sample_id"], "tts_arm": pair["tts_arm"], "left_condition": "tts" if left_tts else NATURAL, "right_condition": NATURAL if left_tts else "tts", "rater_slot": rater_slot, "trial_kind": "primary", "repeat_of": None})

    # Each rater slot receives eight known ±200 ms mismatches.  Controls are
    # not part of the primary estimand and are omitted rather than fabricated
    # when this function is used without real media.
    qc_media_ready = ffmpeg is not None and Path(ffmpeg).is_file()
    control_rows: list[dict[str, Any]] = []
    if pairs and qc_media_ready:
        for rater_slot in range(raters_per_pair):
            for control_index in range(8):
                pair = pairs[(rater_slot * 8 + control_index) % len(pairs)]
                offset_ms = 200 if control_index % 2 else -200
                trial_id = hashlib.sha256(f"{seed}|control|{rater_slot}|{control_index}|{pair['pair_id']}".encode()).hexdigest()[:20]
                correct_source = pair["natural_video"]
                wrong_target = media_dir / f"{trial_id}_wrong.mkv"
                correct_left = bool(rng.getrandbits(1))
                if not make_offset_media(correct_source, wrong_target, offset_ms):
                    continue
                left = correct_source if correct_left else wrong_target
                right = wrong_target if correct_left else correct_source
                if not add_subject_trial(trial_id=trial_id, left=left, right=right, trial_kind="control"):
                    continue
                expected_choice = "left" if correct_left else "right"
                control_rows.append({"trial_id": trial_id, "pair_id": pair["pair_id"], "source_group": pair["source_group"], "sample_id": pair["sample_id"], "tts_arm": pair["tts_arm"], "left_condition": "natural" if correct_left else "offset_control", "right_condition": "offset_control" if correct_left else "natural", "rater_slot": rater_slot, "trial_kind": "control", "repeat_of": None, "offset_ms": offset_ms, "expected_choice": expected_choice})
                # The generated wrong-side media is an implementation detail;
                # subject-facing trial paths remain opaque.
                wrong_target.unlink(missing_ok=True)

    # Repeats reuse the primary media assignment, but receive a fresh opaque
    # ID so agreement can be measured without exposing the key.
    primary_by_slot = defaultdict(list)
    for row in admin_rows:
        if row.get("trial_kind") == "primary":
            primary_by_slot[int(row["rater_slot"])].append(row)
    repeat_rows: list[dict[str, Any]] = []
    subject_primary = {row["trial_id"]: row for row in subject_rows if row.get("trial_kind") == "primary"}
    for rater_slot in range(raters_per_pair):
        for repeat_index, original in enumerate(primary_by_slot.get(rater_slot, [])[:4]):
            original_subject = subject_primary.get(original["trial_id"])
            if original_subject is None:
                continue
            trial_id = hashlib.sha256(f"{seed}|repeat|{rater_slot}|{repeat_index}|{original['trial_id']}".encode()).hexdigest()[:20]
            left_source = output_dir / original_subject["left_media"]
            right_source = output_dir / original_subject["right_media"]
            if not (left_source.is_file() and right_source.is_file()):
                continue
            if not add_subject_trial(trial_id=trial_id, left=left_source, right=right_source, trial_kind="repeat"):
                continue
            repeat_rows.append({**original, "trial_id": trial_id, "rater_slot": rater_slot, "trial_kind": "repeat", "repeat_of": original["trial_id"]})

    admin_rows.extend(control_rows)
    admin_rows.extend(repeat_rows)
    # Do not expose even the control/repeat kind in the subject-facing CSV.
    write_csv(output_dir / "subject_trials.csv", ["trial_id", "left_media", "right_media", "question", "choices", "quality_question", "quality_choices"], subject_rows)
    write_json(output_dir / "admin_key.json", {"schema_version": 1, "seed": int(seed), "rows": admin_rows})
    # The subject-facing file intentionally carries no model names, scores, or
    # condition labels.  It is a local offline pack; no service/upload occurs.
    html_rows = "\n".join(f"<tr><td>{html.escape(row['trial_id'])}</td><td><video controls src='{html.escape(row['left_media'])}'></video></td><td><video controls src='{html.escape(row['right_media'])}'></video></td></tr>" for row in subject_rows)
    (output_dir / "index.html").write_text(f"<!doctype html><meta charset='utf-8'><table><tr><th>ID</th><th>Left</th><th>Right</th></tr>{html_rows}</table>", encoding="utf-8")
    primary_count = sum(1 for row in admin_rows if row.get("trial_kind") == "primary")
    control_count = sum(1 for row in admin_rows if row.get("trial_kind") == "control")
    repeat_count = sum(1 for row in admin_rows if row.get("trial_kind") == "repeat")
    status = "READY_FOR_HUMAN" if control_count == raters_per_pair * 8 and repeat_count == raters_per_pair * 4 else "READY_FOR_HUMAN_QC_MEDIA_PENDING"
    result = {"schema_version": 1, "status": status, "pair_count": len(pairs), "trial_count": len(subject_rows), "primary_trial_count": primary_count, "control_trial_count": control_count, "repeat_trial_count": repeat_count, "raters_per_pair": int(raters_per_pair), "qc_media_ready": bool(control_count == raters_per_pair * 8 and repeat_count == raters_per_pair * 4), "subject_manifest": str((output_dir / "subject_trials.csv").resolve()), "admin_key": str((output_dir / "admin_key.json").resolve())}
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    write_json(output_dir / "manifest.json", result)
    return result


def import_ratings(ratings_path: Path, pack_dir: Path, *, minimum_control_accuracy: float = 7 / 8, minimum_repeat_agreement: float = 3 / 4) -> dict[str, Any]:
    """Validate human ratings and return pair-level scores; never infer missing ratings."""

    admin = read_json(pack_dir / "admin_key.json")
    key = {str(row["trial_id"]): row for row in admin.get("rows", [])}
    with ratings_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"trial_id", "rater_id", "choice"}
    if rows and not required.issubset(rows[0]):
        raise ProtocolError(f"ratings must contain columns: {sorted(required)}")
    choices = {"left", "right", "tie", "unjudgeable"}
    by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ratings: set[tuple[str, str]] = set()
    for row in rows:
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in key:
            raise ProtocolError(f"rating references unknown trial: {trial_id}")
        rater_id = str(row.get("rater_id", "")).strip()
        if not rater_id:
            raise ProtocolError("rating has an empty rater_id")
        identity = (rater_id, trial_id)
        if identity in seen_ratings:
            raise ProtocolError(f"duplicate rating: {rater_id}/{trial_id}")
        seen_ratings.add(identity)
        choice = str(row.get("choice", "")).lower()
        if choice not in choices:
            raise ProtocolError(f"invalid rating choice: {choice}")
        mapping = key[trial_id]
        by_rater[rater_id].append({"mapping": mapping, "rater_id": rater_id, "choice": choice, "quality": str(row.get("quality", "")).lower(), "trial_id": trial_id})

    qc_by_rater: dict[str, dict[str, Any]] = {}
    eligible_raters: set[str] = set()
    for rater_id, rated in sorted(by_rater.items()):
        control = [item for item in rated if item["mapping"].get("trial_kind") == "control"]
        correct = sum(1 for item in control if item["choice"] == item["mapping"].get("expected_choice"))
        repeats = [item for item in rated if item["mapping"].get("trial_kind") == "repeat"]
        primary_choices = {item["trial_id"]: item["choice"] for item in rated if item["mapping"].get("trial_kind") == "primary"}
        repeat_matches = 0
        repeat_comparable = 0
        for item in repeats:
            original = str(item["mapping"].get("repeat_of", ""))
            if original in primary_choices:
                repeat_comparable += 1
                repeat_matches += int(primary_choices[original] == item["choice"])
        control_accuracy = (correct / len(control)) if control else None
        repeat_agreement = (repeat_matches / repeat_comparable) if repeat_comparable else None
        eligible = bool(control_accuracy is not None and len(control) >= 8 and control_accuracy >= minimum_control_accuracy and repeat_agreement is not None and repeat_comparable >= 4 and repeat_agreement >= minimum_repeat_agreement)
        if eligible:
            eligible_raters.add(rater_id)
        qc_by_rater[rater_id] = {"rater_id": rater_id, "control_count": len(control), "control_correct": correct, "control_accuracy": control_accuracy, "repeat_count": len(repeats), "repeat_comparable": repeat_comparable, "repeat_matches": repeat_matches, "repeat_agreement": repeat_agreement, "eligible": eligible}

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rater_id in sorted(eligible_raters):
        for item in by_rater[rater_id]:
            mapping = item["mapping"]
            if mapping.get("trial_kind") != "primary" or item["choice"] == "unjudgeable":
                continue
            choice = item["choice"]
            score = 0.5 if choice == "tie" else (1.0 if (choice == "left") == (mapping["left_condition"] == "tts") else 0.0)
            quality_value = str(item.get("quality", "")).lower() in {"yes", "true", "1"}
            by_pair[str(mapping["pair_id"])].append({"source_group": mapping["source_group"], "sample_id": mapping["sample_id"], "tts_arm": mapping["tts_arm"], "rater_id": rater_id, "choice": choice, "score": score, "quality_confounded": quality_value, "trial_id": item["trial_id"]})
    pair_rows: list[dict[str, Any]] = []
    for pair_id, pair_ratings in sorted(by_pair.items()):
        scores = [float(row["score"]) for row in pair_ratings]
        pair_rows.append({"pair_id": pair_id, "source_group": pair_ratings[0]["source_group"], "sample_id": pair_ratings[0]["sample_id"], "tts_arm": pair_ratings[0]["tts_arm"], "score": float(np.mean(scores, dtype=np.float64)), "rating_count": len(scores), "quality_confounded": any(bool(row.get("quality_confounded")) for row in pair_ratings), "status": "MEASURED" if len(scores) >= 3 else "MISSING_INSUFFICIENT_ELIGIBLE_RATERS", "ratings": pair_ratings})
    measured_pairs = [row for row in pair_rows if row["status"] == "MEASURED"]
    result = {"schema_version": 1, "status": "MEASURED" if measured_pairs else "PENDING_HUMAN", "pair_rows": pair_rows, "raw_rows": rows, "qc": {"minimum_control_accuracy": minimum_control_accuracy, "minimum_repeat_agreement": minimum_repeat_agreement, "raters": list(qc_by_rater.values()), "eligible_rater_count": len(eligible_raters), "measured_pair_count": len(measured_pairs)}}
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    write_json(pack_dir / "ratings.json", result)
    return result


def score_patch_media(
    media: Path,
    source_audio: Path,
    output_dir: Path,
    *,
    scorer: Callable[..., Mapping[str, Any]] | None = None,
    expected_media_sha256: str | None = None,
    expected_pcm_sha256: str | None = None,
) -> dict[str, Any]:
    """Call an explicitly supplied scorer; no hidden official/custom mixing."""

    if scorer is None:
        return {"status": "PENDING_SCORER", "media": str(media), "source_audio": str(source_audio)}
    if expected_media_sha256 is not None and file_sha256(media) != expected_media_sha256:
        raise ProtocolError("patch media hash changed before scoring")
    result = dict(scorer(media=media, source_audio=source_audio, output_dir=output_dir, expected_media_sha256=expected_media_sha256, expected_pcm_sha256=expected_pcm_sha256))
    result.update({"metric_family": "custom_fresh_syncnet", "status": result.get("status", "COMPLETE")})
    return result


def score_patch_official(
    receipts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    syncnet_root: Path,
    syncnet_python: Path,
    syncnet_model: Path,
    min_track: int = 25,
) -> dict[str, Any]:
    """Run the repository's official full-track scorer on patch media only.

    The staging receipts are new-run artifacts and each condition gets one
    unique cell key.  This prevents the official adapter from selecting an
    arbitrary Confidence track or confusing patch conditions with the parent
    108-cell cohort.
    """

    from scripts.experiments.phoneme_tfg_association import official_syncnet_eval

    staging_root = output_dir / "official_input"
    receipt_count = 0
    for receipt in receipts:
        for condition, output in receipt.get("outputs", {}).items():
            cell_key = f"{receipt['source_group']}::{receipt['sample_id']}::{receipt['direction']}::{condition}"
            safe_key = hashlib.sha256(cell_key.encode()).hexdigest()[:20]
            cell_dir = staging_root / "03_video" / "wav2lip" / safe_key
            cell_dir.mkdir(parents=True, exist_ok=True)
            staged_video = cell_dir / f"{condition}.mkv"
            if staged_video.exists() or staged_video.is_symlink():
                staged_video.unlink()
            staged_video.symlink_to(Path(str(output["video"])).resolve())
            staged_receipt = cell_dir / f"{condition}.receipt.json"
            write_json(
                staged_receipt,
                {
                    "schema_version": 1,
                    "protocol_id": "tts_evidence_temporal_patch_v1",
                    "cell_key": cell_key,
                    "sample_id": receipt["sample_id"],
                    "source_group": receipt["source_group"],
                    "arm": condition,
                    "tfg": "wav2lip",
                    "status": "complete",
                    "output": str(Path(str(output["video"])).resolve()),
                    "output_sha256": output["video_sha256"],
                    "checkpoint": receipt.get("checkpoint"),
                    "checkpoint_sha256": receipt.get("checkpoint_sha256"),
                },
            )
            receipt_count += 1
    if receipt_count == 0:
        return {"status": "EMPTY", "rows": [], "receipt_count": 0}
    official_root = output_dir / "official_fulltrack"
    result = official_syncnet_eval.evaluate(
        SimpleNamespace(
            run_root=staging_root,
            syncnet_root=syncnet_root,
            syncnet_python=syncnet_python,
            model=syncnet_model,
            output_root=official_root,
            min_track=int(min_track),
        )
    )
    rows: list[dict[str, Any]] = []
    for record in result.get("records", []):
        cell_key = str(record["cell_key"])
        condition = cell_key.rsplit("::", 1)[-1]
        rows.append(
            {
                "protocol_id": "tts_evidence_temporal_patch_v1",
                "metric_family": "official_patch_fulltrack",
                "endpoint": "confidence",
                "source_group": record.get("source_group"),
                "sample_id": record.get("sample_id"),
                "direction": "::".join(cell_key.split("::")[2:-1]),
                "condition": condition,
                "value": record.get("sync_c"),
                "min_dist": record.get("sync_d"),
                "av_offset": record.get("av_offset"),
                "video_sha256": record.get("video_sha256"),
                "model_sha256": result.get("syncnet_model_sha256"),
                "status": "MEASURED",
                "source": str((official_root / "summary.json").resolve()),
            }
        )
    for row in rows:
        row["model_sha256"] = file_sha256(syncnet_model)
    return {"status": result.get("status", "incomplete"), "rows": rows, "official": result, "receipt_count": receipt_count, "model_sha256": file_sha256(syncnet_model)}
