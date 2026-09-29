from __future__ import annotations

import hashlib
import platform
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DiagnosticError,
    assert_finite,
    compare_values,
    file_sha256,
    read_json,
    resolve_path,
    sample_ids_sha256,
    verify_self_hashed_json,
)


def _asset(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    payload = verify_self_hashed_json(path, expected_hash)
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": file_sha256(path),
        "payload": payload,
    }


def _expected_cell_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str, bool]]:
    return {
        (str(record["sample_id"]), video, audio, repeat)
        for record in records
        for video, audio, repeat in config.ALL_CELL_SPECS
    }


def _score_index(
    score_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str, str, bool], Mapping[str, Any]]:
    rows = score_manifest.get("scores")
    if (
        score_manifest.get("status") != "complete"
        or score_manifest.get("stage") != "control"
        or not isinstance(rows, list)
        or len(rows) != config.EXPECTED_CELL_COUNT
    ):
        raise DiagnosticError("control score manifest is incomplete")
    result: dict[tuple[str, str, str, bool], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DiagnosticError("score row is malformed")
        key = (
            str(row.get("sample_id")),
            str(row.get("video_arm")),
            str(row.get("audio_arm")),
            bool(row.get("repeat", False)),
        )
        if key in result:
            raise DiagnosticError(f"duplicate score cell: {key}")
        result[key] = row
    expected = _expected_cell_keys(records)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise DiagnosticError(f"score cell set differs: missing={missing[:3]} extra={extra[:3]}")
    return result


def _matrix(row: Mapping[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray:
    path = Path(str(row.get("matrix", ""))).resolve()
    if not path.is_file():
        raise DiagnosticError(f"score matrix is missing: {path}")
    expected_hash = str(row.get("matrix_sha256", ""))
    actual_hash = file_sha256(path)
    if not expected_hash or actual_hash != expected_hash:
        raise DiagnosticError(f"score matrix hash changed: {path}")
    if expected_hash in cache:
        return cache[expected_hash]
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise DiagnosticError(f"score matrix cannot be loaded: {path}") from exc
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise DiagnosticError(f"score matrix is malformed: {path}")
    shape = row.get("matrix_shape")
    if shape != [int(value.shape[0]), int(value.shape[1])]:
        raise DiagnosticError(f"score matrix shape binding changed: {path}")
    cache[expected_hash] = value
    return value


def _media_row(media_manifest: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    rows = media_manifest.get("rows")
    if not isinstance(rows, list):
        raise DiagnosticError("media manifest rows are missing")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise DiagnosticError(f"media manifest row is missing: {sample_id}")


def _video_row(video_manifest: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    rows = video_manifest.get("rows")
    if not isinstance(rows, list):
        raise DiagnosticError("video manifest rows are missing")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise DiagnosticError(f"video manifest row is missing: {sample_id}")


def _verify_media_bindings(
    parent_root: Path,
    records: Sequence[Mapping[str, Any]],
    score_index: Mapping[tuple[str, str, str, bool], Mapping[str, Any]],
    media_manifest: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    matrix_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Audit immutable media/PCM/score bindings; no statistics are computed here."""
    pcm_pairs = 0
    matrix_count = 0
    sidecar_count = 0
    stream_keys: dict[tuple[str, str], tuple[str, str]] = {}
    audio_keys: dict[tuple[str, str], tuple[str, str, str]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        media_row = _media_row(media_manifest, sample_id)
        video_row = _video_row(video_manifest, sample_id)
        cells = media_row.get("cells")
        streams = media_row.get("streams")
        arms = video_row.get("arms")
        if not isinstance(cells, Mapping) or not isinstance(streams, Mapping) or not isinstance(arms, Mapping):
            raise DiagnosticError(f"media/video arm maps are missing: {sample_id}")
        predicted = record.get("predicted_frame_counts")
        if not isinstance(predicted, Mapping):
            raise DiagnosticError(f"predicted frame counts are missing: {sample_id}")
        for video_arm in config.VIDEO_ARMS:
            stream = streams.get(video_arm)
            video = arms.get(video_arm)
            if not isinstance(stream, Mapping):
                raise DiagnosticError(f"video arm/stream is missing: {sample_id}/{video_arm}")
            if video_arm == "R":
                source_timeline = record.get("source_video_timeline")
                if not isinstance(source_timeline, Mapping) or int(source_timeline.get("frame_count", -1)) != int(predicted[video_arm]):
                    raise DiagnosticError(f"source video frame count differs: {sample_id}/{video_arm}")
            else:
                if not isinstance(video, Mapping):
                    raise DiagnosticError(f"generated video arm is missing: {sample_id}/{video_arm}")
                output = resolve_path(video.get("output"), config.REPO)
                if not output.is_file() or file_sha256(output) != str(video.get("output_sha256")):
                    raise DiagnosticError(f"video hash changed: {sample_id}/{video_arm}")
                if int(video.get("frame_count", -1)) != int(predicted[video_arm]):
                    raise DiagnosticError(f"video frame count differs: {sample_id}/{video_arm}")
            stream_path = resolve_path(stream.get("output"), config.REPO)
            stream_hash = str(stream.get("output_sha256", ""))
            if not stream_path.is_file() or file_sha256(stream_path) != stream_hash:
                raise DiagnosticError(f"video stream hash changed: {sample_id}/{video_arm}")
            stream_keys[(sample_id, video_arm)] = (str(stream_path), stream_hash)
        for video_arm, audio_arm, repeat in config.MAIN_CELL_SPECS:
            key = (sample_id, video_arm, audio_arm, repeat)
            row = score_index[key]
            cell_key = f"{video_arm}__{audio_arm}"
            cell = cells.get(cell_key)
            if not isinstance(cell, Mapping):
                raise DiagnosticError(f"media cell is missing: {sample_id}/{cell_key}")
            media_path = resolve_path(row.get("media"), config.REPO)
            if media_path != resolve_path(cell.get("output"), config.REPO):
                raise DiagnosticError(f"score/media path mismatch: {key}")
            media_hash = str(row.get("media_sha256", ""))
            if not media_path.is_file() or file_sha256(media_path) != media_hash:
                raise DiagnosticError(f"muxed media hash changed: {media_path}")
            if media_hash != str(cell.get("output_sha256")):
                raise DiagnosticError(f"media manifest hash mismatch: {key}")
            sidecar_path = media_path.with_suffix(".json")
            sidecar = verify_self_hashed_json(sidecar_path)
            sidecar_count += 1
            if str(sidecar.get("output")) != str(media_path) or str(sidecar.get("output_sha256")) != media_hash:
                raise DiagnosticError(f"media sidecar mismatch: {key}")
            if sidecar.get("audio_modified") is not False:
                raise DiagnosticError(f"audio was modified in mux: {key}")
            if str(sidecar.get("video_stream_sha256")) != str(cell.get("video_stream_sha256")):
                raise DiagnosticError(f"video stream binding mismatch: {key}")
            stream_key = (sample_id, video_arm)
            if stream_keys[stream_key][1] != str(cell.get("video_stream_sha256")):
                raise DiagnosticError(f"video stream identity differs: {key}")
            audio_path = resolve_path(row.get("audio"), config.REPO)
            audio_hash = str(row.get("audio_sha256", ""))
            audio_pcm_hash = str(row.get("audio_pcm_sha256", ""))
            if not audio_path.is_file() or file_sha256(audio_path) != audio_hash:
                raise DiagnosticError(f"score audio hash changed: {key}")
            if audio_hash != str(cell.get("audio_container_sha256")) or audio_pcm_hash != str(cell.get("audio_pcm_sha256")):
                raise DiagnosticError(f"score/audio PCM binding mismatch: {key}")
            if audio_pcm_hash != str(sidecar.get("audio_pcm_sha256")):
                raise DiagnosticError(f"mux decoded PCM binding mismatch: {key}")
            audio_key = (sample_id, audio_arm)
            observed_audio = (str(audio_path), audio_hash, audio_pcm_hash)
            if audio_key in audio_keys and audio_keys[audio_key] != observed_audio:
                raise DiagnosticError(f"audio arm identity differs: {key}")
            audio_keys[audio_key] = observed_audio
            worker_path = resolve_path(row.get("worker_result"), config.REPO)
            worker_hash = str(row.get("worker_result_sha256", ""))
            if not worker_path.is_file() or file_sha256(worker_path) != worker_hash:
                raise DiagnosticError(f"worker result hash changed: {key}")
            worker = read_json(worker_path)
            if str(worker.get("matrix_sha256")) != str(row.get("matrix_sha256")) or str(worker.get("media_sha256")) != media_hash:
                raise DiagnosticError(f"worker binding mismatch: {key}")
            if worker.get("input_pcm_sha256") != audio_pcm_hash or worker.get("extracted_pcm_verified") is not True:
                raise DiagnosticError(f"worker PCM verification mismatch: {key}")
            _matrix(row, matrix_cache)
            matrix_count += 1
            pcm_pairs += 1
        for video_arm, audio_arm, repeat in config.REPEAT_CELL_SPECS:
            row = score_index[(sample_id, video_arm, audio_arm, repeat)]
            main = score_index[(sample_id, video_arm, audio_arm, False)]
            if str(row.get("matrix_sha256")) != str(main.get("matrix_sha256")) or str(row.get("media_sha256")) != str(main.get("media_sha256")):
                raise DiagnosticError(f"repeat cell is not paired with main cell: {sample_id}/{video_arm}")
    return {
        "score_matrix_cells_checked": matrix_count,
        "media_pcm_pairs_checked": pcm_pairs,
        "media_sidecars_checked": sidecar_count,
        "video_stream_identity_checked": len(stream_keys),
        "new_generated_videos": 0,
        "new_score_cells": 0,
    }


def load_parent() -> dict[str, Any]:
    root = config.PARENT_ROOT
    if not root.exists():
        raise DiagnosticError(f"parent run is missing: {root}")
    files = {
        "final": root / "final.json",
        "validation": root / "validation.json",
        "control": root / "control.json",
        "protocol": root / "protocol.json",
        "score_manifest": root / "scores/control/manifest.json",
    }
    assets = {name: _asset(path, config.PARENT_HASHES[name]) for name, path in files.items()}
    final = assets["final"]["payload"]
    protocol = assets["protocol"]["payload"]
    control = assets["control"]["payload"]
    score_manifest = assets["score_manifest"]["payload"]
    if final.get("status") != "complete" or final.get("scientific_decision") != "CONTROL_FAILED":
        raise DiagnosticError("parent final is not the fixed CONTROL_FAILED run")
    if file_sha256(files["protocol"]) != str(final.get("protocol_sha256")):
        raise DiagnosticError("parent final/protocol binding differs")
    if not isinstance(protocol.get("records"), list) or len(protocol["records"]) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("parent cohort does not contain 22 records")
    records = protocol["records"]
    ids = [str(item.get("sample_id")) for item in records if isinstance(item, Mapping)]
    groups = [str(item.get("source_group")) for item in records if isinstance(item, Mapping)]
    if (
        len(ids) != config.EXPECTED_RECORD_COUNT
        or len(set(ids)) != config.EXPECTED_RECORD_COUNT
        or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT
        or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256
    ):
        raise DiagnosticError("parent cohort order or source-group identity changed")
    binding_specs = final.get("spec_bindings")
    if not isinstance(binding_specs, Mapping):
        raise DiagnosticError("parent spec bindings are missing")
    inherited_specs: dict[str, Any] = {}
    for name in ("spec", "inherited_timing_spec"):
        item = binding_specs.get(name)
        if not isinstance(item, Mapping):
            raise DiagnosticError(f"parent binding is missing: {name}")
        path = Path(str(item.get("path")))
        if not path.is_file() or file_sha256(path) != str(item.get("sha256")):
            raise DiagnosticError(f"parent binding hash changed: {name}")
        inherited_specs[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": len(path.read_text(encoding="utf-8")),
        }
    video_path = root / "videos/control/manifest.json"
    media_path = root / "media/control/manifest.json"
    video_manifest = verify_self_hashed_json(video_path)
    media_manifest = verify_self_hashed_json(media_path)
    if file_sha256(video_path) != str(final.get("control_video_manifest_sha256")) or file_sha256(media_path) != str(final.get("control_media_manifest_sha256")):
        raise DiagnosticError("parent video/media manifest binding differs")
    if file_sha256(files["score_manifest"]) != str(final.get("control_score_manifest_sha256")):
        raise DiagnosticError("parent score manifest binding differs")
    score_index = _score_index(score_manifest, records)
    matrix_cache: dict[str, np.ndarray] = {}
    media_checks = _verify_media_bindings(
        root,
        records,
        score_index,
        media_manifest,
        video_manifest,
        matrix_cache,
    )
    if len(matrix_cache) == 0 or media_checks["score_matrix_cells_checked"] != config.EXPECTED_CELL_COUNT - config.EXPECTED_REPEAT_CELL_COUNT:
        raise DiagnosticError("parent score matrices were not fully audited")
    return {
        "root": str(root),
        "resolved_root": str(root.resolve()),
        "assets": {name: {key: value for key, value in item.items() if key != "payload"} for name, item in assets.items()},
        "payloads": {"final": final, "validation": assets["validation"]["payload"], "control": control, "protocol": protocol, "score_manifest": score_manifest},
        "records": records,
        "score_index": score_index,
        "matrix_cache": matrix_cache,
        "video_manifest": video_manifest,
        "media_manifest": media_manifest,
        "inherited_specs": inherited_specs,
        "media_checks": media_checks,
    }


def reconstruct_timing(sample_count: int, frame_counts: Mapping[str, Any]) -> dict[str, Any]:
    if sample_count < 2:
        raise DiagnosticError("audio is too short for timing reconstruction")
    n = np.arange(sample_count, dtype=np.float64)
    mapped = n + config.WARP_AMPLITUDE_SAMPLES * np.sin(2.0 * np.pi * n / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise DiagnosticError("timing map is not strictly monotone")
    q = min(
        *(int(frame_counts[arm]) for arm in config.VIDEO_ARMS),
        sample_count // config.SAMPLES_PER_FRAME,
    )
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, n, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, n))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise DiagnosticError("timing masks have fewer than five rows")
    return {
        "sample_count": int(sample_count),
        "q_frames": int(q),
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": hashlib.sha256(np.asarray(mapped, dtype="<f8").tobytes()).hexdigest(),
        "forward_mapping_dtype": "float64-little-endian",
        "forward_mapping_length": int(mapped.size),
    }


def summarize_curve(matrix: np.ndarray, rows: Sequence[int] | None = None) -> dict[str, Any]:
    selected = list(range(matrix.shape[0])) if rows is None else [int(row) for row in rows]
    if not selected or min(selected) < 0 or max(selected) >= matrix.shape[0]:
        raise DiagnosticError("curve rows are outside matrix")
    curve = np.mean(matrix[np.asarray(selected, dtype=np.int64), :], axis=0)
    min_index = int(np.argmin(curve))
    ordered = np.sort(curve, kind="stable")
    second_min = float(ordered[1])
    minimum = float(curve[min_index])
    peak_gap = second_min - minimum
    offset = config.VSHIFT - min_index
    return {
        "rows": selected,
        "curve": [float(value) for value in curve],
        "offsets": [config.VSHIFT - index for index in range(curve.size)],
        "min_index": min_index,
        "offset": int(offset),
        "sync_d": minimum,
        "sync_c": float(np.median(curve) - minimum),
        "min": minimum,
        "second_min": second_min,
        "peak_gap": float(peak_gap),
        "clear": bool(peak_gap > config.PEAK_GAP_THRESHOLD and offset not in (-15, 15)),
    }


def _evidence(matrix: np.ndarray, record: Mapping[str, Any], timing: Mapping[str, Any], video_arm: str) -> dict[str, Any]:
    frame_count = int(record["predicted_frame_counts"][video_arm])
    expected_rows = min(frame_count, int(record["natural_sample_count"]) // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if matrix.shape != (expected_rows, 31):
        raise DiagnosticError(f"matrix shape differs from protocol: {record['sample_id']}/{video_arm}")
    common = [int(row) for row in timing["common_window_rows"]]
    plus = [int(row) for row in timing["plus_rows"]]
    minus = [int(row) for row in timing["minus_rows"]]
    return {
        "matrix": matrix,
        "reconstructed": summarize_curve(matrix),
        "common_global": summarize_curve(matrix, common),
        "local": {
            "supported_rows": list(range(matrix.shape[0])),
            "common_window_rows": common,
            "PLUS": summarize_curve(matrix, plus),
            "MINUS": summarize_curve(matrix, minus),
        },
    }


def _compare_score_derived(
    row: Mapping[str, Any], evidence: Mapping[str, Any], differences: list[dict[str, Any]], key: str
) -> None:
    compare_values(row.get("reconstructed"), evidence["reconstructed"], f"score[{key}].reconstructed", differences)
    compare_values(row.get("common_global"), evidence["common_global"], f"score[{key}].common_global", differences)
    compare_values(row.get("local"), evidence["local"], f"score[{key}].local", differences)


def cluster_bootstrap(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    seed: int = config.BOOTSTRAP_SEED,
    draws: int = config.BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise DiagnosticError("invalid bootstrap inputs")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        number = float(value)
        if not np.isfinite(number):
            raise DiagnosticError("bootstrap input is non-finite")
        by_group[str(group)].append(number)
    labels = sorted(by_group)
    group_means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([group_means[label] for label in sampled]))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "rng": "numpy_default_rng_pcg64",
        "labels": "sorted",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "group_means": group_means,
    }


def _baseline_info(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_matrix = left["matrix"]
    right_matrix = right["matrix"]
    shape_equal = left_matrix.shape == right_matrix.shape
    max_abs = float(np.max(np.abs(left_matrix - right_matrix))) if shape_equal else None
    offsets_equal = {
        "global": int(left["common_global"]["offset"]) == int(right["common_global"]["offset"]),
        "plus": int(left["local"]["PLUS"]["offset"]) == int(right["local"]["PLUS"]["offset"]),
        "minus": int(left["local"]["MINUS"]["offset"]) == int(right["local"]["MINUS"]["offset"]),
    }
    return {
        "matrix_shape_equal": shape_equal,
        "matrix_max_abs": max_abs,
        "offsets_equal": offsets_equal,
        "passes": bool(shape_equal and max_abs is not None and max_abs <= 0.001 and all(offsets_equal.values())),
    }


def _baseline_explainable(evidence: Mapping[str, Any]) -> dict[str, Any]:
    plus = evidence["local"]["PLUS"]
    minus = evidence["local"]["MINUS"]
    return {
        "plus": {key: plus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "minus": {key: minus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "direction_difference": abs(int(plus["offset"]) - int(minus["offset"])),
        "passes": bool(plus["clear"] and minus["clear"] and abs(int(plus["offset"]) - int(minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES),
    }


def _check_pair(left: Mapping[str, Any], right: Mapping[str, Any], timing: Mapping[str, Any], keys: tuple[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected_key in zip(("PLUS", "MINUS"), keys, strict=True):
        actual = float(left["local"][name]["offset"]) - float(right["local"][name]["offset"])
        expected = float(timing[expected_key])
        result[name] = {
            "actual": actual,
            "expected": expected,
            "error": actual - expected,
            "passes": bool(left["local"][name]["clear"] and right["local"][name]["clear"] and abs(actual - expected) <= config.OFFSET_TOLERANCE_FRAMES),
        }
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _offset_noninferiority(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(int(left["common_global"]["offset"]) - int(right["common_global"]["offset"])) <= config.OFFSET_TOLERANCE_FRAMES


def _gate_ci(value: Mapping[str, Any], *, lower: float | None = None, upper: float | None = None, strict_lower: bool = False, strict_upper: bool = False) -> bool:
    ci = value.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return False
    low, high = float(ci[0]), float(ci[1])
    lower_ok = True if lower is None else (low > lower if strict_lower else low >= lower)
    upper_ok = True if upper is None else (high < upper if strict_upper else high <= upper)
    return bool(lower_ok and upper_ok)


def _c_diagnostic(gnn: Mapping[str, Any], gwn: Mapping[str, Any], timing: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    flags = {
        "baseline_invalid": not bool(baseline["passes"]),
        "boundary_peak": False,
        "unclear_peak": False,
        "offset_error": False,
    }
    expected_names = {
        "PLUS": "plus_expected_video_response",
        "MINUS": "minus_expected_video_response",
    }
    for name in ("PLUS", "MINUS"):
        left = gwn["local"][name]
        right = gnn["local"][name]
        actual = float(left["offset"]) - float(right["offset"])
        expected = float(timing[expected_names[name]])
        residual = actual - expected
        segment_flags = {
            "boundary_peak": abs(int(left["offset"])) == 15 or abs(int(right["offset"])) == 15,
            "unclear_peak": float(left["peak_gap"]) <= config.PEAK_GAP_THRESHOLD or float(right["peak_gap"]) <= config.PEAK_GAP_THRESHOLD,
            "offset_error": abs(residual) > config.OFFSET_TOLERANCE_FRAMES,
        }
        for key, value in segment_flags.items():
            flags[key] = bool(flags[key] or value)
        segments[name] = {
            "rows": list(right["rows"]),
            "gnn_curve": list(right["curve"]),
            "gwn_curve": list(left["curve"]),
            "gnn": {key: right[key] for key in ("offset", "peak_gap", "clear")},
            "gwn": {key: left[key] for key in ("offset", "peak_gap", "clear")},
            "expected": expected,
            "actual": actual,
            "residual": residual,
            "flags": segment_flags,
        }
    passes = bool(
        baseline["passes"]
        and all(
            segment["gnn"]["clear"]
            and segment["gwn"]["clear"]
            and abs(float(segment["residual"])) <= config.OFFSET_TOLERANCE_FRAMES
            for segment in segments.values()
        )
    )
    return {"segments": segments, "flags": flags, "passes": passes}


def _history_mismatches(
    parent: Mapping[str, Any],
    per_record: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    differences: list[dict[str, Any]],
) -> None:
    control = parent["payloads"]["control"]
    compare_values(control.get("per_record"), list(per_record), "control.per_record", differences)
    compare_values(control.get("gates"), gates, "control.gates", differences)


def diagnose(parent: Mapping[str, Any]) -> dict[str, Any]:
    records = parent["records"]
    score_index = parent["score_index"]
    matrix_cache = parent["matrix_cache"]
    derived_differences: list[dict[str, Any]] = []
    groups: list[str] = []
    per_record: list[dict[str, Any]] = []
    own_rows: list[dict[str, Any]] = []
    c_flag_ids = {name: [] for name in ("baseline_invalid", "boundary_peak", "unclear_peak", "offset_error")}
    counts = {"repeatability": 0, "r_baseline": 0, "gn_baseline": 0, "gnr_offset": 0, "own_offset": 0, "damage_positive": 0}
    check_counts = {name: 0 for name in ("A", "B", "C", "O")}
    gnr_c: list[float] = []
    gnr_d: list[float] = []
    own_c: list[float] = []
    own_d: list[float] = []
    damage_c: list[float] = []
    damage_d: list[float] = []
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        groups.append(group)
        timing = reconstruct_timing(int(record["natural_sample_count"]), record["predicted_frame_counts"])
        stored_masks = record.get("masks")
        expected_masks = {key: timing[key] for key in ("sample_count", "q_frames", "candidate_rows", "common_window_rows", "plus_rows", "minus_rows", "d_by_row", "a_by_row", "plus_expected_offset", "minus_expected_offset", "plus_expected_video_response", "minus_expected_video_response", "forward_mapping_sha256", "forward_mapping_dtype", "forward_mapping_length")}
        compare_values(stored_masks, expected_masks, f"protocol[{sample_id}].masks", derived_differences, tolerance=1e-6)
        evidence: dict[str, dict[str, Any]] = {}
        names = {
            "rn": ("R", "N", False),
            "rw": ("R", "W", False),
            "gnn": ("G_N", "N", False),
            "gnw": ("G_N", "W", False),
            "gwn": ("G_W", "N", False),
            "gww": ("G_W", "W", False),
            "gnr": ("G_NR", "N", False),
            "rn_repeat": ("R", "N", True),
            "gnn_repeat": ("G_N", "N", True),
        }
        for name, (video, audio, repeat) in names.items():
            key = (sample_id, video, audio, repeat)
            evidence[name] = _evidence(matrix_cache[str(score_index[key]["matrix_sha256"])], record, timing, video)
            _compare_score_derived(score_index[key], evidence[name], derived_differences, "/".join((sample_id, video, audio, str(repeat))))
        repeat_r = _baseline_info(evidence["rn"], evidence["rn_repeat"])
        repeat_gn = _baseline_info(evidence["gnn"], evidence["gnn_repeat"])
        repeat_pass = bool(repeat_r["passes"] and repeat_gn["passes"])
        counts["repeatability"] += int(repeat_pass)
        baseline_r = _baseline_explainable(evidence["rn"])
        baseline_gn = _baseline_explainable(evidence["gnn"])
        counts["r_baseline"] += int(baseline_r["passes"])
        counts["gn_baseline"] += int(baseline_gn["passes"])
        check_a = _check_pair(evidence["rw"], evidence["rn"], timing, ("plus_expected_offset", "minus_expected_offset"))
        check_b = _check_pair(evidence["gnw"], evidence["gnn"], timing, ("plus_expected_offset", "minus_expected_offset"))
        check_c = _check_pair(evidence["gwn"], evidence["gnn"], timing, ("plus_expected_video_response", "minus_expected_video_response"))
        check_o = _check_pair(evidence["gww"], evidence["gnn"], {"plus_expected_offset": 0.0, "minus_expected_offset": 0.0}, ("plus_expected_offset", "minus_expected_offset"))
        for name, value, baseline in (("A", check_a, baseline_r), ("B", check_b, baseline_gn), ("C", check_c, baseline_gn), ("O", check_o, baseline_gn)):
            value["passes"] = bool(value["passes"] and baseline["passes"])
            check_counts[name] += int(value["passes"])
        c_diagnostic = _c_diagnostic(evidence["gnn"], evidence["gwn"], timing, baseline_gn)
        for flag, present in c_diagnostic["flags"].items():
            if present:
                c_flag_ids[flag].append(sample_id)
        gnr_c_value = float(evidence["gnr"]["common_global"]["sync_c"] - evidence["gnn"]["common_global"]["sync_c"])
        gnr_d_value = float(evidence["gnn"]["common_global"]["sync_d"] - evidence["gnr"]["common_global"]["sync_d"])
        gnr_c.append(gnr_c_value)
        gnr_d.append(gnr_d_value)
        gnr_offset = int(evidence["gnr"]["common_global"]["offset"]) - int(evidence["gnn"]["common_global"]["offset"])
        counts["gnr_offset"] += int(abs(gnr_offset) <= config.OFFSET_TOLERANCE_FRAMES)
        own_c_value = float(evidence["gww"]["common_global"]["sync_c"] - evidence["gnn"]["common_global"]["sync_c"])
        own_d_value = float(evidence["gnn"]["common_global"]["sync_d"] - evidence["gww"]["common_global"]["sync_d"])
        n_curve = evidence["gnn"]["common_global"]["curve"]
        w_curve = evidence["gww"]["common_global"]["curve"]
        median_change = float(np.median(w_curve) - np.median(n_curve))
        identity_error = own_c_value - (median_change + own_d_value)
        if abs(identity_error) > 1e-9:
            raise DiagnosticError(f"own-audio decomposition identity failed: {sample_id}")
        own_c.append(own_c_value)
        own_d.append(own_d_value)
        own_offset = _offset_noninferiority(evidence["gww"], evidence["gnn"])
        counts["own_offset"] += int(own_offset)
        own_row = {
            "sample_id": sample_id,
            "source_group": group,
            "N_curve": list(n_curve),
            "W_curve": list(w_curve),
            "median_N": float(np.median(n_curve)),
            "median_W": float(np.median(w_curve)),
            "median_change": median_change,
            "own_C": own_c_value,
            "own_D": own_d_value,
            "identity_error": identity_error,
            "N": {key: evidence["gnn"]["common_global"][key] for key in ("offset", "sync_c", "sync_d")},
            "W": {key: evidence["gww"]["common_global"][key] for key in ("offset", "sync_c", "sync_d")},
            "offset_noninferior": own_offset,
        }
        own_rows.append(own_row)
        damage_c_value = float(evidence["gww"]["common_global"]["sync_c"] - evidence["gwn"]["common_global"]["sync_c"])
        damage_d_value = float(evidence["gwn"]["common_global"]["sync_d"] - evidence["gww"]["common_global"]["sync_d"])
        damage_c.append(damage_c_value)
        damage_d.append(damage_d_value)
        damage_positive = damage_c_value > 0.0 and damage_d_value > 0.0
        counts["damage_positive"] += int(damage_positive)
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": group,
                "repeatability": {"R_N": repeat_r, "G_N_N": repeat_gn, "passes": repeat_pass},
                "baseline": {"R_N": baseline_r, "G_N_N": baseline_gn},
                "checks": {"A": check_a, "B": check_b, "C": check_c, "O": check_o},
                "generated_repeat": {"c_difference": gnr_c_value, "d_difference": gnr_d_value, "offset_difference": gnr_offset},
                "own_audio": {"c": own_c_value, "d": own_d_value, "offset_noninferior": own_offset},
                "replacement_damage": {"c": damage_c_value, "d": damage_d_value, "both_positive": damage_positive},
                "c_diagnostic": c_diagnostic,
            }
        )
    gnr_ci_c = cluster_bootstrap(gnr_c, groups)
    gnr_ci_d = cluster_bootstrap(gnr_d, groups)
    own_ci_c = cluster_bootstrap(own_c, groups)
    own_ci_d = cluster_bootstrap(own_d, groups)
    damage_ci_c = cluster_bootstrap(damage_c, groups)
    damage_ci_d = cluster_bootstrap(damage_d, groups)
    gates = {
        "repeatability": {"count": counts["repeatability"], "required": config.EXPECTED_RECORD_COUNT, "passes": counts["repeatability"] == config.EXPECTED_RECORD_COUNT},
        "baseline": {"R_N_count": counts["r_baseline"], "G_N_N_count": counts["gn_baseline"], "required_each": config.MIN_BASELINE_RECORDS, "passes": counts["r_baseline"] >= config.MIN_BASELINE_RECORDS and counts["gn_baseline"] >= config.MIN_BASELINE_RECORDS},
        "generated_repeat": {"c_difference": gnr_ci_c, "d_difference_N_minus_NR": gnr_ci_d, "offset_count": counts["gnr_offset"], "required_offset_count": config.MIN_BASELINE_RECORDS, "ci_open_interval": bool(_gate_ci(gnr_ci_c, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and _gate_ci(gnr_ci_d, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True)), "passes": bool(_gate_ci(gnr_ci_c, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and _gate_ci(gnr_ci_d, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and counts["gnr_offset"] >= config.MIN_BASELINE_RECORDS)},
        "inherited": {**{name: {"count": check_counts[name], "required": config.MIN_SUCCESS_RECORDS, "denominator": config.EXPECTED_RECORD_COUNT, "passes": check_counts[name] >= config.MIN_SUCCESS_RECORDS} for name in ("A", "B", "C", "O")}, "passes": all(check_counts[name] >= config.MIN_SUCCESS_RECORDS for name in ("A", "B", "C", "O"))},
        "own_audio": {"c": own_ci_c, "d": own_ci_d, "offset_count": counts["own_offset"], "required_offset_count": config.MIN_BASELINE_RECORDS, "passes": bool(_gate_ci(own_ci_c, lower=-0.10, strict_lower=True) and _gate_ci(own_ci_d, lower=-0.10, strict_lower=True) and counts["own_offset"] >= config.MIN_BASELINE_RECORDS)},
        "replacement_damage": {"c": damage_ci_c, "d": damage_ci_d, "both_positive_count": counts["damage_positive"], "required_positive_count": config.MIN_SUCCESS_RECORDS, "passes": bool(_gate_ci(damage_ci_c, lower=0.10, strict_lower=True) and _gate_ci(damage_ci_d, lower=0.10, strict_lower=True) and counts["damage_positive"] >= config.MIN_SUCCESS_RECORDS)},
    }
    control_pass = bool(all(bool(value["passes"]) for value in gates.values()))
    _history_mismatches(parent, per_record, gates, derived_differences)
    assert_finite(per_record)
    diagnostic_decision = "CONTROL_FAILURE_REPRODUCED" if not derived_differences else "AUDIT_MISMATCH"
    return {
        "schema_version": 1,
        "analysis": config.PROTOCOL_ID,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "diagnostic_decision": diagnostic_decision,
        "historical_scientific_decision": "CONTROL_FAILED",
        "recomputed_control_pass": control_pass,
        "record_count": len(records),
        "source_group_count": len(set(groups)),
        "score_count": config.EXPECTED_CELL_COUNT,
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "bridge_executed": False,
        "training_authorized": False,
        "cross_model_spec_eligible": False,
        "generalization_established": False,
        "gates": gates,
        "per_record": per_record,
        "c_failure_summary": {
            "denominator": config.EXPECTED_RECORD_COUNT,
            "failed_count": config.EXPECTED_RECORD_COUNT - check_counts["C"],
            "flag_counts": {name: len(ids) for name, ids in c_flag_ids.items()},
            "flag_ids": c_flag_ids,
        },
        "own_audio_decomposition": {
            "rows": own_rows,
            "C": own_ci_c,
            "D": own_ci_d,
            "C_lower_margin_above_minus_0_10": float(own_ci_c["ci95"][0] + 0.10),
            "D_lower_margin_above_minus_0_10": float(own_ci_d["ci95"][0] + 0.10),
            "offset_count": counts["own_offset"],
            "identity_tolerance": 1e-9,
        },
        "history_comparison": {
            "status": "match" if not derived_differences else "mismatch",
            "difference_count": len(derived_differences),
            "differences": derived_differences[:500],
        },
        "recommendation": (
            "保持视频和音频不变，只对 C 失败记录做一次独立 SyncNet 局部峰复核；若峰/残差仍一致，再把生成器响应不足作为下一单一干预假设；本轮不自动执行。"
            if not derived_differences
            else "先修复或解释列出的字段差异，再决定是否进行任何生成实验；本轮不自动执行。"
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device_policy": "CPU-only; no CUDA or media generation",
        },
    }
