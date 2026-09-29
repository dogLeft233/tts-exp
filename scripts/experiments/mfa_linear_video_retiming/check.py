from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .common import (
    SYNCNET_REQUEST_EXEC_CODE,
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    verify_json,
    write_json,
)
from .config import load_config


def rebuild_q_independently(mapping: Mapping[str, Any]) -> np.ndarray:
    total = int(mapping["frame_count"])
    valid = int(mapping["valid_frame_count"])
    positions = np.asarray(mapping["knot_positions"], dtype=np.int64)
    values = np.asarray(mapping["knot_deltas"], dtype=np.float64)
    if total < 1 or valid < 1 or valid > total or positions.ndim != 1 or values.shape != positions.shape:
        raise ProtocolError("CHECK_MAP_SHAPE_INVALID")
    if positions.size > 16 or (positions.size and (positions[0] < 5 or positions[-1] > valid - 6)):
        raise ProtocolError("CHECK_MAP_KNOT_SUPPORT_INVALID")
    if positions.size > 1 and np.any(np.diff(positions) < 12):
        raise ProtocolError("CHECK_MAP_KNOT_GAP_INVALID")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 3.0 + 1e-12):
        raise ProtocolError("CHECK_MAP_KNOT_DELTA_INVALID")
    if np.any(np.abs(values * 4.0 - np.rint(values * 4.0)) > 1e-10):
        raise ProtocolError("CHECK_MAP_KNOT_GRID_INVALID")
    frame_index = np.arange(total, dtype=np.float64)
    delta = np.zeros(total, dtype=np.float64)
    if positions.size:
        delta[:valid] = np.interp(frame_index[:valid], positions.astype(np.float64), values)
    delta[:5] = 0.0
    delta[max(5, valid - 5):] = 0.0
    delta[valid:] = 0.0
    q = frame_index + delta
    if np.any(np.abs(delta) > 3.0 + 1e-12):
        raise ProtocolError("CHECK_MAP_DISPLACEMENT_INVALID")
    if q[0] != 0.0 or q[-1] != total - 1 or np.any(q < 0) or np.any(q > total - 1):
        raise ProtocolError("CHECK_MAP_END_TIME_OR_BOUNDS_INVALID")
    if total > 1:
        speed = np.diff(q)
        if np.any(speed < 0.5 - 1e-12) or np.any(speed > 1.5 + 1e-12) or np.any(np.abs(np.diff(speed)) > 0.5 + 1e-12):
            raise ProtocolError("CHECK_MAP_SPEED_OR_SLOPE_INVALID")
    for key, rebuilt in (("q", q), ("delta", delta)):
        saved = np.asarray(mapping.get(key, []), dtype=np.float64)
        if saved.shape != rebuilt.shape or not np.array_equal(saved, rebuilt):
            raise ProtocolError(f"CHECK_MAP_{key.upper()}_RECONSTRUCTION_MISMATCH")
    digest = canonical_json_sha256({"frame_count": total, "valid_frame_count": valid,
                                    "knot_positions": positions.astype(int).tolist(),
                                    "knot_deltas": values.tolist(), "delta": delta.tolist(), "q": q.tolist()})
    if digest != mapping.get("map_sha256"):
        raise ProtocolError("CHECK_MAP_HASH_MISMATCH")
    return q


def _decode_frames(path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(frame))
    cap.release()
    if not frames:
        raise ProtocolError(f"CHECK_VIDEO_EMPTY:{path}")
    return np.stack(frames)


def _render_pixels_independently(source: np.ndarray, q: np.ndarray) -> np.ndarray:
    if source.dtype != np.uint8 or source.ndim != 4 or source.shape[0] != q.size:
        raise ProtocolError("CHECK_SOURCE_PIXEL_SHAPE_INVALID")
    lo = np.floor(q).astype(np.int64)
    hi = np.ceil(q).astype(np.int64)
    alpha = (q - lo).reshape(-1, 1, 1, 1)
    blend = source[lo].astype(np.float64) * (1.0 - alpha) + source[hi].astype(np.float64) * alpha
    return np.floor(blend + 0.5).astype(np.uint8)


def np_distance_matrix(visual: np.ndarray, audio: np.ndarray, support_count: int) -> np.ndarray:
    v = np.asarray(visual, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1:] != (1024,) or a.shape[1:] != (1024,):
        raise ProtocolError("CHECK_EMBEDDING_SHAPE_INVALID")
    if int(support_count) > min(v.shape[0], a.shape[0]):
        raise ProtocolError("CHECK_EMBEDDING_SUPPORT_INVALID")
    out = np.full((int(support_count), 31), np.nan, dtype=np.float32)
    eps = np.float32(1e-6)
    for i in range(int(support_count)):
        for column, lag in enumerate(range(-15, 16)):
            index = i + lag
            if 0 <= index < a.shape[0]:
                difference = np.subtract(v[i], a[index], dtype=np.float32)
                shifted = np.add(difference, eps, dtype=np.float32)
                squared = np.multiply(shifted, shifted, dtype=np.float32)
                out[i, column] = np.sqrt(np.sum(squared, dtype=np.float32), dtype=np.float32)
    return out


def np_metrics(matrix: np.ndarray, support_rows: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    rows = np.asarray(support_rows, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 31 or not rows.size or not np.isfinite(values[rows]).all():
        raise ProtocolError("CHECK_DISTANCE_MATRIX_INVALID")
    curve = np.mean(values[rows], axis=0, dtype=np.float32).astype(np.float64)
    minimum = float(np.min(curve))
    index = int(np.flatnonzero(curve == minimum)[0])
    local = []
    for offset in range(0, len(rows), 25):
        chunk = rows[offset:offset + 25]
        if chunk.size != 25:
            continue
        local_curve = np.mean(values[chunk], axis=0, dtype=np.float32).astype(np.float64)
        local_index = int(np.flatnonzero(local_curve == np.min(local_curve))[0])
        local.append({"d0": float(local_curve[15]), "offset": 15 - local_index})
    if not local:
        raise ProtocolError("CHECK_LOCAL_SUPPORT_MISSING")
    return {"curve": curve.tolist(), "sync_c": float(np.median(curve) - minimum), "sync_d": minimum,
            "d0": float(curve[15]), "offset": 15 - index,
            "local_median_d0": float(np.median([row["d0"] for row in local])),
            "local_abs_offset_q90": float(np.quantile([abs(row["offset"]) for row in local], 0.9, method="linear"))}


def select_candidate_independently(candidates: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], gates: Mapping[str, Any]) -> Mapping[str, Any] | None:
    eligible = []
    for row in candidates:
        metric = row.get("metrics", {})
        if not row.get("feasible"):
            continue
        if float(metric["sync_c"]) < float(baseline["sync_c"]) + float(gates["minimum_sync_c_gain"]):
            continue
        if float(metric["d0"]) > float(baseline["d0"]) - float(gates["minimum_d0_improvement"]):
            continue
        if float(metric["sync_d"]) > float(baseline["sync_d"]) + float(gates["maximum_sync_d_worsening"]):
            continue
        if abs(int(metric["offset"])) > 1:
            continue
        if float(metric["local_median_d0"]) > float(baseline["local_median_d0"]):
            continue
        if float(metric["local_abs_offset_q90"]) > float(baseline["local_abs_offset_q90"]):
            continue
        eligible.append(row)
    if not eligible:
        return None
    max_c = max(float(row["metrics"]["sync_c"]) for row in eligible)
    near_best = [row for row in eligible if float(row["metrics"]["sync_c"]) >= max_c - float(gates["local_sync_c_slack"])]
    return min(near_best, key=lambda row: (float(row["regularization"]), str(row["candidate_id"])))


def _run_fresh_worker(
    config: Mapping[str, Any], run_dir: Path, frozen: Mapping[str, Any], sealed: Mapping[str, Any], *,
    timeout_for_remaining: Callable[[int], int] | None = None,
) -> list[dict[str, Any]]:
    generated = verify_json(run_dir / "01_generation" / "manifest.json", self_hash=True)
    frozen_by_id = {str(row["sample_id"]): row for row in frozen["records"]}
    generated_by_id = {str(row["sample_id"]): row for row in generated["records"]}
    rows = []
    total_forwards = 2 * len(sealed["records"])
    for sealed_record in sealed["records"]:
        sid = str(sealed_record["sample_id"])
        box = frozen["portrait_bindings"]["portraits"]["3"]["score_box"]["box"]
        valid = int(sealed_record["valid_frame_count"])
        audio_n = frozen_by_id[sid]["audio"]["natural"]
        for arm, video in (("M", sealed_record["videos"]["M"]["path"]), ("R", sealed_record["videos"]["R"]["path"])):
            base = run_dir / "07_check" / "fresh" / sid / arm
            base.mkdir(parents=True, exist_ok=True)
            request = {"mode": "score_video", "sample_id": sid, "paired_key": sealed_record["paired_key"],
                       "portrait_id": "3", "video_arm": arm, "audio_role": "N", "video": video,
                       "audio": audio_n["path"], "video_sha256": file_sha256(video),
                       "audio_sha256": audio_n["sha256"], "score_box_xyxy": box,
                       "valid_frame_count": valid, "batch_size": int(config["models"]["syncnet_batch_size"]),
                       "syncnet_root": config["paths"]["syncnet_root"], "syncnet_model": config["paths"]["syncnet_model"],
                       "expected_syncnet_sha256": config["models"]["syncnet_model_sha256"],
                       "embedding_dir": str((base / "embeddings").resolve()), "result_path": str((base / "result.json").resolve())}
            request_path = base / "request.json"
            write_json(request_path, request)
            command = [str(Path(config["paths"]["syncnet_python"]).absolute()), "-c", SYNCNET_REQUEST_EXEC_CODE, str(request_path)]
            timeout = int(config["official"]["syncnet_timeout_seconds"])
            if timeout_for_remaining is not None:
                timeout = int(timeout_for_remaining(total_forwards - len(rows)))
                if timeout < 1:
                    raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
            with (base / "worker.log").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(command) + "\n")
                try:
                    result = subprocess.run(command, cwd=config["repo_root"], stdout=handle, stderr=subprocess.STDOUT,
                                            timeout=timeout, check=False)
                except subprocess.TimeoutExpired as exc:
                    handle.write(f"\n[TIMEOUT] after {timeout} seconds\n")
                    if timeout_for_remaining is not None:
                        raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED") from exc
                    raise ProtocolError(f"CHECK_FRESH_FORWARD_TIMEOUT:{sid}:{arm}") from exc
                handle.write(f"\n[returncode] {result.returncode}\n")
            if result.returncode:
                raise ProtocolError(f"CHECK_FRESH_FORWARD_FAILED:{sid}:{arm}")
            rows.append(verify_json(request["result_path"], self_hash=True))
    return rows


def _fresh_metric_parity(config: Mapping[str, Any], run_dir: Path, sealed: Mapping[str, Any], fresh: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by = {(str(row["sample_id"]), str(row["video_arm"])): row for row in fresh}
    out = []
    for record in sealed["records"]:
        sid = str(record["sample_id"])
        state = verify_json(run_dir / "03_search" / sid / "state.json", self_hash=True)
        baseline_candidate = state["candidates"]["identity"]
        candidate_state = (verify_json(run_dir / "03_search" / sid / "state_v2.json", self_hash=True)
                           if config["protocol"].endswith("_v2") else state)
        selected = (_candidate_for_map_sha(candidate_state, str(record["map_sha256"]), sample_id=sid)
                    if str(record["map_sha256"]) != str(baseline_candidate["map_sha256"]) else baseline_candidate)
        frozen_audio = np.load(state["natural_audio_embedding_path"], allow_pickle=False)["audio"]
        frozen_w = [int(value) for value in state["frozen_support"]]
        for arm, candidate in (("M", baseline_candidate), ("R", selected)):
            fresh_row = by[(sid, arm)]
            archive = np.load(fresh_row["embedding_path"], allow_pickle=False)
            fresh_visual, fresh_audio = np.asarray(archive["visual"], dtype=np.float32), np.asarray(archive["audio"], dtype=np.float32)
            saved = np.load(candidate["embedding_path"], allow_pickle=False)
            expected_visual = np.asarray(saved["visual"], dtype=np.float32)
            if fresh_audio.shape != frozen_audio.shape or float(np.max(np.abs(fresh_audio - frozen_audio))) > 1e-4:
                raise ProtocolError(f"CHECK_FRESH_AUDIO_EMBEDDING_MISMATCH:{sid}:{arm}")
            if fresh_visual.shape != expected_visual.shape or float(np.max(np.abs(fresh_visual - expected_visual))) > 1e-4:
                raise ProtocolError(f"CHECK_FRESH_VISUAL_EMBEDDING_MISMATCH:{sid}:{arm}")
            n = int(state["baseline_candidate"]["metrics"]["conservative_window_count"])
            fresh_matrix = np_distance_matrix(fresh_visual, fresh_audio, n)
            saved_matrix = np_distance_matrix(expected_visual, frozen_audio, n)
            if float(np.nanmax(np.abs(fresh_matrix - saved_matrix))) > 1e-4:
                raise ProtocolError(f"CHECK_FRESH_DISTANCE_MATRIX_MISMATCH:{sid}:{arm}")
            fresh_metrics = np_metrics(fresh_matrix, frozen_w)
            saved_metrics = np_metrics(saved_matrix, frozen_w)
            for name in ("sync_c", "sync_d", "d0"):
                if abs(float(fresh_metrics[name]) - float(saved_metrics[name])) > 1e-4:
                    raise ProtocolError(f"CHECK_FRESH_{name.upper()}_MISMATCH:{sid}:{arm}")
            if int(fresh_metrics["offset"]) != int(saved_metrics["offset"]):
                raise ProtocolError(f"CHECK_FRESH_OFFSET_MISMATCH:{sid}:{arm}")
            out.append({"sample_id": sid, "video_arm": arm, "embedding_max_abs": float(np.max(np.abs(fresh_visual - expected_visual))),
                        "distance_matrix_max_abs": float(np.nanmax(np.abs(fresh_matrix - saved_matrix))),
                        "metrics": fresh_metrics, "status": "PASS"})
    return out


def _candidate_for_map_sha(state: Mapping[str, Any], map_sha256: str, *, sample_id: str) -> Mapping[str, Any]:
    candidates = state.get("true_candidates", state.get("candidates"))
    if not isinstance(candidates, Mapping):
        raise ProtocolError(f"CHECK_CANDIDATE_STATE_MISSING:{sample_id}")
    matches = [candidate for candidate in candidates.values()
               if isinstance(candidate, Mapping) and str(candidate.get("map_sha256")) == map_sha256]
    if not matches:
        raise ProtocolError(f"CHECK_SELECTED_MAP_CANDIDATE_MISSING:{sample_id}:{map_sha256}")
    maps = {canonical_json_sha256(candidate.get("map")) for candidate in matches}
    if len(maps) != 1:
        raise ProtocolError(f"CHECK_SELECTED_MAP_HASH_COLLISION:{sample_id}:{map_sha256}")
    return min(matches, key=lambda candidate: str(candidate.get("candidate_id", "")))


def _verify_sealed_pixels(run_dir: Path, sealed: Mapping[str, Any]) -> list[dict[str, Any]]:
    audits = []
    for record in sealed["records"]:
        source = _decode_frames(record["source_M"]["canonical_video"])
        mapping = record["selected_map"]
        q = rebuild_q_independently(mapping)
        if source.shape[0] != q.size:
            raise ProtocolError("CHECK_SEALED_SOURCE_LENGTH_MISMATCH")
        expected = _render_pixels_independently(source, q)
        path = Path(record["videos"]["R"]["path"])
        actual = _decode_frames(path)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ProtocolError(f"CHECK_RETIMED_PIXELS_MISMATCH:{record['sample_id']}")
        if file_sha256(path) != record["videos"]["R"]["sha256"]:
            raise ProtocolError("CHECK_SEALED_VIDEO_HASH_MISMATCH")
        nearest_index = np.floor(q + 0.5).astype(np.int64)
        nearest_expected = source[nearest_index]
        nearest = _decode_frames(record["videos"]["NEAREST"]["path"])
        if not np.array_equal(nearest, nearest_expected):
            raise ProtocolError(f"CHECK_NEAREST_PIXELS_MISMATCH:{record['sample_id']}")
        identity = _decode_frames(record["videos"]["M"]["path"])
        if not np.array_equal(identity, source):
            raise ProtocolError(f"CHECK_IDENTITY_PIXELS_MISMATCH:{record['sample_id']}")
        global_map = (record.get("global_control") or {}).get("map")
        if global_map:
            global_q = rebuild_q_independently(global_map)
            expected_global = _render_pixels_independently(source, global_q)
            actual_global = _decode_frames(record["videos"]["GLOBAL"]["path"])
            if not np.array_equal(actual_global, expected_global):
                raise ProtocolError(f"CHECK_GLOBAL_PIXELS_MISMATCH:{record['sample_id']}")
        mirror_row = record["videos"].get("MIRROR")
        if mirror_row is not None:
            mirror_positions = np.asarray(mapping["knot_positions"], dtype=np.int64)
            mirror_values = -np.asarray(mapping["knot_deltas"], dtype=np.float64)
            frame_index = np.arange(int(mapping["frame_count"]), dtype=np.float64)
            mirror_delta = np.zeros(frame_index.size, dtype=np.float64)
            valid_count = int(mapping["valid_frame_count"])
            if mirror_positions.size:
                mirror_delta[:valid_count] = np.interp(
                    frame_index[:valid_count], mirror_positions.astype(np.float64), mirror_values)
            mirror_delta[:5] = 0.0
            mirror_delta[max(5, valid_count - 5):] = 0.0
            mirror_delta[valid_count:] = 0.0
            mirror_q = frame_index + mirror_delta
            expected_mirror = _render_pixels_independently(source, mirror_q)
            actual_mirror = _decode_frames(mirror_row["path"])
            if not np.array_equal(actual_mirror, expected_mirror):
                raise ProtocolError(f"CHECK_MIRROR_PIXELS_MISMATCH:{record['sample_id']}")
        audits.append({"sample_id": str(record["sample_id"]), "q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                       "R_pixels_exact": True, "M_identity_exact": True, "global_control_exact": bool(global_map),
                       "nearest_pixels_exact": True, "mirror_pixels_exact": mirror_row is not None,
                       "frame_count": int(expected.shape[0])})
    return audits


def _independent_wav2lip_parameter_sha256(config: Mapping[str, Any]) -> str:
    worker = Path(__file__).with_name("model_binding_worker.py").resolve()
    command = [str(Path(config["paths"]["wav2lip_python"]).absolute()), str(worker),
               "--checkpoint", str(Path(config["paths"]["wav2lip_checkpoint"]).resolve()),
               "--wav2lip-root", str(Path(config["paths"]["wav2lip_root"]).resolve())]
    result = subprocess.run(command, cwd=str(Path(config["repo_root"]).resolve()), capture_output=True,
                            text=True, timeout=300, check=False)
    if result.returncode:
        raise ProtocolError(f"CHECK_WAV2LIP_PARAMETER_RELOAD_FAILED:{result.stderr[-1200:]}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        digest = str(payload["loaded_parameter_sha256"])
    except (IndexError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError("CHECK_WAV2LIP_PARAMETER_RELOAD_OUTPUT_INVALID") from exc
    if len(digest) != 64:
        raise ProtocolError("CHECK_WAV2LIP_PARAMETER_RELOAD_OUTPUT_INVALID")
    return digest


def _verify_generation_independently(
    generated: Mapping[str, Any], frozen: Mapping[str, Any], config: Mapping[str, Any], *, portrait_ids: Sequence[str],
) -> list[dict[str, Any]]:
    import cv2

    frozen_by = {str(row["sample_id"]): row for row in frozen["records"]}
    if generated.get("status") != "COMPLETE" or generated.get("audio_roles") != {"N": "natural", "M": "mfa_linear"}:
        raise ProtocolError("CHECK_GENERATION_MANIFEST_ROLE_OR_STATUS_INVALID")
    if [str(value) for value in generated.get("portrait_ids", [])] != [str(value) for value in portrait_ids]:
        raise ProtocolError("CHECK_GENERATION_PORTRAIT_DENOMINATOR_INVALID")
    checkpoint_sha = file_sha256(config["paths"]["wav2lip_checkpoint"])
    parameter_sha = _independent_wav2lip_parameter_sha256(config)
    expected_python = Path(config["paths"]["wav2lip_python"]).absolute()
    expected_prefix = expected_python.parent.parent.resolve()
    wav2lip_root = Path(config["paths"]["wav2lip_root"]).resolve()
    expected_sources = {
        "audio_module": (wav2lip_root / "audio.py").resolve(),
        "models_module": (wav2lip_root / "models" / "__init__.py").resolve(),
        "wav2lip_class_source": (wav2lip_root / "models" / "wav2lip.py").resolve(),
    }
    frozen_portraits = frozen["portrait_bindings"]["portraits"]
    audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in generated.get("records", []):
        sid = str(sample["sample_id"])
        if sid in seen or sid not in frozen_by:
            raise ProtocolError(f"CHECK_GENERATION_SAMPLE_ID_INVALID:{sid}")
        seen.add(sid)
        source = frozen_by[sid]
        if (str(sample.get("paired_key")) != str(source["paired_key"])
                or str(sample.get("speaker_id")) != str(source["speaker_id"])):
            raise ProtocolError(f"CHECK_GENERATION_SAMPLE_IDENTITY_MISMATCH:{sid}")
        if set(sample.get("portraits", {})) != {str(value) for value in portrait_ids}:
            raise ProtocolError(f"CHECK_GENERATION_PORTRAIT_SET_MISMATCH:{sid}")
        for portrait_id in portrait_ids:
            image_row = frozen_portraits[str(portrait_id)]
            image_path = Path(image_row["path"]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ProtocolError(f"CHECK_GENERATION_PORTRAIT_UNREADABLE:{portrait_id}")
            rgb_sha = hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            if rgb_sha != image_row["rgb_pixel_sha256"]:
                raise ProtocolError(f"CHECK_GENERATION_PORTRAIT_PIXEL_HASH_MISMATCH:{portrait_id}")
            box = [int(value) for value in image_row["generation_box_xyxy"]]
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ProtocolError(f"CHECK_GENERATION_BOX_INVALID:{portrait_id}")
            outside = np.ones(image.shape[:2], dtype=bool)
            outside[y1:y2, x1:x2] = False
            for role in ("N", "M"):
                receipt = sample["portraits"][str(portrait_id)][role]
                video = Path(receipt["canonical_video"]).resolve()
                if not video.is_file() or file_sha256(video) != receipt.get("canonical_video_sha256"):
                    raise ProtocolError(f"CHECK_CANONICAL_VIDEO_HASH_MISMATCH:{sid}:{portrait_id}:{role}")
                worker_path = Path(receipt["worker_receipt"]).resolve()
                if not worker_path.is_file() or file_sha256(worker_path) != receipt.get("worker_receipt_sha256"):
                    raise ProtocolError(f"CHECK_WAV2LIP_WORKER_RECEIPT_HASH_MISMATCH:{sid}:{portrait_id}:{role}")
                worker = verify_json(worker_path, self_hash=False)
                audio = source["audio"]["natural" if role == "N" else "mfa_linear"]
                audio_path = Path(audio["path"]).resolve()
                if not audio_path.is_file() or file_sha256(audio_path) != audio["sha256"]:
                    raise ProtocolError(f"CHECK_FROZEN_AUDIO_HASH_MISMATCH:{sid}:{role}")
                if (Path(str(worker.get("audio", ""))).resolve() != audio_path
                        or worker.get("audio_sha256") != audio["sha256"]
                        or Path(str(worker.get("image", ""))).resolve() != image_path
                        or worker.get("image_rgb_sha256") != image_row["rgb_pixel_sha256"]
                        or worker.get("box_xyxy") != box
                        or worker.get("input_mode") != "one_png_only"):
                    raise ProtocolError(f"CHECK_WAV2LIP_INPUT_ROLE_MISMATCH:{sid}:{portrait_id}:{role}")
                if (worker.get("checkpoint_sha256") != checkpoint_sha
                        or worker.get("loaded_parameter_sha256") != parameter_sha):
                    raise ProtocolError(f"CHECK_WAV2LIP_LOADED_MODEL_MISMATCH:{sid}:{portrait_id}:{role}")
                if (Path(str(worker.get("python_executable", ""))).absolute() != expected_python
                        or Path(str(worker.get("python_realpath", ""))).resolve() != expected_python.resolve()
                        or Path(str(worker.get("python_prefix", ""))).resolve() != expected_prefix):
                    raise ProtocolError(f"CHECK_WAV2LIP_PYTHON_ENVIRONMENT_MISMATCH:{sid}:{portrait_id}:{role}")
                if worker.get("outside_generation_box_identity") is not True:
                    raise ProtocolError(f"CHECK_WAV2LIP_OUTSIDE_BOX_RECEIPT_INVALID:{sid}:{portrait_id}:{role}")
                rendered_count = int(worker.get("frames_rendered", 0))
                if list(worker.get("source_frame_indices", [])) != [0] * rendered_count:
                    raise ProtocolError(f"CHECK_WAV2LIP_NOT_STATIC_FRAME_INPUT:{sid}:{portrait_id}:{role}")
                for name, expected_path in expected_sources.items():
                    if (Path(str(worker.get(name, ""))).resolve() != expected_path
                            or worker.get(f"{name}_sha256") != file_sha256(expected_path)):
                        raise ProtocolError(f"CHECK_WAV2LIP_SOURCE_BINDING_MISMATCH:{name}")
                frames = _decode_frames(video)
                target_count, expected_valid = _generation_frame_counts(source["audio"], rendered_count)
                if (frames.shape[0] != target_count
                        or int(receipt.get("valid_frame_count", -1)) != expected_valid
                        or int(receipt.get("canonical_frame_count", -1)) != target_count):
                    raise ProtocolError(f"CHECK_GENERATION_FRAME_SUPPORT_MISMATCH:{sid}:{portrait_id}:{role}")
                if not np.array_equal(frames[:, outside], np.broadcast_to(image[outside], frames[:, outside].shape)):
                    raise ProtocolError(f"CHECK_GENERATION_OUTSIDE_FACE_PIXELS_CHANGED:{sid}:{portrait_id}:{role}")
                audits.append({"sample_id": sid, "portrait_id": str(portrait_id), "audio_role": role,
                               "checkpoint_and_parameters": "PASS", "static_source_and_outside_pixels": "PASS",
                               "canonical_frames": target_count, "status": "PASS"})
    if seen != set(frozen_by):
        raise ProtocolError(f"CHECK_GENERATION_SAMPLE_DENOMINATOR_MISMATCH:{len(seen)}/{len(frozen_by)}")
    return audits


def _generation_frame_counts(audio: Mapping[str, Any], rendered_count: int) -> tuple[int, int]:
    """Derive canonical and valid frame counts from the frozen audio record.

    ``sample_count`` belongs to the paired audio record; the nested natural
    and MFA objects only bind their paths and hashes.
    """

    sample_count = int(audio["sample_count"])
    rendered = int(rendered_count)
    if sample_count < 1 or rendered < 0:
        raise ProtocolError("CHECK_GENERATION_AUDIO_OR_RENDER_COUNT_INVALID")
    target_count = (sample_count + 639) // 640
    return target_count, min(target_count, rendered)


def _expected_official_cells(
    frozen: Mapping[str, Any], sealed: Mapping[str, Any], transfer: Mapping[str, Any], *, engineering_only: bool,
) -> list[dict[str, Any]]:
    frozen_by = {str(row["sample_id"]): row for row in frozen["records"]}
    sealed_by = {str(row["sample_id"]): row for row in sealed["records"]}
    transfer_by = {str(row["sample_id"]): row for row in transfer["records"]}
    expected: list[dict[str, Any]] = []
    for sealed_row in sealed["records"]:
        sid = str(sealed_row["sample_id"])
        source = frozen_by[sid]
        audio_n = str(Path(source["audio"]["natural"]["path"]).resolve())
        audio_m = str(Path(source["audio"]["mfa_linear"]["path"]).resolve())
        videos = sealed_row["videos"]
        for arm in ("N", "M", "R", "GLOBAL", "NEAREST", "MIRROR"):
            video = videos[arm]
            expected.append({
                "cell_key": f"p3_s{sid}_v{arm}_aN", "sample_id": sid,
                "paired_key": str(source["paired_key"]), "speaker_id": str(source["speaker_id"]),
                "portrait_id": "3", "video_arm": arm, "audio_role": "N",
                "video": None if video is None else str(Path(video["path"]).resolve()), "audio": audio_n,
                "status": "UNAVAILABLE" if video is None else "PASS",
            })
        for arm in ("M", "R"):
            video = videos[arm]
            expected.append({
                "cell_key": f"p3_s{sid}_v{arm}_aM", "sample_id": sid,
                "paired_key": str(source["paired_key"]), "speaker_id": str(source["speaker_id"]),
                "portrait_id": "3", "video_arm": arm, "audio_role": "M",
                "video": str(Path(video["path"]).resolve()), "audio": audio_m, "status": "PASS",
            })
        if not engineering_only:
            for portrait_id in ("6", "9"):
                portrait = transfer_by[sid]["portraits"][portrait_id]
                portrait_videos = {
                    "N": portrait["N"]["canonical_video"],
                    "M": portrait["M"]["canonical_video"],
                    "R": portrait["R"]["path"],
                }
                for arm in ("N", "M", "R"):
                    expected.append({
                        "cell_key": f"p{portrait_id}_s{sid}_v{arm}_aN", "sample_id": sid,
                        "paired_key": str(source["paired_key"]), "speaker_id": str(source["speaker_id"]),
                        "portrait_id": portrait_id, "video_arm": arm, "audio_role": "N",
                        "video": str(Path(portrait_videos[arm]).resolve()), "audio": audio_n, "status": "PASS",
                    })
    return expected


def _load_official_matrix_independently(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
    except Exception as exc:
        raise ProtocolError(f"CHECK_OFFICIAL_ACTIVESD_UNREADABLE:{path}") from exc
    found: list[np.ndarray] = []

    def visit(value: Any) -> None:
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray([], dtype=np.float32)
        if array.ndim == 2 and 31 in array.shape and np.issubdtype(array.dtype, np.number):
            matrix = np.asarray(array, dtype=np.float32)
            found.append(matrix if matrix.shape[0] == 31 else matrix.T)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(payload)
    unique = {(item.shape, hashlib.sha256(item.tobytes()).hexdigest()): item for item in found}
    if len(unique) != 1:
        raise ProtocolError(f"CHECK_OFFICIAL_ACTIVESD_MATRIX_AMBIGUOUS:{[item.shape for item in unique.values()]}")
    matrix = next(iter(unique.values()))
    if matrix.shape[0] != 31 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ProtocolError("CHECK_OFFICIAL_ACTIVESD_MATRIX_INVALID")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _official_log_metrics(result: Mapping[str, Any]) -> dict[str, float | int]:
    sync_logs = [Path(str(row["log"])) for row in result.get("commands", [])
                 if any("run_syncnet.py" in str(value) for value in row.get("argv", []))]
    if len(sync_logs) != 1 or not sync_logs[0].is_file():
        raise ProtocolError("CHECK_OFFICIAL_SYNCNET_LOG_BINDING_INVALID")
    text = sync_logs[0].read_text(encoding="utf-8", errors="replace")
    patterns = {
        "sync_c": re.compile(r"Confidence:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"),
        "sync_d": re.compile(r"Min dist:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"),
        "offset": re.compile(r"AV offset:\s*(-?\d+)"),
    }
    matches = {name: pattern.findall(text) for name, pattern in patterns.items()}
    if any(len(values) != 1 for values in matches.values()):
        raise ProtocolError("CHECK_OFFICIAL_SYNCNET_LOG_PARSE_INVALID")
    parsed: dict[str, float | int] = {"sync_c": float(matches["sync_c"][0]),
                                    "sync_d": float(matches["sync_d"][0]),
                                    "offset": int(matches["offset"][0])}
    if not np.isfinite(parsed["sync_c"]) or not np.isfinite(parsed["sync_d"]) or abs(int(parsed["offset"])) == 15:
        raise ProtocolError("CHECK_OFFICIAL_SYNCNET_LOG_VALUE_INVALID")
    return parsed


def _probe_streams(ffprobe: str, path: Path) -> list[dict[str, Any]]:
    command = [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise ProtocolError(f"CHECK_OFFICIAL_FFPROBE_FAILED:{path}")
    try:
        return list(json.loads(completed.stdout.decode("utf-8"))["streams"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"CHECK_OFFICIAL_FFPROBE_OUTPUT_INVALID:{path}") from exc


def _probe_video_pts(ffprobe: str, path: Path) -> np.ndarray:
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise ProtocolError(f"CHECK_OFFICIAL_PTS_PROBE_FAILED:{path}")
    try:
        values = [float(row["best_effort_timestamp_time"]) for row in json.loads(completed.stdout.decode("utf-8")).get("frames", [])]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"CHECK_OFFICIAL_PTS_INVALID:{path}") from exc
    if not values or not np.isfinite(values).all():
        raise ProtocolError(f"CHECK_OFFICIAL_PTS_EMPTY:{path}")
    return np.asarray(values, dtype=np.float64)


def _verify_official(
    run_dir: Path, manifest: Mapping[str, Any], config: Mapping[str, Any], *,
    frozen: Mapping[str, Any], sealed: Mapping[str, Any], transfer: Mapping[str, Any], engineering_only: bool,
) -> list[dict[str, Any]]:
    expected = _expected_official_cells(frozen, sealed, transfer, engineering_only=engineering_only)
    rows = list(manifest.get("rows", []))
    expected_keys = [row["cell_key"] for row in expected]
    actual_keys = [str(row.get("cell_key")) for row in rows]
    if manifest.get("status") != "COMPLETE" or actual_keys != expected_keys or len(rows) != len(expected):
        raise ProtocolError(f"CHECK_OFFICIAL_DENOMINATOR_OR_ORDER_MISMATCH:{len(rows)}/{len(expected)}")
    for row, binding in zip(rows, expected, strict=True):
        for field in ("sample_id", "paired_key", "speaker_id", "portrait_id", "video_arm", "audio_role"):
            if str(row.get(field)) != str(binding[field]):
                raise ProtocolError(f"CHECK_OFFICIAL_CELL_IDENTITY_MISMATCH:{binding['cell_key']}:{field}")
        if row.get("status") != binding["status"]:
            raise ProtocolError(f"CHECK_OFFICIAL_CELL_STATUS_MISMATCH:{binding['cell_key']}")
        if binding["status"] == "UNAVAILABLE":
            if row.get("video") is not None or row.get("reason") != "MIRROR_MAP_INFEASIBLE":
                raise ProtocolError(f"CHECK_OFFICIAL_UNAVAILABLE_CELL_TAMPER:{binding['cell_key']}")
        elif (str(Path(str(row.get("video", ""))).resolve()) != binding["video"]
              or str(Path(str(row.get("audio", ""))).resolve()) != binding["audio"]):
            raise ProtocolError(f"CHECK_OFFICIAL_MEDIA_ROLE_MISMATCH:{binding['cell_key']}")
    reports = []
    ffmpeg = str(Path(config["paths"]["ffmpeg"]).resolve())
    for row, binding in zip(rows, expected, strict=True):
        if row.get("status") == "UNAVAILABLE":
            continue
        result_path = run_dir / "06_official" / "cells" / str(row["cell_key"]) / "result.json"
        result = verify_json(result_path, self_hash=True)
        for field in ("cell_key", "sample_id", "paired_key", "speaker_id", "portrait_id", "video_arm", "audio_role"):
            if str(result.get(field)) != str(binding[field]):
                raise ProtocolError(f"CHECK_OFFICIAL_RESULT_IDENTITY_MISMATCH:{row['cell_key']}:{field}")
        if (str(Path(str(result.get("video", ""))).resolve()) != binding["video"]
                or str(Path(str(result.get("audio", ""))).resolve()) != binding["audio"]):
            raise ProtocolError(f"CHECK_OFFICIAL_RESULT_ROLE_SWAP:{row['cell_key']}")
        if file_sha256(result["video"]) != result["video_sha256"] or file_sha256(result["audio"]) != result["audio_sha256"]:
            raise ProtocolError(f"CHECK_OFFICIAL_INPUT_HASH_MISMATCH:{row['cell_key']}")
        if row.get("video_sha256") != result["video_sha256"] or row.get("audio_sha256") != result["audio_sha256"]:
            raise ProtocolError(f"CHECK_OFFICIAL_MANIFEST_HASH_MISMATCH:{row['cell_key']}")
        source_pcm = subprocess.run([ffmpeg, "-v", "error", "-i", result["audio"], "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"], capture_output=True, check=False)
        muxed_path = run_dir / "06_official" / "cells" / str(row["cell_key"]) / "cell.mkv"
        muxed_pcm = subprocess.run([ffmpeg, "-v", "error", "-i", str(muxed_path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"], capture_output=True, check=False)
        if source_pcm.returncode or muxed_pcm.returncode or source_pcm.stdout != muxed_pcm.stdout:
            raise ProtocolError(f"CHECK_OFFICIAL_PCM_MISMATCH:{row['cell_key']}")
        source_video = Path(binding["video"])
        source_frames = _decode_frames(source_video)
        muxed_frames = _decode_frames(muxed_path)
        if source_frames.shape != muxed_frames.shape or not np.array_equal(source_frames, muxed_frames):
            raise ProtocolError(f"CHECK_OFFICIAL_MUX_PIXELS_MISMATCH:{row['cell_key']}")
        source_pts = _probe_video_pts(str(Path(config["paths"]["ffprobe"]).resolve()), source_video)
        muxed_pts = _probe_video_pts(str(Path(config["paths"]["ffprobe"]).resolve()), muxed_path)
        if source_pts.shape != muxed_pts.shape or not np.allclose(source_pts, muxed_pts, atol=1e-7, rtol=0):
            raise ProtocolError(f"CHECK_OFFICIAL_MUX_PTS_MISMATCH:{row['cell_key']}")
        source_streams = _probe_streams(str(Path(config["paths"]["ffprobe"]).resolve()), source_video)
        muxed_streams = _probe_streams(str(Path(config["paths"]["ffprobe"]).resolve()), muxed_path)
        source_video_streams = [item for item in source_streams if item.get("codec_type") == "video"]
        muxed_video_streams = [item for item in muxed_streams if item.get("codec_type") == "video"]
        muxed_audio_streams = [item for item in muxed_streams if item.get("codec_type") == "audio"]
        if len(source_video_streams) != 1 or len(muxed_video_streams) != 1 or len(muxed_audio_streams) != 1:
            raise ProtocolError(f"CHECK_OFFICIAL_MUX_STREAM_COUNT_MISMATCH:{row['cell_key']}")
        video_fields = ("codec_name", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate")
        if any(source_video_streams[0].get(field) != muxed_video_streams[0].get(field) for field in video_fields):
            raise ProtocolError(f"CHECK_OFFICIAL_MUX_VIDEO_STREAM_CHANGED:{row['cell_key']}")
        audio_stream = muxed_audio_streams[0]
        if (audio_stream.get("codec_name") != "pcm_s16le" or int(audio_stream.get("sample_rate", 0)) != 16000
                or int(audio_stream.get("channels", 0)) != 1):
            raise ProtocolError(f"CHECK_OFFICIAL_MUX_AUDIO_STREAM_INVALID:{row['cell_key']}")
        matrix = np.asarray(result["activesd"]["distance_matrix"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != 31 or not np.isfinite(matrix).all():
            raise ProtocolError(f"CHECK_OFFICIAL_ACTIVESD_INVALID:{row['cell_key']}")
        activesd_path = Path(str(result["activesd"]["path"]))
        if (not activesd_path.is_file() or file_sha256(activesd_path) != result["activesd"]["sha256"]
                or row.get("activesd_sha256") != result["activesd"]["sha256"]
                or str(Path(str(row.get("activesd_path", ""))).resolve()) != str(activesd_path.resolve())
                or matrix.shape[1] != int(row.get("official_support_columns", -1))):
            raise ProtocolError(f"CHECK_OFFICIAL_ACTIVESD_BINDING_MISMATCH:{row['cell_key']}")
        raw_matrix = _load_official_matrix_independently(activesd_path)
        if raw_matrix.shape != matrix.shape or not np.array_equal(raw_matrix, matrix):
            raise ProtocolError(f"CHECK_OFFICIAL_ACTIVESD_JSON_MATRIX_MISMATCH:{row['cell_key']}")
        curve = np.mean(matrix, axis=1, dtype=np.float32).astype(np.float64)
        index = int(np.flatnonzero(curve == np.min(curve))[0])
        recomputed = {"sync_c": float(np.median(curve) - np.min(curve)), "sync_d": float(np.min(curve)),
                      "d0": float(curve[15]), "offset": 15 - index}
        saved = result["recomputed_official_curve"]
        for key in ("sync_c", "sync_d", "d0"):
            if abs(recomputed[key] - float(saved[key])) > 1e-7:
                raise ProtocolError(f"CHECK_OFFICIAL_CURVE_TAMPER:{row['cell_key']}:{key}")
        if int(recomputed["offset"]) != int(saved["offset"]) or int(recomputed["offset"]) != int(result["official_offset"]):
            raise ProtocolError(f"CHECK_OFFICIAL_OFFSET_TAMPER:{row['cell_key']}")
        for field in ("sync_c", "sync_d", "d0", "offset"):
            row_field = "official_" + field if field != "offset" else "official_offset"
            if abs(float(row[row_field]) - float(recomputed[field])) > 0.0011:
                raise ProtocolError(f"CHECK_OFFICIAL_MANIFEST_METRIC_MISMATCH:{row['cell_key']}:{field}")
        logged = _official_log_metrics(result)
        if (abs(float(logged["sync_c"]) - recomputed["sync_c"]) > 0.0011
                or abs(float(logged["sync_d"]) - recomputed["sync_d"]) > 0.0011
                or int(logged["offset"]) != int(recomputed["offset"])):
            raise ProtocolError(f"CHECK_OFFICIAL_LOG_MATRIX_PARITY_MISMATCH:{row['cell_key']}")
        reports.append({"cell_key": row["cell_key"], "pcm_exact": True, "video_pixels_and_pts_exact": True,
                        "activesd_loaded_independently": True, "curve_recomputed": True,
                        "official_log_recomputed": True, "metrics": recomputed, "status": "PASS"})
    return reports


def _verify_search_independently(run_dir: Path, frozen: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for record in frozen["records"]:
        sid = str(record["sample_id"])
        state = verify_json(run_dir / "03_search" / sid / "state.json", self_hash=True)
        result = verify_json(run_dir / "03_search" / sid / "result.json", self_hash=True)
        for candidate_id, candidate in state["candidates"].items():
            mapping = candidate["map"]
            rebuild_q_independently(mapping)
            cache = Path(candidate["embedding_path"])
            if not cache.is_file() or file_sha256(cache) != candidate["embedding_sha256"]:
                raise ProtocolError(f"CHECK_CANDIDATE_EMBEDDING_BINDING_MISMATCH:{sid}:{candidate_id}")
            with np.load(cache, allow_pickle=False) as data:
                visual = np.asarray(data["visual"], dtype=np.float32)
            with np.load(state["natural_audio_embedding_path"], allow_pickle=False) as data:
                audio = np.asarray(data["audio"], dtype=np.float32)
            n = int(state["baseline_candidate"]["metrics"]["conservative_window_count"])
            matrix = np_distance_matrix(visual, audio, n)
            metrics = np_metrics(matrix, state["frozen_support"])
            saved_metrics = candidate["metrics"]
            for name in ("sync_c", "sync_d", "d0"):
                if abs(float(metrics[name]) - float(saved_metrics[name])) > 1e-4:
                    raise ProtocolError(f"CHECK_SEARCH_METRIC_MISMATCH:{sid}:{candidate_id}:{name}")
            if int(metrics["offset"]) != int(saved_metrics["offset"]):
                raise ProtocolError(f"CHECK_SEARCH_OFFSET_MISMATCH:{sid}:{candidate_id}")
            if not np.allclose(np.asarray(metrics["curve"]), np.asarray(saved_metrics["curve"]), atol=1e-4, rtol=0):
                raise ProtocolError(f"CHECK_SEARCH_CURVE_MISMATCH:{sid}:{candidate_id}")
        baseline = state["baseline_candidate"]["metrics"]
        choice = select_candidate_independently(list(state["candidates"].values()), baseline, config["search"])
        expected_id = "identity" if choice is None else str(choice["candidate_id"])
        if str(result["selected"]["candidate_id"]) != expected_id:
            raise ProtocolError(f"CHECK_SELECTED_CANDIDATE_MISMATCH:{sid}")
        output.append({"sample_id": sid, "candidate_count": len(state["candidates"]), "selected_candidate_id": expected_id,
                       "independent_selection": True, "status": "PASS"})
    return output


def check_run(
    run_dir: str | Path, *, config_path: str | Path, force_fresh: bool = True,
    timeout_for_remaining: Callable[[int], int] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    config = load_config(config_path)
    protocol = verify_json(root / "00_protocol" / "protocol.json", self_hash=True)
    if protocol.get("protocol") != config["protocol"]:
        raise ProtocolError("CHECK_CONFIG_RUN_PROTOCOL_MISMATCH")
    frozen = verify_json(root / "00_protocol" / "frozen_inputs.json", self_hash=True)
    configured = {key: value for key, value in config.items() if key not in ("config_path", "repo_root")}
    configured["sample_ids"] = list(frozen["sample_ids"])
    configured["portraits"] = list(protocol["portrait_ids"])
    if canonical_json_sha256(configured) != canonical_json_sha256(protocol["run_bindings"]["config"]):
        raise ProtocolError("CHECK_CONFIG_RUN_FROZEN_FIELDS_MISMATCH")
    generated = verify_json(root / "01_generation" / "manifest.json", self_hash=True)
    sealed = verify_json(root / "04_sealed" / "manifest.json", self_hash=True)
    transfer = verify_json(root / "05_transfer" / "manifest.json", self_hash=True)
    official = verify_json(root / "06_official" / "manifest.json", self_hash=True)
    if protocol.get("run_fingerprint") != sealed.get("run_fingerprint"):
        raise ProtocolError("CHECK_RUN_FINGERPRINT_MISMATCH")
    if file_sha256(config["paths"]["wav2lip_checkpoint"]) != protocol["run_bindings"]["models"]["wav2lip_checkpoint_sha256"]:
        raise ProtocolError("CHECK_WAV2LIP_WEIGHT_CHANGED")
    if file_sha256(config["paths"]["syncnet_model"]) != protocol["run_bindings"]["models"]["syncnet_model_sha256"]:
        raise ProtocolError("CHECK_SYNCNET_WEIGHT_CHANGED")
    generation_audits = _verify_generation_independently(
        generated, frozen, config, portrait_ids=[str(value) for value in protocol["portrait_ids"]])
    pixel_audits = _verify_sealed_pixels(root, sealed)
    if config["protocol"].endswith("_v2"):
        from .check_v2 import verify_search_v2
        search_audits = verify_search_v2(root, frozen, config)
    else:
        search_audits = _verify_search_independently(root, frozen, config)
    if force_fresh:
        fresh = _run_fresh_worker(config, root, frozen, sealed, timeout_for_remaining=timeout_for_remaining)
        fresh_audits = _fresh_metric_parity(config, root, sealed, fresh)
        write_json(root / "07_check" / "fresh_scores_manifest.json", {"status": "PASS", "rows": fresh, "count": len(fresh)}, self_hash=True)
    else:
        fresh_audits = []
    official_audits = _verify_official(root, official, config, frozen=frozen, sealed=sealed,
                                       transfer=transfer, engineering_only=bool(protocol["engineering_only"]))
    report_binding_path = root / "09_report" / "report_binding.json"
    report_audit = None
    if report_binding_path.is_file():
        binding = verify_json(report_binding_path, self_hash=True)
        if file_sha256(binding["summary_path"]) != binding["summary_sha256"] or file_sha256(binding["report_path"]) != binding["report_sha256"]:
            raise ProtocolError("CHECK_REPORT_BINDING_MISMATCH")
        summary = verify_json(binding["summary_path"], self_hash=True)
        if int(summary.get("sample_count", -1)) != len(frozen["records"]):
            raise ProtocolError("CHECK_REPORT_SAMPLE_DENOMINATOR_MISMATCH")
        report_audit = {"status": "PASS", "summary_sha256": binding["summary_sha256"], "report_sha256": binding["report_sha256"]}
    result = {"schema_version": 1, "protocol": config["protocol"], "status": "PASS",
              "engineering_decision": "GO", "scientific_status": "NOT_DETERMINED_BY_CHECKER",
              "run_fingerprint": protocol["run_fingerprint"], "independent_checker": True,
              "producer_validate_map_used": False, "producer_score_metrics_used": False,
              "producer_select_candidate_used": False,
              "checks": {"generation_model_call_and_weights": generation_audits, "audio_roles_and_input_hashes": "PASS",
                         "map_reconstructed_from_saved_knots": pixel_audits, "candidate_scores_recomputed_from_embeddings": search_audits,
                         "fresh_process_reload_and_forward": fresh_audits if force_fresh else "NOT_RUN",
                         "official_pcm_and_curve_recomputed": official_audits,
                         "report_binding": report_audit or "NOT_AVAILABLE"}}
    return write_json(root / "07_check" / "validation.json", result, self_hash=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent MFA-linear retiming evidence checker")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("scripts/configs/mfa_linear_video_retiming_v1.yaml"))
    parser.add_argument("--skip-fresh-forward", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_run(args.run_dir, config_path=args.config.resolve(), force_fresh=not args.skip_fresh_forward)
    except Exception as exc:
        result = {"schema_version": 1, "status": "FAIL", "engineering_decision": "BLOCKED",
                  "scientific_status": "NOT_DETERMINED_BY_CHECKER", "error": f"{type(exc).__name__}: {exc}"}
        write_json(args.run_dir.resolve() / "07_check" / "validation.json", result, self_hash=True)
    print(json.dumps({"status": result["status"], "engineering_decision": result.get("engineering_decision")}, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
