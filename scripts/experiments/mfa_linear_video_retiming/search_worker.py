from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .common import ProtocolError, canonical_json_sha256, file_sha256, read_json, verify_json, write_json
from .retime import build_map, global_seed, regularization, render_map, validate_map
from .scorer import (
    VSHIFT,
    calibration_fingerprint,
    crop_frames,
    forward_video,
    load_audio_embeddings,
    load_frozen_syncnet,
    read_video_frames,
    score_embeddings,
    support_indices,
)


def _float(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if not np.isfinite(value):
        raise ProtocolError(f"candidate metric {key} is not finite")
    return value


def feasibility_reason(metrics: Mapping[str, Any], baseline: Mapping[str, Any], search: Mapping[str, Any]) -> str | None:
    checks = (
        ("sync_c", _float(metrics, "sync_c") >= _float(baseline, "sync_c") - float(search["noninferiority_sync_c"])),
        ("sync_d", _float(metrics, "sync_d") <= _float(baseline, "sync_d") + float(search["noninferiority_sync_d"])),
        ("d0", _float(metrics, "d0") <= _float(baseline, "d0") + float(search["noninferiority_d0"])),
    )
    return next((f"NONINFERIORITY_{name.upper()}" for name, passed in checks if not passed), None)


def _candidate_order(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        _float(metrics, "d0"),
        abs(int(metrics["offset"])),
        -_float(metrics, "sync_c"),
        float(candidate["regularization"]),
        str(candidate["candidate_id"]),
    )


def _better_for_stage_b(left: Mapping[str, Any], right: Mapping[str, Any] | None, tolerance: float) -> bool:
    if right is None:
        return True
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    left_sync = abs(int(left_metrics["offset"])) <= 1
    right_sync = abs(int(right_metrics["offset"])) <= 1
    if left_sync != right_sync:
        return left_sync
    left_c = _float(left_metrics, "sync_c")
    right_c = _float(right_metrics, "sync_c")
    if abs(left_c - right_c) > tolerance:
        return left_c > right_c
    left_d0 = _float(left_metrics, "d0")
    right_d0 = _float(right_metrics, "d0")
    if left_d0 != right_d0:
        return left_d0 < right_d0
    if float(left["regularization"]) != float(right["regularization"]):
        return float(left["regularization"]) < float(right["regularization"])
    return str(left["candidate_id"]) < str(right["candidate_id"])


def final_acceptable(candidate: Mapping[str, Any], baseline: Mapping[str, Any], search: Mapping[str, Any]) -> bool:
    if not candidate.get("feasible"):
        return False
    metrics = candidate["metrics"]
    if _float(metrics, "sync_c") < _float(baseline, "sync_c") + float(search["minimum_sync_c_gain"]):
        return False
    if _float(metrics, "d0") > _float(baseline, "d0") - float(search["minimum_d0_improvement"]):
        return False
    if _float(metrics, "sync_d") > _float(baseline, "sync_d") + float(search["maximum_sync_d_worsening"]):
        return False
    if abs(int(metrics["offset"])) > 1:
        return False
    local = metrics.get("local_windows", [])
    base_local = baseline.get("local_windows", [])
    if not local or not base_local:
        return False
    local_median_d0 = float(np.median([_float(row, "d0") for row in local]))
    base_local_median_d0 = float(np.median([_float(row, "d0") for row in base_local]))
    if local_median_d0 > base_local_median_d0:
        return False
    quantile = float(search["local_offset_quantile"])
    local_q = float(np.quantile([abs(int(row["offset"])) for row in local], quantile, method="linear"))
    base_q = float(np.quantile([abs(int(row["offset"])) for row in base_local], quantile, method="linear"))
    return local_q <= base_q


def select_final_candidate(
    candidates: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    search: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    eligible = [candidate for candidate in candidates if final_acceptable(candidate, baseline, search)]
    if not eligible:
        return None
    max_c = max(_float(candidate["metrics"], "sync_c") for candidate in eligible)
    slack = float(search["local_sync_c_slack"])
    high_c = [candidate for candidate in eligible if _float(candidate["metrics"], "sync_c") >= max_c - slack]
    return min(high_c, key=lambda candidate: (float(candidate["regularization"]), str(candidate["candidate_id"])))


def state_compatible(state: Mapping[str, Any], request_fingerprint: str, scorer_fingerprint: str) -> bool:
    return (
        state.get("request_fingerprint") == request_fingerprint
        and state.get("scorer_fingerprint") == scorer_fingerprint
        and int(state.get("schema_version", 0)) == 1
    )


def _save_embeddings(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return file_sha256(path)


def _load_embedding(path: str | Path, expected_sha256: str) -> dict[str, np.ndarray]:
    target = Path(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise ProtocolError(f"embedding cache is missing or changed: {target}")
    with np.load(target, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _metrics_for_frames(
    frames: np.ndarray,
    audio_embeddings: np.ndarray,
    *,
    score_box: Sequence[int],
    audio_sample_count: int,
    valid_frame_count: int,
    scorer: Any,
    torch: Any,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cropped = crop_frames(frames, score_box)
    visual = forward_video(cropped, scorer, torch, batch_size=batch_size)
    metrics = score_embeddings(
        visual,
        audio_embeddings,
        frame_count=frames.shape[0],
        valid_frame_count=valid_frame_count,
        audio_sample_count=audio_sample_count,
        torch=torch,
        local_window_frames=25,
    )
    return metrics, visual, cropped


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.pop("distance_matrix", None)
    return result


def _shift_pcm(pcm: np.ndarray, shift_frames: int) -> np.ndarray:
    shift = int(shift_frames) * 640
    source = np.asarray(pcm)
    if source.dtype != np.int16 or source.ndim != 1:
        raise ProtocolError("delay diagnostic requires mono PCM16")
    output = np.zeros_like(source)
    if shift > 0:
        output[shift:] = source[:-shift]
    elif shift < 0:
        output[:shift] = source[-shift:]
    else:
        output[:] = source
    return output


def _legacy_curve_from_embeddings(
    visual: np.ndarray,
    audio: np.ndarray,
    rows: np.ndarray,
    *,
    torch: Any,
) -> tuple[np.ndarray, int]:
    """Rebuild the existing fixed-support SyncNet curve independently.

    ``static_image_bridge.score_worker`` owns the historical crop and
    embedding extraction helpers, but deliberately does not expose curve
    reduction helpers.  The legacy fixed-support reducer used by the existing
    experiment workers evaluates one ``pairwise_distance`` vector per lag and
    takes its float32 NumPy mean.  Keep that route separate from
    ``scorer.score_embeddings`` so calibration actually checks the new path.
    """

    selected = np.asarray(rows, dtype=np.int64)
    visual_array = np.asarray(visual, dtype=np.float32)
    audio_array = np.asarray(audio, dtype=np.float32)
    if selected.ndim != 1 or selected.size < 1:
        raise ProtocolError("legacy parity needs a non-empty fixed support")
    if visual_array.ndim != 2 or audio_array.ndim != 2:
        raise ProtocolError("legacy parity embeddings must be 2-D")
    if int(selected.min()) < 0 or int(selected.max()) >= visual_array.shape[0]:
        raise ProtocolError("legacy parity support exceeds visual embeddings")
    if int(selected.min()) - VSHIFT < 0 or int(selected.max()) + VSHIFT >= audio_array.shape[0]:
        raise ProtocolError("legacy parity support exceeds audio lag support")

    values: list[float] = []
    with torch.inference_mode():
        for lag in range(-VSHIFT, VSHIFT + 1):
            left = torch.from_numpy(np.ascontiguousarray(visual_array[selected]))
            right = torch.from_numpy(np.ascontiguousarray(audio_array[selected + lag]))
            distances = torch.nn.functional.pairwise_distance(left, right).cpu().numpy()
            # The legacy implementation calls np.mean on a float32 tensor's
            # NumPy view, which keeps float32 accumulation.
            values.append(float(np.mean(distances)))
    return np.asarray(values, dtype=np.float64), int(selected.size)


def _legacy_fixed_crop_parity(
    request: Mapping[str, Any],
    *,
    scorer: Any,
    torch: Any,
    video_frames: np.ndarray,
    audio_embeddings: np.ndarray,
    expected_visual: np.ndarray,
    audio_sample_count: int,
    valid_frame_count: int,
) -> dict[str, Any]:
    repository = Path(request["repo_root"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    legacy = importlib.import_module("scripts.experiments.static_image_bridge.score_worker")
    actual_path = Path(str(legacy.__file__)).resolve()
    expected_path = Path(request["legacy_score_worker"]).resolve()
    if actual_path != expected_path:
        raise ProtocolError(f"legacy fixed-crop scorer import shadowing: {actual_path}")
    score_box = {"box": [int(value) for value in request["score_box_xyxy"]]}
    legacy_cropped, _hashes, metadata = legacy.read_video(Path(request["video_m"]).resolve(), score_box)
    new_cropped = crop_frames(video_frames, request["score_box_xyxy"])
    if not np.array_equal(legacy_cropped, new_cropped):
        raise ProtocolError("legacy and new SyncNet crops differ on the same V_M input")
    old_visual_tensor, old_visual_count = legacy.visual_embedding(legacy_cropped, scorer, "cuda", torch)
    old_audio_tensor, old_audio_count, old_samples = legacy.audio_embedding(Path(request["audio_n"]).resolve(), scorer, "cuda", torch)
    old_visual = np.asarray(old_visual_tensor, dtype=np.float32)
    old_audio = np.asarray(old_audio_tensor, dtype=np.float32)
    if old_visual_count != old_visual.shape[0] or old_audio_count != old_audio.shape[0] or old_samples != audio_sample_count:
        raise ProtocolError("legacy fixed-crop embedding support differs from the frozen inputs")
    visual_error = float(np.max(np.abs(old_visual - expected_visual)))
    audio_error = float(np.max(np.abs(old_audio - audio_embeddings)))
    if visual_error > 1e-4 or audio_error > 1e-4:
        raise ProtocolError(f"legacy embedding parity failed: visual={visual_error}, audio={audio_error}")
    n = min(int(valid_frame_count), int(audio_sample_count) // 640) - 5
    frozen_w = support_indices(n)
    old_curve, old_support = _legacy_curve_from_embeddings(old_visual, old_audio, frozen_w, torch=torch)
    new_result = score_embeddings(
        old_visual, old_audio,
        frame_count=video_frames.shape[0], valid_frame_count=valid_frame_count,
        audio_sample_count=audio_sample_count, torch=torch,
    )
    old_minimum = float(np.min(old_curve))
    old_metrics = {
        "C": float(np.median(old_curve) - old_minimum),
        "D": old_minimum,
    }
    curve_error = float(np.max(np.abs(np.asarray(old_curve, dtype=np.float64) - np.asarray(new_result["curve"], dtype=np.float64))))
    if curve_error > 1e-4 or abs(float(old_metrics["C"]) - float(new_result["sync_c"])) > 1e-4 or abs(float(old_metrics["D"]) - float(new_result["sync_d"])) > 1e-4:
        raise ProtocolError("legacy fixed-crop Sync-C/D parity exceeded 1e-4")
    old_index = int(np.flatnonzero(np.asarray(old_curve) == np.min(old_curve))[0])
    if int(VSHIFT - old_index) != int(new_result["offset"]):
        raise ProtocolError("legacy fixed-crop offset sign/index differs")
    return {
        "status": "PASS",
        "legacy_worker": str(actual_path),
        "legacy_worker_sha256": file_sha256(actual_path),
        "video_metadata": metadata,
        "crop_pixels_identical": True,
        "visual_embedding_max_abs": visual_error,
        "audio_embedding_max_abs": audio_error,
        "curve_max_abs": curve_error,
        "legacy_support_count": int(old_support),
        "frozen_W_sha256": new_result["W_sha256"],
        "legacy_sync_c": float(old_metrics["C"]),
        "legacy_sync_d": float(old_metrics["D"]),
        "legacy_offset": int(VSHIFT - old_index),
    }


def calibrate_record(request: Mapping[str, Any]) -> dict[str, Any]:
    scorer, torch, model_meta = load_frozen_syncnet(
        request["syncnet_root"], request["syncnet_model"], device="cuda"
    )
    if model_meta["checkpoint_sha256"] != request["expected_syncnet_sha256"]:
        raise ProtocolError("loaded SyncNet model SHA differs from the frozen request")
    m_frames, m_video_meta = read_video_frames(request["video_m"])
    n_frames, n_video_meta = read_video_frames(request["video_n"])
    if m_video_meta["frame_count"] != n_video_meta["frame_count"]:
        raise ProtocolError("N/M baseline videos do not share frame count")
    audio_embeddings, audio_meta = load_audio_embeddings(
        request["audio_n"], scorer, torch, batch_size=int(request["batch_size"])
    )
    audio_sample_count = int(audio_meta["sample_count"])
    valid_count = int(request["valid_frame_count"])
    m_metrics, m_visual, m_cropped = _metrics_for_frames(
        m_frames, audio_embeddings, score_box=request["score_box_xyxy"],
        audio_sample_count=audio_sample_count, valid_frame_count=valid_count,
        scorer=scorer, torch=torch, batch_size=int(request["batch_size"]),
    )
    m_repeat_visual = forward_video(m_cropped, scorer, torch, batch_size=int(request["batch_size"]))
    repeat_max = float(np.max(np.abs(m_visual - m_repeat_visual)))
    repeat_metrics = score_embeddings(
        m_repeat_visual, audio_embeddings,
        frame_count=m_frames.shape[0], valid_frame_count=valid_count,
        audio_sample_count=audio_sample_count, torch=torch,
    )
    if repeat_max > 1e-4 or abs(float(m_metrics["sync_c"]) - float(repeat_metrics["sync_c"])) > 1e-4 or abs(float(m_metrics["sync_d"]) - float(repeat_metrics["sync_d"])) > 1e-4 or int(m_metrics["offset"]) != int(repeat_metrics["offset"]):
        raise ProtocolError("same-input SyncNet forward is not stable within 1e-4")
    n_metrics, n_visual, _n_cropped = _metrics_for_frames(
        n_frames, audio_embeddings, score_box=request["score_box_xyxy"],
        audio_sample_count=audio_sample_count, valid_frame_count=valid_count,
        scorer=scorer, torch=torch, batch_size=int(request["batch_size"]),
    )
    legacy_parity = _legacy_fixed_crop_parity(
        request, scorer=scorer, torch=torch, video_frames=m_frames,
        audio_embeddings=audio_embeddings, expected_visual=m_visual,
        audio_sample_count=audio_sample_count, valid_frame_count=valid_count,
    )

    from scipy.io import wavfile

    rate, pcm = wavfile.read(str(Path(request["audio_n"]).resolve()))
    if rate != 16000 or np.asarray(pcm).dtype != np.int16 or np.asarray(pcm).ndim != 1:
        raise ProtocolError("delay calibration source is not 16 kHz mono PCM16")
    temp_dir = Path(request["work_dir"]).resolve() / "delay_calibration"
    temp_dir.mkdir(parents=True, exist_ok=True)
    delay_rows: list[dict[str, Any]] = []
    baseline_lag = int(n_metrics["best_lag"])
    delay_failed = abs(baseline_lag) == VSHIFT
    for shift in (-3, 3):
        temporary_audio = temp_dir / f"natural_delay_{shift:+d}_frames.wav"
        shifted = _shift_pcm(np.asarray(pcm), shift)
        wavfile.write(str(temporary_audio), 16000, shifted)
        shifted_features, shifted_meta = load_audio_embeddings(
            temporary_audio, scorer, torch, batch_size=int(request["batch_size"])
        )
        shifted_metrics = score_embeddings(
            n_visual, shifted_features,
            frame_count=n_frames.shape[0], valid_frame_count=valid_count,
            audio_sample_count=int(shifted_meta["sample_count"]), torch=torch,
        )
        observed_delta = int(shifted_metrics["best_lag"]) - baseline_lag
        passed = abs(observed_delta - shift) <= 1 and abs(int(shifted_metrics["best_lag"])) < VSHIFT
        delay_failed = delay_failed or not passed
        delay_rows.append({
            "input_delay_frames": shift,
            "expected_best_lag_delta": shift,
            "baseline_best_lag": baseline_lag,
            "observed_best_lag": int(shifted_metrics["best_lag"]),
            "observed_best_lag_delta": observed_delta,
            "offset": int(shifted_metrics["offset"]),
            "passed": passed,
            "temporary_audio": str(temporary_audio),
            "temporary_audio_sha256": file_sha256(temporary_audio),
        })
    n_audio_hash_after = file_sha256(request["audio_n"])
    if n_audio_hash_after != audio_meta["audio_sha256"]:
        raise ProtocolError("delay calibration changed the frozen N source WAV")
    n = min(valid_count, audio_sample_count // 640) - 5
    frozen_w = support_indices(n)
    scorer_fp = calibration_fingerprint(
        model_meta,
        score_box=request["score_box_xyxy"],
        support=frozen_w,
        legacy_score_worker=request["legacy_score_worker"],
    )
    state_path = Path(request["state_path"]).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    audio_embedding_path = Path(request["embedding_dir"]).resolve() / "natural_audio.npz"
    audio_embedding_sha = _save_embeddings(audio_embedding_path, audio=audio_embeddings)
    m_embedding_path = Path(request["embedding_dir"]).resolve() / "baseline_M_visual.npz"
    m_embedding_sha = _save_embeddings(m_embedding_path, visual=m_visual)
    n_embedding_path = Path(request["embedding_dir"]).resolve() / "baseline_N_visual.npz"
    n_embedding_sha = _save_embeddings(n_embedding_path, visual=n_visual)
    compact_baseline = _compact_metrics(m_metrics)
    base_candidate = {
        "candidate_id": str(m_metrics["map_sha256"][:16]) if "map_sha256" in m_metrics else "identity",
        "label": "identity",
        "stage": "A",
        "map": request["identity_map"],
        "map_sha256": request["identity_map"]["map_sha256"],
        "metrics": compact_baseline,
        "feasible": True,
        "reject_reason": None,
        "regularization": 0.0,
        "embedding_path": str(m_embedding_path),
        "embedding_sha256": m_embedding_sha,
    }
    state = {
        "schema_version": 1,
        "request_fingerprint": request["request_fingerprint"],
        "scorer_fingerprint": scorer_fp,
        "status": "CALIBRATED" if not delay_failed and legacy_parity["status"] == "PASS" else "CALIBRATION_FAILED",
        "sample_id": str(request["sample_id"]),
        "paired_key": str(request["paired_key"]),
        "portrait_id": str(request["portrait_id"]),
        "score_box_xyxy": [int(value) for value in request["score_box_xyxy"]],
        "frame_count": int(m_frames.shape[0]),
        "valid_frame_count": valid_count,
        "original_M_video_sha256": file_sha256(request["video_m"]),
        "original_M_frame_sha256": hashlib.sha256(np.ascontiguousarray(m_frames).tobytes()).hexdigest(),
        "natural_audio": audio_meta,
        "natural_audio_embedding_path": str(audio_embedding_path),
        "natural_audio_embedding_sha256": audio_embedding_sha,
        "baseline_N_visual_embedding_path": str(n_embedding_path),
        "baseline_N_visual_embedding_sha256": n_embedding_sha,
        "baseline_N_metrics": _compact_metrics(n_metrics),
        "baseline_candidate": base_candidate,
        "candidates": {"identity": base_candidate},
        "frozen_support": [int(value) for value in frozen_w],
        "W_sha256": canonical_json_sha256([int(value) for value in frozen_w]),
        "model_metadata": model_meta,
        "legacy_parity": legacy_parity,
        "repeat_forward": {"status": "PASS", "visual_embedding_max_abs": repeat_max},
        "delay_calibration": {"status": "FAIL" if delay_failed else "PASS", "rows": delay_rows},
        "calibration_forward_passes": 5,
        "search_candidate_calls": 1,
        "stage_a_candidate_calls": 1,
        "stage_b_candidate_calls": 0,
        "elapsed_search_seconds": 0.0,
        "cache_invalidations": [],
    }
    write_json(state_path, state, self_hash=True)
    result = {
        "schema_version": 1,
        "mode": "calibrate",
        "status": state["status"],
        "state_path": str(state_path),
        "state_sha256": file_sha256(state_path),
        "request_fingerprint": request["request_fingerprint"],
        "scorer_fingerprint": scorer_fp,
        "syncnet_model": model_meta,
        "baseline_M_N": compact_baseline,
        "baseline_N_N": _compact_metrics(n_metrics),
        "legacy_parity": legacy_parity,
        "repeat_forward": state["repeat_forward"],
        "delay_calibration": state["delay_calibration"],
    }
    write_json(request["result_path"], result, self_hash=True)
    return result


def _valid_state_or_none(
    state_path: Path,
    request: Mapping[str, Any],
    scorer_fingerprint: str,
) -> dict[str, Any] | None:
    if not state_path.is_file():
        return None
    try:
        state = read_json(state_path)
    except (OSError, ValueError):
        return None
    body = dict(state)
    recorded = body.pop("artifact_sha256", None)
    if recorded != canonical_json_sha256(body) or not state_compatible(state, str(request["request_fingerprint"]), scorer_fingerprint):
        raise ProtocolError("SEARCH_STATE_FINGERPRINT_MISMATCH; refusing to read cached scores")
    invalid: list[str] = []
    for candidate_id, candidate in state.get("candidates", {}).items():
        path = candidate.get("embedding_path")
        if not path or not Path(path).is_file() or file_sha256(path) != candidate.get("embedding_sha256"):
            invalid.append(str(candidate_id))
    for candidate_id in invalid:
        state["candidates"].pop(candidate_id, None)
        state.setdefault("cache_invalidations", []).append({"candidate_id": candidate_id, "reason": "embedding_missing_or_hash_mismatch"})
    if invalid:
        write_json(state_path, state, self_hash=True)
    return state


def _record_candidate(
    *,
    state: dict[str, Any],
    request: Mapping[str, Any],
    mapping: Mapping[str, Any],
    label: str,
    stage: str,
    frames: np.ndarray,
    audio_embeddings: np.ndarray,
    audio_sample_count: int,
    scorer: Any,
    torch: Any,
    output_path: Path,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    map_sha = str(mapping["map_sha256"])
    candidate_id = map_sha
    cached = state["candidates"].get(candidate_id)
    if cached is not None and cached.get("map_sha256") == map_sha:
        if cached.get("embedding_sha256") and Path(cached["embedding_path"]).is_file() and file_sha256(cached["embedding_path"]) == cached["embedding_sha256"]:
            return cached
    started = time.monotonic()
    audit = validate_map(mapping, frame_count=frames.shape[0], valid_frame_count=int(request["valid_frame_count"]))
    rendered, render_audit = render_map(frames, mapping)
    metrics, visual, _cropped = _metrics_for_frames(
        rendered, audio_embeddings, score_box=request["score_box_xyxy"],
        audio_sample_count=audio_sample_count, valid_frame_count=int(request["valid_frame_count"]),
        scorer=scorer, torch=torch, batch_size=int(request["batch_size"]),
    )
    embedding_path = output_path / f"candidate_{candidate_id}.npz"
    embedding_sha = _save_embeddings(embedding_path, visual=visual)
    compact = _compact_metrics(metrics)
    reject_reason = feasibility_reason(compact, baseline, request["search"])
    row = {
        "candidate_id": candidate_id,
        "label": label,
        "stage": stage,
        "map": dict(mapping),
        "map_sha256": map_sha,
        "metrics": compact,
        "feasible": reject_reason is None,
        "reject_reason": reject_reason,
        "regularization": float(audit["regularization"]),
        "max_abs_displacement": float(audit["max_abs_displacement"]),
        "max_speed_deviation": float(audit["max_speed_deviation"]),
        "max_slope_change": float(audit["max_slope_change"]),
        "blended_fraction": float(audit["blended_fraction"]),
        "rendered_frames_sha256": hashlib.sha256(np.ascontiguousarray(rendered).tobytes()).hexdigest(),
        "embedding_path": str(embedding_path.resolve()),
        "embedding_sha256": embedding_sha,
        "elapsed_seconds": 0.0,
    }
    row["elapsed_seconds"] = time.monotonic() - started
    state["search_candidate_calls"] = int(state.get("search_candidate_calls", 0)) + 1
    if stage == "A":
        state["stage_a_candidate_calls"] = int(state.get("stage_a_candidate_calls", 0)) + 1
    else:
        state["stage_b_candidate_calls"] = int(state.get("stage_b_candidate_calls", 0)) + 1
    state["elapsed_search_seconds"] = float(state.get("elapsed_search_seconds", 0.0)) + row["elapsed_seconds"]
    state["candidates"][candidate_id] = row
    write_json(Path(request["state_path"]), state, self_hash=True)
    return row


def _candidate_invalid_reason(mapping: Mapping[str, Any], frame_count: int, valid_count: int) -> str | None:
    try:
        validate_map(mapping, frame_count=frame_count, valid_frame_count=valid_count)
    except ProtocolError as exc:
        return str(exc)
    return None


def _coordinate_maps(incumbent: Mapping[str, Any], step: float) -> list[tuple[str, dict[str, Any]]]:
    mapping = incumbent["map"]
    positions = [int(value) for value in mapping["knot_positions"]]
    values = np.asarray(mapping["knot_deltas"], dtype=np.float64)
    output: list[tuple[str, dict[str, Any]]] = []

    def append(label: str, candidate_values: np.ndarray) -> None:
        candidate_values = np.rint(candidate_values / 0.25) * 0.25
        if np.any(np.abs(candidate_values) > 3.0 + 1e-12):
            output.append((label, {"candidate_build_error": "knot_displacement_exceeds_3_frames"}))
            return
        try:
            candidate = build_map(
                int(mapping["frame_count"]), int(mapping["valid_frame_count"]),
                candidate_values, positions=positions,
            )
        except ProtocolError as exc:
            output.append((label, {"candidate_build_error": str(exc)}))
            return
        output.append((label, candidate))

    for index in range(values.size):
        for direction, sign in (("minus", -1.0), ("plus", 1.0)):
            candidate_values = values.copy()
            candidate_values[index] += sign * float(step)
            append(f"single_{index:02d}_{direction}_{step:g}", candidate_values)
    for index in range(values.size - 1):
        for direction, sign in (("minus", -1.0), ("plus", 1.0)):
            candidate_values = values.copy()
            candidate_values[index:index + 2] += sign * float(step)
            append(f"adjacent_{index:02d}_{direction}_{step:g}", candidate_values)
    return output


def _choose_best(candidates: Sequence[Mapping[str, Any]], *, stage_a: bool, tolerance: float) -> Mapping[str, Any]:
    feasible = [row for row in candidates if row.get("feasible")]
    if not feasible:
        raise ProtocolError("search has no feasible candidate, although identity should be feasible")
    if stage_a:
        return min(feasible, key=lambda row: _candidate_order(row, {}))
    best: Mapping[str, Any] | None = None
    for row in feasible:
        if _better_for_stage_b(row, best, tolerance):
            best = row
    assert best is not None
    return best


def search_record(request: Mapping[str, Any]) -> dict[str, Any]:
    state_path = Path(request["state_path"]).resolve()
    state = verify_json(state_path, self_hash=True)
    if state.get("status") not in ("CALIBRATED", "SEARCHING", "BUDGET_LIMITED", "SEARCH_COMPLETE"):
        raise ProtocolError("search is blocked because scorer calibration did not pass")
    if not state_compatible(state, str(request["request_fingerprint"]), str(state.get("scorer_fingerprint", ""))):
        raise ProtocolError("SEARCH_STATE_FINGERPRINT_MISMATCH: frozen input/code binding changed")
    state = _valid_state_or_none(state_path, request, str(state["scorer_fingerprint"])) or state
    if "identity" not in state.get("candidates", {}):
        raise ProtocolError("identity embedding cache is invalid; rerun calibration before search")
    if state.get("status") == "SEARCH_COMPLETE":
        cached_result_path = Path(request["result_path"]).resolve()
        if cached_result_path.is_file():
            cached_result = verify_json(cached_result_path, self_hash=True)
            if cached_result.get("search_result", {}).get("candidate_count") == len(state.get("candidates", {})):
                return cached_result
    scorer, torch, model_meta = load_frozen_syncnet(
        request["syncnet_root"], request["syncnet_model"], device="cuda"
    )
    expected_fp = calibration_fingerprint(
        model_meta,
        score_box=request["score_box_xyxy"],
        support=state["frozen_support"],
        legacy_score_worker=request["legacy_score_worker"],
    )
    if expected_fp != state.get("scorer_fingerprint"):
        raise ProtocolError("SEARCH_STATE_FINGERPRINT_MISMATCH: SyncNet/frontend fingerprint changed")
    if not state_compatible(state, str(request["request_fingerprint"]), expected_fp):
        raise ProtocolError("SEARCH_STATE_FINGERPRINT_MISMATCH: frozen input/code binding changed")
    frames, _video_metadata = read_video_frames(request["video_m"])
    frame_hash = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
    if frame_hash != state.get("original_M_frame_sha256") or file_sha256(request["video_m"]) != state.get("original_M_video_sha256"):
        raise ProtocolError("search source V_M changed after calibration")
    audio_cache = _load_embedding(state["natural_audio_embedding_path"], state["natural_audio_embedding_sha256"])
    audio_embeddings = audio_cache["audio"].astype(np.float32, copy=False)
    audio_sample_count = int(state["natural_audio"]["sample_count"])
    base_candidate = state.get("candidates", {}).get("identity")
    if base_candidate is None:
        raise ProtocolError("identity baseline cache is missing; calibration must be repeated")
    baseline = base_candidate["metrics"]
    if int(state["search_candidate_calls"]) > int(request["search"]["max_candidates"]):
        raise ProtocolError("candidate score budget was already exceeded")
    state["status"] = "SEARCHING"
    write_json(state_path, state, self_hash=True)
    output_dir = Path(request["embedding_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = list(state["candidates"].values())
    unique_maps = {str(row["map_sha256"]) for row in candidate_rows}
    stage_a_rows = [row for row in candidate_rows if row.get("stage") == "A" or row.get("label") == "identity"]
    stage_a_seconds = time.monotonic()
    stage_a_elapsed_before = float(state.get("elapsed_search_seconds", 0.0))
    identity = base_candidate
    seeds = []
    for amount in (-3, -2, -1, 1, 2, 3):
        mapping = global_seed(frames.shape[0], int(request["valid_frame_count"]), amount)
        label = f"global_seed_{amount:+d}"
        reason = _candidate_invalid_reason(mapping, frames.shape[0], int(request["valid_frame_count"]))
        map_sha = str(mapping["map_sha256"])
        if reason:
            state.setdefault("candidate_rejections", []).append({"label": label, "map_sha256": map_sha, "stage": "A", "reason": reason, "scored": False})
            continue
        if map_sha in unique_maps:
            existing = state["candidates"].get(map_sha)
            if existing:
                seeds.append(existing)
            continue
        if len(stage_a_rows) >= int(request["search"]["stage_a_max_candidates"]):
            break
        if int(state["search_candidate_calls"]) >= int(request["search"]["max_candidates"]):
            state["status"] = "BUDGET_LIMITED"
            break
        if stage_a_elapsed_before + (time.monotonic() - stage_a_seconds) >= float(request["search"]["max_seconds_per_sample"]):
            state["status"] = "BUDGET_LIMITED"
            break
        row = _record_candidate(
            state=state, request=request, mapping=mapping, label=label, stage="A",
            frames=frames, audio_embeddings=audio_embeddings, audio_sample_count=audio_sample_count,
            scorer=scorer, torch=torch, output_path=output_dir, baseline=baseline,
        )
        unique_maps.add(map_sha)
        seeds.append(row)
        stage_a_rows.append(row)

    stage_a_best = _choose_best(stage_a_rows, stage_a=True, tolerance=float(request["search"]["score_tie_tolerance"]))
    aligned_seed_candidates = [row for row in stage_a_rows if row.get("feasible") and abs(int(row["metrics"]["offset"])) <= 1]
    if aligned_seed_candidates:
        incumbent = _choose_best(aligned_seed_candidates, stage_a=False, tolerance=float(request["search"]["score_tie_tolerance"]))
    else:
        incumbent = stage_a_best

    stop_reason: str | None = None
    stage_b_seconds = time.monotonic()
    stage_b_elapsed_before = float(state.get("elapsed_search_seconds", 0.0))
    if state.get("status") != "BUDGET_LIMITED":
        for step in request["search"]["stage_b_steps"]:
            for round_index in range(int(request["search"]["rounds_per_step"])):
                candidates_this_round: list[Mapping[str, Any]] = [incumbent]
                moves = _coordinate_maps(incumbent, float(step))
                round_exhausted = False
                for label, mapping in moves:
                    if mapping.get("candidate_build_error"):
                        state.setdefault("candidate_rejections", []).append({
                            "label": label,
                            "stage": "B",
                            "reason": str(mapping["candidate_build_error"]),
                            "scored": False,
                        })
                        continue
                    map_sha = str(mapping["map_sha256"])
                    if map_sha in unique_maps:
                        cached = state["candidates"].get(map_sha)
                        if cached is not None:
                            candidates_this_round.append(cached)
                        continue
                    invalid = _candidate_invalid_reason(mapping, frames.shape[0], int(request["valid_frame_count"]))
                    if invalid:
                        state.setdefault("candidate_rejections", []).append({"label": label, "map_sha256": map_sha, "stage": "B", "reason": invalid, "scored": False})
                        continue
                    stage_b_calls = int(state.get("stage_b_candidate_calls", 0))
                    if stage_b_calls >= int(request["search"]["stage_b_max_candidates"]):
                        stop_reason = "STAGE_B_CANDIDATE_BUDGET"
                        round_exhausted = True
                        break
                    if int(state["search_candidate_calls"]) >= int(request["search"]["max_candidates"]):
                        stop_reason = "TOTAL_CANDIDATE_BUDGET"
                        round_exhausted = True
                        break
                    elapsed = stage_b_elapsed_before + (time.monotonic() - stage_b_seconds)
                    if elapsed >= float(request["search"]["max_seconds_per_sample"]):
                        stop_reason = "PER_SAMPLE_TIME_BUDGET"
                        round_exhausted = True
                        break
                    row = _record_candidate(
                        state=state, request=request, mapping=mapping, label=label, stage="B",
                        frames=frames, audio_embeddings=audio_embeddings, audio_sample_count=audio_sample_count,
                        scorer=scorer, torch=torch, output_path=output_dir, baseline=baseline,
                    )
                    unique_maps.add(map_sha)
                    candidates_this_round.append(row)
                incumbent = _choose_best(
                    candidates_this_round,
                    stage_a=False,
                    tolerance=float(request["search"]["score_tie_tolerance"]),
                )
                write_json(state_path, state, self_hash=True)
                if round_exhausted:
                    break
            if stop_reason:
                break
    all_rows = list(state["candidates"].values())
    accepted = select_final_candidate(all_rows, baseline, request["search"])
    if accepted is None:
        selected = identity
        result_status = "BUDGET_LIMITED" if stop_reason else "NO_ACCEPTABLE_WARP"
    else:
        selected = accepted
        result_status = "BUDGET_LIMITED" if stop_reason else "ACCEPTED_WARP"
    global_rows = [row for row in all_rows if str(row.get("label", "")).startswith("global_seed_")]
    selected_global = select_final_candidate(global_rows, baseline, request["search"]) if global_rows else None
    global_control = selected_global or identity
    best_attempt = _choose_best(
        [row for row in all_rows if row.get("feasible")] or [identity],
        stage_a=False,
        tolerance=float(request["search"]["score_tie_tolerance"]),
    )
    if state.get("status") == "BUDGET_LIMITED" and stop_reason is None:
        stop_reason = "STAGE_A_TIME_OR_CANDIDATE_BUDGET"
    state.update({
        "status": "SEARCH_COMPLETE" if result_status != "BUDGET_LIMITED" else "BUDGET_LIMITED",
        "search_result": {
            "status": result_status,
            "stop_reason": stop_reason,
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_map_sha256": selected["map_sha256"],
            "global_candidate_id": global_control["candidate_id"],
            "best_attempt_candidate_id": best_attempt["candidate_id"],
            "actual_score_calls": int(state["search_candidate_calls"]),
            "stage_a_scored": len([row for row in all_rows if row.get("stage") == "A"]),
            "stage_b_scored": len([row for row in all_rows if row.get("stage") == "B"]),
            "elapsed_search_seconds": stage_b_elapsed_before + (time.monotonic() - stage_b_seconds),
            "candidate_count": len(all_rows),
        },
    })
    write_json(state_path, state, self_hash=True)
    result = {
        "schema_version": 1,
        "mode": "search",
        "status": result_status,
        "stop_reason": stop_reason,
        "sample_id": str(request["sample_id"]),
        "paired_key": str(request["paired_key"]),
        "portrait_id": str(request["portrait_id"]),
        "baseline": baseline,
        "selected": selected,
        "global_control": global_control,
        "best_attempt": best_attempt,
        "state_path": str(state_path),
        "state_sha256": file_sha256(state_path),
        "search_result": state["search_result"],
    }
    write_json(request["result_path"], result, self_hash=True)
    return result


def score_exported_video(request: Mapping[str, Any]) -> dict[str, Any]:
    scorer, torch, model_meta = load_frozen_syncnet(
        request["syncnet_root"], request["syncnet_model"], device="cuda"
    )
    if model_meta["checkpoint_sha256"] != request["expected_syncnet_sha256"]:
        raise ProtocolError("fresh SyncNet pass loaded an unexpected checkpoint")
    frames, video_meta = read_video_frames(request["video"])
    audio_embeddings, audio_meta = load_audio_embeddings(
        request["audio"], scorer, torch, batch_size=int(request["batch_size"])
    )
    metrics, visual, cropped = _metrics_for_frames(
        frames, audio_embeddings, score_box=request["score_box_xyxy"],
        audio_sample_count=int(audio_meta["sample_count"]),
        valid_frame_count=int(request["valid_frame_count"]),
        scorer=scorer, torch=torch, batch_size=int(request["batch_size"]),
    )
    output_dir = Path(request["embedding_dir"]).resolve()
    embedding_path = output_dir / f"fresh_{request['video_arm']}_{request['audio_role']}.npz"
    embedding_sha = _save_embeddings(embedding_path, visual=visual, audio=audio_embeddings)
    result = {
        "schema_version": 1,
        "mode": "score_video",
        "status": "COMPLETE",
        "sample_id": str(request["sample_id"]),
        "paired_key": str(request["paired_key"]),
        "portrait_id": str(request["portrait_id"]),
        "video_arm": str(request["video_arm"]),
        "audio_role": str(request["audio_role"]),
        "video": str(Path(request["video"]).resolve()),
        "video_sha256": file_sha256(request["video"]),
        "audio": str(Path(request["audio"]).resolve()),
        "audio_sha256": audio_meta["audio_sha256"],
        "video_metadata": video_meta,
        "audio_metadata": audio_meta,
        "score_box_xyxy": [int(value) for value in request["score_box_xyxy"]],
        "metrics": _compact_metrics(metrics),
        "embedding_path": str(embedding_path),
        "embedding_sha256": embedding_sha,
        "scorer_metadata": model_meta,
        "cropped_video_sha256": hashlib.sha256(np.ascontiguousarray(cropped).tobytes()).hexdigest(),
    }
    write_json(request["result_path"], result, self_hash=True)
    return result


def run_request(request_path: str | Path) -> dict[str, Any]:
    request = read_json(request_path)
    mode = str(request.get("mode", ""))
    if mode == "calibrate":
        return calibrate_record(request)
    if mode == "search":
        return search_record(request)
    if mode == "search_v2":
        from .search_v2 import search_record_v2
        return search_record_v2(request)
    if mode == "score_video":
        return score_exported_video(request)
    raise ProtocolError(f"unsupported SyncNet worker mode: {mode}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_request(args.request_json.resolve())
    print(json.dumps({"status": result["status"], "mode": result["mode"], "sample_id": result.get("sample_id")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] in ("PASS", "CALIBRATED", "COMPLETE", "ACCEPTED_WARP", "NO_ACCEPTABLE_WARP", "BUDGET_LIMITED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
