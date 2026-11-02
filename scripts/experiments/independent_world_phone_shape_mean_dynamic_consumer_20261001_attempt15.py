#!/usr/bin/env python3
"""Independent CPU reconstruction for the frozen 11,776-endpoint WORLD shape design.

The reader imports no WORLD/Wav2Lip/SyncNet producer modules.  It verifies the
sealed five-cell run before opening score arrays, recomputes full V x A raw and
unit distances, applies the frozen full-A time permutations, and constructs the
single speaker-weighted max-|t| family.  It does not fit a phone-specific
causal model: phone-local response is a receptive-field-weighted description
of globally randomized shape interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import wavfile


DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
METRIC_ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
PHONE_INVENTORY_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"
QA_REPAIR_AUTHORIZATION_RAW_SHA256 = "0923847e89ab0c7badf9d82c821ee79bc7da9155bf6e40797d72bcd034ac2f81"
QA_REPAIR_AUTHORIZATION_PAYLOAD_SHA256 = "381e6dfd67e730e42810e71f08c4c4b9b1ebbeddfbd4a03f69ebe95ed1d5800b"
QA_REPAIR_SCHEMA = "ROOT_APPROVED_EVAL_QA_EXECUTION_REPAIR_V1"
SHAPE_MANIFEST_RAW_SHA256 = "55e34a1ffd13f61d27e7a9b9b630a2514cce98c40f4812340a658b581947bdd0"
SHAPE_MANIFEST_PAYLOAD_SHA256 = "2161858caab9283eafe32b7385be892094798d1acc2fbb77fd4313a67eee45ad"
SHAPE_QA_RAW_SHA256 = "d39fbc2a801c6ae4e615b1bf1644eb214bbc576fa08a77aadafb32567bb33238"
SHAPE_QA_PAYLOAD_SHA256 = "9a1a17b9f9bf585a6988ccf9c032dd3d996ebb5a214d69f83c75b8d9bc80ac8d"
QA_REPAIR_CHANGES = (
    "phone inventory payload binding: compare the recomputed inventory_sha256 with frozen input_bindings.phone_inventory.payload_sha256 and payload_field inventory_sha256",
    "QA source self-binding: retain the plan-bound QA03 hash; authorize QA04 only through this root-approved repair manifest",
)
EXPECTED_CONTEXTS = 160
EXPECTED_PAIRS = 80
EXPECTED_SPEAKERS = 40
EXPECTED_ROWS = 1600
EXPECTED_CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
EXPECTED_AUDIO = ("A_RAW", "A_RT")
EXPECTED_DIRECTIONS = ("tts_to_natural", "natural_to_tts")
SUPPORTS = ("full", "valid", "guard20", "guard15")
GEOMETRIES = ("raw", "unit")
METRICS = ("C_fixed_anchor", "B_median", "D_min", "D0", "search_gain", "optimal_offset_frames")
CONTRASTS = (
    "mean", "trajectory", "joint", "grid_cycle", "mean_minus_trajectory",
    "mean_beyond_grid", "trajectory_beyond_grid", "joint_beyond_grid", "factorial_interaction",
)
LOCAL_TREATMENTS = ("q10", "q01", "q11", "SELF_GRID")
LOCAL_ESTIMANDS = ("observed", "time_excess")
PHONE_LABELS = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)
PHONE_OUTCOMES = ("whole", "begin", "middle", "end")
LOCAL_STAT_NAMES = (
    "observed_fixed_denominator_mean", "time_null_mean", "time_excess",
    "exposure_fraction", "exposed_window_count", "exact_token_count",
    "conditional_mean_exposed_windows_only", "stft50_observed_fixed_denominator_mean",
    "stft50_time_null_mean", "stft50_time_excess", "stft50_exposure_fraction",
)
OVERALL_STAT_NAMES = ("observed_fixed_denominator_mean", "time_null_mean", "time_excess")
ANCHOR = {"k": 3, "column": 18, "official_offset_frames": -3}
VSHIFT = 15
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
RAW_NULL_SEED = 20260930
RT_NULL_SEED = 20270930
NULL_DRAWS = 100
CROSS_ATOL = 1.0e-4
CROSS_RTOL = 1.0e-4
METRIC_ATOL = 2.0e-6


class ConsumerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_sha(path: Path, expected: str, label: str) -> str:
    path = path.resolve(strict=True)
    observed = _sha(path)
    if observed != expected:
        raise ConsumerError(f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def _path_under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ConsumerError(f"{label} escapes the sealed run root: {relative}") from exc
    return path


def _read_bridge_wave(ref: dict[str, Any], label: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read and independently bind a float32 endpoint waveform for a bridge."""
    path = Path(str(ref.get("path", ""))).resolve(strict=True)
    file_sha = str(ref.get("file_sha256", ""))
    if not file_sha or _check_sha(path, file_sha, label) != file_sha:
        raise ConsumerError(f"{label} does not bind its waveform file")
    sample_rate, values = wavfile.read(str(path))
    if values.dtype != np.float32 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ConsumerError(f"{label} must be finite mono float32 WAV: {values.dtype}/{values.shape}")
    wave = np.ascontiguousarray(values)
    pcm_sha = _array_sha(wave)
    expected_pcm_sha = ref.get("float32_pcm_sha256")
    if expected_pcm_sha is not None and pcm_sha != expected_pcm_sha:
        raise ConsumerError(f"{label} decoded float32 PCM hash differs from its reference")
    if int(ref.get("sample_rate_hz", -1)) != int(sample_rate) or int(ref.get("samples", -1)) != len(wave):
        raise ConsumerError(f"{label} sample clock differs from its reference")
    return wave, {"path": str(path), "file_sha256": file_sha, "float32_pcm_sha256": pcm_sha, "sample_rate_hz": int(sample_rate), "samples": int(len(wave))}


def _video_pixel_bridge(old_path: Path, fresh_path: Path) -> dict[str, Any]:
    """Stream both videos and reconstruct the runner's pixel bridge summary."""
    old_cap = cv2.VideoCapture(str(old_path))
    fresh_cap = cv2.VideoCapture(str(fresh_path))
    if not old_cap.isOpened() or not fresh_cap.isOpened():
        old_cap.release()
        fresh_cap.release()
        raise ConsumerError(f"cannot decode q00/q11 bridge video: {old_path} / {fresh_path}")
    total_abs = 0.0
    total_pixels = 0
    max_abs = 0
    exact_frames = 0
    frame_count = 0
    try:
        while True:
            old_ok, old_frame = old_cap.read()
            fresh_ok, fresh_frame = fresh_cap.read()
            if old_ok != fresh_ok:
                raise ConsumerError("q00/q11 bridge videos have different decoded frame counts")
            if not old_ok:
                break
            if old_frame.shape != fresh_frame.shape:
                raise ConsumerError("q00/q11 bridge decoded frame shapes differ")
            delta = np.abs(old_frame.astype(np.int16) - fresh_frame.astype(np.int16))
            total_abs += float(delta.sum(dtype=np.float64))
            total_pixels += int(delta.size)
            max_abs = max(max_abs, int(delta.max(initial=0)))
            exact_frames += int(np.array_equal(old_frame, fresh_frame))
            frame_count += 1
    finally:
        old_cap.release()
        fresh_cap.release()
    if frame_count < 1:
        raise ConsumerError("q00/q11 bridge videos decoded no frames")
    return {
        "frame_count": frame_count,
        "pixel_bit_exact_frames": exact_frames,
        "mean_absolute_pixel_difference": total_abs / float(total_pixels),
        "maximum_absolute_pixel_difference": max_abs,
    }


def _close_number(observed: Any, recorded: Any, label: str, *, atol: float = 1e-12) -> None:
    left, right = float(observed), float(recorded)
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > atol:
        raise ConsumerError(f"{label} does not match independent reconstruction: {left} != {right}")


def _verify_endpoint_bridges(
    run_root: Path,
    context: dict[str, Any],
    context_meta: dict[str, Any],
    context_arrays: dict[str, np.ndarray],
    row_by_id: dict[str, dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    run_inventory: dict[str, str],
) -> dict[str, Any]:
    """Rebuild q00/RT and q11/SP waveform, latent, V and pixel bridges."""
    if context_meta.get("original_endpoints") != context.get("original_endpoints"):
        raise ConsumerError("context sidecar changed the sealed original RT/SP endpoint refs")
    if context_meta.get("baseline_artifacts") != context.get("baseline_artifacts"):
        raise ConsumerError("context sidecar changed the sealed baseline context/video refs")
    bridge_records = context_meta.get("endpoint_bridges")
    if not isinstance(bridge_records, dict) or set(bridge_records) != {"q00", "q11"}:
        raise ConsumerError("context sidecar lacks the exact q00/q11 endpoint bridge records")
    rebuilt: dict[str, Any] = {}
    for cell, old_arm in (("q00", "RT"), ("q11", "SP")):
        record = bridge_records[cell]
        original_ref = context["original_endpoints"][old_arm]
        if record.get("old_arm") != old_arm or record.get("old_wave_file_sha256") != original_ref.get("file_sha256"):
            raise ConsumerError(f"{cell} bridge does not name its exact old {old_arm} waveform")
        fresh_ref = context["cells"][cell]["waveform"]
        old_wave, old_wave_meta = _read_bridge_wave(original_ref, f"old/{cell}/{old_arm}")
        fresh_wave, fresh_wave_meta = _read_bridge_wave(fresh_ref, f"fresh/{cell}")
        if old_wave.shape != fresh_wave.shape:
            raise ConsumerError(f"{cell} and original {old_arm} waveform sample clocks differ")
        wave_delta = fresh_wave.astype(np.float64) - old_wave.astype(np.float64)
        wave_max = float(np.max(np.abs(wave_delta), initial=0.0))
        wave_rms = float(np.sqrt(np.mean(wave_delta * wave_delta, dtype=np.float64)))
        if wave_max > 1e-6:
            raise ConsumerError(f"{cell} fresh physical waveform exceeds the frozen 1e-6 endpoint bridge tolerance")
        _close_number(wave_max, record.get("wave_max_abs_error"), f"{cell} waveform max error")
        _close_number(wave_rms, record.get("wave_rms_error"), f"{cell} waveform RMS error")

        baseline_ref = context["baseline_artifacts"]["context"]
        baseline_path = Path(str(baseline_ref["path"])).resolve(strict=True)
        _check_sha(baseline_path, str(baseline_ref["sha256"]), f"baseline/{cell} context")
        with np.load(baseline_path, allow_pickle=False) as old_context:
            z_key, v_key = f"arm_{old_arm}_z", f"arm_{old_arm}_lip_embeddings"
            if z_key not in old_context or v_key not in old_context:
                raise ConsumerError(f"baseline context lacks {z_key}/{v_key}")
            old_z = np.asarray(old_context[z_key])
            old_v = np.asarray(old_context[v_key])
        new_z = np.asarray(context_arrays[f"{cell}_z"])
        new_v = np.asarray(context_arrays[f"{cell}_V"])
        if old_z.shape != new_z.shape or old_v.shape != new_v.shape:
            raise ConsumerError(f"{cell} fresh latent/V clock differs from original {old_arm}")
        latent_exact = bool(np.array_equal(old_z, new_z))
        latent_max = float(np.max(np.abs(old_z.astype(np.float64) - new_z.astype(np.float64)), initial=0.0))
        visual_exact = bool(np.array_equal(old_v, new_v))
        visual_mean_abs = float(np.mean(np.abs(old_v.astype(np.float64) - new_v.astype(np.float64))))
        if record.get("latent_bit_exact") is not latent_exact or record.get("V_bit_exact") is not visual_exact:
            raise ConsumerError(f"{cell} latent/V bit-exact bridge flags differ from independent arrays")
        _close_number(latent_max, record.get("latent_max_abs_error"), f"{cell} latent max error")
        _close_number(visual_mean_abs, record.get("V_mean_abs_error"), f"{cell} V mean absolute error")

        planned = row_by_id[f"{context['sample_id']}__{context['direction']}__{cell}__A_RAW"]
        result = result_by_id[f"{context['sample_id']}__{context['direction']}__{cell}__A_RAW"]
        if "video_path" in planned:
            raise ConsumerError(f"{cell} frozen plan unexpectedly contains a video_path field")
        video_rel = str(result.get("video_path", ""))
        expected_video_rel = f"videos/{context['sample_id']}/{context['direction']}/{cell}.mkv"
        if video_rel != expected_video_rel:
            raise ConsumerError(f"{cell} result video reference differs from the frozen producer output layout")
        visual_meta = context_meta.get("cell_metadata", {}).get(cell, {}).get("visual", {})
        if (
            visual_meta.get("condition") != cell
            or visual_meta.get("video_path") != video_rel
            or visual_meta.get("video_sha256") != result.get("video_sha256")
        ):
            raise ConsumerError(f"{cell} context visual sidecar does not bind the scored fresh video")
        fresh_video = _path_under(run_root, video_rel, f"fresh/{cell} video")
        fresh_video_sha = _check_sha(fresh_video, str(result.get("video_sha256", "")), f"fresh/{cell} video")
        if run_inventory.get(video_rel) != fresh_video_sha:
            raise ConsumerError(f"{cell} scored video SHA differs from the sealed all-files inventory")
        if int(visual_meta.get("video_size_bytes", -1)) != fresh_video.stat().st_size:
            raise ConsumerError(f"{cell} visual sidecar video byte count differs from the sealed video")
        old_video_ref = context["baseline_artifacts"]["videos"][old_arm]
        old_video = Path(str(old_video_ref["path"])).resolve(strict=True)
        old_video_sha = _check_sha(old_video, str(old_video_ref["sha256"]), f"baseline/{cell} video")
        recorded_pixels = record.get("pixel_bridge")
        if not isinstance(recorded_pixels, dict) or recorded_pixels.get("old_video_sha256") != old_video_sha or recorded_pixels.get("fresh_video_sha256") != fresh_video_sha:
            raise ConsumerError(f"{cell} pixel bridge does not bind the sealed old/new videos")
        pixel_rebuilt = _video_pixel_bridge(old_video, fresh_video)
        for name in ("frame_count", "pixel_bit_exact_frames", "maximum_absolute_pixel_difference"):
            if int(recorded_pixels.get(name, -1)) != int(pixel_rebuilt[name]):
                raise ConsumerError(f"{cell} recorded pixel bridge {name} differs from independent decode")
        _close_number(pixel_rebuilt["mean_absolute_pixel_difference"], recorded_pixels.get("mean_absolute_pixel_difference"), f"{cell} mean pixel difference")
        rebuilt[cell] = {
            "old_arm": old_arm,
            "old_waveform": old_wave_meta,
            "fresh_waveform": fresh_wave_meta,
            "wave_max_abs_error": wave_max,
            "wave_rms_error": wave_rms,
            "latent_bit_exact": latent_exact,
            "latent_max_abs_error": latent_max,
            "V_bit_exact": visual_exact,
            "V_mean_abs_error": visual_mean_abs,
            "old_video_sha256": old_video_sha,
            "fresh_video_sha256": fresh_video_sha,
            "pixel_bridge": pixel_rebuilt,
        }
    return rebuilt


def _inventory_map(path: Path) -> dict[str, str]:
    value = _read_json(path)
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if not isinstance(value, dict):
        raise ConsumerError("all-files inventory is not a path-to-SHA map")
    result: dict[str, str] = {}
    for name, item in value.items():
        digest = item if isinstance(item, str) else item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ConsumerError(f"malformed all-files inventory item: {name!r}")
        result[name.replace("\\", "/")] = digest.lower()
    return result


def _verify_inventory(root: Path, inventory_path: Path, expected_raw_sha: str) -> dict[str, Any]:
    _check_sha(inventory_path, expected_raw_sha, "five-cell all-files inventory")
    inventory = _inventory_map(inventory_path)
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*") if path.is_file() and path != inventory_path
    }
    missing = sorted(set(inventory) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(inventory))
    mismatched = []
    for relative in sorted(set(inventory) & set(actual_paths)):
        actual = _sha(actual_paths[relative])
        if actual != inventory[relative]:
            mismatched.append({"path": relative, "expected": inventory[relative], "actual": actual})
    if missing or extra or mismatched:
        raise ConsumerError(f"all-files coverage failure: missing={len(missing)} extra={len(extra)} mismatched={len(mismatched)}")
    return {"status": "PASS", "member_count": len(inventory), "missing": 0, "extra": 0, "mismatched": 0, "inventory_sha256": expected_raw_sha}


def _row_unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(array * array, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (array / np.maximum(norm, np.float32(1.0e-12))).astype(np.float32, copy=False)


def _cross_distance(visual: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy form of the sealed PyTorch p=2, eps=1e-6 distance."""
    v = np.asarray(visual, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or not len(v) or not len(a) or v.shape[1] != a.shape[1]:
        raise ConsumerError(f"invalid full embedding arrays: V={v.shape}, A={a.shape}")
    eps = np.float32(1.0e-6)
    cross = np.empty((len(v), len(a)), dtype=np.float32)
    for start in range(0, len(v), 8):
        stop = min(start + 8, len(v))
        delta = v[start:stop, None, :] - a[None, :, :]
        delta += eps
        squared = np.sum(delta * delta, axis=2, dtype=np.float32)
        np.sqrt(squared, out=squared)
        cross[start:stop] = squared
    zero_delta = v.copy()
    zero_delta += eps
    zero = np.sum(zero_delta * zero_delta, axis=1, dtype=np.float32)
    np.sqrt(zero, out=zero)
    return cross, zero.astype(np.float32, copy=False)


def _lag_from_cross(cross: np.ndarray, zero: np.ndarray, query_n: int, audio_n: int) -> np.ndarray:
    matrix = np.asarray(cross, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    if matrix.ndim != 2 or zero.shape != (matrix.shape[0],) or not 1 <= query_n <= matrix.shape[0] or not 1 <= audio_n <= matrix.shape[1]:
        raise ConsumerError("invalid V×A domains for lag gather")
    q = np.arange(query_n, dtype=np.int64)[:, None]
    source = q + np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, :]
    valid = (source >= 0) & (source < audio_n)
    clipped = np.clip(source, 0, audio_n - 1)
    values = matrix[np.broadcast_to(q, clipped.shape), clipped]
    return np.asarray(np.where(valid, values, zero[:query_n, None]), dtype=np.float32)


def _support_indices(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    g15 = np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32)
    g20 = np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32)
    return {"full": full, "valid": full.copy(), "guard20": g20, "guard15": g15}


def _metrics(lags: np.ndarray, indices: np.ndarray) -> dict[str, float] | None:
    ix = np.asarray(indices, dtype=np.int64)
    if ix.size == 0:
        return None
    curve = np.mean(np.asarray(lags, dtype=np.float32)[ix], axis=0, dtype=np.float32).astype(np.float64)
    median = float(np.median(curve))
    minimum = float(np.min(curve))
    d0 = float(curve[15])
    return {
        "support_n": int(ix.size),
        "curve": curve.tolist(),
        "C_fixed_anchor": median - float(curve[18]),
        "B_median": median,
        "D_min": minimum,
        "D0": d0,
        "search_gain": d0 - minimum,
        "optimal_offset_frames": float(15 - int(np.argmin(curve))),
        "C_best": median - minimum,
        "D_fixed_anchor": float(curve[18]),
    }


def _metrics_batch(lags: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray] | None:
    ix = np.asarray(indices, dtype=np.int64)
    if not ix.size:
        return None
    curves = np.mean(np.asarray(lags, dtype=np.float32)[:, ix, :], axis=1, dtype=np.float32).astype(np.float64)
    med = np.median(curves, axis=1)
    dmin = np.min(curves, axis=1)
    d0 = curves[:, 15]
    return {
        "C_fixed_anchor": med - curves[:, 18], "B_median": med,
        "D_min": dmin, "D0": d0, "search_gain": d0 - dmin,
        "optimal_offset_frames": (15 - np.argmin(curves, axis=1)).astype(np.float64),
        "C_best": med - dmin, "D_fixed_anchor": curves[:, 18],
    }


def _permuted_lags(cross: np.ndarray, zero: np.ndarray, permutations: np.ndarray, query_n: int, audio_n: int) -> np.ndarray:
    matrix = np.asarray(cross, dtype=np.float32)
    perms = np.asarray(permutations, dtype=np.int64)
    if perms.ndim != 2 or perms.shape[1] != audio_n:
        raise ConsumerError("time-null permutation bank has the wrong full-A size")
    if not np.all(np.sort(perms, axis=1) == np.arange(audio_n, dtype=np.int64)[None, :]):
        raise ConsumerError("time-null draws are not permutations of full A")
    q = np.arange(query_n, dtype=np.int64)[None, :, None]
    source = q + np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, None, :]
    valid = (source >= 0) & (source < audio_n)
    clipped = np.clip(source, 0, audio_n - 1)
    audio_ix = perms[:, clipped[0]]
    query_ix = np.broadcast_to(q, audio_ix.shape)
    gathered = matrix[query_ix, audio_ix]
    return np.asarray(np.where(valid, gathered, np.asarray(zero, dtype=np.float32)[None, :query_n, None]), dtype=np.float32)


def _anchor_values(cross: np.ndarray, zero: np.ndarray, q: np.ndarray, audio_n: int) -> np.ndarray:
    query = np.asarray(q, dtype=np.int64)
    source = query + 3
    valid = source < audio_n
    result = np.asarray(zero, dtype=np.float32)[query].astype(np.float64)
    if np.any(valid):
        result[valid] = np.asarray(cross, dtype=np.float32)[query[valid], source[valid]].astype(np.float64)
    return result


def _permuted_anchor(cross: np.ndarray, zero: np.ndarray, q: np.ndarray, permutations: np.ndarray, audio_n: int) -> np.ndarray:
    query = np.asarray(q, dtype=np.int64)
    source = query + 3
    valid = source < audio_n
    result = np.broadcast_to(np.asarray(zero, dtype=np.float32)[query][None, :], (len(permutations), len(query))).astype(np.float64, copy=True)
    if np.any(valid):
        audio_ix = np.asarray(permutations, dtype=np.int64)[:, source[valid]]
        query_ix = np.broadcast_to(query[valid][None, :], audio_ix.shape)
        result[:, valid] = np.asarray(cross, dtype=np.float32)[query_ix, audio_ix].astype(np.float64)
    return result


def _phone_intervals(phone_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete recipient tier, marking exact matched phones.

    The 16 frozen inferential labels control which tokens contribute a weight;
    they must not control the phone-overlap denominator. Other and unmatched
    recipient phones still own time and therefore remain visible in ambiguity,
    mixed-phone and zero-exposure diagnostics.
    """
    matched_rows = phone_map.get("matched_phones")
    recipient_rows = phone_map.get("recipient_phones")
    if not isinstance(matched_rows, list) or not isinstance(recipient_rows, list):
        raise ConsumerError("phone map lacks exact matched and full recipient phone tiers")
    matched_by_index: dict[int, dict[str, Any]] = {}
    for row in matched_rows:
        index = int(row["recipient_phone_index"])
        if index in matched_by_index:
            raise ConsumerError("phone map repeats an exact matched recipient token")
        matched_by_index[index] = row
    full_by_index: dict[int, dict[str, Any]] = {}
    for row in recipient_rows:
        index = int(row["index"])
        if index in full_by_index:
            raise ConsumerError("full recipient phone tier repeats an index")
        full_by_index[index] = row
    if not set(matched_by_index).issubset(full_by_index):
        raise ConsumerError("exact matched phone is absent from the full recipient tier")
    result = []
    for index, row in full_by_index.items():
        is_matched = index in matched_by_index
        matched = matched_by_index.get(index)
        label = str(row.get("normalized_label", ""))
        interval = row.get("interval_s")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ConsumerError("recipient phone has a malformed interval")
        start, end = map(float, interval)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ConsumerError("recipient phone interval is nonfinite or nonpositive")
        if matched is not None:
            matched_interval = matched.get("recipient_interval_s")
            if (
                str(matched.get("recipient_normalized_label")) != label
                or not isinstance(matched_interval, list) or len(matched_interval) != 2
                or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(matched_interval, (start, end)))
            ):
                raise ConsumerError("exact matched phone differs from its full recipient token")
        result.append({
            "phone_index": index, "label": label,
            "start": start, "end": end, "word_index": int(row.get("word_index", -1)),
            "matched": is_matched,
        })
    result.sort(key=lambda value: (value["start"], value["end"], value["phone_index"]))
    return result


def _owner_arrays(times: np.ndarray, intervals: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    count = np.zeros(len(t), dtype=np.int16)
    owner = np.full(len(t), -1, dtype=np.int32)
    for item in intervals:
        mask = (t >= item["start"]) & (t < item["end"])
        count += mask.astype(np.int16)
        owner[mask] = item["phone_index"]
    return count, owner


def _window_unions(starts: np.ndarray, query_n: int, mel_n: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if query_n < 1 or query_n > len(starts) - 5:
        raise ConsumerError("common SyncNet V prefix exceeds the five-frame Wav2Lip clock")
    unions: list[np.ndarray] = []
    indptr = [0]
    flat: list[int] = []
    cols = np.arange(16, dtype=np.int64)
    for query in range(query_n):
        values = np.unique(np.concatenate([starts[frame] + cols for frame in range(query, query + 5)]))
        if values.size == 0 or values.min() < 0 or values.max() >= mel_n:
            raise ConsumerError("five-frame Wav2Lip receptive-field union is outside mel clock")
        unions.append(values)
        flat.extend(int(value) for value in values)
        indptr.append(len(flat))
    return unions, np.asarray(indptr, dtype=np.int32), np.asarray(flat, dtype=np.int32)


def _phone_weights(unions: list[np.ndarray], mel_n: int, intervals: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.arange(mel_n, dtype=np.float64) / 80.0
    label_slot = {label: index for index, label in enumerate(PHONE_LABELS)}
    weights = np.zeros((len(PHONE_LABELS), len(PHONE_OUTCOMES), len(unions)), dtype=np.float64)
    stft_weights = np.zeros_like(weights)
    owner_count, owner_index = _owner_arrays(centers, intervals)
    boundaries = {float(value) for item in intervals for value in (item["start"], item["end"])}
    for token in intervals:
        if token["matched"] and token["label"] in label_slot:
            length = token["end"] - token["start"]
            boundaries.update((token["start"] + length / 3.0, token["start"] + 2.0 * length / 3.0))
    ordered_boundaries = sorted(boundaries)
    stft_segments: list[tuple[float, float, int, int]] = []
    for left, right in zip(ordered_boundaries, ordered_boundaries[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        active = [item for item in intervals if item["start"] <= midpoint < item["end"]]
        if len(active) != 1 or not active[0]["matched"]:
            continue
        token = active[0]
        slot = label_slot.get(token["label"])
        if slot is None:
            continue
        phase = (midpoint - token["start"]) / (token["end"] - token["start"])
        phase_index = 1 if phase < 1 / 3 else 2 if phase < 2 / 3 else 3
        stft_segments.append((left, right, slot, phase_index))
    for row, columns in enumerate(unions):
        cols = np.asarray(columns, dtype=np.int64)
        counts = owner_count[cols]
        owners = owner_index[cols]
        for token in intervals:
            slot = label_slot.get(token["label"])
            if slot is None or not token["matched"]:
                continue
            member = (centers[cols] >= token["start"]) & (centers[cols] < token["end"])
            member &= (counts == 1) & (owners == token["phone_index"])
            if not np.any(member):
                pass
            else:
                phase = (centers[cols] - token["start"]) / (token["end"] - token["start"])
                masks = (member, member & (phase < 1 / 3), member & (phase >= 1 / 3) & (phase < 2 / 3), member & (phase >= 2 / 3) & (phase < 1))
                for outcome, mask in enumerate(masks):
                    weights[slot, outcome, row] += float(np.count_nonzero(mask)) / float(len(cols))

        # Sensitivity only: integrate the centered 50 ms STFT support over
        # uniquely owned phone time. Split each phone at both its phase thirds
        # and every recipient-phone boundary so overlaps remain unassigned.
        col_left = centers[cols] - 0.025
        col_right = centers[cols] + 0.025
        for left, right, slot, phase_index in stft_segments:
            overlap = np.maximum(0.0, np.minimum(col_right, right) - np.maximum(col_left, left)) / 0.05
            mass = float(np.sum(overlap, dtype=np.float64)) / float(len(cols))
            stft_weights[slot, 0, row] += mass
            stft_weights[slot, phase_index, row] += mass
    if not np.allclose(weights[:, 1:, :].sum(axis=1), weights[:, 0, :], rtol=0, atol=1e-7):
        raise ConsumerError("phone phase coverage does not partition whole coverage")
    if not np.allclose(stft_weights[:, 1:, :].sum(axis=1), stft_weights[:, 0, :], rtol=0, atol=1e-7):
        raise ConsumerError("STFT-clean phone phase coverage does not partition whole coverage")
    return weights, stft_weights, owner_count, owner_index, centers


def _boundary_diagnostics(unions: list[np.ndarray], centers: np.ndarray, intervals: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    boundaries = np.asarray(sorted({float(item[key]) for item in intervals for key in ("start", "end")}), dtype=np.float64)
    n = len(unions)
    near40 = np.zeros(n, dtype=np.float32)
    stft50_crossing = np.zeros(n, dtype=np.float32)
    mixed = np.zeros(n, dtype=np.uint8)
    identities = np.zeros(n, dtype=np.int16)
    for i, cols in enumerate(unions):
        t = centers[np.asarray(cols, dtype=np.int64)]
        owners: set[int] = set()
        for item in intervals:
            if np.any((t >= item["start"]) & (t < item["end"])):
                owners.add(item["phone_index"])
        identities[i] = len(owners)
        mixed[i] = int(len(owners) > 1)
        if boundaries.size:
            distance = np.min(np.abs(t[:, None] - boundaries[None, :]), axis=1)
            near40[i] = float(np.mean(distance <= 0.040 + 1e-12))
            stft50_crossing[i] = float(np.mean(distance <= 0.025 + 1e-12))
    return {
        "boundary_within_40ms_column_fraction": near40,
        "stft50_boundary_crossing_column_fraction": stft50_crossing,
        "mixed_phone_rf": mixed,
        "recipient_phone_identity_count": identities,
    }


def _contrast(values: dict[str, float], name: str) -> float:
    if name == "mean": return values["q10"] - values["q00"]
    if name == "trajectory": return values["q01"] - values["q00"]
    if name == "joint": return values["q11"] - values["q00"]
    if name == "grid_cycle": return values["SELF_GRID"] - values["q00"]
    if name == "mean_minus_trajectory": return values["q10"] - values["q01"]
    if name == "mean_beyond_grid": return values["q10"] - values["SELF_GRID"]
    if name == "trajectory_beyond_grid": return values["q01"] - values["SELF_GRID"]
    if name == "joint_beyond_grid": return values["q11"] - values["SELF_GRID"]
    if name == "factorial_interaction": return values["q11"] - values["q10"] - values["q01"] + values["q00"]
    raise ConsumerError(f"unknown global contrast {name}")


def _contrast_batch(values: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "mean": return values["q10"] - values["q00"]
    if name == "trajectory": return values["q01"] - values["q00"]
    if name == "joint": return values["q11"] - values["q00"]
    if name == "grid_cycle": return values["SELF_GRID"] - values["q00"]
    if name == "mean_minus_trajectory": return values["q10"] - values["q01"]
    if name == "mean_beyond_grid": return values["q10"] - values["SELF_GRID"]
    if name == "trajectory_beyond_grid": return values["q01"] - values["SELF_GRID"]
    if name == "joint_beyond_grid": return values["q11"] - values["SELF_GRID"]
    if name == "factorial_interaction": return values["q11"] - values["q10"] - values["q01"] + values["q00"]
    raise ConsumerError(f"unknown global contrast {name}")


def _inventory_identity(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "inventory_sha256"})).hexdigest()


def _fingerprint(value: dict[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != field})).hexdigest()


def _verify_baseline_v6_seal(baseline: Any) -> dict[str, Any]:
    """Recheck the v6 external plan's raw bytes separately from its payload SHA."""
    if not isinstance(baseline, dict):
        raise ConsumerError("run protocol lacks the frozen baseline v6 seal")
    run_root = Path(str(baseline.get("path", ""))).resolve(strict=True)
    expected_files = baseline.get("expected_file_sha256")
    required = {
        "evaluation_protocol.json", "evaluation_rows.jsonl", "evaluation_results.jsonl",
        "all_files_sha256.json", "code_seal.json", "run_status.json",
    }
    if not isinstance(expected_files, dict) or set(expected_files) != required:
        raise ConsumerError("baseline v6 seal file-hash map differs from its exact six-file contract")
    observed: dict[str, str] = {}
    for name in sorted(required):
        observed[name] = _check_sha(run_root / name, str(expected_files[name]), f"baseline/{name}")
    for key, filename in (("status_sha256", "run_status.json"), ("code_seal_sha256", "code_seal.json")):
        if baseline.get(key) != observed[filename]:
            raise ConsumerError(f"baseline seal {key} does not match the actual file")

    inventory = _inventory_map(run_root / "all_files_sha256.json")
    for name in ("evaluation_protocol.json", "evaluation_rows.jsonl", "evaluation_results.jsonl", "code_seal.json", "run_status.json"):
        if inventory.get(name) != observed[name]:
            raise ConsumerError(f"baseline all-files inventory does not bind {name}")

    plan_path = Path(str(baseline.get("execution_plan_path", ""))).resolve(strict=True)
    raw_plan_sha = _check_sha(plan_path, str(baseline.get("execution_plan_raw_sha256", "")), "baseline/execution_plan/raw")
    plan = _read_json(plan_path)
    plan_payload_sha = _fingerprint(plan, "plan_sha256")
    if (
        plan.get("plan_sha256") != plan_payload_sha
        or baseline.get("execution_plan_payload_sha256") != plan_payload_sha
    ):
        raise ConsumerError("baseline external plan raw/payload SHA bindings are inconsistent")
    metadata_path = Path(str(baseline.get("seal_metadata_path", ""))).resolve(strict=True)
    _check_sha(metadata_path, str(baseline.get("seal_metadata_sha256", "")), "baseline/seal_metadata")
    metadata = _read_json(metadata_path)
    plan_meta = metadata.get("files", {}).get("execution_plan", {})
    if (
        metadata.get("execution_plan_sha256") != raw_plan_sha
        or plan_meta.get("path") != str(plan_path)
        or plan_meta.get("sha256") != raw_plan_sha
    ):
        raise ConsumerError("baseline metadata does not bind the exact external plan file")
    protocol = _read_json(run_root / "evaluation_protocol.json")
    status = _read_json(run_root / "run_status.json")
    code_seal = _read_json(run_root / "code_seal.json")
    if (
        protocol.get("execution_plan_sha256") != plan_payload_sha
        or status.get("execution_plan_sha256") != plan_payload_sha
        or code_seal.get("execution_plan_sha256") != plan_payload_sha
        or code_seal.get("protocol_sha256") != observed["evaluation_protocol.json"]
        or code_seal.get("row_manifest_sha256") != observed["evaluation_rows.jsonl"]
        or status.get("protocol_sha256") != observed["evaluation_protocol.json"]
        or status.get("row_manifest_sha256") != observed["evaluation_rows.jsonl"]
    ):
        raise ConsumerError("baseline run files do not bind their external plan payload and protocol/row bytes")
    return {
        "run_root": str(run_root),
        "execution_plan_path": str(plan_path),
        "execution_plan_raw_sha256": raw_plan_sha,
        "execution_plan_payload_sha256": plan_payload_sha,
        "seal_metadata_path": str(metadata_path),
        "seal_metadata_sha256": str(baseline["seal_metadata_sha256"]),
        "verified_files": observed,
    }


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ConsumerError(f"object arrays are forbidden: {path}")
    return arrays


def _expected_support_dict(count: int) -> dict[str, np.ndarray]:
    return _support_indices(count)


def _official_mel_starts(mel_frames: int, fps: int = 25, mel_step: int = 16) -> np.ndarray:
    if mel_frames < mel_step or fps <= 0 or mel_step <= 0:
        raise ConsumerError("invalid mel dimensions for official chunk loop")
    multiplier = 80.0 / float(fps)
    starts: list[int] = []
    index = 0
    while True:
        start = int(index * multiplier)
        if start + mel_step > mel_frames:
            starts.append(mel_frames - mel_step)
            break
        starts.append(start)
        index += 1
    return np.asarray(starts, dtype=np.int32)


def _compare_metric_record(observed: dict[str, float] | None, recorded: dict[str, Any], label: str) -> None:
    if observed is None:
        if set(recorded) != {"status", "support_n", "window_indices"}:
            raise ConsumerError(f"{label} empty-support fields differ from the producer contract")
        if recorded.get("status") not in {"too_short", "incomplete", "unavailable"} or int(recorded.get("support_n", -1)) != 0:
            raise ConsumerError(f"{label} should retain an empty support status")
        return
    emitted_fields = {
        "support_n", "curve", "lag_offsets_frames", "optimal_offset_frames", "optimal_column",
        "D_min", "B_median", "C_best", "fixed_anchor_column", "fixed_anchor_offset_frames",
        "D_fixed_anchor", "C_fixed_anchor", "status", "window_indices",
    }
    if set(recorded) != emitted_fields:
        missing = sorted(emitted_fields - set(recorded))
        extra = sorted(set(recorded) - emitted_fields)
        raise ConsumerError(f"{label} serialized metric field set differs: missing={missing}, extra={extra}")
    if recorded.get("status") != "ok":
        raise ConsumerError(f"{label} stored support status is not ok")
    if int(recorded.get("support_n", -1)) != int(observed["support_n"]):
        raise ConsumerError(f"{label} stored support_n differs from the frozen support")
    curve = recorded.get("curve")
    if not isinstance(curve, list) or len(curve) != 31:
        raise ConsumerError(f"{label} serialized lag curve is not the producer's 31-column curve")
    recorded_curve = np.asarray(curve, dtype=np.float64)
    observed_curve = np.asarray(observed["curve"], dtype=np.float64)
    if not np.isfinite(recorded_curve).all() or not np.allclose(
        recorded_curve, observed_curve, rtol=0, atol=METRIC_ATOL,
    ):
        raise ConsumerError(f"{label} stored lag curve differs from the independent reconstruction")
    expected_lag_offsets = list(range(15, -16, -1))
    if recorded.get("lag_offsets_frames") != expected_lag_offsets:
        raise ConsumerError(f"{label} stored lag_offsets_frames differs from the fixed 31-column axis")
    if int(recorded["fixed_anchor_column"]) != ANCHOR["column"]:
        raise ConsumerError(f"{label} stored fixed_anchor_column differs from the frozen anchor")
    if int(recorded["fixed_anchor_offset_frames"]) != ANCHOR["official_offset_frames"]:
        raise ConsumerError(f"{label} stored fixed_anchor_offset_frames differs from the frozen anchor")
    recorded_optimal_column = int(recorded["optimal_column"])
    observed_optimal_column = 15 - int(observed["optimal_offset_frames"])
    if recorded_optimal_column != observed_optimal_column or recorded_optimal_column != int(np.argmin(recorded_curve)):
        raise ConsumerError(f"{label} stored optimal_column/offset differs from the independently rebuilt curve")

    # The frozen producer serializes the full lag curve but does not emit D0 or
    # search_gain. Derive them independently from each curve instead of treating
    # absent producer fields as metric failures or trusting a second summary.
    recorded_values = {
        "C_fixed_anchor": recorded.get("C_fixed_anchor"),
        "B_median": recorded.get("B_median"),
        "D_min": recorded.get("D_min"),
        "D0": float(recorded_curve[15]),
        "search_gain": float(recorded_curve[15] - float(recorded["D_min"])),
        "optimal_offset_frames": recorded.get("optimal_offset_frames"),
        "C_best": recorded.get("C_best"),
        "D_fixed_anchor": recorded.get("D_fixed_anchor"),
    }
    for metric, value in recorded_values.items():
        if metric not in observed:
            raise ConsumerError(f"{label} lacks independently recomputed {metric}")
        if value is None:
            raise ConsumerError(f"{label} serialized metric {metric} is missing")
        if metric == "optimal_offset_frames":
            if int(value) != int(observed[metric]):
                raise ConsumerError(f"{label} recorded {metric} differs")
        else:
            if not math.isfinite(float(value)) or abs(float(value) - observed[metric]) > METRIC_ATOL:
                raise ConsumerError(f"{label} recorded/derived {metric} differs: {value!r} vs {observed[metric]}")


def _check_runtime_flags(path: Path, expected_sha: str, expected_runner_sha: str) -> dict[str, Any]:
    _check_sha(path, expected_sha, "runtime forward-flag sidecar")
    value = _read_json(path)
    expected = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    }
    if value.get("schema_version") != 5 or value.get("policy") != expected:
        raise ConsumerError("runtime sidecar does not use the frozen five-flag policy")
    if value.get("runtime_forward_gate") != "pass" or value.get("runner_exit_code") != 0:
        raise ConsumerError("runtime forward gate did not finish with exit zero")
    if int(value.get("flag_mismatch_count", -1)) != 0 or int(value.get("nn_module_forward_call_count", 0)) < 1:
        raise ConsumerError("runtime sidecar reports a flag mismatch or no model forwards")
    if value.get("controlled_stop_reason") is not None:
        raise ConsumerError("runtime sidecar records a controlled stop")
    if value.get("runner_sha256") != expected_runner_sha:
        raise ConsumerError("runtime sidecar runner source differs from the run protocol")
    counts = value.get("nn_module_call_counts_by_type")
    if not isinstance(counts, dict) or sum(int(number) for number in counts.values()) != int(value["nn_module_forward_call_count"]):
        raise ConsumerError("runtime module-call counts do not sum to the forward count")
    if value.get("torch_flags_after_runner") is None:
        raise ConsumerError("runtime sidecar lacks the final post-run torch flag state")
    after = value["torch_flags_after_runner"]
    if {key: after.get(key) for key in expected} != expected:
        raise ConsumerError("final post-run torch flags differ from the frozen policy")
    for key in ("hash_chain_initial", "hash_chain_last"):
        if not isinstance(value.get(key), str) or len(value[key]) != 64:
            raise ConsumerError(f"runtime sidecar has invalid {key}")
    return value


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_dir).resolve(strict=True)
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise ConsumerError(f"output directory must be fresh: {out}")
    design_path = Path(args.design_file).resolve(strict=True)
    addendum_path = Path(args.metric_addendum).resolve(strict=True)
    phone_inventory_path = Path(args.phone_inventory).resolve(strict=True)
    design_sha = _check_sha(design_path, DESIGN_SHA256, "frozen shape-factor design")
    addendum_sha = _check_sha(addendum_path, METRIC_ADDENDUM_SHA256, "frozen metric addendum")
    phone_inventory_sha = _check_sha(phone_inventory_path, PHONE_INVENTORY_SHA256, "CAL-only phone inventory")
    phone_inventory = _read_json(phone_inventory_path)
    if (
        phone_inventory.get("status") != "FROZEN_CAL_ONLY_PHONE_INVENTORY"
        or phone_inventory.get("selection_split") != "calibration"
        or phone_inventory.get("evaluation_textgrids_or_endpoints_read") is not False
        or phone_inventory.get("inventory_sha256") != _inventory_identity(phone_inventory)
        or phone_inventory.get("selected_phone_labels") != list(PHONE_LABELS)
        or phone_inventory.get("selected_phone_count") != len(PHONE_LABELS)
    ):
        raise ConsumerError("CAL-only phone inventory identity or frozen label order differs")
    run_files = {
        "evaluation_protocol.json": args.protocol_sha256,
        "evaluation_rows.jsonl": args.rows_sha256,
        "evaluation_results.jsonl": args.results_sha256,
        "run_status.json": args.status_sha256,
        "all_files_sha256.json": args.all_files_sha256,
        "code_seal.json": args.code_seal_sha256,
    }
    observed_run_files = {name: _check_sha(run_root / name, digest, f"run/{name}") for name, digest in run_files.items()}
    inventory_check = _verify_inventory(run_root, run_root / "all_files_sha256.json", args.all_files_sha256)
    run_inventory = _inventory_map(run_root / "all_files_sha256.json")
    protocol = _read_json(run_root / "evaluation_protocol.json")
    baseline_binding_check = _verify_baseline_v6_seal(protocol.get("baseline_seal"))
    if protocol.get("protocol_sha256") != _fingerprint(protocol, "protocol_sha256"):
        raise ConsumerError("run protocol embedded fingerprint is invalid")
    if protocol.get("row_manifest_sha256") != observed_run_files["evaluation_rows.jsonl"]:
        raise ConsumerError("run protocol does not bind the exact row ledger")
    if protocol.get("design_sha256") != DESIGN_SHA256 or protocol.get("metric_addendum_sha256") != METRIC_ADDENDUM_SHA256:
        raise ConsumerError("run protocol is not bound to the frozen design and metric addendum")
    if protocol.get("direction_order") != list(EXPECTED_DIRECTIONS) or protocol.get("cell_order") != list(EXPECTED_CELLS) or protocol.get("audio_domains") != list(EXPECTED_AUDIO):
        raise ConsumerError("run protocol changed the frozen direction/cell/audio domains")
    if protocol.get("anchor") != ANCHOR:
        raise ConsumerError("run protocol changed the fixed k=3 / column18 anchor")
    cohort = protocol.get("cohort", {})
    if {key: cohort.get(key) for key in ("pairs", "speakers", "contexts", "videos", "score_rows")} != {"pairs": 80, "speakers": 40, "contexts": 160, "videos": 800, "score_rows": 1600}:
        raise ConsumerError("run protocol cohort cardinalities differ from the frozen design")
    if protocol.get("common_support_policy") != "minimum full V-window count across all five fresh videos and full A-row counts for A_RAW/A_RT; matrices retain full V×A domains; padding uses each full A domain":
        raise ConsumerError("run protocol common-support/full-A policy differs")

    plan_path = Path(args.execution_plan).resolve(strict=True)
    plan_raw = _check_sha(plan_path, args.execution_plan_sha256, "external execution plan")
    plan = _read_json(plan_path)
    if plan.get("protocol_sha256") != _fingerprint(plan, "protocol_sha256"):
        raise ConsumerError("external execution plan embedded fingerprint is invalid")
    if plan.get("row_manifest_sha256") != observed_run_files["evaluation_rows.jsonl"]:
        raise ConsumerError("external plan does not bind the emitted row ledger")
    code_seal = _read_json(run_root / "code_seal.json")
    if (
        code_seal.get("execution_plan_sha256") != plan_raw
        or code_seal.get("protocol_sha256") != observed_run_files["evaluation_protocol.json"]
        or code_seal.get("row_manifest_sha256") != observed_run_files["evaluation_rows.jsonl"]
    ):
        raise ConsumerError("run code seal does not bind the supplied plan/protocol/row ledger")
    status = _read_json(run_root / "run_status.json")
    expected_status = {
        "status": "complete", "planned_contexts": 160, "completed_contexts": 160,
        "failed_contexts": 0, "planned_rows": 1600, "scored_rows": 1600, "failed_rows": 0,
    }
    if {key: status.get(key) for key in expected_status} != expected_status:
        raise ConsumerError(f"five-cell producer is not complete for the fixed denominator: {status}")
    if status.get("protocol_sha256") != observed_run_files["evaluation_protocol.json"] or status.get("row_manifest_sha256") != observed_run_files["evaluation_rows.jsonl"]:
        raise ConsumerError("final run status does not bind the exact protocol and row manifest")

    # Cross-check every input seal recorded by the frozen GPU plan against the
    # actual source file. These are physical/manifests and model assets only;
    # no score array is opened in preflight.
    input_seals = protocol.get("input_seals")
    required_input_seals = {
        "shape_manifest", "parent_manifest", "input_manifest", "tts_manifest", "evaluation_plan",
        "shape_qa", "cal_qa", "qa_repair_authorization", "evaluation_freeze",
        "design", "metric_addendum", "face_registry",
    }
    if not isinstance(input_seals, dict) or set(input_seals) != required_input_seals:
        raise ConsumerError("run protocol input-seal keys differ from the frozen ten-input contract")
    checked_inputs: dict[str, dict[str, str]] = {}
    for name, seal in input_seals.items():
        if not isinstance(seal, dict) or not seal.get("path") or not seal.get("sha256"):
            raise ConsumerError(f"malformed protocol input seal: {name}")
        source_path = Path(str(seal["path"])).resolve(strict=True)
        digest = _check_sha(source_path, str(seal["sha256"]), f"protocol-input/{name}")
        checked_inputs[name] = {"path": str(source_path), "sha256": digest}
    if checked_inputs.get("shape_manifest", {}).get("sha256") != args.shape_manifest_sha256:
        raise ConsumerError("shape physical manifest SHA differs from the separately frozen EVAL manifest")
    if (
        checked_inputs.get("shape_manifest", {}).get("sha256") != SHAPE_MANIFEST_RAW_SHA256
        or checked_inputs.get("shape_qa", {}).get("sha256") != SHAPE_QA_RAW_SHA256
    ):
        raise ConsumerError("shape manifest/physical QA raw SHA differs from the full root readback")
    shape_path = Path(args.shape_manifest).resolve(strict=True)
    if str(shape_path) != checked_inputs.get("shape_manifest", {}).get("path"):
        raise ConsumerError("shape manifest path differs from the runner's exact bound input")
    shape_manifest = _read_json(shape_path)
    if (
        shape_manifest.get("split") != "evaluation"
        or shape_manifest.get("status") != "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW"
        or shape_manifest.get("manifest_sha256") != SHAPE_MANIFEST_PAYLOAD_SHA256
        or _fingerprint(shape_manifest, "manifest_sha256") != SHAPE_MANIFEST_PAYLOAD_SHA256
    ):
        raise ConsumerError("shape physical EVAL manifest is not in a ready state")
    if shape_manifest.get("manifest_sha256") != _fingerprint(shape_manifest, "manifest_sha256"):
        raise ConsumerError("shape physical EVAL manifest embedded fingerprint is invalid")
    if any(shape_manifest.get(name) is not False for name in ("evaluation_effects_read", "evaluation_outcomes_read", "model_forward_performed")):
        raise ConsumerError("shape physical manifest must certify no EVAL effect/outcome reads or model forwards")
    physical_records = shape_manifest.get("records")
    if not isinstance(physical_records, list) or len(physical_records) != EXPECTED_CONTEXTS:
        raise ConsumerError("shape physical manifest does not retain all 160 intended direction rows")
    physical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for physical_row in physical_records:
        physical_key = (str(physical_row.get("id")), str(physical_row.get("direction")))
        # Split is frozen at the physical manifest root; direction rows from
        # this producer intentionally carry no redundant row-level split.
        if physical_key in physical_by_key or physical_row.get("split") not in (None, "evaluation") or physical_row.get("status") != "PASS":
            raise ConsumerError(f"shape physical manifest row identity/status is invalid: {physical_key}")
        physical_by_key[physical_key] = physical_row
    physical_plan_ref = checked_inputs["evaluation_plan"]
    physical_plan = _read_json(Path(physical_plan_ref["path"]))
    physical_plan_payload_sha = _fingerprint(physical_plan, "manifest_sha256")
    freeze_ref = checked_inputs["evaluation_freeze"]
    freeze = _read_json(Path(freeze_ref["path"]))
    freeze_payload_sha = _fingerprint(freeze, "manifest_sha256")
    repair_ref = checked_inputs["qa_repair_authorization"]
    repair_path = Path(repair_ref["path"])
    repair = _read_json(repair_path)
    repair_payload_sha = _fingerprint(repair, "manifest_sha256")
    if (
        repair_ref["sha256"] != QA_REPAIR_AUTHORIZATION_RAW_SHA256
        or repair_payload_sha != QA_REPAIR_AUTHORIZATION_PAYLOAD_SHA256
        or repair.get("manifest_sha256") != repair_payload_sha
        or repair.get("schema") != QA_REPAIR_SCHEMA
        or repair.get("status") != "APPROVED"
        or repair.get("scope") != "EVAL_PHYSICAL_WAVEFORMS_ONLY"
        or repair.get("repair_changes") != list(QA_REPAIR_CHANGES)
        or repair.get("scientific_controls_changed") is not False
        or repair.get("evaluation_inputs_read") is not True
        or repair.get("evaluation_outcomes_read") is not False
        or repair.get("evaluation_effects_read") is not False
        or repair.get("model_forward_authorized") is not False
        or repair.get("gpu_forward_authorized") is not False
        or repair.get("original_analysis_plan") != {
            "path": physical_plan_ref["path"],
            "raw_sha256": physical_plan_ref["sha256"],
            "payload_sha256": physical_plan_payload_sha,
        }
        or repair.get("original_evaluation_freeze") != {
            "path": freeze_ref["path"],
            "raw_sha256": freeze_ref["sha256"],
            "payload_sha256": freeze_payload_sha,
        }
        or physical_plan.get("manifest_sha256") != physical_plan_payload_sha
        or freeze.get("manifest_sha256") != freeze_payload_sha
        or physical_plan.get("status") != "FROZEN_FOR_EVALUATION_WAVEFORM_PRODUCTION"
        or physical_plan.get("scope") != "evaluation_only"
        or physical_plan.get("evaluation_waveform_production_authorized") is not True
        or physical_plan.get("model_forward_performed") is not False
        or physical_plan.get("evaluation_effects_read") is not False
        or physical_plan.get("evaluation_outcomes_read") is not False
        or freeze.get("analysis_plan_raw_sha256") != physical_plan_ref["sha256"]
        or freeze.get("analysis_plan_sha256") != physical_plan_payload_sha
    ):
        raise ConsumerError("approved QA repair sidecar does not bind the unchanged exact plan/freeze")
    physical_plan_qa_source_ref = physical_plan.get("input_bindings", {}).get("independent_qa_source")
    if not isinstance(physical_plan_qa_source_ref, dict):
        raise ConsumerError("frozen evaluation plan lacks its independent_qa_source input binding")
    physical_plan_qa_source_sha = str(physical_plan_qa_source_ref.get("raw_sha256", ""))
    if (
        not physical_plan_qa_source_sha
        or physical_plan.get("independent_qa_source_path") != physical_plan_qa_source_ref.get("path")
        or physical_plan.get("independent_qa_source_sha256") != physical_plan_qa_source_sha
        or freeze.get("independent_qa_source_sha256") != physical_plan_qa_source_sha
    ):
        raise ConsumerError("plan/freeze QA03 source path and SHA bindings disagree")
    old_qa_ref = repair.get("original_qa_source")
    new_qa_ref = repair.get("repaired_qa_source")
    if not isinstance(old_qa_ref, dict) or not isinstance(new_qa_ref, dict):
        raise ConsumerError("approved QA repair lacks original/repaired QA source refs")
    if (
        old_qa_ref.get("path") != physical_plan_qa_source_ref.get("path")
        or old_qa_ref.get("raw_sha256") != physical_plan_qa_source_sha
        or _check_sha(Path(old_qa_ref["path"]), str(old_qa_ref["raw_sha256"]), "original QA03 source") != old_qa_ref["raw_sha256"]
    ):
        raise ConsumerError("approved QA repair's original QA03 ref differs from the unchanged plan/freeze")
    qa_auth_report_expected = {
        "path": str(repair_path),
        "raw_sha256": repair_ref["sha256"],
        "payload_sha256": repair_payload_sha,
        "schema": QA_REPAIR_SCHEMA,
        "status": "APPROVED",
        "original_qa_source_path": old_qa_ref["path"],
        "original_qa_source_sha256": old_qa_ref["raw_sha256"],
        "repaired_qa_source_path": new_qa_ref.get("path"),
        "repaired_qa_source_sha256": new_qa_ref.get("raw_sha256"),
        "repair_changes": list(QA_REPAIR_CHANGES),
    }
    for qa_name in ("shape_qa", "cal_qa"):
        if qa_name in checked_inputs:
            qa = _read_json(Path(checked_inputs[qa_name]["path"]))
            if qa.get("status") != "PASS":
                raise ConsumerError(f"required {qa_name} artifact is not PASS")
            if qa_name == "shape_qa":
                qa_payload_sha = _fingerprint(qa, "manifest_sha256")
                new_qa_path = Path(str(new_qa_ref.get("path", ""))).resolve(strict=True)
                new_qa_sha = str(new_qa_ref.get("raw_sha256", ""))
                _check_sha(new_qa_path, new_qa_sha, "repaired QA04 source")
                if (
                    qa.get("manifest_sha256") != SHAPE_QA_PAYLOAD_SHA256
                    or qa_payload_sha != SHAPE_QA_PAYLOAD_SHA256
                    or qa.get("scope") != "EVAL physical waveforms only; 80 utterances / 40 speakers / 160 directions / 800 WORLD waveforms; no video/model/scoring inputs"
                    or qa.get("evaluation_inputs_read") is not True
                    or any(qa.get(field) is not False for field in ("effects_read", "evaluation_outcomes_read", "model_forward_performed"))
                ):
                    raise ConsumerError("physical EVAL QA does not certify no EVAL effects/outcomes/model forwards")
                if (
                    qa.get("input_report", {}).get("path") != checked_inputs["shape_manifest"]["path"]
                    or qa.get("input_report", {}).get("raw_sha256") != args.shape_manifest_sha256
                    or qa.get("input_report", {}).get("payload_sha256") != SHAPE_MANIFEST_PAYLOAD_SHA256
                ):
                    raise ConsumerError("physical EVAL QA does not bind the exact shape manifest bytes")
                if (
                    qa.get("direction_rows_verified") != 160
                    or qa.get("cell_waveforms_verified") != 800
                    or qa.get("input_files_verified") != 1440
                    or qa.get("output_files_verified") != 1120
                ):
                    raise ConsumerError("physical EVAL QA does not cover the full frozen 160/800/1440/1120 denominator")
                if (
                    qa.get("bound_world_manifest_raw_sha256") != checked_inputs["parent_manifest"]["sha256"]
                    or qa.get("bound_analysis_source_sha256") != shape_manifest.get("analysis_source_sha256")
                    or qa.get("bound_analysis_spec_sha256") != DESIGN_SHA256
                    or qa.get("independent_qa_source_path") != str(new_qa_path)
                    or qa.get("independent_qa_source_sha256") != new_qa_sha
                    or qa.get("plan_bound_independent_qa_source_sha256") != old_qa_ref.get("raw_sha256")
                    or qa.get("bound_analysis_plan_raw_sha256") != physical_plan_ref.get("sha256")
                    or qa.get("bound_analysis_plan_payload_sha256") != physical_plan_payload_sha
                    or qa.get("qa_repair_authorization") != qa_auth_report_expected
                ):
                    raise ConsumerError("physical EVAL QA is not bound to the approved old/new QA, parent, source, design, and original plan")
            if qa_name == "cal_qa":
                if (
                    not str(qa.get("scope", "")).startswith("CAL only;")
                    or len(qa.get("records", [])) != 40
                    or qa.get("direction_rows_verified") != 40
                    or qa.get("cell_waveforms_verified") != 200
                    or qa.get("manifest_sha256") != _fingerprint(qa, "manifest_sha256")
                    or any(qa.get(field) is not False for field in (
                        "evaluation_inputs_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed",
                    ))
                ):
                    raise ConsumerError("CAL QA is not a calibration-only no-EVAL/no-forward PASS")

    runtime_path = Path(args.runtime_flags).resolve(strict=True)
    runtime = _check_runtime_flags(runtime_path, args.runtime_flags_sha256, str(protocol.get("execution_assets", {}).get("runner_sha256", "")))
    rows_path = run_root / "evaluation_rows.jsonl"
    results_path = run_root / "evaluation_results.jsonl"
    plan_rows = _read_jsonl(rows_path)
    results = _read_jsonl(results_path)
    plan_contexts = plan.get("contexts")
    plan_context_rows = {(str(ctx["sample_id"]), str(ctx["direction"])): ctx for ctx in plan_contexts or []}
    if len(plan_context_rows) != EXPECTED_CONTEXTS or len(plan_rows) != EXPECTED_ROWS or len(results) != EXPECTED_ROWS:
        raise ConsumerError(f"preflight row/context counts are wrong: {len(plan_context_rows)}/{len(plan_rows)}/{len(results)}")
    if [row.get("row_id") for row in plan_rows] != [row.get("row_id") for row in results]:
        raise ConsumerError("planned and scored row identities differ or are out of order")
    expected_row_ids = []
    expected_context_order = []
    for context in plan_contexts:
        key = (str(context["sample_id"]), str(context["direction"]))
        expected_context_order.append(key)
        for cell in EXPECTED_CELLS:
            for domain in EXPECTED_AUDIO:
                expected_row_ids.append(f"{key[0]}__{key[1]}__{cell}__{domain}")
    if expected_row_ids != [str(row.get("row_id")) for row in plan_rows]:
        raise ConsumerError("row order differs from context × five generation cells × two fixed audio domains")
    if len(set(expected_context_order)) != EXPECTED_CONTEXTS or tuple(dict.fromkeys(str(row.get("speaker_id")) for row in plan_rows)) == ():
        raise ConsumerError("context identifiers or speaker roster are invalid")
    speakers = sorted({str(context["speaker_id"]) for context in plan_contexts})
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ConsumerError(f"expected exactly 40 frozen EVAL speakers, observed {len(speakers)}")
    ids_by_speaker: dict[str, set[str]] = defaultdict(set)
    for context in plan_contexts:
        ids_by_speaker[str(context["speaker_id"])].add(str(context["sample_id"]))
    if any(len(ids) != 2 for ids in ids_by_speaker.values()):
        raise ConsumerError("the speaker-weighted roster must retain exactly two utterances per speaker")

    row_by_id: dict[str, dict[str, Any]] = {}
    result_by_id: dict[str, dict[str, Any]] = {}
    context_table: dict[tuple[str, str], dict[str, Any]] = {}
    result_context_table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for context in plan_contexts:
        key = (str(context["sample_id"]), str(context["direction"]))
        context_table[key] = context
    for planned, result in zip(plan_rows, results):
        if planned.get("split") != "evaluation" or result.get("status") != "scored" or result.get("face_id") != "primary":
            raise ConsumerError(f"planned row is not a scored primary-face EVAL row: {planned.get('row_id')}")
        expected_identity = {key: planned.get(key) for key in ("ordered_eval_index", "sample_id", "speaker_id", "direction", "condition", "audio_domain", "context_key", "face_id")}
        if any(result.get(key) != value for key, value in expected_identity.items()):
            raise ConsumerError(f"result row identity differs from plan: {planned.get('row_id')}")
        if result.get("fixed_anchor_k") != 3 or result.get("fixed_anchor_column") != 18 or result.get("fixed_anchor_official_offset") != -3:
            raise ConsumerError(f"result row changed the frozen anchor: {planned.get('row_id')}")
        row_id = str(planned["row_id"])
        if row_id in row_by_id:
            raise ConsumerError(f"duplicate result row id: {row_id}")
        row_by_id[row_id] = planned
        result_by_id[row_id] = result
        context_key = (str(planned["sample_id"]), str(planned["direction"]))
        result_context_table[(context_key[0], context_key[1], str(planned["condition"]), str(planned["audio_domain"]))] = result
    contexts: list[dict[str, Any]] = []
    metadata_checks = []
    for context_index, key in enumerate(expected_context_order):
        context = context_table[key]
        physical_row = physical_by_key.get(key)
        if physical_row is None:
            raise ConsumerError(f"planned context is absent from frozen physical manifest: {key}")
        if (
            str(context.get("speaker_id")) != str(physical_row.get("speaker"))
            or int(context.get("ordered_eval_index", -1)) != context_index
            or str(context.get("source_identity_sha256")) != str(physical_row.get("source_identity_sha256"))
            or str(context.get("recipient_condition")) != str(physical_row.get("recipient_condition"))
            or str(context.get("donor_condition")) != str(physical_row.get("donor_condition"))
        ):
            raise ConsumerError(f"planned context identity differs from its frozen physical source row: {key}")
        physical_phone_ref = physical_row.get("phone_map", {})
        planned_phone_ref = context.get("phone_map", {})
        if (
            str(Path(str(planned_phone_ref.get("path", ""))).resolve()) != str(Path(str(physical_phone_ref.get("path", ""))).resolve())
            or planned_phone_ref.get("file_sha256") != physical_phone_ref.get("file_sha256")
        ):
            raise ConsumerError(f"planned phone map does not point to the exact physical phone map: {key}")
        for domain in EXPECTED_AUDIO:
            planned_audio = context.get("scoring_audio", {}).get(domain, {})
            physical_audio = physical_row.get("scoring_audio", {}).get(domain, {})
            for field in ("path", "file_sha256", "sample_rate_hz", "samples", "float32_pcm_sha256"):
                left = Path(str(planned_audio.get(field, ""))).resolve() if field == "path" else planned_audio.get(field)
                right = Path(str(physical_audio.get(field, ""))).resolve() if field == "path" else physical_audio.get(field)
                if left != right:
                    raise ConsumerError(f"planned {domain} audio ref differs from physical manifest: {key}/{field}")
            if domain == "A_RAW" and planned_audio.get("pcm16_sha256") != physical_audio.get("pcm16_sha256"):
                raise ConsumerError(f"planned A_RAW PCM16 identity differs from physical manifest: {key}")
        for cell in EXPECTED_CELLS:
            planned_cell = context.get("cells", {}).get(cell, {})
            physical_cell = physical_row.get("cells", {}).get(cell, {})
            wave_ref = physical_cell.get("waveform", {})
            params_ref = physical_cell.get("parameter_npz", {})
            if (
                str(Path(str(planned_cell.get("waveform", {}).get("path", ""))).resolve()) != str(Path(str(wave_ref.get("path", ""))).resolve())
                or planned_cell.get("waveform", {}).get("file_sha256") != wave_ref.get("file_sha256")
                or planned_cell.get("waveform", {}).get("float32_pcm_sha256") != wave_ref.get("float32_pcm_sha256")
                or str(Path(str(planned_cell.get("parameter_npz", {}).get("path", ""))).resolve()) != str(Path(str(params_ref.get("path", ""))).resolve())
                or planned_cell.get("parameter_npz", {}).get("file_sha256") != params_ref.get("file_sha256")
            ):
                raise ConsumerError(f"planned {cell} generator cell differs from physical manifest: {key}")
        context_path = _path_under(run_root, str(plan_rows[context_index * 10]["context_npz"]), "context NPZ")
        context_json = context_path.with_suffix(".json")
        context_npz_sha = _sha(context_path)
        if not context_json.is_file():
            raise ConsumerError(f"context sidecar is missing: {key}")
        context_meta = _read_json(context_json)
        if (
            context_meta.get("context_npz", {}).get("path") != str(plan_rows[context_index * 10]["context_npz"])
            or context_meta.get("context_npz", {}).get("sha256") != context_npz_sha
            or int(context_meta.get("context_npz", {}).get("bytes", -1)) != context_path.stat().st_size
        ):
            raise ConsumerError(f"context sidecar does not bind its NPZ: {key}")
        if (context_meta.get("sample_id"), context_meta.get("direction"), context_meta.get("ordered_eval_index")) != (key[0], key[1], context.get("ordered_eval_index")):
            raise ConsumerError(f"context sidecar identity differs from plan: {key}")
        for field in ("source_identity_sha256", "recipient_condition", "donor_condition", "speaker_id"):
            if context_meta.get(field) != context.get(field):
                raise ConsumerError(f"context sidecar differs from plan field {field}: {key}")
        if context_meta.get("cells") != context.get("cells") or context_meta.get("scoring_audio") != context.get("scoring_audio"):
            raise ConsumerError(f"context sidecar changed physical cell or scoring audio refs: {key}")
        phone_ref = context.get("phone_map", {})
        phone_path = Path(str(phone_ref.get("path", ""))).resolve(strict=True)
        if _sha(phone_path) != phone_ref.get("file_sha256"):
            raise ConsumerError(f"phone map SHA mismatch: {key}")
        phone_map = _read_json(phone_path)
        if context_meta.get("phone_map_record") != phone_map or phone_map.get("id") != key[0] or phone_map.get("recipient_condition") != context.get("recipient_condition") or phone_map.get("donor_condition") != context.get("donor_condition"):
            raise ConsumerError(f"context phone map is not the exact sealed direction map: {key}")
        context_arrays = _read_npz(context_path)
        required = {f"{domain}_{name}" for domain in EXPECTED_AUDIO for name in ("wave", "units", "mfcc", "emb")}
        required |= {f"{cell}_{name}" for cell in EXPECTED_CELLS for name in ("mel", "chunks", "starts", "z", "V")}
        required |= {f"support_{support}" for support in SUPPORTS}
        if required - set(context_arrays):
            raise ConsumerError(f"context NPZ lacks required arrays {sorted(required - set(context_arrays))}: {key}")
        arrays_meta = context_meta.get("arrays", {})
        for name, array in context_arrays.items():
            info = arrays_meta.get(name)
            if not isinstance(info, dict) or info.get("shape") != list(array.shape) or info.get("dtype") != str(array.dtype) or info.get("sha256") != _array_sha(array):
                raise ConsumerError(f"context array metadata/hash differs: {key}/{name}")
        starts0 = np.asarray(context_arrays[f"{EXPECTED_CELLS[0]}_starts"], dtype=np.int32)
        audio_counts = {domain: len(context_arrays[f"{domain}_emb"]) for domain in EXPECTED_AUDIO}
        visual_counts = {cell: len(context_arrays[f"{cell}_V"]) for cell in EXPECTED_CELLS}
        audio_metadata = context_meta.get("audio_metadata", {})
        for domain in EXPECTED_AUDIO:
            wave = np.asarray(context_arrays[f"{domain}_wave"])
            units = np.asarray(context_arrays[f"{domain}_units"])
            mfcc = np.asarray(context_arrays[f"{domain}_mfcc"])
            embeddings = np.asarray(context_arrays[f"{domain}_emb"])
            if (
                wave.dtype != np.float32 or wave.ndim != 1 or not np.isfinite(wave).all()
                or units.dtype != np.float64 or units.shape != wave.shape
                or mfcc.dtype != np.float64 or mfcc.ndim != 2 or not np.isfinite(mfcc).all()
                or embeddings.dtype != np.float32 or embeddings.ndim != 2 or not np.isfinite(embeddings).all()
            ):
                raise ConsumerError(f"scoring audio arrays have invalid dtype/shape/finite values: {key}/{domain}")
            if len(embeddings) != len(units) // 640 - 5:
                raise ConsumerError(f"scoring audio embedding clock differs from official SyncNet window count: {key}/{domain}")
            if domain == "A_RAW":
                reconstructed_pcm16 = np.rint(wave.astype(np.float64) * 32768.0).astype(np.int16)
                if not np.array_equal((reconstructed_pcm16.astype(np.float64) / 32768.0).astype(np.float32), wave):
                    raise ConsumerError(f"RAW audio wave is not the exact decoded PCM16 waveform: {key}")
                expected_units = reconstructed_pcm16.astype(np.float64)
            else:
                expected_units = np.multiply(wave.astype(np.float64), np.float64(32768.0), dtype=np.float64)
            if not np.array_equal(units, expected_units):
                raise ConsumerError(f"SyncNet integer-amplitude units do not reconstruct from exact audio PCM: {key}/{domain}")
            audio_meta = audio_metadata.get(domain, {})
            if (
                audio_meta.get("float32_pcm_sha256") != _array_sha(wave)
                or audio_meta.get("syncnet_units_sha256") != _array_sha(units)
                or audio_meta.get("syncnet_units_dtype") != str(units.dtype)
                or int(audio_meta.get("sample_rate_hz", -1)) != 16000
                or int(audio_meta.get("sample_count", -1)) != len(wave)
            ):
                raise ConsumerError(f"scoring audio metadata hashes/clocks differ from its arrays: {key}/{domain}")
        if any(len(context_arrays[f"{cell}_starts"]) != len(context_arrays[f"{cell}_z"]) for cell in EXPECTED_CELLS):
            raise ConsumerError(f"mel starts and encoder latent clocks differ: {key}")
        if any(not np.array_equal(starts0, np.asarray(context_arrays[f"{cell}_starts"], dtype=np.int32)) for cell in EXPECTED_CELLS):
            raise ConsumerError(f"all five cells must share the exact mel start clock: {key}")
        for cell in EXPECTED_CELLS:
            z = np.asarray(context_arrays[f"{cell}_z"])
            visual_values = np.asarray(context_arrays[f"{cell}_V"])
            if z.dtype != np.float32 or z.ndim != 2 or not np.isfinite(z).all() or visual_values.dtype != np.float32 or visual_values.ndim != 2 or not np.isfinite(visual_values).all():
                raise ConsumerError(f"cell latent/V arrays have invalid dtype, dimensions or finite values: {key}/{cell}")
            mel = np.asarray(context_arrays[f"{cell}_mel"])
            chunks = np.asarray(context_arrays[f"{cell}_chunks"])
            starts = np.asarray(context_arrays[f"{cell}_starts"])
            if mel.dtype != np.float32 or mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
                raise ConsumerError(f"cell mel dimensions/dtype/finite values differ from frozen Wav2Lip frontend: {key}/{cell}")
            expected_starts = _official_mel_starts(int(mel.shape[1]))
            if starts.dtype != np.int32 or chunks.dtype != np.float32 or chunks.shape != (len(expected_starts), 80, 16):
                raise ConsumerError(f"cell mel/chunk dimensions differ from frozen Wav2Lip frontend: {key}/{cell}")
            if not np.array_equal(starts, expected_starts):
                raise ConsumerError(f"cell mel start indices differ from official 80Hz/25fps loop: {key}/{cell}")
            expected_chunks = np.stack([mel[:, int(start):int(start) + 16] for start in expected_starts], axis=0)
            if not np.array_equal(chunks, expected_chunks):
                raise ConsumerError(f"cell Wav2Lip chunks differ from its saved mel/start arrays: {key}/{cell}")
        if any(context_arrays[f"{cell}_V"].shape[0] != context_arrays[f"{cell}_z"].shape[0] - 5 for cell in EXPECTED_CELLS):
            raise ConsumerError(f"five-frame encoder→SyncNet V clock must be N−5: {key}")
        common = min(*visual_counts.values(), *audio_counts.values())
        if context_meta.get("common_query_count") != common or common < 1:
            raise ConsumerError(f"common prefix count is not min(full V, full A): {key}")
        supports_expected = _expected_support_dict(common)
        for support, expected_indices in supports_expected.items():
            if not np.array_equal(np.asarray(context_arrays[f"support_{support}"], dtype=np.int32), expected_indices):
                raise ConsumerError(f"context {support} support array differs from frozen rule: {key}")
            if not np.array_equal(np.asarray(context_meta.get("supports", {}).get(support, []), dtype=np.int32), expected_indices):
                raise ConsumerError(f"context JSON {support} support differs from frozen rule: {key}")
        for cell in EXPECTED_CELLS:
            for domain in EXPECTED_AUDIO:
                row_id = f"{key[0]}__{key[1]}__{cell}__{domain}"
                planned, result = row_by_id[row_id], result_by_id[row_id]
                for path_field in ("score_npz",):
                    relative = str(planned[path_field])
                    path = _path_under(run_root, relative, path_field)
                    wanted = result.get(f"{path_field}_sha256")
                    if result.get(path_field) != relative or not wanted or _sha(path) != wanted or run_inventory.get(relative) != wanted:
                        raise ConsumerError(f"{path_field} SHA differs from result ledger: {row_id}")
                plan_visual_rel = str(planned.get("visual_npz", ""))
                expected_plan_visual_rel = f"visuals/{planned['context_key']}/{cell}.npz"
                if plan_visual_rel != expected_plan_visual_rel or "video_path" in planned:
                    raise ConsumerError(f"frozen visual/video plan descriptors differ from the recorded producer schema: {row_id}")
                visual_meta = context_meta.get("cell_metadata", {}).get(cell, {}).get("visual", {})
                actual_visual_rel = str(result.get("visual_npz", ""))
                expected_actual_visual_rel = f"visuals/{key[0]}/{key[1]}/{cell}.npz"
                actual_visual_sha = str(result.get("visual_npz_sha256", ""))
                actual_video_rel = str(result.get("video_path", ""))
                actual_video_sha = str(result.get("video_sha256", ""))
                expected_video_rel = f"videos/{key[0]}/{key[1]}/{cell}.mkv"
                if (
                    actual_visual_rel != expected_actual_visual_rel
                    or visual_meta.get("condition") != cell
                    or visual_meta.get("visual_npz") != actual_visual_rel
                    or visual_meta.get("visual_npz_sha256") != actual_visual_sha
                    or actual_video_rel != expected_video_rel
                    or visual_meta.get("video_path") != actual_video_rel
                    or visual_meta.get("video_sha256") != actual_video_sha
                    or not actual_video_sha
                    or run_inventory.get(actual_video_rel) != actual_video_sha
                ):
                    raise ConsumerError(f"actual visual/video refs differ between result, context sidecar, inventory and producer layout: {row_id}")
                _path_under(run_root, actual_video_rel, "video reference")
                visual_path = _path_under(run_root, actual_visual_rel, "visual NPZ")
                if not actual_visual_sha or _check_sha(visual_path, actual_visual_sha, f"visual/{row_id}") != run_inventory.get(actual_visual_rel):
                    raise ConsumerError(f"visual SHA differs from the all-files inventory: {row_id}")
                if result.get("context_npz") != planned.get("context_npz") or result.get("context_npz_sha256") != context_npz_sha:
                    raise ConsumerError(f"result row does not bind its exact context NPZ: {row_id}")
                visual_arrays = _read_npz(visual_path)
                if set(visual_arrays) != {"latent_z_float32", "lip_embeddings_float32"}:
                    raise ConsumerError(f"visual NPZ key set differs from the frozen producer schema: {row_id}")
                if not np.array_equal(visual_arrays["latent_z_float32"], context_arrays[f"{cell}_z"]) or not np.array_equal(visual_arrays["lip_embeddings_float32"], context_arrays[f"{cell}_V"]):
                    raise ConsumerError(f"visual NPZ does not match context latent/V arrays: {row_id}")
                if int(result.get("full_visual_rows", -1)) != visual_counts[cell] or int(result.get("full_audio_rows", -1)) != audio_counts[domain] or int(result.get("common_query_count", -1)) != common:
                    raise ConsumerError(f"row counts differ from full-domain context arrays: {row_id}")
        endpoint_bridge_reconstruction = _verify_endpoint_bridges(
            run_root, context, context_meta, context_arrays, row_by_id, result_by_id, run_inventory,
        )
        contexts.append({
            "context_index": context_index, "context_key": str(context_meta["key"]), "sample_id": key[0],
            "speaker_id": str(context_meta["speaker_id"]), "direction": key[1], "ordered_eval_index": int(context_meta["ordered_eval_index"]),
            "context_npz": context_path, "context_npz_sha256": context_npz_sha, "context_json": context_json,
            "context_meta": context_meta, "arrays": context_arrays, "phone_map": phone_map,
            "phone_map_path": phone_path, "phone_map_sha256": str(phone_ref["file_sha256"]),
            "query_count": common, "visual_counts": visual_counts, "audio_counts": audio_counts,
        })
        metadata_checks.append({
            "context_key": str(context_meta["key"]), "context_npz_sha256": context_npz_sha,
            "query_count": common, "visual_counts": visual_counts, "audio_counts": audio_counts,
            "support_n": {name: int(len(indices)) for name, indices in supports_expected.items()},
            "endpoint_bridge_reconstruction": endpoint_bridge_reconstruction,
        })

    if len(contexts) != EXPECTED_CONTEXTS or len({ctx["sample_id"] for ctx in contexts}) != EXPECTED_PAIRS or len(speakers) != EXPECTED_SPEAKERS:
        raise ConsumerError("all 160 contexts, 80 utterances and 40 speakers are required")
    return {
        "run_dir": run_root, "output_dir": out, "protocol": protocol, "plan": plan,
        "run_files": observed_run_files, "run_status": status, "code_seal": code_seal,
        "baseline_binding_check": baseline_binding_check,
        "inventory_check": inventory_check, "runtime_path": runtime_path, "runtime": runtime,
        "runtime_flags_sha256": args.runtime_flags_sha256,
        "plan_path": plan_path, "plan_file_sha256": plan_raw,
        "physical_manifest_sha256": args.shape_manifest_sha256,
        "design_path": design_path, "design_sha256": design_sha,
        "metric_addendum_path": addendum_path, "metric_addendum_sha256": addendum_sha,
        "phone_inventory_path": phone_inventory_path, "phone_inventory_sha256": phone_inventory_sha,
        "phone_inventory": phone_inventory,
        "physical_manifest": shape_manifest, "physical_manifest_path": shape_path,
        "contexts": contexts, "rows": plan_rows, "results": results,
        "row_by_id": row_by_id, "result_by_id": result_by_id,
        "speakers": speakers, "input_seals": checked_inputs,
        "metadata_preflight": metadata_checks,
    }


METRIC_ARRAY_NAMES = (*METRICS, "C_best", "D_fixed_anchor")


def _family_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for direction in EXPECTED_DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for metric in METRICS:
                        for audio_domain in EXPECTED_AUDIO:
                            for estimand in ("observed", "time_excess"):
                                key = ("global", contrast, direction, geometry, support, metric, audio_domain, estimand)
                                entries.append({
                                    "key": key, "family": "global", "contrast": contrast,
                                    "direction": direction, "geometry": geometry, "support": support,
                                    "metric": metric, "audio_domain": audio_domain, "estimand": estimand,
                                })
    for treatment in LOCAL_TREATMENTS:
        for direction in EXPECTED_DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for label in PHONE_LABELS:
                        for outcome in PHONE_OUTCOMES:
                            for estimand in LOCAL_ESTIMANDS:
                                key = ("local", treatment, direction, geometry, support, label, outcome, estimand)
                                entries.append({
                                    "key": key, "family": "local_phone", "treatment": treatment,
                                    "direction": direction, "geometry": geometry, "support": support,
                                    "phone": label, "phase": outcome, "estimand": estimand,
                                })
    for treatment in LOCAL_TREATMENTS:
        for direction in EXPECTED_DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for estimand in LOCAL_ESTIMANDS:
                        key = ("overall", treatment, direction, geometry, support, estimand)
                        entries.append({
                            "key": key, "family": "local_overall", "treatment": treatment,
                            "direction": direction, "geometry": geometry, "support": support,
                            "estimand": estimand,
                        })
    if len(entries) != 11_776 or len({entry["key"] for entry in entries}) != 11_776:
        raise ConsumerError(f"planned endpoint family cardinality is {len(entries)}, expected 11,776")
    return entries


def _score_path_arrays(sealed: dict[str, Any], row: dict[str, Any], expected_sha256: str) -> dict[str, np.ndarray]:
    path = _path_under(sealed["run_dir"], str(row["score_npz"]), "score NPZ")
    arrays = _read_npz(path)
    expected = {
        "cross_distance_raw_float32", "cross_distance_to_zero_raw_float32", "lag_distance_raw_float32",
        "cross_distance_unit_float32", "cross_distance_to_zero_unit_float32", "lag_distance_unit_float32",
        "common_query_prefix_indices_int32", "lag_columns_int32", "lag_k_int32",
        *(f"support_{name}_indices_int32" for name in SUPPORTS),
    }
    if set(arrays) != expected:
        raise ConsumerError(f"score NPZ key set differs from producer schema: {row['row_id']}")
    if _sha(path) != expected_sha256:
        raise ConsumerError(f"score NPZ hash changed after preflight: {row['row_id']}")
    return arrays


def _verify_score_and_metrics(
    context: dict[str, Any],
    cell: str,
    domain: str,
    planned: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    arrays = context["arrays"]
    score = _score_path_arrays(context["_sealed"], planned, str(result["score_npz_sha256"]))
    visual = np.asarray(arrays[f"{cell}_V"], dtype=np.float32)
    audio = np.asarray(arrays[f"{domain}_emb"], dtype=np.float32)
    query_n = int(context["query_count"])
    if score["cross_distance_raw_float32"].shape != (len(visual), len(audio)):
        raise ConsumerError(f"stored raw matrix is not full-V×full-A: {planned['row_id']}")
    if score["cross_distance_unit_float32"].shape != (len(visual), len(audio)):
        raise ConsumerError(f"stored unit matrix is not full-V×full-A: {planned['row_id']}")
    raw_cross, raw_zero = _cross_distance(visual, audio)
    unit_cross, unit_zero = _cross_distance(_row_unit(visual), _row_unit(audio))
    checks = (
        ("cross_distance_raw_float32", raw_cross),
        ("cross_distance_to_zero_raw_float32", raw_zero),
        ("cross_distance_unit_float32", unit_cross),
        ("cross_distance_to_zero_unit_float32", unit_zero),
    )
    for name, independent in checks:
        stored = np.asarray(score[name], dtype=np.float32)
        if stored.shape != independent.shape or not np.allclose(stored, independent, rtol=CROSS_RTOL, atol=CROSS_ATOL):
            error = float(np.max(np.abs(stored.astype(np.float64) - independent.astype(np.float64)), initial=0.0))
            raise ConsumerError(f"cached SyncNet {name} parity failed for {planned['row_id']}: max_abs={error}")
    if not np.array_equal(score["common_query_prefix_indices_int32"], np.arange(query_n, dtype=np.int32)):
        raise ConsumerError(f"cached common query prefix is not the exact contiguous prefix: {planned['row_id']}")
    if not np.array_equal(score["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(score["lag_k_int32"], np.arange(-15, 16, dtype=np.int32)):
        raise ConsumerError(f"cached lag column/offset axes differ: {planned['row_id']}")
    supports = {name: np.asarray(arrays[f"support_{name}"], dtype=np.int32) for name in SUPPORTS}
    for name, indices in supports.items():
        if not np.array_equal(score[f"support_{name}_indices_int32"], indices):
            raise ConsumerError(f"score/context {name} support arrays differ: {planned['row_id']}")
    lag_by_geometry = {
        "raw": _lag_from_cross(raw_cross, raw_zero, query_n, len(audio)),
        "unit": _lag_from_cross(unit_cross, unit_zero, query_n, len(audio)),
    }
    for geometry, expected_lags in lag_by_geometry.items():
        stored = np.asarray(score[f"lag_distance_{geometry}_float32"], dtype=np.float32)
        if stored.shape != expected_lags.shape or not np.allclose(stored, expected_lags, rtol=CROSS_RTOL, atol=CROSS_ATOL):
            error = float(np.max(np.abs(stored.astype(np.float64) - expected_lags.astype(np.float64)), initial=0.0))
            raise ConsumerError(f"cached SyncNet {geometry} lag reconstruction failed for {planned['row_id']}: max_abs={error}")
    recorded_by_geometry = {
        "raw": result.get("raw_support_metrics", {}),
        "unit": result.get("unit_support_metrics", {}),
    }
    metrics: dict[str, dict[str, np.ndarray]] = {}
    lags_by_geometry = lag_by_geometry
    for geometry, lags in lags_by_geometry.items():
        recorded_supports = recorded_by_geometry[geometry]
        if set(recorded_supports) != set(SUPPORTS):
            raise ConsumerError(f"result row lacks the frozen four support metrics: {planned['row_id']}")
        metrics[geometry] = {}
        for support, indices in supports.items():
            observed = _metrics(lags, indices)
            recorded = recorded_supports[support]
            _compare_metric_record(observed, recorded, f"{planned['row_id']}/{geometry}/{support}")
            if observed is None:
                if not np.array_equal(np.asarray(recorded.get("window_indices", []), dtype=np.int32), indices):
                    raise ConsumerError(f"stored empty support indices differ: {planned['row_id']}/{support}")
                metrics[geometry][support] = {}
            else:
                if not np.array_equal(np.asarray(recorded.get("window_indices", []), dtype=np.int32), indices):
                    raise ConsumerError(f"stored {support} window indices differ: {planned['row_id']}")
                metrics[geometry][support] = {
                    name: np.asarray(observed[name], dtype=np.float64)
                    for name in METRIC_ARRAY_NAMES
                }
    return metrics, {
        "raw": raw_cross, "raw_zero": raw_zero,
        "unit": unit_cross, "unit_zero": unit_zero,
    }


def _make_permutations(audio_n: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    bank = np.stack([rng.permutation(audio_n) for _ in range(NULL_DRAWS)], axis=0).astype(np.int32)
    expected = np.arange(audio_n, dtype=np.int32)
    if not np.all(np.sort(bank, axis=1) == expected[None, :]):
        raise ConsumerError("constructed time-null bank does not preserve the full A multiset")
    return bank


def _token_counts_by_label(
    intervals: list[dict[str, Any]], centers: np.ndarray, owner_count: np.ndarray,
    owner_index: np.ndarray, unions: list[np.ndarray], supports: dict[str, np.ndarray],
) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for support, q in supports.items():
        selected_cols = np.unique(np.concatenate([unions[int(index)] for index in q])) if len(q) else np.empty(0, dtype=np.int64)
        counts = {label: [0, 0, 0, 0] for label in PHONE_LABELS}
        for token in intervals:
            if token["label"] not in counts:
                continue
            member = (centers[selected_cols] >= token["start"]) & (centers[selected_cols] < token["end"])
            member &= (owner_count[selected_cols] == 1) & (owner_index[selected_cols] == token["phone_index"])
            if np.any(member):
                phase = (centers[selected_cols] - token["start"]) / (token["end"] - token["start"])
                phases = (
                    member, member & (phase < 1 / 3),
                    member & (phase >= 1 / 3) & (phase < 2 / 3),
                    member & (phase >= 2 / 3) & (phase < 1),
                )
                for phase_idx, phase_mask in enumerate(phases):
                    counts[token["label"]][phase_idx] += int(np.any(phase_mask))
        result[support] = counts
    return result


def _speaker_bootstrap(
    family_values: np.ndarray,
    family_mask: np.ndarray,
    endpoint_metadata: list[dict[str, Any]],
    speaker_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, Any]]]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, EXPECTED_SPEAKERS, size=(BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.int16)
    counts = np.zeros((BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.uint8)
    for draw in range(BOOTSTRAP_DRAWS):
        counts[draw] = np.bincount(indices[draw], minlength=EXPECTED_SPEAKERS).astype(np.uint8)
    counts64 = counts.astype(np.float64)
    point = np.full(len(endpoint_metadata), np.nan, dtype=np.float64)
    se = np.full(len(endpoint_metadata), np.nan, dtype=np.float64)
    estimable = np.asarray(family_mask, dtype=bool)
    for idx in np.flatnonzero(estimable):
        values = np.asarray(family_values[idx], dtype=np.float64)
        if values.shape != (EXPECTED_SPEAKERS,) or not np.isfinite(values).all():
            raise ConsumerError(f"endpoint mask marks nonfinite speaker values estimable: {idx}")
        point[idx] = float(np.mean(values, dtype=np.float64))
        boot = counts64 @ values / float(EXPECTED_SPEAKERS)
        se[idx] = float(np.std(boot, ddof=1, dtype=np.float64))
    estimable_indices = np.flatnonzero(estimable)
    if not len(estimable_indices):
        raise ConsumerError("no family endpoint is estimable")
    max_t = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, len(estimable_indices), 96):
        block = estimable_indices[start : start + 96]
        values = np.asarray(family_values[block], dtype=np.float64)
        boot = (counts64 @ values.T) / float(EXPECTED_SPEAKERS)
        errors = se[block]
        centered = np.abs(boot - point[block][None, :])
        tvals = np.divide(centered, errors[None, :], out=np.zeros_like(centered), where=errors[None, :] > 0)
        max_t = np.maximum(max_t, np.max(tvals, axis=1))
    critical = float(np.quantile(max_t, 0.99, method="linear"))
    for idx, item in enumerate(endpoint_metadata):
        if estimable[idx]:
            item.update({
                "status": "estimable",
                "estimate_speaker_equal_mean": float(point[idx]),
                "bootstrap_se": float(se[idx]),
                "ci99_simultaneous_max_t": [
                    float(point[idx] - critical * se[idx]),
                    float(point[idx] + critical * se[idx]),
                ],
                "family_critical_99_max_t": critical,
            })
        else:
            item.update({
                "status": "INCOMPLETE_SUPPORT",
                "estimate_speaker_equal_mean": None,
                "bootstrap_se": None,
                "ci99_simultaneous_max_t": None,
                "family_critical_99_max_t": critical,
            })
    return indices, counts, max_t, critical, endpoint_metadata


def analyze(sealed: dict[str, Any]) -> dict[str, Any]:
    out: Path = sealed["output_dir"]
    out.mkdir(parents=True, exist_ok=False)
    rf_dir = out / "receptive_fields"
    null_dir = out / "permutations"
    rf_dir.mkdir()
    null_dir.mkdir()
    entries = _family_entries()
    entry_index = {entry["key"]: index for index, entry in enumerate(entries)}
    sample_order = list(dict.fromkeys(context["sample_id"] for context in sealed["contexts"]))
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_order)}
    speaker_order = sealed["speakers"]
    speaker_index = {speaker: index for index, speaker in enumerate(speaker_order)}
    if len(sample_order) != EXPECTED_PAIRS:
        raise ConsumerError("ordered context plan does not contain exactly 80 distinct clips")
    for context in sealed["contexts"]:
        context["_sealed"] = sealed

    clip_values = np.full((len(entries), 2, EXPECTED_PAIRS), np.nan, dtype=np.float64)
    exposure_details: dict[str, dict[str, Any]] = {}
    global_context_rows = []
    rf_context_rows = []
    score_rows_verified = 0
    context_started = time.time()
    for context in sealed["contexts"]:
        key = (context["sample_id"], context["direction"])
        sample_id = context["sample_id"]
        direction = context["direction"]
        pair_idx = sample_index[sample_id]
        domain_seed_bases = {"A_RAW": RAW_NULL_SEED, "A_RT": RT_NULL_SEED}
        permutations: dict[str, np.ndarray] = {}
        permutation_records: dict[str, dict[str, Any]] = {}
        for domain in EXPECTED_AUDIO:
            audio_n = int(context["audio_counts"][domain])
            seed = domain_seed_bases[domain] + int(context["ordered_eval_index"])
            bank = _make_permutations(audio_n, seed)
            permutations[domain] = bank
            permutation_path = null_dir / f"{sample_id}__{direction}__{domain}.npz"
            np.savez_compressed(permutation_path, permutation_indices_int32=bank)
            permutation_records[domain] = {
                "path": str(permutation_path.relative_to(out)).replace("\\", "/"),
                "sha256": _sha(permutation_path), "array_sha256": _array_sha(bank),
                "seed": seed, "shape": list(bank.shape), "audio_population_n": audio_n,
            }

        query_n = int(context["query_count"])
        supports = {name: np.asarray(context["arrays"][f"support_{name}"], dtype=np.int32) for name in SUPPORTS}
        n_cells, n_domains, n_geometries, n_supports, n_metrics = (
            len(EXPECTED_CELLS), len(EXPECTED_AUDIO), len(GEOMETRIES), len(SUPPORTS), len(METRIC_ARRAY_NAMES)
        )
        global_obs = np.full((n_cells, n_domains, n_geometries, n_supports, n_metrics), np.nan, dtype=np.float64)
        global_null = np.full_like(global_obs, np.nan)
        global_null_draws = np.full((n_cells, n_domains, n_geometries, n_supports, n_metrics, NULL_DRAWS), np.nan, dtype=np.float64)
        cross_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for cell_idx, cell in enumerate(EXPECTED_CELLS):
            for domain_idx, domain in enumerate(EXPECTED_AUDIO):
                row_id = f"{sample_id}__{direction}__{cell}__{domain}"
                planned = sealed["row_by_id"][row_id]
                result = sealed["result_by_id"][row_id]
                metrics, crosses = _verify_score_and_metrics(context, cell, domain, planned, result)
                score_rows_verified += 1
                cross_cache[(cell, domain, "raw")] = (crosses["raw"], crosses["raw_zero"])
                cross_cache[(cell, domain, "unit")] = (crosses["unit"], crosses["unit_zero"])
                for geometry_idx, geometry in enumerate(GEOMETRIES):
                    cross, zero = cross_cache[(cell, domain, geometry)]
                    null_lags = _permuted_lags(
                        cross, zero, permutations[domain], query_n,
                        int(context["audio_counts"][domain]),
                    )
                    if null_lags.shape != (NULL_DRAWS, query_n, 31):
                        raise ConsumerError(f"full-A null lag bank has unexpected shape: {row_id}/{geometry}")
                    for support_idx, support in enumerate(SUPPORTS):
                        ix = supports[support]
                        if not len(ix):
                            continue
                        observed = metrics[geometry][support]
                        batch = _metrics_batch(null_lags, ix)
                        if observed is None or batch is None:
                            raise ConsumerError(f"nonempty support failed to produce global metrics: {row_id}/{support}")
                        for metric_idx, metric in enumerate(METRIC_ARRAY_NAMES):
                            if metric not in observed or metric not in batch:
                                raise ConsumerError(f"independent metric reconstruction omitted {metric}: {row_id}")
                            draws = np.asarray(batch[metric], dtype=np.float64)
                            global_obs[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx] = float(observed[metric])
                            global_null[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx] = float(np.mean(draws, dtype=np.float64))
                            global_null_draws[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx] = draws

        # Inferential global contrast family; the same null bank is shared across
        # the five generated V cells for each fixed A domain.
        for contrast in CONTRASTS:
            for geometry_idx, geometry in enumerate(GEOMETRIES):
                for support_idx, support in enumerate(SUPPORTS):
                    for metric_idx, metric in enumerate(METRICS):
                        for domain_idx, domain in enumerate(EXPECTED_AUDIO):
                            observed_cells = {
                                cell: float(global_obs[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx])
                                for cell_idx, cell in enumerate(EXPECTED_CELLS)
                            }
                            null_cells = {
                                cell: global_null_draws[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx]
                                for cell_idx, cell in enumerate(EXPECTED_CELLS)
                            }
                            if not all(math.isfinite(value) for value in observed_cells.values()):
                                continue
                            observed_value = _contrast(observed_cells, contrast)
                            null_contrast = _contrast_batch(null_cells, contrast)
                            null_mean = float(np.mean(null_contrast, dtype=np.float64))
                            for estimand, value in (("observed", observed_value), ("time_excess", observed_value - null_mean)):
                                entry_key = ("global", contrast, direction, geometry, support, metric, domain, estimand)
                                clip_values[entry_index[entry_key], 0, pair_idx] = value
                                clip_values[entry_index[entry_key], 1, pair_idx] = null_mean
        # Descriptive C_best and D_fixed_anchor are outside the simultaneous
        # family but retained for every cell/support/domain/geometry and null.
        diagnostic_rows = []
        for cell_idx, cell in enumerate(EXPECTED_CELLS):
            for domain_idx, domain in enumerate(EXPECTED_AUDIO):
                for geometry_idx, geometry in enumerate(GEOMETRIES):
                    for support_idx, support in enumerate(SUPPORTS):
                        for metric in ("C_best", "D_fixed_anchor"):
                            metric_idx = METRIC_ARRAY_NAMES.index(metric)
                            value = global_obs[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx]
                            null_value = global_null[cell_idx, domain_idx, geometry_idx, support_idx, metric_idx]
                            diagnostic_rows.append({
                                "context_key": context["context_key"], "cell": cell, "audio_domain": domain,
                                "geometry": geometry, "support": support, "metric": metric,
                                "observed": float(value) if np.isfinite(value) else None,
                                "time_null_mean": float(null_value) if np.isfinite(null_value) else None,
                                "status": "ok" if np.isfinite(value) else "INCOMPLETE_SUPPORT",
                            })
        global_npz = out / "global_context" / f"{sample_id}__{direction}.npz"
        global_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            global_npz, cell_names_unicode=np.asarray(EXPECTED_CELLS, dtype="U16"),
            audio_names_unicode=np.asarray(EXPECTED_AUDIO, dtype="U8"),
            geometry_names_unicode=np.asarray(GEOMETRIES, dtype="U8"),
            support_names_unicode=np.asarray(SUPPORTS, dtype="U8"),
            metric_names_unicode=np.asarray(METRIC_ARRAY_NAMES, dtype="U32"),
            global_observed_metrics_float64=global_obs,
            global_null_metric_draws_float64=global_null_draws,
            global_null_metric_mean_float64=global_null,
        )
        global_context_rows.append({
            "sample_id": sample_id, "speaker_id": context["speaker_id"], "direction": direction,
            "ordered_eval_index": context["ordered_eval_index"], "context_npz_sha256": context["context_npz_sha256"],
            "global_metrics_path": str(global_npz.relative_to(out)).replace("\\", "/"),
            "global_metrics_sha256": _sha(global_npz), "permutation_banks": permutation_records,
            "descriptive_metrics": diagnostic_rows,
        })

        # Exact Wav2Lip mel-coordinate receptive fields and descriptive local G.
        mel_n = int(np.asarray(context["arrays"][f"{EXPECTED_CELLS[0]}_mel"]).shape[-1])
        starts = np.asarray(context["arrays"][f"{EXPECTED_CELLS[0]}_starts"], dtype=np.int32)
        unions, indptr, flat_columns = _window_unions(starts, query_n, mel_n)
        intervals = _phone_intervals(context["phone_map"])
        weights, stft_weights, owner_count, owner_index, centers = _phone_weights(unions, mel_n, intervals)
        boundary = _boundary_diagnostics(unions, centers, intervals)
        token_counts = _token_counts_by_label(intervals, centers, owner_count, owner_index, unions, supports)
        g_observed = np.zeros((len(LOCAL_TREATMENTS), len(GEOMETRIES), query_n), dtype=np.float64)
        g_null_mean_window = np.zeros_like(g_observed)
        local_stats = np.full((len(LOCAL_TREATMENTS), len(GEOMETRIES), len(SUPPORTS), len(PHONE_LABELS), len(PHONE_OUTCOMES), len(LOCAL_STAT_NAMES)), np.nan, dtype=np.float64)
        overall_stats = np.full((len(LOCAL_TREATMENTS), len(GEOMETRIES), len(SUPPORTS), len(OVERALL_STAT_NAMES)), np.nan, dtype=np.float64)
        q = np.arange(query_n, dtype=np.int64)
        baseline_cache = {}
        for geometry_idx, geometry in enumerate(GEOMETRIES):
            base_cross, base_zero = cross_cache[("q00", "A_RAW", geometry)]
            baseline_cache[geometry] = _anchor_values(base_cross, base_zero, q, int(context["audio_counts"]["A_RAW"]))
            baseline_null = _permuted_anchor(base_cross, base_zero, q, permutations["A_RAW"], int(context["audio_counts"]["A_RAW"]))
            for treatment_idx, treatment in enumerate(LOCAL_TREATMENTS):
                treatment_cross, treatment_zero = cross_cache[(treatment, "A_RAW", geometry)]
                treated_anchor = _anchor_values(treatment_cross, treatment_zero, q, int(context["audio_counts"]["A_RAW"]))
                g = baseline_cache[geometry] - treated_anchor
                treatment_null = _permuted_anchor(treatment_cross, treatment_zero, q, permutations["A_RAW"], int(context["audio_counts"]["A_RAW"]))
                g_null = baseline_null - treatment_null
                g_observed[treatment_idx, geometry_idx] = g
                g_null_mean_window[treatment_idx, geometry_idx] = np.mean(g_null, axis=0, dtype=np.float64)
                for support in SUPPORTS:
                    support_idx = SUPPORTS.index(support)
                    ix = supports[support]
                    support_n = len(ix)
                    if support_n == 0:
                        continue
                    observed_g = g[ix].astype(np.float64)
                    null_g = g_null[:, ix].astype(np.float64)
                    overall_obs = float(np.mean(observed_g, dtype=np.float64))
                    overall_null_draws = np.mean(null_g, axis=1, dtype=np.float64)
                    overall_null_mean = float(np.mean(overall_null_draws, dtype=np.float64))
                    overall_stats[treatment_idx, geometry_idx, support_idx] = (
                        overall_obs, overall_null_mean, overall_obs - overall_null_mean,
                    )
                    overall_key = ("overall", treatment, direction, geometry, support, "observed")
                    clip_values[entry_index[overall_key], 0, pair_idx] = overall_obs
                    clip_values[entry_index[overall_key], 1, pair_idx] = overall_null_mean
                    overall_excess_key = ("overall", treatment, direction, geometry, support, "time_excess")
                    clip_values[entry_index[overall_excess_key], 0, pair_idx] = overall_obs - overall_null_mean
                    clip_values[entry_index[overall_excess_key], 1, pair_idx] = overall_null_mean
                    for phone_idx, label in enumerate(PHONE_LABELS):
                        for phase_idx, phase in enumerate(PHONE_OUTCOMES):
                            weight = np.asarray(weights[phone_idx, phase_idx, ix], dtype=np.float64)
                            denominator = float(support_n)
                            obs_value = float(np.dot(weight, observed_g) / denominator)
                            null_draws = np.sum(null_g * weight[None, :], axis=1, dtype=np.float64) / denominator
                            null_mean = float(np.mean(null_draws, dtype=np.float64))
                            key_obs = ("local", treatment, direction, geometry, support, label, phase, "observed")
                            key_excess = ("local", treatment, direction, geometry, support, label, phase, "time_excess")
                            clip_values[entry_index[key_obs], 0, pair_idx] = obs_value
                            clip_values[entry_index[key_obs], 1, pair_idx] = null_mean
                            clip_values[entry_index[key_excess], 0, pair_idx] = obs_value - null_mean
                            clip_values[entry_index[key_excess], 1, pair_idx] = null_mean
                            exposure_fraction = float(weight.sum(dtype=np.float64) / denominator)
                            exposed_n = int(np.count_nonzero(weight > 0))
                            conditional = float(np.dot(weight, observed_g) / weight.sum(dtype=np.float64)) if weight.sum(dtype=np.float64) > 0 else None
                            token_count = int(token_counts[support][label][phase_idx])
                            stft_weight = np.asarray(stft_weights[phone_idx, phase_idx, ix], dtype=np.float64)
                            stft_obs = float(np.dot(stft_weight, observed_g) / denominator)
                            stft_null = np.mean(
                                np.sum(null_g * stft_weight[None, :], axis=1, dtype=np.float64) / denominator,
                                dtype=np.float64,
                            )
                            local_stats[treatment_idx, geometry_idx, support_idx, phone_idx, phase_idx] = (
                                obs_value, null_mean, obs_value - null_mean,
                                exposure_fraction, float(exposed_n), float(token_count),
                                np.nan if conditional is None else conditional,
                                stft_obs, float(stft_null), stft_obs - float(stft_null),
                                float(stft_weight.sum(dtype=np.float64) / denominator),
                            )
        rf_path = rf_dir / f"{sample_id}__{direction}.npz"
        np.savez_compressed(
            rf_path,
            phone_labels_unicode=np.asarray(PHONE_LABELS, dtype="U8"),
            phase_names_unicode=np.asarray(PHONE_OUTCOMES, dtype="U8"),
            treatment_names_unicode=np.asarray(LOCAL_TREATMENTS, dtype="U16"),
            geometry_names_unicode=np.asarray(GEOMETRIES, dtype="U8"),
            support_names_unicode=np.asarray(SUPPORTS, dtype="U8"),
            mel_centers_seconds_float64=centers,
            matched_phone_owner_count_int16=owner_count,
            matched_phone_owner_index_int32=owner_index,
            window_union_indptr_int32=indptr,
            window_union_mel_columns_int32=flat_columns,
            phone_center_coverage_weights_float64=weights,
            phone_stft50_sensitivity_weights_float64=stft_weights,
            local_g_observed_float64=g_observed,
            local_g_time_null_mean_per_window_float64=g_null_mean_window,
            local_endpoint_stats_float64=local_stats,
            local_endpoint_stat_names_unicode=np.asarray(LOCAL_STAT_NAMES, dtype="U48"),
            overall_endpoint_stats_float64=overall_stats,
            overall_endpoint_stat_names_unicode=np.asarray(OVERALL_STAT_NAMES, dtype="U48"),
            null_permutations_A_RAW_int32=permutations["A_RAW"],
            support_full_int32=supports["full"], support_valid_int32=supports["valid"],
            support_guard20_int32=supports["guard20"], support_guard15_int32=supports["guard15"],
            **{f"boundary_{name}": values for name, values in boundary.items()},
        )
        rf_context_rows.append({
            "sample_id": sample_id, "speaker_id": context["speaker_id"], "direction": direction,
            "ordered_eval_index": context["ordered_eval_index"], "context_npz_sha256": context["context_npz_sha256"],
            "phone_map_path": str(context["phone_map_path"]),
            "phone_map_sha256": context["phone_map_sha256"],
            "rf_npz": str(rf_path.relative_to(out)).replace("\\", "/"), "rf_npz_sha256": _sha(rf_path),
            "mel_center_clock": "column c is assigned center c/80 seconds; this is Wav2Lip mel-coordinate support, not the exact waveform/STFT receptive field",
            "syncnet_query_count": query_n, "mel_frame_count": mel_n,
            "union_column_count": int(len(flat_columns)), "matched_phone_token_count": len(intervals),
            "unowned_or_ambiguous_column_count": int(np.count_nonzero(owner_count != 1)),
            "boundary_diagnostics_full": {name: (values.tolist() if values.size <= 200 else {"sum": int(values.sum()), "mean": float(values.mean())}) for name, values in boundary.items()},
            "permutation_seed": RAW_NULL_SEED + int(context["ordered_eval_index"]),
            "permutation_array_sha256": permutation_records["A_RAW"]["array_sha256"],
            "local_stats_shape": list(local_stats.shape), "local_stat_names": list(LOCAL_STAT_NAMES),
            "overall_stats_shape": list(overall_stats.shape), "overall_stat_names": list(OVERALL_STAT_NAMES),
        })

    for filename, rows in (("global_context_index.jsonl", global_context_rows), ("receptive_field_context_index.jsonl", rf_context_rows)):
        with (out / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    schema = {
        "schema_version": 1,
        "design_sha256": DESIGN_SHA256,
        "metric_addendum_sha256": METRIC_ADDENDUM_SHA256,
        "phone_inventory_sha256": PHONE_INVENTORY_SHA256,
        "planned_family_endpoint_count": len(entries),
        "planned_global_endpoint_count": 3456,
        "planned_local_phone_endpoint_count": 8192,
        "planned_local_overall_endpoint_count": 128,
        "direction_order": list(EXPECTED_DIRECTIONS),
        "cell_order": list(EXPECTED_CELLS),
        "audio_domains": list(EXPECTED_AUDIO),
        "supports": list(SUPPORTS),
        "geometries": list(GEOMETRIES),
        "metrics": list(METRICS),
        "contrasts": list(CONTRASTS),
        "local_treatments": list(LOCAL_TREATMENTS),
        "phone_labels_cal_only": list(PHONE_LABELS),
        "phone_phases": list(PHONE_OUTCOMES),
        "local_stat_fields": list(LOCAL_STAT_NAMES),
        "overall_stat_fields": list(OVERALL_STAT_NAMES),
        "rf_definition": "Union of 16 mel columns for each of five consecutive decoded video frames; ownership uses each mel column center c/80 seconds. This is Wav2Lip mel-coordinate support, not an exact waveform receptive field.",
        "stft50_sensitivity": "Centered 50 ms window around each selected mel center, phase and phone ownership integrated on the exact recipient clock; descriptive sensitivity only.",
        "empty_support_policy": "Keep planned endpoint and full 80-clip/40-speaker denominator; mark it INCOMPLETE_SUPPORT, never substitute guard15 or remove a clip.",
        "time_null": {
            "draws": NULL_DRAWS, "generator": "PCG64", "A_RAW_seed_base": RAW_NULL_SEED,
            "A_RT_seed_base": RT_NULL_SEED, "seed_rule": "seed_base + ordered_eval_index",
            "population": "Each full A row domain; one shared permutation bank across all five generated V cells and every support/geometry/metric/contrast.",
        },
        "speaker_bootstrap": {
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "generator": "PCG64",
            "resample_index_dtype": "int16", "speaker_count_dtype": "uint8",
            "unit": "40 frozen speakers, each represented by the equal mean of its two clip values.",
            "interval": "One 99% studentized max-|t| simultaneous interval across estimable planned endpoints.",
        },
    }
    (out / "analysis_schema.json").write_text(json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Speaker values are averages of each speaker's two clip-level values,
    # preserving the frozen 80-clip/40-speaker denominator per direction.
    family_values = np.full((len(entries), EXPECTED_SPEAKERS), np.nan, dtype=np.float64)
    family_mask = np.zeros(len(entries), dtype=np.uint8)
    sample_speakers: dict[str, str] = {}
    for context in sealed["contexts"]:
        sample_speakers[context["sample_id"]] = context["speaker_id"]
    if len(sample_speakers) != EXPECTED_PAIRS:
        raise ConsumerError("sample-to-speaker mapping is not one-to-one")
    for entry_idx, entry in enumerate(entries):
        values = clip_values[entry_idx, 0]
        if np.isfinite(values).all():
            per_speaker = np.full(EXPECTED_SPEAKERS, np.nan, dtype=np.float64)
            for speaker, sidx in speaker_index.items():
                ids = [sample for sample in sample_order if sample_speakers[sample] == speaker]
                if len(ids) != 2:
                    raise ConsumerError(f"speaker {speaker} lacks exactly two frozen utterances")
                per_speaker[sidx] = float(np.mean([values[sample_index[sample]] for sample in ids], dtype=np.float64))
            if np.isfinite(per_speaker).all():
                family_values[entry_idx] = per_speaker
                family_mask[entry_idx] = 1
    final_metadata = [
        {
            **{key_name: value for key_name, value in entry.items() if key_name != "key"},
            "endpoint_key": json.dumps(entry["key"], ensure_ascii=False, separators=(",", ":")),
            "planned_family_index": index,
        }
        for index, entry in enumerate(entries)
    ]
    bootstrap_indices, bootstrap_counts, max_t, critical, final_metadata = _speaker_bootstrap(
        family_values, family_mask, final_metadata, speaker_order,
    )
    bootstrap_path = out / "family_reconstruction.npz"
    np.savez_compressed(
        bootstrap_path,
        speaker_order_unicode=np.asarray(speaker_order, dtype="U16"),
        endpoint_key_unicode=np.asarray([row["endpoint_key"] for row in final_metadata], dtype="U512"),
        estimable_mask_uint8=family_mask,
        speaker_values_float64=family_values,
        clip_values_by_endpoint_estimand_float64=clip_values[:, 0, :],
        clip_time_null_means_float64=clip_values[:, 1, :],
        bootstrap_indices_int16=bootstrap_indices,
        bootstrap_counts_uint8=bootstrap_counts,
        family_max_t_draws_float64=max_t,
    )
    endpoint_path = out / "family_endpoints.jsonl"
    with endpoint_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in final_metadata:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    inventory_rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "independent_output_manifest.json":
            inventory_rows.append({
                "path": str(path.relative_to(out)).replace("\\", "/"),
                "sha256": _sha(path), "bytes": path.stat().st_size,
            })
    manifest = {
        "schema_version": 1,
        "status": "ANALYSIS_COMPLETE",
        "family_support_status": "COMPLETE" if int(family_mask.sum()) == len(entries) else "INCOMPLETE_SUPPORT",
        "design_sha256": DESIGN_SHA256,
        "metric_addendum_sha256": METRIC_ADDENDUM_SHA256,
        "phone_inventory_sha256": PHONE_INVENTORY_SHA256,
        "consumer_source_path": str(Path(__file__).resolve()),
        "consumer_source_sha256": _sha(Path(__file__).resolve()),
        "producer_run_dir": str(sealed["run_dir"]),
        "execution_plan_path": str(sealed["plan_path"]),
        "execution_plan_sha256": sealed["plan_file_sha256"],
        "baseline_execution_plan_raw_sha256": sealed["baseline_binding_check"]["execution_plan_raw_sha256"],
        "baseline_execution_plan_payload_sha256": sealed["baseline_binding_check"]["execution_plan_payload_sha256"],
        "baseline_v6_file_hashes": sealed["baseline_binding_check"]["verified_files"],
        "physical_manifest_path": str(sealed["physical_manifest_path"]),
        "physical_manifest_sha256": sealed["physical_manifest_sha256"],
        "protocol_raw_sha256": sealed["run_files"]["evaluation_protocol.json"],
        "rows_raw_sha256": sealed["run_files"]["evaluation_rows.jsonl"],
        "results_raw_sha256": sealed["run_files"]["evaluation_results.jsonl"],
        "all_files_raw_sha256": sealed["run_files"]["all_files_sha256.json"],
        "runtime_flags_raw_sha256": sealed["runtime"]["runtime_flags_sha256"],
        "inventory_check": sealed["inventory_check"],
        "runtime_forward_call_count": int(sealed["runtime"]["nn_module_forward_call_count"]),
        "score_rows_independently_rebuilt": score_rows_verified,
        "context_count": len(sealed["contexts"]),
        "unique_clip_count": len(sample_order),
        "speaker_count": len(speaker_order),
        "planned_family_endpoint_count": len(entries),
        "estimable_family_endpoint_count": int(family_mask.sum()),
        "unestimable_family_endpoint_count": int((family_mask == 0).sum()),
        "family_critical_99_max_t": critical,
        "family_mask_sha256": _array_sha(family_mask),
        "speaker_value_matrix_shape": list(family_values.shape),
        "bootstrap_indices_shape": list(bootstrap_indices.shape),
        "bootstrap_indices_dtype": str(bootstrap_indices.dtype),
        "bootstrap_counts_dtype": str(bootstrap_counts.dtype),
        "bootstrap_max_t_draw_count": len(max_t),
        "bootstrap_no_draws_by_endpoint_matrix": True,
        "endpoint_bridge_context_count": len(sealed["metadata_preflight"]),
        "endpoint_bridge_cell_count": sum(
            len(item["endpoint_bridge_reconstruction"]) for item in sealed["metadata_preflight"]
        ),
        "endpoint_bridge_reconstruction": [
            {"context_key": item["context_key"], **item["endpoint_bridge_reconstruction"]}
            for item in sealed["metadata_preflight"]
        ],
        "output_files": inventory_rows,
    }
    manifest_path = out / "independent_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "status": manifest["status"], "family_support_status": manifest["family_support_status"], "output_dir": str(out),
        "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
        "family_endpoint_count": len(entries), "estimable_endpoint_count": int(family_mask.sum()),
        "score_rows_rebuilt": score_rows_verified, "family_critical_99_max_t": critical,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--shape-manifest-sha256", required=True)
    parser.add_argument("--runtime-flags", type=Path, required=True)
    parser.add_argument("--runtime-flags-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--rows-sha256", required=True)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--status-sha256", required=True)
    parser.add_argument("--all-files-sha256", required=True)
    parser.add_argument("--code-seal-sha256", required=True)
    parser.add_argument("--design-file", type=Path, default=root / "openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/eval-design-root-freeze.md")
    parser.add_argument("--metric-addendum", type=Path, default=root / "openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/metric-definitions-root-addendum.md")
    parser.add_argument("--phone-inventory", type=Path, default=root / "openspec/changes/diagnose-world-local-acoustic-response-20261001/cal_phone_inventory.json")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _preflight_summary(sealed: dict[str, Any]) -> dict[str, Any]:
    contexts = sealed["contexts"]
    return {
        "status": "PREFLIGHT_PASS",
        "run_dir": str(sealed["run_dir"]),
        "output_dir": str(sealed["output_dir"]),
        "protocol_sha256": sealed["run_files"]["evaluation_protocol.json"],
        "row_manifest_sha256": sealed["run_files"]["evaluation_rows.jsonl"],
        "results_sha256": sealed["run_files"]["evaluation_results.jsonl"],
        "all_files_sha256": sealed["run_files"]["all_files_sha256.json"],
        "execution_plan_sha256": sealed["plan_file_sha256"],
        "physical_manifest_sha256": sealed["physical_manifest_sha256"],
        "runtime_flags_sha256": sealed["runtime_flags_sha256"],
        "all_files_member_count": sealed["inventory_check"]["member_count"],
        "context_count": len(contexts),
        "score_rows": len(sealed["results"]),
        "clip_count": len({context["sample_id"] for context in contexts}),
        "speaker_count": len(sealed["speakers"]),
        "estimand_family_endpoint_count": len(_family_entries()),
        "endpoint_bridge_context_count": len(sealed["metadata_preflight"]),
        "endpoint_bridge_cell_count": sum(
            len(item["endpoint_bridge_reconstruction"]) for item in sealed["metadata_preflight"]
        ),
        "short_support_context_count": sum(int(context["query_count"]) < 40 for context in contexts),
        "model_forward_call_count": int(sealed["runtime"]["nn_module_forward_call_count"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sealed = preflight(args)
        if args.preflight_only:
            print(json.dumps(_preflight_summary(sealed), ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        result = analyze(sealed)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        # The full planned family remains in the output. Incomplete support is
        # a scientific status, not a consumer execution failure.
        return 0 if result["status"] == "ANALYSIS_COMPLETE" else 2
    except Exception as exc:
        print(json.dumps({
            "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc),
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
