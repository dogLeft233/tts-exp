#!/usr/bin/env python3
"""Independent, outcome-blind reconstruction QA for the sealed CAL factorial.

This consumer intentionally does not import the CAL producer. It rebuilds the
phone-map transfer, token domains, mean/residual shapes, SELF_GRID interpolation,
all five WORLD parameter sets, and synthesized waveforms from the sealed CAL
inputs. It is limited to the 20 CAL utterances and writes an engineering QA
record only; it does not inspect evaluation inputs or any model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPORT_RAW_SHA256 = "dc23a765794b0dd6e735be26955dbcb39e4ce06dff4fbe1be9f1c6b926ab7efa"
REPORT_PAYLOAD_SHA256 = "f3b8045245edcf21ebc30c944f61318f8ac3538669152d64f68c882543d75a1f"
FILE_INVENTORY_PAYLOAD_SHA256 = "abdf8e14d7957f595862f5628ba43c35924d569ca361e22b97eae98dce287e86"
FILE_INVENTORY_RAW_SHA256 = "95069667cc05d380f26de097c25a2d7efbded06bc99ecb39242c8858e844b59a"
OUTPUT_SIZE_INVENTORY_RAW_SHA256 = "1fdb17a135b861f1d45a0accfe35e6765405ecd691b327c854a07a2deee4e6bf"
OUTPUT_GZIP_RAW_SHA256 = "88862372ff202f32e044abca2dc440a485794831acdcd4d84fe43a79003da10b"
WORLD_MANIFEST_RAW_SHA256 = "297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc"
WORLD_MANIFEST_PAYLOAD_SHA256 = "75b47f39a7a462753877db6f03e41758335011764b279061d546c3e812cec29c"
WORLD_PLAN_RAW_SHA256 = "2c5d26b09c0bd3fea4812c16612387ec5c96308082c9da1e2a823ee1f6c9d8df"
# The combined CAL+EVAL plan stays unserialized; this payload identity is the
# explicit value carried by the independently raw-pinned CAL producer manifest.
WORLD_PLAN_PAYLOAD_SHA256 = "e444a0fc6b322b8e73589ce23969a0142642df76527703cab69f6f8cf9f71760"
WORLD_SOURCE_SHA256 = "7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2"
WORLD_SOURCE_DEFAULT = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/hf_selected/world_phone_transfer_20261001.py"
ANALYSIS_SOURCE_SHA256 = "4f4f1fe5d59565c87706cedf5531c3f29c649819d7a1c61413f0dee8508ed88e"
ANALYSIS_SPEC_SHA256 = "01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9"
ANALYSIS_PLAN_RAW_SHA256 = "594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7"
ANALYSIS_PLAN_PAYLOAD_SHA256 = "85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a"
FIXED_RMS = 0.0376838172675746
SAMPLE_RATE = 16000
FRAME_PERIOD = 0.005
EPSILON = 1e-12
MIXED_ABS_TOL = 1e-12
MIXED_REL_TOL = 1e-10
MIXED_REL_FLOOR = 1e-12
PARAM_ABS_TOL = 1e-12
POWER_REL_TOL = 1e-10
WAVE_MAX_ABS_TOL = 1e-6
RMS_ABS_TOL = 1e-9
CONDITIONS = {"tts_to_natural": ("natural", "tts"), "natural_to_tts": ("tts", "natural")}
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
FEATURE_KEYS = {"f0", "time", "sp", "ap", "frequency_hz", "fft_size"}
PARAM_KEYS = {"f0", "time", "sp", "ap", "frequency_hz"}
OUTPUT_PARAMETER_KEYS = {"f0", "time", "ap", "frequency_hz", "fft_size", *(f"sp_{key}" for key in CELLS)}
TRANSFER_KEYS = {"mapped_frames", "mapped_f0", "mapped_ap", "mapped_shape", "donor_time", "donor_frame", "donor_voiced", "joint_voiced", "sp_power_error_by_frame"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes({key: item for key, item in value.items() if key != "manifest_sha256"})).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value: Any, np: Any) -> str:
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise TypeError("object arrays are forbidden")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _array_equal(actual: Any, expected: Any, np: Any, label: str, tolerance: float = 0.0) -> dict[str, Any]:
    a, b = np.asarray(actual), np.asarray(expected)
    _expect(a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}")
    _expect(a.dtype == b.dtype, f"{label}: dtype {a.dtype} != {b.dtype}")
    _expect(np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: nonfinite values")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(np.max(delta, initial=0.0))
    if max_abs > tolerance:
        raise ValueError(f"{label}: max_abs={max_abs} exceeds {tolerance}")
    return {"bit_exact": bool(np.array_equal(a, b)), "max_abs_error": max_abs, "tolerance": tolerance, "dtype": a.dtype.str, "shape": list(a.shape)}


def _mixed_compare(actual: Any, expected: Any, np: Any, label: str) -> dict[str, Any]:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    _expect(a.shape == b.shape, f"{label}: mixed-comparison shape mismatch")
    _expect(np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: nonfinite mixed comparison")
    err = np.abs(a - b)
    rel = err / np.maximum(np.abs(b), MIXED_REL_FLOOR)
    allowed = MIXED_ABS_TOL + MIXED_REL_TOL * np.abs(b)
    bad = int(np.count_nonzero(err > allowed))
    if bad:
        raise ValueError(f"{label}: mixed tolerance failed ({bad} elements, max_abs={np.max(err)}, max_rel={np.max(rel)})")
    return {"passed": True, "elements": int(a.size), "max_abs_error": float(np.max(err, initial=0.0)), "max_relative_error": float(np.max(rel, initial=0.0)), "violations": bad}


def _load_npz(path: Path, np: Any, expected: set[str], cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    key = str(path.resolve(strict=True))
    if cache is not None and key in cache:
        item = cache[key]
        _expect(set(item) == expected, f"{path}: cached NPZ key inventory differs")
        return item
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        _expect(names == expected, f"{path}: NPZ key inventory mismatch, missing={sorted(expected - names)} extra={sorted(names - expected)}")
        result = {name: archive[name].copy() for name in archive.files}
    if cache is not None:
        cache[key] = result
    return result


def _verified_file(path_text: str, expected_sha: str, expected_bytes: int | None = None) -> Path:
    path = Path(path_text).resolve(strict=True)
    _expect(path.is_file(), f"not a regular file: {path}")
    if expected_bytes is not None:
        _expect(path.stat().st_size == int(expected_bytes), f"file size differs: {path}")
    actual = sha256_file(path)
    _expect(actual == expected_sha, f"raw SHA-256 mismatch for {path}: {actual} != {expected_sha}")
    return path


def _norm_sp(sp: Any, np: Any) -> Any:
    powers = np.maximum(np.asarray(sp, dtype=np.float64), 0.0)
    total = np.sum(powers, axis=1, keepdims=True, dtype=np.float64)
    _expect(np.isfinite(total).all() and np.all(total > 0.0), "SP contains a zero/nonfinite row power")
    floored = np.maximum(powers / total, EPSILON)
    return floored / np.sum(floored, axis=1, keepdims=True, dtype=np.float64)


def _row_interp(values: Any, axis: Any, query: float, np: Any) -> Any:
    rows = np.asarray(values, dtype=np.float64)
    times = np.asarray(axis, dtype=np.float64)
    _expect(rows.ndim == 2 and times.ndim == 1 and len(times) == len(rows) and len(times) > 0, "invalid interpolation dimensions")
    right = int(np.searchsorted(times, query, side="left"))
    if right == 0:
        return rows[0].copy()
    if right >= len(times):
        return rows[-1].copy()
    left = right - 1
    span = float(times[right] - times[left])
    alpha = 0.0 if span <= 0.0 else min(1.0, max(0.0, (query - float(times[left])) / span))
    return rows[left] * (1.0 - alpha) + rows[right] * alpha


def _nearest(times: Any, query: float, np: Any) -> int:
    axis = np.asarray(times, dtype=np.float64)
    _expect(axis.ndim == 1 and axis.size > 0, "nearest-center query requires a nonempty axis")
    right = int(np.searchsorted(axis, query, side="left"))
    if right <= 0:
        return 0
    if right >= len(axis):
        return len(axis) - 1
    left = right - 1
    return left if abs(query - float(axis[left])) <= abs(float(axis[right]) - query) else right


def _voiced_f0_at(times: Any, f0: Any, query: float, donor_indices: Any, np: Any) -> tuple[float, int, bool]:
    candidates = np.asarray(donor_indices, dtype=np.int64)
    if not candidates.size:
        return 0.0, -1, False
    local_times = np.asarray(times, dtype=np.float64)[candidates]
    nearest_local = _nearest(local_times, query, np)
    nearest_global = int(candidates[nearest_local])
    pitch = np.asarray(f0, dtype=np.float64)
    if float(pitch[nearest_global]) <= 0.0:
        return 0.0, nearest_global, False
    low = high = nearest_local
    while low > 0 and candidates[low] == candidates[low - 1] + 1 and float(pitch[candidates[low - 1]]) > 0.0:
        low -= 1
    while high + 1 < len(candidates) and candidates[high + 1] == candidates[high] + 1 and float(pitch[candidates[high + 1]]) > 0.0:
        high += 1
    run = candidates[low:high + 1]
    value = float(np.exp(np.interp(query, np.asarray(times)[run], np.log(pitch[run]))))
    return value, nearest_global, True


def _rebuild_transfer(rec: dict[str, Any], donor: dict[str, Any], phone_map: dict[str, Any], np: Any) -> dict[str, Any]:
    rt = np.asarray(rec["time"], dtype=np.float64)
    dt = np.asarray(donor["time"], dtype=np.float64)
    rec_sp = np.asarray(rec["sp"], dtype=np.float64)
    donor_sp = np.asarray(donor["sp"], dtype=np.float64)
    rec_ap = np.asarray(rec["ap"], dtype=np.float64)
    donor_ap = np.asarray(donor["ap"], dtype=np.float64)
    rec_f0 = np.asarray(rec["f0"], dtype=np.float64)
    donor_f0 = np.asarray(donor["f0"], dtype=np.float64)
    _expect(rec_sp.shape[1] == donor_sp.shape[1] and np.array_equal(rec["frequency_hz"], donor["frequency_hz"]), "N/T WORLD grids differ")
    donor_shape = _norm_sp(donor_sp, np)
    frame_count = len(rt)
    mapped = np.zeros(frame_count, dtype=bool)
    mapped_f0 = np.zeros(frame_count, dtype=np.float64)
    mapped_ap = np.zeros_like(rec_ap)
    mapped_shape = np.zeros_like(_norm_sp(rec_sp, np))
    donor_time = np.full(frame_count, np.nan, dtype=np.float64)
    donor_frame = np.full(frame_count, -1, dtype=np.int32)
    donor_voiced = np.zeros(frame_count, dtype=bool)
    rec_duration = float(phone_map.get("recipient_audio_duration_s", -1.0))
    don_duration = float(phone_map.get("donor_audio_duration_s", -1.0))
    for phone in phone_map.get("matched_phones", []):
        r0, r1 = map(float, phone["recipient_interval_s"])
        d0, d1 = map(float, phone["donor_interval_s"])
        rmask = (rt >= r0) & (rt < r1)
        if abs(r1 - rec_duration) <= 1e-9:
            rmask |= np.abs(rt - r1) <= 1e-9
        rec_idx = np.flatnonzero(rmask)
        dmask = (dt >= d0) & (dt < d1)
        if abs(d1 - don_duration) <= 1e-9:
            dmask = (dt >= d0) & (dt <= d1)
        don_idx = np.flatnonzero(dmask)
        if not rec_idx.size or not don_idx.size or r1 <= r0 or d1 <= d0:
            continue
        local_dt = dt[don_idx]
        local_logshape = np.log(np.maximum(donor_shape[don_idx], EPSILON))
        for ri in rec_idx:
            query = d0 + (float(rt[ri]) - r0) * (d1 - d0) / (r1 - r0)
            query = min(d1, max(d0, query))
            mapped_ap[ri] = _row_interp(donor_ap[don_idx], local_dt, query, np)
            row = np.exp(_row_interp(local_logshape, local_dt, query, np))
            total = float(np.sum(row, dtype=np.float64))
            if not math.isfinite(total) or total <= 0.0:
                continue
            mapped_shape[ri] = row / total
            mapped_f0[ri], donor_frame[ri], donor_voiced[ri] = _voiced_f0_at(dt, donor_f0, query, don_idx, np)
            donor_time[ri] = query
            mapped[ri] = True
    joint = mapped & (rec_f0 > 0.0) & donor_voiced
    rec_power = np.sum(rec_sp, axis=1, dtype=np.float64)
    err = np.full(frame_count, np.nan, dtype=np.float64)
    if np.any(mapped):
        err[mapped] = np.sum(mapped_shape[mapped] * rec_power[mapped, None], axis=1, dtype=np.float64) - rec_power[mapped]
    return {"mapped_frames": mapped, "mapped_f0": mapped_f0, "mapped_ap": mapped_ap, "mapped_shape": mapped_shape,
            "donor_time": donor_time, "donor_frame": donor_frame, "donor_voiced": donor_voiced,
            "joint_voiced": joint, "sp_power_error_by_frame": err}


def _stable_softmax(rows: Any, np: Any) -> Any:
    x = np.asarray(rows, dtype=np.float64)
    shift = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(shift)
    denom = np.sum(e, axis=1, keepdims=True, dtype=np.float64)
    _expect(np.isfinite(e).all() and np.isfinite(denom).all() and np.all(denom > 0.0), "invalid softmax")
    return e / denom


def _token_layout(phone_map: dict[str, Any], times: Any, mapped: Any, duration: float, np: Any) -> tuple[list[Any], Any]:
    axis = np.asarray(times, dtype=np.float64)
    mapped_mask = np.asarray(mapped, dtype=bool)
    phones = phone_map.get("matched_phones", [])
    _expect(mapped_mask.shape == axis.shape, "mapped frame and WORLD time shapes differ")
    rids = [int(item["recipient_phone_index"]) for item in phones]
    dids = [int(item["donor_phone_index"]) for item in phones]
    _expect(len(set(rids)) == len(rids) and len(set(dids)) == len(dids), "phone map repeats an exact phone token")
    claims = np.zeros(len(axis), dtype=np.int16)
    masks = []
    for phone in phones:
        _expect(phone.get("recipient_normalized_label") == phone.get("donor_normalized_label"), "paired normalized phone labels differ")
        start, end = map(float, phone["recipient_interval_s"])
        _expect(math.isfinite(start) and math.isfinite(end) and end > start, "invalid phone interval")
        mask = (axis >= start) & (axis < end)
        if abs(end - duration) <= 1e-9:
            mask |= np.abs(axis - end) <= 1e-9
        claims += mask.astype(np.int16)
        masks.append(mask)
    ambiguous = claims > 1
    tokens = []
    assigned = np.zeros(len(axis), dtype=bool)
    for idx, (phone, mask) in enumerate(zip(phones, masks)):
        mapped_phone = mask & mapped_mask
        eligible = mapped_phone & ~ambiguous
        _expect(not np.any(assigned & eligible), "frame assigned to multiple exact tokens")
        assigned |= eligible
        tokens.append({"token_index": idx, "phone": phone, "interval_mask": mask, "mapped_mask": mapped_phone, "eligible_mask": eligible,
                       "recipient_interval_frame_count": int(np.count_nonzero(mask)), "mapped_frame_count": int(np.count_nonzero(mapped_phone)),
                       "ambiguous_frame_count": int(np.count_nonzero(mapped_phone & ambiguous)), "eligible_frame_count": int(np.count_nonzero(eligible))})
    _expect(not np.any(mapped_mask & (claims == 0)), "mapped frame lies outside all matched phone tokens")
    return tokens, ambiguous


def _self_grid(own_log: Any, rec_times: Any, donor_times: Any, phone: dict[str, Any], indices: Any, donor_duration: float, np: Any) -> tuple[Any | None, int]:
    r0, r1 = map(float, phone["recipient_interval_s"])
    d0, d1 = map(float, phone["donor_interval_s"])
    donor_axis = np.asarray(donor_times, dtype=np.float64)
    donor_mask = (donor_axis >= d0) & (donor_axis < d1)
    if abs(d1 - donor_duration) <= 1e-9:
        donor_mask |= np.abs(donor_axis - d1) <= 1e-9
    dgrid = donor_axis[donor_mask]
    ids = np.asarray(indices, dtype=np.int64)
    if ids.size == 0 or dgrid.size == 0:
        return None, int(dgrid.size)
    rgrid = np.asarray(rec_times, dtype=np.float64)[ids]
    own_rows = np.asarray(own_log, dtype=np.float64)[ids]
    interim = np.empty((len(dgrid), own_rows.shape[1]), dtype=np.float64)
    for i, center in enumerate(dgrid):
        phase = min(1.0, max(0.0, (float(center) - d0) / (d1 - d0)))
        interim[i] = _row_interp(own_rows, rgrid, r0 + phase * (r1 - r0), np)
    back = np.empty_like(own_rows)
    for i, center in enumerate(rgrid):
        phase = min(1.0, max(0.0, (float(center) - r0) / (r1 - r0)))
        back[i] = _row_interp(interim, dgrid, d0 + phase * (d1 - d0), np)
    return _stable_softmax(back, np), int(dgrid.size)


def _read_wave(meta: dict[str, Any], sf: Any, np: Any, expected_path_root: Path, role: str) -> tuple[Any, dict[str, Any]]:
    path = Path(meta["path"]).resolve(strict=True)
    _expect(path.is_relative_to(expected_path_root), f"{role}: artifact escaped output root: {path}")
    _expect(sha256_file(path) == meta["file_sha256"], f"{role}: WAV byte SHA mismatch")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    _expect(int(rate) == SAMPLE_RATE and audio.ndim == 2 and audio.shape[1] == 1, f"{role}: wrong rate/channel count")
    samples = np.asarray(audio[:, 0], dtype=np.float32)
    _expect(len(samples) == int(meta["samples"]), f"{role}: decoded EOF sample count mismatch")
    _expect(int(meta["sample_rate_hz"]) == SAMPLE_RATE and int(meta["channels"]) == 1 and meta["subtype"] == "FLOAT", f"{role}: metadata mismatch")
    _expect(np.isfinite(samples).all(), f"{role}: nonfinite PCM")
    pcm_hash = hashlib.sha256(samples.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
    _expect(pcm_hash == meta["float32_pcm_sha256"], f"{role}: decoded PCM SHA mismatch")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    _expect(abs(rms - FIXED_RMS) <= RMS_ABS_TOL, f"{role}: fixed RMS outside target tolerance")
    return samples, {"path": str(path), "file_sha256": meta["file_sha256"], "float32_pcm_sha256": pcm_hash, "samples": len(samples), "rms": rms}


def _synthesize(pyworld: Any, params: dict[str, Any], samples: int, np: Any) -> tuple[Any, dict[str, Any]]:
    generated = np.asarray(pyworld.synthesize(np.asarray(params["f0"]), np.asarray(params["sp"]), np.asarray(params["ap"]), SAMPLE_RATE, frame_period=5.0), dtype=np.float64).reshape(-1)
    _expect(len(generated) >= samples, f"WORLD synthesis too short: {len(generated)} < {samples}")
    _expect(np.isfinite(generated).all(), "WORLD synthesis produced nonfinite samples")
    trimmed = generated[:samples]
    tail = generated[samples:]
    before = float(np.sqrt(np.mean(trimmed * trimmed)))
    _expect(math.isfinite(before) and before > 1e-12, "invalid pre-normalization synthesis RMS")
    gain = FIXED_RMS / before
    fixed = (trimmed * gain).astype(np.float32)
    after = float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2)))
    return fixed, {"generated_samples": len(generated), "right_trim_samples": len(generated) - samples,
                   "discarded_tail_rms": float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0,
                   "input_rms": before, "gain": gain, "gain_db": 20.0 * math.log10(gain), "fixed_rms": after,
                   "peak_abs": float(np.max(np.abs(fixed), initial=0.0)), "samples_over_full_scale": int(np.count_nonzero(np.abs(fixed) > 1.0))}


def _assert_array_ref(array: Any, meta: dict[str, Any], np: Any, label: str) -> None:
    value = np.asarray(array)
    _expect(list(value.shape) == meta["shape"], f"{label}: array shape reference mismatch")
    _expect(value.dtype.str == meta["dtype"], f"{label}: array dtype reference mismatch")
    _expect(array_sha(value, np) == meta["array_sha256"], f"{label}: array SHA mismatch")


def _audit_transfer(saved: dict[str, Any], rebuilt: dict[str, Any], np: Any, label: str) -> dict[str, Any]:
    _expect(set(saved) == TRANSFER_KEYS, f"{label}: transfer-audit key inventory mismatch")
    result = {}
    for key in sorted(TRANSFER_KEYS):
        left, right = np.asarray(saved[key]), np.asarray(rebuilt[key])
        _expect(left.shape == right.shape and left.dtype == right.dtype, f"{label}/{key}: transfer audit shape/dtype mismatch")
        if left.dtype.kind == "f":
            same = np.array_equal(left, right, equal_nan=True)
            finite_delta = np.abs(left[np.isfinite(left)].astype(np.float64) - right[np.isfinite(right)].astype(np.float64))
            max_abs = float(np.max(finite_delta, initial=0.0))
        else:
            same = np.array_equal(left, right)
            max_abs = 0.0 if same else None
        _expect(bool(same), f"{label}/{key}: independently rebuilt transfer differs (max_abs={max_abs})")
        result[key] = {"bit_exact": True, "dtype": left.dtype.str, "shape": list(left.shape), "array_sha256": array_sha(left, np)}
    return result


def _load_and_verify_report(path: Path, inventory_path: Path, np: Any) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    _expect(sha256_file(path) == REPORT_RAW_SHA256, "sealed CAL report raw-byte SHA differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    _expect(report.get("manifest_sha256") == REPORT_PAYLOAD_SHA256 and payload_sha(report) == REPORT_PAYLOAD_SHA256, "sealed CAL report embedded fingerprint differs")
    _expect(report.get("status") == "CAL_FACTORIAL_WAVEFORMS_READY_FOR_ROOT_REVIEW", "CAL producer report is not complete")
    _expect(report.get("split") == "calibration" and report.get("scope") == "calibration_only", "report split/scope mismatch")
    for flag in ("evaluation_waveforms_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"):
        _expect(report.get(flag) is False, f"report has prohibited flag {flag}")
    _expect(report.get("intended_direction_rows") == 40 and report.get("complete_direction_rows") == 40 and report.get("failure_count") == 0 and report.get("failures") == [], "CAL producer did not complete 40 directions")
    _expect(len(report.get("records", [])) == 40 and len(report.get("input_inventory", [])) == 320 and int(report.get("input_file_count_verified", -1)) == 320, "CAL report row/input counts differ")
    bindings = report.get("input_bindings", {})
    _expect(bindings.get("world_manifest", {}).get("sha256") == WORLD_MANIFEST_RAW_SHA256, "WORLD manifest raw identity mismatch")
    _expect(bindings.get("world_plan", {}).get("sha256") == WORLD_PLAN_RAW_SHA256, "WORLD plan raw identity mismatch")
    _expect(bindings.get("analysis_spec", {}).get("sha256") == ANALYSIS_SPEC_SHA256, "analysis spec identity mismatch")
    _expect(bindings.get("analysis_plan", {}).get("sha256") == ANALYSIS_PLAN_RAW_SHA256, "analysis plan raw identity mismatch")
    _expect(report.get("analysis_plan_payload_sha256") == ANALYSIS_PLAN_PAYLOAD_SHA256, "analysis plan payload identity mismatch")
    _expect(report.get("analysis_plan_path") == bindings.get("analysis_plan", {}).get("path") and report.get("analysis_plan_raw_sha256") == ANALYSIS_PLAN_RAW_SHA256, "top-level analysis plan binding differs from input binding")
    _expect(report.get("analysis_spec_path") == bindings.get("analysis_spec", {}).get("path") and report.get("analysis_spec_raw_sha256") == ANALYSIS_SPEC_SHA256, "top-level analysis spec binding differs from input binding")
    _expect(report.get("analysis_source_sha256") == ANALYSIS_SOURCE_SHA256, "top-level analysis source SHA differs")
    for key, binding in bindings.items():
        _verified_file(binding["path"], binding["sha256"], binding.get("bytes"))
        if key == "world_plan":
            # The combined WORLD plan is raw-hash pinned only; its condition list is never parsed.
            continue
    if report.get("analysis_source_path"):
        _verified_file(report["analysis_source_path"], ANALYSIS_SOURCE_SHA256)
    world_manifest = json.loads(Path(bindings["world_manifest"]["path"]).read_text(encoding="utf-8"))
    _expect(world_manifest.get("manifest_sha256") == WORLD_MANIFEST_PAYLOAD_SHA256 and payload_sha(world_manifest) == WORLD_MANIFEST_PAYLOAD_SHA256, "source WORLD CAL manifest payload mismatch")
    _expect(world_manifest.get("manifest_type") == "world_phone_transfer_calibration_producer" and world_manifest.get("split") == "calibration", "source WORLD manifest is not CAL-only")
    _expect(world_manifest.get("status") == "CAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW" and world_manifest.get("failures") == [], "source WORLD CAL manifest is incomplete")
    _expect(world_manifest.get("evaluation_outcomes_read") is False and world_manifest.get("evaluation_waveforms_read") is False, "source WORLD manifest records evaluation access")
    _expect(world_manifest.get("world_transfer_plan_sha256") == WORLD_PLAN_PAYLOAD_SHA256, "source WORLD manifest payload binding differs from the exact frozen plan payload")
    world_records = world_manifest.get("records")
    _expect(isinstance(world_records, list) and len(world_records) == 20, "source WORLD CAL manifest does not contain 20 CAL records")
    world_by_id = {str(item["id"]): item for item in world_records}
    _expect(len(world_by_id) == 20, "source WORLD CAL manifest has duplicate IDs")
    plan_binding = bindings["analysis_plan"]
    analysis_plan = json.loads(Path(plan_binding["path"]).read_text(encoding="utf-8"))
    _expect(payload_sha(analysis_plan) == ANALYSIS_PLAN_PAYLOAD_SHA256, "analysis-only CAL plan payload hash mismatch")
    _expect(analysis_plan.get("scope") == "calibration_only" and analysis_plan.get("evaluation_waveforms_read") is False and analysis_plan.get("evaluation_outcomes_read") is False, "analysis plan is not CAL-only")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    _expect(sha256_file(inventory_path) == FILE_INVENTORY_RAW_SHA256, "output inventory raw-byte SHA differs")
    _expect(inventory.get("manifest_type") == "world_phone_shape_factorial_cal_output_file_inventory", "output inventory type mismatch")
    _expect(inventory.get("manifest_sha256") == FILE_INVENTORY_PAYLOAD_SHA256 and payload_sha(inventory) == FILE_INVENTORY_PAYLOAD_SHA256, "output inventory payload mismatch")
    _expect(report.get("output_file_inventory_sha256") == FILE_INVENTORY_PAYLOAD_SHA256, "report/output inventory link mismatch")
    out_root = Path(path).resolve().parent
    _expect(Path(report.get("output_file_inventory_path", "")).resolve() == inventory_path.resolve(), "report points to another output inventory")
    return report, inventory, out_root, analysis_plan, world_by_id


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pyworld
    import soundfile as sf

    report_path = args.manifest.resolve(strict=True)
    inventory_path = args.output_inventory.resolve(strict=True)
    size_inventory_path = args.output_size_inventory.resolve(strict=True)
    report, output_inventory, output_root, analysis_plan, world_by_id = _load_and_verify_report(report_path, inventory_path, np)
    _expect(sha256_file(size_inventory_path) == OUTPUT_SIZE_INVENTORY_RAW_SHA256, "output-size inventory raw-byte SHA differs")
    size_inventory = json.loads(size_inventory_path.read_text(encoding="utf-8"))
    _expect(size_inventory.get("schema") == "world_shape_mean_dynamic_cal_output_size_inventory_v1" and size_inventory.get("status") == "PASS", "output-size inventory schema/status mismatch")
    _expect(int(size_inventory.get("file_count", -1)) == 283 and int(size_inventory.get("total_bytes", -1)) == 801875807, "output-size inventory count/bytes mismatch")
    _expect(size_inventory.get("producer_inventory_sha256") == FILE_INVENTORY_PAYLOAD_SHA256 and size_inventory.get("report_manifest_sha256") == REPORT_PAYLOAD_SHA256, "output-size inventory binding mismatch")
    world_source_path = _verified_file(str(args.world_source), WORLD_SOURCE_SHA256)
    manifest_world_source = Path(report["input_bindings"]["world_source"]["path"]).resolve(strict=True)
    _expect(world_source_path == manifest_world_source, "CLI WORLD source path differs from the report-bound source path")
    _expect(getattr(pyworld, "__version__", None) == "0.3.5", f"unexpected pyworld version {getattr(pyworld, '__version__', None)}")
    expected_pyworld = report["runtime"]
    ext_path = Path(pyworld.pyworld.__file__).resolve(strict=True)
    pkg_path = Path(pyworld.__file__).resolve(strict=True)
    _expect(sha256_file(ext_path) == expected_pyworld["pyworld_extension_sha256"], "pyworld extension SHA differs from execution")
    _expect(sha256_file(pkg_path) == expected_pyworld["pyworld_package_sha256"], "pyworld package SHA differs from execution")
    _expect(float(expected_pyworld["fixed_rms"]) == FIXED_RMS and int(expected_pyworld["sample_rate_hz"]) == SAMPLE_RATE and float(expected_pyworld["frame_period_ms"]) == 5.0, "runtime target/clock mismatch")

    inventory_rows = output_inventory["files"]
    _expect(len(inventory_rows) == 280, f"expected 280 scientific output files, got {len(inventory_rows)}")
    expected_relative = set()
    expected_output_sha: dict[str, str] = {}
    for entry in inventory_rows:
        rel = Path(entry["path"])
        _expect(not rel.is_absolute() and ".." not in rel.parts, "output inventory contains an unsafe path")
        target = (output_root / rel).resolve(strict=True)
        _expect(target.is_relative_to(output_root), "inventory output escaped output root")
        _expect(target.stat().st_size == int(entry["bytes"]), f"output byte count mismatch: {target}")
        _expect(sha256_file(target) == entry["sha256"], f"output raw SHA mismatch: {target}")
        expected_relative.add(rel.as_posix())
        expected_output_sha[rel.as_posix()] = entry["sha256"]
    output_inventory_by_path = {str(entry["path"]): entry for entry in inventory_rows}
    _expect(len(output_inventory_by_path) == len(inventory_rows), "duplicate path in output file inventory")

    def verify_output_ref(path_text: str, expected_sha: str, label: str) -> Path:
        target = Path(path_text).resolve(strict=True)
        _expect(target.is_relative_to(output_root), f"{label}: output reference escaped output root")
        rel = target.relative_to(output_root).as_posix()
        _expect(rel in output_inventory_by_path, f"{label}: output reference absent from sealed file inventory")
        inventory_entry = output_inventory_by_path[rel]
        _expect(inventory_entry["sha256"] == expected_sha and target.stat().st_size == int(inventory_entry["bytes"]), f"{label}: output reference SHA/size differs from sealed inventory")
        return target

    actual_relative_all = {p.relative_to(output_root).as_posix() for p in output_root.rglob("*") if p.is_file()}
    actual_relative = actual_relative_all - {"calibration_manifest.json", "calibration_manifest.json.gz", "output_file_inventory.json"}
    _expect(actual_relative == expected_relative, f"output tree differs from inventory: missing={len(expected_relative-actual_relative)} extra={len(actual_relative-expected_relative)}")
    size_rows = size_inventory.get("files", [])
    _expect(len(size_rows) == 283, "output-size inventory entry count mismatch")
    size_by_path = {str(item["path"]): item for item in size_rows}
    _expect(len(size_by_path) == len(size_rows), "duplicate path in output-size inventory")
    metadata_hashes = {"calibration_manifest.json": REPORT_RAW_SHA256,
                       "calibration_manifest.json.gz": OUTPUT_GZIP_RAW_SHA256,
                       "output_file_inventory.json": FILE_INVENTORY_RAW_SHA256}
    _expect(set(size_by_path) == expected_relative | set(metadata_hashes), "output-size inventory path set differs from scientific outputs plus three metadata files")
    for rel, item in size_by_path.items():
        _expect(not Path(rel).is_absolute() and ".." not in Path(rel).parts, f"unsafe output-size path: {rel}")
        _expect((output_root / rel).resolve(strict=True).is_relative_to(output_root), f"output-size path escaped run output: {rel}")
        if rel in expected_output_sha:
            _expect(int(item["bytes"]) == next(int(entry["bytes"]) for entry in inventory_rows if entry["path"] == rel) and item["sha256"] == expected_output_sha[rel], f"output-size inventory disagrees with file inventory: {rel}")
        else:
            target = output_root / rel
            _expect(target.stat().st_size == int(item["bytes"]) and item["sha256"] == metadata_hashes[rel] and sha256_file(target) == metadata_hashes[rel], f"output-size metadata file differs: {rel}")
    _expect(sum(int(item["bytes"]) for item in size_rows) == 801875807, "output-size inventory total byte count does not sum")

    input_entries = report["input_inventory"]
    input_roles: dict[str, dict[str, Any]] = {}
    for entry in input_entries:
        role = entry["role"]
        _expect(role not in input_roles, f"duplicate input role: {role}")
        path = _verified_file(entry["path"], entry["sha256"], entry["bytes"])
        input_roles[role] = {**entry, "resolved_path": path}
    _expect(len(input_roles) == 320, "input role inventory is not exactly 320")

    phone_binding = report["input_bindings"].get("phone_inventory")
    _expect(isinstance(phone_binding, dict), "frozen CAL phone inventory binding is missing")
    phone_inventory_path = _verified_file(phone_binding["path"], phone_binding["sha256"], phone_binding.get("bytes"))
    phone_inventory = json.loads(phone_inventory_path.read_text(encoding="utf-8"))
    phone_inventory_payload = hashlib.sha256(canonical_bytes({k: v for k, v in phone_inventory.items() if k != "inventory_sha256"})).hexdigest()
    _expect(phone_inventory.get("status") == "FROZEN_CAL_ONLY_PHONE_INVENTORY" and phone_inventory.get("selection_split") == "calibration", "phone inventory is not the frozen CAL-only selection")
    _expect(phone_inventory.get("evaluation_textgrids_or_endpoints_read") is False, "phone inventory records evaluation read")
    _expect(phone_inventory.get("inventory_sha256") == phone_inventory_payload, "phone inventory payload fingerprint mismatch")
    _expect(phone_inventory_payload == analysis_plan.get("input_bindings", {}).get("phone_inventory_payload_sha256"), "phone inventory payload differs from frozen analysis plan")
    phone_labels = set(phone_inventory.get("selected_phone_labels", []))
    _expect(analysis_plan.get("inferential_phone_labels") == phone_inventory.get("selected_phone_labels"), "frozen analysis label family differs from phone inventory")
    _expect(report.get("factorization", {}).get("inference_labels") == phone_inventory.get("selected_phone_labels"), "reported fixed inference label family differs")

    feature_cache: dict[str, dict[str, Any]] = {}
    endpoint_cache: dict[str, dict[str, Any]] = {}
    audit_cache: dict[str, dict[str, Any]] = {}
    component_cache: dict[str, dict[str, Any]] = {}
    output_parameter_cache: dict[str, dict[str, Any]] = {}
    decoded_input_waves: dict[str, Any] = {}
    row_ids = [str(row.get("id")) for row in report["records"]]
    ids = list(dict.fromkeys(row_ids))
    _expect(len(ids) == 20 and len(set(row_ids)) == 20, "CAL direction record identities are not 20 ordered pairs")
    _expect(ids == analysis_plan.get("ordered_calibration_ids"), "ordered CAL IDs differ from the frozen plan")
    _expect(len({str(row.get("speaker")) for row in report["records"]}) == 10, "CAL direction rows do not span 10 speakers")
    ids_by_speaker: dict[str, set[str]] = {}
    for source_row in report["records"]:
        ids_by_speaker.setdefault(str(source_row["speaker"]), set()).add(str(source_row["id"]))
    _expect(all(len(samples) == 2 for samples in ids_by_speaker.values()), "CAL speakers do not each have two distinct utterances")
    for sample_id in ids:
        got = [row["direction"] for row in report["records"] if row["id"] == sample_id]
        _expect(got == list(CONDITIONS), f"{sample_id}: directions/order differ from frozen plan")
        _expect(sample_id in world_by_id, f"{sample_id}: report row absent from source WORLD CAL manifest")

    qa_rows = []
    cell_total = 0
    total_token_rows = 0
    transfer_array_count = 0
    for row_index, row in enumerate(report["records"]):
        sample_id = str(row["id"])
        direction = str(row["direction"])
        rec_condition, donor_condition = CONDITIONS[direction]
        row_label = f"{sample_id}/{direction}"
        _expect(row.get("status") == "PASS" and row.get("recipient_condition") == rec_condition and row.get("donor_condition") == donor_condition, f"{row_label}: status or direction mismatch")
        source_record = world_by_id[sample_id]
        _expect(row.get("source_identity_sha256") == source_record.get("source_identity_sha256") and row.get("speaker") == source_record.get("speaker"), f"{row_label}: source identity/speaker differs from WORLD CAL manifest")

        def input_entry(role: str) -> dict[str, Any]:
            _expect(role in input_roles, f"{row_label}: missing sealed input role {role}")
            return input_roles[role]

        rec_feature_meta = input_entry(f"{sample_id}/{rec_condition}/WORLD_features")
        don_feature_meta = input_entry(f"{sample_id}/{donor_condition}/WORLD_features")
        rec_feature = _load_npz(rec_feature_meta["resolved_path"], np, FEATURE_KEYS, feature_cache)
        don_feature = _load_npz(don_feature_meta["resolved_path"], np, FEATURE_KEYS, feature_cache)
        for cond, feature in ((rec_condition, rec_feature), (donor_condition, don_feature)):
            _expect(np.isfinite(np.asarray(feature["f0"])).all() and np.isfinite(np.asarray(feature["time"])).all(), f"{row_label}/{cond}: nonfinite F0/time")
            _expect(np.isfinite(np.asarray(feature["sp"])).all() and np.isfinite(np.asarray(feature["ap"])).all(), f"{row_label}/{cond}: nonfinite SP/AP")
            _expect(np.all(np.asarray(feature["sp"]) >= 0.0) and np.all(np.asarray(feature["ap"]) >= -1e-6) and np.all(np.asarray(feature["ap"]) <= 1.0 + 1e-6), f"{row_label}/{cond}: SP/AP domain failure")
            _expect(np.asarray(feature["sp"]).ndim == 2 and np.asarray(feature["sp"]).shape == np.asarray(feature["ap"]).shape and len(feature["time"]) == np.asarray(feature["sp"]).shape[0], f"{row_label}/{cond}: feature dimensions disagree")
            _expect(np.asarray(feature["frequency_hz"]).shape == (np.asarray(feature["sp"]).shape[1],), f"{row_label}/{cond}: frequency dimension mismatch")
            _expect(int(np.asarray(feature["fft_size"]).item()) == 2 * (np.asarray(feature["sp"]).shape[1] - 1), f"{row_label}/{cond}: FFT size mismatch")
        _expect(np.array_equal(rec_feature["frequency_hz"], don_feature["frequency_hz"]), f"{row_label}: donor/recipient frequency bins differ")
        _expect(sha256_file(rec_feature_meta["resolved_path"]) == row["recipient_world_feature_sha256"], f"{row_label}: recipient WORLD feature report binding mismatch")
        _expect(sha256_file(don_feature_meta["resolved_path"]) == row["donor_world_feature_sha256"], f"{row_label}: donor WORLD feature report binding mismatch")

        rec_audio_meta = input_entry(f"{sample_id}/{rec_condition}/fixed_audio")
        _expect(rec_audio_meta["sha256"] == row["recipient_fixed_audio_sha256"], f"{row_label}: recipient fixed-audio SHA differs from row")
        if str(rec_audio_meta["resolved_path"]) not in decoded_input_waves:
            wav, rate = sf.read(str(rec_audio_meta["resolved_path"]), dtype="float32", always_2d=True)
            _expect(int(rate) == SAMPLE_RATE and wav.shape[1] == 1 and np.isfinite(wav).all(), f"{row_label}: recipient input WAV decode invalid")
            decoded_input_waves[str(rec_audio_meta["resolved_path"])] = np.asarray(wav[:, 0], dtype=np.float32)
        recipient_audio = decoded_input_waves[str(rec_audio_meta["resolved_path"])]
        recipient_samples = len(recipient_audio)
        recipient_duration = recipient_samples / SAMPLE_RATE
        _expect(int(row.get("sample_count", -1)) == recipient_samples, f"{row_label}: report sample count differs from decoded recipient PCM")
        don_audio_meta = input_entry(f"{sample_id}/{donor_condition}/fixed_audio")
        _expect(don_audio_meta["sha256"] == row["donor_fixed_audio_sha256"], f"{row_label}: donor fixed-audio SHA differs from row")
        if str(don_audio_meta["resolved_path"]) not in decoded_input_waves:
            wav, rate = sf.read(str(don_audio_meta["resolved_path"]), dtype="float32", always_2d=True)
            _expect(int(rate) == SAMPLE_RATE and wav.shape[1] == 1 and np.isfinite(wav).all(), f"{row_label}: donor input WAV decode invalid")
            decoded_input_waves[str(don_audio_meta["resolved_path"])] = np.asarray(wav[:, 0], dtype=np.float32)
        donor_duration = len(decoded_input_waves[str(don_audio_meta["resolved_path"])]) / SAMPLE_RATE

        rec_param_meta = row["original_parameter_endpoints"]["RT"]
        sp_param_meta = row["original_parameter_endpoints"]["SP"]
        rt_path = _verified_file(rec_param_meta["path"], rec_param_meta["file_sha256"])
        sp_path = _verified_file(sp_param_meta["path"], sp_param_meta["file_sha256"])
        rt_params = _load_npz(rt_path, np, PARAM_KEYS, endpoint_cache)
        sp_params = _load_npz(sp_path, np, PARAM_KEYS, endpoint_cache)
        for key in ("f0", "time", "sp", "ap", "frequency_hz"):
            _array_equal(rt_params[key], rec_feature[key], np, f"{row_label}/RT-vs-recipient-feature/{key}")
        for key in ("f0", "time", "ap", "frequency_hz"):
            _array_equal(sp_params[key], rt_params[key], np, f"{row_label}/SP-vs-RT-recipient-preservation/{key}")
        _expect(len(rt_params["time"]) > 0 and len(rt_params["f0"]) == len(rt_params["time"]), f"{row_label}: empty or mismatched WORLD clock")
        _expect(int(row.get("world_frame_count", -1)) == len(rt_params["time"]), f"{row_label}: report WORLD frame count differs")
        _expect(np.all(np.diff(np.asarray(rt_params["time"], dtype=np.float64)) > 0.0), f"{row_label}: WORLD time centers not increasing")
        _expect(np.allclose(np.diff(np.asarray(rt_params["time"], dtype=np.float64)), FRAME_PERIOD, rtol=0.0, atol=1e-7), f"{row_label}: WORLD clock differs from 5 ms")

        phone_meta = row["phone_map"]
        phone_path = _verified_file(phone_meta["path"], phone_meta["file_sha256"])
        phone_map = json.loads(phone_path.read_text(encoding="utf-8"))
        _expect(phone_map.get("split") == "calibration" and phone_map.get("id") == sample_id and phone_map.get("speaker") == row["speaker"], f"{row_label}: phone-map identity mismatch")
        _expect(phone_map.get("recipient_condition") == rec_condition and phone_map.get("donor_condition") == donor_condition, f"{row_label}: phone-map direction mismatch")
        _expect(int(phone_meta["matched_phone_count"]) == len(phone_map.get("matched_phones", [])), f"{row_label}: phone-map count mismatch")
        _expect(abs(float(phone_map.get("recipient_audio_duration_s", -1.0)) - recipient_duration) <= 1e-9 and abs(float(phone_map.get("donor_audio_duration_s", -1.0)) - donor_duration) <= 1e-9, f"{row_label}: phone-map durations mismatch decoded PCM")

        audit_meta = row["transfer_audit"]
        audit_role = f"{sample_id}/{direction}/transfer_audit"
        audit_input = input_entry(audit_role)
        audit_path = audit_input["resolved_path"]
        _expect(Path(audit_meta["path"]).resolve(strict=True) == audit_path, f"{row_label}: transfer-audit path differs from the sealed WORLD input role")
        _expect(audit_meta["file_sha256"] == audit_input["sha256"], f"{row_label}: transfer-audit SHA differs from the sealed WORLD input role")
        _expect(audit_path.stat().st_size == int(audit_input["bytes"]), f"{row_label}: transfer-audit size differs from the sealed WORLD input role")
        saved_audit = _load_npz(audit_path, np, TRANSFER_KEYS, audit_cache)
        rebuilt_audit = _rebuild_transfer(rec_feature, don_feature, phone_map, np)
        audit_checks = _audit_transfer(saved_audit, rebuilt_audit, np, row_label)
        _expect(set(audit_meta.get("arrays", {})) == TRANSFER_KEYS, f"{row_label}: transfer-audit reference inventory mismatch")
        for array_key, actual_meta in audit_checks.items():
            reported_meta = audit_meta["arrays"][array_key]
            _expect(reported_meta.get("bit_exact") is True and reported_meta.get("shape") == actual_meta["shape"] and reported_meta.get("dtype") == actual_meta["dtype"] and reported_meta.get("array_sha256") == actual_meta["array_sha256"], f"{row_label}/{array_key}: transfer-audit array reference differs")
        transfer_array_count += len(audit_checks)

        own_shape = _norm_sp(rt_params["sp"], np)
        mapped_shape = np.asarray(rebuilt_audit["mapped_shape"], dtype=np.float64)
        mapped = np.asarray(rebuilt_audit["mapped_frames"], dtype=bool)
        times = np.asarray(rt_params["time"], dtype=np.float64)
        rec_power = np.sum(np.asarray(rt_params["sp"], dtype=np.float64), axis=1, dtype=np.float64)
        _expect(np.all(np.isfinite(rec_power)) and np.all(rec_power > 0.0), f"{row_label}: invalid recipient SP power")
        tokens, ambiguous = _token_layout(phone_map, times, mapped, recipient_duration, np)
        _expect(len(row["token_rows"]) == len(tokens), f"{row_label}: matched token records were dropped")
        token_records = row["token_rows"]
        component_meta = row["components_npz"]
        component_path = verify_output_ref(component_meta["path"], component_meta["file_sha256"], f"{row_label}/phone-components")
        expected_component_keys = set()
        for token in tokens:
            if token["eligible_frame_count"] > 0:
                prefix = f"token_{token['token_index']:04d}"
                expected_component_keys.update({f"{prefix}_mu_recipient", f"{prefix}_mu_donor", f"{prefix}_frame_indices"})
        component_arrays = _load_npz(component_path, np, expected_component_keys, component_cache)
        _expect(set(component_meta.get("arrays", {})) == expected_component_keys, f"{row_label}: component array reference inventory mismatch")
        for array_key in sorted(expected_component_keys):
            _assert_array_ref(component_arrays[array_key], component_meta["arrays"][array_key], np, f"{row_label}/component-ref/{array_key}")
            reference = component_meta["arrays"][array_key]
            _expect(reference.get("key") == array_key and reference.get("path") == component_meta["path"] and reference.get("file_sha256") == component_meta["file_sha256"], f"{row_label}/{array_key}: component reference path/key/file SHA differs")

        own_log = np.log(np.maximum(own_shape, EPSILON))
        donor_log = np.log(np.maximum(mapped_shape, EPSILON))
        cell_shapes = {cell: own_shape.copy() for cell in CELLS}
        token_dynamics = []
        epsilon_rows = []
        token_label_acc: dict[str, dict[str, Any]] = {}
        q00_token_checks = []
        q11_map_checks = []
        q11_endpoint_checks = []
        total_self_frames = 0
        eligible_mask = np.zeros(len(times), dtype=bool)
        for index, (token, token_row) in enumerate(zip(tokens, token_records)):
            phone = token["phone"]
            _expect(int(token_row.get("token_index", -1)) == index and token_row.get("phone_label") == phone["recipient_normalized_label"], f"{row_label}: token index/label mismatch")
            _expect(int(token_row.get("recipient_phone_index", -1)) == int(phone["recipient_phone_index"]) and int(token_row.get("donor_phone_index", -1)) == int(phone["donor_phone_index"]), f"{row_label}: token identity mismatch")
            label = str(phone["recipient_normalized_label"])
            _expect(bool(token_row.get("phone_in_frozen16_inventory")) == (label in phone_labels), f"{row_label}/token{index}: frozen-label membership differs")
            label_acc = token_label_acc.setdefault(label, {"matched": 0, "eligible": 0, "dynamics": {cell: {"means": [], "with_pairs": 0} for cell in CELLS}})
            label_acc["matched"] += 1
            for name in ("recipient_interval_frame_count", "mapped_frame_count", "ambiguous_frame_count", "eligible_frame_count"):
                _expect(int(token_row.get(name, -1)) == token[name], f"{row_label}/token{index}: {name} mismatch")
            ids_frame = np.flatnonzero(token["eligible_mask"])
            eligible_mask |= token["eligible_mask"]
            if not len(ids_frame):
                _expect(token_row.get("status") == "no_eligible_frames" and token_row.get("mu_recipient_array_key") is None and token_row.get("mu_donor_array_key") is None, f"{row_label}/token{index}: empty-domain status mismatch")
                continue
            _expect(token_row.get("status") == "ok", f"{row_label}/token{index}: eligible token status mismatch")
            label_acc["eligible"] += 1
            total_self_frames += len(ids_frame)
            prefix = f"token_{index:04d}"
            mu_r_key, mu_d_key, frame_key = f"{prefix}_mu_recipient", f"{prefix}_mu_donor", f"{prefix}_frame_indices"
            _expect(token_row.get("mu_recipient_array_key") == mu_r_key and token_row.get("mu_donor_array_key") == mu_d_key and token_row.get("frame_indices_array_key") == frame_key, f"{row_label}/token{index}: component array refs mismatch")
            index_array = ids_frame.astype(np.int32)
            mu_r = np.mean(own_log[ids_frame], axis=0, dtype=np.float64)
            mu_d = np.mean(donor_log[ids_frame], axis=0, dtype=np.float64)
            _array_equal(component_arrays[frame_key], index_array, np, f"{row_label}/token{index}/frame-index")
            _array_equal(component_arrays[mu_r_key], mu_r, np, f"{row_label}/token{index}/mu-recipient", tolerance=PARAM_ABS_TOL)
            _array_equal(component_arrays[mu_d_key], mu_d, np, f"{row_label}/token{index}/mu-donor", tolerance=PARAM_ABS_TOL)
            rho_r = own_log[ids_frame] - mu_r[None, :]
            rho_d = donor_log[ids_frame] - mu_d[None, :]
            _expect(float(np.max(np.abs((mu_r[None, :] + rho_r) - own_log[ids_frame]))) <= PARAM_ABS_TOL, f"{row_label}/token{index}: recipient mean/residual reconstruction failed")
            _expect(float(np.max(np.abs((mu_d[None, :] + rho_d) - donor_log[ids_frame]))) <= PARAM_ABS_TOL, f"{row_label}/token{index}: donor mean/residual reconstruction failed")
            q00 = _stable_softmax(mu_r[None, :] + rho_r, np)
            q10 = _stable_softmax(mu_d[None, :] + rho_r, np)
            q01 = _stable_softmax(mu_r[None, :] + rho_d, np)
            q11 = _stable_softmax(mu_d[None, :] + rho_d, np)
            q00_token_checks.append(_mixed_compare(q00, own_shape[ids_frame], np, f"{row_label}/token{index}/q00-formula-own-shape"))
            q11_map_checks.append(_mixed_compare(q11, mapped_shape[ids_frame], np, f"{row_label}/token{index}/q11-formula-mapped-shape"))
            q11_endpoint_checks.append(_mixed_compare(q11, (np.asarray(sp_params["sp"], dtype=np.float64) / rec_power[:, None])[ids_frame], np, f"{row_label}/token{index}/q11-formula-SP-endpoint"))
            cell_shapes["q00"][ids_frame] = q00
            cell_shapes["q10"][ids_frame] = q10
            cell_shapes["q01"][ids_frame] = q01
            cell_shapes["q11"][ids_frame] = q11
            self_shape, donor_grid_count = _self_grid(own_log, times, don_feature["time"], phone, ids_frame, donor_duration, np)
            _expect(self_shape is not None, f"{row_label}/token{index}: SELF_GRID had no donor grid")
            cell_shapes["SELF_GRID"][ids_frame] = self_shape
            dynamics = {}
            for cell in CELLS:
                frame_ids = ids_frame
                adjacent = frame_ids[:-1][np.diff(frame_ids) == 1]
                delta = np.abs(np.log(np.maximum(cell_shapes[cell][adjacent + 1], EPSILON)) - np.log(np.maximum(cell_shapes[cell][adjacent], EPSILON))) if len(adjacent) else np.empty((0, own_shape.shape[1]))
                mean_adj = float(np.mean(delta, dtype=np.float64)) if delta.size else None
                median_adj = float(np.median(delta)) if delta.size else None
                expected_dyn = token_row["cell_logshape_dynamics"][cell]
                _expect(int(expected_dyn["consecutive_pair_count"]) == len(adjacent), f"{row_label}/token{index}/{cell}: adjacent index count mismatch")
                if mean_adj is None:
                    _expect(expected_dyn.get("mean_abs_adjacent_delta_log_shape") is None, f"{row_label}/token{index}/{cell}: missing empty dynamic statistic")
                else:
                    _expect(abs(float(expected_dyn["mean_abs_adjacent_delta_log_shape"]) - mean_adj) <= 1e-12, f"{row_label}/token{index}/{cell}: adjacent dynamic mean mismatch")
                    _expect(abs(float(expected_dyn["p50_binwise_abs_delta"]) - median_adj) <= 1e-12, f"{row_label}/token{index}/{cell}: adjacent dynamic median mismatch")
                dynamics[cell] = {"consecutive_pair_count": len(adjacent), "mean_abs_adjacent_delta_log_shape": mean_adj, "p50_binwise_abs_delta": median_adj}
                if mean_adj is not None:
                    label_acc["dynamics"][cell]["means"].append(mean_adj)
                if len(adjacent) > 0:
                    label_acc["dynamics"][cell]["with_pairs"] += 1
            _expect(token_row.get("self_grid", {}).get("status") == "ok" and int(token_row["self_grid"].get("donor_grid_center_count", -1)) == donor_grid_count and token_row["self_grid"].get("recipient_frame_indices") == ids_frame.tolist(), f"{row_label}/token{index}: SELF_GRID index metadata mismatch")
            _expect(abs(float(token_row.get("mean_log_shape_gap_mean_absolute_per_bin", float("nan"))) - float(np.mean(np.abs(mu_d - mu_r)))) <= 1e-12, f"{row_label}/token{index}: mean-shape gap differs")
            _expect(abs(float(token_row.get("recipient_log_residual_rms", float("nan"))) - float(np.sqrt(np.mean(rho_r * rho_r)))) <= 1e-12, f"{row_label}/token{index}: recipient residual RMS differs")
            _expect(abs(float(token_row.get("donor_log_residual_rms", float("nan"))) - float(np.sqrt(np.mean(rho_d * rho_d)))) <= 1e-12, f"{row_label}/token{index}: donor residual RMS differs")
            rec_mass = np.sum(np.maximum(EPSILON - own_shape[ids_frame], 0.0), axis=1, dtype=np.float64)
            don_mass = np.sum(np.maximum(EPSILON - mapped_shape[ids_frame], 0.0), axis=1, dtype=np.float64)
            epsilon_rows.append({"token_index": index, "recipient_floor_mass_max": float(np.max(rec_mass, initial=0.0)),
                                 "donor_floor_mass_max": float(np.max(don_mass, initial=0.0)), "donor_grid_centers": donor_grid_count,
                                 "recipient_row_count": len(ids_frame), "donor_row_count": len(ids_frame),
                                 "recipient_floor_rows": int(np.count_nonzero(np.any(own_shape[ids_frame] < EPSILON, axis=1))),
                                 "donor_floor_rows": int(np.count_nonzero(np.any(mapped_shape[ids_frame] < EPSILON, axis=1)))})
            token_dynamics.append({"token_index": index, "phone_label": phone["recipient_normalized_label"], "eligible_frames": len(ids_frame), "dynamics": dynamics})

        _expect(int(row.get("eligible_factorization_frame_count", -1)) == int(np.count_nonzero(eligible_mask)), f"{row_label}: eligible frame total mismatch")
        _expect(int(row.get("mapped_frame_count", -1)) == int(np.count_nonzero(mapped)), f"{row_label}: mapped frame count mismatch")
        _expect(int(row.get("ambiguous_mapped_frame_count", -1)) == int(np.count_nonzero(ambiguous & mapped)), f"{row_label}: ambiguous count mismatch")
        _expect(row.get("ambiguous_mapped_frames_exactly_own_in_original_SP_endpoint") is True, f"{row_label}: ambiguous endpoint policy not certified")
        _expect(row.get("outside_eligible_frames_exactly_own_in_all_cells") is True, f"{row_label}: outside-eligible policy not certified")
        _expect(int(row.get("self_grid_eligible_frame_count", -1)) == total_self_frames, f"{row_label}: SELF_GRID eligible count mismatch")
        _expect(len(row.get("epsilon_floor_mass_audit", {}).get("token_rows", [])) == len(epsilon_rows), f"{row_label}: epsilon audit token coverage mismatch")
        for actual_epsilon, reported_epsilon in zip(epsilon_rows, row["epsilon_floor_mass_audit"]["token_rows"]):
            _expect(int(reported_epsilon.get("token_index", -1)) == actual_epsilon["token_index"], f"{row_label}: epsilon audit token order mismatch")
            for domain, mass_key, count_key, row_count_key in (("recipient", "recipient_floor_mass_max", "recipient_floor_rows", "recipient_row_count"), ("mapped_donor", "donor_floor_mass_max", "donor_floor_rows", "donor_row_count")):
                detail = reported_epsilon.get(domain, {})
                _expect(abs(float(detail.get("observed_max_added_mass", float("nan"))) - actual_epsilon[mass_key]) <= 1e-18, f"{row_label}: {domain} epsilon max differs")
                _expect(int(detail.get("rows_with_floored_bins", -1)) == actual_epsilon[count_key], f"{row_label}: {domain} floored-row count differs")
                _expect(int(detail.get("frequency_bin_count", -1)) == own_shape.shape[1] and int(detail.get("row_count", -1)) == actual_epsilon[row_count_key], f"{row_label}: {domain} epsilon row/bin count differs")
                _expect(abs(float(detail.get("upper_bound_per_row", float("nan"))) - own_shape.shape[1] * EPSILON) <= 1e-24, f"{row_label}: {domain} epsilon mass upper bound differs")
        epsilon_summary = row["epsilon_floor_mass_audit"]
        _expect(float(epsilon_summary.get("epsilon", float("nan"))) == EPSILON, f"{row_label}: epsilon value differs")
        _expect(float(epsilon_summary.get("theoretical_max_added_mass_per_row", float("nan"))) == own_shape.shape[1] * EPSILON, f"{row_label}: epsilon global row bound differs")
        _expect(int(epsilon_summary.get("recipient_rows_with_floored_bins", -1)) == sum(item["recipient_floor_rows"] for item in epsilon_rows), f"{row_label}: recipient aggregate floor count differs")
        _expect(int(epsilon_summary.get("mapped_donor_rows_with_floored_bins", -1)) == sum(item["donor_floor_rows"] for item in epsilon_rows), f"{row_label}: donor aggregate floor count differs")
        _expect(abs(float(epsilon_summary.get("recipient_observed_max_added_mass", float("nan"))) - max((item["recipient_floor_mass_max"] for item in epsilon_rows), default=0.0)) <= 1e-18, f"{row_label}: recipient aggregate floor mass differs")
        _expect(abs(float(epsilon_summary.get("mapped_donor_observed_max_added_mass", float("nan"))) - max((item["donor_floor_mass_max"] for item in epsilon_rows), default=0.0)) <= 1e-18, f"{row_label}: donor aggregate floor mass differs")
        if np.any(ambiguous & mapped):
            _array_equal(np.asarray(sp_params["sp"])[ambiguous & mapped] / rec_power[ambiguous & mapped, None], own_shape[ambiguous & mapped], np, f"{row_label}: mapped ambiguous SP must equal own shape", tolerance=0.0)

        expected_q00 = cell_shapes["q00"]
        expected_q11 = cell_shapes["q11"]
        rt_shape = np.asarray(rt_params["sp"], dtype=np.float64) / rec_power[:, None]
        sp_shape = np.asarray(sp_params["sp"], dtype=np.float64) / rec_power[:, None]
        q00_bridge = _mixed_compare(expected_q00, rt_shape, np, f"{row_label}/q00-full-grid-RT-bridge")
        q11_bridge = _mixed_compare(expected_q11, sp_shape, np, f"{row_label}/q11-full-grid-SP-bridge")
        for label, actual, summary in (("q00", q00_token_checks, row["softmax_formula_endpoint_audits"]["q00"]),
                                       ("q11_mapped_shape", q11_map_checks, row["softmax_formula_endpoint_audits"]["q11_mapped_shape"]),
                                       ("q11_saved_SP_endpoint", q11_endpoint_checks, row["softmax_formula_endpoint_audits"]["q11_saved_SP_endpoint"])):
            _expect(summary.get("passed") is True and int(summary.get("comparison_count", -1)) == len(actual), f"{row_label}/{label}: token formula audit summary differs")
            _expect(abs(float(summary.get("max_abs_error", float("nan"))) - max((item["max_abs_error"] for item in actual), default=0.0)) <= 1e-18, f"{row_label}/{label}: formula summary max absolute error differs")
            _expect(abs(float(summary.get("max_relative_error", float("nan"))) - max((item["max_relative_error"] for item in actual), default=0.0)) <= 1e-12, f"{row_label}/{label}: formula summary max relative error differs")
        softmax_summary = row["softmax_formula_endpoint_audits"]
        for key, checked, reference in (("q00_full_grid_RT_endpoint", q00_bridge, rt_shape), ("q11_full_grid_SP_endpoint", q11_bridge, sp_shape)):
            bridge_meta = softmax_summary.get(key, {})
            _expect(bridge_meta.get("passed") is True and int(bridge_meta.get("violating_elements", -1)) == 0, f"{row_label}/{key}: full-grid bridge status differs")
            _expect(bridge_meta.get("reference_shape") == list(reference.shape), f"{row_label}/{key}: full-grid bridge reference shape differs")
            _expect(abs(float(bridge_meta.get("max_abs_error", float("nan"))) - checked["max_abs_error"]) <= 1e-18, f"{row_label}/{key}: full-grid bridge max absolute error differs")
            _expect(abs(float(bridge_meta.get("max_relative_error", float("nan"))) - checked["max_relative_error"]) <= 1e-12, f"{row_label}/{key}: full-grid bridge max relative error differs")
            _expect(float(bridge_meta.get("absolute_tolerance", float("nan"))) == MIXED_ABS_TOL and float(bridge_meta.get("relative_tolerance", float("nan"))) == MIXED_REL_TOL and float(bridge_meta.get("relative_denominator_floor", float("nan"))) == MIXED_REL_FLOOR, f"{row_label}/{key}: full-grid bridge tolerance metadata differs")

        params_meta = row["parameters_npz"]
        params_path = verify_output_ref(params_meta["path"], params_meta["file_sha256"], f"{row_label}/factorized-parameters")
        output_params = _load_npz(params_path, np, OUTPUT_PARAMETER_KEYS, output_parameter_cache)
        _expect(int(np.asarray(output_params["fft_size"]).item()) == int(np.asarray(rt_params["sp"]).shape[1] * 2 - 2), f"{row_label}: stored FFT size mismatch")
        for key, reference in (("f0", rt_params["f0"]), ("ap", rt_params["ap"]), ("time", rt_params["time"]), ("frequency_hz", rt_params["frequency_hz"])):
            check = _array_equal(output_params[key], reference, np, f"{row_label}/output-{key}-recipient-preservation")
            summary = row["unchanged_recipient_factor_checks"].get(key, {})
            _expect(summary.get("bit_exact") is True and summary.get("shape") == check["shape"] and summary.get("dtype") == check["dtype"] and float(summary.get("max_abs_error", float("nan"))) == check["max_abs_error"] == 0.0 and float(summary.get("tolerance", float("nan"))) == 0.0, f"{row_label}/{key}: recipient preservation summary differs")
        _expect(set(row["unchanged_recipient_factor_checks"]) == {"f0", "ap", "time", "frequency_hz"}, f"{row_label}: recipient-preservation summary key set mismatch")
        _expect(row.get("f0_ap_time_identity") == "recipient arrays copied unchanged for all five cells", f"{row_label}: recipient-factor summary text differs")
        _array_equal(output_params["sp_q00"], rt_params["sp"], np, f"{row_label}/q00-stored-endpoint-copy")
        _array_equal(output_params["sp_q11"], sp_params["sp"], np, f"{row_label}/q11-stored-endpoint-copy")
        endpoint_copy = row.get("stored_endpoint_sp_array_copy_checks", {})
        _expect(set(endpoint_copy) == {"q00_to_RT_SP", "q11_to_SP_SP"}, f"{row_label}: stored endpoint-copy summary key set mismatch")
        for copy_key, actual in (("q00_to_RT_SP", output_params["sp_q00"]), ("q11_to_SP_SP", output_params["sp_q11"])):
            check = _array_equal(actual, rt_params["sp"] if copy_key == "q00_to_RT_SP" else sp_params["sp"], np, f"{row_label}/{copy_key}/summary")
            reported = endpoint_copy[copy_key]
            _expect(reported.get("bit_exact") is True and reported.get("shape") == check["shape"] and reported.get("dtype") == check["dtype"] and float(reported.get("max_abs_error", float("nan"))) == 0.0 and float(reported.get("tolerance", float("nan"))) == 0.0, f"{row_label}/{copy_key}: endpoint-copy summary differs")
        _expect(set(params_meta.get("arrays", {})) == OUTPUT_PARAMETER_KEYS, f"{row_label}: parameter array ref inventory mismatch")
        for name in OUTPUT_PARAMETER_KEYS:
            _assert_array_ref(output_params[name], params_meta["arrays"][name], np, f"{row_label}/parameter-ref/{name}")
            reference = params_meta["arrays"][name]
            _expect(reference.get("key") == name and reference.get("path") == params_meta["path"] and reference.get("file_sha256") == params_meta["file_sha256"], f"{row_label}/{name}: parameter reference path/key/file SHA differs")

        old_waves = {}
        for arm in ("RT", "SP"):
            wave_entry = input_entry(f"{sample_id}/{direction}/{arm}/waveform")
            wav, rate = sf.read(str(wave_entry["resolved_path"]), dtype="float32", always_2d=True)
            _expect(int(rate) == SAMPLE_RATE and wav.shape == (recipient_samples, 1) and np.isfinite(wav).all(), f"{row_label}/{arm}: original endpoint WAV clock/decode failed")
            old_waves[arm] = np.asarray(wav[:, 0], dtype=np.float32)
            _expect(abs(float(np.sqrt(np.mean(old_waves[arm].astype(np.float64) ** 2))) - FIXED_RMS) <= RMS_ABS_TOL, f"{row_label}/{arm}: original endpoint RMS differs")

        cell_results = {}
        power_results = {}
        for cell in CELLS:
            sp_value = np.asarray(output_params[f"sp_{cell}"])
            _expect(np.isfinite(sp_value).all() and np.all(sp_value >= 0.0), f"{row_label}/{cell}: invalid SP")
            row_power = np.sum(sp_value.astype(np.float64), axis=1, dtype=np.float64)
            power_rel = np.abs(row_power - rec_power) / np.maximum(np.abs(rec_power), 1e-30)
            power_abs = np.abs(row_power - rec_power)
            _expect(float(np.max(power_rel, initial=0.0)) <= POWER_REL_TOL, f"{row_label}/{cell}: SP row power changed")
            stored_shape = sp_value.astype(np.float64) / rec_power[:, None]
            _expect(float(np.max(np.abs(np.sum(stored_shape, axis=1, dtype=np.float64) - 1.0), initial=0.0)) <= PARAM_ABS_TOL, f"{row_label}/{cell}: SP normalized rows do not sum to one")
            if cell in ("q00", "q11"):
                bridge_shape = expected_q00 if cell == "q00" else expected_q11
                _mixed_compare(bridge_shape, stored_shape, np, f"{row_label}/{cell}-formula-stored-endpoint")
            else:
                _mixed_compare(cell_shapes[cell], stored_shape, np, f"{row_label}/{cell}-formula-stored")

            cell_meta = row["cells"][cell]
            _expect(cell_meta.get("status") == "ok", f"{row_label}/{cell}: producer cell not ok")
            power_meta = cell_meta.get("power_preservation", {})
            _expect(int(power_meta.get("frame_count", -1)) == len(rec_power), f"{row_label}/{cell}: row-power frame count differs")
            _expect(abs(float(power_meta.get("max_abs_power_error", float("nan"))) - float(np.max(power_abs, initial=0.0))) <= 1e-12, f"{row_label}/{cell}: row-power absolute diagnostic differs")
            _expect(abs(float(power_meta.get("max_relative_power_error", float("nan"))) - float(np.max(power_rel, initial=0.0))) <= 1e-12, f"{row_label}/{cell}: row-power relative diagnostic differs")
            power_results[cell] = {"frame_count": len(rec_power), "max_abs_power_error": float(np.max(power_abs, initial=0.0)), "max_relative_power_error": float(np.max(power_rel, initial=0.0))}
            _expect(cell_meta["parameter_npz"]["file_sha256"] == params_meta["file_sha256"] and cell_meta["parameter_npz"]["sp_key"] == f"sp_{cell}", f"{row_label}/{cell}: waveform-to-parameter reference differs")
            _expect(cell_meta["parameter_npz"].get("path") == params_meta["path"] and cell_meta["parameter_npz"].get("sp_array", {}).get("key") == f"sp_{cell}" and cell_meta["parameter_npz"].get("sp_array", {}).get("path") == params_meta["path"] and cell_meta["parameter_npz"].get("sp_array", {}).get("file_sha256") == params_meta["file_sha256"], f"{row_label}/{cell}: nested parameter reference path/key/file SHA differs")
            _assert_array_ref(sp_value, cell_meta["parameter_npz"]["sp_array"], np, f"{row_label}/{cell}/SP-array-reference")
            verify_output_ref(cell_meta["waveform"]["path"], cell_meta["waveform"]["file_sha256"], f"{row_label}/{cell}/waveform")
            wave, wave_readback = _read_wave(cell_meta["waveform"], sf, np, output_root, f"{row_label}/{cell}")
            _expect(len(wave) == recipient_samples, f"{row_label}/{cell}: output waveform sample clock differs from recipient")
            generated, synth_meta = _synthesize(pyworld, {"f0": output_params["f0"], "sp": sp_value, "ap": output_params["ap"]}, recipient_samples, np)
            _expect(np.array_equal(generated, wave), f"{row_label}/{cell}: independently resynthesized float PCM differs")
            if cell in ("q00", "q11"):
                old_arm = "RT" if cell == "q00" else "SP"
                old_rms = float(np.sqrt(np.mean(old_waves[old_arm].astype(np.float64) ** 2)))
                bridge_diff = np.abs(wave.astype(np.float64) - old_waves[old_arm].astype(np.float64))
                bridge_max = float(np.max(bridge_diff, initial=0.0))
                bridge_rms = float(np.sqrt(np.mean(bridge_diff * bridge_diff)))
                bridge_relative_rms = bridge_rms / max(old_rms, 1e-30)
                _expect(bridge_max <= WAVE_MAX_ABS_TOL, f"{row_label}/{cell}: original endpoint waveform bridge exceeds tolerance")
                bridge_meta = cell_meta["endpoint_bridge"]
                old_wave_meta = input_entry(f"{sample_id}/{direction}/{old_arm}/waveform")
                _expect(bridge_meta.get("existing_arm") == old_arm and bridge_meta.get("existing_wave_file_sha256") == old_wave_meta["sha256"] and Path(bridge_meta.get("existing_wave_path", "")).resolve(strict=True) == old_wave_meta["resolved_path"], f"{row_label}/{cell}: endpoint bridge refs differ")
                _expect(bool(bridge_meta.get("bit_identical")) == bool(np.array_equal(wave, old_waves[old_arm])), f"{row_label}/{cell}: endpoint byte identity metadata differs")
                _expect(float(bridge_meta.get("max_abs_tolerance", float("nan"))) == WAVE_MAX_ABS_TOL and abs(float(bridge_meta.get("max_abs_error", float("nan"))) - bridge_max) <= 1e-12 and abs(float(bridge_meta.get("rms_error", float("nan"))) - bridge_rms) <= 1e-12, f"{row_label}/{cell}: endpoint bridge error metadata differs")
                repeated, _ = _synthesize(pyworld, {"f0": output_params["f0"], "sp": sp_value, "ap": output_params["ap"]}, recipient_samples, np)
                _expect(np.array_equal(repeated, generated), f"{row_label}/{cell}: repeated WORLD synthesis is not bit-identical")
                _expect(bridge_meta.get("repeat_bit_identical") is True and bridge_meta.get("repeat_pcm_sha256") == hashlib.sha256(repeated.astype("<f4", copy=False).tobytes(order="C")).hexdigest(), f"{row_label}/{cell}: repeat metadata differs")
                repeat_meta = bridge_meta.get("repeat_synthesis", {})
                for repeat_key, actual_value in (("synthesized_samples", recipient_samples + synth_meta["right_trim_samples"]), ("recipient_samples", recipient_samples), ("right_discarded_tail_samples", synth_meta["right_trim_samples"])):
                    _expect(int(repeat_meta.get(repeat_key, -1)) == int(actual_value), f"{row_label}/{cell}: repeat synthesis {repeat_key} differs")
                for repeat_key, actual_value, tol in (("fixed_rms", synth_meta["fixed_rms"], 1e-12), ("peak_abs", synth_meta["peak_abs"], 1e-12), ("input_synthesis_rms_before_global_fix", synth_meta["input_rms"], 1e-12), ("global_gain_to_fixed_rms", synth_meta["gain"], 1e-12), ("global_gain_db", synth_meta["gain_db"], 1e-10), ("discarded_tail_rms", synth_meta["discarded_tail_rms"], 1e-12)):
                    _expect(abs(float(repeat_meta.get(repeat_key, float("nan"))) - actual_value) <= tol, f"{row_label}/{cell}: repeat synthesis {repeat_key} differs")
                _expect(repeat_meta.get("status") == "ok" and int(repeat_meta.get("samples_over_full_scale", -1)) == synth_meta["samples_over_full_scale"], f"{row_label}/{cell}: repeat synthesis status/clipping differs")
            synth_record = cell_meta["synthesis"]
            _expect(synth_record.get("status") == "ok" and int(synth_record.get("synthesized_samples", -1)) == synth_meta["generated_samples"] and int(synth_record.get("recipient_samples", -1)) == recipient_samples and int(synth_record.get("right_discarded_tail_samples", -1)) == synth_meta["right_trim_samples"], f"{row_label}/{cell}: synthesis clock record differs")
            _expect(abs(float(synth_record.get("fixed_rms", float("nan"))) - synth_meta["fixed_rms"]) <= 1e-12 and abs(float(synth_record.get("peak_abs", float("nan"))) - synth_meta["peak_abs"]) <= 1e-12, f"{row_label}/{cell}: synthesis RMS/peak record differs")
            _expect(abs(float(synth_record.get("input_synthesis_rms_before_global_fix", float("nan"))) - synth_meta["input_rms"]) <= 1e-12, f"{row_label}/{cell}: synthesis input RMS diagnostic differs")
            _expect(abs(float(synth_record.get("global_gain_to_fixed_rms", float("nan"))) - synth_meta["gain"]) <= 1e-12 and abs(float(synth_record.get("global_gain_db", float("nan"))) - synth_meta["gain_db"]) <= 1e-10, f"{row_label}/{cell}: synthesis gain diagnostic differs")
            _expect(abs(float(synth_record.get("discarded_tail_rms", float("nan"))) - synth_meta["discarded_tail_rms"]) <= 1e-12, f"{row_label}/{cell}: discarded tail diagnostic differs")
            _expect(abs(synth_meta["fixed_rms"] - FIXED_RMS) <= RMS_ABS_TOL and int(synth_record.get("samples_over_full_scale", -1)) == synth_meta["samples_over_full_scale"], f"{row_label}/{cell}: RMS or clipping count differs")
            cell_results[cell] = {"wave_file_sha256": cell_meta["waveform"]["file_sha256"], "wave_float32_pcm_sha256": wave_readback["float32_pcm_sha256"],
                                  "parameter_sp_array_sha256": array_sha(sp_value, np), "decoded_samples": len(wave), "rms": wave_readback["rms"],
                                  "fixed_rms_abs_error": abs(wave_readback["rms"] - FIXED_RMS), "sp_power_max_abs_error": float(np.max(power_abs, initial=0.0)),
                                  "sp_power_max_relative_error": float(np.max(power_rel, initial=0.0)), "right_trim_samples": synth_meta["right_trim_samples"],
                                  "independent_synthesis_bit_exact": True}
            if cell in ("q00", "q11"):
                cell_results[cell]["endpoint_waveform_bridge"] = {"bit_identical": bool(np.array_equal(wave, old_waves["RT" if cell == "q00" else "SP"])), "max_abs_error": bridge_max, "rms_error": bridge_rms, "relative_rms_error": bridge_relative_rms}
            cell_total += 1

        reported_power = row.get("sp_power_preservation", {})
        _expect(set(reported_power) == set(CELLS), f"{row_label}: row-level SP power summary cell set mismatch")
        for cell, actual in power_results.items():
            reported = reported_power[cell]
            _expect(int(reported.get("frame_count", -1)) == actual["frame_count"] and float(reported.get("relative_tolerance", float("nan"))) == POWER_REL_TOL, f"{row_label}/{cell}: row-level SP power summary clock/tolerance differs")
            _expect(abs(float(reported.get("max_abs_power_error", float("nan"))) - actual["max_abs_power_error"]) <= 1e-12 and abs(float(reported.get("max_relative_power_error", float("nan"))) - actual["max_relative_power_error"]) <= 1e-12, f"{row_label}/{cell}: row-level SP power summary values differ")

        reported_labels = row.get("token_label_summary", {})
        _expect(set(reported_labels) == set(token_label_acc), f"{row_label}: per-label summary inventory differs from matched tokens")
        for label, actual_label in token_label_acc.items():
            reported_label = reported_labels[label]
            _expect(reported_label.get("direction") == direction and bool(reported_label.get("frozen16_inference_member")) == (label in phone_labels), f"{row_label}/{label}: per-label direction/frozen membership differs")
            _expect(int(reported_label.get("matched_token_count", -1)) == actual_label["matched"] and int(reported_label.get("eligible_token_count", -1)) == actual_label["eligible"], f"{row_label}/{label}: per-label token counts differ")
            reported_label_cells = reported_label.get("cells", {})
            _expect(set(reported_label_cells) == set(CELLS), f"{row_label}/{label}: per-label cell set differs")
            for cell in CELLS:
                actual_dynamic = actual_label["dynamics"][cell]
                reported_dynamic = reported_label_cells[cell]
                expected_mean = float(np.mean(actual_dynamic["means"], dtype=np.float64)) if actual_dynamic["means"] else None
                _expect(int(reported_dynamic.get("token_count_with_adjacent_pairs", -1)) == actual_dynamic["with_pairs"], f"{row_label}/{label}/{cell}: per-label adjacent-token count differs")
                reported_mean = reported_dynamic.get("token_unweighted_mean_adjacent_logshape_delta")
                _expect((expected_mean is None and reported_mean is None) or (expected_mean is not None and reported_mean is not None and abs(float(reported_mean) - expected_mean) <= 1e-12), f"{row_label}/{label}/{cell}: per-label dynamics aggregate differs")

        qa_rows.append({"ordered_index": row_index, "id": sample_id, "speaker": row["speaker"], "direction": direction,
                        "recipient_condition": rec_condition, "donor_condition": donor_condition, "status": "PASS",
                        "world_frames": len(times), "mapped_frames": int(np.count_nonzero(mapped)), "eligible_frames": int(np.count_nonzero(eligible_mask)),
                        "ambiguous_mapped_frames": int(np.count_nonzero(ambiguous & mapped)), "phone_tokens": len(tokens),
                        "transfer_arrays_rebuilt": len(audit_checks), "q00_full_grid_bridge": q00_bridge, "q11_full_grid_bridge": q11_bridge,
                        "all_cells": cell_results, "token_dynamics_verified": token_dynamics,
                        "epsilon_floor_token_audit_count": len(epsilon_rows)})
        total_token_rows += len(tokens)

    _expect(cell_total == 200 and len(qa_rows) == 40, f"independent QA coverage mismatch: directions={len(qa_rows)} cells={cell_total}")
    _expect(len({row["id"] for row in qa_rows}) == 20 and len({row["speaker"] for row in qa_rows}) == 10, "independent QA cohort uniqueness mismatch")
    result = {
        "schema_version": 1,
        "manifest_type": "world_phone_shape_mean_dynamic_cal_independent_reconstruction_qa",
        "status": "PASS",
        "scope": "CAL only; 20 utterances / 10 speakers / 40 directions / 200 WORLD waveforms",
        "evaluation_inputs_read": False,
        "evaluation_outcomes_read": False,
        "effects_read": False,
        "model_forward_performed": False,
        "input_report": {"path": str(report_path), "raw_sha256": REPORT_RAW_SHA256, "payload_sha256": REPORT_PAYLOAD_SHA256},
        "input_output_inventory": {"path": str(inventory_path), "payload_sha256": FILE_INVENTORY_PAYLOAD_SHA256, "entries_verified": len(inventory_rows)},
        "input_output_size_inventory": {"path": str(size_inventory_path), "raw_sha256": OUTPUT_SIZE_INVENTORY_RAW_SHA256, "entries_verified": len(size_rows), "total_bytes": 801875807},
        "bound_world_manifest_raw_sha256": WORLD_MANIFEST_RAW_SHA256,
        "bound_world_plan_raw_sha256_only_without_deserialization": WORLD_PLAN_RAW_SHA256,
        "bound_world_source": {"path": str(world_source_path), "sha256": WORLD_SOURCE_SHA256},
        "bound_analysis_source_sha256": ANALYSIS_SOURCE_SHA256,
        "bound_analysis_spec_sha256": ANALYSIS_SPEC_SHA256,
        "bound_analysis_plan_raw_sha256": ANALYSIS_PLAN_RAW_SHA256,
        "bound_analysis_plan_payload_sha256": ANALYSIS_PLAN_PAYLOAD_SHA256,
        "independent_qa_source_path": str(Path(__file__).resolve()),
        "independent_qa_source_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": {"python": sys.version, "numpy": np.__version__, "soundfile": sf.__version__, "pyworld": pyworld.__version__,
                    "pyworld_extension_sha256": sha256_file(ext_path), "fixed_rms": FIXED_RMS, "sample_rate_hz": SAMPLE_RATE, "frame_period_ms": 5.0},
        "input_files_verified": len(input_roles),
        "output_files_verified": len(inventory_rows),
        "direction_rows_verified": len(qa_rows),
        "cell_waveforms_verified": cell_total,
        "transfer_audit_arrays_rebuilt": transfer_array_count,
        "exact_phone_tokens_verified": total_token_rows,
        "records": qa_rows,
    }
    result["manifest_sha256"] = payload_sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-inventory", required=True, type=Path)
    parser.add_argument("--output-size-inventory", required=True, type=Path)
    parser.add_argument("--world-source", type=Path, default=Path(WORLD_SOURCE_DEFAULT))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _expect(not args.out.exists(), f"refusing to overwrite QA output: {args.out}")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    print(json.dumps({"status": result["status"], "directions": result["direction_rows_verified"], "cells": result["cell_waveforms_verified"],
                      "tokens": result["exact_phone_tokens_verified"], "manifest_sha256": result["manifest_sha256"], "path": str(args.out)}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        raise
    raise SystemExit(code)
