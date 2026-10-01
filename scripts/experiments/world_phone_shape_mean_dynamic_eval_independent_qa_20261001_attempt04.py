#!/usr/bin/env python3
"""Independent, outcome-blind reconstruction QA for the sealed shape-factor EVAL waveforms.

This consumer does not import the EVAL producer. It rebuilds phone-map transfer,
token domains, mean/residual shapes, SELF_GRID interpolation, all five WORLD
parameter sets, and synthesized waveforms from sealed physical EVAL inputs. It
does not read model scores, videos, SyncNet outputs, or scientific effects.
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


EXPECTED_REPORT_RAW_SHA256: str | None = None
EXPECTED_OUTPUT_INVENTORY_RAW_SHA256: str | None = None
PHYSICAL_MANIFEST_RAW_SHA256 = "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3"
PHYSICAL_MANIFEST_PAYLOAD_SHA256 = "eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa7ed9de8"
PHYSICAL_QA_RAW_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
PHYSICAL_PLAN_RAW_SHA256 = "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e"
PHYSICAL_PLAN_PAYLOAD_SHA256 = "bfae127eed35c459d1f6a2fff8feebac4f4b5d855b1f26611141138410f22b24"
SOURCE_INPUT_MANIFEST_RAW_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
SOURCE_INPUT_MANIFEST_PAYLOAD_SHA256 = "b18f83ad07a660e6c0bdef3b8780c59b7b12497370a79916eb95c7dfe07303bf"
TTS_PAIR_MANIFEST_RAW_SHA256 = "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2"
TTS_PAIR_MANIFEST_PAYLOAD_SHA256 = "e0b6b2c69db11703cc505a8d4627287d307825609ecfb65383807587ed0ac70f"
WORLD_SOURCE_SHA256 = "7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2"
WORLD_SOURCE_DEFAULT = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/hf_selected/world_phone_transfer_20261001.py"
EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
PHONE_TABLE_RAW_SHA256 = "1673092554bfc03dbc385136c3a66c08821499c0ec6c25602d385d1d0d7cf16f"
PHONE_TABLE_PAYLOAD_SHA256 = "586d86fedd159ba1c654586bf6892d87b36b1ed8d88c666217d648d230fb2259"
PHONE_INVENTORY_RAW_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"
PHONE_INVENTORY_PAYLOAD_SHA256 = "54a51ff1a9af891271b0eaed797dc0bb7508a338ef7e38f922b8f9292fa40f5d"
CAL_WORLD_MANIFEST_RAW_SHA256 = "dc23a765794b0dd6e735be26955dbcb39e4ce06dff4fbe1be9f1c6b926ab7efa"
CAL_WORLD_MANIFEST_PAYLOAD_SHA256 = "f3b8045245edcf21ebc30c944f61318f8ac3538669152d64f68c882543d75a1f"
CAL_WORLD_PLAN_RAW_SHA256 = "594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7"
CAL_WORLD_PLAN_PAYLOAD_SHA256 = "85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a"
CAL_WORLD_SPEC_SHA256 = "01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9"
CAL_QA_RAW_SHA256 = "ae2d5631c3855731e46518fae44b8c35e3542ac0be9be5b56d484a4c0661b292"
CAL_QA_PAYLOAD_SHA256 = "7a16848f106527abc10cc6e68ecc78402a59d5e3a73696f94b2994fdea2bb96a"
CAL_QA_SOURCE_SHA256 = "94688ef9adce2286c374da95da1d2eebbada3b7ef4b3ebe145e3737fe1a7cfe9"
CAL_ROOT_FREEZE_RAW_SHA256 = "14c8c1244e176bb9e2172dfeba0f8dcd78e67c56f4645ec0cc180d4e08dffa20"
CAL_ROOT_FREEZE_PAYLOAD_SHA256 = "892f537848c06887f8072c9ec3a1ddbbcba8dd2eca5d9389377cab3b0f8b0ac3"
EVAL_DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
METRIC_ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
EXPECTED_INPUT_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
QA_REPAIR_AUTHORIZATION_DEFAULT = "/root/tts-research-system-20261001/world_phone_shape_mean_dynamic_eval_20261001_attempt01/control/independent_qa_execution_repair_source04.json"
QA_REPAIR_SCHEMA = "ROOT_APPROVED_EVAL_QA_EXECUTION_REPAIR_V1"
QA_REPAIR_CHANGES = [
    "phone inventory payload binding: compare the recomputed inventory_sha256 with frozen input_bindings.phone_inventory.payload_sha256 and payload_field inventory_sha256",
    "QA source self-binding: retain the plan-bound QA03 hash; authorize QA04 only through this root-approved repair manifest",
]
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


def _load_and_verify_report(args: argparse.Namespace, np: Any) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = args.manifest.resolve(strict=True)
    inventory_path = args.output_inventory.resolve(strict=True)
    report_raw = sha256_file(report_path)
    _expect(args.expected_manifest_raw_sha256 is not None and report_raw == args.expected_manifest_raw_sha256,
            "EVAL report does not match the explicitly supplied raw-byte SHA")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload = payload_sha(report)
    _expect(report.get("manifest_sha256") == report_payload, "EVAL report embedded payload fingerprint mismatch")
    _expect(report.get("status") == "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW", "shape-factor EVAL producer report is not complete")
    _expect(report.get("split") == "evaluation" and report.get("scope") == "evaluation_physical_waveform_production_only", "EVAL report split/scope mismatch")
    for flag in ("evaluation_outcomes_read", "evaluation_effects_read", "effects_read", "model_forward_performed"):
        _expect(report.get(flag) is False, f"EVAL producer report records prohibited access: {flag}")
    _expect(report.get("evaluation_waveforms_read") is True, "EVAL producer report does not disclose physical waveform reads")
    _expect(report.get("intended_pairs") == 80 and report.get("intended_speakers") == 40 and
            report.get("intended_direction_rows") == 160 and report.get("complete_direction_rows") == 160 and
            report.get("intended_cell_waveforms") == 800 and report.get("complete_cell_waveforms") == 800 and
            report.get("failure_count") == 0 and report.get("failures") == [], "EVAL producer did not complete the frozen 80x2x5 cohort")
    _expect(report.get("direction_row_identity_complete") is True and report.get("output_inventory_complete") is True and
            report.get("output_file_count") == 1120 and report.get("output_wav_count") == 800 and report.get("output_npz_count") == 320,
            "EVAL producer output inventory count gates are incomplete")
    _expect(len(report.get("records", [])) == 160 and len(report.get("input_inventory", [])) == 1440 and
            report.get("input_file_count_verified") == 1440, "EVAL report row/input counts differ")

    bindings = report.get("input_bindings", {})
    expected_raw = {
        "physical_manifest": PHYSICAL_MANIFEST_RAW_SHA256,
        "physical_plan": PHYSICAL_PLAN_RAW_SHA256,
        "physical_qa": PHYSICAL_QA_RAW_SHA256,
        "source_input_manifest": SOURCE_INPUT_MANIFEST_RAW_SHA256,
        "tts_pair_manifest": TTS_PAIR_MANIFEST_RAW_SHA256,
        "world_source": WORLD_SOURCE_SHA256,
        "execution_bindings": EXECUTION_BINDINGS_SHA256,
        "phone_table": PHONE_TABLE_RAW_SHA256,
        "phone_inventory": PHONE_INVENTORY_RAW_SHA256,
        "calibration_world_manifest": CAL_WORLD_MANIFEST_RAW_SHA256,
        "calibration_world_plan": CAL_WORLD_PLAN_RAW_SHA256,
        "calibration_world_spec": CAL_WORLD_SPEC_SHA256,
        "calibration_qa": CAL_QA_RAW_SHA256,
        "calibration_freeze": CAL_ROOT_FREEZE_RAW_SHA256,
    }
    for role, expected in expected_raw.items():
        _expect(bindings.get(role, {}).get("sha256") == expected, f"EVAL report input binding differs for {role}")
    _expect(bindings.get("evaluation_design", {}).get("sha256") == EVAL_DESIGN_SHA256 and
            bindings.get("metric_addendum", {}).get("sha256") == METRIC_ADDENDUM_SHA256, "EVAL design/addendum source binding differs")
    _expect(bindings.get("evaluation_freeze", {}).get("path") == str(args.evaluation_freeze.resolve(strict=True)), "EVAL report root-freeze path differs from CLI binding")
    _expect(bindings.get("analysis_plan") is None, "analysis plan must be top-level, not duplicated as an input binding")
    for role, binding in bindings.items():
        _verified_file(binding["path"], binding["sha256"], binding.get("bytes"))

    plan_path = args.analysis_plan.resolve(strict=True)
    analysis_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_payload = payload_sha(analysis_plan)
    planned_inputs = analysis_plan.get("input_bindings", {})
    for role, binding in bindings.items():
        if role == "evaluation_freeze":
            continue
        planned = planned_inputs.get(role)
        _expect(isinstance(planned, dict) and planned.get("raw_sha256") == binding.get("sha256") and
                Path(str(planned.get("path", ""))).resolve() == Path(binding["path"]).resolve(),
                f"EVAL report input binding differs from the frozen plan: {role}")
    _expect(analysis_plan.get("manifest_sha256") == plan_payload, "frozen EVAL plan payload fingerprint mismatch")
    _expect(report.get("analysis_plan_path") == str(plan_path) and report.get("analysis_plan_raw_sha256") == sha256_file(plan_path) and
            report.get("analysis_plan_payload_sha256") == plan_payload, "EVAL report plan path/raw/payload identity differs")
    _expect(analysis_plan.get("analysis_source_sha256") == report.get("analysis_source_sha256"), "analysis plan/report producer-source binding differs")
    source_path = _verified_file(report.get("analysis_source_path", ""), str(report.get("analysis_source_sha256", "")))
    _expect(Path(analysis_plan.get("analysis_source_path", "")).resolve(strict=True) == source_path,
            "EVAL producer source path disagrees with the frozen plan")
    plan_bound_qa_source = planned_inputs.get("independent_qa_source")
    _expect(isinstance(plan_bound_qa_source, dict) and
            analysis_plan.get("independent_qa_source_sha256") == plan_bound_qa_source.get("raw_sha256") and
            Path(str(plan_bound_qa_source.get("path", ""))).resolve() == Path(bindings["independent_qa_source"]["path"]).resolve(),
            "frozen plan/report binding for original QA03 source differs")
    _expect(analysis_plan.get("scope") == "evaluation_only" and analysis_plan.get("split") == "evaluation" and
            analysis_plan.get("evaluation_outcomes_read") is False and analysis_plan.get("evaluation_effects_read") is False and
            analysis_plan.get("model_forward_performed") is False, "frozen EVAL plan records prohibited access or wrong scope")
    _expect(analysis_plan.get("input_identity_sha256") == EXPECTED_INPUT_IDENTITY_SHA256, "EVAL plan input identity differs")
    _expect(analysis_plan.get("directions") == list(CONDITIONS) and analysis_plan.get("cells") == list(CELLS), "EVAL plan direction/cell order differs")
    _expect(analysis_plan.get("intended_pair_count") == 80 and analysis_plan.get("intended_speaker_count") == 40 and
            analysis_plan.get("intended_direction_rows") == 160 and analysis_plan.get("intended_cell_waveforms") == 800, "EVAL plan cohort counts differ")

    freeze_path = args.evaluation_freeze.resolve(strict=True)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _expect(freeze.get("status") == "FROZEN" and freeze.get("evaluation_waveform_production_authorized") is True, "root EVAL freeze does not authorize this waveform plan")
    _expect(freeze.get("analysis_plan_sha256") == plan_payload and freeze.get("analysis_plan_raw_sha256") == sha256_file(plan_path), "root EVAL freeze does not bind the exact plan")
    _expect(freeze.get("analysis_source_sha256") == report.get("analysis_source_sha256"), "root EVAL freeze does not bind the executed producer source")
    _expect(freeze.get("evaluation_effects_read") is False and freeze.get("evaluation_outcomes_read") is False, "root EVAL freeze records forbidden outcome/effect access")
    _expect(bindings["evaluation_freeze"]["sha256"] == sha256_file(freeze_path), "EVAL report root-freeze file SHA differs")

    qa_repair_path = args.qa_repair_authorization.resolve(strict=True)
    _expect(qa_repair_path == Path(QA_REPAIR_AUTHORIZATION_DEFAULT).resolve(strict=True),
            "QA repair authorization path differs from the frozen control path")
    qa_repair_raw_sha = sha256_file(qa_repair_path)
    _expect(args.expected_qa_repair_authorization_raw_sha256 == qa_repair_raw_sha,
            "QA repair authorization does not match the explicitly supplied raw-byte SHA")
    qa_repair = json.loads(qa_repair_path.read_text(encoding="utf-8"))
    qa_repair_payload_sha = payload_sha(qa_repair)
    current_qa_source_path = Path(__file__).resolve(strict=True)
    current_qa_source_sha = sha256_file(current_qa_source_path)
    original_qa_binding = planned_inputs.get("independent_qa_source", {})
    _expect(qa_repair.get("schema") == QA_REPAIR_SCHEMA and qa_repair.get("status") == "APPROVED" and
            qa_repair.get("scope") == "EVAL_PHYSICAL_WAVEFORMS_ONLY" and qa_repair.get("manifest_sha256") == qa_repair_payload_sha,
            "root-approved QA repair authorization schema/status/payload differs")
    _expect(isinstance(qa_repair.get("authorized_at_utc"), str) and bool(qa_repair["authorized_at_utc"].strip()),
            "QA repair authorization timestamp is missing")
    original_plan = qa_repair.get("original_analysis_plan", {})
    original_freeze = qa_repair.get("original_evaluation_freeze", {})
    original_qa = qa_repair.get("original_qa_source", {})
    repaired_qa = qa_repair.get("repaired_qa_source", {})
    _expect(original_plan.get("path") == str(plan_path) and original_plan.get("raw_sha256") == sha256_file(plan_path) and
            original_plan.get("payload_sha256") == plan_payload,
            "QA repair authorization does not bind the immutable executed analysis plan")
    _expect(original_freeze.get("path") == str(freeze_path) and original_freeze.get("raw_sha256") == sha256_file(freeze_path) and
            original_freeze.get("payload_sha256") == payload_sha(freeze),
            "QA repair authorization does not bind the immutable executed evaluation freeze")
    _expect(original_qa.get("path") == original_qa_binding.get("path") and
            original_qa.get("raw_sha256") == analysis_plan.get("independent_qa_source_sha256") == original_qa_binding.get("raw_sha256"),
            "QA repair authorization does not bind the original plan-bound QA03 source")
    _expect(repaired_qa.get("path") == str(current_qa_source_path) and repaired_qa.get("raw_sha256") == current_qa_source_sha,
            "QA repair authorization does not bind this exact QA04 source")
    _expect(qa_repair.get("repair_changes") == QA_REPAIR_CHANGES and qa_repair.get("scientific_controls_changed") is False,
            "QA repair authorization declares unapproved changes or altered scientific controls")
    for key in ("evaluation_outcomes_read", "evaluation_effects_read", "model_forward_authorized", "gpu_forward_authorized"):
        _expect(qa_repair.get(key) is False, f"QA repair authorization records prohibited access/authority: {key}")
    _expect(qa_repair.get("evaluation_inputs_read") is True,
            "QA repair authorization does not disclose the EVAL waveform input read")
    qa_repair_record = {
        "path": str(qa_repair_path), "raw_sha256": qa_repair_raw_sha,
        "payload_sha256": qa_repair_payload_sha, "schema": qa_repair.get("schema"),
        "status": qa_repair.get("status"),
        "original_qa_source_path": original_qa.get("path"),
        "original_qa_source_sha256": original_qa.get("raw_sha256"),
        "repaired_qa_source_path": repaired_qa.get("path"),
        "repaired_qa_source_sha256": repaired_qa.get("raw_sha256"),
        "repair_changes": list(QA_REPAIR_CHANGES),
    }

    cal_freeze_path = Path(bindings["calibration_freeze"]["path"]).resolve(strict=True)
    cal_freeze = json.loads(cal_freeze_path.read_text(encoding="utf-8"))
    _expect(cal_freeze.get("status") == "FROZEN" and
            cal_freeze.get("manifest_sha256") == CAL_ROOT_FREEZE_PAYLOAD_SHA256 and payload_sha(cal_freeze) == CAL_ROOT_FREEZE_PAYLOAD_SHA256 and
            cal_freeze.get("calibration_producer_manifest_raw_sha256") == CAL_WORLD_MANIFEST_RAW_SHA256 and
            cal_freeze.get("calibration_independent_qa_raw_sha256") == CAL_QA_RAW_SHA256,
            "root CAL freeze does not bind the completed shape-factor CAL producer/QA")
    cal_report = json.loads(Path(bindings["calibration_world_manifest"]["path"]).read_text(encoding="utf-8"))
    _expect(cal_report.get("manifest_sha256") == CAL_WORLD_MANIFEST_PAYLOAD_SHA256 and payload_sha(cal_report) == CAL_WORLD_MANIFEST_PAYLOAD_SHA256,
            "shape-factor CAL producer payload fingerprint differs")
    cal_qa = json.loads(Path(bindings["calibration_qa"]["path"]).read_text(encoding="utf-8"))
    _expect(cal_qa.get("manifest_sha256") == CAL_QA_PAYLOAD_SHA256 and payload_sha(cal_qa) == CAL_QA_PAYLOAD_SHA256 and cal_qa.get("status") == "PASS",
            "shape-factor CAL independent QA is not the reviewed PASS report")
    for key, expected in (("direction_rows_verified", 40), ("cell_waveforms_verified", 200),
                          ("input_files_verified", 320), ("output_files_verified", 280)):
        _expect(cal_qa.get(key) == expected, f"shape-factor CAL QA {key} differs")
    for key in ("evaluation_inputs_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"):
        _expect(cal_qa.get(key) is False, f"shape-factor CAL QA records prohibited access: {key}")
    _expect(cal_report.get("status") == "CAL_FACTORIAL_WAVEFORMS_READY_FOR_ROOT_REVIEW" and cal_report.get("split") == "calibration",
            "shape-factor CAL report is incomplete or wrong-split")
    _expect(cal_report.get("evaluation_waveforms_read") is False and cal_report.get("evaluation_outcomes_read") is False and
            cal_report.get("effects_read") is False and cal_report.get("model_forward_performed") is False,
            "shape-factor CAL producer records prohibited EVAL/model access")
    cal_plan = json.loads(Path(bindings["calibration_world_plan"]["path"]).read_text(encoding="utf-8"))
    _expect(payload_sha(cal_plan) == CAL_WORLD_PLAN_PAYLOAD_SHA256 and cal_plan.get("scope") == "calibration_only" and
            cal_plan.get("evaluation_waveforms_read") is False and cal_plan.get("evaluation_outcomes_read") is False,
            "shape-factor CAL source plan is not the exact CAL-only plan")

    physical = json.loads(Path(bindings["physical_manifest"]["path"]).read_text(encoding="utf-8"))
    physical_plan = json.loads(Path(bindings["physical_plan"]["path"]).read_text(encoding="utf-8"))
    physical_qa = json.loads(Path(bindings["physical_qa"]["path"]).read_text(encoding="utf-8"))
    _expect(sha256_file(Path(bindings["physical_manifest"]["path"])) == PHYSICAL_MANIFEST_RAW_SHA256 and
            physical.get("manifest_sha256") == PHYSICAL_MANIFEST_PAYLOAD_SHA256 and payload_sha(physical) == PHYSICAL_MANIFEST_PAYLOAD_SHA256,
            "parent physical manifest raw/payload identity differs")
    _expect(physical.get("manifest_type") == "world_phone_transfer_evaluation_producer" and physical.get("split") == "evaluation" and
            physical.get("status") == "EVALUATION_WAVEFORMS_READY" and physical.get("evaluation_waveforms_read") is True and
            physical.get("evaluation_outcomes_read") is False, "parent physical manifest is not the sealed EVAL input")
    _expect(physical.get("world_transfer_plan_sha256") == PHYSICAL_PLAN_PAYLOAD_SHA256 and
            physical_plan.get("manifest_sha256") == PHYSICAL_PLAN_PAYLOAD_SHA256 and payload_sha(physical_plan) == PHYSICAL_PLAN_PAYLOAD_SHA256,
            "parent physical plan payload binding differs")
    _expect(physical_plan.get("manifest_type") == "world_phone_transfer_plan" and physical_plan.get("status") == "SEALED_BEFORE_ENDPOINT_FORWARD",
            "parent physical plan does not match sealed EVAL waveform run")
    _expect(physical_qa.get("status") == "PASS" and physical_qa.get("failures") == [] and
            physical_qa.get("classification") == "independent EVAL waveform/parameter-dose/local-energy QA; no SyncNet or visual scores read",
            "parent physical waveform-only QA is not PASS")
    for key, expected in (("sample_pairs", 80), ("recipient_contexts", 160), ("completed_recipient_contexts", 160),
                          ("original_mfa_qualified_pairs", 79), ("physical_overlay_qualified_pairs", 80),
                          ("expected_physical_wavs", 1920), ("expected_npz_artifacts", 1920)):
        _expect(physical_qa.get("coverage", {}).get(key) == expected, f"parent physical QA coverage differs: {key}")
    for flag in ("evaluation_effects_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"):
        if flag in physical_qa:
            _expect(physical_qa[flag] is False, f"parent physical QA records prohibited access: {flag}")

    source_input_manifest = json.loads(Path(bindings["source_input_manifest"]["path"]).read_text(encoding="utf-8"))
    tts_pair_manifest = json.loads(Path(bindings["tts_pair_manifest"]["path"]).read_text(encoding="utf-8"))
    _expect(source_input_manifest.get("status") == "complete" and source_input_manifest.get("records_expected") == 100 and
            source_input_manifest.get("frozen_before_tts") is True and tts_pair_manifest.get("status") == "complete" and
            tts_pair_manifest.get("records_expected") == 100 and tts_pair_manifest.get("split_counts") == {"calibration": 20, "evaluation": 80},
            "source/TTS input manifests are not the frozen 20/80 source cohort")
    source_rows = source_input_manifest.get("records", [])
    tts_rows = tts_pair_manifest.get("records", [])
    _expect(len(source_rows) == 100 and len(tts_rows) == 100, "source/TTS input manifest row count differs")
    identities = []
    source_by_id: dict[str, dict[str, Any]] = {}
    tts_by_id: dict[str, dict[str, Any]] = {}
    for source_row, tts_row in zip(source_rows, tts_rows):
        identity = {key: source_row[key] for key in ("id", "speaker", "text", "split")}
        identity["ref_sha256"] = source_row["ref_sha256"]
        identities.append(identity)
        sample_id = str(source_row["id"])
        _expect(sample_id not in source_by_id and sample_id == str(tts_row.get("id")), "source/TTS IDs are duplicated or out of order")
        _expect(tts_row.get("input_identity") == identity and
                tts_row.get("input_identity_sha256") == hashlib.sha256(canonical_bytes(identity)).hexdigest(),
                f"{sample_id}: source/TTS input identity differs")
        _expect(tts_row.get("natural_raw", {}).get("sha256") == source_row.get("ref_sha256") and
                bool(tts_row.get("qwen_canonical_16k", {}).get("sha256")), f"{sample_id}: source/TTS audio identity is incomplete")
        source_by_id[sample_id] = source_row
        tts_by_id[sample_id] = tts_row
    identity_sha = hashlib.sha256(canonical_bytes(identities)).hexdigest()
    _expect(identity_sha == EXPECTED_INPUT_IDENTITY_SHA256 and tts_pair_manifest.get("input_identity_sha256") == identity_sha,
            "source/TTS manifest whole-cohort identity differs")
    _expect(len(physical.get("records", [])) == 80, "physical manifest does not contain 80 EVAL pairs")
    for physical_row in physical["records"]:
        sample_id = str(physical_row.get("id"))
        source_row, tts_row = source_by_id.get(sample_id), tts_by_id.get(sample_id)
        _expect(source_row is not None and tts_row is not None and source_row.get("split") == "evaluation" and
                physical_row.get("split") == "evaluation" and physical_row.get("speaker") == source_row.get("speaker") == tts_row.get("speaker"),
                f"{sample_id}: physical pair differs from the frozen source/TTS manifests")
        identity = {key: source_row[key] for key in ("id", "speaker", "text", "split")}
        identity["ref_sha256"] = source_row["ref_sha256"]
        row_sha = hashlib.sha256(canonical_bytes(identity)).hexdigest()
        _expect(physical_row.get("source_identity_sha256") == row_sha, f"{sample_id}: physical source identity digest differs")
        for condition, expected_sha in (("natural", source_row["ref_sha256"]), ("tts", tts_row["qwen_canonical_16k"]["sha256"])):
            side = physical_row.get(condition, {})
            _expect(side.get("source_audio_sha256") == expected_sha and side.get("source_identity_sha256") == row_sha,
                    f"{sample_id}/{condition}: physical source audio identity differs")

    phone_inventory = json.loads(Path(bindings["phone_inventory"]["path"]).read_text(encoding="utf-8"))
    phone_payload = hashlib.sha256(canonical_bytes({k: v for k, v in phone_inventory.items() if k != "inventory_sha256"})).hexdigest()
    _expect(phone_inventory.get("status") == "FROZEN_CAL_ONLY_PHONE_INVENTORY" and phone_inventory.get("selection_split") == "calibration" and
            phone_inventory.get("evaluation_textgrids_or_endpoints_read") is False and phone_inventory.get("inventory_sha256") == phone_payload,
            "frozen inference phone inventory is not CAL-only or its payload hash differs")
    _expect(analysis_plan.get("inferential_phone_labels") == phone_inventory.get("selected_phone_labels"), "EVAL plan changes the frozen CAL-only label family")
    _expect(report.get("factorization", {}).get("inference_labels") == phone_inventory.get("selected_phone_labels"), "report label family differs from frozen CAL inventory")
    _expect(analysis_plan.get("ordered_evaluation_ids") == [str(row.get("id")) for row in physical.get("records", [])], "EVAL plan ordering differs from parent physical manifest")

    inventory_raw = sha256_file(inventory_path)
    _expect(args.expected_output_inventory_raw_sha256 is not None and inventory_raw == args.expected_output_inventory_raw_sha256,
            "output file inventory does not match its explicit raw-byte SHA")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    _expect(inventory.get("manifest_type") == "world_phone_shape_factorial_eval_output_file_inventory" and
            inventory.get("manifest_sha256") == payload_sha(inventory), "EVAL output inventory type/payload differs")
    _expect(report.get("output_file_inventory_sha256") == inventory.get("manifest_sha256") and
            Path(report.get("output_file_inventory_path", "")).resolve() == inventory_path,
            "EVAL report/output inventory link differs")
    out_root = report_path.resolve().parent
    return report, inventory, out_root, analysis_plan, {str(item["id"]): item for item in physical["records"]}, {
        "physical_manifest": physical, "physical_plan": physical_plan, "physical_qa": physical_qa,
        "source_input_manifest": source_input_manifest,
        "tts_pair_manifest": tts_pair_manifest,
        "calibration_freeze": cal_freeze, "evaluation_freeze": freeze, "calibration_report": cal_report, "calibration_qa": cal_qa,
        "qa_repair_authorization": qa_repair_record,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pyworld
    import soundfile as sf

    report_path = args.manifest.resolve(strict=True)
    inventory_path = args.output_inventory.resolve(strict=True)
    report, output_inventory, output_root, analysis_plan, world_by_id, documents = _load_and_verify_report(args, np)
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
    _expect(len(inventory_rows) == 1120, f"expected 1120 scientific output files, got {len(inventory_rows)}")
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
    actual_relative = actual_relative_all - {"evaluation_manifest.json", "output_file_inventory.json"}
    _expect(actual_relative == expected_relative, f"output tree differs from inventory: missing={len(expected_relative-actual_relative)} extra={len(actual_relative-expected_relative)}")

    input_entries = report["input_inventory"]
    input_roles: dict[str, dict[str, Any]] = {}
    for entry in input_entries:
        role = entry["role"]
        _expect(role not in input_roles, f"duplicate input role: {role}")
        path = _verified_file(entry["path"], entry["sha256"], entry["bytes"])
        input_roles[role] = {**entry, "resolved_path": path}
    _expect(len(input_roles) == 1440, "input role inventory is not exactly 1440")

    phone_binding = report["input_bindings"].get("phone_inventory")
    _expect(isinstance(phone_binding, dict), "frozen CAL phone inventory binding is missing")
    phone_inventory_path = _verified_file(phone_binding["path"], phone_binding["sha256"], phone_binding.get("bytes"))
    phone_inventory = json.loads(phone_inventory_path.read_text(encoding="utf-8"))
    phone_inventory_payload = hashlib.sha256(canonical_bytes({k: v for k, v in phone_inventory.items() if k != "inventory_sha256"})).hexdigest()
    _expect(phone_inventory.get("status") == "FROZEN_CAL_ONLY_PHONE_INVENTORY" and phone_inventory.get("selection_split") == "calibration", "phone inventory is not the frozen CAL-only selection")
    _expect(phone_inventory.get("evaluation_textgrids_or_endpoints_read") is False, "phone inventory records evaluation read")
    _expect(phone_inventory.get("inventory_sha256") == phone_inventory_payload, "phone inventory payload fingerprint mismatch")
    phone_plan_binding = analysis_plan.get("input_bindings", {}).get("phone_inventory", {})
    _expect(isinstance(phone_plan_binding, dict) and phone_plan_binding.get("raw_sha256") == phone_binding.get("sha256") and
            phone_plan_binding.get("payload_field") == "inventory_sha256" and
            phone_plan_binding.get("payload_sha256") == phone_inventory_payload == PHONE_INVENTORY_PAYLOAD_SHA256,
            "phone inventory payload differs from frozen input_bindings.phone_inventory role")
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
    _expect(len(ids) == 80 and len(set(row_ids)) == 80, "EVAL direction record identities are not 80 ordered pairs")
    _expect(ids == analysis_plan.get("ordered_evaluation_ids"), "ordered EVAL IDs differ from the frozen plan")
    _expect(len({str(row.get("speaker")) for row in report["records"]}) == 40, "EVAL direction rows do not span 40 speakers")
    ids_by_speaker: dict[str, set[str]] = {}
    for source_row in report["records"]:
        ids_by_speaker.setdefault(str(source_row["speaker"]), set()).add(str(source_row["id"]))
    _expect(all(len(samples) == 2 for samples in ids_by_speaker.values()), "EVAL speakers do not each have two distinct utterances")
    for sample_id in ids:
        got = [row["direction"] for row in report["records"] if row["id"] == sample_id]
        _expect(got == list(CONDITIONS), f"{sample_id}: directions/order differ from frozen plan")
        _expect(sample_id in world_by_id, f"{sample_id}: report row absent from source physical EVAL manifest")

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
        _expect(row.get("source_identity_sha256") == source_record.get("source_identity_sha256") and row.get("speaker") == source_record.get("speaker"), f"{row_label}: source identity/speaker differs from parent physical EVAL manifest")
        physical_direction = source_record.get("directions", {}).get(direction, {})
        _expect(physical_direction.get("recipient_condition") == rec_condition and physical_direction.get("donor_condition") == donor_condition,
                f"{row_label}: parent physical direction/recipient binding differs")

        def input_entry(role: str) -> dict[str, Any]:
            _expect(role in input_roles, f"{row_label}: missing sealed input role {role}")
            return input_roles[role]

        condition_refs = row.get("condition_audio_refs", {})
        _expect(set(condition_refs) == {"natural", "tts"}, f"{row_label}: source/fixed condition reference inventory differs")
        for condition in ("natural", "tts"):
            refs = condition_refs[condition]
            raw_ref = refs.get("source_raw", {})
            raw_entry = input_entry(f"{sample_id}/{condition}/A_RAW_source")
            physical_condition = source_record.get(condition, {})
            _expect(Path(str(raw_ref.get("staged_audio_path", ""))).resolve(strict=True) == raw_entry["resolved_path"] and
                    raw_ref.get("staged_audio_file_sha256") == raw_entry["sha256"] and
                    raw_ref.get("file_sha256") == raw_entry["sha256"] and
                    raw_ref.get("original_source_file_sha256") == physical_condition.get("source_audio_sha256") and
                    raw_ref.get("original_source_file_sha256") == physical_condition.get("staged_audio_sha256"),
                    f"{row_label}/{condition}: RAW input reference differs from sealed file ledger")
            raw_f32, raw_rate = sf.read(str(raw_entry["resolved_path"]), dtype="float32", always_2d=True)
            raw_i16, raw_rate_i16 = sf.read(str(raw_entry["resolved_path"]), dtype="int16", always_2d=True)
            _expect(int(raw_rate) == SAMPLE_RATE and int(raw_rate_i16) == SAMPLE_RATE and raw_f32.shape[1] == 1 and raw_i16.shape == raw_f32.shape,
                    f"{row_label}/{condition}: RAW audio decode clock/channels differ")
            raw_pcm = np.asarray(raw_f32[:, 0], dtype=np.float32)
            raw_pcm16 = np.asarray(raw_i16[:, 0], dtype="<i2")
            _expect(len(raw_pcm) == int(raw_ref.get("samples", -1)) == int(raw_ref.get("raw_decoded_samples", len(raw_pcm))) and
                    hashlib.sha256(raw_pcm.astype("<f4", copy=False).tobytes(order="C")).hexdigest() == raw_ref.get("float32_pcm_sha256") and
                    hashlib.sha256(raw_pcm16.tobytes(order="C")).hexdigest() == raw_ref.get("pcm16_sha256"),
                    f"{row_label}/{condition}: RAW decoded PCM identity differs")
            _expect(np.isfinite(raw_pcm).all(), f"{row_label}/{condition}: RAW audio contains nonfinite PCM")
            fixed_ref = refs.get("fixed_world_input", {})
            fixed_entry = input_entry(f"{sample_id}/{condition}/fixed_audio")
            physical_fixed = physical_condition.get("fixed_audio", {})
            _expect(Path(str(fixed_ref.get("path", ""))).resolve(strict=True) == fixed_entry["resolved_path"] and
                    fixed_ref.get("file_sha256") == fixed_entry["sha256"] and
                    Path(str(physical_fixed.get("path", ""))).resolve(strict=True) == fixed_entry["resolved_path"] and
                    physical_fixed.get("file_sha256") == fixed_entry["sha256"],
                    f"{row_label}/{condition}: fixed WORLD input reference differs from sealed file ledger")
            feature_ref = refs.get("world_features", {})
            feature_entry = input_entry(f"{sample_id}/{condition}/WORLD_features")
            physical_feature = physical_condition.get("world_features", {})
            _expect(Path(str(feature_ref.get("path", ""))).resolve(strict=True) == feature_entry["resolved_path"] and
                    feature_ref.get("file_sha256") == feature_entry["sha256"] and
                    Path(str(physical_feature.get("path", ""))).resolve(strict=True) == feature_entry["resolved_path"] and
                    physical_feature.get("file_sha256") == feature_entry["sha256"],
                    f"{row_label}/{condition}: WORLD feature reference differs from sealed file ledger")

        scoring_audio = row.get("scoring_audio", {})
        _expect(canonical_bytes(scoring_audio.get("A_RAW")) == canonical_bytes(condition_refs[rec_condition]["source_raw"]),
                f"{row_label}: A_RAW is not the exact recipient source PCM reference")
        raw_arm_meta = physical_direction.get("arms", {}).get("RT", {})
        a_rt_meta = scoring_audio.get("A_RT", {})
        a_rt_input = input_entry(f"{sample_id}/{direction}/RT/waveform")
        _expect(Path(str(a_rt_meta.get("path", ""))).resolve(strict=True) == a_rt_input["resolved_path"] and
                a_rt_meta.get("file_sha256") == a_rt_input["sha256"] and
                Path(str(raw_arm_meta.get("path", ""))).resolve(strict=True) == a_rt_input["resolved_path"] and
                raw_arm_meta.get("file_sha256") == a_rt_input["sha256"],
                f"{row_label}: A_RT is not the exact original recipient RT waveform")

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
        for arm, endpoint_meta in (("RT", rec_param_meta), ("SP", sp_param_meta)):
            physical_endpoint = physical_direction.get("arms", {}).get(arm, {}).get("parameters", {})
            _expect(Path(str(endpoint_meta.get("path", ""))).resolve(strict=True) == Path(str(physical_endpoint.get("path", ""))).resolve(strict=True) and
                    endpoint_meta.get("file_sha256") == physical_endpoint.get("file_sha256"),
                    f"{row_label}/{arm}: original parameter reference differs from parent physical manifest")
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
        physical_phone_meta = physical_direction.get("phone_map", {})
        _expect(Path(str(phone_meta.get("path", ""))).resolve(strict=True) == Path(str(physical_phone_meta.get("path", ""))).resolve(strict=True) and
                phone_meta.get("file_sha256") == physical_phone_meta.get("file_sha256"),
                f"{row_label}: phone-map reference differs from parent physical manifest")
        phone_path = _verified_file(phone_meta["path"], phone_meta["file_sha256"])
        phone_map = json.loads(phone_path.read_text(encoding="utf-8"))
        _expect(canonical_bytes(phone_map) == canonical_bytes(physical_direction.get("mapping")),
                f"{row_label}: on-disk phone map differs from the parent physical manifest mapping")
        _expect(phone_map.get("split") == "evaluation" and phone_map.get("id") == sample_id and phone_map.get("speaker") == row["speaker"], f"{row_label}: phone-map identity mismatch")
        _expect(phone_map.get("recipient_condition") == rec_condition and phone_map.get("donor_condition") == donor_condition, f"{row_label}: phone-map direction mismatch")
        _expect(int(phone_meta["matched_phone_count"]) == len(phone_map.get("matched_phones", [])), f"{row_label}: phone-map count mismatch")
        _expect(abs(float(phone_map.get("recipient_audio_duration_s", -1.0)) - recipient_duration) <= 1e-9 and abs(float(phone_map.get("donor_audio_duration_s", -1.0)) - donor_duration) <= 1e-9, f"{row_label}: phone-map durations mismatch decoded PCM")

        audit_meta = row["transfer_audit"]
        physical_audit_meta = physical_direction.get("parameter_transfer_audit", {})
        _expect(Path(str(audit_meta.get("path", ""))).resolve(strict=True) == Path(str(physical_audit_meta.get("path", ""))).resolve(strict=True) and
                audit_meta.get("file_sha256") == physical_audit_meta.get("file_sha256"),
                f"{row_label}: transfer-audit reference differs from parent physical manifest")
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
        # The CAL producer's q00 metadata uses its own normalized_sp output as
        # the formula reference. Keep that metadata comparison exact, and
        # independently retain the raw RT-SP/row-power bridge as a second gate.
        q00_bridge = _mixed_compare(expected_q00, own_shape, np, f"{row_label}/q00-full-grid-normalized-RT-reference")
        q00_raw_rt_bridge = _mixed_compare(expected_q00, rt_shape, np, f"{row_label}/q00-full-grid-raw-RT-row-power-bridge")
        q11_bridge = _mixed_compare(expected_q11, sp_shape, np, f"{row_label}/q11-full-grid-SP-bridge")
        for label, actual, summary in (("q00", q00_token_checks, row["softmax_formula_endpoint_audits"]["q00"]),
                                       ("q11_mapped_shape", q11_map_checks, row["softmax_formula_endpoint_audits"]["q11_mapped_shape"]),
                                       ("q11_saved_SP_endpoint", q11_endpoint_checks, row["softmax_formula_endpoint_audits"]["q11_saved_SP_endpoint"])):
            _expect(summary.get("passed") is True and int(summary.get("comparison_count", -1)) == len(actual), f"{row_label}/{label}: token formula audit summary differs")
            _expect(abs(float(summary.get("max_abs_error", float("nan"))) - max((item["max_abs_error"] for item in actual), default=0.0)) <= 1e-18, f"{row_label}/{label}: formula summary max absolute error differs")
            _expect(abs(float(summary.get("max_relative_error", float("nan"))) - max((item["max_relative_error"] for item in actual), default=0.0)) <= 1e-12, f"{row_label}/{label}: formula summary max relative error differs")
        softmax_summary = row["softmax_formula_endpoint_audits"]
        for key, checked, reference in (("q00_full_grid_RT_endpoint", q00_bridge, own_shape), ("q11_full_grid_SP_endpoint", q11_bridge, sp_shape)):
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
            physical_arm = physical_direction.get("arms", {}).get(arm, {})
            _expect(wave_entry["resolved_path"] == Path(str(physical_arm.get("path", ""))).resolve(strict=True) and
                    wave_entry["sha256"] == physical_arm.get("file_sha256"),
                    f"{row_label}/{arm}: original waveform reference differs from parent physical manifest")
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
                        "transfer_arrays_rebuilt": len(audit_checks), "q00_full_grid_bridge": q00_bridge,
                        "q00_raw_RT_parameter_bridge": q00_raw_rt_bridge, "q11_full_grid_bridge": q11_bridge,
                        "all_cells": cell_results, "token_dynamics_verified": token_dynamics,
                        "epsilon_floor_token_audit_count": len(epsilon_rows)})
        total_token_rows += len(tokens)

    _expect(cell_total == 800 and len(qa_rows) == 160, f"independent QA coverage mismatch: directions={len(qa_rows)} cells={cell_total}")
    _expect(len({row["id"] for row in qa_rows}) == 80 and len({row["speaker"] for row in qa_rows}) == 40, "independent QA cohort uniqueness mismatch")
    result = {
        "schema_version": 1,
        "manifest_type": "world_phone_shape_mean_dynamic_eval_independent_reconstruction_qa",
        "status": "PASS",
        "scope": "EVAL physical waveforms only; 80 utterances / 40 speakers / 160 directions / 800 WORLD waveforms; no video/model/scoring inputs",
        "evaluation_inputs_read": True,
        "evaluation_outcomes_read": False,
        "effects_read": False,
        "model_forward_performed": False,
        "input_report": {"path": str(report_path), "raw_sha256": sha256_file(report_path), "payload_sha256": report.get("manifest_sha256")},
        "input_output_inventory": {"path": str(inventory_path), "raw_sha256": sha256_file(inventory_path), "payload_sha256": output_inventory.get("manifest_sha256"), "entries_verified": len(inventory_rows)},
        "bound_world_manifest_raw_sha256": PHYSICAL_MANIFEST_RAW_SHA256,
        "bound_world_plan_raw_sha256_only_without_deserialization": PHYSICAL_PLAN_RAW_SHA256,
        "bound_world_source": {"path": str(world_source_path), "sha256": WORLD_SOURCE_SHA256},
        "bound_analysis_source_sha256": report.get("analysis_source_sha256"),
        "bound_analysis_spec_sha256": EVAL_DESIGN_SHA256,
        "bound_analysis_plan_raw_sha256": sha256_file(args.analysis_plan.resolve(strict=True)),
        "bound_analysis_plan_payload_sha256": analysis_plan.get("manifest_sha256"),
        "independent_qa_source_path": str(Path(__file__).resolve()),
        "independent_qa_source_sha256": sha256_file(Path(__file__).resolve()),
        "plan_bound_independent_qa_source_sha256": analysis_plan.get("independent_qa_source_sha256"),
        "qa_repair_authorization": documents["qa_repair_authorization"],
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
    parser.add_argument("--expected-manifest-raw-sha256", required=True)
    parser.add_argument("--expected-output-inventory-raw-sha256", required=True)
    parser.add_argument("--analysis-plan", required=True, type=Path)
    parser.add_argument("--evaluation-freeze", required=True, type=Path)
    parser.add_argument("--qa-repair-authorization", required=True, type=Path)
    parser.add_argument("--expected-qa-repair-authorization-raw-sha256", required=True)
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
