#!/usr/bin/env python3
"""Independent waveform, transfer-dose and energy-control QA for WORLD EVAL.

This consumer reads the sealed physical EVAL manifest and its WAV/NPZ files.
It does not import the WORLD producer, SyncNet evaluator, or any score code.
The report contains only artifact integrity, waveform-contract, intervention-
dose, reconstruction and local-envelope residual diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyworld
import soundfile as sf


RATE = 16_000
FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 65.0
F0_CEILING_HZ = 500.0
FIXED_RMS = 0.0376838172675746
ENERGY_RADIUS = 400
ENERGY_HOP = 80
ENERGY_STEPS = 8
ENERGY_STEP_DB = 3.0
ENERGY_BOUND_DB = 12.0
ENERGY_FLOOR = 1e-5
ARMS = ("RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
BASE_ARMS = ("RT", "EX", "SP", "BOTH", "F0", "AP")
EXPECTED = {
    "input_manifest": "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b",
    "tts_pair_manifest": "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2",
    "physical_manifest": "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3",
    "physical_plan": "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e",
    "cal_freeze": "0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d",
    "spec": "f085e2915349523891a478e3fc4dd1dada8493d68d3d73608bcc042d4a4e62ca",
    "producer": "dba7a98bbe99ba2546062e5918b80481db97038113fc4d96dab3bc09f6b5d8c8",
    "freeze_fingerprint": "0ba872a3f10e039c28c6c9396d801584df4b09fb7b620fa8cea8a4e3edfbcf31",
}


class QAError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_json_fingerprint(value: dict[str, Any], label: str) -> str:
    expected = value.get("manifest_sha256")
    observed = sha_bytes(canonical({key: item for key, item in value.items() if key != "manifest_sha256"}))
    if expected != observed:
        raise QAError(f"{label} fingerprint mismatch: {expected} != {observed}")
    return observed


def finite(value: np.ndarray) -> bool:
    return bool(np.isfinite(value).all())


def rms(value: np.ndarray) -> float:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(x * x)))


def pcm_float32_sha(value: np.ndarray) -> str:
    x = np.asarray(value, dtype=np.float32).reshape(-1)
    return sha_bytes(x.tobytes(order="C"))


def read_audio(meta: dict[str, Any], label: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(meta["path"]).resolve(strict=True)
    expected = str(meta["file_sha256"])
    observed = sha_file(path)
    if observed != expected:
        raise QAError(f"{label} file SHA mismatch: {path}")
    info = sf.info(str(path))
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != RATE or data.shape[1] != 1 or data.shape[0] == 0:
        raise QAError(f"{label} must decode as nonempty mono {RATE} Hz float PCM")
    if str(info.subtype).upper() != "FLOAT":
        raise QAError(f"{label} must use FLOAT WAV subtype, got {info.subtype}")
    wave = np.asarray(data[:, 0], dtype=np.float32)
    pcm_sha = pcm_float32_sha(wave)
    if pcm_sha != meta.get("float32_pcm_sha256"):
        raise QAError(f"{label} decoded float32 PCM SHA mismatch")
    if int(meta.get("samples", -1)) != len(wave):
        raise QAError(f"{label} sample count mismatch")
    return wave, {
        "path": str(path), "file_sha256": observed, "float32_pcm_sha256": pcm_sha,
        "samples": int(len(wave)), "sample_rate_hz": int(rate), "channels": 1,
        "subtype": str(info.subtype), "rms": rms(wave), "peak_abs": float(np.max(np.abs(wave))),
    }


def load_npz(meta: dict[str, Any], label: str) -> dict[str, np.ndarray]:
    path = Path(meta["path"]).resolve(strict=True)
    observed = sha_file(path)
    if observed != meta.get("file_sha256"):
        raise QAError(f"{label} file SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def world_analyze(audio: np.ndarray) -> dict[str, np.ndarray | int]:
    x64 = np.asarray(audio, dtype=np.float32).reshape(-1).astype(np.float64)
    if x64.size == 0 or not finite(x64):
        raise QAError("fixed audio is empty or non-finite")
    f0, time_axis = pyworld.dio(
        x64, RATE, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEILING_HZ, frame_period=FRAME_PERIOD_MS
    )
    f0 = pyworld.stonemask(x64, f0, time_axis, RATE)
    sp = pyworld.cheaptrick(x64, f0, time_axis, RATE)
    ap = pyworld.d4c(x64, f0, time_axis, RATE)
    f0 = np.where(np.asarray(f0, dtype=np.float64) > 0, np.asarray(f0, dtype=np.float64), 0.0)
    time_axis = np.asarray(time_axis, dtype=np.float64)
    sp = np.maximum(np.asarray(sp, dtype=np.float64), 0.0)
    ap = np.clip(np.asarray(ap, dtype=np.float64), 0.0, 1.0)
    if sp.ndim != 2 or ap.shape != sp.shape or len(f0) != sp.shape[0] or len(time_axis) != len(f0):
        raise QAError("independent WORLD feature dimensions disagree")
    fft_size = int(2 * (sp.shape[1] - 1))
    frequency = np.fft.rfftfreq(fft_size, d=1.0 / RATE)
    return {"f0": f0, "time": time_axis, "sp": sp, "ap": ap, "frequency_hz": frequency, "fft_size": fft_size}


def compare_array(expected: np.ndarray, observed: np.ndarray, label: str, atol: float = 1e-10) -> float:
    a = np.asarray(expected)
    b = np.asarray(observed)
    if a.shape != b.shape:
        raise QAError(f"{label} shape mismatch: {a.shape} != {b.shape}")
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        if not np.array_equal(a, b):
            raise QAError(f"{label} integer/bool values mismatch")
        return 0.0
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    delta = delta[np.isfinite(delta)]
    maximum = float(delta.max()) if delta.size else 0.0
    if not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
        raise QAError(f"{label} values differ (max_abs={maximum:.9g})")
    return maximum


def normalized_sp(sp: np.ndarray) -> np.ndarray:
    power = np.maximum(np.asarray(sp, dtype=np.float64), 0.0)
    total = np.sum(power, axis=1, keepdims=True)
    if np.any(total <= 0):
        raise QAError("WORLD SP contains a zero-power row")
    shape = np.maximum(power / total, 1e-12)
    return shape / np.sum(shape, axis=1, keepdims=True)


def nearest_index(times: np.ndarray, value: float) -> int:
    right = int(np.searchsorted(times, value, side="left"))
    if right <= 0:
        return 0
    if right >= len(times):
        return len(times) - 1
    left = right - 1
    return left if value - float(times[left]) <= float(times[right]) - value else right


def interp_rows(values: np.ndarray, times: np.ndarray, query: float) -> np.ndarray:
    right = int(np.searchsorted(times, query, side="left"))
    if right <= 0:
        return values[0].copy()
    if right >= len(times):
        return values[-1].copy()
    left = right - 1
    span = float(times[right] - times[left])
    alpha = 0.0 if span <= 0 else min(1.0, max(0.0, (query - float(times[left])) / span))
    return values[left] * (1.0 - alpha) + values[right] * alpha


def voiced_f0_at(times: np.ndarray, f0: np.ndarray, query: float, indices: np.ndarray) -> tuple[float, int, bool]:
    if indices.size == 0:
        return 0.0, -1, False
    local_times = times[indices]
    nearest_local = nearest_index(local_times, query)
    nearest = int(indices[nearest_local])
    if float(f0[nearest]) <= 0.0:
        return 0.0, nearest, False
    lo = hi = nearest_local
    while lo > 0 and indices[lo] == indices[lo - 1] + 1 and float(f0[indices[lo - 1]]) > 0.0:
        lo -= 1
    while hi + 1 < len(indices) and indices[hi + 1] == indices[hi] + 1 and float(f0[indices[hi + 1]]) > 0.0:
        hi += 1
    run = indices[lo : hi + 1]
    value = float(np.exp(np.interp(query, times[run], np.log(f0[run]))))
    return value, nearest, True


def expected_transfer(recipient: dict[str, np.ndarray], donor: dict[str, np.ndarray], mapping: dict[str, Any]):
    rf0, rt, rsp, rap = (recipient[k] for k in ("f0", "time", "sp", "ap"))
    df0, dt, dsp, dap = (donor[k] for k in ("f0", "time", "sp", "ap"))
    if rsp.shape[1] != dsp.shape[1] or not np.array_equal(recipient["frequency_hz"], donor["frequency_hz"]):
        raise QAError("recipient/donor WORLD frequency grid mismatch")
    dshape = normalized_sp(dsp)
    frame_n = len(rt)
    mapped = np.zeros(frame_n, dtype=bool)
    mapped_f0 = np.zeros(frame_n, dtype=np.float64)
    mapped_ap = np.zeros_like(rap)
    mapped_shape = np.zeros((frame_n, rsp.shape[1]), dtype=np.float64)
    donor_time = np.full(frame_n, np.nan, dtype=np.float64)
    donor_frame = np.full(frame_n, -1, dtype=np.int32)
    donor_voiced = np.zeros(frame_n, dtype=bool)
    for phone in mapping["matched_phones"]:
        r0, r1 = map(float, phone["recipient_interval_s"])
        d0, d1 = map(float, phone["donor_interval_s"])
        rmask = (rt >= r0) & (rt < r1)
        if abs(r1 - float(mapping.get("recipient_audio_duration_s", -1.0))) <= 1e-9:
            rmask |= np.abs(rt - r1) <= 1e-9
        ri = np.flatnonzero(rmask)
        dmask = (dt >= d0) & (dt < d1)
        if abs(d1 - float(mapping.get("donor_audio_duration_s", -1.0))) <= 1e-9:
            dmask = (dt >= d0) & (dt <= d1)
        di = np.flatnonzero(dmask)
        if not ri.size or not di.size or r1 <= r0 or d1 <= d0:
            continue
        donor_local_times = dt[di]
        local_log_shape = np.log(np.maximum(dshape[di], 1e-12))
        for index in ri:
            query = d0 + (float(rt[index]) - r0) * (d1 - d0) / (r1 - r0)
            query = min(d1, max(d0, query))
            mapped_ap[index] = interp_rows(dap[di], donor_local_times, query)
            row = np.exp(interp_rows(local_log_shape, donor_local_times, query))
            total = float(np.sum(row))
            if total <= 0 or not math.isfinite(total):
                continue
            mapped_shape[index] = row / total
            mapped_f0[index], donor_frame[index], donor_voiced[index] = voiced_f0_at(dt, df0, query, di)
            donor_time[index] = query
            mapped[index] = True
    jointly_voiced = mapped & (rf0 > 0.0) & donor_voiced
    power = np.sum(rsp, axis=1, keepdims=True)
    mapped_sp = rsp.copy()
    mapped_sp[mapped] = mapped_shape[mapped] * power[mapped]
    ex_f0, ex_ap = rf0.copy(), rap.copy()
    ex_f0[mapped] = mapped_f0[mapped]
    ex_ap[mapped] = mapped_ap[mapped]
    f0_f0, ap_ap = rf0.copy(), rap.copy()
    f0_f0[jointly_voiced] = mapped_f0[jointly_voiced]
    ap_ap[jointly_voiced] = mapped_ap[jointly_voiced]
    arms = {
        "RT": {"f0": rf0.copy(), "sp": rsp.copy(), "ap": rap.copy()},
        "EX": {"f0": ex_f0, "sp": rsp.copy(), "ap": ex_ap},
        "SP": {"f0": rf0.copy(), "sp": mapped_sp, "ap": rap.copy()},
        "BOTH": {"f0": ex_f0.copy(), "sp": mapped_sp.copy(), "ap": ex_ap.copy()},
        "F0": {"f0": f0_f0, "sp": rsp.copy(), "ap": rap.copy()},
        "AP": {"f0": rf0.copy(), "sp": rsp.copy(), "ap": ap_ap},
    }
    power_error = np.full(frame_n, np.nan, dtype=np.float64)
    power_error[mapped] = np.sum(mapped_sp[mapped], axis=1) - power[mapped, 0]
    audit = {
        "mapped_frames": mapped,
        "mapped_f0": mapped_f0,
        "mapped_ap": mapped_ap,
        "mapped_shape": mapped_shape,
        "donor_time": donor_time,
        "donor_frame": donor_frame,
        "donor_voiced": donor_voiced,
        "joint_voiced": jointly_voiced,
        "sp_power_error_by_frame": power_error,
    }
    return arms, audit


def rms_envelope(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        raise QAError("local envelope input must have at least two samples")
    radius = ENERGY_RADIUS
    weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(-radius, radius + 1, dtype=np.float64) / radius))
    weights /= np.sum(weights)
    padded = np.pad(x * x, (radius, radius), mode="reflect")
    centers = np.arange(0, len(x), ENERGY_HOP, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)[centers]
    return centers.astype(np.float64) / RATE, np.sqrt(np.maximum(windows @ weights, 0.0))


def envelope_match(audio: np.ndarray, target: np.ndarray, times: np.ndarray):
    y0 = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(y0)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if not n or len(target) != len(times) or not target.size:
        raise QAError("invalid local envelope input")
    cumulative = np.zeros(n, dtype=np.float64)
    current = y0.astype(np.float64)
    for _ in range(ENERGY_STEPS):
        _, achieved = rms_envelope(current)
        delta = np.clip(
            20.0 * np.log10(np.maximum(target, ENERGY_FLOOR) / np.maximum(achieved, ENERGY_FLOOR)),
            -ENERGY_STEP_DB,
            ENERGY_STEP_DB,
        )
        samples = np.arange(n, dtype=np.float64) / RATE
        delta_samples = np.interp(samples, times, delta, left=float(delta[0]), right=float(delta[-1]))
        cumulative = np.clip(cumulative + delta_samples, -ENERGY_BOUND_DB, ENERGY_BOUND_DB)
        adjusted = y0.astype(np.float64) * np.power(10.0, cumulative / 20.0)
        before = rms(adjusted)
        if before <= 1e-12:
            raise QAError("local envelope match produced silence")
        current = (adjusted * (FIXED_RMS / before)).astype(np.float32).astype(np.float64)
    _, achieved = rms_envelope(current)
    return current.astype(np.float32), {
        "envelope_times_s": times,
        "target_rms": target.astype(np.float32),
        "achieved_rms": achieved.astype(np.float32),
        "local_gain_db": cumulative.astype(np.float32),
    }


def e_target(rec_fixed: np.ndarray, donor_fixed: np.ndarray, mapping: dict[str, Any]):
    times, recipient_env = rms_envelope(rec_fixed)
    donor_times, donor_env = rms_envelope(donor_fixed)
    ratio = np.ones(len(times), dtype=np.float64)
    exposed = np.zeros(len(times), dtype=bool)
    for phone in mapping["matched_phones"]:
        r0, r1 = map(float, phone["recipient_interval_s"])
        d0, d1 = map(float, phone["donor_interval_s"])
        selected = (times >= r0) & (times < r1)
        if not np.any(selected) or r1 <= r0:
            continue
        mapped_times = d0 + (times[selected] - r0) * (d1 - d0) / (r1 - r0)
        donor_values = np.interp(mapped_times, donor_times, donor_env, left=float(donor_env[0]), right=float(donor_env[-1]))
        own = np.maximum(recipient_env[selected], ENERGY_FLOOR)
        db = np.clip(20.0 * np.log10(np.maximum(donor_values, ENERGY_FLOOR) / own), -ENERGY_BOUND_DB, ENERGY_BOUND_DB)
        ratio[selected] = np.power(10.0, db / 20.0)
        exposed[selected] = True
    return times, recipient_env * ratio, ratio.astype(np.float32), exposed


def residual_summary(target: np.ndarray, achieved: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    achieved = np.asarray(achieved, dtype=np.float64)
    residual = 20.0 * np.log10(np.maximum(achieved, ENERGY_FLOOR) / np.maximum(target, ENERGY_FLOOR))
    active = target >= ENERGY_FLOOR
    residual = residual[active] if np.any(active) else residual
    return {
        "frames": int(len(residual)),
        "median_db": float(np.median(residual)) if len(residual) else 0.0,
        "p95_abs_db": float(np.percentile(np.abs(residual), 95)) if len(residual) else 0.0,
        "max_abs_db": float(np.max(np.abs(residual))) if len(residual) else 0.0,
    }


def verify_path_meta(meta: dict[str, Any], label: str, file_set: set[Path], expected_type: str):
    path = Path(meta["path"]).resolve(strict=True)
    if not path.is_file():
        raise QAError(f"missing {label} file")
    file_set.add(path)
    actual = sha_file(path)
    if actual != meta.get("file_sha256"):
        raise QAError(f"{label} SHA mismatch")
    if expected_type == "npz":
        with np.load(path, allow_pickle=False):
            pass
    return path


def process_context(record: dict[str, Any], input_by_id: dict[str, dict[str, Any]], tts_by_id: dict[str, dict[str, Any]], file_sets: dict[str, set[Path]], global_stats: dict[str, Any]) -> dict[str, Any]:
    sample_id = record["id"]
    input_row = input_by_id[sample_id]
    tts_row = tts_by_id[sample_id]
    if record.get("split") != "evaluation" or record.get("physical_alignment_pair_qualified") is not True:
        raise QAError("physical evaluation row is not qualified")
    speaker = record["speaker"]
    if input_row.get("speaker") != speaker or tts_row.get("speaker") != speaker:
        raise QAError("source/TTS speaker identity mismatch")
    if record.get("source_identity_sha256") is None:
        raise QAError("missing source identity SHA")
    if record["natural"].get("source_audio_sha256") != input_row.get("ref_sha256"):
        raise QAError("natural source hash differs from the frozen input row")
    if record["tts"].get("source_audio_sha256") != tts_row.get("qwen_canonical_16k", {}).get("sha256"):
        raise QAError("TTS source hash differs from the frozen canonical pair row")

    features: dict[str, dict[str, np.ndarray]] = {}
    fixed: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    wave_meta: dict[str, dict[str, Any]] = {}
    max_feature_diff: dict[str, float] = {}
    for condition in ("natural", "tts"):
        side = record[condition]
        source_path = Path(side["staged_audio_path"]).resolve(strict=True)
        source_sha = sha_file(source_path)
        if source_sha != side["staged_audio_sha256"]:
            raise QAError(f"{condition} staged source SHA mismatch")
        source_wave, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
        if int(sr) != RATE or source_wave.shape[1] != 1 or len(source_wave) != int(side["raw_decoded_samples"]):
            raise QAError(f"{condition} staged source audio dimensions differ")
        raw[condition] = np.asarray(source_wave[:, 0], dtype=np.float32)
        file_sets["wav"].add(source_path)

        fixed_wave, fixed_meta = read_audio(side["fixed_audio"], f"{sample_id}/{condition}/fixed")
        fixed[condition] = fixed_wave
        wave_meta[f"fixed_{condition}"] = fixed_meta
        file_sets["wav"].add(Path(side["fixed_audio"]["path"]).resolve())
        if len(fixed_wave) != len(raw[condition]):
            raise QAError(f"{condition} fixed waveform length differs from source")
        global_error = abs(fixed_meta["rms"] - FIXED_RMS)
        if global_error > 1e-6:
            raise QAError(f"{condition} fixed waveform RMS mismatch: {global_error}")

        feature_meta = side["world_features"]
        feature_path = verify_path_meta(feature_meta, f"{condition} WORLD feature", file_sets["npz"], "npz")
        with np.load(feature_path, allow_pickle=False) as archive:
            saved = {key: np.asarray(archive[key]) for key in archive.files}
        if set(saved) != {"f0", "time", "sp", "ap", "frequency_hz", "fft_size"}:
            raise QAError(f"{condition} feature NPZ array set mismatch")
        expected_feature = world_analyze(fixed_wave)
        for key in ("f0", "time", "sp", "ap", "frequency_hz"):
            max_feature_diff[f"{condition}.{key}"] = compare_array(expected_feature[key], saved[key], f"{sample_id}/{condition}/feature/{key}", atol=1e-10)
        if int(np.asarray(saved["fft_size"]).item()) != int(expected_feature["fft_size"]):
            raise QAError(f"{condition} feature FFT size mismatch")
        features[condition] = {key: saved[key] for key in ("f0", "time", "sp", "ap", "frequency_hz")}

    all_directions: dict[str, Any] = {}
    directions = record["directions"]
    expected_contexts = {"tts_to_natural": ("natural", "tts"), "natural_to_tts": ("tts", "natural")}
    for direction, (recipient, donor) in expected_contexts.items():
        entry = directions[direction]
        if (entry.get("recipient_condition"), entry.get("donor_condition")) != (recipient, donor):
            raise QAError(f"{sample_id}/{direction} recipient/donor direction mismatch")
        mapping = entry.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("pair_qualified") is not True:
            raise QAError(f"{sample_id}/{direction} phone map is not qualified")
        phone_meta = entry["phone_map"]
        verify_path_meta(phone_meta, "phone map JSON", file_sets["json"], "json")
        audit_meta = entry["parameter_transfer_audit"]
        audit_path = verify_path_meta(audit_meta, "transfer audit NPZ", file_sets["npz"], "npz")
        with np.load(audit_path, allow_pickle=False) as archive:
            saved_audit = {key: np.asarray(archive[key]) for key in archive.files}
        if set(saved_audit) != {
            "mapped_frames", "mapped_f0", "mapped_ap", "mapped_shape", "donor_time",
            "donor_frame", "donor_voiced", "joint_voiced", "sp_power_error_by_frame",
        }:
            raise QAError("transfer audit NPZ array set mismatch")
        expected_arms, expected_audit = expected_transfer(features[recipient], features[donor], mapping)
        frame_errors = {}
        for key, value in expected_audit.items():
            frame_errors[key] = compare_array(value, saved_audit[key], f"{sample_id}/{direction}/audit/{key}", atol=1e-9)
        mapped = np.asarray(saved_audit["mapped_frames"], dtype=bool)
        joint = np.asarray(saved_audit["joint_voiced"], dtype=bool)
        exposure = {
            "world_frames": int(len(mapped)),
            "mapped_frames": int(np.count_nonzero(mapped)),
            "mapped_fraction": float(np.mean(mapped)) if len(mapped) else 0.0,
            "jointly_voiced_frames": int(np.count_nonzero(joint)),
            "jointly_voiced_fraction_of_all": float(np.mean(joint)) if len(joint) else 0.0,
            "actual_transfer_frames": {
                "EX": int(np.count_nonzero(mapped)), "SP": int(np.count_nonzero(mapped)),
                "BOTH": int(np.count_nonzero(mapped)), "F0": int(np.count_nonzero(joint)),
                "AP": int(np.count_nonzero(joint)),
            },
            "sp_power_abs_error_max": float(np.nanmax(np.abs(saved_audit["sp_power_error_by_frame"]))) if np.any(mapped) else 0.0,
        }

        loaded_waves: dict[str, np.ndarray] = {}
        synthesis_errors: dict[str, float] = {}
        parameter_errors: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            arm_meta = entry["arms"].get(arm, {})
            if arm_meta.get("status") != "ok":
                raise QAError(f"{sample_id}/{direction}/{arm} did not complete")
            wave, item = read_audio(arm_meta, f"{sample_id}/{direction}/{arm}")
            if len(wave) != len(fixed[recipient]):
                raise QAError(f"{sample_id}/{direction}/{arm} did not preserve recipient length")
            if abs(item["rms"] - FIXED_RMS) > 1e-6:
                raise QAError(f"{sample_id}/{direction}/{arm} global RMS differs from frozen target")
            loaded_waves[arm] = wave
            wave_meta[f"{direction}_{arm}"] = item
            file_sets["wav"].add(Path(arm_meta["path"]).resolve())
            if arm in BASE_ARMS:
                pmeta = arm_meta["parameters"]
                ppath = verify_path_meta(pmeta, f"{arm} parameters", file_sets["npz"], "npz")
                with np.load(ppath, allow_pickle=False) as archive:
                    parameters = {key: np.asarray(archive[key]) for key in archive.files}
                if set(parameters) != {"f0", "time", "sp", "ap", "frequency_hz"}:
                    raise QAError(f"{arm} parameter NPZ array set mismatch")
                errors = {}
                for key in ("f0", "sp", "ap"):
                    errors[key] = compare_array(expected_arms[arm][key], parameters[key], f"{sample_id}/{direction}/{arm}/{key}", atol=1e-9)
                for key in ("time", "frequency_hz"):
                    errors[key] = compare_array(features[recipient][key], parameters[key], f"{sample_id}/{direction}/{arm}/{key}", atol=1e-12)
                parameter_errors[arm] = errors
                synthesized = np.asarray(
                    pyworld.synthesize(parameters["f0"], parameters["sp"], parameters["ap"], RATE, frame_period=FRAME_PERIOD_MS),
                    dtype=np.float64,
                ).reshape(-1)
                target_n = len(fixed[recipient])
                if len(synthesized) < target_n:
                    raise QAError(f"{sample_id}/{direction}/{arm} WORLD resynthesis is short")
                expected_tail = len(synthesized) - target_n
                if int(arm_meta.get("right_discarded_tail_samples", -1)) != expected_tail:
                    raise QAError(f"{sample_id}/{direction}/{arm} discarded-tail metadata mismatch")
                trimmed = synthesized[:target_n]
                synth_rms = rms(trimmed)
                if synth_rms <= 1e-12:
                    raise QAError(f"{sample_id}/{direction}/{arm} WORLD reconstruction is silent")
                rebuilt = (trimmed * (FIXED_RMS / synth_rms)).astype(np.float32)
                synthesis_errors[arm] = compare_array(rebuilt, wave, f"{sample_id}/{direction}/{arm}/waveform", atol=0.0)

        repeat_meta = entry["qc"]["rt_repeat"]
        repeat_wave, repeat_info = read_audio(repeat_meta["repeat_audio"], f"{sample_id}/{direction}/RT_REPEAT")
        compare_array(loaded_waves["RT"], repeat_wave, f"{sample_id}/{direction}/RT_REPEAT", atol=0.0)
        file_sets["wav"].add(Path(repeat_meta["repeat_audio"]["path"]).resolve())
        identity_meta = entry["qc"]["rt_envelope_identity"]
        identity_wave, identity_info = read_audio(identity_meta["audio"], f"{sample_id}/{direction}/RT_ENVELOPE_IDENTITY")
        file_sets["wav"].add(Path(identity_meta["audio"]["path"]).resolve())
        rt_times, rt_env = rms_envelope(loaded_waves["RT"])
        expected_identity, identity_arrays = envelope_match(loaded_waves["RT"], rt_env, rt_times)
        compare_array(expected_identity, identity_wave, f"{sample_id}/{direction}/identity/wave", atol=0.0)
        identity_abs = float(np.max(np.abs(identity_wave.astype(np.float64) - loaded_waves["RT"].astype(np.float64))))
        identity_rel = float(rms(identity_wave.astype(np.float64) - loaded_waves["RT"].astype(np.float64)) / max(rms(loaded_waves["RT"]), 1e-15))
        if identity_abs > 1e-7 or identity_rel > 1e-6:
            raise QAError(f"{sample_id}/{direction} RT envelope identity tolerance failed")
        identity_gain_meta = identity_meta["gain_audit"]
        identity_gain_path = verify_path_meta(identity_gain_meta, "RT identity gain audit", file_sets["npz"], "npz")
        with np.load(identity_gain_path, allow_pickle=False) as archive:
            for key, value in identity_arrays.items():
                compare_array(value, np.asarray(archive[key]), f"{sample_id}/{direction}/identity/gain/{key}", atol=1e-6)

        energy_residuals = {}
        for base_arm, energy_arm in (("EX", "EX_EM"), ("SP", "SP_EM")):
            expected_wave, gain_arrays = envelope_match(loaded_waves[base_arm], rt_env, rt_times)
            energy_meta = entry["arms"][energy_arm]
            compare_array(expected_wave, loaded_waves[energy_arm], f"{sample_id}/{direction}/{energy_arm}/wave", atol=0.0)
            gain_path = verify_path_meta(energy_meta["gain_audit"], f"{energy_arm} gain audit", file_sets["npz"], "npz")
            with np.load(gain_path, allow_pickle=False) as archive:
                for key, value in gain_arrays.items():
                    compare_array(value, np.asarray(archive[key]), f"{sample_id}/{direction}/{energy_arm}/gain/{key}", atol=1e-6)
            _, achieved = rms_envelope(loaded_waves[energy_arm])
            energy_residuals[energy_arm] = residual_summary(rt_env, achieved)

        donor_ratio_times, target_e, expected_ratio, expected_exposed = e_target(fixed[recipient], fixed[donor], mapping)
        expected_e, e_arrays = envelope_match(loaded_waves["RT"], target_e, donor_ratio_times)
        compare_array(expected_e, loaded_waves["E"], f"{sample_id}/{direction}/E/wave", atol=0.0)
        e_meta = entry["arms"]["E"]
        e_gain_path = verify_path_meta(e_meta["gain_audit"], "E gain audit", file_sets["npz"], "npz")
        with np.load(e_gain_path, allow_pickle=False) as archive:
            for key, value in e_arrays.items():
                compare_array(value, np.asarray(archive[key]), f"{sample_id}/{direction}/E/gain/{key}", atol=1e-6)
            compare_array(expected_ratio, np.asarray(archive["intended_ratio"]), f"{sample_id}/{direction}/E/intended_ratio", atol=1e-6)
            compare_array(expected_exposed, np.asarray(archive["exposed"]), f"{sample_id}/{direction}/E/exposed")
        _, achieved_e = rms_envelope(loaded_waves["E"])
        energy_residuals["E"] = residual_summary(target_e, achieved_e)
        global_stats["energy_residuals"].append(energy_residuals)
        global_stats["e_exposure_frames"].append(int(np.count_nonzero(expected_exposed)))
        global_stats["mapped_frames"].append(exposure["mapped_frames"])
        global_stats["joint_frames"].append(exposure["jointly_voiced_frames"])
        global_stats["sp_power_error"].append(exposure["sp_power_abs_error_max"])
        for key, value in synthesis_errors.items():
            global_stats["synthesis_exact_checks"][key] += 1
        global_stats["rt_repeat_checks"] += 1
        global_stats["rt_identity_checks"] += 1
        all_directions[direction] = {
            "recipient": recipient, "donor": donor,
            "recipient_audio_samples": int(len(fixed[recipient])),
            "recipient_world_frames": int(len(features[recipient]["time"])),
            "donor_world_frames": int(len(features[donor]["time"])),
            "mapped_exposure": exposure,
            "transfer_audit_max_abs_errors": frame_errors,
            "parameter_max_abs_errors": parameter_errors,
            "synthesis_waveform_max_abs_errors": synthesis_errors,
            "rt_repeat": {"bit_identical": bool(np.array_equal(loaded_waves["RT"], repeat_wave)), "max_abs_error": float(np.max(np.abs(loaded_waves["RT"].astype(np.float64) - repeat_wave.astype(np.float64))))},
            "rt_identity": {"max_abs_error": identity_abs, "relative_rms_error": identity_rel},
            "energy_match_residuals_db": energy_residuals,
            "energy_only_exposed_frames": int(np.count_nonzero(expected_exposed)),
            "energy_local_gain_bound_hit_samples": int(np.count_nonzero(np.abs(e_arrays["local_gain_db"]) >= ENERGY_BOUND_DB - 1e-5)),
        }
    return {
        "sample_id": sample_id, "speaker": speaker,
        "raw_samples": {key: int(len(value)) for key, value in raw.items()},
        "fixed_rms": {key: wave_meta[f"fixed_{key}"]["rms"] for key in ("natural", "tts")},
        "fixed_feature_reanalysis_max_abs_error": max_feature_diff,
        "directions": all_directions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--tts-pair-manifest", required=True, type=Path)
    parser.add_argument("--physical-manifest", required=True, type=Path)
    parser.add_argument("--physical-plan", required=True, type=Path)
    parser.add_argument("--cal-freeze", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"QA output must be a fresh path: {args.out}")
    if str(pyworld.__version__) != "0.3.5":
        raise QAError(f"expected pyworld 0.3.5, got {pyworld.__version__}")
    top_files = {
        "input_manifest": args.input_manifest,
        "tts_pair_manifest": args.tts_pair_manifest,
        "physical_manifest": args.physical_manifest,
        "physical_plan": args.physical_plan,
        "cal_freeze": args.cal_freeze,
        "spec": args.spec,
    }
    actual_seals = {name: {"path": str(path.resolve(strict=True)), "sha256": sha_file(path)} for name, path in top_files.items()}
    for name, expected in EXPECTED.items():
        observed = actual_seals[name]["sha256"] if name in actual_seals else None
        if name == "producer":
            continue
        if observed != expected:
            raise QAError(f"{name} raw SHA differs from frozen value: {observed}")

    source_manifest = read_json(args.input_manifest)
    tts_manifest = read_json(args.tts_pair_manifest)
    physical = read_json(args.physical_manifest)
    plan = read_json(args.physical_plan)
    freeze = read_json(args.cal_freeze)
    for label, value in (("input manifest", source_manifest), ("TTS pair manifest", tts_manifest), ("physical manifest", physical), ("physical plan", plan)):
        verify_json_fingerprint(value, label)
    if freeze.get("status") != "FROZEN" or freeze.get("manifest_sha256") != EXPECTED["freeze_fingerprint"]:
        raise QAError("independent CAL QA freeze status/fingerprint mismatch")
    if physical.get("status") != "EVALUATION_WAVEFORMS_READY" or physical.get("split") != "evaluation":
        raise QAError("physical evaluation manifest is not ready")
    if physical.get("phone_normalizer_source_sha256") != EXPECTED["producer"]:
        raise QAError("physical producer source SHA mismatch")
    if physical.get("sample_counts", {}).get("intended") != 80:
        raise QAError("physical manifest does not contain 80 EVAL pairs")
    if len(physical.get("records", [])) != 80:
        raise QAError("physical manifest record count mismatch")

    source_rows = [row for row in source_manifest.get("records", []) if row.get("split") == "evaluation"]
    tts_rows = [row for row in tts_manifest.get("records", []) if row.get("split") == "evaluation"]
    source_by_id = {row.get("id"): row for row in source_rows}
    tts_by_id = {row.get("id"): row for row in tts_rows}
    records = physical["records"]
    if len(source_rows) != 80 or len(tts_rows) != 80 or len(source_by_id) != 80 or len(tts_by_id) != 80:
        raise QAError("frozen input/TTS EVAL row counts differ from 80")
    if set(source_by_id) != set(tts_by_id) or set(source_by_id) != {row.get("id") for row in records}:
        raise QAError("physical/input/TTS EVAL ID sets differ")
    id_fields = tuple(tts_manifest.get("input_identity_fields", ("id", "speaker", "text", "split", "ref_sha256")))
    identity_all = []
    tts_all_by_id = {row.get("id"): row for row in tts_manifest.get("records", [])}
    for row in source_manifest.get("records", []):
        identity = {field: row.get(field) for field in id_fields}
        if identity != tts_all_by_id.get(row.get("id"), {}).get("input_identity"):
            raise QAError(f"full source/TTS identity mismatch: {row.get('id')}")
        identity_all.append(identity)
    identity_sha = sha_bytes(canonical(identity_all))
    if identity_sha != tts_manifest.get("input_identity_sha256") or identity_sha != "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27":
        raise QAError("full 100-row source/TTS identity digest mismatch")

    count_by_id = Counter(row.get("id") for row in records)
    if any(count != 1 for count in count_by_id.values()) or len(count_by_id) != 80:
        raise QAError("physical manifest sample IDs are not unique")
    if sum(bool(row.get("mfa_pair_qualified")) for row in records) != 79:
        raise QAError("original MFA eligibility count differs from its frozen 79/80 audit")
    if sum(bool(row.get("physical_alignment_pair_qualified")) for row in records) != 80:
        raise QAError("physical overlay eligibility count differs from frozen 80/80")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(mode=0o700)
    file_sets = {"wav": set(), "npz": set(), "json": set()}
    global_stats = {
        "energy_residuals": [], "e_exposure_frames": [], "mapped_frames": [], "joint_frames": [],
        "sp_power_error": [], "synthesis_exact_checks": Counter(), "rt_repeat_checks": 0, "rt_identity_checks": 0,
    }
    rows = []
    failures = []
    for index, record in enumerate(records, 1):
        try:
            rows.append(process_context(record, source_by_id, tts_by_id, file_sets, global_stats))
        except Exception as exc:
            failures.append({"ordered_index": index, "sample_id": record.get("id"), "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    expected_counts = {"wav": 1920, "npz": 1920}
    actual_counts = {key: len(value) for key, value in file_sets.items()}
    if actual_counts["wav"] != expected_counts["wav"]:
        failures.append({"error_type": "coverage", "error": f"unique WAV count {actual_counts['wav']} != {expected_counts['wav']}"})
    if actual_counts["npz"] != expected_counts["npz"]:
        failures.append({"error_type": "coverage", "error": f"unique NPZ count {actual_counts['npz']} != {expected_counts['npz']}"})

    def quantiles(values):
        array = np.asarray(values, dtype=np.float64)
        return {"count": int(array.size), "min": float(np.min(array)) if array.size else None,
                "median": float(np.median(array)) if array.size else None,
                "p95": float(np.percentile(array, 95)) if array.size else None,
                "max": float(np.max(array)) if array.size else None}

    energy_summary = {}
    for arm in ("EX_EM", "SP_EM", "E"):
        summaries = [
            direction["energy_match_residuals_db"][arm]
            for row in rows
            for direction in row.get("directions", {}).values()
        ]
        energy_summary[arm] = {
            stat: quantiles([summary[stat] for summary in summaries])
            for stat in ("median_db", "p95_abs_db", "max_abs_db")
        }
    output = {
        "schema_version": 1,
        "classification": "independent EVAL waveform/parameter-dose/local-energy QA; no SyncNet or visual scores read",
        "status": "PASS" if not failures and len(rows) == 80 else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": actual_seals,
        "input_identity_sha256": identity_sha,
        "producer_sha256": EXPECTED["producer"],
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pyworld": pyworld.__version__, "soundfile": sf.__version__, "platform": platform.platform()},
        "coverage": {
            "sample_pairs": len(records), "recipient_contexts": len(records) * 2,
            "completed_recipient_contexts": len(rows) * 2, "unique_audio_wavs": actual_counts["wav"],
            "unique_npz_artifacts": actual_counts["npz"], "expected_audio_wavs": 1920,
            "expected_npz_artifacts": 1920, "phone_maps_and_other_json_count": len(file_sets["json"]),
            "original_mfa_qualified_pairs": 79, "physical_overlay_qualified_pairs": 80,
        },
        "dose_and_residual_qc": {
            "mapped_world_frames": quantiles(global_stats["mapped_frames"]),
            "jointly_voiced_transfer_frames": quantiles(global_stats["joint_frames"]),
            "energy_only_exposed_envelope_frames": quantiles(global_stats["e_exposure_frames"]),
            "sp_power_absolute_error_max": quantiles(global_stats["sp_power_error"]),
            "energy_match_residual_db": energy_summary,
            "rt_repeat_contexts_recomputed": global_stats["rt_repeat_checks"],
            "rt_envelope_identity_contexts_recomputed": global_stats["rt_identity_checks"],
            "base_synthesis_exact_waveform_checks_by_arm": dict(global_stats["synthesis_exact_checks"]),
        },
        "failures": failures,
        "recipient_rows": rows,
    }
    out_path = out / "independent_physical_waveform_qa.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "execution_record.json").write_text(json.dumps({
        "qa_source_path": str(Path(__file__).resolve()), "qa_source_sha256": sha_file(Path(__file__).resolve()),
        "input_seals": actual_seals, "output_path": str(out_path), "output_sha256": sha_file(out_path),
        "read_policy": "physical WAV/NPZ and manifests only; no SyncNet score artifacts",
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "sample_pairs": len(records), "completed_pairs": len(rows),
        "recipient_contexts": len(rows) * 2, "wav_count": actual_counts["wav"], "npz_count": actual_counts["npz"],
        "failure_count": len(failures), "output": str(out_path), "output_sha256": sha_file(out_path),
    }, separators=(",", ":")))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
