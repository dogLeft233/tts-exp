#!/usr/bin/env python3
"""Prepare auditable CPU-only WORLD phone-transfer waveforms with a sealed
physical-only alignment overlay option for the frozen S0064 compound token.

The producer reads only sealed paired audio, frozen MFA TextGrids and the
CAL-frozen phone table. It does not read SyncNet, video, or any model endpoint.
EVAL waveform production requires an independent CAL waveform-QC freeze.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_RMS = 0.0376838172675746
SAMPLE_RATE = 16_000
FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 65.0
F0_CEILING_HZ = 500.0
ENERGY_WINDOW_RADIUS = 400
ENERGY_HOP = 80
LOCAL_MATCH_ITERATIONS = 8
LOCAL_STEP_DB = 3.0
LOCAL_BOUND_DB = 12.0
LOCAL_RMS_FLOOR = 1e-5
SILENCE = {"", "sil", "sp", "spn", "<sil>"}
UNKNOWN = {"spn", "unk", "unknown", "oov", "<unk>", "<unknown>"}
ARMS = ("RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
TIER_START_RE = re.compile(r"(?m)^\s*item\s+\[\d+\]:\s*$")
TIER_NAME_RE = re.compile(r'(?m)^\s*name\s*=\s*"((?:""|[^"])*)"')
TIER_CLASS_RE = re.compile(r'(?m)^\s*class\s*=\s*"([^"]+)"')
INTERVAL_RE = re.compile(
    r'xmin\s*=\s*([-+\d.eE]+)\s*\n\s*xmax\s*=\s*([-+\d.eE]+)\s*\n\s*text\s*=\s*"((?:""|[^"])*)"',
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({key: item for key, item in value.items() if key != "manifest_sha256"})).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_phone(label: str) -> str:
    """Frozen physical normalizer: NFC and outer whitespace only."""
    return unicodedata.normalize("NFC", str(label)).strip()


def parse_textgrid(path: Path) -> dict[str, dict[str, Any]]:
    """Read Praat long-format IntervalTiers while preserving tier names."""
    raw = path.read_text(encoding="utf-8")
    starts = list(TIER_START_RE.finditer(raw))
    if not starts:
        raise ValueError("no Praat tier blocks")
    tiers: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(raw)
        block = raw[match.end():end]
        name_match = TIER_NAME_RE.search(block)
        class_match = TIER_CLASS_RE.search(block)
        if not name_match or not class_match:
            continue
        name = name_match.group(1).replace('""', '"')
        if name in tiers:
            raise ValueError(f"duplicate TextGrid tier: {name}")
        intervals = []
        for interval in INTERVAL_RE.finditer(block):
            start, stop = float(interval.group(1)), float(interval.group(2))
            label = interval.group(3).replace('""', '"')
            if stop <= start:
                continue
            intervals.append({"raw_label": label, "start_s": start, "end_s": stop})
        tiers[name] = {"class": class_match.group(1), "intervals": intervals}
    return tiers


def select_tier(tiers: dict[str, dict[str, Any]], base_name: str) -> tuple[str, list[dict[str, Any]]]:
    candidates = [
        (name, item["intervals"])
        for name, item in tiers.items()
        if item["class"] == "IntervalTier" and (name == base_name or name.endswith("_" + base_name))
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one {base_name!r} IntervalTier; found {len(candidates)}")
    return candidates[0]


def deterministic_lcs(recipient: list[dict[str, Any]], donor: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Maximum LCS; tie-break by recipient-index vector then donor-index vector."""

    def key(pairs: tuple[tuple[int, int], ...]) -> tuple[Any, ...]:
        return (tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs))

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[tuple[int, int], ...]:
        if i >= len(recipient) or j >= len(donor):
            return ()
        options = [solve(i + 1, j), solve(i, j + 1)]
        left, right = recipient[i], donor[j]
        if left["normalized_label"] == right["normalized_label"] and left["normalized_label"].casefold() not in SILENCE | UNKNOWN:
            options.append(((int(left["index"]), int(right["index"])),) + solve(i + 1, j + 1))
        length = max(map(len, options))
        return min((option for option in options if len(option) == length), key=key)

    return list(solve(0, 0))


def nearest_index(times: Any, value: float) -> int:
    import numpy as np

    ts = np.asarray(times, dtype=np.float64)
    if ts.ndim != 1 or ts.size == 0:
        raise ValueError("nearest-index requires a nonempty 1-D time axis")
    right = int(np.searchsorted(ts, value, side="left"))
    if right <= 0:
        return 0
    if right >= len(ts):
        return len(ts) - 1
    left = right - 1
    return left if value - ts[left] <= ts[right] - value else right


def _clamp_intervals(intervals: list[dict[str, Any]], duration_s: float, tolerance_s: float = 0.04) -> list[dict[str, Any]]:
    result = []
    for index, interval in enumerate(intervals):
        raw_start, raw_end = float(interval["start_s"]), float(interval["end_s"])
        if raw_start < -tolerance_s or raw_end > duration_s + tolerance_s:
            raise ValueError(f"interval {index} is outside audio clock")
        start, end = min(duration_s, max(0.0, raw_start)), min(duration_s, max(0.0, raw_end))
        if end <= start:
            raise ValueError(f"interval {index} has no duration after clock clamp")
        result.append({
            **interval, "index": index, "start_s": start, "end_s": end,
            "raw_start_s": raw_start, "raw_end_s": raw_end,
            "clamp_start_s": start - raw_start, "clamp_end_s": end - raw_end,
        })
    return result


def _assign_phones_to_words(phones: list[dict[str, Any]], words: list[dict[str, Any]], duration_s: float) -> tuple[list[dict[str, Any]], int]:
    assigned, unassigned = [], 0
    for phone in phones:
        mid = (float(phone["start_s"]) + float(phone["end_s"])) / 2.0
        owner = None
        for wi, word in enumerate(words):
            inside = word["start_s"] <= mid < word["end_s"]
            if not inside and abs(mid - duration_s) <= 1e-9 and wi == len(words) - 1 and word["start_s"] <= mid <= word["end_s"]:
                inside = True
            if inside:
                owner = wi
                break
        if owner is None:
            unassigned += 1
        assigned.append({**phone, "word_index": owner})
    return assigned, unassigned


def prepare_grid(path: Path, audio_samples: int, transcript_tokens: list[str], inventory: set[str]) -> dict[str, Any]:
    duration = audio_samples / SAMPLE_RATE
    tiers = parse_textgrid(path)
    word_tier_name, raw_words = select_tier(tiers, "words")
    phone_tier_name, raw_phones = select_tier(tiers, "phones")
    words = _clamp_intervals(raw_words, duration)
    words = [
        {**word, "raw_label": str(word["raw_label"]).strip(), "normalized_label": unicodedata.normalize("NFC", str(word["raw_label"]).strip())}
        for word in words if str(word["raw_label"]).strip()
    ]
    phones_all = _clamp_intervals(raw_phones, duration)
    phones = []
    for phone in phones_all:
        raw_label = str(phone["raw_label"])
        normalized = normalize_phone(raw_label)
        phones.append({**phone, "normalized_label": normalized, "in_frozen_inventory": raw_label in inventory})
    observed_words = [word["normalized_label"] for word in words]
    if observed_words != transcript_tokens:
        raise ValueError("words tier differs from frozen publisher-text segmentation")
    phones, unassigned = _assign_phones_to_words(phones, words, duration)
    return {
        "textgrid_path": str(path.resolve()), "textgrid_sha256": sha256_file(path),
        "duration_s": duration, "audio_samples": audio_samples,
        "word_tier_name": word_tier_name, "phone_tier_name": phone_tier_name,
        "words": words, "phones": phones, "unassigned_phone_count": unassigned,
        "transcript_tokens": transcript_tokens,
    }


def map_phone_intervals(recipient_grid: dict[str, Any], donor_grid: dict[str, Any]) -> dict[str, Any]:
    rec_words, don_words = recipient_grid["words"], donor_grid["words"]
    if [x["normalized_label"] for x in rec_words] != [x["normalized_label"] for x in don_words]:
        raise ValueError("recipient/donor transcript word sequences differ")
    rec_phones = recipient_grid["phones"]
    don_phones = donor_grid["phones"]
    matched: list[dict[str, Any]] = []
    matched_rec: set[int] = set()
    matched_don: set[int] = set()
    for word_index, (rw, dw) in enumerate(zip(rec_words, don_words)):
        rp = [x for x in rec_phones if x["word_index"] == word_index and x["normalized_label"].casefold() not in SILENCE | UNKNOWN and x["in_frozen_inventory"]]
        dp = [x for x in don_phones if x["word_index"] == word_index and x["normalized_label"].casefold() not in SILENCE | UNKNOWN and x["in_frozen_inventory"]]
        for ri, di in deterministic_lcs(rp, dp):
            r, d = rec_phones[ri], don_phones[di]
            matched_rec.add(ri); matched_don.add(di)
            matched.append({
                "word_index": word_index, "word_label": rw["normalized_label"],
                "recipient_phone_index": ri, "donor_phone_index": di,
                "recipient_raw_label": r["raw_label"], "donor_raw_label": d["raw_label"],
                "recipient_normalized_label": r["normalized_label"], "donor_normalized_label": d["normalized_label"],
                "recipient_interval_s": [r["start_s"], r["end_s"]],
                "recipient_raw_interval_s": [r["raw_start_s"], r["raw_end_s"]],
                "recipient_clamp_s": [r["clamp_start_s"], r["clamp_end_s"]],
                "donor_interval_s": [d["start_s"], d["end_s"]],
                "donor_raw_interval_s": [d["raw_start_s"], d["raw_end_s"]],
                "donor_clamp_s": [d["clamp_start_s"], d["clamp_end_s"]],
            })
    unmatched_rec = [x for x in rec_phones if x["index"] not in matched_rec]
    unmatched_don = [x for x in don_phones if x["index"] not in matched_don]
    return {
        "matched_phones": matched,
        "matched_phone_count": len(matched),
        "unmatched_recipient_phone_count": len(unmatched_rec),
        "unmatched_donor_phone_count": len(unmatched_don),
        "unmatched_recipient_phones": [{"index": x["index"], "label": x["raw_label"], "reason": _unmatched_reason(x)} for x in unmatched_rec],
        "unmatched_donor_phones": [{"index": x["index"], "label": x["raw_label"], "reason": _unmatched_reason(x)} for x in unmatched_don],
        "recipient_words": [{"index": x["index"], "raw_label": x["raw_label"], "normalized_label": x["normalized_label"], "interval_s": [x["start_s"], x["end_s"]]} for x in rec_words],
        "donor_words": [{"index": x["index"], "raw_label": x["raw_label"], "normalized_label": x["normalized_label"], "interval_s": [x["start_s"], x["end_s"]]} for x in don_words],
        "recipient_phones": [{"index": x["index"], "raw_label": x["raw_label"], "normalized_label": x["normalized_label"], "word_index": x["word_index"], "interval_s": [x["start_s"], x["end_s"]], "in_frozen_inventory": x["in_frozen_inventory"]} for x in rec_phones],
        "donor_phones": [{"index": x["index"], "raw_label": x["raw_label"], "normalized_label": x["normalized_label"], "word_index": x["word_index"], "interval_s": [x["start_s"], x["end_s"]], "in_frozen_inventory": x["in_frozen_inventory"]} for x in don_phones],
        "recipient_word_count": len(rec_words), "donor_word_count": len(don_words),
        "recipient_audio_duration_s": recipient_grid["duration_s"],
        "donor_audio_duration_s": donor_grid["duration_s"],
        "word_tier_name_recipient": recipient_grid["word_tier_name"],
        "word_tier_name_donor": donor_grid["word_tier_name"],
        "phone_tier_name_recipient": recipient_grid["phone_tier_name"],
        "phone_tier_name_donor": donor_grid["phone_tier_name"],
    }


def _unmatched_reason(phone: dict[str, Any]) -> str:
    label = str(phone["normalized_label"]).casefold()
    if label in UNKNOWN or not phone["in_frozen_inventory"]:
        return "unknown_or_unfrozen_phone_label"
    if label in SILENCE:
        return "silence_preserved"
    if phone["word_index"] is None:
        return "outside_nonempty_word_interval"
    return "no_lcs_match"


def world_analyze(audio: Any, pyworld: Any, np: Any) -> dict[str, Any]:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("WORLD input is empty or non-finite")
    x64 = x.astype(np.float64)
    f0, time_axis = pyworld.dio(x64, SAMPLE_RATE, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEILING_HZ, frame_period=FRAME_PERIOD_MS)
    f0 = pyworld.stonemask(x64, f0, time_axis, SAMPLE_RATE)
    sp = pyworld.cheaptrick(x64, f0, time_axis, SAMPLE_RATE)
    ap = pyworld.d4c(x64, f0, time_axis, SAMPLE_RATE)
    f0 = np.asarray(f0, dtype=np.float64)
    times = np.asarray(time_axis, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)
    if sp.ndim != 2 or ap.shape != sp.shape or len(f0) != sp.shape[0] or len(times) != len(f0):
        raise ValueError("WORLD F0/time/SP/AP dimensions disagree")
    if not all(np.isfinite(a).all() for a in (f0, times, sp, ap)):
        raise ValueError("WORLD returned non-finite parameters")
    sp = np.maximum(sp, 0.0)
    ap = np.clip(ap, 0.0, 1.0)
    f0 = np.where(f0 > 0.0, f0, 0.0)
    fft_size = int(2 * (sp.shape[1] - 1))
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLE_RATE)
    return {"f0": f0, "time": times, "sp": sp, "ap": ap, "frequency_hz": frequencies, "fft_size": fft_size}


def normalized_sp(sp: Any, np: Any, floor: float = 1e-12) -> Any:
    powers = np.maximum(np.asarray(sp, dtype=np.float64), 0.0)
    totals = np.sum(powers, axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("SP row has zero total power")
    shape = np.maximum(powers / totals, floor)
    return shape / np.sum(shape, axis=1, keepdims=True)


def _interp_rows(values: Any, times: Any, query: float, np: Any) -> Any:
    right = int(np.searchsorted(times, query, side="left"))
    if right <= 0:
        return values[0].copy()
    if right >= len(times):
        return values[-1].copy()
    left = right - 1
    span = float(times[right] - times[left])
    alpha = 0.0 if span <= 0.0 else min(1.0, max(0.0, (query - float(times[left])) / span))
    return values[left] * (1.0 - alpha) + values[right] * alpha


def _voiced_f0_at(frame_times: Any, f0: Any, query: float, interval_indices: Any, np: Any) -> tuple[float, int, bool]:
    candidates = np.asarray(interval_indices, dtype=np.int64)
    if candidates.size == 0:
        return 0.0, -1, False
    local_times = np.asarray(frame_times)[candidates]
    nearest_local = nearest_index(local_times, query)
    nearest = int(candidates[nearest_local])
    if float(f0[nearest]) <= 0.0:
        return 0.0, nearest, False
    # Locate the contiguous voiced run containing the nearest donor center,
    # constrained to this phone interval; unvoiced gaps are never crossed.
    lo = nearest_local
    hi = nearest_local
    while lo > 0 and candidates[lo] == candidates[lo - 1] + 1 and float(f0[candidates[lo - 1]]) > 0.0:
        lo -= 1
    while hi + 1 < len(candidates) and candidates[hi + 1] == candidates[hi] + 1 and float(f0[candidates[hi + 1]]) > 0.0:
        hi += 1
    run_indices = candidates[lo:hi + 1]
    run_times = np.asarray(frame_times)[run_indices]
    log_f0 = np.log(np.asarray(f0)[run_indices])
    value = float(np.exp(np.interp(query, run_times, log_f0)))
    return value, nearest, True


def transfer_parameters(recipient: dict[str, Any], donor: dict[str, Any], mapping: dict[str, Any], np: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    rf0, rt = recipient["f0"], recipient["time"]
    rsp, rap = recipient["sp"], recipient["ap"]
    df0, dt = donor["f0"], donor["time"]
    dsp, dap = donor["sp"], donor["ap"]
    if recipient["fft_size"] != donor["fft_size"] or rsp.shape[1] != dsp.shape[1] or not np.array_equal(recipient["frequency_hz"], donor["frequency_hz"]):
        raise ValueError("recipient and donor WORLD frequency grids differ")
    dshape = normalized_sp(dsp, np)
    rec_shape = normalized_sp(rsp, np)
    frame_count = len(rt)
    mapped = np.zeros(frame_count, dtype=bool)
    mapped_f0 = np.zeros(frame_count, dtype=np.float64)
    mapped_ap = np.zeros_like(rap)
    mapped_shape = np.zeros_like(rec_shape)
    donor_time = np.full(frame_count, np.nan, dtype=np.float64)
    donor_frame = np.full(frame_count, -1, dtype=np.int32)
    f0_voiced = np.zeros(frame_count, dtype=bool)
    shape_sum_error = []
    used_intervals = []
    for phone in mapping["matched_phones"]:
        r0, r1 = map(float, phone["recipient_interval_s"])
        d0, d1 = map(float, phone["donor_interval_s"])
        rmask = (rt >= r0) & (rt < r1)
        if abs(r1 - float(mapping.get("recipient_audio_duration_s", -1.0))) <= 1e-9:
            rmask |= np.abs(rt - r1) <= 1e-9
        rec_indices = np.flatnonzero(rmask)
        donor_indices = np.flatnonzero((dt >= d0) & (dt < d1))
        if abs(d1 - float(mapping.get("donor_audio_duration_s", -1.0))) <= 1e-9:
            donor_indices = np.flatnonzero((dt >= d0) & (dt <= d1))
        if not rec_indices.size or not donor_indices.size or r1 <= r0 or d1 <= d0:
            continue
        local_dtimes = dt[donor_indices]
        for ri in rec_indices:
            query = d0 + (float(rt[ri]) - r0) * (d1 - d0) / (r1 - r0)
            query = min(d1, max(d0, query))
            mapped_ap[ri] = _interp_rows(dap[donor_indices], local_dtimes, query, np)
            log_shape = np.log(np.maximum(dshape[donor_indices], 1e-12))
            row = np.exp(_interp_rows(log_shape, local_dtimes, query, np))
            total = float(np.sum(row))
            if total <= 0.0 or not math.isfinite(total):
                continue
            row /= total
            mapped_shape[ri] = row
            shape_sum_error.append(abs(float(np.sum(row)) - 1.0))
            mapped_f0[ri], donor_frame[ri], f0_voiced[ri] = _voiced_f0_at(dt, df0, query, donor_indices, np)
            donor_time[ri] = query
            mapped[ri] = True
        used_intervals.append(phone["recipient_phone_index"])
    own_voiced = rf0 > 0.0
    joint_voiced = mapped & own_voiced & f0_voiced
    ex_f0, ex_ap = rf0.copy(), rap.copy()
    ex_f0[mapped] = mapped_f0[mapped]
    ex_ap[mapped] = mapped_ap[mapped]
    f0_f0 = rf0.copy()
    f0_f0[joint_voiced] = mapped_f0[joint_voiced]
    ap_ap = rap.copy()
    ap_ap[joint_voiced] = mapped_ap[joint_voiced]
    power = np.sum(rsp, axis=1, keepdims=True)
    mapped_sp = rsp.copy()
    mapped_sp[mapped] = mapped_shape[mapped] * power[mapped]
    arms = {
        "RT": {"f0": rf0.copy(), "sp": rsp.copy(), "ap": rap.copy()},
        "EX": {"f0": ex_f0, "sp": rsp.copy(), "ap": ex_ap},
        "SP": {"f0": rf0.copy(), "sp": mapped_sp, "ap": rap.copy()},
        "BOTH": {"f0": ex_f0.copy(), "sp": mapped_sp.copy(), "ap": ex_ap.copy()},
        "F0": {"f0": f0_f0, "sp": rsp.copy(), "ap": rap.copy()},
        "AP": {"f0": rf0.copy(), "sp": rsp.copy(), "ap": ap_ap},
    }
    power_error_by_frame = np.full(frame_count, np.nan, dtype=np.float64)
    power_error_by_frame[mapped] = np.sum(mapped_sp[mapped], axis=1) - power[mapped, 0]
    power_error = np.abs(power_error_by_frame[mapped]) if np.any(mapped) else np.asarray([], dtype=np.float64)
    metrics = {
        "recipient_world_frames": frame_count,
        "matched_frame_count": int(np.count_nonzero(mapped)),
        "matched_frame_fraction": float(np.mean(mapped)) if frame_count else 0.0,
        "jointly_voiced_frame_count": int(np.count_nonzero(joint_voiced)),
        "jointly_voiced_fraction_of_all_frames": float(np.mean(joint_voiced)) if frame_count else 0.0,
        "jointly_voiced_fraction_of_mapped_frames": float(np.count_nonzero(joint_voiced) / max(1, np.count_nonzero(mapped))),
        "actual_frame_intervention_count": {"EX": int(np.count_nonzero(mapped)), "SP": int(np.count_nonzero(mapped)), "BOTH": int(np.count_nonzero(mapped)), "F0": int(np.count_nonzero(joint_voiced)), "AP": int(np.count_nonzero(joint_voiced))},
        "mapped_recipient_phone_indices_with_frames": sorted(set(map(int, used_intervals))),
        "normalized_sp_sum_abs_error_max": max(shape_sum_error, default=0.0),
        "reconstructed_sp_power_abs_error_max": float(np.max(power_error)) if power_error.size else 0.0,
        "reconstructed_sp_power_relative_error_max": float(np.max(power_error / np.maximum(power[mapped, 0], 1e-20))) if power_error.size else 0.0,
    }
    detail = {"mapped_frames": mapped, "mapped_f0": mapped_f0, "mapped_ap": mapped_ap, "mapped_shape": mapped_shape, "donor_time": donor_time, "donor_frame": donor_frame, "donor_voiced": f0_voiced, "joint_voiced": joint_voiced, "sp_power_error_by_frame": power_error_by_frame}
    return arms, {"metrics": metrics, "detail": detail}


def local_rms_envelope(audio: Any, np: Any) -> tuple[Any, Any]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        raise ValueError("local envelope requires at least two samples")
    radius = ENERGY_WINDOW_RADIUS
    weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(-radius, radius + 1, dtype=np.float64) / radius))
    weights /= np.sum(weights)
    padded = np.pad(x * x, (radius, radius), mode="reflect")
    centers = np.arange(0, len(x), ENERGY_HOP, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)[centers]
    power = windows @ weights
    return centers.astype(np.float64) / SAMPLE_RATE, np.sqrt(np.maximum(power, 0.0))


def local_envelope_match(audio: Any, target_envelope: Any, envelope_times: Any, np: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Apply the frozen 8-pass, +/-3 dB step and +/-12 dB cumulative match."""
    y0 = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(y0)
    target = np.asarray(target_envelope, dtype=np.float64).reshape(-1)
    times = np.asarray(envelope_times, dtype=np.float64).reshape(-1)
    if not n or target.size != times.size or target.size == 0:
        raise ValueError("invalid local envelope target")
    cumulative = np.zeros(n, dtype=np.float64)
    centers_samples = np.rint(times * SAMPLE_RATE).astype(np.int64)
    current = y0.astype(np.float64)
    iteration_rows = []
    scalar_gains = []
    for iteration in range(LOCAL_MATCH_ITERATIONS):
        _, achieved = local_rms_envelope(current, np)
        delta = np.clip(20.0 * np.log10(np.maximum(target, LOCAL_RMS_FLOOR) / np.maximum(achieved, LOCAL_RMS_FLOOR)), -LOCAL_STEP_DB, LOCAL_STEP_DB)
        delta_samples = np.interp(np.arange(n, dtype=np.float64) / SAMPLE_RATE, times, delta, left=float(delta[0]), right=float(delta[-1]))
        cumulative = np.clip(cumulative + delta_samples, -LOCAL_BOUND_DB, LOCAL_BOUND_DB)
        adjusted = y0.astype(np.float64) * np.power(10.0, cumulative / 20.0)
        rms_before_global = float(np.sqrt(np.mean(adjusted * adjusted)))
        if rms_before_global <= 1e-12:
            raise ValueError("local energy matching produced silence")
        global_gain = FIXED_RMS / rms_before_global
        scalar_gains.append(global_gain)
        current = (adjusted * global_gain).astype(np.float32).astype(np.float64)
        iteration_rows.append({
            "iteration": iteration + 1,
            "delta_db_min_p50_max": [float(np.min(delta)), float(np.median(delta)), float(np.max(delta))],
            "global_normalization_gain": global_gain,
            "max_abs_local_gain_db": float(np.max(np.abs(cumulative))),
        })
    _, achieved_final = local_rms_envelope(current, np)
    residual_db = 20.0 * np.log10(np.maximum(achieved_final, LOCAL_RMS_FLOOR) / np.maximum(target, LOCAL_RMS_FLOOR))
    active = target >= LOCAL_RMS_FLOOR
    residual_active = residual_db[active] if np.any(active) else residual_db
    audit_arrays = {"envelope_times_s": times, "target_rms": target.astype(np.float32), "achieved_rms": achieved_final.astype(np.float32), "local_gain_db": cumulative.astype(np.float32)}
    audit = {
        "iterations": iteration_rows,
        "iteration_count": LOCAL_MATCH_ITERATIONS,
        "local_step_db_limit": LOCAL_STEP_DB,
        "cumulative_local_gain_db_limit": LOCAL_BOUND_DB,
        "local_gain_bound_hit_count_samples": int(np.count_nonzero(np.abs(cumulative) >= LOCAL_BOUND_DB - 1e-6)),
        "local_gain_p50_p95_abs_max_db": [float(np.percentile(np.abs(cumulative), 50)), float(np.percentile(np.abs(cumulative), 95)), float(np.max(np.abs(cumulative)))],
        "global_normalization_gains": scalar_gains,
        "target_rms_envelope_active_frames": int(np.count_nonzero(active)),
        "achieved_minus_target_log_rms_db_p50_p95_abs_max_active": [float(np.median(residual_active)), float(np.percentile(np.abs(residual_active), 95)), float(np.max(np.abs(residual_active)))],
        "achieved_minus_target_log_rms_db_p50_p95_abs_max_all": [float(np.median(residual_db)), float(np.percentile(np.abs(residual_db), 95)), float(np.max(np.abs(residual_db)))],
        "fixed_global_rms": float(np.sqrt(np.mean(current * current))),
        "identity_target": bool(np.array_equal(target, local_rms_envelope(y0, np)[1])),
        "max_abs_waveform_change_from_input": float(np.max(np.abs(current - y0))) if len(current) else 0.0,
        "relative_rms_change_from_input": float(np.sqrt(np.mean((current - y0) ** 2)) / max(np.sqrt(np.mean(y0.astype(np.float64) ** 2)), 1e-15)),
    }
    return current.astype(np.float32), audit, audit_arrays


def synthesize_fixed(pyworld: Any, params: dict[str, Any], sample_count: int, np: Any) -> tuple[Any | None, dict[str, Any]]:
    y = np.asarray(pyworld.synthesize(params["f0"], params["sp"], params["ap"], SAMPLE_RATE, frame_period=FRAME_PERIOD_MS), dtype=np.float64).reshape(-1)
    delta = int(len(y) - sample_count)
    if delta < 0:
        return None, {"status": "failed_shorter_than_recipient", "synthesized_samples": int(len(y)), "recipient_samples": int(sample_count), "length_delta_samples": delta}
    discarded = y[sample_count:]
    exact = y[:sample_count]
    before = float(np.sqrt(np.mean(exact * exact)))
    if before <= 1e-12 or not np.isfinite(exact).all():
        return None, {"status": "failed_invalid_synthesis", "synthesized_samples": int(len(y)), "recipient_samples": int(sample_count), "length_delta_samples": delta}
    gain = FIXED_RMS / before
    fixed = (exact * gain).astype(np.float32)
    after = float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2)))
    return fixed, {
        "status": "ok", "synthesized_samples": int(len(y)), "recipient_samples": int(sample_count),
        "length_delta_samples": delta, "right_discarded_tail_samples": int(len(discarded)),
        "discarded_tail_rms": float(np.sqrt(np.mean(discarded * discarded))) if discarded.size else 0.0,
        "input_synthesis_rms_before_global_fix": before, "global_gain_to_fixed_rms": gain,
        "global_gain_db": 20.0 * math.log10(gain), "fixed_rms": after,
        "peak_abs": float(np.max(np.abs(fixed))), "samples_over_full_scale": int(np.count_nonzero(np.abs(fixed) > 1.0)),
    }


def pcm_float32_sha(audio: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(audio, dtype=np.float32).reshape(-1).tobytes(order="C")).hexdigest()


def write_float_wav(path: Path, audio: Any, sf: Any, np: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(path), data, SAMPLE_RATE, subtype="FLOAT")
    reread, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE or reread.shape != (len(data), 1) or not np.array_equal(reread[:, 0], data):
        raise ValueError("saved float WAV did not round-trip bit-identically")
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), "float32_pcm_sha256": pcm_float32_sha(data), "samples": int(len(data)), "sample_rate_hz": SAMPLE_RATE, "channels": 1, "subtype": "FLOAT"}


def _load_mfa_normalizer(producer: dict[str, Any]) -> Any:
    source = Path(producer["normalizer_parser_path"])
    if not source.is_absolute():
        source = REPO_ROOT / source
    if sha256_file(source) != producer["normalizer_parser_sha256"]:
        raise ValueError("MFA publisher-text normalizer source hash differs from frozen protocol")
    spec = importlib.util.spec_from_file_location("world_phone_mfa_normalizer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen MFA normalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _world_runtime(np: Any, pyworld: Any) -> dict[str, Any]:
    ext = getattr(getattr(pyworld, "pyworld", None), "__file__", None)
    pkg = Path(pyworld.__file__).resolve()
    return {
        "python_executable": sys.executable, "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        "pyworld_version": str(getattr(pyworld, "__version__", "unknown")), "pyworld_package_path": str(pkg),
        "pyworld_package_sha256": sha256_file(pkg),
        "pyworld_extension_path": str(Path(ext).resolve()) if ext else None,
        "pyworld_extension_sha256": sha256_file(Path(ext)) if ext and Path(ext).is_file() else None,
        "sample_rate_hz": SAMPLE_RATE, "frame_period_ms": FRAME_PERIOD_MS, "f0_floor_hz": F0_FLOOR_HZ, "f0_ceiling_hz": F0_CEILING_HZ,
        "features": ["DIO", "StoneMask", "CheapTrick", "D4C"],
    }


def _read_audio(path: Path, expected_hash: str, sf: Any, np: Any) -> tuple[Any, dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"frozen staged audio file hash mismatch: {path.name}")
    pcm, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE or pcm.shape[1] != 1 or pcm.shape[0] <= 0:
        raise ValueError(f"audio must decode fully as mono 16 kHz: {path.name}")
    audio = pcm[:, 0].astype(np.float32, copy=False)
    pcm16, rate16 = sf.read(str(path), dtype="int16", always_2d=True)
    if int(rate16) != SAMPLE_RATE or pcm16.shape != pcm.shape:
        raise ValueError(f"PCM16 and float decoder frame counts differ: {path.name}")
    pcm16_bytes = np.asarray(pcm16, dtype="<i2").tobytes(order="C")
    return audio, {
        "path": str(path.resolve()), "file_sha256": expected_hash, "decoded_samples": int(len(audio)),
        "sample_rate_hz": int(rate), "channels": int(pcm.shape[1]), "float32_pcm_sha256": pcm_float32_sha(audio),
        "decoded_pcm16_sha256": hashlib.sha256(pcm16_bytes).hexdigest(),
        "raw_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
    }


def _qualification_map(mfa_out: Path, split: str) -> dict[str, dict[str, Any]]:
    if split == "calibration":
        path = mfa_out / "calibration_qualification_frozen.json"
        if not path.is_file():
            raise FileNotFoundError("CAL MFA qualification must be frozen before WORLD production")
        payload = read_json(path)
        if payload.get("status") != "FROZEN" or payload.get("evaluation_textgrids_read") is not False:
            raise ValueError("CAL qualification freeze is not outcome-blind/frozen")
        return {str(key): value for key, value in payload.get("calibration_pairs", {}).items()}
    path = mfa_out / "evaluation_endpoint_eligibility.json"
    if not path.is_file():
        raise FileNotFoundError("EVAL MFA eligibility must be mechanically scored after the CAL gate")
    payload = read_json(path)
    cal_gate = read_json(mfa_out / "calibration_qualification_frozen.json")
    protocol = read_json(mfa_out / "alignment_protocol_frozen.json")
    if payload.get("calibration_qualification_sha256") != cal_gate.get("manifest_sha256"):
        raise ValueError("EVAL eligibility is not bound to the frozen CAL qualification")
    if payload.get("protocol_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("EVAL eligibility is bound to another MFA protocol")
    return {str(row["id"]): row for row in payload.get("records", [])}


def _make_plan(
    mfa_out: Path,
    producer: dict[str, Any],
    protocol: dict[str, Any],
    world_script: Path,
    spec_path: Path,
    *,
    output_path: Path | None = None,
    alignment_overlay: dict[str, Any] | None = None,
    calibration_plan_sha256: str | None = None,
    calibration_qc_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    spec_sha = sha256_file(spec_path)
    condition_rows = []
    for sample_id, rec in sorted(producer["records"].items()):
        condition_rows.append({
            "id": sample_id, "speaker": rec["speaker"], "split": rec["split"],
        "source_identity_sha256": rec["source_identity_sha256"],
            "natural_raw_audio_sha256": rec["natural_raw_audio_sha256"],
            "canonical_tts_16k_audio_sha256": rec["canonical_tts_16k_audio_sha256"],
            "natural_transcript_sha256": rec["natural_transcript_sha256"],
            "tts_transcript_sha256": rec["tts_transcript_sha256"],
            "natural_textgrid_path_expected": rec["conditions"]["natural"]["expected_textgrid_path"],
            "tts_textgrid_path_expected": rec["conditions"]["tts"]["expected_textgrid_path"],
            "physical_alignment_overlay_applied": bool(alignment_overlay and sample_id in alignment_overlay["record_by_id"]),
        })
    payload = {
        "schema_version": 1, "manifest_type": "world_phone_transfer_plan", "status": "SEALED_BEFORE_ENDPOINT_FORWARD",
        "evaluation_outcomes_read": False,
        "world_producer_source": str(world_script.resolve()), "world_producer_sha256": sha256_file(world_script),
        "world_spec_path": str(spec_path.resolve()), "world_spec_sha256": spec_sha,
        "mfa_protocol_manifest_sha256": protocol["manifest_sha256"],
        "mfa_configuration_sha256": producer.get("mfa_configuration", {}).get("configuration_sha256"),
        "fixed_rms": FIXED_RMS,
        "world": {"pyworld": "0.3.5", "f0_floor_hz": F0_FLOOR_HZ, "f0_ceiling_hz": F0_CEILING_HZ, "frame_period_ms": FRAME_PERIOD_MS, "feature_calls": ["dio", "stonemask", "cheaptrick", "d4c"], "frequency_bins_include_dc_and_nyquist": True},
        "phone_normalizer": "world-phone-identity-v1 (Unicode NFC and outer whitespace removal only)",
        "phone_map_rules": {"word_pairing": "position, labels must be identical and nonempty", "phone_pairing": "order-preserving LCS inside paired word", "lcs_tie_break": "recipient index vector, then donor index vector, lexicographic", "phone_midpoint_assigns_word": True, "boundary_tolerance_s": 0.04, "no_global_shift_dtw_or_waveform_resampling": True, "unknown_oov_and_silence_keep_recipient_parameters": True},
        "parameter_transfer_rules": {"donor_frame_voicing": "nearest WORLD center F0>0; exact ties choose earlier", "f0": "log interpolation only within donor contiguous voiced run; hold within run edges", "ap": "linear in time per bin; clamp numerical excursions to [0,1]", "sp": "interpolate log normalized power with 1e-12 floor; renormalize; multiply by recipient SP sum", "outside_matched_phone": "recipient identity"},
        "arms": list(ARMS), "donor_directions": ["natural_to_tts", "tts_to_natural"],
        "energy_controls": {"window_indices": [-ENERGY_WINDOW_RADIUS, ENERGY_WINDOW_RADIUS], "window_samples": 2 * ENERGY_WINDOW_RADIUS + 1, "hop_samples": ENERGY_HOP, "window_weight": "0.5*(1+cos(pi*j/400)); normalized", "iterations": LOCAL_MATCH_ITERATIONS, "step_db_clip": LOCAL_STEP_DB, "cumulative_db_clip": LOCAL_BOUND_DB, "rms_floor": LOCAL_RMS_FLOOR, "boundary_mode": "reflect", "rt_identity_tolerance_max_abs": 1e-7, "rt_identity_tolerance_relative_rms": 1e-6},
        "conditions": condition_rows,
        "evaluation_waveforms_blocked_until_cal_waveform_qc_freeze": True,
    }
    if alignment_overlay is not None:
        row = alignment_overlay["record_by_id"][next(iter(alignment_overlay["record_by_id"]))]
        condition_rows = [
            {
                **condition_row,
                **({
                    "original_natural_textgrid_path": row["conditions"]["natural"]["original_textgrid_path"],
                    "original_natural_textgrid_sha256": row["conditions"]["natural"]["original_textgrid_sha256"],
                    "physical_natural_lab_path": row["conditions"]["natural"]["physical_lab_path"],
                    "physical_natural_lab_sha256": row["conditions"]["natural"]["physical_lab_sha256"],
                    "physical_natural_textgrid_path": row["conditions"]["natural"]["physical_textgrid_path"],
                    "physical_natural_textgrid_sha256": row["conditions"]["natural"]["physical_textgrid_sha256"],
                    "original_tts_textgrid_path": row["conditions"]["tts"]["original_textgrid_path"],
                    "original_tts_textgrid_sha256": row["conditions"]["tts"]["original_textgrid_sha256"],
                    "physical_tts_lab_path": row["conditions"]["tts"]["physical_lab_path"],
                    "physical_tts_lab_sha256": row["conditions"]["tts"]["physical_lab_sha256"],
                    "physical_tts_textgrid_path": row["conditions"]["tts"]["physical_textgrid_path"],
                    "physical_tts_textgrid_sha256": row["conditions"]["tts"]["physical_textgrid_sha256"],
                    "physical_overlay_record_sha256": row["record_sha256"],
                } if condition_row["id"] == row["id"] else {}),
            }
            for condition_row in condition_rows
        ]
        payload["physical_alignment_overlay"] = {
            "manifest_path": alignment_overlay["path"],
            "manifest_raw_sha256": alignment_overlay["raw_sha256"],
            "manifest_fingerprint": alignment_overlay["manifest"]["manifest_sha256"],
            "alignment_overlay_annex_sha256": alignment_overlay["manifest"]["alignment_overlay_annex_sha256"],
            "execution_bindings_sha256": alignment_overlay["manifest"]["execution_bindings_sha256"],
            "probe_manifest_raw_sha256": alignment_overlay["manifest"]["probe_manifest_raw_sha256"],
            "original_evaluation_eligibility_raw_sha256": alignment_overlay["manifest"]["original_evaluation_eligibility_raw_sha256"],
            "original_coverage_pair_qualified": 79,
            "physical_overlay_pair_qualified": 80,
            "overlay_rows": sorted(alignment_overlay["record_by_id"]),
            "visual_and_source_axis_qualification_changed": False,
        }
        payload["calibration_world_transfer_plan_sha256"] = calibration_plan_sha256
        payload["calibration_qc_freeze_sha256"] = calibration_qc_freeze_sha256
    payload["manifest_sha256"] = fingerprint(payload)
    path = output_path or (mfa_out / "world_phone_transfer_plan.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        old = read_json(path)
        if old.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError("existing WORLD plan differs from current inputs, source or spec")
        return old
    write_json(path, payload)
    return payload


def _model_phone_inventory(producer: dict[str, Any]) -> tuple[set[str], str, str]:
    config = producer.get("mfa_configuration")
    if not config:
        raise ValueError("MFA runtime configuration is absent")
    phone_path = Path(config["acoustic_model_path"]) / "phones.txt"
    if not phone_path.is_file():
        raise FileNotFoundError("frozen acoustic-model phones.txt is missing")
    labels = set()
    for line in phone_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            labels.add(fields[0])
    return labels, str(phone_path.resolve()), sha256_file(phone_path)


def _grid_path(meta: dict[str, Any]) -> Path:
    return Path(meta["expected_textgrid_path"])


def _build_cal_phone_table(mfa_out: Path, producer: dict[str, Any], qualification: dict[str, dict[str, Any]], normalizer: Any, model_inventory: set[str], model_path: str, model_sha: str, np: Any) -> dict[str, Any]:
    labels = set(model_inventory)
    observed = set()
    for sample_id, rec in sorted(producer["records"].items()):
        if rec["split"] != "calibration":
            continue
        for condition in ("natural", "tts"):
            grid_path = _grid_path(rec["conditions"][condition])
            if grid_path.is_file():
                tiers = parse_textgrid(grid_path)
                _, phones = select_tier(tiers, "phones")
                observed.update(str(item["raw_label"]) for item in phones)
    labels.update(observed)
    table = {label: normalize_phone(label) for label in sorted(labels)}
    payload = {
        "schema_version": 1, "manifest_type": "world_phone_normalization_table", "status": "FROZEN_FROM_MODEL_INVENTORY_AND_CAL_ONLY_LABELS",
        "selection_split": "calibration", "evaluation_textgrids_read": False,
        "normalizer": "world-phone-identity-v1", "normalizer_source_sha256": sha256_file(Path(__file__).resolve()),
        "mfa_phone_inventory_path": model_path, "mfa_phone_inventory_sha256": model_sha,
        "mfa_normalizer_parser_sha256": producer["normalizer_parser_sha256"],
        "mfa_calibration_qualification_sha256": sha256_file(mfa_out / "calibration_qualification_frozen.json"),
        "model_inventory_label_count": len(model_inventory), "calibration_label_count": len(observed),
        "raw_to_normalized": table, "unknown_labels": sorted(UNKNOWN), "silence_labels": sorted(SILENCE),
    }
    payload["manifest_sha256"] = fingerprint(payload)
    table_path = mfa_out / "world_phone_normalization_table.json"
    if table_path.is_file():
        prior = read_json(table_path)
        if prior.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError("existing CAL phone normalization table differs; create a new producer version")
        return prior
    write_json(table_path, payload)
    return payload


def _grid_pair(rec: dict[str, Any], input_info: dict[str, dict[str, Any]], normalizer: Any, phone_table: dict[str, Any], rec_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    table = phone_table["raw_to_normalized"]
    lab = rec["conditions"]["natural"]["normalized_lab_text"]
    transcript_tokens = [unicodedata.normalize("NFC", x.strip()) for x in str(lab).split() if x.strip()]
    grids = {}
    for condition, key in (("natural", "natural"), ("tts", "tts")):
        grid = _grid_path(rec["conditions"][condition])
        if not grid.is_file():
            raise FileNotFoundError(f"missing {condition} TextGrid for {rec_id}")
        grids[condition] = prepare_grid(grid, int(input_info[key]["decoded_samples"]), transcript_tokens, set(table))
    if [x["normalized_label"] for x in grids["natural"]["words"]] != [x["normalized_label"] for x in grids["tts"]["words"]]:
        raise ValueError("natural and TTS word tier labels differ")
    return grids["natural"], grids["tts"]


def _load_alignment_overlay(
    overlay_path: Path,
    annex_path: Path,
    bindings_path: Path,
    mfa_out: Path,
    producer: dict[str, Any],
    protocol: dict[str, Any],
    original_qualification: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen physical-only S0064 grid replacement.

    Original EVAL grids and qualification are hashed first. The alternate LAB
    and TextGrids are then exposed as a separate physical alignment view.
    """
    overlay_path = overlay_path.resolve(strict=True)
    annex_path = annex_path.resolve(strict=True)
    bindings_path = bindings_path.resolve(strict=True)
    manifest = read_json(overlay_path)
    if manifest.get("manifest_sha256") != fingerprint(manifest):
        raise ValueError("physical alignment overlay embedded fingerprint mismatch")
    if manifest.get("status") != "FROZEN_PHYSICAL_ONLY" or manifest.get("evaluation_effects_read") is not False:
        raise ValueError("physical alignment overlay is not frozen and outcome-blind")
    if manifest.get("alignment_overlay_annex_sha256") != sha256_file(annex_path):
        raise ValueError("alignment overlay annex hash mismatch")
    if manifest.get("execution_bindings_sha256") != sha256_file(bindings_path):
        raise ValueError("WORLD execution-bindings annex hash mismatch")
    if manifest.get("mfa_producer_manifest_sha256") != producer.get("manifest_sha256"):
        raise ValueError("overlay is bound to a different MFA producer manifest")
    if manifest.get("mfa_protocol_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("overlay is bound to a different MFA protocol")
    eligibility_path = mfa_out / "evaluation_endpoint_eligibility.json"
    eligibility_raw_sha256 = sha256_file(eligibility_path)
    if manifest.get("original_evaluation_eligibility_raw_sha256") != eligibility_raw_sha256:
        raise ValueError("overlay is bound to different original EVAL eligibility bytes")
    probe_path = Path(manifest["probe_manifest_path"]).resolve(strict=True)
    if sha256_file(probe_path) != manifest.get("probe_manifest_raw_sha256"):
        raise ValueError("compound-token probe manifest raw hash mismatch")
    probe = read_json(probe_path)
    if probe.get("manifest_sha256") != manifest.get("probe_manifest_fingerprint"):
        raise ValueError("compound-token probe embedded fingerprint mismatch")
    records = manifest.get("records", [])
    if len(records) != 1:
        raise ValueError("physical-only overlay must contain exactly the frozen S0064 pair")
    producer_records = producer.get("records", {})
    for sample_id, rec in producer_records.items():
        if rec.get("split") != "evaluation":
            continue
        qrow = original_qualification.get(str(sample_id), {})
        for condition in ("natural", "tts"):
            meta = rec["conditions"][condition]
            qcondition = qrow.get(condition, {})
            grid_path = Path(meta["expected_textgrid_path"]).resolve(strict=True)
            expected_grid_sha = str(qcondition.get("textgrid_sha256", ""))
            if not expected_grid_sha or sha256_file(grid_path) != expected_grid_sha:
                raise ValueError(f"sealed original EVAL TextGrid hash mismatch: {sample_id}/{condition}")
    record_by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("id", ""))
        if sample_id != "S0064_BAC009S0064W0121" or row.get("speaker") != "S0064" or row.get("split") != "evaluation":
            raise ValueError("physical overlay scope must be only S0064_BAC009S0064W0121")
        rec = producer_records.get(sample_id)
        qrow = original_qualification.get(sample_id)
        if not rec or not qrow or qrow.get("pair_qualified") is not False:
            raise ValueError("physical overlay must preserve the original unqualified S0064 eligibility row")
        if row.get("source_identity_sha256") != rec.get("source_identity_sha256"):
            raise ValueError("physical overlay source identity differs from the original paired input")
        normalized_old = str(rec["conditions"]["natural"]["normalized_lab_text"])
        normalized_tts = str(rec["conditions"]["tts"]["normalized_lab_text"])
        if "".join(normalized_old.split()) != "".join(normalized_tts.split()):
            raise ValueError("original N/T normalized transcript differs for overlay sample")
        if row.get("publisher_transcript") != probe.get("publisher_transcript"):
            raise ValueError("physical overlay publisher transcript differs from probe")
        transcript_sha = hashlib.sha256(str(row["publisher_transcript"]).encode("utf-8")).hexdigest()
        if transcript_sha != rec.get("natural_transcript_sha256") or transcript_sha != rec.get("tts_transcript_sha256"):
            raise ValueError("physical overlay publisher transcript hash differs from sealed N/T input")
        joined_probe = "".join(str(row.get("physical_lab_text", "")).split())
        if joined_probe != "".join(normalized_old.split()):
            raise ValueError("physical LAB token join changes normalized publisher text")
        probe_conditions = probe.get("conditions", {})
        conditions = row.get("conditions", {})
        if set(conditions) != {"natural", "tts"}:
            raise ValueError("physical overlay needs exactly natural and tts conditions")
        for condition in ("natural", "tts"):
            source_meta = rec["conditions"][condition]
            original_q = qrow.get(condition, {})
            physical = conditions[condition]
            original_grid = Path(source_meta["expected_textgrid_path"]).resolve(strict=True)
            original_actual_sha = sha256_file(original_grid)
            expected_old_sha = str(original_q.get("textgrid_sha256", ""))
            if original_actual_sha != expected_old_sha or physical.get("original_textgrid_sha256") != expected_old_sha:
                raise ValueError(f"original EVAL TextGrid seal changed: {sample_id}/{condition}")
            probe_condition = probe_conditions.get(condition, {})
            expected_source_sha = str(source_meta["source_audio_sha256"])
            if physical.get("source_audio_sha256") != expected_source_sha or probe_condition.get("source_audio_sha256") != expected_source_sha:
                raise ValueError(f"overlay audio identity differs from frozen input: {sample_id}/{condition}")
            grid_path = Path(physical["physical_textgrid_path"]).resolve(strict=True)
            lab_path = Path(physical["physical_lab_path"]).resolve(strict=True)
            if sha256_file(grid_path) != physical.get("physical_textgrid_sha256") or physical.get("physical_textgrid_sha256") != probe_condition.get("textgrid_sha256"):
                raise ValueError(f"overlay TextGrid raw hash mismatch: {sample_id}/{condition}")
            if sha256_file(lab_path) != physical.get("physical_lab_sha256") or physical.get("physical_lab_sha256") != probe_condition.get("lab_sha256"):
                raise ValueError(f"overlay LAB raw hash mismatch: {sample_id}/{condition}")
            if lab_path.read_text(encoding="utf-8").strip() != str(physical.get("physical_lab_text", "")):
                raise ValueError(f"overlay LAB content mismatch: {sample_id}/{condition}")
            if physical.get("physical_lab_text") != probe_condition.get("probe_lab_text"):
                raise ValueError(f"overlay LAB text differs from sealed probe: {sample_id}/{condition}")
            if not probe_condition.get("qualified_by_frozen_structure_rules"):
                raise ValueError(f"probe TextGrid did not pass the frozen structure gate: {sample_id}/{condition}")
            if int(probe_condition.get("unknown_oov_phone_count", -1)) != 0:
                raise ValueError(f"probe TextGrid contains unknown/OOV phones: {sample_id}/{condition}")
            physical["original_textgrid_path"] = str(original_grid)
            physical["original_textgrid_sha256"] = original_actual_sha
            physical["physical_textgrid_path"] = str(grid_path)
            physical["physical_lab_path"] = str(lab_path)
        expected_row_sha = row.get("record_sha256")
        actual_row_sha = fingerprint({key: val for key, val in row.items() if key != "record_sha256"})
        if expected_row_sha is not None and expected_row_sha != actual_row_sha:
            raise ValueError(f"overlay row fingerprint mismatch: {sample_id}")
        row["record_sha256"] = actual_row_sha
        record_by_id[sample_id] = row
    return {
        "path": str(overlay_path),
        "raw_sha256": sha256_file(overlay_path),
        "manifest": manifest,
        "probe_manifest": probe,
        "record_by_id": record_by_id,
        "annex_path": str(annex_path),
        "execution_bindings_path": str(bindings_path),
    }


def _physical_run_views(
    producer: dict[str, Any],
    original_qualification: dict[str, dict[str, Any]],
    overlay: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build an in-memory physical view while leaving sealed MFA inputs intact."""
    run_records = copy.deepcopy(producer["records"])
    qualification = copy.deepcopy(original_qualification)
    original_rows: dict[str, dict[str, Any]] = {}
    if overlay is None:
        return run_records, qualification, original_rows
    for sample_id, overlay_row in overlay["record_by_id"].items():
        rec = run_records[sample_id]
        original_rows[sample_id] = copy.deepcopy(original_qualification[sample_id])
        original_qrow = original_qualification[sample_id]
        for condition in ("natural", "tts"):
            meta = rec["conditions"][condition]
            physical = overlay_row["conditions"][condition]
            meta["physical_original_textgrid_path"] = meta["expected_textgrid_path"]
            meta["physical_original_textgrid_sha256"] = physical["original_textgrid_sha256"]
            meta["expected_textgrid_path"] = physical["physical_textgrid_path"]
            meta["normalized_lab_text"] = physical["physical_lab_text"]
        qualification[sample_id] = {
            **original_qrow,
            "original_pair_qualified": bool(original_qrow.get("pair_qualified")),
            "physical_overlay_used": True,
            "pair_qualified": True,
            "natural": {
                **original_qrow.get("natural", {}), "qualified": True,
                "textgrid_path": overlay_row["conditions"]["natural"]["physical_textgrid_path"],
                "textgrid_sha256": overlay_row["conditions"]["natural"]["physical_textgrid_sha256"],
            },
            "tts": {
                **original_qrow.get("tts", {}), "qualified": True,
                "textgrid_path": overlay_row["conditions"]["tts"]["physical_textgrid_path"],
                "textgrid_sha256": overlay_row["conditions"]["tts"]["physical_textgrid_sha256"],
            },
        }
    return run_records, qualification, original_rows


def _direction_specs(natural_grid: dict[str, Any] | None, tts_grid: dict[str, Any] | None) -> tuple[tuple[str, str, str, Any, Any], ...]:
    """Direction names state donor→recipient, matching the actual audio source."""
    return (
        ("tts_to_natural", "natural", "tts", natural_grid, tts_grid),
        ("natural_to_tts", "tts", "natural", tts_grid, natural_grid),
    )


def _save_feature(path: Path, feature: dict[str, Any], meta: dict[str, Any], np: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, f0=feature["f0"], time=feature["time"], sp=feature["sp"], ap=feature["ap"], frequency_hz=feature["frequency_hz"], fft_size=np.asarray(feature["fft_size"], dtype=np.int32))
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), "feature_shapes": {key: list(feature[key].shape) for key in ("f0", "time", "sp", "ap", "frequency_hz")}, "fft_size": feature["fft_size"], **meta}


def _load_feature(path: Path, np: Any) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in ("f0", "time", "sp", "ap", "frequency_hz")} | {"fft_size": int(data["fft_size"])}


def _save_map(path: Path, mapping: dict[str, Any]) -> dict[str, Any]:
    write_json(path, mapping)
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), "matched_phone_count": mapping["matched_phone_count"]}


def _domain_summary(feature: dict[str, Any], audio: Any, np: Any) -> dict[str, Any]:
    times, f0, sp, ap = feature["time"], feature["f0"], feature["sp"], feature["ap"]
    env_times, env = local_rms_envelope(audio, np)
    env_at_world = np.interp(times, env_times, env, left=float(env[0]), right=float(env[-1]))
    active = env_at_world >= 10.0 ** (-45.0 / 20.0)
    voiced = f0 > 0.0
    shape = normalized_sp(sp, np)
    shape_db = 10.0 * np.log10(np.maximum(shape, 1e-12))
    adjacent = np.abs(np.diff(shape_db, axis=0))
    bands = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8001))
    rows = {}
    frequencies = feature["frequency_hz"]
    for domain_name, mask in (
        ("full_track", np.ones(len(times), dtype=bool)),
        ("fixed_activity", active),
        ("world_voiced", voiced),
        ("fixed_activity_and_world_voiced", active & voiced),
    ):
        rows[domain_name] = {"frame_count": int(np.count_nonzero(mask)), "weight_fraction": float(np.mean(mask)) if len(mask) else 0.0}
        if np.any(mask):
            rows[domain_name]["aperiodicity_mean_p10_p50_p90"] = [float(np.mean(ap[mask])), float(np.percentile(ap[mask], 10)), float(np.percentile(ap[mask], 50)), float(np.percentile(ap[mask], 90))]
            rows[domain_name]["f0_voiced_fraction_within_domain"] = float(np.mean(voiced[mask]))
            for low, high in bands:
                freq_mask = (frequencies >= low) & (frequencies < high)
                rows[domain_name].setdefault("aperiodicity_by_band", {})[f"{low}_{high}Hz"] = {
                    "bin_count": int(np.count_nonzero(freq_mask)),
                    "mean_p50": [float(np.mean(ap[mask][:, freq_mask])), float(np.median(ap[mask][:, freq_mask]))] if np.any(freq_mask) else None,
                }
        else:
            rows[domain_name]["aperiodicity_mean_p10_p50_p90"] = None
            rows[domain_name]["aperiodicity_by_band"] = {}
        if adjacent.size and len(times) > 1:
            pair_mask = mask[1:] & mask[:-1]
            rows[domain_name]["sp_adjacent_local_shape_change_db"] = {
                "frame_pair_count": int(np.count_nonzero(pair_mask)),
                "mean_absolute": float(np.mean(adjacent[pair_mask])) if np.any(pair_mask) else None,
                "p90_absolute": float(np.percentile(adjacent[pair_mask], 90)) if np.any(pair_mask) else None,
            }
        else:
            rows[domain_name]["sp_adjacent_local_shape_change_db"] = {"frame_pair_count": 0, "mean_absolute": None, "p90_absolute": None}
    voiced_values = f0[voiced]
    return {
        "world_frame_count": int(len(times)), "fixed_activity_threshold_dbfs": -45.0,
        "fixed_activity_frame_count": int(np.count_nonzero(active)), "world_voiced_frame_count": int(np.count_nonzero(voiced)),
        "f0_voiced_fraction": float(np.mean(voiced)), "f0_median_hz": float(np.median(voiced_values)) if voiced_values.size else None,
        "f0_std_hz": float(np.std(voiced_values)) if voiced_values.size else None,
        "domains": rows,
    }


def _map_env_ratio(rec_audio: Any, donor_audio: Any, phone_map: dict[str, Any], np: Any) -> tuple[Any, Any, dict[str, Any]]:
    rt, re = local_rms_envelope(rec_audio, np)
    dt, de = local_rms_envelope(donor_audio, np)
    ratio = np.ones(len(rt), dtype=np.float64)
    exposed = np.zeros(len(rt), dtype=bool)
    for phone in phone_map["matched_phones"]:
        r0, r1 = map(float, phone["recipient_interval_s"])
        d0, d1 = map(float, phone["donor_interval_s"])
        selected = (rt >= r0) & (rt < r1)
        if not np.any(selected) or r1 <= r0:
            continue
        mapped_times = d0 + (rt[selected] - r0) * (d1 - d0) / (r1 - r0)
        donor_env = np.interp(mapped_times, dt, de, left=float(de[0]), right=float(de[-1]))
        own_env = np.maximum(re[selected], LOCAL_RMS_FLOOR)
        log_ratio = np.clip(20.0 * np.log10(np.maximum(donor_env, LOCAL_RMS_FLOOR) / own_env), -LOCAL_BOUND_DB, LOCAL_BOUND_DB)
        ratio[selected] = np.power(10.0, log_ratio / 20.0)
        exposed[selected] = True
    return rt, re, {"donor_to_recipient_env_ratio": ratio, "exposed": exposed, "envelope_times_s": rt, "recipient_envelope": re, "donor_envelope_times_s": dt, "donor_envelope": de}


def _save_gain_audit(path: Path, arrays: dict[str, Any], np: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path)}


def _bridge_summary(raw_fixed: Any, rt_fixed: Any, rec_feature: dict[str, Any], pyworld: Any, np: Any) -> dict[str, Any]:
    x, y = np.asarray(raw_fixed, dtype=np.float64), np.asarray(rt_fixed, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("own WORLD reconstruction bridge length mismatch")
    error = y - x
    rms_x = float(np.sqrt(np.mean(x * x)))
    rms_error = float(np.sqrt(np.mean(error * error)))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    rt_feature = world_analyze(rt_fixed, pyworld, np)
    original_shape = normalized_sp(rec_feature["sp"], np)
    rt_shape = normalized_sp(rt_feature["sp"], np)
    frame_count = min(len(original_shape), len(rt_shape))
    log_delta_db = 10.0 * np.log10(np.maximum(rt_shape[:frame_count], 1e-12)) - 10.0 * np.log10(np.maximum(original_shape[:frame_count], 1e-12))
    rt_times, rt_env = local_rms_envelope(rt_fixed, np)
    _, in_env = local_rms_envelope(raw_fixed, np)
    env_count = min(len(rt_env), len(in_env))
    env_db = 20.0 * np.log10(np.maximum(rt_env[:env_count], LOCAL_RMS_FLOOR) / np.maximum(in_env[:env_count], LOCAL_RMS_FLOOR))
    return {
        "input_samples": len(x), "roundtrip_samples": len(y), "length_delta_samples": 0,
        "unaligned_no_shift_snr_db": 20.0 * math.log10(max(rms_x, 1e-15) / max(rms_error, 1e-15)),
        "waveform_error_rms_relative_to_input": rms_error / max(rms_x, 1e-15),
        "max_abs_waveform_error": float(np.max(np.abs(error))), "pearson_r_no_shift": corr,
        "analysis_frame_count_compared": frame_count,
        "normalized_log_sp_abs_difference_db_mean_p50_p90_max": [float(np.mean(np.abs(log_delta_db))), float(np.median(np.abs(log_delta_db))), float(np.percentile(np.abs(log_delta_db), 90)), float(np.max(np.abs(log_delta_db)))],
        "local_log_rms_bridge_db_mean_p50_p90_absmax": [float(np.mean(env_db)), float(np.median(env_db)), float(np.percentile(np.abs(env_db), 90)), float(np.max(np.abs(env_db)))],
        "alignment_shift_or_warp_applied": False,
    }


def _build_split(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        import pyworld
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("use the isolated CPU environment with numpy, soundfile and pyworld 0.3.5") from exc
    if str(getattr(pyworld, "__version__", "")) != "0.3.5":
        raise ValueError(f"expected pyworld 0.3.5, got {getattr(pyworld, '__version__', 'unknown')}")
    mfa_out = args.mfa_outdir.resolve()
    out = args.outdir.resolve()
    producer = read_json(mfa_out / "mfa_producer_manifest.json")
    protocol = read_json(mfa_out / "alignment_protocol_frozen.json")
    if producer.get("protocol_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("MFA producer/protocol hash binding mismatch")
    if producer.get("manifest_sha256") != fingerprint(producer):
        raise ValueError("MFA producer manifest fingerprint mismatch")
    if protocol.get("manifest_sha256") != fingerprint(protocol):
        raise ValueError("MFA protocol manifest fingerprint mismatch")
    cal_freeze = None
    calibration_plan = None
    if args.split == "evaluation":
        if not args.cal_qc_freeze:
            raise ValueError("evaluation waveform production requires --cal-qc-freeze")
        cal_freeze = read_json(args.cal_qc_freeze.resolve())
        cal_path = mfa_out / "world_phone_transfer_calibration.json"
        cal_report = read_json(cal_path)
        if cal_freeze.get("status") != "FROZEN" or cal_freeze.get("calibration_producer_manifest_sha256") != cal_report.get("manifest_sha256"):
            raise ValueError("CAL waveform-QC freeze does not bind the current calibration producer")
        calibration_plan = read_json(mfa_out / "world_phone_transfer_plan.json")
        if cal_freeze.get("world_transfer_plan_sha256") != calibration_plan.get("manifest_sha256"):
            raise ValueError("CAL waveform-QC freeze does not bind the frozen WORLD plan")
        if cal_freeze.get("calibration_qc_sha256") != cal_report.get("calibration_waveform_qc", {}).get("manifest_sha256"):
            raise ValueError("CAL waveform-QC freeze does not bind the calibration QA manifest")
    original_qualification = _qualification_map(mfa_out, args.split)
    alignment_overlay = None
    if args.alignment_overlay_manifest:
        if args.split != "evaluation":
            raise ValueError("physical alignment overlay can only be used for evaluation")
        if not args.alignment_overlay_annex or not args.execution_bindings:
            raise ValueError("physical overlay requires its frozen annex and execution-bindings annex")
        alignment_overlay = _load_alignment_overlay(
            args.alignment_overlay_manifest,
            args.alignment_overlay_annex,
            args.execution_bindings,
            mfa_out,
            producer,
            protocol,
            original_qualification,
        )
    run_records, qualification, original_qualification_rows = _physical_run_views(producer, original_qualification, alignment_overlay)
    normalizer = _load_mfa_normalizer(producer)
    model_inventory, model_phone_path, model_phone_sha = _model_phone_inventory(producer)
    script_path = Path(__file__).resolve()
    spec_path = args.spec.resolve()
    plan_path = (
        mfa_out / "world_phone_transfer_evaluation_physical_overlay_plan.json"
        if alignment_overlay is not None
        else mfa_out / "world_phone_transfer_plan.json"
    )
    plan = _make_plan(
        mfa_out,
        producer,
        protocol,
        script_path,
        spec_path,
        output_path=plan_path,
        alignment_overlay=alignment_overlay,
        calibration_plan_sha256=calibration_plan.get("manifest_sha256") if calibration_plan else None,
        calibration_qc_freeze_sha256=sha256_file(args.cal_qc_freeze.resolve()) if args.split == "evaluation" else None,
    )
    if args.split == "calibration":
        phone_table = _build_cal_phone_table(mfa_out, producer, qualification, normalizer, model_inventory, model_phone_path, model_phone_sha, np)
    else:
        phone_table_path = mfa_out / "world_phone_normalization_table.json"
        phone_table = read_json(phone_table_path)
        if phone_table.get("status") != "FROZEN_FROM_MODEL_INVENTORY_AND_CAL_ONLY_LABELS":
            raise ValueError("CAL phone normalization table is absent or not frozen")
    records = [rec for rec in run_records.values() if rec["split"] == args.split]
    if args.sample_id:
        records = [rec for rec in records if str(rec["id"]) == args.sample_id]
        if len(records) != 1:
            raise ValueError(f"--sample-id must identify exactly one {args.split} row: {args.sample_id}")
    expected = 20 if args.split == "calibration" else 80
    if not args.sample_id and len(records) != expected:
        raise ValueError(f"expected {expected} {args.split} records, got {len(records)}")
    runtime = _world_runtime(np, pyworld)
    result_rows = []
    all_failures = []
    for record_index, rec in enumerate(sorted(records, key=lambda x: x["id"]), 1):
        sample_id = str(rec["id"])
        qrow = qualification.get(sample_id, {})
        original_qrow = original_qualification_rows.get(sample_id, original_qualification.get(sample_id, {}))
        pair_qualified = bool(qrow.get("pair_qualified"))
        physical_overlay_used = bool(alignment_overlay and sample_id in alignment_overlay["record_by_id"])
        sides: dict[str, Any] = {}
        features: dict[str, Any] = {}
        fixed_waves: dict[str, Any] = {}
        input_info: dict[str, Any] = {}
        for condition in ("natural", "tts"):
            meta = rec["conditions"][condition]
            source, source_info = _read_audio(Path(meta["staged_audio_path"]), meta["staged_audio_sha256"], sf, np)
            if source_info["decoded_samples"] != int(qrow.get(condition, {}).get("audio_decoded_frames", source_info["decoded_samples"])):
                raise ValueError(f"audio sample count differs from MFA gate: {sample_id}/{condition}")
            raw_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
            if raw_rms <= 1e-12:
                raise ValueError(f"silent source waveform: {sample_id}/{condition}")
            gain = FIXED_RMS / raw_rms
            fixed = (source.astype(np.float64) * gain).astype(np.float32)
            fixed_rms = float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2)))
            if abs(fixed_rms - FIXED_RMS) > 1e-7:
                fixed = (fixed.astype(np.float64) * (FIXED_RMS / fixed_rms)).astype(np.float32)
                fixed_rms = float(np.sqrt(np.mean(fixed.astype(np.float64) ** 2)))
            fixed_path = out / "fixed_audio" / args.split / sample_id / f"{condition}.wav"
            fixed_meta = write_float_wav(fixed_path, fixed, sf, np)
            feature = world_analyze(fixed, pyworld, np)
            feature_path = out / "features" / args.split / sample_id / f"{condition}.npz"
            feature_meta = _save_feature(feature_path, feature, {}, np)
            sides[condition] = {
                "source_audio_path": meta["source_path"], "source_audio_sha256": meta["source_audio_sha256"],
                "staged_audio_path": source_info["path"], "staged_audio_sha256": source_info["file_sha256"],
                "source_identity_sha256": rec["source_identity_sha256"],
                "raw_decoded_samples": source_info["decoded_samples"], "sample_rate_hz": SAMPLE_RATE,
                "raw_float32_pcm_sha256": source_info["float32_pcm_sha256"],
                "raw_decoded_pcm16_sha256": source_info["decoded_pcm16_sha256"], "raw_rms": raw_rms,
                "fixed_gain": gain, "fixed_gain_db": 20.0 * math.log10(gain), "fixed_rms": fixed_rms,
                "fixed_audio": fixed_meta, "world_features": feature_meta,
                "calibration_domain_descriptors": _domain_summary(feature, fixed, np) if args.split == "calibration" else None,
                "textgrid_path": meta["expected_textgrid_path"], "textgrid_sha256": qrow.get(condition, {}).get("textgrid_sha256"),
                "original_textgrid_path": meta.get("physical_original_textgrid_path", meta["expected_textgrid_path"]),
                "original_textgrid_sha256": meta.get("physical_original_textgrid_sha256", original_qrow.get(condition, {}).get("textgrid_sha256")),
                "physical_alignment_overlay_used": physical_overlay_used,
                "condition_qualified": bool(qrow.get(condition, {}).get("qualified", pair_qualified)),
            }
            features[condition] = feature
            fixed_waves[condition] = fixed
            input_info[condition] = source_info
        try:
            natural_grid, tts_grid = _grid_pair(rec, input_info, normalizer, phone_table, sample_id)
            grid_error = None
        except Exception as exc:
            natural_grid = tts_grid = None
            grid_error = f"{type(exc).__name__}:{exc}"
        directions: dict[str, Any] = {}
        alignment_pair_ok = pair_qualified and natural_grid is not None and tts_grid is not None
        pair_ok = alignment_pair_ok
        direction_specs = _direction_specs(natural_grid, tts_grid)
        for direction, rec_cond, don_cond, rec_grid, don_grid in direction_specs:
            direction_dir = out / "directions" / args.split / sample_id / direction
            map_path = direction_dir / "phone_map.json"
            if alignment_pair_ok:
                try:
                    phone_map = map_phone_intervals(rec_grid, don_grid)
                    phone_map.update({
                        "id": sample_id, "speaker": rec["speaker"], "split": args.split,
                        "recipient_condition": rec_cond, "donor_condition": don_cond,
                        "recipient_source_pcm_sha256": rec["conditions"][rec_cond]["source_audio_sha256"],
                        "donor_source_pcm_sha256": rec["conditions"][don_cond]["source_audio_sha256"],
                        "recipient_textgrid_sha256": rec_grid["textgrid_sha256"], "donor_textgrid_sha256": don_grid["textgrid_sha256"],
                        "mfa_protocol_sha256": protocol["manifest_sha256"],
                        "mfa_configuration_sha256": producer.get("mfa_configuration", {}).get("configuration_sha256"),
                        "phone_normalization_table_sha256": phone_table["manifest_sha256"],
                        "normalizer_source_sha256": sha256_file(script_path),
                        "mfa_parser_source_sha256": producer["normalizer_parser_sha256"],
                        "pair_qualified": True,
                    })
                except Exception as exc:
                    phone_map = {"matched_phones": [], "matched_phone_count": 0, "unmatched_recipient_phone_count": 0, "unmatched_donor_phone_count": 0, "status": "failed", "failure": f"{type(exc).__name__}:{exc}", "pair_qualified": False}
            else:
                phone_map = {"matched_phones": [], "matched_phone_count": 0, "unmatched_recipient_phone_count": 0, "unmatched_donor_phone_count": 0, "status": "alignment_unqualified_or_missing", "failure": grid_error or "MFA pair qualification failed", "pair_qualified": False}
            phone_map.update({"recipient_condition": rec_cond, "donor_condition": don_cond})
            map_meta = _save_map(map_path, phone_map)
            if phone_map.get("status") == "failed":
                pair_ok = False
            directions[direction] = {"recipient_condition": rec_cond, "donor_condition": don_cond, "phone_map": map_meta, "mapping": phone_map, "mapping_metrics": None, "arms": {}, "qc": {}}
            if not phone_map.get("pair_qualified"):
                all_failures.append({"id": sample_id, "direction": direction, "failure": phone_map.get("failure")})
            target_feature, donor_feature = features[rec_cond], features[don_cond]
            params_by_arm, transfer = transfer_parameters(target_feature, donor_feature, phone_map, np)
            directions[direction]["mapping_metrics"] = transfer["metrics"]
            detail = transfer["detail"]
            parameter_dir = direction_dir / "parameters"
            parameter_dir.mkdir(parents=True, exist_ok=True)
            transfer_audit_path = parameter_dir / "transfer_audit.npz"
            np.savez_compressed(transfer_audit_path, **{key: np.asarray(value) for key, value in detail.items()})
            directions[direction]["parameter_transfer_audit"] = {"path": str(transfer_audit_path.resolve()), "file_sha256": sha256_file(transfer_audit_path), "array_shapes": {key: list(np.asarray(value).shape) for key, value in detail.items()}}
            synthesized: dict[str, Any] = {}
            for arm in ("RT", "EX", "SP", "BOTH", "F0", "AP"):
                parameter_path = parameter_dir / f"{arm}.npz"
                arm_params = params_by_arm[arm]
                np.savez_compressed(parameter_path, f0=arm_params["f0"], time=target_feature["time"], sp=arm_params["sp"], ap=arm_params["ap"], frequency_hz=target_feature["frequency_hz"])
                parameter_meta = {"path": str(parameter_path.resolve()), "file_sha256": sha256_file(parameter_path), "fft_size": target_feature["fft_size"], "f0_shape": list(arm_params["f0"].shape), "time_shape": list(target_feature["time"].shape), "sp_shape": list(arm_params["sp"].shape), "ap_shape": list(arm_params["ap"].shape), "frequency_grid_sha256": hashlib.sha256(np.asarray(target_feature["frequency_hz"], dtype=np.float64).tobytes()).hexdigest()}
                wave, synth_meta = synthesize_fixed(pyworld, params_by_arm[arm], len(fixed_waves[rec_cond]), np)
                if wave is None:
                    directions[direction]["arms"][arm] = {**synth_meta, "status": "failed", "parameters": parameter_meta}
                    all_failures.append({"id": sample_id, "direction": direction, "arm": arm, "failure": synth_meta})
                    continue
                wav_path = direction_dir / "audio" / f"{arm}.wav"
                saved = write_float_wav(wav_path, wave, sf, np)
                arm_meta = {**synth_meta, "status": "ok", **saved, "parameters": parameter_meta}
                synthesized[arm] = wave
                directions[direction]["arms"][arm] = arm_meta
            if "RT" not in synthesized:
                continue
            rt_repeat, repeat_meta = synthesize_fixed(pyworld, params_by_arm["RT"], len(fixed_waves[rec_cond]), np)
            if rt_repeat is not None:
                repeat_path = direction_dir / "audio" / "RT_REPEAT.wav"
                repeat_saved = write_float_wav(repeat_path, rt_repeat, sf, np)
                diff = rt_repeat.astype(np.float64) - synthesized["RT"].astype(np.float64)
                repeat_meta.update({
                    "bit_identical": bool(np.array_equal(rt_repeat, synthesized["RT"])),
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "rms_difference": float(np.sqrt(np.mean(diff * diff))),
                    "repeat_audio": repeat_saved,
                })
            directions[direction]["qc"]["rt_repeat"] = repeat_meta
            if rt_repeat is None:
                all_failures.append({"id": sample_id, "direction": direction, "arm": "RT_REPEAT", "failure": repeat_meta})
            directions[direction]["qc"]["raw_to_rt_bridge"] = _bridge_summary(fixed_waves[rec_cond], synthesized["RT"], target_feature, pyworld, np)
            rt_times, rt_env = local_rms_envelope(synthesized["RT"], np)
            rt_identity, rt_identity_audit, rt_identity_arrays = local_envelope_match(synthesized["RT"], rt_env, rt_times, np)
            identity_error = float(np.max(np.abs(rt_identity - synthesized["RT"])))
            identity_relative_rms = float(np.sqrt(np.mean((rt_identity.astype(np.float64) - synthesized["RT"].astype(np.float64)) ** 2)) / max(np.sqrt(np.mean(synthesized["RT"].astype(np.float64) ** 2)), 1e-15))
            rt_identity_audit["identity_contract_max_abs_tolerance"] = 1e-7
            rt_identity_audit["identity_contract_relative_rms_tolerance"] = 1e-6
            rt_identity_audit["identity_max_abs_error"] = identity_error
            rt_identity_audit["identity_relative_rms_error"] = identity_relative_rms
            rt_identity_audit["identity_contract_pass"] = identity_error <= 1e-7 and identity_relative_rms <= 1e-6
            if not rt_identity_audit["identity_contract_pass"]:
                all_failures.append({"id": sample_id, "direction": direction, "arm": "RT_ENVELOPE_IDENTITY", "failure": "identity tolerance exceeded", "max_abs_error": identity_error, "relative_rms_error": identity_relative_rms})
            identity_gain_path = direction_dir / "gain_audits" / "RT_ENVELOPE_IDENTITY.npz"
            rt_identity_audit["gain_audit"] = _save_gain_audit(identity_gain_path, rt_identity_arrays, np)
            identity_wave_path = direction_dir / "audio" / "RT_ENVELOPE_IDENTITY.wav"
            rt_identity_audit["audio"] = write_float_wav(identity_wave_path, rt_identity, sf, np)
            directions[direction]["qc"]["rt_envelope_identity"] = rt_identity_audit
            for arm, energy_arm in (("EX", "EX_EM"), ("SP", "SP_EM")):
                if arm not in synthesized:
                    continue
                matched, audit, arrays = local_envelope_match(synthesized[arm], rt_env, rt_times, np)
                gain_path = direction_dir / "gain_audits" / f"{energy_arm}.npz"
                gain_meta = _save_gain_audit(gain_path, arrays, np)
                wav_path = direction_dir / "audio" / f"{energy_arm}.wav"
                saved = write_float_wav(wav_path, matched, sf, np)
                directions[direction]["arms"][energy_arm] = {"status": "ok", **saved, "energy_match_audit": audit, "gain_audit": gain_meta, "source_arm": arm}
            env_ratio_times, rec_fixed_env, ratio_meta = _map_env_ratio(fixed_waves[rec_cond], fixed_waves[don_cond], phone_map, np)
            target_e = rt_env * ratio_meta["donor_to_recipient_env_ratio"]
            e_audio, e_audit, e_arrays = local_envelope_match(synthesized["RT"], target_e, env_ratio_times, np)
            gain_path = direction_dir / "gain_audits" / "E.npz"
            gain_meta = _save_gain_audit(gain_path, {**e_arrays, "intended_ratio": ratio_meta["donor_to_recipient_env_ratio"].astype(np.float32), "exposed": ratio_meta["exposed"]}, np)
            e_path = direction_dir / "audio" / "E.wav"
            e_saved = write_float_wav(e_path, e_audio, sf, np)
            directions[direction]["arms"]["E"] = {"status": "ok", **e_saved, "energy_match_audit": e_audit, "gain_audit": gain_meta, "intended_envelope_ratio_exposed_frames": int(np.count_nonzero(ratio_meta["exposed"]))}
        result_rows.append({
            "id": sample_id, "speaker": rec["speaker"], "split": args.split,
            "source_identity_sha256": rec["source_identity_sha256"], "pair_qualified": pair_ok,
            "mfa_pair_qualified": bool(original_qrow.get("pair_qualified")),
            "physical_alignment_pair_qualified": pair_qualified,
            "physical_alignment_overlay_used": physical_overlay_used,
            "original_mfa_failure_reasons": {
                condition: original_qrow.get(condition, {}).get("errors", [])
                for condition in ("natural", "tts")
            },
            "grid_failure": grid_error,
            "physical_overlay_record_sha256": alignment_overlay["record_by_id"][sample_id]["record_sha256"] if physical_overlay_used else None,
            "natural": sides["natural"], "tts": sides["tts"], "directions": directions,
        })
        print(f"WORLD {args.split} {record_index}/{len(records)} {sample_id}", flush=True)
    payload = {
        "schema_version": 1, "manifest_type": f"world_phone_transfer_{args.split}_producer",
        "status": ("CAL_SMOKE_WAVEFORMS_READY_FOR_ENGINEERING_REVIEW" if args.sample_id else "CAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW") if args.split == "calibration" else "EVALUATION_WAVEFORMS_READY",
        "producer_scope": "single_calibration_engineering_row" if args.sample_id else "full_split",
        "smoke_sample_id": args.sample_id,
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "evaluation_outcomes_read": False, "evaluation_waveforms_read": args.split == "evaluation",
        "world_transfer_plan_path": str(plan_path.resolve()),
        "world_transfer_plan_sha256": plan["manifest_sha256"],
        "mfa_producer_manifest_sha256": producer["manifest_sha256"],
        "mfa_protocol_manifest_sha256": protocol["manifest_sha256"],
        "mfa_qualification_manifest_sha256": sha256_file(mfa_out / ("calibration_qualification_frozen.json" if args.split == "calibration" else "evaluation_endpoint_eligibility.json")),
        "mfa_configuration": producer.get("mfa_configuration"),
        "phone_normalization_table_path": str((mfa_out / "world_phone_normalization_table.json").resolve()),
        "phone_normalization_table_sha256": phone_table["manifest_sha256"],
        "phone_normalizer_source_sha256": sha256_file(script_path),
        "world_runtime": runtime, "fixed_rms": FIXED_RMS,
        "arms": list(ARMS), "donor_directions": ["natural_to_tts", "tts_to_natural"],
        "energy_identity_qc": "RT_ENVELOPE_IDENTITY uses the same eight-iteration routine; max abs <=1e-7 and relative RMS <=1e-6",
        "sample_counts": {
            "intended": len(result_rows),
            "mfa_pair_qualified_original": sum(bool(x["mfa_pair_qualified"]) for x in result_rows),
            "physical_alignment_pair_qualified": sum(bool(x["physical_alignment_pair_qualified"]) for x in result_rows),
            "physical_overlay_rows": sum(bool(x["physical_alignment_overlay_used"]) for x in result_rows),
            "phone_map_failures": len(all_failures),
        },
        "records": result_rows, "failures": all_failures,
    }
    if alignment_overlay is not None:
        payload["physical_alignment_overlay"] = {
            "manifest_path": alignment_overlay["path"],
            "manifest_raw_sha256": alignment_overlay["raw_sha256"],
            "manifest_fingerprint": alignment_overlay["manifest"]["manifest_sha256"],
            "alignment_overlay_annex_path": alignment_overlay["annex_path"],
            "alignment_overlay_annex_sha256": alignment_overlay["manifest"]["alignment_overlay_annex_sha256"],
            "execution_bindings_path": alignment_overlay["execution_bindings_path"],
            "execution_bindings_sha256": alignment_overlay["manifest"]["execution_bindings_sha256"],
            "probe_manifest_path": alignment_overlay["manifest"]["probe_manifest_path"],
            "probe_manifest_raw_sha256": alignment_overlay["manifest"]["probe_manifest_raw_sha256"],
            "original_pair_qualification_preserved": True,
            "original_qualified_pairs": sum(bool(x.get("pair_qualified")) for x in original_qualification.values()),
            "physical_overlay_qualified_pairs": sum(bool(x["physical_alignment_pair_qualified"]) for x in result_rows),
            "visual_source_axis_eligibility_changed": False,
        }
    payload["manifest_sha256"] = fingerprint(payload)
    suffix = "_smoke" if args.sample_id else ""
    filename = (
        "world_phone_transfer_evaluation_physical_overlay.json"
        if args.split == "evaluation" and alignment_overlay is not None
        else f"world_phone_transfer_{args.split}{suffix}.json"
    )
    out_path = mfa_out / filename
    if out_path.is_file() and not args.force:
        prior = read_json(out_path)
        if prior.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError("existing WORLD producer manifest differs; use a new output root or --force")
    write_json(out_path, payload)
    if args.split == "calibration":
        qc = {
            "schema_version": 1, "manifest_type": "world_calibration_waveform_qc",
            "status": "ENGINEERING_SMOKE" if args.sample_id else "READY_FOR_INDEPENDENT_REVIEW", "producer_manifest_sha256": payload["manifest_sha256"],
            "world_transfer_plan_sha256": plan["manifest_sha256"], "evaluation_waveforms_read": False,
            "evaluation_effects_read": False,
            "intended_rows": len(result_rows), "phone_map_failure_count": len(all_failures),
            "rt_repeat_max_abs_error": max((r["qc"]["rt_repeat"].get("max_abs_difference", 0.0) for row in result_rows for r in row["directions"].values()), default=0.0),
            "rt_bridge_snr_db_p10_p50_p90": _quantiles([r["qc"]["raw_to_rt_bridge"].get("unaligned_no_shift_snr_db") for row in result_rows for r in row["directions"].values() if r["qc"].get("raw_to_rt_bridge")], np),
            "local_energy_controls": _local_control_summary(result_rows, np),
        }
        qc["manifest_sha256"] = fingerprint(qc)
        qc_path = mfa_out / f"world_phone_transfer_calibration{suffix}_qc.json"
        write_json(qc_path, qc)
        payload["calibration_waveform_qc"] = qc
        payload["manifest_sha256"] = fingerprint(payload)
        write_json(out_path, payload)
    print(json.dumps({"split": args.split, "records": len(result_rows), "mfa_pair_qualified_original": payload["sample_counts"]["mfa_pair_qualified_original"], "physical_alignment_pair_qualified": payload["sample_counts"]["physical_alignment_pair_qualified"], "failures": len(all_failures), "manifest_sha256": payload["manifest_sha256"], "out": str(out_path)}, separators=(",", ":")))
    return 0


def _quantiles(values: list[Any], np: Any) -> list[float | None]:
    clean = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    return [float(x) for x in np.percentile(clean, [10, 50, 90])] if clean.size else [None, None, None]


def _local_control_summary(records: list[dict[str, Any]], np: Any) -> dict[str, Any]:
    result = {}
    for arm in ("EX_EM", "SP_EM", "E"):
        audits = [direction["arms"].get(arm, {}).get("energy_match_audit", {}) for record in records for direction in record["directions"].values()]
        residual = [audit.get("achieved_minus_target_log_rms_db_p50_p95_abs_max_active", [None, None, None])[1] for audit in audits]
        bound_counts = [audit.get("local_gain_bound_hit_count_samples", 0) for audit in audits]
        result[arm] = {
            "rows_intended": len(audits), "rows_with_audit": sum(bool(audit) for audit in audits),
            "row_active_residual_abs_p95_db_p10_p50_p90": _quantiles(residual, np),
            "bound_hit_sample_count_total": int(sum(bound_counts)),
        }
    repeat_audits = [direction.get("qc", {}).get("rt_repeat", {}) for record in records for direction in record["directions"].values()]
    repeat_errors = [audit.get("max_abs_difference") for audit in repeat_audits if audit.get("max_abs_difference") is not None]
    identity_audits = [direction.get("qc", {}).get("rt_envelope_identity", {}) for record in records for direction in record["directions"].values()]
    identity_errors = [audit.get("identity_max_abs_error") for audit in identity_audits if audit.get("identity_max_abs_error") is not None]
    identity_relative_errors = [audit.get("identity_relative_rms_error") for audit in identity_audits if audit.get("identity_relative_rms_error") is not None]
    result["RT_REPEAT"] = {
        "rows_intended": len(repeat_audits), "rows_with_synthesis": len(repeat_errors),
        "bit_identical_count": sum(bool(audit.get("bit_identical")) for audit in repeat_audits),
        "max_abs_error": max(repeat_errors, default=None),
    }
    result["RT_ENVELOPE_IDENTITY"] = {
        "rows_intended": len(identity_audits),
        "rows_passed": sum(bool(audit.get("identity_contract_pass")) for audit in identity_audits),
        "max_abs_error": max(identity_errors, default=None),
        "max_relative_rms_error": max(identity_relative_errors, default=None),
    }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mfa-outdir", type=Path, required=True, help="MFA producer output containing frozen TextGrids and gates")
    parser.add_argument("--outdir", type=Path, required=True, help="directory for WORLD feature and audio artifacts")
    parser.add_argument("--spec", type=Path, default=REPO_ROOT / "openspec/changes/confirm-world-source-filter-20261001/spec.md")
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--cal-qc-freeze", type=Path, help="independent reviewer freeze JSON; required for evaluation waveforms")
    parser.add_argument("--alignment-overlay-manifest", type=Path, help="frozen physical-only S0064 LAB/TextGrid overlay manifest")
    parser.add_argument("--alignment-overlay-annex", type=Path, help="root-frozen alignment-overlay.md")
    parser.add_argument("--execution-bindings", type=Path, help="root-frozen WORLD execution-bindings.md")
    parser.add_argument("--sample-id", help="run one CAL engineering smoke row into a separate manifest/artifact root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return _build_split(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
