#!/usr/bin/env python3
"""Audio-only artifact audit for the frozen AISHELL-1 n25 MFA-linear cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, get_window

REPO = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000
ARMS = ("natural", "raw_tts", "mfa_linear_resample_poly")
SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
FRAME_LENGTH = 512
HOP_LENGTH = 160
NFFT = 1024
PLOT_SAMPLES = 25


@dataclass(frozen=True)
class AuditConfig:
    cohort: Path
    raw_tts: Path
    tokens: Path
    mfa_linear: Path
    syncnet: Path
    outdir: Path
    make_plots: bool = True


@dataclass
class AudioData:
    values: np.ndarray
    sample_rate: int
    path: Path
    sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> float | int | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return value.item() if isinstance(value, (np.integer, np.floating)) else value
    return None


def finite_list(values: Iterable[Any]) -> list[float | int | None]:
    return [finite_number(value) for value in values]


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return finite_number(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_audio(path: Path, expected_sha: str) -> AudioData:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha):
        raise ValueError(f"audio hash mismatch: {path}")
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim != 1 or int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"expected mono 16 kHz audio: {path} rate={sample_rate} ndim={values.ndim}")
    values = np.asarray(values, dtype=np.float32)
    if values.size < FRAME_LENGTH or not np.isfinite(values).all():
        raise ValueError(f"invalid audio samples: {path}")
    return AudioData(values, int(sample_rate), path, actual_sha)


def _frame_matrix(values: np.ndarray, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH) -> np.ndarray:
    if values.size < frame_length:
        padded = np.pad(values, (0, frame_length - values.size))
    else:
        padded = values
    count = 1 + max(0, (padded.size - frame_length) // hop_length)
    needed = (count - 1) * hop_length + frame_length
    if needed > padded.size:
        padded = np.pad(padded, (0, needed - padded.size))
    starts = np.arange(count)[:, None] * hop_length + np.arange(frame_length)[None, :]
    return padded[starts]


def _spectral_features(values: np.ndarray) -> dict[str, Any]:
    frames = _frame_matrix(values)
    windowed = frames * get_window("hann", FRAME_LENGTH, fftbins=True)[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, n=NFFT, axis=1))
    power = spectrum * spectrum + 1e-12
    frequencies = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    totals = power.sum(axis=1)
    centroid = (power * frequencies[None, :]).sum(axis=1) / totals
    cumulative = np.cumsum(power, axis=1)
    rolloff = np.array([frequencies[min(np.searchsorted(row, total * 0.85), frequencies.size - 1)] for row, total in zip(cumulative, totals, strict=True)])
    geometric = np.exp(np.mean(np.log(power), axis=1))
    flatness = geometric / np.mean(power, axis=1)
    normalized = power / totals[:, None]
    spectral_entropy = -np.sum(normalized * np.log(normalized + 1e-12), axis=1) / np.log(normalized.shape[1])
    flux = np.concatenate([[0.0], np.mean(np.abs(np.diff(normalized, axis=0)), axis=1)])
    bands = ((0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 6000), (6000, 8000), (8000, 12000), (12000, 16000))
    band_energy = {f"band_{lo}_{hi}_hz": power[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1) / totals for lo, hi in bands}
    line_bins = []
    for fundamental in (50.0, 60.0):
        for harmonic in range(1, 9):
            target = fundamental * harmonic
            line_bins.append(np.exp(-0.5 * ((frequencies - target) / 12.0) ** 2))
    line_kernel = np.max(np.stack(line_bins), axis=0)
    narrowband = (power * line_kernel[None, :]).sum(axis=1) / totals
    return {
        "frames": frames,
        "spectrum": spectrum,
        "power": power,
        "frequencies": frequencies,
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "spectral_entropy": spectral_entropy,
        "flux": flux,
        "band_energy": band_energy,
        "narrowband_50_60_hz": narrowband,
    }


def _rough_f0(values: np.ndarray) -> tuple[float | None, float | None, float]:
    frames = _frame_matrix(values).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    spectrum = np.fft.rfft(frames, n=2 * FRAME_LENGTH, axis=1)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2, n=2 * FRAME_LENGTH, axis=1)[:, :FRAME_LENGTH]
    autocorrelation /= autocorrelation[:, :1] + 1e-12
    min_lag, max_lag = int(SAMPLE_RATE / 600), min(FRAME_LENGTH - 1, int(SAMPLE_RATE / 50))
    if max_lag <= min_lag:
        return None, None, 0.0
    region = autocorrelation[:, min_lag:max_lag]
    lags = min_lag + np.argmax(region, axis=1)
    strengths = autocorrelation[np.arange(len(frames)), lags]
    voiced = (energy >= 1e-3) & (strengths >= 0.35)
    frequencies = SAMPLE_RATE / np.maximum(lags[voiced], 1)
    if not frequencies.size:
        return None, None, 0.0
    return float(np.mean(frequencies)), float(np.std(frequencies)), float(np.mean(voiced))


def extract_audio_features(values: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    spectral = _spectral_features(values)
    frames = spectral["frames"]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peaks = np.max(np.abs(frames), axis=1)
    rms_jump = np.abs(np.diff(np.log(rms + 1e-8), prepend=np.log(rms[0] + 1e-8)))
    derivative_samples = np.abs(np.diff(values, prepend=values[0]))
    second_samples = np.abs(np.diff(values, n=2, prepend=[values[0], values[0]]))
    derivative = np.max(_frame_matrix(derivative_samples), axis=1)
    second = np.max(_frame_matrix(second_samples), axis=1)
    f0_mean, f0_std, voiced_ratio = _rough_f0(values)
    duration = values.size / SAMPLE_RATE
    global_features: dict[str, Any] = {
        "duration_s": duration,
        "samples": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values) + 1e-12)),
        "peak": float(np.max(np.abs(values))),
        "clip_fraction": float(np.mean(np.abs(values) >= 0.999)),
        "dc_offset": float(np.mean(values)),
        "crest_factor": float(np.max(np.abs(values)) / (np.sqrt(np.mean(values * values)) + 1e-8)),
        "silence_ratio": float(np.mean(rms < 1e-4)),
        "voiced_ratio": voiced_ratio,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "energy_env_std": float(np.std(rms)),
        "energy_env_smoothness": float(np.std(np.diff(rms))) if rms.size > 1 else 0.0,
        "spectral_centroid_hz": float(np.mean(spectral["centroid"])),
        "spectral_rolloff_hz": float(np.mean(spectral["rolloff"])),
        "spectral_flatness": float(np.mean(spectral["flatness"])),
        "spectral_flux": float(np.mean(spectral["flux"])),
        "zero_crossing_rate": float(np.mean(np.mean(np.diff(np.signbit(frames), axis=1) != 0, axis=1))),
        "mean_narrowband_50_60_hz": float(np.mean(spectral["narrowband_50_60_hz"])),
        "mean_high_band_ratio": float(np.mean(spectral["band_energy"]["band_6000_8000_hz"] + spectral["band_energy"]["band_8000_12000_hz"] + spectral["band_energy"]["band_12000_16000_hz"])),
        "max_frame_peak": float(np.max(peaks)),
        "max_frame_rms_jump": float(np.max(rms_jump)),
        "max_sample_derivative": float(np.max(derivative)),
        "max_sample_second_difference": float(np.max(second)),
    }
    local = {
        "times_s": (np.arange(frames.shape[0]) * HOP_LENGTH / SAMPLE_RATE).astype(np.float32),
        "rms": rms,
        "frame_peak": peaks,
        "rms_jump": rms_jump,
        "derivative": derivative,
        "second_difference": second,
        "flux": spectral["flux"],
        "flatness": spectral["flatness"],
        "spectral_entropy": spectral["spectral_entropy"],
        "narrowband": spectral["narrowband_50_60_hz"],
        "high_band_ratio": spectral["band_energy"]["band_6000_8000_hz"] + spectral["band_energy"]["band_8000_12000_hz"] + spectral["band_energy"]["band_12000_16000_hz"],
        "spectrum": spectral["spectrum"],
        "frequencies": spectral["frequencies"],
        "log_spectrum": np.log(spectral["power"] + 1e-12),
    }
    return global_features, local


def robust_threshold(values: Sequence[float], multiplier: float = 6.0, floor: float = 1e-9) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + multiplier * max(mad, np.finfo(float).eps))


def calibrate_thresholds(control_locals: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("rms_jump", "flux", "derivative", "second_difference", "flatness", "narrowband", "high_band_ratio")
    result: dict[str, float] = {}
    for key in keys:
        values = np.concatenate([np.asarray(local[key], dtype=np.float64) for local in control_locals])
        result[key] = robust_threshold(values, 6.0)
    return result


def _event(start: float, end: float, kind: str, score: float, evidence: Mapping[str, Any], confidence: str = "moderate") -> dict[str, Any]:
    return {"start_s": max(0.0, float(start)), "end_s": max(float(start), float(end)), "duration_s": max(0.0, float(end - start)), "kind": kind, "score": float(score), "confidence": confidence, "evidence": dict(evidence)}


def _merge_events(events: Sequence[Mapping[str, Any]], gap_s: float = 0.03) -> list[dict[str, Any]]:
    ordered = sorted((dict(event) for event in events), key=lambda event: (event["kind"], event["start_s"]))
    merged: list[dict[str, Any]] = []
    for event in ordered:
        if merged and merged[-1]["kind"] == event["kind"] and event["start_s"] <= merged[-1]["end_s"] + gap_s:
            previous = merged[-1]
            previous["end_s"] = max(previous["end_s"], event["end_s"])
            previous["duration_s"] = previous["end_s"] - previous["start_s"]
            previous["score"] = max(previous["score"], event["score"])
            previous["evidence"].update(event.get("evidence", {}))
        else:
            merged.append(event)
    return merged


def detect_local_events(local: Mapping[str, Any], thresholds: Mapping[str, float], sample_count: int) -> list[dict[str, Any]]:
    times = np.asarray(local["times_s"])
    hop_s = HOP_LENGTH / SAMPLE_RATE
    events: list[dict[str, Any]] = []
    for index in range(len(times)):
        end = min(sample_count / SAMPLE_RATE, float(times[index] + FRAME_LENGTH / SAMPLE_RATE))
        start = float(times[index])
        if local["derivative"][index] > thresholds["derivative"] and local["second_difference"][index] > thresholds["second_difference"]:
            events.append(_event(start, end, "impulsive_discontinuity", float(local["derivative"][index] / (thresholds["derivative"] + 1e-12)), {"derivative": float(local["derivative"][index]), "second_difference": float(local["second_difference"][index])}, "strong"))
        if local["flux"][index] > thresholds["flux"] and local["rms_jump"][index] > thresholds["rms_jump"]:
            events.append(_event(start, end, "broadband_burst", float(local["flux"][index] / (thresholds["flux"] + 1e-12)), {"flux": float(local["flux"][index]), "rms_jump": float(local["rms_jump"][index])}, "moderate"))
        if local["narrowband"][index] > thresholds["narrowband"] and local["high_band_ratio"][index] > thresholds["high_band_ratio"]:
            events.append(_event(start, end, "suggestive_narrowband_or_highband_noise", float(local["narrowband"][index] / (thresholds["narrowband"] + 1e-12)), {"narrowband_50_60_hz": float(local["narrowband"][index]), "high_band_ratio": float(local["high_band_ratio"][index])}, "weak"))
    return _merge_events(events, gap_s=hop_s * 2)


def detect_repetitions(local: Mapping[str, Any], sample_rate: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH, min_duration_s: float = 0.10) -> list[dict[str, Any]]:
    """Find repeated spectral chunks only when envelope and spectrum agree."""
    x = np.asarray(local["log_spectrum"], dtype=np.float64)
    envelope = np.asarray(local["rms"], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 40 or envelope.size != x.shape[0]:
        return []
    x = x - x.mean(axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    raw_envelope = envelope.copy()
    envelope = (raw_envelope - np.median(raw_envelope)) / (np.std(raw_envelope) + 1e-12)
    width = max(8, int(round(min_duration_s * sample_rate / hop_length)))
    min_lag = max(width * 2, int(round(0.25 * sample_rate / hop_length)))
    max_lag = min(x.shape[0] - width - 1, int(round(2.5 * sample_rate / hop_length)))
    if max_lag <= min_lag:
        return []
    candidates: list[dict[str, Any]] = []
    for lag in range(min_lag, max_lag + 1):
        spectral_similarity = np.sum(x[:-lag] * x[lag:], axis=1)
        envelope_similarity = envelope[:-lag] * envelope[lag:]
        if spectral_similarity.size < width:
            continue
        spectral_score = np.convolve(spectral_similarity, np.ones(width) / width, mode="valid")
        envelope_score = np.convolve(envelope_similarity, np.ones(width) / width, mode="valid")
        joint_score = spectral_score * np.clip(envelope_score, 0.0, 1.0)
        peaks, _ = find_peaks(joint_score, height=0.985, distance=max(width, 4))
        high = np.flatnonzero(joint_score >= 0.985)
        if high.size and not peaks.size:
            groups = np.split(high, np.where(np.diff(high) > 1)[0] + 1)
            peaks = np.asarray([group[np.argmax(joint_score[group])] for group in groups if group.size], dtype=int)
        for peak in peaks:
            start = int(peak)
            first_envelope = envelope[start : start + width]
            second_envelope = envelope[start + lag : start + lag + width]
            envelope_variability = max(float(np.std(first_envelope)), float(np.std(second_envelope)))
            raw_variability = max(
                float(np.std(raw_envelope[start : start + width]) / (np.mean(raw_envelope[start : start + width]) + 1e-12)),
                float(np.std(raw_envelope[start + lag : start + lag + width]) / (np.mean(raw_envelope[start + lag : start + lag + width]) + 1e-12)),
            )
            trajectory_variability = max(float(np.mean(np.linalg.norm(np.diff(x[start : start + width], axis=0), axis=1))) if width > 1 else 0.0, float(np.mean(np.linalg.norm(np.diff(x[start + lag : start + lag + width], axis=0), axis=1))) if width > 1 else 0.0)
            if raw_variability < 0.02 or envelope_variability < 0.05 or trajectory_variability < 0.02:
                continue
            spectral_value = float(spectral_score[peak])
            envelope_value = float(envelope_score[peak])
            score = float(joint_score[peak])
            candidates.append({"start_s": start * hop_length / sample_rate, "end_s": (start + width) * hop_length / sample_rate, "lag_s": lag * hop_length / sample_rate, "duration_s": width * hop_length / sample_rate, "similarity": score, "envelope_agreement": envelope_value, "kind": "possible_repeated_spectral_segment", "score": score, "confidence": "moderate" if score < 0.995 else "strong", "evidence": {"lag_frames": lag, "chunk_frames": width, "spectral_similarity": spectral_value, "envelope_similarity": envelope_value}})
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["similarity"], reverse=True):
        if all(abs(candidate["start_s"] - prior["start_s"]) > candidate["duration_s"] / 2 or abs(candidate["lag_s"] - prior["lag_s"]) > 0.04 for prior in selected):
            selected.append(candidate)
    return selected[:20]


def boundary_events(local: Mapping[str, Any], tokens: Sequence[Mapping[str, Any]], sample_count: int, threshold: float) -> list[dict[str, Any]]:
    times = np.asarray(local["times_s"])
    if not len(times):
        return []
    events: list[dict[str, Any]] = []
    for token in tokens[1:-1]:
        boundary = float(token.get("end_s", 0.0))
        index = int(np.argmin(np.abs(times - boundary)))
        score = float(local["rms_jump"][index] + local["flux"][index])
        if score > threshold:
            events.append(_event(max(0.0, boundary - 0.04), min(sample_count / SAMPLE_RATE, boundary + 0.04), "mfa_token_boundary_discontinuity", score / (threshold + 1e-12), {"boundary_s": boundary, "token": token.get("token", ""), "is_silence": bool(token.get("is_silence", False)), "rms_jump": float(local["rms_jump"][index]), "flux": float(local["flux"][index])}, "moderate"))
    return events


def _resolve_record_inputs(config: AuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = json.loads(config.cohort.read_text(encoding="utf-8"))
    raw_meta = json.loads(config.raw_tts.read_text(encoding="utf-8"))
    tokens = json.loads(config.tokens.read_text(encoding="utf-8"))
    mfa = json.loads(config.mfa_linear.read_text(encoding="utf-8"))
    sync = json.loads(config.syncnet.read_text(encoding="utf-8"))
    records = list(cohort.get("records", []))
    raw_by = {str(row["sample_id"]): row for row in raw_meta.get("results", {}).values()}
    mfa_by = {str(row["sample_id"]): row for row in mfa.get("results", {}).values()}
    token_by = {str(key): row for key, row in tokens.get("records", {}).items()}
    sync_by = {(str(row["sample_id"]), str(row["arm"])): row for row in sync.get("scores", [])}
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(records) != 25:
        raise ValueError("cohort is not the frozen n25 manifest")
    if set(row.get("speaker_id") for row in records) != set(SPEAKERS) or any(sum(row.get("speaker_id") == speaker for row in records) != 5 for speaker in SPEAKERS):
        raise ValueError("expected five frozen speakers with five records each")
    if "S0770" in {str(row.get("speaker_id")) for row in records}:
        raise ValueError("heldout S0770 is present")
    if len(raw_by) != 25 or len(mfa_by) != 25 or len(token_by) != 25 or len(sync_by) != 75:
        raise ValueError("incomplete audit inputs")
    if mfa.get("arm") != "mfa_linear" or "resample_poly" not in str(next(iter(mfa_by.values())).get("audio_path", "")):
        raise ValueError("MFA input is not the resample-poly arm")
    if sync.get("status") != "complete" or sync.get("face_protocol") != "single_fixed_face1":
        raise ValueError("SyncNet input is not the complete fixed-face1 evaluation")
    output: list[dict[str, Any]] = []
    for row in records:
        sid = str(row["sample_id"])
        raw = raw_by.get(sid); mfa_row = mfa_by.get(sid); token = token_by.get(sid)
        if raw is None or mfa_row is None or token is None:
            raise ValueError(f"missing sample {sid}")
        identity = (str(row["paired_key"]), str(row["speaker_id"]), str(row["transcript"]))
        for other in (raw, mfa_row, token):
            other_identity = (str(other.get("paired_key")), str(other.get("speaker_id")), other.get("transcript"))
            if other_identity[0] != identity[0] or other_identity[1] != identity[1] or (other_identity[2] is not None and str(other_identity[2]) != identity[2]):
                raise ValueError(f"identity mismatch at {sid}")
        natural = read_audio(Path(str(row["audio_path"])), str(row["natural_source_sha256"]))
        raw_audio = read_audio(Path(str(raw["canonical_16k_audio"])), str(raw["canonical_audio_sha256"]))
        mfa_audio = read_audio(Path(str(mfa_row["audio_path"])), str(mfa_row["audio_sha256"]))
        common = {"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"], "split": row["split"], "transcript": row["transcript"]}
        mfa_meta = dict(mfa_row.get("mfa_linear_meta", {}))
        mfa_meta["output_to_conditioning_cosine"] = mfa_row.get("output_to_conditioning_cosine")
        mfa_meta["peak"] = mfa_row.get("peak")
        mfa_meta["length_adjustment"] = mfa_row.get("length_adjustment")
        for arm, audio, expected, arm_tokens, sync_arm in (("natural", natural, row["natural_source_sha256"], token["natural"]["tokens"], "natural_raw"), ("raw_tts", raw_audio, raw["canonical_audio_sha256"], token["tts"]["tokens"], "raw_tts"), ("mfa_linear_resample_poly", mfa_audio, mfa_row["audio_sha256"], token["natural"]["tokens"], "mfa_linear_resample_poly")):
            score = sync_by[(sid, sync_arm)]
            output.append({**common, "arm": arm, "audio": audio, "audio_sha256": expected, "tokens": arm_tokens, "mfa_meta": mfa_meta if arm == "mfa_linear_resample_poly" else {}, "sync": {key: score[key] for key in ("sync_c", "sync_d", "av_offset")}})
    metadata = {"cohort": cohort, "raw_tts": raw_meta, "tokens": tokens, "mfa_linear": mfa, "syncnet": sync}
    return output, metadata


def _control_locals(rows: Sequence[Mapping[str, Any]], locals_by_id: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [locals_by_id[(str(row["sample_id"]), str(row["arm"]))] for row in rows if row["arm"] in ("natural", "raw_tts")]


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(np.mean(array)) if array.size else None, "median": float(np.median(array)) if array.size else None, "mad": float(np.median(np.abs(array - np.median(array)))) if array.size else None, "positive_count": int(np.sum(array > 0)) if array.size else 0, "negative_count": int(np.sum(array < 0)) if array.size else 0}


def _paired_summaries(per_file: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = {(str(row["sample_id"]), str(row["arm"])): row for row in per_file}
    metrics = ("duration_s", "rms", "peak", "spectral_flatness", "spectral_flux", "mean_high_band_ratio", "artifact_event_count", "repetition_count", "boundary_event_count", "artifact_event_duration_s")
    result: dict[str, Any] = {}
    for target, baseline in (("mfa_linear_resample_poly", "natural"), ("mfa_linear_resample_poly", "raw_tts"), ("raw_tts", "natural")):
        comparison: dict[str, Any] = {}
        for metric in metrics:
            values = [float(by[(sid, target)].get(metric, 0.0) or 0.0) - float(by[(sid, baseline)].get(metric, 0.0) or 0.0) for sid in sorted({str(row["sample_id"]) for row in per_file}, key=int)]
            comparison[metric] = _metric_summary(values)
        result[f"{target}_vs_{baseline}"] = comparison
    return result


def _speaker_summary(per_file: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speaker in SPEAKERS:
        rows = [row for row in per_file if row["speaker_id"] == speaker]
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            arms[arm] = {metric: _metric_summary([float(row.get(metric, 0.0) or 0.0) for row in arm_rows]) for metric in ("rms", "spectral_flux", "mean_high_band_ratio", "artifact_event_count", "repetition_count", "boundary_event_count", "artifact_event_duration_s")}
        result[speaker] = {"n": len(rows) // 3, "arms": arms}
    return result


def _correlations(per_file: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in per_file if row["arm"] == "mfa_linear_resample_poly"]
    outputs: dict[str, Any] = {}
    for x_key in ("coverage", "fallback_frames", "unmatched_natural_token_count", "unmatched_tts_token_count", "output_to_conditioning_cosine", "peak"):
        x = np.asarray([float(row.get(x_key, 0.0) or 0.0) for row in rows])
        for y_key in ("artifact_event_count", "repetition_count", "boundary_event_count", "sync_c", "sync_d"):
            y = np.asarray([float(row.get(y_key, 0.0) or 0.0) for row in rows])
            if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
                value = None
            else:
                value = float(np.corrcoef(x, y)[0, 1])
            outputs[f"{x_key}_vs_{y_key}"] = {"n": int(x.size), "pearson_r": value, "interpretation": "exploratory association; not causal"}
    return outputs


def _plot_sample(sample_id: str, rows: Sequence[Mapping[str, Any]], locals_by_id: Mapping[tuple[str, str], Mapping[str, Any]], tokens_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    for index, arm in enumerate(ARMS):
        row = next(row for row in rows if row["arm"] == arm)
        local = locals_by_id[(sample_id, arm)]
        values = row["audio_values"]
        time = np.arange(values.size) / SAMPLE_RATE
        axes[index, 0].plot(time, values, linewidth=0.35, color="#334e68")
        axes[index, 0].set_ylabel(arm)
        axes[index, 0].set_ylim(-1.05, 1.05)
        axes[index, 1].pcolormesh(local["times_s"], local["frequencies"], 20 * np.log10(local["spectrum"].T + 1e-6), shading="auto", cmap="magma", vmin=-80, vmax=0)
        for token in tokens_by_arm[arm]:
            boundary = float(token.get("end_s", 0.0))
            axes[index, 0].axvline(boundary, color="#e76f51", alpha=0.18, linewidth=0.4)
            axes[index, 1].axvline(boundary, color="#f4a261", alpha=0.18, linewidth=0.4)
        for event in row.get("events", []):
            if event["kind"] != "padding_tail":
                axes[index, 0].axvspan(event["start_s"], event["end_s"], color="#e63946", alpha=0.25)
                axes[index, 1].axvspan(event["start_s"], event["end_s"], color="#e63946", alpha=0.25)
        axes[index, 1].set_ylim(0, 8000)
        axes[index, 1].set_ylabel("Hz")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"AISHELL-1 {sample_id} waveform and spectrogram")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"sample_{sample_id}.png", dpi=120)
    plt.close(fig)


def _report(per_file: Sequence[Mapping[str, Any]], paired: Mapping[str, Any], speakers: Mapping[str, Any], correlations: Mapping[str, Any]) -> str:
    lines = ["# AISHELL-1 n25 audio artifact audit", "", "## Scope", "", "This is an audio-only exploratory audit of 25 paired utterances and three arms: natural, raw TTS, and resample-poly MFA-linear. The SyncNet join uses a single fixed face1 and is association-only.", "", "## Key aggregate observations", ""]
    for comparison, values in paired.items():
        lines.append(f"### {comparison}")
        for metric in ("artifact_event_count", "repetition_count", "boundary_event_count", "artifact_event_duration_s", "spectral_flux", "mean_high_band_ratio"):
            summary = values[metric]
            lines.append(f"- `{metric}` delta: mean={summary['mean']}, median={summary['median']}, positive={summary['positive_count']}/{summary['n']}")
        lines.append("")
    lines.extend(["## Speaker comparison", "", "| Speaker | MFA artifact events | MFA repetitions | MFA boundary events | MFA high-band ratio |", "|---|---:|---:|---:|---:|"])
    for speaker in SPEAKERS:
        arm = speakers[speaker]["arms"]["mfa_linear_resample_poly"]
        lines.append(f"| {speaker} | {arm['artifact_event_count']['mean']:.3f} | {arm['repetition_count']['mean']:.3f} | {arm['boundary_event_count']['mean']:.3f} | {arm['mean_high_band_ratio']['mean']:.5f} |")
    lines.extend(["", "## Highest-priority manual review", ""])
    ranked = sorted((row for row in per_file if row["arm"] == "mfa_linear_resample_poly"), key=lambda row: (float(row.get("artifact_event_count", 0)) + 2 * float(row.get("repetition_count", 0)) + float(row.get("boundary_event_count", 0)), str(row["sample_id"])), reverse=True)
    for row in ranked[:10]:
        lines.append(f"- `{row['speaker_id']} sample {row['sample_id']}`: artifact_events={row['artifact_event_count']}, repetitions={row['repetition_count']}, boundary_events={row['boundary_event_count']}, coverage={row.get('coverage')}, fallback_frames={row.get('fallback_frames')}")
    lines.extend(["", "## Interpretation limits", "", "The detectors provide review candidates, not diagnoses. Narrow-band energy near 50/60 Hz can overlap speech harmonics at 16 kHz; repeated spectral chunks can be caused by vocoder regularity; MFA metadata has aggregate fallback counts but no fallback frame indices. Five speakers with five utterances each and a fixed S0765 face cannot establish speaker-independent acoustic causality.", "", "## Correlations", "", "All correlations are exploratory Pearson associations and are not causal claims."])
    for key, value in list(correlations.items())[:12]:
        lines.append(f"- `{key}`: r={value['pearson_r']}")
    return "\n".join(lines) + "\n"


def run_audit(config: AuditConfig) -> dict[str, Any]:
    config = AuditConfig(**{**asdict(config), "cohort": Path(config.cohort).resolve(), "raw_tts": Path(config.raw_tts).resolve(), "tokens": Path(config.tokens).resolve(), "mfa_linear": Path(config.mfa_linear).resolve(), "syncnet": Path(config.syncnet).resolve(), "outdir": Path(config.outdir).resolve()})
    if config.outdir.exists() and any(config.outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {config.outdir}")
    rows, metadata = _resolve_record_inputs(config)
    config.outdir.mkdir(parents=True, exist_ok=True)
    locals_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        global_features, local = extract_audio_features(row["audio"].values)
        locals_by_id[(str(row["sample_id"]), str(row["arm"]))] = local
        feature_rows.append({**{key: value for key, value in row.items() if key not in ("audio", "tokens", "mfa_meta")}, **global_features, "audio": str(row["audio"].path), "audio_values": row["audio"].values, "audio_sha256": row["audio_sha256"], "tokens": row["tokens"], "mfa_meta": row["mfa_meta"]})
    thresholds = calibrate_thresholds(_control_locals(rows, locals_by_id))
    tokens_by_id = {(str(row["sample_id"]), str(row["arm"])): row["tokens"] for row in rows}
    for row in feature_rows:
        key = (str(row["sample_id"]), str(row["arm"]))
        local = locals_by_id[key]
        events = detect_local_events(local, thresholds, int(row["samples"]))
        events.extend(detect_repetitions(local))
        boundary_threshold = thresholds["rms_jump"] + thresholds["flux"]
        events.extend(boundary_events(local, row["tokens"], int(row["samples"]), boundary_threshold))
        adjustment = row.get("mfa_meta", {}).get("length_adjustment", {})
        if row["arm"] == "mfa_linear_resample_poly" and adjustment.get("action") == "right_zero_pad":
            raw_samples = int(adjustment.get("raw_samples", row["samples"]))
            if raw_samples < int(row["samples"]):
                events.append(_event(raw_samples / SAMPLE_RATE, int(row["samples"]) / SAMPLE_RATE, "padding_tail", 1.0, {"raw_samples": raw_samples, "target_samples": int(row["samples"]), "not_generated_audio": True}, "strong"))
        row["events"] = events
        row["artifact_event_count"] = sum(event["kind"] in {"impulsive_discontinuity", "broadband_burst", "suggestive_narrowband_or_highband_noise"} for event in events)
        row["repetition_count"] = sum(event["kind"] == "possible_repeated_spectral_segment" for event in events)
        row["boundary_event_count"] = sum(event["kind"] == "mfa_token_boundary_discontinuity" for event in events)
        row["artifact_event_duration_s"] = sum(float(event["duration_s"]) for event in events if event["kind"] in {"impulsive_discontinuity", "broadband_burst", "suggestive_narrowband_or_highband_noise"})
        mfa_meta = row.get("mfa_meta", {})
        match_stats = mfa_meta.get("match_stats", {})
        row["coverage"] = mfa_meta.get("coverage")
        row["fallback_frames"] = mfa_meta.get("fallback_frames")
        row["matched_frames"] = mfa_meta.get("matched_frames")
        row["unmatched_natural_token_count"] = match_stats.get("unmatched_natural_token_count", 0)
        row["unmatched_tts_token_count"] = match_stats.get("unmatched_tts_token_count", 0)
        row["output_to_conditioning_cosine"] = mfa_meta.get("output_to_conditioning_cosine")
        row["sync_c"] = row["sync"]["sync_c"]
        row["sync_d"] = row["sync"]["sync_d"]
        row["av_offset"] = row["sync"]["av_offset"]
        row.pop("sync", None)
        row.pop("tokens", None)
    per_file = [{key: value for key, value in row.items() if key not in ("audio_values", "tokens")} for row in feature_rows]
    paired = _paired_summaries(per_file)
    speakers = _speaker_summary(per_file)
    correlations = _correlations(per_file)
    input_manifest = {"schema_version": 1, "audit": "aishell1_n25_audio_artifacts", "arms": list(ARMS), "sample_count": 25, "speaker_ids": list(SPEAKERS), "thresholds": thresholds, "inputs": {key: {"path": str(getattr(config, key)), "sha256": sha256_file(getattr(config, key))} for key in ("cohort", "raw_tts", "tokens", "mfa_linear", "syncnet")}, "mfa_linear_contract": "scipy.signal.resample_poly(up=2, down=3, float32, in_memory_equivalent)", "fallback_frame_indices_available": False}
    write_json(config.outdir / "input_manifest.json", input_manifest)
    with (config.outdir / "per_file.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_file:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    with (config.outdir / "per_event.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_file:
            for event in row["events"]:
                handle.write(json.dumps(json_safe({"sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "arm": row["arm"], **event}), ensure_ascii=False, allow_nan=False) + "\n")
    write_json(config.outdir / "paired_summary.json", paired)
    write_json(config.outdir / "speaker_summary.json", speakers)
    write_json(config.outdir / "correlations.json", correlations)
    suspects = sorted(per_file, key=lambda row: (float(row.get("artifact_event_count", 0)) + 2 * float(row.get("repetition_count", 0)) + float(row.get("boundary_event_count", 0)), str(row["sample_id"])), reverse=True)
    with (config.outdir / "top_suspects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "speaker_id", "arm", "artifact_event_count", "repetition_count", "boundary_event_count", "artifact_event_duration_s", "coverage", "fallback_frames"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in suspects[:30])
    if config.make_plots:
        plot_dir = config.outdir / "plots"
        for sample_id in sorted({str(row["sample_id"]) for row in rows}, key=int):
            sample_rows = [row for row in feature_rows if str(row["sample_id"]) == sample_id]
            _plot_sample(sample_id, sample_rows, locals_by_id, {row["arm"]: row["tokens"] for row in sample_rows}, plot_dir)
    report = _report(per_file, paired, speakers, correlations)
    (config.outdir / "report.md").write_text(report, encoding="utf-8")
    return {"input_manifest": input_manifest, "per_file": per_file, "paired": paired, "speakers": speakers, "correlations": correlations, "output_dir": str(config.outdir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-tts", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--mfa-linear", type=Path, required=True)
    parser.add_argument("--syncnet", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run_audit(AuditConfig(args.cohort, args.raw_tts, args.tokens, args.mfa_linear, args.syncnet, args.outdir, not args.no_plots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
