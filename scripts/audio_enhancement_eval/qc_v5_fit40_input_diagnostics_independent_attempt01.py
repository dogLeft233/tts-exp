#!/usr/bin/env python3
"""Independently QC the sealed fit40 input-only power/flux diagnostics.

This reads only the 40 already-materialized canonical PCM16 WAVs, the fit40
transfer manifest, and the pinned V4 diagnostic JSON/NPZ. It does not open the
source tar archives, run a transform/model, render video, score, select, or read
dev/confirmation audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02"
SOURCE_DIR = PREP / "fit40_transfer_bundle_attempt01"
MANIFEST_PATH = SOURCE_DIR / "fit40_source_transfer_manifest.json"
DIAGNOSTIC_DIR = PREP / "fit40_input_diagnostics_attempt02"
DIAGNOSTIC_JSON_PATH = DIAGNOSTIC_DIR / "fit40_input_power_flux_diagnostic.json"
DIAGNOSTIC_NPZ_PATH = DIAGNOSTIC_DIR / "fit40_input_power_flux_arrays.npz"
MATERIALIZER_PATH = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/prepare_v5_fit40_transfer_manifest_attempt01.py"
DIAGNOSTIC_SOURCE_PATH = ROOT / "scripts/audio_enhancement_eval/run_v5_fit40_input_power_flux_diagnostic_attempt01.py"
AUDIO_IO_PATH = ROOT / "scripts/audio_enhancement/audio_io.py"
BASELINES_PATH = ROOT / "scripts/audio_enhancement/baselines.py"
V4_HELPER_PATH = ROOT / "scripts/audio_enhancement_eval/decompose_v4_delta_common_mode_attempt04.py"
R090_CONFIG_PATH = SOURCE_DIR / "localcompress_r090_production_config.json"
R090_SOURCE_PATH = ROOT / "scripts/audio_enhancement/localcompress_r090_control_attempt01.py"

EXPECTED_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_DIAGNOSTIC_JSON_SHA256 = "aacf642acfb693fe3cfeda299418693e2e894230b2adf9aa6334ee44fc7cd8fd"
EXPECTED_DIAGNOSTIC_NPZ_SHA256 = "7b545d26c8c5a43499175361030b2902bd889994e0be78e38737d3689d2f7977"
EXPECTED_MATERIALIZER_SHA256 = "753c5725cf99d2c098ba10b8251b7c5d6100a4da9153d26920ab5e30ec4d63f3"
EXPECTED_DIAGNOSTIC_SOURCE_SHA256 = "1e78f88ef717b26a53e7b0d5e9f398ebece0d0edef4ced3425f3557d7a8c3e54"
EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_V4_HELPER_SHA256 = "9b48d0fefd3c34221f6929da08f7dede9b761f579d7fc56d9fc0c1c1e8d68577"
EXPECTED_R090_SOURCE_SHA256 = "4c24c5d15d280186da2be4c07931a72625cf6d3edc24bb3de4211206aa81fa09"
EXPECTED_R090_CONFIG_SHA256 = "2c9f63d74bc204ff46c766b5cad18dc9655cf79a228f93b76cc0a84533913e62"

SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 128
LOCAL_WINDOW_SAMPLES = 512
PHYSICAL_FRAME_SAMPLES = 640
EPS = 1e-12


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {name: None for name in ("mean", "std", "p05", "p50", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def require_pin(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or symlinked pinned input {label}: {path}")
    digest, size = sha_file(path)
    if digest != expected_sha:
        raise RuntimeError(f"pinned input SHA mismatch for {label}: {digest}")
    return {"path": str(path.resolve()), "bytes": size, "raw_sha256": digest}


def read_source_pcm16(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    wav_path = Path(row["transfer_local_path"]).resolve(strict=True)
    if wav_path.parent != SOURCE_DIR.resolve() / "source_wav":
        raise RuntimeError(f"source WAV escaped the frozen transfer directory: {wav_path}")
    file_sha, file_bytes = sha_file(wav_path)
    if (file_sha, file_bytes) != (row["transfer_file_sha256"], int(row["transfer_file_bytes"])):
        raise RuntimeError(f"materialized WAV file seal mismatch: {row['row_id']}")
    if file_sha != row["source_member_sha256"] or file_bytes != int(row["source_member_bytes"]):
        raise RuntimeError(f"WAV is not the selected raw archive member: {row['row_id']}")

    raw = wav_path.read_bytes()
    with wave.open(str(wav_path), "rb") as reader:
        params = reader.getparams()
        if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (
            1,
            2,
            SAMPLE_RATE,
            "NONE",
        ):
            raise RuntimeError(f"WAV is not mono 16 kHz PCM16: {row['row_id']} {params}")
        if params.nframes != int(row["sample_count"]):
            raise RuntimeError(f"WAV frame count differs from manifest: {row['row_id']}")
        payload = reader.readframes(params.nframes)
    if len(payload) != params.nframes * 2:
        raise RuntimeError(f"truncated PCM16 payload: {row['row_id']}")
    payload_sha = sha_bytes(payload)
    if payload_sha != row["source_pcm_payload_sha256"]:
        raise RuntimeError(f"PCM16 payload SHA mismatch: {row['row_id']}")

    # Independent canonical decode for this frozen same-rate, mono PCM16 cohort.
    pcm_i16 = np.frombuffer(payload, dtype="<i2")
    decoded_f64 = pcm_i16.astype(np.float64) / 32768.0
    samples = np.ascontiguousarray(decoded_f64, dtype=np.float32)
    pcm_f32_sha = sha_bytes(samples.astype("<f4", copy=False).tobytes())
    if pcm_f32_sha != row["canonical_float32_pcm_sha256"]:
        raise RuntimeError(f"canonical float32 PCM digest mismatch: {row['row_id']}")

    # Cross-check the project's pinned decoder without using it for the numeric recomputation.
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    project_samples, project_info = read_canonical_wav(wav_path, target_rate=SAMPLE_RATE)
    if project_info.pcm_float32_sha256 != pcm_f32_sha or not np.array_equal(project_samples, samples):
        raise RuntimeError(f"project decoder differs from independent PCM16 decode: {row['row_id']}")
    return samples, {
        "path": str(wav_path),
        "file_bytes": file_bytes,
        "file_sha256": file_sha,
        "pcm16_payload_sha256": payload_sha,
        "canonical_float32_pcm_sha256": pcm_f32_sha,
        "sample_count": int(samples.size),
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "source_encoding": "PCM_16",
        "archive_opened": False,
    }


def recompute_v4_descriptors(samples: np.ndarray) -> dict[str, np.ndarray | int | bool]:
    """Recompute frozen V4 512/128 features and signed flux from source PCM."""
    n = int(samples.size)
    centers = np.arange(1 + n // HOP, dtype=np.int64) * HOP
    # Frozen V4 uses zero-padding, 256 samples on both sides, centered at these indices.
    padded = np.pad(np.asarray(samples, dtype=np.float32), (N_FFT // 2, N_FFT // 2), mode="constant")
    windows = np.stack([padded[int(c) : int(c) + N_FFT] for c in centers], axis=0)
    k = np.arange(N_FFT, dtype=np.float64)
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * k / N_FFT)).astype(np.float32)
    windowed = windows * hann[None, :]
    frame_rms = np.sqrt(np.mean(windowed.astype(np.float64) ** 2, axis=1))
    envelope_dbfs = 20.0 * np.log10(np.maximum(frame_rms, EPS))
    magnitude = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1))
    log_magnitude_db = 20.0 * np.log10(np.maximum(magnitude, EPS))
    delta = log_magnitude_db[1:, 1:] - log_magnitude_db[:-1, 1:]
    positive = np.full(centers.size, np.nan, dtype=np.float64)
    negative = np.full(centers.size, np.nan, dtype=np.float64)
    signed_mean = np.full(centers.size, np.nan, dtype=np.float64)
    if centers.size > 1:
        positive[1:] = np.maximum(delta, 0.0).mean(axis=1, dtype=np.float64)
        negative[1:] = np.maximum(-delta, 0.0).mean(axis=1, dtype=np.float64)
        signed_mean[1:] = delta.mean(axis=1, dtype=np.float64)
    if centers[-1] > n:
        raise RuntimeError("V4 analysis clock extends beyond the single permitted N endpoint")
    return {
        "centers": centers,
        "envelope_dbfs": envelope_dbfs,
        "hann_power": np.power(10.0, envelope_dbfs / 10.0),
        "positive_flux_db": positive,
        "negative_flux_db": negative,
        "signed_mean_db_change": signed_mean,
        "positive_frequency_bins": N_FFT // 2,
        "endpoint_center_equals_N": bool(int(centers[-1]) == n),
    }


def recompute_local_envelope(samples: np.ndarray, last_center: int) -> tuple[np.ndarray, int]:
    """Replicate the frozen compressor RMS envelope and its explicit N endpoint."""
    n = int(samples.size)
    if last_center < 0 or last_center > n:
        raise RuntimeError(f"local envelope center outside [0,N]: {last_center} / {n}")
    extension = max(0, last_center - (n - 1))
    x = np.asarray(samples, dtype=np.float32)
    if extension:
        # Only the N sample diagnostic point is extended; original input length is unchanged.
        x = np.pad(x, (0, extension), mode="edge")
    window = LOCAL_WINDOW_SAMPLES
    left = window // 2 - 1
    right = window - 1 - left
    power = x.astype(np.float64) ** 2
    padded = np.pad(power, (left, right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    local_power = (cumulative[window:] - cumulative[:-window]) / window
    if local_power.size != x.size:
        raise RuntimeError("local envelope changed waveform sample count")
    return np.sqrt(np.maximum(local_power, 1e-10)), extension


def compare_array(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual)
    e = np.asarray(expected)
    if a.shape != e.shape:
        raise RuntimeError(f"array shape mismatch {name}: {a.shape} != {e.shape}")
    if a.dtype.kind in "iu" or e.dtype.kind in "iu":
        if not np.array_equal(a, e):
            raise RuntimeError(f"integer array mismatch {name}")
        return {"shape": list(a.shape), "exact": True, "max_abs_error": 0.0}
    af = a.astype(np.float64)
    ef = e.astype(np.float64)
    if not np.array_equal(np.isnan(af), np.isnan(ef)):
        raise RuntimeError(f"NaN-position mismatch {name}")
    finite = np.isfinite(af) & np.isfinite(ef)
    max_error = float(np.max(np.abs(af[finite] - ef[finite]))) if finite.any() else 0.0
    if not np.allclose(af, ef, rtol=2e-6, atol=2e-6, equal_nan=True):
        raise RuntimeError(f"numeric array mismatch {name}; max_abs_error={max_error}")
    return {"shape": list(a.shape), "exact": bool(np.array_equal(a, e, equal_nan=True)), "max_abs_error": max_error}


def compare_quantiles(label: str, actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if expected_value is None:
            if actual_value is not None:
                raise RuntimeError(f"{label}.{key}: expected null, got {actual_value}")
            continue
        if actual_value is None or not math.isclose(
            float(actual_value), float(expected_value), rel_tol=3e-6, abs_tol=3e-6
        ):
            raise RuntimeError(f"{label}.{key}: {actual_value} != {expected_value}")
        errors[key] = abs(float(actual_value) - float(expected_value))
    return errors


def verify_rows(
    manifest: dict[str, Any], diagnostic: dict[str, Any], stored: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = manifest.get("rows")
    diagnostic_rows = diagnostic.get("rows")
    if not isinstance(rows, list) or len(rows) != 40 or not isinstance(diagnostic_rows, list) or len(diagnostic_rows) != 40:
        raise RuntimeError("both source manifest and pinned diagnostic must have exactly 40 rows")
    if len({r["row_id"] for r in rows}) != 40 or len({r["sample_id"] for r in rows}) != 40:
        raise RuntimeError("fit40 source rows are not unique")
    if [r["row_id"] for r in rows] != [r["row_id"] for r in diagnostic_rows]:
        raise RuntimeError("diagnostic row order/identity differs from source manifest")
    speakers = {f"S{i:04d}" for i in range(52, 72)}
    if set(manifest.get("fit_speakers", [])) != speakers:
        raise RuntimeError("fit speaker roster differs from S0052-S0071")
    if manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise RuntimeError("source manifest status is not the prepared input-only status")
    for scope in (manifest.get("scope", {}), diagnostic.get("scope", {})):
        if scope.get("dev_audio_read") is not False or scope.get("confirm_audio_read") is not False:
            raise RuntimeError("a pinned source artifact reports dev or confirmation audio access")
    for key in ("candidate_effect_read", "enhancement_transform_executed", "model_forward", "render", "video_render", "score", "selection"):
        if key in diagnostic.get("scope", {}) and diagnostic["scope"][key] is not False:
            raise RuntimeError(f"diagnostic scope unexpectedly reports {key}=true")

    npz_expected_keys: set[str] = set()
    row_reports: list[dict[str, Any]] = []
    output_arrays: dict[str, np.ndarray] = {}
    speaker_rows: dict[str, int] = {}
    for row, diag_row in zip(rows, diagnostic_rows, strict=True):
        row_id = str(row["row_id"])
        sid = str(row["speaker_id"])
        speaker_rows[sid] = speaker_rows.get(sid, 0) + 1
        if sid not in speakers or int(row["canonical_j"]) not in (0, 15):
            raise RuntimeError(f"unexpected fit endpoint row: {row_id}")
        if row["sample_rate_hz"] != SAMPLE_RATE or row["channels"] != 1 or row["source_encoding"] != "PCM_16":
            raise RuntimeError(f"source manifest contract mismatch: {row_id}")
        samples, lineage = read_source_pcm16(row)
        n = int(samples.size)
        if n != int(row["sample_count"]) or not np.isfinite(samples).all():
            raise RuntimeError(f"invalid canonical source samples: {row_id}")
        if lineage["path"] != str(Path(diag_row["verified_source"]["path"]).resolve(strict=True)):
            raise RuntimeError(f"pinned diagnostic source path mismatch: {row_id}")
        if diag_row["verified_source"]["file_sha256"] != lineage["file_sha256"] or diag_row["verified_source"]["pcm_float32_sha256"] != lineage["canonical_float32_pcm_sha256"]:
            raise RuntimeError(f"pinned diagnostic source seal mismatch: {row_id}")

        x64 = samples.astype(np.float64)
        global_rms = float(np.sqrt(np.mean(x64 * x64, dtype=np.float64)))
        global_mean_square = global_rms * global_rms
        peak = float(np.max(np.abs(x64)))
        physical_count = n // PHYSICAL_FRAME_SAMPLES
        physical_used = physical_count * PHYSICAL_FRAME_SAMPLES
        physical_frames = samples[:physical_used].astype(np.float64).reshape(physical_count, PHYSICAL_FRAME_SAMPLES)
        physical_power = np.mean(physical_frames * physical_frames, axis=1, dtype=np.float64)
        physical_dbfs = 10.0 * np.log10(np.maximum(physical_power, 1e-20))
        physical_start = np.arange(physical_count, dtype=np.int64) * PHYSICAL_FRAME_SAMPLES
        physical_end = physical_start + PHYSICAL_FRAME_SAMPLES
        physical_center = physical_start + PHYSICAL_FRAME_SAMPLES // 2

        v4 = recompute_v4_descriptors(samples)
        centers = np.asarray(v4["centers"], dtype=np.int64)
        envelope, endpoint_extension = recompute_local_envelope(samples, int(centers[-1]))
        envelope_at_centers = envelope[centers]
        local_power_at_centers = envelope_at_centers.astype(np.float64) ** 2
        flux_pos = np.asarray(v4["positive_flux_db"], dtype=np.float64)
        flux_neg = np.asarray(v4["negative_flux_db"], dtype=np.float64)
        flux_net = np.asarray(v4["signed_mean_db_change"], dtype=np.float64)
        valid_flux = np.isfinite(flux_pos)
        if int(valid_flux.sum()) != max(0, len(centers) - 1):
            raise RuntimeError(f"positive-flux denominator mismatch: {row_id}")
        if not np.allclose(flux_pos[valid_flux] - flux_neg[valid_flux], flux_net[valid_flux], rtol=2e-12, atol=2e-12):
            raise RuntimeError(f"signed-flux decomposition mismatch: {row_id}")

        arrays_for_row = {
            "frame_center_sample": centers,
            "local_envelope_rms": envelope_at_centers.astype(np.float32),
            "local_power_amplitude_squared": local_power_at_centers.astype(np.float32),
            "hann_envelope_dbfs": np.asarray(v4["envelope_dbfs"], dtype=np.float32),
            "hann_windowed_mean_square_unscaled": np.asarray(v4["hann_power"], dtype=np.float32),
            "positive_spectral_flux_db": flux_pos.astype(np.float32),
            "physical_25fps_frame_power": physical_power.astype(np.float32),
            "physical_25fps_frame_power_dbfs": physical_dbfs.astype(np.float32),
        }
        array_checks: dict[str, Any] = {}
        for suffix, recomputed in arrays_for_row.items():
            key = f"{row_id}_{suffix}"
            npz_expected_keys.add(key)
            if key not in stored.files:
                raise RuntimeError(f"pinned diagnostic NPZ missing array {key}")
            array_checks[suffix] = compare_array(key, stored[key], recomputed)
            output_arrays[f"{row_id}_{suffix}"] = recomputed
        output_arrays[f"{row_id}_negative_spectral_flux_db"] = flux_neg.astype(np.float32)
        output_arrays[f"{row_id}_signed_mean_spectral_change_db"] = flux_net.astype(np.float32)
        output_arrays[f"{row_id}_physical_25fps_frame_start_sample"] = physical_start
        output_arrays[f"{row_id}_physical_25fps_frame_end_sample_exclusive"] = physical_end
        output_arrays[f"{row_id}_physical_25fps_frame_center_sample"] = physical_center

        source_summary = diag_row["global_physical_summary"]
        if not math.isclose(source_summary["rms_amplitude"], global_rms, rel_tol=2e-8, abs_tol=2e-10):
            raise RuntimeError(f"global RMS summary mismatch: {row_id}")
        if not math.isclose(source_summary["mean_square_amplitude"], global_mean_square, rel_tol=2e-8, abs_tol=2e-12):
            raise RuntimeError(f"global mean-square summary mismatch: {row_id}")
        if not math.isclose(source_summary["peak_abs_amplitude"], peak, rel_tol=2e-8, abs_tol=2e-10):
            raise RuntimeError(f"peak summary mismatch: {row_id}")

        local_section = diag_row["r090_compressor_input_envelope"]
        hann_section = diag_row["frozen_profile_clock_descriptors"]
        physical_section = diag_row["physical_25fps_power"]
        summary_checks = {
            "local_rms_quantiles": compare_quantiles("local_rms", local_section["rms_amplitude_quantiles"], quantiles(envelope[:n])),
            "local_power_quantiles": compare_quantiles("local_power", local_section["power_amplitude_squared_quantiles"], quantiles(envelope[:n].astype(np.float64) ** 2)),
            "local_dbfs_quantiles": compare_quantiles("local_dbfs", local_section["rms_dbfs_quantiles"], quantiles(20.0 * np.log10(np.maximum(envelope[:n].astype(np.float64), 1e-12)))),
            "local_center_dbfs_quantiles": compare_quantiles("local_center_dbfs", local_section["frame_centered_8ms_sampled_rms_dbfs_quantiles"], quantiles(20.0 * np.log10(np.maximum(envelope_at_centers.astype(np.float64), 1e-12)))),
            "hann_envelope_dbfs_quantiles": compare_quantiles("hann_env", hann_section["hann_windowed_envelope_dbfs_quantiles"], quantiles(np.asarray(v4["envelope_dbfs"]))),
            "hann_power_quantiles": compare_quantiles("hann_power", hann_section["hann_windowed_mean_square_unscaled_quantiles"], quantiles(np.asarray(v4["hann_power"]))),
            "positive_flux_quantiles": compare_quantiles("positive_flux", hann_section["positive_spectral_flux_db_quantiles"], quantiles(flux_pos)),
            "physical_power_quantiles": compare_quantiles("physical_power", physical_section["mean_square_quantiles"], quantiles(physical_power)),
            "physical_dbfs_quantiles": compare_quantiles("physical_dbfs", physical_section["dbfs_quantiles"], quantiles(physical_dbfs)),
        }
        if int(hann_section["frame_count"]) != len(centers) or hann_section["frame_centers_first_last_samples"] != [int(centers[0]), int(centers[-1])]:
            raise RuntimeError(f"V4 sample-clock mapping mismatch: {row_id}")
        if int(physical_section["frame_count"]) != physical_count or int(physical_section["trailing_samples_excluded"]) != n - physical_used:
            raise RuntimeError(f"25 fps frame mapping mismatch: {row_id}")

        row_reports.append({
            "row_id": row_id,
            "speaker_id": sid,
            "canonical_j": int(row["canonical_j"]),
            "sample_id": row["sample_id"],
            "source": lineage,
            "global": {
                "sample_count": n,
                "mean_square": global_mean_square,
                "rms": global_rms,
                "peak_abs": peak,
            },
            "local_32ms_envelope": {
                "window_samples": LOCAL_WINDOW_SAMPLES,
                "center_left_samples": LOCAL_WINDOW_SAMPLES // 2 - 1,
                "center_right_samples": LOCAL_WINDOW_SAMPLES - (LOCAL_WINDOW_SAMPLES // 2 - 1) - 1,
                "edge_padding": "replicate endpoints for local RMS window",
                "sample_count_unmodified": n,
                "endpoint_center_equals_N": bool(int(centers[-1]) == n),
                "extra_repeated_source_samples_for_endpoint_measurement": endpoint_extension,
                "rms_quantiles": quantiles(envelope[:n]),
            },
            "stft_512_hop128": {
                "window": "periodic Hann; 256 zero samples left/right at source boundaries",
                "center_sample_first_last": [int(centers[0]), int(centers[-1])],
                "center_time_seconds_first_last": [[int(centers[0]), SAMPLE_RATE], [int(centers[-1]), SAMPLE_RATE]],
                "frame_count": int(len(centers)),
                "positive_flux_valid_frames": int(valid_flux.sum()),
                "positive_flux_missing_frames": int((~valid_flux).sum()),
                "negative_flux_valid_frames": int(np.isfinite(flux_neg).sum()),
                "positive_flux_db_quantiles": quantiles(flux_pos),
                "negative_flux_db_quantiles": quantiles(flux_neg),
                "signed_mean_db_change_quantiles": quantiles(flux_net),
                "flux_definition": "over 256 non-DC rFFT bins: positive=mean(max(delta_logmag_db,0)); negative=mean(max(-delta_logmag_db,0)); signed=mean(delta_logmag_db); frame 0 unavailable/NaN",
            },
            "physical_40ms_25fps": {
                "frame_samples": PHYSICAL_FRAME_SAMPLES,
                "fps": 25,
                "frame_count": int(physical_count),
                "trailing_samples_excluded": int(n - physical_used),
                "first_frame_interval_samples": [0, min(n, PHYSICAL_FRAME_SAMPLES)],
                "last_frame_interval_samples": [int(physical_start[-1]), int(physical_end[-1])] if physical_count else None,
                "first_last_frame_center_samples": [int(physical_center[0]), int(physical_center[-1])] if physical_count else None,
                "first_last_frame_center_time_numerators_over_32000": [[int(physical_center[0]), 2 * SAMPLE_RATE], [int(physical_center[-1]), 2 * SAMPLE_RATE]] if physical_count else None,
                "power_mean_square_quantiles": quantiles(physical_power),
            },
            "producer_array_comparisons": array_checks,
            "producer_summary_comparisons": summary_checks,
        })

    if set(stored.files) != npz_expected_keys:
        extra = sorted(set(stored.files) - npz_expected_keys)
        missing = sorted(npz_expected_keys - set(stored.files))
        raise RuntimeError(f"pinned NPZ key set differs; extra={extra[:4]} missing={missing[:4]}")
    if speaker_rows != {sid: 2 for sid in sorted(speakers)}:
        raise RuntimeError(f"fit40 endpoint coverage is not exactly two endpoints per speaker: {speaker_rows}")
    return row_reports, output_arrays


def run(output_dir: Path) -> dict[str, Any]:
    pins = {
        "materializer": require_pin(MATERIALIZER_PATH, EXPECTED_MATERIALIZER_SHA256, "fit40 materializer"),
        "diagnostic_source": require_pin(DIAGNOSTIC_SOURCE_PATH, EXPECTED_DIAGNOSTIC_SOURCE_SHA256, "diagnostic producer"),
        "audio_io": require_pin(AUDIO_IO_PATH, EXPECTED_AUDIO_IO_SHA256, "canonical audio reader"),
        "baselines": require_pin(BASELINES_PATH, EXPECTED_BASELINES_SHA256, "local envelope baseline"),
        "v4_helper": require_pin(V4_HELPER_PATH, EXPECTED_V4_HELPER_SHA256, "V4 diagnostic helper"),
        "r090_source": require_pin(R090_SOURCE_PATH, EXPECTED_R090_SOURCE_SHA256, "r090 prepared control"),
        "r090_config": require_pin(R090_CONFIG_PATH, EXPECTED_R090_CONFIG_SHA256, "r090 prepared config"),
        "source_manifest": require_pin(MANIFEST_PATH, EXPECTED_MANIFEST_SHA256, "40 WAV source manifest"),
        "producer_json": require_pin(DIAGNOSTIC_JSON_PATH, EXPECTED_DIAGNOSTIC_JSON_SHA256, "V4 diagnostic JSON"),
        "producer_npz": require_pin(DIAGNOSTIC_NPZ_PATH, EXPECTED_DIAGNOSTIC_NPZ_SHA256, "V4 diagnostic arrays"),
    }
    manifest = json.loads(MANIFEST_PATH.read_bytes().decode("utf-8"))
    diagnostic = json.loads(DIAGNOSTIC_JSON_PATH.read_bytes().decode("utf-8"))
    if manifest.get("created_by_raw_sha256") != EXPECTED_MATERIALIZER_SHA256:
        raise RuntimeError("source manifest does not bind the pinned materializer")
    if diagnostic.get("source_manifest_raw_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("diagnostic JSON does not bind the exact source manifest")
    if diagnostic.get("array_archive", {}).get("raw_sha256") != EXPECTED_DIAGNOSTIC_NPZ_SHA256:
        raise RuntimeError("diagnostic JSON does not bind the pinned NPZ")
    if diagnostic.get("diagnostic_source_raw_sha256") != EXPECTED_DIAGNOSTIC_SOURCE_SHA256:
        raise RuntimeError("diagnostic JSON does not bind the pinned producer script")
    if diagnostic.get("implementation_pins", {}).get("v4_envelope_flux_helper_sha256") != EXPECTED_V4_HELPER_SHA256:
        raise RuntimeError("diagnostic JSON does not bind the pinned V4 feature helper")
    if diagnostic.get("implementation_pins", {}).get("audio_io_sha256") != EXPECTED_AUDIO_IO_SHA256 or diagnostic.get("implementation_pins", {}).get("baselines_sha256") != EXPECTED_BASELINES_SHA256:
        raise RuntimeError("diagnostic JSON does not bind the pinned source decoder and RMS envelope")
    if manifest.get("source_implementation_pins", {}).get("v4_envelope_flux_helper", {}).get("raw_sha256") != EXPECTED_V4_HELPER_SHA256:
        raise RuntimeError("source manifest does not bind the pinned V4 helper")
    if manifest.get("r090_control", {}).get("status") != "PREPARED_NOT_FROZEN":
        raise RuntimeError("r090 control must remain unfrozen")
    if manifest.get("scope", {}).get("candidate_transform_executed") is not False:
        raise RuntimeError("source manifest reports a transform execution")
    if diagnostic.get("status") != "PASS_INPUT_ONLY_PHYSICAL_POWER_LOCAL_FLUX":
        raise RuntimeError("pinned producer status is not the fit40 input-only diagnostic")
    clock = diagnostic.get("clock", {})
    if clock != {
        "analysis_hop_samples": HOP,
        "analysis_window": "centered periodic-Hann, 512 samples, zero-padded at boundaries",
        "local_compressor_envelope_samples": LOCAL_WINDOW_SAMPLES,
        "physical_power_fps": 25,
        "physical_power_frame_samples": PHYSICAL_FRAME_SAMPLES,
    }:
        raise RuntimeError(f"unexpected producer mapping clock: {clock}")

    stored = np.load(DIAGNOSTIC_NPZ_PATH, allow_pickle=False)
    try:
        row_reports, output_arrays = verify_rows(manifest, diagnostic, stored)
    finally:
        stored.close()

    source = Path(__file__).resolve()
    source_sha, source_bytes = sha_file(source)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite independent QC output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = output_dir / "fit40_independent_power_flux_arrays.npz"
    np.savez_compressed(arrays_path, **output_arrays)
    arrays_sha, arrays_bytes = sha_file(arrays_path)
    report = {
        "schema": "cross_tfg_fit40_independent_input_numeric_qc_v1",
        "status": "PASS_INDEPENDENT_CANONICAL_PCM_POWER_SIGNED_FLUX_MAPPING_QC",
        "created_by": str(source),
        "created_by_bytes": source_bytes,
        "created_by_raw_sha256": source_sha,
        "inputs": pins,
        "cohort": {
            "split": "historical_fit_only",
            "source_rows": len(row_reports),
            "speaker_count": len({r["speaker_id"] for r in row_reports}),
            "endpoints_per_speaker": 2,
            "canonical_j_values": [0, 15],
            "sample_count_min": min(r["global"]["sample_count"] for r in row_reports),
            "sample_count_median": float(np.median([r["global"]["sample_count"] for r in row_reports])),
            "sample_count_max": max(r["global"]["sample_count"] for r in row_reports),
            "source_wave_bytes": sum(r["source"]["file_bytes"] for r in row_reports),
        },
        "numerical_contract": {
            "canonical_source": "materialized source WAV only: mono 16 kHz PCM16 payload / 32768 -> canonical float32; no source archive opened",
            "global_rms": "sqrt(mean(canonical_float32_pcm cast to float64 squared))",
            "physical_power": "non-overlapping 640-sample 40 ms source-clock frames; mean-square; no window/resampling; final partial frame excluded and reported per row",
            "v4_spectral_clock": "zero-origin centers i*128, periodic Hann length 512, 256 zero samples each boundary; frame count 1+floor(N/128); no time shift",
            "positive_flux": "mean(max(delta_log_magnitude_db,0)) over non-DC rFFT bins, first frame NaN/unavailable",
            "negative_flux": "mean(max(-delta_log_magnitude_db,0)) over the same bins/clock, independently added here; first frame NaN/unavailable",
            "signed_flux_identity": "positive_flux - negative_flux == mean(delta_log_magnitude_db) on each valid frame",
            "local_envelope_endpoint": "centered 512-sample RMS uses edge padding; if a 128-hop center equals N, repeat the final source sample once for that endpoint measurement only; source PCM/sample_count is unchanged",
            "frame_mapping": "STFT center sample=i*128 (time i/125 s); physical video-rate frame i spans [640i,640(i+1)) and center is 640i+320 (time (2i+1)/50 s)",
            "missing_denominators": "all per-row frame counts, excluded sample tails, and NaN first flux frame are explicit; no imputation",
        },
        "scope": {
            "fit_audio_read": 40,
            "source_archives_opened": False,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "enhancement_transform_executed": False,
            "candidate_effect_read": False,
            "model_forward": False,
            "video_render": False,
            "score": False,
            "selection": False,
            "method_frozen": False,
        },
        "independent_arrays": {
            "path": str(arrays_path),
            "bytes": arrays_bytes,
            "raw_sha256": arrays_sha,
            "array_count": len(output_arrays),
            "includes_negative_flux": True,
        },
        "summary": {
            "global_rms_quantiles": quantiles(np.asarray([r["global"]["rms"] for r in row_reports])),
            "physical_40ms_power_dbfs_clip_median_quantiles": quantiles(np.asarray([r["physical_40ms_25fps"]["power_mean_square_quantiles"]["p50"] for r in row_reports])),
            "positive_flux_clip_median_quantiles_db": quantiles(np.asarray([r["stft_512_hop128"]["positive_flux_db_quantiles"]["p50"] for r in row_reports])),
            "negative_flux_clip_median_quantiles_db": quantiles(np.asarray([r["stft_512_hop128"]["negative_flux_db_quantiles"]["p50"] for r in row_reports])),
            "total_stft_frames": int(sum(r["stft_512_hop128"]["frame_count"] for r in row_reports)),
            "total_valid_flux_frames_per_sign": int(sum(r["stft_512_hop128"]["positive_flux_valid_frames"] for r in row_reports)),
            "total_25fps_frames": int(sum(r["physical_40ms_25fps"]["frame_count"] for r in row_reports)),
            "total_25fps_trailing_samples_excluded": int(sum(r["physical_40ms_25fps"]["trailing_samples_excluded"] for r in row_reports)),
        },
        "rows": row_reports,
    }
    report_path = output_dir / "fit40_independent_input_numeric_qc.json"
    report_raw = canonical_json(report) + b"\n"
    with report_path.open("xb") as stream:
        stream.write(report_raw)
    report["receipt"] = {
        "path": str(report_path),
        "bytes": len(report_raw),
        "raw_sha256": sha_bytes(report_raw),
    }
    # Receipt fields are written into a sibling final file to avoid self-hash recursion.
    receipt_path = output_dir / "fit40_independent_input_numeric_qc_receipt.json"
    receipt_raw = canonical_json(report) + b"\n"
    with receipt_path.open("xb") as stream:
        stream.write(receipt_raw)
    return {
        "status": report["status"],
        "row_count": len(row_reports),
        "receipt_path": str(receipt_path),
        "receipt_bytes": len(receipt_raw),
        "receipt_sha256": sha_bytes(receipt_raw),
        "arrays_path": str(arrays_path),
        "arrays_bytes": arrays_bytes,
        "arrays_sha256": arrays_sha,
        "scope": report["scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREP / "fit40_independent_numeric_qc_attempt01",
    )
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
