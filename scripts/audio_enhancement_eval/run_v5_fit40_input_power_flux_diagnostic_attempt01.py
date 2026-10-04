#!/usr/bin/env python3
"""Input-only physical-power and local-flux summaries for frozen fit40 audio.

The executable reads only the 40 already-selected ADR013 historical-fit WAVs.
It does not transform audio, render a model, score, select, or read dev/confirm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIAGNOSTIC_SOURCE_SHA256 = "9b48d0fefd3c34221f6929da08f7dede9b761f579d7fc56d9fc0c1c1e8d68577"
EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
SAMPLE_RATE_HZ = 16_000
N_FFT = 512
HOP = 128
LOCAL_WINDOW_SAMPLES = 512
VIDEO_FRAME_SAMPLES = 640


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
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


def load_v4_envelope_flux_helper() -> Any:
    helper_path = ROOT / "scripts/audio_enhancement_eval/decompose_v4_delta_common_mode_attempt04.py"
    actual_sha, _ = sha_file(helper_path)
    if actual_sha != EXPECTED_DIAGNOSTIC_SOURCE_SHA256:
        raise RuntimeError(f"pinned V4 envelope/flux helper changed: {actual_sha}")
    audio_io_path = ROOT / "scripts/audio_enhancement/audio_io.py"
    audio_io_sha, _ = sha_file(audio_io_path)
    if audio_io_sha != EXPECTED_AUDIO_IO_SHA256:
        raise RuntimeError(f"pinned audio IO implementation changed: {audio_io_sha}")
    baseline_path = ROOT / "scripts/audio_enhancement/baselines.py"
    baseline_sha, _ = sha_file(baseline_path)
    if baseline_sha != EXPECTED_BASELINES_SHA256:
        raise RuntimeError(f"pinned local-envelope implementation changed: {baseline_sha}")
    spec = importlib.util.spec_from_file_location("_fit40_v4_envelope_flux_pinned", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned V4 envelope/flux helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if int(module.SAMPLE_RATE) != SAMPLE_RATE_HZ or int(module.N_FFT) != N_FFT or int(module.HOP) != HOP:
        raise RuntimeError("pinned V4 helper clock differs from declared 16 kHz / 512 / 128")
    return module


def local_envelope_rms_from_frozen_compressor(
    samples: np.ndarray, *, last_center_sample: int | None = None
) -> np.ndarray:
    """Compute the baseline's centered edge-padded 32 ms RMS envelope only.

    V4's centered STFT clock can include a final center exactly at N when the
    input length is hop-aligned. For that one diagnostic sample, extend the
    edge value by one sample, matching the compressor's edge-padding policy;
    the source waveform and its sample count are unchanged.
    """
    x = np.asarray(samples, dtype=np.float32)
    requested_last = x.size - 1 if last_center_sample is None else int(last_center_sample)
    if requested_last < 0 or requested_last > x.size:
        raise ValueError("requested local-envelope center is outside the input/endpoint")
    extension = max(0, requested_last - (x.size - 1))
    if extension:
        x = np.pad(x, (0, extension), mode="edge")
    n_window = LOCAL_WINDOW_SAMPLES
    left = n_window // 2 - 1
    right = n_window - 1 - left
    power = x.astype(np.float64) ** 2
    padded = np.pad(power, (left, right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    local_power = (cumulative[n_window:] - cumulative[:-n_window]) / n_window
    if local_power.size != x.size:
        raise RuntimeError("local envelope computation changed sample count")
    return np.sqrt(np.maximum(local_power, 1e-10))


def decode_and_validate(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    wav_path = Path(row["transfer_local_path"]).resolve(strict=True)
    file_sha, file_bytes = sha_file(wav_path)
    if file_sha != row["transfer_file_sha256"] or file_sha != row["source_member_sha256"]:
        raise ValueError(f"source WAV file hash mismatch: {row['sample_id']}")
    if file_bytes != int(row["transfer_file_bytes"]) or file_bytes != int(row["source_member_bytes"]):
        raise ValueError(f"source WAV byte count mismatch: {row['sample_id']}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.audio_enhancement.audio_io import _parse_wave, read_canonical_wav

    raw = wav_path.read_bytes()
    fmt_code, channels, rate, bits, block_align, payload = _parse_wave(raw)
    if (fmt_code, channels, rate, bits) != (1, 1, SAMPLE_RATE_HZ, 16):
        raise ValueError(f"unexpected fit40 WAV encoding for {row['sample_id']}")
    if sha_bytes(payload) != row["source_pcm_payload_sha256"]:
        raise ValueError(f"source PCM payload hash mismatch: {row['sample_id']}")
    if block_align != 2 or len(payload) // block_align != int(row["sample_count"]):
        raise ValueError(f"source PCM sample count mismatch: {row['sample_id']}")
    samples, info = read_canonical_wav(wav_path, target_rate=SAMPLE_RATE_HZ)
    if info.file_sha256 != file_sha or info.source_encoding != "PCM_16" or info.source_channels != 1 or info.source_sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(f"canonical WAV reader lineage mismatch: {row['sample_id']}")
    if info.frames != int(row["sample_count"]) or info.pcm_float32_sha256 != row["canonical_float32_pcm_sha256"]:
        raise ValueError(f"canonical float PCM binding mismatch: {row['sample_id']}")
    return samples, {
        "path": str(wav_path),
        "file_sha256": file_sha,
        "file_bytes": file_bytes,
        "pcm_payload_sha256": sha_bytes(payload),
        "pcm_float32_sha256": info.pcm_float32_sha256,
        "sample_count": info.frames,
        "sample_rate_hz": info.sample_rate,
        "channels": info.source_channels,
        "source_encoding": info.source_encoding,
    }


def analyze_row(row: dict[str, Any], samples: np.ndarray, helper: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_samples = int(samples.size)
    if n_samples < LOCAL_WINDOW_SAMPLES:
        raise ValueError(f"fit40 input shorter than 32 ms window: {row['sample_id']}")
    centers = np.arange(1 + n_samples // HOP, dtype=np.int64) * HOP
    env_db, positive_flux = helper.envelope_and_flux(samples, centers)
    if env_db.shape != centers.shape or positive_flux.shape != centers.shape:
        raise RuntimeError("pinned helper returned arrays off the shared sample clock")

    local_env_extended = local_envelope_rms_from_frozen_compressor(
        samples, last_center_sample=int(centers[-1])
    )
    local_env = local_env_extended[:n_samples]
    local_env_at_centers = local_env_extended[centers]
    local_power = np.square(local_env_at_centers.astype(np.float64))
    hann_power_proxy = np.power(10.0, np.asarray(env_db, dtype=np.float64) / 10.0)
    n_25fps = n_samples // VIDEO_FRAME_SAMPLES
    aligned_samples = n_25fps * VIDEO_FRAME_SAMPLES
    video_frames = samples[:aligned_samples].astype(np.float64).reshape(n_25fps, VIDEO_FRAME_SAMPLES)
    frame_power = np.mean(video_frames * video_frames, axis=1, dtype=np.float64)
    frame_power_dbfs = 10.0 * np.log10(np.maximum(frame_power, 1e-20))
    flux_finite = np.asarray(positive_flux, dtype=np.float64)
    global_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2, dtype=np.float64)))
    global_power = global_rms * global_rms
    peak = float(np.max(np.abs(samples.astype(np.float64))))
    local_env_dbfs = 20.0 * np.log10(np.maximum(local_env_at_centers.astype(np.float64), 1e-12))
    row_summary = {
        "row_id": row["row_id"],
        "speaker_id": row["speaker_id"],
        "fold_index": int(row["fold_index"]),
        "canonical_j": int(row["canonical_j"]),
        "sample_id": row["sample_id"],
        "source": {
            "path": row["transfer_local_path"],
            "archive_sha256": row["source_archive_sha256"],
            "member": row["source_archive_member"],
            "member_sha256": row["source_member_sha256"],
            "pcm_payload_sha256": row["source_pcm_payload_sha256"],
            "canonical_float32_pcm_sha256": row["canonical_float32_pcm_sha256"],
            "sample_count": n_samples,
            "sample_rate_hz": SAMPLE_RATE_HZ,
        },
        "global_physical_summary": {
            "mean_square_amplitude": global_power,
            "rms_amplitude": global_rms,
            "peak_abs_amplitude": peak,
            "peak_to_rms": peak / max(global_rms, 1e-20),
            "samples_above_0_98": int(np.count_nonzero(np.abs(samples) > 0.98)),
        },
        "r090_compressor_input_envelope": {
            "definition": "centered edge-padded moving mean-square, 512 samples at 16 kHz, matching baselines.apply_local_envelope_compression; read-only input measurement",
            "rms_amplitude_quantiles": quantiles(local_env),
            "power_amplitude_squared_quantiles": quantiles(np.square(local_env.astype(np.float64))),
            "rms_dbfs_quantiles": quantiles(20.0 * np.log10(np.maximum(local_env.astype(np.float64), 1e-12))),
            "frame_centered_8ms_sampled_rms_dbfs_quantiles": quantiles(local_env_dbfs),
        },
        "frozen_profile_clock_descriptors": {
            "helper_source_path": "scripts/audio_enhancement_eval/decompose_v4_delta_common_mode_attempt04.py",
            "helper_source_sha256": EXPECTED_DIAGNOSTIC_SOURCE_SHA256,
            "window": "centered periodic-Hann, 512 samples, zero-padded by 256 at each utterance boundary",
            "hop_samples": HOP,
            "frame_count": int(centers.size),
            "frame_centers_first_last_samples": [int(centers[0]), int(centers[-1])],
            "hann_windowed_envelope_dbfs_quantiles": quantiles(env_db),
            "hann_windowed_mean_square_unscaled_quantiles": quantiles(hann_power_proxy),
            "positive_spectral_flux_db_quantiles": quantiles(flux_finite),
            "positive_flux_definition": "mean positive frame-to-frame dB-magnitude difference over non-DC rFFT bins; first frame unavailable, exactly as the pinned V4 helper",
        },
        "physical_25fps_power": {
            "definition": "mean squared decoded PCM amplitude in non-overlapping 640-sample source-clock frames; no window or resampling",
            "frame_count": n_25fps,
            "trailing_samples_excluded": n_samples - aligned_samples,
            "mean_square_quantiles": quantiles(frame_power),
            "dbfs_quantiles": quantiles(frame_power_dbfs),
        },
        "analysis_scope": "input_only; no enhancement transform, candidate comparison, model forward, render, SyncNet score, dev, or confirmation audio",
    }
    arrays = {
        f"{row['row_id']}_frame_center_sample": centers.astype(np.int64),
        f"{row['row_id']}_local_envelope_rms": local_env_at_centers.astype(np.float32),
        f"{row['row_id']}_local_power_amplitude_squared": local_power.astype(np.float32),
        f"{row['row_id']}_hann_envelope_dbfs": np.asarray(env_db, dtype=np.float32),
        f"{row['row_id']}_hann_windowed_mean_square_unscaled": hann_power_proxy.astype(np.float32),
        f"{row['row_id']}_positive_spectral_flux_db": np.asarray(positive_flux, dtype=np.float32),
        f"{row['row_id']}_physical_25fps_frame_power": frame_power.astype(np.float32),
        f"{row['row_id']}_physical_25fps_frame_power_dbfs": frame_power_dbfs.astype(np.float32),
    }
    if not all(np.isfinite(values).all() for key, values in arrays.items() if "positive_spectral_flux_db" not in key):
        raise RuntimeError(f"non-finite physical diagnostic array for {row['sample_id']}")
    if not np.isfinite(flux_finite[1:]).all() or not np.isnan(flux_finite[0]):
        raise RuntimeError(f"flux finite/first-frame contract differs for {row['sample_id']}")
    return row_summary, arrays


def run(manifest_path: Path, expected_manifest_sha: str, output_dir: Path) -> dict[str, Any]:
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = sha_bytes(manifest_raw)
    if manifest_sha != expected_manifest_sha:
        raise ValueError(f"fit40 source manifest raw SHA mismatch: {manifest_sha}")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1" or manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise ValueError("input manifest is not the prepared fit-only 40-endpoint bundle")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40 or len({row.get("sample_id") for row in rows}) != 40:
        raise ValueError("input manifest must contain exactly 40 unique fit endpoints")
    if set(manifest.get("fit_speakers", [])) != {f"S{n:04d}" for n in range(52, 72)}:
        raise ValueError("input manifest fit speakers differ from the frozen S0052-S0071 cohort")
    if manifest.get("scope", {}).get("dev_audio_read") is not False or manifest.get("scope", {}).get("confirm_audio_read") is not False:
        raise ValueError("input transfer manifest reports a prohibited split read")
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    helper = load_v4_envelope_flux_helper()
    summaries = []
    arrays: dict[str, np.ndarray] = {}
    for row in rows:
        if row.get("speaker_id") not in {f"S{n:04d}" for n in range(52, 72)}:
            raise ValueError(f"non-fit speaker in input manifest: {row.get('speaker_id')}")
        samples, source_binding = decode_and_validate(row)
        summary, row_arrays = analyze_row(row, samples, helper)
        summary["verified_source"] = source_binding
        summaries.append(summary)
        arrays.update(row_arrays)

    npz_path = output_dir / "fit40_input_power_flux_arrays.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_sha, npz_bytes = sha_file(npz_path)
    result = {
        "schema": "cross_tfg_v5_fit40_input_power_flux_diagnostic_v1",
        "status": "PASS_INPUT_ONLY_PHYSICAL_POWER_LOCAL_FLUX",
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_raw_sha256": manifest_sha,
        "diagnostic_source_path": Path(__file__).relative_to(ROOT).as_posix(),
        "diagnostic_source_raw_sha256": sha_file(Path(__file__))[0],
        "input_count": len(summaries),
        "speaker_count": len({row["speaker_id"] for row in summaries}),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "clock": {
            "analysis_window": "centered periodic-Hann, 512 samples, zero-padded at boundaries",
            "analysis_hop_samples": HOP,
            "physical_power_frame_samples": VIDEO_FRAME_SAMPLES,
            "physical_power_fps": 25,
            "local_compressor_envelope_samples": LOCAL_WINDOW_SAMPLES,
        },
        "implementation_pins": {
            "v4_envelope_flux_helper_sha256": EXPECTED_DIAGNOSTIC_SOURCE_SHA256,
            "audio_io_sha256": EXPECTED_AUDIO_IO_SHA256,
            "baselines_sha256": EXPECTED_BASELINES_SHA256,
        },
        "array_archive": {
            "path": str(npz_path.resolve()),
            "raw_sha256": npz_sha,
            "bytes": npz_bytes,
            "array_count": len(arrays),
        },
        "scope": {
            "fit_audio_read": 40,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "enhancement_transform_executed": False,
            "candidate_effect_read": False,
            "model_forward": False,
            "video_render": False,
            "score": False,
            "selection": False,
        },
        "rows": summaries,
    }
    out_raw = canonical_json(result) + b"\n"
    report_path = output_dir / "fit40_input_power_flux_diagnostic.json"
    with report_path.open("xb") as stream:
        stream.write(out_raw)
    return {
        "status": result["status"],
        "report_path": str(report_path.resolve()),
        "report_bytes": len(out_raw),
        "report_raw_sha256": sha_bytes(out_raw),
        "array_archive_path": str(npz_path.resolve()),
        "array_archive_bytes": npz_bytes,
        "array_archive_raw_sha256": npz_sha,
        "input_count": len(summaries),
        "speaker_count": result["speaker_count"],
        "scope": result["scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.manifest.resolve(strict=True), args.expected_manifest_sha256, args.output_dir.resolve())
    print(json.dumps(outcome, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
