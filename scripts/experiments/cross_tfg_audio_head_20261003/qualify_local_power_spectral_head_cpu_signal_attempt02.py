"""CPU-only signal qualification for the source-conditioned 11-scalar head.

This is an engineering qualification over the eight sealed historical fit
utterances. It does not run a generator, frontend encoder, SyncNet, optimizer,
or selection procedure. It writes no source or candidate waveform copies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross_tfg_local_power_spectral_head_cpu_signal_qualification_v1"
EXPECTED_PROBE_PLAN_SHA256 = "0b6f7ba4935ccb7b411148f67b8024cf056a58bafdbb55dce7f008761cf88f5a"
EXPECTED_ARCHIVE_MANIFEST_SHA256 = "0d73826ef19a33698c41b4d50408adc7adc76a69e007c2ea8a9f4081aa0f642b"
EXPECTED_FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES_40MS = 640
TARGET_RMS = 0.0376838172675746
CELL_IDS = ("rms_reference", "envelope_only", "shape_only", "full")
POINT_IDS = ("zero", "low_plus", "low_minus", "high_plus", "high_minus")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(values: Any) -> dict[str, Any]:
    import numpy as np

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"count": int(data.size), "finite_count": 0, "mean": None, "median": None, "p05": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "count": int(data.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _verify_file_pin(repo_root: Path, record: Mapping[str, Any], label: str) -> Path:
    rel = record.get("path")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if not isinstance(rel, str) or not isinstance(expected_bytes, int):
        raise ValueError(f"malformed source pin {label}")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != expected_sha:
        raise ValueError(f"invalid SHA-256 in source pin {label}")
    path = (repo_root / rel).resolve(strict=True)
    path.relative_to(repo_root.resolve(strict=True))
    size, actual_sha = _sha256_file(path)
    if size != expected_bytes or actual_sha != expected_sha:
        raise ValueError(f"source pin mismatch {label}: {size}/{actual_sha}")
    return path


def _verify_input_file(task_root: Path, record: Mapping[str, Any], label: str) -> Path:
    rel = record.get("relative_path")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if not isinstance(rel, str) or not isinstance(expected_bytes, int):
        raise ValueError(f"malformed staged input pin {label}")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != expected_sha:
        raise ValueError(f"invalid staged-input SHA-256 {label}")
    path = (task_root / rel).resolve(strict=True)
    path.relative_to(task_root.resolve(strict=True))
    size, actual_sha = _sha256_file(path)
    if size != expected_bytes or actual_sha != expected_sha:
        raise ValueError(f"staged input pin mismatch {label}: {size}/{actual_sha}")
    return path


def _source_rows(config: Mapping[str, Any], probe: Mapping[str, Any], archive: Mapping[str, Any]) -> list[dict[str, Any]]:
    planned = probe.get("input_rows")
    archived = archive.get("inputs")
    configured = config.get("input_rows")
    if not isinstance(planned, list) or len(planned) != 8:
        raise ValueError("frozen fit8 probe plan must retain all eight source rows")
    if not isinstance(archived, list) or len(archived) != 8:
        raise ValueError("restored source archive manifest must retain all eight inputs")
    if not isinstance(configured, list) or len(configured) != 8:
        raise ValueError("qualification config must bind all eight source rows")
    pmap = {row.get("sample_id"): row for row in planned}
    amap = {row.get("sample_id"): row for row in archived}
    cmap = {row.get("sample_id"): row for row in configured}
    if len(pmap) != 8 or len(amap) != 8 or len(cmap) != 8 or set(pmap) != set(amap) or set(pmap) != set(cmap):
        raise ValueError("probe, restored archive, and config source rosters differ")
    rows = []
    for sample_id in sorted(pmap):
        row = pmap[sample_id]
        natural = row.get("natural_pcm", {})
        cfg = cmap[sample_id]
        archive_row = amap[sample_id]
        if row.get("split") != "fit" or row.get("seed") != 20261003:
            raise ValueError(f"non-fit or wrong-seed row in fit8 cohort: {sample_id}")
        if cfg.get("split") != "fit" or cfg.get("speaker_id") != row.get("speaker_id"):
            raise ValueError(f"configured identity mismatch: {sample_id}")
        checks = {
            "source_wav_path": natural.get("path"),
            "wav_file_sha256": natural.get("file_sha256"),
            "canonical_pcm_float32_sha256": natural.get("pcm_sha256"),
            "sample_count": natural.get("sample_count"),
        }
        for key, expected in checks.items():
            if cfg.get(key) != expected:
                raise ValueError(f"probe/config {key} mismatch for {sample_id}")
        if (
            archive_row.get("speaker_id") != row.get("speaker_id")
            or archive_row.get("encoding") != "PCM_16"
            or archive_row.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or archive_row.get("sha256") != natural.get("file_sha256")
            or archive_row.get("pcm_float32_sha256") != natural.get("pcm_sha256")
            or archive_row.get("sample_count") != natural.get("sample_count")
        ):
            raise ValueError(f"restored archive binding mismatch for {sample_id}")
        if cfg.get("wav_file_bytes", 0) <= 44:
            raise ValueError(f"invalid measured WAV byte count for {sample_id}")
        rows.append(dict(cfg))
    return rows


def _set_coordinates(head: Any, shape: Any, envelope: Any, torch: Any) -> None:
    with torch.no_grad():
        head.shape_coordinates.copy_(torch.as_tensor(shape, dtype=torch.float32, device="cpu"))
        head.envelope_coordinates.copy_(torch.as_tensor(envelope, dtype=torch.float32, device="cpu"))


def _torch_stft_callable(torch: Any, v3_config: Mapping[str, Any]):
    n_fft = int(v3_config["n_fft"])
    hop = int(v3_config["hop_length"])
    win_length = int(v3_config["win_length"])
    periodic = bool(v3_config["window_periodic"])
    pad_mode = str(v3_config["stft_pad_mode"])
    window = torch.hann_window(win_length, periodic=periodic, dtype=torch.float32, device="cpu")

    def stft(pcm: Any, *, n_fft: int, hop_length: int):
        import numpy as np

        if n_fft != int(v3_config["n_fft"]) or hop_length != int(v3_config["hop_length"]):
            raise ValueError("phase helper requested an STFT outside the sealed V3 clock")
        x = np.ascontiguousarray(pcm, dtype=np.float32)
        y = torch.from_numpy(x)
        spectrum = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=bool(v3_config["stft_center"]),
            pad_mode=pad_mode,
            normalized=bool(v3_config["stft_normalized"]),
            onesided=bool(v3_config["stft_onesided"]),
            return_complex=bool(v3_config["stft_return_complex"]),
        ).detach().cpu().numpy()
        centers = np.arange(spectrum.shape[1], dtype=np.int64) * hop_length
        return spectrum, centers, window.detach().cpu().numpy()

    return stft


def _phase_summary(phase_result: Any) -> dict[str, Any]:
    import numpy as np

    arrays = phase_result.arrays
    centroid = arrays["output_minus_source_acoustic_energy_centroid_samples_40ms"]
    lag = arrays["local_xcorr_argmax_output_delay_samples"]
    corr = arrays["local_xcorr_argmax_value"]
    usable_lag = lag != np.iinfo(np.int64).min
    phase = phase_result.summary
    return {
        "phase_comparisons": phase.get("phase_comparisons"),
        "phase_shared_nonzero_support_bins_all_three": phase.get("phase_shared_nonzero_support_bins_all_three"),
        "stft_centers_identical_source_and_reanalysis": phase.get("stft_centers_identical_source_and_reanalysis"),
        "stft_window_identical_source_and_reanalysis": phase.get("stft_window_identical_source_and_reanalysis"),
        "local_xcorr": {
            "window_samples": phase.get("local_waveform_xcorr", {}).get("window_samples"),
            "hop_samples": phase.get("local_waveform_xcorr", {}).get("hop_samples"),
            "lag_grid_min_samples": phase.get("local_waveform_xcorr", {}).get("lag_grid_min_samples"),
            "lag_grid_max_samples": phase.get("local_waveform_xcorr", {}).get("lag_grid_max_samples"),
            "valid_window_count": int(np.count_nonzero(usable_lag)),
            "argmax_delay_samples": _summary(lag[usable_lag]),
            "argmax_correlation": _summary(corr[usable_lag]),
        },
        "within_frame_energy_centroid_shift_samples_40ms": _summary(centroid),
        "within_frame_energy_centroid_semantics": phase.get("transient_centroid_semantics"),
        "phase_semantics": phase.get("phase_semantics"),
    }


def _power_arrays(source: Any, output: Any, baseline: Any, np: Any) -> dict[str, Any]:
    n_frames = int(source.size // FRAME_SAMPLES_40MS)
    used = n_frames * FRAME_SAMPLES_40MS
    source_frames = np.asarray(source[:used], dtype=np.float64).reshape(n_frames, FRAME_SAMPLES_40MS)
    output_frames = np.asarray(output[:used], dtype=np.float64).reshape(n_frames, FRAME_SAMPLES_40MS)
    baseline_frames = np.asarray(baseline[:used], dtype=np.float64).reshape(n_frames, FRAME_SAMPLES_40MS)
    source_power = np.mean(source_frames * source_frames, axis=1, dtype=np.float64)
    output_power = np.mean(output_frames * output_frames, axis=1, dtype=np.float64)
    baseline_power = np.mean(baseline_frames * baseline_frames, axis=1, dtype=np.float64)
    valid = source_power > 0.0
    ratio = np.zeros(n_frames, dtype=np.float64)
    ratio[valid] = output_power[valid] / source_power[valid]
    base_ratio = np.zeros(n_frames, dtype=np.float64)
    base_ratio[valid] = baseline_power[valid] / source_power[valid]
    ratio_db = np.full(n_frames, np.nan, dtype=np.float64)
    ratio_db[valid] = 10.0 * np.log10(np.maximum(ratio[valid], np.finfo(np.float64).tiny))
    return {
        "frame_count": n_frames,
        "trailing_samples_excluded": int(source.size - used),
        "source_mean_square": source_power,
        "output_mean_square": output_power,
        "baseline_mean_square": baseline_power,
        "output_ratio_to_natural": ratio,
        "output_ratio_db_to_natural": ratio_db,
        "baseline_ratio_to_natural": base_ratio,
        "source_nonzero_mask": valid,
    }


def _cell_metrics(cell: Mapping[str, Any], source_np: Any, baseline_np: Any, result: Mapping[str, Any], np: Any, torch: Any, tolerance: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output = np.ascontiguousarray(cell["output_pcm_float32"].detach().cpu().numpy(), dtype=np.float32)
    mask = np.ascontiguousarray(cell["log_magnitude_mask_db"].detach().cpu().numpy(), dtype=np.float32)
    constructed = np.ascontiguousarray(cell["constructed_stft"].detach().cpu().numpy())
    source_stft = result["source_stft"].detach().cpu().numpy()
    weights = result["one_sided_power_bin_weights"].detach().cpu().numpy().astype(np.float64, copy=False)
    source_power = np.sum(np.abs(source_stft.astype(np.complex128)) ** 2 * weights[:, None], axis=0, dtype=np.float64)
    output_power = np.sum(np.abs(constructed.astype(np.complex128)) ** 2 * weights[:, None], axis=0, dtype=np.float64)
    support = np.asarray(result["stft_power_support_mask"].detach().cpu().numpy(), dtype=bool)
    weighted_ratio = np.zeros(source_power.shape, dtype=np.float64)
    weighted_ratio[support] = output_power[support] / source_power[support]
    internal_ratio = np.asarray(cell["one_sided_weighted_stft_power_ratio_to_source"].detach().cpu().numpy(), dtype=np.float64)
    ratio_error = np.abs(weighted_ratio[support] - internal_ratio[support]) if np.any(support) else np.asarray([], dtype=np.float64)
    powers = _power_arrays(source_np, output, baseline_np, np)
    output_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2, dtype=np.float64)))
    baseline_rms = float(np.sqrt(np.mean(np.asarray(baseline_np, dtype=np.float64) ** 2, dtype=np.float64)))
    mask_float64 = mask.astype(np.float64, copy=False)
    frequency_mean_mask_db = np.mean(mask_float64, axis=1, dtype=np.float64)
    metrics = {
        "exact_sample_count": int(output.size) == int(source_np.size),
        "output_finite": bool(np.isfinite(output).all()),
        "output_pcm_float32_sha256": _sha256_bytes(np.ascontiguousarray(output, dtype="<f4").tobytes()),
        "output_rms": output_rms,
        "canonical_baseline_rms": baseline_rms,
        "rms_error_to_target": output_rms - TARGET_RMS,
        "output_peak": float(np.max(np.abs(output))),
        "peak_ceiling": None,
        "mask_db": {
            "shape": list(mask.shape),
            "min": float(np.min(mask)),
            "max": float(np.max(mask)),
            "max_abs": float(np.max(np.abs(mask))),
            "mean": float(np.mean(mask_float64, dtype=np.float64)),
            "std": float(np.std(mask_float64, dtype=np.float64)),
        },
        "weighted_stft_power_ratio_to_natural_after_mask": _summary(weighted_ratio[support]),
        "weighted_stft_power_support_frames": int(np.count_nonzero(support)),
        "weighted_stft_ratio_head_vs_independent_numpy_max_abs_error": float(np.max(ratio_error)) if ratio_error.size else None,
        "physical_40ms": {
            "frame_count": powers["frame_count"],
            "trailing_samples_excluded": powers["trailing_samples_excluded"],
            "output_power_ratio_to_natural": _summary(powers["output_ratio_to_natural"][powers["source_nonzero_mask"]]),
            "output_power_change_db_to_natural": _summary(powers["output_ratio_db_to_natural"][powers["source_nonzero_mask"]]),
            "canonical_rms_power_ratio_to_natural": _summary(powers["baseline_ratio_to_natural"][powers["source_nonzero_mask"]]),
            "source_zero_power_frames": int(np.count_nonzero(~powers["source_nonzero_mask"])),
        },
        "normalization_reported_by_head": {
            "rms": float(cell["output_rms"].detach().cpu().item()),
            "peak": float(cell["output_peak"].detach().cpu().item()),
            "peak_limited": bool(cell.get("normalization", {}).get("peak_limited", torch.tensor(False)).detach().cpu().item()) if hasattr(cell.get("normalization", {}).get("peak_limited", False), "detach") else False,
        },
    }
    checks = {
        "exact_sample_count": metrics["exact_sample_count"],
        "output_finite": metrics["output_finite"],
        "final_rms_within_tolerance": abs(metrics["rms_error_to_target"]) <= float(tolerance["final_rms_abs_tolerance"]),
        "mask_within_declared_bound": metrics["mask_db"]["max_abs"] <= 3.0 + float(tolerance["mask_bound_tolerance_db"]),
        "weighted_stft_ratio_finite": bool(np.isfinite(weighted_ratio[support]).all()),
    }
    return metrics, {
        **powers,
        "frequency_mean_mask_db": frequency_mean_mask_db.astype(np.float32),
        "weighted_stft_ratio": weighted_ratio.astype(np.float32),
        "weighted_stft_support": support,
        "checks": checks,
    }


def _verify_plan_bundle(task_root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    inputs = config["staged_inputs"]
    probe_path = _verify_input_file(task_root, inputs["probe_plan"], "probe_plan")
    archive_path = _verify_input_file(task_root, inputs["archive_manifest"], "archive_manifest")
    probe = _read_json(probe_path)
    archive = _read_json(archive_path)
    if probe.get("schema") != "cross_tfg_native_audio_encoder_probe_plan_v1":
        raise ValueError("probe plan schema changed")
    if probe.get("source_plan", {}).get("sha256") != EXPECTED_FIT_PLAN_SHA256:
        raise ValueError("probe plan is not attached to the sealed historical fit plan")
    if probe.get("scope", {}).get("development_audio_opened") is not False or probe.get("scope", {}).get("confirmation_audio_opened") is not False:
        raise ValueError("probe plan no longer proves Dev/Confirm closure")
    rows = _source_rows(config, probe, archive)
    staged_wavs = config["staged_inputs"].get("source_wavs")
    if not isinstance(staged_wavs, list) or len(staged_wavs) != 8:
        raise ValueError("all eight source WAVs must be explicitly staged and pinned")
    wav_map = {item.get("sample_id"): item for item in staged_wavs}
    if len(wav_map) != 8 or set(wav_map) != {row["sample_id"] for row in rows}:
        raise ValueError("staged source WAV roster is incomplete or duplicated")
    for row in rows:
        record = wav_map[row["sample_id"]]
        staged_path = _verify_input_file(task_root, record, f"source_wav:{row['sample_id']}")
        declared_path = Path(row["remote_wav_path"]).resolve(strict=True)
        if declared_path != staged_path or int(record["bytes"]) != int(row["wav_file_bytes"]) or record["sha256"] != row["wav_file_sha256"]:
            raise ValueError(f"staged source WAV identity/path mismatch for {row['sample_id']}")
    return probe, archive, rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    config_path = Path(args.config).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    config_sha = _sha256_bytes(config_bytes)
    if config_sha != args.expected_config_sha256:
        raise ValueError("raw configuration SHA does not match the invocation pin")
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema") != SCHEMA or config.get("status") != "PREPARED_NOT_DISPATCHED":
        raise ValueError("qualification config schema/status mismatch")
    if config.get("scope") != {
        "historical_fit8_only": True,
        "development_audio_read": False,
        "confirmation_audio_read": False,
        "frontend_or_generator_forward": False,
        "syncnet_or_video": False,
        "optimizer_or_parameter_update": False,
        "waveform_persisted": False,
        "candidate_selection": False,
        "api_calls": False,
    }:
        raise ValueError("qualification scope is not the sealed CPU signal-only scope")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU signal qualification requires CUDA_VISIBLE_DEVICES to be empty")

    repo_root = Path(args.repo_root).resolve(strict=True)
    task_root = Path(args.task_root).resolve(strict=True)
    output_dir = Path(args.output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError("fresh output directory already exists")
    if output_dir.parent.resolve(strict=True) != Path(config["output_parent"]).resolve(strict=True):
        raise ValueError("output path differs from the sealed fresh output parent")
    if args.source_root and Path(args.source_root).resolve(strict=True) != repo_root:
        raise ValueError("source-root and repository-root differ")

    for label, record in config["source_pins"].items():
        _verify_file_pin(repo_root, record, label)
    for name, expected in config["runtime"]["environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"runtime environment mismatch for {name}")
    probe, archive, rows = _verify_plan_bundle(task_root, config)
    if len(rows) != 8 or len({row["sample_id"] for row in rows}) != 8:
        raise ValueError("complete fit8 denominator is required")
    limits = config["resource_limits"]
    if any(
        int(row["sample_count"]) > int(limits["max_audio_samples"])
        or int(row["wav_file_bytes"]) > int(limits["max_single_wav_bytes"])
        for row in rows
    ):
        raise ValueError("a fit8 source row exceeds the sealed audio size limits")
    v3_cfg = config["head_constructor"]["v3_configuration"]
    if v3_cfg.get("schema") != "cross_tfg_shared_recipient_transition_head_v3" or v3_cfg.get("stored_trainable_scalar_count") != 9:
        raise ValueError("nested V3 constructor reference is malformed")
    if v3_cfg.get("max_abs_mask_db") != 3.0 or v3_cfg.get("peak_ceiling") is not None or v3_cfg.get("target_rms") != TARGET_RMS:
        raise ValueError("V3 baseline/dose configuration changed")
    points = config["parameter_points"]
    if set(points) != set(POINT_IDS):
        raise ValueError("parameter point set must be exactly zero, ±0.2 and ±2 probes")
    for pair in (("low_plus", "low_minus"), ("high_plus", "high_minus")):
        for coord in ("shape_coordinates", "envelope_coordinates"):
            if not np.array_equal(np.asarray(points[pair[0]][coord], dtype=np.float32), -np.asarray(points[pair[1]][coord], dtype=np.float32)):
                raise ValueError("declared plus/minus probe points are not exact sign pairs")
    if np.any(np.asarray(points["zero"]["shape_coordinates"], dtype=np.float32)) or np.any(np.asarray(points["zero"]["envelope_coordinates"], dtype=np.float32)):
        raise ValueError("zero parameter point is not exact float32 zero")

    # All project sources and staged plans are hash-checked before importing the
    # package (whose __init__ imports the pure waveform pipeline helpers).
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import torch
    from scripts.audio_enhancement.audio_io import pcm_float32_sha256, read_canonical_wav
    from scripts.audio_enhancement.baselines import apply_global_rms
    from scripts.audio_enhancement.phase_clock_diagnostics_v1 import phase_clock_reanalysis_diagnostics_v1
    from scripts.audio_enhancement.shared_recipient_local_power_spectral_head_v1 import (
        ENVELOPE_CENTER_RULE,
        ONE_SIDED_POWER_RULE,
        SCHEMA as HEAD_SCHEMA,
        TRAINABLE_SCALAR_COUNT,
        SharedRecipientLocalPowerSpectralHeadV1,
    )

    if importlib.metadata.version("numpy") != config["runtime"]["numpy_version"]:
        raise RuntimeError("NumPy runtime version differs from the sealed config")
    if torch.__version__ != config["runtime"]["torch_version"]:
        raise RuntimeError("Torch runtime version differs from the sealed config")
    if tuple(sys.version_info[:3]) != tuple(config["runtime"]["python_version"]):
        raise RuntimeError("Python runtime version differs from the sealed config")
    torch.set_num_threads(int(config["runtime"]["torch_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["torch_interop_threads"]))
    if torch.cuda.is_initialized():
        raise RuntimeError("Torch unexpectedly initialized CUDA")
    if HEAD_SCHEMA != "cross_tfg_shared_recipient_local_power_spectral_head_v1" or TRAINABLE_SCALAR_COUNT != 11:
        raise RuntimeError("loaded candidate head schema/scalar count mismatch")
    if config["head_constructor"]["one_sided_power_bin_policy"] != ONE_SIDED_POWER_RULE or config["head_constructor"]["envelope_centering_policy"] != ENVELOPE_CENTER_RULE:
        raise ValueError("explicit local-power policy labels differ from the head source")

    # V1 validates the complete V3 record, including the stored parameter-count
    # declaration and reference source pins. Pass it intact to the constructor.
    head_kwargs = dict(config["head_constructor"])
    v3_behavior = dict(v3_cfg)
    head = SharedRecipientLocalPowerSpectralHeadV1(**head_kwargs).to(device="cpu", dtype=torch.float32).eval()

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    v3_behavior = dict(config["head_constructor"]["v3_configuration"])
    stft_fn = _torch_stft_callable(torch, v3_behavior)
    stft_offsets: dict[str, list[int]] = {point: [0] for point in POINT_IDS}
    source_frame_offsets: list[int] = [0]
    source_frame_sample_ids: list[str] = []
    power_offsets_by_point: dict[str, list[int]] = {point: [0] for point in POINT_IDS}
    power_sample_ids_by_point: dict[str, list[str]] = {point: [] for point in POINT_IDS}
    results = []
    hard_checks: dict[str, bool] = {
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "all_8_source_rows_retained": len(rows) == 8,
        "all_11_parameter_gradients_finite_and_connected": True,
        "every_parameter_has_a_nonzero_observed_gradient": True,
        "all_outputs_exact_length_finite_and_target_rms": True,
        "all_constructed_masks_within_3db": True,
        "all_output_power_diagnostics_finite": True,
        "no_peak_cap_declared": True,
        "supporting_array_offsets_valid": True,
    }
    gradient_nonzero_counts = np.zeros(11, dtype=np.int64)
    gradient_finite_count = 0
    stft_arrays_by_point: dict[str, dict[str, list[Any]]] = {
        point: {cell: [] for cell in CELL_IDS} for point in POINT_IDS
    }
    mask_frequency_means_by_point: dict[str, dict[str, list[Any]]] = {
        point: {cell: [] for cell in CELL_IDS} for point in POINT_IDS
    }
    support_arrays_by_point: dict[str, dict[str, list[Any]]] = {
        point: {cell: [] for cell in CELL_IDS} for point in POINT_IDS
    }
    power_arrays_by_point: dict[str, dict[str, list[Any]]] = {
        point: {cell: [] for cell in CELL_IDS} for point in POINT_IDS
    }
    stft_sample_ids_by_point: dict[str, list[str]] = {point: [] for point in POINT_IDS}
    src_power_parts: list[Any] = []
    rms_power_parts: list[Any] = []

    journal_path = output_dir / "results.partial.jsonl"
    journal = journal_path.open("xb")
    try:
        for row in rows:
            row_record: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "split": "fit",
                "seed": 20261003,
                "source_file_sha256": row["wav_file_sha256"],
                "source_original_path": row["source_wav_path"],
                "source_staged_path": row["remote_wav_path"],
                "source_canonical_pcm_float32_sha256": row["canonical_pcm_float32_sha256"],
                "source_file_bytes": int(row["wav_file_bytes"]),
                "source_sample_count": int(row["sample_count"]),
                "parameter_points": {},
            }
            try:
                source_np, read_info = read_canonical_wav(
                    Path(row["remote_wav_path"]), target_rate=SAMPLE_RATE_HZ
                )
                source_np = np.ascontiguousarray(source_np, dtype=np.float32)
                raw_bytes, raw_sha = _sha256_file(Path(row["remote_wav_path"]))
                pcm_sha = pcm_float32_sha256(source_np)
                if (
                    raw_bytes != int(row["wav_file_bytes"])
                    or raw_sha != row["wav_file_sha256"]
                    or pcm_sha != row["canonical_pcm_float32_sha256"]
                    or source_np.size != int(row["sample_count"])
                    or read_info.sample_rate != SAMPLE_RATE_HZ
                    or read_info.source_channels != 1
                    or read_info.source_encoding != "PCM_16"
                ):
                    raise ValueError("actual decoded WAV does not match the frozen PCM16 source identity")
                baseline_np, rms_info = apply_global_rms(
                    source_np,
                    TARGET_RMS,
                    peak_ceiling=None,
                    silence_epsilon=float(config["numpy_rms_baseline"]["silence_epsilon"]),
                )
                baseline_np = np.ascontiguousarray(baseline_np, dtype=np.float32)
                if not np.isfinite(baseline_np).all() or baseline_np.size != source_np.size:
                    raise FloatingPointError("canonical NumPy RMS baseline is nonfinite or changed sample count")
                source_tensor = torch.from_numpy(source_np)
                baseline_tensor = torch.from_numpy(baseline_np)
                baseline_bytes = np.ascontiguousarray(baseline_np, dtype="<f4").tobytes()
                zero_source_power: Any | None = None
                zero_rms_power: Any | None = None

                for point_id in POINT_IDS:
                    point = points[point_id]
                    pending_masks: dict[str, Any] = {}
                    pending_ratios: dict[str, Any] = {}
                    pending_support: dict[str, Any] = {}
                    pending_physical_power: dict[str, Any] = {}
                    _set_coordinates(head, point["shape_coordinates"], point["envelope_coordinates"], torch)
                    if point_id == "zero":
                        result = head(source_tensor, baseline_tensor)
                        point_grad = None
                        sample_index = torch.arange(source_np.size, dtype=torch.float32)
                        phase_a = (2.0 * math.pi * float(config["gradient_probe"]["cycles"][0]) / source_np.size) * sample_index
                        phase_b = (2.0 * math.pi * float(config["gradient_probe"]["cycles"][1]) / source_np.size) * sample_index
                        phase_c = (2.0 * math.pi * float(config["gradient_probe"]["cycles"][2]) / source_np.size) * sample_index
                        probe_weight = torch.sin(phase_a) + float(config["gradient_probe"]["coefficient_cosine_2"]) * torch.cos(phase_b) + float(config["gradient_probe"]["coefficient_sine_3"]) * torch.sin(phase_c)
                        probe_weight = probe_weight / torch.sqrt(torch.mean(probe_weight.square()).clamp_min(torch.finfo(torch.float32).tiny))
                        residual = result["cells"]["full"]["output_pcm_float32"] - baseline_tensor
                        projection = torch.mean(residual * probe_weight)
                        gradients = torch.autograd.grad(
                            projection,
                            (head.shape_coordinates, head.envelope_coordinates),
                            allow_unused=False,
                            retain_graph=False,
                            create_graph=False,
                        )
                        gradient_matrix = gradients[0].detach().cpu().numpy().astype(np.float64).reshape(-1)
                        gradient_envelope = gradients[1].detach().cpu().numpy().astype(np.float64).reshape(-1)
                        gradient_vector = np.concatenate((gradient_matrix, gradient_envelope))
                        finite_gradient = bool(np.isfinite(gradient_vector).all())
                        connected = gradients[0] is not None and gradients[1] is not None
                        gradient_nonzero_counts += (gradient_vector != 0.0).astype(np.int64)
                        gradient_finite_count += int(finite_gradient)
                        point_grad = {
                            "objective": "mean((full_output - canonical_rms_baseline) * fixed_unit_rms_sinusoidal_projection)",
                            "objective_value": float(projection.detach().cpu().item()),
                            "fixed_projection_cycles": list(config["gradient_probe"]["cycles"]),
                            "gradient_shape": [3, 3],
                            "gradient_envelope": [2],
                            "all_parameter_tensors_connected": bool(connected),
                            "all_finite": finite_gradient,
                            "nonzero_scalar_count": int(np.count_nonzero(gradient_vector)),
                            "gradient_vector_shape_then_envelope": gradient_vector.tolist(),
                        }
                        with torch.no_grad():
                            # The forward result remains valid as a numeric
                            # engineering probe; no optimizer object is made.
                            pass
                    else:
                        with torch.no_grad():
                            result = head(source_tensor, baseline_tensor)
                    kappa = float(result["dose_scale_kappa"].detach().cpu().item())
                    point_record = {
                        "parameter_coordinates": {
                            "shape_coordinates": point["shape_coordinates"],
                            "envelope_coordinates": point["envelope_coordinates"],
                        },
                        "kappa": kappa,
                        "common_kappa_active": bool(kappa < 1.0),
                        "preprojection_shape_extent_db": float(result["preprojection_shape_extent_db"].detach().cpu().item()),
                        "preprojection_envelope_extent_db": float(result["preprojection_envelope_extent_db"].detach().cpu().item()),
                        "power_compensation_db": _summary(result["power_compensation_db"].detach().cpu().numpy()),
                        "source_weighted_stft_power": _summary(result["stft_source_weighted_coefficient_power"].detach().cpu().numpy()),
                        "masked_weighted_stft_power_before_compensation": _summary(result["stft_masked_weighted_coefficient_power_before_compensation"].detach().cpu().numpy()),
                        "stft_support_frame_count": int(result["stft_power_support_mask"].sum().item()),
                        "first_transition_valid": bool(result["valid_transition_mask"][0].item()),
                        "first_transition_positive_flux": float(result["positive_relative_flux"][0].item()),
                        "first_transition_negative_flux": float(result["negative_relative_flux"][0].item()),
                        "positive_gate": _summary(result["positive_flux_gate"].detach().cpu().numpy()),
                        "negative_gate": _summary(result["negative_flux_gate"].detach().cpu().numpy()),
                        "gradient": point_grad,
                        "cells": {},
                    }
                    source_stft_np = result["source_stft"].detach().cpu().numpy()
                    if source_stft_np.ndim != 2:
                        raise ValueError("head source STFT is not a matrix")
                    stft_frames = int(source_stft_np.shape[1])
                    for cell_id in CELL_IDS:
                        cell = result["cells"][cell_id]
                        metrics, arrays = _cell_metrics(cell, source_np, baseline_np, result, np, torch, config["tolerances"])
                        if point_id == "zero":
                            out_bytes = np.ascontiguousarray(cell["output_pcm_float32"].detach().cpu().numpy(), dtype="<f4").tobytes()
                            metrics["bitwise_equal_to_canonical_numpy_rms_baseline"] = out_bytes == baseline_bytes
                            metrics["max_abs_difference_to_baseline"] = float(np.max(np.abs(np.frombuffer(out_bytes, dtype="<f4") - baseline_np)))
                        else:
                            metrics["bitwise_equal_to_canonical_numpy_rms_baseline"] = None
                            metrics["max_abs_difference_to_baseline"] = float(np.max(np.abs(np.asarray(cell["output_pcm_float32"].detach().cpu().numpy(), dtype=np.float32) - baseline_np)))
                        phase_summary = None
                        if point_id == "zero" and cell_id == "rms_reference":
                            phase_input = cell["constructed_stft"].detach().cpu().numpy()
                            phase_result = phase_clock_reanalysis_diagnostics_v1(
                                source_np,
                                np.asarray(cell["output_pcm_float32"].detach().cpu().numpy(), dtype=np.float32),
                                constructed_output_stft=phase_input,
                                stft=stft_fn,
                                stft_source_sha256=config["source_pins"]["head_v3"]["sha256"],
                                coefficient_transform_source_sha256=config["source_pins"]["head11"]["sha256"],
                                sample_rate_hz=SAMPLE_RATE_HZ,
                                n_fft=int(v3_behavior["n_fft"]),
                                hop_length=int(v3_behavior["hop_length"]),
                                local_xcorr_window_samples=int(config["phase_diagnostics"]["local_xcorr_window_samples"]),
                                local_xcorr_hop_samples=int(config["phase_diagnostics"]["local_xcorr_hop_samples"]),
                                max_lag_samples=int(config["phase_diagnostics"]["max_lag_samples"]),
                            )
                            phase_summary = _phase_summary(phase_result)
                        elif point_id == "zero" or cell_id == "full":
                            constructed_np = cell["constructed_stft"].detach().cpu().numpy()
                            phase_result = phase_clock_reanalysis_diagnostics_v1(
                                source_np,
                                np.asarray(cell["output_pcm_float32"].detach().cpu().numpy(), dtype=np.float32),
                                constructed_output_stft=constructed_np,
                                stft=stft_fn,
                                stft_source_sha256=config["source_pins"]["head_v3"]["sha256"],
                                coefficient_transform_source_sha256=config["source_pins"]["head11"]["sha256"],
                                sample_rate_hz=SAMPLE_RATE_HZ,
                                n_fft=int(v3_behavior["n_fft"]),
                                hop_length=int(v3_behavior["hop_length"]),
                                local_xcorr_window_samples=int(config["phase_diagnostics"]["local_xcorr_window_samples"]),
                                local_xcorr_hop_samples=int(config["phase_diagnostics"]["local_xcorr_hop_samples"]),
                                max_lag_samples=int(config["phase_diagnostics"]["max_lag_samples"]),
                            )
                            phase_summary = _phase_summary(phase_result)
                        metrics["phase_reanalysis_and_transient"] = phase_summary
                        point_record["cells"][cell_id] = metrics
                        if point_id == "zero" and cell_id == "rms_reference":
                            zero_source_power = arrays["source_mean_square"].astype(np.float32)
                            zero_rms_power = arrays["baseline_mean_square"].astype(np.float32)
                        pending_masks[cell_id] = arrays["mask_db"]
                        pending_support[cell_id] = arrays["weighted_stft_support"]
                        pending_ratios[cell_id] = arrays["weighted_stft_ratio"]
                        pending_physical_power[cell_id] = arrays["output_ratio_to_natural"].astype(np.float32)
                        checks = arrays["checks"]
                        hard_checks["all_outputs_exact_length_finite_and_target_rms"] &= bool(
                            checks["exact_sample_count"] and checks["output_finite"] and checks["final_rms_within_tolerance"]
                        )
                        hard_checks["all_constructed_masks_within_3db"] &= bool(checks["mask_within_declared_bound"])
                        hard_checks["all_output_power_diagnostics_finite"] &= bool(checks["weighted_stft_ratio_finite"])
                        if point_id == "zero":
                            hard_checks["all_outputs_exact_length_finite_and_target_rms"] &= bool(metrics["bitwise_equal_to_canonical_numpy_rms_baseline"])
                    point_record["actual_constructed_mask_max_abs_db_across_arms"] = max(
                        point_record["cells"][cell]["mask_db"]["max_abs"] for cell in CELL_IDS
                    )
                    frame_count_40ms = int(pending_physical_power["rms_reference"].size)
                    if any(int(pending_physical_power[cell].size) != frame_count_40ms for cell in CELL_IDS):
                        raise RuntimeError("physical 40 ms power arrays differ in length across arms")
                    if point_id == "zero" and (zero_source_power is None or zero_rms_power is None):
                        raise RuntimeError("zero/RMS reference physical-power arrays were not produced")
                    # Commit arrays and offsets only after all four arms for
                    # this row/point finished. Failed rows remain in JSONL but
                    # cannot corrupt concatenated-array offset semantics.
                    for cell_id in CELL_IDS:
                        mask_frequency_means_by_point[point_id][cell_id].append(
                            np.mean(pending_masks[cell_id].astype(np.float64), axis=1).astype(np.float32)
                        )
                        support_arrays_by_point[point_id][cell_id].append(pending_support[cell_id])
                        stft_arrays_by_point[point_id][cell_id].append(pending_ratios[cell_id])
                        power_arrays_by_point[point_id][cell_id].append(pending_physical_power[cell_id])
                    stft_offsets[point_id].append(stft_offsets[point_id][-1] + stft_frames)
                    stft_sample_ids_by_point[point_id].append(row["sample_id"])
                    power_offsets_by_point[point_id].append(power_offsets_by_point[point_id][-1] + frame_count_40ms)
                    power_sample_ids_by_point[point_id].append(row["sample_id"])
                    row_record["parameter_points"][point_id] = point_record
                    if point_id == "zero":
                        src_power_parts.append(zero_source_power)
                        rms_power_parts.append(zero_rms_power)
                        source_frame_offsets.append(source_frame_offsets[-1] + int(zero_source_power.size))
                        source_frame_sample_ids.append(row["sample_id"])
                row_record["status"] = "PASS" if all(
                    cell.get("exact_sample_count") and cell.get("output_finite")
                    for point in row_record["parameter_points"].values()
                    for cell in point["cells"].values()
                ) else "FAIL"
            except Exception as exc:
                row_record["status"] = "FAIL"
                row_record["error"] = f"{type(exc).__name__}: {exc}"
                hard_checks["all_8_source_rows_retained"] = False
                hard_checks["all_outputs_exact_length_finite_and_target_rms"] = False
            results.append(row_record)
            line = _canonical_json(_json_safe(row_record))
            journal.write(line)
            journal.flush()
            os.fsync(journal.fileno())
    finally:
        journal.close()

    hard_checks["all_11_parameter_gradients_finite_and_connected"] = (
        gradient_finite_count == 8
        and all(
            row.get("parameter_points", {}).get("zero", {}).get("gradient", {}).get("all_parameter_tensors_connected") is True
            and row.get("parameter_points", {}).get("zero", {}).get("gradient", {}).get("all_finite") is True
            for row in results
        )
    )
    hard_checks["every_parameter_has_a_nonzero_observed_gradient"] = bool(np.all(gradient_nonzero_counts > 0))
    hard_checks["no_peak_cap_declared"] = all(
        cell.get("peak_ceiling") is None
        for row in results
        for point in row.get("parameter_points", {}).values()
        for cell in point.get("cells", {}).values()
    )

    matrices: dict[str, Any] = {
        "source_frame_power_40ms_f32": np.concatenate(src_power_parts).astype(np.float32) if src_power_parts else np.zeros((0,), dtype=np.float32),
        "canonical_rms_frame_power_40ms_f32": np.concatenate(rms_power_parts).astype(np.float32) if rms_power_parts else np.zeros((0,), dtype=np.float32),
        "source_frame_offsets_i64": np.asarray(source_frame_offsets, dtype=np.int64),
        "source_frame_sample_ids_u": np.asarray(source_frame_sample_ids, dtype="U64"),
        "gradient_nonzero_count_per_parameter_i64": gradient_nonzero_counts.astype(np.int64),
    }
    stft_offsets_json = {}
    for point_id in POINT_IDS:
        if (
            len(stft_offsets[point_id]) != len(stft_sample_ids_by_point[point_id]) + 1
            or not stft_offsets[point_id]
            or stft_offsets[point_id][0] != 0
            or any(a > b for a, b in zip(stft_offsets[point_id], stft_offsets[point_id][1:]))
        ):
            hard_checks["supporting_array_offsets_valid"] = False
        if (
            len(power_offsets_by_point[point_id]) != len(power_sample_ids_by_point[point_id]) + 1
            or not power_offsets_by_point[point_id]
            or power_offsets_by_point[point_id][0] != 0
            or any(a > b for a, b in zip(power_offsets_by_point[point_id], power_offsets_by_point[point_id][1:]))
        ):
            hard_checks["supporting_array_offsets_valid"] = False
        for cell_id in CELL_IDS:
            if (
                len(mask_frequency_means_by_point[point_id][cell_id]) != len(stft_sample_ids_by_point[point_id])
                or len(support_arrays_by_point[point_id][cell_id]) != len(stft_sample_ids_by_point[point_id])
                or len(stft_arrays_by_point[point_id][cell_id]) != len(stft_sample_ids_by_point[point_id])
                or len(power_arrays_by_point[point_id][cell_id]) != len(power_sample_ids_by_point[point_id])
                or sum(int(array.size) for array in stft_arrays_by_point[point_id][cell_id]) != stft_offsets[point_id][-1]
                or sum(int(array.size) for array in support_arrays_by_point[point_id][cell_id]) != stft_offsets[point_id][-1]
                or sum(int(array.size) for array in power_arrays_by_point[point_id][cell_id]) != power_offsets_by_point[point_id][-1]
            ):
                hard_checks["supporting_array_offsets_valid"] = False
    if (
        len(source_frame_offsets) != len(source_frame_sample_ids) + 1
        or not source_frame_offsets
        or source_frame_offsets[0] != 0
        or any(a > b for a, b in zip(source_frame_offsets, source_frame_offsets[1:]))
    ):
        hard_checks["supporting_array_offsets_valid"] = False
    for point_id in POINT_IDS:
        stft_offsets_json[point_id] = list(stft_offsets[point_id])
        matrices[f"stft_frame_offsets__{point_id}_i64"] = np.asarray(stft_offsets[point_id], dtype=np.int64)
        matrices[f"stft_sample_ids__{point_id}_u"] = np.asarray(stft_sample_ids_by_point[point_id], dtype="U64")
        matrices[f"physical_40ms_offsets__{point_id}_i64"] = np.asarray(power_offsets_by_point[point_id], dtype=np.int64)
        matrices[f"physical_40ms_sample_ids__{point_id}_u"] = np.asarray(power_sample_ids_by_point[point_id], dtype="U64")
        for cell_id in CELL_IDS:
            mask_means = mask_frequency_means_by_point[point_id][cell_id]
            supports = support_arrays_by_point[point_id][cell_id]
            stft_ratios = stft_arrays_by_point[point_id][cell_id]
            powers = power_arrays_by_point[point_id][cell_id]
            if mask_means:
                matrices[f"mask_frequency_mean_db__{point_id}__{cell_id}_f32"] = np.stack(mask_means).astype(np.float32, copy=False)
                matrices[f"weighted_stft_support__{point_id}__{cell_id}_u8"] = np.concatenate(supports).astype(np.uint8, copy=False)
                matrices[f"weighted_stft_ratio__{point_id}__{cell_id}_f32"] = np.concatenate(stft_ratios).astype(np.float32, copy=False)
                matrices[f"physical_40ms_ratio__{point_id}__{cell_id}_f32"] = np.concatenate(powers).astype(np.float32, copy=False)
    uncompressed_bytes = sum(int(np.asarray(value).nbytes) for value in matrices.values())
    if uncompressed_bytes > int(limits["max_uncompressed_array_bytes"]):
        hard_checks["bounded_matrix_bank"] = False
        matrix_bytes: bytes | None = None
    else:
        hard_checks["bounded_matrix_bank"] = True
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **matrices)
        matrix_bytes = buffer.getvalue()
        hard_checks["matrix_archive_within_limit"] = len(matrix_bytes) <= int(limits["max_matrix_archive_bytes"])
    if matrix_bytes is not None and len(matrix_bytes) > int(limits["max_matrix_archive_bytes"]):
        hard_checks["matrix_archive_within_limit"] = False
    elif matrix_bytes is not None:
        _atomic_write(output_dir / "qualification_arrays.npz", matrix_bytes)

    summary = {
        "schema": SCHEMA,
        "status": "COMPLETE" if all(hard_checks.values()) and len(results) == 8 and all(r.get("status") == "PASS" for r in results) else "COMPLETE_WITH_QUALIFICATION_FAILURES",
        "source_cohort": {"sample_count": len(rows), "sample_ids": [row["sample_id"] for row in rows], "speaker_count": len({row["speaker_id"] for row in rows}), "seed": 20261003, "split": "fit"},
        "head_schema": HEAD_SCHEMA,
        "trainable_scalar_count": TRAINABLE_SCALAR_COUNT,
        "coordinate_scale_db": {"shape": head.shape_coordinate_scale_db, "envelope": head.envelope_coordinate_scale_db},
        "probe_points": config["parameter_points"],
        "common_kappa_active_count_by_point": {
            point: sum(bool(row.get("parameter_points", {}).get(point, {}).get("common_kappa_active")) for row in results)
            for point in POINT_IDS
        },
        "gradient_nonzero_count_per_parameter": gradient_nonzero_counts.tolist(),
        "gradient_finite_rows": gradient_finite_count,
        "hard_checks": hard_checks,
        "source_bundle": {"probe_plan_raw_sha256": config["staged_inputs"]["probe_plan"]["sha256"], "archive_manifest_raw_sha256": config["staged_inputs"]["archive_manifest"]["sha256"], "frozen_historical_fit_plan_sha256": EXPECTED_FIT_PLAN_SHA256},
        "runtime": {"python": list(sys.version_info[:3]), "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_initialized": bool(torch.cuda.is_initialized())},
        "resource_measurements": {"elapsed_seconds": time.time() - started, "matrix_uncompressed_bytes": uncompressed_bytes, "matrix_archive_bytes": len(matrix_bytes) if matrix_bytes is not None else None, "output_audio_written": False},
        "stft_frame_offsets": stft_offsets_json,
        "rows": results,
        "scope": config["scope"],
        "interpretation_limit": "Signal-construction/gradient-path engineering evidence only; no generator response, SyncNet effect, portability, or candidate efficacy is measured.",
    }
    summary_bytes = _canonical_json(_json_safe(summary))
    _atomic_write(output_dir / "summary.json", summary_bytes)
    os.replace(journal_path, output_dir / "results.jsonl")
    files = {}
    for name in ("summary.json", "results.jsonl", "qualification_arrays.npz"):
        path = output_dir / name
        if path.is_file():
            size, digest = _sha256_file(path)
            files[name] = {"bytes": size, "sha256": digest}
    result_manifest = {
        "schema": "cross_tfg_local_power_spectral_head_cpu_signal_result_manifest_v1",
        "status": summary["status"],
        "configuration_sha256": config_sha,
        "worker_sha256": config["source_pins"]["worker"]["sha256"],
        "source_pins": config["source_pins"],
        "input_rows": [{"sample_id": row["sample_id"], "wav_file_sha256": row["wav_file_sha256"], "canonical_pcm_float32_sha256": row["canonical_pcm_float32_sha256"], "sample_count": row["sample_count"]} for row in rows],
        "files": files,
        "hard_checks": hard_checks,
        "scope": config["scope"],
    }
    manifest_bytes = _canonical_json(_json_safe(result_manifest))
    _atomic_write(output_dir / "result_manifest.json", manifest_bytes)
    terminal = {
        "schema": "cross_tfg_local_power_spectral_head_cpu_signal_receipt_v1",
        "status": summary["status"],
        "summary": {"path": str(output_dir / "summary.json"), "bytes": len(summary_bytes), "sha256": _sha256_bytes(summary_bytes)},
        "result_manifest": {"path": str(output_dir / "result_manifest.json"), "bytes": len(manifest_bytes), "sha256": _sha256_bytes(manifest_bytes)},
        "planned_rows": 8,
        "completed_rows": len(results),
        "pass_rows": sum(row.get("status") == "PASS" for row in results),
        "failed_rows": sum(row.get("status") != "PASS" for row in results),
        "scope": config["scope"],
    }
    terminal_bytes = _canonical_json(terminal)
    _atomic_write(output_dir / "qualification_receipt.json", terminal_bytes)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = _run(args)
        print(json.dumps(_json_safe(result), sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0 if result.get("status") == "COMPLETE" else 1
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
