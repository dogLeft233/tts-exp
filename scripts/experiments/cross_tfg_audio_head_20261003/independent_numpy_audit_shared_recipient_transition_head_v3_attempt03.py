"""Independent NumPy-only audit of a completed V3 CPU qualification packet.

This consumer deliberately does not import Torch or the producer worker. It
reconstructs the producer's declared signal path from the original local
PCM16 fit WAVs and the saved qualification artifacts, retaining numerical
mismatches as measurements instead of converting them to a success claim.
It is not a model, effect, or candidate-selection evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CONSUMER_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_independent_numpy_consumer_v1"
SUMMARY_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_independent_numpy_audit_summary_v1"
RESULT_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_independent_numpy_audit_row_v1"
MANIFEST_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_independent_numpy_audit_manifest_v1"
HEX = frozenset("0123456789abcdef")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _f32le_sha(values: Any, np: Any) -> str:
    array = np.asarray(values, dtype=np.dtype("<f4"))
    if array.ndim != 1:
        raise ValueError("canonical PCM hash requires a mono vector")
    return _sha_bytes(np.ascontiguousarray(array).tobytes())


def _safe_relative(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return rel


def _repo_file(repo_root: Path, rel_value: str) -> Path:
    rel = _safe_relative(rel_value)
    path = repo_root.joinpath(*rel.parts)
    current = repo_root
    if current.is_symlink():
        raise ValueError("repository root must not be a symlink")
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in pinned repository path: {rel.as_posix()}")
    if not path.is_file():
        raise FileNotFoundError(f"pinned source is absent or symlinked: {rel.as_posix()}")
    return path


def _verify_pins(repo_root: Path, pins: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, pin in pins.items():
        expected_sha = str(pin["sha256"])
        expected_bytes = int(pin["bytes"])
        if len(expected_sha) != 64 or any(char not in HEX for char in expected_sha):
            raise ValueError(f"invalid SHA-256 syntax in source pin {name}")
        path = _repo_file(repo_root, str(pin["path"]))
        sha, size = _sha_file(path)
        verified[name] = {"path": str(pin["path"]), "sha256": sha, "bytes": size}
        if sha != expected_sha or size != expected_bytes:
            raise ValueError(f"source pin mismatch: {name}")
    return verified


def _atomic_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int]:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    return _sha_bytes(payload), len(payload)


def _append_jsonl(path: Path, value: Mapping[str, Any], current_size: int, maximum: int) -> int:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    if current_size + len(payload) > maximum:
        raise MemoryError("independent audit JSONL exceeds its declared byte bound")
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return len(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return value


def _inside_no_symlink(root: Path, relative: str) -> Path:
    rel = _safe_relative(relative)
    if root.is_symlink():
        raise ValueError("declared artifact/input root must not be a symlink")
    candidate = root
    for part in rel.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"symlink in artifact/input path: {rel.as_posix()}")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if os.path.commonpath((str(resolved_root), str(resolved))) != str(resolved_root):
        raise ValueError(f"path escapes its declared root: {rel.as_posix()}")
    if not resolved.is_file():
        raise FileNotFoundError(f"expected regular file: {rel.as_posix()}")
    return resolved


def _wav_pcm16_mono(path: Path, row: Mapping[str, Any], numpy: Any, max_bytes: int) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"input WAV exceeds declared file-size bound: {row['row_id']}")
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"input is not a RIFF/WAVE file: {row['row_id']}")
    declared = int.from_bytes(raw[4:8], "little") + 8
    if declared > len(raw) or declared < 44:
        raise ValueError(f"invalid RIFF declared length: {row['row_id']}")
    format_tuple: tuple[int, int, int, int, int, int] | None = None
    data_parts: list[bytes] = []
    position = 12
    while position + 8 <= declared:
        chunk_id = raw[position : position + 4]
        chunk_size = int.from_bytes(raw[position + 4 : position + 8], "little")
        start = position + 8
        stop = start + chunk_size
        if stop > declared:
            raise ValueError(f"WAV chunk exceeds declared RIFF length: {row['row_id']}")
        chunk = raw[start:stop]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise ValueError(f"short WAV fmt chunk: {row['row_id']}")
            fmt = int.from_bytes(chunk[0:2], "little")
            channels = int.from_bytes(chunk[2:4], "little")
            rate = int.from_bytes(chunk[4:8], "little")
            byte_rate = int.from_bytes(chunk[8:12], "little")
            align = int.from_bytes(chunk[12:14], "little")
            bits = int.from_bytes(chunk[14:16], "little")
            format_tuple = (fmt, channels, rate, byte_rate, align, bits)
        elif chunk_id == b"data":
            data_parts.append(chunk)
        position = stop + (chunk_size & 1)
    if format_tuple is None or not data_parts:
        raise ValueError(f"WAV lacks fmt/data chunks: {row['row_id']}")
    fmt, channels, rate, byte_rate, align, bits = format_tuple
    if (fmt, channels, rate, byte_rate, align, bits) != (1, 1, 16000, 32000, 2, 16):
        raise ValueError(f"WAV is not canonical mono PCM16/16k: {row['row_id']}")
    payload = b"".join(data_parts)
    if len(payload) % 2:
        raise ValueError(f"PCM16 data payload has an odd byte count: {row['row_id']}")
    pcm_i16 = numpy.frombuffer(payload, dtype="<i2")
    pcm = (pcm_i16.astype(numpy.float64) / 32768.0).astype(numpy.float32)
    pcm = numpy.ascontiguousarray(pcm, dtype=numpy.float32)
    file_sha = _sha_bytes(raw)
    payload_sha = _sha_bytes(payload)
    pcm_sha = _f32le_sha(pcm, numpy)
    checks = {
        "raw_file_sha256_matches_manifest": file_sha == row["transfer_file_sha256"],
        "raw_file_bytes_match_manifest": len(raw) == int(row["transfer_file_bytes"]),
        "payload_sha256_matches_source_member": payload_sha == row["source_pcm_payload_sha256"],
        "canonical_float32_sha256_matches_manifest": pcm_sha == row["canonical_float32_pcm_sha256"],
        "sample_count_matches_manifest": int(pcm.size) == int(row["sample_count"]),
        "sample_rate_channels_encoding_match_manifest": (
            rate == int(row["sample_rate_hz"])
            and channels == int(row["channels"])
            and f"PCM_{bits}" == row["source_encoding"]
        ),
        "finite": bool(numpy.isfinite(pcm).all()),
    }
    receipt = {
        "path": row["transfer_relative_path"],
        "raw_file_sha256": file_sha,
        "raw_file_bytes": len(raw),
        "source_pcm_payload_sha256": payload_sha,
        "canonical_float32_pcm_sha256": pcm_sha,
        "sample_count": int(pcm.size),
        "checks": checks,
    }
    if not all(checks.values()):
        raise ValueError(f"original fit PCM failed its manifest binding: {row['row_id']}")
    return pcm, receipt


def _numpy_rms_baseline(samples: Any, target_rms: float, peak_ceiling: float | None, epsilon: float, numpy: Any) -> tuple[Any, dict[str, Any]]:
    source = numpy.asarray(samples, dtype=numpy.float32)
    rms = float(numpy.sqrt(numpy.mean(source.astype(numpy.float64) ** 2, dtype=numpy.float64)))
    peak = float(numpy.max(numpy.abs(source.astype(numpy.float64))))
    if rms <= epsilon:
        output = source.copy()
        gain = 1.0
        peak_limited = False
    else:
        gain = float(target_rms) / rms
        peak_limited = False
        if peak_ceiling is not None and peak > 0.0 and peak * gain > peak_ceiling:
            gain = float(peak_ceiling) / peak
            peak_limited = True
        output = numpy.ascontiguousarray(source * numpy.float32(gain), dtype=numpy.float32)
    return output, {
        "input_rms_f64": rms,
        "input_peak_f64": peak,
        "applied_gain_f64": gain,
        "output_rms_f64": float(numpy.sqrt(numpy.mean(output.astype(numpy.float64) ** 2, dtype=numpy.float64))),
        "output_peak_f64": float(numpy.max(numpy.abs(output.astype(numpy.float64)))),
        "peak_limited": peak_limited,
    }


def _hann_periodic(length: int, numpy: Any) -> Any:
    index = numpy.arange(length, dtype=numpy.float64)
    window = 0.5 - 0.5 * numpy.cos((2.0 * numpy.pi * index) / float(length))
    return numpy.ascontiguousarray(window.astype(numpy.float32))


def _centered_stft(samples: Any, config: Mapping[str, Any], numpy: Any) -> tuple[Any, Any]:
    options = config["head_constructor"]
    n_fft = int(options["n_fft"])
    hop = int(options["hop_length"])
    if (
        not bool(options["stft_center"])
        or options["stft_pad_mode"] != "reflect"
        or options["window_kind"] != "hann"
        or options["window_periodic"] is not True
        or int(options["win_length"]) != n_fft
        or options["stft_normalized"] is not False
        or options["stft_onesided"] is not True
        or options["stft_return_complex"] is not True
    ):
        raise ValueError("independent STFT consumer supports only the pinned centered-reflect contract")
    source = numpy.asarray(samples, dtype=numpy.float32)
    half = n_fft // 2
    if source.size <= half:
        raise ValueError("source is too short for centered reflect padding")
    padded = numpy.pad(source, (half, half), mode="reflect")
    frames = numpy.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    window = _hann_periodic(int(options["win_length"]), numpy)
    windowed = numpy.asarray(frames * window[None, :], dtype=numpy.float32)
    spectrum = numpy.fft.rfft(windowed, n=n_fft, axis=1).astype(numpy.complex64, copy=False).T
    spectrum = numpy.ascontiguousarray(spectrum, dtype=numpy.complex64)
    magnitude = numpy.ascontiguousarray(numpy.abs(spectrum), dtype=numpy.float32)
    return spectrum, magnitude


def _reflect_smooth_1d(values: Any, width: int, numpy: Any) -> Any:
    source = numpy.asarray(values, dtype=numpy.float32)
    if width == 1:
        return source.copy()
    if width <= 0 or width % 2 != 1:
        raise ValueError("smoothing width must be a positive odd integer")
    half = width // 2
    padded = numpy.pad(source, (half, half), mode="reflect")
    result = numpy.empty(source.shape, dtype=numpy.float32)
    for index in range(source.size):
        result[index] = numpy.sum(padded[index : index + width], dtype=numpy.float32) / numpy.float32(width)
    return result


def _reflect_smooth_axis(values: Any, width: int, axis: int, numpy: Any) -> Any:
    source = numpy.asarray(values, dtype=numpy.float32)
    if width == 1:
        return source.copy()
    if width <= 0 or width % 2 != 1:
        raise ValueError("smoothing width must be a positive odd integer")
    half = width // 2
    pad_width = ((half, half), (0, 0)) if axis == 0 else ((0, 0), (half, half))
    padded = numpy.pad(source, pad_width, mode="reflect")
    result = numpy.empty(source.shape, dtype=numpy.float32)
    if axis == 0:
        for index in range(source.shape[0]):
            result[index, :] = numpy.sum(padded[index : index + width, :], axis=0, dtype=numpy.float32) / numpy.float32(width)
    else:
        for index in range(source.shape[1]):
            result[:, index] = numpy.sum(padded[:, index : index + width], axis=1, dtype=numpy.float32) / numpy.float32(width)
    return result


def _band_basis(options: Mapping[str, Any], numpy: Any) -> tuple[Any, Any]:
    n_fft = int(options["n_fft"])
    rate = int(options["sample_rate_hz"])
    frequencies = numpy.arange(n_fft // 2 + 1, dtype=numpy.float32) * numpy.float32(float(rate) / float(n_fft))
    edges = [float(value) for value in options["band_edges_hz"]]
    basis = numpy.zeros((4, frequencies.size), dtype=numpy.float32)
    for band in range(4):
        if band == 3:
            selected = (frequencies >= edges[band]) & (frequencies <= edges[band + 1])
        else:
            selected = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
        basis[band, selected] = numpy.float32(1.0)
    if not numpy.all(basis.sum(axis=0, dtype=numpy.float32) == numpy.float32(1.0)) or numpy.any(basis.sum(axis=1) == 0):
        raise ValueError("declared band edges do not partition all one-sided bins")
    return frequencies, basis


def _torch_linear_quantile(values: Any, quantile: float, numpy: Any) -> Any:
    selected = numpy.asarray(values, dtype=numpy.float32)
    if selected.size == 0:
        return numpy.float32(0.0)
    result = numpy.quantile(selected, float(quantile), method="linear")
    return numpy.float32(result)


def _forward_numpy(source: Any, baseline: Any, coefficients: Any, config: Mapping[str, Any], numpy: Any) -> dict[str, Any]:
    options = config["head_constructor"]
    spectrum, magnitude = _centered_stft(source, config, numpy)
    frames = magnitude.shape[1]
    total = numpy.sum(magnitude, axis=0, dtype=numpy.float32)
    active = total > numpy.float32(options["activity_floor"])
    previous = numpy.zeros(frames, dtype=numpy.float32)
    denominator = numpy.zeros(frames, dtype=numpy.float32)
    positive = numpy.zeros(frames, dtype=numpy.float32)
    negative = numpy.zeros(frames, dtype=numpy.float32)
    valid = numpy.zeros(frames, dtype=numpy.bool_)
    if frames > 1:
        previous[1:] = total[:-1]
        denominator[1:] = numpy.maximum(previous[1:], numpy.float32(options["flux_denominator_floor"]))
        change = numpy.asarray(magnitude[:, 1:] - magnitude[:, :-1], dtype=numpy.float32)
        pos_num = numpy.sum(numpy.maximum(change, numpy.float32(0.0)), axis=0, dtype=numpy.float32)
        neg_num = numpy.sum(numpy.maximum(-change, numpy.float32(0.0)), axis=0, dtype=numpy.float32)
        scale = numpy.float32(options["flux_normalization_scale"])
        positive[1:] = (pos_num / denominator[1:]) / scale
        negative[1:] = (neg_num / denominator[1:]) / scale
        valid[1:] = active[:-1] & active[1:] & (previous[1:] > numpy.float32(options["flux_denominator_floor"]))
        positive[~valid] = numpy.float32(0.0)
        negative[~valid] = numpy.float32(0.0)

    def make_gate(flux: Any, quantile: float) -> tuple[Any, Any, Any]:
        gate = numpy.zeros(frames, dtype=numpy.float32)
        if not numpy.any(valid):
            return gate, numpy.float32(0.0), gate.copy()
        threshold = _torch_linear_quantile(flux[valid], quantile, numpy)
        logits = numpy.asarray((flux - threshold) / numpy.float32(options["gate_temperature"]), dtype=numpy.float32)
        raw = numpy.empty(frames, dtype=numpy.float32)
        positive_logits = logits >= numpy.float32(0.0)
        raw[positive_logits] = numpy.float32(1.0) / (numpy.float32(1.0) + numpy.exp(-logits[positive_logits]))
        exp_logits = numpy.exp(logits[~positive_logits])
        raw[~positive_logits] = exp_logits / (numpy.float32(1.0) + exp_logits)
        raw *= valid.astype(numpy.float32)
        smooth = _reflect_smooth_1d(raw, int(options["gate_smoothing_frames"]), numpy)
        smooth *= valid.astype(numpy.float32)
        return smooth, threshold, raw

    positive_gate, positive_threshold, positive_gate_pre_smoothing = make_gate(positive, float(options["positive_flux_quantile"]))
    negative_gate, negative_threshold, negative_gate_pre_smoothing = make_gate(negative, float(options["negative_flux_quantile"]))
    frequencies, basis = _band_basis(options, numpy)
    coeff = numpy.asarray(coefficients, dtype=numpy.float32)
    if coeff.shape != (3, 3):
        raise ValueError("V3 coefficient matrix must be shape [3,3]")
    ref = int(options["reference_band_index"])
    weights = numpy.insert(coeff, ref, numpy.float32(0.0), axis=1).astype(numpy.float32, copy=False)
    band_index = numpy.argmax(basis, axis=0)
    per_frequency = weights[:, band_index].T
    static_mask = numpy.broadcast_to(per_frequency[:, 0:1], (per_frequency.shape[0], frames)).astype(numpy.float32, copy=False)
    positive_mask = (per_frequency[:, 1:2] * positive_gate[None, :]).astype(numpy.float32, copy=False)
    negative_mask = (per_frequency[:, 2:3] * negative_gate[None, :]).astype(numpy.float32, copy=False)
    raw = (static_mask + positive_mask + negative_mask).astype(numpy.float32, copy=False)
    smooth_frequency = _reflect_smooth_axis(raw, int(options["mask_smoothing_frequency_bins"]), 0, numpy)
    smooth_mask = _reflect_smooth_axis(smooth_frequency, int(options["mask_smoothing_time_frames"]), 1, numpy)
    smooth_positive_mask = _reflect_smooth_axis(
        _reflect_smooth_axis(positive_mask, int(options["mask_smoothing_frequency_bins"]), 0, numpy),
        int(options["mask_smoothing_time_frames"]), 1, numpy,
    )
    smooth_negative_mask = _reflect_smooth_axis(
        _reflect_smooth_axis(negative_mask, int(options["mask_smoothing_frequency_bins"]), 0, numpy),
        int(options["mask_smoothing_time_frames"]), 1, numpy,
    )
    common = numpy.mean(smooth_mask, axis=0, keepdims=True, dtype=numpy.float32)
    if options["frequency_common_mode_policy"] == "remove_per_frame":
        policy = numpy.asarray(smooth_mask - common, dtype=numpy.float32)
    elif options["frequency_common_mode_policy"] == "retain_reference_coded":
        policy = smooth_mask
    else:
        raise ValueError("unsupported frequency common-mode policy")
    max_abs = numpy.float32(numpy.max(numpy.abs(policy)))
    bound = numpy.float32(options["max_abs_mask_db"])
    extent = numpy.float32(max_abs / bound)
    if extent > numpy.float32(0.0):
        safe = numpy.maximum(extent, numpy.finfo(numpy.float32).tiny)
        projection_scale = numpy.float32(numpy.tanh(extent) / safe)
    else:
        projection_scale = numpy.float32(1.0)
    projected = numpy.asarray(policy * projection_scale, dtype=numpy.float32)
    gain = numpy.asarray(numpy.power(numpy.float32(10.0), projected / numpy.float32(20.0)), dtype=numpy.float32)
    delta_spectrum = numpy.asarray(spectrum * (gain - numpy.float32(1.0)), dtype=numpy.complex64)
    delta_pcm = _istft_centered(delta_spectrum, int(numpy.asarray(source).size), options, numpy)
    candidate = numpy.asarray(numpy.asarray(source, dtype=numpy.float32) + delta_pcm, dtype=numpy.float32)
    normalized_candidate, candidate_rms = _torch_rms_anchor(candidate, options, numpy)
    normalized_source, source_rms = _torch_rms_anchor(source, options, numpy)
    canonical_output = numpy.asarray(baseline, dtype=numpy.float32) + (normalized_candidate - normalized_source)
    canonical_output = numpy.ascontiguousarray(canonical_output, dtype=numpy.float32)
    samples_per_video_frame = int(options["sample_rate_hz"]) * int(options["video_fps_den"]) // int(options["video_fps_num"])
    physical = {}
    for name, values in (
        ("source_frame_power_f32", source),
        ("candidate_frame_power_f32", candidate),
        ("baseline_frame_power_f32", baseline),
        ("output_frame_power_f32", canonical_output),
    ):
        count = int(numpy.asarray(values).size) // samples_per_video_frame
        blocks = numpy.asarray(values, dtype=numpy.float32)[: count * samples_per_video_frame].reshape(count, samples_per_video_frame)
        physical[name] = numpy.mean(blocks * blocks, axis=1, dtype=numpy.float32)
    physical["valid_source_frame_power_u8"] = (physical["source_frame_power_f32"] > numpy.float32(options["frame_power_floor"])).astype(numpy.uint8)
    physical["video_frame_centers_source_samples_i64"] = (
        numpy.arange(physical["source_frame_power_f32"].size, dtype=numpy.int64) * samples_per_video_frame
        + samples_per_video_frame // 2
    )
    return {
        "source_stft": spectrum,
        "magnitude": magnitude,
        "frame_magnitude_sum_f32": total,
        "previous_frame_magnitude_sum_f32": previous,
        "flux_denominator_f32": denominator,
        "positive_relative_flux_f32": positive,
        "negative_relative_flux_f32": negative,
        "positive_gate_f32": positive_gate,
        "negative_gate_f32": negative_gate,
        "positive_gate_pre_smoothing_f32": positive_gate_pre_smoothing,
        "negative_gate_pre_smoothing_f32": negative_gate_pre_smoothing,
        "static_mask_db_f32": static_mask,
        "positive_mask_db_f32": positive_mask,
        "negative_mask_db_f32": negative_mask,
        "smoothed_mask_db_f32": smooth_mask,
        "smoothed_positive_mask_db_f32": smooth_positive_mask,
        "smoothed_negative_mask_db_f32": smooth_negative_mask,
        "active_frame_mask_u8": active.astype(numpy.uint8),
        "valid_transition_mask_u8": valid.astype(numpy.uint8),
        "stft_frame_centers_source_samples_i64": numpy.arange(frames, dtype=numpy.int64) * int(options["hop_length"]),
        "positive_threshold_f32": positive_threshold,
        "negative_threshold_f32": negative_threshold,
        "band_basis_f32": basis,
        "band_bin_counts_i64": basis.sum(axis=1, dtype=numpy.float32).astype(numpy.int64),
        "raw_mask_db_f32": raw,
        "smoothed_mask_db_f32": smooth_mask,
        "policy_mask_db_f32": policy,
        "projected_mask_db_f32": projected,
        "magnitude_gain_f32": gain,
        "delta_spectrum": delta_spectrum,
        "delta_pcm_f32": delta_pcm,
        "candidate_before_rms_f32": candidate,
        "normalized_candidate_f32": normalized_candidate,
        "normalized_source_f32": normalized_source,
        "output_pcm_f32": canonical_output,
        "candidate_rms": candidate_rms,
        "source_rms": source_rms,
        "physical": physical,
        "trailing_unframed_sample_count": int(numpy.asarray(source).size) % samples_per_video_frame,
        "flux_balance_identity_max_abs_error": float(
            numpy.max(
                numpy.abs(
                    (positive[valid] - negative[valid])
                    - (
                        (total[1:][valid[1:]] - previous[1:][valid[1:]])
                        / previous[1:][valid[1:]]
                        / numpy.float32(options["flux_normalization_scale"])
                    )
                )
            )
        ) if numpy.any(valid) else None,
        "frequencies_hz_f32": frequencies,
        "weights_db_f32": weights,
        "projection_scale_f32": projection_scale,
    }


def _torch_rms_anchor(samples: Any, options: Mapping[str, Any], numpy: Any) -> tuple[Any, dict[str, Any]]:
    source = numpy.asarray(samples, dtype=numpy.float32)
    epsilon = numpy.float32(options["rms_epsilon"])
    rms_square = numpy.mean(source * source, dtype=numpy.float32)
    safe_rms = numpy.sqrt(numpy.maximum(rms_square, epsilon * epsilon), dtype=numpy.float32)
    active = bool(rms_square > epsilon * epsilon)
    gain = numpy.float32(options["target_rms"]) / safe_rms if active else numpy.float32(1.0)
    scaled = numpy.asarray(source * gain, dtype=numpy.float32)
    peak = numpy.float32(numpy.max(numpy.abs(scaled)))
    ceiling = options["peak_ceiling"]
    if ceiling is None:
        peak_gain = numpy.float32(1.0)
        peak_limited = False
    else:
        peak_limited = bool(peak > numpy.float32(ceiling))
        peak_gain = numpy.float32(ceiling) / numpy.maximum(peak, epsilon) if peak_limited else numpy.float32(1.0)
    output = numpy.asarray(scaled * peak_gain, dtype=numpy.float32)
    output_rms = numpy.sqrt(numpy.mean(output * output, dtype=numpy.float32), dtype=numpy.float32)
    return output, {
        "input_rms_f32": numpy.sqrt(rms_square, dtype=numpy.float32),
        "gain_f32": gain,
        "peak_before_uniform_cap_f32": peak,
        "uniform_peak_gain_f32": peak_gain,
        "peak_limited": peak_limited,
        "output_rms_f32": output_rms,
        "output_peak_f32": numpy.max(numpy.abs(output)),
    }


def _istft_centered(spectrum: Any, output_length: int, options: Mapping[str, Any], numpy: Any) -> Any:
    n_fft = int(options["n_fft"])
    hop = int(options["hop_length"])
    win_length = int(options["win_length"])
    window = _hann_periodic(win_length, numpy)
    frames = numpy.fft.irfft(numpy.asarray(spectrum, dtype=numpy.complex64).T, n=n_fft, axis=1).astype(numpy.float32, copy=False)
    total = (frames.shape[0] - 1) * hop + n_fft
    output = numpy.zeros(total, dtype=numpy.float32)
    envelope = numpy.zeros(total, dtype=numpy.float32)
    window_sq = numpy.asarray(window * window, dtype=numpy.float32)
    for frame_index, frame in enumerate(frames):
        start = frame_index * hop
        output[start : start + n_fft] += numpy.asarray(frame * window, dtype=numpy.float32)
        envelope[start : start + n_fft] += window_sq
    valid = envelope > numpy.finfo(numpy.float32).tiny
    output[valid] /= envelope[valid]
    output[~valid] = numpy.float32(0.0)
    half = n_fft // 2
    centered = output[half : half + output_length]
    if centered.size < output_length:
        centered = numpy.pad(centered, (0, output_length - centered.size), mode="constant")
    return numpy.ascontiguousarray(centered, dtype=numpy.float32)


def _compare_array(actual: Any, expected: Any, *, atol: float, rtol: float, numpy: Any) -> dict[str, Any]:
    left = numpy.asarray(actual)
    right = numpy.asarray(expected)
    shape_equal = left.shape == right.shape
    if not shape_equal:
        return {
            "shape_equal": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
            "nonfinite_actual_count": None,
            "max_abs_error": None,
            "max_relative_error": None,
            "out_of_tolerance_count": None,
            "element_count": None,
            "within_tolerance": False,
        }
    if left.dtype.kind in "biu" or right.dtype.kind in "biu":
        equal = bool(numpy.array_equal(left, right))
        return {
            "shape_equal": True,
            "exact_equal": equal,
            "nonfinite_actual_count": 0,
            "max_abs_error": 0.0 if equal else None,
            "max_relative_error": 0.0 if equal else None,
            "out_of_tolerance_count": 0 if equal else int(left.size),
            "element_count": int(left.size),
            "within_tolerance": equal,
        }
    a = left.astype(numpy.float64, copy=False)
    b = right.astype(numpy.float64, copy=False)
    finite = numpy.isfinite(a) & numpy.isfinite(b)
    errors = numpy.full(a.shape, numpy.inf, dtype=numpy.float64)
    errors[finite] = numpy.abs(a[finite] - b[finite])
    limits = numpy.full(a.shape, float(atol), dtype=numpy.float64)
    limits[finite] += float(rtol) * numpy.abs(b[finite])
    rel = numpy.full(a.shape, numpy.inf, dtype=numpy.float64)
    nonzero = finite & (numpy.abs(b) > 0.0)
    rel[nonzero] = errors[nonzero] / numpy.abs(b[nonzero])
    rel[finite & (b == 0.0)] = numpy.where(errors[finite & (b == 0.0)] == 0.0, 0.0, numpy.inf)
    outside = (~finite) | (errors > limits)
    return {
        "shape_equal": True,
        "nonfinite_actual_count": int((~numpy.isfinite(a)).sum()),
        "nonfinite_expected_count": int((~numpy.isfinite(b)).sum()),
        "max_abs_error": float(errors.max()) if errors.size and numpy.isfinite(errors.max()) else None,
        "max_relative_error": float(rel.max()) if rel.size and numpy.isfinite(rel.max()) else None,
        "out_of_tolerance_count": int(outside.sum()),
        "element_count": int(a.size),
        "atol": float(atol),
        "rtol": float(rtol),
        "within_tolerance": not bool(outside.any()),
    }


def _scalar_compare(actual: float, expected: float, atol: float, rtol: float) -> dict[str, Any]:
    a = float(actual)
    b = float(expected)
    finite = math.isfinite(a) and math.isfinite(b)
    error = abs(a - b) if finite else None
    limit = float(atol) + float(rtol) * abs(b) if finite else None
    return {
        "actual": a if math.isfinite(a) else None,
        "expected": b if math.isfinite(b) else None,
        "both_finite": finite,
        "absolute_error": error,
        "allowed_error": limit,
        "within_tolerance": bool(finite and error <= limit),
    }


def _rank(matrix: Any, tolerance: float, numpy: Any) -> int:
    singular = numpy.linalg.svd(numpy.asarray(matrix, dtype=numpy.float64), compute_uv=False)
    return int(numpy.sum(singular > float(tolerance)))


def _parameter_design_matrix(basis: Any, positive: Any, negative: Any, options: Mapping[str, Any], numpy: Any) -> Any:
    reference = int(options["reference_band_index"])
    nonreference = [band for band in range(basis.shape[0]) if band != reference]
    gates = (numpy.ones_like(positive, dtype=numpy.float32), positive, negative)
    columns = []
    for gate in gates:
        for band in nonreference:
            mask = basis[band, :, None] * gate[None, :]
            mask = _reflect_smooth_axis(mask, int(options["mask_smoothing_frequency_bins"]), 0, numpy)
            mask = _reflect_smooth_axis(mask, int(options["mask_smoothing_time_frames"]), 1, numpy)
            if options["frequency_common_mode_policy"] == "remove_per_frame":
                mask = mask - numpy.mean(mask, axis=0, keepdims=True, dtype=numpy.float32)
            columns.append(mask.astype(numpy.float64, copy=False).reshape(-1))
    return numpy.column_stack(columns)


def _design_spectrum(matrix: Any, tolerance: float, numpy: Any) -> dict[str, Any]:
    value = numpy.asarray(matrix, dtype=numpy.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        return {"rank": 0, "singular_values": [], "condition_number": None, "row_count": int(value.shape[0]) if value.ndim == 2 else 0}
    singular = numpy.linalg.svd(value, compute_uv=False)
    rank = int(numpy.sum(singular > float(tolerance)))
    condition = None
    if singular.size and singular[-1] > float(tolerance):
        condition = float(singular[0] / singular[-1])
    return {
        "rank": rank,
        "singular_values": [float(item) for item in singular],
        "condition_number": condition,
        "row_count": int(value.shape[0]),
        "column_count": int(value.shape[1]),
    }


def _supported_distribution(values: Any, support: Any, numpy: Any) -> dict[str, Any]:
    selected = numpy.asarray(values, dtype=numpy.float64)[numpy.asarray(support, dtype=numpy.bool_)]
    if selected.size == 0:
        return {"supported_count": 0, "minimum": None, "maximum": None, "mean": None, "standard_deviation": None}
    return {
        "supported_count": int(selected.size),
        "minimum": float(numpy.min(selected)),
        "maximum": float(numpy.max(selected)),
        "mean": float(numpy.mean(selected, dtype=numpy.float64)),
        "standard_deviation": float(numpy.std(selected, dtype=numpy.float64)),
    }


def _read_npz(path: Path, expected_keys: set[str], limits: Mapping[str, Any], numpy: Any) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        if len(archive.infolist()) != len(expected_keys):
            raise ValueError("qualification NPZ member count differs from the declared key set")
        expanded = sum(member.file_size for member in archive.infolist())
        if expanded > int(limits["max_uncompressed_matrix_bytes"]):
            raise MemoryError("qualification NPZ exceeds its declared uncompressed bound")
    with numpy.load(path, allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        if actual_keys != expected_keys:
            raise ValueError(f"qualification NPZ key set mismatch: missing={sorted(expected_keys-actual_keys)}, unexpected={sorted(actual_keys-expected_keys)}")
        arrays = {key: archive[key] for key in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError("object arrays are prohibited in qualification NPZ")
    return arrays


def _expected_npz_keys() -> set[str]:
    fit_fields = {
        "raw_mask_db_f32", "projected_mask_db_f32", "positive_relative_flux_f32",
        "negative_relative_flux_f32", "positive_gate_f32", "negative_gate_f32",
        "active_frame_mask_u8", "valid_transition_mask_u8", "frame_magnitude_sum_f32",
        "previous_frame_magnitude_sum_f32", "flux_denominator_f32",
        "stft_frame_centers_source_samples_i64", "source_frame_power_f32",
        "candidate_frame_power_f32", "baseline_frame_power_f32", "output_frame_power_f32",
        "valid_source_frame_power_u8", "video_frame_centers_source_samples_i64",
    }
    synthetic = {
        "synthetic_source_float32", "synthetic_baseline_float32", "synthetic_loss_weight_float32",
        "synthetic_quadratic_weight_float32", "synthetic_raw_mask_db_f32",
        "synthetic_projected_mask_db_f32", "synthetic_positive_flux_f32",
        "synthetic_negative_flux_f32", "synthetic_positive_gate_f32", "synthetic_negative_gate_f32",
        "synthetic_frame_magnitude_sum_f32", "synthetic_previous_frame_magnitude_sum_f32",
        "synthetic_flux_denominator_f32", "synthetic_valid_transition_mask_u8",
        "synthetic_active_frame_mask_u8", "synthetic_parameters_db_f32",
        "synthetic_autograd_matrix_f64", "synthetic_coordinate_fd_f64", "synthetic_direction_matrix_f64",
        "synthetic_band_basis_f32", "synthetic_source_stft_real_f32", "synthetic_source_stft_imag_f32",
    }
    metadata = {
        "fit_row_ids_utf8", "fit_sample_ids_utf8", "fit_speaker_ids_utf8", "fit_canonical_j_i64",
        "fit_stft_frame_offsets_i64", "fit_video_frame_offsets_i64", "fit_raw_mask_db_f32",
        "fit_projected_mask_db_f32", "fit_band_basis_f32", "fit_probe_parameters_db_f32",
        "fit_flux_normalization_scale_f32", "fit_flux_denominator_floor_f32", "fit_activity_floor_f32",
    }
    return metadata | {f"fit_{field}" for field in fit_fields - {"raw_mask_db_f32", "projected_mask_db_f32"}} | synthetic


def _verify_worker_packet(output_root: Path, expected_config_sha: str, expected_input_manifest_sha: str, qualification_config: Mapping[str, Any], numpy: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if output_root.is_symlink():
        raise ValueError("qualification output root must not be a symlink")
    output_root = output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise ValueError("qualification output root must be a nonsymlink directory")
    terminal_path = output_root / "result_manifest.json"
    summary_path = output_root / "summary.json"
    rows_path = output_root / "results.jsonl"
    arrays_path = output_root / "qualification_matrices.npz"
    actual_file_pins = {"results.jsonl": rows_path, "summary.json": summary_path, "qualification_matrices.npz": arrays_path}
    terminal = _read_json(terminal_path)
    if terminal.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_manifest_v1":
        raise ValueError("qualification terminal manifest schema mismatch")
    if terminal.get("status") != "COMPLETE" or terminal.get("qualification_status") != "COMPLETE_ENGINEERING_DIAGNOSTICS":
        raise ValueError("qualification output is not a completed engineering diagnostic packet")
    if terminal.get("configuration_sha256") != expected_config_sha:
        raise ValueError("qualification terminal manifest has the wrong config SHA")
    if terminal.get("source_manifest_sha256") != expected_input_manifest_sha:
        raise ValueError("qualification terminal manifest has the wrong fit40 manifest SHA")
    if terminal.get("completed_fit_rows") != int(qualification_config["source_bundle"]["expected_row_count"]):
        raise ValueError("qualification terminal fit-row count mismatch")
    declared = {entry["path"]: entry for entry in terminal.get("files", [])}
    if len(declared) != len(terminal.get("files", [])) or set(declared) != set(actual_file_pins):
        raise ValueError("qualification terminal file roster mismatch")
    for name, path in actual_file_pins.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"qualification file is absent or symlinked: {name}")
        sha, size = _sha_file(path)
        if sha != declared[name]["sha256"] or size != int(declared[name]["bytes"]):
            raise ValueError(f"qualification output file does not match terminal manifest: {name}")
        if name == "results.jsonl" and size > int(qualification_config["resource_limits"]["max_jsonl_bytes"]):
            raise MemoryError("qualification JSONL exceeds its producer-declared size bound")
        if name == "qualification_matrices.npz" and size > int(qualification_config["resource_limits"]["max_matrix_archive_bytes"]):
            raise MemoryError("qualification NPZ exceeds its producer-declared size bound")
    matrix_entry = declared["qualification_matrices.npz"]
    terminal_sha, terminal_bytes = _sha_file(terminal_path)
    summary = _read_json(summary_path)
    if summary.get("configuration_sha256") != expected_config_sha or summary.get("source_manifest_sha256") != expected_input_manifest_sha:
        raise ValueError("qualification summary lineage mismatch")
    if summary.get("completed_fit_rows") != int(qualification_config["source_bundle"]["expected_row_count"]):
        raise ValueError("qualification summary row count mismatch")
    if summary.get("qualification_status") != terminal.get("qualification_status"):
        raise ValueError("qualification terminal and summary status disagree")
    matrix_serialization = summary.get("matrix_serialization_diagnostics")
    if not isinstance(matrix_serialization, dict):
        raise ValueError("qualification summary lacks matrix serialization diagnostics")
    if (
        summary.get("matrix_archive_sha256") != matrix_entry["sha256"]
        or int(summary.get("matrix_archive_bytes", -1)) != int(matrix_entry["bytes"])
        or matrix_serialization.get("compressed_bytes") != int(matrix_entry["bytes"])
    ):
        raise ValueError("qualification matrix bytes/hash do not match the terminal-bound archive")
    if summary.get("head_schema") != qualification_config["head"]["schema"]:
        raise ValueError("qualification summary head schema mismatch")
    if summary.get("head_constructor") != qualification_config["head_constructor"]:
        raise ValueError("qualification summary constructor contract mismatch")
    if summary.get("scope") != qualification_config["scope"]:
        raise ValueError("qualification summary scope mismatch")
    if summary.get("diagnostic_checks") is None:
        raise ValueError("qualification summary lacks its engineering diagnostic ledger")
    if bool(summary.get("engineering_diagnostic_checks_all_pass")) != bool(terminal.get("engineering_diagnostic_checks_all_pass")):
        raise ValueError("qualification summary/terminal engineering status mismatch")
    records = []
    for line_no, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"qualification JSONL row {line_no} is not an object")
        records.append(record)
    if len(records) != int(qualification_config["source_bundle"]["expected_row_count"]):
        raise ValueError("qualification JSONL row count mismatch")
    arrays = _read_npz(arrays_path, _expected_npz_keys(), qualification_config["resource_limits"], numpy)
    return terminal, summary, {"records": records, "arrays": arrays}, {
        **{name: {"sha256": sha, "bytes": size} for name, (sha, size) in ((key, _sha_file(path)) for key, path in actual_file_pins.items())},
        "result_manifest.json": {"sha256": terminal_sha, "bytes": terminal_bytes},
    }


def _npz_array_check(arrays: Mapping[str, Any], key: str, actual: Any, tolerances: Mapping[str, Any], numpy: Any, tolerance_class: str) -> dict[str, Any]:
    t = tolerances[tolerance_class]
    check = _compare_array(actual, arrays[key], atol=float(t["atol"]), rtol=float(t["rtol"]), numpy=numpy)
    check["matrix_key"] = key
    return check


def _objective(source: Any, baseline: Any, weights: Any, quadratic: Any, coefficient_matrix: Any, config: Mapping[str, Any], numpy: Any) -> tuple[float, dict[str, Any]]:
    result = _forward_numpy(source, baseline, coefficient_matrix, config, numpy)
    residual = numpy.asarray(result["output_pcm_f32"] - baseline, dtype=numpy.float32)
    linear_term = numpy.mean(residual * weights, dtype=numpy.float32)
    quadratic_term = numpy.mean((residual * residual) * quadratic, dtype=numpy.float32)
    scale = numpy.float32(config["synthetic_probe"]["objective"]["loss_scale"])
    value = numpy.float32(scale * numpy.float32(linear_term + quadratic_term))
    return float(value), result


def _verify_synthetic(summary: Mapping[str, Any], matrices: Mapping[str, Any], qualification_config: Mapping[str, Any], tolerances: Mapping[str, Any], numpy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = matrices
    source = numpy.asarray(arrays["synthetic_source_float32"], dtype=numpy.float32)
    target = float(qualification_config["head_constructor"]["target_rms"])
    baseline, baseline_info = _numpy_rms_baseline(
        source,
        target,
        qualification_config["head_constructor"]["peak_ceiling"],
        float(qualification_config["numpy_rms_baseline"]["silence_epsilon"]),
        numpy,
    )
    saved_baseline = numpy.asarray(arrays["synthetic_baseline_float32"], dtype=numpy.float32)
    baseline_check = _compare_array(saved_baseline, baseline, atol=0.0, rtol=0.0, numpy=numpy)
    probe_summary = summary["synthetic_nonuniform_probe"]
    checks: dict[str, Any] = {"synthetic_numpy_baseline_exact_array": baseline_check}
    if _f32le_sha(baseline, numpy) != probe_summary["baseline_pcm_sha256"]:
        checks["synthetic_baseline_hash_matches_worker"] = False
    else:
        checks["synthetic_baseline_hash_matches_worker"] = True

    coefficient = numpy.asarray(arrays["synthetic_parameters_db_f32"], dtype=numpy.float32)
    checks["synthetic_source_is_mono_finite_and_declared_length"] = (
        source.ndim == 1
        and source.size == int(qualification_config["synthetic_probe"]["sample_count"])
        and bool(numpy.isfinite(source).all())
    )
    actual = _forward_numpy(source, baseline, coefficient, qualification_config, numpy)
    scalar_tolerance = tolerances["scalar"]
    for local_key, npz_key, category in (
        ("raw_mask_db_f32", "synthetic_raw_mask_db_f32", "mask"),
        ("projected_mask_db_f32", "synthetic_projected_mask_db_f32", "mask"),
        ("positive_relative_flux_f32", "synthetic_positive_flux_f32", "flux"),
        ("negative_relative_flux_f32", "synthetic_negative_flux_f32", "flux"),
        ("positive_gate_f32", "synthetic_positive_gate_f32", "gate"),
        ("negative_gate_f32", "synthetic_negative_gate_f32", "gate"),
        ("frame_magnitude_sum_f32", "synthetic_frame_magnitude_sum_f32", "magnitude_sum"),
        ("previous_frame_magnitude_sum_f32", "synthetic_previous_frame_magnitude_sum_f32", "magnitude_sum"),
        ("flux_denominator_f32", "synthetic_flux_denominator_f32", "magnitude_sum"),
        ("valid_transition_mask_u8", "synthetic_valid_transition_mask_u8", "exact"),
        ("active_frame_mask_u8", "synthetic_active_frame_mask_u8", "exact"),
        ("band_basis_f32", "synthetic_band_basis_f32", "exact"),
    ):
        if category == "exact":
            checks[f"synthetic_{local_key}"] = _compare_array(actual[local_key], arrays[npz_key], atol=0.0, rtol=0.0, numpy=numpy)
        else:
            checks[f"synthetic_{local_key}"] = _npz_array_check(arrays, npz_key, actual[local_key], tolerances, numpy, category)
    checks["synthetic_fft_real"] = _npz_array_check(arrays, "synthetic_source_stft_real_f32", actual["source_stft"].real.astype(numpy.float32), tolerances, numpy, "fft")
    checks["synthetic_fft_imag"] = _npz_array_check(arrays, "synthetic_source_stft_imag_f32", actual["source_stft"].imag.astype(numpy.float32), tolerances, numpy, "fft")

    weight = numpy.asarray(arrays["synthetic_loss_weight_float32"], dtype=numpy.float32)
    quadratic = numpy.asarray(arrays["synthetic_quadratic_weight_float32"], dtype=numpy.float32)
    recomputed_loss, _ = _objective(source, baseline, weight, quadratic, coefficient, qualification_config, numpy)
    checks["synthetic_objective_matches_worker"] = _scalar_compare(
        recomputed_loss, float(probe_summary["loss"]), float(scalar_tolerance["atol"]), float(scalar_tolerance["rtol"])
    )

    gradient = numpy.asarray(arrays["synthetic_autograd_matrix_f64"], dtype=numpy.float64)
    saved_fd = numpy.asarray(arrays["synthetic_coordinate_fd_f64"], dtype=numpy.float64).reshape(3, 3)
    worker_gradient = numpy.asarray(probe_summary["gradient_matrix"], dtype=numpy.float64)
    checks["synthetic_gradient_matrix_matches_worker_summary"] = _compare_array(
        gradient, worker_gradient, atol=0.0, rtol=0.0, numpy=numpy
    )
    checks["synthetic_gradient_finite_flag_matches_worker"] = bool(
        bool(numpy.isfinite(gradient).all()) == bool(probe_summary["gradient_all_finite"])
    )
    checks["synthetic_coordinate_fd_matrix_matches_worker_summary"] = _compare_array(
        saved_fd,
        numpy.asarray([row["central_finite_difference"] for row in probe_summary["coordinate_finite_difference"]], dtype=numpy.float64).reshape(3, 3),
        atol=0.0,
        rtol=0.0,
        numpy=numpy,
    )
    epsilon = float(qualification_config["synthetic_probe"]["finite_difference"]["coordinate_step_db"])
    computed_fd = numpy.empty((3, 3), dtype=numpy.float64)
    fd_checks = []
    for r in range(3):
        for c in range(3):
            plus = coefficient.copy()
            minus = coefficient.copy()
            plus[r, c] = numpy.float32(plus[r, c] + numpy.float32(epsilon))
            minus[r, c] = numpy.float32(minus[r, c] - numpy.float32(epsilon))
            plus_loss, _ = _objective(source, baseline, weight, quadratic, plus, qualification_config, numpy)
            minus_loss, _ = _objective(source, baseline, weight, quadratic, minus, qualification_config, numpy)
            computed_fd[r, c] = (plus_loss - minus_loss) / (2.0 * epsilon)
            fd_checks.append({
                "row": r,
                "column": c,
                "numpy_central_fd": float(computed_fd[r, c]),
                "worker_central_fd": float(saved_fd[r, c]),
                "worker_autograd": float(gradient[r, c]),
                "numpy_fd_vs_worker_fd": _scalar_compare(
                    float(computed_fd[r, c]), float(saved_fd[r, c]),
                    float(tolerances["finite_difference"]["atol"]),
                    float(tolerances["finite_difference"]["rtol"]),
                ),
                "numpy_fd_vs_worker_autograd": _scalar_compare(
                    float(computed_fd[r, c]), float(gradient[r, c]),
                    float(tolerances["finite_difference"]["atol"]),
                    float(tolerances["finite_difference"]["rtol"]),
                ),
            })

    direction_raw = numpy.asarray(qualification_config["synthetic_probe"]["finite_difference"]["direction_matrix"], dtype=numpy.float64)
    direction = direction_raw / numpy.linalg.norm(direction_raw)
    saved_direction = numpy.asarray(arrays["synthetic_direction_matrix_f64"], dtype=numpy.float64)
    checks["synthetic_direction_matrix_matches_worker_summary"] = _compare_array(
        direction, numpy.asarray(probe_summary["normalized_direction_matrix"], dtype=numpy.float64),
        atol=1e-14, rtol=1e-14, numpy=numpy,
    )
    checks["synthetic_direction_matrix_matches_npz"] = _compare_array(
        saved_direction, direction, atol=1e-14, rtol=1e-14, numpy=numpy
    )
    step_direction = float(qualification_config["synthetic_probe"]["finite_difference"]["directional_step_db"])
    plus_direction = (coefficient.astype(numpy.float64) + step_direction * direction).astype(numpy.float32)
    minus_direction = (coefficient.astype(numpy.float64) - step_direction * direction).astype(numpy.float32)
    plus_loss, _ = _objective(source, baseline, weight, quadratic, plus_direction, qualification_config, numpy)
    minus_loss, _ = _objective(source, baseline, weight, quadratic, minus_direction, qualification_config, numpy)
    directional_fd = (plus_loss - minus_loss) / (2.0 * step_direction)
    directional_autograd = float(numpy.sum(gradient * direction))
    checks["synthetic_directional_autograd_matches_worker_summary"] = _scalar_compare(
        directional_autograd, float(probe_summary["directional_autograd"]),
        float(scalar_tolerance["atol"]), float(scalar_tolerance["rtol"]),
    )
    checks["synthetic_all_coordinate_agreement_flag_matches_recomputed_rows"] = bool(
        bool(probe_summary["all_coordinate_agreement"])
        == all(bool(row["agreement"]) for row in probe_summary["coordinate_finite_difference"])
    )
    synthetic_summary = {
        "baseline_info": baseline_info,
        "checks": checks,
        "recomputed_objective": recomputed_loss,
        "coordinate_finite_difference_checks": fd_checks,
        "coordinate_numpy_fd_matrix_f64": computed_fd.tolist(),
        "worker_autograd_matrix_f64": gradient.tolist(),
        "direction_matrix_matches_config": _compare_array(saved_direction, direction, atol=1e-14, rtol=1e-14, numpy=numpy),
        "directional_numpy_fd": float(directional_fd),
        "directional_worker_autograd": directional_autograd,
        "directional_fd_vs_worker_autograd": _scalar_compare(
            float(directional_fd), directional_autograd,
            float(tolerances["finite_difference"]["atol"]),
            float(tolerances["finite_difference"]["rtol"]),
        ),
    }
    extras = {
        "synthetic_numpy_coordinate_fd_f64": computed_fd,
        "synthetic_numpy_autograd_residual_f64": computed_fd - gradient,
        "synthetic_numpy_direction_f64": direction,
    }
    return synthetic_summary, extras


def _read_results_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {index} is not a JSON object")
            rows.append(value)
    return rows


def _run(args: argparse.Namespace) -> int:
    if "torch" in sys.modules:
        raise RuntimeError("Torch is already loaded; this independent consumer must remain no-Torch")
    import numpy as np

    config_path = Path(args.config).resolve(strict=True)
    repo_root = Path(args.repo_root).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema") != CONSUMER_SCHEMA:
        raise ValueError("independent consumer config schema mismatch")
    config_sha = _sha_bytes(config_bytes)
    expected_config_path = _repo_file(repo_root, config["consumer_config_relative_path"]).resolve(strict=True)
    if config_path != expected_config_path:
        raise ValueError("consumer config is not at its declared repository path")
    consumer_source = config["consumer_source"]
    consumer_source_path = _repo_file(repo_root, consumer_source["path"])
    consumer_source_sha, consumer_source_bytes = _sha_file(consumer_source_path)
    if consumer_source_sha != consumer_source["sha256"] or consumer_source_bytes != int(consumer_source["bytes"]):
        raise ValueError("independent NumPy consumer source pin mismatch")
    if config.get("runtime", {}).get("numpy_version") != np.__version__:
        raise RuntimeError(f"NumPy version mismatch: observed {np.__version__}")
    expected_executable = str(config["runtime"]["python_executable_literal"])
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(expected_executable)):
        raise RuntimeError("literal primary Python executable mismatch")
    if "torch" in sys.modules:
        raise RuntimeError("Torch appeared in sys.modules before consumer imports")

    source_pins = _verify_pins(repo_root, config["source_pins"])
    qualification_config_path = _repo_file(repo_root, config["qualification_config_relative_path"])
    qualification_config_bytes = qualification_config_path.read_bytes()
    qualification_config_sha = _sha_bytes(qualification_config_bytes)
    if qualification_config_sha != config["source_pins"]["qualification_config"]["sha256"]:
        raise ValueError("qualification config raw SHA mismatch")
    qualification_config = json.loads(qualification_config_bytes.decode("utf-8"))
    if np.__version__ != config["runtime"]["numpy_version"]:
        raise RuntimeError("NumPy runtime does not match consumer config")

    input_manifest_rel = qualification_config["source_bundle"]["manifest_relative_path"]
    fit_root = _repo_file(repo_root, config["fit40_source_manifest_relative_path"]).parent
    manifest_path = _inside_no_symlink(fit_root, input_manifest_rel)
    manifest_sha, manifest_bytes = _sha_file(manifest_path)
    expected_manifest_pin = qualification_config["source_bundle"]["manifest_pin"]
    if manifest_sha != expected_manifest_pin["sha256"] or manifest_bytes != int(expected_manifest_pin["bytes"]):
        raise ValueError("original fit40 source manifest raw pin mismatch")
    fit_manifest = _read_json(manifest_path)
    if fit_manifest.get("schema") != qualification_config["source_bundle"]["schema"] or fit_manifest.get("status") != qualification_config["source_bundle"]["required_status"]:
        raise ValueError("original fit40 source manifest schema/status mismatch")
    if any(bool(fit_manifest.get("scope", {}).get(key)) for key in (
        "dev_audio_read", "confirm_audio_read", "model_forward", "render", "score", "candidate_transform_executed"
    )):
        raise ValueError("original fit40 manifest has out-of-scope activity flags")
    source_rows = fit_manifest.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != int(qualification_config["source_bundle"]["expected_row_count"]):
        raise ValueError("original fit40 manifest row count mismatch")
    expected_speakers = set(qualification_config["source_bundle"]["expected_speakers"])
    if {str(row.get("speaker_id")) for row in source_rows} != expected_speakers:
        raise ValueError("original fit40 manifest speaker roster mismatch")
    if len({str(row.get("row_id")) for row in source_rows}) != len(source_rows) or len({str(row.get("sample_id")) for row in source_rows}) != len(source_rows):
        raise ValueError("original fit40 manifest row/sample identifiers are not unique")
    if sum(int(row["canonical_j"]) == 0 for row in source_rows) != len(source_rows) // 2 or sum(int(row["canonical_j"]) == 15 for row in source_rows) != len(source_rows) // 2:
        raise ValueError("original fit40 manifest lacks the exact canonical j=0/j=15 endpoint roster")
    for source_pin_name, manifest_pin_name in (("canonical_audio_decoder", "audio_io"), ("numpy_rms_baseline", "baselines")):
        observed = fit_manifest.get("source_implementation_pins", {}).get(manifest_pin_name, {})
        if observed.get("path") != config["source_pins"][source_pin_name]["path"] or observed.get("raw_sha256") != config["source_pins"][source_pin_name]["sha256"]:
            raise ValueError(f"fit40 source manifest implementation pin mismatch: {manifest_pin_name}")
    qualification_output_root = Path(args.qualification_output_dir)
    if not qualification_output_root.is_absolute():
        qualification_output_root = repo_root / qualification_output_root
    if qualification_output_root.is_symlink():
        raise ValueError("qualification output root must not be a symlink")
    terminal, summary, packet, worker_file_hashes = _verify_worker_packet(
        qualification_output_root,
        qualification_config_sha,
        manifest_sha,
        qualification_config,
        np,
    )
    if terminal.get("scope") != qualification_config["scope"]:
        raise ValueError("qualification terminal scope mismatch")
    source_pin_checks = {}
    for name, expected_pin in qualification_config["source_pins"].items():
        observed_pin = summary.get("source_pins", {}).get(name)
        source_pin_checks[name] = bool(
            isinstance(observed_pin, dict)
            and observed_pin.get("path") == expected_pin["path"]
            and observed_pin.get("sha256") == expected_pin["sha256"]
            and int(observed_pin.get("bytes", -1)) == int(expected_pin["bytes"])
        )
    if not all(source_pin_checks.values()):
        raise ValueError("qualification summary source pins differ from its frozen config")
    expected_worker_runtime = qualification_config["runtime"]
    observed_worker_runtime = summary.get("runtime", {})
    for field in ("python_version", "python_executable_literal", "numpy_version", "torch_version", "torch_intraop_threads", "torch_interop_threads", "cuda_initialized"):
        config_field = "python_executable" if field == "python_executable_literal" else field
        expected_value = False if field == "cuda_initialized" else expected_worker_runtime.get(config_field)
        if observed_worker_runtime.get(field) != expected_value:
            raise ValueError(f"qualification worker runtime mismatch: {field}")
    if observed_worker_runtime.get("cuda_initialized") is not False:
        raise ValueError("qualification worker unexpectedly initialized CUDA")
    worker_records = packet["records"]
    matrix = packet["arrays"]
    tolerances = config["comparison_tolerances"]
    expected_rows = [str(row["row_id"]) for row in source_rows]
    if [str(row.get("row_id")) for row in worker_records] != expected_rows:
        raise ValueError("qualification JSONL row order/identity differs from original fit40 manifest")
    if matrix["fit_row_ids_utf8"].astype(str).tolist() != expected_rows:
        raise ValueError("qualification NPZ row identity/order differs from original fit40 manifest")
    if matrix["fit_sample_ids_utf8"].astype(str).tolist() != [str(row["sample_id"]) for row in source_rows]:
        raise ValueError("qualification NPZ sample identity/order mismatch")
    if matrix["fit_speaker_ids_utf8"].astype(str).tolist() != [str(row["speaker_id"]) for row in source_rows]:
        raise ValueError("qualification NPZ speaker identity/order mismatch")
    if not np.array_equal(matrix["fit_canonical_j_i64"], np.asarray([row["canonical_j"] for row in source_rows], dtype=np.int64)):
        raise ValueError("qualification NPZ canonical endpoint order mismatch")

    options = qualification_config["head_constructor"]
    theta = np.asarray(qualification_config["synthetic_probe"]["diagnostic_parameter_matrix_db"], dtype=np.float32)
    rank_tolerance = float(qualification_config["qualification_checks"]["matrix_rank_tolerance"])
    offsets = np.asarray(matrix["fit_stft_frame_offsets_i64"], dtype=np.int64)
    video_offsets = np.asarray(matrix["fit_video_frame_offsets_i64"], dtype=np.int64)
    if offsets.shape != (len(source_rows) + 1,) or video_offsets.shape != (len(source_rows) + 1,):
        raise ValueError("qualification NPZ row-offset vectors have wrong shape")
    if offsets[0] != 0 or video_offsets[0] != 0 or np.any(np.diff(offsets) <= 0) or np.any(np.diff(video_offsets) < 0):
        raise ValueError("qualification NPZ offsets are not monotone and zero-based")
    if int(offsets[-1]) != int(matrix["fit_raw_mask_db_f32"].shape[1]) or matrix["fit_projected_mask_db_f32"].shape != matrix["fit_raw_mask_db_f32"].shape:
        raise ValueError("qualification NPZ mask offsets do not cover the saved matrix extent")
    video_vector_fields = {
        "fit_source_frame_power_f32", "fit_candidate_frame_power_f32", "fit_baseline_frame_power_f32",
        "fit_output_frame_power_f32", "fit_valid_source_frame_power_u8", "fit_video_frame_centers_source_samples_i64",
    }
    for field in _expected_npz_keys():
        if field.startswith("fit_") and field not in {"fit_row_ids_utf8", "fit_sample_ids_utf8", "fit_speaker_ids_utf8", "fit_canonical_j_i64", "fit_stft_frame_offsets_i64", "fit_video_frame_offsets_i64", "fit_raw_mask_db_f32", "fit_projected_mask_db_f32", "fit_band_basis_f32", "fit_probe_parameters_db_f32", "fit_flux_normalization_scale_f32", "fit_flux_denominator_floor_f32", "fit_activity_floor_f32"}:
            expected_extent = int(video_offsets[-1]) if field in video_vector_fields else int(offsets[-1])
            if matrix[field].ndim != 1 or matrix[field].size != expected_extent:
                raise ValueError(f"qualification NPZ vector extent mismatch: {field}")
    if int(video_offsets[-1]) != int(matrix["fit_source_frame_power_f32"].size):
        raise ValueError("qualification NPZ video offsets do not cover the saved 40 ms arrays")
    constructor = qualification_config["head_constructor"]
    for matrix_key, config_key in (
        ("fit_probe_parameters_db_f32", "diagnostic_parameter_matrix_db"),
        ("fit_flux_normalization_scale_f32", "flux_normalization_scale"),
        ("fit_flux_denominator_floor_f32", "flux_denominator_floor"),
        ("fit_activity_floor_f32", "activity_floor"),
    ):
        expected_value = qualification_config["synthetic_probe"][config_key] if config_key == "diagnostic_parameter_matrix_db" else constructor[config_key]
        expected_array = np.asarray(expected_value, dtype=matrix[matrix_key].dtype)
        if config_key != "diagnostic_parameter_matrix_db":
            # The producer deliberately stores these scalar constructor pins
            # as one-element arrays so every archive entry remains an ndarray.
            expected_array = np.asarray([expected_value], dtype=matrix[matrix_key].dtype)
        if not np.array_equal(np.asarray(matrix[matrix_key]), expected_array):
            raise ValueError(f"qualification matrix constructor pin mismatch: {matrix_key}")

    result_jsonl = qualification_output_root / "results.jsonl"
    audit_dir = Path(args.output_dir)
    if not audit_dir.is_absolute():
        audit_dir = repo_root / audit_dir
    audit_dir = audit_dir.absolute()
    try:
        output_relative = audit_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("independent NumPy output must be placed under the repository root") from exc
    current_output_parent = repo_root
    if current_output_parent.is_symlink():
        raise ValueError("repository root must not be a symlink")
    for part in output_relative.parts[:-1]:
        current_output_parent = current_output_parent / part
        if current_output_parent.is_symlink():
            raise ValueError("independent NumPy output parent path contains a symlink")
    if audit_dir.exists() or audit_dir.is_symlink():
        raise FileExistsError("independent NumPy output directory must be a fresh nonsymlink path")
    audit_dir.mkdir(parents=True, exist_ok=False)
    partial = audit_dir / "audit_results.partial.jsonl"
    fit_magnitude_parts = []
    fit_raw_parts = []
    fit_projected_parts = []
    fit_gate_pre_parts: dict[str, list[Any]] = {"positive": [], "negative": []}
    fit_flux_parts: dict[str, list[Any]] = {
        key: [] for key in (
            "positive_relative_flux_f32", "negative_relative_flux_f32", "positive_gate_f32", "negative_gate_f32",
            "positive_gate_pre_smoothing_f32", "negative_gate_pre_smoothing_f32",
            "active_frame_mask_u8", "valid_transition_mask_u8", "frame_magnitude_sum_f32",
            "previous_frame_magnitude_sum_f32", "flux_denominator_f32", "stft_frame_centers_source_samples_i64",
        )
    }
    fit_power_parts: dict[str, list[Any]] = {
        key: [] for key in (
            "source_frame_power_f32", "candidate_frame_power_f32", "baseline_frame_power_f32",
            "output_frame_power_f32", "valid_source_frame_power_u8", "video_frame_centers_source_samples_i64",
        )
    }
    audit_rows = []
    arrays_max_uncompressed = int(config["resource_limits"]["max_uncompressed_audit_matrix_bytes"])
    audit_array_bytes = 0
    row_comparison_summary = []
    try:
        records_bytes = 0
        for index, (source_row, worker_row) in enumerate(zip(source_rows, worker_records, strict=True)):
            wav = _inside_no_symlink(fit_root, str(source_row["transfer_relative_path"]))
            pcm, source_receipt = _wav_pcm16_mono(wav, source_row, np, int(config["resource_limits"]["max_single_wav_bytes"]))
            baseline, baseline_info = _numpy_rms_baseline(
                pcm,
                float(qualification_config["numpy_rms_baseline"]["target_rms"]),
                qualification_config["numpy_rms_baseline"]["peak_ceiling"],
                float(qualification_config["numpy_rms_baseline"]["silence_epsilon"]),
                np,
            )
            baseline_sha = _f32le_sha(baseline, np)
            zero_record = worker_row["zero_parameter_check"]
            diag_coeff = np.asarray(qualification_config["synthetic_probe"]["diagnostic_parameter_matrix_db"], dtype=np.float32)
            result = _forward_numpy(pcm, baseline, diag_coeff, qualification_config, np)
            frame_start, frame_stop = int(offsets[index]), int(offsets[index + 1])
            video_start, video_stop = int(video_offsets[index]), int(video_offsets[index + 1])
            checks: dict[str, Any] = {
                "source_pcm_canonical_hash_matches": source_receipt["canonical_float32_pcm_sha256"] == source_row["canonical_float32_pcm_sha256"],
                "worker_zero_output_matches_independent_baseline_hash": zero_record["output_pcm_float32_sha256"] == baseline_sha,
                "worker_zero_baseline_hash_matches_independent_baseline_hash": zero_record["baseline_pcm_float32_sha256"] == baseline_sha,
                "worker_zero_output_bitwise_equal_flag": bool(zero_record["bitwise_equal_to_numpy_rms_baseline"]),
                "worker_zero_output_max_abs_difference_exact_zero": float(zero_record["max_abs_difference"]) == 0.0,
                "worker_zero_exact_length_flag": bool(zero_record["exact_sample_count"]) == (pcm.size == int(source_row["sample_count"])),
                "worker_zero_finite_flag": bool(zero_record["finite"]) == bool(np.isfinite(baseline).all()),
                "worker_row_config_sha_matches": worker_row.get("configuration_sha256") == qualification_config_sha,
                "worker_row_source_manifest_sha_matches": worker_row.get("source_manifest_sha256") == manifest_sha,
                "worker_row_identity_matches_input": (
                    worker_row.get("row_id") == source_row["row_id"]
                    and worker_row.get("sample_id") == source_row["sample_id"]
                    and worker_row.get("speaker_id") == source_row["speaker_id"]
                    and int(worker_row.get("canonical_j", -1)) == int(source_row["canonical_j"])
                ),
            }
            worker_source_audio = worker_row["source_audio"]
            checks["worker_source_audio_hashes_match_local_wav"] = (
                worker_source_audio.get("file_sha256") == source_receipt["raw_file_sha256"]
                and int(worker_source_audio.get("file_bytes", -1)) == source_receipt["raw_file_bytes"]
                and worker_source_audio.get("canonical_float32_pcm_sha256") == source_receipt["canonical_float32_pcm_sha256"]
                and int(worker_source_audio.get("sample_count", -1)) == int(source_row["sample_count"])
            )
            row_arrays = {
                "raw_mask_db_f32": ("fit_raw_mask_db_f32", result["raw_mask_db_f32"], "mask"),
                "projected_mask_db_f32": ("fit_projected_mask_db_f32", result["projected_mask_db_f32"], "mask"),
                "positive_relative_flux_f32": ("fit_positive_relative_flux_f32", result["positive_relative_flux_f32"], "flux"),
                "negative_relative_flux_f32": ("fit_negative_relative_flux_f32", result["negative_relative_flux_f32"], "flux"),
                "positive_gate_f32": ("fit_positive_gate_f32", result["positive_gate_f32"], "gate"),
                "negative_gate_f32": ("fit_negative_gate_f32", result["negative_gate_f32"], "gate"),
                "active_frame_mask_u8": ("fit_active_frame_mask_u8", result["active_frame_mask_u8"], "exact"),
                "valid_transition_mask_u8": ("fit_valid_transition_mask_u8", result["valid_transition_mask_u8"], "exact"),
                "frame_magnitude_sum_f32": ("fit_frame_magnitude_sum_f32", result["frame_magnitude_sum_f32"], "magnitude_sum"),
                "previous_frame_magnitude_sum_f32": ("fit_previous_frame_magnitude_sum_f32", result["previous_frame_magnitude_sum_f32"], "magnitude_sum"),
                "flux_denominator_f32": ("fit_flux_denominator_f32", result["flux_denominator_f32"], "magnitude_sum"),
                "stft_frame_centers_source_samples_i64": ("fit_stft_frame_centers_source_samples_i64", result["stft_frame_centers_source_samples_i64"], "exact"),
                "source_frame_power_f32": ("fit_source_frame_power_f32", result["physical"]["source_frame_power_f32"], "power"),
                "candidate_frame_power_f32": ("fit_candidate_frame_power_f32", result["physical"]["candidate_frame_power_f32"], "power"),
                "baseline_frame_power_f32": ("fit_baseline_frame_power_f32", result["physical"]["baseline_frame_power_f32"], "power"),
                "output_frame_power_f32": ("fit_output_frame_power_f32", result["physical"]["output_frame_power_f32"], "power"),
                "valid_source_frame_power_u8": ("fit_valid_source_frame_power_u8", result["physical"]["valid_source_frame_power_u8"], "exact"),
                "video_frame_centers_source_samples_i64": ("fit_video_frame_centers_source_samples_i64", result["physical"]["video_frame_centers_source_samples_i64"], "exact"),
            }
            for local_name, (matrix_key, recomputed, tolerance_class) in row_arrays.items():
                actual_ref = matrix[matrix_key]
                if local_name in {"raw_mask_db_f32", "projected_mask_db_f32"}:
                    actual_ref = actual_ref[:, frame_start:frame_stop]
                else:
                    actual_ref = actual_ref[frame_start:frame_stop] if local_name not in {"source_frame_power_f32", "candidate_frame_power_f32", "baseline_frame_power_f32", "output_frame_power_f32", "valid_source_frame_power_u8", "video_frame_centers_source_samples_i64"} else actual_ref[video_start:video_stop]
                tolerance = config["comparison_tolerances"][tolerance_class]
                comparison = _compare_array(
                    recomputed,
                    actual_ref,
                    atol=float(tolerance.get("atol", 0.0)),
                    rtol=float(tolerance.get("rtol", 0.0)),
                    numpy=np,
                )
                comparison["matrix_key"] = matrix_key
                checks[local_name] = comparison
            expected_frames = 1 + int(pcm.size) // int(options["hop_length"])
            checks["frame_count_matches_centered_stft_length"] = result["magnitude"].shape[1] == expected_frames == frame_stop - frame_start
            checks["source_clock_centers_match_declared_hop"] = np.array_equal(
                result["stft_frame_centers_source_samples_i64"],
                np.arange(expected_frames, dtype=np.int64) * int(options["hop_length"]),
            )
            checks["video_clock_centers_match_declared_40ms_clock"] = np.array_equal(
                result["physical"]["video_frame_centers_source_samples_i64"],
                np.arange(video_stop - video_start, dtype=np.int64) * 640 + 320,
            )
            checks["fit_flux_balance_identity"] = (
                None if result["flux_balance_identity_max_abs_error"] is None else
                result["flux_balance_identity_max_abs_error"] <= float(config["comparison_tolerances"]["flux_identity_atol"])
            )

            worker_diag = worker_row["diagnostic_nonzero_parameter_check"]
            temporal = worker_diag["temporal_diagnostics"]
            valid_support = result["valid_transition_mask_u8"].astype(np.bool_)
            positive_valid = result["positive_gate_f32"][valid_support]
            negative_valid = result["negative_gate_f32"][valid_support]
            positive_raw_valid = result["positive_gate_pre_smoothing_f32"][valid_support]
            negative_raw_valid = result["negative_gate_pre_smoothing_f32"][valid_support]
            smoothed_design = np.column_stack((np.ones(positive_valid.size, dtype=np.float64),
                                               positive_valid.astype(np.float64), negative_valid.astype(np.float64)))
            raw_gate_design = np.column_stack((np.ones(positive_raw_valid.size, dtype=np.float64),
                                               positive_raw_valid.astype(np.float64), negative_raw_valid.astype(np.float64)))
            temporal_rank = _rank(smoothed_design, rank_tolerance, np) if smoothed_design.shape[0] else 0
            temporal_spectrum = _design_spectrum(smoothed_design, rank_tolerance, np)
            raw_temporal_spectrum = _design_spectrum(raw_gate_design, rank_tolerance, np)
            frequency_basis, basis = _band_basis(options, np)
            center_basis = basis.T - np.mean(basis.T, axis=0, keepdims=True, dtype=np.float32)
            raw_rank = _rank(result["raw_mask_db_f32"], rank_tolerance, np)
            projected_rank = _rank(result["projected_mask_db_f32"], rank_tolerance, np)
            parameter_design = _parameter_design_matrix(basis, result["positive_gate_f32"], result["negative_gate_f32"], options, np)
            parameter_spectrum = _design_spectrum(parameter_design, rank_tolerance, np)
            source_supported_cells = (
                (result["magnitude"] > np.float32(options["activity_floor"]))
                & valid_support[None, :]
            )
            supported_parameter_design = parameter_design[source_supported_cells.reshape(-1)]
            supported_parameter_spectrum = _design_spectrum(supported_parameter_design, rank_tolerance, np)
            active_frequency_support = np.any(
                result["magnitude"][:, result["active_frame_mask_u8"].astype(np.bool_)] > np.float32(options["activity_floor"]), axis=1
            ) if np.any(result["active_frame_mask_u8"]) else np.zeros(basis.shape[1], dtype=np.bool_)
            supported_basis = basis[:, active_frequency_support]
            source_supported_band_rank = _rank(supported_basis.T, rank_tolerance, np) if np.any(active_frequency_support) else 0
            component_diagnostics = {}
            for component_name, raw_component, smoothed_component in (
                ("positive", result["positive_mask_db_f32"], result["smoothed_positive_mask_db_f32"]),
                ("negative", result["negative_mask_db_f32"], result["smoothed_negative_mask_db_f32"]),
            ):
                support_component = raw_component[:, valid_support]
                smooth_support_component = smoothed_component[:, valid_support]
                component_diagnostics[component_name] = {
                    "raw_mask_db_on_valid_support": {
                        "element_count": int(support_component.size),
                        "rms_db": float(np.sqrt(np.mean(support_component.astype(np.float64) ** 2, dtype=np.float64))) if support_component.size else None,
                        "mean_abs_db": float(np.mean(np.abs(support_component), dtype=np.float64)) if support_component.size else None,
                    },
                    "after_frequency_and_time_mask_smoothing_on_valid_support": {
                        "element_count": int(smooth_support_component.size),
                        "rms_db": float(np.sqrt(np.mean(smooth_support_component.astype(np.float64) ** 2, dtype=np.float64))) if smooth_support_component.size else None,
                        "mean_abs_db": float(np.mean(np.abs(smooth_support_component), dtype=np.float64)) if smooth_support_component.size else None,
                    },
                }
            flux_identity_error = result["flux_balance_identity_max_abs_error"]
            previous_supported = result["previous_frame_magnitude_sum_f32"][valid_support].astype(np.float64)
            transition_sum_change = (
                result["frame_magnitude_sum_f32"][valid_support].astype(np.float64) - previous_supported
            ) / previous_supported / float(options["flux_normalization_scale"]) if np.any(valid_support) else np.asarray([], dtype=np.float64)
            independent_scalars = {
                "final_output_rms": float(np.sqrt(np.mean(result["output_pcm_f32"] * result["output_pcm_f32"], dtype=np.float32))),
                "final_output_peak": float(np.max(np.abs(result["output_pcm_f32"]))),
                "output_peak_delta_to_baseline": float(np.max(np.abs(result["output_pcm_f32"])) - np.max(np.abs(baseline))),
                "final_output_rms_error_to_target": float(np.sqrt(np.mean(result["output_pcm_f32"] * result["output_pcm_f32"], dtype=np.float32)) - np.float32(options["target_rms"])),
                "max_abs_projected_mask_db": float(np.max(np.abs(result["projected_mask_db_f32"]))),
                "target_rms": float(options["target_rms"]),
                "rms_baseline_applied_gain_f64": baseline_info["applied_gain_f64"],
                "trailing_unframed_sample_count": result["trailing_unframed_sample_count"],
                "flux_positive_minus_negative_identity_max_abs_error": flux_identity_error,
                "flux_positive_minus_negative_equals_relative_total_magnitude_change_max_abs_error": flux_identity_error,
                "supported_relative_total_magnitude_change_mean": float(np.mean(transition_sum_change)) if transition_sum_change.size else None,
                "supported_relative_total_magnitude_change_std": float(np.std(transition_sum_change)) if transition_sum_change.size else None,
            }
            worker_scalar_fields = {
                "final_output_rms": "final_output_rms",
                "final_output_peak": "final_output_peak",
                "output_peak_delta_to_baseline": "output_peak_delta_to_baseline",
                "final_output_rms_error_to_target": "final_output_rms_error_to_target",
                "max_abs_projected_mask_db": "postprojection_max_abs_mask_db",
            }
            for key, worker_key in worker_scalar_fields.items():
                checks[f"scalar_{key}"] = _scalar_compare(
                    independent_scalars[key],
                    float(worker_diag[worker_key]),
                    float(config["comparison_tolerances"]["scalar"]["atol"]),
                    float(config["comparison_tolerances"]["scalar"]["rtol"]),
                )
            checks["temporal_effective_design_rank_matches_worker"] = temporal_rank == int(temporal["temporal_design_rank"])
            checks["temporal_positive_negative_gate_distinct_flag_matches_worker"] = bool(
                float(np.max(np.abs(positive_valid - negative_valid))) > 0.0 if positive_valid.size else False
            ) == bool(temporal["positive_negative_gate_arrays_distinct_on_valid_support"])
            checks["temporal_positive_negative_gate_max_abs_matches_worker"] = _scalar_compare(
                float(np.max(np.abs(positive_valid - negative_valid))) if positive_valid.size else 0.0,
                float(temporal["positive_negative_gate_max_abs_difference_on_valid_support"]),
                float(config["comparison_tolerances"]["scalar"]["atol"]),
                float(config["comparison_tolerances"]["scalar"]["rtol"]),
            )
            checks["temporal_positive_negative_gate_mean_abs_matches_worker"] = _scalar_compare(
                float(np.mean(np.abs(positive_valid - negative_valid), dtype=np.float64)) if positive_valid.size else 0.0,
                float(temporal["positive_negative_gate_mean_abs_difference_on_valid_support"]),
                float(config["comparison_tolerances"]["scalar"]["atol"]),
                float(config["comparison_tolerances"]["scalar"]["rtol"]),
            )
            checks["temporal_first_transition_invalid_matches_worker"] = (
                bool(not result["valid_transition_mask_u8"][0]) == bool(temporal["first_transition_invalid"])
            )
            checks["frequency_effective_ranks"] = {
                "four_band_basis_rank": _rank(basis.T, rank_tolerance, np),
                "per_frame_centered_band_basis_rank": _rank(center_basis, rank_tolerance, np),
                "source_supported_band_basis_rank": source_supported_band_rank,
                "source_supported_bin_count": int(np.sum(active_frequency_support)),
                "source_supported_bin_count_by_band": [int(np.sum((basis[band] > 0) & active_frequency_support)) for band in range(basis.shape[0])],
                "recomputed_raw_mask_rank": raw_rank,
                "recomputed_projected_mask_rank": projected_rank,
                "nine_parameter_preprojection_design_rank": parameter_spectrum["rank"],
                "nine_parameter_preprojection_design_singular_values": parameter_spectrum["singular_values"],
                "nine_parameter_preprojection_design_condition_number": parameter_spectrum["condition_number"],
                "nine_parameter_preprojection_design_scope": "algebraic parameter-mask geometry; not waveform-Jacobian rank",
                "source_supported_nine_parameter_preprojection_design_rank": supported_parameter_spectrum["rank"],
                "source_supported_nine_parameter_preprojection_design_singular_values": supported_parameter_spectrum["singular_values"],
                "source_supported_nine_parameter_preprojection_design_condition_number": supported_parameter_spectrum["condition_number"],
                "source_supported_frequency_frame_cell_count": int(np.sum(source_supported_cells)),
                "source_supported_nine_parameter_preprojection_design_scope": "valid transitions crossed with source-magnitude cells above activity_floor; algebraic mask geometry, not waveform-Jacobian rank",
                "band_bin_counts": [int(value) for value in basis.sum(axis=1)],
            }
            gate_diagnostics = {
                "valid_transition_count": int(np.sum(valid_support)),
                "positive_gate_before_gate_smoothing": _supported_distribution(result["positive_gate_pre_smoothing_f32"], valid_support, np),
                "negative_gate_before_gate_smoothing": _supported_distribution(result["negative_gate_pre_smoothing_f32"], valid_support, np),
                "positive_gate_after_gate_smoothing": _supported_distribution(result["positive_gate_f32"], valid_support, np),
                "negative_gate_after_gate_smoothing": _supported_distribution(result["negative_gate_f32"], valid_support, np),
                "positive_negative_gate_max_abs_difference_after_gate_smoothing": float(np.max(np.abs(positive_valid-negative_valid))) if positive_valid.size else None,
                "positive_negative_gate_mean_abs_difference_after_gate_smoothing": float(np.mean(np.abs(positive_valid-negative_valid), dtype=np.float64)) if positive_valid.size else None,
                "raw_gate_temporal_design": raw_temporal_spectrum,
                "smoothed_gate_temporal_design": temporal_spectrum,
                "positive_negative_mask_components_after_frequency_and_time_smoothing": component_diagnostics,
            }
            checks["frequency_band_basis_matches_worker"] = _compare_array(matrix["fit_band_basis_f32"], basis, atol=0.0, rtol=0.0, numpy=np)
            checks["source_pcm_sample_count_matches_worker"] = int(pcm.size) == int(worker_diag["source_sample_count"]) == int(worker_diag["output_sample_count"])

            record = {
                "schema": RESULT_SCHEMA,
                "ordinal": index,
                "row_id": source_row["row_id"],
                "sample_id": source_row["sample_id"],
                "speaker_id": source_row["speaker_id"],
                "canonical_j": int(source_row["canonical_j"]),
                "source_audio": source_receipt,
                "independent_rms_baseline": baseline_info,
                "independent_scalars": independent_scalars,
                "temporal_design_rank": temporal_rank,
                "temporal_design_singular_values": temporal_spectrum["singular_values"],
                "temporal_design_condition_number": temporal_spectrum["condition_number"],
                "gate_diagnostics": gate_diagnostics,
                "frequency_effective_ranks": checks["frequency_effective_ranks"],
                "flux_identity_interpretation": "positive minus negative relative spectral-magnitude flux equals the relative change in total STFT magnitude on valid transitions (divided by the configured flux scale); it includes local amplitude change and is not an articulation-only statistic",
                "measured_numeric_checks": checks,
                "all_numeric_checks_within_declared_tolerance": all(
                    value.get("within_tolerance", value if isinstance(value, bool) else True)
                    for value in checks.values()
                    if isinstance(value, dict) and "within_tolerance" in value
                ) and all(value is not False for value in checks.values() if isinstance(value, bool)),
            }
            records_bytes = _append_jsonl(
                partial, record, sum((audit_dir / "audit_results.partial.jsonl").stat().st_size for _ in [0]) if partial.exists() else 0,
                int(config["resource_limits"]["max_jsonl_bytes"]),
            )
            audit_rows.append(record)
            row_comparison_summary.append({"row_id": source_row["row_id"], "check_count": len(checks), "checks": checks})

            fit_magnitude_parts.append(result["magnitude"])
            fit_raw_parts.append(result["raw_mask_db_f32"])
            fit_projected_parts.append(result["projected_mask_db_f32"])
            fit_gate_pre_parts["positive"].append(result["positive_gate_pre_smoothing_f32"])
            fit_gate_pre_parts["negative"].append(result["negative_gate_pre_smoothing_f32"])
            audit_array_names = {
                "positive_relative_flux_f32": result["positive_relative_flux_f32"],
                "negative_relative_flux_f32": result["negative_relative_flux_f32"],
                "positive_gate_f32": result["positive_gate_f32"],
                "negative_gate_f32": result["negative_gate_f32"],
                "positive_gate_pre_smoothing_f32": result["positive_gate_pre_smoothing_f32"],
                "negative_gate_pre_smoothing_f32": result["negative_gate_pre_smoothing_f32"],
                "active_frame_mask_u8": result["active_frame_mask_u8"],
                "valid_transition_mask_u8": result["valid_transition_mask_u8"],
                "frame_magnitude_sum_f32": result["frame_magnitude_sum_f32"],
                "previous_frame_magnitude_sum_f32": result["previous_frame_magnitude_sum_f32"],
                "flux_denominator_f32": result["flux_denominator_f32"],
                "stft_frame_centers_source_samples_i64": result["stft_frame_centers_source_samples_i64"],
            }
            for key, value in audit_array_names.items():
                fit_flux_parts[key].append(value)
            for key, value in result["physical"].items():
                fit_power_parts[key].append(value)
            audit_array_bytes += sum(int(np.asarray(value).nbytes) for value in (
                result["magnitude"], result["raw_mask_db_f32"], result["projected_mask_db_f32"],
                *audit_array_names.values(), *result["physical"].values(),
            ))
            if audit_array_bytes > arrays_max_uncompressed:
                raise MemoryError("independent output matrix bank exceeds its declared uncompressed bound")
            del pcm, baseline, result

        synthetic, synthetic_extras = _verify_synthetic(summary, matrix, qualification_config, tolerances, np)
        all_rows_numeric = all(row["all_numeric_checks_within_declared_tolerance"] for row in audit_rows)
        synthetic_checks = [
            value.get("within_tolerance", False) if isinstance(value, dict) else bool(value)
            for value in synthetic["checks"].values()
            if isinstance(value, (dict, bool))
        ]
        synthetic_checks.extend(row[key]["within_tolerance"] for row in synthetic["coordinate_finite_difference_checks"] for key in ("numpy_fd_vs_worker_fd", "numpy_fd_vs_worker_autograd"))
        synthetic_checks.append(synthetic["directional_fd_vs_worker_autograd"]["within_tolerance"])
        synthetic_ok = all(bool(value) for value in synthetic_checks)
        status = "COMPLETE" if all_rows_numeric and synthetic_ok else "COMPLETE_WITH_NUMERIC_MISMATCHES"
        worker_matrix_offsets = np.asarray(matrix["fit_stft_frame_offsets_i64"], dtype=np.int64)
        worker_video_offsets = np.asarray(matrix["fit_video_frame_offsets_i64"], dtype=np.int64)
        independent_arrays = {
            "fit_row_ids_utf8": np.asarray(expected_rows, dtype="U64"),
            "fit_sample_ids_utf8": np.asarray([row["sample_id"] for row in source_rows], dtype="U32"),
            "fit_speaker_ids_utf8": np.asarray([row["speaker_id"] for row in source_rows], dtype="U16"),
            "fit_canonical_j_i64": np.asarray([row["canonical_j"] for row in source_rows], dtype=np.int64),
            "fit_stft_frame_offsets_i64": worker_matrix_offsets,
            "fit_video_frame_offsets_i64": worker_video_offsets,
            "fit_source_magnitude_f32": np.concatenate(fit_magnitude_parts, axis=1).astype(np.float32, copy=False),
            "fit_recomputed_raw_mask_db_f32": np.concatenate(fit_raw_parts, axis=1).astype(np.float32, copy=False),
            "fit_recomputed_projected_mask_db_f32": np.concatenate(fit_projected_parts, axis=1).astype(np.float32, copy=False),
            "fit_recomputed_positive_gate_pre_smoothing_f32": np.concatenate(fit_gate_pre_parts["positive"]).astype(np.float32, copy=False),
            "fit_recomputed_negative_gate_pre_smoothing_f32": np.concatenate(fit_gate_pre_parts["negative"]).astype(np.float32, copy=False),
            "synthetic_numpy_coordinate_fd_f64": synthetic_extras["synthetic_numpy_coordinate_fd_f64"],
            "synthetic_numpy_autograd_residual_f64": synthetic_extras["synthetic_numpy_autograd_residual_f64"],
            "synthetic_numpy_direction_f64": synthetic_extras["synthetic_numpy_direction_f64"],
        }
        for key, parts in fit_flux_parts.items():
            dtype = np.uint8 if key.endswith("_u8") else (np.int64 if key.endswith("_i64") else np.float32)
            independent_arrays[f"fit_recomputed_{key}"] = np.concatenate(parts).astype(dtype, copy=False)
        for key, parts in fit_power_parts.items():
            dtype = np.uint8 if key.endswith("_u8") else (np.int64 if key.endswith("_i64") else np.float32)
            independent_arrays[f"fit_recomputed_{key}"] = np.concatenate(parts).astype(dtype, copy=False)
        independent_uncompressed = sum(int(array.nbytes) for array in independent_arrays.values())
        if independent_uncompressed > arrays_max_uncompressed:
            raise MemoryError("independent NPZ matrix output exceeds its declared uncompressed byte bound")
        matrix_path = audit_dir / "independent_numpy_matrices.npz"
        matrix_buffer = io.BytesIO()
        np.savez_compressed(matrix_buffer, **independent_arrays)
        matrix_payload = matrix_buffer.getvalue()
        if len(matrix_payload) > int(config["resource_limits"]["max_independent_matrix_archive_bytes"]):
            raise MemoryError("independent NPZ matrix output exceeds its declared compressed byte bound")
        with matrix_path.open("xb") as stream:
            stream.write(matrix_payload)
            stream.flush()
            os.fsync(stream.fileno())
        (audit_dir / "audit_results.partial.jsonl").replace(audit_dir / "audit_results.jsonl")
        summary_result = {
            "schema": SUMMARY_SCHEMA,
            "status": status,
            "interpretation": "independent numerical audit only; no candidate or effect inference. Positive-minus-negative relative flux also encodes local total-magnitude change, so it is not an articulation-only measure.",
            "consumer_config_sha256": config_sha,
            "qualification_config_sha256": qualification_config_sha,
            "fit40_manifest_sha256": manifest_sha,
            "qualification_result_manifest_sha256": _sha_file(qualification_output_root / "result_manifest.json")[0],
            "qualification_terminal_status": terminal["status"],
            "qualification_source_file_pins": worker_file_hashes,
            "local_fit40_rows": len(source_rows),
            "completed_independent_rows": len(audit_rows),
            "unique_speakers": len({row["speaker_id"] for row in audit_rows}),
            "rows_all_numeric_checks_within_tolerance": sum(row["all_numeric_checks_within_declared_tolerance"] for row in audit_rows),
            "synthetic_numpy_recomputation": synthetic,
            "comparison_tolerances": tolerances,
            "runtime": {"python_version": list(sys.version_info[:3]), "python_executable": sys.executable, "numpy_version": np.__version__, "torch_imported": "torch" in sys.modules, "platform": platform.platform()},
            "scope": dict(config["scope"]),
            "output_matrices_uncompressed_bytes": independent_uncompressed,
            "output_matrices_bytes": len(matrix_payload),
            "input_waveforms_persisted": False,
            "flux_balance_interpretation": "positive minus negative relative spectral-magnitude flux equals relative total-magnitude change on valid transitions, after division by the configured scale; it therefore includes local amplitude change and is not an articulation-only or shape-only statistic",
            "fit_row_comparisons": row_comparison_summary,
        }
        summary_sha, summary_size = _atomic_json(audit_dir / "summary.json", summary_result)
        output_files = []
        for name in ("audit_results.jsonl", "independent_numpy_matrices.npz", "summary.json"):
            sha, size = _sha_file(audit_dir / name)
            output_files.append({"path": name, "sha256": sha, "bytes": size})
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": status,
            "consumer_config_sha256": config_sha,
            "qualification_config_sha256": qualification_config_sha,
            "fit40_manifest_sha256": manifest_sha,
            "qualification_result_manifest_sha256": summary_result["qualification_result_manifest_sha256"],
            "files": output_files,
            "rows": len(audit_rows),
            "scope": dict(config["scope"]),
        }
        manifest_sha, manifest_size = _atomic_json(audit_dir / "result_manifest.json", manifest)
        print(json.dumps({
            "status": status,
            "audit_dir": str(audit_dir),
            "rows": len(audit_rows),
            "within_tolerance_rows": summary_result["rows_all_numeric_checks_within_tolerance"],
            "summary_sha256": summary_sha,
            "result_manifest_sha256": manifest_sha,
            "result_manifest_bytes": manifest_size,
        }, sort_keys=True))
        return 0 if status == "COMPLETE" else 2
    except Exception as exc:
        failure = {
            "schema": MANIFEST_SCHEMA,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "consumer_config_sha256": config_sha,
            "qualification_config_sha256": qualification_config_sha,
            "fit40_manifest_sha256": manifest_sha,
            "completed_rows": len(audit_rows),
            "scope": dict(config["scope"]),
        }
        _atomic_json(audit_dir / "failure.json", failure)
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--qualification-output-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="must be a fresh audit output directory")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
