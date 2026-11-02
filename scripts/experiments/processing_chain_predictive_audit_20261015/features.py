"""Manifest-bound waveform features for the processing-chain audit.

This module only reads existing audio.  It never invokes TTS, ASR, SyncNet,
video generation, or a learned model.  Inputs and decode settings are bound by
the audit protocol; CSV output is deterministic and cache records are keyed by
both source bytes and the extractor/decoder identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "processing_chain_audit_v1"
SAMPLE_RATE_HZ = 16000
BLOCK_SAMPLES = 160
ABS_SILENCE_RMS = 10.0 ** (-50.0 / 20.0)
PAPER_ABS_THRESHOLD = 0.0005
DECODE_ARGS = [
    "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
    "-threads", "1", "-af", "aresample=resampler=swr:dither_method=none", "-f", "f32le", "pipe:1",
]

METADATA_COLUMNS = [
    "audio_uid", "pair_uid", "source_group", "dataset", "language",
    "condition", "source_layer", "file_sha256", "pcm_sha256", "source_path",
    "missing_reasons_json",
]
FEATURE_COLUMNS = [
    "energy_rms_dbfs", "energy_active_rms_dbfs",
    "silence_leading_s", "silence_trailing_s", "silence_fraction",
    "silence_paper_leading_s", "silence_first30ms_peak",
    "silence_leading_s_relative", "silence_trailing_s_relative",
    "silence_fraction_relative", "energy_active_rms_dbfs_relative",
    "duration_s", "duration_active_s", "duration_active_s_relative",
    "band_ratio_0_250", "band_ratio_250_1000", "band_ratio_1000_2000",
    "band_ratio_2000_3500", "band_ratio_5500_7500",
    "band_log_psd_slope_250_3500",
    "sample_rate_source_hz", "sample_rate_generator_hz",
    "sample_rate_mux_hz", "sample_rate_scorer_hz",
    "encoding_codec", "encoding_container", "encoding_bitrate_bps",
    "encoding_intermediate_codec", "encoding_intermediate_bitrate_bps",
    "timeline_audio_start_s", "timeline_video_start_s",
    "timeline_audio_video_duration_difference_s", "timeline_fps",
    "timeline_audio_sample_count", "timeline_video_frame_count",
    "timeline_audio_time_base_s", "timeline_video_time_base_s",
]
CSV_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


class FeatureInputError(RuntimeError):
    """Raised when a source or protocol binding cannot be honored exactly."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _process_startticks(pid: int) -> str | None:
    stat = Path(f"/proc/{pid}/stat")
    try:
        raw = stat.read_text(encoding="ascii")
        tail = raw.rsplit(")", 1)[1].split()
        return tail[19]  # proc field 22 (starttime), after fields 3..22
    except (OSError, IndexError):
        return None


def _run_text(argv: list[str]) -> str:
    result = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout


def _tool_version(path: str) -> dict[str, str]:
    resolved = shutil.which(path) if not Path(path).is_file() else str(Path(path).resolve())
    if resolved is None:
        raise FeatureInputError(f"required binary not found: {path}")
    first = _run_text([resolved, "-version"]).splitlines()
    if not first:
        raise FeatureInputError(f"binary returned no version line: {resolved}")
    return {"path": resolved, "version_line": first[0]}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rational_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return _float_or_none(value)


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "a:0", "-show_streams",
        "-show_format", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise FeatureInputError(f"ffprobe failed for {path}: {exc}") from exc
    streams = data.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "audio":
        raise FeatureInputError(f"expected exactly the selected first audio stream: {path}")
    return {"stream": streams[0], "format": data.get("format", {})}


def _decode(path: Path, ffmpeg: str) -> tuple[np.ndarray, bytes]:
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), *DECODE_ARGS]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-1200:].decode("utf-8", errors="replace")
        raise FeatureInputError(f"ffmpeg decode failed for {path}: {tail}") from exc
    raw = result.stdout
    if len(raw) % 4:
        raise FeatureInputError(f"decoder returned a non-float32 byte count for {path}: {len(raw)}")
    pcm_bytes = np.frombuffer(raw, dtype="<f4").astype("<f4", copy=False).tobytes(order="C")
    x = np.frombuffer(pcm_bytes, dtype="<f4")
    if not np.isfinite(x).all():
        raise FeatureInputError(f"non-finite decoded sample in {path}")
    return x, pcm_bytes


def _block_view(x: np.ndarray, block: int = BLOCK_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    counts = np.full((x.size + block - 1) // block, block, dtype=np.int64)
    counts[-1] = x.size - block * (counts.size - 1)
    rms = np.empty(counts.size, dtype=np.float64)
    for idx, start in enumerate(range(0, x.size, block)):
        segment = x[start:start + block].astype(np.float64, copy=False)
        rms[idx] = math.sqrt(float(np.mean(segment * segment)))
    return rms, counts


def _silence_features(x: np.ndarray, threshold: float) -> dict[str, float | None]:
    duration = x.size / SAMPLE_RATE_HZ
    if x.size == 0:
        return {"leading": None, "trailing": None, "fraction": None, "active_rms": None, "active_duration": None}
    if threshold == 0.0:
        silent = np.zeros(x.size, dtype=bool)
    block_rms, block_counts = _block_view(x)
    silent_blocks = block_rms <= threshold
    silent = np.repeat(silent_blocks, BLOCK_SAMPLES)[:x.size]
    silent_count = int(silent.sum())
    active_mask = ~silent
    active = x[active_mask].astype(np.float64, copy=False)
    active_rms = math.sqrt(float(np.mean(active * active))) if active.size else 0.0
    if bool(silent_blocks.all()):
        leading = trailing = duration
    else:
        first = int(np.flatnonzero(~silent_blocks)[0])
        last = int(np.flatnonzero(~silent_blocks)[-1])
        leading = min(first * BLOCK_SAMPLES, x.size) / SAMPLE_RATE_HZ
        trailing = (x.size - min((last + 1) * BLOCK_SAMPLES, x.size)) / SAMPLE_RATE_HZ
    return {
        "leading": leading,
        "trailing": trailing,
        "fraction": silent_count / x.size,
        "active_rms": active_rms,
        "active_duration": (x.size - silent_count) / SAMPLE_RATE_HZ,
    }


def _dbfs(rms: float | None) -> float | None:
    return None if rms is None else 20.0 * math.log10(max(float(rms), 1e-12))


def _welch_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if x.size == 0:
        return None
    nperseg = min(400, int(x.size))
    hop = max(1, int(round(nperseg * 160.0 / 400.0)))
    if x.size <= nperseg:
        starts = [0]
    else:
        starts = range(0, int(x.size) - nperseg + 1, hop)
    # Periodic Hann matches the conventional Welch Hann window.  Any final
    # incomplete frame is excluded; clips shorter than 400 samples use one
    # complete segment and hop scales linearly with nperseg.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg, dtype=np.float64) / nperseg)
    scale = SAMPLE_RATE_HZ * float(np.sum(window * window))
    if not math.isfinite(scale) or scale <= 0:
        return None
    periodograms = []
    for start in starts:
        frame = x[start:start + nperseg].astype(np.float64, copy=False)
        frame = frame - float(np.mean(frame))
        power = (np.abs(np.fft.rfft(frame * window, n=512)) ** 2) / scale
        if power.size > 2:
            power[1:-1] *= 2.0
        periodograms.append(power)
    if not periodograms:
        return None
    return np.fft.rfftfreq(512, 1.0 / SAMPLE_RATE_HZ), np.mean(periodograms, axis=0)


def _band_features(x: np.ndarray) -> dict[str, float | None]:
    result = {
        "band_ratio_0_250": None, "band_ratio_250_1000": None,
        "band_ratio_1000_2000": None, "band_ratio_2000_3500": None,
        "band_ratio_5500_7500": None, "band_log_psd_slope_250_3500": None,
    }
    psd = _welch_psd(x)
    if psd is None:
        return result
    freq, density = psd
    df = SAMPLE_RATE_HZ / 512.0

    def integral(low: float, high: float, include_high: bool = False) -> float:
        mask = (freq >= low) & ((freq <= high) if include_high else (freq < high))
        return float(np.sum(density[mask]) * df)

    base = integral(0, 3500)
    high_base = integral(0, 7500, include_high=True)
    bands = [(0, 250), (250, 1000), (1000, 2000), (2000, 3500)]
    names = ["band_ratio_0_250", "band_ratio_250_1000", "band_ratio_1000_2000", "band_ratio_2000_3500"]
    if base > 0:
        for name, (low, high) in zip(names, bands):
            result[name] = integral(low, high) / base
    if high_base > 0:
        result["band_ratio_5500_7500"] = integral(5500, 7500, include_high=True) / high_base
    slope_mask = (freq >= 250) & (freq <= 3500)
    if int(slope_mask.sum()) >= 2:
        fx = np.log10(freq[slope_mask])
        py = np.log10(np.maximum(density[slope_mask], 1e-20))
        result["band_log_psd_slope_250_3500"] = float(np.polyfit(fx, py, 1)[0])
    return result


def _lineage_generator_verified(asset: Mapping[str, Any]) -> bool:
    lineage = asset.get("lineage", {})
    expected = lineage.get("run_input_sha256", lineage.get("canonical_run_input_sha256"))
    verified = lineage.get("archive_copy_body_sha256_verified") is True
    return bool(verified and expected and str(expected).lower() == str(asset.get("file_sha256", "")).lower())


def _feature_values(asset: Mapping[str, Any], pcm: np.ndarray, probe: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    stream = probe["stream"]
    fmt = probe["format"]
    missing: dict[str, str] = {}
    duration = pcm.size / SAMPLE_RATE_HZ
    whole_rms = math.sqrt(float(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else None
    absolute = _silence_features(pcm, ABS_SILENCE_RMS)
    relative_threshold = (0.1 * whole_rms) if whole_rms is not None else 0.0
    relative = _silence_features(pcm, relative_threshold)
    if whole_rms == 0.0:
        relative = {"leading": duration, "trailing": duration, "fraction": 1.0, "active_rms": 0.0, "active_duration": 0.0}
    if pcm.size:
        significant = np.flatnonzero(np.abs(pcm) > PAPER_ABS_THRESHOLD)
        paper_leading = (int(significant[0]) / SAMPLE_RATE_HZ) if significant.size else duration
        first30 = float(np.max(np.abs(pcm[:min(pcm.size, 480)])))
    else:
        paper_leading = first30 = None

    source_rate = _int_or_none(stream.get("sample_rate"))
    if source_rate is None:
        missing["sample_rate_source_hz"] = "ffprobe_sample_rate_unavailable"
    generator_rate = source_rate if _lineage_generator_verified(asset) else None
    if generator_rate is None:
        missing["sample_rate_generator_hz"] = "generator_run_input_rate_not_body_bound"
    missing["sample_rate_mux_hz"] = "no_mux_audio_body_bound_to_this_waveform_asset"
    missing["sample_rate_scorer_hz"] = "scorer_rate_not_measured_by_waveform_probe"

    rate_bps = _int_or_none(stream.get("bit_rate"))
    if rate_bps is None:
        rate_bps = _int_or_none(fmt.get("bit_rate"))
    if rate_bps is None:
        missing["encoding_bitrate_bps"] = "ffprobe_measured_stream_and_container_bitrate_unavailable"
    # The canonical source is the generator waveform itself; no intermediate
    # codec is asserted unless a bound manifest supplies one.
    intermediate_codec = None
    intermediate_bitrate = None
    missing["encoding_intermediate_codec"] = "no_intermediate_encode_stage_body_bound"
    missing["encoding_intermediate_bitrate_bps"] = "no_intermediate_encode_stage_body_bound"

    audio_start = _float_or_none(stream.get("start_time"))
    if audio_start is None:
        audio_start = _float_or_none(fmt.get("start_time"))
    audio_tb = _rational_or_none(stream.get("time_base"))
    if audio_tb is None:
        missing["timeline_audio_time_base_s"] = "ffprobe_time_base_unavailable"
    missing["timeline_video_start_s"] = "no_source_video_stream_bound_to_waveform_asset"
    missing["timeline_audio_video_duration_difference_s"] = "no_source_video_stream_bound_to_waveform_asset"
    missing["timeline_fps"] = "no_source_video_stream_bound_to_waveform_asset"
    missing["timeline_video_frame_count"] = "no_source_video_stream_bound_to_waveform_asset"
    missing["timeline_video_time_base_s"] = "no_source_video_stream_bound_to_waveform_asset"

    values: dict[str, Any] = {
        "energy_rms_dbfs": _dbfs(whole_rms),
        "energy_active_rms_dbfs": _dbfs(absolute["active_rms"]),
        "silence_leading_s": absolute["leading"],
        "silence_trailing_s": absolute["trailing"],
        "silence_fraction": absolute["fraction"],
        "silence_paper_leading_s": paper_leading,
        "silence_first30ms_peak": first30,
        "silence_leading_s_relative": relative["leading"],
        "silence_trailing_s_relative": relative["trailing"],
        "silence_fraction_relative": relative["fraction"],
        "energy_active_rms_dbfs_relative": _dbfs(relative["active_rms"]),
        "duration_s": duration,
        "duration_active_s": absolute["active_duration"],
        "duration_active_s_relative": relative["active_duration"],
        **_band_features(pcm),
        "sample_rate_source_hz": source_rate,
        "sample_rate_generator_hz": generator_rate,
        "sample_rate_mux_hz": None,
        "sample_rate_scorer_hz": None,
        "encoding_codec": stream.get("codec_name"),
        "encoding_container": fmt.get("format_name"),
        "encoding_bitrate_bps": rate_bps,
        "encoding_intermediate_codec": intermediate_codec,
        "encoding_intermediate_bitrate_bps": intermediate_bitrate,
        "timeline_audio_start_s": audio_start,
        "timeline_video_start_s": None,
        "timeline_audio_video_duration_difference_s": None,
        "timeline_fps": None,
        "timeline_audio_sample_count": int(pcm.size),
        "timeline_video_frame_count": None,
        "timeline_audio_time_base_s": audio_tb,
        "timeline_video_time_base_s": None,
    }
    for name, value in values.items():
        if value is None and name not in missing:
            missing[name] = "not_available_from_bound_waveform_or_manifest"
    return values, missing


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureInputError(f"cannot read JSON {path}: {exc}") from exc


def _validate_inputs(protocol_path: Path, assets_path: Path, pair_uids: list[str] | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = _read_json(protocol_path)
    assets_payload = _read_json(assets_path)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise FeatureInputError("protocol schema_version mismatch")
    if protocol.get("cohort_binding", {}).get("status") != "BOUND":
        raise FeatureInputError("protocol cohort binding is not BOUND")
    primary = protocol["cohort_binding"]["primary"]
    if primary.get("source_layer") != "generator":
        raise FeatureInputError("smoke expects the frozen generator-layer cohort")
    assets_sha, _ = _sha256_file(assets_path)
    if assets_sha != primary.get("asset_inventory_sha256"):
        raise FeatureInputError("asset inventory SHA-256 differs from the frozen protocol")
    assets = assets_payload.get("assets")
    if not isinstance(assets, list):
        raise FeatureInputError("asset inventory lacks assets list")
    expected_pairs = list(protocol.get("execution", {}).get("smoke_pair_uids", []))
    if pair_uids is None:
        pair_uids = expected_pairs
    if pair_uids != expected_pairs:
        raise FeatureInputError("requested pair list differs from the protocol's exact smoke_pair_uids")
    if len(pair_uids) != 12 or len(set(pair_uids)) != 12:
        raise FeatureInputError("frozen smoke pair set must contain exactly 12 unique pair_uids")
    chosen = [row for row in assets if row.get("pair_uid") in set(pair_uids)]
    if len(chosen) != 24 or len({row.get("audio_uid") for row in chosen}) != 24:
        raise FeatureInputError(f"expected exactly 24 unique bound audio rows, found {len(chosen)}")
    expected_groups = set(protocol.get("execution", {}).get("smoke_source_groups", []))
    actual_groups = {row.get("source_group") for row in chosen}
    if actual_groups != expected_groups:
        raise FeatureInputError("smoke source-group set differs from frozen protocol")
    for pair in pair_uids:
        arms = [row.get("condition") for row in chosen if row.get("pair_uid") == pair]
        if sorted(arms) != ["natural", "tts"]:
            raise FeatureInputError(f"incomplete N/T pair in smoke: {pair}")
    return protocol, assets_payload, sorted(chosen, key=lambda row: (row["pair_uid"], row["condition"]))


def _asset_row(asset: Mapping[str, Any], path: Path, ffprobe: str, ffmpeg: str, binary_versions: Mapping[str, Any], extractor_sha: str, cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_file_sha, file_bytes = _sha256_file(path)
    if actual_file_sha != str(asset.get("file_sha256", "")).lower():
        raise FeatureInputError(f"source WAV SHA mismatch: {asset.get('audio_uid')}")
    expected_bytes = asset.get("source_file_size_bytes")
    if expected_bytes is not None and file_bytes != int(expected_bytes):
        raise FeatureInputError(f"source WAV size mismatch: {asset.get('audio_uid')}")
    cache_binding = {
        "source_file_sha256": actual_file_sha,
        "asset_context": {
            "audio_uid": asset.get("audio_uid"),
            "source_layer": asset.get("source_layer"),
            "generator_lineage_verified": _lineage_generator_verified(asset),
            "source_media_metadata": asset.get("source_media_metadata", {}),
        },
        "ffmpeg": binary_versions["ffmpeg"],
        "ffprobe": binary_versions["ffprobe"],
        "decode_args": DECODE_ARGS,
        "extractor_sha256": extractor_sha,
        "schema_version": SCHEMA_VERSION,
    }
    cache_key = _sha256_bytes(_json_bytes(cache_binding))
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        cached = _read_json(cache_path)
        if cached.get("cache_binding") != cache_binding:
            raise FeatureInputError(f"feature cache binding mismatch: {cache_key}")
        values = cached["feature_values"]
        missing = cached["missing_reasons"]
        pcm_sha = cached["pcm_sha256"]
        observed_pcm_sha = cached["pcm_sha256"]
        probe_record = cached["probe"]
        reused = True
    else:
        probe_data = _probe(path, ffprobe)
        expected_media = asset.get("source_media_metadata", {})
        observed_rate = _int_or_none(probe_data["stream"].get("sample_rate"))
        observed_channels = _int_or_none(probe_data["stream"].get("channels"))
        if expected_media.get("sample_rate_hz") is not None and observed_rate != int(expected_media["sample_rate_hz"]):
            raise FeatureInputError(f"ffprobe sample rate differs from bound inventory: {asset.get('audio_uid')}")
        if expected_media.get("channels") is not None and observed_channels != int(expected_media["channels"]):
            raise FeatureInputError(f"ffprobe channel count differs from bound inventory: {asset.get('audio_uid')}")
        pcm, pcm_bytes = _decode(path, ffmpeg)
        if pcm.size == 0:
            raise FeatureInputError(f"decoder returned an empty waveform: {asset.get('audio_uid')}")
        expected_count = expected_media.get("sample_count")
        if expected_count is not None and pcm.size != int(expected_count):
            raise FeatureInputError(f"decoded sample count differs from bound inventory: {asset.get('audio_uid')}")
        pcm_sha = _sha256_bytes(pcm_bytes)
        observed_pcm_sha = pcm_sha
        values, missing = _feature_values(asset, pcm, probe_data)
        probe_record = probe_data
        reused = False
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(_json_bytes({
            "schema": "processing_chain_wave_feature_cache_v1",
            "cache_binding": cache_binding,
            "pcm_sha256": pcm_sha,
            "probe": probe_data,
            "feature_values": values,
            "missing_reasons": missing,
        }))
    row = {name: asset.get(name) for name in METADATA_COLUMNS if name not in {"pcm_sha256", "missing_reasons_json"}}
    row.update({
        "file_sha256": actual_file_sha,
        "pcm_sha256": pcm_sha,
        "source_path": str(path),
        "missing_reasons_json": json.dumps(missing, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        **values,
    })
    receipt = {
        "audio_uid": asset.get("audio_uid"),
        "pair_uid": asset.get("pair_uid"),
        "condition": asset.get("condition"),
        "source_path": str(path),
        "file_sha256": actual_file_sha,
        "file_bytes": file_bytes,
        "pcm_sha256": observed_pcm_sha,
        "decoded_sample_count": values.get("timeline_audio_sample_count"),
        "cache_key_sha256": cache_key,
        "cache_reused": reused,
        "cache_path": str(cache_path),
        "ffprobe_stream": probe_record.get("stream", {}),
        "ffprobe_format": probe_record.get("format", {}),
        "missing_reasons": missing,
    }
    return row, receipt


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen waveform-only processing-chain features.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--assets-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Optional reusable feature cache; defaults to OUTPUT/cache")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started_utc = _utc_now()
    started = time.perf_counter()
    protocol_path = args.protocol.resolve(strict=True)
    assets_path = args.assets_json.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FeatureInputError(f"output directory must be fresh: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FeatureInputError("output parent must exist and must not be a symlink")
    protocol, assets_payload, selected = _validate_inputs(protocol_path, assets_path, None)
    ffmpeg_info = _tool_version(args.ffmpeg)
    ffprobe_info = _tool_version(args.ffprobe)
    binary_versions = {"ffmpeg": ffmpeg_info, "ffprobe": ffprobe_info}
    source_file = Path(__file__).resolve(strict=True)
    extractor_sha, extractor_bytes = _sha256_file(source_file)
    output.mkdir()
    cache_dir = args.cache_dir.expanduser().resolve(strict=False) if args.cache_dir else output / "cache"
    cache_dir.mkdir()
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for asset in selected:
        path = Path(str(asset.get("source_path", ""))).expanduser()
        if not path.is_absolute():
            raise FeatureInputError(f"asset source_path must be absolute: {asset.get('audio_uid')}")
        row, per_asset = _asset_row(asset, path, ffprobe_info["path"], ffmpeg_info["path"], binary_versions, extractor_sha, cache_dir)
        rows.append(row)
        receipts.append(per_asset)
    if len(rows) != int(protocol["execution"]["smoke_audio_count"]):
        raise FeatureInputError("extracted row count differs from frozen smoke_audio_count")
    features_path = output / "features.csv"
    _write_csv(features_path, rows)
    csv_sha, csv_bytes = _sha256_file(features_path)
    protocol_sha, protocol_bytes = _sha256_file(protocol_path)
    assets_sha, assets_bytes = _sha256_file(assets_path)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "processing_chain_feature_smoke_receipt_v1",
        "status": "COMPLETE",
        "scope": "24-wave engineering smoke only; no model or score inference",
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "process": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "startticks": _process_startticks(os.getpid()),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "returncode": 0,
        },
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha, "bytes": protocol_bytes},
        "asset_inventory": {"path": str(assets_path), "sha256": assets_sha, "bytes": assets_bytes},
        "source": {"path": str(source_file), "sha256": extractor_sha, "bytes": extractor_bytes},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "ffmpeg": ffmpeg_info,
            "ffprobe": ffprobe_info,
            "decode": {"input_stream": "first audio stream", "sample_rate_hz": SAMPLE_RATE_HZ, "channels": 1, "dtype": "float32 little-endian", "dither_method": "none", "args": DECODE_ARGS},
            "cpu_threads_requested": 1,
            "environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            )},
        },
        "cohort": {
            "cohort_id": protocol["cohort_binding"]["primary"]["cohort_id"],
            "source_groups": protocol["execution"]["smoke_source_groups"],
            "pair_uids": protocol["execution"]["smoke_pair_uids"],
            "audio_count": len(rows),
        },
        "row_receipts": receipts,
        "outputs": {"features_csv": {"path": str(features_path), "sha256": csv_sha, "bytes": csv_bytes}},
        "feature_definitions": {
            "absolute_silence_threshold_rms": ABS_SILENCE_RMS,
            "relative_silence_threshold": "0.1 times whole-clip RMS; zero-RMS clips all silent",
            "silence_block_samples": BLOCK_SAMPLES,
            "paper_probe_abs_threshold": PAPER_ABS_THRESHOLD,
            "welch": {"window": "periodic Hann", "nperseg": "min(400,N)", "hop": "round(nperseg*160/400), minimum 1", "nfft": 512, "scaling": "density", "detrend": "constant per segment", "average": "mean", "tail": "discard incomplete final segment"},
            "band_integral": "sum of PSD bins times df; band bins use [low,high), except final 7500 Hz endpoint included",
            "pcm_sha256": "SHA256 of contiguous decoded float32 little-endian mono16k bytes without a WAV header",
        },
    }
    receipt_path = output / "feature_extraction_receipt.json"
    receipt_path.write_bytes(_json_bytes(receipt))
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "output": str(output), "features_sha256": csv_sha, "receipt": str(receipt_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FeatureInputError as exc:
        print(f"FEATURE_INPUT_ERROR:{exc}", file=sys.stderr)
        raise SystemExit(2)
