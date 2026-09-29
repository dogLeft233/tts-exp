"""Independent validator for the LRS3 bridge experiment.

This module intentionally does not import the producer's bridge, scoring, or
decision functions.  It rechecks hashes, PCM construction, matrix keys, the
primary paired contrast, and the terminal artifact from the files on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import (
    ProtocolError,
    decoded_pcm_sha256,
    file_sha256,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .support import common_support_length


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _read_pcm(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (config.SAMPLE_RATE, 1, 2):
                raise ProtocolError(f"PCM format mismatch: {path}")
            values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"cannot read PCM: {path}") from exc
    if values.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError(f"PCM is too short: {path}")
    return values


def _score_triplet(row: Mapping[str, Any]) -> tuple[float, float, int]:
    score = row.get("score")
    if not isinstance(score, Mapping):
        raise ProtocolError("score row has no score mapping")
    sync_c = float(score["sync_c"])
    sync_d = float(score["sync_d"])
    offset = int(score["av_offset"])
    if not math.isfinite(sync_c) or not math.isfinite(sync_d):
        raise ProtocolError("score contains a non-finite SyncNet value")
    return sync_c, sync_d, offset


def _official_log_triplet(path: Path) -> tuple[float, float, int]:
    if not path.is_file():
        raise ProtocolError(f"official SyncNet log is missing: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s*([-+0-9.eE]+)", content)
    distance = re.search(r"Min dist:\s*([-+0-9.eE]+)", content)
    offset = re.search(r"AV offset:\s*(-?\d+)", content)
    if confidence is None or distance is None or offset is None:
        raise ProtocolError(f"official SyncNet log is incomplete: {path}")
    return float(confidence.group(1)), float(distance.group(1)), int(offset.group(1))


def _score_index(payload: Mapping[str, Any]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            raise ProtocolError("score matrix contains a non-object row")
        key = (str(row["sample_id"]), int(row["render_repeat"]), str(row["cell"]))
        if key in result:
            raise ProtocolError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _score_value(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    sample_id: str,
    repeat: int,
    video_arm: str,
    audio_arm: str,
) -> tuple[float, float, int]:
    key = (sample_id, repeat, f"V_{video_arm}/A_{audio_arm}")
    row = index.get(key)
    if row is None:
        raise ProtocolError(f"missing score cell: {key}")
    return _score_triplet(row)


def _bootstrap_interval(values: list[float], *, lower: float = 0.025, upper: float = 0.975) -> tuple[float, float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    rng = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED))
    samples = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(config.BOOTSTRAP_DRAWS):
        samples[draw] = float(np.mean(array[rng.integers(0, array.size, size=array.size)]))
    return float(np.quantile(samples, lower)), float(np.quantile(samples, upper))


def _interval_inside(values: list[float], lower: float, upper: float) -> bool:
    interval = _bootstrap_interval(values)
    return bool(interval is not None and interval[0] > lower and interval[1] < upper)


def _recompute_primary_rows(score_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = _score_index(score_payload)
    rows: list[dict[str, Any]] = []
    for sample_id in config.EXPECTED_SAMPLE_IDS:
        local_c: list[float] = []
        cloud_c: list[float] = []
        local_d: list[float] = []
        cloud_d: list[float] = []
        source_group: str | None = None
        for repeat in config.REPEATS:
            baseline_c, baseline_d, _ = _score_value(index, sample_id, repeat, "N", "N")
            local_c_value, local_d_value, _ = _score_value(index, sample_id, repeat, "B_LOCAL", "N")
            cloud_c_value, cloud_d_value, _ = _score_value(index, sample_id, repeat, "B_CLOUD", "N")
            local_c.append(local_c_value - baseline_c)
            cloud_c.append(cloud_c_value - baseline_c)
            local_d.append(baseline_d - local_d_value)
            cloud_d.append(baseline_d - cloud_d_value)
            row = index[(sample_id, repeat, "V_N/A_N")]
            row_group = str(row.get("source_group", ""))
            if source_group is None:
                source_group = row_group
            elif source_group != row_group:
                raise ProtocolError(f"source group changed within score rows: {sample_id}")
        g_c_local = float(np.mean(local_c))
        g_c_cloud = float(np.mean(cloud_c))
        g_d_local = float(np.mean(local_d))
        g_d_cloud = float(np.mean(cloud_d))
        rows.append({
            "sample_id": sample_id,
            "source_group": source_group or "",
            "gC_LOCAL": g_c_local,
            "gC_CLOUD": g_c_cloud,
            "deltaC": g_c_cloud - g_c_local,
            "gD_LOCAL": g_d_local,
            "gD_CLOUD": g_d_cloud,
            "deltaD": g_d_cloud - g_d_local,
        })
    return rows


def _recompute_controls_and_terminal(
    score_payload: Mapping[str, Any],
    diagnostics_payload: Mapping[str, Any],
) -> dict[str, Any]:
    index = _score_index(score_payload)
    primary_rows = _recompute_primary_rows(score_payload)
    repeatability_pass = True
    for arm in config.ARMS:
        c_differences: list[float] = []
        d_differences: list[float] = []
        offset_count = 0
        for sample_id in config.EXPECTED_SAMPLE_IDS:
            first = _score_value(index, sample_id, config.REPEATS[0], arm, "N")
            second = _score_value(index, sample_id, config.REPEATS[1], arm, "N")
            c_differences.append(second[0] - first[0])
            d_differences.append(second[1] - first[1])
            offset_count += int(abs(second[2] - first[2]) <= config.OFFSET_TOLERANCE_FRAMES)
        repeatability_pass = repeatability_pass and _interval_inside(c_differences, -0.100, 0.100) and _interval_inside(d_differences, -0.100, 0.100) and offset_count >= config.MIN_OFFSET_AGREEMENT_RECORDS

    b0_c: list[float] = []
    b0_d: list[float] = []
    b0_offset = 0
    damage_c: list[float] = []
    damage_d: list[float] = []
    wrong_audio_count = 0
    for sample_id in config.EXPECTED_SAMPLE_IDS:
        b0_c_record: list[float] = []
        b0_d_record: list[float] = []
        damage_c_record: list[float] = []
        damage_d_record: list[float] = []
        b0_offset_record = True
        for repeat in config.REPEATS:
            baseline = _score_value(index, sample_id, repeat, "N", "N")
            b0 = _score_value(index, sample_id, repeat, "B0", "N")
            reversed_audio = _score_value(index, sample_id, repeat, "N", "N_REV")
            b0_c_record.append(b0[0] - baseline[0])
            b0_d_record.append(baseline[1] - b0[1])
            damage_c_record.append(baseline[0] - reversed_audio[0])
            damage_d_record.append(reversed_audio[1] - baseline[1])
            b0_offset_record = b0_offset_record and abs(b0[2] - baseline[2]) <= config.OFFSET_TOLERANCE_FRAMES
        b0_c.append(float(np.mean(b0_c_record)))
        b0_d.append(float(np.mean(b0_d_record)))
        damage_c_value = float(np.mean(damage_c_record))
        damage_d_value = float(np.mean(damage_d_record))
        damage_c.append(damage_c_value)
        damage_d.append(damage_d_value)
        b0_offset += int(b0_offset_record)
        wrong_audio_count += int(damage_c_value > config.WRONG_AUDIO_DAMAGE_THRESHOLD and damage_d_value > config.WRONG_AUDIO_DAMAGE_THRESHOLD)
    b0_pass = _interval_inside(b0_c, -0.100, 0.100) and _interval_inside(b0_d, -0.100, 0.100) and b0_offset >= config.MIN_OFFSET_AGREEMENT_RECORDS
    damage_c_interval = _bootstrap_interval(damage_c)
    damage_d_interval = _bootstrap_interval(damage_d)
    wrong_audio_pass = bool(
        damage_c_interval is not None
        and damage_d_interval is not None
        and damage_c_interval[0] > config.WRONG_AUDIO_DAMAGE_THRESHOLD
        and damage_d_interval[0] > config.WRONG_AUDIO_DAMAGE_THRESHOLD
        and wrong_audio_count >= config.MIN_WRONG_AUDIO_DAMAGE_RECORDS
    )
    controls_pass = bool(repeatability_pass and b0_pass and wrong_audio_pass)

    diagnostic_rows = diagnostics_payload.get("rows", [])
    movement_pass = True
    movement: dict[str, bool] = {}
    for provider in ("LOCAL", "CLOUD"):
        side_rows = [row.get("arms", {}).get(provider, {}) for row in diagnostic_rows if isinstance(row, Mapping)]
        progress = [float(row["progress"]) for row in side_rows if row.get("progress") is not None and math.isfinite(float(row["progress"]))]
        degenerate_count = sum(bool(row.get("degenerate_direction")) for row in side_rows)
        progress_interval = _bootstrap_interval(progress)
        passed = bool(
            len(progress) == config.EXPECTED_RECORD_COUNT
            and sum(value >= config.MOVEMENT_THRESHOLD for value in progress) >= config.MOVEMENT_MIN_RECORDS
            and progress_interval is not None
            and progress_interval[0] > config.MOVEMENT_THRESHOLD
            and degenerate_count == 0
        )
        movement[provider] = passed
        movement_pass = movement_pass and passed

    primary_delta = [float(row["deltaC"]) for row in primary_rows]
    primary_interval = _bootstrap_interval(primary_delta)
    if primary_interval is not None and primary_interval[0] > 0.0:
        source_comparison = "CLOUD_BRIDGE_STRONGER"
    elif primary_interval is not None and primary_interval[1] < 0.0:
        source_comparison = "LOCAL_BRIDGE_STRONGER"
    else:
        source_comparison = "SOURCE_DIFFERENCE_UNRESOLVED"
    if not controls_pass:
        terminal = "CONTROL_FAILED"
    elif not movement_pass:
        terminal = "BRIDGE_MOVEMENT_FAILED"
    else:
        terminal = source_comparison
    return {
        "primary_rows": primary_rows,
        "primary_mean": float(np.mean(primary_delta)),
        "primary_ci": primary_interval,
        "controls_pass": controls_pass,
        "movement": movement,
        "terminal": terminal,
    }


def _independent_bridge(natural: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    if natural.dtype != np.int16 or target.dtype != np.int16 or natural.ndim != 1 or target.ndim != 1 or natural.size != target.size:
        raise ProtocolError("independent bridge inputs are not equal int16 vectors")
    n = torch.from_numpy(natural.astype(np.float64) / 32768.0)
    t = torch.from_numpy(target.astype(np.float64) / 32768.0)
    window = torch.hann_window(config.WIN_LENGTH, periodic=True, dtype=torch.float64)
    sn = torch.stft(n, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH, window=window, center=True, pad_mode="reflect", return_complex=True)
    st = torch.stft(t, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH, window=window, center=True, pad_mode="reflect", return_complex=True)
    mn = torch.abs(sn).clamp_min(config.MAGNITUDE_FLOOR)
    mt = torch.abs(st).clamp_min(config.MAGNITUDE_FLOOR)
    sb = torch.exp((1.0 - alpha) * torch.log(mn) + alpha * torch.log(mt)) * sn / mn
    waveform = torch.istft(sb, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH, window=window, center=True, length=natural.size).numpy()
    n_float = natural.astype(np.float64) / 32768.0
    waveform *= float(np.sqrt(np.mean(n_float * n_float))) / float(np.sqrt(np.mean(waveform * waveform)))
    peak = float(np.max(np.abs(waveform)))
    if peak >= config.PEAK_LIMIT:
        waveform *= config.PEAK_LIMIT / peak
    scaled = np.rint(waveform * 32768.0)
    if float(scaled.min()) < -32768 or float(scaled.max()) > 32767:
        raise ProtocolError("independent PCM quantization exceeds int16 range")
    return scaled.astype("<i2")


def _check_self(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path.is_file():
        _error(errors, f"missing {label}: {path}")
        return None
    try:
        return verify_self_hashed_json(path)
    except Exception as exc:  # noqa: BLE001 - validator converts malformed artifacts to errors
        _error(errors, f"invalid {label}: {exc}")
        return None


def _validate_protocol(root: Path, errors: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    paths = config.RunPaths(root)
    setup = _check_self(paths.protocol / "setup.json", errors, "setup")
    cohort = _check_self(paths.protocol / "cohort.json", errors, "cohort")
    if setup is None or cohort is None:
        return setup, cohort
    if setup.get("protocol_id") != config.PROTOCOL_ID or cohort.get("protocol_id") != config.PROTOCOL_ID:
        _error(errors, "protocol identity mismatch")
    if setup.get("config", {}).get("input_bindings_sha256") != config.INPUT_BINDINGS_SHA256:
        _error(errors, "setup does not bind the registered input-bindings file")
    records = cohort.get("records")
    ids = [str(row.get("sample_id", "")) for row in records] if isinstance(records, list) else []
    groups = [str(row.get("source_group", "")) for row in records] if isinstance(records, list) else []
    if len(records or []) != config.EXPECTED_RECORD_COUNT or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        _error(errors, "cohort record/source-group count mismatch")
    if ids != list(config.EXPECTED_SAMPLE_IDS) or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        _error(errors, "cohort order or ordered-ID hash mismatch")
    if setup.get("cohort", {}).get("ordered_sample_ids") != ids:
        _error(errors, "setup and cohort ordered IDs differ")
    return setup, cohort


def _validate_tts(root: Path, cohort: Mapping[str, Any], errors: list[str]) -> dict[str, Any] | None:
    payload = _check_self(config.RunPaths(root).tts / "tts_manifest.json", errors, "TTS manifest")
    if payload is None:
        return None
    if payload.get("status") != "complete" or payload.get("record_count") != config.EXPECTED_RECORD_COUNT or payload.get("cloud_new_call_count") != 0:
        _error(errors, "TTS manifest is incomplete or reports a cloud call")
    rows = payload.get("rows", [])
    expected = {str(record["sample_id"]): record for record in cohort.get("records", [])}
    if len(rows) != config.EXPECTED_RECORD_COUNT:
        _error(errors, "TTS paired row count mismatch")
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        record = expected.get(sample_id)
        if record is None:
            _error(errors, f"TTS row has unknown sample: {sample_id}")
            continue
        for provider, model in (("LOCAL", config.LOCAL_MODEL), ("CLOUD", config.CLOUD_MODEL)):
            item = row.get(provider)
            if not isinstance(item, Mapping):
                _error(errors, f"missing TTS provider row: {sample_id}/{provider}")
                continue
            if item.get("provider") != (config.LOCAL_PROVIDER if provider == "LOCAL" else config.CLOUD_PROVIDER) or item.get("model") != model:
                _error(errors, f"TTS provider identity mismatch: {sample_id}/{provider}")
            if item.get("reference_audio_sha256") != record["natural_audio"]["sha256"]:
                _error(errors, f"TTS reference mismatch: {sample_id}/{provider}")
            path = Path(str(item.get("canonical_audio", "")))
            digest = str(item.get("canonical_audio_sha256", ""))
            if not path.is_file() or file_sha256(path) != digest:
                _error(errors, f"TTS canonical hash mismatch: {sample_id}/{provider}")
            else:
                try:
                    values = _read_pcm(path)
                    if item.get("decoded_pcm_sha256") != decoded_pcm_sha256(values):
                        _error(errors, f"TTS decoded PCM hash mismatch: {sample_id}/{provider}")
                except Exception as exc:  # noqa: BLE001 - report malformed asset per provider
                    _error(errors, f"TTS PCM invalid: {sample_id}/{provider}: {exc}")
            if provider == "CLOUD" and item.get("new_cloud_call") is not False:
                _error(errors, f"cloud row is not reuse-only: {sample_id}")
    return payload


def _validate_targets(root: Path, cohort: Mapping[str, Any], errors: list[str]) -> dict[str, Any] | None:
    paths = config.RunPaths(root)
    alignment = _check_self(paths.targets / "alignment_manifest.json", errors, "alignment manifest")
    payload = _check_self(paths.targets / "targets_manifest.json", errors, "target manifest")
    if payload is None:
        return None
    if alignment is None or alignment.get("status") != "complete":
        _error(errors, "alignment manifest is not complete")
    rows = payload.get("rows", [])
    if payload.get("status") != "complete" or payload.get("target_count") != config.EXPECTED_RECORD_COUNT * 2 or len(rows) != config.EXPECTED_RECORD_COUNT * 2:
        _error(errors, "target coverage is incomplete")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("sample_id", "")), str(row.get("provider", "")))
        if key in seen:
            _error(errors, f"duplicate target row: {key}")
        seen.add(key)
        path = Path(str(row.get("target_audio", "")))
        if not path.is_file() or file_sha256(path) != str(row.get("target_audio_sha256", "")):
            _error(errors, f"target file hash mismatch: {key}")
            continue
        try:
            values = _read_pcm(path)
            if values.size != int(row.get("target_samples", -1)) or row.get("decoded_pcm_sha256") != decoded_pcm_sha256(values):
                _error(errors, f"target decoded PCM mismatch: {key}")
            trace = Path(str(row.get("trace", "")))
            if not trace.is_file() or file_sha256(trace) != str(row.get("trace_sha256", "")):
                _error(errors, f"target trace mismatch: {key}")
        except Exception as exc:  # noqa: BLE001 - report malformed target per record
            _error(errors, f"target PCM invalid: {key}: {exc}")
    if seen != {(str(record["sample_id"]), provider) for record in cohort.get("records", []) for provider in ("LOCAL", "CLOUD")}:
        _error(errors, "target IDs/providers do not cover the frozen cohort")
    return payload


def _validate_bridge(root: Path, cohort: Mapping[str, Any], errors: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    paths = config.RunPaths(root)
    manifest = _check_self(paths.bridge / "audio_manifest.json", errors, "bridge manifest")
    geometry = _check_self(paths.bridge / "geometry.json", errors, "support geometry")
    diagnostics = _check_self(paths.bridge / "diagnostics.json", errors, "bridge diagnostics")
    if manifest is None:
        return None, geometry, diagnostics
    if manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT:
        _error(errors, "bridge manifest is incomplete")
        return manifest, geometry, diagnostics
    if diagnostics is None or diagnostics.get("status") != "complete" or diagnostics.get("record_count") != config.EXPECTED_RECORD_COUNT:
        _error(errors, "bridge diagnostics are incomplete")
    target_rows = _check_self(paths.targets / "targets_manifest.json", errors, "target manifest for bridge")
    by_target = {(str(row.get("sample_id")), str(row.get("provider"))): row for row in (target_rows or {}).get("rows", [])}
    for record in cohort.get("records", []):
        sample_id = str(record["sample_id"])
        try:
            natural = _read_pcm(Path(str(record["natural_audio"]["path"])))
            bridge_row = next(row for row in manifest["rows"] if str(row["sample_id"]) == sample_id)
            arms = {str(row["arm"]): row for row in bridge_row.get("arms", [])}
            if set(arms) != set(config.ARMS):
                raise ProtocolError("bridge arm set mismatch")
            n_path = Path(str(arms["N"]["output"]))
            n = _read_pcm(n_path)
            if not np.array_equal(n, natural) or file_sha256(n_path) != file_sha256(Path(str(record["natural_audio"]["path"]))):
                raise ProtocolError("N is not the original PCM/file")
            local = _read_pcm(Path(str(by_target[(sample_id, "LOCAL")]["target_audio"])))
            cloud = _read_pcm(Path(str(by_target[(sample_id, "CLOUD")]["target_audio"])))
            for arm, target, alpha in (("B0", natural, 0.0), ("B_LOCAL", local, config.BRIDGE_ALPHA), ("B_CLOUD", cloud, config.BRIDGE_ALPHA)):
                actual = _read_pcm(Path(str(arms[arm]["output"])))
                expected = _independent_bridge(natural, target, alpha)
                if not np.array_equal(actual, expected):
                    raise ProtocolError(f"independent bridge bytes differ: {arm}")
                if arms[arm].get("decoded_pcm_sha256") != decoded_pcm_sha256(actual):
                    raise ProtocolError(f"bridge decoded hash differs: {arm}")
        except Exception as exc:  # noqa: BLE001 - report independent bridge mismatch per record
            _error(errors, f"bridge reconstruction check failed for {sample_id}: {exc}")
    if geometry is not None:
        if geometry.get("status") != "complete" or geometry.get("record_count") != config.EXPECTED_RECORD_COUNT:
            _error(errors, "common support is incomplete")
        else:
            for row in geometry.get("rows", []):
                try:
                    if int(row["support_frame_start"]) != 0 or int(row["support_frames"]) < config.SYNCNET_MIN_TRACK:
                        raise ProtocolError("support does not start at zero or is too short")
                    expected = common_support_length(int(row["natural_sample_count"]), int(row["source_decoded_frame_count"]))
                    if int(row["support_frames"]) != expected:
                        raise ProtocolError("support length formula mismatch")
                except Exception as exc:  # noqa: BLE001 - report support mismatch per record
                    _error(errors, f"support check failed for {row.get('sample_id')}: {exc}")
    return manifest, geometry, diagnostics


def _validate_videos(root: Path, errors: list[str]) -> dict[str, Any] | None:
    payload = _check_self(config.RunPaths(root).videos / "videos_manifest.json", errors, "video manifest")
    if payload is None:
        return None
    if payload.get("status") != "complete" or payload.get("video_count") != config.EXPECTED_VIDEO_COUNT:
        _error(errors, "video coverage is incomplete")
    seen: set[tuple[str, int, str]] = set()
    invocations: set[str] = set()
    for row in payload.get("rows", []):
        key = (str(row.get("sample_id")), int(row.get("render_repeat", -1)), str(row.get("arm")))
        if key in seen:
            _error(errors, f"duplicate video identity: {key}")
        seen.add(key)
        invocation = str(row.get("invocation_id", ""))
        if not invocation or invocation in invocations:
            _error(errors, f"video invocation is missing or reused: {key}")
        invocations.add(invocation)
        output = Path(str(row.get("output", "")))
        if not output.is_file() or file_sha256(output) != str(row.get("output_sha256", "")):
            _error(errors, f"video hash mismatch: {key}")
    expected = {(sample_id, repeat, arm) for sample_id in config.EXPECTED_SAMPLE_IDS for repeat in config.REPEATS for arm in config.ARMS}
    if seen != expected:
        _error(errors, "video identities do not cover the 176-cell render matrix")
    return payload


def _validate_scores(root: Path, cohort: Mapping[str, Any], geometry: Mapping[str, Any] | None, errors: list[str]) -> dict[str, Any] | None:
    payload = _check_self(config.RunPaths(root).scores / "matrix_manifest.json", errors, "score matrix")
    if payload is None:
        return None
    if payload.get("status") != "complete" or payload.get("cell_count") != config.EXPECTED_CELL_COUNT:
        _error(errors, "score matrix is incomplete")
    expected = {(sample_id, repeat, f"V_{video}/A_{audio}") for sample_id in config.EXPECTED_SAMPLE_IDS for repeat in config.REPEATS for video, audio, _ in config.SCORE_CELLS}
    seen: set[tuple[str, int, str]] = set()
    expected_groups = {str(record["sample_id"]): str(record["source_group"]) for record in cohort.get("records", [])}
    expected_cells = {f"V_{video}/A_{audio}": (video, audio) for video, audio, _purpose in config.SCORE_CELLS}
    for row in payload.get("rows", []):
        try:
            key = (str(row.get("sample_id")), int(row.get("render_repeat", -1)), str(row.get("cell")))
        except (TypeError, ValueError) as exc:
            _error(errors, f"malformed score cell identity: {exc}")
            continue
        if key in seen:
            _error(errors, f"duplicate score cell: {key}")
        seen.add(key)
        if expected_groups.get(key[0]) != str(row.get("source_group", "")):
            _error(errors, f"score source-group binding mismatch: {key}")
        cell_parts = expected_cells.get(key[2])
        if cell_parts is not None and (row.get("video_arm") != cell_parts[0] or row.get("score_audio_arm") != cell_parts[1]):
            _error(errors, f"score arm binding mismatch: {key}")
        if row.get("protocol_hash") != file_sha256(config.RunPaths(root).protocol / "setup.json"):
            _error(errors, f"score protocol hash mismatch: {key}")
        mux = Path(str(row.get("mux", "")))
        if not mux.is_file() or file_sha256(mux) != str(row.get("mux_sha256", "")):
            _error(errors, f"mux hash mismatch: {key}")
        meta = row.get("mux_meta", {})
        if meta.get("audio_pcm_verified") is not True or meta.get("video_stream_copy_verified") is not True:
            _error(errors, f"mux verification flag missing: {key}")
        score = row.get("score")
        if not isinstance(score, Mapping):
            _error(errors, f"score mapping is missing: {key}")
            continue
        if score.get("official_forward") is not True or not isinstance(score.get("parity"), Mapping) or score["parity"].get("checked") is not True:
            _error(errors, f"official SyncNet/parity flag missing: {key}")
        try:
            _score_triplet(row)
        except Exception as exc:  # noqa: BLE001 - report malformed score values per cell
            _error(errors, f"score value invalid: {key}: {exc}")
        try:
            parsed = _official_log_triplet(Path(str(score.get("raw_log", ""))))
            recorded = _score_triplet(row)
            if not math.isclose(parsed[0], recorded[0], rel_tol=0.0, abs_tol=1e-12) or not math.isclose(parsed[1], recorded[1], rel_tol=0.0, abs_tol=1e-12) or parsed[2] != recorded[2]:
                _error(errors, f"score differs from official log: {key}")
        except Exception as exc:  # noqa: BLE001 - report missing or malformed official log per cell
            _error(errors, f"official log invalid: {key}: {exc}")
    if seen != expected:
        _error(errors, "score identities do not cover the exact 308-cell matrix")
    if geometry is not None and geometry.get("status") == "complete":
        support_by_id = {str(row["sample_id"]): row for row in geometry.get("rows", [])}
        for sample_id in config.EXPECTED_SAMPLE_IDS:
            support = support_by_id.get(sample_id)
            if support is None:
                continue
            prefix_len = int(support["support_frames"]) * config.SAMPLES_PER_FRAME
            cohort_rows = verify_self_hashed_json(config.RunPaths(root).protocol / "cohort.json")["records"]
            cohort_row = next(row for row in cohort_rows if str(row["sample_id"]) == sample_id)
            natural = _read_pcm(Path(str(cohort_row["natural_audio"]["path"])))
            reverse_path = config.RunPaths(root).scores / "audio" / "N_REV" / f"{sample_id}.wav"
            if not reverse_path.is_file():
                _error(errors, f"N_REV audio is missing: {sample_id}")
            elif not np.array_equal(_read_pcm(reverse_path), natural[:prefix_len][::-1]):
                _error(errors, f"N_REV is not exact int16 reversal: {sample_id}")
    return payload


def _validate_analysis(
    root: Path,
    errors: list[str],
    score_payload: Mapping[str, Any] | None,
    diagnostics_payload: Mapping[str, Any] | None,
    analysis_dir: Path | None = None,
) -> None:
    paths = config.RunPaths(root)
    destination = analysis_dir or paths.analysis
    label_suffix = "" if destination == paths.analysis else f" ({destination.name})"
    final = _check_self(destination / "final.json", errors, f"final analysis{label_suffix}")
    statistics = _check_self(destination / "statistics.json", errors, f"statistics{label_suffix}")
    if final is None or statistics is None:
        return
    if final.get("causal_quality_effect_established") is not False or final.get("generalization_confirmed") is not False:
        _error(errors, "final analysis contains an overclaim flag")
    primary = statistics.get("primary", {}).get("deltaC", {})
    per_record = statistics.get("per_record", [])
    if len(per_record) != config.EXPECTED_RECORD_COUNT:
        _error(errors, "analysis denominator is not 22")
    if score_payload is None or diagnostics_payload is None:
        return
    try:
        recomputed = _recompute_controls_and_terminal(score_payload, diagnostics_payload)
    except Exception as exc:  # noqa: BLE001 - validator reports independent recomputation failure
        _error(errors, f"independent analysis recomputation failed: {exc}")
        return
    reported_by_id = {str(row.get("sample_id")): row for row in per_record if isinstance(row, Mapping)}
    expected_by_id = {str(row["sample_id"]): row for row in recomputed["primary_rows"]}
    if set(reported_by_id) != set(expected_by_id):
        _error(errors, "analysis rows do not cover the score denominator")
    for sample_id, expected in expected_by_id.items():
        actual = reported_by_id.get(sample_id)
        if actual is None:
            continue
        if actual.get("source_group") != expected["source_group"]:
            _error(errors, f"analysis source-group binding mismatch: {sample_id}")
        for field in ("gC_LOCAL", "gC_CLOUD", "deltaC", "gD_LOCAL", "gD_CLOUD", "deltaD"):
            if not math.isclose(float(actual.get(field)), float(expected[field]), rel_tol=0.0, abs_tol=1e-12):
                _error(errors, f"analysis {field} does not match score matrix: {sample_id}")
    if primary.get("mean") is None or not math.isclose(float(recomputed["primary_mean"]), float(primary["mean"]), rel_tol=0.0, abs_tol=1e-12):
        _error(errors, "analysis primary mean does not match score matrix")
    if primary.get("ci") is None or recomputed["primary_ci"] is None or not np.allclose(primary["ci"], recomputed["primary_ci"], rtol=0.0, atol=1e-12):
        _error(errors, "analysis primary CI does not match independent bootstrap")
    if statistics.get("controls_pass") is not recomputed["controls_pass"]:
        _error(errors, "analysis controls_pass does not match independent control recomputation")
    if statistics.get("movement_pass") is not all(recomputed["movement"].values()):
        _error(errors, "analysis movement_pass does not match diagnostics")
    if statistics.get("terminal") != recomputed["terminal"] or final.get("terminal_state") != recomputed["terminal"]:
        _error(errors, "analysis terminal state does not match independent decision recomputation")


def validate_run(root: str | Path) -> dict[str, Any]:
    run_root = Path(root).resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    setup, cohort = _validate_protocol(run_root, errors)
    checks["protocol"] = setup is not None and cohort is not None
    if cohort is None:
        result = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "valid": False, "errors": errors, "checks": checks}
        write_self_hashed_json(run_root / "validation.json", result)
        return result
    checks["tts"] = _validate_tts(run_root, cohort, errors) is not None
    checks["targets"] = _validate_targets(run_root, cohort, errors) is not None
    bridge_manifest, geometry, diagnostics = _validate_bridge(run_root, cohort, errors)
    checks["bridge"] = bridge_manifest is not None and geometry is not None and diagnostics is not None
    checks["videos"] = _validate_videos(run_root, errors) is not None
    score_payload = _validate_scores(run_root, cohort, geometry, errors)
    checks["scores"] = score_payload is not None
    _validate_analysis(run_root, errors, score_payload, diagnostics)
    checks["analysis"] = (run_root / "07_analysis" / "final.json").is_file()
    base_final = run_root / "07_analysis" / "final.json"
    previous_final = base_final if base_final.is_file() else None
    versions_dir = run_root / "07_analysis" / "versions"
    if versions_dir.is_dir():
        version_directories = sorted(
            (directory for directory in versions_dir.iterdir() if directory.is_dir() and directory.name.startswith("v") and directory.name[1:].isdigit()),
            key=lambda directory: int(directory.name[1:]),
        )
        previous_version = 1
        for directory in version_directories:
            version = int(directory.name[1:])
            if version != previous_version + 1:
                _error(errors, f"analysis versions are not contiguous: {directory.name}")
            final_path = directory / "final.json"
            if final_path.is_file():
                try:
                    version_final = verify_self_hashed_json(final_path)
                    if version_final.get("analysis_version") != version:
                        _error(errors, f"analysis version field mismatch: {directory.name}")
                    if previous_final is not None and version_final.get("parent_analysis_sha256") != file_sha256(previous_final):
                        _error(errors, f"analysis parent hash mismatch: {directory.name}")
                except Exception as exc:  # noqa: BLE001 - report version-chain corruption
                    _error(errors, f"analysis version chain is invalid: {directory.name}: {exc}")
                previous_final = final_path
            _validate_analysis(run_root, errors, score_payload, diagnostics, directory)
            previous_version = version
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "run_root": str(run_root),
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "independent": True,
    }
    write_self_hashed_json(run_root / "validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="independent LRS3 bridge run validator")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_run(args.root)
    except Exception as exc:  # noqa: BLE001 - CLI reports validator failure
        print(json.dumps({"valid": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
