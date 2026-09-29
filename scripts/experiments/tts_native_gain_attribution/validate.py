"""Independent completion validator for the native-gain continuation.

The validator shares only file/JSON I/O helpers with the producers.  It
recomputes SyncNet endpoint reductions, the B four-cell effects, the six
source-group bootstrap intervals, and the two controls from the stored
features/matrices.  A forged summary or ``COMPLETE`` field cannot create a
valid B result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    canonical_sha256,
    executable_path,
    ffprobe_json,
    file_sha256,
    read_json,
    read_self_hashed_json,
    write_self_hashed_json,
)

REQUIRED_A_FIELDS = ("stage", "id", "source_group", "source", "video_type", "eval_condition", "video_hash", "pcm_hash", "support_hash", "matrix_hash", "status")
REQUIRED_B_FIELDS = ("stage", "id", "source_group", "source", "seed", "video_cell_id", "video_hash", "pcm_hash", "roi_hash", "support_hash", "matrix_hash", "status")
CONTROL_STATUSES = {"IDENTITY_PASS", "DELAY_DETECTED", "GENERATION_UNSTABLE", "CONTROL_FAILED", "CONTROL_UNINFORMATIVE"}


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _self(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return read_self_hashed_json(path)
    except (OSError, ProtocolError, ValueError, TypeError) as exc:
        _fail(errors, f"{path}: {exc}")
        return {}


def _float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} is not numeric") from exc
    if not np.isfinite(number):
        raise ProtocolError(f"{label} is non-finite")
    return number


def _assert_close(errors: list[str], expected: Any, actual: Any, label: str, tolerance: float = 1e-6) -> None:
    if expected is None or actual is None:
        if expected != actual:
            _fail(errors, f"{label}: {expected!r} != {actual!r}")
    elif isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            _fail(errors, f"{label}: expected mapping")
            return
        for key, value in expected.items():
            if key not in actual:
                _fail(errors, f"{label}: missing {key}")
            else:
                _assert_close(errors, value, actual[key], f"{label}.{key}", tolerance)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            _fail(errors, f"{label}: list differs")
        else:
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                _assert_close(errors, left, right, f"{label}[{index}]", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            if abs(float(expected) - float(actual)) > tolerance:
                _fail(errors, f"{label}: {expected!r} != {actual!r}")
        except (TypeError, ValueError):
            _fail(errors, f"{label}: actual is not numeric")
    elif expected != actual:
        _fail(errors, f"{label}: {expected!r} != {actual!r}")


def _resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError("file binding has no path")
    target = Path(value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def _verify_file_binding(value: Any, label: str, errors: list[str]) -> Path | None:
    try:
        if isinstance(value, Mapping):
            path = _resolve_path(value.get("path"))
            expected_hash = value.get("sha256", value.get("file_sha256"))
            expected_bytes = value.get("bytes")
        else:
            path = _resolve_path(value)
            expected_hash = None
            expected_bytes = None
        if not path.is_file():
            raise ProtocolError(f"missing {path}")
        actual = file_sha256(path)
        if expected_hash is not None and actual != str(expected_hash):
            raise ProtocolError(f"sha256 {actual} != {expected_hash}")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise ProtocolError(f"bytes {path.stat().st_size} != {expected_bytes}")
        return path
    except (OSError, ProtocolError, TypeError, ValueError) as exc:
        _fail(errors, f"{label}: {exc}")
        return None


def _expected_a_code_hash() -> str:
    names = ("config.py", "common.py", "audio.py", "syncnet.py")
    return canonical_sha256({name: file_sha256(config.REPO / "scripts/experiments/tts_native_gain_attribution" / name) for name in names})


def _allowed_audio_code_hashes(paths: config.RunPaths) -> set[str]:
    allowed = {file_sha256(config.REPO / "scripts/experiments/tts_native_gain_attribution/audio.py")}
    if not paths.reuse_manifest.is_file():
        return allowed
    try:
        reuse = read_self_hashed_json(paths.reuse_manifest)
        parent_protocol = read_self_hashed_json(Path(str(reuse["parent_run"])) / "protocol.json")
        frozen = parent_protocol.get("frozen_code", {})
        parent_hash = frozen.get("scripts/experiments/tts_native_gain_attribution/audio.py")
        if isinstance(parent_hash, str):
            allowed.add(parent_hash)
    except (KeyError, OSError, ProtocolError, TypeError, ValueError):
        pass
    return allowed


def _expected_b_generation_code_hash() -> str:
    root = config.REPO / "scripts/experiments/tts_native_gain_attribution"
    return canonical_sha256({name: file_sha256(root / name) for name in ("generation.py", "leaptalk_adapter.py")})


def _provenance_model_hash(provenance: Mapping[str, Any]) -> str:
    body = dict(provenance)
    body.pop("artifact_sha256", None)
    body.pop("protocol_id", None)
    body.pop("schema_version", None)
    return canonical_sha256(body)


def _expected_b_score_code_hash() -> str:
    root = config.REPO / "scripts/experiments/tts_native_gain_attribution"
    return canonical_sha256({name: file_sha256(root / name) for name in ("generation.py", "syncnet.py", "common.py")})


def _allowed_a_code_hashes(paths: config.RunPaths, fixed: Mapping[str, Any], a: Mapping[str, Any]) -> set[str]:
    allowed = {_expected_a_code_hash()}
    if paths.reuse_manifest.is_file():
        if isinstance(fixed.get("code_hash"), str):
            allowed.add(str(fixed["code_hash"]))
        if isinstance(a.get("code_hash"), str):
            allowed.add(str(a["code_hash"]))
        for row in [*a.get("cells", []), *a.get("controls", [])]:
            if isinstance(row, Mapping) and isinstance(row.get("code_hash"), str):
                allowed.add(str(row["code_hash"]))
    return allowed


def _independent_curve(matrix: np.ndarray, support: Sequence[int]) -> dict[str, Any]:
    value = np.asarray(matrix)
    rows = np.asarray(list(support), dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != config.LAG_COUNT or value.shape[0] < 1 or not np.isfinite(value).all() or rows.ndim != 1 or rows.size < 1 or np.any(rows < 0) or np.any(rows >= value.shape[0]) or np.any(np.diff(rows) <= 0):
        raise ProtocolError(f"invalid matrix/support: {value.shape}/{rows.shape}")
    curve = value[rows].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    index = int(np.argmin(curve))
    distance = float(curve[index])
    background = float(np.median(curve))
    return {"sync_c": background - distance, "sync_d": distance, "background_b": background, "d0": float(curve[config.VSHIFT]), "official_offset": int(config.VSHIFT - index), "min_index": index, "support_count": int(rows.size)}


def _bootstrap_indices() -> np.ndarray:
    return np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED)).integers(0, len(config.SAMPLE_IDS), size=(config.BOOTSTRAP_DRAWS, len(config.SAMPLE_IDS)), dtype=np.int64)


def _independent_bootstrap(values: Mapping[str, float], indices: np.ndarray, metric: str) -> dict[str, Any]:
    labels = sorted(values)
    if len(labels) != len(config.SAMPLE_IDS):
        return {"status": "INCOMPLETE", "metric": metric, "group_count": len(labels), "expected_group_count": len(config.SAMPLE_IDS)}
    array = np.asarray([_float(values[label], f"{metric}/{label}") for label in labels], dtype=np.float64)
    draws = array[np.asarray(indices, dtype=np.int64)].mean(axis=1, dtype=np.float64)
    tail = 0.05 / (2 * 6)
    quantile = lambda probability: float(np.quantile(draws, probability, method="linear"))
    return {"status": "COMPLETE", "metric": metric, "mean": float(array.mean()), "ci95": [quantile(0.025), quantile(0.975)], "ci99_166667_bonferroni": [quantile(tail), quantile(1 - tail)], "ci_bonferroni": [quantile(tail), quantile(1 - tail)], "bonferroni_comparisons": 6, "bonferroni_tail_probability": tail, "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, array, strict=True)}, "group_count": len(labels), "draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "rng": "numpy.random.Generator(PCG64)", "quantile_method": "linear", "positive_evidence": bool(quantile(tail) > 0.0), "reverse_evidence": bool(quantile(1 - tail) < 0.0), "practical_threshold_diagnostic": bool(array.mean() >= config.PRACTICAL_THRESHOLD), "equivalent_within_0.2": bool(quantile(tail) >= -0.2 and quantile(1 - tail) <= 0.2)}


def _read_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            fields = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes())
            data = handle.readframes(fields[3])
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"cannot read WAV {path}") from exc
    if fields[:3] != (1, 2, config.SAMPLE_RATE):
        raise ProtocolError(f"WAV format differs: {path}")
    values = np.frombuffer(data, dtype="<i2").copy()
    if values.size != fields[3] or values.size == 0:
        raise ProtocolError(f"WAV sample count differs: {path}")
    return values, {"sample_count": int(values.size), "pcm_sha256": hashlib.sha256(data).hexdigest()}


def _validate_b_audio(
    row: Mapping[str, Any],
    source_audio: Mapping[str, Any] | None,
    errors: list[str],
    label: str,
    source_pcm_cache: dict[str, np.ndarray],
    *,
    shift_samples: int = 0,
) -> None:
    """Verify a B score's cropped PCM against its source-clock interval."""

    if source_audio is None:
        _fail(errors, f"{label}: source audio binding is missing")
        return
    clip_path = _verify_file_binding(
        {"path": row.get("pcm_path"), "sha256": row.get("pcm_file_sha256")},
        f"{label}/pcm",
        errors,
    )
    source_path = _verify_file_binding(
        {"path": source_audio.get("path"), "sha256": source_audio.get("file_sha256")},
        f"{label}/source_pcm",
        errors,
    )
    if clip_path is None or source_path is None:
        return
    source_key = str(source_path)
    try:
        source_pcm = source_pcm_cache.get(source_key)
        if source_pcm is None:
            source_pcm, source_meta = _read_wav(source_path)
            source_pcm_cache[source_key] = source_pcm
        else:
            source_meta = {"pcm_sha256": hashlib.sha256(np.asarray(source_pcm, dtype="<i2").tobytes()).hexdigest()}
        if source_meta.get("pcm_sha256") != source_audio.get("pcm_sha256"):
            _fail(errors, f"{label}: source PCM hash differs")
        if row.get("audio_source_path") != source_audio.get("path") or row.get("audio_source_file_sha256") != source_audio.get("file_sha256") or row.get("audio_source_pcm_sha256") != source_audio.get("pcm_sha256"):
            _fail(errors, f"{label}: source audio provenance differs")
        start = int(row.get("audio_start_samples", -1))
        end = int(row.get("audio_end_samples", -1))
        start_frame = int(row.get("time_start_frame", -1))
        end_frame = int(row.get("time_end_frame_exclusive", -1))
        if start < 0 or end <= start or end > source_pcm.size or start != start_frame * config.SAMPLES_PER_FRAME or end != end_frame * config.SAMPLES_PER_FRAME or end_frame <= start_frame:
            raise ProtocolError(f"{label}: source-clock sample interval is invalid")
        if int(row.get("shift_samples", 0)) != int(shift_samples):
            raise ProtocolError(f"{label}: audio shift metadata differs")
        expected = np.asarray(source_pcm[start:end], dtype=np.int16)
        if shift_samples:
            shifted = np.zeros_like(expected)
            if int(shift_samples) < expected.size:
                shifted[int(shift_samples):] = expected[: expected.size - int(shift_samples)]
            expected = shifted
        actual, meta = _read_wav(clip_path)
        expected_hash = hashlib.sha256(np.asarray(expected, dtype="<i2").tobytes()).hexdigest()
        if meta.get("pcm_sha256") != row.get("pcm_hash") or meta.get("pcm_sha256") != expected_hash or not np.array_equal(actual, expected):
            _fail(errors, f"{label}: cropped PCM does not match source-clock slice")
        clock = _self(clip_path.with_suffix(".json"), errors)
        if clock:
            expected_clock = {
                "sample_id": int(row.get("id", -1)),
                "source": str(row.get("source")),
                "seed": int(row.get("seed", 42)),
                "time_start_frame": start_frame,
                "time_end_frame_exclusive": end_frame,
                "sample_start": start,
                "sample_end": end,
                "source_audio_path": str(source_audio.get("path")),
                "source_audio_file_sha256": source_audio.get("file_sha256"),
                "source_audio_pcm_sha256": source_audio.get("pcm_sha256"),
            }
            for field, expected_value in expected_clock.items():
                if clock.get(field) != expected_value:
                    _fail(errors, f"{label}: clock metadata differs at {field}")
            if clock.get("file_sha256") != file_sha256(clip_path) or clock.get("pcm_sha256") != row.get("pcm_hash") or row.get("audio_clock_hash") != clock.get("artifact_sha256"):
                _fail(errors, f"{label}: clock artifact binding differs")
            if int(clock.get("shift_samples", 0)) != int(shift_samples):
                _fail(errors, f"{label}: clock shift metadata differs")
    except (OSError, ProtocolError, TypeError, ValueError) as exc:
        _fail(errors, f"{label}: {exc}")


def _validate_parent_continuation(paths: config.RunPaths, errors: list[str]) -> None:
    if not paths.parent_evidence.is_file() and not paths.reuse_manifest.is_file():
        return
    evidence = _self(paths.parent_evidence, errors)
    reuse = _self(paths.reuse_manifest, errors)
    if evidence.get("parent_immutable") is not True or reuse.get("read_only") is not True or reuse.get("no_writable_hard_links") is not True:
        _fail(errors, "continuation parent/reuse artifacts do not declare immutable read-only reuse")
    parent = evidence.get("parent_transitive", {}).get("parent")
    if not isinstance(parent, str):
        _fail(errors, "continuation parent path is missing")
        return
    parent_path = Path(parent)
    hashes = evidence.get("parent_transitive", {}).get("manifest_hashes", {})
    expected_paths = {"protocol": parent_path / "protocol.json", "inputs": parent_path / "inputs.json", "claims": parent_path / "claim_registry.json", "provenance": parent_path / "model_provenance.json", "assets": parent_path / "00_audit/assets.json", "audio": parent_path / "01_audio/manifest.json", "fixed": parent_path / "02_fixed_video/manifest.json", "a": parent_path / "02_fixed_video/a_manifest.json", "analysis": parent_path / "05_analysis/summary.json", "perception": parent_path / "06_perception/package.json", "perception_analysis": parent_path / "06_perception/analysis.json", "final": parent_path / "final.json", "validation": parent_path / "validation.json"}
    for name, path in expected_paths.items():
        if not path.is_file():
            _fail(errors, f"immutable parent artifact is missing: {name}/{path}")
        elif isinstance(hashes, Mapping) and name in hashes and file_sha256(path) != str(hashes[name]):
            _fail(errors, f"immutable parent artifact changed: {name}")
    for entry in reuse.get("entries", []):
        if not isinstance(entry, Mapping) or entry.get("mode") != "read_only_symlink":
            _fail(errors, "reuse manifest contains a non-read-only entry")
    if paths.audio.is_symlink() and paths.audio.resolve() != (parent_path / "01_audio").resolve():
        _fail(errors, "continuation audio symlink does not target parent audio")
    if paths.fixed_video.is_symlink() and paths.fixed_video.resolve() != (parent_path / "02_fixed_video").resolve():
        _fail(errors, "continuation fixed-video symlink does not target parent fixed video")


def _validate_audit(paths: config.RunPaths, errors: list[str]) -> dict[str, Any]:
    assets = _self(paths.audit / "assets.json", errors)
    if assets.get("protocol_id") != config.PROTOCOL_ID or assets.get("status") != "COMPLETE":
        _fail(errors, "audit is not complete")
    if assets.get("sample_ids") != list(config.SAMPLE_IDS) or int(assets.get("record_count", -1)) != len(config.SAMPLE_IDS):
        _fail(errors, "audit cohort identity/count differs")
    input_binding = assets.get("input_bindings", {})
    binding_path = _verify_file_binding(input_binding, "input-bindings artifact", errors)
    if binding_path is not None:
        try:
            bindings = read_json(binding_path)
            if not isinstance(bindings, Mapping) or bindings.get("change_id") != "disentangle-tts-native-gain":
                raise ProtocolError("input-bindings change_id differs")
            files = bindings.get("files")
            if not isinstance(files, list) or len(files) != 142:
                raise ProtocolError("input-bindings must contain 142 files")
            for index, row in enumerate(files):
                _verify_file_binding(row, f"input binding {index}", errors)
        except (OSError, ProtocolError, TypeError, ValueError) as exc:
            _fail(errors, f"input-bindings document: {exc}")
    if "continuation" in assets:
        _validate_parent_continuation(paths, errors)
    else:
        parent_evidence = assets.get("parent_evidence", {})
        for key, historical in (("final_sha256", config.PARENT_FINAL), ("validation_sha256", config.PARENT_VALIDATION), ("cohort_sha256", config.PARENT_COHORT), ("inputs_sha256", config.PARENT_INPUTS)):
            if not historical.is_file() or parent_evidence.get(key) != file_sha256(historical):
                _fail(errors, f"parent evidence hash differs: {key}")
    groups: set[str] = set()
    for row in assets.get("records", []):
        sample_id = int(row.get("sample_id", -1))
        group = str(row.get("source_group", ""))
        if group in groups or not group:
            _fail(errors, f"source group is missing or duplicated: {sample_id}/{group}")
        groups.add(group)
        _verify_file_binding({"path": row.get("real_video"), "sha256": row.get("real_video_sha256")}, f"real video/{sample_id}", errors)
        _verify_file_binding(row.get("portrait"), f"portrait/{sample_id}", errors)
        for source in config.SOURCES:
            source_row = row.get("sources", {}).get(source, {})
            if not isinstance(source_row, Mapping):
                _fail(errors, f"missing source binding {sample_id}/{source}")
                continue
            required_keys = ("media", "audio_media", "real_video") if source == "R" else ("media", "audio_media", "crop", "matrix", "crop_selection", "worker_result")
            for key in required_keys:
                if key in source_row:
                    _verify_file_binding(source_row[key], f"audit {sample_id}/{source}/{key}", errors)
                else:
                    _fail(errors, f"missing source binding {sample_id}/{source}/{key}")
    return assets


def _validate_audio(paths: config.RunPaths, assets: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    manifest = _self(paths.audio / "manifest.json", errors)
    rows = manifest.get("records", [])
    if manifest.get("status") != "COMPLETE" or int(manifest.get("record_count", -1)) != 180 or len(rows) != 180:
        _fail(errors, "audio manifest count/status differs")
    if manifest.get("audio_code_hash") not in _allowed_audio_code_hashes(paths):
        _fail(errors, "audio manifest code hash differs")
    by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    by_id: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            _fail(errors, "audio row is malformed")
            continue
        key = (int(row.get("sample_id", -1)), str(row.get("source")), str(row.get("condition")))
        if key in by_key:
            _fail(errors, f"duplicate audio identity {key}")
        by_key[key] = row
        by_id[key[0]].append(row)
        path = _verify_file_binding({"path": row.get("path"), "sha256": row.get("file_sha256")}, f"audio/{key}", errors)
        if path is None:
            continue
        try:
            _, meta = _read_wav(path)
            if meta["pcm_sha256"] != row.get("pcm_sha256") or meta["sample_count"] != int(row.get("sample_count", -1)):
                _fail(errors, f"audio PCM binding differs {key}")
        except ProtocolError as exc:
            _fail(errors, str(exc))
    asset_map = {int(row["sample_id"]): row for row in assets.get("records", []) if isinstance(row, Mapping)}
    for sample_id in config.SAMPLE_IDS:
        if len(by_id[sample_id]) != 15:
            _fail(errors, f"audio denominator differs for {sample_id}")
        headrooms = []
        for source in config.SOURCES:
            source_manifest = _self(paths.audio / str(sample_id) / source / "manifest.json", errors)
            if source_manifest.get("status") != "COMPLETE" or len(source_manifest.get("conditions", [])) != 5:
                _fail(errors, f"source audio manifest incomplete {sample_id}/{source}")
            for artifact_name in ("frame_rms", "activity_mask"):
                _verify_file_binding(source_manifest.get("activity_artifacts", {}).get(artifact_name), f"activity/{sample_id}/{source}/{artifact_name}", errors)
            headrooms.append(source_manifest.get("common_headroom", {}).get("g"))
            original = by_key.get((sample_id, source, "ORIGINAL"))
            expected = asset_map.get(sample_id, {}).get("sources", {}).get(source, {}).get("expected_pcm_sha256")
            if expected and (original is None or original.get("pcm_sha256") != expected):
                _fail(errors, f"ORIGINAL audio hash differs {sample_id}/{source}")
            gain = by_key.get((sample_id, source, "GAIN"))
            noise = by_key.get((sample_id, source, "NOISE"))
            if gain is None or abs(_float(gain.get("operation", {}).get("measured_gain_db", 999), "gain") + 6.0) > 0.01:
                _fail(errors, f"GAIN contract failed {sample_id}/{source}")
            if noise is None or abs(_float(noise.get("operation", {}).get("measured_snr_db", 999), "snr") - 20.0) > 0.1:
                _fail(errors, f"NOISE20 contract failed {sample_id}/{source}")
        if len({str(value) for value in headrooms}) != 1:
            _fail(errors, f"common headroom differs across sources {sample_id}")
    return {**manifest, "by_key": by_key}


def _independent_pixel_hash(path: Path) -> tuple[str, int, list[int]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("opencv is required for independent video validation") from exc
    capture = cv2.VideoCapture(str(path))
    digest = hashlib.sha256()
    count = 0
    shape: list[int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = [int(item) for item in frame.shape]
            if shape is None:
                shape = current
            if shape != current:
                raise ProtocolError(f"video frame geometry changes: {path}")
            digest.update(np.ascontiguousarray(frame).tobytes())
            count += 1
    finally:
        capture.release()
    if shape is None or count < 5:
        raise ProtocolError(f"video has too few decodable frames: {path}")
    return digest.hexdigest(), count, shape


def _first_pts(path: Path) -> dict[str, Any]:
    executable = executable_path(config.FFPROBE, "ffprobe")
    command = (str(executable), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time", "-of", "json", str(path))
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=30)
        payload = json.loads(result.stdout.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read first PTS: {path}") from exc
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        raise ProtocolError(f"first PTS is missing: {path}")
    return dict(frames[0])


def _validate_video_identity(row: Mapping[str, Any], errors: list[str], label: str, *, allow_audio: bool = False) -> None:
    path = _verify_file_binding({"path": row.get("path"), "sha256": row.get("file_sha256")}, f"{label}/file", errors)
    if path is None:
        return
    try:
        pixel, count, shape = _independent_pixel_hash(path)
        if pixel != row.get("pixel_sha256") or count != int(row.get("frame_count", -1)) or (row.get("frame_shape") and shape != [int(item) for item in row["frame_shape"]]):
            _fail(errors, f"{label}: decoded pixel identity differs")
        probe = ffprobe_json(path)
        streams = [stream for stream in probe.get("streams", []) if isinstance(stream, Mapping)]
        if not allow_audio and any(stream.get("codec_type") == "audio" for stream in streams):
            _fail(errors, f"{label}: unexpected audio stream")
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
        if not isinstance(rate, str) or "/" not in rate:
            raise ProtocolError("video fps is missing")
        numerator, denominator = rate.split("/", 1)
        if abs(float(numerator) / float(denominator) - config.FPS) > 0.01:
            raise ProtocolError("video fps differs from registered 25 fps")
        first = _first_pts(path)
        pts_payload = {
            "time_base": video.get("time_base"),
            "start_time": video.get("start_time"),
            "first_frame_pts": first.get("pts"),
            "first_frame_pts_time": first.get("pts_time"),
            "first_frame_best_effort_timestamp": first.get("best_effort_timestamp"),
            "first_frame_best_effort_timestamp_time": first.get("best_effort_timestamp_time"),
            "frame_count": count,
            "fps": rate,
        }
        expected_pts_hash = row.get("pts_sha256")
        actual_pts_hash = canonical_sha256(pts_payload)
        if expected_pts_hash not in (None, "") and expected_pts_hash != actual_pts_hash:
            _fail(errors, f"{label}: PTS identity differs")
        expected_video_hash = row.get("video_hash")
        actual_video_hash = canonical_sha256({"pixel_sha256": pixel, "pts_sha256": actual_pts_hash})
        if expected_video_hash not in (None, "") and expected_video_hash != actual_video_hash:
            _fail(errors, f"{label}: video identity hash differs")
        pts = row.get("pts", {})
        for stored, actual in (("first_frame_pts", "pts"), ("first_frame_pts_time", "pts_time"), ("first_frame_best_effort_timestamp", "best_effort_timestamp"), ("first_frame_best_effort_timestamp_time", "best_effort_timestamp_time")):
            if stored in pts and pts.get(stored) != first.get(actual):
                _fail(errors, f"{label}: {stored} differs")
    except (OSError, ProtocolError, ValueError, TypeError) as exc:
        _fail(errors, f"{label}: {exc}")


def _validate_fixed(paths: config.RunPaths, errors: list[str]) -> tuple[dict[str, Any], dict[tuple[int, str], Mapping[str, Any]]]:
    fixed = _self(paths.fixed_video / "manifest.json", errors)
    videos = fixed.get("videos", [])
    if fixed.get("status") != "COMPLETE" or int(fixed.get("video_count", -1)) != 36 or len(videos) != 36:
        _fail(errors, "fixed video manifest count/status differs")
    seen: set[tuple[int, str]] = set()
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in videos:
        key = (int(row.get("sample_id", -1)), str(row.get("video_type")))
        if key in seen:
            _fail(errors, f"duplicate fixed video identity {key}")
        seen.add(key)
        result[key] = row
        selection = _verify_file_binding({"path": row.get("selection"), "sha256": row.get("selection_sha256")}, f"fixed selection/{key}", errors)
        if selection is not None and selection.name != f"{key[0]}.selection.json":
            _fail(errors, f"selection is not ID-specific {key}")
        # Historical SyncNet crops may carry an embedded PCM stream; A's
        # scientific score still binds the independent audio manifest.  The
        # blind package is where audio must be stripped and is checked below.
        _validate_video_identity(row, errors, f"fixed/{key}", allow_audio=True)
        if row.get("pts_sha256") != canonical_sha256(row.get("pts", {})):
            _fail(errors, f"fixed PTS hash differs {key}")
    expected = {(sample_id, video_type) for sample_id in config.SAMPLE_IDS for video_type in config.VIDEO_TYPES}
    if seen != expected:
        _fail(errors, f"fixed identity set differs: missing={sorted(expected - seen)} extra={sorted(seen - expected)}")
    return fixed, result


def _validate_roi(
    row: Mapping[str, Any],
    errors: list[str],
    label: str,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    path = _verify_file_binding(row.get("roi_path"), f"{label}/roi", errors)
    if path is None:
        return None
    try:
        roi = read_self_hashed_json(path)
        if roi.get("status") != "COMPLETE" or roi.get("candidate_tracking_forbidden") is not True or roi.get("padding_rows") != []:
            _fail(errors, f"{label}: ROI flags differ")
        frames = np.asarray(roi.get("frame_indices"), dtype=np.int64)
        boxes = np.asarray(roi.get("bbox"), dtype=np.float64)
        if frames.ndim != 1 or boxes.shape != (frames.size, 4) or frames.size <= config.OFFICIAL_PIPELINE["min_track"] or np.any(np.diff(frames) != 1):
            _fail(errors, f"{label}: frozen ROI geometry is invalid")
        if row.get("roi_hash") is None:
            _fail(errors, f"{label}: ROI hash is missing")
        else:
            expected_roi_hash = canonical_sha256({"roi": roi.get("artifact_sha256"), "baseline": roi.get("baseline_video_hash"), "candidate": row.get("video_hash"), "frame_indices": roi.get("frame_indices")})
            if row.get("roi_hash") != expected_roi_hash:
                _fail(errors, f"{label}: ROI hash does not bind the candidate and frozen baseline")
        if int(roi.get("sample_id", -1)) != int(row.get("id", -2)) or str(roi.get("source")) != str(row.get("source")) or int(roi.get("seed", -1)) != int(row.get("seed", -2)):
            _fail(errors, f"{label}: ROI sample/source/seed identity differs")
        if baseline is not None:
            if roi.get("baseline_cell_id") != baseline.get("cell_id") or roi.get("baseline_video_hash") != baseline.get("video_hash"):
                _fail(errors, f"{label}: ROI is frozen from a different baseline")
            if roi.get("baseline_video_path") != baseline.get("path"):
                _fail(errors, f"{label}: ROI baseline path differs")
        start_frame = int(frames[0]) if frames.size else -1
        end_frame = int(frames[-1]) + 1 if frames.size else -1
        if roi.get("time_start_frame", start_frame) != start_frame or roi.get("time_end_frame_exclusive", end_frame) != end_frame or roi.get("audio_start_samples", start_frame * config.SAMPLES_PER_FRAME) != start_frame * config.SAMPLES_PER_FRAME or roi.get("audio_end_samples", end_frame * config.SAMPLES_PER_FRAME) != end_frame * config.SAMPLES_PER_FRAME:
            _fail(errors, f"{label}: ROI/audio clock metadata differs")
        return roi
    except (OSError, ProtocolError, TypeError, ValueError) as exc:
        _fail(errors, f"{label}/roi: {exc}")
        return None


def _load_feature(value: Any, expected_hash: Any, errors: list[str], label: str) -> np.ndarray | None:
    path = _verify_file_binding({"path": value, "sha256": expected_hash}, label, errors)
    if path is None:
        return None
    metadata_path = path.with_suffix(".json")
    metadata = _self(metadata_path, errors)
    if metadata and metadata.get("sha256") != file_sha256(path):
        _fail(errors, f"{label}: feature metadata hash differs")
    try:
        array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != config.EMBEDDING_DIM or not np.isfinite(array).all():
            raise ProtocolError(f"feature shape is invalid: {array.shape}")
        return array
    except (OSError, ValueError, ProtocolError) as exc:
        _fail(errors, f"{label}: {exc}")
        return None


def _independent_distance_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    v = np.asarray(visual, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    count = min(v.shape[0], a.shape[0])
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != config.EMBEDDING_DIM or a.shape[1] != config.EMBEDDING_DIM or count < 1:
        raise ProtocolError("invalid feature arrays for distance matrix")
    padded = np.pad(a[:count], ((config.VSHIFT, config.VSHIFT), (0, 0)))
    result = np.empty((count, config.LAG_COUNT), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for index in range(count):
        difference = v[index : index + 1] - padded[index : index + config.LAG_COUNT] + epsilon
        result[index] = np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float32), dtype=np.float32)
    return result.astype(np.float64)


def _independent_common_time_support(
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[str, np.ndarray],
    *,
    trim_rows: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Recompute the common source-frame support without importing the scorer."""

    intervals: list[set[int]] = []
    starts: dict[str, int] = {}
    for row in rows:
        cell_id = str(row.get("cell_id"))
        matrix = matrices.get(cell_id)
        if matrix is None or matrix.ndim != 2:
            raise ProtocolError(f"matrix is missing for support cell {cell_id}")
        start = int(row.get("time_start_frame", 0))
        count = int(matrix.shape[0])
        if count <= 2 * int(trim_rows):
            return [], {}
        starts[cell_id] = start
        intervals.append(set(range(start + int(trim_rows), start + count - int(trim_rows))))
    if not intervals:
        return [], {}
    common = sorted(set.intersection(*intervals))
    return common, {cell_id: [int(frame - starts[cell_id]) for frame in common] for cell_id in starts}


def _validate_a_cells(paths: config.RunPaths, assets: Mapping[str, Any], audio: Mapping[str, Any], fixed: Mapping[str, Any], errors: list[str]) -> tuple[dict[str, Any], dict[tuple[int, str, str], Mapping[str, Any]]]:
    a = _self(paths.fixed_video / "a_manifest.json", errors)
    cells = a.get("cells", [])
    controls = a.get("controls", [])
    if a.get("visual_decode_mode") != config.SYNCNET_VISUAL_DECODE_MODE or len(cells) != config.EXPECTED_A_SCIENCE or len(controls) != config.EXPECTED_A_CONTROLS:
        _fail(errors, "A manifest decoder/count differs")
    allowed_code = _allowed_a_code_hashes(paths, fixed, a)
    source_for = {"V_N": "N", "V_T": "T", "R": "R"}
    seen: set[tuple[int, str, str]] = set()
    by_family: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    audio_by_key = audio.get("by_key", {})
    fixed_by_key = {(int(row.get("sample_id", -1)), str(row.get("video_type"))): row for row in fixed.get("videos", [])}
    for row in cells:
        key = (int(row.get("id", -1)), str(row.get("video_type")), str(row.get("eval_condition")))
        if key in seen:
            _fail(errors, f"duplicate A cell {key}")
        seen.add(key)
        result[key] = row
        by_family[key[:2]].append(row)
        missing = [field for field in REQUIRED_A_FIELDS if field not in row]
        if missing:
            _fail(errors, f"A cell missing fields {key}: {missing}")
            continue
        if row.get("status") != "COMPLETE" or str(row.get("source")) != source_for.get(key[1]) or str(row.get("code_hash")) not in allowed_code:
            _fail(errors, f"A cell identity/status/code differs {key}")
        expected_audio = audio_by_key.get((key[0], str(row.get("source")), str(row.get("eval_condition"))))
        expected_video = fixed_by_key.get((key[0], key[1]))
        if expected_audio is not None and row.get("pcm_hash") != expected_audio.get("pcm_sha256"):
            _fail(errors, f"A cell PCM identity differs {key}")
        if expected_video is not None and row.get("video_hash") != expected_video.get("video_hash"):
            _fail(errors, f"A cell video identity differs {key}")
        matrix_path = _verify_file_binding({"path": row.get("matrix_path"), "sha256": row.get("matrix_hash")}, f"A matrix/{key}", errors)
        if matrix_path is not None:
            try:
                matrix = np.asarray(np.load(matrix_path, allow_pickle=False))
                if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or not np.isfinite(matrix).all() or list(matrix.shape) != [int(item) for item in row.get("matrix_shape", [])]:
                    _fail(errors, f"A matrix shape/content differs {key}")
            except (OSError, ValueError, TypeError) as exc:
                _fail(errors, f"A matrix unreadable {key}: {exc}")
    expected_keys = {(sample_id, video_type, condition) for sample_id in config.SAMPLE_IDS for video_type in config.VIDEO_TYPES for condition in config.AUDIO_CONDITIONS}
    if seen != expected_keys:
        _fail(errors, f"A identity set differs: missing={sorted(expected_keys - seen)} extra={sorted(seen - expected_keys)}")
    for family_key, family in by_family.items():
        supports = {tuple(int(item) for item in row.get("support_rows", [])) for row in family}
        hashes = {str(row.get("support_hash")) for row in family}
        matrices = []
        for row in family:
            try:
                matrices.append(np.load(_resolve_path(row.get("matrix_path")), allow_pickle=False))
            except (OSError, ProtocolError, ValueError, TypeError):
                pass
        expected_support = list(range(config.VSHIFT, min((matrix.shape[0] for matrix in matrices), default=0) - config.VSHIFT))
        expected_hash = canonical_sha256({"support": expected_support, "rule": "common INTERIOR after vshift"})
        if supports != {tuple(expected_support)} or hashes != {expected_hash} or len(expected_support) < config.MIN_INTERIOR_ROWS:
            _fail(errors, f"A support differs {family_key}")
        if len({str(row.get("video_hash")) for row in family}) != 1:
            _fail(errors, f"A family changed frozen video {family_key}")
        by_condition = {str(row.get("eval_condition")): row for row in family}
        if by_condition.get("ORIGINAL", {}).get("reused_from") is not None:
            try:
                original = np.load(_resolve_path(by_condition["ORIGINAL"]["matrix_path"]), allow_pickle=False)
                a0 = np.load(_resolve_path(by_condition["A0"]["matrix_path"]), allow_pickle=False)
                if not np.array_equal(original, a0):
                    _fail(errors, f"A ORIGINAL/A0 reuse is not byte-identical {family_key}")
            except (KeyError, OSError, ProtocolError, ValueError):
                _fail(errors, f"A ORIGINAL/A0 reuse is unreadable {family_key}")
    expected_controls = {(sample_id, video_type, shift) for sample_id in config.CONTROL_IDS for video_type in config.VIDEO_TYPES for shift in config.CONTROL_SHIFTS}
    control_seen: set[tuple[int, str, str]] = set()
    for row in controls:
        key = (int(row.get("id", -1)), str(row.get("video_type")), str(row.get("eval_condition")))
        control_seen.add(key)
        if row.get("status") != "COMPLETE" or row.get("control", {}).get("status") not in CONTROL_STATUSES or str(row.get("code_hash")) not in allowed_code:
            _fail(errors, f"A control status/code differs {key}")
        _verify_file_binding({"path": row.get("matrix_path"), "sha256": row.get("matrix_hash")}, f"A control matrix/{key}", errors)
    if control_seen != expected_controls:
        _fail(errors, f"A control identity set differs: missing={sorted(expected_controls - control_seen)} extra={sorted(control_seen - expected_controls)}")
    replay = a.get("v15_replay", {})
    if replay.get("status") != "PASS" or int(replay.get("comparison_count", -1)) != 24 or _float(replay.get("max_abs_error", float("inf")), "replay error") > _float(replay.get("tolerance_max_abs", 1e-4), "replay tolerance"):
        _fail(errors, "A v15 replay does not pass")
    asset_map = {int(row["sample_id"]): row for row in assets.get("records", []) if isinstance(row, Mapping)}
    for sample_id in config.SAMPLE_IDS:
        for source, video_type in (("N", "V_N"), ("T", "V_T")):
            current = result.get((sample_id, video_type, "ORIGINAL"))
            historical = asset_map.get(sample_id, {}).get("sources", {}).get(source, {}).get("matrix", {})
            if current is None or not isinstance(historical, Mapping):
                _fail(errors, f"A replay binding is missing {sample_id}/{source}")
                continue
            try:
                old = np.load(_resolve_path(historical.get("path")), allow_pickle=False)
                new = np.load(_resolve_path(current.get("matrix_path")), allow_pickle=False)
                if old.shape != new.shape or float(np.max(np.abs(old.astype(np.float64) - new.astype(np.float64)))) > 1e-4:
                    _fail(errors, f"A replay matrix differs {sample_id}/{source}")
            except (OSError, ProtocolError, ValueError, TypeError) as exc:
                _fail(errors, f"A replay matrix unreadable {sample_id}/{source}: {exc}")
    return a, result


def _validate_response_independent(row: Mapping[str, Any], provenance: Mapping[str, Any], audio_row: Mapping[str, Any], asset: Mapping[str, Any], errors: list[str], label: str) -> None:
    request_binding = row.get("request")
    response_binding = row.get("response")
    request_path = _verify_file_binding(request_binding, f"{label}/request", errors)
    response_path = _verify_file_binding(response_binding, f"{label}/response", errors)
    if request_path is None or response_path is None:
        return
    request = _self(request_path, errors)
    response = _self(response_path, errors)
    body = dict(request)
    body.pop("artifact_sha256", None)
    if response.get("status") != "COMPLETE" or response.get("request_sha256") != canonical_sha256(body) or response.get("config_sha256") != request.get("config_hash") or response.get("output_path") != request.get("output_path"):
        _fail(errors, f"{label}: response is not bound to request/config/output")
    consumed = response.get("consumed_audio", {})
    if not isinstance(consumed, Mapping) or consumed.get("path") != audio_row.get("path") or consumed.get("file_sha256") != audio_row.get("file_sha256") or consumed.get("pcm_sha256") != audio_row.get("pcm_sha256"):
        _fail(errors, f"{label}: consumed audio differs")
    frontend = response.get("frontend", {})
    if not isinstance(frontend, Mapping) or any(frontend.get(key) in (None, "", "UNKNOWN") for key in ("path", "sha256", "tensor_sha256", "tensor_shape", "tensor_dtype", "pcm_sha256", "sample_start", "sample_end")):
        _fail(errors, f"{label}: frontend proof is incomplete")
    else:
        _verify_file_binding({"path": frontend.get("path"), "sha256": frontend.get("sha256")}, f"{label}/frontend", errors)
        if frontend.get("pcm_sha256") != audio_row.get("pcm_sha256") or int(frontend.get("sample_start", -1)) < 0 or int(frontend.get("sample_end", -1)) <= int(frontend.get("sample_start", -1)):
            _fail(errors, f"{label}: frontend PCM range/hash differs")
    rng = response.get("rng_reset", {})
    if not isinstance(rng, Mapping) or rng.get("all_streams_reset") is not True or not rng.get("initial_state_sha256") or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(set(rng.get("streams", []))):
        _fail(errors, f"{label}: RNG reset proof is incomplete")
    timing = response.get("chunk_to_frame_timing", {})
    if not isinstance(timing, Mapping) or not timing.get("rows") or timing.get("mapping_sha256") in (None, "", "UNKNOWN"):
        _fail(errors, f"{label}: timing proof is incomplete")
    elif timing.get("path") is not None:
        _verify_file_binding({"path": timing.get("path"), "sha256": timing.get("sha256")}, f"{label}/timing", errors)
    expected_loaded = sorted(str(key) for key in provenance.get("components", {}))
    if sorted(str(item) for item in response.get("loaded_components", [])) != expected_loaded:
        _fail(errors, f"{label}: loaded component set differs")
    portrait = asset.get("portrait", {})
    if request.get("portrait", {}).get("sha256") != portrait.get("file_sha256") or request.get("audio", {}).get("pcm_sha256") != audio_row.get("pcm_sha256"):
        _fail(errors, f"{label}: request asset identity differs")


def _validate_generation_row(row: Mapping[str, Any], provenance: Mapping[str, Any], audio: Mapping[tuple[int, str, str], Mapping[str, Any]], assets: Mapping[int, Mapping[str, Any]], errors: list[str], label: str, paths: config.RunPaths) -> None:
    sid = int(row.get("id", -1))
    source = str(row.get("source"))
    condition = str(row.get("driver_condition"))
    audio_row = audio.get((sid, source, condition))
    asset = assets.get(sid, {})
    if audio_row is None:
        _fail(errors, f"{label}: audio binding is missing")
        return
    if row.get("pcm_hash") != audio_row.get("pcm_sha256") or row.get("pcm_path") != audio_row.get("path"):
        _fail(errors, f"{label}: PCM identity differs")
    if row.get("portrait_hash") != asset.get("portrait", {}).get("file_sha256"):
        _fail(errors, f"{label}: portrait identity differs")
    if row.get("model_hash") != _provenance_model_hash(provenance) or row.get("code_hash") != _expected_b_generation_code_hash():
        _fail(errors, f"{label}: model/code provenance differs")
    _validate_video_identity(row, errors, label, allow_audio=True)
    _validate_response_independent(row, provenance, audio_row, asset, errors, label)


def _validate_feature_and_matrix(row: Mapping[str, Any], errors: list[str], label: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    visual = _load_feature(row.get("visual_feature_path"), row.get("visual_feature_hash"), errors, f"{label}/visual_feature")
    audio = _load_feature(row.get("audio_feature_path"), row.get("audio_feature_hash"), errors, f"{label}/audio_feature")
    matrix_path = _verify_file_binding({"path": row.get("matrix_path"), "sha256": row.get("matrix_hash")}, f"{label}/matrix", errors)
    matrix = None
    if matrix_path is not None:
        try:
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False))
            if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or not np.isfinite(matrix).all():
                raise ProtocolError(f"matrix shape/content is invalid: {matrix.shape}")
        except (OSError, ValueError, ProtocolError) as exc:
            _fail(errors, f"{label}/matrix: {exc}")
    if visual is not None and audio is not None and matrix is not None:
        try:
            expected = _independent_distance_matrix(visual, audio)
            if expected.shape != matrix.shape or not np.allclose(expected, matrix, atol=1e-4, rtol=1e-6):
                _fail(errors, f"{label}: matrix is not independently reproducible")
        except ProtocolError as exc:
            _fail(errors, f"{label}: {exc}")
    return visual, audio


def _validate_b_and_perception(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], errors: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = _self(paths.generation / "manifest.json", errors)
    provenance = _self(paths.provenance, errors)
    audio = {(int(row["sample_id"]), str(row["source"]), str(row["condition"])): row for row in audio_manifest.get("records", []) if isinstance(row, Mapping)}
    asset_map = {int(row["sample_id"]): row for row in assets.get("records", []) if isinstance(row, Mapping)}
    expected_generation = {(sid, source, driver, seed, "main") for sid in config.SAMPLE_IDS for source in ("N", "T") for driver in config.B_DRIVERS for seed in config.SEEDS}
    expected_repeats = {(sid, source, "A0", 42, "repeat1") for sid in config.CONTROL_IDS for source in ("N", "T")}
    generation_rows = generation.get("videos", [])
    generation_keys = {(int(row.get("id", -1)), str(row.get("source")), str(row.get("driver_condition")), int(row.get("seed", -1)), str(row.get("role", "main"))) for row in generation_rows if row.get("stage") == "B_GENERATION"}
    repeat_keys = {(int(row.get("id", -1)), str(row.get("source")), str(row.get("driver_condition")), int(row.get("seed", -1)), str(row.get("role"))) for row in generation_rows if row.get("stage") == "B_GENERATION_REPEAT"}
    if generation_keys != expected_generation or repeat_keys != expected_repeats:
        _fail(errors, "B generation identity set differs")
    generation_status = str(generation.get("status"))
    if generation_status == "DEPENDENCY_BLOCKED":
        if (
            int(generation.get("science_video_count", -1)) != 0
            or int(generation.get("repeat_video_count", -1)) != 0
            or int(generation.get("planned_science_video_count", -1)) != config.EXPECTED_B_VIDEOS
            or int(generation.get("planned_repeat_video_count", -1)) != config.EXPECTED_B_REPEATS
            or int(generation.get("blocked_video_count", -1)) != config.EXPECTED_B_VIDEOS + config.EXPECTED_B_REPEATS
            or int(generation.get("failed_video_count", -1)) != 0
            or int(generation.get("resource_wait_count", -1)) != 0
            or len(generation_rows) != config.EXPECTED_B_VIDEOS + config.EXPECTED_B_REPEATS
        ):
            _fail(errors, "blocked B generation counts are dishonest")
        if any(row.get("status") != "DEPENDENCY_BLOCKED" for row in generation_rows):
            _fail(errors, "blocked B generation contains a non-blocked cell")
    elif generation_status == "COMPLETE":
        if len(generation_rows) != config.EXPECTED_B_VIDEOS + config.EXPECTED_B_REPEATS or int(generation.get("science_video_count", -1)) != config.EXPECTED_B_VIDEOS or int(generation.get("repeat_video_count", -1)) != config.EXPECTED_B_REPEATS:
            _fail(errors, "B generation complete counts differ")
        seen_paths: set[str] = set()
        for row in generation_rows:
            if str(row.get("path")) in seen_paths:
                _fail(errors, "B generation reuses one output path for two cells")
            seen_paths.add(str(row.get("path")))
            _validate_generation_row(row, provenance, audio, asset_map, errors, f"B generation/{row.get('cell_id')}", paths)
        for sid in config.CONTROL_IDS:
            for source in ("N", "T"):
                main = next((row for row in generation_rows if row.get("stage") == "B_GENERATION" and int(row.get("id")) == sid and row.get("source") == source and row.get("driver_condition") == "A0" and int(row.get("seed")) == 42), None)
                repeat = next((row for row in generation_rows if row.get("stage") == "B_GENERATION_REPEAT" and int(row.get("id")) == sid and row.get("source") == source), None)
                if main and repeat and str(main.get("path")) == str(repeat.get("path")):
                    _fail(errors, f"repeat path aliases main path {sid}/{source}")
    else:
        _fail(errors, f"B generation has no explicit complete/blocked status: {generation_status}")

    crossed = _self(paths.crossed / "manifest.json", errors)
    score_rows = crossed.get("cells", [])
    controls = crossed.get("controls", [])
    expected_scores = {(sid, source, seed, driver, condition) for sid in config.SAMPLE_IDS for source in ("N", "T") for seed in config.SEEDS for driver, condition in (("A0", "A0"), ("A0", "NOISE"), ("NOISE", "A0"), ("NOISE", "NOISE"), ("A0", "DENOISE"), ("DENOISE", "A0"), ("DENOISE", "DENOISE"))}
    score_keys = {(int(row.get("id", -1)), str(row.get("source")), int(row.get("seed", -1)), str(row.get("driver_condition")), str(row.get("eval_condition"))) for row in score_rows}
    expected_controls = {(sid, source, control) for sid in config.CONTROL_IDS for source in ("N", "T") for control in ("REPEAT", "DELAY_PLUS_200MS")}
    control_keys = {(int(row.get("id", -1)), str(row.get("source")), str(row.get("driver_condition"))) for row in controls}
    if score_keys != expected_scores or control_keys != expected_controls:
        _fail(errors, "B score/control identity set differs")
    crossed_status = str(crossed.get("status"))
    if crossed_status == "DEPENDENCY_BLOCKED":
        if (
            int(crossed.get("science_cell_count", -1)) != 0
            or int(crossed.get("control_cell_count", -1)) != 0
            or int(crossed.get("planned_science_cell_count", -1)) != config.EXPECTED_B_SCIENCE
            or int(crossed.get("planned_control_cell_count", -1)) != config.EXPECTED_B_CONTROLS
            or int(crossed.get("blocked_cell_count", -1)) != config.EXPECTED_B_SCIENCE + config.EXPECTED_B_CONTROLS
            or int(crossed.get("failed_cell_count", -1)) != 0
            or int(crossed.get("resource_wait_count", -1)) != 0
            or len(score_rows) != config.EXPECTED_B_SCIENCE
            or len(controls) != config.EXPECTED_B_CONTROLS
        ):
            _fail(errors, "blocked B score counts are dishonest")
        if any(row.get("status") != "DEPENDENCY_BLOCKED" for row in [*score_rows, *controls]):
            _fail(errors, "blocked B scores contain a non-blocked cell")
    elif crossed_status == "COMPLETE":
        if len(score_rows) != config.EXPECTED_B_SCIENCE or len(controls) != config.EXPECTED_B_CONTROLS or int(crossed.get("science_cell_count", -1)) != config.EXPECTED_B_SCIENCE or int(crossed.get("control_cell_count", -1)) != config.EXPECTED_B_CONTROLS:
            _fail(errors, "B score complete counts differ")
        generation_map = {(int(row["id"]), str(row["source"]), str(row["driver_condition"]), int(row["seed"])): row for row in generation_rows if row.get("stage") == "B_GENERATION"}
        score_map: dict[tuple[int, str, int, str, str], Mapping[str, Any]] = {}
        matrix_by_cell: dict[str, np.ndarray] = {}
        source_pcm_cache: dict[str, np.ndarray] = {}
        for row in score_rows:
            label = f"B score/{row.get('cell_id')}"
            if any(field not in row for field in REQUIRED_B_FIELDS) or row.get("status") != "COMPLETE" or row.get("code_hash") != _expected_b_score_code_hash():
                _fail(errors, f"{label}: required field/status/code differs")
                continue
            key = (int(row["id"]), str(row["source"]), int(row["seed"]), str(row["driver_condition"]), str(row["eval_condition"]))
            if key in score_map:
                _fail(errors, f"duplicate B score {key}")
            score_map[key] = row
            video = generation_map.get((key[0], key[1], key[3], key[2]))
            if video is None or row.get("video_cell_id") != video.get("cell_id") or row.get("video_hash") != video.get("video_hash"):
                _fail(errors, f"{label}: video identity differs")
            _validate_b_audio(
                row,
                audio.get((key[0], key[1], str(row.get("eval_condition")))),
                errors,
                label,
                source_pcm_cache,
            )
            crop = _verify_file_binding({"path": row.get("cropped_video_path"), "sha256": row.get("cropped_video_hash")}, f"{label}/crop", errors)
            if crop is not None:
                _validate_video_identity({"path": str(crop), "file_sha256": file_sha256(crop), "video_hash": row.get("cropped_video_hash"), "pixel_sha256": row.get("cropped_pixel_sha256"), "pts_sha256": row.get("cropped_pts_sha256"), "frame_count": row.get("cropped_frame_count", 5), "pts": {}}, errors, f"{label}/crop", allow_audio=True)
            _validate_roi(row, errors, label, baseline=generation_map.get((key[0], key[1], "A0", key[2])))
            visual, auditory = _validate_feature_and_matrix(row, errors, label)
            matrix = None
            try:
                matrix = np.load(_resolve_path(row.get("matrix_path")), allow_pickle=False)
                matrix_by_cell[str(row["cell_id"])] = np.asarray(matrix)
            except (OSError, ProtocolError, ValueError, TypeError):
                pass
            if matrix is not None:
                try:
                    metrics = _independent_curve(matrix, row.get("support_rows", []))
                    _assert_close(errors, metrics, row.get("metrics", {}), f"{label}/metrics", tolerance=1e-5)
                except (ProtocolError, TypeError, ValueError) as exc:
                    _fail(errors, f"{label}/metrics: {exc}")
        for scope in {(int(row.get("id")), str(row.get("source"))) for row in score_rows}:
            family = [row for row in score_rows if int(row.get("id")) == scope[0] and str(row.get("source")) == scope[1]]
            try:
                expected_time, expected_rows = _independent_common_time_support(family, matrix_by_cell, trim_rows=config.VSHIFT)
            except (ProtocolError, TypeError, ValueError) as exc:
                _fail(errors, f"B common fourteen-cell support cannot be recomputed {scope}: {exc}")
                expected_time, expected_rows = [], {}
            expected_hash = canonical_sha256({"rows": expected_time, "rule": "common fourteen-cell valid time intersection after vshift", "scope": [str(scope[0]), str(scope[1]), "seeds=42,43"]})
            row_support_ok = all(
                [int(item) for item in row.get("support_time_frames", [])] == expected_time
                and [int(item) for item in row.get("support_rows", [])] == expected_rows.get(str(row.get("cell_id")), [])
                for row in family
            )
            if not row_support_ok or {str(row.get("support_hash")) for row in family} != {expected_hash} or len(expected_time) < config.MIN_INTERIOR_ROWS or len(family) != 14:
                _fail(errors, f"B common fourteen-cell support differs {scope}")
        support_manifest = _self(paths.crossed / "support_manifest.json", errors)
        if support_manifest.get("status") != "COMPLETE" or len(support_manifest.get("groups", [])) != len(config.SAMPLE_IDS) * 2:
            _fail(errors, "B support manifest is incomplete")
        manifest_groups = {
            (int(group.get("scope", {}).get("id", -1)), str(group.get("scope", {}).get("source"))): group
            for group in support_manifest.get("groups", [])
            if isinstance(group, Mapping)
        }
        for scope in {(sid, source) for sid in config.SAMPLE_IDS for source in ("N", "T")}:
            group = manifest_groups.get(scope)
            family = [row for row in score_rows if int(row.get("id")) == scope[0] and str(row.get("source")) == scope[1]]
            try:
                expected_time, expected_rows = _independent_common_time_support(family, matrix_by_cell, trim_rows=config.VSHIFT)
            except (ProtocolError, TypeError, ValueError):
                expected_time, expected_rows = [], {}
            expected_hash = canonical_sha256({"rows": expected_time, "rule": "common fourteen-cell valid time intersection after vshift", "scope": [str(scope[0]), str(scope[1]), "seeds=42,43"]})
            if not isinstance(group, Mapping) or group.get("support_time_frames") != expected_time or group.get("support_hash") != expected_hash or group.get("support_rows_by_cell") != expected_rows:
                _fail(errors, f"B support manifest content differs {scope}")
        decompositions = crossed.get("four_cell_decomposition", [])
        expected_decomp = {(sid, source, seed, driver) for sid in config.SAMPLE_IDS for source in ("N", "T") for seed in config.SEEDS for driver in ("NOISE", "DENOISE")}
        seen_decomp: set[tuple[int, str, int, str]] = set()
        for item in decompositions:
            key = (int(item.get("id", -1)), str(item.get("source")), int(item.get("seed", -1)), str(item.get("driver")))
            seen_decomp.add(key)
            cells = [score_map.get((key[0], key[1], key[2], "A0", "A0")), score_map.get((key[0], key[1], key[2], "A0", key[3])), score_map.get((key[0], key[1], key[2], key[3], "A0")), score_map.get((key[0], key[1], key[2], key[3], key[3]))]
            if any(cell is None for cell in cells):
                _fail(errors, f"B decomposition cell missing {key}")
                continue
            values = [float(cell["metrics"]["sync_c"]) for cell in cells]
            expected = {"generation": values[2] - values[0], "evaluation": values[1] - values[0], "interaction": values[3] - values[2] - values[1] + values[0], "total": values[3] - values[0]}
            for field, value in expected.items():
                if abs(_float(item.get(field), f"decomposition/{field}") - value) > 1e-6:
                    _fail(errors, f"B decomposition arithmetic differs {key}/{field}")
        if seen_decomp != expected_decomp or len(decompositions) != 96:
            _fail(errors, f"B decomposition count/identity differs: {len(decompositions)}")
        expected_primary: dict[str, float] = {}
        for sid in config.SAMPLE_IDS:
            for source in ("N", "T"):
                for driver in ("NOISE", "DENOISE"):
                    values = []
                    for seed in config.SEEDS:
                        q00 = score_map.get((sid, source, seed, "A0", "A0"))
                        q10 = score_map.get((sid, source, seed, driver, "A0"))
                        if q00 is None or q10 is None:
                            continue
                        values.append(float(q10["metrics"]["sync_c"] - q00["metrics"]["sync_c"]))
                    if len(values) == len(config.SEEDS):
                        expected_primary[f"{sid}:{source}:{driver}"] = float(np.mean(values))
            native = []
            for seed in config.SEEDS:
                n = score_map.get((sid, "N", seed, "A0", "A0")); t = score_map.get((sid, "T", seed, "A0", "A0"))
                if n is not None and t is not None:
                    native.append(float(t["metrics"]["sync_c"] - n["metrics"]["sync_c"]))
            if len(native) == len(config.SEEDS):
                expected_primary[f"{sid}:NATIVE"] = float(np.mean(native))
        for key, value in expected_primary.items():
            record = crossed.get("primary_records", {}).get(key)
            if not isinstance(record, Mapping) or abs(_float(record.get("value"), f"primary/{key}") - value) > 1e-6:
                _fail(errors, f"B primary record differs {key}")
        if set(crossed.get("primary_records", {})) != set(expected_primary):
            _fail(errors, "B primary record identity differs")
        for row in controls:
            key = (int(row.get("id")), str(row.get("source")), str(row.get("driver_condition")))
            if row.get("status") != "COMPLETE" or row.get("control", {}).get("status") not in CONTROL_STATUSES or row.get("code_hash") != _expected_b_score_code_hash():
                _fail(errors, f"B control status/code differs {key}")
            baseline_generation = generation_map.get((key[0], key[1], "A0", 42))
            if key[2] == "REPEAT":
                candidate_generation = next((item for item in generation_rows if item.get("stage") == "B_GENERATION_REPEAT" and int(item.get("id", -1)) == key[0] and str(item.get("source")) == key[1]), None)
                expected_shift = 0
            else:
                candidate_generation = baseline_generation
                expected_shift = 3200
            baseline_score = score_map.get((key[0], key[1], 42, "A0", "A0"))
            if candidate_generation is None or baseline_generation is None or row.get("video_cell_id") != candidate_generation.get("cell_id") or row.get("video_hash") != candidate_generation.get("video_hash"):
                _fail(errors, f"B control video identity differs {key}")
            _validate_b_audio(row, audio.get((key[0], key[1], "A0")), errors, f"B control/{key}/audio", source_pcm_cache, shift_samples=expected_shift)
            crop = _verify_file_binding({"path": row.get("cropped_video_path"), "sha256": row.get("cropped_video_hash")}, f"B control/{key}/crop", errors)
            if crop is not None:
                _validate_video_identity({"path": str(crop), "file_sha256": file_sha256(crop), "video_hash": row.get("cropped_video_hash"), "pixel_sha256": row.get("cropped_pixel_sha256"), "pts_sha256": row.get("cropped_pts_sha256"), "frame_count": row.get("cropped_frame_count", 5), "pts": {}}, errors, f"B control/{key}/crop", allow_audio=True)
            _validate_roi(row, errors, f"B control/{key}", baseline=baseline_generation)
            visual, auditory = _validate_feature_and_matrix(row, errors, f"B control/{key}")
            try:
                matrix = np.load(_resolve_path(row.get("matrix_path")), allow_pickle=False)
                expected = _independent_distance_matrix(visual, auditory) if visual is not None and auditory is not None else None
                if expected is not None and not np.allclose(expected, matrix, atol=1e-4, rtol=1e-6):
                    _fail(errors, f"B control matrix differs {key}")
                if baseline_score is None:
                    _fail(errors, f"B control baseline score is missing {key}")
                    continue
                baseline_matrix = np.asarray(np.load(_resolve_path(baseline_score["matrix_path"]), allow_pickle=False))
                control_scope = {"cell_id": str(row.get("cell_id")), "time_start_frame": int(row.get("time_start_frame", 0))}
                expected_time, expected_rows = _independent_common_time_support(
                    [baseline_score, control_scope],
                    {str(baseline_score["cell_id"]): baseline_matrix, str(row["cell_id"]): np.asarray(matrix)},
                    trim_rows=config.VSHIFT + 5,
                )
                expected_base_support = expected_rows.get(str(baseline_score["cell_id"]), [])
                expected_candidate_support = expected_rows.get(str(row["cell_id"]), [])
                expected_support_hash = canonical_sha256({"rows": expected_time, "rule": "control support trims five rows beyond vshift on shared source clock", "scope": [str(key[0]), str(key[1])]})
                if row.get("support_time_frames") != expected_time or row.get("support_rows") != expected_base_support or row.get("candidate_support_rows") != expected_candidate_support or row.get("support_hash") != expected_support_hash or len(expected_time) < config.MIN_INTERIOR_ROWS:
                    _fail(errors, f"B control support differs {key}")
                support = expected_base_support
                candidate_support = expected_candidate_support
                control = row.get("control")
                if not isinstance(control, Mapping):
                    _fail(errors, f"B control result is missing {key}")
                    continue
                if key[2] == "REPEAT":
                    base_metrics = _independent_curve(baseline_matrix, support)
                    repeat_metrics = _independent_curve(matrix, candidate_support)
                    delta_c = float(repeat_metrics["sync_c"] - base_metrics["sync_c"])
                    delta_d = float(repeat_metrics["sync_d"] - base_metrics["sync_d"])
                    offset_delta = int(repeat_metrics["official_offset"] - base_metrics["official_offset"])
                    aligned_error = float(np.max(np.abs(baseline_matrix[np.asarray(support)] - np.asarray(matrix)[np.asarray(candidate_support)])))
                    expected_status = "IDENTITY_PASS" if abs(delta_c) <= 0.05 and abs(delta_d) <= 0.05 and abs(offset_delta) <= 1 else "GENERATION_UNSTABLE"
                    if control.get("status") != expected_status:
                        _fail(errors, f"repeat control status is not independently reproducible {key}")
                    for field, value in (("delta_sync_c", delta_c), ("delta_sync_d", delta_d), ("offset_delta", offset_delta), ("aligned_matrix_max_abs_error", aligned_error)):
                        try:
                            if abs(float(control.get(field)) - float(value)) > 1e-5:
                                _fail(errors, f"repeat control field differs {key}/{field}")
                        except (TypeError, ValueError):
                            _fail(errors, f"repeat control field is invalid {key}/{field}")
                elif key[2] == "DELAY_PLUS_200MS":
                    base_metrics = _independent_curve(baseline_matrix, support)
                    shifted_metrics = _independent_curve(matrix, candidate_support)
                    expected_delta = -5
                    observed_delta = int(shifted_metrics["official_offset"] - base_metrics["official_offset"])
                    base_distance = float(np.mean(baseline_matrix[np.asarray(support), int(base_metrics["min_index"])]))
                    shifted_distance = float(np.mean(np.asarray(matrix)[np.asarray(candidate_support), int(base_metrics["min_index"])]))
                    expected_min_index = int(base_metrics["min_index"]) - expected_delta
                    boundary = bool(int(base_metrics["min_index"]) in (0, config.LAG_COUNT - 1) or expected_min_index < 0 or expected_min_index >= config.LAG_COUNT)
                    expected_status = "CONTROL_UNINFORMATIVE" if boundary else ("DELAY_DETECTED" if abs(observed_delta - expected_delta) <= 1 and shifted_distance > base_distance else "CONTROL_FAILED")
                    if control.get("status") != expected_status:
                        _fail(errors, f"delay control status is not independently reproducible {key}")
                    for field, value in (("expected_offset_delta", expected_delta), ("observed_offset_delta", observed_delta), ("baseline_column", int(base_metrics["min_index"])), ("baseline_column_distance", base_distance), ("shifted_column_distance", shifted_distance)):
                        try:
                            if abs(float(control.get(field)) - float(value)) > 1e-5:
                                _fail(errors, f"delay control field differs {key}/{field}")
                        except (TypeError, ValueError):
                            _fail(errors, f"delay control field is invalid {key}/{field}")
            except (OSError, ProtocolError, ValueError, TypeError, KeyError) as exc:
                _fail(errors, f"B control cannot be recomputed {key}: {exc}")
        expected_gate = "PASS" if len(controls) == config.EXPECTED_B_CONTROLS and all(row.get("control", {}).get("status") in {"IDENTITY_PASS", "DELAY_DETECTED"} for row in controls) else "CONTROL_LIMITED"
        if crossed.get("control_gate") != expected_gate:
            _fail(errors, "B control gate is not independently reproducible")
    elif crossed_status != "NOT_RUN":
        _fail(errors, f"B crossed scoring has no explicit complete/blocked status: {crossed_status}")

    package = _self(paths.perception / "package.json", errors)
    mapping = _self(paths.perception / "private_mapping.json", errors)
    if package.get("blind_schema_version") != 3 or int(package.get("sync_pair_count", -1)) != config.EXPECTED_SYNC_PAIRS or int(package.get("quality_pair_count", -1)) != config.EXPECTED_QUALITY_PAIRS or len(package.get("sync_pairs", [])) != config.EXPECTED_SYNC_PAIRS + 10 or len(package.get("quality_pairs", [])) != config.EXPECTED_QUALITY_PAIRS + 5:
        _fail(errors, "perception package counts/schema differ")
    public_forbidden = {"sample_id", "source", "seed", "driver", "hidden_repeat", "source_pair_id", "left_role", "right_role", "left_source_path", "right_source_path", "audio_source_path", "blind_mapping", "private_mapping"}
    mapping_rows = mapping.get("mapping", [])
    if len(mapping_rows) != config.EXPECTED_SYNC_PAIRS + 10 + config.EXPECTED_QUALITY_PAIRS + 5:
        _fail(errors, "private perception mapping count differs")
    private_by_id = {str(row.get("anonymous_id")): row for row in mapping_rows if isinstance(row, Mapping)}
    for kind, rows, expected, hidden in (("sync", package.get("sync_pairs", []), config.EXPECTED_SYNC_PAIRS, 10), ("quality", package.get("quality_pairs", []), config.EXPECTED_QUALITY_PAIRS, 5)):
        if len(rows) != expected + hidden or len({str(row.get("anonymous_id")) for row in rows}) != expected + hidden:
            _fail(errors, f"perception {kind} identity/count differs")
        for row in rows:
            if public_forbidden & set(row):
                _fail(errors, f"perception {kind} leaks operator fields: {row.get('anonymous_id')}")
            if str(row.get("anonymous_id")) not in private_by_id:
                _fail(errors, f"perception {kind} lacks private join: {row.get('anonymous_id')}")
            for key in ("left_public", "right_public"):
                path = row.get(key)
                if row.get("available") is True:
                    hash_key = f"{key.removesuffix('_public')}_sha256"
                    if not isinstance(path, str) or not Path(path).is_file():
                        _fail(errors, f"perception media missing {kind}/{row.get('anonymous_id')}/{key}")
                    elif row.get(hash_key) != file_sha256(Path(path)):
                        _fail(errors, f"perception media hash differs {kind}/{row.get('anonymous_id')}/{key}")
                    elif kind == "sync":
                        try:
                            probe = ffprobe_json(Path(path))
                            if any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []) if isinstance(stream, Mapping)):
                                _fail(errors, f"perception sync video contains audio {row.get('anonymous_id')}")
                        except (OSError, ProtocolError, ValueError):
                            _fail(errors, f"perception sync video is unreadable {row.get('anonymous_id')}")
                elif package.get("status") == "COMPLETE":
                    _fail(errors, f"COMPLETE perception package has unavailable media {row.get('anonymous_id')}")
            if kind == "sync" and row.get("available") is True and (
                not isinstance(row.get("audio_public"), str)
                or not Path(str(row["audio_public"])).is_file()
                or row.get("audio_sha256") != file_sha256(Path(str(row["audio_public"])))
            ):
                _fail(errors, f"perception sync audio binding differs {row.get('anonymous_id')}")
    for path, expected_header in ((paths.perception / "sync_ratings_template.csv", "anonymous_id"), (paths.perception / "quality_ratings_template.csv", "anonymous_id")):
        if not path.is_file():
            _fail(errors, f"missing perception template {path}")
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                if expected_header not in next(csv.reader(handle), []):
                    _fail(errors, f"perception template header differs {path}")
    perception_analysis = _self(paths.perception / "analysis.json", errors)
    if perception_analysis.get("package_sha256") != file_sha256(paths.perception / "package.json"):
        _fail(errors, "perception analysis is not bound to current package")
    return crossed, package


def _validate_analysis(paths: config.RunPaths, cells: Mapping[tuple[int, str, str], Mapping[str, Any]], assets: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    analysis = _self(paths.analysis / "summary.json", errors)
    if not analysis:
        return analysis
    for field, path in (("a_manifest_sha256", paths.fixed_video / "a_manifest.json"), ("audio_manifest_sha256", paths.audio / "manifest.json"), ("assets_sha256", paths.audit / "assets.json")):
        if analysis.get(field) != file_sha256(path):
            _fail(errors, f"analysis binding differs: {field}")
    crossed_path = paths.crossed / "manifest.json"
    if analysis.get("crossed_manifest_sha256") != (file_sha256(crossed_path) if crossed_path.is_file() else None):
        _fail(errors, "analysis crossed binding differs")
    bootstrap = analysis.get("bootstrap_indices", {})
    bootstrap_path = _verify_file_binding(bootstrap, "analysis bootstrap", errors)
    if bootstrap_path is not None:
        expected_indices = _bootstrap_indices()
        try:
            actual = np.load(bootstrap_path, allow_pickle=False)
            if actual.shape != expected_indices.shape or not np.array_equal(actual, expected_indices):
                _fail(errors, "analysis bootstrap indices differ from PCG64 seed")
        except (OSError, ValueError) as exc:
            _fail(errors, f"analysis bootstrap unreadable: {exc}")
    primary = analysis.get("primary", {}).get("summaries", {})
    specs_a = (("A_N_DENOISE_E", "V_N", "A0", "DENOISE", 1), ("A_T_NOISE_E_HARM", "V_T", "NOISE", "A0", 1), ("A_R_DENOISE_E", "R", "A0", "DENOISE", 1))
    group_values: dict[str, dict[str, float]] = {}
    for name, video_type, left, right, direction in specs_a:
        values: dict[str, float] = {}
        for sid in config.SAMPLE_IDS:
            first = cells.get((sid, video_type, left)); second = cells.get((sid, video_type, right))
            if first is None or second is None:
                continue
            try:
                left_metrics = _independent_curve(np.load(_resolve_path(first["matrix_path"]), allow_pickle=False), first["support_rows"])
                right_metrics = _independent_curve(np.load(_resolve_path(second["matrix_path"]), allow_pickle=False), second["support_rows"])
                values[str(first["source_group"])] = direction * (right_metrics["sync_c"] - left_metrics["sync_c"])
            except (OSError, ProtocolError, ValueError, TypeError) as exc:
                _fail(errors, f"analysis {name}/{sid}: {exc}")
        group_values[name] = values
        _assert_close(errors, _independent_bootstrap(values, _bootstrap_indices(), name), primary.get(name, {}), f"analysis.primary.{name}")
    crossed = _self(paths.crossed / "manifest.json", errors)
    if crossed.get("status") == "COMPLETE":
        score_map = {(int(row["id"]), str(row["source"]), int(row["seed"]), str(row["driver_condition"]), str(row["eval_condition"])): row for row in crossed.get("cells", [])}
        b_values: dict[str, dict[str, float]] = {name: {} for name in ("B_N_DENOISE_G", "B_T_NOISE_G_HARM", "B_NATIVE_FRESH")}
        for sid in config.SAMPLE_IDS:
            for name, source, driver, direction in (("B_N_DENOISE_G", "N", "DENOISE", 1), ("B_T_NOISE_G_HARM", "T", "NOISE", -1)):
                values = []
                for seed in config.SEEDS:
                    q00 = score_map.get((sid, source, seed, "A0", "A0")); q10 = score_map.get((sid, source, seed, driver, "A0"))
                    if q00 and q10:
                        values.append(direction * (float(q10["metrics"]["sync_c"]) - float(q00["metrics"]["sync_c"])))
                if len(values) == len(config.SEEDS):
                    b_values[name][str(next(row["source_group"] for row in crossed["cells"] if int(row["id"]) == sid))] = float(np.mean(values))
            native = []
            for seed in config.SEEDS:
                n = score_map.get((sid, "N", seed, "A0", "A0")); t = score_map.get((sid, "T", seed, "A0", "A0"))
                if n and t:
                    native.append(float(t["metrics"]["sync_c"]) - float(n["metrics"]["sync_c"]))
            if len(native) == len(config.SEEDS):
                b_values["B_NATIVE_FRESH"][str(next(row["source_group"] for row in crossed["cells"] if int(row["id"]) == sid))] = float(np.mean(native))
        for name, values in b_values.items():
            _assert_close(errors, _independent_bootstrap(values, _bootstrap_indices(), name), primary.get(name, {}), f"analysis.primary.{name}")
    elif any(primary.get(name, {}).get("status") == "COMPLETE" for name in ("B_N_DENOISE_G", "B_T_NOISE_G_HARM", "B_NATIVE_FRESH")):
        _fail(errors, "analysis reports complete B effect while B manifest is blocked")
    return analysis


def _check_resource_artifact(path: Path, errors: list[str]) -> dict[str, Any]:
    value = _self(path, errors)
    for key in ("disk_before", "gpu_peak_budget_bytes", "disk_temp_budget_bytes", "disk_persistent_budget_bytes"):
        if key not in value:
            _fail(errors, f"resource plan misses {key}")
    return value


def validate_stage(paths: config.RunPaths) -> dict[str, Any]:
    errors: list[str] = []
    protocol = _self(paths.protocol, errors)
    claims = _self(paths.claims, errors)
    provenance = _self(paths.provenance, errors)
    assets = _validate_audit(paths, errors)
    audio = _validate_audio(paths, assets, errors) if assets else {}
    fixed, _ = _validate_fixed(paths, errors)
    a, cells = _validate_a_cells(paths, assets, audio, fixed, errors) if fixed else ({}, {})
    _validate_analysis(paths, cells, assets, errors) if cells else {}
    crossed, package = _validate_b_and_perception(paths, assets, audio, errors)
    resource = _check_resource_artifact(paths.audit / "resource_plan.json", errors)
    final = _self(paths.final, errors)
    if paths.report.is_file() and final.get("report_sha256") != file_sha256(paths.report):
        _fail(errors, "final report hash differs")
    generation = _self(paths.generation / "manifest.json", errors)
    all_science_complete = generation.get("status") == "COMPLETE" and crossed.get("status") == "COMPLETE" and package.get("status") == "COMPLETE"
    if final.get("automatic_complete") is True and not all_science_complete:
        _fail(errors, "final falsely claims automatic completion")
    if protocol.get("protocol_id") != config.PROTOCOL_ID or claims.get("protocol_id") != config.PROTOCOL_ID or provenance.get("protocol_id") != config.PROTOCOL_ID:
        _fail(errors, "top-level protocol IDs differ")
    expected_code = {str(path.relative_to(config.REPO)): file_sha256(path) for path in config.package_files() if path.is_file()}
    if protocol.get("frozen_code") != expected_code or protocol.get("code_snapshot_refrozen_before_scoring") is not True:
        _fail(errors, "new code snapshot is stale or was not frozen before B scoring")
    continuation_spec = protocol.get("continuation_spec")
    if isinstance(continuation_spec, str):
        spec_path = Path(continuation_spec) / "specs/tts-native-gain-completion/spec.md"
        if not spec_path.is_file() or protocol.get("continuation_spec_sha256") != file_sha256(spec_path):
            _fail(errors, "continuation spec snapshot differs")
    status = "valid" if not errors else "invalid"
    engineering = "AUTOMATIC_COMPLETE" if status == "valid" and all_science_complete and final.get("automatic_complete") is True else "PARTIAL"
    result = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": status, "engineering_status": engineering, "errors": errors, "independent_recompute": "B matrices, endpoint curves, four-cell decomposition, controls, and six bootstrap families are recomputed without importing producer analysis/runner decisions", "stages": {"audit": "valid" if assets.get("status") == "COMPLETE" else "invalid", "A": a.get("status", "NOT_RUN"), "B_generation": generation.get("status", "NOT_RUN"), "B_scores": crossed.get("status", "NOT_RUN"), "perception": package.get("status", "NOT_RUN")}, "resource": resource}
    return write_self_hashed_json(paths.validation, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="independently validate a native-gain attribution run")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_stage(config.RunPaths(args.root))
    print(result["status"])
    for error in result.get("errors", [])[:20]:
        print(error, file=sys.stderr)
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
