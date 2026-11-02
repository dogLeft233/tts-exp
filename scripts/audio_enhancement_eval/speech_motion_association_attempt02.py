#!/usr/bin/env python3
"""Descriptive fixed-clock association of natural-speech energy and mouth motion.

This companion reads the frozen natural PCM and the visual-motion sidecar. It
does not rerender video, modify timing, rescore SyncNet, or select a lag. The
zero-lag association is the declared descriptive value; the fixed −15…+15
curve is reported only as a timing sensitivity profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cross-tfg-speech-motion-association-v1"
FRAME_RATE_NUM = 25
FRAME_RATE_DEN = 1
SAMPLE_RATE_HZ = 16_000
MINIMUM_PAIRS = 30
LAG_OFFSETS = tuple(range(-15, 16))
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
VISUAL_ARRAYS = (
    "frame_index_int32", "timestamp_num_int64", "timestamp_den_int64",
    "tracking_valid_bool", "mouth_aperture_eye_normalized_float64",
    "aperture_velocity_eye_normalized_per_s_float64", "aperture_velocity_valid_bool",
)


class AssociationError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], self_field: str) -> str:
    return _sha_bytes(_canonical_json({key: item for key, item in value.items() if key != self_field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssociationError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssociationError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve(root: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AssociationError(f"{where} must be a nonempty path")
    candidate = Path(value)
    return candidate.resolve(strict=True) if candidate.is_absolute() else (root / candidate).resolve(strict=True)


def _read_f32_wav(path: Path) -> tuple[np.ndarray, str, int]:
    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AssociationError(f"canonical natural PCM is not a RIFF/WAVE file: {path}")
    declared_end = int.from_bytes(raw[4:8], "little") + 8
    if declared_end > len(raw):
        raise AssociationError(f"truncated WAV payload: {path}")
    fmt: tuple[int, int, int, int, int] | None = None
    data_chunks: list[bytes] = []
    position = 12
    while position + 8 <= declared_end:
        chunk_id = raw[position:position + 4]
        chunk_size = int.from_bytes(raw[position + 4:position + 8], "little")
        start, end = position + 8, position + 8 + chunk_size
        if end > declared_end:
            raise AssociationError(f"truncated WAV chunk {chunk_id!r}: {path}")
        payload = raw[start:end]
        if chunk_id == b"fmt ":
            if len(payload) < 16:
                raise AssociationError(f"short WAV fmt chunk: {path}")
            format_code = int.from_bytes(payload[0:2], "little")
            channels = int.from_bytes(payload[2:4], "little")
            rate = int.from_bytes(payload[4:8], "little")
            align = int.from_bytes(payload[12:14], "little")
            bits = int.from_bytes(payload[14:16], "little")
            if format_code == 0xFFFE:
                if len(payload) < 40:
                    raise AssociationError(f"truncated extensible WAV fmt: {path}")
                format_code = int.from_bytes(payload[24:26], "little")
            fmt = format_code, channels, rate, bits, align
        elif chunk_id == b"data":
            data_chunks.append(payload)
        position = end + (chunk_size & 1)
    if fmt != (3, 1, SAMPLE_RATE_HZ, 32, 4) or not data_chunks:
        raise AssociationError(f"natural PCM must be mono 16 kHz IEEE float32: {path}")
    pcm_bytes = b"".join(data_chunks)
    if len(pcm_bytes) % 4:
        raise AssociationError(f"float32 WAV data is not sample aligned: {path}")
    samples = np.frombuffer(pcm_bytes, dtype=np.dtype("<f4"))
    if samples.size == 0 or not np.isfinite(samples).all():
        raise AssociationError(f"natural PCM is empty or non-finite: {path}")
    return samples, _sha_bytes(pcm_bytes), int(samples.size)


def _load_visual_arrays(root: Path, row: Mapping[str, Any], where: str) -> dict[str, np.ndarray]:
    ref = row.get("visual_npz")
    if not isinstance(ref, dict):
        raise AssociationError(f"{where}.visual_npz is missing")
    path = _resolve(root, ref.get("path"), f"{where}.visual_npz.path")
    if _sha_file(path) != ref.get("file_sha256"):
        raise AssociationError(f"{where}.visual_npz file SHA mismatch")
    metadata = ref.get("arrays")
    if not isinstance(metadata, dict):
        raise AssociationError(f"{where}.visual_npz array refs are missing")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in VISUAL_ARRAYS}
    except Exception as exc:
        raise AssociationError(f"cannot read visual trajectory arrays for {where}: {exc}") from exc
    for name, array in arrays.items():
        item = metadata.get(name)
        if not isinstance(item, dict):
            raise AssociationError(f"{where}.visual_npz lacks the {name} seal")
        if item.get("dtype") not in (array.dtype.str, array.dtype.name) or item.get("shape") != list(array.shape):
            raise AssociationError(f"{where}.visual_npz {name} dtype/shape mismatch")
        if item.get("array_sha256") != _sha_bytes(np.ascontiguousarray(array).tobytes(order="C")):
            raise AssociationError(f"{where}.visual_npz {name} array SHA mismatch")
    return arrays


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str]:
    if x.shape != y.shape:
        raise AssociationError("correlation inputs have different lengths")
    if x.size < MINIMUM_PAIRS:
        return None, "INELIGIBLE_SHORT_SUPPORT"
    xc = x.astype(np.float64, copy=False) - float(np.mean(x, dtype=np.float64))
    yc = y.astype(np.float64, copy=False) - float(np.mean(y, dtype=np.float64))
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if not math.isfinite(denom) or denom <= 0:
        return None, "UNDEFINED_CONSTANT_SERIES"
    value = float(np.dot(xc, yc) / denom)
    if not math.isfinite(value):
        return None, "UNDEFINED_NONFINITE"
    return min(1.0, max(-1.0, value)), "PASS"


def _visual_input_manifest(
    visual_root: Path,
    plan_path: Path,
    results_path: Path,
    result_manifest_path: Path,
    plan_raw_sha: str,
    results_raw_sha: str,
    result_manifest_raw_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest_path = visual_root / "visual_motion_manifest.json"
    audit_path = visual_root / "audit.json"
    visual_manifest = _load_json(manifest_path)
    visual_audit = _load_json(audit_path)
    manifest_raw_sha = _sha_file(manifest_path)
    audit_raw_sha = _sha_file(audit_path)
    if visual_manifest.get("schema_version") != "cross-tfg-visual-motion-guard-v1" or visual_manifest.get("manifest_sha256") != _payload_sha(visual_manifest, "manifest_sha256"):
        raise AssociationError("visual-motion manifest schema/fingerprint is invalid")
    if visual_manifest.get("status") not in ("PASS", "INCOMPLETE") or visual_audit.get("status") != visual_manifest.get("status"):
        raise AssociationError("visual-motion manifest/audit status mismatch")
    expected_inputs = {
        "plan_raw_sha256": plan_raw_sha,
        "results_raw_sha256": results_raw_sha,
        "result_manifest_raw_sha256": result_manifest_raw_sha,
    }
    for key, expected in expected_inputs.items():
        if visual_audit.get(key) != expected:
            raise AssociationError(f"visual-motion audit does not bind exact {key}")
    if visual_audit.get("plan_path") != str(plan_path) or visual_audit.get("results_path") != str(results_path):
        raise AssociationError("visual-motion audit input paths differ from the requested inputs")
    files = visual_manifest.get("files")
    if not isinstance(files, list):
        raise AssociationError("visual-motion manifest has no file inventory")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AssociationError("malformed visual-motion inventory item")
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise AssociationError("visual-motion inventory path escapes its output root")
        normalized = rel.as_posix()
        if normalized in seen:
            raise AssociationError(f"duplicate visual-motion inventory path: {normalized}")
        seen.add(normalized)
        path = (visual_root / rel).resolve(strict=True)
        if path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise AssociationError(f"visual-motion output inventory mismatch: {normalized}")
    if "audit.json" not in seen:
        raise AssociationError("visual-motion output inventory omits audit.json")
    return visual_manifest, visual_audit, manifest_raw_sha, audit_raw_sha


def _corr_record(count: int, value: float | None, status: str) -> dict[str, Any]:
    return {"pair_count": int(count), "pearson_r": value, "status": status}


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise AssociationError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    result_manifest_path = Path(args.result_manifest).resolve(strict=True)
    visual_root = Path(args.visual_root).resolve(strict=True)
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    plan = _load_json(plan_path)
    result_manifest = _load_json(result_manifest_path)
    plan_raw_sha, results_raw_sha = _sha_file(plan_path), _sha_file(results_path)
    result_manifest_raw_sha = _sha_file(result_manifest_path)
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise AssociationError("frozen Stage-A plan schema/fingerprint is invalid")
    if result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or result_manifest.get("run_id") != plan.get("run_id") or result_manifest.get("manifest_sha256") != _payload_sha(result_manifest, "manifest_sha256"):
        raise AssociationError("Stage-A result manifest schema/fingerprint is invalid")
    if result_manifest.get("plan_raw_sha256") != plan_raw_sha or result_manifest.get("plan_sha256") != plan.get("plan_sha256") or result_manifest.get("results_jsonl_sha256") != results_raw_sha:
        raise AssociationError("Stage-A result manifest does not bind the requested inputs")
    visual_manifest, visual_audit, visual_manifest_raw_sha, visual_audit_raw_sha = _visual_input_manifest(
        visual_root, plan_path, results_path, result_manifest_path, plan_raw_sha, results_raw_sha, result_manifest_raw_sha,
    )
    planned_rows = plan.get("rows")
    results_rows = _load_jsonl(results_path)
    visual_rows = _load_jsonl(visual_root / "visual_motion_results.jsonl")
    if not isinstance(planned_rows, list) or len(planned_rows) != len(results_rows) or len(planned_rows) != len(visual_rows):
        raise AssociationError("plan, result, and visual row counts differ")
    plan_ids = [row.get("row_id") for row in planned_rows]
    if plan_ids != [row.get("row_id") for row in results_rows] or plan_ids != [row.get("row_id") for row in visual_rows] or len(plan_ids) != len(set(plan_ids)):
        raise AssociationError("plan/result/visual rows do not preserve exact unique plan order")
    methods = plan.get("methods")
    if not isinstance(methods, list):
        raise AssociationError("plan methods are missing")
    method_ids = [str(item.get("method_id")) for item in methods if isinstance(item, dict)]
    if len(method_ids) != len(methods) or not method_ids or len(set(method_ids)) != len(method_ids):
        raise AssociationError("planned methods must have nonempty unique method IDs")
    identity_ids = [str(item["method_id"]) for item in methods if item.get("kind") == "identity"]
    if len(identity_ids) != 1:
        raise AssociationError("plan must have exactly one identity method")
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise AssociationError("plan model registry is missing")
    model_by_id = {str(item.get("model_id")): item for item in models if isinstance(item, dict) and isinstance(item.get("model_id"), str)}
    if len(model_by_id) != len(models):
        raise AssociationError("plan model IDs must be unique")
    syncnet = plan.get("protocol", {}).get("syncnet", {})
    lag_offsets = syncnet.get("lag_offsets_frames")
    if lag_offsets != list(LAG_OFFSETS):
        raise AssociationError("fixed speech-motion sensitivity offsets must match the plan’s −15…+15 axis")
    audio_frame_samples = Fraction(SAMPLE_RATE_HZ * FRAME_RATE_DEN, FRAME_RATE_NUM)
    if audio_frame_samples.denominator != 1:
        raise AssociationError("the frozen 16 kHz audio / 25 fps clock does not yield integer samples per frame")
    samples_per_frame = int(audio_frame_samples)
    if any((int(model.get("fps_num", 0)), int(model.get("fps_den", 0))) != (FRAME_RATE_NUM, FRAME_RATE_DEN) for model in models):
        raise AssociationError("all registered models must use the frozen 25/1 visual/audio frame clock")

    _visual_input_manifest_rows = visual_manifest.get("files", [])
    visual_results_path = visual_root / "visual_motion_results.jsonl"
    visual_results_sha = _sha_file(visual_results_path)
    listed_visual_results = next((item for item in _visual_input_manifest_rows if item.get("path") == "visual_motion_results.jsonl"), None)
    if not isinstance(listed_visual_results, dict) or listed_visual_results.get("sha256") != visual_results_sha:
        raise AssociationError("visual results JSONL is not bound by its output manifest")

    audio_cache: dict[str, dict[str, Any]] = {}
    visual_by_id: dict[str, dict[str, Any]] = {}
    counts = Counter()
    for planned, result, visual in zip(planned_rows, results_rows, visual_rows, strict=True):
        row_id = str(planned["row_id"])
        for field in IDENTITY_FIELDS:
            if result.get(field) != planned.get(field) or visual.get(field) != planned.get(field):
                raise AssociationError(f"row identity mismatch {row_id}.{field}")
        if result.get("score_audio_sha256") != planned.get("natural_pcm", {}).get("pcm_sha256"):
            raise AssociationError(f"SyncNet score audio is not the planned natural PCM for {row_id}")
        if result.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise AssociationError(f"invalid producer status for {row_id}")
        if result.get("status") != "PASS" or visual.get("status") != "PASS":
            state = "SOURCE_OR_VISUAL_ROW_NOT_PASS"
            visual_by_id[row_id] = {"planned": planned, "result": result, "visual": visual, "status": state}
            counts[state] += 1
            continue
        expected_model = model_by_id.get(str(planned.get("model_id")))
        if expected_model is None or result.get("score", {}).get("score_model_sha256") != expected_model.get("score_model_sha256"):
            raise AssociationError(f"score model differs from frozen model registry for {row_id}")
        if result.get("natural_pcm_sha256") != planned["natural_pcm"].get("pcm_sha256"):
            raise AssociationError(f"result natural PCM identity mismatch for {row_id}")
        if visual.get("source_face_sha256") != planned.get("source_face", {}).get("file_sha256"):
            raise AssociationError(f"visual row does not bind planned face for {row_id}")
        render = result.get("render")
        if not isinstance(render, dict) or visual.get("video", {}).get("file_sha256") != render.get("silent_video_sha256"):
            raise AssociationError(f"visual row does not bind the exact render video for {row_id}")
        clock = visual.get("render_clock")
        if not isinstance(clock, dict) or (clock.get("fps_num"), clock.get("fps_den")) != (FRAME_RATE_NUM, FRAME_RATE_DEN):
            raise AssociationError(f"visual row lacks the exact frozen 25 fps clock for {row_id}")
        arrays = _load_visual_arrays(visual_root, visual, row_id)
        nframes = len(arrays["frame_index_int32"])
        if nframes != clock.get("frame_count") or not np.array_equal(arrays["frame_index_int32"], np.arange(nframes, dtype=np.int32)):
            raise AssociationError(f"visual trajectory frame indices/count mismatch for {row_id}")
        nums, dens = arrays["timestamp_num_int64"], arrays["timestamp_den_int64"]
        if nums.shape != (nframes,) or dens.shape != (nframes,) or np.any(dens <= 0):
            raise AssociationError(f"visual trajectory rational clock arrays are malformed for {row_id}")
        for frame_index, (num, den) in enumerate(zip(nums.tolist(), dens.tolist(), strict=True)):
            if Fraction(int(num), int(den)) != Fraction(frame_index, FRAME_RATE_NUM):
                raise AssociationError(f"visual trajectory timestamp disagrees with source clock for {row_id}[{frame_index}]")
        valid = arrays["tracking_valid_bool"].astype(bool)
        aperture = arrays["mouth_aperture_eye_normalized_float64"]
        velocity = arrays["aperture_velocity_eye_normalized_per_s_float64"]
        velocity_valid = arrays["aperture_velocity_valid_bool"].astype(bool)
        if valid.shape != (nframes,) or aperture.shape != (nframes,) or velocity.shape != (max(0, nframes - 1),) or velocity_valid.shape != velocity.shape:
            raise AssociationError(f"visual tracking/trajectory array dimensions differ for {row_id}")
        if not np.isfinite(aperture).all() or not np.isfinite(velocity).all():
            raise AssociationError(f"visual trajectory has non-finite values for {row_id}")

        natural_ref = planned.get("natural_pcm")
        if not isinstance(natural_ref, dict):
            raise AssociationError(f"planned natural PCM ref is missing for {row_id}")
        audio_path = _resolve(artifact_root, natural_ref.get("path"), f"{row_id}.natural_pcm.path")
        cache_key = str(audio_path)
        audio_entry = audio_cache.get(cache_key)
        if audio_entry is None:
            if _sha_file(audio_path) != natural_ref.get("file_sha256"):
                raise AssociationError(f"natural PCM file SHA mismatch for {row_id}")
            samples, pcm_sha, sample_count = _read_f32_wav(audio_path)
            if pcm_sha != natural_ref.get("pcm_sha256") or sample_count != natural_ref.get("sample_count") or sample_count != result.get("score_audio_sample_count", sample_count):
                raise AssociationError(f"natural PCM bytes/count mismatch for {row_id}")
            complete_audio_frames = sample_count // samples_per_frame
            rms = np.empty(complete_audio_frames, dtype=np.float64)
            for frame_index in range(complete_audio_frames):
                start = frame_index * samples_per_frame
                frame = samples[start:start + samples_per_frame].astype(np.float64, copy=False)
                rms[frame_index] = math.sqrt(float(np.mean(frame * frame, dtype=np.float64)))
            log_rms = np.log(np.maximum(rms, 1e-8))
            audio_entry = {"pcm_sha256": pcm_sha, "sample_count": sample_count, "complete_audio_frames": complete_audio_frames, "log_rms": log_rms}
            audio_cache[cache_key] = audio_entry
        elif audio_entry["pcm_sha256"] != natural_ref.get("pcm_sha256"):
            raise AssociationError(f"paired rows disagree on natural PCM bytes for {row_id}")
        visual_by_id[row_id] = {
            "planned": planned, "result": result, "visual": visual, "status": "PASS",
            "arrays": arrays, "valid": valid, "aperture": aperture,
            "velocity": velocity, "velocity_valid": velocity_valid,
            "audio": audio_entry,
        }
        counts["PASS"] += 1

    groups: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for planned in planned_rows:
        group = tuple(planned.get(field) for field in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))
        method_id = str(planned["method_id"])
        if method_id in groups[group]:
            raise AssociationError(f"duplicate method within paired context {group}: {method_id}")
        groups[group][method_id] = str(planned["row_id"])

    diagnostics: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    incomplete_groups = 0
    for group, row_ids_by_method in groups.items():
        if set(row_ids_by_method) != set(method_ids) or any(visual_by_id[row_id]["status"] != "PASS" for row_id in row_ids_by_method.values()):
            incomplete_groups += 1
            reason = "INCOMPLETE_PLANNED_METHOD_GROUP" if set(row_ids_by_method) != set(method_ids) else "INCOMPLETE_RENDER_OR_VISUAL_TRACKING"
            diagnostics.append({"context": list(group), "status": reason, "planned_methods": method_ids, "present_methods": sorted(row_ids_by_method)})
            for method_id, row_id in row_ids_by_method.items():
                diagnostics.append({"context": list(group), "row_id": row_id, "method_id": method_id, "status": reason, "zero_lag": None, "lag_curve": None})
            continue
        states = {method: visual_by_id[row_id] for method, row_id in row_ids_by_method.items()}
        first_state = states[method_ids[0]]
        audio = first_state["audio"]
        if any(state["audio"]["pcm_sha256"] != audio["pcm_sha256"] for state in states.values()):
            raise AssociationError(f"paired methods do not share original natural PCM for {group}")
        minimum_frames = min(len(state["valid"]) for state in states.values())
        common_frame_mask = np.ones(minimum_frames, dtype=bool)
        for state in states.values():
            common_frame_mask &= state["valid"][:minimum_frames]
        common_frames = np.flatnonzero(common_frame_mask)
        row_records: dict[str, dict[str, Any]] = {}
        for method_id in method_ids:
            state = states[method_id]
            lag_records: list[dict[str, Any]] = []
            aperture_curve: list[float | None] = []
            velocity_curve: list[float | None] = []
            aperture_counts: list[int] = []
            velocity_counts: list[int] = []
            for lag in LAG_OFFSETS:
                aperture_frames = np.asarray([frame for frame in common_frames if 0 <= frame + lag < audio["complete_audio_frames"]], dtype=np.int64)
                aperture_values = state["aperture"][aperture_frames]
                audio_values = audio["log_rms"][aperture_frames + lag]
                aperture_r, aperture_status = _pearson(aperture_values, audio_values)
                velocity_frames = np.asarray([
                    frame for frame in common_frames
                    if frame + 1 < minimum_frames
                    and state["velocity_valid"][frame]
                    and all(other["velocity_valid"][frame] for other in states.values())
                    and 0 <= frame + lag
                    and frame + lag + 1 < audio["complete_audio_frames"]
                ], dtype=np.int64)
                velocity_values = state["velocity"][velocity_frames]
                audio_onsets = audio["log_rms"][velocity_frames + lag + 1] - audio["log_rms"][velocity_frames + lag]
                velocity_r, velocity_status = _pearson(velocity_values, audio_onsets)
                lag_records.append({
                    "audio_frame_equals_visual_frame_plus_lag": int(lag),
                    "mouth_aperture_vs_natural_log_rms": _corr_record(len(aperture_frames), aperture_r, aperture_status),
                    "mouth_aperture_velocity_vs_natural_log_rms_change": _corr_record(len(velocity_frames), velocity_r, velocity_status),
                })
                aperture_curve.append(aperture_r)
                velocity_curve.append(velocity_r)
                aperture_counts.append(len(aperture_frames))
                velocity_counts.append(len(velocity_frames))
            zero_index = LAG_OFFSETS.index(0)
            record = {
                "context": list(group), "row_id": row_ids_by_method[method_id], "method_id": method_id,
                "identity_method_id": identity_ids[0], "status": "PASS",
                "natural_pcm_sha256": audio["pcm_sha256"], "audio_sample_count": audio["sample_count"],
                "complete_25fps_audio_frames": audio["complete_audio_frames"],
                "cross_method_common_valid_visual_frames": int(len(common_frames)),
                "minimum_pairs_for_correlation": MINIMUM_PAIRS,
                "declared_zero_lag": lag_records[zero_index],
                "lag_offsets_frames": list(LAG_OFFSETS), "lag_curve": lag_records,
                "aperture_pair_counts_by_lag": aperture_counts,
                "velocity_pair_counts_by_lag": velocity_counts,
                "interpretation": "descriptive association only; zero lag is the declared value and the fixed lag curve is sensitivity, with no retiming, lag selection, or claim that larger motion is better",
            }
            diagnostics.append(record)
            row_records[method_id] = record
        identity_record = row_records[identity_ids[0]]
        for method_id in method_ids:
            if method_id == identity_ids[0]:
                continue
            record = row_records[method_id]
            left_aperture = identity_record["declared_zero_lag"]["mouth_aperture_vs_natural_log_rms"]["pearson_r"]
            right_aperture = record["declared_zero_lag"]["mouth_aperture_vs_natural_log_rms"]["pearson_r"]
            left_velocity = identity_record["declared_zero_lag"]["mouth_aperture_velocity_vs_natural_log_rms_change"]["pearson_r"]
            right_velocity = record["declared_zero_lag"]["mouth_aperture_velocity_vs_natural_log_rms_change"]["pearson_r"]
            paired_rows.append({
                "context": list(group), "method_id": method_id, "identity_method_id": identity_ids[0],
                "status": "PASS",
                "zero_lag_aperture_association_delta_vs_identity": None if left_aperture is None or right_aperture is None else float(right_aperture - left_aperture),
                "zero_lag_aperture_velocity_onset_association_delta_vs_identity": None if left_velocity is None or right_velocity is None else float(right_velocity - left_velocity),
                "interpretation": "paired per-context descriptive correlation differences; no inferential weighting across speakers is applied here",
            })

    diagnostics_path = out / "speech_motion_association.jsonl"
    paired_path = out / "paired_speech_motion_association.jsonl"
    for path, rows in ((diagnostics_path, diagnostics), (paired_path, paired_rows)):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    output_status = "PASS" if not incomplete_groups and counts["PASS"] == len(planned_rows) else "INCOMPLETE"
    audit = {
        "schema_version": SCHEMA_VERSION, "status": output_status, "run_id": plan.get("run_id"),
        "plan_path": str(plan_path), "plan_raw_sha256": plan_raw_sha, "plan_payload_sha256": plan.get("plan_sha256"),
        "results_path": str(results_path), "results_raw_sha256": results_raw_sha,
        "result_manifest_path": str(result_manifest_path), "result_manifest_raw_sha256": result_manifest_raw_sha,
        "result_manifest_payload_sha256": result_manifest.get("manifest_sha256"),
        "visual_root": str(visual_root), "visual_manifest_raw_sha256": visual_manifest_raw_sha,
        "visual_manifest_payload_sha256": visual_manifest.get("manifest_sha256"), "visual_audit_raw_sha256": visual_audit_raw_sha,
        "planned_rows": len(planned_rows), "accounted_source_visual_pass_rows": counts["PASS"],
        "source_or_visual_not_pass_rows": counts["SOURCE_OR_VISUAL_ROW_NOT_PASS"],
        "paired_context_groups": len(groups), "incomplete_context_groups": incomplete_groups,
        "diagnostic_rows": len(diagnostics), "paired_delta_rows": len(paired_rows),
        "frame_rate": {"numerator": FRAME_RATE_NUM, "denominator": FRAME_RATE_DEN},
        "audio_energy": "natural PCM framewise 40 ms log-RMS; full frame only; no padding",
        "lag_definition": "audio frame equals visual source-clock frame plus lag; zero-lag association declared, −15…+15 curve sensitivity only",
        "minimum_pairs": MINIMUM_PAIRS,
        "effects_selected_from_association": False,
    }
    audit_path = out / "audit.json"
    audit_path.write_bytes(_canonical_json(audit) + b"\n")
    files = []
    for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "audit.json"):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    files.append({"path": "audit.json", "bytes": audit_path.stat().st_size, "sha256": _sha_file(audit_path)})
    output_manifest = {"schema_version": SCHEMA_VERSION, "status": output_status, "run_id": plan.get("run_id"), "files": files}
    output_manifest["manifest_sha256"] = _payload_sha(output_manifest, "manifest_sha256")
    (out / "manifest.json").write_bytes(_canonical_json(output_manifest) + b"\n")
    return 0 if output_status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        out = Path(args.out).resolve()
        if out.exists():
            error = {"schema_version": SCHEMA_VERSION, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "argv": sys.argv}
            (out / "audit.json").write_bytes(_canonical_json(error) + b"\n")
        print(f"speech_motion_association_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
