#!/usr/bin/env python3
"""Reconstruct the frozen 54-endpoint visual family from 800 fresh tracks.

This CPU consumer must run only after the root has sealed availability and
issued a geometry-read release bound to the exact track/availability inputs.
It reads no SyncNet scores, model effects, or outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DESIGN_SCHEMA_SHA256 = "0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8"
DESIGN_PIN_CORRECTION_SHA256 = "cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6"
FPS_HZ = 25.0
LATENCY_S = -0.085
SEED = 20260930
DRAWS = 20_000
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
METRICS = ("mean_aperture", "mean_absolute_aperture_velocity_per_s", "phone_class_separation")
CONTRASTS = (
    "mean_q10_minus_q00", "trajectory_q01_minus_q00", "joint_q11_minus_q00",
    "grid_cycle_SELF_GRID_minus_q00", "mean_minus_trajectory_q10_minus_q01",
    "mean_beyond_grid_q10_minus_SELF_GRID", "trajectory_beyond_grid_q01_minus_SELF_GRID",
    "joint_beyond_grid_q11_minus_SELF_GRID", "factorial_interaction_q11_minus_q10_minus_q01_plus_q00",
)
TONE_MARKS = frozenset(("˥", "˦", "˧", "˨", "˩"))
BILABIAL = frozenset(("p", "b", "m"))
OPEN_VOWEL = frozenset(("a", "ɑ", "ɒ", "ɐ", "æ"))
OUTPUT_FILES = (
    "geometry_values.npz", "phone_token_measurements.npz",
    "speaker_endpoints.npz", "bootstrap_family.npz",
)


class GeometryRunError(RuntimeError):
    pass


def canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GeometryRunError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise GeometryRunError(f"json_not_object:{path.name}")
    return value


def verify_embedded(value: dict[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = sha_bytes(canon({key: item for key, item in value.items() if key != field}))
    if expected != actual:
        raise GeometryRunError(f"embedded_fingerprint_mismatch:{field}")
    return actual


def sha_array(value: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(value).tobytes())


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise GeometryRunError("input_path_escapes_root")
    if not candidate.is_file():
        raise GeometryRunError("input_reference_not_regular_file")
    return candidate


def _read_inventory(path: Path) -> dict[str, str]:
    obj = read_json(path)
    files = obj.get("files")
    if not isinstance(files, dict) or not files:
        raise GeometryRunError("file_inventory_missing")
    if any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 for k, v in files.items()):
        raise GeometryRunError("file_inventory_schema_invalid")
    return files


def _safe_inventory_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise GeometryRunError("inventory_path_not_relative")
    return _resolve_inside(root, relative)


def _verify_inventory(root: Path, inventory_path: Path, expected: set[str]) -> dict[str, str]:
    inventory = _read_inventory(inventory_path)
    if set(inventory) != expected:
        missing, extra = sorted(expected - set(inventory)), sorted(set(inventory) - expected)
        raise GeometryRunError(f"inventory_path_set_mismatch:missing={len(missing)}:extra={len(extra)}")
    for relative, expected_sha in inventory.items():
        path = _safe_inventory_path(root, relative)
        if sha_file(path) != expected_sha:
            raise GeometryRunError(f"inventory_hash_mismatch:{relative}")
    return inventory


def _import_core(path: Path, expected_sha: str) -> Any:
    path = path.resolve(strict=True)
    if sha_file(path) != expected_sha:
        raise GeometryRunError("visual_core_source_sha_mismatch")
    spec = importlib.util.spec_from_file_location("world_shape_visual_core_frozen", path)
    if spec is None or spec.loader is None:
        raise GeometryRunError("visual_core_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _phone_class(label: Any, core: Any) -> tuple[str, str]:
    raw = str(label or "").strip().lower()
    tone_stripped = "".join(ch for ch in raw if ch not in TONE_MARKS)
    normalized = core.normalize_phone(tone_stripped)
    if normalized in BILABIAL:
        return normalized, "bilabial"
    if normalized in OPEN_VOWEL:
        return normalized, "open_vowel"
    return normalized, ""


def _match_clock(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.size > 1:
        tolerance = 0.5 * float(np.median(np.diff(reference))) + 1e-12
    else:
        tolerance = 0.5 / FPS_HZ
    output = np.full(reference.shape[0], -1, dtype=np.int32)
    i = j = 0
    while i < len(reference) and j < len(candidate):
        delta = float(candidate[j] - reference[i])
        if abs(delta) <= tolerance:
            output[i] = j
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    return output


def _load_npz_arrays(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise GeometryRunError(f"npz_keyset_mismatch:{path.name}")
            return {key: np.asarray(archive[key]) for key in keys}
    except GeometryRunError:
        raise
    except Exception as exc:
        raise GeometryRunError(f"npz_read_failed:{path.name}:{type(exc).__name__}") from exc


def _context_identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")), str(row.get("direction", ""))


def _track_record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("sample_id", "")), str(row.get("direction", "")), str(row.get("cell", ""))


def _validate_release(
    release_path: Path,
    source_sha: str,
    plan_raw: str,
    plan_payload: str,
    track_raw: str,
    track_payload: str,
    track_seal_raw: str,
    track_inventory_raw: str,
    availability_raw: str,
    availability_payload: str,
    availability_seal_raw: str,
    availability_inventory_raw: str,
) -> tuple[dict[str, Any], str]:
    release = read_json(release_path)
    payload = verify_embedded(release, "manifest_sha256")
    checks = {
        "status": "ROOT_RELEASED",
        "scope": "GEOMETRY_ANALYSIS",
        "geometry_read_authorized": True,
        "geometry_consumer_source_sha256": source_sha,
        "tracking_plan_raw_sha256": plan_raw,
        "tracking_plan_payload_sha256": plan_payload,
        "track_manifest_raw_sha256": track_raw,
        "track_manifest_payload_sha256": track_payload,
        "track_seal_raw_sha256": track_seal_raw,
        "track_inventory_raw_sha256": track_inventory_raw,
        "availability_manifest_raw_sha256": availability_raw,
        "availability_manifest_payload_sha256": availability_payload,
        "availability_seal_raw_sha256": availability_seal_raw,
        "availability_inventory_raw_sha256": availability_inventory_raw,
    }
    for key, expected in checks.items():
        if release.get(key) != expected:
            raise GeometryRunError(f"geometry_release_binding_mismatch:{key}")
    return release, payload


def _load_and_validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve(strict=True)
    track_root = args.track_dir.resolve(strict=True)
    availability_root = args.availability_dir.resolve(strict=True)
    plan_raw = sha_file(plan_path)
    plan = read_json(plan_path)
    plan_payload = verify_embedded(plan, "plan_sha256")
    if plan.get("schema") != "world_shape_visual_tracking_plan_v1" or len(plan.get("context_rows", [])) != 160:
        raise GeometryRunError("tracking_plan_schema_or_context_count_mismatch")

    sources = {
        "tracking_extractor_source": args.extractor_source,
        "visual_design": args.design,
        "evaluation_design": args.evaluation_design,
        "execution_pin_correction": args.pin_correction,
        "feature_helper": args.video_features,
        "landmarker_asset": args.landmarker,
        "visual_core_source": args.visual_core,
        "tone_adapter_source": args.tone_adapter,
        "visual_calibration": args.calibration,
        "visual_calibration_freeze": args.calibration_freeze,
    }
    source_hashes = {}
    for role, path in sources.items():
        resolved = Path(path).resolve(strict=True)
        digest = sha_file(resolved)
        expected = plan.get("input_seals", {}).get(role, {}).get("raw_sha256")
        if expected != digest:
            raise GeometryRunError(f"tracking_plan_source_binding_mismatch:{role}")
        source_hashes[role] = {"path": str(resolved), "raw_sha256": digest}
    if sha_file(args.design) != DESIGN_SCHEMA_SHA256 or sha_file(args.pin_correction) != DESIGN_PIN_CORRECTION_SHA256:
        raise GeometryRunError("formal_visual_design_or_correction_pin_mismatch")

    expected_contexts = [
        (_context_identity(row), row) for row in sorted(plan["context_rows"], key=lambda x: int(x["context_order"]))
    ]
    if [int(row["context_order"]) for _, row in expected_contexts] != list(range(160)):
        raise GeometryRunError("tracking_plan_context_order_not_0_to_159")
    if len({identity for identity, _ in expected_contexts}) != 160:
        raise GeometryRunError("tracking_plan_context_identity_duplicate")
    if len({str(row["speaker_id"]) for _, row in expected_contexts}) != 40:
        raise GeometryRunError("tracking_plan_speaker_roster_not_40")

    track_manifest_path = track_root / "visual_track_manifest.json"
    track_seal_path = track_root / "visual_track_seal.json"
    track_inventory_path = track_root / "all_files_sha256.json"
    track_manifest_raw = sha_file(track_manifest_path)
    track_seal_raw = sha_file(track_seal_path)
    track_inventory_raw = sha_file(track_inventory_path)
    track_manifest = read_json(track_manifest_path)
    track_payload = verify_embedded(track_manifest, "manifest_sha256")
    track_seal = read_json(track_seal_path)
    if (track_manifest.get("schema") != "world_shape_visual_track_manifest_v1"
            or track_manifest.get("status") != "COMPLETE"
            or track_manifest.get("completed_videos") != 800
            or track_manifest.get("failed_videos") != 0
            or len(track_manifest.get("records", [])) != 800
            or track_manifest.get("effects_read") is not False
            or track_manifest.get("model_forward_performed") is not False
            or track_seal.get("status") != "SEALED"
            or track_seal.get("manifest_sha256") != track_manifest_raw
            or track_seal.get("manifest_payload_sha256") != track_payload
            or track_seal.get("all_files_sha256") != track_inventory_raw):
        raise GeometryRunError("track_manifest_or_seal_not_complete_and_bound")
    if track_manifest.get("tracking_plan_raw_sha256") != plan_raw or track_manifest.get("tracking_plan_sha256") != plan_payload:
        raise GeometryRunError("track_manifest_plan_binding_mismatch")

    track_by_key = {}
    for record in track_manifest["records"]:
        key = _track_record_key(record)
        if key in track_by_key or record.get("status") != "complete":
            raise GeometryRunError("track_record_duplicate_or_incomplete")
        track_by_key[key] = record
    if len(track_by_key) != 800:
        raise GeometryRunError("track_record_unique_count_mismatch")
    expected_track_keys = set()
    for identity, row in expected_contexts:
        sample, direction = identity
        for cell in CELLS:
            expected_track_keys.add((sample, direction, cell))
    if set(track_by_key) != expected_track_keys:
        raise GeometryRunError("track_record_identity_set_mismatch")

    track_expected_inventory = {"visual_track_protocol.json", "progress.jsonl", "visual_track_manifest.json"}
    for identity, row in expected_contexts:
        sample, direction = identity
        for cell in CELLS:
            record = track_by_key[(sample, direction, cell)]
            cell_ref = row["cells"][cell]
            checks = {
                "speaker_id": str(row["speaker_id"]),
                "split": str(row["split"]),
                "context_order": int(row["context_order"]),
                "video_relative_path": str(cell_ref["video_relative_path"]),
                "video_sha256": str(cell_ref["video_sha256"]),
                "official_frame_count": int(cell_ref["official_frame_count"]),
                "recipient_grid_sha256": str(row["recipient_grid_sha256"]),
            }
            if any(record.get(name) != expected for name, expected in checks.items()):
                raise GeometryRunError(f"track_record_plan_identity_mismatch:{sample}:{direction}:{cell}")
            if (int(record.get("video_bytes", -1)) != int(cell_ref["video_bytes"])
                    or not math.isclose(float(record.get("fps_hz", 0.0)), FPS_HZ, rel_tol=0, abs_tol=1e-6)):
                raise GeometryRunError(f"track_record_clock_or_size_mismatch:{sample}:{direction}:{cell}")
            track_expected_inventory.add(str(record["feature_relative_path"]))
            track_expected_inventory.add(str(record["availability_relative_path"]))
    track_inventory = _verify_inventory(track_root, track_inventory_path, track_expected_inventory)
    if (track_inventory.get("visual_track_protocol.json") != track_manifest.get("tracking_protocol_sha256")
            or track_inventory.get("progress.jsonl") != track_manifest.get("progress_sha256")
            or track_seal.get("files") != len(track_inventory)):
        raise GeometryRunError("track_protocol_progress_or_inventory_count_binding_mismatch")
    for record in track_manifest["records"]:
        if track_inventory.get(record["feature_relative_path"]) != record["feature_sha256"]:
            raise GeometryRunError("track_feature_inventory_manifest_binding_mismatch")
        if track_inventory.get(record["availability_relative_path"]) != record["availability_sha256"]:
            raise GeometryRunError("track_availability_inventory_manifest_binding_mismatch")

    availability_manifest_path = availability_root / "availability_freeze.json"
    availability_seal_path = availability_root / "availability_seal.json"
    availability_inventory_path = availability_root / "all_files_sha256.json"
    availability_raw = sha_file(availability_manifest_path)
    availability_payload = verify_embedded(read_json(availability_manifest_path), "manifest_sha256")
    availability_manifest = read_json(availability_manifest_path)
    availability_seal_raw = sha_file(availability_seal_path)
    availability_inventory_raw = sha_file(availability_inventory_path)
    availability_seal = read_json(availability_seal_path)
    if (availability_manifest.get("schema") != "world_shape_visual_availability_freeze_v1"
            or availability_manifest.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or availability_manifest.get("tracking_plan_raw_sha256") != plan_raw
            or availability_manifest.get("tracking_plan_payload_sha256") != plan_payload
            or availability_manifest.get("track_manifest_raw_sha256") != track_manifest_raw
            or availability_manifest.get("track_manifest_payload_sha256") != track_payload
            or availability_manifest.get("track_seal_raw_sha256") != track_seal_raw
            or availability_manifest.get("track_files_inventory_raw_sha256") != track_inventory_raw
            or availability_manifest.get("visual_design_sha256") != DESIGN_SCHEMA_SHA256
            or availability_manifest.get("coverage", {}).get("geometry_values_read") is not False
            or availability_manifest.get("coverage", {}).get("evaluation_effects_read") is not False
            or availability_seal.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or availability_seal.get("manifest_raw_sha256") != availability_raw
            or availability_seal.get("manifest_payload_sha256") != availability_payload
            or availability_seal.get("all_files_sha256") != availability_inventory_raw
            or availability_seal.get("geometry_values_read") is not False):
        raise GeometryRunError("availability_freeze_or_seal_binding_mismatch")
    if len(availability_manifest.get("context_rows", [])) != 160 or len(availability_manifest.get("speaker_rows", [])) != 80:
        raise GeometryRunError("availability_freeze_row_count_mismatch")
    availability_expected_inventory = {"availability_freeze.json"}
    for row in availability_manifest["context_rows"]:
        availability_expected_inventory.add(str(row["availability_context_path"]))
    availability_inventory = _verify_inventory(availability_root, availability_inventory_path, availability_expected_inventory)
    if availability_seal.get("context_files") != 160 or len(availability_inventory) != 161:
        raise GeometryRunError("availability_inventory_count_binding_mismatch")
    availability_by_key = {}
    for row in availability_manifest["context_rows"]:
        key = _context_identity(row)
        if key in availability_by_key:
            raise GeometryRunError("availability_context_duplicate")
        availability_by_key[key] = row
        if availability_inventory.get(row["availability_context_path"]) != row["availability_context_sha256"]:
            raise GeometryRunError("availability_context_inventory_binding_mismatch")
        plan_row = next((candidate for _, candidate in expected_contexts if _context_identity(candidate) == key), None)
        if plan_row is None or row.get("recipient_grid_sha256") != plan_row.get("recipient_grid_sha256"):
            raise GeometryRunError("availability_recipient_grid_identity_mismatch")
        for cell in CELLS:
            source_record = track_by_key[(key[0], key[1], cell)]
            frozen_ref = row.get("track_refs", {}).get(cell, {})
            if (frozen_ref.get("feature_relative_path") != source_record.get("feature_relative_path")
                    or frozen_ref.get("feature_sha256") != source_record.get("feature_sha256")
                    or frozen_ref.get("availability_relative_path") != source_record.get("availability_relative_path")
                    or frozen_ref.get("availability_sha256") != source_record.get("availability_sha256")
                    or frozen_ref.get("array_inventory") != source_record.get("array_inventory")):
                raise GeometryRunError("availability_track_ref_differs_from_track_manifest")
    if set(availability_by_key) != {_context_identity(row) for _, row in expected_contexts}:
        raise GeometryRunError("availability_context_identity_set_mismatch")

    source_sha = sha_file(Path(__file__).resolve(strict=True))
    release, release_payload = _validate_release(
        args.geometry_release.resolve(strict=True), source_sha, plan_raw, plan_payload,
        track_manifest_raw, track_payload, track_seal_raw, track_inventory_raw,
        availability_raw, availability_payload, availability_seal_raw, availability_inventory_raw,
    )
    return {
        "plan_path": plan_path, "plan": plan, "plan_raw": plan_raw, "plan_payload": plan_payload,
        "source_hashes": source_hashes,
        "track_root": track_root, "track_manifest": track_manifest, "track_manifest_raw": track_manifest_raw,
        "track_payload": track_payload, "track_seal_raw": track_seal_raw, "track_inventory_raw": track_inventory_raw,
        "track_inventory": track_inventory, "track_by_key": track_by_key,
        "availability_root": availability_root, "availability_manifest": availability_manifest,
        "availability_raw": availability_raw, "availability_payload": availability_payload,
        "availability_seal_raw": availability_seal_raw, "availability_inventory_raw": availability_inventory_raw,
        "availability_inventory": availability_inventory, "availability_by_key": availability_by_key,
        "geometry_release_raw": sha_file(args.geometry_release), "geometry_release_payload": release_payload,
        "geometry_release": release,
    }


def _array_inventory_check(record: dict[str, Any], name: str, array: np.ndarray, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    if array.dtype != dtype or array.shape != shape:
        raise GeometryRunError(f"feature_array_shape_dtype_or_finite_mismatch:{name}")
    if dtype.kind == "f" and not np.isfinite(array).all():
        raise GeometryRunError(f"feature_array_shape_dtype_or_finite_mismatch:{name}")
    entry = record.get("array_inventory", {}).get(name)
    if not isinstance(entry, dict):
        raise GeometryRunError(f"feature_array_inventory_missing:{name}")
    if entry.get("shape") != list(array.shape) or entry.get("dtype") != str(array.dtype) or entry.get("sha256") != sha_array(array):
        raise GeometryRunError(f"feature_array_hash_or_metadata_mismatch:{name}")


def _load_geometry_feature(path: Path, record: dict[str, Any], identity: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sha_file(path) != record["feature_sha256"]:
        raise GeometryRunError("feature_file_sha_mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            needed = {"timestamps_s", "valid", "mouth_open", "metadata_json"}
            expected_keys = {"landmarks", "canonical_mouth", "valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area", "timestamps_s", "metadata_json"}
            if set(archive.files) != expected_keys or not needed.issubset(archive.files):
                raise GeometryRunError("feature_npz_missing_geometry_arrays")
            times = np.asarray(archive["timestamps_s"])
            valid = np.asarray(archive["valid"])
            aperture = np.asarray(archive["mouth_open"])
            metadata = json.loads(str(archive["metadata_json"].item()))
    except GeometryRunError:
        raise
    except Exception as exc:
        raise GeometryRunError(f"feature_npz_read_failed:{type(exc).__name__}") from exc
    n = int(record["official_frame_count"])
    _array_inventory_check(record, "timestamps_s", times, np.dtype(np.float64), (n,))
    _array_inventory_check(record, "valid", valid, np.dtype(np.bool_), (n,))
    _array_inventory_check(record, "mouth_open", aperture, np.dtype(np.float64), (n,))
    if record.get("timestamp_sha256") != sha_array(times):
        raise GeometryRunError("track_record_timestamp_hash_mismatch")
    if not isinstance(metadata, dict):
        raise GeometryRunError("feature_metadata_not_object")
    for key, value in identity.items():
        if key.startswith("_"):
            continue
        if metadata.get(key) != value:
            raise GeometryRunError(f"feature_metadata_identity_mismatch:{key}")
    expected_metadata = {
        "video_sha256": record.get("video_sha256"),
        "video_relative_path": record.get("video_relative_path"),
        "recipient_grid_sha256": record.get("recipient_grid_sha256"),
        "landmarker_sha256": identity.get("_landmarker_sha256"),
        "video_features_sha256": identity.get("_video_features_sha256"),
        "visual_core_sha256": identity.get("_visual_core_sha256"),
        "tone_adapter_sha256": identity.get("_tone_adapter_sha256"),
        "full_recipient_clock": True,
        "fps_hz_required": FPS_HZ,
        "decoded_frame_count": n,
        "declared_frame_count": n,
    }
    if any(metadata.get(key) != expected for key, expected in expected_metadata.items()):
        raise GeometryRunError("feature_metadata_video_or_landmarker_binding_missing")
    if not math.isclose(float(metadata.get("fps", 0.0)), FPS_HZ, rel_tol=0, abs_tol=1e-6):
        raise GeometryRunError("feature_metadata_fps_mismatch")
    if (not np.isfinite(times).all() or not np.isfinite(aperture).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))
            or np.any(aperture[~valid] != 0.0)):
        raise GeometryRunError("feature_clock_or_geometry_invalid")
    return times, valid, aperture


def _load_and_verify_availability(path: Path, record: dict[str, Any], q00_sha: str | None) -> dict[str, np.ndarray]:
    if sha_file(path) != record["availability_sha256"]:
        raise GeometryRunError("track_availability_file_sha_mismatch")
    arrays = _load_npz_arrays(path, {"timestamps_s", "valid"})
    times, valid = arrays["timestamps_s"], arrays["valid"]
    n = int(record["official_frame_count"])
    if times.dtype != np.float64 or times.shape != (n,) or valid.dtype != np.bool_ or valid.shape != (n,):
        raise GeometryRunError("raw_availability_clock_shape_dtype_mismatch")
    if q00_sha is not None and sha_array(times) != q00_sha and record.get("cell") == "q00":
        raise GeometryRunError("q00_timestamp_binding_to_track_mismatch")
    return arrays


def _validate_token_freeze(plan_row: dict[str, Any], availability_row: dict[str, Any], common: np.ndarray,
                           q00_times: np.ndarray, core: Any) -> list[dict[str, Any]]:
    phones = plan_row.get("recipient_phones")
    if phones != availability_row.get("recipient_phones"):
        raise GeometryRunError("availability_phone_source_differs_from_plan")
    frame_minimum = int(availability_row.get("minimum_common_frames_per_token", 0))
    if frame_minimum < 1:
        raise GeometryRunError("availability_token_frame_minimum_invalid")
    frozen = availability_row.get("tokens")
    if not isinstance(frozen, list):
        raise GeometryRunError("availability_token_records_missing")
    expected = []
    for phone in phones:
        normalized, cls = _phone_class(phone.get("label"), core)
        if not cls:
            continue
        start = float(phone["start_s"]) + LATENCY_S
        end = float(phone["end_s"]) + LATENCY_S
        token_mask = common & (q00_times >= start) & (q00_times < end)
        indices = np.flatnonzero(token_mask).astype(np.int32)
        expected.append({
            "phone_index": int(phone["phone_index"]), "raw_label": str(phone["label"]),
            "normalized_label": normalized, "visual_class": cls,
            "start_s": phone["start_s"], "end_s": phone["end_s"],
            "shifted_start_s": start, "shifted_end_s": end,
            "common_frames": int(len(indices)), "supported": bool(len(indices) >= frame_minimum),
            "q00_indices": indices.tolist(),
        })
    if len(frozen) != len(expected):
        raise GeometryRunError("availability_supported_phone_token_count_mismatch")
    for actual, wanted in zip(frozen, expected, strict=True):
        if actual != wanted:
            raise GeometryRunError("availability_phone_token_support_reconstruction_mismatch")
    counts = {name: sum(item["supported"] and item["visual_class"] == name for item in expected) for name in ("bilabial", "open_vowel")}
    expected_counts = {name: sum(item["visual_class"] == name for item in expected) for name in ("bilabial", "open_vowel")}
    if availability_row.get("supported_token_counts") != counts or availability_row.get("expected_token_counts") != expected_counts:
        raise GeometryRunError("availability_phone_support_count_reconstruction_mismatch")
    minimums = availability_row.get("minimum_token_counts")
    if not isinstance(minimums, dict) or set(minimums) != {"bilabial", "open_vowel"}:
        raise GeometryRunError("availability_phone_minimum_schema_invalid")
    plan_qualified = bool(plan_row.get("original_mfa_pair_qualified"))
    if availability_row.get("original_mfa_pair_qualified") is not plan_qualified:
        raise GeometryRunError("availability_mfa_eligibility_differs_from_plan")
    expected_eligible = plan_qualified and all(counts[k] >= int(minimums[k]) for k in minimums)
    if availability_row.get("availability_endpoint_eligible") is not expected_eligible:
        raise GeometryRunError("availability_phone_eligibility_reconstruction_mismatch")
    return expected


def _read_context(
    ctx_index: int,
    plan_row: dict[str, Any],
    availability_row: dict[str, Any],
    binding: dict[str, Any],
    core: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    sample, direction = _context_identity(plan_row)
    av_path = _resolve_inside(binding["availability_root"], availability_row["availability_context_path"])
    if sha_file(av_path) != availability_row["availability_context_sha256"]:
        raise GeometryRunError("availability_context_file_sha_mismatch")
    availability = _load_npz_arrays(av_path, {
        "q00_timestamps_s", "common_valid", "common_adjacent",
        *(f"index_map_{cell}" for cell in CELLS),
        *(f"valid_on_q00_{cell}" for cell in CELLS),
    })
    q00 = availability["q00_timestamps_s"]
    common = availability["common_valid"]
    adjacent = availability["common_adjacent"]
    if q00.dtype != np.float64 or common.dtype != np.bool_ or adjacent.dtype != np.bool_:
        raise GeometryRunError("availability_context_clock_or_mask_dtype_mismatch")
    n = len(q00)
    if common.shape != (n,) or adjacent.shape != (max(0, n - 1),):
        raise GeometryRunError("availability_context_clock_or_mask_shape_mismatch")
    if availability_row.get("q00_frame_count") != n or availability_row.get("q00_clock_sha256") != sha_array(q00):
        raise GeometryRunError("availability_q00_clock_binding_mismatch")
    if availability_row.get("common_valid_sha256") != sha_array(common) or availability_row.get("common_adjacent_sha256") != sha_array(adjacent):
        raise GeometryRunError("availability_common_mask_hash_mismatch")
    if not np.array_equal(adjacent, common[:-1] & common[1:]):
        raise GeometryRunError("availability_adjacent_support_not_reconstructed")
    if np.any(~np.isfinite(q00)) or (n > 1 and not np.allclose(np.diff(q00), 1.0 / FPS_HZ, rtol=0, atol=1e-9)):
        raise GeometryRunError("q00_reference_clock_not_full_fps25")
    phone_tokens = _validate_token_freeze(plan_row, availability_row, common, q00, core)
    expected_context_identity = {
        "sample_id": sample, "speaker_id": str(plan_row["speaker_id"]),
        "direction": direction, "cell": None, "context_order": int(plan_row["context_order"]),
        "split": str(plan_row["split"]),
    }
    aperture_by_cell: dict[str, np.ndarray] = {}
    maps: dict[str, np.ndarray] = {}
    valid_on_clock: dict[str, np.ndarray] = {}
    feature_records: dict[str, dict[str, Any]] = {}
    q00_clock_hash = None
    for cell in CELLS:
        record = binding["track_by_key"][(sample, direction, cell)]
        feature_record = record
        video_ref = plan_row["cells"][cell]
        identity = {**expected_context_identity, "cell": cell, "ordered_eval_index": int(plan_row["ordered_eval_index"])}
        if record.get("recipient_grid_sha256") != plan_row.get("recipient_grid_sha256") or record.get("video_sha256") != video_ref.get("video_sha256"):
            raise GeometryRunError("feature_context_grid_or_video_identity_mismatch")
        feature_path = _resolve_inside(binding["track_root"], str(record["feature_relative_path"]))
        times, valid, aperture = _load_geometry_feature(feature_path, feature_record, identity)
        raw_av_path = _resolve_inside(binding["track_root"], str(record["availability_relative_path"]))
        raw_av = _load_and_verify_availability(raw_av_path, record, q00_clock_hash or "")
        if not np.array_equal(raw_av["timestamps_s"], times) or not np.array_equal(raw_av["valid"], valid):
            raise GeometryRunError("feature_and_availability_raw_arrays_differ")
        index_map = availability[f"index_map_{cell}"]
        valid_saved = availability[f"valid_on_q00_{cell}"]
        if index_map.dtype != np.int32 or index_map.shape != (n,) or valid_saved.dtype != np.bool_ or valid_saved.shape != (n,):
            raise GeometryRunError("availability_cell_map_shape_dtype_mismatch")
        map_expected = _match_clock(q00, times)
        if not np.array_equal(index_map, map_expected):
            raise GeometryRunError("availability_cell_clock_map_reconstruction_mismatch")
        if cell == "q00" and not np.array_equal(index_map, np.arange(n, dtype=np.int32)):
            raise GeometryRunError("q00_reference_map_not_identity")
        found = index_map >= 0
        if np.any(index_map[found] >= len(times)) or np.any(index_map[found][1:] <= index_map[found][:-1]):
            raise GeometryRunError("availability_cell_map_not_strictly_monotone")
        valid_expected = np.zeros(n, dtype=np.bool_)
        valid_expected[found] = valid[index_map[found]]
        if not np.array_equal(valid_saved, valid_expected):
            raise GeometryRunError("availability_cell_valid_support_mismatch")
        if sha_array(index_map) != availability_row.get("cell_index_map_sha256", {}).get(cell):
            raise GeometryRunError("availability_cell_map_array_sha_mismatch")
        if not np.array_equal(valid_saved & common, common):
            raise GeometryRunError("availability_common_support_not_valid_on_every_cell")
        maps[cell], valid_on_clock[cell], aperture_by_cell[cell] = index_map, valid_saved, aperture
        feature_records[cell] = record
        if cell == "q00":
            q00_clock_hash = sha_array(times)
    if not np.array_equal(availability["q00_timestamps_s"], _load_geometry_feature(
        _resolve_inside(binding["track_root"], feature_records["q00"]["feature_relative_path"]),
        feature_records["q00"], {**expected_context_identity, "cell": "q00", "ordered_eval_index": int(plan_row["ordered_eval_index"])}
    )[0]):
        raise GeometryRunError("availability_reference_clock_not_exact_q00_feature_clock")

    metric_values = np.full((len(CELLS), len(METRICS)), np.nan, dtype=np.float64)
    metric_status: list[list[str]] = [["INCOMPLETE" for _ in METRICS] for _ in CELLS]
    common_indices = np.flatnonzero(common)
    adjacent_indices = np.flatnonzero(adjacent)
    dt = np.diff(q00)
    cell_token_values = np.full((len(phone_tokens), len(CELLS)), np.nan, dtype=np.float64)
    token_rows: list[dict[str, Any]] = []
    token_index_by_phone_index = {int(token["phone_index"]): i for i, token in enumerate(phone_tokens)}
    phones = plan_row.get("recipient_phones", [])
    for token_index, phone in enumerate(phones):
        phone_index = int(phone["phone_index"])
        normalized, cls = _phone_class(phone.get("label"), core)
        if not cls:
            continue
        frozen = phone_tokens[token_index_by_phone_index[phone_index]]
        token_supported = bool(frozen["supported"])
        indices = np.asarray(frozen["q00_indices"], dtype=np.int32)
        if token_supported:
            for cell_index, cell in enumerate(CELLS):
                current = aperture_by_cell[cell][maps[cell][indices]]
                cell_token_values[token_index_by_phone_index[phone_index], cell_index] = float(np.mean(current, dtype=np.float64))
        token_rows.append({
            "context_order": int(ctx_index), "context_sample_id": sample,
            "speaker_id": str(plan_row["speaker_id"]), "direction": direction,
            "token_index": token_index, "phone_index": phone_index,
            "phone_label": str(phone["label"]), "normalized_phone": normalized,
            "phone_class": cls, "start_s": float(phone["start_s"]),
            "end_s": float(phone["end_s"]), "shifted_start_s": float(frozen["shifted_start_s"]),
            "shifted_end_s": float(frozen["shifted_end_s"]),
            "common_frame_count": int(frozen["common_frames"]),
            "token_supported": token_supported,
            "cell_mean_aperture": cell_token_values[token_index_by_phone_index[phone_index]].tolist(),
        })
    token_values_index = {int(row["phone_index"]): i for i, row in enumerate(phone_tokens)}
    class_means_by_cell = np.full((len(CELLS), 2), np.nan, dtype=np.float64)
    class_counts = np.zeros((len(CELLS), 2), dtype=np.int32)
    min_counts = availability_row["minimum_token_counts"]
    for cell_index, cell in enumerate(CELLS):
        for class_index, cls in enumerate(("bilabial", "open_vowel")):
            values = [
                cell_token_values[token_values_index[int(token["phone_index"])], cell_index]
                for token in phone_tokens
                if token["visual_class"] == cls and token["supported"]
            ]
            values = [float(value) for value in values if np.isfinite(value)]
            class_counts[cell_index, class_index] = len(values)
            if len(values) >= int(min_counts[cls]):
                class_means_by_cell[cell_index, class_index] = float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
        if np.isfinite(class_means_by_cell[cell_index]).all():
            metric_values[cell_index, 2] = float(class_means_by_cell[cell_index, 1] - class_means_by_cell[cell_index, 0])
            metric_status[cell_index][2] = "COMPLETE"
        else:
            metric_status[cell_index][2] = "INCOMPLETE_PHONE_CLASS_SUPPORT"
    for cell_index, cell in enumerate(CELLS):
        if len(common_indices):
            metric_values[cell_index, 0] = float(np.mean(aperture_by_cell[cell][maps[cell][common_indices]], dtype=np.float64))
            metric_status[cell_index][0] = "COMPLETE"
        else:
            metric_status[cell_index][0] = "INCOMPLETE_NO_COMMON_FRAMES"
        if len(adjacent_indices):
            left = maps[cell][adjacent_indices]
            right = maps[cell][adjacent_indices + 1]
            if np.any(left < 0) or np.any(right < 0) or np.any(dt[adjacent_indices] <= 0):
                raise GeometryRunError("velocity_support_has_unmapped_or_nonpositive_dt")
            velocity = np.abs(aperture_by_cell[cell][right] - aperture_by_cell[cell][left]) / dt[adjacent_indices]
            metric_values[cell_index, 1] = float(np.mean(velocity, dtype=np.float64))
            metric_status[cell_index][1] = "COMPLETE"
        else:
            metric_status[cell_index][1] = "INCOMPLETE_NO_COMMON_ADJACENT_PAIRS"
    result = {
        "context_order": int(ctx_index), "sample_id": sample,
        "speaker_id": str(plan_row["speaker_id"]), "direction": direction,
        "split": str(plan_row["split"]), "original_mfa_pair_qualified": bool(plan_row["original_mfa_pair_qualified"]),
        "grid_overlay_used": bool(plan_row.get("overlay_used")),
        "availability_endpoint_eligible": bool(availability_row["availability_endpoint_eligible"]),
        "availability_context_path": availability_row["availability_context_path"],
        "availability_context_sha256": availability_row["availability_context_sha256"],
        "common_frame_count": int(len(common_indices)), "common_adjacent_pair_count": int(len(adjacent_indices)),
        "q00_frame_count": int(n),
        "track_refs": {
            cell: {"feature_relative_path": feature_records[cell]["feature_relative_path"],
                  "feature_sha256": feature_records[cell]["feature_sha256"],
                  "availability_relative_path": feature_records[cell]["availability_relative_path"],
                  "availability_sha256": feature_records[cell]["availability_sha256"],
                  "array_inventory": feature_records[cell]["array_inventory"]}
            for cell in CELLS
        },
        "cell_metrics": {cell: {metric: (float(metric_values[i, j]) if np.isfinite(metric_values[i, j]) else None)
                                for j, metric in enumerate(METRICS)} for i, cell in enumerate(CELLS)},
        "cell_metric_status": {cell: {metric: metric_status[i][j] for j, metric in enumerate(METRICS)}
                                for i, cell in enumerate(CELLS)},
        "class_token_counts": {cell: {"bilabial": int(class_counts[i, 0]), "open_vowel": int(class_counts[i, 1])}
                               for i, cell in enumerate(CELLS)},
    }
    return metric_values, token_rows, result


def _contrast_values(cells: np.ndarray) -> np.ndarray:
    # cells is [..., cell] in CELLS order.
    q00, q10, q01, q11, self_grid = [cells[..., i] for i in range(5)]
    return np.stack((
        q10 - q00,
        q01 - q00,
        q11 - q00,
        self_grid - q00,
        q10 - q01,
        q10 - self_grid,
        q01 - self_grid,
        q11 - self_grid,
        q11 - q10 - q01 + q00,
    ), axis=-1)


def _bootstrap_family(
    speaker_values: np.ndarray,
    availability_mask: np.ndarray,
    endpoint_status: list[str],
) -> dict[str, np.ndarray]:
    roster_n = int(speaker_values.shape[1])
    if roster_n != 40 or speaker_values.shape != (54, 40) or availability_mask.shape != (54, 40):
        raise GeometryRunError("speaker_endpoint_array_shape_mismatch")
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = rng.integers(0, roster_n, size=(DRAWS, roster_n), dtype=np.int16)
    counts = np.stack([np.bincount(row, minlength=roster_n).astype(np.uint8) for row in indices])
    if indices.shape != (DRAWS, roster_n) or counts.shape != (DRAWS, roster_n):
        raise GeometryRunError("bootstrap_rng_output_shape_mismatch")
    denominators = np.zeros((54, DRAWS), dtype=np.int16)
    estimates = np.full((54, DRAWS), np.nan, dtype=np.float64)
    points = np.full(54, np.nan, dtype=np.float64)
    ses = np.full(54, np.nan, dtype=np.float64)
    deterministic = np.zeros(54, dtype=np.bool_)
    estimable = []
    for endpoint in range(54):
        mask = availability_mask[endpoint]
        selected = np.flatnonzero(mask)
        if not len(selected):
            endpoint_status[endpoint] = "INCOMPLETE_NO_FROZEN_ELIGIBLE_SPEAKERS"
            continue
        denominators[endpoint] = counts[:, selected].astype(np.int16).sum(axis=1, dtype=np.int64).astype(np.int16)
        if np.any(denominators[endpoint] == 0):
            raise GeometryRunError(f"bootstrap_zero_denominator:{endpoint}")
        values = speaker_values[endpoint, selected]
        if not np.isfinite(values).all():
            endpoint_status[endpoint] = "INCOMPLETE_FROZEN_SPEAKER_GEOMETRY_MISSING"
            continue
        points[endpoint] = float(np.mean(values, dtype=np.float64))
        if np.all(values == values[0]):
            estimates[endpoint] = points[endpoint]
            ses[endpoint] = 0.0
            deterministic[endpoint] = True
        else:
            weights = counts[:, selected].astype(np.float64)
            estimates[endpoint] = np.sum(weights * values[None, :], axis=1, dtype=np.float64) / denominators[endpoint].astype(np.float64)
            ses[endpoint] = float(np.std(estimates[endpoint], ddof=1, dtype=np.float64))
            if ses[endpoint] == 0.0:
                if not np.array_equal(estimates[endpoint], np.full(DRAWS, points[endpoint], dtype=np.float64)):
                    raise GeometryRunError(f"zero_se_without_deterministic_draws:{endpoint}")
                deterministic[endpoint] = True
        endpoint_status[endpoint] = "ESTIMABLE_DETERMINISTIC_ZERO_SE" if deterministic[endpoint] else "ESTIMABLE"
        estimable.append(endpoint)
    if not estimable:
        raise GeometryRunError("no_estimable_visual_endpoints")
    max_abs_t = np.zeros(DRAWS, dtype=np.float64)
    for endpoint in estimable:
        if ses[endpoint] == 0.0:
            continue
        standardized = np.abs((estimates[endpoint] - points[endpoint]) / ses[endpoint])
        max_abs_t = np.maximum(max_abs_t, standardized)
    critical = float(np.quantile(max_abs_t, 0.99, method="linear"))
    low = np.full(54, np.nan, dtype=np.float64)
    high = np.full(54, np.nan, dtype=np.float64)
    for endpoint in estimable:
        low[endpoint] = points[endpoint] - critical * ses[endpoint]
        high[endpoint] = points[endpoint] + critical * ses[endpoint]
    return {
        "bootstrap_indices_int16": indices,
        "bootstrap_counts_uint8": counts,
        "draw_denominators_int16": denominators,
        "bootstrap_estimates_float64": estimates,
        "max_abs_t_float64": max_abs_t,
        "estimable_endpoint_indices_int16": np.asarray(estimable, dtype=np.int16),
        "point_estimates_float64": points,
        "bootstrap_se_float64": ses,
        "ci_low_float64": low,
        "ci_high_float64": high,
        "critical_value_float64": np.asarray([critical], dtype=np.float64),
        "deterministic_zero_se_bool": deterministic,
    }


def _endpoint_records(
    speaker_cells: np.ndarray,
    phone_masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    # speaker_cells: [direction, speaker, cell, metric]
    endpoint_keys = []
    values = []
    masks = []
    for direction_index, direction in enumerate(DIRECTIONS):
        cells_by_metric = np.moveaxis(speaker_cells[direction_index], 1, -1)  # [speaker, metric, cell]
        contrast_tensor = _contrast_values(cells_by_metric)  # [speaker, metric, contrast]
        for metric_index, metric in enumerate(METRICS):
            for contrast_index, contrast in enumerate(CONTRASTS):
                endpoint_keys.append(f"{direction}::{metric}::{contrast}")
                values.append(contrast_tensor[:, metric_index, contrast_index])
                if metric == "phone_class_separation":
                    masks.append(phone_masks[direction].copy())
                else:
                    masks.append(np.ones(40, dtype=np.bool_))
    speaker_values = np.asarray(values, dtype=np.float64)
    availability_mask = np.asarray(masks, dtype=np.bool_)
    if speaker_values.shape != (54, 40) or availability_mask.shape != (54, 40):
        raise GeometryRunError("endpoint_order_or_count_mismatch")
    return endpoint_keys, speaker_values, availability_mask, {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise GeometryRunError("output_namespace_must_be_absent")
    binding = _load_and_validate_sources(args)
    core = _import_core(args.visual_core, binding["source_hashes"]["visual_core_source"]["raw_sha256"])
    plan_rows = sorted(binding["plan"]["context_rows"], key=lambda row: int(row["context_order"]))
    availability_rows = binding["availability_by_key"]
    context_values = np.full((160, 5, 3), np.nan, dtype=np.float64)
    context_status = np.empty((160, 5, 3), dtype="<U48")
    context_order = np.arange(160, dtype=np.int32)
    context_ids = np.empty(160, dtype="<U96")
    context_speakers = np.empty(160, dtype="<U32")
    context_directions = np.empty(160, dtype="<U32")
    context_rows = []
    token_records = []
    by_speaker_direction: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, plan_row in enumerate(plan_rows):
        identity = _context_identity(plan_row)
        availability_row = availability_rows[identity]
        values, tokens, context = _read_context(index, plan_row, availability_row, binding, core)
        context_values[index] = values
        context_status[index] = np.asarray(
            [[context["cell_metric_status"][cell][metric] for metric in METRICS] for cell in CELLS], dtype="<U48"
        )
        context_ids[index] = identity[0]
        context_speakers[index] = str(plan_row["speaker_id"])
        context_directions[index] = identity[1]
        context_rows.append(context)
        token_records.extend(tokens)
        by_speaker_direction[(str(plan_row["speaker_id"]), identity[1])].append(index)
    if len(context_rows) != 160 or len(token_records) == 0:
        raise GeometryRunError("geometry_context_or_phone_token_coverage_mismatch")

    speaker_roster = sorted({str(row["speaker_id"]) for row in plan_rows})
    if len(speaker_roster) != 40:
        raise GeometryRunError("speaker_roster_count_mismatch")
    availability_speakers = {(str(row["speaker_id"]), str(row["direction"])): row
                             for row in binding["availability_manifest"]["speaker_rows"]}
    if len(availability_speakers) != 80:
        raise GeometryRunError("availability_speaker_direction_rows_mismatch")
    speaker_cells = np.full((2, 40, 5, 3), np.nan, dtype=np.float64)
    phone_masks = {}
    for direction_index, direction in enumerate(DIRECTIONS):
        phone_masks[direction] = np.zeros(40, dtype=np.bool_)
        for speaker_index, speaker in enumerate(speaker_roster):
            indices = by_speaker_direction[(speaker, direction)]
            if len(indices) != 2:
                raise GeometryRunError(f"not_two_contexts_per_speaker_direction:{speaker}:{direction}")
            if len({context_ids[index] for index in indices}) != 2:
                raise GeometryRunError("duplicate_clip_in_speaker_direction")
            for cell_index in range(5):
                for metric_index in range(3):
                    clip_values = context_values[indices, cell_index, metric_index]
                    if np.isfinite(clip_values).all():
                        speaker_cells[direction_index, speaker_index, cell_index, metric_index] = float(np.mean(clip_values, dtype=np.float64))
            availability_speaker = availability_speakers[(speaker, direction)]
            if sorted(availability_speaker.get("clip_ids", [])) != sorted(str(context_ids[index]) for index in indices):
                raise GeometryRunError("availability_speaker_clip_pair_mismatch")
            phone_masks[direction][speaker_index] = bool(availability_speaker.get("complete_two_clip_phone_eligible"))
    if not all(int(mask.sum()) > 0 for mask in phone_masks.values()):
        raise GeometryRunError("frozen_phone_eligibility_has_zero_speakers")

    endpoint_keys, speaker_values, mask, _ = _endpoint_records(speaker_cells, phone_masks)
    statuses = []
    boot = _bootstrap_family(speaker_values, mask, statuses)
    points = boot["point_estimates_float64"]
    ses = boot["bootstrap_se_float64"]
    lows = boot["ci_low_float64"]
    highs = boot["ci_high_float64"]
    critical = boot["critical_value_float64"]
    endpoint_rows = []
    endpoint_index = 0
    for direction in DIRECTIONS:
        for metric in METRICS:
            for contrast in CONTRASTS:
                status = statuses[endpoint_index]
                endpoint_rows.append({
                    "endpoint_index": endpoint_index,
                    "endpoint_key": endpoint_keys[endpoint_index],
                    "direction": direction,
                    "metric": metric,
                    "contrast": contrast,
                    "estimand_mask": "frozen_phone_complete_pair" if metric == "phone_class_separation" else "all_40_speakers",
                    "availability_speaker_count": int(mask[endpoint_index].sum()),
                    "status": status,
                    "point_estimate": float(points[endpoint_index]) if np.isfinite(points[endpoint_index]) else None,
                    "bootstrap_se_sample_sd_ddof1": float(ses[endpoint_index]) if np.isfinite(ses[endpoint_index]) else None,
                    "simultaneous_99_ci": [float(lows[endpoint_index]), float(highs[endpoint_index])] if np.isfinite(lows[endpoint_index]) and np.isfinite(highs[endpoint_index]) else None,
                    "critical_value": float(critical[0]),
                    "zero_se_deterministic": bool(boot["deterministic_zero_se_bool"][endpoint_index]),
                })
                endpoint_index += 1
    if endpoint_index != 54:
        raise GeometryRunError("endpoint_count_not_54")

    token_records.sort(key=lambda row: (int(row["context_order"]), int(row["token_index"])))
    phone_npz = {
        "context_order_int32": np.asarray([row["context_order"] for row in token_records], dtype=np.int32),
        "sample_id_unicode": np.asarray([row["context_sample_id"] for row in token_records], dtype="<U96"),
        "speaker_id_unicode": np.asarray([row["speaker_id"] for row in token_records], dtype="<U32"),
        "direction_unicode": np.asarray([row["direction"] for row in token_records], dtype="<U32"),
        "token_index_int32": np.asarray([row["token_index"] for row in token_records], dtype=np.int32),
        "phone_index_int32": np.asarray([row["phone_index"] for row in token_records], dtype=np.int32),
        "phone_label_unicode": np.asarray([row["phone_label"] for row in token_records], dtype="<U64"),
        "normalized_phone_unicode": np.asarray([row["normalized_phone"] for row in token_records], dtype="<U64"),
        "phone_class_unicode": np.asarray([row["phone_class"] for row in token_records], dtype="<U16"),
        "start_s_float64": np.asarray([row["start_s"] for row in token_records], dtype=np.float64),
        "end_s_float64": np.asarray([row["end_s"] for row in token_records], dtype=np.float64),
        "shifted_start_s_float64": np.asarray([row["shifted_start_s"] for row in token_records], dtype=np.float64),
        "shifted_end_s_float64": np.asarray([row["shifted_end_s"] for row in token_records], dtype=np.float64),
        "common_frame_count_int32": np.asarray([row["common_frame_count"] for row in token_records], dtype=np.int32),
        "token_supported_bool": np.asarray([row["token_supported"] for row in token_records], dtype=np.bool_),
        "cell_mean_aperture_float64": np.asarray([row["cell_mean_aperture"] for row in token_records], dtype=np.float64),
    }

    speaker_npz = {
        "endpoint_keys_unicode": np.asarray(endpoint_keys, dtype="<U128"),
        "contrast_names_unicode": np.asarray(CONTRASTS, dtype="<U32"),
        "directions_unicode": np.asarray(DIRECTIONS, dtype="<U32"),
        "metric_names_unicode": np.asarray(METRICS, dtype="<U48"),
        "speaker_ids_unicode": np.asarray(speaker_roster, dtype="<U32"),
        "speaker_values_float64": speaker_values,
        "availability_mask_bool": mask,
        "point_estimates_float64": points,
        "bootstrap_se_float64": ses,
        "ci_low_float64": lows,
        "ci_high_float64": highs,
        "endpoint_status_unicode": np.asarray(statuses, dtype="<U64"),
        "critical_value_float64": critical,
        "deterministic_zero_se_bool": boot["deterministic_zero_se_bool"],
    }
    geometry_npz = {
        "context_order_int32": context_order,
        "sample_id_unicode": context_ids,
        "speaker_id_unicode": context_speakers,
        "direction_unicode": context_directions,
        "cell_names_unicode": np.asarray(CELLS, dtype="<U16"),
        "metric_names_unicode": np.asarray(METRICS, dtype="<U48"),
        "cell_metric_values_float64": context_values,
        "cell_metric_status_unicode": context_status,
    }

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "geometry_values.npz", **geometry_npz)
    np.savez_compressed(output / "phone_token_measurements.npz", **phone_npz)
    np.savez_compressed(output / "speaker_endpoints.npz", **speaker_npz)
    np.savez_compressed(output / "bootstrap_family.npz", **{k: boot[k] for k in (
        "bootstrap_indices_int16", "bootstrap_counts_uint8", "draw_denominators_int16",
        "bootstrap_estimates_float64", "max_abs_t_float64", "estimable_endpoint_indices_int16",
    )})

    artifacts = {}
    for name in OUTPUT_FILES:
        path = output / name
        artifacts[name] = {"path": name, "bytes": path.stat().st_size, "raw_sha256": sha_file(path)}
    manifest = {
        "schema": "world_shape_visual_geometry_family_v1",
        "status": "COMPLETE_WITH_INCOMPLETE_ENDPOINTS" if any(s.startswith("INCOMPLETE") for s in statuses) else "COMPLETE",
        "analysis_scope": "visual_geometry_only; no scores/evaluation outcomes/model forward read",
        "tracking_plan": {"path": str(binding["plan_path"]), "raw_sha256": binding["plan_raw"], "payload_sha256": binding["plan_payload"]},
        "tracking_extractor_source": binding["source_hashes"]["tracking_extractor_source"],
        "visual_design": {"path": str(args.design.resolve()), "raw_sha256": DESIGN_SCHEMA_SHA256},
        "visual_design_pin_correction": {"path": str(args.pin_correction.resolve()), "raw_sha256": DESIGN_PIN_CORRECTION_SHA256},
        "visual_core_source": binding["source_hashes"]["visual_core_source"],
        "tone_adapter_source": binding["source_hashes"]["tone_adapter_source"],
        "track_inputs": {
            "root": str(binding["track_root"]),
            "manifest_raw_sha256": binding["track_manifest_raw"],
            "manifest_payload_sha256": binding["track_payload"],
            "seal_raw_sha256": binding["track_seal_raw"],
            "files_inventory_raw_sha256": binding["track_inventory_raw"],
            "completed_videos": 800,
        },
        "availability_inputs": {
            "root": str(binding["availability_root"]),
            "manifest_raw_sha256": binding["availability_raw"],
            "manifest_payload_sha256": binding["availability_payload"],
            "seal_raw_sha256": binding["availability_seal_raw"],
            "files_inventory_raw_sha256": binding["availability_inventory_raw"],
            "geometry_was_read_before_freeze": False,
        },
        "geometry_release": {"path": str(args.geometry_release.resolve()), "raw_sha256": binding["geometry_release_raw"], "payload_sha256": binding["geometry_release_payload"]},
        "endpoint_design": {
            "directions": list(DIRECTIONS), "cells": list(CELLS), "metrics": list(METRICS), "contrasts": list(CONTRASTS),
            "endpoint_order": "direction-major, metric-major, contrast-minor",
            "endpoint_count": 54, "contexts": 160, "clips": 80, "speakers": 40, "video_tracks": 800,
            "latency_s": LATENCY_S, "fps_hz": FPS_HZ,
            "mean_aperture": "mean mouth_open over all common-valid q00-reference frames",
            "mean_abs_aperture_velocity_per_s": "mean absolute first difference over adjacent common-valid q00-reference pairs divided by exact q00 dt; holes are not bridged",
            "phone_class_separation": "within each clip, equal mean of supported token aperture means by class; open_vowel minus bilabial",
            "phone_classifier": "fixed old tone-stripped bilabial/open_vowel classes; recipient phones and original MFA qualification only",
        },
        "bootstrap": {
            "bit_generator": "PCG64", "seed": SEED, "draws": DRAWS, "draw_size": 40,
            "rng_call": "Generator(PCG64(seed)).integers(0, 40, size=(20000, 40), dtype=int16)",
            "count_construction": "row-wise bincount(minlength=40).astype(uint8)",
            "sample_unit": "speaker; one shared draw-count matrix across all endpoints",
            "point": "equal speaker mean; overall all 40, phone endpoints frozen complete-pair eligibility mask",
            "se": "bootstrap sample standard deviation, ddof=1",
            "simultaneous_ci": "studentized max-|t| across all estimable endpoints; np.quantile(method='linear', q=0.99); point +/- critical*SE",
            "zero_denominator_draws": 0,
        },
        "coverage": {
            "planned_contexts": 160, "complete_contexts": len(context_rows),
            "planned_video_tracks": 800, "complete_video_tracks": len(binding["track_by_key"]),
            "phone_token_rows": len(token_records), "planned_endpoints": 54,
            "estimable_endpoints": int(sum(s.startswith("ESTIMABLE") for s in statuses)),
            "incomplete_endpoints": int(sum(s.startswith("INCOMPLETE") for s in statuses)),
            "phone_eligible_speakers_by_direction": {direction: int(mask.sum()) for direction, mask in phone_masks.items()},
        },
        "context_rows": context_rows,
        "endpoint_rows": endpoint_rows,
        "artifact_inventory": artifacts,
        "mouth_geometry_read": True,
        "evaluation_scores_read": False,
        "evaluation_outcomes_read": False,
        "evaluation_effects_read": False,
        "model_forward_performed": False,
    }
    manifest["manifest_sha256"] = sha_bytes(canon(manifest))
    manifest_path = output / "geometry_family_manifest.json"
    _write_json_new(manifest_path, manifest)
    file_inventory = {name: sha_file(output / name) for name in (*OUTPUT_FILES, "geometry_family_manifest.json")}
    inventory = {"schema": "world_shape_visual_geometry_files_v1", "files": file_inventory}
    inventory_path = output / "all_files_sha256.json"
    _write_json_new(inventory_path, inventory)
    seal = {
        "schema": "world_shape_visual_geometry_seal_v1", "status": manifest["status"],
        "manifest_raw_sha256": sha_file(manifest_path), "manifest_payload_sha256": manifest["manifest_sha256"],
        "all_files_sha256": sha_file(inventory_path), "files": len(file_inventory),
        "contexts": 160, "videos": 800, "endpoints": 54,
        "evaluation_effects_read": False, "model_forward_performed": False,
    }
    _write_json_new(output / "geometry_family_seal.json", seal)
    return {
        "status": manifest["status"], "output_dir": str(output),
        "manifest_raw_sha256": sha_file(manifest_path), "manifest_payload_sha256": manifest["manifest_sha256"],
        "all_files_sha256": sha_file(inventory_path), "seal_raw_sha256": sha_file(output / "geometry_family_seal.json"),
        "contexts": 160, "videos": 800, "endpoints": 54,
        "estimable_endpoints": manifest["coverage"]["estimable_endpoints"],
        "evaluation_effects_read": False, "model_forward_performed": False,
    }


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extractor-source", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--evaluation-design", type=Path, required=True)
    parser.add_argument("--pin-correction", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--landmarker", type=Path, required=True)
    parser.add_argument("--visual-core", type=Path, required=True)
    parser.add_argument("--tone-adapter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--geometry-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
