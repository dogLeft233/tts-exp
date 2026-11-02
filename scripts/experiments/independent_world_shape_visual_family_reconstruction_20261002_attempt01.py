#!/usr/bin/env python3
"""Independently rebuild the frozen 54-endpoint WORLD visual family.

This verifier deliberately does not import the primary geometry-family
consumer. It validates the tracking and geometry-blind availability seals,
recomputes the five-cell clip geometry from the sealed track NPZs, then
reconstructs speaker endpoints and the shared PCG64/max-|t| family.

The primary geometry consumer's output is treated as data: every output file
is hash-checked, but no primary computation is imported or called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PLAN_RAW_SHA256 = "13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f"
EXPECTED_PLAN_PAYLOAD_SHA256 = "26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6"
EXPECTED_VISUAL_DESIGN_SHA256 = "0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8"
EXPECTED_PIN_CORRECTION_SHA256 = "cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6"
EXPECTED_EXTRACTOR_SHA256 = "aa86f4011e4a93e345471142ce29d6fcb739d1d4d89254f95f7685d2bf7fb41f"
EXPECTED_FEATURE_HELPER_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
EXPECTED_VISUAL_CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
EXPECTED_TONE_ADAPTER_SHA256 = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
EXPECTED_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
METRICS = (
    "mean_aperture",
    "mean_absolute_aperture_velocity_per_s",
    "open_vowel_minus_bilabial_equal_token_mean",
)
CONTRASTS = (
    "mean_q10_minus_q00",
    "trajectory_q01_minus_q00",
    "joint_q11_minus_q00",
    "grid_cycle_SELF_GRID_minus_q00",
    "mean_minus_trajectory_q10_minus_q01",
    "mean_beyond_grid_q10_minus_SELF_GRID",
    "trajectory_beyond_grid_q01_minus_SELF_GRID",
    "joint_beyond_grid_q11_minus_SELF_GRID",
    "factorial_interaction_q11_minus_q10_minus_q01_plus_q00",
)
CONTRAST_COEFFICIENTS = {
    "mean_q10_minus_q00": {"q10": 1.0, "q00": -1.0},
    "trajectory_q01_minus_q00": {"q01": 1.0, "q00": -1.0},
    "joint_q11_minus_q00": {"q11": 1.0, "q00": -1.0},
    "grid_cycle_SELF_GRID_minus_q00": {"SELF_GRID": 1.0, "q00": -1.0},
    "mean_minus_trajectory_q10_minus_q01": {"q10": 1.0, "q01": -1.0},
    "mean_beyond_grid_q10_minus_SELF_GRID": {"q10": 1.0, "SELF_GRID": -1.0},
    "trajectory_beyond_grid_q01_minus_SELF_GRID": {"q01": 1.0, "SELF_GRID": -1.0},
    "joint_beyond_grid_q11_minus_SELF_GRID": {"q11": 1.0, "SELF_GRID": -1.0},
    "factorial_interaction_q11_minus_q10_minus_q01_plus_q00": {
        "q11": 1.0,
        "q10": -1.0,
        "q01": -1.0,
        "q00": 1.0,
    },
}
EXPECTED_CONTEXTS = 160
EXPECTED_CLIPS = 80
EXPECTED_SPEAKERS = 40
EXPECTED_TRACKS = 800
EXPECTED_ENDPOINTS = 54
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_930
LATENCY_S = -0.085
FPS = 25.0
METRIC_RTOL = 2e-10
METRIC_ATOL = 2e-12
FAMILY_RTOL = 2e-10
FAMILY_ATOL = 2e-12
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return bytes_sha256(np.ascontiguousarray(value).tobytes())


def canon_sha256(value: dict[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return bytes_sha256(canonical)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise VerificationError(f"expected_object:{path}")
    return value


def read_inventory(path: Path) -> dict[str, str]:
    obj = read_json(path)
    mapping = obj.get("files", obj)
    if not isinstance(mapping, dict):
        raise VerificationError(f"inventory_not_mapping:{path}")
    result: dict[str, str] = {}
    for rel, digest in mapping.items():
        if not isinstance(rel, str) or not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            raise VerificationError(f"inventory_entry_invalid:{path}")
        result[rel.replace("\\", "/")] = digest
    return result


def verify_fingerprint(obj: dict[str, Any], field: str, label: str) -> str:
    expected = obj.get(field)
    actual = canon_sha256(obj, field)
    if expected != actual:
        raise VerificationError(f"embedded_fingerprint_mismatch:{label}:{field}")
    return actual


def safe_join(root: Path, rel: str) -> Path:
    posix = Path(rel.replace("\\", "/"))
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts):
        raise VerificationError(f"unsafe_relative_path:{rel}")
    base = root.resolve(strict=True)
    candidate = (base / posix).resolve(strict=True)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise VerificationError(f"path_escapes_root:{rel}") from exc
    return candidate


def verify_tree_inventory(root: Path, inventory_path: Path, inventory: dict[str, str], allowed_extra: set[str]) -> None:
    root = root.resolve(strict=True)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"symlink_in_sealed_tree:{path.relative_to(root)}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    expected = set(inventory) | allowed_extra
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise VerificationError(f"tree_inventory_set_mismatch:missing={missing}:extra={extra}")
    for rel, expected_sha in inventory.items():
        path = safe_join(root, rel)
        if sha256_file(path) != expected_sha:
            raise VerificationError(f"tree_inventory_hash_mismatch:{rel}")
    if inventory_path.resolve(strict=True).parent != root:
        raise VerificationError("inventory_path_outside_tree")


def verify_plan(plan_path: Path) -> dict[str, Any]:
    raw = sha256_file(plan_path)
    if raw != EXPECTED_PLAN_RAW_SHA256:
        raise VerificationError("tracking_plan_raw_sha_mismatch")
    plan = read_json(plan_path)
    payload = verify_fingerprint(plan, "plan_sha256", "tracking_plan")
    if payload != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise VerificationError("tracking_plan_payload_sha_mismatch")
    if plan.get("schema") != "world_shape_visual_tracking_plan_v1":
        raise VerificationError("tracking_plan_schema_mismatch")
    design = plan.get("design", {})
    if design.get("visual_design_raw_sha256") != EXPECTED_VISUAL_DESIGN_SHA256:
        raise VerificationError("visual_design_pin_mismatch")
    if design.get("execution_pin_correction_raw_sha256") != EXPECTED_PIN_CORRECTION_SHA256:
        raise VerificationError("execution_pin_correction_mismatch")
    seals = plan.get("input_seals", {})
    expected = {
        "tracking_extractor_source": EXPECTED_EXTRACTOR_SHA256,
        "feature_helper": EXPECTED_FEATURE_HELPER_SHA256,
        "visual_core_source": EXPECTED_VISUAL_CORE_SHA256,
        "tone_adapter_source": EXPECTED_TONE_ADAPTER_SHA256,
        "landmarker_asset": EXPECTED_LANDMARKER_SHA256,
        "visual_design": EXPECTED_VISUAL_DESIGN_SHA256,
        "execution_pin_correction": EXPECTED_PIN_CORRECTION_SHA256,
    }
    for name, digest in expected.items():
        item = seals.get(name, {})
        if item.get("raw_sha256") != digest:
            raise VerificationError(f"tracking_plan_input_pin_mismatch:{name}")
    if len(plan.get("context_rows", [])) != EXPECTED_CONTEXTS:
        raise VerificationError("tracking_plan_context_count_mismatch")
    if plan.get("cohort", {}).get("videos") != EXPECTED_TRACKS:
        raise VerificationError("tracking_plan_video_count_mismatch")
    if tuple(plan.get("cohort", {}).get("cells", ())) != CELLS:
        raise VerificationError("tracking_plan_cell_order_mismatch")
    return plan


def array_digest_entry(array: np.ndarray) -> dict[str, Any]:
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_sha256(array)}


def verify_npz_array_inventory(npz: Any, expected: dict[str, Any], record_key: str) -> dict[str, np.ndarray]:
    required = {
        "timestamps_s",
        "valid",
        "mouth_open",
        "mouth_width",
        "mouth_height",
        "mouth_area",
        "landmarks",
        "canonical_mouth",
    }
    if set(expected) != required:
        raise VerificationError(f"track_array_inventory_names_invalid:{record_key}")
    if set(npz.files) != required | {"metadata_json"}:
        raise VerificationError(f"track_npz_keys_invalid:{record_key}")
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(required):
        array = np.asarray(npz[name])
        ref = expected[name]
        if ref.get("shape") != list(array.shape) or ref.get("dtype") != str(array.dtype):
            raise VerificationError(f"track_array_descriptor_mismatch:{record_key}:{name}")
        if ref.get("sha256") != array_sha256(array):
            raise VerificationError(f"track_array_hash_mismatch:{record_key}:{name}")
        arrays[name] = array
    return arrays


@dataclass
class ContextTracks:
    plan_row: dict[str, Any]
    availability_row: dict[str, Any]
    q00_times: np.ndarray
    common_valid: np.ndarray
    common_adjacent: np.ndarray
    maps: dict[str, np.ndarray]
    arrays: dict[str, dict[str, np.ndarray]]
    tokens: list[dict[str, Any]]


def verify_track_and_availability_inputs(
    plan: dict[str, Any], track_root: Path, availability_root: Path
) -> tuple[list[ContextTracks], dict[str, Any], dict[str, Any]]:
    track_manifest_path = track_root / "visual_track_manifest.json"
    track_seal_path = track_root / "visual_track_seal.json"
    track_inventory_path = track_root / "all_files_sha256.json"
    track_manifest = read_json(track_manifest_path)
    track_seal = read_json(track_seal_path)
    track_inventory = read_inventory(track_inventory_path)
    track_payload = verify_fingerprint(track_manifest, "manifest_sha256", "track_manifest")
    track_raw = sha256_file(track_manifest_path)
    track_inventory_raw = sha256_file(track_inventory_path)
    if track_manifest.get("status") != "COMPLETE" or track_manifest.get("intended_videos") != EXPECTED_TRACKS or track_manifest.get("completed_videos") != EXPECTED_TRACKS or track_manifest.get("failed_videos") != 0:
        raise VerificationError("track_manifest_not_complete")
    if track_seal.get("status") != "SEALED" or track_seal.get("manifest_sha256") != track_raw or track_seal.get("manifest_payload_sha256") != track_payload or track_seal.get("all_files_sha256") != track_inventory_raw:
        raise VerificationError("track_seal_binding_mismatch")
    verify_tree_inventory(track_root, track_inventory_path, track_inventory, {"visual_track_seal.json", "all_files_sha256.json"})

    av_manifest_path = availability_root / "availability_freeze.json"
    av_seal_path = availability_root / "availability_seal.json"
    av_inventory_path = availability_root / "all_files_sha256.json"
    av_manifest = read_json(av_manifest_path)
    av_seal = read_json(av_seal_path)
    av_inventory = read_inventory(av_inventory_path)
    av_payload = verify_fingerprint(av_manifest, "manifest_sha256", "availability_freeze")
    av_raw = sha256_file(av_manifest_path)
    av_inventory_raw = sha256_file(av_inventory_path)
    if av_manifest.get("schema") != "world_shape_visual_availability_freeze_v1" or av_manifest.get("status") != "FROZEN_BEFORE_GEOMETRY_READ":
        raise VerificationError("availability_freeze_status_invalid")
    if av_manifest.get("geometry_values_read") is not False or av_manifest.get("evaluation_effects_read") is not False:
        raise VerificationError("availability_freeze_not_geometry_blind")
    if av_manifest.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256 or av_manifest.get("tracking_plan_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise VerificationError("availability_plan_binding_mismatch")
    if av_manifest.get("track_manifest_raw_sha256") != track_raw or av_manifest.get("track_manifest_payload_sha256") != track_payload:
        raise VerificationError("availability_track_manifest_binding_mismatch")
    if av_manifest.get("track_seal_raw_sha256") != sha256_file(track_seal_path) or av_manifest.get("track_files_inventory_raw_sha256") != track_inventory_raw:
        raise VerificationError("availability_track_inventory_binding_mismatch")
    if av_seal.get("status") != "FROZEN_BEFORE_GEOMETRY_READ" or av_seal.get("manifest_raw_sha256") != av_raw or av_seal.get("manifest_payload_sha256") != av_payload or av_seal.get("all_files_sha256") != av_inventory_raw:
        raise VerificationError("availability_seal_binding_mismatch")
    verify_tree_inventory(availability_root, av_inventory_path, av_inventory, {"availability_seal.json", "all_files_sha256.json"})

    plan_rows = plan["context_rows"]
    if len({(row["sample_id"], row["direction"]) for row in plan_rows}) != EXPECTED_CONTEXTS:
        raise VerificationError("tracking_plan_context_identity_not_unique")
    records = track_manifest.get("records", [])
    rec_by_key = {(r.get("sample_id"), r.get("direction"), r.get("cell")): r for r in records}
    if len(records) != EXPECTED_TRACKS or len(rec_by_key) != EXPECTED_TRACKS:
        raise VerificationError("track_manifest_record_coverage_invalid")
    av_rows = av_manifest.get("context_rows", [])
    av_by_key = {(r.get("sample_id"), r.get("direction")): r for r in av_rows}
    if len(av_rows) != EXPECTED_CONTEXTS or len(av_by_key) != EXPECTED_CONTEXTS:
        raise VerificationError("availability_context_coverage_invalid")

    contexts: list[ContextTracks] = []
    observed_track_refs: set[str] = set()
    observed_av_refs: set[str] = set()
    for plan_row in sorted(plan_rows, key=lambda row: int(row["context_order"])):
        key = (plan_row["sample_id"], plan_row["direction"])
        av_row = av_by_key.get(key)
        if av_row is None:
            raise VerificationError(f"availability_context_missing:{key}")
        for field in ("context_order", "speaker_id", "recipient_grid_sha256", "original_mfa_pair_qualified"):
            if av_row.get(field) != plan_row.get(field):
                if field == "original_mfa_pair_qualified" and av_row.get("original_mfa_pair_qualified") == plan_row.get(field):
                    continue
                raise VerificationError(f"availability_plan_identity_mismatch:{key}:{field}")
        if av_row.get("recipient_phones") != plan_row.get("recipient_phones"):
            raise VerificationError(f"availability_phone_tier_mismatch:{key}")
        av_context_path = safe_join(availability_root, av_row["availability_context_path"])
        if sha256_file(av_context_path) != av_row["availability_context_sha256"]:
            raise VerificationError(f"availability_context_file_hash_mismatch:{key}")
        observed_av_refs.add(av_row["availability_context_path"].replace("\\", "/"))
        with np.load(av_context_path, allow_pickle=False) as archive:
            expected_context_keys = {
                "q00_timestamps_s", "common_valid", "common_adjacent",
                *{f"index_map_{cell}" for cell in CELLS},
                *{f"valid_on_q00_{cell}" for cell in CELLS},
            }
            if set(archive.files) != expected_context_keys:
                raise VerificationError(f"availability_context_npz_keys_invalid:{key}")
            q00_times = np.asarray(archive["q00_timestamps_s"])
            common_valid = np.asarray(archive["common_valid"])
            common_adjacent = np.asarray(archive["common_adjacent"])
            maps = {cell: np.asarray(archive[f"index_map_{cell}"]) for cell in CELLS}
            valid_on_q00 = {cell: np.asarray(archive[f"valid_on_q00_{cell}"]) for cell in CELLS}
        if q00_times.dtype != np.float64 or common_valid.dtype != np.bool_ or common_adjacent.dtype != np.bool_:
            raise VerificationError(f"availability_context_dtype_invalid:{key}")
        if q00_times.ndim != 1 or common_valid.shape != q00_times.shape or common_adjacent.shape != (max(0, len(q00_times) - 1),):
            raise VerificationError(f"availability_context_shape_invalid:{key}")
        if not np.isfinite(q00_times).all() or (len(q00_times) > 1 and np.any(np.diff(q00_times) <= 0)):
            raise VerificationError(f"availability_q00_clock_invalid:{key}")
        if len(q00_times) > 1 and not np.allclose(np.diff(q00_times), 1.0 / FPS, rtol=0, atol=1e-9):
            raise VerificationError(f"availability_q00_fps_invalid:{key}")
        if array_sha256(q00_times) != av_row.get("q00_clock_sha256") or array_sha256(common_valid) != av_row.get("common_valid_sha256") or array_sha256(common_adjacent) != av_row.get("common_adjacent_sha256"):
            raise VerificationError(f"availability_context_array_hash_mismatch:{key}")
        if not np.array_equal(common_adjacent, common_valid[:-1] & common_valid[1:]):
            raise VerificationError(f"availability_adjacent_mask_mismatch:{key}")
        for cell in CELLS:
            mapping = maps[cell]
            cell_valid = valid_on_q00[cell]
            if mapping.dtype != np.int32 or mapping.shape != q00_times.shape or cell_valid.dtype != np.bool_ or cell_valid.shape != q00_times.shape:
                raise VerificationError(f"availability_cell_map_shape_dtype_invalid:{key}:{cell}")
            if array_sha256(mapping) != av_row.get("cell_index_map_sha256", {}).get(cell):
                raise VerificationError(f"availability_cell_map_hash_mismatch:{key}:{cell}")
        if not np.array_equal(maps["q00"], np.arange(len(q00_times), dtype=np.int32)):
            raise VerificationError(f"availability_q00_map_not_identity:{key}")

        arrays_by_cell: dict[str, dict[str, np.ndarray]] = {}
        for cell in CELLS:
            record = rec_by_key.get((plan_row["sample_id"], plan_row["direction"], cell))
            if record is None or record.get("status") != "complete":
                raise VerificationError(f"track_record_missing_or_failed:{key}:{cell}")
            plan_cell = plan_row["cells"][cell]
            if record.get("speaker_id") != plan_row["speaker_id"] or record.get("context_order") != plan_row["context_order"]:
                raise VerificationError(f"track_identity_mismatch:{key}:{cell}")
            for field in ("video_relative_path", "video_sha256", "video_bytes", "official_frame_count"):
                if record.get(field) != plan_cell.get(field):
                    raise VerificationError(f"track_plan_video_binding_mismatch:{key}:{cell}:{field}")
            feature_rel = record["feature_relative_path"].replace("\\", "/")
            availability_rel = record["availability_relative_path"].replace("\\", "/")
            observed_track_refs.update((feature_rel, availability_rel))
            feature_path = safe_join(track_root, feature_rel)
            availability_path = safe_join(track_root, availability_rel)
            feature_sha = sha256_file(feature_path)
            availability_sha = sha256_file(availability_path)
            if feature_sha != record.get("feature_sha256") or track_inventory.get(feature_rel) != feature_sha:
                raise VerificationError(f"feature_file_hash_mismatch:{key}:{cell}")
            if availability_sha != record.get("availability_sha256") or track_inventory.get(availability_rel) != availability_sha:
                raise VerificationError(f"track_availability_file_hash_mismatch:{key}:{cell}")
            with np.load(feature_path, allow_pickle=False) as archive:
                feature_arrays = verify_npz_array_inventory(archive, record.get("array_inventory", {}), f"{key}:{cell}")
                metadata = json.loads(str(archive["metadata_json"].item()))
            expected_meta = {
                "sample_id": plan_row["sample_id"],
                "speaker_id": plan_row["speaker_id"],
                "direction": plan_row["direction"],
                "cell": cell,
                "video_relative_path": plan_cell["video_relative_path"],
                "video_sha256": plan_cell["video_sha256"],
                "recipient_grid_sha256": plan_row["recipient_grid_sha256"],
                "landmarker_sha256": EXPECTED_LANDMARKER_SHA256,
                "video_features_sha256": EXPECTED_FEATURE_HELPER_SHA256,
            }
            for field, expected_value in expected_meta.items():
                if metadata.get(field) != expected_value:
                    raise VerificationError(f"feature_metadata_binding_mismatch:{key}:{cell}:{field}")
            times = feature_arrays["timestamps_s"]
            valid = feature_arrays["valid"]
            if times.dtype != np.float64 or valid.dtype != np.bool_ or times.ndim != 1 or valid.shape != times.shape:
                raise VerificationError(f"feature_clock_shape_dtype_invalid:{key}:{cell}")
            if len(times) != plan_cell["official_frame_count"] or len(times) != record.get("official_frame_count"):
                raise VerificationError(f"feature_frame_count_mismatch:{key}:{cell}")
            if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
                raise VerificationError(f"feature_clock_not_monotonic:{key}:{cell}")
            if len(times) > 1 and not np.allclose(np.diff(times), 1.0 / FPS, rtol=0, atol=1e-9):
                raise VerificationError(f"feature_clock_fps_mismatch:{key}:{cell}")
            if array_sha256(times) != record.get("timestamp_sha256") or int(valid.sum()) != record.get("valid_frames"):
                raise VerificationError(f"feature_availability_descriptor_mismatch:{key}:{cell}")
            mapping = maps[cell]
            found = mapping >= 0
            aligned_valid = np.zeros(mapping.shape, dtype=np.bool_)
            aligned_valid[found] = valid[mapping[found]]
            if not np.array_equal(aligned_valid, valid_on_q00[cell]):
                raise VerificationError(f"feature_validity_map_mismatch:{key}:{cell}")
            if np.any(common_valid & (mapping < 0)) or np.any(common_valid & ~aligned_valid):
                raise VerificationError(f"common_mask_not_supported_by_track:{key}:{cell}")
            with np.load(availability_path, allow_pickle=False) as availability_archive:
                if set(availability_archive.files) != {"timestamps_s", "valid"}:
                    raise VerificationError(f"track_availability_npz_keys_invalid:{key}:{cell}")
                av_times = np.asarray(availability_archive["timestamps_s"])
                av_valid = np.asarray(availability_archive["valid"])
            if av_times.dtype != np.float64 or av_valid.dtype != np.bool_ or not np.array_equal(av_times, times) or not np.array_equal(av_valid, valid):
                raise VerificationError(f"track_availability_npz_not_feature_identity:{key}:{cell}")
            arrays_by_cell[cell] = feature_arrays
        tokens = av_row.get("tokens", [])
        contexts.append(ContextTracks(plan_row, av_row, q00_times, common_valid, common_adjacent, maps, arrays_by_cell, tokens))

    if observed_track_refs != {k for k in track_inventory if k.startswith("features/") or k.startswith("availability/")}:
        raise VerificationError("track_inventory_has_unbound_or_missing_feature_refs")
    expected_av_contexts = {r["availability_context_path"].replace("\\", "/") for r in av_rows}
    if expected_av_contexts != {k for k in av_inventory if k.startswith("contexts/")}:
        raise VerificationError("availability_inventory_context_refs_mismatch")
    if len(contexts) != EXPECTED_CONTEXTS:
        raise VerificationError("verified_context_count_mismatch")
    return contexts, track_manifest, av_manifest


def _token_means_for_context(context: ContextTracks, values: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    by_class: dict[str, list[float]] = {"bilabial": [], "open_vowel": []}
    token_means: dict[str, Any] = {}
    for token in context.tokens:
        visual_class = token.get("visual_class")
        if visual_class not in by_class:
            continue
        supported = bool(token.get("supported"))
        indices = np.asarray(token.get("q00_indices", []), dtype=np.int64)
        frame_count = int(token.get("common_frames", -1))
        if frame_count != int(indices.size):
            raise VerificationError("availability_token_frame_count_mismatch")
        if np.any(indices < 0) or np.any(indices >= len(values)) or (indices.size and not context.common_valid[indices].all()):
            raise VerificationError("availability_token_indices_outside_common_support")
        if supported:
            if not indices.size:
                raise VerificationError("supported_phone_token_has_no_common_frames")
            vals = values[indices]
            if not np.isfinite(vals).all():
                raise VerificationError("nonfinite_geometry_on_supported_phone_token")
            mean = float(np.mean(vals, dtype=np.float64))
            by_class[visual_class].append(mean)
            token_means[str(token["phone_index"])] = mean
        else:
            token_means[str(token["phone_index"])] = None
    if not bool(context.availability_row.get("availability_endpoint_eligible")):
        return None, token_means
    if not by_class["bilabial"] or not by_class["open_vowel"]:
        raise VerificationError("frozen_phone_eligible_context_lacks_supported_class")
    return float(np.mean(by_class["open_vowel"], dtype=np.float64) - np.mean(by_class["bilabial"], dtype=np.float64)), token_means


def recompute_context_geometry(contexts: list[ContextTracks]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values = np.full((EXPECTED_CONTEXTS, len(CELLS), len(METRICS)), np.nan, dtype=np.float64)
    statuses = np.full(values.shape, "INCOMPLETE", dtype="U40")
    token_rows: list[dict[str, Any]] = []
    for ci, context in enumerate(sorted(contexts, key=lambda item: int(item.plan_row["context_order"]))):
        if int(context.plan_row["context_order"]) != ci:
            raise VerificationError("context_order_not_contiguous")
        has_frames = bool(context.common_valid.any())
        has_adjacent = bool(context.common_adjacent.any())
        for cell_index, cell in enumerate(CELLS):
            mapping = context.maps[cell]
            mouth_open = context.arrays[cell]["mouth_open"]
            aligned = np.full(mapping.shape, np.nan, dtype=np.float64)
            found = mapping >= 0
            aligned[found] = mouth_open[mapping[found]]
            if has_frames and np.isfinite(aligned[context.common_valid]).all():
                values[ci, cell_index, 0] = float(np.mean(aligned[context.common_valid], dtype=np.float64))
                statuses[ci, cell_index, 0] = "COMPLETE"
            if has_adjacent:
                pair_values = np.abs(np.diff(aligned) / np.diff(context.q00_times))
                if np.isfinite(pair_values[context.common_adjacent]).all():
                    values[ci, cell_index, 1] = float(np.mean(pair_values[context.common_adjacent], dtype=np.float64))
                    statuses[ci, cell_index, 1] = "COMPLETE"
            if bool(context.availability_row.get("availability_endpoint_eligible")):
                separation, token_means = _token_means_for_context(context, aligned)
                if separation is not None and math.isfinite(separation):
                    values[ci, cell_index, 2] = separation
                    statuses[ci, cell_index, 2] = "COMPLETE"
            else:
                token_means = {}
            if cell_index == 0:
                for token in context.tokens:
                    token_rows.append({
                        "context_order": ci,
                        "sample_id": context.plan_row["sample_id"],
                        "speaker_id": context.plan_row["speaker_id"],
                        "direction": context.plan_row["direction"],
                        "phone_index": token.get("phone_index"),
                        "visual_class": token.get("visual_class"),
                        "raw_label": token.get("raw_label"),
                        "start_s": token.get("start_s"),
                        "end_s": token.get("end_s"),
                        "supported": bool(token.get("supported")),
                        "frame_count": int(token.get("common_frames", -1)),
                    })
        if context.availability_row.get("availability_endpoint_eligible"):
            # Require the same supported token inventory for all five cells.
            phone_separations = values[ci, :, 2]
            if not np.isfinite(phone_separations).all():
                statuses[ci, :, 2] = "INCOMPLETE_GEOMETRY"
    return values, statuses, token_rows


def expected_endpoint_order() -> list[tuple[str, str, str]]:
    return [(direction, metric, contrast) for direction in DIRECTIONS for metric in METRICS for contrast in CONTRASTS]


def rebuild_family(
    contexts: list[ContextTracks], cell_values: np.ndarray
) -> dict[str, np.ndarray | list[str] | list[tuple[str, str, str]]]:
    if cell_values.shape != (EXPECTED_CONTEXTS, len(CELLS), len(METRICS)):
        raise VerificationError("cell_metric_values_shape_invalid")
    roster = sorted({str(context.plan_row["speaker_id"]) for context in contexts})
    if len(roster) != EXPECTED_SPEAKERS:
        raise VerificationError("speaker_roster_count_invalid")
    plan_roster = list((contexts[0].plan_row.get("_unused", []) if contexts else []))
    del plan_roster
    speaker_index = {speaker: index for index, speaker in enumerate(roster)}
    context_index = {(context.plan_row["sample_id"], context.plan_row["direction"]): index for index, context in enumerate(sorted(contexts, key=lambda item: int(item.plan_row["context_order"])))}
    av_speaker_rows = {(row["speaker_id"], row["direction"]): row for row in contexts[0].availability_row.get("_speaker_rows", [])} if contexts else {}
    # Rebuild complete-two-clip phone eligibility from the frozen context rows,
    # preserving each direction's full 40-speaker roster.
    phone_eligible = {(speaker, direction): True for speaker in roster for direction in DIRECTIONS}
    by_speaker_direction: dict[tuple[str, str], list[int]] = {}
    for context in contexts:
        identity = (str(context.plan_row["speaker_id"]), str(context.plan_row["direction"]))
        by_speaker_direction.setdefault(identity, []).append(context_index[(context.plan_row["sample_id"], context.plan_row["direction"])])
        if not bool(context.availability_row.get("availability_endpoint_eligible")):
            phone_eligible[identity] = False
    expected_groups = {(speaker, direction) for speaker in roster for direction in DIRECTIONS}
    if set(by_speaker_direction) != expected_groups:
        raise VerificationError("speaker_direction_group_coverage_invalid")
    for group, indices in by_speaker_direction.items():
        if len(indices) != 2:
            raise VerificationError(f"speaker_direction_not_two_clips:{group}")
    del av_speaker_rows

    endpoint_keys = expected_endpoint_order()
    speaker_values = np.full((EXPECTED_ENDPOINTS, EXPECTED_SPEAKERS), np.nan, dtype=np.float64)
    availability_mask = np.zeros((EXPECTED_ENDPOINTS, EXPECTED_SPEAKERS), dtype=np.bool_)
    endpoint_status = np.full(EXPECTED_ENDPOINTS, "INCOMPLETE", dtype="U40")
    point = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)

    for ep, (direction, metric, contrast_name) in enumerate(endpoint_keys):
        metric_index = METRICS.index(metric)
        contrast = CONTRAST_COEFFICIENTS[contrast_name]
        is_phone = metric == "open_vowel_minus_bilabial_equal_token_mean"
        mask = np.asarray([
            phone_eligible[(speaker, direction)] if is_phone else True
            for speaker in roster
        ], dtype=np.bool_)
        availability_mask[ep] = mask
        if not mask.any():
            raise VerificationError(f"endpoint_has_empty_frozen_mask:{ep}")
        needed_cells = [CELLS.index(cell) for cell in contrast]
        for speaker, sidx in speaker_index.items():
            clip_indices = by_speaker_direction[(speaker, direction)]
            clip_indices = sorted(clip_indices)
            per_speaker = 0.0
            valid_value = True
            for cell, coefficient in contrast.items():
                cidx = CELLS.index(cell)
                pair = cell_values[clip_indices, cidx, metric_index]
                if not np.isfinite(pair).all():
                    valid_value = False
                    break
                per_speaker += coefficient * float(np.mean(pair, dtype=np.float64))
            if valid_value:
                speaker_values[ep, sidx] = per_speaker
        if not np.isfinite(speaker_values[ep, mask]).all():
            endpoint_status[ep] = "INCOMPLETE_GEOMETRY"
            continue
        endpoint_status[ep] = "ESTIMABLE"
        point[ep] = float(np.mean(speaker_values[ep, mask], dtype=np.float64))

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_indices = rng.integers(0, EXPECTED_SPEAKERS, size=(BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.int16)
    bootstrap_counts = np.zeros((BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.int16)
    for draw in range(BOOTSTRAP_DRAWS):
        bootstrap_counts[draw] = np.bincount(bootstrap_indices[draw], minlength=EXPECTED_SPEAKERS).astype(np.int16)

    draw_denominators = np.zeros((EXPECTED_ENDPOINTS, BOOTSTRAP_DRAWS), dtype=np.int16)
    bootstrap_means = np.full((EXPECTED_ENDPOINTS, BOOTSTRAP_DRAWS), np.nan, dtype=np.float64)
    se = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)
    for ep in range(EXPECTED_ENDPOINTS):
        mask = availability_mask[ep]
        counts = bootstrap_counts[:, mask].astype(np.float64)
        denominator = counts.sum(axis=1).astype(np.int16)
        if np.any(denominator <= 0):
            raise VerificationError(f"zero_bootstrap_denominator:{ep}")
        draw_denominators[ep] = denominator
        if endpoint_status[ep] != "ESTIMABLE":
            continue
        vals = speaker_values[ep, mask]
        draws = counts @ vals / denominator.astype(np.float64)
        if not np.isfinite(draws).all():
            raise VerificationError(f"nonfinite_bootstrap_estimate:{ep}")
        bootstrap_means[ep] = draws
        se[ep] = float(np.std(draws, ddof=1, dtype=np.float64))
        if se[ep] == 0.0 and not np.allclose(draws, point[ep], rtol=0.0, atol=1e-14):
            raise VerificationError(f"zero_se_not_deterministic:{ep}")

    estimable = np.flatnonzero(endpoint_status == "ESTIMABLE").astype(np.int16)
    if estimable.size == 0:
        raise VerificationError("no_estimable_visual_endpoints")
    max_abs_t = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for ep in estimable:
        if se[ep] == 0.0:
            t = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
        else:
            t = np.abs((bootstrap_means[ep] - point[ep]) / se[ep])
        max_abs_t = np.maximum(max_abs_t, t)
    critical = float(np.quantile(max_abs_t, 0.99, method="linear"))
    ci_low = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)
    ci_high = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)
    for ep in estimable:
        ci_low[ep] = point[ep] - critical * se[ep]
        ci_high[ep] = point[ep] + critical * se[ep]

    return {
        "roster": roster,
        "endpoint_keys": endpoint_keys,
        "speaker_values": speaker_values,
        "availability_mask": availability_mask,
        "endpoint_status": endpoint_status,
        "point_estimates": point,
        "bootstrap_se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "critical_value": np.asarray([critical], dtype=np.float64),
        "bootstrap_indices": bootstrap_indices,
        "bootstrap_counts": bootstrap_counts,
        "draw_denominators": draw_denominators,
        "bootstrap_estimates": bootstrap_means,
        "max_abs_t": max_abs_t,
        "estimable_endpoint_indices": estimable,
    }


def compare_float(actual: np.ndarray, expected: np.ndarray, label: str, *, rtol: float = FAMILY_RTOL, atol: float = FAMILY_ATOL) -> float:
    if actual.shape != expected.shape:
        raise VerificationError(f"array_shape_mismatch:{label}:{actual.shape}:{expected.shape}")
    finite_mask = np.isfinite(actual) & np.isfinite(expected)
    if not np.array_equal(np.isnan(actual), np.isnan(expected)) or not np.array_equal(np.isposinf(actual), np.isposinf(expected)) or not np.array_equal(np.isneginf(actual), np.isneginf(expected)):
        raise VerificationError(f"array_nonfinite_pattern_mismatch:{label}")
    if finite_mask.any() and not np.allclose(actual[finite_mask], expected[finite_mask], rtol=rtol, atol=atol):
        delta = np.abs(actual[finite_mask] - expected[finite_mask])
        raise VerificationError(f"array_value_mismatch:{label}:max_abs={float(delta.max())}")
    return float(np.max(np.abs(actual[finite_mask] - expected[finite_mask]))) if finite_mask.any() else 0.0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tracking-plan", type=Path, required=True)
    result.add_argument("--track-root", type=Path, required=True)
    result.add_argument("--availability-root", type=Path, required=True)
    result.add_argument("--geometry-root", type=Path, required=True)
    result.add_argument("--primary-source", type=Path, required=True)
    result.add_argument("--expected-primary-source-sha256", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if not SHA_RE.fullmatch(args.expected_primary_source_sha256):
            raise VerificationError("expected_primary_source_sha256_not_64_hex")
        if sha256_file(args.primary_source.resolve(strict=True)) != args.expected_primary_source_sha256:
            raise VerificationError("primary_source_sha256_mismatch")
        plan = verify_plan(args.tracking_plan.resolve(strict=True))
        contexts, track_manifest, availability_manifest = verify_track_and_availability_inputs(
            plan, args.track_root.resolve(strict=True), args.availability_root.resolve(strict=True)
        )
        independent_cell_values, independent_cell_status, token_rows = recompute_context_geometry(contexts)
        independent_family = rebuild_family(contexts, independent_cell_values)

        # The primary output contract is validated after the source is sealed.
        # This guard intentionally fails closed until each exact JSON pointer and
        # array key has been pinned from the frozen primary implementation.
        geometry_root = args.geometry_root.resolve(strict=True)
        primary_manifest_path = geometry_root / "geometry_family_manifest.json"
        if not primary_manifest_path.is_file():
            raise VerificationError("primary_geometry_manifest_missing")
        raise VerificationError("primary_manifest_contract_not_yet_bound")
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
