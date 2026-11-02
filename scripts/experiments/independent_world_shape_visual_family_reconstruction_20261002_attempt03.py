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
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PLAN_RAW_SHA256 = "13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f"
EXPECTED_PLAN_PAYLOAD_SHA256 = "26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e"
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
    "phone_class_separation",
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
TONE_MARKS = frozenset(("˥", "˦", "˧", "˨", "˩"))
BILABIAL = frozenset(("p", "b", "m"))
OPEN_VOWEL = frozenset(("a", "ɑ", "ɒ", "ɐ", "æ"))


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


def load_visual_core(path: Path) -> Any:
    path = path.resolve(strict=True)
    if sha256_file(path) != EXPECTED_VISUAL_CORE_SHA256:
        raise VerificationError("visual_core_source_sha_mismatch")
    spec = importlib.util.spec_from_file_location("frozen_visual_core_for_independent_verifier", path)
    if spec is None or spec.loader is None:
        raise VerificationError("visual_core_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def phone_class(raw_label: Any, core: Any) -> tuple[str, str]:
    raw = str(raw_label or "").strip().lower()
    stripped = "".join(char for char in raw if char not in TONE_MARKS)
    normalized = core.normalize_phone(stripped)
    if normalized in BILABIAL:
        return normalized, "bilabial"
    if normalized in OPEN_VOWEL:
        return normalized, "open_vowel"
    return normalized, ""


def match_clock(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    tolerance = 0.5 * float(np.median(np.diff(reference))) + 1e-12 if reference.size > 1 else 0.5 / FPS
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
    plan_path = plan_path.resolve(strict=True)
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
    plan["_path"] = str(plan_path)
    return plan


def verify_plan_sources(plan: dict[str, Any], paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    seals = plan.get("input_seals")
    if not isinstance(seals, dict):
        raise VerificationError("tracking_plan_input_seals_missing")
    if not set(paths).issubset(seals):
        raise VerificationError("tracking_plan_source_path_roles_missing")
    for role, seal in seals.items():
        if not isinstance(seal, dict) or not SHA_RE.fullmatch(str(seal.get("raw_sha256", ""))):
            raise VerificationError(f"tracking_plan_input_seal_invalid:{role}")
    verified: dict[str, dict[str, str]] = {}
    for role, source_path in paths.items():
        source_path = source_path.resolve(strict=True)
        actual = sha256_file(source_path)
        expected = seals.get(role, {}).get("raw_sha256")
        if actual != expected:
            raise VerificationError(f"tracking_plan_source_sha_mismatch:{role}")
        verified[role] = {"path": str(source_path), "raw_sha256": actual}
    return verified


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
        # Keep only the three arrays used for independent geometry rebuilds.
        # Hash and descriptor checks still cover landmarks/canonical_mouth.
        if name in {"timestamps_s", "valid", "mouth_open"}:
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
    plan: dict[str, Any], track_root: Path, availability_root: Path, core: Any
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
    if (track_manifest.get("schema") != "world_shape_visual_track_manifest_v1"
            or track_manifest.get("status") != "COMPLETE"
            or track_manifest.get("completed_videos") != EXPECTED_TRACKS
            or track_manifest.get("failed_videos") != 0
            or track_manifest.get("effects_read") is not False
            or track_manifest.get("model_forward_performed") is not False):
        raise VerificationError("track_manifest_not_complete")
    if (track_manifest.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or track_manifest.get("tracking_plan_sha256") != plan.get("plan_sha256")):
        raise VerificationError("track_manifest_plan_binding_mismatch")
    if track_seal.get("status") != "SEALED" or track_seal.get("manifest_sha256") != track_raw or track_seal.get("manifest_payload_sha256") != track_payload or track_seal.get("all_files_sha256") != track_inventory_raw:
        raise VerificationError("track_seal_binding_mismatch")
    verify_tree_inventory(
        track_root, track_inventory_path, track_inventory,
        {"visual_track_seal.json", "all_files_sha256.json", "visual_track_manifest.json"},
    )

    av_manifest_path = availability_root / "availability_freeze.json"
    av_seal_path = availability_root / "availability_seal.json"
    av_inventory_path = availability_root / "all_files_sha256.json"
    av_manifest = read_json(av_manifest_path)
    av_seal = read_json(av_seal_path)
    av_inventory = read_inventory(av_inventory_path)
    av_payload = verify_fingerprint(av_manifest, "manifest_sha256", "availability_freeze")
    av_raw = sha256_file(av_manifest_path)
    av_inventory_raw = sha256_file(av_inventory_path)
    if (av_manifest.get("schema") != "world_shape_visual_availability_freeze_v1"
            or av_manifest.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or av_manifest.get("visual_design_sha256") != EXPECTED_VISUAL_DESIGN_SHA256):
        raise VerificationError("availability_freeze_status_invalid")
    av_coverage = av_manifest.get("coverage", {})
    if av_coverage.get("geometry_values_read") is not False or av_coverage.get("evaluation_effects_read") is not False:
        raise VerificationError("availability_freeze_not_geometry_blind")
    if av_manifest.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256 or av_manifest.get("tracking_plan_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise VerificationError("availability_plan_binding_mismatch")
    if av_manifest.get("track_manifest_raw_sha256") != track_raw or av_manifest.get("track_manifest_payload_sha256") != track_payload:
        raise VerificationError("availability_track_manifest_binding_mismatch")
    if av_manifest.get("track_seal_raw_sha256") != sha256_file(track_seal_path) or av_manifest.get("track_files_inventory_raw_sha256") != track_inventory_raw:
        raise VerificationError("availability_track_inventory_binding_mismatch")
    if (av_seal.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or av_seal.get("manifest_raw_sha256") != av_raw
            or av_seal.get("manifest_payload_sha256") != av_payload
            or av_seal.get("all_files_sha256") != av_inventory_raw
            or av_seal.get("geometry_values_read") is not False):
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
        recomputed_common = np.ones(q00_times.shape, dtype=np.bool_)
        for cell in CELLS:
            mapping = maps[cell]
            cell_valid = valid_on_q00[cell]
            if mapping.dtype != np.int32 or mapping.shape != q00_times.shape or cell_valid.dtype != np.bool_ or cell_valid.shape != q00_times.shape:
                raise VerificationError(f"availability_cell_map_shape_dtype_invalid:{key}:{cell}")
            if array_sha256(mapping) != av_row.get("cell_index_map_sha256", {}).get(cell):
                raise VerificationError(f"availability_cell_map_hash_mismatch:{key}:{cell}")
            if cell == "q00" and not np.array_equal(mapping, np.arange(len(q00_times), dtype=np.int32)):
                raise VerificationError(f"availability_q00_map_not_identity:{key}")
        if not np.array_equal(maps["q00"], np.arange(len(q00_times), dtype=np.int32)):
            raise VerificationError(f"availability_q00_map_not_identity:{key}")

        arrays_by_cell: dict[str, dict[str, np.ndarray]] = {}
        for cell in CELLS:
            record = rec_by_key.get((plan_row["sample_id"], plan_row["direction"], cell))
            if record is None or record.get("status") != "complete":
                raise VerificationError(f"track_record_missing_or_failed:{key}:{cell}")
            plan_cell = plan_row["cells"][cell]
            if (record.get("speaker_id") != plan_row["speaker_id"]
                    or record.get("split") != plan_row["split"]
                    or record.get("context_order") != plan_row["context_order"]
                    or record.get("ordered_eval_index") != plan_row["ordered_eval_index"]
                    or record.get("recipient_grid_sha256") != plan_row["recipient_grid_sha256"]):
                raise VerificationError(f"track_identity_mismatch:{key}:{cell}")
            if (record.get("video_bytes") != plan_cell.get("video_bytes")
                    or not math.isclose(float(record.get("fps_hz", 0.0)), FPS, rel_tol=0, abs_tol=1e-6)):
                raise VerificationError(f"track_video_size_or_fps_mismatch:{key}:{cell}")
            for field in ("video_relative_path", "video_sha256", "video_bytes", "official_frame_count"):
                if record.get(field) != plan_cell.get(field):
                    raise VerificationError(f"track_plan_video_binding_mismatch:{key}:{cell}:{field}")
            feature_rel = record["feature_relative_path"].replace("\\", "/")
            availability_rel = record["availability_relative_path"].replace("\\", "/")
            frozen_ref = av_row.get("track_refs", {}).get(cell, {})
            if (frozen_ref.get("feature_relative_path") != record.get("feature_relative_path")
                    or frozen_ref.get("feature_sha256") != record.get("feature_sha256")
                    or frozen_ref.get("availability_relative_path") != record.get("availability_relative_path")
                    or frozen_ref.get("availability_sha256") != record.get("availability_sha256")
                    or frozen_ref.get("array_inventory") != record.get("array_inventory")):
                raise VerificationError(f"availability_track_ref_mismatch:{key}:{cell}")
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
            frame_count = int(record.get("official_frame_count", -1))
            expected_meta = {
                "sample_id": plan_row["sample_id"],
                "speaker_id": plan_row["speaker_id"],
                "split": plan_row["split"],
                "direction": plan_row["direction"],
                "cell": cell,
                "context_order": plan_row["context_order"],
                "ordered_eval_index": plan_row["ordered_eval_index"],
                "protocol": "world_shape_visual_tracks_20261002_v1",
                "video_relative_path": plan_cell["video_relative_path"],
                "video_sha256": plan_cell["video_sha256"],
                "recipient_grid_sha256": plan_row["recipient_grid_sha256"],
                "landmarker_sha256": EXPECTED_LANDMARKER_SHA256,
                "video_features_sha256": EXPECTED_FEATURE_HELPER_SHA256,
                "visual_core_sha256": EXPECTED_VISUAL_CORE_SHA256,
                "tone_adapter_sha256": EXPECTED_TONE_ADAPTER_SHA256,
                "full_recipient_clock": True,
                "fps_hz_required": FPS,
                "decoded_frame_count": frame_count,
                "declared_frame_count": frame_count,
            }
            for field, expected_value in expected_meta.items():
                if metadata.get(field) != expected_value:
                    raise VerificationError(f"feature_metadata_binding_mismatch:{key}:{cell}:{field}")
            if not math.isclose(float(metadata.get("fps", 0.0)), FPS, rel_tol=0, abs_tol=1e-6):
                raise VerificationError(f"feature_metadata_fps_mismatch:{key}:{cell}")
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
            if cell == "q00" and not np.array_equal(q00_times, times):
                raise VerificationError(f"availability_reference_clock_not_exact_q00_feature_clock:{key}")
            expected_mapping = match_clock(q00_times, times)
            if not np.array_equal(mapping, expected_mapping):
                raise VerificationError(f"availability_clock_match_reconstruction_mismatch:{key}:{cell}")
            found = mapping >= 0
            aligned_valid = np.zeros(mapping.shape, dtype=np.bool_)
            aligned_valid[found] = valid[mapping[found]]
            if not np.array_equal(aligned_valid, valid_on_q00[cell]):
                raise VerificationError(f"feature_validity_map_mismatch:{key}:{cell}")
            if np.any(common_valid & (mapping < 0)) or np.any(common_valid & ~aligned_valid):
                raise VerificationError(f"common_mask_not_supported_by_track:{key}:{cell}")
            recomputed_common &= aligned_valid
            with np.load(availability_path, allow_pickle=False) as availability_archive:
                if set(availability_archive.files) != {"timestamps_s", "valid"}:
                    raise VerificationError(f"track_availability_npz_keys_invalid:{key}:{cell}")
                av_times = np.asarray(availability_archive["timestamps_s"])
                av_valid = np.asarray(availability_archive["valid"])
            if av_times.dtype != np.float64 or av_valid.dtype != np.bool_ or not np.array_equal(av_times, times) or not np.array_equal(av_valid, valid):
                raise VerificationError(f"track_availability_npz_not_feature_identity:{key}:{cell}")
            arrays_by_cell[cell] = feature_arrays
        if not np.array_equal(recomputed_common, common_valid):
            raise VerificationError(f"availability_common_mask_not_reconstructed:{key}")
        expected_tokens = []
        for phone in plan_row.get("recipient_phones", []):
            normalized, cls = phone_class(phone.get("label"), core)
            if not cls:
                continue
            shifted_start = float(phone["start_s"]) + LATENCY_S
            shifted_end = float(phone["end_s"]) + LATENCY_S
            indices = np.flatnonzero(common_valid & (q00_times >= shifted_start) & (q00_times < shifted_end)).astype(np.int32)
            expected_tokens.append({
                "phone_index": int(phone["phone_index"]), "raw_label": str(phone["label"]),
                "normalized_label": normalized, "visual_class": cls,
                "start_s": phone["start_s"], "end_s": phone["end_s"],
                "shifted_start_s": shifted_start, "shifted_end_s": shifted_end,
                "common_frames": int(indices.size), "supported": bool(indices.size >= int(av_row["minimum_common_frames_per_token"])),
                "q00_indices": indices.tolist(),
            })
        tokens = av_row.get("tokens", [])
        if tokens != expected_tokens:
            raise VerificationError(f"availability_token_records_not_reconstructed:{key}")
        if len(expected_tokens) == 0:
            raise VerificationError(f"availability_context_has_no_classified_phone_tokens:{key}")
        supported_counts = {cls: sum(bool(t["supported"]) and t["visual_class"] == cls for t in expected_tokens) for cls in ("bilabial", "open_vowel")}
        expected_counts = {cls: sum(t["visual_class"] == cls for t in expected_tokens) for cls in ("bilabial", "open_vowel")}
        if av_row.get("supported_token_counts") != supported_counts or av_row.get("expected_token_counts") != expected_counts:
            raise VerificationError(f"availability_token_count_mismatch:{key}")
        minimums = av_row.get("minimum_token_counts")
        if not isinstance(minimums, dict) or set(minimums) != {"bilabial", "open_vowel"}:
            raise VerificationError(f"availability_minimum_counts_invalid:{key}")
        expected_eligible = bool(plan_row.get("original_mfa_pair_qualified")) and all(
            supported_counts[cls] >= int(minimums[cls]) for cls in minimums
        )
        if av_row.get("availability_endpoint_eligible") is not expected_eligible:
            raise VerificationError(f"availability_context_eligibility_mismatch:{key}")
        contexts.append(ContextTracks(plan_row, av_row, q00_times, common_valid, common_adjacent, maps, arrays_by_cell, tokens))

    expected_track_paths = observed_track_refs | {
        "visual_track_protocol.json", "progress.jsonl"
    }
    if set(track_inventory) != expected_track_paths:
        raise VerificationError("track_inventory_has_unbound_or_missing_feature_refs")
    if (track_inventory.get("visual_track_protocol.json") != track_manifest.get("tracking_protocol_sha256")
            or track_inventory.get("progress.jsonl") != track_manifest.get("progress_sha256")
            or track_seal.get("files") != len(track_inventory)):
        raise VerificationError("track_protocol_progress_or_inventory_count_binding_mismatch")
    expected_av_contexts = {r["availability_context_path"].replace("\\", "/") for r in av_rows}
    if set(av_inventory) != expected_av_contexts | {"availability_freeze.json"}:
        raise VerificationError("availability_inventory_context_refs_mismatch")
    if (av_inventory.get("availability_freeze.json") != av_raw
            or av_seal.get("context_files") != EXPECTED_CONTEXTS
            or len(av_inventory) != EXPECTED_CONTEXTS + 1):
        raise VerificationError("availability_inventory_count_binding_mismatch")
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
    minimums = context.availability_row.get("minimum_token_counts")
    if not isinstance(minimums, dict) or set(minimums) != {"bilabial", "open_vowel"}:
        raise VerificationError("phone_minimum_token_counts_invalid")
    if any(len(by_class[cls]) < int(minimums[cls]) for cls in by_class):
        return None, token_means
    return float(np.mean(by_class["open_vowel"], dtype=np.float64) - np.mean(by_class["bilabial"], dtype=np.float64)), token_means


def recompute_context_geometry(contexts: list[ContextTracks]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values = np.full((EXPECTED_CONTEXTS, len(CELLS), len(METRICS)), np.nan, dtype=np.float64)
    statuses = np.full(values.shape, "INCOMPLETE", dtype="U48")
    token_rows: list[dict[str, Any]] = []
    for ci, context in enumerate(sorted(contexts, key=lambda item: int(item.plan_row["context_order"]))):
        if int(context.plan_row["context_order"]) != ci:
            raise VerificationError("context_order_not_contiguous")
        has_frames = bool(context.common_valid.any())
        has_adjacent = bool(context.common_adjacent.any())
        plan_phones = context.plan_row.get("recipient_phones", [])
        phone_by_index = {int(phone["phone_index"]): (token_index, phone) for token_index, phone in enumerate(plan_phones)}
        token_value_by_phone: dict[int, list[float]] = {
            int(token["phone_index"]): [float("nan")] * len(CELLS) for token in context.tokens
        }
        for cell_index, cell in enumerate(CELLS):
            mapping = context.maps[cell]
            mouth_open = context.arrays[cell]["mouth_open"]
            aligned = np.full(mapping.shape, np.nan, dtype=np.float64)
            found = mapping >= 0
            aligned[found] = mouth_open[mapping[found]]
            if has_frames and np.isfinite(aligned[context.common_valid]).all():
                values[ci, cell_index, 0] = float(np.mean(aligned[context.common_valid], dtype=np.float64))
                statuses[ci, cell_index, 0] = "COMPLETE"
            else:
                statuses[ci, cell_index, 0] = "INCOMPLETE_NO_COMMON_FRAMES"
            if has_adjacent:
                pair_values = np.abs(np.diff(aligned) / np.diff(context.q00_times))
                if np.isfinite(pair_values[context.common_adjacent]).all():
                    values[ci, cell_index, 1] = float(np.mean(pair_values[context.common_adjacent], dtype=np.float64))
                    statuses[ci, cell_index, 1] = "COMPLETE"
                else:
                    raise VerificationError("nonfinite_velocity_on_common_adjacent_support")
            else:
                statuses[ci, cell_index, 1] = "INCOMPLETE_NO_COMMON_ADJACENT_PAIRS"
            separation, token_means = _token_means_for_context(context, aligned)
            if separation is not None and math.isfinite(separation):
                values[ci, cell_index, 2] = separation
                statuses[ci, cell_index, 2] = "COMPLETE"
            else:
                statuses[ci, cell_index, 2] = "INCOMPLETE_PHONE_CLASS_SUPPORT"
            for token in context.tokens:
                phone_index = int(token["phone_index"])
                measured = token_means.get(str(phone_index))
                if measured is not None:
                    token_value_by_phone[phone_index][cell_index] = float(measured)
        for token in context.tokens:
            phone_index = int(token["phone_index"])
            token_index, phone = phone_by_index[phone_index]
            token_rows.append({
                "context_order": ci,
                "context_sample_id": context.plan_row["sample_id"],
                "speaker_id": str(context.plan_row["speaker_id"]),
                "direction": context.plan_row["direction"],
                "token_index": token_index,
                "phone_index": phone_index,
                "phone_label": str(phone["label"]),
                "normalized_phone": token["normalized_label"],
                "phone_class": token["visual_class"],
                "start_s": float(phone["start_s"]),
                "end_s": float(phone["end_s"]),
                "shifted_start_s": float(token["shifted_start_s"]),
                "shifted_end_s": float(token["shifted_end_s"]),
                "common_frame_count": int(token["common_frames"]),
                "token_supported": bool(token["supported"]),
                "cell_mean_aperture": token_value_by_phone[phone_index],
            })
    return values, statuses, token_rows


def expected_endpoint_order() -> list[tuple[str, str, str]]:
    return [(direction, metric, contrast) for direction in DIRECTIONS for metric in METRICS for contrast in CONTRASTS]


def rebuild_family(
    contexts: list[ContextTracks], cell_values: np.ndarray, availability_manifest: dict[str, Any]
) -> dict[str, np.ndarray | list[str] | list[tuple[str, str, str]]]:
    if cell_values.shape != (EXPECTED_CONTEXTS, len(CELLS), len(METRICS)):
        raise VerificationError("cell_metric_values_shape_invalid")
    roster = sorted({str(context.plan_row["speaker_id"]) for context in contexts})
    if len(roster) != EXPECTED_SPEAKERS:
        raise VerificationError("speaker_roster_count_invalid")
    speaker_index = {speaker: index for index, speaker in enumerate(roster)}
    context_index = {(context.plan_row["sample_id"], context.plan_row["direction"]): index for index, context in enumerate(sorted(contexts, key=lambda item: int(item.plan_row["context_order"])))}
    av_speaker_rows = {
        (str(row["speaker_id"]), str(row["direction"])): row
        for row in availability_manifest.get("speaker_rows", [])
    }
    expected_groups = {(speaker, direction) for speaker in roster for direction in DIRECTIONS}
    if len(av_speaker_rows) != 80 or set(av_speaker_rows) != expected_groups:
        raise VerificationError("availability_speaker_row_coverage_invalid")
    phone_eligible: dict[tuple[str, str], bool] = {}
    by_speaker_direction: dict[tuple[str, str], list[int]] = {}
    context_rows_by_group: dict[tuple[str, str], list[ContextTracks]] = {}
    for context in contexts:
        identity = (str(context.plan_row["speaker_id"]), str(context.plan_row["direction"]))
        by_speaker_direction.setdefault(identity, []).append(context_index[(context.plan_row["sample_id"], context.plan_row["direction"])])
        context_rows_by_group.setdefault(identity, []).append(context)
    if set(by_speaker_direction) != expected_groups:
        raise VerificationError("speaker_direction_group_coverage_invalid")
    for group, indices in by_speaker_direction.items():
        if len(indices) != 2:
            raise VerificationError(f"speaker_direction_not_two_clips:{group}")
        group_contexts = context_rows_by_group[group]
        clip_ids = sorted(str(ctx.plan_row["sample_id"]) for ctx in group_contexts)
        source_row = av_speaker_rows[group]
        if sorted(str(value) for value in source_row.get("clip_ids", [])) != clip_ids:
            raise VerificationError(f"availability_speaker_clip_ids_mismatch:{group}")
        derived_eligible = all(bool(ctx.availability_row.get("availability_endpoint_eligible")) for ctx in group_contexts)
        if source_row.get("complete_two_clip_phone_eligible") is not derived_eligible:
            raise VerificationError(f"availability_speaker_phone_mask_mismatch:{group}")
        phone_eligible[group] = derived_eligible

    # Rebuild clip→speaker means first, in the same tensor orientation as the
    # published estimand, then apply the nine frozen contrasts by direct
    # arithmetic. This avoids a different summation order from a coefficient
    # loop for the factorial interaction.
    speaker_cells = np.full((len(DIRECTIONS), EXPECTED_SPEAKERS, len(CELLS), len(METRICS)), np.nan, dtype=np.float64)
    phone_masks: dict[str, np.ndarray] = {}
    for direction_index, direction in enumerate(DIRECTIONS):
        phone_masks[direction] = np.zeros(EXPECTED_SPEAKERS, dtype=np.bool_)
        for sidx, speaker in enumerate(roster):
            clip_indices = sorted(by_speaker_direction[(speaker, direction)])
            if len(clip_indices) != 2:
                raise VerificationError(f"not_two_clips_per_speaker_direction:{speaker}:{direction}")
            for cidx in range(len(CELLS)):
                for midx in range(len(METRICS)):
                    pair = cell_values[clip_indices, cidx, midx]
                    if np.isfinite(pair).all():
                        speaker_cells[direction_index, sidx, cidx, midx] = float(np.mean(pair, dtype=np.float64))
            phone_masks[direction][sidx] = phone_eligible[(speaker, direction)]

    endpoint_keys = [(direction, metric, contrast) for direction in DIRECTIONS for metric in METRICS for contrast in CONTRASTS]
    speaker_values_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    for direction_index, direction in enumerate(DIRECTIONS):
        cells_by_metric = np.moveaxis(speaker_cells[direction_index], 1, -1)  # [speaker, metric, cell]
        q00, q10, q01, q11, self_grid = [cells_by_metric[..., i] for i in range(len(CELLS))]
        contrast_tensor = np.stack((
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
        for metric_index, metric in enumerate(METRICS):
            for contrast_index, _contrast in enumerate(CONTRASTS):
                speaker_values_list.append(contrast_tensor[:, metric_index, contrast_index])
                masks_list.append(phone_masks[direction].copy() if metric == "phone_class_separation" else np.ones(EXPECTED_SPEAKERS, dtype=np.bool_))
    speaker_values = np.asarray(speaker_values_list, dtype=np.float64)
    availability_mask = np.asarray(masks_list, dtype=np.bool_)
    if speaker_values.shape != (EXPECTED_ENDPOINTS, EXPECTED_SPEAKERS) or availability_mask.shape != speaker_values.shape:
        raise VerificationError("rebuilt_endpoint_array_shape_mismatch")

    endpoint_status = np.full(EXPECTED_ENDPOINTS, "INCOMPLETE_NO_FROZEN_ELIGIBLE_SPEAKERS", dtype="<U64")
    point = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)
    for ep in range(EXPECTED_ENDPOINTS):
        mask = availability_mask[ep]
        if not mask.any():
            continue
        if not np.isfinite(speaker_values[ep, mask]).all():
            endpoint_status[ep] = "INCOMPLETE_FROZEN_SPEAKER_GEOMETRY_MISSING"
            continue
        endpoint_status[ep] = "ESTIMABLE"
        point[ep] = float(np.mean(speaker_values[ep, mask], dtype=np.float64))

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_indices = rng.integers(0, EXPECTED_SPEAKERS, size=(BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.int16)
    bootstrap_counts = np.zeros((BOOTSTRAP_DRAWS, EXPECTED_SPEAKERS), dtype=np.uint8)
    for draw in range(BOOTSTRAP_DRAWS):
        bootstrap_counts[draw] = np.bincount(bootstrap_indices[draw], minlength=EXPECTED_SPEAKERS).astype(np.uint8)

    draw_denominators = np.zeros((EXPECTED_ENDPOINTS, BOOTSTRAP_DRAWS), dtype=np.int16)
    bootstrap_means = np.full((EXPECTED_ENDPOINTS, BOOTSTRAP_DRAWS), np.nan, dtype=np.float64)
    se = np.full(EXPECTED_ENDPOINTS, np.nan, dtype=np.float64)
    deterministic_zero_se = np.zeros(EXPECTED_ENDPOINTS, dtype=np.bool_)
    for ep in range(EXPECTED_ENDPOINTS):
        mask = availability_mask[ep]
        if not mask.any():
            continue
        counts = bootstrap_counts[:, mask].astype(np.float64)
        denominator = counts.sum(axis=1).astype(np.int16)
        if np.any(denominator <= 0):
            raise VerificationError(f"zero_bootstrap_denominator:{ep}")
        draw_denominators[ep] = denominator
        if not endpoint_status[ep].startswith("ESTIMABLE"):
            continue
        vals = speaker_values[ep, mask]
        if np.all(vals == vals[0]):
            draws = np.full(BOOTSTRAP_DRAWS, point[ep], dtype=np.float64)
            se[ep] = 0.0
            deterministic_zero_se[ep] = True
        else:
            draws = np.sum(counts * vals[None, :], axis=1, dtype=np.float64) / denominator.astype(np.float64)
            se[ep] = float(np.std(draws, ddof=1, dtype=np.float64))
            if se[ep] == 0.0:
                if not np.array_equal(draws, np.full(BOOTSTRAP_DRAWS, point[ep], dtype=np.float64)):
                    raise VerificationError(f"zero_se_not_deterministic:{ep}")
                deterministic_zero_se[ep] = True
        if not np.isfinite(draws).all():
            raise VerificationError(f"nonfinite_bootstrap_estimate:{ep}")
        bootstrap_means[ep] = draws
        if deterministic_zero_se[ep]:
            endpoint_status[ep] = "ESTIMABLE_DETERMINISTIC_ZERO_SE"

    estimable = np.flatnonzero(np.char.startswith(endpoint_status, "ESTIMABLE")).astype(np.int16)
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
        "deterministic_zero_se": deterministic_zero_se,
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


def assert_array_contract(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise VerificationError(f"array_dtype_or_shape_mismatch:{label}:{actual.dtype}/{actual.shape}:{expected.dtype}/{expected.shape}")
    if actual.dtype.kind == "f":
        if not np.isfinite(actual[np.isfinite(expected)]).all():
            raise VerificationError(f"unexpected_nonfinite_actual:{label}")
        return compare_float(actual, expected, label)
    if not np.array_equal(actual, expected):
        raise VerificationError(f"array_exact_mismatch:{label}")
    return 0.0


def load_npz_contract(path: Path, expected_keys: set[str], label: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise VerificationError(f"npz_keyset_mismatch:{label}")
        return {key: np.asarray(archive[key]) for key in expected_keys}


def verify_geometry_release(
    path: Path, source_sha: str, plan_raw: str, plan_payload: str,
    track_root: Path, availability_root: Path,
) -> tuple[dict[str, Any], str, str]:
    value = read_json(path)
    payload = verify_fingerprint(value, "manifest_sha256", "geometry_release")
    track_manifest_path = track_root / "visual_track_manifest.json"
    track_seal_path = track_root / "visual_track_seal.json"
    track_inventory_path = track_root / "all_files_sha256.json"
    availability_manifest_path = availability_root / "availability_freeze.json"
    availability_seal_path = availability_root / "availability_seal.json"
    availability_inventory_path = availability_root / "all_files_sha256.json"
    expected = {
        "status": "ROOT_RELEASED", "scope": "GEOMETRY_ANALYSIS", "geometry_read_authorized": True,
        "geometry_consumer_source_sha256": source_sha,
        "tracking_plan_raw_sha256": plan_raw, "tracking_plan_payload_sha256": plan_payload,
        "track_manifest_raw_sha256": sha256_file(track_manifest_path),
        "track_manifest_payload_sha256": read_json(track_manifest_path).get("manifest_sha256"),
        "track_seal_raw_sha256": sha256_file(track_seal_path),
        "track_inventory_raw_sha256": sha256_file(track_inventory_path),
        "availability_manifest_raw_sha256": sha256_file(availability_manifest_path),
        "availability_manifest_payload_sha256": read_json(availability_manifest_path).get("manifest_sha256"),
        "availability_seal_raw_sha256": sha256_file(availability_seal_path),
        "availability_inventory_raw_sha256": sha256_file(availability_inventory_path),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise VerificationError(f"geometry_release_binding_mismatch:{key}")
    value["_path"] = str(path.resolve(strict=True))
    return value, payload, sha256_file(path)


def expected_context_rows(
    contexts: list[ContextTracks], track_manifest: dict[str, Any],
    values: np.ndarray, statuses: np.ndarray,
) -> list[dict[str, Any]]:
    track_by_key = {
        (str(row["sample_id"]), str(row["direction"]), str(row["cell"])): row
        for row in track_manifest.get("records", [])
    }
    result: list[dict[str, Any]] = []
    ordered = sorted(contexts, key=lambda item: int(item.plan_row["context_order"]))
    for index, context in enumerate(ordered):
        plan_row, av_row = context.plan_row, context.availability_row
        if int(plan_row["context_order"]) != index:
            raise VerificationError("context_order_not_contiguous_for_output_manifest")
        sample, direction = str(plan_row["sample_id"]), str(plan_row["direction"])
        cell_metrics = {}
        cell_status = {}
        track_refs = {}
        class_counts = av_row.get("supported_token_counts")
        if not isinstance(class_counts, dict):
            raise VerificationError("availability_supported_counts_missing")
        for cell_index, cell in enumerate(CELLS):
            cell_metrics[cell] = {
                metric: (float(values[index, cell_index, metric_index])
                         if np.isfinite(values[index, cell_index, metric_index]) else None)
                for metric_index, metric in enumerate(METRICS)
            }
            cell_status[cell] = {
                metric: str(statuses[index, cell_index, metric_index])
                for metric_index, metric in enumerate(METRICS)
            }
            record = track_by_key.get((sample, direction, cell))
            if record is None:
                raise VerificationError(f"track_record_missing_for_context_manifest:{sample}:{direction}:{cell}")
            track_refs[cell] = {
                "feature_relative_path": record["feature_relative_path"],
                "feature_sha256": record["feature_sha256"],
                "availability_relative_path": record["availability_relative_path"],
                "availability_sha256": record["availability_sha256"],
                "array_inventory": record["array_inventory"],
            }
        result.append({
            "context_order": index,
            "sample_id": sample,
            "speaker_id": str(plan_row["speaker_id"]),
            "direction": direction,
            "split": str(plan_row["split"]),
            "original_mfa_pair_qualified": bool(plan_row["original_mfa_pair_qualified"]),
            "grid_overlay_used": bool(plan_row.get("overlay_used")),
            "availability_endpoint_eligible": bool(av_row["availability_endpoint_eligible"]),
            "availability_context_path": av_row["availability_context_path"],
            "availability_context_sha256": av_row["availability_context_sha256"],
            "common_frame_count": int(context.common_valid.sum()),
            "common_adjacent_pair_count": int(context.common_adjacent.sum()),
            "q00_frame_count": int(len(context.q00_times)),
            "track_refs": track_refs,
            "cell_metrics": cell_metrics,
            "cell_metric_status": cell_status,
            "class_token_counts": {
                cell: {
                    "bilabial": int(class_counts["bilabial"]),
                    "open_vowel": int(class_counts["open_vowel"]),
                }
                for cell in CELLS
            },
        })
    return result


def expected_endpoint_rows(family: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    critical = float(family["critical_value"][0])
    for endpoint_index, (direction, metric, contrast) in enumerate(family["endpoint_keys"]):
        status = str(family["endpoint_status"][endpoint_index])
        point = float(family["point_estimates"][endpoint_index]) if np.isfinite(family["point_estimates"][endpoint_index]) else None
        se = float(family["bootstrap_se"][endpoint_index]) if np.isfinite(family["bootstrap_se"][endpoint_index]) else None
        low = float(family["ci_low"][endpoint_index]) if np.isfinite(family["ci_low"][endpoint_index]) else None
        high = float(family["ci_high"][endpoint_index]) if np.isfinite(family["ci_high"][endpoint_index]) else None
        rows.append({
            "endpoint_index": endpoint_index,
            "endpoint_key": f"{direction}::{metric}::{contrast}",
            "direction": direction,
            "metric": metric,
            "contrast": contrast,
            "estimand_mask": "frozen_phone_complete_pair" if metric == "phone_class_separation" else "all_40_speakers",
            "availability_speaker_count": int(family["availability_mask"][endpoint_index].sum()),
            "status": status,
            "point_estimate": point,
            "bootstrap_se_sample_sd_ddof1": se,
            "simultaneous_99_ci": [low, high] if low is not None and high is not None else None,
            "critical_value": critical,
            "zero_se_deterministic": bool(family["deterministic_zero_se"][endpoint_index]),
        })
    return rows


def verify_primary_output(
    geometry_root: Path, expected_primary_source_sha: str, plan: dict[str, Any], plan_raw: str,
    track_root: Path, track_manifest: dict[str, Any], availability_root: Path,
    availability_manifest: dict[str, Any], release: dict[str, Any], release_payload: str,
    release_raw: str, independent_values: np.ndarray, independent_status: np.ndarray,
    token_rows: list[dict[str, Any]], independent_family: dict[str, Any], contexts: list[ContextTracks],
    source_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    root = geometry_root.resolve(strict=True)
    required_files = {
        "geometry_values.npz", "phone_token_measurements.npz", "speaker_endpoints.npz",
        "bootstrap_family.npz", "geometry_family_manifest.json", "all_files_sha256.json",
        "geometry_family_seal.json",
    }
    actual_files = {p.relative_to(root).as_posix() for p in root.iterdir() if p.is_file()}
    actual_dirs = {p.name for p in root.iterdir() if p.is_dir()}
    if actual_files != required_files or actual_dirs:
        raise VerificationError(f"geometry_output_inventory_mismatch:files={len(actual_files)}:dirs={len(actual_dirs)}")
    manifest_path = root / "geometry_family_manifest.json"
    inventory_path = root / "all_files_sha256.json"
    seal_path = root / "geometry_family_seal.json"
    manifest = read_json(manifest_path)
    manifest_payload = verify_fingerprint(manifest, "manifest_sha256", "geometry_family_manifest")
    manifest_raw = sha256_file(manifest_path)
    inventory_obj = read_json(inventory_path)
    inventory = inventory_obj.get("files")
    expected_names = required_files - {"all_files_sha256.json", "geometry_family_seal.json"}
    if inventory_obj.get("schema") != "world_shape_visual_geometry_files_v1" or not isinstance(inventory, dict) or set(inventory) != expected_names:
        raise VerificationError("geometry_output_inventory_schema_mismatch")
    if any(not isinstance(value, str) or not SHA_RE.fullmatch(value) for value in inventory.values()):
        raise VerificationError("geometry_output_inventory_sha_invalid")
    for name, expected_sha in inventory.items():
        if sha256_file(root / name) != expected_sha:
            raise VerificationError(f"geometry_output_file_sha_mismatch:{name}")
    seal = read_json(seal_path)
    inventory_raw = sha256_file(inventory_path)
    if (seal.get("schema") != "world_shape_visual_geometry_seal_v1"
            or seal.get("status") != manifest.get("status")
            or seal.get("manifest_raw_sha256") != manifest_raw
            or seal.get("manifest_payload_sha256") != manifest_payload
            or seal.get("all_files_sha256") != inventory_raw
            or seal.get("files") != len(expected_names)
            or seal.get("contexts") != EXPECTED_CONTEXTS
            or seal.get("videos") != EXPECTED_TRACKS
            or seal.get("endpoints") != EXPECTED_ENDPOINTS
            or seal.get("evaluation_effects_read") is not False
            or seal.get("model_forward_performed") is not False):
        raise VerificationError("geometry_output_seal_mismatch")
    if manifest.get("schema") != "world_shape_visual_geometry_family_v1":
        raise VerificationError("geometry_manifest_schema_mismatch")
    if manifest.get("status") != ("COMPLETE_WITH_INCOMPLETE_ENDPOINTS" if np.any(np.char.startswith(independent_family["endpoint_status"], "INCOMPLETE")) else "COMPLETE"):
        raise VerificationError("geometry_manifest_status_mismatch")
    if manifest.get("analysis_scope") != "visual_geometry_only; no scores/evaluation outcomes/model forward read":
        raise VerificationError("geometry_manifest_scope_mismatch")
    if manifest.get("tracking_plan") != {"path": str(Path(plan["_path"]).resolve()), "raw_sha256": plan_raw, "payload_sha256": plan["plan_sha256"]}:
        raise VerificationError("geometry_manifest_plan_binding_mismatch")
    track_raw = sha256_file(track_root / "visual_track_manifest.json")
    track_payload = track_manifest["manifest_sha256"]
    track_seal_raw = sha256_file(track_root / "visual_track_seal.json")
    track_inventory_raw = sha256_file(track_root / "all_files_sha256.json")
    availability_raw = sha256_file(availability_root / "availability_freeze.json")
    availability_payload = availability_manifest["manifest_sha256"]
    availability_seal_raw = sha256_file(availability_root / "availability_seal.json")
    availability_inventory_raw = sha256_file(availability_root / "all_files_sha256.json")
    if manifest.get("track_inputs") != {
        "root": str(track_root.resolve()), "manifest_raw_sha256": track_raw,
        "manifest_payload_sha256": track_payload, "seal_raw_sha256": track_seal_raw,
        "files_inventory_raw_sha256": track_inventory_raw, "completed_videos": EXPECTED_TRACKS,
    }:
        raise VerificationError("geometry_manifest_track_input_binding_mismatch")
    if manifest.get("availability_inputs") != {
        "root": str(availability_root.resolve()), "manifest_raw_sha256": availability_raw,
        "manifest_payload_sha256": availability_payload, "seal_raw_sha256": availability_seal_raw,
        "files_inventory_raw_sha256": availability_inventory_raw, "geometry_was_read_before_freeze": False,
    }:
        raise VerificationError("geometry_manifest_availability_input_binding_mismatch")
    if manifest.get("geometry_release") != {"path": str(release["_path"]), "raw_sha256": release_raw, "payload_sha256": release_payload}:
        raise VerificationError("geometry_manifest_release_binding_mismatch")
    if manifest.get("evaluation_scores_read") is not False or manifest.get("evaluation_outcomes_read") is not False or manifest.get("evaluation_effects_read") is not False or manifest.get("model_forward_performed") is not False:
        raise VerificationError("geometry_manifest_effect_access_policy_mismatch")
    if manifest.get("mouth_geometry_read") is not True:
        raise VerificationError("geometry_manifest_geometry_access_flag_mismatch")

    expected_source_fields = {
        "tracking_extractor_source": source_hashes["tracking_extractor_source"],
        "visual_design": {"path": source_hashes["visual_design"]["path"], "raw_sha256": EXPECTED_VISUAL_DESIGN_SHA256},
        "visual_design_pin_correction": {"path": source_hashes["execution_pin_correction"]["path"], "raw_sha256": EXPECTED_PIN_CORRECTION_SHA256},
        "visual_core_source": source_hashes["visual_core_source"],
        "tone_adapter_source": source_hashes["tone_adapter_source"],
    }
    for field, expected in expected_source_fields.items():
        if manifest.get(field) != expected:
            raise VerificationError(f"geometry_manifest_source_binding_mismatch:{field}")

    expected_endpoint_design = {
        "directions": list(DIRECTIONS), "cells": list(CELLS), "metrics": list(METRICS), "contrasts": list(CONTRASTS),
        "endpoint_order": "direction-major, metric-major, contrast-minor",
        "endpoint_count": EXPECTED_ENDPOINTS, "contexts": EXPECTED_CONTEXTS, "clips": EXPECTED_CLIPS,
        "speakers": EXPECTED_SPEAKERS, "video_tracks": EXPECTED_TRACKS,
        "latency_s": LATENCY_S, "fps_hz": FPS,
        "mean_aperture": "mean mouth_open over all common-valid q00-reference frames",
        "mean_absolute_aperture_velocity_per_s": "mean absolute first difference over adjacent common-valid q00-reference pairs divided by exact q00 dt; holes are not bridged",
        "phone_class_separation": "within each clip, equal mean of supported token aperture means by class; open_vowel minus bilabial",
        "phone_classifier": "fixed old tone-stripped bilabial/open_vowel classes; recipient phones and original MFA qualification only",
    }
    if manifest.get("endpoint_design") != expected_endpoint_design:
        raise VerificationError("geometry_manifest_endpoint_design_mismatch")
    expected_bootstrap = {
        "bit_generator": "PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "draw_size": EXPECTED_SPEAKERS,
        "rng_call": "Generator(PCG64(seed)).integers(0, 40, size=(20000, 40), dtype=int16)",
        "count_construction": "row-wise bincount(minlength=40).astype(uint8)",
        "sample_unit": "speaker; one shared draw-count matrix across all endpoints",
        "point": "equal speaker mean; overall all 40, phone endpoints frozen complete-pair eligibility mask",
        "se": "bootstrap sample standard deviation, ddof=1",
        "simultaneous_ci": "studentized max-|t| across all estimable endpoints; np.quantile(method='linear', q=0.99); point +/- critical*SE",
        "zero_denominator_draws": 0,
    }
    if manifest.get("bootstrap") != expected_bootstrap:
        raise VerificationError("geometry_manifest_bootstrap_contract_mismatch")
    phone_metric_start = (len(METRICS) - 1) * len(CONTRASTS)
    phone_eligible_by_direction = {
        direction: int(independent_family["availability_mask"][phone_metric_start + direction_index * len(METRICS) * len(CONTRASTS)].sum())
        for direction_index, direction in enumerate(DIRECTIONS)
    }
    expected_coverage = {
        "planned_contexts": EXPECTED_CONTEXTS, "complete_contexts": EXPECTED_CONTEXTS,
        "planned_video_tracks": EXPECTED_TRACKS, "complete_video_tracks": EXPECTED_TRACKS,
        "phone_token_rows": len(token_rows), "planned_endpoints": EXPECTED_ENDPOINTS,
        "estimable_endpoints": int(np.sum(np.char.startswith(independent_family["endpoint_status"], "ESTIMABLE"))),
        "incomplete_endpoints": int(np.sum(np.char.startswith(independent_family["endpoint_status"], "INCOMPLETE"))),
        "phone_eligible_speakers_by_direction": phone_eligible_by_direction,
    }
    if manifest.get("coverage") != expected_coverage:
        raise VerificationError("geometry_manifest_coverage_mismatch")
    expected_artifact_inventory = {
        name: {"path": name, "bytes": (root / name).stat().st_size, "raw_sha256": sha256_file(root / name)}
        for name in ("geometry_values.npz", "phone_token_measurements.npz", "speaker_endpoints.npz", "bootstrap_family.npz")
    }
    if manifest.get("artifact_inventory") != expected_artifact_inventory:
        raise VerificationError("geometry_manifest_artifact_inventory_mismatch")

    expected_context_output = expected_context_rows(contexts, track_manifest, independent_values, independent_status)
    if manifest.get("context_rows") != expected_context_output:
        raise VerificationError("geometry_manifest_context_rows_mismatch")
    expected_endpoint_output = expected_endpoint_rows(independent_family)
    if manifest.get("endpoint_rows") != expected_endpoint_output:
        raise VerificationError("geometry_manifest_endpoint_rows_mismatch")

    geometry = load_npz_contract(root / "geometry_values.npz", {
        "context_order_int32", "sample_id_unicode", "speaker_id_unicode", "direction_unicode",
        "cell_names_unicode", "metric_names_unicode", "cell_metric_values_float64", "cell_metric_status_unicode",
    }, "geometry_values")
    plan_rows = sorted(plan["context_rows"], key=lambda row: int(row["context_order"]))
    context_order = np.arange(EXPECTED_CONTEXTS, dtype=np.int32)
    checks = {
        "context_order_int32": context_order,
        "sample_id_unicode": np.asarray([str(row["sample_id"]) for row in plan_rows], dtype="<U96"),
        "speaker_id_unicode": np.asarray([str(row["speaker_id"]) for row in plan_rows], dtype="<U32"),
        "direction_unicode": np.asarray([str(row["direction"]) for row in plan_rows], dtype="<U32"),
        "cell_names_unicode": np.asarray(CELLS, dtype="<U16"),
        "metric_names_unicode": np.asarray(METRICS, dtype="<U48"),
        "cell_metric_values_float64": independent_values.astype(np.float64),
        "cell_metric_status_unicode": independent_status.astype("<U48"),
    }
    residuals = {}
    for name, expected in checks.items():
        residuals[f"geometry_values.{name}"] = assert_array_contract(geometry[name], expected, f"geometry_values.{name}")

    token_rows = sorted(token_rows, key=lambda row: (int(row["context_order"]), int(row["token_index"])))
    phone_expected = {
        "context_order_int32": np.asarray([r["context_order"] for r in token_rows], dtype=np.int32),
        "sample_id_unicode": np.asarray([r["context_sample_id"] for r in token_rows], dtype="<U96"),
        "speaker_id_unicode": np.asarray([r["speaker_id"] for r in token_rows], dtype="<U32"),
        "direction_unicode": np.asarray([r["direction"] for r in token_rows], dtype="<U32"),
        "token_index_int32": np.asarray([r["token_index"] for r in token_rows], dtype=np.int32),
        "phone_index_int32": np.asarray([r["phone_index"] for r in token_rows], dtype=np.int32),
        "phone_label_unicode": np.asarray([r["phone_label"] for r in token_rows], dtype="<U64"),
        "normalized_phone_unicode": np.asarray([r["normalized_phone"] for r in token_rows], dtype="<U64"),
        "phone_class_unicode": np.asarray([r["phone_class"] for r in token_rows], dtype="<U16"),
        "start_s_float64": np.asarray([r["start_s"] for r in token_rows], dtype=np.float64),
        "end_s_float64": np.asarray([r["end_s"] for r in token_rows], dtype=np.float64),
        "shifted_start_s_float64": np.asarray([r["shifted_start_s"] for r in token_rows], dtype=np.float64),
        "shifted_end_s_float64": np.asarray([r["shifted_end_s"] for r in token_rows], dtype=np.float64),
        "common_frame_count_int32": np.asarray([r["common_frame_count"] for r in token_rows], dtype=np.int32),
        "token_supported_bool": np.asarray([r["token_supported"] for r in token_rows], dtype=np.bool_),
        "cell_mean_aperture_float64": np.asarray([r["cell_mean_aperture"] for r in token_rows], dtype=np.float64),
    }
    phone = load_npz_contract(root / "phone_token_measurements.npz", set(phone_expected), "phone_token_measurements")
    for name, expected in phone_expected.items():
        residuals[f"phone_token_measurements.{name}"] = assert_array_contract(phone[name], expected, f"phone_token_measurements.{name}")

    family = independent_family
    endpoint_keys = [f"{direction}::{metric}::{contrast}" for direction, metric, contrast in family["endpoint_keys"]]
    speaker_expected = {
        "endpoint_keys_unicode": np.asarray(endpoint_keys, dtype="<U128"),
        "contrast_names_unicode": np.asarray(CONTRASTS, dtype="<U64"),
        "directions_unicode": np.asarray(DIRECTIONS, dtype="<U32"),
        "metric_names_unicode": np.asarray(METRICS, dtype="<U48"),
        "speaker_ids_unicode": np.asarray(family["roster"], dtype="<U32"),
        "speaker_values_float64": family["speaker_values"],
        "availability_mask_bool": family["availability_mask"],
        "point_estimates_float64": family["point_estimates"],
        "bootstrap_se_float64": family["bootstrap_se"],
        "ci_low_float64": family["ci_low"],
        "ci_high_float64": family["ci_high"],
        "endpoint_status_unicode": family["endpoint_status"].astype("<U64"),
        "critical_value_float64": family["critical_value"],
        "deterministic_zero_se_bool": family["deterministic_zero_se"],
    }
    speaker = load_npz_contract(root / "speaker_endpoints.npz", set(speaker_expected), "speaker_endpoints")
    for name, expected in speaker_expected.items():
        residuals[f"speaker_endpoints.{name}"] = assert_array_contract(speaker[name], expected, f"speaker_endpoints.{name}")
    bootstrap_expected = {
        "bootstrap_indices_int16": family["bootstrap_indices"],
        "bootstrap_counts_uint8": family["bootstrap_counts"],
        "draw_denominators_int16": family["draw_denominators"],
        "bootstrap_estimates_float64": family["bootstrap_estimates"],
        "max_abs_t_float64": family["max_abs_t"],
        "estimable_endpoint_indices_int16": family["estimable_endpoint_indices"],
    }
    bootstrap = load_npz_contract(root / "bootstrap_family.npz", set(bootstrap_expected), "bootstrap_family")
    for name, expected in bootstrap_expected.items():
        residuals[f"bootstrap_family.{name}"] = assert_array_contract(bootstrap[name], expected, f"bootstrap_family.{name}")
    return {
        "status": "INDEPENDENT_RECONSTRUCTION_PASS",
        "primary_source_sha256": expected_primary_source_sha,
        "geometry_manifest_raw_sha256": manifest_raw,
        "geometry_manifest_payload_sha256": manifest_payload,
        "geometry_inventory_raw_sha256": inventory_raw,
        "geometry_seal_raw_sha256": sha256_file(seal_path),
        "track_manifest_raw_sha256": track_raw,
        "track_inventory_raw_sha256": track_inventory_raw,
        "availability_manifest_raw_sha256": availability_raw,
        "availability_inventory_raw_sha256": availability_inventory_raw,
        "planned_contexts": EXPECTED_CONTEXTS,
        "planned_tracks": EXPECTED_TRACKS,
        "planned_endpoints": EXPECTED_ENDPOINTS,
        "verified_phone_token_rows": len(token_rows),
        "estimable_endpoints": int(np.sum(np.char.startswith(family["endpoint_status"], "ESTIMABLE"))),
        "incomplete_endpoints": int(np.sum(np.char.startswith(family["endpoint_status"], "INCOMPLETE"))),
        "max_abs_array_residual": max(residuals.values(), default=0.0),
        "array_residuals": residuals,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--extractor-source", type=Path, required=True)
    result.add_argument("--design", type=Path, required=True)
    result.add_argument("--evaluation-design", type=Path, required=True)
    result.add_argument("--pin-correction", type=Path, required=True)
    result.add_argument("--video-features", type=Path, required=True)
    result.add_argument("--landmarker", type=Path, required=True)
    result.add_argument("--visual-core", type=Path, required=True)
    result.add_argument("--tone-adapter", type=Path, required=True)
    result.add_argument("--calibration", type=Path, required=True)
    result.add_argument("--calibration-freeze", type=Path, required=True)
    result.add_argument("--track-dir", type=Path, required=True)
    result.add_argument("--availability-dir", type=Path, required=True)
    result.add_argument("--geometry-release", type=Path, required=True)
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
        primary_source = args.primary_source.resolve(strict=True)
        if sha256_file(primary_source) != args.expected_primary_source_sha256:
            raise VerificationError("primary_source_sha256_mismatch")
        plan = verify_plan(args.plan.resolve(strict=True))
        source_paths = {
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
        source_hashes = verify_plan_sources(plan, source_paths)
        for role, expected in {
            "tracking_extractor_source": EXPECTED_EXTRACTOR_SHA256,
            "visual_design": EXPECTED_VISUAL_DESIGN_SHA256,
            "execution_pin_correction": EXPECTED_PIN_CORRECTION_SHA256,
            "feature_helper": EXPECTED_FEATURE_HELPER_SHA256,
            "landmarker_asset": EXPECTED_LANDMARKER_SHA256,
            "visual_core_source": EXPECTED_VISUAL_CORE_SHA256,
            "tone_adapter_source": EXPECTED_TONE_ADAPTER_SHA256,
        }.items():
            if source_hashes[role]["raw_sha256"] != expected:
                raise VerificationError(f"frozen_source_constant_mismatch:{role}")
        release, release_payload, release_raw = verify_geometry_release(
            args.geometry_release.resolve(strict=True), args.expected_primary_source_sha256,
            sha256_file(args.plan.resolve(strict=True)), plan["plan_sha256"],
            args.track_dir.resolve(strict=True), args.availability_dir.resolve(strict=True),
        )
        core = load_visual_core(args.visual_core)
        contexts, track_manifest, availability_manifest = verify_track_and_availability_inputs(
            plan, args.track_dir.resolve(strict=True), args.availability_dir.resolve(strict=True), core
        )
        independent_cell_values, independent_cell_status, token_rows = recompute_context_geometry(contexts)
        independent_family = rebuild_family(contexts, independent_cell_values, availability_manifest)
        result = verify_primary_output(
            args.geometry_root.resolve(strict=True), args.expected_primary_source_sha256,
            plan, sha256_file(args.plan.resolve(strict=True)), args.track_dir.resolve(strict=True),
            track_manifest, args.availability_dir.resolve(strict=True), availability_manifest,
            release, release_payload, release_raw, independent_cell_values, independent_cell_status,
            token_rows, independent_family, contexts, source_hashes,
        )
        result.update({
            "independent_source_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "primary_source_sha256": args.expected_primary_source_sha256,
            "tracking_plan_raw_sha256": sha256_file(args.plan.resolve(strict=True)),
            "tracking_plan_payload_sha256": plan["plan_sha256"],
            "track_seal_raw_sha256": sha256_file(args.track_dir.resolve(strict=True) / "visual_track_seal.json"),
            "availability_seal_raw_sha256": sha256_file(args.availability_dir.resolve(strict=True) / "availability_seal.json"),
            "verified_input_source_hashes": source_hashes,
            "numpy_version": np.__version__,
            "geometry_release_raw_sha256": release_raw,
            "verified_source_roles": sorted(source_hashes),
        })
        output = args.output_dir.resolve(strict=False)
        output.mkdir(parents=True, exist_ok=False)
        receipt_path = output / "independent_reconstruction_receipt.json"
        raw = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        with receipt_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        print(json.dumps({
            "status": result["status"], "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "contexts": result["planned_contexts"], "tracks": result["planned_tracks"],
            "endpoints": result["planned_endpoints"],
            "max_abs_array_residual": result["max_abs_array_residual"],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
