#!/usr/bin/env python3
"""Independent reconstruction of the sealed timestamp/valid/phone availability.

This verifier reads only the 160 availability context NPZs and the 800 source
availability NPZs (timestamps_s, valid). It never opens feature NPZ contents,
mouth geometry, model scores, or outcomes. On PASS it writes a deterministic
archive of the exact 163 files in the frozen availability output namespace.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_PLAN_RAW = "13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f"
EXPECTED_PLAN_PAYLOAD = "26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e"
EXPECTED_TRACK_READBACK_RAW = "6d45048ada823b1626274f7f9e4a4435d5ea5065e0e5c520bb77f19f7a18bb16"
EXPECTED_TRACK_READBACK_PAYLOAD = "8aa31e9cdbd1d592dfecce8e7d20508aeb995a5c85875f2683fc7a8d8ec06618"
EXPECTED_ROOT_TRACK_ACCEPTANCE_RAW = "0169cbbdbfba1f86be29c033d433f40f311852631525a04958c8072b4c276d19"
EXPECTED_TRACK_MANIFEST_RAW = "2703afdb4fc1d91959b288dbc4d24ea98420aa34f624e1230c25901546986f7a"
EXPECTED_TRACK_MANIFEST_PAYLOAD = "2deeac2d67fbc7caea4a8bca4a82c70d57acdf855d6348a1113f8483f293c56a"
EXPECTED_TRACK_SEAL_RAW = "48249c94c85a4a8664ff6b7bcc8ec27c4b3fd3d853ea988a66c116d0e0ac73ff"
EXPECTED_TRACK_INVENTORY_RAW = "647a5447a1eef0c21a9a1c1404a83c3947573e76a94b5e7b45a0937645a11f75"
EXPECTED_TRACK_ROOT = "/dev/shm/tts-research-20261002/independent_validation/world_shape_visual_attempt08/output"
EXPECTED_AVAILABILITY_MANIFEST_RAW = "057a9b17d39b3d6b1196e676f378d582d377b5b465726776e8d4305cf935245a"
EXPECTED_AVAILABILITY_PAYLOAD = "1933994f2bb5fe74b64f6ee4bb4d3fc55337f8751eb7b2884e77596ae9b50eeb"
EXPECTED_AVAILABILITY_SEAL_RAW = "b905a13e88f323de5f013e194ae5b9b5e8a664884cd6b3809f1f15ff5995c5fe"
EXPECTED_AVAILABILITY_INVENTORY_RAW = "7b5b40cd177d8d880d96f2a5ebc70a46e29a032d03ecc744641a702ffc24d1ae"
EXPECTED_CORE_SHA = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
EXPECTED_TONE_SHA = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
EXPECTED_CAL_SHA = "0a1897a03248322a4868ceae5e194b848212ef33858b4a1cb002f5fe58bd21a4"
EXPECTED_CAL_FREEZE_SHA = "45cc2ee35980177b126a05b79fb19caa06e43624c24c1c7f85c41575c677b287"
EXPECTED_MFA_ELIGIBILITY_RAW = "acc6cacfb8f5b35c13163157c6de70f90bc6e34ae54364036935923982a85ba5"
EXPECTED_MFA_ELIGIBILITY_PAYLOAD = "4006b3476ad9d32910d24754a0105f26040e386144111a191e6e7bea934730ae"
EXPECTED_EXTRACTOR_SHA = "aa86f4011e4a93e345471142ce29d6fcb739d1d4d89254f95f7685d2bf7fb41f"
EXPECTED_VISUAL_DESIGN_SHA = "0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8"
EXPECTED_EVAL_DESIGN_SHA = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
EXPECTED_PIN_CORRECTION_SHA = "cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6"
FPS = 25.0
LATENCY_S = -0.085
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
CONTEXT_NPZ_KEYS = {
    "q00_timestamps_s", "common_valid", "common_adjacent",
    *(f"index_map_{cell}" for cell in CELLS),
    *(f"valid_on_q00_{cell}" for cell in CELLS),
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class AvailabilityQAError(RuntimeError):
    pass


def canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        raise AvailabilityQAError(f"json_read_failed:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise AvailabilityQAError(f"json_not_object:{path.name}")
    return value


def verify_embedded(value: dict[str, Any], key: str) -> str:
    actual = sha_bytes(canon({k: v for k, v in value.items() if k != key}))
    if value.get(key) != actual:
        raise AvailabilityQAError(f"embedded_fingerprint_mismatch:{key}")
    return actual


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AvailabilityQAError("relative_path_missing")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise AvailabilityQAError("unsafe_relative_path")
    path = (root / Path(*posix.parts)).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if path != resolved_root and resolved_root not in path.parents:
        raise AvailabilityQAError("path_escaped_root")
    if not path.is_file():
        raise AvailabilityQAError("path_not_regular_file")
    return path


def arr_sha(array: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(array).tobytes())


def load_core(path: Path, expected_sha: str) -> Any:
    path = path.resolve(strict=True)
    if sha_file(path) != expected_sha:
        raise AvailabilityQAError("visual_core_sha_mismatch")
    spec = importlib.util.spec_from_file_location("availability_rebuild_visual_core", path)
    if spec is None or spec.loader is None:
        raise AvailabilityQAError("visual_core_import_unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def phone_class(label: Any, core: Any) -> str | None:
    tone_marks = ("˥", "˦", "˧", "˨", "˩")
    value = str(label or "").strip().lower()
    value = "".join(ch for ch in value if ch not in tone_marks)
    normalized = core.normalize_phone(value)
    if normalized in core.BILABIAL_BASES:
        return "bilabial"
    if normalized in core.OPEN_VOWEL_BASES:
        return "open_vowel"
    return None


def match_clock(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    tol = 0.5 * float(np.median(np.diff(reference))) + 1e-12 if reference.size > 1 else 0.5 / FPS
    output = np.full(reference.shape[0], -1, dtype=np.int32)
    i = j = 0
    while i < len(reference) and j < len(candidate):
        delta = float(candidate[j] - reference[i])
        if abs(delta) <= tol:
            output[i] = j
            i += 1
            j += 1
        elif delta < -tol:
            j += 1
        else:
            i += 1
    return output


def read_npz_exact(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise AvailabilityQAError(f"npz_keyset_mismatch:{path.name}")
            return {key: np.asarray(archive[key]) for key in keys}
    except AvailabilityQAError:
        raise
    except Exception as exc:
        raise AvailabilityQAError(f"npz_read_failed:{path.name}:{type(exc).__name__}") from exc


def _record_map(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for record in records:
        if record.get("status") != "complete":
            raise AvailabilityQAError("track_record_not_complete")
        key = (str(record.get("sample_id")), str(record.get("direction")), str(record.get("cell")))
        if key in result:
            raise AvailabilityQAError("duplicate_track_record")
        result[key] = record
    return result


def _rebuild_tokens(phones: list[dict[str, Any]], common: np.ndarray, q00: np.ndarray, core: Any,
                    frame_minimum: int) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    token_rows = []
    expected = {"bilabial": 0, "open_vowel": 0}
    supported = {"bilabial": 0, "open_vowel": 0}
    for phone in phones:
        cls = phone_class(phone.get("label"), core)
        if cls is None:
            continue
        start = float(phone["start_s"])
        end = float(phone["end_s"])
        shifted_start = start + LATENCY_S
        shifted_end = end + LATENCY_S
        mask = common & (q00 >= shifted_start) & (q00 < shifted_end)
        frame_count = int(mask.sum())
        is_supported = frame_count >= frame_minimum
        expected[cls] += 1
        supported[cls] += int(is_supported)
        token_rows.append({
            "phone_index": int(phone["phone_index"]),
            "raw_label": phone["label"],
            "normalized_label": phone["normalized_label"],
            "visual_class": cls,
            "start_s": phone["start_s"],
            "end_s": phone["end_s"],
            "shifted_start_s": shifted_start,
            "shifted_end_s": shifted_end,
            "common_frames": frame_count,
            "supported": is_supported,
            "q00_indices": np.flatnonzero(mask).astype(np.int32).tolist(),
        })
    return token_rows, expected, supported


def verify_and_build(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve(strict=True)
    plan = read_json(plan_path)
    plan_payload = verify_embedded(plan, "plan_sha256")
    plan_raw = sha_file(plan_path)
    if plan_raw != EXPECTED_PLAN_RAW or plan_payload != EXPECTED_PLAN_PAYLOAD:
        raise AvailabilityQAError("tracking_plan_binding_mismatch")
    if len(plan.get("context_rows", [])) != 160:
        raise AvailabilityQAError("tracking_plan_context_count_mismatch")

    track_root = args.track_dir.resolve(strict=True)
    if str(track_root) != EXPECTED_TRACK_ROOT:
        raise AvailabilityQAError("track_root_path_binding_mismatch")
    track_manifest_path = track_root / "visual_track_manifest.json"
    track_seal_path = track_root / "visual_track_seal.json"
    track_inventory_path = track_root / "all_files_sha256.json"
    track_manifest = read_json(track_manifest_path)
    track_seal = read_json(track_seal_path)
    track_inventory = read_json(track_inventory_path).get("files")
    track_raw = sha_file(track_manifest_path)
    track_payload = verify_embedded(track_manifest, "manifest_sha256")
    track_seal_raw = sha_file(track_seal_path)
    track_inventory_raw = sha_file(track_inventory_path)
    if (track_raw != EXPECTED_TRACK_MANIFEST_RAW or track_payload != EXPECTED_TRACK_MANIFEST_PAYLOAD
            or track_seal_raw != EXPECTED_TRACK_SEAL_RAW or track_inventory_raw != EXPECTED_TRACK_INVENTORY_RAW):
        raise AvailabilityQAError("track_source_seal_mismatch")
    if (track_manifest.get("schema") != "world_shape_visual_track_manifest_v1"
            or track_manifest.get("status") != "COMPLETE"
            or track_manifest.get("completed_videos") != 800
            or track_manifest.get("failed_videos") != 0
            or len(track_manifest.get("records", [])) != 800
            or track_manifest.get("tracking_plan_raw_sha256") != plan_raw
            or track_manifest.get("tracking_plan_sha256") != plan_payload
            or track_seal.get("status") != "SEALED"
            or track_seal.get("manifest_sha256") != track_raw
            or track_seal.get("manifest_payload_sha256") != track_payload
            or track_seal.get("all_files_sha256") != track_inventory_raw):
        raise AvailabilityQAError("track_manifest_or_seal_contract_mismatch")
    if not isinstance(track_inventory, dict):
        raise AvailabilityQAError("track_inventory_missing")
    track_records = _record_map(track_manifest["records"])
    if len(track_records) != 800 or len(track_inventory) != 1603:
        raise AvailabilityQAError("track_record_count_mismatch")

    readback_path = args.track_readback.resolve(strict=True)
    readback = read_json(readback_path)
    if sha_file(readback_path) != EXPECTED_TRACK_READBACK_RAW or readback.get("report_payload_sha256") != EXPECTED_TRACK_READBACK_PAYLOAD:
        raise AvailabilityQAError("track_readback_report_binding_mismatch")
    if verify_embedded(readback, "report_payload_sha256") != EXPECTED_TRACK_READBACK_PAYLOAD:
        raise AvailabilityQAError("track_readback_report_fingerprint_mismatch")
    acceptance_path = args.track_acceptance.resolve(strict=True)
    acceptance = read_json(acceptance_path)
    if sha_file(acceptance_path) != EXPECTED_ROOT_TRACK_ACCEPTANCE_RAW:
        raise AvailabilityQAError("root_track_acceptance_sha_mismatch")
    if (acceptance.get("schema") != "ROOT_TRACKING03_METADATA_QUALIFICATION_V1"
            or acceptance.get("status") != "ROOT_ACCEPTED_COMPLETE_800_TRACKING_WITH_PRESERVED_WRAPPER_FAILURE"
            or acceptance.get("geometry_read_authorized") is not False
            or acceptance.get("geometry_values_read") is not False
            or acceptance.get("independent_checker_exit") != 0
            or acceptance.get("original_wrapper_exit") != 1
            or acceptance.get("all_five_cell_clocks_exact_equal") is not True
            or acceptance.get("availability_npz_verified") != 800
            or acceptance.get("opaque_feature_npz_verified") != 800
            or acceptance.get("verified_records") != 800
            or acceptance.get("report_raw_sha256") != EXPECTED_TRACK_READBACK_RAW
            or acceptance.get("report_payload_sha256") != EXPECTED_TRACK_READBACK_PAYLOAD):
        raise AvailabilityQAError("root_track_acceptance_contract_mismatch")
    if (readback.get("status") != "TRACK_OUTPUT_INDEPENDENTLY_VERIFIED_WITH_KNOWN_WRAPPER_FAILURE"
            or readback.get("evaluation_effects_read") is not False
            or readback.get("evaluation_results_parsed") is not False
            or readback.get("model_forward_performed") is not False
            or readback.get("mouth_geometry_values_read") is not False
            or readback.get("wrapper_exitcode_interpreted_as_producer_failure") is not False
            or readback.get("evidence", {}).get("contexts_verified") != 160
            or readback.get("evidence", {}).get("video_records_verified") != 800
            or readback.get("evidence", {}).get("tracking_output_root") != EXPECTED_TRACK_ROOT
            or readback.get("evidence", {}).get("tracking_manifest_raw_sha256") != track_raw
            or readback.get("evidence", {}).get("tracking_seal_raw_sha256") != track_seal_raw
            or readback.get("evidence", {}).get("tracking_inventory_raw_sha256") != track_inventory_raw):
        raise AvailabilityQAError("track_readback_scope_or_binding_mismatch")
    if (readback.get("evidence", {}).get("readback_checker_source", {}).get("raw_sha256")
            != "8e84cb1683881d10aa890af316886a7434f4728b9a190d5a0e8f1f18fed7bdfd"):
        raise AvailabilityQAError("track_readback_checker_binding_mismatch")
    if (readback.get("evidence", {}).get("worker_control", {}).get("worker_exit_code") != 1
            or not str(readback.get("evidence", {}).get("worker_control", {}).get("wrapper_accounting_failure", "")).startswith("known extraction_stdout_object_count_mismatch")):
        raise AvailabilityQAError("known_tracking_wrapper_failure_not_preserved")
    readback_records = readback.get("record_inventory")
    if not isinstance(readback_records, list) or len(readback_records) != 800:
        raise AvailabilityQAError("track_readback_record_inventory_missing")
    readback_by_key = {}
    for row in readback_records:
        key = (str(row.get("sample_id")), str(row.get("direction")), str(row.get("cell")))
        if key in readback_by_key:
            raise AvailabilityQAError("duplicate_track_readback_record")
        readback_by_key[key] = row
    if set(readback_by_key) != set(track_records):
        raise AvailabilityQAError("track_readback_identity_set_mismatch")
    for key, record in track_records.items():
        verified = readback_by_key[key]
        if (verified.get("availability_relative_path") != record.get("availability_relative_path")
                or verified.get("availability_raw_sha256") != record.get("availability_sha256")
                or verified.get("feature_relative_path") != record.get("feature_relative_path")
                or verified.get("feature_raw_sha256") != record.get("feature_sha256")
                or verified.get("official_frame_count") != record.get("official_frame_count")
                or verified.get("timestamp_array_sha256") != record.get("timestamp_sha256")
                or verified.get("valid_frames") != record.get("valid_frames")):
            raise AvailabilityQAError("track_readback_record_binding_mismatch")

    mfa_path = args.mfa_eligibility.resolve(strict=True)
    mfa = read_json(mfa_path)
    mfa_raw = sha_file(mfa_path)
    mfa_payload = verify_embedded(mfa, "manifest_sha256")
    if (mfa_raw != EXPECTED_MFA_ELIGIBILITY_RAW or mfa_payload != EXPECTED_MFA_ELIGIBILITY_PAYLOAD
            or len(mfa.get("records", [])) != 80):
        raise AvailabilityQAError("original_mfa_eligibility_source_mismatch")
    mfa_by_id = {str(row.get("id")): row for row in mfa["records"]}
    if len(mfa_by_id) != 80:
        raise AvailabilityQAError("original_mfa_eligibility_duplicate_id")

    for role, path, expected in (
        ("visual_core", args.visual_core, EXPECTED_CORE_SHA),
        ("tone_adapter", args.tone_adapter, EXPECTED_TONE_SHA),
        ("calibration", args.calibration, EXPECTED_CAL_SHA),
        ("calibration_freeze", args.calibration_freeze, EXPECTED_CAL_FREEZE_SHA),
        ("extractor", args.extractor_source, EXPECTED_EXTRACTOR_SHA),
        ("visual_design", args.visual_design, EXPECTED_VISUAL_DESIGN_SHA),
        ("evaluation_design", args.evaluation_design, EXPECTED_EVAL_DESIGN_SHA),
        ("pin_correction", args.pin_correction, EXPECTED_PIN_CORRECTION_SHA),
    ):
        if sha_file(path.resolve(strict=True)) != expected:
            raise AvailabilityQAError(f"bound_source_sha_mismatch:{role}")
    plan_source_roles = {
        "tracking_extractor_source": EXPECTED_EXTRACTOR_SHA,
        "visual_design": EXPECTED_VISUAL_DESIGN_SHA,
        "evaluation_design": EXPECTED_EVAL_DESIGN_SHA,
        "execution_pin_correction": EXPECTED_PIN_CORRECTION_SHA,
        "feature_helper": "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4",
        "landmarker_asset": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "visual_core_source": EXPECTED_CORE_SHA,
        "tone_adapter_source": EXPECTED_TONE_SHA,
        "visual_calibration": EXPECTED_CAL_SHA,
        "visual_calibration_freeze": EXPECTED_CAL_FREEZE_SHA,
        "original_mfa_eligibility": EXPECTED_MFA_ELIGIBILITY_RAW,
    }
    for role, expected in plan_source_roles.items():
        if plan.get("input_seals", {}).get(role, {}).get("raw_sha256") != expected:
            raise AvailabilityQAError(f"tracking_plan_source_binding_mismatch:{role}")
    core = load_core(args.visual_core, EXPECTED_CORE_SHA)
    calibration = read_json(args.calibration)
    minimums = {str(k): int(v) for k, v in calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {}).items()}
    frame_minimum = int(calibration.get("protocol", {}).get("phone_min_common_frames", 0))
    if minimums != {"bilabial": 1, "open_vowel": 1} or frame_minimum != 1:
        raise AvailabilityQAError("frozen_phone_threshold_mismatch")
    if not math.isclose(float(calibration.get("protocol", {}).get("latency", {}).get("latency_s")), LATENCY_S, rel_tol=0, abs_tol=1e-12):
        raise AvailabilityQAError("frozen_latency_mismatch")

    availability_root = args.availability_dir.resolve(strict=True)
    manifest_path = availability_root / "availability_freeze.json"
    seal_path = availability_root / "availability_seal.json"
    inventory_path = availability_root / "all_files_sha256.json"
    manifest = read_json(manifest_path)
    seal = read_json(seal_path)
    inventory_obj = read_json(inventory_path)
    manifest_raw = sha_file(manifest_path)
    manifest_payload = verify_embedded(manifest, "manifest_sha256")
    seal_raw = sha_file(seal_path)
    inventory_raw = sha_file(inventory_path)
    if (manifest_raw != EXPECTED_AVAILABILITY_MANIFEST_RAW or manifest_payload != EXPECTED_AVAILABILITY_PAYLOAD
            or seal_raw != EXPECTED_AVAILABILITY_SEAL_RAW or inventory_raw != EXPECTED_AVAILABILITY_INVENTORY_RAW):
        raise AvailabilityQAError("availability_output_seal_mismatch")
    if (manifest.get("schema") != "world_shape_visual_availability_freeze_v1"
            or manifest.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or manifest.get("tracking_plan_raw_sha256") != plan_raw
            or manifest.get("tracking_plan_payload_sha256") != plan_payload
            or manifest.get("track_manifest_raw_sha256") != track_raw
            or manifest.get("track_manifest_payload_sha256") != track_payload
            or manifest.get("track_seal_raw_sha256") != track_seal_raw
            or manifest.get("track_files_inventory_raw_sha256") != track_inventory_raw
            or manifest.get("visual_design_sha256") != EXPECTED_VISUAL_DESIGN_SHA
            or manifest.get("tone_adapter_sha256") != EXPECTED_TONE_SHA
            or manifest.get("visual_core_sha256") != EXPECTED_CORE_SHA
            or manifest.get("calibration_raw_sha256") != EXPECTED_CAL_SHA
            or manifest.get("calibration_freeze_raw_sha256") != EXPECTED_CAL_FREEZE_SHA
            or manifest.get("coverage", {}).get("geometry_values_read") is not False
            or manifest.get("coverage", {}).get("evaluation_effects_read") is not False
            or seal.get("status") != "FROZEN_BEFORE_GEOMETRY_READ"
            or seal.get("manifest_raw_sha256") != manifest_raw
            or seal.get("manifest_payload_sha256") != manifest_payload
            or seal.get("all_files_sha256") != inventory_raw
            or seal.get("context_files") != 160
            or seal.get("geometry_values_read") is not False):
        raise AvailabilityQAError("availability_manifest_or_seal_contract_mismatch")
    inventory = inventory_obj.get("files")
    if not isinstance(inventory, dict):
        raise AvailabilityQAError("availability_file_inventory_missing")
    context_rows = manifest.get("context_rows")
    if not isinstance(context_rows, list) or len(context_rows) != 160:
        raise AvailabilityQAError("availability_context_count_mismatch")
    expected_files = {"availability_freeze.json"} | {str(row["availability_context_path"]) for row in context_rows}
    if set(inventory) != expected_files or len(inventory) != 161:
        raise AvailabilityQAError("availability_inventory_member_set_mismatch")
    output_actual = {p.relative_to(availability_root).as_posix() for p in availability_root.rglob("*") if p.is_file()}
    if output_actual != expected_files | {"availability_seal.json", "all_files_sha256.json"}:
        raise AvailabilityQAError("availability_output_directory_member_set_mismatch")
    for relative, expected_sha in inventory.items():
        path = inside(availability_root, relative)
        if sha_file(path) != expected_sha:
            raise AvailabilityQAError(f"availability_inventory_file_hash_mismatch:{relative}")

    plan_rows = plan["context_rows"]
    if [row.get("context_order") for row in plan_rows] != list(range(160)):
        raise AvailabilityQAError("tracking_plan_order_not_dense")
    if len(track_records) != 800 or len({(r.get("sample_id"), r.get("direction")) for r in plan_rows}) != 160:
        raise AvailabilityQAError("source_identity_counts_invalid")
    for plan_row in plan_rows:
        sample = str(plan_row["sample_id"])
        mfa_row = mfa_by_id.get(sample)
        if mfa_row is None or mfa_row.get("speaker") != plan_row.get("speaker_id") or mfa_row.get("split") != "evaluation":
            raise AvailabilityQAError(f"original_mfa_identity_mismatch:{sample}")
        if sha_bytes(canon(mfa_row)) != plan_row.get("original_eligibility_record_sha256"):
            raise AvailabilityQAError(f"original_mfa_record_digest_mismatch:{sample}")
        if bool(mfa_row.get("pair_qualified")) is not bool(plan_row.get("original_mfa_pair_qualified")):
            raise AvailabilityQAError(f"original_mfa_pair_eligibility_mismatch:{sample}")
        condition = str(plan_row.get("recipient_condition"))
        if condition not in {"natural", "tts"}:
            raise AvailabilityQAError(f"recipient_condition_invalid:{sample}")
        mfa_side = mfa_row[condition]
        if mfa_side.get("textgrid_sha256") != plan_row.get("recipient_grid_sha256"):
            raise AvailabilityQAError(f"recipient_grid_mfa_binding_mismatch:{sample}:{condition}")
        mfa_tokens = mfa_side.get("tokens", [])
        phones = plan_row.get("recipient_phones", [])
        if len(mfa_tokens) != len(phones):
            raise AvailabilityQAError(f"recipient_phone_token_count_mismatch:{sample}:{condition}")
        for phone_index, (phone, token) in enumerate(zip(phones, mfa_tokens, strict=True)):
            if (phone.get("phone_index") != phone_index or phone.get("label") != token.get("raw_token")
                    or phone.get("start_s") != token.get("start_s") or phone.get("end_s") != token.get("end_s")):
                raise AvailabilityQAError(f"recipient_phone_token_identity_mismatch:{sample}:{condition}:{phone_index}")
        if bool(plan_row.get("overlay_used")):
            raise AvailabilityQAError(f"availability_must_preserve_original_grid:{sample}")

    plan_by_key = {}
    for row in plan_rows:
        key = (str(row["sample_id"]), str(row["direction"]))
        if key in plan_by_key:
            raise AvailabilityQAError("duplicate_plan_context")
        plan_by_key[key] = row
    avail_by_key = {}
    speaker_contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    frame_total = 0
    map_total = 0
    source_availability_files = 0
    expected_video_ids = set()
    for expected_order, (plan_row, frozen_row) in enumerate(zip(plan_rows, context_rows, strict=True)):
        sample = str(plan_row["sample_id"])
        speaker = str(plan_row["speaker_id"])
        direction = str(plan_row["direction"])
        key = (sample, direction)
        if key in avail_by_key:
            raise AvailabilityQAError("duplicate_availability_context")
        if (frozen_row.get("context_order") != expected_order or frozen_row.get("sample_id") != sample
                or frozen_row.get("speaker_id") != speaker or frozen_row.get("direction") != direction
                or frozen_row.get("recipient_condition") != plan_row.get("recipient_condition")
                or frozen_row.get("recipient_grid_sha256") != plan_row.get("recipient_grid_sha256")
                or frozen_row.get("recipient_phones") != plan_row.get("recipient_phones")
                or frozen_row.get("original_mfa_pair_qualified") is not plan_row.get("original_mfa_pair_qualified")
                or frozen_row.get("grid_overlay_used") is not plan_row.get("overlay_used")):
            raise AvailabilityQAError(f"context_identity_or_phone_binding_mismatch:{expected_order}")
        ctx_path = inside(availability_root, str(frozen_row["availability_context_path"]))
        ctx_raw = sha_file(ctx_path)
        if ctx_raw != frozen_row.get("availability_context_sha256") or inventory.get(frozen_row["availability_context_path"]) != ctx_raw:
            raise AvailabilityQAError(f"availability_context_file_binding_mismatch:{expected_order}")
        arrays = read_npz_exact(ctx_path, CONTEXT_NPZ_KEYS)
        q00 = arrays["q00_timestamps_s"]
        common = arrays["common_valid"]
        adjacent = arrays["common_adjacent"]
        if q00.dtype != np.float64 or common.dtype != np.bool_ or adjacent.dtype != np.bool_ or q00.ndim != 1:
            raise AvailabilityQAError(f"context_array_dtype_mismatch:{expected_order}")
        n = len(q00)
        if n != frozen_row.get("q00_frame_count") or q00.shape != (n,) or common.shape != (n,) or adjacent.shape != (max(0, n-1),):
            raise AvailabilityQAError(f"context_array_shape_mismatch:{expected_order}")
        if n < 2 or not np.isfinite(q00).all() or np.any(np.diff(q00) <= 0) or not np.allclose(np.diff(q00), 1.0 / FPS, rtol=0, atol=1e-9):
            raise AvailabilityQAError(f"context_q00_clock_invalid:{expected_order}")
        if arr_sha(q00) != frozen_row.get("q00_clock_sha256"):
            raise AvailabilityQAError(f"context_q00_clock_sha_mismatch:{expected_order}")
        if arr_sha(common) != frozen_row.get("common_valid_sha256") or arr_sha(adjacent) != frozen_row.get("common_adjacent_sha256"):
            raise AvailabilityQAError(f"context_common_mask_sha_mismatch:{expected_order}")

        projected_valid: dict[str, np.ndarray] = {}
        expected_maps: dict[str, np.ndarray] = {}
        for cell in CELLS:
            plan_cell = plan_row["cells"][cell]
            track_record = track_records.get((sample, direction, cell))
            if track_record is None:
                raise AvailabilityQAError(f"track_source_record_missing:{sample}:{direction}:{cell}")
            expected_video_ids.add((sample, direction, cell))
            if (track_record.get("ordered_eval_index") != plan_row.get("ordered_eval_index")
                    or track_record.get("official_frame_count") != plan_cell.get("official_frame_count")
                    or track_record.get("video_sha256") != plan_cell.get("video_sha256")
                    or track_record.get("video_relative_path") != plan_cell.get("video_relative_path")
                    or track_record.get("availability_relative_path") not in track_inventory
                    or track_inventory.get(track_record.get("availability_relative_path")) != track_record.get("availability_sha256")):
                raise AvailabilityQAError(f"track_source_record_binding_mismatch:{sample}:{direction}:{cell}")
            av_source = inside(track_root, track_record["availability_relative_path"])
            if sha_file(av_source) != track_record.get("availability_sha256"):
                raise AvailabilityQAError(f"track_source_availability_hash_mismatch:{sample}:{direction}:{cell}")
            source_arrays = read_npz_exact(av_source, {"timestamps_s", "valid"})
            times = source_arrays["timestamps_s"]
            valid = source_arrays["valid"]
            if times.dtype != np.float64 or valid.dtype != np.bool_ or times.ndim != 1 or valid.shape != times.shape or len(times) != plan_cell.get("official_frame_count"):
                raise AvailabilityQAError(f"source_availability_shape_dtype_mismatch:{sample}:{direction}:{cell}")
            if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                    or not np.allclose(np.diff(times), 1.0 / FPS, rtol=0, atol=1e-9)):
                raise AvailabilityQAError(f"source_availability_clock_invalid:{sample}:{direction}:{cell}")
            if int(valid.sum()) != track_record.get("valid_frames") or arr_sha(times) != track_record.get("timestamp_sha256"):
                raise AvailabilityQAError(f"source_availability_array_inventory_mismatch:{sample}:{direction}:{cell}")
            frozen_track_ref = frozen_row.get("track_refs", {}).get(cell, {})
            if (frozen_track_ref.get("feature_relative_path") != track_record.get("feature_relative_path")
                    or frozen_track_ref.get("feature_sha256") != track_record.get("feature_sha256")
                    or frozen_track_ref.get("availability_relative_path") != track_record.get("availability_relative_path")
                    or frozen_track_ref.get("availability_sha256") != track_record.get("availability_sha256")
                    or frozen_track_ref.get("array_inventory") != track_record.get("array_inventory")):
                raise AvailabilityQAError(f"availability_track_reference_mismatch:{sample}:{direction}:{cell}")
            source_availability_files += 1
            index_map = arrays[f"index_map_{cell}"]
            saved_valid = arrays[f"valid_on_q00_{cell}"]
            if index_map.dtype != np.int32 or saved_valid.dtype != np.bool_ or index_map.shape != (n,) or saved_valid.shape != (n,):
                raise AvailabilityQAError(f"context_cell_array_shape_dtype_mismatch:{sample}:{direction}:{cell}")
            rebuilt_map = match_clock(q00, times)
            if not np.array_equal(index_map, rebuilt_map):
                raise AvailabilityQAError(f"context_clock_map_reconstruction_mismatch:{sample}:{direction}:{cell}")
            projected = np.zeros(n, dtype=np.bool_)
            matched = rebuilt_map >= 0
            projected[matched] = valid[rebuilt_map[matched]]
            if not np.array_equal(saved_valid, projected):
                raise AvailabilityQAError(f"context_projected_valid_reconstruction_mismatch:{sample}:{direction}:{cell}")
            if arr_sha(index_map) != frozen_row.get("cell_index_map_sha256", {}).get(cell):
                raise AvailabilityQAError(f"context_index_map_sha_mismatch:{sample}:{direction}:{cell}")
            projected_valid[cell] = projected
            expected_maps[cell] = rebuilt_map
            map_total += int(len(rebuilt_map))
        rebuilt_common = np.logical_and.reduce([projected_valid[cell] for cell in CELLS])
        rebuilt_adjacent = rebuilt_common[:-1] & rebuilt_common[1:]
        if not np.array_equal(common, rebuilt_common):
            raise AvailabilityQAError(f"context_common_valid_reconstruction_mismatch:{expected_order}")
        if not np.array_equal(adjacent, rebuilt_adjacent):
            raise AvailabilityQAError(f"context_adjacent_reconstruction_mismatch:{expected_order}")
        if int(common.sum()) != frozen_row.get("common_frame_count") or int(adjacent.sum()) != frozen_row.get("common_adjacent_pair_count"):
            raise AvailabilityQAError(f"context_common_count_mismatch:{expected_order}")

        tokens, expected_counts, supported_counts = _rebuild_tokens(
            plan_row["recipient_phones"], common, q00, core, frame_minimum)
        if tokens != frozen_row.get("tokens"):
            raise AvailabilityQAError(f"context_phone_token_reconstruction_mismatch:{expected_order}")
        if expected_counts != frozen_row.get("expected_token_counts") or supported_counts != frozen_row.get("supported_token_counts"):
            raise AvailabilityQAError(f"context_phone_counts_mismatch:{expected_order}")
        original_qualified = bool(plan_row["original_mfa_pair_qualified"])
        eligible = original_qualified and all(supported_counts[cls] >= minimums[cls] for cls in minimums)
        if frozen_row.get("minimum_token_counts") != minimums or frozen_row.get("minimum_common_frames_per_token") != frame_minimum:
            raise AvailabilityQAError(f"context_phone_threshold_mismatch:{expected_order}")
        if frozen_row.get("availability_endpoint_eligible") is not eligible:
            raise AvailabilityQAError(f"context_phone_eligibility_mismatch:{expected_order}")
        frame_total += n
        avail_by_key[key] = frozen_row
        speaker_contexts.setdefault((speaker, direction), []).append(frozen_row)

    if expected_video_ids != set(track_records):
        raise AvailabilityQAError("track_record_identity_set_mismatch")
    expected_speaker_rows = []
    for speaker in sorted({str(row["speaker_id"]) for row in plan_rows}):
        for direction in DIRECTIONS:
            clips = sorted(speaker_contexts.get((speaker, direction), []), key=lambda row: str(row["sample_id"]))
            if len(clips) != 2:
                raise AvailabilityQAError(f"speaker_clip_count_mismatch:{speaker}:{direction}")
            expected_speaker_rows.append({
                "speaker_id": speaker,
                "direction": direction,
                "clip_ids": [str(row["sample_id"]) for row in clips],
                "both_clips_original_mfa_qualified": all(bool(row["original_mfa_pair_qualified"]) for row in clips),
                "complete_two_clip_phone_eligible": all(bool(row["availability_endpoint_eligible"]) for row in clips),
            })
    if manifest.get("speaker_rows") != expected_speaker_rows:
        raise AvailabilityQAError("speaker_eligibility_reconstruction_mismatch")
    eligible_by_direction = {direction: sum(row["complete_two_clip_phone_eligible"] for row in expected_speaker_rows if row["direction"] == direction) for direction in DIRECTIONS}
    expected_coverage = {
        "complete_contexts": 160,
        "complete_speakers": 40,
        "complete_video_tracks": 800,
        "phone_eligible_speakers_by_direction": eligible_by_direction,
        "original_mfa_unqualified_samples": sum(not bool(row["original_mfa_pair_qualified"]) for row in context_rows),
        "geometry_values_read": False,
        "evaluation_effects_read": False,
    }
    if manifest.get("coverage") != expected_coverage:
        raise AvailabilityQAError("availability_coverage_reconstruction_mismatch")

    # Archive exactly the sealed freeze output: 160 per-context NPZ + three JSONs.
    bundle_dir = args.bundle_dir.resolve(strict=False)
    if bundle_dir.exists():
        raise AvailabilityQAError("bundle_output_must_be_absent")
    bundle_dir.mkdir(parents=True, exist_ok=False)
    tar_path = bundle_dir / "availability_freeze_163_files.tar.gz"
    transfer_dir = bundle_dir / "transfer_chunks"
    transfer_dir.mkdir()
    members = sorted(expected_files | {"availability_seal.json", "all_files_sha256.json"})
    with tar_path.open("xb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for relative in members:
                    source = inside(availability_root, relative)
                    info = archive.gettarinfo(str(source), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
    chunk_size = 6 * 1024 * 1024
    tar_hash = sha_file(tar_path)
    chunks = []
    with tar_path.open("rb") as stream:
        index = 0
        while True:
            data = stream.read(chunk_size)
            if not data:
                break
            name = f"availability_bundle_part_{index:03d}.bin"
            path = transfer_dir / name
            with path.open("xb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            chunks.append({"index": index, "path": f"transfer_chunks/{name}", "bytes": len(data), "sha256": sha_bytes(data)})
            index += 1
    if not chunks or sum(item["bytes"] for item in chunks) != tar_path.stat().st_size or any(item["bytes"] > chunk_size for item in chunks):
        raise AvailabilityQAError("bundle_chunk_accounting_failed")
    report = {
        "schema": "independent_world_shape_visual_availability_reconstruction_v1",
        "status": "PASS",
        "scope": "all 160 availability context NPZs and source timestamps/valid masks only; no feature NPZ contents, mouth geometry, scores, outcomes, or model forward",
        "evaluation_effects_read": False,
        "evaluation_outcomes_read": False,
        "geometry_values_read": False,
        "model_forward_performed": False,
        "input_bindings": {
            "tracking_plan_raw_sha256": plan_raw,
            "tracking_plan_payload_sha256": plan_payload,
            "track_readback_raw_sha256": EXPECTED_TRACK_READBACK_RAW,
            "track_readback_payload_sha256": EXPECTED_TRACK_READBACK_PAYLOAD,
            "root_track_acceptance_raw_sha256": EXPECTED_ROOT_TRACK_ACCEPTANCE_RAW,
            "track_manifest_raw_sha256": track_raw,
            "track_manifest_payload_sha256": track_payload,
            "track_seal_raw_sha256": track_seal_raw,
            "track_inventory_raw_sha256": track_inventory_raw,
            "availability_manifest_raw_sha256": manifest_raw,
            "availability_manifest_payload_sha256": manifest_payload,
            "availability_seal_raw_sha256": seal_raw,
            "availability_inventory_raw_sha256": inventory_raw,
            "visual_core_sha256": EXPECTED_CORE_SHA,
            "tone_adapter_sha256": EXPECTED_TONE_SHA,
            "calibration_raw_sha256": EXPECTED_CAL_SHA,
            "calibration_freeze_raw_sha256": EXPECTED_CAL_FREEZE_SHA,
            "original_mfa_eligibility_raw_sha256": mfa_raw,
            "original_mfa_eligibility_payload_sha256": mfa_payload,
            "extractor_sha256": EXPECTED_EXTRACTOR_SHA,
            "visual_design_sha256": EXPECTED_VISUAL_DESIGN_SHA,
            "evaluation_design_sha256": EXPECTED_EVAL_DESIGN_SHA,
            "pin_correction_sha256": EXPECTED_PIN_CORRECTION_SHA,
        },
        "coverage": {
            "contexts_reconstructed": len(context_rows),
            "source_availability_npz_read": source_availability_files,
            "source_feature_npz_contents_read": 0,
            "context_npz_files_reconstructed": len(context_rows),
            "context_arrays_reconstructed": len(context_rows) * 13,
            "frame_clock_values_checked": frame_total,
            "cell_clock_indices_reconstructed": map_total,
            "phone_tokens_reconstructed": sum(len(row["tokens"]) for row in context_rows),
            "phone_eligible_speakers_by_direction": eligible_by_direction,
            "original_mfa_unqualified_context_rows": expected_coverage["original_mfa_unqualified_samples"],
            "bundle_members": len(members),
        },
        "bundle": {
            "tar_path": str(tar_path),
            "tar_bytes": tar_path.stat().st_size,
            "tar_sha256": tar_hash,
            "members": members,
            "chunks": chunks,
        },
        "context_checks": [
            {"context_order": int(row["context_order"]), "sample_id": row["sample_id"], "speaker_id": row["speaker_id"],
             "direction": row["direction"], "availability_context_sha256": row["availability_context_sha256"],
             "q00_frames": int(row["q00_frame_count"]), "common_frames": int(row["common_frame_count"]),
             "common_adjacent_pairs": int(row["common_adjacent_pair_count"]),
             "phone_token_count": len(row["tokens"]), "original_mfa_pair_qualified": bool(row["original_mfa_pair_qualified"]),
             "endpoint_eligible": bool(row["availability_endpoint_eligible"])}
            for row in context_rows
        ],
    }
    report["report_payload_sha256"] = sha_bytes(canon(report))
    report_path = bundle_dir / "independent_availability_reconstruction.json"
    with report_path.open("xb") as stream:
        stream.write(canon(report))
        stream.flush()
        os.fsync(stream.fileno())
    bundle_manifest = {
        "schema": "world_shape_visual_availability_bundle_v1",
        "status": "SEALED",
        "report_path": report_path.name,
        "report_raw_sha256": sha_file(report_path),
        "report_payload_sha256": report["report_payload_sha256"],
        "tar_path": tar_path.name,
        "tar_bytes": tar_path.stat().st_size,
        "tar_sha256": tar_hash,
        "member_count": len(members),
        "chunks": chunks,
        "bundle_sha256": "",
    }
    bundle_manifest["bundle_sha256"] = sha_bytes(canon({k: v for k, v in bundle_manifest.items() if k != "bundle_sha256"}))
    manifest_out = bundle_dir / "availability_bundle_manifest.json"
    with manifest_out.open("xb") as stream:
        stream.write(canon(bundle_manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "status": "PASS",
        "report_path": str(report_path),
        "report_raw_sha256": sha_file(report_path),
        "report_payload_sha256": report["report_payload_sha256"],
        "bundle_manifest_path": str(manifest_out),
        "bundle_manifest_raw_sha256": sha_file(manifest_out),
        "bundle_manifest_payload_sha256": bundle_manifest["bundle_sha256"],
        "tar_path": str(tar_path),
        "tar_bytes": tar_path.stat().st_size,
        "tar_sha256": tar_hash,
        "members": len(members),
        "chunks": len(chunks),
        "contexts": 160,
        "source_availability_npz_read": source_availability_files,
        "feature_npz_contents_read": 0,
        "geometry_values_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--track-readback", type=Path, required=True)
    parser.add_argument("--track-acceptance", type=Path, required=True)
    parser.add_argument("--availability-dir", type=Path, required=True)
    parser.add_argument("--visual-core", type=Path, required=True)
    parser.add_argument("--tone-adapter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--mfa-eligibility", type=Path, required=True)
    parser.add_argument("--extractor-source", type=Path, required=True)
    parser.add_argument("--visual-design", type=Path, required=True)
    parser.add_argument("--evaluation-design", type=Path, required=True)
    parser.add_argument("--pin-correction", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_and_build(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "geometry_values_read": False,
                          "evaluation_effects_read": False, "model_forward_performed": False}, separators=(",", ":")), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
