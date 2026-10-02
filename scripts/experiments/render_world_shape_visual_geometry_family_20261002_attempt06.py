#!/usr/bin/env python3
"""Render the frozen 54-endpoint WORLD visual-geometry family.

This is a presentation-only consumer. The primary consumer and its independent
reconstruction must already have passed. It reads only their sealed geometry
outputs, makes no new estimates, and retains all planned rows including any
incomplete endpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


PRIMARY_SOURCE_SHA256 = "ae9daa7191c401e8c86852bb1bae4666fd1739e897a19c4496b980ce1a52a568"
INDEPENDENT_SOURCE_SHA256 = "2ba5169d61f4e818a505791ae4f97e358b41cca346f53fefbdb102c9b978c06c"
GEOMETRY_RELEASE_RAW_SHA256 = "4a8a6265de17566ee237e2ed79f034f3df427fd186daa64b345f98253d2005ae"
GEOMETRY_RELEASE_PAYLOAD_SHA256 = "6eb84aed937a227012627533205720bd145b7aeb63d0a457d7b024f471bd542f"
PLAN_RAW_SHA256 = "13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f"
PLAN_PAYLOAD_SHA256 = "26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e"
DESIGN_SHA256 = "0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8"
PIN_CORRECTION_SHA256 = "cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6"
VISUAL_CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
TONE_ADAPTER_SHA256 = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
PLOTDEPS_RUNTIME_MANIFEST_SHA256 = "bf5586ad38ada97add6fb24aa125901d5a80b8bacc5b246482ac9a98975789f4"
PLOTDEPS_RUNTIME_READBACK_SHA256 = "4e420047f9cf4d0e99e74cf799db336708fcb2b4ab3ea0360b4f22567a1e6abe"
PLOTDEPS_TREE_SHA256 = "fe25379439c3401b54309d5b18fdfc388380ef9e568bd7ce6ed1e3d6aaa6df83"
PLOTDEPS_FILE_COUNT = 1712
PLOTDEPS_TOTAL_BYTES = 53345343
ENDPOINT_COUNT = 54
SPEAKER_COUNT = 40
METRICS = (
    "mean_aperture",
    "mean_absolute_aperture_velocity_per_s",
    "phone_class_separation",
)
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
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
EXPECTED_FILES = {
    "geometry_values.npz",
    "phone_token_measurements.npz",
    "speaker_endpoints.npz",
    "bootstrap_family.npz",
    "geometry_family_manifest.json",
    "all_files_sha256.json",
    "geometry_family_seal.json",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UNIT = {
    "mean_aperture": "eye-distance-normalized aperture (dimensionless)",
    "mean_absolute_aperture_velocity_per_s": "normalized aperture change per second (s⁻¹)",
    "phone_class_separation": "open-vowel minus bilabial normalized aperture (dimensionless)",
}
METRIC_TITLE = {
    "mean_aperture": "Mean aperture",
    "mean_absolute_aperture_velocity_per_s": "Mean absolute aperture velocity",
    "phone_class_separation": "Phone-class aperture separation",
}
DIRECTION_LABEL = {"tts_to_natural": "T→N", "natural_to_tts": "N→T"}
DIRECTION_COLOR = {"tts_to_natural": "#286b9f", "natural_to_tts": "#ba5635"}
DIRECTION_MARKER = {"tts_to_natural": "o", "natural_to_tts": "s"}


class RenderError(RuntimeError):
    pass


LINE2D: Any = None


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RenderError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise RenderError(f"json_not_object:{path.name}")
    return value


def _embedded_sha(value: dict[str, Any], field: str, label: str) -> str:
    actual = _sha_bytes(_canonical({key: item for key, item in value.items() if key != field}))
    if value.get(field) != actual:
        raise RenderError(f"embedded_fingerprint_mismatch:{label}")
    return actual


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise RenderError(f"invalid_sha256:{label}")
    return value


def _verify_plotdeps(args: argparse.Namespace) -> dict[str, Any]:
    plotdeps = args.plotdeps_root.resolve(strict=True)
    source_receipt_path = args.plotdeps_runtime_manifest.resolve(strict=True)
    readback_path = args.plotdeps_runtime_readback.resolve(strict=True)
    if _sha_file(source_receipt_path) != PLOTDEPS_RUNTIME_MANIFEST_SHA256:
        raise RenderError("plotdeps_runtime_manifest_raw_sha_mismatch")
    if _sha_file(readback_path) != PLOTDEPS_RUNTIME_READBACK_SHA256:
        raise RenderError("plotdeps_runtime_readback_raw_sha_mismatch")
    readback = _read_json(readback_path)
    if (readback.get("status") != "ROOT_RENDER_RUNTIME_AND_PRIOR_QA_DISK_READBACK_PASS"
            or readback.get("source_receipt_sha256") != PLOTDEPS_RUNTIME_MANIFEST_SHA256
            or readback.get("matplotlib") != "3.11.2" or readback.get("numpy") != "2.3.5"
            or readback.get("python") != "3.12.14"):
        raise RenderError("plotdeps_runtime_readback_contract_mismatch")
    receipt = _read_json(source_receipt_path)
    component = receipt.get("components", {}).get("matplotlib_and_dependencies", {})
    recorded_root = component.get("root")
    if not isinstance(recorded_root, str) or not recorded_root:
        raise RenderError("plotdeps_root_missing_from_receipt")
    try:
        resolved_recorded_root = Path(recorded_root).resolve(strict=True)
    except OSError as exc:
        raise RenderError("plotdeps_recorded_root_unavailable") from exc
    if (receipt.get("status") != "PREPARED_NOT_RENDERED"
            or resolved_recorded_root != plotdeps
            or component.get("tree_sha256") != PLOTDEPS_TREE_SHA256
            or component.get("file_count") != PLOTDEPS_FILE_COUNT
            or component.get("total_bytes") != PLOTDEPS_TOTAL_BYTES):
        raise RenderError("plotdeps_tree_identity_mismatch")
    entries = component.get("files")
    if not isinstance(entries, list) or len(entries) != PLOTDEPS_FILE_COUNT:
        raise RenderError("plotdeps_file_inventory_missing")
    expected: dict[str, tuple[int, str]] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise RenderError("plotdeps_inventory_row_invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RenderError("plotdeps_inventory_path_invalid")
        digest = _sha(row.get("sha256"), f"plotdeps:{relative}")
        size = row.get("bytes")
        if not isinstance(size, int) or size < 0 or relative in expected:
            raise RenderError("plotdeps_inventory_duplicate_or_size_invalid")
        expected[relative] = (size, digest)
    actual_paths: set[str] = set()
    for path in plotdeps.rglob("*"):
        if path.is_symlink():
            raise RenderError("plotdeps_tree_contains_symlink")
        if path.is_file():
            relative = path.relative_to(plotdeps).as_posix()
            actual_paths.add(relative)
            entry = expected.get(relative)
            if entry is None or path.stat().st_size != entry[0] or _sha_file(path) != entry[1]:
                raise RenderError(f"plotdeps_file_mismatch:{relative}")
    if actual_paths != set(expected):
        raise RenderError("plotdeps_tree_fileset_mismatch")
    return {
        "runtime_manifest_path": str(source_receipt_path),
        "runtime_manifest_raw_sha256": PLOTDEPS_RUNTIME_MANIFEST_SHA256,
        "runtime_readback_path": str(readback_path),
        "runtime_readback_raw_sha256": PLOTDEPS_RUNTIME_READBACK_SHA256,
        "plotdeps_root": str(plotdeps),
        "plotdeps_tree_sha256": PLOTDEPS_TREE_SHA256,
        "plotdeps_file_count": PLOTDEPS_FILE_COUNT,
        "plotdeps_total_bytes": PLOTDEPS_TOTAL_BYTES,
    }


def _load_plotting(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    global LINE2D
    plotdeps = args.plotdeps_root.resolve(strict=True)
    if not plotdeps.is_dir():
        raise RenderError("plotdeps_root_not_directory")
    runtime_binding = _verify_plotdeps(args)
    sys.path.insert(0, str(plotdeps))
    import numpy as np  # type: ignore[import-not-found]
    import matplotlib  # type: ignore[import-not-found]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    from matplotlib.lines import Line2D  # type: ignore[import-not-found]
    LINE2D = Line2D

    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__, "backend": "Agg"}
    if versions["numpy"] != "2.3.5" or versions["matplotlib"] != "3.11.2":
        raise RenderError(f"plot_runtime_version_mismatch:{versions}")
    return np, plt, {**versions, **runtime_binding}


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    geometry_root = args.geometry_dir.resolve(strict=True)
    receipt_path = args.independent_receipt.resolve(strict=True)
    primary_source = args.primary_source.resolve(strict=True)
    expected_receipt_raw = _sha(args.expected_independent_receipt_sha256, "independent_receipt")
    for label, expected in (("primary", PRIMARY_SOURCE_SHA256), ("independent", INDEPENDENT_SOURCE_SHA256),
                            ("plan_raw", PLAN_RAW_SHA256), ("plan_payload", PLAN_PAYLOAD_SHA256),
                            ("design", DESIGN_SHA256), ("pin_correction", PIN_CORRECTION_SHA256),
                            ("visual_core", VISUAL_CORE_SHA256), ("tone_adapter", TONE_ADAPTER_SHA256)):
        _sha(expected, label)
    if _sha_file(primary_source) != PRIMARY_SOURCE_SHA256:
        raise RenderError("primary_source_hash_mismatch")
    receipt_raw = _sha_file(receipt_path)
    if receipt_raw != expected_receipt_raw:
        raise RenderError("independent_receipt_raw_hash_mismatch")
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "INDEPENDENT_RECONSTRUCTION_PASS":
        raise RenderError("independent_reconstruction_not_pass")
    if receipt.get("independent_source_sha256") != INDEPENDENT_SOURCE_SHA256:
        raise RenderError("independent_source_pin_mismatch")
    if receipt.get("primary_source_sha256") != PRIMARY_SOURCE_SHA256:
        raise RenderError("receipt_primary_source_mismatch")
    if (receipt.get("planned_contexts"), receipt.get("planned_tracks"), receipt.get("planned_endpoints")) != (160, 800, 54):
        raise RenderError("independent_receipt_coverage_mismatch")

    manifest_path = geometry_root / "geometry_family_manifest.json"
    inventory_path = geometry_root / "all_files_sha256.json"
    seal_path = geometry_root / "geometry_family_seal.json"
    if {item.name for item in geometry_root.iterdir()} != EXPECTED_FILES:
        raise RenderError("geometry_output_exact_fileset_mismatch")
    manifest = _read_json(manifest_path)
    manifest_payload = _embedded_sha(manifest, "manifest_sha256", "geometry_manifest")
    manifest_raw = _sha_file(manifest_path)
    if receipt.get("geometry_manifest_raw_sha256") != manifest_raw or receipt.get("geometry_manifest_payload_sha256") != manifest_payload:
        raise RenderError("independent_receipt_geometry_binding_mismatch")
    if receipt.get("geometry_inventory_raw_sha256") != _sha_file(inventory_path):
        raise RenderError("independent_receipt_inventory_binding_mismatch")
    if receipt.get("geometry_seal_raw_sha256") != _sha_file(seal_path):
        raise RenderError("independent_receipt_seal_binding_mismatch")

    if manifest.get("schema") != "world_shape_visual_geometry_family_v1":
        raise RenderError("geometry_manifest_schema_mismatch")
    if manifest.get("status") not in ("COMPLETE", "COMPLETE_WITH_INCOMPLETE_ENDPOINTS"):
        raise RenderError("geometry_manifest_status_not_complete")
    if manifest.get("tracking_plan") != {
        "path": manifest.get("tracking_plan", {}).get("path"),
        "raw_sha256": PLAN_RAW_SHA256,
        "payload_sha256": PLAN_PAYLOAD_SHA256,
    }:
        raise RenderError("tracking_plan_binding_mismatch")
    for field, expected in (("visual_design", DESIGN_SHA256), ("visual_design_pin_correction", PIN_CORRECTION_SHA256)):
        value = manifest.get(field)
        if not isinstance(value, dict) or value.get("raw_sha256") != expected:
            raise RenderError(f"geometry_source_binding_mismatch:{field}")
    for field, expected in (("visual_core_source", VISUAL_CORE_SHA256), ("tone_adapter_source", TONE_ADAPTER_SHA256)):
        value = manifest.get(field)
        if not isinstance(value, dict) or value.get("raw_sha256") != expected:
            raise RenderError(f"geometry_source_binding_mismatch:{field}")
    release = manifest.get("geometry_release")
    if (not isinstance(release, dict)
            or release.get("raw_sha256") != GEOMETRY_RELEASE_RAW_SHA256
            or release.get("payload_sha256") != GEOMETRY_RELEASE_PAYLOAD_SHA256):
        raise RenderError("geometry_release_binding_mismatch")
    if (manifest.get("mouth_geometry_read") is not True
            or manifest.get("evaluation_scores_read") is not False
            or manifest.get("evaluation_outcomes_read") is not False
            or manifest.get("evaluation_effects_read") is not False
            or manifest.get("model_forward_performed") is not False):
        raise RenderError("geometry_access_scope_mismatch")
    if (manifest.get("coverage", {}).get("planned_endpoints") != 54
            or manifest.get("coverage", {}).get("complete_contexts") != 160
            or manifest.get("coverage", {}).get("complete_video_tracks") != 800):
        raise RenderError("geometry_coverage_mismatch")
    endpoint_design = manifest.get("endpoint_design", {})
    if (endpoint_design.get("directions") != list(DIRECTIONS)
            or endpoint_design.get("metrics") != list(METRICS)
            or endpoint_design.get("contrasts") != list(CONTRASTS)
            or endpoint_design.get("endpoint_order") != "direction-major, metric-major, contrast-minor"
            or endpoint_design.get("endpoint_count") != 54):
        raise RenderError("endpoint_design_mismatch")

    inventory = _read_json(inventory_path)
    inventory_files = inventory.get("files")
    expected_inventory_names = EXPECTED_FILES - {"all_files_sha256.json", "geometry_family_seal.json"}
    if inventory.get("schema") != "world_shape_visual_geometry_files_v1" or not isinstance(inventory_files, dict) or set(inventory_files) != expected_inventory_names:
        raise RenderError("geometry_inventory_schema_or_fileset_mismatch")
    for name, digest in inventory_files.items():
        _sha(digest, f"inventory:{name}")
        path = geometry_root / name
        if not path.is_file() or path.is_symlink() or _sha_file(path) != digest:
            raise RenderError(f"geometry_inventory_hash_mismatch:{name}")
    seal = _read_json(seal_path)
    if (seal.get("status") != manifest.get("status")
            or seal.get("files") != len(inventory_files)
            or seal.get("manifest_raw_sha256") != manifest_raw
            or seal.get("manifest_payload_sha256") != manifest_payload
            or seal.get("all_files_sha256") != _sha_file(inventory_path)
            or seal.get("contexts") != 160 or seal.get("videos") != 800 or seal.get("endpoints") != 54
            or seal.get("evaluation_effects_read") is not False or seal.get("model_forward_performed") is not False):
        raise RenderError("geometry_seal_contract_mismatch")
    if receipt.get("verified_input_source_hashes", {}).get("visual_design", {}).get("raw_sha256") != DESIGN_SHA256:
        raise RenderError("independent_receipt_design_binding_mismatch")
    if receipt.get("verified_input_source_hashes", {}).get("visual_core_source", {}).get("raw_sha256") != VISUAL_CORE_SHA256:
        raise RenderError("independent_receipt_visual_core_binding_mismatch")
    if receipt.get("verified_input_source_hashes", {}).get("tone_adapter_source", {}).get("raw_sha256") != TONE_ADAPTER_SHA256:
        raise RenderError("independent_receipt_tone_adapter_binding_mismatch")
    return geometry_root, manifest, receipt, inventory, manifest_raw


def _endpoint_rows(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    rows = manifest.get("endpoint_rows")
    if not isinstance(rows, list) or len(rows) != ENDPOINT_COUNT:
        raise RenderError("endpoint_rows_not_exactly_54")
    expected_keys = [(direction, metric, contrast) for direction in DIRECTIONS for metric in METRICS for contrast in CONTRASTS]
    if [(row.get("direction"), row.get("metric"), row.get("contrast")) for row in rows] != expected_keys:
        raise RenderError("endpoint_order_mismatch")
    critical_values: set[float] = set()
    phone_counts = manifest.get("coverage", {}).get("phone_eligible_speakers_by_direction")
    if not isinstance(phone_counts, dict) or set(phone_counts) != set(DIRECTIONS):
        raise RenderError("frozen_phone_eligibility_counts_missing")
    for direction in DIRECTIONS:
        if not isinstance(phone_counts[direction], int) or not 0 < phone_counts[direction] <= SPEAKER_COUNT:
            raise RenderError(f"invalid_frozen_phone_eligibility_count:{direction}")
    for index, row in enumerate(rows):
        if row.get("endpoint_index") != index or row.get("endpoint_key") != "::".join(expected_keys[index]):
            raise RenderError(f"endpoint_identity_mismatch:{index}")
        metric = str(row["metric"])
        count = row.get("availability_speaker_count")
        expected_mask = "frozen_phone_complete_pair" if metric == "phone_class_separation" else "all_40_speakers"
        if row.get("estimand_mask") != expected_mask or not isinstance(count, int) or not 0 <= count <= SPEAKER_COUNT:
            raise RenderError(f"endpoint_mask_or_count_invalid:{index}")
        if metric != "phone_class_separation" and count != SPEAKER_COUNT:
            raise RenderError(f"overall_endpoint_lost_speakers:{index}")
        if metric == "phone_class_separation" and count != phone_counts[row["direction"]]:
            raise RenderError(f"phone_endpoint_frozen_count_mismatch:{index}")
        critical = row.get("critical_value")
        if not isinstance(critical, (int, float)) or not math.isfinite(float(critical)):
            raise RenderError(f"endpoint_critical_value_invalid:{index}")
        critical_values.add(float(critical))
        status = str(row.get("status", ""))
        point, se, interval = row.get("point_estimate"), row.get("bootstrap_se_sample_sd_ddof1"), row.get("simultaneous_99_ci")
        if status.startswith("ESTIMABLE"):
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (point, se)):
                raise RenderError(f"estimable_endpoint_missing_point_or_se:{index}")
            if not isinstance(interval, list) or len(interval) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in interval):
                raise RenderError(f"estimable_endpoint_missing_interval:{index}")
            if float(interval[0]) > float(point) or float(point) > float(interval[1]) or float(se) < 0:
                raise RenderError(f"estimable_endpoint_invalid_interval:{index}")
        elif status.startswith("INCOMPLETE"):
            if point is not None or se is not None or interval is not None:
                raise RenderError(f"incomplete_endpoint_has_estimate:{index}")
        else:
            raise RenderError(f"endpoint_status_unrecognized:{index}:{status}")
    if len(critical_values) != 1:
        raise RenderError("endpoint_rows_do_not_share_one_family_critical_value")
    critical = next(iter(critical_values))
    bootstrap = manifest.get("bootstrap", {})
    if bootstrap.get("seed") != 20260930 or bootstrap.get("draws") != 20000 or bootstrap.get("draw_size") != SPEAKER_COUNT:
        raise RenderError("visual_bootstrap_contract_mismatch")

    return rows, critical


def _validate_speaker_arrays(geometry_root: Path, rows: list[dict[str, Any]], critical: float, np: Any) -> None:
    expected = {
        "endpoint_keys_unicode", "contrast_names_unicode", "directions_unicode", "metric_names_unicode",
        "speaker_ids_unicode", "speaker_values_float64", "availability_mask_bool", "point_estimates_float64",
        "bootstrap_se_float64", "ci_low_float64", "ci_high_float64", "endpoint_status_unicode",
        "critical_value_float64", "deterministic_zero_se_bool",
    }
    with np.load(geometry_root / "speaker_endpoints.npz", allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RenderError("speaker_endpoint_npz_keyset_mismatch")
        if archive["speaker_values_float64"].shape != (54, 40) or archive["availability_mask_bool"].shape != (54, 40):
            raise RenderError("speaker_endpoint_npz_shape_mismatch")
        keys = [f"{row['direction']}::{row['metric']}::{row['contrast']}" for row in rows]
        if archive["endpoint_keys_unicode"].astype(str).tolist() != keys:
            raise RenderError("speaker_endpoint_npz_order_mismatch")
        if not np.allclose(archive["critical_value_float64"], np.asarray([critical]), rtol=0.0, atol=0.0):
            raise RenderError("speaker_endpoint_npz_critical_value_mismatch")
        if archive["speaker_ids_unicode"].shape != (40,):
            raise RenderError("speaker_endpoint_npz_roster_shape_mismatch")
        for index, row in enumerate(rows):
            status = str(row["status"])
            point, se, ci = row["point_estimate"], row["bootstrap_se_sample_sd_ddof1"], row["simultaneous_99_ci"]
            expected_status = str(archive["endpoint_status_unicode"][index])
            if expected_status != status:
                raise RenderError(f"speaker_endpoint_npz_status_mismatch:{index}")
            expected_count = int(archive["availability_mask_bool"][index].sum())
            if expected_count != row["availability_speaker_count"]:
                raise RenderError(f"speaker_endpoint_npz_mask_count_mismatch:{index}")
            if bool(archive["deterministic_zero_se_bool"][index]) is not bool(row["zero_se_deterministic"]):
                raise RenderError(f"speaker_endpoint_npz_zero_se_mismatch:{index}")
            for key, value in (("point_estimates_float64", point), ("bootstrap_se_float64", se)):
                actual = float(archive[key][index])
                if value is None:
                    if math.isfinite(actual):
                        raise RenderError(f"speaker_endpoint_npz_expected_nan:{key}:{index}")
                elif not math.isclose(actual, float(value), rel_tol=0.0, abs_tol=0.0):
                    raise RenderError(f"speaker_endpoint_npz_value_mismatch:{key}:{index}")
            for key, value in (("ci_low_float64", None if ci is None else ci[0]), ("ci_high_float64", None if ci is None else ci[1])):
                actual = float(archive[key][index])
                if value is None:
                    if math.isfinite(actual):
                        raise RenderError(f"speaker_endpoint_npz_expected_nan:{key}:{index}")
                elif not math.isclose(actual, float(value), rel_tol=0.0, abs_tol=0.0):
                    raise RenderError(f"speaker_endpoint_npz_value_mismatch:{key}:{index}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "endpoint_index", "endpoint_key", "metric", "unit", "direction", "direction_label",
        "contrast", "estimand_mask", "eligible_speakers", "planned_speakers", "status",
        "point_estimate", "bootstrap_se_sample_sd_ddof1", "ci99_low_simultaneous", "ci99_high_simultaneous",
        "critical_value_shared_family", "zero_se_deterministic",
    ]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            interval = row["simultaneous_99_ci"]
            writer.writerow({
                "endpoint_index": row["endpoint_index"], "endpoint_key": row["endpoint_key"],
                "metric": row["metric"], "unit": UNIT[row["metric"]], "direction": row["direction"],
                "direction_label": DIRECTION_LABEL[row["direction"]], "contrast": row["contrast"],
                "estimand_mask": row["estimand_mask"], "eligible_speakers": row["availability_speaker_count"],
                "planned_speakers": SPEAKER_COUNT, "status": row["status"],
                "point_estimate": "" if row["point_estimate"] is None else repr(float(row["point_estimate"])),
                "bootstrap_se_sample_sd_ddof1": "" if row["bootstrap_se_sample_sd_ddof1"] is None else repr(float(row["bootstrap_se_sample_sd_ddof1"])),
                "ci99_low_simultaneous": "" if interval is None else repr(float(interval[0])),
                "ci99_high_simultaneous": "" if interval is None else repr(float(interval[1])),
                "critical_value_shared_family": repr(float(row["critical_value"])),
                "zero_se_deterministic": row["zero_se_deterministic"],
            })
        stream.flush()
        os.fsync(stream.fileno())


def _draw_metric(metric: str, rows: list[dict[str, Any]], critical: float, output_dir: Path, np: Any, plt: Any) -> dict[str, Any]:
    page_rows = [row for row in rows if row["metric"] == metric]
    if len(page_rows) != len(DIRECTIONS) * len(CONTRASTS):
        raise RenderError(f"metric_page_row_count_mismatch:{metric}")
    by_key = {(row["direction"], row["contrast"]): row for row in page_rows}
    if len(by_key) != len(page_rows):
        raise RenderError(f"duplicate_metric_endpoint:{metric}")
    finite_bounds = [float(value) for row in page_rows if row["simultaneous_99_ci"] is not None for value in row["simultaneous_99_ci"]]
    if finite_bounds:
        extent = max(abs(value) for value in finite_bounds)
        margin = max(extent * 0.12, 1e-6)
        xlim = (-extent - margin, extent + margin)
    else:
        xlim = (-1.0, 1.0)

    fig, ax = plt.subplots(figsize=(18.0, 11.5), constrained_layout=False)
    y_positions = []
    y_labels = []
    for ci, contrast in enumerate(CONTRASTS):
        for di, direction in enumerate(DIRECTIONS):
            row = by_key[(direction, contrast)]
            y = ci * 2 + di
            y_positions.append(y)
            n = int(row["availability_speaker_count"])
            y_labels.append(f"{contrast}  ·  {DIRECTION_LABEL[direction]}  ·  n={n}/40")
            point = row["point_estimate"]
            interval = row["simultaneous_99_ci"]
            color = DIRECTION_COLOR[direction]
            if point is None or interval is None:
                ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.015, y, "N/A", color="#59636f", fontsize=8,
                        va="center", ha="left")
            else:
                point, low, high = float(point), float(interval[0]), float(interval[1])
                ax.errorbar(
                    point, y, xerr=np.asarray([[point - low], [high - point]]),
                    fmt=DIRECTION_MARKER[direction], color=color, ecolor=color,
                    elinewidth=1.15, capsize=2.6, markersize=5.1, markeredgewidth=0.8,
                    markerfacecolor=color, zorder=3,
                )
    ax.axvline(0.0, color="#27313d", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlim(*xlim)
    ax.set_yticks(y_positions, y_labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel(UNIT[metric], fontsize=10)
    ax.set_title(
        f"{METRIC_TITLE[metric]} — 18 endpoints in the frozen 54-endpoint family\n"
        f"shared two-sided 99% simultaneous max-|t| intervals; family critical value {critical:.5g}",
        loc="left", fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(axis="x", color="#dce1e7", linewidth=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.legend(
        handles=[
            LINE2D([0], [0], color=DIRECTION_COLOR[direction], marker=DIRECTION_MARKER[direction],
                   linestyle="none", label=DIRECTION_LABEL[direction])
            for direction in DIRECTIONS
        ], loc="lower right", frameon=False, ncol=2,
    )
    caption = (
        "Aperture is normalized by inter-eye distance (dimensionless); velocity is per second. "
        "Phone-class separation uses the fixed −85 ms label alignment and each direction's frozen complete-pair speaker mask. "
        "Overall aperture endpoints use all 40 speakers. Incomplete planned endpoints remain marked N/A; no row is omitted. "
        "Intervals use the single frozen 54-endpoint family and shared 20,000-draw speaker bootstrap."
    )
    fig.text(0.07, 0.018, caption, fontsize=8.0, color="#4c5662", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.48, right=0.985, top=0.90, bottom=0.105)
    png = output_dir / f"visual_geometry_{metric}.png"
    fig.savefig(png, dpi=100, facecolor="white", metadata={"Title": METRIC_TITLE[metric]})
    plt.close(fig)
    return {"path": png.name, "width_px": 1800, "height_px": 1150, "metric": metric, "rows": len(page_rows), "xlim": list(xlim)}


def render(args: argparse.Namespace) -> dict[str, Any]:
    np, plt, runtime = _load_plotting(args)
    geometry_root, manifest, independent_receipt, inventory, manifest_raw = _validate_inputs(args)
    rows, critical = _endpoint_rows(manifest)
    _validate_speaker_arrays(geometry_root, rows, critical, np)
    output_dir = args.output_dir.resolve(strict=False)
    if output_dir.exists():
        raise RenderError("render_output_directory_must_be_absent")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "visual_geometry_endpoints_all54.csv", rows)
    pages = {metric: _draw_metric(metric, rows, critical, output_dir, np, plt) for metric in METRICS}
    pdf_path = output_dir / "visual_geometry_family_all54.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path, metadata={"Title": "WORLD visual geometry frozen 54-endpoint family", "Author": "TTS-TFG analysis"}) as pdf:
        for metric in METRICS:
            # Recreate each forest deterministically for the vector PDF page.
            page_rows = [row for row in rows if row["metric"] == metric]
            by_key = {(row["direction"], row["contrast"]): row for row in page_rows}
            fig, ax = plt.subplots(figsize=(11.7, 8.3), constrained_layout=False)
            extent_values = [float(v) for row in page_rows if row["simultaneous_99_ci"] is not None for v in row["simultaneous_99_ci"]]
            extent = max((abs(v) for v in extent_values), default=1.0)
            margin = max(extent * 0.12, 1e-6)
            xlim = (-extent - margin, extent + margin)
            positions, labels = [], []
            for ci, contrast in enumerate(CONTRASTS):
                for di, direction in enumerate(DIRECTIONS):
                    row = by_key[(direction, contrast)]
                    y = ci * 2 + di
                    positions.append(y)
                    labels.append(f"{contrast} · {DIRECTION_LABEL[direction]} · n={row['availability_speaker_count']}/40")
                    if row["point_estimate"] is None:
                        ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.015, y, "N/A", color="#59636f", fontsize=7, va="center")
                    else:
                        point = float(row["point_estimate"])
                        low, high = map(float, row["simultaneous_99_ci"])
                        ax.errorbar(point, y, xerr=np.asarray([[point - low], [high - point]]),
                                    fmt=DIRECTION_MARKER[direction], color=DIRECTION_COLOR[direction],
                                    ecolor=DIRECTION_COLOR[direction], elinewidth=0.95, capsize=2.2, markersize=4.0)
            ax.axvline(0.0, color="#27313d", linewidth=0.8, linestyle="--")
            ax.set_xlim(*xlim)
            ax.set_yticks(positions, labels, fontsize=6.2)
            ax.invert_yaxis()
            ax.set_xlabel(UNIT[metric], fontsize=8.5)
            ax.set_title(f"{METRIC_TITLE[metric]} — 18 endpoints in the frozen 54-endpoint family\nTwo-sided 99% simultaneous max-|t|; critical={critical:.5g}",
                         loc="left", fontsize=10.5, fontweight="bold")
            ax.grid(axis="x", color="#dce1e7", linewidth=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=5)
            ax.legend(handles=[LINE2D([0], [0], color=DIRECTION_COLOR[d], marker=DIRECTION_MARKER[d], linestyle="none", label=DIRECTION_LABEL[d]) for d in DIRECTIONS],
                      loc="lower right", frameon=False, ncol=2, fontsize=7)
            fig.text(0.06, 0.012,
                     "Aperture is inter-eye normalized (dimensionless); velocity uses s⁻¹. Phone separation uses the frozen −85 ms label alignment and pre-geometry complete-pair mask. Incomplete planned endpoints stay N/A.",
                     fontsize=6.8, color="#4c5662", va="bottom", wrap=True)
            fig.subplots_adjust(left=0.40, right=0.985, top=0.90, bottom=0.10)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

    csv_path = output_dir / "visual_geometry_endpoints_all54.csv"
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise RenderError("render_output_contains_nonregular_member")
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    renderer_path = Path(__file__).resolve(strict=True)
    payload = {
        "schema": "world_shape_visual_geometry_render_v1",
        "status": "RENDERED_ALL_54_PLANNED_ENDPOINTS",
        "renderer_source": {"path": str(renderer_path), "raw_sha256": _sha_file(renderer_path)},
        "primary_source_sha256": PRIMARY_SOURCE_SHA256,
        "independent_source_sha256": INDEPENDENT_SOURCE_SHA256,
        "input": {
            "geometry_output_root": str(geometry_root),
            "geometry_manifest_raw_sha256": manifest_raw,
            "geometry_manifest_payload_sha256": manifest["manifest_sha256"],
            "geometry_inventory_raw_sha256": _sha_file(geometry_root / "all_files_sha256.json"),
            "geometry_seal_raw_sha256": _sha_file(geometry_root / "geometry_family_seal.json"),
            "geometry_release_raw_sha256": GEOMETRY_RELEASE_RAW_SHA256,
            "geometry_release_payload_sha256": GEOMETRY_RELEASE_PAYLOAD_SHA256,
            "independent_receipt_path": str(args.independent_receipt.resolve(strict=True)),
            "independent_receipt_raw_sha256": _sha_file(args.independent_receipt.resolve(strict=True)),
            "independent_receipt_status": independent_receipt["status"],
        },
        "design": {
            "planned_endpoints": 54, "rendered_endpoints": len(rows), "directions": list(DIRECTIONS),
            "metrics": list(METRICS), "contrasts": list(CONTRASTS),
            "endpoint_order": "direction-major, metric-major, contrast-minor",
            "units": UNIT, "shared_99_max_t_critical_value": critical,
            "bootstrap_draws": 20000, "bootstrap_seed": 20260930, "speaker_roster": 40,
            "phone_eligible_speakers_by_direction": manifest["coverage"]["phone_eligible_speakers_by_direction"],
            "incomplete_endpoints_retained": sum(str(row["status"]).startswith("INCOMPLETE") for row in rows),
            "planned_speakers": 40,
            "phone_mask_note": "Phone-class endpoints use the frozen complete-pair mask per direction; CSV reports eligible count out of the 40-speaker roster.",
            "no_significance_count_or_strength_ranking": True,
        },
        "runtime": runtime,
        "figures": {"pdf": {"path": pdf_path.name, "pages": 3}, "pngs": pages},
        "csv": {"path": csv_path.name, "rows": len(rows)},
        "output_files": artifacts,
    }
    payload["manifest_sha256"] = _sha_bytes(_canonical(payload))
    manifest_path = output_dir / "visual_geometry_render_manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": payload["status"], "manifest_path": str(manifest_path), "manifest_sha256": _sha_file(manifest_path), "payload_sha256": payload["manifest_sha256"], "endpoint_count": len(rows), "output_files": len(artifacts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--expected-independent-receipt-sha256", required=True)
    parser.add_argument("--primary-source", type=Path, required=True)
    parser.add_argument("--plotdeps-root", type=Path, required=True)
    parser.add_argument("--plotdeps-runtime-manifest", type=Path, required=True)
    parser.add_argument("--plotdeps-runtime-readback", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = render(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
