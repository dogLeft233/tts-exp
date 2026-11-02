#!/usr/bin/env python3
"""Independent readback of the 54-endpoint visual-geometry render.

The verifier uses only the sealed primary family manifest, the independent
reconstruction receipt, CSV, PDF, PNG headers, and file hashes. It does not
import Matplotlib or recompute scientific estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any


PRIMARY_SOURCE_SHA256 = "ae9daa7191c401e8c86852bb1bae4666fd1739e897a19c4496b980ce1a52a568"
INDEPENDENT_SOURCE_SHA256 = "84f9056e09579c636180f3365e8dd9a7021faff8da1375e0e4dc89290d021cc6"
GEOMETRY_RELEASE_RAW_SHA256 = "4a8a6265de17566ee237e2ed79f034f3df427fd186daa64b345f98253d2005ae"
GEOMETRY_RELEASE_PAYLOAD_SHA256 = "6eb84aed937a227012627533205720bd145b7aeb63d0a457d7b024f471bd542f"
PLOTDEPS_RUNTIME_MANIFEST_SHA256 = "bf5586ad38ada97add6fb24aa125901d5a80b8bacc5b246482ac9a98975789f4"
PLOTDEPS_RUNTIME_READBACK_SHA256 = "4e420047f9cf4d0ee99e74cf799db336708fcb2b4ab3ea036b0f22567a1e6abe"
PLOTDEPS_TREE_SHA256 = "fe25379439c3401b54309d5b18fdfc388380ef9e568bd7ce6ed1e3d6aaa6df83"
METRICS = (
    "mean_aperture",
    "mean_absolute_aperture_velocity_per_s",
    "phone_class_separation",
)
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
CONTRASTS = (
    "mean_q10_minus_q00", "trajectory_q01_minus_q00", "joint_q11_minus_q00",
    "grid_cycle_SELF_GRID_minus_q00", "mean_minus_trajectory_q10_minus_q01",
    "mean_beyond_grid_q10_minus_SELF_GRID", "trajectory_beyond_grid_q01_minus_SELF_GRID",
    "joint_beyond_grid_q11_minus_SELF_GRID", "factorial_interaction_q11_minus_q10_minus_q01_plus_q00",
)
PNG_NAMES = tuple(f"visual_geometry_{metric}.png" for metric in METRICS)
CSV_NAME = "visual_geometry_endpoints_all54.csv"
PDF_NAME = "visual_geometry_family_all54.pdf"
RENDER_MANIFEST_NAME = "visual_geometry_render_manifest.json"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PDF_PAGE_RE = re.compile(rb"/Type\s*/Page\b")
UNITS = {
    "mean_aperture": "eye-distance-normalized aperture (dimensionless)",
    "mean_absolute_aperture_velocity_per_s": "normalized aperture change per second (s⁻¹)",
    "phone_class_separation": "open-vowel minus bilabial normalized aperture (dimensionless)",
}
DIRECTION_LABEL = {"tts_to_natural": "T→N", "natural_to_tts": "N→T"}


class ReadbackError(RuntimeError):
    pass


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


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReadbackError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ReadbackError(f"json_not_object:{path.name}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ReadbackError(f"invalid_sha256:{label}")
    return value


def _float_csv(value: str, expected: Any, label: str) -> None:
    if expected is None:
        if value != "":
            raise ReadbackError(f"incomplete_value_present:{label}")
        return
    try:
        actual = float(value)
    except Exception as exc:
        raise ReadbackError(f"invalid_csv_float:{label}") from exc
    if not math.isfinite(actual) or actual != float(expected):
        raise ReadbackError(f"csv_float_mismatch:{label}")


def _check_geometry_inputs(geometry_root: Path, geometry_raw_expected: str, geometry_payload_expected: str) -> dict[str, Any]:
    manifest_path = geometry_root / "geometry_family_manifest.json"
    if _sha_file(manifest_path) != geometry_raw_expected:
        raise ReadbackError("geometry_manifest_raw_sha_mismatch")
    manifest = _json(manifest_path)
    payload = _sha_bytes(_canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"}))
    if payload != geometry_payload_expected or manifest.get("manifest_sha256") != payload:
        raise ReadbackError("geometry_manifest_payload_sha_mismatch")
    if manifest.get("schema") != "world_shape_visual_geometry_family_v1":
        raise ReadbackError("geometry_manifest_schema_mismatch")
    return manifest


def verify(args: argparse.Namespace) -> dict[str, Any]:
    geometry_root = args.geometry_dir.resolve(strict=True)
    receipt_path = args.independent_receipt.resolve(strict=True)
    render_root = args.render_dir.resolve(strict=True)
    renderer_source = args.renderer_source.resolve(strict=True)
    independent_source = args.independent_source.resolve(strict=True)
    receipt_output = args.receipt_path.resolve(strict=False)
    try:
        receipt_output.relative_to(render_root)
    except ValueError:
        pass
    else:
        raise ReadbackError("readback_receipt_must_be_outside_render_tree")
    expected_renderer = _sha(args.expected_renderer_sha256, "renderer_source")
    expected_independent = _sha(args.expected_independent_receipt_sha256, "independent_receipt")
    expected_receipt_source = _sha(args.expected_independent_source_sha256, "independent_source")
    geometry_raw = _sha(args.geometry_manifest_raw_sha256, "geometry_manifest_raw")
    geometry_payload = _sha(args.geometry_manifest_payload_sha256, "geometry_manifest_payload")
    if _sha_file(renderer_source) != expected_renderer:
        raise ReadbackError("renderer_source_sha_mismatch")
    if _sha_file(independent_source) != INDEPENDENT_SOURCE_SHA256 or expected_receipt_source != INDEPENDENT_SOURCE_SHA256:
        raise ReadbackError("independent_source_sha_mismatch")
    if _sha_file(receipt_path) != expected_independent:
        raise ReadbackError("independent_receipt_sha_mismatch")
    receipt = _json(receipt_path)
    if receipt.get("status") != "INDEPENDENT_RECONSTRUCTION_PASS" or receipt.get("independent_source_sha256") != INDEPENDENT_SOURCE_SHA256:
        raise ReadbackError("independent_reconstruction_not_qualified")
    if (receipt.get("planned_contexts"), receipt.get("planned_tracks"), receipt.get("planned_endpoints")) != (160, 800, 54):
        raise ReadbackError("independent_reconstruction_coverage_mismatch")
    geometry = _check_geometry_inputs(geometry_root, geometry_raw, geometry_payload)
    release = geometry.get("geometry_release")
    if (not isinstance(release, dict)
            or release.get("raw_sha256") != GEOMETRY_RELEASE_RAW_SHA256
            or release.get("payload_sha256") != GEOMETRY_RELEASE_PAYLOAD_SHA256):
        raise ReadbackError("geometry_release_binding_mismatch")
    if receipt.get("geometry_manifest_raw_sha256") != geometry_raw or receipt.get("geometry_manifest_payload_sha256") != geometry_payload:
        raise ReadbackError("independent_receipt_not_bound_to_geometry_manifest")

    render_manifest_path = render_root / RENDER_MANIFEST_NAME
    if render_manifest_path.is_symlink():
        raise ReadbackError("render_manifest_must_be_regular_file")
    render_manifest = _json(render_manifest_path)
    render_payload = _sha_bytes(_canonical({key: value for key, value in render_manifest.items() if key != "manifest_sha256"}))
    if render_manifest.get("manifest_sha256") != render_payload:
        raise ReadbackError("render_manifest_payload_mismatch")
    if render_manifest.get("schema") != "world_shape_visual_geometry_render_v1" or render_manifest.get("status") != "RENDERED_ALL_54_PLANNED_ENDPOINTS":
        raise ReadbackError("render_manifest_status_or_schema_mismatch")
    if render_manifest.get("renderer_source", {}).get("raw_sha256") != expected_renderer:
        raise ReadbackError("render_manifest_source_binding_mismatch")
    recorded_renderer_path = render_manifest.get("renderer_source", {}).get("path")
    if not isinstance(recorded_renderer_path, str) or Path(recorded_renderer_path).resolve(strict=True) != renderer_source:
        raise ReadbackError("render_manifest_source_path_mismatch")
    if (render_manifest.get("primary_source_sha256") != PRIMARY_SOURCE_SHA256
            or render_manifest.get("independent_source_sha256") != INDEPENDENT_SOURCE_SHA256
            or receipt.get("primary_source_sha256") != PRIMARY_SOURCE_SHA256):
        raise ReadbackError("render_manifest_science_source_binding_mismatch")
    input_fields = render_manifest.get("input", {})
    if (input_fields.get("geometry_manifest_raw_sha256") != geometry_raw
            or input_fields.get("geometry_manifest_payload_sha256") != geometry_payload
            or input_fields.get("geometry_output_root") != str(geometry_root)
            or input_fields.get("geometry_release_raw_sha256") != GEOMETRY_RELEASE_RAW_SHA256
            or input_fields.get("geometry_release_payload_sha256") != GEOMETRY_RELEASE_PAYLOAD_SHA256
            or input_fields.get("independent_receipt_raw_sha256") != expected_independent
            or input_fields.get("independent_receipt_path") != str(receipt_path)
            or input_fields.get("independent_receipt_status") != "INDEPENDENT_RECONSTRUCTION_PASS"):
        raise ReadbackError("render_manifest_input_binding_mismatch")
    expected_names = {CSV_NAME, PDF_NAME, *PNG_NAMES, RENDER_MANIFEST_NAME}
    if {path.name for path in render_root.iterdir()} != expected_names:
        raise ReadbackError("render_output_exact_fileset_mismatch")
    output_entries = render_manifest.get("output_files")
    expected_artifact_names = expected_names - {RENDER_MANIFEST_NAME}
    if (not isinstance(output_entries, list) or len(output_entries) != len(expected_artifact_names)
            or any(not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}
                   or not isinstance(entry.get("path"), str) for entry in output_entries)
            or {entry["path"] for entry in output_entries} != expected_artifact_names):
        raise ReadbackError("render_manifest_output_fileset_mismatch")
    for entry in output_entries:
        name = entry["path"]
        _sha(entry.get("sha256"), f"render_output:{name}")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            raise ReadbackError(f"render_artifact_size_invalid:{name}")
        path = render_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry.get("bytes") or _sha_file(path) != entry.get("sha256"):
            raise ReadbackError(f"render_artifact_hash_or_size_mismatch:{name}")

    endpoints = geometry.get("endpoint_rows")
    expected_order = [(direction, metric, contrast) for direction in DIRECTIONS for metric in METRICS for contrast in CONTRASTS]
    if not isinstance(endpoints, list) or len(endpoints) != 54:
        raise ReadbackError("geometry_endpoint_coverage_invalid")
    observed_order = [(row.get("direction"), row.get("metric"), row.get("contrast")) for row in endpoints]
    if observed_order != expected_order:
        raise ReadbackError("geometry_endpoint_order_invalid")
    expected_csv_fields = [
        "endpoint_index", "endpoint_key", "metric", "unit", "direction", "direction_label",
        "contrast", "estimand_mask", "eligible_speakers", "planned_speakers", "status",
        "point_estimate", "bootstrap_se_sample_sd_ddof1", "ci99_low_simultaneous", "ci99_high_simultaneous",
        "critical_value_shared_family", "zero_se_deterministic",
    ]
    with (render_root / CSV_NAME).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_csv_fields:
            raise ReadbackError("render_csv_header_mismatch")
        csv_rows = list(reader)
    if len(csv_rows) != 54:
        raise ReadbackError("render_csv_row_count_not_54")
    checked = 0
    for index, (source, row, expected_key) in enumerate(zip(endpoints, csv_rows, expected_order, strict=True)):
        direction, metric, contrast = expected_key
        if (row.get("endpoint_index") != str(index)
                or row.get("endpoint_key") != f"{direction}::{metric}::{contrast}"
                or row.get("direction") != direction or row.get("metric") != metric or row.get("contrast") != contrast
                or row.get("direction_label") != DIRECTION_LABEL[direction] or row.get("unit") != UNITS[metric]
                or row.get("estimand_mask") != source.get("estimand_mask")
                or row.get("status") != source.get("status")
                or row.get("eligible_speakers") != str(source.get("availability_speaker_count"))
                or row.get("planned_speakers") != "40"):
            raise ReadbackError(f"render_csv_identity_or_support_mismatch:{index}")
        if row.get("zero_se_deterministic") != str(source.get("zero_se_deterministic")):
            raise ReadbackError(f"render_csv_zero_se_mismatch:{index}")
        for field, expected in (
            ("point_estimate", source.get("point_estimate")),
            ("bootstrap_se_sample_sd_ddof1", source.get("bootstrap_se_sample_sd_ddof1")),
            ("ci99_low_simultaneous", None if source.get("simultaneous_99_ci") is None else source["simultaneous_99_ci"][0]),
            ("ci99_high_simultaneous", None if source.get("simultaneous_99_ci") is None else source["simultaneous_99_ci"][1]),
            ("critical_value_shared_family", source.get("critical_value")),
        ):
            _float_csv(row.get(field, ""), expected, f"{field}:{index}")
        checked += 1

    pdf_bytes = (render_root / PDF_NAME).read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or len(PDF_PAGE_RE.findall(pdf_bytes)) != 3:
        raise ReadbackError("render_pdf_not_three_pages")
    png_dimensions = {}
    for name in PNG_NAMES:
        raw = (render_root / name).read_bytes()
        if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
            raise ReadbackError(f"invalid_png:{name}")
        width, height = struct.unpack(">II", raw[16:24])
        if width != 1800 or height != 1150:
            raise ReadbackError(f"png_dimensions_invalid:{name}")
        png_dimensions[name] = {"width_px": width, "height_px": height}
    design = render_manifest.get("design", {})
    if (design.get("planned_endpoints") != 54 or design.get("rendered_endpoints") != 54
            or design.get("directions") != list(DIRECTIONS) or design.get("metrics") != list(METRICS)
            or design.get("contrasts") != list(CONTRASTS) or design.get("no_significance_count_or_strength_ranking") is not True
            or design.get("bootstrap_draws") != 20000 or design.get("bootstrap_seed") != 20260930
            or design.get("speaker_roster") != 40 or design.get("planned_speakers") != 40
            or design.get("shared_99_max_t_critical_value") != float(endpoints[0]["critical_value"])
            or design.get("incomplete_endpoints_retained") != sum(str(row.get("status", "")).startswith("INCOMPLETE") for row in endpoints)):
        raise ReadbackError("render_manifest_design_coverage_mismatch")
    runtime = render_manifest.get("runtime", {})
    if (runtime.get("python") != "3.12.14" or runtime.get("numpy") != "2.3.5"
            or runtime.get("matplotlib") != "3.11.2" or runtime.get("backend") != "Agg"
            or runtime.get("runtime_manifest_raw_sha256") != PLOTDEPS_RUNTIME_MANIFEST_SHA256
            or runtime.get("runtime_readback_raw_sha256") != PLOTDEPS_RUNTIME_READBACK_SHA256
            or runtime.get("plotdeps_tree_sha256") != PLOTDEPS_TREE_SHA256
            or runtime.get("plotdeps_file_count") != 1712 or runtime.get("plotdeps_total_bytes") != 53345343):
        raise ReadbackError("render_manifest_runtime_binding_mismatch")
    figures = render_manifest.get("figures", {})
    if figures.get("pdf", {}).get("path") != PDF_NAME or figures.get("pdf", {}).get("pages") != 3:
        raise ReadbackError("render_manifest_pdf_contract_mismatch")
    png_meta = figures.get("pngs", {})
    if set(png_meta) != set(METRICS) or any(
        png_meta[metric].get("path") != f"visual_geometry_{metric}.png" or png_meta[metric].get("rows") != 18
        for metric in METRICS
    ):
        raise ReadbackError("render_manifest_png_contract_mismatch")
    if render_manifest.get("csv", {}).get("path") != CSV_NAME or render_manifest.get("csv", {}).get("rows") != 54:
        raise ReadbackError("render_manifest_csv_contract_mismatch")
    expected_phone_counts = geometry.get("coverage", {}).get("phone_eligible_speakers_by_direction")
    if design.get("phone_eligible_speakers_by_direction") != expected_phone_counts or design.get("planned_speakers") != 40:
        raise ReadbackError("render_phone_roster_counts_mismatch")
    if design.get("units") != UNITS:
        raise ReadbackError("render_unit_labels_mismatch")

    return {
        "schema": "world_shape_visual_geometry_render_readback_v1",
        "status": "INDEPENDENT_RENDER_READBACK_PASS",
        "renderer_source_sha256": expected_renderer,
        "independent_source_sha256": INDEPENDENT_SOURCE_SHA256,
        "independent_receipt_sha256": expected_independent,
        "geometry_manifest_raw_sha256": geometry_raw,
        "geometry_manifest_payload_sha256": geometry_payload,
        "render_manifest_raw_sha256": _sha_file(render_manifest_path),
        "render_manifest_payload_sha256": render_payload,
        "verified_endpoint_rows": checked,
        "planned_endpoint_rows": 54,
        "pdf_pages": 3,
        "png_dimensions": png_dimensions,
        "verified_output_files": len(output_entries),
        "no_estimates_recomputed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--geometry-manifest-raw-sha256", required=True)
    parser.add_argument("--geometry-manifest-payload-sha256", required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--expected-independent-receipt-sha256", required=True)
    parser.add_argument("--expected-independent-source-sha256", required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--renderer-source", type=Path, required=True)
    parser.add_argument("--expected-renderer-sha256", required=True)
    parser.add_argument("--independent-source", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
        receipt_path = args.receipt_path.resolve(strict=False)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        with receipt_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        result["receipt_path"] = str(receipt_path)
        result["receipt_raw_sha256"] = _sha_file(receipt_path)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
