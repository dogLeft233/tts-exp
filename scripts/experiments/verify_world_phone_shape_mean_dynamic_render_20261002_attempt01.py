#!/usr/bin/env python3
"""Verify the sealed WORLD phone-shape render and its lossless CSV lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


RENDERER_SHA256 = "a7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd"
FIGURE_BINDING_SHA256 = "f95142b36ae82363f5d0841408818fc749184a1d5b960828a13e313f94305211"
CONSUMER_SOURCE_SHA256 = "dec1148b8883ff964142cdaf3716b15f27e6e6c5aa21e710a4cf8636dc8d9a44"
ANALYSIS_MANIFEST_SHA256 = "7c65c8e58ff73ee3747789f19731f6a3d76a39960c478fb872510bc4dd7439da"
FAMILY_ENDPOINTS_SHA256 = "c47cb2e0fb5b6eb014a54ae9b13fc86774cf8ce983753495e8a020e5a819de1c"
EXTRACT_MANIFEST_SHA256 = "0aef8129d24090abec3f3c688b268f620415fbae594cf2f10f0091bd627f35f4"
EXACT_JOIN_SHA256 = "c87fd4d6945a914a7321f0fef321355933f09799c37ee641337d3c00060e6dd2"
GLOBAL_EXTRACT_SHA256 = "2f1996f9cadd08211e322105c9d45c954beffa9b0b04a2f6da6d825f9d8771e4"
LOCAL_EXTRACT_SHA256 = "166352df40d2c0904071007c9194533cf6aaf89b9c38292599007ae75aef398d"
NEGATIVE_EXTRACT_SHA256 = "9712e27029793aa8b3a7c19d19f22cf356eeec0e4690c17bc252aa3d497c7e59"

FAMILY_SIZE = 11_776
GLOBAL_SIZE = 3_456
LOCAL_PHONE_SIZE = 8_192
LOCAL_OVERALL_SIZE = 128
CSV_FILES = {
    "family_endpoints_all.csv": FAMILY_SIZE,
    "global_all_metrics.csv": GLOBAL_SIZE,
    "local_phone_endpoints.csv": LOCAL_PHONE_SIZE,
    "local_overall_endpoints.csv": LOCAL_OVERALL_SIZE,
}
RENDER_FILES = {
    *CSV_FILES,
    "global_C_fixed_anchor_forest.pdf",
    "local_phone_phase_heatmaps.pdf",
    "guard15_C_fixed_anchor_overview.png",
    "guard15_C_fixed_anchor_overview.svg",
}
CSV_FIELDS = (
    "planned_family_index", "endpoint_key", "family", "contrast", "treatment", "direction", "geometry",
    "support", "metric", "audio_domain", "estimand", "phone", "phase", "status",
    "estimate_speaker_equal_mean", "bootstrap_se", "ci99_lower", "ci99_upper", "family_critical_99_max_t",
)
RENDER_SCHEMA = "ROOT_APPROVED_WORLD_PHONE_SHAPE_FIGURE_INPUT_V2"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_sha(value: dict[str, Any], excluded: str) -> str:
    raw = json.dumps({key: item for key, item in value.items() if key != excluded}, ensure_ascii=False,
                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha_bytes(raw)


def projected_csv_record(row: dict[str, Any]) -> dict[str, str]:
    fields = (
        "planned_family_index", "endpoint_key", "family", "contrast", "treatment", "direction", "geometry",
        "support", "metric", "audio_domain", "estimand", "phone", "phase", "status",
        "estimate_speaker_equal_mean", "bootstrap_se", "family_critical_99_max_t",
    )
    record: dict[str, Any] = {field: row.get(field) for field in fields}
    ci = row.get("ci99_simultaneous_max_t")
    record["ci99_lower"] = ci[0] if isinstance(ci, list) and len(ci) == 2 else None
    record["ci99_upper"] = ci[1] if isinstance(ci, list) and len(ci) == 2 else None
    return {field: "" if record.get(field) is None else str(record[field]) for field in CSV_FIELDS}


def read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == expected_fields, f"CSV header differs: {path.name}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"CSV row has extra columns: {path.name}")
    return rows


def verify_csv_exact(path: Path, source_rows: list[dict[str, Any]], label: str) -> int:
    actual = read_csv(path, CSV_FIELDS)
    expected = [projected_csv_record(row) for row in source_rows]
    require(len(actual) == len(expected), f"{label} row count differs: {len(actual)}/{len(expected)}")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if left != right:
            raise VerifyError(f"{label} lossless projection differs at row {index}")
    return len(actual)


def verify_extracts(extract_root: Path, global_render_rows: list[dict[str, Any]], all_local_rows: list[dict[str, Any]]) -> dict[str, int]:
    manifest_path = extract_root / "extract_manifest.json"
    require(sha_file(manifest_path) == EXTRACT_MANIFEST_SHA256, "report extract manifest SHA mismatch")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "EXTRACT_COMPLETE" and manifest.get("schema") == "WORLD_PHONE_SHAPE_DYNAMIC_REPORT_TABLE_EXTRACT_V1", "report extract manifest status/schema mismatch")
    require(manifest.get("family_counts", {}).get("planned") == FAMILY_SIZE and manifest.get("family_counts", {}).get("global") == GLOBAL_SIZE, "report extract family counts mismatch")
    join_path = extract_root / "root_exact_join_readback_attempt01.json"
    require(sha_file(join_path) == EXACT_JOIN_SHA256, "root exact-join receipt SHA mismatch")
    join = read_json(join_path)
    require(join.get("status") == "ROOT_REPORT_TABLE_FULL_EXACT_JOIN_PASS" and join.get("source_family_rows") == FAMILY_SIZE and join.get("projected_cells_compared") == 33_868 and join.get("mismatches") == 0 and join.get("subset_is_descriptive") is True, "root exact-join receipt contract mismatch")

    global_path = extract_root / "guard15_global_C_raw_A_RAW_all9_contrasts.csv"
    local_path = extract_root / "full_local_phone_all_endpoints_with_negative_flag.csv"
    negative_path = extract_root / "full_local_phone_simultaneous_ci_below_zero_35.csv"
    for path, expected in ((global_path, GLOBAL_EXTRACT_SHA256), (local_path, LOCAL_EXTRACT_SHA256), (negative_path, NEGATIVE_EXTRACT_SHA256)):
        require(sha_file(path) == expected, f"report extract CSV SHA mismatch: {path.name}")

    global_extract = read_csv(global_path, (
        "planned_family_index", "endpoint_key", "contrast", "direction", "estimand", "status", "support", "metric",
        "geometry", "audio_domain", "estimate_speaker_equal_mean", "bootstrap_se", "ci99_lower", "ci99_upper",
        "family_critical_99_max_t",
    ))
    global_lookup = {row["endpoint_key"]: row for row in global_extract}
    require(len(global_extract) == 36 and len(global_lookup) == 36, "guard15 global extract does not have 36 distinct rows")
    projected_global_fields = (
        "planned_family_index", "endpoint_key", "contrast", "direction", "estimand", "status", "support", "metric",
        "geometry", "audio_domain", "estimate_speaker_equal_mean", "bootstrap_se", "ci99_lower", "ci99_upper",
        "family_critical_99_max_t",
    )
    render_global = {row["endpoint_key"]: projected_csv_record(row) for row in global_render_rows}
    for extract_row in global_extract:
        actual = render_global.get(extract_row["endpoint_key"])
        require(actual is not None, "guard15 extract endpoint is absent from rendered global rows")
        require({field: actual[field] for field in projected_global_fields} == extract_row, "guard15 table extract differs from rendered global source row")

    local_fields = (
        "planned_family_index", "endpoint_key", "phone", "phase", "treatment", "direction", "geometry", "support",
        "estimand", "status", "estimate_speaker_equal_mean", "bootstrap_se", "ci99_lower", "ci99_upper",
        "family_critical_99_max_t", "ci99_upper_below_zero",
    )
    local_extract = read_csv(local_path, local_fields)
    local_lookup = {row["endpoint_key"]: row for row in local_extract}
    require(len(local_extract) == 2_048 and len(local_lookup) == 2_048, "FULL local extract does not have 2,048 distinct rows")
    render_local = {row["endpoint_key"]: projected_csv_record(row) for row in all_local_rows}
    local_projection_fields = local_fields[:-1]
    for extract_row in local_extract:
        actual = render_local.get(extract_row["endpoint_key"])
        require(actual is not None, "FULL local extract endpoint is absent from rendered phone rows")
        require({field: actual[field] for field in local_projection_fields} == {field: extract_row[field] for field in local_projection_fields}, "FULL local extract differs from rendered local source row")
        should_be_below = extract_row["status"] == "estimable" and float(extract_row["ci99_upper"]) < 0
        require(extract_row["ci99_upper_below_zero"] == ("True" if should_be_below else "False"), "FULL local descriptive negative flag differs from its interval/status")

    negative = read_csv(negative_path, local_fields)
    expected_negative = [row for row in local_extract if row["ci99_upper_below_zero"] == "True"]
    require(len(negative) == 35 and len(expected_negative) == 35, "descriptive local negative subset count differs from frozen full extract")
    require(negative == expected_negative, "descriptive local negative subset is not an exact ordered slice of the full extract")
    return {"guard15_global_rows_joined": len(global_extract), "full_local_rows_joined": len(local_extract), "descriptive_negative_rows": len(negative)}


def pdf_pages(path: Path) -> int:
    raw = path.read_bytes()
    require(raw.startswith(b"%PDF-") and raw.rstrip().endswith(b"%%EOF"), f"PDF trailer/header invalid: {path.name}")
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
        return len(PdfReader(str(path)).pages)
    except ImportError:
        pages = re.findall(rb"/Type\s*/Page(?!s)\b", raw)
        return len(pages)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    analysis_root = args.analysis_root.resolve(strict=True)
    render_root = args.render_dir.resolve(strict=True)
    extract_root = args.extract_dir.resolve(strict=True)
    binding_path = args.figure_binding.resolve(strict=True)
    require(sha_file(binding_path) == FIGURE_BINDING_SHA256 and args.figure_binding_raw_sha256 == FIGURE_BINDING_SHA256, "figure binding SHA differs from frozen receipt")
    binding = read_json(binding_path)
    require(binding.get("status") == "FROZEN" and binding.get("schema") == RENDER_SCHEMA, "figure binding is not frozen under the expected schema")
    require(sha_file(analysis_root / "independent_output_manifest.json") == ANALYSIS_MANIFEST_SHA256, "consumer manifest SHA differs")
    require(sha_file(analysis_root / "family_endpoints.jsonl") == FAMILY_ENDPOINTS_SHA256, "family endpoint source SHA differs")
    endpoints = read_jsonl(analysis_root / "family_endpoints.jsonl")
    require(len(endpoints) == FAMILY_SIZE, "family endpoint source row count differs")
    require([row.get("planned_family_index") for row in endpoints] == list(range(FAMILY_SIZE)), "family source endpoint order/index differs")
    global_rows = [row for row in endpoints if row.get("family") == "global"]
    local_rows = [row for row in endpoints if row.get("family") == "local_phone"]
    overall_rows = [row for row in endpoints if row.get("family") == "local_overall"]
    require((len(global_rows), len(local_rows), len(overall_rows)) == (GLOBAL_SIZE, LOCAL_PHONE_SIZE, LOCAL_OVERALL_SIZE), "family endpoint category counts differ")

    manifest_path = render_root / "render_manifest.json"
    require(sha_file(manifest_path) == args.render_manifest_raw_sha256, "render manifest raw SHA differs from supplied seal")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "RENDERED", "render manifest status/schema mismatch")
    require(manifest.get("payload_sha256") == canonical_sha(manifest, "payload_sha256"), "render manifest payload self-hash mismatch")
    require(manifest.get("renderer_source_sha256") == RENDERER_SHA256, "rendered renderer source SHA mismatch")
    source_path = Path(manifest.get("renderer_source_path", "")).resolve(strict=True)
    require(sha_file(source_path) == RENDERER_SHA256, "actual renderer source bytes differ")
    rendered_input = manifest.get("input", {})
    for field, value in {
        "binding_raw_sha256": FIGURE_BINDING_SHA256,
        "binding_payload_sha256": binding.get("payload_sha256"),
        "analysis_manifest_raw_sha256": ANALYSIS_MANIFEST_SHA256,
        "consumer_source_sha256": CONSUMER_SOURCE_SHA256,
        "family_endpoints_sha256": FAMILY_ENDPOINTS_SHA256,
        "verified_output_file_count": 645,
        "analysis_output_root": str(analysis_root),
    }.items():
        require(rendered_input.get(field) == value, f"render manifest input {field} mismatch")
    require(manifest.get("csv_rows") == {
        "family_endpoints_all": FAMILY_SIZE, "global_all_metrics": GLOBAL_SIZE,
        "local_phone": LOCAL_PHONE_SIZE, "local_overall": LOCAL_OVERALL_SIZE,
    }, "render manifest CSV counts differ from the frozen family")
    figures = manifest.get("rendered_figures", {})
    require(figures.get("global_forest", {}).get("pages") == 9, "global figure page count differs")
    require(figures.get("local_heatmaps", {}).get("pages") == 32, "local figure page count differs")
    require(manifest.get("guard15_overview", {}).get("endpoint_count") == 144, "guard15 overview endpoint count differs")

    inventory = manifest.get("output_files")
    require(isinstance(inventory, list) and len(inventory) == 8, "render output inventory must have eight assets")
    inventory_map = {row.get("path"): row for row in inventory}
    require(len(inventory_map) == 8 and set(inventory_map) == RENDER_FILES, "render output asset names differ")
    for name, entry in inventory_map.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe render output path: {name}")
        path = (render_root / rel).resolve(strict=True)
        require(path.is_file() and not (render_root / rel).is_symlink(), f"render output is not a regular file: {name}")
        require(path.is_relative_to(render_root), f"render output escapes its root: {name}")
        require(path.stat().st_size == entry.get("bytes") and entry.get("bytes", 0) > 0, f"render output bytes differ: {name}")
        require(sha_file(path) == entry.get("sha256"), f"render output SHA differs: {name}")
    actual_names = {path.relative_to(render_root).as_posix() for path in render_root.rglob("*") if path.is_file() and path.name != "render_manifest.json"}
    require(not any(path.is_symlink() for path in render_root.rglob("*")), "render output contains a symlink")
    require(actual_names == RENDER_FILES, "render output directory has missing or extra files")

    csv_counts = {
        "family_endpoints_all.csv": verify_csv_exact(render_root / "family_endpoints_all.csv", endpoints, "family_endpoints_all"),
        "global_all_metrics.csv": verify_csv_exact(render_root / "global_all_metrics.csv", global_rows, "global_all_metrics"),
        "local_phone_endpoints.csv": verify_csv_exact(render_root / "local_phone_endpoints.csv", local_rows, "local_phone_endpoints"),
        "local_overall_endpoints.csv": verify_csv_exact(render_root / "local_overall_endpoints.csv", overall_rows, "local_overall_endpoints"),
    }
    extract_counts = verify_extracts(extract_root, global_rows, local_rows)
    page_counts = {
        "global_C_fixed_anchor_forest.pdf": pdf_pages(render_root / "global_C_fixed_anchor_forest.pdf"),
        "local_phone_phase_heatmaps.pdf": pdf_pages(render_root / "local_phone_phase_heatmaps.pdf"),
    }
    require(page_counts == {"global_C_fixed_anchor_forest.pdf": 9, "local_phone_phase_heatmaps.pdf": 32}, "PDF pages differ from the renderer contract")
    png = (render_root / "guard15_C_fixed_anchor_overview.png").read_bytes()
    require(png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) >= 24, "overview PNG signature is invalid")
    png_width, png_height = struct.unpack(">II", png[16:24])
    require(png_width > 0 and png_height > 0, "overview PNG dimensions are invalid")
    ET.parse(render_root / "guard15_C_fixed_anchor_overview.svg")
    return {
        "status": "RENDER_READBACK_PASS",
        "render_manifest_sha256": sha_file(manifest_path),
        "renderer_source_sha256": RENDERER_SHA256,
        "analysis_manifest_sha256": ANALYSIS_MANIFEST_SHA256,
        "family_endpoint_rows": len(endpoints),
        "csv_rows": csv_counts,
        "extract_exact_join_rows": extract_counts,
        "assets": len(inventory), "pdf_pages": page_counts,
        "overview_png_dimensions": [png_width, png_height],
        "scope": "Manifest, file hashes, exact CSV projections, extract joins, and figure structure only; no statistics are recomputed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--figure-binding", type=Path, required=True)
    parser.add_argument("--figure-binding-raw-sha256", required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--render-manifest-raw-sha256", required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = args.receipt.resolve(strict=False)
        require(not receipt.exists(), f"refusing to overwrite verifier receipt: {receipt}")
        result = verify(args)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        report = {"schema": "WORLD_PHONE_SHAPE_DYNAMIC_RENDER_READBACK_V1", **result}
        receipt.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"status": report["status"], "receipt": str(receipt), "receipt_sha256": sha_file(receipt),
                          "assets": report["assets"], "family_rows": report["family_endpoint_rows"], "pdf_pages": report["pdf_pages"]}, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
