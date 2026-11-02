#!/usr/bin/env python3
"""Rasterize and inventory the three-page visual-geometry PDF for review.

This utility is deliberately presentation-only. It verifies the frozen render
and readback receipts, rasterizes all PDF pages with Poppler, and writes an
HTML contact sheet. It does not recompute or inspect scientific estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RENDERER_SHA256 = "0be54043fce6cf4e7ee8f7bee8cc46ed162957c563f0e78a8142b05c6b4f3c93"
VERIFIER_SHA256 = "0bda07981455286c133ef5f898bd276e7580d16d1d6a37644de30d14f1849489"
PRIMARY_SHA256 = "ae9daa7191c401e8c86852bb1bae4666fd1739e897a19c4496b980ce1a52a568"
INDEPENDENT_SHA256 = "2ba5169d61f4e818a505791ae4f97e358b41cca346f53fefbdb102c9b978c06c"
RELEASE_RAW_SHA256 = "4a8a6265de17566ee237e2ed79f034f3df427fd186daa64b345f98253d2005ae"
RELEASE_PAYLOAD_SHA256 = "6eb84aed937a227012627533205720bd145b7aeb63d0a457d7b024f471bd542f"
PDF_NAME = "visual_geometry_family_all54.pdf"
RENDER_MANIFEST_NAME = "visual_geometry_render_manifest.json"
PNG_NAMES = (
    "visual_geometry_mean_aperture.png",
    "visual_geometry_mean_absolute_aperture_velocity_per_s.png",
    "visual_geometry_phone_class_separation.png",
)
CSV_NAME = "visual_geometry_endpoints_all54.csv"
EXPECTED_RENDER_MEMBERS = {PDF_NAME, CSV_NAME, *PNG_NAMES, RENDER_MANIFEST_NAME}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class QaError(RuntimeError):
    pass


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QaError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise QaError(f"json_not_object:{path}")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise QaError(f"invalid_sha256:{label}")
    return value


def png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != PNG_SIGNATURE or raw[12:16] != b"IHDR":
        raise QaError(f"invalid_png:{path.name}")
    return struct.unpack(">II", raw[16:24])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    render_dir = args.render_dir.resolve(strict=True)
    geometry_dir = args.geometry_dir.resolve(strict=True)
    independent_receipt_path = args.independent_receipt.resolve(strict=True)
    verifier_receipt_path = args.verifier_receipt.resolve(strict=True)
    renderer_source = args.renderer_source.resolve(strict=True)
    verifier_source = args.verifier_source.resolve(strict=True)
    primary_source = args.primary_source.resolve(strict=True)
    independent_source = args.independent_source.resolve(strict=True)
    pdfinfo = args.pdfinfo.resolve(strict=True)
    pdftoppm = args.pdftoppm.resolve(strict=True)
    qa_dir = args.qa_dir.resolve(strict=False)
    receipt_path = qa_dir / "visual_geometry_pdf_qa_manifest.json"

    for path, expected, label in (
        (renderer_source, RENDERER_SHA256, "renderer"),
        (verifier_source, VERIFIER_SHA256, "verifier"),
        (primary_source, PRIMARY_SHA256, "primary"),
        (independent_source, INDEPENDENT_SHA256, "independent"),
    ):
        if sha_file(path) != expected:
            raise QaError(f"source_sha_mismatch:{label}")

    if qa_dir.exists():
        raise QaError("qa_output_directory_already_exists")
    if qa_dir == render_dir or qa_dir in render_dir.parents or render_dir in qa_dir.parents:
        raise QaError("qa_receipt_must_be_outside_render_directory")
    qa_dir.mkdir(parents=True, exist_ok=False)
    pages_dir = qa_dir / "pdf_pages"
    pages_dir.mkdir()

    geometry_manifest_path = geometry_dir / "geometry_family_manifest.json"
    geometry_manifest_raw = sha_file(geometry_manifest_path)
    geometry_manifest = read_json(geometry_manifest_path)
    geometry_payload = sha_bytes(canonical({k: v for k, v in geometry_manifest.items() if k != "manifest_sha256"}))
    if geometry_manifest.get("manifest_sha256") != geometry_payload:
        raise QaError("geometry_manifest_payload_mismatch")

    release = geometry_manifest.get("geometry_release")
    if not isinstance(release, dict) or (release.get("raw_sha256"), release.get("payload_sha256")) != (RELEASE_RAW_SHA256, RELEASE_PAYLOAD_SHA256):
        raise QaError("geometry_release_binding_mismatch")

    independent_sha = sha_file(independent_receipt_path)
    if args.expected_independent_receipt_sha256 != independent_sha:
        raise QaError("independent_receipt_raw_sha_mismatch")
    independent_receipt = read_json(independent_receipt_path)
    if (independent_receipt.get("status") != "INDEPENDENT_RECONSTRUCTION_PASS"
            or independent_receipt.get("independent_source_sha256") != INDEPENDENT_SHA256
            or independent_receipt.get("primary_source_sha256") != PRIMARY_SHA256
            or (independent_receipt.get("geometry_manifest_raw_sha256"), independent_receipt.get("geometry_manifest_payload_sha256")) != (geometry_manifest_raw, geometry_payload)
            or (independent_receipt.get("planned_contexts"), independent_receipt.get("planned_tracks"), independent_receipt.get("planned_endpoints")) != (160, 800, 54)):
        raise QaError("independent_receipt_contract_mismatch")

    render_manifest_path = render_dir / RENDER_MANIFEST_NAME
    render_manifest = read_json(render_manifest_path)
    render_manifest_raw = sha_file(render_manifest_path)
    render_payload = sha_bytes(canonical({k: v for k, v in render_manifest.items() if k != "manifest_sha256"}))
    if (render_manifest.get("schema") != "world_shape_visual_geometry_render_v1"
            or render_manifest.get("status") != "RENDERED_ALL_54_PLANNED_ENDPOINTS"
            or render_manifest.get("manifest_sha256") != render_payload):
        raise QaError("render_manifest_contract_mismatch")
    if (render_manifest.get("renderer_source", {}).get("raw_sha256") != RENDERER_SHA256
            or render_manifest.get("primary_source_sha256") != PRIMARY_SHA256
            or render_manifest.get("independent_source_sha256") != INDEPENDENT_SHA256):
        raise QaError("render_source_binding_mismatch")
    render_input = render_manifest.get("input", {})
    if (render_input.get("geometry_manifest_raw_sha256"), render_input.get("geometry_manifest_payload_sha256")) != (geometry_manifest_raw, geometry_payload):
        raise QaError("render_geometry_binding_mismatch")
    if (render_input.get("independent_receipt_raw_sha256"), render_input.get("independent_receipt_status")) != (independent_sha, "INDEPENDENT_RECONSTRUCTION_PASS"):
        raise QaError("render_independent_receipt_binding_mismatch")

    verifier_sha = sha_file(verifier_receipt_path)
    verifier_receipt = read_json(verifier_receipt_path)
    if (verifier_receipt.get("status") != "INDEPENDENT_RENDER_READBACK_PASS"
            or verifier_receipt.get("renderer_source_sha256") != RENDERER_SHA256
            or verifier_receipt.get("independent_source_sha256") != INDEPENDENT_SHA256
            or verifier_receipt.get("independent_receipt_sha256") != independent_sha
            or (verifier_receipt.get("geometry_manifest_raw_sha256"), verifier_receipt.get("geometry_manifest_payload_sha256")) != (geometry_manifest_raw, geometry_payload)
            or (verifier_receipt.get("render_manifest_raw_sha256"), verifier_receipt.get("render_manifest_payload_sha256")) != (render_manifest_raw, render_payload)
            or verifier_receipt.get("pdf_pages") != 3
            or verifier_receipt.get("verified_output_files") != 5):
        raise QaError("verifier_receipt_contract_mismatch")
    recorded_dimensions = verifier_receipt.get("png_dimensions", {})
    if set(recorded_dimensions) != set(PNG_NAMES) or any(
        recorded_dimensions[name] != {"width_px": 1800, "height_px": 1150} for name in PNG_NAMES
    ):
        raise QaError("verifier_png_dimensions_mismatch")

    if {p.name for p in render_dir.iterdir()} != EXPECTED_RENDER_MEMBERS:
        raise QaError("render_directory_fileset_mismatch")
    entries = render_manifest.get("output_files")
    expected_artifacts = EXPECTED_RENDER_MEMBERS - {RENDER_MANIFEST_NAME}
    if not isinstance(entries, list) or {row.get("path") for row in entries if isinstance(row, dict)} != expected_artifacts or len(entries) != 5:
        raise QaError("render_output_inventory_mismatch")
    checked_outputs: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for row in entries:
        if set(row) != {"path", "bytes", "sha256"}:
            raise QaError("render_output_inventory_schema_mismatch")
        name = row["path"]
        expected_hash = require_sha(row["sha256"], f"render:{name}")
        path = render_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha_file(path) != expected_hash:
            raise QaError(f"render_output_hash_mismatch:{name}")
        checked = {"path": str(path), "bytes": path.stat().st_size, "sha256": expected_hash}
        checked_outputs.append(checked)
        by_name[name] = checked

    pdf_path = render_dir / PDF_NAME
    started = utc_now()
    pdfinfo_result = subprocess.run([str(pdfinfo), str(pdf_path)], capture_output=True, check=False)
    ended = utc_now()
    pdfinfo_stdout = pdfinfo_result.stdout.decode("utf-8", errors="replace")
    pdfinfo_stderr = pdfinfo_result.stderr.decode("utf-8", errors="replace")
    if pdfinfo_result.returncode != 0:
        raise QaError(f"pdfinfo_failed:{pdfinfo_result.returncode}:{pdfinfo_stderr[:500]}")
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", pdfinfo_stdout, flags=re.MULTILINE)
    if not page_match or int(page_match.group(1)) != 3:
        raise QaError("pdfinfo_page_count_not_three")
    pdfinfo_log = {
        "argv": [str(pdfinfo), str(pdf_path)], "started_utc": started, "ended_utc": ended,
        "exit_code": pdfinfo_result.returncode, "stdout": pdfinfo_stdout, "stderr": pdfinfo_stderr,
        "stdout_sha256": sha_bytes(pdfinfo_result.stdout), "stderr_sha256": sha_bytes(pdfinfo_result.stderr),
    }

    page_prefix = pages_dir / "visual_geometry_pdf_page"
    started = utc_now()
    pdftoppm_result = subprocess.run(
        [str(pdftoppm), "-f", "1", "-l", "3", "-r", "120", "-png", str(pdf_path), str(page_prefix)],
        capture_output=True, check=False,
    )
    ended = utc_now()
    pdftoppm_stdout = pdftoppm_result.stdout.decode("utf-8", errors="replace")
    pdftoppm_stderr = pdftoppm_result.stderr.decode("utf-8", errors="replace")
    if pdftoppm_result.returncode != 0:
        raise QaError(f"pdftoppm_failed:{pdftoppm_result.returncode}:{pdftoppm_stderr[:500]}")
    page_files = sorted(pages_dir.glob("visual_geometry_pdf_page-*.png"), key=lambda p: p.name)
    if len(page_files) != 3 or [p.name for p in page_files] != [f"visual_geometry_pdf_page-{n}.png" for n in (1, 2, 3)]:
        raise QaError("pdf_page_raster_fileset_mismatch")
    raster_pages = []
    for page_number, path in enumerate(page_files, start=1):
        width, height = png_dimensions(path)
        raster_pages.append({"page": page_number, "path": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path), "width_px": width, "height_px": height})
    raster_log = {
        "argv": [str(pdftoppm), "-f", "1", "-l", "3", "-r", "120", "-png", str(pdf_path), str(page_prefix)],
        "started_utc": started, "ended_utc": ended, "exit_code": pdftoppm_result.returncode,
        "stdout_sha256": sha_bytes(pdftoppm_result.stdout), "stderr_sha256": sha_bytes(pdftoppm_result.stderr),
        "stdout": pdftoppm_stdout, "stderr": pdftoppm_stderr,
    }

    image_rows = []
    for name in PNG_NAMES:
        width, height = png_dimensions(render_dir / name)
        if (width, height) != (1800, 1150):
            raise QaError(f"render_png_dimensions_mismatch:{name}")
        href = Path(os.path.relpath(render_dir / name, qa_dir)).as_posix()
        image_rows.append({**by_name[name], "label": name, "width_px": width, "height_px": height, "href": href})
    for row in raster_pages:
        image_rows.append({**row, "label": f"PDF page {row['page']}", "href": f"pdf_pages/{row['path']}"})
    cards = []
    for row in image_rows:
        cards.append(
            '<section><h2>{}</h2><p>{} bytes<br>SHA-256 <code>{}</code><br>{} × {} px</p>'
            '<a href="{}"><img src="{}" alt="{}"></a></section>'.format(
                html.escape(row["label"]), row["bytes"], html.escape(row["sha256"]), row["width_px"], row["height_px"],
                html.escape(row["href"], quote=True), html.escape(row["href"], quote=True), html.escape(row["label"], quote=True),
            )
        )
    gallery = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Visual geometry render QA</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#18212b}main{display:grid;"
        "grid-template-columns:repeat(2,minmax(0,1fr));gap:1.5rem}section{border:1px solid #ccd3dc;"
        "padding:1rem;overflow-wrap:anywhere}img{max-width:100%;height:auto}code{font-size:.8rem}"
        "</style></head><body><h1>54-endpoint geometry render: page/image review</h1>"
        "<p>Six images are linked below: the three standalone metric PNGs and all three PDF pages at 120 DPI."
        " Hashes are recorded for independent confirmation. This page is prepared for ROOT visual review; the contact sheet itself does not approve layout quality.</p>"
        "<main>" + "".join(cards) + "</main></body></html>"
    )
    gallery_path = qa_dir / "visual_geometry_contact_sheet.html"
    with gallery_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(gallery)
        stream.flush()
        os.fsync(stream.fileno())

    result = {
        "schema": "world_shape_visual_geometry_pdf_page_qa_v1",
        "status": "PDF_RASTERIZED_AWAITING_ROOT_VISUAL_REVIEW",
        "visual_quality_approved": False,
        "root_visual_review_pending": True,
        "manual_visual_review_required": False,
        "no_estimates_recomputed": True,
        "sources": {
            "renderer": {"path": str(renderer_source), "sha256": RENDERER_SHA256},
            "verifier": {"path": str(verifier_source), "sha256": VERIFIER_SHA256},
            "primary": {"path": str(primary_source), "sha256": PRIMARY_SHA256},
            "independent": {"path": str(independent_source), "sha256": INDEPENDENT_SHA256},
        },
        "inputs": {
            "geometry_dir": str(geometry_dir), "geometry_manifest_raw_sha256": geometry_manifest_raw,
            "geometry_manifest_payload_sha256": geometry_payload,
            "independent_receipt_path": str(independent_receipt_path), "independent_receipt_raw_sha256": independent_sha,
            "render_dir": str(render_dir), "render_manifest_path": str(render_manifest_path),
            "render_manifest_raw_sha256": render_manifest_raw, "render_manifest_payload_sha256": render_payload,
            "verifier_receipt_path": str(verifier_receipt_path), "verifier_receipt_raw_sha256": verifier_sha,
            "verifier_status": verifier_receipt["status"],
        },
        "tools": {
            "pdfinfo": {"path": str(pdfinfo), "sha256": sha_file(pdfinfo), "result": pdfinfo_log},
            "pdftoppm": {"path": str(pdftoppm), "sha256": sha_file(pdftoppm), "result": raster_log},
        },
        "pdf": {"path": str(pdf_path), "bytes": pdf_path.stat().st_size, "sha256": by_name[PDF_NAME]["sha256"], "pages_reported": 3, "raster_dpi": 120, "raster_pages": raster_pages},
        "standalone_pngs": image_rows[:3],
        "contact_sheet": {"path": str(gallery_path), "bytes": gallery_path.stat().st_size, "sha256": sha_file(gallery_path)},
        "rendered_artifacts": checked_outputs,
    }
    result["manifest_sha256"] = sha_bytes(canonical(result))
    with receipt_path.open("xb") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": result["status"], "manifest_path": str(receipt_path), "manifest_sha256": sha_file(receipt_path), "pages": 3, "images": len(image_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--expected-independent-receipt-sha256", required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--verifier-receipt", type=Path, required=True)
    parser.add_argument("--renderer-source", type=Path, required=True)
    parser.add_argument("--verifier-source", type=Path, required=True)
    parser.add_argument("--primary-source", type=Path, required=True)
    parser.add_argument("--independent-source", type=Path, required=True)
    parser.add_argument("--pdfinfo", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
