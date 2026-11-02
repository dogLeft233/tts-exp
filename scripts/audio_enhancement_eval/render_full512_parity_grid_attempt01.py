"""Render a read-only visual grid from sealed full512 proxy/native arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "cross_tfg_full512_parity_visual_grid_v1"
SUMMARY_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
FRAME_SAMPLES = (0, 8, 16, 24, 31)
ARRAY_KEYS = (
    "proxy_center_frames_bgr_u8",
    "official_center_frames_bgr_u8",
    "official_full512_center_generated_u8",
    "official_full512_center_decoded_u8",
)
TILE = 192
MARGIN = 14


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_image_stack(name: str, value: np.ndarray) -> np.ndarray:
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[0] != 32 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 [32,H,W,3], got {value.dtype} {value.shape}")
    if value.shape[1] < 32 or value.shape[2] < 32:
        raise ValueError(f"{name} has implausibly small frames: {value.shape}")
    return np.ascontiguousarray(value)


def tile_image(frame: np.ndarray, label: str, box: list[int] | None = None) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    if box is not None:
        y1, y2, x1, x2 = (int(v) for v in box)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    height, width = image.shape[:2]
    scale = min(TILE / width, TILE / height)
    out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE + 30, TILE, 3), dtype=np.uint8)
    xoff = (TILE - out_w) // 2
    yoff = (TILE - out_h) // 2 + 22
    tile[yoff : yoff + out_h, xoff : xoff + out_w] = resized
    cv2.putText(tile, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def render_row(row: dict[str, Any], row_manifest: dict[str, Any], portrait: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    npz_path = Path(row_manifest["npz_path"])
    expected_bytes = int(row_manifest["npz_bytes"])
    expected_sha = str(row_manifest["npz_sha256"])
    if not npz_path.is_file() or npz_path.stat().st_size != expected_bytes:
        raise ValueError(f"missing or wrong-sized source arrays: {row['sample_id']}")
    before_sha = sha256_file(npz_path)
    if before_sha != expected_sha:
        raise ValueError(f"source array SHA mismatch: {row['sample_id']}")

    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(set(ARRAY_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"missing visual arrays for {row['sample_id']}: {missing}")
        arrays = {key: require_image_stack(key, archive[key]) for key in ARRAY_KEYS}

    if len({value.shape[0] for value in arrays.values()}) != 1:
        raise ValueError(f"32-frame arrays do not share a common leading dimension: {row['sample_id']}")
    q0 = int(row_manifest["center_query_start_source_frame"])
    if q0 != int(row_manifest["center_query_start_selected_track_local"]):
        raise ValueError(f"selected track local/source center starts differ: {row['sample_id']}")
    if int(row_manifest["selected_track_start_source_frame"]) != 0:
        raise ValueError(f"grid renderer expects the reviewed source-frame-zero track: {row['sample_id']}")
    face_box = [int(v) for v in row_manifest["face_box_y1_y2_x1_x2"]]
    if len(face_box) != 4:
        raise ValueError("face box must be [y1,y2,x1,x2]")

    columns = ("Natural portrait / W2L box", "Proxy", "Full512 generated", "Full512 decoded", "SyncNet crop")
    header_h = 54
    row_h = TILE + 42
    label_w = 118
    block_w = label_w + len(columns) * (TILE + MARGIN) + MARGIN
    canvas = np.zeros((header_h + len(FRAME_SAMPLES) * row_h + MARGIN, block_w, 3), dtype=np.uint8)
    title = f"{row['sample_id']} | {row['method_id']} | 25 fps | source-clock q0={q0}; 27 queries, lag window q0-15..q0+41"
    cv2.putText(canvas, title, (MARGIN, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    for c, name in enumerate(columns):
        x = label_w + MARGIN + c * (TILE + MARGIN)
        cv2.putText(canvas, name, (x, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (210, 210, 210), 1, cv2.LINE_AA)

    array_hashes: dict[str, str] = {}
    for key, value in arrays.items():
        array_hashes[key] = hashlib.sha256(value.tobytes(order="C")).hexdigest()

    for ri, frame_offset in enumerate(FRAME_SAMPLES):
        clock_frame = q0 + frame_offset
        y = header_h + ri * row_h
        timestamp = f"{clock_frame}/25 s\n({clock_frame / 25:.2f}s)"
        for line_i, line in enumerate(timestamp.splitlines()):
            cv2.putText(canvas, line, (6, y + 28 + 22 * line_i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

        frames = (
            tile_image(portrait, "", box=face_box),
            tile_image(arrays[ARRAY_KEYS[0]][frame_offset], ""),
            tile_image(arrays[ARRAY_KEYS[2]][frame_offset], "", box=face_box),
            tile_image(arrays[ARRAY_KEYS[3]][frame_offset], "", box=face_box),
            tile_image(arrays[ARRAY_KEYS[1]][frame_offset], ""),
        )
        for ci, tile in enumerate(frames):
            x = label_w + MARGIN + ci * (TILE + MARGIN)
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile

    after_bytes = npz_path.stat().st_size
    after_sha = sha256_file(npz_path)
    if after_bytes != expected_bytes or after_sha != expected_sha:
        raise ValueError(f"source arrays changed while rendering: {row['sample_id']}")
    return canvas, {
        "row_id": row["row_id"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "historical_exposed": row["historical_exposed"],
        "method_id": row["method_id"],
        "source_npz_path": str(npz_path),
        "source_npz_bytes": expected_bytes,
        "source_npz_sha256": before_sha,
        "source_npz_after_bytes": after_bytes,
        "source_npz_after_sha256": after_sha,
        "source_unchanged": True,
        "face_box_y1_y2_x1_x2": face_box,
        "selected_track_start_source_frame": int(row_manifest["selected_track_start_source_frame"]),
        "center_query_start_source_frame": q0,
        "center_query_start_selected_track_local": int(row_manifest["center_query_start_selected_track_local"]),
        "center_query_count": 27,
        "lag_offsets_frames": list(range(-15, 16)),
        "displayed_frame_offsets_within_center32": list(FRAME_SAMPLES),
        "displayed_source_clock_frames": [q0 + i for i in FRAME_SAMPLES],
        "displayed_source_clock_pts": [{"num": q0 + i, "den": 25} for i in FRAME_SAMPLES],
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_sha256": array_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    manifest_raw = args.input_manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest.get("schema") != "cross_tfg_full512_visual_grid_inputs_v1":
        raise SystemExit("unexpected visual-grid input schema")
    if manifest.get("summary_raw_sha256") != SUMMARY_SHA256:
        raise SystemExit("full512 summary pin mismatch")
    summary_path = Path(manifest["summary_path"])
    if not summary_path.is_file() or summary_path.stat().st_size != int(manifest["summary_bytes"]):
        raise SystemExit("full512 summary missing or wrong-sized")
    if sha256_file(summary_path) != SUMMARY_SHA256:
        raise SystemExit("full512 summary raw SHA mismatch")
    if args.output_root.exists():
        raise SystemExit("output root already exists; refusing overwrite")

    portrait_path = Path(manifest["source_portrait_path"])
    if not portrait_path.is_file() or sha256_file(portrait_path) != manifest["source_portrait_sha256"]:
        raise SystemExit("source portrait missing or SHA mismatch")
    portrait = cv2.imread(str(portrait_path), cv2.IMREAD_COLOR)
    if portrait is None or portrait.shape != (512, 512, 3):
        raise SystemExit("source portrait must decode as 512x512 BGR")

    if len(manifest.get("rows", [])) != 2:
        raise SystemExit("expected exactly two pinned historical fit rows")
    if any(row.get("split") != "fit" or row.get("historical_exposed") is not True or row.get("method_id") != "rms_fixed" for row in manifest["rows"]):
        raise SystemExit("visual grid rows must be the pinned fit-only rms_fixed diagnostics")

    args.output_root.mkdir(parents=True, exist_ok=False)
    receipt_rows: list[dict[str, Any]] = []
    output_path = args.output_root / "proxy_native_full512_frame_grid.png"
    sheets = []
    for row in manifest["rows"]:
        sheet, row_receipt = render_row(row, row["visual_arrays"], portrait)
        sheets.append(sheet)
        receipt_rows.append(row_receipt)
    combined = np.vstack(sheets)
    if not cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise SystemExit("OpenCV failed to write the visual grid")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "read-only rendered view of sealed arrays; no model forward, score recomputation, or source mutation",
        "summary_raw_sha256": SUMMARY_SHA256,
        "input_manifest_path": str(args.input_manifest),
        "input_manifest_bytes": len(manifest_raw),
        "input_manifest_sha256": manifest_sha,
        "opencv_version": cv2.__version__,
        "source_portrait_path": str(portrait_path),
        "source_portrait_sha256": manifest["source_portrait_sha256"],
        "display_transform": "BGR arrays resized only for display to fit 192px tiles; face-box overlay is W2L box [y1,y2,x1,x2], not SyncNet crop geometry",
        "columns": ["source portrait with W2L box", "proxy frames", "native full512 generated", "native full512 decoded", "official SyncNet tracked crop"],
        "rows": receipt_rows,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_root / "visual_grid_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "status": "PASS",
        "rows": len(receipt_rows),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "receipt_path": str(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
        "receipt_sha256": sha256_file(receipt_path),
        "payload_sha256": receipt["payload_sha256"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
