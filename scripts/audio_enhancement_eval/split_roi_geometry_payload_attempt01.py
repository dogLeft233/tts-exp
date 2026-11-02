#!/usr/bin/env python3
"""Read-only chunk exporter for a completed ROI NPZ over a capped bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


MAX_PART_BYTES = 6_500_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-part-bytes", type=int, default=MAX_PART_BYTES)
    args = parser.parse_args()
    started = time.time()
    output_dir = args.output_dir
    receipt_path = output_dir / "roi_geometry_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("completed collector receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_READ_ONLY_SAVED_FRAMES":
        raise RuntimeError("collector receipt is not PASS_READ_ONLY_SAVED_FRAMES")
    npz_meta = receipt.get("selected_arrays_npz", {})
    npz_path = Path(str(npz_meta.get("path", "")))
    expected_path = output_dir / "roi_geometry_selected_frames.npz"
    if npz_path != expected_path or not npz_path.is_file():
        raise RuntimeError("receipt NPZ path does not resolve to the sealed collector output")
    expected_bytes = int(npz_meta.get("bytes", -1))
    expected_sha = str(npz_meta.get("sha256", ""))
    if expected_bytes <= 0 or len(expected_sha) != 64:
        raise RuntimeError("collector receipt lacks a valid NPZ size/SHA")
    before = {"bytes": npz_path.stat().st_size, "sha256": sha256_file(npz_path)}
    if before["bytes"] != expected_bytes or before["sha256"] != expected_sha:
        raise RuntimeError("NPZ bytes differ from the completed collector receipt")
    if args.max_part_bytes <= 0 or args.max_part_bytes > 7_000_000:
        raise ValueError("part size must be positive and no greater than 7,000,000 bytes")

    parts_dir = output_dir / "roi_geometry_transfer_parts"
    parts_dir.mkdir(parents=False, exist_ok=False)
    expected_count = math.ceil(expected_bytes / args.max_part_bytes)
    parts: list[dict[str, Any]] = []
    with npz_path.open("rb") as source:
        for index in range(expected_count):
            filename = f"roi_geometry_selected_frames.part{index:03d}"
            part_path = parts_dir / filename
            digest = hashlib.sha256()
            written = 0
            with part_path.open("xb") as target:
                remaining = min(args.max_part_bytes, expected_bytes - index * args.max_part_bytes)
                while remaining:
                    block = source.read(min(1 << 20, remaining))
                    if not block:
                        raise RuntimeError(f"unexpected NPZ EOF while writing part {index}")
                    target.write(block)
                    digest.update(block)
                    written += len(block)
                    remaining -= len(block)
                target.flush()
                os.fsync(target.fileno())
            parts.append({"index": index, "filename": filename, "bytes": written,
                          "sha256": digest.hexdigest()})
        if source.read(1):
            raise RuntimeError("unpartitioned bytes remain after the expected chunk count")
    after = {"bytes": npz_path.stat().st_size, "sha256": sha256_file(npz_path)}
    if before != after:
        raise RuntimeError("source NPZ changed while creating transport copies")
    if sum(item["bytes"] for item in parts) != expected_bytes:
        raise RuntimeError("part byte counts do not sum to the source NPZ size")

    manifest: dict[str, Any] = {
        "schema": "cross_tfg_roi_geometry_transport_parts_v1",
        "status": "PASS_READ_ONLY_CHUNK_COPY",
        "started_unix": started,
        "source_npz": {"path": str(npz_path), "bytes": expected_bytes, "sha256": expected_sha},
        "source_before": before,
        "source_after": after,
        "max_part_bytes": args.max_part_bytes,
        "part_count": len(parts),
        "parts": parts,
        "source_receipt_sha256": sha256_file(receipt_path),
    }
    manifest["finished_unix"] = time.time()
    manifest["payload_sha256"] = canonical_sha(manifest)
    manifest_path = output_dir / "roi_geometry_transfer_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "manifest_path": str(manifest_path),
                      "manifest_bytes": manifest_path.stat().st_size,
                      "manifest_sha256": sha256_file(manifest_path),
                      "payload_sha256": manifest["payload_sha256"],
                      "source_bytes": expected_bytes, "source_sha256": expected_sha,
                      "part_count": len(parts), "parts": parts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
