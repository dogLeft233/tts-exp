#!/usr/bin/env python3
"""Verify and reassemble a read-only chunk transfer of the ROI NPZ."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collector-receipt", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-receipt", type=Path, required=True)
    args = parser.parse_args()
    for target in (args.output, args.restore_receipt):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite local readback target: {target}")
    started = time.time()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    collector = json.loads(args.collector_receipt.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_READ_ONLY_CHUNK_COPY":
        raise RuntimeError("transfer manifest is not PASS_READ_ONLY_CHUNK_COPY")
    if collector.get("status") != "PASS_READ_ONLY_SAVED_FRAMES":
        raise RuntimeError("collector receipt is not PASS_READ_ONLY_SAVED_FRAMES")
    manifest_payload = dict(manifest)
    payload_sha = manifest_payload.pop("payload_sha256", None)
    if payload_sha != canonical_sha(manifest_payload):
        raise RuntimeError("transfer manifest canonical payload digest mismatch")
    source = manifest.get("source_npz", {})
    selected = collector.get("selected_arrays_npz", {})
    if (source.get("bytes") != selected.get("bytes")
            or source.get("sha256") != selected.get("sha256")
            or source.get("path") != selected.get("path")
            or manifest.get("source_receipt_sha256") != sha256_file(args.collector_receipt)):
        raise RuntimeError("transfer manifest is not bound to this collector receipt")
    if (manifest.get("source_before") != {"bytes": source.get("bytes"), "sha256": source.get("sha256")}
            or manifest.get("source_after") != manifest.get("source_before")):
        raise RuntimeError("transfer manifest does not show an unchanged source NPZ")
    if not isinstance(manifest.get("parts"), list) or len(manifest["parts"]) != int(manifest.get("part_count", -1)):
        raise RuntimeError("transfer manifest part inventory is malformed")
    ordered = sorted(manifest["parts"], key=lambda part: int(part.get("index", -1)))
    if [int(part.get("index", -1)) for part in ordered] != list(range(len(ordered))):
        raise RuntimeError("transfer part indices are not contiguous from zero")
    if sum(int(part.get("bytes", -1)) for part in ordered) != int(source["bytes"]):
        raise RuntimeError("transfer part sizes do not sum to the sealed NPZ size")
    max_part_bytes = int(manifest.get("max_part_bytes", -1))
    if max_part_bytes <= 0 or max_part_bytes > 6_500_000:
        raise RuntimeError("transfer manifest declares an unsupported part size")
    part_readback: list[dict[str, Any]] = []
    temp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    whole = hashlib.sha256()
    total = 0
    with temp.open("xb") as target:
        for part in ordered:
            filename = str(part.get("filename", ""))
            if not filename.startswith("roi_geometry_selected_frames.part") or Path(filename).name != filename:
                raise RuntimeError("transfer part filename is not a safe expected basename")
            path = args.parts_dir / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"downloaded part is not a regular file: {filename}")
            observed_bytes = path.stat().st_size
            if observed_bytes <= 0 or observed_bytes > max_part_bytes:
                raise RuntimeError(f"downloaded part size is outside the declared limit: {filename}")
            observed_sha = sha256_file(path)
            if observed_bytes != int(part["bytes"]) or observed_sha != part["sha256"]:
                raise RuntimeError(f"downloaded part failed its size/SHA seal: {filename}")
            part_readback.append({"filename": filename, "bytes": observed_bytes, "sha256": observed_sha})
            with path.open("rb") as source_stream:
                for block in iter(lambda: source_stream.read(1 << 20), b""):
                    target.write(block)
                    whole.update(block)
                    total += len(block)
        target.flush()
        os.fsync(target.fileno())
    if total != int(source["bytes"]) or whole.hexdigest() != source["sha256"]:
        temp.unlink(missing_ok=True)
        raise RuntimeError("reassembled NPZ differs from sealed collector bytes")
    if args.output.exists():
        temp.unlink(missing_ok=True)
        raise FileExistsError(f"local restore output appeared during reassembly: {args.output}")
    os.rename(temp, args.output)
    receipt: dict[str, Any] = {
        "schema": "cross_tfg_roi_geometry_local_chunk_restore_v1",
        "status": "PASS_REASSEMBLED_EXACT_COLLECTOR_BYTES",
        "started_unix": started,
        "finished_unix": time.time(),
        "collector_receipt": {"path": str(args.collector_receipt),
                               "bytes": args.collector_receipt.stat().st_size,
                               "sha256": sha256_file(args.collector_receipt)},
        "transfer_manifest": {"path": str(args.manifest), "bytes": args.manifest.stat().st_size,
                              "sha256": sha256_file(args.manifest), "payload_sha256": payload_sha},
        "parts_dir": str(args.parts_dir),
        "parts": part_readback,
        "reassembled_npz": {"path": str(args.output), "bytes": total, "sha256": whole.hexdigest()},
    }
    receipt["payload_sha256"] = canonical_sha(receipt)
    write_json(args.restore_receipt, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "bytes": total,
                      "sha256": whole.hexdigest(), "receipt": str(args.restore_receipt),
                      "receipt_sha256": sha256_file(args.restore_receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
