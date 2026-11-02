#!/usr/bin/env python3
"""Reassemble the pinned NPZ from attempt04 plus attempt05 split ranges."""
from __future__ import annotations

import base64
import hashlib
import json
import argparse
from pathlib import Path
from typing import Any

BASE_INDEX_SHA256 = "6f01ed1548aa2e3f2eb9339e637afc8f8bde0beeda29532b6054c798a303b7d8"
SUPPLEMENT_INDEX_SHA256 = "3072097c76f941846a1fd6c1baf5921893e1e41b88b11f5004d88b115d672518"
NPZ_SHA256 = "69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be"
NPZ_BYTES = 46_753_644


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_obj(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return obj, raw


def checked_chunk(req_path: Path, response_path: Path, expected_remote_index: int,
                  expected_offset: int, expected_length: int,
                  expected_source_path: str) -> tuple[bytes, dict[str, Any]]:
    req, req_raw = read_obj(req_path)
    outer, outer_raw = read_obj(response_path)
    trunc = outer.get("truncated")
    trunc_ok = trunc is False or (isinstance(trunc, dict) and trunc.get("stdout") is False and trunc.get("stderr") is False)
    if outer.get("ok") is not True or outer.get("code") != 0 or outer.get("timed_out") is not False or not trunc_ok:
        raise RuntimeError(f"incomplete remote transport response: {response_path}")
    stdout = outer.get("stdout")
    if not isinstance(stdout, str):
        raise RuntimeError(f"missing stdout payload: {response_path}")
    try:
        inner = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"truncated/invalid inner response JSON: {response_path}: {exc}") from exc
    if inner.get("status") != "PASS" or inner.get("chunk_index") != expected_remote_index:
        raise RuntimeError(f"remote chunk status/index mismatch: {response_path}")
    if inner.get("offset") != expected_offset or inner.get("length") != expected_length:
        raise RuntimeError(f"remote chunk range mismatch: {response_path}")
    if inner.get("source_path") != expected_source_path or inner.get("source_bytes") != NPZ_BYTES or inner.get("source_sha256") != NPZ_SHA256:
        raise RuntimeError(f"remote source binding mismatch: {response_path}")
    stat = inner.get("file_stat")
    if not isinstance(stat, dict) or stat.get("size") != NPZ_BYTES:
        raise RuntimeError(f"missing or invalid remote source identity: {response_path}")
    data = base64.b64decode(inner["chunk_base64"], validate=True)
    if len(data) != expected_length or sha(data) != inner.get("chunk_sha256"):
        raise RuntimeError(f"range bytes/SHA mismatch: {response_path}")
    return data, {"request_path": str(req_path.resolve()), "request_bytes": len(req_raw),
                  "request_sha256": sha(req_raw), "response_path": str(response_path.resolve()),
                  "response_bytes": len(outer_raw), "response_sha256": sha(outer_raw),
                  "offset": expected_offset, "bytes": len(data), "remote_chunk_index": expected_remote_index,
                  "chunk_sha256": sha(data), "file_stat": stat}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--base-responses-dir", type=Path, required=True)
    parser.add_argument("--supplement-index", type=Path, required=True)
    parser.add_argument("--supplement-responses-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_index.resolve(strict=True)
    supplement_path = args.supplement_index.resolve(strict=True)
    base, base_raw = read_obj(base_path)
    supplement, supplement_raw = read_obj(supplement_path)
    if sha(base_raw) != BASE_INDEX_SHA256 or sha(supplement_raw) != SUPPLEMENT_INDEX_SHA256:
        raise RuntimeError("range index seal mismatch")
    src = base.get("source", {})
    if src.get("sha256") != NPZ_SHA256 or src.get("bytes") != NPZ_BYTES:
        raise RuntimeError("base index source seal mismatch")
    if supplement.get("source", {}).get("sha256") != NPZ_SHA256 or supplement.get("source", {}).get("bytes") != NPZ_BYTES:
        raise RuntimeError("supplement index source seal mismatch")
    source_path = src["path"]
    base_requests = base.get("requests")
    if not isinstance(base_requests, list) or len(base_requests) != 16:
        raise RuntimeError("base index must contain 16 original ranges")
    new_requests = supplement.get("new_ranges")
    if not isinstance(new_requests, list) or len(new_requests) != 3:
        raise RuntimeError("supplement must contain three 1-MB ranges")

    chunks: list[tuple[int, bytes, dict[str, Any]]] = []
    for idx, r in enumerate(base_requests):
        if idx == 11:
            continue
        req_path = Path(r["path"]).resolve(strict=True)
        response_path = Path(str(req_path) + ".response.json").resolve(strict=True)
        req_obj, req_raw = read_obj(req_path)
        if len(req_raw) != r["bytes"] or sha(req_raw) != r["sha256"]:
            raise RuntimeError(f"original request pin mismatch: {idx}")
        block, audit = checked_chunk(req_path, response_path, idx, r["offset"], r["length"], source_path)
        chunks.append((r["offset"], block, audit))

    for entry in new_requests:
        req_path = Path(entry["request_path"]).resolve(strict=True)
        response_path = Path(entry["response_path"]).resolve(strict=True)
        block, audit = checked_chunk(req_path, response_path, entry["remote_chunk_index"],
                                     entry["offset"], entry["length"], source_path)
        if len(block) != 1_000_000:
            raise RuntimeError("supplement block is not exactly 1,000,000 bytes")
        chunks.append((entry["offset"], block, audit))

    chunks.sort(key=lambda x: x[0])
    offset = 0
    identity = None
    for start, data, audit in chunks:
        if start != offset:
            raise RuntimeError(f"noncontiguous range coverage at {start}, expected {offset}")
        if identity is None:
            identity = audit["file_stat"]
        elif audit["file_stat"] != identity:
            raise RuntimeError("remote source stat changed across reads")
        offset += len(data)
    if offset != NPZ_BYTES:
        raise RuntimeError(f"range coverage length mismatch: {offset}")
    raw = b"".join(data for _, data, _ in chunks)
    if sha(raw) != NPZ_SHA256:
        raise RuntimeError("reassembled whole-file SHA mismatch")

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise RuntimeError(f"output directory must be new: {out_dir}")
    out_dir.mkdir(parents=True)
    npz_path = out_dir / "v4_logmask_arrays.npz"
    with npz_path.open("xb") as f:
        f.write(raw)
        f.flush()
    receipt = {"schema": "v4_logmask_npz_range_reassembly_v2", "status": "PASS",
               "source_path": source_path, "source_bytes": NPZ_BYTES, "source_sha256": NPZ_SHA256,
               "base_index_path": str(base_path), "base_index_sha256": sha(base_raw),
               "supplement_index_path": str(supplement_path), "supplement_index_sha256": sha(supplement_raw),
               "remote_file_stat": identity, "range_count": len(chunks),
               "ranges": [a for _, _, a in chunks], "local_output_path": str(npz_path),
               "local_output_bytes": npz_path.stat().st_size, "local_output_sha256": sha_file(npz_path),
               "scope": supplement.get("scope"), "confirmation_audio_read": False}
    receipt_path = out_dir / "reassembly_receipt.json"
    receipt_raw = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with receipt_path.open("xb") as f:
        f.write(receipt_raw)
        f.flush()
    print(json.dumps({"status": "PASS", "range_count": len(chunks), "bytes": len(raw),
                      "npz_sha256": sha(raw), "receipt_sha256": sha(receipt_raw),
                      "confirmation_audio_read": False}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
