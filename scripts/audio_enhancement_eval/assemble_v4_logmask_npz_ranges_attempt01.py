#!/usr/bin/env python3
"""Validate and assemble read-only bounded remote byte-range responses."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_INDEX_SHA256 = "6f01ed1548aa2e3f2eb9339e637afc8f8bde0beeda29532b6054c798a303b7d8"
EXPECTED_NPZ_SHA256 = "69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be"
EXPECTED_NPZ_BYTES = 46_753_644


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    index_path = args.index.resolve(strict=True)
    index, index_raw = load_json(index_path)
    index_sha = sha256(index_raw)
    if index_sha != EXPECTED_INDEX_SHA256:
        raise RuntimeError(f"index SHA mismatch: {index_sha}")
    source = index["source"]
    if source.get("sha256") != EXPECTED_NPZ_SHA256 or source.get("bytes") != EXPECTED_NPZ_BYTES:
        raise RuntimeError("index source seal differs from the frozen NPZ")
    requests = index.get("requests")
    if not isinstance(requests, list) or len(requests) != 16:
        raise RuntimeError("range index must declare exactly 16 chunks")

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise RuntimeError(f"output directory must be fresh: {out_dir}")

    responses_dir = args.responses_dir.resolve(strict=True)
    blocks: list[tuple[int, bytes]] = []
    audit: list[dict[str, Any]] = []
    common_stat: dict[str, Any] | None = None
    expected_offset = 0
    for expected_i, req in enumerate(requests):
        req_path = Path(req["path"]).resolve(strict=True)
        req_raw = req_path.read_bytes()
        if len(req_raw) != req["bytes"] or sha256(req_raw) != req["sha256"]:
            raise RuntimeError(f"request source seal mismatch: {req_path}")
        response_path = responses_dir / (req_path.name + ".response.json")
        outer, outer_raw = load_json(response_path)
        trunc = outer.get("truncated")
        trunc_ok = (trunc is False) or (isinstance(trunc, dict) and
                                        trunc.get("stdout") is False and trunc.get("stderr") is False)
        if (outer.get("ok") is not True or outer.get("code") != 0 or
                outer.get("timed_out") is not False or not trunc_ok):
            raise RuntimeError(f"outer transport result is not complete PASS: chunk {expected_i}")
        inner_raw = outer.get("stdout")
        if not isinstance(inner_raw, str):
            raise RuntimeError(f"missing stdout JSON for chunk {expected_i}")
        try:
            inner = json.loads(inner_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"inner response JSON invalid for chunk {expected_i}: chars={len(inner_raw)} error={exc}") from exc
        if inner.get("status") != "PASS":
            raise RuntimeError(f"remote range did not report PASS: chunk {expected_i}")
        offset = req.get("offset")
        length = req.get("length")
        if (inner.get("chunk_index") != expected_i or inner.get("offset") != offset or
                inner.get("length") != length or offset != expected_offset):
            raise RuntimeError(f"range mapping mismatch for chunk {expected_i}")
        for key, expected in (("source_bytes", EXPECTED_NPZ_BYTES),
                              ("source_sha256", EXPECTED_NPZ_SHA256),
                              ("source_path", source["path"])):
            if inner.get(key) != expected:
                raise RuntimeError(f"remote source binding mismatch in chunk {expected_i}: {key}")
        stat = inner.get("file_stat")
        if not isinstance(stat, dict) or stat.get("size") != EXPECTED_NPZ_BYTES:
            raise RuntimeError(f"missing remote file identity in chunk {expected_i}")
        if common_stat is None:
            common_stat = stat
        elif stat != common_stat:
            raise RuntimeError(f"remote file identity changed across chunks at {expected_i}")
        encoded = inner.get("chunk_base64")
        if not isinstance(encoded, str):
            raise RuntimeError(f"missing base64 chunk {expected_i}")
        raw_chunk = base64.b64decode(encoded, validate=True)
        if len(raw_chunk) != length or sha256(raw_chunk) != inner.get("chunk_sha256"):
            raise RuntimeError(f"chunk byte count/SHA mismatch: {expected_i}")
        blocks.append((offset, raw_chunk))
        audit.append({"chunk_index": expected_i, "offset": offset, "bytes": len(raw_chunk),
                      "sha256": sha256(raw_chunk), "request_path": str(req_path),
                      "request_sha256": req["sha256"], "response_path": str(response_path),
                      "response_bytes": len(outer_raw), "response_sha256": sha256(outer_raw),
                      "outer_transport_pass": True, "inner_range_pass": True})
        expected_offset += len(raw_chunk)

    if expected_offset != EXPECTED_NPZ_BYTES:
        raise RuntimeError(f"assembled length mismatch: {expected_offset}")
    joined = b"".join(chunk for _, chunk in blocks)
    if len(joined) != EXPECTED_NPZ_BYTES or sha256(joined) != EXPECTED_NPZ_SHA256:
        raise RuntimeError("assembled NPZ bytes do not match the frozen whole-file seal")
    out_dir.mkdir(parents=True)
    npz_path = out_dir / "v4_logmask_arrays.npz"
    with npz_path.open("xb") as stream:
        stream.write(joined)
        stream.flush()
    receipt = {"schema": "v4_logmask_npz_range_reassembly_v1", "status": "PASS",
               "source_path": source["path"], "source_bytes": EXPECTED_NPZ_BYTES,
               "source_sha256": EXPECTED_NPZ_SHA256, "index_path": str(index_path),
               "index_sha256": index_sha, "remote_file_stat": common_stat,
               "range_count": len(audit), "ranges": audit,
               "local_output_path": str(npz_path), "local_output_bytes": npz_path.stat().st_size,
               "local_output_sha256": hash_file(npz_path), "scope": index.get("scope")}
    receipt_path = out_dir / "reassembly_receipt.json"
    receipt_raw = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n").encode("utf-8")
    with receipt_path.open("xb") as stream:
        stream.write(receipt_raw)
        stream.flush()
    print(json.dumps({"status": "PASS", "ranges": len(audit), "bytes": len(joined),
                      "npz_sha256": hash_file(npz_path), "receipt_sha256": sha256(receipt_raw),
                      "confirmation_audio_read": False}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
