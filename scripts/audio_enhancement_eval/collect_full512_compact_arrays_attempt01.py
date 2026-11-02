"""Read-only compaction of the full512 parity run's matrices/embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "cross_tfg_full512_compact_arrays_v1"
EXPECTED_INPUT_MANIFEST_SHA256 = "f357982f89c25854313d7641a37db80bcbff80bc26153a73657cc4e352882dfe"
EXPECTED_SUMMARY_RAW_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
ARRAY_NAMES = (
    "proxy_distance_by_query_lag_float32",
    "native_distance_by_query_lag_float32",
    "proxy_video_embedding_float32",
    "native_video_embedding_center_float32",
    "proxy_audio_embedding_full_float32",
    "proxy_audio_embedding_aligned_segment_float32",
    "native_audio_embedding_full_float32",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    if sha256_file(args.input_manifest) != EXPECTED_INPUT_MANIFEST_SHA256:
        raise SystemExit("input manifest SHA256 mismatch")
    if args.output_root.exists():
        raise SystemExit("output root already exists; refusing overwrite")
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cross_tfg_full512_compact_array_inputs_v1":
        raise SystemExit("unexpected input manifest schema")
    if manifest.get("summary_raw_sha256") != EXPECTED_SUMMARY_RAW_SHA256:
        raise SystemExit("full parity summary pin mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 8:
        raise SystemExit("expected exactly eight pinned input arrays")
    row_ids = [Path(str(item["path"])).parent.parent.name for item in inputs]
    if len(set(row_ids)) != 8:
        raise SystemExit("input rows are not unique")

    args.output_root.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, Any]] = []
    for row_id, item in zip(row_ids, inputs, strict=True):
        source = Path(item["path"])
        expected_bytes = int(item["bytes"])
        expected_sha = str(item["sha256"])
        if not source.is_file() or source.stat().st_size != expected_bytes:
            raise SystemExit(f"pinned source file missing or wrong-sized: {row_id}")
        source_sha = sha256_file(source)
        if source_sha != expected_sha:
            raise SystemExit(f"pinned source SHA256 mismatch: {row_id}")
        with np.load(source, allow_pickle=False) as archive:
            missing = sorted(set(ARRAY_NAMES) - set(archive.files))
            if missing:
                raise SystemExit(f"missing required arrays for {row_id}: {missing}")
            arrays: dict[str, np.ndarray] = {}
            array_receipts: dict[str, dict[str, Any]] = {}
            for name in ARRAY_NAMES:
                value = np.ascontiguousarray(archive[name])
                if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                    raise SystemExit(f"non-finite or non-float32 array {name} for {row_id}")
                if name.endswith("distance_by_query_lag_float32"):
                    if value.ndim != 2 or value.shape[1] != 31:
                        raise SystemExit(f"distance matrix is not N x 31 for {row_id}")
                elif "video_embedding" in name:
                    if value.ndim != 2 or value.shape[1] != 1024:
                        raise SystemExit(f"video embedding is not N x 1024 for {row_id}")
                elif value.ndim != 2 or value.shape[1] != 1024:
                    raise SystemExit(f"audio embedding is not N x 1024 for {row_id}")
                arrays[name] = value
                array_receipts[name] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "array_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                }
        output = args.output_root / f"{row_id}_compact_arrays.npz"
        np.savez_compressed(output, **arrays)
        source_after_bytes = source.stat().st_size
        source_after_sha = sha256_file(source)
        if source_after_bytes != expected_bytes or source_after_sha != expected_sha:
            raise SystemExit(f"source changed while compacting: {row_id}")
        output_rows.append({
            "row_id": row_id,
            "source_path": str(source),
            "source_bytes": expected_bytes,
            "source_sha256": source_sha,
            "source_after_bytes": source_after_bytes,
            "source_after_sha256": source_after_sha,
            "source_unchanged": True,
            "output_path": str(output),
            "output_bytes": output.stat().st_size,
            "output_sha256": sha256_file(output),
            "arrays": array_receipts,
        })

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "read-only selection of saved matrices and embeddings; no forward pass or metric recomputation",
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "summary_raw_sha256": EXPECTED_SUMMARY_RAW_SHA256,
        "array_names": list(ARRAY_NAMES),
        "rows": output_rows,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_root / "compact_collection_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "status": "PASS",
        "rows": len(output_rows),
        "receipt_path": str(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
        "receipt_sha256": sha256_file(receipt_path),
        "payload_sha256": receipt["payload_sha256"],
        "output_bytes_total": sum(int(row["output_bytes"]) for row in output_rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
