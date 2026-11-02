#!/usr/bin/env python3
"""Verify and restore the sealed MuseTalk Stage-B score-matrix export."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_EXPORT_BYTES = 3_618_123
EXPECTED_EXPORT_SHA256 = "874211e28f50846dc4fbac8afb7d36489d47cb26ade021be3d1c41b24c047245"
EXPECTED_REPEAT_NPZ_SHA256 = "d0e62a513688aab5b51512f5317a62e169880af1b9ce7031a87b8c1c54189db5"
EXPECTED_RUN_ID = "cross_tfg_audio_head_20261003_musetalk_stageb_attempt04"
EXPECTED_PLAN_SHA256 = "d8414f0c41919b627bbbc24f26b97e989c0a059e82164ebe3f34b2dce34b9116"
EXPECTED_RESULTS_SHA256 = "a5556d7821286669cee4f741de88443b05172584121fa7c2643f5c9b4660ba55"
EXPECTED_RESULT_MANIFEST_SHA256 = "d064ff327b23d3d7dfb7be559fd17b62a87c88d7bcc7a19aa04ab2b1d052a4ce"
EXPECTED_REPEAT_JSONL_SHA256 = "8ed516751a1f0504b8260dc0eda11e7ecc7269be81e31161b069a5df7aaa3892"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def npz_array_seals(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = np.ascontiguousarray(archive[key])
            records[key] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "array_sha256": sha_bytes(array.tobytes(order="C")),
            }
    return records


def json_from_member(tar: tarfile.TarFile, name: str) -> Any:
    member = tar.getmember(name)
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"archive member cannot be read: {name}")
    return json.loads(stream.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--repeat-npz", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    repeat_npz_path = args.repeat_npz.resolve()
    destination = args.destination.resolve()
    receipt_path = args.receipt.resolve()
    archive_sha, archive_bytes = sha_file(archive_path)
    if (archive_sha, archive_bytes) != (EXPECTED_EXPORT_SHA256, EXPECTED_EXPORT_BYTES):
        raise ValueError("raw-matrix export archive size/SHA mismatch")
    repeat_sha, repeat_bytes = sha_file(repeat_npz_path)
    if repeat_sha != EXPECTED_REPEAT_NPZ_SHA256:
        raise ValueError("separately downloaded repeat-QC matrix SHA mismatch")
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite extraction destination or receipt")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate paths in raw-matrix archive")
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"unsafe or non-regular archive member: {member.name}")
        export_manifest = json_from_member(tar, "export_manifest.json")
        if export_manifest.get("run_id") != EXPECTED_RUN_ID:
            raise ValueError("raw-matrix export run_id mismatch")
        member_seals = export_manifest.get("archive_members")
        if not isinstance(member_seals, dict):
            raise ValueError("export manifest has no archive member seals")
        if set(member_seals) | {"export_manifest.json"} != set(names):
            raise ValueError("archive member inventory differs from export manifest")
        expected_sources = {
            "producer/plan.json": EXPECTED_PLAN_SHA256,
            "producer/results.jsonl": EXPECTED_RESULTS_SHA256,
            "producer/result_manifest.json": EXPECTED_RESULT_MANIFEST_SHA256,
            "producer/generation_repeat_qc.jsonl": EXPECTED_REPEAT_JSONL_SHA256,
        }
        for name, expected in expected_sources.items():
            seal = member_seals.get(name)
            member = tar.getmember(name)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read producer source member {name}")
            content = stream.read()
            if sha_bytes(content) != expected or seal.get("sha256") != expected or len(content) != seal.get("bytes"):
                raise ValueError(f"producer source seal mismatch: {name}")
        plan = json_from_member(tar, "producer/plan.json")
        manifest = json_from_member(tar, "producer/result_manifest.json")
        audit = export_manifest.get("consumer_audit")
        if plan.get("run_id") != EXPECTED_RUN_ID or manifest.get("run_id") != EXPECTED_RUN_ID:
            raise ValueError("producer plan/manifest run_id mismatch")
        if plan.get("plan_sha256") != sha_bytes(canonical({key: value for key, value in plan.items() if key != "plan_sha256"})):
            raise ValueError("producer plan payload self-seal mismatch")
        if manifest.get("manifest_sha256") != sha_bytes(canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})):
            raise ValueError("producer result-manifest payload self-seal mismatch")
        if manifest.get("status") != "PASS" or manifest.get("planned_rows") != 240 or manifest.get("result_rows") != 240:
            raise ValueError("producer result manifest is not the sealed 240-row PASS run")
        if not isinstance(audit, dict) or audit.get("issues") != [] or audit.get("verified_scored_row_count") != 240:
            raise ValueError("independent consumer audit is incomplete or reports issues")
        if audit.get("status") != "INCOMPLETE":
            raise ValueError("expected the known GUARD20 coverage-limited audit status")
        if (audit.get("plan_raw_sha256") != EXPECTED_PLAN_SHA256
                or audit.get("results_raw_sha256") != EXPECTED_RESULTS_SHA256
                or audit.get("result_manifest_raw_sha256") != EXPECTED_RESULT_MANIFEST_SHA256
                or audit.get("generation_repeat_qc_raw_sha256") != EXPECTED_REPEAT_JSONL_SHA256):
            raise ValueError("consumer audit does not bind the exact producer artifacts")

        consumer_manifest = json_from_member(tar, "consumer/consumer_manifest.json")
        if consumer_manifest.get("run_id") != EXPECTED_RUN_ID or consumer_manifest.get("status") != audit.get("status"):
            raise ValueError("consumer manifest run/status does not match audit")
        if consumer_manifest.get("manifest_sha256") != sha_bytes(canonical({
            key: value for key, value in consumer_manifest.items() if key != "manifest_sha256"
        })):
            raise ValueError("consumer-manifest payload self-seal mismatch")
        consumer_files = consumer_manifest.get("files")
        if not isinstance(consumer_files, list):
            raise ValueError("consumer manifest has no output file inventory")
        for item in consumer_files:
            member_name = f"consumer/{item['path']}"
            seal = member_seals.get(member_name)
            if seal is None or seal.get("sha256") != item.get("sha256") or seal.get("bytes") != item.get("bytes"):
                raise ValueError(f"consumer output inventory mismatch: {item.get('path')}")

        results_member = tar.extractfile(tar.getmember("producer/results.jsonl"))
        if results_member is None:
            raise ValueError("missing producer results JSONL")
        result_rows = [json.loads(line) for line in results_member.read().splitlines() if line.strip()]
        matrix_records = export_manifest.get("distance_matrix_rows", [])
        if len(result_rows) != 240 or len(matrix_records) != 240:
            raise ValueError("expected 240 result and distance-matrix rows")
        rows_by_id = {row["row_id"]: row for row in result_rows}
        if len(rows_by_id) != 240:
            raise ValueError("duplicate producer result row IDs")
        for record in matrix_records:
            row = rows_by_id.get(record.get("row_id"))
            if row is None or row.get("status") != "PASS" or record.get("status") != "PASS":
                raise ValueError(f"unmatched or non-PASS matrix record: {record.get('row_id')}")
            ref = row.get("score", {}).get("raw_matrix_npz", {})
            if ref.get("file_sha256") != record.get("file_sha256") or ref.get("arrays") != record.get("arrays"):
                raise ValueError(f"matrix reference differs from immutable producer result: {record['row_id']}")
            if record.get("member") != f"raw_matrix_npz/{record['row_id']}.npz":
                raise ValueError(f"unexpected matrix archive member name: {record['row_id']}")
            seal = member_seals[record["member"]]
            if seal.get("sha256") != record.get("file_sha256") or seal.get("bytes") != record.get("bytes"):
                raise ValueError(f"matrix archive/member seal mismatch: {record['row_id']}")

        # Restore without tar.extract() so only the previously validated regular files are written.
        destination.mkdir(parents=True, exist_ok=False)
        restored = []
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"cannot restore member {member.name}")
            payload = source.read()
            target.write_bytes(payload)
            restored.append({"path": member.name, "bytes": len(payload), "sha256": sha_bytes(payload)})

    # Validate every restored matrix against both the sealed NPZ reference and its array-level seals.
    restored_by_name = {row["path"]: row for row in restored}
    for record in matrix_records:
        member = record["member"]
        path = destination.joinpath(*PurePosixPath(member).parts)
        actual_arrays = npz_array_seals(path)
        if actual_arrays != record["arrays"]:
            raise ValueError(f"restored NPZ array seal mismatch: {record['row_id']}")
        if restored_by_name[member]["sha256"] != record["file_sha256"]:
            raise ValueError(f"restored NPZ file seal mismatch: {record['row_id']}")

    repeat_rows_path = destination / "producer" / "generation_repeat_qc.jsonl"
    repeat_rows = [json.loads(line) for line in repeat_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(repeat_rows) != 1:
        raise ValueError("expected exactly one generation-repeat-QC row")
    repeat_ref = repeat_rows[0].get("raw_matrix_npz", {})
    if repeat_ref.get("file_sha256") != repeat_sha:
        # Some producer QCs nest this exact ref; find the sealed mapping without guessing a media path.
        consumer_repeat = ((export_manifest.get("consumer_audit") or {}).get("generation_repeat_qc") or [{}])[0]
        repeat_ref = consumer_repeat.get("raw_matrix_npz", {})
    if repeat_ref.get("file_sha256") != repeat_sha or repeat_ref.get("arrays") != npz_array_seals(repeat_npz_path):
        raise ValueError("repeat-QC NPZ does not match the sealed producer/consumer reference")

    receipt = {
        "schema": "musetalk_stageb_raw_matrix_restore_readback_v1",
        "run_id": EXPECTED_RUN_ID,
        "archive": {"path": str(archive_path), "bytes": archive_bytes, "sha256": archive_sha},
        "repeat_matrix": {"path": str(repeat_npz_path), "bytes": repeat_bytes, "sha256": repeat_sha},
        "restored_root": str(destination),
        "restored_member_count": len(restored),
        "matrix_row_count": len(matrix_records),
        "repeat_matrix_row_count": 1,
        "producer_seals": {
            "plan_raw_sha256": EXPECTED_PLAN_SHA256,
            "results_raw_sha256": EXPECTED_RESULTS_SHA256,
            "result_manifest_raw_sha256": EXPECTED_RESULT_MANIFEST_SHA256,
            "repeat_qc_raw_sha256": EXPECTED_REPEAT_JSONL_SHA256,
        },
        "all_matrix_file_and_array_seals_verified": True,
        "all_producer_rows_pass": True,
        "consumer_issue_count": 0,
        "consumer_status": audit["status"],
        "guard20_status": "INCOMPLETE_COVERAGE_RETAINED",
        "restored_members": restored,
    }
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
    print(json.dumps({"status": "PASS", "receipt": str(receipt_path), "matrix_rows": len(matrix_records), "members": len(restored)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
