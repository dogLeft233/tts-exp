#!/usr/bin/env python3
"""Build a source-only, transfer-ready fit40 package without running a method.

The package contains exactly the 40 sealed PCM16 WAVs, the original sealed fit40
transfer manifest, and a binding registry. It is intentionally not an executable
candidate plan, and it excludes enhancement implementations and final method
configuration files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02"
SOURCE_DIR = PREP / "fit40_transfer_bundle_attempt01"
MANIFEST_PATH = SOURCE_DIR / "fit40_source_transfer_manifest.json"
QC_RECEIPT_PATH = PREP / "fit40_independent_numeric_qc_attempt03/fit40_independent_input_numeric_qc_receipt.json"
QC_ARRAYS_PATH = PREP / "fit40_independent_numeric_qc_attempt03/fit40_independent_power_flux_arrays.npz"
DEFAULT_OUTPUT = PREP / "fit40_source_stage_attempt02"
EXPECTED_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_QC_RECEIPT_SHA256 = "4a47124b34cf7684d5d236339e957a570baed2cef9871529b8ed7eed3f109ca9"
EXPECTED_QC_ARRAYS_SHA256 = "544199f39058505711779112779c369b487f0bff62584c8c8afc4ca11ae957ed"
MAX_PACKAGE_BYTES = 8 * 1024 * 1024
FIT_SPEAKERS = tuple(f"S{value:04d}" for value in range(52, 72))


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sealed_file(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or symlinked source: {path}")
    digest, size = sha_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise RuntimeError(f"source SHA mismatch for {path}: {digest}")
    return {"path": str(path.resolve()), "bytes": size, "raw_sha256": digest}


def source_registry(manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RuntimeError("source transfer manifest does not have exactly 40 rows")
    if manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise RuntimeError("unexpected source manifest status")
    if manifest.get("split") != "historical_fit_only" or set(manifest.get("fit_speakers", [])) != set(FIT_SPEAKERS):
        raise RuntimeError("fit40 roster/split does not match the frozen historical fit cohort")
    scope = manifest.get("scope", {})
    if scope.get("dev_audio_read") is not False or scope.get("confirm_audio_read") is not False:
        raise RuntimeError("source manifest indicates a dev/confirmation read")
    if scope.get("candidate_transform_executed") is not False or scope.get("model_forward") is not False or scope.get("score") is not False:
        raise RuntimeError("source manifest indicates candidate/model/score execution")

    expected_pairs = {(sid, j) for sid in FIT_SPEAKERS for j in (0, 15)}
    observed_pairs = {(str(row["speaker_id"]), int(row["canonical_j"])) for row in rows}
    if observed_pairs != expected_pairs or len({r["sample_id"] for r in rows}) != 40 or len({r["row_id"] for r in rows}) != 40:
        raise RuntimeError("source rows do not match the exact 40 frozen j=0/j=15 fit endpoints")

    package_rows: list[dict[str, Any]] = []
    package_members: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["transfer_local_path"]).resolve(strict=True)
        if path.parent != SOURCE_DIR.resolve() / "source_wav":
            raise RuntimeError(f"WAV outside the fit40 materialization directory: {path}")
        sealed = sealed_file(path, str(row["transfer_file_sha256"]))
        if sealed["bytes"] != int(row["transfer_file_bytes"]) or sealed["bytes"] != int(row["source_member_bytes"]):
            raise RuntimeError(f"WAV byte count mismatch: {row['row_id']}")
        if sealed["raw_sha256"] != row["source_member_sha256"]:
            raise RuntimeError(f"WAV is not byte-identical to the selected corpus member: {row['row_id']}")
        if int(row["sample_rate_hz"]) != 16000 or int(row["channels"]) != 1 or row["source_encoding"] != "PCM_16":
            raise RuntimeError(f"unexpected source audio contract: {row['row_id']}")
        relative_path = str(row["transfer_relative_path"])
        if relative_path != f"source_wav/{row['sample_id']}.wav":
            raise RuntimeError(f"unexpected package member path: {relative_path}")
        package_rows.append({
            "row_id": row["row_id"],
            "speaker_id": row["speaker_id"],
            "canonical_j": int(row["canonical_j"]),
            "sample_id": row["sample_id"],
            "package_path": relative_path,
            "file_bytes": sealed["bytes"],
            "file_sha256": sealed["raw_sha256"],
            "pcm16_payload_sha256": row["source_pcm_payload_sha256"],
            "canonical_float32_pcm_sha256": row["canonical_float32_pcm_sha256"],
            "sample_count": int(row["sample_count"]),
            "sample_rate_hz": int(row["sample_rate_hz"]),
            "channels": int(row["channels"]),
            "source_encoding": row["source_encoding"],
            "source_archive_path_reference": row["source_archive_path"],
            "source_archive_sha256_reference": row["source_archive_sha256"],
            "source_archive_member": row["source_archive_member"],
            "source_member_sha256": row["source_member_sha256"],
            "source_member_bytes": int(row["source_member_bytes"]),
        })
        package_members.append({**sealed, "package_path": relative_path})

    source_refs: dict[str, Any] = {}
    for name, data in manifest.get("source_implementation_pins", {}).items():
        path = ROOT / str(data["path"])
        source_refs[name] = sealed_file(path, str(data["raw_sha256"]))
    materializer_path = ROOT / str(manifest["created_by"])
    materializer = sealed_file(materializer_path, str(manifest["created_by_raw_sha256"]))
    scan_path = ROOT / str(manifest["selection_scan"]["path"])
    scan = sealed_file(scan_path, str(manifest["selection_scan"]["raw_sha256"]))
    roster_path = PREP / "v5_fit_expansion_roster.json"
    roster = sealed_file(roster_path, str(manifest["selection_scan"]["roster_raw_sha256"]))

    config_ref = manifest.get("r090_control", {})
    runtime_meta = manifest.get("runtime_transfer_metadata", {})
    if config_ref.get("status") != "PREPARED_NOT_FROZEN":
        raise RuntimeError("r090 is expected to remain prepared and unfrozen")
    if runtime_meta.get("max_v5_candidate_or_ablation_methods") != 6 or runtime_meta.get("localcompress_r090_uses_v5_candidate_or_ablation_slot") is not True:
        raise RuntimeError("V5 six-slot budget or r090 slot binding changed")

    qc_receipt = sealed_file(QC_RECEIPT_PATH, EXPECTED_QC_RECEIPT_SHA256)
    qc_arrays = sealed_file(QC_ARRAYS_PATH, EXPECTED_QC_ARRAYS_SHA256)
    qc_doc = json.loads(QC_RECEIPT_PATH.read_bytes().decode("utf-8"))
    if qc_doc.get("status") != "PASS_INDEPENDENT_CANONICAL_PCM_POWER_SIGNED_FLUX_MAPPING_QC" or qc_doc.get("cohort", {}).get("source_rows") != 40:
        raise RuntimeError("independent source-only QC receipt is not the expected 40-row PASS")
    if qc_doc.get("scope", {}).get("dev_audio_read") is not False or qc_doc.get("scope", {}).get("confirm_audio_read") is not False:
        raise RuntimeError("independent QC receipt reports a prohibited split read")

    manifest_seal = sealed_file(MANIFEST_PATH, manifest_sha)
    return {
        "schema": "cross_tfg_v5_fit40_source_binding_registry_v1",
        "status": "SEALED_SOURCE_ONLY_TRANSFER_READY_NOT_EXECUTABLE",
        "source_manifest": manifest_seal,
        "source_rows": package_rows,
        "package_members": package_members,
        "source_provenance": {
            "materializer": materializer,
            "member_scan": scan,
            "frozen_roster": roster,
            "fit_plan_raw_sha256_reference": manifest["selection_scan"]["fit_plan_raw_sha256"],
            "adr013_raw_sha256_reference": manifest["selection_scan"]["adr013_raw_sha256"],
            "implementation_pins": source_refs,
        },
        "input_numeric_qc": {
            "receipt": qc_receipt,
            "arrays": qc_arrays,
            "scope": "input-only numeric consistency; includes derived negative log-magnitude flux; no transform/effect",
        },
        "r090_status": {
            "method_id": "localcompress_r090",
            "status": "PREPARED_NOT_FROZEN",
            "uses_one_of_six_v5_candidate_or_ablation_slots": True,
            "remaining_v5_slots_if_retained": 5,
            "production_config_included_as_final_method": False,
            "r090_implementation_included": False,
        },
        "remote_stage": {
            "namespace_name": "v5_fit40_input_stage_attempt01",
            "namespace_created": False,
            "transfer_performed": False,
            "plan_is_executable": False,
            "model_or_transform_execution": False,
        },
        "scope": {
            "historical_fit_only": True,
            "rows": 40,
            "speakers": 20,
            "dev_audio_read": False,
            "confirm_audio_read": False,
            "source_archives_opened": False,
            "candidate_effect_read": False,
            "transform_executed": False,
            "model_forward": False,
            "render": False,
            "score": False,
            "selection": False,
        },
        "archive_member_policy": "exactly fit40_source_transfer_manifest.json, source_binding_registry.json, and the 40 sealed source_wav/*.wav members; no candidate code or final method configuration",
    }


def build(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fit40 source stage directory: {output_dir}")
    manifest_raw = MANIFEST_PATH.read_bytes()
    manifest_sha = sha_bytes(manifest_raw)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"fit40 source manifest SHA mismatch: {manifest_sha}")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    registry = source_registry(manifest, manifest_sha)
    registry_raw = canonical_json(registry) + b"\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    registry_path = output_dir / "source_binding_registry.json"
    with registry_path.open("xb") as stream:
        stream.write(registry_raw)

    archive_path = output_dir / "fit40_source_stage_attempt02.zip"
    with zipfile.ZipFile(archive_path, mode="x", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.writestr("fit40_source_transfer_manifest.json", manifest_raw, compress_type=zipfile.ZIP_STORED)
        archive.writestr("source_binding_registry.json", registry_raw, compress_type=zipfile.ZIP_STORED)
        for row in registry["source_rows"]:
            source_path = SOURCE_DIR / row["package_path"]
            archive.write(source_path, arcname=row["package_path"], compress_type=zipfile.ZIP_STORED)

    archive_sha, archive_bytes = sha_file(archive_path)
    if archive_bytes > MAX_PACKAGE_BYTES:
        raise RuntimeError(f"source-only stage package exceeds the bridge file limit: {archive_bytes}")
    expected_members = {
        "fit40_source_transfer_manifest.json",
        "source_binding_registry.json",
        *(r["package_path"] for r in registry["source_rows"]),
    }
    member_checks: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("source stage ZIP failed CRC validation")
        names = archive.namelist()
        if len(names) != 42 or set(names) != expected_members:
            raise RuntimeError(f"source stage ZIP member set is wrong: {len(names)} members")
        for info in archive.infolist():
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, mode="r") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
            if info.filename == "fit40_source_transfer_manifest.json":
                expected_sha, expected_size = EXPECTED_MANIFEST_SHA256, len(manifest_raw)
            elif info.filename == "source_binding_registry.json":
                expected_sha, expected_size = sha_bytes(registry_raw), len(registry_raw)
            else:
                row = next(r for r in registry["source_rows"] if r["package_path"] == info.filename)
                expected_sha, expected_size = row["file_sha256"], row["file_bytes"]
            if digest.hexdigest() != expected_sha or size != expected_size:
                raise RuntimeError(f"source stage ZIP member mismatch: {info.filename}")
            member_checks.append({
                "path": info.filename,
                "bytes": size,
                "raw_sha256": digest.hexdigest(),
                "zip_crc32": f"{info.CRC:08x}",
            })

    script_sha, script_bytes = sha_file(Path(__file__).resolve())
    receipt = {
        "schema": "cross_tfg_v5_fit40_source_stage_package_receipt_v1",
        "status": "PASS_SOURCE_ONLY_FIT40_TRANSFER_PACKAGE_SEALED",
        "created_by": str(Path(__file__).resolve()),
        "created_by_bytes": script_bytes,
        "created_by_raw_sha256": script_sha,
        "package": {"path": str(archive_path.resolve()), "bytes": archive_bytes, "raw_sha256": archive_sha, "bridge_limit_bytes": MAX_PACKAGE_BYTES},
        "source_manifest": {"path": str(MANIFEST_PATH.resolve()), "bytes": len(manifest_raw), "raw_sha256": manifest_sha},
        "binding_registry": {"path": str(registry_path.resolve()), "bytes": len(registry_raw), "raw_sha256": sha_bytes(registry_raw)},
        "member_count": len(member_checks),
        "member_checks": member_checks,
        "scope": registry["scope"],
        "r090_status": registry["r090_status"],
        "remote_stage": registry["remote_stage"],
    }
    receipt_raw = canonical_json(receipt) + b"\n"
    receipt_path = output_dir / "source_stage_package_receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(receipt_raw)
    return {
        "status": receipt["status"],
        "package_path": str(archive_path.resolve()),
        "package_bytes": archive_bytes,
        "package_sha256": archive_sha,
        "binding_registry_path": str(registry_path.resolve()),
        "binding_registry_bytes": len(registry_raw),
        "binding_registry_sha256": sha_bytes(registry_raw),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_bytes": len(receipt_raw),
        "receipt_sha256": sha_bytes(receipt_raw),
        "member_count": len(member_checks),
        "rows": 40,
        "remote_namespace_created": False,
        "effect_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir.resolve()), sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
