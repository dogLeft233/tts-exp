"""Seal-gated CPU consumer for the retry3 source-axis evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .independent_consumer import consume_evaluation_run


EXPECTED_PROTOCOL_SHA256 = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
EXPECTED_ROW_MANIFEST_SHA256 = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
EXPECTED_RECOVERY_CODE_SEAL_SHA256 = "55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417"
EXPECTED_CORE_MANIFEST_SHA256 = "98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73"
EXPECTED_INHERITED_SUMMARY_SHA256 = "0aa21a38aa7b645fb556a589ff132ad71b2083d443df73ecf795f073a522c39c"
EXPECTED_INVENTORY = {
    "items": 6311,
    "archives": 301,
    "planned_row_count": 2560,
    "unique_context_count": 80,
    "unique_visual_feature_count": 1760,
    "unique_score_file_count": 2560,
    "total_archive_bytes": 1587388518,
    "recovery_code_items": 5,
    "recovery_metadata_items": 8,
}


class SealError(RuntimeError):
    """The frozen package or continuation lineage failed independent verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SealError(f"expected a JSON object: {path}")
    return value


def _safe_member(name: str) -> str:
    value = PurePosixPath(name)
    if not name or value == PurePosixPath(".") or value.is_absolute() or any(part in ("..", "") for part in value.parts):
        raise SealError(f"unsafe archive member path: {name}")
    return value.as_posix()


def verify_core_transfer(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    transfer_dir: str | Path,
    inherited_summary_path: str | Path,
    expected_manifest_sha256: str = EXPECTED_CORE_MANIFEST_SHA256,
    expected_inherited_summary_sha256: str = EXPECTED_INHERITED_SUMMARY_SHA256,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
    expected_row_manifest_sha256: str = EXPECTED_ROW_MANIFEST_SHA256,
    expected_recovery_code_seal_sha256: str = EXPECTED_RECOVERY_CODE_SEAL_SHA256,
    expected_face_row_counts: Mapping[str, int] = {"primary": 1280, "secondary": 1280},
    expected_inventory: Mapping[str, int] = EXPECTED_INVENTORY,
) -> dict[str, Any]:
    """Verify immutable core tar inventory, file members, split results, and parent seal."""

    run_root = Path(run_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    package_root = Path(transfer_dir).resolve()
    inherited_file = Path(inherited_summary_path).resolve()
    manifest_sha = sha256_file(manifest_file)
    inherited_sha = sha256_file(inherited_file)
    if manifest_sha != expected_manifest_sha256:
        raise SealError("core transfer manifest SHA differs from the frozen manifest")
    if inherited_sha != expected_inherited_summary_sha256:
        raise SealError("continuation parent inheritance summary SHA differs from the frozen summary")

    manifest = _read_json(manifest_file)
    if (manifest.get("package_type") != "analysis_core"
            or manifest.get("protocol_sha256") != expected_protocol_sha256
            or manifest.get("row_manifest_sha256") != expected_row_manifest_sha256
            or manifest.get("planned_row_count") != expected_inventory["planned_row_count"]
            or manifest.get("contains_effect_summary") is not False):
        raise SealError("core manifest is not the frozen effect-summary-free retry3 package")
    items = manifest.get("items")
    archives = manifest.get("archives")
    if not isinstance(items, list) or not isinstance(archives, list):
        raise SealError("core manifest is missing item or archive inventories")
    archive_bytes = sum(int(item.get("size_bytes", -1)) for item in archives)
    inventory_observed = {
        "items": len(items), "archives": len(archives),
        "planned_row_count": int(manifest.get("planned_row_count", -1)),
        "unique_context_count": int(manifest.get("unique_context_count", -1)),
        "unique_visual_feature_count": int(manifest.get("unique_visual_feature_count", -1)),
        "unique_score_file_count": int(manifest.get("unique_score_file_count", -1)),
        "total_archive_bytes": archive_bytes,
        "recovery_code_items": sum(item.get("category") == "recovery_code" for item in items),
        "recovery_metadata_items": sum(item.get("category") == "sealed_run_metadata" for item in items),
    }
    if inventory_observed != dict(expected_inventory):
        raise SealError(f"core inventory differs from the declared retry3 package: {inventory_observed}")
    face_counts = manifest.get("face_row_counts", {})
    if face_counts != dict(expected_face_row_counts):
        raise SealError("core package does not contain both complete 1,280-row faces")

    by_archive: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in items:
        name = str(item.get("archive_name", ""))
        member = _safe_member(str(item.get("archive_path", "")))
        if not name or member in by_archive.setdefault(name, {}):
            raise SealError(f"duplicate or unbound core archive member: {name}/{member}")
        by_archive[name][member] = item
    if set(by_archive) != {str(archive.get("name", "")) for archive in archives}:
        raise SealError("core archive names differ between item and archive inventories")

    verified_members = 0
    verified_archive_bytes = 0
    for archive in archives:
        name = str(archive["name"])
        archive_path = package_root / name
        if not archive_path.is_file():
            raise SealError(f"core archive is missing: {name}")
        expected_size = int(archive["size_bytes"])
        if archive_path.stat().st_size != expected_size or sha256_file(archive_path) != archive.get("sha256"):
            raise SealError(f"core archive size or SHA mismatch: {name}")
        verified_archive_bytes += expected_size
        expected_members = by_archive[name]
        seen: set[str] = set()
        with tarfile.open(archive_path, mode="r:gz") as tar:
            for info in tar.getmembers():
                if not info.isfile():
                    continue
                member = _safe_member(info.name)
                item = expected_members.get(member)
                if item is None or member in seen:
                    raise SealError(f"unexpected or duplicate member in {name}: {member}")
                if info.size != int(item.get("size_bytes", -1)):
                    raise SealError(f"member byte count mismatch: {name}/{member}")
                source = tar.extractfile(info)
                if source is None:
                    raise SealError(f"cannot read core member: {name}/{member}")
                with source:
                    member_sha, member_size = _sha256_stream(source)
                if member_size != int(item["size_bytes"]) or member_sha != item.get("sha256"):
                    raise SealError(f"member SHA mismatch: {name}/{member}")
                seen.add(member)
                verified_members += 1
        if seen != set(expected_members):
            missing = sorted(set(expected_members) - seen)
            raise SealError(f"archive member inventory is incomplete in {name}: {missing[:3]}")

    if verified_members != len(items) or verified_archive_bytes != expected_inventory["total_archive_bytes"]:
        raise SealError("verified archive/member totals do not match the frozen core inventory")

    for relative, expected_sha in (
        ("evaluation_protocol.json", expected_protocol_sha256),
        ("evaluation_rows.jsonl", expected_row_manifest_sha256),
        ("recovery_code_seal.json", expected_recovery_code_seal_sha256),
    ):
        path = run_root / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise SealError(f"run artifact differs from its frozen core SHA: {relative}")
    result_split = manifest.get("split_files", {}).get("evaluation_results.jsonl")
    if not isinstance(result_split, Mapping):
        raise SealError("core manifest does not describe the sealed evaluation result split")
    reconstructed = hashlib.sha256()
    reconstructed_size = 0
    for part in result_split.get("parts", []):
        part_path = _safe_member(str(part.get("archive_path", "")))
        item = next((candidate for candidate in items if candidate.get("archive_path") == part_path), None)
        if item is None or item.get("sha256") != part.get("sha256") or int(item.get("size_bytes", -1)) != int(part.get("size_bytes", -2)):
            raise SealError(f"evaluation result split part is not bound to the core member: {part_path}")
        archive_path = package_root / str(item["archive_name"])
        with tarfile.open(archive_path, mode="r:gz") as tar:
            member_stream = tar.extractfile(part_path)
            if member_stream is None:
                raise SealError(f"cannot reconstruct result part: {part_path}")
            with member_stream:
                for block in iter(lambda: member_stream.read(1024 * 1024), b""):
                    reconstructed.update(block)
                    reconstructed_size += len(block)
    if (reconstructed_size != int(result_split.get("original_size_bytes", -1))
            or reconstructed.hexdigest() != result_split.get("original_sha256")):
        raise SealError("concatenated evaluation result parts differ from the declared complete result log")
    result_path = run_root / "evaluation_results.jsonl"
    if (not isinstance(result_split, Mapping) or not result_path.is_file()
            or result_path.stat().st_size != int(result_split.get("original_size_bytes", -1))
            or sha256_file(result_path) != result_split.get("original_sha256")):
        raise SealError("run evaluation_results.jsonl differs from its frozen split-file seal")
    status_path = run_root / "evaluation_clip_status.jsonl"
    attempts_path = run_root / "evaluation_attempts.jsonl"
    item_hashes = {str(item["archive_path"]): str(item["sha256"]) for item in items}
    for relative in (
        "continuation_parent.json", "recovery_artifact_hashes.jsonl",
        "recovery_detached_metadata_manifest.jsonl", "evaluation_clip_status.jsonl",
        "evaluation_attempts.jsonl",
    ):
        path = run_root / relative
        if not path.is_file() or sha256_file(path) != item_hashes.get(relative):
            raise SealError(f"run lineage/completion artifact differs from its core archive member: {relative}")
    if not status_path.stat().st_size or not attempts_path.stat().st_size:
        raise SealError("run status and append-only attempts logs must be present")

    return {
        "status": "verified",
        "manifest_sha256": manifest_sha,
        "inherited_summary_sha256": inherited_sha,
        "protocol_sha256": expected_protocol_sha256,
        "row_manifest_sha256": expected_row_manifest_sha256,
        "recovery_code_seal_sha256": expected_recovery_code_seal_sha256,
        "archive_count": len(archives), "archive_bytes": verified_archive_bytes,
        "member_count": verified_members, "member_count_expected": len(items),
        "contains_effect_summary": False,
        "complete_faces": face_counts,
        "input_result_log_sha256": str(result_split["original_sha256"]),
    }


def _execution_environment() -> dict[str, Any]:
    flags = {
        name: os.environ.get(name, "")
        for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if flags["CUDA_VISIBLE_DEVICES"] not in ("", "-1"):
        raise SealError("CPU validator requires CUDA_VISIBLE_DEVICES to be empty")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            thread_count = int(flags[name])
        except (TypeError, ValueError) as exc:
            raise SealError(f"CPU validator requires an explicit {name}=1 or 2") from exc
        if thread_count not in (1, 2):
            raise SealError(f"CPU validator requires {name}=1 or 2")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_only_environment": flags,
    }


def run_sealed_consumer(
    *,
    run_dir: str | Path,
    calibration_dir: str | Path,
    manifest_path: str | Path,
    transfer_dir: str | Path,
    inherited_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify immutable lineage and all core members, then run frozen CPU analyses."""

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    preflight = verify_core_transfer(
        run_dir=run_dir,
        manifest_path=manifest_path,
        transfer_dir=transfer_dir,
        inherited_summary_path=inherited_summary_path,
    )
    preflight_path = output_root / "seal_preflight.json"
    preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    consumer_output = output_root / "independent_consumer_summary.json"
    consumer_result = consume_evaluation_run(
        run_dir,
        calibration_dir=calibration_dir,
        null_count=100,
        bootstrap_draws=20_000,
        verify_cached_embeddings=True,
    )
    consumer_output.write_text(json.dumps(consumer_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "schema_version": 1,
        "status": "complete" if consumer_result.get("status") == "complete" else consumer_result.get("status"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_parameters": {
            "run_dir": str(Path(run_dir).resolve()),
            "calibration_dir": str(Path(calibration_dir).resolve()),
            "manifest_path": str(Path(manifest_path).resolve()),
            "transfer_dir": str(Path(transfer_dir).resolve()),
            "inherited_summary_path": str(Path(inherited_summary_path).resolve()),
            "null_permutations": 100,
            "speaker_bootstrap_draws": 20_000,
            "cached_embedding_parity": "full raw/unit V×A matrices plus selected aligned-lag parity",
        },
        "input_seals": preflight,
        "source_hashes": {
            "independent_sealed_run.py": sha256_file(source_root / "independent_sealed_run.py"),
            "independent_consumer.py": sha256_file(source_root / "independent_consumer.py"),
            "independent_validator.py": sha256_file(source_root / "independent_validator.py"),
        },
        "runtime": _execution_environment(),
        "outputs": {
            "preflight_path": str(preflight_path), "preflight_sha256": sha256_file(preflight_path),
            "consumer_summary_path": str(consumer_output), "consumer_summary_sha256": sha256_file(consumer_output),
        },
    }
    record_path = output_root / "execution_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--transfer-dir", required=True)
    parser.add_argument("--inherited-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    record = run_sealed_consumer(
        run_dir=args.run_dir,
        calibration_dir=args.calibration_dir,
        manifest_path=args.manifest,
        transfer_dir=args.transfer_dir,
        inherited_summary_path=args.inherited_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": record["status"], "record_path": record["outputs"]["consumer_summary_path"], "consumer_summary_sha256": record["outputs"]["consumer_summary_sha256"]}, separators=(",", ":")))
    return 0 if record["status"] in ("complete", "INCOMPLETE_SUPPORT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
