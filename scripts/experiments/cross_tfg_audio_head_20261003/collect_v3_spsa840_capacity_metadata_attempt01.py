#!/usr/bin/env python3
"""Read retained V3 row metadata and report bounded capacity evidence.

This utility reads only JSON/JSONL members from the sealed 540-row evidence
archive. It never extracts media, opens candidate audio/video, invokes a model,
or contacts a storage/API service. An optional directory can later supply the
actual 180 step-120 Dev result ledgers for candidate timing summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_REL = Path(
    "runs/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt07_readback_attempt01"
    "/static_outputs_attempt02/provenance/baseline_540_evidence.tar.gz"
)
ARCHIVE_BYTES = 18_784_301
ARCHIVE_SHA256 = "349c5b62f09c1a775013039d2abb3bd72d7c6b6233531fb7f87e6398fff981d4"
INDEX_MEMBER = "row_asset_index.json"
RESULT_MEMBERS = {
    "wav2lip_gan_bac9a81e": "producer/w2l/results.jsonl",
    "musetalk_fixed_portrait_attempt16": "producer/muse/results.jsonl",
    "ditto_hubert_fixed_portrait_attempt18": "producer/ditto/results.jsonl",
}
MODEL_DIRS = {
    "wav2lip_gan_bac9a81e": "w2l",
    "musetalk_fixed_portrait_attempt16": "muse",
    "ditto_hubert_fixed_portrait_attempt18": "ditto",
}
CONTROL_METHODS = {"identity", "rms_fixed"}
METHOD_CANDIDATE = "shared_recipient_transition_head_v3_fit40_step120"
EXPECTED_CANDIDATE = {
    "wav2lip_gan_bac9a81e": "w2l/results.jsonl",
    "musetalk_fixed_portrait_attempt16": "muse/results.jsonl",
    "ditto_hubert_fixed_portrait_attempt18": "ditto/results.jsonl",
}
STAGE_NAMES = ("generation", "strip_audio", "natural_pcm_mux", "syncnet_tracking", "official_evaluator")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def nearest_rank(values: list[float | int], percentile: float = 0.95) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def distribution(values: list[float | int], unit: str) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95_nearest_rank": None, "max": None, "unit": unit}
    return {
        "count": len(ordered),
        "p50": ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)],
        "p95_nearest_rank": nearest_rank(ordered),
        "max": ordered[-1],
        "unit": unit,
    }


def read_member(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str,
                max_bytes: int) -> bytes:
    info = members.get(name)
    if info is None or not info.isfile() or info.size < 0 or info.size > max_bytes:
        raise ValueError(f"missing, nonregular, or oversized metadata member: {name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"could not read metadata member: {name}")
    raw = stream.read(max_bytes + 1)
    if len(raw) != info.size:
        raise ValueError(f"metadata member byte count differs from archive index: {name}")
    return raw


def parse_jsonl(raw: bytes, label: str) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    line_sizes: list[int] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} is not a JSON object")
        rows.append(row)
        line_sizes.append(len(line))
    return rows, line_sizes


def stage_receipt(row: dict[str, Any], stage: str) -> dict[str, Any] | None:
    if stage == "official_evaluator":
        value = row.get("score", {}).get("official_evaluator", {}).get("execution")
    else:
        value = row.get("render", {}).get(stage)
    return value if isinstance(value, dict) else None


def elapsed(receipt: dict[str, Any] | None) -> float | None:
    if not isinstance(receipt, dict):
        return None
    started, ended = receipt.get("started_unix"), receipt.get("ended_unix")
    if not isinstance(started, (int, float)) or not isinstance(ended, (int, float)) or ended < started:
        return None
    return float(ended - started)


def stage_log_bytes(receipt: dict[str, Any] | None) -> int | None:
    if not isinstance(receipt, dict):
        return None
    values = (receipt.get("stdout_bytes", 0), receipt.get("stderr_bytes", 0))
    if any(type(value) is not int or value < 0 for value in values):
        return None
    return sum(values)


def collect_asset_refs(node: Any, result: dict[str, dict[str, Any]]) -> None:
    if isinstance(node, dict):
        path = node.get("path")
        sha = node.get("sha256")
        size = node.get("bytes")
        if (isinstance(path, str) and path.startswith("assets/") and isinstance(sha, str)
                and type(size) is int and size >= 0):
            old = result.setdefault(path, {"sha256": sha, "bytes": size})
            if old != {"sha256": sha, "bytes": size}:
                raise ValueError(f"row has inconsistent archive member descriptor: {path}")
        for value in node.values():
            collect_asset_refs(value, result)
    elif isinstance(node, list):
        for value in node:
            collect_asset_refs(value, result)


def collect_baseline(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo],
                     archive_manifest: dict[str, Any]) -> dict[str, Any]:
    index_raw = read_member(archive, members, INDEX_MEMBER, 16 * 1024 * 1024)
    static_files = {item.get("path"): item for item in archive_manifest.get("static_files", []) if isinstance(item, dict)}
    index_pin = static_files.get(INDEX_MEMBER)
    if not isinstance(index_pin, dict) or index_pin.get("bytes") != len(index_raw) or index_pin.get("sha256") != sha_bytes(index_raw):
        raise ValueError("row_asset_index does not match the protected archive manifest")
    index = json.loads(index_raw.decode("utf-8"))
    if not isinstance(index, dict) or index.get("schema") != "cross_tfg_v4_paired_score_row_asset_index_v1":
        raise ValueError("row_asset_index schema differs from the protected baseline package")
    asset_manifest = {
        item.get("path"): {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
        for item in archive_manifest.get("assets", []) if isinstance(item, dict)
    }
    rows_by_id = {row.get("row_id"): row for row in index.get("rows", []) if isinstance(row, dict)}
    model_reports: dict[str, Any] = {}
    for model_id, member_name in RESULT_MEMBERS.items():
        results_raw = read_member(archive, members, member_name, 16 * 1024 * 1024)
        result_pin = static_files.get(member_name)
        if not isinstance(result_pin, dict) or result_pin.get("bytes") != len(results_raw) or result_pin.get("sha256") != sha_bytes(results_raw):
            raise ValueError(f"protected archive manifest does not bind {member_name}")
        rows, line_sizes = parse_jsonl(results_raw, member_name)
        result_by_id = {row.get("row_id"): (row, size) for row, size in zip(rows, line_sizes)}
        selected = [(row, size) for row, size in zip(rows, line_sizes)
                    if row.get("model_id") == model_id and row.get("method_id") in CONTROL_METHODS and row.get("status") == "PASS"]
        if len(selected) != 120 or len({row.get("row_id") for row, _size in selected}) != 120:
            raise ValueError(f"{model_id} baseline must contain exactly 120 PASS identity/RMS rows")
        stage_values: dict[str, list[float]] = {stage: [] for stage in STAGE_NAMES}
        stage_logs: dict[str, list[int]] = {stage: [] for stage in STAGE_NAMES}
        row_total_seconds: list[float] = []
        selected_line_bytes: list[int] = []
        row_archive_asset_bytes: list[int] = []
        row_archive_asset_counts: list[int] = []
        row_archive_ids: set[str] = set()
        generation_log_bytes: list[int] = []
        for row, line_size in selected:
            selected_line_bytes.append(line_size)
            durations: list[float] = []
            for stage in STAGE_NAMES:
                receipt = stage_receipt(row, stage)
                duration = elapsed(receipt)
                log_bytes = stage_log_bytes(receipt)
                if duration is not None:
                    stage_values[stage].append(duration)
                    durations.append(duration)
                if log_bytes is not None:
                    stage_logs[stage].append(log_bytes)
                if stage == "generation" and log_bytes is not None:
                    generation_log_bytes.append(log_bytes)
            if len(durations) == len(STAGE_NAMES):
                row_total_seconds.append(sum(durations))
            index_row = rows_by_id.get(row.get("row_id"))
            if not isinstance(index_row, dict):
                raise ValueError(f"row_asset_index lacks protected row {row.get('row_id')}")
            refs: dict[str, dict[str, Any]] = {}
            collect_asset_refs(index_row, refs)
            total = 0
            for asset_path, pin in refs.items():
                actual = asset_manifest.get(asset_path)
                if actual != pin:
                    raise ValueError(f"row asset reference differs from archive member manifest: {asset_path}")
                total += pin["bytes"]
                row_archive_ids.add(asset_path)
            row_archive_asset_bytes.append(total)
            row_archive_asset_counts.append(len(refs))
        stage_report = {
            stage: {
                "elapsed_seconds": distribution(values, "seconds"),
                "stdout_plus_stderr_bytes": distribution(stage_logs[stage], "bytes"),
            }
            for stage, values in stage_values.items()
        }
        model_reports[model_id] = {
            "source": {"member": member_name, "bytes": len(results_raw), "raw_sha256": sha_bytes(results_raw)},
            "baseline_scope": {"methods": sorted(CONTROL_METHODS), "pass_rows": len(selected), "rows_per_method": 60,
                               "candidate_step120_rows": 0},
            "actual_p95": {
                "per_stage": stage_report,
                "sequential_sum_of_five_stage_elapsed_seconds": distribution(row_total_seconds, "seconds"),
                "producer_result_record_bytes": distribution(selected_line_bytes, "bytes"),
                "row_referenced_retained_archive_asset_bytes": distribution(row_archive_asset_bytes, "bytes"),
                "row_referenced_retained_archive_asset_count": distribution(row_archive_asset_counts, "members"),
                "generation_stdout_plus_stderr_bytes": distribution(generation_log_bytes, "bytes"),
            },
            "archive_asset_members_referenced_by_selected_rows": len(row_archive_ids),
            "archive_asset_bytes_referenced_by_selected_rows_deduplicated": sum(asset_manifest[path]["bytes"] for path in row_archive_ids),
            "archive_asset_policy": "deduplicated archive members only; not per-row disk usage; generated MP4/JPEG/crop media are excluded",
        }
    return {
        "status": "PASS",
        "row_asset_index": {"member": INDEX_MEMBER, "bytes": len(index_raw), "raw_sha256": sha_bytes(index_raw),
                            "rows": len(index.get("rows", [])), "manifest_sha256": index.get("manifest_sha256")},
        "models": model_reports,
    }


def collect_candidate(candidate_root: Path | None) -> dict[str, Any]:
    if candidate_root is None:
        return {"status": "NOT_AVAILABLE", "candidate_step120_generation_p95": None,
                "reason": "no actual completed Dev180 W2L/MuseTalk/Ditto result arrays were supplied; plan-only rows are not timing evidence"}
    root = candidate_root.resolve(strict=True)
    models: dict[str, Any] = {}
    for model_id, relative in EXPECTED_CANDIDATE.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"candidate results file is missing or symlinked: {path}")
        raw = path.read_bytes()
        rows, line_sizes = parse_jsonl(raw, str(path))
        if len(rows) != 60:
            raise ValueError(f"candidate {model_id} result array must contain exactly 60 clip×seed rows")
        if any(row.get("model_id") != model_id or row.get("method_id") != METHOD_CANDIDATE
               or row.get("split") != "dev" for row in rows):
            raise ValueError(f"candidate {model_id} rows do not match the frozen step-120 Dev method")
        if len({(row.get("sample_id"), row.get("speaker_id"), row.get("seed")) for row in rows}) != 60:
            raise ValueError(f"candidate {model_id} rows contain duplicate clip×seed identities")
        durations: list[float] = []
        logs: list[int] = []
        for row in rows:
            receipt = stage_receipt(row, "generation")
            duration = elapsed(receipt)
            log_bytes = stage_log_bytes(receipt)
            if duration is not None:
                durations.append(duration)
            if log_bytes is not None:
                logs.append(log_bytes)
        models[model_id] = {
            "rows": len(rows), "pass_rows": sum(row.get("status") == "PASS" for row in rows),
            "results_path": str(path), "results_bytes": len(raw), "results_raw_sha256": sha_bytes(raw),
            "producer_result_record_bytes": distribution(line_sizes, "bytes"),
            "actual_generation_elapsed_seconds": distribution(durations, "seconds"),
            "generation_stdout_plus_stderr_bytes": distribution(logs, "bytes"),
            "complete_generation_timing_receipts": len(durations) == 60,
        }
    return {"status": "PASS", "candidate_method_id": METHOD_CANDIDATE,
            "planned_candidate_rows": 180, "models": models,
            "media_bytes": "not present in producer result ledgers; obtain media member/file-size metadata separately"}


def collect_windows_capacity() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "UNAVAILABLE", "reason": "Windows drive inventory is not available on this host"}
    command = (
        "Get-CimInstance Win32_LogicalDisk | Where-Object { $_.DeviceID -in @('B:','C:','E:','F:') } "
        "| Select-Object DeviceID,DriveType,ProviderName,Size,FreeSpace | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
    decoded = json.loads(proc.stdout)
    entries = decoded if isinstance(decoded, list) else [decoded]
    normalized = []
    for item in entries:
        normalized.append({
            "device_id": item.get("DeviceID"), "drive_type": item.get("DriveType"),
            "provider_name": item.get("ProviderName") or None,
            "total_bytes": item.get("Size"), "free_bytes": item.get("FreeSpace"),
        })
    return {
        "status": "OBSERVED_UNATTESTED_AS_BACKUP",
        "observed_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Windows Win32_LogicalDisk read-only inventory; no filesystem/API write performed",
        "drives": normalized,
        "baidu_account_or_quota_verified": False,
        "configured_backup_destination_identified": False,
        "capacity_gate": "INCOMPLETE: mapped CloudDrive and local volumes are observed, but Baidu identity and protected backup target are not attested",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / ARCHIVE_REL)
    parser.add_argument("--candidate-results-root", type=Path,
                        help="optional actual Dev180 producer-ledger root with w2l/muse/ditto/results.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/cross_tfg_audio_head_20261003/spsa840_capacity_metadata_attempt01.json")
    args = parser.parse_args()

    archive_path = args.archive.resolve(strict=True)
    archive_bytes = archive_path.stat().st_size
    archive_sha = sha_file(archive_path)
    if archive_bytes != ARCHIVE_BYTES or archive_sha != ARCHIVE_SHA256:
        raise ValueError("baseline archive differs from the exact protected 540-row source")
    with tarfile.open(archive_path, "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for info in archive.getmembers():
            name, parsed = info.name, PurePosixPath(info.name)
            if (not name or "\\" in name or parsed.is_absolute() or ".." in parsed.parts
                    or name in members or info.issym() or info.islnk() or not (info.isdir() or info.isfile())):
                raise ValueError(f"unsafe or nonregular archive member: {name!r}")
            members[name] = info
        manifest_raw = read_member(archive, members, "evidence_archive_manifest.json", 2 * 1024 * 1024)
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if (manifest.get("schema") != "cross_tfg_v4_cross_frontend_score_evidence_archive_v1"
                or manifest.get("role") != "read_only_raw_score_and_common_support_evidence"
                or manifest.get("row_count") != 540 or manifest.get("confirm_speakers_read") is not False):
            raise ValueError("protected baseline archive manifest scope/schema differs")
        baseline = collect_baseline(archive, members, manifest)
        archive_manifest_sha = sha_bytes(manifest_raw)
        archive_member_count = len(members)
        archive_media_policy = manifest.get("media_policy")

    report = {
        "schema": "cross_tfg_v3_spsa840_capacity_metadata_v1",
        "status": "PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collector_source": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                             "bytes": Path(__file__).stat().st_size, "raw_sha256": sha_file(Path(__file__))},
        "scope": {"plan_only": True, "dev_audio_opened": False, "confirm_audio_opened": False,
                  "model_or_api_called": False, "network_api_calls": 0},
        "baseline_archive": {"path": archive_path.relative_to(ROOT).as_posix(), "bytes": archive_bytes,
                             "raw_sha256": archive_sha, "manifest_raw_sha256": archive_manifest_sha,
                             "members": archive_member_count, "row_count": manifest.get("row_count"),
                             "media_policy": archive_media_policy,
                             "full_render_media_bytes_included": False},
        "baseline_360_identity_rms_controls": baseline,
        "candidate_step120_dev180": collect_candidate(args.candidate_results_root),
        "windows_storage_inventory": collect_windows_capacity(),
        "required_followup_for_spsa840_gate": [
            "actual step-120 Dev180 result arrays with complete per-model generation receipts",
            "generated MP4/crop/JPEG or exact archive member size distribution by backend; excluded from protected baseline archive",
            "attested Baidu account quota/free space and the exact protected backup destination",
            "current target filesystem free-space and SHM/native-cache dependency inventory on the migrated remote host",
        ],
        "capacity_gate": "INCOMPLETE; retained score/evidence bytes and actual baseline p95 timing are available, but candidate media size, Baidu quota, backup target, and migrated-host SHM/cache inventory are not evidenced here",
    }
    raw = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing capacity metadata output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "baseline_models": len(baseline["models"]),
                      "step120_candidate_receipts": report["candidate_step120_dev180"]["status"],
                      "capacity_gate": report["capacity_gate"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
