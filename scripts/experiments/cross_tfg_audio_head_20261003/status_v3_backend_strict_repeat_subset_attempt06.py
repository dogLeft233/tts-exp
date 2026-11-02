#!/usr/bin/env python3
"""Read-only compact status view for an already-dispatched pilot06 output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value, raw


def process_identity(pid: int, expected_start_ticks: int) -> dict[str, Any]:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
    except FileNotFoundError:
        return {"status": "EXITED", "pid": pid, "process_start_ticks": None}
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("Linux process stat lacks a closing comm delimiter")
    fields = raw[closing + 2:].split()
    if len(fields) < 20:
        raise ValueError("Linux process stat is truncated before starttime")
    state = fields[0]
    start_ticks = int(fields[19])
    if start_ticks != expected_start_ticks:
        return {"status": "PID_REUSED", "pid": pid, "process_start_ticks": start_ticks,
                "expected_process_start_ticks": expected_start_ticks, "process_state": state}
    return {"status": "RUNNING" if state != "Z" else "EXITED", "pid": pid,
            "process_start_ticks": start_ticks, "process_state": state}


def load_launch_receipt(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    receipt, raw = read_object(path)
    if (receipt.get("schema") != "cross_tfg_backend_strict_repeat_detached_launch_v1"
            or receipt.get("launch_receipt_sha256") != payload_sha(receipt, "launch_receipt_sha256")
            or receipt.get("start_new_session") is not True):
        raise ValueError("detached launch receipt schema/hash/session gate failed")
    pid, start_ticks = receipt.get("pid"), receipt.get("process_start_ticks")
    if type(pid) is not int or type(start_ticks) is not int:
        raise ValueError("detached launch receipt lacks integer PID/start-tick identity")
    identity = process_identity(pid, start_ticks)
    log_path = Path(receipt.get("stdout_stderr_log", ""))
    if log_path.is_symlink() or not log_path.is_file():
        log = {"status": "MISSING_OR_UNSAFE", "path": str(log_path)}
    else:
        log = {"status": "PRESENT", "path": str(log_path.resolve(strict=True)),
               "bytes": log_path.stat().st_size}
    metadata = {"path": str(path.resolve(strict=True)), "bytes": len(raw),
                "sha256": sha_bytes(raw), "launch_receipt_sha256": receipt["launch_receipt_sha256"],
                "launch_status": receipt.get("status"), "process": identity, "log": log,
                "output_root": receipt.get("output_root"), "requested_models": receipt.get("requested_models"),
                "started_unix": receipt.get("started_unix"),
                "external_timeout": receipt.get("external_timeout")}
    return receipt, metadata


def status(output_root: Path, launch_receipt_path: Path | None) -> dict[str, Any]:
    if output_root.is_symlink() or not output_root.is_dir():
        if output_root.exists() or output_root.is_symlink():
            raise ValueError(f"pilot output root is unsafe: {output_root}")
    final_path = output_root / "pilot_manifest_final.json"
    if output_root.is_dir() and final_path.is_file() and not final_path.is_symlink():
        path = final_path
        final = True
    else:
        final = False
        progress = sorted(output_root.glob("pilot_manifest_progress_*.json")) if output_root.is_dir() else []
        progress = [item for item in progress if item.is_file() and not item.is_symlink()]
        initial = output_root / "pilot_manifest.json"
        if progress:
            path = progress[-1]
        elif initial.is_file() and not initial.is_symlink():
            path = initial
        else:
            manifest, raw, path, final = {}, b"", None, False

    if path is not None:
        manifest, raw = read_object(path)
        if manifest.get("schema") != "cross_tfg_backend_strict_repeat_pilot_manifest_v1":
            raise ValueError("pilot manifest schema does not match attempt06")
        if manifest.get("planned_rows") != 6 or len(manifest.get("row_ids", [])) != 6:
            raise ValueError("full six-row planned denominator is absent")
    rows = manifest.get("rows", [])
    row_status_counts = {key: sum(row.get("status") == key for row in rows)
                         for key in ("PASS", "FAILED", "INCOMPLETE")}
    qualifications = manifest.get("backend_qualifications", {})
    receipt_checks: dict[str, Any] = {}
    for model_id, entry in qualifications.items():
        receipt_ref = entry.get("receipt") if isinstance(entry, dict) else None
        if not isinstance(receipt_ref, dict):
            receipt_checks[model_id] = {"status": entry.get("status") if isinstance(entry, dict) else "INVALID"}
            continue
        receipt_path = Path(receipt_ref.get("path", ""))
        if not receipt_path.is_absolute() or receipt_path.is_symlink() or not receipt_path.is_file():
            receipt_checks[model_id] = {"status": "RECEIPT_MISSING"}
            continue
        receipt, receipt_raw = read_object(receipt_path)
        digest = sha_bytes(receipt_raw)
        self_sha_ok = receipt.get("receipt_sha256") == payload_sha(receipt, "receipt_sha256")
        ref_ok = (len(receipt_raw) == receipt_ref.get("bytes") == receipt_path.stat().st_size
                  and digest == receipt_ref.get("sha256"))
        receipt_checks[model_id] = {"status": receipt.get("status"),
                                    "receipt_sha256": digest,
                                    "canonical_self_hash_valid": self_sha_ok,
                                    "manifest_reference_valid": ref_ok}

    final_hash_valid = None
    if final:
        final_hash_valid = manifest.get("manifest_sha256") == payload_sha(manifest, "manifest_sha256")
    launch_receipt, launch_metadata = load_launch_receipt(launch_receipt_path)
    if launch_receipt is not None and launch_receipt.get("output_root") != str(output_root.resolve()):
        raise ValueError("launch receipt output path does not match requested status root")
    process_status = (launch_metadata or {}).get("process", {}).get("status")
    overall_status = manifest.get("status", "NO_MANIFEST")
    if (not final and process_status == "EXITED"):
        overall_status = "PROCESS_EXITED_WITHOUT_FINAL_MANIFEST"
    return {
        "schema": "cross_tfg_backend_strict_repeat_status_v1",
        "status": overall_status,
        "final_manifest": final,
        "manifest_path": str(path.resolve(strict=True)) if path else None,
        "manifest_bytes": len(raw),
        "manifest_file_sha256": sha_bytes(raw) if raw else None,
        "manifest_canonical_self_hash_valid": final_hash_valid,
        "run_id": manifest.get("run_id"),
        "planned_rows": manifest.get("planned_rows"),
        "requested_render_rows": manifest.get("requested_render_rows"),
        "completed_rows": manifest.get("completed_rows"),
        "requested_models": manifest.get("requested_models"),
        "row_status_counts": row_status_counts,
        "row_statuses": [{"row_id": row.get("row_id"), "model_id": row.get("model_id"),
                           "status": row.get("status"), "failure_reason": row.get("failure_reason")}
                          for row in rows],
        "backend_qualifications": {key: value.get("status") for key, value in qualifications.items()},
        "receipt_checks": receipt_checks,
        "failures": manifest.get("failures", []),
        "resource_before": manifest.get("resource_before"),
        "resource_after": manifest.get("resource_after"),
        "detached_launch": launch_metadata,
        "root_cause_scope": "engineering-only strict repeat status; not Dev/Confirm qualification or method selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path)
    args = parser.parse_args()
    try:
        result = status(args.output_root, args.launch_receipt)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if result.get("status") not in {"FAILED_OR_INCOMPLETE", "PROCESS_EXITED_WITHOUT_FINAL_MANIFEST",
                                                  "STATUS_READ_FAILED"} else 2
    except Exception as exc:
        print(json.dumps({"schema": "cross_tfg_backend_strict_repeat_status_v1",
                          "status": "STATUS_READ_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
