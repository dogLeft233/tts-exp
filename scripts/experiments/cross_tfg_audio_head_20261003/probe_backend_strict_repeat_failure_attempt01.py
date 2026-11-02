#!/usr/bin/env python3
"""Bounded, read-only diagnostics for strict-repeat attempt05 row failures.

Reads only the pinned pilot manifest, the four row renderer logs, JSON startup /
adapter receipts if present, and the detached-launch receipt/log. It never opens
audio or video payloads, imports a backend, reads process environments, or uses
GPU/API resources. Diagnostic output is capped and environment data is limited
to fields emitted by the strict-startup receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path(
    "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/"
    "samehost_dev360_latebind_attempt01/"
    "backend_strict_repeat_attempt03_wav2lip_ditto_attempt05"
)
EXPECTED_MANIFEST_BYTES = 61129
EXPECTED_MANIFEST_SHA256 = "7df7068bad5dce152e894c0497117aa4173c28406b69b989f1ac3a2ab809e552"
LAUNCH_RECEIPT = Path(
    "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/"
    "job_control/backend_strict_repeat_attempt03_w2l_ditto_attempt05/"
    "detached_launch_receipt.json"
)
EXPECTED_LAUNCH_RECEIPT_SHA256 = "f19b85fddca6f1cdb394f8d7fa95fd958fb3a671ad37286b5aa9e4375699c48d"
SUPERVISOR_LOG = Path(
    "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/"
    "job_control/backend_strict_repeat_attempt03_w2l_ditto_attempt05/"
    "pilot03_stdout_stderr.log"
)
ROWS = (
    ("strictrepeat_wav2lip_gan_bac9a81e_01", "wav2lip_gan_bac9a81e", "repeat_01"),
    ("strictrepeat_wav2lip_gan_bac9a81e_02", "wav2lip_gan_bac9a81e", "repeat_02"),
    ("strictrepeat_ditto_hubert_fixed_portrait_attempt18_01",
     "ditto_hubert_fixed_portrait_attempt18", "repeat_01"),
    ("strictrepeat_ditto_hubert_fixed_portrait_attempt18_02",
     "ditto_hubert_fixed_portrait_attempt18", "repeat_02"),
)
MAX_MANIFEST_BYTES = 1 << 20
MAX_RECEIPT_BYTES = 1 << 16
MAX_LOG_BYTES_TO_HASH = 1 << 20
MAX_LOG_TAIL_BYTES = 16 << 10
MAX_TRACEBACK_CHARS = 8000
SENSITIVE_VALUE = re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)[^\s,;]+")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path, limit: int) -> tuple[int, str | None]:
    size = path.stat().st_size
    if size > limit:
        return size, None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            h.update(block)
    return size, h.hexdigest()


def bounded_text(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    return SENSITIVE_VALUE.sub(r"\1\2[REDACTED]", text)


def safe_ref(path: Path, *, max_bytes: int, tail_bytes: int = 0) -> dict[str, Any]:
    try:
        st = os.lstat(path)
    except OSError as exc:
        return {"path": str(path), "status": "MISSING", "error_type": type(exc).__name__}
    if not stat.S_ISREG(st.st_mode):
        return {"path": str(path), "status": "NOT_REGULAR", "mode": stat.S_IFMT(st.st_mode)}
    size = int(st.st_size)
    result: dict[str, Any] = {"path": str(path), "status": "PRESENT", "bytes": size}
    actual_size, digest = file_digest(path, max_bytes)
    result["sha256"] = digest
    result["sha256_status"] = "FULL_FILE" if digest else "SKIPPED_FILE_EXCEEDS_HASH_BOUND"
    if tail_bytes:
        with path.open("rb") as stream:
            stream.seek(max(0, size - tail_bytes))
            tail = stream.read(tail_bytes)
        result["tail_bytes"] = len(tail)
        result["tail_sha256"] = sha256(tail)
        result["tail_text"] = bounded_text(tail)
    return result


def safe_json(path: Path, max_bytes: int = MAX_RECEIPT_BYTES) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ref = safe_ref(path, max_bytes=max_bytes)
    if ref.get("status") != "PRESENT" or ref.get("bytes", max_bytes + 1) > max_bytes:
        return None, ref
    try:
        return json.loads(path.read_text(encoding="utf-8")), ref
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        ref["json_error_type"] = type(exc).__name__
        return None, ref


def receipt_view(record: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema", "status", "pid", "ppid", "row_id", "model_id", "audio_frontend",
        "seed", "python_executable", "entry_path", "expected_backend_entry",
        "entry_matches_backend", "bootstrap_source_sha256", "policy",
        "policy_payload_sha256", "cuda_available", "torch_version", "receipt_sha256",
    )
    return {key: record[key] for key in allowed if key in record}


def main() -> None:
    out = OUTPUT_ROOT.resolve(strict=True)
    manifest_path = out / "pilot_manifest_final.json"
    manifest_ref = safe_ref(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
    result: dict[str, Any] = {
        "schema": "cross_tfg_backend_strict_repeat_failure_probe_v1",
        "read_only": True,
        "backend_import_or_forward_performed": False,
        "gpu_touched": False,
        "api_calls_performed": False,
        "audio_or_video_payloads_opened": False,
        "selection_or_dev_confirm_eligibility": False,
        "manifest": manifest_ref,
        "launch_receipt": {},
        "supervisor_log": {},
        "rows": [],
    }
    if (manifest_ref.get("status") != "PRESENT"
            or manifest_ref.get("bytes") != EXPECTED_MANIFEST_BYTES
            or manifest_ref.get("sha256") != EXPECTED_MANIFEST_SHA256):
        result["status"] = "BLOCKED_MANIFEST_PIN_MISMATCH"
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["manifest_fields"] = {
        key: manifest.get(key) for key in (
            "schema", "run_id", "status", "planned_rows", "requested_render_rows",
            "completed_rows", "requested_models", "determinism_policy_sha256",
        )
    }
    result["manifest_fields"]["runtime_identity_fields"] = {
        key: manifest.get("runtime_identity", {}).get(key)
        for key in ("strict_policy_sha256", "qc06_manifest_sha256")
        if key in manifest.get("runtime_identity", {})
    }
    result["manifest_failure_rows"] = [
        {"row_id": row.get("row_id"), "reason": row.get("reason")}
        for row in manifest.get("failures", [])
        if row.get("row_id") in {row_id for row_id, _, _ in ROWS}
    ]
    result["launch_receipt"], launch_ref = safe_json(LAUNCH_RECEIPT)
    result["launch_receipt_ref"] = launch_ref
    result["launch_receipt_pin_match"] = (
        launch_ref.get("status") == "PRESENT"
        and launch_ref.get("sha256") == EXPECTED_LAUNCH_RECEIPT_SHA256
    )
    if result["launch_receipt"] is not None:
        launch = result["launch_receipt"]
        result["launch_receipt"] = {
            key: launch[key] for key in (
                "schema", "status", "pid", "process_start_ticks", "started_unix",
                "command_sha256", "stdout_path", "stderr_path", "exit_code",
            ) if key in launch
        }
    result["launch_receipt_expected_sha256"] = EXPECTED_LAUNCH_RECEIPT_SHA256
    result["process_environment_captured_by_original_run"] = False
    result["environment_scope_note"] = (
        "Only actual strict-startup receipt fields and bounded renderer logs are returned; "
        "the original child environment was not persisted as a full environment dump."
    )
    result["supervisor_log"] = safe_ref(SUPERVISOR_LOG, max_bytes=MAX_LOG_BYTES_TO_HASH,
                                         tail_bytes=MAX_LOG_TAIL_BYTES)

    manifest_rows = {row.get("row_id"): row for row in manifest.get("rows", [])}
    expected_ids = {row_id for row_id, _, _ in ROWS}
    result["manifest_row_ids_exact"] = expected_ids == set(manifest_rows) & expected_ids
    for row_id, model_id, repeat_name in ROWS:
        row_root = out / "models" / model_id / repeat_name
        row = manifest_rows.get(row_id, {})
        item: dict[str, Any] = {
            "row_id": row_id,
            "model_id": model_id,
            "manifest_status": row.get("status"),
            "failure_reason": row.get("failure_reason"),
            "traceback_tail": str(row.get("traceback", ""))[-MAX_TRACEBACK_CHARS:],
            "row_root": str(row_root),
            "logs": [],
            "strict_startup_receipts": [],
            "adapter_receipt": None,
            "control_entries": [],
        }
        log_root = row_root / "logs"
        for name in ("renderer.stdout.log", "renderer.stderr.log"):
            item["logs"].append(safe_ref(log_root / name, max_bytes=MAX_LOG_BYTES_TO_HASH,
                                          tail_bytes=MAX_LOG_TAIL_BYTES))
        receipt_dir = row_root / "strict_startup_receipts"
        if receipt_dir.is_dir() and not receipt_dir.is_symlink():
            for path in sorted(receipt_dir.glob("*.json"))[:8]:
                record, ref = safe_json(path)
                receipt_item: dict[str, Any] = {"file": ref}
                if isinstance(record, dict):
                    receipt_item["receipt"] = receipt_view(record)
                item["strict_startup_receipts"].append(receipt_item)
        control_dir = row_root / "control"
        if control_dir.is_dir() and not control_dir.is_symlink():
            item["control_entries"] = sorted(p.name for p in control_dir.iterdir())[:32]
        adapter_path = row_root / "render" / "generated_video.adapter_receipt.json"
        adapter, adapter_ref = safe_json(adapter_path)
        item["adapter_receipt"] = {
            "file": adapter_ref,
            "receipt": receipt_view(adapter) if isinstance(adapter, dict) else None,
        }
        # Report output filenames and byte counts only. Never hash or open media.
        render_dir = row_root / "render"
        if render_dir.is_dir() and not render_dir.is_symlink():
            item["render_entries_metadata_only"] = [
                {"name": p.name, "kind": "directory" if p.is_dir() else "file",
                 "bytes": p.stat().st_size if p.is_file() else None}
                for p in sorted(render_dir.iterdir())[:24]
            ]
        result["rows"].append(item)
    result["status"] = "PASS_BOUNDED_FAILURE_DIAGNOSTICS" if len(result["rows"]) == 4 else "INCOMPLETE"
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
