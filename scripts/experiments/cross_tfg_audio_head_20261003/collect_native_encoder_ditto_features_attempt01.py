#!/usr/bin/env python3
"""Collect and strictly restore the completed CPU Ditto encoder probe.

This is a local receiver for a raw `tar -cf - .` SSH stdout stream. It never
connects to the remote host and contains no credentials. The remote output tree
is read-only; no remote archive or copy is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from typing import Any, BinaryIO

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
COPY_BLOCK = 1024 * 1024
EXPECTED_CONDITIONS = (
    "natural",
    "global_minus_6db",
    "global_plus_6db",
    "fixed_rms_0p0376838173",
    "local_r090_accent015_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent_0.30_tilt0",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(COPY_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def verify_local_pin(pin: dict[str, Any], label: str) -> Path:
    path = Path(pin["path"])
    if path.is_symlink() or not path.is_file():
        fail(f"{label}: expected a regular, non-symlink local file")
    if path.stat().st_size != pin["bytes"] or sha_file(path) != pin["sha256"]:
        fail(f"{label}: local size/SHA-256 differs from the sealed handoff pin")
    return path


def read_outer_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    outer = json.loads(raw)
    trunc = outer.get("truncated")
    if (
        outer.get("ok") is not True
        or outer.get("code") != 0
        or outer.get("timed_out") is not False
        or outer.get("incomplete_stream") is not False
        or outer.get("stream_error") is not None
        or not isinstance(trunc, dict)
        or trunc.get("stdout") is not False
        or trunc.get("stderr") is not False
        or outer.get("stderr") != ""
    ):
        fail(f"{label}: outer bridge response is not a complete clean success")
    inner = json.loads(outer["stdout"])
    if not isinstance(inner, dict):
        fail(f"{label}: stdout payload is not a JSON object")
    return outer, inner


def normalize_member_name(raw_name: str) -> str:
    if not raw_name or "\\" in raw_name:
        fail(f"unsafe tar member name: {raw_name!r}")
    if raw_name.startswith("/"):
        fail(f"absolute tar member path is forbidden: {raw_name!r}")
    name = raw_name
    while name.startswith("./"):
        name = name[2:]
    if name in ("", "."):
        return "."
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        fail(f"unsafe tar member path: {raw_name!r}")
    return pure.as_posix()


def read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    f = tar.extractfile(member)
    if f is None:
        fail(f"cannot read regular tar member {member.name!r}")
    data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES or len(data) != member.size:
        fail(f"tar member size exceeds cap or is truncated: {member.name!r}")
    return data


def copy_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> tuple[int, str]:
    f = tar.extractfile(member)
    if f is None:
        fail(f"cannot extract regular tar member {member.name!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    count = 0
    with target.open("xb") as out:
        while True:
            block = f.read(COPY_BLOCK)
            if not block:
                break
            count += len(block)
            if count > MAX_MEMBER_BYTES:
                fail(f"extracted member exceeded byte cap: {member.name!r}")
            h.update(block)
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    if count != member.size:
        fail(f"extracted member length mismatch: {member.name!r}")
    return count, h.hexdigest()


def stream_stdin_to_archive(archive_path: Path) -> tuple[int, str]:
    if archive_path.exists() or archive_path.is_symlink():
        fail(f"archive destination already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        fail(f"partial archive path already exists: {partial}")
    h = hashlib.sha256()
    count = 0
    try:
        with partial.open("xb") as out:
            while True:
                block = sys.stdin.buffer.read(COPY_BLOCK)
                if not block:
                    break
                count += len(block)
                if count > MAX_ARCHIVE_BYTES:
                    fail("incoming tar stream exceeded the 64 MiB bound")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if count == 0:
            fail("empty tar stream")
        os.replace(partial, archive_path)
        return count, h.hexdigest()
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def validate_plan_and_terminal(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pinned: dict[str, Path] = {}
    for label, pin in contract["local_inputs"].items():
        pinned[label] = verify_local_pin(pin, label)

    plan = json.loads(pinned["probe_plan"].read_text(encoding="utf-8"))
    manifest = json.loads(pinned["source_manifest"].read_text(encoding="utf-8"))
    fit_plan = json.loads(pinned["frozen_fit_plan"].read_text(encoding="utf-8"))
    _, preflight = read_outer_json(pinned["preflight_response"], "CPU-only preflight")
    _, launch = read_outer_json(pinned["launch_response"], "async launch")
    _, status = read_outer_json(pinned["status_response"], "terminal status")

    if preflight.get("schema") != "cross_tfg_native_audio_encoder_preflight_v1" or preflight.get("status") != "PASS_CPU_PREFLIGHT_ONLY":
        fail("pinned preflight is not PASS_CPU_PREFLIGHT_ONLY")
    if preflight.get("frontend") != "ditto" or preflight.get("scope", {}).get("cuda_context_initialized") is not False:
        fail("preflight did not establish the frozen Ditto CPU-only/no-CUDA route")
    if preflight.get("scope", {}).get("development_audio_opened") is not False or preflight.get("scope", {}).get("confirmation_audio_opened") is not False:
        fail("preflight scope opened Dev or confirmation")
    if launch.get("schema") != "cross_tfg_native_encoder_ditto_async_launch_v1" or launch.get("status") != "LAUNCHED":
        fail("launch evidence is not a successful async launch")
    if launch.get("admission", {}).get("gate") != "PASS" or launch.get("admission", {}).get("status") != "ADMITTED":
        fail("launch admission did not pass")
    if status.get("schema") != "cross_tfg_native_encoder_execution_status_v1" or status.get("run_id") != contract["expected_run_id"]:
        fail("terminal status schema/run identity mismatch")
    if status.get("summary", {}).get("status") != "COMPLETE" or status.get("summary", {}).get("completed_units") != 56 or status.get("summary", {}).get("planned_units") != 56:
        fail("Ditto source-condition run is not 56/56 COMPLETE")
    if status.get("feature_npz_count") != 56 or status.get("output_file_count") != 58 or status.get("output_bytes") != contract["expected_output_bytes"]:
        fail("Ditto terminal file/byte denominator differs from the pinned status contract")
    if status.get("output_cap_exceeded") != "0" or status.get("stderr_tail") != [] or status.get("supervisor_stderr_tail") != []:
        fail("Ditto terminal status reports output-cap or stderr failures")
    if status.get("worker", {}).get("alive") is not False or status.get("supervisor", {}).get("alive") is not False:
        fail("Ditto processes are not terminal")
    if status.get("exit_code") not in (0, "0") or status.get("receipt", {}).get("status") != "COMPLETE":
        fail("Ditto process or output receipt did not complete successfully")
    if status.get("receipt", {}).get("output_file_count") != 57:
        fail("remote output receipt does not declare 56 features plus summary")
    if launch.get("output_dir") != status.get("out_dir"):
        fail("launch and terminal output paths differ")
    if (launch.get("supervisor", {}).get("pid"), launch.get("supervisor", {}).get("start_ticks")) != (
        status.get("supervisor", {}).get("pid"), status.get("supervisor", {}).get("expected_start_ticks")
    ):
        fail("launch process identity differs from the terminal status identity")
    if preflight.get("probe_protocol", {}).get("sha256") != contract["local_inputs"]["probe_plan"]["sha256"]:
        fail("preflight used a different probe plan than the pinned local plan")
    if preflight.get("plan", {}).get("sha256") != contract["local_inputs"]["frozen_fit_plan"]["sha256"]:
        fail("preflight used a different frozen fit plan")
    if plan.get("status") != "PREPARED_FOR_ROOT_REVIEW_NOT_DISPATCHED":
        fail("probe-plan preparation status differs from the exact executed attempt")
    if plan.get("scope", {}).get("confirmation_audio_opened") is not False or plan.get("scope", {}).get("development_audio_opened") is not False:
        fail("probe plan scope is not closed to Dev/confirmation")
    if len(plan.get("input_rows", [])) != 8 or len(plan.get("conditions", [])) != 7:
        fail("frozen source-condition plan is not exactly 8×7")
    if tuple(x.get("condition_id") for x in plan["conditions"]) != EXPECTED_CONDITIONS:
        fail("condition roster/order differs from the frozen seven-condition plan")
    if len({x.get("sample_id") for x in plan["input_rows"]}) != 8:
        fail("source roster has duplicates or missing sample IDs")
    frontend = plan.get("frontends", [{}])[0]
    if frontend.get("model_id") != "ditto_hubert_fixed_portrait_attempt18" or frontend.get("onnx_sha256") != "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697":
        fail("probe plan model identity/ONNX pin is not the frozen Ditto frontend")
    if manifest.get("source_model", {}).get("model_id") != frontend.get("model_id") or manifest.get("source_model", {}).get("hubert_onnx_sha256") != frontend.get("onnx_sha256"):
        fail("source manifest model/ONNX binding differs from the frozen plan")
    source_map = {x["sample_id"]: x for x in plan["input_rows"]}
    if len(source_map) != 8 or not manifest.get("probe_plan", {}).get("sha256"):
        fail("source manifest/probe plan bindings are incomplete")
    if manifest.get("runner", {}).get("sha256") != contract["local_inputs"]["runner"]["sha256"]:
        fail("source manifest runner hash differs from the actual executed source")
    if manifest.get("probe_plan", {}).get("sha256") != contract["local_inputs"]["probe_plan"]["sha256"]:
        fail("source manifest plan hash differs from the actual executed plan")
    if manifest.get("frozen_fit_plan", {}).get("sha256") != contract["local_inputs"]["frozen_fit_plan"]["sha256"]:
        fail("source manifest fit-plan hash differs from the frozen plan")
    if fit_plan.get("run_id") != plan.get("source_plan", {}).get("run_id"):
        fail("Ditto probe is not bound to the declared shared fit plan")
    return plan, manifest, status, source_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("schema") != "cross_tfg_native_ditto_feature_collection_contract_v1":
        fail("unrecognized collection contract")
    if args.restore_root.exists() or args.restore_root.is_symlink() or args.receipt.exists() or args.receipt.is_symlink():
        fail("restore output or receipt path already exists")
    plan, manifest, status, source_map = validate_plan_and_terminal(contract)
    archive_bytes, archive_sha = stream_stdin_to_archive(args.archive)

    with tarfile.open(args.archive, mode="r:") as tar:
        raw_members = tar.getmembers()
        normalized: dict[str, tarfile.TarInfo] = {}
        for member in raw_members:
            name = normalize_member_name(member.name)
            if name in normalized:
                fail(f"duplicate archive member: {name}")
            normalized[name] = member
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                fail(f"non-regular tar member is forbidden: {name}")
            if not (member.isdir() or member.isfile()):
                fail(f"unsupported tar member type: {name}")
            if member.isfile() and (member.size < 0 or member.size > MAX_MEMBER_BYTES):
                fail(f"tar member size outside bound: {name}")
        if len(normalized) > 64:
            fail("archive member count exceeds the fixed 64-member bound")

        sample_ids = [row["sample_id"] for row in plan["input_rows"]]
        expected_files = {"summary.json", "receipt.json"}
        expected_files.update(f"features/{sid}__{condition}.npz" for sid in sample_ids for condition in EXPECTED_CONDITIONS)
        expected_dirs = {"features"}
        actual_files = {name for name, member in normalized.items() if member.isfile()}
        actual_dirs = {name for name, member in normalized.items() if member.isdir() and name != "."}
        if actual_files != expected_files or not actual_dirs.issubset(expected_dirs) or actual_dirs != expected_dirs:
            fail(f"archive member set mismatch: files={len(actual_files)}, dirs={sorted(actual_dirs)}")

        receipt_raw = read_member(tar, normalized["receipt.json"])
        summary_raw = read_member(tar, normalized["summary.json"])
        remote_receipt = json.loads(receipt_raw)
        summary = json.loads(summary_raw)
        if remote_receipt.get("schema") != "cross_tfg_native_audio_encoder_probe_receipt_v1" or remote_receipt.get("status") != "COMPLETE":
            fail("embedded native encoder receipt is not COMPLETE")
        if remote_receipt.get("summary_sha256") != status.get("receipt", {}).get("summary_sha256") or sha_bytes(summary_raw) != remote_receipt.get("summary_sha256"):
            fail("embedded summary hash differs from both receipt and terminal status")
        if remote_receipt.get("output_file_count") != 57:
            fail("embedded receipt denominator is not 56 feature files plus summary")
        if remote_receipt.get("scope", {}).get("development_audio_opened") is not False or remote_receipt.get("scope", {}).get("confirmation_audio_opened") is not False:
            fail("embedded receipt scope includes Dev or confirmation")
        if summary.get("schema") != "cross_tfg_native_audio_encoder_probe_v1" or summary.get("status") != "COMPLETE" or summary.get("frontend") != "ditto":
            fail("embedded summary schema/status/frontend mismatch")
        if summary.get("plan", {}).get("sha256") != contract["local_inputs"]["frozen_fit_plan"]["sha256"]:
            fail("embedded summary does not bind the exact frozen fit plan")
        if summary.get("row_count") != 56 or len(summary.get("rows", [])) != 56:
            fail("embedded summary does not contain all 56 source-condition rows")
        if summary.get("scope", {}).get("development_audio_opened") is not False or summary.get("scope", {}).get("confirmation_audio_opened") is not False:
            fail("summary scope includes Dev or confirmation")
        if summary.get("scope", {}).get("generator_model_forward") is not False or summary.get("scope", {}).get("video_created_or_read") is not False or summary.get("scope", {}).get("syncnet_forward_or_scores_read") is not False:
            fail("summary includes generator/video/SyncNet activity outside the probe scope")
        runtime = summary.get("runtime", {})
        if runtime.get("device") != "cpu" or runtime.get("onnxruntime") is None or summary.get("model_id") != "ditto_hubert_fixed_portrait_attempt18":
            fail("summary runtime/model is not the frozen CPU ONNX Ditto path")
        if summary.get("condition_order") != list(EXPECTED_CONDITIONS):
            fail("summary condition ordering differs from the frozen plan")

        remote_entries: dict[str, dict[str, Any]] = {}
        for record in remote_receipt.get("outputs", []):
            remote_path = PurePosixPath(record.get("path", ""))
            if remote_path.is_absolute() is False or ".." in remote_path.parts:
                fail("remote output receipt contains an unsafe path")
            basename = remote_path.name
            rel = f"features/{basename}" if remote_path.parent.name == "features" else basename
            if rel in remote_entries:
                fail(f"duplicate output in embedded receipt: {rel}")
            remote_entries[rel] = record
        if set(remote_entries) != (expected_files - {"receipt.json"}):
            fail("embedded output receipt does not bind exactly the summary and 56 features")
        total_file_bytes = 0
        extracted_records = []
        row_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in summary["rows"]:
            key = (row.get("sample_id"), row.get("condition"))
            if key in row_by_key or key[0] not in source_map or key[1] not in EXPECTED_CONDITIONS:
                fail(f"duplicate/unplanned summary row: {key}")
            row_by_key[key] = row
        expected_pairs = {(sid, condition) for sid in source_map for condition in EXPECTED_CONDITIONS}
        if set(row_by_key) != expected_pairs:
            fail("summary rows do not exactly cover the frozen 8×7 roster")

        validated_files: dict[str, tuple[tarfile.TarInfo, int, str]] = {}
        for rel in sorted(expected_files - {"receipt.json"}):
            member = normalized[rel]
            payload_hash = hashlib.sha256()
            size = 0
            f = tar.extractfile(member)
            if f is None:
                fail(f"cannot hash archived member: {rel}")
            while True:
                block = f.read(COPY_BLOCK)
                if not block:
                    break
                size += len(block)
                if size > MAX_MEMBER_BYTES:
                    fail(f"archived member exceeded size limit: {rel}")
                payload_hash.update(block)
            digest = payload_hash.hexdigest()
            record = remote_entries[rel]
            if size != record.get("bytes") or digest != record.get("sha256"):
                fail(f"receipt raw member pin mismatch: {rel}")
            total_file_bytes += size
            validated_files[rel] = (member, size, digest)

        if total_file_bytes + normalized["receipt.json"].size != contract["expected_output_bytes"]:
            fail("sum of all 58 archive files differs from terminal output byte count")
        summary_digest = sha_bytes(summary_raw)
        if summary_digest != status["receipt"]["summary_sha256"]:
            fail("summary digest differs from terminal receipt")

        for (sid, condition), row in row_by_key.items():
            expected_input = source_map[sid]
            source = row.get("source_pcm", {})
            if source.get("sample_count") != expected_input["natural_pcm"]["sample_count"] or source.get("sample_rate_hz") != 16000:
                fail(f"source clock/sample count differs for {sid}/{condition}")
            if source.get("pcm_float32_sha256") != expected_input["natural_pcm"]["pcm_sha256"] or source.get("wav_sha256") != expected_input["natural_pcm"]["file_sha256"]:
                fail(f"source PCM digest differs for {sid}/{condition}")
            if row.get("status") != "PASS" or row.get("speaker_id") != expected_input["speaker_id"]:
                fail(f"non-PASS or speaker mismatch for {sid}/{condition}")
            arr = row.get("feature_array", {})
            rel = f"features/{sid}__{condition}.npz"
            feature_path = PurePosixPath(arr.get("path", ""))
            if feature_path.name != f"{sid}__{condition}.npz" or feature_path.parent.name != "features":
                fail(f"summary feature path does not identify its own source/condition file for {rel}")
            if arr.get("bytes") != validated_files[rel][1] or arr.get("sha256") != validated_files[rel][2]:
                fail(f"summary array reference differs from receipt/archive for {rel}")
            if arr.get("shape") != [((expected_input["natural_pcm"]["sample_count"] + 639) // 640), 1024]:
                fail(f"Ditto feature shape is not ceil(N/640)×1024 for {rel}")
            if row.get("clock_map", {}).get("fps_num") != 25 or row.get("clock_map", {}).get("fps_den") != 1:
                fail(f"query clock is not exactly i/25 for {rel}")
            clock_map = row.get("clock_map", {})
            if clock_map.get("query_count") != arr["shape"][0] or len(clock_map.get("queries", [])) != arr["shape"][0]:
                fail(f"clock-map query count differs from feature count for {rel}")
            for q, query in enumerate(clock_map["queries"]):
                expected_start = q * 640
                expected_stop = min(expected_input["natural_pcm"]["sample_count"], (q + 1) * 640)
                if (query.get("query_index"), query.get("query_timestamp_num"), query.get("query_timestamp_den"), query.get("source_clock_frame_index"), query.get("source_sample_start"), query.get("source_sample_stop_exclusive")) != (q, q, 25, q, expected_start, expected_stop):
                    fail(f"clock-map source/query sample alignment mismatch at {rel} query {q}")

        # Create the restore only after the entire tar, receipt, summary, and all
        # member hashes have passed the strict input validation above.
        args.restore_root.mkdir(parents=True, exist_ok=False)
        for dirname in sorted(expected_dirs):
            (args.restore_root / dirname).mkdir(parents=True, exist_ok=True)
        for rel, (member, expected_size, expected_sha) in validated_files.items():
            size, digest = copy_member(tar, member, args.restore_root / Path(*PurePosixPath(rel).parts))
            if size != expected_size or digest != expected_sha:
                fail(f"restored member differs from already-validated archive member: {rel}")
            extracted_records.append({"path": rel, "bytes": size, "sha256": digest})
        receipt_size, receipt_sha = copy_member(tar, normalized["receipt.json"], args.restore_root / "receipt.json")
        if receipt_size != len(receipt_raw) or receipt_sha != sha_bytes(receipt_raw):
            fail("restored embedded receipt differs from archive bytes")
        extracted_records.append({"path": "receipt.json", "bytes": receipt_size, "sha256": receipt_sha})

    local_receipt = {
        "schema": "cross_tfg_native_ditto_feature_collection_receipt_v1",
        "status": "PASS",
        "source_scope": "historical_fit_only_8_clips_4_speakers_7_conditions",
        "remote_run_id": status["run_id"],
        "remote_output_dir": status["out_dir"],
        "terminal_summary_sha256": status["receipt"]["summary_sha256"],
        "terminal_status_response": contract["local_inputs"]["status_response"],
        "launch_response": contract["local_inputs"]["launch_response"],
        "cpu_preflight_response": contract["local_inputs"]["preflight_response"],
        "probe_plan": contract["local_inputs"]["probe_plan"],
        "runner": contract["local_inputs"]["runner"],
        "source_manifest": contract["local_inputs"]["source_manifest"],
        "frozen_fit_plan": contract["local_inputs"]["frozen_fit_plan"],
        "archive": {"path": str(args.archive), "bytes": archive_bytes, "sha256": archive_sha},
        "member_count": len(normalized),
        "file_count": len(actual_files),
        "total_remote_file_bytes": total_file_bytes + normalized["receipt.json"].size,
        "restored_root": str(args.restore_root),
        "restored_files": sorted(extracted_records, key=lambda x: x["path"]),
        "checks": {
            "56_of_56_source_condition_rows": True,
            "8_sources_times_7_conditions": True,
            "every_member_pinned_by_remote_receipt": True,
            "every_restored_member_rehashed": True,
            "Ditto_CPU_ONNX_no_CUDA": True,
            "exact_source_clock_25fps": True,
            "development_and_confirmation_closed": True,
            "generator_video_and_syncnet_not_run": True,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(local_receipt, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": "PASS", "archive_bytes": archive_bytes, "archive_sha256": archive_sha, "restored_files": len(extracted_records)+1, "receipt": str(args.receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
