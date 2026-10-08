#!/usr/bin/env python3
"""Verify the pinned local Dev360 transfers and reconstruct the split ledger.

Local-only integrity operation. It does not contact the server or make a scientific
acceptance decision. Every output is create-only under verify_output_attempt01.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / ".codex/scratch/dev360_terminal_transfer_attempt06/after_plan_download_attempt01/terminal_file_download_request_index_attempt01.json"
PLAN = ROOT / ".codex/scratch/dev360_terminal_transfer_attempt05/after_inventory_download_attempt01/download_plan_attempt02.json"
INVENTORY = ROOT / ".codex/scratch/dev360_terminal_transfer_attempt04/inventory_download_attempt03/downloads_attempt03/terminal_inventory_attempt02.json"
OUTDIR = ROOT / ".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01"
LEDGER_OUT = OUTDIR / "results.jsonl"
REPORT_OUT = OUTDIR / "transfer_reconstruction_receipt_attempt01.json"

INDEX_PIN = (25198, "66d8a97e5274e559eb2cdbbfee4e736558de20281ded6cfd28825408ba72e348", "7de51026f33ca95075d3999e24a121753116b30dc4e9f8f134fd461d71d0b55a")
PLAN_PIN = (4165, "c3e9b5cd0228d08e42d1f678f05c3ea917e5f28d9e5fb49e422f5ab23882be57", "50cedbd5e2e32324ab393467af34a09a09afc662b78ca2d8fcf4e2f0f6597528")
INVENTORY_PIN = (13593, "3f343b6ff73f35e838fbecdc290ae2112dbb6be8f55f8474469c1c7b52c254ef", "e27816df44f4bbe7f21daebb2274d845cc04502309d5e3781e9625e3d55de929")
BUILDER_PIN = (100651, "4671ca2472dd6d4b20af4599560e1da65ca0e86bc5ec4cf51f4f6d54158dd49e")
RUN_ID = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10"
STAGE_ID = "controls360"
NAMES = {"result_manifest", "results_ledger", "run_status", "execution_record", "strict_determinism_attestation", "generation_repeat_qc_ledger"}


def need(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(obj: dict[str, Any], field: str) -> str:
    body = dict(obj)
    body.pop(field, None)
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha(raw)


def read_regular(path: Path, cap: int | None = None) -> bytes:
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"not a non-symlink regular file: {path}")
    need(cap is None or before.st_size <= cap, f"read cap exceeded: {path}")
    raw = path.read_bytes()
    after = path.stat()
    need(len(raw) == before.st_size and after.st_size == before.st_size and after.st_mtime_ns == before.st_mtime_ns, f"file changed while read: {path}")
    return raw


def check_pin(path: Path, n: int, digest: str) -> bytes:
    raw = read_regular(path)
    need(len(raw) == n and sha(raw) == digest, f"file pin mismatch: {path}")
    return raw


def obj(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON at {label}: {exc}") from exc
    need(isinstance(value, dict), f"JSON object expected at {label}")
    return value


def local_key(text: str) -> str:
    return os.path.normcase(os.path.abspath(text))


def response_record(path: Path, req_obj: dict[str, Any], size: int, digest: str) -> dict[str, Any]:
    raw = read_regular(path, 64 * 1024)
    response = obj(raw, str(path))
    need(response.get("ok") is True and not response.get("error"), f"download response not clean success: {path}")
    need(local_key(str(response.get("local", ""))) == local_key(str(req_obj.get("local", ""))), f"response local path mismatch: {path}")
    need(response.get("remote") == req_obj.get("remote"), f"response remote path mismatch: {path}")
    need(response.get("size") == size and response.get("sha256") == digest, f"response payload pin mismatch: {path}")
    return {"path": str(path), "bytes": len(raw), "raw_sha256": sha(raw), "status": "CLEAN_DOWNLOAD", "size": size, "sha256": digest}


def main() -> None:
    need(OUTDIR.is_dir() and not OUTDIR.is_symlink(), f"output directory missing/unsafe: {OUTDIR}")
    need(not LEDGER_OUT.exists() and not REPORT_OUT.exists(), "create-only verifier output already exists")

    index_raw = check_pin(INDEX, *INDEX_PIN[:2])
    index = obj(index_raw, "transfer index")
    need(index.get("schema") == "cross_tfg_dev360_terminal_transfer_request_index_v1", "index schema mismatch")
    need(index.get("run_id") == RUN_ID and index.get("stage_id") == STAGE_ID, "index run/stage mismatch")
    need(index.get("index_sha256") == INDEX_PIN[2] and canonical_sha(index, "index_sha256") == INDEX_PIN[2], "index self hash mismatch")
    need(index.get("status") == "READY_FOR_ROOT_DISPATCH_OF_PINNED_TERMINAL_FILE_TRANSFERS", "index status mismatch")

    evidence = index.get("source_evidence", {})
    builder_pin = evidence.get("transfer_builder", {})
    builder_path = Path(str(builder_pin.get("path", "")))
    builder_raw = check_pin(builder_path, *BUILDER_PIN)
    need(builder_pin.get("bytes") == len(builder_raw) and builder_pin.get("sha256") == sha(builder_raw), "historical transfer builder pin mismatch")

    plan_raw = check_pin(PLAN, *PLAN_PIN[:2])
    plan = obj(plan_raw, "actual download plan")
    need(plan.get("schema") == "cross_tfg_dev360_terminal_download_plan_v1" and plan.get("status") == "DOWNLOAD_PLAN_READY", "download plan status/schema mismatch")
    need(plan.get("plan_sha256") == PLAN_PIN[2] and canonical_sha(plan, "plan_sha256") == PLAN_PIN[2], "download plan self hash mismatch")
    need(plan.get("run_id") == RUN_ID and plan.get("stage_id") == STAGE_ID, "plan run/stage mismatch")
    need(plan.get("source_terminal_files_modified") is False and plan.get("source_terminal_files_deleted") is False, "plan source-preservation declaration failed")

    inventory_raw = check_pin(INVENTORY, *INVENTORY_PIN[:2])
    inventory = obj(inventory_raw, "actual terminal inventory")
    need(inventory.get("schema") == "cross_tfg_dev360_terminal_file_inventory_v1" and inventory.get("status") == "TERMINAL_INVENTORY_READY", "inventory status/schema mismatch")
    need(inventory.get("inventory_sha256") == INVENTORY_PIN[2] and canonical_sha(inventory, "inventory_sha256") == INVENTORY_PIN[2], "inventory self hash mismatch")
    need(inventory.get("run_id") == RUN_ID and inventory.get("stage_id") == STAGE_ID, "inventory run/stage mismatch")
    need(inventory.get("errors") == [] and inventory.get("all_rows_and_failures_preserved") is True, "inventory errors/source-denominator gate failed")
    need(index.get("actual_plan", {}).get("raw_sha256") == PLAN_PIN[1] and index.get("actual_inventory", {}).get("raw_sha256") == INVENTORY_PIN[1], "index actual plan/inventory raw links mismatch")
    need(index.get("actual_plan", {}).get("plan_sha256") == PLAN_PIN[2] and index.get("actual_inventory", {}).get("inventory_sha256") == INVENTORY_PIN[2], "index actual plan/inventory canonical links mismatch")
    need(plan.get("inventory_raw_sha256") == INVENTORY_PIN[1] and plan.get("inventory_sha256") == INVENTORY_PIN[2], "plan does not cite actual inventory")

    # Verify the two previous phase responses that produced the actual plan.
    for ev_role in ("actual_plan_exec_response", "actual_plan_download_response"):
        pin = evidence[ev_role]
        raw = check_pin(Path(pin["path"]), int(pin["bytes"]), str(pin["sha256"]))
        response = obj(raw, ev_role)
        need(response.get("ok") is True and not response.get("error"), f"non-clean prior response: {ev_role}")
    exec_envelope = obj(read_regular(Path(evidence["actual_plan_exec_response"]["path"])), "plan execution response envelope")
    need(exec_envelope.get("ok") is True, "plan execution bridge envelope is not ok")
    need(exec_envelope.get("code") == 0 and exec_envelope.get("timed_out") is False and exec_envelope.get("incomplete_stream") is False, "plan execution bridge did not finish cleanly")
    need(exec_envelope.get("stream_error") is None and exec_envelope.get("stderr") == "", "plan execution bridge has stream/stderr errors")
    truncated = exec_envelope.get("truncated")
    need(isinstance(truncated, dict) and truncated.get("stdout") is False and truncated.get("stderr") is False, "plan execution truncation fields are missing or non-false")
    exec_stdout = exec_envelope.get("stdout")
    need(isinstance(exec_stdout, str) and exec_stdout.strip(), "plan execution stdout is empty")
    exec_obj = obj(exec_stdout.encode("utf-8"), "plan execution stdout JSON")
    need(exec_obj.get("status") == "DOWNLOAD_PLAN_READY" and exec_obj.get("schema") == plan.get("schema"), "plan execution stdout status/schema mismatch")
    need(exec_obj.get("plan_bytes") == len(plan_raw) and exec_obj.get("plan_raw_sha256") == PLAN_PIN[1] and exec_obj.get("plan_sha256") == PLAN_PIN[2], "plan execution stdout does not bind actual downloaded plan")
    dl_obj = obj(read_regular(Path(evidence["actual_plan_download_response"]["path"])), "plan download response")
    need(dl_obj.get("size") == len(plan_raw) and dl_obj.get("sha256") == PLAN_PIN[1], "downloaded plan response differs from actual plan")
    need(local_key(str(dl_obj.get("local", ""))) == local_key(str(PLAN)) and exec_obj.get("plan_path") == dl_obj.get("remote"), "plan execution/download path linkage mismatch")

    transfers_list = index.get("artifact_transfers", [])
    transfers = {item.get("name"): item for item in transfers_list}
    need(len(transfers_list) == 6 and set(transfers) == NAMES, "transfer artifact list is not exact six")
    inv_items = {item.get("name"): item for item in inventory.get("artifacts", [])}
    plan_items = {item.get("name"): item for item in plan.get("download_items", [])}
    need(set(inv_items) == NAMES and set(plan_items) == NAMES, "inventory/plan artifact allowlist mismatch")
    for name in NAMES:
        inv = inv_items[name]
        item = plan_items[name]
        transfer = transfers[name]
        need(inv.get("exists") is True and inv.get("status") == "PINNED", f"inventory artifact not pinned: {name}")
        need((inv.get("bytes"), inv.get("raw_sha256"), inv.get("path")) == (item.get("bytes"), item.get("raw_sha256"), item.get("source_path")), f"plan/inventory mismatch: {name}")
        need((item.get("bytes"), item.get("raw_sha256"), item.get("source_path")) == (transfer.get("bytes"), transfer.get("raw_sha256"), transfer.get("source_path")), f"plan/transfer mismatch: {name}")
    requests = index.get("requests", [])
    need(len(requests) == 7, "expected exactly seven request rows")
    request_by_path = {local_key(r.get("path", "")): r for r in requests}
    need(len(request_by_path) == 7, "request paths are missing/duplicated")
    response_rows: list[dict[str, Any]] = []

    def verify_one(request_path: str, request_n: int, request_sha: str, response_path: str, local_path: str, remote_path: str, payload_n: int, payload_sha: str) -> bytes:
        req_path = Path(request_path)
        req_raw = check_pin(req_path, int(request_n), str(request_sha))
        req = obj(req_raw, f"request:{req_path.name}")
        reqrow = request_by_path.get(local_key(str(req_path)))
        need(reqrow is not None and reqrow.get("bytes") == len(req_raw) and reqrow.get("sha256") == sha(req_raw), f"request not bound by index: {req_path}")
        need(req.get("op") == "download" and local_key(str(req.get("local", ""))) == local_key(local_path) and req.get("remote") == remote_path, f"request target mismatch: {req_path}")
        response = response_record(Path(response_path), req, int(payload_n), str(payload_sha))
        response_rows.append({"request_id": reqrow.get("request_id"), "request_path": str(req_path), "request_bytes": len(req_raw), "request_sha256": sha(req_raw), "response": response})
        data = check_pin(Path(local_path), int(payload_n), str(payload_sha))
        return data

    chunks = sorted(transfers["results_ledger"].get("parts", []), key=lambda part: int(part.get("part_index", -1)))
    need(len(chunks) == 2, "ledger must have two pinned chunks")
    need(not transfers["results_ledger"].get("reconstructed_path") or not Path(transfers["results_ledger"]["reconstructed_path"]).exists(), "legacy reconstructed target already exists; refusing overwrite")
    digest = hashlib.sha256()
    offset = 0
    total = 0
    fd = os.open(str(LEDGER_OUT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as out:
            fd = -1
            for ix, part in enumerate(chunks):
                need(part.get("part_index") == ix and part.get("offset") == offset, "ledger chunks have duplicate/gapped index or offset")
                need(int(part.get("bytes", -1)) <= 6 * 1024 * 1024, "chunk exceeds staged 6 MiB limit")
                data = verify_one(part["request_path"], part["request_bytes"], part["request_sha256"], part["response_path"], part["local_path"], part["source_path"], part["bytes"], part["raw_sha256"])
                out.write(data)
                digest.update(data)
                total += len(data)
                offset += len(data)
            out.flush()
            os.fsync(out.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    ledger = transfers["results_ledger"]
    need(total == ledger.get("bytes") and offset == ledger.get("bytes") and digest.hexdigest() == ledger.get("raw_sha256"), "chunk concatenation differs from whole ledger pin")
    ledger_readback = read_regular(LEDGER_OUT)
    need(len(ledger_readback) == ledger.get("bytes") and sha(ledger_readback) == ledger.get("raw_sha256"), "reconstructed ledger file readback mismatch")

    direct: list[dict[str, Any]] = []
    for name in sorted(NAMES - {"results_ledger"}):
        item = transfers[name]
        need(item.get("status") == "DIRECT_DOWNLOAD", f"unexpected transfer mode: {name}")
        data = verify_one(item["request_path"], item["request_bytes"], item["request_sha256"], item["response_path"], item["local_path"], item["source_path"], item["bytes"], item["raw_sha256"])
        direct.append({"name": name, "source_path": item["source_path"], "local_path": item["local_path"], "bytes": len(data), "raw_sha256": sha(data)})
    need(len(response_rows) == 7 and len(request_by_path) == 7, "not all seven request/response pairs verified")

    artifacts = sorted(direct + [{"name": "results_ledger", "source_path": ledger["source_path"], "local_path": str(LEDGER_OUT), "bytes": len(ledger_readback), "raw_sha256": sha(ledger_readback)}], key=lambda r: r["name"])
    report: dict[str, Any] = {
        "schema": "cross_tfg_dev360_terminal_transfer_reconstruction_readback_v1",
        "status": "PASS_TRANSFER_INTEGRITY_AND_LEDGER_RECONSTRUCTION",
        "run_id": RUN_ID,
        "stage_id": STAGE_ID,
        "scope": {"local_only": True, "remote_dispatch": False, "scientific_acceptance_inferred": False, "producer_files_modified": False, "source_files_deleted": False},
        "transfer_index": {"path": str(INDEX), "bytes": len(index_raw), "raw_sha256": sha(index_raw), "self_sha256": INDEX_PIN[2]},
        "historical_transfer_builder": {"path": str(builder_path), "bytes": len(builder_raw), "raw_sha256": sha(builder_raw)},
        "actual_plan": {"path": str(PLAN), "bytes": len(plan_raw), "raw_sha256": sha(plan_raw), "self_sha256": PLAN_PIN[2]},
        "actual_inventory": {"path": str(INVENTORY), "bytes": len(inventory_raw), "raw_sha256": sha(inventory_raw), "self_sha256": INVENTORY_PIN[2]},
        "request_response_pairs": sorted(response_rows, key=lambda item: str(item["request_id"])),
        "actual_six_artifacts": artifacts,
        "ledger_reconstruction": {"part_count": len(chunks), "contiguous_offsets": True, "part_byte_sum": total, "bytes": len(ledger_readback), "raw_sha256": sha(ledger_readback), "path": str(LEDGER_OUT)},
    }
    report["receipt_sha256"] = canonical_sha(report, "receipt_sha256")
    report_raw = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    fd = os.open(str(REPORT_OUT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(report_raw)
        stream.flush()
        os.fsync(stream.fileno())
    readback = read_regular(REPORT_OUT)
    need(readback == report_raw, "receipt readback differs from written bytes")
    print(json.dumps({"status": report["status"], "receipt_path": str(REPORT_OUT), "receipt_bytes": len(readback), "receipt_raw_sha256": sha(readback), "receipt_self_sha256": report["receipt_sha256"], "results_jsonl_path": str(LEDGER_OUT), "results_jsonl_bytes": len(ledger_readback), "results_jsonl_sha256": sha(ledger_readback)}, sort_keys=True))


if __name__ == "__main__":
    main()
