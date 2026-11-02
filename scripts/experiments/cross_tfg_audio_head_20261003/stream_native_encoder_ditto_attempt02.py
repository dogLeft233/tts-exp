#!/usr/bin/env python3
"""Stream the sealed Ditto feature output to a local strict receiver.

The SSH command only runs tar over the completed read-only output tree. No
remote archive, chunk files, base64 payloads, or credentials are written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, BinaryIO

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 1024 * 1024
BLOCK_BYTES = 1024 * 1024
PASSWORD_ENV = "TTS_SSH_PASSWORD_PRIVATE"
LOCAL_PYTHON = Path(r"E:\Documents\tts-exp\.codex\scratch\world_v5_archive_runtime\Scripts\python.exe")
REPOSITORY_ROOT = Path(r"E:\Documents\tts-exp")
V4_STREAM_HELPER = Path(r"E:\Documents\tts-exp\.codex\scratch\cross_tfg_audio_head_20261003\v4_memfd_direct_stream_attempt03\stream_v4_memfd_direct_to_c_attempt03.py")
V4_STREAM_HELPER_SHA256 = "d8c6b367facfc03261996637955205d152db477a0eb6fefb91477b08bea36825"
REMOTE_OUTPUT = "/dev/shm/cross_tfg_audio_head_20261003/native_encoder_ditto_retry_execution_attempt04/output"
REMOTE_HOST = "connect.weste.seetacloud.com"
REMOTE_PORT = 12349
REMOTE_USERNAME = "root"
REMOTE_HOST_KEY_SHA256 = "SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BLOCK_BYTES), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def load_module(name: str, path: Path, expected_sha256: str):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PINNED_SOURCE_NOT_REGULAR:{name}")
    digest, _ = sha_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"PINNED_SOURCE_SHA_MISMATCH:{name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"PINNED_SOURCE_IMPORT_SPEC_FAILED:{name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_clean_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CONTRACT_NOT_REGULAR")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("CONTRACT_NOT_OBJECT")
    return value


def create_local_receiver(
    collector: Path,
    contract: Path,
    output_root: Path,
) -> tuple[subprocess.Popen[bytes], dict[str, Path], BinaryIO, BinaryIO]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    if output_root.drive.upper() != "C:":
        raise RuntimeError("OUTPUT_ROOT_MUST_BE_FRESH_C_PATH")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    parent = output_root.parent.resolve(strict=True)
    if not parent.is_dir() or parent.drive.upper() != "C:":
        raise RuntimeError("OUTPUT_PARENT_MUST_BE_C_DIRECTORY")
    free_bytes = shutil.disk_usage(parent).free
    required = 2 * MAX_ARCHIVE_BYTES + (1 << 30)
    if free_bytes < required:
        raise RuntimeError(f"LOCAL_STORAGE_GATE_FAILED:{free_bytes}:{required}")
    output_root.mkdir(parents=False, exist_ok=False)
    archive = output_root / "ditto_native_encoder_features.tar"
    restore_root = output_root / "restored"
    receipt = output_root / "collection_receipt.json"
    stdout_log = output_root / "collector.stdout.jsonl"
    stderr_log = output_root / "collector.stderr.log"
    for path in (archive, restore_root, receipt, stdout_log, stderr_log):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"RECEIVER_OUTPUT_ALREADY_EXISTS:{path}")
    command = [
        str(LOCAL_PYTHON),
        str(collector.resolve()),
        "--contract",
        str(contract.resolve()),
        "--archive",
        str(archive),
        "--restore-root",
        str(restore_root),
        "--receipt",
        str(receipt),
    ]
    if not LOCAL_PYTHON.is_file() or Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError("PINNED_LOCAL_PYTHON_MISMATCH")
    stdout_file = stdout_log.open("xb")
    stderr_file = stderr_log.open("xb")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        cwd=str(REPOSITORY_ROOT.resolve()),
        )
    except BaseException:
        stdout_file.close()
        stderr_file.close()
        raise
    return proc, {
        "archive": archive,
        "restore_root": restore_root,
        "receipt": receipt,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
    }, stdout_file, stderr_file


def start_local_drain(
    stream: BinaryIO,
    target: BinaryIO,
    label: str,
    state: dict[str, Any],
) -> threading.Thread:
    def run() -> None:
        count = 0
        try:
            while True:
                block = stream.read(64 * 1024)
                if not block:
                    break
                count += len(block)
                if count > MAX_DIAGNOSTIC_BYTES:
                    state["overflow"] = f"{label}_DIAGNOSTIC_EXCEEDED_CAP"
                    break
                target.write(block)
            target.flush()
        except BaseException as exc:
            state["error"] = f"{label}:{type(exc).__name__}:{exc}"
        finally:
            state[f"{label}_bytes"] = count
            try:
                target.close()
            except OSError:
                pass

    thread = threading.Thread(target=run, name=f"drain-{label}", daemon=True)
    thread.start()
    return thread


def remote_readonly_command() -> str:
    return "set -eu; test -d " + shlex.quote(REMOTE_OUTPUT) + "; exec tar -C " + shlex.quote(REMOTE_OUTPUT) + " -cf - ."


def stream_to_receiver(
    client: Any,
    receiver: subprocess.Popen[bytes],
    stdout_file: BinaryIO,
    stderr_file: BinaryIO,
) -> dict[str, Any]:
    if receiver.stdin is None or receiver.stdout is None or receiver.stderr is None:
        raise RuntimeError("LOCAL_RECEIVER_PIPES_MISSING")
    local_state: dict[str, Any] = {}
    stdout_thread = start_local_drain(receiver.stdout, stdout_file, "collector_stdout", local_state)
    stderr_thread = start_local_drain(receiver.stderr, stderr_file, "collector_stderr", local_state)
    remote_stdin, remote_stdout, remote_stderr = client.exec_command(
        remote_readonly_command(), timeout=None, get_pty=False
    )
    remote_channel = remote_stdout.channel
    remote_stdin.close()
    remote_state: dict[str, Any] = {"stderr_bytes": 0, "stderr_prefix": bytearray(), "stderr_error": None}

    def drain_remote_stderr() -> None:
        held = remote_state["stderr_prefix"]
        try:
            while True:
                block = remote_channel.recv_stderr(64 * 1024)
                if not block:
                    if remote_channel.exit_status_ready() and not remote_channel.recv_stderr_ready():
                        break
                    time.sleep(0.01)
                    continue
                remote_state["stderr_bytes"] += len(block)
                if len(held) < MAX_DIAGNOSTIC_BYTES:
                    held.extend(block[: MAX_DIAGNOSTIC_BYTES - len(held)])
        except BaseException as exc:
            remote_state["stderr_error"] = f"{type(exc).__name__}:{exc}"

    remote_stderr_thread = threading.Thread(target=drain_remote_stderr, name="drain-ssh-stderr", daemon=True)
    remote_stderr_thread.start()
    archive_hash = hashlib.sha256()
    archive_bytes = 0
    try:
        while True:
            if remote_channel.recv_ready():
                block = remote_channel.recv(BLOCK_BYTES)
                if not block:
                    if remote_channel.exit_status_ready() and not remote_channel.recv_ready():
                        break
                    continue
                archive_bytes += len(block)
                if archive_bytes > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("REMOTE_TAR_STREAM_EXCEEDED_64MIB_BOUND")
                archive_hash.update(block)
                try:
                    receiver.stdin.write(block)
                    receiver.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(f"LOCAL_RECEIVER_REJECTED_STREAM:{exc}") from exc
                continue
            if remote_channel.exit_status_ready():
                if not remote_channel.recv_ready():
                    break
            else:
                time.sleep(0.01)
        receiver.stdin.close()
        rc_remote = remote_channel.recv_exit_status()
        remote_stderr_thread.join(timeout=120)
        if remote_stderr_thread.is_alive():
            raise TimeoutError("REMOTE_STDERR_DRAIN_TIMEOUT")
        if remote_state["stderr_error"] is not None:
            raise RuntimeError("REMOTE_STDERR_DRAIN_FAILED:" + str(remote_state["stderr_error"]))
        if rc_remote != 0:
            diagnostic = bytes(remote_state["stderr_prefix"]).decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"REMOTE_TAR_FAILED:{rc_remote}:{diagnostic}")
        if remote_state["stderr_bytes"] != 0:
            diagnostic = bytes(remote_state["stderr_prefix"]).decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"REMOTE_TAR_STDERR_NOT_EMPTY:{remote_state['stderr_bytes']}:{diagnostic}")
        receiver_rc = receiver.wait(timeout=3600)
        stdout_thread.join(timeout=60)
        stderr_thread.join(timeout=60)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise TimeoutError("LOCAL_RECEIVER_LOG_DRAIN_TIMEOUT")
        if local_state.get("overflow") or local_state.get("error"):
            raise RuntimeError("LOCAL_RECEIVER_LOG_FAILURE:" + str(local_state))
        output_root = Path(stdout_file.name).parent
        stdout_path = output_root / "collector.stdout.jsonl"
        stderr_path = output_root / "collector.stderr.log"
        stdout_raw = stdout_path.read_bytes()
        stderr_raw = stderr_path.read_bytes()
        if receiver_rc != 0 or stderr_raw:
            raise RuntimeError(
                "LOCAL_RECEIVER_FAILED:" + str(receiver_rc) + ":" + stderr_raw.decode("utf-8", "replace")[-3000:]
            )
        lines = [line for line in stdout_raw.decode("utf-8", "strict").splitlines() if line]
        if len(lines) != 1:
            raise RuntimeError("LOCAL_RECEIVER_DID_NOT_RETURN_ONE_JSON_RECORD")
        collection = json.loads(lines[0])
        if not isinstance(collection, dict) or collection.get("status") != "PASS":
            raise RuntimeError("LOCAL_COLLECTION_RECEIPT_NOT_PASS")
        if collection.get("archive_bytes") != archive_bytes or collection.get("archive_sha256") != archive_hash.hexdigest():
            raise RuntimeError("STREAM_HASH_DIFFERS_FROM_STRICT_RECEIVER")
        receipt_path = output_root / "collection_receipt.json"
        raw_receipt = receipt_path.read_bytes()
        receipt = json.loads(raw_receipt.decode("utf-8"))
        if receipt.get("status") != "PASS" or receipt.get("archive", {}).get("bytes") != archive_bytes or receipt.get("archive", {}).get("sha256") != archive_hash.hexdigest():
            raise RuntimeError("COLLECTION_RECEIPT_STREAM_BINDING_MISMATCH")
        if receipt.get("file_count") != 58 or len(receipt.get("restored_files", [])) != 58:
            raise RuntimeError("STRICT_RESTORE_FILE_DENOMINATOR_MISMATCH")
        return {
            "schema": "cross_tfg_native_ditto_direct_stream_receipt_v1",
            "status": "DIRECT_STREAM_AND_STRICT_RESTORE_PASS",
            "remote_read_only_command": remote_readonly_command(),
            "remote_exit_code": rc_remote,
            "remote_stderr_bytes": remote_state["stderr_bytes"],
            "stream_bytes": archive_bytes,
            "stream_sha256": archive_hash.hexdigest(),
            "collector_exit_code": receiver_rc,
            "collector_stdout_bytes": len(stdout_raw),
            "collector_stderr_bytes": len(stderr_raw),
            "collection_receipt": {"path": str(receipt_path), "bytes": len(raw_receipt), "sha256": sha_bytes(raw_receipt)},
            "restored_file_count": len(receipt["restored_files"]),
            "output_root": str(output_root),
            "source_mutation_performed": False,
            "remote_archive_copy_created": False,
            "credential_persisted": False,
        }
    except BaseException:
        try:
            receiver.stdin.close()
        except (OSError, ValueError):
            pass
        if receiver.poll() is None:
            try:
                receiver.terminate()
                receiver.wait(timeout=10)
            except BaseException:
                try:
                    receiver.kill()
                except OSError:
                    pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--port", type=int, default=REMOTE_PORT)
    parser.add_argument("--username", default=REMOTE_USERNAME)
    parser.add_argument("--host-key-sha256", default=REMOTE_HOST_KEY_SHA256)
    args = parser.parse_args()
    if sys.version.split()[0] != "3.13.5" or Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError("PINNED_LOCAL_PYTHON_MISMATCH")
    collector_raw = args.collector.read_bytes()
    contract_raw = args.contract.read_bytes()
    contract = json.loads(contract_raw.decode("utf-8"))
    if contract.get("schema") != "cross_tfg_native_ditto_feature_collection_contract_v1":
        raise RuntimeError("COLLECTION_CONTRACT_SCHEMA_MISMATCH")
    collector_module = load_module("native_encoder_ditto_collector", args.collector, sha_bytes(collector_raw))
    collector_pin = contract.get("source", {})
    if (
        Path(collector_pin.get("path", "")).resolve() != args.collector.resolve()
        or collector_pin.get("bytes") != len(collector_raw)
        or collector_pin.get("sha256") != sha_bytes(collector_raw)
    ):
        raise RuntimeError("COLLECTOR_CONTRACT_SOURCE_BINDING_MISMATCH")
    driver_pin = contract.get("stream_driver", {})
    driver_raw = Path(__file__).read_bytes()
    if (
        Path(driver_pin.get("path", "")).resolve() != Path(__file__).resolve()
        or driver_pin.get("bytes") != len(driver_raw)
        or driver_pin.get("sha256") != sha_bytes(driver_raw)
    ):
        raise RuntimeError("STREAM_DRIVER_CONTRACT_SOURCE_BINDING_MISMATCH")
    helper_pin = contract.get("stream_helper", {})
    helper_raw = V4_STREAM_HELPER.read_bytes()
    if (
        Path(helper_pin.get("path", "")).resolve() != V4_STREAM_HELPER.resolve()
        or helper_pin.get("bytes") != len(helper_raw)
        or helper_pin.get("sha256") != V4_STREAM_HELPER_SHA256
        or sha_bytes(helper_raw) != V4_STREAM_HELPER_SHA256
    ):
        raise RuntimeError("STREAM_HELPER_CONTRACT_SOURCE_BINDING_MISMATCH")
    if contract.get("remote_read_only", {}).get("run_id") != contract.get("expected_run_id") or contract.get("remote_read_only", {}).get("command") != remote_readonly_command():
        raise RuntimeError("REMOTE_READ_ONLY_COMMAND_BINDING_MISMATCH")
    if str(args.output_root) != contract.get("local_output", {}).get("output_root"):
        raise RuntimeError("LOCAL_OUTPUT_ROOT_CONTRACT_MISMATCH")
    collector_module.validate_plan_and_terminal(contract)
    helper = load_module("v4_direct_stream_helper", V4_STREAM_HELPER, V4_STREAM_HELPER_SHA256)
    if not helper.hostpin_ok(args.host_key_sha256):
        raise RuntimeError("HOST_KEY_PIN_FORMAT_INVALID")
    password = os.environ.pop(PASSWORD_ENV, None)
    if not isinstance(password, str) or not password:
        raise RuntimeError("PRIVATE_PASSWORD_ENV_MISSING")
    receiver = None
    client = None
    stdout_file = None
    stderr_file = None
    try:
        receiver, paths, stdout_file, stderr_file = create_local_receiver(args.collector, args.contract, args.output_root)
        client, fingerprint = helper.connect(
            SimpleNamespace(host=args.host, port=args.port, username=args.username, host_key_sha256=args.host_key_sha256),
            password,
        )
        password = "\0" * len(password)
        result = stream_to_receiver(client, receiver, stdout_file, stderr_file)
        contract_sha_after, contract_bytes_after = sha_file(args.contract)
        if contract_sha_after != sha_bytes(contract_raw) or contract_bytes_after != len(contract_raw):
            raise RuntimeError("COLLECTION_CONTRACT_CHANGED_DURING_STREAM")
        result["connection"] = {
            "host": args.host,
            "port": args.port,
            "username": args.username,
            "host_key_sha256": fingerprint,
        }
        result["local_inputs"] = {
            "collector": {"path": str(args.collector.resolve()), "bytes": len(collector_raw), "sha256": sha_bytes(collector_raw)},
            "contract": {"path": str(args.contract.resolve()), "bytes": len(contract_raw), "sha256": sha_bytes(contract_raw)},
            "stream_driver": {"path": str(Path(__file__).resolve()), "bytes": len(driver_raw), "sha256": sha_bytes(driver_raw)},
            "stream_helper": {"path": str(V4_STREAM_HELPER), "bytes": V4_STREAM_HELPER.stat().st_size, "sha256": V4_STREAM_HELPER_SHA256},
        }
        result["receipt_sha256"] = sha_bytes(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        receipt_path = args.output_root / "direct_stream_receipt.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise FileExistsError("DIRECT_STREAM_RECEIPT_ALREADY_EXISTS")
        raw_receipt = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with receipt_path.open("xb") as output:
            output.write(raw_receipt)
            output.flush()
            os.fsync(output.fileno())
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stream_bytes": result["stream_bytes"],
                    "stream_sha256": result["stream_sha256"],
                    "restored_file_count": result["restored_file_count"],
                    "direct_stream_receipt": str(receipt_path),
                    "receipt_sha256": result["receipt_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    finally:
        password = ""
        if client is not None:
            client.close()
        if receiver is not None and receiver.poll() is None:
            try:
                if receiver.stdin is not None and not receiver.stdin.closed:
                    receiver.stdin.close()
                receiver.terminate()
                receiver.wait(timeout=10)
            except BaseException:
                try:
                    receiver.kill()
                except OSError:
                    pass
        if stdout_file is not None and not stdout_file.closed:
            stdout_file.close()
        if stderr_file is not None and not stderr_file.closed:
            stderr_file.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
