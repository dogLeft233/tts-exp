#!/usr/bin/env python3
"""Stream a sealed WORLD NPZ member set over an already-authenticated SSH client.

Authentication is intentionally external. The caller passes a connected
Paramiko SSHClient and SFTPClient; this module accepts no password, token, or
credential path and never writes credentials to disk or argv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class StreamArchiveError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_member_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    embedded = value.get("manifest_sha256")
    computed = sha_bytes(canonical({key: item for key, item in value.items() if key != "manifest_sha256"}))
    if embedded != computed:
        raise StreamArchiveError("member manifest fingerprint mismatch")
    if value.get("status") != "SEALED_READ_ONLY_INVENTORY":
        raise StreamArchiveError("member manifest is not sealed read-only inventory")
    by_name: dict[str, dict[str, Any]] = {}
    for row in value.get("members", []):
        name = row.get("archive_path")
        if not isinstance(name, str) or not name or name in by_name:
            raise StreamArchiveError("empty or duplicate archive member path")
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise StreamArchiveError(f"unsafe archive path: {name}")
        if not name.endswith(".npz") or len(str(row.get("sha256", ""))) != 64 or int(row.get("size_bytes", -1)) < 0:
            raise StreamArchiveError(f"invalid archive member metadata: {name}")
        by_name[name] = row
    coverage = value.get("coverage", {})
    if len(by_name) != coverage.get("npz_member_count") or len(by_name) != 1920:
        raise StreamArchiveError("member inventory does not contain exactly 1,920 NPZ files")
    if sum(int(row["size_bytes"]) for row in by_name.values()) != int(coverage.get("total_member_bytes", -1)):
        raise StreamArchiveError("member inventory total byte count mismatch")
    return value, by_name


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temp.rename(path)


def _verify_tar_archive(archive_path: Path, expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    total_uncompressed = 0
    with archive_path.open("rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as archive:
            for member in archive:
                name = member.name
                if name not in expected or name in seen:
                    raise StreamArchiveError(f"archive contains unknown or repeated member: {name}")
                if not member.isfile() or member.issym() or member.islnk():
                    raise StreamArchiveError(f"archive member is not a regular file: {name}")
                row = expected[name]
                if member.size != int(row["size_bytes"]):
                    raise StreamArchiveError(f"archive member size mismatch: {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise StreamArchiveError(f"archive member has no byte stream: {name}")
                digest = hashlib.sha256()
                read_bytes = 0
                with source:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        digest.update(block)
                        read_bytes += len(block)
                if read_bytes != member.size or digest.hexdigest() != row["sha256"]:
                    raise StreamArchiveError(f"archive member content SHA mismatch: {name}")
                seen.add(name)
                total_uncompressed += read_bytes
    missing = set(expected) - seen
    if missing:
        raise StreamArchiveError(f"archive is missing {len(missing)} manifest members")
    return {"member_count": len(seen), "uncompressed_bytes": total_uncompressed, "all_member_hashes_pass": True}


def _copy_verified(source: Path, destination: Path, expected_sha: str, progress_path: Path | None = None) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite archive copy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial copy: {partial}")
    digest = hashlib.sha256()
    copied = 0
    last_report = time.monotonic()
    with source.open("rb") as src:
        fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
                digest.update(block)
                copied += len(block)
                if progress_path is not None and time.monotonic() - last_report >= 15:
                    write_json_atomic_progress(progress_path, {"phase": "copy_to_secondary", "bytes_copied": copied, "archive_bytes": source.stat().st_size})
                    last_report = time.monotonic()
            dst.flush()
            os.fsync(dst.fileno())
    observed = digest.hexdigest()
    if observed != expected_sha or copied != source.stat().st_size:
        raise StreamArchiveError("secondary copy SHA or byte count mismatch; partial retained")
    partial.rename(destination)
    readback_sha = sha_file(destination)
    readback_bytes = destination.stat().st_size
    if readback_sha != expected_sha or readback_bytes != copied:
        raise StreamArchiveError("secondary copy readback mismatch; copied artifact retained for inspection")
    return {"path": str(destination), "bytes": copied, "sha256": observed, "readback_sha256": readback_sha, "verified": True}


def write_json_atomic_progress(path: Path, value: dict[str, Any]) -> None:
    """Replace only this task's mutable progress sidecar atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temp, path)


def stream_archive_with_authenticated_client(
    ssh_client: Any,
    sftp_client: Any,
    member_manifest_path: Path,
    remote_member_list_path: str,
    local_archive_path: Path,
    receipt_path: Path,
    progress_path: Path | None = None,
    secondary_copy_path: Path | None = None,
) -> dict[str, Any]:
    """Stream an exact tar.gz over SSH and verify every archived NPZ.

    `ssh_client` must already be authenticated by the caller. The member list
    uploaded via SFTP contains only relative paths from the sealed manifest.
    No target artifact is deleted or changed by this function.
    """
    manifest_path = member_manifest_path.resolve(strict=True)
    manifest, expected = verify_member_manifest(manifest_path)
    archive_path = local_archive_path.resolve()
    receipt = receipt_path.resolve()
    if archive_path.exists() or archive_path.with_name(archive_path.name + ".partial").exists() or receipt.exists():
        raise FileExistsError("archive, partial, or receipt path already exists")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)

    list_bytes = b"".join(name.encode("utf-8") + b"\0" for name in sorted(expected))
    try:
        sftp_client.stat(remote_member_list_path)
    except OSError:
        pass
    else:
        raise FileExistsError(f"refusing to overwrite remote member list: {remote_member_list_path}")
    with sftp_client.open(remote_member_list_path, "wb") as remote_list:
        remote_list.write(list_bytes)
        remote_list.flush()
    remote_size = int(sftp_client.stat(remote_member_list_path).st_size)
    if remote_size != len(list_bytes):
        raise StreamArchiveError("uploaded remote NUL member list has the wrong byte count")

    root = shlex.quote(manifest["artifact_root"])
    member_list = shlex.quote(remote_member_list_path)
    inner_command = f"set -euo pipefail; tar --create --file=- --null --directory={root} --files-from={member_list} | gzip --no-name --stdout"
    command = f"bash -lc {shlex.quote(inner_command)}"
    stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=False)
    channel = stdout.channel
    stderr_parts: list[bytes] = []

    def drain_stderr() -> None:
        while True:
            if channel.recv_stderr_ready():
                stderr_parts.append(channel.recv_stderr(1 << 16))
            elif channel.exit_status_ready():
                while channel.recv_stderr_ready():
                    stderr_parts.append(channel.recv_stderr(1 << 16))
                return
            else:
                time.sleep(0.02)

    stderr_thread = threading.Thread(target=drain_stderr, name="world-npz-archive-stderr", daemon=True)
    stderr_thread.start()
    partial_path = archive_path.with_name(archive_path.name + ".partial")
    archive_digest = hashlib.sha256()
    bytes_received = 0
    last_report = time.monotonic()
    try:
        fd = os.open(partial_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as local_stream:
            while True:
                block = channel.recv(1 << 20)
                if not block:
                    if channel.exit_status_ready():
                        break
                    continue
                local_stream.write(block)
                archive_digest.update(block)
                bytes_received += len(block)
                now = time.monotonic()
                if progress_path is not None and now - last_report >= 15:
                    write_json_atomic_progress(progress_path, {"phase": "streaming_remote_tar", "bytes_received": bytes_received, "started_at_utc": manifest.get("created_utc")})
                    last_report = now
            local_stream.flush()
            os.fsync(local_stream.fileno())
        exit_code = channel.recv_exit_status()
    except Exception:
        channel.close()
        raise
    finally:
        stderr_thread.join(timeout=5)
        stdin.close()
        stdout.close()
        stderr.close()
    stderr_text = b"".join(stderr_parts).decode("utf-8", errors="replace")
    if exit_code != 0:
        raise StreamArchiveError(f"remote tar/gzip exit {exit_code}; partial archive retained; stderr={stderr_text[:1000]}")
    if bytes_received == 0:
        raise StreamArchiveError("remote tar stream was empty")

    archive_sha = archive_digest.hexdigest()
    archive_size = partial_path.stat().st_size
    if archive_size != bytes_received or sha_file(partial_path) != archive_sha:
        raise StreamArchiveError("streamed archive byte/hash reconciliation failed")
    tar_verification = _verify_tar_archive(partial_path, expected)
    partial_path.rename(archive_path)

    secondary_copy = None
    if secondary_copy_path is not None:
        secondary_copy = _copy_verified(archive_path, secondary_copy_path.resolve(), archive_sha, progress_path)

    payload = {
        "schema_version": 1,
        "classification": "verified streamed archive of retained WORLD physical-EVAL NPZ source artifacts; source files were not deleted",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "member_manifest": str(manifest_path),
        "member_manifest_sha256": sha_file(manifest_path),
        "physical_manifest_sha256": manifest["source_seals"]["physical_manifest"]["sha256"],
        "physical_qa_report_sha256": manifest["qa_report"]["sha256"],
        "archive_root": manifest["artifact_root"],
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "remote_tar_exit_code": exit_code,
        "remote_member_list_path": remote_member_list_path,
        "remote_member_list_bytes": remote_size,
        "member_verification": tar_verification,
        "secondary_copy": secondary_copy,
        "stderr_nonempty": bool(stderr_text.strip()),
        "stderr_sha256": sha_bytes(stderr_text.encode("utf-8")),
    }
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite archive receipt: {receipt}")
    write_json_atomic(receipt, payload)
    if progress_path is not None:
        write_json_atomic_progress(progress_path, {"phase": "complete", "archive_bytes": archive_size, "archive_sha256": archive_sha, "verified_members": tar_verification["member_count"]})
    return payload
