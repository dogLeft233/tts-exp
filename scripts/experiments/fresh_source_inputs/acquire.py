"""Bounded, provenance-preserving acquisition for the fresh-source P stage.

The acquisition worker deliberately has a small surface: it inventories the
already mounted pretrain directory and the two recorded pretrain archives,
extracts only the first eight clips of at most 24 previously unused source
groups, and writes a frozen pool for the visual worker.  It never walks train,
val, or test media and it never changes an existing run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .protocol import (
    GROUP_ID_RE,
    REPO_ROOT,
    file_sha256,
    scan_history,
    write_json,
)

DATA_ROOT_NAME = "data/dataset_samples/lrs3/pretrain"
ARCHIVE_ROOT_NAME = "data/lrs3/ellipsis-lrs3-raw/ainncy"
MAX_GROUPS = 24
CLIPS_PER_GROUP = 8
MAX_ACQUISITION_BYTES = 10 * (1 << 30)
MAX_WALL_SECONDS = 45 * 60
MIN_FREE_BYTES = 5 * (1 << 30)
SCHEMA_VERSION = 1


class AcquisitionError(RuntimeError):
    """An acquisition contract error that should be recorded as blocked."""


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_member(member_name: str) -> PurePosixPath:
    """Return a safe archive member path or raise.

    Tar files are untrusted inputs.  Absolute paths, parent traversal, links,
    and non-pretrain members are rejected before any bytes are written.
    """

    path = PurePosixPath(member_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AcquisitionError(f"unsafe archive member: {member_name}")
    if any(part.startswith("/") for part in path.parts):
        raise AcquisitionError(f"unsafe archive member: {member_name}")
    return path


def _member_group(path: PurePosixPath) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if GROUP_ID_RE.fullmatch(part) and parts[index + 1].lower().endswith((".mp4", ".txt")):
            return part
    return None


def _clip_id(path: PurePosixPath) -> str | None:
    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".txt"}:
        return None
    stem = path.stem
    return stem if stem.isdigit() else None


def _archive_inventory(archive: Path) -> tuple[dict[str, dict[str, tarfile.TarInfo]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, tarfile.TarInfo]] = {}
    ledger: list[dict[str, Any]] = []
    try:
        with tarfile.open(archive, "r") as handle:
            for member in handle.getmembers():
                try:
                    safe = _safe_member(member.name)
                except AcquisitionError as exc:
                    ledger.append({"source": str(archive.resolve()), "member": member.name, "status": "REJECT", "reason": str(exc)})
                    continue
                if member.issym() or member.islnk():
                    ledger.append({"source": str(archive.resolve()), "member": member.name, "status": "REJECT", "reason": "archive link is forbidden"})
                    continue
                if not member.isfile():
                    continue
                group = _member_group(safe)
                clip_id = _clip_id(safe)
                if group is None or clip_id is None:
                    continue
                suffix = safe.suffix.lower()
                by_group.setdefault(group, {})[f"{clip_id}{suffix}"] = member
    except (OSError, tarfile.TarError) as exc:
        ledger.append({"source": str(archive.resolve()), "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
    return by_group, ledger


def _source_descriptor(path: Path, *, kind: str, split: str = "pretrain") -> dict[str, Any]:
    row: dict[str, Any] = {"kind": kind, "path": str(path.resolve()), "split": split, "accessible": path.exists()}
    if path.is_file():
        row.update({"size": path.stat().st_size, "sha256": file_sha256(path)})
    elif path.is_dir():
        stat = path.stat()
        row.update({"mtime_ns": stat.st_mtime_ns})
    return row


def _write_ledger(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _local_inventory(data_root: Path, historical: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    ledger: list[dict[str, Any]] = []
    if not data_root.is_dir():
        return groups, [{"source": str(data_root), "status": "ERROR", "reason": "local pretrain root is missing"}]
    for group_dir in sorted(data_root.iterdir(), key=lambda item: item.name):
        group = group_dir.name
        if not group_dir.is_dir() or not GROUP_ID_RE.fullmatch(group):
            continue
        if group in historical:
            ledger.append({"source_group": group, "status": "HISTORICAL_EXCLUDED", "source": str(group_dir.resolve())})
            continue
        clips: list[dict[str, Any]] = []
        mp4s = sorted(group_dir.glob("*.mp4"), key=lambda item: item.name)[:CLIPS_PER_GROUP]
        for mp4 in mp4s:
            clip_id = mp4.stem
            txt = mp4.with_suffix(".txt")
            row = {
                "source_group": group,
                "clip_id": clip_id,
                "source_kind": "local",
                "source": str(data_root.resolve()),
                "member": str(mp4.resolve().relative_to(data_root.resolve())),
                "mp4_path": str(mp4.resolve()),
                "txt_path": str(txt.resolve()),
                "mp4_sha256": file_sha256(mp4),
                "txt_sha256": file_sha256(txt) if txt.is_file() else None,
                "status": "READY" if txt.is_file() else "MISSING_TRANSCRIPT",
            }
            clips.append(row)
            ledger.append({"source_group": group, "clip_id": clip_id, "status": row["status"], "source": row["source"], "member": row["member"]})
        if clips:
            groups[group] = clips
        else:
            ledger.append({"source_group": group, "status": "NO_MP4", "source": str(group_dir.resolve())})
    return groups, ledger


def _archive_group_candidates(
    archive: Path,
    by_group: dict[str, dict[str, tarfile.TarInfo]],
    historical: set[str],
    local_groups: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for group in sorted(by_group):
        if group in historical or group in local_groups:
            continue
        members = by_group[group]
        clip_ids = sorted({name[:-4] for name in members if name.endswith(".mp4")})[:CLIPS_PER_GROUP]
        rows = []
        for clip_id in clip_ids:
            mp4_member = members.get(f"{clip_id}.mp4")
            txt_member = members.get(f"{clip_id}.txt")
            if mp4_member is None:
                continue
            rows.append({
                "source_group": group,
                "clip_id": clip_id,
                "source_kind": "archive",
                "source": str(archive.resolve()),
                "member": mp4_member.name,
                "txt_member": txt_member.name if txt_member else None,
                "mp4_size": int(mp4_member.size),
                "txt_size": int(txt_member.size) if txt_member else 0,
                "status": "READY" if txt_member is not None else "MISSING_TRANSCRIPT",
            })
        if rows:
            result[group] = rows
    return result


def _extract_archive_group(
    archive: Path,
    rows: list[dict[str, Any]],
    data_root: Path,
    *,
    started: float,
    bytes_used: int,
    max_bytes: int = MAX_ACQUISITION_BYTES,
    max_seconds: float = MAX_WALL_SECONDS,
) -> tuple[list[dict[str, Any]], int]:
    target_group = data_root / str(rows[0]["source_group"])
    if not _path_under(target_group, data_root):
        raise AcquisitionError(f"archive extraction escaped data root: {target_group}")
    row_by_member: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(archive, "r") as handle:
            members = {member.name: member for member in handle.getmembers()}
            for row in rows:
                for key, suffix in (("member", ".mp4"), ("txt_member", ".txt")):
                    member_name = row.get(key)
                    if not member_name:
                        continue
                    member = members.get(str(member_name))
                    if member is None or not member.isfile() or member.issym() or member.islnk():
                        raise AcquisitionError(f"archive member is unavailable or unsafe: {member_name}")
                    safe = _safe_member(member.name)
                    if _member_group(safe) != row["source_group"] or safe.suffix.lower() != suffix:
                        raise AcquisitionError(f"archive member/group mismatch: {member_name}")
                    target = target_group / f"{row['clip_id']}{suffix}"
                    if not _path_under(target, data_root):
                        raise AcquisitionError(f"archive target escaped data root: {target}")
                    if time.monotonic() - started > max_seconds:
                        raise AcquisitionError("acquisition wall-clock budget exceeded")
                    expected_size = int(member.size)
                    if bytes_used + expected_size > max_bytes:
                        raise AcquisitionError("acquisition byte budget exceeded")
                    if target.exists():
                        if not target.is_file() or target.stat().st_size != expected_size:
                            raise AcquisitionError(f"existing extraction conflicts with archive member: {target}")
                        # Size alone is not enough to claim identity.  Hash the
                        # existing file and the member before reusing it.
                        extracted = handle.extractfile(member)
                        if extracted is None:
                            raise AcquisitionError(f"cannot read archive member: {member_name}")
                        digest = hashlib.sha256()
                        while True:
                            chunk = extracted.read(1 << 20)
                            if not chunk:
                                break
                            digest.update(chunk)
                        if digest.hexdigest() != file_sha256(target):
                            raise AcquisitionError(f"existing extraction SHA conflicts with archive member: {target}")
                    else:
                        target_group.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(f".{target.name}.{time.monotonic_ns()}.tmp")
                        extracted = handle.extractfile(member)
                        if extracted is None:
                            raise AcquisitionError(f"cannot read archive member: {member_name}")
                        try:
                            with temporary.open("wb") as output:
                                copied = 0
                                while True:
                                    chunk = extracted.read(1 << 20)
                                    if not chunk:
                                        break
                                    copied += len(chunk)
                                    if bytes_used + copied > max_bytes:
                                        raise AcquisitionError("acquisition byte budget exceeded")
                                    output.write(chunk)
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
                        bytes_used += copied
                    row_by_member[member_name] = {"path": str(target.resolve()), "sha256": file_sha256(target), "size": target.stat().st_size}
    except (OSError, tarfile.TarError) as exc:
        raise AcquisitionError(f"archive extraction failed: {type(exc).__name__}: {exc}") from exc
    materialized = []
    for row in rows:
        mp4_info = row_by_member.get(str(row["member"]))
        txt_info = row_by_member.get(str(row["txt_member"])) if row.get("txt_member") else None
        materialized.append({
            "source_group": row["source_group"],
            "clip_id": row["clip_id"],
            "source_kind": "archive",
            "source": row["source"],
            "member": row["member"],
            "txt_member": row.get("txt_member"),
            "mp4_path": mp4_info["path"] if mp4_info else None,
            "txt_path": txt_info["path"] if txt_info else None,
            "mp4_sha256": mp4_info["sha256"] if mp4_info else None,
            "txt_sha256": txt_info["sha256"] if txt_info else None,
            "status": "READY" if mp4_info and txt_info else "MISSING_TRANSCRIPT",
        })
    return materialized, bytes_used


def acquire_pool(
    run_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    max_groups: int = MAX_GROUPS,
    clips_per_group: int = CLIPS_PER_GROUP,
    max_bytes: int = MAX_ACQUISITION_BYTES,
    max_seconds: float = MAX_WALL_SECONDS,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, Any]:
    """Create the bounded current-run source pool and its audit artifacts."""

    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    if max_groups <= 0 or max_groups > MAX_GROUPS:
        raise ValueError(f"max_groups must be in 1..{MAX_GROUPS}")
    if clips_per_group != CLIPS_PER_GROUP:
        raise ValueError(f"clips_per_group is frozen at {CLIPS_PER_GROUP}")
    acquisition = run_root / "acquisition"
    acquisition.mkdir(parents=True, exist_ok=True)
    data_root = repo_root / DATA_ROOT_NAME
    archive_root = repo_root / ARCHIVE_ROOT_NAME
    started = time.monotonic()
    history = scan_history(repo_root, ignore_run_root=run_root)
    write_json(acquisition / "history_snapshot.json", history)
    source_paths = [data_root, *sorted(archive_root.glob("pretrain*.tar"))]
    sources = [_source_descriptor(path, kind="local" if path == data_root else "archive") for path in source_paths]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "fresh_source_inputs_unblock",
        "run_root": str(run_root),
        "split": "pretrain",
        "history_snapshot": str((acquisition / "history_snapshot.json").resolve()),
        "history_snapshot_sha256": file_sha256(acquisition / "history_snapshot.json"),
        "history_file_manifest_sha256": history.get("scan_file_manifest_sha256"),
        "sources": sources,
        "selection": {"max_groups": max_groups, "clips_per_group": clips_per_group, "group_order": "source_group_lexicographic", "clip_order": "filename_lexicographic_first_8"},
        "visual": {
            "method": "mediapipe_face_mesh_v1",
            "model_path": str((repo_root / "checkpoints/mediapipe/face_landmarker.task").resolve()),
            "options": {"running_mode": "IMAGE", "num_faces": 1, "frame_count": 140, "fps": 25.0, "zero_pts": True},
        },
        "budget": {"max_acquisition_bytes": max_bytes, "max_wall_seconds": max_seconds, "min_free_bytes": min_free_bytes},
        "outputs": {"source_manifest": str((acquisition / "source_manifest.json").resolve()), "pool": str((acquisition / "pool.json").resolve()), "ledger": str((acquisition / "ledger.jsonl").resolve())},
        "status": "PLANNED",
    }
    write_json(acquisition / "plan.json", plan)

    ledger: list[dict[str, Any]] = []
    local_groups, local_ledger = _local_inventory(data_root, set(history.get("groups", [])))
    ledger.extend(local_ledger)
    archive_groups: dict[str, list[dict[str, Any]]] = {}
    for archive in sorted(archive_root.glob("pretrain*.tar")):
        inventory, archive_ledger = _archive_inventory(archive)
        ledger.extend(archive_ledger)
        archive_groups.update(_archive_group_candidates(archive, inventory, set(history.get("groups", [])), set(local_groups)))

    selected_groups = sorted(set(local_groups) | set(archive_groups))[:max_groups]
    materialized: dict[str, list[dict[str, Any]]] = {}
    bytes_used = 0
    blocked_reason: str | None = None
    try:
        if time.monotonic() - started > max_seconds:
            raise AcquisitionError("acquisition wall-clock budget exceeded")
        if selected_groups and shutil.disk_usage(data_root if data_root.exists() else repo_root).free < min_free_bytes:
            raise AcquisitionError("free disk space is below the frozen 5 GiB reserve")
        for group in selected_groups:
            if group in local_groups:
                materialized[group] = local_groups[group][:clips_per_group]
                continue
            rows = archive_groups[group]
            source = Path(str(rows[0]["source"]))
            materialized[group], bytes_used = _extract_archive_group(source, rows[:clips_per_group], data_root, started=started, bytes_used=bytes_used, max_bytes=max_bytes, max_seconds=max_seconds)
    except AcquisitionError as exc:
        blocked_reason = str(exc)
        ledger.append({"status": "BLOCKED_ACQUISITION_BUDGET" if "budget" in str(exc) or "space" in str(exc) else "ERROR", "reason": str(exc)})

    manifest_rows: list[dict[str, Any]] = []
    groups_payload: list[dict[str, Any]] = []
    for position, group in enumerate(selected_groups):
        clips = materialized.get(group, [])[:clips_per_group]
        clips = sorted(clips, key=lambda row: str(row["clip_id"]))
        for clip_position, clip in enumerate(clips):
            clip_row = dict(clip, group_order=position, clip_order=clip_position)
            manifest_rows.append(clip_row)
        groups_payload.append({"source_group": group, "group_order": position, "clips": clips, "clip_ids": [str(row["clip_id"]) for row in clips]})
    if len(selected_groups) < 14 and blocked_reason is None:
        blocked_reason = "no recorded accessible pretrain source can supply fourteen fresh groups"
        ledger.append({"status": "BLOCKED_SOURCE_ACCESS", "reason": blocked_reason, "fresh_group_count": len(selected_groups), "required_group_count": 14})
    status = "READY" if len(selected_groups) >= 14 and blocked_reason is None else ("BLOCKED_ACQUISITION_BUDGET" if blocked_reason and ("budget" in blocked_reason or "space" in blocked_reason) else "BLOCKED_SOURCE_ACCESS")
    _write_ledger(acquisition / "ledger.jsonl", ledger)
    write_json(acquisition / "source_manifest.json", {"schema_version": SCHEMA_VERSION, "status": status, "source_count": len(sources), "rows": manifest_rows, "ledger_sha256": file_sha256(acquisition / "ledger.jsonl")})
    pool = write_json(acquisition / "pool.json", {"schema_version": SCHEMA_VERSION, "status": status, "source_plan_sha256": file_sha256(acquisition / "plan.json"), "source_manifest_sha256": file_sha256(acquisition / "source_manifest.json"), "max_groups": max_groups, "clips_per_group": clips_per_group, "groups": groups_payload, "group_order": selected_groups, "bytes_acquired": bytes_used, "blocked_reason": blocked_reason})
    plan.update({"status": status, "selected_group_count": len(selected_groups), "bytes_acquired": bytes_used, "ledger_sha256": file_sha256(acquisition / "ledger.jsonl"), "source_manifest_sha256": file_sha256(acquisition / "source_manifest.json"), "elapsed_seconds": time.monotonic() - started})
    write_json(acquisition / "plan.json", plan)
    # Rebind the pool to the final plan revision.  This is intentionally the
    # last write: a pool from an earlier plan must never be accepted.
    pool["source_plan_sha256"] = file_sha256(acquisition / "plan.json")
    write_json(acquisition / "pool.json", pool)
    print(f"acquisition status={status} groups={len(selected_groups)} bytes={bytes_used}")
    return pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-groups", type=int, default=MAX_GROUPS)
    parser.add_argument("--max-bytes", type=int, default=MAX_ACQUISITION_BYTES)
    parser.add_argument("--max-seconds", type=float, default=MAX_WALL_SECONDS)
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)
    args = parser.parse_args(argv)
    try:
        result = acquire_pool(args.run_root, repo_root=args.repo_root, max_groups=args.max_groups, max_bytes=args.max_bytes, max_seconds=args.max_seconds, min_free_bytes=args.min_free_bytes)
    except (AcquisitionError, OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        print(f"acquisition status=BLOCKED_SOURCE_ACCESS error={type(exc).__name__}: {exc}")
        return 2
    return 0 if result.get("status") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
