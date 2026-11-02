#!/usr/bin/env python3
"""Install a sealed attempt03 source closure without overwriting unknown files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def safe_relative(text: Any) -> PurePosixPath:
    if not isinstance(text, str) or "\\" in text:
        raise ValueError("bundle and destination names must be POSIX relative paths")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path in source manifest: {text!r}")
    return path


def no_symlink_ancestors(root: Path, relative: PurePosixPath, *, include_leaf: bool) -> Path:
    current = root
    parts = relative.parts if include_leaf else relative.parts[:-1]
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source staging path traverses a symlink: {current}")
        if current.exists() and not current.is_dir() and current != root / relative:
            raise ValueError(f"source staging path parent is not a directory: {current}")
    return root.joinpath(*relative.parts)


def install(manifest_path: Path, stage_root: Path, repo_root: Path,
            expected_manifest_sha256: str) -> dict[str, Any]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("sealed source manifest is missing or unsafe")
    manifest_raw = manifest_path.read_bytes()
    observed_manifest_sha = sha_bytes(manifest_raw)
    if observed_manifest_sha != expected_manifest_sha256:
        raise ValueError("source manifest raw SHA differs from dispatcher request pin")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if (manifest.get("schema") != "cross_tfg_backend_subset_source_bundle_v1"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("target_repo_root") != str(repo_root.resolve(strict=True))):
        raise ValueError("source manifest self-hash or target root identity differs")
    if stage_root.is_symlink() or not stage_root.is_dir():
        raise ValueError("sealed source bundle root must be a real directory")
    repo_root = repo_root.resolve(strict=True)
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ValueError("target repository root must be a real directory")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sealed source manifest has no file list")

    verified: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid source manifest row")
        staged_rel = safe_relative(entry.get("staged_rel"))
        target_rel = safe_relative(entry.get("target_rel"))
        if target_rel.as_posix() in seen_targets:
            raise ValueError(f"duplicate repository target in source manifest: {target_rel}")
        seen_targets.add(target_rel.as_posix())
        source = no_symlink_ancestors(stage_root, staged_rel, include_leaf=True)
        target = no_symlink_ancestors(repo_root, target_rel, include_leaf=True)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"staged source file is missing or unsafe: {source}")
        expected_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")
        if (type(expected_bytes) is not int or expected_bytes < 0
                or not isinstance(expected_sha, str) or len(expected_sha) != 64
                or source.stat().st_size != expected_bytes or sha_file(source) != expected_sha):
            raise ValueError(f"staged source file does not match its byte/SHA pin: {source}")
        if target.is_symlink():
            raise ValueError(f"repository target is a symlink: {target}")
        if target.exists():
            if not target.is_file() or target.stat().st_size != expected_bytes or sha_file(target) != expected_sha:
                raise FileExistsError(f"repository target exists with different bytes; refusing overwrite: {target}")
            state = "already_identical"
        else:
            state = "install_required"
        verified.append({"staged_rel": staged_rel.as_posix(), "target_rel": target_rel.as_posix(),
                         "bytes": expected_bytes, "sha256": expected_sha, "state": state})

    for row in verified:
        if row["state"] == "already_identical":
            continue
        source = stage_root.joinpath(*safe_relative(row["staged_rel"]).parts)
        target = repo_root.joinpath(*safe_relative(row["target_rel"]).parts)
        no_symlink_ancestors(repo_root, safe_relative(row["target_rel"]), include_leaf=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if (target.is_file() and not target.is_symlink()
                    and target.stat().st_size == row["bytes"] and sha_file(target) == row["sha256"]):
                row["state"] = "already_identical_after_race"
                continue
            raise FileExistsError(f"repository target changed during staging; refusing overwrite: {target}")
        temp = target.with_name(f".{target.name}.stage-{os.getpid()}")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as out_stream, source.open("rb") as in_stream:
                shutil.copyfileobj(in_stream, out_stream, 1 << 20)
                out_stream.flush()
                os.fsync(out_stream.fileno())
            if temp.stat().st_size != row["bytes"] or sha_file(temp) != row["sha256"]:
                raise IOError(f"staged copy does not match its sealed source: {temp}")
            os.link(temp, target)
            row["state"] = "installed"
        finally:
            temp.unlink(missing_ok=True)

    return {
        "schema": "cross_tfg_backend_subset_source_stage_receipt_v1",
        "status": "PASS_STAGED_OR_ALREADY_IDENTICAL",
        "manifest": {"path": str(manifest_path.resolve(strict=True)),
                     "bytes": len(manifest_raw), "sha256": observed_manifest_sha,
                     "manifest_sha256": manifest["manifest_sha256"]},
        "target_repo_root": str(repo_root),
        "file_count": len(verified),
        "installed_count": sum(row["state"] == "installed" for row in verified),
        "already_identical_count": sum(row["state"].startswith("already_identical") for row in verified),
        "files": verified,
        "model_import_or_forward_performed": False,
        "media_generation_or_scoring_performed": False,
        "api_calls_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        result = install(args.manifest, args.stage_root, args.repo_root,
                         args.expected_manifest_sha256)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"schema": "cross_tfg_backend_subset_source_stage_receipt_v1",
                          "status": "FAILED_SOURCE_STAGE", "error_type": type(exc).__name__,
                          "error": str(exc), "model_import_or_forward_performed": False,
                          "media_generation_or_scoring_performed": False, "api_calls_performed": False},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
