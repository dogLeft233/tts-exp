#!/usr/bin/env python3
"""Acquire only the ten frozen AISHELL development speaker archives.

This utility does not read audio values, select utterances, or touch the
confirmation roster. Each downloaded archive is checked against the pinned
Hugging Face dataset-tree LFS SHA-256 and byte count before it is published.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json"
DEFAULT_REVISION = ROOT / "data/native_mechanism_holdout_20260926/provenance/hf_dataset_revision.json"
DEFAULT_TREE = ROOT / "data/native_mechanism_holdout_20260926/provenance/hf_archive_tree.json"
DATASET_ID = "AISHELL/AISHELL-1"
URL_TEMPLATE = "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{revision}/data_aishell/wav/{speaker}.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_contract(protocol_path: Path, revision_path: Path, tree_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    revision_info = json.loads(revision_path.read_text(encoding="utf-8"))
    tree_rows = json.loads(tree_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ROSTER_FROZEN_ROWS_PENDING":
        raise ValueError("speaker roster is not in the expected frozen-before-effects state")
    dev_ids = protocol.get("dev_speaker_ids")
    confirm_ids = protocol.get("confirm_speaker_ids")
    if not isinstance(dev_ids, list) or len(dev_ids) != 10 or len(set(dev_ids)) != 10:
        raise ValueError("the frozen development roster must contain ten unique speakers")
    if not isinstance(confirm_ids, list) or set(dev_ids) & set(confirm_ids):
        raise ValueError("development and confirmation speakers must be disjoint")
    if any(speaker in protocol.get("historically_exposed_speaker_ids", []) for speaker in confirm_ids):
        raise ValueError("confirmation roster intersects the historical exposure list")
    revision = revision_info.get("sha")
    if revision != protocol.get("dataset_revision"):
        raise ValueError("dataset revision does not match the frozen speaker protocol")
    if revision_info.get("id") != DATASET_ID:
        raise ValueError("unexpected Hugging Face dataset identity")
    tree = {row.get("path"): row for row in tree_rows if isinstance(row, dict)}
    return protocol, revision, tree


def acquire_one(
    speaker: str,
    revision: str,
    tree: dict[str, Any],
    destination: Path,
    attempts: int,
) -> dict[str, Any]:
    relative = f"data_aishell/wav/{speaker}.tar.gz"
    entry = tree.get(relative)
    if not isinstance(entry, dict) or not isinstance(entry.get("lfs"), dict):
        raise ValueError(f"publisher tree has no LFS seal for {speaker}")
    expected_bytes = int(entry.get("size", -1))
    expected_sha = str(entry["lfs"].get("oid", "")).lower()
    if expected_bytes <= 0 or len(expected_sha) != 64:
        raise ValueError(f"publisher tree has invalid size/SHA for {speaker}")
    output = destination / f"{speaker}.tar.gz"
    partial = output.with_suffix(output.suffix + ".partial")
    source_url = URL_TEMPLATE.format(revision=revision, speaker=speaker)
    attempt_log: list[dict[str, Any]] = []
    if output.exists():
        actual_bytes = output.stat().st_size
        actual_sha = sha256_file(output)
        if (actual_bytes, actual_sha) != (expected_bytes, expected_sha):
            raise FileExistsError(f"existing archive fails pinned publisher seal: {output}")
        status = "REUSED_VERIFIED"
    else:
        if partial.exists():
            raise FileExistsError(f"stale partial exists; preserve and inspect before retry: {partial}")
        status = "DOWNLOADED_VERIFIED"
        for attempt in range(1, attempts + 1):
            digest = hashlib.sha256()
            count = 0
            try:
                request = urllib.request.Request(
                    source_url,
                    headers={"User-Agent": "cross-tfg-audio-head-acquisition/1"},
                )
                with urllib.request.urlopen(request, timeout=180) as response, partial.open("xb") as handle:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        count += len(block)
                        digest.update(block)
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                actual_sha = digest.hexdigest()
                if (count, actual_sha) != (expected_bytes, expected_sha):
                    raise ValueError(
                        f"download seal mismatch for {speaker}: {count}/{actual_sha}"
                    )
                partial.replace(output)
                attempt_log.append({"attempt": attempt, "status": "PASS"})
                break
            except (OSError, urllib.error.URLError, ValueError) as exc:
                attempt_log.append({"attempt": attempt, "status": "FAILED", "error_type": type(exc).__name__})
                if partial.exists():
                    partial.unlink()
                if attempt >= attempts:
                    raise RuntimeError(f"archive acquisition failed for {speaker}; see error type only") from None
                time.sleep(min(2 * attempt, 10))
    return {
        "speaker_id": speaker,
        "split": "dev",
        "source_dataset": DATASET_ID,
        "revision": revision,
        "repository_path": relative,
        "publisher_url": source_url,
        "path": str(output.resolve()),
        "bytes": int(output.stat().st_size),
        "sha256": sha256_file(output),
        "publisher_lfs_sha256": expected_sha,
        "publisher_size_sha256_verified": True,
        "status": status,
        "attempts": attempt_log,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--revision-file", type=Path, default=DEFAULT_REVISION)
    parser.add_argument("--archive-tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1 or args.workers > 6 or args.attempts < 1 or args.attempts > 8:
            raise ValueError("workers/attempts are outside the bounded acquisition settings")
        protocol, revision, tree = load_contract(args.protocol, args.revision_file, args.archive_tree)
        args.destination.mkdir(parents=True, exist_ok=True)
        manifest_path = args.destination / "development_pack_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("manifest_sha256") != canonical_sha(old):
                raise ValueError("existing acquisition manifest has an invalid canonical self-hash")
            if old.get("dev_speaker_ids") == protocol["dev_speaker_ids"] and old.get("revision") == revision:
                raise FileExistsError("a verified development acquisition manifest already exists")
            raise FileExistsError("refusing to replace an existing acquisition manifest")
        protocol_sha = sha256_file(args.protocol)
        revision_sha = sha256_file(args.revision_file)
        tree_sha = sha256_file(args.archive_tree)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                speaker: pool.submit(
                    acquire_one, speaker, revision, tree, args.destination, args.attempts
                )
                for speaker in protocol["dev_speaker_ids"]
            }
            records = [futures[speaker].result() for speaker in protocol["dev_speaker_ids"]]
        manifest = {
            "schema": "cross_tfg_dev_archive_acquisition_v1",
            "status": "COMPLETE",
            "dataset": DATASET_ID,
            "revision": revision,
            "protocol_path": str(args.protocol.resolve()),
            "protocol_raw_sha256": protocol_sha,
            "dataset_revision_path": str(args.revision_file.resolve()),
            "dataset_revision_file_sha256": revision_sha,
            "publisher_tree_path": str(args.archive_tree.resolve()),
            "publisher_tree_file_sha256": tree_sha,
            "dev_speaker_ids": protocol["dev_speaker_ids"],
            "confirm_speaker_ids_read": False,
            "audio_samples_read_or_decoded": False,
            "records": records,
        }
        manifest["manifest_sha256"] = canonical_sha(manifest)
        save_json(manifest_path, manifest)
        print(json.dumps({"status": "PASS", "manifest_path": str(manifest_path.resolve()),
                          "manifest_sha256": sha256_file(manifest_path),
                          "canonical_manifest_sha256": manifest["manifest_sha256"],
                          "speaker_count": len(records),
                          "total_archive_bytes": sum(row["bytes"] for row in records)},
                         sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
