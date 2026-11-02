from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INVENTORY_SHA256 = "8e52bfabbaf525fae307888929d688eecd35c5f33b25105502a64705f15561d5"
ROOTS = (
    Path("/root/tts-research-system-20261001/cache-migration/tts-research-20260930-conda-pkgs").resolve(),
    Path("/root/tts-research-system-20261001/cache-migration/autodl-tmp-conda-pkgs").resolve(),
)
REPODATA_BASENAME = re.compile(r"^[0-9a-f]{8}\.json$")
INFO_BASENAME = re.compile(r"^[0-9a-f]{8}\.info\.json$")


class InspectionError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def skip_string(data: mmap.mmap, start: int) -> int:
    if data[start] != 34:
        raise InspectionError("string parser did not start on a quote")
    cursor = start + 1
    while True:
        quote = data.find(b'"', cursor)
        if quote < 0:
            raise InspectionError("unterminated JSON string")
        backslash = quote - 1
        slash_count = 0
        while backslash >= start and data[backslash] == 92:
            slash_count += 1
            backslash -= 1
        if slash_count % 2 == 0:
            return quote + 1
        cursor = quote + 1


def skip_container(data: mmap.mmap, start: int) -> int:
    opening = data[start]
    if opening not in (123, 91):
        raise InspectionError("container parser did not start on object/array")
    stack = [125 if opening == 123 else 93]
    cursor = start + 1
    while stack:
        candidates = [(data.find(bytes((token,)), cursor), token) for token in (34, 123, 91, 125, 93)]
        candidates = [(position, token) for position, token in candidates if position >= 0]
        if not candidates:
            raise InspectionError("unterminated JSON container")
        position, token = min(candidates)
        if token == 34:
            cursor = skip_string(data, position)
            continue
        if token == 123:
            stack.append(125)
        elif token == 91:
            stack.append(93)
        else:
            if not stack or token != stack[-1]:
                raise InspectionError("mismatched JSON container closer")
            stack.pop()
        cursor = position + 1
    return cursor


def skip_scalar(data: mmap.mmap, start: int) -> int:
    cursor = start
    while cursor < len(data) and data[cursor] not in (44, 125, 9, 10, 13, 32):
        cursor += 1
    if cursor == start:
        raise InspectionError("empty JSON scalar")
    return cursor


def inspect_top_level(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0:
        raise InspectionError("empty JSON file")
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            cursor = 0
            while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                cursor += 1
            if cursor >= len(data) or data[cursor] != 123:
                raise InspectionError("top-level JSON is not an object")
            cursor += 1
            keys: list[dict[str, Any]] = []
            while True:
                while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                    cursor += 1
                if cursor < len(data) and data[cursor] == 125:
                    cursor += 1
                    break
                if cursor >= len(data) or data[cursor] != 34:
                    raise InspectionError("top-level object key is not a JSON string")
                key_start = cursor
                cursor = skip_string(data, cursor)
                key = json.loads(data[key_start:cursor].decode("utf-8"))
                while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                    cursor += 1
                if cursor >= len(data) or data[cursor] != 58:
                    raise InspectionError("missing colon after top-level key")
                cursor += 1
                while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                    cursor += 1
                value_start = cursor
                if cursor >= len(data):
                    raise InspectionError("missing top-level value")
                first = data[cursor]
                if first == 123:
                    value_type = "object"
                    cursor = skip_container(data, cursor)
                elif first == 91:
                    value_type = "array"
                    cursor = skip_container(data, cursor)
                elif first == 34:
                    value_type = "string"
                    cursor = skip_string(data, cursor)
                else:
                    cursor = skip_scalar(data, cursor)
                    raw = data[value_start:cursor]
                    value_type = "boolean" if raw in (b"true", b"false") else "null" if raw == b"null" else "number"
                keys.append({"key": key, "value_type": value_type, "value_bytes": cursor - value_start})
                while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                    cursor += 1
                if cursor < len(data) and data[cursor] == 44:
                    cursor += 1
                    continue
                if cursor < len(data) and data[cursor] == 125:
                    cursor += 1
                    break
                raise InspectionError("top-level object separator is invalid")
            while cursor < len(data) and data[cursor] in (9, 10, 13, 32):
                cursor += 1
            if cursor != len(data):
                raise InspectionError("trailing bytes after JSON object")
            return {"top_level_keys": keys, "top_level_key_count": len(keys)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    inventory_bytes = args.inventory.read_bytes()
    inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_sha != INVENTORY_SHA256:
        raise InspectionError("cache inventory file SHA does not match sealed attempt03")
    inventory = json.loads(inventory_bytes)
    if inventory.get("regular_file_count") != 78 or inventory.get("regular_file_total_bytes") != 2692591059:
        raise InspectionError("cache inventory coverage differs from sealed attempt03")
    if inventory.get("receipt_binding", {}).get("all_verified_tree_pairs_match") is not True:
        raise InspectionError("cache migration tree receipt is not fully verified")
    if inventory.get("symlinks_inside_cache_roots"):
        raise InspectionError("unexpected internal symlink in cache root")
    per_file = {row["absolute_resolved_path"]: row for row in inventory["per_file"]}
    candidates = []
    for row in inventory["per_file"]:
        name = row["basename"]
        path = Path(row["absolute_resolved_path"])
        if not REPODATA_BASENAME.fullmatch(name) and not INFO_BASENAME.fullmatch(name):
            continue
        root = next((candidate for candidate in ROOTS if path.is_relative_to(candidate)), None)
        if root is None or "/cache/" not in path.as_posix():
            raise InspectionError(f"JSON candidate is outside exact verified cache roots: {path}")
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise InspectionError(f"JSON candidate is not a regular non-symlink file: {path}")
        if path.stat().st_size != int(row["size_bytes"]) or sha_file(path) != row["sha256"]:
            raise InspectionError(f"JSON candidate no longer matches inventory03: {path}")
        schema = inspect_top_level(path)
        sibling_base = path.name.replace(".info.json", "").replace(".json", "")
        sibling_info = path.with_name(sibling_base + ".info.json")
        sibling_json = path.with_name(sibling_base + ".json")
        sibling_solv = path.with_name(sibling_base + ".solv")
        if not sibling_info.exists() or str(sibling_info) not in per_file:
            raise InspectionError(f"repodata JSON lacks inventoried info sidecar: {path}")
        keys = {entry["key"] for entry in schema["top_level_keys"]}
        if REPODATA_BASENAME.fullmatch(name):
            allowed_repodata_keys = {"info", "packages", "packages.conda", "repodata_version", "removed", "signatures"}
            if "packages" not in keys or "info" not in keys or not keys.issubset(allowed_repodata_keys):
                raise InspectionError(f"repodata JSON has an unexpected top-level schema: {path}")
            classification = "conda_repodata_json_cache"
        else:
            if not keys.issubset({"_url", "_etag", "_mod", "_cache_control", "_write_time", "_mtime", "_size", "etag", "mod", "cache_control", "url"}):
                raise InspectionError(f"repodata info JSON has unexpected top-level schema: {path}")
            if not keys.intersection({"_url", "url"}):
                raise InspectionError(f"repodata info JSON lacks a URL metadata field: {path}")
            classification = "conda_repodata_info_metadata_preserve"
        candidates.append({
            "path": str(path), "basename": name, "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"], "classification": classification,
            "top_level_keys_and_types": schema["top_level_keys"],
            "sibling_repodata_json": str(sibling_json) if INFO_BASENAME.fullmatch(name) else None,
            "sibling_info_metadata": str(sibling_info) if REPODATA_BASENAME.fullmatch(name) else None,
            "has_matching_solv_cache": bool(sibling_solv.is_file()),
        })
    repodata = [row for row in candidates if row["classification"] == "conda_repodata_json_cache"]
    info = [row for row in candidates if row["classification"] == "conda_repodata_info_metadata_preserve"]
    if len(repodata) != 28 or len(info) != 28:
        raise InspectionError(f"expected 28 repodata JSON and 28 info sidecars, got {len(repodata)} and {len(info)}")
    payload = {
        "schema_version": 1,
        "status": "PASS_STRUCTURAL_REPODATA_CLASSIFICATION",
        "classification": "top-level JSON key/type structure only; no URL/body values emitted; this report does not authorize deletion",
        "source_inventory_path": str(args.inventory.resolve()),
        "source_inventory_sha256": inventory_sha,
        "coverage": {"json_and_info_files": len(candidates), "repodata_json_count": len(repodata), "repodata_json_bytes": sum(r["size_bytes"] for r in repodata), "info_metadata_count": len(info), "info_metadata_bytes": sum(r["size_bytes"] for r in info)},
        "preserve_policy": "info.json, urls, urls.txt, all non-repodata/unknown files are retained",
        "files": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "file_count": len(candidates), "repodata_json_count": len(repodata), "repodata_json_bytes": payload["coverage"]["repodata_json_bytes"], "info_metadata_count": len(info), "info_metadata_bytes": payload["coverage"]["info_metadata_bytes"], "output": str(args.out), "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
