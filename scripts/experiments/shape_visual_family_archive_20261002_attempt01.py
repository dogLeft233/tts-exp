#!/usr/bin/env python3
"""Create and restore a hash-sealed, chunked archive for visual-family evidence.

The program is deliberately data-agnostic and standard-library-only. It does
not discover project files. A caller supplies an immutable JSON selection
specification. Every selected tree must be exhausted exactly against its own
sealed inventory, every loose file must have an explicit byte count and SHA,
and an optional tracking-plan binding verifies the raw plan, canonical payload,
and all declared input seals. External model archives are recorded as
references only; this program never copies them implicitly.

Specification shape (all hashes are lowercase SHA-256 hex):

{
  "schema": "TTS_VISUAL_FAMILY_ARCHIVE_SPEC_V1",
  "status": "SEALED_SOURCE_SELECTION",
  "transfer_id": "...",
  "pack_root": "/absolute/fresh/remote/target",
  "chunk_limit_bytes": 7000000,
  "spec_member": "transfer/archive_spec.json",
  "trees": [{
    "alias": "track_output",
    "source_root": "/absolute/sealed/tree",
    "archive_prefix": "track_output",
    "inventory": {"relative_path": "all_files_sha256.json",
      "sha256": "...", "format": "path_sha256_mapping"},
    "seal_files": [{"relative_path": "track_manifest.json",
      "bytes": 123, "sha256": "..."}],
    "extra_files": []
  }],
  "files": [{"alias": "tracking_plan", "source_path": "/abs/plan.json",
    "archive_member": "design/tracking_plan.json", "bytes": 123,
    "sha256": "..."}],
  "plan_binding": {"plan_member": "design/tracking_plan.json",
    "raw_sha256": "...", "payload_sha256": "...",
    "input_seal_count": 29,
    "input_seal_bindings": [{"seal_key": "...",
      "archive_member": "inputs/sealed_file.json"}]},
  "external_references": [{"kind": "verified_model_archive_reference",
    "archive_sha256": "...", "verification_receipt_sha256": "..."}]
}

Inventory formats are explicit: ``path_sha256_mapping`` accepts a JSON object
whose keys are normalized POSIX relative paths and values are SHA strings or
objects containing sha256/bytes. ``rows`` accepts a JSON list of row objects,
or an object plus ``inventory.container_key`` naming that list; row fields
default to path/sha256/bytes and can be explicitly overridden. Inventory and
seal/extra files are archived alongside inventory-listed members. No tree may
contain an unlisted regular file. Empty directories are harmless but are not
archive members.

Use ``pack --spec ... --expected-script-sha256 ...`` on the remote host,
``prepare-downloads`` on the local host after separately verifying pack
metadata, then ``restore`` into a fresh, absent directory. Receipts are written
outside the restored tree by design.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

SCHEMA = "TTS_VISUAL_FAMILY_ARCHIVE_SPEC_V1"
TRANSFER_SCHEMA = "TTS_VISUAL_FAMILY_TRANSFER_MANIFEST_V1"
DEFAULT_CHUNK_LIMIT = 7_000_000
BLOCK_SIZE = 1 << 20


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"invalid_sha256:{label}")
    return value.lower()


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid_integer:{label}")
    return value


def safe_rel(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"unsafe_relative_path:{label}")
    rel = PurePosixPath(value)
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"unsafe_relative_path:{label}")
    if any(part.endswith((".", " ")) or part.split(".", 1)[0].upper() in reserved or any(ord(ch) < 32 for ch in part) for part in rel.parts):
        raise ValueError(f"nonportable_relative_path:{label}")
    if rel.as_posix() != value:
        raise ValueError(f"noncanonical_relative_path:{label}")
    return rel


def safe_abs_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing_absolute_file:{label}")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or ".." in path.parts:
        raise ValueError(f"unsafe_absolute_file:{label}")
    resolved = path.resolve(strict=True)
    if path.absolute() != resolved:
        raise ValueError(f"symlink_or_noncanonical_path_component:{label}")
    mode = resolved.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"not_regular_file:{label}")
    return resolved


def safe_abs_dir(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing_absolute_directory:{label}")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or ".." in path.parts:
        raise ValueError(f"unsafe_absolute_directory:{label}")
    resolved = path.resolve(strict=True)
    if path.absolute() != resolved:
        raise ValueError(f"symlink_or_noncanonical_path_component:{label}")
    if not stat.S_ISDIR(resolved.lstat().st_mode):
        raise ValueError(f"not_directory:{label}")
    return resolved


def write_new(path: Path, raw: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return len(raw), sha_bytes(raw)


def write_json_new(path: Path, value: Any) -> tuple[int, str]:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return write_new(path, raw)


def load_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    return value, raw


def inventory_rows(spec: dict[str, Any], tree: dict[str, Any], inventory_path: Path) -> list[dict[str, Any]]:
    inv = tree.get("inventory")
    if not isinstance(inv, dict):
        raise ValueError("tree_inventory_spec_missing")
    raw = inventory_path.read_bytes()
    expected_inventory_sha = require_sha(inv.get("sha256"), f"inventory:{tree.get('alias')}")
    if sha_bytes(raw) != expected_inventory_sha:
        raise ValueError(f"tree_inventory_file_sha_mismatch:{tree.get('alias')}")
    document = json.loads(raw)
    fmt = inv.get("format")
    path_field = inv.get("path_field", "path")
    sha_field = inv.get("sha_field", "sha256")
    bytes_field = inv.get("bytes_field", "bytes")
    if fmt == "path_sha256_mapping":
        pointer = inv.get("mapping_pointer")
        mapping = json_pointer(document, pointer) if pointer else document
        if not isinstance(mapping, dict):
            raise ValueError(f"inventory_mapping_not_object:{tree.get('alias')}")
        pairs = list(mapping.items())
        rows: list[dict[str, Any]] = []
        for rel_value, value in pairs:
            rel = safe_rel(rel_value, "inventory_path").as_posix()
            if isinstance(value, str):
                digest, size = value, None
            elif isinstance(value, dict):
                digest, size = value.get(sha_field), value.get(bytes_field)
            else:
                raise ValueError(f"inventory_value_invalid:{tree.get('alias')}:{rel}")
            rows.append({"relative_path": rel, "sha256": require_sha(digest, f"inventory:{rel}"),
                         "bytes": require_int(size, f"inventory_bytes:{rel}") if size is not None else None})
    elif fmt == "rows":
        pointer = inv.get("rows_pointer")
        entries = json_pointer(document, pointer) if pointer else document
        if isinstance(entries, dict) and isinstance(inv.get("container_key"), str):
            entries = entries.get(inv["container_key"])
        if not isinstance(entries, list):
            raise ValueError(f"inventory_rows_not_list:{tree.get('alias')}")
        rows = []
        for index, value in enumerate(entries):
            if not isinstance(value, dict):
                raise ValueError(f"inventory_row_not_object:{tree.get('alias')}:{index}")
            rel = safe_rel(value.get(path_field), f"inventory:{index}").as_posix()
            size_value = value.get(bytes_field)
            rows.append({"relative_path": rel,
                         "sha256": require_sha(value.get(sha_field), f"inventory:{rel}"),
                         "bytes": require_int(size_value, f"inventory_bytes:{rel}") if size_value is not None else None})
    else:
        raise ValueError(f"unsupported_inventory_format:{fmt}")
    names = [row["relative_path"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate_inventory_path:{tree.get('alias')}")
    return sorted(rows, key=lambda row: row["relative_path"])


def json_pointer(value: Any, pointer: Any) -> Any:
    if pointer in (None, ""):
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("invalid_json_pointer")
    current = value
    for encoded in pointer.split("/")[1:]:
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, list):
            current = current[int(key)]
        else:
            raise ValueError("json_pointer_parent_not_container")
    return current


def walk_regular_tree(root: Path) -> tuple[list[Path], int]:
    files: list[Path] = []
    directory_count = 0
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        here = Path(current)
        kept: list[str] = []
        for name in dirs:
            path = here / name
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode):
                raise ValueError(f"non_directory_or_symlink:{path}")
            directory_count += 1
            kept.append(name)
        dirs[:] = kept
        for name in names:
            path = here / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"non_regular_or_symlink:{path}")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix()), directory_count


def add_source_row(rows: list[dict[str, Any]], *, member: str, path: Path,
                   alias: str, expected_sha: Any = None, expected_bytes: Any = None) -> None:
    rel = safe_rel(member, "archive_member").as_posix()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source_missing_or_unsafe:{alias}:{path}")
    path = path.resolve(strict=True)
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"source_not_regular:{alias}:{path}")
    size = path.stat().st_size
    digest = sha_file(path)
    if expected_bytes is not None and size != require_int(expected_bytes, f"source_bytes:{alias}"):
        raise ValueError(f"source_size_mismatch:{alias}:{path}")
    if expected_sha is not None and digest != require_sha(expected_sha, f"source_sha:{alias}"):
        raise ValueError(f"source_sha_mismatch:{alias}:{path}")
    rows.append({"alias": alias, "member": rel, "source_path": str(path), "bytes": size, "sha256": digest})


def collect_sources(spec: dict[str, Any], spec_path: Path, spec_raw: bytes) -> list[dict[str, Any]]:
    if spec.get("schema") != SCHEMA or spec.get("status") != "SEALED_SOURCE_SELECTION":
        raise ValueError("archive_spec_schema_or_status_invalid")
    rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_members: set[str] = set()
    seen_members_casefold: set[str] = set()
    trees = spec.get("trees")
    files = spec.get("files")
    if not isinstance(trees, list) or not isinstance(files, list):
        raise ValueError("archive_spec_trees_or_files_missing")
    tree_names: set[str] = set()
    for tree in trees:
        if not isinstance(tree, dict):
            raise ValueError("tree_spec_not_object")
        alias = tree.get("alias")
        if not isinstance(alias, str) or not alias or alias in tree_names:
            raise ValueError("tree_alias_invalid_or_duplicate")
        tree_names.add(alias)
        root = safe_abs_dir(tree.get("source_root"), f"tree:{alias}")
        prefix = safe_rel(tree.get("archive_prefix"), f"tree_prefix:{alias}").as_posix()
        inventory_rel = safe_rel(tree.get("inventory", {}).get("relative_path"), f"inventory_path:{alias}").as_posix()
        inventory_file = root.joinpath(*PurePosixPath(inventory_rel).parts)
        if not inventory_file.is_file() or inventory_file.is_symlink():
            raise ValueError(f"inventory_file_missing_or_unsafe:{alias}")
        inv_rows = inventory_rows(spec, tree, inventory_file)
        declared: dict[str, tuple[str, int | None, str]] = {}
        for row in inv_rows:
            if row["relative_path"] == inventory_rel:
                raise ValueError(f"inventory_lists_itself:{alias}")
            declared[row["relative_path"]] = (row["sha256"], row["bytes"], "sealed_inventory")
        extras = []
        for category in ("seal_files", "extra_files"):
            values = tree.get(category, [])
            if not isinstance(values, list):
                raise ValueError(f"tree_{category}_not_list:{alias}")
            for row in values:
                if not isinstance(row, dict):
                    raise ValueError(f"tree_{category}_row_not_object:{alias}")
                rel = safe_rel(row.get("relative_path"), f"{category}:{alias}").as_posix()
                digest = require_sha(row.get("sha256"), f"{category}:{alias}:{rel}")
                size = require_int(row.get("bytes"), f"{category}_bytes:{alias}:{rel}")
                if rel in declared:
                    if declared[rel][0] != digest or (declared[rel][1] is not None and declared[rel][1] != size):
                        raise ValueError(f"extra_file_conflicts_with_inventory:{alias}:{rel}")
                else:
                    declared[rel] = (digest, size, category)
                extras.append(rel)
        declared[inventory_rel] = (require_sha(tree["inventory"]["sha256"], f"inventory_file:{alias}"),
                                   inventory_file.stat().st_size, "inventory_file")
        actual, dir_count = walk_regular_tree(root)
        actual_rel = {path.relative_to(root).as_posix() for path in actual}
        if actual_rel != set(declared):
            missing = sorted(set(declared) - actual_rel)[:5]
            unlisted = sorted(actual_rel - set(declared))[:5]
            raise ValueError(f"tree_file_set_mismatch:{alias}:missing={missing}:unlisted={unlisted}")
        for rel, (digest, size, source_kind) in sorted(declared.items()):
            source = root.joinpath(*PurePosixPath(rel).parts)
            add_source_row(rows, member=f"{prefix}/{rel}", path=source, alias=f"{alias}:{source_kind}",
                           expected_sha=digest, expected_bytes=size)
        tree["_resolved_directory_count"] = dir_count
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("loose_file_spec_not_object")
        alias = row.get("alias")
        if not isinstance(alias, str) or not alias:
            raise ValueError("loose_file_alias_missing")
        member = safe_rel(row.get("archive_member"), f"file_member:{alias}").as_posix()
        path = safe_abs_file(row.get("source_path"), f"file:{alias}")
        require_sha(row.get("sha256"), f"file_sha:{alias}")
        require_int(row.get("bytes"), f"file_bytes:{alias}")
        add_source_row(rows, member=member, path=path, alias=alias,
                       expected_sha=row.get("sha256"), expected_bytes=row.get("bytes"))
    spec_member = safe_rel(spec.get("spec_member"), "spec_member").as_posix()
    add_source_row(rows, member=spec_member, path=spec_path, alias="archive_selection_spec",
                   expected_sha=sha_bytes(spec_raw), expected_bytes=len(spec_raw))
    for row in rows:
        if row["source_path"] in seen_sources:
            raise ValueError(f"duplicate_source_path:{row['source_path']}")
        if row["member"] in seen_members:
            raise ValueError(f"duplicate_archive_member:{row['member']}")
        if row["member"].casefold() in seen_members_casefold:
            raise ValueError(f"case_insensitive_archive_member_collision:{row['member']}")
        seen_sources.add(row["source_path"])
        seen_members.add(row["member"])
        seen_members_casefold.add(row["member"].casefold())
    reserved = {"transfer/source_inventory_before.json", "transfer/transfer_provenance.json"}
    reserved_casefold = {item.casefold() for item in reserved}
    collision = sorted(seen_members_casefold & reserved_casefold)
    if collision:
        raise ValueError(f"source_selection_uses_reserved_transfer_member:{collision[0]}")
    if spec.get("model_bulk_included", False) is not False:
        raise ValueError("model_bulk_must_be_reference_only")
    references = spec.get("external_references", [])
    if not isinstance(references, list):
        raise ValueError("external_references_not_list")
    for ref in references:
        if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str):
            raise ValueError("external_reference_invalid")
        for key, value in ref.items():
            if key.endswith("sha256"):
                require_sha(value, f"external_reference:{key}")
        if ref.get("kind") == "verified_model_archive_reference":
            if (ref.get("verification_status") != "PASS"
                    or "archive_sha256" not in ref
                    or "verification_receipt_sha256" not in ref):
                raise ValueError("model_archive_reference_not_verified")
    rows.sort(key=lambda row: row["member"])
    verify_plan_binding(spec, rows)
    return rows


def verify_plan_binding(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    binding = spec.get("plan_binding")
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError("plan_binding_not_object")
    member = safe_rel(binding.get("plan_member"), "plan_member").as_posix()
    row = next((item for item in rows if item["member"] == member), None)
    if row is None:
        raise ValueError("plan_member_not_in_archive_selection")
    raw_sha = require_sha(binding.get("raw_sha256"), "plan_raw_sha256")
    payload_sha = require_sha(binding.get("payload_sha256"), "plan_payload_sha256")
    if row["sha256"] != raw_sha:
        raise ValueError("plan_raw_sha_mismatch")
    document_path = Path(row.get("data_path", row["source_path"]))
    document = json.loads(document_path.read_bytes())
    if not isinstance(document, dict) or document.get("plan_sha256") != payload_sha:
        raise ValueError("plan_embedded_payload_sha_mismatch")
    payload = dict(document)
    payload.pop("plan_sha256", None)
    computed = sha_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if computed != payload_sha:
        raise ValueError("plan_canonical_payload_sha_mismatch")
    seals = document.get("input_seals")
    if not isinstance(seals, dict):
        raise ValueError("plan_input_seals_not_object")
    expected_count = require_int(binding.get("input_seal_count"), "plan_input_seal_count", minimum=1)
    if len(seals) != expected_count:
        raise ValueError("plan_input_seal_count_mismatch")
    seal_bindings = binding.get("input_seal_bindings")
    if not isinstance(seal_bindings, list) or len(seal_bindings) != expected_count:
        raise ValueError("plan_input_seal_bindings_count_mismatch")
    by_key: dict[str, str] = {}
    archive_rows = {item["member"]: item for item in rows}
    for item in seal_bindings:
        if not isinstance(item, dict):
            raise ValueError("plan_input_seal_binding_invalid")
        key = item.get("seal_key")
        if not isinstance(key, str) or key not in seals or key in by_key:
            raise ValueError("plan_input_seal_key_invalid_or_duplicate")
        archive_member = safe_rel(item.get("archive_member"), f"input_seal_member:{key}").as_posix()
        seal = seals[key]
        if not isinstance(seal, dict):
            raise ValueError(f"plan_input_seal_row_invalid:{key}")
        digest = require_sha(seal.get("raw_sha256"), f"plan_input_seal:{key}")
        size = require_int(seal.get("bytes"), f"plan_input_seal_bytes:{key}")
        archived = archive_rows.get(archive_member)
        if archived is None or archived["sha256"] != digest or archived["bytes"] != size:
            raise ValueError(f"plan_input_seal_archive_binding_mismatch:{key}")
        sealed_source_path = seal.get("path")
        archived_source_path = archived.get("source_path")
        if not isinstance(sealed_source_path, str) or not isinstance(archived_source_path, str):
            raise ValueError(f"plan_input_seal_path_missing:{key}")
        override = item.get("source_path_override")
        if sealed_source_path != archived_source_path:
            if not isinstance(override, dict):
                raise ValueError(f"plan_input_seal_source_override_missing:{key}")
            if (override.get("source_path") != archived_source_path
                    or not isinstance(override.get("reason"), str)
                    or not override["reason"].strip()):
                raise ValueError(f"plan_input_seal_source_override_invalid:{key}")
        elif override is not None and (not isinstance(override, dict) or override.get("source_path") != archived_source_path):
            raise ValueError(f"plan_input_seal_source_override_mismatch:{key}")
        by_key[key] = archive_member
    if set(by_key) != set(seals):
        raise ValueError("plan_input_seal_binding_coverage_mismatch")
    return {"plan_member": member, "plan_raw_sha256": raw_sha, "plan_payload_sha256": payload_sha,
            "input_seal_count": expected_count, "input_seal_bindings": by_key,
            "source_path_overrides": {item["seal_key"]: item["source_path_override"]
                                      for item in seal_bindings if item.get("source_path_override") is not None}}


class ChunkSink:
    def __init__(self, directory: Path, limit: int) -> None:
        self.directory = directory
        self.limit = limit
        self.index = 0
        self.stream: BinaryIO | None = None
        self.chunk_hash: Any = None
        self.chunk_bytes = 0
        self.total_hash = hashlib.sha256()
        self.total_bytes = 0
        self.chunks: list[dict[str, Any]] = []

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.total_bytes

    def _open(self) -> None:
        path = self.directory / f"part{self.index:06d}.bin"
        self.stream = path.open("xb")
        self.chunk_hash = hashlib.sha256()
        self.chunk_bytes = 0

    def _close(self) -> None:
        if self.stream is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        path = self.directory / f"part{self.index:06d}.bin"
        self.chunks.append({"name": path.name, "bytes": self.chunk_bytes, "sha256": self.chunk_hash.hexdigest()})
        self.index += 1
        self.stream = None

    def write(self, data: bytes | bytearray | memoryview) -> int:
        view = memoryview(data)
        count = len(view)
        offset = 0
        while offset < count:
            if self.stream is None:
                self._open()
            take = min(self.limit - self.chunk_bytes, count - offset)
            block = view[offset:offset + take]
            self.stream.write(block)
            self.chunk_hash.update(block)
            self.total_hash.update(block)
            self.chunk_bytes += take
            self.total_bytes += take
            offset += take
            if self.chunk_bytes == self.limit:
                self._close()
        return count

    def flush(self) -> None:
        if self.stream is not None:
            self.stream.flush()

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return self.chunks


class ChunkSource:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.i = 0
        self.stream: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = BLOCK_SIZE
        blocks: list[bytes] = []
        remain = size
        while remain:
            if self.stream is None:
                if self.i >= len(self.paths):
                    break
                self.stream = self.paths[self.i].open("rb")
                self.i += 1
            block = self.stream.read(remain)
            if block:
                blocks.append(block)
                remain -= len(block)
            else:
                self.stream.close()
                self.stream = None
        return b"".join(blocks)

    def close(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> "ChunkSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def scan_rows(root: Path) -> list[dict[str, Any]]:
    root = safe_abs_dir(str(root), "scan_root")
    files, _ = walk_regular_tree(root)
    out = []
    for path in files:
        out.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return out


def snapshot_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        path = Path(row["source_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source_changed_or_missing:{row['member']}")
        size, digest = path.stat().st_size, sha_file(path)
        if size != row["bytes"] or digest != row["sha256"]:
            raise ValueError(f"source_changed_from_selection:{row['member']}")
        out.append({k: row[k] for k in ("alias", "member", "source_path", "bytes", "sha256")})
    return out


def verify_archive(chunks_dir: Path, chunks: list[dict[str, Any]], archive_sha: str,
                   archive_bytes: int, expected_members: list[dict[str, Any]], chunk_limit: int) -> dict[str, Any]:
    expected_chunk_names = [f"part{i:06d}.bin" for i in range(len(chunks))]
    if sorted(p.name for p in chunks_dir.iterdir()) != expected_chunk_names:
        raise ValueError("chunk_output_set_mismatch")
    paths: list[Path] = []
    full_hash = hashlib.sha256()
    total = 0
    for i, row in enumerate(chunks):
        name = expected_chunk_names[i]
        path = chunks_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or path.stat().st_size > chunk_limit or sha_file(path) != row["sha256"]:
            raise ValueError(f"created_chunk_verification_failed:{name}")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                full_hash.update(block)
                total += len(block)
        paths.append(path)
    if total != archive_bytes or full_hash.hexdigest() != archive_sha:
        raise ValueError("created_archive_hash_or_size_mismatch")
    expected = {r["member"]: r for r in expected_members}
    seen: list[str] = []
    with ChunkSource(paths) as source:
        with gzip.GzipFile(fileobj=source, mode="rb") as ungzip:
            with tarfile.open(fileobj=ungzip, mode="r|") as tar:
                for info in tar:
                    safe_rel(info.name, "tar_member")
                    if not info.isfile() or info.name not in expected or info.name in seen:
                        raise ValueError("archive_member_unexpected_duplicate_or_nonregular")
                    row = expected[info.name]
                    if info.size != row["bytes"]:
                        raise ValueError(f"tar_member_size_mismatch:{info.name}")
                    extracted = tar.extractfile(info)
                    if extracted is None:
                        raise ValueError(f"tar_member_unreadable:{info.name}")
                    h = hashlib.sha256()
                    n = 0
                    with extracted:
                        for block in iter(lambda: extracted.read(BLOCK_SIZE), b""):
                            h.update(block)
                            n += len(block)
                    if n != row["bytes"] or h.hexdigest() != row["sha256"]:
                        raise ValueError(f"tar_member_sha_mismatch:{info.name}")
                    seen.append(info.name)
    if seen != [r["member"] for r in expected_members]:
        raise ValueError("archive_member_order_or_coverage_mismatch")
    return {"archive_bytes": total, "archive_sha256": full_hash.hexdigest(), "verified_member_count": len(seen)}


def package(spec_path: Path, expected_script_sha: str, script_path: Path) -> dict[str, Any]:
    spec_path = safe_abs_file(str(spec_path), "spec")
    script_path = safe_abs_file(str(script_path), "script")
    expected_script_sha = require_sha(expected_script_sha, "expected_script")
    if sha_file(script_path) != expected_script_sha:
        raise ValueError("packer_self_sha_mismatch")
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    if not isinstance(spec, dict):
        raise ValueError("spec_root_not_object")
    pack_root_value = spec.get("pack_root")
    if not isinstance(pack_root_value, str) or not Path(pack_root_value).is_absolute():
        raise ValueError("pack_root_must_be_absolute")
    pack_root = Path(pack_root_value)
    if ".." in pack_root.parts or pack_root.resolve(strict=False) != pack_root.absolute():
        raise ValueError("pack_root_not_canonical")
    if pack_root.exists() or pack_root.is_symlink():
        raise ValueError("refusing_existing_pack_root")
    chunk_limit = require_int(spec.get("chunk_limit_bytes", DEFAULT_CHUNK_LIMIT), "chunk_limit", minimum=1)
    if chunk_limit > DEFAULT_CHUNK_LIMIT:
        raise ValueError("chunk_limit_exceeds_7000000")
    transfer_id = spec.get("transfer_id")
    if not isinstance(transfer_id, str) or not transfer_id or "/" in transfer_id or "\\" in transfer_id:
        raise ValueError("transfer_id_invalid")
    # Prevent writing an archive beneath any selected source tree.
    for tree in spec.get("trees", []):
        root = safe_abs_dir(tree.get("source_root"), f"tree:{tree.get('alias')}")
        if pack_root.resolve(strict=False).is_relative_to(root):
            raise ValueError("pack_root_inside_source_tree")
    source_rows = collect_sources(spec, spec_path, spec_raw)
    before = snapshot_sources(source_rows)
    tree_before = [{"alias": tree["alias"], "source_root": str(Path(tree["source_root"]).resolve(strict=True)),
                    "directory_count": tree["_resolved_directory_count"],
                    "inventory_relative_path": tree["inventory"]["relative_path"],
                    "inventory_sha256": tree["inventory"]["sha256"]}
                   for tree in spec.get("trees", [])]
    source_bytes = sum(row["bytes"] for row in before)
    parent = safe_abs_dir(str(pack_root.parent), "pack_root_parent")
    st = os.statvfs(parent)
    available = st.f_bavail * st.f_frsize
    required = source_bytes + max(64 << 20, source_bytes // 20)
    if available < required:
        raise ValueError(f"insufficient_free_space:{available}<{required}")
    pack_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    payload = pack_root / "payload"
    chunks_dir = payload / "chunks"
    chunks_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    before_path = pack_root / "source_inventory_before.json"
    before_obj = {"schema": "TTS_ARCHIVE_SOURCE_INVENTORY_V1", "status": "SOURCE_INVENTORY_BEFORE_ARCHIVE",
                  "transfer_id": transfer_id, "source_files": before}
    before_bytes, before_sha = write_json_new(before_path, before_obj)
    provenance = {
        "schema": "TTS_VISUAL_FAMILY_TRANSFER_PROVENANCE_V1",
        "status": "SEALED_SOURCE_SELECTION",
        "transfer_id": transfer_id,
        "archive_spec_sha256": sha_bytes(spec_raw),
        "archive_spec_member": safe_rel(spec.get("spec_member"), "spec_member").as_posix(),
        "plan_binding": verify_plan_binding(spec, source_rows),
        "external_references": spec.get("external_references", []),
        "source_tree_seals": [
            {"alias": t["alias"], "archive_prefix": t["archive_prefix"], "inventory": t["inventory"],
             "seal_files": t.get("seal_files", []), "extra_files": t.get("extra_files", []),
             "source_root": str(Path(t["source_root"]).resolve(strict=True)),
             "directory_count": t["_resolved_directory_count"]}
            for t in spec.get("trees", [])
        ],
        "scope": spec.get("scope", "explicitly selected sealed visual-family inputs and outputs; external model archives are references only"),
        "model_bulk_included": False,
        "created_utc_epoch": time.time(),
    }
    provenance_path = payload / "transfer_provenance.json"
    prov_bytes, prov_sha = write_json_new(provenance_path, provenance)
    members = [dict(row) for row in before]
    members.extend([
        {"member": "transfer/source_inventory_before.json", "bytes": before_bytes, "sha256": before_sha,
         "source_path": str(before_path), "alias": "transfer_inventory"},
        {"member": "transfer/transfer_provenance.json", "bytes": prov_bytes, "sha256": prov_sha,
         "source_path": str(provenance_path), "alias": "transfer_provenance"},
    ])
    members.sort(key=lambda row: row["member"])
    member_names = [row["member"] for row in members]
    if len(member_names) != len(set(member_names)):
        raise ValueError("duplicate_generated_archive_member")
    sink = ChunkSink(chunks_dir, chunk_limit)
    with gzip.GzipFile(fileobj=sink, mode="wb", filename="", compresslevel=6, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for row in members:
                path = Path(row["source_path"])
                info = tarfile.TarInfo(name=row["member"])
                info.size = row["bytes"]
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as source:
                    tar.addfile(info, source)
    chunks = sink.finish()
    archive_sha = sink.total_hash.hexdigest()
    archive_bytes = sink.total_bytes
    # Re-enumerate inventories and roots after streaming. This catches new or
    # removed files as well as mutations to files selected before the archive.
    source_rows_after = collect_sources(spec, spec_path, spec_raw)
    after = snapshot_sources(source_rows_after)
    tree_after = [{"alias": tree["alias"], "source_root": str(Path(tree["source_root"]).resolve(strict=True)),
                   "directory_count": tree["_resolved_directory_count"],
                   "inventory_relative_path": tree["inventory"]["relative_path"],
                   "inventory_sha256": tree["inventory"]["sha256"]}
                  for tree in spec.get("trees", [])]
    if after != before or tree_after != tree_before:
        raise ValueError("source_tree_changed_during_archive")
    after_path = pack_root / "source_inventory_after.json"
    after_obj = {"schema": "TTS_ARCHIVE_SOURCE_INVENTORY_V1", "status": "SOURCE_INVENTORY_AFTER_ARCHIVE",
                 "transfer_id": transfer_id, "source_files": after, "matches_source_inventory_before": True}
    after_bytes, after_sha = write_json_new(after_path, after_obj)
    archive_readback = verify_archive(chunks_dir, chunks, archive_sha, archive_bytes, members, chunk_limit)
    manifest = {
        "schema": TRANSFER_SCHEMA, "status": "OUTPUT_TRANSFER_SEALED", "transfer_id": transfer_id,
        "pack_root": str(pack_root), "payload_root": str(payload),
        "chunk_limit_bytes": chunk_limit, "archive_format": "deterministic tar streamed through gzip; ordered chunks concatenate byte-for-byte",
        "archive_bytes": archive_bytes, "archive_sha256": archive_sha,
        "chunks": [{"name": row["name"], "path": f"{pack_root}/payload/chunks/{row['name']}",
                    "bytes": row["bytes"], "sha256": row["sha256"]} for row in chunks],
        "chunk_count": len(chunks), "source_inventory_before_path": str(before_path),
        "source_inventory_before_bytes": before_bytes, "source_inventory_before_sha256": before_sha,
        "source_inventory_after_path": str(after_path), "source_inventory_after_bytes": after_bytes,
        "source_inventory_after_sha256": after_sha, "source_inventory_before_after_identical": True,
        "source_file_count": len(before), "source_bytes": source_bytes,
        "archive_member_count": len(members), "archive_member_bytes": sum(row["bytes"] for row in members),
        "archive_member_inventory": [{"member": row["member"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in members],
        "archive_readback_verification": archive_readback,
        "archive_spec_sha256": sha_bytes(spec_raw), "archive_spec_member": spec["spec_member"],
        "source_tree_snapshots_before": tree_before, "source_tree_snapshots_after": tree_after,
        "model_bulk_included": False,
    }
    manifest_path = pack_root / "transfer_manifest.json"
    manifest_bytes, manifest_sha = write_json_new(manifest_path, manifest)
    receipt = {
        "schema": "TTS_VISUAL_FAMILY_TRANSFER_RECEIPT_V1", "status": "OUTPUT_TRANSFER_PASS",
        "transfer_id": transfer_id, "transfer_manifest_path": str(manifest_path),
        "transfer_manifest_bytes": manifest_bytes, "transfer_manifest_sha256": manifest_sha,
        "archive_sha256": archive_sha, "archive_bytes": archive_bytes,
        "chunk_count": len(chunks), "archive_member_count": len(members),
        "source_inventory_before_after_identical": True, "archive_readback_verified": True,
    }
    receipt_path = pack_root / "pack_receipt.json"
    receipt_bytes, receipt_sha = write_json_new(receipt_path, receipt)
    return {**receipt, "pack_receipt_bytes": receipt_bytes, "pack_receipt_sha256": receipt_sha,
            "archive_readback_verification": archive_readback}


def verify_transfer_metadata(metadata_dir: Path, *, expected_manifest_sha: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    metadata_dir = safe_abs_dir(str(metadata_dir), "metadata_dir")
    names = ("transfer_manifest.json", "pack_receipt.json", "source_inventory_before.json", "source_inventory_after.json")
    values: dict[str, Any] = {}
    raw: dict[str, bytes] = {}
    for name in names:
        path = metadata_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"transfer_metadata_missing_or_unsafe:{name}")
        values[name], raw[name] = load_json(path)
    manifest = values[names[0]]
    receipt = values[names[1]]
    before = values[names[2]]
    after = values[names[3]]
    manifest_sha = sha_bytes(raw[names[0]])
    if expected_manifest_sha and manifest_sha != require_sha(expected_manifest_sha, "expected_manifest"):
        raise ValueError("expected_transfer_manifest_sha_mismatch")
    if manifest.get("schema") != TRANSFER_SCHEMA or manifest.get("status") != "OUTPUT_TRANSFER_SEALED" or receipt.get("status") != "OUTPUT_TRANSFER_PASS":
        raise ValueError("transfer_status_or_schema_not_pass")
    if receipt.get("transfer_manifest_sha256") != manifest_sha or receipt.get("transfer_manifest_bytes") != len(raw[names[0]]):
        raise ValueError("receipt_manifest_binding_mismatch")
    pack_root = manifest.get("pack_root")
    if not isinstance(pack_root, str) or not Path(pack_root).is_absolute():
        raise ValueError("manifest_pack_root_invalid")
    if (receipt.get("transfer_manifest_path") != f"{pack_root}/transfer_manifest.json"
            or manifest.get("source_inventory_before_path") != f"{pack_root}/source_inventory_before.json"
            or manifest.get("source_inventory_after_path") != f"{pack_root}/source_inventory_after.json"
            or manifest.get("payload_root") != f"{pack_root}/payload"):
        raise ValueError("manifest_path_bindings_invalid")
    if (manifest.get("source_inventory_before_sha256") != sha_bytes(raw[names[2]])
            or manifest.get("source_inventory_after_sha256") != sha_bytes(raw[names[3]])
            or manifest.get("source_inventory_before_bytes") != len(raw[names[2]])
            or manifest.get("source_inventory_after_bytes") != len(raw[names[3]])
            or before.get("status") != "SOURCE_INVENTORY_BEFORE_ARCHIVE"
            or after.get("status") != "SOURCE_INVENTORY_AFTER_ARCHIVE"
            or before.get("schema") != "TTS_ARCHIVE_SOURCE_INVENTORY_V1"
            or after.get("schema") != "TTS_ARCHIVE_SOURCE_INVENTORY_V1"
            or before.get("transfer_id") != manifest.get("transfer_id")
            or after.get("transfer_id") != manifest.get("transfer_id")
            or after.get("matches_source_inventory_before") is not True
            or before.get("source_files") != after.get("source_files")
            or manifest.get("source_inventory_before_after_identical") is not True):
        raise ValueError("source_inventory_before_after_binding_invalid")
    if (receipt.get("archive_sha256") != manifest.get("archive_sha256")
            or receipt.get("archive_bytes") != manifest.get("archive_bytes")
            or receipt.get("chunk_count") != manifest.get("chunk_count")
            or receipt.get("archive_member_count") != manifest.get("archive_member_count")
            or receipt.get("archive_readback_verified") is not True):
        raise ValueError("receipt_archive_binding_invalid")
    archive_readback = manifest.get("archive_readback_verification")
    if (not isinstance(archive_readback, dict)
            or archive_readback.get("archive_sha256") != manifest.get("archive_sha256")
            or archive_readback.get("archive_bytes") != manifest.get("archive_bytes")
            or archive_readback.get("verified_member_count") != manifest.get("archive_member_count")):
        raise ValueError("remote_archive_readback_binding_invalid")
    if manifest.get("model_bulk_included") is not False:
        raise ValueError("model_bulk_inclusion_policy_invalid")
    source_files = before.get("source_files")
    if (not isinstance(source_files, list)
            or manifest.get("source_file_count") != len(source_files)
            or manifest.get("source_bytes") != sum(require_int(row.get("bytes"), "source_file_bytes") for row in source_files)):
        raise ValueError("source_inventory_count_or_bytes_invalid")
    tree_before = manifest.get("source_tree_snapshots_before")
    tree_after = manifest.get("source_tree_snapshots_after")
    if not isinstance(tree_before, list) or tree_before != tree_after:
        raise ValueError("source_tree_snapshot_before_after_mismatch")
    return manifest, receipt, before, after, manifest_sha


def prepare_downloads(metadata_dir: Path, chunks_dir: Path, request_dir: Path,
                      expected_manifest_sha: str | None = None) -> dict[str, Any]:
    manifest, receipt, before, after, manifest_sha = verify_transfer_metadata(metadata_dir, expected_manifest_sha=expected_manifest_sha)
    chunks = manifest.get("chunks")
    limit = require_int(manifest.get("chunk_limit_bytes"), "chunk_limit", minimum=1)
    if limit > DEFAULT_CHUNK_LIMIT or not isinstance(chunks, list) or not chunks:
        raise ValueError("chunk_policy_or_list_invalid")
    if not chunks_dir.is_absolute() or not request_dir.is_absolute():
        raise ValueError("download_targets_must_be_absolute")
    if chunks_dir.resolve(strict=False).is_relative_to(request_dir.resolve(strict=False)) or request_dir.resolve(strict=False).is_relative_to(chunks_dir.resolve(strict=False)):
        raise ValueError("download_targets_must_not_overlap")
    metadata_resolved = metadata_dir.resolve(strict=True)
    if (chunks_dir.resolve(strict=False).is_relative_to(metadata_resolved)
            or metadata_resolved.is_relative_to(chunks_dir.resolve(strict=False))
            or request_dir.resolve(strict=False).is_relative_to(metadata_resolved)
            or metadata_resolved.is_relative_to(request_dir.resolve(strict=False))):
        raise ValueError("download_targets_must_not_overlap_metadata")
    safe_abs_dir(str(chunks_dir.parent), "chunks_parent")
    safe_abs_dir(str(request_dir.parent), "request_parent")
    if chunks_dir.exists() or chunks_dir.is_symlink() or request_dir.exists() or request_dir.is_symlink():
        raise ValueError("refusing_existing_download_target")
    expected_total = 0
    requests = []
    remote_root = manifest["pack_root"]
    for i, row in enumerate(chunks):
        name = f"part{i:06d}.bin"
        if not isinstance(row, dict) or row.get("name") != name or row.get("path") != f"{remote_root}/payload/chunks/{name}":
            raise ValueError("chunk_manifest_order_or_path_invalid")
        size = require_int(row.get("bytes"), f"chunk_bytes:{name}", minimum=1)
        digest = require_sha(row.get("sha256"), f"chunk_sha:{name}")
        if size > limit:
            raise ValueError(f"chunk_exceeds_limit:{name}")
        expected_total += size
        requests.append({"op": "download", "remote": row["path"], "local": str(chunks_dir / name),
                         "expected_bytes": size, "expected_sha256": digest,
                         "dispatch_gate": "transfer manifest, receipt, and source before/after inventories verified"})
    if expected_total != manifest.get("archive_bytes") or len(chunks) != manifest.get("chunk_count"):
        raise ValueError("chunk_aggregate_mismatch")
    request_dir.mkdir(exist_ok=False)
    chunks_dir.mkdir(exist_ok=False)
    request_rows = []
    for i, request in enumerate(requests):
        path = request_dir / f"{i:06d}_download_{Path(request['local']).name}.json"
        size, digest = write_json_new(path, request)
        request_rows.append({"path": str(path), "bytes": size, "sha256": digest,
                             "remote": request["remote"], "local": request["local"],
                             "expected_bytes": request["expected_bytes"], "expected_sha256": request["expected_sha256"]})
    index = {"schema": "TTS_VISUAL_FAMILY_CHUNK_DOWNLOAD_INDEX_V1", "status": "PREPARED_NOT_DISPATCHED",
             "transfer_id": manifest["transfer_id"], "transfer_manifest_sha256": manifest_sha,
             "pack_receipt_sha256": sha_file(Path(metadata_dir) / "pack_receipt.json"),
             "archive_sha256": manifest["archive_sha256"], "archive_bytes": expected_total,
             "chunk_limit_bytes": limit, "chunk_count": len(chunks), "requests": request_rows}
    index_path = request_dir / "request_index.json"
    index_size, index_sha = write_json_new(index_path, index)
    return {"status": index["status"], "request_index_path": str(index_path), "request_index_bytes": index_size,
            "request_index_sha256": index_sha, "chunk_count": len(chunks), "archive_bytes": expected_total}


def write_restored_member(root: Path, rel: PurePosixPath, stream: BinaryIO, size_expected: int, sha_expected: str) -> dict[str, Any]:
    target = root.joinpath(*rel.parts)
    if not target.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
        raise ValueError("restore_target_escaped")
    parent = root
    for part in rel.parts[:-1]:
        parent = parent / part
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("unsafe_restore_parent")
        else:
            parent.mkdir(mode=0o700)
    if target.exists() or target.is_symlink():
        raise ValueError("refusing_existing_restore_member")
    hasher = hashlib.sha256()
    count = 0
    with target.open("xb") as output:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            count += len(block)
            if count > size_expected:
                raise ValueError("restored_member_over_expected_size")
            hasher.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if count != size_expected or hasher.hexdigest() != sha_expected:
        raise ValueError("restored_member_hash_or_size_mismatch")
    return {"member": rel.as_posix(), "bytes": count, "sha256": hasher.hexdigest()}


def audit_restored_spec(restore_root: Path, manifest: dict[str, Any], member_map: dict[str, tuple[int, str]]) -> dict[str, Any]:
    spec_member = safe_rel(manifest.get("archive_spec_member"), "archive_spec_member").as_posix()
    spec_path = restore_root.joinpath(*PurePosixPath(spec_member).parts)
    spec_raw = spec_path.read_bytes()
    if sha_bytes(spec_raw) != require_sha(manifest.get("archive_spec_sha256"), "manifest_spec_sha"):
        raise ValueError("restored_archive_spec_sha_mismatch")
    spec = json.loads(spec_raw)
    if spec.get("transfer_id") != manifest.get("transfer_id") or spec.get("pack_root") != manifest.get("pack_root"):
        raise ValueError("restored_archive_spec_identity_mismatch")
    if spec.get("model_bulk_included", False) is not False:
        raise ValueError("restored_spec_model_bulk_policy_invalid")
    references = spec.get("external_references", [])
    if not isinstance(references, list):
        raise ValueError("restored_external_references_invalid")
    for ref in references:
        if not isinstance(ref, dict) or not isinstance(ref.get("kind"), str):
            raise ValueError("restored_external_reference_invalid")
        for key, value in ref.items():
            if key.endswith("sha256"):
                require_sha(value, f"restored_external_reference:{key}")
        if ref.get("kind") == "verified_model_archive_reference":
            if ref.get("verification_status") != "PASS" or "archive_sha256" not in ref or "verification_receipt_sha256" not in ref:
                raise ValueError("restored_model_archive_reference_not_verified")
    # Check each sealed source tree using archive member paths. Each file was
    # already hashed during extraction; this pass replays the tree's own index.
    checked_trees = 0
    for tree in spec.get("trees", []):
        prefix = safe_rel(tree.get("archive_prefix"), "restored_tree_prefix").as_posix()
        inv = tree.get("inventory")
        if not isinstance(inv, dict):
            raise ValueError("restored_tree_inventory_missing")
        rel_inventory = safe_rel(inv.get("relative_path"), "restored_inventory_path").as_posix()
        inventory_member = f"{prefix}/{rel_inventory}"
        inventory_sha = require_sha(inv.get("sha256"), f"restored_inventory_file:{tree.get('alias')}")
        if member_map.get(inventory_member, (None, None))[1] != inventory_sha:
            raise ValueError(f"restored_inventory_file_sha_mismatch:{tree.get('alias')}")
        inv_data = json.loads(restore_root.joinpath(*PurePosixPath(inventory_member).parts).read_bytes())
        # Reuse inventory parser semantics on the already restored file.
        rows = inventory_rows_from_document(inv_data, inv)
        expected_paths = {row["relative_path"] for row in rows}
        expected_paths.add(rel_inventory)
        expected_paths.update(safe_rel(row.get("relative_path"), "seal_file").as_posix() for row in tree.get("seal_files", []))
        expected_paths.update(safe_rel(row.get("relative_path"), "extra_file").as_posix() for row in tree.get("extra_files", []))
        archive_paths = {name[len(prefix) + 1:] for name in member_map if name.startswith(prefix + "/")}
        if archive_paths != expected_paths:
            raise ValueError(f"restored_sealed_tree_member_set_mismatch:{tree.get('alias')}")
        for row in rows:
            member = f"{prefix}/{row['relative_path']}"
            expected = member_map.get(member)
            if expected is None or expected[1] != row["sha256"] or (row["bytes"] is not None and expected[0] != row["bytes"]):
                raise ValueError(f"restored_tree_inventory_binding_mismatch:{member}")
        for category in ("seal_files", "extra_files"):
            for selected in tree.get(category, []):
                rel = safe_rel(selected.get("relative_path"), f"restored_{category}").as_posix()
                member = f"{prefix}/{rel}"
                expected = (require_int(selected.get("bytes"), f"restored_{category}_bytes"),
                            require_sha(selected.get("sha256"), f"restored_{category}_sha"))
                if member_map.get(member) != expected:
                    raise ValueError(f"restored_{category}_binding_mismatch:{member}")
        checked_trees += 1
    source_inventory = json.loads((restore_root / "transfer/source_inventory_before.json").read_bytes())
    source_rows = source_inventory.get("source_files")
    if not isinstance(source_rows, list):
        raise ValueError("restored_source_inventory_invalid")
    source_by_member = {row["member"]: row for row in source_rows if isinstance(row, dict)}
    if len(source_by_member) != len(source_rows):
        raise ValueError("restored_source_inventory_row_invalid")
    rows = []
    for member, (size, digest) in member_map.items():
        path = restore_root.joinpath(*PurePosixPath(member).parts)
        recorded = source_by_member.get(member)
        rows.append({"member": member, "bytes": size, "sha256": digest,
                     "source_path": recorded.get("source_path") if recorded else str(path),
                     "data_path": str(path)})
    plan_result = verify_plan_binding(spec, rows)
    for selected in spec.get("files", []):
        member = safe_rel(selected.get("archive_member"), "restored_loose_member").as_posix()
        expected = (require_int(selected.get("bytes"), "restored_loose_bytes"),
                    require_sha(selected.get("sha256"), "restored_loose_sha"))
        if member_map.get(member) != expected:
            raise ValueError(f"restored_loose_file_binding_mismatch:{member}")
    return {"restored_spec_sha256": sha_bytes(spec_raw), "sealed_tree_count": checked_trees,
            "plan_binding": plan_result}


def inventory_rows_from_document(document: Any, inv: dict[str, Any]) -> list[dict[str, Any]]:
    # Inventory parser without a source file. Used only after the inventory
    # file itself has been restored and its hash matched to the archive.
    fmt = inv.get("format")
    path_field, sha_field, bytes_field = inv.get("path_field", "path"), inv.get("sha_field", "sha256"), inv.get("bytes_field", "bytes")
    if fmt == "path_sha256_mapping":
        mapping = json_pointer(document, inv.get("mapping_pointer")) if inv.get("mapping_pointer") else document
        if not isinstance(mapping, dict):
            raise ValueError("restored_inventory_mapping_invalid")
        source = []
        for path, value in mapping.items():
            if isinstance(value, str):
                digest, size = value, None
            elif isinstance(value, dict):
                digest, size = value.get(sha_field), value.get(bytes_field)
            else:
                raise ValueError("restored_inventory_value_invalid")
            source.append((path, digest, size))
    elif fmt == "rows":
        entries = json_pointer(document, inv.get("rows_pointer")) if inv.get("rows_pointer") else document
        if isinstance(entries, dict) and isinstance(inv.get("container_key"), str):
            entries = entries.get(inv["container_key"])
        if not isinstance(entries, list):
            raise ValueError("restored_inventory_rows_invalid")
        source = [(row.get(path_field), row.get(sha_field), row.get(bytes_field)) for row in entries if isinstance(row, dict)]
        if len(source) != len(entries):
            raise ValueError("restored_inventory_row_invalid")
    else:
        raise ValueError("restored_inventory_format_invalid")
    rows = []
    for path, digest, size in source:
        rel = safe_rel(path, "restored_inventory_path").as_posix()
        rows.append({"relative_path": rel, "sha256": require_sha(digest, f"restored_inventory:{rel}"),
                     "bytes": require_int(size, f"restored_inventory_bytes:{rel}") if size is not None else None})
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("restored_inventory_duplicate_path")
    return sorted(rows, key=lambda row: row["relative_path"])


def restore(metadata_dir: Path, chunks_dir: Path, restore_root: Path, receipt_path: Path,
            expected_manifest_sha: str | None = None) -> dict[str, Any]:
    if restore_root.exists() or restore_root.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("refusing_existing_restore_root_or_receipt")
    if (not restore_root.is_absolute() or not receipt_path.is_absolute()
            or ".." in restore_root.parts or ".." in receipt_path.parts
            or receipt_path.resolve(strict=False).is_relative_to(restore_root.resolve(strict=False))):
        raise ValueError("restore_root_and_receipt_must_be_absolute_and_disjoint")
    metadata_dir = safe_abs_dir(str(metadata_dir), "metadata_dir")
    chunks_dir = safe_abs_dir(str(chunks_dir), "chunks_dir")
    safe_abs_dir(str(restore_root.parent), "restore_root_parent")
    safe_abs_dir(str(receipt_path.parent), "restore_receipt_parent")
    if (restore_root.resolve(strict=False).is_relative_to(metadata_dir)
            or restore_root.resolve(strict=False).is_relative_to(chunks_dir)
            or metadata_dir.is_relative_to(restore_root.resolve(strict=False))
            or chunks_dir.is_relative_to(restore_root.resolve(strict=False))):
        raise ValueError("restore_root_must_be_disjoint_from_metadata_and_chunks")
    manifest, pack_receipt, before, after, manifest_sha = verify_transfer_metadata(metadata_dir, expected_manifest_sha=expected_manifest_sha)
    chunk_limit = require_int(manifest.get("chunk_limit_bytes"), "chunk_limit", minimum=1)
    if chunk_limit > DEFAULT_CHUNK_LIMIT:
        raise ValueError("manifest_chunk_limit_exceeds_policy")
    source_rows = before.get("source_files")
    member_rows = manifest.get("archive_member_inventory")
    chunks = manifest.get("chunks")
    if not isinstance(source_rows, list) or not isinstance(member_rows, list) or not isinstance(chunks, list) or not chunks:
        raise ValueError("transfer_member_or_chunk_list_missing")
    if (manifest.get("source_file_count") != len(source_rows)
            or manifest.get("source_bytes") != sum(require_int(row.get("bytes"), "source_bytes") for row in source_rows)
            or manifest.get("archive_member_count") != len(member_rows)
            or manifest.get("archive_member_bytes") != sum(require_int(row.get("bytes"), "archive_member_bytes") for row in member_rows)):
        raise ValueError("transfer_member_count_or_bytes_mismatch")
    member_map: dict[str, tuple[int, str]] = {}
    member_names: list[str] = []
    for row in member_rows:
        if not isinstance(row, dict):
            raise ValueError("archive_member_row_invalid")
        name = safe_rel(row.get("member"), "archive_member").as_posix()
        if name in member_map:
            raise ValueError("duplicate_archive_member")
        member_map[name] = (require_int(row.get("bytes"), f"member_bytes:{name}"), require_sha(row.get("sha256"), f"member_sha:{name}"))
        member_names.append(name)
    if member_names != sorted(member_names):
        raise ValueError("archive_member_inventory_unsorted")
    source_map: dict[str, tuple[int, str]] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError("source_inventory_row_invalid")
        name = safe_rel(row.get("member"), "source_member").as_posix()
        if name in source_map:
            raise ValueError("duplicate_source_inventory_member")
        source_map[name] = (require_int(row.get("bytes"), "source_bytes"), require_sha(row.get("sha256"), "source_sha"))
    expected = dict(source_map)
    # The exact raw before-inventory bytes are also archived, so compare the
    # archive member against the raw downloaded file rather than a re-encode.
    before_raw = (Path(metadata_dir) / "source_inventory_before.json").read_bytes()
    embedded_before = (len(before_raw), sha_bytes(before_raw))
    embedded_provenance = member_map.get("transfer/transfer_provenance.json")
    if member_map.get("transfer/source_inventory_before.json") != embedded_before or embedded_provenance is None:
        raise ValueError("transfer_internal_member_binding_invalid")
    expected["transfer/source_inventory_before.json"] = embedded_before
    expected["transfer/transfer_provenance.json"] = embedded_provenance
    if expected != member_map:
        raise ValueError("archive_members_do_not_match_source_inventory")
    if len(chunks) != manifest.get("chunk_count") or pack_receipt.get("chunk_count") != len(chunks):
        raise ValueError("chunk_count_mismatch")
    chunk_paths: list[Path] = []
    chunk_names: list[str] = []
    total_hash = hashlib.sha256()
    total_bytes = 0
    for i, row in enumerate(chunks):
        name = f"part{i:06d}.bin"
        if row.get("name") != name or row.get("path") != f"{manifest['pack_root']}/payload/chunks/{name}":
            raise ValueError("chunk_order_or_remote_path_mismatch")
        size = require_int(row.get("bytes"), f"chunk_bytes:{name}", minimum=1)
        digest = require_sha(row.get("sha256"), f"chunk_sha:{name}")
        if size > chunk_limit:
            raise ValueError("chunk_oversize")
        path = chunks_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or sha_file(path) != digest:
            raise ValueError(f"downloaded_chunk_hash_mismatch:{name}")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                total_hash.update(block)
                total_bytes += len(block)
        chunk_paths.append(path)
        chunk_names.append(name)
    if chunks_dir.is_symlink() or not chunks_dir.is_dir():
        raise ValueError("chunk_directory_missing_or_unsafe")
    actual_names = []
    for path in chunks_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("chunk_directory_contains_nonregular_member")
        actual_names.append(path.name)
    if sorted(actual_names) != chunk_names:
        raise ValueError("downloaded_chunk_set_mismatch")
    if total_bytes != manifest.get("archive_bytes") or total_hash.hexdigest() != require_sha(manifest.get("archive_sha256"), "archive_sha"):
        raise ValueError("concatenated_archive_hash_or_size_mismatch")
    if manifest.get("archive_readback_verification", {}).get("verified_member_count") != len(member_rows):
        raise ValueError("remote_archive_member_readback_missing")
    safe_abs_dir(str(restore_root.parent), "restore_root_parent")
    restore_root.mkdir(mode=0o700, exist_ok=False)
    restored: list[dict[str, Any]] = []
    observed: list[str] = []
    with ChunkSource(chunk_paths) as source:
        with gzip.GzipFile(fileobj=source, mode="rb") as ungzip:
            with tarfile.open(fileobj=ungzip, mode="r|") as tar:
                for info in tar:
                    name = safe_rel(info.name, "tar_member").as_posix()
                    if not info.isfile() or name not in member_map or name in observed:
                        raise ValueError("archive_has_unexpected_duplicate_or_nonregular_member")
                    size, digest = member_map[name]
                    if info.size != size:
                        raise ValueError(f"tar_member_size_mismatch:{name}")
                    source_file = tar.extractfile(info)
                    if source_file is None:
                        raise ValueError(f"tar_member_unreadable:{name}")
                    with source_file:
                        restored.append(write_restored_member(restore_root, PurePosixPath(name), source_file, size, digest))
                    observed.append(name)
    if observed != member_names:
        raise ValueError("restored_member_order_or_coverage_mismatch")
    restored_map = {row["member"]: (row["bytes"], row["sha256"]) for row in restored}
    if restored_map != member_map:
        raise ValueError("restored_member_inventory_mismatch")
    semantic = audit_restored_spec(restore_root, manifest, member_map)
    receipt = {
        "schema": "TTS_VISUAL_FAMILY_FULL_RESTORE_RECEIPT_V1", "status": "FULL_ARCHIVE_RESTORE_PASS",
        "transfer_id": manifest["transfer_id"], "manifest_sha256": manifest_sha,
        "pack_receipt_sha256": sha_file(Path(metadata_dir) / "pack_receipt.json"),
        "archive_sha256": manifest["archive_sha256"], "archive_bytes": total_bytes,
        "chunk_count": len(chunks), "chunk_limit_bytes": chunk_limit,
        "source_inventory_before_sha256": sha_file(Path(metadata_dir) / "source_inventory_before.json"),
        "source_inventory_after_sha256": sha_file(Path(metadata_dir) / "source_inventory_after.json"),
        "source_inventory_identical": True, "source_file_count": len(source_rows),
        "archive_member_count": len(member_rows), "archive_member_bytes": sum(size for size, _ in member_map.values()),
        "restored_member_count": len(restored), "restored_bytes": sum(row["bytes"] for row in restored),
        "restore_root": str(restore_root.resolve()), "semantic_rechecks": semantic,
        "model_bulk_included": False, "receipt_outside_restore_root": not receipt_path.resolve(strict=False).is_relative_to(restore_root.resolve()),
    }
    if not receipt["receipt_outside_restore_root"]:
        raise ValueError("restore_receipt_must_be_outside_restore_tree")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    size, digest = write_json_new(receipt_path, receipt)
    return {**receipt, "receipt_bytes": size, "receipt_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack_p = sub.add_parser("pack", help="create and fully verify remote archive chunks")
    pack_p.add_argument("--spec", type=Path, required=True)
    pack_p.add_argument("--expected-script-sha256", required=True)
    pack_p.add_argument("--script-path", type=Path, default=Path(__file__))
    dl_p = sub.add_parser("prepare-downloads", help="create fresh bridge requests for every verified archive chunk")
    dl_p.add_argument("--metadata-dir", type=Path, required=True)
    dl_p.add_argument("--chunks-dir", type=Path, required=True)
    dl_p.add_argument("--request-dir", type=Path, required=True)
    dl_p.add_argument("--expected-manifest-sha256")
    restore_p = sub.add_parser("restore", help="verify all chunks and restore/hash every member")
    restore_p.add_argument("--metadata-dir", type=Path, required=True)
    restore_p.add_argument("--chunks-dir", type=Path, required=True)
    restore_p.add_argument("--restore-root", type=Path, required=True)
    restore_p.add_argument("--receipt", type=Path, required=True)
    restore_p.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            result = package(args.spec, args.expected_script_sha256, args.script_path)
        elif args.command == "prepare-downloads":
            result = prepare_downloads(args.metadata_dir, args.chunks_dir, args.request_dir, args.expected_manifest_sha256)
        else:
            result = restore(args.metadata_dir, args.chunks_dir, args.restore_root, args.receipt, args.expected_manifest_sha256)
    except Exception as exc:
        print(json.dumps({"status": "TRANSFER_FAILED", "command": args.command,
                          "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
