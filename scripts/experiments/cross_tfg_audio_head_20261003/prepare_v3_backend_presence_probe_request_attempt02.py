#!/usr/bin/env python3
"""Create a root-dispatched, read-only current-host backend path/hash probe.

This packet embeds the frozen literal-path pins but runs nowhere by itself.
Its remote command only stats those files and known directories, hashes listed
regular files within a strict aggregate byte budget, and reads local Git HEADs.
It does not invoke Python model code, render, score, touch media, or write on
the inspected host.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
INPUT_REL = Path("runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_restore_identity_manifest_attempt02.json")
INPUT_RAW_SHA256 = "05df89942010960e3ddfc0a84d27eac7a447c7bfe14c78c684bd136d12dc8c1d"
OUT_DIR_REL = Path("runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_presence_probe_attempt02/requests_attempt02")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


REMOTE_PROBE = r'''import base64, hashlib, json, os, platform, stat, subprocess, time
PINS = json.loads(base64.b64decode("__PIN_B64__").decode("ascii"))
DIRECTORIES = json.loads(base64.b64decode("__DIRECTORY_B64__").decode("ascii"))
MAX_HASH_BYTES = 10 * 1024**3 - 1

def path_kind(mode):
    if stat.S_ISREG(mode): return 'regular_file'
    if stat.S_ISDIR(mode): return 'directory'
    if stat.S_ISLNK(mode): return 'symlink'
    return 'other'

observed = []
hash_candidates = []
for pin in PINS:
    path = pin['legacy_literal_path']
    try:
        link_st = os.lstat(path)
    except OSError as exc:
        observed.append({**pin, 'presence': 'MISSING', 'error_type': type(exc).__name__})
        continue
    try:
        target_st = os.stat(path)
    except OSError as exc:
        observed.append({**pin, 'presence': 'BROKEN_OR_UNREADABLE_LINK',
                         'lstat_kind': path_kind(link_st.st_mode), 'error_type': type(exc).__name__})
        continue
    row = {**pin, 'presence': 'PRESENT', 'lstat_kind': path_kind(link_st.st_mode),
           'resolved_path': os.path.realpath(path), 'target_kind': path_kind(target_st.st_mode),
           'actual_bytes': target_st.st_size if stat.S_ISREG(target_st.st_mode) else None,
           'device': target_st.st_dev, 'inode': target_st.st_ino}
    if not stat.S_ISREG(target_st.st_mode):
        row['hash_status'] = 'NOT_A_REGULAR_FILE'
    else:
        row['hash_status'] = 'PENDING_BUDGET'
        hash_candidates.append((target_st.st_size, target_st.st_dev, target_st.st_ino, row, path))
    observed.append(row)

directory_rows = []
for entry in DIRECTORIES:
    path = entry['legacy_literal_path']
    try:
        lst = os.lstat(path)
        target = os.stat(path)
        directory_rows.append({**entry, 'presence': 'PRESENT', 'lstat_kind': path_kind(lst.st_mode),
                               'resolved_path': os.path.realpath(path), 'target_kind': path_kind(target.st_mode),
                               'device': target.st_dev, 'inode': target.st_ino})
    except OSError as exc:
        directory_rows.append({**entry, 'presence': 'MISSING_OR_UNREADABLE', 'error_type': type(exc).__name__})

hash_candidates.sort(key=lambda item: (-item[0], item[3]['backend_id'], item[4]))
hashed_by_inode = {}
hashed_bytes = 0
for size, dev, ino, row, path in hash_candidates:
    key = (dev, ino)
    if key in hashed_by_inode:
        value = hashed_by_inode[key]['sha256']
        row['current_sha256'] = value
        row['hashed_bytes'] = 0
        row['sha_matches'] = value == row['expected_sha256']
        row['size_matches'] = row['expected_bytes'] is None or row['actual_bytes'] == row['expected_bytes']
        row['hash_status'] = 'HASHED_SHARED_INODE_MATCH' if row['sha_matches'] and row['size_matches'] else 'HASHED_SHARED_INODE_MISMATCH'
        continue
    if size < 0 or hashed_bytes + size > MAX_HASH_BYTES:
        row['hash_status'] = 'SKIPPED_10_GIB_HASH_BUDGET'
        continue
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                read_bytes += len(block)
                if read_bytes > MAX_HASH_BYTES or hashed_bytes + read_bytes > MAX_HASH_BYTES:
                    raise RuntimeError('hash byte budget exceeded during read')
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = os.stat(path)
        before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if read_bytes != size or before_key != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or before_key != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            row['hash_status'] = 'CHANGED_DURING_HASH'
            row['hashed_bytes'] = read_bytes
            hashed_bytes += read_bytes
            continue
        value = digest.hexdigest()
        hashed_bytes += read_bytes
        hashed_by_inode[key] = {'sha256': value}
        row['current_sha256'] = value
        row['hashed_bytes'] = read_bytes
        row['sha_matches'] = value == row['expected_sha256']
        row['size_matches'] = row['expected_bytes'] is None or row['actual_bytes'] == row['expected_bytes']
        if not row['size_matches']:
            row['hash_status'] = 'SIZE_AND_SHA_MATCH' if row['sha_matches'] else 'SIZE_AND_SHA_MISMATCH'
        else:
            row['hash_status'] = 'MATCH' if row['sha_matches'] else 'SHA_MISMATCH'
    except Exception as exc:
        row['hash_status'] = 'HASH_ERROR_' + type(exc).__name__
        row['hashed_bytes'] = read_bytes
        hashed_bytes += read_bytes

git_rows = []
for entry in DIRECTORIES:
    path = entry['legacy_literal_path']
    if entry['role'] not in ('backend_repository', 'model_source_root', 'project_repository'):
        continue
    try:
        proc = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, timeout=5, check=False)
        head = proc.stdout.strip() if proc.returncode == 0 else None
        git_rows.append({'path': path, 'status': 'PASS' if head else 'NOT_GIT_REPO_OR_UNAVAILABLE', 'head': head})
    except Exception as exc:
        git_rows.append({'path': path, 'status': 'ERROR_' + type(exc).__name__, 'head': None})

out = {'schema': 'cross_tfg_v3_samehost_backend_presence_probe_v1',
       'captured_unix': int(time.time()),
       'host': {'hostname': platform.node(), 'platform': platform.platform(), 'cpu_count': os.cpu_count()},
       'file_hash_budget_bytes': MAX_HASH_BYTES, 'hashed_file_bytes': hashed_bytes,
       'file_count': len(observed), 'present_count': sum(x['presence'] == 'PRESENT' for x in observed),
       'missing_count': sum(x['presence'] == 'MISSING' for x in observed),
       'files': observed, 'directories': directory_rows, 'git_heads': git_rows,
       'scope': {'model_or_scorer_forward': False, 'renderer_or_scorer_invoked': False,
                 'development_or_confirmation_media_opened': False,
                 'file_contents_emitted': False, 'file_contents_read_only_for_sha256': True,
                 'filesystem_mutation': False, 'network_or_cloud_operation': False,
                 'environment_values_or_credentials_emitted': False}}
print(json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUT_DIR_REL)
    args = parser.parse_args()
    raw = INPUT_REL
    if not raw.is_absolute():
        raw = ROOT / raw
    inventory_raw = raw.read_bytes()
    if sha_bytes(inventory_raw) != INPUT_RAW_SHA256:
        raise ValueError("backend identity manifest raw SHA changed")
    inventory = json.loads(inventory_raw.decode("utf-8"))
    if (inventory.get("schema") != "cross_tfg_v3_samehost_backend_restore_identity_manifest_v2"
            or inventory.get("manifest_sha256") != payload_sha(inventory, "manifest_sha256")
            or inventory.get("status") != "FROZEN_IDENTITIES_AND_LEGACY_PATHS_ENUMERATED_CURRENT_HOST_PRESENCE_NOT_VERIFIED"):
        raise ValueError("backend identity manifest is not the reviewed sealed attempt02 inventory")
    pins = []
    for backend in inventory["backends"]:
        for file in backend["files"]:
            pins.append({
                "backend_id": backend["model_id"], "role": file["role"],
                "logical_path": file["logical_path"], "legacy_literal_path": file["legacy_literal_path"],
                "expected_bytes": file["expected_bytes"], "expected_sha256": file["expected_sha256"],
            })
    pins.sort(key=lambda item: (item["backend_id"], item["legacy_literal_path"], item["role"]))
    directories = []
    seen_dirs: set[tuple[str, str]] = set()
    for backend in inventory["backends"]:
        for entry in backend.get("directories", []):
            key = (entry["role"], entry["legacy_literal_path"])
            if key not in seen_dirs:
                directories.append({"backend_id": backend["model_id"], "role": entry["role"],
                                    "legacy_literal_path": entry["legacy_literal_path"]})
                seen_dirs.add(key)
    # Include the project repository parent for one local Git HEAD observation.
    project_path = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003"
    directories.append({"backend_id": "shared", "role": "project_repository", "legacy_literal_path": project_path})
    pins_text = json.dumps(pins, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    directories_text = json.dumps(directories, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    pins_b64 = base64.b64encode(pins_text).decode("ascii")
    directories_b64 = base64.b64encode(directories_text).decode("ascii")
    command = ("set -euo pipefail\npython3 - <<'PYCODE'\n" +
               REMOTE_PROBE.replace("__PIN_B64__", pins_b64).replace("__DIRECTORY_B64__", directories_b64) +
               "\nPYCODE")
    request = {
        "op": "exec",
        "packet_attempt": "samehost_dev360_backend_presence_probe_attempt02",
        "timeout": 120,
        "read_only_contract": {
            "stat_exact_frozen_backend_paths_only": True,
            "hash_existing_expected_files_only": True,
            "maximum_aggregate_hashed_bytes": 10 * 1024**3 - 1,
            "source_repository_git_heads_only": True,
            "model_or_scorer_forward": False,
            "development_or_confirmation_media_opened": False,
            "renderer_or_scorer_invoked": False,
            "filesystem_mutation": False,
            "environment_values_or_credentials_emitted": False,
            "file_contents_emitted": False,
        },
        "inventory_input": {
            "path": INPUT_REL.as_posix(), "bytes": len(inventory_raw),
            "raw_sha256": INPUT_RAW_SHA256, "payload_sha256": inventory["manifest_sha256"],
            "file_pin_count": len(pins),
        },
        "command": command,
    }
    request_raw = pretty(request)
    request_path = args.output_dir / "01_read_backend_presence_attempt02_request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(request_raw)
    index = {
        "schema": "cross_tfg_backend_probe_request_index_v1",
        "packet_attempt": "samehost_dev360_backend_presence_probe_attempt02",
        "requests": [{"index": 0, "name": "read current-host backend path presence and exact pins",
                      "path": request_path.relative_to(ROOT).as_posix(),
                      "bytes": len(request_raw), "raw_sha256": sha_bytes(request_raw),
                      "op": "exec", "remote_execution_authorized_by_root": False}],
        "source_inventory": {"path": INPUT_REL.as_posix(), "raw_sha256": INPUT_RAW_SHA256,
                             "payload_sha256": inventory["manifest_sha256"], "file_pin_count": len(pins)},
    }
    index["index_sha256"] = payload_sha(index, "index_sha256")
    index_path = args.output_dir / "request_index_attempt02.json"
    index_raw = pretty(index)
    index_path.write_bytes(index_raw)
    print(json.dumps({"request_path": str(request_path.resolve()), "request_bytes": len(request_raw),
                      "request_raw_sha256": sha_bytes(request_raw), "index_path": str(index_path.resolve()),
                      "index_bytes": len(index_raw), "index_raw_sha256": sha_bytes(index_raw),
                      "index_payload_sha256": index["index_sha256"], "request_index": 0,
                      "backend_file_pins": len(pins), "known_directories": len(directories),
                      "max_hashed_bytes": 10 * 1024**3 - 1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
