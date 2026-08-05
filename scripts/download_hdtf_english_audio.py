#!/usr/bin/env python3
"""Download HDTF audio from Synology FileStation without exposing credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NAS_HOST = "10.108.21.182"
ACCOUNT = "dhg_mm"
AUTH_ENDPOINT = f"https://{NAS_HOST}:5001/webapi/auth.cgi"
ENTRY_ENDPOINT = f"https://{NAS_HOST}:5001/webapi/entry.cgi"
DEFAULT_REMOTE_ROOT = "/dhg_mm/HDTF/_audio_raw"
DEFAULT_OUTPUT = ROOT / "data" / "hdtf_audio_en"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SSL_CONTEXT = ssl._create_unverified_context()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_request(endpoint: str, params: dict[str, Any], method: str = "GET") -> dict[str, Any]:
    encoded = urllib.parse.urlencode({key: str(value) for key, value in params.items()})
    if method == "POST":
        request = urllib.request.Request(
            endpoint,
            data=encoded.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
    else:
        request = urllib.request.Request(f"{endpoint}?{encoded}")
    with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=30) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise RuntimeError(f"Synology API request failed: {payload}")
    return payload


def login(password: str) -> tuple[str, str]:
    payload = json_request(
        AUTH_ENDPOINT,
        {
            "api": "SYNO.API.Auth",
            "version": "6",
            "method": "login",
            "session": "FileStation",
            "format": "sid",
            "enable_syno_token": "yes",
            "account": ACCOUNT,
            "passwd": password,
        },
        method="POST",
    )
    data = payload["data"]
    return str(data["sid"]), str(data["synotoken"])


def list_folder(folder: str, sid: str, token: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = json_request(
            ENTRY_ENDPOINT,
            {
                "api": "SYNO.FileStation.List",
                "version": "2",
                "method": "list",
                "folder_path": folder,
                "offset": offset,
                "limit": 1000,
                "sort_by": "name",
                "sort_direction": "asc",
                "filetype": "all",
                "SynoToken": token,
                "_sid": sid,
            },
        )
        data = payload.get("data", {})
        batch = data.get("files", [])
        entries.extend(batch)
        total = int(data.get("total", len(entries)))
        offset += len(batch)
        if not batch or offset >= total:
            return entries


def is_directory(entry: dict[str, Any]) -> bool:
    return bool(entry.get("isdir")) or entry.get("type") in {"dir", "folder"}


def enumerate_audio(remote_root: str, sid: str, token: str) -> list[dict[str, Any]]:
    pending = [remote_root]
    audio: list[dict[str, Any]] = []
    while pending:
        folder = pending.pop()
        for entry in list_folder(folder, sid, token):
            path = str(entry.get("path") or f"{folder}/{entry['name']}")
            if is_directory(entry):
                pending.append(path)
            elif PurePosixPath(path).suffix.lower() in AUDIO_SUFFIXES:
                audio.append({**entry, "path": path})
    return sorted(audio, key=lambda entry: str(entry["path"]))


def download_file(remote_path: str, destination: Path, sid: str, token: str) -> None:
    params = urllib.parse.urlencode(
        {
            "api": "SYNO.FileStation.Download",
            "version": "2",
            "method": "download",
            "path": remote_path,
            "mode": "download",
            "SynoToken": token,
            "_sid": sid,
        }
    )
    request = urllib.request.Request(
        f"{ENTRY_ENDPOINT}?{params}",
        headers={"User-Agent": "tts-exp-hdtf-acquisition/1.0"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(
                request, context=SSL_CONTEXT, timeout=180
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type.lower():
                    raise RuntimeError(response.read().decode("utf-8", "replace"))
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
            temporary.flush()
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List remote audio files without downloading them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("NAS_DHG_MM_PASSWORD")
    if not password:
        raise SystemExit("Set NAS_DHG_MM_PASSWORD in the current shell")
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")

    sid, token = login(password)
    remote_files = enumerate_audio(args.remote_root, sid, token)
    if args.limit:
        remote_files = remote_files[: args.limit]
    print(f"Found {len(remote_files)} audio files under {args.remote_root}")
    if args.dry_run:
        for entry in remote_files:
            print(entry["path"])
        return 0

    audio_root = args.output_dir / "audio"
    records: list[dict[str, Any]] = []
    remote_root_path = PurePosixPath(args.remote_root)
    for entry in remote_files:
        remote_path = PurePosixPath(str(entry["path"]))
        relative = remote_path.relative_to(remote_root_path)
        destination = audio_root.joinpath(*relative.parts)
        if not destination.exists():
            download_file(str(remote_path), destination, sid, token)
        record = {
            "remote_path": str(remote_path),
            "local_path": str(destination.relative_to(ROOT)),
            "remote_size_bytes": entry.get("size"),
            "local_size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        records.append(record)
        print(f"Downloaded {record['local_path']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "nas_host": NAS_HOST,
        "nas_account": ACCOUNT,
        "remote_root": args.remote_root,
        "audio_role": "HDTF_English_source_audio",
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
