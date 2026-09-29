from __future__ import annotations

import random
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    DiagnosticError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def create_review_package(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], paths) -> dict[str, Any]:
    paths.review.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["sample_id"]): row for row in media_manifest.get("rows", [])}
    records = list(protocol.get("records", []))[: config.BLIND_RECORD_COUNT]
    if len(records) != config.BLIND_RECORD_COUNT:
        raise DiagnosticError("review cohort has fewer than four records")
    rng = random.Random(config.BLIND_SEED)
    entries: list[dict[str, Any]] = []
    key: list[dict[str, str]] = []
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        media_row = by_id.get(sample_id)
        if media_row is None:
            raise DiagnosticError(f"review media row is missing: {sample_id}")
        order = [config.REAL_ARM, config.WARP_ARM]
        rng.shuffle(order)
        pair_id = f"pair_{index:02d}"
        display_files: list[str] = []
        pair_key: dict[str, str] = {"pair_id": pair_id, "sample_id": sample_id}
        for label, arm in zip(("A", "B"), order, strict=True):
            source = Path(str(media_row["arms"][arm]["output"]))
            destination = paths.review / f"{pair_id}_{label}.mkv"
            if destination.is_file():
                if file_sha256(destination) != file_sha256(source):
                    raise DiagnosticError(f"existing blind media differs: {destination}")
            else:
                shutil.copy2(source, destination)
            display_files.append(str(destination))
            pair_key[label] = arm
        entries.append({"pair_id": pair_id, "sample_id": sample_id, "files": display_files})
        key.append(pair_key)
    form = """# Blind review form

For each pair, listen/watch with the same volume and playback settings.

| pair | which is more synchronized (A / B / indistinguishable) | obvious stutter (yes / no) | notes |
|---|---|---|---|
"""
    form_path = paths.review / "form.md"
    if form_path.exists() and form_path.read_text(encoding="utf-8") != form:
        raise DiagnosticError("existing blind review form differs")
    form_path.write_text(form, encoding="utf-8")
    key_payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "seed": config.BLIND_SEED,
        "mapping": key,
        "warning": "This key is separate from the display files; do not expose it to reviewers.",
    }
    key_path = paths.review / "key.json"
    if key_path.is_file():
        prior = verify_self_hashed_json(key_path)
        body = dict(prior)
        body.pop("artifact_sha256", None)
        if body != key_payload:
            raise DiagnosticError("existing blind key differs")
    else:
        write_self_hashed_json(key_path, key_payload)
    manifest = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "PENDING",
        "seed": config.BLIND_SEED,
        "record_count": len(entries),
        "entries": entries,
        "form": str(form_path),
        "key": str(key_path),
        "reviewer_responses": None,
    }
    manifest_path = paths.review / "manifest.json"
    if manifest_path.is_file():
        prior = verify_self_hashed_json(manifest_path)
        body = dict(prior)
        body.pop("artifact_sha256", None)
        if body != manifest:
            raise DiagnosticError("existing blind review manifest differs")
    else:
        write_self_hashed_json(manifest_path, manifest)
    return {"status": "PENDING", "manifest_sha256": file_sha256(manifest_path), "manifest": manifest}
