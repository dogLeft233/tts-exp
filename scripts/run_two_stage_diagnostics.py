#!/usr/bin/env python3
"""Validate and materialize deterministic two-stage diagnostic manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from prepare_two_stage_hubert_manifest import (
    STAGE_CONTRACT,
    file_sha256,
    load_two_stage_manifest,
)

REMOTE_REPO_ROOT = Path("/mnt/e/Documents/tts-audio/tts-exp")
PATH_FIELDS = (
    "natural_path",
    "tts_path",
    "matched_span_metadata_path",
)
DEFAULT_KS = (1, 4, 16)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("diagnostic manifest must be an object")
    return payload


def _localize_path(
    value: Any,
    *,
    source_manifest: Path,
    repo_root: Path,
    field: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = source_manifest.parent / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(REMOTE_REPO_ROOT)
    except ValueError:
        localized = candidate
    else:
        localized = (repo_root / relative).resolve()
    if not localized.is_file():
        raise FileNotFoundError(f"{field} is unavailable after local path mapping: {localized}")
    return str(localized)


def localize_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    source_manifest = Path(manifest_path).resolve()
    root = Path(repo_root).resolve()
    payload = _strict_json(source_manifest)
    if payload.get("contract") != STAGE_CONTRACT:
        raise ValueError("diagnostic manifest has an incompatible contract")
    localized = copy.deepcopy(payload)
    parent = localized.get("source_target_manifest")
    if not isinstance(parent, Mapping):
        raise ValueError("source_target_manifest is malformed")
    parent["path"] = _localize_path(
        parent.get("path"),
        source_manifest=source_manifest,
        repo_root=root,
        field="source_target_manifest.path",
    )
    records = localized.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("diagnostic manifest requires records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        for field in PATH_FIELDS:
            record[field] = _localize_path(
                record.get(field),
                source_manifest=source_manifest,
                repo_root=root,
                field=f"record[{index}].{field}",
            )
    return localized


def _split_boundary(records: Sequence[Mapping[str, Any]], excluded: Mapping[str, Any]) -> dict[str, Any]:
    split_speakers: dict[str, set[str]] = defaultdict(set)
    paired_keys: set[str] = set()
    for record in records:
        key = str(record["paired_key"])
        if key in paired_keys:
            raise ValueError(f"duplicate paired_key: {key}")
        paired_keys.add(key)
        split_speakers[str(record["split"])].add(str(record["speaker_group"]))
    heldout = excluded.get("heldout")
    if not isinstance(heldout, list):
        raise ValueError("heldout provenance must be a list")
    heldout_speakers = {str(row["speaker_group"]) for row in heldout if isinstance(row, Mapping)}
    if not heldout_speakers:
        raise ValueError("heldout provenance has no speakers")
    stage_speakers = set().union(*split_speakers.values())
    if heldout_speakers & stage_speakers:
        raise ValueError("heldout speaker crosses the train/valid boundary")
    if split_speakers.get("train", set()) & split_speakers.get("valid", set()):
        raise ValueError("train speaker crosses the valid boundary")
    return {
        "counts": {
            split: sum(1 for record in records if str(record["split"]) == split)
            for split in ("train", "valid")
        },
        "speakers_by_split": {
            split: sorted(values) for split, values in sorted(split_speakers.items())
        },
        "heldout_speakers": sorted(heldout_speakers),
        "paired_key_count": len(paired_keys),
    }


def _round_robin_train_subset(records: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("subset size must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["speaker_group"])].append(dict(record))
    speakers = sorted(buckets)
    for rows in buckets.values():
        rows.sort(key=lambda row: str(row["paired_key"]))
    if count > len(records):
        raise ValueError(f"subset size {count} exceeds train record count {len(records)}")
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        progressed = False
        for speaker in speakers:
            rows = buckets[speaker]
            if cursor < len(rows):
                selected.append(rows[cursor])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise AssertionError("round-robin subset selection stalled")
        cursor += 1
    return selected


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return file_sha256(path)


def build_diagnostic_manifests(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    source_path = Path(manifest_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent
    output = Path(output_dir).resolve()
    if not ks or any(int(value) <= 0 for value in ks):
        raise ValueError("ks must contain positive integers")
    requested_ks = tuple(dict.fromkeys(int(value) for value in ks))

    localized_payload = localize_manifest(source_path, repo_root=root)
    local_full_path = output / "localized_stage_manifest.json"
    local_manifest_sha = _write_json(local_full_path, localized_payload)
    records = load_two_stage_manifest(local_full_path)
    boundary = _split_boundary(records, localized_payload["excluded_provenance"])
    train_records = [record for record in records if record["split"] == "train"]
    valid_records = [record for record in records if record["split"] == "valid"]

    subsets: dict[str, Any] = {}
    for count in requested_ks:
        selected = _round_robin_train_subset(train_records, count)
        subset_payload = copy.deepcopy(localized_payload)
        subset_payload["records"] = selected + [dict(record) for record in valid_records]
        subset_path = output / f"subset_k{count}.json"
        subset_sha = _write_json(subset_path, subset_payload)
        checked = load_two_stage_manifest(subset_path)
        subset_train = [record for record in checked if record["split"] == "train"]
        subsets[str(count)] = {
            "path": str(subset_path),
            "sha256": subset_sha,
            "train_count": len(subset_train),
            "valid_count": sum(record["split"] == "valid" for record in checked),
            "paired_keys": [str(record["paired_key"]) for record in subset_train],
            "speakers": sorted({str(record["speaker_group"]) for record in subset_train}),
            "natural_sha256": [str(record["natural_sha256"]) for record in subset_train],
            "tts_sha256": [str(record["tts_sha256"]) for record in subset_train],
            "matched_span_sha256": [str(record["matched_spans_sha256"]) for record in subset_train],
        }

    report = {
        "schema_version": 1,
        "diagnostic_type": "two_stage_overfit_manifest_preflight",
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "localized_manifest": str(local_full_path),
        "localized_manifest_sha256": local_manifest_sha,
        "repo_root": str(root),
        "contract": STAGE_CONTRACT,
        "boundary": boundary,
        "subsets": subsets,
        "heldout_loaded": False,
    }
    _write_json(output / "preflight.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--k", type=int, action="append", dest="ks")
    args = parser.parse_args(argv)
    report = build_diagnostic_manifests(
        args.manifest,
        args.output_dir,
        repo_root=args.repo_root,
        ks=args.ks or DEFAULT_KS,
    )
    print(
        json.dumps(
            {
                "diagnostic_type": report["diagnostic_type"],
                "localized_manifest": report["localized_manifest"],
                "subsets": {
                    key: {
                        "train_count": value["train_count"],
                        "valid_count": value["valid_count"],
                        "sha256": value["sha256"],
                    }
                    for key, value in report["subsets"].items()
                },
                "heldout_loaded": report["heldout_loaded"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
