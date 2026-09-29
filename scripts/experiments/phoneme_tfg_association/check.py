"""Independent artifact and denominator checker for the association run."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .protocol import PAIR_TYPES, PROTOCOL_ID, ProtocolError, file_sha256, read_json, read_jsonl, validate_receipt, write_json


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _contains_lrs3(path: str | Path) -> bool:
    return any("lrs3" in part.lower() for part in Path(str(path)).resolve().parts)


def validate_run(run_dir: Path, *, require_media: bool = False, require_scores: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    protocol_path = run_dir / "00_protocol/protocol.json"
    if not protocol_path.is_file():
        return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "FAIL", "engineering_status": "BLOCKED", "errors": [f"missing {protocol_path}"]}
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        _error(errors, "protocol_id mismatch")
    blocks = protocol.get("blocks", [])
    wav2lip_cells = protocol.get("cells", [])
    ditto_cells = protocol.get("ditto_cells", [])
    cells = [*wav2lip_cells, *ditto_cells]
    keys = [str(cell.get("cell_key")) for cell in cells]
    if len(keys) != len(set(keys)):
        _error(errors, "duplicate cell_key in frozen protocol")
    if len(wav2lip_cells) != int(protocol.get("video_budget", {}).get("planned_unique_wav2lip_videos", -1)):
        _error(errors, "frozen Wav2Lip cell budget does not match cell count")
    if len(ditto_cells) != int(protocol.get("video_budget", {}).get("planned_unique_ditto_videos", 0)):
        _error(errors, "frozen Ditto cell budget does not match cell count")
    pair_counts = Counter("|".join(str(item) for item in block.get("pair", [])) for block in blocks)
    expected_pairs = {"|".join(pair): int(protocol.get("config", {}).get("pair_replicates", 6)) for pair in PAIR_TYPES}
    if not bool(protocol.get("smoke")) and dict(pair_counts) != expected_pairs:
        _error(errors, f"pair counts mismatch: {dict(pair_counts)} != {expected_pairs}")

    for block in blocks:
        reference = block.get("reference")
        if not isinstance(reference, Mapping) or "visual_source" not in reference:
            continue  # legacy/minimal checker fixtures have no visual binding
        visual_source = str(reference.get("visual_source", ""))
        if not visual_source or _contains_lrs3(visual_source):
            _error(errors, f"forbidden or missing external visual source: {block.get('sample_id')}")
        else:
            visual_path = Path(visual_source)
            expected_visual_sha = str(reference.get("visual_source_sha256", ""))
            if not visual_path.is_file():
                _error(errors, f"external visual source is missing: {block.get('sample_id')}")
            elif not expected_visual_sha or file_sha256(visual_path) != expected_visual_sha:
                _error(errors, f"external visual source hash mismatch: {block.get('sample_id')}")
        if reference.get("visual_source_frame_policy") != "first_frame_only":
            _error(errors, f"visual source is not first-frame-only: {block.get('sample_id')}")
        if reference.get("visual_source_used_for_generation") != "one_frame_repeated_for_each_mel_chunk":
            _error(errors, f"visual generation policy is not frozen-frame-only: {block.get('sample_id')}")

    generation_receipts: list[dict[str, Any]] = []
    missing_media: list[str] = []
    for cell in cells:
        receipt_path = run_dir / "03_video" / str(cell.get("tfg")) / str(cell.get("sample_id")) / f"{cell.get('arm')}.receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            generation_receipts.append(receipt)
            try:
                validate_receipt(receipt, cell)
            except ProtocolError as exc:
                _error(errors, f"receipt identity mismatch: {cell.get('cell_key')}: {exc}")
            media_path = Path(str(receipt.get("output", ""))) if receipt.get("output") else run_dir / "03_video" / str(cell.get("tfg")) / str(cell.get("sample_id")) / f"{cell.get('arm')}.mkv"
            if receipt.get("status") != "complete":
                missing_media.append(str(cell.get("cell_key")))
            elif not media_path.is_file() or receipt.get("output_sha256") != file_sha256(media_path):
                _error(errors, f"complete receipt/media mismatch: {cell.get('cell_key')}")
            reference = cell.get("reference")
            if isinstance(reference, Mapping) and "visual_source" in reference and receipt.get("status") == "complete":
                visual_source = str(receipt.get("visual_source", ""))
                expected_visual = str(reference.get("visual_source", ""))
                if not visual_source or _contains_lrs3(visual_source):
                    _error(errors, f"receipt uses forbidden/missing visual source: {cell.get('cell_key')}")
                if visual_source != expected_visual:
                    _error(errors, f"receipt visual source binding changed: {cell.get('cell_key')}")
                if receipt.get("visual_source_sha256") != reference.get("visual_source_sha256"):
                    _error(errors, f"receipt visual source hash binding changed: {cell.get('cell_key')}")
                if visual_source and Path(visual_source).is_file() and file_sha256(Path(visual_source)) != receipt.get("visual_source_sha256"):
                    _error(errors, f"receipt external visual source hash mismatch: {cell.get('cell_key')}")
                if receipt.get("visual_source_frame_policy") != "first_frame_only" or receipt.get("fixed_reference_frame_repeated") is not True:
                    _error(errors, f"receipt does not prove frozen external frame input: {cell.get('cell_key')}")
        else:
            missing_media.append(str(cell.get("cell_key")))
    if require_media and missing_media:
        _error(errors, f"missing media receipts: {len(missing_media)}")
    elif missing_media:
        warnings.append(f"media not generated: {len(missing_media)} cells")

    score_path = run_dir / "04_syncnet/scores.jsonl"
    score_rows = read_jsonl(score_path)
    score_keys: set[tuple[str, str]] = set()
    expected_score_keys = {
        (str(cell.get("sample_id")), str(cell.get("tfg")))
        for cell in cells
    }
    expected_score_keys = {
        key for key in expected_score_keys
        if sum(1 for cell in cells if (str(cell.get("sample_id")), str(cell.get("tfg"))) == key) >= 3
    }
    if require_scores and not score_rows:
        _error(errors, "required scores.jsonl is missing or empty")
    elif not score_rows:
        warnings.append("scores.jsonl is not present")
    for row in score_rows:
        score_key = (str(row.get("sample_id")), str(row.get("tfg", "wav2lip")))
        if score_key in score_keys:
            _error(errors, f"duplicate score row: {score_key}")
        score_keys.add(score_key)
        endpoints = row.get("endpoints", [])
        if not isinstance(endpoints, list):
            _error(errors, f"score row endpoints is not a list: {row.get('sample_id')}")
            continue
        for endpoint in endpoints:
            if endpoint.get("protocol_id") not in {None, PROTOCOL_ID}:
                _error(errors, f"score endpoint protocol mismatch: {endpoint.get('sample_id')}")
            for field in ("sync_c", "sync_d", "background_b"):
                if endpoint.get(field) is not None and not math.isfinite(float(endpoint[field])):
                    _error(errors, f"non-finite score {field}: {endpoint.get('sample_id')}")
            curve = endpoint.get("curve")
            if curve is not None and (len(curve) != 31 or not np.isfinite(np.asarray(curve, dtype=np.float64)).all()):
                _error(errors, f"invalid score curve: {endpoint.get('sample_id')}")
        by_condition_support = {(str(endpoint.get("condition")), str(endpoint.get("support"))): endpoint for endpoint in endpoints}
        for support in ("FULL", "INTERIOR", "EQUAL_COUNT"):
            natural = by_condition_support.get(("natural", support))
            if natural is None:
                continue
            for condition in ("qwen_cloud", "qwen_local", "index_tts2", "cosyvoice2"):
                tts = by_condition_support.get((condition, support))
                if tts is None:
                    continue
                values = (natural.get("sync_c"), tts.get("sync_c"), natural.get("sync_d"), tts.get("sync_d"), natural.get("background_b"), tts.get("background_b"))
                if all(value is not None and math.isfinite(float(value)) for value in values):
                    lhs = float(tts["sync_c"]) - float(natural["sync_c"])
                    rhs = float(natural["sync_d"]) - float(tts["sync_d"]) + float(tts["background_b"]) - float(natural["background_b"])
                    if abs(lhs - rhs) > 1e-4:
                        _error(errors, f"C decomposition mismatch: {row.get('sample_id')}/{support}/{condition}")
    if require_scores and expected_score_keys - score_keys:
        _error(errors, f"missing score rows: {sorted(expected_score_keys - score_keys)}")
    elif expected_score_keys - score_keys:
        warnings.append(f"score rows not present: {len(expected_score_keys - score_keys)}")

    validation = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not errors else "FAIL",
        "engineering_status": "READY" if not errors else "BLOCKED",
        "independent": True,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "source_groups": len({str(block.get("source_group")) for block in blocks}),
            "blocks": len(blocks),
            "expected_cells": len(cells),
            "receipt_count": len(generation_receipts),
            "complete_receipt_count": sum(row.get("status") == "complete" for row in generation_receipts),
            "missing_media_count": len(missing_media),
            "score_rows": len(score_rows),
            "score_keys": len(score_keys),
        },
        "missing_media": missing_media,
        "score_path": str(score_path),
    }
    write_json(run_dir / "validation.json", validation)
    return validation


__all__ = ["validate_run"]
