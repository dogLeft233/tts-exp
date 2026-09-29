from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import _matrix
from .common import DiagnosticError, file_sha256, read_json, verify_self_hashed_json, write_self_hashed_json
from .scoring import reconstruct_global


def validate_outputs(paths: config.RunPaths) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        inputs = verify_self_hashed_json(paths.inputs)
        audit = verify_self_hashed_json(paths.historical_audit)
        protocol = verify_self_hashed_json(paths.protocol)
        media = verify_self_hashed_json(paths.media / "manifest.json")
        scores = verify_self_hashed_json(paths.scores / "manifest.json")
        support = verify_self_hashed_json(paths.support)
        review = verify_self_hashed_json(paths.review / "manifest.json")
        checks["self_hashes"] = True
    except Exception as exc:
        errors.append(f"self_hash:{type(exc).__name__}:{exc}")
        inputs = audit = protocol = media = scores = support = review = {}
        checks["self_hashes"] = False
    checks["history_audit_22"] = bool(audit.get("record_count") == config.EXPECTED_HISTORY_COUNT and audit.get("passed_count") == config.EXPECTED_HISTORY_COUNT)
    checks["media_24_rows"] = bool(media.get("cell_count") == config.EXPECTED_NEW_CELL_COUNT and len(media.get("cells", [])) == config.EXPECTED_NEW_CELL_COUNT)
    checks["scores_24_rows"] = bool(scores.get("cell_count") == config.EXPECTED_NEW_CELL_COUNT and len(scores.get("scores", [])) == config.EXPECTED_NEW_CELL_COUNT)
    complete_scores = 0
    for row in scores.get("scores", []):
        if row.get("status") != "complete":
            continue
        try:
            matrix = _matrix(row)
            reconstructed = reconstruct_global(matrix)
            parity = row.get("parity", {})
            if float(parity.get("worker_matrix_reconstruction_c_abs_error", 1.0)) > 0.001 or not bool(parity.get("worker_matrix_reconstruction_offset_equal")):
                raise DiagnosticError(f"parity failed: {row.get('cell_id')}")
            complete_scores += 1
        except Exception as exc:
            errors.append(f"score:{row.get('cell_id')}:{type(exc).__name__}:{exc}")
    checks["score_matrices_verified"] = complete_scores == int(scores.get("complete_count", -1))
    checks["scores_csv_24_rows"] = False
    if paths.scores_csv.is_file():
        with paths.scores_csv.open(encoding="utf-8", newline="") as handle:
            checks["scores_csv_24_rows"] = sum(1 for _ in csv.DictReader(handle)) == config.EXPECTED_NEW_CELL_COUNT
    else:
        errors.append("scores.csv missing")
    checks["playback_page"] = bool((paths.playback / "index.html").is_file() and review.get("human_review_status") == "NOT_HUMAN_REVIEWED")
    checks["support_traceable"] = bool(support.get("protocol_id") == config.PROTOCOL_ID and "samples" in support)
    checks["c_uses_a_gn_crop"] = all(
        bool(support.get("samples", {}).get(sample_id, {}).get("C", {}).get("fixed_same_pipeline_crop_frames"))
        and bool(support.get("samples", {}).get(sample_id, {}).get("C", {}).get("a_pipeline_source", {}).get("scored_media_sha256"))
        for sample_id in config.SAMPLE_IDS
    )
    checks["b_pts_zero_based"] = all(
        bool(support.get("samples", {}).get(sample_id, {}).get("B", {}).get("crop", {}).get("pts_validation", {}).get("zero_based"))
        and bool(support.get("samples", {}).get(sample_id, {}).get("B", {}).get("crop", {}).get("pts_validation", {}).get("contiguous"))
        for sample_id in config.SAMPLE_IDS
    )
    passed = all(checks.values()) and not errors
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if passed else "blocked",
        "checks": checks,
        "errors": errors,
        "complete_score_count": complete_scores,
        "expected_score_count": config.EXPECTED_NEW_CELL_COUNT,
    }
    write_self_hashed_json(paths.root / "validation.json", result)
    return result
