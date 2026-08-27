from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import assert_finite_json, canonical_sha256, write_json
from .statistics import analyze_replacement, validate_complete_matrix

AUTHORITATIVE_CELLS = ("G_N_E_N", "G_M_E_N")
DIAGNOSTIC_CELLS = ("G_N_E_M", "G_M_E_M")


def validate_cohort_binding(
    rows: Sequence[Mapping[str, Any]],
    expected_records: Sequence[Mapping[str, Any]] | None,
    expected_cohort_sha256: str | None,
) -> None:
    if not expected_records or not expected_cohort_sha256:
        raise ValueError("authoritative score matrix requires a frozen cohort and cohort hash")
    if canonical_sha256(list(expected_records)) != expected_cohort_sha256:
        raise ValueError("expected cohort hash does not match frozen records")
    observed_identity = [(str(row.get("sample_id", "")), str(row.get("source_group", ""))) for row in rows]
    expected_identity = [(str(row.get("sample_id", "")), str(row.get("source_group", ""))) for row in expected_records]
    if observed_identity != expected_identity:
        raise ValueError("score matrix is not bound to the ordered frozen cohort")


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{name} must be a non-empty SHA-256 hex digest")
    return text.lower()


def validate_score_provenance(row: Mapping[str, Any], *, model_sha256: str, source_sha256: str) -> None:
    model_hash = _require_sha256(model_sha256, "SyncNet model hash")
    source_hash = _require_sha256(source_sha256, "SyncNet source hash")
    if row.get("syncnet_model_sha256") != model_hash or row.get("syncnet_source_sha256") != source_hash:
        raise ValueError(f"SyncNet provenance mismatch for {row.get('sample_id')}")
    for cell in AUTHORITATIVE_CELLS + DIAGNOSTIC_CELLS:
        cell_row = row.get(cell)
        if not isinstance(cell_row, Mapping):
            raise ValueError(f"missing score cell {cell}")
        _require_sha256(cell_row.get("mux_sha256"), f"{cell} mux hash")


def build_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_sha256: str,
    source_sha256: str,
    expected_records: Sequence[Mapping[str, Any]] | None,
    expected_cohort_sha256: str | None,
    engineering_complete: bool,
    bootstrap_draws: int = 10000,
) -> dict[str, Any]:
    matrix = validate_complete_matrix(rows)
    validate_cohort_binding(matrix, expected_records, expected_cohort_sha256)
    for row in matrix:
        validate_score_provenance(row, model_sha256=model_sha256, source_sha256=source_sha256)
    analysis = analyze_replacement(matrix, engineering_complete=engineering_complete, bootstrap_draws=bootstrap_draws)
    analysis["syncnet_provenance"] = {"model_sha256": model_sha256, "source_sha256": source_sha256}
    analysis["cohort_provenance"] = {
        "ordered_records_sha256": expected_cohort_sha256,
        "records": [dict(record) for record in expected_records or []],
    }
    analysis["authoritative_cells"] = list(AUTHORITATIVE_CELLS)
    analysis["diagnostic_cells"] = list(DIAGNOSTIC_CELLS)
    analysis["paired_rows"] = [
        {
            "sample_id": str(row["sample_id"]),
            "source_group": str(row["source_group"]),
            "benefit_C": float(row["G_M_E_N"]["sync_c"] - row["G_N_E_N"]["sync_c"]),
            "benefit_D": float(row["G_N_E_N"]["sync_d"] - row["G_M_E_N"]["sync_d"]),
        }
        for row in matrix
    ]
    assert_finite_json(analysis)
    return analysis


def write_analysis(path: str | Path, analysis: Mapping[str, Any]) -> str:
    return write_json(path, analysis)


def score_matrix_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
