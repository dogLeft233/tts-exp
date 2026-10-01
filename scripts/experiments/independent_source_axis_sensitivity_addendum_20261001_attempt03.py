#!/usr/bin/env python3
"""Rebuild source-axis sensitivity metrics from sealed lag matrices.

This addendum intentionally does not trust the legacy report's search_gain
field: that field was copied from C in the earlier consumer.  It reads only
the sealed row plan/results and score NPZs, verifies file hashes and matrix
reconstruction, then recomputes every metric over all four frozen supports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_SHA256 = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
ROW_PLAN_SHA256 = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
RESULTS_SHA256 = "fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa"
LINEAGE_SHA256 = "0aa21a38aa7b645fb556a589ff132ad71b2083d443df73ecf795f073a522c39c"
EXPECTED_PROTOCOL = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
EXPECTED_ROWS = 2560
EXPECTED_SELECTED = 480
EXPECTED_CLIPS = 80
EXPECTED_SPEAKERS = 40
FACES = ("primary", "secondary")
CONDITIONS = ("baseline", "orth_plus", "orth_minus")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
CONTRASTS = {
    "orth_plus_minus_baseline": ("orth_plus", "baseline"),
    "orth_plus_minus_orth_minus": ("orth_plus", "orth_minus"),
}
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
ANCHOR_COLUMN = 18
VSHIFT = 15


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_no} is not a JSON object")
            output.append(value)
    return output


def _safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if path != root and root not in path.parents:
        raise AuditError(f"artifact escapes frozen run directory: {relative}")
    if not path.is_file():
        raise AuditError(f"artifact is not a file: {relative}")
    return path


def _expected_support(count: int, guard: int) -> np.ndarray:
    start = min(guard, count)
    stop = max(start, count - guard)
    return np.arange(start, stop, dtype=np.int32)


def _expected_supports(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": _expected_support(count, 20),
        "guard15": _expected_support(count, 15),
    }


def _rebuild_lag(cross: np.ndarray, zero: np.ndarray, query_n: int) -> np.ndarray:
    """Rebuild legacy observed score with query_n as both V and A cutoff."""
    cross32 = np.asarray(cross, dtype=np.float32)
    zero32 = np.asarray(zero, dtype=np.float32)
    if cross32.ndim != 2 or zero32.shape != (cross32.shape[0],):
        raise AuditError("full cross-distance/zero arrays have inconsistent shapes")
    if query_n < 1 or query_n > cross32.shape[0] or query_n > cross32.shape[1]:
        raise AuditError("legacy scoring cutoff is outside saved full V×A matrix")
    matrix = np.empty((query_n, 2 * VSHIFT + 1), dtype=np.float32)
    for i in range(query_n):
        for column, lag in enumerate(range(-VSHIFT, VSHIFT + 1)):
            j = i + lag
            matrix[i, column] = cross32[i, j] if 0 <= j < query_n else zero32[i]
    return matrix


def _curve_metrics(matrix: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    subset = np.asarray(matrix, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    if subset.ndim != 2 or subset.shape[0] == 0 or subset.shape[1] != 31:
        raise AuditError("a selected support has no windows or wrong lag count")
    # This exactly follows the frozen reporting convention: mean in float32,
    # then scalar curve summaries in float64.
    curve = subset.mean(axis=0, dtype=np.float32).astype(np.float64)
    minimum = float(np.min(curve))
    median = float(np.median(curve))
    return {
        "C": median - minimum,
        "C_anchor": median - float(curve[ANCHOR_COLUMN]),
        "B": median,
        "D": minimum,
        # Correct definition: fixed query at column 15 (k=0) minus D.
        "search_gain": float(curve[15]) - minimum,
    }


def _speaker(sample_id: str) -> str:
    match = re.match(r"^(S\d+)_", sample_id)
    if not match:
        raise AuditError(f"sample ID has no frozen speaker prefix: {sample_id}")
    return match.group(1)


def _bootstrap(values_by_sample: dict[str, float], samples: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        grouped[_speaker(sample)].append(float(values_by_sample[sample]))
    if len(grouped) != EXPECTED_SPEAKERS or any(len(v) != 2 for v in grouped.values()):
        raise AuditError("contrast is not complete for 40 speakers with two clips each")
    speakers = sorted(grouped)
    per_speaker = np.asarray(
        [np.mean(np.asarray(grouped[s], dtype=np.float64), dtype=np.float64) for s in speakers],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = rng.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)), endpoint=False)
    means = per_speaker[draws].mean(axis=1, dtype=np.float64)
    lo, hi = np.quantile(means, (0.005, 0.995), method="linear")
    return {
        "equal_speaker_mean": float(per_speaker.mean(dtype=np.float64)),
        "speaker_bootstrap_99_ci": [float(lo), float(hi)],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "complete_speaker_count": len(speakers),
        "complete_clip_count": len(samples),
        "speaker_values": {speaker: float(value) for speaker, value in zip(speakers, per_speaker)},
    }


def _load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    protocol_path = run_dir / "evaluation_protocol.json"
    plan_path = run_dir / "evaluation_rows.jsonl"
    results_path = run_dir / "evaluation_results.jsonl"
    for path in (protocol_path, plan_path, results_path):
        if not path.is_file():
            raise AuditError(f"missing sealed input: {path}")
    observed = {
        "protocol_sha256": sha256_file(protocol_path),
        "row_plan_sha256": sha256_file(plan_path),
        "results_sha256": sha256_file(results_path),
    }
    expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "row_plan_sha256": ROW_PLAN_SHA256,
        "results_sha256": RESULTS_SHA256,
    }
    if observed != expected:
        raise AuditError(f"sealed run source hashes differ: observed={observed}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("row_manifest_sha256") != ROW_PLAN_SHA256:
        raise AuditError("protocol binds a different row plan")
    method = protocol.get("method", {})
    if int(method.get("global_anchor_column", -1)) != ANCHOR_COLUMN:
        raise AuditError("protocol global anchor column differs from frozen column 18")
    plan = _jsonl(plan_path)
    results = _jsonl(results_path)
    if len(plan) != EXPECTED_ROWS or len(results) != EXPECTED_ROWS:
        raise AuditError(f"expected 2,560 planned/results rows, saw {len(plan)}/{len(results)}")
    return plan, results, protocol


def analyze(run_dir: Path, lineage_path: Path, output_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    lineage_path = lineage_path.resolve(strict=True)
    if sha256_file(lineage_path) != LINEAGE_SHA256:
        raise AuditError("inherited-artifact lineage verification SHA mismatch")

    plan_rows, results, protocol = _load_rows(run_dir)
    plan_by_id = {str(row["row_id"]): row for row in plan_rows}
    result_by_id = {str(row["row_id"]): row for row in results}
    if len(plan_by_id) != EXPECTED_ROWS or len(result_by_id) != EXPECTED_ROWS:
        raise AuditError("sealed plan/results contain duplicate row IDs")
    if set(plan_by_id) != set(result_by_id):
        raise AuditError("sealed plan and result row identities differ")

    selected_plans: dict[tuple[str, str, str], dict[str, Any]] = {}
    selected_speakers: dict[str, str] = {}
    for row_id, row in plan_by_id.items():
        if row.get("family") == "source_axis" and row.get("source") == "natural" and row.get("condition") in CONDITIONS:
            face = str(row.get("face_id"))
            sample = str(row.get("sample_id"))
            condition = str(row.get("condition"))
            if face not in FACES:
                raise AuditError(f"unexpected face in selected row {row_id}")
            key = (sample, face, condition)
            if key in selected_plans:
                raise AuditError(f"duplicate selected identity: {key}")
            selected_plans[key] = row
            speaker = str(row.get("speaker_id"))
            if speaker != _speaker(sample):
                raise AuditError(f"speaker/sample identity mismatch for {sample}")
            selected_speakers[sample] = speaker
    samples = sorted(selected_speakers)
    if len(samples) != EXPECTED_CLIPS or len(set(selected_speakers.values())) != EXPECTED_SPEAKERS:
        raise AuditError("selected rows do not represent the frozen 80-clip/40-speaker cohort")
    expected_keys = {(s, face, cond) for s in samples for face in FACES for cond in CONDITIONS}
    if set(selected_plans) != expected_keys or len(selected_plans) != EXPECTED_SELECTED:
        raise AuditError("selected source-axis baseline/orth± row set is incomplete")

    metric_rows: dict[tuple[str, str, str], dict[str, dict[str, dict[str, float]]]] = {}
    artifact_records = []
    supports_by_sample_face: dict[tuple[str, str], np.ndarray] = {}
    for (sample, face, condition), plan_row in sorted(selected_plans.items()):
        row_id = str(plan_row["row_id"])
        result = result_by_id[row_id]
        for field in ("sample_id", "speaker_id", "family", "source", "condition", "face_id"):
            if str(result.get(field)) != str(plan_row.get(field)):
                raise AuditError(f"result identity field {field} differs for {row_id}")
        if result.get("status") != "scored":
            raise AuditError(f"selected row is not scored: {row_id}")
        score_path = _safe_artifact(run_dir, str(plan_row["score_npz"]))
        score_sha = sha256_file(score_path)
        if score_sha != result.get("score_npz_sha256"):
            raise AuditError(f"score artifact SHA differs from sealed result: {row_id}")
        with np.load(score_path, allow_pickle=False) as archive:
            required = {
                "distance_matrix_raw_float32", "distance_matrix_unit_float32",
                "cross_distance_raw_float32", "cross_distance_unit_float32",
                "cross_distance_to_zero_raw_float32", "cross_distance_to_zero_unit_float32",
                "common_support_indices_int32", "full_support_indices_int32", "valid_support_indices_int32",
                "guard20_indices_int32", "guard15_indices_int32", "lag_columns_int32", "lag_k_int32",
            }
            if not required.issubset(archive.files):
                raise AuditError(f"score NPZ misses required fields: {row_id}")
            arrays = {key: archive[key] for key in required}
        n = int(result.get("common_support_n", -1))
        if n < 41 or int(result.get("scoring_audio_domain_n", -1)) != n:
            raise AuditError(f"selected row common support/count is invalid: {row_id}")
        if int(result.get("visual_query_domain_n", -1)) != arrays["cross_distance_raw_float32"].shape[0]:
            raise AuditError(f"visual domain metadata differs from saved cross matrix: {row_id}")
        if int(result.get("audio_row_domain_n", -1)) != arrays["cross_distance_raw_float32"].shape[1]:
            raise AuditError(f"audio domain metadata differs from saved cross matrix: {row_id}")
        if not np.array_equal(arrays["lag_columns_int32"], np.arange(31, dtype=np.int32)):
            raise AuditError(f"lag columns differ from frozen 31-column mapping: {row_id}")
        if not np.array_equal(arrays["lag_k_int32"], np.arange(-15, 16, dtype=np.int32)):
            raise AuditError(f"lag k differs from frozen -15…15 mapping: {row_id}")
        expected_support = _expected_supports(n)
        if not np.array_equal(arrays["common_support_indices_int32"], np.arange(n, dtype=np.int32)):
            raise AuditError(f"common query support differs from frozen recipient clock: {row_id}")
        for name, expected in expected_support.items():
            stored_name = f"{name}_support_indices_int32" if name in ("full", "valid") else f"{name}_indices_int32"
            if not np.array_equal(arrays[stored_name], expected):
                raise AuditError(f"{name} support indices differ from frozen rule: {row_id}")
        group_key = (sample, face)
        previous_support = supports_by_sample_face.get(group_key)
        if previous_support is None:
            supports_by_sample_face[group_key] = np.arange(n, dtype=np.int32)
        elif len(previous_support) != n:
            raise AuditError(f"selected conditions do not share common clock support: {sample}/{face}")

        metrics_for_row: dict[str, dict[str, dict[str, float]]] = {}
        for geometry in GEOMETRIES:
            matrix = arrays[f"distance_matrix_{geometry}_float32"]
            cross = arrays[f"cross_distance_{geometry}_float32"]
            zero = arrays[f"cross_distance_to_zero_{geometry}_float32"]
            if matrix.dtype != np.float32 or matrix.shape != (n, 31):
                raise AuditError(f"saved {geometry} lag matrix has wrong dtype/shape: {row_id}")
            if cross.dtype != np.float32 or zero.dtype != np.float32:
                raise AuditError(f"saved {geometry} cross/zero dtypes are wrong: {row_id}")
            if not np.isfinite(matrix).all() or not np.isfinite(cross).all() or not np.isfinite(zero).all():
                raise AuditError(f"nonfinite score matrix values: {row_id}")
            rebuilt = _rebuild_lag(cross, zero, n)
            if not np.array_equal(matrix, rebuilt):
                raise AuditError(f"saved {geometry} lag matrix does not rebuild from full V×A scores: {row_id}")
            metrics_for_row[geometry] = {
                support: _curve_metrics(matrix, indices)
                for support, indices in expected_support.items()
            }
        metric_rows[(sample, face, condition)] = metrics_for_row
        artifact_records.append({"row_id": row_id, "score_npz": str(plan_row["score_npz"]), "sha256": score_sha, "common_support_n": n})

    contrasts: dict[str, Any] = {}
    for contrast_name, (left, right) in CONTRASTS.items():
        contrasts[contrast_name] = {}
        for face in FACES:
            contrasts[contrast_name][face] = {}
            for geometry in GEOMETRIES:
                contrasts[contrast_name][face][geometry] = {}
                for support in SUPPORTS:
                    per_metric = {}
                    for metric in METRICS:
                        clip_deltas = {}
                        for sample in samples:
                            left_value = metric_rows[(sample, face, left)][geometry][support][metric]
                            right_value = metric_rows[(sample, face, right)][geometry][support][metric]
                            clip_deltas[sample] = float(left_value - right_value)
                        per_metric[metric] = {**_bootstrap(clip_deltas, samples), "clip_values": clip_deltas}
                    contrasts[contrast_name][face][geometry][support] = per_metric

    output = {
        "schema_version": 1,
        "analysis_type": "independent_source_axis_sensitivity_addendum_matrix_rebuilt",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "run_dir": str(run_dir),
            "protocol_sha256": PROTOCOL_SHA256,
            "row_plan_sha256": ROW_PLAN_SHA256,
            "results_sha256": RESULTS_SHA256,
            "lineage_verification_path": str(lineage_path),
            "lineage_verification_sha256": LINEAGE_SHA256,
            "lineage_artifact_count": 3033,
        },
        "verification": {
            "planned_rows": EXPECTED_ROWS,
            "result_rows": EXPECTED_ROWS,
            "selected_score_artifacts": len(artifact_records),
            "matrix_rebuild_rows": len(artifact_records),
            "selected_clips": len(samples),
            "selected_speakers": len(set(selected_speakers.values())),
            "conditions": list(CONDITIONS),
            "faces": list(FACES),
            "geometries": list(GEOMETRIES),
            "supports": list(SUPPORTS),
            "metrics": list(METRICS),
            "anchor_column": ANCHOR_COLUMN,
            "anchor_k": 3,
            "official_syncnet_offset": -3,
            "speaker_weighting": "two clips averaged within speaker, then 40 speakers equally",
            "confidence_interval": "20,000 PCG64 speaker bootstrap draws; seed 20260930; percentile 0.005–0.995",
            "search_gain_definition": "mean lag-distance at fixed zero-lag column 15 minus D=min(mean lag-distance curve)",
            "score_matrices_rebuilt_from_full_cross_domain": True,
        },
        "contrasts": contrasts,
        "verified_score_artifacts": artifact_records,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise AuditError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lineage-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.lineage_verification, args.output)
    print(json.dumps({"status": report["status"], "selected_score_artifacts": report["verification"]["selected_score_artifacts"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
