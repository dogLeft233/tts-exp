#!/usr/bin/env python3
"""Independent all-support temporal-null addendum for random specificity.

Reads the immutable auxiliary score matrices, reconstructs the observed and
permuted lag curves, and reports the frozen equal-dose contrast over all four
query supports.  It does not import producer scoring code or run model forwards.
The earlier independent audit is required as a sealed input for the full
matrix and float32-dose checks; this script adds the missing all-domain nulls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


SPEC_SHA256 = "679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79"
PRIOR_AUDIT_SHA256 = "dc73acddc9f745c6c4a8e87ec635a9abc21b1920406ad99a3eb63ec40744867e"
RUN_ROOT = "/root/tts-research-system-20261001/source_axis/random_specificity_aux_20261001_attempt03"
RUN_SEALS = {
    "evaluation_protocol.json": "2a5f834939164a75b68b9d059c58580b4ad260dc7d1848536b3206e1018a579b",
    "evaluation_rows.jsonl": "33b0d36697e5c28f12210bcea4cbab164b2c12353f9b5df0bafbe2d0a62a1aa3",
    "evaluation_results.jsonl": "0e783920d5a7fae1883b005177ad467149c1896df263aab5d6973f76cf1977e3",
    "run_status.json": "e3ef70034ac08052c331a4d7910145f3302043e08c0ab58c610887ce8c3cb753",
    "code_seal.json": "73c45d80e1e904aa091f3577fe8574224dd48b17d1c5532e97b86b987782331d",
    "all_files_sha256.json": "555c854cb42508377c2de5cc689e63b308dd9ebf12d02c24969e1fa050d7b535",
    "source_artifact_inventory.json": "e1d239d20966deab34cb67ee66846f6c422b55d32069d442644d44769aa16f2a",
}
CONDITIONS = ("orth_plus", "random1", "random2", "random3")
FACES = ("primary", "secondary")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "D0", "search_gain")
N_CLIPS = 80
N_SPEAKERS = 40
CLIPS_PER_SPEAKER = 2
LAGS = np.arange(-15, 16, dtype=np.int64)
ANCHOR_COLUMN = 18
ZERO_COLUMN = 15
NULL_DRAWS = 100
NULL_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
DISTANCE_ATOL = 5e-4


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AuditError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(row)
    return rows


def safe_artifact(root: Path, relative: str) -> Path:
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise AuditError(f"unsafe run-relative artifact path: {relative}")
    path = root.joinpath(*rel.parts).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AuditError(f"artifact path escapes run root: {relative}") from exc
    return path


def support_map(query_count: int) -> dict[str, np.ndarray]:
    full = np.arange(query_count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(min(20, query_count), max(min(20, query_count), query_count - 20), dtype=np.int32),
        "guard15": np.arange(min(15, query_count), max(min(15, query_count), query_count - 15), dtype=np.int32),
    }


def make_lag_matrix(cross: np.ndarray, zero: np.ndarray, queries: np.ndarray, audio_indices: np.ndarray) -> np.ndarray:
    """Use full-A rows at in-range lags and SyncNet's zero vector at boundaries."""
    x = np.asarray(cross)
    z = np.asarray(zero)
    q = np.asarray(queries, dtype=np.int64)
    indices = np.asarray(audio_indices, dtype=np.int64)
    if x.dtype != np.float32 or x.ndim != 2 or z.dtype != np.float32 or z.shape != (x.shape[0],):
        raise AuditError("cross/zero score arrays must be float32 with V×A and V shapes")
    if q.ndim != 1 or indices.shape != (x.shape[1],) or np.any(q < 0) or np.any(q >= x.shape[0]):
        raise AuditError("query/permutation domains do not match the cached full V×A matrix")
    source = q[:, None] + LAGS[None, :]
    in_range = (source >= 0) & (source < len(indices))
    clipped = np.clip(source, 0, len(indices) - 1)
    values = x[q[:, None], indices[clipped]]
    return np.where(in_range, values, z[q, None]).astype(np.float32, copy=False)


def metric_values(matrix: np.ndarray) -> dict[str, float]:
    curve = np.asarray(matrix, dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    if curve.shape != (31,) or not np.isfinite(curve).all():
        raise AuditError("lag curve must contain 31 finite columns")
    minimum = float(np.min(curve))
    baseline = float(curve[ZERO_COLUMN])
    median = float(np.median(curve))
    return {
        "C": median - minimum,
        "C_anchor": median - float(curve[ANCHOR_COLUMN]),
        "B": median,
        "D": minimum,
        "D0": baseline,
        "search_gain": baseline - minimum,
    }


def permutation_set(audio_count: int, ordered_index: int) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.Generator(np.random.PCG64(NULL_SEED + ordered_index))
    vectors: list[np.ndarray] = []
    hashes: list[str] = []
    expected = np.arange(audio_count, dtype=np.int64)
    for _ in range(NULL_DRAWS):
        vector = rng.permutation(audio_count).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(vector), expected):
            raise AuditError("generated temporal-null vector is not a full RAW-N A permutation")
        vectors.append(vector)
        hashes.append(sha256_array(vector.astype("<i8", copy=False)))
    return vectors, hashes


def bootstrap_summary(values_by_sample: dict[str, float], speaker_by_sample: dict[str, str], sample_ids: list[str]) -> dict[str, Any]:
    clips_by_speaker: dict[str, list[float]] = defaultdict(list)
    for sample_id, value in values_by_sample.items():
        clips_by_speaker[speaker_by_sample[sample_id]].append(float(value))
    speaker_values = {
        speaker: float(np.mean(values, dtype=np.float64))
        for speaker, values in clips_by_speaker.items() if len(values) == CLIPS_PER_SPEAKER
    }
    all_speakers = sorted(set(speaker_by_sample.values()))
    incomplete = [speaker for speaker in all_speakers if speaker not in speaker_values]
    if speaker_values:
        values = np.asarray([speaker_values[speaker] for speaker in sorted(speaker_values)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise AuditError("nonfinite speaker-level contrast")
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)), endpoint=False)
        boot_means = values[indices].mean(axis=1, dtype=np.float64)
        low, high = np.quantile(boot_means, [0.005, 0.995], method="linear")
        mean = float(values.mean(dtype=np.float64))
        sd = float(values.std(ddof=1, dtype=np.float64)) if len(values) > 1 else None
        ci = [float(low), float(high)]
    else:
        mean = sd = ci = None
    return {
        "equal_speaker_mean_complete_pairs": mean,
        "speaker_mean_sd": sd,
        "speaker_bootstrap_99_ci": ci,
        "bootstrap_draws": BOOTSTRAP_DRAWS if speaker_values else 0,
        "bootstrap_rng": "PCG64",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "intended_speakers": len(all_speakers),
        "complete_two_clip_speakers": len(speaker_values),
        "incomplete_speakers": incomplete,
        "complete_clip_count": sum(len(values) for values in clips_by_speaker.values() if len(values) == CLIPS_PER_SPEAKER),
        "intended_clip_count": len(sample_ids),
        "eligible_for_intended_cohort": len(speaker_values) == N_SPEAKERS and not incomplete and len(sample_ids) == N_CLIPS,
        "speaker_values": speaker_values,
    }


def expected_prior_audit(path: Path, root: Path) -> dict[str, Any]:
    if sha256_file(path) != PRIOR_AUDIT_SHA256:
        raise AuditError("prior independent full-matrix/dose audit SHA differs")
    prior = read_json(path)
    if prior.get("status") != "COMPLETE":
        raise AuditError("prior independent audit is not COMPLETE")
    seals = prior.get("input_seals")
    if not isinstance(seals, dict) or seals.get("run_root") != RUN_ROOT:
        raise AuditError("prior independent audit binds another auxiliary run")
    for field, filename in (
        ("protocol_sha256", "evaluation_protocol.json"),
        ("row_manifest_sha256", "evaluation_rows.jsonl"),
        ("results_jsonl_sha256", "evaluation_results.jsonl"),
        ("run_status_sha256", "run_status.json"),
        ("all_files_manifest_sha256", "all_files_sha256.json"),
    ):
        if seals.get(field) != RUN_SEALS[filename]:
            raise AuditError(f"prior independent audit does not bind expected {filename}")
    verification = prior.get("verification", {})
    if (verification.get("planned_rows") != 640 or verification.get("scored_rows") != 640
            or verification.get("score_artifacts_recomputed") != 640
            or verification.get("full_v_by_full_a_matrices_recomputed") is not True
            or verification.get("all_random_frame_doses_reconstructed_from_independent_kkt") is not True
            or verification.get("all_listed_files_exist_and_match_sha") is not True
            or verification.get("exact_all_files_path_set_match") is not True):
        raise AuditError("prior audit does not verify all 640 score rows, full matrices, and random doses")
    dose = verification.get("strict_random_dose_totals")
    if not isinstance(dose, dict) or set(dose) != {"random1", "random2", "random3"}:
        raise AuditError("prior audit lacks the three strict random-dose summaries")
    for arm, item in dose.items():
        if (item.get("strict_equal_dose_eligible_clips") != 80 or item.get("miss_frames") != 0
                or item.get("capped_infeasible_frames") != 0 or item.get("pass_frames") != item.get("active_frames")):
            raise AuditError(f"prior strict random-dose check is incomplete: {arm}")
    return {
        "path": str(path.resolve()), "sha256": PRIOR_AUDIT_SHA256,
        "verified_score_rows": 640, "verified_random_dose_frames": dose,
        "all_files_match": verification.get("all_listed_files_exist_and_match_sha") is True,
    }


def verify_inputs(root: Path, prior_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    if str(root.resolve()) != RUN_ROOT:
        raise AuditError(f"run root differs from frozen path: {root.resolve()}")
    for name, expected in RUN_SEALS.items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise AuditError(f"sealed input hash mismatch for {name}: {actual}")
    protocol = read_json(root / "evaluation_protocol.json")
    rows = read_jsonl(root / "evaluation_rows.jsonl")
    results = read_jsonl(root / "evaluation_results.jsonl")
    status = read_json(root / "run_status.json")
    all_files = read_json(root / "all_files_sha256.json")
    if protocol.get("row_manifest", {}).get("sha256") != RUN_SEALS["evaluation_rows.jsonl"]:
        raise AuditError("protocol does not bind the frozen row-plan hash")
    if status.get("status") != "complete" or status.get("completed_samples") != 80 or status.get("failed_samples") != 0:
        raise AuditError("source auxiliary run is not complete 80/80 with no failed clips")
    if len(rows) != 640 or len(results) != 640 or not isinstance(all_files, dict):
        raise AuditError("source auxiliary artifact row counts differ from 640 planned/scored rows")
    prior = expected_prior_audit(prior_path, root)
    return {"protocol": protocol, "run_status": status, "prior_audit": prior}, rows, results, all_files


def load_score(path: Path, expected_sha: str, all_files: dict[str, str], relative: str, row: dict[str, Any], result: dict[str, Any], query_count: int) -> dict[str, np.ndarray]:
    if all_files.get(relative) != expected_sha or result.get("score_npz_sha256") != expected_sha:
        raise AuditError(f"score hash is not bound by result row and all-files seal: {row['row_id']}")
    if sha256_file(path) != expected_sha:
        raise AuditError(f"score artifact hash mismatch: {row['row_id']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "cross_distance_raw_float32", "cross_distance_to_zero_raw_float32", "distance_matrix_raw_float32",
        "cross_distance_unit_float32", "cross_distance_to_zero_unit_float32", "distance_matrix_unit_float32",
        "common_support_full_int32", "common_support_valid_int32", "common_support_guard20_int32", "common_support_guard15_int32",
        "lag_columns_int32", "lag_k_int32",
    }
    if set(arrays) != required:
        raise AuditError(f"score array key set differs from sealed contract: {row['row_id']}")
    for geometry in GEOMETRIES:
        cross = arrays[f"cross_distance_{geometry}_float32"]
        zero = arrays[f"cross_distance_to_zero_{geometry}_float32"]
        lag = arrays[f"distance_matrix_{geometry}_float32"]
        if cross.dtype != np.float32 or cross.ndim != 2 or zero.dtype != np.float32 or zero.shape != (cross.shape[0],):
            raise AuditError(f"invalid {geometry} V×A/zero arrays: {row['row_id']}")
        if lag.dtype != np.float32 or lag.shape != (query_count, 31) or query_count > cross.shape[0]:
            raise AuditError(f"invalid {geometry} lag array: {row['row_id']}")
        if not np.isfinite(cross).all() or not np.isfinite(zero).all() or not np.isfinite(lag).all():
            raise AuditError(f"nonfinite {geometry} score array: {row['row_id']}")
    return arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve(strict=True)
    out = args.outdir.resolve()
    if out.exists():
        raise AuditError(f"output directory already exists: {out}")
    prior_path = args.prior_audit.resolve(strict=True)
    seals, rows, results, all_files = verify_inputs(root, prior_path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    result_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in rows_by_id:
            raise AuditError("planned score rows have missing/duplicate row_id")
        rows_by_id[row_id] = row
    for result in results:
        row_id = str(result.get("row_id", ""))
        if not row_id or row_id in result_by_id:
            raise AuditError("result rows have missing/duplicate row_id")
        result_by_id[row_id] = result
    if set(rows_by_id) != set(result_by_id):
        raise AuditError("planned and scored row identities differ")

    sample_order: list[str] = []
    speaker_by_sample: dict[str, str] = {}
    ordered_by_sample: dict[str, int] = {}
    row_grid: dict[tuple[str, str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row_id, row in rows_by_id.items():
        result = result_by_id[row_id]
        sample = str(row.get("sample_id", ""))
        speaker = str(row.get("speaker_id", ""))
        face = str(row.get("face_id", ""))
        condition = str(row.get("condition", ""))
        ordered = int(row.get("ordered_eval_index", -1))
        if (not sample or not speaker or face not in FACES or condition not in CONDITIONS or ordered < 0
                or row.get("family") != "random_specificity_aux" or row.get("source") != "natural"
                or row.get("audio_level") != "RAW_N" or row.get("split") != "evaluation" or row.get("status") != "planned"):
            raise AuditError(f"planned row identity differs from frozen auxiliary design: {row_id}")
        if result.get("status") != "scored" or any(result.get(key) != row.get(key) for key in ("sample_id", "speaker_id", "face_id", "condition", "ordered_eval_index")):
            raise AuditError(f"result row is not an exact scored match for its plan: {row_id}")
        if result.get("score_computed") is not True or result.get("anchor") != {"k": 3, "column": 18, "official_syncnet_offset": -3}:
            raise AuditError(f"result row score/anchor record differs from the frozen contract: {row_id}")
        if row.get("score_npz") != result.get("score_npz"):
            raise AuditError(f"result score path differs from row plan: {row_id}")
        speaker_prior = speaker_by_sample.setdefault(sample, speaker)
        ordered_prior = ordered_by_sample.setdefault(sample, ordered)
        if speaker_prior != speaker or ordered_prior != ordered:
            raise AuditError(f"sample speaker/index changes across score rows: {sample}")
        key = (sample, face, condition)
        if key in row_grid:
            raise AuditError(f"duplicate sample/face/condition row: {key}")
        row_grid[key] = (row, result)
    sample_order = sorted(speaker_by_sample, key=lambda sample: ordered_by_sample[sample])
    if len(sample_order) != N_CLIPS or sorted(ordered_by_sample[s] for s in sample_order) != list(range(N_CLIPS)):
        raise AuditError("sample order does not contain the frozen 80 unique indexes")
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for sample in sample_order:
        by_speaker[speaker_by_sample[sample]].append(sample)
    if len(by_speaker) != N_SPEAKERS or any(len(items) != CLIPS_PER_SPEAKER for items in by_speaker.values()):
        raise AuditError("frozen cohort is not 40 speakers with two clips each")
    if len(row_grid) != 640 or any((sample, face, cond) not in row_grid for sample in sample_order for face in FACES for cond in CONDITIONS):
        raise AuditError("score grid is incomplete; no rows may be selected away")

    score_arrays: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    supports_by_sample_face: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audio_count_by_sample: dict[str, int] = {}
    score_hashes: dict[str, str] = {}
    observed_by_domain: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    null_by_domain: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    permutation_hashes: dict[str, list[str]] = {}
    score_count = 0

    # Load all 640 immutable score matrices and verify each stored lag matrix
    # against its full V×A cross matrix before producing any contrasts.
    for sample in sample_order:
        for face in FACES:
            expected_supports: dict[str, np.ndarray] | None = None
            query_count: int | None = None
            for condition in CONDITIONS:
                row, result = row_grid[(sample, face, condition)]
                relative = str(row["score_npz"])
                path = safe_artifact(root, relative)
                score_sha = str(result.get("score_npz_sha256", ""))
                row_a = int(result.get("audio_row_domain_n", -1))
                row_v = int(result.get("visual_query_domain_n", -1))
                row_q = int(result.get("common_support_n", -1))
                arrays = load_score(path, score_sha, all_files, relative, row, result, row_q)
                cross_shape = arrays["cross_distance_raw_float32"].shape
                if row_a != cross_shape[1] or row_v != cross_shape[0] or row_q < 1 or row_q > min(row_a, row_v):
                    raise AuditError(f"score domain counts differ from full matrix: {row_id if (row_id := row['row_id']) else sample}")
                if arrays["cross_distance_unit_float32"].shape != cross_shape:
                    raise AuditError(f"raw/unit full cross shapes differ: {row['row_id']}")
                if query_count is None:
                    query_count = row_q
                elif query_count != row_q:
                    raise AuditError(f"common query count differs across the four conditions: {sample}/{face}")
                if arrays["cross_distance_raw_float32"].shape[1] != row_a:
                    raise AuditError(f"full RAW-N A domain mismatch: {row['row_id']}")
                supports = support_map(row_q)
                expected_keys = {
                    "full": "common_support_full_int32", "valid": "common_support_valid_int32",
                    "guard20": "common_support_guard20_int32", "guard15": "common_support_guard15_int32",
                }
                if not np.array_equal(arrays["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(arrays["lag_k_int32"], LAGS.astype(np.int32)):
                    raise AuditError(f"lag-column identity differs: {row['row_id']}")
                for name, array_key in expected_keys.items():
                    idx = arrays[array_key]
                    if idx.dtype != np.int32 or not np.array_equal(idx, supports[name]):
                        raise AuditError(f"{name} support differs from frozen query rule: {row['row_id']}")
                if expected_supports is None:
                    expected_supports = supports
                    supports_by_sample_face[(sample, face)] = supports
                else:
                    for name in SUPPORTS:
                        if not np.array_equal(expected_supports[name], supports[name]):
                            raise AuditError(f"query support differs across conditions: {sample}/{face}/{name}")
                identity = np.arange(row_a, dtype=np.int64)
                for geometry in GEOMETRIES:
                    observed = make_lag_matrix(
                        arrays[f"cross_distance_{geometry}_float32"],
                        arrays[f"cross_distance_to_zero_{geometry}_float32"],
                        supports["full"], identity,
                    )
                    stored = arrays[f"distance_matrix_{geometry}_float32"][:row_q]
                    max_diff = float(np.max(np.abs(observed - stored), initial=0.0))
                    if max_diff > DISTANCE_ATOL:
                        raise AuditError(f"independent observed lag matrix mismatch ({geometry}, {max_diff}): {row['row_id']}")
                    for support in SUPPORTS:
                        indices = supports[support]
                        if len(indices):
                            observed_by_domain[(face, geometry, support, condition)][sample] = metric_values(observed[indices])
                score_arrays[(sample, face, condition)] = arrays
                score_hashes[row["row_id"]] = score_sha
                score_count += 1
            if query_count is None:
                raise AuditError(f"missing query count: {sample}/{face}")
        counts = {
            int(score_arrays[(sample, face, CONDITIONS[0])]["cross_distance_raw_float32"].shape[1])
            for face in FACES
        }
        if len(counts) != 1:
            raise AuditError(f"RAW-N A row count differs across faces: {sample}")
        audio_count_by_sample[sample] = counts.pop()

    if score_count != 640:
        raise AuditError(f"only {score_count}/640 score matrices were verified")

    # The same 100 full-A permutations are shared by every condition and face
    # for a sample. Full/valid use the official zero boundary at out-of-range
    # lag indices; guarded domains contain no such padding.
    for sample in sample_order:
        audio_n = audio_count_by_sample[sample]
        vectors, hashes = permutation_set(audio_n, ordered_by_sample[sample])
        permutation_hashes[sample] = hashes
        for face in FACES:
            supports = supports_by_sample_face[(sample, face)]
            for geometry in GEOMETRIES:
                by_condition_draws = {
                    condition: {support: {metric: np.empty(NULL_DRAWS, dtype=np.float64) for metric in METRICS} for support in SUPPORTS}
                    for condition in CONDITIONS
                }
                for draw, permutation in enumerate(vectors):
                    for condition in CONDITIONS:
                        arrays = score_arrays[(sample, face, condition)]
                        null_matrix = make_lag_matrix(
                            arrays[f"cross_distance_{geometry}_float32"],
                            arrays[f"cross_distance_to_zero_{geometry}_float32"],
                            supports["full"], permutation,
                        )
                        for support in SUPPORTS:
                            indices = supports[support]
                            if len(indices):
                                values = metric_values(null_matrix[indices])
                                for metric in METRICS:
                                    by_condition_draws[condition][support][metric][draw] = values[metric]
                for support in SUPPORTS:
                    for metric in METRICS:
                        if not len(supports[support]):
                            continue
                        random_draws = np.stack([by_condition_draws[f"random{i}"][support][metric] for i in (1, 2, 3)], axis=0)
                        specificity = by_condition_draws["orth_plus"][support][metric] - random_draws.mean(axis=0, dtype=np.float64)
                        null_by_domain[(face, geometry, support, metric)][sample] = specificity

    observed_effects: dict[str, Any] = {}
    null_effects: dict[str, Any] = {}
    per_clip_rows: list[dict[str, Any]] = []
    permutation_group_means: dict[str, Any] = {}
    for face in FACES:
        observed_effects[face] = {}
        null_effects[face] = {}
        permutation_group_means[face] = {}
        for geometry in GEOMETRIES:
            observed_effects[face][geometry] = {}
            null_effects[face][geometry] = {}
            permutation_group_means[face][geometry] = {}
            for support in SUPPORTS:
                observed_effects[face][geometry][support] = {}
                null_effects[face][geometry][support] = {}
                permutation_group_means[face][geometry][support] = {}
                for metric in METRICS:
                    key = (face, geometry, support, metric)
                    observed_clip: dict[str, float] = {}
                    mean_null_clip: dict[str, float] = {}
                    excess_clip: dict[str, float] = {}
                    all_draws = null_by_domain.get(key, {})
                    for sample in sample_order:
                        cells = [observed_by_domain.get((face, geometry, support, condition), {}).get(sample) for condition in CONDITIONS]
                        draws = all_draws.get(sample)
                        if any(cell is None for cell in cells) or draws is None or len(draws) != NULL_DRAWS:
                            continue
                        effect = float(cells[0][metric] - np.mean([cells[i][metric] for i in (1, 2, 3)], dtype=np.float64))
                        null_mean = float(np.mean(draws, dtype=np.float64))
                        observed_clip[sample] = effect
                        mean_null_clip[sample] = null_mean
                        excess_clip[sample] = effect - null_mean
                        per_clip_rows.append({
                            "sample_id": sample, "speaker_id": speaker_by_sample[sample],
                            "ordered_eval_index": ordered_by_sample[sample], "face_id": face,
                            "geometry": geometry, "support": support, "metric": metric,
                            "observed_specificity": effect, "mean_permutation_null_specificity": null_mean,
                            "observed_minus_mean_null_excess": effect - null_mean,
                            "support_n": int(len(supports_by_sample_face[(sample, face)][support])),
                            "permutation_count": NULL_DRAWS,
                        })
                    ids = sample_order
                    observed_effects[face][geometry][support][metric] = bootstrap_summary(observed_clip, speaker_by_sample, ids)
                    null_effects[face][geometry][support][metric] = {
                        "mean_permutation_null_specificity": bootstrap_summary(mean_null_clip, speaker_by_sample, ids),
                        "observed_minus_mean_null_excess": bootstrap_summary(excess_clip, speaker_by_sample, ids),
                    }

                    # One equal-speaker mean per permutation supports a paired
                    # descriptive null distribution without pooling clips.
                    draw_means: list[float | None] = []
                    for draw in range(NULL_DRAWS):
                        speaker_vals: list[float] = []
                        for speaker, clips in sorted(by_speaker.items()):
                            if len(clips) != CLIPS_PER_SPEAKER or any(clip not in all_draws for clip in clips):
                                continue
                            speaker_vals.append(float(np.mean([all_draws[clip][draw] for clip in clips], dtype=np.float64)))
                        draw_means.append(float(np.mean(speaker_vals, dtype=np.float64)) if len(speaker_vals) == N_SPEAKERS else None)
                    permutation_group_means[face][geometry][support][metric] = {
                        "equal_speaker_null_mean_by_permutation": draw_means,
                        "valid_permutation_count": sum(value is not None for value in draw_means),
                        "permutation_hashes_shared_across_conditions_and_faces": True,
                    }

    report = {
        "schema_version": 1,
        "analysis_id": "independent-random-specificity-all-domain-null-20261001",
        "status": "PASS" if score_count == 640 else "INCOMPLETE_SUPPORT",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "frozen_spec": {"sha256": SPEC_SHA256, "run_root": RUN_ROOT},
        "input_seals": {
            "protocol_sha256": RUN_SEALS["evaluation_protocol.json"],
            "row_manifest_sha256": RUN_SEALS["evaluation_rows.jsonl"],
            "results_jsonl_sha256": RUN_SEALS["evaluation_results.jsonl"],
            "run_status_sha256": RUN_SEALS["run_status.json"],
            "code_seal_sha256": RUN_SEALS["code_seal.json"],
            "all_files_manifest_sha256": RUN_SEALS["all_files_sha256.json"],
            "source_artifact_inventory_sha256": RUN_SEALS["source_artifact_inventory.json"],
            "prior_independent_audit": seals["prior_audit"],
        },
        "verification": {
            "planned_rows": 640, "score_rows_hash_verified": score_count,
            "intended_clips": N_CLIPS, "intended_speakers": N_SPEAKERS,
            "all_scores_match_all_files_and_result_hashes": score_count == 640,
            "all_observed_lag_matrices_reconstructed": True,
            "all_random_dose_and_full_matrix_checks_inherited_from_sealed_prior_audit": True,
            "permutation_count_per_clip": NULL_DRAWS,
            "permutation_domain": "full RAW-N A rows",
            "same_permutation_vectors_shared_across_conditions_and_faces": True,
            "supports": {"full": "all common V/A queries; zero boundary where source index is outside full A", "valid": "all common V/A queries before lag-edge guards", "guard20": "indices [20,n-20)", "guard15": "indices [15,n-15)"},
            "support_padding": "for full/valid nulls, preserve each V query and use its saved zero-boundary distance at any out-of-range A lag; do not drop boundary rows",
            "observed_search_gain": "D0-D at lag column 15 minus minimum D",
            "legacy_search_gain_note": "The earlier guard20-only independent consumer mapped search_gain to C; this addendum recomputes D0-D from saved matrices for every support.",
            "complete_score_grid": score_count == 640,
        },
        "analysis": {
            "contrast": "orth_plus minus the arithmetic mean of random1/random2/random3 within clip",
            "aggregation": "two clips averaged within each speaker, then 40 speakers equally weighted",
            "bootstrap": {"rng": "PCG64", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "ci": "percentile 0.5/99.5"},
            "metrics": list(METRICS), "geometries": list(GEOMETRIES), "supports": list(SUPPORTS), "faces": list(FACES),
            "observed_specificity": observed_effects,
            "null_and_excess": null_effects,
            "equal_speaker_null_mean_by_permutation": permutation_group_means,
        },
        "permutation_sha256_by_sample": permutation_hashes,
        "score_artifact_sha256_by_row_id": score_hashes,
        "per_clip_metrics_path": "independent_aux_all_domain_null_clip_rows.jsonl",
        "effects_read": True,
        "model_forward_performed_by_consumer": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "independent_aux_all_domain_null.json"
    rows_path = out / "independent_aux_all_domain_null_clip_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in per_clip_rows), encoding="utf-8", newline="\n")
    report["per_clip_metrics_sha256"] = sha256_file(rows_path)
    report["payload_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return {"status": report["status"], "report": str(report_path), "report_sha256": sha256_file(report_path), "score_rows": score_count, "per_clip_rows": len(per_clip_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--prior-audit", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:2000], "source_sha256": sha256_file(Path(__file__).resolve())}, sort_keys=True), file=__import__("sys").stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
