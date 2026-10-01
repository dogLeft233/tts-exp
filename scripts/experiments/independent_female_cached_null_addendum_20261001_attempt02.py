#!/usr/bin/env python3
"""Female-cohort cached temporal-null supplement for native and generator cells.

Reads the already sealed 40-clip female run and its saved V×A matrices. It
does not load a model or alter the frozen score rows. The legacy score boundary
and a matched full-A boundary are both reported so observed/null padding is
identical within each analysis.
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


PROTOCOL_SHA256 = "ce03790adf556e881ff7bafebf08270e12d3266aa534684d70e47893d0976e44"
ROW_PLAN_SHA256 = "8efd0d0ac56e34e647836c8f6b8a01aa27d826b3252356f88f094840414e88a5"
RESULTS_SHA256 = "d2e737e4a2c59ea8c2d9cbdf2df36f4717035260cab52a128c4b0bb3c27036ad"
STATUS_SHA256 = "21986cfa04610d5c4b5066f356e9c0933a7550e2a764c8641b2fc1b164b0ebf1"
ALL_FILES_SHA256 = "991f6d22953df7f6609a0cb40dc593b019f1d659f7803ba1df95df1eed86d94e"
CODE_SEAL_SHA256 = "9b124a81c45b598294f63b04d178c7b891065c0ed065d985e201219871ba2e80"
SUMMARY_SHA256 = "cd31ca9541d690ebd34b557215209e094d3a0431aaedaf9a2af4fe45a51062d6"
EXPECTED_ROWS = 640
EXPECTED_CLIPS = 40
EXPECTED_SPEAKERS = 20
CELLS = ("q00", "q10", "q01", "q11")
SOURCES = ("natural", "tts")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard15", "guard20")
PADDING = ("legacy_query_cutoff", "full_A_cutoff")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
NULL_COUNT = 100
NULL_SEED = 20260930
ANCHOR_COLUMN = 18
LAGS = np.arange(-15, 16, dtype=np.int64)


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise AuditError(f"{path}:{line_no} is not an object")
            rows.append(item)
    return rows


def _safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if path != root and root not in path.parents:
        raise AuditError(f"artifact escapes the sealed run: {relative}")
    if not path.is_file():
        raise AuditError(f"missing artifact: {relative}")
    return path


def _speaker(sample_id: str) -> str:
    match = re.match(r"^(S\d+)_", sample_id)
    if not match:
        raise AuditError(f"sample ID lacks frozen speaker prefix: {sample_id}")
    return match.group(1)


def _support(count: int, guard: int) -> np.ndarray:
    start = min(guard, count)
    stop = max(start, count - guard)
    return np.arange(start, stop, dtype=np.int32)


def _supports(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard15": _support(count, 15),
        "guard20": _support(count, 20),
    }


def _observed_lag(cross: np.ndarray, zero: np.ndarray, query_n: int, audio_cutoff: int) -> np.ndarray:
    cross = np.asarray(cross, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    if cross.ndim != 2 or zero.shape != (cross.shape[0],):
        raise AuditError("saved full V×A matrix/zero vector shapes disagree")
    if query_n < 1 or query_n > cross.shape[0] or not 1 <= audio_cutoff <= cross.shape[1]:
        raise AuditError("observed query or audio cutoff is outside saved matrix")
    source = np.arange(query_n, dtype=np.int64)[:, None] + LAGS[None, :]
    valid = (source >= 0) & (source < audio_cutoff)
    clipped = np.clip(source, 0, audio_cutoff - 1)
    gathered = cross[np.arange(query_n, dtype=np.int64)[:, None], clipped]
    return np.where(valid, gathered, zero[:query_n, None]).astype(np.float32, copy=False)


def _null_lag(cross: np.ndarray, zero: np.ndarray, query_n: int, audio_cutoff: int, pi: np.ndarray) -> np.ndarray:
    cross = np.asarray(cross, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    if len(pi) != cross.shape[1] or audio_cutoff > len(pi):
        raise AuditError("permutation does not span the complete saved A domain")
    source = np.arange(query_n, dtype=np.int64)[:, None] + LAGS[None, :]
    valid = (source >= 0) & (source < audio_cutoff)
    clipped = np.clip(source, 0, audio_cutoff - 1)
    permuted = pi[clipped]
    gathered = cross[np.arange(query_n, dtype=np.int64)[:, None], permuted]
    return np.where(valid, gathered, zero[:query_n, None]).astype(np.float32, copy=False)


def _metrics(matrix: np.ndarray, supports: dict[str, np.ndarray]) -> dict[str, dict[str, float] | None]:
    output: dict[str, dict[str, float] | None] = {}
    matrix = np.asarray(matrix, dtype=np.float32)
    for support_name, indices in supports.items():
        if len(indices) == 0:
            output[support_name] = None
            continue
        curve = matrix[indices.astype(np.int64)].mean(axis=0, dtype=np.float32).astype(np.float64)
        d_min = float(np.min(curve))
        b = float(np.median(curve))
        output[support_name] = {
            "C": b - d_min,
            "C_anchor": b - float(curve[ANCHOR_COLUMN]),
            "B": b,
            "D": d_min,
            "search_gain": float(curve[15]) - d_min,
        }
    return output


def _permutations(audio_n: int, ordered_index: int) -> tuple[list[np.ndarray], list[str]]:
    if audio_n < 1:
        raise AuditError("cannot permute an empty full-A domain")
    generator = np.random.Generator(np.random.PCG64(NULL_SEED + ordered_index))
    perms: list[np.ndarray] = []
    hashes: list[str] = []
    target = np.arange(audio_n, dtype=np.int64)
    for _ in range(NULL_COUNT):
        pi = generator.permutation(audio_n).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(pi), target):
            raise AuditError("generated permutation does not preserve the full A multiset")
        perms.append(pi)
        hashes.append(hashlib.sha256(np.asarray(pi, dtype="<i8").tobytes(order="C")).hexdigest())
    return perms, hashes


def _summary(values: dict[str, float | None], samples: list[str], speakers: dict[str, str]) -> dict[str, Any]:
    kept = {sample: float(values[sample]) for sample in samples if values.get(sample) is not None and np.isfinite(float(values[sample]))}
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample, value in kept.items():
        grouped[speakers[sample]].append(value)
    complete = {speaker: float(np.mean(items, dtype=np.float64)) for speaker, items in grouped.items() if len(items) == 2}
    missing_samples = sorted(set(samples) - set(kept))
    missing_speakers = sorted(set(speakers.values()) - set(complete))
    if complete:
        speaker_ids = sorted(complete)
        vector = np.asarray([complete[s] for s in speaker_ids], dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        idx = rng.integers(0, len(vector), size=(BOOTSTRAP_DRAWS, len(vector)), endpoint=False)
        means = vector[idx].mean(axis=1, dtype=np.float64)
        lo, hi = np.quantile(means, [0.005, 0.995], method="linear")
        estimate = float(vector.mean(dtype=np.float64))
        ci = [float(lo), float(hi)]
    else:
        estimate, ci = None, None
    return {
        "intended_clip_count": len(samples),
        "observed_clip_count": len(kept),
        "complete_two_clip_speaker_count": len(complete),
        "intended_speaker_count": len(set(speakers.values())),
        "missing_clip_ids": missing_samples,
        "missing_or_incomplete_speakers": missing_speakers,
        "complete_pair_equal_speaker_mean": estimate,
        "speaker_bootstrap_99_ci": ci,
        "bootstrap_draws": BOOTSTRAP_DRAWS if complete else None,
        "bootstrap_seed": BOOTSTRAP_SEED if complete else None,
        "eligible_for_intended_denominator": len(kept) == len(samples) and len(complete) == len(set(speakers.values())),
    }


def _mean_null(permutations: np.ndarray | None) -> float | None:
    if permutations is None or permutations.size == 0 or not np.isfinite(permutations).all():
        return None
    return float(np.mean(permutations, dtype=np.float64))


def _draw_equal_speaker_means(
    values_by_sample: dict[str, np.ndarray | None], samples: list[str], speakers: dict[str, str]
) -> list[float | None]:
    counts = [len(v) for v in values_by_sample.values() if v is not None]
    if not counts:
        return []
    draw_count = counts[0]
    if any(count != draw_count for count in counts):
        raise AuditError("permutation count changed across rows in one contrast")
    output: list[float | None] = []
    for p in range(draw_count):
        vals = {sample: (None if values_by_sample.get(sample) is None else float(values_by_sample[sample][p])) for sample in samples}
        grouped: dict[str, list[float]] = defaultdict(list)
        for sample, value in vals.items():
            if value is not None and np.isfinite(value):
                grouped[speakers[sample]].append(value)
        complete = [float(np.mean(items, dtype=np.float64)) for items in grouped.values() if len(items) == 2]
        output.append(float(np.mean(complete, dtype=np.float64)) if complete else None)
    return output


def _contrast_report(
    observed: dict[str, float | None],
    null_draws: dict[str, np.ndarray | None],
    samples: list[str],
    speakers: dict[str, str],
) -> dict[str, Any]:
    null_mean = {sample: _mean_null(null_draws.get(sample)) for sample in samples}
    excess: dict[str, float | None] = {}
    for sample in samples:
        ov = observed.get(sample)
        nv = null_mean.get(sample)
        excess[sample] = None if ov is None or nv is None else float(ov - nv)
    return {
        "observed": _summary(observed, samples, speakers),
        "mean_null": _summary(null_mean, samples, speakers),
        "observed_minus_mean_null": _summary(excess, samples, speakers),
        "null_equal_speaker_mean_by_permutation": _draw_equal_speaker_means(null_draws, samples, speakers),
    }


def _load_sealed_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    expected_files = {
        "evaluation_protocol.json": PROTOCOL_SHA256,
        "evaluation_rows.jsonl": ROW_PLAN_SHA256,
        "evaluation_results.jsonl": RESULTS_SHA256,
        "evaluation_status.json": STATUS_SHA256,
        "all_files_sha256.json": ALL_FILES_SHA256,
        "code_seal.json": CODE_SEAL_SHA256,
    }
    for name, expected in expected_files.items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"sealed run file hash mismatch or missing file: {name}")
    protocol = json.loads((run_dir / "evaluation_protocol.json").read_text(encoding="utf-8"))
    rows = _jsonl(run_dir / "evaluation_rows.jsonl")
    results = _jsonl(run_dir / "evaluation_results.jsonl")
    if len(rows) != EXPECTED_ROWS or len(results) != EXPECTED_ROWS:
        raise AuditError(f"expected {EXPECTED_ROWS} planned/result rows, saw {len(rows)}/{len(results)}")
    plan_by_id = {str(row["row_id"]): row for row in rows}
    result_by_id = {str(row["row_id"]): row for row in results}
    if len(plan_by_id) != EXPECTED_ROWS or len(result_by_id) != EXPECTED_ROWS or set(plan_by_id) != set(result_by_id):
        raise AuditError("sealed female plan/result row identities are incomplete or duplicated")
    for row_id in plan_by_id:
        if result_by_id[row_id].get("status") != "scored":
            raise AuditError(f"planned female row is not scored: {row_id}")
    return rows, result_by_id, protocol


def analyze(run_dir: Path, consumer_summary: Path, output_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    consumer_summary = consumer_summary.resolve(strict=True)
    if sha256_file(consumer_summary) != SUMMARY_SHA256:
        raise AuditError("female consumer summary SHA does not match the completed independent audit")
    rows, results_by_id, protocol = _load_sealed_inputs(run_dir)
    plans = {str(row["row_id"]): row for row in rows}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    speakers: dict[str, str] = {}
    ordered: dict[str, int] = {}
    for row in rows:
        if row.get("family") != "fourcell" or row.get("face_id") != "primary":
            continue
        sample = str(row.get("sample_id"))
        source = str(row.get("source"))
        cell = str(row.get("condition"))
        if source not in SOURCES or cell not in CELLS:
            raise AuditError(f"unexpected female four-cell identity: {row.get('row_id')}")
        visual_level = "fixed" if cell in ("q10", "q11") else "raw"
        audio_level = "fixed" if cell in ("q01", "q11") else "raw"
        expected_visual = f"{source}_{visual_level}"
        expected_audio = f"{source}_audio_embeddings_{audio_level}_float32"
        if str(row.get("visual_id")) != expected_visual or str(row.get("audio_key")) != expected_audio:
            raise AuditError(f"female four-cell visual/audio mapping differs from frozen q definition: {row.get('row_id')}")
        key = (sample, source, cell)
        if key in selected:
            raise AuditError(f"duplicate female source/four-cell row: {key}")
        selected[key] = row
        speaker = str(row.get("speaker_id"))
        if speaker != _speaker(sample):
            raise AuditError(f"sample/speaker identity mismatch: {sample}")
        speakers[sample] = speaker
        row_order = int(row["ordered_eval_index"])
        if sample in ordered and ordered[sample] != row_order:
            raise AuditError(f"ordered index changed within sample {sample}")
        ordered[sample] = row_order
    samples = sorted(speakers)
    if len(samples) != EXPECTED_CLIPS or len(set(speakers.values())) != EXPECTED_SPEAKERS:
        raise AuditError("female four-cell input is not the frozen 40-clip/20-speaker cohort")
    expected_keys = {(s, source, cell) for s in samples for source in SOURCES for cell in CELLS}
    if set(selected) != expected_keys:
        raise AuditError(f"female four-cell matrix is incomplete: {len(selected)} of {len(expected_keys)}")
    if int(protocol.get("row_count", EXPECTED_ROWS)) != EXPECTED_ROWS:
        raise AuditError("female protocol row count differs from 640")

    observed: dict[tuple[str, str, str, str, str, str, str], float | None] = {}
    null_draws: dict[tuple[str, str, str, str, str, str, str], np.ndarray | None] = {}
    row_coverage: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []
    artifact_count = 0

    # A source-specific permutation is reused across all four cells; N and T
    # have distinct full-A domains when their embedding lengths differ.
    for sample in samples:
        for source in SOURCES:
            source_rows = {cell: selected[(sample, source, cell)] for cell in CELLS}
            loaded: dict[str, dict[str, np.ndarray]] = {}
            result_group: dict[str, dict[str, Any]] = {}
            n_values: set[int] = set()
            a_domains: set[int] = set()
            for cell, plan in source_rows.items():
                row_id = str(plan["row_id"])
                result = results_by_id[row_id]
                for field in ("sample_id", "speaker_id", "family", "source", "condition", "face_id"):
                    if str(plan.get(field)) != str(result.get(field)):
                        raise AuditError(f"result row identity differs at {field}: {row_id}")
                anchor = result.get("anchor")
                if anchor is not None and (
                    int(anchor.get("column", -1)) != ANCHOR_COLUMN
                    or int(anchor.get("k", -999)) != 3
                    or int(anchor.get("official_offset_frames", anchor.get("official_syncnet_offset", -999))) != -3
                ):
                    raise AuditError(f"female row anchor differs from frozen male CAL k=3 anchor: {row_id}")
                if str(result.get("score_npz")) != str(plan.get("score_npz")):
                    raise AuditError(f"result/plan score path differs: {row_id}")
                score_path = _safe_artifact(run_dir, str(plan["score_npz"]))
                score_sha = sha256_file(score_path)
                if score_sha != result.get("score_npz_sha256"):
                    raise AuditError(f"score NPZ SHA differs from sealed result row: {row_id}")
                with np.load(score_path, allow_pickle=False) as archive:
                    need = {
                        "distance_matrix_raw_float32", "distance_matrix_unit_float32",
                        "cross_distance_raw_float32", "cross_distance_unit_float32",
                        "cross_distance_to_zero_raw_float32", "cross_distance_to_zero_unit_float32",
                        "common_support_indices_int32", "full_support_indices_int32", "valid_support_indices_int32",
                        "guard15_indices_int32", "guard20_indices_int32", "lag_columns_int32", "lag_k_int32",
                    }
                    if not need.issubset(archive.files):
                        raise AuditError(f"female score NPZ lacks required arrays: {row_id}")
                    loaded[cell] = {key: archive[key] for key in need}
                result_group[cell] = result
                n = int(result.get("common_support_n", -1))
                if n < 1 or int(result.get("scoring_audio_domain_n", -1)) != n:
                    raise AuditError(f"female result row has invalid common support: {row_id}")
                n_values.add(n)
                a_domains.add(int(loaded[cell]["cross_distance_raw_float32"].shape[1]))
                artifact_count += 1
            if len(n_values) != 1 or len(a_domains) != 1:
                raise AuditError(f"four female cells do not share query/A support for {sample}/{source}")
            n = next(iter(n_values))
            audio_n = next(iter(a_domains))
            if audio_n < n:
                raise AuditError("saved full-A domain is shorter than common scoring support")
            support_arrays = _supports(n)
            permutation_list, permutation_hashes = _permutations(audio_n, ordered[sample])
            permutation_records.append({
                "sample_id": sample,
                "source": source,
                "ordered_eval_index": ordered[sample],
                "seed": NULL_SEED + ordered[sample],
                "permutation_domain_n": audio_n,
                "permutation_count": NULL_COUNT,
                "permutation_sha256": permutation_hashes,
                "reused_for_cells": list(CELLS),
            })
            for cell in CELLS:
                row_id = str(source_rows[cell]["row_id"])
                result = result_group[cell]
                arrays = loaded[cell]
                if int(result.get("audio_row_domain_n", -1)) != audio_n:
                    raise AuditError(f"A domain metadata differs from NPZ for {row_id}")
                if int(result.get("visual_query_domain_n", -1)) != arrays["cross_distance_raw_float32"].shape[0]:
                    raise AuditError(f"V domain metadata differs from NPZ for {row_id}")
                if not np.array_equal(arrays["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(arrays["lag_k_int32"], LAGS.astype(np.int32)):
                    raise AuditError(f"lag mapping differs from frozen 31-column support for {row_id}")
                if not np.array_equal(arrays["common_support_indices_int32"], np.arange(n, dtype=np.int32)):
                    raise AuditError(f"common query support differs from 0..n-1 for {row_id}")
                for support_name, expected in support_arrays.items():
                    key = f"{support_name}_support_indices_int32" if support_name in ("full", "valid") else f"{support_name}_indices_int32"
                    if not np.array_equal(arrays[key], expected):
                        raise AuditError(f"{support_name} support indices differ from frozen rule for {row_id}")

                for geometry in GEOMETRIES:
                    matrix = arrays[f"distance_matrix_{geometry}_float32"]
                    cross = arrays[f"cross_distance_{geometry}_float32"]
                    zero = arrays[f"cross_distance_to_zero_{geometry}_float32"]
                    paired_cross = arrays[f"cross_distance_{'unit' if geometry == 'raw' else 'raw'}_float32"]
                    paired_zero = arrays[f"cross_distance_to_zero_{'unit' if geometry == 'raw' else 'raw'}_float32"]
                    paired_matrix = arrays[f"distance_matrix_{'unit' if geometry == 'raw' else 'raw'}_float32"]
                    if matrix.dtype != np.float32 or matrix.shape != (n, 31):
                        raise AuditError(f"saved observed lag matrix shape/dtype invalid: {row_id}/{geometry}")
                    if cross.dtype != np.float32 or zero.dtype != np.float32 or not np.isfinite(cross).all() or not np.isfinite(zero).all():
                        raise AuditError(f"saved full cross matrix/zero vector invalid: {row_id}/{geometry}")
                    if paired_cross.shape != cross.shape or paired_zero.shape != zero.shape or paired_matrix.shape != matrix.shape:
                        raise AuditError(f"raw/unit saved score domains differ: {row_id}")
                    legacy_rebuilt = _observed_lag(cross, zero, n, n)
                    if not np.array_equal(matrix, legacy_rebuilt):
                        raise AuditError(f"legacy observed lag matrix does not rebuild: {row_id}/{geometry}")
                    full_a_observed = _observed_lag(cross, zero, n, audio_n)
                    observed_by_padding = {
                        "legacy_query_cutoff": matrix,
                        "full_A_cutoff": full_a_observed,
                    }
                    metrics_by_padding = {
                        padding: _metrics(observed_matrix, support_arrays)
                        for padding, observed_matrix in observed_by_padding.items()
                    }
                    null_by_padding = {
                        padding: {
                            support_name: {
                                metric: (None if metrics_by_padding[padding][support_name] is None else np.empty(NULL_COUNT, dtype=np.float64))
                                for metric in METRICS
                            }
                            for support_name in SUPPORTS
                        }
                        for padding in PADDING
                    }
                    for p_index, pi in enumerate(permutation_list):
                        for padding in PADDING:
                            audio_cutoff = n if padding == "legacy_query_cutoff" else audio_n
                            null_matrix = _null_lag(cross, zero, n, audio_cutoff, pi)
                            null_metrics = _metrics(null_matrix, support_arrays)
                            for support_name in SUPPORTS:
                                support_metrics = null_metrics[support_name]
                                for metric in METRICS:
                                    values = null_by_padding[padding][support_name][metric]
                                    if values is None:
                                        if support_metrics is not None:
                                            raise AuditError("null support unexpectedly available when observed support is empty")
                                        continue
                                    if support_metrics is None:
                                        raise AuditError("null support unexpectedly unavailable for observed support")
                                    values[p_index] = float(support_metrics[metric])
                    for padding, observed_matrix in observed_by_padding.items():
                        metrics_obs = metrics_by_padding[padding]
                        for support_name, metric_dict in metrics_obs.items():
                            for metric in METRICS:
                                key = (sample, source, cell, geometry, padding, support_name, metric)
                                if metric_dict is None:
                                    observed[key] = None
                                    null_draws[key] = None
                                    continue
                                observed[key] = float(metric_dict[metric])
                                null_draws[key] = null_by_padding[padding][support_name][metric]
                artifact_hash = str(result["score_npz_sha256"])
                row_coverage.append({
                    "row_id": row_id,
                    "sample_id": sample,
                    "source": source,
                    "condition": cell,
                    "score_npz_sha256": artifact_hash,
                    "common_support_n": n,
                    "full_A_n": audio_n,
                    "support_counts": {name: int(len(idx)) for name, idx in support_arrays.items()},
                })

    if artifact_count != 320:
        raise AuditError(f"female addendum requires 320 native four-cell matrices, got {artifact_count}")

    def report_for_expression(
        expression_name: str,
        cell_terms: tuple[tuple[str, str, float], ...],
        source_terms: tuple[tuple[str, float], ...],
        geometry: str,
        padding: str,
        support_name: str,
        metric: str,
    ) -> dict[str, Any]:
        obs: dict[str, float | None] = {}
        null: dict[str, np.ndarray | None] = {}
        for sample in samples:
            obs_values: list[float] = []
            null_vectors: list[np.ndarray] = []
            missing = False
            for source, source_weight in source_terms:
                for cell, cell_weight, sign in cell_terms:
                    key = (sample, source, cell, geometry, padding, support_name, metric)
                    ov = observed.get(key)
                    nv = null_draws.get(key)
                    if ov is None or nv is None:
                        missing = True
                        break
                    weight = source_weight * cell_weight * sign
                    obs_values.append(weight * float(ov))
                    null_vectors.append(weight * nv)
                if missing:
                    break
            if missing:
                obs[sample] = None
                null[sample] = None
            else:
                obs[sample] = float(sum(obs_values))
                null[sample] = np.sum(np.stack(null_vectors, axis=0), axis=0, dtype=np.float64)
        return _contrast_report(obs, null, samples, speakers)

    reports: dict[str, Any] = {
        "cell_native_and_processed": {},
        "generator_G": {},
        "T_minus_N_by_cell": {},
        "paired_T_minus_N_gap_change_q00_to_q11": {},
    }
    for padding in PADDING:
        reports["cell_native_and_processed"][padding] = {}
        reports["generator_G"][padding] = {}
        reports["T_minus_N_by_cell"][padding] = {}
        reports["paired_T_minus_N_gap_change_q00_to_q11"][padding] = {}
        for geometry in GEOMETRIES:
            reports["cell_native_and_processed"][padding][geometry] = {}
            reports["generator_G"][padding][geometry] = {}
            reports["T_minus_N_by_cell"][padding][geometry] = {}
            reports["paired_T_minus_N_gap_change_q00_to_q11"][padding][geometry] = {}
            for support_name in SUPPORTS:
                reports["cell_native_and_processed"][padding][geometry][support_name] = {}
                reports["generator_G"][padding][geometry][support_name] = {}
                reports["T_minus_N_by_cell"][padding][geometry][support_name] = {}
                reports["paired_T_minus_N_gap_change_q00_to_q11"][padding][geometry][support_name] = {}
                for metric in METRICS:
                    reports["cell_native_and_processed"][padding][geometry][support_name][metric] = {}
                    for source in SOURCES:
                        reports["cell_native_and_processed"][padding][geometry][support_name][metric][source] = {}
                        for cell in CELLS:
                            reports["cell_native_and_processed"][padding][geometry][support_name][metric][source][cell] = report_for_expression(
                                f"{source}_{cell}", ((cell, 1.0, 1.0),), ((source, 1.0),), geometry, padding, support_name, metric
                            )
                    reports["generator_G"][padding][geometry][support_name][metric] = {}
                    for source in SOURCES:
                        reports["generator_G"][padding][geometry][support_name][metric][source] = {
                            "q10_minus_q00": report_for_expression(
                                f"{source}_G_q10_minus_q00", (("q10", 1.0, 1.0), ("q00", 1.0, -1.0)), ((source, 1.0),), geometry, padding, support_name, metric
                            ),
                            "q11_minus_q01": report_for_expression(
                                f"{source}_G_q11_minus_q01", (("q11", 1.0, 1.0), ("q01", 1.0, -1.0)), ((source, 1.0),), geometry, padding, support_name, metric
                            ),
                        }
                    reports["T_minus_N_by_cell"][padding][geometry][support_name][metric] = {}
                    for cell in CELLS:
                        reports["T_minus_N_by_cell"][padding][geometry][support_name][metric][cell] = report_for_expression(
                            f"T_minus_N_{cell}", ((cell, 1.0, 1.0),), (("tts", 1.0), ("natural", -1.0)), geometry, padding, support_name, metric
                        )
                    reports["paired_T_minus_N_gap_change_q00_to_q11"][padding][geometry][support_name][metric] = report_for_expression(
                        "(Tq00-Nq00)-(Tq11-Nq11)",
                        (("q00", 1.0, 1.0), ("q11", 1.0, -1.0)),
                        (("tts", 1.0), ("natural", -1.0)), geometry, padding, support_name, metric,
                    )

    output = {
        "schema_version": 1,
        "analysis_type": "independent_female_native_generator_cached_temporal_null_addendum",
        "attempt": "02",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "CPU-only cached-matrix analysis; no model forwards or score artifacts modified",
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "input_seals": {
            "run_dir": str(run_dir),
            "protocol_sha256": PROTOCOL_SHA256,
            "row_plan_sha256": ROW_PLAN_SHA256,
            "results_sha256": RESULTS_SHA256,
            "status_sha256": STATUS_SHA256,
            "all_files_sha256": ALL_FILES_SHA256,
            "code_seal_sha256": CODE_SEAL_SHA256,
            "prior_independent_consumer_summary_sha256": SUMMARY_SHA256,
            "prior_summary_path": str(consumer_summary),
        },
        "protocol": {
            "clips": EXPECTED_CLIPS,
            "speakers": EXPECTED_SPEAKERS,
            "intended_denominator_retained": True,
            "conditions": list(CELLS),
            "sources": list(SOURCES),
            "face": "primary",
            "geometries": list(GEOMETRIES),
            "supports": list(SUPPORTS),
            "metrics": list(METRICS),
            "anchor_column": ANCHOR_COLUMN,
            "null_generator": "PCG64(seed=20260930+ordered_eval_index), 100 full-A permutations per source; reused for the four cells within source",
            "legacy_padding_pair": "observed and permuted score both use common_support_n as A cutoff; permutation remains over the complete A domain",
            "full_A_padding_pair": "observed and permuted score both use the complete saved A domain as cutoff",
            "speaker_bootstrap": "two clips averaged within speaker; equal-speaker mean; 20,000 draws, seed 20260930, 99% percentile interval",
            "search_gain": "zero-lag column 15 mean minus D",
        },
        "verification": {
            "planned_and_scored_rows": EXPECTED_ROWS,
            "independently_hashed_native_score_artifacts": artifact_count,
            "clips": len(samples),
            "speakers": len(set(speakers.values())),
            "permutation_domains": len(permutation_records),
            "short_supports": {
                support_name: sum(1 for record in row_coverage if record["support_counts"][support_name] == 0)
                for support_name in SUPPORTS
            },
            "all_row_artifact_sha_checks_passed": True,
            "all_legacy_score_matrices_rebuilt_exactly": True,
            "full_A_observed_and_null_rebuilt_with_matched_boundary": True,
        },
        "permutation_records": permutation_records,
        "row_coverage": row_coverage,
        "results": reports,
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
    parser.add_argument("--consumer-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = analyze(args.run_dir, args.consumer_summary, args.output)
    print(json.dumps({"status": output["status"], "artifacts": output["verification"]["independently_hashed_native_score_artifacts"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
