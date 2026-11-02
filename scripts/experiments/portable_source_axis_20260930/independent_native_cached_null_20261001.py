#!/usr/bin/env python3
"""Independent CPU audit of the sealed native four-cell cached score run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

PROTOCOL_SHA = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
ROWS_SHA = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
RESULTS_SHA = "fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa"
PARENT_SHA = "d5fbfa99865b958a006f504e0dd7c6e5120ec194651aa639acad4d540528b862"
RECOVERY_SHA = "55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417"
SPEC_SHA = "b11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd"
ATTEMPT08_SHA = "55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c"
ANCHOR = {"column": 18, "k": 3, "official_offset": -3}
CONDITIONS = ("q00", "q10", "q01", "q11")
SOURCES = ("natural", "tts")
FACES = ("primary", "secondary")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
LAGS = np.arange(-15, 16, dtype=np.int32)
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
NULL_DRAWS = 100
EPS = np.float32(1e-6)
ARRAY_ATOL = 5e-4


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise AuditError(f"expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise AuditError(f"non-object JSONL at {path}:{line_no}")
                rows.append(row)
    return rows


def safe_path(root: Path, ref: str) -> Path:
    part = PurePosixPath(str(ref))
    if part.is_absolute() or not part.parts or any(item in ("", ".", "..") for item in part.parts):
        raise AuditError(f"unsafe artifact reference: {ref}")
    path = root.joinpath(*part.parts).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AuditError(f"artifact escapes run root: {ref}") from exc
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def supports_for(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    guard20 = np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32)
    guard15 = np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32)
    return {"full": full, "valid": full.copy(), "guard20": guard20, "guard15": guard15}


def row_unit(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    norm = np.sqrt(np.sum(value * value, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (value / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def cross_and_zero(visual: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v, a = np.asarray(visual, dtype=np.float32), np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or not len(v) or not len(a) or v.shape[1] != a.shape[1]:
        raise AuditError("V/A embeddings are not nonempty aligned matrices")
    cross = np.empty((len(v), len(a)), dtype=np.float32)
    for start in range(0, len(v), 4):
        stop = min(start + 4, len(v))
        delta = v[start:stop, None, :] - a[None, :, :]
        np.add(delta, EPS, out=delta)
        np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32), out=cross[start:stop])
    zero_v = v.copy()
    np.add(zero_v, EPS, out=zero_v)
    zero = np.sqrt(np.sum(zero_v * zero_v, axis=1, dtype=np.float32), dtype=np.float32)
    return cross, zero.astype(np.float32, copy=False)


def lag_matrix(cross: np.ndarray, zero: np.ndarray, query_n: int, scoring_audio_n: int) -> np.ndarray:
    x, z = np.asarray(cross, dtype=np.float32), np.asarray(zero, dtype=np.float32)
    if query_n < 1 or query_n > x.shape[0] or scoring_audio_n < 1 or scoring_audio_n > x.shape[1] or z.shape != (x.shape[0],):
        raise AuditError("invalid score query/audio domain")
    out = np.empty((query_n, len(LAGS)), dtype=np.float32)
    for q in range(query_n):
        for col, lag in enumerate(LAGS):
            j = q + int(lag)
            out[q, col] = x[q, j] if 0 <= j < scoring_audio_n else z[q]
    return out


def metrics_by_support(matrix: np.ndarray, supports: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, indices in supports.items():
        if len(indices) == 0:
            result[name] = {"status": "too_short", "support_n": 0}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        b, d = float(np.median(curve)), float(np.min(curve))
        result[name] = {
            "status": "ok", "support_n": int(len(indices)), "curve": curve,
            "C": b - d, "C_anchor": b - float(curve[18]), "B": b, "D": d,
            "search_gain": float(curve[15]) - d,
        }
    return result


def compare_float32(saved: np.ndarray, expected: np.ndarray, name: str, max_diffs: dict[str, float]) -> None:
    value = np.asarray(saved)
    if value.dtype != np.float32 or value.shape != expected.shape or not np.isfinite(value).all():
        raise AuditError(f"{name} has wrong dtype/shape or nonfinite values")
    delta = float(np.max(np.abs(value - expected), initial=0.0))
    max_diffs[name] = max(max_diffs.get(name, 0.0), delta)
    if delta > ARRAY_ATOL:
        raise AuditError(f"{name} differs by {delta:.8g} > {ARRAY_ATOL}")


def compare_saved_metrics(saved: dict[str, Any], rebuilt: dict[str, dict[str, Any]], row_id: str) -> None:
    for support in SUPPORTS:
        old = saved.get(support)
        new = rebuilt.get(support)
        if not isinstance(old, dict) or not isinstance(new, dict) or old.get("status") != new.get("status"):
            raise AuditError(f"{row_id}: saved {support} status differs")
        if int(old.get("support_n", -1)) != int(new.get("support_n", -2)):
            raise AuditError(f"{row_id}: saved {support} count differs")
        if new["status"] == "too_short":
            continue
        for key in ("C_best", "C_fixed_anchor", "B_median", "D_min"):
            expected_key = {"C_best": "C", "C_fixed_anchor": "C_anchor", "B_median": "B", "D_min": "D"}[key]
            a, b = float(old[key]), float(new[expected_key])
            if abs(a - b) > 5e-4:
                raise AuditError(f"{row_id}: {support}.{key} differs by {abs(a-b):.8g}")
        curve = np.asarray(old.get("curve"), dtype=np.float64)
        if curve.shape != (31,) or np.max(np.abs(curve - new["curve"]), initial=0.0) > 5e-4:
            raise AuditError(f"{row_id}: saved {support} lag curve differs")


def permutation_list(n_audio: int, ordered_index: int) -> tuple[list[np.ndarray], list[str]]:
    if n_audio < 1:
        raise AuditError("empty audio permutation domain")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED + ordered_index))
    expected = np.arange(n_audio, dtype=np.int64)
    vectors, hashes = [], []
    for _ in range(NULL_DRAWS):
        permutation = rng.permutation(n_audio).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(permutation), expected):
            raise AuditError("PCG64 draw is not a full-domain audio permutation")
        vectors.append(permutation)
        hashes.append(hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest())
    return vectors, hashes


def null_metrics(cross: np.ndarray, zero: np.ndarray, queries: np.ndarray, permutations: list[np.ndarray]) -> dict[str, np.ndarray] | None:
    if len(queries) == 0:
        return None
    x, boundary = np.asarray(cross, dtype=np.float32), np.asarray(zero, dtype=np.float32)
    q = np.asarray(queries, dtype=np.int64)
    source = q[:, None] + LAGS.astype(np.int64)[None, :]
    in_bounds = (source >= 0) & (source < x.shape[1])
    clipped = np.clip(source, 0, x.shape[1] - 1)
    if boundary.shape != (x.shape[0],) or np.any(q < 0) or np.any(q >= x.shape[0]):
        raise AuditError("malformed null distance/query arrays")
    for pi in permutations:
        if np.asarray(pi).shape != (x.shape[1],) or not np.array_equal(np.sort(pi), np.arange(x.shape[1], dtype=np.int64)):
            raise AuditError("temporal null does not permute the full A domain")
    source_rows = np.stack([pi[clipped] for pi in permutations], axis=0)
    query_rows = np.broadcast_to(q[None, :, None], source_rows.shape)
    gathered = x[query_rows, source_rows]
    gathered = np.where(in_bounds[None, :, :], gathered, boundary[q][None, :, None]).astype(np.float32, copy=False)
    curves = gathered.mean(axis=1, dtype=np.float32).astype(np.float64)
    b = np.median(curves, axis=1)
    d = np.min(curves, axis=1)
    return {"C": b-d, "C_anchor": b-curves[:, 18], "B": b, "D": d, "search_gain": curves[:, 15]-d}


class Bootstrapper:
    def __init__(self) -> None:
        self.rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        self.indices: dict[tuple[str, ...], np.ndarray] = {}

    def summarize(self, values: dict[str, float], speakers: dict[str, str], samples: list[str]) -> dict[str, Any]:
        by_speaker: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            if sample in values:
                by_speaker[speakers[sample]].append(float(values[sample]))
        complete = {speaker: float(np.mean(items, dtype=np.float64)) for speaker, items in by_speaker.items() if len(items) == 2}
        names = tuple(sorted(complete))
        mean = float(np.mean([complete[name] for name in names], dtype=np.float64)) if names else None
        ci = None
        if names:
            if names not in self.indices:
                self.indices[names] = self.rng.integers(0, len(names), size=(BOOTSTRAP_DRAWS, len(names)), dtype=np.int32)
            estimates = np.asarray([complete[name] for name in names], dtype=np.float64)
            draws = estimates[self.indices[names]].mean(axis=1, dtype=np.float64)
            ci = [float(x) for x in np.quantile(draws, [0.005, 0.995], method="linear")]
        return {
            "equal_speaker_mean": mean, "speaker_bootstrap_99_ci": ci,
            "complete_speakers": len(names), "complete_clips": 2*len(names),
            "available_clip_count": len(values), "intended_clip_count": len(samples),
            "missing_clip_count": len(samples)-len(values), "intended_speaker_count": len(set(speakers.values())),
            "speaker_values": complete,
            "eligible_for_intended_cohort": len(names) == 40 and len(values) == 80,
        }


def summarize_effect(
    observed: dict[str, float], null_vectors: dict[str, np.ndarray], speakers: dict[str, str], samples: list[str],
    metric: str, bootstrapper: Bootstrapper,
) -> dict[str, Any]:
    obs_summary = bootstrapper.summarize(observed, speakers, samples)
    null_mean = {sample: float(np.mean(vector, dtype=np.float64)) for sample, vector in null_vectors.items() if sample in observed}
    null_summary = bootstrapper.summarize(null_mean, speakers, samples)
    difference = {sample: observed[sample]-null_mean[sample] for sample in observed.keys() & null_mean.keys()}
    difference_summary = bootstrapper.summarize(difference, speakers, samples)
    group_draws: list[float] = []
    if null_vectors:
        eligible_samples = [sample for sample in samples if sample in null_vectors and len(null_vectors[sample]) == NULL_DRAWS]
        for draw in range(NULL_DRAWS):
            per_sample = {sample: float(null_vectors[sample][draw]) for sample in eligible_samples}
            by_speaker: dict[str, list[float]] = defaultdict(list)
            for sample, value in per_sample.items():
                by_speaker[speakers[sample]].append(value)
            speaker_values = [float(np.mean(values, dtype=np.float64)) for values in by_speaker.values() if len(values) == 2]
            if speaker_values:
                group_draws.append(float(np.mean(speaker_values, dtype=np.float64)))
    observed_mean = obs_summary["equal_speaker_mean"]
    p_upper = None
    if metric == "C" and observed_mean is not None and len(group_draws) == NULL_DRAWS:
        p_upper = float((1 + sum(value >= observed_mean for value in group_draws)) / (NULL_DRAWS + 1))
    return {
        "observed": obs_summary, "mean_null": null_summary,
        "observed_minus_mean_null": difference_summary,
        "null_equal_speaker_mean_by_permutation": group_draws,
        "null_group_mean_percentile_99": [float(x) for x in np.quantile(group_draws, [0.005, 0.995], method="linear")] if group_draws else None,
        "one_sided_p_null_ge_observed": p_upper,
        "null_permutation_count": len(group_draws),
        "null_complete_speaker_count": null_summary["complete_speakers"],
        "support_eligible_for_full_cohort": bool(obs_summary["eligible_for_intended_cohort"] and null_summary["eligible_for_intended_cohort"] and difference_summary["eligible_for_intended_cohort"] and len(group_draws) == NULL_DRAWS),
    }


def signed_map(
    source_values: dict[str, dict[str, float | None]], coefficients: dict[str, float], metric: str,
) -> dict[str, float]:
    out = {}
    for sample, conditions in source_values.items():
        if all(conditions.get(condition) is not None for condition in coefficients):
            out[sample] = float(sum(coeff * float(conditions[condition]) for condition, coeff in coefficients.items()))
    return out


def signed_null_map(
    source_values: dict[str, dict[str, dict[str, np.ndarray | None]]], coefficients: dict[str, float], metric: str,
) -> dict[str, np.ndarray]:
    out = {}
    for sample, conditions in source_values.items():
        arrays = [conditions.get(condition, {}).get(metric) for condition in coefficients]
        if arrays and all(value is not None for value in arrays):
            value = np.zeros(NULL_DRAWS, dtype=np.float64)
            for coefficient, condition in zip(coefficients.values(), coefficients, strict=True):
                value += float(coefficient) * np.asarray(conditions[condition][metric], dtype=np.float64)
            out[sample] = value
    return out


def condition_values(
    observed: dict[str, dict[str, dict[str, float | None]]], face: str, source: str, geometry: str, support: str, metric: str,
) -> dict[str, dict[str, float | None]]:
    return {
        sample: {condition: observed[sample][face][source][condition][geometry][support].get(metric) for condition in CONDITIONS}
        for sample in observed
    }


def condition_null_values(
    nulls: dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray | None]]]]], face: str, source: str, geometry: str, support: str,
) -> dict[str, dict[str, dict[str, np.ndarray | None]]]:
    return {
        sample: {
            condition: nulls[sample][face][source][condition][geometry][support]
            for condition in CONDITIONS
        }
        for sample in nulls
    }


def validate_and_analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    root = args.run_root.resolve(strict=True)
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise AuditError(f"output directory must be empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    protocol_path, rows_path, results_path = (root / name for name in ("evaluation_protocol.json", "evaluation_rows.jsonl", "evaluation_results.jsonl"))
    parent_path, recovery_path = root / "continuation_parent.json", root / "recovery_code_seal.json"
    for path, expected, label in (
        (protocol_path, PROTOCOL_SHA, "retry3 protocol"), (rows_path, ROWS_SHA, "retry3 row plan"),
        (results_path, RESULTS_SHA, "retry3 result ledger"), (parent_path, PARENT_SHA, "continuation parent"),
        (recovery_path, RECOVERY_SHA, "recovery code seal"),
    ):
        observed_sha = sha256_file(path)
        if observed_sha != expected:
            raise AuditError(f"{label} SHA mismatch: {observed_sha}")
    spec_path = args.spec.resolve(strict=True)
    if sha256_file(spec_path) != SPEC_SHA:
        raise AuditError("native diagnostic spec SHA mismatch")
    prior_path = args.attempt08_summary.resolve(strict=True)
    if sha256_file(prior_path) != ATTEMPT08_SHA:
        raise AuditError("attempt08 independent summary SHA mismatch")

    protocol = read_json(protocol_path)
    plan, results = read_jsonl(rows_path), read_jsonl(results_path)
    if len(plan) != 2560 or len(results) != 2560:
        raise AuditError(f"expected sealed 2,560-row plan/results; got {len(plan)}/{len(results)}")
    # The retry3 artifact nests this field under method; the top-level value is
    # absent in the frozen producer protocol.
    if protocol.get("row_count") != 2560 or protocol.get("method", {}).get("global_anchor_column") != ANCHOR["column"]:
        raise AuditError("protocol row count or fixed anchor differs from the frozen run")
    if protocol.get("method", {}).get("level_cells") != {
        "q00": ["RAW", "RAW"], "q10": ["FIXED", "RAW"],
        "q01": ["RAW", "FIXED"], "q11": ["FIXED", "FIXED"],
    }:
        raise AuditError("native four-cell identity differs from the frozen protocol")
    if protocol.get("method", {}).get("lag_domain", {}).get("columns") != 31:
        raise AuditError("protocol lag domain is not the frozen 31 columns")
    if protocol.get("method", {}).get("lag_domain", {}).get("vshift") != 15:
        raise AuditError("protocol lag half-width is not 15")

    plan_by_id = {str(row["row_id"]): row for row in plan}
    result_by_id = {str(row["row_id"]): row for row in results}
    if len(plan_by_id) != 2560 or len(result_by_id) != 2560 or set(plan_by_id) != set(result_by_id):
        raise AuditError("plan/result row IDs are duplicate, missing, or unmatched")
    if sum(row.get("family") == "fourcell" for row in plan) != 1280:
        raise AuditError("expected exactly 1,280 native four-cell rows")
    planned_four = [row for row in plan if row.get("family") == "fourcell"]
    samples = sorted({str(row["sample_id"]) for row in planned_four}, key=lambda sample: min(int(r["ordered_eval_index"]) for r in planned_four if r["sample_id"] == sample))
    speaker_by_sample = {str(row["sample_id"]): str(row["speaker_id"]) for row in planned_four if row.get("source") == "natural" and row.get("face_id") == "primary" and row.get("condition") == "q00"}
    ordered_by_sample = {str(row["sample_id"]): int(row["ordered_eval_index"]) for row in planned_four if row.get("face_id") == "primary" and row.get("source") == "natural" and row.get("condition") == "q00"}
    if len(samples) != 80 or len(speaker_by_sample) != 80 or len(ordered_by_sample) != 80:
        raise AuditError("native cohort does not contain 80 ordered samples")
    speaker_counts = Counter(speaker_by_sample.values())
    if len(speaker_counts) != 40 or set(speaker_counts.values()) != {2}:
        raise AuditError("native cohort does not contain 40 speakers × 2 clips")

    expected_ids = {
        (sample, face, source, condition)
        for sample in samples for face in FACES for source in SOURCES for condition in CONDITIONS
    }
    seen_ids = set()
    by_sample: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for planned in planned_four:
        sample, face, source, condition = (str(planned[key]) for key in ("sample_id", "face_id", "source", "condition"))
        identity = (sample, face, source, condition)
        if identity in seen_ids or identity not in expected_ids:
            raise AuditError(f"unexpected/duplicate native row identity: {identity}")
        seen_ids.add(identity)
        if planned.get("split") != "evaluation" or planned.get("visual_id") != f"{source}_{'raw' if condition in ('q00','q01') else 'fixed'}":
            raise AuditError(f"native row has inconsistent split/video mapping: {planned['row_id']}")
        expected_audio = f"{source}_audio_embeddings_{'raw' if condition in ('q00','q10') else 'fixed'}_float32"
        if planned.get("audio_key") != expected_audio:
            raise AuditError(f"native row has inconsistent audio mapping: {planned['row_id']}")
        actual = result_by_id[str(planned["row_id"])]
        for field in ("ordered_eval_index", "sample_id", "speaker_id", "split", "family", "source", "condition", "face_id", "visual_id", "audio_key", "visual_level", "audio_level"):
            if actual.get(field) != planned.get(field):
                raise AuditError(f"row/result identity mismatch at {planned['row_id']} field {field}")
        if actual.get("status") != "scored":
            raise AuditError(f"native row is not scored: {planned['row_id']} status={actual.get('status')}")
        anchor = actual.get("anchor", {})
        if anchor.get("column") != ANCHOR["column"] or anchor.get("k") != ANCHOR["k"] or anchor.get("official_offset_frames") != ANCHOR["official_offset"]:
            raise AuditError(f"native row anchor differs: {planned['row_id']}")
        by_sample[sample][(face, source, condition)] = {"plan": planned, "result": actual}
    if seen_ids != expected_ids:
        raise AuditError(f"native four-cell matrix incomplete: {len(seen_ids)}/{len(expected_ids)}")

    prior = read_json(prior_path)
    if prior.get("status") != "complete" or prior.get("protocol_sha256") != PROTOCOL_SHA or prior.get("row_manifest_sha256") != ROWS_SHA:
        raise AuditError("attempt08 summary is not complete or does not bind retry3")
    prior_perm = {str(row["sample_id"]): row for row in prior.get("temporal_null", {}).get("permutation_rows", [])}
    if len(prior_perm) != 80:
        raise AuditError(f"attempt08 permutation inventory missing/duplicated: {len(prior_perm)}/80")

    observed: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    nulls: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray | None]]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    cross_records: dict[tuple[str, str, str, str, str, str], dict[str, np.ndarray | int]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    permutation_records: dict[str, dict[str, Any]] = {}
    max_diffs: dict[str, float] = {}
    verification = Counter()

    for sample in samples:
        grid = by_sample[sample]
        index = ordered_by_sample[sample]
        context_refs = {str(grid[(face, source, cond)]["plan"]["context_npz"]) for face in FACES for source in SOURCES for cond in CONDITIONS}
        if len(context_refs) != 1:
            raise AuditError(f"native rows do not share one per-sample A context: {sample}")
        context_ref = next(iter(context_refs))
        context_path = safe_path(root, context_ref)
        context_sha = sha256_file(context_path)
        context_arrays = load_npz(context_path)
        context_json_ref = PurePosixPath(context_ref).with_suffix(".json").as_posix()
        context_json_path = safe_path(root, context_json_ref)
        context_json_sha = sha256_file(context_json_path)
        context_json = read_json(context_json_path)
        context_json_npz_sha = context_json.get("context_npz", {}).get("sha256")
        if context_json_npz_sha is not None and context_json_npz_sha != context_sha:
            raise AuditError(f"context JSON does not bind NPZ: {sample}")
        expected_audio_keys = {f"{source}_audio_embeddings_{level}_float32" for source in SOURCES for level in ("raw", "fixed")}
        if not expected_audio_keys <= set(context_arrays):
            raise AuditError(f"per-sample context misses native A embeddings: {sample}")
        audio: dict[tuple[str, str], np.ndarray] = {}
        for source in SOURCES:
            for level in ("raw", "fixed"):
                value = context_arrays[f"{source}_audio_embeddings_{level}_float32"]
                if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 1024 or not len(value) or not np.isfinite(value).all():
                    raise AuditError(f"malformed cached A embeddings: {sample}/{source}/{level}")
                audio[(source, level)] = value
            if len(audio[(source, "raw")]) != len(audio[(source, "fixed")]):
                raise AuditError(f"raw/fixed A row domain differs within recipient: {sample}/{source}")
        artifact_hashes[f"context:{sample}"] = {"path": context_ref, "sha256": context_sha, "json_path": context_json_ref, "json_sha256": context_json_sha}

        for source in SOURCES:
            ordered_indices = {int(grid[(face, source, cond)]["plan"]["ordered_eval_index"]) for face in FACES for cond in CONDITIONS}
            if ordered_indices != {index}:
                raise AuditError(f"ordered index differs within recipient/source: {sample}/{source}")
            n_audio = len(audio[(source, "raw")])
            permutations, hashes = permutation_list(n_audio, index)
            permutation_records[f"{sample}:{source}"] = {
                "sample_id": sample, "source": source, "ordered_eval_index": index,
                "seed": BOOTSTRAP_SEED + index, "audio_domain_n": n_audio,
                "permutation_count": NULL_DRAWS, "permutation_sha256": hashes,
                "same_permutation_for_raw_fixed_audio": True, "same_permutation_across_faces_and_cells": True,
            }
            if source == "natural":
                prior_row = prior_perm.get(sample)
                if prior_row is None or int(prior_row.get("ordered_eval_index", -1)) != index or int(prior_row.get("seed", -1)) != BOOTSTRAP_SEED + index:
                    raise AuditError(f"attempt08 N permutation identity differs for {sample}")
                if int(prior_row.get("permutation_domain_n", -1)) != n_audio or int(prior_row.get("permutation_count", -1)) != NULL_DRAWS or prior_row.get("permutation_sha256") != hashes:
                    raise AuditError(f"native N raw-A permutations do not match sealed attempt08: {sample}")
                permutation_records[f"{sample}:{source}"]["matches_attempt08_N_raw_domain"] = True

            for face in FACES:
                visual_arrays: dict[str, np.ndarray] = {}
                visual_paths: dict[str, str] = {}
                visual_shas: dict[str, str] = {}
                visual_ids = (f"{source}_raw", f"{source}_fixed")
                for visual_id in visual_ids:
                    row_plan = next(grid[(face, source, condition)]["plan"] for condition in CONDITIONS if grid[(face, source, condition)]["plan"]["visual_id"] == visual_id)
                    ref = str(row_plan["visual_npz"])
                    path = safe_path(root, ref)
                    digest = sha256_file(path)
                    arrays = load_npz(path)
                    if "lip_embeddings_float32" not in arrays or "z_condition_float32" not in arrays:
                        raise AuditError(f"visual archive has missing V/z arrays: {sample}/{face}/{visual_id}")
                    lips = arrays["lip_embeddings_float32"]
                    latent = arrays["z_condition_float32"]
                    if lips.dtype != np.float32 or lips.ndim != 2 or lips.shape[1] != 1024 or not len(lips) or not np.isfinite(lips).all():
                        raise AuditError(f"malformed cached V embeddings: {sample}/{face}/{visual_id}")
                    if latent.dtype != np.float32 or latent.ndim != 2 or latent.shape[1] != 512 or not np.isfinite(latent).all():
                        raise AuditError(f"malformed cached video latent: {sample}/{face}/{visual_id}")
                    for condition in CONDITIONS:
                        row = grid[(face, source, condition)]
                        if row["plan"]["visual_id"] == visual_id:
                            result = row["result"]
                            if result.get("visual_npz_sha256") != digest or result.get("lip_embedding_sha256") != array_sha256(lips):
                                raise AuditError(f"visual file/embedding hash mismatch: {row['plan']['row_id']}")
                    visual_arrays[visual_id] = lips
                    visual_paths[visual_id] = ref
                    visual_shas[visual_id] = digest
                    artifact_hashes[f"visual:{sample}:{face}:{visual_id}"] = {"path": ref, "sha256": digest, "lip_embedding_sha256": array_sha256(lips)}

                query_n = min(*(len(value) for value in visual_arrays.values()), len(audio[(source, "raw")]), len(audio[(source, "fixed")]))
                supports = supports_for(query_n)
                for condition in CONDITIONS:
                    cell = grid[(face, source, condition)]
                    plan_row, result_row = cell["plan"], cell["result"]
                    audio_level = "raw" if condition in ("q00", "q10") else "fixed"
                    audio_value = audio[(source, audio_level)]
                    visual_value = visual_arrays[str(plan_row["visual_id"])]
                    if int(result_row.get("common_support_n", -1)) != query_n or int(result_row.get("audio_row_domain_n", -1)) != len(audio_value) or int(result_row.get("visual_query_domain_n", -1)) != len(visual_value):
                        raise AuditError(f"recorded A/V lengths differ from independent common support: {plan_row['row_id']}")
                    if int(result_row.get("scoring_audio_domain_n", -1)) != query_n:
                        raise AuditError(f"scoring audio domain must equal common truncated support: {plan_row['row_id']}")
                    if result_row.get("context_npz_sha256") != context_sha or result_row.get("context_json_sha256") != context_json_sha:
                        raise AuditError(f"result row context hashes differ from per-sample context: {plan_row['row_id']}")
                    if array_sha256(audio_value) != result_row.get("audio_embedding_sha256"):
                        raise AuditError(f"A embedding array hash differs: {plan_row['row_id']}")
                    score_path = safe_path(root, str(plan_row["score_npz"]))
                    score_sha = sha256_file(score_path)
                    if result_row.get("score_npz_sha256") != score_sha:
                        raise AuditError(f"score NPZ hash differs from result metadata: {plan_row['row_id']}")
                    score = load_npz(score_path)
                    required = {
                        "distance_matrix_raw_float32", "distance_matrix_unit_float32",
                        "cross_distance_raw_float32", "cross_distance_unit_float32",
                        "cross_distance_to_zero_raw_float32", "cross_distance_to_zero_unit_float32",
                        "common_support_indices_int32", "full_support_indices_int32", "valid_support_indices_int32",
                        "guard20_indices_int32", "guard15_indices_int32", "lag_columns_int32", "lag_k_int32",
                    }
                    if not required <= set(score):
                        raise AuditError(f"score NPZ lacks required matrices/supports: {plan_row['row_id']}")
                    support_arrays = {
                        "common_support_indices_int32": np.arange(query_n, dtype=np.int32),
                        "full_support_indices_int32": supports["full"],
                        "valid_support_indices_int32": supports["valid"],
                        "guard20_indices_int32": supports["guard20"],
                        "guard15_indices_int32": supports["guard15"],
                        "lag_columns_int32": np.arange(31, dtype=np.int32),
                        "lag_k_int32": LAGS,
                    }
                    for key, expected in support_arrays.items():
                        if score[key].dtype != np.int32 or not np.array_equal(score[key], expected):
                            raise AuditError(f"saved support/lag array differs for {plan_row['row_id']}/{key}")

                    expected_cross: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    expected_cross["raw"] = cross_and_zero(visual_value, audio_value)
                    expected_cross["unit"] = cross_and_zero(row_unit(visual_value), row_unit(audio_value))
                    rebuilt_metrics: dict[str, dict[str, dict[str, Any]]] = {}
                    for geometry in GEOMETRIES:
                        cross, zero = expected_cross[geometry]
                        compare_float32(score[f"cross_distance_{geometry}_float32"], cross, f"cross:{geometry}", max_diffs)
                        compare_float32(score[f"cross_distance_to_zero_{geometry}_float32"], zero, f"zero:{geometry}", max_diffs)
                        matrix = lag_matrix(cross, zero, query_n, query_n)
                        compare_float32(score[f"distance_matrix_{geometry}_float32"], matrix, f"lag:{geometry}", max_diffs)
                        rebuilt_metrics[geometry] = metrics_by_support(matrix, supports)
                        saved_metrics = result_row["raw_geometry"] if geometry == "raw" else result_row["row_unit_geometry"]
                        compare_saved_metrics(saved_metrics, rebuilt_metrics[geometry], str(plan_row["row_id"]))
                        for support in SUPPORTS:
                            item = rebuilt_metrics[geometry][support]
                            observed[sample][face][source][condition].setdefault(geometry, {})[support] = {
                                "status": item["status"], "support_n": item["support_n"],
                                **({metric: float(item[metric]) for metric in METRICS} if item["status"] == "ok" else {}),
                            }
                            cross_records[(sample, face, source, condition, geometry, support)] = {
                                "cross": cross, "zero": zero, "queries": supports[support], "audio_n": len(audio_value),
                            }
                        artifact_hashes[f"score:{plan_row['row_id']}"] = {"path": str(score_path.relative_to(root).as_posix()), "sha256": score_sha}
                    verification["fourcell_score_rows"] += 1
                for condition in CONDITIONS:
                    cell = grid[(face, source, condition)]
                    visual_id = str(cell["plan"]["visual_id"])
                    visual = visual_arrays[visual_id]
                    if condition in ("q00", "q01"):
                        q = "q00"
                        if not np.array_equal(visual, visual_arrays[f"{source}_raw"]):
                            raise AuditError(f"raw visual cache mapping changed: {sample}/{face}/{source}")
                    else:
                        q = "q10"
                        if not np.array_equal(visual, visual_arrays[f"{source}_fixed"]):
                            raise AuditError(f"fixed visual cache mapping changed: {sample}/{face}/{source}")
            verification["face_source_groups"] += 1
        verification["contexts"] += 1

    if verification["contexts"] != 80 or verification["fourcell_score_rows"] != 1280 or verification["face_source_groups"] != 320:
        raise AuditError(f"independent native matrix verification did not cover the full intended row set: {dict(verification)}")

    # All seals and cached A/V/cross/lag parity checks have passed. Only now compute null statistics.
    bootstrapper = Bootstrapper()
    for sample in samples:
        for face in FACES:
            for source in SOURCES:
                for condition in CONDITIONS:
                    for geometry in GEOMETRIES:
                        for support in SUPPORTS:
                            record = cross_records[(sample, face, source, condition, geometry, support)]
                            nulls[sample][face][source][condition].setdefault(geometry, {})[support] = null_metrics(
                                record["cross"], record["zero"], record["queries"],
                                permutation_list(int(record["audio_n"]), ordered_by_sample[sample])[0],
                            )

    analysis: dict[str, Any] = {"by_face": {}}
    level_coefficients = {
        "G_visual_fixed_minus_raw_audio_raw": {"q10": 1.0, "q00": -1.0},
        "E_audio_fixed_minus_raw_visual_raw": {"q01": 1.0, "q00": -1.0},
        "I_interaction": {"q11": 1.0, "q10": -1.0, "q01": -1.0, "q00": 1.0},
        "q11_minus_q00_joint": {"q11": 1.0, "q00": -1.0},
    }
    for face in FACES:
        analysis["by_face"][face] = {"by_geometry": {}}
        for geometry in GEOMETRIES:
            analysis["by_face"][face]["by_geometry"][geometry] = {"by_support": {}}
            for support in SUPPORTS:
                domain: dict[str, Any] = {"cells": {}, "within_recipient_G_E_I": {}, "native_T_minus_N": {}}
                for source in SOURCES:
                    source_summary = {}
                    for condition in CONDITIONS:
                        source_summary[condition] = {}
                        for metric in METRICS:
                            values = {}
                            vectors = {}
                            for sample in samples:
                                item = observed[sample][face][source][condition][geometry][support]
                                if item["status"] == "ok":
                                    values[sample] = float(item[metric])
                                vector = nulls[sample][face][source][condition][geometry][support]
                                if vector is not None:
                                    vectors[sample] = vector[metric]
                            source_summary[condition][metric] = summarize_effect(values, vectors, speaker_by_sample, samples, metric, bootstrapper)
                    domain["cells"][source] = source_summary
                    domain["within_recipient_G_E_I"][source] = {}
                    observed_source = condition_values(observed, face, source, geometry, support, "C")
                    null_source = condition_null_values(nulls, face, source, geometry, support)
                    for contrast, coefficients in level_coefficients.items():
                        contrast_summary = {}
                        for metric in METRICS:
                            obs_by_sample = signed_map(observed_source if metric == "C" else condition_values(observed, face, source, geometry, support, metric), coefficients, metric)
                            null_by_sample = signed_null_map(null_source, coefficients, metric)
                            contrast_summary[metric] = summarize_effect(obs_by_sample, null_by_sample, speaker_by_sample, samples, metric, bootstrapper)
                        domain["within_recipient_G_E_I"][source][contrast] = contrast_summary
                    # Verify clip-level algebraic closure on the fixed common support.
                    closure = {}
                    for metric in METRICS:
                        metric_values = condition_values(observed, face, source, geometry, support, metric)
                        g = signed_map(metric_values, level_coefficients["G_visual_fixed_minus_raw_audio_raw"], metric)
                        e = signed_map(metric_values, level_coefficients["E_audio_fixed_minus_raw_visual_raw"], metric)
                        i = signed_map(metric_values, level_coefficients["I_interaction"], metric)
                        joint = signed_map(metric_values, level_coefficients["q11_minus_q00_joint"], metric)
                        common = set(g) & set(e) & set(i) & set(joint)
                        closure[metric] = {"common_clip_count": len(common), "max_abs_clip_closure_error": max((abs(joint[s]-g[s]-e[s]-i[s]) for s in common), default=0.0)}
                    domain["within_recipient_G_E_I"][source]["algebraic_closure"] = closure

                for condition in ("q00", "q11"):
                    contrast_summary = {}
                    for metric in METRICS:
                        obs = {}
                        vec = {}
                        for sample in samples:
                            t_item = observed[sample][face]["tts"][condition][geometry][support]
                            n_item = observed[sample][face]["natural"][condition][geometry][support]
                            if t_item["status"] == "ok" and n_item["status"] == "ok":
                                obs[sample] = float(t_item[metric]) - float(n_item[metric])
                            t_null = nulls[sample][face]["tts"][condition][geometry][support]
                            n_null = nulls[sample][face]["natural"][condition][geometry][support]
                            if t_null is not None and n_null is not None:
                                vec[sample] = t_null[metric] - n_null[metric]
                        contrast_summary[metric] = summarize_effect(obs, vec, speaker_by_sample, samples, metric, bootstrapper)
                    domain["native_T_minus_N"][condition] = contrast_summary
                analysis["by_face"][face]["by_geometry"][geometry]["by_support"][support] = domain

    runtime_hashes = sorted({
        json.dumps(row.get("runtime_code_sha256"), sort_keys=True, separators=(",", ":"))
        for row in results if row.get("runtime_code_sha256") is not None
    })
    report = {
        "schema_version": 1,
        "analysis_type": "independent_native_fourcell_cached_temporal_null",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "python": platform.python_version(), "numpy": np.__version__, "device": "CPU", "imports_producer_or_scoring": False},
        "input_seals": {
            "run_root": str(root), "protocol_sha256": sha256_file(protocol_path), "row_plan_sha256": sha256_file(rows_path),
            "results_sha256": sha256_file(results_path), "continuation_parent_sha256": sha256_file(parent_path),
            "recovery_code_seal_sha256": sha256_file(recovery_path), "diagnostic_spec_sha256": sha256_file(spec_path),
            "attempt08_summary_sha256": sha256_file(prior_path),
        },
        "verification": {
            "planned_rows": len(plan), "result_rows": len(results), "native_fourcell_rows": verification["fourcell_score_rows"],
            "verified_contexts": verification["contexts"], "verified_face_source_groups": verification["face_source_groups"],
            "speakers": len(speaker_counts), "clips_per_speaker": 2, "all_fourcell_rows_scored": True,
            "all_context_visual_score_hashes_verified": True, "cross_and_lag_matrices_independently_rebuilt": True,
            "max_absolute_float32_rebuild_differences": max_diffs,
            "fixed_anchor": ANCHOR,
        },
        "permutation_protocol": {
            "generator": "NumPy PCG64", "seed_rule": "20260930 + ordered_eval_index", "draws": NULL_DRAWS,
            "domain_rule": "one full-A permutation sequence per sample×recipient; raw/fixed A share the sequence, both faces and all cells share it; N/T each instantiate a separate generator with the same declared seed and their own domain length",
            "boundary_rule": "for null sources outside the true full A row domain, use the saved V-to-zero distance; do not drop query rows",
            "N_raw_hashes_match_attempt08": True,
            "domains_and_hashes": permutation_records,
        },
        "execution_provenance": {
            "sealed_protocol_method": protocol.get("method", {}),
            "distinct_row_runtime_code_sha256_json": runtime_hashes,
            "per_forward_cudnn_state": "not established by the sealed EVAL artifacts; no execution value inferred",
        },
        "estimand_notes": {
            "G": "visual/generator FIXED versus RAW at RAW scoring audio: q10−q00",
            "E": "scoring audio FIXED versus RAW at RAW generation video: q01−q00",
            "I": "two-factor interaction: q11−q10−q01+q00",
            "joint": "q11−q00; G+E+I algebraic closure is reported",
            "native_gap": "T−N for q00 RAW/RAW and q11 FIXED/FIXED",
            "weighting": "clip contrasts, then two clips per speaker, then 40 speakers equally",
            "interpretation": "cached SyncNet representation-pairing diagnostic; not waveform retiming or an independent perceptual measure",
        },
        "bootstrap": {"unit": "40 speakers, each equal-weighted after its two clips are averaged", "generator": "PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "ci99": "percentile 0.005–0.995"},
        "metric_definitions": {"C": "B−D", "C_anchor": "B−D at fixed lag column 18 (k=+3; official offset=−3)", "B": "median lag distance", "D": "minimum lag distance", "search_gain": "D0−D, where D0 is zero-lag column 15"},
        "consumed_artifact_hashes": artifact_hashes,
        "summaries": analysis,
        "elapsed_seconds": time.monotonic()-started,
    }
    report_path = outdir / "independent_native_cached_null.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--attempt08-summary", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    report = validate_and_analyze(args)
    print(json.dumps({"status": report["status"], "verification": report["verification"], "summary_path": str(args.outdir.resolve()/"independent_native_cached_null.json")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
