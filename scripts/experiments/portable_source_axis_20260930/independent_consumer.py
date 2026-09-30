"""CPU-only consumer for the sealed source-axis score artifacts.

This code intentionally does not import the producer's ops/scoring modules. It
validates the immutable row plan and score artifacts, computes the predeclared
speaker-level summaries, and gathers the cached temporal null without model
forwards. It never selects or repairs rows based on their measured effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .independent_validator import (
    CONDITIONS,
    LEVEL_CELLS,
    ValidationError,
    _array_sha256,
    _file_sha256,
    _recompute_audio_mask,
    bootstrap_speaker_mean,
    curve_summary,
    reduce_two_clip_speaker_effects,
    validate_calibration_artifacts,
    validate_cross_embedding_artifacts,
    validate_intervention,
    validate_random_direction,
    validate_random_dose,
)


FACE_IDS = ("primary", "secondary")
SOURCES = ("natural", "tts")
GEOMETRIES = ("raw", "unit")
METRICS = ("C", "C_anchor", "B", "D")
AXIS_METRICS = ("C", "C_anchor", "B", "D", "search_gain")
DEFAULT_NULL_PERMUTATIONS = 100
DEFAULT_BOOTSTRAP_DRAWS = 20_000
DEFAULT_SEED = 20_260_930
SOURCE_AUDIO_KEY = "natural_audio_embeddings_raw_float32"
PRIMARY_FACE = "primary"
PRIMARY_AXIS_CONDITIONS = ("baseline", "orth_plus", "orth_minus")


class ConsumerError(ValidationError):
    """The sealed run cannot support the requested independent analysis."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConsumerError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    if not path.is_file():
        return rows, [{"path": str(path), "error": "missing_jsonl"}]
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
            except (json.JSONDecodeError, ValueError) as exc:
                malformed.append({"path": str(path), "line": line_number, "error": str(exc)})
    return rows, malformed


def _safe_artifact_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(str(relative)).parts
    if not parts or PurePosixPath(str(relative)).is_absolute() or any(part in ("..", "") for part in parts):
        raise ConsumerError(f"artifact path escapes the sealed run: {relative}")
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ConsumerError(f"artifact path escapes the sealed run: {relative}") from exc
    return path


def _canonical_permutation_hash(permutation: np.ndarray) -> str:
    values = np.asarray(permutation, dtype="<i8")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def make_temporal_permutations(
    audio_domain_n: int,
    *,
    ordered_eval_index: int,
    count: int = DEFAULT_NULL_PERMUTATIONS,
) -> tuple[list[np.ndarray], list[str]]:
    """Draw the frozen per-clip PCG64 full-A permutation sequence."""

    if audio_domain_n < 1 or ordered_eval_index < 0 or count < 1:
        raise ConsumerError("temporal-null domain, ordered index, and count must be positive")
    generator = np.random.Generator(np.random.PCG64(DEFAULT_SEED + int(ordered_eval_index)))
    permutations: list[np.ndarray] = []
    hashes: list[str] = []
    expected = np.arange(audio_domain_n, dtype=np.int64)
    for _ in range(count):
        permutation = generator.permutation(audio_domain_n).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(permutation), expected):
            raise ConsumerError("PCG64 output is not a full-domain permutation")
        permutations.append(permutation)
        hashes.append(_canonical_permutation_hash(permutation))
    return permutations, hashes


def gather_temporal_null_matrix(
    cross_distance: np.ndarray,
    distance_to_zero: np.ndarray,
    query_indices: np.ndarray,
    permutation: np.ndarray,
    lag_k: np.ndarray | None = None,
) -> np.ndarray:
    """Gather V[i] against permuted A[i+k] on fixed query support.

    Guard-20 support should keep every source index valid at all 31 lags. An
    out-of-domain source is treated as a support error, never dropped.
    """

    cross = np.asarray(cross_distance, dtype=np.float32)
    zero = np.asarray(distance_to_zero, dtype=np.float32)
    queries = np.asarray(query_indices, dtype=np.int64)
    pi = np.asarray(permutation, dtype=np.int64)
    lags = np.arange(-15, 16, dtype=np.int64) if lag_k is None else np.asarray(lag_k, dtype=np.int64)
    if cross.ndim != 2 or zero.shape != (cross.shape[0],) or queries.ndim != 1 or pi.ndim != 1:
        raise ConsumerError("temporal-null distance arrays have invalid dimensions")
    if not len(queries) or not len(pi) or cross.shape[1] != len(pi) or not len(lags):
        raise ConsumerError("temporal-null query, lag, or A domain is empty/misaligned")
    if not np.array_equal(np.sort(pi), np.arange(len(pi), dtype=np.int64)):
        raise ConsumerError("temporal-null index vector is not a permutation")
    if np.any(queries < 0) or np.any(queries >= cross.shape[0]):
        raise ConsumerError("temporal-null query lies outside the V domain")
    result = np.empty((len(queries), len(lags)), dtype=np.float32)
    for row, query in enumerate(queries):
        for column, lag in enumerate(lags):
            source = int(query + lag)
            if source < 0 or source >= len(pi):
                raise ConsumerError("guard-20 query has an invalid source index at a frozen lag")
            result[row, column] = cross[int(query), int(pi[source])]
    return result


def _expected_support(count: int, guard: int) -> np.ndarray:
    start = min(guard, count)
    stop = max(start, count - guard)
    return np.arange(start, stop, dtype=np.int32)


def _gather_scoring_matrix(
    cross: np.ndarray,
    zero: np.ndarray,
    support_n: int,
    *,
    lag_k: np.ndarray,
) -> np.ndarray:
    value = np.asarray(cross, dtype=np.float32)
    boundary = np.asarray(zero, dtype=np.float32)
    if support_n < 1 or support_n > value.shape[0] or support_n > value.shape[1] or boundary.shape != (value.shape[0],):
        raise ConsumerError("score matrix support differs from saved V/A distance domains")
    out = np.empty((support_n, len(lag_k)), dtype=np.float32)
    for query in range(support_n):
        for column, lag in enumerate(lag_k):
            source = query + int(lag)
            out[query, column] = value[query, source] if 0 <= source < support_n else boundary[query]
    return out


def _metric_payload(matrix: np.ndarray, support_arrays: Mapping[str, np.ndarray], anchor_column: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("full", "valid", "guard20", "guard15"):
        indices = np.asarray(support_arrays[name], dtype=np.int64)
        if not len(indices):
            result[name] = {"status": "too_short", "support_n": 0}
            continue
        result[name] = curve_summary(np.asarray(matrix, dtype=np.float32)[indices], anchor_column=anchor_column)
    return result


def _check_producer_metrics(producer: Mapping[str, Any], independent: Mapping[str, Any], label: str) -> None:
    for geometry in ("full", "valid", "guard20", "guard15"):
        expected = independent[geometry]
        actual = producer.get(geometry)
        if not isinstance(actual, Mapping):
            raise ConsumerError(f"{label} is missing {geometry} metrics")
        if expected.get("status") == "too_short":
            if actual.get("status") != "too_short" or int(actual.get("support_n", -1)) != 0:
                raise ConsumerError(f"{label} {geometry} short-support status differs")
            continue
        for producer_key, independent_key in (("D_min", "D"), ("B_median", "B"), ("C_best", "C"),
                                                ("C_fixed_anchor", "C_anchor"),
                                                ("D_fixed_anchor", "D_anchor")):
            if producer_key not in actual or independent_key not in expected:
                raise ConsumerError(f"{label} {geometry} metric {producer_key} is missing")
            if not np.isclose(float(actual[producer_key]), float(expected[independent_key]), rtol=2e-6, atol=2e-6):
                raise ConsumerError(f"{label} {geometry} {producer_key} differs from independent recomputation")


def _context_visual_key(row: Mapping[str, Any]) -> str:
    if row["family"] == "source_axis":
        condition = str(row["condition"])
        return "natural_z_fixed_float32" if condition == "baseline" else f"axis_z_{condition}_float32"
    return f"{row['source']}_z_{row['visual_level']}_float32"


def _compare_sample_context(
    sample_id: str,
    context_arrays: Mapping[str, np.ndarray],
    context_meta: Mapping[str, Any],
    protocol: Mapping[str, Any],
    calibration_axes: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {"sample_id": sample_id, "raw_pcm_identity": {}, "mask": {}, "axes": {}, "interventions": {}}
    rms_target = float(protocol.get("method", {}).get("rms_target", 0.0376838172675746))
    mask_spec = protocol.get("method", {}).get("speech_mask", {})
    threshold = 10.0 ** (float(mask_spec.get("threshold_dbfs", -45.0)) / 20.0)
    window_samples = int(mask_spec.get("window_samples", 3200))
    hop_samples = int(mask_spec.get("mel_hop_samples", 200))

    for source in SOURCES:
        pcm = np.asarray(context_arrays[f"{source}_pcm16"])
        raw = np.asarray(context_arrays[f"{source}_raw_float32"])
        fixed = np.asarray(context_arrays[f"{source}_fixed_float32"])
        units = np.asarray(context_arrays[f"{source}_audio_units_raw_float64"])
        if pcm.dtype != np.int16 or pcm.ndim != 1 or raw.dtype != np.float32 or fixed.dtype != np.float32:
            raise ConsumerError(f"{sample_id} {source}: PCM/RMS dtypes violate the sealed contract")
        raw_expected = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
        if raw.shape != raw_expected.shape or not np.array_equal(raw, raw_expected):
            raise ConsumerError(f"{sample_id} {source}: raw float waveform differs from PCM16 identity")
        if units.dtype != np.float64 or not np.array_equal(units, pcm.astype(np.float64)):
            raise ConsumerError(f"{sample_id} {source}: raw SyncNet units are not exact PCM16 integer units")
        if fixed.shape != raw.shape or not len(fixed) or not np.isfinite(fixed).all():
            raise ConsumerError(f"{sample_id} {source}: FIXED waveform is malformed")
        rms = float(np.sqrt(np.mean(np.square(fixed.astype(np.float64)), dtype=np.float64)))
        if abs(rms - rms_target) > 2e-7:
            raise ConsumerError(f"{sample_id} {source}: FIXED waveform RMS differs from target")
        nonzero = np.flatnonzero(raw)
        if len(nonzero):
            gain = float(fixed[int(nonzero[0])] / raw[int(nonzero[0])])
            if gain <= 0 or np.max(np.abs(fixed - raw * np.float32(gain))) > 2e-7:
                raise ConsumerError(f"{sample_id} {source}: FIXED waveform is not a uniform no-clip rescale")
        checks["raw_pcm_identity"][source] = True

    mask_starts = np.asarray(context_arrays["natural_mel_starts_int32"])
    mask_saved = np.asarray(context_arrays["natural_speech_mask_fixed_uint8"], dtype=np.uint8)
    mask_expected = _recompute_audio_mask(
        context_arrays["natural_fixed_float32"], mask_starts,
        threshold=threshold, window_samples=window_samples, mel_hop_samples=hop_samples,
    ).astype(np.uint8)
    if not np.array_equal(mask_saved, mask_expected):
        raise ConsumerError(f"{sample_id}: source-axis activity mask is not waveform-derived")
    mask = mask_saved.astype(bool)
    checks["mask"] = {"frames": len(mask), "active": int(mask.sum()), "inactive": int((~mask).sum()), "valid": True}

    d = np.asarray(context_arrays["calibration_d_source_float64"], dtype=np.float64)
    u = np.asarray(context_arrays["calibration_u_level_float64"], dtype=np.float64)
    if d.shape != (512,) or u.shape != (512,):
        raise ConsumerError(f"{sample_id}: source/level axes must be 512-dimensional")
    if calibration_axes is not None and (not np.array_equal(d, calibration_axes[0]) or not np.array_equal(u, calibration_axes[1])):
        raise ConsumerError(f"{sample_id}: context axes differ from the sealed CAL axes")
    if not np.array_equal(context_arrays["axis_z_baseline_float32"], context_arrays["natural_z_fixed_float32"]):
        raise ConsumerError(f"{sample_id}: axis baseline differs from FIXED natural generator input")
    checks["axes"] = {"source_norm": float(np.linalg.norm(d)), "level_norm": float(np.linalg.norm(u)), "valid": True}

    base = np.asarray(context_arrays["axis_z_baseline_float32"])
    for condition, sign, strict in (
        ("full_plus", 1, False), ("full_minus", -1, False),
        ("orth_plus", 1, True), ("orth_minus", -1, True),
    ):
        checks["interventions"][condition] = validate_intervention(
            base, mask, context_arrays[f"axis_z_{condition}_float32"], d, u,
            sign=sign, level_orthogonal=strict,
        )
    random_checks = {}
    diagnostics = context_meta.get("axis_diagnostics", {})
    for index in (1, 2, 3):
        condition = f"random{index}"
        direction = np.asarray(context_arrays[f"axis_direction_{condition}_float64"], dtype=np.float64)
        direction_meta = diagnostics.get(condition, {})
        if int(direction_meta.get("seed", -1)) != DEFAULT_SEED + index - 1:
            raise ConsumerError(f"{sample_id}: {condition} seed differs from the frozen control seed")
        if direction_meta.get("direction_sha256") != _array_sha256(direction):
            raise ConsumerError(f"{sample_id}: {condition} direction SHA differs from its context array")
        random_checks[condition] = {
            "direction": validate_random_direction(direction, d, u),
            "dose": validate_random_dose(
                base, mask, context_arrays[f"axis_z_{condition}_float32"],
                context_arrays["axis_z_orth_plus_float32"], direction, u,
                direction_meta.get("per_frame_projection_diagnostics", []),
            ),
        }
    checks["interventions"].update(random_checks)

    array_hashes = context_meta.get("array_sha256", {})
    required_hashed = (
        "natural_pcm16", "natural_raw_float32", "natural_fixed_float32",
        "natural_audio_units_raw_float64", "natural_audio_embeddings_raw_float32",
        "natural_z_fixed_float32", "natural_speech_mask_fixed_uint8",
        "calibration_d_source_float64", "calibration_u_level_float64",
    ) + tuple(f"axis_z_{name}_float32" for name in CONDITIONS) + tuple(
        f"axis_direction_random{index}_float64" for index in (1, 2, 3)
    )
    for key in required_hashed:
        if key not in context_arrays or array_hashes.get(key) != _array_sha256(context_arrays[key]):
            raise ConsumerError(f"{sample_id}: context array hash mismatch for {key}")
    for condition in CONDITIONS:
        condition_meta = diagnostics.get(condition, {})
        latent_hash_key = "z_sha256" if condition == "baseline" else "output_z_sha256"
        if condition_meta.get(latent_hash_key) != _array_sha256(context_arrays[f"axis_z_{condition}_float32"]):
            raise ConsumerError(f"{sample_id}: {condition} diagnostics do not bind its saved latent")
    for index in (1, 2, 3):
        condition = f"random{index}"
        if diagnostics[condition].get("random_direction_sha256") != _array_sha256(context_arrays[f"axis_direction_{condition}_float64"]):
            raise ConsumerError(f"{sample_id}: {condition} dose metadata do not bind its random direction")
    checks["valid"] = True
    return checks


def _read_score_artifact(
    run_dir: Path,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    anchor_column: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    score_path = _safe_artifact_path(run_dir, str(plan["score_npz"]))
    if not score_path.is_file():
        raise ConsumerError(f"missing score artifact: {plan['row_id']}")
    if _file_sha256(score_path) != result.get("score_npz_sha256"):
        raise ConsumerError(f"score artifact SHA mismatch: {plan['row_id']}")
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    needed = (
        "distance_matrix_raw_float32", "distance_matrix_unit_float32",
        "cross_distance_raw_float32", "cross_distance_unit_float32",
        "cross_distance_to_zero_raw_float32", "cross_distance_to_zero_unit_float32",
        "common_support_indices_int32", "full_support_indices_int32", "valid_support_indices_int32",
        "guard20_indices_int32", "guard15_indices_int32", "lag_columns_int32", "lag_k_int32",
    )
    if any(key not in arrays for key in needed):
        raise ConsumerError(f"score artifact misses required arrays: {plan['row_id']}")
    support_n = int(result.get("common_support_n", -1))
    if support_n < 1 or int(result.get("scoring_audio_domain_n", -1)) != support_n:
        raise ConsumerError(f"score row does not retain the frozen common support: {plan['row_id']}")
    lag_k = np.arange(-15, 16, dtype=np.int32)
    if not np.array_equal(arrays["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(arrays["lag_k_int32"], lag_k):
        raise ConsumerError(f"lag column/k mapping differs from the 31-lag frozen domain: {plan['row_id']}")
    supports = {
        "full": _expected_support(support_n, 0),
        "valid": _expected_support(support_n, 0),
        "guard20": _expected_support(support_n, 20),
        "guard15": _expected_support(support_n, 15),
    }
    if not np.array_equal(arrays["common_support_indices_int32"], np.arange(support_n, dtype=np.int32)):
        raise ConsumerError(f"common score support is not the complete fixed query clock: {plan['row_id']}")
    for name, expected in supports.items():
        key = f"{name}_support_indices_int32" if name in ("full", "valid") else f"{name}_indices_int32"
        if not np.array_equal(arrays[key], expected):
            raise ConsumerError(f"{name} support differs from the frozen guard rule: {plan['row_id']}")

    metric_values = {}
    for geometry in GEOMETRIES:
        matrix = np.asarray(arrays[f"distance_matrix_{geometry}_float32"])
        cross = np.asarray(arrays[f"cross_distance_{geometry}_float32"])
        zero = np.asarray(arrays[f"cross_distance_to_zero_{geometry}_float32"])
        if matrix.dtype != np.float32 or matrix.shape != (support_n, 31):
            raise ConsumerError(f"{geometry} lag matrix shape/dtype differs from the frozen support: {plan['row_id']}")
        if cross.dtype != np.float32 or cross.ndim != 2 or cross.shape[0] < support_n or zero.shape != (cross.shape[0],):
            raise ConsumerError(f"{geometry} full V×A distance domain is malformed: {plan['row_id']}")
        expected_matrix = _gather_scoring_matrix(cross, zero, support_n, lag_k=lag_k)
        if not np.array_equal(matrix, expected_matrix):
            raise ConsumerError(f"{geometry} lag matrix is not gathered from the sealed full cross-distance domain: {plan['row_id']}")
        metric_values[geometry] = _metric_payload(matrix, supports, anchor_column)
        _check_producer_metrics(result.get("raw_geometry" if geometry == "raw" else "row_unit_geometry", {}), metric_values[geometry], plan["row_id"])
    if int(result.get("audio_row_domain_n", -1)) != int(arrays["cross_distance_raw_float32"].shape[1]):
        raise ConsumerError(f"saved full A domain differs from row metadata: {plan['row_id']}")
    if int(result.get("visual_query_domain_n", -1)) != int(arrays["cross_distance_raw_float32"].shape[0]):
        raise ConsumerError(f"saved V domain differs from row metadata: {plan['row_id']}")
    arrays["_support_n"] = np.asarray(support_n)
    return metric_values, arrays


def _effect_summary(
    values_by_sample: Mapping[str, float],
    speaker_by_sample: Mapping[str, str],
    expected_samples: Sequence[str],
    *,
    draws: int,
) -> dict[str, Any]:
    retained = {sample: float(value) for sample, value in values_by_sample.items() if sample in speaker_by_sample}
    groups: dict[str, list[float]] = defaultdict(list)
    group_samples: dict[str, list[str]] = defaultdict(list)
    for sample, value in sorted(retained.items()):
        speaker = speaker_by_sample[sample]
        groups[speaker].append(value)
        group_samples[speaker].append(sample)
    complete_speakers = {speaker: float(np.mean(items, dtype=np.float64)) for speaker, items in groups.items() if len(items) == 2}
    complete_clip_ids = sorted(sample for speaker, ids in group_samples.items() if len(ids) == 2 for sample in ids)
    missing_clips = sorted(set(expected_samples) - set(retained))
    missing_speakers = sorted(set(speaker_by_sample[s] for s in expected_samples) - set(complete_speakers))
    bootstrap = None
    if len(complete_speakers) >= 2:
        bootstrap = bootstrap_speaker_mean(complete_speakers, draws=draws, seed=DEFAULT_SEED)
    return {
        "complete_paired_clip_count": len(retained),
        "expected_clip_count": len(expected_samples),
        "complete_two_clip_speaker_count": len(complete_speakers),
        "expected_speaker_count": len(set(speaker_by_sample.values())),
        "missing_clip_ids": missing_clips,
        "missing_or_incomplete_speakers": missing_speakers,
        "paired_clip_mean": float(np.mean(list(retained.values()), dtype=np.float64)) if retained else None,
        "equal_speaker_mean": float(np.mean(list(complete_speakers.values()), dtype=np.float64)) if complete_speakers else None,
        "complete_speaker_values": dict(sorted(complete_speakers.items())),
        "speaker_bootstrap_99_ci": bootstrap["ci99"] if bootstrap else None,
        "bootstrap_draws": draws if bootstrap else None,
        "bootstrap_seed": DEFAULT_SEED if bootstrap else None,
        "eligible_for_intended_cohort": len(retained) == len(expected_samples) and len(complete_speakers) == len(set(speaker_by_sample.values())),
    }


def _primary_scope_status(
    expected_samples: Sequence[str],
    speaker_by_sample: Mapping[str, str],
    metric_rows: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    score_supports: Mapping[tuple[str, str, str, str], Mapping[str, tuple[str, str]]],
    *,
    run_rows_sealed: bool,
) -> dict[str, Any]:
    """Determine primary source-axis support independently of secondary rows."""

    validated: set[tuple[str, str]] = set()
    support_groups_valid = 0
    support_failures: list[dict[str, Any]] = []
    for sample in expected_samples:
        group = score_supports.get((sample, PRIMARY_FACE, "source_axis", "natural"), {})
        digests = []
        for condition in PRIMARY_AXIS_CONDITIONS:
            metrics = metric_rows.get((PRIMARY_FACE, sample, "source_axis", "natural", condition), {})
            raw = metrics.get("raw")
            if isinstance(raw, Mapping) and all(key in raw for key in ("C", "C_anchor", "B", "D")):
                validated.add((sample, condition))
            digest = group.get(condition)
            if digest is not None:
                digests.append(digest)
        common_support = len(digests) == len(PRIMARY_AXIS_CONDITIONS) and len(set(digests)) == 1
        if common_support:
            support_groups_valid += 1
        else:
            support_failures.append({
                "sample_id": sample,
                "conditions_expected": list(PRIMARY_AXIS_CONDITIONS),
                "conditions_with_support": [condition for condition in PRIMARY_AXIS_CONDITIONS if condition in group],
                "common_support": common_support,
            })
    speaker_counts: dict[str, int] = defaultdict(int)
    for sample in expected_samples:
        if sample in speaker_by_sample:
            speaker_counts[str(speaker_by_sample[sample])] += 1
    expected_row_count = len(expected_samples) * len(PRIMARY_AXIS_CONDITIONS)
    eligible = bool(
        run_rows_sealed
        and len(expected_samples) == 80
        and len(speaker_counts) == 40
        and all(count == 2 for count in speaker_counts.values())
        and len(validated) == expected_row_count
        and support_groups_valid == len(expected_samples)
    )
    return {
        "eligible": eligible,
        "required_conditions": list(PRIMARY_AXIS_CONDITIONS),
        "validated_required_score_rows": len(validated),
        "expected_required_score_rows": expected_row_count,
        "common_support_groups_valid": support_groups_valid,
        "expected_common_support_groups": len(expected_samples),
        "support_failures": support_failures,
    }


def _sample_rows(plan_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    speaker_by_sample: dict[str, str] = {}
    order_by_sample: dict[str, int] = {}
    for row in plan_rows:
        sample = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        ordered = int(row["ordered_eval_index"])
        if sample in speaker_by_sample and (speaker_by_sample[sample] != speaker or order_by_sample[sample] != ordered):
            raise ConsumerError(f"inconsistent sample identity in sealed row plan: {sample}")
        speaker_by_sample[sample] = speaker
        order_by_sample[sample] = ordered
    return speaker_by_sample, order_by_sample


def _validate_plan(plan_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, int]]:
    planned_count = int(protocol.get("row_count", -1))
    if len(plan_rows) != planned_count or planned_count != 2560:
        raise ConsumerError(f"sealed plan must contain exactly 2,560 score rows; found {len(plan_rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in by_id:
            raise ConsumerError(f"empty or duplicate row_id in sealed plan: {row_id}")
        by_id[row_id] = dict(row)
        if row.get("split") != "evaluation" or row.get("face_id") not in FACE_IDS:
            raise ConsumerError(f"unexpected split or face in sealed plan: {row_id}")
        if row.get("family") == "source_axis":
            if row.get("source") != "natural" or row.get("audio_key") != SOURCE_AUDIO_KEY or row.get("audio_level") != "raw":
                raise ConsumerError(f"source-axis plan does not bind unchanged RAW natural A: {row_id}")
            if row.get("condition") not in CONDITIONS:
                raise ConsumerError(f"unknown source-axis condition in sealed plan: {row_id}")
            expected_row_id = f"{row['sample_id']}__{row['face_id']}__source_axis__natural__{row['condition']}"
        elif row.get("family") == "fourcell":
            if row.get("source") not in SOURCES or row.get("condition") not in LEVEL_CELLS:
                raise ConsumerError(f"unknown four-cell source/condition in sealed plan: {row_id}")
            cell = str(row["condition"])
            expected_visual = "raw" if cell in ("q00", "q01") else "fixed"
            expected_audio = "raw" if cell in ("q00", "q10") else "fixed"
            if row.get("visual_level") != expected_visual or row.get("audio_level") != expected_audio:
                raise ConsumerError(f"four-cell plan does not encode the declared G/E/I cells: {row_id}")
            expected_row_id = f"{row['sample_id']}__{row['face_id']}__fourcell__{row['source']}__{row['condition']}"
        else:
            raise ConsumerError(f"unknown row family in sealed plan: {row_id}")
        if row_id != expected_row_id:
            raise ConsumerError(f"row_id does not bind its sample/face/family/condition identity: {row_id}")
    speaker_by_sample, order_by_sample = _sample_rows(plan_rows)
    if len(speaker_by_sample) != 80 or len(set(speaker_by_sample.values())) != 40:
        raise ConsumerError("sealed row plan must identify 80 clips from 40 speakers")
    if set(order_by_sample.values()) != set(range(80)):
        raise ConsumerError("ordered_eval_index must cover the frozen 0..79 order")
    if any(sum(value == speaker for value in speaker_by_sample.values()) != 2 for speaker in set(speaker_by_sample.values())):
        raise ConsumerError("the sealed cohort must retain exactly two clips per speaker")
    for sample in speaker_by_sample:
        rows = [row for row in plan_rows if row["sample_id"] == sample]
        for face in FACE_IDS:
            face_rows = [row for row in rows if row["face_id"] == face]
            source_axis = {str(row["condition"]) for row in face_rows if row["family"] == "source_axis"}
            fourcell = {(str(row["source"]), str(row["condition"])) for row in face_rows if row["family"] == "fourcell"}
            if (len(face_rows) != 16 or source_axis != set(CONDITIONS)
                    or fourcell != {(source, cell) for source in SOURCES for cell in LEVEL_CELLS}):
                raise ConsumerError(f"sealed row grid is incomplete for {sample}/{face}")
    return by_id, speaker_by_sample, order_by_sample


def _support_digest(arrays: Mapping[str, np.ndarray]) -> tuple[str, str]:
    return (
        _array_sha256(arrays["common_support_indices_int32"]),
        _array_sha256(arrays["guard20_indices_int32"]),
    )


def consume_evaluation_run(
    run_dir: str | Path,
    *,
    calibration_dir: str | Path | None = None,
    null_count: int = DEFAULT_NULL_PERMUTATIONS,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    verify_cached_embeddings: bool = True,
) -> dict[str, Any]:
    """Validate row artifacts and produce the frozen CPU statistics/null JSON."""

    if null_count != DEFAULT_NULL_PERMUTATIONS:
        raise ConsumerError("the frozen temporal null uses exactly 100 permutations")
    if bootstrap_draws != DEFAULT_BOOTSTRAP_DRAWS:
        raise ConsumerError("the frozen speaker bootstrap uses exactly 20,000 PCG64 draws")
    root = Path(run_dir).resolve()
    protocol_path = root / "evaluation_protocol.json"
    rows_path = root / "evaluation_rows.jsonl"
    protocol = _read_json(protocol_path)
    plan_rows, malformed_plan = _read_jsonl(rows_path)
    if malformed_plan:
        raise ConsumerError(f"pre-score row manifest is malformed: {malformed_plan[:2]}")
    row_bytes_sha = _file_sha256(rows_path)
    if protocol.get("row_manifest_sha256") != row_bytes_sha:
        raise ConsumerError("pre-score row-manifest SHA differs from the frozen protocol")
    if protocol.get("status") not in ("frozen_pre_effects", "evaluation_started", "sealed_row_scores"):
        raise ConsumerError(f"unexpected evaluation protocol status: {protocol.get('status')}")
    cohort = protocol.get("cohort", {})
    if (protocol.get("input_identity_rows_match") is not True
            or int(cohort.get("evaluation_rows", -1)) != 80
            or int(cohort.get("evaluation_speakers", -1)) != 40
            or cohort.get("two_clips_per_speaker") is not True):
        raise ConsumerError("protocol does not seal the intended 80-clip/40-speaker cohort and input identity")
    plan_by_id, speaker_by_sample, order_by_sample = _validate_plan(plan_rows, protocol)
    expected_samples = sorted(speaker_by_sample)
    anchor = protocol.get("global_anchor", {})
    anchor_column = int(anchor.get("anchor_column", -1))
    anchor_k = int(anchor.get("selected_k", anchor.get("anchor_k", -999)))
    official_offset = int(anchor.get("official_syncnet_offset", -999))
    if anchor_column != anchor_k + 15 or official_offset != -anchor_k:
        raise ConsumerError("global calibration anchor column/k/official-offset sign is inconsistent")

    cal_root = Path(calibration_dir).resolve() if calibration_dir else root.parent / "calibration_20260930"
    cal_summary_path = cal_root / "calibration_summary.json"
    cal_summary = None
    cal_axes = None
    cal_audit = None
    if cal_summary_path.is_file():
        cal_audit = validate_calibration_artifacts(cal_summary_path, require_all_features=True)
        cal_summary = _read_json(cal_summary_path)
        cal_anchor = cal_summary.get("global_anchor", {})
        if int(cal_anchor.get("anchor_column", -1)) != anchor_column or int(cal_anchor.get("selected_k", -999)) != anchor_k:
            raise ConsumerError("evaluation anchor differs from independent CAL q10 anchor")
        protocol_cal = protocol.get("calibration_summary", {})
        if protocol_cal.get("sha256") != cal_audit["summary_sha256"]:
            raise ConsumerError("evaluation protocol does not bind the independently audited CAL summary")
        axes_path = cal_root / "calibration_axes.npz"
        protocol_axes = protocol.get("calibration_axes", {})
        if protocol_axes.get("sha256") != _file_sha256(axes_path):
            raise ConsumerError("evaluation protocol does not bind the independently audited CAL axes")
        with np.load(axes_path, allow_pickle=False) as archive:
            cal_axes = (archive["d_source"].astype(np.float64), archive["u_level"].astype(np.float64))

    result_rows, malformed_results = _read_jsonl(root / "evaluation_results.jsonl")
    status_rows, malformed_status = _read_jsonl(root / "evaluation_clip_status.jsonl")
    attempt_rows, malformed_attempts = _read_jsonl(root / "evaluation_attempts.jsonl")
    attempt_failures = [
        row for row in attempt_rows
        if row.get("event") == "clip_failed" or row.get("status") == "failed"
    ]
    row_failures: list[dict[str, Any]] = []
    for collection, malformed in (("results", malformed_results), ("clip_status", malformed_status), ("attempts", malformed_attempts)):
        row_failures.extend({"artifact": collection, **item} for item in malformed)
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        row_id = str(row.get("row_id", ""))
        if row_id not in plan_by_id:
            row_failures.append({"artifact": "result", "row_id": row_id, "error": "unexpected_row_id"})
            continue
        if row_id in results_by_id:
            row_failures.append({"artifact": "result", "row_id": row_id, "error": "duplicate_result_row"})
            continue
        results_by_id[row_id] = row
    missing_row_ids = sorted(set(plan_by_id) - set(results_by_id))
    for row_id in missing_row_ids:
        row_failures.append({"artifact": "result", "row_id": row_id, "sample_id": plan_by_id[row_id]["sample_id"], "error": "missing_scored_row"})

    context_cache: OrderedDict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]] = OrderedDict()
    context_checks_by_sample: dict[str, dict[str, Any]] = {}
    visual_cache: OrderedDict[tuple[str, str, str], tuple[dict[str, np.ndarray], Path]] = OrderedDict()
    metric_rows: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    score_supports: dict[tuple[str, str, str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    score_support_summaries: dict[str, Any] = {}
    axis_values: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    score_paths: dict[str, Path] = {}
    cached_parity_rows: list[dict[str, Any]] = []
    cross_pair_parity: dict[tuple[str, str], dict[str, Any]] = {}
    cross_parity_row_count = 0
    cross_pair_use_count: dict[tuple[str, str], int] = defaultdict(int)
    row_validation_count = 0
    context_errors: list[dict[str, Any]] = []

    for row_id, result in results_by_id.items():
        plan = plan_by_id[row_id]
        row_parity: dict[str, Any] | None = None
        try:
            for key in ("sample_id", "speaker_id", "split", "family", "source", "condition", "face_id", "visual_id", "audio_key", "ordered_eval_index"):
                if result.get(key) != plan.get(key):
                    raise ConsumerError(f"result identity field {key} differs from pre-score plan")
            if result.get("status") != "scored":
                raise ConsumerError("result row is not marked scored")
            if (result.get("natural_input_sha256") != plan.get("natural_input_sha256")
                    or result.get("tts_input_sha256") != plan.get("tts_input_sha256")):
                raise ConsumerError(f"scored row input hashes differ from the pre-score plan: {row_id}")
            row_anchor = result.get("anchor", {})
            if (row_anchor.get("source") != "calibration natural q10"
                    or int(row_anchor.get("column", -1)) != anchor_column
                    or int(row_anchor.get("k", -999)) != anchor_k
                    or int(row_anchor.get("official_offset_frames", 999)) != official_offset):
                raise ConsumerError(f"row anchor is not bound to the one CAL q10 anchor: {row_id}")
            if plan["family"] == "source_axis" and result.get("audio_embedding_key") != SOURCE_AUDIO_KEY:
                raise ConsumerError("source-axis result does not use unchanged RAW-natural A")
            if plan["family"] == "source_axis" and result.get("input_pcm_identity", {}).get("natural_raw_exact") is not True:
                raise ConsumerError("source-axis RAW-natural PCM identity is not exact")

            sample_id = str(plan["sample_id"])
            if sample_id not in context_cache:
                context_path = _safe_artifact_path(root, str(plan["context_npz"]))
                context_json_path = root / "samples" / f"{sample_id}_context.json"
                if not context_path.is_file() or not context_json_path.is_file():
                    raise ConsumerError(f"missing context NPZ/JSON for {sample_id}")
                if _file_sha256(context_path) != result.get("context_npz_sha256"):
                    raise ConsumerError(f"context NPZ SHA mismatch for {sample_id}")
                if _file_sha256(context_json_path) != result.get("context_json_sha256"):
                    raise ConsumerError(f"context JSON SHA mismatch for {sample_id}")
                context_meta = _read_json(context_json_path)
                with np.load(context_path, allow_pickle=False) as archive:
                    all_context_arrays = {key: archive[key] for key in archive.files}
                context_checks = _compare_sample_context(sample_id, all_context_arrays, context_meta, protocol, cal_axes)
                if context_meta.get("sample_id") != sample_id or context_meta.get("speaker_id") != speaker_by_sample[sample_id]:
                    raise ConsumerError(f"context sample/speaker identity differs for {sample_id}")
                inputs = context_meta.get("inputs", {})
                if inputs.get("natural", {}).get("sha256") != plan.get("natural_input_sha256") or inputs.get("tts", {}).get("sha256") != plan.get("tts_input_sha256"):
                    raise ConsumerError(f"context natural/TTS input hashes differ from plan for {sample_id}")
                used_context_keys = {
                    f"{source}_audio_embeddings_{level}_float32"
                    for source in SOURCES for level in ("raw", "fixed")
                }
                used_context_keys.update({
                    "natural_z_raw_float32", "natural_z_fixed_float32", "tts_z_raw_float32", "tts_z_fixed_float32",
                    *(f"axis_z_{condition}_float32" for condition in CONDITIONS),
                })
                context_arrays = {key: all_context_arrays[key] for key in used_context_keys}
                context_meta_small = {
                    "sample_id": context_meta.get("sample_id"),
                    "speaker_id": context_meta.get("speaker_id"),
                    "context_npz": context_meta.get("context_npz", {}),
                    "inputs": inputs,
                    "raw_unit_hashes": {
                        source: context_meta.get("array_sha256", {}).get(f"{source}_audio_units_raw_float64")
                        for source in SOURCES
                    },
                }
                context_cache[sample_id] = (context_arrays, context_meta_small, context_checks)
                context_cache.move_to_end(sample_id)
                context_checks_by_sample[sample_id] = context_checks
                while len(context_cache) > 2:
                    context_cache.popitem(last=False)
            context_arrays, context_meta, _context_checks = context_cache[sample_id]
            context_cache.move_to_end(sample_id)
            if result.get("context_npz_sha256") != context_meta.get("context_npz", {}).get("sha256"):
                raise ConsumerError(f"context meta does not bind the NPZ hash for {sample_id}")
            if result.get("input_pcm_identity", {}).get("tts_raw_exact") is not True:
                raise ConsumerError(f"TTS raw PCM identity failed for {sample_id}")
            pcm_identity = result.get("input_pcm_identity", {})
            if pcm_identity.get("natural_syncnet_units_sha256") != context_meta["raw_unit_hashes"].get("natural"):
                raise ConsumerError(f"natural SyncNet integer-unit hash differs from exact PCM16 units: {row_id}")
            if pcm_identity.get("tts_syncnet_units_sha256") != context_meta["raw_unit_hashes"].get("tts"):
                raise ConsumerError(f"TTS SyncNet integer-unit hash differs from exact PCM16 units: {row_id}")

            visual_relative = str(plan["visual_npz"])
            visual_path = _safe_artifact_path(root, visual_relative)
            visual_key = (str(plan["face_id"]), sample_id, str(plan["visual_id"]))
            if visual_key not in visual_cache:
                if not visual_path.is_file() or _file_sha256(visual_path) != result.get("visual_npz_sha256"):
                    raise ConsumerError(f"visual feature NPZ missing/hash mismatch: {row_id}")
                with np.load(visual_path, allow_pickle=False) as archive:
                    visual_arrays = {key: archive[key] for key in archive.files}
                visual_cache[visual_key] = (visual_arrays, visual_path)
                visual_cache.move_to_end(visual_key)
                while len(visual_cache) > 12:
                    visual_cache.popitem(last=False)
            else:
                visual_cache.move_to_end(visual_key)
            visual_arrays, _visual_path = visual_cache[visual_key]
            latent_key = _context_visual_key(plan)
            if "z_condition_float32" not in visual_arrays or "lip_embeddings_float32" not in visual_arrays:
                raise ConsumerError(f"visual feature NPZ is missing latent/lip arrays: {row_id}")
            if not np.array_equal(visual_arrays["z_condition_float32"], context_arrays[latent_key]):
                raise ConsumerError(f"saved visual generator latent differs from the sealed context: {row_id}")
            lip = np.asarray(visual_arrays["lip_embeddings_float32"])
            if _array_sha256(lip) != result.get("lip_embedding_sha256"):
                raise ConsumerError(f"lip embedding hash mismatch: {row_id}")
            audio = np.asarray(context_arrays[str(plan["audio_key"])])
            if _array_sha256(audio) != result.get("audio_embedding_sha256"):
                raise ConsumerError(f"audio embedding hash mismatch: {row_id}")
            if plan["family"] == "source_axis" and not np.array_equal(audio, context_arrays[SOURCE_AUDIO_KEY]):
                raise ConsumerError(f"source-axis A rows changed across condition: {row_id}")

            metric_payload, score_arrays = _read_score_artifact(root, plan, result, anchor_column)
            score_path_for_row = _safe_artifact_path(root, str(plan["score_npz"]))
            lip_sha = _array_sha256(lip)
            audio_sha = _array_sha256(audio)
            cross_pair_key = (lip_sha, audio_sha)
            cross_array_keys = (
                "cross_distance_raw_float32", "cross_distance_unit_float32",
                "cross_distance_to_zero_raw_float32", "cross_distance_to_zero_unit_float32",
            )
            cross_array_hashes = {key: _array_sha256(score_arrays[key]) for key in cross_array_keys}
            if cross_pair_key not in cross_pair_parity:
                full_matrix_parity = validate_cross_embedding_artifacts(
                    lip, audio,
                    score_arrays["cross_distance_raw_float32"],
                    score_arrays["cross_distance_unit_float32"],
                    score_arrays["cross_distance_to_zero_raw_float32"],
                    score_arrays["cross_distance_to_zero_unit_float32"],
                )
                cross_pair_parity[cross_pair_key] = {
                    "first_row_id": row_id,
                    "lip_embedding_sha256": lip_sha,
                    "audio_embedding_sha256": audio_sha,
                    "cross_artifact_sha256": cross_array_hashes,
                    **full_matrix_parity,
                }
            elif cross_pair_parity[cross_pair_key]["cross_artifact_sha256"] != cross_array_hashes:
                raise ConsumerError(f"duplicate cached lip/audio pair has different full cross-distance arrays: {row_id}")
            support_digest = _support_digest(score_arrays)
            if plan["family"] == "source_axis":
                support_group = (sample_id, str(plan["face_id"]), "source_axis", "natural")
            else:
                support_group = (sample_id, str(plan["face_id"]), "fourcell", str(plan["source"]))

            row_metrics = {}
            support_report: dict[str, Any] = {}
            for geometry in GEOMETRIES:
                support_report[geometry] = {}
                for support_name in ("full", "valid", "guard20", "guard15"):
                    current = metric_payload[geometry][support_name]
                    if current.get("status") == "too_short":
                        support_report[geometry][support_name] = current
                        continue
                    compact = {key: float(current[key]) for key in ("C", "C_anchor", "B", "D")}
                    compact["search_gain"] = compact["C"]
                    support_report[geometry][support_name] = compact
                    if support_name == "guard20":
                        row_metrics[geometry] = compact
            key = (str(plan["face_id"]), sample_id, str(plan["family"]), str(plan["source"]), str(plan["condition"]))
            row_axis_metrics = {}
            for geometry in GEOMETRIES:
                if plan["family"] == "source_axis" and geometry in row_metrics:
                    row_axis_metrics[(str(plan["face_id"]), sample_id, geometry, str(plan["condition"]))] = row_metrics[geometry]

            # A prespecified, effect-blind parity sample checks saved distances
            # directly against the cached A/V embeddings. Every row is still
            # checked against the complete saved V×A matrix above.
            ordered_index = int(plan["ordered_eval_index"])
            if verify_cached_embeddings and ordered_index in {0, 39, 79}:
                from .independent_validator import validate_embedding_distance_artifacts

                support_n = int(result["common_support_n"])
                parity = validate_embedding_distance_artifacts(
                    lip[:support_n], audio[:support_n],
                    score_arrays["distance_matrix_raw_float32"],
                    score_arrays["distance_matrix_unit_float32"],
                    vshift=15,
                )
                row_parity = {"row_id": row_id, "ordered_eval_index": ordered_index, **parity}

            # Commit this row only after every independent check, including the
            # preselected embedding-distance parity, has passed.
            metric_rows[key] = row_metrics
            score_supports[support_group][str(plan["condition"])] = support_digest
            score_support_summaries[row_id] = support_report
            score_paths[row_id] = score_path_for_row
            axis_values.update(row_axis_metrics)
            if row_parity is not None:
                cached_parity_rows.append(row_parity)
            row_validation_count += 1
            cross_parity_row_count += 1
            cross_pair_use_count[cross_pair_key] += 1
        except Exception as exc:
            row_failures.append({
                "artifact": "row_validation", "row_id": row_id,
                "sample_id": plan.get("sample_id"), "face_id": plan.get("face_id"),
                "error_type": type(exc).__name__, "error": str(exc),
            })
            context_errors.append({"row_id": row_id, "sample_id": plan.get("sample_id"), "error": str(exc)})

    # Same query support is a property of the entire planned family, not a
    # selected subset of successful cells.
    support_failures = []
    for key, conditions in score_supports.items():
        expected = set(CONDITIONS) if key[2] == "source_axis" else set(LEVEL_CELLS)
        if set(conditions) != expected:
            support_failures.append({"group": list(key), "conditions_present": sorted(conditions), "conditions_expected": sorted(expected)})
        elif len(set(conditions.values())) != 1:
            support_failures.append({"group": list(key), "error": "condition_supports_differ"})
    if support_failures:
        row_failures.extend({"artifact": "support", **item} for item in support_failures)

    # Derive the expected cell and condition scalars only after each score row
    # has passed identity, hash, support, matrix-gather and geometry checks.
    fourcell_contrasts: dict[str, Any] = {}
    fourcell_clip: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for face in FACE_IDS:
        for sample in expected_samples:
            for geometry in GEOMETRIES:
                for source in SOURCES:
                    cells = {
                        cell: metric_rows.get((face, sample, "fourcell", source, cell), {}).get(geometry)
                        for cell in LEVEL_CELLS
                    }
                    if any(value is None for value in cells.values()):
                        continue
                    for metric in METRICS:
                        q00, q10, q01, q11 = (float(cells[cell][metric]) for cell in LEVEL_CELLS)
                        contrasts = {
                            "G": q10 - q00,
                            "E": q01 - q00,
                            "I": q11 - q10 - q01 + q00,
                            "native": q00,
                            "processed": q11,
                        }
                        for contrast, value in contrasts.items():
                            fourcell_clip[(face, source, geometry, metric, contrast)] = fourcell_clip.get((face, source, geometry, metric, contrast), {})
                            fourcell_clip[(face, source, geometry, metric, contrast)][sample] = value
    for face in FACE_IDS:
        for geometry in GEOMETRIES:
            for metric in METRICS:
                for contrast in ("G", "E", "I", "native", "processed"):
                    fourcell_contrasts.setdefault(face, {}).setdefault(geometry, {}).setdefault(metric, {}).setdefault(contrast, {})
                    for source in SOURCES:
                        values = fourcell_clip.get((face, source, geometry, metric, contrast), {})
                        fourcell_contrasts[face][geometry][metric][contrast][source] = _effect_summary(
                            values, speaker_by_sample, expected_samples, draws=bootstrap_draws,
                        )

    t_minus_n: dict[str, Any] = {}
    for face in FACE_IDS:
        for geometry in GEOMETRIES:
            for metric in METRICS:
                for cell in LEVEL_CELLS:
                    values = {}
                    for sample in expected_samples:
                        n = metric_rows.get((face, sample, "fourcell", "natural", cell), {}).get(geometry)
                        t = metric_rows.get((face, sample, "fourcell", "tts", cell), {}).get(geometry)
                        if n is not None and t is not None:
                            values[sample] = float(t[metric] - n[metric])
                    t_minus_n.setdefault(face, {}).setdefault(geometry, {}).setdefault(metric, {})[cell] = _effect_summary(
                        values, speaker_by_sample, expected_samples, draws=bootstrap_draws,
                    )
                for contrast in ("G", "E", "I", "native", "processed"):
                    values = {}
                    for sample in expected_samples:
                        n = fourcell_clip.get((face, "natural", geometry, metric, contrast), {}).get(sample)
                        t = fourcell_clip.get((face, "tts", geometry, metric, contrast), {}).get(sample)
                        if n is not None and t is not None:
                            values[sample] = float(t - n)
                    t_minus_n[face][geometry][metric][contrast] = _effect_summary(
                        values, speaker_by_sample, expected_samples, draws=bootstrap_draws,
                    )
            native_gap = {}
            for metric in METRICS:
                values = {}
                for sample in expected_samples:
                    n = metric_rows.get((face, sample, "fourcell", "natural", "q00"), {}).get(geometry)
                    t = metric_rows.get((face, sample, "fourcell", "tts", "q00"), {}).get(geometry)
                    if n is not None and t is not None:
                        values[sample] = float(t[metric] - n[metric])
                native_gap[metric] = _effect_summary(values, speaker_by_sample, expected_samples, draws=bootstrap_draws)
            processed_gap = {}
            attenuation = {}
            for metric in METRICS:
                values = {}
                for sample in expected_samples:
                    n = metric_rows.get((face, sample, "fourcell", "natural", "q11"), {}).get(geometry)
                    t = metric_rows.get((face, sample, "fourcell", "tts", "q11"), {}).get(geometry)
                    if n is not None and t is not None:
                        values[sample] = float(t[metric] - n[metric])
                processed_gap[metric] = _effect_summary(values, speaker_by_sample, expected_samples, draws=bootstrap_draws)
                native_by_speaker = native_gap[metric]["complete_speaker_values"]
                processed_by_speaker = processed_gap[metric]["complete_speaker_values"]
                common_speakers = sorted(set(native_by_speaker) & set(processed_by_speaker))
                attenuation_by_speaker = {
                    speaker: float(native_by_speaker[speaker] - processed_by_speaker[speaker])
                    for speaker in common_speakers
                }
                attenuation[metric] = {
                    "speaker_values": attenuation_by_speaker,
                    "equal_speaker_mean": float(np.mean(list(attenuation_by_speaker.values()))) if attenuation_by_speaker else None,
                    "speaker_bootstrap_99_ci": bootstrap_speaker_mean(attenuation_by_speaker, draws=bootstrap_draws, seed=DEFAULT_SEED)["ci99"] if len(attenuation_by_speaker) >= 2 else None,
                    "complete_speaker_count": len(attenuation_by_speaker),
                    "expected_speaker_count": 40,
                }
            t_minus_n.setdefault(face, {}).setdefault(geometry, {})["native_gap"] = native_gap
            t_minus_n[face][geometry]["processed_gap"] = processed_gap
            t_minus_n[face][geometry]["native_minus_processed_gap"] = attenuation

    axis_contrast_names = {
        "orth_plus_minus_baseline": ("orth_plus", "baseline"),
        "orth_plus_minus_orth_minus": ("orth_plus", "orth_minus"),
        "full_plus_minus_baseline": ("full_plus", "baseline"),
        "full_minus_minus_baseline": ("full_minus", "baseline"),
        "orth_plus_minus_random_mean": ("orth_plus", "random_mean"),
        "random_mean_minus_baseline": ("random_mean", "baseline"),
    }
    axis_effects: dict[str, Any] = {}
    axis_clip_contrasts: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for face in FACE_IDS:
        axis_effects[face] = {}
        for geometry in GEOMETRIES:
            axis_effects[face][geometry] = {"conditions": {}, "contrasts": {}}
            for condition in CONDITIONS:
                for metric in AXIS_METRICS:
                    values = {
                        sample: axis_values[(face, sample, geometry, condition)][metric]
                        for sample in expected_samples if (face, sample, geometry, condition) in axis_values
                    }
                    axis_effects[face][geometry]["conditions"].setdefault(condition, {})[metric] = _effect_summary(
                        values, speaker_by_sample, expected_samples, draws=bootstrap_draws,
                    )
            for contrast_name, (positive, negative) in axis_contrast_names.items():
                for metric in AXIS_METRICS:
                    values: dict[str, float] = {}
                    for sample in expected_samples:
                        if positive == "random_mean":
                            positive_values = [axis_values.get((face, sample, geometry, f"random{i}"), {}).get(metric) for i in (1, 2, 3)]
                            p = float(np.mean(positive_values)) if all(value is not None for value in positive_values) else None
                        else:
                            p = axis_values.get((face, sample, geometry, positive), {}).get(metric)
                        if negative == "random_mean":
                            negative_values = [axis_values.get((face, sample, geometry, f"random{i}"), {}).get(metric) for i in (1, 2, 3)]
                            n = float(np.mean(negative_values)) if all(value is not None for value in negative_values) else None
                        else:
                            n = axis_values.get((face, sample, geometry, negative), {}).get(metric)
                        if p is not None and n is not None:
                            values[sample] = p - n
                    axis_clip_contrasts[(face, geometry, metric, contrast_name)] = values
                    axis_effects[face][geometry]["contrasts"].setdefault(contrast_name, {})[metric] = _effect_summary(
                        values, speaker_by_sample, expected_samples, draws=bootstrap_draws,
                    )

    # Completeness includes every predeclared row, including the secondary
    # face and secondary arms. Paired estimates above remain visible if not.
    result_grid_complete = len(results_by_id) == len(plan_by_id) and not missing_row_ids
    artifact_complete = result_grid_complete and row_validation_count == len(plan_by_id) and not row_failures
    complete_status_samples = {
        (str(item.get("face_id")), str(item.get("sample_id")))
        for item in status_rows if item.get("status") == "complete" and item.get("event") == "clip_complete"
    }
    expected_status = {(face, sample) for face in FACE_IDS for sample in expected_samples}
    status_grid_complete = complete_status_samples == expected_status
    run_rows_sealed = result_grid_complete and status_grid_complete
    if not status_grid_complete:
        artifact_complete = False

    # The primary source-axis contrast needs baseline, orth_plus, and
    # orth_minus on the primary face. Secondary T four-cell rows may have no
    # guard-20 window and secondary arms may fail validation without making a
    # complete primary cohort ineligible.
    primary_face = PRIMARY_FACE
    primary_scope = _primary_scope_status(
        expected_samples, speaker_by_sample, metric_rows, score_supports,
        run_rows_sealed=run_rows_sealed,
    )
    primary_scope_eligible = bool(primary_scope["eligible"])

    gates: dict[str, Any] = {}
    for contrast_name in ("orth_plus_minus_baseline", "orth_plus_minus_orth_minus"):
        for metric in ("C", "C_anchor"):
            summary = axis_effects.get(primary_face, {}).get("raw", {}).get("contrasts", {}).get(contrast_name, {}).get(metric)
            lower = summary.get("speaker_bootstrap_99_ci", [None, None])[0] if summary else None
            eligible = bool(primary_scope_eligible and summary and summary["eligible_for_intended_cohort"])
            key = f"{contrast_name}:{metric}"
            gates[key] = {
                "contrast": contrast_name, "metric": metric, "face_id": primary_face,
                "estimate": summary,
                "eligible": eligible,
                "passed": bool(eligible and lower is not None and float(lower) > 0.0),
            }

    # The cached temporal null keeps all 80 clip identities and all 31 lags.
    # The same permutation vector is re-used for every arm and both faces.
    null_clip: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    null_diag: list[dict[str, Any]] = []
    null_failures: list[dict[str, Any]] = []
    if null_count > 0:
        for sample in expected_samples:
            ordered = order_by_sample[sample]
            source_rows_by_face = {
                face: {
                    condition: plan_by_id.get(f"{sample}__{face}__source_axis__natural__{condition}")
                    for condition in CONDITIONS
                }
                for face in FACE_IDS
            }
            first_row = source_rows_by_face["primary"]["baseline"]
            if first_row is None:
                null_failures.append({"sample_id": sample, "error": "missing_primary_baseline_plan"})
                continue
            n_a: int | None = None
            hashes: list[str] = []
            face_query_counts: dict[str, int] = {}
            staged_null: dict[tuple[str, str, str, str, str], np.ndarray] = {}
            try:
                sample_context_path = _safe_artifact_path(root, str(first_row["context_npz"]))
                if sample in context_cache:
                    n_a = int(len(context_cache[sample][0][SOURCE_AUDIO_KEY]))
                elif sample_context_path.is_file():
                    with np.load(sample_context_path, allow_pickle=False) as archive:
                        n_a = int(len(archive[SOURCE_AUDIO_KEY]))
                else:
                    raise ConsumerError("cannot establish full RAW-natural A permutation domain")
                permutations, hashes = make_temporal_permutations(n_a, ordered_eval_index=ordered, count=null_count)
                for face in FACE_IDS:
                    group_n_a = None
                    query_ref: np.ndarray | None = None
                    for condition in CONDITIONS:
                        planned = source_rows_by_face[face][condition]
                        if planned is None or str(planned["row_id"]) not in score_paths:
                            raise ConsumerError(f"null lacks {face}/{condition} score row")
                        with np.load(score_paths[str(planned["row_id"])], allow_pickle=False) as archive:
                            cross_raw = archive["cross_distance_raw_float32"]
                            cross_unit = archive["cross_distance_unit_float32"]
                            zero_raw = archive["cross_distance_to_zero_raw_float32"]
                            zero_unit = archive["cross_distance_to_zero_unit_float32"]
                            queries = archive["guard20_indices_int32"].astype(np.int64)
                            lags = archive["lag_k_int32"].astype(np.int64)
                            if group_n_a is None:
                                group_n_a = cross_raw.shape[1]
                                query_ref = queries
                            if cross_raw.shape[1] != n_a or cross_unit.shape != cross_raw.shape:
                                raise ConsumerError("permutation domain differs across arms/faces")
                            if query_ref is None:
                                query_ref = queries.copy()
                            elif not np.array_equal(queries, query_ref):
                                raise ConsumerError("source-axis guard20 queries differ across arms")
                            if not np.array_equal(lags, np.arange(-15, 16, dtype=np.int64)):
                                raise ConsumerError("null lag domain differs from the frozen 31 columns")
                            observed_raw = np.asarray(archive["distance_matrix_raw_float32"])[queries]
                            observed_unit = np.asarray(archive["distance_matrix_unit_float32"])[queries]
                            null_scores = {"raw": {metric: np.empty(null_count, dtype=np.float64) for metric in AXIS_METRICS},
                                           "unit": {metric: np.empty(null_count, dtype=np.float64) for metric in AXIS_METRICS}}
                            for p_index, permutation in enumerate(permutations):
                                for geometry, cross, zero in (("raw", cross_raw, zero_raw), ("unit", cross_unit, zero_unit)):
                                    matrix = gather_temporal_null_matrix(cross, zero, queries, permutation, lags)
                                    summary = curve_summary(matrix, anchor_column=anchor_column)
                                    for metric in ("C", "C_anchor", "B", "D"):
                                        null_scores[geometry][metric][p_index] = float(summary[metric])
                                    null_scores[geometry]["search_gain"][p_index] = float(summary["C"])
                            for geometry, observed_matrix in (("raw", observed_raw), ("unit", observed_unit)):
                                observed_summary = curve_summary(observed_matrix, anchor_column=anchor_column)
                                for metric in AXIS_METRICS:
                                    staged_null[(face, geometry, condition, metric, sample)] = null_scores[geometry][metric].copy()
                                    staged_null[(face, geometry, f"observed:{condition}", metric, sample)] = np.full(
                                        null_count,
                                        float(observed_summary["C"] if metric == "search_gain" else observed_summary[metric]),
                                        dtype=np.float64,
                                    )
                    face_query_counts[face] = int(len(query_ref)) if query_ref is not None else 0
                null_diag.append({
                    "sample_id": sample, "ordered_eval_index": ordered, "seed": DEFAULT_SEED + ordered,
                    "permutation_domain_n": n_a, "permutation_count": null_count,
                    "permutation_sha256": hashes,
                    "guard20_queries_by_face": face_query_counts,
                    "same_permutation_across_arms_and_faces": True,
                    "scores_available_for_all_arms_and_faces": True,
                })
                for (face, geometry, condition, metric, sample_id), values in staged_null.items():
                    null_clip[(face, geometry, condition, metric)][sample_id] = values
            except Exception as exc:
                null_failures.append({"sample_id": sample, "ordered_eval_index": ordered, "error_type": type(exc).__name__, "error": str(exc)})
                if n_a is not None and hashes:
                    null_diag.append({
                        "sample_id": sample, "ordered_eval_index": ordered, "seed": DEFAULT_SEED + ordered,
                        "permutation_domain_n": n_a, "permutation_count": null_count,
                        "permutation_sha256": hashes,
                        "guard20_queries_by_face": face_query_counts,
                        "same_permutation_across_arms_and_faces": True,
                        "scores_available_for_all_arms_and_faces": False,
                    })

    null_summary: dict[str, Any] = {"permutation_count": null_count, "generator": "PCG64", "seed_rule": "20260930+ordered_eval_index", "permutation_rows": null_diag, "failures": null_failures, "by_face": {}}
    if null_count > 0:
        for face in FACE_IDS:
            null_summary["by_face"][face] = {}
            for geometry in GEOMETRIES:
                null_summary["by_face"][face][geometry] = {"conditions": {}, "orth_plus_minus_baseline": {}}
                for condition in CONDITIONS:
                    condition_values = {}
                    for metric in AXIS_METRICS:
                        observed_map = null_clip.get((face, geometry, f"observed:{condition}", metric), {})
                        null_map = null_clip.get((face, geometry, condition, metric), {})
                        observed_scalar = {sample: float(np.mean(values, dtype=np.float64)) for sample, values in observed_map.items()}
                        null_scalar = {sample: float(np.mean(values, dtype=np.float64)) for sample, values in null_map.items()}
                        condition_values[metric] = {
                            "observed": _effect_summary(observed_scalar, speaker_by_sample, expected_samples, draws=bootstrap_draws),
                            "mean_null": _effect_summary(null_scalar, speaker_by_sample, expected_samples, draws=bootstrap_draws),
                        }
                    null_summary["by_face"][face][geometry]["conditions"][condition] = condition_values

                for metric in AXIS_METRICS:
                    observed_values: dict[str, float] = {}
                    clip_mean_null: dict[str, float] = {}
                    sample_diffs: dict[str, np.ndarray] = {}
                    for sample in expected_samples:
                        plus_obs = null_clip.get((face, geometry, "observed:orth_plus", metric), {}).get(sample)
                        base_obs = null_clip.get((face, geometry, "observed:baseline", metric), {}).get(sample)
                        plus_null = null_clip.get((face, geometry, "orth_plus", metric), {}).get(sample)
                        base_null = null_clip.get((face, geometry, "baseline", metric), {}).get(sample)
                        if any(value is None for value in (plus_obs, base_obs, plus_null, base_null)):
                            continue
                        observed_values[sample] = float(plus_obs[0] - base_obs[0])
                        diffs = plus_null - base_null
                        sample_diffs[sample] = diffs
                        clip_mean_null[sample] = float(np.mean(diffs, dtype=np.float64))
                    observed_effect = _effect_summary(observed_values, speaker_by_sample, expected_samples, draws=bootstrap_draws)
                    null_effect = _effect_summary(clip_mean_null, speaker_by_sample, expected_samples, draws=bootstrap_draws)
                    null_means_by_draw: list[float] = []
                    for p_index in range(null_count):
                        draw_by_sample = {sample: float(values[p_index]) for sample, values in sample_diffs.items()}
                        group: dict[str, list[float]] = defaultdict(list)
                        for sample, value in draw_by_sample.items():
                            group[speaker_by_sample[sample]].append(value)
                        complete = [float(np.mean(items, dtype=np.float64)) for items in group.values() if len(items) == 2]
                        if complete:
                            null_means_by_draw.append(float(np.mean(complete, dtype=np.float64)))
                    observed_mean = observed_effect["equal_speaker_mean"]
                    null_p = None
                    if metric == "C" and observed_mean is not None and len(null_means_by_draw) == null_count:
                        null_p = float((1 + np.count_nonzero(np.asarray(null_means_by_draw) >= float(observed_mean))) / (null_count + 1))
                    speaker_diff = {}
                    observed_speakers = observed_effect["complete_speaker_values"]
                    null_speakers = null_effect["complete_speaker_values"]
                    for speaker in sorted(set(observed_speakers) & set(null_speakers)):
                        speaker_diff[speaker] = float(observed_speakers[speaker] - null_speakers[speaker])
                    difference_bootstrap = bootstrap_speaker_mean(speaker_diff, draws=bootstrap_draws, seed=DEFAULT_SEED) if len(speaker_diff) >= 2 else None
                    null_summary["by_face"][face][geometry]["orth_plus_minus_baseline"][metric] = {
                        "observed": observed_effect,
                        "mean_null": null_effect,
                        "null_mean_by_permutation": null_means_by_draw,
                        "one_sided_p": null_p if metric == "C" else None,
                        "observed_minus_mean_null_speaker_values": speaker_diff,
                        "observed_minus_mean_null_equal_speaker_mean": difference_bootstrap["mean"] if difference_bootstrap else None,
                        "observed_minus_mean_null_speaker_bootstrap_99_ci": difference_bootstrap["ci99"] if difference_bootstrap else None,
                        "eligible_for_intended_cohort": len(null_means_by_draw) == null_count and observed_effect["eligible_for_intended_cohort"],
                    }

    all_expected_result_rows = len(plan_by_id)
    status = "complete" if artifact_complete else "INCOMPLETE_SUPPORT"
    return {
        "schema_version": 1,
        "status": status,
        "classification": "independent CPU validation and predeclared summary; no model forwards",
        "run_dir": str(root),
        "protocol_sha256": _file_sha256(protocol_path),
        "row_manifest_sha256": row_bytes_sha,
        "calibration_independent_audit": cal_audit,
        "global_anchor": {"column": anchor_column, "k": anchor_k, "official_syncnet_offset": official_offset},
        "support_denominator": {
            "expected_clips": 80, "expected_speakers": 40,
            "expected_score_rows": all_expected_result_rows, "scored_result_rows": len(results_by_id),
            "independently_validated_rows": row_validation_count,
            "missing_row_count": len(missing_row_ids), "missing_row_ids": missing_row_ids,
            "all_rows_complete": artifact_complete,
            "complete_clip_status_rows": len(complete_status_samples),
            "expected_clip_status_rows": len(expected_status),
            "all_planned_result_and_status_rows_sealed": run_rows_sealed,
        },
        "artifact_failures": row_failures,
        "attempt_failures_preserved": attempt_failures,
        "context_validation": {
            "sample_count": len(context_checks_by_sample), "valid_sample_count": sum(bool(v.get("valid")) for v in context_checks_by_sample.values()),
            "errors": context_errors,
            "intervention_checks": {sample: value["interventions"] for sample, value in context_checks_by_sample.items()},
        },
        "support_checks": {"groups_checked": len(score_supports), "failures": support_failures},
        "cached_embedding_parity": {
            "preselected_ordered_indices": [0, 39, 79] if verify_cached_embeddings else [],
            "preselected_aligned_lag_rows_validated": len(cached_parity_rows),
            "preselected_aligned_lag_rows": cached_parity_rows,
            "full_cross_matrix": {
                "algorithm": "independent NumPy float32 pairwise L2 with SyncNet eps=1e-6; raw and row-unit geometry",
                "validated_score_rows": cross_parity_row_count,
                "expected_score_rows": len(plan_by_id),
                "unique_lip_audio_embedding_pairs": len(cross_pair_parity),
                "duplicate_pair_rows_hash_matched": sum(max(0, uses - 1) for uses in cross_pair_use_count.values()),
                "tolerance_atol": 5e-4,
                "valid": cross_parity_row_count == len(plan_by_id),
                "pairs": [
                    {**record, "score_row_references": cross_pair_use_count[key]}
                    for key, record in sorted(cross_pair_parity.items())
                ],
            },
        },
        "score_support_summaries_by_row": score_support_summaries,
        "four_cell": {"C/B/D and C_anchor G/E/I; clip then speaker weighting": fourcell_contrasts, "T_minus_N": t_minus_n},
        "source_axis": axis_effects,
        "primary_gates": {
            "face_id": primary_face, "geometry": "raw", "support": "guard20",
            **primary_scope,
            "gates": gates, "conjunction_pass": bool(gates) and all(item["passed"] for item in gates.values()),
            "eligible": primary_scope_eligible,
        },
        "temporal_null": null_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--calibration-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--null-count", type=int, default=DEFAULT_NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--skip-cached-embedding-parity", action="store_true")
    args = parser.parse_args(argv)
    result = consume_evaluation_run(
        args.run_dir, calibration_dir=args.calibration_dir, null_count=args.null_count,
        bootstrap_draws=args.bootstrap_draws,
        verify_cached_embeddings=not args.skip_cached_embedding_parity,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "validated_rows": result["support_denominator"]["independently_validated_rows"], "missing_rows": result["support_denominator"]["missing_row_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
