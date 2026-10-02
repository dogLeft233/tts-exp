#!/usr/bin/env python3
"""Second-pass verifier for the frozen 11,776-endpoint WORLD family artifact.

This program is deliberately independent of the producer and of the primary
consumer implementation. It checks the consumer's sealed output inventory,
reassembles clip-level endpoints from its global/RF context artifacts, checks
the 80-clip/40-speaker map, and independently regenerates the shared speaker
bootstrap and max-|t| intervals. It prints only integrity/coverage status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CONSUMER_SHA256 = "85f5c83ab1c0a5ab2eaeae5b9af4b385b447a423a3d9a42f385439136d89847b"
EXPECTED_DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
EXPECTED_ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
EXPECTED_PHONE_INVENTORY_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"

CONSUMER_DIR_NAME = "independent_world_phone_shape_mean_dynamic_consumer_20261001_attempt18.py"
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
AUDIO = ("A_RAW", "A_RT")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C_fixed_anchor", "B_median", "D_min", "D0", "search_gain", "optimal_offset_frames")
METRIC_ARRAYS = (*METRICS, "C_best", "D_fixed_anchor")
CONTRASTS = (
    "mean", "trajectory", "joint", "grid_cycle", "mean_minus_trajectory",
    "mean_beyond_grid", "trajectory_beyond_grid", "joint_beyond_grid", "factorial_interaction",
)
TREATMENTS = ("q10", "q01", "q11", "SELF_GRID")
PHONE_LABELS = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)
PHASES = ("whole", "begin", "middle", "end")
LOCAL_ESTIMANDS = ("observed", "time_excess")
LOCAL_STAT_NAMES = (
    "observed_fixed_denominator_mean", "time_null_mean", "time_excess",
    "exposure_fraction", "exposed_window_count", "exact_token_count",
    "conditional_mean_exposed_windows_only", "stft50_observed_fixed_denominator_mean",
    "stft50_time_null_mean", "stft50_time_excess", "stft50_exposure_fraction",
)
OVERALL_STAT_NAMES = ("observed_fixed_denominator_mean", "time_null_mean", "time_excess")
GLOBAL_NULL_SEEDS = {"A_RAW": 20260930, "A_RT": 20270930}
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
SPEAKERS = 40
CLIPS = 80
CONTEXTS = 160
FAMILY_SIZE = 11_776


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close_array(actual: np.ndarray, expected: np.ndarray, name: str, *, atol: float = 2e-11) -> None:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.allclose(a, e, rtol=2e-11, atol=atol, equal_nan=True):
        delta = np.abs(a - e) if a.shape == e.shape else np.asarray([math.inf])
        raise VerificationError(f"{name} differs from independent reconstruction; shape={a.shape}/{e.shape}, max_abs={np.nanmax(delta, initial=0.0)}")


def expected_endpoint_keys() -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    for contrast in CONTRASTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for metric in METRICS:
                        for audio_domain in AUDIO:
                            for estimand in ("observed", "time_excess"):
                                keys.append(("global", contrast, direction, geometry, support, metric, audio_domain, estimand))
    for treatment in TREATMENTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for label in PHONE_LABELS:
                        for phase in PHASES:
                            for estimand in LOCAL_ESTIMANDS:
                                keys.append(("local", treatment, direction, geometry, support, label, phase, estimand))
    for treatment in TREATMENTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for estimand in LOCAL_ESTIMANDS:
                        keys.append(("overall", treatment, direction, geometry, support, estimand))
    if len(keys) != FAMILY_SIZE or len(set(keys)) != FAMILY_SIZE:
        raise VerificationError("frozen endpoint key construction does not yield 11,776 unique rows")
    return keys


def key_json(key: tuple[Any, ...]) -> str:
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def contrast(values: dict[str, float], name: str) -> float:
    if name == "mean": return values["q10"] - values["q00"]
    if name == "trajectory": return values["q01"] - values["q00"]
    if name == "joint": return values["q11"] - values["q00"]
    if name == "grid_cycle": return values["SELF_GRID"] - values["q00"]
    if name == "mean_minus_trajectory": return values["q10"] - values["q01"]
    if name == "mean_beyond_grid": return values["q10"] - values["SELF_GRID"]
    if name == "trajectory_beyond_grid": return values["q01"] - values["SELF_GRID"]
    if name == "joint_beyond_grid": return values["q11"] - values["SELF_GRID"]
    if name == "factorial_interaction": return values["q11"] - values["q10"] - values["q01"] + values["q00"]
    raise VerificationError(f"unknown contrast {name}")


def contrast_batch(values: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "mean": return values["q10"] - values["q00"]
    if name == "trajectory": return values["q01"] - values["q00"]
    if name == "joint": return values["q11"] - values["q00"]
    if name == "grid_cycle": return values["SELF_GRID"] - values["q00"]
    if name == "mean_minus_trajectory": return values["q10"] - values["q01"]
    if name == "mean_beyond_grid": return values["q10"] - values["SELF_GRID"]
    if name == "trajectory_beyond_grid": return values["q01"] - values["SELF_GRID"]
    if name == "joint_beyond_grid": return values["q11"] - values["SELF_GRID"]
    if name == "factorial_interaction": return values["q11"] - values["q10"] - values["q01"] + values["q00"]
    raise VerificationError(f"unknown contrast {name}")


def _safe_rel(root: Path, rel: str) -> Path:
    base = root.resolve(strict=True)
    path = (base / rel).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise VerificationError(f"output ref escapes consumer root: {rel}") from exc
    return path


def verify_inventory(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("output_files")
    if not isinstance(rows, list):
        raise VerificationError("output manifest has no file inventory")
    expected: dict[str, tuple[str, int]] = {}
    for row in rows:
        rel = str(row.get("path", ""))
        if not rel or rel in expected:
            raise VerificationError("output inventory has an empty or duplicate member path")
        expected[rel] = (str(row.get("sha256", "")), int(row.get("bytes", -1)))
    observed_paths: set[str] = set()
    mismatches = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"consumer output contains symlink: {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel == "independent_output_manifest.json":
                continue
            observed_paths.add(rel)
            if rel not in expected:
                raise VerificationError(f"unlisted output member: {rel}")
            digest, size = expected[rel]
            if path.stat().st_size != size or sha256_file(path) != digest:
                mismatches += 1
                raise VerificationError(f"output member SHA/size mismatch: {rel}")
    if observed_paths != set(expected):
        missing = sorted(set(expected) - observed_paths)
        raise VerificationError(f"listed output files missing: {missing[:5]} (count={len(missing)})")
    return {"entry_count": len(expected), "verified_count": len(observed_paths), "mismatch_count": mismatches}


def verify_permutations(root: Path, index_row: dict[str, Any], domain: str) -> tuple[np.ndarray, int]:
    record = index_row["permutation_banks"][domain]
    seed = GLOBAL_NULL_SEEDS[domain] + int(index_row["ordered_eval_index"])
    if int(record.get("seed", -1)) != seed:
        raise VerificationError(f"{domain} null seed does not match ordered context index")
    path = _safe_rel(root, str(record["path"]))
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"permutation_indices_int32"}:
            raise VerificationError(f"unexpected {domain} permutation NPZ key set")
        bank = np.asarray(data["permutation_indices_int32"])
    audio_n = int(record["audio_population_n"])
    if bank.dtype != np.int32 or bank.shape != (100, audio_n) or list(bank.shape) != record.get("shape"):
        raise VerificationError(f"{domain} permutation bank has wrong shape/dtype")
    if sha256_file(path) != record.get("sha256") or array_sha(bank) != record.get("array_sha256"):
        raise VerificationError(f"{domain} permutation bank hash differs from its context index")
    rng = np.random.Generator(np.random.PCG64(seed))
    expected = np.stack([rng.permutation(audio_n) for _ in range(100)], axis=0).astype(np.int32)
    if not np.array_equal(bank, expected):
        raise VerificationError(f"{domain} permutation bank is not the frozen PCG64 sequence")
    return bank, audio_n


def metric_index(value: np.ndarray, names: np.ndarray, name: str) -> float:
    pos = np.flatnonzero(names == name)
    if len(pos) != 1:
        raise VerificationError(f"global metric name {name} is missing or duplicated")
    return float(value[int(pos[0])])


def verify_context_indexes(root: Path, inventory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    global_rows = read_jsonl(root / "global_context_index.jsonl")
    rf_rows = read_jsonl(root / "receptive_field_context_index.jsonl")
    if len(global_rows) != CONTEXTS or len(rf_rows) != CONTEXTS:
        raise VerificationError("global/RF context index count is not 160")
    key = lambda row: (str(row.get("sample_id")), str(row.get("direction")))
    gr = {key(row): row for row in global_rows}
    rr = {key(row): row for row in rf_rows}
    if len(gr) != CONTEXTS or set(gr) != set(rr):
        raise VerificationError("global/RF context indexes have duplicate or mismatched contexts")
    if {k[1] for k in gr} != set(DIRECTIONS):
        raise VerificationError("context indexes do not cover both frozen directions")
    eval_indices = sorted(int(row["ordered_eval_index"]) for row in global_rows)
    if eval_indices != list(range(CONTEXTS)):
        raise VerificationError("ordered_eval_index is not exactly 0..159")
    samples = sorted({k[0] for k in gr}, key=lambda sample: min(int(r["ordered_eval_index"]) for r in global_rows if r["sample_id"] == sample))
    if len(samples) != CLIPS:
        raise VerificationError("context indexes do not cover 80 unique clips")
    sample_speaker: dict[str, str] = {}
    for row in global_rows:
        sample = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        if sample in sample_speaker and sample_speaker[sample] != speaker:
            raise VerificationError(f"clip {sample} maps to multiple speakers")
        sample_speaker[sample] = speaker
    speaker_samples: dict[str, set[str]] = defaultdict(set)
    for sample, speaker in sample_speaker.items():
        speaker_samples[speaker].add(sample)
    if len(speaker_samples) != SPEAKERS or any(len(items) != 2 for items in speaker_samples.values()):
        raise VerificationError("40-speaker map does not contain exactly two distinct clips per speaker")
    return global_rows, rf_rows, samples, sorted(speaker_samples)


def reconstruct_clip_family(
    root: Path,
    global_rows: list[dict[str, Any]],
    rf_rows: list[dict[str, Any]],
    samples: list[str],
) -> tuple[list[tuple[Any, ...]], np.ndarray, np.ndarray, dict[str, Any]]:
    keys = expected_endpoint_keys()
    key_index = {key: i for i, key in enumerate(keys)}
    clip_values = np.full((FAMILY_SIZE, CLIPS), np.nan, dtype=np.float64)
    clip_null = np.full((FAMILY_SIZE, CLIPS), np.nan, dtype=np.float64)
    sample_index = {sample: i for i, sample in enumerate(samples)}
    rf_by_key = {(str(row["sample_id"]), str(row["direction"])): row for row in rf_rows}
    supported_contexts = 0
    for row in sorted(global_rows, key=lambda x: int(x["ordered_eval_index"])):
        sample = str(row["sample_id"])
        direction = str(row["direction"])
        pair_ix = sample_index[sample]
        global_path = _safe_rel(root, str(row["global_metrics_path"]))
        if sha256_file(global_path) != row.get("global_metrics_sha256"):
            raise VerificationError(f"global context hash mismatch: {sample}/{direction}")
        raw_bank, raw_n = verify_permutations(root, row, "A_RAW")
        rt_bank, rt_n = verify_permutations(root, row, "A_RT")
        with np.load(global_path, allow_pickle=False) as loaded:
            required = {
                "cell_names_unicode", "audio_names_unicode", "geometry_names_unicode", "support_names_unicode",
                "metric_names_unicode", "global_observed_metrics_float64", "global_null_metric_draws_float64",
                "global_null_metric_mean_float64",
            }
            if set(loaded.files) != required:
                raise VerificationError(f"unexpected global metrics keys: {sample}/{direction}")
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        names_expected = {
            "cell_names_unicode": CELLS, "audio_names_unicode": AUDIO,
            "geometry_names_unicode": GEOMETRIES, "support_names_unicode": SUPPORTS,
            "metric_names_unicode": METRIC_ARRAYS,
        }
        for name, expected in names_expected.items():
            if tuple(str(x) for x in arrays[name].tolist()) != tuple(expected):
                raise VerificationError(f"global axis labels differ: {sample}/{direction}/{name}")
        obs = arrays["global_observed_metrics_float64"]
        draws = arrays["global_null_metric_draws_float64"]
        null_means = arrays["global_null_metric_mean_float64"]
        if obs.shape != (5, 2, 2, 4, 8) or draws.shape != (5, 2, 2, 4, 8, 100) or null_means.shape != obs.shape:
            raise VerificationError(f"global context array shapes differ: {sample}/{direction}")
        if not all(np.issubdtype(a.dtype, np.floating) for a in (obs, draws, null_means)):
            raise VerificationError(f"global context arrays are not floating point: {sample}/{direction}")
        close_array(null_means, np.mean(draws, axis=-1, dtype=np.float64), f"global per-metric null means {sample}/{direction}")
        if raw_bank.shape[1] != raw_n or rt_bank.shape[1] != rt_n:
            raise VerificationError("permutation population mismatch")
        if raw_n < 1 or rt_n < 1:
            raise VerificationError("empty audio population")

        for contrast_name in CONTRASTS:
            for geo_i, geometry in enumerate(GEOMETRIES):
                for support_i, support in enumerate(SUPPORTS):
                    for metric_name in METRICS:
                        metric_i = METRIC_ARRAYS.index(metric_name)
                        for domain_i, domain in enumerate(AUDIO):
                            observed_cells = {cell: float(obs[CELLS.index(cell), domain_i, geo_i, support_i, metric_i]) for cell in CELLS}
                            if not all(math.isfinite(v) for v in observed_cells.values()):
                                continue
                            observed_value = contrast(observed_cells, contrast_name)
                            draw_cells = {cell: np.asarray(draws[CELLS.index(cell), domain_i, geo_i, support_i, metric_i], dtype=np.float64) for cell in CELLS}
                            null_contrast = contrast_batch(draw_cells, contrast_name)
                            null_mean = float(np.mean(null_contrast, dtype=np.float64))
                            for estimand, value in (("observed", observed_value), ("time_excess", observed_value - null_mean)):
                                idx = key_index[("global", contrast_name, direction, geometry, support, metric_name, domain, estimand)]
                                clip_values[idx, pair_ix] = value
                                clip_null[idx, pair_ix] = null_mean

        rf_row = rf_by_key[(sample, direction)]
        if rf_row.get("permutation_seed") != GLOBAL_NULL_SEEDS["A_RAW"] + int(row["ordered_eval_index"]):
            raise VerificationError(f"RF A_RAW permutation seed differs: {sample}/{direction}")
        if rf_row.get("permutation_array_sha256") != row["permutation_banks"]["A_RAW"].get("array_sha256"):
            raise VerificationError(f"RF does not bind shared A_RAW permutation bank: {sample}/{direction}")
        if not str(rf_row.get("phone_map_path", "")) or len(str(rf_row.get("phone_map_sha256", ""))) != 64:
            raise VerificationError(f"RF context lacks its recipient phone-map provenance: {sample}/{direction}")
        rf_path = _safe_rel(root, str(rf_row["rf_npz"]))
        if sha256_file(rf_path) != rf_row.get("rf_npz_sha256"):
            raise VerificationError(f"RF NPZ hash mismatch: {sample}/{direction}")
        with np.load(rf_path, allow_pickle=False) as loaded:
            required = {
                "phone_labels_unicode", "phase_names_unicode", "treatment_names_unicode", "geometry_names_unicode",
                "support_names_unicode", "local_endpoint_stats_float64", "local_endpoint_stat_names_unicode",
                "overall_endpoint_stats_float64", "overall_endpoint_stat_names_unicode", "null_permutations_A_RAW_int32",
                "support_full_int32", "support_valid_int32", "support_guard20_int32", "support_guard15_int32",
            }
            if not required.issubset(loaded.files):
                raise VerificationError(f"RF NPZ lacks required arrays: {sample}/{direction}")
            rf = {name: np.asarray(loaded[name]) for name in required}
        axis_names = {
            "phone_labels_unicode": PHONE_LABELS, "phase_names_unicode": PHASES,
            "treatment_names_unicode": TREATMENTS, "geometry_names_unicode": GEOMETRIES,
            "support_names_unicode": SUPPORTS, "local_endpoint_stat_names_unicode": LOCAL_STAT_NAMES,
            "overall_endpoint_stat_names_unicode": OVERALL_STAT_NAMES,
        }
        for name, expected in axis_names.items():
            if tuple(str(x) for x in rf[name].tolist()) != tuple(expected):
                raise VerificationError(f"RF axis labels differ: {sample}/{direction}/{name}")
        local = rf["local_endpoint_stats_float64"]
        overall = rf["overall_endpoint_stats_float64"]
        if local.shape != (4, 2, 4, 16, 4, 11) or overall.shape != (4, 2, 4, 3):
            raise VerificationError(f"RF endpoint array shape differs: {sample}/{direction}")
        if rf["null_permutations_A_RAW_int32"].dtype != np.int32 or not np.array_equal(rf["null_permutations_A_RAW_int32"], raw_bank):
            raise VerificationError(f"RF did not preserve the same full-domain A_RAW bank: {sample}/{direction}")
        support_map = {name: rf[f"support_{name}_int32"] for name in SUPPORTS}
        query_n = int(rf_row.get("syncnet_query_count", -1))
        if query_n < 1:
            raise VerificationError(f"RF query count is invalid: {sample}/{direction}")
        full_expected = np.arange(query_n, dtype=np.int32)
        g15_expected = np.arange(min(15, query_n), max(min(15, query_n), query_n - 15), dtype=np.int32)
        g20_expected = np.arange(min(20, query_n), max(min(20, query_n), query_n - 20), dtype=np.int32)
        support_expected = {"full": full_expected, "valid": full_expected, "guard20": g20_expected, "guard15": g15_expected}
        for name, ix in support_map.items():
            if ix.dtype != np.int32 or ix.ndim != 1 or (len(ix) and (np.any(ix < 0) or np.any(ix[1:] <= ix[:-1]))):
                raise VerificationError(f"RF support indices malformed: {sample}/{direction}/{name}")
            if not np.array_equal(ix, support_expected[name]):
                raise VerificationError(f"RF support indices differ from frozen full/valid/guard rule: {sample}/{direction}/{name}")
        # The phone-local stream is intentionally A_RAW-only. There is no local A_RT bank.
        if any("A_RT" in name for name in rf):
            raise VerificationError(f"local receptive-field artifact unexpectedly includes A_RT: {sample}/{direction}")
        for treatment_i, treatment in enumerate(TREATMENTS):
            for geo_i, geometry in enumerate(GEOMETRIES):
                for support_i, support in enumerate(SUPPORTS):
                    for phone_i, label in enumerate(PHONE_LABELS):
                        for phase_i, phase in enumerate(PHASES):
                            stats = local[treatment_i, geo_i, support_i, phone_i, phase_i]
                            observed, null_mean, excess = (float(stats[0]), float(stats[1]), float(stats[2]))
                            if math.isfinite(observed) and math.isfinite(null_mean) and math.isfinite(excess):
                                if abs((observed - null_mean) - excess) > 2e-11:
                                    raise VerificationError(f"local phone excess identity fails: {sample}/{direction}")
                            for estimand, value in (("observed", observed), ("time_excess", excess)):
                                idx = key_index[("local", treatment, direction, geometry, support, label, phase, estimand)]
                                clip_values[idx, pair_ix] = value
                                clip_null[idx, pair_ix] = null_mean
                    for estimand_i, estimand in enumerate(LOCAL_ESTIMANDS):
                        stats = overall[treatment_i, geo_i, support_i]
                        observed, null_mean, excess = map(float, stats)
                        if math.isfinite(observed) and math.isfinite(null_mean) and math.isfinite(excess) and abs((observed - null_mean) - excess) > 2e-11:
                            raise VerificationError(f"local overall excess identity fails: {sample}/{direction}")
                        value = observed if estimand == "observed" else excess
                        idx = key_index[("overall", treatment, direction, geometry, support, estimand)]
                        clip_values[idx, pair_ix] = value
                        clip_null[idx, pair_ix] = null_mean
        supported_contexts += 1
    if supported_contexts != CONTEXTS:
        raise VerificationError("not all 160 contexts were independently reconstructed")
    return keys, clip_values, clip_null, {"reconstructed_contexts": supported_contexts, "clip_count": CLIPS}


def verify_family(root: Path, clip_values: np.ndarray, clip_null: np.ndarray, samples: list[str]) -> dict[str, Any]:
    family_path = root / "family_reconstruction.npz"
    endpoints_path = root / "family_endpoints.jsonl"
    with np.load(family_path, allow_pickle=False) as loaded:
        required = {
            "speaker_order_unicode", "endpoint_key_unicode", "estimable_mask_uint8", "speaker_values_float64",
            "clip_values_by_endpoint_estimand_float64", "clip_time_null_means_float64", "bootstrap_indices_int16",
            "bootstrap_counts_uint8", "family_max_t_draws_float64",
        }
        if set(loaded.files) != required:
            raise VerificationError("family reconstruction NPZ key set differs")
        family = {name: np.asarray(loaded[name]) for name in required}
    speakers = [str(x) for x in family["speaker_order_unicode"].tolist()]
    if len(speakers) != SPEAKERS or len(set(speakers)) != SPEAKERS:
        raise VerificationError("speaker order does not contain 40 unique speakers")
    endpoint_rows = read_jsonl(endpoints_path)
    keys = expected_endpoint_keys()
    expected_keys = [key_json(key) for key in keys]
    if len(endpoint_rows) != FAMILY_SIZE or len(expected_keys) != FAMILY_SIZE:
        raise VerificationError("family endpoint output does not preserve all 11,776 planned endpoints")
    if [str(row.get("endpoint_key")) for row in endpoint_rows] != expected_keys:
        raise VerificationError("family endpoint JSONL ordering/key set differs from frozen family")
    if [str(x) for x in family["endpoint_key_unicode"].tolist()] != expected_keys:
        raise VerificationError("family reconstruction endpoint ordering differs from frozen family")
    if [int(row.get("planned_family_index", -1)) for row in endpoint_rows] != list(range(FAMILY_SIZE)):
        raise VerificationError("family endpoint index sequence is not exact")
    mask = family["estimable_mask_uint8"]
    values = family["speaker_values_float64"]
    stored_clips = family["clip_values_by_endpoint_estimand_float64"]
    stored_null = family["clip_time_null_means_float64"]
    if mask.dtype != np.uint8 or mask.shape != (FAMILY_SIZE,):
        raise VerificationError("family estimability mask shape/dtype differs")
    if values.shape != (FAMILY_SIZE, SPEAKERS) or stored_clips.shape != (FAMILY_SIZE, CLIPS) or stored_null.shape != (FAMILY_SIZE, CLIPS):
        raise VerificationError("family reconstruction matrix shapes differ")
    close_array(stored_clips, clip_values, "family clip values")
    close_array(stored_null, clip_null, "family clip time-null means")

    # Build the plan-order sample -> speaker map from the two independent context indexes.
    global_rows = read_jsonl(root / "global_context_index.jsonl")
    sample_speaker = {str(row["sample_id"]): str(row["speaker_id"]) for row in global_rows}
    speaker_index = {speaker: i for i, speaker in enumerate(speakers)}
    if set(sample_speaker) != set(samples) or set(sample_speaker.values()) != set(speakers):
        raise VerificationError("family speaker axis does not bind context-index speaker identities")
    members: dict[str, list[int]] = defaultdict(list)
    for sample_i, sample in enumerate(samples):
        members[sample_speaker[sample]].append(sample_i)
    if any(len(members[speaker]) != 2 for speaker in speakers):
        raise VerificationError("speaker bootstrap unit is not exactly two clips per speaker")
    expected_values = np.full((FAMILY_SIZE, SPEAKERS), np.nan, dtype=np.float64)
    expected_mask = np.zeros(FAMILY_SIZE, dtype=np.uint8)
    for ep in range(FAMILY_SIZE):
        clip = clip_values[ep]
        if np.isfinite(clip).all():
            for speaker, sidx in speaker_index.items():
                expected_values[ep, sidx] = float(np.mean(clip[members[speaker]], dtype=np.float64))
            if np.isfinite(expected_values[ep]).all():
                expected_mask[ep] = 1
    if not np.array_equal(mask, expected_mask):
        raise VerificationError("estimability mask does not preserve all 80 clips/40 speakers")
    close_array(values, expected_values, "speaker equal-weight values")
    if [row.get("status") for row in endpoint_rows] != ["estimable" if x else "INCOMPLETE_SUPPORT" for x in expected_mask]:
        raise VerificationError("endpoint statuses do not correspond to the full planned estimability mask")

    # Reproduce exact shared PCG64 index draws and their per-draw speaker counts.
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    boot_idx = rng.integers(0, SPEAKERS, size=(BOOTSTRAP_DRAWS, SPEAKERS), dtype=np.int16)
    counts = np.zeros((BOOTSTRAP_DRAWS, SPEAKERS), dtype=np.uint8)
    for draw in range(BOOTSTRAP_DRAWS):
        counts[draw] = np.bincount(boot_idx[draw], minlength=SPEAKERS).astype(np.uint8)
    if family["bootstrap_indices_int16"].dtype != np.int16 or not np.array_equal(family["bootstrap_indices_int16"], boot_idx):
        raise VerificationError("20,000×40 PCG64 speaker index matrix differs from frozen generator")
    if family["bootstrap_counts_uint8"].dtype != np.uint8 or not np.array_equal(family["bootstrap_counts_uint8"], counts):
        raise VerificationError("bootstrap per-draw speaker counts differ from the regenerated index matrix")

    active = np.flatnonzero(expected_mask)
    if not len(active):
        raise VerificationError("no estimable endpoints in the family")
    count64 = counts.astype(np.float64)
    point = np.mean(expected_values[active], axis=1, dtype=np.float64)
    se = np.empty(len(active), dtype=np.float64)
    for start in range(0, len(active), 96):
        block_pos = np.arange(start, min(start + 96, len(active)))
        block_vals = expected_values[active[block_pos]]
        boot = count64 @ block_vals.T / float(SPEAKERS)
        se[block_pos] = np.std(boot, axis=0, ddof=1, dtype=np.float64)
    max_t = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, len(active), 96):
        block_pos = np.arange(start, min(start + 96, len(active)))
        block_vals = expected_values[active[block_pos]]
        boot = count64 @ block_vals.T / float(SPEAKERS)
        errors = se[block_pos]
        centered = np.abs(boot - point[block_pos][None, :])
        tvals = np.divide(centered, errors[None, :], out=np.zeros_like(centered), where=errors[None, :] > 0)
        max_t = np.maximum(max_t, np.max(tvals, axis=1))
    if family["family_max_t_draws_float64"].shape != (BOOTSTRAP_DRAWS,):
        raise VerificationError("family max-|t| draw vector shape differs")
    close_array(family["family_max_t_draws_float64"], max_t, "20,000-draw family max-|t| distribution", atol=1e-9)
    critical = float(np.quantile(max_t, 0.99, method="linear"))
    for pos, ep in enumerate(active):
        record = endpoint_rows[int(ep)]
        expected_ci = [float(point[pos] - critical * se[pos]), float(point[pos] + critical * se[pos])]
        if not math.isclose(float(record["estimate_speaker_equal_mean"]), float(point[pos]), rel_tol=2e-11, abs_tol=2e-11):
            raise VerificationError(f"speaker point estimate differs at endpoint {ep}")
        if not math.isclose(float(record["bootstrap_se"]), float(se[pos]), rel_tol=2e-11, abs_tol=2e-11):
            raise VerificationError(f"speaker bootstrap SE differs at endpoint {ep}")
        close_array(np.asarray(record["ci99_simultaneous_max_t"], dtype=np.float64), np.asarray(expected_ci), f"simultaneous CI endpoint {ep}")
        if not math.isclose(float(record["family_critical_99_max_t"]), critical, rel_tol=2e-11, abs_tol=2e-11):
            raise VerificationError(f"family max-|t| critical value differs at endpoint {ep}")
    for ep in np.flatnonzero(expected_mask == 0):
        row = endpoint_rows[int(ep)]
        if row.get("estimate_speaker_equal_mean") is not None or row.get("bootstrap_se") is not None or row.get("ci99_simultaneous_max_t") is not None:
            raise VerificationError(f"incomplete endpoint {ep} has an inferential estimate")
    return {
        "endpoint_count": FAMILY_SIZE,
        "estimable_endpoint_count": int(expected_mask.sum()),
        "incomplete_endpoint_count": int((expected_mask == 0).sum()),
        "speaker_count": SPEAKERS,
        "clip_count": CLIPS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_index_sha256": array_sha(boot_idx),
        "bootstrap_count_sha256": array_sha(counts),
        "family_max_t_sha256": array_sha(max_t),
    }


def verify(root: Path, expected_consumer_sha: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = root / "independent_output_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "ANALYSIS_COMPLETE":
        raise VerificationError("primary consumer did not mark analysis complete")
    if manifest.get("consumer_source_sha256") != expected_consumer_sha or expected_consumer_sha != EXPECTED_CONSUMER_SHA256:
        raise VerificationError("primary consumer source SHA differs from the frozen attempt18 source")
    if manifest.get("design_sha256") != EXPECTED_DESIGN_SHA256 or manifest.get("metric_addendum_sha256") != EXPECTED_ADDENDUM_SHA256 or manifest.get("phone_inventory_sha256") != EXPECTED_PHONE_INVENTORY_SHA256:
        raise VerificationError("consumer manifest does not bind frozen design/addendum/CAL phone inventory")
    inventory = verify_inventory(root, manifest)
    schema = read_json(root / "analysis_schema.json")
    if schema.get("planned_family_endpoint_count") != FAMILY_SIZE or schema.get("planned_global_endpoint_count") != 3456 or schema.get("planned_local_phone_endpoint_count") != 8192 or schema.get("planned_local_overall_endpoint_count") != 128:
        raise VerificationError("analysis schema family cardinalities differ")
    if schema.get("speaker_bootstrap", {}).get("draws") != BOOTSTRAP_DRAWS or schema.get("speaker_bootstrap", {}).get("seed") != BOOTSTRAP_SEED:
        raise VerificationError("analysis schema bootstrap policy differs")
    if schema.get("time_null", {}).get("A_RAW_seed_base") != GLOBAL_NULL_SEEDS["A_RAW"] or schema.get("time_null", {}).get("A_RT_seed_base") != GLOBAL_NULL_SEEDS["A_RT"]:
        raise VerificationError("analysis schema time-null seed policy differs")
    global_rows, rf_rows, samples, speakers = verify_context_indexes(root, inventory)
    keys, clip_values, clip_null, reconstruction = reconstruct_clip_family(root, global_rows, rf_rows, samples)
    family = verify_family(root, clip_values, clip_null, samples)
    if int(manifest.get("context_count", -1)) != CONTEXTS or int(manifest.get("unique_clip_count", -1)) != CLIPS or int(manifest.get("speaker_count", -1)) != SPEAKERS:
        raise VerificationError("output manifest cohort counts differ")
    if int(manifest.get("planned_family_endpoint_count", -1)) != FAMILY_SIZE:
        raise VerificationError("output manifest omitted planned endpoints")
    if int(manifest.get("unestimable_family_endpoint_count", -1)) != family["incomplete_endpoint_count"] or int(manifest.get("estimable_family_endpoint_count", -1)) != family["estimable_endpoint_count"]:
        raise VerificationError("output manifest family support counts differ")
    if not math.isclose(float(manifest.get("family_critical_99_max_t", math.nan)), float(np.quantile(np.load(root / "family_reconstruction.npz", allow_pickle=False)["family_max_t_draws_float64"], 0.99, method="linear")), rel_tol=2e-11, abs_tol=2e-11):
        raise VerificationError("output manifest family max-|t| critical value differs from reconstruction")
    return {
        "status": "INDEPENDENT_FAMILY_RECHECK_PASS",
        "consumer_manifest_sha256": sha256_file(manifest_path),
        "consumer_source_sha256": expected_consumer_sha,
        "output_inventory": inventory,
        "cohort": {"contexts": CONTEXTS, "clips": CLIPS, "speakers": SPEAKERS, "contexts_reconstructed": reconstruction["reconstructed_contexts"]},
        "family": family,
        "null_scope": {
            "global_audio_domains": list(AUDIO),
            "phone_local_audio_domain": "A_RAW only; verified shared full-domain A_RAW permutations",
            "shared_permutation_banks_rebuilt": CONTEXTS * len(AUDIO),
            "speaker_unit": "40 speakers, equal mean of two clips each",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-consumer-sha256", default=EXPECTED_CONSUMER_SHA256)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.output_dir, args.expected_consumer_sha256)
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            if args.receipt.exists():
                raise VerificationError(f"refusing to overwrite verifier receipt: {args.receipt}")
            args.receipt.write_text(payload, encoding="utf-8")
        print(json.dumps({
            "status": result["status"],
            "contexts": result["cohort"]["contexts"],
            "clips": result["cohort"]["clips"],
            "speakers": result["cohort"]["speakers"],
            "planned_endpoints": result["family"]["endpoint_count"],
            "estimable_endpoints": result["family"]["estimable_endpoint_count"],
            "incomplete_endpoints": result["family"]["incomplete_endpoint_count"],
            "inventory_verified": result["output_inventory"]["verified_count"],
            "receipt": str(args.receipt) if args.receipt else None,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
