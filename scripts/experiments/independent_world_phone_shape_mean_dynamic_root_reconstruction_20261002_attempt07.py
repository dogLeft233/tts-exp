"""Independent local-endpoint reconstruction for the sealed WORLD phone-shape audit.

This root-side audit starts from the saved per-window G arrays, phone/RF weights,
and exact support indices. It reconstructs every phone-phase and overall local
endpoint, compares those values with the saved local arrays and family clip
matrix, then invokes the separately reviewed family verifier for global
contrasts, the 11,776-endpoint mask, shared speaker bootstrap, max-|t| draws,
and simultaneous intervals. It does not import producer/scoring code or expose
effect estimates in its receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


CONSUMER_SHA256 = "0aea5d17558c3985453119525890529f5a24addac41539b07d6b49a52d355470"
FAMILY_VERIFIER_SHA256 = "ec678ea13fc3cc5dba364d272eb590cb7e93597120b8adccf0dfc815a54d03ff"
DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
PHONE_INVENTORY_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"

CONTEXTS = 160
CLIPS = 80
SPEAKERS = 40
FAMILY_SIZE = 11_776
PHONE_ENDPOINTS = 8_192
OVERALL_ENDPOINTS = 128
TREATMENTS = ("q10", "q01", "q11", "SELF_GRID")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
PHONES = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)
PHASES = ("whole", "begin", "middle", "end")
LOCAL_STATS = (
    "observed_fixed_denominator_mean", "time_null_mean", "time_excess",
    "exposure_fraction", "exposed_window_count", "exact_token_count",
    "conditional_mean_exposed_windows_only", "stft50_observed_fixed_denominator_mean",
    "stft50_time_null_mean", "stft50_time_excess", "stft50_exposure_fraction",
)
OVERALL_STATS = ("observed_fixed_denominator_mean", "time_null_mean", "time_excess")
STAT_TOL = 2e-11
WEIGHT_TOL = 2e-11


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close_array(actual: np.ndarray, expected: np.ndarray, label: str, *, atol: float = STAT_TOL) -> float:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.allclose(a, e, rtol=STAT_TOL, atol=atol, equal_nan=True):
        delta = np.abs(a - e) if a.shape == e.shape else np.asarray([math.inf])
        raise AuditError(
            f"{label} differs: shape={a.shape}/{e.shape}, "
            f"max_abs={float(np.nanmax(delta, initial=0.0))}"
        )
    finite = np.isfinite(a) & np.isfinite(e)
    return float(np.max(np.abs(a[finite] - e[finite]), initial=0.0))


def import_family_verifier(path: Path) -> ModuleType:
    path = path.resolve(strict=True)
    observed = sha256_file(path)
    if observed != FAMILY_VERIFIER_SHA256:
        raise AuditError(f"family verifier source SHA differs: {observed}")
    spec = importlib.util.spec_from_file_location("sealed_world_family_verifier", path)
    if spec is None or spec.loader is None:
        raise AuditError("cannot load the pinned family verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "EXPECTED_CONSUMER_SHA256", None) != CONSUMER_SHA256:
        raise AuditError("family verifier is not bound to consumer attempt17")
    return module


def expected_supports(query_n: int) -> dict[str, np.ndarray]:
    full = np.arange(query_n, dtype=np.int32)
    g20 = np.arange(min(20, query_n), max(min(20, query_n), query_n - 20), dtype=np.int32)
    g15 = np.arange(min(15, query_n), max(min(15, query_n), query_n - 15), dtype=np.int32)
    return {"full": full, "valid": full.copy(), "guard20": g20, "guard15": g15}


def recipient_intervals(phone_map: dict[str, Any]) -> list[dict[str, Any]]:
    matched = phone_map.get("matched_phones")
    tier = phone_map.get("recipient_phones")
    if not isinstance(matched, list) or not isinstance(tier, list):
        raise AuditError("phone map lacks exact matched and complete recipient phone tiers")
    matched_by_index: dict[int, dict[str, Any]] = {}
    for row in matched:
        idx = int(row["recipient_phone_index"])
        if idx in matched_by_index:
            raise AuditError("phone map repeats a matched recipient token index")
        matched_by_index[idx] = row
    full_by_index: dict[int, dict[str, Any]] = {}
    for row in tier:
        idx = int(row["index"])
        if idx in full_by_index:
            raise AuditError("phone map repeats a recipient token index")
        full_by_index[idx] = row
    if not set(matched_by_index).issubset(full_by_index):
        raise AuditError("matched phone is absent from the complete recipient tier")
    result: list[dict[str, Any]] = []
    for idx, row in full_by_index.items():
        is_matched = idx in matched_by_index
        match = matched_by_index.get(idx)
        interval = row.get("interval_s")
        if not isinstance(interval, list) or len(interval) != 2:
            raise AuditError("recipient phone interval is malformed")
        start, end = map(float, interval)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise AuditError("recipient phone interval is not finite and positive")
        label = str(row.get("normalized_label", ""))
        if match is not None:
            match_interval = match.get("recipient_interval_s")
            if (
                str(match.get("recipient_normalized_label")) != label
                or not isinstance(match_interval, list) or len(match_interval) != 2
                or any(abs(float(x) - float(y)) > 1e-12 for x, y in zip(match_interval, (start, end)))
            ):
                raise AuditError("matched phone label or interval differs from the full recipient tier")
        result.append({
            "phone_index": idx, "label": label, "start": start, "end": end,
            "matched": is_matched, "word_index": int(row.get("word_index", -1)),
        })
    result.sort(key=lambda item: (item["start"], item["end"], item["phone_index"]))
    return result


def owner_arrays(times: np.ndarray, intervals: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(len(times), dtype=np.int16)
    owner = np.full(len(times), -1, dtype=np.int32)
    for token in intervals:
        mask = (times >= token["start"]) & (times < token["end"])
        count += mask.astype(np.int16)
        owner[mask] = token["phone_index"]
    return count, owner


def rebuild_rf_weights(
    unions: list[np.ndarray], centers: np.ndarray, intervals: list[dict[str, Any]], owner_count: np.ndarray,
    owner_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phone_slot = {phone: index for index, phone in enumerate(PHONES)}
    weights = np.zeros((len(PHONES), len(PHASES), len(unions)), dtype=np.float64)
    stft = np.zeros_like(weights)
    boundaries = {float(value) for token in intervals for value in (token["start"], token["end"])}
    for token in intervals:
        if token["matched"] and token["label"] in phone_slot:
            duration = token["end"] - token["start"]
            boundaries.add(token["start"] + duration / 3.0)
            boundaries.add(token["start"] + 2.0 * duration / 3.0)
    segments: list[tuple[float, float, int, int]] = []
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        active = [token for token in intervals if token["start"] <= midpoint < token["end"]]
        if len(active) != 1 or not active[0]["matched"]:
            continue
        token = active[0]
        slot = phone_slot.get(token["label"])
        if slot is None:
            continue
        phase = (midpoint - token["start"]) / (token["end"] - token["start"])
        phase_index = 1 if phase < 1 / 3 else 2 if phase < 2 / 3 else 3
        segments.append((left, right, slot, phase_index))
    for q, columns in enumerate(unions):
        cols = np.asarray(columns, dtype=np.int64)
        for token in intervals:
            slot = phone_slot.get(token["label"])
            if slot is None or not token["matched"]:
                continue
            member = (centers[cols] >= token["start"]) & (centers[cols] < token["end"])
            member &= (owner_count[cols] == 1) & (owner_index[cols] == token["phone_index"])
            if not np.any(member):
                continue
            phase = (centers[cols] - token["start"]) / (token["end"] - token["start"])
            phase_masks = (
                member,
                member & (phase < 1 / 3),
                member & (phase >= 1 / 3) & (phase < 2 / 3),
                member & (phase >= 2 / 3) & (phase < 1),
            )
            for phase_idx, mask in enumerate(phase_masks):
                weights[slot, phase_idx, q] += float(np.count_nonzero(mask)) / float(len(cols))
        left = centers[cols] - 0.025
        right = centers[cols] + 0.025
        for seg_left, seg_right, slot, phase_idx in segments:
            overlap = np.maximum(0.0, np.minimum(right, seg_right) - np.maximum(left, seg_left)) / 0.05
            mass = float(np.sum(overlap, dtype=np.float64)) / float(len(cols))
            stft[slot, 0, q] += mass
            stft[slot, phase_idx, q] += mass
    return weights, stft


def token_counts_by_label(
    intervals: list[dict[str, Any]], centers: np.ndarray, owner_count: np.ndarray, owner_index: np.ndarray,
    unions: list[np.ndarray], supports: dict[str, np.ndarray],
) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for support, indices in supports.items():
        cols = np.unique(np.concatenate([unions[int(index)] for index in indices])) if len(indices) else np.empty(0, dtype=np.int64)
        counts = {phone: [0, 0, 0, 0] for phone in PHONES}
        for token in intervals:
            if not token["matched"] or token["label"] not in counts:
                continue
            member = (centers[cols] >= token["start"]) & (centers[cols] < token["end"])
            member &= (owner_count[cols] == 1) & (owner_index[cols] == token["phone_index"])
            if not np.any(member):
                continue
            phase = (centers[cols] - token["start"]) / (token["end"] - token["start"])
            phase_masks = (
                member,
                member & (phase < 1 / 3),
                member & (phase >= 1 / 3) & (phase < 2 / 3),
                member & (phase >= 2 / 3) & (phase < 1),
            )
            for phase_idx, mask in enumerate(phase_masks):
                counts[token["label"]][phase_idx] += int(np.any(mask))
        result[support] = counts
    return result


def official_mel_starts(mel_n: int) -> np.ndarray:
    """Rebuild the frozen 80-Hz/25-fps, 16-column chunk loop independently."""
    mel_step = 16
    fps = 25
    if mel_n < mel_step:
        raise AuditError("mel clock is shorter than the frozen 16-column chunk")
    multiplier = 80.0 / float(fps)
    starts: list[int] = []
    frame_index = 0
    while True:
        start = int(frame_index * multiplier)
        if start + mel_step > mel_n:
            starts.append(mel_n - mel_step)
            break
        starts.append(start)
        frame_index += 1
    return np.asarray(starts, dtype=np.int32)


def union_rows(rf: dict[str, np.ndarray], query_n: int, mel_n: int) -> tuple[list[np.ndarray], int, int]:
    """Rebuild official chunk starts and exact five-video-frame CSR unions."""
    starts = official_mel_starts(mel_n)
    if query_n < 1 or query_n > len(starts) - 5:
        raise AuditError(
            f"SyncNet query prefix exceeds official five-frame clock: query_n={query_n}, starts_n={len(starts)}"
        )
    expected_indptr = [0]
    expected_flat: list[int] = []
    rows: list[np.ndarray] = []
    for query in range(query_n):
        columns = sorted({
            int(starts[frame_index]) + mel_column
            for frame_index in range(query, query + 5)
            for mel_column in range(16)
        })
        if not columns or columns[0] < 0 or columns[-1] >= mel_n:
            raise AuditError(f"independently rebuilt window union is outside mel clock at query {query}")
        row = np.asarray(columns, dtype=np.int32)
        rows.append(row.astype(np.int64, copy=False))
        expected_flat.extend(columns)
        expected_indptr.append(len(expected_flat))

    observed_indptr = np.asarray(rf["window_union_indptr_int32"])
    observed_flat = np.asarray(rf["window_union_mel_columns_int32"])
    wanted_indptr = np.asarray(expected_indptr, dtype=np.int32)
    wanted_flat = np.asarray(expected_flat, dtype=np.int32)
    if observed_indptr.dtype != np.int32 or observed_flat.dtype != np.int32:
        raise AuditError("window-union CSR arrays must be int32")
    if observed_indptr.shape != wanted_indptr.shape or observed_flat.shape != wanted_flat.shape:
        raise AuditError(
            f"window-union CSR dimensions differ from independent rebuild: "
            f"indptr={observed_indptr.shape}/{wanted_indptr.shape}, "
            f"columns={observed_flat.shape}/{wanted_flat.shape}"
        )
    if not np.array_equal(observed_indptr, wanted_indptr) or not np.array_equal(observed_flat, wanted_flat):
        raise AuditError("saved complete window-union CSR differs from independent official-clock rebuild")
    return rows, len(starts), len(expected_flat)


def compare_local_stats(
    rf: dict[str, np.ndarray], context: dict[str, Any], sample_index: int,
    family_clip: np.ndarray, family_null: np.ndarray, endpoint_index: dict[str, int],
) -> tuple[int, int, int, float, int, int]:
    query_n = int(context["syncnet_query_count"])
    mel_n = int(context["mel_frame_count"])
    g = np.asarray(rf["local_g_observed_float64"])
    null_g_mean = np.asarray(rf["local_g_time_null_mean_per_window_float64"])
    weights = np.asarray(rf["phone_center_coverage_weights_float64"])
    stft = np.asarray(rf["phone_stft50_sensitivity_weights_float64"])
    local = np.asarray(rf["local_endpoint_stats_float64"])
    overall = np.asarray(rf["overall_endpoint_stats_float64"])
    expected_local_shape = (4, 2, 4, 16, 4, 11)
    if g.dtype != np.float64 or g.shape != (4, 2, query_n) or not np.isfinite(g).all():
        raise AuditError(f"per-window G shape/dtype/finite check failed: {context['sample_id']}/{context['direction']}")
    if null_g_mean.dtype != np.float64 or null_g_mean.shape != g.shape or not np.isfinite(null_g_mean).all():
        raise AuditError(f"per-window null-G shape/dtype/finite check failed: {context['sample_id']}/{context['direction']}")
    if weights.dtype != np.float64 or weights.shape != (16, 4, query_n) or not np.isfinite(weights).all():
        raise AuditError(f"center weights shape/dtype/finite check failed: {context['sample_id']}/{context['direction']}")
    if stft.dtype != np.float64 or stft.shape != weights.shape or not np.isfinite(stft).all():
        raise AuditError(f"STFT weights shape/dtype/finite check failed: {context['sample_id']}/{context['direction']}")
    if local.dtype != np.float64 or local.shape != expected_local_shape:
        raise AuditError(f"phone-phase endpoint array shape/dtype differs: {context['sample_id']}/{context['direction']}")
    if overall.dtype != np.float64 or overall.shape != (4, 2, 4, 3):
        raise AuditError(f"overall endpoint array shape/dtype differs: {context['sample_id']}/{context['direction']}")
    if np.isinf(local).any() or np.isinf(overall).any():
        raise AuditError("saved local endpoint arrays contain infinity")
    if np.any(weights < -WEIGHT_TOL) or np.any(stft < -WEIGHT_TOL):
        raise AuditError("receptive-field weights contain negative mass")
    if np.any(weights > 1 + WEIGHT_TOL) or np.any(stft > 1 + WEIGHT_TOL):
        raise AuditError("per-phone receptive-field mass exceeds one")
    if not np.allclose(weights[:, 1:, :].sum(axis=1), weights[:, 0, :], rtol=0, atol=WEIGHT_TOL):
        raise AuditError("center whole-phone weights do not equal the three phase weights")
    if not np.allclose(stft[:, 1:, :].sum(axis=1), stft[:, 0, :], rtol=0, atol=WEIGHT_TOL):
        raise AuditError("STFT whole-phone weights do not equal the three phase weights")
    if np.any(weights[:, 0, :].sum(axis=0) > 1 + WEIGHT_TOL) or np.any(stft[:, 0, :].sum(axis=0) > 1 + WEIGHT_TOL):
        raise AuditError("selected phone labels assign more than the available window mass")

    supports = expected_supports(query_n)
    for name in SUPPORTS:
        stored = np.asarray(rf[f"support_{name}_int32"])
        if stored.dtype != np.int32 or not np.array_equal(stored, supports[name]):
            raise AuditError(f"{name} support does not match its frozen exact query rule")
    intervals = recipient_intervals(read_json(Path(str(context["phone_map_path"])).resolve(strict=True)))
    centers = np.arange(mel_n, dtype=np.float64) / 80.0
    saved_centers = np.asarray(rf["mel_centers_seconds_float64"])
    if saved_centers.dtype != np.float64 or not np.array_equal(saved_centers, centers):
        raise AuditError("saved mel center clock is not exact column/80")
    unions, starts_n, union_column_n = union_rows(rf, query_n, mel_n)
    expected_owner_count, expected_owner_index = owner_arrays(centers, intervals)
    owner_count = np.asarray(rf["matched_phone_owner_count_int16"])
    owner_index = np.asarray(rf["matched_phone_owner_index_int32"])
    if owner_count.dtype != np.int16 or owner_index.dtype != np.int32 or owner_count.shape != (mel_n,) or owner_index.shape != (mel_n,):
        raise AuditError("phone ownership arrays have wrong shapes or dtypes")
    if not np.array_equal(owner_count, expected_owner_count) or not np.array_equal(owner_index, expected_owner_index):
        raise AuditError("saved phone ownership differs from the complete recipient tier")
    rebuilt_weights, rebuilt_stft = rebuild_rf_weights(unions, centers, intervals, owner_count, owner_index)
    max_weight_error = close_array(weights, rebuilt_weights, "reconstructed center RF weights", atol=WEIGHT_TOL)
    max_stft_error = close_array(stft, rebuilt_stft, "reconstructed STFT sensitivity weights", atol=WEIGHT_TOL)
    token_counts = token_counts_by_label(intervals, centers, owner_count, owner_index, unions, supports)

    maximum_stats_error = 0.0
    local_values_written = 0
    local_estimable_clip_pairs = 0
    incomplete_clip_pairs = 0
    for treatment_i, treatment in enumerate(TREATMENTS):
        for geometry_i, geometry in enumerate(GEOMETRIES):
            for support_i, support in enumerate(SUPPORTS):
                ix = supports[support].astype(np.int64)
                n = len(ix)
                if n:
                    obs_g = g[treatment_i, geometry_i, ix]
                    null_g = null_g_mean[treatment_i, geometry_i, ix]
                    overall_obs = float(np.mean(obs_g, dtype=np.float64))
                    overall_null = float(np.mean(null_g, dtype=np.float64))
                    overall_expected = np.asarray((overall_obs, overall_null, overall_obs - overall_null), dtype=np.float64)
                else:
                    overall_expected = np.full(3, np.nan, dtype=np.float64)
                maximum_stats_error = max(maximum_stats_error, close_array(
                    overall[treatment_i, geometry_i, support_i], overall_expected,
                    f"overall stats {context['sample_id']}/{context['direction']}/{treatment}/{geometry}/{support}",
                ))
                for estimand, value, null_mean in (
                    ("observed", overall_expected[0], overall_expected[1]),
                    ("time_excess", overall_expected[2], overall_expected[1]),
                ):
                    key = json.dumps(("overall", treatment, context["direction"], geometry, support, estimand), ensure_ascii=False, separators=(",", ":"))
                    ep = endpoint_index[key]
                    if not (np.isnan(value) and np.isnan(family_clip[ep, sample_index])) and not np.isclose(value, family_clip[ep, sample_index], rtol=STAT_TOL, atol=STAT_TOL, equal_nan=True):
                        raise AuditError(f"overall clip endpoint differs from family matrix: {key}/{context['sample_id']}")
                    if not (np.isnan(null_mean) and np.isnan(family_null[ep, sample_index])) and not np.isclose(null_mean, family_null[ep, sample_index], rtol=STAT_TOL, atol=STAT_TOL, equal_nan=True):
                        raise AuditError(f"overall time-null clip value differs from family matrix: {key}/{context['sample_id']}")
                    local_values_written += 1
                    local_estimable_clip_pairs += int(np.isfinite(value))
                    incomplete_clip_pairs += int(not np.isfinite(value))
                for phone_i, phone in enumerate(PHONES):
                    for phase_i, phase in enumerate(PHASES):
                        if n:
                            weight = weights[phone_i, phase_i, ix]
                            stft_weight = stft[phone_i, phase_i, ix]
                            obs = float(np.dot(weight, obs_g) / float(n))
                            null_mean = float(np.dot(weight, null_g) / float(n))
                            exposure = float(np.sum(weight, dtype=np.float64) / float(n))
                            exposed = float(np.count_nonzero(weight > 0))
                            weight_sum = float(np.sum(weight, dtype=np.float64))
                            conditional = float(np.dot(weight, obs_g) / weight_sum) if weight_sum > 0 else math.nan
                            exact_tokens = float(token_counts[support][phone][phase_i])
                            stft_obs = float(np.dot(stft_weight, obs_g) / float(n))
                            stft_null = float(np.dot(stft_weight, null_g) / float(n))
                            stft_exposure = float(np.sum(stft_weight, dtype=np.float64) / float(n))
                            expected_stats = np.asarray((
                                obs, null_mean, obs - null_mean, exposure, exposed, exact_tokens,
                                conditional, stft_obs, stft_null, stft_obs - stft_null, stft_exposure,
                            ), dtype=np.float64)
                        else:
                            obs = null_mean = math.nan
                            expected_stats = np.full(len(LOCAL_STATS), np.nan, dtype=np.float64)
                        maximum_stats_error = max(maximum_stats_error, close_array(
                            local[treatment_i, geometry_i, support_i, phone_i, phase_i], expected_stats,
                            f"phone stats {context['sample_id']}/{context['direction']}/{treatment}/{geometry}/{support}/{phone}/{phase}",
                        ))
                        for estimand, value in (("observed", obs), ("time_excess", obs - null_mean)):
                            key = json.dumps(("local", treatment, context["direction"], geometry, support, phone, phase, estimand), ensure_ascii=False, separators=(",", ":"))
                            ep = endpoint_index[key]
                            stored_value = family_clip[ep, sample_index]
                            stored_null = family_null[ep, sample_index]
                            if not np.isclose(value, stored_value, rtol=STAT_TOL, atol=STAT_TOL, equal_nan=True):
                                raise AuditError(f"local clip endpoint differs from family matrix: {key}/{context['sample_id']}")
                            if not np.isclose(null_mean, stored_null, rtol=STAT_TOL, atol=STAT_TOL, equal_nan=True):
                                raise AuditError(f"local time-null clip value differs from family matrix: {key}/{context['sample_id']}")
                            local_values_written += 1
                            local_estimable_clip_pairs += int(np.isfinite(value))
                            incomplete_clip_pairs += int(not np.isfinite(value))
    return (
        local_values_written, local_estimable_clip_pairs, incomplete_clip_pairs,
        max(maximum_stats_error, max_weight_error, max_stft_error), starts_n, union_column_n,
    )


def reconstruct_local(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve(strict=True)
    manifest = read_json(root / "independent_output_manifest.json")
    if manifest.get("status") != "ANALYSIS_COMPLETE":
        raise AuditError("consumer output is not marked ANALYSIS_COMPLETE")
    if manifest.get("consumer_source_sha256") != CONSUMER_SHA256:
        raise AuditError("output manifest is not bound to sealed consumer17")
    if (manifest.get("design_sha256"), manifest.get("metric_addendum_sha256"), manifest.get("phone_inventory_sha256")) != (DESIGN_SHA256, ADDENDUM_SHA256, PHONE_INVENTORY_SHA256):
        raise AuditError("output manifest does not bind the frozen design/addendum/phone inventory")
    if (
        int(manifest.get("context_count", -1)) != CONTEXTS
        or int(manifest.get("unique_clip_count", -1)) != CLIPS
        or int(manifest.get("speaker_count", -1)) != SPEAKERS
        or int(manifest.get("score_rows_independently_rebuilt", -1)) != 1600
        or int(manifest.get("planned_family_endpoint_count", -1)) != FAMILY_SIZE
    ):
        raise AuditError("output manifest cohort, score rebuild, or planned-family count differs")

    global_rows = read_jsonl(root / "global_context_index.jsonl")
    rf_rows = read_jsonl(root / "receptive_field_context_index.jsonl")
    if len(global_rows) != CONTEXTS or len(rf_rows) != CONTEXTS:
        raise AuditError("global/RF indexes do not contain exactly 160 contexts")
    key = lambda row: (str(row.get("sample_id")), str(row.get("direction")))
    global_map = {key(row): row for row in global_rows}
    rf_map = {key(row): row for row in rf_rows}
    if len(global_map) != CONTEXTS or set(global_map) != set(rf_map):
        raise AuditError("global/RF context identities are duplicated or differ")
    if {ctx[1] for ctx in global_map} != set(DIRECTIONS):
        raise AuditError("context indexes do not include both frozen directions")
    ordered = sorted(global_rows, key=lambda row: int(row["ordered_eval_index"]))
    if [int(row["ordered_eval_index"]) for row in ordered] != list(range(CONTEXTS)):
        raise AuditError("ordered context indexes are not exactly 0..159")
    sample_order = list(dict.fromkeys(str(row["sample_id"]) for row in ordered))
    sample_index = {sample: idx for idx, sample in enumerate(sample_order)}
    sample_speaker: dict[str, str] = {}
    for row in ordered:
        sample = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        if sample in sample_speaker and sample_speaker[sample] != speaker:
            raise AuditError(f"sample maps to multiple speakers: {sample}")
        sample_speaker[sample] = speaker
    speaker_counts: dict[str, int] = {}
    for speaker in sample_speaker.values():
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    if len(sample_order) != CLIPS or len(speaker_counts) != SPEAKERS or any(n != 2 for n in speaker_counts.values()):
        raise AuditError("context index is not 80 clips with two clips per 40 speakers")

    with np.load(root / "family_reconstruction.npz", allow_pickle=False) as data:
        family = {name: np.asarray(data[name]) for name in data.files}
    required = {
        "endpoint_key_unicode", "clip_values_by_endpoint_estimand_float64", "clip_time_null_means_float64",
    }
    if set(family) != {
        "speaker_order_unicode", "endpoint_key_unicode", "estimable_mask_uint8", "speaker_values_float64",
        "clip_values_by_endpoint_estimand_float64", "clip_time_null_means_float64", "bootstrap_indices_int16",
        "bootstrap_counts_uint8", "family_max_t_draws_float64",
    }:
        raise AuditError("family reconstruction NPZ keys differ from frozen consumer schema")
    if not required.issubset(family):
        raise AuditError("family reconstruction NPZ lacks clip-value arrays")
    keys = [str(value) for value in family["endpoint_key_unicode"].tolist()]
    endpoint_index = {value: idx for idx, value in enumerate(keys)}
    if len(keys) != FAMILY_SIZE or len(endpoint_index) != FAMILY_SIZE:
        raise AuditError("family clip-value endpoint axis is not 11,776 unique keys")
    clip_values = np.asarray(family["clip_values_by_endpoint_estimand_float64"])
    clip_null = np.asarray(family["clip_time_null_means_float64"])
    if clip_values.dtype != np.float64 or clip_null.dtype != np.float64 or clip_values.shape != (FAMILY_SIZE, CLIPS) or clip_null.shape != clip_values.shape:
        raise AuditError("family clip and null matrices have wrong dtype/shape")
    if np.isinf(clip_values).any() or np.isinf(clip_null).any():
        raise AuditError("family clip/null arrays contain infinity")
    local_key_count = sum(1 for value in keys if value.startswith('["local"') or value.startswith('["overall"'))
    if local_key_count != PHONE_ENDPOINTS + OVERALL_ENDPOINTS:
        raise AuditError(f"family local endpoint cardinality is {local_key_count}, expected 8,320")

    local_clip_values = 0
    local_estimable = 0
    local_incomplete = 0
    max_reconstruction_error = 0.0
    seen_phone_maps = 0
    total_official_mel_starts = 0
    total_window_union_rows = 0
    total_window_union_columns = 0
    for row in ordered:
        ctx_key = (str(row["sample_id"]), str(row["direction"]))
        rf_row = rf_map[ctx_key]
        rf_path = (root / str(rf_row["rf_npz"])).resolve(strict=True)
        try:
            rf_path.relative_to(root)
        except ValueError as exc:
            raise AuditError("RF artifact path escapes the sealed consumer output") from exc
        if sha256_file(rf_path) != rf_row.get("rf_npz_sha256"):
            raise AuditError(f"RF NPZ hash differs from RF context index: {ctx_key}")
        phone_path = Path(str(rf_row.get("phone_map_path", ""))).resolve(strict=True)
        if sha256_file(phone_path) != rf_row.get("phone_map_sha256"):
            raise AuditError(f"recipient phone-map hash differs from RF context index: {ctx_key}")
        with np.load(rf_path, allow_pickle=False) as loaded:
            rf = {name: np.asarray(loaded[name]) for name in loaded.files}
        expected_axes = {
            "phone_labels_unicode": PHONES, "phase_names_unicode": PHASES,
            "treatment_names_unicode": TREATMENTS, "geometry_names_unicode": GEOMETRIES,
            "support_names_unicode": SUPPORTS, "local_endpoint_stat_names_unicode": LOCAL_STATS,
            "overall_endpoint_stat_names_unicode": OVERALL_STATS,
        }
        for name, values in expected_axes.items():
            if name not in rf or tuple(str(x) for x in rf[name].tolist()) != tuple(values):
                raise AuditError(f"RF axis labels differ: {ctx_key}/{name}")
        if str(rf_row.get("sample_id")) != ctx_key[0] or str(rf_row.get("direction")) != ctx_key[1]:
            raise AuditError("RF index row identity differs from its global context row")
        if str(rf_row.get("phone_map_path")) == "" or len(str(rf_row.get("phone_map_sha256", ""))) != 64:
            raise AuditError("RF context lacks a hash-bound phone map")
        n_values, n_estimable, n_incomplete, error, starts_n, union_column_n = compare_local_stats(
            rf, rf_row, sample_index[ctx_key[0]], clip_values, clip_null, endpoint_index,
        )
        local_clip_values += n_values
        local_estimable += n_estimable
        local_incomplete += n_incomplete
        max_reconstruction_error = max(max_reconstruction_error, error)
        seen_phone_maps += 1
        total_official_mel_starts += starts_n
        total_window_union_rows += int(rf_row["syncnet_query_count"])
        total_window_union_columns += union_column_n
    # The family cardinalities include both directions; each context row is one
    # direction, so it contributes half of each direction-indexed family slice.
    expected_values = CONTEXTS * ((PHONE_ENDPOINTS + OVERALL_ENDPOINTS) // len(DIRECTIONS))
    if local_clip_values != expected_values:
        raise AuditError(f"local family clip-value comparisons={local_clip_values}, expected {expected_values}")
    return {
        "status": "LOCAL_ENDPOINT_RECONSTRUCTION_PASS",
        "consumer_manifest_sha256": sha256_file(root / "independent_output_manifest.json"),
        "consumer_source_sha256": CONSUMER_SHA256,
        "contexts_reconstructed": seen_phone_maps,
        "clips": len(sample_order),
        "speakers": len(speaker_counts),
        "local_endpoint_count": PHONE_ENDPOINTS + OVERALL_ENDPOINTS,
        "local_clip_estimand_comparisons": local_clip_values,
        "finite_local_clip_estimands": local_estimable,
        "incomplete_local_clip_estimands": local_incomplete,
        "window_union_csr_reconstruction": {
            "context_count": seen_phone_maps,
            "official_mel_start_count": total_official_mel_starts,
            "syncnet_query_rows": total_window_union_rows,
            "mel_column_entries": total_window_union_columns,
            "query_bound": "query_n <= len(official_80Hz_25fps_16column_starts) - 5",
            "exact_indptr_and_column_equality": True,
        },
        "max_abs_reconstruction_residual": max_reconstruction_error,
        "local_reconstruction_scope": {
            "source_arrays": [
                "local_g_observed_float64", "local_g_time_null_mean_per_window_float64",
                "phone_center_coverage_weights_float64", "phone_stft50_sensitivity_weights_float64",
                "support_full_int32", "support_valid_int32", "support_guard20_int32", "support_guard15_int32",
            ],
            "recomputed": [
                "complete-recipient phone ownership", "mel-center and STFT sensitivity weights",
                "exact matched-token counts", "phone-phase observed/null/excess and exposure stats",
                "overall observed/null/excess stats", "official mel-start and full CSR window-union exact reconstruction",
                "all local clip values against the 11,776 family matrix",
            ],
            "score_matrix_rebuilds": "consumer manifest binds the primary consumer's independent 1,600/1,600 full-V×full-A score reconstruction; this root audit does not duplicate the expensive score-matrix pass",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consumer-source", type=Path, required=True)
    parser.add_argument("--family-verifier", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        out = args.output_dir.resolve(strict=True)
        receipt = args.receipt.resolve() if args.receipt.exists() else args.receipt.absolute()
        if receipt.exists():
            raise AuditError(f"refusing to overwrite reconstruction receipt: {receipt}")
        try:
            receipt.relative_to(out)
        except ValueError:
            pass
        else:
            raise AuditError("receipt must be outside the immutable consumer output root")
        consumer_source = args.consumer_source.resolve(strict=True)
        actual_consumer_sha = sha256_file(consumer_source)
        if actual_consumer_sha != CONSUMER_SHA256:
            raise AuditError(f"consumer source SHA differs: {actual_consumer_sha}")
        family_verifier = import_family_verifier(args.family_verifier)
        family_result = family_verifier.verify(out, CONSUMER_SHA256)
        local_result = reconstruct_local(out)
        report = {
            "status": "ROOT_FULL_RECONSTRUCTION_PASS",
            "consumer_source_path": str(consumer_source),
            "consumer_source_sha256": actual_consumer_sha,
            "family_verifier_source_path": str(args.family_verifier.resolve(strict=True)),
            "family_verifier_source_sha256": FAMILY_VERIFIER_SHA256,
            "family_reverification": family_result,
            "local_reconstruction": local_result,
            "interpretation_boundary": "Integrity and reconstruction receipt only; no effect estimates or endpoint values are emitted.",
        }
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "contexts": local_result["contexts_reconstructed"],
            "local_endpoint_count": local_result["local_endpoint_count"],
            "local_clip_estimand_comparisons": local_result["local_clip_estimand_comparisons"],
            "family_endpoints": family_result["family"]["endpoint_count"],
            "estimable_endpoints": family_result["family"]["estimable_endpoint_count"],
            "incomplete_endpoints": family_result["family"]["incomplete_endpoint_count"],
            "receipt": str(receipt),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
