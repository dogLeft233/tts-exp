"""Fit-only selection for the frozen 12-profile shared-frequency family.

Input rows are produced by the independent SyncNet consumer after method-
common support is sealed.  This module never reads audio or chooses support.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.audio_enhancement.shared_frequency_profile_v1 import PROFILES


SCHEMA = "cross_tfg_shared_profile_fit_selection_v3"
CONTROL_METHODS = ("identity", "rms_fixed", "localcompress")
PROFILE_METHODS = tuple(profile.profile_id for profile in PROFILES)
ALL_METHODS = (*CONTROL_METHODS, *PROFILE_METHODS)
LAG_OFFSETS = tuple(range(-15, 16))
FAR_LAG_MIN_ABS_FRAMES = 4
FAR_LAG_WEIGHT = 0.25
SHORTLIST_COUNT = 3


def _context_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model_id"]),
        str(row["speaker_id"]),
        str(row["sample_id"]),
        int(row["seed"]),
    )


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"metric {key} is non-finite")
    return value


def _validate_curve(row: Mapping[str, Any]) -> np.ndarray:
    offsets = tuple(int(x) for x in row["lag_offsets_frames"])
    if offsets != LAG_OFFSETS:
        raise ValueError(f"unexpected lag-offset contract for {row.get('row_id')}")
    curve = np.asarray(row["lag_curve_f32"], dtype=np.float32)
    if curve.shape != (31,) or not np.isfinite(curve).all():
        raise ValueError(f"invalid 31-lag curve for {row.get('row_id')}")
    return curve


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[tuple[str, str, str, int], str], Mapping[str, Any]]:
    indexed: dict[tuple[tuple[str, str, str, int], str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("split") != "fit":
            raise ValueError(f"result row is not explicitly marked split=fit: {row.get('row_id')}")
        key = (_context_key(row), str(row["method_id"]))
        if key in indexed:
            raise ValueError(f"duplicate context/method row: {key}")
        indexed[key] = row
        if row.get("status") != "PASS":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"PASS row has no metrics object: {row.get('row_id')}")
        curve = _validate_curve(row)
        for name in ("B", "Dmin", "D0", "Cbest", "Canchor"):
            _finite_metric(metrics, name)
        curve_median = float(np.float32(np.median(curve)))
        curve_min = float(np.min(curve))
        curve_zero = float(curve[LAG_OFFSETS.index(0)])
        if float(np.float32(metrics["B"])) != curve_median:
            raise ValueError(f"B does not equal the float32 median of the saved common-support curve: {row.get('row_id')}")
        if float(np.float32(metrics["Dmin"])) != curve_min:
            raise ValueError(f"Dmin does not equal the minimum of the saved common-support curve: {row.get('row_id')}")
        if float(np.float32(metrics["D0"])) != curve_zero:
            raise ValueError(f"D0 does not equal the saved zero-lag curve entry: {row.get('row_id')}")
        expected_cbest = float(np.float32(np.float32(metrics["B"]) - np.float32(metrics["Dmin"])))
        expected_canchor = float(np.float32(np.float32(metrics["B"]) - np.float32(metrics["D0"])))
        if float(np.float32(metrics["Cbest"])) != expected_cbest:
            raise ValueError(f"Cbest does not match the canonical float32 B-Dmin contract: {row.get('row_id')}")
        if float(np.float32(metrics["Canchor"])) != expected_canchor:
            raise ValueError(f"Canchor does not match the canonical float32 B-D0 contract: {row.get('row_id')}")
        support_sha = row.get("support_sha256")
        if not isinstance(support_sha, str) or len(support_sha) != 64 or any(c not in "0123456789abcdef" for c in support_sha):
            raise ValueError(f"PASS row is missing a lowercase 64-hex support digest: {row.get('row_id')}")
        first_minimum_index = int(np.argmin(curve))
        expected_best_lag = LAG_OFFSETS[first_minimum_index]
        if int(metrics["best_lag_frames"]) != expected_best_lag:
            raise ValueError(f"best_lag_frames does not use the first minimum in the saved curve: {row.get('row_id')}")
        if int(metrics["official_av_offset_frames"]) != -expected_best_lag:
            raise ValueError(f"offset sign contract failed: {row.get('row_id')}")
        # Keep a derived validation marker out of caller-owned result mappings.
        if curve.size != len(LAG_OFFSETS):
            raise AssertionError("unreachable curve-length mismatch")
    return indexed


def _paired_delta(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    if candidate.get("status") != "PASS" or reference.get("status") != "PASS":
        raise ValueError("paired fit deltas require both method rows to pass")
    if candidate["support_sha256"] != reference["support_sha256"]:
        raise ValueError("paired profile/control rows have different common-support seals")
    cand = candidate["metrics"]
    ref = reference["metrics"]
    curve_c = _validate_curve(candidate).astype(np.float64)
    curve_r = _validate_curve(reference).astype(np.float64)
    far_indices = np.asarray([abs(k) >= FAR_LAG_MIN_ABS_FRAMES for k in LAG_OFFSETS], dtype=bool)
    cand_far = float(curve_c[far_indices].mean(dtype=np.float64) - _finite_metric(cand, "D0"))
    ref_far = float(curve_r[far_indices].mean(dtype=np.float64) - _finite_metric(ref, "D0"))
    return {
        "delta_B": _finite_metric(cand, "B") - _finite_metric(ref, "B"),
        "delta_Dmin": _finite_metric(cand, "Dmin") - _finite_metric(ref, "Dmin"),
        "delta_D0": _finite_metric(cand, "D0") - _finite_metric(ref, "D0"),
        "delta_Cbest": _finite_metric(cand, "Cbest") - _finite_metric(ref, "Cbest"),
        "delta_Canchor": _finite_metric(cand, "Canchor") - _finite_metric(ref, "Canchor"),
        "delta_far_lag_contrast": cand_far - ref_far,
        "delta_best_lag_frames": float(int(cand["best_lag_frames"]) - int(ref["best_lag_frames"])),
        "delta_official_av_offset_frames": float(
            int(cand["official_av_offset_frames"]) - int(ref["official_av_offset_frames"])
        ),
    }


def _equal_speaker_model_summary(
    deltas: Sequence[tuple[str, Mapping[str, float]]],
) -> dict[str, Any]:
    by_speaker: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for speaker_id, values in deltas:
        by_speaker[speaker_id].append(values)
    if not by_speaker:
        return {"status": "INCOMPLETE", "speaker_count": 0, "context_count": 0}
    fields = tuple(next(iter(by_speaker.values()))[0].keys())
    per_speaker: dict[str, dict[str, float]] = {}
    for speaker_id, items in sorted(by_speaker.items()):
        per_speaker[speaker_id] = {
            field: float(np.mean([float(row[field]) for row in items], dtype=np.float64))
            for field in fields
        }
    means = {
        field: float(np.mean([row[field] for row in per_speaker.values()], dtype=np.float64))
        for field in fields
    }
    return {
        "status": "COMPLETE",
        "speaker_count": len(per_speaker),
        "context_count": sum(len(items) for items in by_speaker.values()),
        "equal_speaker_means": means,
        "per_speaker": per_speaker,
    }


def select_shared_profiles(
    *,
    expected_contexts: Sequence[Mapping[str, Any]],
    predeclared_fit_roster: Sequence[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    model_roster: Sequence[str],
    shortlist_count: int = SHORTLIST_COUNT,
) -> dict[str, Any]:
    """Summarize the fit-only grid with equal-speaker/equal-model weighting.

    ``predeclared_fit_roster`` is the frozen eight-row speaker/sample/seed
    roster. ``expected_contexts`` must be its exact cross-product with the
    model roster, and every context/result must explicitly say ``split=fit``.
    Failed/missing rows remain explicit; they are never dropped to create a
    complete comparison.
    """

    if len(model_roster) < 3 or len(set(model_roster)) != len(model_roster):
        raise ValueError("model_roster must contain at least three unique qualified frontends")
    if shortlist_count < 1 or shortlist_count > SHORTLIST_COUNT:
        raise ValueError(f"shortlist_count must be between 1 and {SHORTLIST_COUNT}")
    if len(predeclared_fit_roster) != 8:
        raise ValueError("the fit-only selector requires the predeclared eight-utterance roster")
    roster_keys: set[tuple[str, str, int]] = set()
    speaker_counts: dict[str, int] = defaultdict(int)
    for roster_row in predeclared_fit_roster:
        if roster_row.get("split") != "fit":
            raise ValueError("predeclared fit roster contains a non-fit row")
        speaker_id = str(roster_row["speaker_id"])
        sample_id = str(roster_row["sample_id"])
        seed = int(roster_row["seed"])
        key = (speaker_id, sample_id, seed)
        if key in roster_keys:
            raise ValueError(f"duplicate predeclared fit roster row: {key}")
        roster_keys.add(key)
        speaker_counts[speaker_id] += 1
    if len(speaker_counts) != 4 or any(count != 2 for count in speaker_counts.values()):
        raise ValueError("the predeclared roster must contain two utterances for each of four fit speakers")
    if len({seed for _, _, seed in roster_keys}) != 1:
        raise ValueError("the predeclared fit roster must use one shared paired render seed")
    sample_ids = [sample_id for _, sample_id, _ in roster_keys]
    if len(set(sample_ids)) != 8:
        raise ValueError("each predeclared fit utterance must occur exactly once")

    contexts: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for context in expected_contexts:
        if context.get("split") != "fit":
            raise ValueError("expected context is not explicitly marked split=fit")
        key = _context_key(context)
        if key in contexts:
            raise ValueError(f"duplicate expected context: {key}")
        if key[0] not in model_roster:
            raise ValueError(f"context model not in frozen roster: {key[0]}")
        contexts[key] = context
    if not contexts:
        raise ValueError("the frozen plan has no expected contexts")
    contexts_by_model = {
        model_id: {
            (context_key[1], context_key[2], context_key[3])
            for context_key in contexts
            if context_key[0] == model_id
        }
        for model_id in model_roster
    }
    for model_id, observed_roster in contexts_by_model.items():
        if observed_roster != roster_keys:
            raise ValueError(f"model {model_id} contexts do not exactly match the predeclared fit roster")
    first_model = model_roster[0]
    if any(contexts_by_model[model_id] != contexts_by_model[first_model] for model_id in model_roster[1:]):
        raise ValueError("all frontends must use the same speaker/sample/seed fit contexts")
    indexed = _validated_rows(result_rows)
    expected_keys = {
        (context_key, method_id)
        for context_key in contexts
        for method_id in ALL_METHODS
    }
    unexpected = sorted(set(indexed) - expected_keys)
    missing = sorted(expected_keys - set(indexed))
    context_integrity_problems: list[dict[str, Any]] = []
    for context_key in sorted(contexts):
        for method_id in ALL_METHODS:
            row = indexed.get((context_key, method_id))
            if row is None:
                context_integrity_problems.append({
                    "context": context_key,
                    "method_id": method_id,
                    "reason": "MISSING_PLANNED_ROW",
                })
            elif row.get("status") != "PASS":
                context_integrity_problems.append({
                    "context": context_key,
                    "method_id": method_id,
                    "status": row.get("status"),
                    "reason": row.get("failure_reason") or "METHOD_ROW_NOT_PASS",
                })

    support_errors: list[dict[str, Any]] = []
    for context_key in contexts:
        observed = [indexed.get((context_key, method)) for method in ALL_METHODS]
        passing = [row for row in observed if row is not None and row.get("status") == "PASS"]
        support_hashes = {str(row.get("support_sha256")) for row in passing}
        if len(support_hashes) > 1:
            support_errors.append({"context": context_key, "support_sha256_values": sorted(support_hashes)})

    candidate_rows: list[dict[str, Any]] = []
    for profile_id in PROFILE_METHODS:
        problems: list[dict[str, Any]] = []
        per_model_deltas: dict[str, list[tuple[str, Mapping[str, float]]]] = {
            model_id: [] for model_id in model_roster
        }
        for context_key in sorted(contexts):
            candidate = indexed.get((context_key, profile_id))
            reference = indexed.get((context_key, "rms_fixed"))
            if candidate is None or reference is None:
                problems.append({"context": context_key, "reason": "MISSING_PLANNED_ROW"})
                continue
            if candidate.get("status") != "PASS" or reference.get("status") != "PASS":
                problems.append({
                    "context": context_key,
                    "reason": "FAILED_OR_INCOMPLETE_ROW",
                    "candidate_status": candidate.get("status"),
                    "reference_status": reference.get("status"),
                })
                continue
            try:
                delta = _paired_delta(candidate, reference)
            except (KeyError, TypeError, ValueError) as exc:
                problems.append({"context": context_key, "reason": "PAIR_VALIDATION_FAILED", "detail": str(exc)})
                continue
            per_model_deltas[context_key[0]].append((context_key[1], delta))

        model_summaries = {
            model_id: _equal_speaker_model_summary(per_model_deltas[model_id])
            for model_id in model_roster
        }
        complete = not problems and not missing and not unexpected and not context_integrity_problems and not support_errors and all(
            summary.get("status") == "COMPLETE" for summary in model_summaries.values()
        )
        if complete:
            means = [summary["equal_speaker_means"] for summary in model_summaries.values()]
            model_utilities: dict[str, float] = {}
            for model_id, summary in model_summaries.items():
                values = summary["equal_speaker_means"]
                # Cbest is the primary fit objective. The per-model terms make
                # regressions in fixed-time D0 or anchored confidence costly,
                # while far-lag separation is a small positive tie-breaker.
                utility = (
                    values["delta_Cbest"]
                    + min(0.0, values["delta_Cbest"])
                    - max(0.0, values["delta_D0"])
                    - max(0.0, -values["delta_Canchor"])
                    + FAR_LAG_WEIGHT * values["delta_far_lag_contrast"]
                )
                model_utilities[model_id] = float(utility)
            objective = float(np.mean(list(model_utilities.values()), dtype=np.float64))
            objective += min(0.0, min(model_utilities.values()))
            equal_model_delta = {
                field: float(np.mean([x[field] for x in means], dtype=np.float64))
                for field in means[0]
            }
            cross_model_guards = {
                "all_models_nonincreasing_D0": all(x["delta_D0"] <= 0.0 for x in means),
                "all_models_nonnegative_delta_Canchor": all(x["delta_Canchor"] >= 0.0 for x in means),
                "all_models_nonnegative_delta_Cbest": all(x["delta_Cbest"] >= 0.0 for x in means),
            }
        else:
            objective = None
            equal_model_delta = None
            model_utilities = {}
            cross_model_guards = {"status": "NOT_EVALUABLE"}
        candidate_rows.append({
            "profile_id": profile_id,
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "expected_context_count": len(contexts),
            "paired_context_count": sum(len(v) for v in per_model_deltas.values()),
            "problems": problems,
            "equal_model_deltas_vs_rms_fixed": equal_model_delta,
            "per_model": model_summaries,
            "per_model_selection_utility": model_utilities,
            "selection_objective": objective,
            "cross_model_nonregression_guards": cross_model_guards,
        })

    complete_candidates = [row for row in candidate_rows if row["status"] == "COMPLETE"]
    complete_candidates.sort(key=lambda row: (-float(row["selection_objective"]), str(row["profile_id"])))
    shortlisted_ids = [row["profile_id"] for row in complete_candidates[:shortlist_count]]
    return {
        "schema": SCHEMA,
        "split_role": "fit_only_historical_exposed",
        "qualified_model_roster": list(model_roster),
        "predeclared_fit_roster": [
            {"split": "fit", "speaker_id": speaker_id, "sample_id": sample_id, "seed": seed}
            for speaker_id, sample_id, seed in sorted(roster_keys)
        ],
        "selection_policy": {
            "candidate_method_ids": list(PROFILE_METHODS),
            "controls": list(CONTROL_METHODS),
            "primary_reference": "rms_fixed",
            "weights": "equal speaker within model; equal model across roster",
            "primary_objective": "equal-model delta Cbest vs fixed RMS, with per-model penalties for negative Cbest, worse D0, or worse Canchor",
            "auxiliary_weighted_term": f"{FAR_LAG_WEIGHT}*delta mean(D_lag-D0) for abs(lag)>={FAR_LAG_MIN_ABS_FRAMES}; this is an auxiliary objective term, not a tie-breaker",
            "offset_policy": "report best_lag_frames and official_av_offset_frames; never retime or optimize offset",
            "support_policy": "use sealed complete method-common support; missing/failed rows remain in the planned denominator",
            "shortlist_count": int(shortlist_count),
            "inference": "descriptive fit selection only; no confidence intervals or confirmatory claim",
        },
        "expected_context_count": len(contexts),
        "expected_method_row_count": len(expected_keys),
        "observed_method_row_count": len(indexed),
        "missing_rows": [
            {"context": context_key, "method_id": method_id}
            for context_key, method_id in missing
        ],
        "context_integrity_problems": context_integrity_problems,
        "unexpected_rows": [
            {"context": context_key, "method_id": method_id}
            for context_key, method_id in unexpected
        ],
        "common_support_errors": support_errors,
        "candidate_results": candidate_rows,
        "shortlist_profile_ids": shortlisted_ids,
        "shortlist_status": "READY_FOR_FIT_ONLY_REVIEW" if len(shortlisted_ids) == shortlist_count else "INSUFFICIENT_COMPLETE_CANDIDATES",
    }
