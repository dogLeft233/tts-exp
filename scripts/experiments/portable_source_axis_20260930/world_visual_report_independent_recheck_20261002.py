#!/usr/bin/env python3
"""Independently reconstruct sealed WORLD visual summaries from context rows."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("runs/recovered_20260930/remote/world_visual_analysis_attempt09")
REPORT = ROOT / "world_visual_analysis.json"
AUDIT = ROOT / "world_visual_analysis_audit.json"
ALL_FILES = ROOT / "world_visual_analysis_all_files_sha256.json"
SEAL = ROOT / "world_visual_analysis_seal.json"
FREEZE = ROOT / "world_visual_availability_freeze.json"
FREEZE_SEAL = ROOT / "world_visual_availability_freeze.seal.json"
EXPECTED_REPORT_SHA256 = "5d0b8d18ec4565572292d5dbbdf43d25b92e72fbb3a0120eac0731536f31f98a"
EXPECTED_FREEZE_SHA256 = "67131b9bd381dd2e393ab739ae51408e5edf15594cd3e9b67b844c5857621ccf"
EXPECTED_FREEZE_SEAL_SHA256 = "1e7bab71c2d16689f5f316b543883dec3058f27aa053f7689d9495e95b3b37d2"
VISUAL_IDS = ("V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E")
TREATMENTS = ("V_AP", "V_BOTH", "V_EX_EM", "V_EX", "V_E", "V_F0", "V_SP_EM", "V_SP")
GEOMETRY = ("aperture", "mouth_width", "absolute_aperture_velocity_per_s", "yaw_proxy", "pitch_proxy", "mouth_height", "mouth_area")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def close(a: Any, b: Any, *, atol: float = 2e-15, rtol: float = 2e-13) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def reconstructed_speaker_summary(
    clip_values: dict[str, float | None],
    speaker_by_sample: dict[str, str],
    intended_samples: list[str],
    intended_speakers: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in intended_samples:
        value = clip_values.get(sample)
        if value is not None and math.isfinite(float(value)):
            grouped[speaker_by_sample[sample]].append(float(value))
    complete = {
        speaker: float(np.mean(values, dtype=np.float64))
        for speaker, values in grouped.items()
        if len(values) == 2
    }
    names = sorted(complete)
    data = np.asarray([complete[name] for name in names], dtype=np.float64)
    if len(data):
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indices = rng.integers(0, len(data), size=(BOOTSTRAP_DRAWS, len(data)), endpoint=False)
        means = data[indices].mean(axis=1)
        interval = [
            float(np.quantile(means, 0.005, method="linear")),
            float(np.quantile(means, 0.995, method="linear")),
        ]
        index_sha = array_sha(indices.astype(np.int32))
        estimate = float(data.mean())
    else:
        interval = None
        index_sha = None
        estimate = None
    return {
        "equal_speaker_mean": estimate,
        "ci99": interval,
        "complete_two_clip_speakers": len(names),
        "intended_speakers": len(intended_speakers),
        "complete_clips": sum(len(values) for values in grouped.values()),
        "intended_clips": len(intended_samples),
        "missing_speakers": sorted(set(intended_speakers) - set(names)),
        "speaker_values": dict(sorted(complete.items())),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_index_sha256": index_sha,
        "unit": "speaker; mean of two clip-level outcomes before equal-speaker averaging",
    }


def compare_summary(actual: dict[str, Any], expected: dict[str, Any], label: str, errors: list[str], counters: dict[str, int]) -> None:
    counters["summary_objects"] += 1
    for key in (
        "complete_two_clip_speakers", "intended_speakers", "complete_clips", "intended_clips",
        "missing_speakers", "bootstrap_draws", "bootstrap_seed", "bootstrap_index_sha256", "unit",
    ):
        counters["summary_fields"] += 1
        if actual.get(key) != expected.get(key):
            errors.append(f"{label}:{key}")
    for key in ("equal_speaker_mean",):
        counters["summary_fields"] += 1
        if not close(actual.get(key), expected.get(key)):
            errors.append(f"{label}:{key}")
    actual_ci, expected_ci = actual.get("ci99"), expected.get("ci99")
    counters["summary_fields"] += 1
    if actual_ci is None or expected_ci is None:
        if actual_ci is not expected_ci:
            errors.append(f"{label}:ci99")
    elif len(actual_ci) != 2 or any(not close(a, b) for a, b in zip(actual_ci, expected_ci)):
        errors.append(f"{label}:ci99")
    actual_speakers, expected_speakers = actual.get("speaker_values", {}), expected.get("speaker_values", {})
    counters["summary_fields"] += 1
    if set(actual_speakers) != set(expected_speakers) or any(
        not close(actual_speakers[name], expected_speakers[name]) for name in expected_speakers
    ):
        errors.append(f"{label}:speaker_values")


def main() -> None:
    report_sha = file_sha(REPORT)
    require(len(EXPECTED_REPORT_SHA256) == 64, "pin_full_report_sha256_before_run")
    require(report_sha == EXPECTED_REPORT_SHA256, "analysis_report_sha_mismatch")
    require(file_sha(FREEZE) == EXPECTED_FREEZE_SHA256, "availability_freeze_sha_mismatch")
    require(file_sha(FREEZE_SEAL) == EXPECTED_FREEZE_SEAL_SHA256, "availability_freeze_seal_sha_mismatch")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    all_doc = json.loads(ALL_FILES.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    require(report.get("status") == "COMPLETE" and report.get("analysis_type") == "world_physical_two_class_visual_endpoint_and_geometry", "analysis_report_status_or_type_mismatch")
    require(audit.get("status") == "PASS" and audit.get("error_count") == 0 and audit.get("analysis_sha256") == report_sha, "analysis_audit_binding_mismatch")
    require(seal.get("status") == "SEALED" and seal.get("analysis_sha256") == report_sha and seal.get("audit_sha256") == file_sha(AUDIT), "analysis_seal_binding_mismatch")
    files = all_doc.get("files")
    require(isinstance(files, dict) and set(files) == {REPORT.name, AUDIT.name, FREEZE.name, FREEZE_SEAL.name}, "analysis_all_files_coverage_mismatch")
    require(files[REPORT.name] == report_sha and files[AUDIT.name] == file_sha(AUDIT) and files[FREEZE.name] == EXPECTED_FREEZE_SHA256 and files[FREEZE_SEAL.name] == EXPECTED_FREEZE_SEAL_SHA256, "analysis_all_files_hash_mismatch")
    require(seal.get("all_files_sha256") == file_sha(ALL_FILES) and seal.get("file_count") == len(files), "analysis_all_files_seal_mismatch")
    require(report.get("input_seals", {}).get("availability_freeze_sha256") == EXPECTED_FREEZE_SHA256, "report_freeze_input_binding_mismatch")
    require(report.get("intended_denominator") == {"samples": 80, "speakers": 40, "clips_per_speaker": 2, "directions": 2, "contexts": 160, "videos": 1600}, "analysis_intended_denominator_mismatch")
    require(report.get("method", {}).get("bootstrap") == {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "ci_quantiles": [0.005, 0.995], "unit": "equal speaker; two clips averaged first"}, "analysis_bootstrap_protocol_mismatch")

    rows = report.get("context_rows")
    require(isinstance(rows, list) and len(rows) == 160, "context_row_count_mismatch")
    freeze_rows = {(r["sample_id"], r["direction"]): r for r in freeze.get("context_rows", [])}
    require(len(freeze_rows) == 160, "freeze_context_count_mismatch")
    require({(r.get("sample_id"), r.get("direction")) for r in rows} == set(freeze_rows), "analysis_freeze_context_join_mismatch")
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_speaker: dict[str, str] = {}
    for row in rows:
        by_direction[str(row["direction"])].append(row)
        prior = sample_speaker.setdefault(str(row["sample_id"]), str(row["speaker_id"]))
        require(prior == row["speaker_id"], "sample_speaker_mapping_inconsistent")
        fr = freeze_rows[(row["sample_id"], row["direction"])]
        require(row["original_mfa_pair_qualified"] == fr["original_mfa_pair_qualified"], "original_eligibility_mismatch")
        require(row["pre_effect_availability"] == fr["availability_only_endpoint"], "frozen_availability_mismatch")
        require(row["effective_grid"] == fr["effective_grid"], "effective_grid_mismatch")
    samples = sorted(sample_speaker)
    speakers = sorted(set(sample_speaker.values()))
    require(len(samples) == 80 and len(speakers) == 40, "intended_sample_or_speaker_count_mismatch")
    sample_counts: dict[str, int] = defaultdict(int)
    for speaker in sample_speaker.values():
        sample_counts[speaker] += 1
    require(set(sample_counts.values()) == {2}, "not_exactly_two_clips_per_speaker")
    require({direction: len(items) for direction, items in by_direction.items()} == {"tts_to_natural": 80, "natural_to_tts": 80}, "direction_context_coverage_mismatch")

    expected_arm_ids = set(VISUAL_IDS)
    expected_contrasts = {f"{arm}_minus_V_RT" for arm in TREATMENTS} | {"V_RT_minus_V_FIXED"}
    errors: list[str] = []
    counters = {"summary_objects": 0, "summary_fields": 0}
    availability_counts = {}
    for direction in ("tts_to_natural", "natural_to_tts"):
        dir_rows = by_direction[direction]
        direction_summary = report["direction_summaries"][direction]
        require(direction_summary.get("intended_clips") == 80 and direction_summary.get("intended_speakers") == 40, f"{direction}:intended_denominator_mismatch")
        eligible = {
            r["sample_id"]: bool(r["original_mfa_pair_qualified"] and r["pre_effect_availability"]["primary_visual_endpoint_eligible"])
            for r in dir_rows
        }
        original_qualified = sum(bool(r["original_mfa_pair_qualified"]) for r in dir_rows)
        preavailable = sum(bool(r["pre_effect_availability"]["primary_visual_endpoint_eligible"]) for r in dir_rows)
        complete_speakers = sum(
            all(eligible[r["sample_id"]] for r in dir_rows if r["speaker_id"] == speaker)
            for speaker in speakers
        )
        availability_counts[direction] = {
            "original_mfa_pair_qualified_clips": original_qualified,
            "primary_available_clips": preavailable,
            "complete_two_clip_speakers_primary_available": complete_speakers,
        }
        for key, value in availability_counts[direction].items():
            require(direction_summary.get(key) == value, f"{direction}:{key}_mismatch")
        require(set(direction_summary.get("arm_levels", {})) == expected_arm_ids, f"{direction}:arm_coverage_mismatch")
        require(set(direction_summary.get("primary_contrasts", {})) == expected_contrasts, f"{direction}:contrast_coverage_mismatch")
        by_sample = {str(row["sample_id"]): row for row in dir_rows}
        speaker_by_sample = {str(row["sample_id"]): str(row["speaker_id"]) for row in dir_rows}

        for visual_id in VISUAL_IDS:
            observed_arm = direction_summary["arm_levels"][visual_id]
            require(set(observed_arm) == {f"{latency}_{metric}" for latency in ("minus_85ms", "zero_latency") for metric in ("phone_separation", "phone_token_auc")} | {f"geometry_{metric}" for metric in GEOMETRY}, f"{direction}:{visual_id}:metric_coverage_mismatch")
            for latency in ("minus_85ms", "zero_latency"):
                for metric in ("phone_separation", "phone_token_auc"):
                    values = {
                        sample: (by_sample[sample]["per_arm_phone_metrics"][visual_id][latency].get(metric) if eligible[sample] else None)
                        for sample in samples
                    }
                    expected = reconstructed_speaker_summary(values, speaker_by_sample, samples, speakers)
                    compare_summary(observed_arm[f"{latency}_{metric}"], expected, f"{direction}:{visual_id}:{latency}:{metric}", errors, counters)
            for metric in GEOMETRY:
                values = {sample: by_sample[sample]["per_arm_geometry_metrics"][visual_id][metric]["mean"] for sample in samples}
                expected = reconstructed_speaker_summary(values, speaker_by_sample, samples, speakers)
                compare_summary(observed_arm[f"geometry_{metric}"], expected, f"{direction}:{visual_id}:geometry:{metric}", errors, counters)

        for contrast in sorted(expected_contrasts):
            observed_contrast = direction_summary["primary_contrasts"][contrast]
            expected_fields = {f"{latency}_{metric}" for latency in ("minus_85ms", "zero_latency") for metric in ("phone_separation", "phone_token_auc")} | {f"geometry_{metric}" for metric in GEOMETRY}
            require(set(observed_contrast) == expected_fields, f"{direction}:{contrast}:metric_coverage_mismatch")
            for latency in ("minus_85ms", "zero_latency"):
                for metric in ("phone_separation", "phone_token_auc"):
                    values = {
                        sample: (by_sample[sample]["contrasts"][contrast][latency][metric] if eligible[sample] else None)
                        for sample in samples
                    }
                    expected = reconstructed_speaker_summary(values, speaker_by_sample, samples, speakers)
                    compare_summary(observed_contrast[f"{latency}_{metric}"], expected, f"{direction}:{contrast}:{latency}:{metric}", errors, counters)
            for metric in GEOMETRY:
                values = {sample: by_sample[sample]["contrasts"][contrast]["geometry"][metric] for sample in samples}
                expected = reconstructed_speaker_summary(values, speaker_by_sample, samples, speakers)
                compare_summary(observed_contrast[f"geometry_{metric}"], expected, f"{direction}:{contrast}:geometry:{metric}", errors, counters)

    require(not errors, f"summary_reconstruction_mismatches:{len(errors)}:{errors[:20]}")
    report_result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "independent reconstruction from sealed context rows; no visual geometry re-extraction",
        "analysis_path": str(REPORT),
        "analysis_sha256": report_sha,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "analysis_seal_sha256": file_sha(SEAL),
        "directions": availability_counts,
        "intended_denominators": {"samples": len(samples), "speakers": len(speakers), "clips_per_speaker": 2, "contexts": len(rows)},
        "summary_objects_recomputed": counters["summary_objects"],
        "summary_fields_compared": counters["summary_fields"],
        "mismatches": len(errors),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "interval": "linear quantiles 0.005 and 0.995", "unit": "speaker means after averaging two clips"},
        "versions": {"numpy": np.__version__},
    }
    output = ROOT / "world_visual_independent_reconstruction_attempt01.json"
    output.write_text(json.dumps(report_result, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "equal_speaker_mean": summary["equal_speaker_mean"],
            "ci99": summary["ci99"],
            "complete_clips": summary["complete_clips"],
            "complete_two_clip_speakers": summary["complete_two_clip_speakers"],
            "intended_clips": summary["intended_clips"],
            "intended_speakers": summary["intended_speakers"],
            "missing_speakers": summary["missing_speakers"],
            "bootstrap_index_sha256": summary["bootstrap_index_sha256"],
        }

    compact = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_sha256": report_sha,
        "independent_recheck_sha256": file_sha(output),
        "reconstruction": {
            direction: {
                "arm_levels": {
                    arm: {metric: compact_summary(summary) for metric, summary in metrics.items()}
                    for arm, metrics in report["direction_summaries"][direction]["arm_levels"].items()
                },
                "primary_contrasts": {
                    contrast: {metric: compact_summary(summary) for metric, summary in metrics.items()}
                    for contrast, metrics in report["direction_summaries"][direction]["primary_contrasts"].items()
                },
            }
            for direction in ("tts_to_natural", "natural_to_tts")
        },
    }
    compact_path = ROOT / "world_visual_independent_endpoint_table_attempt01.json"
    compact_path.write_text(json.dumps(compact, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report_result, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
