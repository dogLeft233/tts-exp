#!/usr/bin/env python3
"""Reproduce the predeclared paired controls360 RMS diagnostic (stdlib only)."""

from __future__ import annotations

import array
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "scripts").is_dir())
PROTOCOL_PATH = HERE / "analysis_protocol_attempt01.json"
PROTOCOL_SHA256 = "b27b6d1fd2ed7e646b4697da209d9f9ebc50838388457e41d5e3b047b459709a"
OUTPUT_PATH = HERE / "effects_table_attempt01.json"
MANIFEST_PATH = HERE / "analysis_manifest_attempt01.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), sha256_bytes(data)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_value(metrics: dict[str, Any], metric_spec: dict[str, Any]) -> float:
    name = metric_spec["name"]
    if name == "abs_best_lag_frames":
        return float(abs(metrics["best_lag_frames"]))
    return float(metrics[metric_spec["source_field"]])


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of an empty set")
    return math.fsum(values) / len(values)


def as_counter(counter: collections.Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def main() -> None:
    protocol_bytes, protocol_sha = sha256_file(PROTOCOL_PATH)
    if protocol_sha != PROTOCOL_SHA256:
        raise SystemExit(f"protocol hash mismatch: {protocol_sha}")
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "PRECOMPUTE_LOCKED":
        raise SystemExit("preanalysis protocol is not locked")

    pinned_inputs: dict[str, dict[str, Any]] = {}
    for label, pin in protocol["source_inputs"].items():
        path = REPO_ROOT / pin["path"]
        size, digest = sha256_file(path)
        if size != pin["bytes"] or digest != pin["sha256"]:
            raise SystemExit(f"pinned input mismatch for {label}: bytes={size} sha256={digest}")
        pinned_inputs[label] = {"path": pin["path"], "bytes": size, "sha256": digest}

    ledger_pin = protocol["source_inputs"]["ledger"]
    ledger_path = REPO_ROOT / ledger_pin["path"]
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != protocol["population_and_pairing"]["ledger_rows"]:
        raise SystemExit(f"unexpected ledger row count: {len(rows)}")

    pairing_fields = protocol["population_and_pairing"]["paired_identity_fields"]
    methods = protocol["population_and_pairing"]["methods"]
    models = protocol["population_and_pairing"]["models"]
    speakers = protocol["population_and_pairing"]["speakers"]
    metric_specs = protocol["estimands"]["metric_family"]
    metric_names = [spec["name"] for spec in metric_specs]
    target_names = protocol["estimands"]["target_family"]
    if target_names != models + ["equal_model_mean"]:
        raise SystemExit("target family does not match the predeclared model order")

    row_statuses = collections.Counter(row.get("status") for row in rows)
    method_counts = collections.Counter(row.get("method_id") for row in rows)
    model_method_counts = collections.Counter((row.get("model_id"), row.get("method_id")) for row in rows)
    if row_statuses != {"PASS": len(rows)}:
        raise SystemExit(f"non-PASS terminal rows: {dict(row_statuses)}")
    if method_counts != {"identity": 180, "rms_fixed": 180}:
        raise SystemExit(f"unexpected method denominator: {dict(method_counts)}")
    for model in models:
        for method in methods:
            if model_method_counts[(model, method)] != 60:
                raise SystemExit(f"unexpected model/method cell count for {model}/{method}")

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in pairing_fields)
        grouped.setdefault(key, {})[row["method_id"]] = row
    expected_pairs = len(rows) // len(methods)
    if len(grouped) != expected_pairs or any(set(pair) != set(methods) for pair in grouped.values()):
        raise SystemExit("paired identity join is incomplete or duplicated")
    if any(len(pair) != len(methods) for pair in grouped.values()):
        raise SystemExit("paired identity key has duplicate method rows")

    observed_speakers = sorted({row["speaker_id"] for row in rows})
    if observed_speakers != speakers:
        raise SystemExit(f"speaker set differs from the frozen ten: {observed_speakers}")
    observed_models = sorted({row["model_id"] for row in rows})
    if observed_models != sorted(models):
        raise SystemExit(f"model set differs from the predeclared set: {observed_models}")

    pairs: list[dict[str, Any]] = []
    support_inventory: list[dict[str, Any]] = []
    support_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for key, pair in grouped.items():
        identity = pair["identity"]
        rms = pair["rms_fixed"]
        if identity["score"].get("status") != "PASS" or rms["score"].get("status") != "PASS":
            raise SystemExit(f"scorer failed inside pair {key}")
        identity_score = identity["score"]
        rms_score = rms["score"]
        identity_full = identity_score["support_by_support"]["full"]
        rms_full = rms_score["support_by_support"]["full"]
        identity_guard = identity_score["support_by_support"]["guard20"]
        rms_guard = rms_score["support_by_support"]["guard20"]
        if not identity_full["eligible"] or not rms_full["eligible"]:
            raise SystemExit(f"full-support row ineligible inside pair {key}")
        if identity_full != rms_full or identity_guard != rms_guard:
            raise SystemExit(f"paired support metadata differs inside pair {key}")
        identity_render_frames = identity["render"]["frame_count"]
        rms_render_frames = rms["render"]["frame_count"]
        identity_eval_frames = identity_score["official_evaluator"]["receipt"]["video_frame_count"]
        rms_eval_frames = rms_score["official_evaluator"]["receipt"]["video_frame_count"]
        if identity_render_frames != rms_render_frames or identity_eval_frames != rms_eval_frames:
            raise SystemExit(f"paired video frame counts differ inside pair {key}")

        identity_metrics = identity_score["metrics_by_support"]["full"]
        rms_metrics = rms_score["metrics_by_support"]["full"]
        for metric in (identity_metrics, rms_metrics):
            for spec in metric_specs:
                value = metric_value(metric, spec)
                if not math.isfinite(value):
                    raise SystemExit(f"non-finite metric {spec['name']} in pair {key}")
            if metric["anchor_column"] != metric["zero_lag_column"]:
                raise SystemExit(f"anchor and zero-lag columns differ in pair {key}")
            if metric["anchor_column"] != 15 or metric["zero_lag_column"] != 15:
                raise SystemExit(f"unexpected anchor/zero-lag column in pair {key}")

        identity_values = {spec["name"]: metric_value(identity_metrics, spec) for spec in metric_specs}
        rms_values = {spec["name"]: metric_value(rms_metrics, spec) for spec in metric_specs}
        differences = {name: rms_values[name] - identity_values[name] for name in metric_names}
        pairs.append({
            "speaker_id": identity["speaker_id"],
            "model_id": identity["model_id"],
            "identity_values": identity_values,
            "rms_fixed_values": rms_values,
            "differences": differences,
        })
        item = {
            "split": identity["split"],
            "sample_id": identity["sample_id"],
            "speaker_id": identity["speaker_id"],
            "model_id": identity["model_id"],
            "face_id": identity["face_id"],
            "seed": identity["seed"],
            "full_support": {
                "eligible": identity_full["eligible"],
                "query_count": identity_full["query_count"],
                "support_mask_sha256": identity_full["support_mask_sha256"],
            },
            "guard20_support": {
                "eligible": identity_guard["eligible"],
                "query_count": identity_guard["query_count"],
                "support_mask_sha256": identity_guard["support_mask_sha256"],
            },
            "render_frame_count": identity_render_frames,
            "evaluator_video_frame_count": identity_eval_frames,
        }
        support_inventory.append(item)
        support_by_model[identity["model_id"]].append(item)

    cluster_pair_counts: dict[str, dict[str, int]] = {speaker: {model: 0 for model in models} for speaker in speakers}
    for pair in pairs:
        cluster_pair_counts[pair["speaker_id"]][pair["model_id"]] += 1
    if any(count != 6 for by_model in cluster_pair_counts.values() for count in by_model.values()):
        raise SystemExit(f"unbalanced speaker/model pair support: {cluster_pair_counts}")

    # Speaker cluster means; each frozen speaker has exactly six paired rows per model.
    cluster_values: dict[str, dict[str, dict[str, list[float]]]] = {
        target: {speaker: {name: [] for name in metric_names} for speaker in speakers}
        for target in target_names
    }
    method_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        model: {method: {speaker: {name: [] for name in metric_names} for speaker in speakers} for method in methods}
        for model in models
    }
    for pair in pairs:
        speaker, model = pair["speaker_id"], pair["model_id"]
        for name in metric_names:
            cluster_values[model][speaker][name].append(pair["differences"][name])
            method_values[model]["identity"][speaker][name].append(pair["identity_values"][name])
            method_values[model]["rms_fixed"][speaker][name].append(pair["rms_fixed_values"][name])

    cluster_means: dict[str, dict[str, dict[str, float]]] = {
        target: {speaker: {} for speaker in speakers} for target in target_names
    }
    for model in models:
        for speaker in speakers:
            for name in metric_names:
                cluster_means[model][speaker][name] = mean(cluster_values[model][speaker][name])
    for speaker in speakers:
        for name in metric_names:
            cluster_means["equal_model_mean"][speaker][name] = mean(
                [cluster_means[model][speaker][name] for model in models]
            )

    observed: dict[str, dict[str, float]] = {target: {} for target in target_names}
    for target in target_names:
        for name in metric_names:
            observed[target][name] = mean([cluster_means[target][speaker][name] for speaker in speakers])

    # The declared target family has 4 x 7 = 28 paired-mean effects.
    expected_family_size = protocol["estimands"]["simultaneous_family_size"]
    family_cells = [(target, name) for target in target_names for name in metric_names]
    if len(family_cells) != expected_family_size:
        raise SystemExit(f"family size mismatch: {len(family_cells)}")

    uncertainty = protocol["uncertainty"]
    resamples = int(uncertainty["resamples"])
    rng = random.Random(int(uncertainty["seed"]))
    bootstrap: dict[str, dict[str, array.array[float]]] = {
        target: {name: array.array("d") for name in metric_names} for target in target_names
    }
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    for _ in range(resamples):
        counts = [0] * len(speakers)
        for _draw in speakers:
            counts[rng.randrange(len(speakers))] += 1
        for target, name in family_cells:
            boot_value = math.fsum(
                cluster_means[target][speaker][name] * counts[speaker_index[speaker]]
                for speaker in speakers
            ) / len(speakers)
            bootstrap[target][name].append(boot_value)

    bootstrap_sd: dict[str, dict[str, float]] = {target: {} for target in target_names}
    centered_columns: dict[tuple[str, str], array.array[float]] = {}
    for target, name in family_cells:
        values = bootstrap[target][name]
        if len(values) != resamples:
            raise SystemExit("bootstrap output length mismatch")
        boot_mean = math.fsum(values) / resamples
        sd = math.sqrt(math.fsum((value - boot_mean) ** 2 for value in values) / (resamples - 1))
        bootstrap_sd[target][name] = sd
        centered = array.array("d")
        estimate = observed[target][name]
        for value in values:
            if sd == 0.0:
                if value != estimate:
                    raise SystemExit(f"undefined studentization for {target}/{name}")
                centered.append(0.0)
            else:
                centered.append(abs((value - estimate) / sd))
        centered_columns[(target, name)] = centered

    max_t = array.array("d")
    for replicate in range(resamples):
        max_t.append(max(centered_columns[cell][replicate] for cell in family_cells))
    max_t = array.array("d", sorted(max_t))
    rank = math.ceil(0.99 * resamples) - 1
    critical_value = max_t[rank]

    table_rows: list[dict[str, Any]] = []
    for target in target_names:
        for name in metric_names:
            if target == "equal_model_mean":
                identity_mean = mean([
                    mean([value for speaker in speakers for value in method_values[model]["identity"][speaker][name]])
                    for model in models
                ])
                rms_mean = mean([
                    mean([value for speaker in speakers for value in method_values[model]["rms_fixed"][speaker][name]])
                    for model in models
                ])
                pair_count = len(pairs)
            else:
                identity_mean = mean([
                    mean(method_values[target]["identity"][speaker][name]) for speaker in speakers
                ])
                rms_mean = mean([
                    mean(method_values[target]["rms_fixed"][speaker][name]) for speaker in speakers
                ])
                pair_count = sum(pair["model_id"] == target for pair in pairs)
            delta_mean = observed[target][name]
            if abs((rms_mean - identity_mean) - delta_mean) > 1e-10:
                raise SystemExit(f"paired contrast does not equal difference of method means for {target}/{name}")
            sd = bootstrap_sd[target][name]
            table_rows.append({
                "target": target,
                "metric": name,
                "support": "full",
                "identity_mean": identity_mean,
                "rms_fixed_mean": rms_mean,
                "paired_delta_mean_rms_fixed_minus_identity": delta_mean,
                "cluster_bootstrap_se": sd,
                "simultaneous_99_ci_low": delta_mean - critical_value * sd,
                "simultaneous_99_ci_high": delta_mean + critical_value * sd,
                "pair_count": pair_count,
                "speaker_cluster_count": len(speakers),
            })

    def support_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        guard_eligibility = collections.Counter(item["guard20_support"]["eligible"] for item in items)
        guard_query_histogram = collections.Counter(item["guard20_support"]["query_count"] for item in items)
        full_query_histogram = collections.Counter(item["full_support"]["query_count"] for item in items)
        render_histogram = collections.Counter(item["render_frame_count"] for item in items)
        eval_histogram = collections.Counter(item["evaluator_video_frame_count"] for item in items)
        return {
            "paired_clip_model_identity_count": len(items),
            "full_eligible_count": sum(bool(item["full_support"]["eligible"]) for item in items),
            "full_query_count_histogram": as_counter(full_query_histogram),
            "guard20_eligible_count": guard_eligibility[True],
            "guard20_ineligible_count": guard_eligibility[False],
            "guard20_query_count_histogram": as_counter(guard_query_histogram),
            "render_frame_count_histogram": as_counter(render_histogram),
            "evaluator_video_frame_count_histogram": as_counter(eval_histogram),
        }

    qc = read_json(REPO_ROOT / protocol["source_inputs"]["terminal_content_qc"]["path"])
    terminal_qc = qc.get("terminal_qc", {})
    result = {
        "schema": "controls360_rms_diagnostic_effects_v1",
        "analysis_id": protocol["analysis_id"],
        "status": "PASS_DESCRIPTIVE_ANALYSIS",
        "runtime": {"python_version": sys.version.split()[0], "implementation": sys.implementation.name},
        "analysis_protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "bytes": protocol_bytes,
            "sha256": protocol_sha,
        },
        "analysis_code": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "bytes": sha256_file(Path(__file__))[0],
            "sha256": sha256_file(Path(__file__))[1],
        },
        "source_inputs": pinned_inputs,
        "terminal_readback_context": {
            "status": qc.get("status"),
            "run_status": terminal_qc.get("run_status"),
            "manifest_status": terminal_qc.get("manifest_status"),
            "child_returncode": terminal_qc.get("child_returncode"),
            "status_counts": terminal_qc.get("status_counts"),
            "planned_and_result_rows": terminal_qc.get("planned_and_result_rows"),
            "strict_attestation_status": terminal_qc.get("strict_attestation_status"),
            "repeat_policy": terminal_qc.get("repeat_policy"),
            "empty_repeat_ledger_bytes": terminal_qc.get("empty_repeat_ledger_bytes"),
            "native_receipt_download_scope": terminal_qc.get("interpretation"),
        },
        "denominator": {
            "ledger_rows": len(rows),
            "paired_method_rows": len(pairs),
            "pairs_by_model": {model: sum(pair["model_id"] == model for pair in pairs) for model in models},
            "methods": dict(method_counts),
            "model_method_cells": {
                f"{model}/{method}": model_method_counts[(model, method)]
                for model in models for method in methods
            },
            "speaker_ids": speakers,
            "pairs_per_speaker_model": 6,
            "cluster_pair_counts": cluster_pair_counts,
        },
        "bootstrap": {
            "method": uncertainty["method"],
            "cluster_unit": uncertainty["cluster_unit"],
            "resamples": resamples,
            "seed": uncertainty["seed"],
            "rng": uncertainty["rng"],
            "family_size": len(family_cells),
            "simultaneous_max_abs_t_99_critical_value": critical_value,
            "quantile_rule": "nearest_rank_99th_percentile",
            "interval_rule": uncertainty["interval"],
        },
        "effects_table": table_rows,
        "support_availability_summary": {
            "all_pairs": support_summary(support_inventory),
            "by_model": {model: support_summary(support_by_model[model]) for model in models},
            "guard20_interpretation": "Actual emitted support_by_support.guard20 fields; eligibility was not inferred from the support label. Query counts are scored query positions, not video frame counts.",
        },
        "support_availability_by_paired_clip_model_identity": support_inventory,
        "limitations": [
            "The full-support comparisons are diagnostic summaries of this completed controls ledger, not candidate qualification or method selection.",
            "The strict attestation QC validates self-hashed per-row references; the six-file transfer QC did not separately download all 360 referenced native receipt files.",
            "The simultaneous intervals are approximate cluster-bootstrap intervals over ten frozen speaker clusters.",
        ],
    }
    result["table_self_sha256"] = sha256_bytes(canonical_bytes({key: value for key, value in result.items() if key != "table_self_sha256"}))
    OUTPUT_PATH.write_bytes(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    protocol_size, protocol_hash = sha256_file(PROTOCOL_PATH)
    code_size, code_hash = sha256_file(Path(__file__))
    table_size, table_hash = sha256_file(OUTPUT_PATH)
    manifest = {
        "schema": "controls360_rms_diagnostic_manifest_v1",
        "analysis_id": protocol["analysis_id"],
        "status": "PASS_DESCRIPTIVE_ANALYSIS",
        "self_hash_rule": "manifest_self_sha256 is SHA-256 over canonical JSON with this field omitted.",
        "source_inputs": pinned_inputs,
        "artifacts": [
            {"role": "preanalysis_protocol", "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)), "bytes": protocol_size, "sha256": protocol_hash},
            {"role": "analysis_code", "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "bytes": code_size, "sha256": code_hash},
            {"role": "effects_table", "path": str(OUTPUT_PATH.relative_to(REPO_ROOT)), "bytes": table_size, "sha256": table_hash, "table_self_sha256": result["table_self_sha256"]},
        ],
        "bootstrap": result["bootstrap"],
        "denominator": result["denominator"],
    }
    manifest["manifest_self_sha256"] = sha256_bytes(canonical_bytes(manifest))
    MANIFEST_PATH.write_bytes(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    manifest_size, manifest_raw_hash = sha256_file(MANIFEST_PATH)
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT_PATH),
        "table_self_sha256": result["table_self_sha256"],
        "analysis_code_sha256": result["analysis_code"]["sha256"],
        "manifest": str(MANIFEST_PATH),
        "manifest_bytes": manifest_size,
        "manifest_file_sha256": manifest_raw_hash,
        "manifest_self_sha256": manifest["manifest_self_sha256"],
        "bootstrap_critical_value": critical_value,
        "rows": len(rows),
        "paired_rows": len(pairs),
        "guard20_ineligible_pairs": result["support_availability_summary"]["all_pairs"]["guard20_ineligible_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
