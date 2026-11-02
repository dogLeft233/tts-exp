#!/usr/bin/env python3
"""Freeze candidate-vs-control development contrasts from a sealed StageB plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seal(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha(canonical(body))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-plan-raw-sha256", required=True)
    parser.add_argument("--candidate-method-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.source_plan.read_bytes()
    source_raw_sha = sha(raw)
    if source_raw_sha != args.source_plan_raw_sha256.lower():
        raise ValueError("source plan raw SHA256 differs from the requested frozen source")
    source = json.loads(raw)
    if source.get("schema_version") != "cross-tfg-audio-eval-v1" or source.get("status") != "FROZEN":
        raise ValueError("source plan must be frozen cross-tfg-audio-eval-v1")
    if source.get("plan_sha256") != seal(source, "plan_sha256"):
        raise ValueError("source plan embedded payload seal is invalid")
    if source.get("run_id") != "cross_tfg_audio_head_20261003_stageb_dev_attempt03":
        raise ValueError("source plan run id differs from finalized four-method StageB run")
    if len(source.get("rows", [])) != 240:
        raise ValueError("source plan does not contain the frozen 240 planned contexts")
    method_ids = [row.get("method_id") for row in source.get("methods", [])]
    required = ["identity", "rms_fixed", "rms_localcompress", "spectral_head_fit24"]
    if method_ids != required:
        raise ValueError(f"source method roster mismatch: {method_ids}")
    candidate_id = args.candidate_method_id
    if not candidate_id or candidate_id in required:
        raise ValueError("candidate method id must be a new nonempty id distinct from the four source methods")
    analysis = source.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("source plan is missing analysis policy")
    for key in ("bootstrap", "aggregation"):
        if not isinstance(analysis.get(key), dict):
            raise ValueError(f"source analysis is missing frozen {key}")
    if not isinstance(source.get("supports"), list) or not source["supports"]:
        raise ValueError("source plan is missing frozen FULL/GUARD20 support definitions")

    candidate_contrasts = [
        {"contrast_id": f"{candidate_id}_minus_identity", "method_a": candidate_id, "method_b": "identity", "role": "main"},
        {"contrast_id": f"{candidate_id}_minus_rms_fixed", "method_a": candidate_id, "method_b": "rms_fixed", "role": "main"},
        {"contrast_id": f"{candidate_id}_minus_spectral_head_fit24", "method_a": candidate_id,
         "method_b": "spectral_head_fit24", "role": "diagnostic"},
    ]
    source_contrast_ids = {item.get("contrast_id") for item in analysis.get("contrasts", []) if isinstance(item, dict)}
    if source_contrast_ids.intersection(item["contrast_id"] for item in candidate_contrasts):
        raise ValueError("candidate contrast id collides with a source contrast")
    model_ids = [item.get("model_id") for item in source.get("models", []) if isinstance(item, dict)]
    if len(model_ids) != 1 or not isinstance(model_ids[0], str):
        raise ValueError("StageB source must pin exactly one model for this v2 comparison")
    endpoints = []
    for contrast in candidate_contrasts:
        for support_id in ("full", "guard20"):
            if not any(x.get("support_id") == support_id for x in source["supports"]):
                raise ValueError(f"frozen source plan does not define required support {support_id}")
            endpoints.append({
                "endpoint_id": f"dev_{contrast['contrast_id']}_{support_id}_Cbest",
                "split": "dev", "contrast_id": contrast["contrast_id"],
                "support_id": support_id, "metric": "Cbest", "model_scope": "model",
                "model_id": model_ids[0], "roi_estimand": "native_official",
                "audio_estimand": "native_roi_original_pcm16", "exploratory": True,
            })

    policy = {
        "schema_version": "cross_tfg_joint_analysis_policy_v1",
        "status": "FROZEN",
        "source_run_id": source["run_id"],
        "source_plan_raw_sha256": source_raw_sha,
        "source_plan_payload_sha256": source["plan_sha256"],
        "source_analysis_contrasts_sha256": sha(canonical(analysis.get("contrasts", []))),
        "source_summary_endpoints_sha256": sha(canonical(analysis.get("summary_endpoints", []))),
        "candidate_method_id": candidate_id,
        "candidate_method_roster_role": "new_historical_fit_only_candidate",
        "contrasts": candidate_contrasts,
        "contrast_family": {
            "main": [candidate_contrasts[0]["contrast_id"], candidate_contrasts[1]["contrast_id"]],
            "diagnostic": [candidate_contrasts[2]["contrast_id"]],
            "status": "EXPLORATORY_DEV_ONLY_NOT_CONFIRMATORY",
        },
        "summary_endpoints": endpoints,
        "bootstrap": analysis["bootstrap"],
        "aggregation": analysis["aggregation"],
        "supports": source["supports"],
        "joint_common_support": {
            "methods": required + [candidate_id],
            "scope": "within_sample_model_face_seed",
            "full": "intersection_of_valid_query_keys_across_all_five_planned_methods",
            "guard20": "remove_20_queries_from_both_ends_of_each_five_way_full_intersection",
            "empty_support": "retain_planned_rows_and_contexts_as_unestimable_without_denominator_shrinkage",
        },
        "cross_plan_lineage": {
            "source_four_method_estimand_remains_separate": True,
            "joint_package_provenance": "DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE",
            "candidate_only_run_requires_all_60_planned_context_rows": True,
            "reuse_only_hash_bound_final_PASS_source_rows": True,
            "rebuild_all_support_summaries_over_five_methods": True,
        },
        "confirm_audio_read": False,
        "confirm_effect_read": False,
        "status_note": "Development selection remains exploratory; freeze confirmation roles/family before any confirmation effects are opened.",
    }
    policy["policy_sha256"] = seal(policy, "policy_sha256")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing policy: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(json.dumps(policy, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
    print(json.dumps({"status": "FROZEN", "candidate_method_id": candidate_id,
        "source_plan_raw_sha256": source_raw_sha, "policy_sha256": policy["policy_sha256"],
        "contrast_ids": [item["contrast_id"] for item in candidate_contrasts],
        "endpoint_count": len(endpoints), "confirm_audio_read": False, "confirm_effect_read": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
