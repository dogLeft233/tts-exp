#!/usr/bin/env python3
"""Freeze a learned-head method spec from an audited fit-only checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seal(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha(canonical(body))


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-plan-raw-sha256", required=True)
    parser.add_argument("--fit-plan", type=Path, required=True)
    parser.add_argument("--fit-plan-raw-sha256", required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-report-sha256", required=True)
    parser.add_argument("--fit-steps", type=Path, required=True)
    parser.add_argument("--fit-steps-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method-id", default="spectral_head_fit_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for label, path, expected in (
        ("source plan", args.source_plan, args.source_plan_raw_sha256),
        ("fit plan", args.fit_plan, args.fit_plan_raw_sha256),
        ("fit report", args.fit_report, args.fit_report_sha256),
        ("fit steps", args.fit_steps, args.fit_steps_sha256),
        ("checkpoint", args.checkpoint, args.checkpoint_sha256),
    ):
        actual = sha_file(path)
        if len(expected) != 64 or actual != expected.lower():
            raise ValueError(f"{label} raw SHA256 mismatch: {actual}")

    source_raw_sha = sha_file(args.source_plan)
    source = read_json(args.source_plan, "source StageB plan")
    if source.get("schema_version") != "cross-tfg-audio-eval-v1" or source.get("status") != "FROZEN":
        raise ValueError("source StageB plan is not frozen")
    if source.get("run_id") != "cross_tfg_audio_head_20261003_stageb_dev_attempt03":
        raise ValueError("source StageB run id mismatch")
    if source.get("plan_sha256") != seal(source, "plan_sha256"):
        raise ValueError("source StageB plan payload seal mismatch")
    method24 = next((m for m in source.get("methods", []) if m.get("method_id") == "spectral_head_fit24"), None)
    if not isinstance(method24, dict) or not isinstance(method24.get("config"), dict):
        raise ValueError("source plan is missing the fit24 learned-head method config")

    fit_plan = read_json(args.fit_plan, "historical fit plan")
    if fit_plan.get("schema") != "cross_tfg_audio_head_historical_fullcontext_fit_v2" or fit_plan.get("status") != "FIT_ONLY_FROZEN":
        raise ValueError("fit plan is not the frozen full-context historical-fit v2 plan")
    fit_plan_hash = fit_plan.get("plan_payload_sha256")
    if not isinstance(fit_plan_hash, str) or len(fit_plan_hash) != 64:
        raise ValueError("fit plan payload SHA is missing")
    fit_plan_body = {k: v for k, v in fit_plan.items() if k != "plan_payload_sha256"}
    if sha(canonical(fit_plan_body)) != fit_plan_hash:
        raise ValueError("fit plan embedded payload SHA mismatch")
    fit_scope = fit_plan.get("full_context_fit_v2", {})
    fit_config = fit_plan.get("fit_config", {})
    if fit_plan.get("historical_fit_subset") is not True or fit_scope.get("fit_rows") != 8:
        raise ValueError("fit plan is not restricted to the eight historical-fit rows")
    if fit_scope.get("confirmation_audio_opened") is not False or fit_scope.get("development_audio_opened") is not False or fit_scope.get("model_scoring_or_effects_read") is not False:
        raise ValueError("fit plan scope flags are not clean")
    if fit_config.get("steps_requested") != 96 or fit_config.get("epochs") != 12 or fit_config.get("fit_seed") != 20261003:
        raise ValueError("fit plan step/epoch/seed values differ from the completed 96-step run")

    report = read_json(args.fit_report, "fit report")
    if report.get("status") != "FIT_ONLY_COMPLETE" or report.get("split") != "fit":
        raise ValueError("fit report is not a completed fit-only report")
    if report.get("optimizer_steps") != 96 or report.get("planned_fit_rows") != 8 or report.get("used_fit_rows") != 8:
        raise ValueError("fit report row/step counts differ from the frozen fit plan")
    if report.get("development_audio_opened") is not False or report.get("confirmation_audio_opened") is not False or report.get("effects_selected_on_development") is not False:
        raise ValueError("fit report scope flags are not clean")
    if report.get("checkpoint_sha256") != args.checkpoint_sha256.lower():
        raise ValueError("fit report checkpoint SHA differs from the supplied checkpoint")
    if report.get("steps_jsonl_sha256") != args.fit_steps_sha256.lower():
        raise ValueError("fit report steps SHA differs from the supplied steps ledger")
    step_lines = [line for line in args.fit_steps.read_bytes().splitlines() if line.strip()]
    if len(step_lines) != 96:
        raise ValueError("fit steps ledger does not contain exactly 96 records")

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "cross_tfg_audio_head_checkpoint_v1":
        raise ValueError("checkpoint schema is not the pinned learned audio-head format")
    architecture = checkpoint.get("architecture")
    if architecture != method24["config"].get("head_architecture"):
        raise ValueError("fit-v2 checkpoint architecture differs from StageB fit24 transform architecture")

    method_id = args.method_id
    if not method_id or method_id in {m.get("method_id") for m in source.get("methods", [])}:
        raise ValueError("candidate method id must be new relative to the four-method StageB run")
    config = copy.deepcopy(method24["config"])
    config.update({
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "fit_report_sha256": args.fit_report_sha256.lower(),
        "fit_plan_raw_sha256": args.fit_plan_raw_sha256.lower(),
        "fit_plan_payload_sha256": fit_plan_hash,
        "fit_optimizer_steps": 96,
        "training_objective": "full-audio center-32 27-query median-lag distance minus normalized softmin; 31 lags, tau=0.1; historical fit only",
        "fit_context_mode": "full_original_audio_center_32_frames_27_queries_all31_lags_no_padding",
        "fit_context_rule": fit_config.get("context"),
        "fit_runtime_source_sha256": sha(canonical(fit_plan.get("source_revision", {}).get("fit_package_source_files", []))),
    })
    method = {
        "method_id": method_id,
        "kind": "learned_audio_head_candidate",
        "config": config,
        "config_sha256": sha(canonical(config)),
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "fit_report_sha256": args.fit_report_sha256.lower(),
        "fit_plan_raw_sha256": args.fit_plan_raw_sha256.lower(),
        "training_steps": 96,
        "training_objective": config["training_objective"],
        "selection_scope": "development_only_after_candidate_run_and_independent_consumer_audit; no confirmation audio/effects read",
    }
    spec = {
        "schema_version": "cross_tfg_candidate_method_spec_v1",
        "status": "FROZEN",
        "fit_scope": "historical_fit_only",
        "fit_data_splits": ["fit"],
        "fit_speakers": sorted(fit_scope.get("fit_speakers", [])),
        "fit_rows": 8,
        "optimizer_steps": 96,
        "fit_plan": {"path": str(args.fit_plan), "file_sha256": args.fit_plan_raw_sha256.lower(), "payload_sha256": fit_plan_hash},
        "fit_report": {"path": str(args.fit_report), "file_sha256": args.fit_report_sha256.lower()},
        "fit_steps": {"path": str(args.fit_steps), "file_sha256": args.fit_steps_sha256.lower(), "records": 96},
        "checkpoint": {"path": str(args.checkpoint), "file_sha256": args.checkpoint_sha256.lower()},
        "source_stageb_plan": {"run_id": source["run_id"], "path": str(args.source_plan), "file_sha256": source_raw_sha, "payload_sha256": source["plan_sha256"]},
        "method": method,
        "fit_scope_attestation": {
            "training_objective": fit_config.get("proxy_loss"),
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "development_effects_selected": False,
            "full_context_plan_payload_sha256": fit_plan_hash,
            "checkpoint_architecture": architecture,
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate method spec: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(json.dumps({"status": "FROZEN", "method_id": method_id, "method_config_sha256": method["config_sha256"],
        "candidate_method_spec_raw_sha256": sha(raw), "source_plan_raw_sha256": source_raw_sha,
        "fit_plan_raw_sha256": args.fit_plan_raw_sha256.lower(), "fit_report_sha256": args.fit_report_sha256.lower(),
        "fit_steps_sha256": args.fit_steps_sha256.lower(), "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "training_steps": 96, "fit_scope": "historical_fit_only", "development_audio_opened": False,
        "confirmation_audio_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
