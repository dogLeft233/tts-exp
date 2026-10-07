#!/usr/bin/env python3
"""Build the fixed-step V3 180-row Dev plan from its actual materialization.

This metadata-only adapter requires both the METHOD_CANDIDATE CLI config and
the materializer receipt that binds it.  It derives all candidate digests from
those actual files, then delegates the already-qualified baseline/row logic to
build_v3_candidate_only_dev_plan_attempt03.py without changing that source.
It does not open Dev/Confirm audio, render, score, or call a model/API.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LEGACY_BUILDER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_candidate_only_dev_plan_attempt03.py"
LEGACY_BUILDER_BYTES = 47_301
LEGACY_BUILDER_SHA256 = "94c786591f4bb59f69abd15f5400116fc0b4c0e06655907f20ef7e4ef50282c2"
MATERIALIZATION_SCHEMA = "cross_tfg_v3_step120_method_candidate_materialization_v1"
METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
FIXED_STEPS = 120
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _regular_bytes(path: Path, label: str, cap: int = 8 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink or junction: {cursor}")
    info = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > cap:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != info.st_size:
        raise ValueError(f"{label} changed while it was read")
    return raw


def _load_builder(project_root: Path):
    path = project_root / Path(LEGACY_BUILDER_REL)
    raw = _regular_bytes(path, "qualified legacy candidate builder")
    if len(raw) != LEGACY_BUILDER_BYTES or sha_bytes(raw) != LEGACY_BUILDER_SHA256:
        raise ValueError("qualified legacy candidate builder differs from its frozen source identity")
    spec = importlib.util.spec_from_file_location("qualified_v3_candidate_builder_attempt03", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen candidate builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return path, raw, module


def _repo_relative(root: Path, path: Path, label: str) -> str:
    try:
        rel = path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if not rel or "\\" in rel or ".." in Path(rel).parts:
        raise ValueError(f"{label} is not a safe repository-relative path")
    return rel


def validate_actual_materialization(project_root: Path, config_path: Path,
                                    receipt_path: Path, builder: Any) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    config_rel = _repo_relative(project_root, config_path, "candidate CLI config")
    receipt_rel = _repo_relative(project_root, receipt_path, "materialization receipt")
    config_raw = _regular_bytes(config_path, "candidate CLI config")
    receipt_raw = _regular_bytes(receipt_path, "materialization receipt")
    config_sha = sha_bytes(config_raw)
    receipt_sha = sha_bytes(receipt_raw)
    config = json.loads(config_raw.decode("utf-8"))
    receipt = json.loads(receipt_raw.decode("utf-8"))
    if not isinstance(config, dict) or not isinstance(receipt, dict):
        raise ValueError("candidate config and materialization receipt must be JSON objects")
    for key, expected in (
        ("schema", MATERIALIZATION_SCHEMA),
        ("status", "MATERIALIZED_FIXED_STEP120_METHOD_CANDIDATE"),
        ("profile_role", "METHOD_CANDIDATE"),
        ("optimizer_steps", FIXED_STEPS),
        ("fit_row_count", 40),
        ("fit_speaker_count", 20),
        ("zero_head_gate_rows_passed", 40),
        ("development_audio_opened", False),
        ("confirmation_audio_opened", False),
        ("method_selected_using_development", False),
        ("model_execution_performed", False),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"materialization receipt {key} differs from the fixed fit-only contract")
    if config.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1" or config.get("profile_role") != "METHOD_CANDIDATE":
        raise ValueError("candidate config is not the materialized V3 METHOD_CANDIDATE")
    if config.get("candidate_provenance") != receipt.get("candidate_provenance"):
        raise ValueError("candidate config provenance does not exactly match its materializer receipt")
    cfg_pin = receipt.get("candidate_config")
    if not isinstance(cfg_pin, dict) or set(cfg_pin) != {"path", "bytes", "sha256"}:
        raise ValueError("materializer receipt lacks its exact candidate-config pin")
    if (cfg_pin.get("path") != config_rel or cfg_pin.get("bytes") != len(config_raw)
            or cfg_pin.get("sha256") != config_sha):
        raise ValueError("supplied METHOD_CANDIDATE config is not the exact file bound by its receipt")
    source_bindings = receipt.get("source_bindings")
    provenance = receipt.get("candidate_provenance")
    if not isinstance(source_bindings, dict) or not isinstance(provenance, dict):
        raise ValueError("materializer receipt lacks source/provenance bindings")
    if set(provenance) != {"checkpoint", "training_record", "selection_record"}:
        raise ValueError("materializer receipt provenance keys differ from the fixed V3 materializer contract")
    for name, source_name in (("checkpoint", "original_checkpoint"), ("training_record", "original_training_record")):
        candidate_pin = provenance.get(name)
        original_pin = source_bindings.get(source_name)
        if not isinstance(candidate_pin, dict) or not isinstance(original_pin, dict):
            raise ValueError(f"materializer receipt lacks {name} source/candidate pins")
        if (candidate_pin.get("bytes") != original_pin.get("bytes")
                or candidate_pin.get("sha256") != original_pin.get("sha256")):
            raise ValueError(f"materialized {name} does not preserve the exact fit artifact bytes")
    if set(receipt.get("candidate_provenance", {})) != set(config.get("candidate_provenance", {})):
        raise ValueError("config and receipt candidate provenance roster differs")

    # The already-reviewed builder validates all config source pins, record
    # schemas, checkpoint/training/selection bindings, and the fixed terminal
    # step.  Calling that validator is still metadata-only.
    candidate = builder._candidate_bundle(project_root, config_path, config_sha)
    if candidate.get("optimizer_steps") != FIXED_STEPS:
        raise ValueError("materialized candidate is not the fixed terminal step-120 checkpoint")
    return {
        "config_path": config_rel,
        "config_bytes": len(config_raw),
        "config_raw_sha256": config_sha,
        "receipt_path": receipt_rel,
        "receipt_bytes": len(receipt_raw),
        "receipt_raw_sha256": receipt_sha,
        "receipt_schema": receipt["schema"],
        "receipt_status": receipt["status"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_record_sha256": candidate["training_record_sha256"],
        "selection_record_sha256": candidate["selection_record_sha256"],
        "optimizer_steps": FIXED_STEPS,
    }


def _rewrite_sealed_plan(output_dir: Path, run_id: str, actual: dict[str, Any],
                         wrapper_rel: str, wrapper_sha: str) -> dict[str, Any]:
    plan_path = output_dir / "plan.json"
    rows_path = output_dir / "planned_rows.jsonl"
    manifest_path = output_dir / "plan_manifest.json"
    plan = json.loads(_regular_bytes(plan_path, "candidate plan", 16 * 1024 * 1024).decode("utf-8"))
    manifest = json.loads(_regular_bytes(manifest_path, "candidate plan manifest", 2 * 1024 * 1024).decode("utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("execution", {}).get("planned_rows") != 180:
        raise ValueError("qualified builder did not return the frozen 180-row candidate plan")
    if tuple(model.get("model_id") for model in plan.get("models", [])) != MODELS:
        raise ValueError("qualified builder returned an unexpected model roster")
    if len(plan.get("rows", [])) != 180 or len({row.get("row_id") for row in plan["rows"]}) != 180:
        raise ValueError("qualified builder did not return 180 unique candidate rows")
    if any(row.get("method_id") != METHOD_ID or row.get("split") != "dev" for row in plan["rows"]):
        raise ValueError("candidate rows do not all use the materialized step-120 Dev method")
    cohort = {(row.get("sample_id"), row.get("speaker_id")) for row in plan["rows"]}
    if len(cohort) != 20 or len({sample for sample, _speaker in cohort}) != 20 or len({speaker for _sample, speaker in cohort}) != 10:
        raise ValueError("candidate plan is not bound to the fixed 20-clip cohort")
    seeds = plan.get("generation_seed_family", {}).get("dev")
    if seeds != [20261003, 20261004, 20261005]:
        raise ValueError("candidate plan does not retain the fixed three-seed Dev family")
    expected_rows = {
        (sample, speaker, model_id, seed)
        for sample, speaker in cohort for model_id in MODELS for seed in seeds
    }
    actual_rows = {(row.get("sample_id"), row.get("speaker_id"), row.get("model_id"), row.get("seed"))
                   for row in plan["rows"]}
    if len(actual_rows) != 180 or actual_rows != expected_rows:
        raise ValueError("candidate plan does not contain the complete 20×3×3 model/seed row grid")
    if (manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("planned_rows") != 180
            or manifest.get("candidate_checkpoint_sha256") != actual["checkpoint_sha256"]):
        raise ValueError("qualified builder manifest does not bind the actual step-120 candidate")
    rows_raw = _regular_bytes(rows_path, "planned candidate rows", 16 * 1024 * 1024)
    if rows_raw != b"".join(canonical(row) + b"\n" for row in plan["rows"]):
        raise ValueError("qualified builder rows differ from its frozen candidate plan")
    if manifest.get("planned_rows_sha256") != sha_bytes(rows_raw):
        raise ValueError("qualified builder manifest planned-row hash differs before receipt binding")
    plan["run_id"] = run_id
    plan["source_reference_plans"]["materialization_receipt"] = {
        "path": actual["receipt_path"],
        "bytes": actual["receipt_bytes"],
        "raw_sha256": actual["receipt_raw_sha256"],
        "schema": actual["receipt_schema"],
        "status": actual["receipt_status"],
        "optimizer_steps": FIXED_STEPS,
    }
    plan["source_reference_plans"]["materialized_plan_builder"] = {
        "path": wrapper_rel,
        "raw_sha256": wrapper_sha,
        "qualified_row_builder_path": LEGACY_BUILDER_REL,
        "qualified_row_builder_bytes": LEGACY_BUILDER_BYTES,
        "qualified_row_builder_sha256": LEGACY_BUILDER_SHA256,
    }
    plan["execution"]["materialization_receipt_raw_sha256"] = actual["receipt_raw_sha256"]
    plan["execution"]["materialization_receipt_path"] = actual["receipt_path"]
    plan["execution"]["candidate_config_raw_sha256"] = actual["config_raw_sha256"]
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")
    plan_raw = pretty(plan)
    plan_path.write_bytes(plan_raw)

    manifest["run_id"] = run_id
    manifest["plan_sha256"] = plan["plan_sha256"]
    manifest["plan_raw_sha256"] = sha_bytes(plan_raw)
    manifest["materialization_receipt_raw_sha256"] = actual["receipt_raw_sha256"]
    manifest["candidate_config_raw_sha256"] = actual["config_raw_sha256"]
    manifest["materialized_plan_builder_sha256"] = wrapper_sha
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path.write_bytes(pretty(manifest))
    return {"plan_sha256": plan["plan_sha256"], "plan_raw_sha256": sha_bytes(plan_raw),
            "planned_rows_sha256": sha_bytes(rows_raw), "manifest_sha256": manifest["manifest_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-dev-plan", type=Path, required=True)
    parser.add_argument("--frontend-roster-plan", type=Path, required=True)
    parser.add_argument("--candidate-cli-config", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--baseline-evidence-archive", type=Path, required=True)
    parser.add_argument("--stagea-core", type=Path, required=True)
    parser.add_argument("--cross-frontend-runner", type=Path, required=True)
    parser.add_argument("--v3-dispatcher", type=Path, required=True)
    parser.add_argument("--v3-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from this repository")
    builder_path, builder_raw, builder = _load_builder(project_root)
    actual = validate_actual_materialization(project_root, args.candidate_cli_config,
                                             args.materialization_receipt, builder)
    wrapper_path = Path(__file__).resolve(strict=True)
    wrapper_rel = _repo_relative(project_root, wrapper_path, "materialized Dev plan builder")
    wrapper_raw = _regular_bytes(wrapper_path, "materialized Dev plan builder", 2 * 1024 * 1024)
    wrapper_sha = sha_bytes(wrapper_raw)
    out = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    if out.exists() or out.is_symlink() or getattr(out, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing existing candidate-plan output directory: {out}")

    forwarded = [
        "--project-root", str(project_root),
        "--base-dev-plan", str(args.base_dev_plan),
        "--frontend-roster-plan", str(args.frontend_roster_plan),
        "--candidate-cli-config", str(args.candidate_cli_config.resolve(strict=True)),
        "--candidate-cli-config-raw-sha256", actual["config_raw_sha256"],
        "--baseline-evidence-archive", str(args.baseline_evidence_archive),
        "--stagea-core", str(args.stagea_core),
        "--cross-frontend-runner", str(args.cross_frontend_runner),
        "--v3-dispatcher", str(args.v3_dispatcher),
        "--v3-cli", str(args.v3_cli),
        "--output-dir", str(out),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [str(builder_path), *forwarded]
        with contextlib.redirect_stdout(io.StringIO()):
            code = builder.main()
    finally:
        sys.argv = previous_argv
    if code != 0:
        return code

    sealed = _rewrite_sealed_plan(out, "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02",
                                  actual, wrapper_rel, wrapper_sha)
    print(json.dumps({
        "status": "FROZEN",
        "run_id": "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02",
        "planned_rows": 180,
        "clips": 20,
        "seeds": [20261003, 20261004, 20261005],
        "models": list(MODELS),
        "method_id": METHOD_ID,
        "materialization_receipt_raw_sha256": actual["receipt_raw_sha256"],
        "candidate_config_raw_sha256": actual["config_raw_sha256"],
        "checkpoint_sha256": actual["checkpoint_sha256"],
        "baseline_rows_reused": 360,
        "confirmation_audio_read": False,
        "development_audio_opened_by_builder": False,
        "rendering_or_scoring_performed": False,
        "output_dir": str(out),
        **sealed,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
