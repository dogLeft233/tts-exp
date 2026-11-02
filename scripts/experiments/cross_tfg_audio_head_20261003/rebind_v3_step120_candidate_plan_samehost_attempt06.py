#!/usr/bin/env python3
"""Rebind a materialized step-120 Dev180 slate to actual same-host controls.

This metadata-only join runs after the reviewed step-120 materializer/builder
has produced its exact METHOD_CANDIDATE plan and after the same-host Dev360
identity/RMS plan exists. It replaces stale deployment paths with the exact
current-host runtime receipt, original-source paths with metadata-matched
current paths, and historical baseline references with fresh Dev360 row IDs.
No audio is opened and no model, renderer, scorer, API, or Confirm path is run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHOD_CANDIDATE = "shared_recipient_transition_head_v3_fit40_step120"
METHODS = ("identity", "rms_fixed")
MODEL_BINDING_FIELDS = (
    "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
    "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den",
)
EXPECTED_GPU = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, label: str, cap: int = 24 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > cap:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return raw


def read_json(path: Path, label: str, expected_sha: str | None = None) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA differs from supplied pin")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def load_rows(path: Path, label: str, expected_sha: str | None = None) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_regular(path, label)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA differs from supplied pin")
    rows = [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} includes a non-object row")
    return rows, raw


def require_frozen_plan(plan: dict[str, Any], raw: bytes, expected_sha: str, label: str) -> None:
    if (sha_bytes(raw) != expected_sha.lower() or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")):
        raise ValueError(f"{label} raw/payload SHA or frozen status mismatch")


def rebind_source_paths(plan: dict[str, Any], binding: dict[str, Any]) -> None:
    dev_contexts = binding["dev_contexts"]
    sources = binding["shared_sources"]
    for row in plan["rows"]:
        record = dev_contexts[row["sample_id"]]
        for field in ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"):
            if row["natural_pcm"].get(field) != record.get(field):
                raise ValueError(f"{row['row_id']} Dev PCM differs from runtime binding at {field}")
        if row.get("score_audio", {}).get("pcm_sha256") != record.get("pcm_sha256"):
            raise ValueError(f"{row['row_id']} score-audio PCM no longer matches original natural PCM")
        row["natural_pcm"]["path"] = record["path"]
        row["score_audio"]["path"] = record["path"]
        for key in ("source_face", "source_video"):
            current = sources[key]
            if row[key].get("file_sha256") != current.get("file_sha256"):
                raise ValueError(f"{row['row_id']} {key} differs from current-host binding")
            if key == "source_video" and (row[key].get("fps_num"), row[key].get("fps_den")) != (
                current.get("fps_num"), current.get("fps_den")
            ):
                raise ValueError("current source video clock differs from candidate plan")
            row[key]["path"] = current["path"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan-raw-sha256", required=True)
    parser.add_argument("--candidate-planned-rows", type=Path, required=True)
    parser.add_argument("--candidate-planned-rows-sha256", required=True)
    parser.add_argument("--candidate-plan-manifest", type=Path, required=True)
    parser.add_argument("--candidate-plan-manifest-raw-sha256", required=True)
    parser.add_argument("--controls-plan", type=Path, required=True)
    parser.add_argument("--controls-plan-raw-sha256", required=True)
    parser.add_argument("--controls-planned-rows", type=Path, required=True)
    parser.add_argument("--controls-planned-rows-sha256", required=True)
    parser.add_argument("--controls-plan-manifest", type=Path, required=True)
    parser.add_argument("--controls-plan-manifest-raw-sha256", required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]

    candidate, candidate_raw = read_json(args.candidate_plan, "candidate plan", args.candidate_plan_raw_sha256)
    require_frozen_plan(candidate, candidate_raw, args.candidate_plan_raw_sha256, "candidate plan")
    controls, controls_raw = read_json(args.controls_plan, "same-host controls plan", args.controls_plan_raw_sha256)
    require_frozen_plan(controls, controls_raw, args.controls_plan_raw_sha256, "same-host controls plan")
    candidate_rows_file, candidate_rows_raw = load_rows(args.candidate_planned_rows, "candidate rows", args.candidate_planned_rows_sha256)
    controls_rows_file, controls_rows_raw = load_rows(args.controls_planned_rows, "controls rows", args.controls_planned_rows_sha256)
    candidate_manifest, candidate_manifest_raw = read_json(
        args.candidate_plan_manifest, "candidate plan manifest", args.candidate_plan_manifest_raw_sha256
    )
    controls_manifest, controls_manifest_raw = read_json(
        args.controls_plan_manifest, "controls plan manifest", args.controls_plan_manifest_raw_sha256
    )
    if candidate_rows_file != candidate.get("rows") or len(candidate_rows_file) != 180:
        raise ValueError("candidate planned-row file differs from its frozen 180-row plan")
    if controls_rows_file != controls.get("rows") or len(controls_rows_file) != 360:
        raise ValueError("controls planned-row file differs from its frozen 360-row plan")
    for value, rows_raw, plan in ((candidate_manifest, candidate_rows_raw, candidate),
                                  (controls_manifest, controls_rows_raw, controls)):
        if value.get("plan_sha256") != plan.get("plan_sha256") or value.get("planned_rows_sha256") != sha_bytes(rows_raw):
            raise ValueError("one plan manifest is not bound to its exact plan and row file")

    runtime_raw = read_regular(args.runtime_binding, "runtime binding", 8 * 1024 * 1024)
    runtime_sha = sha_bytes(runtime_raw)
    if runtime_sha != args.runtime_binding_raw_sha256.lower():
        raise ValueError("runtime binding raw SHA differs from supplied pin")
    binding = json.loads(runtime_raw.decode("utf-8"))
    if not isinstance(binding, dict) or binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS":
        raise ValueError("runtime binding is not a PASS same-host receipt")
    if binding.get("gpu_uuid") != EXPECTED_GPU:
        raise ValueError("runtime GPU identity differs from the qualified same-host QC06 device")
    for name, plan in (("candidate", candidate), ("controls", controls)):
        runtime_ref = plan.get("execution", {}).get("samehost_runtime_binding", {})
        if runtime_ref.get("raw_sha256") != runtime_sha or runtime_ref.get("gpu_uuid") != EXPECTED_GPU:
            raise ValueError(f"{name} plan is not bound to this exact same-host runtime receipt")

    if (candidate.get("execution", {}).get("candidate_checkpoint_sha256") is None
            or candidate.get("execution", {}).get("fixed_optimizer_steps") != 120
            or candidate.get("execution", {}).get("confirm_audio_read") is not False
            or candidate.get("execution", {}).get("confirm_effects_read") is not False):
        raise ValueError("candidate plan is not a closed-Confirm fixed step-120 slate")
    if len(candidate.get("methods", [])) != 1 or candidate["methods"][0].get("method_id") != METHOD_CANDIDATE:
        raise ValueError("candidate plan has an unexpected method roster")
    if (len(controls.get("methods", [])) != 2
            or tuple(method.get("method_id") for method in controls["methods"]) != METHODS):
        raise ValueError("controls plan must contain only identity and rms_fixed")
    if candidate.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ValueError("candidate seed family differs from the frozen Dev context")
    if tuple(model.get("model_id") for model in candidate.get("models", [])) != MODELS:
        raise ValueError("candidate model roster differs from the exact three-front-end roster")
    if tuple(model.get("model_id") for model in controls.get("models", [])) != MODELS:
        raise ValueError("controls model roster differs from the exact three-front-end roster")

    models_by_id = {model["model_id"]: model for model in controls["models"]}
    runtime_backends = binding.get("backends", {})
    for model in candidate["models"]:
        model_id = model["model_id"]
        control_model = models_by_id[model_id]
        if {key: model.get(key) for key in MODEL_BINDING_FIELDS} != {
            key: control_model.get(key) for key in MODEL_BINDING_FIELDS
        }:
            raise ValueError(f"candidate and new-host control model pins differ: {model_id}")
        backend = runtime_backends.get(model_id)
        if not isinstance(backend, dict) or backend.get("model_binding") != {
            key: model.get(key) for key in MODEL_BINDING_FIELDS
        }:
            raise ValueError(f"runtime model binding does not match candidate model identity: {model_id}")
        model["generator"] = copy.deepcopy(backend["generator"])
        model["runtime_dependency_receipt_sha256"] = sha_bytes(canonical(backend))
        model["runtime_dependency_files"] = copy.deepcopy(backend["dependencies"])

    rebind_source_paths(candidate, binding)
    controls_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    control_run_id = controls.get("run_id")
    for row in controls["rows"]:
        key = (row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))
        control_key = (*key, row["method_id"])
        if control_key in controls_by_key:
            raise ValueError(f"duplicate same-host control context: {control_key}")
        controls_by_key[control_key] = row
    candidate_rows: list[dict[str, Any]] = []
    for row in candidate["rows"]:
        key = (row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))
        if (*key, "identity") not in controls_by_key or (*key, "rms_fixed") not in controls_by_key:
            raise ValueError(f"candidate has no exact identity/RMS control pair: {key}")
        row["baseline_reference_row_ids"] = {
            method: controls_by_key[(*key, method)]["row_id"] for method in METHODS
        }
        row["baseline_reference_run_ids"] = {method: control_run_id for method in METHODS}
        candidate_rows.append(row)
    if len(candidate_rows) != 180 or len(controls_by_key) != 360:
        raise ValueError("the new same-host baseline/candidate denominator changed")

    # Keep the old archive identity only as a historical comparator. No current
    # candidate row may retain any reference derived from its old control IDs.
    historical_archive = {
        "path": "runs/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt07_readback_attempt01/static_outputs_attempt02/provenance/baseline_540_evidence.tar.gz",
        "bytes": 18784301,
        "sha256": "349c5b62f09c1a775013039d2abb3bd72d7c6b6233531fb7f87e6398fff981d4",
        "role": "historical_only_not_joined",
    }
    candidate["execution"]["fresh_baseline_rerender"] = True
    candidate["execution"]["same_host_controls"] = True
    candidate["execution"]["baseline_controls"] = {
        "run_id": control_run_id,
        "plan_sha256": controls["plan_sha256"],
        "plan_raw_sha256": sha_bytes(controls_raw),
        "planned_rows_sha256": sha_bytes(controls_rows_raw),
        "runtime_binding_raw_sha256": runtime_sha,
        "gpu_uuid": EXPECTED_GPU,
        "control_rows": {method: sum(row["method_id"] == method for row in controls["rows"]) for method in METHODS},
        "historical_archive_not_used": historical_archive,
    }
    candidate.setdefault("source_reference_plans", {})["samehost_controls_dev360"] = {
        "path": str(args.controls_plan.resolve()), "plan_raw_sha256": sha_bytes(controls_raw),
        "plan_sha256": controls["plan_sha256"], "rows_raw_sha256": sha_bytes(controls_rows_raw),
        "manifest_raw_sha256": sha_bytes(controls_manifest_raw), "run_id": control_run_id,
    }
    candidate["source_reference_plans"]["samehost_runtime_binding"] = {
        "path": str(args.runtime_binding.resolve()), "bytes": len(runtime_raw), "raw_sha256": runtime_sha,
        "gpu_uuid": EXPECTED_GPU,
    }
    candidate["execution"]["samehost_runtime_binding"] = {
        "path": str(args.runtime_binding.resolve()), "bytes": len(runtime_raw), "raw_sha256": runtime_sha,
        "gpu_uuid": EXPECTED_GPU,
    }
    controls_execution = controls.get("execution", {})
    strict_policy = controls_execution.get("strict_renderer_determinism")
    qc06_policy = controls_execution.get("qc06_determinism_qualification")
    source_closure = controls_execution.get("samehost_dev360_source_closure")
    if not isinstance(strict_policy, dict) or not isinstance(qc06_policy, dict) or not isinstance(source_closure, dict):
        raise ValueError("same-host controls plan lacks the frozen strict renderer policy/QC/source closure")
    candidate["execution"]["strict_renderer_determinism"] = copy.deepcopy(strict_policy)
    candidate["execution"]["qc06_determinism_qualification"] = copy.deepcopy(qc06_policy)
    candidate["execution"]["samehost_dev360_source_closure"] = copy.deepcopy(source_closure)
    candidate["execution"]["samehost_runtime_binding"]["determinism_policy_sha256"] = qc06_policy.get("determinism_policy_sha256")
    candidate_guard_path = project_root / (
        "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt04.py"
    )
    candidate_guard_path = candidate_guard_path.resolve(strict=True)
    candidate_guard_raw = read_regular(candidate_guard_path, "same-host step-120 guarded launcher", 16 * 1024 * 1024)
    candidate_guard_ref = {
        "path": candidate_guard_path.relative_to(Path(__file__).resolve().parents[3]).as_posix(),
        "bytes": len(candidate_guard_raw), "raw_sha256": sha_bytes(candidate_guard_raw),
    }
    candidate["execution"]["samehost_candidate_guarded_launcher"] = candidate_guard_ref
    candidate["source_reference_plans"]["samehost_candidate_guarded_launcher"] = copy.deepcopy(candidate_guard_ref)
    candidate["source_reference_plans"]["samehost_strict_renderer_policy"] = {
        "bootstrap_source": copy.deepcopy(strict_policy.get("bootstrap_source")),
        "policy_values": copy.deepcopy(strict_policy.get("policy_values")),
        "qualified_policy_reference_sha256": strict_policy.get("qualified_policy_reference_sha256"),
        "controls_plan_raw_sha256": sha_bytes(controls_raw),
        "runtime_binding_raw_sha256": runtime_sha,
    }
    candidate.pop("plan_sha256", None)
    candidate["plan_sha256"] = payload_sha(candidate, "plan_sha256")

    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink() or getattr(output, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing existing candidate rebind output: {output}")
    output.mkdir(parents=True)
    plan_raw_out = pretty(candidate)
    rows_raw_out = b"".join(canonical(row) + b"\n" for row in candidate_rows)
    (output / "plan.json").write_bytes(plan_raw_out)
    (output / "planned_rows.jsonl").write_bytes(rows_raw_out)
    manifest = copy.deepcopy(candidate_manifest)
    manifest["run_id"] = candidate["run_id"]
    manifest["plan_sha256"] = candidate["plan_sha256"]
    manifest["plan_raw_sha256"] = sha_bytes(plan_raw_out)
    manifest["planned_rows_sha256"] = sha_bytes(rows_raw_out)
    manifest["samehost_controls_plan_sha256"] = controls["plan_sha256"]
    manifest["samehost_controls_plan_raw_sha256"] = sha_bytes(controls_raw)
    manifest["samehost_controls_plan_manifest_raw_sha256"] = sha_bytes(controls_manifest_raw)
    manifest["samehost_runtime_binding_raw_sha256"] = runtime_sha
    manifest["historical_baseline_archive_used"] = False
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    (output / "plan_manifest.json").write_bytes(pretty(manifest))
    print(json.dumps({"status": "FROZEN", "run_id": candidate["run_id"], "planned_rows": 180,
                      "controls_run_id": control_run_id, "controls_rows": 360,
                      "gpu_uuid": EXPECTED_GPU, "runtime_binding_raw_sha256": runtime_sha,
                      "historical_baseline_archive_used": False,
                      "confirm_audio_read": False, "confirm_effects_read": False,
                      "development_audio_opened_by_rebinder": False,
                      "plan_raw_sha256": sha_bytes(plan_raw_out),
                      "planned_rows_sha256": sha_bytes(rows_raw_out)},
                     sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
