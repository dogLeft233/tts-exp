#!/usr/bin/env python3
"""Freeze a v4-only 20-clip × 3-seed × 3-frontend development render plan.

The plan contains only new v4 candidate rows. Existing identity/RMS/local
control rows remain in their original ledgers and may be joined downstream
only after the consumer verifies the complete source, clock, model, face, and
audio bindings. This builder reads plan/report metadata only; it opens no
development or confirmation audio and performs no model inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_PLAN_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FRONTEND_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_FRONTEND_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
PROFILE_DISPATCHER_SHA256 = "dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660"
PROFILE_SOURCE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHOD_ID = "spectral_head_fit_v4_native_attempt01"
RUN_ID = "cross_tfg_audio_head_20261003_v4_dev180_attempt02"
CHECKPOINT_SHA256 = "3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4"
PARENT_PLAN_RAW_SHA256 = "49a50a4b5b15727ab99398c38d462e87d86ecae8b73615babfebe2a7c1fc1702"
PARENT_PLAN_PAYLOAD_SHA256 = "2d3dafdb4616ccfb27ef286dc02c614d48bb96025e8856d518ae043d4c06760c"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def load_plan(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    if actual != expected_sha:
        raise ValueError(f"{label} raw SHA mismatch: {actual}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("status") != "FROZEN":
        raise ValueError(f"{label} is not a frozen plan object")
    if value.get("plan_sha256") != payload_sha(value, "plan_sha256"):
        raise ValueError(f"{label} embedded plan payload seal mismatch")
    return value, raw


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dev-plan", type=Path, required=True)
    parser.add_argument("--frontend-roster-plan", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--expected-fit-report-raw-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--fit-report-remote-path", required=True)
    parser.add_argument("--stagea-core", type=Path, required=True)
    parser.add_argument("--cross-frontend-runner", type=Path, required=True)
    parser.add_argument("--audio-pipeline", type=Path, required=True)
    parser.add_argument("--audio-heads", type=Path, required=True)
    parser.add_argument("--profile-dispatcher", type=Path, required=True)
    parser.add_argument("--profile-transform-source", type=Path, required=True)
    parser.add_argument("--parent-candidate-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.expected_checkpoint_sha256.lower() != CHECKPOINT_SHA256:
        raise ValueError("v4 checkpoint SHA differs from the pinned final96 checkpoint")
    parent_raw = args.parent_candidate_plan.read_bytes()
    if sha_bytes(parent_raw) != PARENT_PLAN_RAW_SHA256:
        raise ValueError("parent v4 candidate plan raw SHA mismatch")
    parent_plan = json.loads(parent_raw.decode("utf-8"))
    if (parent_plan.get("status") != "FROZEN"
            or parent_plan.get("run_id") != "cross_tfg_audio_head_20261003_v4_dev180_attempt01"
            or parent_plan.get("plan_sha256") != PARENT_PLAN_PAYLOAD_SHA256
            or parent_plan.get("plan_sha256") != payload_sha(parent_plan, "plan_sha256")):
        raise ValueError("parent v4 candidate plan payload/run binding mismatch")
    base, base_raw = load_plan(args.base_dev_plan, BASE_PLAN_SHA256, "base development plan")
    frontends, frontend_raw = load_plan(args.frontend_roster_plan, FRONTEND_PLAN_SHA256,
                                        "three-frontend roster plan")

    report_path = args.fit_report.resolve(strict=True)
    report_raw = report_path.read_bytes()
    report_sha = sha_bytes(report_raw)
    if report_sha != args.expected_fit_report_raw_sha256.lower():
        raise ValueError(f"v4 fit-report raw SHA mismatch: {report_sha}")
    report = json.loads(report_raw.decode("utf-8"))
    if report.get("schema") != "cross_tfg_audio_head_fit_run_v1" or report.get("status") != "FIT_ONLY_COMPLETE":
        raise ValueError("fit report does not declare the completed fit-only v4 run")
    if report.get("split") != "fit" or int(report.get("optimizer_steps", -1)) != 96:
        raise ValueError("v4 candidate must be the frozen 96-step fit-only checkpoint")
    if report.get("checkpoint_sha256", "").lower() != CHECKPOINT_SHA256:
        raise ValueError("fit report does not bind the pinned final96 checkpoint")
    if report.get("development_audio_opened") is not False or report.get("confirmation_audio_opened") is not False:
        raise ValueError("fit report does not certify closed development and confirmation audio")
    transform = report.get("transform_config")
    architecture = report.get("architecture")
    if not isinstance(transform, dict) or not isinstance(architecture, dict):
        raise ValueError("fit report is missing transform or head architecture metadata")
    if transform.get("method") != "spectral_head" or transform.get("head_architecture") != architecture:
        raise ValueError("fit report transform/head metadata disagree")
    if transform.get("use_global_rms") is not True or transform.get("use_spectral_head") is not True:
        raise ValueError("v4 deployment transform does not match the trained RMS-relative spectral head")

    if base.get("schema_version") != "cross-tfg-audio-eval-v1":
        raise ValueError("base plan schema differs from the frozen Stage-A development contract")
    if base.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ValueError("the frozen development render-seed family changed")
    if base.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("base dev plan unexpectedly declares confirmation seeds")
    base_rows = base.get("rows")
    if not isinstance(base_rows, list) or len(base_rows) != 180:
        raise ValueError("base dev plan must contain the frozen 20 × 3-seed × 3-control roster")
    if [method.get("method_id") for method in base.get("methods", [])] != ["identity", "rms_fixed", "rms_localcompress"]:
        raise ValueError("base dev plan control roster differs from the frozen three-control source")

    model_records = frontends.get("models")
    if not isinstance(model_records, list) or tuple(model.get("model_id") for model in model_records) != EXPECTED_MODELS:
        raise ValueError("frontend roster differs from the three qualified frontend IDs")
    model_by_id = {model["model_id"]: model for model in model_records}
    model_face_ids = {model.get("face_id", "primary_face_static_portrait_v1") for model in model_records}
    if model_face_ids != {"primary_face_static_portrait_v1"}:
        raise ValueError("three-front-end roster does not use the common frozen primary portrait")

    core_path = args.stagea_core.resolve(strict=True)
    wrapper_path = args.cross_frontend_runner.resolve(strict=True)
    if sha_file(core_path) != STAGEA_CORE_SHA256:
        raise ValueError("Stage-A core source differs from the qualified cross-frontend core")
    if sha_file(wrapper_path) != CROSS_FRONTEND_RUNNER_SHA256:
        raise ValueError("cross-frontend adapter wrapper differs from the qualified runner")
    pipeline_path = args.audio_pipeline.resolve(strict=True)
    heads_path = args.audio_heads.resolve(strict=True)
    dispatcher_path = args.profile_dispatcher.resolve(strict=True)
    profile_source_path = args.profile_transform_source.resolve(strict=True)
    if sha_file(dispatcher_path) != PROFILE_DISPATCHER_SHA256:
        raise ValueError("profile cross-frontend dispatcher source differs from the qualified SHA")
    if sha_file(profile_source_path) != PROFILE_SOURCE_SHA256:
        raise ValueError("shared-frequency transform module differs from the qualified SHA")

    methods = base["methods"]
    identity_rows = [row for row in base_rows if row.get("method_id") == "identity"]
    expected_contexts = 20 * 3
    if len(identity_rows) != expected_contexts:
        raise ValueError("base dev plan does not contain 20 identity source rows per seed")
    context_keys = [(row["sample_id"], row["speaker_id"], row["seed"]) for row in identity_rows]
    if len(context_keys) != len(set(context_keys)):
        raise ValueError("base dev identity context keys are duplicated")
    face_bindings = {canonical(row.get("source_face")) for row in identity_rows}
    video_bindings = {canonical(row.get("source_video")) for row in identity_rows}
    if len(face_bindings) != 1 or len(video_bindings) != 1:
        raise ValueError("base dev identity rows do not share one exact static face/video binding")
    samples = {row["sample_id"] for row in identity_rows}
    speakers = {row["speaker_id"] for row in identity_rows}
    seeds = {row["seed"] for row in identity_rows}
    if len(samples) != 20 or len(speakers) != 10 or seeds != {20261003, 20261004, 20261005}:
        raise ValueError("base dev source roster is not the frozen 20 clips/10 speakers/3 seeds")
    for sample_id in samples:
        sample_rows = [row for row in identity_rows if row["sample_id"] == sample_id]
        if len(sample_rows) != 3 or len({row["speaker_id"] for row in sample_rows}) != 1:
            raise ValueError(f"dev sample/seed roster is incomplete or ambiguous: {sample_id}")
        audio_refs = {canonical([row["natural_pcm"], row["score_audio"]]) for row in sample_rows}
        if len(audio_refs) != 1:
            raise ValueError(f"natural/score PCM binding changed across seeds: {sample_id}")

    method_config = copy.deepcopy(transform)
    method_config.update({
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fit_report_path": str(args.fit_report_remote_path),
        "fit_report_sha256": report_sha,
        "fit_plan_raw_sha256": str(report["plan_raw_sha256"]).lower(),
        "fit_optimizer_steps": 96,
        "training_objective": report.get("optimization_config", {}),
        "transform_source_sha256": {
            "run_stagea_dev_attempt03.py": sha_file(core_path),
            "run_cross_frontend_dev_attempt03.py": sha_file(wrapper_path),
            "run_cross_frontend_fit_profiles_attempt02.py": sha_file(dispatcher_path),
            "audio_enhancement_pipeline.py": sha_file(pipeline_path),
            "audio_enhancement_heads.py": sha_file(heads_path),
        },
    })
    method = {
        "method_id": METHOD_ID,
        "kind": "learned_audio_head_candidate",
        "config": method_config,
        "config_sha256": sha_bytes(canonical(method_config)),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fit_report_sha256": report_sha,
        "fit_plan_raw_sha256": str(report["plan_raw_sha256"]).lower(),
        "training_steps": 96,
        "selection_scope": "development_only; final96 checkpoint fixed before these dev renders; confirmation remains closed",
    }

    rows: list[dict[str, Any]] = []
    for source_row in identity_rows:
        for model_id in EXPECTED_MODELS:
            model = model_by_id[model_id]
            row = copy.deepcopy(source_row)
            row["model_id"] = model_id
            row["audio_frontend"] = model["audio_frontend"]
            row["model_role"] = model.get("role", "candidate")
            row["method_id"] = METHOD_ID
            row["method_config_sha256"] = method["config_sha256"]
            row["checkpoint_sha256"] = CHECKPOINT_SHA256
            row["historical_exposed"] = False
            row["generation_settings"] = {
                **row["generation_settings"],
                "generator": model_id,
                "input_audio": "v4_enhanced_generator_condition_only; score_audio_is_original_natural_pcm",
                "audio_transform": METHOD_ID,
                "head_checkpoint_sha256": CHECKPOINT_SHA256,
                "fit_report_sha256": report_sha,
                "seed": int(source_row["seed"]),
            }
            row["row_id"] = (
                f"dev_{row['speaker_id']}_{row['sample_id']}_{model_id}_"
                f"{row['face_id']}_{int(row['seed'])}_{METHOD_ID}"
            )
            rows.append(row)
    if len(rows) != 180:
        raise ValueError(f"v4-only plan has {len(rows)} rows; expected 180")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("generated v4 plan has duplicate row IDs")

    plan = copy.deepcopy(base)
    plan["run_id"] = RUN_ID
    plan["methods"] = [method]
    plan["models"] = copy.deepcopy(model_records)
    plan["rows"] = rows
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["simultaneous_family_ids"] = []
    plan["analysis"]["simultaneous_family_status"] = "NOT_APPLICABLE_CANDIDATE_ONLY_SUBRUN"
    plan["analysis"]["candidate_only_subrun"] = {
        "new_candidate_method_id": METHOD_ID,
        "baseline_join_policy": "Join only previously completed identity/RMS/local controls with exact matching natural PCM, face, model, seed, score frontend, and normalized query/frame clock; otherwise leave comparator unavailable.",
        "selection_claim": "No method selection or inferential claim is produced by this render-only plan.",
    }
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "v4_candidate_only_development_render",
        "scope": "development-only; 20 fixed dev clips × 3 frozen render seeds × 3 qualified TFG frontends; confirmation audio/effects remain closed",
        "planned_rows": 180,
        "candidate_only": True,
        "fresh_baseline_rerender": False,
        "baseline_join": "pending exact input/face/model/seed/clock verification after candidate completion",
        "v4_checkpoint_sha256": CHECKPOINT_SHA256,
        "v4_fit_report_sha256": report_sha,
        "v4_fit_plan_raw_sha256": str(report["plan_raw_sha256"]).lower(),
        "stagea_core_sha256": STAGEA_CORE_SHA256,
        "stagea_runner_sha256": STAGEA_CORE_SHA256,
        "cross_frontend_runner_sha256": CROSS_FRONTEND_RUNNER_SHA256,
        "cross_frontend_dispatcher_sha256": PROFILE_DISPATCHER_SHA256,
        "profile_transform_source_sha256": PROFILE_SOURCE_SHA256,
        "generation_repeat_qc": "NOT_RUN_IN_CANDIDATE_ONLY_SUBRUN; prior baseline repeat records remain separate and are not relabeled as v4 repeats",
        "confirm_audio_read": False,
        "confirm_effects_read": False,
    }
    plan["protocol"]["generator"]["stagea_runner_sha256"] = STAGEA_CORE_SHA256
    plan["protocol"]["generator"]["cross_frontend_runner_sha256"] = CROSS_FRONTEND_RUNNER_SHA256
    plan["protocol"]["generator"]["cross_frontend_dispatcher_sha256"] = PROFILE_DISPATCHER_SHA256
    plan["protocol"]["generator"]["profile_transform_source_sha256"] = PROFILE_SOURCE_SHA256
    for model in plan["models"]:
        generator = model.get("generator")
        if isinstance(generator, dict):
            generator["runner_sha256"] = STAGEA_CORE_SHA256
    plan["source_reference_plans"] = {
        "parent_candidate_plan_attempt01": {
            "path": str(args.parent_candidate_plan),
            "raw_sha256": sha_bytes(parent_raw),
            "plan_sha256": parent_plan.get("plan_sha256"),
            "change": "metadata_only_runner_binding_correction; candidate rows, inputs, models, transform, and checkpoint unchanged",
        },
        "base_dev_plan": {"path": str(args.base_dev_plan), "raw_sha256": sha_bytes(base_raw),
                          "plan_sha256": base["plan_sha256"], "planned_rows": len(base_rows)},
        "frontend_roster_plan": {"path": str(args.frontend_roster_plan), "raw_sha256": sha_bytes(frontend_raw),
                                 "plan_sha256": frontends["plan_sha256"], "model_ids": list(EXPECTED_MODELS)},
        "profile_dispatcher": {"path": str(dispatcher_path), "raw_sha256": sha_file(dispatcher_path),
                                "sha256": PROFILE_DISPATCHER_SHA256},
        "profile_transform_source": {"path": str(profile_source_path), "raw_sha256": sha_file(profile_source_path),
                                      "sha256": PROFILE_SOURCE_SHA256},
        "fit_report": {"path": str(args.fit_report_remote_path), "raw_sha256": report_sha,
                       "fit_plan_raw_sha256": report["plan_raw_sha256"],
                       "checkpoint_sha256": CHECKPOINT_SHA256, "optimizer_steps": 96},
    }
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")

    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    plan_bytes = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    rows_bytes = b"".join(canonical(row) + b"\n" for row in rows)
    plan_path = args.output_dir / "plan.json"
    rows_path = args.output_dir / "planned_rows.jsonl"
    write_new(plan_path, plan_bytes)
    write_new(rows_path, rows_bytes)
    manifest = {
        "schema": "cross_tfg_v4_candidate_only_dev_plan_v1",
        "run_id": RUN_ID,
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": sha_bytes(plan_bytes),
        "planned_rows_sha256": sha_bytes(rows_bytes),
        "planned_rows": len(rows),
        "method_counts": {METHOD_ID: len(rows)},
        "model_counts": {model_id: sum(row["model_id"] == model_id for row in rows) for model_id in EXPECTED_MODELS},
        "sample_count": 20,
        "speaker_count": 10,
        "render_seed_family": [20261003, 20261004, 20261005],
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "baseline_join_pending_exact_row_qualification": True,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path = args.output_dir / "plan_manifest.json"
    write_new(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({
        "status": "FROZEN",
        "run_id": RUN_ID,
        "plan_raw_sha256": sha_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "planned_rows_sha256": sha_file(rows_path),
        "rows": len(rows),
        "models": list(EXPECTED_MODELS),
        "method_id": METHOD_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "confirm_audio_read": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
