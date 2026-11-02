#!/usr/bin/env python3
"""Freeze a V3 step-120 candidate-only 180-row Dev render slate.

Reads frozen plan/result metadata only. It does not open development audio,
run a model, choose a checkpoint, or read confirmation data. Identity and RMS
references are pinned to a complete existing Stage-A result ledger and are
joined by the exact sample/speaker/model/face/seed and original-PCM bindings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


BASE_PLAN_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FRONTEND_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_FRONTEND_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
V3_METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
V3_METHOD_CONFIG = "v3_parameter_checkpoint"
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
BASE_METHODS = ("identity", "rms_fixed", "rms_localcompress")
REQUIRED_BASELINE_METHODS = ("identity", "rms_fixed")
FIXED_TRAINING_STEPS = 120


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()


def load_frozen_plan(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if sha_bytes(raw) != expected_sha:
        raise ValueError(f"{label} raw SHA mismatch: {sha_bytes(raw)}")
    value = json.loads(raw.decode("utf-8"))
    if value.get("status") != "FROZEN" or value.get("plan_sha256") != payload_sha(value, "plan_sha256"):
        raise ValueError(f"{label} is not a valid frozen plan")
    return value, raw


def _safe_repo_file(project_root: Path, pin: dict[str, Any], label: str) -> tuple[Path, bytes]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} must contain exactly path/bytes/sha256")
    rel = pin["path"]
    if not isinstance(rel, str) or not rel or "\\" in rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"{label} path must be safe and repository-relative")
    lexical = project_root.joinpath(*Path(rel).parts)
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} has a symlink path component: {cursor}")
    path = lexical.resolve(strict=True)
    try:
        path.relative_to(project_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes the declared project root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular repository file")
    raw = path.read_bytes()
    if len(raw) != pin["bytes"] or sha_bytes(raw) != pin["sha256"]:
        raise ValueError(f"{label} file differs from its declared bytes/SHA")
    return path, raw


def _candidate_bundle(project_root: Path, cli_config_path: Path, expected_raw_sha: str) -> dict[str, Any]:
    config_raw = cli_config_path.read_bytes()
    if sha_bytes(config_raw) != expected_raw_sha:
        raise ValueError("V3 CLI config raw SHA differs from its supplied exact pin")
    config = json.loads(config_raw.decode("utf-8"))
    if config.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1":
        raise ValueError("candidate config is not the frozen V3 CLI schema")
    if config.get("profile_role") != "METHOD_CANDIDATE":
        raise ValueError("Dev slate requires the frozen METHOD_CANDIDATE V3 CLI role")
    parameter = config.get("parameter")
    if not isinstance(parameter, dict) or parameter.get("shape") != [3, 3] or parameter.get("dtype") != "float32":
        raise ValueError("V3 candidate must contain the frozen nine-scalar float32 parameter")
    values = parameter.get("values")
    if not isinstance(values, list) or len(values) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in values):
        raise ValueError("V3 candidate must contain a 3×3 parameter matrix")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for row in values for v in row):
        raise ValueError("V3 candidate parameters must be nine finite JSON numbers")

    source_pins = config.get("source_pins")
    implementation = config.get("implementation")
    provenance = config.get("candidate_provenance")
    if not isinstance(source_pins, dict) or not isinstance(implementation, dict):
        raise ValueError("V3 CLI config is missing implementation/source pins")
    if not isinstance(provenance, dict) or set(provenance) != {"checkpoint", "training_record", "selection_record"}:
        raise ValueError("V3 candidate config requires checkpoint, training record, and fixed selection record")
    entry = source_pins.get("cli")
    if not isinstance(entry, dict) or entry.get("path") != "scripts/audio_enhancement/enhance_v3_attempt02.py":
        raise ValueError("candidate config must pin the exact supported V3 CLI entrypoint")
    for name, pin in source_pins.items():
        if not isinstance(pin, dict):
            raise ValueError(f"invalid V3 source pin: {name}")
        _safe_repo_file(project_root, pin, f"V3 source {name}")
    if implementation.get("path") != "scripts/audio_enhancement/shared_recipient_transition_head_v3.py":
        raise ValueError("candidate config implementation is not the frozen V3 head")
    if implementation != source_pins.get("head_v3"):
        raise ValueError("V3 implementation pin differs from its source closure")

    loaded: dict[str, tuple[dict[str, Any], bytes, Path]] = {}
    for name in ("checkpoint", "training_record", "selection_record"):
        pin = provenance[name]
        path, raw = _safe_repo_file(project_root, pin, f"candidate provenance {name}")
        record = json.loads(raw.decode("utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"candidate {name} must be a JSON object")
        loaded[name] = (record, raw, path)
    checkpoint, checkpoint_raw, checkpoint_path = loaded["checkpoint"]
    training, training_raw, training_path = loaded["training_record"]
    selection, selection_raw, selection_path = loaded["selection_record"]
    constructor = config.get("constructor")
    if not isinstance(constructor, dict):
        raise ValueError("V3 candidate constructor is missing")
    constructor_sha = sha_bytes(canonical(constructor))
    if checkpoint.get("schema") != "cross_tfg_v3_parameter_checkpoint_v1" or checkpoint.get("head_schema") != config.get("head_schema"):
        raise ValueError("step120 checkpoint schema/head mismatch")
    if checkpoint.get("implementation") != implementation or checkpoint.get("constructor_sha256") != constructor_sha:
        raise ValueError("step120 checkpoint implementation/constructor binding mismatch")
    if checkpoint.get("parameter") != parameter:
        raise ValueError("V3 CLI theta differs from the frozen step120 checkpoint")
    if checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise ValueError("step120 fit checkpoint must not contain a Dev-selected weight")
    bindings = checkpoint.get("training_artifact_bindings")
    if not isinstance(bindings, dict) or bindings.get("completed_optimizer_steps") != FIXED_TRAINING_STEPS:
        raise ValueError("checkpoint does not bind exactly 120 optimizer steps")
    if training.get("schema") != "cross_tfg_v3_candidate_training_record_v1" or training.get("status") != "COMPLETE":
        raise ValueError("V3 training record is not complete")
    if training.get("completed_optimizer_steps") != FIXED_TRAINING_STEPS or training.get("planned_optimizer_steps") != FIXED_TRAINING_STEPS:
        raise ValueError("V3 training record is not fixed at the terminal 120 steps")
    if training.get("checkpoint_sha256") != sha_bytes(checkpoint_raw):
        raise ValueError("V3 training record does not bind the exact checkpoint JSON")
    if training.get("head_implementation_sha256") != implementation["sha256"]:
        raise ValueError("V3 training record does not bind the V3 implementation")
    if training.get("development_audio_opened") is not False or training.get("confirmation_audio_opened") is not False:
        raise ValueError("V3 training record must certify that Dev/Confirm stayed closed during fit")
    if training.get("selection_record_created") is not False or training.get("selection_status") != "NOT_SELECTED":
        raise ValueError("fit record must remain unselected before Dev evaluation")
    if selection.get("schema") != "cross_tfg_v3_candidate_selection_record_v1" or selection.get("status") != "FROZEN":
        raise ValueError("the V3 method freeze record must be present before Dev")
    if selection.get("checkpoint_sha256") != sha_bytes(checkpoint_raw):
        raise ValueError("V3 method freeze record does not bind the step120 checkpoint")
    if selection.get("training_record_sha256") != sha_bytes(training_raw):
        raise ValueError("V3 method freeze record does not bind its complete fit record")
    if selection.get("selection_rule") != "fixed_terminal_optimizer_step_120_no_development_selection":
        raise ValueError("V3 freeze record must name the predeclared fixed-step120 rule")
    if selection.get("selected_optimizer_step") != FIXED_TRAINING_STEPS:
        raise ValueError("V3 freeze record must bind the terminal optimizer step 120")
    if selection.get("development_audio_opened") is not False or selection.get("confirmation_audio_opened") is not False:
        raise ValueError("V3 fixed-step freeze record must certify Dev/Confirm were not used to choose weights")
    return {
        "config": config,
        "config_raw_sha256": expected_raw_sha,
        "checkpoint_sha256": sha_bytes(checkpoint_raw),
        "checkpoint_path": str(checkpoint_path),
        "training_record_sha256": sha_bytes(training_raw),
        "training_record_path": str(training_path),
        "selection_record_sha256": sha_bytes(selection_raw),
        "selection_record_path": str(selection_path),
        "optimizer_steps": FIXED_TRAINING_STEPS,
    }


def _baseline_rows(results_path: Path, manifest_path: Path, expected_methods: tuple[str, ...],
                   identity_contexts: list[dict[str, Any]], models: list[dict[str, Any]]) -> tuple[dict[str, dict[tuple[str, str, str, int], dict[str, Any]]], dict[str, Any]]:
    results_raw = results_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("results_jsonl_sha256") != sha_bytes(results_raw):
        raise ValueError("frozen control result manifest does not bind the supplied terminal JSONL")
    if manifest.get("result_rows") != manifest.get("planned_rows"):
        raise ValueError("frozen control ledger is incomplete")
    model_ids = {model["model_id"] for model in models}
    expected_by_context: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in identity_contexts:
        for model_id in model_ids:
            key = (row["sample_id"], row["speaker_id"], model_id, int(row["seed"]))
            expected_by_context[key] = row
    controls = {method: {} for method in expected_methods}
    for line_no, line in enumerate(results_raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        method_id = row.get("method_id")
        if method_id not in controls:
            continue
        key = (row.get("sample_id"), row.get("speaker_id"), row.get("model_id"), row.get("seed"))
        if key not in expected_by_context:
            raise ValueError(f"control row {line_no} has an unexpected sample/model/seed context")
        if key in controls[method_id]:
            raise ValueError(f"control row {line_no} duplicates method/context {method_id}/{key}")
        plan_context = expected_by_context[key]
        if row.get("status") != "PASS":
            raise ValueError(f"frozen {method_id} control failed for {key}; no row replacement is allowed")
        if row.get("natural_pcm_sha256") != plan_context["natural_pcm"]["pcm_sha256"]:
            raise ValueError(f"frozen {method_id} control source PCM differs for {key}")
        if row.get("score_audio_sha256") != plan_context["natural_pcm"]["pcm_sha256"]:
            raise ValueError(f"frozen {method_id} control scored enhanced audio instead of original PCM for {key}")
        if row.get("source_face_sha256") != plan_context["source_face"]["file_sha256"]:
            raise ValueError(f"frozen {method_id} control portrait differs for {key}")
        if row.get("source_video_sha256") != plan_context["source_video"]["file_sha256"]:
            raise ValueError(f"frozen {method_id} control source video differs for {key}")
        if row.get("face_id") != plan_context["face_id"]:
            raise ValueError(f"frozen {method_id} control face ID differs for {key}")
        controls[method_id][key] = row
    expected_keys = set(expected_by_context)
    for method_id, rows in controls.items():
        if set(rows) != expected_keys or len(rows) != 20 * 3 * 3:
            raise ValueError(f"frozen {method_id} control denominator is incomplete: {len(rows)} / {len(expected_keys)}")
    refs: dict[str, dict[tuple[str, str, str, int], dict[str, Any]]] = controls
    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_raw_sha256": sha_bytes(manifest_raw),
        "run_id": manifest.get("run_id"),
        "results_path": str(results_path.resolve()),
        "results_raw_sha256": sha_bytes(results_raw),
        "control_rows": {method_id: len(rows) for method_id, rows in controls.items()},
    }
    return refs, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-dev-plan", type=Path, required=True)
    parser.add_argument("--frontend-roster-plan", type=Path, required=True)
    parser.add_argument("--candidate-cli-config", type=Path, required=True)
    parser.add_argument("--candidate-cli-config-raw-sha256", required=True)
    parser.add_argument("--baseline-results-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-result-manifest", type=Path, required=True)
    parser.add_argument("--stagea-core", type=Path, required=True)
    parser.add_argument("--cross-frontend-runner", type=Path, required=True)
    parser.add_argument("--v3-dispatcher", type=Path, required=True)
    parser.add_argument("--v3-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve(strict=True)
    base, base_raw = load_frozen_plan(args.base_dev_plan, BASE_PLAN_SHA256, "base Dev plan")
    frontend, frontend_raw = load_frozen_plan(args.frontend_roster_plan, FRONTEND_PLAN_SHA256, "frontend roster plan")
    if base.get("schema_version") != "cross-tfg-audio-eval-v1" or frontend.get("schema") not in {
        "cross_tfg_audio_head_fit_plan_v1", "cross_tfg_audio_head_fit_plan_parent_overlay_v1", None
    }:
        raise ValueError("frozen base/frontend plan schemas differ from the qualified inputs")
    if base.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ValueError("frozen Dev render seed family changed")
    if base.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("frozen base plan unexpectedly has confirmation seeds")
    if len(base.get("rows", [])) != 180 or tuple(method["method_id"] for method in base.get("methods", [])) != BASE_METHODS:
        raise ValueError("frozen Stage-A baseline plan row/method roster changed")
    models = frontend.get("models")
    if not isinstance(models, list) or tuple(model.get("model_id") for model in models) != EXPECTED_MODELS:
        raise ValueError("the exact Wav2Lip/MuseTalk/Ditto model roster changed")
    if {model.get("face_id", "primary_face_static_portrait_v1") for model in models} != {"primary_face_static_portrait_v1"}:
        raise ValueError("candidate Dev slate must retain the common static primary portrait")

    core_path = args.stagea_core.resolve(strict=True)
    cross_path = args.cross_frontend_runner.resolve(strict=True)
    dispatcher_path = args.v3_dispatcher.resolve(strict=True)
    cli_path = args.v3_cli.resolve(strict=True)
    if sha_file(core_path) != STAGEA_CORE_SHA256 or sha_file(cross_path) != CROSS_FRONTEND_RUNNER_SHA256:
        raise ValueError("Stage-A core/cross-frontend adapter differs from the frozen qualified sources")
    cli_sha = sha_file(cli_path)
    # The dispatcher embeds the qualified core/adapter hashes and checks the
    # CLI hash supplied by the plan at invocation. It must be pinned here.
    if len(cli_sha) != 64 or not dispatcher_path.is_file():
        raise ValueError("V3 CLI/dispatcher source identity could not be established")
    candidate = _candidate_bundle(
        project_root,
        args.candidate_cli_config.resolve(strict=True),
        args.candidate_cli_config_raw_sha256.lower(),
    )
    config_cli_pin = candidate["config"].get("source_pins", {}).get("cli", {})
    if config_cli_pin.get("path") != "scripts/audio_enhancement/enhance_v3_attempt02.py" or config_cli_pin.get("sha256") != cli_sha:
        raise ValueError("V3 config CLI pin does not match the exact CLI source passed to the candidate dispatcher")

    identity_rows = [row for row in base["rows"] if row.get("method_id") == "identity"]
    contexts = {(row["sample_id"], row["speaker_id"], int(row["seed"])) for row in identity_rows}
    if len(identity_rows) != 60 or len(contexts) != 60:
        raise ValueError("base plan must contain exactly one identity context for each 20 clips × 3 seeds")
    samples = {row["sample_id"] for row in identity_rows}
    speakers = {row["speaker_id"] for row in identity_rows}
    seeds = {int(row["seed"]) for row in identity_rows}
    if len(samples) != 20 or len(speakers) != 10 or seeds != {20261003, 20261004, 20261005}:
        raise ValueError("base identity contexts no longer match the frozen 20-clip/10-speaker/3-seed cohort")
    identity_rows.sort(key=lambda row: (row["sample_id"], int(row["seed"])))
    face_bindings = {canonical(row.get("source_face")) for row in identity_rows}
    video_bindings = {canonical(row.get("source_video")) for row in identity_rows}
    if len(face_bindings) != 1 or len(video_bindings) != 1:
        raise ValueError("baseline identity contexts do not share one frozen source portrait/video")
    for sample_id in samples:
        sample_rows = [row for row in identity_rows if row["sample_id"] == sample_id]
        if len(sample_rows) != 3 or len({row["speaker_id"] for row in sample_rows}) != 1:
            raise ValueError(f"incomplete speaker/seed context for {sample_id}")
        if len({canonical([row["natural_pcm"], row["score_audio"]]) for row in sample_rows}) != 1:
            raise ValueError(f"natural source/score PCM changed across seeds for {sample_id}")

    controls, control_provenance = _baseline_rows(
        args.baseline_results_jsonl.resolve(strict=True),
        args.baseline_result_manifest.resolve(strict=True),
        REQUIRED_BASELINE_METHODS,
        identity_rows,
        models,
    )
    candidate_config_rel = args.candidate_cli_config.resolve(strict=True).relative_to(project_root).as_posix()
    method_config = {
        "method": V3_METHOD_CONFIG,
        "v3_cli_config_path": candidate_config_rel,
        "v3_cli_config_sha256": candidate["config_raw_sha256"],
        "head_schema": candidate["config"]["head_schema"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_training_record_sha256": candidate["training_record_sha256"],
        "candidate_fixed_step_selection_record_sha256": candidate["selection_record_sha256"],
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "peak_policy": "use the exact frozen V3 CLI constructor; reject any configured peak violation without clipping/post-gain",
        "generator_audio_only": True,
        "score_audio": "exact original natural PCM16 source under the frozen native evaluator",
    }
    method = {
        "method_id": V3_METHOD_ID,
        "kind": "learned_audio_head_candidate",
        "config": method_config,
        "config_sha256": sha_bytes(canonical(method_config)),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_record_sha256": candidate["training_record_sha256"],
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "selection_scope": "fixed terminal fit step 120; weights frozen before Dev and not chosen using Dev; confirmation remains closed",
    }

    # Use the 60 exact identity contexts as the row source and expand each
    # across the three pinned model records. Each planned row gets explicit
    # identity/RMS references from the already complete frozen ledger.
    rows: list[dict[str, Any]] = []
    expected_contexts = {(row["sample_id"], row["speaker_id"], int(row["seed"])) for row in identity_rows}
    model_by_id = {model["model_id"]: model for model in models}
    for source in identity_rows:
        for model_id in EXPECTED_MODELS:
            key = (source["sample_id"], source["speaker_id"], model_id, int(source["seed"]))
            identity_ref = controls["identity"][key]
            rms_ref = controls["rms_fixed"][key]
            model = model_by_id[model_id]
            row = copy.deepcopy(source)
            row["model_id"] = model_id
            row["audio_frontend"] = model["audio_frontend"]
            row["model_role"] = model.get("role", "candidate")
            row["method_id"] = V3_METHOD_ID
            row["method_config_sha256"] = method["config_sha256"]
            row["checkpoint_sha256"] = candidate["checkpoint_sha256"]
            row["baseline_reference_row_ids"] = {
                "identity": identity_ref["row_id"],
                "rms_fixed": rms_ref["row_id"],
            }
            row["baseline_reference_run_ids"] = {
                "identity": control_provenance["run_id"],
                "rms_fixed": control_provenance["run_id"],
            }
            row["generation_settings"] = {
                **row["generation_settings"],
                "generator": model_id,
                "input_audio": "V3 step120 candidate drives generator only; exact original natural PCM is muxed/scored",
                "audio_transform": V3_METHOD_ID,
                "v3_cli_config_sha256": candidate["config_raw_sha256"],
                "head_checkpoint_sha256": candidate["checkpoint_sha256"],
                "training_record_sha256": candidate["training_record_sha256"],
                "seed": int(source["seed"]),
            }
            row["row_id"] = (
                f"dev_{row['speaker_id']}_{row['sample_id']}_{model_id}_"
                f"{row['face_id']}_{int(row['seed'])}_{V3_METHOD_ID}"
            )
            rows.append(row)
    if len(rows) != 180 or len({row["row_id"] for row in rows}) != 180:
        raise ValueError("V3 candidate plan must contain exactly 180 unique learned rows")
    if len(expected_contexts) != 20 * 3:
        raise ValueError("candidate source context denominator changed")

    plan = copy.deepcopy(base)
    plan["run_id"] = "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt01"
    plan["methods"] = [method]
    plan["models"] = copy.deepcopy(models)
    plan["rows"] = rows
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["simultaneous_family_ids"] = []
    plan["analysis"]["simultaneous_family_status"] = "NOT_APPLICABLE_CANDIDATE_ONLY_SUBRUN"
    plan["analysis"]["candidate_only_subrun"] = {
        "candidate_method_id": V3_METHOD_ID,
        "fixed_checkpoint_rule": "terminal_optimizer_step_120_from_fit_only_40_endpoint_roster",
        "weight_selection_using_dev": False,
        "baseline_join": "identity and rms_fixed are immutable reference rows pinned by row IDs in the full common-support control ledger",
    }
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "v3_fixed_step120_candidate_only_development_render",
        "scope": "development only; 20 fixed Dev clips × 3 frozen seeds × 3 qualified native frontends",
        "planned_rows": 180,
        "candidate_only": True,
        "fresh_baseline_rerender": False,
        "baseline_controls": control_provenance,
        "baseline_reference_method_ids": list(REQUIRED_BASELINE_METHODS),
        "baseline_join_policy": "Use explicit per-row identity/rms_fixed references; verify exact original PCM, source face/video, model, seed, score model/preprocess, and native query clock before paired support is computed.",
        "v3_cli_sha256": cli_sha,
        "v3_dispatcher_sha256": sha_file(dispatcher_path),
        "stagea_core_sha256": STAGEA_CORE_SHA256,
        "stagea_runner_sha256": STAGEA_CORE_SHA256,
        "cross_frontend_runner_sha256": CROSS_FRONTEND_RUNNER_SHA256,
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_training_record_sha256": candidate["training_record_sha256"],
        "candidate_selection_record_sha256": candidate["selection_record_sha256"],
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "development_weights_selected": False,
    }
    plan["protocol"]["generator"]["stagea_runner_sha256"] = STAGEA_CORE_SHA256
    plan["protocol"]["generator"]["cross_frontend_runner_sha256"] = CROSS_FRONTEND_RUNNER_SHA256
    for model in plan["models"]:
        generator = model.get("generator")
        if isinstance(generator, dict):
            generator["runner_sha256"] = STAGEA_CORE_SHA256
    plan["source_reference_plans"] = {
        "base_dev_plan": {"path": str(args.base_dev_plan.resolve()), "raw_sha256": sha_bytes(base_raw), "plan_sha256": base["plan_sha256"]},
        "frontend_roster_plan": {"path": str(args.frontend_roster_plan.resolve()), "raw_sha256": sha_bytes(frontend_raw), "plan_sha256": frontend["plan_sha256"], "model_ids": list(EXPECTED_MODELS)},
        "v3_cli_config": {"path": candidate_config_rel, "raw_sha256": candidate["config_raw_sha256"], "checkpoint_sha256": candidate["checkpoint_sha256"], "training_record_sha256": candidate["training_record_sha256"], "selection_record_sha256": candidate["selection_record_sha256"]},
        "v3_cli": {"path": "scripts/audio_enhancement/enhance_v3_attempt02.py", "raw_sha256": cli_sha},
        "v3_candidate_dispatcher": {"path": str(dispatcher_path), "raw_sha256": sha_file(dispatcher_path)},
        "stagea_core": {"path": str(core_path), "raw_sha256": sha_file(core_path)},
        "cross_frontend_adapter_runner": {"path": str(cross_path), "raw_sha256": sha_file(cross_path)},
        "control_result_ledger": control_provenance,
    }
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing plan output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    plan_raw = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    write_new(args.output_dir / "plan.json", plan_raw)
    write_new(args.output_dir / "planned_rows.jsonl", rows_raw)
    manifest = {
        "schema": "cross_tfg_v3_candidate_only_dev_plan_v1",
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": sha_bytes(plan_raw),
        "planned_rows_sha256": sha_bytes(rows_raw),
        "planned_rows": len(rows),
        "method_counts": {V3_METHOD_ID: len(rows)},
        "model_counts": {model_id: sum(row["model_id"] == model_id for row in rows) for model_id in EXPECTED_MODELS},
        "sample_count": len(samples),
        "speaker_count": len(speakers),
        "render_seed_family": sorted(seeds),
        "fixed_optimizer_steps": FIXED_TRAINING_STEPS,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "identity_control_rows": 180,
        "rms_fixed_control_rows": 180,
        "fresh_baseline_rerender": False,
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "development_weights_selected": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    write_new(args.output_dir / "plan_manifest.json", (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": "FROZEN", "run_id": plan["run_id"], "plan_raw_sha256": sha_file(args.output_dir / "plan.json"),
                      "planned_rows_sha256": sha_file(args.output_dir / "planned_rows.jsonl"), "rows": 180,
                      "models": list(EXPECTED_MODELS), "checkpoint_sha256": candidate["checkpoint_sha256"],
                      "identity_control_rows": 180, "rms_fixed_control_rows": 180,
                      "confirm_audio_read": False, "development_weights_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
