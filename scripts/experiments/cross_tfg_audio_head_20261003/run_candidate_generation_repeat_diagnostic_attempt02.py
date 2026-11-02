#!/usr/bin/env python3
"""Prepare or execute the ADR018 three-backend candidate repeat diagnostic.

This diagnostic reruns the already frozen step-120 method from the original
natural PCM for one predeclared candidate row per backend, twice per row.  Its
outputs live outside the candidate180 and joined540 roots and can never add
primary rows or choose a method.  The default mode is metadata-only; renderer
execution requires the separate --execute-repeat-diagnostic flag and a fresh,
exactly pinned preflight receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import stat
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT_DEFAULT = "/root/tts-research-system-20261001"
BASE_REL = "scripts/experiments/cross_tfg_audio_head_20261003"
CORE_REL = BASE_REL + "/run_stagea_dev_attempt06.py"
CORE_BYTES = 79_475
CORE_SHA256 = "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"
FRONTEND_REL = BASE_REL + "/run_cross_frontend_dev_attempt09.py"
FRONTEND_BYTES = 23_249
FRONTEND_SHA256 = "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"
CANDIDATE_DISPATCHER_REL = BASE_REL + "/run_cross_frontend_v3_candidate_attempt07.py"
CANDIDATE_DISPATCHER_BYTES = 12_824
CANDIDATE_DISPATCHER_SHA256 = "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3"
CANDIDATE_LAUNCHER_REL = BASE_REL + "/run_v3_step120_candidate_samehost_attempt13.py"
CANDIDATE_LAUNCHER_BYTES = 35_309
CANDIDATE_LAUNCHER_SHA256 = "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"
GUARD_REL = BASE_REL + "/run_v3_samehost_dev360_controls_attempt18.py"
GUARD_BYTES = 78_172
GUARD_SHA256 = "30381ee8deb6d0e7880379fc91a042e1e2bb3cca375dd05423319259fd9e8756"
DIAGNOSTIC_MANIFEST_REL = (
    "cross_tfg_audio_head_20261003/job_control/dev540_candidate_generation_repeat_diagnostic_attempt01/"
    "inputs/candidate_generation_repeat_diagnostic_manifest_attempt02.json"
)
DIAGNOSTIC_MANIFEST_BYTES = 7_926
DIAGNOSTIC_MANIFEST_SHA256 = "b65432f786e93e7c3c26fbdcd8d1038129e8ee1c6a3d7dd5c132dabb2d61d9d5"
DIAGNOSTIC_MANIFEST_SELF_SHA256 = "6e0fc4dcf3b8111fe908eb41487a3621832997397e411db228921c845b3386fe"
EXPECTED_OUTPUT_ROOT = (
    "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/"
    "controls360_repair_attempt10/candidate_generation_repeat_diagnostic_attempt02"
)
EXPECTED_METHOD = "shared_recipient_transition_head_v3_fit40_step120"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
RESULT_SCHEMA = "cross_tfg_v3_candidate_generation_repeat_diagnostic_result_v1"
PREFLIGHT_SCHEMA = "cross_tfg_v3_candidate_generation_repeat_diagnostic_preflight_v1"
MAX_JSON_BYTES = 32 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def raw_file(path: Path, label: str, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical.is_symlink() or getattr(lexical, "is_junction", lambda: False)():
        raise ValueError(f"{label} must not be a symlink or junction: {lexical}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > max_bytes:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    with lexical.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    after = lexical.stat(follow_symlinks=False)
    if (len(raw) != before.st_size or len(raw) > max_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
               (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError(f"{label} changed while being read: {lexical}")
    return raw, {"path": str(lexical.resolve(strict=True)), "bytes": len(raw),
                 "sha256": sha_bytes(raw), "device": int(after.st_dev),
                 "inode": int(after.st_ino)}


def read_json_exact(path: Path, label: str, expected_sha: str,
                    expected_bytes: int | None = None, max_bytes: int = MAX_JSON_BYTES
                    ) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw, ref = raw_file(path, label, max_bytes)
    if ref["sha256"] != expected_sha.lower() or (
            expected_bytes is not None and ref["bytes"] != expected_bytes):
        raise ValueError(f"{label} raw byte identity differs from its pin")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, raw, ref


def create_json(path: Path, value: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    raw = pretty(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw, {"path": str(path.resolve(strict=True)), "bytes": len(raw),
                 "sha256": sha_bytes(raw)}


def replace_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = pretty(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def load_pinned_module(project_root: Path, relative: str, expected_bytes: int,
                       expected_sha: str, module_name: str) -> tuple[Any, dict[str, Any]]:
    path = project_root / relative
    raw, ref = raw_file(path, relative, 64 * 1024 * 1024)
    if ref["bytes"] != expected_bytes or ref["sha256"] != expected_sha:
        raise ValueError(f"pinned source changed: {relative}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned Python source: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, {"path": str(path.resolve(strict=True)), "bytes": len(raw),
                    "sha256": sha_bytes(raw)}


def _payload_self_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def _assert_manifest(manifest: Mapping[str, Any], raw: bytes) -> None:
    if (len(raw) != DIAGNOSTIC_MANIFEST_BYTES
            or sha_bytes(raw) != DIAGNOSTIC_MANIFEST_SHA256
            or manifest.get("schema") != "cross_tfg_v3_candidate180_generation_repeat_diagnostic_manifest_v1"
            or manifest.get("status") != "SOURCE_ONLY_PREDECLARED_NOT_RUN"
            or manifest.get("receipt_sha256") != DIAGNOSTIC_MANIFEST_SELF_SHA256
            or manifest.get("receipt_sha256") != _payload_self_sha(manifest, "receipt_sha256")
            or manifest.get("rendering_performed") is not False
            or manifest.get("scoring_performed") is not False
            or manifest.get("method_selected") is not False
            or manifest.get("primary_rows_added") != 0
            or manifest.get("row_count") != 3
            or manifest.get("confirm_opened") is not False):
        raise ValueError("ADR018 immutable predeclared repeat manifest identity or scope is invalid")
    rows = manifest.get("planned_repeats")
    shared = manifest.get("shared_context")
    if (not isinstance(shared, Mapping)
            or shared.get("sample_id") != "BAC009S0005W0121"
            or shared.get("speaker_id") != "S0005"
            or shared.get("seed") != 20261003
            or shared.get("natural_pcm_sha256") != "bce584f82bad71ea2501a976a512471e0dacbdc24a4b8c047e6981c747aa2a8e"
            or not isinstance(rows, list) or len(rows) != 3
            or tuple(item.get("model_id") for item in rows) != MODELS
            or any(item.get("status") != "PREDECLARED_NOT_RUN"
                   or item.get("future_output_paths") is not None
                   or item.get("future_output_hashes") is not None
                   or item.get("sample_id") != shared.get("sample_id")
                   or item.get("speaker_id") != shared.get("speaker_id")
                   or item.get("seed") != shared.get("seed")
                   or item.get("natural_pcm", {}).get("pcm_sha256") != shared.get("natural_pcm_sha256")
                   or item.get("fps") != {"num": 25, "den": 1} for item in rows)):
        raise ValueError("repeat manifest does not contain the exact ordered three-row predeclaration")


def _expected_manifest_row(plan: Mapping[str, Any], diagnostic_item: Mapping[str, Any]
                           ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = plan.get("rows")
    if not isinstance(rows, list):
        raise ValueError("candidate plan rows are absent")
    matching = [row for row in rows if row.get("row_id") == diagnostic_item.get("row_id")]
    if len(matching) != 1:
        raise ValueError("predeclared repeat row is absent or duplicated in the candidate plan")
    row = matching[0]
    models = [item for item in plan.get("models", []) if item.get("model_id") == row.get("model_id")]
    methods = [item for item in plan.get("methods", []) if item.get("method_id") == row.get("method_id")]
    if len(models) != 1 or len(methods) != 1:
        raise ValueError("predeclared row lacks one exact model or method descriptor")
    model, method = models[0], methods[0]
    identity = {
        "row_id": row.get("row_id"), "sample_id": row.get("sample_id"),
        "speaker_id": row.get("speaker_id"), "model_id": row.get("model_id"),
        "audio_frontend": row.get("audio_frontend"), "seed": row.get("seed"),
        "method_id": row.get("method_id"),
    }
    expected = {
        "row_id": diagnostic_item.get("row_id"),
        "sample_id": diagnostic_item.get("sample_id"),
        "speaker_id": diagnostic_item.get("speaker_id"),
        "model_id": diagnostic_item.get("model_id"),
        "audio_frontend": diagnostic_item.get("audio_frontend"),
        "seed": diagnostic_item.get("seed"),
        "method_id": diagnostic_item.get("method_id"),
    }
    if identity != expected or row.get("method_id") != EXPECTED_METHOD:
        raise ValueError("predeclared diagnostic identity does not exactly match its frozen candidate row")
    natural = row.get("natural_pcm")
    face = row.get("source_face")
    video = row.get("source_video")
    if (not isinstance(natural, dict) or not isinstance(face, dict) or not isinstance(video, dict)
            or natural.get("file_sha256") != diagnostic_item.get("natural_pcm", {}).get("file_sha256")
            or natural.get("pcm_sha256") != diagnostic_item.get("natural_pcm", {}).get("pcm_sha256")
            or natural.get("sample_count") != diagnostic_item.get("natural_pcm", {}).get("sample_count")
            or natural.get("sample_rate_hz") != diagnostic_item.get("natural_pcm", {}).get("sample_rate_hz")
            or natural.get("channels") != diagnostic_item.get("natural_pcm", {}).get("channels")
            or face.get("file_sha256") != diagnostic_item.get("source_face_sha256")
            or video.get("file_sha256") != diagnostic_item.get("source_video_sha256")
            or row.get("checkpoint_sha256") != diagnostic_item.get("checkpoint_sha256")
            or row.get("method_config_sha256") != diagnostic_item.get("method_config_sha256")
            or method.get("config_sha256") != diagnostic_item.get("method_config_sha256")
            or model.get("audio_frontend") != diagnostic_item.get("audio_frontend")
            or (model.get("fps_num"), model.get("fps_den")) != (25, 1)):
        raise ValueError("diagnostic natural PCM/source/model/method fields differ from the frozen candidate plan")
    generator = model.get("generator")
    renderer = diagnostic_item.get("renderer_binding")
    if not isinstance(generator, dict) or not isinstance(renderer, dict):
        raise ValueError("diagnostic row lacks its generator descriptor or renderer identity")
    if row["model_id"] == MODELS[0]:
        frame_binding = row.get("wav2lip_native_frame_binding")
        if (not isinstance(frame_binding, dict)
                or renderer.get("entry_path") != generator.get("native_entry_path")
                or renderer.get("entry_sha256") != generator.get("native_entry_sha256")
                or renderer.get("expected_frame_count") != frame_binding.get("expected_inference_frame_count")
                or renderer.get("frame_probe_raw_sha256") != frame_binding.get("probe_raw_sha256")
                or renderer.get("runtime_compat_sha256") != generator.get("runtime_compat_sitecustomize_sha256")
                or renderer.get("strict_bootstrap_sha256") != generator.get("strict_bootstrap_sha256")):
            raise ValueError("Wav2Lip predeclared entry/frame/runtime pins differ from the candidate row")
    else:
        if (renderer.get("adapter_path") != generator.get("adapter_path")
                or renderer.get("adapter_sha256") != generator.get("adapter_sha256")
                or renderer.get("entry_path") != generator.get("render_entry_path")
                or renderer.get("entry_sha256") != generator.get("renderer_sha256")
                or renderer.get("strict_bootstrap_sha256") != generator.get("strict_bootstrap_sha256")):
            raise ValueError("native adapter or renderer pins differ from the candidate descriptor")
    return row, model, method


def _check_plan_and_binding(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError("project root is absent")
    sys.path.insert(0, str(project_root))
    guard_source = project_root / GUARD_REL
    guard_raw, guard_source_ref = raw_file(guard_source, "samehost candidate guard", 16 * 1024 * 1024)
    if guard_source_ref["bytes"] != GUARD_BYTES or guard_source_ref["sha256"] != GUARD_SHA256:
        raise ValueError("samehost candidate guard source differs from its frozen attempt18 pin")
    # The candidate launcher performs the comprehensive 180-row and source-closure validation.
    candidate, candidate_ref = load_pinned_module(
        project_root, CANDIDATE_LAUNCHER_REL, CANDIDATE_LAUNCHER_BYTES,
        CANDIDATE_LAUNCHER_SHA256, "adr018_candidate_launcher")
    core, core_ref = load_pinned_module(project_root, CORE_REL, CORE_BYTES, CORE_SHA256,
                                        "adr018_stagea_core")
    frontend, frontend_ref = load_pinned_module(project_root, FRONTEND_REL, FRONTEND_BYTES,
                                                FRONTEND_SHA256, "adr018_cross_frontend")
    dispatcher, dispatcher_ref = load_pinned_module(
        project_root, CANDIDATE_DISPATCHER_REL, CANDIDATE_DISPATCHER_BYTES,
        CANDIDATE_DISPATCHER_SHA256, "adr018_step120_candidate_dispatcher")
    if (frontend.CORE_DEFAULT != str(project_root / CORE_REL)
            or frontend.CORE_SHA256_DEFAULT != CORE_SHA256):
        raise ValueError("cross-frontend wrapper does not bind the exact Stage-A core source")
    if (dispatcher.CORE_SHA256 != CORE_SHA256
            or dispatcher.ADAPTER_RUNNER_SHA256 != FRONTEND_SHA256
            or dispatcher.TRANSFORM_METHOD != "v3_parameter_checkpoint"):
        raise ValueError("step-120 dispatcher does not bind the exact core, adapter, and candidate method route")
    candidate_args = argparse.Namespace(
        project_root=project_root,
        candidate_plan=args.candidate_plan,
        candidate_plan_raw_sha256=args.candidate_plan_raw_sha256,
        planned_rows=args.planned_rows,
        planned_rows_raw_sha256=args.planned_rows_raw_sha256,
        plan_manifest=args.plan_manifest,
        plan_manifest_raw_sha256=args.plan_manifest_raw_sha256,
    )
    guard, _guard_raw = candidate.load_guard(project_root)
    runtime, runtime_raw, runtime_ref = read_json_exact(
        args.runtime_binding, "samehost runtime binding", args.runtime_binding_raw_sha256,
        args.runtime_binding_bytes, 16 * 1024 * 1024)
    plan, plan_raw, plan_manifest, plan_manifest_raw, rows_raw = candidate.validate_candidate_plan(
        candidate_args, project_root, guard, sha_bytes(runtime_raw))
    if (runtime.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1"
            or runtime.get("status") != "PASS"
            or runtime.get("binding_sha256") != _payload_self_sha(runtime, "binding_sha256")
            or plan.get("execution", {}).get("samehost_runtime_binding", {}).get("raw_sha256") != sha_bytes(runtime_raw)
            or plan.get("execution", {}).get("samehost_runtime_binding", {}).get("path") != str(args.runtime_binding)):
        raise ValueError("candidate plan and exact samehost binding raw/self identities disagree")
    source_refs = plan.get("source_reference_plans")
    if not isinstance(source_refs, Mapping):
        raise ValueError("candidate plan lacks its source reference map")
    dispatcher_ref_from_plan = source_refs.get("v3_candidate_dispatcher")
    if dispatcher_ref_from_plan != {
            "path": CANDIDATE_DISPATCHER_REL,
            "bytes": CANDIDATE_DISPATCHER_BYTES,
            "raw_sha256": CANDIDATE_DISPATCHER_SHA256}:
        raise ValueError("candidate plan does not pin the exact V3 candidate dispatcher")
    cli_ref = source_refs.get("v3_cli")
    config_ref = source_refs.get("v3_cli_config")
    if (not isinstance(cli_ref, Mapping) or not isinstance(config_ref, Mapping)
            or not isinstance(cli_ref.get("path"), str) or not isinstance(cli_ref.get("raw_sha256"), str)
            or not isinstance(config_ref.get("path"), str) or not isinstance(config_ref.get("raw_sha256"), str)):
        raise ValueError("candidate plan lacks pinned V3 CLI/configuration references")
    cli_raw, cli_source_ref = raw_file(project_root / cli_ref["path"], "pinned V3 CLI source", 16 * 1024 * 1024)
    if sha_bytes(cli_raw) != cli_ref["raw_sha256"]:
        raise ValueError("pinned V3 CLI source raw SHA differs from the candidate plan")
    config_raw, config_source_ref = raw_file(project_root / config_ref["path"], "pinned V3 CLI configuration", 16 * 1024 * 1024)
    if (sha_bytes(config_raw) != config_ref["raw_sha256"]
            or config_ref.get("checkpoint_sha256") !=
               plan.get("execution", {}).get("candidate_checkpoint_sha256")):
        raise ValueError("pinned V3 CLI config/checkpoint source reference differs from the candidate plan")
    manifest, diagnostic_raw, diagnostic_ref = read_json_exact(
        args.diagnostic_manifest, "ADR018 repeat manifest", DIAGNOSTIC_MANIFEST_SHA256,
        DIAGNOSTIC_MANIFEST_BYTES, 2 * 1024 * 1024)
    _assert_manifest(manifest, diagnostic_raw)
    if Path(args.output_root).is_symlink() or Path(args.output_root).resolve(strict=False) != Path(EXPECTED_OUTPUT_ROOT):
        raise ValueError("repeat diagnostic output root differs from its fixed fresh /dev/shm namespace")
    expected_namespace = Path(runtime["execution_bindings"]["output_root"]).resolve(strict=True)
    candidate_output = guard.subrun_output_root(runtime, "candidate180")
    if (candidate_output.parent != expected_namespace
            or Path(args.output_root).parent.resolve(strict=True) != expected_namespace
            or Path(args.output_root) == candidate_output):
        raise ValueError("diagnostic output must be a sibling of candidate180 in the bound /dev/shm namespace")
    if Path(args.output_root).exists():
        raise FileExistsError("diagnostic output leaf already exists; refusing to reuse prior output")
    # The exact candidate row plan is cross-checked to all three predeclared diagnostic identities.
    row_models: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for diag_item in manifest["planned_repeats"]:
        row, model, method = _expected_manifest_row(plan, diag_item)
        method_config = method.get("config")
        if (not isinstance(method_config, Mapping)
                or method_config.get("method") != dispatcher.TRANSFORM_METHOD
                or method_config.get("v3_cli_config_sha256") != config_ref.get("raw_sha256")
                or method_config.get("v3_cli_config_path") != config_ref.get("path")):
            raise ValueError("candidate method config is not the pinned V3 step-120 transform route")
        if model["generator"] != runtime["backends"][row["model_id"]]["generator"]:
            raise ValueError("candidate plan generator descriptor differs from the measured samehost runtime binding")
        backend = runtime["backends"][row["model_id"]]
        model_binding = backend.get("model_binding")
        if not isinstance(model_binding, dict) or any(
                model.get(key) != value for key, value in model_binding.items()):
            raise ValueError("candidate plan model/scorer identity differs from its actual runtime backend binding")
        row_models.append((row, model, method))
    if tuple(model["model_id"] for _row, model, _method in row_models) != MODELS:
        raise ValueError("predeclared diagnostic model order is not the frozen three-backend order")
    if plan.get("generation_repeat_qc") != []:
        raise ValueError("candidate's frozen repeat-QC policy changed; the separate ADR018 diagnostic cannot alter it")
    helper = guard.load_resource_helper(project_root)
    live, identities = guard.validate_bindings(project_root, plan, runtime, helper,
                                                candidate_output, capacity_stage="candidate180")
    guard._assert_output_namespace_disjoint(plan, runtime, Path(args.output_root))

    # Validate natural, face, and source-video bytes against the exact row refs without rendering.
    inputs: list[dict[str, Any]] = []
    for row, _model, _method in row_models:
        natural_path, samples, natural_ref = core._reference(row["natural_pcm"])
        if (natural_ref["sample_rate_hz"] != 16000 or natural_ref["channels"] != 1
                or natural_ref["sample_count"] != row["natural_pcm"].get("sample_count")):
            raise ValueError(f"natural PCM rate/channel/count differs from the frozen candidate row: {row['row_id']}")
        file_records = {"natural_pcm": {"path": str(natural_path),
                                        "file_sha256": natural_ref["file_sha256"],
                                        "pcm_sha256": natural_ref["pcm_sha256"],
                                        "sample_count": natural_ref["sample_count"],
                                        "sample_rate_hz": 16000, "channels": 1}}
        for key in ("source_face", "source_video"):
            ref = row[key]
            path = Path(str(ref.get("path", "")))
            expected = str(ref.get("file_sha256", "")).lower()
            if not path.is_absolute() or len(expected) != 64:
                raise ValueError(f"candidate row {key} path/hash is incomplete: {row['row_id']}")
            resolved, raw = guard.safe_regular(path, f"{row['row_id']} {key}", 2 * 1024 * 1024 * 1024)
            if sha_bytes(raw) != expected:
                raise ValueError(f"candidate row {key} raw bytes differ from the frozen row: {row['row_id']}")
            if expected != next(item for item in manifest["planned_repeats"]
                                if item["row_id"] == row["row_id"])[
                                    "source_face_sha256" if key == "source_face" else "source_video_sha256"]:
                raise ValueError(f"candidate row {key} differs from immutable diagnostic manifest")
            file_records[key] = {"path": str(resolved), "bytes": len(raw), "sha256": sha_bytes(raw)}
        inputs.append({"row_id": row["row_id"], "natural_pcm": file_records["natural_pcm"],
                       "source_face": file_records["source_face"],
                       "source_video": file_records["source_video"]})
        del samples
    return {
        "project_root": str(project_root), "plan": plan, "plan_raw": plan_raw,
        "plan_raw_ref": {"path": str(args.candidate_plan), "bytes": len(plan_raw), "sha256": sha_bytes(plan_raw)},
        "planned_rows_raw_ref": {"path": str(args.planned_rows), "bytes": len(rows_raw), "sha256": sha_bytes(rows_raw)},
        "plan_manifest_raw_ref": {"path": str(args.plan_manifest), "bytes": len(plan_manifest_raw), "sha256": sha_bytes(plan_manifest_raw)},
        "plan_manifest": plan_manifest, "runtime": runtime, "runtime_raw": runtime_raw,
        "runtime_raw_ref": runtime_ref, "diagnostic_manifest": manifest,
        "diagnostic_manifest_raw_ref": diagnostic_ref, "candidate_output_root": str(candidate_output),
        "diagnostic_output_root": str(Path(args.output_root)), "row_models": row_models,
        "inputs": inputs, "guard": guard, "core": core, "frontend": frontend,
        "candidate_dispatcher": dispatcher,
        "v3_cli_source_ref": cli_source_ref, "v3_cli_config_source_ref": config_source_ref,
        "source_pins": {"stagea_core": core_ref, "cross_frontend": frontend_ref,
                        "candidate_launcher": candidate_ref,
                        "candidate_v3_dispatcher": dispatcher_ref,
                        "v3_cli": cli_source_ref, "v3_cli_config": config_source_ref,
                        "samehost_guard": guard_source_ref},
        "live_resource_probe": live, "runtime_identity": identities,
    }


def _resource_environment(runtime_ref: Mapping[str, Any]) -> None:
    for key in tuple(os.environ):
        if key.startswith("CROSS_TFG_"):
            os.environ.pop(key, None)
    os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(runtime_ref["path"])
    os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = str(runtime_ref["sha256"])
    os.environ["CROSS_TFG_CAPACITY_STAGE"] = "candidate180"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"


def _array_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "bytes": len(raw), "array_sha256": sha_bytes(raw)}


def _npz_array_identities(path: Path, expected_arrays: Mapping[str, Any] | None = None
                          ) -> dict[str, Any]:
    import numpy as np

    raw, ref = raw_file(path, "official raw score NPZ", 512 * 1024 * 1024)
    identities: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files)
        for key in names:
            array = np.ascontiguousarray(archive[key])
            identities[key] = _array_identity(array)
            if expected_arrays is not None:
                if key not in expected_arrays or not np.array_equal(array, expected_arrays[key]):
                    raise ValueError(f"score NPZ array does not match the official in-memory array: {key}")
        if expected_arrays is not None and set(names) != set(expected_arrays):
            raise ValueError("score NPZ field roster differs from the full official score matrix payload")
    return {**ref, "npz_file_sha256": sha_bytes(raw), "arrays": identities}


def _read_native_receipts(result: Mapping[str, Any], row: Mapping[str, Any], model: Mapping[str, Any],
                          frontend: Any, core: Any, receipt_dir: Path,
                          guard: Any) -> dict[str, Any]:
    files = sorted(receipt_dir.glob("*.json"))
    model_id = row["model_id"]
    generator = model["generator"]
    expected_python = Path(str(generator["python"])).resolve(strict=True)
    if model_id == MODELS[0]:
        native_entry = Path(str(generator["native_entry_path"])).resolve(strict=True)
        expected_receipts = {native_entry: (True, None)}
        if len(files) != 1 or not files[0].stem.isdecimal():
            raise ValueError(f"expected one PID-named Wav2Lip bootstrap receipt, found {len(files)}")
    else:
        native_entry = Path(str(generator["render_entry_path"])).resolve(strict=True)
        renderer_alias = Path(str(generator["renderer_path"])).resolve(strict=True)
        shim_entry = Path(str(generator["adapter_path"])).resolve(strict=True)
        if (renderer_alias != native_entry or shim_entry == native_entry or len(files) != 2
                or {item.name for item in files} != {
                    "strict_startup_native.json", "strict_startup_shim.json"}):
            raise ValueError(f"expected the exact native+shim bootstrap receipt pair: {row['row_id']}")
        expected_receipts = {native_entry: (True, "strict_startup_native.json"),
                             shim_entry: (False, "strict_startup_shim.json")}
    if set(expected_receipts) == set():
        raise ValueError("strict renderer receipt descriptor is empty")

    loaded: dict[str, dict[str, Any]] = {}
    for path in files:
        raw, ref = raw_file(path, "strict renderer bootstrap receipt", 256 * 1024)
        bootstrap = json.loads(raw.decode("utf-8"))
        try:
            entry_path = Path(str(bootstrap.get("entry_path"))).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"strict bootstrap receipt entry path is not a resolved file: {path.name}") from exc
        match_spec = expected_receipts.get(entry_path)
        if match_spec is None:
            raise ValueError(f"unexpected strict bootstrap entry in receipt directory: {path.name}")
        should_match_backend, expected_filename = match_spec
        pid = bootstrap.get("pid")
        if (expected_filename is not None and path.name != expected_filename
                or type(pid) is not int or pid <= 0
                or (model_id == MODELS[0] and path.stem != str(pid))
                or bootstrap.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
                or bootstrap.get("status") != "PASS"
                or bootstrap.get("receipt_sha256") != _payload_self_sha(bootstrap, "receipt_sha256")
                or bootstrap.get("bootstrap_source_sha256") != guard.STRICT_BOOTSTRAP_SHA256
                or bootstrap.get("policy") != guard.STRICT_BOOTSTRAP_POLICY
                or bootstrap.get("policy_payload_sha256") != sha_bytes(canonical(bootstrap.get("policy")))
                or bootstrap.get("cuda_available") is not True
                or (bootstrap.get("row_id"), bootstrap.get("model_id"), bootstrap.get("audio_frontend"), bootstrap.get("seed"))
                    != (row["row_id"], model_id, row["audio_frontend"], row["seed"])
                or bootstrap.get("expected_backend_entry") != str(native_entry)
                or bootstrap.get("entry_path") != str(entry_path)
                or bootstrap.get("entry_matches_backend") is not should_match_backend
                or bootstrap.get("python_executable") != str(expected_python)):
            raise ValueError(f"strict bootstrap receipt differs from its exact planned process: {row['row_id']}:{path.name}")
        if entry_path in loaded:
            raise ValueError(f"duplicate strict bootstrap receipt for entry: {entry_path}")
        ref["receipt_sha256"] = bootstrap["receipt_sha256"]
        loaded[entry_path] = {"ref": ref, "receipt": bootstrap}

    if set(loaded) != set(expected_receipts):
        raise ValueError(f"strict bootstrap receipt entries are incomplete: {row['row_id']}")
    renderer_record = loaded[native_entry]
    bootstrap = renderer_record["receipt"]
    if model_id != MODELS[0]:
        shim_record = loaded[shim_entry]
        shim = shim_record["receipt"]
        if (type(shim.get("ppid")) is not int or type(bootstrap.get("ppid")) is not int
                or bootstrap["ppid"] != shim["pid"]
                or shim.get("policy") != bootstrap.get("policy")
                or shim.get("policy_payload_sha256") != bootstrap.get("policy_payload_sha256")
                or shim.get("bootstrap_source_sha256") != bootstrap.get("bootstrap_source_sha256")):
            raise ValueError(f"native renderer process is not the strict shim child for row {row['row_id']}")
    native: dict[str, Any] = {"renderer_startup_receipt": {**renderer_record["ref"],
                       "entry_path": bootstrap["entry_path"],
                       "python_executable": bootstrap["python_executable"],
                       "policy_payload_sha256": bootstrap["policy_payload_sha256"]}}
    if model_id != MODELS[0]:
        shim_bootstrap = loaded[shim_entry]["receipt"]
        native["shim_startup_receipt"] = {**loaded[shim_entry]["ref"],
                       "entry_path": shim_bootstrap["entry_path"],
                       "python_executable": shim_bootstrap["python_executable"],
                       "policy_payload_sha256": shim_bootstrap["policy_payload_sha256"],
                       "pid": shim_bootstrap["pid"], "ppid": shim_bootstrap["ppid"]}
    generation = result["render"]["generation"]
    if model_id == MODELS[0]:
        compat = generation.get("wav2lip_runtime_compatibility_receipts")
        if not isinstance(compat, Mapping) or set(compat) != {
                "constructor_policy", "actual_forward_policy", "prewrite_pixel_arrays"}:
            raise ValueError("Wav2Lip generation omitted one or more strict runtime receipts")
        records = {}
        for name, schema, status in (
            ("constructor_policy", "cross_tfg_w2l_facealignment_constructor_policy_receipt_v1", "PASS_CONSTRUCTOR_RETURN_RESTORED"),
            ("actual_forward_policy", "cross_tfg_w2l_actual_forward_policy_receipt_v1", "PASS_ACTUAL_FORWARD_ENTRIES"),
            ("prewrite_pixel_arrays", "cross_tfg_w2l_prewrite_pixel_array_receipt_v1", "PASS"),
        ):
            record_ref = compat[name]
            record, record_file_ref = frontend._read_w2l_receipt(Path(record_ref["path"]), schema)
            if (record_file_ref.get("bytes") != record_ref.get("bytes")
                    or record_file_ref.get("sha256") != record_ref.get("sha256")
                    or record.get("status") != status
                    or (record.get("row_id"), record.get("model_id"), record.get("seed"))
                        != (row["row_id"], model_id, row["seed"])):
                raise ValueError(f"Wav2Lip {name} receipt identity mismatch")
            evidence_keys = {
                "constructor_policy": ("policy_before_constructor", "policy_at_constructor_return",
                                        "expected_after_constructor", "policy_after_restore",
                                        "changed_policy_fields", "benchmark_override_observed",
                                        "policy_restored_after_constructor", "sfd_network_fallback_prohibited"),
                "actual_forward_policy": ("policy_sha256", "policy_payload_sha256",
                                           "sfd_network_fallback_prohibited",
                                           "all_observed_actual_forward_entries_match_strict_policy",
                                           "forward_entry_observations", "sfd_forward_count",
                                           "wav2lip_generator_forward_count"),
                "prewrite_pixel_arrays": ("expected_frame_count", "frame_count",
                                          "frame_pixel_array_sha256s", "ordered_frame_hashes_sha256",
                                          "raw_intermediate_avi"),
            }[name]
            records[name] = {**record_file_ref, "receipt_sha256": record.get("receipt_sha256"),
                             "status": record.get("status"), "schema": record.get("schema"),
                             "evidence": {key: record.get(key) for key in evidence_keys}}
        forward, _ = frontend._read_w2l_receipt(
            Path(compat["actual_forward_policy"]["path"]), "cross_tfg_w2l_actual_forward_policy_receipt_v1")
        prewrite, _ = frontend._read_w2l_receipt(
            Path(compat["prewrite_pixel_arrays"]["path"]), "cross_tfg_w2l_prewrite_pixel_array_receipt_v1")
        expected_frames = row["wav2lip_native_frame_binding"]["expected_inference_frame_count"]
        intermediate = prewrite.get("raw_intermediate_avi")
        if (not isinstance(intermediate, Mapping) or not isinstance(intermediate.get("path"), str)
                or type(intermediate.get("bytes")) is not int or intermediate.get("bytes") <= 0
                or not isinstance(intermediate.get("sha256"), str)):
            raise ValueError("Wav2Lip prewrite receipt lacks its exact intermediate AVI identity")
        intermediate_raw, intermediate_ref = raw_file(
            Path(intermediate["path"]), "Wav2Lip intermediate AVI", 2 * 1024 * 1024 * 1024)
        if (len(intermediate_raw) != intermediate["bytes"]
                or intermediate_ref["sha256"] != intermediate["sha256"]):
            raise ValueError("Wav2Lip intermediate AVI bytes differ from the prewrite receipt")
        if (forward.get("wav2lip_generator_forward_count", 0) <= 0
                or forward.get("sfd_forward_count", 0) <= 0
                or forward.get("sfd_network_fallback_prohibited") is not True
                or forward.get("all_observed_actual_forward_entries_match_strict_policy") is not True
                or forward.get("policy_sha256") != frontend.W2L_POLICY_CONTRACT_SHA256
                or forward.get("policy_payload_sha256") != frontend.W2L_POLICY_PAYLOAD_SHA256
                or prewrite.get("expected_frame_count") != expected_frames
                or prewrite.get("frame_count") != expected_frames
                or len(prewrite.get("frame_pixel_array_sha256s", [])) != expected_frames
                or prewrite.get("ordered_frame_hashes_sha256") != sha_bytes(canonical(prewrite.get("frame_pixel_array_sha256s")))):
            raise ValueError("Wav2Lip forward/prewrite receipt content differs from its strict bound row")
        native["wav2lip_runtime_compatibility_receipts"] = records
        native["forward_counts"] = {"sfd_forward_count": forward["sfd_forward_count"],
                                     "wav2lip_generator_forward_count": forward["wav2lip_generator_forward_count"]}
        events = forward.get("forward_entry_observations")
        if (not isinstance(events, list)
                or len(events) != forward["sfd_forward_count"] + forward["wav2lip_generator_forward_count"]
                or any((item.get("row_id"), item.get("model_id"), item.get("seed"))
                       != (row["row_id"], model_id, row["seed"]) for item in events)
                or any(item.get("policy_at_forward_entry") != guard.STRICT_BOOTSTRAP_POLICY
                       for item in events)
                or any(item.get("benchmark_override_observed_and_restored") is not True
                       for item in events if item.get("kind") == "sfd_s3fd_forward_pre_hook")):
            raise ValueError("Wav2Lip actual forward event stream is incomplete or differs from the strict row policy")
        native["runtime_policy_observations"] = {
            "policy_contract_sha256": forward.get("policy_sha256"),
            "policy_payload_sha256": forward.get("policy_payload_sha256"),
            "sfd_network_fallback_prohibited": forward.get("sfd_network_fallback_prohibited"),
            "all_observed_actual_forward_entries_match_strict_policy": forward.get(
                "all_observed_actual_forward_entries_match_strict_policy"),
            "constructor_policy_before": records["constructor_policy"]["evidence"].get("policy_before_constructor"),
            "constructor_policy_at_return": records["constructor_policy"]["evidence"].get("policy_at_constructor_return"),
            "constructor_policy_after_restore": records["constructor_policy"]["evidence"].get("policy_after_restore"),
            "normalized_forward_entry_observations": [
                {key: value for key, value in item.items() if key != "pid"} for item in events],
        }
        native["prewrite_pixel_arrays"] = {
            "expected_frame_count": expected_frames, "frame_count": len(prewrite["frame_pixel_array_sha256s"]),
            "frame_pixel_array_sha256s": prewrite["frame_pixel_array_sha256s"],
            "ordered_frame_hashes_sha256": prewrite["ordered_frame_hashes_sha256"],
            "raw_intermediate_avi": {key: intermediate.get(key) for key in ("bytes", "sha256")},
        }
    else:
        receipt_ref = generation.get("adapter_receipt")
        if not isinstance(receipt_ref, Mapping) or not isinstance(receipt_ref.get("path"), str):
            raise ValueError("native model adapter receipt is missing")
        receipt_raw, adapter_file_ref = raw_file(Path(receipt_ref["path"]), "native adapter receipt", 4 * 1024 * 1024)
        adapter_obj = json.loads(receipt_raw.decode("utf-8"))
        expected_schema = ("cross_tfg_musetalk_stagea_adapter_receipt_v1" if model_id == MODELS[1]
                           else "cross_tfg_ditto_stagea_adapter_receipt_v1")
        expected_status = "PASS" if model_id == MODELS[1] else "PASS_ENGINEERING_ONLY"
        if (adapter_obj.get("schema") != expected_schema or adapter_obj.get("status") != expected_status
                or (adapter_obj.get("row_id"), adapter_obj.get("model_id"),
                    adapter_obj.get("audio_frontend"), adapter_obj.get("seed"))
                    != (row["row_id"], model_id, row["audio_frontend"], row["seed"])
                or adapter_obj.get("adapter_sha256") != model["generator"].get("adapter_sha256")
                or adapter_obj.get("renderer_sha256") != model["generator"].get("renderer_sha256")
                or adapter_file_ref["sha256"] != receipt_ref.get("file_sha256")):
            raise ValueError("native adapter receipt schema/row/seed/source identity mismatch")
        input_ref = adapter_obj.get("input_audio")
        pcm_ref = result.get("generator_pcm", {})
        if (not isinstance(input_ref, Mapping)
                or input_ref.get("file_sha256") != pcm_ref.get("file_sha256")
                or input_ref.get("path") != pcm_ref.get("path")
                or input_ref.get("pcm_float32_sha256") != pcm_ref.get("pcm_sha256")
                or input_ref.get("sample_rate_hz") != 16000
                or input_ref.get("channels") != 1
                or input_ref.get("sample_count") != pcm_ref.get("sample_count")):
            raise ValueError("native adapter receipt does not bind the exact method-transformed input WAV")
        silent_ref = adapter_obj.get("silent_video")
        render = result.get("render", {})
        if (not isinstance(silent_ref, Mapping)
                or silent_ref.get("path") != render.get("generated_video_path")
                or silent_ref.get("file_sha256") != render.get("generated_video_sha256")
                or silent_ref.get("frame_count") != render.get("frame_count")
                or silent_ref.get("fps_num", 25) != 25
                or silent_ref.get("fps_den", 1) != 1):
            raise ValueError("native adapter receipt does not bind the actual generated video and 25/1 clock")
        face_ref = adapter_obj.get("face") if model_id == MODELS[1] else adapter_obj.get("source_face")
        source_face = row.get("source_face", {})
        if (not isinstance(face_ref, Mapping)
                or face_ref.get("path") != str(Path(source_face.get("path", "")).resolve(strict=True))
                or face_ref.get("file_sha256") != source_face.get("file_sha256")):
            raise ValueError("native adapter receipt does not bind the exact frozen source face")
        native["native_adapter_receipt"] = {
            **adapter_file_ref, "schema": adapter_obj.get("schema"), "status": adapter_obj.get("status"),
            "adapter_sha256": adapter_obj.get("adapter_sha256"),
            "renderer_sha256": adapter_obj.get("renderer_sha256"),
            "native_report_sha256": adapter_obj.get("native_report_sha256"),
            "model_profile": adapter_obj.get("model_profile"),
            "source_commit": adapter_obj.get("source_commit"),
            "input_audio_file_sha256": input_ref.get("file_sha256"),
            "input_audio_pcm_float32_sha256": input_ref.get("pcm_float32_sha256"),
            "input_audio_sample_count": input_ref.get("sample_count"),
            "input_audio_path": input_ref.get("path"),
            "silent_video_file_sha256": silent_ref.get("file_sha256"),
            "silent_video_frame_count": silent_ref.get("frame_count"),
            "silent_video_fps": silent_ref.get("fps"),
            "native_report_sha256": adapter_obj.get("native_report_sha256"),
        }
    native["renderer_policy"] = {
        "row_id": bootstrap.get("row_id"), "model_id": bootstrap.get("model_id"),
        "audio_frontend": bootstrap.get("audio_frontend"), "seed": bootstrap.get("seed"),
        "entry_path": bootstrap.get("entry_path"),
        "expected_backend_entry": bootstrap.get("expected_backend_entry"),
        "entry_matches_backend": bootstrap.get("entry_matches_backend"),
        "bootstrap_source_sha256": bootstrap.get("bootstrap_source_sha256"),
        "policy_payload_sha256": bootstrap.get("policy_payload_sha256"),
        "policy": bootstrap.get("policy"),
    }
    return native


def _capture_attempt(core: Any, frontend: Any, guard: Any,
                     row: Mapping[str, Any], model: Mapping[str, Any],
                     method: Mapping[str, Any], natural_path: Path, samples: Any,
                     natural_ref: Mapping[str, Any], attempt_root: Path, receipt_dir: Path,
                     args: argparse.Namespace, gate: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["CROSS_TFG_DETERMINISM_RECEIPT_DIR"] = str(receipt_dir.resolve(strict=True))
    execution_flags = {"renderer_invoked": False, "official_score_invoked": False}
    try:
        current_path, current_samples, current_natural_ref = core._reference(row["natural_pcm"])
        if (str(current_path) != str(natural_path)
                or current_natural_ref.get("file_sha256") != natural_ref.get("file_sha256")
                or current_natural_ref.get("pcm_sha256") != natural_ref.get("pcm_sha256")
                or current_natural_ref.get("sample_count") != natural_ref.get("sample_count")):
            raise ValueError("original natural PCM changed between diagnostic attempts")
        source_media_refs: dict[str, Any] = {}
        for key in ("source_face", "source_video"):
            source_ref = row.get(key, {})
            path_text = source_ref.get("path") if isinstance(source_ref, Mapping) else None
            expected_sha = source_ref.get("file_sha256") if isinstance(source_ref, Mapping) else None
            if not isinstance(path_text, str) or not isinstance(expected_sha, str):
                raise ValueError(f"candidate row is missing its exact {key} input binding")
            resolved, raw = guard.safe_regular(Path(path_text), f"repeat attempt {key}", 2 * 1024 * 1024 * 1024)
            if sha_bytes(raw) != expected_sha:
                raise ValueError(f"candidate row {key} changed between diagnostic attempts")
            source_media_refs[key] = {"path": str(resolved), "bytes": len(raw), "sha256": sha_bytes(raw)}
        original_generator = core._run_generator
        original_run_logged = core.run_logged
        generator_descriptor = model["generator"]
        renderer_entry = (generator_descriptor["native_entry_path"] if row["model_id"] == MODELS[0]
                          else generator_descriptor["adapter_path"])
        renderer_entry = str(Path(str(renderer_entry)).resolve(strict=True))
        official_evaluator = str(Path(str(args.syncnet_evaluator)).resolve(strict=True))

        def generator_dispatch(**kwargs: Any) -> dict[str, Any]:
            actual_row = kwargs.get("row", {})
            actual_model = kwargs.get("model", {})
            if actual_row.get("model_id") == MODELS[0]:
                return frontend._wav2lip_native_generator(core, **kwargs)
            return frontend._adapter_generator(core, **kwargs)

        def logged_dispatch(argv: Any, *run_args: Any, **run_kwargs: Any) -> Any:
            execution = original_run_logged(argv, *run_args, **run_kwargs)
            if isinstance(argv, (list, tuple)) and len(argv) > 1:
                executable = str(argv[1])
                if executable == renderer_entry:
                    execution_flags["renderer_invoked"] = True
                if executable == official_evaluator:
                    execution_flags["official_score_invoked"] = True
            return execution

        core._run_generator = generator_dispatch
        core.run_logged = logged_dispatch
        try:
            result = core._render_and_score(row, model, method, current_path, current_samples,
                                            current_natural_ref, attempt_root, args, qc_repeat=True)
        finally:
            core._run_generator = original_generator
            core.run_logged = original_run_logged
            os.environ.pop("CROSS_TFG_DETERMINISM_RECEIPT_DIR", None)
        if (result.get("status") != "PASS"
                or result.get("natural_pcm_sha256") != natural_ref.get("pcm_sha256")
                or result.get("score_audio_sha256") != natural_ref.get("pcm_sha256")
                or result.get("source_face_sha256") != row.get("source_face", {}).get("file_sha256")
                or result.get("source_video_sha256") != row.get("source_video", {}).get("file_sha256")):
            raise ValueError("Stage-A result did not retain the exact natural source context and scorer audio")
        generated_ref = result.get("generator_pcm", {})
        generated_path = Path(str(generated_ref.get("path", "")))
        generated_samples, generated_info = core._reference({
            "path": str(generated_path), "file_sha256": generated_ref.get("file_sha256"),
            "pcm_sha256": generated_ref.get("pcm_sha256"), "sample_count": generated_ref.get("sample_count"),
        })
        if (generated_info.get("pcm_sha256") != generated_ref.get("pcm_sha256")
                or generated_info.get("sample_count") != natural_ref.get("sample_count")
                or generated_info.get("sample_rate_hz") != 16000
                or generated_info.get("channels") != 1):
            raise ValueError("method-transformed generator WAV failed exact canonical readback")
        del generated_samples
        transform_diagnostics = result.get("transform_diagnostics")
        method_config = method.get("config", {})
        if not isinstance(transform_diagnostics, Mapping):
            raise ValueError("candidate dispatcher did not return transform provenance")
        expected_config_path = Path(str(method_config.get("v3_cli_config_path", "")))
        if not expected_config_path.is_absolute():
            expected_config_path = Path(args.project_root) / expected_config_path
        if (transform_diagnostics.get("operation") != "pinned_v3_shared_recipient_transition_head"
                or transform_diagnostics.get("v3_cli_config_path") != str(expected_config_path.resolve(strict=True))
                or transform_diagnostics.get("v3_cli_config_sha256") != method_config.get("v3_cli_config_sha256")
                or transform_diagnostics.get("candidate_checkpoint_sha256") != method_config.get("candidate_checkpoint_sha256")
                or transform_diagnostics.get("candidate_training_record_sha256") != method_config.get("candidate_training_record_sha256")
                or transform_diagnostics.get("candidate_selection_record_sha256") != method_config.get("candidate_fixed_step_selection_record_sha256")
                or transform_diagnostics.get("completed_optimizer_steps") != 120
                or transform_diagnostics.get("input_pcm_float32_sha256") != natural_ref.get("pcm_sha256")
                or transform_diagnostics.get("output_pcm_float32_sha256") != generated_ref.get("pcm_sha256")):
            raise ValueError("actual transformed float32 PCM is not bound to the frozen step-120 head and generator WAV")
        if (result.get("generator_pcm_sha256") != generated_ref.get("pcm_sha256")
                or result.get("render", {}).get("generator_pcm_sha256") != generated_ref.get("pcm_sha256")
                or result.get("mux_audio_readback", {}).get("decoded_float32_pcm_sha256") != natural_ref.get("pcm_sha256")
                or result.get("mux_audio_readback", {}).get("matches_natural_canonical_pcm") is not True):
            raise ValueError("generator condition or original-natural-PCM mux content guard failed")

        generation = result["render"]["generation"]
        if row["model_id"] == MODELS[0]:
            audio_ref = generation.get("generator_input_audio", {})
            if (audio_ref.get("path") != generated_ref.get("path")
                    or audio_ref.get("file_sha256") != generated_ref.get("file_sha256")
                    or audio_ref.get("pcm_float32_sha256") != generated_ref.get("pcm_sha256")
                    or audio_ref.get("sample_count") != generated_ref.get("sample_count")
                    or audio_ref.get("sample_rate_hz") != 16000 or audio_ref.get("channels") != 1):
                raise ValueError("Wav2Lip actual native WAV input differs from method output PCM")
        else:
            adapter_ref = generation.get("adapter_receipt", {})
            if (adapter_ref.get("path") is None
                    or adapter_ref.get("file_sha256") is None):
                raise ValueError("native M/D adapter output has no actual input/video receipt reference")

        native_refs = _read_native_receipts(result, row, model, frontend, core,
                                            receipt_dir, guard)
        render = result["render"]
        silent_path = Path(render["silent_video_path"])
        generated_video = Path(render["generated_video_path"])
        muxed_video = Path(render["muxed_video_path"])
        probe = core._ffprobe(silent_path, args.ffprobe)
        video_stream = core._stream_ref(probe, "video")
        width, height = int(video_stream.get("width", 0)), int(video_stream.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("decoded renderer video has invalid dimensions")
        frames, decode_rc = core._decode_rgb(silent_path, args.ffmpeg, width, height)
        if decode_rc != 0 or frames.ndim != 4 or frames.shape[1:] != (height, width, 3):
            raise ValueError("renderer video could not be decoded as exact RGB24 frames")
        frame_hashes = [sha_bytes(np.ascontiguousarray(frame).tobytes(order="C")) for frame in frames]
        clock_ref = Path(render["timestamps_path"])
        clock_raw, clock_file_ref = raw_file(clock_ref, "source clock sidecar", 8 * 1024 * 1024)
        clock = json.loads(clock_raw.decode("utf-8"))
        if (clock.get("frame_count") != int(render.get("frame_count", -1))
                or len(clock.get("frames", [])) != int(render.get("frame_count", -1))
                or len(frame_hashes) != int(render.get("frame_count", -1))
                or (clock.get("fps_num"), clock.get("fps_den")) != (25, 1)):
            raise ValueError("decoded RGB frame count, Stage-A frame count, and rational clock disagree")
        elementary = {
            "generated_video_sha256": core.sha256_file(generated_video),
            "generated_video_elementary_sha256": core._video_es_sha(generated_video, args.ffmpeg),
            "silent_video_sha256": core.sha256_file(silent_path),
            "silent_video_elementary_sha256": core._video_es_sha(silent_path, args.ffmpeg),
            "muxed_video_sha256": core.sha256_file(muxed_video),
            "muxed_video_elementary_sha256": core._video_es_sha(muxed_video, args.ffmpeg),
            "render_claimed_video_elementary_sha256": render.get("video_stream_sha256"),
        }
        if (elementary["silent_video_elementary_sha256"] != elementary["muxed_video_elementary_sha256"]
                or elementary["render_claimed_video_elementary_sha256"] != elementary["muxed_video_elementary_sha256"]):
            raise ValueError("Stage-A original-PCM mux changed or misreported the renderer video elementary stream")

        arrays = result.get("_internal_arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != set(core.ARRAY_DTYPES):
            raise ValueError("Stage-A official score matrix does not have the exact complete array roster")
        array_refs = {key: _array_identity(value) for key, value in sorted(arrays.items())}
        query_keys = [list(key) for key in core._query_keys(arrays)]
        query_key_sha = sha_bytes(canonical(query_keys))
        score_npz_path = Path(result["score"]["raw_matrix_npz"]["path"])
        score_npz = _npz_array_identities(score_npz_path, arrays)
        evaluator_ref = result["score"]["official_evaluator"]
        evaluator_npz = _npz_array_identities(Path(evaluator_ref["raw_npz_path"]), {
            "distance_by_query_lag": arrays["distance_by_query_lag"],
            "lag_offsets_frames": arrays["lag_offsets_frames"],
        })
        if (score_npz["sha256"] != result["score"]["raw_matrix_npz"].get("file_sha256")
                or evaluator_npz["sha256"] != evaluator_ref.get("raw_npz_sha256")):
            raise ValueError("raw score archive bytes differ from the Stage-A official score receipts")
        frame_map_ref = result["score"]["primary_syncnet_input"]
        frame_map_raw, frame_map_file_ref = raw_file(
            Path(frame_map_ref["frame_map_path"]), "selected-track source-clock map", 16 * 1024 * 1024)
        if sha_bytes(frame_map_raw) != frame_map_ref["frame_map_sha256"]:
            raise ValueError("selected-track source-clock map raw SHA differs from its Stage-A ref")
        result_meta = {
            "row_id": row["row_id"], "model_id": row["model_id"],
            "audio_frontend": row["audio_frontend"], "seed": row["seed"],
            "method_id": row["method_id"],
            "natural_pcm": {"file_sha256": natural_ref["file_sha256"],
                            "pcm_sha256": natural_ref["pcm_sha256"],
                            "sample_count": natural_ref["sample_count"],
                            "sample_rate_hz": 16000, "channels": 1},
            "generator_pcm": {key: generated_ref.get(key) for key in (
                "path", "file_sha256", "pcm_sha256", "sample_rate_hz", "channels", "sample_count", "subtype")},
            "method_transform": {
                "dispatcher_path": CANDIDATE_DISPATCHER_REL,
                "dispatcher_sha256": CANDIDATE_DISPATCHER_SHA256,
                "v3_cli_path": transform_diagnostics.get("v3_cli_path"),
                "v3_cli_sha256": transform_diagnostics.get("v3_cli_sha256"),
                "v3_cli_config_path": transform_diagnostics.get("v3_cli_config_path"),
                "v3_cli_config_sha256": transform_diagnostics.get("v3_cli_config_sha256"),
                "head_implementation_sha256": transform_diagnostics.get("head_implementation_sha256"),
                "candidate_checkpoint_sha256": transform_diagnostics.get("candidate_checkpoint_sha256"),
                "candidate_training_record_sha256": transform_diagnostics.get("candidate_training_record_sha256"),
                "candidate_selection_record_sha256": transform_diagnostics.get("candidate_selection_record_sha256"),
                "completed_optimizer_steps": transform_diagnostics.get("completed_optimizer_steps"),
                "input_pcm_float32_sha256": transform_diagnostics.get("input_pcm_float32_sha256"),
                "output_pcm_float32_sha256": transform_diagnostics.get("output_pcm_float32_sha256"),
                "matches_actual_generator_pcm":
                    transform_diagnostics.get("output_pcm_float32_sha256") == generated_ref.get("pcm_sha256"),
                "diagnostics_sha256": sha_bytes(canonical(transform_diagnostics)),
            },
            "native_evidence": native_refs,
            "native_runtime_observations": native_refs.get("runtime_policy_observations"),
            "native_adapter_semantics": ({key: value for key, value in native_refs.get(
                "native_adapter_receipt", {}).items() if key not in {"path", "bytes", "sha256",
                                                                      "native_report_sha256", "input_audio_path"}}
                if row["model_id"] != MODELS[0] else None),
            "source_media_refs": source_media_refs,
            "video": {**elementary,
                      "fps_num": render.get("fps_num"), "fps_den": render.get("fps_den"),
                      "frame_count": len(frames), "width": width, "height": height,
                      "rgb24_frame_hashes": frame_hashes,
                      "rgb24_ordered_hashes_sha256": sha_bytes(canonical(frame_hashes)),
                      "timestamp_sidecar": clock_file_ref,
                      "timestamp_payload": clock,
                      "selected_track_id": render.get("selected_track_id"),
                      "selected_track_index": render.get("selected_track_index"),
                      "source_clock_frame_map": {**frame_map_file_ref,
                                                  "payload": json.loads(frame_map_raw.decode("utf-8"))},
                      "frame_map_semantics": {
                          "fps_num": json.loads(frame_map_raw.decode("utf-8")).get("fps_num"),
                          "fps_den": json.loads(frame_map_raw.decode("utf-8")).get("fps_den"),
                          "selected_track_id": json.loads(frame_map_raw.decode("utf-8")).get("selected_track_id"),
                          "selected_track_index": json.loads(frame_map_raw.decode("utf-8")).get("selected_track_index"),
                          "frames": json.loads(frame_map_raw.decode("utf-8")).get("frames")}},
            "official_score": {"query_keys": query_keys,
                               "query_keys_sha256": query_key_sha,
                               "score_model_sha256": result["score"].get("score_model_sha256"),
                               "preprocessing_sha256": result["score"].get("preprocessing_sha256"),
                               "score_audio_pcm_sha256": result["score"].get("score_audio_pcm_sha256"),
                               "score_preprocessed_pcm16_sha256": result["score"].get("score_preprocessed_pcm16_sha256"),
                               "score_preprocessed_sample_count": result["score"].get("score_preprocessed_sample_count"),
                               "primary_pcm16_window": {key: result["score"].get("primary_pcm16_window", {}).get(key)
                                                        for key in ("source_float32_pcm_sha256", "start_sample",
                                                                    "requested_stop_sample_exclusive",
                                                                    "stop_sample_exclusive", "tail_clipped_to_natural_pcm",
                                                                    "source_start_pts_num", "source_start_pts_den",
                                                                    "source_stop_pts_num", "source_stop_pts_den",
                                                                    "source_frame_start", "source_frame_stop_exclusive",
                                                                    "fps_num", "fps_den", "pcm_sha256", "sample_count")},
                               "arrays": array_refs,
                               "stagea_score_npz": score_npz,
                               "official_evaluator_npz": evaluator_npz,
                               "distance_matrix_shape": list(arrays["distance_by_query_lag"].shape),
                               "lag_offsets_frames": np.asarray(arrays["lag_offsets_frames"]).tolist(),
                               "official_evaluator_receipt": evaluator_ref.get("receipt")},
            "natural_pcm_mux": result["mux_audio_readback"],
            "source_context": {"source_face_sha256": row.get("source_face", {}).get("file_sha256"),
                               "source_video_sha256": row.get("source_video", {}).get("file_sha256"),
                               "natural_pcm_sha256": natural_ref.get("pcm_sha256"),
                               "natural_pcm_file_sha256": natural_ref.get("file_sha256")},
            "primary_pcm16_window": result["score"].get("primary_pcm16_window"),
            "primary_syncnet_input": result["score"].get("primary_syncnet_input"),
            "row_root": str(attempt_root.resolve(strict=True)),
            "status": "PASS",
            "resource_gate": gate,
            "execution_flags": dict(execution_flags),
        }
        # Runtime receipt directory is separate from Stage-A row content, but is pinned in this attempt record.
        return result_meta
    except Exception as exc:
        os.environ.pop("CROSS_TFG_DETERMINISM_RECEIPT_DIR", None)
        return {"row_id": row["row_id"], "model_id": row["model_id"], "seed": row["seed"],
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "attempt_root": str(attempt_root),
                "receipt_dir": str(receipt_dir), "resource_gate": gate,
                "execution_flags": dict(execution_flags)}


def _pair_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if first.get("status") != "PASS" or second.get("status") != "PASS":
        return {"status": "INCOMPLETE", "all_exact": False,
                "reason": "at least one paired render/official-score attempt did not pass"}
    pcm_a, pcm_b = first["generator_pcm"], second["generator_pcm"]
    video_a, video_b = first["video"], second["video"]
    score_a, score_b = first["official_score"], second["official_score"]
    audio_exact = all(pcm_a.get(key) == pcm_b.get(key) for key in (
        "file_sha256", "pcm_sha256", "sample_rate_hz", "channels", "sample_count", "subtype"))
    prewrite_a = first["native_evidence"].get("prewrite_pixel_arrays")
    prewrite_b = second["native_evidence"].get("prewrite_pixel_arrays")
    prewrite_exact = True
    if prewrite_a is not None or prewrite_b is not None:
        prewrite_exact = (isinstance(prewrite_a, Mapping) and isinstance(prewrite_b, Mapping)
                          and prewrite_a.get("expected_frame_count") == prewrite_b.get("expected_frame_count")
                          and prewrite_a.get("frame_count") == prewrite_b.get("frame_count")
                          and prewrite_a.get("frame_pixel_array_sha256s") == prewrite_b.get("frame_pixel_array_sha256s")
                          and prewrite_a.get("ordered_frame_hashes_sha256") == prewrite_b.get("ordered_frame_hashes_sha256")
                          and prewrite_a.get("raw_intermediate_avi", {}).get("sha256") ==
                              prewrite_b.get("raw_intermediate_avi", {}).get("sha256"))
    exact_fields = {
        "transformed_generator_pcm": audio_exact,
        "pinned_step120_transform_and_actual_pcm": (
            first.get("method_transform") == second.get("method_transform")
            and isinstance(first.get("method_transform"), Mapping)
            and first["method_transform"].get("matches_actual_generator_pcm") is True
            and second.get("method_transform", {}).get("matches_actual_generator_pcm") is True
            and first["method_transform"].get("input_pcm_float32_sha256") == first.get("source_context", {}).get("natural_pcm_sha256")
            and second["method_transform"].get("input_pcm_float32_sha256") == second.get("source_context", {}).get("natural_pcm_sha256")
            and first["method_transform"].get("output_pcm_float32_sha256") == pcm_a.get("pcm_sha256")
            and second["method_transform"].get("output_pcm_float32_sha256") == pcm_b.get("pcm_sha256")
        ),
        "source_context": first.get("source_context") == second.get("source_context"),
        "renderer_strict_policy_and_native_entry": first.get("native_evidence", {}).get("renderer_policy") ==
                                                    second.get("native_evidence", {}).get("renderer_policy"),
        "native_runtime_policy_observations": first.get("native_runtime_observations") ==
                                              second.get("native_runtime_observations"),
        "native_adapter_semantics": first.get("native_adapter_semantics") ==
                                    second.get("native_adapter_semantics"),
        "native_model_forward_evidence": first.get("native_evidence", {}).get("forward_counts") ==
                                         second.get("native_evidence", {}).get("forward_counts")
                                         and first.get("native_evidence", {}).get("native_adapter_receipt", {}).get("adapter_sha256") ==
                                             second.get("native_evidence", {}).get("native_adapter_receipt", {}).get("adapter_sha256"),
        "generated_video_container": video_a.get("generated_video_sha256") == video_b.get("generated_video_sha256"),
        "generated_video_elementary_stream": video_a.get("generated_video_elementary_sha256") == video_b.get("generated_video_elementary_sha256"),
        "silent_video_container": video_a.get("silent_video_sha256") == video_b.get("silent_video_sha256"),
        "silent_video_elementary_stream": video_a.get("silent_video_elementary_sha256") == video_b.get("silent_video_elementary_sha256"),
        "original_pcm_mux_container": video_a.get("muxed_video_sha256") == video_b.get("muxed_video_sha256"),
        "original_pcm_mux_video_elementary_stream": video_a.get("muxed_video_elementary_sha256") == video_b.get("muxed_video_elementary_sha256"),
        "decoded_rgb24_frames": (video_a.get("width"), video_a.get("height"), video_a.get("frame_count"),
                                  video_a.get("rgb24_frame_hashes")) ==
                                 (video_b.get("width"), video_b.get("height"), video_b.get("frame_count"),
                                  video_b.get("rgb24_frame_hashes")),
        "rational_frame_clock": (video_a.get("fps_num"), video_a.get("fps_den"),
                                  video_a.get("timestamp_payload")) ==
                                 (video_b.get("fps_num"), video_b.get("fps_den"),
                                  video_b.get("timestamp_payload")),
        "selected_track_and_source_clock_map": video_a.get("frame_map_semantics") == video_b.get("frame_map_semantics"),
        "official_query_map": (score_a.get("query_keys"), score_a.get("query_keys_sha256")) ==
                              (score_b.get("query_keys"), score_b.get("query_keys_sha256")),
        "official_score_source_and_pcm_window": all(score_a.get(key) == score_b.get(key) for key in (
            "score_model_sha256", "preprocessing_sha256", "score_audio_pcm_sha256",
            "score_preprocessed_pcm16_sha256", "score_preprocessed_sample_count", "primary_pcm16_window")),
        "full_score_matrix_and_arrays": score_a.get("arrays") == score_b.get("arrays"),
        "stagea_raw_score_npz_arrays": score_a.get("stagea_score_npz", {}).get("arrays") ==
                                       score_b.get("stagea_score_npz", {}).get("arrays"),
        "official_evaluator_matrix_arrays": score_a.get("official_evaluator_npz", {}).get("arrays") ==
                                            score_b.get("official_evaluator_npz", {}).get("arrays"),
        "Wav2Lip_prewrite_pixel_arrays": prewrite_exact,
        "natural_PCM_F32_delivery": first.get("natural_pcm_mux") == second.get("natural_pcm_mux"),
    }
    return {"status": "EXACT_MATCH" if all(exact_fields.values()) else "DIFFERENCES_OBSERVED",
            "all_exact": all(exact_fields.values()), "exact_fields": exact_fields,
            "common_query_key_count": len(set(tuple(x) for x in score_a.get("query_keys", []))
                                           & set(tuple(x) for x in score_b.get("query_keys", []))),
            "distance_matrix_shape_pair": [score_a.get("distance_matrix_shape"), score_b.get("distance_matrix_shape")],
            "lag_offsets_frames_pair": [score_a.get("lag_offsets_frames"), score_b.get("lag_offsets_frames")]}


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    context = _check_plan_and_binding(args)
    return {
        "schema": PREFLIGHT_SCHEMA, "status": "PASS_METADATA_ONLY",
        "host": platform.node(), "created_unix": time.time(), "valid_for_seconds": 1800,
        "worker_path": str(Path(__file__).resolve(strict=True)),
        "worker_bytes": Path(__file__).stat().st_size,
        "worker_sha256": sha_file(Path(__file__)),
        "candidate_plan": context["plan_raw_ref"],
        "candidate_rows": context["planned_rows_raw_ref"],
        "candidate_plan_manifest": context["plan_manifest_raw_ref"],
        "runtime_binding": context["runtime_raw_ref"],
        "diagnostic_manifest": context["diagnostic_manifest_raw_ref"],
        "source_pins": context["source_pins"],
        "candidate_output_root": context["candidate_output_root"],
        "diagnostic_output_root": context["diagnostic_output_root"],
        "planned_diagnostic_outputs": 6,
        "predeclared_rows": [row[0]["row_id"] for row in context["row_models"]],
        "live_resource_probe": context["live_resource_probe"],
        "verified_runtime_identity": {
            "gpu_uuid": context["runtime"].get("gpu_uuid"),
            "runtime_binding_raw_sha256": context["runtime_raw_ref"]["sha256"],
            "capacity_basis_status": context["runtime"].get("resources", {}).get("capacity_basis", {}).get("status"),
            "media_bytes_estimated": context["runtime"].get("resources", {}).get("capacity_basis", {}).get("media_bytes_estimated"),
        },
        "execution_performed": False, "rendering_performed": False,
        "scoring_performed": False, "confirm_opened": False,
        "primary_rows_added": 0, "method_selected": False,
    }


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_step120_transform(context: Mapping[str, Any], method: Mapping[str, Any]):
    dispatcher = context["candidate_dispatcher"]
    project_root = Path(context["project_root"]).resolve(strict=True)
    config = method.get("config")
    refs = context["plan"].get("source_reference_plans", {})
    cli_ref = refs.get("v3_cli", {})
    config_ref = refs.get("v3_cli_config", {})
    if (not isinstance(config, Mapping)
            or config.get("method") != dispatcher.TRANSFORM_METHOD
            or config.get("v3_cli_config_path") != config_ref.get("path")
            or config.get("v3_cli_config_sha256") != config_ref.get("raw_sha256")
            or config.get("candidate_checkpoint_sha256") !=
               context["plan"].get("execution", {}).get("candidate_checkpoint_sha256")
            or config.get("fixed_optimizer_steps") != 120):
        raise ValueError("repeat diagnostic method config is not the frozen step-120 candidate")
    config_path = Path(str(config["v3_cli_config_path"]))
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("V3 candidate configuration must remain inside the pinned project root") from exc
    cli_path = project_root / str(cli_ref.get("path", ""))
    candidate = dispatcher._CandidateTransform(
        cli_path=cli_path,
        cli_sha256=str(cli_ref.get("raw_sha256", "")),
        config_path=config_path,
        config_sha256=str(config_ref.get("raw_sha256", "")),
        repo_root=project_root,
    )
    if candidate.candidate_binding.get("completed_optimizer_steps") != 120:
        raise ValueError("pinned V3 candidate transform did not resolve the terminal step-120 checkpoint")
    return candidate


def _execute(args: argparse.Namespace) -> int:
    if not args.execute_repeat_diagnostic:
        raise ValueError("repeat execution requires --execute-repeat-diagnostic")
    if not args.preflight_receipt or not args.preflight_receipt_raw_sha256:
        raise ValueError("repeat execution requires the exact approved metadata preflight receipt path and raw SHA")
    preflight, preflight_raw, preflight_ref = read_json_exact(
        args.preflight_receipt, "repeat diagnostic preflight receipt",
        args.preflight_receipt_raw_sha256, None, 16 * 1024 * 1024)
    if (preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("status") != "PASS_METADATA_ONLY"
            or preflight.get("receipt_sha256") != _payload_self_sha(preflight, "receipt_sha256")
            or preflight.get("host") != platform.node()
            or time.time() - float(preflight.get("created_unix", 0)) > float(preflight.get("valid_for_seconds", 0))):
        raise ValueError("metadata preflight receipt is invalid, stale, or from another host")
    context = _check_plan_and_binding(args)
    for key, expected in (
        ("candidate_plan", context["plan_raw_ref"]),
        ("candidate_rows", context["planned_rows_raw_ref"]),
        ("candidate_plan_manifest", context["plan_manifest_raw_ref"]),
        ("runtime_binding", context["runtime_raw_ref"]),
        ("diagnostic_manifest", context["diagnostic_manifest_raw_ref"]),
    ):
        if preflight.get(key) != expected:
            raise ValueError(f"metadata preflight receipt no longer matches current input: {key}")
    runtime = context["runtime"]
    runtime_ref = context["runtime_raw_ref"]
    plan = context["plan"]
    core = context["core"]
    frontend = context["frontend"]
    guard = context["guard"]
    output_root = Path(args.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("diagnostic output root must be new and empty")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise FileNotFoundError("diagnostic output namespace parent is absent or unsafe")
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    status_path = output_root / "diagnostic_status.json"
    initial = {"schema": RESULT_SCHEMA, "status": "RUNNING", "run_id": plan.get("run_id"),
               "runtime_binding_raw_sha256": runtime_ref["sha256"],
               "candidate_plan_raw_sha256": context["plan_raw_ref"]["sha256"],
               "diagnostic_manifest_raw_sha256": context["diagnostic_manifest_raw_ref"]["sha256"],
               "planned_outputs": 6, "completed_outputs": 0, "primary_rows_added": 0,
               "method_selected": False, "rendering_performed": False, "scoring_performed": False,
               "pid": os.getpid(), "started_unix": time.time()}
    create_json(status_path, initial)
    args_for_core = argparse.Namespace(
        project_root=str(context["project_root"]),
        generator_child=runtime["execution_bindings"]["generator_child"],
        generator_timeout=float(runtime.get("timeouts", {}).get("generator_seconds", 900)),
        ffmpeg=runtime["execution_bindings"]["ffmpeg"],
        ffmpeg_timeout=float(runtime.get("timeouts", {}).get("ffmpeg_seconds", 300)),
        ffprobe=runtime["execution_bindings"]["ffprobe"],
        syncnet_root=runtime["execution_bindings"]["syncnet_root"],
        syncnet_python=runtime["execution_bindings"]["syncnet_python"],
        syncnet_checkpoint=runtime["execution_bindings"]["syncnet_checkpoint"],
        syncnet_evaluator=runtime["execution_bindings"]["syncnet_evaluator"],
        syncnet_timeout=float(runtime.get("timeouts", {}).get("syncnet_seconds", 1800)),
        ffmpeg_sha256=runtime["expected_file_sha256"]["ffmpeg"],
        pcm16_lineage_sha256=runtime["expected_file_sha256"]["pcm16_lineage"],
    )
    _resource_environment(runtime_ref)
    result_rows: list[dict[str, Any]] = []
    abort_remaining = False
    original_transform = core._transform
    candidate_transform = None
    candidate_transform_key: tuple[str, str] | None = None

    def method_transform(samples: Any, method: Mapping[str, Any]):
        nonlocal candidate_transform, candidate_transform_key
        config = method.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("candidate row lacks its method configuration")
        config_path = Path(str(config.get("v3_cli_config_path", "")))
        if not config_path.is_absolute():
            config_path = Path(context["project_root"]) / config_path
        key = (str(config_path.resolve(strict=True)), str(config.get("v3_cli_config_sha256", "")))
        if candidate_transform_key is not None and key != candidate_transform_key:
            raise ValueError("repeat diagnostic cannot mix step-120 candidate configs")
        if candidate_transform is None:
            candidate_transform = _make_step120_transform(context, method)
            candidate_transform_key = key
        return candidate_transform.transform(samples)

    # Mirror the frozen V3 candidate dispatcher: use its exact _CandidateTransform
    # implementation on the original natural PCM instead of Stage-A's base method
    # switch, which does not recognize v3_parameter_checkpoint.
    core._transform = method_transform
    try:
        for row, model, method in context["row_models"]:
            natural_path, natural_samples, natural_ref = core._reference(row["natural_pcm"])
            attempts: list[dict[str, Any]] = []
            for attempt_number in (1, 2):
                attempt_name = f"attempt{attempt_number:02d}"
                attempt_parent = output_root / "rows" / row["model_id"] / attempt_name
                attempt_parent.mkdir(mode=0o700, parents=True, exist_ok=False)
                row_root = attempt_parent / "stagea_row"
                receipt_dir = attempt_parent / "renderer_strict_receipts"
                gate = core._row_resource_gate(plan, row, args_for_core, output_root)
                if gate.get("status") != "PASS":
                    attempts.append({"status": "FAILED_RESOURCE_GATE", "row_id": row["row_id"],
                                     "attempt": attempt_name, "resource_gate": gate})
                    abort_remaining = True
                    break
                captured = _capture_attempt(core, frontend, guard, row, model, method,
                                            natural_path, natural_samples, natural_ref,
                                            row_root, receipt_dir, args_for_core, gate)
                captured["attempt"] = attempt_name
                captured["renderer_receipt_dir"] = str(receipt_dir.resolve(strict=True)) if receipt_dir.exists() else str(receipt_dir)
                attempts.append(captured)
                initial["completed_outputs"] += 1
                flags = captured.get("execution_flags", {})
                initial["rendering_performed"] = bool(initial["rendering_performed"] or flags.get("renderer_invoked") is True)
                initial["scoring_performed"] = bool(initial["scoring_performed"] or flags.get("official_score_invoked") is True)
                initial["current_row_id"] = row["row_id"]
                initial["current_attempt"] = attempt_name
                replace_json(status_path, initial)
            if abort_remaining:
                while len(attempts) < 2:
                    attempts.append({"status": "NOT_RUN_RESOURCE_GATE", "row_id": row["row_id"],
                                     "attempt": f"attempt{len(attempts)+1:02d}"})
                comparison = {"status": "INCOMPLETE", "all_exact": False,
                              "reason": "live resource gate stopped remaining diagnostic renders"}
            else:
                comparison = _pair_comparison(attempts[0], attempts[1])
            result_rows.append({"row_id": row["row_id"], "model_id": row["model_id"],
                                "audio_frontend": row["audio_frontend"], "seed": row["seed"],
                                "method_id": row["method_id"], "attempts": attempts,
                                "comparison": comparison})
            if abort_remaining:
                for remaining_row, _remaining_model, _remaining_method in context["row_models"][len(result_rows):]:
                    result_rows.append({"row_id": remaining_row["row_id"],
                                        "model_id": remaining_row["model_id"],
                                        "audio_frontend": remaining_row["audio_frontend"],
                                        "seed": remaining_row["seed"], "method_id": remaining_row["method_id"],
                                        "attempts":[{"status":"NOT_RUN_RESOURCE_GATE","attempt":"attempt01"},
                                                    {"status":"NOT_RUN_RESOURCE_GATE","attempt":"attempt02"}],
                                        "comparison":{"status":"INCOMPLETE","all_exact":False,
                                                      "reason":"resource gate stopped the serial diagnostic"}})
                break
        all_pass = all(all(attempt.get("status") == "PASS" for attempt in row_result["attempts"])
                       for row_result in result_rows)
        all_exact = all(row_result.get("comparison", {}).get("all_exact") is True
                        for row_result in result_rows)
        status = ("COMPLETE_EXACT_MATCH" if all_pass and all_exact else
                  "COMPLETE_DIFFERENCES_OBSERVED" if all_pass else "INCOMPLETE")
        final = {
            "schema": RESULT_SCHEMA, "status": status,
            "purpose": "non-efficacy renderer/source-clock repeat diagnostic only",
            "candidate_plan": context["plan_raw_ref"],
            "candidate_rows": context["planned_rows_raw_ref"],
            "candidate_plan_manifest": context["plan_manifest_raw_ref"],
            "samehost_runtime_binding": context["runtime_raw_ref"],
            "immutable_predeclared_manifest": context["diagnostic_manifest_raw_ref"],
            "source_pins": context["source_pins"],
            "output_root": str(output_root.resolve(strict=True)),
            "planned_rows": 3, "planned_outputs": 6,
            "result_rows": result_rows,
            "counts": {"PASS_OUTPUTS": sum(attempt.get("status") == "PASS"
                                             for rr in result_rows for attempt in rr["attempts"]),
                       "FAILED_OUTPUTS": sum(attempt.get("status", "").startswith("FAILED")
                                             for rr in result_rows for attempt in rr["attempts"]),
                       "NOT_RUN_OUTPUTS": sum(attempt.get("status", "").startswith("NOT_RUN")
                                               for rr in result_rows for attempt in rr["attempts"])},
            "primary_rows_added": 0, "method_selected": False,
            "confirm_opened": False, "confirm_audio_read": False,
            "rendering_performed": any(attempt.get("execution_flags", {}).get("renderer_invoked") is True
                                        for rr in result_rows for attempt in rr["attempts"]),
            "scoring_performed": any(attempt.get("execution_flags", {}).get("official_score_invoked") is True
                                      for rr in result_rows for attempt in rr["attempts"]),
            "started_unix": initial["started_unix"], "finished_unix": time.time(),
            "pid": os.getpid(), "preflight_receipt": preflight_ref,
        }
        final["result_sha256"] = _payload_self_sha(final, "result_sha256")
        create_json(output_root / "candidate_generation_repeat_diagnostic_result_attempt02.json", final)
        initial.update({"schema": RESULT_SCHEMA, "status": status,
                        "completed_outputs": final["counts"]["PASS_OUTPUTS"] + final["counts"]["FAILED_OUTPUTS"],
                        "result_path": str((output_root / "candidate_generation_repeat_diagnostic_result_attempt02.json").resolve()),
                        "finished_unix": final["finished_unix"],
                        "primary_rows_added": 0, "method_selected": False})
        replace_json(status_path, initial)
        print(json.dumps({"status": status, "result_path": initial["result_path"],
                          "planned_outputs": 6, "primary_rows_added": 0,
                          "method_selected": False}, sort_keys=True))
        return 0 if all_pass else 2
    except BaseException as exc:
        failure = {"schema": RESULT_SCHEMA, "status": "INCOMPLETE",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "pid": os.getpid(),
                   "started_unix": initial["started_unix"], "finished_unix": time.time(),
                   "planned_outputs": 6, "completed_outputs": initial["completed_outputs"],
                   "result_rows": result_rows, "primary_rows_added": 0,
                   "method_selected": False, "confirm_opened": False}
        failure["result_sha256"] = _payload_self_sha(failure, "result_sha256")
        failure_path = output_root / "candidate_generation_repeat_diagnostic_result_attempt02.json"
        if not failure_path.exists():
            create_json(failure_path, failure)
        initial.update({"schema": RESULT_SCHEMA, "status": "INCOMPLETE",
                        "error": failure["error"], "finished_unix": failure["finished_unix"],
                        "result_path": str(failure_path.resolve()),
                        "primary_rows_added": 0, "method_selected": False})
        replace_json(status_path, initial)
        raise
    finally:
        core._transform = original_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(ROOT_DEFAULT))
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--planned-rows-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-bytes", type=int, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(EXPECTED_OUTPUT_ROOT))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--preflight-receipt-raw-sha256")
    parser.add_argument("--execute-repeat-diagnostic", action="store_true")
    parser.add_argument("--worker-expected-sha256", required=True)
    args = parser.parse_args(argv)
    self_ref = raw_file(Path(__file__), "repeat diagnostic worker", 16 * 1024 * 1024)[1]
    if self_ref["sha256"] != args.worker_expected_sha256.lower():
        raise ValueError("repeat diagnostic worker raw SHA does not match its exact dispatch pin")
    if args.preflight_only == args.execute_repeat_diagnostic:
        raise ValueError("select exactly one mode: --preflight-only or --execute-repeat-diagnostic")
    if args.preflight_only:
        if not args.preflight_receipt:
            raise ValueError("metadata-only mode requires a create-only preflight receipt path")
        payload = _run_preflight(args)
        payload["receipt_sha256"] = _payload_self_sha(payload, "receipt_sha256")
        raw, ref = create_json(args.preflight_receipt, payload)
        print(json.dumps({"status": payload["status"], "preflight_receipt": ref,
                          "receipt_sha256": payload["receipt_sha256"],
                          "execution_performed": False}, sort_keys=True))
        return 0
    return _execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
