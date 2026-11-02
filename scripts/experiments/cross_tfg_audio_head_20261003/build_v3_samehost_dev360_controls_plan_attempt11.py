#!/usr/bin/env python3
"""Build attempt04 of the source-only same-host identity/RMS Dev360 plan.

The builder consumes frozen context/model metadata and a separately collected,
raw-pinned current-host runtime/input binding. It does not open Dev or Confirm
audio, execute a model, render/score, call an API, or choose candidate weights.
Paths for the current deployment and Dev PCM are late-bound from the runtime
receipt; immutable model, scorer, source, face/video, and PCM identities must
still match the qualified frozen roster.
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


ROOT = Path(__file__).resolve().parents[3]
BASE_REL = "runs/cross_tfg_audio_head_20261003/stagea_dev_plan_attempt01/plan.json"
BASE_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
ROSTER_REL = "runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json"
ROSTER_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
STAGEA_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt05.py"
STAGEA_BYTES = 69_184
STAGEA_SHA256 = "6fcef9ac59983af9aeb70fa9fe4ec8bdb6e16e2c2ff3368d142f2702f9352034"
CROSS_FRONTEND_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt07.py"
CROSS_FRONTEND_BYTES = 20_951
CROSS_FRONTEND_SHA256 = "0183b1d09068df32fb71be42662a453930ed72529f904d82a14360de8ce3089a"
RUNNER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_samehost_dev_attempt05.py"
RUNNER_BYTES = 3_010
RUNNER_SHA256 = "dc964eb5fd9eec241148a8779a839e6a155f48bbc51f4454c4a4de0003cd32ef"
METHODS = ("identity", "rms_fixed")
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
MODEL_BINDING_FIELDS = (
    "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
    "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den",
)
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
QC06 = {
    "preflight_raw_sha256": "c0dd743e3b9b250c88d8ffade8cdaa8c21c18411c40647264f9e9ee5779b3171",
    "terminal_raw_sha256": "a4fa225dc299f1b89c5cd8fc20b915df7f613c51b491280624895eef052f7269",
    "manifest_raw_sha256": "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436",
    "determinism_policy_sha256": "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc",
    "source_sha256": "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d",
}
SEEDS = (20261003, 20261004, 20261005)
STRICT_BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
W2L_FRAME_PROBE_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/dev20_native_frame_binding_attempt03_w2l_expected_frames.json"
STRICT_BOOTSTRAP_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}
STRICT_SHIM_SOURCE_REL = {
    MODELS[1]: "scripts/experiments/cross_tfg_audio_head_20261003/musetalk_five_arg_shim_attempt07.py",
    MODELS[2]: "scripts/experiments/cross_tfg_audio_head_20261003/ditto_five_arg_shim_attempt06.py",
}
MODEL_ENTRY_PINS = {
    MODELS[1]: ("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code/render_silent_video_attempt09.py",
                "b75004340fbf90b937fb516c5d8ee013f193f763f5cf61bbfc7d28cb12ce88c8"),
    MODELS[2]: ("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt18/code/render_silent_video_attempt07.py",
                "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0"),
}


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


def read_regular(path: Path, label: str, limit: int = 16 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return raw


def load_plan(path: Path, sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label)
    if sha_bytes(raw) != sha:
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if (not isinstance(value, dict) or value.get("status") != "FROZEN"
            or value.get("plan_sha256") != payload_sha(value, "plan_sha256")):
        raise ValueError(f"{label} is not a self-sealed FROZEN plan")
    return value, raw


def require_safe_abs_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if "\x00" in value:
        raise ValueError(f"{label} contains a NUL")
    return value


def _check_runtime_binding(binding: dict[str, Any], raw_sha: str,
                           models: list[dict[str, Any]], contexts: list[dict[str, Any]],
                           project_root: Path) -> None:
    if binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS":
        raise ValueError("runtime binding must be a PASS current-host receipt")
    if binding.get("gpu_uuid") != GPU_UUID:
        raise ValueError("runtime binding GPU UUID differs from QC06 deterministic qualification")
    if binding.get("qc06_source_sha256") != QC06["source_sha256"]:
        raise ValueError("runtime binding does not reference the exact QC06 source")
    if binding.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]:
        raise ValueError("runtime deterministic policy differs from QC06")
    frame_binding = binding.get("wav2lip_native_frame_count_binding")
    if (not isinstance(frame_binding, dict)
            or frame_binding.get("schema") != "cross_tfg_v3_wav2lip_native_frame_count_binding_v1"
            or frame_binding.get("status") != "PASS_SOURCE_DERIVED_METADATA_ONLY"
            or frame_binding.get("frozen_plan_raw_sha256") != BASE_SHA256
            or frame_binding.get("model_or_video_output_observed") is not False):
        raise ValueError("runtime binding lacks the actual source-derived Wav2Lip per-clip frame map")
    qualifications = binding.get("backend_determinism_qualifications")
    if not isinstance(qualifications, dict) or set(qualifications) != set(MODELS):
        raise ValueError("runtime binding must contain actual strict repeat qualifications for all three backends")
    for model_id in MODELS:
        entry = qualifications.get(model_id)
        if (not isinstance(entry, dict) or entry.get("model_id") != model_id
                or entry.get("status") != "PASS_EXACT_MATCH"
                or entry.get("gpu_uuid") != GPU_UUID
                or entry.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
                or entry.get("policy_values") != STRICT_BOOTSTRAP_POLICY):
            raise ValueError(f"runtime binding lacks a same-GPU strict deterministic repeat for {model_id}")
    resources = binding.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("runtime binding lacks actual resource observations")
    if resources.get("gpu_uuid") != GPU_UUID or not isinstance(resources.get("gpu_free_mib"), int):
        raise ValueError("runtime GPU resource observation is incomplete or from another GPU")
    for name in ("effective_ram_available_bytes", "dev_shm_free_bytes", "output_free_bytes",
                 "output_required_bytes", "output_reserve_bytes"):
        if isinstance(resources.get(name), bool) or not isinstance(resources.get(name), int) or resources[name] < 0:
            raise ValueError(f"runtime resource field is invalid: {name}")
    if resources["output_free_bytes"] < resources["output_required_bytes"] + resources["output_reserve_bytes"]:
        raise ValueError("runtime output mount does not meet its explicit reserved-space gate")
    if resources.get("storage_gate_status") != "PASS":
        raise ValueError("runtime receipt has no passing explicit storage-capacity gate")
    if not isinstance(resources.get("capacity_basis_sha256"), str) or len(resources["capacity_basis_sha256"]) != 64:
        raise ValueError("runtime receipt lacks the exact capacity-basis SHA")

    context_map = binding.get("dev_contexts")
    by_sample: dict[str, dict[str, Any]] = {}
    for row in contexts:
        by_sample.setdefault(row["sample_id"], row)
    if not isinstance(context_map, dict) or set(context_map) != set(by_sample):
        raise ValueError("runtime Dev PCM source map must cover exactly the frozen 20 clips")
    for sample_id, expected in by_sample.items():
        observed = context_map[sample_id]
        if not isinstance(observed, dict):
            raise ValueError(f"runtime Dev PCM mapping is invalid for {sample_id}")
        for field in ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"):
            if observed.get(field) != expected["natural_pcm"].get(field):
                raise ValueError(f"runtime Dev PCM identity differs from frozen context for {sample_id}: {field}")
        require_safe_abs_path(observed.get("path"), f"runtime Dev PCM path {sample_id}")
    source_bindings = binding.get("shared_sources")
    if not isinstance(source_bindings, dict):
        raise ValueError("runtime binding lacks current face/video source paths")
    reference = contexts[0]
    for key, plan_key, identity_key in (("source_face", "source_face", "file_sha256"),
                                        ("source_video", "source_video", "file_sha256")):
        value = source_bindings.get(key)
        if not isinstance(value, dict) or value.get(identity_key) != reference[plan_key].get(identity_key):
            raise ValueError(f"runtime {key} identity differs from frozen Dev metadata")
        require_safe_abs_path(value.get("path"), f"runtime {key} path")
    if (source_bindings["source_video"].get("fps_num"), source_bindings["source_video"].get("fps_den")) != (
        reference["source_video"].get("fps_num"), reference["source_video"].get("fps_den")
    ):
        raise ValueError("runtime reference video frame clock differs from frozen metadata")

    backend_map = binding.get("backends")
    if not isinstance(backend_map, dict) or set(backend_map) != set(MODELS):
        raise ValueError("runtime backend map must cover the exact three qualified frontends")
    model_map = {model["model_id"]: model for model in models}
    for model_id in MODELS:
        expected = model_map[model_id]
        observed = backend_map[model_id]
        if not isinstance(observed, dict):
            raise ValueError(f"runtime backend binding is invalid: {model_id}")
        if observed.get("model_binding") != {field: expected.get(field) for field in MODEL_BINDING_FIELDS}:
            raise ValueError(f"runtime backend immutable model/scorer pins differ: {model_id}")
        generator = observed.get("generator")
        if not isinstance(generator, dict):
            raise ValueError(f"runtime backend generator descriptor is missing: {model_id}")
        for field in ("python", "fps_num", "fps_den"):
            if field not in generator:
                raise ValueError(f"runtime backend generator descriptor lacks {field}: {model_id}")
        require_safe_abs_path(generator["python"], f"runtime backend Python {model_id}")
        if (generator.get("fps_num"), generator.get("fps_den")) != (expected.get("fps_num"), expected.get("fps_den")):
            raise ValueError(f"runtime backend frame rate differs from frozen model pin: {model_id}")
        if model_id in STRICT_SHIM_SOURCE_REL:
            shim_rel = STRICT_SHIM_SOURCE_REL[model_id]
            shim_path = (project_root / shim_rel).resolve(strict=True)
            shim_raw = read_regular(shim_path, f"{model_id} strict runtime shim", 4 * 1024 * 1024)
            shim_sha = sha_bytes(shim_raw)
            if (generator.get("adapter_path") != str(shim_path)
                    or generator.get("adapter_sha256") != shim_sha):
                raise ValueError(f"runtime {model_id} adapter path/SHA must bind the new strict-policy shim")
            shim_deps = [item for item in observed.get("dependencies", []) if isinstance(item, dict)
                         and item.get("path") == str(shim_path) and item.get("bytes") == len(shim_raw)
                         and item.get("sha256") == shim_sha]
            if len(shim_deps) != 1:
                raise ValueError(f"runtime dependency closure lacks the exact {model_id} strict-policy shim")
        if model_id == MODELS[0]:
            if generator.get("checkpoint_sha256") != expected.get("model_sha256"):
                raise ValueError("runtime Wav2Lip checkpoint identity differs from frozen model pin")
            require_safe_abs_path(generator.get("repo"), "runtime Wav2Lip repository")
            require_safe_abs_path(generator.get("checkpoint"), "runtime Wav2Lip checkpoint")
            expected_native = str(Path(generator["repo"]) / "inference.py")
            if (generator.get("native_entry_path") != expected_native
                    or generator.get("native_entry_sha256") != "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
                    or generator.get("runtime_compat_sitecustomize_path") != str((project_root / "scripts/experiments/cross_tfg_audio_head_20261003/wav2lip_runtime_compat_attempt04/sitecustomize.py").resolve(strict=True))
                    or generator.get("runtime_compat_sitecustomize_sha256") != "462884a2534f73d93eb09cd05c87fefda677e3647366360fb81a9622fa7592ca"
                    or generator.get("native_frame_count_probe_raw_sha256") != frame_binding.get("probe", {}).get("raw_sha256")):
                raise ValueError("Wav2Lip generator does not bind the exact native entry, compatibility hook, and frame probe")
            clip_map = frame_binding.get("clips")
            if not isinstance(clip_map, dict) or len(clip_map) != 20:
                raise ValueError("runtime Wav2Lip frame map must cover the exact 20 natural clips")
            for sample_id, context in by_sample.items():
                clip = clip_map.get(sample_id)
                if (not isinstance(clip, dict)
                        or clip.get("natural_pcm_sha256") != context["natural_pcm"].get("pcm_sha256")
                        or clip.get("sample_count") != context["natural_pcm"].get("sample_count")
                        or (clip.get("fps_num"), clip.get("fps_den")) != (25, 1)
                        or type(clip.get("expected_inference_frame_count")) is not int
                        or clip["expected_inference_frame_count"] <= 0):
                    raise ValueError(f"runtime Wav2Lip frame count is not bound to natural PCM {sample_id}")
        else:
            if generator.get("contract") != "cross_tfg_silent_video_cli_v1":
                raise ValueError(f"runtime {model_id} does not use the qualified silent-video adapter contract")
            for field in ("adapter_path", "cwd"):
                require_safe_abs_path(generator.get(field), f"runtime {model_id} {field}")
            expected_entry, expected_sha = MODEL_ENTRY_PINS[model_id]
            if generator.get("render_entry_path") != expected_entry:
                raise ValueError(f"runtime {model_id} native renderer entry differs from the qualified path")
            if not any(item.get("path") == expected_entry and item.get("sha256") == expected_sha
                       for item in observed.get("dependencies", []) if isinstance(item, dict)):
                raise ValueError(f"runtime {model_id} native renderer source is not fixed-pin bound")
        dependencies = observed.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"runtime dependency closure is absent for {model_id}")
        seen: set[str] = set()
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {"path", "bytes", "sha256", "role"}:
                raise ValueError(f"runtime dependency entry schema mismatch: {model_id}")
            require_safe_abs_path(dependency.get("path"), f"runtime dependency path {model_id}")
            if (dependency["path"] in seen or isinstance(dependency["bytes"], bool)
                    or not isinstance(dependency["bytes"], int) or dependency["bytes"] < 0
                    or not isinstance(dependency["sha256"], str) or len(dependency["sha256"]) != 64
                    or not isinstance(dependency["role"], str)):
                raise ValueError(f"runtime dependency pin is invalid or duplicated: {model_id}")
            seen.add(dependency["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--base-dev-plan", type=Path, default=ROOT / BASE_REL)
    parser.add_argument("--frontend-roster-plan", type=Path, default=ROOT / ROSTER_REL)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from the source repository")
    base, base_raw = load_plan(args.base_dev_plan, BASE_SHA256, "frozen 20-clip base Dev plan")
    roster, roster_raw = load_plan(args.frontend_roster_plan, ROSTER_SHA256, "frozen three-backend roster")
    if base.get("schema_version") != "cross-tfg-audio-eval-v1" or roster.get("status") != "FROZEN":
        raise ValueError("frozen Dev/roster schemas differ from the reviewed source inputs")
    if base.get("generation_seed_family", {}).get("dev") != list(SEEDS) or base.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("base plan seed/split scope differs from the fixed Dev-only family")
    if len(base.get("rows", [])) != 180 or tuple(m.get("method_id") for m in base.get("methods", [])) != (
        "identity", "rms_fixed", "rms_localcompress"
    ):
        raise ValueError("base plan no longer contains the exact frozen 20×3 source context")
    source_models = roster.get("models")
    if not isinstance(source_models, list) or tuple(m.get("model_id") for m in source_models) != MODELS:
        raise ValueError("qualified backend roster differs from Wav2Lip/MuseTalk/Ditto")
    if sha_bytes(read_regular(project_root / STAGEA_REL, "Stage-A core")) != STAGEA_SHA256:
        raise ValueError("Stage-A core source differs from its qualified pin")
    if sha_bytes(read_regular(project_root / CROSS_FRONTEND_REL, "silent-video adapter")) != CROSS_FRONTEND_SHA256:
        raise ValueError("cross-front-end adapter differs from its qualified pin")
    runner_raw = read_regular(project_root / RUNNER_REL, "same-host Stage-A runner")
    if len(runner_raw) != RUNNER_BYTES or sha_bytes(runner_raw) != RUNNER_SHA256:
        raise ValueError("same-host Stage-A runner source pin has not been frozen")
    bootstrap_raw = read_regular(project_root / STRICT_BOOTSTRAP_REL, "strict renderer determinism bootstrap")
    bootstrap_sha = sha_bytes(bootstrap_raw)

    contexts_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in base["rows"]:
        if row.get("method_id") != "identity" or row.get("split") != "dev":
            continue
        key = (row["sample_id"], row["speaker_id"], int(row["seed"]))
        if key in contexts_by_key:
            raise ValueError(f"duplicate frozen source context: {key}")
        contexts_by_key[key] = row
    contexts = list(contexts_by_key.values())
    if (len(contexts) != 60 or len({row["sample_id"] for row in contexts}) != 20
            or len({row["speaker_id"] for row in contexts}) != 10
            or {int(row["seed"]) for row in contexts} != set(SEEDS)):
        raise ValueError("Dev contexts are no longer exactly 20 clips × 10 speakers × 3 seeds")
    for sample_id in {row["sample_id"] for row in contexts}:
        group = [row for row in contexts if row["sample_id"] == sample_id]
        if (len(group) != 3 or len({row["speaker_id"] for row in group}) != 1
                or len({canonical([row["natural_pcm"], row["score_audio"]]) for row in group}) != 1):
            raise ValueError(f"frozen natural PCM source changed across seeds: {sample_id}")
    if len({canonical(row["source_face"]) for row in contexts}) != 1 or len({canonical(row["source_video"]) for row in contexts}) != 1:
        raise ValueError("the Dev contexts do not share one frozen source face and video")

    runtime_raw = read_regular(args.runtime_binding, "current-host runtime binding", 8 * 1024 * 1024)
    runtime_sha = sha_bytes(runtime_raw)
    if runtime_sha != args.runtime_binding_raw_sha256.lower():
        raise ValueError("current-host runtime binding raw SHA differs from the supplied pin")
    runtime = json.loads(runtime_raw.decode("utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError("current-host runtime binding must be a JSON object")
    _check_runtime_binding(runtime, runtime_sha, source_models, contexts, project_root)

    base_methods = {method["method_id"]: method for method in base["methods"]}
    if set(METHODS) - set(base_methods):
        raise ValueError("base plan is missing identity/RMS method definitions")
    methods = [copy.deepcopy(base_methods[name]) for name in METHODS]
    method_sha = {name: sha_bytes(canonical(method["config"])) for name, method in zip(METHODS, methods)}
    rebound_models: list[dict[str, Any]] = []
    runtime_backends = runtime["backends"]
    for source_model in source_models:
        model = copy.deepcopy(source_model)
        binding = runtime_backends[model["model_id"]]
        model["generator"] = copy.deepcopy(binding["generator"])
        model["runtime_dependency_receipt_sha256"] = sha_bytes(canonical(binding))
        model["runtime_dependency_files"] = copy.deepcopy(binding["dependencies"])
        rebound_models.append(model)

    identity_source = sorted(contexts, key=lambda row: (row["sample_id"], int(row["seed"])))
    rows: list[dict[str, Any]] = []
    for source in identity_source:
        for model in rebound_models:
            for method_id in METHODS:
                row = copy.deepcopy(source)
                row["model_id"] = model["model_id"]
                row["audio_frontend"] = model["audio_frontend"]
                row["model_role"] = model.get("role", "candidate")
                row["method_id"] = method_id
                row["method_config_sha256"] = method_sha[method_id]
                row["checkpoint_sha256"] = None
                row["generation_settings"] = {
                    **row.get("generation_settings", {}),
                    "generator": model["model_id"],
                    "input_audio": "identity_or_fixed_rms_condition_only; score_audio_is_exact_original_natural_pcm",
                    "audio_transform": method_id,
                    "seed": int(source["seed"]),
                }
                row["row_id"] = (
                    f"dev_{row['speaker_id']}_{row['sample_id']}_{model['model_id']}_"
                    f"{row['face_id']}_{int(row['seed'])}_{method_id}"
                )
                if model["model_id"] == MODELS[0]:
                    frame_binding = runtime.get("wav2lip_native_frame_count_binding", {})
                    if (frame_binding.get("schema") != "cross_tfg_v3_wav2lip_native_frame_count_binding_v1"
                            or frame_binding.get("status") != "PASS_SOURCE_DERIVED_METADATA_ONLY"
                            or frame_binding.get("model_or_video_output_observed") is not False):
                        raise ValueError("runtime binding lacks the source-derived Wav2Lip native frame-count map")
                    clip = frame_binding.get("clips", {}).get(source["sample_id"])
                    natural = row.get("natural_pcm", {})
                    if (not isinstance(clip, dict)
                            or clip.get("natural_pcm_sha256") != natural.get("pcm_sha256")
                            or clip.get("sample_count") != natural.get("sample_count")
                            or (clip.get("fps_num"), clip.get("fps_den")) != (25, 1)):
                        raise ValueError("Wav2Lip row frame count is not tied to its exact frozen natural PCM")
                    row["wav2lip_native_frame_binding"] = copy.deepcopy(clip)
                rows.append(row)
    if len(rows) != 360 or len({row["row_id"] for row in rows}) != 360:
        raise ValueError("same-host baseline plan must contain exactly 360 unique identity/RMS rows")
    if not {row["row_id"] for row in rows}:
        raise ValueError("empty baseline plan")

    shared_sources = runtime["shared_sources"]
    for row in rows:
        pcm = runtime["dev_contexts"][row["sample_id"]]
        row["natural_pcm"] = {**row["natural_pcm"], "path": pcm["path"]}
        row["score_audio"] = {**row["score_audio"], "path": pcm["path"]}
        row["source_face"] = {**row["source_face"], "path": shared_sources["source_face"]["path"]}
        row["source_video"] = {**row["source_video"], "path": shared_sources["source_video"]["path"]}

    plan = copy.deepcopy(base)
    plan["run_id"] = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt09"
    plan["protocol"]["generator"]["child_script_sha256"] = QC06["source_sha256"]
    plan["methods"] = methods
    plan["models"] = rebound_models
    plan["rows"] = rows
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["simultaneous_family_ids"] = []
    plan["analysis"]["simultaneous_family_status"] = "NOT_APPLICABLE_CONTROL_RENDER_SUBRUN"
    plan["analysis"]["samehost_control_subrun"] = {
        "methods": list(METHODS), "rows": 360, "clip_count": 20, "seeds": list(SEEDS),
        "frontends": list(MODELS), "no_method_selection": True,
        "identity_rms_pair_policy": "each method retains distinct generated-video/raw-matrix products; pair support is exact original PCM, face/video, backend, seed, scorer, FPS, and source-clock/query-array identity",
    }
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "same_host_identity_rms_dev360_controls",
        "scope": "development only; 20 fixed clips × 3 frozen seeds × 3 qualified frontends × identity/RMS",
        "planned_rows": 360,
        "fresh_baseline_rerender": True,
        "same_host_controls": True,
        "actual_controls_run_id": plan["run_id"],
        "samehost_runtime_binding": {
            "path": str(args.runtime_binding.resolve(strict=True)),
            "bytes": len(runtime_raw), "raw_sha256": runtime_sha,
            "gpu_uuid": runtime["gpu_uuid"],
            "determinism_policy_sha256": runtime["determinism_policy_sha256"],
            "capacity_basis_sha256": runtime["resources"]["capacity_basis_sha256"],
            "backend_determinism_qualifications": runtime["backend_determinism_qualifications"],
        },
        "wav2lip_native_frame_count_binding": copy.deepcopy(runtime["wav2lip_native_frame_count_binding"]),
        "strict_renderer_determinism": {
            "policy_values": STRICT_BOOTSTRAP_POLICY,
            "qualified_policy_reference_sha256": QC06["determinism_policy_sha256"],
            "bootstrap_source": {"path": STRICT_BOOTSTRAP_REL, "bytes": len(bootstrap_raw),
                                  "raw_sha256": bootstrap_sha},
            "per_row_runtime_receipts_required": True,
            "confirm_audio_read": False,
        },
        "qc06_determinism_qualification": QC06,
        "control_methods": list(METHODS),
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "candidate_weights_included": False,
        "candidate_weights_selected": False,
        "development_weights_selected": False,
    }
    plan["protocol"]["generator"]["stagea_runner_sha256"] = STAGEA_SHA256
    plan["protocol"]["generator"]["cross_frontend_runner_sha256"] = RUNNER_SHA256
    pipeline_source_paths = {
        "samehost_dev360_plan_builder": "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_samehost_dev360_controls_plan_attempt11.py",
        "samehost_dev360_guarded_launcher": "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt12.py",
        "samehost_stagea_core": STAGEA_REL,
        "samehost_cross_frontend_adapter": CROSS_FRONTEND_REL,
        "samehost_stagea_dispatcher": RUNNER_REL,
        "samehost_runtime_binding_collector": "scripts/experiments/cross_tfg_audio_head_20261003/collect_v3_samehost_runtime_binding_attempt05.py",
        "samehost_w2l_runtime_sitecustomize": "scripts/experiments/cross_tfg_audio_head_20261003/wav2lip_runtime_compat_attempt04/sitecustomize.py",
        "samehost_w2l_native_frame_probe": W2L_FRAME_PROBE_REL,
        "samehost_dev360_evidence_collector": "scripts/experiments/cross_tfg_audio_head_20261003/collect_v3_samehost_dev360_controls_evidence_attempt03.py",
        "samehost_dev360_raw_media_collector": "scripts/experiments/cross_tfg_audio_head_20261003/collect_v3_stagea_raw_media_evidence_attempt03.py",
        "samehost_v3_baseline_reader_adapter": "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_candidate_samehost_controls_adapter_attempt04.py",
        "samehost_dynamic_540_joiner": "scripts/experiments/cross_tfg_audio_head_20261003/join_v3_candidate_540_samehost_controls_attempt06.py",
        "samehost_strict_determinism_bootstrap": STRICT_BOOTSTRAP_REL,
        "samehost_muse_strict_runtime_shim": "scripts/experiments/cross_tfg_audio_head_20261003/musetalk_five_arg_shim_attempt07.py",
        "samehost_ditto_strict_runtime_shim": "scripts/experiments/cross_tfg_audio_head_20261003/ditto_five_arg_shim_attempt06.py",
        "step120_method_materializer": "scripts/experiments/cross_tfg_audio_head_20261003/materialize_v3_step120_method_candidate_attempt02.py",
        "step120_samehost_plan_rebinder": "scripts/experiments/cross_tfg_audio_head_20261003/rebind_v3_step120_candidate_plan_samehost_attempt08.py",
        "step120_candidate_controls_adapter": "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_candidate_samehost_controls_adapter_attempt04.py",
        "step120_candidate_dispatcher": "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt05.py",
        "step120_candidate_guarded_launcher": "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt07.py",
        "step120_latebind_packet": "scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt02.py",
    }
    pipeline_sources: dict[str, dict[str, Any]] = {}
    for key, relative in pipeline_source_paths.items():
        raw = read_regular(project_root / relative, f"same-host pipeline source {key}")
        pipeline_sources[key] = {"path": relative, "bytes": len(raw), "raw_sha256": sha_bytes(raw)}
    plan["source_reference_plans"] = {
        "base_dev_plan": {"path": str(args.base_dev_plan.resolve()), "bytes": len(base_raw), "raw_sha256": sha_bytes(base_raw), "plan_sha256": base["plan_sha256"]},
        "frontend_roster_plan": {"path": str(args.frontend_roster_plan.resolve()), "bytes": len(roster_raw), "raw_sha256": sha_bytes(roster_raw), "plan_sha256": roster["plan_sha256"], "model_ids": list(MODELS)},
        "stagea_core": {"path": STAGEA_REL, "bytes": STAGEA_BYTES, "raw_sha256": STAGEA_SHA256},
        "cross_frontend_adapter": {"path": CROSS_FRONTEND_REL, "bytes": CROSS_FRONTEND_BYTES, "raw_sha256": CROSS_FRONTEND_SHA256},
        "samehost_stagea_runner": {"path": RUNNER_REL, "bytes": len(runner_raw), "raw_sha256": RUNNER_SHA256},
        "samehost_runtime_binding": {"path": str(args.runtime_binding.resolve()), "bytes": len(runtime_raw), "raw_sha256": runtime_sha},
        "qc06_actual_determinism": QC06,
        "samehost_dev360_source_closure": pipeline_sources,
        "actual_360_identity_rms_result_ledger": "LATE_BOUND_AFTER_STAGE_A_COMPLETES",
    }
    plan["execution"]["samehost_dev360_source_closure"] = pipeline_sources
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    if output.exists() or output.is_symlink() or getattr(output, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing existing plan output directory: {output}")
    output.mkdir(parents=True)
    plan_raw = pretty(plan)
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    (output / "plan.json").write_bytes(plan_raw)
    (output / "planned_rows.jsonl").write_bytes(rows_raw)
    manifest = {
        "schema": "cross_tfg_v3_samehost_dev360_controls_plan_v1",
        "run_id": plan["run_id"], "status": "FROZEN",
        "plan_sha256": plan["plan_sha256"], "plan_raw_sha256": sha_bytes(plan_raw),
        "planned_rows_sha256": sha_bytes(rows_raw), "planned_rows": 360,
        "method_counts": {method: sum(row["method_id"] == method for row in rows) for method in METHODS},
        "model_counts": {model: sum(row["model_id"] == model for row in rows) for model in MODELS},
        "sample_count": 20, "speaker_count": 10, "render_seed_family": list(SEEDS),
        "runtime_binding_raw_sha256": runtime_sha,
        "qc06_actual_determinism_manifest_sha256": QC06["manifest_raw_sha256"],
        "samehost_dev360_source_closure_sha256": sha_bytes(canonical(pipeline_sources)),
        "confirm_audio_read": False, "confirm_effects_read": False,
        "development_audio_opened_by_builder": False,
        "model_execution_performed": False, "rendering_or_scoring_performed": False,
        "candidate_weights_present": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    (output / "plan_manifest.json").write_bytes(pretty(manifest))
    print(json.dumps({"status": "FROZEN", "run_id": plan["run_id"], "planned_rows": 360,
                      "clips": 20, "seeds": list(SEEDS), "models": list(MODELS),
                      "methods": list(METHODS), "gpu_uuid": GPU_UUID,
                      "runtime_binding_raw_sha256": runtime_sha,
                      "confirmation_audio_opened": False,
                      "development_audio_opened_by_builder": False,
                      "model_execution_performed": False,
                      "plan_raw_sha256": sha_bytes(plan_raw), "planned_rows_sha256": sha_bytes(rows_raw)},
                     sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
