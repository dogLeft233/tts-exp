from __future__ import annotations

import hashlib
import json
import pathlib
import shlex
from typing import Any


REPO = pathlib.Path.cwd()
PACKET = REPO / ".codex/scratch/candidate_v3_cpu_preflight_attempt03"
REQ_DIR = PACKET / "requests_attempt01"
DOWNLOAD_DIR = PACKET / "downloads_attempt01"
for path in (REQ_DIR, DOWNLOAD_DIR, PACKET / "source_manifest_attempt01.json", PACKET / "request_index_attempt01.json"):
    if path.exists() or path.is_symlink():
        raise SystemExit(f"attempt03 packet output already exists; refusing overwrite: {path}")
REQ_DIR.mkdir()
DOWNLOAD_DIR.mkdir()


def raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def pin(name: str, rel: str, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    path = REPO / rel
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SystemExit(f"source exceeds bound: {rel}")
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": raw_sha(raw), "name": name}


def flat(pin_obj: dict[str, Any]) -> dict[str, Any]:
    return {key: pin_obj[key] for key in ("path", "bytes", "sha256")}


def write_json(path: pathlib.Path, obj: Any) -> bytes:
    raw = (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    return raw


source_paths = {
    "worker_attempt03": ".codex/scratch/candidate_v3_cpu_preflight_attempt03/preflight_candidate_v3_cpu_attempt03.py",
    "worker_attempt02_parent": ".codex/scratch/candidate_v3_cpu_preflight_attempt02/preflight_candidate_v3_cpu_attempt02.py",
    "worker_attempt02_to_attempt03_diff": ".codex/scratch/candidate_v3_cpu_preflight_attempt03/attempt02_to_attempt03.diff",
    "load_plan_metadata_qc": ".codex/scratch/candidate_v3_cpu_preflight_attempt03/load_plan_metadata_qc_attempt01.json",
    "strict_bootstrap03": ".codex/scratch/dev360_cpu_bind_plan_preflight_attempt01/staging_files/015/sitecustomize.py",
    "candidate_plan13": ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/actual_readback_attempt01/candidate_rebound_plan_attempt13.json",
    "candidate_manifest14": ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/actual_readback_attempt01/plan_manifest_metadata_repair_attempt14.json",
    "candidate_config03": ".codex/scratch/candidate_v3_interop_thread_repair_attempt03/v3_method_candidate_config_attempt03.json",
    "candidate_config02_predecessor": ".codex/scratch/root_candidate_original_metadata_readback_attempt01/candidate_config.json",
    "reference_enhanced_wav": ".codex/scratch/v3_step120_bundle_cpu_repro_attempt04/downloads/enhanced.wav",
    "reference_cli_metadata": ".codex/scratch/v3_step120_bundle_cpu_repro_attempt04/downloads/cli_metadata.json",
    "audio_io": "scripts/audio_enhancement/audio_io.py",
    "candidate_cli03": "scripts/audio_enhancement/enhance_v3_attempt03.py",
    "head_v3": "scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
    "overdub": "scripts/audio_enhancement/overdub.py",
    "package_initializer": "scripts/audio_enhancement/__init__.py",
    "baselines": "scripts/audio_enhancement/baselines.py",
    "pipeline": "scripts/audio_enhancement/pipeline.py",
}
source_pins = {name: pin(name, rel) for name, rel in source_paths.items()}

task_root = "/dev/shm/cross_tfg_audio_head_20261003/candidate_v3_cpu_preflight_attempt03"
control_root = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/candidate_v3_cpu_preflight_attempt03"
project_root = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003"
python = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
plan_remote = f"{control_root}/inputs/candidate_rebound_plan_attempt13.json"
manifest_remote = f"{control_root}/inputs/plan_manifest_metadata_repair_attempt14.json"
config_remote = f"{task_root}/project/v3_method_candidate_config_attempt03.json"
config02_remote = f"{control_root}/inputs/candidate_config_cli02.json"
ref_wav_remote = f"{control_root}/inputs/same_profile_enhanced.wav"
ref_meta_remote = f"{control_root}/inputs/same_profile_cli_metadata.json"
bootstrap_remote = f"{task_root}/bootstrap/sitecustomize.py"
worker_remote = f"{control_root}/workers/preflight_candidate_v3_cpu_attempt03.py"
output_remote = f"{task_root}/output"
receipt_remote = f"{control_root}/output/cpu_preflight_receipt_attempt03.json"

q00 = f'''set -eu
CONTROL={shlex.quote(control_root)}
TASK={shlex.quote(task_root)}
for p in /root /root/tts-research-system-20261001 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control /dev /dev/shm /dev/shm/cross_tfg_audio_head_20261003; do test -d "$p"; test ! -L "$p"; done
test ! -e "$CONTROL"; test ! -L "$CONTROL"
test ! -e "$TASK"; test ! -L "$TASK"
mkdir -m 700 "$CONTROL" "$TASK"
mkdir -m 700 "$CONTROL/inputs" "$CONTROL/workers" "$CONTROL/output"
mkdir -m 700 "$TASK/bootstrap" "$TASK/project" "$TASK/output"
mkdir -m 700 "$TASK/project/scripts" "$TASK/project/scripts/audio_enhancement"
test -d "$CONTROL/inputs" && test -z "$(find "$CONTROL/inputs" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$CONTROL/workers" && test -z "$(find "$CONTROL/workers" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$CONTROL/output" && test -z "$(find "$CONTROL/output" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$TASK/project" && test -z "$(find "$TASK/project" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$TASK/bootstrap" && test -z "$(find "$TASK/bootstrap" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$TASK/output" && test -z "$(find "$TASK/output" -mindepth 1 -maxdepth 1 -print -quit)"
printf '%s\\n' '{{"status":"PASS_CREATED_FRESH_CPU_PREFLIGHT_NAMESPACE","control_root":"{control_root}","task_root":"{task_root}","gpu_or_model_execution":false}}'
'''

requests: list[dict[str, Any]] = []


def add_request(name: str, obj: dict[str, Any]) -> None:
    path = REQ_DIR / f"{name}.json"
    raw = write_json(path, obj)
    requests.append({"name": name, "op": obj["op"], "path": str(path.resolve()), "bytes": len(raw), "sha256": raw_sha(raw)})


add_request("q00_create_fresh_cpu_preflight_namespace", {"op": "exec", "command": q00, "timeout": 60})
upload_map = {
    "worker_attempt03": worker_remote,
    "strict_bootstrap03": bootstrap_remote,
    "candidate_plan13": plan_remote,
    "candidate_manifest14": manifest_remote,
    "candidate_config03": config_remote,
    "candidate_config02_predecessor": config02_remote,
    "reference_enhanced_wav": ref_wav_remote,
    "reference_cli_metadata": ref_meta_remote,
}
for key, remote in upload_map.items():
    item = source_pins[key]
    add_request(f"q_upload_{key}", {"op": "upload", "local": item["path"], "remote": remote, "expected_bytes": item["bytes"], "expected_sha256": item["sha256"], "timeout": 120})

source_uploads = {
    "audio_io": f"{task_root}/project/scripts/audio_enhancement/audio_io.py",
    "candidate_cli03": f"{task_root}/project/scripts/audio_enhancement/enhance_v3_attempt03.py",
    "head_v3": f"{task_root}/project/scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
    "overdub": f"{task_root}/project/scripts/audio_enhancement/overdub.py",
    "package_initializer": f"{task_root}/project/scripts/audio_enhancement/__init__.py",
    "baselines": f"{task_root}/project/scripts/audio_enhancement/baselines.py",
    "pipeline": f"{task_root}/project/scripts/audio_enhancement/pipeline.py",
}
for key, remote in source_uploads.items():
    item = source_pins[key]
    add_request(f"q_upload_source_{key}", {"op": "upload", "local": item["path"], "remote": remote, "expected_bytes": item["bytes"], "expected_sha256": item["sha256"], "timeout": 120})

python_command = f'''set -eu
CONTROL={shlex.quote(control_root)}
TASK={shlex.quote(task_root)}
WORKER={shlex.quote(worker_remote)}
EXPECTED_WORKER_SHA={shlex.quote(source_pins["worker_attempt03"]["sha256"])}
test -f "$WORKER" && test ! -L "$WORKER"
test "$(sha256sum "$WORKER" | awk '{{print $1}}')" = "$EXPECTED_WORKER_SHA"
test -d {shlex.quote(output_remote)} && test -z "$(find {shlex.quote(output_remote)} -mindepth 1 -maxdepth 1 -print -quit)"
test ! -e {shlex.quote(receipt_remote)} && test ! -L {shlex.quote(receipt_remote)}
for name in ${{!CROSS_TFG_@}}; do unset "$name"; done
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={shlex.quote(str(pathlib.PurePosixPath(task_root) / "bootstrap"))}
export PYTHONHASHSEED=20261003 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CROSS_TFG_STRICT_STARTUP_POLICY=1 CUDA_VISIBLE_DEVICES=
exec {shlex.quote(python)} -B "$WORKER" \\
  --repo-root {shlex.quote(project_root)} \\
  --code-root {shlex.quote(str(pathlib.PurePosixPath(task_root) / "project"))} \\
  --plan {shlex.quote(plan_remote)} \\
  --plan-manifest {shlex.quote(manifest_remote)} \\
  --config {shlex.quote(config_remote)} \\
  --predecessor-config {shlex.quote(config02_remote)} \\
  --reference-wav {shlex.quote(ref_wav_remote)} \\
  --reference-metadata {shlex.quote(ref_meta_remote)} \\
  --output-root {shlex.quote(output_remote)} \\
  --receipt {shlex.quote(receipt_remote)}
'''
add_request("q_run_strict_cpu_preflight", {"op": "exec", "command": python_command, "timeout": 1800})
download_local = str((DOWNLOAD_DIR / "cpu_preflight_receipt_attempt03.json").resolve())
add_request("q_download_cpu_preflight_receipt", {"op": "download", "remote": receipt_remote, "local": download_local, "timeout": 60})

plan_obj = json.loads((REPO / source_paths["candidate_plan13"]).read_text(encoding="utf-8"))
manifest_obj = json.loads((REPO / source_paths["candidate_manifest14"]).read_text(encoding="utf-8"))
if manifest_obj.get("manifest_sha256") != "420d73cd5fb95692ded1c8bbd8cd82363ccb40169a064f21121be70f041909e5":
    raise SystemExit("actual plan manifest14 embedded self-hash differs")
remote_only_inputs = {
    "same_profile_reference_source_wav": {
        "path": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/v5_fit40_input_stage_attempt03/fit40_source_inputs/source_wav/BAC009S0052W0121.wav",
        "bytes": 125322,
        "sha256": "2d45f2b65dec29f77bcd56f05f883b16207a99e7848df3a06a203819a915b261",
        "pcm_float32_sha256": "f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c",
        "sample_count": 62639,
        "sample_rate_hz": 16000,
    },
    "candidate_provenance_project_files": json.loads((REPO / source_paths["candidate_config03"]).read_text(encoding="utf-8"))["candidate_provenance"],
    "candidate_dev_inputs": {
        row["sample_id"]: row["natural_pcm"]
        for row in plan_obj["rows"]
        if row["model_id"] == "wav2lip_gan_bac9a81e" and row["seed"] == 20261003
    },
}

manifest = {
    "schema": "cross_tfg_v3_candidate_cpu_preflight_source_manifest_v1",
    "attempt": "candidate_v3_cpu_preflight_attempt03",
    "status": "SOURCE_AND_REQUEST_PACKET_READY_FOR_ROOT_REVIEW_NOT_DISPATCHED",
    "scope": {
        "model_or_gpu_execution_performed": False,
        "remote_action_performed": False,
        "confirm_audio_or_effects_read": False,
        "candidate180_started": False,
        "same_profile_anchor_and_20_unique_dev_audios_planned": True,
        "pcm_delta_against_old_anchor_is_non_gating": True,
    },
    "worker": flat(source_pins["worker_attempt03"]),
    "parent_worker": flat(source_pins["worker_attempt02_parent"]),
    "source_evidence": {key: flat(value) for key, value in source_pins.items()},
    "worker_diff": flat(source_pins["worker_attempt02_to_attempt03_diff"]),
    "metadata_only_load_plan_qc": flat(source_pins["load_plan_metadata_qc"]),
    "local_sealer": {"path": str(pathlib.Path(__file__).resolve()), "bytes": pathlib.Path(__file__).stat().st_size, "sha256": raw_sha(pathlib.Path(__file__).read_bytes())},
    "remote_roots": {"control_root": control_root, "task_root": task_root, "project_root": project_root, "fresh_output_root": output_remote, "receipt": receipt_remote},
    "interpreter": {"lexical_path": python, "resolved_path": "/usr/bin/python3.10", "python_version": [3, 10, 12], "numpy": "1.26.4", "torch": "2.5.1+cu124", "strict_bootstrap_sha256": source_pins["strict_bootstrap03"]["sha256"], "cuda_visible_devices": "", "cuda_initialized_must_remain": False},
    "actual_plan_manifest": {"plan_raw_sha256": source_pins["candidate_plan13"]["sha256"], "plan_sha256": plan_obj["plan_sha256"], "manifest_raw_sha256": source_pins["candidate_manifest14"]["sha256"], "manifest_sha256": manifest_obj["manifest_sha256"]},
    "remote_only_expected_inputs": remote_only_inputs,
    "execution_contract": {
        "one_process": True,
        "rows": ["same-profile trained step120 anchor", "20 distinct frozen Dev natural PCM inputs"],
        "no_video_render_or_syncnet": True,
        "no_confirm_inputs": True,
        "all_outputs_create_only_under_fresh_shm_root": True,
        "strict_policy": {"PYTHONHASHSEED": "20261003", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CROSS_TFG_STRICT_STARTUP_POLICY": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        "old_vs_new_anchor_pcm_equality_is_gate": False,
    },
    "request_names": [request["name"] for request in requests],
}
manifest["manifest_sha256"] = raw_sha(canonical(manifest))
manifest_raw = write_json(PACKET / "source_manifest_attempt01.json", manifest)

evidence = {key: flat(value) for key, value in source_pins.items()}
evidence["source_manifest_attempt01"] = {"path": str((PACKET / "source_manifest_attempt01.json").resolve()), "bytes": len(manifest_raw), "sha256": raw_sha(manifest_raw)}
evidence["sealer_attempt03"] = {"path": str(pathlib.Path(__file__).resolve()), "bytes": pathlib.Path(__file__).stat().st_size, "sha256": raw_sha(pathlib.Path(__file__).read_bytes())}

index: dict[str, Any] = {
    "schema": "cross_tfg_v3_candidate_cpu_preflight_request_index_v1",
    "status": "READY_FOR_ROOT_REVIEW_NOT_DISPATCHED",
    "purpose": "Upload the frozen CPU-preflight source/dependencies, run one strict-bootstrap CPU process on the same-profile anchor plus 20 distinct Dev WAVs, and retrieve its receipt; no candidate180 render or score.",
    "control_root": control_root,
    "task_root": task_root,
    "project_root": project_root,
    "candidate_plan": {"path": source_pins["candidate_plan13"]["path"], "remote": plan_remote, "bytes": source_pins["candidate_plan13"]["bytes"], "raw_sha256": source_pins["candidate_plan13"]["sha256"], "plan_sha256": plan_obj["plan_sha256"]},
    "candidate_manifest": {"path": source_pins["candidate_manifest14"]["path"], "remote": manifest_remote, "bytes": source_pins["candidate_manifest14"]["bytes"], "raw_sha256": source_pins["candidate_manifest14"]["sha256"], "manifest_sha256": manifest_obj["manifest_sha256"]},
    "reference": {"output_wav": flat(source_pins["reference_enhanced_wav"]), "metadata": flat(source_pins["reference_cli_metadata"]), "reference_input": remote_only_inputs["same_profile_reference_source_wav"]},
    "worker": {**flat(source_pins["worker_attempt03"]), "remote": worker_remote},
    "output": {"root": output_remote, "receipt": receipt_remote, "download_local": download_local, "runtime_bytes_sha256": None},
    "source_manifest": {"path": str((PACKET / "source_manifest_attempt01.json").resolve()), "bytes": len(manifest_raw), "raw_sha256": raw_sha(manifest_raw), "manifest_sha256": manifest["manifest_sha256"]},
    "requests": requests,
    "source_evidence": evidence,
    "scope": manifest["scope"],
}
index["index_sha256"] = raw_sha(canonical(index))
index_raw = write_json(PACKET / "request_index_attempt01.json", index)

print(json.dumps({
    "worker": flat(source_pins["worker_attempt03"]),
    "worker_diff": flat(source_pins["worker_attempt02_to_attempt03_diff"]),
    "metadata_only_load_plan_qc": flat(source_pins["load_plan_metadata_qc"]),
    "manifest": {"path": str((PACKET / "source_manifest_attempt01.json").resolve()), "bytes": len(manifest_raw), "sha256": raw_sha(manifest_raw), "self_sha256": manifest["manifest_sha256"]},
    "index": {"path": str((PACKET / "request_index_attempt01.json").resolve()), "bytes": len(index_raw), "sha256": raw_sha(index_raw), "self_sha256": index["index_sha256"]},
    "requests": len(requests),
    "remote_action_performed": False,
}, sort_keys=True))
