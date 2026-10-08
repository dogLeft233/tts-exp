from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


REPO = pathlib.Path.cwd()
PACKET = REPO / ".codex/scratch/candidate_v3_cpu_preflight_attempt03"
OLD_INDEX_PATH = PACKET / "request_index_attempt01.json"
OLD_MANIFEST_PATH = PACKET / "source_manifest_attempt01.json"
OLD_REQUEST_DIR = PACKET / "requests_attempt01"
NEW_REQUEST_DIR = PACKET / "requests_attempt02_revision02"
NEW_DOWNLOAD_DIR = PACKET / "downloads_attempt02_revision02"
NEW_MANIFEST_PATH = PACKET / "source_manifest_attempt02.json"
NEW_INDEX_PATH = PACKET / "request_index_attempt02.json"
for path in (NEW_REQUEST_DIR, NEW_DOWNLOAD_DIR, NEW_MANIFEST_PATH, NEW_INDEX_PATH):
    if path.exists() or path.is_symlink():
        raise SystemExit(f"attempt02 output exists; refusing overwrite: {path}")
NEW_REQUEST_DIR.mkdir()
NEW_DOWNLOAD_DIR.mkdir()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path: pathlib.Path, value: Any) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    return raw


def local_pin(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)}


def verify_self_hash(raw: bytes, key: str, expected_schema: str) -> dict[str, Any]:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or obj.get("schema") != expected_schema:
        raise SystemExit(f"unexpected schema in {key}")
    value = obj.pop(key, None)
    if not isinstance(value, str) or sha(canonical(obj)) != value:
        raise SystemExit(f"self-hash mismatch in {key}")
    obj[key] = value
    return obj


old_index_raw = OLD_INDEX_PATH.read_bytes()
old_index = verify_self_hash(old_index_raw, "index_sha256", "cross_tfg_v3_candidate_cpu_preflight_request_index_v1")
old_manifest_raw = OLD_MANIFEST_PATH.read_bytes()
old_manifest = verify_self_hash(old_manifest_raw, "manifest_sha256", "cross_tfg_v3_candidate_cpu_preflight_source_manifest_v1")
if old_index.get("status") != "READY_FOR_ROOT_REVIEW_NOT_DISPATCHED":
    raise SystemExit("attempt01 was not the expected undispatched source packet")
if old_index.get("requests") is None or len(old_index["requests"]) != 18:
    raise SystemExit("attempt01 request set differs from the expected 18 operations")

launcher_path = REPO / ".codex/scratch/dev360_candidate180_detached_launch_attempt10/launch_candidate180_detached_attempt10.py"
config_path = REPO / ".codex/scratch/candidate_v3_interop_thread_repair_attempt03/v3_method_candidate_config_attempt03.json"
launcher_text = launcher_path.read_text(encoding="utf-8")
config = json.loads(config_path.read_text(encoding="utf-8"))
expected_root = "/root/tts-research-system-20261001"
required_launcher_fragments = (
    'ROOT = Path("/root/tts-research-system-20261001")',
    'MATERIAL_DIR = ROOT / "runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01"',
    'METHOD_CONFIG = MATERIAL_DIR / "v3_method_candidate_config.json"',
    'material.get("candidate_config", {}).get("path") != str(METHOD_CONFIG.relative_to(ROOT))',
)
if any(fragment not in launcher_text for fragment in required_launcher_fragments):
    raise SystemExit("pinned candidate launcher no longer proves the repository root derivation")
provenance = config.get("candidate_provenance")
if not isinstance(provenance, dict) or set(provenance) != {"checkpoint", "training_record", "selection_record"}:
    raise SystemExit("candidate config provenance map is not the frozen three-file map")
for name, item in provenance.items():
    rel = pathlib.PurePosixPath(item.get("path", ""))
    if rel.is_absolute() or rel.parts[:2] != ("runs", "cross_tfg_audio_head_20261003"):
        raise SystemExit(f"unexpected repository-relative provenance path: {name}")

launcher_pin = local_pin(launcher_path)
config_pin = local_pin(config_path)
if launcher_pin["bytes"] != 51722 or launcher_pin["sha256"] != "1391ceb1a6b206359d16ddbbbbb7511e70260c69c7a9cdfc0e1d07a1a56cb133":
    raise SystemExit("actual candidate launcher10 source pin changed")
if config_pin["bytes"] != 5240 or config_pin["sha256"] != "fd2ce3509c457e40e618818828f6040ab65641315d40efe8f48e5dc3c5c3d1df":
    raise SystemExit("candidate config03 source pin changed")

source_pins: dict[str, dict[str, Any]] = {
    key: {field: value[field] for field in ("path", "bytes", "sha256")}
    for key, value in old_index["source_evidence"].items()
}
source_pins.update({
    "previous_request_index_attempt01": local_pin(OLD_INDEX_PATH),
    "previous_source_manifest_attempt01": local_pin(OLD_MANIFEST_PATH),
    "candidate_detached_launcher_attempt10": launcher_pin,
})

# The q00 guard was below mkdir of project/scripts in attempt01. Move that
# exact empty-project test directly after creation of the project directory.
req_by_name = {r["name"]: r for r in old_index["requests"]}
old_q00_ref = req_by_name["q00_create_fresh_cpu_preflight_namespace"]
old_qrun_ref = req_by_name["q_run_strict_cpu_preflight"]
old_q00_raw = pathlib.Path(old_q00_ref["path"]).read_bytes()
old_qrun_raw = pathlib.Path(old_qrun_ref["path"]).read_bytes()
old_q00 = json.loads(old_q00_raw)
old_qrun = json.loads(old_qrun_raw)
q00_command = old_q00["command"]
late_project_guard = 'test -d "$TASK/project" && test -z "$(find "$TASK/project" -mindepth 1 -maxdepth 1 -print -quit)"\n'
early_project_guard = 'test -d "$TASK/project" && test -z "$(find "$TASK/project" -mindepth 1 -maxdepth 1 -print -quit)"\n'
task_project_mkdir = 'mkdir -m 700 "$TASK/bootstrap" "$TASK/project" "$TASK/output"\n'
scripts_mkdir = 'mkdir -m 700 "$TASK/project/scripts" "$TASK/project/scripts/audio_enhancement"\n'
if q00_command.count(late_project_guard) != 1 or q00_command.count(task_project_mkdir) != 1 or q00_command.count(scripts_mkdir) != 1:
    raise SystemExit("q00 source command no longer has the exact expected project guard and mkdir sequence")
q00_command = q00_command.replace(late_project_guard, "", 1)
q00_command = q00_command.replace(task_project_mkdir + scripts_mkdir, task_project_mkdir + early_project_guard + scripts_mkdir, 1)
if q00_command.index(early_project_guard) > q00_command.index(scripts_mkdir):
    raise SystemExit("empty project guard still follows nested project directory creation")
new_q00 = dict(old_q00)
new_q00["command"] = q00_command

# The pinned actual candidate launcher roots the repository at the directory
# containing runs/ and scripts/, not at the experiment-specific subdirectory.
old_wrong_arg = "--repo-root /root/tts-research-system-20261001/cross_tfg_audio_head_20261003 \\\n"
new_correct_arg = "--repo-root /root/tts-research-system-20261001 \\\n"
qrun_command = old_qrun["command"]
if qrun_command.count(old_wrong_arg) != 1:
    raise SystemExit("attempt01 q_run repo-root argument differs from the recorded old value")
qrun_command = qrun_command.replace(old_wrong_arg, new_correct_arg, 1)
new_qrun = dict(old_qrun)
new_qrun["command"] = qrun_command

new_requests: list[dict[str, Any]] = []
for old_ref in old_index["requests"]:
    old_path = pathlib.Path(old_ref["path"])
    old_raw = old_path.read_bytes()
    if len(old_raw) != old_ref["bytes"] or sha(old_raw) != old_ref["sha256"]:
        raise SystemExit(f"attempt01 request pin mismatch: {old_ref['name']}")
    obj = json.loads(old_raw)
    if old_ref["name"] == "q00_create_fresh_cpu_preflight_namespace":
        obj = new_q00
    elif old_ref["name"] == "q_run_strict_cpu_preflight":
        obj = new_qrun
    elif old_ref["name"] == "q_download_cpu_preflight_receipt":
        obj["local"] = str((NEW_DOWNLOAD_DIR / "cpu_preflight_receipt_attempt03.json").resolve())
    new_path = NEW_REQUEST_DIR / old_path.name
    if old_ref["name"] == "q00_create_fresh_cpu_preflight_namespace":
        new_path = NEW_REQUEST_DIR / "q00_create_fresh_cpu_preflight_namespace_revision02.json"
    elif old_ref["name"] == "q_run_strict_cpu_preflight":
        new_path = NEW_REQUEST_DIR / "q_run_strict_cpu_preflight_revision02.json"
    raw = write_json(new_path, obj)
    new_requests.append({"name": old_ref["name"], "op": obj["op"], "path": str(new_path.resolve()), "bytes": len(raw), "sha256": sha(raw)})

for changed_name, old_command, new_command in (
    ("q00", old_q00["command"], new_q00["command"]),
    ("q_run", old_qrun["command"], new_qrun["command"]),
):
    if old_command == new_command:
        raise SystemExit(f"expected {changed_name} request correction did not occur")

new_manifest = dict(old_manifest)
new_manifest["attempt"] = "candidate_v3_cpu_preflight_attempt03_request_revision02"
new_manifest["remote_roots"] = dict(old_manifest["remote_roots"])
new_manifest["remote_roots"]["repo_root"] = expected_root
new_manifest["remote_roots"].pop("project_root", None)
new_manifest["repo_root_derivation"] = {
    "status": "PINNED_FROM_ACTUAL_CANDIDATE_LAUNCHER_AND_CONFIG",
    "repo_root": expected_root,
    "launcher_source": launcher_pin,
    "launcher_root_literal": expected_root,
    "launcher_material_config_relative_path": "runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/v3_method_candidate_config.json",
    "config03_provenance_paths": {name: value["path"] for name, value in sorted(provenance.items())},
    "relative_path_anchor": "The actual candidate launcher sets ROOT to the repo root and validates materialization candidate_config.path against METHOD_CONFIG.relative_to(ROOT); config03 provenance paths are rooted below runs/.",
}
new_manifest["source_evidence"] = {**old_manifest["source_evidence"],
    "previous_request_index_attempt01": source_pins["previous_request_index_attempt01"],
    "previous_source_manifest_attempt01": source_pins["previous_source_manifest_attempt01"],
    "candidate_detached_launcher_attempt10": launcher_pin,
}
new_manifest["request_names"] = [r["name"] for r in new_requests]
new_manifest.pop("manifest_sha256", None)
new_manifest["manifest_sha256"] = sha(canonical(new_manifest))
new_manifest_raw = write_json(NEW_MANIFEST_PATH, new_manifest)

source_evidence = {key: dict(value) for key, value in source_pins.items()}
source_evidence["source_manifest_attempt02"] = local_pin(NEW_MANIFEST_PATH)
source_evidence["packet_revisioner_attempt02"] = local_pin(pathlib.Path(__file__))

new_index = dict(old_index)
new_index["attempt"] = "candidate_v3_cpu_preflight_attempt03_request_revision02"
new_index["repo_root"] = expected_root
new_index["project_root"] = expected_root
new_index["repo_root_derivation"] = new_manifest["repo_root_derivation"]
new_index["candidate_manifest"] = dict(old_index["candidate_manifest"])
new_index["candidate_manifest"]["manifest_sha256"] = "420d73cd5fb95692ded1c8bbd8cd82363ccb40169a064f21121be70f041909e5"
new_index["q00_ordering_fix"] = "The TASK/project emptiness guard now runs immediately after creating TASK/project and before creating TASK/project/scripts or audio_enhancement."
new_index["output"] = dict(old_index["output"])
new_index["output"]["download_local"] = str((NEW_DOWNLOAD_DIR / "cpu_preflight_receipt_attempt03.json").resolve())
new_index["requests"] = new_requests
new_index["source_manifest"] = {"path": str(NEW_MANIFEST_PATH.resolve()), "bytes": len(new_manifest_raw), "raw_sha256": sha(new_manifest_raw), "manifest_sha256": new_manifest["manifest_sha256"]}
new_index["source_evidence"] = source_evidence
new_index.pop("index_sha256", None)
new_index["index_sha256"] = sha(canonical(new_index))
new_index_raw = write_json(NEW_INDEX_PATH, new_index)

print(json.dumps({
    "index": {"path": str(NEW_INDEX_PATH.resolve()), "bytes": len(new_index_raw), "raw_sha256": sha(new_index_raw), "self_sha256": new_index["index_sha256"]},
    "source_manifest": {"path": str(NEW_MANIFEST_PATH.resolve()), "bytes": len(new_manifest_raw), "raw_sha256": sha(new_manifest_raw), "self_sha256": new_manifest["manifest_sha256"]},
    "repo_root": expected_root,
    "q00_request": next(x for x in new_requests if x["name"] == "q00_create_fresh_cpu_preflight_namespace"),
    "q_run_request": next(x for x in new_requests if x["name"] == "q_run_strict_cpu_preflight"),
    "requests": len(new_requests),
    "remote_action_performed": False,
}, sort_keys=True))
