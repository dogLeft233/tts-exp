#!/usr/bin/env python3
"""Preflight and, only with an explicit switch, run the frozen same-host Dev360 controls.

Preflight verifies the exact plan/runtime receipt, current GPU and cgroup-v2/v1
resource identity, source/model/scorer files, and declared capacity gates. It
does not open audio or video. Execution requires a same-host preflight receipt
and --execute-controls; it delegates generation and scoring to the qualified
Stage-A core plus its existing cross-frontend adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
STAGEA_REL = BASE + "/run_stagea_dev_attempt03.py"
STAGEA_BYTES = 67_141
STAGEA_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_ADAPTER_REL = BASE + "/run_cross_frontend_dev_attempt03.py"
CROSS_ADAPTER_BYTES = 7_505
CROSS_ADAPTER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
STAGEA_DISPATCHER_REL = BASE + "/run_cross_frontend_samehost_dev_attempt01.py"
STAGEA_DISPATCHER_BYTES = 2_724
STAGEA_DISPATCHER_SHA256 = "48345db1197dccf047c48061a95e9db505e72795e3daeb51fd98616afef63f59"
RESOURCE_HELPER_REL = BASE + "/v3_dev180_newhost_latebind_packet_attempt02.py"
RESOURCE_HELPER_BYTES = 62_830
RESOURCE_HELPER_SHA256 = "406f95e4a262fb62bb02bc67249e6a7bf211b803a014622e042329d772fdf7"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
QC06 = {
    "preflight_raw_sha256": "c0dd743e3b9b250c88d8ffade8cdaa8c21c18411c40647264f9e9ee5779b3171",
    "terminal_raw_sha256": "a4fa225dc299f1b89c5cd8fc20b915df7f613c51b491280624895eef052f7269",
    "manifest_raw_sha256": "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436",
    "determinism_policy_sha256": "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc",
    "source_sha256": "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d",
}
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHODS = ("identity", "rms_fixed")
ATTESTATION_SCHEMA = "cross_tfg_v3_samehost_dev360_preflight_v1"
MAX_DEPENDENCY_BYTES = 16 * 1024 * 1024 * 1024
STRICT_BOOTSTRAP_REL = BASE + "/runtime_determinism_attempt03/sitecustomize.py"
STRICT_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
STRICT_BOOTSTRAP_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}
MUSE_WORKSPACE = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace"
MUSE_ALLOWED_SYMLINKS = {
    MUSE_WORKSPACE + "/models/musetalkV15/unet.pth": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/musetalk/musetalkV15/unet.pth",
    MUSE_WORKSPACE + "/models/musetalkV15/musetalk.json": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code/assets/musetalkV15/musetalk.json",
    MUSE_WORKSPACE + "/models/sd-vae-ft-mse": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/musetalk/sd-vae",
    MUSE_WORKSPACE + "/models/whisper": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/musetalk/whisper",
    MUSE_WORKSPACE + "/models/face-parse-bisent/79999_iter.pth": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt09/staging/assembled_assets/79999_iter.pth",
    MUSE_WORKSPACE + "/models/face-parse-bisent/resnet18-5c106cde.pth": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt09/staging/assembled_assets/resnet18-5c106cde.pth",
}
MUSE_FILE_PINS = {
    MUSE_WORKSPACE + "/models/musetalkV15/unet.pth": (3_400_074_924, "7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007"),
    MUSE_WORKSPACE + "/models/musetalkV15/musetalk.json": (748, "5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47"),
    MUSE_WORKSPACE + "/models/sd-vae-ft-mse/config.json": (547, "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e"),
    MUSE_WORKSPACE + "/models/sd-vae-ft-mse/diffusion_pytorch_model.bin": (334_707_217, "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"),
    MUSE_WORKSPACE + "/models/whisper/config.json": (1_983, "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98"),
    MUSE_WORKSPACE + "/models/whisper/preprocessor_config.json": (184_990, "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"),
    MUSE_WORKSPACE + "/models/whisper/pytorch_model.bin": (151_095_027, "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d"),
    MUSE_WORKSPACE + "/models/face-parse-bisent/79999_iter.pth": (53_289_463, "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567"),
    MUSE_WORKSPACE + "/models/face-parse-bisent/resnet18-5c106cde.pth": (46_827_520, "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"),
}
STRICT_SHIM_SOURCE_REL = {
    MODELS[1]: BASE + "/musetalk_five_arg_shim_attempt07.py",
    MODELS[2]: BASE + "/ditto_five_arg_shim_attempt06.py",
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


def safe_regular(path: Path, label: str, limit: int = 16 * 1024 * 1024) -> tuple[Path, bytes]:
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
    return lexical.resolve(strict=True), raw


def _stat_identity(path: Path, *, follow: bool) -> dict[str, int]:
    info = path.stat(follow_symlinks=follow)
    return {"device": int(info.st_dev), "inode": int(info.st_ino),
            "mode": int(info.st_mode), "mtime_ns": int(info.st_mtime_ns), "size": int(info.st_size)}


def _symlink_identity(path: Path, expected_target: str) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    before = lexical.lstat()
    if not stat.S_ISLNK(before.st_mode):
        raise ValueError(f"approved Muse alias is not a literal symlink: {lexical}")
    resolved = lexical.resolve(strict=True)
    if str(resolved) != expected_target:
        raise ValueError(f"approved Muse alias resolves outside its frozen target: {lexical}")
    after = lexical.lstat()
    target_info = resolved.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
        after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size
    ):
        raise ValueError(f"approved Muse alias changed while inspected: {lexical}")
    return {
        "path": str(lexical), "target_path": str(resolved), "link_text": os.readlink(lexical),
        "link_identity": {"device": int(before.st_dev), "inode": int(before.st_ino),
                          "mode": int(before.st_mode), "mtime_ns": int(before.st_mtime_ns),
                          "size": int(before.st_size)},
        "target_identity": {"device": int(target_info.st_dev), "inode": int(target_info.st_ino),
                            "mode": int(target_info.st_mode), "mtime_ns": int(target_info.st_mtime_ns),
                            "size": int(target_info.st_size)},
    }


def validate_muse_alias_binding(backend: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    aliases = backend.get("approved_symlink_aliases")
    if not isinstance(aliases, dict) or set(aliases) != set(MUSE_ALLOWED_SYMLINKS):
        raise ValueError("Muse runtime must bind exactly the six qualified literal symlink aliases")
    observed: dict[str, dict[str, Any]] = {}
    for path_text, target in MUSE_ALLOWED_SYMLINKS.items():
        expected = aliases.get(path_text)
        if not isinstance(expected, dict) or expected.get("target_path") != target:
            raise ValueError(f"Muse symlink binding target differs from the qualified alias map: {path_text}")
        current = _symlink_identity(Path(path_text), target)
        if current != expected:
            raise ValueError(f"Muse symlink/current root identity differs from its runtime receipt: {path_text}")
        observed[path_text] = current
    dep_by_path = {item.get("path"): item for item in backend.get("dependencies", [])
                   if isinstance(item, dict)}
    for path_text, (size, digest) in MUSE_FILE_PINS.items():
        pin = dep_by_path.get(path_text)
        if not isinstance(pin, dict) or pin.get("bytes") != size or pin.get("sha256") != digest:
            raise ValueError(f"Muse alias file is absent from exact fixed dependency pins: {path_text}")
    return observed


def validate_backend_determinism_qualifications(binding: Mapping[str, Any]) -> dict[str, Any]:
    qualifications = binding.get("backend_determinism_qualifications")
    if not isinstance(qualifications, dict) or set(qualifications) != set(MODELS):
        raise ValueError("runtime binding lacks exact strict-repeat qualification receipts for all three backends")
    required_equal_gates = ("generated_video_sha256_equal", "score_npz_sha256_equal",
                            "crop_video_sha256_equal", "tracks_pickle_sha256_equal",
                            "face_bbox_arrays_sha256_equal")
    verified: dict[str, Any] = {}
    for model_id in MODELS:
        entry = qualifications.get(model_id)
        if not isinstance(entry, dict) or entry.get("status") != "PASS_EXACT_MATCH":
            raise ValueError(f"backend lacks a passing same-GPU strict deterministic repeat: {model_id}")
        if (entry.get("schema") != "cross_tfg_backend_strict_determinism_qualification_v1"
                or entry.get("model_id") != model_id or entry.get("gpu_uuid") != GPU_UUID
                or entry.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
                or entry.get("policy_values") != STRICT_BOOTSTRAP_POLICY
                or not isinstance(entry.get("input_context_sha256"), str)
                or len(entry["input_context_sha256"]) != 64):
            raise ValueError(f"backend deterministic repeat receipt has wrong model/GPU/policy/input binding: {model_id}")
        repeat = entry.get("repeat_gate")
        if (not isinstance(repeat, dict) or repeat.get("status") != "PASS_EXACT_MATCH"
                or any(repeat.get(key) is not True for key in required_equal_gates)):
            raise ValueError(f"backend strict repeat did not match video, SyncNet matrix, crops, tracks, and boxes: {model_id}")
        receipt_ref = entry.get("receipt")
        if not isinstance(receipt_ref, dict) or not isinstance(receipt_ref.get("path"), str):
            raise ValueError(f"backend deterministic repeat receipt path is missing: {model_id}")
        receipt_path = Path(receipt_ref["path"])
        if not receipt_path.is_absolute() or type(receipt_ref.get("bytes")) is not int or receipt_ref["bytes"] <= 0:
            raise ValueError(f"backend deterministic repeat receipt pin is invalid: {model_id}")
        receipt_obj, receipt_raw = read_json(receipt_path, f"{model_id} deterministic repeat receipt",
                                              str(receipt_ref.get("sha256", "")), 16 * 1024 * 1024)
        if (len(receipt_raw) != receipt_ref["bytes"]
                or receipt_obj.get("schema") != "cross_tfg_backend_strict_determinism_qualification_v1"
                or receipt_obj.get("status") != "PASS_EXACT_MATCH"
                or receipt_obj.get("model_id") != model_id
                or receipt_obj.get("gpu_uuid") != GPU_UUID
                or receipt_obj.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
                or receipt_obj.get("policy_values") != STRICT_BOOTSTRAP_POLICY
                or receipt_obj.get("input_context_sha256") != entry["input_context_sha256"]
                or receipt_obj.get("repeat_gate") != repeat
                or receipt_obj.get("receipt_sha256") != payload_sha(receipt_obj, "receipt_sha256")):
            raise ValueError(f"backend deterministic repeat receipt bytes/content mismatch: {model_id}")
        verified[model_id] = {"path": str(receipt_path), "bytes": len(receipt_raw),
                              "sha256": sha_bytes(receipt_raw),
                              "input_context_sha256": entry["input_context_sha256"],
                              "repeat_gate": repeat}
    return verified


def hash_regular(path: Path, label: str, expected_sha: str, expected_bytes: int | None = None,
                 *, allowed_aliases: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    alias_map = allowed_aliases or {}
    cursor = Path(lexical.anchor)
    used_aliases: list[str] = []
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            alias = alias_map.get(str(cursor))
            target = MUSE_ALLOWED_SYMLINKS.get(str(cursor))
            if alias is None or target is None or _symlink_identity(cursor, target) != alias:
                raise ValueError(f"{label} traverses an unapproved or changed symlink: {cursor}")
            used_aliases.append(str(cursor))
        elif getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses an unapproved junction: {cursor}")
    resolved = lexical.resolve(strict=True)
    before = resolved.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
        raise ValueError(f"{label} is not a regular file: {lexical}")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ValueError(f"{label} byte count differs from frozen runtime metadata")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = resolved.stat(follow_symlinks=False)
    actual = digest.hexdigest()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or actual != expected_sha.lower()):
        raise ValueError(f"{label} changed or differs from its SHA-256 pin: {lexical}")
    return {"path": str(lexical), "resolved_path": str(resolved), "bytes": before.st_size,
            "sha256": actual, "approved_symlink_aliases": used_aliases}


def read_json(path: Path, label: str, expected_sha: str | None = None,
              limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    _, raw = safe_regular(path, label, limit)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def load_resource_helper(project_root: Path) -> Any:
    path = project_root / RESOURCE_HELPER_REL
    raw = path.read_bytes()
    if len(raw) != RESOURCE_HELPER_BYTES or sha_bytes(raw) != RESOURCE_HELPER_SHA256:
        raise ValueError("pinned cgroup-aware resource helper differs from the reviewed source")
    spec = importlib.util.spec_from_file_location("samehost_dev360_resource_helper_attempt02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the cgroup-aware resource helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_plan(project_root: Path, plan_path: Path, plan_sha: str,
                  rows_path: Path, rows_sha: str, manifest_path: Path,
                  manifest_sha: str, runtime_sha: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    plan, plan_raw = read_json(plan_path, "frozen controls plan", plan_sha)
    manifest, manifest_raw = read_json(manifest_path, "controls plan manifest", manifest_sha, 4 * 1024 * 1024)
    _, rows_raw = safe_regular(rows_path, "controls planned rows", 32 * 1024 * 1024)
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("schema_version") != "cross-tfg-audio-eval-v1" or len(plan.get("rows", [])) != 360):
        raise ValueError("controls plan status/payload/schema/denominator mismatch")
    if rows != plan["rows"] or rows_sha.lower() != sha_bytes(rows_raw) or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw:
        raise ValueError("planned row ledger does not byte-bind the exact frozen plan")
    if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_plan_v1"
            or manifest.get("status") != "FROZEN"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or manifest.get("planned_rows") != 360):
        raise ValueError("controls plan manifest does not bind this plan and 360 rows")
    binding = plan.get("execution", {}).get("samehost_runtime_binding", {})
    if binding.get("raw_sha256") != runtime_sha.lower():
        raise ValueError("frozen plan is not bound to the supplied current-host runtime receipt")
    if tuple(m.get("model_id") for m in plan.get("models", [])) != MODELS:
        raise ValueError("frozen model roster differs from the exact qualified W2L/MuseTalk/Ditto roster")
    if tuple(m.get("method_id") for m in plan.get("methods", [])) != METHODS:
        raise ValueError("frozen control method roster is not identity/RMS")
    if plan.get("protocol", {}).get("generator", {}).get("child_script_sha256") != QC06["source_sha256"]:
        raise ValueError("same-host plan does not use the exact strict QC06 Wav2Lip child bootstrap source")
    strict_plan = plan.get("execution", {}).get("strict_renderer_determinism", {})
    if (strict_plan.get("policy_values") != STRICT_BOOTSTRAP_POLICY
            or strict_plan.get("qualified_policy_reference_sha256") != QC06["determinism_policy_sha256"]
            or strict_plan.get("bootstrap_source", {}).get("path") != BASE + "/runtime_determinism_attempt03/sitecustomize.py"
            or strict_plan.get("bootstrap_source", {}).get("raw_sha256") != STRICT_BOOTSTRAP_SHA256
            or strict_plan.get("per_row_runtime_receipts_required") is not True):
        raise ValueError("same-host plan lacks the strict renderer bootstrap/receipt protocol")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise ValueError("frozen plan does not keep Confirm sealed")
    source_refs = plan.get("source_reference_plans", {})
    for key, rel, size, digest in (
        ("stagea_core", STAGEA_REL, STAGEA_BYTES, STAGEA_SHA256),
        ("cross_frontend_adapter", CROSS_ADAPTER_REL, CROSS_ADAPTER_BYTES, CROSS_ADAPTER_SHA256),
        ("samehost_stagea_runner", STAGEA_DISPATCHER_REL, STAGEA_DISPATCHER_BYTES, STAGEA_DISPATCHER_SHA256),
    ):
        reference = source_refs.get(key, {})
        if reference.get("path") != rel or reference.get("bytes") != size or reference.get("raw_sha256") != digest:
            raise ValueError(f"frozen plan source reference differs from {key}")
        hash_regular(project_root / rel, key, digest, size)
    expected_closure = {
        "samehost_dev360_plan_builder": BASE + "/build_v3_samehost_dev360_controls_plan_attempt07.py",
        "samehost_dev360_guarded_launcher": BASE + "/run_v3_samehost_dev360_controls_attempt06.py",
        "samehost_dev360_evidence_collector": BASE + "/collect_v3_samehost_dev360_controls_evidence_attempt03.py",
        "samehost_dev360_raw_media_collector": BASE + "/collect_v3_stagea_raw_media_evidence_attempt02.py",
        "samehost_v3_baseline_reader_adapter": BASE + "/build_v3_candidate_samehost_controls_adapter_attempt01.py",
        "samehost_dynamic_540_joiner": BASE + "/join_v3_candidate_540_samehost_controls_attempt03.py",
        "samehost_strict_determinism_bootstrap": BASE + "/runtime_determinism_attempt03/sitecustomize.py",
        "samehost_muse_strict_runtime_shim": BASE + "/musetalk_five_arg_shim_attempt07.py",
        "samehost_ditto_strict_runtime_shim": BASE + "/ditto_five_arg_shim_attempt06.py",
        "step120_method_materializer": BASE + "/materialize_v3_step120_method_candidate_attempt02.py",
        "step120_samehost_plan_rebinder": BASE + "/rebind_v3_step120_candidate_plan_samehost_attempt03.py",
        "step120_latebind_packet": BASE + "/v3_dev180_newhost_latebind_packet_attempt02.py",
    }
    closure = plan.get("execution", {}).get("samehost_dev360_source_closure")
    if not isinstance(closure, dict) or set(closure) != set(expected_closure):
        raise ValueError("frozen plan does not contain the full same-host Dev360/step120 source closure")
    for key, relative in expected_closure.items():
        pin = closure[key]
        if not isinstance(pin, dict) or pin.get("path") != relative:
            raise ValueError(f"same-host source-closure path differs from its reviewed route: {key}")
        hash_regular(project_root / relative, key, str(pin.get("raw_sha256", "")), pin.get("bytes"))
    return plan, plan_raw, manifest, manifest_raw


def validate_bindings(project_root: Path, plan: Mapping[str, Any], binding: dict[str, Any],
                      helper: Any, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS":
        raise ValueError("runtime binding must be a current-host PASS receipt")
    if binding.get("gpu_uuid") != GPU_UUID or binding.get("qc06_source_sha256") != QC06["source_sha256"]:
        raise ValueError("runtime binding GPU/QC06 source identity mismatch")
    if binding.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]:
        raise ValueError("runtime binding determinism policy differs from QC06")
    qc_receipts = binding.get("qc06_receipts", {})
    for key, digest in QC06.items():
        if qc_receipts.get(key) != digest:
            raise ValueError(f"runtime binding does not cite the exact QC06 receipt {key}")
    backend_repeat_receipts = validate_backend_determinism_qualifications(binding)
    resources = binding.get("resources")
    if not isinstance(resources, dict) or resources.get("storage_gate_status") != "PASS":
        raise ValueError("runtime resource/capacity binding is incomplete or pending")
    if resources.get("gpu_uuid") != GPU_UUID:
        raise ValueError("runtime resource receipt is bound to another GPU")
    for key in ("min_gpu_free_mib", "min_effective_ram_available_bytes", "min_dev_shm_free_bytes",
                "output_required_bytes", "output_reserve_bytes", "output_free_bytes",
                "effective_ram_available_bytes", "dev_shm_free_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise ValueError(f"runtime resource field {key} must be an explicit nonnegative integer")
    if resources["output_free_bytes"] < resources["output_required_bytes"] + resources["output_reserve_bytes"]:
        raise ValueError("runtime output capacity does not include required bytes plus reserve")
    basis = resources.get("capacity_basis")
    if not isinstance(basis, dict) or basis.get("status") != "MEASURED_OR_EXPLICITLY_BOUNDED":
        raise ValueError("capacity basis must be measured or explicitly bounded; unknown media bytes cannot pass")
    basis_sha = sha_bytes(canonical(basis))
    if resources.get("capacity_basis_sha256") != basis_sha:
        raise ValueError("capacity-basis payload and raw SHA do not match")
    cgroup = resources.get("cgroup")
    if not isinstance(cgroup, dict) or cgroup.get("version") not in ("v1", "v2"):
        raise ValueError("runtime cgroup identity is missing")

    exec_binding = binding.get("execution_bindings")
    if not isinstance(exec_binding, dict):
        raise ValueError("runtime receipt lacks explicit Stage-A tool and output bindings")
    required = ("output_root", "stagea_python", "syncnet_python", "generator_child",
                "syncnet_root", "syncnet_evaluator", "syncnet_checkpoint", "ffmpeg", "ffprobe",
                "pcm16_lineage", "wav2lip_python", "wav2lip_root", "strict_determinism_bootstrap")
    for key in required:
        value = exec_binding.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"runtime execution binding {key} must be an explicit absolute path")
    expected_hashes = binding.get("expected_file_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("runtime binding must contain exact tool-file SHA pins")
    tool_hashes = {
        "generator_child": plan.get("protocol", {}).get("generator", {}).get("child_script_sha256"),
        "strict_determinism_bootstrap": STRICT_BOOTSTRAP_SHA256,
        "syncnet_evaluator": plan.get("execution", {}).get("official_syncnet_evaluator_wrapper_sha256"),
        "syncnet_checkpoint": plan.get("protocol", {}).get("syncnet", {}).get("score_model_sha256"),
        "ffmpeg": plan.get("protocol", {}).get("syncnet", {}).get("primary_audio_input", {}).get("ffmpeg_sha256"),
        "pcm16_lineage": plan.get("execution", {}).get("primary_pcm16_lineage_code_sha256"),
    }
    for key, expected in tool_hashes.items():
        if not isinstance(expected, str) or expected_hashes.get(key) != expected:
            raise ValueError(f"execution tool pin does not match the frozen plan: {key}")
        hash_regular(Path(exec_binding[key]), key, expected)
    if Path(exec_binding["strict_determinism_bootstrap"]).resolve(strict=True) != (
            project_root / STRICT_BOOTSTRAP_REL).resolve(strict=True):
        raise ValueError("runtime bootstrap path differs from the pinned same-host source")
    hash_regular(Path(exec_binding["syncnet_checkpoint"]), "SyncNet checkpoint",
                 expected_hashes["syncnet_checkpoint"])
    ffprobe_sha = expected_hashes.get("ffprobe")
    if not isinstance(ffprobe_sha, str) or len(ffprobe_sha) != 64:
        raise ValueError("runtime binding must pin the FFprobe executable SHA-256")
    hash_regular(Path(exec_binding["ffprobe"]), "FFprobe executable", ffprobe_sha)
    for key in ("stagea_python", "syncnet_python", "wav2lip_python"):
        if not Path(exec_binding[key]).is_file():
            raise FileNotFoundError(f"runtime Python interpreter is absent: {key}")
    for key in ("syncnet_root", "wav2lip_root"):
        if not Path(exec_binding[key]).is_dir():
            raise FileNotFoundError(f"runtime source/model directory is absent: {key}")

    backend_map = binding.get("backends")
    if not isinstance(backend_map, dict) or set(backend_map) != set(MODELS):
        raise ValueError("runtime backends must contain exactly the three frozen models")
    muse_aliases = validate_muse_alias_binding(backend_map[MODELS[1]])
    dependencies_by_path: dict[str, dict[str, Any]] = {}
    python_paths: dict[str, str] = {
        "stagea": exec_binding["stagea_python"],
        "syncnet": exec_binding["syncnet_python"],
    }
    backend_directories: dict[str, dict[str, Any]] = {}
    for model in plan["models"]:
        model_id = model["model_id"]
        backend = backend_map[model_id]
        if backend.get("model_binding") != {key: model.get(key) for key in (
                "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
                "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den")}:
            raise ValueError(f"current runtime backend model/scorer identity changed: {model_id}")
        generator = backend.get("generator")
        if not isinstance(generator, dict) or generator.get("fps_num") != model.get("fps_num") or generator.get("fps_den") != model.get("fps_den"):
            raise ValueError(f"current runtime generator descriptor changed frame rate/model: {model_id}")
        generator_python = generator.get("python")
        if not isinstance(generator_python, str) or not Path(generator_python).is_absolute() or not Path(generator_python).is_file():
            raise ValueError(f"runtime generator Python is absent or not absolute: {model_id}")
        if model_id in STRICT_SHIM_SOURCE_REL:
            shim_path = (project_root / STRICT_SHIM_SOURCE_REL[model_id]).resolve(strict=True)
            shim_raw = read_regular(shim_path, f"{model_id} strict runtime shim", 4 * 1024 * 1024)
            shim_sha = sha_bytes(shim_raw)
            if (generator.get("adapter_path") != str(shim_path)
                    or generator.get("adapter_sha256") != shim_sha):
                raise ValueError(f"runtime {model_id} adapter must use the exact new strict-policy shim path/SHA")
            shim_dep = [dep for dep in backend.get("dependencies", []) if isinstance(dep, dict)
                        and dep.get("path") == str(shim_path) and dep.get("sha256") == shim_sha
                        and dep.get("bytes") == len(shim_raw)]
            if len(shim_dep) != 1:
                raise ValueError(f"runtime {model_id} dependency inventory lacks its exact strict-policy shim")
        if model_id in MODEL_ENTRY_PINS:
            expected_entry, expected_entry_sha = MODEL_ENTRY_PINS[model_id]
            if generator.get("render_entry_path") != expected_entry:
                raise ValueError(f"runtime native renderer entry differs from its qualified path: {model_id}")
            dependency_match = any(isinstance(dep, dict) and dep.get("path") == expected_entry
                                   and dep.get("sha256") == expected_entry_sha
                                   for dep in backend.get("dependencies", []))
            if not dependency_match:
                raise ValueError(f"native renderer entry is absent from fixed source/model pins: {model_id}")
        backend_key = {MODELS[0]: "wav2lip", MODELS[1]: "muse", MODELS[2]: "ditto"}[model_id]
        python_paths[backend_key] = generator_python
        if model_id == MODELS[0] and Path(generator_python).resolve(strict=True) != Path(exec_binding["wav2lip_python"]).resolve(strict=True):
            raise ValueError("Wav2Lip execution interpreter differs from the frozen backend descriptor")

        directory_values: dict[str, str] = {}
        for key in ("cwd", "repo", "model_source_root", "ditto_source_root", "model_code_root",
                    "model_workspace", "ditto_weights_root"):
            value = generator.get(key)
            if isinstance(value, str) and value:
                directory_values[key] = value
        for key in ("model_overlay_roots",):
            values = generator.get(key)
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str) and value:
                        directory_values[f"{key}[{index}]"] = value
        env_overrides = generator.get("env", {})
        pythonpath = env_overrides.get("PYTHONPATH") if isinstance(env_overrides, Mapping) else None
        if isinstance(pythonpath, str):
            for index, value in enumerate(item for item in pythonpath.split(os.pathsep) if item):
                directory_values[f"PYTHONPATH[{index}]"] = value
        directory_records = {key: helper.directory_identity(Path(value), f"{model_id} {key}")
                             for key, value in sorted(directory_values.items())}
        backend_directories[model_id] = directory_records
        dependencies = backend.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"runtime backend dependency inventory is empty: {model_id}")
        for item in dependencies:
            if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256", "role"}:
                raise ValueError(f"backend dependency receipt schema mismatch: {model_id}")
            path = Path(str(item["path"]))
            if not path.is_absolute() or type(item["bytes"]) is not int or item["bytes"] < 0 or not isinstance(item["role"], str):
                raise ValueError(f"backend dependency receipt has invalid path/size/role: {model_id}")
            previous = dependencies_by_path.get(str(path))
            identity = {"path": str(path), "bytes": item["bytes"], "sha256": item["sha256"]}
            if previous and (previous["bytes"], previous["sha256"]) != (identity["bytes"], identity["sha256"]):
                raise ValueError(f"one dependency path has conflicting pins: {path}")
            dependencies_by_path[str(path)] = identity
    expected_directories = binding.get("backend_directories")
    if expected_directories != backend_directories:
        raise ValueError("current backend directory path/device/inode/mtime snapshots differ from the runtime binding")
    expected_pythons = binding.get("python_environments")
    if not isinstance(expected_pythons, dict) or set(expected_pythons) != set(python_paths):
        raise ValueError("runtime receipt lacks exact Python environment identities for Stage-A, SyncNet, and all backends")
    actual_pythons = {key: helper.probe_python(value) for key, value in sorted(python_paths.items())}
    if expected_pythons != actual_pythons:
        raise ValueError("current Python executable/version/package identities differ from the runtime receipt")
    dependency_total = sum(item["bytes"] for item in dependencies_by_path.values())
    if dependency_total > MAX_DEPENDENCY_BYTES:
        raise ValueError("runtime backend dependency hash budget exceeds 16 GiB")
    verified_dependencies = []
    for item in sorted(dependencies_by_path.values(), key=lambda value: value["path"]):
        verified_dependencies.append(hash_regular(Path(item["path"]), "backend source/model dependency",
                                                  item["sha256"], item["bytes"],
                                                  allowed_aliases=muse_aliases))

    thresholds = {
        "min_gpu_free_mib": resources["min_gpu_free_mib"],
        "min_ram_available_bytes": resources["min_effective_ram_available_bytes"],
        "min_shm_free_bytes": resources["min_dev_shm_free_bytes"],
        "min_output_free_bytes": resources["output_required_bytes"] + resources["output_reserve_bytes"],
    }
    current = helper.resource_snapshot(output_root, thresholds)
    gpu = next((item for item in current["gpus"] if item.get("uuid") == GPU_UUID), None)
    if gpu is None:
        raise ValueError("QC06 physical GPU UUID is absent from current nvidia-smi inventory")
    if current["ram"]["cgroup"].get("version") != cgroup.get("version"):
        raise ValueError("current cgroup v1/v2 resource controller differs from the bound runtime identity")
    stable = current["ram"]["cgroup"].get("identity", {})
    stable_expect = cgroup.get("identity")
    if not isinstance(stable_expect, dict):
        raise ValueError("runtime binding lacks the stable cgroup identity record")
    for key in ("version", "mount_target", "mount_root", "membership_path", "leaf_path", "finite_limit", "limit_bytes"):
        if stable.get(key) != stable_expect.get(key):
            raise ValueError(f"current cgroup identity differs from the bound resource receipt: {key}")
    if current["output_disk"].get("target") != resources.get("output_mount_target"):
        raise ValueError("current output mount target differs from the capacity receipt")
    if current["output_disk"].get("available_bytes", -1) < resources["output_required_bytes"] + resources["output_reserve_bytes"]:
        raise ValueError("current output mount cannot fit declared measured/bounded bytes plus reserve")
    return current, {"dependencies": verified_dependencies, "dependency_total_bytes": dependency_total,
                     "tool_file_pins": tool_hashes, "execution_bindings": exec_binding,
                     "python_environments": actual_pythons,
                     "backend_directories": backend_directories,
                     "backend_determinism_qualifications": backend_repeat_receipts,
                     "approved_muse_symlink_aliases": muse_aliases}


def collect_strict_determinism_receipts(plan: Mapping[str, Any], binding: Mapping[str, Any],
                                        receipt_dir: Path, plan_raw_sha256: str,
                                        runtime_binding_raw_sha256: str) -> dict[str, Any]:
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise ValueError("renderer deterministic receipt directory is missing or unsafe")
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or len(plan_rows) != 360:
        raise ValueError("strict deterministic receipt check requires the frozen 360-row plan")
    expected = {row["row_id"]: row for row in plan_rows}
    if len(expected) != len(plan_rows):
        raise ValueError("frozen strict deterministic row IDs are not unique")
    exec_binding = binding["execution_bindings"]
    backend_map = binding["backends"]
    expected_entries: dict[str, tuple[str, str]] = {
        MODELS[0]: (str(Path(exec_binding["generator_child"]).resolve(strict=True)),
                    str(Path(exec_binding["wav2lip_python"]).resolve(strict=True)),),
    }
    for model_id in MODELS[1:]:
        generator = backend_map[model_id]["generator"]
        expected_entries[model_id] = (
            str(Path(generator["render_entry_path"]).resolve(strict=True)),
            str(Path(generator["python"]).resolve(strict=True)),
        )
    files = sorted(receipt_dir.glob("*.json"))
    if not files:
        raise ValueError("no renderer bootstrap receipts were recorded")
    receipts: list[dict[str, Any]] = []
    matching_by_row: dict[str, list[dict[str, Any]]] = {row_id: [] for row_id in expected}
    for path in files:
        _, raw = safe_regular(path, "strict renderer receipt", 256 * 1024)
        record = json.loads(raw.decode("utf-8"))
        if not isinstance(record, dict) or record.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1":
            raise ValueError(f"renderer bootstrap receipt schema is invalid: {path.name}")
        if (record.get("status") != "PASS" or record.get("receipt_sha256") != payload_sha(record, "receipt_sha256")
                or record.get("bootstrap_source_sha256") != STRICT_BOOTSTRAP_SHA256
                or record.get("policy") != STRICT_BOOTSTRAP_POLICY
                or record.get("policy_payload_sha256") != sha_bytes(canonical(record.get("policy")))
                or record.get("cuda_available") is not True
                or record.get("pid") != int(path.stem)):
            raise ValueError(f"renderer bootstrap receipt does not attest the applied strict policy: {path.name}")
        row_id = record.get("row_id")
        if row_id not in expected:
            raise ValueError(f"renderer bootstrap receipt belongs to an unknown planned row: {path.name}")
        row = expected[row_id]
        model_id = row.get("model_id")
        expected_entry, expected_python = expected_entries[model_id]
        if (record.get("model_id") != model_id or record.get("audio_frontend") != row.get("audio_frontend")
                or record.get("seed") != row.get("seed")
                or record.get("expected_backend_entry") != expected_entry
                or record.get("python_executable") != expected_python):
            raise ValueError(f"renderer bootstrap receipt differs from planned row/runtime binding: {row_id}")
        entry_matches = record.get("entry_path") == expected_entry
        if record.get("entry_matches_backend") is not entry_matches:
            raise ValueError(f"renderer receipt entry-match flag is inconsistent: {row_id}")
        if entry_matches:
            matching_by_row[row_id].append({"path": str(path.resolve(strict=True)),
                                            "bytes": len(raw), "sha256": sha_bytes(raw),
                                            "pid": record["pid"], "entry_path": expected_entry})
        receipts.append({"path": str(path.resolve(strict=True)), "bytes": len(raw),
                         "sha256": sha_bytes(raw), "row_id": row_id,
                         "model_id": model_id, "pid": record["pid"],
                         "entry_matches_backend": entry_matches})
    if any(len(items) != 1 for items in matching_by_row.values()):
        missing = sorted(row_id for row_id, items in matching_by_row.items() if not items)
        duplicate = sorted(row_id for row_id, items in matching_by_row.items() if len(items) > 1)
        raise ValueError(f"strict bootstrap lacks one exact native renderer process per row; missing={missing[:5]} duplicate={duplicate[:5]}")
    result: dict[str, Any] = {
        "schema": "cross_tfg_v3_samehost_renderer_determinism_attestation_v1",
        "status": "PASS",
        "gpu_uuid": GPU_UUID,
        "plan_raw_sha256": plan_raw_sha256,
        "runtime_binding_raw_sha256": runtime_binding_raw_sha256,
        "strict_bootstrap_source_sha256": STRICT_BOOTSTRAP_SHA256,
        "policy_values": STRICT_BOOTSTRAP_POLICY,
        "qualified_policy_reference_sha256": QC06["determinism_policy_sha256"],
        "per_backend_repeat_qualification_receipts": binding.get("backend_determinism_qualifications"),
        "row_count": len(expected),
        "native_renderer_row_receipt_count": sum(len(items) for items in matching_by_row.values()),
        "process_receipt_count": len(receipts),
        "native_renderer_receipts_by_row": {row_id: items[0] for row_id, items in sorted(matching_by_row.items())},
        "process_receipts": receipts,
        "confirm_audio_read": False,
        "confirm_data_opened": False,
    }
    result["attestation_sha256"] = payload_sha(result, "attestation_sha256")
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from this source checkout")
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, plan_manifest, plan_manifest_raw = validate_plan(
        project_root, args.plan, args.plan_raw_sha256, args.planned_rows, args.planned_rows_raw_sha256,
        args.plan_manifest, args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    if binding.get("status") != "PASS":
        raise ValueError("runtime binding must be a passing late-bound receipt")
    helper = load_resource_helper(project_root)
    resources, identities = validate_bindings(project_root, plan, binding, helper, Path(binding["execution_bindings"]["output_root"]))
    output_root = Path(binding["execution_bindings"]["output_root"])
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError("Stage-A output root must not exist or must be empty before execution")
    payload = {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS_METADATA_ONLY",
        "host": platform.node(),
        "created_unix": time.time(),
        "valid_for_seconds": 1800,
        "plan_raw_sha256": sha_bytes(plan_raw),
        "plan_manifest_raw_sha256": sha_bytes(plan_manifest_raw),
        "runtime_binding_raw_sha256": sha_bytes(binding_raw),
        "runtime_gpu_uuid": GPU_UUID,
        "runtime_resources": resources,
        "verified_dependency_files": identities["dependencies"],
        "verified_dependency_bytes": identities["dependency_total_bytes"],
        "verified_tool_pins": identities["tool_file_pins"],
        "verified_python_environments": identities["python_environments"],
        "verified_backend_directories": identities["backend_directories"],
        "execution_bindings": identities["execution_bindings"],
        "development_audio_opened": False,
        "confirm_audio_opened": False,
        "generated_media_opened": False,
        "model_execution_performed": False,
    }
    payload["attestation_sha256"] = payload_sha(payload, "attestation_sha256")
    attestation_path = Path(args.preflight_receipt)
    if attestation_path.exists() or attestation_path.is_symlink():
        raise FileExistsError(f"refusing existing preflight receipt: {attestation_path}")
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(attestation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(payload)); stream.flush(); os.fsync(stream.fileno())
    return {"status": payload["status"], "attestation_path": str(attestation_path.resolve()),
            "attestation_raw_sha256": sha_bytes(attestation_path.read_bytes()),
            "plan_raw_sha256": sha_bytes(plan_raw), "runtime_binding_raw_sha256": sha_bytes(binding_raw),
            "resource_snapshot": resources, "model_execution_performed": False}


def execute(args: argparse.Namespace) -> int:
    if not args.execute_controls:
        raise ValueError("execution requires --execute-controls")
    receipt_sha256 = args.preflight_receipt_raw_sha256
    if (not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in receipt_sha256)):
        raise ValueError("execution requires the exact 64-character preflight receipt raw SHA-256")
    receipt, receipt_raw = read_json(args.preflight_receipt, "same-host controls preflight receipt",
                                    receipt_sha256, 16 * 1024 * 1024)
    if (receipt.get("schema") != ATTESTATION_SCHEMA or receipt.get("status") != "PASS_METADATA_ONLY"
            or receipt.get("attestation_sha256") != payload_sha(receipt, "attestation_sha256")
            or receipt.get("host") != platform.node()
            or time.time() - float(receipt.get("created_unix", 0)) > float(receipt.get("valid_for_seconds", 0))):
        raise ValueError("preflight receipt is invalid, stale, or from another host")
    now = preflight_recheck(args, receipt)
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, _manifest, _manifest_raw = validate_plan(
        Path(args.project_root).resolve(strict=True), args.plan, args.plan_raw_sha256,
        args.planned_rows, args.planned_rows_raw_sha256, args.plan_manifest,
        args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    if (receipt.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or receipt.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or now.get("runtime_resources") is None):
        raise ValueError("preflight does not bind the exact current plan/runtime receipt")
    tools = binding["execution_bindings"]
    out = Path(tools["output_root"])
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Stage-A output root became nonempty after preflight")
    out.mkdir(parents=True, exist_ok=True)
    strict_bootstrap_dir = Path(args.project_root).resolve(strict=True) / Path(STRICT_BOOTSTRAP_REL).parent
    receipt_dir = out / "runtime_determinism_receipts"
    if receipt_dir.exists() or receipt_dir.is_symlink():
        raise FileExistsError("strict renderer receipt directory already exists")
    receipt_dir.mkdir(mode=0o700)
    command = [
        tools["stagea_python"], str(Path(args.project_root) / STAGEA_DISPATCHER_REL),
        "--stagea-core", str(Path(args.project_root) / STAGEA_REL),
        "--cross-frontend-adapter-runner", str(Path(args.project_root) / CROSS_ADAPTER_REL),
        "--plan", str(args.plan.resolve(strict=True)),
        "--planned-rows", str(args.planned_rows.resolve(strict=True)),
        "--expected-plan-raw-sha256", sha_bytes(plan_raw),
        "--output-root", str(out), "--project-root", str(Path(args.project_root).resolve(strict=True)),
        "--wav2lip-python", tools["wav2lip_python"], "--wav2lip-root", tools["wav2lip_root"],
        "--generator-child", tools["generator_child"],
        "--syncnet-root", tools["syncnet_root"], "--syncnet-python", tools["syncnet_python"],
        "--syncnet-checkpoint", tools["syncnet_checkpoint"], "--syncnet-evaluator", tools["syncnet_evaluator"],
        "--ffmpeg", tools["ffmpeg"], "--ffprobe", tools["ffprobe"],
        "--ffmpeg-sha256", binding["expected_file_sha256"]["ffmpeg"],
        "--pcm16-lineage-sha256", binding["expected_file_sha256"]["pcm16_lineage"],
        "--generator-timeout", str(binding.get("timeouts", {}).get("generator_seconds", 900)),
        "--syncnet-timeout", str(binding.get("timeouts", {}).get("syncnet_seconds", 1800)),
        "--ffmpeg-timeout", str(binding.get("timeouts", {}).get("ffmpeg_seconds", 300)),
    ]
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(strict_bootstrap_dir) + os.pathsep + str(Path(args.project_root).resolve(strict=True))
                            + os.pathsep + env.get("PYTHONPATH", ""),
                "CUDA_VISIBLE_DEVICES": str(binding.get("cuda_visible_devices", "0")),
                "PYTHONHASHSEED": "20261003", "TFG_DETERMINISM_SEED": "20261003",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(receipt_dir.resolve()),
                "CROSS_TFG_STRICT_STARTUP_POLICY": "1"})
    receipt_copy = dict(receipt)
    receipt_copy["execution_started_unix"] = time.time()
    receipt_copy["execution_command"] = command
    receipt_copy["execution_output_root"] = str(out.resolve())
    execution_record = out.parent / f"{out.name}.execution.json"
    if execution_record.exists() or execution_record.is_symlink():
        raise FileExistsError(f"refusing existing execution record: {execution_record}")
    execution_record.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(execution_record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt_copy)); stream.flush(); os.fsync(stream.fileno())
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, env=env, check=False)
    receipt_copy["execution_finished_unix"] = time.time()
    receipt_copy["child_returncode"] = completed.returncode
    deterministic = collect_strict_determinism_receipts(
        plan, binding, receipt_dir, sha_bytes(plan_raw), sha_bytes(binding_raw))
    det_path = out / "runtime_determinism_attestation.json"
    fd_det = os.open(det_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd_det, "wb") as stream:
        det_raw = pretty(deterministic)
        stream.write(det_raw); stream.flush(); os.fsync(stream.fileno())
    receipt_copy["strict_determinism_attestation"] = {
        "path": str(det_path.resolve()), "bytes": len(det_raw), "raw_sha256": sha_bytes(det_raw),
        "attestation_sha256": deterministic["attestation_sha256"],
        "status": deterministic["status"],
    }
    fd = os.open(execution_record, os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt_copy)); stream.flush(); os.fsync(stream.fileno())
    return completed.returncode if deterministic["status"] == "PASS" else (completed.returncode or 2)


def preflight_recheck(args: argparse.Namespace, previous: Mapping[str, Any]) -> dict[str, Any]:
    """Re-evaluate runtime gates without creating another receipt or touching media."""
    project_root = Path(args.project_root).resolve(strict=True)
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, _manifest, _manifest_raw = validate_plan(
        project_root, args.plan, args.plan_raw_sha256, args.planned_rows, args.planned_rows_raw_sha256,
        args.plan_manifest, args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    helper = load_resource_helper(project_root)
    resources, _ids = validate_bindings(project_root, plan, binding, helper,
                                        Path(binding["execution_bindings"]["output_root"]))
    if (previous.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or previous.get("plan_manifest_raw_sha256") != sha_bytes(_manifest_raw)
            or previous.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)):
        raise ValueError("preflight receipt belongs to a different plan or runtime binding")
    old_resources = previous.get("runtime_resources", {})
    old_gpu = next((item for item in old_resources.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    new_gpu = next((item for item in resources.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    if old_gpu is None or new_gpu is None or old_gpu.get("uuid") != new_gpu.get("uuid"):
        raise ValueError("physical GPU UUID changed after preflight")
    old_cgroup = old_resources.get("ram", {}).get("cgroup", {}).get("identity", {})
    new_cgroup = resources.get("ram", {}).get("cgroup", {}).get("identity", {})
    if old_cgroup != new_cgroup:
        raise ValueError("cgroup resource-controller identity changed after preflight")
    if resources.get("output_disk", {}).get("available_bytes", -1) < binding["resources"]["output_required_bytes"] + binding["resources"]["output_reserve_bytes"]:
        raise ValueError("current output capacity fell below its measured/bounded requirement plus reserve")
    return {"runtime_resources": resources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--execute-controls", action="store_true", help="explicitly authorize Dev360 rendering/scoring")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--planned-rows-raw-sha256", required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt-raw-sha256")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        result = preflight(args)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
