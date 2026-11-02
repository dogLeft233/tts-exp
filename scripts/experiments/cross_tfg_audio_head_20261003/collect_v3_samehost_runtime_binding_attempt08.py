#!/usr/bin/env python3
"""Collect a current-host, metadata-only runtime binding for Dev360 and step120.

The seed is filled only after backend assets have been restored and all three
strict repeat receipts exist. This collector hashes every frozen runtime file,
records interpreter/directory/resource identities, and writes a self-sealed
PASS binding only when the complete input and capacity gates pass. It does not
open Dev/Confirm audio, inspect images/video, render, score, train, or call APIs.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
ROSTER_REL = "runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json"
ROSTER_BYTES = 997_947
ROSTER_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
BASE_PLAN_REL = "runs/cross_tfg_audio_head_20261003/stagea_dev_plan_attempt01/plan.json"
BASE_PLAN_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
RESTORE_INVENTORY_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_restore_identity_manifest_attempt02.json"
RESTORE_INVENTORY_BYTES = 42_642
RESTORE_INVENTORY_SHA256 = "05df89942010960e3ddfc0a84d27eac7a447c7bfe14c78c684bd136d12dc8c1d"
OLD_TEMPLATE_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_template_attempt05.json"
OLD_TEMPLATE_BYTES = 48_942
OLD_TEMPLATE_SHA256 = "226acf9169d9d9e55734c4199cee01581fbc415aef01dae107c9a6fb376eeeab"
NEW_TEMPLATE_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_template_attempt07.json"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
MODEL_BINDING_FIELDS = (
    "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
    "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den",
)
QC06 = {
    "preflight_raw_sha256": "c0dd743e3b9b250c88d8ffade8cdaa8c21c18411c40647264f9e9ee5779b3171",
    "terminal_raw_sha256": "a4fa225dc299f1b89c5cd8fc20b915df7f613c51b491280624895eef052f7269",
    "manifest_raw_sha256": "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436",
    "determinism_policy_sha256": "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc",
    "source_sha256": "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d",
}
STRICT_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}
STRICT_BOOTSTRAP_REL = BASE + "/runtime_determinism_attempt03/sitecustomize.py"
STRICT_BOOTSTRAP_BYTES = 9_525
STRICT_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
STRICT_SHIMS = {
    MODELS[1]: (BASE + "/musetalk_five_arg_shim_attempt07.py", 25_103,
                "39d75863cdbd94fc8942c395a2ebf16cacbff74e13bed5dbc8cbd62b74239d14"),
    MODELS[2]: (BASE + "/ditto_five_arg_shim_attempt06.py", 20_529,
                "7522e08402dab6dc6386f20e4ef667e97fe6ea62febc232e49953ba99b74ddda"),
}
W2L_NATIVE_ENTRY = "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py"
W2L_NATIVE_ENTRY_SHA256 = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
W2L_RUNTIME_COMPAT_REL = BASE + "/wav2lip_runtime_compat_attempt04/sitecustomize.py"
W2L_RUNTIME_COMPAT_BYTES = 41_074
W2L_RUNTIME_COMPAT_SHA256 = "462884a2534f73d93eb09cd05c87fefda677e3647366360fb81a9622fa7592ca"
W2L_FRAME_PROBE_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/dev20_native_frame_binding_attempt03_w2l_expected_frames.json"
W2L_FRAME_PROBE_BYTES = 72_121
W2L_FRAME_PROBE_SHA256 = "ca1ec73328a2c22eee4260d3890a29d87d394dae6a3d97bba61482993d302223"
W2L_FRAME_PROBE_RECEIPT_SHA256 = "b3099fdcd15a5ab3b32faafba71f9e26f7ffc5fa2aea9914c5bf27f8a95e3adb"
W2L_FACE_DETECTION_PINS = (
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/__init__.py", 183, "7215846a10fcb2162dab619cf1cb45bd6c0faac2b7d1a71ed8cc1cc0811db86a"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/api.py", 2_266, "c050676db49d8eb5de19f13c171101892b60b992b2a730cfbc8534967e4faeaa"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/__init__.py", 30, "1f3233930bfb7ac8b1726e4af8ed9b93925046d16edc640afd218f9d02b6d235"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/core.py", 4_868, "ba16a80fac378da9b5bbfc9c190975ab4fe153b13b0d18d892ab5bbfcbb9c71e"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/__init__.py", 53, "6fbfb62eccc1a14de7c0a8cb1ca6f3c59af42479955ac08acbf548f28ec1feed"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/bbox.py", 4_279, "3ed1487243bbde8ab47c567f46ee4b177a549e31440db8d71376bffc70a7bd86"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/detect.py", 3_769, "1848d75a551131c786a5e10d4ffab517dda606af27913e19ab48467c1a779d71"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/net_s3fd.py", 5_291, "80960aaefd9fb97a5e9784dc0a48bc38f43bdcaacb268782ede16a959e77a5a9"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/sfd_detector.py", 1_809, "155be501e83825f9b0508ee7d596f13047559350adbdbf17fcbf361cfab22e92"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/models.py", 8_619, "2cde93a32467b9c5cfd009499c917afe8c89f512622237c2d865ba8fbb8a37e9"),
    ("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/utils.py", 11_808, "561bfe5ec349a28f086e04efbb2d43cbc45c15fbcac48c311d497cb2bd2efe9a"),
)
W2L_EXTRA_DEPENDENCIES = (
    ("wav2lip-runtime-compat", "@PROJECT_ROOT@/" + W2L_RUNTIME_COMPAT_REL,
     W2L_RUNTIME_COMPAT_BYTES, W2L_RUNTIME_COMPAT_SHA256, "strict_runtime_adapter"),
    *tuple(("wav2lip-face-source:" + path.rsplit("/", 1)[-1] + ":" + str(index), path, size, digest, "backend_source")
           for index, (path, size, digest) in enumerate(W2L_FACE_DETECTION_PINS)),
    ("wav2lip-models-package-init", "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/models/__init__.py",
     82, "d6883f7d5c41623280c9cbce4963ad719347b9d4cdac614cadfff7b544535215", "backend_source"),
    ("wav2lip-sfd-checkpoint", "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
     89_843_225, "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543", "backend_checkpoint"),
)
QUALIFICATION_LINEAGE_PINS = {
    MODELS[0]: {
        "manifest": {"path": "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt12/pilot_manifest_final.json",
                     "bytes": 146_401, "sha256": "c55b390b6d5189983a4a3e2d181d5ff7606cc191817f125a859800c3e09508d9",
                     "run_id": "cross_tfg_backend_strict_repeat_20261008_attempt12"},
        "qualification": {"path": "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt12/backend_qualifications/wav2lip_gan_bac9a81e.qualification.json",
                          "bytes": 125_550, "sha256": "d1b778e4a85ae7861f68683dfa1959abb030c6f0962271ed10ff80c70542645a"},
    },
    MODELS[1]: {
        "manifest": {"path": "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt09/pilot_manifest_final.json",
                     "bytes": 287_148, "sha256": "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d",
                     "run_id": "cross_tfg_backend_strict_repeat_20261008_attempt09"},
        "qualification": {"path": "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt09/backend_qualifications/musetalk_fixed_portrait_attempt16.qualification.json",
                          "bytes": 103_921, "sha256": "38599cc6de038c3d15b3890a6b89ee4397fa9c925dc71dd6f5edf9fb9888b268"},
    },
    MODELS[2]: {
        "manifest": {"path": "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt09/pilot_manifest_final.json",
                     "bytes": 287_148, "sha256": "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d",
                     "run_id": "cross_tfg_backend_strict_repeat_20261008_attempt09"},
        "qualification": {"path": "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt09/backend_qualifications/ditto_hubert_fixed_portrait_attempt18.qualification.json",
                          "bytes": 142_943, "sha256": "218e2856a74aa6aa22c8b3791ef07fcd475e83830d33193a8257ef1941cacc04"},
    },
}
SOURCE_TEMPLATE_SCHEMA = "cross_tfg_v3_samehost_runtime_binding_seed_v1"
RUNTIME_SCHEMA = "cross_tfg_v3_samehost_runtime_binding_v1"
MAX_DEPENDENCY_BYTES = 16 * 1024 * 1024 * 1024
CAPACITY_EVIDENCE_BASE_REL = "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/capacity_evidence"
PILOT_FOOTPRINT_RECEIPT = {
    "path": "cross_tfg_audio_head_20261003/job_control/dev360_resource_footprint_probe_attempt01/output/resource_footprint_receipt_attempt01.json",
    "bytes": 57_746,
    "sha256": "164bff9e05094428af4de0901e4d38e1139754cbfcfa7e693a80251be5479bf7",
    "schema": "cross_tfg_dev360_resource_footprint_probe_attempt01_v1",
    "status": "PASS_READ_ONLY_RESOURCE_AND_PILOT_FOOTPRINT",
}
CAPACITY_ANALYSIS_RECEIPT = {
    "path": CAPACITY_EVIDENCE_BASE_REL + "/capacity_analysis_attempt01.json",
    "bytes": 19_072,
    "sha256": "0e8de8f39ff72ada8e61ebab6950604020c6674a5bb9c61e0ede221fe0745315",
    "schema": "cross_tfg_dev360_allocated_capacity_analysis_attempt01_v1",
    "status": "SEALED_LOCAL_SOURCE_BACKED_SCENARIO",
}
CAPACITY_SCENARIO = {
    "classification": "SCENARIO_ONLY_NOT_A_HARD_BOUND_OR_FUTURE_OUTPUT_SIZE_CLAIM",
    "allocated_pilot_base_bytes": 4_986_961_920,
    "stagea_additions_bytes": 879_246_792,
    "enriched_projected_base_bytes": 5_866_208_712,
    "explicit_slack_multiplier": 2,
    "explicit_slack_bytes": 11_732_417_424,
    "shared_source_pcm_bytes_counted_once": 5_465_536,
    "output_required_scenario_bytes": 17_604_091_672,
    "policy_reserve_bytes": 5_368_709_120,
    "preflight_free_bytes_needed": 22_972_800_792,
    "formula": "sum(max allocated PASS pilot row per backend × 180 rows) + source-backed Stage-A additions = enriched projected base; add 2× explicit scenario slack to enriched base; add one shared source PCM copy. The 5 GiB reserve is a separate admission policy, not a measured peak bound.",
}
CAPACITY_ROW_BYTES = {
    MODELS[0]: 8_830_976,
    MODELS[1]: 9_826_304,
    MODELS[2]: 9_048_064,
}
CAPACITY_CORE_CURRENT = {
    "path": "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py",
    "bytes": 79_475,
    "sha256": "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483",
}
CAPACITY_CORE_PRIOR = {
    "path": "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt05.py",
    "bytes": 69_184,
    "sha256": "6fcef9ac59983af9aeb70fa9fe4ec8bdb6e16e2c2ff3368d142f2702f9352034",
}
CAPACITY_CORE_RETENTION_DIFF = {
    "path": CAPACITY_EVIDENCE_BASE_REL + "/core06_vs_core05.diff",
    "bytes": 12_837,
    "sha256": "57955e90bfeb96ef5827b8ad7da8418b311dfa391916688e91b2a8e8d7d00d3f",
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


def read_regular(path: Path, label: str, limit: int = 32 * 1024 * 1024) -> bytes:
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
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size):
        raise ValueError(f"{label} changed while being read")
    return raw


def load_json(path: Path, label: str, expected_sha: str | None = None,
              limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, limit)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA-256 mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def write_exclusive(path: Path, raw: bytes) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    if target.exists() or target.is_symlink() or getattr(target, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing to replace existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def reject_secret_fields(value: Any, trail: str = "seed") -> None:
    forbidden = ("token", "password", "secret", "api_key", "access_key", "private_key")
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(term in key_text for term in forbidden):
                raise ValueError(f"seed contains forbidden credential field {trail}.{key}")
            reject_secret_fields(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_fields(child, f"{trail}[{index}]")


def inventory_id(model_id: str, row: Mapping[str, Any]) -> str:
    return sha_bytes(canonical({"model_id": model_id, "path": row["legacy_literal_path"],
                                "sha256": row["expected_sha256"], "role": row["role"]}))


def prepare_template(output: Path) -> dict[str, Any]:
    roster, _roster_raw = load_json(ROOT / ROSTER_REL, "frozen backend roster", ROSTER_SHA256)
    base_plan, base_raw = load_json(ROOT / BASE_PLAN_REL, "frozen Dev context plan", BASE_PLAN_SHA256)
    inventory, inventory_raw = load_json(ROOT / RESTORE_INVENTORY_REL, "frozen restore inventory",
                                          RESTORE_INVENTORY_SHA256)
    old, old_raw = load_json(ROOT / OLD_TEMPLATE_REL, "prior runtime binding template", OLD_TEMPLATE_SHA256)
    if len(_roster_raw) != ROSTER_BYTES or len(inventory_raw) != RESTORE_INVENTORY_BYTES or len(old_raw) != OLD_TEMPLATE_BYTES:
        raise ValueError("a reviewed input has the wrong exact byte length")
    if (roster.get("status") != "FROZEN" or inventory.get("status") !=
            "FROZEN_IDENTITIES_AND_LEGACY_PATHS_ENUMERATED_CURRENT_HOST_PRESENCE_NOT_VERIFIED"):
        raise ValueError("frozen roster or restore identity status changed")
    if tuple(model.get("model_id") for model in roster.get("models", [])) != MODELS:
        raise ValueError("frozen backend roster order changed")
    inv_backends = {item["model_id"]: item for item in inventory.get("backends", [])}
    roster_models = {item["model_id"]: item for item in roster["models"]}
    seed = copy.deepcopy(old)
    seed["schema"] = SOURCE_TEMPLATE_SCHEMA
    seed["status"] = "PENDING_CURRENT_HOST_RUNTIME_VALUES_THREE_QUALIFICATION_LINEAGES_AND_FRAME_BINDING"
    seed["authority"] = (
        "Metadata-only input template. Per-backend strict-repeat manifests remain distinct, and Wav2Lip frame counts come from the pinned native frontend/chunk-loop probe. Fill current runtime paths and explicit capacity evidence before collection."
    )
    seed["source_pins"] = {
        "frozen_roster": {"path": ROSTER_REL, "bytes": ROSTER_BYTES, "raw_sha256": ROSTER_SHA256},
        "frozen_dev_context_plan": {"path": BASE_PLAN_REL, "bytes": len(base_raw),
                                     "raw_sha256": sha_bytes(base_raw), "plan_sha256": base_plan.get("plan_sha256")},
        "restore_inventory": {"path": RESTORE_INVENTORY_REL, "bytes": RESTORE_INVENTORY_BYTES,
                              "raw_sha256": sha_bytes(inventory_raw),
                              "manifest_sha256": inventory.get("manifest_sha256")},
        "prior_template": {"path": OLD_TEMPLATE_REL, "bytes": len(old_raw), "raw_sha256": sha_bytes(old_raw)},
        "dev20_w2l_native_frame_binding": {
            "path": W2L_FRAME_PROBE_REL, "bytes": W2L_FRAME_PROBE_BYTES,
            "raw_sha256": W2L_FRAME_PROBE_SHA256,
            "receipt_sha256": W2L_FRAME_PROBE_RECEIPT_SHA256,
        },
        "backend_strict_repeat_lineages": copy.deepcopy(QUALIFICATION_LINEAGE_PINS),
    }
    seed["gpu_uuid"] = GPU_UUID
    seed["qc06_source_sha256"] = QC06["source_sha256"]
    seed["determinism_policy_sha256"] = QC06["determinism_policy_sha256"]
    seed["qc06_receipts"] = {key: QC06[key] for key in ("preflight_raw_sha256", "terminal_raw_sha256", "manifest_raw_sha256")}
    seed["strict_determinism_policy"] = copy.deepcopy(STRICT_POLICY)
    seed.pop("secret_policy", None)
    seed["strict_determinism_bootstrap"] = {
        "path": "@PROJECT_ROOT@/" + STRICT_BOOTSTRAP_REL,
        "bytes": STRICT_BOOTSTRAP_BYTES, "sha256": STRICT_BOOTSTRAP_SHA256,
    }
    seed["backend_determinism_qualification_receipts"] = {
        model_id: copy.deepcopy(QUALIFICATION_LINEAGE_PINS[model_id]["qualification"])
        for model_id in MODELS
    }
    seed["backend_directories"] = {model_id: None for model_id in MODELS}
    seed["python_environments"] = {key: None for key in ("stagea", "syncnet", "wav2lip", "muse", "ditto")}
    seed["execution_bindings"]["strict_determinism_bootstrap"] = "@PROJECT_ROOT@/" + STRICT_BOOTSTRAP_REL
    seed["execution_bindings"]["w2l_native_frame_binding_source"] = "@PROJECT_ROOT@/" + W2L_FRAME_PROBE_REL
    seed["execution_bindings"]["w2l_native_frame_binding_source_sha256"] = W2L_FRAME_PROBE_SHA256
    seed["execution_bindings"]["generator_child"] = "@PROJECT_ROOT@/scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt06.py"
    seed["execution_bindings"]["syncnet_evaluator"] = "@PROJECT_ROOT@/" + BASE + "/stagea_syncnet_evaluator.py"
    seed["execution_bindings"]["pcm16_lineage"] = "@PROJECT_ROOT@/" + BASE + "/stagea_pcm16_lineage.py"
    seed["expected_file_sha256"]["generator_child"] = QC06["source_sha256"]
    seed["expected_file_sha256"]["strict_determinism_bootstrap"] = STRICT_BOOTSTRAP_SHA256
    seed["credential_policy"] = "No tokens, passwords, API keys, credential values, or private mount source strings."
    seed["backend_source_policy"] = {
        "max_unique_dependency_bytes": MAX_DEPENDENCY_BYTES,
        "hash_all_frozen_backend_source_and_model_files": True,
        "hash_strict_bootstrap_and_adapter_shims": True,
        "only_allowed_symlinks": "the six frozen Muse aliases in the guarded launcher",
        "dev_confirm_face_video_content_read": False,
        "source_presence_checks_are_stat_only": True,
    }
    seed["backends"] = {}
    for model_id in MODELS:
        model = roster_models[model_id]
        inv = inv_backends.get(model_id)
        if not isinstance(inv, dict):
            raise ValueError(f"restore inventory lacks {model_id}")
        generator = copy.deepcopy(model.get("generator", {}))
        dependencies: list[dict[str, Any]] = []
        for item in inv.get("files", []):
            if model_id in STRICT_SHIMS and item.get("role") == "adapter":
                continue
            old_path = item.get("legacy_literal_path")
            if not isinstance(old_path, str) or not old_path.startswith("/"):
                raise ValueError(f"restore inventory path is not absolute for {model_id}")
            dependencies.append({
                "inventory_id": inventory_id(model_id, item),
                "expected_legacy_path": old_path,
                "path": old_path,
                "bytes": item.get("expected_bytes"),
                "sha256": item["expected_sha256"],
                "role": item["role"],
            })
        if model_id in STRICT_SHIMS:
            rel, byte_count, digest = STRICT_SHIMS[model_id]
            shim_placeholder = "@PROJECT_ROOT@/" + rel
            generator["adapter_path"] = shim_placeholder
            generator["adapter_sha256"] = digest
            generator["child_script_sha256"] = digest
            dependencies.append({"inventory_id": "runtime-strict-shim:" + model_id,
                                 "expected_legacy_path": shim_placeholder, "path": shim_placeholder,
                                 "bytes": byte_count, "sha256": digest, "role": "strict_runtime_adapter"})
            entry_path = generator.get("renderer_path")
            if not isinstance(entry_path, str):
                raise ValueError(f"qualified renderer path is absent for {model_id}")
            generator["render_entry_path"] = entry_path
        dependencies.append({"inventory_id": "strict-bootstrap:" + model_id,
                             "expected_legacy_path": "@PROJECT_ROOT@/" + STRICT_BOOTSTRAP_REL,
                             "path": "@PROJECT_ROOT@/" + STRICT_BOOTSTRAP_REL,
                             "bytes": STRICT_BOOTSTRAP_BYTES, "sha256": STRICT_BOOTSTRAP_SHA256,
                             "role": "strict_determinism_bootstrap"})
        if model_id == MODELS[0]:
            generator["child_script_sha256"] = W2L_NATIVE_ENTRY_SHA256
            generator.update({
                "native_entry_path": W2L_NATIVE_ENTRY,
                "native_entry_sha256": W2L_NATIVE_ENTRY_SHA256,
                "runtime_compat_sitecustomize_path": "@PROJECT_ROOT@/" + W2L_RUNTIME_COMPAT_REL,
                "runtime_compat_sitecustomize_sha256": W2L_RUNTIME_COMPAT_SHA256,
                "strict_bootstrap_path": "@PROJECT_ROOT@/" + STRICT_BOOTSTRAP_REL,
                "strict_bootstrap_sha256": STRICT_BOOTSTRAP_SHA256,
                "native_frame_count_probe_path": "@PROJECT_ROOT@/" + W2L_FRAME_PROBE_REL,
                "native_frame_count_probe_raw_sha256": W2L_FRAME_PROBE_SHA256,
                "native_frame_count_probe_receipt_sha256": W2L_FRAME_PROBE_RECEIPT_SHA256,
                "implementation": "official pinned inference.py entry with strict runtime compatibility hooks; source-derived frame-count binding per natural PCM",
            })
            for dep_id, dep_path, dep_bytes, dep_sha, role in W2L_EXTRA_DEPENDENCIES:
                dependencies.append({
                    "inventory_id": dep_id, "expected_legacy_path": dep_path,
                    "path": dep_path, "bytes": dep_bytes, "sha256": dep_sha, "role": role,
                })
        seed["backends"][model_id] = {
            "model_binding": {key: model.get(key) for key in MODEL_BINDING_FIELDS},
            "generator": generator,
            "dependencies": dependencies,
            "approved_symlink_aliases": {},
        }
    seed["seed_sha256"] = payload_sha(seed, "seed_sha256")
    reject_secret_fields(seed)
    raw = (json.dumps(seed, sort_keys=True, indent=0, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    write_exclusive(output, raw)
    return {"status": seed["status"], "path": str(output.resolve()), "bytes": len(raw),
            "raw_sha256": sha_bytes(raw), "seed_sha256": seed["seed_sha256"],
            "dependency_pin_count": sum(len(item["dependencies"]) for item in seed["backends"].values()),
            "confirm_audio_read": False, "model_execution_performed": False}


def seal_seed(input_path: Path, output: Path, expected_sha: str) -> dict[str, Any]:
    seed, raw = load_json(input_path, "runtime binding seed", expected_sha, 8 * 1024 * 1024)
    if seed.get("schema") != SOURCE_TEMPLATE_SCHEMA:
        raise ValueError("seed schema is not the reviewed runtime binding template")
    if seed.get("status") != "PENDING_CURRENT_HOST_RUNTIME_VALUES_THREE_QUALIFICATION_LINEAGES_AND_FRAME_BINDING":
        raise ValueError("only the pending template can be sealed")
    seed["status"] = "READY_FOR_COLLECTION"
    seed["seed_sha256"] = payload_sha(seed, "seed_sha256")
    raw_out = pretty(seed)
    write_exclusive(output, raw_out)
    return {"status": seed["status"], "path": str(output.resolve()), "bytes": len(raw_out),
            "raw_sha256": sha_bytes(raw_out), "seed_sha256": seed["seed_sha256"],
            "confirmation_audio_opened": False}


def _resolve_root_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty explicit path")
    if value.startswith("@PROJECT_ROOT@/"):
        value = str(project_root / value[len("@PROJECT_ROOT@/"):])
    result = Path(value)
    if not result.is_absolute():
        if ".." in result.parts:
            raise ValueError(f"{label} relative path traverses above the project root")
        result = project_root / result
    return result


def _source_stat(path_value: Any, label: str, expected_sha: str,
                 expected_bytes: int | None = None) -> dict[str, Any]:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError(f"{label} source path must be absolute")
    path = Path(path_value)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} is not a regular non-symlink file")
    if expected_bytes is not None and info.st_size != expected_bytes:
        raise ValueError(f"{label} file size differs from the frozen source metadata")
    if len(expected_sha) != 64:
        raise ValueError(f"{label} expected source digest is malformed")
    # Content is deliberately not opened: this is a path/presence receipt only.
    return {"path": str(path), "bytes": int(info.st_size), "device": int(info.st_dev),
            "inode": int(info.st_ino), "mtime_ns": int(info.st_mtime_ns),
            "expected_file_sha256": expected_sha, "content_sha256_rechecked": False}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed metadata helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_qualification_lineage(project_root: Path, model_id: str,
                                   lineage: Mapping[str, Any], qualification_ref: Mapping[str, Any],
                                   record: Mapping[str, Any], qualification_raw: bytes) -> dict[str, Any]:
    """Bind one PASS qualification to its own exact final manifest and rows."""
    manifest_pin = lineage.get("manifest")
    qualification_pin = lineage.get("qualification")
    if not isinstance(manifest_pin, Mapping) or not isinstance(qualification_pin, Mapping):
        raise ValueError(f"{model_id} lineage lacks both manifest and qualification pins")
    for key in ("path", "bytes", "sha256"):
        if qualification_ref.get(key) != qualification_pin.get(key):
            raise ValueError(f"{model_id} seed qualification {key} differs from its reviewed lineage pin")
    qualification_path = _resolve_root_path(project_root, qualification_pin.get("path"),
                                            f"{model_id} pinned qualification")
    if (len(qualification_raw) != qualification_pin.get("bytes")
            or sha_bytes(qualification_raw) != qualification_pin.get("sha256")):
        raise ValueError(f"{model_id} qualification bytes do not match the reviewed lineage pin")

    manifest_path = _resolve_root_path(project_root, manifest_pin.get("path"),
                                       f"{model_id} strict-repeat manifest")
    manifest_raw = read_regular(manifest_path, f"{model_id} strict-repeat manifest", 8 * 1024 * 1024)
    if (len(manifest_raw) != manifest_pin.get("bytes")
            or sha_bytes(manifest_raw) != manifest_pin.get("sha256")):
        raise ValueError(f"{model_id} strict-repeat manifest raw bytes differ from the independently read-back pin")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "cross_tfg_backend_strict_repeat_pilot_manifest_v1"
            or manifest.get("run_id") != manifest_pin.get("run_id")
            or manifest.get("planned_rows") != 6
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
        raise ValueError(f"{model_id} strict-repeat manifest schema, run, denominator, or self-hash is invalid")

    expected_ids = [f"strictrepeat_{model_id}_01", f"strictrepeat_{model_id}_02"]
    manifest_rows = manifest.get("rows")
    qualification_rows = record.get("repeat_rows")
    if not isinstance(manifest_rows, list) or not isinstance(qualification_rows, list):
        raise ValueError(f"{model_id} strict-repeat manifest or qualification lacks row evidence")
    selected_rows = [row for row in manifest_rows if isinstance(row, dict) and row.get("model_id") == model_id]
    selected_rows.sort(key=lambda row: str(row.get("row_id", "")))
    selected_ids = [row.get("row_id") for row in selected_rows]
    if (selected_ids != expected_ids
            or [row.get("row_id") for row in qualification_rows if isinstance(row, dict)] != expected_ids
            or selected_rows != qualification_rows):
        raise ValueError(f"{model_id} qualification rows do not exactly match its two final-manifest rows")
    if any(row.get("status") != "PASS" or row.get("seed") != 20261003 for row in selected_rows):
        raise ValueError(f"{model_id} strict-repeat lineage contains a nonpassing or wrong-seed row")

    manifest_qualification = manifest.get("backend_qualifications", {}).get(model_id)
    receipt = manifest_qualification.get("receipt") if isinstance(manifest_qualification, dict) else None
    expected_receipt = {key: qualification_pin[key] for key in ("path", "bytes", "sha256")}
    if (not isinstance(manifest_qualification, dict)
            or manifest_qualification.get("status") != "PASS_EXACT_MATCH"
            or manifest_qualification.get("model_id") not in (None, model_id)
            or receipt != expected_receipt):
        raise ValueError(f"{model_id} final manifest does not point to this exact passing qualification file")

    repeat_gate = record.get("repeat_gate")
    if (not isinstance(repeat_gate, dict)
            or repeat_gate.get("status") != "PASS_EXACT_MATCH"
            or repeat_gate.get("repeat_row_ids") != expected_ids
            or any(key.endswith("_equal") and value is not True for key, value in repeat_gate.items())):
        raise ValueError(f"{model_id} repeat gate lacks exact matching evidence for its two rows")

    if model_id == MODELS[0]:
        for index, row in enumerate(selected_rows):
            receipts = row.get("wav2lip_runtime_compatibility_receipts")
            if not isinstance(receipts, dict) or set(receipts) != {
                    "actual_forward_policy_receipt", "constructor_policy_receipt", "prewrite_pixel_array_receipt"}:
                raise ValueError(f"{model_id} row lacks its three runtime-compatibility receipt pins")
            for key, value in receipts.items():
                if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
                        or not Path(value["path"]).is_absolute() or type(value.get("bytes")) is not int
                        or value["bytes"] <= 0 or not isinstance(value.get("sha256"), str)
                        or len(value["sha256"]) != 64):
                    raise ValueError(f"{model_id}/{expected_ids[index]} has an invalid {key} reference")
            forward = receipts["actual_forward_policy_receipt"]
            prewrite = receipts["prewrite_pixel_array_receipt"]
            expected_stream = repeat_gate.get("prewrite_pixel_frame_hash_streams", [])[index]
            if (forward.get("sfd_forward_count", 0) <= 0 or forward.get("wav2lip_forward_count", 0) <= 0
                    or prewrite.get("frame_pixel_array_sha256s") != expected_stream
                    or len(expected_stream) != 141
                    or prewrite.get("ordered_frame_hashes_sha256") !=
                    repeat_gate.get("prewrite_pixel_frame_stream_sha256", [])[index]):
                raise ValueError(f"{model_id}/{expected_ids[index]} actual forward/pre-write evidence differs from its qualification gate")
    else:
        for row in selected_rows:
            native = row.get("native_adapter_receipt")
            embedded = native.get("receipt") if isinstance(native, dict) else None
            expected_schema = "cross_tfg_musetalk_stagea_adapter_receipt_v1" if model_id == MODELS[1] else "cross_tfg_ditto_stagea_adapter_receipt_v1"
            expected_status = "PASS" if model_id == MODELS[1] else "PASS_ENGINEERING_ONLY"
            if (not isinstance(native, dict) or type(native.get("bytes")) is not int
                    or not isinstance(native.get("path"), str) or not Path(native["path"]).is_absolute()
                    or not isinstance(native.get("sha256"), str) or len(native["sha256"]) != 64
                    or not isinstance(embedded, dict) or embedded.get("schema") != expected_schema
                    or embedded.get("status") != expected_status or embedded.get("row_id") != row["row_id"]
                    or embedded.get("model_id") != model_id or embedded.get("seed") != 20261003):
                raise ValueError(f"{model_id}/{row['row_id']} native adapter receipt is absent or mismatched")
            native_path = Path(native["path"])
            native_raw = read_regular(native_path, f"{model_id}/{row['row_id']} native adapter receipt", 4 * 1024 * 1024)
            native_obj = json.loads(native_raw.decode("utf-8"))
            if (len(native_raw) != native.get("bytes") or sha_bytes(native_raw) != native.get("sha256")
                    or not isinstance(native_obj, dict) or native_obj != embedded
                    or native_obj.get("schema") != expected_schema or native_obj.get("status") != expected_status
                    or native_obj.get("row_id") != row["row_id"] or native_obj.get("model_id") != model_id
                    or native_obj.get("seed") != 20261003):
                raise ValueError(f"{model_id}/{row['row_id']} native adapter raw file/SHA/JSON differs from its exact qualification row")

    input_context = record.get("input_context")
    if (not isinstance(input_context, dict)
            or record.get("input_context_sha256") != payload_sha(input_context, "input_context_sha256")
            or input_context.get("scope") != "engineering_qc06_exposed_input_only"
            or input_context.get("seed") != 20261003
            or input_context.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
            or input_context.get("strict_bootstrap_sha256") != STRICT_BOOTSTRAP_SHA256):
        raise ValueError(f"{model_id} qualification input context does not match frozen QC06 audio/policy")
    return {
        "path": str(manifest_path.resolve(strict=True)), "bytes": len(manifest_raw),
        "raw_sha256": sha_bytes(manifest_raw), "manifest_sha256": manifest["manifest_sha256"],
        "run_id": manifest["run_id"], "planned_rows": manifest["planned_rows"],
        "qualified_model_row_ids": expected_ids,
        "qualification_receipt": expected_receipt,
        "input_context": {key: input_context.get(key) for key in (
            "scope", "sample_id", "seed", "canonical_audio_file_sha256", "canonical_audio_pcm_sha256",
            "source_file_sha256", "sample_count", "sample_rate_hz", "channels", "face_file_sha256",
            "syncnet_checkpoint_sha256", "determinism_policy_sha256", "strict_bootstrap_sha256")},
    }


def _directory_inputs(generator: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("cwd", "repo", "model_source_root", "ditto_source_root", "model_code_root",
                "model_workspace", "ditto_weights_root"):
        value = generator.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    roots = generator.get("model_overlay_roots")
    if isinstance(roots, list):
        for index, value in enumerate(roots):
            if isinstance(value, str) and value:
                result[f"model_overlay_roots[{index}]"] = value
    env = generator.get("env", {})
    pythonpath = env.get("PYTHONPATH") if isinstance(env, Mapping) else None
    if isinstance(pythonpath, str):
        for index, value in enumerate(item for item in pythonpath.split(os.pathsep) if item):
            result[f"PYTHONPATH[{index}]"] = value
    return result


def _capacity(seed: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    resources = seed.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("runtime seed lacks resource/capacity inputs")
    basis = resources.get("capacity_basis")
    if (not isinstance(basis, dict)
            or basis.get("status") != "MEASURED_PILOT_WITH_EXPLICIT_SCENARIO_AND_LIVE_GATES"
            or basis.get("media_bytes_estimated") is not True
            or basis.get("basis_kind") != "MEASURED_PILOT_WITH_EXPLICIT_SCENARIO_AND_LIVE_GATES"):
        raise ValueError("capacity basis must preserve the measured pilot plus explicit non-bound scenario status")
    required = basis.get("output_required_bytes")
    reserve = basis.get("output_reserve_bytes")
    if (required != CAPACITY_SCENARIO["output_required_scenario_bytes"]
            or reserve != CAPACITY_SCENARIO["policy_reserve_bytes"]):
        raise ValueError("capacity byte totals differ from the sealed source-backed scenario and separate reserve")
    timings = basis.get("per_backend_controls_row_timing_seconds")
    sizes = basis.get("per_backend_controls_row_output_bytes")
    if not isinstance(timings, dict) or not isinstance(sizes, dict):
        raise ValueError("capacity basis needs per-backend observed pilot timing and row byte metadata")
    if set(timings) != set(MODELS) or set(sizes) != set(MODELS):
        raise ValueError("capacity pilot timing/byte metadata must cover the exact three backends")
    for model_id in MODELS:
        timing, size = timings[model_id], sizes[model_id]
        if isinstance(timing, bool) or not isinstance(timing, (int, float)) or timing <= 0:
            raise ValueError(f"historical pilot timing metadata is invalid for {model_id}")
        if type(size) is not int or size != CAPACITY_ROW_BYTES[model_id]:
            raise ValueError(f"measured pilot allocated row bytes differ for {model_id}")
    if resources.get("output_required_bytes") != required or resources.get("output_reserve_bytes") != reserve:
        raise ValueError("resource gate byte counts differ from the explicit scenario basis")

    pilot_path = project_root / PILOT_FOOTPRINT_RECEIPT["path"]
    pilot, pilot_raw = load_json(pilot_path, "measured pilot footprint receipt",
                                 PILOT_FOOTPRINT_RECEIPT["sha256"], 8 * 1024 * 1024)
    if (len(pilot_raw) != PILOT_FOOTPRINT_RECEIPT["bytes"]
            or pilot.get("schema") != PILOT_FOOTPRINT_RECEIPT["schema"]
            or pilot.get("status") != PILOT_FOOTPRINT_RECEIPT["status"]
            or pilot.get("selected_pilot_row_count") != 6
            or pilot.get("scope", {}).get("read_only_targets") is not True
            or pilot.get("scope", {}).get("gpu_compute_started") is not False
            or pilot.get("scope", {}).get("inference_or_scoring_run") is not False
            or pilot.get("scope", {}).get("candidate_selection_or_confirm") is not False):
        raise ValueError("measured pilot receipt schema/scope/row count differs from the accepted read-only evidence")
    analysis_path = project_root / CAPACITY_ANALYSIS_RECEIPT["path"]
    analysis, analysis_raw = load_json(analysis_path, "Stage-A capacity scenario analysis",
                                       CAPACITY_ANALYSIS_RECEIPT["sha256"], 8 * 1024 * 1024)
    if (len(analysis_raw) != CAPACITY_ANALYSIS_RECEIPT["bytes"]
            or analysis.get("schema") != CAPACITY_ANALYSIS_RECEIPT["schema"]
            or analysis.get("status") != CAPACITY_ANALYSIS_RECEIPT["status"]
            or analysis.get("capacity_scenario") != CAPACITY_SCENARIO
            or analysis.get("pilot_evidence", {}).get("raw_resource_receipt", {}).get("sha256") != PILOT_FOOTPRINT_RECEIPT["sha256"]):
        raise ValueError("source-backed capacity analysis does not reproduce the accepted scenario/pilot receipt")
    if analysis.get("pilot_evidence", {}).get("raw_resource_receipt", {}).get("bytes") != PILOT_FOOTPRINT_RECEIPT["bytes"]:
        raise ValueError("capacity analysis links a different measured pilot receipt size")

    current_core_path = project_root / CAPACITY_CORE_CURRENT["path"]
    current_core_raw = read_regular(current_core_path, "current Stage-A core", 8 * 1024 * 1024)
    if (len(current_core_raw) != CAPACITY_CORE_CURRENT["bytes"]
            or sha_bytes(current_core_raw) != CAPACITY_CORE_CURRENT["sha256"]):
        raise ValueError("current Stage-A core differs from the reviewed retention-equivalence pin")
    prior_core_path = project_root / CAPACITY_CORE_PRIOR["path"]
    prior_core_raw = read_regular(prior_core_path, "capacity-analysis Stage-A source", 8 * 1024 * 1024)
    if (len(prior_core_raw) != CAPACITY_CORE_PRIOR["bytes"]
            or sha_bytes(prior_core_raw) != CAPACITY_CORE_PRIOR["sha256"]):
        raise ValueError("prior Stage-A source differs from the capacity analysis source pin")
    core_diff_path = project_root / CAPACITY_CORE_RETENTION_DIFF["path"]
    core_diff_raw = read_regular(core_diff_path, "current/prior Stage-A retention diff", 2 * 1024 * 1024)
    if (len(core_diff_raw) != CAPACITY_CORE_RETENTION_DIFF["bytes"]
            or sha_bytes(core_diff_raw) != CAPACITY_CORE_RETENTION_DIFF["sha256"]):
        raise ValueError("current/prior Stage-A retention diff differs from its reviewed pin")

    # Bind both local evidence artifacts and derive the stage shares from the
    # source-backed scenario. The two shares exclude the one shared PCM copy,
    # which is counted in the initial full-540 admission check.
    bound = copy.deepcopy(basis)
    bound["evidence_receipts"] = {
        "measured_pilot_footprint": {**PILOT_FOOTPRINT_RECEIPT, "raw_sha256": sha_bytes(pilot_raw)},
        "stagea_source_backed_capacity_analysis": {**CAPACITY_ANALYSIS_RECEIPT,
                                                    "raw_sha256": sha_bytes(analysis_raw)},
    }
    scenario = dict(CAPACITY_SCENARIO)
    row_outputs = required - scenario["shared_source_pcm_bytes_counted_once"]
    controls_share = (row_outputs * 2) // 3
    candidate_share = row_outputs - controls_share
    bound["capacity_scenario"] = scenario
    bound["stage_output_required_bytes"] = {"controls360": controls_share,
                                            "candidate180": candidate_share}
    bound["per_row_scenario_allowance_bytes"] = {
        model_id: CAPACITY_ROW_BYTES[model_id] * scenario["explicit_slack_multiplier"]
        for model_id in MODELS
    }
    bound["row_allowance_classification"] = "MEASURED_PILOT_ROW_ALLOCATION_TIMES_EXPLICIT_SCENARIO_SLACK; NOT_A_HARD_BOUND"
    bound["retention_equivalence"] = {
        "status": "SOURCE_REVIEWED_RETENTION_PATHS_PRESERVED; CURRENT_CORE_NOT_MEASURED",
        "capacity_analysis_core_source": {**CAPACITY_CORE_PRIOR,
                                          "raw_sha256": sha_bytes(prior_core_raw)},
        "current_stagea_core": {**CAPACITY_CORE_CURRENT,
                                 "raw_sha256": sha_bytes(current_core_raw)},
        "current_vs_prior_core_diff": {**CAPACITY_CORE_RETENTION_DIFF,
                                       "raw_sha256": sha_bytes(core_diff_raw)},
        "claim": "The measured pilot and source-backed Stage-A artifact-size scenario are historical evidence. The pinned source diff supports retention-path equivalence only; it does not measure future core06 output bytes.",
    }
    return bound


def collect(seed_path: Path, seed_sha: str, output: Path) -> dict[str, Any]:
    project_root = ROOT.resolve(strict=True)
    seed, seed_raw = load_json(seed_path, "ready runtime binding seed", seed_sha, 8 * 1024 * 1024)
    if seed.get("schema") != SOURCE_TEMPLATE_SCHEMA or seed.get("status") != "READY_FOR_COLLECTION":
        raise ValueError("runtime seed must be sealed and marked READY_FOR_COLLECTION")
    if seed.get("seed_sha256") != payload_sha(seed, "seed_sha256"):
        raise ValueError("runtime seed self-hash is invalid")
    reject_secret_fields(seed)
    if (seed.get("gpu_uuid") != GPU_UUID or seed.get("qc06_source_sha256") != QC06["source_sha256"]
            or seed.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
            or seed.get("strict_determinism_policy") != STRICT_POLICY):
        raise ValueError("seed GPU or strict deterministic policy differs from the reviewed QC06 identity")

    roster, roster_raw = load_json(project_root / ROSTER_REL, "frozen backend roster", ROSTER_SHA256)
    base_plan, base_raw = load_json(project_root / BASE_PLAN_REL, "frozen Dev context plan", BASE_PLAN_SHA256)
    inventory, inventory_raw = load_json(project_root / RESTORE_INVENTORY_REL, "frozen restore inventory",
                                          RESTORE_INVENTORY_SHA256)
    if len(roster_raw) != ROSTER_BYTES or len(inventory_raw) != RESTORE_INVENTORY_BYTES:
        raise ValueError("frozen source identity byte counts changed")
    source_pins = seed.get("source_pins", {})
    if source_pins.get("frozen_roster", {}).get("raw_sha256") != sha_bytes(roster_raw):
        raise ValueError("seed references a different frozen backend roster")
    if source_pins.get("frozen_dev_context_plan", {}).get("raw_sha256") != sha_bytes(base_raw):
        raise ValueError("seed references a different frozen Dev context plan")
    if source_pins.get("restore_inventory", {}).get("raw_sha256") != sha_bytes(inventory_raw):
        raise ValueError("seed references a different backend restore inventory")
    if source_pins.get("dev20_w2l_native_frame_binding") != {
            "path": W2L_FRAME_PROBE_REL, "bytes": W2L_FRAME_PROBE_BYTES,
            "raw_sha256": W2L_FRAME_PROBE_SHA256,
            "receipt_sha256": W2L_FRAME_PROBE_RECEIPT_SHA256}:
        raise ValueError("seed does not bind the exact CPU-only native Wav2Lip frame-count probe")
    if source_pins.get("backend_strict_repeat_lineages") != QUALIFICATION_LINEAGE_PINS:
        raise ValueError("seed per-backend strict-repeat lineage paths/pins differ from the actual manifests")
    if tuple(item.get("model_id") for item in roster.get("models", [])) != MODELS:
        raise ValueError("frozen backend roster changed")

    runner_rel = BASE + "/run_v3_samehost_dev360_controls_attempt15.py"
    runner_path = project_root / runner_rel
    runner_raw = read_regular(runner_path, "reviewed Dev360 guarded launcher", 8 * 1024 * 1024)
    runner = _load_module(runner_path, "samehost_dev360_guard_attempt15_for_binding")
    if (runner.SOURCE_TEMPLATE_SCHEMA if hasattr(runner, "SOURCE_TEMPLATE_SCHEMA") else None) is not None:
        raise ValueError("unexpected guard namespace collision")
    helper = runner.load_resource_helper(project_root)

    runtime: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA, "status": "PENDING", "gpu_uuid": GPU_UUID,
        "qc06_source_sha256": QC06["source_sha256"],
        "determinism_policy_sha256": QC06["determinism_policy_sha256"],
        "qc06_receipts": {key: QC06[key] for key in ("preflight_raw_sha256", "terminal_raw_sha256", "manifest_raw_sha256")},
        "backend_determinism_qualifications": {},
        "backends": {},
        "execution_bindings": copy.deepcopy(seed.get("execution_bindings", {})),
        "expected_file_sha256": copy.deepcopy(seed.get("expected_file_sha256", {})),
        "resources": {},
        "shared_sources": copy.deepcopy(seed.get("shared_sources", {})),
        "dev_contexts": copy.deepcopy(seed.get("dev_contexts", {})),
        "strict_determinism_policy": copy.deepcopy(STRICT_POLICY),
        "source_pins": copy.deepcopy(source_pins),
        "binding_collector": {"path": str(Path(__file__).resolve(strict=True)),
                              "bytes": len(read_regular(Path(__file__), "binding collector")),
                              "raw_sha256": sha_bytes(read_regular(Path(__file__), "binding collector"))},
        "binding_seed": {"path": str(seed_path.resolve(strict=True)), "bytes": len(seed_raw), "raw_sha256": sha_bytes(seed_raw),
                         "seed_sha256": seed["seed_sha256"]},
        "restore_inventory": {"path": RESTORE_INVENTORY_REL, "bytes": len(inventory_raw),
                              "raw_sha256": sha_bytes(inventory_raw), "manifest_sha256": inventory.get("manifest_sha256")},
        "source_presence_observations": {"scope": "stat-only for Dev PCM, source face, and reference video; no source bytes were opened"},
        "confirm_audio_read": False, "confirm_effects_read": False,
    }

    frame_probe_pin = source_pins["dev20_w2l_native_frame_binding"]
    frame_probe_path = _resolve_root_path(project_root, frame_probe_pin["path"], "Dev20 Wav2Lip frame-count probe")
    frame_probe_raw = read_regular(frame_probe_path, "Dev20 Wav2Lip frame-count probe", 8 * 1024 * 1024)
    if len(frame_probe_raw) != frame_probe_pin["bytes"] or sha_bytes(frame_probe_raw) != frame_probe_pin["raw_sha256"]:
        raise ValueError("Dev20 Wav2Lip frame-count probe raw bytes differ from the independently read-back pin")
    frame_probe = json.loads(frame_probe_raw.decode("utf-8"))
    if (frame_probe.get("schema") != "cross_tfg_dev20_w2l_native_frame_binding_v1"
            or frame_probe.get("status") != "PASS_CPU_NATIVE_FRONTEND_AND_EXPECTED_FRAME_COUNT_BINDING"
            or frame_probe.get("receipt_sha256") != frame_probe_pin["receipt_sha256"]
            or frame_probe.get("source_and_model_filesystem_mutation_by_probe") is not False
            or frame_probe.get("frozen_plan", {}).get("plan_raw_sha256") != BASE_PLAN_SHA256
            or frame_probe.get("frozen_plan", {}).get("planned_rows") != 180
            or frame_probe.get("scope", {}).get("confirmation_audio_opened") is not False
            or frame_probe.get("scope", {}).get("model_or_scorer_imported_or_invoked") is not False):
        raise ValueError("Dev20 Wav2Lip frame-count probe schema/scope/source binding is not the accepted CPU-only result")
    clips = frame_probe.get("clips")
    if not isinstance(clips, list) or len(clips) != 20:
        raise ValueError("Dev20 Wav2Lip frame-count probe must bind exactly 20 natural PCM clips")
    base_context_by_id: dict[str, dict[str, Any]] = {}
    for row in base_plan.get("rows", []):
        if row.get("split") == "dev":
            base_context_by_id.setdefault(row["sample_id"], row)
    frame_rows: dict[str, dict[str, Any]] = {}
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("Dev20 Wav2Lip frame-count probe contains a malformed clip row")
        sample_id = clip.get("sample_id")
        natural = base_context_by_id.get(sample_id, {}).get("natural_pcm", {})
        frames = clip.get("expected_inference_frame_count")
        if (not isinstance(sample_id, str) or sample_id in frame_rows
                or clip.get("source_pcm_float32_sha256") != natural.get("pcm_sha256")
                or clip.get("sample_count") != natural.get("sample_count")
                or (clip.get("fps_num"), clip.get("fps_den")) != (25, 1)
                or type(frames) is not int or frames <= 0
                or clip.get("frame_count_observed_from_model_or_video") is not False):
            raise ValueError(f"Dev20 native frame count is not joined to the exact frozen PCM row: {sample_id}")
        frame_rows[sample_id] = {
            "sample_id": sample_id, "natural_pcm_sha256": natural["pcm_sha256"],
            "sample_count": natural["sample_count"], "fps_num": 25, "fps_den": 1,
            "expected_inference_frame_count": frames,
            "mel_shape": clip.get("mel_shape"), "mel_step_size": clip.get("mel_step_size"),
            "chunk_start_mel_frame_indices": clip.get("chunk_start_mel_frame_indices"),
            "final_tail_start_mel_frame_index": clip.get("final_tail_start_mel_frame_index"),
            "probe_raw_sha256": sha_bytes(frame_probe_raw),
            "probe_receipt_sha256": frame_probe["receipt_sha256"],
        }
    if set(frame_rows) != set(base_context_by_id):
        raise ValueError("Dev20 frame-count map must cover the exact 20 frozen clip IDs")
    runtime["wav2lip_native_frame_count_binding"] = {
        "schema": "cross_tfg_v3_wav2lip_native_frame_count_binding_v1",
        "status": "PASS_SOURCE_DERIVED_METADATA_ONLY",
        "probe": {"path": str(frame_probe_path.resolve(strict=True)), "bytes": len(frame_probe_raw),
                  "raw_sha256": sha_bytes(frame_probe_raw), "receipt_sha256": frame_probe["receipt_sha256"]},
        "frozen_plan_raw_sha256": BASE_PLAN_SHA256,
        "frame_count_semantics": frame_probe["frame_count_semantics"],
        "clips": frame_rows,
        "model_or_video_output_observed": False,
    }

    execution = runtime["execution_bindings"]
    required_exec = ("output_root", "stagea_python", "syncnet_python", "generator_child", "syncnet_root",
                     "syncnet_evaluator", "syncnet_checkpoint", "ffmpeg", "ffprobe", "pcm16_lineage",
                     "wav2lip_python", "wav2lip_root", "strict_determinism_bootstrap")
    for key in required_exec:
        execution[key] = str(_resolve_root_path(project_root, execution.get(key), f"execution binding {key}"))
    strict_bootstrap = project_root / STRICT_BOOTSTRAP_REL
    if execution["strict_determinism_bootstrap"] != str(strict_bootstrap.resolve(strict=True)):
        raise ValueError("runtime seed strict bootstrap path is not attempt03 sitecustomize.py")
    if execution["generator_child"] != str((project_root / BASE / "engineering_qc_oldpilot_attempt06.py").resolve(strict=True)):
        raise ValueError("Wav2Lip child must use the exact QC06 deterministic engineering source")
    frame_source_path = _resolve_root_path(project_root, execution.get("w2l_native_frame_binding_source"),
                                           "Wav2Lip native frame-count source")
    if (str(frame_source_path.resolve(strict=True)) != str(frame_probe_path.resolve(strict=True))
            or execution.get("w2l_native_frame_binding_source_sha256") != W2L_FRAME_PROBE_SHA256):
        raise ValueError("Wav2Lip execution binding does not match the accepted source-derived frame-count map")

    expected_hashes = runtime["expected_file_sha256"]
    fixed_hashes = {
        "generator_child": QC06["source_sha256"],
        "strict_determinism_bootstrap": STRICT_BOOTSTRAP_SHA256,
        "syncnet_evaluator": "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b",
        "syncnet_checkpoint": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
        "ffmpeg": "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45",
        "ffprobe": "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd",
        "pcm16_lineage": "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0",
    }
    for key, digest in fixed_hashes.items():
        if expected_hashes.get(key) != digest:
            raise ValueError(f"runtime seed has an incorrect fixed tool SHA pin: {key}")
    for key, relative in (("syncnet_evaluator", BASE + "/stagea_syncnet_evaluator.py"),
                          ("pcm16_lineage", BASE + "/stagea_pcm16_lineage.py")):
        expected = str((project_root / relative).resolve(strict=True))
        if execution[key] != expected:
            raise ValueError(f"runtime {key} path differs from the reviewed source")

    # Stat-only checks preserve the frozen PCM/face/video identities without opening media.
    source_observations: dict[str, Any] = {"dev_contexts": {}, "shared_sources": {}}
    base_contexts: dict[str, dict[str, Any]] = {}
    for row in base_plan.get("rows", []):
        if row.get("split") != "dev":
            continue
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            prior = base_contexts.get(sample_id)
            if prior is None:
                base_contexts[sample_id] = row
            elif prior.get("natural_pcm") != row.get("natural_pcm"):
                raise ValueError(f"frozen Dev PCM context changes across rows for {sample_id}")
    if len(base_contexts) != 20:
        raise ValueError("frozen Dev source plan no longer contains exactly 20 unique clips")
    for sample_id, source in runtime["dev_contexts"].items():
        if sample_id not in base_contexts or not isinstance(source, dict):
            raise ValueError(f"runtime Dev context is outside the frozen 20-clip roster: {sample_id}")
        expected = base_contexts[sample_id].get("natural_pcm", {})
        for key in ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"):
            if source.get(key) != expected.get(key):
                raise ValueError(f"Dev context does not match frozen {sample_id} {key}")
        path = _resolve_root_path(project_root, source.get("path"), f"Dev PCM path {sample_id}")
        observed = _source_stat(str(path), f"Dev PCM {sample_id}", expected["file_sha256"], source.get("file_bytes"))
        source["path"] = str(path)
        source["file_bytes"] = observed["bytes"]
        source_observations["dev_contexts"][sample_id] = observed
    if set(source_observations["dev_contexts"]) != set(base_contexts):
        raise ValueError("runtime Dev source map must cover the exact frozen 20 clips")

    shared = runtime["shared_sources"]
    expected_ref = next(iter(base_contexts.values()))
    for key, plan_key in (("source_face", "source_face"), ("source_video", "source_video")):
        item = shared.get(key)
        expected = expected_ref.get(plan_key, {})
        if not isinstance(item, dict) or item.get("file_sha256") != expected.get("file_sha256"):
            raise ValueError(f"runtime shared {key} identity differs from frozen roster")
        if key == "source_video" and (item.get("fps_num"), item.get("fps_den")) != (
                expected.get("fps_num"), expected.get("fps_den")):
            raise ValueError("runtime reference-video frame clock differs from frozen roster")
        path = _resolve_root_path(project_root, item.get("path"), f"shared {key} path")
        observed = _source_stat(str(path), key, expected["file_sha256"], item.get("file_bytes"))
        item["path"] = str(path)
        item["file_bytes"] = observed["bytes"]
        source_observations["shared_sources"][key] = observed
    runtime["source_presence_observations"] = source_observations

    # Attach three actual same-GPU, same-policy repeat reports; status is never synthesized.
    qualification_inputs = seed.get("backend_determinism_qualification_receipts")
    if not isinstance(qualification_inputs, dict) or set(qualification_inputs) != set(MODELS):
        raise ValueError("seed lacks exact Wav2Lip/MuseTalk/Ditto strict repeat receipt paths")
    common_qualification_context: dict[str, Any] | None = None
    for model_id in MODELS:
        ref = qualification_inputs[model_id]
        if not isinstance(ref, dict):
            raise ValueError(f"qualification reference is invalid for {model_id}")
        lineage = source_pins["backend_strict_repeat_lineages"][model_id]
        qualification_pin = lineage.get("qualification", {})
        if any(ref.get(key) != qualification_pin.get(key) for key in ("path", "bytes", "sha256")):
            raise ValueError(f"qualification reference differs from the actual per-backend pin: {model_id}")
        path = _resolve_root_path(project_root, ref.get("path"), f"{model_id} qualification receipt")
        raw = read_regular(path, f"{model_id} qualification receipt", 16 * 1024 * 1024)
        digest = sha_bytes(raw)
        if ref.get("sha256") != digest or ref.get("bytes") != len(raw):
            raise ValueError(f"qualification receipt path/size/SHA differs from the filled seed for {model_id}")
        record = json.loads(raw.decode("utf-8"))
        if (not isinstance(record, dict) or record.get("schema") != "cross_tfg_backend_strict_determinism_qualification_v1"
                or record.get("status") != "PASS_EXACT_MATCH" or record.get("model_id") != model_id
                or record.get("gpu_uuid") != GPU_UUID
                or record.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]
                or record.get("policy_values") != STRICT_POLICY
                or record.get("receipt_sha256") != payload_sha(record, "receipt_sha256")):
            raise ValueError(f"actual strict repeat report is not a self-sealed same-GPU pass: {model_id}")
        if record.get("run_id") != lineage.get("manifest", {}).get("run_id"):
            raise ValueError(f"{model_id} qualification run ID differs from its own strict-repeat manifest pin")
        validated_lineage = _validate_qualification_lineage(
            project_root, model_id, lineage, ref, record, raw)
        if common_qualification_context is None:
            common_qualification_context = validated_lineage["input_context"]
        elif validated_lineage["input_context"] != common_qualification_context:
            raise ValueError(f"{model_id} strict-repeat input differs from the shared same-GPU QC06 context")
        entry = copy.deepcopy(record)
        entry["receipt"] = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": digest}
        entry["lineage"] = {
            "manifest": validated_lineage,
            "repeat_row_ids": validated_lineage["qualified_model_row_ids"],
        }
        runtime["backend_determinism_qualifications"][model_id] = entry

    roster_models = {model["model_id"]: model for model in roster["models"]}
    inv_backends = {item["model_id"]: item for item in inventory["backends"]}
    alias_paths = getattr(runner, "MUSE_ALLOWED_SYMLINKS")
    actual_dependencies: dict[str, Any] = {}
    total_unique: dict[str, tuple[int, str]] = {}
    python_paths = {"stagea": execution["stagea_python"], "syncnet": execution["syncnet_python"]}
    backend_directories: dict[str, Any] = {}
    for model_id in MODELS:
        observed = seed.get("backends", {}).get(model_id)
        if not isinstance(observed, dict):
            raise ValueError(f"seed lacks runtime generator descriptor for {model_id}")
        model = roster_models[model_id]
        expected_model_binding = {key: model.get(key) for key in MODEL_BINDING_FIELDS}
        if observed.get("model_binding") != expected_model_binding:
            raise ValueError(f"runtime model/scorer binding differs from frozen roster: {model_id}")
        generator = copy.deepcopy(observed.get("generator"))
        if not isinstance(generator, dict):
            raise ValueError(f"runtime generator descriptor is absent: {model_id}")
        for key, value in list(generator.items()):
            if isinstance(value, str) and value.startswith("@PROJECT_ROOT@/"):
                generator[key] = str(_resolve_root_path(project_root, value, f"{model_id} generator {key}"))
        if (generator.get("fps_num"), generator.get("fps_den")) != (model["fps_num"], model["fps_den"]):
            raise ValueError(f"runtime generator FPS differs from frozen roster: {model_id}")
        if model_id in STRICT_SHIMS:
            rel, byte_count, digest = STRICT_SHIMS[model_id]
            shim_path = (project_root / rel).resolve(strict=True)
            shim_raw = read_regular(shim_path, f"{model_id} strict runtime shim", 4 * 1024 * 1024)
            if len(shim_raw) != byte_count or sha_bytes(shim_raw) != digest:
                raise ValueError(f"current {model_id} strict shim differs from its reviewed source pin")
            generator["adapter_path"] = str(shim_path)
            generator["adapter_sha256"] = digest
            generator["child_script_sha256"] = digest
            generator["render_entry_path"] = model["generator"]["renderer_path"]
        else:
            generator["child_script_sha256"] = QC06["source_sha256"]
            if generator.get("python") != execution["wav2lip_python"] or generator.get("repo") != execution["wav2lip_root"]:
                raise ValueError("Wav2Lip runtime generator Python/repo differs from execution binding")
        if not isinstance(generator.get("python"), str) or not Path(generator["python"]).is_file():
            raise FileNotFoundError(f"backend interpreter is absent: {model_id}")
        if model_id in STRICT_SHIMS and generator.get("contract") != "cross_tfg_silent_video_cli_v1":
            raise ValueError(f"runtime adapter contract changed for {model_id}")
        if model_id in STRICT_SHIMS and generator.get("render_entry_path") != model["generator"].get("renderer_path"):
            raise ValueError(f"runtime native render entry changed for {model_id}")
        python_key = {MODELS[0]: "wav2lip", MODELS[1]: "muse", MODELS[2]: "ditto"}[model_id]
        python_paths[python_key] = generator["python"]
        observed["generator"] = generator

        expected_inv = inv_backends[model_id].get("files", [])
        expected_inventory = {inventory_id(model_id, item): item for item in expected_inv
                              if not (model_id in STRICT_SHIMS and item.get("role") == "adapter")}
        supplied = observed.get("dependencies")
        if not isinstance(supplied, list):
            raise ValueError(f"runtime dependency list is absent: {model_id}")
        supplied_by_id = {item.get("inventory_id"): item for item in supplied if isinstance(item, dict)}
        required_ids = set(expected_inventory) | {"strict-bootstrap:" + model_id}
        if model_id in STRICT_SHIMS:
            required_ids.add("runtime-strict-shim:" + model_id)
        if model_id == MODELS[0]:
            required_ids.update(item[0] for item in W2L_EXTRA_DEPENDENCIES)
        if set(supplied_by_id) != required_ids:
            raise ValueError(f"runtime dependency set is incomplete/extra for {model_id}")
        clean_deps: list[dict[str, Any]] = []
        for key in sorted(expected_inventory):
            pin = expected_inventory[key]
            item = supplied_by_id[key]
            if (item.get("sha256") != pin.get("expected_sha256")
                    or item.get("role") != pin.get("role")
                    or item.get("expected_legacy_path") != pin.get("legacy_literal_path")):
                raise ValueError(f"runtime dependency pin differs from immutable inventory: {model_id}/{key}")
            path = _resolve_root_path(project_root, item.get("path"), f"{model_id} dependency path")
            if not path.is_absolute():
                raise ValueError(f"dependency path is not absolute: {path}")
            expected_bytes = pin.get("expected_bytes")
            if item.get("bytes") not in (None, expected_bytes):
                raise ValueError(f"runtime dependency byte pin differs from frozen inventory: {path}")
            clean_deps.append({"path": str(path), "bytes": expected_bytes, "sha256": pin["expected_sha256"],
                               "role": pin["role"]})
        bootstrap_path = (project_root / STRICT_BOOTSTRAP_REL).resolve(strict=True)
        bootstrap_raw = read_regular(bootstrap_path, "strict startup bootstrap", 4 * 1024 * 1024)
        if len(bootstrap_raw) != STRICT_BOOTSTRAP_BYTES or sha_bytes(bootstrap_raw) != STRICT_BOOTSTRAP_SHA256:
            raise ValueError("strict startup bootstrap attempt03 source differs from its fixed pin")
        bootstrap_item = supplied_by_id["strict-bootstrap:" + model_id]
        if bootstrap_item.get("sha256") != STRICT_BOOTSTRAP_SHA256:
            raise ValueError("strict bootstrap dependency pin differs from attempt03")
        clean_deps.append({"path": str(bootstrap_path), "bytes": len(bootstrap_raw),
                           "sha256": STRICT_BOOTSTRAP_SHA256, "role": "strict_determinism_bootstrap"})
        if model_id in STRICT_SHIMS:
            rel, byte_count, digest = STRICT_SHIMS[model_id]
            shim_path = (project_root / rel).resolve(strict=True)
            shim_raw = read_regular(shim_path, f"{model_id} strict runtime shim", 4 * 1024 * 1024)
            shim_item = supplied_by_id["runtime-strict-shim:" + model_id]
            if shim_item.get("sha256") != digest or shim_item.get("bytes") != byte_count:
                raise ValueError(f"strict shim dependency pin is wrong: {model_id}")
            clean_deps.append({"path": str(shim_path), "bytes": len(shim_raw),
                               "sha256": digest, "role": "strict_runtime_adapter"})
        if model_id == MODELS[0]:
            for dep_id, expected_path, expected_bytes, expected_sha, role in W2L_EXTRA_DEPENDENCIES:
                item = supplied_by_id[dep_id]
                if (item.get("expected_legacy_path") != expected_path
                        or item.get("sha256") != expected_sha
                        or item.get("bytes") not in (None, expected_bytes)
                        or item.get("role") != role):
                    raise ValueError(f"Wav2Lip native runtime dependency pin differs from the reviewed source closure: {dep_id}")
                path = _resolve_root_path(project_root, item.get("path"), f"Wav2Lip native dependency {dep_id}")
                clean_deps.append({"path": str(path), "bytes": expected_bytes,
                                   "sha256": expected_sha, "role": role})
        # The native entry source remains a fixed frozen pin independent of the shim.
        if model_id in STRICT_SHIMS:
            entry_path, entry_sha = runner.MODEL_ENTRY_PINS[model_id]
            if not any(item["path"] == entry_path and item["sha256"] == entry_sha for item in clean_deps):
                raise ValueError(f"native renderer entry pin is absent from dependency closure: {model_id}")
        observed["dependencies"] = clean_deps
        if model_id == MODELS[1]:
            observed["approved_symlink_aliases"] = {
                path: runner._symlink_identity(Path(path), target)
                for path, target in alias_paths.items()
            }
            runner.validate_muse_alias_binding(observed)
        else:
            observed["approved_symlink_aliases"] = {}
        if model_id == MODELS[0]:
            native_entry_path = Path(str(generator.get("repo", ""))) / "inference.py"
            if str(native_entry_path) != W2L_NATIVE_ENTRY:
                raise ValueError("Wav2Lip runtime repo does not resolve to the exact official native entry")
            generator["native_entry_path"] = W2L_NATIVE_ENTRY
            generator["native_entry_sha256"] = W2L_NATIVE_ENTRY_SHA256
            generator["runtime_compat_sitecustomize_path"] = str(
                (project_root / W2L_RUNTIME_COMPAT_REL).resolve(strict=True))
            generator["runtime_compat_sitecustomize_sha256"] = W2L_RUNTIME_COMPAT_SHA256
            generator["strict_bootstrap_path"] = str(bootstrap_path)
            generator["strict_bootstrap_sha256"] = STRICT_BOOTSTRAP_SHA256
            generator["native_frame_count_probe_path"] = str(frame_probe_path.resolve(strict=True))
            generator["native_frame_count_probe_raw_sha256"] = W2L_FRAME_PROBE_SHA256
            generator["native_frame_count_probe_receipt_sha256"] = W2L_FRAME_PROBE_RECEIPT_SHA256
            if generator.get("repo") != execution["wav2lip_root"]:
                raise ValueError("Wav2Lip runtime native repo differs from the model inventory root")
            generator["child_script_sha256"] = W2L_NATIVE_ENTRY_SHA256
        backend_map = observed
        backend_map["model_binding"] = expected_model_binding
        runtime["backends"][model_id] = backend_map

        for dep in clean_deps:
            path = Path(dep["path"])
            if not path.is_absolute():
                raise ValueError(f"backend dependency path is not absolute: {path}")
            previous = total_unique.get(str(path))
            ident = (dep["bytes"] if isinstance(dep["bytes"], int) else -1, dep["sha256"])
            if previous and previous[1] != ident[1]:
                raise ValueError(f"one dependency path has conflicting whole-file SHA pins: {path}")
            total_unique[str(path)] = ident

        dirs: dict[str, Any] = {}
        for key, value in _directory_inputs(generator).items():
            dirs[key] = helper.directory_identity(Path(value), f"{model_id} {key}")
        backend_directories[model_id] = dirs

    dependency_total = 0
    verified_dependencies: list[dict[str, Any]] = []
    muse_aliases = runtime["backends"][MODELS[1]]["approved_symlink_aliases"]
    for model_id in MODELS:
        for dep in runtime["backends"][model_id]["dependencies"]:
            verified = runner.hash_regular(Path(dep["path"]), f"{model_id} whole dependency",
                                           dep["sha256"], dep["bytes"], allowed_aliases=muse_aliases)
            verified_dependencies.append({"model_id": model_id, "path": dep["path"],
                                          "bytes": verified["bytes"], "sha256": verified["sha256"],
                                          "role": dep["role"],
                                          "approved_symlink_aliases": verified["approved_symlink_aliases"]})
    unique: dict[str, tuple[int, str]] = {}
    for dep in verified_dependencies:
        prior = unique.get(dep["path"])
        ident = (dep["bytes"], dep["sha256"])
        if prior is not None and prior != ident:
            raise ValueError(f"shared dependency conflicts across backends: {dep['path']}")
        unique[dep["path"]] = ident
    dependency_total = sum(size for size, _digest in unique.values())
    if dependency_total > MAX_DEPENDENCY_BYTES:
        raise ValueError("whole dependency hash byte budget exceeds 16 GiB")
    runtime["backend_directories"] = backend_directories
    runtime["verified_dependency_files"] = verified_dependencies
    runtime["verified_unique_dependency_bytes"] = dependency_total

    # Verify the exact tool file hashes. Python executables are separately probed.
    tool_hashes = {}
    for key in ("generator_child", "strict_determinism_bootstrap", "syncnet_evaluator",
                "syncnet_checkpoint", "ffmpeg", "pcm16_lineage"):
        tool_hashes[key] = runner.hash_regular(Path(execution[key]), key, expected_hashes[key])
    if not isinstance(expected_hashes.get("ffprobe"), str) or len(expected_hashes["ffprobe"]) != 64:
        raise ValueError("runtime seed must provide an exact FFprobe SHA-256")
    tool_hashes["ffprobe"] = runner.hash_regular(Path(execution["ffprobe"]), "ffprobe", expected_hashes["ffprobe"])
    runtime["verified_tool_files"] = tool_hashes

    for key in ("stagea_python", "syncnet_python", "wav2lip_python"):
        if not Path(execution[key]).is_file():
            raise FileNotFoundError(f"required runtime interpreter is absent: {key}")
    for key in ("syncnet_root", "wav2lip_root"):
        if not Path(execution[key]).is_dir():
            raise FileNotFoundError(f"required runtime source directory is absent: {key}")
    runtime["python_environments"] = {
        "stagea": helper.probe_python(execution["stagea_python"]),
        "syncnet": helper.probe_python(execution["syncnet_python"]),
        "wav2lip": helper.probe_python(runtime["backends"][MODELS[0]]["generator"]["python"]),
        "muse": helper.probe_python(runtime["backends"][MODELS[1]]["generator"]["python"]),
        "ditto": helper.probe_python(runtime["backends"][MODELS[2]]["generator"]["python"]),
    }

    basis = _capacity(seed, project_root)
    resources_seed = seed["resources"]
    thresholds = {
        "min_gpu_free_mib": resources_seed.get("min_gpu_free_mib"),
        "min_ram_available_bytes": resources_seed.get("min_effective_ram_available_bytes"),
        "min_shm_free_bytes": basis["output_required_bytes"] + basis["output_reserve_bytes"],
        "min_output_free_bytes": basis["output_required_bytes"] + basis["output_reserve_bytes"],
    }
    if any(type(value) is not int or value <= 0 for value in thresholds.values()):
        raise ValueError("all current-host GPU/RAM/SHM/output resource minima must be positive integers")
    snapshot = helper.resource_snapshot(Path(execution["output_root"]), thresholds)
    gpu = next((item for item in snapshot["gpus"] if item.get("uuid") == GPU_UUID), None)
    if gpu is None:
        raise ValueError("current host does not expose the frozen QC06 GPU UUID")
    ram = snapshot["ram"]
    cg = ram["cgroup"]
    identity = cg.get("identity")
    if not isinstance(identity, dict) or cg.get("version") not in ("v1", "v2"):
        raise ValueError("current cgroup memory identity is incomplete")
    disk = snapshot["output_disk"]
    resource_record = {
        "gpu_uuid": GPU_UUID, "gpu_free_mib": int(gpu["memory_free_mib"]),
        "effective_ram_available_bytes": int(ram["available_bytes"]),
        "dev_shm_free_bytes": int(snapshot["shm"]["available_bytes"]),
        "output_mount_target": disk["target"], "output_free_bytes": int(disk["available_bytes"]),
        "output_required_bytes": basis["output_required_bytes"],
        "output_reserve_bytes": basis["output_reserve_bytes"],
        "min_gpu_free_mib": thresholds["min_gpu_free_mib"],
        "min_effective_ram_available_bytes": thresholds["min_ram_available_bytes"],
        "min_dev_shm_free_bytes": thresholds["min_shm_free_bytes"],
        "capacity_basis": basis, "capacity_basis_sha256": sha_bytes(canonical(basis)),
        "storage_gate_status": "PASS",
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cgroup": {
            "version": cg["version"], "current_bytes": cg["current_bytes"],
            "finite_limit": cg["finite_limit"], "limit_bytes": cg["limit_bytes"],
            "identity": identity,
            "membership_identity_sha256": sha_bytes(canonical({"membership_path": identity.get("membership_path"),
                                                                  "leaf_path": identity.get("leaf_path")})),
            "mount_identity_sha256": sha_bytes(canonical({"mount_target": identity.get("mount_target"),
                                                            "mount_root": identity.get("mount_root")})),
        },
        "resource_snapshot": snapshot,
        "resource_snapshot_sha256": sha_bytes(canonical(snapshot)),
    }
    runtime["resources"] = resource_record

    # Validate the exact three actual qualification receipt contents using the same guard as preflight.
    runner.validate_backend_determinism_qualifications(project_root, runtime)
    runtime["status"] = "PASS"
    runtime["binding_sha256"] = payload_sha(runtime, "binding_sha256")
    raw_out = pretty(runtime)
    write_exclusive(output, raw_out)
    return {"status": "PASS", "path": str(output.resolve()), "bytes": len(raw_out),
            "raw_sha256": sha_bytes(raw_out), "binding_sha256": runtime["binding_sha256"],
            "gpu_uuid": GPU_UUID, "unique_dependency_count": len(unique),
            "unique_dependency_bytes": dependency_total, "qualification_backend_count": len(MODELS),
            "resource_snapshot_sha256": resource_record["resource_snapshot_sha256"],
            "dev_confirm_content_opened": False, "model_execution_performed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-template", help="derive an unbound seed from frozen roster/inventory metadata")
    prep.add_argument("--output", type=Path, default=ROOT / NEW_TEMPLATE_REL)
    seal = sub.add_parser("seal-seed", help="mark a completed seed ready and self-seal it")
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--input-raw-sha256", required=True)
    seal.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("collect", help="hash current runtime files and write a PASS binding when all gates pass")
    run.add_argument("--seed", type=Path, required=True)
    run.add_argument("--seed-raw-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-template":
            result = prepare_template(args.output)
        elif args.command == "seal-seed":
            result = seal_seed(args.input, args.output, args.input_raw_sha256)
        else:
            result = collect(args.seed, args.seed_raw_sha256, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
