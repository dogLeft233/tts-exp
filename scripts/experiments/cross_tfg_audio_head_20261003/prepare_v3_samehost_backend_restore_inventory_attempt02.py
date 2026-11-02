#!/usr/bin/env python3
"""Build a metadata-only literal-path restore inventory for the three Stage-A backends.

The inventory is generated from the frozen backend roster plus the qualified
MuseTalk preflight path report. It lists old-host paths, expected SHA-256 pins,
and byte lengths where frozen metadata provides them. It does not stat or read
any current backend file and does not access development or confirmation media.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
ROSTER_REL = Path("runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json")
ROSTER_BYTES = 997_947
ROSTER_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
MUSE_PREFLIGHT_REL = Path("runs/cross_tfg_audio_head_20261003/native_encoder_wm_archive_readback_attempt01/restored_attempt01/frontends/muse/summary.json")
RESOURCE_RESPONSE_REL = Path("runs/cross_tfg_audio_head_20261003/dev180_newhost_runtime_inventory_attempt03/requests_attempt03/01_read_newhost_mount_resources_attempt03_request.json.response.json")
RESOURCE_RESPONSE_SHA256 = "be2d89c336b8835df0b79a545e21e9f51e9cc4abe509529a2b2f89810113fd27"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
CURRENT_RESOURCES = {
    "gpu_uuid": GPU_UUID,
    "cgroup_v2_available_bytes": 64_910_016_512,
    "dev_shm_available_bytes": 33_249_136_640,
    "overlay_available_bytes": 7_657_783_296,
    "autodl_tmp_xfs_available_bytes": 188_764_160,
    "backup_write_target_verified": False,
}
MUSE_ASSET_ROOTS = {
    "unet": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace/models/musetalkV15",
    "vae": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace/models/sd-vae-ft-mse",
    "whisper": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace/models/whisper",
    "face_parser": "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace/models/face-parse-bisent",
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_pinned(path: Path, label: str, expected_sha: str, expected_bytes: int | None = None) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    digest = sha_bytes(raw)
    if digest != expected_sha or (expected_bytes is not None and len(raw) != expected_bytes):
        raise ValueError(f"{label} does not match its reviewed raw pin")
    return raw, {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw), "raw_sha256": digest}


def storage_class(path: str) -> str:
    if path == "/dev/shm" or path.startswith("/dev/shm/"):
        return "legacy_shm_namespace_unverified_after_host_migration"
    if path.startswith("/root/autodl-tmp/"):
        return "legacy_autodl_tmp_mount_current_presence_unverified"
    if path.startswith("/root/tts-research-system-"):
        return "legacy_project_overlay_current_presence_unverified"
    if path.startswith("/root/autodl-tmp/"):
        return "legacy_local_mount_current_presence_unverified"
    return "legacy_path_current_presence_unverified"


def add_file(files: dict[tuple[str, str], dict[str, Any]], *, backend: str, role: str,
             logical_path: str, path: str, sha256: str, byte_count: int | None = None,
             evidence: str) -> None:
    if not PurePosixPath(path).is_absolute():
        raise ValueError(f"inventory path is not an absolute POSIX path: {path}")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError(f"invalid SHA-256 pin for {backend}:{logical_path}")
    key = (path, sha256)
    row = {
        "backend_id": backend,
        "role": role,
        "logical_path": logical_path,
        "legacy_literal_path": path,
        "expected_bytes": byte_count,
        "expected_sha256": sha256,
        "legacy_storage_class": storage_class(path),
        "current_host_presence": "NOT_PROBED",
        "restoration_source": "RECOVER_OR_RESTAGE_EXACT_PIN",
        "pin_evidence": evidence,
    }
    old = files.get(key)
    if old is None:
        files[key] = row
    elif old["expected_bytes"] is None and byte_count is not None:
        old["expected_bytes"] = byte_count
    elif old["expected_bytes"] is not None and byte_count is not None and old["expected_bytes"] != byte_count:
        raise ValueError(f"conflicting byte lengths for pinned file: {path}")


def make_inventory(roster: dict[str, Any], roster_meta: dict[str, Any],
                   muse_evidence: dict[str, Any], resource_evidence: dict[str, Any]) -> dict[str, Any]:
    if tuple(model.get("model_id") for model in roster.get("models", [])) != MODELS:
        raise ValueError("frozen three-backend roster order changed")
    files: dict[tuple[str, str], dict[str, Any]] = {}
    directories: list[dict[str, str]] = []
    for model in roster["models"]:
        backend = model["model_id"]
        generator = model.get("generator", {})
        if backend == MODELS[0]:
            repo = generator["repo"]
            for rel, digest in sorted(model.get("source_sha256", {}).items()):
                add_file(files, backend=backend, role="backend_source", logical_path=rel,
                         path=f"{repo}/{rel}", sha256=digest, byte_count=None,
                         evidence="frozen_roster.model.source_sha256")
            add_file(files, backend=backend, role="backend_checkpoint", logical_path="checkpoint",
                     path=generator["checkpoint"], sha256=generator["checkpoint_sha256"],
                     byte_count=None, evidence="frozen_roster.model.generator.checkpoint_sha256")
            directories.extend([
                {"backend_id": backend, "role": "backend_repository", "legacy_literal_path": repo,
                 "current_host_presence": "NOT_PROBED"},
                {"backend_id": backend, "role": "python_environment", "legacy_literal_path": generator["python"],
                 "current_host_presence": "NOT_PROBED"},
            ])
        elif backend == MODELS[1]:
            source_root = generator["model_source_root"]
            for rel, digest in sorted(model.get("source_sha256", {}).items()):
                add_file(files, backend=backend, role="backend_source", logical_path=rel,
                         path=f"{source_root}/{rel}", sha256=digest, byte_count=None,
                         evidence="frozen_roster.model.source_sha256")
            for category, records in sorted(model.get("model_assets", {}).items()):
                for asset in records:
                    asset_path = f"{MUSE_ASSET_ROOTS[category]}/{asset['relative_path']}"
                    add_file(files, backend=backend, role=f"model_asset:{category}",
                             logical_path=f"{category}/{asset['relative_path']}", path=asset_path,
                             sha256=asset["sha256"], byte_count=int(asset["bytes"]),
                             evidence="frozen_roster.model.model_assets")
            extra = (
                ("adapter", "generator.adapter_path", generator["adapter_path"], generator["adapter_sha256"]),
                ("renderer", "generator.renderer_path", generator["renderer_path"], generator["renderer_sha256"]),
                ("musetalk_adapter", "generator.musetalk_adapter_path", generator["musetalk_adapter_path"], generator["musetalk_adapter_sha256"]),
                ("overdub_helper", "generator.overdub_helper_path", generator["overdub_helper_path"], generator["overdub_helper_sha256"]),
                ("qualification_report", "generator.qualification_report_path", generator["qualification_report_path"], generator["qualification_report_sha256"]),
            )
            for role, evidence, path, digest in extra:
                add_file(files, backend=backend, role=role, logical_path=evidence,
                         path=path, sha256=digest, byte_count=None, evidence=f"frozen_roster.model.{evidence.rsplit('.',1)[-1]}")
            whisper_module = muse_evidence["runtime_source_details"]["source_files"]["transformers_whisper_feature_extractor.py"]
            add_file(files, backend=backend, role="runtime_overlay_source",
                     logical_path="transformers/models/whisper/feature_extraction_whisper.py",
                     path=whisper_module["path"], sha256=whisper_module["sha256"],
                     byte_count=int(whisper_module["bytes"]), evidence="historical_qualified_MuseTalk_preflight")
            for role, path in (
                ("python_environment", generator["python"]),
                ("model_source_root", source_root),
                ("model_workspace", generator["model_workspace"]),
                ("model_code_root", generator["model_code_root"]),
                ("model_overlay_root", generator["model_overlay_roots"][0]),
                ("model_overlay_root", generator["model_overlay_roots"][1]),
            ):
                directories.append({"backend_id": backend, "role": role,
                                    "legacy_literal_path": path, "current_host_presence": "NOT_PROBED"})
        elif backend == MODELS[2]:
            for rel, source in sorted(model.get("source_sha256", {}).items()):
                if isinstance(source, str):
                    path = f"{generator['ditto_source_root']}/{rel}"
                    digest = source
                    byte_count = None
                else:
                    path = source["path"]
                    digest = source["sha256"]
                    byte_count = int(source["bytes"])
                add_file(files, backend=backend, role="backend_source", logical_path=rel,
                         path=path, sha256=digest, byte_count=byte_count,
                         evidence="frozen_roster.model.source_sha256")
            assets = model.get("model_assets", {})
            for section, records in sorted(assets.items()):
                if isinstance(records, Mapping):
                    iterator = records.items()
                elif isinstance(records, list):
                    iterator = enumerate(records)
                else:
                    continue
                for name, asset in iterator:
                    if not isinstance(asset, Mapping) or not isinstance(asset.get("path"), str):
                        continue
                    add_file(files, backend=backend, role=f"model_asset:{section}",
                             logical_path=str(name), path=asset["path"],
                             sha256=asset["sha256"], byte_count=int(asset["bytes"]),
                             evidence="frozen_roster.model.model_assets")
            for role, field, digest_field in (
                ("adapter", "adapter_path", "adapter_sha256"),
                ("renderer", "renderer_path", "renderer_sha256"),
                ("runtime_extension", "runtime_extension_path", "runtime_extension_sha256"),
            ):
                path, digest = generator.get(field), generator.get(digest_field)
                if path and digest:
                    add_file(files, backend=backend, role=role, logical_path=field,
                             path=path, sha256=digest, byte_count=None,
                             evidence=f"frozen_roster.model.generator.{digest_field}")
            for role, path in (
                ("python_environment", generator["python"]),
                ("model_source_root", generator["ditto_source_root"]),
                ("model_overlay_root", generator["model_overlay_roots"][0]),
                ("model_overlay_root", generator["model_overlay_roots"][1]),
                ("model_overlay_root", generator["model_overlay_roots"][2]),
                ("model_overlay_root", generator["model_overlay_roots"][3]),
                ("model_overlay_root", generator["model_overlay_roots"][4]),
                ("weights_root", generator["ditto_weights_root"]),
            ):
                directories.append({"backend_id": backend, "role": role,
                                    "legacy_literal_path": path, "current_host_presence": "NOT_PROBED"})
        else:
            raise AssertionError(backend)

    ordered_files = sorted(files.values(), key=lambda item: (item["backend_id"], item["legacy_literal_path"], item["role"]))
    known_bytes = sum(item["expected_bytes"] or 0 for item in ordered_files)
    unknown_bytes = sum(item["expected_bytes"] is None for item in ordered_files)
    result: dict[str, Any] = {
        "schema": "cross_tfg_v3_samehost_backend_restore_identity_manifest_v2",
        "status": "FROZEN_IDENTITIES_AND_LEGACY_PATHS_ENUMERATED_CURRENT_HOST_PRESENCE_NOT_VERIFIED",
        "scope": "Metadata-only restore inventory. No current backend path was statted; no model file, media, or confirmation input was opened; no model, scorer, renderer, or API was invoked.",
        "current_host": {
            "gpu_uuid": GPU_UUID,
            "resource_observation": CURRENT_RESOURCES,
            "probe_response": resource_evidence,
            "meaning": "The probe measured current capacity and mounts only. It did not test backend path presence or writable backup capacity.",
        },
        "frozen_roster": roster_meta | {"plan_sha256": roster.get("plan_sha256"), "status": roster.get("status")},
        "muse_qualified_preflight_evidence": muse_evidence["evidence_pin"],
        "inventory_summary": {
            "backend_count": len(MODELS),
            "file_pin_count": len(ordered_files),
            "unique_file_identity_count": len({(item["legacy_literal_path"], item["expected_sha256"]) for item in ordered_files}),
            "known_file_bytes_lower_bound": known_bytes,
            "unknown_file_byte_count": unknown_bytes,
            "full_source_environment_size_known": False,
            "generated_media_or_output_capacity_included": False,
        },
        "backends": [
            {
                "model_id": backend,
                "model_sha256": next(m["model_sha256"] for m in roster["models"] if m["model_id"] == backend),
                "source_commit": next(m.get("source_commit", m.get("git_head")) for m in roster["models"] if m["model_id"] == backend),
                "status": "NOT_VERIFIED_ON_CURRENT_MIGRATED_HOST",
                "files": [row for row in ordered_files if row["backend_id"] == backend],
                "directories": [row for row in directories if row["backend_id"] == backend],
            }
            for backend in MODELS
        ],
        "restore_rules": [
            "Treat every /dev/shm path as a legacy namespace path; restage from a protected source or verified persistent copy, then verify exact byte count and SHA-256.",
            "For non-SHM paths, check the literal path first; if moved, provide a new runtime binding with actual absolute paths and file pins.",
            "The current SHM has room for the listed known model assets as a byte-count lower bound, but the total deployment, dependency, cache, and run-output footprint is not established by this inventory.",
            "No directory or endpoint is a backup target until current writable access and required retained-output space are measured and explicitly attested.",
        ],
    }
    result["manifest_sha256"] = payload_sha(result, "manifest_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_restore_identity_manifest_attempt02.json")
    args = parser.parse_args()
    roster_raw, roster_meta = read_pinned(ROOT / ROSTER_REL, "frozen roster", ROSTER_SHA256, ROSTER_BYTES)
    roster = json.loads(roster_raw.decode("utf-8"))
    if roster.get("status") != "FROZEN":
        raise ValueError("backend roster is not frozen")
    muse_raw = (ROOT / MUSE_PREFLIGHT_REL).read_bytes()
    muse = json.loads(muse_raw.decode("utf-8"))
    if (muse.get("schema") != "cross_tfg_native_audio_encoder_probe_v1"
            or muse.get("model_id") != MODELS[1]
            or muse.get("runtime_source_details", {}).get("source_commit") != "0a89dec45a0192b824e3cf4daf96c239440c5ed8"):
        raise ValueError("MuseTalk historical preflight path evidence does not match the frozen roster")
    resource_raw, resource_meta = read_pinned(ROOT / RESOURCE_RESPONSE_REL, "current mount/resource response", RESOURCE_RESPONSE_SHA256)
    resource_value = json.loads(resource_raw.decode("utf-8"))
    if not isinstance(resource_value, dict) or not resource_value.get("ok"):
        raise ValueError("current mount/resource response is not a successful raw response")
    stdout = json.loads(resource_value["stdout"])
    memory = stdout["memory_bytes"]
    cgroup = memory["cgroup"]
    if (stdout.get("gpu", {}).get("rows", [None])[0] is None
            or "GPU-d09622d0-3535-e662-bb45-3df2b049fcea" not in stdout["gpu"]["rows"][0]
            or cgroup.get("available_bytes") != CURRENT_RESOURCES["cgroup_v2_available_bytes"]
            or next(item["available_bytes"] for item in stdout["candidate_paths"] if item["target"] == "/dev/shm") != CURRENT_RESOURCES["dev_shm_available_bytes"]
            or next(item["available_bytes"] for item in stdout["candidate_paths"] if item["target"] == "/") != CURRENT_RESOURCES["overlay_available_bytes"]
            or next(item["available_bytes"] for item in stdout["candidate_paths"] if item["target"] == "/root/autodl-tmp") != CURRENT_RESOURCES["autodl_tmp_xfs_available_bytes"]):
        raise ValueError("actual current resource values differ from the reviewed probe response")
    evidence = {
        "path": MUSE_PREFLIGHT_REL.as_posix(),
        "bytes": len(muse_raw),
        "raw_sha256": sha_bytes(muse_raw),
        "status": muse.get("status"),
        "use": "historical qualified path/asset identity metadata only",
    }
    value = make_inventory(roster, roster_meta, {"runtime_source_details": muse["runtime_source_details"], "evidence_pin": evidence}, resource_meta)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pretty(value)
    out.write_bytes(raw)
    print(json.dumps({"status": value["status"], "path": str(out), "bytes": len(raw), "raw_sha256": sha_bytes(raw),
                      "manifest_sha256": value["manifest_sha256"], "file_pin_count": value["inventory_summary"]["file_pin_count"],
                      "known_file_bytes_lower_bound": value["inventory_summary"]["known_file_bytes_lower_bound"],
                      "unknown_file_byte_count": value["inventory_summary"]["unknown_file_byte_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
