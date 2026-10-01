#!/usr/bin/env python3
"""Recover the isolated visual CPU runtime from the verified attempt12 tree.

Attempt12's extracted wheels are immutable.  This utility copies that already
verified Python target to an executable system filesystem, compares every tree
member before and after the copy, then validates the pinned imports and
MediaPipe Tasks API without changing the base Wav2Lip environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

SOURCE = Path("/dev/shm/tts-research-20261001/world_visual_runtime_attempt12/python_target")
TARGET = Path("/root/tts-research-system-20261001/deps/world_visual_python_target_attempt13")
JOB = Path("/root/tts-research-system-20261001/job_control/world_visual_runtime_attempt13")
PYTHON = Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
ASSET = Path("/dev/shm/tts-research-20261001/world_visual_runtime_attempt05/assets/face_landmarker.task")
HELPER = Path("/root/tts-research-system-20261001/code/world_visual_attempt01/video_features.py")
STAGE_RECEIPT = Path("/root/autodl-tmp/tts-research-storage-20261001/job_control/world_visual_runtime_attempt12/stage_receipt.json")
BUNDLE_MANIFEST = Path("/dev/shm/tts-research-20261001/world_visual_runtime_attempt12/offline_bundle/bundle_manifest.json")
ASSET_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
HELPER_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
ATTEMPT12_INSTALLER_SHA256 = "308b9713ad129056672762b919726e792f4e2cad8b8195c0609bf7b3ae336422"
ATTEMPT12_WORKER_SHA256 = "e28a26b73e1893f2c0f5b9c08c6f4bf3efea25e31296dd3583f427abd96ae461"
ATTEMPT12_BUNDLE_SHA256 = "a0c2b1230cc195e56947f8c5c546cd1eb65eb4b0ca5d815658cee47d49be6c8f"
ATTEMPT12_TAR_SHA256 = "d2508a14905d87e8b9304f21b3bf2e0dc58e4b5a0c847ed71b8ce5991186afa9"
ATTEMPT12_WHEEL_MANIFEST_SHA256 = "006a7bc191eb39b350e302ffad341459953284a0048aa0e75311fc5951c1cae3"
MIN_FILE_COUNT = 2059
MAX_FILE_COUNT = 2070
MIN_FILE_BYTES = 315650474
MAX_FILE_BYTES = 330000000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    root = root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            entries.append({"path": relative, "type": "symlink", "mode": mode, "target": os.readlink(path)})
        elif stat.S_ISDIR(metadata.st_mode):
            entries.append({"path": relative, "type": "directory", "mode": mode})
        elif stat.S_ISREG(metadata.st_mode):
            file_sha = sha256_file(path)
            entries.append({"path": relative, "type": "file", "mode": mode,
                            "size_bytes": metadata.st_size, "sha256": file_sha})
            total_bytes += metadata.st_size
        else:
            raise RuntimeError(f"unsupported_filesystem_entry:{relative}")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return entries, hashlib.sha256(encoded).hexdigest(), total_bytes


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def copy_verify() -> dict[str, Any]:
    if not SOURCE.is_dir() or not JOB.is_dir():
        raise RuntimeError("source_or_job_directory_missing")
    if TARGET.exists():
        raise RuntimeError("refusing_existing_target")
    if sha256_file(BUNDLE_MANIFEST) != ATTEMPT12_BUNDLE_SHA256:
        raise RuntimeError("attempt12_bundle_manifest_hash_mismatch")
    with BUNDLE_MANIFEST.open("r", encoding="utf-8") as stream:
        bundle = json.load(stream)
    stage = json.loads(STAGE_RECEIPT.read_text(encoding="utf-8"))
    if (stage.get("status") != "STAGE_PASS"
            or stage.get("installer_sha256") != ATTEMPT12_INSTALLER_SHA256
            or stage.get("worker_sha256") != ATTEMPT12_WORKER_SHA256
            or stage.get("bundle_manifest_sha256") != ATTEMPT12_BUNDLE_SHA256
            or stage.get("tar_sha256") != ATTEMPT12_TAR_SHA256
            or stage.get("wheel_manifest_sha256") != ATTEMPT12_WHEEL_MANIFEST_SHA256
            or bundle.get("tar_sha256") != ATTEMPT12_TAR_SHA256
            or bundle.get("wheel_manifest_sha256") != ATTEMPT12_WHEEL_MANIFEST_SHA256):
        raise RuntimeError("attempt12_stage_receipt_or_bundle_lineage_mismatch")
    before, before_hash, before_bytes = tree_inventory(SOURCE)
    files = [entry for entry in before if entry["type"] == "file"]
    links = [entry for entry in before if entry["type"] == "symlink"]
    if not (MIN_FILE_COUNT <= len(files) <= MAX_FILE_COUNT and MIN_FILE_BYTES <= before_bytes <= MAX_FILE_BYTES):
        raise RuntimeError("attempt12_source_tree_outside_pinned_safety_bounds")
    if links:
        raise RuntimeError("unexpected_symlinks_in_offline_target")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, TARGET, symlinks=False, copy_function=shutil.copy2)
    after, after_hash, after_bytes = tree_inventory(TARGET)
    if before_hash != after_hash or before != after or before_bytes != after_bytes:
        raise RuntimeError("copied_tree_differs_from_verified_source")
    receipt = {
        "status": "TREE_COPY_PASS",
        "recovery_utility_sha256": sha256_file(Path(__file__).resolve()),
        "source": str(SOURCE),
        "target": str(TARGET),
        "attempt12_stage_receipt_sha256": sha256_file(STAGE_RECEIPT),
        "attempt12_bundle_manifest_sha256": sha256_file(BUNDLE_MANIFEST),
        "attempt12_installer_sha256": ATTEMPT12_INSTALLER_SHA256,
        "attempt12_worker_sha256": ATTEMPT12_WORKER_SHA256,
        "attempt12_tar_sha256": ATTEMPT12_TAR_SHA256,
        "attempt12_wheel_manifest_sha256": ATTEMPT12_WHEEL_MANIFEST_SHA256,
        "source_member_count": len(before),
        "regular_file_count": len(files),
        "symlink_count": len(links),
        "regular_file_bytes": before_bytes,
        "source_tree_inventory_sha256": before_hash,
        "target_tree_inventory_sha256": after_hash,
        "source_file_sha256_index": {entry["path"]: entry["sha256"] for entry in files},
        "target_file_sha256_index": {entry["path"]: entry["sha256"] for entry in after if entry["type"] == "file"},
        "executable_mount_required": True,
        "base_environment_modified": False,
        "gpu_forward": False,
    }
    write_receipt(JOB / "tree_copy_receipt.json", receipt)
    return {key: value for key, value in receipt.items() if not key.endswith("sha256_index")}


def runtime_check() -> dict[str, Any]:
    if not TARGET.is_dir() or not JOB.is_dir():
        raise RuntimeError("verified_target_or_job_missing")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("unexpected_python_executable")
    if sha256_file(ASSET) != ASSET_SHA256 or sha256_file(HELPER) != HELPER_SHA256:
        raise RuntimeError("landmarker_or_helper_hash_mismatch")
    with (JOB / "tree_copy_receipt.json").open("r", encoding="utf-8") as stream:
        copy_receipt = json.load(stream)
    _, current_tree_hash, current_bytes = tree_inventory(TARGET)
    if (copy_receipt.get("status") != "TREE_COPY_PASS"
            or copy_receipt.get("target_tree_inventory_sha256") != current_tree_hash
            or current_bytes != copy_receipt.get("regular_file_bytes")
            or not (MIN_FILE_BYTES <= current_bytes <= MAX_FILE_BYTES)):
        raise RuntimeError("target_tree_no_longer_matches_copy_receipt")

    sys.path.insert(0, str(TARGET))
    sys.path.insert(0, str(HELPER.parent))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import numpy as np
    import cv2
    import mediapipe as mp
    import google.protobuf
    import absl
    import flatbuffers
    import attrs
    import matplotlib
    import matplotlib.pyplot  # noqa: F401
    import PIL
    import video_features

    actual_versions = {
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "mediapipe": mp.__version__,
        "protobuf": google.protobuf.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
    }
    expected_versions = {
        "numpy": "1.26.4", "opencv": "4.10.0", "mediapipe": "0.10.21",
        "protobuf": "4.25.8", "matplotlib": "3.9.4", "pillow": "11.0.0",
    }
    if any(actual_versions.get(name) != expected for name, expected in expected_versions.items()):
        raise RuntimeError("pinned_runtime_version_mismatch")
    if Path(video_features.__file__).resolve() != HELPER.resolve():
        raise RuntimeError("unexpected_video_features_module_path")
    if sha256_file(Path(video_features.__file__).resolve()) != HELPER_SHA256:
        raise RuntimeError("video_features_helper_changed")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    _mp, task = video_features.create_landmarker(ASSET.resolve())
    task.close()
    module_paths = {
        "numpy": str(Path(np.__file__).resolve()),
        "opencv": str(Path(cv2.__file__).resolve()),
        "mediapipe": str(Path(mp.__file__).resolve()),
        "protobuf": str(Path(google.protobuf.__file__).resolve()),
        "absl": str(Path(absl.__file__).resolve()),
        "flatbuffers": str(Path(flatbuffers.__file__).resolve()),
        "attrs": str(Path(attrs.__file__).resolve()),
        "matplotlib": str(Path(matplotlib.__file__).resolve()),
        "pillow": str(Path(PIL.__file__).resolve()),
        "video_features": str(Path(video_features.__file__).resolve()),
    }
    for module in ("opencv", "mediapipe", "protobuf", "absl", "flatbuffers", "attrs", "matplotlib", "pillow"):
        if not module_paths[module].startswith(str(TARGET)):
            raise RuntimeError(f"module_did_not_load_from_isolated_target:{module}")
    if str(TARGET) in module_paths["numpy"]:
        raise RuntimeError("numpy_was_shadowed_by_isolated_target")
    receipt = {
        "status": "RUNTIME_VALIDATION_PASS",
        "recovery_utility_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "target": str(TARGET),
        "target_tree_inventory_sha256": current_tree_hash,
        "target_regular_file_bytes": current_bytes,
        "landmarker_asset_sha256": sha256_file(ASSET),
        "video_features_sha256": sha256_file(HELPER),
        "versions": actual_versions,
        "module_paths": module_paths,
        "mediapipe_tasks_landmarker_create_close": "PASS",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "base_environment_modified": False,
        "gpu_forward": False,
    }
    write_receipt(JOB / "runtime_validation_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("copy-verify", "runtime-check"))
    args = parser.parse_args()
    value = copy_verify() if args.mode == "copy-verify" else runtime_check()
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
