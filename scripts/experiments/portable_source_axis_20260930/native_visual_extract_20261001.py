#!/usr/bin/env python3
"""Extract frozen MediaPipe tracks from the sealed native 4-cell video package.

This stage reads video pixels but does not compute phone effects. It verifies
the input package and writes one immutable feature file per intended clip/cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPECTED_PACKAGE_MANIFEST_SHA = "73c2f0d9993b9af5f070259ffb869efdf8105030e06b9be4f81e8098330ad0c8"
EXPECTED_EMBEDDED_MANIFEST_SHA = "1d274f37013793f722473590bd23f424bc0e66bfec17be65f909865984d65143"
EXPECTED_PACKAGE_FILES_SHA = "843f6f9914a0ef04947869c79004773b49fc35f12a2aeaafb21238dbea854add"
EXPECTED_MFA_ARCHIVE_SHA = "37fc220464d3eab6788f67fd88e79f76cf9af97963be9d7c65f7eb9a4fdbd6a9"
EXPECTED_SPEC_SHA = "b11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd"
EXPECTED_VISUAL_CORE_SHA = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
EXPECTED_TONE_ADAPTER_SHA = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
EXPECTED_VIDEO_FEATURES_SHA = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
EXPECTED_LANDMARKER_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_extract_tar(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[Path] = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(f"unsafe tar member path: {member.name!r}")
            if not member.isfile():
                raise ValueError(f"non-regular tar member is not allowed: {member.name!r}")
            target = destination.joinpath(*relative.parts).resolve()
            if root not in target.parents:
                raise ValueError(f"tar member escapes destination: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member: {member.name!r}")
            with source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            if target.stat().st_size != member.size:
                raise IOError(f"extracted size mismatch: {member.name}")
            extracted.append(target)
    return extracted


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def run(package_root: Path, outdir: Path, asset: Path) -> dict[str, Any]:
    started = time.monotonic()
    package_root = package_root.resolve(strict=True)
    asset = asset.resolve(strict=True)
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"output must be new/empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = package_root / "native_package_manifest.json"
    package_files_path = package_root / "package_files_sha256.json"
    mfa_archive = package_root / "original_mfa_and_source_plan.tar.gz"
    if sha256_file(manifest_path) != EXPECTED_PACKAGE_MANIFEST_SHA:
        raise ValueError("native package manifest raw SHA mismatch")
    if sha256_file(package_files_path) != EXPECTED_PACKAGE_FILES_SHA:
        raise ValueError("native package file inventory SHA mismatch")
    if sha256_file(mfa_archive) != EXPECTED_MFA_ARCHIVE_SHA:
        raise ValueError("original MFA bundle SHA mismatch")
    if sha256_file(asset) != EXPECTED_LANDMARKER_SHA:
        raise ValueError("MediaPipe task asset SHA mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_files = json.loads(package_files_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("manifest_sha256") != EXPECTED_EMBEDDED_MANIFEST_SHA:
        raise ValueError("native package embedded manifest seal mismatch")
    if manifest.get("spec_sha256") != EXPECTED_SPEC_SHA or manifest.get("archives_only_contain_inputs") is not True:
        raise ValueError("native package scope/spec binding mismatch")
    videos = manifest.get("videos", [])
    if len(videos) != 320 or manifest.get("cell_counts") != {
        "natural_fixed": 80, "natural_raw": 80, "tts_fixed": 80, "tts_raw": 80,
    }:
        raise ValueError("native video package is not the frozen 80x4 grid")
    sample_ids = sorted({str(row["sample_id"]) for row in videos})
    if len(sample_ids) != 80 or manifest.get("source_cohort", {}).get("speakers") != 40:
        raise ValueError("native video package cohort differs from 80 clips / 40 speakers")
    if manifest.get("original_mfa_evidence", {}).get("physical_overlay_applied") is not False:
        raise ValueError("native package did not preserve original physical-free MFA qualification")

    # Verify every packaged input before extraction. The inventory itself is
    # sealed, and each archive's bytes are bound by the native manifest.
    expected_inventory_names = {
        "native_package_manifest.json", "native_package_manifest.raw_sha256",
        "original_mfa_and_source_plan.tar.gz",
        *(f"video_shards/{row['name']}" for row in manifest["video_shards"]),
    }
    if set(package_files) != expected_inventory_names:
        raise ValueError("package file inventory member names differ from the sealed manifest")
    verified_package_files = 0
    for name, record in package_files.items():
        if name == "native_package_manifest.raw_sha256":
            raw_sha_text = f"{sha256_file(manifest_path)}  native_package_manifest.json\n".encode("ascii")
            if len(raw_sha_text) != int(record["bytes"]) or hashlib.sha256(raw_sha_text).hexdigest() != record["sha256"]:
                raise ValueError("manifest raw-hash sidecar binding mismatch")
            verified_package_files += 1
            continue
        local_name = PurePosixPath(name).name if name.startswith("video_shards/") else name
        path = package_root / local_name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"native package input hash/size mismatch: {name}")
        verified_package_files += 1

    source_root = outdir / "source"
    video_root = source_root / "videos"
    mfa_root = source_root / "mfa"
    feature_root = outdir / "features"
    video_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    extracted_mfa = safe_extract_tar(mfa_archive, mfa_root)
    if len([p for p in extracted_mfa if p.suffix == ".TextGrid"]) != 160:
        raise ValueError("original MFA archive did not extract exactly 160 TextGrids")

    video_by_destination: dict[str, dict[str, Any]] = {}
    for row in videos:
        cell = str(row["cell"])
        sample = str(row["sample_id"])
        relative = PurePosixPath("videos") / cell / f"{sample}__{cell}.mkv"
        if str(relative) in video_by_destination:
            raise ValueError(f"duplicate native video destination: {relative}")
        video_by_destination[str(relative)] = row
    extracted_video_records: dict[tuple[str, str], Path] = {}
    for shard in manifest["video_shards"]:
        shard_path = package_root / str(shard["name"])
        if sha256_file(shard_path) != shard["sha256"] or shard_path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"video shard hash/size mismatch: {shard['name']}")
        with tarfile.open(shard_path, "r:") as tar:
            members = tar.getmembers()
            if len(members) != int(shard["video_count"]):
                raise ValueError(f"video shard count mismatch: {shard['name']}")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"video shard has non-regular member: {member.name}")
                row = video_by_destination.get(member.name)
                if row is None:
                    raise ValueError(f"unexpected video member in shard: {member.name}")
                key = (str(row["sample_id"]), str(row["cell"]))
                if key in extracted_video_records:
                    raise ValueError(f"duplicate video member across shards: {key}")
                target = video_root.joinpath(*PurePosixPath(member.name).parts[1:]).resolve()
                if video_root.resolve() not in target.parents:
                    raise ValueError(f"video member escapes output root: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read video member: {member.name}")
                with source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                if target.stat().st_size != int(row["bytes"]) or sha256_file(target) != row["sha256"]:
                    raise ValueError(f"video member hash/size mismatch: {member.name}")
                extracted_video_records[key] = target
    if len(extracted_video_records) != 320:
        raise ValueError(f"expected 320 verified extracted videos, got {len(extracted_video_records)}")

    # Frozen extractor source and task asset checks precede all MediaPipe work.
    core_path = REPO / "scripts/experiments/tts_mechanism_visual_20260930.py"
    tone_path = REPO / "scripts/experiments/tts_mechanism_visual_tone_20261001.py"
    features_path = REPO / "scripts/experiments/lrs3_tts_visual_advantage/video_features.py"
    for path, expected, label in (
        (core_path, EXPECTED_VISUAL_CORE_SHA, "visual core"),
        (tone_path, EXPECTED_TONE_ADAPTER_SHA, "tone adapter"),
        (features_path, EXPECTED_VIDEO_FEATURES_SHA, "video feature extractor"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen {label} SHA mismatch")

    import cv2
    import mediapipe as mp
    from scripts.experiments.lrs3_tts_visual_advantage.video_features import (
        create_landmarker, extract_video_features, save_visual_sequence,
    )

    progress_path = outdir / "feature_attempts.jsonl"
    feature_records: list[dict[str, Any]] = []
    handle = create_landmarker(asset)
    try:
        for index, video_row in enumerate(sorted(videos, key=lambda r: (str(r["sample_id"]), str(r["cell"]))), 1):
            sample, cell = str(video_row["sample_id"]), str(video_row["cell"])
            video_path = extracted_video_records[(sample, cell)]
            feature_path = feature_root / sample / f"{cell}.npz"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                sequence = extract_video_features(video_path, asset, landmarker_handle=handle)
                frame_count = len(sequence.valid)
                expected_video = video_row["video"]
                if frame_count != int(expected_video["decoded_frames"]):
                    raise ValueError(f"decoded frame count differs from packaged probe: {frame_count} vs {expected_video['decoded_frames']}")
                if sequence.metadata["width"] != 512 or sequence.metadata["height"] != 512 or not np.isclose(float(sequence.metadata["fps"]), 25.0, atol=1e-6):
                    raise ValueError("decoded video geometry/FPS differs from frozen package")
                save_visual_sequence(feature_path, sequence)
                record = {
                    "sample_id": sample, "speaker_id": str(video_row["speaker_id"]), "split": "evaluation",
                    "cell": cell, "video_path": str(video_path.resolve()), "video_sha256": sha256_file(video_path),
                    "feature_path": str(feature_path.resolve()), "feature_sha256": sha256_file(feature_path),
                    "frame_count": int(frame_count), "fps": float(sequence.metadata["fps"]),
                    "valid_frames": int(np.asarray(sequence.valid, dtype=bool).sum()),
                    "valid_fraction": float(np.asarray(sequence.valid, dtype=bool).mean()),
                    "width": int(sequence.metadata["width"]), "height": int(sequence.metadata["height"]),
                    "timestamp_sha256": hashlib.sha256(np.asarray(sequence.timestamps_s, dtype="<f8").tobytes()).hexdigest(),
                    "valid_mask_sha256": hashlib.sha256(np.asarray(sequence.valid, dtype=np.uint8).tobytes()).hexdigest(),
                    "status": "extracted",
                }
                feature_records.append(record)
                write_jsonl(progress_path, {"index": index, **record})
            except Exception as exc:  # preserve failures without selecting rows
                failure = {"index": index, "sample_id": sample, "speaker_id": str(video_row["speaker_id"]),
                           "cell": cell, "video_sha256": str(video_row["sha256"]),
                           "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                write_jsonl(progress_path, failure)
                feature_records.append(failure)
    finally:
        handle[1].close()

    complete_rows = [row for row in feature_records if row.get("status") == "extracted"]
    failed_rows = [row for row in feature_records if row.get("status") != "extracted"]
    by_cell = {cell: sum(row.get("cell") == cell and row.get("status") == "extracted" for row in feature_records)
               for cell in ("natural_raw", "natural_fixed", "tts_raw", "tts_fixed")}
    result = {
        "schema_version": 1,
        "stage": "native_4cell_mediapipe_features_extracted",
        "status": "COMPLETE" if len(complete_rows) == 320 and not failed_rows else "INCOMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"planned_clips": 80, "speakers": 40, "clips_per_speaker": 2, "planned_videos": 320,
                   "completed_videos": len(complete_rows), "failed_videos": len(failed_rows), "completed_by_cell": by_cell},
        "input_seals": {"package_manifest_raw_sha256": sha256_file(manifest_path),
                        "package_manifest_embedded_sha256": manifest["manifest_sha256"],
                        "package_files_manifest_sha256": sha256_file(package_files_path),
                        "mfa_archive_sha256": sha256_file(mfa_archive), "spec_sha256": EXPECTED_SPEC_SHA,
                        "mfa_textgrid_count": 160, "physical_overlay_applied": False},
        "extraction": {"landmarker_asset_path": str(asset), "landmarker_asset_sha256": sha256_file(asset),
                       "visual_core_sha256": sha256_file(core_path), "tone_adapter_sha256": sha256_file(tone_path),
                       "video_features_source_sha256": sha256_file(features_path),
                       "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                                   "mediapipe": str(mp.__version__), "opencv": str(cv2.__version__),
                                   "platform": platform.platform(), "device": "CPU"},
                       "clock": "one timestamp per decoded frame at frame_index/fps; original 25fps clocks retained"},
        "features": feature_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(outdir / "extraction_manifest.json", result)
    feature_inventory = {
        row["feature_path"]: {"sha256": row["feature_sha256"], "bytes": Path(row["feature_path"]).stat().st_size}
        for row in complete_rows
    }
    atomic_json(outdir / "feature_inventory_sha256.json", feature_inventory)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--landmarker-asset", type=Path, default=REPO / "checkpoints/mediapipe/face_landmarker.task")
    args = parser.parse_args()
    result = run(args.package_root, args.outdir, args.landmarker_asset)
    print(json.dumps({"status": result["status"], "completed": result["cohort"]["completed_videos"],
                      "failed": result["cohort"]["failed_videos"], "output": str(args.outdir.resolve())}, separators=(",", ":")))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
