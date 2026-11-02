#!/usr/bin/env python3
"""Verify and summarize a locally reassembled frozen ROI-geometry extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def payload_object(path: Path, expected_status: str) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    obj = json.loads(raw)
    if obj.get("status") != expected_status:
        raise RuntimeError(f"unexpected status in {path.name}")
    payload = dict(obj)
    recorded = payload.pop("payload_sha256", None)
    computed = canonical_sha(payload)
    if recorded != computed:
        raise RuntimeError(f"canonical payload digest mismatch in {path.name}")
    return obj, hashlib.sha256(raw).hexdigest(), recorded


def diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        return {"same_shape_uint8": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "same_shape_uint8": True,
        "exact": bool(np.array_equal(left, right)),
        "max_abs_u8": int(delta.max(initial=0)),
        "mean_abs_u8": float(delta.mean()),
        "changed_fraction": float(np.count_nonzero(delta) / delta.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readback-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {args.output}")
    started = time.time()
    root = args.readback_dir
    collector_path = root / "roi_geometry_receipt.json"
    restore_path = root / "local_restore_receipt.json"
    transfer_path = root / "roi_geometry_transfer_manifest.json"
    npz_path = root / "roi_geometry_selected_frames.npz"
    collector, collector_raw_sha, collector_payload_sha = payload_object(
        collector_path, "PASS_READ_ONLY_SAVED_FRAMES"
    )
    restore, restore_raw_sha, restore_payload_sha = payload_object(
        restore_path, "PASS_REASSEMBLED_EXACT_COLLECTOR_BYTES"
    )
    transfer, transfer_raw_sha, transfer_payload_sha = payload_object(
        transfer_path, "PASS_READ_ONLY_CHUNK_COPY"
    )
    source_npz = collector.get("selected_arrays_npz", {})
    restored_npz = restore.get("reassembled_npz", {})
    if (npz_path.stat().st_size != int(source_npz.get("bytes", -1))
            or sha256_file(npz_path) != source_npz.get("sha256")
            or restored_npz.get("bytes") != source_npz.get("bytes")
            or restored_npz.get("sha256") != source_npz.get("sha256")):
        raise RuntimeError("reassembled NPZ does not match collector receipt")
    if (transfer.get("source_npz", {}).get("sha256") != source_npz.get("sha256")
            or transfer.get("source_receipt_sha256") != collector_raw_sha):
        raise RuntimeError("transfer manifest does not bind the collector NPZ and receipt")

    points = collector.get("points", [])
    if len(points) != 10:
        raise RuntimeError(f"expected ten saved-frame points, got {len(points)}")
    expected: dict[str, dict[str, Any]] = {}
    for point in points:
        sample = str(point["sample_id"])
        offset = int(point["offset_from_q0"])
        frame = int(point["source_clock_frame"])
        row_key = "".join(ch if ch.isalnum() else "_" for ch in sample)
        prefix = f"{row_key}_offset{offset:02d}_src{frame:03d}"
        expected[prefix] = point["arrays"]

    computed_points: list[dict[str, Any]] = []
    pair_diffs = {"native_vs_proxy_crop_224": [], "generated512_crop_vs_official_crop": [],
                  "decoded512_crop_vs_official_crop": [], "official_npz_vs_decoded_crop": [],
                  "native_vs_proxy_projected_patch_roi": []}
    with np.load(npz_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected_keys: set[str] = set()
        for prefix, arrays in expected.items():
            for name, seal in arrays.items():
                key = f"{prefix}_{name}"
                expected_keys.add(key)
                if key not in keys:
                    raise RuntimeError(f"ROI NPZ is missing sealed image array {key}")
                value = np.asarray(archive[key])
                if (list(value.shape) != seal.get("shape") or str(value.dtype) != seal.get("dtype")
                        or array_sha256(value) != seal.get("sha256")):
                    raise RuntimeError(f"ROI NPZ array seal mismatch: {key}")
        if keys != expected_keys:
            raise RuntimeError(f"ROI NPZ member set mismatch: expected {len(expected_keys)}, got {len(keys)}")

        for point in points:
            sample = str(point["sample_id"])
            offset = int(point["offset_from_q0"])
            frame = int(point["source_clock_frame"])
            row_key = "".join(ch if ch.isalnum() else "_" for ch in sample)
            prefix = f"{row_key}_offset{offset:02d}_src{frame:03d}"
            def arr(name: str) -> np.ndarray:
                return np.asarray(archive[f"{prefix}_{name}"])

            native = arr("native_official_crop_224")
            proxy = arr("proxy_224")
            native_reconstructed_generated = arr("native_reconstructed_from_generated512_224")
            native_reconstructed_decoded = arr("native_reconstructed_from_decoded512_224")
            native_crop_npz = arr("native_crop_npz_224")
            native_patch = arr("patch_native_u8")
            proxy_patch = arr("patch_proxy_u8")
            metrics = {
                "native_vs_proxy_crop_224": diff_stats(native, proxy),
                "generated512_crop_vs_official_crop": diff_stats(native_reconstructed_generated, native),
                "decoded512_crop_vs_official_crop": diff_stats(native_reconstructed_decoded, native),
                "official_npz_vs_decoded_crop": diff_stats(native_crop_npz, native),
                "native_vs_proxy_projected_patch_roi": diff_stats(native_patch, proxy_patch),
            }
            for key, value in metrics.items():
                pair_diffs[key].append(value)
            computed_points.append({
                "sample_id": sample,
                "row_id": point["row_id"],
                "source_clock_frame": frame,
                "timestamp_seconds": point["source_timestamp_seconds"],
                "crop_local_frame_index": point["crop_local_frame_index"],
                "context_frame_index": point["context_frame_index"],
                "processed_track_crop_geometry": point["processed_track_crop_geometry"],
                "frozen_proxy_patch_box_xyxy_in_224": point["frozen_proxy_patch_box_xyxy_in_224"],
                "projected_native_patch_box_xyxy_in_224": point["projected_native_patch_box_xyxy_in_224"],
                "patch_shapes": {"native": list(native_patch.shape), "proxy": list(proxy_patch.shape)},
                "metrics": metrics,
            })

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [item for item in items if item.get("same_shape_uint8")]
        if not valid:
            return {"comparable_count": 0, "count": len(items)}
        return {
            "count": len(items),
            "comparable_count": len(valid),
            "exact_count": sum(bool(item["exact"]) for item in valid),
            "max_abs_u8_max": max(int(item["max_abs_u8"]) for item in valid),
            "mean_abs_u8_mean": float(np.mean([item["mean_abs_u8"] for item in valid])),
            "changed_fraction_mean": float(np.mean([item["changed_fraction"] for item in valid])),
        }

    report = {
        "schema": "cross_tfg_full512_roi_geometry_local_analysis_v1",
        "status": "PASS_LOCAL_SEAL_AND_PIXEL_SUMMARY",
        "started_unix": started,
        "finished_unix": time.time(),
        "scope": "Seal verification and pixel-level comparison of ten saved source-clock frames from two historical fit clips; no new forward, scoring, or effects analysis.",
        "collector_receipt": {"path": str(collector_path), "bytes": collector_path.stat().st_size,
                               "raw_sha256": collector_raw_sha, "payload_sha256": collector_payload_sha},
        "transfer_manifest": {"path": str(transfer_path), "bytes": transfer_path.stat().st_size,
                              "raw_sha256": transfer_raw_sha, "payload_sha256": transfer_payload_sha},
        "restore_receipt": {"path": str(restore_path), "bytes": restore_path.stat().st_size,
                            "raw_sha256": restore_raw_sha, "payload_sha256": restore_payload_sha},
        "source_npz": {"path": str(npz_path), "bytes": npz_path.stat().st_size,
                       "sha256": sha256_file(npz_path), "array_count": len(expected) * 12},
        "point_count": len(computed_points),
        "aggregates": {key: aggregate(values) for key, values in pair_diffs.items()},
        "points": computed_points,
    }
    report["payload_sha256"] = canonical_sha(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "points": report["point_count"],
                      "array_count": report["source_npz"]["array_count"],
                      "output": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": sha256_file(args.output), "aggregates": report["aggregates"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
