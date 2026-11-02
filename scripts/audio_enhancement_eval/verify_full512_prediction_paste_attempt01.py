#!/usr/bin/env python3
"""Compare saved Wav2Lip 96px predictions with their saved 512px face patches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUMMARY_PATH = Path("/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json")
SUMMARY_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
ROI_OUTPUT = Path("/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05/output")
ROI_RECEIPT_SHA256 = "c5be31bfaa2f3fe5e1b937a63720292db85858399d91d39bc4ddbe79d5688523"
ROI_NPZ_SHA256 = "6c14ebce3c2704f34aae39e82a3c345988b83e8af78577d921bc51dbf7a960ea"
ROI_NPZ_BYTES = 15603354
W2L_INFERENCE = Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py")
W2L_INFERENCE_SHA256 = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
SELECTED = ("S0052_BAC009S0052W0121", "S0055_BAC009S0055W0122")


class ProbeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(body)


def checked(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError(f"required pinned input is missing: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_bytes is not None and size != expected_bytes:
        raise ProbeError(f"input size mismatch for {path}")
    if digest != expected_sha:
        raise ProbeError(f"input SHA mismatch for {path}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def pixel_diff(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        return {"comparable": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"comparable": True, "exact": bool(np.array_equal(left, right)),
            "max_abs_u8": int(delta.max(initial=0)), "mean_abs_u8": float(delta.mean()),
            "changed_fraction": float(np.count_nonzero(delta) / delta.size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite saved-frame paste probe output: {output_dir}")
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_files = {
        "parity_summary": checked(SUMMARY_PATH, SUMMARY_SHA256, 171086),
        "collector_receipt": checked(ROI_OUTPUT / "roi_geometry_receipt.json", ROI_RECEIPT_SHA256, 73632),
        "saved_frame_npz": checked(ROI_OUTPUT / "roi_geometry_selected_frames.npz", ROI_NPZ_SHA256, ROI_NPZ_BYTES),
        "wav2lip_inference_source": checked(W2L_INFERENCE, W2L_INFERENCE_SHA256),
    }
    receipt = json.loads((ROI_OUTPUT / "roi_geometry_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_READ_ONLY_SAVED_FRAMES":
        raise ProbeError("saved-frame collector receipt is not PASS")
    payload = dict(receipt)
    recorded_payload = payload.pop("payload_sha256", None)
    if recorded_payload != canonical_sha(payload):
        raise ProbeError("saved-frame collector payload digest mismatch")
    if receipt.get("selected_arrays_npz", {}).get("sha256") != ROI_NPZ_SHA256:
        raise ProbeError("collector receipt does not bind the expected NPZ")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ProbeError("pinned parity summary is not COMPLETE")
    rows = {}
    for row in summary.get("rows", []):
        if row.get("sample_id") in SELECTED and row.get("method_id") == "rms_fixed":
            rows[str(row["sample_id"])] = row
    if set(rows) != set(SELECTED):
        raise ProbeError("pinned parity summary does not contain the two selected RMS fit rows")

    points_by_sample: dict[str, list[dict[str, Any]]] = {sample: [] for sample in SELECTED}
    for point in receipt.get("points", []):
        points_by_sample[str(point.get("sample_id"))].append(point)
    if any(len(points_by_sample[sample]) != 5 for sample in SELECTED):
        raise ProbeError("collector receipt does not contain five frames per selected sample")

    results: list[dict[str, Any]] = []
    with np.load(ROI_OUTPUT / "roi_geometry_selected_frames.npz", allow_pickle=False) as archive:
        for sample_id in SELECTED:
            for point in points_by_sample[sample_id]:
                offset = int(point["offset_from_q0"])
                frame_index = int(point["source_clock_frame"])
                row_key = "".join(ch if ch.isalnum() else "_" for ch in sample_id)
                prefix = f"{row_key}_offset{offset:02d}_src{frame_index:03d}"
                predicted96 = np.asarray(archive[f"{prefix}_pred96_rgb_or_bgr"])
                generated512 = np.asarray(archive[f"{prefix}_native_full512_generated"])
                if predicted96.shape != (96, 96, 3) or predicted96.dtype != np.uint8:
                    raise ProbeError(f"saved prediction geometry/dtype mismatch for {prefix}")
                if generated512.shape != (512, 512, 3) or generated512.dtype != np.uint8:
                    raise ProbeError(f"saved generated 512 geometry/dtype mismatch for {prefix}")
                array_refs = point.get("arrays", {})
                for suffix, value in (("pred96_rgb_or_bgr", predicted96),
                                      ("native_full512_generated", generated512)):
                    seal = array_refs.get(suffix, {})
                    if (list(value.shape) != seal.get("shape") or str(value.dtype) != seal.get("dtype")
                            or array_sha(value) != seal.get("sha256")):
                        raise ProbeError(f"selected array seal mismatch for {prefix}/{suffix}")
                y1, y2, x1, x2 = [int(v) for v in point["wav2lip_static_face_box_y1_y2_x1_x2"]]
                if not (0 <= y1 < y2 <= 512 and 0 <= x1 < x2 <= 512):
                    raise ProbeError(f"saved static face box is invalid for {prefix}")
                observed_patch = np.ascontiguousarray(generated512[y1:y2, x1:x2])
                resized_linear = cv2.resize(predicted96, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                resized_linear_swapped = cv2.resize(predicted96[:, :, ::-1].copy(),
                                                    (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                results.append({
                    "sample_id": sample_id,
                    "row_id": point["row_id"],
                    "source_clock_frame": frame_index,
                    "timestamp_seconds": point["source_timestamp_seconds"],
                    "prediction96_array_sha256": array_sha(predicted96),
                    "generated512_array_sha256": array_sha(generated512),
                    "static_face_box_y1_y2_x1_x2": [y1, y2, x1, x2],
                    "resize": {"implementation": "OpenCV cv2.resize INTER_LINEAR", "target_wh": [x2 - x1, y2 - y1]},
                    "prediction_channel_order_vs_generated_patch": pixel_diff(resized_linear, observed_patch),
                    "diagnostic_channel_swapped_vs_generated_patch": pixel_diff(resized_linear_swapped, observed_patch),
                })

    after = {
        "parity_summary": {"bytes": SUMMARY_PATH.stat().st_size, "sha256": sha256_file(SUMMARY_PATH)},
        "collector_receipt": {"bytes": (ROI_OUTPUT / "roi_geometry_receipt.json").stat().st_size,
                              "sha256": sha256_file(ROI_OUTPUT / "roi_geometry_receipt.json")},
        "saved_frame_npz": {"bytes": (ROI_OUTPUT / "roi_geometry_selected_frames.npz").stat().st_size,
                            "sha256": sha256_file(ROI_OUTPUT / "roi_geometry_selected_frames.npz")},
        "wav2lip_inference_source": {"bytes": W2L_INFERENCE.stat().st_size,
                                    "sha256": sha256_file(W2L_INFERENCE)},
    }
    for name, pin in source_files.items():
        if after[name]["bytes"] != pin["bytes"] or after[name]["sha256"] != pin["sha256"]:
            raise ProbeError(f"pinned source changed during probe: {name}")
    exact_matches = sum(bool(item["prediction_channel_order_vs_generated_patch"].get("exact")) for item in results)
    report = {
        "schema": "cross_tfg_saved_prediction_full512_paste_probe_v1",
        "status": "PASS_READ_ONLY_SAVED_FRAME_PASTE_COMPARISON",
        "started_unix": started,
        "finished_unix": time.time(),
        "scope": "Ten saved predictions/full512 frames from two historical-fit RMS contexts; exact crop-box pixel comparison only. No model forward, SyncNet scoring, dev/confirmation media, or effects recomputation.",
        "operation": "cv2.resize(saved uint8 96x96 prediction, (static face width,height), INTER_LINEAR) compared with the same face-box pixels in the saved pre-codec full512 generated frame; channel-swapped resize is an explicitly diagnostic comparison.",
        "sources_before": source_files,
        "sources_after": after,
        "point_count": len(results),
        "direct_resize_exact_count": exact_matches,
        "points": results,
    }
    report["payload_sha256"] = canonical_sha(report)
    report_path = output_dir / "prediction_paste_probe_receipt.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(report_path),
                      "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path),
                      "point_count": len(results), "direct_resize_exact_count": exact_matches,
                      "mean_mae": float(np.mean([x["prediction_channel_order_vs_generated_patch"]["mean_abs_u8"] for x in results])),
                      "channel_swapped_mean_mae": float(np.mean([x["diagnostic_channel_swapped_vs_generated_patch"]["mean_abs_u8"] for x in results]))},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
