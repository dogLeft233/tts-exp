#!/usr/bin/env python3
"""Read-only pixel/track geometry probe for the sealed full-512 fit parity run.

No model is loaded or forwarded. The worker selects ten saved frames from two
historical fit clips and exports their saved Wav2Lip predictions, full-frame
paste results, SyncNet crops, frozen proxy crops/boxes, and official tracker
geometry. Every input is SHA/size checked against the pinned parity summary or
the frozen fit-context receipt before it is opened, then rechecked afterward.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUMMARY_PATH = Path(
    "/dev/shm/cross_tfg_audio_head_20261003/"
    "proxy_native_full512_parity_attempt03/output/parity_summary.json"
)
SUMMARY_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
SUMMARY_BYTES = 171086
CONTEXT_NPZ_PATH = Path(
    "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/"
    "context_export_attempt01/context.npz"
)
CONTEXT_NPZ_SHA256 = "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1"
CONTEXT_JSON_PATH = Path(
    "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/"
    "context_export_attempt01/context.json"
)
CONTEXT_JSON_SHA256 = "46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450"
RUN_PIPELINE_PATH = Path("/root/autodl-tmp/wav2lip-syncnet/syncnet_python/run_pipeline.py")
RUN_PIPELINE_SHA256 = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
EXPORT_CONTEXT_SOURCE = Path(__file__).with_name("export_engineering_context_attempt01.py")
EXPORT_CONTEXT_SOURCE_BYTES = 23332
EXPORT_CONTEXT_SOURCE_SHA256 = "e487d137f268d76f9dd68a1d27bdc1bb6d56fd8c9d760416c21596f5eecbe983"
CROP_SCALE = 0.40
FRAME_OFFSETS = (0, 8, 16, 24, 31)
SELECTED_SAMPLE_IDS = (
    "S0052_BAC009S0052W0121",
    "S0055_BAC009S0055W0122",
)


class ProbeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256_bytes(body)


def safe_source(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError(f"required input is missing: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_bytes is not None and size != expected_bytes:
        raise ProbeError(f"input size mismatch for {path}: {size} != {expected_bytes}")
    if digest != expected_sha:
        raise ProbeError(f"input SHA mismatch for {path}: {digest} != {expected_sha}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def array_digest(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def crop_geometry(proc_track: dict[str, Any], local_index: int) -> dict[str, Any]:
    """Reproduce the crop bounds calculation recorded by the old context exporter."""
    bs = float(proc_track["s"][local_index])
    cx = float(proc_track["x"][local_index])
    cy = float(proc_track["y"][local_index])
    if not np.isfinite([bs, cx, cy]).all() or bs <= 0:
        raise ProbeError("official processed track has invalid crop center/scale")
    bsi = int(bs * (1.0 + 2.0 * CROP_SCALE))
    # run_pipeline pads every decoded JPEG by bsi constant pixels (value 110)
    # before applying its crop. Keep both padded-image indices and the
    # corresponding coordinates in the original 512x512 frame.
    y0 = int(cy + bsi - bs)
    y1 = int(cy + bsi + bs * (1.0 + 2.0 * CROP_SCALE))
    x0 = int(cx + bsi - bs * (1.0 + CROP_SCALE))
    x1 = int(cx + bsi + bs * (1.0 + CROP_SCALE))
    if x1 <= x0 or y1 <= y0:
        raise ProbeError("official processed track computes an empty source crop")
    return {
        "center_x": cx, "center_y": cy, "scale": bs, "crop_scale": CROP_SCALE,
        "bsi": bsi, "border_value_bgr": [110, 110, 110],
        "padded_crop_xyxy": [x0, y0, x1, y1],
        "source_crop_xyxy": [x0 - bsi, y0 - bsi, x1 - bsi, y1 - bsi],
        "source_crop_width_height": [x1 - x0, y1 - y0],
        "source_crop_area": (x1 - x0) * (y1 - y0),
    }


def crop_resize(frame: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in geometry["padded_crop_xyxy"])
    bsi = int(geometry["bsi"])
    h, w = frame.shape[:2]
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ProbeError("official crop source must be a uint8 BGR frame")
    padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant",
                    constant_values=110)
    ph, pw = padded.shape[:2]
    if not (0 <= x0 < x1 <= pw and 0 <= y0 < y1 <= ph):
        raise ProbeError(f"selected padded crop box is out of bounds: {geometry['padded_crop_xyxy']} vs {pw}x{ph}")
    crop = np.ascontiguousarray(padded[y0:y1, x0:x1])
    return np.ascontiguousarray(cv2.resize(crop, (224, 224)), dtype=np.uint8)


def snippet_for_cropper(source: str) -> dict[str, Any]:
    """Record crop-related function source as provenance, without executing it."""
    tree = ast.parse(source)
    lines = source.splitlines()
    picked = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            token in node.name.lower() for token in ("crop", "track", "smooth")
        ):
            start = max(0, int(node.lineno) - 1)
            end = int(getattr(node, "end_lineno", node.lineno))
            text = "\n".join(lines[start:end])
            if any(key in text for key in ("proc_track", "crop_scale", "bsi", "bbox")):
                picked.append({"function": node.name, "line_start": start + 1,
                               "line_end": end, "source": text})
    picked.sort(key=lambda row: (row["line_start"], row["function"]))
    return {"path": str(RUN_PIPELINE_PATH), "source_sha256": RUN_PIPELINE_SHA256,
            "relevant_function_count": len(picked), "relevant_functions": picked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-npz", type=Path, default=CONTEXT_NPZ_PATH)
    parser.add_argument("--context-json", type=Path, default=CONTEXT_JSON_PATH)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"refusing to overwrite ROI probe output: {out}")
    started = time.time()
    out.mkdir(parents=True, exist_ok=False)
    sources: dict[str, dict[str, Any]] = {}
    arrays_out: dict[str, np.ndarray] = {}
    point_rows: list[dict[str, Any]] = []

    summary_pin = safe_source(SUMMARY_PATH, SUMMARY_SHA256, SUMMARY_BYTES)
    sources["parity_summary"] = summary_pin
    summary_bytes = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    if summary.get("status") != "COMPLETE" or summary.get("scope") is None:
        raise ProbeError("pinned parity summary is not a completed read-only scoring receipt")
    sources["fit_context_npz"] = safe_source(args.context_npz, CONTEXT_NPZ_SHA256)
    sources["fit_context_json"] = safe_source(args.context_json, CONTEXT_JSON_SHA256)
    sources["run_pipeline_source"] = safe_source(RUN_PIPELINE_PATH, RUN_PIPELINE_SHA256)
    sources["crop_geometry_formula_source"] = safe_source(
        EXPORT_CONTEXT_SOURCE, EXPORT_CONTEXT_SOURCE_SHA256, EXPORT_CONTEXT_SOURCE_BYTES
    )
    sources["crop_geometry_formula_source"]["role"] = "reference_only_crop_bounds_formula"
    pipeline_source = RUN_PIPELINE_PATH.read_text(encoding="utf-8")
    pipeline_crop_source = snippet_for_cropper(pipeline_source)

    context_meta = json.loads(args.context_json.read_text(encoding="utf-8"))
    if context_meta.get("npz_file_sha256") != CONTEXT_NPZ_SHA256:
        raise ProbeError("fit context metadata does not bind the expected context NPZ")
    expected_context_arrays = context_meta.get("array_seals", {})
    required_context_keys = {
        "face_crops_bgr_u8", "syncnet_crops_bgr_u8", "patch_boxes_xyxy_i32",
        "frame_timestamps_seconds_f64",
    }
    with np.load(args.context_npz, allow_pickle=False) as context_npz:
        if not required_context_keys.issubset(context_npz.files):
            raise ProbeError(f"frozen context NPZ is missing keys: {sorted(required_context_keys - set(context_npz.files))}")
        context_arrays_raw = {key: np.array(context_npz[key], copy=True) for key in required_context_keys}
    for key, array in context_arrays_raw.items():
        seal = expected_context_arrays.get(key, {})
        if list(array.shape) != seal.get("shape") or array_digest(array) != seal.get("sha256"):
            raise ProbeError(f"fit context array seal mismatch: {key}")

    # The exporter stores all source-clock frames, not only the 32-frame
    # training window. Bind the count and exact 25-fps timestamp grid from the
    # sealed metadata before mapping selected parity frames into these arrays.
    context_frame_count = int(context_meta.get("frame_count", -1))
    context_track = context_meta.get("syncnet_track", {})
    context_track_start = int(context_track.get("track_start_frame", -1))
    context_fps_num = int(context_meta.get("fps_num", -1))
    context_fps_den = int(context_meta.get("fps_den", -1))
    if context_frame_count <= 0 or context_fps_num != 25 or context_fps_den != 1:
        raise ProbeError("frozen fit context has invalid frame count or source clock")
    context_timestamps = context_arrays_raw["frame_timestamps_seconds_f64"]
    if context_timestamps.shape != (context_frame_count,) or not np.isfinite(context_timestamps).all():
        raise ProbeError("frozen fit context timestamp array does not match its declared frame count")
    expected_timestamps = (context_track_start + np.arange(context_frame_count, dtype=np.float64)) / 25.0
    if not np.array_equal(context_timestamps, expected_timestamps):
        raise ProbeError("frozen fit context timestamps are not the declared contiguous 25-fps source clock")
    if context_arrays_raw["face_crops_bgr_u8"].shape != (context_frame_count, 3, 96, 96):
        raise ProbeError("frozen fit face-crop array does not match the declared full-context layout")

    # Preserve and validate the exact sealed storage layout above. Normalize
    # only the two explicit channel layouts the context schema can declare;
    # no reshape or inferred channel reorder is permitted.
    context_arrays = {key: np.array(value, copy=True) for key, value in context_arrays_raw.items()}
    stored_sync = context_arrays_raw["syncnet_crops_bgr_u8"]
    stored_sync_shape = tuple(int(value) for value in expected_context_arrays["syncnet_crops_bgr_u8"]["shape"])
    if stored_sync_shape == (context_frame_count, 3, 224, 224):
        sync_layout = "TCHW"
        normalized_sync = stored_sync
    elif stored_sync_shape == (context_frame_count, 224, 224, 3):
        sync_layout = "THWC"
        normalized_sync = stored_sync.transpose(0, 3, 1, 2)
    else:
        raise ProbeError(f"declared frozen context SyncNet image layout is unsupported: {stored_sync_shape}")
    context_arrays["syncnet_crops_bgr_u8"] = np.ascontiguousarray(normalized_sync)
    if context_arrays["syncnet_crops_bgr_u8"].shape != (context_frame_count, 3, 224, 224):
        raise ProbeError("frozen fit SyncNet context crop could not be normalized to TCHW")
    if context_arrays["patch_boxes_xyxy_i32"].shape != (context_frame_count, 4):
        raise ProbeError("frozen fit patch-box geometry shape does not match the declared frame count")

    rows_by_sample: dict[str, dict[str, Any]] = {}
    for row in summary.get("rows", []):
        if row.get("sample_id") in SELECTED_SAMPLE_IDS and row.get("method_id") == "rms_fixed":
            if row.get("split") != "fit" or row.get("historical_exposed") is not True:
                raise ProbeError("selected row is not explicitly historical-fit-only")
            if row["sample_id"] in rows_by_sample:
                raise ProbeError("parity summary contains duplicate selected sample rows")
            rows_by_sample[row["sample_id"]] = row
    if set(rows_by_sample) != set(SELECTED_SAMPLE_IDS):
        raise ProbeError("pinned parity summary does not contain both selected fit rows")

    for sample_id in SELECTED_SAMPLE_IDS:
        row = rows_by_sample[sample_id]
        artifact = row["artifacts"]
        details = row["native_render_details"]
        video = details["video"]
        track_meta = details["tracking"]
        q0 = int(row["center_query_start_source_frame"])
        offsets = list(FRAME_OFFSETS)
        source_paths = {
            "parity_arrays": (Path(artifact["parity_arrays_path"]), artifact["parity_arrays_sha256"], int(artifact["parity_arrays_bytes"])),
            "native_video": (Path(video["path"]), video["file_sha256"], int(video["bytes"])),
            "selected_crop_video": (Path(video["selected_track_crop_path"]), video["selected_track_crop_sha256"], int(video["selected_track_crop_bytes"])),
            "tracks_pickle": (Path(track_meta["tracks_pickle_path"]), track_meta["tracks_pickle_sha256"], int(track_meta["tracks_pickle_bytes"])),
        }
        pinned = {}
        for name, (path, digest, size) in source_paths.items():
            pinned[name] = safe_source(path, str(digest), size)
        sources[f"{sample_id}:artifacts"] = pinned

        with np.load(source_paths["parity_arrays"][0], allow_pickle=False) as data:
            keys = (
                "proxy_center_frames_bgr_u8", "official_center_frames_bgr_u8",
                "official_full512_center_generated_u8", "official_full512_center_decoded_u8",
                "official_wav2lip_prediction_float32",
            )
            if not set(keys).issubset(data.files):
                raise ProbeError(f"parity NPZ missing required geometry arrays for {sample_id}")
            saved = {key: np.array(data[key], copy=True) for key in keys}
        expected_shapes = {
            "proxy_center_frames_bgr_u8": (32, 224, 224, 3),
            "official_center_frames_bgr_u8": (32, 224, 224, 3),
            "official_full512_center_generated_u8": (32, 512, 512, 3),
            "official_full512_center_decoded_u8": (32, 512, 512, 3),
            "official_wav2lip_prediction_float32": (32, 3, 96, 96),
        }
        for key, shape in expected_shapes.items():
            if saved[key].shape != shape:
                raise ProbeError(f"saved parity geometry shape mismatch for {sample_id}/{key}: {saved[key].shape}")
        tracks_before = pinned["tracks_pickle"]
        with source_paths["tracks_pickle"][0].open("rb") as stream:
            tracks = pickle.load(stream)
        if not isinstance(tracks, list) or not tracks:
            raise ProbeError(f"selected track pickle is empty for {sample_id}")
        selected_index = int(video["selected_track_index"])
        if selected_index != 0 or selected_index >= len(tracks):
            raise ProbeError(f"unexpected selected track index for {sample_id}: {selected_index}")
        entry = tracks[selected_index]
        if not isinstance(entry, dict) or not isinstance(entry.get("track"), dict) or not isinstance(entry.get("proc_track"), dict):
            raise ProbeError(f"selected pickle entry lacks track/proc_track maps for {sample_id}")
        raw_track = entry["track"]
        proc_track = entry["proc_track"]
        source_frames = np.asarray(raw_track.get("frame"), dtype=np.int64).reshape(-1)
        raw_boxes = np.asarray(raw_track.get("bbox"), dtype=np.float32)
        if raw_boxes.shape != (source_frames.size, 4):
            raise ProbeError(f"raw track bbox shape mismatch for {sample_id}")
        if int(source_frames[0]) != int(video["selected_track_start_source_frame"]):
            raise ProbeError(f"selected track start differs from summary for {sample_id}")
        if source_frames.size != int(video["selected_track_frame_count"]):
            raise ProbeError(f"selected track length differs from summary for {sample_id}")
        if array_digest(source_frames.astype("<i8", copy=False)) != video["selected_track_source_frames_sha256"]:
            raise ProbeError(f"selected track source-frame vector hash mismatch for {sample_id}")
        if not np.array_equal(np.diff(source_frames), np.ones(source_frames.size - 1, dtype=np.int64)):
            raise ProbeError(f"selected track source frames are noncontiguous for {sample_id}")

        cap = cv2.VideoCapture(str(source_paths["selected_crop_video"][0]))
        decoded_crop = []
        try:
            if not cap.isOpened():
                raise ProbeError(f"cannot decode selected crop AVI for {sample_id}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded_crop.append(np.ascontiguousarray(frame, dtype=np.uint8))
        finally:
            cap.release()
        if len(decoded_crop) != source_frames.size or any(f.shape != (224, 224, 3) for f in decoded_crop):
            raise ProbeError(f"decoded official crop frames disagree with track vector for {sample_id}")

        source_face_box = [int(v) for v in details["face_crop"]["official_face_box_y1_y2_x1_x2"]]
        y1, y2, x1, x2 = source_face_box
        row_key = re.sub(r"[^A-Za-z0-9]+", "_", sample_id)
        for offset in offsets:
            absolute_frame = q0 + offset
            context_index = absolute_frame - context_track_start
            if not 0 <= context_index < context_frame_count:
                raise ProbeError(f"selected source frame {absolute_frame} is outside the sealed full-context arrays")
            if context_arrays["frame_timestamps_seconds_f64"][context_index] != absolute_frame / 25.0:
                raise ProbeError(f"context timestamp does not bind selected source frame {absolute_frame}")
            matches = np.flatnonzero(source_frames == absolute_frame)
            if matches.size != 1:
                raise ProbeError(f"selected source frame {absolute_frame} is absent/duplicated in track {sample_id}")
            crop_local_index = int(matches[0])
            if crop_local_index >= len(decoded_crop):
                raise ProbeError("track source-frame index does not address a decoded crop frame")
            geometry = crop_geometry(proc_track, crop_local_index)
            bbox = raw_boxes[crop_local_index].astype(float).tolist()
            generated_full = saved["official_full512_center_generated_u8"][offset]
            decoded_full = saved["official_full512_center_decoded_u8"][offset]
            official_crop = decoded_crop[crop_local_index]
            official_crop_npz = saved["official_center_frames_bgr_u8"][offset]
            proxy_crop = saved["proxy_center_frames_bgr_u8"][offset]
            context_crop = np.ascontiguousarray(context_arrays["syncnet_crops_bgr_u8"][context_index].transpose(1, 2, 0))
            proxy_box = context_arrays["patch_boxes_xyxy_i32"][context_index].astype(int).tolist()
            crop_x0, crop_y0, crop_x1, crop_y1 = geometry["source_crop_xyxy"]
            crop_w = crop_x1 - crop_x0
            crop_h = crop_y1 - crop_y0
            native_patch_box = [
                max(0, min(224, int(np.floor((x1 - crop_x0) * 224.0 / crop_w)))),
                max(0, min(224, int(np.floor((y1 - crop_y0) * 224.0 / crop_h)))),
                max(0, min(224, int(np.ceil((x2 - crop_x0) * 224.0 / crop_w)))),
                max(0, min(224, int(np.ceil((y2 - crop_y0) * 224.0 / crop_h)))),
            ]
            predicted = np.ascontiguousarray(
                saved["official_wav2lip_prediction_float32"][offset].transpose(1, 2, 0) * 255.0,
                dtype=np.float32,
            ).astype(np.uint8)
            # Reconstruct full-frame postprocess from the exact saved Wav2Lip output.
            direct_patch_full = generated_full[y1:y2, x1:x2]
            if direct_patch_full.size == 0:
                raise ProbeError("Wav2Lip full-frame patch is empty")
            reconstructed_crop = crop_resize(decoded_full, geometry)
            reconstructed_gen_crop = crop_resize(generated_full, geometry)
            proxy_mouth = proxy_crop[proxy_box[1]:proxy_box[3], proxy_box[0]:proxy_box[2]]
            native_mouth = official_crop[native_patch_box[1]:native_patch_box[3], native_patch_box[0]:native_patch_box[2]]
            if proxy_mouth.size == 0 or native_mouth.size == 0:
                raise ProbeError("context patch box has an empty mouth ROI")

            prefix = f"{row_key}_offset{offset:02d}_src{absolute_frame:03d}"
            arrays_out[f"{prefix}_pred96_rgb_or_bgr"] = predicted
            arrays_out[f"{prefix}_proxy_224"] = proxy_crop
            arrays_out[f"{prefix}_context_224"] = context_crop
            arrays_out[f"{prefix}_native_official_crop_224"] = official_crop
            arrays_out[f"{prefix}_native_crop_npz_224"] = official_crop_npz
            arrays_out[f"{prefix}_native_reconstructed_from_decoded512_224"] = reconstructed_crop
            arrays_out[f"{prefix}_native_reconstructed_from_generated512_224"] = reconstructed_gen_crop
            arrays_out[f"{prefix}_native_full512_generated"] = generated_full
            arrays_out[f"{prefix}_native_full512_decoded"] = decoded_full
            arrays_out[f"{prefix}_patch_context_crop_u8"] = np.ascontiguousarray(
                context_crop[proxy_box[1]:proxy_box[3], proxy_box[0]:proxy_box[2]])
            arrays_out[f"{prefix}_patch_proxy_u8"] = np.ascontiguousarray(proxy_mouth)
            arrays_out[f"{prefix}_patch_native_u8"] = np.ascontiguousarray(native_mouth)

            def diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
                if left.shape != right.shape:
                    return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
                delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
                return {
                    "same_shape": True,
                    "exact": bool(np.array_equal(left, right)),
                    "max_abs_u8": int(delta.max(initial=0)),
                    "mean_abs_u8": float(delta.mean()),
                    "changed_pixel_fraction": float(np.count_nonzero(delta) / delta.size),
                }

            point_rows.append({
                "sample_id": sample_id,
                "row_id": row["row_id"],
                "method_id": row["method_id"],
                "split": row["split"],
                "historical_exposed": row["historical_exposed"],
                "query_start_source_frame": q0,
                "offset_from_q0": offset,
                "source_clock_frame": absolute_frame,
                "crop_local_frame_index": crop_local_index,
                "context_frame_index": context_index,
                "source_timestamp_seconds": absolute_frame / 25.0,
                "track_source_frame": int(source_frames[crop_local_index]),
                "raw_track_bbox_xyxy": bbox,
                "processed_track_crop_geometry": geometry,
                "wav2lip_static_face_box_y1_y2_x1_x2": source_face_box,
                "frozen_proxy_patch_box_xyxy_in_224": proxy_box,
                "projected_native_patch_box_xyxy_in_224": native_patch_box,
                "proxy_vs_context_224": diff_stats(proxy_crop, context_crop),
                "official_crop_npz_vs_decoded_crop_avi": diff_stats(official_crop_npz, official_crop),
                "reconstructed_decoded512_crop_vs_official_crop": diff_stats(reconstructed_crop, official_crop),
                "reconstructed_generated512_crop_vs_official_crop": diff_stats(reconstructed_gen_crop, official_crop),
                "full512_generated_vs_decoded": diff_stats(generated_full, decoded_full),
                "native_vs_proxy_crop_224": diff_stats(official_crop, proxy_crop),
                "native_vs_proxy_projected_patch_roi": diff_stats(native_mouth, proxy_mouth),
                "arrays": {
                    name: {"shape": list(arrays_out[f"{prefix}_{name}"].shape),
                           "dtype": str(arrays_out[f"{prefix}_{name}"].dtype),
                           "sha256": array_digest(arrays_out[f"{prefix}_{name}"])}
                    for name in (
                        "pred96_rgb_or_bgr", "proxy_224", "context_224",
                        "native_official_crop_224", "native_crop_npz_224",
                        "native_reconstructed_from_decoded512_224",
                        "native_reconstructed_from_generated512_224",
                        "native_full512_generated", "native_full512_decoded",
                        "patch_context_crop_u8", "patch_proxy_u8", "patch_native_u8",
                    )
                },
            })

    # Recheck every input that was read before writing a PASS receipt.
    after = {}
    for name, entries in sources.items():
        if name == "parity_summary":
            path = SUMMARY_PATH
        elif name == "fit_context_npz":
            path = args.context_npz
        elif name == "fit_context_json":
            path = args.context_json
        elif name == "run_pipeline_source":
            path = RUN_PIPELINE_PATH
        elif name == "crop_geometry_formula_source":
            path = EXPORT_CONTEXT_SOURCE
        else:
            continue
        after[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for sample_id in SELECTED_SAMPLE_IDS:
        row = rows_by_sample[sample_id]
        details = row["native_render_details"]
        before = sources[f"{sample_id}:artifacts"]
        for name, rec in before.items():
            after[f"{sample_id}:{name}"] = {"bytes": Path(rec["path"]).stat().st_size,
                                             "sha256": sha256_file(Path(rec["path"]))}
    for name, pin in sources.items():
        if name in after and (after[name]["bytes"] != pin["bytes"] or after[name]["sha256"] != pin["sha256"]):
            raise ProbeError(f"source changed while extracting read-only geometry: {name}")
    for sample_id in SELECTED_SAMPLE_IDS:
        for name, pin in sources[f"{sample_id}:artifacts"].items():
            observed = after[f"{sample_id}:{name}"]
            if observed["bytes"] != pin["bytes"] or observed["sha256"] != pin["sha256"]:
                raise ProbeError(f"row source changed during read-only extraction: {sample_id}/{name}")

    npz_path = out / "roi_geometry_selected_frames.npz"
    np.savez_compressed(npz_path, **arrays_out)
    receipt = {
        "schema": "cross_tfg_full512_roi_geometry_probe_v1",
        "status": "PASS_READ_ONLY_SAVED_FRAMES",
        "started_unix": started,
        "finished_unix": time.time(),
        "scope": "two historical fit rows; five saved source-clock frames per row; no model import, inference, SyncNet forward, or effects recomputation",
        "selected_sample_ids": list(SELECTED_SAMPLE_IDS),
        "frame_offsets_from_q0": list(FRAME_OFFSETS),
        "crop_scale": CROP_SCALE,
        "context_npz_array_hashes": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_digest(value)}
            for key, value in context_arrays_raw.items()
        },
        "context_syncnet_image_storage_layout": sync_layout,
        "normalized_context_array_hashes": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_digest(value)}
            for key, value in context_arrays.items()
        },
        "run_pipeline_crop_source": pipeline_crop_source,
        "sources_before": sources,
        "sources_after": after,
        "points": point_rows,
        "selected_arrays_npz": {"path": str(npz_path), "bytes": npz_path.stat().st_size,
                                 "sha256": sha256_file(npz_path), "member_count": len(arrays_out)},
    }
    receipt["payload_sha256"] = canonical_sha(receipt)
    json_path = out / "roi_geometry_receipt.json"
    json_write(json_path, receipt)
    print(json.dumps({
        "status": receipt["status"], "receipt_path": str(json_path),
        "receipt_bytes": json_path.stat().st_size, "receipt_sha256": sha256_file(json_path),
        "payload_sha256": receipt["payload_sha256"],
        "npz_path": str(npz_path), "npz_bytes": npz_path.stat().st_size,
        "npz_sha256": sha256_file(npz_path), "point_count": len(point_rows),
    }, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
