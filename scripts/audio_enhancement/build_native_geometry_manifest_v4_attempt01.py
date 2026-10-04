"""Build the frozen eight-row native-crop geometry manifest for v4 fit.

This is a metadata-only join over the already sealed full512 parity summary
and JPEG-to-source frame map.  It does not open audio/video/model outputs or
select another track.  The runtime rechecks each tracks.pkl SHA before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
FRAME_MAP_SHA256 = "926e08f84f2624fd8800974f2778c0ed875187aa6de065354c0db4701b0e8447"
SOURCE_PORTRAIT_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
EXPECTED_SAMPLE_IDS = (
    "S0052_BAC009S0052W0121", "S0052_BAC009S0052W0122",
    "S0053_BAC009S0053W0121", "S0053_BAC009S0053W0122",
    "S0054_BAC009S0054W0121", "S0054_BAC009S0054W0122",
    "S0055_BAC009S0055W0121", "S0055_BAC009S0055W0122",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_hex(value: Any, label: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} is not a 64-character SHA-256")
    return value


def build(summary_path: Path, frame_map_path: Path, output_path: Path) -> dict[str, Any]:
    summary_path = summary_path.resolve(strict=True)
    frame_map_path = frame_map_path.resolve(strict=True)
    summary_sha = sha256_file(summary_path)
    frame_map_sha = sha256_file(frame_map_path)
    if summary_sha != SUMMARY_SHA256:
        raise ValueError(f"full512 parity summary pin mismatch: {summary_sha}")
    if frame_map_sha != FRAME_MAP_SHA256:
        raise ValueError(f"minimum-MAE source-frame map pin mismatch: {frame_map_sha}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "cross_tfg_fit_proxy_native_full512_parity_summary_v1":
        raise ValueError("unexpected full512 parity summary schema")
    if summary.get("status") != "COMPLETE_READ_ONLY_NATIVE_FULL512_PARITY":
        if not str(summary.get("status", "")).startswith("COMPLETE"):
            raise ValueError("full512 parity summary is not complete")
    if frame_map.get("schema") != "cross_tfg_pipeline_jpeg_source_frame_map_v1":
        raise ValueError("unexpected JPEG source-map schema")
    if frame_map.get("status") != "COMPLETE_READ_ONLY_JPEG_SOURCE_FRAME_MAPPING":
        raise ValueError("JPEG source-map is not complete")
    if frame_map.get("scope") != "eight predeclared historical fit clips only; no inference/scoring/audio reads; full frame source mapping with exact q0 windows; no source edits":
        raise ValueError("JPEG source-map scope differs from the frozen fit-only audit")

    rows_by_sample = {str(row["sample_id"]): row for row in summary.get("rows", [])}
    maps_by_sample = {str(row["sample_id"]): row for row in frame_map.get("rows", [])}
    if set(rows_by_sample) != set(EXPECTED_SAMPLE_IDS) or set(maps_by_sample) != set(EXPECTED_SAMPLE_IDS):
        raise ValueError("native geometry inputs do not contain the exact eight frozen fit utterances")
    if len(rows_by_sample) != 8 or len(maps_by_sample) != 8:
        raise ValueError("native geometry inputs contain duplicate sample IDs")

    first_crop = rows_by_sample[EXPECTED_SAMPLE_IDS[0]]["native_render_details"]["face_crop"]
    source_portrait = {
        "path": str(first_crop["source_portrait_path"]),
        "sha256": require_hex(first_crop["source_portrait_sha256"], "source portrait"),
        "shape_hwc": list(first_crop["source_portrait_shape_hwc"]),
        "face_box_y1_y2_x1_x2": [int(value) for value in first_crop["official_face_box_y1_y2_x1_x2"]],
        "face96_matches_frozen_fit_context_exact": bool(first_crop["face96_matches_frozen_fit_context_exact"]),
    }
    if source_portrait["sha256"] != SOURCE_PORTRAIT_SHA256 or source_portrait["shape_hwc"] != [512, 512, 3]:
        raise ValueError("source portrait pin/shape differs from the frozen static Wav2Lip portrait")
    if not source_portrait["face96_matches_frozen_fit_context_exact"]:
        raise ValueError("native portrait face96 is not bit-exact with the frozen fit context")

    output_rows = []
    for sample_id in EXPECTED_SAMPLE_IDS:
        row = rows_by_sample[sample_id]
        mapping = maps_by_sample[sample_id]
        if row.get("split") != "fit" or row.get("historical_exposed") is not True or row.get("method_id") != "rms_fixed":
            raise ValueError(f"fit-source row is not the frozen historical RMS row: {sample_id}")
        details = row["native_render_details"]
        face = details["face_crop"]
        tracking = details["tracking"]
        video = details["video"]
        if str(face["source_portrait_sha256"]).lower() != source_portrait["sha256"]:
            raise ValueError(f"static portrait differs across fit rows: {sample_id}")
        if [int(value) for value in face["official_face_box_y1_y2_x1_x2"]] != source_portrait["face_box_y1_y2_x1_x2"]:
            raise ValueError(f"static face box differs across fit rows: {sample_id}")
        if int(details["mel"]["all_official_mel_chunk_count"]) <= 0:
            raise ValueError(f"fit row has no official full-audio mel chunks: {sample_id}")

        q0 = int(video["center_query_start_source_clock_frame"])
        track_local_start = int(video["center_query_start_local"])
        track_source_start = int(video["selected_track_start_source_frame"])
        selected_track_index = int(video["selected_track_index"])
        if not (0 <= selected_track_index < int(tracking["track_count"])):
            raise ValueError(f"selected track index is outside the frozen track inventory: {sample_id}")
        if track_source_start != 0 or track_local_start != q0:
            raise ValueError(f"fit track is not indexed from source-clock frame zero: {sample_id}")
        if int(mapping["query_anchor"]["q0_source_clock_frame"]) != q0:
            raise ValueError(f"JPEG-map q0 differs from the native scored row: {sample_id}")
        if int(mapping["query_anchor"]["q0_selected_track_local_frame"]) != track_local_start:
            raise ValueError(f"JPEG-map tracker-local q0 differs from the native scored row: {sample_id}")

        composed = mapping.get("composed_jpeg_to_full512_source")
        if not isinstance(composed, list):
            raise ValueError(f"full-frame JPEG mapping is absent: {sample_id}")
        by_jpeg = {int(item["jpeg_index_zero_based"]): item for item in composed}
        if len(by_jpeg) != len(composed):
            raise ValueError(f"JPEG source map has duplicate output indices: {sample_id}")
        output_frames = list(range(q0, q0 + 32))
        if output_frames[-1] >= int(video["selected_track_frame_count"]):
            raise ValueError(f"selected tracker does not cover the full native 32-frame window: {sample_id}")
        prediction_frames = []
        mapping_entries = []
        mel_count = int(details["mel"]["all_official_mel_chunk_count"])
        for output_frame in output_frames:
            item = by_jpeg.get(output_frame)
            if item is None:
                raise ValueError(f"JPEG source map is missing native output frame {output_frame}: {sample_id}")
            prediction_frame = int(item["mapped_full512_source_frame_index_zero_based"])
            if prediction_frame < 0 or prediction_frame >= mel_count:
                raise ValueError(f"mapped full512 mel index is outside the full-audio mel clock: {sample_id}")
            prediction_frames.append(prediction_frame)
            mapping_entries.append({
                "jpeg_index_zero_based": output_frame,
                "target_source_clock_frame": output_frame,
                "mapped_full512_prediction_mel_index": prediction_frame,
                "best_candidate_mae_0_255": float(item["best_candidate_mae_0_255"]),
                "second_candidate_mae_0_255": float(item["second_candidate_mae_0_255"]),
                "mapping_ambiguity_delta_mae_0_255": float(item["second_candidate_mae_0_255"]) - float(item["best_candidate_mae_0_255"]),
            })
        if len(prediction_frames) != 32:
            raise ValueError(f"fit row does not have 32 measured prediction indices: {sample_id}")

        tracks_path = str(tracking["tracks_pickle_path"])
        tracks_sha = require_hex(tracking["tracks_pickle_sha256"], f"tracks pickle {sample_id}")
        tracks_bytes = int(tracking["tracks_pickle_bytes"])
        if tracks_bytes <= 0:
            raise ValueError(f"tracks pickle size is invalid: {sample_id}")
        output_rows.append({
            "row_id": str(row["row_id"]),
            "sample_id": sample_id,
            "speaker_id": str(row["speaker_id"]),
            "split": "fit",
            "historical_exposed": True,
            "method_id": "rms_fixed",
            "selected_track_index": selected_track_index,
            "selected_track_id": str(video["selected_track_id"]),
            "tracks_pickle": {"path": tracks_path, "bytes": tracks_bytes, "sha256": tracks_sha},
            "track_frame_count": int(video["selected_track_frame_count"]),
            "track_source_start_frame": track_source_start,
            "face_box_y1_y2_x1_x2": [int(value) for value in face["official_face_box_y1_y2_x1_x2"]],
            "query_start_frame": q0,
            "output_source_clock_frames": output_frames,
            "track_local_frame_indices": list(range(track_local_start, track_local_start + 32)),
            "prediction_source_clock_frames": prediction_frames,
            "prediction_map_evidence": {
                "schema": str(frame_map["schema"]),
                "receipt_path": str(frame_map_path),
                "receipt_sha256": frame_map_sha,
                "row_sample_id": sample_id,
                "map_kind": "minimum_mae_monotone_correspondence; association evidence, not an exact frame identity claim",
                "syncnet_query_policy": "output/query source-clock j is unchanged; only Wav2Lip conditioning mel index follows this measured per-row mapping",
                "source_composition": "full512_source_to_pipeline_avi_to_jpeg",
                "entries": mapping_entries,
            },
        })

    payload: dict[str, Any] = {
        "schema": "cross_tfg_native_full512_fit_geometry_manifest_v1",
        "created_by": "build_native_geometry_manifest_v4_attempt01.py",
        "source_clock_fps_num": 25,
        "source_clock_fps_den": 1,
        "audio_clock_policy": "unchanged_original_natural_pcm_mfcc_queries",
        "scope": "eight historical fit-only RMS full512 contexts; metadata join only; no audio/video/model outputs were opened or changed",
        "source_pins": {
            "parity_summary": {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": summary_sha},
            "jpeg_source_frame_map": {"path": str(frame_map_path), "bytes": frame_map_path.stat().st_size, "sha256": frame_map_sha},
        },
        "source_portrait": source_portrait,
        "rows": output_rows,
    }
    payload["payload_sha256"] = canonical_sha(payload)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output_path.write_bytes(raw)
    return {
        "status": "NATIVE_GEOMETRY_MANIFEST_BUILT",
        "path": str(output_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": payload["payload_sha256"],
        "rows": len(output_rows),
        "sample_ids": [row["sample_id"] for row in output_rows],
        "prediction_index_ranges": [
            {"sample_id": row["sample_id"], "q0": row["query_start_frame"], "input_min": min(row["prediction_source_clock_frames"]),
             "input_max": max(row["prediction_source_clock_frames"]), "input_indices": row["prediction_source_clock_frames"]}
            for row in output_rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--frame-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.summary, args.frame_map, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
