from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .analysis import analyze_full, analyze_stage_a
from .common import (
    ProtocolError,
    bytes_sha256,
    file_sha256,
    read_pcm16,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import load_inputs, load_protocol
from .scoring import _audio_path, load_score_rows


def _video_rows(paths: config.RunPaths) -> dict[tuple[str, str], Mapping[str, Any]]:
    manifest = verify_self_hashed_json(paths.video_manifest)
    rows = {}
    for row in manifest.get("rows", []):
        key = (str(row["sample_id"]), str(row["video_arm"]))
        if key in rows:
            raise ProtocolError(f"duplicate video row: {key}")
        output = Path(str(row["output"]))
        if not output.is_file() or file_sha256(output) != str(row["output_sha256"]):
            raise ProtocolError(f"video output hash changed: {key}")
        if row.get("input_mode") != "one_png_only" or row.get("source_frame_index") != 0:
            raise ProtocolError(f"video input mode is not static: {key}")
        if row.get("source_frame_indices") != [0] * int(row.get("frame_count", 0)):
            raise ProtocolError(f"dynamic source frame mixed into static video: {key}")
        rows[key] = row
    return rows


def _validate_static_inputs(inputs: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for item in inputs.get("records", []):
        sid = str(item["sample_id"])
        static = item.get("static_reference")
        try:
            if static.get("source_frame_index") != 0 or static.get("input_mode") != "one_png_only":
                raise ProtocolError("source frame/input mode contract")
            image = Path(str(static["image"]["path"]))
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ProtocolError("reference PNG missing")
            rgb_hash = bytes_sha256(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).tobytes())
            if rgb_hash != str(static["image"]["rgb_pixel_sha256"]):
                raise ProtocolError("reference RGB hash changed")
            if file_sha256(image) != str(static["image"]["container_sha256"]):
                raise ProtocolError("reference PNG container hash changed")
            crop = Path(str(static["score_crop"]["path"]))
            if not crop.is_file() or file_sha256(crop) != str(static["score_crop"]["container_sha256"]):
                raise ProtocolError("score crop changed")
            if float(static["detection"]["threshold"]) != config.FACE_DET_THRESHOLD:
                raise ProtocolError("face threshold changed")
            if len(static["generation_box_xyxy"]) != 4 or len(static["score_box"]["box"]) != 4:
                raise ProtocolError("box geometry is incomplete")
        except (KeyError, TypeError, ValueError, OSError, ProtocolError) as exc:
            failures.append(f"{sid}: {exc}")
    if len(inputs.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        failures.append("record_count != 22")
    return failures


def _validate_scores(
    paths: config.RunPaths,
    inputs: Mapping[str, Any],
    expected: tuple[tuple[str, str], ...],
    audio_manifest: Mapping[str, Any],
    video_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    scores = load_score_rows(paths)
    required = {(str(item["sample_id"]), video, audio) for item in inputs["records"] for video, audio in expected}
    actual = set(scores)
    if actual != required:
        raise ProtocolError(f"score cell set mismatch; missing={sorted(required - actual)[:3]}, extra={sorted(actual - required)[:3]}")
    audio_by_id = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
    for key, row in scores.items():
        sid, video_arm, audio_arm = key
        sample = next((item for item in inputs["records"] if str(item["sample_id"]) == sid), None)
        if sample is None or sid not in audio_by_id:
            raise ProtocolError(f"score cell is not keyed to a frozen input: {key}")
        video = video_rows.get((sid, video_arm))
        if video is None:
            raise ProtocolError(f"score cell video binding is missing: {key}")
        expected_video = Path(str(video["output"])).resolve()
        actual_video = Path(str(row.get("video_path", ""))).resolve()
        if actual_video != expected_video or file_sha256(actual_video) != str(row.get("video_sha256")):
            raise ProtocolError(f"score cell video binding/hash changed: {key}")
        audio_row = audio_by_id[sid]
        expected_audio = _audio_path(audio_row, audio_arm).resolve()
        actual_audio = Path(str(row.get("audio_path", ""))).resolve()
        if actual_audio != expected_audio:
            raise ProtocolError(f"score cell audio binding changed: {key}")
        _audio_values, audio_meta = read_pcm16(actual_audio)
        if file_sha256(actual_audio) != str(row.get("audio_sha256")) or audio_meta["pcm_sha256"] != str(row.get("audio_pcm_sha256")):
            raise ProtocolError(f"score cell audio hash changed: {key}")
        expected_static = sample["static_reference"]
        if row.get("fixed_score_box") != expected_static.get("score_box"):
            raise ProtocolError(f"score cell fixed crop binding changed: {key}")
        if row.get("video_arm") != video_arm or row.get("audio_arm") != audio_arm or row.get("protocol_id") != config.PROTOCOL_ID:
            raise ProtocolError(f"score cell identity changed: {key}")
        matrix = np.asarray(np.load(Path(str(row["matrix_path"])), allow_pickle=False), dtype=np.float64)
        matrix_path = Path(str(row["matrix_path"]))
        if not matrix_path.is_file() or file_sha256(matrix_path) != str(row.get("matrix_sha256")):
            raise ProtocolError(f"score matrix hash changed: {key}")
        if matrix.shape != tuple(int(value) for value in row["matrix_shape"]):
            raise ProtocolError(f"matrix shape binding changed: {key}")
        if matrix.ndim != 2 or matrix.shape[1] != 31:
            raise ProtocolError(f"matrix shape invalid: {key}")
        if row.get("device") != "cuda" or row.get("model_sha256") != config.SYNCNET_MODEL_SHA256:
            raise ProtocolError(f"score runtime binding invalid: {key}")
    return scores


def validate_stage_a(paths: config.RunPaths) -> dict[str, Any]:
    protocol = load_protocol(paths)
    inputs = load_inputs(paths)
    failures = _validate_static_inputs(inputs)
    videos = _video_rows(paths)
    audio = verify_self_hashed_json(paths.audio_manifest)
    required_videos = {(str(item["sample_id"]), arm) for item in inputs["records"] for arm in config.STAGE_A_VIDEOS}
    if not required_videos.issubset(videos):
        failures.append("stage A video set incomplete")
    scores = _validate_scores(paths, inputs, config.STAGE_A_CELLS, audio, videos)
    stage_a = verify_self_hashed_json(paths.stage_a)
    recomputed = analyze_stage_a(paths, inputs, scores)
    if stage_a != recomputed:
        failures.append("stage_a.json does not match independent gate recomputation")
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "A",
        "status": "PASS" if not failures else "FAIL",
        "independent": True,
        "protocol_sha256": protocol.get("artifact_sha256"),
        "failures": failures,
        "measurement_decision": recomputed.get("measurement_decision"),
    }
    return result


def validate_full(paths: config.RunPaths) -> dict[str, Any]:
    failures: list[str] = []
    try:
        load_protocol(paths)
        inputs = load_inputs(paths)
        failures.extend(_validate_static_inputs(inputs))
        videos = _video_rows(paths)
        audio = verify_self_hashed_json(paths.audio_manifest)
        scores = load_score_rows(paths)
        stage_a = verify_self_hashed_json(paths.stage_a)
        recomputed_stage_a = analyze_stage_a(paths, inputs, scores)
        if stage_a != recomputed_stage_a:
            failures.append("stage A recomputation differs")
        if stage_a.get("measurement_decision") == "PASS":
            required_videos = {(str(item["sample_id"]), arm) for item in inputs["records"] for arm in config.ARMS}
            required_scores = {(str(item["sample_id"]), video, audio) for item in inputs["records"] for video, audio in config.ALL_CELLS}
            if set(videos) != required_videos:
                failures.append(f"full video set mismatch: {len(videos)}/{len(required_videos)}")
            if set(scores) != required_scores:
                failures.append(f"full score set mismatch: {len(scores)}/{len(required_scores)}")
            else:
                _validate_scores(paths, inputs, config.ALL_CELLS, audio, videos)
            analysis = verify_self_hashed_json(paths.analysis)
            recomputed = analyze_full(paths, inputs, scores, recomputed_stage_a)
            if analysis != recomputed:
                failures.append("analysis.json does not match independent recomputation")
        else:
            b_videos = [key for key in videos if key[1] in config.STAGE_B_VIDEOS]
            b_scores = [key for key in scores if key[1] in {video for video, _ in config.STAGE_B_CELLS}]
            if b_videos or b_scores:
                failures.append("stage B ran despite a failed stage A gate")
        validation = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "PASS" if not failures else "FAIL",
            "independent": True,
            "failures": failures,
            "checks": [
                "frozen parent hashes and 22/22 keyed join",
                "static PNG RGB/container hashes and source_frame_index=0",
                "fixed generation/score boxes and no candidate-specific crop",
                "decoded PCM identities and matrix hashes",
                "independent stage-A and final recomputation",
            ],
            "training_authorized": False,
        }
    except (OSError, ProtocolError, KeyError, TypeError, ValueError) as exc:
        validation = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "FAIL", "independent": True, "failures": [f"{type(exc).__name__}: {exc}"], "training_authorized": False}
    write_self_hashed_json(paths.validation, validation)
    return verify_self_hashed_json(paths.validation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("a", "full"), default="full")
    args = parser.parse_args()
    paths = config.RunPaths(args.run_root.resolve())
    result = validate_stage_a(paths) if args.stage == "a" else validate_full(paths)
    print(result)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
