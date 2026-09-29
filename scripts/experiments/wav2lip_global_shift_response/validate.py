from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import extract_pcm_from_media, read_pcm16_wav

from . import config, media
from .common import ExperimentError, canonical_sha256, file_sha256, verify_self_hashed_json, write_self_hashed_json


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _endpoint(matrix: np.ndarray, rows: Sequence[int], anchor: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    indices = np.asarray([int(row) for row in rows], dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= values.shape[0] or not np.isfinite(values).all():
        raise ExperimentError("validator endpoint input is malformed")
    curve = values[indices].mean(axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    result: dict[str, Any] = {"D": minimum, "M": float(np.median(curve)), "C": float(np.median(curve) - minimum), "offset": int(config.VSHIFT - index), "min_index": index, "curve": [float(item) for item in curve], "rows": [int(item) for item in indices]}
    if anchor is not None:
        column = config.VSHIFT - int(anchor)
        if not 0 <= column < config.MATRIX_COLUMNS:
            raise ExperimentError("validator anchor column is invalid")
        result["anchor_offset"] = int(anchor)
        result["anchor_column"] = int(column)
        result["D_anchor"] = float(curve[column])
        result["C_anchor"] = float(np.median(curve) - curve[column])
    return result


def _close_float(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _validate_audio(protocol: Mapping[str, Any]) -> int:
    count = 0
    for record in protocol["records"]:
        arms = record["audio"]["arms"]
        pcm: dict[str, bytes] = {}
        for arm in config.AUDIO_ARMS:
            item = arms[arm]
            path = Path(str(item["path"]))
            decoded, values, params = read_pcm16_wav(path)
            if file_sha256(path) != str(item["container_sha256"]) or decoded.hex() == "" or file_sha256(path) != str(item["container_sha256"]):
                raise ExperimentError(f"audio container binding changed: {record['sample_id']}/{arm}")
            if decoded and __import__("hashlib").sha256(decoded).hexdigest() != str(item["decoded_pcm_sha256"]):
                raise ExperimentError(f"audio PCM binding changed: {record['sample_id']}/{arm}")
            if values.size != int(record["sample_count"]) or params["sample_rate"] != config.SAMPLE_RATE or params["channels"] != 1 or params["sample_width"] != 2:
                raise ExperimentError(f"audio shape/format changed: {record['sample_id']}/{arm}")
            pcm[arm] = decoded
        if pcm[config.AUDIO_N] != pcm[config.AUDIO_N_REPEAT]:
            raise ExperimentError(f"N/N_REPEAT PCM differs: {record['sample_id']}")
        from .audio import assert_shift_contract
        assert_shift_contract(pcm[config.AUDIO_N], pcm[config.AUDIO_DELAY], pcm[config.AUDIO_ADVANCE])
        count += len(config.AUDIO_ARMS)
    return count


def _validate_faces(protocol: Mapping[str, Any]) -> int:
    count = 0
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        source, _ = media.decode_frames(Path(str(record["face_video"]["path"])))
        f = int(record["frame_count"])
        if source.shape[0] < f:
            raise ExperimentError(f"source face frame count changed: {sid}")
        for mode in config.FACE_MODES:
            payload = record["face_modes"][mode]
            decoded, _ = media.decode_frames(Path(str(payload["video"]["path"])))
            if decoded.shape != source[:f].shape:
                raise ExperimentError(f"derived face shape changed: {sid}/{mode}")
            if mode == config.FACE_DYNAMIC:
                if not np.array_equal(decoded, source[:f]):
                    raise ExperimentError(f"DYNAMIC face pixels changed: {sid}")
            elif not np.all(decoded == decoded[:1]):
                raise ExperimentError(f"STATIC face pixels are not frozen: {sid}")
            boxes = payload["boxes"]["boxes"]
            if len(boxes) != f:
                raise ExperimentError(f"face box count changed: {sid}/{mode}")
            if mode == config.FACE_STATIC and any(list(box) != list(boxes[0]) for box in boxes):
                raise ExperimentError(f"STATIC boxes are not frozen: {sid}")
            if str(payload["boxes"]["sha256"]) != file_sha256(Path(str(payload["boxes"]["path"]))):
                raise ExperimentError(f"face box sidecar hash changed: {sid}/{mode}")
            count += 1
    return count


def _validate_videos(protocol: Mapping[str, Any], videos: Mapping[str, Any]) -> tuple[int, dict[tuple[str, str, str], Mapping[str, Any]]]:
    rows = videos.get("rows")
    if videos.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.EXPECTED_VIDEO_COUNT:
        raise ExperimentError("video manifest is incomplete")
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    record_index = {str(row["sample_id"]): row for row in protocol["records"]}
    for row in rows:
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]))
        if key in result or key[0] not in record_index or key[1] not in config.FACE_MODES or key[2] not in config.VIDEO_ARMS:
            raise ExperimentError(f"video identity/duplicate changed: {key}")
        path = Path(str(row["output"]))
        if not path.is_file() or file_sha256(path) != str(row["output_sha256"]):
            raise ExperimentError(f"video artifact hash changed: {key}")
        decoded, evidence = media.decode_frames(path)
        if decoded.shape[0] != int(record_index[key[0]]["frame_count"]) or row.get("device") != "cuda" or row.get("batch_size") != config.GENERATION_BATCH_SIZE:
            raise ExperimentError(f"video runtime/frame contract changed: {key}")
        generation = _load(Path(str(row["generation_result"])))
        if file_sha256(Path(str(row["generation_result"]))) != str(row["generation_result_sha256"]) or generation.get("status") != "complete":
            raise ExperimentError(f"generation result binding changed: {key}")
        result[key] = row
    if len(result) != config.EXPECTED_VIDEO_COUNT:
        raise ExperimentError("video denominator shrank")
    return len(result), result


def _validate_scores(protocol: Mapping[str, Any], scores: Mapping[str, Any], videos: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> tuple[int, dict[tuple[str, str, str, str], Mapping[str, Any]]]:
    rows = scores.get("scores")
    if scores.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.EXPECTED_SCORE_COUNT:
        raise ExperimentError("score manifest is incomplete")
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    record_index = {str(row["sample_id"]): row for row in protocol["records"]}
    expected_cells = {(sid, mode, video, audio) for sid in record_index for mode in config.FACE_MODES for video, audio in config.SCORE_CELLS}
    for row in rows:
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result or key not in expected_cells:
            raise ExperimentError(f"score identity/duplicate changed: {key}")
        media_path = Path(str(row["media"]))
        sidecar_path = Path(str(row["media_sidecar"]))
        if not media_path.is_file() or file_sha256(media_path) != str(row["media_sha256"]) or not sidecar_path.is_file() or file_sha256(sidecar_path) != str(row["media_sidecar_sha256"]):
            raise ExperimentError(f"score media binding changed: {key}")
        sidecar = _load(sidecar_path)
        if sidecar.get("audio_modified") is not False or sidecar.get("video_stream_copy") is not True or "-shortest" in [str(item) for item in sidecar.get("mux_command", [])]:
            raise ExperimentError(f"mux contract changed: {key}")
        audio_item = record_index[key[0]]["audio"]["arms"][key[3]]
        audio_path = Path(str(audio_item["path"]))
        expected_pcm, _values, _params = read_pcm16_wav(audio_path)
        if extract_pcm_from_media(media_path) != expected_pcm or str(row["audio_pcm_sha256"]) != __import__("hashlib").sha256(expected_pcm).hexdigest():
            raise ExperimentError(f"score media PCM changed: {key}")
        worker_path = Path(str(row["worker"]))
        worker = _load(worker_path)
        matrix_path = Path(str(row["matrix"]))
        if file_sha256(worker_path) != str(row["worker_sha256"]) or file_sha256(matrix_path) != str(row["matrix_sha256"]):
            raise ExperimentError(f"worker/matrix hash changed: {key}")
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
        expected_shape = (int(record_index[key[0]]["frame_count"]) - config.WINDOW_FRAMES, config.MATRIX_COLUMNS)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all() or worker.get("new_forward") is not True or worker.get("model_sha256") != config.SYNCNET_MODEL_SHA256:
            raise ExperimentError(f"score matrix/runtime contract changed: {key}")
        result[key] = row
    if set(result) != expected_cells:
        raise ExperimentError(f"score cell set differs: {len(result)} != {len(expected_cells)}")
    return len(result), result


def _validate_endpoints(protocol: Mapping[str, Any], endpoints: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> None:
    rows = endpoints.get("rows")
    if endpoints.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.EXPECTED_SCORE_COUNT:
        raise ExperimentError("endpoint artifact is incomplete")
    record_index = {str(row["sample_id"]): row for row in protocol["records"]}
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in seen or key not in scores:
            raise ExperimentError(f"endpoint key changed: {key}")
        seen.add(key)
        matrix = np.asarray(np.load(Path(str(scores[key]["matrix"])), allow_pickle=False), dtype=np.float32)
        interior = list(range(30, matrix.shape[0] - 30))
        baseline_matrix = np.asarray(np.load(Path(str(scores[(key[0], key[1], config.VIDEO_N, config.AUDIO_N)]["matrix"])), allow_pickle=False), dtype=np.float32)
        full_anchor = _endpoint(baseline_matrix, range(baseline_matrix.shape[0]))["offset"]
        i_anchor = _endpoint(baseline_matrix, interior)["offset"]
        expected_full = _endpoint(matrix, range(matrix.shape[0]), full_anchor)
        expected_i = _endpoint(matrix, interior, i_anchor)
        for support in ("full", "I"):
            actual = row[support]
            expected = expected_full if support == "full" else expected_i
            for field in ("D", "M", "C", "D_anchor", "C_anchor"):
                if not _close_float(actual[field], expected[field], 1e-6):
                    raise ExperimentError(f"endpoint {field} changed: {key}/{support}")
            if int(actual["offset"]) != int(expected["offset"]):
                raise ExperimentError(f"endpoint offset changed: {key}/{support}")
        if list(map(int, row["support_rows"])) != interior:
            raise ExperimentError(f"endpoint interior support changed: {key}")
    if len(seen) != config.EXPECTED_SCORE_COUNT:
        raise ExperimentError("endpoint denominator shrank")


def validate_complete(root: Path) -> dict[str, Any]:
    try:
        paths = config.RunPaths(root)
        protocol = _load(paths.protocol)
        history = _load(paths.history)
        if protocol.get("status") != "locked" or history.get("parity_pass") is not True:
            raise ExperimentError("protocol/history are not locked and parity-complete")
        audio_count = _validate_audio(protocol)
        face_count = _validate_faces(protocol)
        videos = _load(paths.videos_manifest)
        video_count, video_index = _validate_videos(protocol, videos)
        scores = _load(paths.scores_manifest)
        score_count, score_index = _validate_scores(protocol, scores, video_index)
        endpoints = _load(paths.endpoints)
        _validate_endpoints(protocol, endpoints, score_index)
        analysis = _load(paths.analysis)
        visual = _load(paths.visual_response)
        review = _load(paths.review)
        if analysis.get("status") != "complete" or analysis.get("scientific_decision") != "NOT_A_CONFIRMATION" or visual.get("status") != "complete" or review.get("status") != "complete":
            raise ExperimentError("analysis/visual/review artifacts are incomplete")
        fixed = analysis.get("flags", {})
        if fixed.get("training_authorized") is not False or fixed.get("generalization_established") is not False or fixed.get("historical_gate_repaired") is not False:
            raise ExperimentError("fixed scientific flags changed")
        return {"schema_version": 1, "status": "valid", "mode": "complete", "protocol_id": "wav2lip_global_shift_response", "record_count": len(protocol["records"]), "audio_artifacts": audio_count, "face_streams": face_count, "generated_videos": video_count, "score_cells": score_count, "endpoint_cells": len(endpoints["rows"]), "matrix_tolerance": 1e-4, "derived_tolerance": 1e-6, "independent": True, "checks": ["input protocol/history binding", "exact PCM shifts", "DYNAMIC/STATIC frame-box binding", "video/mux/score counts and hashes", "independent FULL/I endpoint reconstruction", "scientific flags"]}
    except Exception as exc:  # noqa: BLE001
        return {"schema_version": 1, "status": "invalid", "mode": "complete", "protocol_id": "wav2lip_global_shift_response", "error": str(exc), "independent": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate a Wav2Lip global shift response run")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_complete(args.run_root.resolve())
    output = args.run_root.resolve() / "validation.json"
    write_self_hashed_json(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
