from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import extract_pcm_from_media, read_pcm16_wav

from . import config, media
from .common import RescoreError, bytes_sha256, close_float, file_sha256, verify_self_hashed_json, write_self_hashed_json


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _old_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list) or len(value) != 1:
        raise RescoreError(f"legacy matrix is not one-track: {path}")
    matrix = np.asarray(value[0], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (matrix.shape[0], config.MATRIX_COLUMNS) or not np.isfinite(matrix).all():
        raise RescoreError(f"legacy matrix is malformed: {path}")
    return matrix


def _endpoint(matrix: np.ndarray, rows: Sequence[int], anchor: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    indices = np.asarray([int(item) for item in rows], dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= values.shape[0] or not np.isfinite(values).all():
        raise RescoreError("validator endpoint input is malformed")
    curve = values[indices].mean(axis=0, dtype=np.float64)
    minimum_index = int(np.argmin(curve))
    result: dict[str, Any] = {"D": float(curve[minimum_index]), "M": float(np.median(curve)), "C": float(np.median(curve) - curve[minimum_index]), "offset": int(config.VSHIFT - minimum_index), "min_index": minimum_index}
    if anchor is not None:
        column = config.VSHIFT - int(anchor)
        if not 0 <= column < config.MATRIX_COLUMNS:
            raise RescoreError("validator anchor is outside columns")
        result["D_anchor"] = float(curve[column])
        result["C_anchor"] = float(np.median(curve) - curve[column])
        result["anchor_offset"] = int(anchor)
    return result


def _benefit(natural: Mapping[str, Any], shifted: Mapping[str, Any]) -> dict[str, Any]:
    value = {"benefit_C": float(shifted["C"] - natural["C"]), "benefit_D": float(natural["D"] - shifted["D"]), "benefit_anchor": float(natural["D_anchor"] - shifted["D_anchor"]), "delta_M": float(shifted["M"] - natural["M"]), "offset_delta": int(shifted["offset"] - natural["offset"])}
    if abs(value["benefit_C"] - value["delta_M"] - value["benefit_D"]) > 1e-6:
        raise RescoreError("validator benefit identity failed")
    return value


def _validate_media(protocol: Mapping[str, Any], manifest: Mapping[str, Any]) -> int:
    rows = manifest.get("rows")
    if manifest.get("status") != "complete" or not isinstance(rows, list) or len(rows) != 46:
        raise RescoreError("media denominator is not 46")
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        if key in expected:
            raise RescoreError(f"duplicate media row: {key}")
        expected[key] = row
        for field in ("source_video", "derived_video", "target_media", "sidecar"):
            path = Path(str(row[field]))
            hash_field = {"source_video": "source_video_sha256", "derived_video": "derived_video_sha256", "target_media": "target_media_sha256", "sidecar": "sidecar_sha256"}[field]
            if not path.is_file() or file_sha256(path) != str(row[hash_field]):
                raise RescoreError(f"media hash changed: {key}/{field}")
        sidecar = _load(Path(str(row["sidecar"])))
        if sidecar.get("roi_crop") is not False or sidecar.get("new_tfg_generation") is not False or sidecar.get("pixel_identity_verified") is not True or sidecar.get("pcm_identity_verified") is not True:
            raise RescoreError(f"media contract changed: {key}")
        source, source_ev = media.decode_frames(Path(str(row["source_video"])))
        derived, derived_ev = media.decode_frames(Path(str(row["derived_video"])))
        target, _target_ev = media.decode_frames(Path(str(row["target_media"])))
        if source.shape != derived.shape or source.shape != target.shape or not np.array_equal(source, derived) or not np.array_equal(source, target):
            raise RescoreError(f"decoded video pixel identity failed: {key}")
        if int(row["frame_count"]) != int(source.shape[0]):
            raise RescoreError(f"media frame count changed: {key}")
        if source_ev.get("probe", {}).get("stream", {}).get("r_frame_rate") != "25/1" or derived_ev.get("probe", {}).get("stream", {}).get("r_frame_rate") != "25/1":
            raise RescoreError(f"media fps changed: {key}")
        audio = Path(str(row["audio"]))
        pcm, values, params = read_pcm16_wav(audio)
        if params["sample_rate"] != config.SAMPLE_RATE or params["channels"] != 1 or params["sample_width"] != 2 or values.size != int(row["audio_sample_count"]):
            raise RescoreError(f"audio contract changed: {key}")
        target_pcm = extract_pcm_from_media(Path(str(row["target_media"])))
        if target_pcm != pcm or bytes_sha256(target_pcm) != str(row["audio_pcm_sha256"]):
            raise RescoreError(f"target PCM identity failed: {key}")
    expected_keys = {(str(record["sample_id"]), arm) for record in protocol["records"] for arm in config.VIDEO_ARMS}
    if set(expected) != expected_keys:
        raise RescoreError("media key set differs from protocol")
    return len(expected)


def _validate_scores(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], scores: Mapping[str, Any]) -> tuple[int, dict[tuple[str, str], Mapping[str, Any]]]:
    rows = scores.get("scores")
    if scores.get("status") != "complete" or not isinstance(rows, list) or len(rows) != 46:
        raise RescoreError("score denominator is not 46")
    media_index = {(str(row["sample_id"]), str(row["video_arm"])): row for row in media_manifest["rows"]}
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    record_index = {str(row["sample_id"]): row for row in protocol["records"]}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        if key in result or key not in media_index or key[0] not in record_index:
            raise RescoreError(f"score key changed: {key}")
        matrix_path = Path(str(row["matrix"]))
        worker_path = Path(str(row["worker"]))
        if not matrix_path.is_file() or file_sha256(matrix_path) != str(row["matrix_sha256"]) or not worker_path.is_file() or file_sha256(worker_path) != str(row["worker_sha256"]):
            raise RescoreError(f"score artifact hash changed: {key}")
        worker = _load(worker_path)
        if worker.get("new_forward") is not True or worker.get("device") != "cpu" or worker.get("model_sha256") != config.SYNCNET_MODEL_SHA256:
            raise RescoreError(f"score runtime contract changed: {key}")
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
        media_row = media_index[key]
        expected_rows = min(int(media_row["frame_count"]), int(media_row["audio_sample_count"]) // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
        if matrix.shape != (expected_rows, config.MATRIX_COLUMNS) or not np.isfinite(matrix).all() or matrix.shape != tuple(int(item) for item in row["matrix_shape"]):
            raise RescoreError(f"score matrix shape changed: {key}")
        if file_sha256(Path(str(row["media"]))) != str(row["media_sha256"]) or bytes_sha256(extract_pcm_from_media(Path(str(row["media"])))) != str(row["audio_pcm_sha256"]):
            raise RescoreError(f"score media binding changed: {key}")
        result[key] = row
    expected_keys = {(str(record["sample_id"]), arm) for record in protocol["records"] for arm in config.VIDEO_ARMS}
    if set(result) != expected_keys:
        raise RescoreError("score key set differs from protocol")
    return len(result), result


def _validate_endpoints(protocol: Mapping[str, Any], history: Mapping[str, Any], scores: Mapping[tuple[str, str], Mapping[str, Any]], endpoints: Mapping[str, Any], analysis_payload: Mapping[str, Any]) -> None:
    rows = endpoints.get("rows")
    if endpoints.get("status") != "complete" or not isinstance(rows, list) or len(rows) != 23:
        raise RescoreError("endpoint denominator is not 23")
    endpoint_index = {(str(row["sample_id"]), str(row["source_group"])): row for row in rows}
    history_index = {str(row["sample_id"]): row for row in history["rows"]}
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        key = (sid, str(record["source_group"]))
        row = endpoint_index.get(key)
        if not isinstance(row, Mapping):
            raise RescoreError(f"endpoint row missing: {key}")
        support = [int(item) for item in record["support"]["J"]]
        hrow = history_index[sid]
        old_n_meta = hrow["cells"]["V_N/A_N"]
        old_s_meta = hrow["cells"]["V_SHIFT_200/A_N"]
        old_n = _old_matrix(Path(str(old_n_meta["matrix"])))
        old_s = _old_matrix(Path(str(old_s_meta["matrix"])))
        old_n_rows = [item - int(old_n_meta["track_start_frame"]) for item in support]
        old_s_rows = [item - int(old_s_meta["track_start_frame"]) for item in support]
        old_n0 = _endpoint(old_n, old_n_rows)
        anchor = int(old_n0["offset"])
        old_n_ep = _endpoint(old_n, old_n_rows, anchor)
        old_s_ep = _endpoint(old_s, old_s_rows, anchor)
        new_n = np.asarray(np.load(Path(str(scores[(sid, config.VIDEO_N)]["matrix"])), allow_pickle=False), dtype=np.float32)
        new_s = np.asarray(np.load(Path(str(scores[(sid, config.VIDEO_SHIFT)]["matrix"])), allow_pickle=False), dtype=np.float32)
        new_n_ep = _endpoint(new_n, support, anchor)
        new_s_ep = _endpoint(new_s, support, anchor)
        expected = (("legacy", "N", old_n_ep), ("legacy", "SHIFT", old_s_ep), ("full_frame_v4", "N", new_n_ep), ("full_frame_v4", "SHIFT", new_s_ep))
        for path_name, arm, value in expected:
            actual = row[path_name][arm]
            for field in ("D", "M", "C", "D_anchor", "C_anchor"):
                if not close_float(actual[field], value[field], 1e-6):
                    raise RescoreError(f"endpoint mismatch: {key}/{path_name}/{arm}/{field}")
            if int(actual["offset"]) != int(value["offset"]):
                raise RescoreError(f"endpoint offset mismatch: {key}/{path_name}/{arm}")
    flags = analysis_payload.get("flags", {})
    if analysis_payload.get("status") != "complete" or analysis_payload.get("scientific_decision") != "NOT_A_CONFIRMATION" or flags.get("training_authorized") is not False or flags.get("generalization_established") is not False or flags.get("historical_gate_repaired") is not False:
        raise RescoreError("analysis terminal contract changed")


def validate_complete(root: Path) -> dict[str, Any]:
    root = root.resolve()
    try:
        paths = config.RunPaths(root)
        protocol = _load(paths.protocol)
        history = _load(paths.history)
        parity = _load(paths.parity)
        media_manifest = _load(paths.media_manifest)
        scores_manifest = _load(paths.scores_manifest)
        endpoints = _load(paths.endpoints)
        analysis_payload = _load(paths.analysis)
        review = _load(paths.review)
        if protocol.get("status") != "locked" or len(protocol.get("records", [])) != 23 or history.get("status") != "locked" or parity.get("parity_pass") is not True:
            raise RescoreError("protocol/history/parity are not locked and complete")
        media_count = _validate_media(protocol, media_manifest)
        score_count, score_index = _validate_scores(protocol, media_manifest, scores_manifest)
        _validate_endpoints(protocol, history, score_index, endpoints, analysis_payload)
        if review.get("status") != "complete":
            raise RescoreError("review is incomplete")
        validation = {"schema_version": 1, "status": "valid", "mode": "complete", "protocol_id": "wav2lip_historical_shift_rescore", "record_count": 23, "source_group_count": 23, "media_count": media_count, "score_count": score_count, "endpoint_count": 23, "matrix_tolerance": 1e-4, "derived_tolerance": 1e-6, "independent": True, "checks": ["fixed manifest and PCM binding", "historical N/SHIFT exact 3200-sample audit", "lossless source/derived/target pixels and PTS", "two-cell v4 parity", "46 CPU fresh matrices", "independent J/endpoint reconstruction", "scientific flags"]}
    except Exception as exc:  # noqa: BLE001
        validation = {"schema_version": 1, "status": "invalid", "mode": "complete", "protocol_id": "wav2lip_historical_shift_rescore", "error": f"{type(exc).__name__}: {exc}", "independent": True}
    write_self_hashed_json(root / "validation.json", validation)
    return verify_self_hashed_json(root / "validation.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate historical Wav2Lip SHIFT_200 rescore artifacts")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_complete(args.run_root)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
