from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media
from scripts.experiments.wav2lip_roi_retiming_oracle.common import extract_pcm_from_media, read_pcm16_wav

from . import config
from .audio import build_arms
from .common import (
    ExperimentError,
    bytes_sha256,
    canonical_sha256,
    file_sha256,
    require_hash,
    verify_self_hashed_json,
    write_or_verify,
    write_self_hashed_json,
)
from .metrics import endpoint, make_bootstrap, bootstrap, compare


def _fixed_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    require_hash(path, expected, label)
    return verify_self_hashed_json(path)


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (config.REPO / path).resolve()


def _index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExperimentError(f"{label} contains a malformed row")
        key = str(row.get("sample_id", ""))
        if not key or key in result:
            raise ExperimentError(f"{label} has duplicate/empty sample_id: {key}")
        result[key] = row
    return result


def _spec_bindings() -> dict[str, dict[str, str]]:
    return {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in config.spec_paths().items()}


def load_selected_inputs() -> list[dict[str, Any]]:
    source = _fixed_json(config.S_PROTOCOL, config.H_HASHES["s_protocol"], "S protocol")
    if source.get("status") not in ("locked", "complete", "frozen"):
        raise ExperimentError(f"S protocol is not locked/complete: {source.get('status')}")
    records = source.get("records")
    if not isinstance(records, list) or len(records) != 22:
        raise ExperimentError("S protocol does not contain the registered 22 records")
    ordered = sorted(records, key=lambda row: (str(row.get("source_group")), str(row.get("sample_id"))))
    selected = [row for row in ordered[: config.EXPECTED_RECORD_COUNT]]
    ids = [str(row.get("sample_id")) for row in selected]
    if tuple(ids) != config.EXPECTED_IDS:
        raise ExperimentError(f"fixed 12-record selection differs: {ids}")
    if len({str(row.get("source_group")) for row in selected}) != config.EXPECTED_RECORD_COUNT:
        raise ExperimentError("selected records do not have one source group each")

    result: list[dict[str, Any]] = []
    for raw in selected:
        sid = str(raw["sample_id"])
        group = str(raw["source_group"])
        natural = raw.get("natural_audio")
        face = raw.get("face_video")
        roi = raw.get("roi")
        if not all(isinstance(item, Mapping) for item in (natural, face, roi)):
            raise ExperimentError(f"source bindings are incomplete: {sid}")
        natural_path = require_hash(_resolve(natural["path"]), str(natural["container_sha256"]), f"natural audio {sid}")
        natural_pcm, natural_values, params = read_pcm16_wav(natural_path)
        if bytes_sha256(natural_pcm) != str(natural.get("decoded_pcm_sha256")):
            raise ExperimentError(f"natural PCM hash changed: {sid}")
        if natural_values.size != int(raw["sample_count"]) or natural_values.size != int(natural.get("sample_count")):
            raise ExperimentError(f"natural sample count changed: {sid}")
        if params["sample_rate"] != config.SAMPLE_RATE or params["channels"] != 1 or params["sample_width"] != 2:
            raise ExperimentError(f"natural WAV format changed: {sid}")
        face_path = require_hash(_resolve(face["path"]), str(face["sha256"]), f"face video {sid}")
        frames, frame_evidence = parent_media.extract_bgr24_frames(face_path)
        frame_count = int(raw["frame_count"])
        if frames.ndim != 4 or frames.shape[1:] != (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3) or frames.shape[0] < frame_count:
            raise ExperimentError(f"face frame support changed: {sid}: {frames.shape}")
        boxes_path = require_hash(_resolve(roi["boxes_path"]), str(roi["boxes_sha256"]), f"ROI boxes {sid}")
        boxes_payload = verify_self_hashed_json(boxes_path)
        boxes = boxes_payload.get("boxes")
        if not isinstance(boxes, list) or len(boxes) < frame_count:
            raise ExperimentError(f"ROI boxes do not cover F frames: {sid}")
        if boxes_payload.get("source_video_sha256") != str(face["sha256"]):
            raise ExperimentError(f"ROI/video source binding changed: {sid}")
        selected_boxes: list[list[int]] = []
        for index, box in enumerate(boxes[:frame_count]):
            if not isinstance(box, list) or len(box) != 4:
                raise ExperimentError(f"ROI box malformed: {sid}/{index}")
            top, bottom, left, right = (int(value) for value in box)
            if not (0 <= top < bottom <= config.FRAME_HEIGHT and 0 <= left < right <= config.FRAME_WIDTH):
                raise ExperimentError(f"ROI box outside frame: {sid}/{index}")
            selected_boxes.append([top, bottom, left, right])
        result.append(
            {
                "sample_id": sid,
                "source_group": group,
                "frame_count": frame_count,
                "sample_count": int(natural_values.size),
                "natural_audio": {
                    "path": str(natural_path),
                    "container_sha256": file_sha256(natural_path),
                    "decoded_pcm_sha256": bytes_sha256(natural_pcm),
                    "sample_count": int(natural_values.size),
                },
                "face_video": {"path": str(face_path), "sha256": file_sha256(face_path), "source_frame_count": int(frames.shape[0]), "source_pixel_sha256": bytes_sha256(np.ascontiguousarray(frames[:frame_count]).tobytes()), "probe": frame_evidence.get("probe")},
                "roi": {"source_path": str(boxes_path), "source_sha256": file_sha256(boxes_path), "frame_count": len(boxes), "boxes": selected_boxes, "source_box_track_sha256": str(boxes_payload.get("box_track_sha256", ""))},
                "_natural_pcm": bytes(natural_pcm),
                "_frames": np.ascontiguousarray(frames[:frame_count]),
                "_boxes": selected_boxes,
            }
        )
    return result


def _face_payload(row: Mapping[str, Any], mode: str, video: Mapping[str, Any], box: Mapping[str, Any]) -> dict[str, Any]:
    return {"mode": mode, "video": dict(video), "boxes": dict(box), "frame_count": int(row["frame_count"]), "sample_id": str(row["sample_id"]), "source_group": str(row["source_group"])}


def prepare_faces(rows: Sequence[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        source_frames = np.asarray(row["_frames"], dtype=np.uint8)
        source_boxes = [[int(value) for value in box] for box in row["_boxes"]]
        row_modes: dict[str, Any] = {}
        for mode in config.FACE_MODES:
            frames = source_frames if mode == config.FACE_DYNAMIC else np.repeat(source_frames[:1], source_frames.shape[0], axis=0)
            boxes = source_boxes if mode == config.FACE_DYNAMIC else [list(source_boxes[0]) for _ in source_boxes]
            video_path = root / "faces" / f"{sid}__{mode}.mkv"
            if video_path.is_file():
                decoded, _evidence = parent_media.extract_bgr24_frames(video_path)
                if not np.array_equal(decoded, frames):
                    raise ExperimentError(f"derived face video differs from frozen {mode} frames: {sid}")
            else:
                parent_media.encode_video(frames, video_path)
            decoded, evidence = parent_media.extract_bgr24_frames(video_path)
            if not np.array_equal(decoded, frames):
                raise ExperimentError(f"derived face pixel round-trip failed: {sid}/{mode}")
            box_payload = {
                "schema_version": 1,
                "protocol_id": "wav2lip_global_shift_response",
                "sample_id": sid,
                "source_group": str(row["source_group"]),
                "face_mode": mode,
                "box_order": ["top", "bottom", "left", "right"],
                "source_boxes": str(row["roi"]["source_path"]),
                "source_boxes_sha256": str(row["roi"]["source_sha256"]),
                "source_video_sha256": str(row["face_video"]["sha256"]),
                "boxes": boxes,
                "box_track_sha256": canonical_sha256(boxes),
                "construction": "dynamic decoded source frames and source boxes" if mode == config.FACE_DYNAMIC else "repeat decoded frame zero and box zero together",
            }
            box_path = root / "faces" / f"{sid}__{mode}__boxes.json"
            box_info = write_or_verify(box_path, box_payload)
            video_info = {"path": str(video_path.resolve()), "sha256": file_sha256(video_path), "pixel_sha256": bytes_sha256(np.ascontiguousarray(decoded).tobytes()), "frame_count": int(decoded.shape[0]), "probe": evidence.get("probe"), "pixel_identity_verified": True}
            row_modes[mode] = _face_payload(row, mode, video_info, {**box_payload, "path": str(box_path.resolve()), "sha256": file_sha256(box_path), "artifact_sha256": box_info.get("artifact_sha256")})
        records.append({"sample_id": sid, "source_group": str(row["source_group"]), "modes": row_modes})
    payload = {"schema_version": 1, "stage_id": "faces", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "record_count": len(records), "face_modes": list(config.FACE_MODES), "rows": records}
    write_self_hashed_json(root / "faces/manifest.json", payload)
    return records


def prepare_audio(rows: Sequence[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        arms = build_arms(row["_natural_pcm"], root / "audio", str(row["sample_id"]))
        output.append({"sample_id": str(row["sample_id"]), "source_group": str(row["source_group"]), "arms": arms})
    payload = {"schema_version": 1, "stage_id": "audio", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "record_count": len(output), "arms": list(config.AUDIO_ARMS), "rows": output}
    write_self_hashed_json(root / "audio/manifest.json", payload)
    return output


def _public_input(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_protocol(run_id: str, rows: Sequence[dict[str, Any]], face_rows: Sequence[Mapping[str, Any]], audio_rows: Sequence[Mapping[str, Any]], audit_sha: str) -> dict[str, Any]:
    face_index = {str(row["sample_id"]): row for row in face_rows}
    audio_index = {str(row["sample_id"]): row for row in audio_rows}
    records: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        modes = face_index[sid]["modes"]
        records.append({"sample_id": sid, "source_group": str(row["source_group"]), "frame_count": int(row["frame_count"]), "sample_count": int(row["sample_count"]), "natural_audio": dict(row["natural_audio"]), "face_video": dict(row["face_video"]), "roi": {key: value for key, value in row["roi"].items() if key != "boxes"}, "audio": audio_index[sid], "face_modes": modes})
        for mode in config.FACE_MODES:
            for video_arm, audio_arm in config.SCORE_CELLS:
                cells.append({"sample_id": sid, "source_group": str(row["source_group"]), "face_mode": mode, "video_arm": video_arm, "audio_arm": audio_arm, "cell_key": config.cell_key(sid, mode, video_arm, audio_arm)})
    payload = {
        "schema_version": 1,
        "protocol_id": "wav2lip_global_shift_response",
        "protocol_revision": "global_shift_response_v1",
        "run_id": run_id,
        "status": "locked",
        "classification": "seen_fit_diagnostic",
        "input_audit_sha256": audit_sha,
        "spec_bindings": _spec_bindings(),
        "frozen_config": config.FrozenConfig().to_dict(),
        "record_count": len(records),
        "video_count": config.EXPECTED_VIDEO_COUNT,
        "score_count": config.EXPECTED_SCORE_COUNT,
        "selected_ids": [str(row["sample_id"]) for row in rows],
        "records": records,
        "cells": cells,
        "fresh_matrix_policy": "join by sample_id+face_mode+video_arm+audio_arm; no zip or score-based retry",
    }
    return payload


def _h_audio_index(payload: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            raise ExperimentError("historical audio row is malformed")
        arms = row.get("arms")
        if not isinstance(arms, list):
            raise ExperimentError("historical audio arms are not a list")
        result[str(row["sample_id"])] = {str(item["arm"]): item for item in arms if isinstance(item, Mapping)}
    return result


def _h_video_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sample_id"]): row for row in payload.get("rows", []) if isinstance(row, Mapping)}


def _h_score_index(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("scores", []):
        if not isinstance(row, Mapping):
            raise ExperimentError("historical score row is malformed")
        key = (str(row["sample_id"]), str(row["cell"]))
        if key in result:
            raise ExperimentError(f"duplicate historical score cell: {key}")
        result[key] = row
    return result


def _load_history_matrix(reference: str) -> tuple[Path, Path, np.ndarray, int, int]:
    base = config.H_ROOT / "04_scores/syncnet" / reference / "pywork" / reference
    tracks_path = base / "tracks.pckl"
    matrix_path = base / "activesd.pckl"
    if not tracks_path.is_file() or not matrix_path.is_file():
        raise ExperimentError(f"historical SyncNet cache is incomplete: {reference}")
    tracks = pickle.loads(tracks_path.read_bytes())
    active = pickle.loads(matrix_path.read_bytes())
    if not isinstance(tracks, list) or len(tracks) != 1 or not isinstance(active, list) or len(active) != 1:
        raise ExperimentError(f"historical cache has multiple/empty tracks: {reference}")
    track = tracks[0].get("track") if isinstance(tracks[0], Mapping) else None
    frames = np.asarray(track.get("frame")) if isinstance(track, Mapping) else np.asarray([])
    matrix = np.asarray(active[0], dtype=np.float32)
    if frames.ndim != 1 or frames.size == 0 or matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or matrix.shape[0] < 1 or matrix.dtype != np.dtype("float32") or not np.isfinite(matrix).all():
        raise ExperimentError(f"historical cache matrix/track malformed: {reference}")
    if not np.array_equal(frames, np.arange(int(frames[0]), int(frames[0]) + frames.size)):
        raise ExperimentError(f"historical track frames are not contiguous: {reference}")
    if matrix.shape[0] > frames.size:
        raise ExperimentError(f"historical matrix exceeds track support: {reference}")
    return tracks_path, matrix_path, matrix, int(frames[0]), int(matrix.shape[0])


def _history_endpoint(matrix: np.ndarray, rows: Sequence[int], anchor: int | None = None) -> dict[str, Any]:
    return endpoint(matrix, rows, anchor_offset=anchor)


def _original_bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    labels = sorted({str(group) for group in groups})
    rng = np.random.default_rng(20260904)
    draws = rng.integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    value_by_group = {str(group): float(value) for group, value in zip(groups, values, strict=True)}
    ordered = np.asarray([value_by_group[label] for label in labels], dtype=np.float64)
    estimate = ordered[draws].mean(axis=1)
    return {"mean": float(np.mean(ordered)), "ci95": [float(np.quantile(estimate, 0.025, method="linear")), float(np.quantile(estimate, 0.975, method="linear"))], "draws": config.BOOTSTRAP_DRAWS, "seed": 20260904, "source_group_count": len(labels), "rng": "numpy_default_rng_pcg64", "quantile_method": "linear"}


def audit_history(root: Path) -> dict[str, Any]:
    cohort = _fixed_json(config.H_COHORT, config.H_HASHES["cohort"], "historical cohort")
    audio = _fixed_json(config.H_AUDIO, config.H_HASHES["audio"], "historical audio manifest")
    videos = _fixed_json(config.H_VIDEOS, config.H_HASHES["videos"], "historical videos manifest")
    scores = _fixed_json(config.H_SCORES, config.H_HASHES["scores"], "historical scores manifest")
    _fixed_json(config.H_FINAL, config.H_HASHES["final"], "historical final")
    cohort_rows = cohort.get("records")
    if not isinstance(cohort_rows, list) or len(cohort_rows) != 23:
        raise ExperimentError("historical cohort count differs")
    audio_by_id, video_by_id, score_by_key = _h_audio_index(audio), _h_video_index(videos), _h_score_index(scores)
    cells = ("V_N/A_N", "V_SHIFT_200/A_N", "V_SHIFT_200/A_SHIFT_200")
    history_rows: list[dict[str, Any]] = []
    scalar_c: list[float] = []
    scalar_d: list[float] = []
    scalar_groups: list[str] = []
    for record in cohort_rows:
        sid = str(record["sample_id"])
        group = str(record["source_group"])
        if sid not in audio_by_id or sid not in video_by_id:
            raise ExperimentError(f"historical source join is incomplete: {sid}")
        cell_rows: dict[str, dict[str, Any]] = {}
        for cell in cells:
            score = score_by_key.get((sid, cell))
            if score is None:
                raise ExperimentError(f"historical score cell missing: {sid}/{cell}")
            reference = str(score["reference"])
            tracks_path, matrix_path, matrix, f0, t = _load_history_matrix(reference)
            media_path = _resolve(score["media"])
            if file_sha256(media_path) != str(score["media_sha256"]):
                raise ExperimentError(f"historical mux hash changed: {sid}/{cell}")
            audio_arm = "SHIFT_200" if cell.endswith("A_SHIFT_200") else "N"
            audio_item = audio_by_id[sid].get(audio_arm)
            if not isinstance(audio_item, Mapping):
                raise ExperimentError(f"historical audio arm missing: {sid}/{audio_arm}")
            audio_path = _resolve(audio_item["output"])
            pcm, values, _params = read_pcm16_wav(audio_path)
            if file_sha256(audio_path) != str(audio_item["output_sha256"]):
                raise ExperimentError(f"historical audio hash changed: {sid}/{audio_arm}")
            if extract_pcm_from_media(media_path) != pcm:
                raise ExperimentError(f"historical mux audio differs: {sid}/{cell}")
            full = _history_endpoint(matrix, range(t))
            if abs(float(full["C"]) - float(score["sync_c"])) > 0.000501 or abs(float(full["D"]) - float(score["sync_d"])) > 0.000501 or int(full["offset"]) != int(score["av_offset"]):
                raise ExperimentError(f"historical FULL parity failed: {sid}/{cell}")
            cell_rows[cell] = {"sample_id": sid, "source_group": group, "cell": cell, "reference": reference, "tracks": str(tracks_path.resolve()), "tracks_sha256": file_sha256(tracks_path), "matrix": str(matrix_path.resolve()), "matrix_sha256": file_sha256(matrix_path), "shape": [int(x) for x in matrix.shape], "track_start_frame": f0, "track_end_frame": f0 + t - 1, "full": full, "scalar": {"sync_c": float(score["sync_c"]), "sync_d": float(score["sync_d"]), "offset": int(score["av_offset"])}, "audio": {"path": str(audio_path), "sha256": file_sha256(audio_path), "pcm_sha256": bytes_sha256(pcm), "sample_count": int(values.size), "arm": audio_arm}, "media": {"path": str(media_path), "sha256": file_sha256(media_path)}}
        starts = [int(item["track_start_frame"]) for item in cell_rows.values()]
        stops = [int(item["track_end_frame"]) + 1 for item in cell_rows.values()]
        support_start = max(start + 30 for start in starts)
        support_stop = min(stop - 30 for stop in stops)
        if support_stop <= support_start:
            raise ExperimentError(f"historical common interior support is empty: {sid}")
        baseline = cell_rows["V_N/A_N"]
        baseline_offset_full = int(baseline["full"]["offset"])
        ih: dict[str, Any] = {}
        for cell, item in cell_rows.items():
            matrix = np.asarray(pickle.loads(Path(item["matrix"]).read_bytes())[0], dtype=np.float32)
            local_rows = list(range(support_start - int(item["track_start_frame"]), support_stop - int(item["track_start_frame"])))
            item["interior_absolute_frames"] = [support_start, support_stop]
            item["interior_rows"] = local_rows
            item["interior"] = _history_endpoint(matrix, local_rows, anchor=baseline_offset_full)
            ih[cell] = item["interior"]
        for cell in ("V_SHIFT_200/A_N", "V_SHIFT_200/A_SHIFT_200"):
            scalar_item = cell_rows[cell]
            if cell == "V_SHIFT_200/A_N":
                scalar_c.append(float(scalar_item["scalar"]["sync_c"] - baseline["scalar"]["sync_c"]))
                scalar_d.append(float(baseline["scalar"]["sync_d"] - scalar_item["scalar"]["sync_d"]))
                scalar_groups.append(group)
        shift_pcm = cell_rows["V_SHIFT_200/A_SHIFT_200"]["audio"]
        n_pcm = cell_rows["V_N/A_N"]["audio"]
        natural_pcm = read_pcm16_wav(Path(n_pcm["path"]))[0]
        delayed = np.concatenate((np.zeros(min(config.AUDIO_SHIFT, len(natural_pcm) // 2), dtype="<i2"), np.frombuffer(natural_pcm, dtype="<i2")[: max(0, len(natural_pcm) // 2 - config.AUDIO_SHIFT)])).astype("<i2").tobytes()
        if shift_pcm["pcm_sha256"] != bytes_sha256(delayed):
            raise ExperimentError(f"historical SHIFT_200 construction differs: {sid}")
        history_rows.append({"sample_id": sid, "source_group": group, "cells": cell_rows, "common_absolute_support": [support_start, support_stop], "common_row_count": support_stop - support_start, "baseline_offset_full": baseline_offset_full, "track_start_frames": {cell: int(item["track_start_frame"]) for cell, item in cell_rows.items()}})
    if len(history_rows) != 23:
        raise ExperimentError("historical row denominator changed")
    labels = sorted(scalar_groups)
    result = {"schema_version": 1, "stage_id": "history", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "parity_pass": True, "record_count": 23, "cell_count": 69, "rows": history_rows, "scalar_original_shift_vs_n": {"C": _original_bootstrap(scalar_c, scalar_groups), "D": _original_bootstrap(scalar_d, scalar_groups)}, "bootstrap_seed_new_derived": config.BOOTSTRAP_SEED, "legacy_cache_locked_at_audit": True, "track_start_caveat": "common absolute-frame support is mapped to each matrix local row; tracks are not assumed to start at zero"}
    write_self_hashed_json(root / "history.json", result)
    return result


def build_input_audit(rows: Sequence[Mapping[str, Any]], history: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": "wav2lip_global_shift_response",
        "status": "complete",
        "classification": "seen_fit_diagnostic",
        "record_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in rows}),
        "selected_ids": [str(row["sample_id"]) for row in rows],
        "historical_record_count": int(history["record_count"]),
        "historical_cell_count": int(history["cell_count"]),
        "history_parity_pass": bool(history["parity_pass"]),
        "records": [{"sample_id": str(row["sample_id"]), "source_group": str(row["source_group"]), "frame_count": int(row["frame_count"]), "sample_count": int(row["sample_count"]), "natural_audio": dict(row["natural_audio"]), "face_video": dict(row["face_video"]), "roi": {key: value for key, value in row["roi"].items() if key != "boxes"}} for row in rows],
    }


def prepare(run_id: str, paths: config.RunPaths) -> dict[str, Any]:
    if paths.final.is_file():
        raise ExperimentError(f"terminal run already exists: {paths.final}")
    rows = load_selected_inputs()
    # Historical input audit must complete before any fresh GPU generation.
    history = audit_history(paths.root)
    audit = build_input_audit(rows, history)
    audit_sha = write_self_hashed_json(paths.input_audit, audit)
    audio_rows = prepare_audio(rows, paths.root)
    face_rows = prepare_faces(rows, paths.root)
    protocol = build_protocol(run_id, rows, face_rows, audio_rows, audit_sha)
    protocol_sha = write_self_hashed_json(paths.protocol, protocol)
    protocol = verify_self_hashed_json(paths.protocol)
    return {"protocol": protocol, "protocol_sha256": protocol_sha, "history": history, "rows": rows, "audio_rows": audio_rows, "face_rows": face_rows, "audit": audit}


def load_prepared(paths: config.RunPaths) -> dict[str, Any]:
    protocol = verify_self_hashed_json(paths.protocol)
    history = verify_self_hashed_json(paths.history)
    audit = verify_self_hashed_json(paths.input_audit)
    audio = verify_self_hashed_json(paths.audio_manifest)
    faces = verify_self_hashed_json(paths.faces_manifest)
    if protocol.get("status") != "locked" or history.get("parity_pass") is not True or audit.get("history_parity_pass") is not True:
        raise ExperimentError("prepared artifacts are not locked and history-complete")
    return {"protocol": protocol, "history": history, "audit": audit, "audio": audio, "faces": faces}
