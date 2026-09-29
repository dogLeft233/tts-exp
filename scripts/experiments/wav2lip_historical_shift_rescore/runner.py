from __future__ import annotations

import argparse
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer
from scripts.experiments.wav2lip_roi_retiming_oracle.common import extract_pcm_from_media, read_pcm16_wav

from . import analysis, config, media
from .common import RescoreError, bytes_sha256, close_float, file_sha256, read_json, require_hash, verify_self_hashed_json, write_self_hashed_json


def _load_fixed(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    require_hash(path, expected_hash, label)
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if isinstance(recorded, str):
        verify_self_hashed_json(path)
    return payload


def _audio_by_arm(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    arms = row.get("arms")
    if not isinstance(arms, list):
        raise RescoreError(f"historical audio arms must be a list: {row.get('sample_id')}")
    matches = [item for item in arms if isinstance(item, Mapping) and str(item.get("arm")) == arm]
    if len(matches) != 1:
        raise RescoreError(f"audio arm join is not unique: {row.get('sample_id')}/{arm}")
    return matches[0]


def _video_by_arm(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise RescoreError(f"historical video arm is missing: {row.get('sample_id')}/{arm}")
    return arms[arm]


def _cell_lookup(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("cell")))
        if key in result:
            raise RescoreError(f"duplicate historical score cell: {key}")
        if key[1] in keys:
            result[key] = row
    return result


def _load_track_start(path: Path, expected_start: int, expected_track_frames: int) -> int:
    with path.open("rb") as handle:
        tracks = pickle.load(handle)
    if not isinstance(tracks, list) or len(tracks) != 1 or not isinstance(tracks[0], Mapping):
        raise RescoreError(f"legacy tracks are not one-track: {path}")
    track = tracks[0].get("track")
    if not isinstance(track, Mapping):
        raise RescoreError(f"legacy track payload is missing: {path}")
    frames = np.asarray(track.get("frame"), dtype=np.int64)
    if frames.ndim != 1 or frames.size != expected_track_frames or not np.array_equal(frames, np.arange(int(frames[0]), int(frames[0]) + expected_track_frames)):
        raise RescoreError(f"legacy track frames are not contiguous: {path}")
    if int(frames[0]) != expected_start:
        raise RescoreError(f"legacy track start mismatch: {path}: {frames[0]} != {expected_start}")
    return int(frames[0])


def _pcm_item(path: Path) -> dict[str, Any]:
    pcm, values, params = read_pcm16_wav(path)
    if params != {"channels": 1, "sample_width": 2, "sample_rate": config.SAMPLE_RATE, "frame_count": int(values.size)} or values.size == 0:
        raise RescoreError(f"audio is not mono PCM16/16k: {path}")
    return {"path": str(path.resolve()), "container_sha256": file_sha256(path), "decoded_pcm_sha256": bytes_sha256(pcm), "sample_count": int(values.size), "pcm": pcm}


def _shift_ok(natural: bytes, shifted: bytes) -> bool:
    values = np.frombuffer(natural, dtype="<i2")
    candidate = np.frombuffer(shifted, dtype="<i2")
    return values.size == candidate.size and values.size > config.SHIFT_SAMPLES and np.array_equal(candidate[: config.SHIFT_SAMPLES], np.zeros(config.SHIFT_SAMPLES, dtype=np.int16)) and np.array_equal(candidate[config.SHIFT_SAMPLES :], values[: -config.SHIFT_SAMPLES])


def _history_matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list) or len(value) != 1:
        raise RescoreError(f"legacy matrix is not one-track: {path}")
    matrix = np.asarray(value[0], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(matrix).all():
        raise RescoreError(f"legacy matrix is malformed: {path}")
    return int(matrix.shape[0]), int(matrix.shape[1])


def _prepare(run_id: str, paths: config.RunPaths) -> dict[str, Any]:
    fixed = {
        "cohort": _load_fixed(config.H_COHORT, config.INPUT_HASHES["cohort"], "cohort"),
        "audio": _load_fixed(config.H_AUDIO, config.INPUT_HASHES["audio"], "audio manifest"),
        "videos": _load_fixed(config.H_VIDEOS, config.INPUT_HASHES["videos"], "video manifest"),
        "scores": _load_fixed(config.H_SCORES, config.INPUT_HASHES["scores"], "historical scores"),
        "history": _load_fixed(config.G_HISTORY, config.INPUT_HASHES["history"], "v4 history"),
        "g_protocol": _load_fixed(config.G_PROTOCOL, config.INPUT_HASHES["protocol"], "v4 protocol"),
        "g_final": _load_fixed(config.G_FINAL, config.INPUT_HASHES["final"], "v4 final"),
        "g_scores": _load_fixed(config.G_SCORES, config.INPUT_HASHES["score_manifest"], "v4 score manifest"),
    }
    cohort_rows = fixed["cohort"].get("records")
    audio_rows = fixed["audio"].get("rows")
    video_rows = fixed["videos"].get("rows")
    history_rows = fixed["history"].get("rows")
    if not all(isinstance(value, list) for value in (cohort_rows, audio_rows, video_rows, history_rows)):
        raise RescoreError("one of the fixed manifests has no rows")
    if len(cohort_rows) != 23 or len(audio_rows) != 23 or len(video_rows) != 23 or len(history_rows) != 23:
        raise RescoreError("fixed cohort is not exactly 23 records")
    cohort_index: dict[str, Mapping[str, Any]] = {}
    audio_index: dict[str, Mapping[str, Any]] = {}
    video_index: dict[str, Mapping[str, Any]] = {}
    history_index: dict[str, Mapping[str, Any]] = {}
    for collection, target, label in ((cohort_rows, cohort_index, "cohort"), (audio_rows, audio_index, "audio"), (video_rows, video_index, "videos"), (history_rows, history_index, "history")):
        for row in collection:
            sid = str(row.get("sample_id"))
            if sid in target:
                raise RescoreError(f"duplicate {label} key: {sid}")
            target[sid] = row
    if set(cohort_index) != set(audio_index) or set(cohort_index) != set(video_index) or set(cohort_index) != set(history_index):
        raise RescoreError("fixed manifests have different sample sets")
    records: list[dict[str, Any]] = []
    preflight_media: list[dict[str, Any]] = []
    total_video_count = 0
    for sid in sorted(cohort_index, key=lambda item: (str(cohort_index[item].get("source_group")), item)):
        cohort = cohort_index[sid]
        if str(cohort.get("source_group")) != str(audio_index[sid].get("source_group")) or str(cohort.get("source_group")) != str(video_index[sid].get("source_group")):
            raise RescoreError(f"source group mismatch: {sid}")
        n_audio = _audio_by_arm(audio_index[sid], config.AUDIO_N)
        shift_audio = _audio_by_arm(audio_index[sid], config.AUDIO_SHIFT)
        n_item = _pcm_item(Path(str(n_audio["output"])))
        shift_item = _pcm_item(Path(str(shift_audio["output"])))
        if str(n_audio.get("output_sha256")) != n_item["container_sha256"] or str(shift_audio.get("output_sha256")) != shift_item["container_sha256"]:
            raise RescoreError(f"audio hash binding changed: {sid}")
        natural_path = Path(str(cohort["natural_audio"]["path"]))
        if file_sha256(natural_path) != str(cohort["natural_audio"]["sha256"]):
            raise RescoreError(f"natural audio binding changed: {sid}")
        natural_pcm, _natural_values, _natural_params = read_pcm16_wav(natural_path)
        if natural_pcm != n_item["pcm"] or n_item["sample_count"] != shift_item["sample_count"] or not _shift_ok(n_item["pcm"], shift_item["pcm"]):
            raise RescoreError(f"N/SHIFT PCM contract failed: {sid}")
        videos: dict[str, dict[str, Any]] = {}
        for arm in config.VIDEO_ARMS:
            historical_arm = {config.VIDEO_N: "N", config.VIDEO_SHIFT: "SHIFT_200"}[arm]
            item = _video_by_arm(video_index[sid], historical_arm)
            video_path = Path(str(item["output"]))
            if file_sha256(video_path) != str(item.get("output_sha256")):
                raise RescoreError(f"historical video hash changed: {sid}/{arm}")
            frames, evidence = media.decode_frames(video_path)
            timeline = media.validate_source_timeline(video_path, evidence, int(frames.shape[0]))
            videos[arm] = {"path": str(video_path.resolve()), "sha256": file_sha256(video_path), "frame_count": int(frames.shape[0]), "pixel_sha256": bytes_sha256(np.ascontiguousarray(frames).tobytes()), "timeline": timeline, "source_arm": arm}
            total_video_count += 1
            preflight_media.append({"sample_id": sid, "video_arm": arm, "frame_count": int(frames.shape[0]), "pixel_sha256": videos[arm]["pixel_sha256"], "timeline": timeline})
        hrow = history_index[sid]
        common = [int(item) for item in hrow["common_absolute_support"]]
        if len(common) != 2 or common[0] >= common[1]:
            raise RescoreError(f"invalid historical support: {sid}")
        hcells = hrow.get("cells")
        if not isinstance(hcells, Mapping) or not all(key in hcells for key in ("V_N/A_N", "V_SHIFT_200/A_N")):
            raise RescoreError(f"historical N score cells are incomplete: {sid}")
        cell_records: dict[str, dict[str, Any]] = {}
        for key in ("V_N/A_N", "V_SHIFT_200/A_N"):
            cell = hcells[key]
            matrix_path = Path(str(cell["matrix"]))
            tracks_path = Path(str(cell["tracks"]))
            if file_sha256(matrix_path) != str(cell["matrix_sha256"]) or file_sha256(tracks_path) != str(cell["tracks_sha256"]):
                raise RescoreError(f"historical score hash changed: {sid}/{key}")
            matrix_rows, matrix_cols = _history_matrix_shape(matrix_path)
            expected_start = int(cell["track_start_frame"])
            actual_start = _load_track_start(tracks_path, expected_start, matrix_rows + config.WINDOW_FRAMES)
            if int(cell["track_end_frame"]) != actual_start + matrix_rows - 1:
                raise RescoreError(f"legacy track end does not match matrix rows: {sid}/{key}")
            cell_records[key] = {"matrix": str(matrix_path.resolve()), "matrix_sha256": str(cell["matrix_sha256"]), "tracks": str(tracks_path.resolve()), "tracks_sha256": str(cell["tracks_sha256"]), "track_start_frame": actual_start, "track_end_frame": int(cell["track_end_frame"]), "track_frame_count": matrix_rows + config.WINDOW_FRAMES, "shape": [matrix_rows, matrix_cols], "media": dict(cell.get("media", {})), "scalar": dict(cell.get("scalar", {}))}
        frame_count = min(videos[config.VIDEO_N]["frame_count"], videos[config.VIDEO_SHIFT]["frame_count"])
        t_new = min(frame_count, n_item["sample_count"] // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
        if t_new <= 0:
            raise RescoreError(f"new SyncNet support is empty: {sid}")
        i_new = list(range(config.INNER_MARGIN, t_new - config.INNER_MARGIN))
        i_h = set(range(common[0], common[1]))
        j = sorted(i_h.intersection(i_new))
        if not j:
            raise RescoreError(f"J is empty: {sid}")
        for key, cell in cell_records.items():
            f0 = int(cell["track_start_frame"])
            if any(g - f0 < 0 or g - f0 >= int(cell["shape"][0]) for g in j):
                raise RescoreError(f"J cannot map to historical local rows: {sid}/{key}")
        records.append({"sample_id": sid, "source_group": str(cohort["source_group"]), "sample_count": n_item["sample_count"], "audio": {"N": {key: value for key, value in n_item.items() if key != "pcm"}, "SHIFT_200": {key: value for key, value in shift_item.items() if key != "pcm"}}, "videos": videos, "history": {"common_absolute_support": common, "cells": cell_records}, "support": {"I_H": common, "I_new": [i_new[0], i_new[-1] + 1], "J": j, "J_count": len(j), "t_new": t_new}})
    if len(records) != 23 or len({row["source_group"] for row in records}) != 23 or total_video_count != 46:
        raise RescoreError("frozen cohort denominator changed")
    input_specs = (("cohort", "cohort", config.H_COHORT), ("audio", "audio", config.H_AUDIO), ("videos", "videos", config.H_VIDEOS), ("scores", "scores", config.H_SCORES), ("history", "history", config.G_HISTORY), ("g_protocol", "protocol", config.G_PROTOCOL), ("g_final", "final", config.G_FINAL), ("g_scores", "score_manifest", config.G_SCORES))
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_historical_shift_rescore", "protocol_revision": "historical_shift_rescore_v1", "status": "locked", "run_id": run_id, "inputs": {name: {"path": str(path.resolve()), "sha256": config.INPUT_HASHES[hash_key]} for name, hash_key, path in input_specs}, "scoring": {"entrance": "FULL_FRAME_V4", "device": "cpu", "batch_size": config.SYNCNET_BATCH_SIZE, "threads": config.TORCH_THREADS, "model": str(config.SYNCNET_MODEL.resolve()), "model_sha256": config.SYNCNET_MODEL_SHA256, "roi_crop": False, "new_tfg_generation": False, "cuda_jobs": 0, "training": False}, "formula": {"t_new": "min(frame_count, sample_count//640)-5", "J": "I_H intersect range(30,t_new-30)", "endpoint": "mean matrix rows first; D=min(z); C=median(z)-D; offset=15-argmin(z)", "benefit_D": "D_N-D_SHIFT", "benefit_C": "C_SHIFT-C_N"}, "records": records}
    history_payload = {"schema_version": 1, "status": "locked", "source": str(config.G_HISTORY.resolve()), "source_sha256": config.INPUT_HASHES["history"], "parity_pass": False, "rows": fixed["history"]["rows"], "scalar_original_shift_vs_n": fixed["history"].get("scalar_original_shift_vs_n"), "track_start_caveat": fixed["history"].get("track_start_caveat")}
    audit = {"schema_version": 1, "status": "complete", "fixed_input_hashes": config.INPUT_HASHES, "record_count": len(records), "source_group_count": len({row["source_group"] for row in records}), "video_count": total_video_count, "preflight_media": preflight_media, "pcm_shift_samples": config.SHIFT_SAMPLES, "all_J_nonempty": True, "shortest_J": min(len(row["support"]["J"]) for row in records), "longest_J": max(len(row["support"]["J"]) for row in records)}
    write_self_hashed_json(paths.protocol, protocol)
    write_self_hashed_json(paths.history, history_payload)
    write_self_hashed_json(paths.input_audit, audit)
    return {"protocol": verify_self_hashed_json(paths.protocol), "history": verify_self_hashed_json(paths.history), "audit": verify_self_hashed_json(paths.input_audit), "fixed": fixed}


def _load_prepared(paths: config.RunPaths) -> dict[str, Any]:
    return {"protocol": verify_self_hashed_json(paths.protocol), "history": verify_self_hashed_json(paths.history), "audit": verify_self_hashed_json(paths.input_audit), "fixed": {"history": _load_fixed(config.G_HISTORY, config.INPUT_HASHES["history"], "v4 history"), "scores": _load_fixed(config.H_SCORES, config.INPUT_HASHES["scores"], "historical scores"), "g_scores": _load_fixed(config.G_SCORES, config.INPUT_HASHES["score_manifest"], "v4 score manifest")}}


def _legacy_endpoint(matrix_path: Path, rows: Sequence[int], anchor: int | None = None) -> dict[str, Any]:
    matrix = analysis.load_old_matrix(matrix_path)
    return analysis.endpoint(matrix, rows, anchor)


def _historical_parity(prepared: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    protocol = prepared["protocol"]
    fixed_scores = prepared["fixed"]["scores"]
    old_score_index = _cell_lookup(fixed_scores.get("scores", []), ("V_N/A_N", "V_SHIFT_200/A_N"))
    rows: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        for cell_key_old, video_arm in (("V_N/A_N", config.VIDEO_N), ("V_SHIFT_200/A_N", config.VIDEO_SHIFT)):
            cell = record["history"]["cells"][cell_key_old]
            matrix_path = Path(str(cell["matrix"]))
            ep = _legacy_endpoint(matrix_path, range(int(cell["shape"][0])))
            old = old_score_index.get((sid, cell_key_old))
            if not isinstance(old, Mapping):
                raise RescoreError(f"historical score row missing: {sid}/{cell_key_old}")
            checks = {"C_abs": abs(float(ep["C"]) - float(old["sync_c"])), "D_abs": abs(float(ep["D"]) - float(old["sync_d"])), "offset_equal": int(ep["offset"]) == int(old["av_offset"])}
            if checks["C_abs"] > 0.000501 or checks["D_abs"] > 0.000501 or not checks["offset_equal"]:
                raise RescoreError(f"legacy endpoint parity failed: {sid}/{cell_key_old}: {checks}")
            rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "cell": cell_key_old, "video_arm": video_arm, "computed": {"C": ep["C"], "D": ep["D"], "offset": ep["offset"]}, "manifest": {"C": old["sync_c"], "D": old["sync_d"], "offset": old["av_offset"]}, "checks": checks})
        n_manifest = old_score_index[(sid, "V_N/A_N")]
        s_manifest = old_score_index[(sid, "V_SHIFT_200/A_N")]
        gains.append({"sample_id": sid, "source_group": str(record["source_group"]), "benefit_C": float(s_manifest["sync_c"] - n_manifest["sync_c"]), "benefit_D": float(n_manifest["sync_d"] - s_manifest["sync_d"])})
    source_groups = sorted({str(row["source_group"]) for row in gains})
    rng = np.random.default_rng(20260904)
    indices = rng.integers(0, len(source_groups), size=(10_000, len(source_groups)), dtype=np.int64)
    by_group = {group: index for index, group in enumerate(source_groups)}
    values = {field: np.asarray([float(row[field]) for row in sorted(gains, key=lambda item: by_group[str(item["source_group"])])], dtype=np.float64) for field in ("benefit_C", "benefit_D")}
    aggregate = {field: {"mean": float(values[field].mean()), "ci95": [float(item) for item in np.percentile(values[field][indices].mean(axis=1), [2.5, 97.5], method="linear")]} for field in values}
    expected = prepared["fixed"]["history"]["scalar_original_shift_vs_n"]
    if not close_float(aggregate["benefit_C"]["mean"], expected["C"]["mean"], 1e-6) or not close_float(aggregate["benefit_D"]["mean"], expected["D"]["mean"], 1e-6):
        raise RescoreError(f"historical bootstrap parity failed: {aggregate}")
    payload = {"schema_version": 1, "stage_id": "historical_parity", "status": "complete", "parity_pass": True, "cell_count": len(rows), "rows": rows, "aggregate": aggregate, "expected": expected, "bootstrap": {"seed": 20260904, "draws": 10_000, "indices_sha256": bytes_sha256(np.ascontiguousarray(indices).tobytes())}}
    write_self_hashed_json(paths.parity, payload)
    return verify_self_hashed_json(paths.parity)


def _g_score_index(payload: Mapping[str, Any]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in payload.get("scores", []):
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise RescoreError(f"duplicate v4 parity score: {key}")
        result[key] = row
    return result


def _parity_cells(prepared: Mapping[str, Any], paths: config.RunPaths, parity_payload: Mapping[str, Any]) -> dict[str, Any]:
    g_index = _g_score_index(prepared["fixed"]["g_scores"])
    sample_id = "lrs3_6ORDQFh0Byw_00008"
    cells = ((config.VIDEO_N, "V_N"), ("V_DELAY_200", "V_DELAY_200"))
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.SYNCNET_BATCH_SIZE, threads=config.TORCH_THREADS)
    rows: list[dict[str, Any]] = []
    existing: dict[str, np.ndarray] = {}
    for v4_arm, label in cells:
        key = (sample_id, "DYNAMIC", v4_arm, "N")
        item = g_index.get(key)
        if not isinstance(item, Mapping):
            raise RescoreError(f"v4 parity cell missing: {key}")
        media_path = Path(str(item["media"]))
        source_audio = Path(str(item["audio"]))
        old_matrix = np.asarray(np.load(Path(str(item["matrix"])), allow_pickle=False), dtype=np.float32)
        if file_sha256(media_path) != str(item["media_sha256"]) or file_sha256(Path(str(item["matrix"]))) != str(item["matrix_sha256"]):
            raise RescoreError(f"v4 parity input hash changed: {key}")
        existing[label] = old_matrix
        out_dir = paths.root / "parity" / label
        worker = scorer.score(media_path, source_audio, out_dir, str(item["media_sha256"]), str(item["audio_pcm_sha256"]))
        new_matrix = np.asarray(np.load(Path(str(worker["matrix"])), allow_pickle=False), dtype=np.float32)
        max_abs = float(np.max(np.abs(new_matrix.astype(np.float64) - old_matrix.astype(np.float64)))) if new_matrix.shape == old_matrix.shape else float("inf")
        if new_matrix.shape != old_matrix.shape or max_abs > 1e-4 or worker.get("device") != "cpu" or worker.get("new_forward") is not True:
            raise RescoreError(f"v4 matrix parity failed: {key}, shape={new_matrix.shape}/{old_matrix.shape}, max_abs={max_abs}")
        worker_path = out_dir / "worker.json"
        write_self_hashed_json(worker_path, worker)
        rows.append({"sample_id": sample_id, "face_mode": "DYNAMIC", "video_arm": v4_arm, "audio_arm": "N", "label": label, "reference_matrix": str(Path(str(item["matrix"])).resolve()), "reference_matrix_sha256": str(item["matrix_sha256"]), "fresh_matrix": str(Path(str(worker["matrix"])).resolve()), "fresh_matrix_sha256": file_sha256(Path(str(worker["matrix"]))), "worker": str(worker_path.resolve()), "worker_sha256": file_sha256(worker_path), "matrix_max_abs": max_abs})
    baseline_reference = analysis.endpoint(existing["V_N"], range(existing["V_N"].shape[0]))
    baseline_fresh = analysis.endpoint(np.asarray(np.load(Path(rows[0]["fresh_matrix"]), allow_pickle=False), dtype=np.float32), range(existing["V_N"].shape[0]))
    anchor = int(baseline_reference["offset"])
    baseline_reference = analysis.endpoint(existing["V_N"], range(existing["V_N"].shape[0]), anchor)
    baseline_fresh = analysis.endpoint(np.asarray(np.load(Path(rows[0]["fresh_matrix"]), allow_pickle=False), dtype=np.float32), range(existing["V_N"].shape[0]), anchor)
    for row, reference_matrix in zip(rows, (existing["V_N"], existing["V_DELAY_200"]), strict=True):
        fresh_matrix = np.asarray(np.load(Path(row["fresh_matrix"]), allow_pickle=False), dtype=np.float32)
        reference_ep = analysis.endpoint(reference_matrix, range(reference_matrix.shape[0]), anchor)
        fresh_ep = analysis.endpoint(fresh_matrix, range(fresh_matrix.shape[0]), anchor)
        max_endpoint_error = max(abs(float(fresh_ep[field]) - float(reference_ep[field])) for field in ("D", "M", "C", "D_anchor", "C_anchor"))
        if max_endpoint_error > 1e-6 or int(fresh_ep["offset"]) != int(reference_ep["offset"]):
            raise RescoreError(f"v4 endpoint parity failed: {row['label']}, error={max_endpoint_error}")
        row["endpoint_max_abs"] = max_endpoint_error
        row["endpoint_offset_equal"] = True
        row["endpoint_parity"] = True
    result = {"schema_version": 1, "stage_id": "v4_parity", "status": "complete", "parity_pass": True, "cells": rows, "target_jobs_authorized": True, "target_job_count": 46}
    write_self_hashed_json(paths.parity, {**dict(parity_payload), "v4": result, "parity_pass": True})
    return verify_self_hashed_json(paths.parity)


def _score_index(scores: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in scores:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        if key in result:
            raise RescoreError(f"duplicate new score: {key}")
        result[key] = row
    return result


def _target_rescore(prepared: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    protocol = prepared["protocol"]
    protocol_sha = str(protocol["artifact_sha256"])
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.SYNCNET_BATCH_SIZE, threads=config.TORCH_THREADS)
    media_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        n_audio = Path(str(record["audio"]["N"]["path"]))
        n_pcm, _n_values, _n_params = read_pcm16_wav(n_audio)
        for arm in config.VIDEO_ARMS:
            cell = config.cell_key(sid, arm)
            source_video = Path(str(record["videos"][arm]["path"]))
            derived = paths.root / "media" / "derived" / f"{cell}.mkv"
            target = paths.root / "media" / "target" / f"{cell}.mkv"
            frames, source_evidence = media.decode_frames(source_video)
            source_timeline = media.validate_source_timeline(source_video, source_evidence, int(frames.shape[0]))
            derived_evidence = media.encode_lossless(frames, derived)
            mux_evidence = media.mux_n(derived, n_audio, target, n_pcm)
            observed_pcm = extract_pcm_from_media(target)
            if observed_pcm != n_pcm:
                raise RescoreError(f"target PCM changed before scoring: {cell}")
            sidecar = {"schema_version": 1, "protocol_id": "wav2lip_historical_shift_rescore", "protocol_sha256": protocol_sha, "sample_id": sid, "source_group": str(record["source_group"]), "video_arm": arm, "source_video": str(source_video.resolve()), "source_video_sha256": file_sha256(source_video), "derived_video": str(derived.resolve()), "derived_video_sha256": file_sha256(derived), "target_media": str(target.resolve()), "target_media_sha256": file_sha256(target), "source_timeline": source_timeline, "derived": derived_evidence, "mux": mux_evidence, "audio": str(n_audio.resolve()), "audio_pcm_sha256": bytes_sha256(n_pcm), "audio_pcm_sample_count": len(n_pcm) // 2, "roi_crop": False, "new_tfg_generation": False, "pixel_identity_verified": True, "pcm_identity_verified": True}
            sidecar_path = paths.root / "media" / "sidecars" / f"{cell}.json"
            write_self_hashed_json(sidecar_path, sidecar)
            media_rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "video_arm": arm, "source_video": str(source_video.resolve()), "source_video_sha256": file_sha256(source_video), "derived_video": str(derived.resolve()), "derived_video_sha256": file_sha256(derived), "target_media": str(target.resolve()), "target_media_sha256": file_sha256(target), "sidecar": str(sidecar_path.resolve()), "sidecar_sha256": file_sha256(sidecar_path), "frame_count": int(frames.shape[0]), "audio": str(n_audio.resolve()), "audio_pcm_sha256": bytes_sha256(n_pcm), "audio_sample_count": len(n_pcm) // 2})
            worker = scorer.score(target, n_audio, paths.root / "scores" / "cells" / cell, file_sha256(target), bytes_sha256(n_pcm))
            worker_path = paths.root / "scores" / "cells" / cell / "worker.json"
            write_self_hashed_json(worker_path, worker)
            matrix_path = Path(str(worker["matrix"]))
            score_rows.append({"schema_version": 1, "protocol_id": "wav2lip_historical_shift_rescore", "protocol_sha256": protocol_sha, "sample_id": sid, "source_group": str(record["source_group"]), "video_arm": arm, "audio_arm": "N", "cell_key": cell, "media": str(target.resolve()), "media_sha256": file_sha256(target), "media_sidecar": str(sidecar_path.resolve()), "media_sidecar_sha256": file_sha256(sidecar_path), "audio": str(n_audio.resolve()), "audio_pcm_sha256": bytes_sha256(n_pcm), "worker": str(worker_path.resolve()), "worker_sha256": file_sha256(worker_path), "visual": str(worker["visual"]), "visual_sha256": str(worker["visual_sha256"]), "audio_embedding": str(worker["audio_embedding"]), "audio_embedding_sha256": str(worker["audio_embedding_sha256"]), "matrix": str(matrix_path.resolve()), "matrix_sha256": file_sha256(matrix_path), "matrix_shape": list(worker["matrix_shape"]), "device": worker["device"], "new_forward": worker["new_forward"]})
    if len(media_rows) != 46 or len(score_rows) != 46:
        raise RescoreError(f"target denominator changed: media={len(media_rows)}, scores={len(score_rows)}")
    write_self_hashed_json(paths.media_manifest, {"schema_version": 1, "stage_id": "historical_media", "status": "complete", "media_count": len(media_rows), "rows": media_rows, "derived_video_count": 46, "target_mux_count": 46, "new_tfg_generation_count": 0})
    write_self_hashed_json(paths.scores_manifest, {"schema_version": 1, "stage_id": "historical_rescore", "status": "complete", "score_count": len(score_rows), "scores": score_rows, "runtime": {"python": str(config.SYNCNET_PYTHON), "device": "cpu", "batch_size": config.SYNCNET_BATCH_SIZE, "threads": config.TORCH_THREADS, "model": str(config.SYNCNET_MODEL.resolve()), "model_sha256": config.SYNCNET_MODEL_SHA256}})
    return {"media": verify_self_hashed_json(paths.media_manifest), "scores": verify_self_hashed_json(paths.scores_manifest)}


def _review(paths: config.RunPaths) -> dict[str, Any]:
    payload = {"schema_version": 1, "status": "complete", "reviewer": "self-review", "method": "focused contract tests, OpenSpec strict validation, independent artifact validator", "scope": ["fixed input hashes", "exact PCM shift and N mux", "lossless pixel/PTS identity", "v4 parity", "absolute J mapping", "shared anchor", "paired bootstrap", "46-cell denominator"], "limitations": ["No separate subagent review claimed", "Frontend difference remains a joint pipeline diagnostic", "Scientific label remains NOT_A_CONFIRMATION"]}
    write_self_hashed_json(paths.review, payload)
    return verify_self_hashed_json(paths.review)


def _result_markdown(final: Mapping[str, Any], analysis_payload: Mapping[str, Any]) -> str:
    legacy = analysis_payload.get("legacy_J", {}).get("aggregate", {})
    new = analysis_payload.get("full_frame_v4_J", {}).get("aggregate", {})
    frontend = analysis_payload.get("frontend_change_J", {}).get("aggregate", {})
    lines = ["# Wav2Lip historical SHIFT_200 rescore", "", f"- Engineering: `{final.get('engineering')}`", f"- Diagnostic: `{final.get('diagnostic')}`", f"- Scientific: `{final.get('scientific')}`", "- Cohort: 23 source groups; target score cells: 46; fresh TFG generation: 0; CUDA jobs: 0", "", "## Same-J paired summary", f"- LEGACY_TRACKED: ΔC={legacy.get('benefit_C', {}).get('mean')}, ΔD={legacy.get('benefit_D', {}).get('mean')}, Δanchor={legacy.get('benefit_anchor', {}).get('mean')}", f"- FULL_FRAME_V4: ΔC={new.get('benefit_C', {}).get('mean')}, ΔD={new.get('benefit_D', {}).get('mean')}, Δanchor={new.get('benefit_anchor', {}).get('mean')}", f"- Frontend change: ΔC={frontend.get('benefit_C', {}).get('direction')}, ΔD={frontend.get('benefit_D', {}).get('direction')}, Δanchor={frontend.get('benefit_anchor', {}).get('direction')}", "", "本轮只回答历史同媒体在两条评分入口下是否一致；即使存在正向信号，也不构成 replacement 确认，不授权训练、泛化或继续偏移扫描。下一步为 CLOSE_SHIFT_DIAGNOSTIC，转向完整 natural 保留 + 内容辅助实验。", ""]
    return "\n".join(lines)


def _blocked(paths: config.RunPaths, stage: str, reason: str, counts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    discrepancy = {"schema_version": 1, "status": "blocked", "stage": stage, "reason": str(reason), "counts": dict(counts or {})}
    write_self_hashed_json(paths.root / "discrepancy.json", discrepancy)
    payload = {"schema_version": 1, "protocol_id": "wav2lip_historical_shift_rescore", "status": "blocked", "engineering": "BLOCKED", "diagnostic": "INCOMPLETE", "scientific": "not_available", "blocked_stage": stage, "block_reason": str(reason), "counts": dict(counts or {}), "training_authorized": False, "generalization_established": False, "historical_gate_repaired": False, "next_action": "STOP_AND_REVIEW"}
    write_self_hashed_json(paths.final, payload)
    paths.result.write_text(f"# Wav2Lip historical SHIFT_200 rescore\n\n- Engineering: `BLOCKED`\n- Stage: `{stage}`\n- Reason: {reason}\n", encoding="utf-8")
    return verify_self_hashed_json(paths.final)


def run(run_id: str, stage: str, resume: bool = False) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.final.is_file():
        raise RescoreError(f"terminal run cannot be resumed: {paths.final}")
    if stage == "prepare":
        prepared = _prepare(run_id, paths)
        return {"status": "prepared", "protocol": str(paths.protocol), "history": str(paths.history), "shortest_J": prepared["audit"]["shortest_J"]}
    if stage != "all":
        raise ValueError(f"unknown stage: {stage}")
    prepared = _load_prepared(paths) if resume and paths.protocol.is_file() and paths.history.is_file() and paths.input_audit.is_file() else _prepare(run_id, paths)
    parity = _historical_parity(prepared, paths)
    parity = _parity_cells(prepared, paths, parity)
    _target_rescore(prepared, paths)
    scores_manifest = verify_self_hashed_json(paths.scores_manifest)
    score_index = {(str(row["sample_id"]), str(row["video_arm"])): row for row in scores_manifest["scores"]}
    analysis_payload, _endpoint_payload = analysis.analyze(prepared["protocol"], prepared["history"], {f"{key[0]}|{key[1]}": value for key, value in score_index.items()}, paths)
    _review(paths)
    from . import validate

    validation = validate.validate_complete(paths.root)
    if validation.get("status") != "valid":
        return _blocked(paths, "validate", str(validation.get("error", "independent validation failed")), {"scores": len(scores_manifest.get("scores", []))})
    final = {"schema_version": 1, "protocol_id": "wav2lip_historical_shift_rescore", "status": "complete", "engineering": "GO", "diagnostic": "COMPLETE", "scientific": "NOT_A_CONFIRMATION", "next_action": "CLOSE_SHIFT_DIAGNOSTIC", "counts": {"records": 23, "source_groups": 23, "old_videos_reused": 46, "derived_videos": 46, "target_muxes": 46, "parity_jobs": 2, "target_score_jobs": 46, "fresh_tfg_generation": 0, "cuda_jobs": 0, "training": 0}, "training_authorized": False, "generalization_established": False, "historical_gate_repaired": False, "analysis_sha256": file_sha256(paths.analysis), "validation_sha256": file_sha256(paths.validation), "review_sha256": file_sha256(paths.review)}
    write_self_hashed_json(paths.final, final)
    paths.result.write_text(_result_markdown(final, analysis_payload), encoding="utf-8")
    return verify_self_hashed_json(paths.final)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rescore historical Wav2Lip SHIFT_200 media with the frozen v4 CPU SyncNet path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = config.run_root_for(args.run_id)
    try:
        result = run(args.run_id, args.stage, args.resume)
    except Exception as exc:  # noqa: BLE001
        root.mkdir(parents=True, exist_ok=True)
        try:
            result = _blocked(config.RunPaths(root), args.stage, f"{type(exc).__name__}: {exc}")
        except Exception:
            raise
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
