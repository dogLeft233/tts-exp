from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config
from .common import ProtocolError, assert_finite, bytes_sha256, file_sha256, read_json, source_pcm16, verify_self_hashed_json, write_json_atomic, write_self_hashed_json
from .media import mux_and_verify
from .scoring import ScoreEngine, score_metrics


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ProtocolError(f"immutable asset changed or missing: {path}")


def _fixed_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for relative, expected in config.PARENT_FILES.items():
        _hash(config.PARENT / relative, expected)
    for relative, expected in config.DIAGNOSTIC_FILES.items():
        _hash(config.DIAGNOSTIC / relative, expected)
    for path in (config.PARENT_PROTOCOL, config.PARENT_DRIVERS, config.PARENT_SCORES, config.PARENT_CONTROL, config.PARENT_VALIDATION, config.PARENT_FINAL, config.DIAGNOSTIC_PROTOCOL, config.DIAGNOSTIC_ANALYSIS, config.DIAGNOSTIC_VALIDATION, config.DIAGNOSTIC_FINAL):
        _load(path)
    if file_sha256(config.ORIGINAL_DESIGN) != config.ORIGINAL_DESIGN_SHA256 or file_sha256(config.ORIGINAL_SPEC) != config.ORIGINAL_SPEC_SHA256:
        raise ProtocolError("original content-residual protocol binding changed")
    if not config.PARITY_MANIFEST.is_file() or file_sha256(config.PARITY_MANIFEST) != config.PARITY_MANIFEST_SHA256:
        raise ProtocolError("historical parity manifest binding changed")
    if not config.WAV2LIP_CHECKPOINT.is_file() or file_sha256(config.WAV2LIP_CHECKPOINT) != config.WAV2LIP_CHECKPOINT_SHA256:
        raise ProtocolError("Wav2Lip checkpoint binding changed")
    if not config.SYNCNET_MODEL.is_file() or file_sha256(config.SYNCNET_MODEL) != config.SYNCNET_MODEL_SHA256:
        raise ProtocolError("SyncNet model binding changed")
    parent_final = _load(config.PARENT_FINAL)
    diagnostic_final = _load(config.DIAGNOSTIC_FINAL)
    if parent_final.get("scientific_decision") != "CONTROL_FAILED" or parent_final.get("replacement_confirmed") is not False:
        raise ProtocolError("parent terminal state is not the frozen CONTROL_FAILED record")
    if diagnostic_final.get("stage_b_authorized") is not False or diagnostic_final.get("content_probe_revision_eligible") is not True:
        raise ProtocolError("diagnostic terminal state is not the frozen recovered-support record")
    return _load(config.PARENT_PROTOCOL), _load(config.PARENT_DRIVERS), _load(config.PARENT_SCORES)


def _records(parent_protocol: dict[str, Any], drivers: dict[str, Any]) -> list[dict[str, Any]]:
    records = parent_protocol.get("records")
    rows = drivers.get("rows")
    if not isinstance(records, list) or not isinstance(rows, list) or len(records) != config.EXPECTED_RECORD_COUNT or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent record/driver count is not 16")
    records = sorted(records, key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    rows = sorted(rows, key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if [str(row["sample_id"]) for row in records] != [str(row["sample_id"]) for row in rows]:
        raise ProtocolError("parent protocol and driver order/ids differ")
    if len({str(row["source_group"]) for row in records}) != config.EXPECTED_GROUP_COUNT:
        raise ProtocolError("parent source-group count is not 8")
    return records


def _input_audit(parent_protocol: dict[str, Any], drivers: dict[str, Any], parent_scores: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _records(parent_protocol, drivers)
    driver_by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    audit_rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        row = driver_by_id[sample_id]
        for arm in config.ARMS:
            item = row["arms"][arm]
            _hash(Path(str(item["path"])), str(item["sha256"]))
        static = row["static_face"]
        _hash(Path(str(static["path"])), str(static["sha256"]))
        audio = Path(str(row["natural_audio"]["path"]))
        _hash(audio, str(row["natural_audio"]["sha256"]))
        pcm = source_pcm16(audio)
        if len(pcm) < config.SUPPORT_SAMPLES * 2:
            raise ProtocolError(f"natural audio lacks frozen support: {sample_id}")
        audit_rows.append({"sample_id": sample_id, "source_group": str(row["source_group"]), "driver_manifest": "parent", "natural_audio_sha256": file_sha256(audio), "natural_pcm_sha256": bytes_sha256(pcm), "static_face_sha256": str(static["sha256"]), "arms": {arm: str(row["arms"][arm]["sha256"]) for arm in config.ARMS}, "passed": True})
    control_rows = [row for row in parent_scores.get("rows", []) if not str(row.get("video_arm", "")).startswith("PARITY_")]
    if len(parent_scores.get("rows", [])) != 36 or len(control_rows) != 34:
        raise ProtocolError("parent control score count is not 36/34 scientific")
    reuse_rows = [{"sample_id": str(row["sample_id"]), "video_arm": str(row["video_arm"]), "audio_arm": str(row["audio_arm"]), "source": str(config.PARENT_SCORES.resolve()), "source_sha256": config.PARENT_FILES["control_scores/manifest.json"], "reuse": "parent"} for row in parent_scores["rows"]]
    audit = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete", "record_count": len(audit_rows), "source_group_count": config.EXPECTED_GROUP_COUNT, "parent_control_rows": len(control_rows), "parent_total_score_rows": len(parent_scores["rows"]), "rows": audit_rows, "parent_terminal_state": "CONTROL_FAILED", "diagnostic_terminal_state": "SEARCH_SUPPORT_RECOVERED"}
    return audit, reuse_rows


def _prepare(paths: config.RunPaths, resume: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_protocol, parent_drivers, parent_scores = _fixed_parent()
    if paths.protocol.is_file():
        if not resume:
            raise ProtocolError(f"run already exists; use --resume: {paths.root}")
        protocol = _load(paths.protocol)
        drivers = _load(paths.drivers)
        return protocol, drivers, parent_scores
    audit, reuse_rows = _input_audit(parent_protocol, parent_drivers, parent_scores)
    spec_bindings = {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in config.spec_paths().items()}
    protocol = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "protocol_revision": config.PROTOCOL_REVISION, "run_id": paths.root.name.removeprefix(f"{config.RUN_PREFIX}_"), "status": "locked", "parent": {"root": str(config.PARENT.resolve()), "files": dict(config.PARENT_FILES)}, "diagnostic": {"root": str(config.DIAGNOSTIC.resolve()), "files": dict(config.DIAGNOSTIC_FILES)}, "spec_bindings": spec_bindings, "configuration": config.configuration(), "runtime": {"wav2lip_python": str(config.WAV2LIP_PYTHON), "syncnet_python": str(config.SYNCNET_PYTHON), "wav2lip_checkpoint": str(config.WAV2LIP_CHECKPOINT.resolve()), "wav2lip_checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256, "syncnet_model": str(config.SYNCNET_MODEL.resolve()), "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256}, "records": parent_protocol["records"], "record_count": config.EXPECTED_RECORD_COUNT, "source_group_count": config.EXPECTED_GROUP_COUNT, "coordinates": {"natural_lags": list(range(config.NATURAL_LAG_START, config.NATURAL_LAG_START + config.MATRIX_COLUMNS)), "delay_lags": list(range(config.DELAY_LAG_START, config.DELAY_LAG_START + config.MATRIX_COLUMNS)), "natural_offset": "15-j", "delay_offset": "10-j", "candidate_domain": "natural_only_-15..15", "fixed_anchor": "uncompensated_legacy_delay_curve_at_k_N"}, "limits": {"new_videos": config.MAX_NEW_VIDEO_COUNT, "new_scores": config.MAX_NEW_SCORE_COUNT, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    drivers = dict(parent_drivers)
    write_self_hashed_json(paths.input_audit, audit)
    write_self_hashed_json(paths.reuse_manifest, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "parent_run": str(config.PARENT.resolve()), "rows": reuse_rows, "reused_video_count": 18, "reused_score_count": 36, "fresh_video_budget": config.MAX_NEW_VIDEO_COUNT, "fresh_score_budget": config.MAX_NEW_SCORE_COUNT})
    write_self_hashed_json(paths.protocol, protocol)
    write_self_hashed_json(paths.drivers, drivers)
    return protocol, drivers, parent_scores


def _parent_control_rows(parent_scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in parent_scores["rows"] if str(row.get("video_arm")) in {"N", "N_REPEAT"} and str(row.get("audio_arm")) in {"N", "A_DELAY"}]
    if len(rows) != 34:
        raise ProtocolError(f"expected 34 cached scientific control rows, got {len(rows)}")
    return rows


def _matrix_from_score(row: dict[str, Any]) -> np.ndarray:
    score = row.get("score", row)
    path = Path(str(score["matrix"]))
    _hash(path, str(score["matrix_sha256"]))
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if matrix.shape != (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS) or not np.isfinite(matrix).all():
        raise ProtocolError(f"invalid cached matrix: {path}")
    return matrix


def _worker_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = row.get("score", row)
    worker_path = Path(str(score["worker"]))
    worker = verify_self_hashed_json(worker_path)
    arrays: list[np.ndarray] = []
    for key in ("visual", "audio_embedding", "matrix"):
        path = Path(str(worker[key]))
        _hash(path, str(worker[f"{key}_sha256"]))
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        expected = (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) if key != "matrix" else (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS)
        if value.shape != expected or not np.isfinite(value).all():
            raise ProtocolError(f"invalid cached {key}: {path}")
        arrays.append(value)
    return arrays[0], arrays[1], arrays[2]


def _legacy_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(audio, dtype=np.float32), ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    result = np.empty((config.EMBEDDING_ROWS, config.MATRIX_COLUMNS), dtype=np.float32)
    for row in range(config.EMBEDDING_ROWS):
        diff = np.asarray(visual[row : row + 1], dtype=np.float32) - padded[row : row + config.MATRIX_COLUMNS]
        result[row] = np.sqrt(np.sum(np.square(diff + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
    return result


def _matched_matrix(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    rows = np.asarray(config.U_ROWS, dtype=np.int64)
    lags = np.arange(lag_start, lag_start + config.MATRIX_COLUMNS, dtype=np.int64)
    result = np.empty((len(rows), config.MATRIX_COLUMNS), dtype=np.float32)
    for index, row in enumerate(rows):
        audio_rows = row + lags
        if audio_rows.min() < 0 or audio_rows.max() >= audio.shape[0]:
            raise ProtocolError(f"matched lag domain lacks real support at row {row}")
        diff = np.asarray(visual[row : row + 1], dtype=np.float32) - np.asarray(audio[audio_rows], dtype=np.float32)
        result[index] = np.sqrt(np.sum(np.square(diff + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
    return result


def _curve(matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(matrix, dtype=np.float64), axis=0, dtype=np.float64)


def _metrics(curve: np.ndarray, offset_base: int) -> dict[str, Any]:
    index = int(np.argmin(curve))
    return {"curve": [float(value) for value in curve], "min_index": index, "offset": int(offset_base - index), "D": float(curve[index]), "C": float(np.median(curve) - curve[index])}


def _bootstrap(values: Iterable[float], groups: Iterable[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != config.EXPECTED_GROUP_COUNT or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("bootstrap requires eight groups with two records")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    indices = np.random.default_rng(config.BOOTSTRAP_SEED).integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    estimates = np.mean(means[indices], axis=1, dtype=np.float64)
    return {"mean": float(np.mean(means)), "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0.0)), "draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "rng": "numpy_default_rng_pcg64", "quantile_method": "linear", "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}


def _reproduce_controls(protocol: dict[str, Any], parent_scores: dict[str, Any]) -> dict[str, Any]:
    rows = _parent_control_rows(parent_scores)
    by_key = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in rows}
    diagnostic = _load(config.DIAGNOSTIC_ANALYSIS)
    record_by_id = {str(row["sample_id"]): row for row in protocol["records"]}
    records: list[dict[str, Any]] = []
    old_failures: set[str] = set()
    matched_pass: set[str] = set()
    anchors: list[float] = []
    groups: list[str] = []
    for sample_id, record in record_by_id.items():
        n = by_key[(sample_id, "N", "N")]
        delay = by_key[(sample_id, "N", "A_DELAY")]
        visual_n, audio_n, cached_n = _worker_arrays(n)
        visual_d, audio_d, cached_d = _worker_arrays(delay)
        if not np.array_equal(visual_n, visual_d):
            raise ProtocolError(f"visual embedding mismatch in delayed pair: {sample_id}")
        rebuilt_n = _legacy_matrix(visual_n, audio_n)
        rebuilt_d = _legacy_matrix(visual_d, audio_d)
        if float(np.max(np.abs(rebuilt_n.astype(np.float64) - cached_n.astype(np.float64)))) > config.MATRIX_TOLERANCE or float(np.max(np.abs(rebuilt_d.astype(np.float64) - cached_d.astype(np.float64)))) > config.MATRIX_TOLERANCE:
            raise ProtocolError(f"cached legacy matrix cannot be independently rebuilt: {sample_id}")
        u_rows = np.asarray(config.U_ROWS, dtype=np.int64)
        legacy_n = _metrics(_curve(rebuilt_n[u_rows]), config.VSHIFT)
        legacy_d = _metrics(_curve(rebuilt_d[u_rows]), config.VSHIFT)
        matched_n_matrix = _matched_matrix(visual_n, audio_n, config.NATURAL_LAG_START)
        matched_d_matrix = _matched_matrix(visual_n, audio_d, config.DELAY_LAG_START)
        matched_n = _metrics(_curve(matched_n_matrix), config.VSHIFT)
        matched_d = _metrics(_curve(matched_d_matrix), 10)
        old_diff = int(legacy_d["offset"] - legacy_n["offset"])
        matched_diff = int(matched_d["offset"] - matched_n["offset"])
        anchor = float(_curve(rebuilt_d[u_rows])[legacy_n["min_index"]] - _curve(rebuilt_n[u_rows])[legacy_n["min_index"]])
        groups.append(str(record["source_group"]))
        anchors.append(anchor)
        if not (config.OFFSET_LOW <= old_diff <= config.OFFSET_HIGH):
            old_failures.add(sample_id)
        if config.OFFSET_LOW <= matched_diff <= config.OFFSET_HIGH:
            matched_pass.add(sample_id)
        records.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "legacy": {"natural": legacy_n, "delay": legacy_d, "offset_difference": old_diff, "expected_column": int(legacy_n["min_index"] + config.DELAY_FRAMES), "old_failure": sample_id in old_failures, "natural_matrix_max_abs": float(np.max(np.abs(rebuilt_n.astype(np.float64) - cached_n.astype(np.float64)))), "delay_matrix_max_abs": float(np.max(np.abs(rebuilt_d.astype(np.float64) - cached_d.astype(np.float64))))}, "matched": {"natural": matched_n, "delay": matched_d, "offset_difference": matched_diff, "offset_pass": sample_id in matched_pass, "same_array_index": matched_n["min_index"] == matched_d["min_index"]}, "anchor_damage": anchor})
    records.sort(key=lambda row: (row["source_group"], row["sample_id"]))
    anchor_result = _bootstrap(anchors, groups)
    expected_failures = {"lrs3_7JVTirBEfho_00039", "lrs3_7PwvGfs6Pok_00003", "lrs3_7c5t6FkvUG0_00001"}
    diagnostic_ok = bool(diagnostic.get("terminal_decision") == "SEARCH_SUPPORT_RECOVERED" and diagnostic.get("content_probe_revision_eligible") is True and diagnostic.get("matched_offset_pass_count") == 16 and set(diagnostic.get("recovered_ids", [])) == expected_failures)
    matched_ok = len(matched_pass) == config.EXPECTED_RECORD_COUNT and matched_pass == set(record_by_id)
    anchor_ok = bool(anchor_result["ci95"][0] > 0.0 and anchor_result["group_positive_count"] >= 7)
    result = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": "controls", "engineering_status": "GO", "old_offset_pass_count": config.EXPECTED_RECORD_COUNT - len(old_failures), "old_failure_ids": sorted(old_failures), "expected_old_failure_ids": sorted(expected_failures), "matched_offset_pass_count": len(matched_pass), "matched_pass_ids": sorted(matched_pass), "anchor_damage": anchor_result, "anchor_pass": anchor_ok, "diagnostic_reproduction": {"status": "PASS" if diagnostic_ok else "FAIL", "diagnostic_final": str(config.DIAGNOSTIC_FINAL.resolve()), "diagnostic_final_sha256": config.DIAGNOSTIC_FILES["final.json"], "terminal_decision": diagnostic.get("terminal_decision"), "content_probe_revision_eligible": diagnostic.get("content_probe_revision_eligible")}, "records": records, "counts": {"natural": 16, "delay": 16, "repeat": 2, "cached_scientific": 34, "parent_total": 36}, "gates": {"legacy_offset_pass": len(old_failures) == 3, "matched_offset_pass": matched_ok, "anchor_pass": anchor_ok, "diagnostic_reproduced": diagnostic_ok}, "stage_b_authorized": False, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}
    assert_finite(result)
    return result


def _run_gpu(plan_rows: list[dict[str, Any]], paths: config.RunPaths, label: str, resume: bool) -> dict[str, Any]:
    plan_path = paths.root / "plans" / f"{label}.json"
    result_path = paths.root / "generation" / f"{label}.json"
    payload = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "label": label, "rows": plan_rows}
    if plan_path.is_file():
        if read_json(plan_path) != payload:
            raise ProtocolError(f"existing GPU plan differs: {plan_path}")
    else:
        write_json_atomic(plan_path, payload)
    if result_path.is_file():
        if not resume:
            raise ProtocolError(f"GPU result already exists; use --resume: {result_path}")
        return read_json(result_path)
    log_path = paths.root / "logs" / f"{label}.gpu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO) + os.pathsep + env.get("PYTHONPATH", "")
    command = [str(config.WAV2LIP_PYTHON), "-m", "scripts.experiments.wav2lip_natural_content_residual_continuation.worker", "--plan", str(plan_path), "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG), "--result", str(result_path)]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(config.REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not result_path.is_file():
        raise ProtocolError(f"GPU generation failed; see {log_path}")
    result = read_json(result_path)
    result["command"] = command
    result["log"] = str(log_path.resolve())
    write_json_atomic(result_path, result)
    return result


def _replay_plan(paths: config.RunPaths, drivers: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(drivers["rows"], key=lambda row: (str(row["source_group"]), str(row["sample_id"])))[: config.EXPECTED_REPLAY_COUNT]
    return [{"sample_id": str(row["sample_id"]), "static_face": row["static_face"]["path"], "arms": {"N_REPLAY": row["arms"]["N"]["path"]}, "outputs": {"N_REPLAY": str((paths.root / "generation/replay" / f"{row['sample_id']}__N_REPLAY.mkv").resolve())}} for row in rows]


def _candidate_plan(paths: config.RunPaths, drivers: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"sample_id": str(row["sample_id"]), "static_face": row["static_face"]["path"], "arms": {arm: row["arms"][arm]["path"] for arm in config.CANDIDATE_ARMS}, "outputs": {arm: str((paths.root / "generation/candidates" / f"{row['sample_id']}__{arm}.mkv").resolve()) for arm in config.CANDIDATE_ARMS}} for row in sorted(drivers["rows"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))) ]


def _driver_by_id(drivers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in drivers["rows"]}


def _mux_replay(paths: config.RunPaths, drivers: dict[str, Any], generated: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    by_id = _driver_by_id(drivers)
    rows: list[dict[str, Any]] = []
    for generated_row in generated["rows"]:
        sample_id = str(generated_row["sample_id"])
        output = paths.media / "replay" / f"{sample_id}__N_REPLAY.mkv"
        audio = Path(str(by_id[sample_id]["natural_audio"]["path"]))
        if output.is_file() and resume:
            rows.append({"sample_id": sample_id, "arm": "N_REPLAY", "output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": generated_row["output"], "video_only_sha256": generated_row["output_sha256"], "audio": str(audio.resolve()), "audio_sha256": file_sha256(audio), "audio_pcm_sha256": bytes_sha256(source_pcm16(audio)), "frame_count": config.WAV2LIP_FRAMES, "audio_arm": "N", "reused": True})
        else:
            row = mux_and_verify(Path(str(generated_row["output"])), audio, output, sample_id=sample_id, arm="N_REPLAY")
            row["audio_arm"] = "N"
            row["reused"] = False
            rows.append(row)
    write_self_hashed_json(paths.root / "media/replay_manifest.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete", "rows": rows, "video_count": len(rows), "score_count": len(rows)})
    return rows


def _parent_row(parent_scores: dict[str, Any], sample_id: str, video_arm: str, audio_arm: str) -> dict[str, Any]:
    for row in parent_scores["rows"]:
        if str(row.get("sample_id")) == sample_id and str(row.get("video_arm")) == video_arm and str(row.get("audio_arm")) == audio_arm:
            return row
    raise ProtocolError(f"parent score row missing: {sample_id}/{video_arm}/{audio_arm}")


def _score_parity(paths: config.RunPaths, parent_scores: dict[str, Any], engine: ScoreEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent_row in [row for row in parent_scores["rows"] if str(row.get("video_arm", "")).startswith("PARITY_")]:
        label = str(parent_row["parity"]["label"])
        score = parent_row["score"]
        reference = Path(str(parent_row["parity"]["reference_matrix"]))
        actual = engine.score(Path(str(score["media"])), Path(str(score["source_audio"])), paths.scores / "parity" / label, sample_id=str(score["sample_id"]), video_arm=f"PARITY_{label}_CONTINUATION", audio_arm="N", reference_matrix=reference, expected_rows=None)
        matrix = np.asarray(np.load(actual["matrix"], allow_pickle=False), dtype=np.float64)
        reference_matrix = np.asarray(np.load(reference, allow_pickle=False), dtype=np.float64)
        matrix_error = float(np.max(np.abs(matrix - reference_matrix))) if matrix.shape == reference_matrix.shape else float("inf")
        actual_full = score_metrics(matrix)
        reference_full = score_metrics(reference_matrix)
        endpoint_error = max(abs(actual_full[name] - reference_full[name]) for name in ("C", "D"))
        passed = bool(matrix_error <= config.MATRIX_TOLERANCE and endpoint_error <= config.MATRIX_TOLERANCE and actual_full["offset"] == reference_full["offset"])
        rows.append({"sample_id": str(score["sample_id"]), "video_arm": f"PARITY_{label}_CONTINUATION", "audio_arm": "N", "parity": {"label": label, "reference_matrix": str(reference.resolve()), "reference_matrix_sha256": file_sha256(reference), "matrix_max_abs": matrix_error, "endpoint_max_abs": float(endpoint_error), "offset_equal": actual_full["offset"] == reference_full["offset"], "passes": passed}, "score": actual, "fresh": True})
    if len(rows) != config.EXPECTED_PARITY_COUNT:
        raise ProtocolError("registered parity cells are not exactly two")
    return rows


def _score_replay(paths: config.RunPaths, drivers: dict[str, Any], parent_scores: dict[str, Any], media_rows: list[dict[str, Any]], engine: ScoreEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for media_row in media_rows:
        sample_id = str(media_row["sample_id"])
        actual = engine.score(Path(str(media_row["output"])), Path(str(media_row["audio"])), paths.scores / "replay" / f"{sample_id}__N_REPLAY__N", sample_id=sample_id, video_arm="N_REPLAY", audio_arm="N")
        parent = _parent_row(parent_scores, sample_id, "N", "N")
        parent_score = parent["score"]
        parent_matrix = np.asarray(np.load(Path(str(parent_score["matrix"])), allow_pickle=False), dtype=np.float64)
        actual_matrix = np.asarray(np.load(Path(str(actual["matrix"])), allow_pickle=False), dtype=np.float64)
        matrix_error = float(np.max(np.abs(parent_matrix - actual_matrix))) if parent_matrix.shape == actual_matrix.shape else float("inf")
        metric_error = max(abs(float(actual["U"][name]) - float(parent_score["U"][name])) if name != "curve" else float(np.max(np.abs(np.asarray(actual["U"][name]) - np.asarray(parent_score["U"][name])))) for name in ("C", "D", "offset", "curve"))
        from scripts.experiments.lrs3_real_video_local_timing.media import decode_video_frames

        parent_frames = decode_video_frames(Path(str(parent_score["media"])))
        fresh_frames = decode_video_frames(Path(str(media_row["output"])))
        pixel_equal = len(parent_frames) == len(fresh_frames) and all(np.array_equal(a, b) for a, b in zip(parent_frames, fresh_frames, strict=True))
        passed = bool(pixel_equal and matrix_error <= config.MATRIX_TOLERANCE and metric_error <= config.MATRIX_TOLERANCE and actual["U"]["offset"] == parent_score["U"]["offset"] and actual["source_pcm_sha256"] == parent_score["source_pcm_sha256"])
        rows.append({"sample_id": sample_id, "video_arm": "N_REPLAY", "audio_arm": "N", "replay": {"parent_media": parent_score["media"], "parent_media_sha256": parent_score["media_sha256"], "pixel_equal": pixel_equal, "matrix_max_abs": matrix_error, "metric_max_abs": float(metric_error), "offset_equal": actual["U"]["offset"] == parent_score["U"]["offset"], "pcm_equal": actual["source_pcm_sha256"] == parent_score["source_pcm_sha256"], "passes": passed}, "score": actual, "fresh": True})
    if len(rows) != config.EXPECTED_REPLAY_COUNT:
        raise ProtocolError("N_REPLAY count is not two")
    return rows


def run_controls(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], parent_scores: dict[str, Any], resume: bool) -> dict[str, Any]:
    if paths.control_analysis.is_file() and paths.control_scores.is_file() and resume:
        return _load(paths.control_analysis)
    control = _reproduce_controls(protocol, parent_scores)
    engine = ScoreEngine()
    parity_rows = _score_parity(paths, parent_scores, engine)
    generated = _run_gpu(_replay_plan(paths, drivers), paths, "replay", resume)
    replay_media = _mux_replay(paths, drivers, generated, resume)
    replay_rows = _score_replay(paths, drivers, parent_scores, replay_media, engine)
    parity_pass = all(bool(row["parity"]["passes"]) for row in parity_rows)
    replay_pass = all(bool(row["replay"]["passes"]) for row in replay_rows)
    gates = dict(control["gates"])
    gates.update({"parity_pass": parity_pass, "replay_pass": replay_pass})
    control_pass = bool(all(gates.values()))
    control["gates"] = gates
    control["control_pass"] = control_pass
    control["scientific_decision"] = "CONTROL_PASS" if control_pass else "CONTROL_FAILED"
    control["stage_b_authorized"] = control_pass
    control["fresh_checks"] = {"parity": parity_rows, "replay": replay_rows}
    control["budget"] = {"reused_parent_videos": 18, "reused_parent_scores": 36, "new_videos": 2, "new_scores": 4, "new_video_limit": config.MAX_NEW_VIDEO_COUNT, "new_score_limit": config.MAX_NEW_SCORE_COUNT}
    write_self_hashed_json(paths.control_scores, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete", "rows": _parent_control_rows(parent_scores) + parity_rows + replay_rows, "counts": {"reused_scientific": 34, "fresh_parity": 2, "fresh_replay": 2, "parent_total": 36}})
    write_self_hashed_json(paths.control_analysis, control)
    print(f"CONTROLS {control['scientific_decision']} old={control['old_offset_pass_count']}/16 matched={control['matched_offset_pass_count']}/16", flush=True)
    return control


def _mux_candidates(paths: config.RunPaths, drivers: dict[str, Any], generated: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    driver_by_id = _driver_by_id(drivers)
    rows: list[dict[str, Any]] = []
    for generated_row in generated["rows"]:
        sample_id = str(generated_row["sample_id"])
        arm = str(generated_row["arm"])
        audio = Path(str(driver_by_id[sample_id]["natural_audio"]["path"]))
        output = paths.media / "candidates" / f"{sample_id}__{arm}.mkv"
        if output.is_file() and resume:
            rows.append({"sample_id": sample_id, "arm": arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": generated_row["output"], "video_only_sha256": generated_row["output_sha256"], "audio": str(audio.resolve()), "audio_sha256": file_sha256(audio), "audio_pcm_sha256": bytes_sha256(source_pcm16(audio)), "frame_count": config.WAV2LIP_FRAMES, "audio_arm": "N", "reused": True})
        else:
            row = mux_and_verify(Path(str(generated_row["output"])), audio, output, sample_id=sample_id, arm=arm)
            row["audio_arm"] = "N"
            row["reused"] = False
            rows.append(row)
    if len(rows) != config.EXPECTED_CANDIDATE_VIDEO_COUNT:
        raise ProtocolError(f"candidate media count is {len(rows)}")
    write_self_hashed_json(paths.root / "media/candidate_manifest.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete", "rows": rows, "video_count": len(rows), "score_count": len(rows)})
    return rows


def _score_candidates(paths: config.RunPaths, media_rows: list[dict[str, Any]], resume: bool) -> list[dict[str, Any]]:
    engine = ScoreEngine()
    rows: list[dict[str, Any]] = []
    for row in media_rows:
        sample_id = str(row["sample_id"])
        arm = str(row["arm"])
        score = engine.score(Path(str(row["output"])), Path(str(row["audio"])), paths.scores / "candidates" / f"{sample_id}__{arm}__N", sample_id=sample_id, video_arm=arm, audio_arm="N")
        rows.append({"sample_id": sample_id, "video_arm": arm, "audio_arm": "N", "score": score, "fresh": not bool(row.get("reused"))})
    write_self_hashed_json(paths.candidate_scores, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete", "rows": rows, "count": len(rows)})
    return rows


def run_candidates(paths: config.RunPaths, drivers: dict[str, Any], control: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    if not control.get("control_pass"):
        raise ProtocolError("candidate generation is blocked by controls")
    generated = _run_gpu(_candidate_plan(paths, drivers), paths, "candidates", resume)
    media_rows = _mux_candidates(paths, drivers, generated, resume)
    return _score_candidates(paths, media_rows, resume)


def _pass_contrast(result: dict[str, Any]) -> bool:
    return bool(all(float(result["metrics"][metric]["ci95"][0]) > 0.0 for metric in ("C", "D", "A")) and int(result["group_joint_positive_count"]) >= 7)


def _contrast(values: dict[str, list[float]], groups: list[str]) -> dict[str, Any]:
    metrics = {metric: _bootstrap(values[metric], groups) for metric in ("C", "D", "A")}
    by_group: dict[str, dict[str, float]] = {}
    for group in sorted(set(groups)):
        positions = [index for index, value in enumerate(groups) if value == group]
        by_group[group] = {metric: float(np.mean([values[metric][position] for position in positions])) for metric in ("C", "D", "A")}
    joint = sum(all(row[metric] > 0.0 for metric in ("C", "D", "A")) for row in by_group.values())
    return {"metrics": metrics, "group_joint_positive_count": int(joint), "group_count": len(by_group), "group_values": by_group}


def analyze_candidates(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if not control.get("control_pass"):
        raise ProtocolError("candidate analysis is unauthorized")
    candidate_manifest = _load(paths.candidate_scores)
    candidate_rows = candidate_manifest.get("rows", [])
    if len(candidate_rows) != config.EXPECTED_CANDIDATE_SCORE_COUNT:
        raise ProtocolError("candidate score count is not 48")
    control_rows = _parent_control_rows(_load(config.PARENT_SCORES))
    baseline = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in control_rows}
    candidate = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row["score"] for row in candidate_rows}
    if len(candidate) != 48:
        raise ProtocolError("candidate keys are not unique")
    values = {name: {metric: [] for metric in ("C", "D", "A")} for name in ("CORRECT_vs_N", "CORRECT_vs_WRONG", "CORRECT_vs_SHUFFLE")}
    groups: list[str] = []
    per_record: list[dict[str, Any]] = []
    driver_by_id = _driver_by_id(drivers)
    for record in sorted(protocol["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))):
        sample_id = str(record["sample_id"])
        groups.append(str(record["source_group"]))
        n = baseline[(sample_id, "N", "N")]["score"]
        c, w, s = (candidate[(sample_id, arm, "N")] for arm in config.CANDIDATE_ARMS)
        k = int(n["U"]["min_index"])
        correct_n = {"C": float(c["U"]["C"] - n["U"]["C"]), "D": float(n["U"]["D"] - c["U"]["D"]), "A": float(n["U"]["curve"][k] - c["U"]["curve"][k])}
        correct_wrong = {"C": float(c["U"]["C"] - w["U"]["C"]), "D": float(w["U"]["D"] - c["U"]["D"]), "A": float(w["U"]["curve"][k] - c["U"]["curve"][k])}
        correct_shuffle = {"C": float(c["U"]["C"] - s["U"]["C"]), "D": float(s["U"]["D"] - c["U"]["D"]), "A": float(s["U"]["curve"][k] - c["U"]["curve"][k])}
        for name, row in (("CORRECT_vs_N", correct_n), ("CORRECT_vs_WRONG", correct_wrong), ("CORRECT_vs_SHUFFLE", correct_shuffle)):
            for metric in ("C", "D", "A"):
                values[name][metric].append(row[metric])
        per_record.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "k_N": k, "correct_vs_n": correct_n, "correct_vs_wrong": correct_wrong, "correct_vs_shuffle": correct_shuffle, "norm_control_valid": bool(driver_by_id[sample_id]["construct"].get("norm_control_valid"))})
    contrasts = {name: _contrast(values[name], groups) for name in values}
    norm_valid = all(row["norm_control_valid"] for row in per_record)
    primary_pass = bool(_pass_contrast(contrasts["CORRECT_vs_N"]) and contrasts["CORRECT_vs_N"]["metrics"]["C"]["mean"] > 0.05)
    content_signal = bool(primary_pass and norm_valid and _pass_contrast(contrasts["CORRECT_vs_WRONG"]) and _pass_contrast(contrasts["CORRECT_vs_SHUFFLE"]))
    decision = "CONTENT_RESIDUAL_SIGNAL" if content_signal else "NATURAL_GAIN_MECHANISM_UNRESOLVED" if primary_pass else "NO_INCREMENT_ESTABLISHED"
    result = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": "candidates", "scientific_decision": decision, "primary_pass": primary_pass, "content_signal": content_signal, "norm_control_valid": norm_valid, "contrasts": contrasts, "per_record": per_record, "limits": ["seen records only", "3.84-second mel support", "exploratory, not independent confirmation", "replacement_confirmed=false", "waveform_head_authorized=false", "generalization_established=false", "historical_shift_gate_repaired=false"]}
    assert_finite(result)
    write_self_hashed_json(paths.analysis, result)
    return result


def _run_validator(paths: config.RunPaths, stage: str) -> tuple[int, Path]:
    output = paths.control_validation if stage == "controls" else paths.validation
    log = paths.root / "logs" / f"validate_{stage}.log"
    command = [str(config.SYNCNET_PYTHON), "-m", "scripts.experiments.wav2lip_natural_content_residual_continuation.validate", "--run-root", str(paths.root), "--stage", stage]
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(config.REPO), env={**os.environ, "PYTHONPATH": str(config.REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return completed.returncode, output


def _finalize(paths: config.RunPaths, control: dict[str, Any], analysis: dict[str, Any] | None, validation: dict[str, Any] | None, engineering_status: str) -> None:
    scientific = str(analysis.get("scientific_decision")) if analysis is not None else str(control.get("scientific_decision", "CONTROL_FAILED"))
    final = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "complete" if engineering_status == "GO" else "blocked", "engineering_status": engineering_status, "scientific_decision": scientific, "control_decision": control.get("scientific_decision"), "analysis": str(paths.analysis.resolve()) if analysis is not None else None, "analysis_sha256": file_sha256(paths.analysis) if paths.analysis.is_file() else None, "validation": str(paths.validation.resolve()) if validation is not None and paths.validation.is_file() else None, "validation_sha256": file_sha256(paths.validation) if validation is not None and paths.validation.is_file() else None, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "budget": {"new_videos": 50, "new_scores": 52, "reused_parent_videos": 18, "reused_parent_scores": 36}}
    write_self_hashed_json(paths.final, final)
    lines = ["# Wav2Lip natural content residual continuation", "", f"- engineering_status: `{engineering_status}`", f"- scientific_decision: `{scientific}`", f"- control: `{control.get('scientific_decision')}`", f"- fresh budget: `50 videos / 52 scores` maximum", "- replacement_confirmed: `false`", "- waveform_head_authorized: `false`", "- generalization_established: `false`", ""]
    if analysis is not None:
        lines.extend(["## Main contrasts", "", "| Contrast | ΔC mean [95% CI] | ΔD mean [95% CI] | ΔA mean [95% CI] |", "|---|---:|---:|---:|"])
        for name, result in analysis["contrasts"].items():
            metrics = result["metrics"]
            lines.append(f"| {name} | {metrics['C']['mean']:.6f} [{metrics['C']['ci95'][0]:.6f}, {metrics['C']['ci95'][1]:.6f}] | {metrics['D']['mean']:.6f} [{metrics['D']['ci95'][0]:.6f}, {metrics['D']['ci95'][1]:.6f}] | {metrics['A']['mean']:.6f} [{metrics['A']['ci95'][0]:.6f}, {metrics['A']['ci95'][1]:.6f}] |")
    paths.result.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "review_type": "self-review", "self_review": True, "status": engineering_status, "checks": ["parent and diagnostic hashes rechecked", "translated delay domain is control-only", "candidate audio remains complete natural PCM", "independent validator invoked", "all confirmation flags remain false"], "remaining_limits": ["seen records", "short mel support", "not independent confirmation", "WRONG and SHUFFLE are specificity controls, not semantic proof"]}
    write_self_hashed_json(paths.review, review)


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = _paths(run_id)
    if paths.root.exists() and any(paths.root.iterdir()) and not resume and not paths.protocol.is_file():
        raise ProtocolError(f"refusing non-empty output directory: {paths.root}")
    protocol, drivers, parent_scores = _prepare(paths, resume)
    if stage == "prepare":
        return 0
    control = run_controls(paths, protocol, drivers, parent_scores, resume)
    control_code, control_path = _run_validator(paths, "controls")
    control_validation = _load(control_path) if control_path.is_file() else None
    if control_code != 0 or not control_validation or control_validation.get("status") != "PASS":
        _finalize(paths, control, None, None, "BLOCKED")
        return 1
    if stage == "controls" or not control.get("control_pass"):
        _finalize(paths, control, None, control_validation, "GO")
        return 0
    if stage == "candidates" or stage == "all":
        run_candidates(paths, drivers, control, resume)
    if stage == "candidates":
        return 0
    analysis = analyze_candidates(paths, protocol, drivers, control)
    validation_code, validation_path = _run_validator(paths, "all")
    validation = _load(validation_path) if validation_path.is_file() else None
    engineering = "GO" if validation_code == 0 and validation and validation.get("status") == "PASS" else "BLOCKED"
    _finalize(paths, control, analysis, validation, engineering)
    return 0 if engineering == "GO" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "controls", "candidates", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.run_id, args.stage, args.resume)
    except Exception as exc:
        root = config.run_root_for(args.run_id)
        root.mkdir(parents=True, exist_ok=True)
        error = write_self_hashed_json(root / "error.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        write_self_hashed_json(root / "final.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "engineering_status": "BLOCKED", "scientific_decision": "BLOCKED", "error_sha256": file_sha256(root / "error.json"), "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})
        print(json.dumps({"run_root": str(root), "error": error}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
