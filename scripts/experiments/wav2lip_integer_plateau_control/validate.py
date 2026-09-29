from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import (
    PlateauError,
    bytes_sha256,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _matrix_from_row(row: dict[str, Any]) -> np.ndarray:
    path = Path(str(row["matrix"])).resolve()
    if file_sha256(path) != str(row["matrix_sha256"]):
        raise PlateauError(f"matrix hash changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 31:
        raise PlateauError(f"matrix shape invalid: {path}")
    return value


def _reconstruct_distance(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual = np.asarray(visual, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if visual.shape != audio.shape or visual.ndim != 2 or visual.shape[1] != config.EMBEDDING_DIM:
        raise PlateauError("embedding shapes are invalid")
    padded = np.pad(audio, ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    output = np.empty((visual.shape[0], 2 * config.VSHIFT + 1), dtype=np.float32)
    for row in range(visual.shape[0]):
        difference = visual[row : row + 1] - padded[row : row + 2 * config.VSHIFT + 1]
        output[row] = np.sqrt(np.sum((difference + np.float32(config.EPSILON)) ** 2, axis=1, dtype=np.float32), dtype=np.float32)
    return output


def _verify_worker(row: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    errors: list[str] = []
    worker_path = Path(str(row["worker_result"])).resolve()
    try:
        worker = verify_self_hashed_json(worker_path)
        visual_path = Path(str(worker["visual"])).resolve()
        audio_path = Path(str(worker["audio_embedding"])).resolve()
        for item, expected in ((visual_path, worker["visual_sha256"]), (audio_path, worker["audio_embedding_sha256"])):
            if file_sha256(item) != str(expected):
                errors.append(f"embedding hash mismatch: {item}")
        visual = np.asarray(np.load(visual_path, allow_pickle=False), dtype=np.float32)
        audio = np.asarray(np.load(audio_path, allow_pickle=False), dtype=np.float32)
        expected = _reconstruct_distance(visual, audio)
        matrix = _matrix_from_row(row)
        if expected.shape != matrix.shape or float(np.max(np.abs(expected - matrix))) > config.VALIDATOR_MATRIX_TOLERANCE:
            errors.append(f"independent distance reconstruction differs: {row['sample_id']}/{row['video_arm']}/{row['audio_arm']}")
        return matrix.astype(np.float64), errors
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"worker validation failed: {row.get('sample_id')}/{row.get('video_arm')}/{row.get('audio_arm')}: {exc}")
        return np.empty((0, 31), dtype=np.float64), errors


def _peak(curve: np.ndarray) -> dict[str, Any]:
    order = np.sort(curve, kind="stable")
    index = int(np.argmin(curve))
    offset = config.VSHIFT - index
    return {"offset": offset, "sync_d": float(order[0]), "sync_c": float(np.median(curve) - order[0]), "clear": bool(order[1] - order[0] > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT)}


def _summary(matrix: np.ndarray, rows: list[int]) -> dict[str, Any]:
    return _peak(np.mean(matrix[rows, :], axis=0, dtype=np.float64))


def _sig(matrix: np.ndarray, plus: list[int], minus: list[int]) -> dict[str, Any]:
    return {"PLUS": _summary(matrix, plus), "MINUS": _summary(matrix, minus), "common": _summary(matrix, plus + minus)}


def _bootstrap(values: list[float], groups: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        grouped[group].append(float(value))
    labels = sorted(grouped)
    means = {label: float(np.mean(grouped[label])) for label in labels}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        selected = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = np.mean([means[str(label)] for label in selected])
    return [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))]


def _independent_oracle(protocol: dict[str, Any], matrices: dict[tuple[str, str, str], np.ndarray]) -> dict[str, Any]:
    parity: list[float] = []
    baseline_count = 0
    timing = {"B": 0, "C": 0, "O": 0}
    own_c: list[float] = []
    own_d: list[float] = []
    damage_c: list[float] = []
    damage_d: list[float] = []
    own_offsets = 0
    damage_positive = 0
    groups: list[str] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        plus = [int(value) for value in record["plus_rows"]]
        minus = [int(value) for value in record["minus_rows"]]
        q_plus = [value + config.FRAME_SHIFT for value in plus]
        q_minus = [value - config.FRAME_SHIFT for value in minus]
        baseline = np.asarray(np.load(Path(str(record["parent_v_id_n"]["matrix"])), allow_pickle=False), dtype=np.float64)
        id_p = _sig(matrices[(sid, config.VIDEO_ID, config.AUDIO_P)], plus, minus)
        p_n = _sig(matrices[(sid, config.VIDEO_P, config.AUDIO_N)], plus, minus)
        p_p = _sig(matrices[(sid, config.VIDEO_P, config.AUDIO_P)], plus, minus)
        id_n = _sig(baseline, plus, minus)
        id_n_q = _sig(baseline, q_plus, q_minus)
        parity.append(float(np.max(np.abs(matrices[(sid, config.VIDEO_P, config.AUDIO_P)][plus + minus, :] - baseline[q_plus + q_minus, :]))))
        baseline_ok = all(id_n[name]["clear"] for name in ("PLUS", "MINUS")) and all(id_n_q[name]["clear"] for name in ("PLUS", "MINUS")) and abs(id_n["PLUS"]["offset"] - id_n["MINUS"]["offset"]) <= 1 and abs(id_n_q["PLUS"]["offset"] - id_n_q["MINUS"]["offset"]) <= 1
        baseline_count += int(baseline_ok)
        for name, left, right, expected in (("B", id_p, id_n, (5, -5)), ("C", p_n, id_n_q, (-5, 5)), ("O", p_p, id_n_q, (0, 0))):
            passed = True
            for segment, target in zip(("PLUS", "MINUS"), expected, strict=True):
                actual = left[segment]["offset"] - right[segment]["offset"]
                passed = passed and left[segment]["clear"] and right[segment]["clear"] and abs(actual - target) <= 1
            timing[name] += int(passed)
        own_c.append(p_p["common"]["sync_c"] - id_n_q["common"]["sync_c"])
        own_d.append(id_n_q["common"]["sync_d"] - p_p["common"]["sync_d"])
        own_offsets += int(abs(p_p["common"]["offset"] - id_n_q["common"]["offset"]) <= 1)
        damage_c.append(p_p["common"]["sync_c"] - p_n["common"]["sync_c"])
        damage_d.append(p_n["common"]["sync_d"] - p_p["common"]["sync_d"])
        damage_positive += int(p_p["common"]["sync_c"] > p_n["common"]["sync_c"] and p_n["common"]["sync_d"] > p_p["common"]["sync_d"])
        groups.append(str(record["source_group"]))
    own_ci = {"c": _bootstrap(own_c, groups), "d": _bootstrap(own_d, groups)}
    damage_ci = {"c": _bootstrap(damage_c, groups), "d": _bootstrap(damage_d, groups)}
    own_ok = own_ci["c"][0] > -0.10 and own_ci["d"][0] > -0.10 and own_offsets >= config.MIN_BASELINE_RECORDS
    damage_ok = damage_ci["c"][0] > 0.10 and damage_ci["d"][0] > 0.10 and damage_positive >= config.MIN_SUCCESS_RECORDS
    decision = "INTEGER_PLATEAU_CONTROL_SUPPORTED" if max(parity) <= config.MATRIX_DIFF_TOLERANCE and baseline_count >= 20 and min(timing.values()) >= 18 and own_ok and damage_ok else "ORACLE_PLATEAU_UNRESOLVED"
    return {"decision": decision, "max_parity": max(parity), "baseline_count": baseline_count, "timing_counts": timing, "own_ci": own_ci, "damage_ci": damage_ci, "own_offset_count": own_offsets, "damage_positive_count": damage_positive}


def validate_oracle(run_root: Path, *, final_sha256: str | None, write: bool) -> dict[str, Any]:
    paths = config.RunPaths(run_root.resolve())
    errors: list[str] = []
    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    try:
        protocol = _load(paths.protocol)
        audit = _load(paths.input_audit)
        frames_manifest = _load(paths.frames_manifest)
        audio_manifest = _load(paths.audio_manifest)
        media_manifest = _load(paths.media_manifest)
        scores = _load(paths.score_manifest)
        oracle_analysis = _load(paths.oracle_analysis)
        if protocol.get("protocol_id") != "wav2lip_integer_plateau_control" or protocol.get("status") != "frozen":
            errors.append("protocol identity/status invalid")
        if len(protocol.get("records", [])) != config.EXPECTED_RECORD_COUNT or len(scores.get("scores", [])) != 66:
            errors.append("Stage A count mismatch")
        if audit.get("record_count") != config.EXPECTED_RECORD_COUNT or len(frames_manifest.get("rows", [])) != 44 or len(media_manifest.get("rows", [])) != 22:
            errors.append("Stage A manifest count mismatch")
        score_index = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in scores.get("scores", [])}
        if len(score_index) != 66:
            errors.append("duplicate or missing Stage A score cells")
        frame_index = {(str(row["sample_id"]), str(row.get("video_arm"))): row for row in frames_manifest.get("rows", [])}
        audio_index = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
        media_index = {str(row["sample_id"]): row for row in media_manifest.get("rows", [])}
        for record in protocol.get("records", []):
            sid = str(record["sample_id"])
            parent_frames, _ = parent_media.extract_bgr24_frames(Path(str(record["parent_stream"]["path"])))
            p_frame_row = frame_index.get((sid, config.VIDEO_P))
            if not isinstance(p_frame_row, dict):
                errors.append(f"missing V_P frame row: {sid}")
                continue
            p_path = Path(str(p_frame_row["evidence"]["output"])).resolve()
            p_frames, _ = parent_media.extract_bgr24_frames(p_path)
            q = np.asarray(record["q"], dtype=np.int64)
            if p_frames.shape != parent_frames[q].shape or not np.array_equal(p_frames, parent_frames[q]):
                errors.append(f"V_P pixels differ from q-indexed source: {sid}")
            arow = audio_index.get(sid)
            if not isinstance(arow, dict):
                errors.append(f"missing audio row: {sid}")
            else:
                n_pcm, n_values, _ = read_pcm16_wav(Path(str(arow["arms"][config.AUDIO_N]["path"])))
                p_pcm, _p_values, _ = read_pcm16_wav(Path(str(arow["arms"][config.AUDIO_P]["path"])))
                source_indices = np.arange(n_values.size, dtype=np.int64) + np.where(np.arange(n_values.size) < int(record["break_sample"]), config.AUDIO_SHIFT, -config.AUDIO_SHIFT)
                expected = np.asarray(n_values, dtype=np.int16)[source_indices].astype("<i2", copy=False).tobytes()
                original_pcm, _values, _params = read_pcm16_wav(Path(str(record["natural_audio"]["path"])))
                if p_pcm != expected or n_pcm != original_pcm:
                    errors.append(f"N/P PCM identity or plateau mismatch: {sid}")
            mrow = media_index.get(sid)
            if not isinstance(mrow, dict):
                errors.append(f"missing media row: {sid}")
            for video, audio in config.FRESH_CELL_SPECS:
                srow = score_index.get((sid, video, audio))
                if not isinstance(srow, dict):
                    errors.append(f"missing score cell: {sid}/{video}/{audio}")
                    continue
                matrices[(sid, video, audio)], worker_errors = _verify_worker(srow)
                errors.extend(worker_errors)
                media_path = Path(str(srow["media"])).resolve()
                if file_sha256(media_path) != str(srow["media_sha256"]):
                    errors.append(f"media hash mismatch: {media_path}")
                try:
                    if bytes_sha256(extract_pcm_from_media(media_path)) != str(srow["media_pcm_sha256"]):
                        errors.append(f"media PCM mismatch: {media_path}")
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    errors.append(f"media decode failed: {media_path}: {exc}")
        if not errors:
            independent = _independent_oracle(protocol, matrices)
            if independent["decision"] != oracle_analysis.get("decision"):
                errors.append(f"independent decision differs: {independent['decision']} != {oracle_analysis.get('decision')}")
        else:
            independent = None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"validator exception: {type(exc).__name__}: {exc}")
        independent = None
    payload = {
        "schema_version": 1,
        "stage_id": "oracle_validation",
        "protocol_id": "wav2lip_integer_plateau_control",
        "status": "valid" if not errors else "invalid",
        "valid": not errors,
        "final_sha256": final_sha256,
        "independent_difference_count": len(errors),
        "new_generated_videos": config.EXPECTED_RECORD_COUNT if not errors else 0,
        "new_score_cells": 66 if not errors else 0,
        "independent_summary": independent,
        "errors": errors,
        "bridge_executed": False,
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
    }
    if write:
        write_self_hashed_json(paths.oracle_validation, payload)
    return payload


def validate_generated(run_root: Path, *, write: bool) -> dict[str, Any]:
    paths = config.RunPaths(run_root.resolve())
    errors: list[str] = []
    try:
        media_manifest = _load(paths.root / "generated_media_manifest.json")
        scores = _load(paths.root / "generated_score_manifest.json")
        analysis = _load(paths.generated_analysis)
        rows = scores.get("scores", [])
        if len(media_manifest.get("rows", [])) != config.EXPECTED_RECORD_COUNT or len(rows) != 44:
            errors.append("Stage B manifest count mismatch")
        if len({(str(row.get("sample_id")), str(row.get("audio_arm"))) for row in rows}) != 44:
            errors.append("Stage B score cell identity mismatch")
        for row in rows:
            _matrix, worker_errors = _verify_worker(row)
            errors.extend(worker_errors)
            media_path = Path(str(row["media"])).resolve()
            if file_sha256(media_path) != str(row["media_sha256"]):
                errors.append(f"generated media hash mismatch: {media_path}")
            if bytes_sha256(extract_pcm_from_media(media_path)) != str(row["media_pcm_sha256"]):
                errors.append(f"generated media PCM mismatch: {media_path}")
        if analysis.get("decision") not in {"INTEGER_PLATEAU_CONTROL_SUPPORTED", "GENERATED_PLATEAU_UNRESOLVED"}:
            errors.append("generated analysis has invalid decision")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"generated validator exception: {type(exc).__name__}: {exc}")
    payload = {"schema_version": 1, "stage_id": "generated_validation", "protocol_id": "wav2lip_integer_plateau_control", "status": "valid" if not errors else "invalid", "valid": not errors, "independent_difference_count": len(errors), "errors": errors, "bridge_executed": False, "training_authorized": False, "reference_conditioned_audio_head_spec_eligible": False, "generalization_established": False}
    if write:
        write_self_hashed_json(paths.generated_validation, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate the integer plateau control artifacts")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("oracle", "all"), default="all")
    args = parser.parse_args(argv)
    root = args.run_root if args.run_root.is_absolute() else (config.REPO / args.run_root).resolve()
    oracle = validate_oracle(root, final_sha256=None, write=True)
    result = {"oracle": oracle}
    if args.stage == "all" and oracle.get("valid"):
        result["generated"] = validate_generated(root, write=True)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if all(bool(item.get("valid")) for item in result.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
