from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import decode_pcm16, decode_video_frames

from . import config
from .common import ProtocolError, bytes_sha256, file_sha256, source_pcm16, verify_self_hashed_json, write_self_hashed_json
from .scoring import score_metrics


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ProtocolError(f"hash mismatch: {path}")


def _fixed_assets() -> None:
    for relative, expected in config.PARENT_FILES.items():
        _hash(config.PARENT / relative, expected)
    for relative, expected in config.DIAGNOSTIC_FILES.items():
        _hash(config.DIAGNOSTIC / relative, expected)
    _hash(config.ORIGINAL_DESIGN, config.ORIGINAL_DESIGN_SHA256)
    _hash(config.ORIGINAL_SPEC, config.ORIGINAL_SPEC_SHA256)
    _hash(config.PARITY_MANIFEST, config.PARITY_MANIFEST_SHA256)


def _matrix(row: dict[str, Any]) -> np.ndarray:
    score = row.get("score", row)
    path = Path(str(score["matrix"]))
    _hash(path, str(score["matrix_sha256"]))
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.shape != (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS) or not np.isfinite(value).all():
        raise ProtocolError(f"invalid score matrix: {path}")
    return value


def _worker_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = row.get("score", row)
    worker = _load(Path(str(score["worker"])))
    result: list[np.ndarray] = []
    for key in ("visual", "audio_embedding", "matrix"):
        path = Path(str(worker[key]))
        _hash(path, str(worker[f"{key}_sha256"]))
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        expected = (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) if key != "matrix" else (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS)
        if value.shape != expected or not np.isfinite(value).all():
            raise ProtocolError(f"invalid worker {key}: {path}")
        result.append(value)
    return result[0], result[1], result[2]


def _legacy(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(audio, dtype=np.float32), ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    value = np.empty((config.EMBEDDING_ROWS, config.MATRIX_COLUMNS), dtype=np.float32)
    for row in range(config.EMBEDDING_ROWS):
        diff = np.asarray(visual[row : row + 1], dtype=np.float32) - padded[row : row + config.MATRIX_COLUMNS]
        value[row] = np.sqrt(np.sum(np.square(diff + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
    return value


def _matched(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    rows = np.asarray(config.U_ROWS, dtype=np.int64)
    lags = np.arange(lag_start, lag_start + config.MATRIX_COLUMNS, dtype=np.int64)
    value = np.empty((len(rows), config.MATRIX_COLUMNS), dtype=np.float32)
    for index, row in enumerate(rows):
        positions = row + lags
        if positions.min() < 0 or positions.max() >= audio.shape[0]:
            raise ProtocolError("matched support is not real")
        diff = np.asarray(visual[row : row + 1], dtype=np.float32) - np.asarray(audio[positions], dtype=np.float32)
        value[index] = np.sqrt(np.sum(np.square(diff + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
    return value


def _curve(matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(matrix, dtype=np.float64), axis=0, dtype=np.float64)


def _metric(curve: np.ndarray, offset_base: int) -> dict[str, Any]:
    index = int(np.argmin(curve))
    return {"curve": [float(value) for value in curve], "min_index": index, "offset": int(offset_base - index), "C": float(np.median(curve) - curve[index]), "D": float(curve[index])}


def _bootstrap(values: Iterable[float], groups: Iterable[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != config.EXPECTED_GROUP_COUNT or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("invalid source-group bootstrap input")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    indices = np.random.default_rng(config.BOOTSTRAP_SEED).integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    samples = np.mean(means[indices], axis=1, dtype=np.float64)
    return {"mean": float(np.mean(means)), "ci95": [float(np.quantile(samples, 0.025, method="linear")), float(np.quantile(samples, 0.975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0.0)), "draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "rng": "numpy_default_rng_pcg64", "quantile_method": "linear", "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}


def _parent_rows() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = _load(config.PARENT_PROTOCOL)
    drivers = _load(config.PARENT_DRIVERS)
    scores = _load(config.PARENT_SCORES)
    records = sorted(protocol["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len(records) != config.EXPECTED_RECORD_COUNT or len(drivers.get("rows", [])) != config.EXPECTED_RECORD_COUNT or len(scores.get("rows", [])) != 36:
        raise ProtocolError("frozen parent counts changed")
    scientific = [row for row in scores["rows"] if str(row.get("video_arm")) in {"N", "N_REPEAT"} and str(row.get("audio_arm")) in {"N", "A_DELAY"}]
    if len(scientific) != 34:
        raise ProtocolError("frozen scientific control count changed")
    return protocol, drivers, scores, scientific


def _recompute_control(protocol: dict[str, Any], scientific: list[dict[str, Any]], diagnostic: dict[str, Any]) -> dict[str, Any]:
    by_key = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in scientific}
    expected_ids = {str(row["sample_id"]) for row in protocol["records"]}
    records: list[dict[str, Any]] = []
    old_failures: set[str] = set()
    matched_pass: set[str] = set()
    anchors: list[float] = []
    groups: list[str] = []
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        n = by_key[(sample_id, "N", "N")]
        d = by_key[(sample_id, "N", "A_DELAY")]
        vn, an, mn = _worker_arrays(n)
        vd, ad, md = _worker_arrays(d)
        if not np.array_equal(vn, vd):
            raise ProtocolError(f"visual pair mismatch: {sample_id}")
        rebuilt_n = _legacy(vn, an)
        rebuilt_d = _legacy(vd, ad)
        if np.max(np.abs(rebuilt_n.astype(np.float64) - mn.astype(np.float64))) > config.MATRIX_TOLERANCE or np.max(np.abs(rebuilt_d.astype(np.float64) - md.astype(np.float64))) > config.MATRIX_TOLERANCE:
            raise ProtocolError(f"legacy matrix reproduction mismatch: {sample_id}")
        u_rows = np.asarray(config.U_ROWS, dtype=np.int64)
        curve_n = _curve(rebuilt_n[u_rows])
        curve_d = _curve(rebuilt_d[u_rows])
        n_legacy = _metric(curve_n, config.VSHIFT)
        d_legacy = _metric(curve_d, config.VSHIFT)
        n_matched = _metric(_curve(_matched(vn, an, config.NATURAL_LAG_START)), config.VSHIFT)
        d_matched = _metric(_curve(_matched(vn, ad, config.DELAY_LAG_START)), 10)
        old_diff = int(d_legacy["offset"] - n_legacy["offset"])
        matched_diff = int(d_matched["offset"] - n_matched["offset"])
        anchor = float(curve_d[n_legacy["min_index"]] - curve_n[n_legacy["min_index"]])
        if not config.OFFSET_LOW <= old_diff <= config.OFFSET_HIGH:
            old_failures.add(sample_id)
        if config.OFFSET_LOW <= matched_diff <= config.OFFSET_HIGH:
            matched_pass.add(sample_id)
        groups.append(group)
        anchors.append(anchor)
        records.append({"sample_id": sample_id, "source_group": group, "legacy": {"natural": n_legacy, "delay": d_legacy, "offset_difference": old_diff, "expected_column": n_legacy["min_index"] + config.DELAY_FRAMES, "old_failure": not config.OFFSET_LOW <= old_diff <= config.OFFSET_HIGH}, "matched": {"natural": n_matched, "delay": d_matched, "offset_difference": matched_diff, "offset_pass": config.OFFSET_LOW <= matched_diff <= config.OFFSET_HIGH, "same_array_index": n_matched["min_index"] == d_matched["min_index"]}, "anchor_damage": anchor})
    records.sort(key=lambda row: (row["source_group"], row["sample_id"]))
    expected_failures = {"lrs3_7JVTirBEfho_00039", "lrs3_7PwvGfs6Pok_00003", "lrs3_7c5t6FkvUG0_00001"}
    anchor_result = _bootstrap(anchors, groups)
    diagnostic_ok = bool(diagnostic.get("terminal_decision") == "SEARCH_SUPPORT_RECOVERED" and diagnostic.get("content_probe_revision_eligible") is True and diagnostic.get("matched_offset_pass_count") == 16 and set(diagnostic.get("recovered_ids", [])) == expected_failures)
    return {"records": records, "old_failure_ids": sorted(old_failures), "matched_pass_ids": sorted(matched_pass), "old_offset_pass_count": config.EXPECTED_RECORD_COUNT - len(old_failures), "matched_offset_pass_count": len(matched_pass), "anchor_damage": anchor_result, "anchor_pass": anchor_result["ci95"][0] > 0.0 and anchor_result["group_positive_count"] >= 7, "diagnostic_ok": diagnostic_ok, "expected_ids": expected_ids, "expected_failures": expected_failures}


def _check_media(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    media = Path(str(row["output"]))
    audio = Path(str(row["audio"]))
    try:
        frames = decode_video_frames(media)
        pcm = decode_pcm16(media)
        expected = source_pcm16(audio)
    except Exception as exc:  # noqa: BLE001 - report artifact failure
        return [f"media_decode:{row.get('sample_id')}:{exc}"]
    if len(frames) != config.WAV2LIP_FRAMES or any(frame.shape != (224, 224, 3) or frame.dtype != np.uint8 for frame in frames):
        errors.append(f"media_timeline:{row.get('sample_id')}:{row.get('arm')}")
    if pcm != expected:
        errors.append(f"media_pcm:{row.get('sample_id')}:{row.get('arm')}")
    if str(row.get("output_sha256")) != file_sha256(media):
        errors.append(f"media_hash:{row.get('sample_id')}:{row.get('arm')}")
    return errors


def _compare(expected: Any, actual: Any, tolerance: float = config.METRIC_TOLERANCE) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(_compare(expected[key], actual[key], tolerance) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_compare(left, right, tolerance) for left, right in zip(expected, actual, strict=True))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= tolerance
    return expected == actual


def _validate_fresh_controls(run_root: Path, control_manifest: dict[str, Any], parent_scores: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = control_manifest.get("rows", [])
    if len(rows) != 38:
        errors.append("control_manifest_count")
    reused = [row for row in rows if str(row.get("video_arm")) not in {"N_REPLAY"} and not str(row.get("video_arm", "")).startswith("PARITY_")]
    parent_scientific = [row for row in parent_scores["rows"] if str(row.get("video_arm")) in {"N", "N_REPEAT"} and str(row.get("audio_arm")) in {"N", "A_DELAY"}]
    if reused != parent_scientific:
        errors.append("reused_control_rows_changed")
    parity = [row for row in rows if str(row.get("video_arm", "")).startswith("PARITY_")]
    replay = [row for row in rows if str(row.get("video_arm")) == "N_REPLAY"]
    if len(parity) != 2 or len(replay) != 2:
        errors.append("fresh_control_counts")
    for row in parity:
        score = row.get("score", {})
        try:
            matrix = np.asarray(np.load(Path(str(score["matrix"])), allow_pickle=False), dtype=np.float64)
            reference = np.asarray(np.load(Path(str(row["parity"]["reference_matrix"])), allow_pickle=False), dtype=np.float64)
            if matrix.shape != reference.shape or np.max(np.abs(matrix - reference)) > config.MATRIX_TOLERANCE or not row["parity"]["passes"]:
                errors.append(f"parity:{row.get('sample_id')}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"parity_exception:{exc}")
    for row in replay:
        if not row.get("replay", {}).get("passes"):
            errors.append(f"replay_flag:{row.get('sample_id')}")
        errors.extend(_check_media({"sample_id": row["sample_id"], "arm": "N_REPLAY", "output": row["score"]["media"], "output_sha256": row["score"]["media_sha256"], "audio": row["score"]["source_audio"]}))
    return errors


def _recompute_analysis(protocol: dict[str, Any], drivers: dict[str, Any], control_scores: list[dict[str, Any]], candidate_scores: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in control_scores if str(row.get("video_arm")) in {"N", "N_REPEAT"} and str(row.get("audio_arm")) in {"N", "A_DELAY"}}
    candidate = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row["score"] for row in candidate_scores}
    if len(candidate) != 48:
        raise ProtocolError("candidate keys are not 48")
    values = {name: {metric: [] for metric in ("C", "D", "A")} for name in ("CORRECT_vs_N", "CORRECT_vs_WRONG", "CORRECT_vs_SHUFFLE")}
    groups: list[str] = []
    per_record: list[dict[str, Any]] = []
    driver_by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    for record in sorted(protocol["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))):
        sid = str(record["sample_id"])
        groups.append(str(record["source_group"]))
        n = baseline[(sid, "N", "N")]["score"]
        c, w, s = (candidate[(sid, arm, "N")] for arm in config.CANDIDATE_ARMS)
        k = int(n["U"]["min_index"])
        rows = {
            "CORRECT_vs_N": {"C": c["U"]["C"] - n["U"]["C"], "D": n["U"]["D"] - c["U"]["D"], "A": n["U"]["curve"][k] - c["U"]["curve"][k]},
            "CORRECT_vs_WRONG": {"C": c["U"]["C"] - w["U"]["C"], "D": w["U"]["D"] - c["U"]["D"], "A": w["U"]["curve"][k] - c["U"]["curve"][k]},
            "CORRECT_vs_SHUFFLE": {"C": c["U"]["C"] - s["U"]["C"], "D": s["U"]["D"] - c["U"]["D"], "A": s["U"]["curve"][k] - c["U"]["curve"][k]},
        }
        for name in rows:
            for metric in rows[name]: values[name][metric].append(float(rows[name][metric]))
        per_record.append({"sample_id": sid, "source_group": str(record["source_group"]), "k_N": k, "correct_vs_n": rows["CORRECT_vs_N"], "correct_vs_wrong": rows["CORRECT_vs_WRONG"], "correct_vs_shuffle": rows["CORRECT_vs_SHUFFLE"], "norm_control_valid": bool(driver_by_id[sid]["construct"].get("norm_control_valid"))})
    contrasts: dict[str, Any] = {}
    for name in values:
        metrics = {metric: _bootstrap(values[name][metric], groups) for metric in ("C", "D", "A")}
        group_values: dict[str, dict[str, float]] = {}
        for group in sorted(set(groups)):
            positions = [index for index, value in enumerate(groups) if value == group]
            group_values[group] = {metric: float(np.mean([values[name][metric][position] for position in positions])) for metric in ("C", "D", "A")}
        contrasts[name] = {"metrics": metrics, "group_joint_positive_count": sum(all(value[metric] > 0 for metric in ("C", "D", "A")) for value in group_values.values()), "group_count": len(group_values), "group_values": group_values}
    norm_valid = all(row["norm_control_valid"] for row in per_record)
    primary_pass = bool(all(float(contrasts["CORRECT_vs_N"]["metrics"][metric]["ci95"][0]) > 0 for metric in ("C", "D", "A")) and contrasts["CORRECT_vs_N"]["group_joint_positive_count"] >= 7 and contrasts["CORRECT_vs_N"]["metrics"]["C"]["mean"] > 0.05)
    content_signal = bool(primary_pass and norm_valid and all(all(float(contrasts[name]["metrics"][metric]["ci95"][0]) > 0 for metric in ("C", "D", "A")) and contrasts[name]["group_joint_positive_count"] >= 7 for name in ("CORRECT_vs_WRONG", "CORRECT_vs_SHUFFLE")))
    decision = "CONTENT_RESIDUAL_SIGNAL" if content_signal else "NATURAL_GAIN_MECHANISM_UNRESOLVED" if primary_pass else "NO_INCREMENT_ESTABLISHED"
    return {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": "candidates", "scientific_decision": decision, "primary_pass": primary_pass, "content_signal": content_signal, "norm_control_valid": norm_valid, "contrasts": contrasts, "per_record": per_record, "limits": ["seen records only", "3.84-second mel support", "exploratory, not independent confirmation", "replacement_confirmed=false", "waveform_head_authorized=false", "generalization_established=false", "historical_shift_gate_repaired=false"]}


def validate(run_root: Path, stage: str) -> dict[str, Any]:
    errors: list[str] = []
    _fixed_assets()
    protocol = _load(run_root / "protocol.json")
    drivers = _load(run_root / "drivers/manifest.json")
    parent_protocol, parent_drivers, parent_scores, scientific = _parent_rows()
    if protocol.get("protocol_revision") != config.PROTOCOL_REVISION or protocol.get("record_count") != 16 or protocol.get("source_group_count") != 8:
        errors.append("protocol_binding")
    if drivers != parent_drivers:
        errors.append("driver_manifest_not_exact_parent")
    input_audit = _load(run_root / "input_audit.json")
    if input_audit.get("record_count") != 16 or input_audit.get("passed_count") is False:
        errors.append("input_audit")
    control_manifest = _load(run_root / "control_scores/manifest.json")
    errors.extend(_validate_fresh_controls(run_root, control_manifest, parent_scores))
    diagnostic = _load(config.DIAGNOSTIC_ANALYSIS)
    recomputed_control = _recompute_control(protocol, scientific, diagnostic)
    control = _load(run_root / "control_analysis.json")
    fresh_parity = [row for row in control_manifest.get("rows", []) if str(row.get("video_arm", "")).startswith("PARITY_")]
    fresh_replay = [row for row in control_manifest.get("rows", []) if str(row.get("video_arm")) == "N_REPLAY"]
    actual_control_pass = bool(recomputed_control["old_offset_pass_count"] == 13 and recomputed_control["matched_offset_pass_count"] == 16 and recomputed_control["anchor_pass"] and recomputed_control["diagnostic_ok"] and all(row.get("parity", {}).get("passes") for row in fresh_parity) and all(row.get("replay", {}).get("passes") for row in fresh_replay))
    if bool(control.get("control_pass")) != actual_control_pass or control.get("scientific_decision") != ("CONTROL_PASS" if actual_control_pass else "CONTROL_FAILED") or bool(control.get("stage_b_authorized")) != actual_control_pass:
        errors.append("control_decision_reconstruction")
    if control.get("old_offset_pass_count") != recomputed_control["old_offset_pass_count"] or control.get("matched_offset_pass_count") != recomputed_control["matched_offset_pass_count"] or not _compare(control.get("anchor_damage"), recomputed_control["anchor_damage"], config.METRIC_TOLERANCE):
        errors.append("control_numbers")
    if stage == "all":
        if not actual_control_pass:
            errors.append("candidate_artifacts_present_without_authorization" if (run_root / "candidate_scores/manifest.json").exists() else "all_requested_but_controls_failed")
        else:
            media_manifest = _load(run_root / "media/candidate_manifest.json")
            media_rows = media_manifest.get("rows", [])
            if len(media_rows) != config.EXPECTED_CANDIDATE_VIDEO_COUNT:
                errors.append("candidate_media_count")
            for row in media_rows:
                errors.extend(_check_media(row))
            candidate_manifest = _load(run_root / "candidate_scores/manifest.json")
            candidate_rows = candidate_manifest.get("rows", [])
            if len(candidate_rows) != config.EXPECTED_CANDIDATE_SCORE_COUNT:
                errors.append("candidate_score_count")
            media_by_key = {(str(row["sample_id"]), str(row["arm"])): row for row in media_rows}
            for row in candidate_rows:
                score = row.get("score", {})
                key = (str(row.get("sample_id")), str(row.get("video_arm")))
                if key not in media_by_key or str(score.get("audio_arm")) != "N" or list(score.get("U", {}).get("rows", [])) != list(config.U_ROWS):
                    errors.append(f"candidate_binding:{key}")
                try:
                    _matrix(row)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"candidate_matrix:{key}:{exc}")
            expected_analysis = _recompute_analysis(protocol, drivers, control_manifest.get("rows", []), candidate_rows)
            analysis = _load(run_root / "analysis.json")
            analysis_body = dict(analysis)
            analysis_body.pop("artifact_sha256", None)
            if not _compare(analysis_body, expected_analysis, config.METRIC_TOLERANCE):
                errors.append("analysis_reconstruction")
    status = "PASS" if not errors else "FAIL"
    return {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": status, "stage": stage, "error_count": len(errors), "errors": errors, "independent": True, "checked": {"record_count": 16, "control_rows": len(control_manifest.get("rows", [])), "candidate_rows": len(_load(run_root / "candidate_scores/manifest.json").get("rows", [])) if (run_root / "candidate_scores/manifest.json").is_file() else 0}, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent validator for the Wav2Lip continuation")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("controls", "all"), default="all")
    args = parser.parse_args(argv)
    output = args.run_root / ("control_validation.json" if args.stage == "controls" else "validation.json")
    try:
        result = validate(args.run_root, args.stage)
    except Exception as exc:  # noqa: BLE001 - preserve a truthful validation artifact
        result = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "FAIL", "stage": args.stage, "error_count": 1, "errors": [f"validator_exception:{type(exc).__name__}:{exc}"], "independent": True, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}
    write_self_hashed_json(output, result)
    print(result, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
