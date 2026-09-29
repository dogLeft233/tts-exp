from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    OracleError,
    bytes_sha256,
    extract_bgr24_frames,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)

from . import config


def _distance_from_embeddings(visual: np.ndarray, audio: np.ndarray, vshift: int = config.VSHIFT) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.ndim != 2 or audio_value.shape != visual_value.shape or visual_value.shape[1] != config.EMBEDDING_DIM:
        raise OracleError(f"embedding shapes are invalid: {visual_value.shape}, {audio_value.shape}")
    padded = np.pad(audio_value, ((vshift, vshift), (0, 0)), mode="constant")
    rows = []
    for row in range(visual_value.shape[0]):
        difference = visual_value[row : row + 1] - padded[row : row + 2 * vshift + 1]
        rows.append(np.sqrt(np.sum((difference + np.float32(config.EPSILON)) ** 2, axis=1, dtype=np.float32)))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def validate_stored_arrays(visual: np.ndarray, audio: np.ndarray, stored: np.ndarray) -> None:
    expected = _distance_from_embeddings(visual, audio)
    actual = np.asarray(stored, dtype=np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise OracleError(f"stored matrix shape or finiteness is invalid: {actual.shape}")
    difference = float(np.max(np.abs(actual - expected), initial=0.0))
    if difference > config.VALIDATOR_MATRIX_TOLERANCE:
        raise OracleError(f"stored matrix differs from independent distance: {difference}")


def _fixed_parent() -> dict[str, Any]:
    payloads = {
        name: verify_self_hashed_json(path, config.PARENT_HASHES[name])
        for name, path in config.PARENT_FILES.items()
    }
    if payloads["validation"].get("valid") is not True:
        raise OracleError("parent validation is not valid")
    if payloads["final"].get("diagnostic_decision") != "ORACLE_OWN_AUDIO_UNRESOLVED":
        raise OracleError("parent oracle decision changed")
    records = payloads["protocol"].get("records", [])
    score_rows = payloads["score_manifest"].get("scores", [])
    if len(records) != config.EXPECTED_RECORD_COUNT or len(score_rows) != config.EXPECTED_CACHED_SCORE_CELL_COUNT:
        raise OracleError("parent cohort or cached score count is incomplete")
    return {
        "payloads": payloads,
        "protocol": payloads["protocol"],
        "records": records,
        "score_rows": score_rows,
    }


def _load_array(row: Mapping[str, Any], field: str, hash_field: str) -> np.ndarray:
    path = Path(str(row[field]))
    expected = str(row[hash_field])
    if not path.is_file() or file_sha256(path) != expected:
        raise OracleError(f"{field} binding is invalid: {path}")
    value = np.asarray(np.load(path, allow_pickle=False))
    if not np.isfinite(value).all():
        raise OracleError(f"{field} contains non-finite values: {path}")
    return value


def _forward_map(sample_count: int) -> np.ndarray:
    coordinates = np.arange(sample_count, dtype=np.float64)
    mapped = coordinates + config.WARP_AMPLITUDE_SAMPLES * np.sin(2.0 * np.pi * coordinates / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise OracleError("independent warp mapping is not strictly increasing")
    return mapped


def _independent_indices(sample_count: int, frame_count: int) -> dict[str, Any]:
    mapped = _forward_map(sample_count)
    t = config.SAMPLES_PER_FRAME * np.arange(frame_count, dtype=np.float64)
    u = np.interp(t, np.arange(sample_count, dtype=np.float64), mapped) / config.SAMPLES_PER_FRAME
    j = np.floor(u).astype(np.int64)
    k = np.ceil(u).astype(np.int64)
    if float(t[-1]) > sample_count - 1 or np.any(j < 0) or np.any(k >= frame_count):
        raise OracleError("independent interpolation coordinate is out of bounds")
    w = u - j.astype(np.float64)
    return {
        "u": u,
        "j": j,
        "k": k,
        "w": w,
        "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
    }


def _independent_pixels(source: np.ndarray, indices: Mapping[str, Any]) -> np.ndarray:
    j = np.asarray(indices["j"], dtype=np.int64)
    k = np.asarray(indices["k"], dtype=np.int64)
    w = np.asarray(indices["w"], dtype=np.float64)
    value = (1.0 - w[:, None, None, None]) * source[j].astype(np.float64) + w[:, None, None, None] * source[k].astype(np.float64)
    return np.floor(value + 0.5).astype(np.uint8)


def _timeline_ok(evidence: Mapping[str, Any], expected_count: int) -> None:
    probe = evidence.get("probe", {})
    stream = probe.get("stream", {})
    required = {
        "codec_name": "ffv1",
        "pix_fmt": "bgr0",
        "color_range": "pc",
        "color_space": "gbr",
        "width": config.FRAME_WIDTH,
        "height": config.FRAME_HEIGHT,
        "r_frame_rate": "25/1",
    }
    if any(stream.get(key) != value for key, value in required.items()):
        raise OracleError(f"video metadata does not match the frozen contract: {stream}")
    frames = probe.get("frames", [])
    if len(frames) != expected_count:
        raise OracleError("video frame count differs from the protocol")
    times = [float(item["best_effort_timestamp_time"]) for item in frames]
    if not times or any(abs(value - index / config.FPS) * 1000.0 > 1.0 for index, value in enumerate(times)):
        raise OracleError("video PTS differs from the frozen timeline")


def _peak(curve: np.ndarray) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    order = np.sort(values, kind="stable")
    index = int(np.argmin(values))
    offset = config.VSHIFT - index
    return {
        "offset": int(offset),
        "sync_c": float(np.median(values) - order[0]),
        "sync_d": float(order[0]),
        "clear": bool(order[1] - order[0] > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT),
    }


def _signature(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, rows in {
        "common": masks["common_window_rows"],
        "PLUS": masks["plus_rows"],
        "MINUS": masks["minus_rows"],
    }.items():
        selected = np.asarray(rows, dtype=np.int64)
        if selected.size < config.MIN_LOCAL_ROWS or selected.min() < 0 or selected.max() >= matrix.shape[0]:
            raise OracleError(f"invalid validator mask: {name}")
        result[name] = _peak(np.mean(matrix[selected], axis=0, dtype=np.float64))
    return result


def _timing(left: Mapping[str, Any], right: Mapping[str, Any], masks: Mapping[str, Any], mode: str) -> bool:
    if mode == "B":
        targets = [float(np.mean([masks["a_by_row"][str(row)] for row in masks[name]])) for name in ("plus_rows", "minus_rows")]
    elif mode == "C":
        targets = [float(-np.mean([masks["d_by_row"][str(row)] for row in masks[name]])) for name in ("plus_rows", "minus_rows")]
    else:
        targets = [0.0, 0.0]
    for name, target in zip(("PLUS", "MINUS"), targets, strict=True):
        actual = int(left[name]["offset"] - right[name]["offset"])
        if not left[name]["clear"] or not right[name]["clear"] or abs(actual - target) > config.OFFSET_TOLERANCE_FRAMES:
            return False
    return True


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([np.mean(grouped[str(label)]) for label in sampled]))
    return {
        "mean": float(statistics.fmean(values)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
    }


def _compare_summary(parent: Mapping[str, Any], protocol: Mapping[str, Any], fresh_by_key: Mapping[tuple[str, str, str], np.ndarray], parent_by_key: Mapping[tuple[str, str, str], np.ndarray], analysis: Mapping[str, Any]) -> None:
    records = []
    for row in protocol["records"]:
        sample_id = str(row["sample_id"])
        masks = row["masks"]
        parent_sig = {(video, audio): _signature(parent_by_key[(sample_id, video, audio)], masks) for video in ("V_ID", "V_ORACLE") for audio in config.AUDIO_ARMS}
        linear_sig = {(config.VIDEO_ARM, audio): _signature(fresh_by_key[(sample_id, config.VIDEO_ARM, audio)], masks) for audio in config.AUDIO_ARMS}
        own_c = linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_c"] - parent_sig["V_ID", "N"]["common"]["sync_c"]
        own_d = parent_sig["V_ID", "N"]["common"]["sync_d"] - linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_d"]
        damage_c = linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_c"] - linear_sig[config.VIDEO_ARM, "N"]["common"]["sync_c"]
        damage_d = linear_sig[config.VIDEO_ARM, "N"]["common"]["sync_d"] - linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_d"]
        gain_c = linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_c"] - parent_sig["V_ORACLE", "W"]["common"]["sync_c"]
        gain_d = parent_sig["V_ORACLE", "W"]["common"]["sync_d"] - linear_sig[config.VIDEO_ARM, "W"]["common"]["sync_d"]
        records.append(
            {
                "group": str(row["source_group"]),
                "baseline": bool(parent_sig["V_ID", "N"]["PLUS"]["clear"] and parent_sig["V_ID", "N"]["MINUS"]["clear"] and abs(parent_sig["V_ID", "N"]["PLUS"]["offset"] - parent_sig["V_ID", "N"]["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES),
                "B": _timing(parent_sig["V_ID", "W"], parent_sig["V_ID", "N"], masks, "B"),
                "C": _timing(linear_sig[config.VIDEO_ARM, "N"], parent_sig["V_ID", "N"], masks, "C"),
                "O": _timing(linear_sig[config.VIDEO_ARM, "W"], parent_sig["V_ID", "N"], masks, "O"),
                "own_c": own_c,
                "own_d": own_d,
                "own_offset": abs(linear_sig[config.VIDEO_ARM, "W"]["common"]["offset"] - parent_sig["V_ID", "N"]["common"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES,
                "damage_c": damage_c,
                "damage_d": damage_d,
                "damage_positive": bool(damage_c > 0.0 and damage_d > 0.0),
                "gain_c": gain_c,
                "gain_d": gain_d,
            }
        )
    groups = [row["group"] for row in records]
    expected = {
        "baseline_count": sum(row["baseline"] for row in records),
        "timing_counts": {key: sum(row[key] for row in records) for key in ("B", "C", "O")},
        "own_c": _bootstrap([row["own_c"] for row in records], groups),
        "own_d": _bootstrap([row["own_d"] for row in records], groups),
        "damage_c": _bootstrap([row["damage_c"] for row in records], groups),
        "damage_d": _bootstrap([row["damage_d"] for row in records], groups),
        "gain_c": _bootstrap([row["gain_c"] for row in records], groups),
        "gain_d": _bootstrap([row["gain_d"] for row in records], groups),
        "own_offset_count": sum(row["own_offset"] for row in records),
        "damage_positive_count": sum(row["damage_positive"] for row in records),
    }
    if expected["baseline_count"] != analysis["baseline_count"] or expected["timing_counts"] != {
        "B": analysis["timing_counts"]["B"],
        "C": analysis["timing_counts"]["C_linear"],
        "O": analysis["timing_counts"]["O_linear"],
    }:
        raise OracleError("independent timing counts differ from analysis")
    if expected["own_offset_count"] != analysis["own_gate"]["offset_agreement_count"] or expected["damage_positive_count"] != analysis["damage_gate"]["both_positive_count"]:
        raise OracleError("independent gate counts differ from analysis")
    for key, section, metric in (
        ("own_c", "own_bootstrap", "c"),
        ("own_d", "own_bootstrap", "d"),
        ("damage_c", "damage_bootstrap", "c"),
        ("damage_d", "damage_bootstrap", "d"),
        ("gain_c", "gain_bootstrap", "c"),
        ("gain_d", "gain_bootstrap", "d"),
    ):
        for field in ("mean", "ci95"):
            actual = expected[key][field]
            recorded = analysis[section][metric][field]
            if isinstance(actual, list):
                if any(abs(float(left) - float(right)) > config.STAT_TOLERANCE for left, right in zip(actual, recorded, strict=True)):
                    raise OracleError(f"independent bootstrap differs: {section}/{metric}/{field}")
            elif abs(float(actual) - float(recorded)) > config.STAT_TOLERANCE:
                raise OracleError(f"independent bootstrap differs: {section}/{metric}/{field}")
    timing_ready = expected["baseline_count"] >= config.MIN_BASELINE_RECORDS and all(value >= config.MIN_SUCCESS_RECORDS for value in expected["timing_counts"].values())
    own_ci_pass = expected["own_c"]["ci95"][0] > -0.10 and expected["own_d"]["ci95"][0] > -0.10
    own_offset_pass = expected["own_offset_count"] >= config.MIN_BASELINE_RECORDS
    own_pass = bool(own_ci_pass and own_offset_pass)
    damage_ci_pass = expected["damage_c"]["ci95"][0] > 0.10 and expected["damage_d"]["ci95"][0] > 0.10
    damage_positive_pass = expected["damage_positive_count"] >= config.MIN_SUCCESS_RECORDS
    damage_pass = bool(damage_ci_pass and damage_positive_pass)
    if analysis["own_gate"]["ci_lower_gt_negative_0_10"] != own_ci_pass or analysis["own_gate"]["offset_agreement_pass"] != own_offset_pass or analysis["own_gate"]["passes"] != own_pass:
        raise OracleError("independent own gate differs from analysis")
    if analysis["damage_gate"]["ci_lower_gt_0_10"] != damage_ci_pass or analysis["damage_gate"]["both_positive_pass"] != damage_positive_pass or analysis["damage_gate"]["passes"] != damage_pass:
        raise OracleError("independent damage gate differs from analysis")
    if not timing_ready:
        decision = "LINEAR_TIMING_UNRESOLVED"
    elif not own_pass:
        decision = "LINEAR_OWN_AUDIO_UNRESOLVED"
    elif not damage_pass:
        decision = "LINEAR_DAMAGE_UNRESOLVED"
    else:
        decision = "LINEAR_ORACLE_CONTROL_SUPPORTED"
    gain_supported = bool(expected["gain_c"]["ci95"][0] > 0.10 and expected["gain_d"]["ci95"][0] > 0.10)
    if decision != analysis["decision"] or gain_supported != analysis["interpolation_improvement_supported"]:
        raise OracleError("independent scientific decision differs from analysis")


def validate_run(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    protocol = verify_self_hashed_json(root / "protocol.json")
    verify_self_hashed_json(root / "input_audit.json")
    frames_manifest = verify_self_hashed_json(root / "frames/manifest.json")
    media_manifest = verify_self_hashed_json(root / "media/manifest.json")
    score_manifest = verify_self_hashed_json(root / "scores/manifest.json")
    analysis = verify_self_hashed_json(root / "analysis.json")
    final_path = root / "final.json"
    final = verify_self_hashed_json(final_path)
    parent = _fixed_parent()
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("status") != "frozen":
        raise OracleError("protocol identity/status is invalid")
    if file_sha256(root / "input_audit.json") != protocol.get("input_audit_sha256"):
        raise OracleError("protocol input-audit binding is invalid")
    if len(protocol.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("protocol record count is invalid")
    if len(frames_manifest.get("rows", [])) != config.EXPECTED_FRESH_STREAM_COUNT:
        raise OracleError("fresh frame stream count is invalid")
    if len(media_manifest.get("rows", [])) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("fresh media record count is invalid")
    fresh_rows = score_manifest.get("scores", [])
    if len(fresh_rows) != config.EXPECTED_FRESH_SCORE_CELL_COUNT:
        raise OracleError("fresh score count is invalid")
    fresh_by_key = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in fresh_rows}
    if len(fresh_by_key) != config.EXPECTED_FRESH_SCORE_CELL_COUNT or any(row.get("origin") != "fresh" for row in fresh_rows):
        raise OracleError("fresh score identity/count is invalid")

    frame_by_id = {str(row["sample_id"]): row for row in frames_manifest["rows"]}
    media_by_id = {str(row["sample_id"]): row for row in media_manifest["rows"]}
    if len(frame_by_id) != config.EXPECTED_RECORD_COUNT or len(media_by_id) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("duplicate or missing fresh media record")

    parent_score_rows = parent["score_rows"]
    parent_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    for row in parent_score_rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        parent_by_key[key] = _load_array(row, "matrix", "matrix_sha256")
    fresh_matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        source_path = Path(str(record["source_stream"]["path"])).resolve()
        source_frames, _source_evidence = extract_bgr24_frames(source_path)
        indices = _independent_indices(int(record["sample_count"]), int(record["frame_count"]))
        recorded_indices = record["linear_indices"]
        if not np.allclose(indices["u"], recorded_indices["u"], atol=1e-9, rtol=0.0) or indices["mapping_sha256"] != recorded_indices["mapping_sha256"]:
            raise OracleError(f"protocol interpolation indices differ: {sample_id}")
        expected_frames = _independent_pixels(source_frames, indices)
        frame_row = frame_by_id[sample_id]
        video = Path(str(frame_row["evidence"]["output"])).resolve()
        if file_sha256(video) != str(frame_row["evidence"]["output_sha256"]):
            raise OracleError(f"fresh video hash changed: {sample_id}")
        decoded, evidence = extract_bgr24_frames(video)
        _timeline_ok(evidence, int(record["frame_count"]))
        if not np.array_equal(decoded, expected_frames):
            raise OracleError(f"fresh video pixels differ from independent interpolation: {sample_id}")
        media_row = media_by_id[sample_id]
        for audio in config.AUDIO_ARMS:
            key = (sample_id, config.VIDEO_ARM, audio)
            score = fresh_by_key[key]
            media_cell = media_row["cells"][audio]
            media_path = Path(str(media_cell["output"])).resolve()
            if file_sha256(media_path) != str(media_cell["output_sha256"]) or str(score["media_sha256"]) != str(media_cell["output_sha256"]):
                raise OracleError(f"fresh media binding changed: {sample_id}/{audio}")
            mux_frames, mux_evidence = extract_bgr24_frames(media_path)
            _timeline_ok(mux_evidence, int(record["frame_count"]))
            if not np.array_equal(mux_frames, expected_frames):
                raise OracleError(f"muxed video pixels changed: {sample_id}/{audio}")
            audio_path = Path(str(record["audio_paths"][audio])).resolve()
            expected_pcm, _audio_values, _params = read_pcm16_wav(audio_path)
            observed_pcm = extract_pcm_from_media(media_path)
            if observed_pcm != expected_pcm or bytes_sha256(observed_pcm) != str(score["media_pcm_sha256"]):
                raise OracleError(f"muxed PCM changed: {sample_id}/{audio}")
            visual = _load_array(score, "visual", "visual_sha256").astype(np.float32, copy=False)
            audio_embedding = _load_array(score, "audio_embedding", "audio_embedding_sha256").astype(np.float32, copy=False)
            matrix = _load_array(score, "matrix", "matrix_sha256").astype(np.float32, copy=False)
            validate_stored_arrays(visual, audio_embedding, matrix)
            fresh_matrices[key] = matrix.astype(np.float64)

    _compare_summary(parent, protocol, fresh_matrices, parent_by_key, analysis)
    final_expected = {
        "status": "complete",
        "engineering_decision": "GO",
        "diagnostic_decision": analysis["decision"],
        "historical_scientific_decision": "CONTROL_FAILED",
        "parent_oracle_decision": "ORACLE_OWN_AUDIO_UNRESOLVED",
        "parent_own_audio_retested": False,
        "new_generated_videos": 0,
        "new_video_stream_count": config.EXPECTED_FRESH_STREAM_COUNT,
        "new_media_count": config.EXPECTED_FRESH_MEDIA_COUNT,
        "new_score_cells": config.EXPECTED_FRESH_SCORE_CELL_COUNT,
        "cached_score_cells": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
        "interpolation_improvement_supported": analysis["interpolation_improvement_supported"],
        "bridge_executed": False,
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
    }
    for key, expected in final_expected.items():
        if final.get(key) != expected:
            raise OracleError(f"final field differs: {key}")
    for key, path in {
        "protocol_sha256": root / "protocol.json",
        "input_audit_sha256": root / "input_audit.json",
        "analysis_sha256": root / "analysis.json",
        "result_sha256": root / "result.md",
        "frame_manifest_sha256": root / "frames/manifest.json",
        "media_manifest_sha256": root / "media/manifest.json",
        "score_manifest_sha256": root / "scores/manifest.json",
    }.items():
        if final.get(key) != file_sha256(path):
            raise OracleError(f"final artifact binding differs: {key}")
    result = {
        "schema_version": 1,
        "stage_id": "validation",
        "protocol_id": config.PROTOCOL_ID,
        "status": "valid",
        "valid": True,
        "mode": "independent_pixel_embedding_matrix_statistics",
        "final_sha256": file_sha256(final_path),
        "diagnostic_decision": analysis["decision"],
        "historical_scientific_decision": "CONTROL_FAILED",
        "independent_difference_count": 0,
        "fresh_video_stream_count": config.EXPECTED_FRESH_STREAM_COUNT,
        "fresh_media_count": config.EXPECTED_FRESH_MEDIA_COUNT,
        "fresh_score_cells": config.EXPECTED_FRESH_SCORE_CELL_COUNT,
        "cached_score_cells": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
        "bridge_executed": False,
        "errors": [],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate the linear pixel oracle experiment")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_run(args.run_root)
    except Exception as exc:  # noqa: BLE001 - validation artifact records the exact failure
        result = {
            "schema_version": 1,
            "stage_id": "validation",
            "protocol_id": config.PROTOCOL_ID,
            "status": "invalid",
            "valid": False,
            "mode": "independent_pixel_embedding_matrix_statistics",
            "errors": [str(exc)],
        }
        path = args.run_root.resolve() / "validation.json"
        write_self_hashed_json(path, result)
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    path = args.run_root.resolve() / "validation.json"
    write_self_hashed_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
