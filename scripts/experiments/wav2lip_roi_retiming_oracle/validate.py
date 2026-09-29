from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    OracleError,
    bytes_sha256,
    compare_values,
    extract_bgr24_frames,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _fixed_parent() -> dict[str, Any]:
    try:
        from scripts.experiments.wav2lip_roi_control_diagnostic.analysis import (
            load_parent,
        )

        return load_parent()
    except Exception as exc:
        raise OracleError(f"fixed parent audit failed: {exc}") from exc


def _formula_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * np.pi * config.WARP_AMPLITUDE_SAMPLES:
        raise OracleError("audio is too short for the registered warp")
    coordinates = np.arange(sample_count, dtype=np.float64)
    mapped = coordinates + config.WARP_AMPLITUDE_SAMPLES * np.sin(
        2.0 * np.pi * coordinates / float(sample_count - 1)
    )
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise OracleError("registered map is not strictly increasing")
    return mapped


def _oracle_q(sample_count: int, frame_count: int) -> list[int]:
    mapped = _formula_map(sample_count)
    coordinates = np.arange(sample_count, dtype=np.float64)
    t = config.SAMPLES_PER_FRAME * np.arange(frame_count, dtype=np.float64)
    sampled = np.interp(t, coordinates, mapped)
    q_float = sampled / config.SAMPLES_PER_FRAME
    q = np.floor(q_float + 0.5).astype(np.int64)
    if np.any(np.diff(q) < 0) or np.any(q < 0) or np.any(q >= frame_count):
        raise OracleError("independent oracle indices are invalid")
    if np.max(np.abs(q.astype(np.float64) - q_float), initial=0.0) > 0.5 + 1e-9:
        raise OracleError("independent oracle quantization bound failed")
    return [int(value) for value in q]


def _expected_warp(natural_pcm: bytes) -> bytes:
    if len(natural_pcm) % 2:
        raise OracleError("natural PCM is not int16 aligned")
    values = np.frombuffer(natural_pcm, dtype="<i2")
    mapped = _formula_map(int(values.size))
    warped = np.interp(mapped, np.arange(values.size, dtype=np.float64), values.astype(np.float64))
    return np.rint(warped).astype("<i2", copy=False).tobytes()


def _masks(sample_count: int, frame_count: int) -> dict[str, Any]:
    mapped = _formula_map(sample_count)
    coordinates = np.arange(sample_count, dtype=np.float64)
    q = min(frame_count, sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, coordinates, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, coordinates))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    return {
        "sample_count": sample_count,
        "q_frames": q,
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
    }


def _same_masks(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("sample_count", "q_frames", "candidate_rows", "common_window_rows", "plus_rows", "minus_rows"):
        if left.get(key) != right.get(key):
            return False
    for key in ("d_by_row", "a_by_row"):
        lvalue, rvalue = left.get(key), right.get(key)
        if not isinstance(lvalue, Mapping) or not isinstance(rvalue, Mapping) or set(lvalue) != set(rvalue):
            return False
        if any(abs(float(lvalue[item]) - float(rvalue[item])) > 1e-12 for item in lvalue):
            return False
    return True


def _peak(curve: np.ndarray) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    order = np.sort(values, kind="stable")
    min_index = int(np.argmin(values))
    offset = config.VSHIFT - min_index
    gap = float(order[1] - order[0])
    return {
        "offset": int(offset),
        "min_index": min_index,
        "sync_d": float(order[0]),
        "sync_c": float(np.median(values) - order[0]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT),
    }


def _summary(matrix: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    selected = [int(row) for row in rows]
    if not selected or min(selected) < 0 or max(selected) >= matrix.shape[0]:
        raise OracleError("validator mask is outside matrix")
    curve = np.mean(matrix[selected, :], axis=0, dtype=np.float64)
    result = _peak(curve)
    result["rows"] = selected
    result["curve"] = [float(value) for value in curve]
    return result


def _signature(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "common": _summary(matrix, masks["common_window_rows"]),
        "PLUS": _summary(matrix, masks["plus_rows"]),
        "MINUS": _summary(matrix, masks["minus_rows"]),
    }


def _distance_from_embeddings(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape != audio.shape or visual.shape[1] != config.EMBEDDING_DIM:
        raise OracleError("stored embedding shapes are invalid")
    padded = np.zeros((audio.shape[0] + 2 * config.VSHIFT, audio.shape[1]), dtype=np.float32)
    padded[config.VSHIFT : config.VSHIFT + audio.shape[0]] = audio.astype(np.float32, copy=False)
    result = np.empty((visual.shape[0], 2 * config.VSHIFT + 1), dtype=np.float32)
    epsilon = np.float32(config.EPSILON)
    for row in range(visual.shape[0]):
        difference = visual[row : row + 1].astype(np.float32) - padded[row : row + 2 * config.VSHIFT + 1]
        result[row] = np.sqrt(np.sum((difference + epsilon) ** 2, axis=1, dtype=np.float32), dtype=np.float32)
    return result


def validate_stored_arrays(visual: np.ndarray, audio: np.ndarray, matrix: np.ndarray) -> None:
    expected = _distance_from_embeddings(visual, audio)
    if matrix.shape != expected.shape or not np.isfinite(matrix).all():
        raise OracleError("stored distance matrix shape or finiteness is invalid")
    if float(np.max(np.abs(expected.astype(np.float64) - matrix.astype(np.float64)))) > config.VALIDATOR_MATRIX_TOLERANCE:
        raise OracleError("stored matrix differs from independently recomputed embeddings")


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    if len(values) != len(groups) or not values:
        raise OracleError("invalid validator bootstrap input")
    for group, value in zip(groups, values, strict=True):
        if not math.isfinite(float(value)):
            raise OracleError("validator bootstrap input is non-finite")
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    means = {label: float(np.mean(grouped[label])) for label in labels}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    draws = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        draws[index] = float(np.mean([means[str(label)] for label in sampled]))
    return {
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [
            float(np.quantile(draws, 0.025, method="linear")),
            float(np.quantile(draws, 0.975, method="linear")),
        ],
        "group_means": means,
    }


def _cell_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")))
        if key in result:
            raise OracleError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _protocol_record(protocol: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for row in protocol.get("records", []):
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise OracleError(f"validator protocol record is missing: {sample_id}")


def _parent_matrix(parent: Mapping[str, Any], sample_id: str, audio: str) -> np.ndarray:
    row = parent["score_index"].get((sample_id, "G_N", audio, False))
    if not isinstance(row, Mapping):
        raise OracleError(f"validator parent matrix is missing: {sample_id}/{audio}")
    cached = parent["matrix_cache"].get(str(row["matrix_sha256"]))
    if not isinstance(cached, np.ndarray):
        raise OracleError("validator parent matrix cache is incomplete")
    return np.asarray(cached, dtype=np.float64)


def _derive_summary(parent: Mapping[str, Any], protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = _cell_index(score_rows)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for parent_record in parent["records"]:
        sample_id = str(parent_record["sample_id"])
        record = _protocol_record(protocol, sample_id)
        masks = record["masks"]
        matrices: dict[tuple[str, str], np.ndarray] = {}
        for video, audio in config.CELL_SPECS:
            row = cells[(sample_id, video, audio)]
            matrices[(video, audio)] = np.asarray(np.load(Path(str(row["matrix"])), allow_pickle=False), dtype=np.float64)
        old_n = _parent_matrix(parent, sample_id, "N")
        old_w = _parent_matrix(parent, sample_id, "W")
        for old, new, audio in ((old_n, matrices["V_ID", "N"], "N"), (old_w, matrices["V_ID", "W"], "W")):
            shape_equal = old.shape == new.shape
            max_abs = float(np.max(np.abs(old - new))) if shape_equal else None
            comparisons.append({"sample_id": sample_id, "audio_arm": audio, "shape_equal": shape_equal, "max_abs": max_abs, "pass": bool(shape_equal and max_abs is not None and max_abs <= config.MATRIX_DIFF_TOLERANCE)})
        sig = {key: _signature(value, masks) for key, value in matrices.items()}
        baseline = bool(sig["V_ID", "N"]["PLUS"]["clear"] and sig["V_ID", "N"]["MINUS"]["clear"] and abs(sig["V_ID", "N"]["PLUS"]["offset"] - sig["V_ID", "N"]["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES)
        expected = {
            "B": {"PLUS": float(np.mean([masks["a_by_row"][str(item)] for item in masks["plus_rows"]])), "MINUS": float(np.mean([masks["a_by_row"][str(item)] for item in masks["minus_rows"]]))},
            "C_oracle": {"PLUS": float(-np.mean([masks["d_by_row"][str(item)] for item in masks["plus_rows"]])), "MINUS": float(-np.mean([masks["d_by_row"][str(item)] for item in masks["minus_rows"]]))},
            "O_oracle": {"PLUS": 0.0, "MINUS": 0.0},
        }
        checks: dict[str, Any] = {}
        for name, left_key, right_key in (("B", ("V_ID", "W"), ("V_ID", "N")), ("C_oracle", ("V_ORACLE", "N"), ("V_ID", "N")), ("O_oracle", ("V_ORACLE", "W"), ("V_ID", "N"))):
            parts: dict[str, Any] = {}
            for segment in ("PLUS", "MINUS"):
                actual = float(sig[left_key][segment]["offset"] - sig[right_key][segment]["offset"])
                error = actual - expected[name][segment]
                parts[segment] = {"actual": actual, "expected": expected[name][segment], "error": error, "passes": bool(sig[left_key][segment]["clear"] and sig[right_key][segment]["clear"] and abs(error) <= config.OFFSET_TOLERANCE_FRAMES)}
            parts["passes"] = bool(parts["PLUS"]["passes"] and parts["MINUS"]["passes"])
            checks[name] = parts
        own = {
            "c": float(sig["V_ORACLE", "W"]["common"]["sync_c"] - sig["V_ID", "N"]["common"]["sync_c"]),
            "d": float(sig["V_ID", "N"]["common"]["sync_d"] - sig["V_ORACLE", "W"]["common"]["sync_d"]),
            "offset_agreement": bool(abs(sig["V_ORACLE", "W"]["common"]["offset"] - sig["V_ID", "N"]["common"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES),
        }
        damage = {
            "c": float(sig["V_ORACLE", "W"]["common"]["sync_c"] - sig["V_ORACLE", "N"]["common"]["sync_c"]),
            "d": float(sig["V_ORACLE", "N"]["common"]["sync_d"] - sig["V_ORACLE", "W"]["common"]["sync_d"]),
            "both_positive": bool(sig["V_ORACLE", "W"]["common"]["sync_c"] > sig["V_ORACLE", "N"]["common"]["sync_c"] and sig["V_ORACLE", "N"]["common"]["sync_d"] > sig["V_ORACLE", "W"]["common"]["sync_d"]),
        }
        identity_pass = all(item["pass"] for item in comparisons[-2:])
        records.append({"sample_id": sample_id, "source_group": str(parent_record["source_group"]), "identity_pass": identity_pass, "baseline": baseline, "checks": checks, "own": own, "damage": damage})
    groups = [row["source_group"] for row in records]
    own_bootstrap = {"c": _bootstrap([row["own"]["c"] for row in records], groups), "d": _bootstrap([row["own"]["d"] for row in records], groups)}
    damage_bootstrap = {"c": _bootstrap([row["damage"]["c"] for row in records], groups), "d": _bootstrap([row["damage"]["d"] for row in records], groups)}
    baseline_count = sum(bool(row["baseline"]) for row in records)
    timing_counts = {name: sum(bool(row["checks"][name]["passes"]) for row in records) for name in ("B", "C_oracle", "O_oracle")}
    identity_pass = all(bool(row["identity_pass"]) for row in records)
    own_gate = {
        "ci_lower_gt_negative_0_10": bool(own_bootstrap["c"]["ci95"][0] > -0.10 and own_bootstrap["d"]["ci95"][0] > -0.10),
        "offset_agreement_count": sum(bool(row["own"]["offset_agreement"]) for row in records),
    }
    own_gate["offset_agreement_pass"] = own_gate["offset_agreement_count"] >= config.MIN_BASELINE_RECORDS
    own_gate["passes"] = bool(own_gate["ci_lower_gt_negative_0_10"] and own_gate["offset_agreement_pass"])
    damage_gate = {
        "ci_lower_gt_0_10": bool(damage_bootstrap["c"]["ci95"][0] > 0.10 and damage_bootstrap["d"]["ci95"][0] > 0.10),
        "both_positive_count": sum(bool(row["damage"]["both_positive"]) for row in records),
    }
    damage_gate["both_positive_pass"] = damage_gate["both_positive_count"] >= config.MIN_SUCCESS_RECORDS
    damage_gate["passes"] = bool(damage_gate["ci_lower_gt_0_10"] and damage_gate["both_positive_pass"])
    if not identity_pass:
        decision = "BASELINE_NOT_REPRODUCED"
    elif baseline_count < config.MIN_BASELINE_RECORDS or any(count < config.MIN_SUCCESS_RECORDS for count in timing_counts.values()):
        decision = "ORACLE_TIMING_UNRESOLVED"
    elif not own_gate["passes"]:
        decision = "ORACLE_OWN_AUDIO_UNRESOLVED"
    elif not damage_gate["passes"]:
        decision = "ORACLE_DAMAGE_UNRESOLVED"
    else:
        decision = "ORACLE_CONTROL_SUPPORTED"
    return {
        "decision": decision,
        "baseline_count": baseline_count,
        "timing_counts": timing_counts,
        "identity_reproduction": {"pass": identity_pass, "comparisons": comparisons, "max_abs_difference": float(max((row["max_abs"] or 0.0) for row in comparisons))},
        "own_bootstrap": own_bootstrap,
        "own_gate": own_gate,
        "damage_bootstrap": damage_bootstrap,
        "damage_gate": damage_gate,
        "per_record": records,
    }


def _verify_parent_bindings(parent: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    for name, expected in config.PARENT_HASHES.items():
        path = {
            "final": config.PARENT_ROOT / "final.json",
            "protocol": config.PARENT_ROOT / "protocol.json",
            "control": config.PARENT_ROOT / "control.json",
            "media_manifest": config.PARENT_ROOT / "media/control/manifest.json",
            "score_manifest": config.PARENT_ROOT / "scores/control/manifest.json",
        }[name]
        if file_sha256(path) != expected:
            raise OracleError(f"parent binding changed: {name}")
    if protocol.get("parent", {}).get("fixed_hashes") != config.PARENT_HASHES:
        raise OracleError("protocol parent hash binding changed")
    if protocol.get("parent", {}).get("scientific_decision") != "CONTROL_FAILED":
        raise OracleError("protocol does not preserve historical CONTROL_FAILED")
    for binding in list(protocol.get("change_bindings", {}).values()) + list(protocol.get("source_bindings", {}).values()):
        if not isinstance(binding, Mapping):
            raise OracleError("source binding is malformed")
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or file_sha256(path) != str(binding.get("sha256", "")):
            raise OracleError(f"bound source changed: {path}")
    recheck_final = verify_self_hashed_json(config.RECHECK_ROOT / "final.json", config.RECHECK_HASHES["final"])
    recheck_validation = verify_self_hashed_json(config.RECHECK_ROOT / "validation.json", config.RECHECK_HASHES["validation"])
    if recheck_final.get("diagnostic_decision") != "PEAKS_REPRODUCED" or recheck_validation.get("valid") is not True:
        raise OracleError("recheck binding is not valid")
    if parent["records"] and [str(row["sample_id"]) for row in parent["records"]] != [str(row["sample_id"]) for row in protocol.get("records", [])]:
        raise OracleError("protocol record order differs from parent")


def _validate_media(root: Path, parent: Mapping[str, Any], protocol: Mapping[str, Any], frame_manifest: Mapping[str, Any], audio_manifest: Mapping[str, Any], media_manifest: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> None:
    frame_rows = frame_manifest.get("rows")
    media_rows = media_manifest.get("rows")
    if frame_manifest.get("status") != "complete" or not isinstance(frame_rows, list) or len(frame_rows) != config.EXPECTED_STREAM_COUNT:
        raise OracleError("frame manifest is incomplete")
    if media_manifest.get("status") != "complete" or not isinstance(media_rows, list) or len(media_rows) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("media manifest is incomplete")
    audio_dir = root / "audio"
    frame_index = {(str(row["sample_id"]), str(row["video_arm"])): row for row in frame_rows}
    media_index = {str(row["sample_id"]): row for row in media_rows}
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        natural_pcm, natural, _ = read_pcm16_wav(Path(str(record["natural_audio"]["path"])))
        expected_warp = _expected_warp(natural_pcm)
        audio_paths = {
            "N": audio_dir / f"{sample_id}__N.wav",
            "W": audio_dir / f"{sample_id}__W.wav",
        }
        for arm, path in audio_paths.items():
            if not path.is_file():
                raise OracleError(f"audio artifact is missing: {path}")
            pcm, values, _params = read_pcm16_wav(path)
            if arm == "N" and pcm != natural_pcm:
                raise OracleError(f"N audio changed: {sample_id}")
            if arm == "W" and pcm != expected_warp:
                raise OracleError(f"W audio reconstruction changed: {sample_id}")
            if values.size != natural.size:
                raise OracleError(f"audio length changed: {sample_id}/{arm}")
        source_path = Path(str(record["parent_stream"]["path"]))
        source_frames, _source_evidence = extract_bgr24_frames(source_path)
        for video in config.VIDEO_ARMS:
            frame_row = frame_index.get((sample_id, video))
            if not isinstance(frame_row, Mapping):
                raise OracleError(f"frame evidence is missing: {sample_id}/{video}")
            stream_path = Path(str(frame_row["evidence"]["output"]))
            if file_sha256(stream_path) != str(frame_row["evidence"]["output_sha256"]):
                raise OracleError(f"new stream hash changed: {sample_id}/{video}")
            observed, _obs_evidence = extract_bgr24_frames(stream_path)
            q = frame_row.get("q")
            expected_q = list(range(source_frames.shape[0])) if video == "V_ID" else _oracle_q(natural.size, source_frames.shape[0])
            if q != expected_q:
                raise OracleError(f"frame index mapping changed: {sample_id}/{video}")
            expected_frames = source_frames[np.asarray(expected_q, dtype=np.int64)]
            if not np.array_equal(observed, expected_frames):
                raise OracleError(f"pixel identity mismatch: {sample_id}/{video}")
        media_row = media_index[sample_id]
        for video, audio in config.CELL_SPECS:
            cell = media_row.get("cells", {}).get(f"{video}__{audio}")
            if not isinstance(cell, Mapping):
                raise OracleError(f"media cell is missing: {sample_id}/{video}/{audio}")
            media_path = Path(str(cell["output"]))
            if file_sha256(media_path) != str(cell["output_sha256"]):
                raise OracleError(f"media hash changed: {media_path}")
            observed_frames, _ = extract_bgr24_frames(media_path)
            stream_frames, _ = extract_bgr24_frames(Path(str(media_row["streams"][video]["output"])))
            if not np.array_equal(observed_frames, stream_frames):
                raise OracleError(f"mux video identity changed: {sample_id}/{video}/{audio}")
            pcm = extract_pcm_from_media(media_path)
            expected_pcm = natural_pcm if audio == "N" else expected_warp
            if pcm != expected_pcm or bytes_sha256(pcm) != str(cell["audio_pcm_sha256"]):
                raise OracleError(f"mux PCM identity changed: {sample_id}/{video}/{audio}")


def _validate_scores(root: Path, score_rows: Sequence[Mapping[str, Any]], protocol_sha: str) -> None:
    expected = {(str(row["sample_id"]), video, audio) for row in score_rows for video, audio in ()}
    del expected
    if len(score_rows) != config.EXPECTED_SCORE_CELL_COUNT:
        raise OracleError("score cell count is not 88")
    cells = _cell_index(score_rows)
    if len(cells) != config.EXPECTED_SCORE_CELL_COUNT:
        raise OracleError("score cell keys are incomplete")
    for row in score_rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        for field in ("media", "audio", "visual", "audio_embedding", "matrix", "worker_result"):
            path = Path(str(row[field]))
            if field in ("visual", "audio_embedding", "matrix", "worker_result") and root.resolve() not in path.resolve().parents:
                raise OracleError(f"score artifact escapes run root: {key}/{field}")
            if not path.is_file():
                raise OracleError(f"score artifact is missing: {key}/{field}")
        if file_sha256(Path(str(row["media"]))) != str(row["media_sha256"]):
            raise OracleError(f"score media hash changed: {key}")
        if file_sha256(Path(str(row["audio"]))) != str(row["audio_container_sha256"]):
            raise OracleError(f"score audio hash changed: {key}")
        for field, hash_field in (("visual", "visual_sha256"), ("audio_embedding", "audio_embedding_sha256"), ("matrix", "matrix_sha256"), ("worker_result", "worker_result_sha256")):
            if file_sha256(Path(str(row[field]))) != str(row[hash_field]):
                raise OracleError(f"score artifact hash changed: {key}/{field}")
        worker = verify_self_hashed_json(Path(str(row["worker_result"])))
        if worker.get("protocol_sha256") != protocol_sha or worker.get("new_forward") is not True or worker.get("extracted_pcm_verified") is not True:
            raise OracleError(f"worker provenance is invalid: {key}")
        visual = np.asarray(np.load(Path(str(row["visual"])), allow_pickle=False), dtype=np.float32)
        audio = np.asarray(np.load(Path(str(row["audio_embedding"])), allow_pickle=False), dtype=np.float32)
        matrix = np.asarray(np.load(Path(str(row["matrix"])), allow_pickle=False), dtype=np.float32)
        validate_stored_arrays(visual, audio, matrix)


def validate_run(run_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    final_sha: str | None = None
    errors: list[str] = []
    try:
        final_path = root / "final.json"
        final_sha = file_sha256(final_path) if final_path.is_file() else None
        final = verify_self_hashed_json(final_path)
        protocol = verify_self_hashed_json(root / "protocol.json")
        audit = verify_self_hashed_json(root / "input_audit.json")
        frame_manifest = verify_self_hashed_json(root / "frames/manifest.json")
        audio_manifest = verify_self_hashed_json(root / "audio/manifest.json")
        media_manifest = verify_self_hashed_json(root / "media/manifest.json")
        score_manifest = verify_self_hashed_json(root / "scores/manifest.json")
        analysis = verify_self_hashed_json(root / "analysis.json")
        parent = _fixed_parent()
        if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("status") != "frozen":
            raise OracleError("protocol identity/status is invalid")
        _verify_parent_bindings(parent, protocol)
        if audit.get("status") != "complete" or audit.get("record_count") != config.EXPECTED_RECORD_COUNT:
            raise OracleError("input audit is incomplete")
        _validate_media(root, parent, protocol, frame_manifest, audio_manifest, media_manifest, score_manifest["scores"])
        protocol_sha = file_sha256(root / "protocol.json")
        _validate_scores(root, score_manifest["scores"], protocol_sha)
        independent = _derive_summary(parent, protocol, score_manifest["scores"])
        actual_summary = {
            key: analysis.get(key)
            for key in ("decision", "baseline_count", "timing_counts", "identity_reproduction", "own_bootstrap", "own_gate", "damage_bootstrap", "damage_gate")
        }
        expected_summary = {key: independent[key] for key in actual_summary}
        differences: list[dict[str, Any]] = []
        compare_values(expected_summary, actual_summary, "analysis", differences, 1e-6)
        if differences:
            raise OracleError(f"analysis differs from independent validator: {differences[:3]}")
        expected_final = {
            "status": "complete",
            "engineering_decision": "GO",
            "diagnostic_decision": independent["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "parent_own_audio_retested": False,
            "oracle_own_audio_tested": True,
            "bridge_executed": False,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "new_generated_videos": 0,
            "new_video_stream_count": config.EXPECTED_STREAM_COUNT,
            "new_media_count": config.EXPECTED_MEDIA_COUNT,
            "new_score_cells": config.EXPECTED_SCORE_CELL_COUNT,
        }
        for key, value in expected_final.items():
            if final.get(key) != value:
                raise OracleError(f"final contract mismatch: {key}")
        if final.get("protocol_sha256") != protocol_sha or final.get("analysis_sha256") != file_sha256(root / "analysis.json"):
            raise OracleError("final evidence binding mismatch")
        for field, path in (("input_audit_sha256", root / "input_audit.json"), ("media_manifest_sha256", root / "media/manifest.json"), ("score_manifest_sha256", root / "scores/manifest.json"), ("result_sha256", root / "result.md")):
            if final.get(field) != file_sha256(path):
                raise OracleError(f"final artifact hash mismatch: {field}")
        result = {
            "schema_version": 1,
            "stage_id": "validation",
            "protocol_id": config.PROTOCOL_ID,
            "status": "valid",
            "valid": True,
            "mode": "independent_pixel_embedding_matrix_statistics",
            "final_sha256": final_sha,
            "diagnostic_decision": independent["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "independent_difference_count": 0,
            "new_generated_videos": 0,
            "new_video_stream_count": config.EXPECTED_STREAM_COUNT,
            "new_score_cells": config.EXPECTED_SCORE_CELL_COUNT,
            "bridge_executed": False,
            "errors": [],
        }
    except Exception as exc:  # noqa: BLE001 - validator always emits an artifact
        errors.append(str(exc))
        result = {
            "schema_version": 1,
            "stage_id": "validation",
            "protocol_id": config.PROTOCOL_ID,
            "status": "invalid",
            "valid": False,
            "mode": "independent_pixel_embedding_matrix_statistics",
            "final_sha256": final_sha,
            "independent_difference_count": None,
            "errors": errors,
        }
    write_self_hashed_json(root / "validation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Wav2Lip ROI retiming oracle run")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    print(result)
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
