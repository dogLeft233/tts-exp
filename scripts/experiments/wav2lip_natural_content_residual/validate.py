from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import decode_pcm16, decode_video_frames

from . import config
from .common import ProtocolError, bytes_sha256, file_sha256, read_json, source_pcm16, verify_self_hashed_json, write_self_hashed_json


def _mask_identity(mask: dict[str, Any]) -> tuple[Any, ...]:
    return (str(mask.get("mask_sha256")), int(mask.get("global_start_frame")), int(mask.get("global_end_frame")), int(mask.get("core_start")), int(mask.get("core_end")))


def _parent_driver_index() -> dict[tuple[str, int, str], dict[str, Any]]:
    payload = read_json(config.DRIVERS)
    rows = payload.get("drivers")
    if not isinstance(rows, list):
        raise ProtocolError("parent drivers are malformed")
    return {(str(row["sample_id"]), int(row["seed"]), str(row["condition"])): row for row in rows}


def _shuffle(correct: np.ndarray, sample_id: str, columns: list[int]) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(config.SHUFFLE_SALT + sample_id.encode("utf-8")).digest()[:8], "little")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(columns)).astype(np.int64)
    if np.array_equal(permutation, np.arange(len(columns), dtype=np.int64)):
        permutation[[0, 1]] = permutation[[1, 0]]
    value = np.zeros_like(correct, dtype=np.float64)
    index = np.asarray(columns, dtype=np.int64)
    value[:, index] = correct[:, index][:, permutation]
    return value


def _expected_arms(sample_id: str, natural: np.ndarray, parent_index: dict[tuple[str, int, str], dict[str, Any]], columns: list[int]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {condition: [] for condition in config.DRIVER_CONDITIONS}
    identities: list[tuple[Any, ...]] | None = None
    for seed in config.SEEDS:
        for condition in config.DRIVER_CONDITIONS:
            row = parent_index[(sample_id, seed, condition)]
            current = [_mask_identity(mask) for mask in row["used_masks"]]
            if identities is None:
                identities = current
            if current != identities:
                raise ProtocolError(f"validator mask contract changed: {sample_id}")
            arrays[condition].append(np.asarray(np.load(Path(str(row["path"])), allow_pickle=False), dtype=np.float64))
    means = {condition: np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64) for condition, values in arrays.items()}
    correct = means["PAIRED_TTS"] - means["NAT_ONLY"]
    wrong = means["SAME_PHONE_WRONG_INSTANCE"] - means["NAT_ONLY"]
    correct_norm = float(np.linalg.norm(correct[:, columns]))
    wrong_norm = float(np.linalg.norm(wrong[:, columns]))
    wrong = wrong * correct_norm / wrong_norm
    shuffled = _shuffle(correct, sample_id, columns)
    amplitude = min(0.25, 0.5 / float(max(np.max(np.abs(correct)), np.max(np.abs(wrong)), np.max(np.abs(shuffled)))) )
    result = {"N": np.asarray(natural, dtype=np.float32).copy()}
    for name, residual in (("CORRECT", correct), ("WRONG", wrong), ("SHUFFLE", shuffled)):
        value = np.asarray(natural, dtype=np.float64).copy()
        index = np.asarray(columns, dtype=np.int64)
        value[:, index] = np.clip(value[:, index] + amplitude * residual[:, index], -4.0, 4.0)
        result[name] = np.asarray(value, dtype=np.float32)
    return result


def _check_static_faces(protocol: dict[str, Any], driver_manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for record, row in zip(protocol["records"], driver_manifest["rows"], strict=True):
        sample_id = str(record["sample_id"])
        face = Path(str(row["face_video"]["path"]))
        boxes = json.loads(Path(str(row["box"]["path"])).read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(str(face))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            errors.append(f"static_face_source:{sample_id}")
            continue
        x1, y1, x2, y2 = [int(value) for value in boxes[0]]
        expected = cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_LINEAR)
        actual = np.asarray(np.load(Path(str(row["static_face"]["path"])), allow_pickle=False))
        if actual.shape != expected.shape or not np.array_equal(actual, expected) or file_sha256(Path(str(row["static_face"]["path"]))) != str(row["static_face"]["sha256"]):
            errors.append(f"static_face:{sample_id}")
    return errors


def _check_media(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    media = Path(str(row["output"]))
    audio = Path(str(row["audio"]))
    try:
        frames = decode_video_frames(media)
        pcm = decode_pcm16(media)
        expected = source_pcm16(audio)
    except Exception as exc:  # noqa: BLE001 - validator reports the artifact failure
        return [f"media_decode:{row.get('sample_id')}:{exc}"]
    if len(frames) != config.WAV2LIP_FRAMES or any(frame.shape != (224, 224, 3) or frame.dtype != np.uint8 for frame in frames):
        errors.append(f"media_timeline:{row.get('sample_id')}:{row.get('arm')}")
    if pcm != expected:
        errors.append(f"media_pcm:{row.get('sample_id')}:{row.get('arm')}")
    if str(row.get("output_sha256")) != file_sha256(media):
        errors.append(f"media_hash:{row.get('sample_id')}:{row.get('arm')}")
    return errors


def _score_metrics(row: dict[str, Any]) -> dict[str, Any]:
    matrix_path = Path(str(row["matrix"]))
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (config.MATRIX_ROWS, 31) or str(row["matrix_sha256"]) != file_sha256(matrix_path):
        raise ProtocolError(f"score matrix invalid: {matrix_path}")
    selected = matrix[config.U_START : config.U_STOP]
    curve = np.mean(selected, axis=0)
    index = int(np.argmin(curve))
    return {"C": float(np.median(curve) - curve[index]), "D": float(curve[index]), "A_curve": curve, "min_index": index, "offset": config.VSHIFT - index}


def _bootstrap(values: list[float], groups: list[str], indices: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[group].append(float(value))
    labels = sorted(grouped)
    means = np.asarray([np.mean(grouped[label]) for label in labels], dtype=np.float64)
    estimates = np.mean(means[indices], axis=1)
    return {"mean": float(np.mean(means)), "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))], "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}}


def _check_analysis(root: Path, protocol: dict[str, Any], control_scores: list[dict[str, Any]], candidate_scores: list[dict[str, Any]], driver_manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    analysis_path = root / "analysis.json"
    if not analysis_path.is_file():
        return ["analysis_missing"]
    analysis = verify_self_hashed_json(analysis_path)
    by_key = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in control_scores + candidate_scores}
    groups = [str(record["source_group"]) for record in protocol["records"]]
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    indices = rng.integers(0, config.EXPECTED_GROUP_COUNT, size=(config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT), endpoint=False)
    baseline = {(str(row["sample_id"]), "N", "N"): row for row in control_scores if str(row.get("video_arm")) == "N"}
    values = {name: {metric: [] for metric in ("C", "D", "A")} for name in ("CORRECT_vs_N", "CORRECT_vs_WRONG", "CORRECT_vs_SHUFFLE")}
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        n = _score_metrics(baseline[(sample_id, "N", "N")])
        c = _score_metrics(by_key[(sample_id, "CORRECT", "N")])
        w = _score_metrics(by_key[(sample_id, "WRONG", "N")])
        s = _score_metrics(by_key[(sample_id, "SHUFFLE", "N")])
        k = n["min_index"]
        for name, left, right in (("CORRECT_vs_N", c, n), ("CORRECT_vs_WRONG", c, w), ("CORRECT_vs_SHUFFLE", c, s)):
            values[name]["C"].append(left["C"] - right["C"])
            values[name]["D"].append(right["D"] - left["D"])
            values[name]["A"].append(right["A_curve"][k] - left["A_curve"][k])
    for name, metrics in values.items():
        expected = analysis["contrasts"][name]["metrics"]
        for metric, value in metrics.items():
            actual = _bootstrap(value, groups, indices)
            for key in ("mean",):
                if abs(float(actual[key]) - float(expected[metric][key])) > 1e-6:
                    errors.append(f"analysis_{name}_{metric}_{key}")
            for index, key in enumerate(("low", "high")):
                if abs(float(actual["ci95"][index]) - float(expected[metric]["ci95"][index])) > 1e-6:
                    errors.append(f"analysis_{name}_{metric}_{key}")
    return errors


def validate(run_root: Path, stage: str) -> dict[str, Any]:
    errors: list[str] = []
    protocol = verify_self_hashed_json(run_root / "protocol.json")
    drivers = verify_self_hashed_json(run_root / "drivers/manifest.json")
    if protocol.get("record_count") != config.EXPECTED_RECORD_COUNT or protocol.get("source_group_count") != config.EXPECTED_GROUP_COUNT:
        errors.append("protocol_counts")
    if len(drivers.get("rows", [])) != config.EXPECTED_RECORD_COUNT:
        errors.append("driver_counts")
    parent_index = _parent_driver_index()
    with np.load(config.NATURAL_MELS, allow_pickle=False) as archive:
        for row in drivers["rows"]:
            sample_id = str(row["sample_id"])
            natural = np.asarray(archive[sample_id], dtype=np.float32)
            columns = [int(value) for value in row["construct"]["K"]]
            expected = _expected_arms(sample_id, natural, parent_index, columns)
            for arm, value in expected.items():
                path = Path(str(row["arms"][arm]["path"]))
                actual = np.asarray(np.load(path, allow_pickle=False))
                if actual.dtype != np.dtype("float32") or not np.array_equal(actual, value) or file_sha256(path) != str(row["arms"][arm]["sha256"]):
                    errors.append(f"driver:{sample_id}:{arm}")
    errors.extend(_check_static_faces(protocol, drivers))
    control_scores: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    control_media_path = run_root / "media/control_manifest.json"
    if control_media_path.is_file():
        control_media = verify_self_hashed_json(control_media_path)
        for row in control_media.get("rows", []):
            errors.extend(_check_media(row))
    else:
        errors.append("control_media_manifest_missing")
    control_score_path = run_root / "control_scores/manifest.json"
    if control_score_path.is_file():
        control_scores = list(verify_self_hashed_json(control_score_path).get("rows", []))
        if len(control_scores) != config.EXPECTED_CONTROL_SCORE_COUNT:
            errors.append("control_score_count")
        for row in control_scores:
            nested = row.get("score", row)
            if str(row.get("video_arm", "")).startswith("PARITY"):
                matrix = np.asarray(np.load(Path(str(nested["matrix"])), allow_pickle=False), dtype=np.float64)
                if matrix.ndim != 2 or matrix.shape[1] != 31:
                    errors.append(f"parity_matrix:{row.get('sample_id')}")
            else:
                try:
                    _score_metrics(nested)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"control_score:{row.get('sample_id')}:{exc}")
    else:
        errors.append("control_score_manifest_missing")
    if stage == "all":
        candidate_media_path = run_root / "media/candidate_manifest.json"
        if candidate_media_path.is_file():
            candidate_media = verify_self_hashed_json(candidate_media_path)
            for row in candidate_media.get("rows", []):
                errors.extend(_check_media(row))
        else:
            errors.append("candidate_media_manifest_missing")
        candidate_score_path = run_root / "candidate_scores/manifest.json"
        if candidate_score_path.is_file():
            candidate_scores = list(verify_self_hashed_json(candidate_score_path).get("rows", []))
            if len(candidate_scores) != config.EXPECTED_CANDIDATE_SCORE_COUNT:
                errors.append("candidate_score_count")
            for row in candidate_scores:
                try:
                    _score_metrics(row.get("score", row))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"candidate_score:{row.get('sample_id')}:{exc}")
        else:
            errors.append("candidate_score_manifest_missing")
        if candidate_scores:
            errors.extend(_check_analysis(run_root, protocol, control_scores, candidate_scores, drivers))
    status = "PASS" if not errors else "FAIL"
    return {"schema_version": 1, "status": status, "stage": stage, "error_count": len(errors), "errors": errors, "checked": {"record_count": len(drivers.get("rows", [])), "control_score_count": len(control_scores), "candidate_score_count": len(candidate_scores)}, "independent": True, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent validator for the natural-content residual probe")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("controls", "all"), default="all")
    args = parser.parse_args(argv)
    output = args.run_root / ("control_validation.json" if args.stage == "controls" else "validation.json")
    try:
        result = validate(args.run_root, args.stage)
    except Exception as exc:  # noqa: BLE001 - preserve a truthful validator artifact
        result = {"schema_version": 1, "status": "FAIL", "stage": args.stage, "error_count": 1, "errors": [f"validator_exception:{type(exc).__name__}:{exc}"], "independent": True, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}
    write_self_hashed_json(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
