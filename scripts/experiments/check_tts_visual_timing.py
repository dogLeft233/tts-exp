#!/usr/bin/env python3
"""Independent evidence checker for ``tts_visual_timing_v1``.

The checker intentionally duplicates the small scoring rules instead of
importing the runner or its metrics module.  Its job is to detect stale,
edited, or internally inconsistent artifacts, not to produce a second
scientific interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


REQUIRED = (13, 14, 61, 291, 33, 263)
PENALTY = 0.240
FPS = 25.0
WORD_COVERAGE = 0.80
SUPPORT_EPSILON = 1e-6


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read_json_from_text(value: str) -> Any:
    return json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_binding(binding: Mapping[str, Any], failures: List[str], label: str) -> None:
    path = Path(str(binding.get("path", "")))
    if not path.is_file():
        failures.append("bound input missing: %s" % label)
        return
    expected = binding.get("sha256")
    if expected and sha256_file(path) != expected:
        failures.append("bound input hash mismatch: %s" % label)


def _distance(reference: Sequence[Mapping[str, Any]], prediction: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ref = [float(item["time_s"]) for item in reference]
    pred = [float(item["time_s"]) for item in prediction]
    if not ref and not pred:
        raise ValueError("two empty event sets")
    n, m = len(ref), len(pred)
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    choice = [["" for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + PENALTY
        choice[i][0] = "delete_reference"
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + PENALTY
        choice[0][j] = "delete_prediction"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            values = (dp[i - 1, j - 1] + 2.0 * min(abs(ref[i - 1] - pred[j - 1]), PENALTY), dp[i - 1, j] + PENALTY, dp[i, j - 1] + PENALTY)
            best = min(values)
            if math.isclose(values[0], best, abs_tol=1e-12, rel_tol=0.0):
                choice[i][j] = "match"
            elif math.isclose(values[1], best, abs_tol=1e-12, rel_tol=0.0):
                choice[i][j] = "delete_reference"
            else:
                choice[i][j] = "delete_prediction"
            dp[i, j] = best
    matches: List[Dict[str, Any]] = []
    missing = extra = 0
    i, j = n, m
    while i or j:
        if choice[i][j] == "match":
            matches.append({"reference_index": i - 1, "prediction_index": j - 1, "error_s": abs(ref[i - 1] - pred[j - 1])})
            i -= 1; j -= 1
        elif choice[i][j] == "delete_reference":
            missing += 1; i -= 1
        elif choice[i][j] == "delete_prediction":
            extra += 1; j -= 1
        elif i:
            missing += 1; i -= 1
        else:
            extra += 1; j -= 1
    matches.reverse()
    denominator = n + m
    return {"distance_ms": float(dp[n, m] / denominator * 1000.0), "reference_count": n, "prediction_count": m, "matched_count": len(matches), "missing_reference_count": missing, "extra_prediction_count": extra, "matches": matches}


def _independent_events(landmarks: np.ndarray, valid: np.ndarray, timestamps: np.ndarray, width: float, height: float) -> Dict[str, Any]:
    if landmarks.ndim != 3 or landmarks.shape[1:] != (478, 2):
        raise ValueError("invalid landmark array shape")
    points = np.asarray(landmarks, dtype=np.float64).copy()
    points[:, :, 0] *= float(width); points[:, :, 1] *= float(height)
    finite = np.isfinite(points[:, REQUIRED, :]).all(axis=(1, 2)) & np.asarray(valid, dtype=bool)
    aperture = np.full(len(points), np.nan, dtype=np.float64)
    if finite.any():
        indices = np.flatnonzero(finite)
        denominator = np.linalg.norm(points[indices, 263] - points[indices, 33], axis=1)
        ok = denominator > 1e-8
        aperture[indices[ok]] = np.linalg.norm(points[indices[ok], 14] - points[indices[ok], 13], axis=1) / denominator[ok]
        finite[indices[~ok]] = False
    invalid_run = current = 0
    for value in finite:
        if value: current = 0
        else: current += 1; invalid_run = max(invalid_run, current)
    if not len(points) or float(np.mean(finite)) < 0.95 or invalid_run > 5:
        return {"status": "DETECTION_FAILURE", "events": [], "valid_fraction": float(np.mean(finite)) if len(points) else 0.0}
    values = aperture[finite]
    p10, p90 = float(np.percentile(values, 10)), float(np.percentile(values, 90))
    spread = p90 - p10
    if spread < 0.015:
        return {"status": "LOW_MOTION", "events": [], "p10": p10, "p90": p90, "valid_fraction": float(np.mean(finite))}
    smooth = aperture.copy()
    for index in range(1, len(smooth) - 1):
        if finite[index - 1] and finite[index] and finite[index + 1]:
            smooth[index] = float(np.median(aperture[index - 1:index + 2]))
    z = np.full(len(smooth), np.nan, dtype=np.float64); z[finite] = (smooth[finite] - p10) / spread
    candidates: List[Dict[str, Any]] = []
    index = 0
    while index < len(z):
        if not finite[index] or z[index] > 0.25:
            index += 1; continue
        start = index
        while index + 1 < len(z) and finite[index + 1] and z[index + 1] <= 0.25: index += 1
        end = index
        if end - start + 1 <= 8:
            left, right = max(0, start - 5), min(len(z), end + 6)
            if finite[left:right].all() and np.any(z[left:start] >= 0.60) and np.any(z[end + 1:right] >= 0.60):
                minimum = float(np.min(smooth[start:end + 1])); minima = [i for i in range(start, end + 1) if smooth[i] == minimum]; position_frames = 0.5 * (minima[0] + minima[-1]); chosen = int(math.floor(position_frames + 0.5))
                candidates.append({"index": chosen, "event_position_frames": float(position_frames), "time_s": float(0.5 * (timestamps[minima[0]] + timestamps[minima[-1]])), "aperture": float(smooth[chosen]), "start_index": start, "end_index": end, "z": float(z[chosen])})
        index += 1
    selected: List[Dict[str, Any]] = []
    for item in candidates:
        if selected and item["time_s"] - selected[-1]["time_s"] < 0.200:
            if (item["aperture"], item["time_s"]) < (selected[-1]["aperture"], selected[-1]["time_s"]): selected[-1] = item
        else: selected.append(item)
    return {"status": "MEASURABLE", "events": selected, "p10": p10, "p90": p90, "valid_fraction": float(np.mean(finite))}


def _compare_events(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Optional[str]:
    if expected.get("status") != actual.get("status"):
        return "status mismatch"
    left, right = expected.get("events", []), actual.get("events", [])
    if len(left) != len(right):
        return "event count mismatch"
    for index, (one, two) in enumerate(zip(left, right)):
        for key in ("index", "start_index", "end_index"):
            if int(one.get(key, -1)) != int(two.get(key, -1)):
                return "event %d %s mismatch" % (index, key)
        if abs(float(one["time_s"]) - float(two["time_s"])) > 1e-9:
            return "event %d time mismatch" % index
    return None


def _usable_token(token: Mapping[str, Any]) -> bool:
    try:
        return bool(token.get("label")) and not bool(token.get("silence")) and not bool(token.get("unknown")) and float(token.get("end_s", 0.0)) > float(token.get("start_s", 0.0))
    except (TypeError, ValueError):
        return False


def _independent_mapping(n_tokens: Sequence[Mapping[str, Any]], t_tokens: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def groups(tokens: Sequence[Mapping[str, Any]]) -> Optional[List[tuple[str, List[Mapping[str, Any]]]]]:
        grouped: Dict[int, List[Mapping[str, Any]]] = {}
        for token in (item for item in tokens if _usable_token(item)):
            if token.get("word_index") is None or not token.get("word_normalized"):
                return None
            grouped.setdefault(int(token["word_index"]), []).append(token)
        return [(str(items[0].get("word_normalized", "")).strip().lower(), items) for _, items in sorted(grouped.items())]

    left, right = groups(n_tokens), groups(t_tokens)
    if left is None or right is None:
        return {"status": "ALIGNMENT_UNSUPPORTED", "reason": "WORDS_TIER_REQUIRED", "segments": [], "support_blocks": []}
    if len(left) != len(right) or any(one[0] != two[0] for one, two in zip(left, right)):
        return {"status": "ALIGNMENT_UNSUPPORTED", "reason": "WORD_ORDINAL_MISMATCH", "segments": [], "support_blocks": []}

    def lcs(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> List[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        labels_a = [str(item.get("normalized", item.get("label", ""))).strip().lower() for item in a]
        labels_b = [str(item.get("normalized", item.get("label", ""))).strip().lower() for item in b]
        cache: Dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        def solve(i: int, j: int) -> tuple[tuple[int, int], ...]:
            if i >= len(a) or j >= len(b):
                return ()
            key = (i, j)
            if key in cache:
                return cache[key]
            candidates = [solve(i + 1, j), solve(i, j + 1)]
            if labels_a[i] and labels_a[i] == labels_b[j]:
                candidates.append(((i, j),) + solve(i + 1, j + 1))
            best_len = max(len(value) for value in candidates)
            cache[key] = min((value for value in candidates if len(value) == best_len), key=lambda value: tuple(value))
            return cache[key]
        return [(a[i], b[j]) for i, j in solve(0, 0)]

    segments: List[Dict[str, Any]] = []
    for (left_label, left_group), (right_label, right_group) in zip(left, right):
        if left_label != right_label:
            continue
        for n_token, t_token in lcs(left_group, right_group):
            n0, n1 = float(n_token["start_s"]), float(n_token["end_s"])
            t0, t1 = float(t_token["start_s"]), float(t_token["end_s"])
            if n1 > n0 and t1 > t0:
                segments.append({"n_start_s": n0, "n_end_s": n1, "t_start_s": t0, "t_end_s": t1, "slope": (t1 - t0) / (n1 - n0), "phone": str(n_token.get("normalized", n_token.get("label", ""))).strip().lower()})
    blocks: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if blocks and abs(float(segment["n_start_s"]) - float(blocks[-1]["n_end_s"])) <= SUPPORT_EPSILON and abs(float(segment["t_start_s"]) - float(blocks[-1]["t_end_s"])) <= SUPPORT_EPSILON:
            blocks[-1]["n_end_s"] = segment["n_end_s"]; blocks[-1]["t_end_s"] = segment["t_end_s"]; blocks[-1]["segment_indices"].append(index)
        else:
            blocks.append({"n_start_s": segment["n_start_s"], "n_end_s": segment["n_end_s"], "t_start_s": segment["t_start_s"], "t_end_s": segment["t_end_s"], "segment_indices": [index]})
    def coverage(tokens: Sequence[Mapping[str, Any]], side: str) -> float:
        total = sum(float(item["end_s"]) - float(item["start_s"]) for item in tokens if _usable_token(item))
        covered = sum(float(item[side + "_end_s"]) - float(item[side + "_start_s"]) for item in segments)
        return float(covered / total) if total > 0 else 0.0
    n_coverage, t_coverage = coverage(n_tokens, "n"), coverage(t_tokens, "t")
    status = "COMPLETE" if segments and n_coverage >= WORD_COVERAGE and t_coverage >= WORD_COVERAGE else "INSUFFICIENT_COVERAGE"
    return {"status": status, "reason": None if status == "COMPLETE" else "MATCHED_SPEECH_COVERAGE_BELOW_0_80", "segments": segments, "support_blocks": blocks, "n_speech_coverage": n_coverage, "t_speech_coverage": t_coverage}


def _event_position(event: Mapping[str, Any]) -> float:
    return float(event.get("event_position_frames", event.get("index")))


def _inverse_event_time(source_position: float, mapping: Sequence[int]) -> Optional[float]:
    left, right = int(math.floor(source_position)), int(math.ceil(source_position))
    if left == right:
        positions = [index for index, value in enumerate(mapping) if int(value) == left]
        return float(0.5 * (positions[0] + positions[-1]) / FPS) if positions else None
    if left < 0 or right >= len(mapping):
        return None
    left_positions = [index for index, value in enumerate(mapping) if int(value) == left]
    right_positions = [index for index, value in enumerate(mapping) if int(value) == right]
    if not left_positions or not right_positions:
        return None
    ltime = 0.5 * (left_positions[0] + left_positions[-1]) / FPS
    rtime = 0.5 * (right_positions[0] + right_positions[-1]) / FPS
    return float(ltime + (source_position - left) * (rtime - ltime))


def _recompute_v2_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    refs = [dict(item) for item in record.get("reference_events", record.get("summary", {}).get("reference_events", []))]
    interval = tuple(float(value) for value in record.get("source_interval_s", record.get("summary", {}).get("source_interval_s", [0.0, 0.0])))
    if len(refs) < 2:
        return {"status": "INSUFFICIENT_SUPPORT", "passed": False, "checks": {}, "reason": "REFERENCE_EVENT_COUNT_LT_2"}
    arms: Dict[str, Any] = {}
    for name in ("REPEAT", "LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160", "FROZEN"):
        raw = record.get("arms", {}).get(name, {})
        mapping = [int(value) for value in raw.get("mapping", [])]
        events = [dict(item) for item in raw.get("events", [])]
        supported: List[Dict[str, Any]] = []
        for event in events:
            position = _event_position(event); lo, hi = int(math.floor(position)), int(math.ceil(position))
            if lo < 0 or hi >= len(mapping):
                continue
            source = float(mapping[lo]) if lo == hi else float(mapping[lo]) + (position - lo) * float(mapping[hi] - mapping[lo])
            if interval[0] <= source / FPS < interval[1]:
                supported.append(event)
        expected: List[Dict[str, Any]] = []
        unsupported: List[int] = []
        for index, event in enumerate(refs):
            value = _inverse_event_time(_event_position(event), mapping)
            if value is None:
                unsupported.append(index)
            else:
                expected.append({"reference_index": index, "time_s": value})
        distance = _distance(expected, supported) if expected or supported else None
        good = [match for match in (distance or {}).get("matches", []) if float(match["error_s"]) <= 0.040]
        shifts = [float(supported[int(match["prediction_index"])]["time_s"]) - float(refs[int(expected[int(match["reference_index"])]["reference_index"])]["time_s"]) for match in good]
        delta_s = raw.get("delta_s")
        direction_ok = True if delta_s is None or abs(float(delta_s)) <= 1e-12 else bool(shifts and np.median(shifts) * (-1 if float(delta_s) > 0 else 1) > 0)
        arms[name] = {"status": raw.get("status"), "distance_to_real_ms": None if distance is None else float(_distance(refs, supported)["distance_ms"] if supported or refs else 0.0), "recovery_rate": float(len(good) / len(refs)) if refs else None, "median_abs_error_ms": float(np.median([float(match["error_s"]) for match in good]) * 1000.0) if good else None, "direction_ok": direction_ok, "unsupported_reference_indices": unsupported}
    repeat_ok = arms["REPEAT"]["distance_to_real_ms"] is not None and arms["REPEAT"]["distance_to_real_ms"] <= 20.0
    warp_ok = all(arms[name]["recovery_rate"] is not None and arms[name]["recovery_rate"] >= 0.8 and arms[name]["median_abs_error_ms"] is not None and arms[name]["median_abs_error_ms"] <= 40.0 for name in ("LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160"))
    sign_ok = all(arms[name]["direction_ok"] for name in ("LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160"))
    order_ok = all(arms[large]["distance_to_real_ms"] is not None and arms[small]["distance_to_real_ms"] is not None and arms[large]["distance_to_real_ms"] > arms[small]["distance_to_real_ms"] > arms["REPEAT"]["distance_to_real_ms"] for large, small in (("LOCAL_+160", "LOCAL_+80"), ("LOCAL_-160", "LOCAL_-80")))
    frozen_ok = arms["FROZEN"]["status"] == "LOW_MOTION"
    checks = {"repeat_e_le_20ms": repeat_ok, "warp_recovery_and_error": warp_ok, "sign_correct": sign_ok, "monotone_error": order_ok, "frozen_not_perfect": frozen_ok}
    passed = bool(all(checks.values()))
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "checks": checks, "arms": arms, "reason": None if passed else "CALIBRATION_GATE_FAILED"}


def check_run(run_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    checks: Dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        result = {"protocol": "tts_visual_timing_v1", "status": "FAIL", "failures": ["manifest missing"], "checks": {}}
        write_json(run_dir / "validation.json", result)
        return result
    manifest = read_json(manifest_path)
    protocol = str(manifest.get("protocol", "tts_visual_timing_v1"))
    checks["protocol"] = protocol in ("tts_visual_timing_v1", "tts_visual_timing_v2")
    if not checks["protocol"]: failures.append("protocol mismatch")
    if protocol == "tts_visual_timing_v2":
        protocol_path = run_dir / "protocol.json"
        if not protocol_path.is_file():
            failures.append("protocol.json missing")
        else:
            protocol_payload = read_json(protocol_path)
            if protocol_payload.get("protocol") != protocol or protocol_payload.get("identity_hash") != manifest.get("identity_hash"):
                failures.append("protocol identity mismatch")
        if not (run_dir / "diagnosis.json").is_file():
            failures.append("diagnosis missing")
    if not (run_dir / "environment.json").is_file(): failures.append("environment missing")
    for label, binding in manifest.get("bindings", {}).items():
        if isinstance(binding, Mapping) and "path" in binding:
            _check_binding(binding, failures, label)
        elif label == "code" and isinstance(binding, list):
            for index, item in enumerate(binding):
                if isinstance(item, Mapping):
                    _check_binding(item, failures, "code[%d]" % index)
                else:
                    failures.append("code binding is not an object: %d" % index)
    for group_name in ("main_records", "calibration_records"):
        for index, row in enumerate(manifest.get(group_name, [])):
            for label, binding in row.get("input_hashes", {}).items():
                if isinstance(binding, Mapping):
                    _check_binding(binding, failures, "%s[%d].%s" % (group_name, index, label))
    feature_count = 0
    event_count = 0
    for feature_path in sorted((run_dir / "features").glob("*.npz")):
        feature_count += 1
        meta_path = feature_path.with_suffix(".json")
        if not meta_path.is_file():
            failures.append("metadata missing: %s" % feature_path.name); continue
        meta = read_json(meta_path)
        if meta.get("npz_sha256") != sha256_file(feature_path):
            failures.append("feature hash mismatch: %s" % feature_path.name)
        with np.load(feature_path, allow_pickle=False) as data:
            arrays = {"landmarks": np.asarray(data["landmarks"]), "valid": np.asarray(data["valid"], dtype=bool), "timestamps_s": np.asarray(data["timestamps_s"], dtype=np.float64)}
        event_path = run_dir / "events" / (feature_path.stem + ".json")
        if not event_path.is_file():
            failures.append("event artifact missing: %s" % feature_path.stem); continue
        event_count += 1
        previews = meta.get("previews", [])
        if len(previews) < 3 or any(not Path(str(item.get("path", ""))).is_file() or (item.get("sha256") and sha256_file(Path(str(item["path"]))) != item.get("sha256")) for item in previews):
            failures.append("preview audit failed: %s" % feature_path.name)
        pts_qc = meta.get("pts_qc", {})
        if pts_qc and (not pts_qc.get("count_matches_decoded") or not pts_qc.get("strictly_increasing")):
            failures.append("PTS audit failed: %s" % feature_path.name)
        expected = read_json(event_path)
        if meta.get("status") == "MULTIPLE_FACES":
            # The main runner intentionally makes a whole multi-face video
            # unmeasurable; do not silently turn it into a normal sequence.
            continue
        try:
            actual = _independent_events(arrays["landmarks"], arrays["valid"], arrays["timestamps_s"], float(meta["width"]), float(meta["height"]))
            mismatch = _compare_events(expected, actual)
            if mismatch:
                failures.append("%s: %s" % (event_path.name, mismatch))
        except Exception as exc:
            failures.append("%s: independent score failed: %s" % (feature_path.name, exc))
    checks["feature_count"] = feature_count
    checks["event_count"] = event_count
    calibration_path = run_dir / "calibration.json"
    if calibration_path.is_file():
        calibration = read_json(calibration_path)
        expected_cells = 7 if manifest.get("smoke") else 28
        cell_path = run_dir / "cells.jsonl"
        cells = []
        if not cell_path.is_file():
            failures.append("cells.jsonl missing")
        else:
            for line_no, line in enumerate(cell_path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    cells.append(read_json_from_text(line))
                except Exception as exc:
                    failures.append("invalid cell line %d: %s" % (line_no, exc))
            required_cell_fields = ("stage", "id", "source_group", "arm", "input_hash", "config_hash", "status", "reason", "artifacts")
            allowed_statuses = {"COMPLETE", "PARTIAL", "FAILED", "DEPENDENCY_BLOCKED", "SKIPPED_BY_GATE"}
            for line_no, row in enumerate(cells, 1):
                if not isinstance(row, Mapping):
                    failures.append("cell line %d is not an object" % line_no)
                    continue
                missing = [field for field in required_cell_fields if field not in row]
                if missing:
                    failures.append("cell line %d missing cell fields: %s" % (line_no, ",".join(missing)))
                if row.get("status") not in allowed_statuses:
                    failures.append("cell line %d has invalid status: %s" % (line_no, row.get("status")))
                if row.get("status") in {"FAILED", "DEPENDENCY_BLOCKED", "SKIPPED_BY_GATE"} and not row.get("reason"):
                    failures.append("cell line %d missing failure reason" % line_no)
                if not isinstance(row.get("artifacts"), list):
                    failures.append("cell line %d artifacts is not a list" % line_no)
            keys = {(str(row.get("stage")), str(row.get("id")), str(row.get("arm"))) for row in cells if isinstance(row, Mapping)}
            calibration_keys = {("calibrate", str(row.get("id")), str(row.get("arm"))) for row in cells if isinstance(row, Mapping) and row.get("stage") == "calibrate"}
            checks["calibration_cell_count"] = len(calibration_keys)
            if protocol == "tts_visual_timing_v2":
                expected_ids = [str(value) for value in manifest.get("selected_calibration_ids", [])]
                expected_arms = ("REAL", "REPEAT", "LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160", "FROZEN")
                expected_keys = {("calibrate", sid, arm) for sid in expected_ids for arm in expected_arms}
                if calibration_keys != expected_keys:
                    failures.append("calibration cells do not match exact v2 key set: %d/%d" % (len(calibration_keys & expected_keys), len(expected_keys)))
            elif len(calibration_keys) < expected_cells:
                failures.append("calibration cells incomplete: %d/%d" % (len(calibration_keys), expected_cells))
        for row in calibration.get("records", []):
            summary = row.get("summary", {})
            for arm, item in summary.get("arms", {}).items():
                mapping = item.get("mapping")
                if mapping is not None:
                    if not isinstance(mapping, list) or not mapping or any(int(value) < 0 for value in mapping) or any(int(right) < int(left) for left, right in zip(mapping, mapping[1:])):
                        failures.append("invalid calibration mapping: %s:%s" % (row.get("sample_id"), arm))
            if row.get("status") == "PASS":
                checks_value = row.get("checks", {})
                recomputed = all(bool(checks_value.get(name)) for name in ("repeat_e_le_20ms", "warp_recovery_and_error", "sign_correct", "monotone_error", "frozen_not_perfect"))
                if bool(row.get("passed")) != recomputed:
                    failures.append("calibration decision mismatch: %s" % row.get("sample_id"))
            if protocol == "tts_visual_timing_v2":
                recomputed = _recompute_v2_record(row)
                if bool(row.get("passed")) != bool(recomputed.get("passed")):
                    failures.append("v2 calibration passed mismatch: %s" % row.get("sample_id"))
                if row.get("checks") != recomputed.get("checks"):
                    failures.append("v2 calibration checks mismatch: %s" % row.get("sample_id"))
                saved_summary = row.get("summary", {})
                for name, arm in recomputed.get("arms", {}).items():
                    saved_arm = saved_summary.get("arms", {}).get(name, {}) if isinstance(saved_summary, Mapping) else {}
                    if saved_arm and saved_arm.get("mapping") is not None and [int(value) for value in saved_arm.get("mapping", [])] != [int(value) for value in row.get("arms", {}).get(name, {}).get("mapping", [])]:
                        failures.append("v2 calibration mapping disagreement: %s:%s" % (row.get("sample_id"), name))
    else:
        failures.append("calibration missing")
    required_artifacts = ("vsr_audit.json", "native.json", "generation.json", "analysis.json", "report.md")
    if protocol == "tts_visual_timing_v2":
        required_artifacts = ("diagnosis.json",) + required_artifacts
    for name in required_artifacts:
        checks[name] = (run_dir / name).is_file()
        if not checks[name]: failures.append("artifact missing: %s" % name)
    result = {"protocol": protocol, "status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks, "independent_recompute": {"features": feature_count, "events": event_count, "artifact_integrity": "PASS" if not failures else "FAIL", "calibration_recompute": "PASS" if (run_dir / "calibration.json").is_file() and protocol == "tts_visual_timing_v2" and not any("calibration" in item or "v2 calibration" in item for item in failures) else ("NOT_RUN" if not (run_dir / "calibration.json").is_file() else "FAIL"), "native_recompute": "NOT_RUN", "generation_recompute": "NOT_RUN"}}
    write_json(run_dir / "validation.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check_run(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
