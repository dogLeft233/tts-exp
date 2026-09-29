"""Independent validator for the fresh-source visual branch.

This file intentionally reimplements the small numerical route instead of
calling the producer's analysis/gate functions.  It is therefore able to
reject an edited-and-resigned ``analysis.json`` while sharing only the raw
feature-file I/O contract.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.fresh_source_inputs.protocol import (
    ProtocolError,
    file_sha256,
    load_self_hashed,
    write_json,
)
from scripts.experiments.fresh_source_visual.common import (
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    CALIBRATION_ERROR_THRESHOLD,
    FRAME_COUNT,
    FRAME_TOLERANCE_S,
    J0,
    MIN_OBSERVED_FRAMES,
    MODELS,
    VisualProtocolError,
    load_feature,
)


def _root(run_root: Path) -> Path:
    return run_root.parent.parent if run_root.name == "shared" else run_root


def _branch(run_root: Path) -> Path:
    return _root(run_root.resolve()) / "run" / "B"


def _manifest(root: Path, name: str) -> Path | None:
    candidates = [root / "run/shared" / name, root / "shared" / name, root / name]
    if root.name == "shared":
        candidates.insert(0, root / name)
    return next((path for path in candidates if path.is_file()), None)


def _load(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return load_self_hashed(path)
    except (OSError, ProtocolError, ValueError) as exc:
        errors.append(f"{path}: {exc}")
        return {}


def _records(root: Path, errors: list[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    cohort_path = _manifest(root, "cohort.json")
    inputs_path = _manifest(root, "inputs.json")
    if cohort_path is None or inputs_path is None:
        errors.append("missing frozen cohort/inputs manifest")
        return [], {}
    cohort = _load(cohort_path, errors)
    inputs = _load(inputs_path, errors)
    groups = [str(item.get("source_group")) for item in cohort.get("formal", []) if isinstance(item, dict)]
    rows = {str(item.get("source_group")): dict(item) for item in inputs.get("records", []) if isinstance(item, dict)}
    if len(groups) != 12 or len(set(groups)) != 12:
        errors.append("frozen formal cohort is not 12 unique groups")
    if set(groups) - set(rows):
        errors.append("inputs missing a frozen formal group")
    return groups, rows


def _feature_index(branch: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    path = branch / "features.json"
    if not path.is_file():
        errors.append("missing B/features.json")
        return {}, {}
    index = _load(path, errors)
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for item in index.get("rows", []):
        if not isinstance(item, dict):
            errors.append("malformed feature index row")
            continue
        group, arm = str(item.get("source_group")), str(item.get("arm"))
        key = (group, arm)
        if key in seen:
            # The real trajectory is shared by the two generator analyses.
            # Repeated identical R rows are harmless; conflicting duplicates
            # remain an integrity error.
            if arm == "R":
                prior = by_group.get(group, {}).get(arm)
                prior_entry = prior.get("entry") if isinstance(prior, Mapping) else None
                if isinstance(prior_entry, Mapping) and prior_entry.get("path") == item.get("path") and prior_entry.get("sha256") == item.get("sha256"):
                    continue
            errors.append(f"duplicate feature index row: {group}/{arm}")
            continue
        seen.add(key)
        feature_path = Path(str(item.get("path", "")))
        if not feature_path.is_file():
            errors.append(f"missing feature: {group}/{arm}")
            continue
        actual_sha = file_sha256(feature_path)
        if str(item.get("sha256")) != actual_sha:
            errors.append(f"feature SHA mismatch: {group}/{arm}")
            continue
        try:
            feature = load_feature(feature_path)
        except VisualProtocolError as exc:
            errors.append(f"invalid feature {group}/{arm}: {exc}")
            continue
        if feature.frame_count < FRAME_COUNT:
            errors.append(f"feature shorter than fixed clock: {group}/{arm}")
        by_group.setdefault(group, {})[arm] = {"feature": feature, "path": feature_path, "sha256": actual_sha, "entry": item}
    for item in index.get("generated", []):
        if not isinstance(item, dict) or item.get("status") != "complete":
            continue
        group = str(item.get("source_group"))
        arm = f"{item.get('model')}:{item.get('arm')}"
        key = (group, arm)
        if key in seen:
            errors.append(f"duplicate generated feature index row: {group}/{arm}")
            continue
        seen.add(key)
        feature_path = Path(str(item.get("path", "")))
        if not feature_path.is_file() or str(item.get("sha256")) != file_sha256(feature_path):
            errors.append(f"generated feature binding changed: {group}/{arm}")
            continue
        try:
            feature = load_feature(feature_path)
        except VisualProtocolError as exc:
            errors.append(f"invalid generated feature {group}/{arm}: {exc}")
            continue
        by_group.setdefault(group, {})[arm] = {"feature": feature, "path": feature_path, "sha256": file_sha256(feature_path), "entry": item}
    return index, by_group


def _feature_provenance(
    index: Mapping[str, Any],
    by_group: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: list[str],
    records: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    asset = index.get("asset")
    if not isinstance(asset, Mapping):
        errors.append("features.json is missing extractor asset provenance")
        return
    required_asset = (
        "extractor",
        "extractor_api",
        "mediapipe_version",
        "landmarker_asset",
        "landmarker_asset_sha256",
        "landmarker_options",
        "fixed_clock_frames",
        "fps",
        "canonical_feature",
        "mouth_shape",
    )
    for key in required_asset:
        if key not in asset:
            errors.append(f"features.json asset provenance missing: {key}")
    if not isinstance(asset.get("landmarker_asset_sha256"), str) or not asset.get("landmarker_asset_sha256"):
        errors.append("features.json landmarker asset hash is not pinned")
    expected_extractor = "scripts.experiments.lrs3_tts_visual_advantage.video_features"
    for group, arms in by_group.items():
        record = records.get(group, {})
        for arm, wrapped in arms.items():
            feature = wrapped["feature"]
            metadata = feature.metadata
            label = f"{group}/{arm}"
            if feature.frame_count != FRAME_COUNT:
                errors.append(f"{label}: feature frame_count is not {FRAME_COUNT}")
            for key in required_asset:
                if key in asset and metadata.get(key) != asset.get(key):
                    errors.append(f"{label}: feature provenance mismatch for {key}")
            if metadata.get("extractor") != expected_extractor:
                errors.append(f"{label}: extractor provenance is missing or unexpected")
            if metadata.get("source_group") != group:
                errors.append(f"{label}: feature source_group binding mismatch")
            if arm == "R":
                expected_path = record.get("real_video")
                expected_sha = record.get("real_video_sha256")
                if metadata.get("source_kind") != "real":
                    errors.append(f"{label}: real feature source_kind is not real")
            else:
                expected_path = None
                expected_sha = None
                entry = wrapped.get("entry", {})
                manifest = entry.get("manifest") if isinstance(entry, Mapping) else None
                if isinstance(manifest, Mapping):
                    # Generated media is extracted from A's exact fixed-clock
                    # normalized stream when present; the muxed path remains
                    # the manifest identity used by A and is retained above.
                    expected_path = manifest.get("normalized_path") or manifest.get("path")
                    expected_sha = manifest.get("normalized_sha256") or manifest.get("sha256")
                if not str(metadata.get("source_kind", "")).startswith("generated:"):
                    errors.append(f"{label}: generated feature source_kind is missing")
            if expected_path and metadata.get("video_path") != str(Path(str(expected_path)).resolve()):
                errors.append(f"{label}: feature video_path binding mismatch")
            if expected_sha and metadata.get("video_sha256") != str(expected_sha):
                errors.append(f"{label}: feature video SHA binding mismatch")


def _a_manifest_path(root: Path) -> Path | None:
    candidates = [
        root / "run" / "A" / "videos.json",
        root / "run" / "A" / "videos_manifest.json",
        root / "run" / "replacement" / "videos.json",
        root / "videos.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _a_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: Any = payload.get("records", payload.get("rows", payload.get("videos", payload.get("cells", []))))
    if isinstance(values, Mapping):
        result: list[dict[str, Any]] = []
        for key, value in values.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("key", key)
                result.append(row)
        return result
    return [dict(value) for value in values if isinstance(value, Mapping)] if isinstance(values, list) else []


def _a_identity(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None, int | None, int]:
    sample = row.get("sample_id")
    model = row.get("model")
    arm = row.get("arm")
    seed_value = row.get("seed")
    repeat_value = row.get("repeat_index", row.get("repeat", 0))
    key = row.get("key")
    if isinstance(key, str):
        parts = key.split("/")
        if len(parts) >= 5:
            sample = sample or parts[0]
            model = model or parts[1]
            arm = arm or parts[2]
            seed_value = seed_value if seed_value is not None else parts[3]
            repeat_value = repeat_value if repeat_value is not None else parts[4]
    arm_text = str(arm or "")
    match = re.fullmatch(r"([NC])(?:42|43)", arm_text.upper().replace("_REPEAT", ""))
    if seed_value is None and match:
        seed_value = arm_text.upper().replace("_REPEAT", "")[1:]
    if "_REPEAT" in arm_text.upper() and repeat_value in (None, 0, "0"):
        repeat_value = 1
    try:
        seed = int(seed_value) if seed_value is not None else None
    except (TypeError, ValueError):
        seed = None
    try:
        repeat = int(repeat_value)
    except (TypeError, ValueError):
        repeat = -1
    return (str(sample) if sample else None, str(model).lower() if model else None, arm_text.upper().replace("_REPEAT", ""), seed, repeat)


def _expected_audio(record: Mapping[str, Any], arm: str) -> tuple[str | None, str | None, str | None]:
    if arm == "N":
        return (
            str(record.get("natural_audio")) if record.get("natural_audio") else None,
            str(record.get("natural_audio_sha256")) if record.get("natural_audio_sha256") else None,
            str(record.get("natural_pcm_sha256")) if record.get("natural_pcm_sha256") else None,
        )
    direct = record.get("direct_audio")
    if isinstance(direct, Mapping):
        return (
            str(direct.get("output_path")) if direct.get("output_path") else None,
            str(direct.get("output_sha256")) if direct.get("output_sha256") else None,
            str(direct.get("decoded_pcm_sha256")) if direct.get("decoded_pcm_sha256") else None,
        )
    candidate = record.get("candidate_audio")
    if isinstance(candidate, Mapping):
        return (
            str(candidate.get("path")) if candidate.get("path") else None,
            str(candidate.get("sha256")) if candidate.get("sha256") else None,
            str(candidate.get("pcm_sha256")) if candidate.get("pcm_sha256") else None,
        )
    return (
        str(record.get("candidate_audio")) if record.get("candidate_audio") else None,
        str(record.get("candidate_audio_sha256")) if record.get("candidate_audio_sha256") else None,
        str(record.get("candidate_pcm_sha256")) if record.get("candidate_pcm_sha256") else None,
    )


def _probe_a_video(path: Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        import json

        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping):
        return None
    fps_text = str(video.get("r_frame_rate", "0/1"))
    try:
        numerator, denominator = fps_text.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = float("nan")
    frames = video.get("nb_read_frames") or video.get("nb_frames")
    try:
        frame_count = int(frames)
    except (TypeError, ValueError):
        frame_count = 0
    return {"frame_count": frame_count, "fps": fps, "has_audio": audio is not None}


def _validate_a_manifest(
    root: Path,
    groups: list[str],
    records: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    path = _a_manifest_path(root)
    if path is None:
        return {"status": "missing", "complete_count": 0, "rows": {}}
    payload = _load(path, errors)
    rows = _a_rows(payload)
    by_key: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    by_sample = {str(record.get("sample_id")): (group, record) for group, record in records.items()}
    for row in rows:
        if str(row.get("status", "complete")).lower() not in {"complete", "pass", "ok"}:
            continue
        # C's fixed-delay forward is an additional audio-control arm.  B's
        # visual hypothesis is frozen to N/C and must not reject a valid A
        # manifest merely because DELAY rows were added later.
        raw_arm = row.get("arm")
        if raw_arm is None and isinstance(row.get("key"), str):
            key_parts = str(row["key"]).split("/")
            raw_arm = key_parts[2] if len(key_parts) >= 3 else None
        if str(raw_arm or "").upper().startswith("DELAY"):
            continue
        sample, model, arm, seed, repeat = _a_identity(row)
        if sample is None or model is None or arm is None or seed is None or repeat < 0:
            errors.append("A complete video cell has incomplete sample/model/arm/seed/repeat identity")
            continue
        group_record = by_sample.get(sample)
        if group_record is None:
            errors.append(f"A complete video cell references unknown sample: {sample}")
            continue
        group, record = group_record
        key = (sample, model, arm, seed, repeat)
        if key in by_key:
            errors.append(f"duplicate A complete cell: {key}")
            continue
        by_key[key] = dict(row)
        if model not in MODELS or arm not in {"N", "C"} or seed not in {42, 43}:
            errors.append(f"A cell has invalid model/arm/seed: {key}")
        if str(row.get("source_group")) != group:
            errors.append(f"A cell source_group mismatch: {key}")
        expected_key = f"{sample}/{model}/{arm}/{seed}/{repeat}"
        if row.get("key") is not None and str(row.get("key")) != expected_key:
            errors.append(f"A cell key mismatch: {key}")
        video_path = Path(str(row.get("path", "")))
        if not video_path.is_file():
            errors.append(f"A cell video missing: {key}")
        elif str(row.get("sha256")) != file_sha256(video_path):
            errors.append(f"A cell video SHA mismatch: {key}")
        if row.get("frame_count") != FRAME_COUNT:
            errors.append(f"A cell frame_count mismatch: {key}")
        try:
            fps = float(row.get("fps"))
        except (TypeError, ValueError):
            fps = float("nan")
        if not np.isfinite(fps) or abs(fps - 25.0) > 1e-6:
            errors.append(f"A cell fps mismatch: {key}")
        probe = _probe_a_video(video_path) if video_path.is_file() else None
        if probe is None:
            errors.append(f"A cell media probe failed: {key}")
        else:
            if probe["frame_count"] != FRAME_COUNT or abs(float(probe["fps"]) - 25.0) > 1e-6:
                errors.append(f"A cell decoded frame/fps mismatch: {key}")
            if not probe["has_audio"]:
                errors.append(f"A cell has no muxed audio: {key}")
        expected_audio, expected_audio_sha, expected_pcm_sha = _expected_audio(record, arm)
        if expected_audio and str(row.get("audio_path")) != str(Path(expected_audio).resolve()):
            errors.append(f"A cell audio path binding mismatch: {key}")
        audio_path = Path(str(row.get("audio_path", "")))
        if expected_audio_sha and audio_path.is_file() and file_sha256(audio_path) != expected_audio_sha:
            errors.append(f"A cell audio container SHA mismatch: {key}")
        if expected_pcm_sha and str(row.get("audio_pcm_sha256")) != expected_pcm_sha:
            errors.append(f"A cell decoded PCM binding mismatch: {key}")
    return {"status": str(payload.get("status", "unknown")), "complete_count": len(by_key), "rows": by_key}


def _calibration_pair(feature: Any, transform: str) -> dict[str, Any]:
    base = J0.copy()
    if transform == "identity":
        other = base.copy()
    elif transform == "shift_plus5":
        other = base + 5
    elif transform == "shift_minus5":
        other = base - 5
    elif transform == "reverse":
        other = (FRAME_COUNT - 1) - base
    else:
        raise ValueError(transform)
    in_range = (other >= 0) & (other < FRAME_COUNT) & (base < feature.frame_count)
    base, other = base[in_range], other[in_range]
    # The calibration is an index-only transform of R.  Its shifted/reversed
    # partner intentionally has a different timestamp; PTS equality is only
    # required when comparing independently extracted generated arms.
    ok = feature.valid[base] & feature.valid[other]
    base, other = base[ok], other[ok]
    error = None
    if len(base):
        left = feature.mouth[base] - np.mean(feature.mouth[base], axis=0, dtype=np.float64)
        right = feature.mouth[other] - np.mean(feature.mouth[other], axis=0, dtype=np.float64)
        error = float(np.mean((left - right) ** 2, dtype=np.float64))
    return {"transform": transform, "valid_count": len(base), "support_total": len(J0), "coverage": float(len(base) / len(J0)), "indices": base.tolist(), "other_indices": other.tolist(), "mse": error, "observed": bool(len(base) / len(J0) >= 0.90)}


def _mse_components_independent(real: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Independent copy of the decomposition (producer helpers are not used)."""
    real = np.asarray(real, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if real.shape != candidate.shape or real.ndim != 3 or not len(real):
        raise VisualProtocolError("invalid MSE arrays")
    real_mean = np.mean(real, axis=0, dtype=np.float64)
    candidate_mean = np.mean(candidate, axis=0, dtype=np.float64)
    real_centered = real - real_mean
    candidate_centered = candidate - candidate_mean
    total = float(np.mean((real - candidate) ** 2, dtype=np.float64))
    static = float(np.mean((real_mean - candidate_mean) ** 2, dtype=np.float64))
    dynamic = float(np.mean((real_centered - candidate_centered) ** 2, dtype=np.float64))
    reverse = float(np.mean((real_centered - candidate_centered[::-1]) ** 2, dtype=np.float64))
    identity_error = total - static - dynamic
    if abs(identity_error) > 1e-10 * max(1.0, total):
        raise VisualProtocolError("independent MSE decomposition identity failed")
    return {"e_total": total, "e_static": static, "e_dynamic": dynamic, "e_reverse": reverse, "identity_error": identity_error}


def _recompute_calibration(groups: list[str], by_group: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        wrapped = by_group.get(group, {}).get("R")
        if wrapped is None:
            rows.append({"source_group": group, "observed": False, "missing_reason": "real_feature_missing", "identity_mse": None, "conditions": {}})
            continue
        feature = wrapped["feature"]
        conditions = {name: _calibration_pair(feature, name) for name in ("identity", "shift_plus5", "shift_minus5", "reverse")}
        rows.append({"source_group": group, "observed": True, "identity_mse": conditions["identity"]["mse"], "identity_pass": bool(conditions["identity"]["observed"] and conditions["identity"]["mse"] is not None and conditions["identity"]["mse"] <= 1e-12), "conditions": conditions})
    counts = {}
    observed = {}
    for name in ("shift_plus5", "shift_minus5", "reverse"):
        valid = [row for row in rows if row.get("conditions", {}).get(name, {}).get("observed")]
        observed[name] = len(valid)
        counts[name] = sum(float(row["conditions"][name]["mse"] or 0.0) > CALIBRATION_ERROR_THRESHOLD for row in valid)
    identity_pass = bool(rows and all(row.get("identity_pass") for row in rows))
    distortion_pass = all(counts.get(name, 0) >= 10 for name in ("shift_plus5", "shift_minus5", "reverse"))
    support_pass = all(observed.get(name, 0) == len(rows) for name in ("shift_plus5", "shift_minus5", "reverse"))
    calibrated = bool(len(rows) == 12 and identity_pass and distortion_pass and support_pass)
    return {"rows": rows, "identity_pass": identity_pass, "distortion_counts": counts, "distortion_observed": observed, "distortion_pass": distortion_pass, "support_pass": support_pass, "status": "METRIC_CALIBRATED" if calibrated else "METRIC_NOT_CALIBRATED", "calibrated": calibrated}


def _support(named: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    base = J0.copy()
    real = named.get("real")
    if real is None:
        return np.asarray([], dtype=np.int64), {"requested_indices": base.tolist(), "requested_count": len(base), "count": 0, "indices": [], "missing_by_arm": {key: len(base) for key, value in named.items() if value is None}}
    keep = np.ones(len(base), dtype=bool)
    missing: dict[str, int] = {}
    for name, feature in named.items():
        if feature is None:
            keep[:] = False
            missing[name] = len(base)
            continue
        valid = (base < feature.frame_count)
        ok = np.zeros(len(base), dtype=bool)
        ok[valid] = feature.valid[base[valid]]
        timestamp = np.zeros(len(base), dtype=bool)
        timestamp[valid] = np.abs(feature.timestamps[base[valid]] - real.timestamps[base[valid]]) <= FRAME_TOLERANCE_S + 1e-12
        ok &= timestamp
        missing[name] = int(np.sum(~ok))
        keep &= ok
    return base[keep], {"requested_indices": base.tolist(), "requested_count": len(base), "count": int(np.sum(keep)), "indices": base[keep].tolist(), "coverage": float(np.mean(keep)), "missing_by_arm": missing, "timestamp_tolerance_s": FRAME_TOLERANCE_S}


def _decompose(real: Any, natural: Any, candidate: Any, indices: np.ndarray) -> dict[str, float]:
    n = _mse_components_independent(real.mouth[indices], natural.mouth[indices])
    c = _mse_components_independent(real.mouth[indices], candidate.mouth[indices])
    return {"b": 0.0 if n["e_dynamic"] == 0 and c["e_dynamic"] == 0 else float((n["e_dynamic"] - c["e_dynamic"]) / (n["e_dynamic"] + c["e_dynamic"] + 1e-12)), "q_n": 0.0 if n["e_reverse"] == 0 and n["e_dynamic"] == 0 else float((n["e_reverse"] - n["e_dynamic"]) / (n["e_reverse"] + n["e_dynamic"] + 1e-12)), "q_c": 0.0 if c["e_reverse"] == 0 and c["e_dynamic"] == 0 else float((c["e_reverse"] - c["e_dynamic"]) / (c["e_reverse"] + c["e_dynamic"] + 1e-12)), "n": n, "c": c}


def _model_rows(groups: list[str], by_group: dict[str, dict[str, dict[str, Any]]], model: str) -> list[dict[str, Any]]:
    result = []
    for group in groups:
        arms = by_group.get(group, {})
        named = {"real": arms.get("R", {}).get("feature") if arms.get("R") else None}
        for arm in ("N42", "C42", "N43", "C43"):
            wrapped = arms.get(f"{model}:{arm}")
            named[arm] = wrapped["feature"] if wrapped else None
        support, info = _support(named)
        row: dict[str, Any] = {"source_group": group, "observed": bool(len(support) >= MIN_OBSERVED_FRAMES), "support": info, "missing_arms": [name for name, value in named.items() if value is None]}
        if len(support) < MIN_OBSERVED_FRAMES:
            row["missing_reason"] = "common_support_below_81" if named["real"] is not None else "feature_missing"
            for key in ("b_dynamic", "q_natural", "q_candidate", "e_total_natural", "e_total_candidate", "e_static_natural", "e_static_candidate", "e_dynamic_natural", "e_dynamic_candidate", "e_reverse_natural", "e_reverse_candidate"):
                row[key] = None
            result.append(row)
            continue
        seeds = [_decompose(named["real"], named[n], named[c], support) for n, c in (("N42", "C42"), ("N43", "C43"))]
        n_components = [item["n"] for item in seeds]
        c_components = [item["c"] for item in seeds]
        row.update({
            "support_count": len(support),
            "seed_values": {"natural": {f"N{42+i}": n_components[i] for i in range(2)}, "candidate": {f"C{42+i}": c_components[i] for i in range(2)}},
            "e_total_natural": float(np.mean([item["e_total"] for item in n_components])),
            "e_total_candidate": float(np.mean([item["e_total"] for item in c_components])),
            "e_static_natural": float(np.mean([item["e_static"] for item in n_components])),
            "e_static_candidate": float(np.mean([item["e_static"] for item in c_components])),
            "e_dynamic_natural": float(np.mean([item["e_dynamic"] for item in n_components])),
            "e_dynamic_candidate": float(np.mean([item["e_dynamic"] for item in c_components])),
            "e_reverse_natural": float(np.mean([item["e_reverse"] for item in n_components])),
            "e_reverse_candidate": float(np.mean([item["e_reverse"] for item in c_components])),
            "identity_error_natural": float(np.mean([item["identity_error"] for item in n_components])),
            "identity_error_candidate": float(np.mean([item["identity_error"] for item in c_components])),
            "b_dynamic_seeds": [item["b"] for item in seeds],
            "q_natural_seeds": [item["q_n"] for item in seeds],
            "q_candidate_seeds": [item["q_c"] for item in seeds],
            "b_dynamic": float(np.mean([item["b"] for item in seeds])),
            "q_natural": float(np.mean([item["q_n"] for item in seeds])),
            "q_candidate": float(np.mean([item["q_c"] for item in seeds])),
            "missing_reason": None,
        })
        result.append(row)
    return result


def _stats(rows: list[dict[str, Any]], groups: list[str], metrics: tuple[str, ...]) -> dict[str, Any]:
    values: dict[str, list[float]] = {group: [] for group in groups}
    for row in rows:
        if row.get("observed"):
            values[str(row["source_group"])].append(row)
    active = [group for group in groups if values[group]]
    matrix = np.asarray([[float(np.mean([row[metric] for row in values[group]])) for metric in metrics] for group in active], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = matrix[rng.integers(0, len(active), size=(BOOTSTRAP_COUNT, len(active)), dtype=np.int64)].mean(axis=1) if active else np.empty((BOOTSTRAP_COUNT, len(metrics)))
    result: dict[str, Any] = {"label": "observed_only", "observed_group_count": len(active), "observed_groups": active, "bootstrap_count": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED, "quantile_method": "linear"}
    for column, metric in enumerate(metrics):
        column_values = matrix[:, column] if active else np.asarray([], dtype=np.float64)
        result[metric] = {"mean": float(np.mean(column_values)) if active else None, "positive_groups": int(np.sum(column_values > 0)) if active else 0, "observed_group_count": len(active), "ci99": [float(np.quantile(draws[:, column], 0.005, method="linear")), float(np.quantile(draws[:, column], 0.995, method="linear"))] if active else [None, None], "group_values": column_values.tolist()}
    return result


def _bounds(rows: list[dict[str, Any]], groups: list[str], metric: str) -> dict[str, Any]:
    vals = [float(row[metric]) for row in rows if row.get("observed") and row.get(metric) is not None]
    missing = len(groups) - len(vals)
    total = float(np.sum(vals)) if vals else 0.0
    return {"group_count": len(groups), "observed_group_count": len(vals), "missing_group_count": missing, "sum_observed": total, "lower": float((total - missing) / len(groups)) if groups else -1.0, "upper": float((total + missing) / len(groups)) if groups else 1.0, "interpretation": "finite-cohort bounds; missing group values are bounded in [-1,1], not a CI"}


def _recompute_model(groups: list[str], by_group: dict[str, dict[str, dict[str, Any]]], model: str, calibrated: bool) -> dict[str, Any]:
    rows = _model_rows(groups, by_group, model)
    stats = _stats(rows, groups, ("b_dynamic", "q_natural", "q_candidate"))
    bounds = {metric: _bounds(rows, groups, metric) for metric in ("b_dynamic", "q_natural", "q_candidate")}
    q_gate = bool(stats["q_natural"]["ci99"][0] is not None and stats["q_candidate"]["ci99"][0] is not None and stats["q_natural"]["ci99"][0] > 0 and stats["q_candidate"]["ci99"][0] > 0 and stats["q_natural"]["positive_groups"] >= 10 and stats["q_candidate"]["positive_groups"] >= 10)
    b = stats["b_dynamic"]
    observed_metric_gate = bool(b["mean"] is not None and b["mean"] > 0.02 and b["ci99"][0] is not None and b["ci99"][0] > 0 and b["positive_groups"] >= 10)
    coverage_gate = bool(stats["observed_group_count"] >= 11)
    full = bool(bounds["b_dynamic"]["lower"] > 0)
    status = "METRIC_NOT_CALIBRATED" if not calibrated else "GENERATED_TIMING_SPECIFICITY_UNRESOLVED" if not q_gate else "SIGNAL" if observed_metric_gate and coverage_gate and full else "MISSINGNESS_LIMITED" if observed_metric_gate else "NO_DYNAMIC_ADVANTAGE_ESTABLISHED"
    return {"rows": rows, "observed_group_count": int(stats["observed_group_count"]), "missing_group_count": int(len(groups)-stats["observed_group_count"]), "stats": stats, "bounds": bounds, "gates": {"real_calibration": calibrated, "generated_timing_specificity": q_gate, "observed_dynamic": observed_metric_gate, "coverage": coverage_gate, "full_cohort_bound": full}, "status": status, "dynamic_signal": status == "SIGNAL", "replacement_confirmed": False, "training_authorized": False, "generalization_established": False}


def _json_equal(left: Any, right: Any, *, path: str = "") -> list[str]:
    """Compare producer and independently recomputed JSON with tight floats."""
    if isinstance(left, float) or isinstance(right, float):
        try:
            if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12, equal_nan=False):
                return [path]
            return []
        except (TypeError, ValueError):
            return [path]
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        errors: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                errors.append(f"{path}/{key}")
            else:
                errors.extend(_json_equal(left[key], right[key], path=f"{path}/{key}"))
        return errors
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        errors: list[str] = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            errors.extend(_json_equal(a, b, path=f"{path}/{index}"))
        return errors
    return [] if left == right else [path]


def _check_blind(branch: Path, errors: list[str]) -> dict[str, Any]:
    package_path = branch / "blind/package.json"
    if not package_path.is_file():
        errors.append("missing blind package")
        return {}
    package = _load(package_path, errors)
    if int(package.get("primary_trials", -1)) != 24 or int(package.get("qc_trials", -1)) != 4:
        errors.append("blind package must contain 24 primary and 4 QC trials")
    if package.get("human_status") != "pending" or package.get("visual_verified") is not False:
        errors.append("blind package human flags are invalid")
    secret = Path(str(package.get("secret_mapping_path", "")))
    if not secret.is_file():
        errors.append("blind secret mapping missing outside reviewer package")
    elif package.get("secret_mapping_sha256") != file_sha256(secret):
        errors.append("blind secret mapping SHA mismatch")
    for reviewer in (1, 2):
        reviewer_path = branch / f"blind/reviewer_{reviewer}/package.json"
        reviewer_package = _load(reviewer_path, errors)
        if int(reviewer_package.get("primary_trials", -1)) != 24 or int(reviewer_package.get("qc_trials", -1)) != 4:
            errors.append(f"reviewer {reviewer} package trial counts invalid")
        for trial in reviewer_package.get("trials", []):
            forbidden = {"model", "source_group", "sample_id", "condition", "seed", "score", "sync_c"}
            if forbidden.intersection(trial):
                errors.append(f"reviewer package leaks condition metadata: reviewer={reviewer}")
    # The public package may duplicate entries for two reviewers, but it must
    # not expose the private key or any condition labels either.
    for trial in package.get("trials", []):
        if {"model", "source_group", "sample_id", "condition", "seed", "score", "sync_c"}.intersection(trial):
            errors.append("public blind package leaks condition metadata")
    return package


def _human_semantic(answer: str, trial: Mapping[str, Any]) -> str:
    if answer in {"same", "unable", "missing"}:
        return answer
    side_map = trial.get("condition_by_side")
    if not isinstance(side_map, Mapping):
        return "unable"
    value = side_map.get(answer)
    return str(value) if value in {"N", "C", "identity", "transform"} else "unable"


def _human_ci(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_COUNT, len(array)), dtype=np.int64)
    draws = array[indices].mean(axis=1)
    return [
        float(np.quantile(draws, 0.005, method="linear")),
        float(np.quantile(draws, 0.995, method="linear")),
    ]


def _recompute_human(
    branch: Path,
    analysis: Mapping[str, Any],
    dynamic_by_model: Mapping[str, bool],
    errors: list[str],
) -> dict[str, Any] | None:
    ratings_path = branch / "blind/ratings.json"
    if not ratings_path.is_file():
        if analysis.get("human_status") != "pending" or analysis.get("visual_verified") is not False:
            errors.append("analysis must remain human pending without ratings.json")
        return None
    ratings = _load(ratings_path, errors)
    secret = _load(branch / "blind/secret_mapping.json", errors)
    trial_map = secret.get("trials")
    rows = ratings.get("ratings")
    if not isinstance(trial_map, Mapping) or not isinstance(rows, list):
        errors.append("ratings/secret artifact is malformed")
        return None
    by_reviewer: dict[int, dict[str, str]] = {1: {}, 2: {}}
    allowed = {"left", "right", "same", "unable", "missing"}
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("ratings contains a malformed row")
            continue
        try:
            reviewer = int(row.get("reviewer", -1))
        except (TypeError, ValueError):
            reviewer = -1
        trial_id = str(row.get("trial_id", ""))
        answer = str(row.get("answer", ""))
        if reviewer not in by_reviewer or trial_id not in trial_map or answer not in allowed:
            errors.append(f"ratings row has invalid reviewer/trial/answer: {reviewer}/{trial_id}")
            continue
        if trial_id in by_reviewer[reviewer]:
            errors.append(f"duplicate ratings row: {reviewer}/{trial_id}")
            continue
        by_reviewer[reviewer][trial_id] = answer
    trial_ids = list(trial_map)
    qc_ids = [trial_id for trial_id in trial_ids if trial_map[trial_id].get("kind") == "qc"]
    primary_ids = [trial_id for trial_id in trial_ids if trial_map[trial_id].get("kind") == "primary"]
    reviewer_reports: list[dict[str, Any]] = []
    for reviewer in (1, 2):
        answers = by_reviewer[reviewer]
        missing = [trial_id for trial_id in trial_ids if trial_id not in answers]
        unjudgeable = [trial_id for trial_id, answer in answers.items() if answer in {"unable", "missing"}]
        qc_correct = sum(
            _human_semantic(answers[trial_id], trial_map[trial_id]) == "identity"
            for trial_id in qc_ids
            if trial_id in answers
        )
        qc_answered = sum(answers.get(trial_id) in {"left", "right"} for trial_id in qc_ids)
        qualified = bool(qc_correct >= 3)
        complete = not missing
        reviewer_reports.append(
            {
                "reviewer": reviewer,
                "qc_total": len(qc_ids),
                "qc_answered": int(qc_answered),
                "qc_identity_correct": int(qc_correct),
                "qualified": qualified,
                "complete": complete,
                "qualified_complete": bool(qualified and complete),
                "missing_trial_ids": missing,
                "unjudgeable_trial_ids": unjudgeable,
                "rated_trial_count": len(answers),
            }
        )
    qualified = [report["reviewer"] for report in reviewer_reports if report["qualified_complete"]]
    agreement: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    for model in MODELS:
        model_ids = [trial_id for trial_id in primary_ids if str(trial_map[trial_id].get("model")) == model]
        pairs: list[bool] = []
        if len(qualified) >= 2:
            pairs = [
                _human_semantic(by_reviewer[qualified[0]][trial_id], trial_map[trial_id])
                == _human_semantic(by_reviewer[qualified[1]][trial_id], trial_map[trial_id])
                for trial_id in model_ids
            ]
        agreement[model] = {
            "total": len(pairs),
            "agree": int(sum(pairs)),
            "fraction": float(np.mean(pairs)) if pairs else None,
        }
        groups: list[str] = []
        values: list[float] = []
        if len(qualified) >= 2:
            for trial_id in model_ids:
                trial = trial_map[trial_id]
                semantic = [_human_semantic(by_reviewer[reviewer][trial_id], trial) for reviewer in qualified[:2]]
                scores = [1.0 if value == "C" else -1.0 if value == "N" else 0.0 for value in semantic]
                groups.append(str(trial.get("source_group", trial_id)))
                values.append(float(np.mean(scores)))
        mean = float(np.mean(values)) if values else None
        ci = _human_ci(values)
        positive = int(sum(value > 0.0 for value in values))
        support = bool(mean is not None and ci[0] is not None and mean > 0.0 and ci[0] > 0.0 and positive >= 9)
        model_reports[model] = {
            "reviewer_count": len(qualified),
            "group_count": len(values),
            "group_values": values,
            "groups": groups,
            "mean": mean,
            "positive_groups": positive,
            "bootstrap_count": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci99": ci,
            "support": support,
        }
    human_status = "complete" if len(qualified) >= 2 else "insufficient"
    expected = {
        "schema_version": 1,
        "status": human_status,
        "human_status": human_status,
        "reviewers": reviewer_reports,
        "qualified_complete_reviewers": qualified,
        "agreement": agreement,
        "models": model_reports,
    }
    actual = analysis.get("human_summary")
    if not isinstance(actual, Mapping):
        errors.append("analysis is missing human_summary despite ratings.json")
    else:
        errors.extend(f"human_summary/{path}" for path in _json_equal(actual, expected))
    verified_models: list[str] = []
    for model in MODELS:
        support = bool(model_reports[model]["support"])
        expected_verified = bool(dynamic_by_model.get(model) and support)
        model_row = analysis.get("models", {}).get(model, {}) if isinstance(analysis.get("models"), Mapping) else {}
        if model_row.get("human") != model_reports[model]:
            errors.extend(f"analysis/{model}/human/{path}" for path in _json_equal(model_row.get("human"), model_reports[model]))
        if model_row.get("human_support") is not support:
            errors.append(f"analysis/{model}/human_support mismatch")
        if model_row.get("visual_verified") is not expected_verified:
            errors.append(f"analysis/{model}/visual_verified mismatch")
        if expected_verified:
            verified_models.append(model)
    if analysis.get("human_status") != human_status:
        errors.append("analysis human_status mismatch")
    if analysis.get("visual_verified_models") != verified_models or analysis.get("visual_verified") is not bool(verified_models):
        errors.append("analysis visual verification flags mismatch")
    return expected


def validate_run(run_root: Path) -> dict[str, Any]:
    root = _root(run_root.resolve())
    branch = _branch(root)
    errors: list[str] = []
    cohort_path = _manifest(root, "cohort.json")
    if cohort_path is None:
        errors.append("cohort manifest missing")
        result = {"schema_version": 1, "status": "NO_GO", "errors": errors}
        write_json(branch / "validation_independent.json", result)
        return result
    cohort = _load(cohort_path, errors)
    if not cohort:
        result = {"schema_version": 1, "status": "NO_GO", "errors": errors or ["cohort manifest is invalid"]}
        write_json(branch / "validation_independent.json", result)
        return result
    if str(cohort.get("status")) not in {"GO", "COHORT_READY"} and str(cohort.get("readiness")) != "COHORT_READY":
        # Preserve old blocked runs as a valid engineering terminal; this is
        # not a scientific negative and has no generated cells.
        final_path = branch / "final.json"
        if final_path.is_file():
            final = _load(final_path, errors)
            if final.get("visual_verified") is not False or final.get("replacement_confirmed") is not False:
                errors.append("blocked visual branch contains a positive terminal flag")
        result = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors, "blocked_compatibility": True}
        write_json(branch / "validation_independent.json", result)
        return result
    groups, records = _records(root, errors)
    index, by_group = _feature_index(branch, errors)
    _feature_provenance(index, by_group, groups, records, errors)
    a_manifest = _validate_a_manifest(root, groups, records, errors)
    if a_manifest["status"] == "missing":
        # A is allowed to be unfinished while B reports explicit missingness;
        # complete cells, when present, are checked strictly above.
        pass
    calibration_path = branch / "calibration.json"
    analysis_path = branch / "analysis.json"
    if not calibration_path.is_file():
        errors.append("missing B/calibration.json")
    if not analysis_path.is_file():
        errors.append("missing B/analysis.json")
    calibration = _load(calibration_path, errors) if calibration_path.is_file() else {}
    analysis = _load(analysis_path, errors) if analysis_path.is_file() else {}
    if calibration and index:
        expected_calibration = _recompute_calibration(groups, by_group)
        for field in ("rows", "identity_pass", "distortion_counts", "distortion_observed", "distortion_pass", "support_pass", "status", "calibrated"):
            errors.extend(f"calibration/{field}" for _ in _json_equal(calibration.get(field), expected_calibration.get(field), path=field))
    expected_models: dict[str, dict[str, Any]] = {}
    if analysis.get("models"):
        calibration_ok = bool(calibration.get("calibrated"))
        for model in MODELS:
            if model not in analysis["models"]:
                errors.append(f"analysis missing model: {model}")
                continue
            expected = _recompute_model(groups, by_group, model, calibration_ok)
            expected_models[model] = expected
            actual = analysis["models"][model]
            for field in ("rows", "observed_group_count", "missing_group_count", "stats", "bounds", "gates", "status", "dynamic_signal", "replacement_confirmed", "training_authorized", "generalization_established"):
                errors.extend(f"analysis/{model}/{field}/{path}" for path in _json_equal(actual.get(field), expected.get(field), path=field))
    package = _check_blind(branch, errors)
    _recompute_human(branch, analysis, {model: bool(value.get("dynamic_signal")) for model, value in expected_models.items()}, errors)
    final_path = branch / "final.json"
    if not final_path.is_file():
        errors.append("missing B/final.json")
    else:
        final = _load(final_path, errors)
        if final.get("replacement_confirmed") is not False:
            errors.append("replacement_confirmed must remain false")
        if final.get("training_authorized") is not False or final.get("generalization_established") is not False:
            errors.append("final positive authorization flag is invalid")
        if final.get("analysis_sha256") != analysis.get("artifact_sha256"):
            errors.append("final analysis_sha256 does not bind analysis.json")
        if final.get("blind_package_sha256") != package.get("artifact_sha256"):
            errors.append("final blind_package_sha256 does not bind blind/package.json")
        validation_sha = final.get("validation_sha256")
        if validation_sha is not None:
            validation_path = branch / "validation.json"
            if not validation_path.is_file() or _load(validation_path, errors).get("artifact_sha256") != validation_sha:
                errors.append("final validation_sha256 does not bind validation.json")
        ratings_present = (branch / "blind/ratings.json").is_file()
        if not ratings_present:
            if final.get("human_status") != "pending" or final.get("visual_verified") is not False:
                errors.append("final human flags must remain pending without ratings")
        elif final.get("human_status") != analysis.get("human_status") or final.get("visual_verified") is not analysis.get("visual_verified"):
            errors.append("final human flags are not bound to analysis")
    result = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "independent": True, "errors": errors, "group_count": len(groups), "feature_index_sha256": file_sha256(branch / "features.json") if (branch / "features.json").is_file() else None}
    write_json(branch / "validation_independent.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    print(result)
    return 0 if result.get("status") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
