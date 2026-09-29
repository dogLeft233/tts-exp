"""Independent A-route validator.

This module intentionally repeats the fixed-support arithmetic instead of
loading ``analyze_model_cells``.  A signed/rewritten ``analysis.json`` cannot
therefore make the terminal artifact pass without matching the raw embedding
arrays and frozen media bindings.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.fresh_source_inputs.protocol import read_json

try:
    from .common import (
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
        EPSILON,
        FPS,
        LAG_MAX,
        LAG_MIN,
        SEEDS,
        SHIFT_SAMPLES,
        U_START,
        U_STOP,
        VIDEO_FRAMES,
        ReplacementError,
        branch_dir,
        cell_key,
        decode_pcm,
        file_sha256,
        media_probe,
        read_pcm16,
        read_self_hashed,
        resolve_run_paths,
        sha256_bytes,
        validate_frozen_inputs,
        write_self_hashed,
    )
except ImportError:  # ``python scripts/.../validate.py`` CLI invocation
    from scripts.experiments.fresh_source_replacement.common import (
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
        EPSILON,
        FPS,
        LAG_MAX,
        LAG_MIN,
        SEEDS,
        SHIFT_SAMPLES,
        U_START,
        U_STOP,
        VIDEO_FRAMES,
        ReplacementError,
        branch_dir,
        cell_key,
        decode_pcm,
        file_sha256,
        media_probe,
        read_pcm16,
        read_self_hashed,
        resolve_run_paths,
        sha256_bytes,
        validate_frozen_inputs,
        write_self_hashed,
    )


MODEL = "wav2lip"
DELAY_ARM = "DELAY"


def _metric(visual: np.ndarray, audio: np.ndarray) -> dict[str, Any]:
    """Independent implementation of fixed-support z/C/D/offset."""

    visual = np.asarray(visual, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != audio.shape[1]:
        raise ReplacementError(f"malformed embeddings: visual={visual.shape}, audio={audio.shape}")
    if visual.shape[0] < U_STOP or audio.shape[0] <= (U_STOP - 1) + LAG_MAX or U_START + LAG_MIN < 0:
        raise ReplacementError(f"embedding support is too short: visual={visual.shape}, audio={audio.shape}")
    values = np.empty(LAG_MAX - LAG_MIN + 1, dtype=np.float64)
    for j, lag in enumerate(range(LAG_MIN, LAG_MAX + 1)):
        distances: list[np.float32] = []
        for r in range(U_START, U_STOP):
            q = r + lag
            delta = visual[r] - audio[q] + np.float32(EPSILON)
            distances.append(np.sqrt(np.sum(delta * delta, dtype=np.float32), dtype=np.float32))
        values[j] = np.mean(np.asarray(distances, dtype=np.float32), dtype=np.float64)
    best = int(np.argmin(values))
    return {
        "z": [float(value) for value in values],
        "C": float(np.median(values) - np.min(values)),
        "D": float(np.min(values)),
        "offset": int(15 - best),
        "best_index": best,
        "best_lag": int(LAG_MIN + best),
    }


def _delay_metric(visual: np.ndarray, audio: np.ndarray, delay: int = 5) -> tuple[dict[str, Any], dict[str, Any], float, int]:
    natural = _metric(visual, audio)
    shifted = np.empty(LAG_MAX - LAG_MIN + 1, dtype=np.float64)
    for j, lag in enumerate(range(LAG_MIN, LAG_MAX + 1)):
        values: list[np.float32] = []
        for r in range(U_START, U_STOP):
            q = r + lag - delay
            if q < 0 or q >= audio.shape[0]:
                raise ReplacementError("delayed audio support is unavailable")
            delta = np.asarray(visual[r], dtype=np.float32) - np.asarray(audio[q], dtype=np.float32) + np.float32(EPSILON)
            values.append(np.sqrt(np.sum(delta * delta, dtype=np.float32), dtype=np.float32))
        shifted[j] = np.mean(np.asarray(values, dtype=np.float32), dtype=np.float64)
    best = int(np.argmin(shifted))
    delayed = {
        "z": [float(value) for value in shifted],
        "C": float(np.median(shifted) - np.min(shifted)),
        "D": float(np.min(shifted)),
        "offset": int(15 - best),
        "best_index": best,
        "best_lag": int(LAG_MIN + best),
    }
    k0 = int(natural["best_index"])
    return natural, delayed, float(shifted[k0] - np.asarray(natural["z"], dtype=np.float64)[k0]), best - k0


def _bootstrap_indices(n: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0, n, size=(BOOTSTRAP_DRAWS, n), dtype=np.int64)


def _bootstrap(values: Sequence[float], indices: np.ndarray, confidence: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    estimates = np.mean(array[np.asarray(indices, dtype=np.int64)], axis=1, dtype=np.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    bounds = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return {
        "mean": float(np.mean(array, dtype=np.float64)), "ci": [float(bounds[0]), float(bounds[1])],
        "confidence": float(confidence), "draws": int(indices.shape[0]), "group_count": int(array.size),
        "group_positive_count": int(np.count_nonzero(array > 0)),
        "indices_sha256": sha256_bytes(np.ascontiguousarray(indices, dtype=np.int64).tobytes()),
        "seed": BOOTSTRAP_SEED, "method": "numpy.PCG64; np.quantile(method=linear)",
    }


def _lookup(cells: Mapping[str, Mapping[str, Any]], sample_id: str, arm: str, seed: int, repeat: int = 0) -> Mapping[str, Any] | None:
    wanted = cell_key(sample_id, MODEL, arm, seed, repeat)
    value = cells.get(wanted)
    if isinstance(value, Mapping):
        return value
    for candidate in cells.values():
        if isinstance(candidate, Mapping) and str(candidate.get("sample_id")) == sample_id and str(candidate.get("arm")) == arm and int(candidate.get("seed", -1)) == seed and int(candidate.get("repeat_index", 0)) == repeat:
            return candidate
    return None


def _embedding(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row.get("embedding_path", "")))
    if not path.is_file() or file_sha256(path) != row.get("embedding_sha256"):
        raise ReplacementError(f"embedding hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "visual" not in data or "audio" not in data:
            raise ReplacementError(f"embedding artifact lacks visual/audio: {path}")
        visual = np.asarray(data["visual"], dtype=np.float32)
        audio = np.asarray(data["audio"], dtype=np.float32)
    return visual, audio


def _cell_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    visual, audio = _embedding(row)
    return _metric(visual, audio)


def _critical_analysis(bundle: Mapping[str, Any], score_cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in bundle["formal_records"]:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        values: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
        for arm in ("N", "C"):
            for seed in SEEDS:
                cell = _lookup(score_cells, sample_id, arm, seed)
                if cell is None or cell.get("status") != "complete":
                    missing.append(cell_key(sample_id, MODEL, arm, seed, 0))
                else:
                    values[f"{arm}{seed}"] = (cell, _cell_metrics(cell))
        repeat_cell = _lookup(score_cells, sample_id, "N", 42, 1)
        if sample_id in [str(item["sample_id"]) for item in bundle["formal_records"][:2]] and (repeat_cell is None or repeat_cell.get("status") != "complete"):
            missing.append(cell_key(sample_id, MODEL, "N", 42, 1))
        if any(name not in values for name in ("N42", "N43", "C42", "C43")):
            continue
        n42, n42m = values["N42"]
        _n43, n43m = values["N43"]
        _c42, c42m = values["C42"]
        _c43, c43m = values["C43"]
        k0 = int(n42m["best_index"])
        visual, audio = _embedding(n42)
        _natural_delay, delayed, damage, best_shift = _delay_metric(visual, audio)
        row: dict[str, Any] = {
            "sample_id": sample_id, "source_group": group, "k0": k0, "k0_lag": int(n42m["best_lag"]),
            "natural_42": n42m, "natural_43": n43m, "candidate_42": c42m, "candidate_43": c43m,
            "fixed_anchor_gain_42": float(n42m["z"][k0] - c42m["z"][k0]),
            "fixed_anchor_gain_43": float(n43m["z"][k0] - c43m["z"][k0]),
            "sync_c_gain_42": float(c42m["C"] - n42m["C"]), "sync_c_gain_43": float(c43m["C"] - n43m["C"]),
            "sync_d_gain_42": float(n42m["D"] - c42m["D"]), "sync_d_gain_43": float(n43m["D"] - c43m["D"]),
            "natural_seed_difference": {
                "A": float(n43m["z"][k0] - n42m["z"][k0]), "C": float(n43m["C"] - n42m["C"]), "D": float(n43m["D"] - n42m["D"]),
            },
            "candidate_seed_difference": {
                "A": float(c43m["z"][k0] - c42m["z"][k0]), "C": float(c43m["C"] - c42m["C"]), "D": float(c43m["D"] - c42m["D"]),
            },
            "delay_control": {
                "natural": n42m, "delayed": delayed, "natural_k0": k0, "uncompensated_damage": damage,
                "best_index_shift": best_shift, "offset_change": int(delayed["offset"] - n42m["offset"]),
                # The matched-domain curves are checked separately below; the
                # arrays are not needed for the signal gate.
            },
        }
        if repeat_cell is not None and repeat_cell.get("status") == "complete":
            repeatm = _cell_metrics(repeat_cell)
            row["natural_42_repeat"] = repeatm
            row["repeat_differences"] = {
                "C": float(repeatm["C"] - n42m["C"]), "D": float(repeatm["D"] - n42m["D"]),
                "A_at_k0": float(repeatm["z"][k0] - n42m["z"][k0]), "offset": int(repeatm["offset"] - n42m["offset"]),
                "pixel_max_abs": repeat_cell.get("pixel_max_abs"), "pixel_mean_abs": repeat_cell.get("pixel_mean_abs"),
                "pixel_different_bytes": repeat_cell.get("pixel_different_bytes"),
            }
        rows.append(row)
    complete = not missing and len(rows) == 12
    indices = _bootstrap_indices(12)
    gain_a = [(row["fixed_anchor_gain_42"] + row["fixed_anchor_gain_43"]) / 2.0 for row in rows]
    gain_c = [(row["sync_c_gain_42"] + row["sync_c_gain_43"]) / 2.0 for row in rows]
    gain_d = [(row["sync_d_gain_42"] + row["sync_d_gain_43"]) / 2.0 for row in rows]
    noise = {
        "fixed_anchor": float(np.mean([abs(row["natural_seed_difference"]["A"]) for row in rows], dtype=np.float64)) if rows else None,
        "sync_c": float(np.mean([abs(row["natural_seed_difference"]["C"]) for row in rows], dtype=np.float64)) if rows else None,
        "sync_d": float(np.mean([abs(row["natural_seed_difference"]["D"]) for row in rows], dtype=np.float64)) if rows else None,
    }
    repeat_rows = [row for row in rows if "repeat_differences" in row]
    shifted_damage = [float(row["delay_control"]["uncompensated_damage"]) for row in rows]
    repeat_pass = len(repeat_rows) == 2 and all(abs(float(row["repeat_differences"][key])) <= 0.01 for row in repeat_rows for key in ("C", "D", "A_at_k0")) and all(abs(int(row["repeat_differences"]["offset"])) <= 1 for row in repeat_rows)
    natural_interior = complete and all(1 <= int(row["k0"]) <= 29 for row in rows)
    delay95 = _bootstrap(shifted_damage, indices, 0.95) if complete else None
    controls_pass = bool(complete and natural_interior and sum(value > 0 for value in shifted_damage) >= 10 and delay95 and delay95["ci"][0] > 0 and repeat_pass)
    ba = _bootstrap(gain_a, indices, 0.99) if complete else None
    bc = _bootstrap(gain_c, indices, 0.99) if complete else None
    bd = _bootstrap(gain_d, indices, 0.99) if complete else None
    joint = int(sum(a > 0 and c > 0 for a, c in zip(gain_a, gain_c, strict=True)))
    signal = bool(complete and controls_pass and ba and bc and bd and float(np.mean(gain_c)) > 0.05 and bc["ci"][0] > 0 and ba["ci"][0] > 0 and float(np.mean(gain_c)) > float(noise["sync_c"]) and float(np.mean(gain_a)) > float(noise["fixed_anchor"]) and bd["ci"][0] > -0.1 and joint >= 10)
    status = "ENGINEERING_INCOMPLETE" if not complete else ("CONTROL_FAILED" if not controls_pass else ("SIGNAL" if signal else "NO_REPLACEMENT_SIGNAL_ESTABLISHED"))
    return {
        "schema_version": 1, "model": MODEL, "status": status, "formal_groups": len(rows),
        "formal_cells": len(rows) * 4 + (2 if complete else 0), "expected_formal_cells": 50, "missing_cells": missing,
        "rows": rows,
        "controls": {
            "pass": controls_pass, "natural_k0_interior": natural_interior,
            "k0_values": [int(row["k0"]) for row in rows], "shifted_damage_positive_count": int(sum(value > 0 for value in shifted_damage)),
            "shifted_damage": shifted_damage, "shifted_damage_bootstrap_95": delay95,
            "repeat_pass": repeat_pass, "repeat_tolerance": {"C": 0.01, "D": 0.01, "fixed_anchor": 0.01, "offset_frames": 1},
        },
        "noise_floor": noise, "gains": {"fixed_anchor": gain_a, "sync_c": gain_c, "sync_d": gain_d},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "indices_sha256": sha256_bytes(np.ascontiguousarray(indices).tobytes()), "fixed_anchor": ba, "sync_c": bc, "sync_d": bd},
        "signal_gate": {"pass": signal, "mean_delta_sync_c_threshold": 0.05, "mean_delta_sync_c": float(np.mean(gain_c)) if gain_c else None, "joint_positive_count": joint, "joint_positive_required": 10, "replacement_confirmed": False},
        "replacement_confirmed": False, "training_authorized": False, "generalization_established": False, "human_status": "pending",
    }


def _compare(expected: Any, actual: Any, path: str, errors: list[str], tolerance: float = 1e-6) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            errors.append(f"{path}: expected object")
            return
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                _compare(value, actual[key], f"{path}.{key}", errors, tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(f"{path}: list shape differs")
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]", errors, tolerance)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            errors.append(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float, np.number)) and isinstance(actual, (int, float, np.number)):
        if not np.isfinite(float(expected)) or not np.isfinite(float(actual)) or abs(float(expected) - float(actual)) > tolerance:
            errors.append(f"{path}: {actual!r} != {expected!r}")
        return
    if expected != actual:
        errors.append(f"{path}: {actual!r} != {expected!r}")


def _validate_delay_contract(bundle: Mapping[str, Any], errors: list[str]) -> dict[str, str]:
    """Independently verify C's exact DELAY PCM/source-index bindings."""

    root = Path(bundle["run_root"])
    manifest_path = root / "run" / "C" / "delay_inputs.json"
    try:
        manifest = read_self_hashed(manifest_path)
    except ReplacementError as exc:
        errors.append(f"DELAY manifest: {exc}")
        return {}
    if manifest.get("status") != "READY":
        errors.append(f"DELAY manifest is not READY: {manifest.get('status')!r}")
        return {}
    if int(manifest.get("delay_samples", -1)) != SHIFT_SAMPLES or int(manifest.get("delay_frames", -1)) != 5:
        errors.append("DELAY manifest does not use the frozen +3200/+5 contract")
        return {}
    groups = manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != len(bundle["formal_records"]):
        errors.append("DELAY manifest does not contain exactly twelve formal groups")
        return {}
    by_key = {
        (str(item.get("source_group")), str(item.get("sample_id"))): item
        for item in groups
        if isinstance(item, Mapping)
    }
    result: dict[str, str] = {}
    for record in bundle["formal_records"]:
        sid = str(record["sample_id"])
        key = (str(record["source_group"]), sid)
        entry = by_key.get(key)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("delay"), Mapping):
            errors.append(f"DELAY manifest has no binding for {sid}")
            continue
        delayed = entry["delay"]
        delay_path = Path(str(delayed.get("path", ""))).expanduser().resolve()
        try:
            delayed_pcm, _delayed_values, delayed_params = read_pcm16(delay_path)
            natural_pcm, natural_values, natural_params = read_pcm16(Path(str(record["_natural_audio_path"])))
        except ReplacementError as exc:
            errors.append(f"DELAY PCM {sid}: {exc}")
            continue
        if not delay_path.is_file() or file_sha256(delay_path) != delayed.get("sha256"):
            errors.append(f"DELAY container hash mismatch: {sid}")
        if sha256_bytes(delayed_pcm) != delayed.get("pcm_sha256"):
            errors.append(f"DELAY PCM hash mismatch: {sid}")
        if delayed_params != natural_params or delayed_params["frame_count"] != int(record["natural_pcm_sample_count"]):
            errors.append(f"DELAY PCM format/length differs from natural audio: {sid}")
        expected_pcm = b"\0" * (SHIFT_SAMPLES * 2) + natural_pcm[: -(SHIFT_SAMPLES * 2)]
        if delayed_pcm != expected_pcm:
            errors.append(f"DELAY PCM is not exact N[n-3200] with zero prefix: {sid}")
        mapping = entry.get("source_index_map")
        map_path = Path(str(mapping.get("path", ""))).expanduser().resolve() if isinstance(mapping, Mapping) else None
        try:
            source_map = np.asarray(np.load(map_path, allow_pickle=False), dtype=np.int64) if map_path is not None else None
        except (OSError, ValueError):
            source_map = None
        expected_map = np.full(natural_values.size, -1, dtype=np.int64)
        expected_map[SHIFT_SAMPLES:] = np.arange(natural_values.size - SHIFT_SAMPLES, dtype=np.int64)
        if source_map is None or source_map.shape != expected_map.shape or not np.array_equal(source_map, expected_map):
            errors.append(f"DELAY source-index map mismatch: {sid}")
        result[sid] = sha256_bytes(delayed_pcm)
    return result


def _validate_media(bundle: Mapping[str, Any], branch_root: Path, errors: list[str]) -> None:
    videos_path = branch_root / "videos.json"
    if not videos_path.is_file():
        errors.append("videos.json is missing")
        return
    videos = read_self_hashed(videos_path)
    cells = videos.get("cells")
    if not isinstance(cells, Mapping):
        errors.append("videos.json cells is not an object")
        return
    records = {str(row["sample_id"]): row for row in bundle["records"]}
    # Four cells (two seeds plus the repeat) can intentionally bind the same
    # media.  Probe/decode each unique path once; this preserves independent
    # verification without making validation needlessly quadratic in repeats.
    media_cache: dict[str, tuple[dict[str, Any], bytes] | Exception] = {}
    delay_rows = [row for row in cells.values() if isinstance(row, Mapping) and row.get("arm") == "DELAY"]
    delay_pcm = _validate_delay_contract(bundle, errors) if delay_rows else {}
    expected = set()
    for record in bundle["formal_records"]:
        sid = str(record["sample_id"])
        for arm in ("N", "C"):
            for seed in SEEDS:
                expected.add(cell_key(sid, MODEL, arm, seed, 0))
        if delay_rows:
            expected.add(cell_key(sid, MODEL, "DELAY", 42, 0))
        if sid in [str(item["sample_id"]) for item in bundle["formal_records"][:2]]:
            expected.add(cell_key(sid, MODEL, "N", 42, 1))
    present = {str(key) for key, row in cells.items() if isinstance(row, Mapping) and row.get("status") == "complete"}
    missing = sorted(expected - present)
    if missing:
        errors.append(f"videos.json missing complete expected cells: {missing[:8]}")
    for key in sorted(present & expected):
        row = cells[key]
        media = Path(str(row.get("path", "")))
        if not media.is_file() or file_sha256(media) != row.get("sha256"):
            errors.append(f"{key}: media hash mismatch")
            continue
        try:
            cached = media_cache.get(str(media))
            if cached is None:
                cached = (media_probe(media), decode_pcm(media))
                media_cache[str(media)] = cached
            if isinstance(cached, Exception):
                raise cached
            probe, pcm = cached
            if probe["frame_count"] != VIDEO_FRAMES or abs(float(probe["fps"]) - FPS) > 1e-6 or not probe.get("has_audio"):
                errors.append(f"{key}: video clock/audio contract failed")
            record = records.get(str(row.get("sample_id")))
            if record is None:
                errors.append(f"{key}: unknown sample")
                continue
            arm = str(row.get("arm"))
            if arm == "N":
                expected_pcm = str(record["natural_pcm_sha256"])
            elif arm == "C":
                expected_pcm = str((record.get("direct_audio") or {}).get("decoded_pcm_sha256"))
            elif arm == DELAY_ARM:
                expected_pcm = str(delay_pcm.get(str(row.get("sample_id")), ""))
            else:
                errors.append(f"{key}: unsupported media arm {arm!r}")
                continue
            if sha256_bytes(pcm) != expected_pcm:
                errors.append(f"{key}: mux PCM differs from frozen {arm}")
            mux = row.get("mux")
            if isinstance(mux, Mapping) and (mux.get("audio_pcm_verified") is not True or mux.get("video_stream_copy_verified") is not True):
                errors.append(f"{key}: mux evidence is not verified")
        except (ReplacementError, OSError) as exc:
            errors.append(f"{key}: {exc}")


def _validate_cross_generator(branch_root: Path, errors: list[str]) -> tuple[str, str | None]:
    """Require an explicit read-only Ditto branch status."""

    path = branch_root / "ditto_ingest.json"
    if not path.is_file():
        errors.append("ditto_ingest.json is missing; cross-generator status is unresolved")
        return "BLOCKED_CROSS_GENERATOR", None
    try:
        value = read_self_hashed(path)
    except ReplacementError as exc:
        errors.append(f"ditto_ingest.json: {exc}")
        return "BLOCKED_CROSS_GENERATOR", None
    status = str(value.get("status", "BLOCKED_CROSS_GENERATOR"))
    if value.get("model") != "ditto" or value.get("read_only") is not True:
        errors.append("ditto_ingest.json is not a read-only Ditto branch artifact")
    if status not in {"COMPLETE", "BLOCKED_CROSS_GENERATOR"}:
        errors.append(f"unsupported Ditto branch status: {status!r}")
        status = "BLOCKED_CROSS_GENERATOR"
    if status == "COMPLETE" and value.get("record_count") != value.get("expected_record_count"):
        errors.append("Ditto branch claims COMPLETE with an incomplete denominator")
        status = "BLOCKED_CROSS_GENERATOR"
    if value.get("replacement_confirmed") is not False:
        errors.append("Ditto branch contains a forbidden positive terminal flag")
    return status, file_sha256(path)


def independent_recompute(run_root: Path, branch: str = "A") -> tuple[dict[str, Any] | None, list[str]]:
    """Recompute A statistics from raw embedding artifacts."""

    errors: list[str] = []
    bundle = validate_frozen_inputs(run_root, require_files=True, require_ready=True)
    root = branch_dir(bundle["run_root"], branch)
    scores = read_self_hashed(root / "scores.json")
    cells = scores.get("cells")
    if not isinstance(cells, Mapping):
        raise ReplacementError("scores.json cells is not an object")
    score_cells = {str(key): dict(value) for key, value in cells.items() if isinstance(value, Mapping)}
    return _critical_analysis(bundle, score_cells), errors


def _blocked_validate(run_root: Path, branch: str, root: Path) -> dict[str, Any]:
    errors: list[str] = []
    for name in ("control_validation.json", "analysis.json", "validation.json", "final.json"):
        try:
            read_self_hashed(root / name)
        except ReplacementError as exc:
            errors.append(f"{name}: {exc}")
    final: dict[str, Any] = {}
    try:
        final = read_self_hashed(root / "final.json")
        if final.get("replacement_confirmed") is not False or final.get("formal_cells") != 0:
            errors.append("blocked branch contains scientific/positive terminal fields")
    except ReplacementError:
        final = {}
    for name in ("analysis", "validation"):
        target = root / f"{name}.json"
        if final and (not target.is_file() or final.get(f"{name}_sha256") != file_sha256(target)):
            errors.append(f"blocked final does not bind {name}.json")
    result = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors, "branch": branch, "scientific_cells": 0, "replacement_confirmed": False}
    write_self_hashed(root / "validation_independent.json", result)
    return result


def validate(run_root: Path, branch: str = "A") -> dict[str, Any]:
    root, shared = resolve_run_paths(run_root)
    cohort_path = shared / "cohort.json"
    cohort = read_json(cohort_path) if cohort_path.is_file() else {}
    branch_root = branch_dir(root, branch)
    if cohort.get("status") not in {"GO", "COHORT_READY"} or cohort.get("readiness") not in {"COHORT_READY", None}:
        return _blocked_validate(root, branch, branch_root)
    errors: list[str] = []
    try:
        bundle = validate_frozen_inputs(root, require_files=True, require_ready=True)
    except (KeyError, OSError, ReplacementError, TypeError, ValueError) as exc:
        errors.append(f"frozen inputs: {exc}")
        result = {"schema_version": 1, "status": "NO_GO", "errors": errors, "branch": branch, "scientific_cells": 0, "replacement_confirmed": False}
        write_self_hashed(branch_root / "validation_independent.json", result)
        return result
    analysis_path = branch_root / "analysis.json"
    try:
        analysis = read_self_hashed(analysis_path)
    except (KeyError, OSError, ReplacementError, TypeError, ValueError) as exc:
        analysis = {}
        errors.append(f"analysis.json: {exc}")
    _validate_media(bundle, branch_root, errors)
    cross_status, cross_sha = _validate_cross_generator(branch_root, errors)
    if analysis.get("cross_generator_status") != cross_status:
        errors.append(
            f"analysis.cross_generator_status does not bind ditto_ingest.json: "
            f"{analysis.get('cross_generator_status')!r} != {cross_status!r}"
        )
    if cross_sha is not None and analysis.get("ditto_ingest_sha256") != cross_sha:
        errors.append("analysis.ditto_ingest_sha256 does not bind ditto_ingest.json")
    expected: dict[str, Any] | None = None
    try:
        expected, recompute_errors = independent_recompute(root, branch)
        errors.extend(recompute_errors)
    except (KeyError, OSError, ReplacementError, TypeError, ValueError) as exc:
        errors.append(f"independent recomputation: {exc}")
    if expected is not None and analysis:
        # Compare every producer field that can affect a scientific endpoint;
        # provenance hashes are checked separately and are allowed to differ
        # only in the self-hash wrapper.
        for key in ("status", "formal_groups", "formal_cells", "expected_formal_cells", "missing_cells", "rows", "controls", "noise_floor", "gains", "bootstrap", "signal_gate", "replacement_confirmed", "training_authorized", "generalization_established", "human_status"):
            if key in expected:
                _compare(expected[key], analysis.get(key), f"analysis.{key}", errors)
    expected_cells = 50
    if analysis.get("formal_cells") != expected_cells:
        errors.append(f"analysis formal_cells must be {expected_cells}, got {analysis.get('formal_cells')!r}")
    if analysis.get("replacement_confirmed") is not False or analysis.get("training_authorized") is not False or analysis.get("generalization_established") is not False:
        errors.append("A terminal positive/training flag is forbidden")
    if analysis.get("scores_sha256") and (branch_root / "scores.json").is_file() and file_sha256(branch_root / "scores.json") != analysis.get("scores_sha256"):
        errors.append("analysis.scores_sha256 does not bind scores.json")
    result = {
        "schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors, "branch": branch,
        "scientific_cells": int(analysis.get("formal_cells") or 0), "replacement_confirmed": False,
        "analysis_sha256": file_sha256(analysis_path) if analysis_path.is_file() else None,
        "cross_generator_status": cross_status,
        "ditto_ingest_sha256": cross_sha,
        "independent_recompute": expected is not None,
    }
    result = write_self_hashed(branch_root / "validation.json", result)
    final = {
        "schema_version": 1,
        "status": (
            "VALIDATION_FAILED"
            if errors
            else ("BLOCKED_CROSS_GENERATOR" if cross_status != "COMPLETE" else analysis.get("status", "ENGINEERING_INCOMPLETE"))
        ),
        "replacement_confirmed": False, "training_authorized": False, "generalization_established": False,
        "formal_cells": int(analysis.get("formal_cells") or 0), "validation_sha256": file_sha256(branch_root / "validation.json"),
        "analysis_sha256": result.get("analysis_sha256"), "cross_generator_status": cross_status,
        "ditto_ingest_sha256": cross_sha, "human_status": "pending",
    }
    write_self_hashed(branch_root / "final.json", final)
    independent = dict(result)
    independent["validation_sha256"] = file_sha256(branch_root / "validation.json")
    write_self_hashed(branch_root / "validation_independent.json", independent)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--branch", choices=("A", "D"), default="A")
    args = parser.parse_args(argv)
    result = validate(args.run_root.resolve(), args.branch)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
