from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, load_self_hashed, read_json, write_json


U_ROWS = tuple(range(30, 58))


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path).get("rows")
    if not isinstance(rows, list):
        raise ProtocolError(f"rows missing: {path}")
    return rows


def _records() -> list[dict[str, str]]:
    rows = read_json(config.A / "protocol.json")["records"]
    return sorted(({"sample_id": str(row["sample_id"]), "source_group": str(row["source_group"])} for row in rows), key=lambda row: (row["source_group"], row["sample_id"]))


def _score_map(rows: list[dict[str, Any]], arms: set[str] | None = None, audio: set[str] | None = None) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if arms is not None and key[1] not in arms:
            continue
        if audio is not None and key[2] not in audio:
            continue
        if key in result:
            raise ProtocolError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _load_worker(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = row["score"]
    worker_path = Path(str(score["worker"]))
    worker = load_self_hashed(worker_path)
    arrays = []
    for key, shape in (("visual", (88, 1024)), ("audio_embedding", (88, 1024)), ("matrix", (88, 31))):
        path = Path(str(worker[key]))
        if file_sha256(path) != str(worker[f"{key}_sha256"]):
            raise ProtocolError(f"worker array hash mismatch: {path}")
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ProtocolError(f"invalid worker array: {path}")
        arrays.append(value)
    return arrays[0], arrays[1], arrays[2]


def _matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if visual.shape != (88, 1024) or audio.shape != (88, 1024):
        raise ProtocolError("unexpected embedding shape")
    padded = np.pad(audio, ((15, 15), (0, 0)), mode="constant")
    result = np.empty((88, 31), dtype=np.float64)
    for row in range(88):
        for column in range(31):
            result[row, column] = np.sqrt(np.sum((visual[row] - padded[row + column] + 1e-6) ** 2, dtype=np.float64))
    return result


def _metrics(matrix: np.ndarray) -> dict[str, Any]:
    curve = np.mean(np.asarray(matrix, dtype=np.float64)[np.asarray(U_ROWS)], axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {"curve": curve.tolist(), "min_index": index, "offset": 15 - index, "D": float(curve[index]), "C": float(np.median(curve) - curve[index])}


def _gain(candidate: dict[str, Any], natural: dict[str, Any]) -> dict[str, float]:
    anchor = int(natural["min_index"])
    return {"C": float(candidate["C"] - natural["C"]), "D": float(natural["D"] - candidate["D"]), "A": float(natural["curve"][anchor] - candidate["curve"][anchor])}


def _bootstrap(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(float(row[field]))
    labels = sorted(grouped)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("expected eight groups with two records")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    indices = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED)).integers(0, 8, size=(config.BOOTSTRAP_DRAWS, 8))
    estimates = means[indices].mean(axis=1, dtype=np.float64)
    return {"mean": float(means.mean()), "ci99": [float(np.quantile(estimates, 0.005, method="linear")), float(np.quantile(estimates, 0.995, method="linear"))], "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "indices": indices.tolist()}


def _audit_a(records: list[dict[str, str]]) -> dict[str, Any]:
    parent = _score_map(_rows(config.P / "control_scores/manifest.json"), arms={"N"}, audio={"N"})
    candidates = _score_map(_rows(config.A / "candidate_scores/manifest.json"), arms={"SMOOTH", "SHARP"}, audio={"N"})
    output = []
    for record in records:
        sid = record["sample_id"]
        visual, audio, producer = _load_worker(parent[(sid, "N", "N")])
        natural = _matrix(visual, audio)
        natural_metrics = _metrics(natural)
        item = {"sample_id": sid, "source_group": record["source_group"], "matrix_reproduction_max_abs": {"N": float(np.max(np.abs(natural - producer)))}}
        for arm in ("SMOOTH", "SHARP"):
            candidate_visual, candidate_audio, candidate_matrix = _load_worker(candidates[(sid, arm, "N")])
            matrix = _matrix(candidate_visual, candidate_audio)
            item[arm] = _metrics(matrix)
            item["matrix_reproduction_max_abs"][arm] = float(np.max(np.abs(matrix - candidate_matrix)))
        item["N"] = natural_metrics
        item["gain"] = {arm: _gain(item[arm], natural_metrics) for arm in ("SMOOTH", "SHARP")}
        output.append(item)
    contrasts = {}
    for arm in ("SMOOTH", "SHARP"):
        contrasts[arm] = {metric: _bootstrap([{**row, metric: row["gain"][arm][metric]} for row in output], metric) for metric in ("C", "D", "A")}
    return {"records": output, "contrasts": contrasts}


def _curve(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    values = []
    for row in U_ROWS:
        # `lag_start` is the positive offset used by the frozen audit.  Keep
        # this expression explicit so the independent validator cannot
        # accidentally reverse the delayed-audio convention.
        indices = row + np.arange(31, dtype=np.int64) - lag_start
        if int(indices.min()) < 0 or int(indices.max()) >= audio.shape[0]:
            raise ProtocolError("B support is outside the audio embedding")
        values.append(np.sqrt(np.sum((visual[row : row + 1] - audio[indices] + 1e-6) ** 2, axis=1, dtype=np.float64)))
    return np.mean(np.asarray(values, dtype=np.float64), axis=0, dtype=np.float64)


def _audit_b(records: list[dict[str, str]]) -> dict[str, Any]:
    rows = _rows(config.P / "control_scores/manifest.json")
    score_map = _score_map(rows, arms={"N"}, audio={"N", "A_DELAY"})
    output = []
    old_pass = matched_pass = 0
    for record in records:
        sid = record["sample_id"]
        visual_n, audio_n, producer_n = _load_worker(score_map[(sid, "N", "N")])
        visual_d, audio_d, producer_d = _load_worker(score_map[(sid, "N", "A_DELAY")])
        if not np.array_equal(visual_n, visual_d):
            raise ProtocolError(f"B visual embeddings differ: {sid}")
        natural_curve = _curve(visual_n, audio_n, 15)
        matched_curve = _curve(visual_n, audio_d, 10)
        legacy_curve = _curve(visual_n, audio_d, 15)
        natural_index = int(np.argmin(natural_curve)); matched_index = int(np.argmin(matched_curve)); legacy_index = int(np.argmin(legacy_curve))
        natural_offset = 15 - natural_index; matched_offset = 10 - matched_index; legacy_offset = 15 - legacy_index
        old_pass += int(-6 <= legacy_offset - natural_offset <= -4)
        matched_pass += int(-6 <= matched_offset - natural_offset <= -4)
        output.append({"sample_id": sid, "source_group": record["source_group"], "legacy_offset": legacy_offset, "matched_offset": matched_offset, "natural_peak_index": natural_index, "delay_peak_index": matched_index, "legacy_curve": legacy_curve.tolist(), "matched_curve": matched_curve.tolist(), "legacy_damage_at_natural_k0": float(legacy_curve[natural_index] - natural_curve[natural_index]), "embedding_visual_max_abs": float(np.max(np.abs(visual_n - visual_d))), "producer_matrix_max_abs": {"natural": float(np.max(np.abs(_matrix(visual_n, audio_n) - producer_n))), "delay": float(np.max(np.abs(_matrix(visual_d, audio_d) - producer_d)))}})
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in output:
        grouped[row["source_group"]].append(float(row["legacy_damage_at_natural_k0"]))
    labels = sorted(grouped); means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64); indices = np.random.Generator(np.random.PCG64(20260909)).integers(0, 8, size=(10_000, 8)); estimates = means[indices].mean(axis=1, dtype=np.float64)
    damage = {"mean": float(means.mean()), "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "seed": 20260909, "draws": 10_000, "indices": indices.tolist()}
    return {"records": output, "legacy_offset_pass_count": old_pass, "matched_offset_pass_count": matched_pass, "damage": damage, "old_control_pass": bool(old_pass >= 14 and damage["ci95"][0] > 0 and damage["group_positive_count"] >= 7), "f46_boundary_restored": bool(old_pass == 16), "f46_control_as_spec": "PASS" if matched_pass >= 14 and damage["ci95"][0] > 0 and damage["group_positive_count"] >= 7 else "FAIL"}


def _supports() -> dict[int, np.ndarray]:
    chunks = []; index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > 308:
            chunks.append(np.arange(292, 308, dtype=np.int64)); break
        chunks.append(np.arange(start, start + 16, dtype=np.int64)); index += 1
    return {row: np.unique(np.concatenate(chunks[row : row + 5])).astype(np.int64) for row in U_ROWS}


def _k_for(sample_id: str, rows: list[dict[str, Any]], masks: dict[str, dict[str, Any]]) -> list[int]:
    sets = []
    for row in rows:
        if str(row["sample_id"]) != sample_id:
            continue
        current: set[int] = set()
        for used in row.get("used_masks", []):
            mask = masks.get(str(used["mask_sha256"]))
            if mask is None:
                raise ProtocolError(f"unknown mask: {sample_id}")
            start = int(used["global_start_frame"]); end = int(used["global_end_frame"])
            if (start, end) != (int(mask["natural_core_start_frame"]), int(mask["natural_core_end_frame"])):
                raise ProtocolError(f"mask identity mismatch: {sample_id}")
            current.update(range(start, end))
        sets.append(current)
    if not sets or any(value != sets[0] for value in sets[1:]):
        raise ProtocolError(f"C support differs: {sample_id}")
    return sorted(sets[0])


def _stats(values: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped = {str(row["source_group"]): float(row[field]) for row in values}
    labels = sorted(grouped)
    if len(labels) != 8:
        raise ProtocolError("C statistic requires eight group values")
    means = np.asarray([grouped[label] for label in labels], dtype=np.float64); indices = np.random.Generator(np.random.PCG64(20260910)).integers(0, 8, size=(20_000, 8)); estimates = means[indices].mean(axis=1, dtype=np.float64)
    return {"mean": float(means.mean()), "ci99": [float(np.quantile(estimates, 0.005, method="linear")), float(np.quantile(estimates, 0.995, method="linear"))], "group_labels": labels, "group_means": grouped, "group_positive_count": int(np.sum(means > 0)), "seed": 20260910, "draws": 20_000, "indices": indices.tolist()}


def _audit_c(records: list[dict[str, str]]) -> dict[str, Any]:
    parent = _score_map(_rows(config.P / "control_scores/manifest.json"), arms={"N"}, audio={"N"})
    candidates = _score_map(_rows(config.Q / "candidate_scores/manifest.json"), arms={"CORRECT", "WRONG", "SHUFFLE"}, audio={"N"})
    driver_rows = read_json(config.D_FILES["drivers"])["drivers"]; masks = {str(row["mask_sha256"]): row for row in read_json(config.D_FILES["mask_manifest"])["masks"]}; supports = _supports(); per = []
    for record in records:
        sid = record["sample_id"]; support = _k_for(sid, driver_rows, masks); exposure = {row: float(len(set(supports[row].tolist()).intersection(support)) / len(supports[row])) for row in supports}; matrices = {}; visuals = {}
        visual, audio, producer = _load_worker(parent[(sid, "N", "N")]); matrices["N"] = _matrix(visual, audio); visuals["N"] = visual
        for arm in ("CORRECT", "WRONG", "SHUFFLE"):
            visual, audio, producer = _load_worker(candidates[(sid, arm, "N")]); matrices[arm] = _matrix(visual, audio); visuals[arm] = visual
        exposure = {str(key): value for key, value in exposure.items()}
        k0 = int(np.argmin(np.mean(matrices["N"][np.asarray(U_ROWS)], axis=0))); row = {"sample_id": sid, "source_group": record["source_group"], "K": support, "K_count": len(support), "exposure": exposure, "k0": k0, "matrix_max_abs": {}, "zero_exposure_rows": [], "zero_exposure_violations": []}
        for arm in ("CORRECT", "WRONG", "SHUFFLE"):
            delta = matrices["N"][:, k0] - matrices[arm][:, k0]; d_rows = delta[np.asarray(U_ROWS)].tolist(); row[arm] = {"d_mean": float(np.mean(d_rows)), "d_rows": d_rows}; row["matrix_max_abs"][arm] = float(np.max(np.abs(matrices[arm] - matrices["N"])))
        for r in U_ROWS:
            if exposure[str(r)] == 0.0:
                row["zero_exposure_rows"].append(r); visual_delta = float(np.max(np.abs(visuals["N"][r] - visuals["CORRECT"][r]))); matrix_delta = float(np.max(np.abs(matrices["N"][r] - matrices["CORRECT"][r])))
                if visual_delta > 1e-4 or matrix_delta > 1e-4:
                    row["zero_exposure_violations"].append({"row": r, "visual_max_abs": visual_delta, "matrix_max_abs": matrix_delta})
        row["exposure_min"] = min(exposure.values()); row["exposure_max"] = max(exposure.values()); per.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per:
        grouped[row["source_group"]].append(row)
    stats = {}
    for arm in ("CORRECT", "WRONG", "SHUFFLE"):
        groups = []; identifiable = True
        for group, items in sorted(grouped.items()):
            beta_num = beta_den = local_num = local_den = 0.0
            for item in items:
                exposure = np.asarray([item["exposure"][str(r)] for r in U_ROWS], dtype=np.float64); delta = np.asarray(item[arm]["d_rows"], dtype=np.float64); centered_exposure = exposure - np.mean(exposure); centered_delta = delta - np.mean(delta); beta_num += float(np.mean(centered_exposure * centered_delta)); beta_den += float(np.mean(centered_exposure * centered_exposure)); local_num += float(np.sum(exposure * delta)); local_den += float(np.sum(exposure))
            if beta_den <= 1e-12 or local_den <= 1e-12:
                identifiable = False
            groups.append({"source_group": group, "beta": beta_num / beta_den if beta_den > 1e-12 else None, "local_gain": local_num / local_den if local_den > 1e-12 else None})
        stats[arm] = {"identifiable": True, "groups": groups, "beta": _stats([{"source_group": row["source_group"], "value": row["beta"]} for row in groups], "value"), "local_gain": _stats([{"source_group": row["source_group"], "value": row["local_gain"]} for row in groups], "value")} if identifiable else {"identifiable": False, "groups": groups}
    return {"records": per, "stats": stats, "K_recomputed": True, "zero_exposure_checked": True, "zero_exposure_violation_count": int(sum(len(row["zero_exposure_violations"]) for row in per)), "parent_full_U_delta_A": float(np.mean([row["CORRECT"]["d_mean"] for row in per]))}


def _recompute() -> dict[str, Any]:
    records = _records(); a = _audit_a(records); b = _audit_b(records); c = _audit_c(records)
    return {"schema_version": 1, "record_count": len(records), "group_count": 8, "A": a, "B": b, "C": c, "contract_violations": {"A": ["old validator reused producer statistics; independent matrix/stat recomputation performed"], "B": ["legacy helper uses offset=15-j for delayed audio; matched domain recomputed with 10-j", "old 13/16 gate is not the matched-domain result"], "C": ["old validator reused producer supports/row_supports", "old joint-positive rule was marginal counts; independent same-group conjunction used", "zero-exposure rows checked one by one"], "final_binding": ["old B final does not bind independent validation"]}, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "f46_candidate_created": False}


def validate(root: Path) -> dict[str, Any]:
    paths = config.RunPaths(root)
    load_self_hashed(paths.protocol)
    expected = _recompute()
    actual = load_self_hashed(paths.recomputed)
    for field in ("record_count", "group_count", "A", "B", "C", "contract_violations", "replacement_confirmed", "waveform_head_authorized", "generalization_established", "historical_shift_gate_repaired", "f46_candidate_created"):
        if expected[field] != actual.get(field):
            raise ProtocolError(f"independent E0 mismatch: {field}")
    result = {"schema_version": 1, "status": "PASS", "engineering_decision": "GO", "scientific_decision": "CONTRACT_VIOLATION_REPRODUCED", "independent": True, "A": expected["A"]["contrasts"], "B": {"legacy_offset_pass_count": expected["B"]["legacy_offset_pass_count"], "matched_offset_pass_count": expected["B"]["matched_offset_pass_count"], "f46_boundary_restored": expected["B"]["f46_boundary_restored"], "f46_control_as_spec": expected["B"]["f46_control_as_spec"], "damage": expected["B"]["damage"]}, "C": {"stats": expected["C"]["stats"], "zero_exposure_violation_count": expected["C"]["zero_exposure_violation_count"]}, "candidate_authorized": False, "checks": ["independent A embeddings-to-matrix and statistics", "independent B lag domains", "independent C support and same-group rule", "every zero-exposure row recorded", "no F46_C authorization"]}
    return write_json(paths.validation, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); args = parser.parse_args(argv)
    try:
        validate(args.run_root)
        return 0
    except Exception as exc:
        write_json(args.run_root / "validation.json", {"schema_version": 1, "status": "FAIL", "engineering_decision": "BLOCKED", "scientific_decision": "not_available", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
