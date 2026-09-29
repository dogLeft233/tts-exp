from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, verify_self_hashed_json, write_self_hashed_json


def bootstrap_stats(values: Sequence[float], groups: Sequence[str], *, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.BOOTSTRAP_SEED) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray([str(value) for value in groups])
    if values.ndim != 1 or values.size != groups.size or values.size == 0 or not np.isfinite(values).all():
        raise ProtocolError("bootstrap values/groups are invalid")
    labels = list(dict.fromkeys(groups.tolist()))
    group_means = np.asarray([float(np.mean(values[groups == label])) for label in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(labels), size=(draws, len(labels)))
    boot = group_means[sample_indices].mean(axis=1)
    return {
        "mean": float(np.mean(group_means)),
        "median": float(np.median(group_means)),
        "ci95": [float(value) for value in np.percentile(boot, [2.5, 97.5], method="linear")],
        "group_count": len(labels),
        "draws": int(draws),
        "seed": int(seed),
        "group_means": {label: float(value) for label, value in zip(labels, group_means, strict=True)},
    }


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row["matrix_path"]))
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 * config.SYNCNET_VSHIFT + 1:
        raise ProtocolError(f"invalid matrix shape: {path}")
    return value


def common_w(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        raise ProtocolError("cannot compute W from no score rows")
    max_rows = min(_matrix(row).shape[0] for row in rows)
    mask = np.ones(max_rows, dtype=bool)
    for row in rows:
        matrix = _matrix(row)[:max_rows]
        mask &= np.isfinite(matrix).all(axis=1)
    indices = np.flatnonzero(mask)
    if indices.size < config.MIN_COMMON_WINDOWS:
        raise ProtocolError(f"common W has only {indices.size} windows")
    return indices.astype(np.int64)


def score_metrics(row: Mapping[str, Any], W: np.ndarray, k0: int | None = None) -> dict[str, Any]:
    matrix = _matrix(row)
    if W.size == 0 or int(W.max()) >= matrix.shape[0]:
        raise ProtocolError("W is outside matrix support")
    curve = np.mean(matrix[W], axis=0)
    if not np.isfinite(curve).all():
        raise ProtocolError("score curve contains non-finite values")
    min_value = float(np.min(curve))
    min_index = int(np.flatnonzero(np.isclose(curve, min_value, rtol=0.0, atol=1e-12))[0])
    lags = list(range(-config.SYNCNET_VSHIFT, config.SYNCNET_VSHIFT + 1))
    best_lag = int(lags[min_index])
    result = {
        "C": float(np.median(curve) - min_value),
        "D": min_value,
        "k": best_lag,
        "k_index": min_index,
        "curve": [float(value) for value in curve],
        "lags": lags,
        "W": [int(value) for value in W],
        "window_count": int(W.size),
        "D0": float(curve[config.SYNCNET_VSHIFT]),
    }
    if k0 is not None:
        if not -config.SYNCNET_VSHIFT <= int(k0) <= config.SYNCNET_VSHIFT:
            raise ProtocolError(f"k0 is outside lag support: {k0}")
        result["k0"] = int(k0)
        result["D_anchor"] = float(curve[int(k0) + config.SYNCNET_VSHIFT])
    return result


def _row(score_rows: Mapping[tuple[str, str, str], Mapping[str, Any]], sid: str, video: str, audio: str) -> Mapping[str, Any]:
    try:
        return score_rows[(sid, video, audio)]
    except KeyError as exc:
        raise ProtocolError(f"missing score cell {sid}/{video}/{audio}") from exc


def _pair_values(metrics: Mapping[str, Mapping[str, Any]], left: str, right: str, field: str, direction: int = 1) -> float:
    return direction * (float(metrics[left][field]) - float(metrics[right][field]))


def _counts(values: Sequence[float], threshold: float = 0.0) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "improved": int(sum(value > threshold for value in values)),
        "tied": int(sum(abs(value) <= 1e-12 for value in values)),
        "declined": int(sum(value < -threshold for value in values)),
        "positive_any": int(sum(value > 0.0 for value in values)),
        "denominator": len(values),
        "values": values,
    }


def analyze_stage_a(paths: config.RunPaths, inputs: Mapping[str, Any], scores: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    groups = []
    repeat_c, repeat_d, repeat_k = [], [], []
    delay_k, delay_anchor = [], []
    repeat_ok = []
    delay_ok = []
    delay_boundary_ok = []
    for item in inputs["records"]:
        sid = str(item["sample_id"])
        group = str(item["source_group"])
        baseline_row = _row(scores, sid, "N", "N")
        repeat_row = _row(scores, sid, "N_REPEAT", "N")
        delay_row = _row(scores, sid, "N", "ND")
        W = common_w((baseline_row, repeat_row, delay_row))
        base = score_metrics(baseline_row, W)
        base["D_anchor"] = base["D"]
        repeat = score_metrics(repeat_row, W, base["k"])
        delayed = score_metrics(delay_row, W, base["k"])
        dc = float(repeat["C"] - base["C"])
        dd = float(repeat["D"] - base["D"])
        dk = int(repeat["k"] - base["k"])
        shift = int(delayed["k"] - base["k"])
        anchor = float(delayed["D_anchor"] - base["D_anchor"])
        repeat_c.append(dc)
        repeat_d.append(dd)
        repeat_k.append(dk)
        delay_k.append(shift)
        delay_anchor.append(anchor)
        repeat_ok.append(abs(dc) <= config.REPEAT_RECORD_ABS_BOUND and abs(dd) <= config.REPEAT_RECORD_ABS_BOUND and abs(dk) <= config.REPEAT_OFFSET_BOUND)
        delay_ok.append(abs(shift - 5) <= config.DELAY_OFFSET_TOLERANCE)
        delay_boundary_ok.append(abs(delayed["k"]) < config.SYNCNET_VSHIFT and abs(base["k"]) < config.SYNCNET_VSHIFT)
        records.append({"sample_id": sid, "source_group": group, "W": W.tolist(), "baseline": base, "repeat": repeat, "delay": delayed, "repeat_delta": {"C": dc, "D": dd, "k": dk}, "delay_control": {"k_shift": shift, "expected_shift": 5, "D_anchor_damage": anchor}})
        groups.append(group)
    repeat_stats_c = bootstrap_stats(repeat_c, groups)
    repeat_stats_d = bootstrap_stats(repeat_d, groups)
    delay_stats = bootstrap_stats(delay_anchor, groups)
    repeat_pass = bool(
        repeat_stats_c["ci95"][0] >= -config.REPEAT_DIFF_BOUND and repeat_stats_c["ci95"][1] <= config.REPEAT_DIFF_BOUND
        and repeat_stats_d["ci95"][0] >= -config.REPEAT_DIFF_BOUND and repeat_stats_d["ci95"][1] <= config.REPEAT_DIFF_BOUND
        and sum(repeat_ok) >= 20
    )
    delay_pass = bool(
        sum(delay_ok) >= config.DELAY_MIN_RECORDS
        and sum(delay_boundary_ok) >= config.DELAY_MIN_RECORDS
        and delay_stats["ci95"][0] > config.DELAY_D_ANCHOR_THRESHOLD
        and sum(value > 0.0 for value in delay_anchor) >= config.DELAY_POSITIVE_MIN_RECORDS
    )
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "A",
        "status": "complete",
        "record_count": len(records),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group", "ci": "two-sided percentile with linear interpolation"},
        "repeatability": {"C": repeat_stats_c, "D": repeat_stats_d, "k": {"values": repeat_k, "count_within_bound": int(sum(abs(value) <= config.REPEAT_OFFSET_BOUND for value in repeat_k))}, "record_pass_count": int(sum(repeat_ok)), "passes": repeat_pass},
        "delay_control": {"k_shift": {"values": delay_k, "expected": 5, "count_within_tolerance": int(sum(delay_ok))}, "boundary_count": int(sum(delay_boundary_ok)), "D_anchor_damage": delay_stats, "positive_count": int(sum(value > 0.0 for value in delay_anchor)), "passes": delay_pass},
        "measurement_decision": "PASS" if repeat_pass and delay_pass else "MEASUREMENT_CONTROL_FAILED",
        "records": records,
    }
    write_self_hashed_json(paths.stage_a, payload)
    return verify_self_hashed_json(paths.stage_a)


def _segment_rows(matrix_rows: Sequence[Mapping[str, Any]], k0: int, start: int, end: int) -> np.ndarray:
    if end - start < config.MIN_SWAP_WINDOWS:
        return np.asarray([], dtype=np.int64)
    max_rows = min(_matrix(row).shape[0] for row in matrix_rows)
    mask = np.zeros(max_rows, dtype=bool)
    column = k0 + config.SYNCNET_VSHIFT
    for t in range(max(0, start), min(end - 5, max_rows)):
        audio_t = t + k0
        if start <= audio_t and audio_t + 4 < end and all(np.isfinite(_matrix(row)[t, column]) for row in matrix_rows):
            mask[t] = True
    return np.flatnonzero(mask).astype(np.int64)


def analyze_full(paths: config.RunPaths, inputs: Mapping[str, Any], scores: Mapping[tuple[str, str, str], Mapping[str, Any]], stage_a: Mapping[str, Any]) -> dict[str, Any]:
    if stage_a.get("measurement_decision") != "PASS":
        final = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "integrity": "VALID",
            "measurement": "MEASUREMENT_CONTROL_FAILED",
            "bridge_gain": "NOT_EVALUATED",
            "swap_transfer": "NOT_EVALUATED",
            "reason": "stage A measurement gate failed; stage B was not run",
            "training_authorized": False,
            "generalization_established": False,
            "mouth_leakage_proven": False,
            "historical_gate_repaired": False,
        }
        write_self_hashed_json(paths.final, final)
        return verify_self_hashed_json(paths.final)

    records = []
    gain_n_c, gain_rt_c, gain_n_d, gain_rt_d, gain_n_anchor, gain_rt_anchor = [], [], [], [], [], []
    b_offset_n, b_offset_rt = [], []
    local_values: dict[str, list[float]] = {"p_N_2": [], "p_S_2": [], "p_N_3": [], "p_S_3": []}
    local_groups: list[str] = []
    local_positive = []
    for item in inputs["records"]:
        sid = str(item["sample_id"])
        group = str(item["source_group"])
        cells = [_row(scores, sid, video, audio) for video, audio in (("N", "N"), ("RT", "N"), ("B", "N"), ("B", "B"), ("N", "S"), ("S", "N"), ("S", "S"))]
        W = common_w(cells)
        baseline = score_metrics(cells[0], W)
        baseline["D_anchor"] = baseline["D"]
        metrics = {
            name: score_metrics(row, W, baseline["k"])
            for name, row in zip(("N_N", "RT_N", "B_N", "B_B", "N_S", "S_N", "S_S"), cells, strict=True)
        }
        n_c = metrics["B_N"]["C"] - metrics["N_N"]["C"]
        rt_c = metrics["B_N"]["C"] - metrics["RT_N"]["C"]
        n_d = metrics["N_N"]["D"] - metrics["B_N"]["D"]
        rt_d = metrics["RT_N"]["D"] - metrics["B_N"]["D"]
        n_a = metrics["N_N"]["D_anchor"] - metrics["B_N"]["D_anchor"]
        rt_a = metrics["RT_N"]["D_anchor"] - metrics["B_N"]["D_anchor"]
        gain_n_c.append(n_c); gain_rt_c.append(rt_c); gain_n_d.append(n_d); gain_rt_d.append(rt_d); gain_n_anchor.append(n_a); gain_rt_anchor.append(rt_a)
        b_offset_n.append(abs(int(metrics["B_N"]["k"]) - int(metrics["N_N"]["k"])))
        b_offset_rt.append(abs(int(metrics["B_N"]["k"]) - int(metrics["RT_N"]["k"])))
        natural_samples = int(item["natural_sample_count"])
        b1, b2, b3 = natural_samples // 4, natural_samples // 2, (3 * natural_samples) // 4
        boundaries = [int(value // config.SAMPLES_PER_FRAME) for value in (b1, b2, b3)]
        segments = {"2": (boundaries[0], boundaries[1]), "3": (boundaries[1], boundaries[2])}
        local_record: dict[str, Any] = {"sample_id": sid, "source_group": group, "W": W.tolist(), "metrics": metrics, "segments": {}}
        positive_terms: list[float] = []
        for segment_name, (start, end) in segments.items():
            segment_w = _segment_rows((cells[0], cells[2], cells[4], cells[5], cells[6]), baseline["k"], start, end)
            if segment_w.size < config.MIN_SWAP_WINDOWS:
                raise ProtocolError(f"LOCAL_SWAP segment {segment_name} has insufficient support: {sid}")
            column = baseline["k"] + config.SYNCNET_VSHIFT
            p_n = float(np.mean(_matrix(cells[4])[segment_w, column] - _matrix(cells[0])[segment_w, column]))
            p_s = float(np.mean(_matrix(cells[5])[segment_w, column] - _matrix(cells[6])[segment_w, column]))
            d0 = {name: float(metrics[name]["curve"][config.SYNCNET_VSHIFT]) for name in ("N_S", "N_N", "S_N", "S_S")}
            local_values[f"p_N_{segment_name}"].append(p_n)
            local_values[f"p_S_{segment_name}"].append(p_s)
            positive_terms.extend([p_n, p_s])
            local_record["segments"][segment_name] = {"W": segment_w.tolist(), "p_N": p_n, "p_S": p_s, "D0": d0, "curve_support": {name: _matrix(row)[segment_w].shape[0] for name, row in (("N_S", cells[4]), ("N_N", cells[0]), ("S_N", cells[5]), ("S_S", cells[6]))}}
        local_positive.append(all(value > 0.0 for value in positive_terms))
        local_groups.append(group)
        records.append({"sample_id": sid, "source_group": group, "gains": {"B_minus_N": {"C": n_c, "D": n_d, "D_anchor": n_a}, "B_minus_RT": {"C": rt_c, "D": rt_d, "D_anchor": rt_a}}, "offset_diff": {"B_N": b_offset_n[-1], "B_RT": b_offset_rt[-1]}, "local_swap": local_record})

    def gain_bundle(c: Sequence[float], d: Sequence[float], anchor: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
        return {"C": bootstrap_stats(c, groups), "D": bootstrap_stats(d, groups), "D_anchor": bootstrap_stats(anchor, groups), "counts": {"C": _counts(c), "D": _counts(d), "D_anchor": _counts(anchor)}}

    gains = {"B_minus_N": gain_bundle(gain_n_c, gain_n_d, gain_n_anchor, [row["source_group"] for row in records]), "B_minus_RT": gain_bundle(gain_rt_c, gain_rt_d, gain_rt_anchor, [row["source_group"] for row in records])}
    gain_pass = all(
        gains[key]["C"]["mean"] > config.GAIN_THRESHOLD
        and gains[key]["C"]["ci95"][0] > 0.0
        and gains[key]["D"]["ci95"][0] > -0.10
        and gains[key]["D_anchor"]["ci95"][0] > -0.10
        for key in ("B_minus_N", "B_minus_RT")
    ) and sum(value <= 1 for value in b_offset_n) >= 20 and sum(value <= 1 for value in b_offset_rt) >= 20
    swap_stats = {key: bootstrap_stats(values, local_groups) for key, values in local_values.items()}
    swap_pass = all(swap_stats[key]["ci95"][0] > config.SWAP_MARGIN for key in swap_stats) and sum(local_positive) >= config.SWAP_MIN_POSITIVE_RECORDS
    analysis = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(records),
        "source_group_count": len(records),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group", "ci": "two-sided percentile with linear interpolation"},
        "stage_a": stage_a,
        "bridge_gain": {"comparisons": gains, "offset_agreement": {"B_minus_N": {"differences": b_offset_n, "count_within_1": int(sum(value <= 1 for value in b_offset_n))}, "B_minus_RT": {"differences": b_offset_rt, "count_within_1": int(sum(value <= 1 for value in b_offset_rt))}}, "passes": gain_pass, "decision": "STATIC_BRIDGE_GAIN_OBSERVED" if gain_pass else "NO_STATIC_BRIDGE_GAIN_ESTABLISHED", "mel_movement_is_descriptive_only": True},
        "swap_transfer": {"segments": swap_stats, "positive_all_four_count": int(sum(local_positive)), "passes": swap_pass, "decision": "SWAP_TRANSFER_OBSERVED" if swap_pass else "SWAP_TRANSFER_UNRESOLVED", "reason": None if swap_pass else "LOCAL_SEGMENT_RESPONSE_GATE_NOT_MET"},
        "records": records,
        "training_authorized": False,
        "generalization_established": False,
        "mouth_leakage_proven": False,
        "historical_gate_repaired": False,
    }
    write_self_hashed_json(paths.analysis, analysis)
    final = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "integrity": "VALID",
        "measurement": "PASS",
        "bridge_gain": analysis["bridge_gain"]["decision"],
        "swap_transfer": analysis["swap_transfer"]["decision"],
        "human_review": "NOT_HUMAN_REVIEWED",
        "scientific_decision": analysis["bridge_gain"]["decision"],
        "training_authorized": False,
        "generalization_established": False,
        "mouth_leakage_proven": False,
        "historical_gate_repaired": False,
    }
    write_self_hashed_json(paths.final, final)
    return verify_self_hashed_json(paths.analysis)


def write_per_record_csv(paths: config.RunPaths, analysis: Mapping[str, Any]) -> None:
    rows = analysis.get("records", [])
    if not rows:
        return
    with paths.per_record_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_group", "B_minus_N_C", "B_minus_N_D", "B_minus_RT_C", "B_minus_RT_D", "swap_p_N_2", "swap_p_S_2", "swap_p_N_3", "swap_p_S_3"])
        for row in rows:
            gains = row["gains"]
            segments = row["local_swap"]["segments"]
            writer.writerow([row["sample_id"], row["source_group"], gains["B_minus_N"]["C"], gains["B_minus_N"]["D"], gains["B_minus_RT"]["C"], gains["B_minus_RT"]["D"], segments["2"]["p_N"], segments["2"]["p_S"], segments["3"]["p_N"], segments["3"]["p_S"]])
