from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DiagnosticError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .scoring import _common_global, _local_evidence, reconstruct_global


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(number)
    labels = sorted(by_group)
    group_means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([group_means[label] for label in sampled]))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "rng": "numpy_default_rng_pcg64",
        "labels": "sorted",
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(statistics.fmean(values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "group_means": group_means,
    }


def _score_index(score_manifest: Mapping[str, Any]) -> dict[tuple[str, str, str, bool], Mapping[str, Any]]:
    rows = score_manifest.get("scores")
    if score_manifest.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.EXPECTED_TOTAL_CELL_COUNT:
        raise DiagnosticError("score manifest is incomplete")
    result: dict[tuple[str, str, str, bool], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DiagnosticError("score row is malformed")
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")), bool(row.get("repeat", False)))
        if key in result or key[1] not in config.VIDEO_ARMS or key[2] not in config.AUDIO_ARMS:
            raise DiagnosticError(f"duplicate or unknown score cell: {key}")
        result[key] = row
    return result


def _record_cell(index: Mapping[tuple[str, str, str, bool], Mapping[str, Any]], sample_id: str, video_arm: str, audio_arm: str, repeat: bool = False) -> Mapping[str, Any]:
    try:
        return index[(sample_id, video_arm, audio_arm, repeat)]
    except KeyError as exc:
        raise DiagnosticError(f"score cell is missing: {sample_id}/{video_arm}/{audio_arm}/{repeat}") from exc


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", "")))
    if not path.is_file() or row.get("matrix_sha256") != file_sha256(path):
        raise DiagnosticError(f"score matrix is missing or changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise DiagnosticError(f"score matrix is malformed: {path}")
    return value


def _cell_evidence(record: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    video_arm = str(row["video_arm"])
    matrix = _matrix(row)
    expected = min(int(record["masks"]["frame_counts"][video_arm]), int(record["natural_audio"]["sample_count"]) // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if matrix.shape != (expected, 31):
        raise DiagnosticError(f"score matrix shape differs from protocol: {row.get('cell_id')}")
    mask = record["masks"]
    local = {"PLUS": _local_evidence(matrix, mask, "PLUS"), "MINUS": _local_evidence(matrix, mask, "MINUS")}
    return {"matrix": matrix, "official_global": reconstruct_global(matrix), "common_global": _common_global(matrix, mask), "local": local}


def _baseline_info(main: Mapping[str, Any], repeat: Mapping[str, Any], main_evidence: Mapping[str, Any], repeat_evidence: Mapping[str, Any]) -> dict[str, Any]:
    main_matrix = main_evidence["matrix"]
    repeat_matrix = repeat_evidence["matrix"]
    shape_equal = main_matrix.shape == repeat_matrix.shape
    matrix_max_abs = float(np.max(np.abs(main_matrix - repeat_matrix))) if shape_equal else None
    offsets_equal = {
        "global": int(main_evidence["official_global"]["offset"]) == int(repeat_evidence["official_global"]["offset"]),
        "plus": int(main_evidence["local"]["PLUS"]["offset"]) == int(repeat_evidence["local"]["PLUS"]["offset"]),
        "minus": int(main_evidence["local"]["MINUS"]["offset"]) == int(repeat_evidence["local"]["MINUS"]["offset"]),
    }
    return {
        "matrix_shape_equal": shape_equal,
        "matrix_max_abs": matrix_max_abs,
        "offsets_equal": offsets_equal,
        "passes": bool(shape_equal and matrix_max_abs is not None and matrix_max_abs <= 0.001 and all(offsets_equal.values())),
    }


def _baseline_explainable(evidence: Mapping[str, Any]) -> dict[str, Any]:
    plus = evidence["local"]["PLUS"]
    minus = evidence["local"]["MINUS"]
    return {
        "plus": {key: plus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "minus": {key: minus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "direction_difference": abs(int(plus["offset"]) - int(minus["offset"])),
        "passes": bool(plus["clear"] and minus["clear"] and abs(int(plus["offset"]) - int(minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES),
    }


def _check_pair(left: Mapping[str, Any], right: Mapping[str, Any], expected: Mapping[str, Any], expected_keys: tuple[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected_key in zip(("PLUS", "MINUS"), expected_keys, strict=True):
        actual = float(left["local"][name]["offset"]) - float(right["local"][name]["offset"])
        target = float(expected[expected_key])
        result[name] = {"actual": actual, "expected": target, "error": actual - target, "passes": bool(left["local"][name]["clear"] and right["local"][name]["clear"] and abs(actual - target) <= config.OFFSET_TOLERANCE_FRAMES)}
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _descriptive_pair(left: Mapping[str, Any], right: Mapping[str, Any], metric: str) -> float:
    return float(left["common_global"][metric]) - float(right["common_global"][metric])


def decision_from_counts(repeatability_count: int, r_baseline_count: int, gn_baseline_count: int, check_counts: Mapping[str, int]) -> str:
    """Apply the frozen priority tree to already computed counts."""
    if repeatability_count != config.EXPECTED_RECORD_COUNT:
        return "REPEATABILITY_FAILED"
    if r_baseline_count < config.MIN_BASELINE_RECORDS or gn_baseline_count < config.MIN_BASELINE_RECORDS:
        return "BASELINE_INCONCLUSIVE"
    if int(check_counts.get("A", 0)) < config.MIN_SUCCESS_RECORDS:
        return "AUDIO_CONTROL_UNRESOLVED"
    if int(check_counts.get("B", 0)) < config.MIN_SUCCESS_RECORDS:
        return "GENERATED_ENDPOINT_UNRESOLVED"
    if int(check_counts.get("C", 0)) < config.MIN_SUCCESS_RECORDS or int(check_counts.get("O", 0)) < config.MIN_SUCCESS_RECORDS:
        return "GENERATED_RESPONSE_UNRESOLVED"
    return "LOCAL_RESPONSE_ESTABLISHED"


def analyze(protocol: Mapping[str, Any], score_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol record count is invalid")
    index = _score_index(score_manifest)
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol sample identity is invalid")
    per_record: list[dict[str, Any]] = []
    groups: list[str] = []
    self_c: list[float] = []
    self_d: list[float] = []
    replacement_c: list[float] = []
    replacement_d: list[float] = []
    repeatability_count = 0
    r_baseline_count = 0
    gn_baseline_count = 0
    check_counts = {"A": 0, "B": 0, "C": 0, "O": 0}
    for record in records:
        sample_id = str(record["sample_id"])
        groups.append(str(record["source_group"]))
        rn = _record_cell(index, sample_id, config.VIDEO_R, config.AUDIO_N)
        rw = _record_cell(index, sample_id, config.VIDEO_R, config.AUDIO_W)
        gnn = _record_cell(index, sample_id, config.VIDEO_GN, config.AUDIO_N)
        gnw = _record_cell(index, sample_id, config.VIDEO_GN, config.AUDIO_W)
        gwn = _record_cell(index, sample_id, config.VIDEO_GW, config.AUDIO_N)
        gww = _record_cell(index, sample_id, config.VIDEO_GW, config.AUDIO_W)
        rn_repeat = _record_cell(index, sample_id, config.VIDEO_R, config.AUDIO_N, True)
        gnn_repeat = _record_cell(index, sample_id, config.VIDEO_GN, config.AUDIO_N, True)
        ev = {
            "rn": _cell_evidence(record, rn),
            "rw": _cell_evidence(record, rw),
            "gnn": _cell_evidence(record, gnn),
            "gnw": _cell_evidence(record, gnw),
            "gwn": _cell_evidence(record, gwn),
            "gww": _cell_evidence(record, gww),
            "rn_repeat": _cell_evidence(record, rn_repeat),
            "gnn_repeat": _cell_evidence(record, gnn_repeat),
        }
        r_repeat = _baseline_info(rn, rn_repeat, ev["rn"], ev["rn_repeat"])
        gn_repeat = _baseline_info(gnn, gnn_repeat, ev["gnn"], ev["gnn_repeat"])
        repeat_pass = bool(r_repeat["passes"] and gn_repeat["passes"])
        repeatability_count += int(repeat_pass)
        r_baseline = _baseline_explainable(ev["rn"])
        gn_baseline = _baseline_explainable(ev["gnn"])
        r_baseline_count += int(r_baseline["passes"])
        gn_baseline_count += int(gn_baseline["passes"])
        masks = record["masks"]
        check_a = _check_pair(ev["rw"], ev["rn"], masks, ("plus_expected_offset", "minus_expected_offset"))
        check_b = _check_pair(ev["gnw"], ev["gnn"], masks, ("plus_expected_offset", "minus_expected_offset"))
        check_c = _check_pair(ev["gwn"], ev["gnn"], masks, ("plus_expected_video_response", "minus_expected_video_response"))
        check_o = _check_pair(ev["gww"], ev["gnn"], {"plus_expected_offset": 0.0, "minus_expected_offset": 0.0}, ("plus_expected_offset", "minus_expected_offset"))
        # A/B/C/O additionally require their natural baseline to be interpretable.
        check_a["passes"] = bool(check_a["passes"] and r_baseline["passes"])
        check_b["passes"] = bool(check_b["passes"] and gn_baseline["passes"])
        check_c["passes"] = bool(check_c["passes"] and gn_baseline["passes"])
        check_o["passes"] = bool(check_o["passes"] and gn_baseline["passes"])
        checks = {"A": check_a, "B": check_b, "C": check_c, "O": check_o}
        for name, value in checks.items():
            check_counts[name] += int(bool(value["passes"]))
        self_c.append(_descriptive_pair(ev["gww"], ev["gnn"], "sync_c"))
        self_d.append(_descriptive_pair(ev["gnn"], ev["gww"], "sync_d"))
        replacement_c.append(_descriptive_pair(ev["gww"], ev["gwn"], "sync_c"))
        replacement_d.append(_descriptive_pair(ev["gwn"], ev["gww"], "sync_d"))
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "repeatability": {"R_N": r_repeat, "G_N_N": gn_repeat, "passes": repeat_pass},
                "baseline": {"R_N": r_baseline, "G_N_N": gn_baseline},
                "checks": checks,
                "descriptive": {"self_c": self_c[-1], "self_d": self_d[-1], "replacement_c": replacement_c[-1], "replacement_d": replacement_d[-1]},
            }
        )
    baseline_gate = bool(r_baseline_count >= config.MIN_BASELINE_RECORDS and gn_baseline_count >= config.MIN_BASELINE_RECORDS)
    decision = decision_from_counts(repeatability_count, r_baseline_count, gn_baseline_count, check_counts)
    return {
        "schema_version": 1,
        "analysis": "lrs3_wav2lip_timing_transfer_diagnostic",
        "protocol_revision": config.PROTOCOL_REVISION,
        "input_audit_sha256": str(protocol["input_audit"]["sha256"]),
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "repeatability": {"count": repeatability_count, "required": config.EXPECTED_RECORD_COUNT, "passes": repeatability_count == config.EXPECTED_RECORD_COUNT},
        "baseline": {"R_N_count": r_baseline_count, "G_N_N_count": gn_baseline_count, "required_each": config.MIN_BASELINE_RECORDS, "passes": baseline_gate},
        "checks": {name: {"count": count, "required": config.MIN_SUCCESS_RECORDS, "denominator": config.EXPECTED_RECORD_COUNT, "passes": count >= config.MIN_SUCCESS_RECORDS} for name, count in check_counts.items()},
        "descriptive": {
            "self_C_GW_W_minus_GN_N": cluster_bootstrap(self_c, groups, seed=config.BOOTSTRAP_SEED),
            "self_D_GN_N_minus_GW_W": cluster_bootstrap(self_d, groups, seed=config.BOOTSTRAP_SEED),
            "replacement_C_GW_W_minus_GW_N": cluster_bootstrap(replacement_c, groups, seed=config.BOOTSTRAP_SEED),
            "replacement_D_GW_N_minus_GW_W": cluster_bootstrap(replacement_d, groups, seed=config.BOOTSTRAP_SEED),
        },
        "per_record": per_record,
        "reference_conditioned_audio_head_spec_eligible": False,
        "limits": [
            "fit-only diagnostic on the already generated constant_full_frame_fallback Wav2Lip videos",
            "linear interpolation changes acoustic content as well as timing; A does not isolate pure acoustic damage",
            "fixed tail_v2 crop track and generated geometry are diagnostic controls, not a new Wav2Lip generation policy",
            "does not prove architecture causality, TTS benefit, training eligibility, or subjective evidence",
            "historical CONTROL_FAILED and bridge conclusions remain unchanged",
        ],
    }


def result_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# LRS3 Wav2Lip 局部时间传递诊断",
        "",
        f"- Engineering: `{analysis['engineering_decision']}`",
        f"- Scientific decision: `{analysis['scientific_decision']}`",
        f"- Cohort: {analysis['record_count']} records / {analysis['source_group_count']} source groups",
        f"- Scores: {config.EXPECTED_MAIN_CELL_COUNT} main + {config.EXPECTED_REPEAT_CELL_COUNT} repeat = {config.EXPECTED_TOTAL_CELL_COUNT}",
        f"- Repeatability: {analysis['repeatability']['count']}/{analysis['repeatability']['required']}",
        f"- Baseline R/N: {analysis['baseline']['R_N_count']}/{analysis['record_count']}; G_N/N: {analysis['baseline']['G_N_N_count']}/{analysis['record_count']}",
        "",
        "## Four diagnostic checks",
        "",
        "| check | meaning | pass |",
        "|---|---|---:|",
        f"| A | real-domain audio sensitivity | {analysis['checks']['A']['count']}/{analysis['checks']['A']['denominator']} |",
        f"| B | generated-domain audio sensitivity | {analysis['checks']['B']['count']}/{analysis['checks']['B']['denominator']} |",
        f"| C | generated video response | {analysis['checks']['C']['count']}/{analysis['checks']['C']['denominator']} |",
        f"| O | generated self local alignment | {analysis['checks']['O']['count']}/{analysis['checks']['O']['denominator']} |",
        "",
        "## Descriptive common-window changes",
        "",
    ]
    for key, label in (
        ("self_C_GW_W_minus_GN_N", "C(G_W/W) − C(G_N/N)"),
        ("replacement_C_GW_W_minus_GW_N", "C(G_W/W) − C(G_W/N)"),
        ("self_D_GN_N_minus_GW_W", "D(G_N/N) − D(G_W/W)"),
        ("replacement_D_GW_N_minus_GW_W", "D(G_W/N) − D(G_W/W)"),
    ):
        value = analysis["descriptive"][key]
        lines.append(f"- {label}: mean `{value['mean']:.3f}`, 95% CI `[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]`.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "本轮只诊断固定 22 条 fit-only 媒体、固定裁脸轨迹和官方 SyncNet 下的局部 timing 传递。历史 calibration 的 full-frame fallback geometry、W 的线性插值声学变化及固定裁脸均可能影响结果；上一轮盲审 PENDING 不构成本轮主观证据。无论本轮是否通过，`reference_conditioned_audio_head_spec_eligible` 均为 `false`，历史 CONTROL_FAILED 不被改写。",
            "",
        ]
    )
    return "\n".join(lines)


def report(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], score_manifest: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    if paths.final.is_file():
        raise DiagnosticError("terminal run cannot be reported again")
    analysis = analyze(protocol, score_manifest)
    analysis_path = paths.analysis
    write_self_hashed_json(analysis_path, analysis)
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(result_markdown(analysis), encoding="utf-8")
    final_payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": analysis["scientific_decision"],
        "reference_conditioned_audio_head_spec_eligible": False,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "main_cell_count": config.EXPECTED_MAIN_CELL_COUNT,
        "repeat_cell_count": config.EXPECTED_REPEAT_CELL_COUNT,
        "cell_count": config.EXPECTED_TOTAL_CELL_COUNT,
        "expected_counts": {"records": config.EXPECTED_RECORD_COUNT, "main_cells": config.EXPECTED_MAIN_CELL_COUNT, "repeat_cells": config.EXPECTED_REPEAT_CELL_COUNT, "total_cells": config.EXPECTED_TOTAL_CELL_COUNT},
        "protocol_sha256": str(protocol["_sha256"]),
        "input_audit": dict(protocol["input_audit"]),
        "input_audit_sha256": str(protocol["input_audit"]["sha256"]),
        "spec_bindings": dict(protocol["spec_bindings"]),
        "parent_runs": dict(protocol["parent_runs"]),
        "media_manifest_sha256": file_sha256(paths.media / "manifest.json"),
        "score_manifest_sha256": file_sha256(paths.scores / "manifest.json"),
        "analysis_sha256": file_sha256(analysis_path),
        "result_sha256": file_sha256(paths.result),
        "gate_counts": {name: int(value["count"]) for name, value in analysis["checks"].items()},
        "baseline_counts": {"R_N": int(analysis["baseline"]["R_N_count"]), "G_N_N": int(analysis["baseline"]["G_N_N_count"])},
        "repeatability_count": int(analysis["repeatability"]["count"]),
        "eligibility": False,
    }
    write_self_hashed_json(paths.final, final_payload)
    return verify_self_hashed_json(paths.final)
