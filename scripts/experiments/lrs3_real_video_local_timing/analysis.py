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
from .review import create_review_package


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(numeric)
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
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(statistics.fmean(values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "group_means": group_means,
    }


def _score_index(score_manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = score_manifest.get("scores")
    if score_manifest.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.expected_cell_count():
        raise DiagnosticError("score manifest is incomplete")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("arm")))
        if key in result or key[1] not in config.ARMS:
            raise DiagnosticError(f"duplicate or unknown score cell: {key}")
        result[key] = row
    return result


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    matrix = np.load(Path(str(row["matrix"])), allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != 31 or not np.isfinite(matrix).all():
        raise DiagnosticError("score matrix is malformed")
    return np.asarray(matrix, dtype=np.float64)


def _local(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row.get("local", {}).get(name)
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"local curve is missing: {row.get('sample_id')}/{row.get('arm')}/{name}")
    return value


def analyze(protocol: Mapping[str, Any], score_manifest: Mapping[str, Any]) -> dict[str, Any]:
    index = _score_index(score_manifest)
    records = list(protocol.get("records", []))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol record count is invalid")
    groups = [str(record["source_group"]) for record in records]
    per_record: list[dict[str, Any]] = []
    repeatability_ok = 0
    baseline_ok = 0
    success_count = 0
    c_damage: list[float] = []
    d_damage: list[float] = []
    for record in records:
        sample_id = str(record["sample_id"])
        real = index[(sample_id, config.REAL_ARM)]
        repeat = index[(sample_id, config.REPEAT_ARM)]
        warp = index[(sample_id, config.WARP_ARM)]
        real_matrix = _matrix(real)
        repeat_matrix = _matrix(repeat)
        if real_matrix.shape != repeat_matrix.shape:
            raise DiagnosticError(f"repeat matrix shape differs: {sample_id}")
        matrix_max_abs = float(np.max(np.abs(real_matrix - repeat_matrix)))
        real_local_plus = _local(real, "PLUS")
        real_local_minus = _local(real, "MINUS")
        repeat_local_plus = _local(repeat, "PLUS")
        repeat_local_minus = _local(repeat, "MINUS")
        warp_local_plus = _local(warp, "PLUS")
        warp_local_minus = _local(warp, "MINUS")
        repeat_pass = bool(
            matrix_max_abs <= 0.001
            and int(real["reconstructed"]["offset"]) == int(repeat["reconstructed"]["offset"])
            and int(real_local_plus["offset"]) == int(repeat_local_plus["offset"])
            and int(real_local_minus["offset"]) == int(repeat_local_minus["offset"])
        )
        if repeat_pass:
            repeatability_ok += 1
        baseline_peak_clear = bool(real_local_plus["clear"] and real_local_minus["clear"])
        baseline_direction_agrees = abs(int(real_local_plus["offset"]) - int(real_local_minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES
        baseline_pass = bool(baseline_peak_clear and baseline_direction_agrees)
        if baseline_pass:
            baseline_ok += 1
        delta_plus = int(warp_local_plus["offset"]) - int(real_local_plus["offset"])
        delta_minus = int(warp_local_minus["offset"]) - int(real_local_minus["offset"])
        plus_error = delta_plus + 3
        minus_error = delta_minus - 3
        warp_peak_clear = bool(warp_local_plus["clear"] and warp_local_minus["clear"])
        recovered = bool(baseline_pass and warp_peak_clear and abs(plus_error) <= config.OFFSET_TOLERANCE_FRAMES and abs(minus_error) <= config.OFFSET_TOLERANCE_FRAMES)
        if recovered:
            success_count += 1
        c_value = float(real["reconstructed"]["sync_c"]) - float(warp["reconstructed"]["sync_c"])
        d_value = float(warp["reconstructed"]["sync_d"]) - float(real["reconstructed"]["sync_d"])
        c_damage.append(c_value)
        d_damage.append(d_value)
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "repeatability": {
                    "matrix_max_abs": matrix_max_abs,
                    "global_offset_equal": int(real["reconstructed"]["offset"]) == int(repeat["reconstructed"]["offset"]),
                    "plus_offset_equal": int(real_local_plus["offset"]) == int(repeat_local_plus["offset"]),
                    "minus_offset_equal": int(real_local_minus["offset"]) == int(repeat_local_minus["offset"]),
                    "passes": repeat_pass,
                },
                "baseline": {
                    "plus": {key: real_local_plus[key] for key in ("offset", "peak_gap", "clear", "rows")},
                    "minus": {key: real_local_minus[key] for key in ("offset", "peak_gap", "clear", "rows")},
                    "direction_difference": abs(int(real_local_plus["offset"]) - int(real_local_minus["offset"])),
                    "passes": baseline_pass,
                },
                "warp": {
                    "plus": {key: warp_local_plus[key] for key in ("offset", "peak_gap", "clear", "rows")},
                    "minus": {key: warp_local_minus[key] for key in ("offset", "peak_gap", "clear", "rows")},
                    "passes": warp_peak_clear,
                },
                "recovery": {
                    "delta_plus": delta_plus,
                    "delta_minus": delta_minus,
                    "plus_error": plus_error,
                    "minus_error": minus_error,
                    "passes": recovered,
                },
                "global": {
                    "real_sync_c": float(real["reconstructed"]["sync_c"]),
                    "warp_sync_c": float(warp["reconstructed"]["sync_c"]),
                    "c_real_minus_warp": c_value,
                    "real_sync_d": float(real["reconstructed"]["sync_d"]),
                    "warp_sync_d": float(warp["reconstructed"]["sync_d"]),
                    "d_warp_minus_real": d_value,
                },
            }
        )
    if repeatability_ok == config.EXPECTED_RECORD_COUNT and baseline_ok >= config.MIN_BASELINE_RECORDS:
        scientific_decision = "LOCAL_TIMING_DETECTED" if success_count >= config.MIN_SUCCESS_RECORDS else "LOCAL_TIMING_NOT_ESTABLISHED"
    elif repeatability_ok != config.EXPECTED_RECORD_COUNT:
        scientific_decision = "REPEATABILITY_FAILED"
    else:
        scientific_decision = "BASELINE_INCONCLUSIVE"
    return {
        "schema_version": 2,
        "analysis": "lrs3_real_video_local_timing_diagnostic",
        "protocol_revision": config.PROTOCOL_REVISION,
        "input_audit_sha256": str(protocol.get("input_audit", {}).get("sha256", "")),
        "scope": "fixed 22-record fit-only real-video cohort; natural audio unchanged; forward SyncNet only",
        "engineering_decision": "GO",
        "scientific_decision": scientific_decision,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "repeatability": {"count": repeatability_ok, "required": config.EXPECTED_RECORD_COUNT, "passes": repeatability_ok == config.EXPECTED_RECORD_COUNT},
        "baseline": {"clear_and_aligned_count": baseline_ok, "required": config.MIN_BASELINE_RECORDS, "passes": baseline_ok >= config.MIN_BASELINE_RECORDS},
        "recovery": {"success_count": success_count, "required": config.MIN_SUCCESS_RECORDS, "passes": success_count >= config.MIN_SUCCESS_RECORDS},
        "global_descriptive": {
            "c_real_minus_warp_sync_c": cluster_bootstrap(c_damage, groups),
            "d_warp_minus_real_sync_d": cluster_bootstrap(d_damage, groups, seed=config.BOOTSTRAP_SEED),
        },
        "per_record": per_record,
        "review_status": "PENDING",
        "limits": [
            "positive result supports only forward local sensitivity under this discrete real-video frame mapping",
            "frame repeats/skips and global visual-motion changes remain confounded",
            "fit-only, non-gradient, non-replacement-benefit, and non-TTS boundary",
        ],
    }


def result_markdown(analysis: Mapping[str, Any], review_status: str) -> str:
    lines = [
        "# LRS3 真实视频局部时间敏感性诊断",
        "",
        f"- Engineering: `{analysis['engineering_decision']}`",
        f"- Scientific decision: `{analysis['scientific_decision']}`",
        f"- Cohort: {analysis['record_count']} records / {analysis['source_group_count']} source groups",
        f"- Repeatability: {analysis['repeatability']['count']}/{analysis['repeatability']['required']}",
        f"- Baseline clear and aligned: {analysis['baseline']['clear_and_aligned_count']}/{analysis['record_count']}",
        f"- Local recovery: {analysis['recovery']['success_count']}/{analysis['record_count']}",
        f"- Blind review: `{review_status}`",
        "",
        "## Per-record diagnostic",
        "",
        "| sample_id | REAL + | WARP + | Δ+ | REAL − | WARP − | Δ− | recovered |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in analysis["per_record"]:
        lines.append(
            f"| {row['sample_id']} | {row['baseline']['plus']['offset']} | {row['warp']['plus']['offset']} | {row['recovery']['delta_plus']} | "
            f"{row['baseline']['minus']['offset']} | {row['warp']['minus']['offset']} | {row['recovery']['delta_minus']} | {row['recovery']['passes']} |"
        )
    c = analysis["global_descriptive"]["c_real_minus_warp_sync_c"]
    d = analysis["global_descriptive"]["d_warp_minus_real_sync_d"]
    lines.extend(
        [
            "",
            "## Descriptive global changes",
            "",
            f"- C(REAL) − C(WARP): mean `{c['mean']:.6f}`, 95% CI `[{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}]` (display C to 3 decimals: `{c['mean']:.3f}`).",
            f"- D(WARP) − D(REAL): mean `{d['mean']:.6f}`, 95% CI `[{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]`.",
            "",
            "## Boundary",
            "",
            "本报告只回答固定真实视频、自然音频不变、离散视频帧采样扰动下的 SyncNet 前向局部敏感性。它不证明 TTS/TFG 收益、Wav2Lip 因果、梯度可用性或可部署策略；帧重复/跳帧及全局视觉运动变化仍是混杂因素。",
            "",
        ]
    )
    return "\n".join(lines)


def report(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], score_manifest: Mapping[str, Any], paths) -> dict[str, Any]:
    if paths.final.is_file():
        raise DiagnosticError("terminal run cannot be reported again")
    analysis = analyze(protocol, score_manifest)
    review = create_review_package(protocol, media_manifest, paths)
    analysis["review_status"] = review["status"]
    analysis_path = paths.root / "analysis.json"
    write_self_hashed_json(analysis_path, analysis)
    paths.result.write_text(result_markdown(analysis, review["status"]), encoding="utf-8")
    final_payload = {
        "schema_version": 2,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": analysis["scientific_decision"],
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "cell_count": config.expected_cell_count(),
        "protocol_sha256": str(protocol["_sha256"]),
        "input_audit": dict(protocol["input_audit"]),
        "input_audit_sha256": str(protocol["input_audit"]["sha256"]),
        "spec_bindings": dict(protocol["spec_bindings"]),
        "parent_blocked_run": dict(protocol["parent_blocked_run"]),
        "media_manifest_sha256": file_sha256(paths.media / "manifest.json"),
        "score_manifest_sha256": file_sha256(paths.scores / "manifest.json"),
        "analysis_sha256": file_sha256(analysis_path),
        "result_sha256": file_sha256(paths.result),
        "review_status": review["status"],
        "review_manifest_sha256": review["manifest_sha256"],
        "summary": {
            "repeatability_count": analysis["repeatability"]["count"],
            "baseline_count": analysis["baseline"]["clear_and_aligned_count"],
            "recovery_count": analysis["recovery"]["success_count"],
        },
        "eligibility": False,
    }
    write_self_hashed_json(paths.final, final_payload)
    return verify_self_hashed_json(paths.final)
