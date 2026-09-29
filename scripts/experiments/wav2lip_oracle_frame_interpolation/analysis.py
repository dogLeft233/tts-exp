from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    OracleError,
    file_sha256,
)

from . import config


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", "")))
    expected = str(row.get("matrix_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise OracleError(f"matrix binding is invalid: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise OracleError(f"matrix is malformed: {path}")
    return value


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise OracleError("SyncNet curve must have 31 finite values")
    order = np.sort(values, kind="stable")
    index = int(np.argmin(values))
    offset = config.VSHIFT - index
    gap = float(order[1] - order[0])
    return {
        "min_index": index,
        "offset": int(offset),
        "sync_d": float(order[0]),
        "sync_c": float(np.median(values) - order[0]),
        "minimum": float(order[0]),
        "second_minimum": float(order[1]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT),
    }


def _signature(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_name = {
        "global": list(range(matrix.shape[0])),
        "common": [int(row) for row in masks["common_window_rows"]],
        "PLUS": [int(row) for row in masks["plus_rows"]],
        "MINUS": [int(row) for row in masks["minus_rows"]],
    }
    result: dict[str, Any] = {}
    for name, rows in rows_by_name.items():
        if not rows or min(rows) < 0 or max(rows) >= matrix.shape[0]:
            raise OracleError(f"invalid {name} rows for matrix with {matrix.shape[0]} rows")
        curve = np.mean(matrix[rows, :], axis=0, dtype=np.float64)
        evidence = _peak(curve)
        evidence["rows"] = rows
        evidence["curve"] = [float(item) for item in curve]
        result[name] = evidence
    return result


def _timing(left: Mapping[str, Any], right: Mapping[str, Any], masks: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == "B":
        expected = {
            "PLUS": float(np.mean([masks["a_by_row"][str(row)] for row in masks["plus_rows"]])),
            "MINUS": float(np.mean([masks["a_by_row"][str(row)] for row in masks["minus_rows"]])),
        }
    elif mode == "C_linear":
        expected = {
            "PLUS": float(-np.mean([masks["d_by_row"][str(row)] for row in masks["plus_rows"]])),
            "MINUS": float(-np.mean([masks["d_by_row"][str(row)] for row in masks["minus_rows"]])),
        }
    else:
        expected = {"PLUS": 0.0, "MINUS": 0.0}
    result: dict[str, Any] = {}
    for segment in ("PLUS", "MINUS"):
        actual = float(left[segment]["offset"] - right[segment]["offset"])
        target = expected[segment]
        result[segment] = {
            "actual": actual,
            "expected": target,
            "error": actual - target,
            "passes": bool(
                left[segment]["clear"]
                and right[segment]["clear"]
                and abs(actual - target) <= config.OFFSET_TOLERANCE_FRAMES
            ),
        }
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _baseline(signature: Mapping[str, Any]) -> bool:
    return bool(
        signature["PLUS"]["clear"]
        and signature["MINUS"]["clear"]
        and abs(signature["PLUS"]["offset"] - signature["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES
    )


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise OracleError("invalid bootstrap input")
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        if not np.isfinite(float(value)):
            raise OracleError("bootstrap input is non-finite")
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    means = {label: float(np.mean(grouped[label])) for label in labels}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([means[str(label)] for label in sampled]))
    return {
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [
            float(np.quantile(estimates, 0.025, method="linear")),
            float(np.quantile(estimates, 0.975, method="linear")),
        ],
        "group_means": means,
    }


def _cell_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise OracleError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "c": float(left["common"]["sync_c"] - right["common"]["sync_c"]),
        "d": float(left["common"]["sync_d"] - right["common"]["sync_d"]),
        "offset": int(left["common"]["offset"] - right["common"]["offset"]),
    }


def analyze(
    parent: Mapping[str, Any],
    protocol: Mapping[str, Any],
    fresh_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parent_cells = _cell_map(parent["score_rows"])
    fresh_cells = _cell_map(fresh_rows)
    expected = {(str(row["sample_id"]), config.VIDEO_ARM, audio) for row in protocol["records"] for audio in config.AUDIO_ARMS}
    if set(fresh_cells) != expected:
        raise OracleError(f"fresh cell set differs: {len(fresh_cells)}/{len(expected)}")

    per_record: list[dict[str, Any]] = []
    history_by_id = {
        str(row["sample_id"]): row
        for row in parent["analysis"].get("per_record", [])
        if isinstance(row, Mapping)
    }
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        masks = record["masks"]
        parent_signatures = {
            (video, audio): _signature(
                _matrix(parent_cells[(sample_id, video, audio)]), masks
            )
            for video in ("V_ID", "V_ORACLE")
            for audio in ("N", "W")
        }
        linear_signatures = {
            (config.VIDEO_ARM, audio): _signature(_matrix(fresh_cells[(sample_id, config.VIDEO_ARM, audio)]), masks)
            for audio in config.AUDIO_ARMS
        }
        baseline = _baseline(parent_signatures["V_ID", "N"])
        checks = {
            "B": _timing(parent_signatures["V_ID", "W"], parent_signatures["V_ID", "N"], masks, "B"),
            "C_linear": _timing(linear_signatures[config.VIDEO_ARM, "N"], parent_signatures["V_ID", "N"], masks, "C_linear"),
            "O_linear": _timing(linear_signatures[config.VIDEO_ARM, "W"], parent_signatures["V_ID", "N"], masks, "O_linear"),
        }
        own = _delta(linear_signatures[config.VIDEO_ARM, "W"], parent_signatures["V_ID", "N"])
        own["d"] = float(parent_signatures["V_ID", "N"]["common"]["sync_d"] - linear_signatures[config.VIDEO_ARM, "W"]["common"]["sync_d"])
        own["offset_agreement"] = bool(abs(int(own["offset"])) <= config.OFFSET_TOLERANCE_FRAMES)
        damage = _delta(linear_signatures[config.VIDEO_ARM, "W"], linear_signatures[config.VIDEO_ARM, "N"])
        damage["d"] = float(linear_signatures[config.VIDEO_ARM, "N"]["common"]["sync_d"] - linear_signatures[config.VIDEO_ARM, "W"]["common"]["sync_d"])
        damage["both_positive"] = bool(damage["c"] > 0.0 and damage["d"] > 0.0)
        gain = _delta(linear_signatures[config.VIDEO_ARM, "W"], parent_signatures["V_ORACLE", "W"])
        gain["d"] = float(parent_signatures["V_ORACLE", "W"]["common"]["sync_d"] - linear_signatures[config.VIDEO_ARM, "W"]["common"]["sync_d"])
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "historical_c_pass": history_by_id.get(sample_id, {}).get("historical_c_pass"),
                "baseline": baseline,
                "checks": checks,
                "own": own,
                "damage": damage,
                "gain": gain,
                "signatures": {f"{config.VIDEO_ARM}__{audio}": value for (video, audio), value in linear_signatures.items()},
                "parent_signatures": {f"{video}__{audio}": value for (video, audio), value in parent_signatures.items()},
            }
        )

    groups = [str(row["source_group"]) for row in per_record]
    own_bootstrap = {
        "c": _bootstrap([float(row["own"]["c"]) for row in per_record], groups),
        "d": _bootstrap([float(row["own"]["d"]) for row in per_record], groups),
    }
    damage_bootstrap = {
        "c": _bootstrap([float(row["damage"]["c"]) for row in per_record], groups),
        "d": _bootstrap([float(row["damage"]["d"]) for row in per_record], groups),
    }
    gain_bootstrap = {
        "c": _bootstrap([float(row["gain"]["c"]) for row in per_record], groups),
        "d": _bootstrap([float(row["gain"]["d"]) for row in per_record], groups),
    }
    baseline_count = sum(bool(row["baseline"]) for row in per_record)
    timing_counts = {
        name: sum(bool(row["checks"][name]["passes"]) for row in per_record)
        for name in ("B", "C_linear", "O_linear")
    }
    own_gate = {
        "ci_lower_gt_negative_0_10": own_bootstrap["c"]["ci95"][0] > -0.10 and own_bootstrap["d"]["ci95"][0] > -0.10,
        "offset_agreement_count": sum(bool(row["own"]["offset_agreement"]) for row in per_record),
    }
    own_gate["offset_agreement_pass"] = own_gate["offset_agreement_count"] >= config.MIN_BASELINE_RECORDS
    own_gate["passes"] = bool(own_gate["ci_lower_gt_negative_0_10"] and own_gate["offset_agreement_pass"])
    damage_gate = {
        "ci_lower_gt_0_10": damage_bootstrap["c"]["ci95"][0] > 0.10 and damage_bootstrap["d"]["ci95"][0] > 0.10,
        "both_positive_count": sum(bool(row["damage"]["both_positive"]) for row in per_record),
    }
    damage_gate["both_positive_pass"] = damage_gate["both_positive_count"] >= config.MIN_SUCCESS_RECORDS
    damage_gate["passes"] = bool(damage_gate["ci_lower_gt_0_10"] and damage_gate["both_positive_pass"])
    interpolation_improvement_supported = bool(
        gain_bootstrap["c"]["ci95"][0] > 0.10 and gain_bootstrap["d"]["ci95"][0] > 0.10
    )
    timing_ready = bool(
        baseline_count >= config.MIN_BASELINE_RECORDS
        and all(timing_counts[name] >= config.MIN_SUCCESS_RECORDS for name in timing_counts)
    )
    if not timing_ready:
        decision = "LINEAR_TIMING_UNRESOLVED"
    elif not own_gate["passes"]:
        decision = "LINEAR_OWN_AUDIO_UNRESOLVED"
    elif not damage_gate["passes"]:
        decision = "LINEAR_DAMAGE_UNRESOLVED"
    else:
        decision = "LINEAR_ORACLE_CONTROL_SUPPORTED"
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "decision": decision,
        "record_count": len(per_record),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "fresh_score_cell_count": len(fresh_rows),
        "expected_fresh_score_cell_count": config.EXPECTED_FRESH_SCORE_CELL_COUNT,
        "cached_score_cell_count": len(parent["score_rows"]),
        "expected_cached_score_cell_count": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
        "baseline_count": baseline_count,
        "baseline_minimum": config.MIN_BASELINE_RECORDS,
        "timing_counts": timing_counts,
        "timing_minimum": config.MIN_SUCCESS_RECORDS,
        "own_bootstrap": own_bootstrap,
        "own_gate": own_gate,
        "damage_bootstrap": damage_bootstrap,
        "damage_gate": damage_gate,
        "gain_bootstrap": gain_bootstrap,
        "interpolation_improvement_supported": interpolation_improvement_supported,
        "per_record": per_record,
        "parent_oracle_decision": "ORACLE_OWN_AUDIO_UNRESOLVED",
        "parent_own_audio_retested": False,
        "oracle_own_audio_tested": True,
        "interpretation_boundary": "V_LINEAR 是从父 V_ID 像素构造的线性混帧参照，不能证明实际 G_W 生成响应或 replacement effect。",
    }


def result_markdown(analysis: Mapping[str, Any]) -> str:
    own_c = analysis["own_bootstrap"]["c"]
    own_d = analysis["own_bootstrap"]["d"]
    damage_c = analysis["damage_bootstrap"]["c"]
    damage_d = analysis["damage_bootstrap"]["d"]
    gain_c = analysis["gain_bootstrap"]["c"]
    gain_d = analysis["gain_bootstrap"]["d"]
    lines = [
        "# Wav2Lip oracle frame interpolation 2026-09-07",
        "",
        "本轮从父 V_ID 的原始无损像素构造 V_LINEAR；没有使用已取整的 V_ORACLE，也没有重新生成 Wav2Lip 视频。",
        "",
        f"- 科学终态：`{analysis['decision']}`",
        f"- fresh/cached SyncNet cells：{analysis['fresh_score_cell_count']}/{analysis['cached_score_cell_count']}",
        f"- baseline：{analysis['baseline_count']}/{analysis['record_count']}",
        f"- B/C_linear/O_linear：{analysis['timing_counts']['B']}/{analysis['timing_counts']['C_linear']}/{analysis['timing_counts']['O_linear']}",
        f"- own C：mean={own_c['mean']:.6f}, CI=[{own_c['ci95'][0]:.6f}, {own_c['ci95'][1]:.6f}]；own D：mean={own_d['mean']:.6f}, CI=[{own_d['ci95'][0]:.6f}, {own_d['ci95'][1]:.6f}]",
        f"- damage C：mean={damage_c['mean']:.6f}, CI=[{damage_c['ci95'][0]:.6f}, {damage_c['ci95'][1]:.6f}]；damage D：mean={damage_d['mean']:.6f}, CI=[{damage_d['ci95'][0]:.6f}, {damage_d['ci95'][1]:.6f}]",
        f"- gain vs V_ORACLE/W：C CI=[{gain_c['ci95'][0]:.6f}, {gain_c['ci95'][1]:.6f}]；D CI=[{gain_d['ci95'][0]:.6f}, {gain_d['ci95'][1]:.6f}]；supported={analysis['interpolation_improvement_supported']}",
        "",
        "父实验 ORACLE_OWN_AUDIO_UNRESOLVED 与历史 CONTROL_FAILED 保持不变。V_LINEAR 的任何改善都同时包含混帧平滑、纹理变化和时间量化变化，不能解释为单一原因。",
        "",
        "| sample | baseline | B | C_linear | O_linear | own offset | damage positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_record"]:
        lines.append(
            f"| {row['sample_id']} | {row['baseline']} | {row['checks']['B']['passes']} | "
            f"{row['checks']['C_linear']['passes']} | {row['checks']['O_linear']['passes']} | "
            f"{row['own']['offset_agreement']} | {row['damage']['both_positive']} |"
        )
    lines.extend(["", "validator 必须独立重算像素、PCM、embedding 距离、统计和终态；本报告只有在 validator valid 后才算正式完成。", ""])
    return "\n".join(lines)
