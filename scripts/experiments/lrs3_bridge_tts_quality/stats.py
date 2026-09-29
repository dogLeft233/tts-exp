"""Pre-registered paired statistics and terminal-state calculation."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def bootstrap_ci(values: Sequence[float], *, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.BOOTSTRAP_SEED, lower: float = 0.025, upper: float = 0.975) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return {"mean": None, "ci": None, "draws": int(draws), "valid_draws": 0, "seed": int(seed)}
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        samples[index] = float(np.mean(array[rng.integers(0, array.size, size=array.size)]))
    return {
        "mean": float(np.mean(array)),
        "ci": [float(np.quantile(samples, lower)), float(np.quantile(samples, upper))],
        "draws": int(draws),
        "valid_draws": int(samples.size),
        "seed": int(seed),
        "interval": "two_sided_percentile",
        "unit": "source_group",
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def bootstrap_spearman(x: Sequence[float], y: Sequence[float], *, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.ASSOCIATION_BOOTSTRAP_SEED) -> dict[str, Any]:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    point = spearman_rho(left, right)
    if point is None:
        return {"rho": None, "ci95": None, "draws": int(draws), "valid_draws": 0, "seed": int(seed), "reason": "constant or invalid input"}
    rng = np.random.Generator(np.random.PCG64(seed))
    values: list[float] = []
    for _ in range(int(draws)):
        indices = rng.integers(0, left.size, size=left.size)
        value = spearman_rho(left[indices], right[indices])
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < config.MIN_VALID_BOOTSTRAPS:
        interval = None
    else:
        interval = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {"rho": point, "ci95": interval, "draws": int(draws), "valid_draws": len(values), "seed": int(seed), "unit": "record"}


def _metric(score: Mapping[str, Any]) -> tuple[float, float, int]:
    value = score.get("score", score)
    return float(value["sync_c"]), float(value["sync_d"]), int(value["av_offset"])


def _score_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id", "")), int(row.get("render_repeat", -1)), str(row.get("cell", "")))
        if key in result:
            raise ProtocolError(f"duplicate score cell: {key}")
        result[key] = row
    expected = config.EXPECTED_CELL_COUNT
    if len(result) != expected:
        raise ProtocolError(f"score cell count is {len(result)}/{expected}")
    return result


def _cell(index: Mapping[tuple[str, int, str], Mapping[str, Any]], sample_id: str, repeat: int, video: str, audio: str) -> Mapping[str, Any]:
    key = (sample_id, repeat, f"V_{video}/A_{audio}")
    if key not in index:
        raise ProtocolError(f"missing score cell: {key}")
    return index[key]


def _record_values(index: Mapping[tuple[str, int, str], Mapping[str, Any]], sample_id: str) -> dict[str, Any]:
    per_repeat: dict[int, dict[str, tuple[float, float, int]]] = {}
    for repeat in config.REPEATS:
        values: dict[str, tuple[float, float, int]] = {}
        for arm, audio in (("N", "N"), ("B0", "N"), ("B_LOCAL", "N"), ("B_CLOUD", "N"), ("B_LOCAL_OWN", "B_LOCAL"), ("B_CLOUD_OWN", "B_CLOUD"), ("N_REV", "N_REV")):
            video = "B_LOCAL" if arm == "B_LOCAL_OWN" else "B_CLOUD" if arm == "B_CLOUD_OWN" else "N" if arm in ("N", "N_REV") else arm
            values[arm] = _metric(_cell(index, sample_id, repeat, video, audio))
        per_repeat[repeat] = values
    return {"sample_id": sample_id, "repeats": per_repeat}


def _endpoint(values: Sequence[float], *, seed: int = config.BOOTSTRAP_SEED, lower: float = 0.025, upper: float = 0.975) -> dict[str, Any]:
    return bootstrap_ci(values, seed=seed, lower=lower, upper=upper)


def _compatibility(values: Sequence[float], offsets: Sequence[bool]) -> dict[str, Any]:
    result = _endpoint(values)
    result["passes_interval"] = bool(result.get("ci") and result["ci"][0] > config.REPLACEMENT_MARGIN and result["ci"][1] < -config.REPLACEMENT_MARGIN)
    result["offset_agreement_count"] = int(sum(bool(item) for item in offsets))
    result["required_offset_agreement_count"] = config.MIN_OFFSET_AGREEMENT_RECORDS
    result["passes"] = bool(result["passes_interval"] and result["offset_agreement_count"] >= config.MIN_OFFSET_AGREEMENT_RECORDS)
    return result


def _movement(diagnostics: Sequence[Mapping[str, Any]], provider: str) -> dict[str, Any]:
    values: list[float] = []
    degenerate = 0
    for row in diagnostics:
        side = row.get("arms", {}).get(provider, {})
        progress = float(side.get("progress", float("nan")))
        if bool(side.get("degenerate_direction")):
            degenerate += 1
        if math.isfinite(progress):
            values.append(progress)
    endpoint = _endpoint(values)
    moved = int(sum(value >= config.MOVEMENT_THRESHOLD for value in values))
    endpoint.update({
        "provider": provider,
        "records": len(values),
        "progress_threshold": config.MOVEMENT_THRESHOLD,
        "progress_pass_count": moved,
        "required_progress_pass_count": config.MOVEMENT_MIN_RECORDS,
        "degenerate_direction_count": degenerate,
    })
    endpoint["passes"] = bool(len(values) == config.EXPECTED_RECORD_COUNT and moved >= config.MOVEMENT_MIN_RECORDS and endpoint.get("ci") and endpoint["ci"][0] > config.MOVEMENT_THRESHOLD and degenerate == 0)
    return endpoint


def _quality_association(quality: Mapping[str, Any] | None, delta_c: Mapping[str, float]) -> dict[str, Any]:
    if not quality or quality.get("status") in (None, "QUALITY_NOT_ASSESSED"):
        return {"status": "QUALITY_NOT_ASSESSED", "target": {"rho": None}, "raw": {"rho": None}}
    result: dict[str, Any] = {}
    for key, stage in (("raw", "raw"), ("target", "target")):
        stage_payload = quality.get(stage, {})
        paired = stage_payload.get("paired", []) if isinstance(stage_payload, Mapping) else []
        by_id = {str(row.get("sample_id")): float(row["difference"]) for row in paired if isinstance(row, Mapping) and row.get("difference") is not None}
        common = [sample_id for sample_id in delta_c if sample_id in by_id]
        if len(common) != config.EXPECTED_RECORD_COUNT:
            result[key] = {"rho": None, "ci95": None, "status": "NOT_ASSESSED", "record_count": len(common)}
            continue
        result[key] = {**bootstrap_spearman([by_id[sample_id] for sample_id in common], [delta_c[sample_id] for sample_id in common]), "status": "ASSESSED", "record_count": len(common)}
    result["status"] = "ASSESSED" if all(result[key].get("status") == "ASSESSED" for key in ("raw", "target")) else "PARTIAL"
    return result


def analyze_score_rows(
    score_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    quality: Mapping[str, Any] | None = None,
    engineering_complete: bool = True,
) -> dict[str, Any]:
    index = _score_index(score_rows)
    sample_ids = list(config.EXPECTED_SAMPLE_IDS)
    source_group_by_id: dict[str, str] = {}
    for row in score_rows:
        sample_id = str(row.get("sample_id", ""))
        source_group = str(row.get("source_group", ""))
        if sample_id in source_group_by_id and source_group_by_id[sample_id] != source_group:
            raise ProtocolError(f"source group changed within score rows: {sample_id}")
        source_group_by_id[sample_id] = source_group
    if set(source_group_by_id) != set(sample_ids) or len(set(source_group_by_id.values())) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("score rows do not contain the frozen source-group denominator")
    records = [_record_values(index, sample_id) for sample_id in sample_ids]
    paired_rows: list[dict[str, Any]] = []
    endpoints: dict[str, list[float]] = {"delta_c": [], "delta_d": [], "g_c_local": [], "g_c_cloud": [], "g_d_local": [], "g_d_cloud": [], "b0_c": [], "b0_d": [], "damage_c": [], "damage_d": []}
    repeatability: dict[str, Any] = {}
    offset_candidate: dict[str, int] = {"LOCAL": 0, "CLOUD": 0}
    b0_offset = 0
    wrong_audio_both_positive = 0
    for record in records:
        sample_id = record["sample_id"]
        repeat_rows = record["repeats"]
        baseline_values = [repeat_rows[r]["N"] for r in config.REPEATS]
        local_values = [repeat_rows[r]["B_LOCAL"] for r in config.REPEATS]
        cloud_values = [repeat_rows[r]["B_CLOUD"] for r in config.REPEATS]
        b0_values = [repeat_rows[r]["B0"] for r in config.REPEATS]
        nrev_values = [repeat_rows[r]["N_REV"] for r in config.REPEATS]
        g_c_local = float(np.mean([local_values[i][0] - baseline_values[i][0] for i in range(len(config.REPEATS))]))
        g_c_cloud = float(np.mean([cloud_values[i][0] - baseline_values[i][0] for i in range(len(config.REPEATS))]))
        g_d_local = float(np.mean([baseline_values[i][1] - local_values[i][1] for i in range(len(config.REPEATS))]))
        g_d_cloud = float(np.mean([baseline_values[i][1] - cloud_values[i][1] for i in range(len(config.REPEATS))]))
        delta_c = g_c_cloud - g_c_local
        delta_d = g_d_cloud - g_d_local
        b0_c = float(np.mean([b0_values[i][0] - baseline_values[i][0] for i in range(len(config.REPEATS))]))
        b0_d = float(np.mean([baseline_values[i][1] - b0_values[i][1] for i in range(len(config.REPEATS))]))
        damage_c = float(np.mean([baseline_values[i][0] - nrev_values[i][0] for i in range(len(config.REPEATS))]))
        damage_d = float(np.mean([nrev_values[i][1] - baseline_values[i][1] for i in range(len(config.REPEATS))]))
        endpoints["delta_c"].append(delta_c)
        endpoints["delta_d"].append(delta_d)
        endpoints["g_c_local"].append(g_c_local)
        endpoints["g_c_cloud"].append(g_c_cloud)
        endpoints["g_d_local"].append(g_d_local)
        endpoints["g_d_cloud"].append(g_d_cloud)
        endpoints["b0_c"].append(b0_c)
        endpoints["b0_d"].append(b0_d)
        endpoints["damage_c"].append(damage_c)
        endpoints["damage_d"].append(damage_d)
        for provider, candidate in (("LOCAL", local_values), ("CLOUD", cloud_values)):
            if all(abs(candidate[i][2] - baseline_values[i][2]) <= config.OFFSET_TOLERANCE_FRAMES for i in range(len(config.REPEATS))):
                offset_candidate[provider] += 1
        if all(abs(b0_values[i][2] - baseline_values[i][2]) <= config.OFFSET_TOLERANCE_FRAMES for i in range(len(config.REPEATS))):
            b0_offset += 1
        if damage_c > config.WRONG_AUDIO_DAMAGE_THRESHOLD and damage_d > config.WRONG_AUDIO_DAMAGE_THRESHOLD:
            wrong_audio_both_positive += 1
        paired_rows.append({
            "sample_id": sample_id,
            "source_group": source_group_by_id[sample_id],
            "gC_LOCAL": g_c_local,
            "gC_CLOUD": g_c_cloud,
            "deltaC": delta_c,
            "gD_LOCAL": g_d_local,
            "gD_CLOUD": g_d_cloud,
            "deltaD": delta_d,
            "B0_C": b0_c,
            "B0_D": b0_d,
            "damageC": damage_c,
            "damageD": damage_d,
            "offset_candidate_local": all(abs(local_values[i][2] - baseline_values[i][2]) <= config.OFFSET_TOLERANCE_FRAMES for i in range(len(config.REPEATS))),
            "offset_candidate_cloud": all(abs(cloud_values[i][2] - baseline_values[i][2]) <= config.OFFSET_TOLERANCE_FRAMES for i in range(len(config.REPEATS))),
        })
    delta_c_endpoint = _endpoint(endpoints["delta_c"])
    delta_d_endpoint = _endpoint(endpoints["delta_d"])
    absolute = {}
    for provider in ("LOCAL", "CLOUD"):
        prefix = provider.lower()
        c = _endpoint(endpoints[f"g_c_{prefix}"], lower=0.0125, upper=0.9875)
        d = _endpoint(endpoints[f"g_d_{prefix}"])
        absolute[provider] = {
            "gC": c,
            "gD": d,
            "offset_compatible_records": offset_candidate[provider],
            "replacement_gain_observed": bool(c.get("ci") and c["ci"][0] > 0.0 and d.get("ci") and d["ci"][0] > config.REPLACEMENT_MARGIN and offset_candidate[provider] >= config.MIN_OFFSET_AGREEMENT_RECORDS),
        }
    repeatability = {}
    for arm in config.ARMS:
        c_diff: list[float] = []
        d_diff: list[float] = []
        offset_count = 0
        for record in records:
            first = record["repeats"][config.REPEATS[0]][arm]
            second = record["repeats"][config.REPEATS[1]][arm]
            c_diff.append(second[0] - first[0])
            d_diff.append(second[1] - first[1])
            offset_count += int(abs(second[2] - first[2]) <= config.OFFSET_TOLERANCE_FRAMES)
        c_ep = _endpoint(c_diff)
        d_ep = _endpoint(d_diff)
        arm_pass = bool(c_ep.get("ci") and d_ep.get("ci") and c_ep["ci"][0] > config.REPLACEMENT_MARGIN and c_ep["ci"][1] < -config.REPLACEMENT_MARGIN and d_ep["ci"][0] > config.REPLACEMENT_MARGIN and d_ep["ci"][1] < -config.REPLACEMENT_MARGIN and offset_count >= config.MIN_OFFSET_AGREEMENT_RECORDS)
        repeatability[arm] = {"C_difference": c_ep, "D_difference": d_ep, "offset_agreement_records": offset_count, "passes": arm_pass}
    b0 = {"C": _endpoint(endpoints["b0_c"]), "D": _endpoint(endpoints["b0_d"]), "offset_agreement_records": b0_offset}
    b0["passes"] = bool(b0["C"].get("ci") and b0["D"].get("ci") and b0["C"]["ci"][0] > config.REPLACEMENT_MARGIN and b0["C"]["ci"][1] < -config.REPLACEMENT_MARGIN and b0["D"]["ci"][0] > config.REPLACEMENT_MARGIN and b0["D"]["ci"][1] < -config.REPLACEMENT_MARGIN and b0_offset >= config.MIN_OFFSET_AGREEMENT_RECORDS)
    wrong_audio = {"damageC": _endpoint(endpoints["damage_c"]), "damageD": _endpoint(endpoints["damage_d"]), "both_damage_above_threshold_records": wrong_audio_both_positive, "threshold": config.WRONG_AUDIO_DAMAGE_THRESHOLD}
    wrong_audio["passes"] = bool(wrong_audio["damageC"].get("ci") and wrong_audio["damageD"].get("ci") and wrong_audio["damageC"]["ci"][0] > config.WRONG_AUDIO_DAMAGE_THRESHOLD and wrong_audio["damageD"]["ci"][0] > config.WRONG_AUDIO_DAMAGE_THRESHOLD and wrong_audio_both_positive >= config.MIN_WRONG_AUDIO_DAMAGE_RECORDS)
    controls_pass = bool(all(row["passes"] for row in repeatability.values()) and b0["passes"] and wrong_audio["passes"])
    diagnostic_by_id = {str(row["sample_id"]): row for row in diagnostics}
    movement = {provider: _movement([diagnostic_by_id[sample_id] for sample_id in sample_ids if sample_id in diagnostic_by_id], provider) for provider in ("LOCAL", "CLOUD")}
    movement_pass = bool(movement["LOCAL"]["passes"] and movement["CLOUD"]["passes"])
    source_comparison = "SOURCE_DIFFERENCE_UNRESOLVED"
    if delta_c_endpoint.get("ci") and delta_c_endpoint["ci"][0] > 0.0:
        source_comparison = "CLOUD_BRIDGE_STRONGER"
    elif delta_c_endpoint.get("ci") and delta_c_endpoint["ci"][1] < 0.0:
        source_comparison = "LOCAL_BRIDGE_STRONGER"
    quality_association = _quality_association(quality, {row["sample_id"]: row["deltaC"] for row in paired_rows})
    if not engineering_complete:
        terminal = "ENGINEERING_BLOCKED"
    elif not controls_pass:
        terminal = "CONTROL_FAILED"
    elif not movement_pass:
        terminal = "BRIDGE_MOVEMENT_FAILED"
    else:
        terminal = source_comparison
    statistics = {
        "schema_version": 1,
        "stage_id": "07_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "record/source_group after repeat mean", "interval": "two_sided_percentile"},
        "primary": {"deltaC": delta_c_endpoint, "deltaD": delta_d_endpoint, "formula_check": "deltaC=mean_r(C_CLOUD-C_LOCAL); gC_CLOUD-gC_LOCAL agrees per record"},
        "absolute": absolute,
        "repeatability": repeatability,
        "B0": b0,
        "wrong_audio": wrong_audio,
        "controls_pass": controls_pass,
        "movement": movement,
        "movement_pass": movement_pass,
        "quality_association": quality_association,
        "per_record": paired_rows,
        "terminal": terminal,
    }
    return statistics


def write_analysis_stage(
    run_root: Path,
    *,
    quality: Mapping[str, Any] | None = None,
    engineering_complete: bool = True,
    output_dir: Path | None = None,
    analysis_version: int = 1,
    parent_analysis_sha256: str | None = None,
) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    destination = output_dir or paths.analysis
    destination.mkdir(parents=True, exist_ok=True)
    score_manifest = verify_self_hashed_json(paths.scores / "matrix_manifest.json")
    diagnostics = verify_self_hashed_json(paths.bridge / "diagnostics.json")
    score_rows = score_manifest.get("rows", [])
    if not isinstance(score_rows, list):
        raise ProtocolError("score manifest has no rows")
    statistics = analyze_score_rows(score_rows, diagnostics.get("rows", []), quality=quality, engineering_complete=engineering_complete and score_manifest.get("status") == "complete")
    with (destination / "paired.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(statistics["per_record"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(statistics["per_record"])
    statistics["analysis_version"] = int(analysis_version)
    statistics["parent_analysis_sha256"] = parent_analysis_sha256
    write_self_hashed_json(destination / "controls.json", {"schema_version": 1, "stage_id": "07_analysis", "protocol_id": config.PROTOCOL_ID, "analysis_version": int(analysis_version), "parent_analysis_sha256": parent_analysis_sha256, "controls": {key: statistics[key] for key in ("repeatability", "B0", "wrong_audio")}, "controls_pass": statistics["controls_pass"]})
    write_self_hashed_json(destination / "statistics.json", statistics)
    final = {
        "schema_version": 1,
        "stage_id": "07_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "analysis_version": int(analysis_version),
        "parent_analysis_sha256": parent_analysis_sha256,
        "engineering_status": "COMPLETE" if engineering_complete and score_manifest.get("status") == "complete" else "ENGINEERING_BLOCKED",
        "controls_status": "PASSED" if statistics["controls_pass"] else "CONTROL_FAILED",
        "movement_status": {provider: "PASSED" if statistics["movement"][provider]["passes"] else "FAILED" for provider in ("LOCAL", "CLOUD")},
        "source_comparison": statistics["terminal"] if statistics["terminal"] in ("CLOUD_BRIDGE_STRONGER", "LOCAL_BRIDGE_STRONGER", "SOURCE_DIFFERENCE_UNRESOLVED") else "SOURCE_DIFFERENCE_UNRESOLVED",
        "terminal_state": statistics["terminal"],
        "replacement_gain_observed": {provider: statistics["absolute"][provider]["replacement_gain_observed"] for provider in ("LOCAL", "CLOUD")},
        "quality_status": (quality or {}).get("status", "QUALITY_NOT_ASSESSED"),
        "quality_association": statistics["quality_association"],
        "scope": "fixed 22-record/22-source-group LRS3 fit-only paired pilot; historical samples were previously observed",
        "causal_quality_effect_established": False,
        "generalization_confirmed": False,
        "statistics_sha256": file_sha256(destination / "statistics.json"),
        "matrix_manifest_sha256": file_sha256(paths.scores / "matrix_manifest.json"),
        "quality_input_sha256": (file_sha256(paths.quality / "quality.json") if (paths.quality / "quality.json").is_file() else None),
    }
    write_self_hashed_json(destination / "final.json", final)
    report = render_report(statistics, final)
    (destination / "report.md").write_text(report, encoding="utf-8")
    write_self_hashed_json(destination / "report_manifest.json", {"schema_version": 1, "stage_id": "07_analysis", "protocol_id": config.PROTOCOL_ID, "analysis_version": int(analysis_version), "parent_analysis_sha256": parent_analysis_sha256, "report_sha256": file_sha256(destination / "report.md"), "final_sha256": file_sha256(destination / "final.json")})
    return final


def render_report(statistics: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    primary = statistics["primary"]["deltaC"]
    lines = [
        "# LRS3 bridge TTS 来源比较",
        "",
        "本报告把驱动音轨和评分音轨分开：驱动音轨用于生成口型视频，评分音轨是在同一冻结视频上送入 SyncNet 的音频。主比较先在每条记录内平均两次独立渲染，再比较云端和本地的 replacement Sync-C 差值。",
        "",
        f"终态：`{final['terminal_state']}`。范围是固定22条LRS3记录、22个source group的fit-only paired pilot；这些样本此前已见。因果音质效应和泛化确认均为 `false`。",
        "",
        f"主端点 deltaC = 云端相对自然的增益 − 本地相对自然的增益，均值 `{primary.get('mean')}`，95% CI `{primary.get('ci')}`。",
        "",
        "## 控制与中间结果",
        "",
        f"控制通过：`{statistics['controls_pass']}`；bridge movement 通过：LOCAL=`{statistics['movement']['LOCAL']['passes']}`，CLOUD=`{statistics['movement']['CLOUD']['passes']}`。",
        f"本地绝对 replacement 增益成立：`{statistics['absolute']['LOCAL']['replacement_gain_observed']}`；云端：`{statistics['absolute']['CLOUD']['replacement_gain_observed']}`。",
        f"质量状态：`{final['quality_status']}`。质量听评副本只用于人工评分，未进入模型或 SyncNet。",
        "",
        "完整的逐条 C/D/offset、重复性、B0、N_REV、听评关联和失败台账保存在同目录的 JSON/CSV 中。",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["analyze_score_rows", "bootstrap_ci", "bootstrap_spearman", "spearman_rho", "write_analysis_stage"]
