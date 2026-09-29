from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import DiagnosticError, file_sha256, read_json, verify_self_hashed_json, write_self_hashed_json


def _score_index(score_manifest: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in score_manifest.get("scores", []):
        key = (str(row.get("sample_id")), str(row.get("family")), str(row.get("cell")))
        if key in result:
            raise DiagnosticError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    if row.get("status") != "complete":
        raise DiagnosticError(f"score cell is not complete: {row.get('cell_id')}")
    path = Path(str(row["matrix"]))
    if file_sha256(path) != str(row["matrix_sha256"]):
        raise DiagnosticError(f"score matrix hash changed: {path}")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise DiagnosticError(f"score matrix is malformed: {path}")
    return np.asarray(value, dtype=np.float64)


def _d0(row: Mapping[str, Any], selected_rows: list[int] | None = None) -> float:
    matrix = _matrix(row)
    rows = selected_rows if selected_rows is not None else list(range(matrix.shape[0]))
    if not rows:
        raise DiagnosticError(f"local support is empty: {row.get('cell_id')}")
    return float(np.mean(matrix[rows, config.VSHIFT]))


def _segment_rows(matrix_rows: int, k: int, segment: int) -> list[int]:
    start = segment * k
    end = (segment + 1) * k
    low = max(config.VSHIFT, start + config.LOCAL_MARGIN_FRAMES)
    high = min(matrix_rows - config.VSHIFT - 1, end - config.WINDOW_FRAMES - config.LOCAL_MARGIN_FRAMES)
    return list(range(low, high + 1)) if high >= low else []


def _all_local_rows(matrix_rows: int) -> list[int]:
    low = config.VSHIFT + config.LOCAL_MARGIN_FRAMES
    high = matrix_rows - config.VSHIFT - 1 - config.WINDOW_FRAMES - config.LOCAL_MARGIN_FRAMES
    return list(range(low, high + 1)) if high >= low else []


def _metric(row: Mapping[str, Any], name: str) -> float | int | None:
    value = row.get("reconstructed", {}).get(name)
    return float(value) if name != "offset" and value is not None else (int(value) if value is not None else None)


def _history_metric(audit_row: Mapping[str, Any], cell: str, name: str) -> float | int | None:
    values = audit_row.get("checks", {}).get(cell, {}).get("log_values", [])
    if not values:
        return None
    value = values[0].get(name)
    return float(value) if name != "av_offset" and value is not None else (int(value) if value is not None else None)


def _local_pair(index: Mapping[tuple[str, str, str], Mapping[str, Any]], sample_id: str, left: tuple[str, str, str], right: tuple[str, str, str], support_rows: list[int]) -> dict[str, Any]:
    left_row = index[left]
    right_row = index[right]
    if len(support_rows) < config.MIN_LOCAL_WINDOWS:
        return {"status": "INSUFFICIENT_SUPPORT", "rows": support_rows, "count": len(support_rows)}
    left_d0 = _d0(left_row, support_rows)
    right_d0 = _d0(right_row, support_rows)
    left_curve = [float(value) for value in np.mean(_matrix(left_row)[support_rows, :], axis=0)]
    right_curve = [float(value) for value in np.mean(_matrix(right_row)[support_rows, :], axis=0)]
    return {
        "status": "complete",
        "rows": support_rows,
        "count": len(support_rows),
        "left_cell": left_row["cell"],
        "right_cell": right_row["cell"],
        "left_d0": left_d0,
        "right_d0": right_d0,
        "preference_left_better": float(right_d0 - left_d0),
        "offsets": [config.VSHIFT - column for column in range(2 * config.VSHIFT + 1)],
        "left_curve": left_curve,
        "right_curve": right_curve,
    }


def analyze(inputs: Mapping[str, Any], audit: Mapping[str, Any], media_manifest: Mapping[str, Any], score_manifest: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    index = _score_index(score_manifest)
    audit_index = {str(row["sample_id"]): row for row in audit.get("records", [])}
    media_index = {(str(row.get("sample_id")), str(row.get("family")), str(row.get("cell"))): row for row in media_manifest.get("cells", [])}
    per_sample: list[dict[str, Any]] = []
    for sample_id in config.SAMPLE_IDS:
        audit_row = audit_index[sample_id]
        a_n = index.get((sample_id, "A", "G_N"), {"status": "blocked"})
        a_s = index.get((sample_id, "A", "G_S"), {"status": "blocked"})
        a: dict[str, Any] = {"status": "complete" if a_n.get("status") == "complete" and a_s.get("status") == "complete" else "blocked"}
        if a["status"] == "complete":
            a_c_delta = float(_metric(a_n, "sync_c")) - float(_metric(a_s, "sync_c"))
            a_d_delta = float(_metric(a_s, "sync_d")) - float(_metric(a_n, "sync_d"))
            a_off_delta = abs(int(_metric(a_n, "offset")) - int(_metric(a_s, "offset")))
            old_n_c = _history_metric(audit_row, "V_LOCAL_SWAP/A_N", "sync_c")
            old_s_c = _history_metric(audit_row, "V_LOCAL_SWAP/A_LOCAL_SWAP", "sync_c")
            old_n_d = _history_metric(audit_row, "V_LOCAL_SWAP/A_N", "sync_d")
            old_s_d = _history_metric(audit_row, "V_LOCAL_SWAP/A_LOCAL_SWAP", "sync_d")
            old_n_o = _history_metric(audit_row, "V_LOCAL_SWAP/A_N", "av_offset")
            old_s_o = _history_metric(audit_row, "V_LOCAL_SWAP/A_LOCAL_SWAP", "av_offset")
            a = {
                "status": "complete",
                "new_G_N": {"sync_c": _metric(a_n, "sync_c"), "sync_d": _metric(a_n, "sync_d"), "offset": _metric(a_n, "offset")},
                "new_G_S": {"sync_c": _metric(a_s, "sync_c"), "sync_d": _metric(a_s, "sync_d"), "offset": _metric(a_s, "offset")},
                "new_natural_preference": bool(a_c_delta > config.FLOAT_TIE_TOLERANCE and a_d_delta > config.FLOAT_TIE_TOLERANCE),
                "new_c_delta_G_N_minus_G_S": a_c_delta,
                "new_d_delta_G_S_minus_G_N": a_d_delta,
                "new_offset_delta_abs": a_off_delta,
                "historical": {"G_N": {"sync_c": old_n_c, "sync_d": old_n_d, "offset": old_n_o}, "G_S": {"sync_c": old_s_c, "sync_d": old_s_d, "offset": old_s_o}},
                "numeric_agreement": bool(
                    old_n_c is not None and old_s_c is not None and old_n_d is not None and old_s_d is not None
                    and abs(float(_metric(a_n, "sync_c")) - float(old_n_c)) <= config.NUMERIC_REPLAY_TOLERANCE
                    and abs(float(_metric(a_s, "sync_c")) - float(old_s_c)) <= config.NUMERIC_REPLAY_TOLERANCE
                    and abs(float(_metric(a_n, "sync_d")) - float(old_n_d)) <= config.NUMERIC_REPLAY_TOLERANCE
                    and abs(float(_metric(a_s, "sync_d")) - float(old_s_d)) <= config.NUMERIC_REPLAY_TOLERANCE
                    and (old_n_o is None or abs(int(_metric(a_n, "offset")) - int(old_n_o)) <= config.OFFSET_REPLAY_TOLERANCE)
                    and (old_s_o is None or abs(int(_metric(a_s, "offset")) - int(old_s_o)) <= config.OFFSET_REPLAY_TOLERANCE)
                ),
            }
        b: dict[str, Any] = {"status": "blocked"}
        b_rows = [index.get((sample_id, "B", cell), {"status": "blocked"}) for cell in ("R_N0", "R_S0", "Rs_N0", "Rs_S0")]
        if all(row.get("status") == "complete" for row in b_rows):
            k = int(support["samples"][sample_id]["B"]["k"])
            r_n, r_s, rs_n, rs_s = b_rows
            row_count = _matrix(r_n).shape[0]
            segments: dict[str, Any] = {}
            for segment in (1, 2):
                rows = _segment_rows(row_count, k, segment)
                # Put the expected-matched condition on the left.  The helper
                # reports D(right) - D(left), which is exactly the positive
                # preference specified by the protocol: wrong minus matched.
                original = _local_pair(index, sample_id, (sample_id, "B", "R_N0"), (sample_id, "B", "R_S0"), rows)
                swapped = _local_pair(index, sample_id, (sample_id, "B", "Rs_S0"), (sample_id, "B", "Rs_N0"), rows)
                segments[str(segment)] = {"rows": rows, "original_preference": original, "swapped_preference": swapped}
            original_rows = sorted({row for item in segments.values() for row in item["original_preference"].get("rows", [])})
            swapped_rows = sorted({row for item in segments.values() for row in item["swapped_preference"].get("rows", [])})
            original_pref = float(_d0(r_s, original_rows) - _d0(r_n, original_rows)) if len(original_rows) >= config.MIN_LOCAL_WINDOWS else None
            swapped_pref = float(_d0(rs_n, swapped_rows) - _d0(rs_s, swapped_rows)) if len(swapped_rows) >= config.MIN_LOCAL_WINDOWS else None
            expected_offsets = {"R_N0": int(_metric(r_n, "offset")), "Rs_S0": int(_metric(rs_s, "offset"))}
            baseline_alignment_suspect = any(abs(value) > config.OFFSET_REPLAY_TOLERANCE for value in expected_offsets.values())
            segment_direction_ok = all(
                item["original_preference"].get("status") == "complete"
                and item["swapped_preference"].get("status") == "complete"
                and float(item["original_preference"]["preference_left_better"]) > config.FLOAT_TIE_TOLERANCE
                and float(item["swapped_preference"]["preference_left_better"]) > config.FLOAT_TIE_TOLERANCE
                for item in segments.values()
            )
            b = {
                "status": "complete",
                "k": k,
                "global": {"R_N0": {"sync_c": _metric(r_n, "sync_c"), "sync_d": _metric(r_n, "sync_d"), "offset": _metric(r_n, "offset")}, "R_S0": {"sync_c": _metric(r_s, "sync_c"), "sync_d": _metric(r_s, "sync_d"), "offset": _metric(r_s, "offset")}, "Rs_N0": {"sync_c": _metric(rs_n, "sync_c"), "sync_d": _metric(rs_n, "sync_d"), "offset": _metric(rs_n, "offset")}, "Rs_S0": {"sync_c": _metric(rs_s, "sync_c"), "sync_d": _metric(rs_s, "sync_d"), "offset": _metric(rs_s, "offset")}},
                "real_original_preference": original_pref,
                "real_swapped_preference": swapped_pref,
                "segments": segments,
                "baseline_offsets": expected_offsets,
                "baseline_alignment_suspect": baseline_alignment_suspect,
                "known_pairing_sensitive": bool(segment_direction_ok and not baseline_alignment_suspect),
            }
        c: dict[str, Any] = {"status": "blocked"}
        c_n = index.get((sample_id, "C", "Gc_Nc"), {"status": "blocked"})
        c_s = index.get((sample_id, "C", "Gc_Sc"), {"status": "blocked"})
        if c_n.get("status") == "complete" and c_s.get("status") == "complete":
            matrix_rows = _matrix(c_n).shape[0]
            rows = _all_local_rows(matrix_rows)
            if len(rows) >= config.MIN_LOCAL_WINDOWS:
                preference = float(_d0(c_s, rows) - _d0(c_n, rows))
                c = {"status": "complete", "rows": rows, "count": len(rows), "offsets": [config.VSHIFT - column for column in range(2 * config.VSHIFT + 1)], "natural_curve": [float(value) for value in np.mean(_matrix(c_n)[rows, :], axis=0)], "local_swap_curve": [float(value) for value in np.mean(_matrix(c_s)[rows, :], axis=0)], "generated_natural_preference": preference, "natural_better": preference > config.FLOAT_TIE_TOLERANCE, "global": {"Gc_Nc": {"sync_c": _metric(c_n, "sync_c"), "sync_d": _metric(c_n, "sync_d"), "offset": _metric(c_n, "offset")}, "Gc_Sc": {"sync_c": _metric(c_s, "sync_c"), "sync_d": _metric(c_s, "sync_d"), "offset": _metric(c_s, "offset")}}}
            else:
                c = {"status": "INSUFFICIENT_SUPPORT", "rows": rows, "count": len(rows)}
        per_sample.append({"sample_id": sample_id, "source_group": audit_row.get("source_group"), "historical_record_verified": bool(audit_row.get("passed")), "A": a, "B": b, "C": c})
    replay_direction_count = sum(bool(row["A"].get("new_natural_preference")) for row in per_sample if row["A"].get("status") == "complete")
    replay_numeric_count = sum(bool(row["A"].get("numeric_agreement")) for row in per_sample if row["A"].get("status") == "complete")
    known_count = sum(bool(row["B"].get("known_pairing_sensitive")) for row in per_sample if row["B"].get("status") == "complete")
    c_preferences = [float(row["C"]["generated_natural_preference"]) for row in per_sample if row["C"].get("status") == "complete" and row["C"].get("generated_natural_preference") is not None]
    generated_preference = "INSUFFICIENT_SUPPORT"
    if len(c_preferences) == len(config.SAMPLE_IDS):
        if all(value > config.FLOAT_TIE_TOLERANCE for value in c_preferences):
            generated_preference = "NATURAL"
        elif all(value < -config.FLOAT_TIE_TOLERANCE for value in c_preferences):
            generated_preference = "LOCAL_SWAP"
        else:
            generated_preference = "MIXED_OR_UNRESOLVED"
    known_status = "KNOWN_PAIRING_SENSITIVE" if known_count == len(config.SAMPLE_IDS) else "MIXED_OR_UNRESOLVED"
    if not bool(audit.get("claims", {}).get("history_record_verified")):
        localization = "HISTORICAL_INPUT_INCONSISTENT"
    elif known_status == "KNOWN_PAIRING_SENSITIVE" and generated_preference == "NATURAL":
        localization = "SUPPORTS_INSUFFICIENT_DRIVER_TRANSFER"
    elif any(row["B"].get("baseline_alignment_suspect") for row in per_sample if row["B"].get("status") == "complete"):
        localization = "BASELINE_ALIGNMENT_SUSPECT"
    elif known_status != "KNOWN_PAIRING_SENSITIVE":
        localization = "SYNCNET_ENDPOINT_SENSITIVITY_SUSPECT_OR_MIXED"
    else:
        localization = "INCONCLUSIVE"
    analysis = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "scope": "three fixed fit-only LRS3 samples; 24 new cells; no Wav2Lip/TTS/training/bridge rerun",
        "historical_record_verified": bool(audit.get("claims", {}).get("history_record_verified")),
        "replay_direction": {"count": replay_direction_count, "total": len(config.SAMPLE_IDS), "status": "REPRODUCED_3_OF_3" if replay_direction_count == len(config.SAMPLE_IDS) else "MIXED_OR_UNRESOLVED"},
        "replay_numeric_agreement": {"count": replay_numeric_count, "total": len(config.SAMPLE_IDS), "tolerance": config.NUMERIC_REPLAY_TOLERANCE, "status": "AGREE_3_OF_3" if replay_numeric_count == len(config.SAMPLE_IDS) else "MIXED_OR_UNRESOLVED"},
        "known_pairing_sensitivity": {"count": known_count, "total": len(config.SAMPLE_IDS), "status": known_status},
        "generated_audio_preference": {"status": generated_preference, "per_sample_preference": c_preferences},
        "localization": localization,
        "per_sample": per_sample,
        "limits": ["three samples diagnose endpoint and pairing; they do not establish replacement benefit", "B/C global and local evidence are reported separately", "human review is not automated"],
    }
    return analysis


def write_scores_csv(paths: config.RunPaths, media_manifest: Mapping[str, Any], score_manifest: Mapping[str, Any]) -> None:
    score_index = _score_index(score_manifest)
    fieldnames = ["sample_id", "source_group", "family", "cell", "status", "video_kind", "audio_kind", "score_mode", "sync_c", "sync_d", "av_offset", "matrix", "media", "media_sha256", "error_type", "error"]
    with paths.scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for media_row in media_manifest.get("cells", []):
            key = (str(media_row.get("sample_id")), str(media_row.get("family")), str(media_row.get("cell")))
            score = score_index.get(key, {"status": "blocked", "error": "score_row_missing"})
            reconstructed = score.get("reconstructed", {})
            writer.writerow({
                "sample_id": key[0], "source_group": media_row.get("source_group", ""), "family": key[1], "cell": key[2], "status": score.get("status", "blocked"), "video_kind": media_row.get("video_kind", ""), "audio_kind": media_row.get("audio_kind", ""), "score_mode": media_row.get("score_mode", ""), "sync_c": reconstructed.get("sync_c", ""), "sync_d": reconstructed.get("sync_d", ""), "av_offset": reconstructed.get("offset", ""), "matrix": score.get("matrix", ""), "media": media_row.get("media", ""), "media_sha256": media_row.get("media_sha256", ""), "error_type": score.get("error_type", ""), "error": score.get("error", ""),
            })


def update_support(paths: config.RunPaths, analysis: Mapping[str, Any]) -> dict[str, Any]:
    support = verify_self_hashed_json(paths.support)
    support["local_evidence"] = {str(row["sample_id"]): {"B": row["B"], "C": row["C"]} for row in analysis.get("per_sample", [])}
    write_self_hashed_json(paths.support, support)
    return support


def report_markdown(analysis: Mapping[str, Any], audit: Mapping[str, Any], review_status: str) -> str:
    lines = [
        "# LOCAL_SWAP 最小重放诊断",
        "",
        "本实验固定历史 cohort 的前三条样本，复核全部 22 条历史 LOCAL_SWAP 记录，并对三条样本执行 A/B/C 共 24 个新评分 cell。没有重新生成 Wav2Lip 视频、TTS 或 bridge 音频。",
        "",
        "## 结论摘要",
        "",
        f"- 历史记录可信度：`{analysis['historical_record_verified']}`；审计 `{audit.get('passed_count', 0)}/{audit.get('expected_record_count', 0)}` 条通过。",
        f"- A 原生成视频重放方向：`{analysis['replay_direction']['status']}`（{analysis['replay_direction']['count']}/{analysis['replay_direction']['total']}）。",
        f"- A 数值近似重放：`{analysis['replay_numeric_agreement']['status']}`，公差 C/D `±{config.NUMERIC_REPLAY_TOLERANCE:.3f}`、offset `±{config.OFFSET_REPLAY_TOLERANCE}` 帧。",
        f"- B 已知真实配对敏感性：`{analysis['known_pairing_sensitivity']['status']}`（{analysis['known_pairing_sensitivity']['count']}/{analysis['known_pairing_sensitivity']['total']}）。",
        f"- C 固定生成视频更偏向：`{analysis['generated_audio_preference']['status']}`。",
        f"- 原因定位字段：`{analysis['localization']}`；人工观看：`{review_status}`。",
        "",
        "## 实验流程",
        "",
        "A 使用历史 LOCAL_SWAP 驱动生成视频 G，分别用完整自然音频 N 与完整 LOCAL_SWAP 音频 S strict-mux 后重跑官方 SyncNet pipeline。B 对真实视频先固定一次官方 S3FD 轨迹，构造 R/N0、R/S0、Rs/N0、Rs/S0；R 与 N0 是原始同步排列，Rs 与 S0 同时采用 A-C-B-D 排列。C 使用 A 的 G/N pipeline 实际选中的首条 crop，固定同一组 crop 帧和绝对时间支持，只配旧 N/S 的准确切片。每个 cell 都保存距离矩阵、offset 轴和输入 hash。",
        "",
        "## 逐样本结果",
        "",
        "| sample_id | A natural preference | A numeric | B pairing | C natural preference | B baseline |",
        "|---|:---:|:---:|:---:|---:|:---:|",
    ]
    for row in analysis["per_sample"]:
        c_value = row["C"].get("generated_natural_preference")
        b_original = row["B"].get("real_original_preference")
        b_swapped = row["B"].get("real_swapped_preference")
        lines.append(f"| {row['sample_id']} | {row['A'].get('new_natural_preference', 'blocked')} | {row['A'].get('numeric_agreement', 'blocked')} | {row['B'].get('known_pairing_sensitive', 'blocked')} | {c_value if c_value is not None else 'blocked'} | {row['B'].get('baseline_alignment_suspect', 'blocked')} |")
        lines.append(f"| ↳ B local D0 preference | original wrong−matched: `{b_original if b_original is not None else 'blocked'}` | swapped wrong−matched: `{b_swapped if b_swapped is not None else 'blocked'}` | | | |")
    def _score_line(sample_id: str, label: str, metrics: Mapping[str, Any]) -> str:
        c_value = metrics.get("sync_c")
        d_value = metrics.get("sync_d")
        offset = metrics.get("offset")
        c_text = f"{float(c_value):.3f}" if c_value is not None else "blocked"
        d_text = f"{float(d_value):.3f}" if d_value is not None else "blocked"
        offset_text = str(int(offset)) if offset is not None else "blocked"
        return f"| {sample_id} | `{label}` | {c_text} | {d_text} | {offset_text} |"
    lines.extend([
        "",
        "## 全局分数（各端点内部比较）",
        "",
        "| sample_id | cell | Sync-C | Sync-D | best offset |",
        "|---|---|---:|---:|---:|",
    ])
    for row in analysis["per_sample"]:
        sample_id = str(row["sample_id"])
        a = row["A"]
        b = row["B"]
        c = row["C"]
        lines.append(_score_line(sample_id, "A/G_N", a.get("new_G_N", {})))
        lines.append(_score_line(sample_id, "A/G_S", a.get("new_G_S", {})))
        for label in ("R_N0", "R_S0", "Rs_N0", "Rs_S0"):
            lines.append(_score_line(sample_id, f"B/{label}", b.get("global", {}).get(label, {})))
        lines.append(_score_line(sample_id, "C/Gc_Nc", c.get("global", {}).get("Gc_Nc", {})))
        lines.append(_score_line(sample_id, "C/Gc_Sc", c.get("global", {}).get("Gc_Sc", {})))
    lines.extend([
        "",
        "## B 的中间段局部 D0",
        "",
        "偏好定义为 `D0(wrong) − D0(matched)`，正值表示 matched 更好；每行两种条件使用同一组有效窗口，具体 offset 曲线保存在 `analysis.json` / `support.json`。",
        "",
        "| sample_id | segment | valid windows | original wrong−matched | swapped wrong−matched |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in analysis["per_sample"]:
        for segment, evidence in row["B"].get("segments", {}).items():
            original = evidence.get("original_preference", {})
            swapped = evidence.get("swapped_preference", {})
            original_value = original.get("preference_left_better")
            swapped_value = swapped.get("preference_left_better")
            original_text = f"{float(original_value):.6f}" if original_value is not None else "blocked"
            swapped_text = f"{float(swapped_value):.6f}" if swapped_value is not None else "blocked"
            lines.append(f"| {row['sample_id']} | {segment} | {original.get('count', 0)} | {original_text} | {swapped_text} |")
    lines.extend([
        "",
        "## C 的固定 crop 音频偏好",
        "",
        "| sample_id | valid windows | `D0(Sc) − D0(Nc)` |",
        "|---|---:|---:|",
    ])
    for row in analysis["per_sample"]:
        c = row["C"]
        value = c.get("generated_natural_preference")
        value_text = f"{float(value):.6f}" if value is not None else "blocked"
        lines.append(f"| {row['sample_id']} | {c.get('count', 0)} | {value_text} |")
    lines.extend([
        "",
        "## 如何解释",
        "",
        "若 B 能在两段中区分已知同步/错配，而 C 仍显示自然音频偏好，证据支持“生成视频中的驱动时序没有充分转移到 TTS/LOCAL_SWAP 音频”的解释；它不能单独证明 mouth leakage、full-frame box 或任何单一生成器缺陷。若 B 失败，则当前 SyncNet 端点对该扰动/样本的敏感性需要保留怀疑，不能外推为所有 SyncNet 结果无效。",
        "",
        "## 局限与人工复核",
        "",
        "三条样本只用于定位，不用于总体显著性检验；A 与 C 的绝对分数来自不同端点，不能混合平均。播放页已生成，但本轮未进行人工观看，因此不能把自动分数写成人工观察。",
        "B/C 局部曲线均保存 offset=−15…+15 的同一支持窗口聚合值；B 的 `D0` 偏好只表示该固定 offset=0 支持上的距离差，不冒充官方全局 Sync-D。跨段感受野和边界各剔除 5 帧。",
        f"播放页：`playback/index.html`；人工复核状态：`{review_status}`。",
        "",
    ])
    return "\n".join(lines)
