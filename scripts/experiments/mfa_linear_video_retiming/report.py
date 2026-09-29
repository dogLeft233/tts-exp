from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import ProtocolError, file_sha256, verify_json, write_json


def _official_results(run_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in manifest.get("rows", []):
        if row.get("status") != "PASS":
            output.append(dict(row))
            continue
        result_path = Path(str(row.get("result_path", ""))).resolve()
        if not result_path.is_file():
            result_path = run_dir / "06_official" / "cells" / str(row["cell_key"]) / "result.json"
        result = verify_json(result_path, self_hash=True)
        if result.get("cell_key") != row.get("cell_key") or result.get("status") != "PASS":
            raise ProtocolError(f"OFFICIAL_RESULT_BINDING_MISMATCH:{row.get('cell_key')}")
        output.append(result)
    return output


def _plot_sample(run_dir: Path, record: Mapping[str, Any], official_rows: list[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapping = record["selected_map"]
    q = np.asarray(mapping["q"], dtype=np.float64)
    delta = q - np.arange(q.size, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].plot(np.arange(q.size), q, linewidth=1.4, label="q(j), source frame")
    axes[0].plot(np.arange(q.size), np.arange(q.size), linewidth=0.8, linestyle="--", label="identity")
    axes[0].set(xlabel="output frame j", ylabel="source coordinate", title=f"sample {record['sample_id']} / portrait 3")
    axes[0].legend()
    axes[1].plot(np.arange(delta.size), delta, linewidth=1.1)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(xlabel="output frame j", ylabel="delta (frames)", ylim=(-3.2, 3.2))

    by_arm = {str(row.get("video_arm")): row for row in official_rows
              if str(row.get("sample_id")) == str(record["sample_id"])
              and str(row.get("portrait_id")) == "3" and str(row.get("audio_role")) == "N"
              and row.get("status") == "PASS"}
    # Put official lag curves in a separate small axis so temporal map units stay clear.
    inset = axes[1].inset_axes([0.58, 0.10, 0.38, 0.72])
    for arm in ("M", "R", "GLOBAL", "NEAREST", "MIRROR"):
        row = by_arm.get(arm)
        curve = row.get("recomputed_official_curve", {}).get("curve") if row else None
        if curve:
            inset.plot(np.arange(-15, 16), curve, linewidth=1.0, label=arm)
    inset.set(title="official distance by lag", xlabel="lag", ylabel="distance")
    inset.legend(fontsize=6, loc="best")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"path": str(out.resolve()), "sha256": file_sha256(out), "sample_id": str(record["sample_id"])}


def _scientific_status(protocol: Mapping[str, Any], official: Mapping[str, Any], search: list[Mapping[str, Any]], comparisons: list[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    if protocol.get("engineering_only"):
        return "NOT_RUN", {"reason": "smoke run is engineering-only"}
    required = {"M/N": 0, "R/N": 0}
    rows = [row for row in official.get("rows", []) if row.get("portrait_id") == "3" and row.get("audio_role") == "N" and row.get("video_arm") in {"M", "R"} and row.get("status") == "PASS"]
    for row in rows:
        required[f"{row['video_arm']}/N"] += 1
    expected_n = len(protocol.get("sample_ids", []))
    if required["M/N"] != expected_n or required["R/N"] != expected_n:
        return "ENGINEERING_BLOCKED", {"reason": "official M/N or R/N denominator incomplete", "counts": required}
    by_pair = {(str(row.get("sample_id")), str(row.get("portrait_id")), str(row.get("audio_role"))): row for row in comparisons}
    deltas: list[float] = []
    passes = 0
    mismatch: list[str] = []
    for sample_id in protocol["sample_ids"]:
        pair = by_pair.get((str(sample_id), "3", "N"))
        if not pair or pair.get("status") != "MATCHED":
            mismatch.append(str(sample_id))
            continue
        dc = float(pair["official_delta_sync_c"])
        dd0 = float(pair["official_delta_d0"])
        m = next(row for row in rows if str(row["sample_id"]) == str(sample_id) and row["video_arm"] == "M")
        r = next(row for row in rows if str(row["sample_id"]) == str(sample_id) and row["video_arm"] == "R")
        okay = dc >= 0.100 and dd0 <= -0.020 and abs(int(r["official_offset"])) <= 1
        passes += int(okay)
        deltas.append(dc)
    if mismatch:
        return "ENGINEERING_BLOCKED", {"reason": "SUPPORT_MISMATCH blocks primary official contrast", "sample_ids": mismatch}
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    # This protocol's promising threshold is fixed to exactly three samples.
    if expected_n == 3 and passes >= 2 and all(value >= -0.050 for value in deltas) and mean_delta >= 0.100:
        return "ORACLE_RETIMING_PROMISING", {"official_delta_sync_c": deltas, "mean_delta_sync_c": mean_delta, "passing_samples": passes}
    has_search_gain = any(float(row.get("selected", {}).get("metrics", {}).get("sync_c", 0.0)) - float(row.get("baseline", {}).get("sync_c", 0.0)) >= 0.100 for row in search)
    if has_search_gain:
        return "SEARCH_GAIN_ONLY", {"official_delta_sync_c": deltas, "mean_delta_sync_c": mean_delta, "passing_samples": passes}
    if all(row.get("status") in {"NO_ACCEPTABLE_WARP", "BUDGET_LIMITED"} and row.get("selected", {}).get("label") == "identity" for row in search):
        return "SEARCH_NO_ACCEPTABLE_WARP", {"official_delta_sync_c": deltas, "mean_delta_sync_c": mean_delta, "passing_samples": passes}
    return "SEARCH_GAIN_ONLY", {"official_delta_sync_c": deltas, "mean_delta_sync_c": mean_delta, "passing_samples": passes}


def _build_blind_package(run_dir: Path, official: Mapping[str, Any], sample_ids: list[str]) -> dict[str, Any]:
    seed = 20260923
    rng = np.random.default_rng(seed)
    by_key = {str(row.get("cell_key")): row for row in official.get("rows", [])}
    packet_rows, key_rows = [], []
    for sample_id in sample_ids:
        mkey = f"p3_s{sample_id}_vM_aN"
        rkey = f"p3_s{sample_id}_vR_aN"
        mrow, rrow = by_key.get(mkey), by_key.get(rkey)
        if not mrow or not rrow or mrow.get("status") != "PASS" or rrow.get("status") != "PASS":
            packet_rows.append({"sample_id": str(sample_id), "status": "UNAVAILABLE", "reason": "M/N or R/N official media unavailable"})
            continue
        left_is_r = bool(rng.integers(0, 2))
        left = rrow if left_is_r else mrow
        right = mrow if left_is_r else rrow
        packet_rows.append({"sample_id": str(sample_id), "status": "READY",
                            "left_video": str((run_dir / "06_official" / "cells" / left["cell_key"] / "cell.mkv").resolve()),
                            "right_video": str((run_dir / "06_official" / "cells" / right["cell_key"] / "cell.mkv").resolve()),
                            "same_natural_audio": True,
                            "response_form": {"sync_preference": "", "double_image_or_jitter": ""}})
        key_rows.append({"sample_id": str(sample_id), "left_video_arm": "R" if left_is_r else "M", "right_video_arm": "M" if left_is_r else "R"})
    packet = {"schema_version": 1, "status": "READY" if all(row["status"] == "READY" for row in packet_rows) else "PARTIAL",
              "randomization_seed": seed, "sample_count_fixed": len(sample_ids), "all_samples_included": len(packet_rows) == len(sample_ids),
              "human_status": "HUMAN_NOT_ASSESSED", "rows": packet_rows}
    packet_path = run_dir / "09_report" / "blind_review" / "packet.json"
    key_path = packet_path.with_name("sealed_key.json")
    write_json(packet_path, packet, self_hash=True)
    key_payload = {"schema_version": 1, "sealed_before_responses": True, "packet_sha256": file_sha256(packet_path), "rows": key_rows}
    key_payload["key_sha256"] = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(key_path, key_payload)
    return {"packet": str(packet_path.resolve()), "packet_sha256": file_sha256(packet_path),
            "sealed_key": str(key_path.resolve()), "sealed_key_sha256": file_sha256(key_path),
            "human_status": "HUMAN_NOT_ASSESSED", "included_sample_ids": sample_ids}


def _stride5_local_diagnostics(visual_path: str, audio_path: str, support: list[int]) -> dict[str, Any]:
    from .check import np_distance_matrix
    with np.load(visual_path, allow_pickle=False) as handle:
        visual = np.asarray(handle["visual"], dtype=np.float32)
    with np.load(audio_path, allow_pickle=False) as handle:
        audio = np.asarray(handle["audio"], dtype=np.float32)
    rows = np.asarray(support, dtype=np.int64)
    matrix = np_distance_matrix(visual, audio, min(len(visual), len(audio)))
    if rows.size < 25:
        raise ProtocolError("STRIDE5_LOCAL_SUPPORT_TOO_SHORT")
    starts = sorted(set(range(0, rows.size - 24, 5)) | {rows.size - 25})
    windows = []
    for start in starts:
        block = rows[start:start + 25]
        curve = np.mean(matrix[block], axis=0, dtype=np.float32)
        index = int(np.argmin(curve))
        windows.append({"row_start": int(block[0]), "row_stop_exclusive": int(block[-1] + 1),
                        "d0": float(curve[15]), "offset": 15 - index})
    return {"stride_rows": 5, "window_rows": 25, "window_count": len(windows),
            "d0_p90": float(np.percentile([row["d0"] for row in windows], 90)),
            "abs_offset_p90": float(np.percentile([abs(row["offset"]) for row in windows], 90)),
            "windows": windows}


def _v2_sample_diagnostics(root: Path, sid: str, search_row: Mapping[str, Any],
                           official_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    state = verify_json(search_row["state_v2_path"], self_hash=True)
    calibration = verify_json(search_row["state_path"], self_hash=True)
    candidates = state["true_candidates"]
    selected_sha = str(search_row["selected"]["map_sha256"])
    wanted = {selected_sha, str(search_row["best_d0"]["map_sha256"]), str(search_row["best_c"]["map_sha256"])}
    proxy = {}
    for ref in state["proxy_chunks"]:
        path = Path(search_row["state_v2_path"]).parent / ref["name"]
        if file_sha256(path) != ref["sha256"]:
            raise ProtocolError(f"REPORT_PROXY_CHUNK_CHANGED:{sid}")
        for row in verify_json(path, self_hash=True)["rows"]:
            if row["map_sha256"] in wanted:
                proxy[row["map_sha256"]] = row["proxy_metrics"]
    selected_proxy = proxy.get(selected_sha)
    selected_true = search_row["selected"]["metrics"]
    calibration_forwards = int(calibration.get("calibration_forward_passes", 0))
    fresh_manifest = verify_json(root / "07_check" / "fresh_scores_manifest.json", self_hash=True)
    fresh_count = sum(str(row.get("sample_id")) == sid for row in fresh_manifest.get("rows", []))
    cells = [row for row in official_rows if str(row.get("sample_id")) == sid and row.get("status") == "PASS"]
    official_seconds = sum(float(command.get("elapsed_seconds", 0.0)) for row in cells for command in row.get("commands", []))
    support = calibration["frozen_support"]
    local = {"M": _stride5_local_diagnostics(calibration["baseline_candidate"]["embedding_path"],
                                               calibration["natural_audio_embedding_path"], support),
             "R": _stride5_local_diagnostics(search_row["selected"]["embedding_path"],
                                               calibration["natural_audio_embedding_path"], support)}
    baseline = search_row["baseline"]
    high_c_without_alignment = sum(float(row["metrics"]["sync_c"]) > float(baseline["sync_c"])
                                   and (float(row["metrics"]["d0"]) >= float(baseline["d0"])
                                        or abs(int(row["metrics"]["offset"])) >= abs(int(baseline["offset"])))
                                   for row in candidates.values())
    elapsed = float(state["elapsed_search_seconds"])
    true_seconds = float(state.get("true_forward_seconds", 0.0))
    return {"proxy_selected": selected_proxy,
            "proxy_selected_error": ({name: float(selected_proxy[name]) - float(selected_true[name])
                                      for name in ("sync_c", "sync_d", "d0")}
                                     if selected_proxy else None),
            "best_true_d0_in_shortlist": str(search_row["best_d0"]["map_sha256"]) in set(state["shortlist_a"] + state["shortlist_b"]),
            "best_true_c_in_shortlist": str(search_row["best_c"]["map_sha256"]) in set(state["shortlist_a"] + state["shortlist_b"]),
            "high_c_without_alignment_true_count": high_c_without_alignment,
            "local_stride5": local,
            "calibration_forward_count": calibration_forwards, "fresh_forward_count": fresh_count,
            "official_forward_count": len(cells), "official_command_seconds": official_seconds,
            "search_true_forward_seconds": true_seconds,
            "search_cpu_and_overhead_seconds": max(0.0, elapsed - true_seconds),
            "search_proxy_scoring_seconds": float(state.get("proxy_scoring_seconds", 0.0)),
            "search_total_seconds": elapsed}


def _write_report_v2(root: Path, protocol: Mapping[str, Any], sample_rows: list[dict[str, Any]],
                     search: list[Mapping[str, Any]], portrait_rows: list[Mapping[str, Any]],
                     official: Mapping[str, Any], comparisons: list[Mapping[str, Any]],
                     blind: Mapping[str, Any], plots: list[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_speaker[row["speaker_id"]].append(row)

    def summarize(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
        matched = [r for r in rows if (r.get("official_pair") or {}).get("status") == "MATCHED"]
        dc = np.asarray([float(r["official_pair"]["official_delta_sync_c"]) for r in matched], dtype=np.float64)
        dd0 = np.asarray([float(r["official_pair"]["official_delta_d0"]) for r in matched], dtype=np.float64)
        return {"denominator": denominator, "completed_pairs": len(matched),
                "alignment_gain_count": sum(r["selection_status"] in {"ALIGNMENT_GAIN", "ALIGNMENT_AND_C_GAIN"} for r in rows),
                "alignment_gain_rate_fixed_denominator": sum(r["selection_status"] in {"ALIGNMENT_GAIN", "ALIGNMENT_AND_C_GAIN"} for r in rows) / denominator,
                "identity_fallback_count": sum(r["identity_fallback"] for r in rows),
                "identity_fallback_rate_fixed_denominator": sum(r["identity_fallback"] for r in rows) / denominator,
                "fixed_c_ge_7_M": sum(float(r["search_baseline"]["sync_c"]) >= 7.0 for r in rows),
                "fixed_c_ge_7_R": sum(float(r["search_selected"]["sync_c"]) >= 7.0 for r in rows),
                "official_c_ge_7_M_N": sum(r.get("official_M_N") is not None and float(r["official_M_N"]["official_sync_c"]) >= 7.0 for r in rows),
                "official_c_ge_7_R_N": sum(r.get("official_R_N") is not None and float(r["official_R_N"]["official_sync_c"]) >= 7.0 for r in rows),
                "delta_sync_c_mean": float(dc.mean()) if dc.size else None,
                "delta_sync_c_p10": float(np.percentile(dc, 10)) if dc.size else None,
                "delta_sync_c_min": float(dc.min()) if dc.size else None,
                "delta_d0_mean": float(dd0.mean()) if dd0.size else None,
                "delta_d0_p10": float(np.percentile(dd0, 10)) if dd0.size else None,
                "delta_d0_min": float(dd0.min()) if dd0.size else None}

    nondev = [r for r in sample_rows if r["sample_id"] != "1"]
    groups = {"all64": summarize(sample_rows, 64), "nondevelopment63": summarize(nondev, 63),
              "S0770": summarize(by_speaker.get("S0770", []), 8),
              "prior_seen24": summarize([r for r in nondev if r["prior_seen"]], 24),
              "new39": summarize([r for r in nondev if not r["prior_seen"]], 39)}
    speaker_summary = {speaker: summarize(rows, 8) for speaker, rows in sorted(by_speaker.items())}
    rng = np.random.default_rng(20260923)
    speakers = sorted(by_speaker)
    means = []
    if len(speakers) == 8:
        for _ in range(20000):
            draw = rng.choice(speakers, size=8, replace=True)
            values = [float(r["official_pair"]["official_delta_sync_c"]) for speaker in draw for r in by_speaker[speaker]
                      if (r.get("official_pair") or {}).get("status") == "MATCHED"]
            if values:
                means.append(float(np.mean(values)))
    cluster_interval = [float(x) for x in np.percentile(means, [2.5, 97.5])] if means else None
    payload = {"schema_version": 2, "protocol": protocol["protocol"], "run_id": protocol["run_id"],
               "run_fingerprint": protocol["run_fingerprint"], "status": "ENGINEERING_ONLY" if protocol["engineering_only"] else "COMPLETE",
               "scientific_status": "NOT_RUN" if protocol["engineering_only"] else "DESCRIPTIVE_ONLY",
               "sample_count": len(sample_rows), "fixed_cohort_count": 64, "main_analysis_denominator": 63,
               "portrait_rows_are_not_independent_n": True, "samples": sample_rows, "groups": groups,
               "speakers": speaker_summary, "speaker_cluster_bootstrap_delta_c_95pct": cluster_interval,
               "portrait_transfer": portrait_rows, "official_cell_count": official["row_count"],
               "official_expected_cell_count": official["expected_row_count"],
               "official_support_comparisons": comparisons, "quality_status": "HUMAN_NOT_ASSESSED",
               "blind_review": blind, "plot_files": plots,
               "official_same_weight_is_not_independent_model_evidence": True}
    summary_path = root / "09_report" / "summary.json"
    write_json(summary_path, payload, self_hash=True)
    lines = ["# MFA-linear 视频重定时 v2：64 条固定队列", "",
             f"- 状态：{payload['status']}；官方 cell：{payload['official_cell_count']}/{payload['official_expected_cell_count']}。",
             "- 主分析固定分母 63（开发样本 1 单列）；S0770 八条是单说话人保留组。",
             "- 搜索用固定裁剪 SyncNet；官方链使用同一权重，均非独立模型验证。",
             "- 画质：HUMAN_NOT_ASSESSED。", "", "## 分组", "",
             "| 分组 | 完成/分母 | 对齐通过 | identity 回退 | 官方平均 ΔC | 官方平均 ΔD0 | ΔC p10/min |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, row in groups.items():
        fmt = lambda value: "—" if value is None else f"{value:+.3f}"
        lines.append(f"| {name} | {row['completed_pairs']}/{row['denominator']} | {row['alignment_gain_count']} | {row['identity_fallback_count']} | {fmt(row['delta_sync_c_mean'])} | {fmt(row['delta_d0_mean'])} | {fmt(row['delta_sync_c_p10'])}/{fmt(row['delta_sync_c_min'])} |")
    lines += ["", f"8 说话人簇 bootstrap 的 ΔC 均值 95% 区间：{cluster_interval if cluster_interval else '—'}。",
              "搜索 GPU 前向耗时包含像素重采样和评分；CPU/调度耗时为搜索墙钟扣除该部分。官方耗时为各 cell 命令之和，未包含视频生成和校准耗时。",
              "", "## C ≥ 7.000 达标数", "",
              "| 分组 | 固定裁剪 M/R | 官方 M/N、R/N |",
              "|---|---:|---:|"]
    for name, row in groups.items():
        lines.append(f"| {name} | {row['fixed_c_ge_7_M']}/{row['fixed_c_ge_7_R']} | {row['official_c_ge_7_M_N']}/{row['official_c_ge_7_R_N']} |")
    lines += ["", "## 搜索诊断", "",
              "| sample | 真实尝试/完成 | 校准/fresh/官方前向 | 搜索 CPU/真实前向/官方命令秒 | 选中代理误差 C/D0 | 短名单命中真实最佳 D0/C | stride5 D0 p90 M→R |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in sample_rows:
        error = row["proxy_selected_error"]
        error_text = "—" if error is None else f"{error['sync_c']:+.3f}/{error['d0']:+.3f}"
        local = row["local_stride5"]
        lines.append(f"| {row['sample_id']} | {row['true_candidate_forwards']}/{row['true_candidate_completed']} | "
                     f"{row['calibration_forward_count']}/{row['fresh_forward_count']}/{row['official_forward_count']} | "
                     f"{row['search_cpu_and_overhead_seconds']:.1f}/{row['search_true_forward_seconds']:.1f}/{row['official_command_seconds']:.1f} | "
                     f"{error_text} | {int(row['best_true_d0_in_shortlist'])}/{int(row['best_true_c_in_shortlist'])} | "
                     f"{local['M']['d0_p90']:.3f}→{local['R']['d0_p90']:.3f} |")
    lines += ["", "## 逐样本", "",
              "| sample | speaker | 终止 | 选择 | proxy/true | fixed C M→R | fixed D0 M→R | official C M/N→R/N |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in sample_rows:
        m, r = row["search_baseline"], row["search_selected"]
        om, orow = row["official_M_N"], row["official_R_N"]
        official_text = "—" if om is None or orow is None else f"{om['official_sync_c']:.3f}→{orow['official_sync_c']:.3f}"
        lines.append(f"| {row['sample_id']} | {row['speaker_id']} | {row['termination']} | {row['selection_status']} | {row['proxy_unique_maps']}/{row['true_candidate_forwards']} | {m['sync_c']:.3f}→{r['sync_c']:.3f} | {m['d0']:.3f}→{r['d0']:.3f} | {official_text} |")
    report_path = root / "09_report" / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "09_report" / "report_binding.json", {"summary_path": str(summary_path.resolve()), "summary_sha256": file_sha256(summary_path),
                                                              "report_path": str(report_path.resolve()), "report_sha256": file_sha256(report_path),
                                                              "plot_files": plots}, self_hash=True)
    return payload


def write_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    protocol = verify_json(root / "00_protocol" / "protocol.json", self_hash=True)
    frozen = verify_json(root / "00_protocol" / "frozen_inputs.json", self_hash=True)
    sealed = verify_json(root / "04_sealed" / "manifest.json", self_hash=True)
    official = verify_json(root / "06_official" / "manifest.json", self_hash=True)
    search = [verify_json(Path(str(row["search_result_path"])), self_hash=True) for row in sealed["records"]]
    comparisons = list(official.get("matched_support_comparisons", []))
    scientific_status, decision = _scientific_status(protocol, official, search, comparisons)
    official_results = _official_results(root, official)
    plots = [_plot_sample(root, record, official_results, root / "09_report" / "plots" / f"sample_{record['sample_id']}.png") for record in sealed["records"]]
    blind = _build_blind_package(root, official, [str(value) for value in protocol["sample_ids"]])
    by_cell = {(str(row.get("sample_id")), str(row.get("portrait_id")), str(row.get("video_arm")), str(row.get("audio_role"))): row for row in official_results}
    sample_rows = []
    for record, search_row in zip(sealed["records"], search, strict=True):
        sid = str(record["sample_id"])
        m, r, n = (by_cell.get((sid, "3", arm, "N")) for arm in ("M", "R", "N"))
        pair = next((row for row in comparisons if str(row.get("sample_id")) == sid and row.get("portrait_id") == "3" and row.get("audio_role") == "N"), None)
        sample_row = {"sample_id": sid, "paired_key": record["paired_key"], "speaker_id": record["speaker_id"],
                            "search_status": search_row["status"], "search_candidate_calls": search_row.get("search_result", {}).get("actual_score_calls"),
                            "search_elapsed_seconds": search_row.get("search_result", {}).get("elapsed_search_seconds"),
                            "search_baseline": search_row["baseline"], "search_selected": search_row["selected"]["metrics"],
                            "search_delta_sync_c": float(search_row["selected"]["metrics"]["sync_c"] - search_row["baseline"]["sync_c"]),
                            "official_M_N": m, "official_R_N": r, "official_N_N": n,
                            "official_pair": pair, "identity_fallback": bool(search_row["selected"].get("label") == "identity"),
                            "maximum_displacement_frames": float(record["interpolation_audit"]["max_abs_displacement"]),
                            "regularization": float(record["interpolation_audit"]["regularization"]),
                            "prior_seen": bool(next((r.get("prior_seen", False) for r in frozen["records"] if str(r["sample_id"]) == sid), False)),
                            "termination": search_row.get("termination"), "selection_status": search_row.get("selection_status"),
                            "proxy_unique_maps": search_row.get("search_result", {}).get("proxy_unique_maps"),
                            "true_candidate_forwards": search_row.get("search_result", {}).get("actual_score_calls"),
                            "true_candidate_completed": search_row.get("search_result", {}).get("true_completed"),
                            "best_true_d0": search_row.get("best_d0", {}).get("metrics"),
                            "best_true_c": search_row.get("best_c", {}).get("metrics"),
                            "global_true": search_row.get("global_control", {}).get("metrics")}
        if protocol.get("protocol") == "mfa_linear_video_retiming_v2":
            sample_row.update(_v2_sample_diagnostics(root, sid, search_row, official_results))
        sample_rows.append(sample_row)
    portrait_rows = []
    for portrait in ("6", "9"):
        local = []
        for sid in protocol["sample_ids"]:
            m = by_cell.get((str(sid), portrait, "M", "N"))
            r = by_cell.get((str(sid), portrait, "R", "N"))
            if m and r and m.get("status") == "PASS" and r.get("status") == "PASS":
                local.append({"sample_id": str(sid), "delta_sync_c": float(r["official_sync_c"] - m["official_sync_c"]),
                              "delta_d0": float(r["recomputed_official_curve"]["d0"] - m["recomputed_official_curve"]["d0"]),
                              "offset_m": m["official_offset"], "offset_r": r["official_offset"],
                              "support_m": m["recomputed_official_curve"]["support_columns"],
                              "support_r": r["recomputed_official_curve"]["support_columns"]})
        portrait_rows.append({"portrait_id": portrait, "paired_count": len(local), "pairs": local})
    if protocol.get("protocol") == "mfa_linear_video_retiming_v2":
        return _write_report_v2(root, protocol, sample_rows, search, portrait_rows, official, comparisons, blind, plots)
    payload = {"schema_version": 1, "protocol": "mfa_linear_video_retiming_v1", "run_id": protocol["run_id"],
               "run_fingerprint": protocol["run_fingerprint"], "status": "ENGINEERING_ONLY" if protocol.get("engineering_only") else "COMPLETE",
               "engineering_only": bool(protocol.get("engineering_only")), "scientific_status": scientific_status,
               "decision": decision, "sample_count": len(sample_rows), "independent_sample_count": len(sample_rows),
               "portrait_rows_are_not_independent_n": True, "samples": sample_rows,
               "portrait_transfer": portrait_rows, "official_cell_count": int(official.get("row_count", 0)),
               "official_expected_cell_count": int(official.get("expected_row_count", 0)),
               "official_same_weight_is_not_independent_model_evidence": True,
               "quality_status": "HUMAN_NOT_ASSESSED", "blind_review": blind,
               "plot_files": plots, "official_support_comparisons": comparisons,
               "mean_official_delta_sync_c": (float(np.mean([row["official_pair"]["official_delta_sync_c"] for row in sample_rows if row.get("official_pair") and row["official_pair"].get("status") == "MATCHED"]))
                                              if any(row.get("official_pair", {}).get("status") == "MATCHED" for row in sample_rows) else None),
               "caveat": "Search and official evaluation use the same frozen SyncNet weights; this is not independent-model validation. Three utterances support descriptive results only."}
    summary_path = root / "09_report" / "summary.json"
    write_json(summary_path, payload, self_hash=True)
    lines = ["# MFA-linear 视频重定时与自然音频同步", "",
             f"- 实验状态：`{payload['status']}`", f"- 科学判读：`{scientific_status}`",
             f"- 样本数：{len(sample_rows)}（独立语音样本；肖像迁移不增加 n）",
             f"- 官方评分 cell：{payload['official_cell_count']}/{payload['official_expected_cell_count']}",
             "- 画质状态：`HUMAN_NOT_ASSESSED`", "",
             "## 样本结果", "",
             "| sample | 搜索状态 | 搜索 ΔC | 固定裁剪 C/D/D0/offset（M→R） | 官方 C/D/D0/offset（M/N→R/N） | max |δ| | 官方 ΔC |",
             "|---|---|---:|---|---|---:|---:|"]
    for row in sample_rows:
        m, r = row.get("official_M_N"), row.get("official_R_N")
        pair = row.get("official_pair") or {}
        fixed_m, fixed_r = row["search_baseline"], row["search_selected"]
        official_text = "—" if not m or not r else f"{m['official_sync_c']:.3f}/{m['official_sync_d']:.3f}/{m['recomputed_official_curve']['d0']:.3f}/{m['official_offset']} → {r['official_sync_c']:.3f}/{r['official_sync_d']:.3f}/{r['recomputed_official_curve']['d0']:.3f}/{r['official_offset']}"
        delta_text = "—" if pair.get("status") != "MATCHED" else f"{float(pair['official_delta_sync_c']):+.3f}"
        lines.append(f"| {row['sample_id']} | {row['search_status']} | {row['search_delta_sync_c']:+.3f} | {fixed_m['sync_c']:.3f}/{fixed_m['sync_d']:.3f}/{fixed_m['d0']:.3f}/{fixed_m['offset']} → {fixed_r['sync_c']:.3f}/{fixed_r['sync_d']:.3f}/{fixed_r['d0']:.3f}/{fixed_r['offset']} | {official_text} | {row['maximum_displacement_frames']:.2f} | {delta_text} |")
    lines += ["", "## 判读", "", str(decision.get("reason", decision)), "",
              "官方完整链和固定裁剪搜索使用同一 SyncNet 权重；重新评分验证流程一致性，不是独立模型验证。3 条语音仅作描述性结果，不作稳定泛化或显著性结论。",
              "", "## 肖像迁移", ""]
    for row in portrait_rows:
        lines.append(f"- 肖像 {row['portrait_id']}：{row['paired_count']}/{len(protocol['sample_ids'])} 个可配对样本。")
        for pair in row["pairs"]:
            lines.append(f"  - {pair['sample_id']}：ΔC={pair['delta_sync_c']:+.3f}，ΔD0={pair['delta_d0']:+.3f}，offset {pair['offset_m']}→{pair['offset_r']}。")
    lines += ["", "## 画质与复核", "", f"盲评包：`{blind['packet']}`（固定包含所有入组样本；未收到人工评分）。", ""]
    for plot in plots:
        lines.append(f"- [{Path(plot['path']).name}]({Path(plot['path']).name})")
    report_path = root / "09_report" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "09_report" / "report_binding.json", {"summary_path": str(summary_path.resolve()), "summary_sha256": file_sha256(summary_path),
                                                              "report_path": str(report_path.resolve()), "report_sha256": file_sha256(report_path),
                                                              "plot_files": plots}, self_hash=True)
    return payload
