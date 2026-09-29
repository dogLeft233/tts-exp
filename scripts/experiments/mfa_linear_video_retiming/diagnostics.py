"""Offline retrospective of the embedding surrogate on historical true candidates."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .common import ProtocolError, file_sha256, verify_json, write_json
from .retime import build_map
from .search_worker import _load_embedding
from .surrogate import score_proxy


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    result: dict[str, Any] = {"count": len(rows)}
    for metric in ("sync_c", "sync_d", "d0"):
        actual = np.asarray([r[f"true_{metric}"] for r in rows], dtype=np.float64)
        proxy = np.asarray([r[f"proxy_{metric}"] for r in rows], dtype=np.float64)
        result[f"{metric}_mae"] = float(np.mean(np.abs(actual - proxy)))
        result[f"{metric}_spearman"] = float(spearmanr(actual, proxy).statistic)
    result["offset_exact_fraction"] = float(np.mean([r["true_offset"] == r["proxy_offset"] for r in rows]))
    actual_top = sorted(rows, key=lambda r: (-r["true_sync_c"], r["map_sha256"]))[:10]
    for k in (10, 25, 50):
        proxy_top = {r["map_sha256"] for r in sorted(rows, key=lambda r: (-r["proxy_sync_c"], r["map_sha256"]))[:k]}
        result[f"true_c_top10_in_proxy_top{k}"] = sum(r["map_sha256"] in proxy_top for r in actual_top) / len(actual_top)
    proxy_best = max(rows, key=lambda r: (r["proxy_sync_c"], -r["proxy_d0"]))
    actual_best = actual_top[0]
    result["proxy_selected_true_sync_c"] = proxy_best["true_sync_c"]
    result["true_best_proxy_c_rank"] = next(i for i, r in enumerate(sorted(rows, key=lambda r: (-r["proxy_sync_c"], r["map_sha256"])), 1) if r["map_sha256"] == actual_best["map_sha256"])
    return result


def compare_cached_candidates(source_run: Path, output_dir: Path) -> dict[str, Any]:
    state_path = source_run / "03_search" / "1" / "state.json"
    state = verify_json(state_path, self_hash=True)
    original = _load_embedding(state["baseline_candidate"]["embedding_path"], state["baseline_candidate"]["embedding_sha256"])["visual"]
    audio = _load_embedding(state["natural_audio_embedding_path"], state["natural_audio_embedding_sha256"])["audio"]
    rows = np.asarray(state["frozen_support"], dtype=np.int64)
    identity = build_map(state["frame_count"], state["valid_frame_count"])
    start = time.monotonic()
    output_rows: list[dict[str, Any]] = []
    for candidate in state["candidates"].values():
        actual_visual = _load_embedding(candidate["embedding_path"], candidate["embedding_sha256"])["visual"]
        actual = score_proxy(actual_visual, audio, identity, rows)
        for key in ("sync_c", "sync_d", "d0"):
            if abs(float(actual[key]) - float(candidate["metrics"][key])) > 1e-4:
                raise ProtocolError(f"HISTORICAL_TRUE_METRIC_REBUILD_FAILED:{candidate['candidate_id']}:{key}")
        if int(actual["offset"]) != int(candidate["metrics"]["offset"]):
            raise ProtocolError("HISTORICAL_TRUE_OFFSET_REBUILD_FAILED")
        row: dict[str, Any] = {"candidate_id": candidate["candidate_id"], "label": candidate["label"],
                               "map_sha256": candidate["map_sha256"], "stage": candidate["stage"]}
        for anchor in ("window_center", "window_start"):
            predicted = score_proxy(original, audio, candidate["map"], rows, anchor=anchor)
            for key in ("sync_c", "sync_d", "d0", "offset"):
                row[f"{anchor}_{key}"] = predicted[key]
        for key in ("sync_c", "sync_d", "d0", "offset"):
            row[f"true_{key}"] = actual[key]
            row[f"proxy_{key}"] = row[f"window_center_{key}"]
        output_rows.append(row)
    elapsed = time.monotonic() - start
    local = [r for r in output_rows if r["label"] != "identity" and not r["label"].startswith("global_seed_")]
    diagnostics = {"schema_version": 2, "protocol": "mfa_linear_video_retiming_v2",
                   "source_state": str(state_path.resolve()), "source_state_sha256": file_sha256(state_path),
                   "source_run": str(source_run.resolve()), "anchor": "window_center", "support_count": len(rows),
                   "elapsed_cpu_seconds": elapsed, "additional_model_forwards": 0,
                   "all_candidates": _rank_summary(output_rows), "local_candidates": _rank_summary(local)}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnostics.json", diagnostics, self_hash=True)
    with (output_dir / "per_candidate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = ["# Embedding 代理回溯诊断", "", f"历史真实候选：{len(output_rows)}；局部候选：{len(local)}。",
              f"CPU 耗时：{elapsed:.3f} 秒；新增模型前向：0。", "",
              "| 集合 | C Spearman | D0 Spearman | C MAE | offset 一致率 | true C top10 ∩ proxy C top10 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, stats in (("全部", diagnostics["all_candidates"]), ("局部", diagnostics["local_candidates"])):
        report.append(f"| {label} | {stats['sync_c_spearman']:.3f} | {stats['d0_spearman']:.3f} | {stats['sync_c_mae']:.3f} | {stats['offset_exact_fraction']:.3f} | {stats['true_c_top10_in_proxy_top10']:.3f} |")
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_cached_candidates(args.source_run.resolve(), args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
