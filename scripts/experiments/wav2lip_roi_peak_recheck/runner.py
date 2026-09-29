from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config
from .analysis import analyze, build_input_rows, selected_records
from .common import (
    RecheckError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .worker import SyncNetScorer


def _fixed_parent() -> dict[str, Any]:
    try:
        from scripts.experiments.wav2lip_roi_control_diagnostic.analysis import (
            load_parent,
        )

        return load_parent()
    except Exception as exc:
        raise RecheckError(f"fixed parent audit failed: {exc}") from exc


def _diagnostic_binding() -> dict[str, Any]:
    final_path = config.DIAGNOSTIC_ROOT / "final.json"
    diagnostics_path = config.DIAGNOSTIC_ROOT / "diagnostics.json"
    final = verify_self_hashed_json(final_path, config.DIAGNOSTIC_FINAL_SHA256)
    diagnostics = verify_self_hashed_json(diagnostics_path, config.DIAGNOSTIC_DIAGNOSTICS_SHA256)
    validation = verify_self_hashed_json(config.DIAGNOSTIC_VALIDATION)
    if final.get("status") != "complete" or final.get("diagnostic_decision") != "CONTROL_FAILURE_REPRODUCED":
        raise RecheckError("fixed diagnostic final is not the expected complete reproduction")
    if diagnostics.get("status") != "complete" or validation.get("status") != "valid":
        raise RecheckError("fixed diagnostic evidence is not valid")
    if final.get("diagnostics_sha256") != config.DIAGNOSTIC_DIAGNOSTICS_SHA256:
        raise RecheckError("fixed diagnostic final does not bind diagnostics.json")
    if validation.get("final_sha256") != config.DIAGNOSTIC_FINAL_SHA256:
        raise RecheckError("fixed diagnostic validation does not bind final.json")
    baseline_by_sample: dict[str, Any] = {}
    for row in diagnostics.get("per_record", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("baseline"), Mapping):
            raise RecheckError("fixed diagnostic baseline evidence is incomplete")
        baseline = row["baseline"].get("G_N_N")
        if not isinstance(baseline, Mapping) or not isinstance(baseline.get("passes"), bool):
            raise RecheckError(f"fixed diagnostic G_N/N baseline is incomplete: {row.get('sample_id')}")
        baseline_by_sample[str(row["sample_id"])] = {
            "passes": bool(baseline["passes"]),
            "plus": baseline.get("plus"),
            "minus": baseline.get("minus"),
            "direction_difference": baseline.get("direction_difference"),
        }
    return {
        "final": {"path": str(final_path.resolve()), "sha256": file_sha256(final_path)},
        "diagnostics": {"path": str(diagnostics_path.resolve()), "sha256": file_sha256(diagnostics_path)},
        "validation": {"path": str(config.DIAGNOSTIC_VALIDATION.resolve()), "sha256": file_sha256(config.DIAGNOSTIC_VALIDATION)},
        "decision": final["diagnostic_decision"],
        "baseline_g_n_n": baseline_by_sample,
    }


def _source_bindings() -> dict[str, dict[str, str]]:
    package = Path(__file__).resolve().parent
    sources = [(f"new/{path.name}", path) for path in sorted(package.glob("*.py"))]
    sources.extend(
        [
            ("official/SyncNetModel.py", config.SYNCNET_ROOT / "SyncNetModel.py"),
            ("official/SyncNetInstance.py", config.SYNCNET_ROOT / "SyncNetInstance.py"),
            ("official/syncnet_v2.model", config.SYNCNET_MODEL),
            ("audit_helper/analysis.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/analysis.py"),
            ("audit_helper/common.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/common.py"),
            ("audit_helper/config.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/config.py"),
        ]
    )
    return {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in sources}


def _change_bindings() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("proposal", config.PROPOSAL), ("design", config.DESIGN), ("spec", config.SPEC))
    }


def _environment(device: str, threads: int) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import python_speech_features
        import torch

        package_versions = {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "python_speech_features": getattr(python_speech_features, "__version__", "unknown"),
        }
        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as exc:
        raise RecheckError(f"scoring environment import failed: {exc}") from exc
    ffmpeg_version = subprocess.run([str(config.FFMPEG), "-version"], capture_output=True, text=True, check=False)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "device_requested": device,
        "torch_threads": int(threads),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "ffmpeg_first_line": (ffmpeg_version.stdout.splitlines() or [""])[0],
        "syncnet_python": str(config.SYNCNET_PYTHON),
        "syncnet_model": str(config.SYNCNET_MODEL.resolve()),
    }


def _protocol(parent: Mapping[str, Any], audit: Mapping[str, Any], run_id: str, device: str, threads: int) -> dict[str, Any]:
    inputs = build_input_rows(parent)
    diagnostic = _diagnostic_binding()
    records = []
    for record in selected_records(parent):
        masks = record["masks"]
        sample_id = str(record["sample_id"])
        baseline = diagnostic["baseline_g_n_n"].get(sample_id)
        if not isinstance(baseline, Mapping):
            raise RecheckError(f"fixed G_N/N baseline is missing for selected record: {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "masks": {
                    "plus_rows": [int(item) for item in masks["plus_rows"]],
                    "minus_rows": [int(item) for item in masks["minus_rows"]],
                    "d_by_row": {str(key): float(value) for key, value in masks["d_by_row"].items()},
                },
                "baseline_g_n_n": dict(baseline),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "run_id": run_id,
        "status": "frozen",
        "parent": {
            "root": str(config.PARENT_ROOT),
            "resolved_root": str(config.PARENT_ROOT.resolve()),
            "fixed_entry_hashes": dict(config.PARENT_HASHES),
            "scientific_decision": "CONTROL_FAILED",
            "record_count": 22,
            "source_group_count": 22,
        },
        "diagnostic_binding": diagnostic,
        "change_bindings": _change_bindings(),
        "source_bindings": _source_bindings(),
        "frozen_config": config.FrozenConfig().to_dict(),
        "environment": _environment(device, threads),
        "selected_records": records,
        "selected_cells": inputs,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
        "audit_sha256": file_sha256(config.run_root_for(run_id) / "audit.json"),
    }


def _audit(parent: Mapping[str, Any], run_id: str, device: str, threads: int) -> dict[str, Any]:
    inputs = build_input_rows(parent)
    for item in inputs:
        media = Path(item["media"])
        source_audio = Path(item["source_audio"])
        if not media.is_file() or file_sha256(media) != item["media_sha256"]:
            raise RecheckError(f"selected media binding changed: {media}")
        if not source_audio.is_file() or file_sha256(source_audio) != item["source_audio_sha256"]:
            raise RecheckError(f"selected natural audio binding changed: {source_audio}")
    diagnostic = _diagnostic_binding()
    selected = selected_records(parent)
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "run_id": run_id,
        "device": device,
        "torch_threads": int(threads),
        "parent_final": parent["assets"]["final"],
        "parent_validation": parent["assets"]["validation"],
        "parent_control": parent["assets"]["control"],
        "parent_protocol": parent["assets"]["protocol"],
        "parent_score_manifest": parent["assets"]["score_manifest"],
        "parent_media_checks": parent["media_checks"],
        "diagnostic_binding": diagnostic,
        "selected_sample_ids": [str(record["sample_id"]) for record in selected],
        "selected_record_count": len(selected),
        "selected_cell_count": len(inputs),
        "historical_failure_ids": list(config.FAIL_IDS),
        "historical_pass_ids": list(config.PASS_IDS),
        "new_generated_videos": 0,
    }


def _result_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Wav2Lip ROI local peak recheck 2026-09-06",
        "",
        "本报告对固定父媒体重新进行 SyncNet forward，并保存了每个 cell 的新 visual/audio embedding 与 distance matrix。它不重测 own-audio，不生成视频，不运行 bridge。",
        "",
        f"- 终态：`{analysis['decision']}`",
        f"- 记录：{analysis['record_count']}/{analysis['expected_record_count']}；新评分 cell：{analysis['score_cell_count']}/{analysis['expected_score_cell_count']}",
        f"- 历史失败复现：{analysis['historical_failure_reproduced']}/{analysis['historical_failure_count']}；历史通过复现：{analysis['historical_pass_reproduced']}/{analysis['historical_pass_count']}",
        f"- 矩阵最大绝对差：`{analysis['matrix_max_abs_difference']:.9f}`（容限 `0.001`）",
        f"- 局部曲线最大绝对差：`{analysis['curve_max_abs_difference']:.9f}`（容限 `0.001`）",
        "",
        "`actual`、`expected`、`residual` 均按固定父 masks 逐段保留。这里的 residual 表示 SyncNet 局部峰响应偏差/误差；即使终态为 `PEAKS_REPRODUCED`，也不能据此将结果解释为生成器响应不足。",
        "",
        "| sample | group | historical | PLUS actual / expected / residual | MINUS actual / expected / residual | old→new offsets (G_N,G_W) | new C |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in analysis["per_record"]:
        sample = row["sample_id"]
        label = row["historical_label"]
        plus = row["new"]["PLUS"]["gn"]
        minus = row["new"]["MINUS"]["gn"]
        old_plus = row["historical"]["PLUS"]
        old_minus = row["historical"]["MINUS"]
        offsets = (
            f"P {old_plus['gn']['offset']}/{old_plus['gw']['offset']}→{row['new']['PLUS']['gn']['offset']}/{row['new']['PLUS']['gw']['offset']}; "
            f"M {old_minus['gn']['offset']}/{old_minus['gw']['offset']}→{row['new']['MINUS']['gn']['offset']}/{row['new']['MINUS']['gw']['offset']}"
        )
        lines.append(
            f"| `{sample}` | `{row['source_group']}` | `{label}` | "
            f"{plus['actual']:.3f} / {plus['expected']:.3f} / {plus['residual']:.3f} | "
            f"{minus['actual']:.3f} / {minus['expected']:.3f} / {minus['residual']:.3f} | {offsets} | `{row['new']['c_pass']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "新推理与独立距离/峰实现若复现历史矩阵与峰，只能说明独立评分实现复现了 SyncNet 所见的局部响应误差；它不能区分 SyncNet 表征限制、warp 后音频变化和生成器响应。若发生 `SCORER_MISMATCH`，本报告只列出具体 cell/列/数值，不调整容限。",
            "",
            "历史科学终态仍为 `CONTROL_FAILED`。own-audio 未重测，bridge、训练和跨模型泛化均未授权。",
            "",
            "建议：若需要继续，只针对本轮发现的具体评分差异设计单一预处理/距离修复；若 `PEAKS_REPRODUCED`，先停止本线实验，不把它扩展成生成器因果结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def _completed_cells(root: Path) -> tuple[int, int]:
    completed: dict[tuple[str, str], Path] = {}
    for worker_path in sorted((root / "scores" / "cells").glob("*/worker.json")):
        try:
            worker = verify_self_hashed_json(worker_path)
        except RecheckError:
            continue
        if worker.get("new_forward") is True:
            completed[(str(worker.get("sample_id")), str(worker.get("video_arm")))] = worker_path
    records = {sample_id for sample_id, _arm in completed}
    return len(completed), len(records)


def _write_blocked(
    paths: config.RunPaths,
    run_id: str,
    reason: str,
    completed_cells: int | None = None,
    completed_records: int | None = None,
    exception_type: str = "RecheckError",
    *,
    allow_existing: bool = False,
) -> None:
    if paths.root.exists() and any(paths.root.iterdir()) and not allow_existing:
        return
    paths.root.mkdir(parents=True, exist_ok=True)
    recovered_cells, recovered_records = _completed_cells(paths.root)
    if completed_cells is None:
        completed_cells = recovered_cells
    if completed_records is None:
        completed_records = recovered_records
    result = paths.result
    result.write_text(
        "# Wav2Lip ROI local peak recheck\n\n"
        f"终态：`BLOCKED`。已完成新评分 cell：{completed_cells}/{config.EXPECTED_CELL_COUNT}。\n\n"
        f"原因：{reason}\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "blocked",
        "diagnostic_decision": "BLOCKED",
        "historical_scientific_decision": "CONTROL_FAILED",
        "parent_root": str(config.PARENT_ROOT),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "record_count": completed_records,
        "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
        "score_cell_count": completed_cells,
        "new_generated_videos": 0,
        "new_score_cells": completed_cells,
        "own_audio_retested": False,
        "bridge_executed": False,
        "training_authorized": False,
        "generalization_established": False,
        "blocked_reason": reason,
        "exception_type": exception_type,
        "result_sha256": file_sha256(result),
    }
    write_self_hashed_json(paths.final, payload)


def run_recheck(run_id: str, device: str = "cpu", threads: int = config.TORCH_THREADS) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise RecheckError(f"run root already contains artifacts; use a new run-id: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    parent = _fixed_parent()
    audit = _audit(parent, run_id, device, threads)
    audit_sha = write_self_hashed_json(paths.audit, audit)
    protocol = _protocol(parent, audit, run_id, device, threads)
    protocol["audit_sha256"] = audit_sha
    protocol_sha = write_self_hashed_json(paths.protocol, protocol)
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device=device, batch_size=config.BATCH_SIZE, threads=threads)
    input_rows = [dict(item) for item in protocol["selected_cells"]]
    if len({(str(item["sample_id"]), str(item["video_arm"])) for item in input_rows}) != config.EXPECTED_CELL_COUNT:
        raise RecheckError("protocol selected cell order contains duplicate or missing cells")
    score_rows: list[dict[str, Any]] = []
    for index, item in enumerate(input_rows, 1):
        sample_id = str(item["sample_id"])
        video_arm = str(item["video_arm"])
        cell_dir = paths.scores / "cells" / config.cell_key(sample_id, video_arm)
        print(f"CELL_START {index}/{config.EXPECTED_CELL_COUNT} {sample_id} {video_arm}/N", flush=True)
        worker = scorer.score(
            Path(item["media"]),
            Path(item["source_audio"]),
            cell_dir,
            item["media_sha256"],
            item["media_pcm_sha256"],
        )
        worker.update(
            {
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "sample_id": sample_id,
                "source_group": item["source_group"],
                "video_arm": video_arm,
                "audio_arm": config.AUDIO_ARM,
                "repeat": config.REPEAT,
            }
        )
        worker_path = cell_dir / "worker.json"
        worker_sha = write_self_hashed_json(worker_path, worker)
        score_rows.append(
            {
                "schema_version": 1,
                "stage_id": "score",
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "sample_id": sample_id,
                "source_group": item["source_group"],
                "video_arm": video_arm,
                "audio_arm": config.AUDIO_ARM,
                "repeat": config.REPEAT,
                "media": item["media"],
                "media_sha256": item["media_sha256"],
                "media_pcm_sha256": item["media_pcm_sha256"],
                "source_audio": item["source_audio"],
                "source_audio_sha256": item["source_audio_sha256"],
                "source_audio_pcm_sha256": item["source_audio_pcm_sha256"],
                "parent_matrix": item["parent_matrix"],
                "parent_matrix_sha256": item["parent_matrix_sha256"],
                "parent_matrix_shape": item["parent_matrix_shape"],
                "cell_dir": str(cell_dir.resolve()),
                "worker_result": str(worker_path.resolve()),
                "worker_result_sha256": worker_sha,
                "visual": worker["visual"],
                "visual_sha256": worker["visual_sha256"],
                "audio_embedding": worker["audio_embedding"],
                "audio_embedding_sha256": worker["audio_embedding_sha256"],
                "matrix": worker["matrix"],
                "matrix_sha256": worker["matrix_sha256"],
                "matrix_shape": worker["matrix_shape"],
            }
        )
        print(f"CELL_DONE {index}/{config.EXPECTED_CELL_COUNT} {sample_id} {video_arm}/N matrix={worker['matrix_shape']}", flush=True)
    if len(score_rows) != config.EXPECTED_CELL_COUNT:
        raise RecheckError(f"new scoring ended with {len(score_rows)}/{config.EXPECTED_CELL_COUNT} cells")
    score_manifest = {
        "schema_version": 1,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "protocol_sha256": protocol_sha,
        "status": "complete",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(score_rows),
        "expected_cell_count": config.EXPECTED_CELL_COUNT,
        "scores": score_rows,
    }
    score_manifest_sha = write_self_hashed_json(paths.score_manifest, score_manifest)
    analysis = analyze(parent, protocol, score_rows)
    analysis["audit_sha256"] = audit_sha
    analysis["protocol_sha256"] = protocol_sha
    analysis["score_manifest_sha256"] = score_manifest_sha
    analysis_sha = write_self_hashed_json(paths.analysis, analysis)
    paths.result.write_text(_result_markdown(analysis), encoding="utf-8")
    final = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "diagnostic_decision": analysis["decision"],
        "historical_scientific_decision": "CONTROL_FAILED",
        "parent_root": str(config.PARENT_ROOT),
        "parent_final_sha256": config.PARENT_HASHES["final"],
        "protocol_sha256": protocol_sha,
        "audit_sha256": audit_sha,
        "score_manifest_sha256": score_manifest_sha,
        "analysis_sha256": analysis_sha,
        "result_sha256": file_sha256(paths.result),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "record_count": analysis["record_count"],
        "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
        "score_cell_count": analysis["score_cell_count"],
        "historical_failure_count": analysis["historical_failure_count"],
        "historical_failure_reproduced": analysis["historical_failure_reproduced"],
        "historical_pass_count": analysis["historical_pass_count"],
        "historical_pass_reproduced": analysis["historical_pass_reproduced"],
        "new_generated_videos": 0,
        "new_score_cells": config.EXPECTED_CELL_COUNT,
        "own_audio_retested": False,
        "bridge_executed": False,
        "training_authorized": False,
        "generalization_established": False,
        "causal_interpretation": "局部峰复核只检查独立评分实现；actual/expected/residual 是响应偏差/误差，不是生成器响应不足的证明。",
    }
    final_sha = write_self_hashed_json(paths.final, final)
    from .validate import validate_run

    validation = validate_run(paths.root)
    if not validation.get("valid"):
        raise RecheckError(f"offline validator rejected the complete run: {validation.get('errors')}")
    print(f"VALIDATION valid decision={analysis['decision']} final_sha256={final_sha}", flush=True)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recheck Wav2Lip ROI local SyncNet peaks with fresh forward passes")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    args = parser.parse_args(argv)
    paths = config.RunPaths(config.run_root_for(args.run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        print(f"BLOCKED: run root already contains artifacts; use a new run-id: {paths.root}", file=sys.stderr, flush=True)
        return 2
    try:
        run_recheck(args.run_id, args.device, args.threads)
    except Exception as exc:  # noqa: BLE001 - persist every CLI failure as blocked evidence
        _write_blocked(paths, args.run_id, str(exc), exception_type=type(exc).__name__, allow_existing=True)
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
