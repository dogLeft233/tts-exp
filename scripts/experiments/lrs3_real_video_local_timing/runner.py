from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .analysis import report
from .common import (
    DiagnosticError,
    assert_run_root_compatible,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .media import materialize, prepare
from .protocol import load_protocol
from .scoring import run_scores


def _manifest(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _write_blocked(paths: config.RunPaths, error: Exception, stage: str) -> None:
    if paths.final.is_file():
        return
    paths.root.mkdir(parents=True, exist_ok=True)
    result = paths.root / "result.md"
    audit_sha = file_sha256(paths.input_audit) if paths.input_audit.is_file() else None
    result.write_text(
        "# LRS3 真实视频局部时间敏感性诊断\n\n"
        f"- status: `BLOCKED`\n- failed stage: `{stage}`\n- error: `{type(error).__name__}: {error}`\n\n"
        f"- protocol revision: `{config.PROTOCOL_REVISION}`\n- input audit: `{paths.input_audit}`\n\n"
        "科学结论为空；实验在工程前提未满足处停止，未替补样本、调参或重试。\n",
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": None,
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "protocol_sha256": file_sha256(paths.protocol) if paths.protocol.is_file() else None,
        "input_audit": {
            "path": str(paths.input_audit.resolve()),
            "sha256": audit_sha,
        },
        "input_audit_sha256": audit_sha,
        "tail_contract": {
            "revision": config.PROTOCOL_REVISION,
            "min_samples": config.AUDIO_TAIL_MIN_SAMPLES,
            "max_samples": config.AUDIO_TAIL_MAX_SAMPLES,
            "comparison": "integer",
        },
        "spec_bindings": {
            "parent_spec": {"path": str(config.PARENT_SPEC.resolve()), "sha256": config.PARENT_SPEC_SHA256},
            "amendment_spec": {"path": str(config.AMENDMENT_SPEC.resolve()), "sha256": config.AMENDMENT_SPEC_SHA256},
        },
        "parent_blocked_run": {
            "run_id": "lrs3_real_video_local_timing_20260905_v2",
            "path": str(config.PARENT_BLOCKED_FINAL.resolve()),
            "sha256": config.PARENT_BLOCKED_FINAL_SHA256,
            "result_sha256": config.PARENT_BLOCKED_RESULT_SHA256,
        },
        "result_sha256": file_sha256(result),
        "eligibility": False,
    }
    write_self_hashed_json(paths.final, payload)


def _run_stage(stage: str, paths: config.RunPaths) -> dict[str, Any]:
    if stage == "prepare":
        return prepare(paths)
    protocol = load_protocol(paths)
    if stage == "media":
        return materialize(paths, protocol)
    media_manifest = _manifest(paths.media / "manifest.json")
    if stage == "score":
        return run_scores(protocol, media_manifest, paths)
    score_manifest = _manifest(paths.scores / "manifest.json")
    if stage == "report":
        return report(protocol, media_manifest, score_manifest, paths)
    raise ValueError(f"unknown stage: {stage}")


def run(run_id: str, stage: str = "all") -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    assert_run_root_compatible(paths.root)
    if paths.final.is_file():
        raise DiagnosticError("terminal run cannot be overwritten; use a new run id")
    stages = ("prepare", "media", "score", "report")
    if stage == "all":
        selected = stages
    elif stage in stages:
        selected = (stage,)
    else:
        raise ValueError(f"unknown stage: {stage}")
    try:
        result: dict[str, Any] = {}
        for current in selected:
            result = _run_stage(current, paths)
        return result
    except Exception as exc:
        _write_blocked(paths, exc, selected[-1])
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed LRS3 real-video local timing diagnostic")
    parser.add_argument("--run-id", required=True, help="safe suffix for the new run root")
    parser.add_argument("--stage", choices=("prepare", "media", "score", "report", "all"), default="all")
    args = parser.parse_args(argv)
    result = run(args.run_id, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
