from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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
from .media import materialize
from .protocol import load_protocol, prepare
from .scoring import run_scores


def _manifest(path):
    return verify_self_hashed_json(path)


def _write_blocked(paths: config.RunPaths, error: Exception, stage: str) -> None:
    if paths.final.is_file():
        return
    paths.root.mkdir(parents=True, exist_ok=True)
    if not paths.result.is_file():
        paths.result.write_text(
            "# LRS3 Wav2Lip 局部时间传递诊断\n\n"
            f"- status: `BLOCKED`\n- failed stage: `{stage}`\n- error: `{type(error).__name__}: {error}`\n\n"
            f"- protocol: `{paths.protocol}`\n- input audit: `{paths.input_audit}`\n\n"
            "工程前提未满足，实验停止；未替补样本、调参、重试或改写历史 run。科学结论为空。\n",
            encoding="utf-8",
        )
    audit_binding = {"path": str(paths.input_audit.resolve()), "sha256": file_sha256(paths.input_audit)} if paths.input_audit.is_file() else {"path": str(paths.input_audit.resolve()), "sha256": None}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": None,
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "input_audit": audit_binding,
        "input_audit_sha256": audit_binding["sha256"],
        "spec_bindings": config.spec_bindings(),
        "parent_runs": {
            "calibration_final": {"path": str(config.CALIBRATION_FINAL.resolve()), "sha256": config.CALIBRATION_FINAL_SHA256},
            "tail_final": {"path": str(config.TAIL_FINAL.resolve()), "sha256": config.TAIL_FINAL_SHA256},
        },
        "main_cell_count": 0,
        "repeat_cell_count": 0,
        "cell_count": 0,
        "reference_conditioned_audio_head_spec_eligible": False,
        "eligibility": False,
        "result_sha256": file_sha256(paths.result),
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
    selected = stages if stage == "all" else (stage,)
    if stage not in (*stages, "all"):
        raise ValueError(f"unknown stage: {stage}")
    try:
        result: dict[str, Any] = {}
        for current in selected:
            result = _run_stage(current, paths)
        if stage == "all":
            from .validate import validate_run, write_validation

            validation = validate_run(paths.root)
            write_validation(paths, validation)
            result = verify_self_hashed_json(paths.final)
            result["validation_status"] = validation.get("status")
        return result
    except Exception as exc:
        _write_blocked(paths, exc, selected[-1])
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen LRS3 Wav2Lip timing-transfer diagnostic")
    parser.add_argument("--run-id", required=True, help="safe suffix for the new run root")
    parser.add_argument("--stage", choices=("prepare", "media", "score", "report", "all"), default="all")
    args = parser.parse_args(argv)
    result = run(args.run_id, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
