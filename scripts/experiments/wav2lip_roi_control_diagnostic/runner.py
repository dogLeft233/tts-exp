from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import diagnose, load_parent
from .common import DiagnosticError, file_sha256, write_self_hashed_json


def _source_bindings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[path.name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return result


def _spec_bindings(parent: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in config.SPEC_PATHS.items():
        if not path.is_file():
            raise DiagnosticError(f"diagnostic spec binding is missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    result["parent_roi_spec"] = dict(parent["inherited_specs"]["spec"])
    result["parent_timing_spec"] = dict(parent["inherited_specs"]["inherited_timing_spec"])
    return result


def _audit_payload(parent: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "parent_root": {"logical": parent["root"], "resolved": parent["resolved_root"]},
        "parent_assets": parent["assets"],
        "record_count": len(parent["records"]),
        "source_group_count": len({str(row["source_group"]) for row in parent["records"]}),
        "score_cell_count": config.EXPECTED_CELL_COUNT,
        "main_score_cell_count": config.EXPECTED_MAIN_CELL_COUNT,
        "repeat_score_cell_count": config.EXPECTED_REPEAT_CELL_COUNT,
        "matrix_file_count": len(parent["matrix_cache"]),
        "media_checks": parent["media_checks"],
        "fixed_parent_symlink_policy": "logical fix5 path and resolved fix1 path are both retained and accepted",
        "forbidden_actions_observed": {
            "new_generated_videos": 0,
            "new_score_cells": 0,
            "audio_modified": False,
        },
    }


def _protocol_payload(parent: dict[str, Any], audit: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "run_id": run_id,
        "status": "locked",
        "classification": "seen_fit_diagnostic",
        "parent_root": {"logical": parent["root"], "resolved": parent["resolved_root"]},
        "parent_assets": parent["assets"],
        "spec_bindings": _spec_bindings(parent),
        "source_bindings": _source_bindings(),
        "config": config.FrozenConfig().to_dict(),
        "audit_sha256": file_sha256(config.run_root_for(run_id) / "audit.json"),
        "record_count": len(parent["records"]),
        "source_group_count": len({str(row["source_group"]) for row in parent["records"]}),
        "score_cell_count": config.EXPECTED_CELL_COUNT,
        "matrix_file_count": len(parent["matrix_cache"]),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "bridge_executed": False,
        "training_authorized": False,
        "cross_model_spec_eligible": False,
        "generalization_established": False,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device_policy": "CPU-only; no CUDA or media generation",
        },
    }


def _result_markdown(diagnostics: dict[str, Any]) -> str:
    gates = diagnostics["gates"]
    own = diagnostics["own_audio_decomposition"]
    summary = diagnostics["c_failure_summary"]
    lines = [
        "# Wav2Lip ROI control failure diagnostic",
        "",
        f"- Diagnostic: `{diagnostics['diagnostic_decision']}`",
        f"- Historical scientific decision: `{diagnostics['historical_scientific_decision']}`",
        f"- Records/source groups: `{diagnostics['record_count']}/{diagnostics['source_group_count']}`",
        f"- New generated videos / score cells: `{diagnostics['new_generated_videos']}/{diagnostics['new_score_cells']}`",
        "",
        "## C and own-audio summary",
        "",
        f"- C failure: `{summary['failed_count']}/22`; flags are overlapping and use denominator 22.",
        "- C flags: " + ", ".join(f"`{key}`={value}" for key, value in summary["flag_counts"].items()),
        f"- own_C 95% CI: `[{own['C']['ci95'][0]:.6f}, {own['C']['ci95'][1]:.6f}]`; lower margin above -0.10: `{own['C_lower_margin_above_minus_0_10']:.6f}`",
        f"- own_D 95% CI: `[{own['D']['ci95'][0]:.6f}, {own['D']['ci95'][1]:.6f}]`; lower margin above -0.10: `{own['D_lower_margin_above_minus_0_10']:.6f}`",
        "- own_C is decomposed as `median_change + own_D`; zero-crossing is not itself a failure rule.",
        "",
        "## All records",
        "",
        "| sample_id | C pass | C flags | C+ residual | C− residual | own_C | own_D | own offset |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics["per_record"]:
        c = row["c_diagnostic"]
        flags = ",".join(key for key, value in c["flags"].items() if value) or "—"
        lines.append(
            "| {sample} | {passed} | {flags} | {plus:.6f} | {minus:.6f} | {own_c:.6f} | {own_d:.6f} | {offset} |".format(
                sample=row["sample_id"],
                passed="yes" if row["checks"]["C"]["passes"] else "no",
                flags=flags,
                plus=c["segments"]["PLUS"]["residual"],
                minus=c["segments"]["MINUS"]["residual"],
                own_c=row["own_audio"]["c"],
                own_d=row["own_audio"]["d"],
                offset="yes" if row["own_audio"]["offset_noninferior"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"- Recomputed inherited C gate: `{gates['inherited']['C']['count']}/22`; historical control remains `CONTROL_FAILED`.",
            f"- own-audio gate: `{gates['own_audio']['passes']}`; the lower bound is compared only with `-0.10`.",
            f"- History comparison: `{diagnostics['history_comparison']['status']}` ({diagnostics['history_comparison']['difference_count']} differences).",
            "",
            f"最小下一步建议：{diagnostics['recommendation']}",
            "",
            "本诊断不运行 bridge，不授权训练，不建立 replacement 或跨模型泛化结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_blocked(paths: config.RunPaths, run_id: str, reason: str) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    if not paths.audit.exists():
        write_self_hashed_json(
            paths.audit,
            {
                "schema_version": 1,
                "stage_id": "input_audit",
                "protocol_id": config.PROTOCOL_ID,
                "status": "blocked",
                "errors": [reason],
                "new_generated_videos": 0,
                "new_score_cells": 0,
            },
        )
    if not paths.protocol.exists():
        write_self_hashed_json(
            paths.protocol,
            {
                "schema_version": 1,
                "protocol_id": config.PROTOCOL_ID,
                "protocol_revision": config.PROTOCOL_REVISION,
                "status": "blocked",
                "run_id": run_id,
                "audit_sha256": file_sha256(paths.audit),
                "errors": [reason],
                "new_generated_videos": 0,
                "new_score_cells": 0,
            },
        )
    write_self_hashed_json(
        paths.diagnostics,
        {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "diagnostic_decision": "BLOCKED",
            "historical_scientific_decision": "CONTROL_FAILED",
            "reason": reason,
            "new_generated_videos": 0,
            "new_score_cells": 0,
            "bridge_executed": False,
            "training_authorized": False,
            "cross_model_spec_eligible": False,
            "generalization_established": False,
        },
    )
    paths.result.write_text(
        "# Wav2Lip ROI control failure diagnostic\n\n"
        "- Status: `BLOCKED`\n"
        f"- Reason: {reason}\n\n"
        "No new videos, scores, bridge, or training were run.\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "blocked",
        "diagnostic_decision": "BLOCKED",
        "historical_scientific_decision": "CONTROL_FAILED",
        "reason": reason,
        "protocol_sha256": file_sha256(paths.protocol),
        "audit_sha256": file_sha256(paths.audit),
        "diagnostics_sha256": file_sha256(paths.diagnostics),
        "result_sha256": file_sha256(paths.result),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "bridge_executed": False,
        "training_authorized": False,
        "cross_model_spec_eligible": False,
        "generalization_established": False,
    }
    write_self_hashed_json(paths.final, payload)
    return payload


def run_diagnostic(run_id: str) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise DiagnosticError(f"refusing non-empty diagnostic run root: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    parent = load_parent()
    audit = _audit_payload(parent)
    write_self_hashed_json(paths.audit, audit)
    protocol = _protocol_payload(parent, audit, run_id)
    write_self_hashed_json(paths.protocol, protocol)
    diagnostics = diagnose(parent)
    diagnostics["protocol_sha256"] = file_sha256(paths.protocol)
    diagnostics["audit_sha256"] = file_sha256(paths.audit)
    write_self_hashed_json(paths.diagnostics, diagnostics)
    paths.result.write_text(_result_markdown(diagnostics), encoding="utf-8")
    final_payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "diagnostic_decision": diagnostics["diagnostic_decision"],
        "historical_scientific_decision": "CONTROL_FAILED",
        "parent_final": parent["assets"]["final"],
        "protocol_sha256": file_sha256(paths.protocol),
        "audit_sha256": file_sha256(paths.audit),
        "diagnostics_sha256": file_sha256(paths.diagnostics),
        "result_sha256": file_sha256(paths.result),
        "record_count": diagnostics["record_count"],
        "source_group_count": diagnostics["source_group_count"],
        "score_cell_count": diagnostics["score_count"],
        "matrix_file_count": len(parent["matrix_cache"]),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "bridge_executed": False,
        "training_authorized": False,
        "cross_model_spec_eligible": False,
        "generalization_established": False,
    }
    write_self_hashed_json(paths.final, final_payload)
    from .validate import validate_run

    validation = validate_run(paths.root)
    if validation.get("status") != "valid":
        raise DiagnosticError(f"diagnostic validator failed: {validation}")
    print(json.dumps({"status": "complete", "diagnostic_decision": final_payload["diagnostic_decision"], "validation": validation}, ensure_ascii=False), flush=True)
    return final_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU-only Wav2Lip ROI control failure diagnostic")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    paths = config.RunPaths(config.run_root_for(args.run_id))
    try:
        run_diagnostic(args.run_id)
    except Exception as exc:  # noqa: BLE001 - a failed frozen audit must become a terminal BLOCKED marker
        if not paths.final.exists():
            try:
                payload = _write_blocked(paths, args.run_id, str(exc))
                print(json.dumps(payload, ensure_ascii=False), flush=True)
            except Exception as block_exc:  # noqa: BLE001 - preserve both failure causes
                print(json.dumps({"status": "invalid", "error": str(exc), "block_write_error": str(block_exc)}, ensure_ascii=False), flush=True)
        else:
            print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
