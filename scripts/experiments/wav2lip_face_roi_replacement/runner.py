from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .analysis import analyze_bridge, analyze_control, write_analysis
from .common import (
    ProtocolError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .media import materialize_stage
from .protocol import load_protocol, prepare
from .render import render_stage
from .scoring import run_scores
from .validate import validate_run


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _require_gpu() -> dict[str, Any]:
    if not config.WAV2LIP_PYTHON.is_file():
        raise ProtocolError(f"Wav2Lip Python is missing: {config.WAV2LIP_PYTHON}")
    environment = dict(os.environ)
    result = subprocess.run([str(config.WAV2LIP_PYTHON), "-c", "import torch; print(torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"], cwd=str(config.WAV2LIP_ROOT), env=environment, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"CUDA is unavailable for the registered Wav2Lip run: {result.stdout.strip()} {result.stderr.strip()}".strip())
    return {"python": str(config.WAV2LIP_PYTHON), "cuda_available": True, "stdout": result.stdout.strip()}


def _write_blocked(paths: config.RunPaths, stage: str, reason: str, *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    if paths.final.exists():
        raise ProtocolError(f"terminal run cannot be overwritten: {paths.final}")
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": None,
        "blocked_stage": stage,
        "block_reason": str(reason),
        "runtime": runtime or {},
        "counts": {"control_videos": 0, "control_scores": 0, "bridge_videos": 0, "bridge_scores": 0},
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
        "cross_model_spec_eligible": False,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(f"# Wav2Lip face-ROI replacement pilot\n\n- Engineering: `BLOCKED`\n- Stage: `{stage}`\n- Reason: {reason}\n\n科学终态为 `null`；未声称实验通过。\n", encoding="utf-8")
    payload["result_sha256"] = file_sha256(paths.result)
    write_self_hashed_json(paths.final, payload)
    try:
        validation = validate_run(paths.root)
    except Exception as exc:  # noqa: BLE001  # preserve the original block reason if validation itself has a defect
        validation = {"status": "invalid", "error": str(exc)}
    print(json.dumps({"status": "blocked", "stage": stage, "reason": reason, "validation": validation}, ensure_ascii=False), flush=True)
    return payload


def _run_control(paths: config.RunPaths, protocol: dict[str, Any]) -> dict[str, Any]:
    videos = render_stage(protocol, paths, "control")
    media = materialize_stage(protocol, videos, paths, "control")
    scores = run_scores(protocol, media, paths, "control")
    control = analyze_control(protocol, scores)
    return write_analysis(paths.control, {**control, "video_manifest_sha256": file_sha256(paths.videos / "control/manifest.json"), "media_manifest_sha256": file_sha256(paths.media / "control/manifest.json"), "score_manifest_sha256": file_sha256(paths.scores / "control/manifest.json")})


def _run_bridge(paths: config.RunPaths, protocol: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if not bool(control.get("control_pass")):
        raise ProtocolError("bridge stage is forbidden because control did not pass")
    videos = render_stage(protocol, paths, "bridge")
    media = materialize_stage(protocol, videos, paths, "bridge")
    scores = run_scores(protocol, media, paths, "bridge")
    bridge = analyze_bridge(protocol, control, _load(paths.scores / "control/manifest.json"), scores)
    return write_analysis(paths.bridge, {**bridge, "video_manifest_sha256": file_sha256(paths.videos / "bridge/manifest.json"), "media_manifest_sha256": file_sha256(paths.media / "bridge/manifest.json"), "score_manifest_sha256": file_sha256(paths.scores / "bridge/manifest.json")})


def _result_markdown(protocol: dict[str, Any], control: dict[str, Any], bridge: dict[str, Any] | None, *, final_decision: str) -> str:
    lines = [
        "# Wav2Lip face-ROI replacement pilot",
        "",
        f"- Scientific terminal: `{final_decision}`",
        f"- Cohort: `{protocol['record_count'] if 'record_count' in protocol else len(protocol['records'])}` seen-fit records / 22 source groups",
        f"- Control: `{control['scientific_decision']}`; 66 generated videos, 198 new scores",
    ]
    if bridge is None:
        lines.append("- Bridge: `NOT_RUN_CONTROL_FAILED`")
    else:
        lines.append(f"- Bridge: `{bridge['scientific_decision']}`; 22 generated videos, 44 new scores")
        gain = bridge["gates"]["positive_primary_gain"]["gain_C"]["ci95"]
        lines.append(f"- Primary gain-C 95% CI: `[{gain[0]:.3f}, {gain[1]:.3f}]`")
    lines.extend(["", "本轮仅覆盖已见 fit-only 的 22 条记录；不授权训练、不证明 reference-conditioned audio head，也不建立跨模型泛化。"])
    return "\n".join(lines) + "\n"


def _finalize(paths: config.RunPaths, protocol: dict[str, Any], control: dict[str, Any], bridge: dict[str, Any] | None) -> dict[str, Any]:
    if paths.final.exists():
        raise ProtocolError("terminal run cannot be finalized twice")
    if bridge is None:
        decision = "CONTROL_FAILED"
        bridge_status = "NOT_RUN_CONTROL_FAILED"
    else:
        decision = str(bridge["scientific_decision"])
        bridge_status = "COMPLETE"
    paths.result.write_text(_result_markdown(protocol, control, bridge, final_decision=decision), encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "classification": "seen_fit_pilot",
        "protocol_sha256": str(protocol["_sha256"]),
        "input_audit": dict(protocol["input_audit"]),
        "spec_bindings": dict(protocol["spec_bindings"]),
        "parent_runs": dict(protocol["parent_runs"]),
        "control_status": str(control["scientific_decision"]),
        "control_pass": bool(control.get("control_pass")),
        "bridge_status": bridge_status,
        "control_analysis_sha256": file_sha256(paths.control),
        "control_video_manifest_sha256": file_sha256(paths.videos / "control/manifest.json"),
        "control_media_manifest_sha256": file_sha256(paths.media / "control/manifest.json"),
        "control_score_manifest_sha256": file_sha256(paths.scores / "control/manifest.json"),
        "counts": {"control_generated_videos": config.EXPECTED_CONTROL_VIDEO_COUNT, "control_scores": config.EXPECTED_CONTROL_SCORE_COUNT, "bridge_generated_videos": config.EXPECTED_BRIDGE_VIDEO_COUNT if bridge is not None else 0, "bridge_scores": config.EXPECTED_BRIDGE_CELL_COUNT if bridge is not None else 0},
        "expected_counts": {"control_generated_videos": config.EXPECTED_CONTROL_VIDEO_COUNT, "control_scores": config.EXPECTED_CONTROL_SCORE_COUNT, "bridge_generated_videos": config.EXPECTED_BRIDGE_VIDEO_COUNT if bridge is not None else 0, "bridge_scores": config.EXPECTED_BRIDGE_CELL_COUNT if bridge is not None else 0},
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
        "cross_model_spec_eligible": bool(bridge is not None and decision == "WAV2LIP_REPLACEMENT_PILOT_PASS"),
        "result_sha256": file_sha256(paths.result),
    }
    if bridge is not None:
        payload.update({"bridge_analysis_sha256": file_sha256(paths.bridge), "bridge_video_manifest_sha256": file_sha256(paths.videos / "bridge/manifest.json"), "bridge_media_manifest_sha256": file_sha256(paths.media / "bridge/manifest.json"), "bridge_score_manifest_sha256": file_sha256(paths.scores / "bridge/manifest.json")})
    write_self_hashed_json(paths.final, payload)
    validation = validate_run(paths.root)
    print(json.dumps({"status": "complete", "scientific_decision": decision, "validation": validation}, ensure_ascii=False), flush=True)
    return verify_self_hashed_json(paths.final)


def run(run_id: str, stage: str) -> dict[str, Any]:
    paths = _paths(run_id)
    if paths.final.is_file():
        raise ProtocolError(f"terminal run cannot be resumed or overwritten: {paths.final}")
    if stage in ("all", "control", "bridge"):
        _require_gpu()
    if stage == "prepare":
        result = prepare(paths)
        print(json.dumps({"stage": "prepare", "status": result.get("status"), "protocol": str(paths.protocol)}, ensure_ascii=False), flush=True)
        return result
    if stage == "all":
        protocol = prepare(paths)
        protocol = load_protocol(paths)
        control = _run_control(paths, protocol)
        if bool(control.get("control_pass")):
            bridge = _run_bridge(paths, protocol, control)
        else:
            bridge = None
        return _finalize(paths, protocol, control, bridge)
    protocol = load_protocol(paths)
    if stage == "control":
        control = _run_control(paths, protocol)
        if not bool(control.get("control_pass")):
            _finalize(paths, protocol, control, None)
        return control
    if stage == "bridge":
        control = _load(paths.control)
        bridge = _run_bridge(paths, protocol, control)
        return _finalize(paths, protocol, control, bridge)
    if stage == "report":
        control = _load(paths.control)
        bridge = _load(paths.bridge) if paths.bridge.is_file() else None
        if bool(control.get("control_pass")) and bridge is None:
            raise ProtocolError("control passed but bridge artifacts are missing; run --stage bridge")
        return _finalize(paths, protocol, control, bridge)
    raise ValueError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Wav2Lip face-ROI replacement pilot")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "control", "bridge", "report", "all"), default="all")
    args = parser.parse_args(argv)
    try:
        run(args.run_id, args.stage)
    except Exception as exc:  # noqa: BLE001  # runner converts execution defects to the required terminal BLOCKED state
        paths = _paths(args.run_id)
        if not paths.final.exists():
            try:
                _write_blocked(paths, args.stage, str(exc))
            except Exception as block_exc:  # noqa: BLE001  # report both the original and block-write errors
                print(json.dumps({"status": "invalid", "error": str(exc), "block_write_error": str(block_exc)}, ensure_ascii=False), flush=True)
        else:
            print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
