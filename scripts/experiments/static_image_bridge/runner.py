from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .analysis import analyze_full, analyze_stage_a, write_per_record_csv
from .audio import prepare_audio
from .common import ProtocolError, verify_self_hashed_json, write_self_hashed_json
from .generate import render_stage
from .images import prepare_images, validate_static_inputs
from .protocol import load_frozen_inputs, prepare_protocol
from .report import write_report
from .scoring import score_stage
from .validate import validate_full, validate_stage_a


def _paths(run_root: Path) -> config.RunPaths:
    root = run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def prepare(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    base = load_frozen_inputs()
    if paths.inputs.is_file():
        existing = verify_self_hashed_json(paths.inputs)
        if existing.get("static_reference_contract"):
            inputs = existing
            failures = validate_static_inputs(inputs)
            if failures:
                raise ProtocolError("existing static inputs failed validation: " + "; ".join(failures))
        else:
            inputs = prepare_images(paths, base)
    else:
        inputs = prepare_images(paths, base)
    if paths.inputs.is_file():
        existing = verify_self_hashed_json(paths.inputs)
        if existing.get("records") != inputs.get("records") or existing.get("parents") != inputs.get("parents"):
            raise ProtocolError("existing inputs.json differs from current frozen/static inputs")
        inputs = existing
    else:
        write_self_hashed_json(paths.inputs, inputs)
        inputs = verify_self_hashed_json(paths.inputs)
    protocol, inputs = prepare_protocol(paths, inputs)
    if paths.audio_manifest.is_file():
        audio = verify_self_hashed_json(paths.audio_manifest)
        if audio.get("record_count") != config.EXPECTED_RECORD_COUNT:
            raise ProtocolError("existing audio manifest is incomplete")
    else:
        audio = prepare_audio(paths, inputs)
    return {"protocol": protocol, "inputs": inputs, "audio": audio, "run_root": str(paths.root)}


def stage_a(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    if not paths.protocol.is_file() or not paths.audio_manifest.is_file():
        prepare(paths.root)
    _protocol, inputs = prepare_protocol(paths, verify_self_hashed_json(paths.inputs))
    audio = verify_self_hashed_json(paths.audio_manifest)
    render_stage(paths, inputs, audio, config.STAGE_A_VIDEOS)
    videos = verify_self_hashed_json(paths.video_manifest)
    scores = score_stage(paths, inputs, audio, videos, "A")
    result = analyze_stage_a(paths, inputs, {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in scores["rows"]})
    return result


def validate_a(run_root: Path) -> dict[str, Any]:
    return validate_stage_a(_paths(run_root))


def stage_b(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    if not paths.stage_a.is_file():
        raise ProtocolError("stage A must be run before stage B")
    stage_a_payload = verify_self_hashed_json(paths.stage_a)
    if stage_a_payload.get("measurement_decision") != "PASS":
        raise ProtocolError("stage A measurement gate failed; stage B is intentionally zero")
    inputs = verify_self_hashed_json(paths.inputs)
    audio = verify_self_hashed_json(paths.audio_manifest)
    render_stage(paths, inputs, audio, config.STAGE_B_VIDEOS)
    videos = verify_self_hashed_json(paths.video_manifest)
    scores = score_stage(paths, inputs, audio, videos, "B")
    result = analyze_full(paths, inputs, {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in scores["rows"]}, stage_a_payload)
    if paths.analysis.is_file():
        write_per_record_csv(paths, verify_self_hashed_json(paths.analysis))
    return result


def analyze(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    inputs = verify_self_hashed_json(paths.inputs)
    stage_a_payload = verify_self_hashed_json(paths.stage_a)
    scores = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in verify_self_hashed_json(paths.scores_manifest).get("rows", [])}
    result = analyze_full(paths, inputs, scores, stage_a_payload)
    if paths.analysis.is_file():
        write_per_record_csv(paths, verify_self_hashed_json(paths.analysis))
    return result


def validate(run_root: Path) -> dict[str, Any]:
    return validate_full(_paths(run_root))


def report(run_root: Path) -> dict[str, Any]:
    return write_report(_paths(run_root))


def run_all(run_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {"prepare": prepare(run_root)}
    output["stage_a"] = stage_a(run_root)
    output["validate_a"] = validate_a(run_root)
    if output["stage_a"].get("measurement_decision") == "PASS":
        output["stage_b"] = stage_b(run_root)
    else:
        output["analysis"] = analyze(run_root)
    output["validation"] = validate(run_root)
    output["report"] = report(run_root)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static-image Natural-to-TTS bridge experiment")
    parser.add_argument("command", choices=("prepare", "stage-a", "validate-a", "stage-b", "analyze", "validate", "report", "all"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.run_root.resolve() if args.run_root is not None else config.run_root_for(args.run_id or config.default_run_id())
    functions = {"prepare": prepare, "stage-a": stage_a, "validate-a": validate_a, "stage-b": stage_b, "analyze": analyze, "validate": validate, "report": report, "all": run_all}
    result = functions[args.command](root)
    print(result)
    if args.command == "validate-a":
        return 0 if result.get("status") == "PASS" else 1
    if args.command == "validate":
        return 0 if result.get("status") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
