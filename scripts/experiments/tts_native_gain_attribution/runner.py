"""Command-line orchestration for the native-gain attribution experiment."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.experiments.tts_native_gain_attribution"

from . import config
from .audio import audio_stage
from .audit import audit_stage, refresh_code_snapshot
from .common import (
    DependencyBlockedError,
    ProtocolError,
    ResourceWaitError,
    read_self_hashed_json,
    write_self_hashed_json,
)
from .continuation import import_parent_stage, preflight_stage
from .generation import crossed_score_stage, generation_stage
from .perception import build_perception_package, perception_analyze_stage
from .report import write_report
from .syncnet import fixed_video_stage, score_a_stage


def _empty_or_read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_self_hashed_json(path)
    except ProtocolError:
        return {}
    return value if isinstance(value, dict) else {}


def _stage_status(paths: config.RunPaths, stage: str, status: str, *, reason: str | None = None, traceback_text: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": stage, "status": status}
    if reason is not None:
        value["reason"] = reason
    if traceback_text is not None:
        value["traceback"] = traceback_text[-4000:]
    return write_self_hashed_json(paths.root / "stage_status" / f"{stage}.json", value)


def _safe(stage: str, paths: config.RunPaths, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = fn()
        _stage_status(paths, stage, str(result.get("status", "COMPLETE")))
        return result
    except ResourceWaitError as exc:
        message = str(exc)
        _stage_status(paths, stage, "RESOURCE_WAIT", reason=message, traceback_text=traceback.format_exc())
        return {"status": "RESOURCE_WAIT", "stage": stage, "failures": [{"error_type": "RESOURCE_WAIT", "reason": message}]}
    except DependencyBlockedError as exc:
        message = str(exc)
        _stage_status(paths, stage, "DEPENDENCY_BLOCKED", reason=message, traceback_text=traceback.format_exc())
        return {"status": "DEPENDENCY_BLOCKED", "stage": stage, "failures": [{"error_type": "DEPENDENCY_BLOCKED", "reason": message}]}
    except ProtocolError as exc:
        message = str(exc)
        status = "RESOURCE_WAIT" if "RESOURCE_WAIT" in message else "DEPENDENCY_BLOCKED" if "DEPENDENCY_BLOCKED" in message else "INCOMPLETE"
        _stage_status(paths, stage, status, reason=message, traceback_text=traceback.format_exc())
        return {"status": status, "stage": stage, "failures": [{"error_type": status, "reason": message}]}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        _stage_status(paths, stage, "INCOMPLETE", reason=str(exc), traceback_text=traceback.format_exc())
        return {"status": "INCOMPLETE", "stage": stage, "failures": [{"error_type": "UNHANDLED_STAGE_ERROR", "reason": str(exc)}]}
    except Exception as exc:  # pragma: no cover - final honest boundary  # noqa: BLE001
        _stage_status(paths, stage, "INCOMPLETE", reason=str(exc), traceback_text=traceback.format_exc())
        return {"status": "INCOMPLETE", "stage": stage, "failures": [{"error_type": "UNHANDLED_STAGE_ERROR", "reason": str(exc)}]}


def _audit(paths: config.RunPaths, run_id: str) -> dict[str, Any]:
    return audit_stage(paths, run_id=run_id)


def _audio(paths: config.RunPaths) -> dict[str, Any]:
    if paths.reuse_manifest.is_file():
        reuse = read_self_hashed_json(paths.reuse_manifest)
        if reuse.get("read_only") is True and any(Path(str(item.get("path"))).resolve() == paths.audio.resolve() for item in reuse.get("entries", [])):
            return read_self_hashed_json(paths.audio / "manifest.json")
    assets = read_self_hashed_json(paths.audit / "assets.json")
    return audio_stage(paths, assets)


def _fixed(paths: config.RunPaths) -> dict[str, Any]:
    assets = read_self_hashed_json(paths.audit / "assets.json")
    if paths.reuse_manifest.is_file():
        reuse = read_self_hashed_json(paths.reuse_manifest)
        if reuse.get("read_only") is True and any(Path(str(item.get("path"))).resolve() == paths.fixed_video.resolve() for item in reuse.get("entries", [])):
            return read_self_hashed_json(paths.fixed_video / "manifest.json")
    # P0 is refrozen at the last safe boundary before fixed-video/A scoring;
    # no implementation edits are allowed once this call has produced cells.
    refresh_code_snapshot(paths)
    fixed = fixed_video_stage(paths, assets)
    if fixed.get("status") != "COMPLETE":
        return fixed
    audio = read_self_hashed_json(paths.audio / "manifest.json")
    return score_a_stage(paths, assets, audio, fixed)


def _generate(paths: config.RunPaths, explicit_model_config: Path | None = None) -> dict[str, Any]:
    protocol = read_self_hashed_json(paths.protocol)
    if protocol.get("code_snapshot_refrozen_before_scoring") is not True:
        # This is the first new-B boundary.  It freezes the reviewed current
        # code without touching the read-only A parent.
        refresh_code_snapshot(paths)
    assets = read_self_hashed_json(paths.audit / "assets.json")
    audio = read_self_hashed_json(paths.audio / "manifest.json")
    model_config = explicit_model_config
    if model_config is None and paths.preflight.is_file():
        preflight = read_self_hashed_json(paths.preflight)
        if isinstance(preflight.get("model_config"), str):
            model_config = Path(str(preflight["model_config"]))
    return generation_stage(paths, assets, audio, model_config=model_config)


def _crossed(paths: config.RunPaths) -> dict[str, Any]:
    assets = read_self_hashed_json(paths.audit / "assets.json")
    audio = read_self_hashed_json(paths.audio / "manifest.json")
    generation = read_self_hashed_json(paths.generation / "manifest.json")
    return crossed_score_stage(paths, assets, audio, generation)


def _analyze(paths: config.RunPaths) -> dict[str, Any]:
    from .analysis import analyze_stage

    assets = read_self_hashed_json(paths.audit / "assets.json")
    audio = read_self_hashed_json(paths.audio / "manifest.json")
    fixed = read_self_hashed_json(paths.fixed_video / "manifest.json")
    a = read_self_hashed_json(paths.fixed_video / "a_manifest.json")
    crossed = _empty_or_read(paths.crossed / "manifest.json")
    if a:
        return analyze_stage(paths, assets, audio, fixed, crossed)
    raise ProtocolError("A scoring manifest is unavailable; analysis remains INCOMPLETE")


def _perception_pack(paths: config.RunPaths) -> dict[str, Any]:
    assets = read_self_hashed_json(paths.audit / "assets.json")
    audio = read_self_hashed_json(paths.audio / "manifest.json")
    generation = _empty_or_read(paths.generation / "manifest.json")
    return build_perception_package(paths, assets, audio, generation)


def _perception_analyze(paths: config.RunPaths) -> dict[str, Any]:
    return perception_analyze_stage(paths)


def _validation(paths: config.RunPaths) -> dict[str, Any]:
    from .validate import validate_stage

    return validate_stage(paths)


def _report(paths: config.RunPaths) -> dict[str, Any]:
    write_report(paths, validation_status=None)
    validation = _safe("validate", paths, lambda: _validation(paths))
    final = write_report(paths, validation_status=str(validation.get("status", "invalid")))
    # The report hash and final state changed after the first validation.
    validation = _safe("validate_final", paths, lambda: _validation(paths))
    final["validation_status"] = validation.get("status")
    return final


STAGES: dict[str, Callable[..., dict[str, Any]]] = {
    "audit": _audit,
    "audio": lambda paths, run_id=None: _audio(paths),
    "fixed-video": lambda paths, run_id=None: _fixed(paths),
    "generate": lambda paths, run_id=None: _generate(paths),
    "crossed-score": lambda paths, run_id=None: _crossed(paths),
    "analyze": lambda paths, run_id=None: _analyze(paths),
    "perception-pack": lambda paths, run_id=None: _perception_pack(paths),
    "perception-analyze": lambda paths, run_id=None: _perception_analyze(paths),
    "report": lambda paths, run_id=None: _report(paths),
}


def _aggregate_status(results: Mapping[str, Any]) -> str:
    statuses = [str(value.get("status", "INCOMPLETE")) for value in results.values() if isinstance(value, Mapping)]
    if any(status == "RESOURCE_WAIT" for status in statuses):
        return "RESOURCE_WAIT"
    if any(status == "DEPENDENCY_BLOCKED" for status in statuses):
        return "DEPENDENCY_BLOCKED"
    if any(status in {"INCOMPLETE", "PARTIAL", "INVALID", "IMPLEMENTATION_INCOMPLETE"} for status in statuses):
        return "INCOMPLETE"
    return "COMPLETE" if statuses and all(status == "COMPLETE" for status in statuses) else "INCOMPLETE"


def run(
    run_id: str,
    stage: str,
    *,
    parent_run: Path | None = None,
    completion_spec: Path | None = None,
    model_config: Path | None = None,
) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    paths.root.mkdir(parents=True, exist_ok=True)
    if stage == "import-parent":
        if parent_run is None or completion_spec is None:
            raise ValueError("import-parent requires --parent-run and --completion-spec")
        return _safe("import-parent", paths, lambda: import_parent_stage(paths, parent_run=parent_run, completion_spec=completion_spec, run_id=run_id))
    if stage == "preflight":
        if model_config is None:
            raise ValueError("preflight requires --model-config")
        return _safe("preflight", paths, lambda: preflight_stage(paths, model_config=model_config))
    if stage == "all":
        result: dict[str, Any] = {}
        for name in ("audit", "audio", "fixed-video", "generate", "crossed-score", "analyze", "perception-pack", "perception-analyze"):
            if name == "generate":
                result[name] = _safe(name, paths, lambda: _generate(paths, model_config))
            else:
                result[name] = _safe(name, paths, lambda name=name: STAGES[name](paths, run_id))
        result["report"] = _safe("report", paths, lambda: STAGES["report"](paths, run_id))
        result["status"] = _aggregate_status(result)
        return result
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if stage == "generate":
        return _safe(stage, paths, lambda: _generate(paths, model_config))
    return _safe(stage, paths, lambda: STAGES[stage](paths, run_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered TTS native-gain attribution experiment")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("import-parent", "preflight", "audit", "audio", "fixed-video", "generate", "crossed-score", "analyze", "perception-pack", "perception-analyze", "report", "all"), default="all")
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--completion-spec", type=Path)
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.run_id, args.stage, parent_run=args.parent_run, completion_spec=args.completion_spec, model_config=args.model_config)
    except (ProtocolError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    status = result.get("status", "INCOMPLETE") if isinstance(result, Mapping) else "INCOMPLETE"
    print(status)
    if status == "COMPLETE":
        return 0
    if status == "RESOURCE_WAIT":
        return 75
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
