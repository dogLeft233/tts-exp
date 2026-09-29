from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .audit import run_audit
from .common import assert_run_root_compatible, verify_self_hashed_json
from .control import materialize_audio
from .final import finalize, write_blocked_terminal
from .protocol import load_cohort, load_history, load_protocol, run_protocol
from .render import render_videos
from .scoring import run_scores
from .stats import analyze_scores


def _artifact(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _write_blocked_if_unterminated(paths: config.RunPaths, error: Exception, failed_stage: str) -> None:
    """Record a new failure without replacing an existing terminal artifact."""

    if (paths.final / "final.json").exists():
        return
    write_blocked_terminal(paths, error, failed_stage)


def _run_stage(stage: str, paths: config.RunPaths) -> dict[str, Any]:
    if stage == "audit":
        return run_audit(paths)
    audit = _artifact(paths.audit / "audit.json")
    if stage == "protocol":
        return run_protocol(paths, audit, load_history())
    protocol = load_protocol(paths)
    cohort = load_cohort(paths)
    if stage == "audio":
        return materialize_audio(cohort, protocol, paths.audio)
    audio = _artifact(paths.audio / "audio_manifest.json")
    if stage == "videos":
        history = load_history()
        return render_videos(cohort, protocol, audio, history["videos"], paths)
    videos = _artifact(paths.videos / "videos_manifest.json")
    if stage == "scores":
        return run_scores(cohort, protocol, audio, videos, paths)
    scores = _artifact(paths.scores / "scores_manifest.json")
    if stage == "analysis":
        return analyze_scores(cohort, protocol, scores, paths.final)
    analysis = _artifact(paths.final / "analysis.json")
    if stage == "final":
        audit = _artifact(paths.audit / "audit.json")
        return finalize(protocol, cohort, audio, videos, scores, analysis, paths, audit)
    raise ValueError(f"unknown stage: {stage}")


def run(run_id: str, stage: str = "all") -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    assert_run_root_compatible(paths.root)
    stages = ("audit", "protocol", "audio", "videos", "scores", "analysis", "final")
    if stage == "all":
        try:
            result: dict[str, Any] = {}
            for current in stages:
                result = _run_stage(current, paths)
            return result
        except Exception as exc:
            _write_blocked_if_unterminated(paths, exc, "all")
            raise
    if stage not in stages:
        raise ValueError(f"unknown stage: {stage}")
    try:
        return _run_stage(stage, paths)
    except Exception as exc:
        _write_blocked_if_unterminated(paths, exc, stage)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LRS3 local timing control calibration")
    parser.add_argument("--run-id", required=True, help="safe suffix for the new run root")
    parser.add_argument("--stage", choices=("audit", "protocol", "audio", "videos", "scores", "analysis", "final", "all"), default="all")
    args = parser.parse_args(argv)
    result = run(args.run_id, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
