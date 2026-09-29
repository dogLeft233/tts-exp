from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .audio import run_stage01
from .common import verify_self_hashed_json
from .diagnostics import run_stage01_diagnostics
from .final import run_stage04_final, write_blocked_terminal
from .protocol import run_stage00
from .render import run_stage02
from .scoring import run_stage03
from .stats import run_stage04_analysis


def _stage(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _run(stage: str) -> dict[str, Any]:
    if stage == "00_protocol":
        return run_stage00()
    cohort = _stage(config.STAGES["00_protocol"] / "cohort.json")
    protocol = _stage(config.STAGES["00_protocol"] / "protocol.json")
    if stage == "01_audio":
        audio = run_stage01(cohort, protocol, config.STAGES["01_audio"])
        diagnostics = run_stage01_diagnostics(cohort, audio, config.STAGES["01_audio"])
        return {"audio": audio, "diagnostics": diagnostics}
    audio = _stage(config.STAGES["01_audio"] / "audio_manifest.json")
    diagnostics = _stage(config.STAGES["01_audio"] / "diagnostics.json")
    if stage == "02_videos":
        return run_stage02(cohort, audio, config.STAGES["02_videos"])
    videos = _stage(config.STAGES["02_videos"] / "videos_manifest.json")
    if stage == "03_scores":
        return run_stage03(cohort, audio, diagnostics, videos, config.STAGES["03_scores"])
    scores = _stage(config.STAGES["03_scores"] / "scores_manifest.json")
    if stage == "04_analysis":
        return run_stage04_analysis(cohort, diagnostics, scores, config.STAGES["04_final"])
    analysis = _stage(config.STAGES["04_final"] / "analysis.json")
    if stage == "04_final":
        return run_stage04_final(protocol, cohort, diagnostics, scores, analysis, config.STAGES["04_final"])
    if stage == "all":
        audio = run_stage01(cohort, protocol, config.STAGES["01_audio"])
        diagnostics = run_stage01_diagnostics(cohort, audio, config.STAGES["01_audio"])
        videos = run_stage02(cohort, audio, config.STAGES["02_videos"])
        scores = run_stage03(cohort, audio, diagnostics, videos, config.STAGES["03_scores"])
        analysis = run_stage04_analysis(cohort, diagnostics, scores, config.STAGES["04_final"])
        return run_stage04_final(protocol, cohort, diagnostics, scores, analysis, config.STAGES["04_final"])
    raise ValueError(f"unknown stage: {stage}")


def run(stage: str) -> dict[str, Any]:
    try:
        return _run(stage)
    except Exception as exc:
        write_blocked_terminal(exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("00_protocol", "01_audio", "02_videos", "03_scores", "04_analysis", "04_final", "all"), default="00_protocol")
    args = parser.parse_args(argv)
    print(run(args.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
