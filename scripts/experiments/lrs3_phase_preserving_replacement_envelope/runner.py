from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .audio import run_stage01
from .common import verify_self_hashed_json
from .diagnostics import run_stage02
from .final import run_stage05_final
from .protocol import run_stage00
from .render import run_stage03
from .scoring import run_stage04
from .stats import run_stage05


def _stage(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def run(stage: str) -> dict[str, Any]:
    if stage == "00_protocol":
        return run_stage00()
    cohort = _stage(config.STAGES["00_protocol"] / "cohort.json")
    protocol = _stage(config.STAGES["00_protocol"] / "protocol.json")
    if stage == "01_candidates":
        return run_stage01(cohort, protocol, config.STAGES["01_candidates"])
    audio = _stage(config.STAGES["01_candidates"] / "audio_manifest.json")
    if stage == "02_audio_diagnostics":
        return run_stage02(cohort, audio, config.STAGES["02_audio_diagnostics"])
    diagnostics = _stage(config.STAGES["02_audio_diagnostics"] / "diagnostics.json")
    if stage == "03_videos":
        return run_stage03(cohort, audio, config.STAGES["03_videos"])
    videos = _stage(config.STAGES["03_videos"] / "videos_manifest.json")
    if stage == "04_scores":
        return run_stage04(cohort, audio, diagnostics, videos, config.STAGES["04_scores"])
    scores = _stage(config.STAGES["04_scores"] / "scores_manifest.json")
    if stage == "05_analysis":
        return run_stage05(cohort, diagnostics, scores, config.STAGES["05_final"])
    analysis = _stage(config.STAGES["05_final"] / "analysis.json")
    if stage == "05_final":
        return run_stage05_final(protocol, cohort, diagnostics, scores, analysis, config.STAGES["05_final"])
    if stage == "all":
        run_stage01(cohort, protocol, config.STAGES["01_candidates"])
        audio = _stage(config.STAGES["01_candidates"] / "audio_manifest.json")
        run_stage02(cohort, audio, config.STAGES["02_audio_diagnostics"])
        diagnostics = _stage(config.STAGES["02_audio_diagnostics"] / "diagnostics.json")
        run_stage03(cohort, audio, config.STAGES["03_videos"])
        videos = _stage(config.STAGES["03_videos"] / "videos_manifest.json")
        run_stage04(cohort, audio, diagnostics, videos, config.STAGES["04_scores"])
        scores = _stage(config.STAGES["04_scores"] / "scores_manifest.json")
        run_stage05(cohort, diagnostics, scores, config.STAGES["05_final"])
        analysis = _stage(config.STAGES["05_final"] / "analysis.json")
        return run_stage05_final(protocol, cohort, diagnostics, scores, analysis, config.STAGES["05_final"])
    raise ValueError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("00_protocol", "01_candidates", "02_audio_diagnostics", "03_videos", "04_scores", "05_analysis", "05_final", "all"), default="00_protocol")
    args = parser.parse_args(argv)
    print(run(args.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
