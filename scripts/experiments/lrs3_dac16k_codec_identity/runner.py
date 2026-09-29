from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import config
from .audio import run_stage01
from .common import verify_self_hashed_json
from .fidelity import run_stage02
from .final import run_stage06
from .matrix import run_stage04
from .protocol import run_stage00
from .render import run_stage03
from .stats import run_stage05


def _stage(path: Path) -> dict:
    return verify_self_hashed_json(path)


def run(stage: str, device: str) -> dict:
    if stage == "00_protocol":
        return run_stage00()
    cohort = _stage(config.STAGES["00_protocol"] / "cohort.json")
    protocol = _stage(config.STAGES["00_protocol"] / "protocol.json")
    if stage == "01_audio":
        return run_stage01(cohort, protocol, config.STAGES["01_audio"], device)
    audio = _stage(config.STAGES["01_audio"] / "audio_manifest.json")
    if stage == "02_fidelity":
        return run_stage02(cohort, audio, config.STAGES["02_fidelity"])
    fidelity = _stage(config.STAGES["02_fidelity"] / "fidelity.json")
    if stage == "03_videos":
        return run_stage03(cohort, audio, config.STAGES["03_videos"])
    videos = _stage(config.STAGES["03_videos"] / "videos_manifest.json")
    if stage == "04_matrix":
        return run_stage04(cohort, audio, videos, config.STAGES["04_matrix"])
    matrix = _stage(config.STAGES["04_matrix"] / "matrix_manifest.json")
    if stage == "05_analysis":
        return run_stage05(cohort, fidelity, matrix, config.STAGES["05_analysis"])
    analysis = _stage(config.STAGES["05_analysis"] / "analysis.json")
    if stage == "06_final":
        return run_stage06(protocol, cohort, fidelity, matrix, analysis, config.STAGES["06_final"])
    if stage == "all":
        run_stage01(cohort, protocol, config.STAGES["01_audio"], device)
        audio = _stage(config.STAGES["01_audio"] / "audio_manifest.json")
        run_stage02(cohort, audio, config.STAGES["02_fidelity"])
        fidelity = _stage(config.STAGES["02_fidelity"] / "fidelity.json")
        run_stage03(cohort, audio, config.STAGES["03_videos"])
        videos = _stage(config.STAGES["03_videos"] / "videos_manifest.json")
        run_stage04(cohort, audio, videos, config.STAGES["04_matrix"])
        matrix = _stage(config.STAGES["04_matrix"] / "matrix_manifest.json")
        run_stage05(cohort, fidelity, matrix, config.STAGES["05_analysis"])
        analysis = _stage(config.STAGES["05_analysis"] / "analysis.json")
        return run_stage06(protocol, cohort, fidelity, matrix, analysis, config.STAGES["06_final"])
    raise ValueError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("00_protocol", "01_audio", "02_fidelity", "03_videos", "04_matrix", "05_analysis", "06_final", "all"), default="00_protocol")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(args.stage, args.device)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
