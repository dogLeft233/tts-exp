from __future__ import annotations

import argparse
import json

from . import config

config.PROTOCOL_ID = "lrs3_mfa_dtw_short_phone_support_20260904"
config.RUN_ROOT = config.REPO / "runs/lrs3_mfa_dtw_replacement_short_phone_20260904"
config.STAGE00 = config.RUN_ROOT / "00_protocol"
config.STAGE01 = config.RUN_ROOT / "01_candidates"
config.STAGE02 = config.RUN_ROOT / "02_diagonal"
config.STAGE03 = config.RUN_ROOT / "03_diagonal_analysis"
config.STAGE04 = config.RUN_ROOT / "04_replacement"
config.STAGE05 = config.RUN_ROOT / "05_final"
config.DTW_FRAME_OWNERSHIP_POLICY = "nominal_half_stride_overlap_shared_v1"


def main(argv: list[str] | None = None) -> int:
    from .analysis import run_analysis
    from .candidates import run_candidates
    from .diagonal import run_diagonal
    from .final import run_final
    from .protocol import run_stage00
    from .replacement import run_replacement

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage00", "candidates", "diagonal", "analysis", "replacement", "final"))
    args = parser.parse_args(argv)
    handlers = {
        "stage00": run_stage00,
        "candidates": run_candidates,
        "diagonal": run_diagonal,
        "analysis": run_analysis,
        "replacement": run_replacement,
        "final": run_final,
    }
    result = handlers[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("engineering_decision") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
