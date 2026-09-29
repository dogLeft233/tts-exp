from __future__ import annotations

import argparse
import json

from .analysis import run_analysis
from .candidates import run_candidates
from .diagonal import run_diagonal
from .final import run_final
from .protocol import run_stage00
from .replacement import run_replacement


def main(argv: list[str] | None = None) -> int:
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
