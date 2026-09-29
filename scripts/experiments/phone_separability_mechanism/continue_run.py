"""Continue non-extraction stages on a frozen run after a code correction.

The main runner intentionally rejects ``--resume`` when its protocol hash
changes. This command is the explicit, auditable escape hatch for a run that
already contains immutable inventory/features and has been strict-rescored.
It records the old protocol hash and the current implementation hash before
writing mechanism, construction, and report artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .run import (
    _code_hashes,
    _load_config,
    _write_json,
    stage_construct,
    stage_mechanisms,
    stage_report,
    stage_train,
)
from .candidate import run_candidates


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def continue_run(run_dir: str | Path, *, stage: str = "all", smoke: bool = False) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = _read_json(protocol_path)
    config_path = Path(str(protocol["config_path"])).resolve()
    config = _load_config(config_path)
    if not (root / "03_atlas" / "strict_summary.json").is_file():
        raise ValueError("strict atlas rescore is required before continuation")
    if stage in {"candidate", "construct", "train", "report", "all"} and not (root / "04_mechanisms" / "mechanism_evidence.json").is_file():
        stage_mechanisms(config, root, smoke=smoke)
    if stage in {"candidate", "construct", "train", "report", "all"} and not (root / "05_construction" / "construction.jsonl").is_file():
        stage_construct(config, root, smoke=smoke)
    if stage in {"candidate", "train", "report", "all"} and not (root / "07_candidates" / "summary.json").is_file():
        run_candidates(root)
    decision = None
    training_path = root / "06_training" / "decision.json"
    training_needs_refresh = not training_path.is_file()
    if (root / "07_candidates" / "summary.json").is_file() and training_path.is_file():
        training_needs_refresh = str(_read_json(training_path).get("status", "")) == "SKIPPED_CONDITIONAL_GATE"
    if stage in {"report", "all"} and training_needs_refresh:
        stage_train(config, root, smoke=smoke)
    decision = stage_report(config, root, smoke=smoke) if stage in {"report", "all"} else decision
    continuation = {
        "schema_version": 1,
        "status": "COMPLETE",
        "stage": stage,
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "current_code_sha256": _code_hashes(config_path),
        "strict_summary_sha256": hashlib.sha256((root / "03_atlas" / "strict_summary.json").read_bytes()).hexdigest(),
        "decision": decision,
    }
    _write_json(root / "07_continuation" / "continuation.json", continuation)
    return continuation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=("mechanisms", "construct", "candidate", "train", "report", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = continue_run(args.run_dir, stage=args.stage, smoke=bool(args.smoke))
    print(json.dumps({"status": result["status"], "stage": result["stage"], "run_dir": str(Path(args.run_dir).resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["continue_run", "main"]
