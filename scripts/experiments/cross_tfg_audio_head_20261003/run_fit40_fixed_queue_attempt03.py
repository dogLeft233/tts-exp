#!/usr/bin/env python3
"""Verify or initialize the historical Fit40 queue baseline; never launches a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.cross_tfg_audio_head_20261003.fit40_fixed_queue_controller_attempt03 import (
    FixedFitQueueController,
    _load_historical_prefix_seed,
    linux_process_start_ticks,
    load_pinned_bundles,
    sha256,
)


def verify_only(repo_root: Path, historical_index: Path, expected_index_sha256: str,
                state_dir: Path) -> dict[str, Any]:
    """Read and validate all source pins/index data without creating queue state."""
    bundles = load_pinned_bundles(repo_root)
    seed = _load_historical_prefix_seed(historical_index, expected_index_sha256,
                                        {item.spec.model_id: item for item in bundles})
    return {
        "schema": "fit40_fixed_queue_preflight_attempt03_v1",
        "status": "PASS_VERIFY_ONLY_NO_LAUNCH",
        "queue_state_dir": str(state_dir),
        "queue_state_dir_exists": state_dir.exists(),
        "models": [
            {"model_id": item.spec.model_id,
             "plan_bytes": item.spec.plan.bytes,
             "plan_sha256": item.spec.plan.sha256,
             "planned_rows_bytes": item.spec.planned_rows.bytes,
             "planned_rows_sha256": item.spec.planned_rows.sha256,
             "runtime_binding_bytes": item.spec.runtime_binding.bytes,
             "runtime_binding_sha256": item.spec.runtime_binding.sha256,
             "next_start": seed.next_starts[item.spec.model_id]}
            for item in bundles
        ],
        "historical_index": {
            "path": str(historical_index.resolve(strict=True)),
            "bytes": seed.index_bytes,
            "sha256": seed.index_sha256,
        },
        "latest_terminal_model": seed.latest_terminal_model,
        "launches_models": False,
    }


def initialize_history(repo_root: Path, historical_index: Path,
                       expected_index_sha256: str, state_dir: Path) -> dict[str, Any]:
    """Create one exclusive history-import receipt; this mode also never launches."""
    bundles = load_pinned_bundles(repo_root)
    controller = FixedFitQueueController(
        bundles, state_dir,
        historical_prefix_index=historical_index,
        historical_prefix_index_sha256=expected_index_sha256,
    )
    ticks = linux_process_start_ticks(os.getpid())
    receipt_path = controller.create_historical_prefix_import(
        owner_pid=os.getpid(), owner_start_ticks=ticks,
    )
    raw = receipt_path.read_bytes()
    return {
        "schema": "fit40_fixed_queue_preflight_attempt03_v1",
        "status": "PASS_HISTORY_INITIALIZED_NO_LAUNCH",
        "queue_state_dir": str(state_dir),
        "history_import_receipt": {
            "path": str(receipt_path.resolve()), "bytes": len(raw), "sha256": sha256(raw),
        },
        "launches_models": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-only", "initialize-history"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "verify-only":
        result = verify_only(args.repo_root, args.historical_index,
                             args.expected_index_sha256, args.state_dir)
    else:
        result = initialize_history(args.repo_root, args.historical_index,
                                    args.expected_index_sha256, args.state_dir)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
