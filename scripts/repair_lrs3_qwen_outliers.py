#!/usr/bin/env python3
"""Repair known Qwen stop failures in the 100-sample LRS3 comparison run.

The default seed used by the comparison protocol reaches the 512-token limit
for two clips from one source group.  This script regenerates only those clips
with a deterministic alternate seed, keeps the same model and token cap, and
updates the existing Qwen manifest in place.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_lrs3_local_tts import (  # noqa: E402
    DEFAULT_COHORT,
    DEFAULT_OUTPUT_ROOT,
    QWEN_MAX_NEW_TOKENS,
    load_qwen_provider,
    load_selection,
    resolve_repo_path,
    synthesize_one,
    write_json,
)


DEFAULT_SAMPLE_IDS = (
    "lrs3_6tSlMoMNSlY_00002",
    "lrs3_6tSlMoMNSlY_00003",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        default=list(DEFAULT_SAMPLE_IDS),
        help="sample to repair; may be repeated (defaults to the two observed stop failures)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=303000,
        help="alternate seed base; final seed is seed_base + selection_index",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cohort_path = resolve_repo_path(args.cohort)
    output_root = resolve_repo_path(args.output_root)
    records = load_selection(cohort_path, 100)
    wanted = set(args.sample_ids)
    selected = [record for record in records if str(record["sample_id"]) in wanted]
    found = {str(record["sample_id"]) for record in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"sample IDs not found in the first 100 cohort records: {missing}")

    backend_dir = output_root / "01_qwen_local"
    manifest_path = backend_dir / "tts_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Qwen manifest not found: {manifest_path}")
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("sample_id")): row
        for row in payload.get("results", [])
        if isinstance(row, dict)
    }

    print(f"repairing {len(selected)} Qwen rows with max_new_tokens={QWEN_MAX_NEW_TOKENS}", flush=True)
    provider = load_qwen_provider()
    for record in selected:
        sample_id = str(record["sample_id"])
        seed = int(args.seed_base) + int(record["selection_index"])
        raw_path = backend_dir / "raw" / f"{sample_id}.wav"
        canonical_path = backend_dir / "canonical_16k" / f"{sample_id}.wav"
        row = synthesize_one("qwen", provider, record, raw_path, canonical_path, seed)
        if float(row["canonical_duration_s"]) >= 30.0:
            raise RuntimeError(
                f"alternate seed did not repair {sample_id}: "
                f"duration={row['canonical_duration_s']}s"
            )
        row["repair"] = {
            "reason": "default_seed_reached_generation_limit",
            "seed_base": int(args.seed_base),
            "generation_protocol": {"max_new_tokens": QWEN_MAX_NEW_TOKENS},
        }
        rows[sample_id] = row
        print(f"{sample_id}: repaired duration={row['canonical_duration_s']:.2f}s seed={seed}", flush=True)

    ordered = [rows[str(record["sample_id"])] for record in records]
    payload["results"] = ordered
    payload["completed_sample_count"] = sum(row.get("status") == "ok" for row in ordered)
    payload["failures"] = [row for row in ordered if row.get("status") != "ok"]
    payload["failed_sample_count"] = len(payload["failures"])
    payload["quality_control"] = {
        "generation_limit_outlier_repaired": sorted(wanted),
        "qwen_max_new_tokens": QWEN_MAX_NEW_TOKENS,
    }
    payload.setdefault("repair_history", []).append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "sample_ids": sorted(wanted),
            "seed_base": int(args.seed_base),
            "reason": "default_seed_reached_generation_limit",
        }
    )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "complete" if not payload["failures"] else "partial_failure"
    write_json(manifest_path, payload)
    print(
        f"updated {manifest_path}: {payload['completed_sample_count']}/"
        f"{len(ordered)} ok, {payload['failed_sample_count']} failed",
        flush=True,
    )
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
