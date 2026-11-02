"""Freeze attempt02 full-natural-audio rows from the qualified attempt01 copies."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.cross_tfg_audio_head_20261003.build_fullcontext_fit_plan_v2_attempt02 import build_plan


RUN_ROOT = ROOT / "runs/cross_tfg_audio_head_20261003"
ATTEMPT01_PLAN = RUN_ROOT / "fit_v2_fullcontext_attempt01/plan.json"
ATTEMPT01_MANIFEST = RUN_ROOT / "fit_v2_full_audio_attempt01/full_audio_manifest.json"
BASE_PLAN = RUN_ROOT / "fit_subset_attempt02/historical_fit_subset_plan.json"
OUT = RUN_ROOT / "fit_v2_fullcontext_attempt02"
REMOTE_RUN = "/dev/shm/cross_tfg_audio_head_20261003/fit_v2_fullcontext_attempt02"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite fresh attempt02 plan root: {OUT}")
    parent_plan_raw = ATTEMPT01_PLAN.read_bytes()
    parent_plan = json.loads(parent_plan_raw.decode("utf-8"))
    parent_manifest_raw = ATTEMPT01_MANIFEST.read_bytes()
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    if parent_plan.get("plan_payload_sha256") != canonical_sha(
        {key: value for key, value in parent_plan.items() if key != "plan_payload_sha256"}
    ):
        raise ValueError("attempt01 plan payload self-hash mismatch")
    if parent_manifest.get("manifest_sha256") != canonical_sha(
        {key: value for key, value in parent_manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("attempt01 full-audio manifest payload self-hash mismatch")
    if len(parent_plan.get("rows", [])) != 8 or len(parent_manifest.get("rows", [])) != 8:
        raise ValueError("attempt01 provenance does not contain exactly eight fit rows")
    if parent_plan.get("full_context_fit_v2", {}).get("fit_rows") != 8:
        raise ValueError("attempt01 full-context plan fit-row count changed")

    local_by_row = {str(row["row_id"]): row for row in parent_manifest["rows"]}
    if len(local_by_row) != 8:
        raise ValueError("attempt01 audio transfer rows are missing or duplicated")
    rows = []
    for parent_row in parent_manifest["rows"]:
        row = copy.deepcopy(parent_row)
        row_id = str(row["row_id"])
        local_path = Path(str(row["transfer_local_path"])).resolve(strict=True)
        actual_sha = sha256_bytes(local_path.read_bytes())
        if actual_sha != str(row.get("file_sha256", "")).lower():
            raise ValueError(f"attempt01 full-audio local copy changed: {row_id}")
        row["remote_path"] = f"{REMOTE_RUN}/input/full_audio/{local_path.name}"
        row["transfer_local_path"] = str(local_path)
        row["parent_attempt01_remote_path"] = parent_row.get("remote_path")
        row["parent_attempt01_plan_row_id"] = row_id
        rows.append(row)
    transfer_manifest: dict[str, Any] = {
        "schema": "cross_tfg_audio_head_full_audio_transfer_manifest_attempt02_v1",
        "status": "READY_FOR_ROOT_REVIEW",
        "run_id": "cross_tfg_audio_head_fullcontext_fit_20261003_attempt02",
        "source_plan_path": str(ATTEMPT01_PLAN.resolve()),
        "source_plan_raw_sha256": sha256_bytes(parent_plan_raw),
        "source_plan_payload_sha256": parent_plan["plan_payload_sha256"],
        "source_manifest_path": str(ATTEMPT01_MANIFEST.resolve()),
        "source_manifest_raw_sha256": sha256_bytes(parent_manifest_raw),
        "source_manifest_payload_sha256": parent_manifest["manifest_sha256"],
        "rows": rows,
        "confirmation_audio_opened": False,
        "development_audio_opened": False,
        "effects_or_scores_read": False,
    }
    transfer_manifest["manifest_sha256"] = canonical_sha(transfer_manifest)
    OUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUT / "full_audio_transfer_manifest.json"
    manifest_raw = json.dumps(transfer_manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    manifest_path.write_bytes(manifest_raw)
    base_path = BASE_PLAN.resolve(strict=True)
    plan_receipt = build_plan(
        base_path=base_path,
        full_audio_manifest=manifest_path.resolve(strict=True),
        output_path=OUT / "plan.json",
        cache_root=f"{REMOTE_RUN}/audio_embedding_cache",
        fit_seed=20261003,
        epochs=12,
        steps=96,
    )
    result = {
        "status": "FIT_V2_ATTEMPT02_PLAN_READY_FOR_REVIEW",
        "plan_path": str((OUT / "plan.json").resolve()),
        "plan_receipt_path": str((OUT / "fullcontext_fit_plan_build_receipt.json").resolve()),
        "transfer_manifest_path": str(manifest_path.resolve()),
        "transfer_manifest_bytes": len(manifest_raw),
        "transfer_manifest_raw_sha256": sha256_bytes(manifest_raw),
        "transfer_manifest_payload_sha256": transfer_manifest["manifest_sha256"],
        "attempt01_plan_raw_sha256": sha256_bytes(parent_plan_raw),
        "attempt01_transfer_manifest_raw_sha256": sha256_bytes(parent_manifest_raw),
        "plan_build_receipt": plan_receipt,
        "forward_performed": False,
        "fit_steps_performed": 0,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "effects_or_scores_read": False,
    }
    receipt_path = OUT / "attempt02_preparation_receipt.json"
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
