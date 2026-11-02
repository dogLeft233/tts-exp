#!/usr/bin/env python3
"""Seal fit-only shared-profile selection from a consumer-verified StageA run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audio_enhancement_eval.shared_profile_selection_v3 import (  # noqa: E402
    ALL_METHODS,
    LAG_OFFSETS,
    select_shared_profiles,
)


SELECTOR_SHA256 = "1e329d4b1d2f5918230e5cdcf2d3c3c91256daf7a4536e7a7424ad33ff609a6d"
SCHEMA = "cross_tfg_shared_profile_fit_selection_artifact_v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            output.append(row)
    return output


def verify_payload_seal(value: dict[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    actual = sha_bytes(canonical({key: item for key, item in value.items() if key != field}))
    if expected != actual:
        raise ValueError(f"{label} payload seal mismatch: {expected!r} != {actual}")


def load_and_verify_consumer(out_dir: Path, plan: dict[str, Any], bindings: dict[str, str], result_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_path = out_dir / "audit.json"
    manifest_path = out_dir / "consumer_manifest.json"
    audit = read_json(audit_path)
    consumer_manifest = read_json(manifest_path)
    verify_payload_seal(consumer_manifest, "manifest_sha256", "consumer manifest")

    if audit.get("run_id") != plan.get("run_id") or consumer_manifest.get("run_id") != plan.get("run_id"):
        raise ValueError("consumer output run_id does not match plan")
    for audit_key, binding_key in (
        ("plan_raw_sha256", "plan_raw_sha256"),
        ("results_raw_sha256", "results_raw_sha256"),
        ("result_manifest_raw_sha256", "result_manifest_raw_sha256"),
        ("generation_repeat_qc_raw_sha256", "generation_repeat_qc_raw_sha256"),
    ):
        if audit.get(audit_key) != bindings[binding_key]:
            raise ValueError(f"consumer audit input seal mismatch: {audit_key}")
    if audit.get("issues") != []:
        raise ValueError("consumer audit reports integrity issues")
    if int(audit.get("result_rows", -1)) != result_count:
        raise ValueError("consumer audit result row count mismatch")
    if int(audit.get("verified_scored_row_count", -1)) != int(audit.get("status_counts", {}).get("PASS", -2)):
        raise ValueError("consumer verified-scored count differs from PASS row count")
    if consumer_manifest.get("status") != audit.get("status"):
        raise ValueError("consumer manifest and audit statuses differ")

    files = consumer_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("consumer manifest has no output file inventory")
    seen = set()
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe consumer output path: {relative}")
        if str(relative) in seen:
            raise ValueError(f"duplicate consumer output path: {relative}")
        seen.add(str(relative))
        path = out_dir / relative
        if path.stat().st_size != int(item["bytes"]) or sha_file(path) != item["sha256"]:
            raise ValueError(f"consumer output file seal mismatch: {relative}")
    if "audit.json" not in seen:
        raise ValueError("consumer manifest omits audit.json")
    return audit, consumer_manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve()
    rows_path = Path(args.results).resolve()
    result_manifest_path = Path(args.result_manifest).resolve()
    repeat_path = Path(args.generation_repeat_qc).resolve()
    consumer_dir = Path(args.consumer_out).resolve()
    out_path = Path(args.output).resolve()

    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if not isinstance(plan, dict):
        raise ValueError("plan JSON root is not an object")
    verify_payload_seal(plan, "plan_sha256", "plan")

    selector_path = ROOT / "scripts/audio_enhancement_eval/shared_profile_selection_v3.py"
    selector_sha = sha_file(selector_path)
    if selector_sha != SELECTOR_SHA256:
        raise ValueError(f"selector source SHA mismatch: {selector_sha}")
    selector_pins = {
        plan.get("execution", {}).get("selector_source_sha256"),
        plan.get("analysis", {}).get("selection", {}).get("selector_source_sha256"),
    }
    if selector_pins != {SELECTOR_SHA256}:
        raise ValueError("plan does not pin this exact selector source")

    plan_rows = plan.get("rows")
    methods = [str(row["method_id"]) for row in plan.get("methods", [])]
    models = [str(row["model_id"]) for row in plan.get("models", [])]
    if len(plan_rows or []) != 360 or len(methods) != 15 or len(models) != 3:
        raise ValueError("plan does not match the frozen 360-row, 15-method, 3-model fit grid")
    if len(set(methods)) != len(methods) or set(methods) != set(ALL_METHODS):
        raise ValueError("plan method roster differs from selector's frozen family")
    if plan.get("dev_speaker_ids") or plan.get("confirm_speaker_ids"):
        raise ValueError("selection input plan contains non-fit speaker rosters")
    if any(row.get("split") != "fit" for row in plan_rows):
        raise ValueError("selection input plan contains non-fit rows")
    if tuple(int(x) for x in plan["protocol"]["syncnet"]["lag_offsets_frames"]) != LAG_OFFSETS:
        raise ValueError("plan lag axis does not match the frozen -15..+15 selector contract")

    results = read_jsonl(rows_path)
    if len(results) != len(plan_rows):
        raise ValueError(f"planned/result rows differ: {len(plan_rows)} != {len(results)}")
    for position, (planned, actual) in enumerate(zip(plan_rows, results)):
        for field in ("row_id", "split", "model_id", "speaker_id", "sample_id", "face_id", "seed", "method_id"):
            if planned.get(field) != actual.get(field):
                raise ValueError(f"row identity mismatch at position {position}: {field}")

    result_manifest_raw = result_manifest_path.read_bytes()
    result_manifest = json.loads(result_manifest_raw)
    verify_payload_seal(result_manifest, "manifest_sha256", "result manifest")
    results_sha = sha_bytes(rows_path.read_bytes())
    repeat_sha = sha_bytes(repeat_path.read_bytes())
    raw_bindings = {
        "plan_raw_sha256": sha_bytes(plan_raw),
        "results_raw_sha256": results_sha,
        "result_manifest_raw_sha256": sha_bytes(result_manifest_raw),
        "generation_repeat_qc_raw_sha256": repeat_sha,
    }
    if result_manifest.get("run_id") != plan.get("run_id"):
        raise ValueError("result manifest run_id mismatch")
    if result_manifest.get("plan_raw_sha256") != raw_bindings["plan_raw_sha256"]:
        raise ValueError("result manifest plan raw SHA mismatch")
    if result_manifest.get("results_jsonl_sha256") != results_sha:
        raise ValueError("result manifest results raw SHA mismatch")
    if result_manifest.get("generation_repeat_qc_jsonl_sha256") != repeat_sha:
        raise ValueError("result manifest repeat-QC raw SHA mismatch")
    if int(result_manifest.get("planned_rows", -1)) != len(plan_rows) or int(result_manifest.get("result_rows", -1)) != len(results):
        raise ValueError("result manifest row totals mismatch")
    if result_manifest.get("status_counts") != {
        key: sum(row.get("status") == key for row in results) for key in ("PASS", "FAILED", "INCOMPLETE")
    }:
        raise ValueError("result manifest status counts mismatch")

    audit, consumer_manifest = load_and_verify_consumer(consumer_dir, plan, raw_bindings, len(results))

    expected_contexts = []
    context_keys = set()
    for row in plan_rows:
        key = (row["split"], row["model_id"], row["speaker_id"], row["sample_id"], int(row["seed"]))
        if key in context_keys:
            continue
        context_keys.add(key)
        expected_contexts.append({
            "split": row["split"], "model_id": row["model_id"], "speaker_id": row["speaker_id"],
            "sample_id": row["sample_id"], "seed": int(row["seed"]),
        })

    selection_rows = []
    for planned, actual in zip(plan_rows, results):
        row = {
            "row_id": actual["row_id"], "split": actual["split"], "model_id": actual["model_id"],
            "speaker_id": actual["speaker_id"], "sample_id": actual["sample_id"],
            "seed": int(actual["seed"]), "method_id": actual["method_id"],
            "status": actual.get("status"), "failure_reason": actual.get("failure_reason"),
            "source_status": actual.get("status"),
        }
        if actual.get("status") == "PASS":
            score = actual.get("score", {})
            support = score.get("support_by_support", {}).get("full", {})
            curve = score.get("lag_curve_by_support", {}).get("full")
            metrics = score.get("metrics_by_support", {}).get("full")
            if int(support.get("query_count", 0)) < 1 or not isinstance(curve, list) or not isinstance(metrics, dict):
                row["status"] = "INCOMPLETE"
                row["failure_reason"] = "FULL method-common support or summary unavailable"
            else:
                row["support_sha256"] = support.get("support_mask_sha256")
                row["lag_offsets_frames"] = list(LAG_OFFSETS)
                row["lag_curve_f32"] = curve
                row["metrics"] = metrics
        selection_rows.append(row)

    selected = select_shared_profiles(
        expected_contexts=expected_contexts,
        predeclared_fit_roster=plan["fit_roster"],
        result_rows=selection_rows,
        model_roster=models,
        shortlist_count=int(plan["analysis"]["selection"]["shortlist_count_max"]),
    )
    envelope: dict[str, Any] = {
        "schema": SCHEMA,
        "status": selected["shortlist_status"],
        "run_id": plan["run_id"],
        "bindings": {
            **raw_bindings,
            "plan_payload_sha256": plan["plan_sha256"],
            "result_manifest_payload_sha256": result_manifest["manifest_sha256"],
            "consumer_audit_raw_sha256": sha_file(consumer_dir / "audit.json"),
            "consumer_manifest_raw_sha256": sha_file(consumer_dir / "consumer_manifest.json"),
            "consumer_manifest_payload_sha256": consumer_manifest["manifest_sha256"],
            "consumer_status": audit["status"],
            "consumer_issue_count": len(audit["issues"]),
            "selector_source_sha256": selector_sha,
            "adapter_source_sha256": sha_file(Path(__file__).resolve()),
        },
        "selection": selected,
    }
    envelope["artifact_sha256"] = sha_bytes(canonical(envelope))
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite selection output: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as stream:
        stream.write(canonical(envelope) + b"\n")
    return envelope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--generation-repeat-qc", required=True)
    parser.add_argument("--consumer-out", required=True, help="directory with audit.json and consumer_manifest.json")
    parser.add_argument("--output", required=True, help="new selection JSON path; existing files are never overwritten")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"], "run_id": result["run_id"],
        "shortlist_profile_ids": result["selection"]["shortlist_profile_ids"],
        "artifact_sha256": result["artifact_sha256"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
