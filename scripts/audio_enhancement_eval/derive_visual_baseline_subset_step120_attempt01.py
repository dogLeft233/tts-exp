#!/usr/bin/env python3
"""Create a sealed single-model identity/RMS visual view from controls360.

This copies exact planned/result rows and does not rerender, rewrite source
ledgers, or remove failures from the declared visual denominator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-visual-baseline-subset-v1"
METHODS = ("identity", "rms_fixed")


class SubsetError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SubsetError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise SubsetError(f"expected JSON object at {path}:{line_no}")
                rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    manifest_path = Path(args.result_manifest).resolve(strict=True)
    out = Path(args.out).resolve()
    if out.exists():
        raise SubsetError(f"output directory already exists: {out}")
    plan, source_manifest = read_json(plan_path), read_json(manifest_path)
    source_results = read_jsonl(results_path)
    plan_raw_sha, results_raw_sha, manifest_raw_sha = sha_file(plan_path), sha_file(results_path), sha_file(manifest_path)
    expected = {
        "plan_raw_sha256": args.expected_plan_sha256,
        "results_raw_sha256": args.expected_results_sha256,
        "result_manifest_raw_sha256": args.expected_result_manifest_sha256,
    }
    actual = {"plan_raw_sha256": plan_raw_sha, "results_raw_sha256": results_raw_sha,
              "result_manifest_raw_sha256": manifest_raw_sha}
    if actual != expected:
        raise SubsetError(f"source raw pins mismatch: {actual}")
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise SubsetError("source plan is not frozen cross-TFG audio evaluation data")
    if plan.get("plan_sha256") != payload_sha(plan, "plan_sha256"):
        raise SubsetError("source plan payload SHA mismatch")
    if source_manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or source_manifest.get("run_id") != plan.get("run_id"):
        raise SubsetError("source manifest schema/run_id mismatch")
    if source_manifest.get("manifest_sha256") != payload_sha(source_manifest, "manifest_sha256"):
        raise SubsetError("source result-manifest payload SHA mismatch")
    if (source_manifest.get("plan_raw_sha256") != plan_raw_sha or
            source_manifest.get("plan_sha256") != plan.get("plan_sha256") or
            source_manifest.get("results_jsonl_sha256") != results_raw_sha):
        raise SubsetError("source result manifest does not bind exact frozen inputs")
    status_counts = Counter(str(row.get("status")) for row in source_results)
    expected_status_counts = {state: int(status_counts.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")}
    if (source_manifest.get("planned_rows") != len(plan.get("rows", []))
            or source_manifest.get("result_rows") != len(source_results)
            or source_manifest.get("status_counts") != expected_status_counts):
        raise SubsetError("source result-manifest row/status totals differ from the exact result ledger")
    source_rows = plan.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != len(source_results):
        raise SubsetError("source planned/result row counts differ")
    if [row.get("row_id") for row in source_rows] != [row.get("row_id") for row in source_results]:
        raise SubsetError("source result rows are not in frozen plan order")
    for idx, (planned, result) in enumerate(zip(source_rows, source_results, strict=True)):
        if any(planned.get(field) != result.get(field) for field in
               ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256")):
            raise SubsetError(f"source plan/result identity differs at row {idx}")

    source_methods = plan.get("methods")
    if not isinstance(source_methods, list):
        raise SubsetError("source plan has no method registry")
    method_registry = {str(row.get("method_id")): row for row in source_methods if isinstance(row, dict)}
    if not set(METHODS).issubset(method_registry):
        raise SubsetError(f"source method roster does not include {METHODS}")
    source_models = {str(row.get("model_id")) for row in plan.get("models", []) if isinstance(row, dict)}
    if args.model_id not in source_models:
        raise SubsetError(f"requested model is absent from source plan: {args.model_id}")
    selected_plan = [copy.deepcopy(row) for row in source_rows
                     if row.get("method_id") in METHODS and row.get("model_id") == args.model_id]
    selected_ids = {str(row.get("row_id")) for row in selected_plan}
    selected_results = [copy.deepcopy(row) for row in source_results if str(row.get("row_id")) in selected_ids]
    if [row.get("row_id") for row in selected_plan] != [row.get("row_id") for row in selected_results]:
        raise SubsetError("selected plan/result identities differ")
    counts_by_method = Counter(str(row.get("method_id")) for row in selected_plan)
    if counts_by_method != {"identity": 60, "rms_fixed": 60}:
        raise SubsetError(f"expected exactly 60 identity + 60 rms rows; got {dict(counts_by_method)}")
    models = {str(row.get("model_id")) for row in selected_plan}
    if models != {args.model_id}:
        raise SubsetError(f"visual subset must contain the exact requested model: {sorted(models)}")

    out.mkdir(parents=True)
    derived = copy.deepcopy(plan)
    derived["run_id"] = f"{plan['run_id']}_visual_{args.model_id}_identity_rms_subset_v1"
    derived["methods"] = [copy.deepcopy(method_registry[method]) for method in METHODS]
    derived["models"] = [copy.deepcopy(row) for row in plan.get("models", [])
                         if isinstance(row, dict) and row.get("model_id") == args.model_id]
    derived["rows"] = selected_plan
    derived["source_visual_subset"] = {
        "schema_version": SCHEMA,
        "source_run_id": plan.get("run_id"),
        "source_plan_path": str(plan_path), "source_plan_raw_sha256": plan_raw_sha,
        "source_results_path": str(results_path), "source_results_raw_sha256": results_raw_sha,
        "source_result_manifest_path": str(manifest_path), "source_result_manifest_raw_sha256": manifest_raw_sha,
        "selected_model_id": args.model_id,
        "selected_method_ids": list(METHODS), "selected_rows": len(selected_plan),
        "selection": "copy exact identity and rms_fixed rows in original frozen order; no rerender; failed rows retained",
    }
    derived.pop("plan_sha256", None)
    derived["plan_sha256"] = payload_sha(derived, "plan_sha256")
    plan_out = out / "plan.json"
    plan_out.write_bytes(canonical(derived) + b"\n")
    results_out = out / "results.jsonl"
    with results_out.open("x", encoding="utf-8", newline="\n") as stream:
        for row in selected_results:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    status_counts = Counter(str(row.get("status")) for row in selected_results)
    model_counts = []
    for method in METHODS:
        method_rows = [row for row in selected_results if row.get("method_id") == method]
        model_counts.append({
            "model_id": next(iter(models)), "method_id": method,
            "planned_rows": len([row for row in selected_plan if row.get("method_id") == method]),
            "result_rows": len(method_rows),
            "status_counts": {state: sum(row.get("status") == state for row in method_rows) for state in ("PASS", "FAILED", "INCOMPLETE")},
        })
    manifest = {
        "schema_version": "cross-tfg-audio-eval-v1", "run_id": derived["run_id"],
        "status": "PASS" if status_counts["FAILED"] == 0 and status_counts["INCOMPLETE"] == 0 else "INCOMPLETE",
        "plan_raw_sha256": sha_file(plan_out), "plan_sha256": derived["plan_sha256"],
        "results_jsonl_sha256": sha_file(results_out),
        "planned_rows": len(selected_plan), "result_rows": len(selected_results),
        "status_counts": {state: int(status_counts[state]) for state in ("PASS", "FAILED", "INCOMPLETE")},
        "counts_by_model_method": model_counts, "failure_reasons": [],
        "source_visual_subset": derived["source_visual_subset"],
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    (out / "result_manifest.json").write_bytes(canonical(manifest) + b"\n")
    receipt = {
        "schema_version": SCHEMA, "status": "SUBSET_CREATED",
        "source": {"plan_path": str(plan_path), "plan_raw_sha256": plan_raw_sha,
                   "results_path": str(results_path), "results_raw_sha256": results_raw_sha,
                   "result_manifest_path": str(manifest_path), "result_manifest_raw_sha256": manifest_raw_sha},
        "derived": {"directory": str(out), "plan_raw_sha256": sha_file(plan_out),
                    "results_raw_sha256": sha_file(results_out), "result_manifest_raw_sha256": sha_file(out / "result_manifest.json"),
                    "planned_rows": len(selected_plan), "status_counts": manifest["status_counts"],
                    "methods": list(METHODS), "model_id": next(iter(models))},
        "not_rerendered": True, "effects_not_read": True,
    }
    (out / "subset_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return 0 if manifest["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-results-sha256", required=True)
    parser.add_argument("--expected-result-manifest-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"visual_baseline_subset_failed:{type(exc).__name__}:{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
