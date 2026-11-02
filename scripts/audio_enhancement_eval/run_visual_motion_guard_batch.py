#!/usr/bin/env python3
"""Run the pinned CPU visual-motion guard over sealed existing render ledgers.

This runner only reads source plans/results/videos and writes descriptive mouth
geometry trajectories. It never renders, rescores, shifts audio, or selects a
method. It preserves every planned row, including failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-visual-motion-batch-v1"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
VISUAL_SCHEMA = "cross-tfg-visual-motion-guard-v1"
FEATURE_SHA = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
LANDMARKER_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class BatchError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise BatchError(f"expected JSON object at {path}:{line_no}")
                rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value) + b"\n")


def verify_job(job: Mapping[str, Any]) -> dict[str, Any]:
    key = str(job.get("key"))
    paths: dict[str, Path] = {}
    for name in ("plan", "results", "result_manifest"):
        path = Path(str(job[name])).resolve(strict=True)
        paths[name] = path
        expected = job.get(f"{name}_raw_sha256")
        actual = sha_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise BatchError(f"{key}: exact {name} raw SHA mismatch")
    plan = read_json(paths["plan"])
    manifest = read_json(paths["result_manifest"])
    results = read_jsonl(paths["results"])
    if (plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN" or
            plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")):
        raise BatchError(f"{key}: plan fingerprint/status mismatch")
    if (manifest.get("schema_version") != PLAN_SCHEMA or manifest.get("run_id") != plan.get("run_id") or
            manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
        raise BatchError(f"{key}: result manifest fingerprint/run mismatch")
    if (manifest.get("plan_raw_sha256") != sha_file(paths["plan"]) or
            manifest.get("plan_sha256") != plan.get("plan_sha256") or
            manifest.get("results_jsonl_sha256") != sha_file(paths["results"])):
        raise BatchError(f"{key}: result manifest does not bind exact source plan/results")
    expected_parent = job.get("parent_source")
    if expected_parent is not None:
        parent = plan.get("source_visual_subset")
        if not isinstance(parent, dict):
            raise BatchError(f"{key}: derived baseline plan lacks its parent-source binding")
        parent_fields = {
            "source_run_id": "source_run_id",
            "source_plan_raw_sha256": "source_plan_raw_sha256",
            "source_results_raw_sha256": "source_results_raw_sha256",
            "source_result_manifest_raw_sha256": "source_result_manifest_raw_sha256",
        }
        for expected_field, plan_field in parent_fields.items():
            if expected_parent.get(expected_field) != parent.get(plan_field):
                raise BatchError(f"{key}: derived baseline parent-source pin mismatch: {expected_field}")
        if parent.get("selected_method_ids") != ["identity", "rms_fixed"] or parent.get("selected_rows") != int(job.get("expected_rows", -1)):
            raise BatchError(f"{key}: derived baseline method/row selection differs")
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != len(results):
        raise BatchError(f"{key}: plan/result row counts differ")
    planned_ids = [str(row.get("row_id")) for row in planned]
    result_ids = [str(row.get("row_id")) for row in results]
    if planned_ids != result_ids or len(set(planned_ids)) != len(planned_ids):
        raise BatchError(f"{key}: row identities/order do not match frozen plan")
    status_counts = Counter(str(row.get("status")) for row in results)
    if manifest.get("planned_rows") != len(planned) or manifest.get("result_rows") != len(results):
        raise BatchError(f"{key}: manifest row totals differ")
    if manifest.get("status_counts") != {state: int(status_counts.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")}:
        raise BatchError(f"{key}: manifest status counts differ")
    method_counts = {str(method): sum(row.get("method_id") == method for row in planned)
                     for method in job.get("expected_methods", [])}
    if len(planned) != int(job.get("expected_rows", -1)):
        raise BatchError(f"{key}: planned row count differs from frozen expectation")
    if set(method_counts) != {str(row.get("method_id")) for row in plan.get("methods", [])}:
        raise BatchError(f"{key}: method roster differs from visual job contract")
    if any(count != int(job.get("expected_rows_by_method", {}).get(method, count))
           for method, count in method_counts.items()):
        raise BatchError(f"{key}: per-method row count differs")
    model_ids = sorted({str(row.get("model_id")) for row in planned})
    if model_ids != sorted(str(x) for x in job.get("expected_model_ids", [])):
        raise BatchError(f"{key}: model roster differs")
    return {"key": key, "plan": plan, "manifest": manifest, "planned_rows": planned, "results": results,
            "paths": paths, "raw_sha256": {name: sha_file(path) for name, path in paths.items()},
            "status_counts": {state: int(status_counts.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")},
            "method_counts": method_counts, "model_ids": model_ids}


def output_inventory(root: Path) -> list[dict[str, Any]]:
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
            for path in sorted(item for item in root.rglob("*") if item.is_file())]


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--python", required=True)
    p.add_argument("--visual-script", required=True, type=Path)
    p.add_argument("--feature-source", required=True, type=Path)
    p.add_argument("--landmarker-asset", required=True, type=Path)
    p.add_argument("--overlay", required=True)
    p.add_argument("--expected_visual_script_sha256", required=True)
    return p


def run(args: argparse.Namespace) -> int:
    jobs_path = args.jobs.resolve(strict=True)
    jobs_doc = read_json(jobs_path)
    if jobs_doc.get("schema_version") != "cross-tfg-visual-motion-batch-input-v1" or jobs_doc.get("manifest_sha256") != payload_sha(jobs_doc, "manifest_sha256"):
        raise BatchError("job manifest schema/payload SHA mismatch")
    if sha_file(args.visual_script.resolve(strict=True)) != args.expected_visual_script_sha256:
        raise BatchError("visual guard script SHA mismatch")
    if sha_file(args.feature_source.resolve(strict=True)) != FEATURE_SHA or sha_file(args.landmarker_asset.resolve(strict=True)) != LANDMARKER_SHA:
        raise BatchError("pinned visual feature source or landmarker asset SHA mismatch")
    job_entries = jobs_doc.get("jobs")
    if not isinstance(job_entries, list) or not job_entries:
        raise BatchError("job manifest has no visual jobs")
    prepared = [verify_job(job) for job in job_entries]
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise BatchError(f"fresh output root already exists: {out_root}")
    out_root.mkdir(parents=True)
    (out_root / "worker_logs").mkdir()
    control = out_root / "control"
    control.mkdir()
    progress_path = control / "progress.jsonl"
    started = {"schema_version": SCHEMA, "status": "STARTED", "jobs_manifest_raw_sha256": sha_file(jobs_path),
               "visual_script_sha256": sha_file(args.visual_script), "feature_source_sha256": FEATURE_SHA,
               "landmarker_asset_sha256": LANDMARKER_SHA, "job_count": len(prepared),
               "planned_rows": sum(len(job["planned_rows"]) for job in prepared), "started_unix": time.time(),
               "scope": "existing-render CPU visual extraction only; no render, SyncNet, audio modification, or selection"}
    write_json(control / "started.json", started)
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": args.overlay, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    version_probe = subprocess.run([args.python, "-B", "-c", "import sys,numpy,cv2,mediapipe; print(sys.version.split()[0]+'|'+numpy.__version__+'|'+cv2.__version__+'|'+mediapipe.__version__)"],
                                   capture_output=True, text=True, env=environment, timeout=90)
    if version_probe.returncode != 0:
        raise BatchError(f"qualified CPU visual runtime import failed ({version_probe.returncode})")
    expected_runtime = jobs_doc.get("runtime", {})
    runtime = version_probe.stdout.strip()
    if runtime != expected_runtime.get("version_signature"):
        raise BatchError("visual runtime version signature differs from frozen job manifest")
    job_records = []
    for ordinal, job in enumerate(prepared, 1):
        entry = next(item for item in job_entries if str(item.get("key")) == job["key"])
        out = Path(str(entry["output"])).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise BatchError(f"visual job output path already exists: {out}")
        log_base = out_root / "worker_logs" / job["key"]
        argv = [args.python, "-B", str(args.visual_script.resolve()), "--plan", str(job["paths"]["plan"]),
                "--results", str(job["paths"]["results"]), "--result-manifest", str(job["paths"]["result_manifest"]),
                "--artifact-root", str(Path(str(entry.get("artifact_root", "/"))).resolve(strict=True)), "--out", str(out),
                "--video-features", str(args.feature_source.resolve(strict=True)),
                "--landmarker-asset", str(args.landmarker_asset.resolve(strict=True))]
        append_jsonl(progress_path, {"ordinal": ordinal, "job_key": job["key"], "status": "RUNNING", "rows": len(job["planned_rows"]), "started_unix": time.time()})
        started_at = time.time()
        completed = subprocess.run(argv, capture_output=True, text=True, env=environment, timeout=int(entry.get("timeout_seconds", 21600)))
        (log_base.with_suffix(".stdout.log")).write_text(completed.stdout, encoding="utf-8")
        (log_base.with_suffix(".stderr.log")).write_text(completed.stderr, encoding="utf-8")
        visual_manifest_path = out / "visual_motion_manifest.json"
        audit_path = out / "audit.json"
        visual_manifest = read_json(visual_manifest_path) if visual_manifest_path.is_file() else None
        audit = read_json(audit_path) if audit_path.is_file() else None
        good = bool(completed.returncode == 0 and visual_manifest and audit and
                    visual_manifest.get("status") == "PASS" and audit.get("planned_rows") == len(job["planned_rows"]) and
                    audit.get("accounted_for_planned_rows") == len(job["planned_rows"]) and
                    audit.get("plan_raw_sha256") == job["raw_sha256"]["plan"] and
                    audit.get("results_raw_sha256") == job["raw_sha256"]["results"] and
                    audit.get("result_manifest_raw_sha256") == job["raw_sha256"]["result_manifest"])
        record = {"job_key": job["key"], "status": "PASS" if good else ("INCOMPLETE" if visual_manifest else "FAILED"),
                  "return_code": completed.returncode, "elapsed_seconds": time.time() - started_at,
                  "source_pins": job["raw_sha256"], "output_path": str(out),
                  "visual_manifest_raw_sha256": sha_file(visual_manifest_path) if visual_manifest_path.is_file() else None,
                  "visual_manifest_status": visual_manifest.get("status") if visual_manifest else None,
                  "visual_status_counts": audit.get("visual_status_counts") if audit else None,
                  "accounted_for_planned_rows": audit.get("accounted_for_planned_rows") if audit else None,
                  "stderr_sha256": sha_bytes(completed.stderr.encode("utf-8")),
                  "stdout_sha256": sha_bytes(completed.stdout.encode("utf-8")),
                  "stderr_bytes": len(completed.stderr.encode("utf-8")), "stdout_bytes": len(completed.stdout.encode("utf-8"))}
        job_records.append(record)
        append_jsonl(progress_path, {"ordinal": ordinal, **record, "finished_unix": time.time()})

    state = "PASS" if all(row["status"] == "PASS" for row in job_records) else "INCOMPLETE"
    inventory = []
    for job in prepared:
        entry = next(item for item in job_entries if str(item.get("key")) == job["key"])
        visual_out = Path(str(entry["output"])).resolve()
        if visual_out.is_dir():
            for item in output_inventory(visual_out):
                inventory.append({"job_key": job["key"], "path": str((visual_out / item["path"]).relative_to(out_root).as_posix()),
                                 "bytes": item["bytes"], "sha256": item["sha256"]})
    terminal = {"schema_version": SCHEMA, "status": state, "job_count": len(job_records), "planned_rows": started["planned_rows"],
                "job_results": job_records, "visual_output_inventory": inventory, "runtime_version_signature": runtime,
                "feature_source_sha256": FEATURE_SHA, "landmarker_asset_sha256": LANDMARKER_SHA,
                "confirmation_accessed": False, "rendering_or_syncnet_rescore": False,
                "interpretation": "descriptive geometry/tracking only; mouth-motion magnitude is not a quality criterion"}
    write_json(control / "batch_terminal.json", terminal)
    archive_path = out_root / "visual_motion_outputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(item for item in out_root.rglob("*") if item.is_file() and item != archive_path):
            archive.add(path, arcname=path.relative_to(out_root).as_posix(), recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive_member_count = len(archive.getmembers())
    receipt = {"schema_version": SCHEMA, "status": state, "output_root": str(out_root),
               "batch_terminal_sha256": sha_file(control / "batch_terminal.json"),
               "archive_path": str(archive_path), "archive_bytes": archive_path.stat().st_size,
               "archive_sha256": sha_file(archive_path), "archive_members": archive_member_count,
               "job_count": len(job_records), "planned_rows": started["planned_rows"],
               "job_status_counts": dict(Counter(row["status"] for row in job_records))}
    write_json(control / "archive_receipt.json", receipt)
    return 0 if state == "PASS" else 2


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"visual_motion_batch_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
