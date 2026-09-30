#!/usr/bin/env python3
"""Metadata-only preflight for the frozen WORLD physical alignment overlay.

This checks all original EVAL TextGrid hashes and invokes the runner's actual
overlay loader. It does not open or decode source waveforms and does not create
any WORLD features or intervention audio.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_asts(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--reference-runner", type=Path, required=True)
    parser.add_argument("--mfa-out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--annex", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner_path = args.runner.resolve(strict=True)
    reference_path = args.reference_runner.resolve(strict=True)
    mfa_out = args.mfa_out.resolve(strict=True)
    overlay_path = args.overlay.resolve(strict=True)
    annex_path = args.annex.resolve(strict=True)
    bindings_path = args.bindings.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a preflight report")

    runner = load_module(runner_path, "world_phone_transfer_overlay_preflight")
    producer_path = mfa_out / "mfa_producer_manifest.json"
    protocol_path = mfa_out / "alignment_protocol_frozen.json"
    producer = runner.read_json(producer_path)
    protocol = runner.read_json(protocol_path)
    qualification = runner._qualification_map(mfa_out, "evaluation")

    grid_checks: list[dict[str, Any]] = []
    for sample_id, record in sorted(producer.get("records", {}).items()):
        if record.get("split") != "evaluation":
            continue
        qrow = qualification.get(str(sample_id), {})
        for condition in ("natural", "tts"):
            grid_meta = record["conditions"][condition]
            expected = str(qrow.get(condition, {}).get("textgrid_sha256", ""))
            path = Path(grid_meta["expected_textgrid_path"]).resolve(strict=True)
            actual = sha256_file(path)
            grid_checks.append({
                "id": str(sample_id),
                "condition": condition,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matched": bool(expected) and actual == expected,
            })
    if len(grid_checks) != 160 or len({row["id"] for row in grid_checks}) != 80 or not all(row["matched"] for row in grid_checks):
        raise ValueError("original EVAL TextGrid hash audit failed its 80-pair/160-grid requirement")

    overlay = runner._load_alignment_overlay(
        overlay_path,
        annex_path,
        bindings_path,
        mfa_out,
        producer,
        protocol,
        qualification,
    )
    if len(overlay.get("record_by_id", {})) != 1:
        raise ValueError("loader did not return exactly the single physical overlay pair")

    probe = overlay["probe_manifest"]
    overlay_row = overlay["record_by_id"]["S0064_BAC009S0064W0121"]
    overlay_grid_checks = []
    for condition in ("natural", "tts"):
        physical = overlay_row["conditions"][condition]
        probe_grid_sha = probe["alignment_rows"][condition]["textgrid_sha256"]
        probe_row = probe["conditions"][condition]
        grid_path = Path(physical["physical_textgrid_path"]).resolve(strict=True)
        lab_path = Path(physical["physical_lab_path"]).resolve(strict=True)
        overlay_grid_checks.append({
            "condition": condition,
            "physical_textgrid_sha256": sha256_file(grid_path),
            "probe_alignment_rows_textgrid_sha256": probe_grid_sha,
            "physical_lab_sha256": sha256_file(lab_path),
            "probe_conditions_lab_sha256": probe_row["lab_sha256"],
            "source_audio_sha256_from_metadata": physical["source_audio_sha256"],
            "textgrid_match": sha256_file(grid_path) == probe_grid_sha == physical["physical_textgrid_sha256"],
            "lab_match": sha256_file(lab_path) == probe_row["lab_sha256"] == physical["physical_lab_sha256"],
        })
    if not all(row["textgrid_match"] and row["lab_match"] for row in overlay_grid_checks):
        raise ValueError("physical overlay Grid/LAB linkage failed")

    current_asts = function_asts(runner_path)
    reference_asts = function_asts(reference_path)
    changed_functions = sorted(
        name for name in set(current_asts) | set(reference_asts)
        if current_asts.get(name) != reference_asts.get(name)
    )
    if changed_functions != ["_load_alignment_overlay"]:
        raise ValueError(f"attempt03 code differs outside the loader function: {changed_functions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "world_physical_overlay_metadata_preflight",
        "status": "PASS_METADATA_ONLY",
        "runner_path": str(runner_path),
        "runner_source_sha256": sha256_file(runner_path),
        "reference_runner_path": str(reference_path),
        "reference_runner_source_sha256": sha256_file(reference_path),
        "changed_top_level_functions_vs_reference": changed_functions,
        "preflight_source_path": str(Path(__file__).resolve()),
        "preflight_source_sha256": sha256_file(Path(__file__).resolve()),
        "mfa_producer_manifest_raw_sha256": sha256_file(producer_path),
        "mfa_protocol_raw_sha256": sha256_file(protocol_path),
        "evaluation_eligibility_raw_sha256": sha256_file(mfa_out / "evaluation_endpoint_eligibility.json"),
        "alignment_overlay_annex_sha256": sha256_file(annex_path),
        "execution_bindings_sha256": sha256_file(bindings_path),
        "overlay_manifest_path": str(overlay_path),
        "overlay_manifest_raw_sha256": sha256_file(overlay_path),
        "overlay_manifest_fingerprint": overlay["manifest"]["manifest_sha256"],
        "overlay_pair_count": len(overlay["record_by_id"]),
        "probe_manifest_path": str(Path(overlay["manifest"]["probe_manifest_path"]).resolve(strict=True)),
        "probe_manifest_raw_sha256": sha256_file(Path(overlay["manifest"]["probe_manifest_path"]).resolve(strict=True)),
        "probe_manifest_fingerprint": probe["manifest_sha256"],
        "original_eval_grid_count": len(grid_checks),
        "original_eval_pair_count": len({row["id"] for row in grid_checks}),
        "original_eval_grid_checks": grid_checks,
        "original_eval_grid_checklist_sha256": canonical_sha(grid_checks),
        "physical_overlay_grid_checks": overlay_grid_checks,
        "physical_overlay_record_sha256": overlay_row["record_sha256"],
        "physical_overlay_source_identity_sha256": overlay_row["source_identity_sha256"],
        "loader_returned_pair_count": len(overlay["record_by_id"]),
        "evaluation_waveforms_read": False,
        "evaluation_audio_decoded": False,
        "evaluation_intervention_waveforms_created": False,
        "evaluation_effects_read": False,
    }
    report["manifest_sha256"] = canonical_sha(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "original_eval_grids": len(grid_checks),
        "original_eval_pairs": report["original_eval_pair_count"],
        "physical_overlay_pairs": len(overlay["record_by_id"]),
        "changed_top_level_functions_vs_reference": changed_functions,
        "runner_source_sha256": report["runner_source_sha256"],
        "overlay_manifest_raw_sha256": report["overlay_manifest_raw_sha256"],
        "overlay_manifest_fingerprint": report["overlay_manifest_fingerprint"],
        "preflight_manifest_sha256": report["manifest_sha256"],
        "output": str(output_path),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
