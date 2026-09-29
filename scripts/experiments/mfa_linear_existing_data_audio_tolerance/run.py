#!/usr/bin/env python3
"""Run the post-hoc 0.11 audio-tolerance diagnostic once."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..mfa_linear_real_video_sync.evaluate import write_pcm16_wav_once
from ..mfa_linear_real_video_sync.protocol import load_pcm16_waveform, write_json_once
from ..mfa_linear_sync_transfer.replacement import materialize_matrix, render_arm_once, render_p0_seam, replacement_record_evidence
from ..mfa_linear_sync_existing_data.run import _audio_arms
from .config import (
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_NOT_EVALUATED,
    DIAGNOSTIC_PARENT_INVALID,
    EVAL_RECORD_COUNT,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_NOT_RUN,
    REPLACEMENT_OBSERVED,
    DiagnosticConfig,
    default_output,
    default_parent,
)
from .protocol import load_adapter, materialize_candidate, read_object, sha256_file, validate_parent
from .real_video import decide_diagnostic, evaluate_record
from .replacement import decide_replacement
from .validate import resume_terminal_run, validate_run

FOCUSED_TESTS = (
    "tests/experiments/mfa_linear_existing_data_audio_tolerance",
    "tests/experiments/mfa_linear_sync_existing_data",
    "tests/experiments/mfa_linear_real_video_sync",
)


def _report(output: Path, decision: Mapping[str, Any]) -> None:
    path = output / "report.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        "\n".join(
            [
                "# MFA-linear post-hoc audio-tolerance diagnostic",
                "",
                f"- Terminal status: `{decision.get('status')}`",
                f"- Original normalized log-mel limit: `{DiagnosticConfig().original_mel_limit}`",
                f"- Diagnostic normalized log-mel limit: `{DiagnosticConfig().diagnostic_mel_limit}`",
                "- This limit was chosen after observing one candidate at `0.1005600542`; all results are exploratory and post-hoc.",
                "",
                "The parent model and eight record-heldout IDs are reused without retraining or selection. Source groups may overlap between training and evaluation; this is not a source-group, sealed-test, population, perceptual, content, speaker, multi-generator, or production claim.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _failure(output: Path, status: str, error: BaseException) -> dict[str, Any]:
    decision = {
        "schema_version": 1,
        "status": status,
        "pass": False,
        "engineering_status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "config": DiagnosticConfig().to_dict(),
        "scientific_claim_available": False,
        "replacement_status": REPLACEMENT_NOT_EVALUATED,
    }
    write_json_once(output / "decision.json", decision)
    _report(output, decision)
    return decision


def _tests(repo: Path, output: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *FOCUSED_TESTS]
    completed = subprocess.run(command, cwd=str(repo), check=False, capture_output=True, text=True)
    write_json_once(
        output / "00_metadata/focused_tests.json",
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "command": command,
            "cwd": str(repo),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
        },
    )
    if completed.returncode != 0:
        raise RuntimeError("diagnostic focused tests failed")


def run(repo: Path, output: Path, *, parent: Path | None = None, resume: bool = False) -> dict[str, Any]:
    repo = repo.resolve()
    output = output.resolve()
    parent = (parent or default_parent(repo)).resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(f"diagnostic run root already exists: {output}")
        from .validate import resume_terminal_run
        return resume_terminal_run(output)
    output.mkdir(parents=True)
    try:
        DiagnosticConfig().validate()
        write_json_once(output / "00_metadata/config.json", DiagnosticConfig().to_dict())
        _tests(repo, output)
        parent_lock = validate_parent(parent)
        write_json_once(output / "01_parent/parent_lock.json", parent_lock)
        parent_manifest = read_object(parent_lock["manifest_path"])
        write_json_once(output / "01_parent/manifest.json", parent_manifest)
        adapter = load_adapter(parent_lock, device="cuda")
        before = sha256_file(parent_lock["checkpoint_path"])
        candidate_results = [
            materialize_candidate(row, adapter, output, mel_limit=DiagnosticConfig().diagnostic_mel_limit)
            for row in parent_manifest["evaluation"]
        ]
        if any(row.get("status") != "complete" for row in candidate_results):
            decision = {
                "stage": "DIAGNOSTIC_CANDIDATES",
                "status": DIAGNOSTIC_NOT_EVALUATED,
                "pass": False,
                "engineering_status": "invalid_or_incomplete",
                "records": candidate_results,
                "scientific_claim_available": False,
            }
            write_json_once(output / "03_real_video/decision.json", decision)
            terminal = {
                "schema_version": 1,
                "status": DIAGNOSTIC_NOT_EVALUATED,
                "pass": False,
                "engineering_status": "invalid_or_incomplete",
                "config": DiagnosticConfig().to_dict(),
                "parent_lock": parent_lock,
                "candidate_results": candidate_results,
                "scientific_claim_available": False,
                "replacement_status": REPLACEMENT_NOT_EVALUATED,
            }
            write_json_once(output / "decision.json", terminal)
            _report(output, terminal)
            validation = validate_run(output)
            write_json_once(output / "validation.json", validation)
            return terminal
        if sha256_file(parent_lock["checkpoint_path"]) != before:
            raise ExistingDataProtocolError("DIAGNOSTIC_PARENT_INVALID: parent changed during inference")
        candidate_by_id = {str(row["sample_id"]): Path(str(item["candidate_path"])) for row, item in zip(parent_manifest["evaluation"], candidate_results, strict=True)}
    except BaseException as error:
        decision = _failure(output, DIAGNOSTIC_PARENT_INVALID if "parent" in str(error).lower() else DIAGNOSTIC_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        syncnet = __import__("scripts.experiments.mfa_linear_real_video_sync.model", fromlist=["load_frozen_syncnet"]).load_frozen_syncnet(
            repo / "third_party/syncnet_python/data/syncnet_v2.model", device="cuda"
        )
        rows = []
        for row in parent_manifest["evaluation"]:
            candidate, _ = load_pcm16_waveform(candidate_by_id[str(row["sample_id"])], expected_sha256=sha256_file(candidate_by_id[str(row["sample_id"])]))
            rows.append(evaluate_record(row, torch.from_numpy(candidate)[None, None], syncnet, output, device=torch.device("cuda")))
        diagnostic = decide_diagnostic(rows)
        write_json_once(output / "03_real_video/decision.json", diagnostic)
    except BaseException as error:
        decision = _failure(output, DIAGNOSTIC_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    if not diagnostic.get("unchanged_real_video_gate_pass"):
        terminal = {
            "schema_version": 1,
            "status": DIAGNOSTIC_COMPLETE,
            "pass": False,
            "engineering_status": diagnostic["engineering_status"],
            "config": DiagnosticConfig().to_dict(),
            "parent_lock": parent_lock,
            "diagnostic": diagnostic,
            "scientific_claim_available": False,
            "replacement_status": REPLACEMENT_NOT_RUN,
        }
        write_json_once(output / "decision.json", terminal)
        _report(output, terminal)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return terminal

    try:
        selected = parent_manifest["evaluation"]
        fixture = selected[0]
        fixture_candidate = candidate_by_id[str(fixture["sample_id"])]
        fixture_wavs = _audio_arms(fixture, fixture_candidate, output, "p0_fixture")
        write_json_once(output / "04_replacement/p0_seam.json", render_p0_seam(fixture, fixture_wavs, output, repo=repo))
        render_results: dict[str, dict[str, dict[str, Any]]] = {}
        eval_wavs: dict[str, dict[str, str]] = {}
        for row in selected:
            sid = str(row["sample_id"])
            eval_wavs[sid] = _audio_arms(row, candidate_by_id[sid], output, "evaluation_audio")
            render_results[sid] = {
                arm: render_arm_once(row, arm, eval_wavs[sid][arm], output, repo=repo)
                for arm in ("N", "B", "C")
            }
        cells = materialize_matrix(selected, render_results, eval_wavs, syncnet, output, device=torch.device("cuda"))
        replacement_rows = []
        for row in selected:
            sid = str(row["sample_id"])
            oracle = next(cell["metrics"] for cell in cells if cell["sample_id"] == sid and cell["key"] == "G_N_E_N")
            replacement_rows.append(replacement_record_evidence(cells, sid, int(oracle["best_signed_shift"])))
        replacement = decide_replacement(replacement_rows, cell_count=len(cells))
        write_json_once(output / "04_replacement/decision.json", replacement)
    except BaseException as error:
        decision = _failure(output, REPLACEMENT_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision
    terminal = {
        "schema_version": 1,
        "status": DIAGNOSTIC_COMPLETE,
        "pass": False,
        "engineering_status": "complete",
        "config": DiagnosticConfig().to_dict(),
        "parent_lock": parent_lock,
        "diagnostic": diagnostic,
        "replacement": replacement,
        "scientific_claim_available": False,
        "replacement_status": replacement["status"],
    }
    write_json_once(output / "decision.json", terminal)
    _report(output, terminal)
    validation = validate_run(output)
    write_json_once(output / "validation.json", validation)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.repo, args.output or default_output(args.repo), parent=args.parent, resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
