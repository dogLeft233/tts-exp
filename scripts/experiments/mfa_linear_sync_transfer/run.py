#!/usr/bin/env python3
"""Run the preregistered P2-gated transfer and replacement evaluation once."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..mfa_linear_real_video_sync.config import AssetRoots
from ..mfa_linear_real_video_sync.model import load_frozen_syncnet, module_state_sha256
from ..mfa_linear_real_video_sync.protocol import load_mfa_linear_waveform, load_pcm16_waveform, write_json_once
from ..mfa_linear_real_video_sync.evaluate import write_pcm16_wav_once
from ..mfa_linear_real_video_sync.syncnet_loss import log_mel_trust_components, waveform_qc
from .config import (
    BLOCKED_COHORT,
    BLOCKED_P2,
    P2_STATUS,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    REPLACEMENT_GATE_FAILED,
    REPLACEMENT_NOT_EVALUATED,
    SYNCNET_CHECKPOINT_SHA256,
    TransferConfig,
    default_paths,
    p2_parent_run_name,
)
from .protocol import (
    TransferProtocolError,
    calibrate_and_materialize_cohort,
    load_adapter,
    read_object,
    scan_structural_eligibility,
    select_cohort,
    sha256_file,
    validate_p2_parent,
)
from .real_video import decide_real_video, evaluate_real_video_record
from .replacement import (
    decide_replacement,
    materialize_matrix,
    render_arm_once,
    render_p0_seam,
    replacement_record_evidence,
)
from .validate import resume_terminal_run, validate_run

RUN_NAME = "lrs3_mfa_linear_sync_transfer_20260903"
PREDECESSOR_TESTS = (
    "tests/experiments/mfa_linear_real_video_sync",
    "tests/test_pnp_audio_enhancer.py",
    "tests/experiments/lrs3_mfa_linear_replacement/test_protocol.py",
    "tests/experiments/lrs3_mfa_linear_replacement/test_candidate_audio.py",
)


def _public(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("natural_audio_values", None)
    return result


def _write_report(output: Path, decision: Mapping[str, Any]) -> None:
    report = output / "report.md"
    if report.exists():
        raise FileExistsError(f"create-once report already exists: {report}")
    status = str(decision.get("status"))
    lines = [
        "# MFA-linear sync transfer and replacement evaluation",
        "",
        f"- Terminal status: `{status}`",
        f"- P2 prerequisite: `{decision.get('p2_prerequisite_status')}`",
        f"- Real-video transfer: `{decision.get('real_video_transfer_status')}`",
        f"- Replacement: `{decision.get('replacement_status')}`",
        "",
        "## Claim boundary",
        "",
        "Any positive result is limited to the preregistered adapter-heldout source groups and, for replacement, this one hash-locked Wav2Lip/SyncNet fixed-crop protocol. It is not a sealed-test or population claim and does not establish perceptual quality, intelligibility, content or speaker preservation, multi-TFG safety, or production readiness.",
        "",
        "## Integrity",
        "",
        "Scientific constants, cohort membership, parent checkpoint, waveform/video hashes, official curves, and terminal decisions are immutable and fail closed.",
    ]
    report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def _write_failure(output: Path, status: str, error: BaseException) -> dict[str, Any]:
    decision = {
        "schema_version": 1,
        "status": status,
        "pass": False,
        "engineering_status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "config": TransferConfig().to_dict(),
        "p2_prerequisite_status": BLOCKED_P2 if status == BLOCKED_P2 else "not_validated",
        "real_video_transfer_status": REAL_VIDEO_NOT_EVALUATED,
        "replacement_status": REPLACEMENT_NOT_EVALUATED,
        "claim_scope": "no scientific result; engineering or prerequisite failure",
    }
    write_json_once(output / "decision.json", decision)
    _write_report(output, decision)
    return decision


def run_predecessor_p2(repo: Path, parent_root: Path) -> dict[str, Any] | None:
    if parent_root.exists():
        raise FileExistsError(f"fresh P2 parent root already exists; no retry is allowed: {parent_root}")
    from ..mfa_linear_real_video_sync.run import run as predecessor_run
    try:
        return predecessor_run(repo, parent_root, run_p2=True, resume=False)
    except BaseException:
        return None


def run_focused_tests(repo: Path, output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *PREDECESSOR_TESTS]
    completed = subprocess.run(command, cwd=str(repo), check=False, capture_output=True, text=True)
    result = {
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
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json_once(output / "00_metadata/focused_tests.json", result)
    if completed.returncode != 0:
        raise RuntimeError("predecessor focused suite failed")
    return result


def _materialize_audio_arms(row: Mapping[str, Any], candidate_path: Path, output: Path, prefix: str) -> dict[str, str]:
    natural, _ = load_pcm16_waveform(row["natural_audio"], expected_sha256=str(row["natural_audio_sha256"]))
    baseline, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
    paths = {
        "N": write_pcm16_wav_once(output / prefix / str(row["sample_id"]) / "natural.wav", torch.from_numpy(natural)[None, None])["path"],
        "B": write_pcm16_wav_once(output / prefix / str(row["sample_id"]) / "baseline.wav", torch.from_numpy(baseline)[None, None])["path"],
        "C": str(candidate_path.resolve()),
    }
    return paths


def run(repo: Path, output: Path, *, p2_root: Path | None = None, resume: bool = False, run_p2: bool = False) -> dict[str, Any]:
    output = output.resolve()
    repo = repo.resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(f"run root already exists; no overwrite/retry is allowed: {output}")
        return resume_terminal_run(output)
    if p2_root is None:
        p2_root = repo / "runs" / p2_parent_run_name()
    output.mkdir(parents=True)
    try:
        run_focused_tests(repo, output)
        if run_p2:
            run_predecessor_p2(repo, p2_root)
        parent_lock = validate_p2_parent(p2_root, repo=repo)
        write_json_once(output / "00_metadata/parent_lock.json", parent_lock)
    except BaseException as error:
        status = BLOCKED_P2 if not (p2_root / "decision.json").is_file() else BLOCKED_P2
        decision = _write_failure(output, status, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        paths = default_paths(repo)
        structural, structural_exclusions = scan_structural_eligibility(
            paths["policy_records"], paths["mfa_linear"] / "summary.json"
        )
        write_json_once(
            output / "01_cohort_lock/structural_scan.json",
            {
                "status": "complete",
                "mfa_summary_sha256": sha256_file(paths["mfa_linear"] / "summary.json"),
                "eligible_count": len(structural),
                "eligible": [_public(row) for row in structural],
                "exclusions": structural_exclusions,
                "outcome_fields_used": False,
            },
        )
        syncnet = load_frozen_syncnet(paths["syncnet_model"], device="cuda")
        calibrated, calibration_info = calibrate_and_materialize_cohort(
            structural, structural_exclusions, syncnet, output, device="cuda"
        )
        write_json_once(output / "01_cohort_lock/calibration.json", calibration_info)
        manifest = select_cohort(calibrated, calibration_info["exclusions"], output)
    except BaseException as error:
        decision = _write_failure(output, BLOCKED_COHORT, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        adapter = load_adapter(parent_lock, device="cuda")
        adapter_hash_before = module_state_sha256(adapter)
        candidate_paths: dict[str, Path] = {}
        for row in manifest["selected"]:
            mfa, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
            source = torch.from_numpy(mfa).to("cuda")[None, None]
            with torch.inference_mode():
                candidate = adapter(source)
            if tuple(candidate.shape) != tuple(source.shape):
                raise TransferProtocolError("NATURAL_OR_SIDE_CHANNEL_LEAKAGE: adapter output shape mismatch")
            qc = waveform_qc(candidate, source, residual_bound=0.05)
            qc["mel_distance"] = float(log_mel_trust_components(candidate, source)["distance"].cpu())
            if not qc["residual_bound_pass"] or not qc["pcm_saturation_pass"] or qc["mel_distance"] > 0.10:
                raise TransferProtocolError("candidate waveform failed frozen QC")
            path = Path(write_pcm16_wav_once(output / "02_candidates" / str(row["sample_id"]) / "candidate.wav", candidate.detach().cpu())["path"])
            candidate_paths[str(row["sample_id"])] = path
            write_json_once(
                path.parent / "candidate.json",
                {
                    "status": "complete", "sample_id": row["sample_id"], "source_mfa_sha256": row["mfa_audio_sha256"],
                    "parent_lock_sha256": parent_lock["step100_model_sha256"], "candidate_wav_sha256": sha256_file(path),
                    "model_input": ["mfa_linear_tts_waveform"], "natural_in_adapter": False, "qc": qc,
                },
            )
            del source, candidate
        adapter_hash_after = module_state_sha256(adapter)
        if adapter_hash_after != adapter_hash_before:
            raise TransferProtocolError("FROZEN_ADAPTER_MUTATION")
    except BaseException as error:
        write_json_once(
            output / "03_real_video/decision.json",
            {
                "stage": "P3_ADAPTER_HELDOUT_REAL_VIDEO",
                "status": REAL_VIDEO_NOT_EVALUATED,
                "pass": False,
                "engineering_status": "invalid_or_incomplete",
                "error": str(error),
                "records": [],
            },
        )
        decision = _write_failure(output, REAL_VIDEO_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        real_rows = []
        for row in manifest["selected"]:
            candidate, _ = load_pcm16_waveform(candidate_paths[str(row["sample_id"])], expected_sha256=None)
            real_rows.append(
                evaluate_real_video_record(
                    row,
                    torch.from_numpy(candidate)[None, None],
                    syncnet,
                    output,
                    device=torch.device("cuda"),
                )
            )
    except BaseException as error:
        decision = _write_failure(output, REAL_VIDEO_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    p3 = decide_real_video(real_rows)
    write_json_once(output / "03_real_video/decision.json", p3)
    if not p3["pass"]:
        terminal = {
            "status": p3["status"], "pass": False, "engineering_status": p3["engineering_status"],
            "config": TransferConfig().to_dict(), "p2_prerequisite_status": P2_STATUS,
            "real_video_transfer_status": p3["status"],
            "replacement_status": REPLACEMENT_GATE_FAILED if p3["status"] == "NO_ADAPTER_HELDOUT_REAL_VIDEO_TRANSFER" else REPLACEMENT_NOT_EVALUATED,
            "parent_lock": parent_lock, "cohort_manifest_sha256": sha256_file(output / "01_cohort_lock/manifest.json"),
            "claim_scope": p3["claim_scope"], "real_video": p3,
        }
        write_json_once(output / "decision.json", terminal)
        _write_report(output, terminal)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return terminal

    try:
        selected = manifest["selected"]
        fixture = next(row for row in calibrated if str(row["sample_id"]) not in set(manifest["selected_sample_ids"]))
        fixture_candidate = Path(str(fixture["mfa_audio"]))
        fixture_wavs = _materialize_audio_arms(fixture, fixture_candidate, output, "02_p0_seams/audio")
        p0 = render_p0_seam(fixture, fixture_wavs, output, repo=repo)
        write_json_once(output / "02_p0_seams/evidence.json", p0)
        render_results: dict[str, dict[str, dict[str, Any]]] = {}
        eval_wavs: dict[str, dict[str, str]] = {}
        for row in selected:
            sid = str(row["sample_id"])
            candidate_path = candidate_paths[sid]
            wavs = _materialize_audio_arms(row, candidate_path, output, "04_replacement/evaluation_audio")
            eval_wavs[sid] = wavs
            render_results[sid] = {
                arm: render_arm_once(row, arm, wavs[arm], output, repo=repo) for arm in ("N", "B", "C")
            }
        cells = materialize_matrix(selected, render_results, eval_wavs, syncnet, output, device=torch.device("cuda"))
        replacement_rows = []
        for row in selected:
            sid = str(row["sample_id"])
            oracle = next(cell["metrics"] for cell in cells if cell["sample_id"] == sid and cell["key"] == "G_N_E_N")
            evidence = replacement_record_evidence(cells, sid, int(oracle["best_signed_shift"]))
            if oracle["best_second_gap"] <= 0.002:
                evidence["engineering_valid"] = False
                evidence["scientific_success"] = False
                evidence["reason"] = "natural downstream oracle is ambiguous"
            replacement_rows.append(evidence)
        p4 = decide_replacement(replacement_rows)
        write_json_once(output / "04_replacement/decision.json", p4)
    except BaseException as error:
        decision = _write_failure(output, REPLACEMENT_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    terminal = {
        "status": p4["status"], "pass": p4["pass"], "engineering_status": p4["engineering_status"],
        "config": TransferConfig().to_dict(), "p2_prerequisite_status": P2_STATUS,
        "real_video_transfer_status": p3["status"], "replacement_status": p4["status"],
        "parent_lock": parent_lock, "cohort_manifest_sha256": sha256_file(output / "01_cohort_lock/manifest.json"),
        "claim_scope": p4["claim_scope"], "real_video": p3, "replacement": p4,
    }
    write_json_once(output / "decision.json", terminal)
    _write_report(output, terminal)
    validation = validate_run(output)
    write_json_once(output / "validation.json", validation)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--run-p2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = (args.output or repo / "runs" / RUN_NAME).resolve()
    result = run(repo, output, p2_root=args.p2_root, resume=args.resume, run_p2=args.run_p2)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
