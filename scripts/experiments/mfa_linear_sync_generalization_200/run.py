#!/usr/bin/env python3
"""Run the immutable 200-record MFA-linear generalization experiment once."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..mfa_linear_real_video_sync.model import load_frozen_syncnet, module_state_sha256
from ..mfa_linear_real_video_sync.protocol import (
    load_mfa_linear_waveform,
    load_pcm16_waveform,
    save_torch_once,
    sha256_file,
    write_json_once,
)
from ..mfa_linear_real_video_sync.evaluate import write_pcm16_wav_once
from ..mfa_linear_real_video_sync.syncnet_loss import log_mel_trust_components, waveform_qc
from ..mfa_linear_sync_transfer.protocol import (
    calibrate_and_materialize_cohort,
    scan_structural_eligibility,
)
from .config import (
    BLOCKED_DATASET_SCALE,
    EXPECTED_MATRIX_CELLS,
    REAL_VIDEO_NOT_EVALUATED,
    REPLACEMENT_GATE_FAILED,
    REPLACEMENT_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    ScaleConfig,
    default_paths,
    run_name,
)
from .protocol import (
    FrozenTTSOnlyAdapter,
    ScaleProtocolError,
    prepare_training_records,
    select_scale_cohorts,
)
from .real_video import decide_real_video, evaluate_real_video_record
from .replacement import (
    decide_replacement,
    materialize_matrix,
    render_arm_once,
    render_p0_seam,
    replacement_record_evidence,
)
from .train import train_scale
from .validate import resume_terminal_run, validate_run

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
    report.write_text(
        "\n".join(
            [
                "# MFA-linear 200-record generalization evaluation",
                "",
                f"- Terminal status: `{decision.get('status')}`",
                f"- Data scale: `{decision.get('data_scale_status')}`",
                f"- Training: `{decision.get('training_status')}`",
                f"- Real-video transfer: `{decision.get('real_video_transfer_status')}`",
                f"- Replacement: `{decision.get('replacement_status')}`",
                "",
                "The evaluation cohort is source-group-heldout within the permitted fit-only LRS3 universe. It is not a sealed-test or population-generalization claim and does not establish perceptual quality, intelligibility, content or speaker preservation, multi-generator safety, or production readiness.",
                "",
                "Scientific constants, memberships, source/checkpoint hashes, curves, matrix cells, and terminal decisions are immutable and fail closed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_failure(output: Path, status: str, error: BaseException, *, data_scale_status: str | None = None) -> dict[str, Any]:
    decision = {
        "schema_version": 1,
        "status": status,
        "pass": False,
        "engineering_status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "config": ScaleConfig().to_dict(),
        "data_scale_status": data_scale_status or status,
        "training_status": "NOT_RUN",
        "real_video_transfer_status": REAL_VIDEO_NOT_EVALUATED,
        "replacement_status": REPLACEMENT_NOT_EVALUATED,
        "claim_scope": "no scientific result; engineering or data-scale prerequisite failure",
    }
    write_json_once(output / "decision.json", decision)
    _write_report(output, decision)
    return decision


def run_focused_tests(repo: Path, output: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *PREDECESSOR_TESTS]
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
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        },
    )
    if completed.returncode != 0:
        raise RuntimeError("focused predecessor suite failed")


def _materialize_candidate(row: Mapping[str, Any], adapter: FrozenTTSOnlyAdapter, output: Path) -> Path:
    sid = str(row["sample_id"])
    mfa, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
    source = torch.from_numpy(mfa).to(next(adapter.parameters()).device)[None, None]
    with torch.inference_mode():
        candidate = adapter(source)
    if tuple(candidate.shape) != tuple(source.shape):
        raise ScaleProtocolError("NATURAL_OR_SIDE_CHANNEL_LEAKAGE: candidate shape changed")
    qc = waveform_qc(candidate, source, residual_bound=ScaleConfig().residual_scale)
    qc["mel_distance"] = float(log_mel_trust_components(candidate, source)["distance"].cpu())
    if not qc["residual_bound_pass"] or not qc["pcm_saturation_pass"] or qc["mel_distance"] > ScaleConfig().mel_trust_limit:
        raise ScaleProtocolError(f"candidate waveform failed frozen QC for {sid}")
    path = Path(write_pcm16_wav_once(output / "03_candidates" / sid / "candidate.wav", candidate.cpu())["path"])
    write_json_once(
        path.parent / "candidate.json",
        {
            "status": "complete",
            "sample_id": sid,
            "source_mfa_sha256": row["mfa_audio_sha256"],
            "candidate_wav_sha256": sha256_file(path),
            "model_input": ["mfa_linear_tts_waveform"],
            "natural_in_adapter": False,
            "qc": qc,
        },
    )
    return path


def _materialize_audio_arms(row: Mapping[str, Any], candidate_path: Path, output: Path) -> dict[str, str]:
    natural, _ = load_pcm16_waveform(row["natural_audio"], expected_sha256=str(row["natural_audio_sha256"]))
    baseline, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
    return {
        "N": write_pcm16_wav_once(output / "06_replacement/evaluation_audio" / str(row["sample_id"]) / "natural.wav", torch.from_numpy(natural)[None, None])["path"],
        "B": write_pcm16_wav_once(output / "06_replacement/evaluation_audio" / str(row["sample_id"]) / "baseline.wav", torch.from_numpy(baseline)[None, None])["path"],
        "C": str(candidate_path.resolve()),
    }


def run(repo: Path, output: Path, *, resume: bool = False) -> dict[str, Any]:
    output = output.resolve()
    repo = repo.resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(f"run root already exists; no overwrite/retry is allowed: {output}")
        return resume_terminal_run(output)
    output.mkdir(parents=True)
    try:
        ScaleConfig().validate()
        write_json_once(output / "00_metadata/config.json", ScaleConfig().to_dict())
        run_focused_tests(repo, output)
        paths = default_paths(repo)
        structural, exclusions = scan_structural_eligibility(
            paths["policy_records"], paths["mfa_linear"] / "summary.json"
        )
        structural_groups = sorted({str(row["source_group"]) for row in structural})
        write_json_once(
            output / "01_data_lock/structural_scan.json",
            {
                "status": "complete",
                "policy_records": str(paths["policy_records"]),
                "mfa_summary": str((paths["mfa_linear"] / "summary.json").resolve()),
                "mfa_summary_sha256": sha256_file(paths["mfa_linear"] / "summary.json"),
                "eligible_count": len(structural),
                "eligible_source_group_count": len(structural_groups),
                "eligible_source_groups": structural_groups,
                "eligible": [_public(row) for row in structural],
                "exclusions": exclusions,
                "outcome_fields_used": False,
                "sealed_splits_accessed": False,
            },
        )
        if len(structural_groups) < ScaleConfig().evaluation_record_count:
            raise ScaleProtocolError(
                f"{BLOCKED_DATASET_SCALE}: current inventory has {len(structural_groups)} eligible source groups and {len(structural)} records; need 40 groups and 200 training records"
            )
        syncnet = load_frozen_syncnet(paths["syncnet_model"], device="cuda")
        calibrated, calibration_info = calibrate_and_materialize_cohort(
            structural, exclusions, syncnet, output / "01_data_lock", device="cuda"
        )
        write_json_once(output / "01_data_lock/calibration.json", calibration_info)
        manifest = select_scale_cohorts(calibrated, calibration_info["exclusions"], output)
    except BaseException as error:
        decision = _write_failure(output, BLOCKED_DATASET_SCALE, error, data_scale_status=BLOCKED_DATASET_SCALE)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        training_records = prepare_training_records(manifest["training"], syncnet, output, device=torch.device("cuda"))
        result = train_scale(training_records, syncnet)
        train_dir = output / "02_training"
        save_torch_once(
            train_dir / "step0/model.pt",
            {"state_dict": result["initial_model_state"], "model_sha256": result["initial_model_sha256"]},
        )
        save_torch_once(
            train_dir / "step100/model.pt",
            {"state_dict": result["model"].state_dict(), "model_sha256": result["final_model_sha256"]},
        )
        serial_history = [
            row for row in result["history"]
            if row["step"] in {1, ScaleConfig().train_steps}
        ]
        write_json_once(
            train_dir / "history.json",
            {
                "status": "complete",
                "stage": result["stage"],
                "steps": result["steps"],
                "record_order": result["record_order"],
                "config": result["config"],
                "initial_model_sha256": result["initial_model_sha256"],
                "final_model_sha256": result["final_model_sha256"],
                "syncnet_sha256_before": result["syncnet_sha256_before"],
                "syncnet_sha256_after": result["syncnet_sha256_after"],
                "history_sampled": serial_history,
            },
        )
        adapter = FrozenTTSOnlyAdapter(result["model"])
        adapter_hash = module_state_sha256(adapter)
        candidate_paths = {
            str(row["sample_id"]): _materialize_candidate(row, adapter, output)
            for row in manifest["evaluation"]
        }
        if module_state_sha256(adapter) != adapter_hash:
            raise ScaleProtocolError("FROZEN_ADAPTER_MUTATION")
    except BaseException as error:
        decision = _write_failure(output, REAL_VIDEO_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    try:
        rows = []
        for row in manifest["evaluation"]:
            candidate, _ = load_pcm16_waveform(
                candidate_paths[str(row["sample_id"])], expected_sha256=sha256_file(candidate_paths[str(row["sample_id"])])
            )
            rows.append(evaluate_real_video_record(row, torch.from_numpy(candidate)[None, None], syncnet, str(output), device=torch.device("cuda")))
        real_video = decide_real_video(rows)
        write_json_once(output / "05_real_video/decision.json", real_video)
    except BaseException as error:
        decision = _write_failure(output, REAL_VIDEO_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    if not real_video["pass"]:
        terminal = {
            "status": real_video["status"],
            "pass": False,
            "engineering_status": real_video["engineering_status"],
            "config": ScaleConfig().to_dict(),
            "data_scale_status": "complete",
            "training_status": "complete",
            "real_video_transfer_status": real_video["status"],
            "replacement_status": REPLACEMENT_GATE_FAILED if real_video["status"] != REAL_VIDEO_NOT_EVALUATED else REPLACEMENT_NOT_EVALUATED,
            "real_video": real_video,
            "claim_scope": real_video["claim_scope"],
        }
        write_json_once(output / "decision.json", terminal)
        _write_report(output, terminal)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return terminal

    try:
        selected = manifest["evaluation"]
        fixture = next(row for row in calibrated if str(row["sample_id"]) not in set(manifest["selected_sample_ids"]))
        fixture_candidate = Path(str(fixture["mfa_audio"]))
        fixture_wavs = _materialize_audio_arms(fixture, fixture_candidate, output)
        write_json_once(output / "04_replacement/p0_seam.json", render_p0_seam(fixture, fixture_wavs, output, repo=repo))
        render_results = {}
        eval_wavs = {}
        for row in selected:
            sid = str(row["sample_id"])
            eval_wavs[sid] = _materialize_audio_arms(row, candidate_paths[sid], output)
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
        write_json_once(output / "06_replacement/decision.json", replacement)
    except BaseException as error:
        decision = _write_failure(output, REPLACEMENT_NOT_EVALUATED, error)
        validation = validate_run(output)
        write_json_once(output / "validation.json", validation)
        return decision

    terminal = {
        "status": replacement["status"],
        "pass": replacement["pass"],
        "engineering_status": replacement["engineering_status"],
        "config": ScaleConfig().to_dict(),
        "data_scale_status": "complete",
        "training_status": "complete",
        "real_video_transfer_status": real_video["status"],
        "replacement_status": replacement["status"],
        "real_video": real_video,
        "replacement": replacement,
        "claim_scope": replacement["claim_scope"],
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.repo, args.output or args.repo / "runs" / run_name(), resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
