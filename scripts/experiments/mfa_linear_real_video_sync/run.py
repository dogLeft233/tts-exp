#!/usr/bin/env python3
"""Run P0 then the single authorized P1 fit; P2 requires explicit opt-in."""
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
from typing import Any, Sequence

import torch

from .config import AssetRoots, P1_SAMPLE_ID, PrototypeConfig, Q, SEED
from .evaluate import (
    decide_p1,
    decide_p2,
    mux_condition_once,
    official_file_curve,
    proxy_file_parity,
    write_pcm16_wav_once,
)
from .model import (
    assert_identity,
    build_waveform_model,
    disposable_two_backward_smoke,
    module_state_sha256,
)
from .prepare import prepare_p1, prepare_p2
from .protocol import ProtocolError, save_torch_once, sha256_file, write_json_once
from .train import compute_record_loss, train_fixed_steps

RUN_NAME = "lrs3_mfa_linear_real_video_sync_20260903"
TEST_TARGETS = (
    "tests/experiments/mfa_linear_real_video_sync",
    "tests/test_pnp_audio_enhancer.py",
    "tests/experiments/lrs3_mfa_linear_replacement/test_protocol.py",
    "tests/experiments/lrs3_mfa_linear_replacement/test_candidate_audio.py",
)


def _jsonable_training_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"model", "optimizer", "initial_model_state"}
    }
    for key in ("step0", "final"):
        payload[key] = [
            {name: value for name, value in row.items() if name != "candidate"}
            for row in result[key]
        ]
    return payload


def _write_terminal_failure(output: Path, *, stage: str, status: str, error: BaseException) -> None:
    path = output / "decision.json"
    if path.exists():
        return
    write_json_once(
        path,
        {
            "status": status,
            "pass": False,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "claim_scope": "no scientific result; engineering stage failed",
            "p2_status": "P2_NOT_RUN",
        },
    )


def run_focused_tests(repo: Path, output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *TEST_TARGETS]
    completed = subprocess.run(command, cwd=repo, check=False, capture_output=True, text=True)
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
    write_json_once(output / "01_p0_seam/test_results.json", result)
    if completed.returncode != 0:
        raise RuntimeError("focused P0/regression tests failed")
    return result


def run_p0(repo: Path, output: Path) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    test_result = run_focused_tests(repo, output)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    roots = AssetRoots.defaults(repo)
    record, syncnet, lock = prepare_p1(roots, output, device=device)

    fresh = build_waveform_model(seed=SEED, device=device)
    assert_identity(fresh, record.mfa_linear_tts_waveform)
    initial_hash = module_state_sha256(fresh)
    canonical_video = Path(lock["visual"]["path"])
    step0_dir = output / "02_p1_one_record/step0"
    wav_lock = write_pcm16_wav_once(step0_dir / f"{P1_SAMPLE_ID}.wav", record.mfa_linear_tts_waveform)
    condition = step0_dir / f"{P1_SAMPLE_ID}.avi"
    mux_condition_once(canonical_video, wav_lock["path"], condition)
    official = official_file_curve(
        syncnet,
        condition,
        expected_frame_hashes=lock["visual"]["decoded_bgr_frame_sha256"],
        expected_scoring_frame_hashes=lock["visual"]["scoring_bgr_frame_sha256"],
        expected_pcm_sha256=wav_lock["pcm_sha256"],
        device=device,
    )
    parity = proxy_file_parity(
        [float(value) for value in record.pristine_curve.cpu()], official["curve"], q=Q
    )
    if not parity["pass"]:
        raise RuntimeError(f"FILE_PROXY_PARITY_FAILURE: max error {parity['max_abs_error']}")

    syncnet_before = module_state_sha256(syncnet)

    def objective(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
        del waveform
        return compute_record_loss(model, syncnet, record)["total"]

    gradient = disposable_two_backward_smoke(
        seed=SEED,
        waveform=record.mfa_linear_tts_waveform,
        objective=objective,
    )
    syncnet_after = module_state_sha256(syncnet)
    if syncnet_after != syncnet_before:
        raise RuntimeError("FROZEN_STATE_MUTATION")
    if gradient["fresh_model_sha256"] != initial_hash:
        raise RuntimeError("P0 disposable smoke changed the real initialization")
    evidence = {
        "status": "GO",
        "tests": {"status": test_result["status"], "result_sha256": sha256_file(output / "01_p0_seam/test_results.json")},
        "input_lock": True,
        "input_isolation": True,
        "visual_coordinates": True,
        "target_unambiguous": lock["target_offset_artifact"]["best_second_gap"] > 2 * Q,
        "offset_sign": lock["target_offset_artifact"]["official_av_offset"] == -lock["target_offset_artifact"]["target_offset"],
        "mfcc_parity": True,
        "file_proxy_parity": parity,
        "identity": True,
        "frozen_state": syncnet_before == syncnet_after,
        "candidate_gradient": True,
        "gradient_smoke": gradient,
        "initial_model_sha256": initial_hash,
        "syncnet_state_sha256_before": syncnet_before,
        "syncnet_state_sha256_after": syncnet_after,
        "step0_pcm": wav_lock,
        "step0_official": official,
        "step0_proxy_curve": [float(value) for value in record.pristine_curve.cpu()],
        "window_count": 91,
    }
    write_json_once(output / "01_p0_seam/evidence.json", evidence)
    write_json_once(step0_dir / "official_curve.json", {"status": "complete", **official, "proxy_parity": parity})
    return record, syncnet, evidence


def run_p1(output: Path, record: Any, syncnet: torch.nn.Module, p0: dict[str, Any]) -> dict[str, Any]:
    result = train_fixed_steps([record], syncnet, stage="P1_ONE_RECORD", config=PrototypeConfig())
    p1_dir = output / "02_p1_one_record"
    save_torch_once(
        p1_dir / "step0/model.pt",
        {"state_dict": result["initial_model_state"], "model_sha256": result["initial_model_sha256"]},
    )
    save_torch_once(
        p1_dir / "step20/model.pt",
        {"state_dict": result["model"].state_dict(), "model_sha256": result["final_model_sha256"]},
    )
    write_json_once(p1_dir / "history.json", {"status": "complete", **_jsonable_training_result(result)})

    final = result["final"][0]
    final_wav = write_pcm16_wav_once(p1_dir / "step20" / f"{P1_SAMPLE_ID}.wav", final["candidate"])
    final_condition = p1_dir / "step20" / f"{P1_SAMPLE_ID}.avi"
    canonical = output / "00_lock/visual" / f"{P1_SAMPLE_ID}.avi"
    mux_condition_once(canonical, final_wav["path"], final_condition)
    lock = json.loads((output / "00_lock/input_lock.json").read_text(encoding="utf-8"))
    official_final = official_file_curve(
        syncnet,
        final_condition,
        expected_frame_hashes=lock["visual"]["decoded_bgr_frame_sha256"],
        expected_pcm_sha256=final_wav["pcm_sha256"],
        device=torch.device("cuda"),
    )
    parity = proxy_file_parity(final["curve"], official_final["curve"])
    write_json_once(
        p1_dir / "step20/official_curve.json",
        {"status": "complete", **official_final, "proxy_parity": parity},
    )
    evidence = {
        "input_lock": p0["input_lock"],
        "input_isolation": p0["input_isolation"],
        "visual_coordinates": p0["visual_coordinates"],
        "target_unambiguous": p0["target_unambiguous"],
        "offset_sign": p0["offset_sign"],
        "mfcc_parity": p0["mfcc_parity"],
        "identity": p0["identity"],
        "frozen_state": result["syncnet_sha256_before"] == result["syncnet_sha256_after"],
        "candidate_gradient": p0["candidate_gradient"],
        "proxy_step0_loss": result["step0"][0]["sync_loss"],
        "proxy_final_loss": final["sync_loss"],
        "official_step0_curve": p0["step0_official"]["curve"],
        "official_final_curve": official_final["curve"],
        "proxy_final_curve": final["curve"],
        "target_offset": record.target_offset,
        "mel_distance": final["mel_distance"],
        "saturation_fraction": final["qc"]["pcm_saturation_fraction"],
        "final_qc": final["qc"],
    }
    decision = decide_p1(evidence)
    decision.update(
        {
            "status_detail": evidence,
            "config_sha256": sha256_file(p1_dir / "history.json"),
            "step0_model_sha256": result["initial_model_sha256"],
            "step20_model_sha256": result["final_model_sha256"],
        }
    )
    write_json_once(p1_dir / "decision.json", decision)
    return decision


def run_p2_stage(
    repo: Path,
    output: Path,
    p1_record: Any,
    syncnet: torch.nn.Module,
    p0: dict[str, Any],
) -> dict[str, Any]:
    roots = AssetRoots.defaults(repo)
    p1_lock = json.loads((output / "00_lock/input_lock.json").read_text(encoding="utf-8"))
    records, locks = prepare_p2(
        roots,
        output,
        p1_record=p1_record,
        p1_lock=p1_lock,
        syncnet=syncnet,
        device=torch.device("cuda"),
    )
    result = train_fixed_steps(records, syncnet, stage="P2_SHARED_FOUR", config=PrototypeConfig())
    p2_dir = output / "03_p2_shared_four"
    save_torch_once(
        p2_dir / "step0/model.pt",
        {"state_dict": result["initial_model_state"], "model_sha256": result["initial_model_sha256"]},
    )
    save_torch_once(
        p2_dir / "step100/model.pt",
        {"state_dict": result["model"].state_dict(), "model_sha256": result["final_model_sha256"]},
    )
    write_json_once(p2_dir / "history.json", {"status": "complete", **_jsonable_training_result(result)})
    evidence_rows = []
    for record, lock, step0, final in zip(records, locks, result["step0"], result["final"], strict=True):
        sample_id = record.sample_id
        record_root = p2_dir / sample_id
        official_rows = []
        for label, evaluated in (("step0", step0), ("step100", final)):
            condition_dir = record_root / label
            wav_lock = write_pcm16_wav_once(condition_dir / f"{sample_id}.wav", evaluated["candidate"])
            condition = condition_dir / f"{sample_id}.avi"
            mux_condition_once(lock["visual"]["path"], wav_lock["path"], condition)
            official = official_file_curve(
                syncnet,
                condition,
                expected_frame_hashes=lock["visual"]["decoded_bgr_frame_sha256"],
                expected_pcm_sha256=wav_lock["pcm_sha256"],
                device=torch.device("cuda"),
            )
            parity = proxy_file_parity(evaluated["curve"], official["curve"])
            write_json_once(condition_dir / "official_curve.json", {"status": "complete", **official, "proxy_parity": parity})
            official_rows.append((official, parity))
        evidence_rows.append(
            {
                "sample_id": sample_id,
                "input_lock": True,
                "input_isolation": True,
                "visual_coordinates": True,
                "target_unambiguous": lock["target_offset_artifact"]["best_second_gap"] > 2 * Q,
                "offset_sign": lock["target_offset_artifact"]["official_av_offset"] == -record.target_offset,
                "mfcc_parity": p0["mfcc_parity"],
                "identity": True,
                "frozen_state": result["syncnet_sha256_before"] == result["syncnet_sha256_after"],
                "candidate_gradient": p0["candidate_gradient"],
                "proxy_step0_loss": step0["sync_loss"],
                "proxy_final_loss": final["sync_loss"],
                "official_step0_curve": official_rows[0][0]["curve"],
                "official_final_curve": official_rows[1][0]["curve"],
                "proxy_final_curve": final["curve"],
                "target_offset": record.target_offset,
                "mel_distance": final["mel_distance"],
                "saturation_fraction": final["qc"]["pcm_saturation_fraction"],
                "final_qc": final["qc"],
            }
        )
    decision = decide_p2(evidence_rows)
    decision.update(
        {
            "status_detail": evidence_rows,
            "step0_model_sha256": result["initial_model_sha256"],
            "step100_model_sha256": result["final_model_sha256"],
            "fresh_from_p1": result["initial_model_sha256"] == p0["initial_model_sha256"],
        }
    )
    write_json_once(p2_dir / "decision.json", decision)
    return decision


def resume_terminal_run(output: Path) -> dict[str, Any]:
    validation_path = output / "validation.json"
    decision_path = output / "decision.json"
    if not validation_path.is_file() or not decision_path.is_file():
        raise ProtocolError("resume requires a completed validated terminal run")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if validation.get("artifact_graph_valid") is not True or validation.get("status") != "valid":
        raise ProtocolError("resume validation is incomplete or invalid")
    if validation.get("decision_sha256") != sha256_file(decision_path):
        raise ProtocolError("resume terminal decision hash mismatch")
    if validation.get("terminal_status") != decision.get("status"):
        raise ProtocolError("resume terminal status mismatch")
    return decision


def run(repo: Path, output: Path, *, run_p2: bool, resume: bool = False) -> dict[str, Any]:
    if output.exists():
        if not resume:
            raise FileExistsError(f"run root already exists; no retry/overwrite is allowed: {output}")
        if run_p2:
            raise ProtocolError("cannot append P2 to a sealed terminal run")
        return resume_terminal_run(output)
    output.mkdir(parents=True)
    try:
        record, syncnet, p0 = run_p0(repo, output)
    except BaseException as error:
        _write_terminal_failure(output, stage="P0_SEAM", status="ENGINEERING_NO_GO", error=error)
        raise
    try:
        p1_decision = run_p1(output, record, syncnet, p0)
    except BaseException as error:
        _write_terminal_failure(output, stage="P1_ONE_RECORD", status="P1_EXECUTION_FAILURE", error=error)
        raise
    if not p1_decision["pass"]:
        terminal = {**p1_decision, "p2_status": "P2_NOT_RUN_P1_FAILED"}
    elif run_p2:
        try:
            p2_decision = run_p2_stage(repo, output, record, syncnet, p0)
            terminal = {**p2_decision, "p1_status": p1_decision["status"]}
        except BaseException as error:
            _write_terminal_failure(output, stage="P2_SHARED_FOUR", status="P2_EXECUTION_FAILURE", error=error)
            raise
    else:
        terminal = {**p1_decision, "p2_status": "P2_NOT_REQUESTED"}
    write_json_once(output / "decision.json", terminal)
    from .validate import validate_run

    validation = validate_run(output)
    write_json_once(output / "validation.json", validation)
    if not validation["artifact_graph_valid"]:
        raise RuntimeError("ARTIFACT_VALIDATION_FAILURE: terminal artifact graph is invalid")
    return terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-p2", action="store_true", help="request optional P2 only after P1 passes")
    parser.add_argument("--resume", action="store_true", help="read a hash-valid sealed terminal run without changing it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    output = (args.output or repo / "runs" / RUN_NAME).resolve()
    decision = run(repo, output, run_p2=bool(args.run_p2), resume=bool(args.resume))
    print(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if decision.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
