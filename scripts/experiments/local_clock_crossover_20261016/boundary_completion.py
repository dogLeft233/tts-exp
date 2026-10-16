#!/usr/bin/env python3
"""Complete the one inclusive-slope boundary pair omitted by pilot_main."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.experiments.local_clock_crossover_20261016 import pilot_main as core


OUT = Path("/dev/shm/local_clock_crossover_20261016_boundary_completion_attempt01")
MAIN_OUT = Path("/dev/shm/local_clock_crossover_20261016_main_attempt01")
SPEAKER, REFERENCE, SOURCE = "S0003", "N1", "N2"
BOUNDARY_TOLERANCE = 1e-12
CONTROL_SHA256 = "381bd121b8b3a8fd2a4016243084a651986437c729aa462df5da2bff5e55245c"
MAIN_RECEIPT_SHA256 = "58c74599cf3e45613b1257765f675434417a90b37328882315aea64c97e3026d"
KERNEL_SHA256 = "f3383047e09a7e30ccee3ef937f43bc0bef8465cce239d4c4a94f86ef683a45d"
SCORE_WORKER_SHA256 = "29892f38c080a484e793b1b11b8b17161062e0e5e1c50eb38d7567ba64691ccc"
SPEC_SHA256 = "fa8f004f7b159410c961080b5ba053509480d7910f62e7c9982c36e35f30780d"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def verify_control_admission() -> tuple[Path, dict[str, Any]]:
    source = MAIN_OUT / "controls" / "control_summary.json"
    if not source.is_file() or sha256_file(source) != CONTROL_SHA256:
        raise RuntimeError("control admission summary is missing or differs from frozen root admission")
    admission = json.loads(source.read_text(encoding="utf-8"))
    if admission.get("status") != "CONTROL_GATE_PASS":
        raise RuntimeError("root control admission is not passing")
    if admission.get("passing_speakers") != 3 or admission.get("required_speakers") != 3:
        raise RuntimeError("root control admission counts changed")
    return source, admission


def verify_prior_pair() -> tuple[Path, dict[str, Any]]:
    path = MAIN_OUT / "main" / "main_receipt.json"
    if not path.is_file() or sha256_file(path) != MAIN_RECEIPT_SHA256:
        raise RuntimeError("preserved main receipt is missing or differs from its readback pin")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    rows = receipt.get("outputs", {}).get(SPEAKER, [])
    matches = [row for row in rows if row.get("reference_role") == REFERENCE and row.get("source_role") == SOURCE]
    if len(matches) != 1:
        raise RuntimeError("preserved main receipt does not contain exactly one target pair")
    pair = matches[0]
    if pair.get("status") != "UNSUPPORTED_LOCAL_MAP" or pair.get("outputs"):
        raise RuntimeError("target pair was already executed or had a different prior status")
    if pair.get("local_map", {}).get("reason") != "slope_outside_0.25_4":
        raise RuntimeError("prior target rejection was not solely the slope boundary")
    if receipt.get("audio_job_count") != 8 or receipt.get("audio_job_count", 0) + 2 > 48:
        raise RuntimeError("prior main audio-forward count or frozen cap changed")
    return path, pair


def accept_inclusive_fp_boundary(raw: dict[str, Any], prior_pair: dict[str, Any]) -> dict[str, Any]:
    if raw.get("status") != "UNSUPPORTED" or raw.get("reason") != "slope_outside_0.25_4":
        raise RuntimeError("recomputed target map is not the expected slope-only rejection")
    old_map = prior_pair["local_map"]
    if raw.get("segment_slopes") != old_map.get("segment_slopes"):
        raise RuntimeError("recomputed target slopes differ from the preserved main receipt")
    slopes = [float(value) for value in raw.get("segment_slopes", [])]
    if not slopes or min(slopes) < 0.25:
        raise RuntimeError("target has a lower-bound slope violation")
    above = [value for value in slopes if value > 4.0]
    if len(above) != 1 or above[0] - 4.0 > BOUNDARY_TOLERANCE:
        raise RuntimeError("target does not have exactly one floating-point upper-bound excess")
    if max(slopes) != float(old_map.get("slope_max")):
        raise RuntimeError("target maximum slope differs from the preserved main receipt")
    corrected = dict(raw)
    corrected.update(
        {
            "status": "SUPPORTED",
            "kind": "piecewise_linear_phone_centers",
            "unmatched_span_rule": "linear interpolation between nearest surrounding matched anchors",
            "inclusive_boundary_classification": {
                "comparison": "frozen inclusive upper limit is 4.0",
                "computed_max_slope": max(slopes),
                "excess_over_4": max(slopes) - 4.0,
                "absolute_rounding_tolerance": BOUNDARY_TOLERANCE,
                "decision": "classify as the inclusive 4.0 boundary; preserve anchors and slope values unchanged",
            },
        }
    )
    return corrected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-admitted", action="store_true")
    args = parser.parse_args()
    if not args.root_admitted:
        raise RuntimeError("explicit root admission flag is required")

    control_source, admission = verify_control_admission()
    prior_receipt_path, prior_pair = verify_prior_pair()
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"refusing preexisting boundary output: {OUT}")
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / "controls").mkdir()
    (OUT / "provenance").mkdir()
    control_copy = OUT / "controls" / "control_summary.json"
    shutil.copyfile(control_source, control_copy)
    if sha256_file(control_copy) != CONTROL_SHA256:
        raise RuntimeError("copied root control admission does not match its frozen hash")

    expected_paths = {
        "kernel": REPO / "scripts/experiments/local_clock_crossover_20261016/pilot_main.py",
        "score_worker": REPO / "scripts/experiments/local_clock_crossover_20261016/score_worker.py",
        "protocol": REPO / ".codex/scratch/local_clock_spec_20261016.md",
        "syncnet_model": REPO / "third_party/syncnet_python/data/syncnet_v2.model",
    }
    pins = {
        "kernel": KERNEL_SHA256,
        "score_worker": SCORE_WORKER_SHA256,
        "protocol": SPEC_SHA256,
        "syncnet_model": SYNCNET_MODEL_SHA256,
    }
    source_records: dict[str, Any] = {}
    for name, path in expected_paths.items():
        actual = hash_record(path)
        if actual["sha256"] != pins[name]:
            raise RuntimeError(f"{name} source pin changed: {actual['sha256']}")
        source_records[name] = actual

    old_main_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    manifest = {
        "schema": "local_clock_boundary_completion_bindings_v1",
        "stage": "boundary_completion",
        "authorized_pair": {"speaker": SPEAKER, "reference_role": REFERENCE, "source_role": SOURCE},
        "max_new_audio_forwards": 2,
        "max_new_mfa_wavs": 2,
        "controls_admission": {"path": str(control_copy.resolve()), "sha256": CONTROL_SHA256, "status": admission["status"], "passing_speakers": 3, "required_speakers": 3},
        "prior_main_receipt": hash_record(prior_receipt_path),
        "prior_target_pair": {"status": prior_pair["status"], "reason": prior_pair["local_map"]["reason"], "slope_min": prior_pair["local_map"]["slope_min"], "slope_max": prior_pair["local_map"]["slope_max"], "outputs": prior_pair["outputs"]},
        "source_pins": source_records,
        "wrapper": hash_record(Path(__file__)),
        "prior_main_audio_forwards": old_main_receipt["audio_job_count"],
        "combined_main_audio_forward_cap": 48,
        "numeric_classification_rule": "inclusive frozen segment-slope bounds [0.25, 4.0]; correct one upper-bound floating representation only when its excess is <=1e-12; never clamp or alter map geometry",
    }
    write_new_json(OUT / "provenance" / "boundary_completion_bindings.json", manifest)

    inputs, bindings = core.load_bindings()
    rows = bindings[SPEAKER]
    ref_row, src_row = rows[REFERENCE], rows[SOURCE]
    helper = core.load_timing_helper()
    _, ref_pcm = core.read_pcm16(ref_row["audio_path"])
    _, src_pcm = core.read_pcm16(src_row["audio_path"])
    raw_map = core.fit_phone_map(
        helper,
        ref_row["textgrid_path"],
        src_row["textgrid_path"],
        ref_pcm["duration_s"],
        src_pcm["duration_s"],
    )
    accepted_map = accept_inclusive_fp_boundary(raw_map, prior_pair)
    if accepted_map["matched_phones"] != prior_pair["local_map"].get("matched_phones"):
        raise RuntimeError("recomputed target phone anchors differ from the preserved main receipt")
    write_new_json(
        OUT / "provenance" / "target_map_boundary_classification.json",
        {"raw_map": raw_map, "accepted_map": accepted_map, "only_change": "status/kind/unmatched-span classification; all anchors and slopes remain identical"},
    )

    original_fit = core.fit_phone_map
    expected_reference_grid = Path(ref_row["textgrid_path"]).resolve()
    expected_source_grid = Path(src_row["textgrid_path"]).resolve()
    expected_durations = (ref_pcm["duration_s"], src_pcm["duration_s"])

    def one_pair_boundary_fit(helper_arg: Any, reference_grid: Path, source_grid: Path, reference_duration: float, source_duration: float) -> dict[str, Any]:
        if Path(reference_grid).resolve() != expected_reference_grid or Path(source_grid).resolve() != expected_source_grid:
            raise RuntimeError("boundary classification was invoked for a pair outside the authorized target")
        if (reference_duration, source_duration) != expected_durations:
            raise RuntimeError("authorized pair durations changed during boundary stage")
        result = dict(accepted_map)
        return result

    core.OUT = OUT
    core.SPEAKERS = (SPEAKER,)
    core.REFERENCES = (REFERENCE,)
    core.ROLES = (REFERENCE, SOURCE)
    core.fit_phone_map = one_pair_boundary_fit
    stage_dir = OUT / "boundary_completion"
    stage_dir.mkdir(exist_ok=False)
    try:
        core.run_main(stage_dir, inputs, bindings, admitted=True)
    finally:
        core.fit_phone_map = original_fit

    main_receipt_path = stage_dir / "main_receipt.json"
    run_receipt_path = stage_dir / "run_receipt.json"
    if not main_receipt_path.is_file() or not run_receipt_path.is_file():
        raise RuntimeError("preserved kernel did not write complete main and run receipts")
    main_receipt = json.loads(main_receipt_path.read_text(encoding="utf-8"))
    result_rows = main_receipt.get("outputs", {}).get(SPEAKER, [])
    if len(result_rows) != 1 or result_rows[0].get("reference_role") != REFERENCE or result_rows[0].get("source_role") != SOURCE:
        raise RuntimeError("boundary output contains anything outside the one admitted pair")
    pair = result_rows[0]
    mfa_receipt_path = stage_dir / "mfa_receipts" / f"{SPEAKER}__main_outputs.json"
    score_receipt_path = stage_dir / "scoring" / "main_worker_receipt.json"
    mfa_doc = json.loads(mfa_receipt_path.read_text(encoding="utf-8")) if mfa_receipt_path.is_file() else {}
    score_doc = json.loads(score_receipt_path.read_text(encoding="utf-8")) if score_receipt_path.is_file() else {}
    status = "BOUNDARY_COMPLETION_PASS"
    if main_receipt.get("audio_job_count") != 2 or pair.get("status") != "QUALIFIED":
        status = "BOUNDARY_COMPLETION_RETAINED_UNQUALIFIED"
    if mfa_doc.get("status") != "PASS" or len(mfa_doc.get("textgrids", {})) != 2:
        status = "BOUNDARY_COMPLETION_RETAINED_MFA_FAILURE"
    if score_doc.get("status") != "PASS" or score_doc.get("returncode") != 0:
        status = "BOUNDARY_COMPLETION_RETAINED_SCORING_FAILURE"
    completion = {
        "schema": "local_clock_boundary_completion_receipt_v1",
        "status": status,
        "target_pair_key": f"{SPEAKER}/{REFERENCE}/{SOURCE}",
        "scope": {"speaker": SPEAKER, "reference_role": REFERENCE, "source_role": SOURCE, "new_audio_forwards": main_receipt.get("audio_job_count"), "max_new_audio_forwards": 2, "max_new_mfa_wavs": 2},
        "prior_main_pair": {"status": prior_pair["status"], "reason": prior_pair["local_map"]["reason"], "prior_receipt_sha256": MAIN_RECEIPT_SHA256},
        "numeric_boundary_correction": accepted_map["inclusive_boundary_classification"],
        "main_receipt": hash_record(main_receipt_path),
        "run_receipt": hash_record(run_receipt_path),
        "mfa_receipt": hash_record(mfa_receipt_path) if mfa_receipt_path.is_file() else {"status": "MISSING"},
        "score_worker_receipt": hash_record(score_receipt_path) if score_receipt_path.is_file() else {"status": "MISSING"},
        "pair_result": pair,
    }
    write_new_json(OUT / "boundary_completion_receipt.json", completion)
    print(json.dumps({"status": status, "target_pair_key": completion["target_pair_key"], "audio_job_count": main_receipt.get("audio_job_count"), "mfa_status": mfa_doc.get("status"), "score_status": score_doc.get("status"), "receipt": str((OUT / "boundary_completion_receipt.json").resolve())}, ensure_ascii=False, sort_keys=True))
    return 0 if status == "BOUNDARY_COMPLETION_PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if OUT.is_dir():
            failure = OUT / "boundary_wrapper_failure.json"
            if not failure.exists():
                write_new_json(failure, {"status": "FAILED", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "time_unix": time.time()})
        raise
