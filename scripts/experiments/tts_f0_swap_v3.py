#!/usr/bin/env python3
"""Run the paired-Harvest F0 repair and the frozen Wav2Lip experiment.

The v2 pilot was blocked because DIO disagreed with Harvest on the raw audio
before WORLD synthesis.  This runner keeps DIO as a diagnostic, but validates
the intervention against the same Harvest measurement used for the WORLD
carrier and against the measured ID resynthesis baseline.  The input list is
copied from the frozen v1 parent; no sample is selected from a score.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_f0_swap as core  # noqa: E402


PROTOCOL_ID = "tts_f0_swap_v3_paired_harvest"
PROTOCOL_REVISION = PROTOCOL_ID
MEASUREMENT_MODE = "paired_harvest"
PARENT_RUN = REPO / "runs/tts_f0_swap_f0spec_audit2"
SPEC_PATH = REPO / "basic-memory/Research/TTS 音高起伏交换的配对 Harvest 测量修复与 Wav2Lip 执行 Spec.md"
RECOMPUTE = Path(__file__).with_name("tts_f0_swap_recompute.py")


class V3Error(RuntimeError):
    """A frozen v3 protocol or artifact is invalid."""


def _install_protocol_globals() -> None:
    """Make shared render/score helpers emit the v3 fingerprint."""

    core.PROTOCOL_ID = PROTOCOL_ID
    core.PROTOCOL_REVISION = PROTOCOL_REVISION
    core.SPEC_PATH = SPEC_PATH


_install_protocol_globals()


def run_root(run_id: str) -> Path:
    return REPO / "runs" / f"tts_f0_swap_v3_{core.run_id_valid(run_id)}"


def _read_self_hashed(path: Path) -> dict[str, Any]:
    try:
        return core.verify_json(path, self_hash=True)
    except Exception as exc:  # noqa: BLE001 - add the path to the error
        raise V3Error(f"invalid self-hashed JSON: {path}: {exc}") from exc


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    core.write_json(path, payload, self_hash=True)


def _copy_frozen_inputs(paths: core.RunPaths) -> dict[str, Any]:
    source = PARENT_RUN / "inputs.json"
    if not source.is_file():
        raise V3Error(f"frozen parent inputs are missing: {source}")
    parent = _read_self_hashed(source)
    if int(parent.get("pilot_pair_count", -1)) != 4 or int(parent.get("formal_pair_count", -1)) != 24:
        raise V3Error("parent input denominator is not 4 pilot / 24 formal")
    if set(parent.get("pilot_keys", [])) != {
        "aishell1_test_400__BAC009S0765W0312",
        "aishell1_test_400__BAC009S0770W0414",
        "aishell1_test_400__BAC009S0901W0487",
        "aishell1_test_400__BAC009S0906W0401",
    }:
        raise V3Error("parent pilot keys changed")
    records = parent.get("records", [])
    if len(records) != 28 or len({str(row.get("paired_key")) for row in records}) != 28:
        raise V3Error("parent records are not exactly 28 unique rows")
    # Recheck the audio/textgrid bindings before copying the list.  This is
    # intentionally independent of the v3 measurement choice.
    for row in records:
        for field, hash_field in (
            ("natural_audio", "natural_container_sha256"),
            ("tts_audio", "tts_container_sha256"),
            ("natural_textgrid", "natural_textgrid_sha256"),
            ("tts_textgrid", "tts_textgrid_sha256"),
        ):
            path = Path(str(row.get(field, "")))
            if not path.is_file() or core.file_sha256(path) != str(row.get(hash_field, "")):
                raise V3Error(f"parent input hash mismatch: {row.get('paired_key')}/{field}")
    if paths.inputs.is_file():
        existing = _read_self_hashed(paths.inputs)
        if existing.get("source_parent_inputs_sha256") != core.file_sha256(source):
            raise V3Error("existing v3 inputs bind to a different parent")
        return existing
    payload = dict(parent)
    payload.pop("artifact_sha256", None)
    payload.update(
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "source_parent_run": str(PARENT_RUN.resolve()),
            "source_parent_inputs_sha256": core.file_sha256(source),
            "measurement_mode": MEASUREMENT_MODE,
            "selection_frozen": True,
            "v3_created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write(paths.inputs, payload)
    return _read_self_hashed(paths.inputs)


def _write_protocol(paths: core.RunPaths, inputs: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "status": "complete",
        "parent_run": str(PARENT_RUN.resolve()),
        "parent_inputs_sha256": inputs.get("source_parent_inputs_sha256"),
        "selection": "exact copy of frozen v1 inputs; no score-based reselection",
        "measurement_mode": MEASUREMENT_MODE,
        "hard_gates": {
            "pilot_required_pairs": 3,
            "pilot_denominator": 4,
            "harvest_search_f0_hz": [core.F0_FLOOR, core.F0_CEIL],
            "post_stonemask_f0_hz": [core.F0_POST_STONEMASK_FLOOR, core.F0_CEIL],
            "id_coverage_vs_h_source": 0.80,
            "id_f0_median_st": 1.0,
            "id_f0_p90_st": 3.0,
            "candidate_support_coverage": 0.80,
            "candidate_delta_median_st": 1.0,
            "candidate_delta_p90_st": 3.0,
        },
        "cross_check": "DIO->StoneMask is diagnostic only; it is never used as the v3 gate reference",
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": core.file_sha256(Path(__file__)),
        "core_runner": str(Path(core.__file__).resolve()),
        "core_runner_sha256": core.file_sha256(Path(core.__file__)),
        "spec": str(SPEC_PATH.resolve()),
        "spec_sha256": core.file_sha256(SPEC_PATH),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "git": core._git_commit(), "created_at": datetime.now(timezone.utc).isoformat()},
    }
    if paths.protocol.is_file():
        existing = _read_self_hashed(paths.protocol)
        if existing.get("protocol_id") != PROTOCOL_ID or existing.get("spec_sha256") != payload["spec_sha256"]:
            raise V3Error("existing protocol is stale; use a new run-id")
        if existing.get("runner_sha256") != payload["runner_sha256"] or existing.get("core_runner_sha256") != payload["core_runner_sha256"]:
            _write(paths.protocol, payload)
            return _read_self_hashed(paths.protocol)
        return existing
    _write(paths.protocol, payload)
    return _read_self_hashed(paths.protocol)


def _pilot(paths: core.RunPaths, inputs: Mapping[str, Any]) -> dict[str, Any]:
    pilot = core.run_audio_stage(paths, inputs, pilot_only=True, measurement_mode=MEASUREMENT_MODE, protocol_id=PROTOCOL_ID)
    gate = core.pilot_gate(pilot, expected_keys=[str(row["paired_key"]) for row in inputs["records"] if row.get("role") == "pilot"])
    pilot["gate"] = gate
    _write(paths.pilot_qc, pilot)
    return _read_self_hashed(paths.pilot_qc)


def _formal_audio(paths: core.RunPaths, inputs: Mapping[str, Any], pilot: Mapping[str, Any]) -> dict[str, Any]:
    gate = pilot.get("gate") or core.pilot_gate(pilot)
    if gate.get("status") != "PASS":
        payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            **core._implementation_fingerprint(),
            "status": "blocked",
            "engineering_status": "MANIPULATION_NOT_VALIDATED",
            "expected_pair_count": int(inputs.get("formal_pair_count", 24)),
            "pair_count": 0,
            "failures": [{"stage": "audio", "status": "MANIPULATION_NOT_VALIDATED", "gate": gate}],
            "manifests": [],
        }
        _write(paths.root / "audio_manifest.json", payload)
        return _read_self_hashed(paths.root / "audio_manifest.json")
    return core.run_audio_stage(paths, inputs, pilot_only=False, measurement_mode=MEASUREMENT_MODE, protocol_id=PROTOCOL_ID)


def _independent_recompute(paths: core.RunPaths) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(RECOMPUTE), "--run-dir", str(paths.root)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output = paths.root / "recompute.json"
    if not output.is_file():
        raise V3Error(f"independent recompute did not write {output}: {completed.stderr[-1000:]}")
    result = _read_self_hashed(output)
    if completed.returncode != 0 or result.get("status") != "PASS":
        raise V3Error(f"independent recompute failed: {result.get('reason', result.get('status'))}")
    return result


def _validate(paths: core.RunPaths, inputs: Mapping[str, Any], *, recompute: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = core.validate_run(paths)
    errors = list(base.get("errors", []))
    warnings = list(base.get("warnings", []))
    if base.get("status") != "PASS":
        errors.append("shared core validation failed")
    pilot = _read_self_hashed(paths.pilot_qc) if paths.pilot_qc.is_file() else {}
    gate = pilot.get("gate", {})
    if gate.get("pilot_pair_count") != 4 or gate.get("required_pairs") != 3:
        errors.append("pilot gate denominator/threshold mismatch")
    if paths.inputs.is_file():
        frozen = _read_self_hashed(paths.inputs)
        if frozen.get("source_parent_inputs_sha256") != inputs.get("source_parent_inputs_sha256"):
            errors.append("frozen parent binding changed")
        if frozen.get("measurement_mode") != MEASUREMENT_MODE:
            errors.append("measurement mode binding missing")
    audio = _read_self_hashed(paths.root / "audio_manifest.json") if (paths.root / "audio_manifest.json").is_file() else {}
    if gate.get("status") == "PASS":
        formal_count = int(inputs.get("formal_pair_count", 24))
        if audio.get("protocol_id") != PROTOCOL_ID or int(audio.get("expected_pair_count", -1)) != formal_count:
            errors.append("formal audio denominator/protocol mismatch")
        if len(audio.get("manifests", [])) != formal_count:
            errors.append("formal audio manifest is incomplete")
        if paths.videos_manifest.is_file():
            videos = _read_self_hashed(paths.videos_manifest)
            if videos.get("status") != "complete":
                errors.append("formal render is incomplete")
            # The shared renderer intentionally keeps RAW/ID for an audio-QC
            # failure and omits only that receiver's candidate arms.  Check
            # the exact QC-conditioned denominator instead of accepting the
            # old coarse four-videos-per-pair lower bound.
            qc_by_key_receiver: dict[tuple[str, str], Mapping[str, str]] = {}
            if paths.audio_qc.is_file():
                with paths.audio_qc.open(encoding="utf-8", newline="") as handle:
                    qc_by_key_receiver = {
                        (str(row.get("paired_key")), str(row.get("receiver"))): row
                        for row in csv.DictReader(handle)
                    }
            expected_video_arms: set[tuple[str, str]] = set()
            for record in inputs.get("records", []):
                if record.get("role") != "formal":
                    continue
                key = str(record["paired_key"])
                for receiver in ("N", "T"):
                    expected_video_arms.update({(key, f"{receiver}_RAW"), (key, f"{receiver}_ID")})
                    qc = qc_by_key_receiver.get((key, receiver), {})
                    if str(qc.get("status")) == "PASS":
                        expected_video_arms.update({(key, f"{receiver}_LEVEL"), (key, f"{receiver}_CONTOUR")})
            actual_video_arms = {
                (str(row.get("paired_key")), str(row.get("video_arm")))
                for row in videos.get("videos", [])
            }
            missing_video_arms = sorted(expected_video_arms - actual_video_arms)
            if missing_video_arms:
                errors.append(f"render missing QC-conditioned arms: {missing_video_arms[:5]}")
            qc_failures = [
                (str(row.get("paired_key")), str(row.get("receiver")), str(row.get("reasons", "")))
                for row in qc_by_key_receiver.values()
                if str(row.get("status")) != "PASS"
            ]
            if qc_failures:
                warnings.append(f"formal audio QC exclusions: {len(qc_failures)} receiver rows")
        else:
            errors.append("videos_manifest.json missing after an allowed pilot")
        if paths.score_manifest.is_file():
            scores = _read_self_hashed(paths.score_manifest)
            if scores.get("status") != "complete":
                errors.append("formal SyncNet score stage is incomplete")
            elif scores.get("engineering_status") == "INSUFFICIENT_SUPPORT":
                warnings.append("SyncNet score support is below the preregistered 50-window gate for some receiver rows")
            elif scores.get("engineering_status") != "PASS":
                errors.append("formal SyncNet score stage has an unclassified engineering status")
        else:
            errors.append("scores_manifest.json missing after an allowed pilot")
        if paths.analysis.is_file():
            analysis = _read_self_hashed(paths.analysis)
            if analysis.get("status") != "complete":
                errors.append("analysis is incomplete")
            if int(analysis.get("main_pair_count", 0)) < core.MIN_FORMAL_PAIRS:
                warnings.append("complete-case formal pairs are below the preregistered 18-pair support")
            if analysis.get("scientific_status") == "INSUFFICIENT_SUPPORT":
                warnings.append("scientific status is INSUFFICIENT_SUPPORT; no directional claim is licensed")
        else:
            errors.append("analysis.json missing after an allowed pilot")
        if recompute is None or recompute.get("status") != "PASS":
            errors.append("independent recompute did not pass")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "pilot_gate": gate,
        "formal_audio_status": audio.get("engineering_status", audio.get("status")),
        "independent_recompute": "PASS" if recompute and recompute.get("status") == "PASS" else "NOT_RUN",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(paths.validation, payload)
    return _read_self_hashed(paths.validation)


def _append_v3_report(paths: core.RunPaths, validation: Mapping[str, Any]) -> None:
    base = paths.report.read_text(encoding="utf-8") if paths.report.is_file() else f"# {PROTOCOL_ID}\n"
    extra = [
        "",
        "## v3 测量修复",
        "",
        "- 生产验收参考：H_ID/H_LEVEL/H_CONTOUR（同一 Harvest 测量）；DIO 轨迹仅作交叉诊断。",
        "- 这是为了避免把 RAW 阶段的 Harvest/DIO 检测器分歧当作 F0 操作失败；目标变化误差、共同支持、RMS、长度和 PCM16 门未放宽。Harvest 搜索仍为 60–600 Hz，StoneMask 精修后的固定验收下限为 50 Hz，未做 clip。",
        f"- validation：{validation.get('status')}；independent recompute：{validation.get('independent_recompute')}",
    ]
    paths.report.write_text(base.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


def run_stage(run_id: str, stage: str, *, device: str = "cuda") -> dict[str, Any]:
    _install_protocol_globals()
    paths = core.RunPaths(run_root(run_id))
    paths.root.mkdir(parents=True, exist_ok=True)
    inputs = _copy_frozen_inputs(paths)
    _write_protocol(paths, inputs)
    if stage == "audit":
        return inputs
    if stage == "pilot":
        return _pilot(paths, inputs)
    if stage == "audio":
        pilot = _read_self_hashed(paths.pilot_qc) if paths.pilot_qc.is_file() else _pilot(paths, inputs)
        return _formal_audio(paths, inputs, pilot)
    if stage == "render":
        audio = _read_self_hashed(paths.root / "audio_manifest.json")
        return core.run_render_stage(paths, inputs, audio, device=device)
    if stage == "score":
        videos = _read_self_hashed(paths.videos_manifest)
        audio = _read_self_hashed(paths.root / "audio_manifest.json")
        controls = core.run_controls(paths, inputs, audio, videos, device=device)
        result = core.run_score_stage(paths, inputs, videos, device=device)
        result["controls"] = controls
        _write(paths.score_manifest, result)
        return _read_self_hashed(paths.score_manifest)
    if stage == "analyze":
        scores = _read_self_hashed(paths.score_manifest)
        audio = _read_self_hashed(paths.root / "audio_manifest.json")
        controls = _read_self_hashed(paths.root / "controls.json") if (paths.root / "controls.json").is_file() else None
        return core.analyze_scores(paths, inputs, scores, audio, controls=controls)
    if stage == "validate":
        recompute = _independent_recompute(paths) if paths.score_manifest.is_file() and paths.analysis.is_file() else None
        return _validate(paths, inputs, recompute=recompute)
    if stage == "report":
        result = core.write_report(paths)
        validation = _read_self_hashed(paths.validation) if paths.validation.is_file() else {"status": "NOT_RUN", "independent_recompute": "NOT_RUN"}
        _append_v3_report(paths, validation)
        return result
    raise ValueError(f"unknown stage: {stage}")


def run_all(run_id: str, *, device: str = "cuda") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("audit", "pilot", "audio", "render", "score", "analyze", "validate", "report"):
        try:
            result[stage] = run_stage(run_id, stage, device=device)
        except Exception as exc:  # noqa: BLE001 - leave stage evidence on disk
            result[stage] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            paths = core.RunPaths(run_root(run_id))
            _write(paths.root / f"{stage}_error.json", result[stage])
            if stage in {"audit", "pilot", "audio"}:
                break
        if stage == "pilot" and result.get("pilot", {}).get("gate", {}).get("status") != "PASS":
            break
    paths = core.RunPaths(run_root(run_id))
    try:
        if "validate" not in result:
            result["validate"] = run_stage(run_id, "validate", device=device)
        if "report" not in result:
            result["report"] = run_stage(run_id, "report", device=device)
    except Exception as exc:  # pragma: no cover - preserve a readable failure
        result["report"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        _write(paths.root / "report_error.json", result["report"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("audit", "pilot", "audio", "render", "score", "analyze", "validate", "report", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        result = run_all(args.run_id, device=args.device) if args.stage == "all" else run_stage(args.run_id, args.stage, device=args.device)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
