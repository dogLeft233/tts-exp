from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config, construct


def _fixed() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in ((config.P_DRIVERS, config.P_DRIVERS_SHA256), (config.P_CONTROL, config.P_CONTROL_SHA256), (config.WAV2LIP_CHECKPOINT, config.WAV2LIP_CHECKPOINT_SHA256), (config.SYNCNET_MODEL, config.SYNCNET_MODEL_SHA256)):
        if not path.is_file() or rt.file_sha256(path) != expected:
            raise rt.ProtocolError(f"fixed asset changed or missing: {path}")
    return rt.load_self(config.P_DRIVERS), rt.load_self(config.P_CONTROL)


def _records(drivers: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(drivers.get("rows", []), key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len(rows) != 16 or len({str(row["source_group"]) for row in rows}) != 8:
        raise rt.ProtocolError("frozen cohort is not 16 records/8 groups")
    return rows


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    drivers, _ = _fixed()
    if paths.protocol.is_file():
        if not resume:
            raise rt.ProtocolError("run exists; use --resume")
        return rt.load_self(paths.protocol)
    rows = _records(drivers)
    audit_rows: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    degenerate: list[str] = []
    for row in rows:
        sid = str(row["sample_id"])
        natural_audio = Path(str(row["natural_audio"]["path"]))
        if rt.file_sha256(natural_audio) != str(row["natural_audio"]["sha256"]):
            raise rt.ProtocolError(f"natural audio changed: {sid}")
        raw = rt.source_pcm16(natural_audio)
        pcm = rt.pcm_array(raw)
        if pcm.size < config.SUPPORT_SAMPLES:
            raise rt.ProtocolError(f"natural audio lacks first 61440 samples: {sid}")
        natural_mel_path = Path(str(row["arms"]["N"]["path"]))
        if rt.file_sha256(natural_mel_path) != str(row["arms"]["N"]["sha256"]):
            raise rt.ProtocolError(f"natural mel changed: {sid}")
        natural_mel = rt.load_mel(natural_mel_path)
        recomputed_mel = rt.natural_mel_from_pcm(raw)
        mel_error = float(np.max(np.abs(recomputed_mel.astype(np.float64) - natural_mel.astype(np.float64))))
        if mel_error > 1e-6:
            raise rt.ProtocolError(f"natural mel binding mismatch: {sid}: {mel_error}")
        result = construct.construct_gains(pcm)
        if result["status"] != "GO":
            degenerate.append(sid)
            continue
        arms: dict[str, dict[str, Any]] = {"N": {"path": str(natural_mel_path.resolve()), "sha256": rt.file_sha256(natural_mel_path)}}
        audio_rows: dict[str, Any] = {"N": {"path": str(natural_audio.resolve()), "sha256": rt.file_sha256(natural_audio), "pcm_sha256": rt.bytes_sha256(raw), "sample_count": int(pcm.size)}}
        for arm, values in (("GAIN_PLUS", result["plus"]), ("GAIN_MINUS", result["minus"])):
            audio_path = paths.root / "audio" / f"{sid}__{arm}.wav"
            rt.write_pcm16(audio_path, values)
            candidate_raw = rt.source_pcm16(audio_path)
            candidate_mel = rt.natural_mel_from_pcm(candidate_raw)
            mel_path = paths.root / "drivers" / f"{sid}__{arm}.npy"
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, candidate_mel.astype(np.float32), allow_pickle=False)
            arms[arm] = {"path": str(mel_path.resolve()), "sha256": rt.file_sha256(mel_path), "audio": str(audio_path.resolve()), "audio_sha256": rt.file_sha256(audio_path), "pcm_sha256": rt.bytes_sha256(candidate_raw)}
            audio_rows[arm] = {"path": str(audio_path.resolve()), "sha256": rt.file_sha256(audio_path), "pcm_sha256": rt.bytes_sha256(candidate_raw), "sample_count": int(values.size), "gain": float(result["gain_plus"] if arm == "GAIN_PLUS" else result["gain_minus"])}
        driver_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), "natural_audio": audio_rows["N"], "arms": arms, "gain": {key: result[key] for key in ("gain_plus", "gain_minus", "plus_db", "minus_db", "peak")}, "natural_mel_max_abs_error": mel_error, "candidate_pcm_equal_natural": {arm: bool(np.array_equal(result["plus" if arm == "GAIN_PLUS" else "minus"], pcm)) for arm in ("GAIN_PLUS", "GAIN_MINUS")}})
        audit_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), "natural_audio_sha256": rt.file_sha256(natural_audio), "natural_pcm_sha256": rt.bytes_sha256(raw), "mel_max_abs_error": mel_error, "status": "GO"})
    equal_candidate_ids = [str(row["sample_id"]) for row in driver_rows if any(row["candidate_pcm_equal_natural"].values())]
    if equal_candidate_ids:
        raise rt.ProtocolError(f"INPUT_DEGENERATE: quantized candidate equals natural PCM for {equal_candidate_ids}")
    if degenerate:
        # The rule is cohort-level: retain the IDs and do not silently attenuate or drop them.
        raise rt.ProtocolError(f"INPUT_DEGENERATE: {degenerate}")
    bindings = {name: {"path": str(path.resolve()), "sha256": rt.file_sha256(path)} for name, path in {"proposal": config.PROPOSAL, "design": config.DESIGN, "spec": config.SPEC, "parent_drivers": config.P_DRIVERS, "parent_control": config.P_CONTROL}.items()}
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "status": "locked", "run_id": paths.root.name.removeprefix("wav2lip_waveform_gain_"), "bindings": bindings, "records": [{"sample_id": row["sample_id"], "source_group": row["source_group"]} for row in driver_rows], "record_count": 16, "source_group_count": 8, "budget": {"new_videos": 34, "new_scores": 36, "training": 0}, "flags": {"replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    rt.write_json(paths.input_audit, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": audit_rows, "degenerate_ids": degenerate})
    rt.write_json(paths.audio_manifest, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": driver_rows, "evaluation_audio": "original_natural_pcm_only"})
    rt.write_json(paths.drivers, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": driver_rows})
    return rt.write_json(paths.protocol, protocol)


def _controls(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    if paths.controls.is_file() and resume:
        old = rt.load_self(paths.controls)
        if "ci95" in old.get("anchor_damage", {}): return old
    result = rt.parent_control_gate(config.P_CONTROL, config.P_DRIVERS)
    result.update({"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "candidate_authorized": result["status"] == "PASS", "new_video_count": 0, "new_score_count": 0})
    return rt.write_json(paths.controls, result)


def _plan(paths: config.RunPaths, drivers: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"])
        rows.append({"sample_id": sid, "static_face": str(next(item for item in _records(drivers) if str(item["sample_id"]) == sid)["static_face"]["path"]), "source_audio": row["natural_audio"]["path"], "arms": {arm: row["arms"][arm]["path"] for arm in ("GAIN_PLUS", "GAIN_MINUS")}, "outputs": {arm: str((paths.root / "media" / f"{sid}__{arm}.mkv").resolve()) for arm in ("GAIN_PLUS", "GAIN_MINUS")}})
    payload = {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "rows": rows}
    return rt.write_json(paths.plan, payload)


def _candidates(paths: config.RunPaths, drivers: dict[str, Any], resume: bool) -> None:
    controls = rt.load_self(paths.controls)
    if controls.get("status") != "PASS":
        raise rt.ProtocolError("CONTROL_FAILED: candidate generation is not authorized")
    plan = _plan(paths, drivers)
    generation = rt.run_gpu_plan(paths.plan, paths.generation, "wav2lip_waveform_gain", config.WAV2LIP_CHECKPOINT, config.FFMPEG, config.WAV2LIP_PYTHON)
    scores = rt.score_videos(generation["rows"], paths.root / "scores" / "candidates", config.SYNCNET_MODEL, "wav2lip_waveform_gain")
    controls = rt.fresh_control_rows(config.P_CONTROL, paths.root / "scores" / "controls", config.SYNCNET_MODEL, "wav2lip_waveform_gain")
    rt.write_json(paths.scores, {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "status": "complete", "rows": scores + controls, "count": len(scores) + len(controls), "candidate_count": len(scores), "fresh_control_count": len(controls), "evaluation_audio": "original_natural_pcm_only"})


def _parent_natural(control: dict[str, Any], sid: str) -> dict[str, Any]:
    for row in control["rows"]:
        if str(row["sample_id"]) == sid and str(row["video_arm"]) == "N" and str(row["audio_arm"]) == "N":
            _, _, matrix = rt.load_worker_arrays(row)
            return rt.score_metrics(matrix)
    raise rt.ProtocolError(f"parent natural score missing: {sid}")


def _analysis(paths: config.RunPaths, control: dict[str, Any], resume: bool) -> dict[str, Any]:
    if not paths.scores.is_file():
        raise rt.ProtocolError("candidate scores are missing")
    scores = rt.load_self(paths.scores)
    by_id = {(str(row["sample_id"]), str(row["video_arm"])): row for row in scores["rows"]}
    records = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"])
        record = {"sample_id": sid, "source_group": str(row["source_group"]), "N": _parent_natural(control, sid)}
        for arm in ("GAIN_PLUS", "GAIN_MINUS"):
            record[arm] = by_id[(sid, arm)]["score"]["full"]
        records.append(record)
    contrasts = {arm: rt.contrast_summary(records, arm) for arm in ("GAIN_PLUS", "GAIN_MINUS")}
    decision = "NO_WAVEFORM_GAIN_ESTABLISHED"
    passing = {arm: rt.gain_pass(contrasts[arm]) for arm in contrasts}
    if sum(passing.values()) == 1:
        decision = "AMPLIFY_SIGNAL_TO_CONFIRM" if passing["GAIN_PLUS"] else "ATTENUATE_SIGNAL_TO_CONFIRM"
    elif sum(passing.values()) == 2:
        decision = "GAIN_RESPONSE_MECHANISM_UNRESOLVED"
    return rt.write_json(paths.analysis, {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "status": "complete", "records": records, "contrasts": contrasts, "passing": passing, "scientific_decision": decision, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})


def _finalize(paths: config.RunPaths, validation: dict[str, Any], analysis: dict[str, Any]) -> None:
    review = rt.write_json(paths.review, {"schema_version": 1, "status": "self_reviewed", "checks": ["PCM16 construction", "original natural evaluation audio", "shared 99% grouped statistics", "candidate flags remain false"], "independent_validator": True})
    actual_budget = {"new_videos": len(rt.load_self(paths.generation)["rows"]), "new_scores": len(rt.load_self(paths.scores)["rows"])}
    final = rt.write_json(paths.final, {"schema_version": 1, "status": "complete", "engineering_status": "GO", "scientific_decision": analysis["scientific_decision"], "analysis_sha256": analysis["artifact_sha256"], "validation": str(paths.validation.resolve()), "validation_sha256": validation["artifact_sha256"], "review_sha256": review["artifact_sha256"], "actual_budget": actual_budget, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})
    lines = ["# Wav2Lip waveform gain probe", "", f"- decision: `{final['scientific_decision']}`", f"- engineering: `{final['engineering_status']}`", "- evaluation audio: `original natural PCM`", f"- actual budget: `{actual_budget['new_videos']} videos / {actual_budget['new_scores']} score cells`", "- replacement_confirmed: `false`", ""]
    for arm, summary in analysis["contrasts"].items():
        lines.append(f"- {arm}: ΔC={summary['metrics']['C']['mean']:.6f}, CI99=[{summary['metrics']['C']['ci99'][0]:.6f}, {summary['metrics']['C']['ci99'][1]:.6f}], joint_positive={summary['joint_positive_count']}")
    paths.result.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = config.RunPaths(config.run_root_for(run_id)); paths.root.mkdir(parents=True, exist_ok=True)
    try:
        protocol = _prepare(paths, resume)
        drivers, control = _fixed()
        if stage == "prepare": return 0
        controls = _controls(paths, resume)
        if stage == "controls": return 0
        if controls["status"] != "PASS": return 1
        if stage == "analyze":
            analysis = _analysis(paths, control, resume)
            from .validate import validate
            validation = validate(paths.root)
            _finalize(paths, validation, analysis)
            paths.root.joinpath("error.json").unlink(missing_ok=True)
            return 0
        _candidates(paths, drivers, resume)
        if stage == "candidates": return 0
        analysis = _analysis(paths, control, resume)
        from .validate import validate
        validation = validate(paths.root)
        _finalize(paths, validation, analysis)
        paths.root.joinpath("error.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:
        rt.write_json(paths.root / "error.json", {"schema_version": 1, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "controls", "candidates", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run(args.run_id, args.stage, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
