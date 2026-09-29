from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config, construct


def _fixed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in ((config.P_DRIVERS, config.P_DRIVERS_SHA256), (config.P_CONTROL, config.P_CONTROL_SHA256), (config.Q_SCORES, config.Q_SCORES_SHA256), (config.D_DRIVERS, config.D_DRIVERS_SHA256), (config.D_MASKS, config.D_MASKS_SHA256), (config.D_RECONSTRUCTION, config.D_RECONSTRUCTION_SHA256), (config.WAV2LIP_CHECKPOINT, config.WAV2LIP_CHECKPOINT_SHA256), (config.SYNCNET_MODEL, config.SYNCNET_MODEL_SHA256)):
        if not path.is_file() or rt.file_sha256(path) != expected: raise rt.ProtocolError(f"fixed asset changed or missing: {path}")
    return rt.load_self(config.P_DRIVERS), rt.load_self(config.P_CONTROL), rt.load_self(config.Q_SCORES), rt.read_json(config.D_DRIVERS)


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    p_drivers, _, _, d_payload = _fixed()
    if paths.protocol.is_file():
        if not resume: raise rt.ProtocolError("run exists; use --resume")
        return rt.load_self(paths.protocol)
    masks = {str(item["mask_sha256"]): item for item in rt.read_json(config.D_MASKS)["masks"]}; records = []; audit = []
    for parent in sorted(p_drivers["rows"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))):
        sid = str(parent["sample_id"]); natural_path = Path(str(parent["arms"]["N"]["path"])); natural = rt.load_mel(natural_path); result = construct.construct_factorial(natural, d_payload["drivers"], masks, sid); arms = {}
        for arm in config.ARMS:
            path = paths.root / "drivers" / f"{sid}__{arm}.npy"; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, result[arm], allow_pickle=False); arms[arm] = {"path": str(path.resolve()), "sha256": rt.file_sha256(path), "shape": list(result[arm].shape), "dtype": str(result[arm].dtype)}
        records.append({"sample_id": sid, "source_group": str(parent["source_group"]), "static_face": parent["static_face"], "source_audio": parent["natural_audio"], "arms": arms, "construction": result["metadata"]}); audit.append({"sample_id": sid, "source_group": str(parent["source_group"]), "K_count": len(result["metadata"]["K"]), "interaction_norm_valid": result["metadata"]["interaction_norm_valid"]})
    bindings = {name: {"path": str(path.resolve()), "sha256": rt.file_sha256(path)} for name, path in {"proposal": config.PROPOSAL, "design": config.DESIGN, "spec": config.SPEC, "parent_drivers": config.P_DRIVERS, "parent_control": config.P_CONTROL, "q_scores": config.Q_SCORES, "d_drivers": config.D_DRIVERS, "mask_manifest": config.D_MASKS, "reconstruction": config.D_RECONSTRUCTION}.items()}
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "status": "locked", "run_id": paths.root.name.removeprefix("wav2lip_reconstruction_base_interaction_"), "bindings": bindings, "records": [{"sample_id": row["sample_id"], "source_group": row["source_group"]} for row in records], "record_count": 16, "source_group_count": 8, "budget": {"new_videos": 50, "new_scores": 52}, "flags": {"replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    rt.write_json(paths.input_audit, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": audit})
    rt.write_json(paths.drivers, {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete", "rows": records})
    return rt.write_json(paths.protocol, protocol)


def _controls(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    if paths.controls.is_file() and resume:
        old = rt.load_self(paths.controls)
        if "ci95" in old.get("anchor_damage", {}): return old
    value = rt.parent_control_gate(config.P_CONTROL, config.P_DRIVERS); value.update({"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "candidate_authorized": value["status"] == "PASS"}); return rt.write_json(paths.controls, value)


def _plan(paths: config.RunPaths, parent: dict[str, Any]) -> dict[str, Any]:
    p_by_id = {str(row["sample_id"]): row for row in parent["rows"]}; rows = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"]); p = p_by_id[sid]; arms = {arm: row["arms"][arm]["path"] for arm in ("BASE", "BASE_CONTENT", "BASE_WRONG")}; rows.append({"sample_id": sid, "static_face": p["static_face"]["path"], "source_audio": p["natural_audio"]["path"], "arms": arms, "outputs": {arm: str((paths.root / "media" / f"{sid}__{arm}.mkv").resolve()) for arm in arms}})
    return rt.write_json(paths.plan, {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "rows": rows})


def _candidates(paths: config.RunPaths, parent: dict[str, Any]) -> None:
    if rt.load_self(paths.controls).get("status") != "PASS": raise rt.ProtocolError("CONTROL_FAILED")
    _plan(paths, parent)
    generation = rt.run_gpu_plan(paths.plan, paths.generation, "wav2lip_reconstruction_base_interaction", config.WAV2LIP_CHECKPOINT, config.FFMPEG, config.WAV2LIP_PYTHON)
    scores = rt.score_videos(generation["rows"], paths.root / "scores" / "candidates", config.SYNCNET_MODEL, "wav2lip_reconstruction_base_interaction")
    controls = rt.fresh_control_rows(config.P_CONTROL, paths.root / "scores" / "controls", config.SYNCNET_MODEL, "wav2lip_reconstruction_base_interaction")
    rt.write_json(paths.scores, {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "status": "complete", "rows": scores + controls, "count": len(scores) + len(controls), "candidate_count": len(scores), "fresh_control_count": len(controls), "evaluation_audio": "original_natural_pcm_only"})


def _natural(control: dict[str, Any], sid: str) -> dict[str, Any]:
    row = next(item for item in control["rows"] if str(item["sample_id"]) == sid and str(item["video_arm"]) == "N" and str(item["audio_arm"]) == "N"); _, _, matrix = rt.load_worker_arrays(row); return rt.score_metrics(matrix)


def _q_n_content(q_scores: dict[str, Any], sid: str) -> dict[str, Any]:
    row = next(item for item in q_scores["rows"] if str(item["sample_id"]) == sid and str(item["video_arm"]) == "CORRECT" and str(item["audio_arm"]) == "N"); _, _, matrix = rt.load_worker_arrays(row); return rt.score_metrics(matrix)


def _pair(records: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    temporary = [{"sample_id": row["sample_id"], "source_group": row["source_group"], "N": row[right], "LEFT": row[left]} for row in records]
    return rt.contrast_summary(temporary, "LEFT")


def _interaction(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = []; groups = [str(row["source_group"]) for row in records]
    for row in records:
        k = int(row["N"]["min_index"]); g_n = row["N"]["curve"][k] - row["N_CONTENT"]["curve"][k]; g_b = row["BASE"]["curve"][k] - row["BASE_CONTENT"]["curve"][k]; values.append(g_b - g_n)
    labels = sorted(set(groups)); indices = rt.bootstrap_indices(labels); stats = rt.grouped_stats(values, groups, indices); joint = int(sum(float(np.mean([value for value, group in zip(values, groups, strict=True) if group == label])) > 0 for label in labels)); return {"values": values, "stats": stats, "joint_positive_count": joint, "signal": bool(stats["mean"] > 0.05 and stats["ci99"][0] > 0 and joint >= 7)}


def _analysis(paths: config.RunPaths, control: dict[str, Any], q_scores: dict[str, Any]) -> dict[str, Any]:
    score_map = {(str(row["sample_id"]), str(row["video_arm"])): row for row in rt.load_self(paths.scores)["rows"]}; records = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"]); item = {"sample_id": sid, "source_group": str(row["source_group"]), "N": _natural(control, sid), "N_CONTENT": _q_n_content(q_scores, sid)}
        for arm in ("BASE", "BASE_CONTENT", "BASE_WRONG"): item[arm] = score_map[(sid, arm)]["score"]["full"]
        records.append(item)
    contrasts = {(left, right): _pair(records, left, right) for left, right in (("BASE", "N"), ("BASE_CONTENT", "N"), ("BASE_WRONG", "N"), ("BASE_CONTENT", "BASE"), ("BASE_CONTENT", "BASE_WRONG"))}; interaction = _interaction(records); main_pass = rt.gain_pass(contrasts[("BASE_CONTENT", "N")]); mechanism_pass = bool(interaction["signal"] and rt.gain_pass(contrasts[("BASE_CONTENT", "BASE_WRONG")]))
    decision = "NO_BASE_CONTENT_GAIN_ESTABLISHED" if not main_pass else "BASE_DEPENDENT_CONTENT_SIGNAL_TO_CONFIRM" if mechanism_pass else "NATURAL_GAIN_MECHANISM_UNRESOLVED"
    return rt.write_json(paths.analysis, {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "status": "complete", "records": records, "contrasts": {f"{left}_vs_{right}": value for (left, right), value in contrasts.items()}, "interaction": interaction, "main_pass": main_pass, "scientific_decision": decision, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})


def _finalize(paths: config.RunPaths, validation: dict[str, Any], analysis: dict[str, Any]) -> None:
    actual_budget = {"new_videos": len(rt.load_self(paths.generation)["rows"]), "new_scores": len(rt.load_self(paths.scores)["rows"])}
    review = rt.write_json(paths.review, {"schema_version": 1, "status": "self_reviewed", "checks": ["same fixed content increment", "natural and reconstructed bases separated", "interaction not substituted for replacement", "independent validator"]}); final = rt.write_json(paths.final, {"schema_version": 1, "status": "complete", "engineering_status": "GO", "scientific_decision": analysis["scientific_decision"], "analysis_sha256": analysis["artifact_sha256"], "validation_sha256": validation["artifact_sha256"], "review_sha256": review["artifact_sha256"], "actual_budget": actual_budget, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}); paths.result.write_text(f"# Wav2Lip reconstruction-base interaction probe\n\n- decision: `{final['scientific_decision']}`\n- interaction signal: `{analysis['interaction']['signal']}`\n- interaction mean: `{analysis['interaction']['stats']['mean']:.6f}`\n- actual budget: `{actual_budget['new_videos']} videos / {actual_budget['new_scores']} score cells`\n- replacement_confirmed: `false`\n", encoding="utf-8")


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = config.RunPaths(config.run_root_for(run_id)); paths.root.mkdir(parents=True, exist_ok=True)
    try:
        _prepare(paths, resume); parent, control, q_scores, _ = _fixed()
        if stage == "prepare": return 0
        controls = _controls(paths, resume)
        if stage == "controls": return 0
        if controls["status"] != "PASS": return 1
        if stage == "analyze":
            analysis = _analysis(paths, control, q_scores)
            from .validate import validate
            validation = validate(paths.root)
            _finalize(paths, validation, analysis)
            paths.root.joinpath("error.json").unlink(missing_ok=True)
            return 0
        _candidates(paths, parent)
        if stage == "candidates": return 0
        analysis = _analysis(paths, control, q_scores)
        from .validate import validate
        validation = validate(paths.root)
        _finalize(paths, validation, analysis); paths.root.joinpath("error.json").unlink(missing_ok=True); return 0
    except Exception as exc:
        rt.write_json(paths.root / "error.json", {"schema_version": 1, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}); return 1


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--stage", choices=("prepare", "controls", "candidates", "analyze", "all"), default="all"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); return run(args.run_id, args.stage, args.resume)


if __name__ == "__main__": raise SystemExit(main())
