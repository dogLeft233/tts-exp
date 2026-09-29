from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, canonical_hash, file_sha256, gpu_lock, load_self_hashed, pcm16, read_json, write_json
from .media import decode_video_frames, mux_natural
from .scoring import ScoreEngine, score_metrics
from .transform import temporal_contrast


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _fixed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name, path in {"drivers": config.DRIVERS, "control_scores": config.CONTROL_SCORES, "static_faces": config.STATIC_FACES, "q_candidates": config.Q_CANDIDATES, "checkpoint": config.WAV2LIP_CHECKPOINT, "syncnet": config.SYNCNET_MODEL}.items():
        if not path.is_file() or file_sha256(path) != config.FIXED_HASHES[name]:
            raise ProtocolError(f"fixed asset changed or missing: {path}")
    return load_self_hashed(config.DRIVERS), load_self_hashed(config.CONTROL_SCORES), load_self_hashed(config.Q_CANDIDATES)


def _records(drivers: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(drivers.get("rows", []), key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len(rows) != config.RECORD_COUNT or len({str(row["source_group"]) for row in rows}) != config.GROUP_COUNT:
        raise ProtocolError("frozen cohort is not 16 records/8 groups")
    return rows


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    drivers, parent_scores, q_candidates = _fixed()
    records = _records(drivers)
    if paths.protocol.is_file():
        if not resume:
            raise ProtocolError(f"run exists; use --resume: {paths.root}")
        return load_self_hashed(paths.protocol)
    driver_rows: list[dict[str, Any]] = []
    for row in records:
        sid = str(row["sample_id"])
        n = np.asarray(np.load(Path(str(row["arms"]["N"]["path"])), allow_pickle=False), dtype=np.float32)
        transformed = temporal_contrast(n)
        mel_paths: dict[str, str] = {}
        for arm in ("N", "SMOOTH", "SHARP"):
            path = paths.root / "drivers" / f"{sid}__{arm}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, transformed[arm], allow_pickle=False)
            mel_paths[arm] = str(path.resolve())
        driver_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), "natural_audio": row["natural_audio"], "static_face": row["static_face"], "arms": {arm: {"path": mel_paths[arm], "sha256": file_sha256(mel_paths[arm]), "shape": [80, 308], "dtype": "float32"} for arm in mel_paths}, "transform": {"amplitude": float(transformed["amplitude"]), "residual_sha256": hashlib.sha256(np.asarray(transformed["residual"], dtype=np.float64).tobytes()).hexdigest(), "preclip_l2": transformed["preclip_l2"], "postclip_l2": transformed["postclip_l2"], "clip_fraction": transformed["clip_fraction"], "edge_columns_bit_exact": bool(np.array_equal(transformed["N"][:, :2], transformed["SMOOTH"][:, :2]) and np.array_equal(transformed["N"][:, -2:], transformed["SHARP"][:, -2:]))}})
    audit = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "complete", "record_count": len(driver_rows), "source_group_count": len({row["source_group"] for row in driver_rows}), "parent_assets": {"drivers": str(config.DRIVERS.resolve()), "control_scores": str(config.CONTROL_SCORES.resolve()), "q_candidates": str(config.Q_CANDIDATES.resolve())}, "rows": [{"sample_id": row["sample_id"], "source_group": row["source_group"], "natural_audio_sha256": row["natural_audio"]["sha256"], "static_face_sha256": row["static_face"]["sha256"], "mel_hashes": {arm: item["sha256"] for arm, item in row["arms"].items()}} for row in driver_rows]}
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "protocol_revision": "temporal_binomial_v1", "status": "locked", "configuration": config.configuration(), "spec_bindings": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in {"proposal": config.REPO / "openspec/changes/probe-wav2lip-natural-temporal-contrast/proposal.md", "design": config.REPO / "openspec/changes/probe-wav2lip-natural-temporal-contrast/design.md", "spec": config.REPO / "openspec/changes/probe-wav2lip-natural-temporal-contrast/specs/wav2lip-natural-temporal-contrast/spec.md", "contract": config.REPO / "openspec/parallel-replacement-probes-20260909.md"}.items()}, "records": [{"sample_id": row["sample_id"], "source_group": row["source_group"]} for row in driver_rows], "record_count": 16, "source_group_count": 8, "limits": {"fresh_videos": 34, "fresh_scores": 36, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}}
    write_json(paths.input_audit, audit)
    write_json(paths.drivers, {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "complete", "rows": driver_rows})
    write_json(paths.protocol, protocol)
    return protocol


def _parent_row(parent_scores: dict[str, Any], sid: str, video: str, audio: str) -> dict[str, Any]:
    for row in parent_scores.get("rows", []):
        if str(row.get("sample_id")) == sid and str(row.get("video_arm")) == video and str(row.get("audio_arm")) == audio:
            return row.get("score", row)
    raise ProtocolError(f"parent score missing: {sid}/{video}/{audio}")


def _gpu(paths: config.RunPaths, label: str, rows: list[dict[str, Any]], resume: bool) -> dict[str, Any]:
    plan = paths.root / "plans" / f"{label}.json"
    result = paths.root / "generation" / f"{label}.json"
    payload = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "label": label, "rows": rows}
    if plan.is_file() and read_json(plan) != payload:
        raise ProtocolError(f"GPU plan changed: {plan}")
    if not plan.is_file():
        write_json(plan, payload)
    if result.is_file():
        if resume:
            return load_self_hashed(result)
        raise ProtocolError(f"GPU result exists; use --resume: {result}")
    log = paths.root / "logs" / f"{label}.gpu.log"
    command = [str(config.WAV2LIP_PYTHON), "-m", "scripts.experiments.wav2lip_natural_temporal_contrast.gpu_worker", "--plan", str(plan), "--result", str(result)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO) + os.pathsep + env.get("PYTHONPATH", "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock(config.GPU_LOCK):
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=str(config.REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not result.is_file():
        raise ProtocolError(f"GPU worker failed; see {log}")
    value = load_self_hashed(result)
    value["command"] = command
    value["log"] = str(log.resolve())
    return write_json(result, value)


def _plan(paths: config.RunPaths, drivers: dict[str, Any], arms: tuple[str, ...], label: str, subset: int | None = None) -> list[dict[str, Any]]:
    rows = sorted(drivers["rows"], key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if subset is not None:
        rows = rows[:subset]
    return [{"sample_id": row["sample_id"], "static_face": row["static_face"]["path"], "arms": {arm: row["arms"][arm]["path"] for arm in arms}, "outputs": {arm: str((paths.root / "generation" / label / f"{row['sample_id']}__{arm}.video.mkv").resolve()) for arm in arms}} for row in rows]


def _mux_and_score(paths: config.RunPaths, generated: dict[str, Any], drivers: dict[str, Any], engine: Any, label: str, audio_arm: str = "N") -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    rows: list[dict[str, Any]] = []
    for item in generated["rows"]:
        sid, arm = str(item["sample_id"]), str(item["arm"])
        audio = Path(str(by_id[sid]["natural_audio"]["path"]))
        media = paths.root / "media" / label / f"{sid}__{arm}.mkv"
        if not media.exists():
            mux = mux_natural(Path(str(item["video_only"])), audio, media)
        else:
            mux = {"output": str(media.resolve()), "output_sha256": file_sha256(media), "video_only": item["video_only"], "video_only_sha256": item["video_only_sha256"], "audio": str(audio.resolve()), "audio_sha256": file_sha256(audio), "frame_count": config.FRAME_COUNT}
        score = engine.score(media, audio, paths.root / "scores" / label / f"{sid}__{arm}", sample_id=sid, video_arm=arm, audio_arm=audio_arm)
        rows.append({"sample_id": sid, "video_arm": arm, "audio_arm": audio_arm, "media": mux, "score": score, "fresh": True})
    return rows


def _controls(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], parent_scores: dict[str, Any], resume: bool) -> dict[str, Any]:
    if paths.control_analysis.is_file() and resume:
        return load_self_hashed(paths.control_analysis)
    engine = ScoreEngine()
    generated = _gpu(paths, "replay", _plan(paths, drivers, ("N",), "replay", config.REPLAY_COUNT), resume)
    replay_rows = _mux_and_score(paths, generated, drivers, engine, "replay")
    parity_rows: list[dict[str, Any]] = []
    for parent in [row for row in parent_scores["rows"] if str(row.get("video_arm", "")).startswith("PARITY_")]:
        pscore = parent["score"]
        sid = str(pscore["sample_id"])
        fresh = engine.score(Path(str(pscore["media"])), Path(str(pscore["source_audio"])), paths.root / "scores" / "parity" / str(parent["parity"]["label"]), sample_id=sid, video_arm=str(parent["video_arm"]), audio_arm="N", expected_rows=None)
        actual = np.asarray(np.load(fresh["matrix"], allow_pickle=False), dtype=np.float64)
        reference = np.asarray(np.load(str(parent["parity"]["reference_matrix"]), allow_pickle=False), dtype=np.float64)
        parity_rows.append({"sample_id": sid, "video_arm": parent["video_arm"], "audio_arm": "N", "parity": {"matrix_max_abs": float(np.max(np.abs(actual - reference))), "offset_equal": score_metrics(actual)["offset"] == score_metrics(reference)["offset"], "passes": bool(actual.shape == reference.shape and np.max(np.abs(actual - reference)) <= 1e-4 and score_metrics(actual)["offset"] == score_metrics(reference)["offset"])}, "score": fresh, "fresh": True})
    replay_pass = []
    for row in replay_rows:
        parent = _parent_row(parent_scores, row["sample_id"], "N", "N")
        fresh_frames = decode_video_frames(Path(str(row["media"]["output"])))
        parent_frames = decode_video_frames(Path(str(parent["media"])))
        fresh_matrix = np.asarray(np.load(row["score"]["matrix"], allow_pickle=False), dtype=np.float64)
        parent_matrix = np.asarray(np.load(parent["matrix"], allow_pickle=False), dtype=np.float64)
        passed = bool(len(fresh_frames) == len(parent_frames) and all(np.array_equal(a, b) for a, b in zip(fresh_frames, parent_frames, strict=True)) and np.max(np.abs(fresh_matrix - parent_matrix)) <= 1e-4 and row["score"]["source_pcm_sha256"] == parent["source_pcm_sha256"])
        row["replay"] = {"pixel_equal": passed, "matrix_max_abs": float(np.max(np.abs(fresh_matrix - parent_matrix))), "pcm_equal": row["score"]["source_pcm_sha256"] == parent["source_pcm_sha256"], "passes": passed}
        replay_pass.append(passed)
    passed = bool(len(replay_rows) == 2 and all(replay_pass) and len(parity_rows) == 2 and all(row["parity"]["passes"] for row in parity_rows))
    control = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "engineering_status": "GO", "control_pass": passed, "scientific_decision": "CONTROL_PASS" if passed else "CONTROL_FAILED", "fresh_video_count": 2, "fresh_score_count": 4, "replay_rows": replay_rows, "parity_rows": parity_rows, "parent_control_manifest": str(config.CONTROL_SCORES.resolve()), "parent_control_manifest_sha256": config.FIXED_HASHES["control_scores"], "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}
    return write_json(paths.control_analysis, control) | {"replay_rows": replay_rows, "parity_rows": parity_rows}


def _candidates(paths: config.RunPaths, drivers: dict[str, Any], resume: bool) -> dict[str, Any]:
    control = load_self_hashed(paths.control_analysis)
    if not control.get("control_pass"):
        raise ProtocolError("control gate failed; refusing candidate generation")
    if not paths.validation.is_file() or load_self_hashed(paths.validation).get("status") != "PASS":
        raise ProtocolError("independent control validator is not PASS; refusing candidate generation")
    if paths.candidate_scores.is_file() and resume:
        return load_self_hashed(paths.candidate_scores)
    engine = ScoreEngine()
    generated = _gpu(paths, "candidates", _plan(paths, drivers, ("SMOOTH", "SHARP"), "candidates"), resume)
    rows = _mux_and_score(paths, generated, drivers, engine, "candidates")
    if len(rows) != config.CANDIDATE_COUNT:
        raise ProtocolError(f"candidate count changed: {len(rows)}")
    return write_json(paths.candidate_scores, {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "complete", "rows": rows, "count": len(rows)})


def _analysis(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], parent_scores: dict[str, Any], resume: bool) -> dict[str, Any]:
    if paths.analysis.is_file() and resume:
        return load_self_hashed(paths.analysis)
    control = load_self_hashed(paths.control_analysis)
    if not control.get("control_pass"):
        return write_json(paths.analysis, {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "engineering_status": "GO", "scientific_decision": "CONTROL_FAILED", "contrasts": {}, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})
    candidates = load_self_hashed(paths.candidate_scores)
    by_candidate = {(str(row["sample_id"]), str(row["video_arm"])): row["score"] for row in candidates["rows"]}
    by_driver = {str(row["sample_id"]): row for row in drivers["rows"]}
    values: dict[str, dict[str, list[float]]] = {"SMOOTH_vs_N": {m: [] for m in ("C", "D", "A")}, "SHARP_vs_N": {m: [] for m in ("C", "D", "A")}}
    groups: list[str] = []
    per_record: list[dict[str, Any]] = []
    for rec in sorted(protocol["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"]))):
        sid = str(rec["sample_id"])
        n = _parent_row(parent_scores, sid, "N", "N"); k = int(n["U"]["min_index"])
        row = {"sample_id": sid, "source_group": str(rec["source_group"]), "k0": k}
        groups.append(str(rec["source_group"]))
        for arm in ("SMOOTH", "SHARP"):
            c = by_candidate[(sid, arm)]
            delta = {"C": float(c["U"]["C"] - n["U"]["C"]), "D": float(n["U"]["D"] - c["U"]["D"]), "A": float(n["U"]["curve"][k] - c["U"]["curve"][k])}
            name = f"{arm}_vs_N"; row[name] = delta
            for metric, value in delta.items(): values[name][metric].append(value)
        per_record.append(row)
    contrasts: dict[str, Any] = {}
    for name, metric_values in values.items():
        metrics = {metric: __import__("scripts.experiments.wav2lip_natural_temporal_contrast.common", fromlist=["group_bootstrap"]).group_bootstrap(items, groups) for metric, items in metric_values.items()}
        joint = sum(all(metric_values[m][i] > 0 for m in ("C", "D", "A")) for i in range(len(groups)))
        passed = bool(all(metrics[m]["ci99"][0] > 0 for m in ("C", "D", "A")) and metrics["C"]["mean"] > 0.05 and joint >= 7)
        contrasts[name] = {"metrics": metrics, "group_joint_positive_count": int(joint), "pass": passed}
    if all(item["pass"] for item in contrasts.values()): decision = "BIDIRECTIONAL_SIGNAL_MECHANISM_UNRESOLVED"
    elif contrasts["SMOOTH_vs_N"]["pass"]: decision = "SMOOTHING_SIGNAL_TO_CONFIRM"
    elif contrasts["SHARP_vs_N"]["pass"]: decision = "SHARPENING_SIGNAL_TO_CONFIRM"
    else: decision = "NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED"
    return write_json(paths.analysis, {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "engineering_status": "GO", "scientific_decision": decision, "contrasts": contrasts, "per_record": per_record, "candidate_count": len(candidates["rows"]), "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False})


def _finalize(paths: config.RunPaths, stage: str) -> dict[str, Any]:
    analysis = load_self_hashed(paths.analysis) if paths.analysis.is_file() else {"scientific_decision": "BLOCKED"}
    validation = {"status": "PENDING", "stage": stage}
    write_json(paths.validation, validation)
    review = write_json(paths.review, {"schema_version": 1, "status": "self_reviewed", "scope": "A natural temporal contrast", "checks": ["fixed input hashes", "control gate before candidates", "both directions reported", "no training authorization"], "scientific_decision": analysis.get("scientific_decision")})
    final = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "complete", "engineering_status": "GO", "scientific_decision": analysis.get("scientific_decision", "BLOCKED"), "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "review_sha256": review["artifact_sha256"]}
    write_json(paths.final, final)
    paths.result.write_text(f"# Wav2Lip natural temporal contrast\\n\\n- engineering: GO\\n- decision: {final['scientific_decision']}\\n- replacement_confirmed: false\\n", encoding="utf-8")
    return final


def run(options: argparse.Namespace) -> int:
    paths = _paths(options.run_id)
    try:
        protocol = _prepare(paths, options.resume)
        drivers = load_self_hashed(paths.drivers); _, parent_scores, _ = _fixed()
        if options.stage in ("controls", "candidates", "analyze", "all"):
            _controls(paths, protocol, drivers, parent_scores, options.resume)
            if options.stage in ("candidates", "analyze", "all"):
                subprocess.run([sys.executable, "-m", "scripts.experiments.wav2lip_natural_temporal_contrast.validate", "--run-root", str(paths.root), "--stage", "controls"], cwd=str(config.REPO), check=True)
                if not load_self_hashed(paths.control_analysis).get("control_pass") and options.stage == "all":
                    _analysis(paths, protocol, drivers, parent_scores, options.resume)
                    _finalize(paths, options.stage)
                    subprocess.run([sys.executable, "-m", "scripts.experiments.wav2lip_natural_temporal_contrast.validate", "--run-root", str(paths.root), "--stage", "all"], cwd=str(config.REPO), check=True)
                    return 0
        if options.stage in ("candidates", "analyze", "all"):
            _candidates(paths, drivers, options.resume)
        if options.stage in ("analyze", "all"):
            _analysis(paths, protocol, drivers, parent_scores, options.resume)
            _finalize(paths, options.stage)
            subprocess.run([sys.executable, "-m", "scripts.experiments.wav2lip_natural_temporal_contrast.validate", "--run-root", str(paths.root), "--stage", "all"], cwd=str(config.REPO), check=True)
        return 0
    except Exception as exc:
        write_json(paths.p("error.json"), {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "controls", "candidates", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
