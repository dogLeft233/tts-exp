from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import analyze_candidates, analyze_controls
from .common import ProtocolError, file_sha256, read_json, verify_self_hashed_json, write_json_atomic, write_self_hashed_json
from .drivers import write_prepared
from .media import delay_audio, mux_and_verify
from .scoring import ScoreEngine


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _prepare(paths: config.RunPaths, resume: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if paths.protocol.is_file():
        if not resume:
            raise ProtocolError(f"run already exists; use --resume: {paths.root}")
        return _load(paths.protocol), _load(paths.driver_manifest)
    audit, protocol, driver_manifest = write_prepared(paths.root)
    print(f"PREPARE {audit['passed_count']}/{audit['record_count']} records", flush=True)
    return _load(paths.protocol), _load(paths.driver_manifest)


def _run_gpu(plan_rows: list[dict[str, Any]], paths: config.RunPaths, label: str, resume: bool) -> dict[str, Any]:
    plan_path = paths.root / "plans" / f"{label}.json"
    result_path = paths.root / "generation" / f"{label}.json"
    payload = {"schema_version": 1, "protocol_id": "wav2lip_natural_content_residual", "label": label, "rows": plan_rows}
    if plan_path.is_file():
        old = read_json(plan_path)
        if old != payload:
            raise ProtocolError(f"existing GPU plan differs: {plan_path}")
    else:
        write_json_atomic(plan_path, payload)
    if result_path.is_file():
        if not resume:
            raise ProtocolError(f"GPU result already exists; use --resume: {result_path}")
        return read_json(result_path)
    log_path = paths.root / "logs" / f"{label}.gpu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(config.REPO) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [str(config.WAV2LIP_PYTHON), "-m", "scripts.experiments.wav2lip_natural_content_residual.worker", "--plan", str(plan_path), "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG), "--result", str(result_path)]
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.REPO), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not result_path.is_file():
        raise ProtocolError(f"GPU generation failed; see {log_path}")
    generated = read_json(result_path)
    generated["command"] = command
    generated["log"] = str(log_path.resolve())
    write_json_atomic(result_path, generated)
    return generated


def _driver_rows(paths: config.RunPaths) -> list[dict[str, Any]]:
    return list(_load(paths.driver_manifest)["rows"])


def _control_plan(paths: config.RunPaths, driver_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(driver_rows):
        arms = {"N": row["arms"]["N"]["path"]}
        outputs = {"N": str((paths.root / "generation/controls" / f"{row['sample_id']}__N.video.mkv").resolve())}
        if index < 2:
            arms["N_REPEAT"] = row["arms"]["N"]["path"]
            outputs["N_REPEAT"] = str((paths.root / "generation/controls" / f"{row['sample_id']}__N_REPEAT.video.mkv").resolve())
        rows.append({"sample_id": row["sample_id"], "static_face": row["static_face"]["path"], "arms": arms, "outputs": outputs})
    return rows


def _candidate_plan(paths: config.RunPaths, driver_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in driver_rows:
        arms = {arm: row["arms"][arm]["path"] for arm in config.CANDIDATE_ARMS}
        outputs = {arm: str((paths.root / "generation/candidates" / f"{row['sample_id']}__{arm}.video.mkv").resolve()) for arm in config.CANDIDATE_ARMS}
        rows.append({"sample_id": row["sample_id"], "static_face": row["static_face"]["path"], "arms": arms, "outputs": outputs})
    return rows


def _mux_control_media(paths: config.RunPaths, driver_rows: list[dict[str, Any]], generated: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    generation = {(str(row["sample_id"]), str(row["arm"])): row for row in generated["rows"]}
    media_rows: list[dict[str, Any]] = []
    delayed_audio: dict[str, Path] = {}
    for index, row in enumerate(driver_rows):
        sample_id = str(row["sample_id"])
        natural_audio = Path(str(row["natural_audio"]["path"]))
        delayed_path = paths.root / "audio/A_DELAY" / f"{sample_id}.wav"
        if not delayed_path.is_file():
            delay_audio(natural_audio, delayed_path)
        delayed_audio[sample_id] = delayed_path
        for arm in ("N", "N_REPEAT") if index < 2 else ("N",):
            generated_row = generation[(sample_id, arm)]
            output = paths.media / "controls" / f"{sample_id}__{arm}.mkv"
            if output.is_file() and resume:
                media_row = {"sample_id": sample_id, "arm": arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": generated_row["output"], "video_only_sha256": generated_row["output_sha256"], "audio": str(natural_audio.resolve()), "audio_sha256": file_sha256(natural_audio), "audio_pcm_sha256": row["natural_audio"]["pcm_sha256"], "frame_count": config.WAV2LIP_FRAMES, "audio_arm": "N", "audio_modified": False}
            else:
                media_row = mux_and_verify(Path(str(generated_row["output"])), natural_audio, output, sample_id=sample_id, arm=arm)
                media_row["audio_arm"] = "N"
            media_rows.append(media_row)
        n_generated = generation[(sample_id, "N")]
        delay_output = paths.media / "controls" / f"{sample_id}__A_DELAY.mkv"
        if delay_output.is_file() and resume:
            delay_row = {"sample_id": sample_id, "arm": "A_DELAY", "output": str(delay_output.resolve()), "output_sha256": file_sha256(delay_output), "video_only": n_generated["output"], "video_only_sha256": n_generated["output_sha256"], "audio": str(delayed_audio[sample_id].resolve()), "audio_sha256": file_sha256(delayed_audio[sample_id]), "audio_pcm_sha256": __import__("scripts.experiments.wav2lip_natural_content_residual.common", fromlist=["source_pcm16"]).bytes_sha256(__import__("scripts.experiments.wav2lip_natural_content_residual.common", fromlist=["source_pcm16"]).source_pcm16(delayed_audio[sample_id])), "frame_count": config.WAV2LIP_FRAMES, "audio_arm": "A_DELAY", "audio_modified": True}
        else:
            delay_row = mux_and_verify(Path(str(n_generated["output"])), delayed_audio[sample_id], delay_output, sample_id=sample_id, arm="A_DELAY")
            delay_row["audio_arm"] = "A_DELAY"
        media_rows.append(delay_row)
    manifest = {"schema_version": 1, "status": "complete", "stage": "controls", "rows": media_rows, "video_count": 18, "score_count": 36}
    write_self_hashed_json(paths.root / "media/control_manifest.json", manifest)
    return media_rows


def _mux_candidate_media(paths: config.RunPaths, driver_rows: list[dict[str, Any]], generated: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    generation = {(str(row["sample_id"]), str(row["arm"])): row for row in generated["rows"]}
    media_rows: list[dict[str, Any]] = []
    for row in driver_rows:
        sample_id = str(row["sample_id"])
        natural_audio = Path(str(row["natural_audio"]["path"]))
        for arm in config.CANDIDATE_ARMS:
            generated_row = generation[(sample_id, arm)]
            output = paths.media / "candidates" / f"{sample_id}__{arm}.mkv"
            if output.is_file() and resume:
                media_rows.append({"sample_id": sample_id, "arm": arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": generated_row["output"], "video_only_sha256": generated_row["output_sha256"], "audio": str(natural_audio.resolve()), "audio_sha256": file_sha256(natural_audio), "audio_pcm_sha256": row["natural_audio"]["pcm_sha256"], "frame_count": config.WAV2LIP_FRAMES, "audio_arm": "N", "audio_modified": False})
            else:
                item = mux_and_verify(Path(str(generated_row["output"])), natural_audio, output, sample_id=sample_id, arm=arm)
                item["audio_arm"] = "N"
                media_rows.append(item)
    manifest = {"schema_version": 1, "status": "complete", "stage": "candidates", "rows": media_rows, "video_count": 48, "score_count": 48}
    write_self_hashed_json(paths.root / "media/candidate_manifest.json", manifest)
    return media_rows


def _score_parity(paths: config.RunPaths, engine: ScoreEngine) -> list[dict[str, Any]]:
    parity = read_json(config.PARITY)
    cells = parity.get("v4", {}).get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ProtocolError("registered v4 parity cells are missing")
    result_rows: list[dict[str, Any]] = []
    for cell in cells:
        label = str(cell["label"])
        media = Path(str(cell["reference_matrix"]).replace("/scores/cells/", "/scores/media/").replace(".mkv", ".mkv"))
        # The source media path is authoritative in the frozen worker artifact.
        worker_path = Path(str(cell["worker"]))
        worker = read_json(worker_path)
        media = Path(str(worker["media"]))
        source_audio = Path(str(worker["source_audio"]))
        output_dir = paths.scores / "parity" / label
        score_row = engine.score(media, source_audio, output_dir, sample_id=str(cell["sample_id"]), video_arm=f"PARITY_{label}", audio_arm="N", reference_matrix=Path(str(cell["reference_matrix"])), expected_rows=None)
        actual = np.asarray(np.load(score_row["matrix"], allow_pickle=False), dtype=np.float64)
        reference_path = Path(str(cell["reference_matrix"]))
        reference = np.asarray(np.load(reference_path, allow_pickle=False), dtype=np.float64)
        actual_full = score_row["full"]
        reference_full = __import__("scripts.experiments.wav2lip_natural_content_residual.scoring", fromlist=["score_metrics"]).score_metrics(reference)
        matrix_max_abs = float(np.max(np.abs(actual - reference))) if actual.shape == reference.shape else float("inf")
        endpoint_error = max(abs(float(actual_full[name]) - float(reference_full[name])) for name in ("C", "D"))
        endpoint_error = max(endpoint_error, abs(int(actual_full["offset"]) - int(reference_full["offset"])))
        result_rows.append({"sample_id": str(cell["sample_id"]), "video_arm": f"PARITY_{label}", "audio_arm": "N", "parity": {"label": label, "reference_matrix": str(reference_path.resolve()), "reference_matrix_sha256": str(cell["reference_matrix_sha256"]), "actual_matrix": score_row["matrix"], "matrix_max_abs": matrix_max_abs, "endpoint_max_abs": endpoint_error, "offset_equal": int(actual_full["offset"]) == int(reference_full["offset"]), "passes": bool(matrix_max_abs <= 1e-4 and endpoint_error <= 1e-4 and int(actual_full["offset"]) == int(reference_full["offset"]))}, "score": score_row})
    return result_rows


def _score_controls(paths: config.RunPaths, media_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engine = ScoreEngine()
    rows = _score_parity(paths, engine)
    for media_row in media_rows:
        sample_id = str(media_row["sample_id"])
        arm = str(media_row["arm"])
        audio_arm = str(media_row["audio_arm"])
        video_arm = "N" if arm == "A_DELAY" else arm
        score = engine.score(Path(str(media_row["output"])), Path(str(media_row["audio"])), paths.scores / "controls" / f"{sample_id}__{video_arm}__{audio_arm}", sample_id=sample_id, video_arm=video_arm, audio_arm=audio_arm)
        rows.append({"sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, "score": score, **{key: score[key] for key in ("media", "media_sha256", "source_audio", "source_audio_sha256", "matrix", "matrix_sha256", "matrix_shape", "full", "U")}})
        print(f"SCORED controls {sample_id} {arm}", flush=True)
    manifest = {"schema_version": 1, "status": "complete", "stage": "controls", "rows": rows, "count": len(rows)}
    write_self_hashed_json(paths.control_scores, manifest)
    return rows


def _score_candidates(paths: config.RunPaths, media_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engine = ScoreEngine()
    rows: list[dict[str, Any]] = []
    for media_row in media_rows:
        sample_id = str(media_row["sample_id"])
        arm = str(media_row["arm"])
        score = engine.score(Path(str(media_row["output"])), Path(str(media_row["audio"])), paths.scores / "candidates" / f"{sample_id}__{arm}__N", sample_id=sample_id, video_arm=arm, audio_arm="N")
        rows.append({"sample_id": sample_id, "video_arm": arm, "audio_arm": "N", "score": score, **{key: score[key] for key in ("media", "media_sha256", "source_audio", "source_audio_sha256", "matrix", "matrix_sha256", "matrix_shape", "full", "U")}})
        print(f"SCORED candidates {sample_id} {arm}", flush=True)
    manifest = {"schema_version": 1, "status": "complete", "stage": "candidates", "rows": rows, "count": len(rows)}
    write_self_hashed_json(paths.candidates, manifest)
    return rows


def run_controls(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], resume: bool) -> dict[str, Any]:
    if paths.controls.is_file() and paths.control_scores.is_file() and resume:
        score_rows = list(_load(paths.control_scores)["rows"])
        control = analyze_controls(protocol, score_rows)
        write_self_hashed_json(paths.controls, control)
        return control
    driver_rows = list(drivers["rows"])
    generated = _run_gpu(_control_plan(paths, driver_rows), paths, "controls", resume)
    media_rows = _mux_control_media(paths, driver_rows, generated, resume)
    score_rows = _score_controls(paths, media_rows)
    control = analyze_controls(protocol, score_rows)
    write_self_hashed_json(paths.controls, control)
    print(f"STAGE_A {control['scientific_decision']}", flush=True)
    return control


def run_candidates(paths: config.RunPaths, drivers: dict[str, Any], control: dict[str, Any], resume: bool) -> list[dict[str, Any]]:
    if not bool(control.get("stage_b_authorized")):
        raise ProtocolError("Stage B is blocked by Stage A")
    if paths.candidates.is_file() and resume:
        return list(_load(paths.candidates)["rows"])
    generated = _run_gpu(_candidate_plan(paths, list(drivers["rows"])), paths, "candidates", resume)
    media_rows = _mux_candidate_media(paths, list(drivers["rows"]), generated, resume)
    rows = _score_candidates(paths, media_rows)
    print("STAGE_B candidate scoring complete", flush=True)
    return rows


def _run_validator(paths: config.RunPaths, stage: str) -> tuple[int, Path]:
    output = paths.control_validation if stage == "controls" else paths.validation
    log = paths.root / "logs" / f"validate_{stage}.log"
    command = [str(config.SYNCNET_PYTHON), "-m", "scripts.experiments.wav2lip_natural_content_residual.validate", "--run-root", str(paths.root), "--stage", stage]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.REPO), env={**os.environ, "PYTHONPATH": str(config.REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode, output


def finalize(paths: config.RunPaths, protocol: dict[str, Any], control: dict[str, Any], analysis: dict[str, Any] | None, validation: dict[str, Any] | None, *, engineering_status: str) -> None:
    scientific = str(control.get("scientific_decision")) if analysis is None else str(analysis.get("scientific_decision"))
    final = {"schema_version": 1, "protocol_id": protocol["protocol_id"], "status": "complete" if engineering_status == "GO" else "blocked", "engineering_status": engineering_status, "scientific_decision": scientific, "control_decision": control.get("scientific_decision"), "analysis": str(paths.analysis.resolve()) if analysis is not None else None, "analysis_sha256": file_sha256(paths.analysis) if paths.analysis.is_file() else None, "validation": str(paths.validation.resolve()) if validation is not None and paths.validation.is_file() else None, "validation_sha256": file_sha256(paths.validation) if validation is not None and paths.validation.is_file() else None, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "budget": {"max_videos": config.MAX_VIDEO_COUNT, "max_scores": config.MAX_SCORE_COUNT}}
    write_self_hashed_json(paths.final, final)
    lines = ["# Wav2Lip natural content residual probe", "", f"- engineering_status: `{engineering_status}`", f"- scientific_decision: `{scientific}`", f"- control: `{control.get('scientific_decision')}`", "- scope: seen records, first 3.84s mel support, full natural PCM mux", "- replacement_confirmed: `false`", "- waveform_head_authorized: `false`", "- generalization_established: `false`", ""]
    if analysis is not None:
        lines.extend(["## Main contrasts", "", "| Contrast | ΔC mean [95% CI] | ΔD mean [95% CI] | ΔA mean [95% CI] |", "|---|---:|---:|---:|"])
        for name, result in analysis.get("contrasts", {}).items():
            metrics = result["metrics"]
            lines.append(f"| {name} | {metrics['C']['mean']:.6f} [{metrics['C']['ci95'][0]:.6f}, {metrics['C']['ci95'][1]:.6f}] | {metrics['D']['mean']:.6f} [{metrics['D']['ci95'][0]:.6f}, {metrics['D']['ci95'][1]:.6f}] | {metrics['A']['mean']:.6f} [{metrics['A']['ci95'][0]:.6f}, {metrics['A']['ci95'][1]:.6f}] |")
        lines.extend(["", f"- norm_control_valid: `{analysis.get('norm_control_valid')}`", f"- content_signal: `{analysis.get('content_signal')}`"])
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = {"schema_version": 1, "review_type": "self-review", "status": "complete", "checks": ["OpenSpec scope respected", "free offset not used as replacement proof", "natural baseline is unmasked N", "all result flags remain false", "validator was run separately"], "remaining_limits": ["seen records", "short mel support", "not independent confirmation", "WRONG and SHUFFLE are imperfect specificity controls"]}
    write_self_hashed_json(paths.review, review)


def run_analysis(paths: config.RunPaths, protocol: dict[str, Any], drivers: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if not bool(control.get("stage_b_authorized")):
        finalize(paths, protocol, control, None, _load(paths.control_validation) if paths.control_validation.is_file() else None, engineering_status="GO")
        return control
    control_rows = list(_load(paths.control_scores)["rows"])
    candidate_rows = list(_load(paths.candidates)["rows"])
    analysis = analyze_candidates(protocol, drivers, control, control_rows + candidate_rows)
    write_self_hashed_json(paths.analysis, analysis)
    code, validation_path = _run_validator(paths, "all")
    validation = _load(validation_path) if validation_path.is_file() else None
    engineering = "GO" if code == 0 and validation and validation.get("status") == "PASS" else "BLOCKED"
    finalize(paths, protocol, control, analysis, validation, engineering_status=engineering)
    return analysis


def run(run_id: str, stage: str, resume: bool) -> dict[str, Any]:
    paths = _paths(run_id)
    protocol, drivers = _prepare(paths, resume)
    if stage == "prepare":
        return protocol
    control = run_controls(paths, protocol, drivers, resume)
    if stage == "controls":
        code, output = _run_validator(paths, "controls")
        if output.is_file():
            print(f"CONTROL_VALIDATION {'PASS' if code == 0 else 'FAIL'}", flush=True)
        return control
    if not bool(control.get("stage_b_authorized")):
        return run_analysis(paths, protocol, drivers, control)
    run_candidates(paths, drivers, control, resume)
    if stage == "candidates":
        return _load(paths.candidates)
    return run_analysis(paths, protocol, drivers, control)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "controls", "candidates", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.run_id, args.stage, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
