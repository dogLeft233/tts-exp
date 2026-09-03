from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.asr_sync_error_correlation.io import append_failure, atomic_write_json, canonical_hash, file_sha256, read_json, read_npz
from scripts.experiments.asr_sync_error_correlation.local_sync import compute_local_scores

from .analyze import make_analysis_rows, write_analysis, plot_sample
from .config import CONDITIONS, RuntimeOptions, config_json, frozen_config, parse_args
from .evaluate import render_wav2lip_cell, score_syncnet_cell, strict_replacement_cell, validate_isolated_work_paths
from .patch_audio import build_condition_audio, read_pcm16, validate_condition_output, write_pcm16
from .protocol import build_target_control_manifest, load_frozen_inputs


STAGES = ("lock", "candidates", "audio", "render", "replacement", "sync", "analysis")


def _stage_dir(run_dir: Path, name: str) -> Path:
    return run_dir / {"lock": "00_lock", "candidates": "01_candidates", "audio": "01_candidates", "render": "02_renders", "replacement": "03_replacement", "sync": "04_sync", "analysis": "05_analysis"}[name]


def _decision(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    path = _stage_dir(run_dir, stage) / "decision.json"
    atomic_write_json(path, {"schema_version": 1, "stage": stage, "status": status, **details})


def _require_go(run_dir: Path, stage: str) -> None:
    path = _stage_dir(run_dir, stage) / "decision.json"
    if not path.is_file():
        raise RuntimeError(f"required predecessor decision is missing: {path}")
    decision = read_json(path)
    if decision.get("status") != "GO":
        raise RuntimeError(f"predecessor stage is not GO: {stage}")


def _guard(options: RuntimeOptions) -> None:
    if options.run_dir.exists() and any(options.run_dir.iterdir()) and not options.resume:
        raise RuntimeError(f"run directory is non-empty; use --resume explicitly: {options.run_dir}")
    options.run_dir.mkdir(parents=True, exist_ok=True)


def _failure(run_dir: Path, stage: str, sample_id: str, condition: str, exc: BaseException) -> None:
    append_failure(run_dir / _stage_dir(run_dir, stage).name / "failures.json", stage=stage, sample_id=sample_id, arm=condition, exception=exc, retryable=True)


def run_lock(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "lock")
    try:
        config = frozen_config(options.repo_root)
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "config.json", config)
        frozen = load_frozen_inputs(options.repo_root)
        lock = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "claim_class": "discovery_reuse_fit_only",
            "config_sha256": file_sha256(stage_dir / "config.json"),
            "frozen_inputs": frozen,
            "sealed_splits_accessed": False,
        }
        atomic_write_json(stage_dir / "lock.json", lock)
        _decision(options.run_dir, "lock", "GO", sample_count=24, source_group_count=24, sealed_splits_accessed=False)
        return True
    except Exception as exc:
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "lock", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def _load_lock(run_dir: Path) -> dict[str, Any]:
    path = _stage_dir(run_dir, "lock") / "lock.json"
    if not path.is_file():
        raise RuntimeError(f"frozen lock is missing: {path}")
    lock = read_json(path)
    if lock.get("sealed_splits_accessed") is not False:
        raise ValueError("prototype lock records sealed split access")
    return lock


def run_candidates(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "candidates")
    try:
        _require_go(options.run_dir, "lock")
        lock = _load_lock(options.run_dir)
        manifest = build_target_control_manifest(lock["frozen_inputs"])
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "manifest.json", manifest)
        status = "GO" if int(manifest["eligible_sample_count"]) >= 12 else "INSUFFICIENT"
        _decision(options.run_dir, "candidates", status, eligible_sample_count=manifest["eligible_sample_count"], eligible_source_group_count=manifest["eligible_source_group_count"], readiness_counts=manifest["readiness_counts"])
        return status == "GO"
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "candidates", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def _load_candidates(run_dir: Path) -> dict[str, Any]:
    return read_json(_stage_dir(run_dir, "candidates") / "manifest.json")


def run_audio(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "audio")
    try:
        _require_go(options.run_dir, "candidates")
        candidates = _load_candidates(options.run_dir)
        frozen_samples = _load_lock(options.run_dir)["frozen_inputs"]["samples"]
        frozen_by_id = {str(row["sample_id"]): row for row in frozen_samples}
        successes = 0
        failures = 0
        for sample in candidates["samples"]:
            sample_id = str(sample["sample_id"])
            source = frozen_by_id[sample_id]
            natural_path = options.repo_root / source["arms"]["natural"]["audio_path"]
            tts_path = options.repo_root / source["arms"]["tts"]["audio_path"]
            natural, rate = read_pcm16(natural_path)
            tts, tts_rate = read_pcm16(tts_path)
            if rate != 16000 or tts_rate != 16000:
                raise ValueError(f"audio sample rate mismatch: {sample_id}")
            condition_patches = {row["condition"]: row["patch_blocks"] for row in sample["conditions"]}
            for condition in CONDITIONS:
                output = stage_dir / "audio" / sample_id / f"{condition}.wav"
                metadata_path = stage_dir / "records" / sample_id / f"{condition}.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                if options.resume and output.is_file() and metadata_path.is_file():
                    try:
                        existing = read_json(metadata_path)
                        if (existing.get("natural_audio_sha256") == file_sha256(natural_path) and existing.get("tts_audio_sha256") == file_sha256(tts_path) and existing.get("output_sha256") == file_sha256(output) and existing.get("validation", {}).get("sample_count") == int(natural.size)):
                            successes += 1
                            continue
                    except (OSError, ValueError, TypeError, KeyError):
                        pass
                try:
                    values, meta = build_condition_audio(natural, tts, condition, condition_patches[condition])
                    write_meta = write_pcm16(output, values)
                    validation = validate_condition_output(output, natural_path, natural.size)
                    if condition == "natural":
                        decoded, _ = read_pcm16(output)
                        if decoded.tobytes() != natural.tobytes():
                            raise ValueError("natural identity PCM is not byte-identical")
                    record = {"schema_version": 1, "sample_id": sample_id, "source_group": sample["source_group"], "condition": condition, "natural_audio": str(natural_path), "natural_audio_sha256": file_sha256(natural_path), "tts_audio": str(tts_path), "tts_audio_sha256": file_sha256(tts_path), "output_path": str(output), "output_sha256": file_sha256(output), "meta": meta, "writer": write_meta, "validation": validation}
                    atomic_write_json(metadata_path, record)
                    successes += 1
                except Exception as exc:
                    failures += 1
                    _failure(options.run_dir, "audio", sample_id, condition, exc)
        status = successes == 4 * len(candidates["samples"]) and failures == 0
        _decision(options.run_dir, "audio", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=4 * len(candidates["samples"]), failures=failures)
        return status
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "audio", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def _source_sample(lock: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for sample in lock["frozen_inputs"]["samples"]:
        if str(sample["sample_id"]) == sample_id:
            return sample
    raise KeyError(sample_id)


def _candidate_audio_path(run_dir: Path, sample_id: str, condition: str) -> Path:
    return _stage_dir(run_dir, "audio") / "audio" / sample_id / f"{condition}.wav"


def run_render(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "render")
    try:
        _require_go(options.run_dir, "audio")
        candidates = _load_candidates(options.run_dir)
        lock = _load_lock(options.run_dir)
        cells = [{"work_dir": stage_dir / "work" / str(sample["sample_id"]) / condition, "temp_dir": stage_dir / "work" / str(sample["sample_id"]) / condition / "temp"} for sample in candidates["samples"] for condition in CONDITIONS]
        validate_isolated_work_paths(cells)
        successes = 0
        for sample in candidates["samples"]:
            sample_id = str(sample["sample_id"])
            source = _source_sample(lock, sample_id)
            face = options.repo_root / source["video_path"]
            for condition in CONDITIONS:
                try:
                    record = render_wav2lip_cell(repo_root=options.repo_root, run_dir=options.run_dir, sample_id=sample_id, condition=condition, face_path=face, driver_audio=_candidate_audio_path(options.run_dir, sample_id, condition), device=options.device, resume=options.resume)
                    atomic_write_json(stage_dir / "records" / sample_id / f"{condition}.json", record)
                    successes += 1
                except Exception as exc:
                    _failure(options.run_dir, "render", sample_id, condition, exc)
        status = successes == 4 * len(candidates["samples"])
        _decision(options.run_dir, "render", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=4 * len(candidates["samples"]))
        return status
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "render", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def run_replacement(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "replacement")
    try:
        _require_go(options.run_dir, "render")
        candidates = _load_candidates(options.run_dir)
        lock = _load_lock(options.run_dir)
        successes = 0
        for sample in candidates["samples"]:
            sample_id = str(sample["sample_id"])
            source = _source_sample(lock, sample_id)
            natural = options.repo_root / source["arms"]["natural"]["audio_path"]
            for condition in CONDITIONS:
                try:
                    render_record = read_json(_stage_dir(options.run_dir, "render") / "records" / sample_id / f"{condition}.json")
                    result = strict_replacement_cell(repo_root=options.repo_root, run_dir=options.run_dir, sample_id=sample_id, condition=condition, rendered_video=Path(render_record["video_path"]), natural_audio=natural, resume=options.resume)
                    atomic_write_json(stage_dir / "records" / sample_id / f"{condition}.json", result)
                    successes += 1
                except Exception as exc:
                    _failure(options.run_dir, "replacement", sample_id, condition, exc)
        status = successes == 4 * len(candidates["samples"])
        _decision(options.run_dir, "replacement", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=4 * len(candidates["samples"]))
        return status
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "replacement", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def run_sync(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "sync")
    try:
        _require_go(options.run_dir, "replacement")
        candidates = _load_candidates(options.run_dir)
        successes = 0
        for sample in candidates["samples"]:
            sample_id = str(sample["sample_id"])
            result_by_condition: dict[str, Any] = {}
            for condition in CONDITIONS:
                try:
                    replacement = read_json(_stage_dir(options.run_dir, "replacement") / "records" / sample_id / f"{condition}.json")
                    patches = next(row["patch_blocks"] for row in sample["conditions"] if row["condition"] == condition)
                    result = score_syncnet_cell(repo_root=options.repo_root, run_dir=options.run_dir, sample_id=sample_id, condition=condition, replacement_media=Path(replacement["output_path"]), natural_sync_record=result_by_condition.get("natural"), patches=patches, audio_duration_s=float(_source_sample(_load_lock(options.run_dir), sample_id)["arms"]["natural"]["audio_duration_s"]), device=options.device, resume=options.resume)
                    result_by_condition[condition] = result
                    atomic_write_json(stage_dir / "records" / sample_id / f"{condition}.json", result)
                    successes += 1
                except Exception as exc:
                    _failure(options.run_dir, "sync", sample_id, condition, exc)
            if set(result_by_condition) == set(CONDITIONS):
                track_keys = ("selected_track_index", "selected_frame_count", "track_start_frame", "track_end_frame", "track_ranges")
                if any(result_by_condition[condition].get(key) != result_by_condition["natural"].get(key) for condition in CONDITIONS for key in track_keys):
                    _failure(options.run_dir, "sync", sample_id, "all", ValueError("paired_track_mismatch"))
                    successes -= 4
        status = successes == 4 * len(candidates["samples"])
        _decision(options.run_dir, "sync", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=4 * len(candidates["samples"]))
        return status
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "sync", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def run_analysis(options: RuntimeOptions) -> bool:
    stage_dir = _stage_dir(options.run_dir, "analysis")
    try:
        _require_go(options.run_dir, "sync")
        candidates = _load_candidates(options.run_dir)
        sync_records: dict[str, dict[str, Any]] = {}
        for sample in candidates["samples"]:
            sample_id = str(sample["sample_id"])
            sync_records[sample_id] = {condition: read_json(_stage_dir(options.run_dir, "sync") / "records" / sample_id / f"{condition}.json") for condition in CONDITIONS}
        rows = make_analysis_rows(candidates["samples"], sync_records)
        payload = write_analysis(stage_dir, rows)
        plot_failures: list[dict[str, str]] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            sample_manifest = next(sample for sample in candidates["samples"] if str(sample["sample_id"]) == sample_id)
            budgets = {"target": sum(float(block["destination_samples"]) / 16000.0 for block in sample_manifest["target_blocks"]), "control_0": sum(float(block["destination_samples"]) / 16000.0 for block in sample_manifest["control_replicates"][0]["blocks"]), "control_1": sum(float(block["destination_samples"]) / 16000.0 for block in sample_manifest["control_replicates"][1]["blocks"])}
            local = {condition: sync_records[sample_id][condition].get("fixed_coordinate_local") for condition in CONDITIONS}
            try:
                plot_sample(stage_dir / "plots" / f"{sample_id}.png", sample_id, row["condition_scores"], budgets, local=local)
            except Exception as exc:
                plot_failures.append({"sample_id": sample_id, "type": type(exc).__name__, "message": str(exc)[:500]})
        payload["plot_status"] = "complete" if not plot_failures else "incomplete"
        payload["plot_failures"] = plot_failures
        atomic_write_json(stage_dir / "analysis.json", payload)
        atomic_write_json(stage_dir / "plot_status.json", {"status": payload["plot_status"], "failures": plot_failures})
        decision = dict(payload["decision"])
        decision["schema_version"] = 1
        decision["plot_status"] = payload["plot_status"]
        atomic_write_json(options.run_dir / "summary.json", {"schema_version": 1, "experiment": "lrs3_asr_targeted_local_replacement_prototype", "analysis": payload["summary"], "bootstrap": payload["bootstrap"], "complete_samples": len(rows), "source_group_count": len({str(row["unit_id"]) for row in rows}), "sealed_splits_accessed": False, "plot_status": payload["plot_status"]})
        atomic_write_json(options.run_dir / "decision.json", decision)
        validation = validate_run_artifacts(options.run_dir)
        atomic_write_json(options.run_dir / "validation.json", validation)
        _decision(options.run_dir, "analysis", "GO", complete_samples=len(rows), science=decision["science"], plot_status=payload["plot_status"])
        return True
    except Exception as exc:
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        _decision(options.run_dir, "analysis", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def validate_run_artifacts(run_dir: Path) -> dict[str, Any]:
    candidates = _load_candidates(run_dir)
    samples = list(candidates["samples"])
    expected_conditions = tuple(candidates["required_conditions"])
    if expected_conditions != CONDITIONS:
        raise ValueError("candidate manifest condition order differs from frozen contract")
    sync_records: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        condition_records: dict[str, Any] = {}
        for condition in CONDITIONS:
            audio = read_json(_stage_dir(run_dir, "audio") / "records" / sample_id / f"{condition}.json")
            if file_sha256(audio["output_path"]) != audio["output_sha256"]:
                raise ValueError(f"candidate hash mismatch: {sample_id}/{condition}")
            render = read_json(_stage_dir(run_dir, "render") / "records" / sample_id / f"{condition}.json")
            if file_sha256(render["video_path"]) != render["video_sha256"]:
                raise ValueError(f"render hash mismatch: {sample_id}/{condition}")
            replacement = read_json(_stage_dir(run_dir, "replacement") / "records" / sample_id / f"{condition}.json")
            if file_sha256(replacement["output_path"]) != replacement["output_sha256"] or replacement["strict_contract"].get("audio_pcm_verified") is not True or replacement["strict_contract"].get("video_stream_copy_verified") is not True:
                raise ValueError(f"replacement validation failed: {sample_id}/{condition}")
            sync = read_json(_stage_dir(run_dir, "sync") / "records" / sample_id / f"{condition}.json")
            if sync.get("sample_id") != sample_id or sync.get("condition") != condition or sync.get("parity", {}).get("passed") is not True:
                raise ValueError(f"sync binding/parity failed: {sample_id}/{condition}")
            if file_sha256(sync["distance_path"]) != sync["distance_sha256"]:
                raise ValueError(f"distance hash mismatch: {sample_id}/{condition}")
            matrix = read_npz(sync["distance_path"], required=("dists",))["dists"]
            recomputed = compute_local_scores(matrix)
            if recomputed["av_offset_frames"] != sync["official"]["av_offset_frames"] or abs(recomputed["sync_c"] - sync["official"]["sync_c"]) > 1e-6 or abs(recomputed["sync_d"] - sync["official"]["sync_d"]) > 1e-6:
                raise ValueError(f"official score cannot be recomputed: {sample_id}/{condition}")
            condition_records[condition] = sync
        reference = condition_records["natural"]
        for condition in CONDITIONS[1:]:
            for key in ("selected_track_index", "selected_frame_count", "track_start_frame", "track_end_frame", "track_ranges"):
                if condition_records[condition].get(key) != reference.get(key):
                    raise ValueError(f"paired track mismatch: {sample_id}")
        if not (_stage_dir(run_dir, "analysis") / "plots" / f"{sample_id}.png").is_file():
            raise ValueError(f"missing sample diagnostic plot: {sample_id}")
        sync_records[sample_id] = condition_records
    analysis = read_json(_stage_dir(run_dir, "analysis") / "analysis.json")
    rows = make_analysis_rows(samples, sync_records)
    if rows != analysis["rows"]:
        raise ValueError("analysis rows do not recompute from SyncNet records")
    summary = read_json(run_dir / "summary.json")
    if summary.get("analysis") != analysis["summary"] or summary.get("bootstrap") != analysis["bootstrap"]:
        raise ValueError("root summary differs from analysis artifact")
    decision = read_json(run_dir / "decision.json")
    if decision.get("science") != analysis["decision"].get("science"):
        raise ValueError("root decision differs from analysis decision")
    return {"schema_version": 1, "status": "GO", "sample_count": len(samples), "condition_count": len(samples) * len(CONDITIONS), "source_group_count": len({str(row["source_group"]) for row in samples}), "sealed_splits_accessed": False}


def run(options: RuntimeOptions) -> int:
    _guard(options)
    handlers = {"lock": run_lock, "candidates": run_candidates, "audio": run_audio, "render": run_render, "replacement": run_replacement, "sync": run_sync, "analysis": run_analysis}
    if options.stage == "all":
        for stage in STAGES:
            if stage == "lock":
                ok = run_lock(options)
            elif stage == "candidates":
                ok = run_candidates(options)
            elif stage == "audio":
                ok = run_audio(options)
            elif stage == "render":
                ok = run_render(options)
            elif stage == "replacement":
                ok = run_replacement(options)
            elif stage == "sync":
                ok = run_sync(options)
            else:
                ok = run_analysis(options)
            if not ok:
                return 1
        return 0
    return 0 if handlers[options.stage](options) else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
