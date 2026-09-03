"""Operator-facing runner for the frozen ASR/SyncNet experiment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analyze import aggregate_records, build_grid, compute_metrics, decisions
from .asr_ctc import compare_official_timings, export_emissions, forced_align_reference, greedy_decode, infer_waveform
from .config import RuntimeOptions, config_json, frozen_config, parse_args
from .io import (
    append_failure,
    atomic_write_json,
    atomic_write_npz,
    canonical_hash,
    file_sha256,
    output_hashes,
    read_json,
    read_npz,
    success_marker_valid,
    write_success_marker,
)
from .local_sync import compute_local_scores, map_local_scores, parity_report, validate_paired_track_metadata
from .preflight import (
    acquire_asr_snapshot,
    check_environment,
    load_locked_processor_model,
    select_device,
    smoke_test_locked_model,
)
from .protocol import build_frozen_manifest, load_manifest
from .strict_inputs import mux_arm
from .syncnet_adapter import build_adapter_command, build_pipeline_command, run_subprocess
from .visualize import plot_sample, write_plot_index
from .word_errors import error_alignment, normalized_words


STAGES = ("preflight", "prepare", "mux", "asr", "sync", "align", "analyze", "plot")


def _stage_dir(run_dir: Path, number: str) -> Path:
    return run_dir / number


def _cell_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["sample_id"]), str(record["arm"])


def _record_path(run_dir: Path, stage: str, record: Mapping[str, Any], suffix: str = "json") -> Path:
    sample_id, arm = _cell_key(record)
    return run_dir / stage / "records" / arm / f"{sample_id}.{suffix}"


def _marker_path(run_dir: Path, stage: str, record: Mapping[str, Any]) -> Path:
    sample_id, arm = _cell_key(record)
    return run_dir / stage / "cells" / f"{sample_id}.{arm}.success.json"


def _failure_path(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "failures.json"


def _binding(config: Mapping[str, Any], *values: Any) -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    code_hash = canonical_hash({path.name: file_sha256(path) for path in sorted(package_dir.glob("*.py"))})
    return {"config_hash": canonical_hash(config), "code_hash": code_hash, "values": list(values), "python": sys.version.split()[0]}


def _write_stage_decision(run_dir: Path, stage_dir: str, status: str, **details: Any) -> None:
    atomic_write_json(run_dir / stage_dir / "decision.json", {"schema_version": 1, "stage": stage_dir, "status": status, **details})


def _read_stage_decision(run_dir: Path, stage_dir: str) -> Mapping[str, Any]:
    path = run_dir / stage_dir / "decision.json"
    if not path.is_file():
        raise RuntimeError(f"required predecessor marker is missing: {path}")
    decision = read_json(path)
    if decision.get("status") != "GO":
        raise RuntimeError(f"predecessor stage is not GO: {stage_dir}")
    return decision


def _run_dir_guard(options: RuntimeOptions) -> None:
    if options.run_dir.exists() and any(options.run_dir.iterdir()) and not options.resume:
        raise RuntimeError(f"run directory is non-empty; use --resume explicitly: {options.run_dir}")
    options.run_dir.mkdir(parents=True, exist_ok=True)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "00_preflight" / "config.json"
    if not path.is_file():
        raise RuntimeError(f"frozen config is missing: {path}")
    return read_json(path)


def run_preflight(options: RuntimeOptions) -> bool:
    config = frozen_config(options.repo_root)
    preflight_dir = _stage_dir(options.run_dir, "00_preflight")
    preflight_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(preflight_dir / "config.json", json.loads(config_json(config)))
    cache = Path(config["paths"]["asr_cache"])
    lock_path = preflight_dir / "asr_model_lock.json"
    try:
        if options.acquire_asr_model:
            acquire_asr_snapshot(cache_dir=cache, lock_path=lock_path)
        environment = check_environment(
            options.repo_root,
            requested_device=options.device,
            asr_cache=cache,
            syncnet_python=Path(config["paths"]["syncnet_python"]),
            lock_path=lock_path,
        )
        if environment["status"] == "GO":
            selected_device = str(environment["selected_device"])
            processor, model, model_info = load_locked_processor_model(lock_path, device=selected_device)
            source = read_json(options.repo_root / config["cohort"]["source_manifest"])
            smoke = smoke_test_locked_model(processor, model, device=selected_device, transcript=source["records"][0]["transcript"])
            environment["model"] = model_info
            environment["smoke"] = smoke
            environment["status"] = "GO"
        else:
            environment["status"] = "NO_GO"
        atomic_write_json(preflight_dir / "environment.json", environment)
        atomic_write_json(preflight_dir / "assets.json", {
            "syncnet_checkpoint_path": environment.get("syncnet_checkpoint"),
            "syncnet_checkpoint_sha256": file_sha256(options.repo_root / "third_party" / "syncnet_python" / "data" / "syncnet_v2.model"),
            "asr_cache": str(cache),
            "asr_model_lock_sha256": file_sha256(lock_path) if lock_path.is_file() else None,
        })
        _write_stage_decision(options.run_dir, "00_preflight", environment["status"], remediation=environment.get("remediation"))
        return environment["status"] == "GO"
    except Exception as exc:
        remediation = "HF_ENDPOINT=https://hf-mirror.com python -m scripts.experiments.asr_sync_error_correlation.run --stage preflight --acquire-asr-model"
        atomic_write_json(preflight_dir / "environment.json", {"status": "NO_GO", "error": f"{type(exc).__name__}: {exc}", "remediation": remediation})
        _write_stage_decision(options.run_dir, "00_preflight", "NO_GO", remediation=remediation)
        return False


def run_manifest(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "00_preflight")
        manifest = build_frozen_manifest(options.repo_root)
        manifest_path = options.run_dir / "01_manifest" / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(options.run_dir / "01_manifest" / "test_lock.json", {
            "status": "sealed_unvisited",
            "sealed_splits_accessed": False,
            "parent_test_lock_sha256": manifest["parent_test_lock_sha256"],
        })
        _write_stage_decision(options.run_dir, "01_manifest", "GO", arm_count=48, sample_count=24)
        return True
    except Exception as exc:
        append_failure(options.run_dir / "01_manifest" / "failures.json", stage="01_manifest", sample_id="manifest", arm="all", exception=exc, retryable=False)
        _write_stage_decision(options.run_dir, "01_manifest", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def _manifest_records(run_dir: Path) -> list[dict[str, Any]]:
    return list(load_manifest(run_dir / "01_manifest" / "manifest.json")["records"])


def run_mux(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "01_manifest")
        records = _manifest_records(options.run_dir)
    except Exception as exc:
        _write_stage_decision(options.run_dir, "02_inputs", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False
    successes = 0
    for record in records:
        output = options.run_dir / "02_inputs" / str(record["arm"]) / f"{record['sample_id']}.mkv"
        metadata = options.run_dir / "02_inputs" / "cells" / f"{record['sample_id']}.{record['arm']}.json"
        marker = _marker_path(options.run_dir, "02_inputs", record)
        bindings = _binding(_load_run_config(options.run_dir), record["video_sha256"], record["audio_sha256"], "strict_mux")
        if options.resume and success_marker_valid(marker, bindings=bindings, outputs={"mux": output, "metadata": metadata}):
            successes += 1
            continue
        try:
            result = mux_arm(record, repo_root=options.repo_root, output_path=output)
            atomic_write_json(metadata, result)
            write_success_marker(marker, stage="02_inputs", sample_id=record["sample_id"], arm=record["arm"], bindings=bindings, outputs={"mux": output, "metadata": metadata})
            successes += 1
        except Exception as exc:
            append_failure(_failure_path(options.run_dir, "02_inputs"), stage="02_inputs", sample_id=record["sample_id"], arm=record["arm"], exception=exc, log_path=None)
    status = successes == len(records)
    _write_stage_decision(options.run_dir, "02_inputs", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=len(records))
    return status


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if waveform.shape[1] != 1 or sample_rate != 16000:
        raise ValueError(f"audio must be mono 16 kHz PCM: {path}")
    values = waveform[:, 0]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values, int(sample_rate)


def run_asr(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "02_inputs")
        records = _manifest_records(options.run_dir)
        config = _load_run_config(options.run_dir)
        selected_device, _ = select_device(options.device)
        processor, model, model_info = load_locked_processor_model(options.run_dir / "00_preflight" / "asr_model_lock.json", device=selected_device)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        blank_id = int(getattr(model.config, "pad_token_id", 0))
        ratio = float(model.config.inputs_to_logits_ratio)
        delimiter_id = getattr(tokenizer, "word_delimiter_token_id", None)
        if delimiter_id is None:
            delimiter_id = int(tokenizer.get_vocab()[getattr(tokenizer, "word_delimiter_token", "|")])
        delimiter_id = int(delimiter_id)
        id_to_token = lambda token_id: str(tokenizer.convert_ids_to_tokens([int(token_id)])[0])
    except Exception as exc:
        _write_stage_decision(options.run_dir, "03_asr", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False
    successes = 0
    for record in records:
        emission_path = options.run_dir / "03_asr" / "emissions" / str(record["arm"]) / f"{record['sample_id']}.npz"
        asr_path = _record_path(options.run_dir, "03_asr", record)
        marker = _marker_path(options.run_dir, "03_asr", record)
        mux_marker = _marker_path(options.run_dir, "02_inputs", record)
        bindings = _binding(config, record["audio_sha256"], model_info["snapshot_hash"], model_info["tokenizer_vocabulary_hash"], selected_device)
        if options.resume and success_marker_valid(marker, bindings=bindings, outputs={"emissions": emission_path, "record": asr_path}):
            successes += 1
            continue
        try:
            if not mux_marker.is_file():
                raise RuntimeError("strict mux success marker is missing")
            audio_path = options.repo_root / record["audio_path"] if not Path(record["audio_path"]).is_absolute() else Path(record["audio_path"])
            waveform, sample_rate = _load_audio(audio_path)
            log_probs, runtime = infer_waveform(waveform, processor, model, device=selected_device)
            emission = export_emissions(emission_path, log_probs, provenance={**model_info, "sample_rate_hz": sample_rate, "input_audio_sha256": record["audio_sha256"]})
            decoded = greedy_decode(log_probs, blank_id=blank_id, delimiter_id=delimiter_id, frame_stride_s=ratio / 16000.0, audio_duration_s=len(waveform) / sample_rate, id_to_token=id_to_token)
            reference = forced_align_reference(log_probs, tokenizer, record["transcript"], blank_id=blank_id, frame_stride_s=ratio / 16000.0, audio_duration_s=len(waveform) / sample_rate)
            asr_record: dict[str, Any] = {
                "schema_version": 1,
                "sample_id": record["sample_id"], "source_group": record["source_group"], "arm": record["arm"],
                "audio_path": record["audio_path"], "audio_sha256": record["audio_sha256"], "audio_duration_s": len(waveform) / sample_rate,
                "model": model_info, "runtime": runtime, "frame_stride_s": ratio / 16000.0,
                "blank_id": blank_id, "delimiter_id": int(getattr(tokenizer, "word_delimiter_token_id")),
                "emissions": emission, "greedy": {**decoded, "frame_argmax_ids": decoded["frame_argmax_ids"].tolist()},
                "reference_alignment": {**reference, "aligned_token_ids": reference["aligned_token_ids"].tolist(), "aligned_scores": reference["aligned_scores"].tolist()},
            }
            if record["arm"] == "natural":
                asr_record["official_timing_qa"] = compare_official_timings(reference["words"], record["official_word_timings"])
            atomic_write_json(asr_path, asr_record)
            write_success_marker(marker, stage="03_asr", sample_id=record["sample_id"], arm=record["arm"], bindings=bindings, outputs={"emissions": emission_path, "record": asr_path})
            successes += 1
        except Exception as exc:
            append_failure(_failure_path(options.run_dir, "03_asr"), stage="03_asr", sample_id=record["sample_id"], arm=record["arm"], exception=exc)
    status = successes == len(records)
    _write_stage_decision(options.run_dir, "03_asr", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=len(records))
    return status


def run_sync(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "02_inputs")
        records = _manifest_records(options.run_dir)
        config = _load_run_config(options.run_dir)
        syncnet_python = Path(config["paths"]["syncnet_python"])
        syncnet_dir = Path(config["paths"]["syncnet_dir"])
        model_path = syncnet_dir / "data" / "syncnet_v2.model"
        adapter_path = options.repo_root / "scripts" / "experiments" / "asr_sync_error_correlation" / "syncnet_adapter.py"
    except Exception as exc:
        _write_stage_decision(options.run_dir, "04_sync", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False
    successes = 0
    for record in records:
        input_marker = _marker_path(options.run_dir, "02_inputs", record)
        work = options.run_dir / "04_sync" / "work" / str(record["arm"]) / str(record["sample_id"])
        distance_path = options.run_dir / "04_sync" / "distances" / str(record["arm"]) / f"{record['sample_id']}.npz"
        sync_path = _record_path(options.run_dir, "04_sync", record)
        adapter_json = work / "adapter.json"
        marker = _marker_path(options.run_dir, "04_sync", record)
        bindings = _binding(config, record["video_sha256"], record["audio_sha256"], file_sha256(model_path), "vshift=15", "min_track=50")
        if options.resume and success_marker_valid(marker, bindings=bindings, outputs={"distances": distance_path, "record": sync_path}):
            successes += 1
            continue
        try:
            if not input_marker.is_file():
                raise RuntimeError("strict mux success marker is missing")
            muxed = options.run_dir / "02_inputs" / str(record["arm"]) / f"{record['sample_id']}.mkv"
            reference = f"asr_sync_{record['sample_id']}_{record['arm']}"
            pipeline_cmd = build_pipeline_command(syncnet_python=syncnet_python, syncnet_dir=syncnet_dir, video_path=muxed, reference=reference, data_dir=work, min_track=50)
            run_subprocess(pipeline_cmd, cwd=syncnet_dir, log_path=options.run_dir / "logs" / "04_sync" / str(record["arm"]) / f"{record['sample_id']}.pipeline.log")
            adapter_cmd = build_adapter_command(syncnet_python=syncnet_python, adapter_path=adapter_path, video_path=muxed, reference=reference, data_dir=work, model_path=model_path, output_npz=work / "adapter_distances.npz", output_json=adapter_json, device=options.device)
            run_subprocess(adapter_cmd, cwd=syncnet_dir, log_path=options.run_dir / "logs" / "04_sync" / str(record["arm"]) / f"{record['sample_id']}.adapter.log")
            adapter = read_json(adapter_json)
            matrix = read_npz(work / "adapter_distances.npz", required=("dists",))["dists"].astype(np.float64)
            scores = compute_local_scores(matrix, vshift=15, median_width=9)
            parity = parity_report(scores, upstream_offset=adapter["upstream_offset"], upstream_confidence=adapter["upstream_sync_c"])
            if not parity["passed"]:
                raise ValueError("SyncNet upstream parity check failed")
            mapped = map_local_scores(scores, track_start_frame=int(adapter["track_start_frame"]), track_end_frame=int(adapter["track_end_frame"]), audio_duration_s=float(record["audio_duration_s"]), fps=25.0)
            arrays = {"dists": matrix.astype(np.float32), "mdist": scores["mdist"].astype(np.float32), "local_c_raw": scores["local_c_raw"].astype(np.float32), "local_c_filtered": scores["local_c_filtered"].astype(np.float32), "timestamps_s": mapped["timestamps_s"].astype(np.float64), "local_c": mapped["local_c"].astype(np.float64), "source_rows": mapped["source_rows"].astype(np.int64)}
            atomic_write_npz(distance_path, arrays)
            sync_record = {"schema_version": 1, "sample_id": record["sample_id"], "source_group": record["source_group"], "arm": record["arm"], "video_sha256": record["video_sha256"], "audio_sha256": record["audio_sha256"], "distance_sha256": file_sha256(distance_path), "distance_shape": list(matrix.shape), "track": adapter, "scores": {key: value for key, value in scores.items() if not isinstance(value, np.ndarray)}, "parity": parity, "mapped": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in mapped.items()}}
            atomic_write_json(sync_path, sync_record)
            write_success_marker(marker, stage="04_sync", sample_id=record["sample_id"], arm=record["arm"], bindings=bindings, outputs={"distances": distance_path, "record": sync_path})
            successes += 1
        except Exception as exc:
            append_failure(_failure_path(options.run_dir, "04_sync"), stage="04_sync", sample_id=record["sample_id"], arm=record["arm"], exception=exc, log_path=options.run_dir / "logs" / "04_sync" / str(record["arm"]) / f"{record['sample_id']}.adapter.log")
    paired_tracks_ok = True
    if successes == len(records):
        by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
        for record in records:
            sync_path = _record_path(options.run_dir, "04_sync", record)
            by_sample.setdefault(str(record["sample_id"]), {})[str(record["arm"])] = read_json(sync_path)
        try:
            for sample_id, pair in by_sample.items():
                validate_paired_track_metadata(pair["natural"]["track"], pair["tts"]["track"])
        except Exception as exc:
            paired_tracks_ok = False
            append_failure(_failure_path(options.run_dir, "04_sync"), stage="04_sync", sample_id=sample_id, arm="paired", exception=exc, retryable=False)
    status = successes == len(records) and paired_tracks_ok
    _write_stage_decision(options.run_dir, "04_sync", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=len(records), paired_track_metadata=paired_tracks_ok)
    return status


def _excluded_regions(sync_record: Mapping[str, Any], audio_duration_s: float) -> list[list[float]]:
    support = sync_record.get("mapped", {}).get("track_support_s", [0.0, audio_duration_s])
    regions: list[list[float]] = []
    if float(support[0]) > 0:
        regions.append([0.0, float(support[0])])
    if float(support[1]) < float(audio_duration_s):
        regions.append([float(support[1]), float(audio_duration_s)])
    return regions


def run_alignment(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "03_asr")
        _read_stage_decision(options.run_dir, "04_sync")
        records = _manifest_records(options.run_dir)
        config = _load_run_config(options.run_dir)
    except Exception as exc:
        _write_stage_decision(options.run_dir, "05_alignment", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False
    successes = 0
    for record in records:
        asr_path = _record_path(options.run_dir, "03_asr", record)
        sync_path = _record_path(options.run_dir, "04_sync", record)
        align_path = _record_path(options.run_dir, "05_alignment", record)
        grid_path = options.run_dir / "05_alignment" / "grids" / str(record["arm"]) / f"{record['sample_id']}.npz"
        marker = _marker_path(options.run_dir, "05_alignment", record)
        bindings = _binding(config, file_sha256(asr_path), file_sha256(sync_path), record["audio_sha256"])
        if options.resume and success_marker_valid(marker, bindings=bindings, outputs={"grid": grid_path, "record": align_path}):
            successes += 1
            continue
        try:
            asr = read_json(asr_path)
            sync = read_json(sync_path)
            ref_words = [str(row["word"]) for row in asr["reference_alignment"]["words"]]
            pred_words = [str(row["word"]) for row in asr["greedy"]["words"]]
            alignment = error_alignment(ref_words, pred_words, asr["reference_alignment"]["words"], asr["greedy"]["words"], audio_duration_s=float(asr["audio_duration_s"]))
            timestamps = np.asarray(sync["mapped"]["timestamps_s"], dtype=np.float64)
            local_c = np.asarray(sync["mapped"]["local_c"], dtype=np.float64)
            grid = build_grid(timestamps, local_c, alignment["error_spans"], low_score_k=float(config["analysis"]["low_score_k"]), std_ddof=int(config["analysis"]["std_ddof"]))
            atomic_write_npz(grid_path, {"timestamps_s": grid["timestamps_s"], "local_c": grid["local_c"], "badness": grid["badness"], "error_mask": grid["error_mask"], "low_sync_mask": grid["low_sync_mask"]})
            metrics = compute_metrics(grid, word_error_rate=float(alignment["word_error_rate"]))
            aligned = {"schema_version": 1, "sample_id": record["sample_id"], "source_group": record["source_group"], "arm": record["arm"], "audio_duration_s": record["audio_duration_s"], "error_alignment": alignment, "error_spans": alignment["error_spans"], "threshold": grid["threshold"], "metrics": metrics, "grid": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in grid.items()}, "excluded_regions": _excluded_regions(sync, float(record["audio_duration_s"])), "distance_sha256": sync["distance_sha256"]}
            atomic_write_json(align_path, aligned)
            write_success_marker(marker, stage="05_alignment", sample_id=record["sample_id"], arm=record["arm"], bindings=bindings, outputs={"grid": grid_path, "record": align_path})
            successes += 1
        except Exception as exc:
            append_failure(_failure_path(options.run_dir, "05_alignment"), stage="05_alignment", sample_id=record["sample_id"], arm=record["arm"], exception=exc)
    status = successes == len(records)
    _write_stage_decision(options.run_dir, "05_alignment", "GO" if status else "NO_GO", successful_cells=successes, expected_cells=len(records))
    return status


def _alignment_records(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for record in _manifest_records(run_dir):
        path = _record_path(run_dir, "05_alignment", record)
        if not path.is_file():
            continue
        rows.append(read_json(path))
    return rows


def run_analysis(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "05_alignment")
        records = _alignment_records(options.run_dir)
        config = _load_run_config(options.run_dir)
        if len(records) != 48:
            raise RuntimeError(f"expected 48 alignment records, found {len(records)}")
        per_record = [{"sample_id": row["sample_id"], "source_group": row["source_group"], "arm": row["arm"], **row["metrics"]} for row in records]
        aggregate = aggregate_records(per_record, seed=int(config["analysis"]["bootstrap_seed"]), draws=int(config["analysis"]["bootstrap_draws"]))
        aggregate["sample_count"] = 24
        aggregate["successful_arm_count"] = len(records)
        atomic_write_json(options.run_dir / "06_analysis" / "per_record.json", {"schema_version": 1, "records": per_record})
        atomic_write_json(options.run_dir / "06_analysis" / "summary.json", aggregate)
        engineering_checks = {"preflight": read_json(options.run_dir / "00_preflight/decision.json").get("status") == "GO", "manifest": read_json(options.run_dir / "01_manifest/decision.json").get("status") == "GO", "inputs": read_json(options.run_dir / "02_inputs/decision.json").get("status") == "GO", "asr": read_json(options.run_dir / "03_asr/decision.json").get("status") == "GO", "sync": read_json(options.run_dir / "04_sync/decision.json").get("status") == "GO", "alignment": read_json(options.run_dir / "05_alignment/decision.json").get("status") == "GO", "plots": (options.run_dir / "07_plots" / "index.json").is_file()}
        decision = decisions({**aggregate, "natural": aggregate["natural"], "tts": aggregate["tts"]}, engineering_checks=engineering_checks)
        atomic_write_json(options.run_dir / "06_analysis" / "decision.json", decision)
        _write_stage_decision(options.run_dir, "06_analysis", "GO", successful_arm_count=len(records))
        return True
    except Exception as exc:
        append_failure(_failure_path(options.run_dir, "06_analysis"), stage="06_analysis", sample_id="analysis", arm="all", exception=exc, retryable=False)
        _write_stage_decision(options.run_dir, "06_analysis", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def _marker_outputs_valid(marker_path: Path, outputs: Mapping[str, Path]) -> bool:
    try:
        marker = read_json(marker_path)
        return marker.get("outputs") == output_hashes(outputs)
    except (OSError, ValueError, TypeError):
        return False


def _required_outputs_complete(run_dir: Path, records: Sequence[Mapping[str, Any]]) -> bool:
    required = [
        run_dir / "00_preflight" / name for name in ("config.json", "environment.json", "assets.json", "decision.json")
    ] + [run_dir / "01_manifest" / name for name in ("manifest.json", "test_lock.json", "decision.json")]
    if not all(path.is_file() for path in required):
        return False
    for record in records:
        sample_id, arm = _cell_key(record)
        outputs_by_stage = {
            "02_inputs": {"mux": run_dir / "02_inputs" / arm / f"{sample_id}.mkv", "metadata": run_dir / "02_inputs" / "cells" / f"{sample_id}.{arm}.json"},
            "03_asr": {"emissions": run_dir / "03_asr" / "emissions" / arm / f"{sample_id}.npz", "record": _record_path(run_dir, "03_asr", record)},
            "04_sync": {"distances": run_dir / "04_sync" / "distances" / arm / f"{sample_id}.npz", "record": _record_path(run_dir, "04_sync", record)},
            "05_alignment": {"grid": run_dir / "05_alignment" / "grids" / arm / f"{sample_id}.npz", "record": _record_path(run_dir, "05_alignment", record)},
        }
        for stage, outputs in outputs_by_stage.items():
            if any(not path.is_file() for path in outputs.values()) or not _marker_outputs_valid(_marker_path(run_dir, stage, record), outputs):
                return False
    return True


def _failure_ledger_summary(run_dir: Path) -> dict[str, Any]:
    by_stage: dict[str, Any] = {}
    for stage in ("01_manifest", "02_inputs", "03_asr", "04_sync", "05_alignment", "06_analysis", "07_plots"):
        path = run_dir / stage / "failures.json"
        if not path.is_file():
            continue
        rows = read_json(path)
        if not isinstance(rows, list):
            continue
        by_exception: dict[str, int] = {}
        for row in rows:
            key = str(row.get("exception_type", "unknown")) if isinstance(row, Mapping) else "unknown"
            by_exception[key] = by_exception.get(key, 0) + 1
        by_stage[stage] = {"total": len(rows), "by_exception_type": dict(sorted(by_exception.items()))}
    return {"historical_attempts": sum(int(row["total"]) for row in by_stage.values()), "by_stage": by_stage}


def _excluded_duration_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {"natural": 0.0, "tts": 0.0}
    counts: dict[str, int] = {"natural": 0, "tts": 0}
    for record in records:
        arm = str(record.get("arm"))
        regions = record.get("excluded_regions", [])
        duration = sum(float(region[1]) - float(region[0]) for region in regions)
        totals[arm] = totals.get(arm, 0.0) + duration
        counts[arm] = counts.get(arm, 0) + len(regions)
    return {"seconds_by_arm": totals, "region_count_by_arm": counts}


def _finalize(options: RuntimeOptions) -> bool:
    try:
        summary = read_json(options.run_dir / "06_analysis" / "summary.json")
        plot_index = read_json(options.run_dir / "07_plots" / "index.json")
        records = _manifest_records(options.run_dir)
        artifact_complete = _required_outputs_complete(options.run_dir, records)
        checks = {
            "preflight": read_json(options.run_dir / "00_preflight/decision.json").get("status") == "GO",
            "manifest_24_samples_48_arms": read_json(options.run_dir / "01_manifest/manifest.json").get("arm_count") == 48 and len(records) == 48,
            "inputs": read_json(options.run_dir / "02_inputs/decision.json").get("successful_cells") == 48,
            "asr": read_json(options.run_dir / "03_asr/decision.json").get("successful_cells") == 48,
            "sync": read_json(options.run_dir / "04_sync/decision.json").get("successful_cells") == 48,
            "alignment": read_json(options.run_dir / "05_alignment/decision.json").get("successful_cells") == 48,
            "artifacts_and_markers": artifact_complete,
            "plots": len(plot_index.get("plots", [])) == 24 and all((options.run_dir / "07_plots" / f"{sample_id}.png").is_file() for sample_id in {str(row["sample_id"]) for row in records}),
        }
        decision = decisions(summary, engineering_checks=checks)
        root_summary = {"schema_version": 1, "sample_count": 24, "successful_arm_count": int(summary.get("successful_arm_count", 0)), "engineering_checks": checks, "analysis": summary, "plot_count": len(plot_index.get("plots", [])), "failure_ledger": _failure_ledger_summary(options.run_dir), "excluded_timeline": _excluded_duration_summary(_alignment_records(options.run_dir))}
        atomic_write_json(options.run_dir / "summary.json", root_summary)
        atomic_write_json(options.run_dir / "decision.json", decision)
        return decision["engineering"] == "GO"
    except Exception:
        return False


def run_plots(options: RuntimeOptions) -> bool:
    try:
        _read_stage_decision(options.run_dir, "06_analysis")
        records = _alignment_records(options.run_dir)
        config = _load_run_config(options.run_dir)
        config_hash = canonical_hash(config)
        by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in records:
            by_sample.setdefault(str(row["sample_id"]), {})[str(row["arm"])] = row
        plot_rows = []
        for sample_id in sorted(by_sample):
            plot_rows.append(plot_sample(sample_id, by_sample[sample_id], output_path=options.run_dir / "07_plots" / f"{sample_id}.png", config_hash=config_hash))
        write_plot_index(options.run_dir / "07_plots" / "index.json", plot_rows)
        _write_stage_decision(options.run_dir, "07_plots", "GO", plot_count=len(plot_rows))
        return _finalize(options)
    except Exception as exc:
        append_failure(_failure_path(options.run_dir, "07_plots"), stage="07_plots", sample_id="plots", arm="all", exception=exc, retryable=True)
        _write_stage_decision(options.run_dir, "07_plots", "NO_GO", error=f"{type(exc).__name__}: {exc}")
        return False


def run(options: RuntimeOptions) -> int:
    _run_dir_guard(options)
    requested = list(STAGES if options.stage is None else (options.stage,))
    result = True
    for stage in requested:
        if stage == "preflight":
            result = run_preflight(options) and result
        elif stage == "prepare":
            result = run_manifest(options) and result
        elif stage == "mux":
            result = run_mux(options) and result
        elif stage == "asr":
            result = run_asr(options) and result
        elif stage == "sync":
            result = run_sync(options) and result
        elif stage == "align":
            result = run_alignment(options) and result
        elif stage == "analyze":
            result = run_analysis(options) and result
        elif stage == "plot":
            result = run_plots(options) and result
    if options.stage is None and result:
        result = _finalize(options)
    return 0 if result else 1


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(list(argv) if argv is not None else None)
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
