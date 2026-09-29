from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
import wave
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.experiments.lrs3_tts_gain_mechanism"

from . import config
from .analysis import (
    bootstrap_indices,
    curve_analysis,
    decompose_pair,
    group_pairs,
    historical_analysis,
    pair_curve_metrics,
    plot_outputs,
    score_row,
    write_curve_tables,
    write_historical_tables,
)
from .common import (
    ProtocolError,
    canonical_sha256,
    csv_read,
    csv_write,
    read_json,
    read_self_hashed_json,
    require_file,
    sample_ids_sha256,
    sha256_bytes,
    sha256_file,
    write_json,
    write_self_hashed_json,
)


def _path(value: str | Path) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def _bound_file(path: str | Path, expected: str | None = None) -> dict[str, Any]:
    target = _path(path)
    item: dict[str, Any] = {"path": str(target), "expected_sha256": expected, "actual_sha256": None, "status": "MISSING"}
    if target.is_file():
        actual = sha256_file(target)
        item["actual_sha256"] = actual
        item["status"] = "PASS" if expected is None or actual == expected else "HASH_MISMATCH"
    return item


def load_input_bindings() -> dict[str, Any]:
    value = read_json(config.INPUT_BINDINGS)
    if not isinstance(value, dict):
        raise ProtocolError("input-bindings must be an object")
    if value.get("protocol_id") != config.PROTOCOL_ID or value.get("record_count") != 50 or value.get("source_group_count") != 45:
        raise ProtocolError("input-bindings protocol or cohort dimensions changed")
    if value.get("curve_sample_ids") != list(config.CURVE_SAMPLE_IDS):
        raise ProtocolError("curve sample selection changed")
    budget = value.get("budget")
    if not isinstance(budget, Mapping) or budget.get("new_syncnet_cells_max") != config.MAX_NEW_CELLS:
        raise ProtocolError("SyncNet cell budget changed")
    return value


def verify_bound_files(bindings: Mapping[str, Any], *, include_curve_media: bool = True) -> list[dict[str, Any]]:
    files = bindings.get("files")
    media = bindings.get("curve_media")
    if not isinstance(files, list) or not isinstance(media, list):
        raise ProtocolError("input-bindings files or curve_media is missing")
    if len(files) != 57 or len(media) != config.MAIN_CURVE_CELLS:
        raise ProtocolError("input-bindings file denominator changed")
    if any(not isinstance(item, Mapping) or not isinstance(item.get("path"), str) for item in files):
        raise ProtocolError("malformed input file binding")
    if any(not isinstance(item, Mapping) or not isinstance(item.get("sample_id"), (int, str)) or not isinstance(item.get("condition"), str) for item in media):
        raise ProtocolError("malformed curve media binding")
    if len({str(item.get("path")) for item in files}) != len(files) or len({(str(item.get("sample_id")), str(item.get("condition"))) for item in media}) != len(media):
        raise ProtocolError("input-bindings contains duplicate file or media identities")
    result: list[dict[str, Any]] = []
    bound_items = [*files, *media] if include_curve_media else list(files)
    for item in bound_items:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ProtocolError("malformed input binding")
        target = _path(str(item["path"]))
        try:
            target.relative_to(config.REPO)
        except ValueError as exc:
            raise ProtocolError(f"binding escapes repository: {target}") from exc
        result.append(_bound_file(target, str(item["sha256"])))
    failures = [item for item in result if item["status"] != "PASS"]
    if failures:
        labels = ", ".join(str(item["path"]) for item in failures[:5])
        raise ProtocolError(f"bound input is missing or changed: {labels}")
    return result


def build_cohort(bindings: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    manifest = read_json(config.MANIFEST)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("records"), list):
        raise ProtocolError("dataset manifest records are missing")
    records: list[dict[str, Any]] = []
    for original_index, raw in enumerate(manifest["records"], start=1):
        if not isinstance(raw, Mapping) or raw.get("dataset") != "lrs3":
            continue
        video_value = raw.get("video_local_path")
        stem = str(raw.get("stem", ""))
        if not isinstance(video_value, str) or not stem:
            raise ProtocolError(f"LRS3 manifest record is incomplete: {original_index}")
        video = _path(video_value)
        source_group = video.parent.name
        if not source_group or not stem.startswith(source_group + "_"):
            raise ProtocolError(f"source group does not agree with stem: {original_index}")
        records.append(
            {
                "sample_id": original_index,
                "stem": stem,
                "source_group": source_group,
                "speaker_key": str(raw.get("speaker_key", "")),
                "dataset": "lrs3",
                "video_local_path": str(video),
                "video_exists": video.is_file(),
            }
        )
    if [int(row["sample_id"]) for row in records] != list(config.HISTORICAL_SAMPLE_IDS):
        raise ProtocolError("LRS3 sample IDs are not the frozen original 1-based rows 151..200")
    if len({str(row["source_group"]) for row in records}) != 45:
        raise ProtocolError("LRS3 source-group count is not 45")
    bound_records = bindings.get("records")
    if not isinstance(bound_records, list) or len(bound_records) != len(records):
        raise ProtocolError("input-binding record list is inconsistent")
    for actual, expected in zip(records, bound_records, strict=True):
        if {key: actual[key] for key in ("sample_id", "stem", "source_group")} != {key: expected.get(key) for key in ("sample_id", "stem", "source_group")}:
            raise ProtocolError(f"cohort identity differs at {actual['sample_id']}")
    first_by_group: dict[str, dict[str, Any]] = {}
    for row in records:
        first_by_group.setdefault(str(row["source_group"]), row)
    first = list(first_by_group.values())[: len(config.CURVE_SAMPLE_IDS)]
    if [int(row["sample_id"]) for row in first] != list(config.CURVE_SAMPLE_IDS):
        raise ProtocolError("curve queue is not the first 12 source groups in manifest order")
    return records, {int(row["sample_id"]): str(row["source_group"]) for row in records}


def _hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def current_run_identity() -> dict[str, Any]:
    code_paths = {
        "config.py": Path(config.__file__),
        "common.py": Path(__file__).with_name("common.py"),
        "analysis.py": Path(__file__).with_name("analysis.py"),
        "runner.py": Path(__file__),
        "syncnet_worker.py": Path(__file__).with_name("syncnet_worker.py"),
        "validate.py": Path(__file__).with_name("validate.py"),
        "SyncNetInstance.py": config.SYNCNET_ROOT / "SyncNetInstance.py",
        "SyncNetModel.py": config.SYNCNET_ROOT / "SyncNetModel.py",
        "run_pipeline.py": config.SYNCNET_ROOT / "run_pipeline.py",
        "run_syncnet.py": config.SYNCNET_ROOT / "run_syncnet.py",
        "detectors/__init__.py": config.SYNCNET_ROOT / "detectors" / "__init__.py",
        "detectors/s3fd/__init__.py": config.SYNCNET_ROOT / "detectors" / "s3fd" / "__init__.py",
        "detectors/s3fd/box_utils.py": config.SYNCNET_ROOT / "detectors" / "s3fd" / "box_utils.py",
        "detectors/s3fd/nets.py": config.SYNCNET_ROOT / "detectors" / "s3fd" / "nets.py",
    }
    code_hashes = {name: _hash_or_none(path) for name, path in code_paths.items()}
    runtime_hashes = {
        "syncnet_python": _hash_or_none(config.SYNCNET_PYTHON),
        "syncnet_model": _hash_or_none(config.SYNCNET_MODEL),
        "s3fd_face": _hash_or_none(config.SYNCNET_ROOT / "detectors" / "s3fd" / "weights" / "sfd_face.pth"),
        # Bind the executable that the runner will actually invoke, including
        # the PATH fallback used on machines without the configured conda path.
        "ffmpeg": _hash_or_none(_ffmpeg_path()),
        "ffprobe": _hash_or_none(_ffprobe_path()),
    }
    configuration = {
        "sample_rate": config.SAMPLE_RATE,
        "fps": config.FPS,
        "vshift": config.VSHIFT,
        "lag_count": config.LAG_COUNT,
        "min_track": config.MIN_TRACK,
        "batch_size": config.BATCH_SIZE,
        "curve_sample_ids": list(config.CURVE_SAMPLE_IDS),
        "conditions": list(config.CONDITIONS),
        "official_pipeline": config.OFFICIAL_PIPELINE_CONFIG,
        "bootstrap_seed": config.BOOTSTRAP_SEED,
        "bootstrap_draws": config.BOOTSTRAP_DRAWS,
    }
    body = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "input_bindings_sha256": _hash_or_none(config.INPUT_BINDINGS),
        "code_sha256": code_hashes,
        "runtime_sha256": runtime_hashes,
        "configuration": configuration,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def _score_path(model: str, sample_id: int, condition: str) -> Path:
    root = config.DITTO_CELL_ROOT if model == "Ditto" else config.LEAPTALK_CELL_ROOT
    return root / condition / str(sample_id) / "syncnet.json"


def load_historical_records(cohort: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ditto = read_json(config.DITTO_SCORES)
    leaptalk = read_json(config.LEAPTALK_SCORES)
    ditto_meta = read_json(config.DITTO_META)
    if not isinstance(ditto, list) or not isinstance(leaptalk, list):
        raise ProtocolError("historical score files must be arrays")
    if (
        not isinstance(ditto_meta, Mapping)
        or ditto_meta.get("conditions") != list(config.CONDITIONS)
        or ditto_meta.get("complete") is not True
        or ditto_meta.get("samples_failed") not in (0, [], None)
        or not isinstance(ditto_meta.get("results"), Mapping)
    ):
        raise ProtocolError("Ditto eval_meta is incomplete or has changed conditions")
    by_model: dict[str, list[dict[str, Any]]] = {"Ditto": [], "LeapTalk": []}
    group_for = {int(row["sample_id"]): str(row["source_group"]) for row in cohort}
    seen: set[tuple[str, int, str]] = set()
    for model, rows in (("Ditto", ditto), ("LeapTalk", leaptalk)):
        for raw in rows:
            if model == "Ditto":
                sample_id = raw.get("sample_id")
                condition = raw.get("condition")
                c_value = raw.get("sync_c")
                d_value = raw.get("sync_d")
            else:
                sample_id = raw.get("i")
                condition = raw.get("cond")
                c_value = raw.get("c")
                d_value = raw.get("d")
            try:
                sample = int(sample_id)
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"invalid historical sample ID for {model}") from exc
            if sample not in group_for or condition not in config.CONDITIONS:
                continue
            key = (model, sample, str(condition))
            if key in seen:
                raise ProtocolError(f"duplicate historical score cell: {key}")
            seen.add(key)
            path = _score_path(model, sample, str(condition))
            normalized = score_row(model, sample, str(condition), c_value, d_value, group_for[sample], str(path) if path.is_file() else None)
            if model == "Ditto":
                meta_row = ditto_meta["results"].get(f"{condition}:{sample}")
                if not isinstance(meta_row, Mapping) or abs(float(meta_row.get("sync_c", float("nan"))) - float(c_value)) > 1e-10 or abs(float(meta_row.get("sync_d", float("nan"))) - float(d_value)) > 1e-10:
                    raise ProtocolError(f"Ditto eval_meta differs from scores_250.json: {sample}/{condition}")
            by_model[model].append(normalized)
    expected = len(config.MODELS) * len(config.HISTORICAL_SAMPLE_IDS) * len(config.CONDITIONS)
    if len(seen) != expected:
        raise ProtocolError(f"historical score denominator is incomplete: {len(seen)}/{expected}")
    cells = sorted([row for rows in by_model.values() for row in rows], key=lambda row: (str(row["model"]), int(row["sample_id"]), str(row["condition"])))
    pairs: list[dict[str, Any]] = []
    for model in config.MODELS:
        for sample_id in config.HISTORICAL_SAMPLE_IDS:
            row_by_condition = {
                str(row["condition"]): row
                for row in by_model[model]
                if int(row["sample_id"]) == sample_id
            }
            pairs.append(decompose_pair(row_by_condition["natural_raw"], row_by_condition["tts_raw"]))
    return cells, pairs


def verify_bound_score_cells(cells: Sequence[Mapping[str, Any]], bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check every bound per-cell JSON against its historical array."""

    expected = {
        (str(row["model"]), int(row["sample_id"]), str(row["condition"])): row
        for row in cells
    }
    checked = 0
    checked_paths: set[Path] = set()
    for item in bindings.get("files", []):
        path_value = str(item.get("path", ""))
        if not path_value.endswith("/syncnet.json"):
            continue
        model = "Ditto" if "/runs/multiset_pipeline/" in f"/{path_value}" else "LeapTalk" if "/runs/leaptalk_eval/" in f"/{path_value}" else None
        if model is None:
            continue
        path = _path(path_value)
        if path in checked_paths:
            raise ProtocolError(f"duplicate bound score cell path: {path}")
        checked_paths.add(path)
        raw = read_json(path)
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"bound score cell is not an object: {path}")
        sample_id = int(raw.get("sample_id", -1))
        condition = str(raw.get("condition", ""))
        reference = expected.get((model, sample_id, condition))
        if reference is None or abs(float(raw.get("sync_c", float("nan"))) - float(reference["sync_c"])) > 1e-10 or abs(float(raw.get("sync_d", float("nan"))) - float(reference["sync_d"])) > 1e-10:
            raise ProtocolError(f"bound score cell differs from historical array: {path}")
        checked += 1
    if checked != 48 or len(checked_paths) != 48:
        raise ProtocolError(f"expected 48 bound per-cell score cross-checks, found {checked}")
    return {"checked": checked, "status": "complete"}


def audit_inputs(paths: config.RunPaths) -> dict[str, Any]:
    bindings = load_input_bindings()
    bound_files = verify_bound_files(bindings, include_curve_media=False)
    cohort, group_for = build_cohort(bindings)
    cells, pairs = load_historical_records(cohort)
    score_crosscheck = verify_bound_score_cells(cells, bindings)
    source_groups = sorted({str(row["source_group"]) for row in cohort})
    curve_records = [row for row in cohort if int(row["sample_id"]) in config.CURVE_SAMPLE_IDS]
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "bindings": {
            "path": str(config.INPUT_BINDINGS.resolve()),
            "sha256": sha256_file(config.INPUT_BINDINGS),
            "file_count": len(bound_files),
            "files": bound_files,
            "curve_media_count": len(bindings.get("curve_media", [])),
        },
        "historical_sources": {
            "manifest": str(config.MANIFEST.resolve()),
            "ditto_scores": str(config.DITTO_SCORES.resolve()),
            "ditto_eval_meta": str(config.DITTO_META.resolve()),
            "leaptalk_scores": str(config.LEAPTALK_SCORES.resolve()),
            "models": {"Ditto": "historical TTS/TFG run; faster_qwen3 0.6B ICL declaration", "LeapTalk": "historical TTS/TFG run; faster_qwen3 0.6B ICL declaration"},
            "declaration_limit": "language/provider fields are provenance declarations, not a new audio-quality causal measurement",
        },
        "record_count": len(cohort),
        "source_group_count": len(source_groups),
        "sample_ids": [int(row["sample_id"]) for row in cohort],
        "sample_ids_sha256": sample_ids_sha256([int(row["sample_id"]) for row in cohort]),
        "speaker_key_values": sorted({str(row["speaker_key"]) for row in cohort}),
        "curve_sample_ids": list(config.CURVE_SAMPLE_IDS),
        "curve_records": curve_records,
        "historical_cell_count": len(cells),
        "historical_pair_count": len(pairs),
        "historical_score_crosscheck": score_crosscheck,
        "source_group_order": source_groups,
        "zero_new_tts": True,
        "zero_new_tfg": True,
        "zero_training": True,
        "zero_cloud_calls": True,
        "run_identity": current_run_identity(),
    }
    write_self_hashed_json(paths.audit / "inputs.json", payload)
    cohort_payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(cohort),
        "source_group_count": len(source_groups),
        "sample_ids_sha256": payload["sample_ids_sha256"],
        "records": cohort,
        "curve_records": curve_records,
        "source_groups": source_groups,
        "group_for_sample_id": group_for,
    }
    write_self_hashed_json(paths.audit / "cohort.json", cohort_payload)
    print(f"AUDIT historical={len(cells)}/200 records={len(cohort)}/50 groups={len(source_groups)}/45", flush=True)
    return payload


def decompose(paths: config.RunPaths) -> dict[str, Any]:
    inputs_path = paths.audit / "inputs.json"
    if not inputs_path.is_file():
        audit_inputs(paths)
    inputs = read_self_hashed_json(inputs_path)
    cohort = read_self_hashed_json(paths.audit / "cohort.json")["records"]
    cells, pairs = load_historical_records(cohort)
    groups = group_pairs(pairs)
    write_historical_tables(paths.decomposition, cells, pairs, groups)
    summary = historical_analysis(pairs, groups)
    summary.update({
        "protocol_id": config.PROTOCOL_ID,
        "input_audit_sha256": sha256_file(inputs_path),
        "input_bindings_sha256": str(inputs["bindings"]["sha256"]),
        "run_identity": inputs["run_identity"],
        "historical_cell_count": len(cells),
        "historical_pair_count": len(pairs),
        "source_group_count": len(groups) // len(config.MODELS),
    })
    expected = read_json(config.INPUT_BINDINGS)["historical_record_mean_checks"]
    for model in config.MODELS:
        actual = summary["record_means"][model]
        bound = expected[model.lower()]
        for field, bound_field in (("gain_c", "gain_c"), ("benefit_d", "benefit_d"), ("gain_background", "gain_background"), ("c_positive", "c_positive")):
            if field == "c_positive":
                if int(actual[field]) != int(bound[bound_field]):
                    raise ProtocolError(f"historical check failed: {model}/{field}")
            elif abs(float(actual[field]) - float(bound[bound_field])) > 1e-9:
                raise ProtocolError(f"historical check failed: {model}/{field}")
    write_self_hashed_json(paths.decomposition / "summary.json", summary)
    print("DECOMPOSE status=complete Ditto/LeapTalk historical 200 cells", flush=True)
    return summary


def _tool_output(command: Sequence[str], *, input_bytes: bytes | None = None) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(command, input=input_bytes, capture_output=True, check=False)
    except OSError as exc:
        return 127, b"", str(exc).encode("utf-8")
    return int(result.returncode), bytes(result.stdout), bytes(result.stderr)


def _ffmpeg_path() -> Path:
    return config.FFMPEG if config.FFMPEG.is_file() else Path(shutil.which("ffmpeg") or "ffmpeg")


def _ffprobe_path() -> Path:
    return config.FFPROBE if config.FFPROBE.is_file() else Path(shutil.which("ffprobe") or "ffprobe")


def probe_media(path: Path) -> dict[str, Any]:
    command = [str(_ffprobe_path()), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)]
    returncode, stdout, stderr = _tool_output(command)
    if returncode != 0:
        raise ProtocolError(f"ffprobe failed for {path}: {stderr.decode('utf-8', errors='replace')[-500:]}")
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe output is invalid: {path}") from exc
    streams = payload.get("streams", [])
    videos = [item for item in streams if isinstance(item, Mapping) and item.get("codec_type") == "video"]
    audios = [item for item in streams if isinstance(item, Mapping) and item.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) < 1:
        raise ProtocolError(f"media must have one video and an audio stream: {path}")
    video = videos[0]
    frame_count = int(video.get("nb_read_frames", video.get("nb_frames", 0)) or 0)
    try:
        fps = float(Fraction(str(video.get("r_frame_rate", "0/1"))))
    except (ValueError, ZeroDivisionError) as exc:
        raise ProtocolError(f"video frame rate is invalid: {path}") from exc
    pts_command = [str(_ffprobe_path()), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time", "-of", "json", str(path)]
    pts_rc, pts_stdout, _ = _tool_output(pts_command)
    if pts_rc != 0:
        raise ProtocolError(f"first video PTS probe failed: {path}")
    try:
        frame_rows = json.loads(pts_stdout.decode("utf-8")).get("frames", [])
        if not frame_rows:
            raise ValueError("no video frame PTS")
        first_pts = dict(frame_rows[0])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"first video PTS is unavailable: {path}") from exc
    returncode, pcm, stderr = _tool_output([
        str(_ffmpeg_path()), "-v", "error", "-i", str(path), "-map", "0:a:0", "-async", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"
    ])
    if returncode != 0 or len(pcm) == 0 or len(pcm) % 2 or frame_count <= 0 or fps <= 0 or int(video.get("width", 0) or 0) <= 0 or int(video.get("height", 0) or 0) <= 0:
        raise ProtocolError(f"audio decode failed for {path}: {stderr.decode('utf-8', errors='replace')[-500:]}")
    return {
        "path": str(path.resolve()),
        "media_sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "video": {
            "codec_name": video.get("codec_name"),
            "width": int(video.get("width", 0) or 0),
            "height": int(video.get("height", 0) or 0),
            "frame_count": frame_count,
            "fps": fps,
            "start_time": video.get("start_time"),
            "duration": video.get("duration"),
            "time_base": video.get("time_base"),
            "first_frame_pts": first_pts,
        },
        "audio": {
            "codec_name": audios[0].get("codec_name"),
            "sample_rate": int(audios[0].get("sample_rate", 0) or 0),
            "channels": int(audios[0].get("channels", 0) or 0),
            "decoded_sample_count": len(pcm) // 2,
            "decoded_pcm_sha256": sha256_bytes(pcm),
        },
        "ffprobe_command": command,
        "ffprobe_pts_command": pts_command,
    }


def gpu_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    rc, stdout, stderr = _tool_output(command)
    result: dict[str, Any] = {"available": rc == 0, "command": command, "gpus": [], "compute_apps": []}
    if rc != 0:
        result["reason"] = stderr.decode("utf-8", errors="replace").strip() or "nvidia-smi unavailable"
        return result
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 6:
            result["gpus"].append({"index": fields[0], "name": fields[1], "memory_total_mib": fields[2], "memory_used_mib": fields[3], "memory_free_mib": fields[4], "utilization_gpu_percent": fields[5]})
    if not result["gpus"]:
        result["available"] = False
        result["reason"] = "nvidia-smi returned no GPU rows"
    rc, stdout, _ = _tool_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if rc == 0:
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            fields = [item.strip() for item in line.split(",")]
            if fields and any(fields):
                result["compute_apps"].append(fields)
    return result


def _process_tree_pids(root_pid: int) -> set[int]:
    """Return a best-effort set of this run's process IDs for GPU polling."""

    pending = [int(root_pid)]
    result: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="utf-8").split()
        except OSError:
            continue
        pending.extend(int(child) for child in children if child.isdigit())
    return result


def _foreign_gpu_apps(snapshot: Mapping[str, Any], allowed_pids: set[int]) -> list[list[str]]:
    foreign: list[list[str]] = []
    for app in snapshot.get("compute_apps", []):
        if not isinstance(app, list) or len(app) < 2:
            foreign.append(app)
            continue
        try:
            pid = int(str(app[1]))
        except ValueError:
            foreign.append(app)
            continue
        if pid not in allowed_pids:
            foreign.append(app)
    return foreign


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the subprocess group created for this cell."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def estimate_resources(media_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage = shutil.disk_usage(config.REPO)
    valid = [int(row.get("video", {}).get("frame_count", 0)) for row in media_rows if int(row.get("video", {}).get("frame_count", 0)) > 0]
    max_frames = max(valid, default=0)
    # This is deliberately conservative for the official AVI/JPEG pipeline.
    jpeg_and_avi_peak = 64 * (1 << 20) + max_frames * 512 * 1024
    matrix_bytes = config.MAX_NEW_CELLS * max(1, max_frames) * config.LAG_COUNT * 8
    persistent_bytes = matrix_bytes + 24 * 2 * (1 << 20)
    required_free = jpeg_and_avi_peak + persistent_bytes + config.RESERVE_BYTES
    gpu = gpu_snapshot()
    blockers: list[str] = []
    if int(usage.free) < int(required_free):
        blockers.append(f"disk_free={usage.free} < required={required_free} (includes 1GiB reserve)")
    runtime = runtime_bindings()
    blockers.extend(f"runtime_missing={name}" for name, item in runtime.items() if item.get("status") != "PASS")
    if not gpu.get("available"):
        blockers.append("GPU_UNAVAILABLE")
    if gpu.get("compute_apps"):
        blockers.append("GPU_OCCUPIED_BY_OTHER_PROCESS")
    return {
        "status": "READY" if not blockers else "RESOURCE_WAIT",
        "disk": {"path": str(config.REPO), "free_bytes": int(usage.free), "required_free_bytes": int(required_free), "reserve_bytes": config.RESERVE_BYTES},
        "estimate": {"max_video_frames": max_frames, "jpeg_and_avi_peak_bytes": int(jpeg_and_avi_peak), "persistent_matrix_and_metadata_bytes": int(persistent_bytes)},
        "gpu": gpu,
        "runtime_bindings": runtime,
        "blockers": blockers,
        "lease_policy": "one exclusive lease; one official pipeline or SyncNet worker at a time; poll nvidia-smi every 30 seconds",
    }


def runtime_bindings() -> dict[str, Any]:
    paths = {
        "syncnet_python": config.SYNCNET_PYTHON,
        "syncnet_model": config.SYNCNET_MODEL,
        "s3fd_face": config.SYNCNET_ROOT / "detectors" / "s3fd" / "weights" / "sfd_face.pth",
        "syncnet_instance": config.SYNCNET_ROOT / "SyncNetInstance.py",
        "syncnet_model_definition": config.SYNCNET_ROOT / "SyncNetModel.py",
        "run_pipeline": config.SYNCNET_ROOT / "run_pipeline.py",
        "run_syncnet": config.SYNCNET_ROOT / "run_syncnet.py",
        "syncnet_worker": Path(__file__).with_name("syncnet_worker.py"),
        "ffmpeg": _ffmpeg_path(),
        "ffprobe": _ffprobe_path(),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            result[name] = {"path": str(path), "status": "MISSING"}
        else:
            result[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "status": "PASS"}
    return result


def _curve_media_bindings(bindings: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    media = bindings.get("curve_media")
    if not isinstance(media, list) or len(media) != 24:
        raise ProtocolError("curve media binding must contain 24 videos")
    return media


def audit_curve_media(paths: config.RunPaths, bindings: Mapping[str, Any], cohort: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_for = {int(row["sample_id"]): str(row["source_group"]) for row in cohort}
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for bound in _curve_media_bindings(bindings):
        sample_id = int(bound["sample_id"])
        condition = str(bound["condition"])
        by_key[(sample_id, condition)] = bound
        try:
            path = _path(str(bound["path"]))
            if not path.is_file() or sha256_file(path) != str(bound["sha256"]):
                raise ProtocolError("media hash changed or file is missing")
            metadata = probe_media(path)
            metadata.update({"sample_id": sample_id, "condition": condition, "source_group": group_for[sample_id], "bound_sha256": str(bound["sha256"]), "status": "PASS"})
            rows.append(metadata)
        except (KeyError, OSError, ProtocolError) as exc:
            failures.append(f"{sample_id}/{condition}: {type(exc).__name__}: {exc}")
            rows.append({"sample_id": sample_id, "condition": condition, "source_group": group_for.get(sample_id), "status": "FAIL", "reason": str(exc)})
    expected_keys = {(sample_id, condition) for sample_id in config.CURVE_SAMPLE_IDS for condition in config.CONDITIONS}
    if set(by_key) != expected_keys:
        failures.append("curve media binding key set changed")
    resource = estimate_resources([row for row in rows if row.get("status") == "PASS"])
    resource["runtime_bindings"] = runtime_bindings()
    resource_artifact = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "official_pipeline_config": dict(config.OFFICIAL_PIPELINE_CONFIG),
        **resource,
        "media_audit_sha256": None,
    }
    media_payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if not failures else "BLOCKED_INPUT_BINDING",
        "record_count": len(config.CURVE_SAMPLE_IDS),
        "cell_count": len(rows),
        "expected_cell_count": config.MAIN_CURVE_CELLS,
        "rows": sorted(rows, key=lambda row: (int(row["sample_id"]), str(row["condition"]))),
        "failures": failures,
        "resource_plan": resource,
        "bindings_sha256": sha256_file(config.INPUT_BINDINGS),
    }
    write_self_hashed_json(paths.curves / "media_audit.json", media_payload)
    resource_artifact["media_audit_sha256"] = sha256_file(paths.curves / "media_audit.json")
    write_self_hashed_json(paths.audit / "resource_plan.json", resource_artifact)
    return rows, resource


@contextmanager
def gpu_lease(lock_path: Path = config.GPU_LOCK) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("RESOURCE_WAIT: GPU lease is held by another task") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_monitored(command: Sequence[str], *, cwd: Path, log_path: Path, gpu_log: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_samples: list[dict[str, Any]] = []
    environment = os.environ.copy()
    ffmpeg = _ffmpeg_path()
    if ffmpeg.is_file():
        environment["PATH"] = f"{ffmpeg.parent}{os.pathsep}{environment.get('PATH', '')}"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            snapshot = gpu_snapshot()
            gpu_samples.append({"time": time.time(), "snapshot": snapshot})
            foreign = _foreign_gpu_apps(snapshot, _process_tree_pids(process.pid))
            if not snapshot.get("available"):
                _terminate_owned_process(process)
                write_json(gpu_log, {"command": list(command), "samples": gpu_samples, "resource_wait": "GPU_UNAVAILABLE"})
                raise ProtocolError("RESOURCE_WAIT: GPU became unavailable during the cell")
            if foreign:
                _terminate_owned_process(process)
                write_json(gpu_log, {"command": list(command), "samples": gpu_samples, "resource_wait": {"foreign_compute_apps": foreign}})
                raise ProtocolError(f"RESOURCE_WAIT: GPU became occupied by another process: {foreign}")
            time.sleep(30)
        returncode = int(process.returncode or 0)
    gpu_samples.append({"time": time.time(), "snapshot": gpu_snapshot()})
    write_json(gpu_log, {"command": list(command), "samples": gpu_samples})
    if returncode != 0:
        raise ProtocolError(f"command failed with exit code {returncode}: {log_path}")


def _pipeline_selection(cell_dir: Path, reference: str) -> tuple[Path, dict[str, Any]]:
    track_path = cell_dir / "official" / "pywork" / reference / "tracks.pckl"
    crop_dir = cell_dir / "official" / "pycrop" / reference
    require_file(track_path, "official tracks")
    with track_path.open("rb") as handle:
        tracks = pickle.load(handle)
    if not isinstance(tracks, list):
        raise ProtocolError(f"official tracks are malformed: {reference}")
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(tracks):
        if not isinstance(item, Mapping) or not isinstance(item.get("track"), Mapping):
            continue
        frame = np.asarray(item["track"].get("frame"), dtype=np.int64)
        crop = crop_dir / f"{index:05d}.avi"
        if frame.size <= config.MIN_TRACK or not crop.is_file():
            continue
        bbox = np.asarray(item["track"].get("bbox"), dtype=np.float64)
        candidates.append({
            "track_index": index,
            "crop_path": str(crop),
            "frame_count": int(frame.size),
            "start_frame": int(frame[0]),
            "end_frame": int(frame[-1]),
            "frame_indices": [int(value) for value in frame.tolist()],
            "bbox_sha256": sha256_bytes(np.ascontiguousarray(bbox).tobytes()),
            "crop_sha256": sha256_file(crop),
        })
    if not candidates:
        raise ProtocolError(f"no valid official face track: {reference}")
    candidates.sort(key=lambda row: (-int(row["frame_count"]), int(row["start_frame"]), Path(str(row["crop_path"])).name))
    selected = candidates[0]
    destination = cell_dir / "crop.avi"
    if destination.exists():
        if sha256_file(destination) != selected["crop_sha256"]:
            raise ProtocolError(f"existing selected crop differs: {reference}")
    else:
        Path(str(selected["crop_path"])).replace(destination)
        selected["crop_sha256"] = sha256_file(destination)
    evidence = {
        "schema_version": 1,
        "selection_rule": "sort by actual frame count descending, start frame ascending, crop filename lexicographically",
        "official_pipeline_config": dict(config.OFFICIAL_PIPELINE_CONFIG),
        "reference": reference,
        "candidates": candidates,
        "selected": {**selected, "crop_path": str(destination), "crop_sha256": sha256_file(destination)},
    }
    write_self_hashed_json(cell_dir / "crop_selection.json", evidence)
    return destination, evidence


def _cleanup_official_temp(cell_dir: Path) -> None:
    official = cell_dir / "official"
    if official.is_dir():
        shutil.rmtree(official)


def _cleanup_interrupted_cell(
    cell_dir: Path,
    *,
    matrix_path: Path,
    result_path: Path,
    crop: Path,
    selection_path: Path,
    repeat: bool,
) -> None:
    """Remove only uncommitted artifacts after a resource-triggered stop.

    A resource wait is resumable only when the next invocation cannot mistake
    a partially written crop/matrix/result for a valid cache.  The frozen crop
    belongs to the main cell and is intentionally retained for repeat controls;
    a main cell's own crop is removed because it may have been truncated by a
    killed official pipeline.
    """

    _cleanup_official_temp(cell_dir)
    shutil.rmtree(cell_dir / "worker_tmp", ignore_errors=True)
    matrix_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    if not repeat:
        crop.unlink(missing_ok=True)
        selection_path.unlink(missing_ok=True)


def _score_crop(crop: Path, *, cell_dir: Path, reference: str, matrix_path: Path, result_path: Path, log_name: str) -> dict[str, Any]:
    worker = Path(__file__).with_name("syncnet_worker.py")
    log_path = cell_dir / "logs" / log_name
    tmp_dir = cell_dir / "worker_tmp"
    command = [
        str(config.SYNCNET_PYTHON), str(worker), "--media", str(crop), "--model", str(config.SYNCNET_MODEL),
        "--tmp-dir", str(tmp_dir), "--reference", reference, "--matrix", str(matrix_path), "--result", str(result_path),
        "--batch-size", str(config.BATCH_SIZE), "--vshift", str(config.VSHIFT), "--ffmpeg", str(_ffmpeg_path()),
    ]
    _run_monitored(command, cwd=config.SYNCNET_ROOT, log_path=log_path, gpu_log=cell_dir / "logs" / f"{log_name}.gpu.json")
    if not matrix_path.is_file() or not result_path.is_file():
        raise ProtocolError(f"SyncNet worker did not produce a matrix: {reference}")
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ProtocolError(f"SyncNet matrix is malformed: {reference}")
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ProtocolError(f"SyncNet worker result is malformed: {reference}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def _run_one_curve_cell(media_row: Mapping[str, Any], paths: config.RunPaths, *, repeat: bool = False) -> dict[str, Any]:
    sample_id = int(media_row["sample_id"])
    condition = str(media_row["condition"])
    cell_name = f"{sample_id}_{condition}" + ("_repeat" if repeat else "")
    cell_dir = paths.curves / "cells" / cell_name
    matrix_path = paths.curves / "matrices" / f"{cell_name}.npy"
    result_path = cell_dir / "worker.json"
    cell_dir.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    crop = cell_dir / "crop.avi"
    selection_path = cell_dir / "crop_selection.json"
    if not repeat:
        if matrix_path.is_file() and selection_path.is_file() and result_path.is_file() and crop.is_file():
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
            selection = read_self_hashed_json(selection_path)
            worker = read_json(result_path)
            if not isinstance(worker, dict):
                raise ProtocolError(f"cached worker result is malformed: {cell_name}")
            matrix_sha256 = sha256_file(matrix_path)
            if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or matrix.shape[0] < 1 or not np.isfinite(matrix).all() or worker.get("matrix_sha256") != matrix_sha256 or worker.get("matrix_shape") != [int(item) for item in matrix.shape] or selection.get("official_pipeline_config") != config.OFFICIAL_PIPELINE_CONFIG:
                raise ProtocolError(f"cached curve cell metadata is inconsistent: {cell_name}")
            if selection.get("selected", {}).get("crop_sha256") != sha256_file(crop):
                raise ProtocolError(f"cached selected crop metadata is inconsistent: {cell_name}")
            cached_row = {
                "sample_id": sample_id, "condition": condition, "source_group": str(media_row["source_group"]), "status": "complete",
                "cell_name": cell_name, "matrix": str(matrix_path), "matrix_sha256": matrix_sha256, "matrix_shape": [int(item) for item in matrix.shape],
                "crop": str(crop), "crop_sha256": sha256_file(crop), "crop_selection": str(selection_path), "media": media_row["path"], "media_sha256": media_row["media_sha256"],
                "worker_result": str(result_path), "worker_result_sha256": sha256_file(result_path), "worker": worker, "reused": True,
                "official_config": {**config.OFFICIAL_PIPELINE_CONFIG, "vshift": config.VSHIFT, "batch_size": config.BATCH_SIZE},
                "selection_sha256": sha256_file(selection_path), "selection": selection,
            }
            _validate_resume_main_row(paths, cached_row, media_row, f"cached/{cell_name}")
            return cached_row
        if any(item.exists() for item in (matrix_path, selection_path, result_path, crop)):
            raise ProtocolError(f"partial curve cell cannot be resumed: {cell_name}")
    else:
        crop = paths.curves / "cells" / f"151_{condition}" / "crop.avi"
        selection_path = paths.curves / "cells" / f"151_{condition}" / "crop_selection.json"
        if not crop.is_file():
            raise ProtocolError(f"repeat crop is missing: {crop}")
    reference = f"lrs3_tts_gain_{cell_name}"
    selection: dict[str, Any]
    try:
        if not repeat:
            command = [str(config.SYNCNET_PYTHON), str(config.SYNCNET_ROOT / "run_pipeline.py"), "--videofile", str(media_row["path"]), "--reference", reference, "--data_dir", str(cell_dir / "official")]
            for option, value in config.OFFICIAL_PIPELINE_CONFIG.items():
                command.extend([f"--{option}", str(value)])
            command.append("--overwrite")
            try:
                _run_monitored(command, cwd=config.SYNCNET_ROOT, log_path=cell_dir / "logs" / "pipeline.log", gpu_log=cell_dir / "logs" / "pipeline.gpu.json")
                crop, selection = _pipeline_selection(cell_dir, reference)
            finally:
                _cleanup_official_temp(cell_dir)
        else:
            selection = read_self_hashed_json(paths.curves / "cells" / f"151_{condition}" / "crop_selection.json")
        result = _score_crop(crop, cell_dir=cell_dir, reference=reference, matrix_path=matrix_path, result_path=result_path, log_name="syncnet.log")
    except ProtocolError as exc:
        if str(exc).startswith("RESOURCE_WAIT"):
            _cleanup_interrupted_cell(
                cell_dir,
                matrix_path=matrix_path,
                result_path=result_path,
                crop=crop,
                selection_path=selection_path,
                repeat=repeat,
            )
        raise
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    row = {
        "sample_id": sample_id, "condition": condition, "source_group": str(media_row["source_group"]), "status": "complete", "cell_name": cell_name,
        "matrix": str(matrix_path), "matrix_sha256": sha256_file(matrix_path), "matrix_shape": [int(item) for item in matrix.shape],
        "crop": str(crop), "crop_sha256": sha256_file(crop), "crop_selection": str(selection_path),
        "media": media_row["path"], "media_sha256": media_row["media_sha256"], "worker_result": str(result_path), "worker_result_sha256": sha256_file(result_path),
        "worker": result, "reused": False, "official_config": {**config.OFFICIAL_PIPELINE_CONFIG, "vshift": config.VSHIFT, "batch_size": config.BATCH_SIZE},
        "selection_sha256": sha256_file(selection_path), "selection": selection,
    }
    return row


def _decode_pcm(path: Path) -> bytes:
    rc, stdout, stderr = _tool_output([str(_ffmpeg_path()), "-v", "error", "-i", str(path), "-map", "0:a:0", "-async", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"])
    if rc != 0:
        raise ProtocolError(f"cannot decode PCM: {path}: {stderr.decode('utf-8', errors='replace')[-500:]}")
    return stdout


def _make_delayed_media(crop: Path, output: Path, work_dir: Path) -> dict[str, Any]:
    pcm = _decode_pcm(crop)
    shift_bytes = 3_200 * 2
    if len(pcm) <= shift_bytes:
        raise ProtocolError("crop audio is too short for 200 ms delay")
    delayed = b"\0" * shift_bytes + pcm[:-shift_bytes]
    delayed_wav = work_dir / "delay.wav"
    work_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(delayed_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(delayed)
    video_only = work_dir / "video.mkv"
    rc, _, stderr = _tool_output([str(_ffmpeg_path()), "-y", "-v", "error", "-i", str(crop), "-map", "0:v:0", "-c:v", "copy", "-an", "-f", "matroska", str(video_only)])
    if rc != 0:
        raise ProtocolError(f"cannot isolate control video: {stderr.decode('utf-8', errors='replace')[-500:]}")
    rc, _, stderr = _tool_output([str(_ffmpeg_path()), "-y", "-v", "error", "-i", str(video_only), "-i", str(delayed_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-f", "matroska", str(output)])
    if rc != 0:
        raise ProtocolError(f"cannot mux delayed control media: {stderr.decode('utf-8', errors='replace')[-500:]}")
    original_video = probe_media(crop)["video"]
    delayed_video = probe_media(output)["video"]
    video_fields = ("codec_name", "width", "height", "frame_count", "fps", "start_time", "duration", "time_base", "first_frame_pts")
    video_unchanged = all(original_video.get(field) == delayed_video.get(field) for field in video_fields)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "input_pcm_sha256": sha256_bytes(pcm),
        "delayed_pcm_sha256": sha256_bytes(delayed),
        "delay_samples": 3_200,
        "sample_count": len(pcm) // 2,
        "original_video": original_video,
        "delayed_video": delayed_video,
        "video_unchanged": video_unchanged,
    }


def _control_metrics(original: np.ndarray, delayed: np.ndarray) -> dict[str, Any]:
    from .analysis import _curve_metrics

    common = list(range(20, min(original.shape[0], delayed.shape[0]) - 20))
    if len(common) < 25:
        return {"status": "CONTROL_FAILED", "reason": "common support has fewer than 25 rows", "common_rows": common}
    original_metrics = _curve_metrics(original, common, label="control/original")
    delayed_metrics = _curve_metrics(delayed, common, label="control/delayed")
    expected = int(delayed_metrics["official_offset"]) - int(original_metrics["official_offset"])
    original_argmin = int(original_metrics["min_index"])
    delayed_curve = np.asarray(delayed_metrics["curve"], dtype=np.float64)
    original_curve = np.asarray(original_metrics["curve"], dtype=np.float64)
    boundary_ok = not original_metrics["boundary_best"] and not delayed_metrics["boundary_best"]
    pass_control = abs(expected + 5) <= 1 and float(delayed_curve[original_argmin]) > float(original_curve[original_argmin]) and boundary_ok
    return {
        "status": "CONTROL_PASS" if pass_control else "CONTROL_FAILED",
        "common_rows": common,
        "original": original_metrics,
        "delayed": delayed_metrics,
        "offset_change_delayed_minus_original": expected,
        "expected_offset_change": -5,
        "original_argmin_delayed_distance_higher": bool(float(delayed_curve[original_argmin]) > float(original_curve[original_argmin])),
        "original_boundary_best": bool(original_metrics["boundary_best"]),
        "delayed_boundary_best": bool(delayed_metrics["boundary_best"]),
        "boundary_ok": boundary_ok,
        "rule": "delayed official offset must move -5 +/- 1 and delayed distance at original argmin must rise",
    }


def _control_rows_from_checkpoint(paths: config.RunPaths) -> list[dict[str, Any]]:
    checkpoint = paths.curves / "controls.json"
    if not checkpoint.is_file():
        return []
    value = read_self_hashed_json(checkpoint)
    if value.get("protocol_id") != config.PROTOCOL_ID:
        raise ProtocolError("control checkpoint belongs to another protocol")
    if value.get("run_identity", {}).get("identity_sha256") != current_run_identity().get("identity_sha256"):
        raise ProtocolError("control checkpoint run identity differs; choose a new run id")
    rows = value.get("rows", [])
    if not isinstance(rows, list):
        raise ProtocolError("control checkpoint rows are malformed")
    return [dict(row) for row in rows]


def _write_control_checkpoint(paths: config.RunPaths, rows: Sequence[Mapping[str, Any]], *, status: str) -> None:
    write_self_hashed_json(paths.curves / "controls.json", {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": status,
        "budget": config.MAX_NEW_CELLS,
        "main_cells": 24,
        "repeat_cells": 2,
        "delay_cells": 2,
        "completed_cells": len(rows),
        "rows": list(rows),
        "run_identity": current_run_identity(),
    })


def _failed_control_row(original: Mapping[str, Any], control: str, condition: str, error: BaseException) -> dict[str, Any]:
    return {
        "control": control,
        "sample_id": 151,
        "condition": condition,
        "crop": original["crop"],
        "crop_sha256": original["crop_sha256"],
        "crop_selection": original["crop_selection"],
        "selection_sha256": original["selection_sha256"],
        "official_config": original["official_config"],
        "status": "CONTROL_FAILED",
        "execution_failed": True,
        "reason": f"{type(error).__name__}: {error}",
    }


def _failed_curve_row(media_row: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    sample_id = int(media_row["sample_id"])
    condition = str(media_row["condition"])
    return {
        "sample_id": sample_id,
        "condition": condition,
        "source_group": str(media_row["source_group"]),
        "status": "FAILED",
        "cell_name": f"{sample_id}_{condition}",
        "media": media_row["path"],
        "media_sha256": media_row["media_sha256"],
        "reason": f"{type(error).__name__}: {error}",
    }


def _run_artifact(paths: config.RunPaths, value: Any, label: str) -> Path:
    target = Path(str(value))
    if not target.is_absolute():
        target = paths.root / target
    resolved = target.resolve()
    try:
        resolved.relative_to(paths.root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the run root") from exc
    if not resolved.is_file():
        raise ProtocolError(f"{label} is missing: {resolved}")
    return resolved


def _run_artifact_reference(paths: config.RunPaths, value: Any, label: str) -> Path:
    target = Path(str(value))
    if not target.is_absolute():
        target = paths.root / target
    resolved = target.resolve()
    try:
        resolved.relative_to(paths.root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the run root") from exc
    return resolved


def _validate_resume_selection(paths: config.RunPaths, row: Mapping[str, Any], label: str) -> None:
    selection_path = _run_artifact(paths, row.get("crop_selection"), f"{label} selection")
    if sha256_file(selection_path) != row.get("selection_sha256"):
        raise ProtocolError(f"{label} selection hash changed")
    selection = read_self_hashed_json(selection_path)
    if selection.get("official_pipeline_config") != config.OFFICIAL_PIPELINE_CONFIG:
        raise ProtocolError(f"{label} selection configuration changed")
    if selection.get("selection_rule") != "sort by actual frame count descending, start frame ascending, crop filename lexicographically":
        raise ProtocolError(f"{label} selection rule changed")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolError(f"{label} selection candidates are missing")
    ordered = sorted(candidates, key=lambda item: (-int(item["frame_count"]), int(item["start_frame"]), Path(str(item["crop_path"])).name))
    if candidates != ordered:
        raise ProtocolError(f"{label} selection order changed")
    track_indices: set[int] = set()
    for candidate in candidates:
        frame_indices = candidate.get("frame_indices")
        try:
            frame_count = int(candidate.get("frame_count", 0))
            track_index = int(candidate.get("track_index"))
            start_frame = int(candidate.get("start_frame"))
            end_frame = int(candidate.get("end_frame"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{label} track candidate metadata is malformed") from exc
        if (
            not isinstance(frame_indices, list)
            or not frame_indices
            or track_index < 0
            or track_index in track_indices
            or frame_count != len(frame_indices)
            or frame_count <= config.MIN_TRACK
            or start_frame != int(frame_indices[0])
            or end_frame != int(frame_indices[-1])
        ):
            raise ProtocolError(f"{label} track candidate metadata is invalid")
        _run_artifact_reference(paths, candidate.get("crop_path"), f"{label} candidate crop")
        track_indices.add(track_index)
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ProtocolError(f"{label} selected track is missing")
    if selected != candidates[0] and any(
        selected.get(field) != candidates[0].get(field)
        for field in ("track_index", "frame_count", "start_frame", "end_frame", "frame_indices", "bbox_sha256")
    ):
        raise ProtocolError(f"{label} selected track differs from the first candidate")
    crop = _run_artifact(paths, row.get("crop"), f"{label} crop")
    if str(crop) != str(selected.get("crop_path")) or row.get("crop_sha256") != selected.get("crop_sha256") or sha256_file(crop) != selected.get("crop_sha256"):
        raise ProtocolError(f"{label} selected crop binding changed")
    if row.get("official_config") != {**config.OFFICIAL_PIPELINE_CONFIG, "vshift": config.VSHIFT, "batch_size": config.BATCH_SIZE}:
        raise ProtocolError(f"{label} scoring configuration changed")


def _validate_resume_main_row(paths: config.RunPaths, row: Mapping[str, Any], media_row: Mapping[str, Any], label: str) -> None:
    if row.get("status") != "complete" or int(row.get("sample_id", -1)) != int(media_row["sample_id"]) or str(row.get("condition")) != str(media_row["condition"]):
        raise ProtocolError(f"{label} cell identity or status changed")
    if str(row.get("media")) != str(media_row["path"]) or row.get("media_sha256") != media_row.get("media_sha256"):
        raise ProtocolError(f"{label} media binding changed")
    _validate_resume_selection(paths, row, label)
    _validate_resume_worker(paths, {**row, "worker_media": row.get("crop")}, label)


def _validate_resume_failed_main_row(row: Mapping[str, Any], media_row: Mapping[str, Any], label: str) -> None:
    if row.get("status") != "FAILED" or int(row.get("sample_id", -1)) != int(media_row["sample_id"]) or str(row.get("condition")) != str(media_row["condition"]):
        raise ProtocolError(f"{label} failed-cell identity or status changed")
    if str(row.get("media")) != str(media_row["path"]) or row.get("media_sha256") != media_row.get("media_sha256") or not str(row.get("reason", "")).strip():
        raise ProtocolError(f"{label} failed-cell binding is malformed")


def _validate_resume_worker(paths: config.RunPaths, row: Mapping[str, Any], label: str) -> np.ndarray:
    matrix_path = _run_artifact(paths, row.get("matrix"), f"{label} matrix")
    if sha256_file(matrix_path) != row.get("matrix_sha256"):
        raise ProtocolError(f"{label} matrix hash changed")
    raw_matrix = np.load(matrix_path, allow_pickle=False)
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if str(raw_matrix.dtype) != "float64" or matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or matrix.shape[0] < 1 or not np.isfinite(matrix).all() or row.get("matrix_shape") != [int(item) for item in matrix.shape]:
        raise ProtocolError(f"{label} matrix metadata is invalid")
    result_path = _run_artifact(paths, row.get("worker_result"), f"{label} worker result")
    if sha256_file(result_path) != row.get("worker_result_sha256"):
        raise ProtocolError(f"{label} worker result hash changed")
    result = read_json(result_path)
    if not isinstance(result, dict) or result != row.get("worker"):
        raise ProtocolError(f"{label} worker payload changed")
    expected_shape = [int(item) for item in matrix.shape]
    if result.get("matrix_sha256") != row.get("matrix_sha256") or result.get("matrix_shape") != expected_shape or result.get("matrix_dtype") != str(raw_matrix.dtype) or result.get("vshift") != config.VSHIFT or result.get("batch_size") != config.BATCH_SIZE or result.get("new_forward") is not True or result.get("extracted_pcm_verified") is not True:
        raise ProtocolError(f"{label} worker metadata is invalid")
    if result.get("model_sha256") != sha256_file(config.SYNCNET_MODEL) or result.get("worker_code_sha256") != sha256_file(Path(__file__).with_name("syncnet_worker.py")) or result.get("syncnet_instance_sha256") != sha256_file(config.SYNCNET_ROOT / "SyncNetInstance.py") or result.get("syncnet_model_definition_sha256") != sha256_file(config.SYNCNET_ROOT / "SyncNetModel.py"):
        raise ProtocolError(f"{label} worker implementation hash changed")
    ffmpeg = _ffmpeg_path()
    if result.get("ffmpeg") != str(ffmpeg.resolve()) or result.get("ffmpeg_sha256") != sha256_file(ffmpeg):
        raise ProtocolError(f"{label} ffmpeg binding changed")
    worker_media = Path(str(row.get("worker_media", row.get("media", "")))).resolve()
    if not worker_media.is_file() or result.get("media") != str(worker_media) or result.get("media_sha256") != sha256_file(worker_media):
        raise ProtocolError(f"{label} worker media binding changed")
    pcm = _decode_pcm(worker_media)
    pcm_sha256 = sha256_bytes(pcm)
    if result.get("input_pcm_sha256") != pcm_sha256 or result.get("extracted_pcm_sha256") != pcm_sha256:
        raise ProtocolError(f"{label} PCM binding changed")
    mean_curve = np.asarray(matrix.astype(np.float32).mean(axis=0, dtype=np.float32), dtype=np.float64)
    expected_offset = config.VSHIFT - int(np.argmin(mean_curve))
    expected_confidence = float(np.median(mean_curve) - np.min(mean_curve))
    confidence = result.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} worker confidence is invalid") from exc
    if not math.isfinite(confidence_value) or result.get("offset") != expected_offset or abs(confidence_value - expected_confidence) > config.MATRIX_TOLERANCE:
        raise ProtocolError(f"{label} worker scalar summary changed")
    return matrix


def _validate_resume_controls(paths: config.RunPaths, controls: Sequence[Mapping[str, Any]], main_rows: Mapping[tuple[int, str], Mapping[str, Any]]) -> None:
    expected_keys = {(control, condition) for control in ("repeat", "delay_200ms") for condition in config.CONDITIONS}
    actual_keys = {(str(row.get("control")), str(row.get("condition"))) for row in controls}
    if len(controls) > config.REPEAT_CELLS + config.DELAY_CELLS or not actual_keys.issubset(expected_keys) or len(actual_keys) != len(controls):
        raise ProtocolError("control checkpoint identities are invalid")
    for row in controls:
        control = str(row["control"])
        condition = str(row["condition"])
        if int(row.get("sample_id", -1)) != 151 or condition not in config.CONDITIONS:
            raise ProtocolError(f"control checkpoint sample identity is invalid: {control}/{condition}")
        main = main_rows.get((151, condition))
        if main is None:
            raise ProtocolError(f"control checkpoint has no main cell: {condition}")
        for field in ("crop", "crop_sha256", "crop_selection", "selection_sha256", "official_config"):
            if row.get(field) != main.get(field):
                raise ProtocolError(f"control checkpoint {field} differs from main cell: {control}/{condition}")
        selection_path = _run_artifact(paths, row.get("crop_selection"), f"control/{control}/{condition} selection")
        selection = read_self_hashed_json(selection_path)
        if selection.get("official_pipeline_config") != config.OFFICIAL_PIPELINE_CONFIG:
            raise ProtocolError(f"control selection configuration changed: {control}/{condition}")
        if row.get("execution_failed") is True:
            if row.get("status") != "CONTROL_FAILED" or not str(row.get("reason", "")).strip():
                raise ProtocolError(f"failed control checkpoint is malformed: {control}/{condition}")
            continue
        worker_row: Mapping[str, Any] = {**row, "worker_media": row.get("crop")}
        if control == "delay_200ms":
            delayed_media = row.get("media")
            if not isinstance(delayed_media, Mapping):
                raise ProtocolError(f"delay control media metadata is missing: {condition}")
            worker_row = {**row, "media": delayed_media.get("output"), "worker_media": delayed_media.get("output")}
        matrix = _validate_resume_worker(paths, worker_row, f"control/{control}/{condition}")
        original_matrix = np.asarray(np.load(_run_artifact(paths, main.get("matrix"), f"original/{condition} matrix"), allow_pickle=False), dtype=np.float64)
        if control == "repeat":
            same_shape = original_matrix.shape == matrix.shape
            max_abs_error = float(np.max(np.abs(original_matrix - matrix))) if same_shape else None
            same_offset = row.get("worker", {}).get("offset") == main.get("worker", {}).get("offset")
            if row.get("same_shape") != same_shape or row.get("max_abs_error") != max_abs_error or row.get("same_offset") != same_offset:
                raise ProtocolError(f"repeat control checkpoint changed: {condition}")
            expected_status = "CONTROL_PASS" if same_shape and max_abs_error is not None and max_abs_error <= config.MATRIX_TOLERANCE and same_offset else "CONTROL_FAILED"
            if row.get("status") != expected_status:
                raise ProtocolError(f"repeat control status changed: {condition}")
            continue
        media = row.get("media")
        if not isinstance(media, Mapping):
            raise ProtocolError(f"delay control media metadata is missing: {condition}")
        delayed_path = _run_artifact(paths, media.get("output"), f"delay/{condition} media")
        if media.get("output_sha256") != sha256_file(delayed_path):
            raise ProtocolError(f"delay control media hash changed: {condition}")
        original_pcm = _decode_pcm(Path(str(main["crop"])).resolve())
        delayed_pcm = _decode_pcm(delayed_path)
        shift_bytes = 3_200 * 2
        expected_pcm = b"\0" * shift_bytes + original_pcm[:-shift_bytes]
        if delayed_pcm != expected_pcm or media.get("input_pcm_sha256") != sha256_bytes(original_pcm) or media.get("delayed_pcm_sha256") != sha256_bytes(expected_pcm) or media.get("delay_samples") != 3_200 or media.get("sample_count") != len(original_pcm) // 2:
            raise ProtocolError(f"delay control audio transform changed: {condition}")
        original_video = probe_media(Path(str(main["crop"])).resolve())["video"]
        delayed_video = probe_media(delayed_path)["video"]
        video_fields = ("codec_name", "width", "height", "frame_count", "fps", "start_time", "duration", "time_base", "first_frame_pts")
        video_unchanged = all(original_video.get(field) == delayed_video.get(field) for field in video_fields)
        if media.get("original_video") != original_video or media.get("delayed_video") != delayed_video or media.get("video_unchanged") != video_unchanged:
            raise ProtocolError(f"delay control video binding changed: {condition}")
        expected_metrics = _control_metrics(original_matrix, matrix)
        if not video_unchanged:
            expected_metrics["status"] = "CONTROL_FAILED"
        for field in (
            "status",
            "common_rows",
            "offset_change_delayed_minus_original",
            "expected_offset_change",
            "original_argmin_delayed_distance_higher",
            "original_boundary_best",
            "delayed_boundary_best",
            "boundary_ok",
            "rule",
        ):
            if row.get(field) != expected_metrics.get(field):
                raise ProtocolError(f"delay control metric changed: {condition}/{field}")


def _main_row_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    complete = sum(row.get("status") == "complete" for row in rows)
    failed = sum(row.get("status") == "FAILED" for row in rows)
    return int(complete), int(failed)


def _write_curve_checkpoint(
    paths: config.RunPaths,
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    status: str,
    failures: Sequence[str],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    complete_count, failed_count = _main_row_counts(rows)
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": status,
        "expected_main_cells": config.MAIN_CURVE_CELLS,
        "completed_main_cells": complete_count,
        "failed_main_cells": failed_count,
        "completed_control_cells": len(controls),
        "new_syncnet_cells_attempted": len(rows) + len(controls),
        "rows": list(rows),
        "failures": list(failures),
        "resource_plan": dict(resource),
        "input_audit_sha256": sha256_file(paths.audit / "inputs.json"),
        "run_identity": current_run_identity(),
    }
    if status == "complete":
        payload["total_new_syncnet_cells"] = config.MAX_NEW_CELLS
        payload["model_sha256"] = sha256_file(config.SYNCNET_MODEL)
        payload["code_sha256"] = current_run_identity()["code_sha256"]
    return write_self_hashed_json(paths.curves / "manifest.json", payload)


def run_curves(paths: config.RunPaths) -> dict[str, Any]:
    if not (paths.audit / "inputs.json").is_file():
        audit_inputs(paths)
    bindings = read_json(config.INPUT_BINDINGS)
    cohort = read_self_hashed_json(paths.audit / "cohort.json")["records"]
    media_rows, resource = audit_curve_media(paths, bindings, cohort)
    existing_manifest = read_self_hashed_json(paths.curves / "manifest.json") if (paths.curves / "manifest.json").is_file() else None
    existing_controls = _control_rows_from_checkpoint(paths)
    if existing_manifest is not None and existing_manifest.get("status") == "complete" and len(existing_manifest.get("rows", [])) == config.MAIN_CURVE_CELLS and len(existing_controls) == 4:
        expected_media = {(int(row["sample_id"]), str(row["condition"])): row for row in media_rows}
        cached_rows = existing_manifest.get("rows", [])
        cached_keys = {(int(row.get("sample_id", -1)), str(row.get("condition", ""))) for row in cached_rows}
        expected_keys = {(sample_id, condition) for sample_id in config.CURVE_SAMPLE_IDS for condition in config.CONDITIONS}
        if cached_keys != expected_keys or existing_manifest.get("input_audit_sha256") != sha256_file(paths.audit / "inputs.json") or existing_manifest.get("model_sha256") != sha256_file(config.SYNCNET_MODEL) or existing_manifest.get("total_new_syncnet_cells") != config.MAX_NEW_CELLS:
            raise ProtocolError("complete curve cache metadata is inconsistent; choose a new run id")
        cached_by_key = {(int(row["sample_id"]), str(row["condition"])): row for row in cached_rows}
        for key, row in cached_by_key.items():
            _validate_resume_main_row(paths, row, expected_media[key], f"cached/{key[0]}/{key[1]}")
        controls_path = paths.curves / "controls.json"
        controls_artifact = read_self_hashed_json(controls_path)
        if controls_artifact.get("status") != "complete" or controls_artifact.get("completed_cells") != config.REPEAT_CELLS + config.DELAY_CELLS:
            raise ProtocolError("complete curve cache control checkpoint is inconsistent; choose a new run id")
        _validate_resume_controls(paths, existing_controls, cached_by_key)
        return existing_manifest
    if any(row.get("status") != "PASS" for row in media_rows):
        failures = [row.get("reason", "media audit failed") for row in media_rows if row.get("status") != "PASS"]
        payload = _write_curve_checkpoint(paths, [], [], status="BLOCKED_INPUT_BINDING", failures=failures, resource=resource)
        _write_control_checkpoint(paths, [], status="BLOCKED_INPUT_BINDING")
        return payload
    if resource["status"] != "READY":
        retained_rows = existing_manifest.get("rows", []) if isinstance(existing_manifest, Mapping) and existing_manifest.get("status") == "RESOURCE_WAIT" else []
        if retained_rows:
            expected_media = {(int(row["sample_id"]), str(row["condition"])): row for row in media_rows}
            retained_keys = {(int(row.get("sample_id", -1)), str(row.get("condition", ""))) for row in retained_rows}
            if not retained_keys.issubset(set(expected_media)):
                raise ProtocolError("resource-wait cache contains an unexpected cell")
            for row in retained_rows:
                key = (int(row["sample_id"]), str(row["condition"]))
                if row.get("status") == "FAILED":
                    _validate_resume_failed_main_row(row, expected_media[key], f"retained/{key[0]}/{key[1]}")
                else:
                    _validate_resume_main_row(paths, row, expected_media[key], f"retained/{key[0]}/{key[1]}")
            if existing_controls:
                _validate_resume_controls(
                    paths,
                    existing_controls,
                    {(int(row["sample_id"]), str(row["condition"])): row for row in retained_rows},
                )
        payload = _write_curve_checkpoint(paths, retained_rows, existing_controls, status="RESOURCE_WAIT", failures=resource["blockers"], resource=resource)
        payload["zero_new_forward_executed"] = not retained_rows and not existing_controls
        payload = write_self_hashed_json(paths.curves / "manifest.json", payload)
        _write_control_checkpoint(paths, existing_controls, status="RESOURCE_WAIT")
        print(f"CURVES status=RESOURCE_WAIT reason={resource['blockers'][0] if resource['blockers'] else 'unknown'}", flush=True)
        return payload
    if not config.SYNCNET_PYTHON.is_file() or not config.SYNCNET_MODEL.is_file():
        payload = _write_curve_checkpoint(paths, [], existing_controls, status="RESOURCE_WAIT", failures=["SyncNet runtime or model is missing"], resource=resource)
        _write_control_checkpoint(paths, existing_controls, status="RESOURCE_WAIT")
        return payload
    media_index = {(int(row["sample_id"]), str(row["condition"])): row for row in media_rows}
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = list(existing_controls)
    control_keys = {(str(row.get("control")), str(row.get("condition"))) for row in controls}
    main_failures: list[str] = []
    try:
        with gpu_lease(config.GPU_LOCK):
            initial_snapshot = gpu_snapshot()
            foreign = _foreign_gpu_apps(initial_snapshot, {os.getpid()})
            if not initial_snapshot.get("available"):
                raise ProtocolError("RESOURCE_WAIT: GPU became unavailable before the first cell")
            if foreign:
                raise ProtocolError(f"RESOURCE_WAIT: GPU is occupied by another process: {foreign}")
            for sample_id in config.CURVE_SAMPLE_IDS:
                for condition in config.CONDITIONS:
                    try:
                        rows.append(_run_one_curve_cell(media_index[(sample_id, condition)], paths))
                    except Exception as exc:
                        if str(exc).startswith("RESOURCE_WAIT"):
                            raise
                        failed = _failed_curve_row(media_index[(sample_id, condition)], exc)
                        rows.append(failed)
                        main_failures.append(f"{sample_id}/{condition}: {failed['reason']}")
                        _write_curve_checkpoint(paths, rows, controls, status="PARTIAL", failures=main_failures, resource=resource)
                        _write_control_checkpoint(paths, controls, status="partial")
                    complete_count, _ = _main_row_counts(rows)
                    print(f"CURVES main={complete_count}/24 {sample_id}/{condition}", flush=True)
            row_by_key = {(int(row["sample_id"]), str(row["condition"])): row for row in rows}
            controls_ready = all(row_by_key.get((151, condition), {}).get("status") == "complete" for condition in config.CONDITIONS)
            if controls_ready:
                _validate_resume_controls(paths, controls, row_by_key)
            control_conditions = config.CONDITIONS if controls_ready else ()
            for condition in control_conditions:
                original = row_by_key[(151, condition)]
                original_matrix = np.asarray(np.load(original["matrix"], allow_pickle=False), dtype=np.float64)
                if ("repeat", condition) not in control_keys:
                    try:
                        repeat_cell = _run_one_curve_cell(media_index[(151, condition)], paths, repeat=True)
                        repeat_matrix = np.asarray(np.load(repeat_cell["matrix"], allow_pickle=False), dtype=np.float64)
                        repeat_result = {
                            "control": "repeat",
                            "sample_id": 151,
                            "condition": condition,
                            "matrix": repeat_cell["matrix"],
                            "matrix_sha256": repeat_cell["matrix_sha256"],
                            "matrix_shape": [int(item) for item in repeat_matrix.shape],
                            "worker": repeat_cell["worker"],
                            "worker_result": repeat_cell["worker_result"],
                            "worker_result_sha256": repeat_cell["worker_result_sha256"],
                            "crop": repeat_cell["crop"],
                            "crop_sha256": repeat_cell["crop_sha256"],
                            "crop_selection": repeat_cell["crop_selection"],
                            "selection_sha256": repeat_cell["selection_sha256"],
                            "official_config": repeat_cell["official_config"],
                            "max_abs_error": float(np.max(np.abs(original_matrix - repeat_matrix))) if original_matrix.shape == repeat_matrix.shape else None,
                            "same_shape": bool(original_matrix.shape == repeat_matrix.shape),
                            "same_offset": bool(original.get("worker", {}).get("offset") == repeat_cell.get("worker", {}).get("offset")),
                        }
                        repeat_result["status"] = "CONTROL_PASS" if repeat_result["same_shape"] and repeat_result["max_abs_error"] is not None and float(repeat_result["max_abs_error"]) <= config.MATRIX_TOLERANCE and repeat_result["same_offset"] else "CONTROL_FAILED"
                    except Exception as exc:
                        if str(exc).startswith("RESOURCE_WAIT"):
                            raise
                        repeat_result = _failed_control_row(original, "repeat", condition, exc)
                    controls.append(repeat_result)
                    control_keys.add(("repeat", condition))
                    _write_control_checkpoint(paths, controls, status="partial")
                delayed_media = paths.curves / "controls" / f"151_{condition}_delay_200ms.mkv"
                if ("delay_200ms", condition) not in control_keys:
                    try:
                        delayed_info = _make_delayed_media(Path(str(original["crop"])), delayed_media, paths.curves / "controls" / f"151_{condition}_delay_work")
                        delayed_dir = paths.curves / "controls" / f"151_{condition}_delay"
                        delayed_matrix = delayed_dir / "distance.npy"
                        delayed_worker = delayed_dir / "worker.json"
                        delayed_dir.mkdir(parents=True, exist_ok=True)
                        delayed_result = _score_crop(delayed_media, cell_dir=delayed_dir, reference=f"lrs3_tts_gain_151_{condition}_delay", matrix_path=delayed_matrix, result_path=delayed_worker, log_name="syncnet.log")
                        delayed_matrix_value = np.asarray(np.load(delayed_matrix, allow_pickle=False), dtype=np.float64)
                        delay_metrics = _control_metrics(original_matrix, delayed_matrix_value)
                        if delayed_info.get("video_unchanged") is not True:
                            delay_metrics["status"] = "CONTROL_FAILED"
                        delay_result = {"control": "delay_200ms", "sample_id": 151, "condition": condition, "media": delayed_info, "crop": original["crop"], "crop_sha256": original["crop_sha256"], "crop_selection": original["crop_selection"], "selection_sha256": original["selection_sha256"], "official_config": original["official_config"], "worker_result": str(delayed_worker), "worker_result_sha256": sha256_file(delayed_worker), "matrix": str(delayed_matrix), "matrix_sha256": sha256_file(delayed_matrix), "matrix_shape": [int(item) for item in delayed_matrix_value.shape], "worker": delayed_result, **delay_metrics}
                    except Exception as exc:
                        if str(exc).startswith("RESOURCE_WAIT"):
                            raise
                        delay_result = _failed_control_row(original, "delay_200ms", condition, exc)
                    controls.append(delay_result)
                    control_keys.add(("delay_200ms", condition))
                    _write_control_checkpoint(paths, controls, status="partial")
            if controls_ready:
                _write_control_checkpoint(paths, controls, status="complete")
            else:
                _write_control_checkpoint(paths, controls, status="partial")
    except ProtocolError as exc:
        if str(exc).startswith("RESOURCE_WAIT"):
            _write_control_checkpoint(paths, controls, status="RESOURCE_WAIT")
            payload = _write_curve_checkpoint(paths, rows, controls, status="RESOURCE_WAIT", failures=[*main_failures, str(exc)], resource=resource)
            return payload
        raise
    if main_failures:
        payload = _write_curve_checkpoint(paths, rows, controls, status="PARTIAL", failures=main_failures, resource=resource)
    else:
        payload = _write_curve_checkpoint(paths, rows, controls, status="complete", failures=[], resource=resource)
    return payload


def analyze_curves(paths: config.RunPaths) -> dict[str, Any]:
    historical_path = paths.decomposition / "summary.json"
    historical = read_self_hashed_json(historical_path) if historical_path.is_file() else None
    manifest_path = paths.curves / "manifest.json"
    if not manifest_path.is_file():
        run_curves(paths)
    manifest = read_self_hashed_json(manifest_path)
    if manifest.get("status") != "complete":
        empty_fields = ("model", "sample_id", "source_group", "condition", "support")
        csv_write(paths.analysis / "endpoints.csv", empty_fields, [])
        csv_write(paths.analysis / "paired.csv", empty_fields, [])
        summary = {"status": str(manifest.get("status", "INCOMPLETE")), "protocol_id": config.PROTOCOL_ID, "reason": manifest.get("failures", []), "main_cells": manifest.get("completed_main_cells", 0), "controls": manifest.get("completed_control_cells", 0)}
        write_self_hashed_json(paths.analysis / "summary.json", summary)
        return summary
    source_groups = {int(row["sample_id"]): str(row["source_group"]) for row in read_self_hashed_json(paths.audit / "cohort.json")["records"]}
    endpoints: list[dict[str, Any]] = []
    pair_supports: list[dict[str, Any]] = []
    for sample_id in config.CURVE_SAMPLE_IDS:
        rows = {str(row["condition"]): row for row in manifest["rows"] if int(row["sample_id"]) == sample_id}
        n_path = _run_artifact(paths, rows["natural_raw"]["matrix"], f"natural/{sample_id} matrix")
        t_path = _run_artifact(paths, rows["tts_raw"]["matrix"], f"tts/{sample_id} matrix")
        n = np.asarray(np.load(n_path, allow_pickle=False), dtype=np.float64)
        t = np.asarray(np.load(t_path, allow_pickle=False), dtype=np.float64)
        current_endpoints, support_info = pair_curve_metrics(n, t, sample_id=sample_id, source_group=source_groups[sample_id])
        endpoints.extend(current_endpoints)
        pair_supports.append(support_info)
    curve = curve_analysis(endpoints, source_groups)
    curve["_endpoints"] = endpoints
    figure_paths = plot_outputs(paths.analysis, historical, curve)
    curve.pop("_endpoints", None)
    curve["figures"] = figure_paths
    curve["protocol_id"] = config.PROTOCOL_ID
    curve["manifest_sha256"] = sha256_file(manifest_path)
    write_curve_tables(paths.analysis, endpoints, curve["paired"])
    write_self_hashed_json(paths.analysis / "summary.json", curve)
    print("ANALYZE curves status=complete", flush=True)
    return curve


def perception_pack(paths: config.RunPaths) -> dict[str, Any]:
    paths.perception.mkdir(parents=True, exist_ok=True)
    bindings = read_json(config.INPUT_BINDINGS)
    media = {(int(row["sample_id"]), str(row["condition"])): row for row in _curve_media_bindings(bindings)}
    media_failures: list[str] = []
    for (sample_id, condition), bound in media.items():
        source = _path(str(bound["path"]))
        if not source.is_file() or sha256_file(source) != str(bound["sha256"]):
            media_failures.append(f"{sample_id}/{condition}: media is missing or its hash changed")
    if media_failures:
        blocked = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "BLOCKED_INPUT_BINDING",
            "pair_count": len(config.CURVE_SAMPLE_IDS),
            "clip_count": 0,
            "failures": media_failures,
        }
        write_self_hashed_json(paths.perception / "status.json", blocked)
        return blocked
    (paths.perception / "status.json").unlink(missing_ok=True)
    required = [
        paths.perception / "clips.json",
        paths.perception / "assignments.csv",
        paths.perception / "ratings_template.csv",
        paths.perception / "private_key.json",
    ]
    if all(path.is_file() for path in required):
        return {"status": "READY", "pair_count": 12, "clip_count": 24, "reused": True}
    cohort = read_self_hashed_json(paths.audit / "cohort.json")["records"] if (paths.audit / "cohort.json").is_file() else (audit_inputs(paths) and read_self_hashed_json(paths.audit / "cohort.json")["records"])
    group_for = {int(row["sample_id"]): str(row["source_group"]) for row in cohort}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    clips: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for sample_id in config.CURVE_SAMPLE_IDS:
        order = list(rng.permutation(list(config.CONDITIONS)))
        a_condition, b_condition = order
        for label, condition in (("A", a_condition), ("B", b_condition)):
            source = _path(str(media[(sample_id, condition)]["path"]))
            public_path = paths.perception / "media" / f"{sample_id}_{label}.mp4"
            public_path.parent.mkdir(parents=True, exist_ok=True)
            if public_path.exists() or public_path.is_symlink():
                if public_path.resolve() != source:
                    raise ProtocolError(f"perception clip reference points to the wrong media: {public_path}")
            else:
                public_path.symlink_to(source)
            clips.append({
                "clip_id": f"{sample_id}_{label}",
                "sample_id": sample_id,
                "label": label,
                "path": str(public_path),
                "sha256": str(media[(sample_id, condition)]["sha256"]),
                "source_group": group_for[sample_id],
                "full_clip_reference": True,
            })
        assignments.append({
            "sample_id": sample_id,
            "source_group": group_for[sample_id],
            "clip_a": str(paths.perception / "media" / f"{sample_id}_A.mp4"),
            "clip_b": str(paths.perception / "media" / f"{sample_id}_B.mp4"),
            "a_id": "A",
            "b_id": "B",
        })
        key_rows.append({
            "sample_id": sample_id,
            "source_group": group_for[sample_id],
            "a_condition": a_condition,
            "b_condition": b_condition,
        })
    clips_payload = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "READY", "clip_count": len(clips), "pair_count": len(config.CURVE_SAMPLE_IDS), "clips": clips, "note": "public A/B paths use opaque labels and symlinks to the bound native media; the original audio may still reveal the condition"}
    write_self_hashed_json(paths.perception / "clips.json", clips_payload)
    csv_write(paths.perception / "assignments.csv", ("sample_id", "source_group", "clip_a", "clip_b", "a_id", "b_id"), assignments)
    csv_write(paths.perception / "ratings_template.csv", ("rater_id", "sample_id", "sync_preference", "sync_rating_a_1_5", "sync_rating_b_1_5", "naturalness_rating_a_1_5", "naturalness_rating_b_1_5", "notes"), [])
    private = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "seed": config.BOOTSTRAP_SEED, "rows": key_rows, "warning": "keep this file separate from raters; it contains the true A/B condition mapping"}
    write_self_hashed_json(paths.perception / "private_key.json", private)
    os.chmod(paths.perception / "private_key.json", 0o600)
    return {"status": "READY", "pair_count": 12, "clip_count": 24, "reused": False}


def _rating_number(value: str, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or result < 1.0 or result > 5.0 or result != round(result):
        raise ProtocolError(f"{label} must be an integer from 1 to 5")
    return result


def _perception_group_bootstrap(values: Mapping[str, float], indices: np.ndarray, *, metric: str) -> dict[str, Any]:
    labels = sorted(values)
    array = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    sampled = array[np.asarray(indices, dtype=np.int64)].mean(axis=1, dtype=np.float64)
    return {
        "metric": metric,
        "mean": float(array.mean(dtype=np.float64)),
        "ci95": [float(np.quantile(sampled, 0.025, method="linear")), float(np.quantile(sampled, 0.975, method="linear"))],
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, array, strict=True)},
        "group_count": len(labels),
        "draws": int(indices.shape[0]),
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy.default_rng(PCG64)",
        "quantile_method": "linear",
    }


def analyze_perception(paths: config.RunPaths) -> dict[str, Any]:
    ratings_path = paths.perception / "ratings_template.csv"
    assignments_path = paths.perception / "assignments.csv"
    key_path = paths.perception / "private_key.json"
    if not all(path.is_file() for path in (ratings_path, assignments_path, key_path)):
        raise ProtocolError("perception package is incomplete")
    assignments = {int(row["sample_id"]): row for row in csv_read(assignments_path)}
    private = read_self_hashed_json(key_path)
    key_rows = {int(row["sample_id"]): row for row in private.get("rows", [])}
    expected_ids = set(config.CURVE_SAMPLE_IDS)
    if set(assignments) != expected_ids or set(key_rows) != expected_ids:
        raise ProtocolError("perception assignment/key IDs are incomplete")
    rows = csv_read(ratings_path)
    if not rows:
        result = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "status": "NOT_ASSESSED",
            "pair_count": len(expected_ids),
            "rating_row_count": 0,
            "rater_count": 0,
            "complete_rater_count": 0,
            "complete_raters": [],
            "ratings_by_rater": {},
            "unjudgeable_count": 0,
            "ratings_sha256": sha256_file(ratings_path),
            "assignments_sha256": sha256_file(assignments_path),
            "private_key_sha256": sha256_file(key_path),
            "reason": "ratings_template.csv is empty; no human score is inferred",
        }
        write_self_hashed_json(paths.perception / "analysis.json", result)
        return result
    allowed_preferences = {"A", "B", "tie", "unjudgeable"}
    by_rater: dict[str, dict[int, Mapping[str, str]]] = {}
    for row in rows:
        rater = str(row.get("rater_id", "")).strip()
        if not rater:
            raise ProtocolError("perception rating has an empty rater_id")
        try:
            sample_id = int(row.get("sample_id", ""))
        except ValueError as exc:
            raise ProtocolError("perception rating has an invalid sample_id") from exc
        if sample_id not in expected_ids:
            raise ProtocolError(f"perception rating has an unexpected sample_id: {sample_id}")
        if sample_id in by_rater.setdefault(rater, {}):
            raise ProtocolError(f"duplicate perception rating: {rater}/{sample_id}")
        preference = str(row.get("sync_preference", "")).strip().lower()
        if preference not in {item.lower() for item in allowed_preferences}:
            raise ProtocolError(f"invalid sync_preference: {preference}")
        normalized = dict(row)
        normalized["sync_preference"] = preference
        for field in ("sync_rating_a_1_5", "sync_rating_b_1_5", "naturalness_rating_a_1_5", "naturalness_rating_b_1_5"):
            normalized[field] = str(_rating_number(row.get(field, ""), label=f"{rater}/{sample_id}/{field}"))
        by_rater[rater][sample_id] = normalized
    complete_raters = sorted(rater for rater, rating_rows in by_rater.items() if set(rating_rows) == expected_ids)
    status = "COMPLETE" if len(complete_raters) >= 3 else "PARTIAL"
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": status,
        "pair_count": len(expected_ids),
        "rating_row_count": len(rows),
        "rater_count": len(by_rater),
        "complete_rater_count": len(complete_raters),
        "complete_raters": complete_raters,
        "ratings_by_rater": {rater: len(rating_rows) for rater, rating_rows in sorted(by_rater.items())},
        "ratings_sha256": sha256_file(ratings_path),
        "assignments_sha256": sha256_file(assignments_path),
        "private_key_sha256": sha256_file(key_path),
        "unjudgeable_count": sum(row["sync_preference"] == "unjudgeable" for row in (item for rater in complete_raters for item in by_rater[rater].values())),
        "reason": "fewer than three common raters completed all 12 pairs" if status == "PARTIAL" else "at least three common raters completed all 12 pairs",
    }
    if status == "COMPLETE":
        pair_rows: list[dict[str, Any]] = []
        for sample_id in sorted(expected_ids):
            key = key_rows[sample_id]
            deltas: list[float] = []
            naturalness_deltas: list[float] = []
            preferences: list[bool] = []
            ties = 0
            unjudgeable = 0
            for rater in complete_raters:
                row = by_rater[rater][sample_id]
                a_sync = float(row["sync_rating_a_1_5"])
                b_sync = float(row["sync_rating_b_1_5"])
                a_naturalness = float(row["naturalness_rating_a_1_5"])
                b_naturalness = float(row["naturalness_rating_b_1_5"])
                sync_by_condition = {key["a_condition"]: a_sync, key["b_condition"]: b_sync}
                naturalness_by_condition = {key["a_condition"]: a_naturalness, key["b_condition"]: b_naturalness}
                deltas.append(sync_by_condition["tts_raw"] - sync_by_condition["natural_raw"])
                naturalness_deltas.append(naturalness_by_condition["tts_raw"] - naturalness_by_condition["natural_raw"])
                preference = row["sync_preference"]
                if preference == "tie":
                    ties += 1
                elif preference == "unjudgeable":
                    unjudgeable += 1
                else:
                    preferred_condition = key[f"{preference.lower()}_condition"]
                    preferences.append(preferred_condition == "tts_raw")
            pair_rows.append({
                "sample_id": sample_id,
                "source_group": key["source_group"],
                "sync_rating_delta_tts_minus_natural": float(np.mean(deltas)),
                "naturalness_delta_tts_minus_natural": float(np.mean(naturalness_deltas)),
                "tts_preference_rate_among_judged": None if not preferences else float(np.mean(preferences)),
                "tie_rate": float(ties / len(complete_raters)),
                "unjudgeable_rate": float(unjudgeable / len(complete_raters)),
            })
        indices = bootstrap_indices(len(pair_rows))
        result["pair_summaries"] = pair_rows
        result["group_bootstrap"] = {
            metric: _perception_group_bootstrap(
                {str(row["source_group"]): float(row[metric]) for row in pair_rows if row[metric] is not None},
                indices,
                metric=metric,
            )
            for metric in ("sync_rating_delta_tts_minus_natural", "naturalness_delta_tts_minus_natural")
        }
        preference_values = {str(row["source_group"]): float(row["tts_preference_rate_among_judged"]) for row in pair_rows if row["tts_preference_rate_among_judged"] is not None}
        if len(preference_values) == len(pair_rows):
            result["group_bootstrap"]["tts_preference_rate_among_judged"] = _perception_group_bootstrap(preference_values, indices, metric="tts_preference_rate_among_judged")
    write_self_hashed_json(paths.perception / "analysis.json", result)
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def report(paths: config.RunPaths) -> dict[str, Any]:
    if not (paths.decomposition / "summary.json").is_file():
        decompose(paths)
    manifest_path = paths.curves / "manifest.json"
    if manifest_path.is_file():
        manifest = read_self_hashed_json(manifest_path)
        analysis_path = paths.analysis / "summary.json"
        analysis_is_current = False
        if analysis_path.is_file():
            analysis = read_self_hashed_json(analysis_path)
            if manifest.get("status") == "complete":
                analysis_is_current = analysis.get("status") == "complete" and analysis.get("manifest_sha256") == sha256_file(manifest_path)
            else:
                analysis_is_current = (
                    analysis.get("status") == manifest.get("status")
                    and analysis.get("main_cells") == manifest.get("completed_main_cells", 0)
                    and analysis.get("controls") == manifest.get("completed_control_cells", 0)
                    and analysis.get("reason") == manifest.get("failures", [])
                )
        if not analysis_is_current:
            analyze_curves(paths)
    if not (paths.perception / "clips.json").is_file():
        perception_pack(paths)
    historical = read_self_hashed_json(paths.decomposition / "summary.json")
    curve = read_self_hashed_json(paths.analysis / "summary.json") if (paths.analysis / "summary.json").is_file() else {"status": "NOT_RUN"}
    controls = read_self_hashed_json(paths.curves / "controls.json") if (paths.curves / "controls.json").is_file() else {"status": "NOT_RUN"}
    curve_manifest = read_self_hashed_json(paths.curves / "manifest.json") if (paths.curves / "manifest.json").is_file() else {}
    perception_package = perception_pack(paths)
    perception = perception_package if perception_package.get("status") == "BLOCKED_INPUT_BINDING" else analyze_perception(paths)
    auto_complete = historical.get("status") == "complete" and curve.get("status") == "complete"
    run_status = "complete" if auto_complete else ("RESOURCE_WAIT" if curve.get("status") == "RESOURCE_WAIT" else "partial")
    record_means = historical.get("record_means", {})
    control_rows = controls.get("rows", []) if isinstance(controls.get("rows", []), list) else []
    control_failed_count = sum(row.get("status") == "CONTROL_FAILED" for row in control_rows if isinstance(row, Mapping))
    main_cells = config.MAIN_CURVE_CELLS if curve.get("status") == "complete" else curve.get("main_cells", curve.get("completed_main_cells", 0))
    attempted_new_cells = int(curve_manifest.get("new_syncnet_cells_attempted", len(control_rows) + int(main_cells)))
    lines = [
        "# LRS3 English TTS SyncNet gain mechanism experiment",
        "",
        f"Protocol: `{config.PROTOCOL_ID}`  ",
        f"Automatic status: **{run_status}**; human perception: **{perception.get('status', 'NOT_ASSESSED')}**.",
        "",
        "## Why this experiment exists",
        "",
        "The LRS3 evidence contains a native TTS advantage. It is kept separate from strict replacement, NAT_ONLY conditioning, MFA-linear, and bridge endpoints. This run does not claim that English as a language is universally effective.",
        "",
        "| historical model | records | record-weighted ΔC | D improvement D_N−D_T | background change B_T−B_N | C-positive |",
    ]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in config.MODELS:
        item = record_means[model]
        lines.append(f"| {model} | {item['n']} | {_fmt(item['gain_c'])} | {_fmt(item['benefit_d'])} | {_fmt(item['gain_background'])} | {item['c_positive']}/{item['n']} |")
    lines.extend(["", "The primary inference table uses 45 source groups with the same bootstrap draws for every metric.", "", "| model | component | group-weighted mean | 95% descriptive interval | 98.75% Bonferroni interval | 98.75% lower bound > 0 |", "|---|---|---:|---:|---:|:---:|"])
    for model in config.MODELS:
        for metric, label in (("gain_c", "ΔC"), ("gain_match", "best-match"), ("gain_background", "background")):
            item = historical["group_summaries"][f"{model}.{metric}"]
            interval = item["ci98_75"]
            lines.append(f"| {model} | {label} | {_fmt(item['mean'])} | [{_fmt(item['ci95'][0])}, {_fmt(item['ci95'][1])}] | [{_fmt(interval[0])}, {_fmt(interval[1])}] | {'yes' if float(interval[0]) > 0 else 'no'} |")
    lines.extend([
        "",
        "Here `B_hat = C_saved + D_saved`; because the historical files store rounded scalar C/D, this recovers the scalar background only, not an original full curve.",
        "",
        "## Calculation",
        "",
        "For every natural/TTS pair the run computes `gain_C = C_T − C_N`, `gain_match = D_N − D_T`, and `gain_background = B_T − B_N`. The identity `gain_C = gain_match + gain_background` is checked before group aggregation. Records sharing one source directory are first averaged within source group; the primary interval then gives each of the 45 source groups equal weight. The displayed historical check values are record-weighted inputs; they are not a second independent replication.",
        "",
        f"The curve stage is restricted to LeapTalk IDs 151–162, 24 already generated native videos, and at most {config.MAX_NEW_CELLS} new SyncNet cells. FULL, INTERIOR, and EQUAL_COUNT are calculated from each saved `[T,31]` matrix after time-row averaging. Official offset is `15 − column_index`; C_5, D0, S, and trough width are exploratory diagnostics.",
        "",
        "## Curve and control status",
        "",
        f"Curve analysis: **{curve.get('status', 'NOT_RUN')}**. Main cells: {main_cells}/24; new SyncNet cells attempted: {attempted_new_cells}/{config.MAX_NEW_CELLS}. New TTS/TFG/training/cloud calls: 0.",
        f"Controls: **{controls.get('status', 'NOT_RUN')}** ({len(control_rows)}/4 attempted; CONTROL_FAILED={control_failed_count}). A 200 ms audio delay is interpreted by its searched offset separately from zero-lag matching; a high searched C would not erase the delay.",
        "",
        "## Perception boundary",
        "",
        f"The run creates a 12-pair A/B package. Human perception status is **{perception.get('status', 'NOT_ASSESSED')}**; without at least three common raters completing all 12 pairs, it remains descriptive only. No human preference is inferred from SyncNet.",
        "",
        "## Resource and interpretation boundary",
        "",
        "GPU work is serialized under an exclusive lease. The official pipeline is allowed to run only after a disk preflight leaves the required peak temporary space plus 1 GiB. If that gate is closed, the CPU historical result remains valid and the curve result stays RESOURCE_WAIT; old runs are never cleaned to force execution.",
        "",
        "The decomposition is algebraic. A lower best distance is evidence of a lower SyncNet embedding distance, and a higher background term is a change in the distance distribution. Neither term alone identifies a perceptual or acoustic causal mechanism. The next experiment should target one concrete timing or evaluator-support phenomenon located by these curves.",
    ])
    if curve.get("status") == "complete":
        lines.insert(-4, "")
        lines.insert(-4, "The three pre-registered curve contrasts use 12 source groups and 95%/98.333333% intervals; all other curve quantities remain exploratory.")
        for name in ("interior_delta_c", "interior_gain_match", "interior_minus_full_delta_c"):
            item = curve["core_summaries"][name]
            lines.insert(-4, f"- `{name}` mean={_fmt(item['mean'])}, 95%=[{_fmt(item['ci95'][0])}, {_fmt(item['ci95'][1])}], 98.333333%=[{_fmt(item['ci98_333'][0])}, {_fmt(item['ci98_333'][1])}]")
    if perception.get("status") == "COMPLETE":
        sync = perception["group_bootstrap"]["sync_rating_delta_tts_minus_natural"]
        naturalness = perception["group_bootstrap"]["naturalness_delta_tts_minus_natural"]
        lines.extend([
            "",
            f"Human rating group bootstrap (12 source groups): sync TTS−natural mean={_fmt(sync['mean'])}, 95%=[{_fmt(sync['ci95'][0])}, {_fmt(sync['ci95'][1])}]; naturalness TTS−natural mean={_fmt(naturalness['mean'])}, 95%=[{_fmt(naturalness['ci95'][0])}, {_fmt(naturalness['ci95'][1])}].",
        ])
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": run_status,
        "automatic_complete": bool(auto_complete),
        "historical_decomposition": {"status": historical.get("status"), "summary": str((paths.decomposition / "summary.json").resolve()), "summary_sha256": sha256_file(paths.decomposition / "summary.json")},
        "curve_analysis": {
            "status": curve.get("status"),
            "summary": str((paths.analysis / "summary.json").resolve()) if (paths.analysis / "summary.json").is_file() else None,
            "summary_sha256": sha256_file(paths.analysis / "summary.json") if (paths.analysis / "summary.json").is_file() else None,
        },
        "controls": {
            "status": controls.get("status"),
            "summary": str((paths.curves / "controls.json").resolve()) if (paths.curves / "controls.json").is_file() else None,
            "summary_sha256": sha256_file(paths.curves / "controls.json") if (paths.curves / "controls.json").is_file() else None,
        },
        "perception": perception,
        "run_identity": historical.get("run_identity"),
        "counts": {"historical_cells": 200, "historical_pairs": 100, "historical_source_groups": 45, "curve_main_cells": 24, "curve_control_cells": 4, "new_syncnet_cells_max": 28, "perception_pairs": 12},
        "budget": {"max_new_syncnet_cells": config.MAX_NEW_CELLS, "attempted_new_syncnet_cells": attempted_new_cells},
        "report_sha256": sha256_file(paths.report),
        "input_bindings_sha256": sha256_file(config.INPUT_BINDINGS),
        "zero_new_tts": True,
        "zero_new_tfg": True,
        "zero_training": True,
        "zero_cloud_calls": True,
    }
    write_self_hashed_json(paths.final, final)
    print(f"REPORT status={run_status} automatic_complete={auto_complete}", flush=True)
    return final


def _check_run_identity(paths: config.RunPaths) -> None:
    inputs_path = paths.audit / "inputs.json"
    if not inputs_path.is_file():
        return
    inputs = read_self_hashed_json(inputs_path)
    recorded = inputs.get("run_identity")
    if not isinstance(recorded, Mapping) or recorded.get("identity_sha256") != current_run_identity()["identity_sha256"]:
        raise ProtocolError("run identity changed or is missing; choose a new run id")


def run(run_id: str, stage: str) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.final.is_file():
        final = read_self_hashed_json(paths.final)
        if str(final.get("status")) != "RESOURCE_WAIT" and stage not in {"perception-pack", "perception-analyze", "report"}:
            raise ProtocolError("terminal run cannot be overwritten; choose a new run id")
    _check_run_identity(paths)
    paths.root.mkdir(parents=True, exist_ok=True)
    if stage == "audit":
        return audit_inputs(paths)
    if stage == "decompose":
        return decompose(paths)
    if stage == "curves":
        if not (paths.audit / "inputs.json").is_file():
            audit_inputs(paths)
        return run_curves(paths)
    if stage == "analyze":
        if not (paths.audit / "inputs.json").is_file():
            audit_inputs(paths)
        if not (paths.decomposition / "summary.json").is_file():
            decompose(paths)
        return analyze_curves(paths)
    if stage == "perception-pack":
        if not (paths.audit / "inputs.json").is_file():
            audit_inputs(paths)
        return perception_pack(paths)
    if stage == "perception-analyze":
        if not (paths.audit / "inputs.json").is_file():
            audit_inputs(paths)
        package = perception_pack(paths)
        return package if package.get("status") == "BLOCKED_INPUT_BINDING" else analyze_perception(paths)
    if stage == "report":
        return report(paths)
    if stage == "all":
        audit_inputs(paths)
        decompose(paths)
        run_curves(paths)
        analyze_curves(paths)
        perception_pack(paths)
        return report(paths)
    raise ValueError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the auditable LRS3 English TTS SyncNet gain mechanism experiment")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("audit", "decompose", "curves", "analyze", "perception-pack", "perception-analyze", "report", "all"), default="all")
    args = parser.parse_args(argv)
    result = run(args.run_id, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
