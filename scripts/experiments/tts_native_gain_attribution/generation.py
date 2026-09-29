"""LeapTalk generation and the complete crossed four-cell B experiment."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DependencyBlockedError,
    ImplementationIncompleteError,
    InputInvalidError,
    ProtocolError,
    ResourceWaitError,
    canonical_sha256,
    file_sha256,
    gpu_lease,
    read_self_hashed_json,
    run_monitored,
    write_self_hashed_json,
)
from .leaptalk_adapter import (
    command_argv,
    discover,
    request_payload,
    validate_response,
    write_request,
)


def discover_leaptalk(model_config: Path | None = None) -> dict[str, Any]:
    """Return an honest provenance record without falling back to Wav2Lip."""

    try:
        return discover(model_config)
    except DependencyBlockedError as exc:
        roots = [{"root": str(candidate), "exists": candidate.is_dir()} for candidate in config.leaptalk_candidates()]
        return {
            "family": "LeapTalk",
            "status": "DEPENDENCY_BLOCKED",
            "reason": str(exc),
            "roots": roots,
            "repo_commit": "UNKNOWN",
            "weights": {},
            "training_distribution": "TRAINING_DISTRIBUTION_UNKNOWN",
            "no_wav2lip_substitution": True,
        }


def _cell_id(row: Mapping[str, Any]) -> str:
    role = str(row.get("role", "main"))
    return f"BGEN:{int(row['id'])}:{row['source']}:{row['driver_condition']}:seed{int(row['seed'])}:{role}"


def _role(row: Mapping[str, Any]) -> str:
    return "repeat1" if bool(row.get("repeat")) or str(row.get("role")) == "repeat1" else "main"


def _output_dir(paths: config.RunPaths, row: Mapping[str, Any]) -> Path:
    return paths.generation / "videos" / str(row["id"]) / str(row["source"]) / str(row["driver_condition"]) / f"seed{int(row['seed'])}" / _role(row)


def _expected_generation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for driver in config.B_DRIVERS:
                for seed in config.SEEDS:
                    rows.append({"stage": "B_GENERATION", "id": sample_id, "source": source, "driver_condition": driver, "seed": seed, "repeat": False, "role": "main", "cell_id": f"BGEN:{sample_id}:{source}:{driver}:seed{seed}:main"})
    for sample_id in config.CONTROL_IDS:
        for source in ("N", "T"):
            rows.append({"stage": "B_GENERATION_REPEAT", "id": sample_id, "source": source, "driver_condition": "A0", "seed": 42, "repeat": True, "role": "repeat1", "cell_id": f"BGEN:{sample_id}:{source}:A0:seed42:repeat1"})
    return rows


def _audio_lookup(audio_manifest: Mapping[str, Any]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    return {(int(row["sample_id"]), str(row["source"]), str(row["condition"])): row for row in audio_manifest.get("records", [])}


def _asset_lookup(assets: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(row["sample_id"]): row for row in assets.get("records", [])}


def _base_row(
    paths: config.RunPaths,
    row: Mapping[str, Any],
    audio: Mapping[str, Any],
    assets: Mapping[str, Any],
    *,
    status: str,
    reason: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    item = dict(row)
    item.update({
        "source_group": str(_asset_lookup(assets).get(int(row["id"]), {}).get("source_group", "UNKNOWN")),
        "video_type": "V0" if str(row["driver_condition"]) == "A0" else f"V_{row['driver_condition']}",
        "eval_condition": str(row["driver_condition"]),
        "path": str(_output_dir(paths, row) / "video.mp4"),
        "video_hash": None,
        "file_sha256": None,
        "pixel_sha256": None,
        "pts_sha256": None,
        "pcm_path": str(audio.get("path", "")),
        "pcm_hash": audio.get("pcm_sha256"),
        "portrait_path": None,
        "portrait_hash": None,
        "roi_hash": None,
        "support_hash": None,
        "model_hash": None,
        "code_hash": None,
        "matrix_hash": None,
        "reused_from": None,
        "response": None,
        "frontend": None,
        "status": status,
    })
    if reason is not None:
        item["reason"] = reason
    if error_type is not None:
        item["error_type"] = error_type
    return item


def _blocked_generation_rows(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], reason: str) -> list[dict[str, Any]]:
    audio = _audio_lookup(audio_manifest)
    rows: list[dict[str, Any]] = []
    for raw in _expected_generation_rows():
        condition = str(raw["driver_condition"])
        record = audio.get((int(raw["id"]), str(raw["source"]), condition), {})
        rows.append(_base_row(paths, raw, record, assets, status="DEPENDENCY_BLOCKED", reason=reason, error_type="DEPENDENCY_BLOCKED"))
    return rows


def _generation_code_hash() -> str:
    paths = (Path(__file__), Path(__file__).with_name("leaptalk_adapter.py"))
    return canonical_sha256({path.name: file_sha256(path) for path in paths if path.is_file()})


def _provenance_model_hash(provenance: Mapping[str, Any]) -> str:
    """Hash the discovered model binding, excluding the run JSON envelope."""

    body = dict(provenance)
    body.pop("artifact_sha256", None)
    body.pop("protocol_id", None)
    body.pop("schema_version", None)
    return canonical_sha256(body)


def _request_for_row(
    paths: config.RunPaths,
    row: Mapping[str, Any],
    audio_record: Mapping[str, Any],
    portrait: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = _output_dir(paths, row)
    config_hash = str(provenance.get("config_sha256") or canonical_sha256(provenance))
    return request_payload(
        cell_id=str(row.get("cell_id") or _cell_id({**row, "role": _role(row)})),
        role=_role(row),
        sample_id=int(row["id"]),
        source=str(row["source"]),
        driver=str(row["driver_condition"]),
        seed=int(row["seed"]),
        portrait=portrait,
        audio=audio_record,
        config_hash=config_hash,
        output_dir=output_dir,
    )


def _quarantine_partial(output_dir: Path) -> None:
    """Retain orphaned media while making the identity directory retryable."""

    candidates = [output_dir / "video.mp4", output_dir / "response.json"]
    present = [path for path in candidates if path.is_file()]
    if not present:
        return
    target = output_dir / "attempts"
    target.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (target / f"attempt{attempt}").exists():
        attempt += 1
    attempt_dir = target / f"attempt{attempt}"
    attempt_dir.mkdir()
    for path in present:
        shutil.move(str(path), str(attempt_dir / path.name))


def _validate_video_rate(signature: Mapping[str, Any]) -> None:
    value = signature.get("pts", {}).get("fps")
    if not isinstance(value, str) or "/" not in value:
        raise ProtocolError("generated video has no auditable frame rate")
    numerator, denominator = value.split("/", 1)
    try:
        fps = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProtocolError("generated video frame rate is malformed") from exc
    if abs(fps - config.FPS) > 0.01:
        raise ProtocolError(f"generated video frame rate differs from {config.FPS}: {fps}")


def _run_one_generation(
    paths: config.RunPaths,
    row: Mapping[str, Any],
    audio_record: Mapping[str, Any],
    portrait: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    command_template = provenance.get("command")
    if not isinstance(command_template, (str, list)):
        raise DependencyBlockedError("LeapTalk provenance has no argv command")
    output_dir = _output_dir(paths, row)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "video.mp4"
    response_path = output_dir / "response.json"
    request_path = output_dir / "request.json"
    config_hash = str(provenance.get("config_sha256") or canonical_sha256(provenance))
    request = _request_for_row(paths, row, audio_record, portrait, provenance)
    if request_path.is_file():
        existing = read_self_hashed_json(request_path)
        existing_body = dict(existing)
        existing_body.pop("artifact_sha256", None)
        if existing_body != request:
            raise InputInvalidError(f"immutable LeapTalk request changed: {request_path}")
    else:
        write_request(request_path, request)
    _quarantine_partial(output_dir)
    values = {
        "root": str(provenance.get("root", "")),
        "request": str(request_path),
        "response": str(response_path),
        "output_dir": str(output_dir),
        "output": str(output),
        "audio": str(audio_record["path"]),
        "portrait": str(portrait["path"]),
        "id": int(row["id"]),
        "source": str(row["source"]),
        "driver": str(row["driver_condition"]),
        "seed": int(row["seed"]),
        "role": _role(row),
    }
    command = command_argv(command_template, values)
    log = output_dir / "adapter.log"
    with gpu_lease(
        gpu_peak_bytes=config.GPU_PEAK_BUDGET_BYTES,
        disk_temp_bytes=config.CELL_TEMP_BUDGET_BYTES,
        disk_persistent_bytes=16 << 20,
    ):
        run_monitored(command, cwd=Path(str(provenance["root"])), log_path=log, interval_seconds=30.0)
    if not output.is_file():
        raise ProtocolError(f"LeapTalk adapter completed without output: {output}")
    if not response_path.is_file():
        raise ProtocolError(f"LeapTalk response is missing: {response_path}")
    response = validate_response(response_path, request, provenance)
    from .syncnet import video_signature

    signature = video_signature(output)
    _validate_video_rate(signature)
    if int(signature["frame_count"]) < 5:
        raise ProtocolError(f"generated video has too few frames: {output}")
    return {
        "stage": str(row["stage"]),
        "id": int(row["id"]),
        "sample_id": int(row["id"]),
        "source_group": str(row["source_group"]),
        "source": str(row["source"]),
        "video_type": "V0" if str(row["driver_condition"]) == "A0" else f"V_{row['driver_condition']}",
        "driver_condition": str(row["driver_condition"]),
        "eval_condition": str(row["driver_condition"]),
        "seed": int(row["seed"]),
        "role": _role(row),
        "cell_id": str(request["cell_id"]),
        "path": str(output),
        "file_sha256": file_sha256(output),
        "video_hash": str(signature["video_hash"]),
        "pixel_sha256": str(signature["pixel_sha256"]),
        "pts_sha256": str(signature["pts_sha256"]),
        "frame_count": int(signature["frame_count"]),
        "pts": signature["pts"],
        "pcm_path": str(audio_record["path"]),
        "pcm_hash": str(audio_record["pcm_sha256"]),
        "portrait_path": str(portrait["path"]),
        "portrait_hash": str(portrait["file_sha256"]),
        "roi_hash": None,
        "support_hash": None,
        "model_hash": _provenance_model_hash(provenance),
        "code_hash": _generation_code_hash(),
        "matrix_hash": None,
        "reused_from": None,
        "request": {"path": str(request_path), "sha256": file_sha256(request_path)},
        "response": {"path": str(response_path), "sha256": file_sha256(response_path)},
        "frontend": response["frontend"],
        "rng_reset": response["rng_reset"],
        "chunk_to_frame_timing": response["chunk_to_frame_timing"],
        "generation_parameters": {"command": command, "provenance_status": provenance.get("status"), "config_sha256": config_hash},
        "status": "COMPLETE",
    }


def _row_from_failure(
    paths: config.RunPaths,
    raw: Mapping[str, Any],
    audio: Mapping[str, Any],
    assets: Mapping[str, Any],
    exc: BaseException,
    status: str = "INCOMPLETE",
) -> dict[str, Any]:
    if isinstance(exc, ResourceWaitError):
        error_type = "RESOURCE_WAIT"
    elif isinstance(exc, DependencyBlockedError):
        error_type = "DEPENDENCY_BLOCKED"
    elif isinstance(exc, InputInvalidError):
        error_type = "INPUT_INVALID"
    elif isinstance(exc, ImplementationIncompleteError):
        error_type = "IMPLEMENTATION_INCOMPLETE"
    else:
        error_type = "GENERATION_FAILED"
    return _base_row(paths, raw, audio, assets, status=status, reason=str(exc), error_type=error_type)


def _valid_generation_cache(
    paths: config.RunPaths,
    row: Mapping[str, Any],
    audio_record: Mapping[str, Any],
    portrait: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> bool:
    if row.get("status") != "COMPLETE" or row.get("model_hash") != _provenance_model_hash(provenance):
        return False
    path = Path(str(row.get("path", "")))
    response = row.get("response")
    request = row.get("request")
    try:
        if not path.is_file() or file_sha256(path) != str(row.get("file_sha256")):
            return False
        from .syncnet import video_signature

        signature = video_signature(path)
        if (
            signature.get("video_hash") != row.get("video_hash")
            or signature.get("pixel_sha256") != row.get("pixel_sha256")
            or signature.get("pts_sha256") != row.get("pts_sha256")
            or signature.get("frame_count") != row.get("frame_count")
        ):
            return False
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            return False
        expected_request = _request_for_row(paths, row, audio_record, portrait, provenance)
        request_path = Path(str(request["path"]))
        if request_path.resolve() != Path(str(expected_request["output_dir"])).resolve() / "request.json":
            return False
        request_value = read_self_hashed_json(request_path)
        request_body = dict(request_value)
        request_body.pop("artifact_sha256", None)
        if request_body != expected_request:
            return False
        validate_response(Path(str(response["path"])), request_value, provenance)
    except (OSError, ProtocolError, KeyError, TypeError, ValueError):
        return False
    return True


def generation_stage(
    paths: config.RunPaths,
    assets: Mapping[str, Any],
    audio_manifest: Mapping[str, Any],
    *,
    model_config: Path | None = None,
) -> dict[str, Any]:
    root = paths.generation
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    provenance = discover_leaptalk(model_config)
    write_self_hashed_json(paths.provenance, {**provenance, "protocol_id": config.PROTOCOL_ID, "schema_version": 1})
    reason = str(provenance.get("reason", "LeapTalk dependency is unavailable"))
    expected = _expected_generation_rows()
    if provenance.get("status") == "DEPENDENCY_BLOCKED":
        result = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "stage": "03_generation",
            "status": "DEPENDENCY_BLOCKED",
            "expected_science_video_count": config.EXPECTED_B_VIDEOS,
            "expected_repeat_video_count": config.EXPECTED_B_REPEATS,
            "science_video_count": 0,
            "repeat_video_count": 0,
            "planned_science_video_count": config.EXPECTED_B_VIDEOS,
            "planned_repeat_video_count": config.EXPECTED_B_REPEATS,
            "blocked_video_count": config.EXPECTED_B_VIDEOS + config.EXPECTED_B_REPEATS,
            "failed_video_count": 0,
            "resource_wait_count": 0,
            "videos": _blocked_generation_rows(paths, assets, audio_manifest, reason),
            "failures": [{"error_type": "DEPENDENCY_BLOCKED", "reason": reason}],
            "provenance_status": provenance.get("status"),
            "no_wav2lip_substitution": True,
        }
        return write_self_hashed_json(manifest_path, result)

    audio = _audio_lookup(audio_manifest)
    asset = _asset_lookup(assets)
    existing: dict[str, Mapping[str, Any]] = {}
    if manifest_path.is_file():
        try:
            cached = read_self_hashed_json(manifest_path)
            existing = {str(row.get("cell_id")): row for row in cached.get("videos", []) if row.get("cell_id")}
        except ProtocolError:
            existing = {}
    videos: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def persist() -> None:
        complete_science = sum(row.get("stage") == "B_GENERATION" and row.get("status") == "COMPLETE" for row in videos)
        complete_repeats = sum(row.get("stage") == "B_GENERATION_REPEAT" and row.get("status") == "COMPLETE" for row in videos)
        blocked = sum(row.get("status") == "DEPENDENCY_BLOCKED" for row in videos)
        resource_wait = sum(row.get("status") == "RESOURCE_WAIT" for row in videos)
        failed = sum(row.get("status") in {"INCOMPLETE", "INPUT_INVALID"} for row in videos)
        if any(row.get("status") == "RESOURCE_WAIT" for row in videos):
            stage_status = "RESOURCE_WAIT"
        elif any(row.get("status") == "DEPENDENCY_BLOCKED" for row in videos):
            stage_status = "DEPENDENCY_BLOCKED"
        elif len(videos) == len(expected) and complete_science == config.EXPECTED_B_VIDEOS and complete_repeats == config.EXPECTED_B_REPEATS and not failures:
            stage_status = "COMPLETE"
        else:
            stage_status = "INCOMPLETE"
        write_self_hashed_json(manifest_path, {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "stage": "03_generation",
            "status": stage_status,
            "expected_science_video_count": config.EXPECTED_B_VIDEOS,
            "expected_repeat_video_count": config.EXPECTED_B_REPEATS,
            "science_video_count": complete_science,
            "repeat_video_count": complete_repeats,
            "planned_science_video_count": config.EXPECTED_B_VIDEOS,
            "planned_repeat_video_count": config.EXPECTED_B_REPEATS,
            "blocked_video_count": blocked,
            "failed_video_count": failed,
            "resource_wait_count": resource_wait,
            "videos": videos,
            "failures": failures,
            "provenance_status": provenance.get("status"),
            "no_wav2lip_substitution": True,
        })

    for raw in expected:
        sid = int(raw["id"])
        source = str(raw["source"])
        condition = str(raw["driver_condition"])
        audio_record = audio.get((sid, source, condition))
        portrait = asset.get(sid, {}).get("portrait") if isinstance(asset.get(sid), Mapping) else None
        if not isinstance(audio_record, Mapping) or not isinstance(portrait, Mapping):
            exc = InputInvalidError(f"missing audio or portrait binding for {sid}/{source}/{condition}")
            row = _row_from_failure(paths, raw, audio_record or {}, assets, exc, status="INPUT_INVALID")
            failures.append({**raw, "error_type": "INPUT_INVALID", "reason": str(exc)})
            videos.append(row)
            persist()
            continue
        cached = existing.get(str(raw["cell_id"]))
        if cached is not None and _valid_generation_cache(paths, cached, audio_record, portrait, provenance):
            videos.append({**dict(cached), "resumed": True, "execution_record_preserved": True})
            persist()
            continue
        try:
            videos.append(_run_one_generation(paths, {**raw, "source_group": asset[sid]["source_group"]}, audio_record, portrait, provenance))
        except ResourceWaitError as exc:
            failures.append({**raw, "error_type": "RESOURCE_WAIT", "reason": str(exc)})
            videos.append(_row_from_failure(paths, raw, audio_record, assets, exc, status="RESOURCE_WAIT"))
            # A foreign GPU process or lease must stop the queue.  Keep the
            # remaining planned identities explicit without starting them.
            for pending in expected[len(videos) :]:
                pending_audio = audio.get((int(pending["id"]), str(pending["source"]), str(pending["driver_condition"])), {})
                videos.append(_row_from_failure(paths, pending, pending_audio, assets, exc, status="RESOURCE_WAIT"))
            persist()
            break
        except DependencyBlockedError as exc:
            failures.append({**raw, "error_type": "DEPENDENCY_BLOCKED", "reason": str(exc)})
            videos.append(_row_from_failure(paths, raw, audio_record, assets, exc, status="DEPENDENCY_BLOCKED"))
            for pending in expected[len(videos) :]:
                pending_audio = audio.get((int(pending["id"]), str(pending["source"]), str(pending["driver_condition"])), {})
                videos.append(_row_from_failure(paths, pending, pending_audio, assets, exc, status="DEPENDENCY_BLOCKED"))
            persist()
            break
        except (OSError, ProtocolError, KeyError, TypeError, ValueError) as exc:
            failures.append({**raw, "error_type": "GENERATION_FAILED", "reason": str(exc)})
            videos.append(_row_from_failure(paths, raw, audio_record, assets, exc))
            persist()
            continue
        persist()
    if len(videos) < len(expected):
        for pending in expected[len(videos) :]:
            pending_audio = audio.get((int(pending["id"]), str(pending["source"]), str(pending["driver_condition"])), {})
            videos.append(_row_from_failure(paths, pending, pending_audio, assets, ResourceWaitError("RESOURCE_WAIT: generation queue stopped"), status="RESOURCE_WAIT"))
        persist()
    return read_self_hashed_json(manifest_path)


def _select_track(tracks: Any) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(tracks if isinstance(tracks, list) else []):
        if not isinstance(item, Mapping) or not isinstance(item.get("track"), Mapping):
            continue
        frame = np.asarray(item["track"].get("frame"), dtype=np.int64)
        bbox = np.asarray(item["track"].get("bbox"), dtype=np.float64)
        if (
            frame.ndim != 1
            or bbox.ndim != 2
            or bbox.shape != (frame.size, 4)
            or frame.size <= config.OFFICIAL_PIPELINE["min_track"]
            or np.any(np.diff(frame) != 1)
        ):
            continue
        candidates.append({
            "track_index": index,
            "frame_indices": [int(value) for value in frame.tolist()],
            "bbox": [[float(value) for value in row] for row in bbox.tolist()],
            "frame_count": int(frame.size),
            "start_frame": int(frame[0]),
            "end_frame": int(frame[-1]),
            "bbox_sha256": hashlib.sha256(np.ascontiguousarray(bbox).tobytes()).hexdigest(),
        })
    if not candidates:
        raise ProtocolError("official face pipeline returned no usable baseline track")
    candidates.sort(key=lambda row: (-int(row["frame_count"]), int(row["start_frame"]), int(row["track_index"])))
    return candidates[0]


def _freeze_baseline_roi(paths: config.RunPaths, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Run the official tracker once and freeze its trajectory for a B cell."""

    sample_id = int(baseline["id"])
    source = str(baseline["source"])
    seed = int(baseline["seed"])
    roi_path = paths.crossed / "roi" / str(sample_id) / source / f"seed{seed}.json"
    from .syncnet import video_signature

    baseline_path = Path(str(baseline["path"]))
    signature = video_signature(baseline_path)
    if roi_path.is_file():
        try:
            cached = read_self_hashed_json(roi_path)
            frame_indices = np.asarray(cached.get("frame_indices", []), dtype=np.int64)
            boxes = np.asarray(cached.get("bbox", []), dtype=np.float64)
            frame_shape = [int(item) for item in signature.get("frame_shape", [])]
            height, width = frame_shape[:2]
            start_frame = int(frame_indices[0]) if frame_indices.size else -1
            end_frame = int(frame_indices[-1]) + 1 if frame_indices.size else -1
            cache_is_valid = bool(
                cached.get("status") == "COMPLETE"
                and cached.get("protocol_id") == config.PROTOCOL_ID
                and int(cached.get("sample_id", -1)) == sample_id
                and str(cached.get("source")) == source
                and int(cached.get("seed", -1)) == seed
                and cached.get("baseline_cell_id") == baseline.get("cell_id")
                and str(cached.get("baseline_video_path")) == str(baseline_path)
                and cached.get("baseline_video_hash") == baseline.get("video_hash")
                and cached.get("baseline_pixel_sha256") == baseline.get("pixel_sha256")
                and cached.get("source_frame_shape") == signature.get("frame_shape")
                and frame_indices.ndim == 1
                and frame_indices.size >= config.MIN_INTERIOR_ROWS
                and np.all(np.diff(frame_indices) == 1)
                and boxes.shape == (frame_indices.size, 4)
                and np.all(np.isfinite(boxes))
                and np.all(boxes[:, 0] >= 0)
                and np.all(boxes[:, 1] >= 0)
                and np.all(boxes[:, 2] <= width)
                and np.all(boxes[:, 3] <= height)
                and np.all(boxes[:, 2] > boxes[:, 0])
                and np.all(boxes[:, 3] > boxes[:, 1])
                and 0 <= start_frame < end_frame <= int(signature["frame_count"])
                and int(cached.get("time_start_frame", -1)) == start_frame
                and int(cached.get("time_end_frame_exclusive", -1)) == end_frame
                and int(cached.get("audio_start_samples", -1)) == start_frame * config.SAMPLES_PER_FRAME
                and int(cached.get("audio_end_samples", -1)) == end_frame * config.SAMPLES_PER_FRAME
                and cached.get("official_pipeline_config") == config.OFFICIAL_PIPELINE
                and cached.get("padding_rows") == []
                and cached.get("candidate_tracking_forbidden") is True
                and isinstance(cached.get("track"), Mapping)
                and cached["track"].get("frame_indices") == cached.get("frame_indices")
                and cached["track"].get("bbox") == cached.get("bbox")
            )
            if cache_is_valid:
                return cached
        except (ProtocolError, KeyError, TypeError, ValueError):
            pass
    work_dir = paths.crossed / "roi_work" / str(sample_id) / source / f"seed{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference = f"tts_native_gain_b_{sample_id}_{source}_{seed}"
    command = [
        str(config.SYNCNET_PYTHON), str(config.SYNCNET_PIPELINE),
        "--videofile", str(baseline_path), "--reference", reference,
        "--data_dir", str(work_dir),
        "--facedet_scale", str(config.OFFICIAL_PIPELINE["facedet_scale"]),
        "--crop_scale", str(config.OFFICIAL_PIPELINE["crop_scale"]),
        "--min_track", str(config.OFFICIAL_PIPELINE["min_track"]),
        "--frame_rate", str(config.OFFICIAL_PIPELINE["frame_rate"]),
        "--num_failed_det", str(config.OFFICIAL_PIPELINE["num_failed_det"]),
        "--min_face_size", str(config.OFFICIAL_PIPELINE["min_face_size"]),
        "--overwrite",
    ]
    with gpu_lease(gpu_peak_bytes=config.GPU_PEAK_BUDGET_BYTES, disk_temp_bytes=config.CELL_TEMP_BUDGET_BYTES, disk_persistent_bytes=8 << 20):
        run_monitored(command, cwd=config.SYNCNET_ROOT, log_path=work_dir / "pipeline.log", interval_seconds=30.0)
    track_path = work_dir / "pywork" / reference / "tracks.pckl"
    if not track_path.is_file():
        raise ProtocolError(f"baseline ROI tracks are missing: {track_path}")
    try:
        with track_path.open("rb") as handle:
            track = _select_track(pickle.load(handle))
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise ProtocolError(f"cannot read baseline ROI tracks: {track_path}") from exc
    frame_shape = [int(item) for item in signature.get("frame_shape", [])]
    if len(frame_shape) < 2:
        raise ProtocolError("baseline ROI source frame geometry is missing")
    height, width = frame_shape[0], frame_shape[1]
    boxes = np.asarray(track["bbox"], dtype=np.float64)
    if (
        np.any(~np.isfinite(boxes))
        or np.any(boxes[:, 0] < 0)
        or np.any(boxes[:, 2] > width)
        or np.any(boxes[:, 1] < 0)
        or np.any(boxes[:, 3] > height)
        or np.any(boxes[:, 2] <= boxes[:, 0])
        or np.any(boxes[:, 3] <= boxes[:, 1])
        or int(track["frame_indices"][-1]) >= int(signature["frame_count"])
    ):
        raise ProtocolError("official baseline ROI is outside the source video")
    start_frame = int(track["frame_indices"][0])
    end_frame = int(track["frame_indices"][-1]) + 1
    roi = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": sample_id,
        "source": source,
        "seed": seed,
        "baseline_cell_id": baseline["cell_id"],
        "baseline_video_path": str(baseline_path),
        "baseline_video_hash": baseline["video_hash"],
        "baseline_pixel_sha256": baseline.get("pixel_sha256"),
        "source_frame_shape": signature.get("frame_shape"),
        "time_start_frame": start_frame,
        "time_end_frame_exclusive": end_frame,
        "audio_start_samples": start_frame * config.SAMPLES_PER_FRAME,
        "audio_end_samples": end_frame * config.SAMPLES_PER_FRAME,
        "official_pipeline_config": dict(config.OFFICIAL_PIPELINE),
        "selection_rule": "longest continuous track, earliest start frame, track index",
        "track": track,
        "frame_indices": track["frame_indices"],
        "bbox": track["bbox"],
        "padding_rows": [],
        "candidate_tracking_forbidden": True,
        "status": "COMPLETE",
    }
    return write_self_hashed_json(roi_path, roi)


def _median_filter(values: np.ndarray, width: int = 13) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return array.copy()
    half = width // 2
    padded = np.pad(array, (half, half), mode="edge")
    return np.asarray([np.median(padded[index : index + width]) for index in range(array.size)], dtype=np.float64)


def apply_frozen_roi(video_path: Path, roi: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    """Apply baseline boxes to a candidate without candidate-specific tracking."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("opencv is required for frozen ROI application") from exc
    from .syncnet import video_signature

    frame_indices = np.asarray(roi.get("frame_indices"), dtype=np.int64)
    boxes = np.asarray(roi.get("bbox"), dtype=np.float64)
    if frame_indices.ndim != 1 or boxes.shape != (frame_indices.size, 4) or frame_indices.size <= config.OFFICIAL_PIPELINE["min_track"] or np.any(np.diff(frame_indices) != 1):
        raise ProtocolError("frozen ROI trajectory is malformed")
    source_signature = video_signature(video_path)
    _validate_video_rate(source_signature["pts"])
    if int(frame_indices[-1]) >= int(source_signature["frame_count"]):
        raise ProtocolError(f"candidate does not support frozen ROI frames: {video_path}")
    source_shape = [int(item) for item in source_signature.get("frame_shape", [])]
    if len(source_shape) < 2:
        raise ProtocolError(f"candidate frame geometry is unavailable: {video_path}")
    height, width = source_shape[0], source_shape[1]
    if (
        np.any(~np.isfinite(boxes))
        or np.any(boxes[:, 0] < 0)
        or np.any(boxes[:, 2] > width)
        or np.any(boxes[:, 1] < 0)
        or np.any(boxes[:, 3] > height)
        or np.any(boxes[:, 2] <= boxes[:, 0])
        or np.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ProtocolError(f"frozen ROI geometry leaves the candidate frame: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames or frame_indices[-1] >= len(frames) or frame_indices[0] < 0:
        raise ProtocolError(f"candidate does not support frozen ROI frames: {video_path}")
    if list(roi.get("source_frame_shape", [])) and [int(item) for item in frames[0].shape] != [int(item) for item in roi["source_frame_shape"]]:
        raise ProtocolError(f"candidate frame geometry differs from baseline: {video_path}")
    centers_x = _median_filter((boxes[:, 0] + boxes[:, 2]) / 2.0)
    centers_y = _median_filter((boxes[:, 1] + boxes[:, 3]) / 2.0)
    sizes = _median_filter(np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0])) / 2.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.avi")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"XVID"), config.FPS, (224, 224))
    if not writer.isOpened():
        raise ProtocolError(f"cannot open frozen ROI writer: {temporary}")
    try:
        for frame_index, x, y, size in zip(frame_indices, centers_x, centers_y, sizes, strict=True):
            image = frames[int(frame_index)]
            bs = max(1.0, float(size))
            bsi = int(bs * (1.0 + 2.0 * float(config.OFFICIAL_PIPELINE["crop_scale"])))
            padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=(110, 110))
            my = float(y) + bsi
            mx = float(x) + bsi
            crop = padded[int(my - bs) : int(my + bs * (1.0 + 2.0 * float(config.OFFICIAL_PIPELINE["crop_scale"]))), int(mx - bs * (1.0 + float(config.OFFICIAL_PIPELINE["crop_scale"]))) : int(mx + bs * (1.0 + float(config.OFFICIAL_PIPELINE["crop_scale"]))) ]
            if crop.size == 0:
                raise ProtocolError(f"frozen ROI leaves the frame: {video_path}/{frame_index}")
            writer.write(cv2.resize(crop, (224, 224)))
    finally:
        writer.release()
    temporary.replace(destination)
    signature = video_signature(destination)
    if int(signature["frame_count"]) != int(frame_indices.size):
        raise ProtocolError(f"frozen ROI output frame count differs: {destination}")
    return signature


def _save_feature(path: Path, value: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)
    payload = {"path": str(path), "sha256": file_sha256(path), "shape": [int(item) for item in value.shape], "dtype": str(value.dtype), **dict(metadata)}
    return write_self_hashed_json(path.with_suffix(".json"), payload)


def _load_feature(path: Path, expected: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_self_hashed_json(metadata_path)
        if file_sha256(path) != str(metadata.get("sha256")):
            return None
        for key, value in expected.items():
            if value is not None and metadata.get(key) != value:
                return None
        array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != config.EMBEDDING_DIM or not np.isfinite(array).all():
            return None
        return array, metadata
    except (OSError, ProtocolError, ValueError):
        return None


def _matrix_write(path: Path, matrix: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(matrix), allow_pickle=False)
    temporary.replace(path)
    return file_sha256(path)


def _support_hash(rows: Sequence[int], *, rule: str, scope: Sequence[Any]) -> str:
    return canonical_sha256({"rows": [int(item) for item in rows], "rule": rule, "scope": [str(item) for item in scope]})


def _score_code_hash() -> str:
    paths = (Path(__file__), Path(__file__).with_name("syncnet.py"), Path(__file__).with_name("common.py"))
    return canonical_sha256({path.name: file_sha256(path) for path in paths if path.is_file()})


def _score_row(
    generation: Mapping[str, Any],
    audio: Mapping[str, Any],
    crop: Mapping[str, Any],
    matrix_path: Path,
    matrix_hash: str,
    *,
    video_driver: str,
    eval_condition: str,
) -> dict[str, Any]:
    return {
        "stage": "B_SCORE",
        "id": int(generation["id"]),
        "sample_id": int(generation["id"]),
        "source_group": str(generation["source_group"]),
        "source": str(generation["source"]),
        "video_type": "V0" if video_driver == "A0" else f"V_{video_driver}",
        "driver_condition": video_driver,
        "eval_condition": eval_condition,
        "seed": int(generation["seed"]),
        "cell_id": f"BSCORE:{generation['id']}:{generation['source']}:{generation['seed']}:{video_driver}:{eval_condition}",
        "video_cell_id": generation["cell_id"],
        "video_path": generation["path"],
        "video_hash": generation["video_hash"],
        "cropped_video_path": crop["path"],
        "cropped_video_hash": crop["video_hash"],
        "cropped_pixel_sha256": crop.get("pixel_sha256"),
        "cropped_pts_sha256": crop.get("pts_sha256"),
        "cropped_frame_count": crop.get("frame_count"),
        "pcm_path": audio["path"],
        "pcm_file_sha256": audio.get("file_sha256"),
        "pcm_hash": audio["pcm_sha256"],
        "audio_clock_hash": audio.get("artifact_sha256"),
        "audio_source_path": audio.get("source_audio_path"),
        "audio_source_file_sha256": audio.get("source_audio_file_sha256"),
        "audio_source_pcm_sha256": audio.get("source_audio_pcm_sha256"),
        "audio_start_samples": audio.get("sample_start"),
        "audio_end_samples": audio.get("sample_end"),
        "time_start_frame": int(audio.get("time_start_frame", crop.get("time_start_frame", 0))),
        "time_end_frame_exclusive": int(audio.get("time_end_frame_exclusive", crop.get("time_end_frame_exclusive", 0))),
        "roi_hash": crop["roi_hash"],
        "roi_path": crop["roi_path"],
        "support_hash": None,
        "support_rows": [],
        "support_scope": None,
        "visual_feature_path": crop["visual_feature_path"],
        "visual_feature_hash": crop["visual_feature_hash"],
        "audio_feature_path": audio["audio_feature_path"],
        "audio_feature_hash": audio["audio_feature_hash"],
        "model_hash": generation["model_hash"],
        "code_hash": _score_code_hash(),
        "matrix_path": str(matrix_path),
        "matrix_hash": matrix_hash,
        "matrix_shape": [int(item) for item in np.load(matrix_path, allow_pickle=False).shape],
        "reused_from": None,
        "status": "COMPLETE",
    }


def _repeat_control_result(
    baseline: np.ndarray,
    repeat: np.ndarray,
    baseline_support: Sequence[int],
    repeat_support: Sequence[int],
) -> dict[str, Any]:
    from .analysis import curve_metrics

    base = curve_metrics(baseline, baseline_support)
    current = curve_metrics(repeat, repeat_support)
    if len(baseline_support) != len(repeat_support):
        raise ProtocolError("repeat control support lengths differ")
    aligned_error = float(
        np.max(
            np.abs(
                baseline[np.asarray(baseline_support, dtype=np.int64)]
                - repeat[np.asarray(repeat_support, dtype=np.int64)]
            )
        )
    )
    delta_c = float(current["sync_c"] - base["sync_c"])
    delta_d = float(current["sync_d"] - base["sync_d"])
    offset_delta = int(current["official_offset"] - base["official_offset"])
    status = "IDENTITY_PASS" if abs(delta_c) <= 0.05 and abs(delta_d) <= 0.05 and abs(offset_delta) <= 1 else "GENERATION_UNSTABLE"
    return {"status": status, "delta_sync_c": delta_c, "delta_sync_d": delta_d, "offset_delta": offset_delta, "aligned_matrix_max_abs_error": aligned_error, "baseline": base, "repeat": current, "tolerances": {"sync_c": 0.05, "sync_d": 0.05, "offset_frames": 1}}


def _independent_audio_shift(values: np.ndarray, amount: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int16)
    result = np.zeros_like(array)
    if amount >= 0:
        if amount < array.size:
            result[amount:] = array[: array.size - amount]
    else:
        amount = -amount
        if amount < array.size:
            result[: array.size - amount] = array[amount:]
    return result


def _prepare_b_media(
    paths: config.RunPaths,
    generation: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    baseline = {(int(row["id"]), str(row["source"]), int(row["seed"])): row for row in generation if row.get("stage") == "B_GENERATION" and row.get("driver_condition") == "A0" and row.get("status") == "COMPLETE"}
    roi_by_scope: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for key, row in baseline.items():
        roi_by_scope[key] = _freeze_baseline_roi(paths, row)
    crops: dict[str, Mapping[str, Any]] = {}
    for row in generation:
        if row.get("status") != "COMPLETE":
            continue
        key = (int(row["id"]), str(row["source"]), int(row["seed"]))
        roi = roi_by_scope.get(key)
        if roi is None:
            raise ProtocolError(f"baseline ROI is missing for {key}")
        destination = paths.crossed / "roi_crops" / str(row["id"]) / str(row["source"]) / f"seed{row['seed']}" / str(row.get("role", "main")) / f"{row['driver_condition']}.avi"
        if not destination.is_file():
            signature = apply_frozen_roi(Path(str(row["path"])), roi, destination)
        else:
            from .syncnet import video_signature

            signature = video_signature(destination)
        roi_hash = canonical_sha256({"roi": roi.get("artifact_sha256"), "baseline": roi.get("baseline_video_hash"), "candidate": row.get("video_hash"), "frame_indices": roi.get("frame_indices")})
        crops[str(row["cell_id"])] = {
            "path": str(destination),
            "video_hash": signature["video_hash"],
            "pixel_sha256": signature["pixel_sha256"],
            "pts_sha256": signature["pts_sha256"],
            "roi_hash": roi_hash,
            "roi_path": str(paths.crossed / "roi" / str(row["id"]) / str(row["source"]) / f"seed{row['seed']}.json"),
            "roi_artifact_sha256": roi.get("artifact_sha256"),
            "baseline_cell_id": roi.get("baseline_cell_id"),
            "time_start_frame": int(roi.get("time_start_frame", roi["frame_indices"][0])),
            "time_end_frame_exclusive": int(roi.get("time_end_frame_exclusive", roi["frame_indices"][-1] + 1)),
            "audio_start_samples": int(roi.get("audio_start_samples", int(roi["frame_indices"][0]) * config.SAMPLES_PER_FRAME)),
            "audio_end_samples": int(roi.get("audio_end_samples", (int(roi["frame_indices"][-1]) + 1) * config.SAMPLES_PER_FRAME)),
            "frame_count": signature["frame_count"],
            "padding_rows": roi.get("padding_rows", []),
        }
    return crops


def _clock_audio_record(
    root: Path,
    audio_record: Mapping[str, Any],
    crop: Mapping[str, Any],
    *,
    sample_id: int,
    source: str,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    """Materialize the exact audio interval covered by a frozen ROI.

    The official SyncNet cropper starts a face crop at the first tracked video
    frame and crops its audio to the same interval.  B must preserve that
    clock even though the generated MP4 is scored through a separate PCM
    input.  In particular, using the full source WAV here would silently
    introduce a seed-dependent leading offset whenever the baseline track
    starts after frame zero.
    """

    from .audio import read_pcm16_wav, write_pcm16_wav

    source_path = Path(str(audio_record.get("path", "")))
    if not source_path.is_file():
        raise InputInvalidError(f"B source audio is missing: {source_path}")
    source_file_hash = file_sha256(source_path)
    if source_file_hash != str(audio_record.get("file_sha256")):
        raise InputInvalidError(f"B source audio container hash differs: {source_path}")
    source_pcm = read_pcm16_wav(source_path)
    source_pcm_hash = hashlib.sha256(np.asarray(source_pcm, dtype="<i2").tobytes()).hexdigest()
    if source_pcm_hash != str(audio_record.get("pcm_sha256")):
        raise InputInvalidError(f"B source audio PCM hash differs: {source_path}")

    start_frame = int(crop.get("time_start_frame", 0))
    end_frame = int(crop.get("time_end_frame_exclusive", start_frame + int(crop.get("frame_count", 0))))
    start_samples = int(crop.get("audio_start_samples", start_frame * config.SAMPLES_PER_FRAME))
    end_samples = int(crop.get("audio_end_samples", end_frame * config.SAMPLES_PER_FRAME))
    if start_frame < 0 or end_frame <= start_frame or start_samples != start_frame * config.SAMPLES_PER_FRAME or end_samples != end_frame * config.SAMPLES_PER_FRAME:
        raise ProtocolError(f"B ROI/audio clock binding is malformed: {sample_id}/{source}/seed{seed}")
    if end_samples > source_pcm.size:
        raise ProtocolError(
            f"B ROI audio interval exceeds source WAV: {sample_id}/{source}/seed{seed} "
            f"[{start_samples},{end_samples})/{source_pcm.size}"
        )
    clipped = np.asarray(source_pcm[start_samples:end_samples], dtype=np.int16)
    if clipped.size < config.SAMPLES_PER_FRAME * 6:
        raise ProtocolError(f"B ROI audio interval is too short: {sample_id}/{source}/seed{seed}")
    destination = root / "clock_audio" / str(sample_id) / source / f"seed{seed}" / f"{condition}.wav"
    metadata_path = destination.with_suffix(".json")
    clock_body = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "04_crossed_scores_clock_audio",
        "sample_id": sample_id,
        "source": source,
        "seed": seed,
        "condition": condition,
        "source_audio_path": str(source_path),
        "source_audio_file_sha256": source_file_hash,
        "source_audio_pcm_sha256": source_pcm_hash,
        "sample_rate": config.SAMPLE_RATE,
        "fps": config.FPS,
        "time_start_frame": start_frame,
        "time_end_frame_exclusive": end_frame,
        "sample_start": start_samples,
        "sample_end": end_samples,
        "roi_hash": crop.get("roi_hash"),
        "roi_artifact_sha256": crop.get("roi_artifact_sha256"),
        "operation": "exact int16 slice on the source clock; no resampling, padding, or retiming",
    }
    if destination.is_file() and metadata_path.is_file():
        try:
            cached = read_self_hashed_json(metadata_path)
            if all(cached.get(key) == value for key, value in clock_body.items()) and file_sha256(destination) == cached.get("file_sha256"):
                cached_pcm = read_pcm16_wav(destination)
                if hashlib.sha256(np.asarray(cached_pcm, dtype="<i2").tobytes()).hexdigest() == cached.get("pcm_sha256") and np.array_equal(cached_pcm, clipped):
                    return cached
        except (OSError, ProtocolError, ValueError):
            pass
    file_hash = write_pcm16_wav(destination, clipped)
    return write_self_hashed_json(
        metadata_path,
        {
            **clock_body,
            "path": str(destination),
            "file_sha256": file_hash,
            "pcm_sha256": hashlib.sha256(np.asarray(clipped, dtype="<i2").tobytes()).hexdigest(),
            "sample_count": int(clipped.size),
        },
    )


def _common_time_support(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    *,
    trim_rows: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Intersect valid rows on the source frame clock, then map to each cell."""

    intervals: list[set[int]] = []
    starts: dict[str, int] = {}
    for row in rows:
        cell_id = str(row["cell_id"])
        matrix = arrays[cell_id]
        start = int(row.get("time_start_frame", 0))
        count = int(matrix.shape[0])
        if count <= 2 * int(trim_rows):
            return [], {}
        starts[cell_id] = start
        intervals.append(set(range(start + int(trim_rows), start + count - int(trim_rows))))
    if not intervals:
        return [], {}
    common = sorted(set.intersection(*intervals))
    mapped = {cell_id: [int(value - starts[cell_id]) for value in common] for cell_id in starts}
    return common, mapped


def _blocked_score_rows(assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], reason: str = "B generation is unavailable") -> list[dict[str, Any]]:
    groups = {int(row["sample_id"]): str(row["source_group"]) for row in assets.get("records", [])}
    pcm = {(int(row["sample_id"]), str(row["source"]), str(row["condition"])): str(row["pcm_sha256"]) for row in audio_manifest.get("records", [])}
    rows: list[dict[str, Any]] = []
    specs = (("A0", "A0"), ("A0", "NOISE"), ("NOISE", "A0"), ("NOISE", "NOISE"), ("A0", "DENOISE"), ("DENOISE", "A0"), ("DENOISE", "DENOISE"))
    for sample_id in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for seed in config.SEEDS:
                for driver, condition in specs:
                    rows.append({"stage": "B_SCORE", "id": sample_id, "sample_id": sample_id, "source_group": groups.get(sample_id), "source": source, "video_type": "V0" if driver == "A0" else f"V_{driver}", "driver_condition": driver, "eval_condition": condition, "seed": seed, "cell_id": f"BSCORE:{sample_id}:{source}:{seed}:{driver}:{condition}", "video_hash": None, "pcm_hash": pcm.get((sample_id, source, condition)), "roi_hash": None, "support_hash": None, "support_rows": [], "support_scope": None, "model_hash": None, "code_hash": None, "matrix_hash": None, "reused_from": None, "status": "DEPENDENCY_BLOCKED", "error_type": "DEPENDENCY_BLOCKED", "reason": reason})
    for sample_id in config.CONTROL_IDS:
        for source in ("N", "T"):
            for control in ("REPEAT", "DELAY_PLUS_200MS"):
                rows.append({"stage": "B_CONTROL", "id": sample_id, "sample_id": sample_id, "source_group": groups.get(sample_id), "source": source, "video_type": "V0_REPEAT" if control == "REPEAT" else "V0", "driver_condition": control, "eval_condition": "A0" if control == "REPEAT" else control, "seed": 42, "cell_id": f"BCONTROL:{sample_id}:{source}:{control}", "video_hash": None, "pcm_hash": pcm.get((sample_id, source, "A0")), "roi_hash": None, "support_hash": None, "support_rows": [], "support_scope": None, "model_hash": None, "code_hash": None, "matrix_hash": None, "reused_from": None, "status": "DEPENDENCY_BLOCKED", "error_type": "DEPENDENCY_BLOCKED", "reason": reason})
    return rows


def _score_identity_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, int, str, str], Mapping[str, Any]]:
    return {(int(row["id"]), str(row["source"]), int(row["seed"]), str(row["driver_condition"]), str(row["eval_condition"])): row for row in rows if row.get("stage") == "B_SCORE"}


def crossed_score_stage(
    paths: config.RunPaths,
    assets: Mapping[str, Any],
    audio_manifest: Mapping[str, Any],
    generation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the complete B matrix with frozen ROI and common support."""

    root = paths.crossed
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if generation_manifest.get("status") != "COMPLETE":
        reason = "B generation is incomplete; no crossed score was started"
        blocked = _blocked_score_rows(assets, audio_manifest, reason)
        return write_self_hashed_json(manifest_path, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": "04_crossed_scores", "status": "DEPENDENCY_BLOCKED", "expected_science_cell_count": config.EXPECTED_B_SCIENCE, "expected_control_cell_count": config.EXPECTED_B_CONTROLS, "science_cell_count": 0, "control_cell_count": 0, "planned_science_cell_count": config.EXPECTED_B_SCIENCE, "planned_control_cell_count": config.EXPECTED_B_CONTROLS, "blocked_cell_count": config.EXPECTED_B_SCIENCE + config.EXPECTED_B_CONTROLS, "failed_cell_count": sum(row.get("status") not in {"COMPLETE", "DEPENDENCY_BLOCKED", "RESOURCE_WAIT"} for row in blocked), "resource_wait_count": sum(row.get("status") == "RESOURCE_WAIT" for row in blocked), "cells": [row for row in blocked if row.get("stage") == "B_SCORE"], "controls": [row for row in blocked if row.get("stage") == "B_CONTROL"], "primary_records": {}, "four_cell_decomposition": [], "failures": [{"error_type": "DEPENDENCY_BLOCKED", "reason": reason}], "no_a_substitution": True})

    generation = [row for row in generation_manifest.get("videos", []) if row.get("status") == "COMPLETE"]
    if len(generation) != config.EXPECTED_B_VIDEOS + config.EXPECTED_B_REPEATS:
        raise ImplementationIncompleteError("generation manifest says COMPLETE but does not contain 148 valid videos")
    audio = _audio_lookup(audio_manifest)
    crops = _prepare_b_media(paths, generation)
    gen_by_key = {
        (int(row["id"]), str(row["source"]), int(row["seed"]), str(row["driver_condition"])): row
        for row in generation
        if row.get("stage") == "B_GENERATION"
    }
    clock_audio: dict[tuple[int, str, int, str], Mapping[str, Any]] = {}
    for sid in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for seed in config.SEEDS:
                baseline = gen_by_key[(sid, source, seed, "A0")]
                baseline_crop = crops[str(baseline["cell_id"])]
                for condition in config.B_DRIVERS:
                    clock_audio[(sid, source, seed, condition)] = _clock_audio_record(
                        root,
                        audio[(sid, source, condition)],
                        baseline_crop,
                        sample_id=sid,
                        source=source,
                        seed=seed,
                        condition=condition,
                    )
    from .audio import read_pcm16_wav, write_pcm16_wav
    from .syncnet import SyncNetEngine

    visual_features: dict[str, np.ndarray] = {}
    visual_meta: dict[str, dict[str, Any]] = {}
    audio_features: dict[tuple[int, str, int, str], np.ndarray] = {}
    audio_meta: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    shifted_features: dict[tuple[int, str], np.ndarray] = {}
    shifted_meta: dict[tuple[int, str], dict[str, Any]] = {}
    with gpu_lease(gpu_peak_bytes=config.GPU_PEAK_BUDGET_BYTES, disk_temp_bytes=config.SYNCNET_CELL_TEMP_BUDGET_BYTES, disk_persistent_bytes=128 << 20):
        engine = SyncNetEngine()
        try:
            for row in generation:
                cell_id = str(row["cell_id"])
                crop = crops[cell_id]
                feature_path = root / "features" / "visual" / f"{cell_id}.npy"
                cached = _load_feature(feature_path, {"video_hash": crop["video_hash"], "roi_hash": crop["roi_hash"]})
                if cached is None:
                    feature, metadata = engine.extract_visual(Path(str(crop["path"])))
                    meta = _save_feature(feature_path, feature, {"video_hash": crop["video_hash"], "roi_hash": crop["roi_hash"], "source_video_hash": row["video_hash"], "visual_decode_mode": config.SYNCNET_VISUAL_DECODE_MODE, **metadata})
                else:
                    feature, meta = cached
                visual_features[cell_id] = feature
                visual_meta[cell_id] = meta
            for sid in config.SAMPLE_IDS:
                for source in ("N", "T"):
                    for seed in config.SEEDS:
                        for condition in config.B_DRIVERS:
                            record = clock_audio[(sid, source, seed, condition)]
                            feature_path = root / "features" / "audio" / str(sid) / source / f"seed{seed}" / f"{condition}.npy"
                            cached = _load_feature(feature_path, {"pcm_hash": record["pcm_sha256"], "sample_start": record["sample_start"], "sample_end": record["sample_end"]})
                            if cached is None:
                                feature, metadata = engine.extract_audio(Path(str(record["path"])))
                                meta = _save_feature(
                                    feature_path,
                                    feature,
                                    {
                                        "pcm_hash": record["pcm_sha256"],
                                        "audio_path": record["path"],
                                        "clock_audio_hash": record.get("artifact_sha256"),
                                        "sample_start": record["sample_start"],
                                        "sample_end": record["sample_end"],
                                        "time_start_frame": record["time_start_frame"],
                                        "time_end_frame_exclusive": record["time_end_frame_exclusive"],
                                        **metadata,
                                    },
                                )
                            else:
                                feature, meta = cached
                            audio_features[(sid, source, seed, condition)] = feature
                            audio_meta[(sid, source, seed, condition)] = meta
            for sid in config.CONTROL_IDS:
                for source in ("N", "T"):
                    base_record = clock_audio[(sid, source, 42, "A0")]
                    base = read_pcm16_wav(Path(str(base_record["path"])))
                    shifted = _independent_audio_shift(base, 3200)
                    shifted_path = root / "controls" / str(sid) / source / "PLUS_200MS.wav"
                    if not shifted_path.is_file() or not np.array_equal(read_pcm16_wav(shifted_path), shifted):
                        write_pcm16_wav(shifted_path, shifted)
                    shifted_hash = hashlib.sha256(np.asarray(shifted, dtype="<i2").tobytes()).hexdigest()
                    shifted_clock = write_self_hashed_json(
                        shifted_path.with_suffix(".json"),
                        {
                            **{key: base_record[key] for key in ("schema_version", "protocol_id", "sample_id", "source", "time_start_frame", "time_end_frame_exclusive", "sample_start", "sample_end", "source_audio_path", "source_audio_file_sha256", "source_audio_pcm_sha256")},
                            "stage": "04_crossed_scores_control_audio",
                            "condition": "DELAY_PLUS_200MS",
                            "path": str(shifted_path),
                            "file_sha256": file_sha256(shifted_path),
                            "pcm_sha256": shifted_hash,
                            "sample_count": int(shifted.size),
                            "shift_samples": 3200,
                            "operation": "exact 3200-sample forward shift of the frozen A0 clock slice; no resampling or padding beyond the registered shift",
                        },
                    )
                    feature_path = root / "features" / "audio" / "controls" / str(sid) / source / "PLUS_200MS.npy"
                    cached = _load_feature(feature_path, {"pcm_hash": shifted_hash, "shift_samples": 3200})
                    if cached is None:
                        feature, metadata = engine.extract_audio(shifted_path)
                        meta = _save_feature(feature_path, feature, {"pcm_hash": shifted_hash, "shift_samples": 3200, "audio_path": str(shifted_path), **metadata})
                    else:
                        feature, meta = cached
                    shifted_features[(sid, source)] = feature
                    shifted_meta[(sid, source)] = {**meta, "clock_artifact_sha256": shifted_clock.get("artifact_sha256")}
        finally:
            engine.close()

    score_rows: list[dict[str, Any]] = []
    score_arrays: dict[str, np.ndarray] = {}
    crop_by_id = {str(row["cell_id"]): crops[str(row["cell_id"])] for row in generation}
    for sid in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for seed in config.SEEDS:
                for driver, condition in (("A0", "A0"), ("A0", "NOISE"), ("NOISE", "A0"), ("NOISE", "NOISE"), ("A0", "DENOISE"), ("DENOISE", "A0"), ("DENOISE", "DENOISE")):
                    video = gen_by_key[(sid, source, seed, driver)]
                    crop = crop_by_id[str(video["cell_id"])]
                    audio_record = clock_audio[(sid, source, seed, condition)]
                    visual = visual_features[str(video["cell_id"])]
                    auditory = audio_features[(sid, source, seed, condition)]
                    matrix = SyncNetEngine.distance_matrix(visual, auditory)
                    matrix_path = root / "matrices" / "science" / str(sid) / source / f"seed{seed}" / f"{driver}__{condition}.npy"
                    matrix_hash = _matrix_write(matrix_path, matrix)
                    crop_info = {**crop, "visual_feature_path": visual_meta[str(video["cell_id"])]["path"], "visual_feature_hash": visual_meta[str(video["cell_id"])]["sha256"]}
                    audio_info = {**audio_record, "audio_feature_path": audio_meta[(sid, source, seed, condition)]["path"], "audio_feature_hash": audio_meta[(sid, source, seed, condition)]["sha256"]}
                    row = _score_row(video, audio_info, crop_info, matrix_path, matrix_hash, video_driver=driver, eval_condition=condition)
                    score_rows.append(row)
                    score_arrays[str(row["cell_id"])] = matrix

    support_groups: dict[tuple[int, str], dict[str, Any]] = {}
    for sid in config.SAMPLE_IDS:
        for source in ("N", "T"):
            current = [row for row in score_rows if int(row["id"]) == sid and row["source"] == source]
            expected_group_size = len(config.SEEDS) * 7
            if len(current) != expected_group_size:
                raise ImplementationIncompleteError(f"B score group has {len(current)}/{expected_group_size} cells: {sid}/{source}")
            support_time_frames, mapped_support = _common_time_support(current, score_arrays, trim_rows=config.VSHIFT)
            if len(support_time_frames) < config.MIN_INTERIOR_ROWS:
                for row in current:
                    row["status"] = "INCOMPLETE"
                    row["error_type"] = "SUPPORT_TOO_SHORT"
                    row["reason"] = f"common support has {len(support_time_frames)} rows"
                continue
            support_hash = _support_hash(support_time_frames, rule="common fourteen-cell valid time intersection after vshift", scope=(sid, source, "seeds=42,43"))
            support_groups[(sid, source)] = {"support_time_frames": support_time_frames, "support_rows_by_cell": mapped_support, "support_hash": support_hash, "count": len(support_time_frames), "scope": {"id": sid, "source": source, "seeds": list(config.SEEDS), "cell_count": expected_group_size}}
            for row in current:
                row.update({"support_rows": mapped_support[str(row["cell_id"])], "support_time_frames": support_time_frames, "support_hash": support_hash, "support_scope": support_groups[(sid, source)]["scope"]})
                from .analysis import curve_metrics

                row["metrics"] = curve_metrics(score_arrays[str(row["cell_id"])], mapped_support[str(row["cell_id"])], label=str(row["cell_id"]))
    write_self_hashed_json(root / "support_manifest.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "COMPLETE" if len(support_groups) == len(config.SAMPLE_IDS) * 2 else "INCOMPLETE", "groups": [support_groups[key] for key in sorted(support_groups)], "forbid_candidate_tracking": True, "padding_excluded": True, "support_scope": "all fourteen cells for each id/source across both seeds"})

    score_map = _score_identity_map(score_rows)
    decompositions: list[dict[str, Any]] = []
    primary_records: dict[str, Any] = {}
    for sid in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for driver in ("NOISE", "DENOISE"):
                seed_values: list[float] = []
                for seed in config.SEEDS:
                    cells = [score_map.get((sid, source, seed, "A0", "A0")), score_map.get((sid, source, seed, "A0", driver)), score_map.get((sid, source, seed, driver, "A0")), score_map.get((sid, source, seed, driver, driver))]
                    if any(item is None or item.get("status") != "COMPLETE" for item in cells):
                        continue
                    from .analysis import four_cell

                    values = [float(item["metrics"]["sync_c"]) for item in cells]
                    decomposition = four_cell(*values, metric="C")
                    decomposition.update({"id": sid, "source_group": cells[0]["source_group"], "source": source, "seed": seed, "driver": driver, "support_hash": cells[0]["support_hash"], "q00_cell_id": cells[0]["cell_id"], "q01_cell_id": cells[1]["cell_id"], "q10_cell_id": cells[2]["cell_id"], "q11_cell_id": cells[3]["cell_id"]})
                    decompositions.append(decomposition)
                    seed_values.append(float(decomposition["generation"]))
                if len(seed_values) == len(config.SEEDS):
                    primary_records[f"{sid}:{source}:{driver}"] = {"id": sid, "source_group": str(score_map[(sid, source, config.SEEDS[0], "A0", "A0")]["source_group"]), "source": source, "driver": driver, "value": float(np.mean(seed_values, dtype=np.float64)), "seed_values": seed_values, "estimand": "average seeds before source-group bootstrap"}
    for sid in config.SAMPLE_IDS:
        native: list[float] = []
        for seed in config.SEEDS:
            n = score_map.get((sid, "N", seed, "A0", "A0"))
            t = score_map.get((sid, "T", seed, "A0", "A0"))
            if n is not None and t is not None and n.get("status") == "COMPLETE" and t.get("status") == "COMPLETE":
                native.append(float(t["metrics"]["sync_c"] - n["metrics"]["sync_c"]))
        if len(native) == len(config.SEEDS):
            primary_records[f"{sid}:NATIVE"] = {"id": sid, "source_group": str(next(row["source_group"] for row in score_rows if int(row["id"]) == sid and row["source"] == "N")), "value": float(np.mean(native, dtype=np.float64)), "seed_values": native, "estimand": "TTS native minus natural; average seeds within source-specific supports"}

    controls: list[dict[str, Any]] = []
    generation_by_repeat = {(int(row["id"]), str(row["source"]), int(row["seed"])): row for row in generation if row.get("stage") == "B_GENERATION_REPEAT"}
    for sid in config.CONTROL_IDS:
        for source in ("N", "T"):
            base = score_map[(sid, source, 42, "A0", "A0")]
            repeat_generation = generation_by_repeat[(sid, source, 42)]
            repeat_visual = visual_features[str(repeat_generation["cell_id"])]
            base_audio_feature = audio_features[(sid, source, 42, "A0")]
            repeat_matrix = SyncNetEngine.distance_matrix(repeat_visual, base_audio_feature)
            repeat_crop = crops[str(repeat_generation["cell_id"])]
            repeat_scope = {"cell_id": str(repeat_generation["cell_id"]), "time_start_frame": repeat_crop.get("time_start_frame", 0)}
            repeat_arrays = {str(base["cell_id"]): score_arrays[str(base["cell_id"])], str(repeat_generation["cell_id"]): repeat_matrix}
            repeat_time_frames, repeat_supports = _common_time_support([base, repeat_scope], repeat_arrays, trim_rows=config.VSHIFT + 5)
            base_repeat_support = repeat_supports.get(str(base["cell_id"]), [])
            repeat_support = repeat_supports.get(str(repeat_generation["cell_id"]), [])
            if len(repeat_time_frames) < config.MIN_INTERIOR_ROWS:
                repeat_status = {"status": "CONTROL_UNINFORMATIVE", "reason": "control support below minimum"}
            else:
                repeat_status = _repeat_control_result(score_arrays[str(base["cell_id"])], repeat_matrix, base_repeat_support, repeat_support)
            repeat_path = root / "matrices" / "controls" / str(sid) / source / "REPEAT.npy"
            repeat_hash = _matrix_write(repeat_path, repeat_matrix)
            repeat_audio = clock_audio[(sid, source, 42, "A0")]
            repeat_feature = audio_meta[(sid, source, 42, "A0")]
            controls.append({
                "stage": "B_CONTROL", "id": sid, "sample_id": sid, "source_group": base["source_group"], "source": source,
                "video_type": "V0_REPEAT", "driver_condition": "REPEAT", "eval_condition": "A0", "seed": 42,
                "cell_id": f"BCONTROL:{sid}:{source}:REPEAT", "video_cell_id": repeat_generation["cell_id"],
                "video_path": repeat_generation["path"], "video_hash": repeat_generation["video_hash"],
                "cropped_video_path": repeat_crop["path"], "cropped_video_hash": repeat_crop["video_hash"],
                "cropped_pixel_sha256": repeat_crop.get("pixel_sha256"), "cropped_pts_sha256": repeat_crop.get("pts_sha256"), "cropped_frame_count": repeat_crop.get("frame_count"),
                "pcm_path": repeat_audio["path"], "pcm_file_sha256": repeat_audio.get("file_sha256"), "pcm_hash": repeat_audio["pcm_sha256"], "audio_clock_hash": repeat_audio.get("artifact_sha256"),
                "audio_source_path": repeat_audio.get("source_audio_path"), "audio_source_file_sha256": repeat_audio.get("source_audio_file_sha256"), "audio_source_pcm_sha256": repeat_audio.get("source_audio_pcm_sha256"),
                "audio_start_samples": repeat_audio.get("sample_start"), "audio_end_samples": repeat_audio.get("sample_end"),
                "time_start_frame": repeat_audio.get("time_start_frame", repeat_crop.get("time_start_frame", 0)), "time_end_frame_exclusive": repeat_audio.get("time_end_frame_exclusive", repeat_crop.get("time_end_frame_exclusive", 0)),
                "roi_hash": repeat_crop["roi_hash"], "roi_path": repeat_crop["roi_path"],
                "visual_feature_path": visual_meta[str(repeat_generation["cell_id"])]["path"], "visual_feature_hash": visual_meta[str(repeat_generation["cell_id"])]["sha256"],
                "audio_feature_path": repeat_feature["path"], "audio_feature_hash": repeat_feature["sha256"],
                "support_hash": _support_hash(repeat_time_frames, rule="control support trims five rows beyond vshift on shared source clock", scope=(sid, source)) if repeat_time_frames else None,
                "support_rows": base_repeat_support, "candidate_support_rows": repeat_support, "support_time_frames": repeat_time_frames,
                "support_scope": {"id": sid, "source": source, "control": "REPEAT"}, "model_hash": repeat_generation["model_hash"], "code_hash": _score_code_hash(),
                "matrix_path": str(repeat_path), "matrix_hash": repeat_hash, "matrix_shape": [int(item) for item in repeat_matrix.shape], "control": repeat_status, "status": "COMPLETE",
            })
            shifted = shifted_features[(sid, source)]
            delay_matrix = SyncNetEngine.distance_matrix(visual_features[str(gen_by_key[(sid, source, 42, "A0")]["cell_id"])], shifted)
            delay_path = root / "matrices" / "controls" / str(sid) / source / "DELAY_PLUS_200MS.npy"
            delay_hash = _matrix_write(delay_path, delay_matrix)
            base_generation = gen_by_key[(sid, source, 42, "A0")]
            base_crop = crops[str(base_generation["cell_id"])]
            delay_scope = {"cell_id": f"DELAY:{sid}:{source}", "time_start_frame": base.get("time_start_frame", base_crop.get("time_start_frame", 0))}
            delay_arrays = {str(base["cell_id"]): score_arrays[str(base["cell_id"])], str(delay_scope["cell_id"]): delay_matrix}
            delay_time_frames, delay_supports = _common_time_support([base, delay_scope], delay_arrays, trim_rows=config.VSHIFT + 5)
            base_delay_support = delay_supports.get(str(base["cell_id"]), [])
            shifted_support = delay_supports.get(str(delay_scope["cell_id"]), [])
            if len(delay_time_frames) < config.MIN_INTERIOR_ROWS:
                delay_status = {"status": "CONTROL_UNINFORMATIVE", "reason": "control support below minimum"}
            else:
                delay_status = _delay_control_result(score_arrays[str(base["cell_id"])], delay_matrix, base_delay_support, shifted_support)
            shifted_path = root / "controls" / str(sid) / source / "PLUS_200MS.wav"
            base_record = clock_audio[(sid, source, 42, "A0")]
            shifted_audio = {
                **base_record,
                "path": str(shifted_path),
                "file_sha256": file_sha256(shifted_path),
                "pcm_sha256": shifted_meta[(sid, source)]["pcm_hash"],
                "sample_count": int(read_pcm16_wav(shifted_path).size),
                "shift_samples": 3200,
                "artifact_sha256": shifted_meta[(sid, source)].get("clock_artifact_sha256"),
            }
            controls.append({
                "stage": "B_CONTROL", "id": sid, "sample_id": sid, "source_group": base["source_group"], "source": source,
                "video_type": "V0", "driver_condition": "DELAY_PLUS_200MS", "eval_condition": "DELAY_PLUS_200MS", "seed": 42,
                "cell_id": f"BCONTROL:{sid}:{source}:DELAY_PLUS_200MS", "video_cell_id": base_generation["cell_id"],
                "video_path": base_generation["path"], "video_hash": base_generation["video_hash"],
                "cropped_video_path": base_crop["path"], "cropped_video_hash": base_crop["video_hash"],
                "cropped_pixel_sha256": base_crop.get("pixel_sha256"), "cropped_pts_sha256": base_crop.get("pts_sha256"), "cropped_frame_count": base_crop.get("frame_count"),
                "pcm_path": shifted_audio["path"], "pcm_file_sha256": shifted_audio.get("file_sha256"), "pcm_hash": shifted_audio["pcm_sha256"], "audio_clock_hash": shifted_audio.get("artifact_sha256"),
                "audio_source_path": shifted_audio.get("source_audio_path"), "audio_source_file_sha256": shifted_audio.get("source_audio_file_sha256"), "audio_source_pcm_sha256": shifted_audio.get("source_audio_pcm_sha256"),
                "audio_start_samples": shifted_audio.get("sample_start"), "audio_end_samples": shifted_audio.get("sample_end"),
                "time_start_frame": shifted_audio.get("time_start_frame", base_crop.get("time_start_frame", 0)), "time_end_frame_exclusive": shifted_audio.get("time_end_frame_exclusive", base_crop.get("time_end_frame_exclusive", 0)), "shift_samples": 3200,
                "roi_hash": base_crop["roi_hash"], "roi_path": base_crop["roi_path"],
                "visual_feature_path": visual_meta[str(base_generation["cell_id"])]["path"], "visual_feature_hash": visual_meta[str(base_generation["cell_id"])]["sha256"],
                "audio_feature_path": shifted_meta[(sid, source)]["path"], "audio_feature_hash": shifted_meta[(sid, source)]["sha256"],
                "support_hash": _support_hash(delay_time_frames, rule="control support trims five rows beyond vshift on shared source clock", scope=(sid, source)) if delay_time_frames else None,
                "support_rows": base_delay_support, "candidate_support_rows": shifted_support, "support_time_frames": delay_time_frames,
                "support_scope": {"id": sid, "source": source, "control": "DELAY_PLUS_200MS"}, "model_hash": base_generation["model_hash"], "code_hash": _score_code_hash(),
                "matrix_path": str(delay_path), "matrix_hash": delay_hash, "matrix_shape": [int(item) for item in delay_matrix.shape], "control": delay_status, "status": "COMPLETE",
            })

    all_complete = len(score_rows) == config.EXPECTED_B_SCIENCE and all(row.get("status") == "COMPLETE" for row in score_rows)
    control_statuses = [str(row.get("control", {}).get("status")) for row in controls]
    control_gate = "PASS" if len(controls) == config.EXPECTED_B_CONTROLS and all(status in {"IDENTITY_PASS", "DELAY_DETECTED"} for status in control_statuses) else "CONTROL_LIMITED"
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "04_crossed_scores",
        "status": "COMPLETE" if all_complete and len(controls) == config.EXPECTED_B_CONTROLS else "INCOMPLETE",
        "expected_science_cell_count": config.EXPECTED_B_SCIENCE,
        "expected_control_cell_count": config.EXPECTED_B_CONTROLS,
        "science_cell_count": len(score_rows),
        "control_cell_count": len(controls),
        "planned_science_cell_count": config.EXPECTED_B_SCIENCE,
        "planned_control_cell_count": config.EXPECTED_B_CONTROLS,
        "blocked_cell_count": sum(row.get("status") == "DEPENDENCY_BLOCKED" for row in score_rows + controls),
        "failed_cell_count": sum(row.get("status") not in {"COMPLETE", "DEPENDENCY_BLOCKED", "RESOURCE_WAIT"} for row in score_rows + controls),
        "resource_wait_count": sum(row.get("status") == "RESOURCE_WAIT" for row in score_rows + controls),
        "cells": score_rows,
        "controls": controls,
        "primary_records": primary_records,
        "four_cell_decomposition": decompositions,
        "support_manifest": str(root / "support_manifest.json"),
        "control_gate": control_gate,
        "control_statuses": {status: control_statuses.count(status) for status in sorted(set(control_statuses))},
        "failures": [] if all_complete else [{"error_type": "B_SCORE_INCOMPLETE", "reason": "one or more B score cells lack valid common support"}],
        "no_a_substitution": True,
        "time_mean_before_lag_reduction": True,
        "distance_matrices_are_cpu_post_forward": True,
        "shared_support_across_seven_cells_and_two_seeds": True,
        "candidate_specific_tracking_forbidden": True,
    }
    return write_self_hashed_json(manifest_path, result)


def _delay_control_result(
    baseline: np.ndarray,
    shifted: np.ndarray,
    baseline_support: Sequence[int],
    shifted_support: Sequence[int],
) -> dict[str, Any]:
    from .analysis import curve_metrics

    base = curve_metrics(baseline, baseline_support)
    current = curve_metrics(shifted, shifted_support)
    expected = -5
    delta = int(current["official_offset"] - base["official_offset"])
    base_distance = float(np.mean(baseline[np.asarray(baseline_support), int(base["min_index"])]))
    shifted_distance = float(np.mean(shifted[np.asarray(shifted_support), int(base["min_index"])]))
    expected_min_index = int(base["min_index"]) - expected
    boundary = bool(
        int(base["min_index"]) in (0, config.LAG_COUNT - 1)
        or expected_min_index < 0
        or expected_min_index >= config.LAG_COUNT
    )
    if boundary:
        status = "CONTROL_UNINFORMATIVE"
    elif abs(delta - expected) <= 1 and shifted_distance > base_distance:
        status = "DELAY_DETECTED"
    else:
        status = "CONTROL_FAILED"
    return {"status": status, "expected_offset_delta": expected, "observed_offset_delta": delta, "baseline_column": int(base["min_index"]), "baseline_column_distance": base_distance, "shifted_column_distance": shifted_distance, "baseline": base, "candidate": current}


def four_cell_from_scores(q00: float, q01: float, q10: float, q11: float) -> dict[str, float]:
    from .analysis import four_cell

    return four_cell(q00, q01, q10, q11)
