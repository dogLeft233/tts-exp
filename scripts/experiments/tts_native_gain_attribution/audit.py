"""P0 input, provenance, and resource audit for the native-gain study."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    decode_media_pcm16,
    executable_path,
    ffprobe_json,
    file_sha256,
    read_json,
    read_self_hashed_json,
    resource_plan,
    run_command,
    sample_ids_hash,
    write_self_hashed_json,
)


def _path(value: str | Path) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def _json_bytes_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _manifest_records() -> dict[tuple[str, str], Mapping[str, Any]]:
    manifest = read_json(config.REPO / "data/dataset_samples/video_manifest_250.json")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("records"), list):
        raise ProtocolError("original video manifest is malformed")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in manifest["records"]:
        if not isinstance(raw, Mapping) or raw.get("dataset") != "lrs3":
            continue
        stem = str(raw.get("stem", ""))
        path = str(raw.get("video_local_path", ""))
        if not stem or not path:
            raise ProtocolError("LRS3 manifest record has no stem/path")
        result[(stem, path)] = raw
    return result


def _text_for_video(video: Path) -> tuple[str | None, str | None, str]:
    text_path = video.with_suffix(".txt")
    if not text_path.is_file():
        return None, None, "TRANSCRIPT_UNAVAILABLE"
    first_line = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = None
    for line in first_line:
        if line.startswith("Text:"):
            text = re.sub(r"\s+", " ", line.split(":", 1)[1]).strip()
            break
    if not text:
        return None, file_sha256(text_path), "TRANSCRIPT_UNVERIFIED"
    return text, file_sha256(text_path), "TRANSCRIPT_VERIFIED"


def _stream_summary(path: Path) -> dict[str, Any]:
    probe = ffprobe_json(path)
    streams = probe.get("streams", [])
    if not isinstance(streams, list):
        raise ProtocolError(f"ffprobe streams are malformed: {path}")
    result: dict[str, Any] = {"file_sha256": file_sha256(path), "bytes": path.stat().st_size, "streams": []}
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        fields = {
            key: stream.get(key)
            for key in (
                "index",
                "codec_type",
                "codec_name",
                "profile",
                "width",
                "height",
                "r_frame_rate",
                "avg_frame_rate",
                "time_base",
                "start_time",
                "duration",
                "nb_frames",
                "nb_read_frames",
                "sample_rate",
                "channels",
                "channel_layout",
            )
            if key in stream
        }
        if stream.get("codec_type") == "video":
            pts_result = run_command(
                (
                    str(executable_path(config.FFPROBE, "ffprobe")),
                    "-v",
                    "error",
                    "-select_streams",
                    f"{stream.get('index', 0)}",
                    "-read_intervals",
                    "%+#1",
                    "-show_entries",
                    "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time",
                    "-of",
                    "json",
                    str(path),
                )
            )
            try:
                pts_payload = json.loads(pts_result.stdout.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"ffprobe PTS is malformed: {path}") from exc
            frames = pts_payload.get("frames", []) if isinstance(pts_payload, Mapping) else []
            fields["first_frame_pts"] = frames[0] if frames else None
        result["streams"].append(fields)
    return result


def _video_frame_count(summary: Mapping[str, Any]) -> int | None:
    for stream in summary.get("streams", []):
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video":
            for key in ("nb_read_frames", "nb_frames"):
                value = stream.get(key)
                try:
                    if value is not None:
                        return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _first_frame(path: Path, target: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("opencv is required to freeze the portrait") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ProtocolError(f"cannot decode first frame: {path}")
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ProtocolError(f"cannot encode first frame: {path}")
    payload = bytes(encoded)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return {
        "path": str(target),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": hashlib.sha256(np.asarray(frame).tobytes()).hexdigest(),
        "shape": [int(item) for item in frame.shape],
        "encoding": "OpenCV imencode PNG from decoded frame 0",
    }


def _verify_input_bindings(bindings: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    files = bindings.get("files")
    if not isinstance(files, list):
        raise ProtocolError("input-bindings.files is missing")
    for index, raw in enumerate(files):
        if not isinstance(raw, Mapping):
            failures.append({"index": index, "error_type": "BINDING_MALFORMED", "reason": "file row is not an object"})
            continue
        path_value = raw.get("path")
        expected = raw.get("sha256")
        target = _path(str(path_value)) if isinstance(path_value, str) else Path("<missing>")
        item = {"index": index, "path": str(target), "role": raw.get("role"), "expected_sha256": expected, "expected_bytes": raw.get("bytes")}
        try:
            actual = file_sha256(target)
            if not isinstance(expected, str) or actual != expected:
                raise ProtocolError(f"sha256 {actual} != {expected}")
            if raw.get("bytes") is not None and int(raw["bytes"]) != target.stat().st_size:
                raise ProtocolError(f"bytes {target.stat().st_size} != {raw['bytes']}")
            item.update({"actual_sha256": actual, "actual_bytes": target.stat().st_size, "status": "PASS"})
            verified.append(item)
        except (OSError, ProtocolError, TypeError, ValueError) as exc:
            item.update({"status": "FAIL", "error_type": "BINDING_MISMATCH", "reason": str(exc)})
            failures.append(item)
    return verified, {"expected_count": len(files), "verified_count": len(verified), "failure_count": len(failures), "failures": failures}


def _parent_audit(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if str(bindings.get("parent_run")) != str(config.PARENT_RUN.relative_to(config.REPO)):
        raise ProtocolError("input binding points to an unexpected parent run")
    final = read_self_hashed_json(config.PARENT_FINAL)
    validation = read_self_hashed_json(config.PARENT_VALIDATION)
    cohort = read_self_hashed_json(config.PARENT_COHORT)
    parent_inputs = read_self_hashed_json(config.PARENT_INPUTS)
    if final.get("status") != "complete" or final.get("automatic_complete") is not True:
        raise ProtocolError("v15 final is not a completed automatic run")
    if validation.get("status") != "valid":
        raise ProtocolError("v15 independent validation is not valid")
    if cohort.get("status") != "complete" or int(cohort.get("source_group_count", -1)) != 45:
        raise ProtocolError("v15 cohort is not complete")
    if parent_inputs.get("status") != "complete":
        raise ProtocolError("v15 inputs audit is not complete")
    return {
        "run": str(config.PARENT_RUN),
        "final_sha256": file_sha256(config.PARENT_FINAL),
        "validation_sha256": file_sha256(config.PARENT_VALIDATION),
        "cohort_sha256": file_sha256(config.PARENT_COHORT),
        "inputs_sha256": file_sha256(config.PARENT_INPUTS),
        "final_status": final.get("status"),
        "validation_status": validation.get("status"),
        "record_count": int(cohort.get("record_count", -1)),
        "source_group_count": int(cohort.get("source_group_count", -1)),
        "historical_v13_v14_excluded": True,
    }


def _condition_assets(record: Mapping[str, Any], source: str) -> dict[str, Any]:
    condition_name = "natural_raw" if source == "N" else "tts_raw"
    condition = record.get("conditions", {}).get(condition_name)
    if not isinstance(condition, Mapping):
        raise ProtocolError(f"missing historical condition {record.get('sample_id')}/{condition_name}")
    required = ("media", "crop", "matrix", "crop_selection", "worker_result", "original_pcm_sha256", "matrix_shape")
    paths: dict[str, Any] = {}
    for key in required[:-2]:
        value = condition.get(key)
        if not isinstance(value, str):
            raise ProtocolError(f"missing {key} for {record.get('sample_id')}/{condition_name}")
        target = _path(value)
        if not target.is_file():
            raise ProtocolError(f"missing historical {key}: {target}")
        paths[key] = {"path": str(target), "sha256": file_sha256(target)}
    paths["original_pcm_sha256"] = str(condition["original_pcm_sha256"])
    paths["matrix_shape"] = [int(item) for item in condition["matrix_shape"]]
    worker = read_json(paths["worker_result"]["path"])
    if not isinstance(worker, Mapping):
        raise ProtocolError(f"worker result is malformed: {record.get('sample_id')}/{source}")
    if str(worker.get("input_pcm_sha256")) != paths["original_pcm_sha256"] or str(worker.get("extracted_pcm_sha256")) != paths["original_pcm_sha256"]:
        raise ProtocolError(f"worker PCM identity differs: {record.get('sample_id')}/{source}")
    matrix = np.load(paths["matrix"]["path"], allow_pickle=False)
    if matrix.shape != tuple(paths["matrix_shape"]) or matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or not np.isfinite(matrix).all():
        raise ProtocolError(f"historical matrix shape/content differs: {record.get('sample_id')}/{source}")
    crop_pcm = decode_media_pcm16(_path(condition["crop"]))
    crop_pcm_hash = hashlib.sha256(np.asarray(crop_pcm, dtype="<i2").tobytes()).hexdigest()
    if crop_pcm_hash != paths["original_pcm_sha256"]:
        raise ProtocolError(f"crop PCM identity differs: {record.get('sample_id')}/{source}: {crop_pcm_hash}")
    selection = read_self_hashed_json(paths["crop_selection"]["path"])
    if not isinstance(selection.get("official_pipeline_config"), Mapping):
        raise ProtocolError(f"crop selection lacks official config: {record.get('sample_id')}/{source}")
    return {
        "source": source,
        "condition": condition_name,
        "media": paths["media"],
        # The historical worker's baseline is the PCM carried by the frozen
        # crop, not the lossy audio track in the rendered MP4.
        "audio_media": paths["crop"],
        "crop": paths["crop"],
        "matrix": paths["matrix"],
        "crop_selection": paths["crop_selection"],
        "worker_result": paths["worker_result"],
        "original_pcm_sha256": paths["original_pcm_sha256"],
        "matrix_shape": paths["matrix_shape"],
        "crop_pcm_sample_count": int(crop_pcm.size),
        "worker_input_pcm_sha256": str(worker.get("input_pcm_sha256")),
        "selection_rule": selection.get("selection_rule"),
        "official_pipeline_config": dict(selection["official_pipeline_config"]),
        "video_probe": _stream_summary(_path(condition["crop"])),
        "media_probe": _stream_summary(_path(condition["media"])),
    }


def _leaptalk_provenance() -> dict[str, Any]:
    candidates = []
    for root in config.leaptalk_candidates():
        candidates.append({"root": str(root), "exists": root.is_dir()})
    explicit = os.environ.get("LEAPTALK_ROOT")
    command = os.environ.get("LEAPTALK_COMMAND")
    return {
        "family": "LeapTalk",
        "status": "DEPENDENCY_BLOCKED" if not explicit or not command else "CONFIGURATION_REQUIRES_VERIFICATION",
        "identity": "LeapTalk family is required; no Wav2Lip substitution is permitted",
        "historical_checkpoint": "UNKNOWN; the historical closed remote deployment is not locally reproducible",
        "repo_commit": "UNKNOWN",
        "base_weights": "UNKNOWN",
        "lora_weights": "UNKNOWN",
        "audio_encoder_weights": "UNKNOWN",
        "vae_weights": "UNKNOWN",
        "tae_weights": "UNKNOWN",
        "mode": "UNKNOWN",
        "sampling": "UNKNOWN",
        "dtype": "UNKNOWN",
        "training_distribution": "TRAINING_DISTRIBUTION_UNKNOWN",
        "software_device": "UNKNOWN until an explicit adapter is bound",
        "candidate_roots": candidates,
        "explicit_root_set": bool(explicit),
        "explicit_command_set": bool(command),
        "no_wav2lip_substitution": True,
    }


def audit_stage(paths: config.RunPaths, *, run_id: str) -> dict[str, Any]:
    """Create the immutable P0 ledger before any new model inference."""

    paths.audit.mkdir(parents=True, exist_ok=True)
    existing = paths.audit / "assets.json"
    if existing.is_file():
        try:
            cached = read_self_hashed_json(existing)
            if cached.get("status") in {"COMPLETE", "INCOMPLETE"} and cached.get("protocol_id") == config.PROTOCOL_ID:
                return cached
        except ProtocolError:
            pass

    bindings = read_json(config.INPUT_BINDINGS)
    if not isinstance(bindings, Mapping):
        raise ProtocolError("input-bindings is not an object")
    if bindings.get("change_id") != "disentangle-tts-native-gain" or list(bindings.get("sample_ids", [])) != list(config.SAMPLE_IDS):
        raise ProtocolError("input-bindings does not match the frozen cohort")
    verified_files, file_audit = _verify_input_bindings(bindings)
    parent = _parent_audit(bindings)
    original = _manifest_records()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = list(file_audit["failures"])
    portraits = paths.audit / "portraits"
    for raw in bindings.get("records", []):
        sample_id = int(raw["sample_id"])
        group = str(raw["source_group"])
        real_video = _path(str(raw["real_video"]))
        record: dict[str, Any] = {"sample_id": sample_id, "source_group": group, "real_video": str(real_video), "sources": {}, "status": "COMPLETE"}
        try:
            if not real_video.is_file():
                raise ProtocolError(f"real video is missing: {real_video}")
            stem = f"{group}_{real_video.stem}"
            manifest_match = original.get((stem, str(raw["real_video"])))
            if not isinstance(manifest_match, Mapping) or str(manifest_match.get("video_local_path")) != str(raw["real_video"]):
                raise ProtocolError(f"original manifest mapping differs for {sample_id}")
            if Path(str(raw["real_video"])).parent.name != group:
                raise ProtocolError(f"real video parent is not source_group for {sample_id}")
            text, text_hash, text_status = _text_for_video(real_video)
            real_probe = _stream_summary(real_video)
            portrait = _first_frame(real_video, portraits / f"{sample_id}.png")
            record.update({
                "manifest": {"stem": stem, "speaker_key": manifest_match.get("speaker_key"), "dataset": manifest_match.get("dataset"), "audio_source": manifest_match.get("audio_source")},
                "transcript": {"text": text, "text_sha256": text_hash, "status": text_status},
                "real_video_sha256": file_sha256(real_video),
                "real_video_probe": real_probe,
                "real_video_frame_count": _video_frame_count(real_probe),
                "real_audio_video_pts_status": "RECORDED_FROM_SINGLE_CONTAINER",
                "portrait": portrait,
                "sources": {
                    "R": {
                        "source": "R",
                        "media": {"path": str(real_video), "sha256": file_sha256(real_video)},
                        "audio_media": {"path": str(real_video), "sha256": file_sha256(real_video)},
                        "real_video": {"path": str(real_video), "sha256": file_sha256(real_video)},
                        "expected_pcm_sha256": None,
                    },
                    "N": _condition_assets(raw, "N"),
                    "T": _condition_assets(raw, "T"),
                },
            })
        except (OSError, ProtocolError, TypeError, ValueError) as exc:
            record.update({"status": "INPUT_INVALID", "error_type": "INPUT_INVALID", "reason": str(exc)})
            failures.append({"sample_id": sample_id, "source_group": group, "error_type": "INPUT_INVALID", "reason": str(exc)})
        records.append(record)

    if len(records) != len(config.SAMPLE_IDS):
        failures.append({"error_type": "COHORT_COUNT", "reason": f"records {len(records)} != {len(config.SAMPLE_IDS)}"})
    statuses = {str(row.get("status")) for row in records}
    status = "COMPLETE" if not failures and statuses == {"COMPLETE"} else "INCOMPLETE"
    if status == "INCOMPLETE":
        # Do not let a partially audited cohort be consumed as a complete run.
        for row in records:
            if row.get("status") == "COMPLETE" and not row.get("sources"):
                row["status"] = "INPUT_INVALID"

    assets = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "00_audit",
        "run_id": run_id,
        "status": status,
        "sample_ids": list(config.SAMPLE_IDS),
        "source_groups": [str(row["source_group"]) for row in records],
        "record_count": len(records),
        "expected_record_count": len(config.SAMPLE_IDS),
        "records": records,
        "input_bindings": {"path": str(config.INPUT_BINDINGS), "sha256": file_sha256(config.INPUT_BINDINGS), "file_audit": file_audit},
        "parent_evidence": parent,
        "failures": failures,
        "verified_file_count": len(verified_files),
    }
    write_self_hashed_json(paths.audit / "assets.json", assets)

    code_hashes = {str(path.relative_to(config.REPO)): file_sha256(path) for path in config.package_files() if path.is_file()}
    protocol = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "run_id": run_id,
        "status": status,
        "frozen_documents": {str(path.relative_to(config.REPO)): file_sha256(path) for path in (config.PROPOSAL, config.DESIGN, config.PROTOCOL, config.SPEC, config.TASKS, config.INPUT_BINDINGS)},
        "frozen_code": code_hashes,
        "cohort": {"sample_ids": list(config.SAMPLE_IDS), "sample_ids_sha256": sample_ids_hash(config.SAMPLE_IDS), "source_group_count": len(records)},
        "parameters": {
            "sample_rate": config.SAMPLE_RATE,
            "fps": config.FPS,
            "n_fft": config.NFFT,
            "hop": config.HOP,
            "vshift": config.VSHIFT,
            "conditions": list(config.AUDIO_CONDITIONS),
            "seeds": list(config.SEEDS),
            "bootstrap_seed": config.BOOTSTRAP_SEED,
            "bootstrap_draws": config.BOOTSTRAP_DRAWS,
            "min_interior_rows": config.MIN_INTERIOR_ROWS,
        },
        "support_rule": "common INTERIOR row intersection; remove vshift=15 at each edge; minimum 25",
        "zero_new_tts": True,
        "zero_new_cloud_calls": True,
        "zero_training": True,
        "failures": failures,
    }
    write_self_hashed_json(paths.protocol, protocol)
    write_self_hashed_json(paths.inputs, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "assets": str(paths.audit / "assets.json"), "assets_sha256": file_sha256(paths.audit / "assets.json"), "input_bindings_sha256": file_sha256(config.INPUT_BINDINGS), "status": status, "records": records})
    write_self_hashed_json(paths.provenance, _leaptalk_provenance() | {"schema_version": 1, "protocol_id": config.PROTOCOL_ID})
    claims = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "claims": [
            {"id": "H1", "status": "UNTESTED", "boundary": "fixed-video evaluation path and geometry diagnostics"},
            {"id": "H2", "status": "UNTESTED", "boundary": "two registered acoustic interventions; not a general quality claim"},
            {"id": "H3", "status": "DESCRIPTIVE_ONLY", "boundary": "prespecified audio rhythm summaries; no causal intervention"},
            {"id": "H4", "status": "NOT_CAUSALLY_IDENTIFIED", "boundary": "training distribution/checkpoint selection evidence incomplete"},
            {"id": "PERCEPTION", "status": "NOT_ASSESSED", "boundary": "packages are built; no human scores are inferred"},
        ],
        "training_distribution": "TRAINING_DISTRIBUTION_UNKNOWN",
        "historical_vs_fresh": "v15 is historical evidence; B must use a fresh provenance-bound LeapTalk baseline",
        "status": status,
    }
    write_self_hashed_json(paths.claims, claims)
    write_self_hashed_json(paths.audit / "resource_plan.json", resource_plan() | {"estimated_persistent_bytes": config.PERSISTENT_BUDGET_BYTES, "estimated_single_cell_temp_bytes": config.CELL_TEMP_BUDGET_BYTES, "disk_policy": "no deletion of historical or unrelated data"})
    return read_self_hashed_json(paths.audit / "assets.json")


def refresh_code_snapshot(paths: config.RunPaths) -> dict[str, Any]:
    """Freeze the reviewed implementation immediately before video scoring."""

    protocol = read_self_hashed_json(paths.protocol)
    code_hashes = {str(path.relative_to(config.REPO)): file_sha256(path) for path in config.package_files() if path.is_file()}
    document_paths = [config.PROPOSAL, config.DESIGN, config.PROTOCOL, config.SPEC, config.TASKS, config.INPUT_BINDINGS]
    continuation_spec = protocol.get("continuation_spec")
    if isinstance(continuation_spec, str):
        continuation_root = Path(continuation_spec)
        document_paths.extend(
            continuation_root / relative
            for relative in (
                "README.md",
                "design.md",
                "tasks.md",
                "evidence-bindings.json",
                "specs/tts-native-gain-completion/spec.md",
            )
        )
    document_hashes = {str(path.relative_to(config.REPO)): file_sha256(path) for path in document_paths if path.is_file()}
    runtime_hashes = {}
    for path in config.runtime_files():
        if not path.is_file():
            continue
        try:
            key = str(path.relative_to(config.REPO))
        except ValueError:
            key = str(path.resolve())
        runtime_hashes[key] = file_sha256(path)
    return write_self_hashed_json(
        paths.protocol,
        {
            **protocol,
            "frozen_documents": document_hashes,
            "frozen_code": code_hashes,
            "frozen_runtime": runtime_hashes,
            "code_snapshot_refrozen_before_scoring": True,
        },
    )
