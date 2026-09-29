from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import build_audio_manifest, probe_mels, read_pcm16
from .common import (
    ProtocolError,
    assert_finite,
    assert_not_sealed,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_or_verify,
)
from .roi import build_roi_manifest


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return (config.REPO / path).resolve() if not path.is_absolute() else path.resolve()


def _asset(path: Path, expected: str, label: str, *, self_hashed: bool = True) -> dict[str, Any]:
    assert_not_sealed(path)
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ProtocolError(f"{label} hash changed: {path}")
    payload: dict[str, Any] = {}
    if self_hashed:
        payload = verify_self_hashed_json(path)
    return {"path": str(path.resolve()), "sha256": actual, "payload": payload}


def _verify_spec_bindings() -> dict[str, dict[str, str]]:
    return config.spec_bindings()


def _load_arm(manifest: Mapping[str, Any], sample_id: str, arm: str) -> Mapping[str, Any]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ProtocolError("audio manifest rows are missing")
    row = next((item for item in rows if isinstance(item, Mapping) and str(item.get("sample_id")) == sample_id), None)
    if not isinstance(row, Mapping):
        raise ProtocolError(f"audio manifest row is missing: {sample_id}")
    arms = row.get("arms")
    if not isinstance(arms, list):
        raise ProtocolError(f"audio manifest arms are missing: {sample_id}")
    item = next((value for value in arms if isinstance(value, Mapping) and str(value.get("arm")) == arm), None)
    if not isinstance(item, Mapping):
        raise ProtocolError(f"audio manifest arm is missing: {sample_id}/{arm}")
    path = _resolve(item.get("output", item.get("path", "")))
    expected = str(item.get("output_sha256", item.get("container_sha256", "")))
    if not expected or not path.is_file() or file_sha256(path) != expected:
        raise ProtocolError(f"audio manifest output hash changed: {sample_id}/{arm}")
    return item


def _load_parent_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort_asset = _asset(config.COHORT, config.COHORT_SHA256, "cohort")
    timing_final_asset = _asset(config.TIMING_FINAL, config.TIMING_FINAL_SHA256, "timing-transfer final")
    bridge_final_asset = _asset(config.BRIDGE_FINAL, config.BRIDGE_FINAL_SHA256, "bridge final")
    calibration_audio_asset = _asset(config.CALIBRATION_AUDIO_MANIFEST, config.CALIBRATION_AUDIO_SHA256, "calibration audio manifest")
    bridge_audio_asset = _asset(config.BRIDGE_AUDIO_MANIFEST, config.BRIDGE_AUDIO_SHA256, "bridge audio manifest")
    tail_final_asset = _asset(config.TAIL_FINAL, config.TAIL_FINAL_SHA256, "tail final")
    timing_protocol_asset = _asset(config.TIMING_PROTOCOL, str(timing_final_asset["payload"].get("protocol_sha256", "")), "timing-transfer protocol")
    bridge_protocol_asset = _asset(config.BRIDGE_PROTOCOL, str(bridge_final_asset["payload"].get("protocol_sha256", "")), "bridge protocol")
    tail_protocol_asset = _asset(config.TAIL_PROTOCOL, str(tail_final_asset["payload"].get("protocol_sha256", "")), "tail protocol")
    timing_final = timing_final_asset["payload"]
    bridge_final = bridge_final_asset["payload"]
    timing_protocol = timing_protocol_asset["payload"]
    bridge_protocol = bridge_protocol_asset["payload"]
    tail_protocol = tail_protocol_asset["payload"]
    cohort = cohort_asset["payload"]
    calibration_audio = calibration_audio_asset["payload"]
    bridge_audio = bridge_audio_asset["payload"]
    if timing_final.get("status") != "complete" or timing_final.get("engineering_decision") != "GO":
        raise ProtocolError("timing-transfer parent is not a complete engineering run")
    if bridge_final.get("status") != "complete" or bridge_final.get("engineering_decision") != "GO":
        raise ProtocolError("bridge parent is not a complete engineering run")
    if timing_protocol.get("status") != "locked" or timing_protocol.get("protocol_id") != "lrs3_wav2lip_timing_transfer":
        raise ProtocolError("timing-transfer parent protocol is not locked")
    if bridge_protocol.get("protocol_id") != "lrs3_natural_to_tts_bridge_confirmation_20260904" or bridge_protocol.get("stage_id") != "00_protocol":
        raise ProtocolError("bridge parent protocol is not locked")
    if tail_protocol.get("status") != "locked" or tail_protocol.get("protocol_id") != "lrs3_real_video_local_timing":
        raise ProtocolError("tail parent protocol is not locked")
    if bridge_final.get("audio_manifest_sha256") != config.BRIDGE_AUDIO_SHA256 or calibration_audio.get("status") != "complete" or bridge_audio.get("status") != "complete":
        raise ProtocolError("bound audio manifests are incomplete or not bound by bridge final")
    records = cohort.get("records")
    timing_records = timing_protocol.get("records")
    tail_records = tail_protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT or not isinstance(timing_records, list) or len(timing_records) != config.EXPECTED_RECORD_COUNT or not isinstance(tail_records, list) or len(tail_records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("frozen parent cohort count is not 22")
    timing_by_id = {str(row.get("sample_id")): row for row in timing_records if isinstance(row, Mapping)}
    tail_by_id = {str(row.get("sample_id")): row for row in tail_records if isinstance(row, Mapping)}
    if len(timing_by_id) != config.EXPECTED_RECORD_COUNT or len(tail_by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent sample IDs are not unique")
    ordered_ids = [str(row.get("sample_id", "")) for row in records if isinstance(row, Mapping)]
    ordered_groups = [str(row.get("source_group", "")) for row in records if isinstance(row, Mapping)]
    if sample_ids_sha256(ordered_ids) != config.ORDERED_SAMPLE_ID_SHA256 or len(set(ordered_ids)) != config.EXPECTED_RECORD_COUNT or len(set(ordered_groups)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("frozen cohort order or source-group hash changed")
    result: list[dict[str, Any]] = []
    for source in records:
        if not isinstance(source, Mapping):
            raise ProtocolError("cohort record is malformed")
        sample_id = str(source.get("sample_id", ""))
        group = str(source.get("source_group", ""))
        timing = timing_by_id.get(sample_id)
        tail = tail_by_id.get(sample_id)
        if timing is None or tail is None or str(timing.get("source_group")) != group or str(tail.get("source_group")) != group:
            raise ProtocolError(f"parent record identity differs: {sample_id}")
        face = source.get("face_video")
        natural = source.get("natural_audio")
        mfa = source.get("mfa_linear_audio")
        tail_face = tail.get("face_video")
        tail_natural = tail.get("natural_audio")
        crop = tail.get("crop")
        track = crop.get("processed_track") if isinstance(crop, Mapping) else None
        if not isinstance(face, Mapping) or not isinstance(natural, Mapping) or not isinstance(mfa, Mapping) or not isinstance(tail_face, Mapping) or not isinstance(tail_natural, Mapping) or not isinstance(track, list):
            raise ProtocolError(f"parent assets are incomplete: {sample_id}")
        if face.get("path") != tail_face.get("path") or face.get("sha256") != tail_face.get("sha256") or natural.get("path") != tail_natural.get("path") or natural.get("sha256") != tail_natural.get("sha256"):
            raise ProtocolError(f"tail source binding differs: {sample_id}")
        face_path = _resolve(face.get("path"))
        natural_path = _resolve(natural.get("path"))
        mfa_path = _resolve(mfa.get("path"))
        for path, expected, label in ((face_path, str(face.get("sha256")), "face video"), (natural_path, str(natural.get("sha256")), "natural audio"), (mfa_path, str(mfa.get("sha256")), "MFA-linear audio")):
            if not path.is_file() or file_sha256(path) != expected:
                raise ProtocolError(f"{label} hash changed: {sample_id}")
        natural_values, natural_meta = read_pcm16(natural_path)
        mfa_values, _ = read_pcm16(mfa_path)
        if natural_values.size != int(source.get("natural_sample_count", -1)) or mfa_values.size != natural_values.size:
            raise ProtocolError(f"source audio length differs: {sample_id}")
        for item in track:
            if not isinstance(item, Mapping) or not all(key in item for key in ("x", "y", "s")) or float(item["s"]) <= 0:
                raise ProtocolError(f"tail crop track is malformed: {sample_id}")
        calibration_arms = {"N_REPEAT": _load_arm(calibration_audio, sample_id, "N_REPEAT"), "W": _load_arm(calibration_audio, sample_id, "LOCAL_WARP_120")}
        bridge_arm = _load_arm(bridge_audio, sample_id, config.AUDIO_BRIDGE)
        candidate_audio_paths = {
            config.AUDIO_N: natural_path,
            config.AUDIO_N_REPEAT: _resolve(calibration_arms["N_REPEAT"].get("output")),
            config.AUDIO_W: _resolve(calibration_arms["W"].get("output")),
            config.AUDIO_BRIDGE: _resolve(bridge_arm.get("output")),
        }
        result.append({
            "sample_id": sample_id,
            "source_group": group,
            "transcript": str(source.get("transcript", "")),
            "face_video": {"path": str(face_path), "sha256": str(face.get("sha256"))},
            "natural_audio": {"path": str(natural_path), "sha256": str(natural.get("sha256")), "sample_count": int(natural_values.size), "decoded_pcm_sha256": natural_meta["decoded_pcm_sha256"]},
            "mfa_linear_audio": {"path": str(mfa_path), "sha256": str(mfa.get("sha256")), "sample_count": int(mfa_values.size)},
            "natural_sample_count": int(natural_values.size),
            "candidate_audio_paths": {key: str(value) for key, value in candidate_audio_paths.items()},
            "source_video_timeline": dict(tail.get("source_video_timeline", timing.get("real_video", {}).get("timeline", {}))),
            "processed_track": [{"x": float(item["x"]), "y": float(item["y"]), "s": float(item["s"])} for item in track],
            "track_sha256": canonical_json_sha256([{"x": float(item["x"]), "y": float(item["y"]), "s": float(item["s"])} for item in track]),
            "parent_record_hashes": {"timing_protocol": canonical_json_sha256(dict(timing)), "tail_protocol": canonical_json_sha256(dict(tail))},
        })
    return result, {
        "cohort": cohort_asset,
        "timing_final": timing_final_asset,
        "timing_protocol": timing_protocol_asset,
        "bridge_final": bridge_final_asset,
        "bridge_protocol": bridge_protocol_asset,
        "calibration_audio": calibration_audio_asset,
        "bridge_audio": bridge_audio_asset,
        "tail_final": tail_final_asset,
        "tail_protocol": tail_protocol_asset,
    }


def audio_forward_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * math.pi * 1920.0:
        raise ProtocolError("audio is too short for LOCAL_WARP_120")
    n = np.arange(sample_count, dtype=np.float64)
    mapped = n + 1920.0 * np.sin(2.0 * np.pi * n / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise ProtocolError("LOCAL_WARP_120 map is not monotone")
    return mapped


def timing_masks(sample_count: int, frame_counts: Mapping[str, int]) -> dict[str, Any]:
    mapped = audio_forward_map(sample_count)
    n = np.arange(sample_count, dtype=np.float64)
    q = min(int(frame_counts[config.VIDEO_R]), int(frame_counts[config.VIDEO_GN]), int(frame_counts[config.VIDEO_GNR]), int(frame_counts[config.VIDEO_GW]), int(frame_counts[config.VIDEO_GB]))
    q = min(q, sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, n, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, n))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise ProtocolError(f"timing masks have fewer than {config.MIN_LOCAL_ROWS} rows")
    return {
        "sample_count": int(sample_count),
        "q_frames": int(q),
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "forward_mapping_dtype": "float64-little-endian",
        "forward_mapping_length": int(mapped.size),
    }


def _runtime_bindings() -> dict[str, Any]:
    paths = {
        "wav2lip_python": config.WAV2LIP_PYTHON,
        "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT,
        "wav2lip_inference": config.WAV2LIP_ROOT / "inference.py",
        "wav2lip_models": config.WAV2LIP_ROOT / "models/wav2lip.py",
        "wav2lip_audio": config.WAV2LIP_ROOT / "audio.py",
        "face_detection_api": config.WAV2LIP_ROOT / "face_detection/api.py",
        "face_detection_sfd": config.WAV2LIP_ROOT / "face_detection/detection/sfd/sfd_detector.py",
        "face_detection_weights": config.WAV2LIP_ROOT / "face_detection/detection/sfd/s3fd.pth",
        "syncnet_python": config.SYNCNET_PYTHON,
        "syncnet_model": config.SYNCNET_MODEL,
        "syncnet_worker": config.SYNCNET_WORKER,
        "syncnet_instance": config.SYNCNET_ROOT / "SyncNetInstance.py",
        "syncnet_model_code": config.SYNCNET_ROOT / "SyncNetModel.py",
        "ffmpeg": config.FFMPEG,
        "ffprobe": config.FFPROBE,
        "roi_worker": Path(__file__).with_name("roi_worker.py"),
        "generation_worker": Path(__file__).with_name("generation_worker.py"),
        "mel_probe": Path(__file__).with_name("mel_probe.py"),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ProtocolError(f"runtime binding is missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if result["wav2lip_checkpoint"]["sha256"] != config.WAV2LIP_CHECKPOINT_SHA256 or result["syncnet_model"]["sha256"] != config.SYNCNET_MODEL_SHA256:
        raise ProtocolError("registered model hash differs")
    return result


def _frame_predictions(records: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    by_id = {str(row.get("sample_id")): row for row in diagnostics.get("rows", []) if isinstance(row, Mapping)}
    if len(by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("mel diagnostics are incomplete")
    result: dict[str, dict[str, int]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        diag = by_id.get(sample_id)
        if diag is None:
            raise ProtocolError(f"mel diagnostic row is missing: {sample_id}")
        arms = diag.get("arms")
        if not isinstance(arms, Mapping):
            raise ProtocolError(f"mel diagnostic arms are missing: {sample_id}")
        counts = {arm: int(arms[arm]["mel_chunk_count"]) for arm in config.AUDIO_ARMS}
        source_count = int(record["source_video_timeline"]["frame_count"])
        if len(set(counts.values())) != 1 or next(iter(counts.values())) <= 0 or next(iter(counts.values())) > source_count:
            raise ProtocolError(f"predicted frame counts differ or exceed source: {sample_id}")
        result[sample_id] = {"R": source_count, "G_N": counts[config.AUDIO_N], "G_NR": counts[config.AUDIO_N_REPEAT], "G_W": counts[config.AUDIO_W], "G_B": counts[config.AUDIO_BRIDGE]}
    return result


def _input_audit(paths: config.RunPaths, records: Sequence[Mapping[str, Any]], parent_assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], roi_manifest: Mapping[str, Any], review: Mapping[str, Any], diagnostics: Mapping[str, Any], frame_counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    roi_rows = {str(row.get("sample_id")): row for row in roi_manifest.get("rows", []) if isinstance(row, Mapping)}
    review_rows = {str(row.get("sample_id")): row for row in review.get("rows", []) if isinstance(row, Mapping)}
    audio_rows = {str(row.get("sample_id")): row for row in audio_manifest.get("rows", []) if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        checks = {
            "parent_assets": "PASS",
            "audio_identity": "PASS" if sample_id in audio_rows else "FAIL",
            "roi_identity": "PASS" if sample_id in roi_rows else "FAIL",
            "roi_review": "PASS" if review_rows.get(sample_id, {}).get("decision") == "PASS" else "FAIL",
            "mel_prediction": "PASS" if sample_id in frame_counts else "FAIL",
        }
        errors = [name for name, value in checks.items() if value != "PASS"]
        rows.append({"index": index, "sample_id": sample_id, "source_group": str(record["source_group"]), "checks": checks, "passed": not errors, "status": "PASS" if not errors else "BLOCKED", "errors": errors})
    payload = {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if all(row["passed"] for row in rows) else "blocked",
        "classification": "seen_fit_pilot",
        "record_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in records}),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "blocked_count": sum(not bool(row["passed"]) for row in rows),
        "ordered_sample_id_sha256": config.ORDERED_SAMPLE_ID_SHA256,
        "frozen_parents": {name: {key: value for key, value in item.items() if key != "payload"} for name, item in parent_assets.items()},
        "frozen_manifests": {"audio": {"path": str(paths.audio_manifest.resolve()), "sha256": file_sha256(paths.audio_manifest)}, "roi": {"path": str(paths.roi_manifest.resolve()), "sha256": file_sha256(paths.roi_manifest)}, "roi_review": {"path": str(paths.roi_review.resolve()), "sha256": file_sha256(paths.roi_review)}, "mel_diagnostics": {"path": str((paths.root / "audio_probe/mel_diagnostics.json").resolve()), "sha256": file_sha256(paths.root / "audio_probe/mel_diagnostics.json")}},
        "records": rows,
        "sealed_scope": {"seen_fit": True, "tts_generation": False, "training": False, "mfa": False, "dtw": False, "heldout": False, "history_overwrite": False},
    }
    return payload


def prepare(paths: config.RunPaths) -> dict[str, Any]:
    if paths.final.is_file():
        raise ProtocolError("terminal run cannot be prepared again")
    records, parents = _load_parent_sources()
    diagnostics = probe_mels(records, paths.root / "audio_probe")
    audio_manifest = build_audio_manifest(records, paths.audio_manifest, diagnostics)
    audio_manifest = write_or_verify(paths.audio_manifest, {key: value for key, value in audio_manifest.items() if key != "artifact_sha256"})
    roi_manifest, review = build_roi_manifest(records, paths.roi_manifest, paths.roi_review, paths.root / "roi" / "logs")
    frame_counts = _frame_predictions(records, diagnostics)
    if review.get("status") != "complete":
        raise ProtocolError("ROI geometry review is blocked")
    audit_payload = _input_audit(paths, records, parents, audio_manifest, roi_manifest, review, diagnostics, frame_counts)
    audit = write_or_verify(paths.input_audit, audit_payload)
    if audit.get("status") != "complete":
        raise ProtocolError("input audit is blocked")
    roi_by_id = {str(row["sample_id"]): row for row in roi_manifest["rows"]}
    audio_by_id = {str(row["sample_id"]): row for row in audio_manifest["rows"]}
    protocol_records: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        counts = frame_counts[sample_id]
        masks = timing_masks(int(record["natural_sample_count"]), counts)
        roi_row = roi_by_id[sample_id]
        audio_row = audio_by_id[sample_id]
        protocol_records.append({
            **{key: value for key, value in record.items() if key not in ("candidate_audio_paths", "processed_track")},
            "processed_track": record["processed_track"],
            "track_sha256": record["track_sha256"],
            "roi": {"boxes_path": roi_row["boxes_path"], "boxes_sha256": roi_row["boxes_sha256"], "box_track_sha256": roi_row["box_track_sha256"], "frame_count": roi_row["frame_count"], "width": roi_row["width"], "height": roi_row["height"], "detector": roi_row["detector"]},
            "audio": {"manifest_row": audio_row, "manifest_sha256": file_sha256(paths.audio_manifest)},
            "predicted_frame_counts": counts,
            "masks": masks,
        })
    runtime = _runtime_bindings()
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "locked",
        "classification": "seen_fit_pilot",
        "experiment": "validate-wav2lip-face-roi-replacement",
        "cohort": {"path": str(config.COHORT.resolve()), "sha256": config.COHORT_SHA256, "record_count": config.EXPECTED_RECORD_COUNT, "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT, "ordered_sample_id_sha256": config.ORDERED_SAMPLE_ID_SHA256},
        "parent_runs": {name: {key: value for key, value in item.items() if key != "payload"} for name, item in parents.items()},
        "spec_bindings": _verify_spec_bindings(),
        "input_audit": {"path": str(paths.input_audit.resolve()), "sha256": file_sha256(paths.input_audit), "artifact_sha256": audit["artifact_sha256"]},
        "audio_manifest": {"path": str(paths.audio_manifest.resolve()), "sha256": file_sha256(paths.audio_manifest)},
        "roi_manifest": {"path": str(paths.roi_manifest.resolve()), "sha256": file_sha256(paths.roi_manifest)},
        "roi_review": {"path": str(paths.roi_review.resolve()), "sha256": file_sha256(paths.roi_review)},
        "mel_diagnostics": {"path": str((paths.root / "audio_probe/mel_diagnostics.json").resolve()), "sha256": file_sha256(paths.root / "audio_probe/mel_diagnostics.json")},
        "runtime_bindings": runtime,
        "config": config.FrozenConfig().to_dict(),
        "stages": {"prepare": {"roi_review_required": True}, "control": {"new_video_count": config.EXPECTED_CONTROL_VIDEO_COUNT, "new_score_count": config.EXPECTED_CONTROL_SCORE_COUNT}, "bridge": {"new_video_count": config.EXPECTED_BRIDGE_VIDEO_COUNT, "new_score_count": config.EXPECTED_BRIDGE_CELL_COUNT}},
        "records": protocol_records,
        "sealed_scope": {"seen_fit": True, "tts_generation": False, "training": False, "mfa": False, "dtw": False, "heldout": False, "history_overwrite": False, "training_authorized": False, "reference_conditioned_audio_head_spec_eligible": False, "generalization_established": False},
    }
    assert_finite(payload)
    return write_or_verify(paths.protocol, payload)


def load_protocol(paths: config.RunPaths) -> dict[str, Any]:
    protocol = verify_self_hashed_json(paths.protocol)
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("protocol_revision") != config.PROTOCOL_REVISION or protocol.get("status") != "locked":
        raise ProtocolError("protocol marker is invalid")
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol records are incomplete")
    if file_sha256(paths.input_audit) != str(protocol["input_audit"]["sha256"]) or file_sha256(paths.audio_manifest) != str(protocol["audio_manifest"]["sha256"]) or file_sha256(paths.roi_manifest) != str(protocol["roi_manifest"]["sha256"]) or file_sha256(paths.roi_review) != str(protocol["roi_review"]["sha256"]):
        raise ProtocolError("prepare artifact hash changed")
    if verify_self_hashed_json(paths.roi_review).get("status") != "complete":
        raise ProtocolError("ROI review is not complete")
    result = dict(protocol)
    result["_path"] = str(paths.protocol.resolve())
    result["_sha256"] = file_sha256(paths.protocol)
    return result
