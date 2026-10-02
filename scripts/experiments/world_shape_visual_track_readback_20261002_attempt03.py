#!/usr/bin/env python3
"""Independent metadata/clock/hash readback for the sealed shape tracking run.

This checker never loads mouth-geometry arrays. It hashes feature NPZ files as
opaque bytes, reads only timestamps_s, valid, and metadata_json from each
feature archive, and reads the timestamp/valid arrays from availability NPZs.
It also records the known launcher accounting failure caused by the producer's
40 progress lines plus final result line; it does not rewrite the original
worker's exit code or any tracking outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_EXTRACTOR_SHA256 = "aa86f4011e4a93e345471142ce29d6fcb739d1d4d89254f95f7685d2bf7fb41f"
EXPECTED_HELPER_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
EXPECTED_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
EXPECTED_CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
EXPECTED_TONE_SHA256 = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
EXPECTED_PLAN_RAW_SHA256 = "13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f"
EXPECTED_PLAN_PAYLOAD_SHA256 = "26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e"
EXPECTED_RELEASE_RAW_SHA256 = "09ca0731613c59fc9cbbd4082fd74133e8bbc84079bc7fe5a27de5db73df22e1"
EXPECTED_RELEASE_PAYLOAD_SHA256 = "c4c37a54ba0b6a5d6ac38e89a8abec1a903d9b5cc438a9f4884bfc6d8b41bf5c"
EXPECTED_PREFLIGHT_RAW_SHA256 = "df0fec4270f5af9c753771e6c3a618f027ebcb2596c838f8463a4231095c47f6"
EXPECTED_PREFLIGHT_STATUS_RAW_SHA256 = "c19d2f8425911292b6a6bd9f6ad2125e9347b82233263e561ccac7b1f5f8f589"
EXPECTED_STAGE_RECEIPT_RAW_SHA256 = "3079f1c50cb8349061bbf48809f0ec16a16a9cba805e6de317a5cbb7ef192388"
EXPECTED_PREFLIGHT_LAUNCH_RAW_SHA256 = "a9a4e668477942e38ed71711d934b0286ff35aa298da48024566d20feaf0a14c"
EXPECTED_READBACK_RECEIPT_RAW_SHA256 = "34bc294c17d3f258a3816709c5fb82a0175517201b08810237817edfd421e88a"
EXPECTED_PREFLIGHT_PID = 30275
EXPECTED_PREFLIGHT_START_TICKS = "751001169"
EXPECTED_EXTRACTION_PID = 40813
EXPECTED_EXTRACTION_START_TICKS = "751203604"
EXPECTED_EXTRACTION_LAUNCH_RAW_SHA256 = "0f1d484ef61a5b537e557bff503bb3b1e77f236a8f848fad497047a80940ae77"
EXPECTED_EXTRACTION_STDOUT_RAW_SHA256 = "b538654cc886b17ab297e7dbda8ca4b683ff3b1ebadd118069b23575b372c865"
EXPECTED_EXTRACTION_STDERR_RAW_SHA256 = "8c69e56cc14e1fccc8eb31098accfdcd71bfe2ff8f44ae761def92d93c61ce5b"
EXPECTED_EXTRACTION_LAUNCH_STDERR_RAW_SHA256 = "73cfff249131bcd1df0b43a0227c89e2174433beb8e8d08019d080b3901c2896"
EXPECTED_EXTRACTION_LAUNCH_STDOUT_RAW_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
EXPECTED_EXTRACTION_EXITCODE_RAW_SHA256 = "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
EXPECTED_ROOT_RELEASE_STATUS = "ROOT_RELEASED"
EXPECTED_CONTEXTS = 160
EXPECTED_VIDEOS = 800
EXPECTED_FPS = 25.0
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
FEATURE_MEMBERS = {
    "landmarks.npy", "canonical_mouth.npy", "valid.npy", "mouth_open.npy",
    "mouth_width.npy", "mouth_height.npy", "mouth_area.npy", "timestamps_s.npy",
    "metadata_json.npy",
}
ARRAY_NAMES = {
    "timestamps_s", "valid", "mouth_open", "mouth_width", "mouth_height",
    "mouth_area", "landmarks", "canonical_mouth",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ReadbackError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReadbackError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ReadbackError(f"json_not_object:{path.name}")
    return value


def verify_fingerprint(value: dict[str, Any], field: str) -> str:
    stored = value.get(field)
    actual = sha_bytes(canonical_bytes({key: item for key, item in value.items() if key != field}))
    if stored != actual:
        raise ReadbackError(f"embedded_fingerprint_mismatch:{field}")
    return actual


def array_sha(value: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(value).tobytes())


def safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ReadbackError("inventory_relative_path_missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ReadbackError("inventory_path_not_safe_posix_relative")
    return value


def inside(root: Path, relative: str) -> Path:
    rel = safe_relative(relative)
    path = (root / Path(*PurePosixPath(rel).parts)).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if path != resolved_root and resolved_root not in path.parents:
        raise ReadbackError("tracked_file_escaped_output_root")
    if not path.is_file():
        raise ReadbackError("tracked_reference_not_regular_file")
    return path


def _read_npz_subset(path: Path, expected_members: set[str], selected_keys: set[str]) -> dict[str, np.ndarray]:
    # The central directory is inspected first. Only selected arrays are ever
    # decompressed by NumPy; geometry-bearing arrays remain unread.
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if names != expected_members:
            raise ReadbackError(f"npz_member_set_mismatch:{path.name}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {name[:-4] for name in expected_members}:
            raise ReadbackError(f"npz_array_key_set_mismatch:{path.name}")
        for key in selected_keys:
            result[key] = np.asarray(archive[key])
    return result


def _progress_lines(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise ReadbackError(f"extraction_stdout_unreadable:{type(exc).__name__}") from exc
    if len(lines) != 41:
        raise ReadbackError(f"extraction_stdout_line_count_mismatch:{len(lines)}")
    values = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except Exception as exc:
            raise ReadbackError(f"extraction_stdout_invalid_json:{index}:{type(exc).__name__}") from exc
        if not isinstance(value, dict):
            raise ReadbackError(f"extraction_stdout_not_object:{index}")
        values.append(value)
    progress, final = values[:-1], values[-1]
    expected_done = list(range(20, 801, 20))
    for row, done in zip(progress, expected_done, strict=True):
        if row != {"done": done, "complete": done, "failed": 0, "planned": 800}:
            raise ReadbackError(f"progress_record_mismatch:{done}")
    return progress, final


def _check_wrapper_controls(job: Path, output: Path, plan_raw: str, plan_payload: str, release_raw: str, release_payload: str,
                            stage_raw: str, preflight_raw: str, preflight_status_raw: str, preflight_launch_raw: str) -> dict[str, Any]:
    expected_files = {
        "stage_receipt.json": (stage_raw, None),
        "preflight.result.json": (preflight_raw, None),
        "preflight.status.json": (preflight_status_raw, None),
        "preflight.launch.json": (preflight_launch_raw, None),
        "extraction.launch.json": (EXPECTED_EXTRACTION_LAUNCH_RAW_SHA256, 1001),
        "extraction.stdout": (EXPECTED_EXTRACTION_STDOUT_RAW_SHA256, 2451),
        "extraction.stderr": (EXPECTED_EXTRACTION_STDERR_RAW_SHA256, 3612),
        "extraction.launch.stdout": (EXPECTED_EXTRACTION_LAUNCH_STDOUT_RAW_SHA256, 0),
        "extraction.launch.stderr": (EXPECTED_EXTRACTION_LAUNCH_STDERR_RAW_SHA256, 40),
        "extraction.exitcode": (EXPECTED_EXTRACTION_EXITCODE_RAW_SHA256, 2),
    }
    controls = {}
    for name, (expected_sha, expected_bytes) in expected_files.items():
        path = job / name
        if not path.is_file():
            raise ReadbackError(f"required_job_control_missing:{name}")
        actual_sha = sha_file(path)
        size = path.stat().st_size
        if expected_sha is not None and actual_sha != expected_sha:
            raise ReadbackError(f"job_control_sha_mismatch:{name}")
        if expected_bytes is not None and size != expected_bytes:
            raise ReadbackError(f"job_control_size_mismatch:{name}")
        controls[name] = {"bytes": size, "raw_sha256": actual_sha}
    launch = read_json(job / "extraction.launch.json")
    if (launch.get("status") != "TRACKING_EXTRACTION_STARTED"
            or launch.get("pid") != EXPECTED_EXTRACTION_PID
            or launch.get("process_start_ticks") != EXPECTED_EXTRACTION_START_TICKS
            or launch.get("tracking_plan_raw_sha256") != plan_raw
            or launch.get("tracking_plan_payload_sha256") != plan_payload
            or launch.get("root_release_raw_sha256") != release_raw
            or launch.get("root_release_payload_sha256") != release_payload
            or launch.get("preflight_result_raw_sha256") != EXPECTED_PREFLIGHT_RAW_SHA256
            or launch.get("preflight_result_status") != "PREFLIGHT_PASS"
            or launch.get("extractor_worker_sha256") != "c87db60fdebcfd3ed4605135a941a39cb094dac97a5ecdf55256c90f524d94d1"
            or launch.get("extractor_source_sha256") != EXPECTED_EXTRACTOR_SHA256
            or launch.get("geometry_read_authorized") is not False):
        raise ReadbackError("extraction_launch_receipt_binding_mismatch")
    try:
        worker_exit = int((job / "extraction.exitcode").read_text(encoding="ascii").strip())
    except Exception as exc:
        raise ReadbackError(f"extraction_worker_exitcode_invalid:{type(exc).__name__}") from exc
    stderr = (job / "extraction.launch.stderr").read_text(encoding="utf-8", errors="replace")
    known_wrapper_failure = worker_exit == 1 and "extraction_stdout_object_count_mismatch" in stderr
    if not known_wrapper_failure:
        raise ReadbackError("extraction_wrapper_failure_differs_from_known_multiline_stdout_accounting_bug")
    if (job / "extraction.result.json").exists():
        raise ReadbackError("unexpected_wrapper_result_file_exists")
    producer_stderr = job / "extraction.stderr"
    producer_stderr_size = producer_stderr.stat().st_size
    producer_stderr_sha256 = sha_file(producer_stderr)
    return {
        "files": controls,
        "worker_exit_code": worker_exit,
        "wrapper_accounting_failure": "known extraction_stdout_object_count_mismatch after 40 progress JSON lines and one final result JSON line",
        "producer_result_file_missing_due_wrapper": True,
        "producer_stderr_bytes": producer_stderr_size,
        "producer_stderr_raw_sha256": producer_stderr_sha256,
        "producer_stderr_content_interpreted": False,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    code = args.code.resolve(strict=True)
    job = args.job.resolve(strict=True)
    output = args.output.resolve(strict=True)
    plan_path = args.plan.resolve(strict=True)
    release_path = args.root_release.resolve(strict=True)
    root_readback_path = args.root_readback_receipt.resolve(strict=True)
    preflight_path = args.preflight_result.resolve(strict=True)
    preflight_status_path = args.preflight_status.resolve(strict=True)
    stage_receipt_path = args.stage_receipt.resolve(strict=True)
    preflight_launch_path = args.preflight_launch.resolve(strict=True)
    extractor = code / "world_shape_visual_track_extract_20261002_attempt07.py"
    helper = code / "video_features.py"
    landmarker = args.landmarker.resolve(strict=True)
    core_path = code / "tts_mechanism_visual_20260930.py"
    tone_path = code / "tts_mechanism_visual_tone_20261001.py"
    pins = {
        "extractor": (extractor, EXPECTED_EXTRACTOR_SHA256),
        "feature_helper": (helper, EXPECTED_HELPER_SHA256),
        "landmarker": (landmarker, EXPECTED_LANDMARKER_SHA256),
        "visual_core": (core_path, EXPECTED_CORE_SHA256),
        "tone_adapter": (tone_path, EXPECTED_TONE_SHA256),
        "tracking_plan": (plan_path, EXPECTED_PLAN_RAW_SHA256),
        "root_release": (release_path, EXPECTED_RELEASE_RAW_SHA256),
        "root_preflight_readback": (root_readback_path, EXPECTED_READBACK_RECEIPT_RAW_SHA256),
        "preflight_result": (preflight_path, EXPECTED_PREFLIGHT_RAW_SHA256),
        "preflight_status": (preflight_status_path, EXPECTED_PREFLIGHT_STATUS_RAW_SHA256),
        "stage_receipt": (stage_receipt_path, EXPECTED_STAGE_RECEIPT_RAW_SHA256),
        "preflight_launch": (preflight_launch_path, EXPECTED_PREFLIGHT_LAUNCH_RAW_SHA256),
    }
    pin_receipts = {}
    for role, (path, expected) in pins.items():
        digest = sha_file(path)
        if digest != expected:
            raise ReadbackError(f"input_pin_mismatch:{role}")
        pin_receipts[role] = {"path": str(path), "bytes": path.stat().st_size, "raw_sha256": digest}

    plan = read_json(plan_path)
    plan_payload = verify_fingerprint(plan, "plan_sha256")
    if (plan_payload != EXPECTED_PLAN_PAYLOAD_SHA256 or plan.get("schema") != "world_shape_visual_tracking_plan_v1"
            or len(plan.get("context_rows", [])) != EXPECTED_CONTEXTS):
        raise ReadbackError("tracking_plan_payload_or_cohort_mismatch")
    release = read_json(release_path)
    release_payload = verify_fingerprint(release, "manifest_sha256")
    if (release_payload != EXPECTED_RELEASE_PAYLOAD_SHA256 or release.get("status") != EXPECTED_ROOT_RELEASE_STATUS
            or release.get("scope") != "TRACKING_EXTRACTION" or release.get("tracking_extraction_authorized") is not True
            or release.get("geometry_read_authorized") is not False
            or release.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or release.get("tracking_plan_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256):
        raise ReadbackError("tracking_root_release_scope_or_binding_mismatch")
    root_readback = read_json(root_readback_path)
    if (root_readback.get("status") != "ROOT_FULL_CPU_PREFLIGHT_METADATA_READBACK_PASS"
            or root_readback.get("worker_exitcode") != 0
            or root_readback.get("pid") != EXPECTED_PREFLIGHT_PID
            or root_readback.get("start_ticks") != EXPECTED_PREFLIGHT_START_TICKS
            or root_readback.get("preflight_result_raw_sha256") != EXPECTED_PREFLIGHT_RAW_SHA256
            or root_readback.get("all_six_model_seals_checked") is not True
            or root_readback.get("all_video_identity_and_clock_rows_checked") is not True
            or root_readback.get("contexts") != EXPECTED_CONTEXTS
            or root_readback.get("videos") != EXPECTED_VIDEOS
            or root_readback.get("full_logs") != 9
            or root_readback.get("stage_sources") != 17
            or root_readback.get("tracking_extraction_authorized") is not False
            or root_readback.get("new_geometry_read") is not False):
        raise ReadbackError("root_preflight_readback_receipt_mismatch")
    expected_preflight_controls = {
        "preflight.exitcode", "preflight.launch.json", "preflight.launch.stderr",
        "preflight.launch.stdout", "preflight.result.json", "preflight.status.json",
        "preflight.stderr", "preflight.stdout", "stage_receipt.json",
    }
    readback_files = root_readback.get("files")
    if not isinstance(readback_files, dict) or set(readback_files) != expected_preflight_controls:
        raise ReadbackError("root_preflight_readback_file_set_mismatch")
    for name, expected in readback_files.items():
        path = job / name
        if not path.is_file():
            raise ReadbackError(f"preflight_control_missing_from_root_readback:{name}")
        actual = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
        if actual != expected:
            raise ReadbackError(f"preflight_control_differs_from_root_readback:{name}")
    preflight = read_json(preflight_path)
    if (preflight.get("status") != "PREFLIGHT_PASS" or preflight.get("tracking_plan_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256
            or preflight.get("contexts") != EXPECTED_CONTEXTS or preflight.get("videos") != EXPECTED_VIDEOS
            or preflight.get("video_pixels_or_mouth_geometry_read") is not False
            or preflight.get("evaluation_results_parsed") is not False):
        raise ReadbackError("preflight_scope_or_cohort_mismatch")
    stage = read_json(stage_receipt_path)
    if (stage.get("status") != "STAGE_PASS" or stage.get("verified_file_count") != 17
            or stage.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or stage.get("tracking_plan_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256
            or stage.get("tracking_extraction_authorized") is not False
            or stage.get("evaluation_results_parsed") is not False):
        raise ReadbackError("stage_receipt_scope_or_binding_mismatch")
    preflight_status = read_json(preflight_status_path)
    if (preflight_status.get("status") != "PREFLIGHT_PASS" or preflight_status.get("preflight_exitcode") != 0
            or preflight_status.get("tracking_extraction_authorized") is not False
            or preflight_status.get("evaluation_results_parsed") is not False):
        raise ReadbackError("preflight_status_scope_mismatch")
    preflight_launch = read_json(preflight_launch_path)
    if (preflight_launch.get("pid") != EXPECTED_PREFLIGHT_PID
            or preflight_launch.get("process_start_ticks") != EXPECTED_PREFLIGHT_START_TICKS
            or preflight_launch.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or preflight_launch.get("tracking_plan_payload_sha256") != EXPECTED_PLAN_PAYLOAD_SHA256
            or preflight_launch.get("tracking_extraction_authorized") is not False):
        raise ReadbackError("preflight_launch_pid_or_binding_mismatch")

    controls = _check_wrapper_controls(job, output, EXPECTED_PLAN_RAW_SHA256, plan_payload,
                                       EXPECTED_RELEASE_RAW_SHA256, release_payload,
                                       EXPECTED_STAGE_RECEIPT_RAW_SHA256, EXPECTED_PREFLIGHT_RAW_SHA256,
                                       EXPECTED_PREFLIGHT_STATUS_RAW_SHA256, EXPECTED_PREFLIGHT_LAUNCH_RAW_SHA256)
    progress, final_result = _progress_lines(job / "extraction.stdout")
    if (final_result.get("status") != "COMPLETE" or final_result.get("completed_videos") != EXPECTED_VIDEOS
            or final_result.get("failed_videos") != 0 or final_result.get("output_dir") != str(output)
            or not SHA_RE.fullmatch(str(final_result.get("manifest_sha256", "")))
            or not SHA_RE.fullmatch(str(final_result.get("all_files_sha256", "")))):
        raise ReadbackError("final_extractor_result_not_complete")

    manifest_path = output / "visual_track_manifest.json"
    protocol_path = output / "visual_track_protocol.json"
    progress_path = output / "progress.jsonl"
    inventory_path = output / "all_files_sha256.json"
    seal_path = output / "visual_track_seal.json"
    for path in (manifest_path, protocol_path, progress_path, inventory_path, seal_path):
        if not path.is_file():
            raise ReadbackError(f"track_output_control_missing:{path.name}")
    manifest_raw = sha_file(manifest_path)
    inventory_raw = sha_file(inventory_path)
    manifest = read_json(manifest_path)
    manifest_payload = verify_fingerprint(manifest, "manifest_sha256")
    if final_result.get("manifest_sha256") != manifest_raw or final_result.get("all_files_sha256") != inventory_raw:
        raise ReadbackError("final_stdout_result_does_not_bind_actual_outputs")
    if (manifest.get("schema") != "world_shape_visual_track_manifest_v1" or manifest.get("status") != "COMPLETE"
            or manifest.get("intended_videos") != EXPECTED_VIDEOS or manifest.get("completed_videos") != EXPECTED_VIDEOS
            or manifest.get("failed_videos") != 0 or manifest.get("failures") != []
            or manifest.get("effects_read") is not False or manifest.get("model_forward_performed") is not False
            or manifest.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or manifest.get("tracking_plan_sha256") != plan_payload):
        raise ReadbackError("track_manifest_scope_or_cohort_mismatch")
    protocol = read_json(protocol_path)
    if (manifest.get("tracking_protocol_sha256") != sha_file(protocol_path)
            or manifest.get("progress_sha256") != sha_file(progress_path)
            or protocol.get("status") != "TRACKING_STARTED"
            or protocol.get("tracking_plan_raw_sha256") != EXPECTED_PLAN_RAW_SHA256
            or protocol.get("tracking_plan_sha256") != plan_payload
            or protocol.get("root_release_raw_sha256") != EXPECTED_RELEASE_RAW_SHA256
            or protocol.get("root_release_payload_sha256") != release_payload
            or protocol.get("extractor_source_sha256") != EXPECTED_EXTRACTOR_SHA256
            or protocol.get("video_features_sha256") != EXPECTED_HELPER_SHA256
            or protocol.get("landmarker_sha256") != EXPECTED_LANDMARKER_SHA256
            or protocol.get("effects_read") is not False or protocol.get("model_forward_performed") is not False
            or protocol.get("geometry_read_by_consumer") is not False):
        raise ReadbackError("track_protocol_source_or_scope_mismatch")
    try:
        journal_rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise ReadbackError(f"track_progress_jsonl_invalid:{type(exc).__name__}") from exc
    if len(journal_rows) != EXPECTED_VIDEOS:
        raise ReadbackError("track_progress_record_count_mismatch")

    seal = read_json(seal_path)
    inventory_obj = read_json(inventory_path)
    inventory = inventory_obj.get("files")
    if not isinstance(inventory, dict):
        raise ReadbackError("track_inventory_files_missing")
    if (seal.get("schema") != "world_shape_visual_track_seal_v1" or seal.get("status") != "SEALED"
            or seal.get("manifest_sha256") != manifest_raw or seal.get("manifest_payload_sha256") != manifest_payload
            or seal.get("all_files_sha256") != inventory_raw or seal.get("completed_videos") != EXPECTED_VIDEOS
            or seal.get("failed_videos") != 0):
        raise ReadbackError("track_seal_manifest_or_count_mismatch")

    expected_records = []
    seen_contexts = set()
    for order, row in enumerate(sorted(plan["context_rows"], key=lambda value: int(value["context_order"]))):
        if int(row["context_order"]) != order:
            raise ReadbackError("plan_context_order_not_contiguous")
        context_key = (str(row["sample_id"]), str(row["direction"]))
        if context_key in seen_contexts:
            raise ReadbackError("plan_context_identity_duplicate")
        seen_contexts.add(context_key)
        for cell in CELLS:
            expected_records.append((row, cell, row["cells"][cell]))
    if len(expected_records) != EXPECTED_VIDEOS or len(manifest.get("records", [])) != EXPECTED_VIDEOS:
        raise ReadbackError("track_manifest_expected_record_count_mismatch")

    expected_inventory = {"visual_track_protocol.json", "progress.jsonl", "visual_track_manifest.json"}
    journal_by_key = {}
    for row in journal_rows:
        key = (str(row.get("sample_id", "")), str(row.get("direction", "")), str(row.get("cell", "")))
        if key in journal_by_key or row.get("status") != "complete":
            raise ReadbackError("track_progress_duplicate_or_failed_row")
        journal_by_key[key] = row
    details = []
    for record, (plan_row, cell, video_ref) in zip(manifest["records"], expected_records, strict=True):
        sample, speaker, direction = str(plan_row["sample_id"]), str(plan_row["speaker_id"]), str(plan_row["direction"])
        expected_identity = {
            "sample_id": sample, "speaker_id": speaker, "split": str(plan_row["split"]),
            "direction": direction, "cell": cell, "context_order": int(plan_row["context_order"]),
            "ordered_eval_index": int(plan_row["ordered_eval_index"]),
        }
        for key, value in expected_identity.items():
            if record.get(key) != value:
                raise ReadbackError(f"track_record_identity_mismatch:{sample}:{direction}:{cell}:{key}")
        expected_video_fields = {
            "status": "complete",
            "video_relative_path": str(video_ref["video_relative_path"]),
            "video_sha256": str(video_ref["video_sha256"]),
            "video_bytes": int(video_ref["video_bytes"]),
            "official_frame_count": int(video_ref["official_frame_count"]),
        }
        for key, value in expected_video_fields.items():
            if record.get(key) != value:
                raise ReadbackError(f"track_record_video_binding_mismatch:{sample}:{direction}:{cell}:{key}")
        if (not math.isclose(float(record.get("fps_hz", 0.0)), EXPECTED_FPS, rel_tol=0, abs_tol=1e-6)
                or int(record.get("valid_frames", -1)) < 0
                or int(record.get("valid_frames", -1)) > int(video_ref["official_frame_count"])):
            raise ReadbackError(f"track_record_fps_or_valid_count_mismatch:{sample}:{direction}:{cell}")
        expected_stem = f"{sample}__{direction}__{cell}.npz"
        feature_rel = f"features/{expected_stem}"
        availability_rel = f"availability/{expected_stem}"
        if record.get("feature_relative_path") != feature_rel or record.get("availability_relative_path") != availability_rel:
            raise ReadbackError(f"track_record_npz_path_mismatch:{sample}:{direction}:{cell}")
        array_inventory = record.get("array_inventory")
        if not isinstance(array_inventory, dict) or set(array_inventory) != ARRAY_NAMES:
            raise ReadbackError(f"track_record_array_inventory_schema_mismatch:{sample}:{direction}:{cell}")
        for name, entry in array_inventory.items():
            if (not isinstance(entry, dict) or not SHA_RE.fullmatch(str(entry.get("sha256", "")))
                    or not isinstance(entry.get("shape"), list) or not isinstance(entry.get("dtype"), str)):
                raise ReadbackError(f"track_record_array_inventory_entry_malformed:{name}")
        n = int(video_ref["official_frame_count"])
        for name in ("timestamps_s", "valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area"):
            if array_inventory[name].get("shape") != [n]:
                raise ReadbackError(f"track_array_inventory_frame_shape_mismatch:{name}")
        if array_inventory["timestamps_s"].get("dtype") != "float64" or array_inventory["valid"].get("dtype") != "bool":
            raise ReadbackError("track_timestamp_or_valid_inventory_dtype_mismatch")

        feature_path = inside(output, feature_rel)
        availability_path = inside(output, availability_rel)
        feature_raw, availability_raw = sha_file(feature_path), sha_file(availability_path)
        if feature_raw != record.get("feature_sha256") or availability_raw != record.get("availability_sha256"):
            raise ReadbackError(f"track_npz_raw_sha_mismatch:{sample}:{direction}:{cell}")
        if inventory.get(feature_rel) != feature_raw or inventory.get(availability_rel) != availability_raw:
            raise ReadbackError(f"track_npz_inventory_mismatch:{sample}:{direction}:{cell}")
        expected_inventory.update((feature_rel, availability_rel))

        feature_arrays = _read_npz_subset(feature_path, FEATURE_MEMBERS,
                                          {"timestamps_s.npy"[:-4], "valid.npy"[:-4], "metadata_json.npy"[:-4]})
        availability_arrays = _read_npz_subset(availability_path, {"timestamps_s.npy", "valid.npy"}, {"timestamps_s", "valid"})
        feature_times, feature_valid = feature_arrays["timestamps_s"], feature_arrays["valid"]
        avail_times, avail_valid = availability_arrays["timestamps_s"], availability_arrays["valid"]
        if (feature_times.dtype != np.float64 or feature_valid.dtype != np.bool_
                or avail_times.dtype != np.float64 or avail_valid.dtype != np.bool_
                or feature_times.shape != (n,) or feature_valid.shape != (n,)
                or avail_times.shape != (n,) or avail_valid.shape != (n,)):
            raise ReadbackError(f"track_timestamp_or_valid_shape_dtype_mismatch:{sample}:{direction}:{cell}")
        if not np.array_equal(feature_times, avail_times) or not np.array_equal(feature_valid, avail_valid):
            raise ReadbackError(f"feature_and_availability_clock_or_valid_mismatch:{sample}:{direction}:{cell}")
        if (not np.isfinite(feature_times).all() or (n > 1 and np.any(np.diff(feature_times) <= 0))
                or (n > 1 and not np.allclose(np.diff(feature_times), 1.0 / EXPECTED_FPS, rtol=0, atol=1e-9))):
            raise ReadbackError(f"track_clock_invalid:{sample}:{direction}:{cell}")
        metadata_raw = feature_arrays["metadata_json"]
        if metadata_raw.shape != ():
            raise ReadbackError("feature_metadata_array_not_scalar")
        try:
            metadata = json.loads(str(metadata_raw.item()))
        except Exception as exc:
            raise ReadbackError(f"feature_metadata_json_invalid:{type(exc).__name__}") from exc
        expected_metadata = {
            **expected_identity,
            "protocol": "world_shape_visual_tracks_20261002_v1",
            "video_relative_path": str(video_ref["video_relative_path"]),
            "video_sha256": str(video_ref["video_sha256"]),
            "recipient_grid_sha256": str(plan_row["recipient_grid_sha256"]),
            "landmarker_sha256": EXPECTED_LANDMARKER_SHA256,
            "video_features_sha256": EXPECTED_HELPER_SHA256,
            "visual_core_sha256": EXPECTED_CORE_SHA256,
            "tone_adapter_sha256": EXPECTED_TONE_SHA256,
            "full_recipient_clock": True,
            "fps_hz_required": EXPECTED_FPS,
            "decoded_frame_count": n,
            "declared_frame_count": n,
        }
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected_metadata.items()):
            raise ReadbackError(f"feature_metadata_plan_identity_mismatch:{sample}:{direction}:{cell}")
        if not math.isclose(float(metadata.get("fps", 0)), EXPECTED_FPS, rel_tol=0, abs_tol=1e-6):
            raise ReadbackError("feature_metadata_fps_mismatch")
        if (array_sha(feature_times) != record.get("timestamp_sha256")
                or array_sha(feature_times) != array_inventory["timestamps_s"].get("sha256")
                or array_sha(feature_valid) != array_inventory["valid"].get("sha256")
                or int(record.get("valid_frames", -1)) != int(feature_valid.sum())):
            raise ReadbackError(f"track_clock_or_valid_array_digest_mismatch:{sample}:{direction}:{cell}")
        progress_key = (sample, direction, cell)
        progress_record = journal_by_key.get(progress_key)
        if (progress_record is None or progress_record.get("feature_sha256") != feature_raw
                or progress_record.get("availability_sha256") != availability_raw):
            raise ReadbackError(f"progress_log_identity_or_sha_mismatch:{sample}:{direction}:{cell}")
        details.append({
            **expected_identity,
            "video_relative_path": str(video_ref["video_relative_path"]),
            "video_sha256": str(video_ref["video_sha256"]),
            "official_frame_count": n,
            "feature_relative_path": feature_rel,
            "feature_bytes": feature_path.stat().st_size,
            "feature_raw_sha256": feature_raw,
            "availability_relative_path": availability_rel,
            "availability_bytes": availability_path.stat().st_size,
            "availability_raw_sha256": availability_raw,
            "timestamp_array_sha256": array_sha(feature_times),
            "valid_frames": int(feature_valid.sum()),
        })
    if set(inventory) != expected_inventory or len(inventory) != 1603:
        raise ReadbackError(f"track_inventory_path_set_mismatch:{len(inventory)}")
    if set(journal_by_key) != {(str(r["sample_id"]), str(r["direction"]), cell) for r in plan["context_rows"] for cell in CELLS}:
        raise ReadbackError("track_progress_identity_set_mismatch")
    for relative in ("visual_track_protocol.json", "progress.jsonl", "visual_track_manifest.json"):
        path = inside(output, relative)
        if sha_file(path) != inventory.get(relative):
            raise ReadbackError(f"track_control_inventory_mismatch:{relative}")
    if seal.get("files") != len(inventory):
        raise ReadbackError("track_seal_inventory_count_mismatch")

    if args.report.exists():
        raise ReadbackError("readback_report_must_be_absent")
    report = {
        "schema": "world_shape_visual_track_independent_readback_v1",
        "status": "TRACK_OUTPUT_INDEPENDENTLY_VERIFIED_WITH_KNOWN_WRAPPER_FAILURE",
        "scope": "metadata, opaque NPZ bytes, frame clocks and valid masks only; no mouth geometry values loaded",
        "evidence": {
            "readback_checker_source": {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).resolve().stat().st_size,
                "raw_sha256": sha_file(Path(__file__).resolve()),
            },
            **pin_receipts,
            "tracking_output_root": str(output),
            "tracking_manifest_raw_sha256": manifest_raw,
            "tracking_manifest_payload_sha256": manifest_payload,
            "tracking_seal_raw_sha256": sha_file(seal_path),
            "tracking_inventory_raw_sha256": inventory_raw,
            "tracking_protocol_raw_sha256": sha_file(protocol_path),
            "tracking_progress_raw_sha256": sha_file(progress_path),
            "tracking_inventory_entries": len(inventory),
            "feature_npz_files_verified_raw_bytes": EXPECTED_VIDEOS,
            "availability_npz_files_verified_raw_bytes": EXPECTED_VIDEOS,
            "feature_arrays_loaded": ["timestamps_s", "valid", "metadata_json"],
            "availability_arrays_loaded": ["timestamps_s", "valid"],
            "geometry_arrays_loaded": [],
            "contexts_verified": EXPECTED_CONTEXTS,
            "video_records_verified": EXPECTED_VIDEOS,
            "valid_video_records": len(details),
            "progress_stdout_rows": len(progress),
            "final_producer_result": final_result,
            "worker_control": controls,
        },
        "record_inventory": details,
        "evaluation_results_parsed": False,
        "evaluation_effects_read": False,
        "model_forward_performed": False,
        "mouth_geometry_values_read": False,
        "wrapper_exitcode_interpreted_as_producer_failure": False,
    }
    report["report_payload_sha256"] = sha_bytes(canonical_bytes(report))
    raw = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "status": report["status"],
        "report_path": str(args.report.resolve()),
        "report_bytes": len(raw),
        "report_raw_sha256": sha_bytes(raw),
        "report_payload_sha256": report["report_payload_sha256"],
        "contexts": EXPECTED_CONTEXTS,
        "videos": EXPECTED_VIDEOS,
        "geometry_arrays_loaded": 0,
        "evaluation_results_parsed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root-release", type=Path, required=True)
    parser.add_argument("--root-readback-receipt", type=Path, required=True)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--preflight-status", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--preflight-launch", type=Path, required=True)
    parser.add_argument("--landmarker", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "READBACK_FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
