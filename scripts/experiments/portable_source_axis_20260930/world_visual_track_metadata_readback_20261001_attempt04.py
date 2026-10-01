#!/usr/bin/env python3
"""Build a sealed, score-free readback of WORLD visual-track metadata."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACK = Path("/dev/shm/tts-research-20261001/independent_validation/world_visual_attempt05/output")
RUN = Path("/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run")
PLAN = Path("/root/tts-research-system-20261001/world_phone_transfer_gpu_20261001_attempt05/preflight/execution_plan.json")
GRID = Path("/root/tts-research-system-20261001/job_control/world_visual_grid_inventory_attempt03/metadata_inventory.json")
FREEZE = Path("/dev/shm/tts-research-20261001/independent_validation/world_visual_analysis_attempt09/world_visual_availability_freeze.json")
FREEZE_SEAL = Path("/dev/shm/tts-research-20261001/independent_validation/world_visual_analysis_attempt09/world_visual_availability_freeze.seal.json")
RUNTIME_FLAGS = Path("/root/autodl-tmp/tts-research-storage-20261001/job_control/world_phone_transfer_gpu_attempt07_v6_tmpfs_worker03/world_runtime_flags.json")
CONSUMER = Path("/dev/shm/tts-research-20261001/independent_validation/world_consumer04_attempt02/output")
JOB = Path("/root/tts-research-system-20261001/job_control/world_visual_track_metadata_readback_attempt04")
EXPORT = JOB / "track_metadata_readback_attempt04"
EXPECTED_PLAN_RAW = "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d"
EXPECTED_PLAN_FINGERPRINT = "1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc"
EXPECTED_GRID = "7b3e879b9a34a6ea20159e770d6530fc77b19fd013381ce4017e87cbbc19203b"
EXPECTED_FREEZE = "67131b9bd381dd2e393ab739ae51408e5edf15594cd3e9b67b844c5857621ccf"
EXPECTED_FREEZE_SEAL = "1e7bab71c2d16689f5f316b543883dec3058f27aa053f7689d9495e95b3b37d2"
EXPECTED_RUN_FILES = "dfbbc35bea359d12ae7383671534fb6e1e08e7f62532de9faad49c37226ffe8c"
EXPECTED_RUN_PROTOCOL = "8c29c8f757c29a88642f69301197560e2356ed641a959bb0bd59fa0f109473ee"
EXPECTED_RUN_ROWS = "d0f81f8d35fa9761d746c7420259d8605fc9191b1665ab171b0813fce844881f"
EXPECTED_RUN_RESULTS = "e494f4dacaeb30520631f061a2b7f0e80ca05da716f7b609bcd7bdba64feaf9d"
EXPECTED_RUN_STATUS = "361b4514f0bab18edcd5e114517cd5b80e3ec76fcb2bd56f873e237e7cda4b7f"
EXPECTED_RUNTIME_FLAGS = "7455c2594579dca28ddd099aea25b7b9006694e5fab40baa4a064bae301581b1"
EXPECTED_EXTRACTOR = "6bec294adcfeff320f70a40e6579b9cb67bc69677fcc07f3a9950ad4186dd87b"
EXPECTED_CONSUMER = "8e40c3017ca7c53924302522678c18854c9b0702ce2223821b03d7fd05166a92"
EXPECTED_LATENCY = -0.085
VISUAL_IDS = ("V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E")
ARRAY_NAMES = {"landmarks", "canonical_mouth", "valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area", "timestamps_s"}
TOP_FILES = {"visual_track_protocol.json", "progress.jsonl"}
CHUNK_BYTES = 7_000_000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non_object_jsonl_row:{path.name}:{line_no}")
                rows.append(value)
    return rows


def skip_json_value(text: str, start: int) -> int:
    """Return the end offset of a JSON value without decoding its payload."""
    n = len(text)
    while start < n and text[start].isspace():
        start += 1
    if start >= n:
        raise ValueError("missing_json_value")
    if text[start] == '"':
        i = start + 1
        escaped = False
        while i < n:
            ch = text[i]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return i + 1
            i += 1
        raise ValueError("unterminated_json_string")
    if text[start] in "[{":
        stack = ["]" if text[start] == "[" else "}"]
        in_string = False
        escaped = False
        for i in range(start + 1, n):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                stack.append("]")
            elif ch == "{":
                stack.append("}")
            elif ch in "]}":
                if not stack or ch != stack.pop():
                    raise ValueError("malformed_json_container")
                if not stack:
                    return i + 1
        raise ValueError("unterminated_json_container")
    i = start
    while i < n and text[i] not in ",}]" and not text[i].isspace():
        i += 1
    if i == start:
        raise ValueError("empty_json_primitive")
    return i


def read_identity_jsonl(path: Path, wanted: set[str]) -> list[dict[str, Any]]:
    """Decode only requested identity fields; skip metric/effect JSON values."""
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, text in enumerate(stream, 1):
            if not text.strip():
                continue
            i = 0
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] != "{":
                raise ValueError(f"expected_json_object:{path.name}:{line_no}")
            i += 1
            row: dict[str, Any] = {}
            while True:
                while i < len(text) and (text[i].isspace() or text[i] == ","):
                    i += 1
                if i < len(text) and text[i] == "}":
                    i += 1
                    break
                key, i = decoder.raw_decode(text, i)
                if not isinstance(key, str):
                    raise ValueError(f"non_string_json_key:{path.name}:{line_no}")
                while i < len(text) and text[i].isspace():
                    i += 1
                if i >= len(text) or text[i] != ":":
                    raise ValueError(f"missing_json_colon:{path.name}:{line_no}")
                i += 1
                while i < len(text) and text[i].isspace():
                    i += 1
                if key in wanted:
                    value, i = decoder.raw_decode(text, i)
                    row[key] = value
                else:
                    i = skip_json_value(text, i)
            if set(row) != wanted:
                raise ValueError(f"identity_field_coverage_mismatch:{path.name}:{line_no}")
            rows.append(row)
    return rows


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def main() -> None:
    if EXPORT.exists():
        raise FileExistsError(f"refusing_existing_export:{EXPORT}")
    TRACK.resolve(strict=True)
    RUN.resolve(strict=True)
    PLAN.resolve(strict=True)
    GRID.resolve(strict=True)
    FREEZE.resolve(strict=True)
    FREEZE_SEAL.resolve(strict=True)
    RUNTIME_FLAGS.resolve(strict=True)
    CONSUMER.resolve(strict=True)
    require(sha(PLAN) == EXPECTED_PLAN_RAW, "execution_plan_raw_sha_mismatch")
    plan = read_json(PLAN)
    require(plan.get("plan_sha256") == EXPECTED_PLAN_FINGERPRINT and hashlib.sha256(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest() == EXPECTED_PLAN_FINGERPRINT, "execution_plan_fingerprint_mismatch")
    require(sha(GRID) == EXPECTED_GRID, "grid_inventory_sha_mismatch")
    require(sha(FREEZE) == EXPECTED_FREEZE, "availability_freeze_sha_mismatch")
    require(sha(FREEZE_SEAL) == EXPECTED_FREEZE_SEAL, "availability_freeze_seal_sha_mismatch")
    require(sha(RUNTIME_FLAGS) == EXPECTED_RUNTIME_FLAGS, "runtime_flags_sha_mismatch")
    require(sha(RUN / "all_files_sha256.json") == EXPECTED_RUN_FILES, "producer_all_files_sha_mismatch")
    require(sha(RUN / "evaluation_protocol.json") == EXPECTED_RUN_PROTOCOL, "producer_protocol_sha_mismatch")
    require(sha(RUN / "evaluation_rows.jsonl") == EXPECTED_RUN_ROWS, "producer_rows_sha_mismatch")
    require(sha(RUN / "evaluation_results.jsonl") == EXPECTED_RUN_RESULTS, "producer_results_sha_mismatch")
    require(sha(RUN / "run_status.json") == EXPECTED_RUN_STATUS, "producer_status_sha_mismatch")

    status_path = TRACK / "visual_track_status.json"
    protocol_path = TRACK / "visual_track_protocol.json"
    all_path = TRACK / "all_files_sha256.json"
    seal_path = TRACK / "visual_track_seal.json"
    progress_path = TRACK / "progress.jsonl"
    status, protocol, all_doc, seal = map(read_json, (status_path, protocol_path, all_path, seal_path))
    require(status.get("status") == "COMPLETE" and status.get("planned_videos") == 1600 and status.get("completed_videos") == 1600 and status.get("failed_videos") == 0 and not status.get("errors"), "track_status_not_complete")
    require(seal.get("status") == "SEALED" and seal.get("completed_videos") == 1600 and seal.get("failed_videos") == 0, "track_seal_not_complete")
    require(seal.get("protocol_sha256") == sha(protocol_path) and seal.get("status_sha256") == sha(status_path) and seal.get("all_files_sha256") == sha(all_path), "track_seal_hash_chain_mismatch")
    require(status.get("progress_sha256") == sha(progress_path), "track_status_progress_hash_mismatch")
    require(protocol.get("status") == "PLAN_FROZEN" and protocol.get("protocol_id") == "world_visual_tracks_20261001_v1", "track_protocol_status_or_id_mismatch")
    require(protocol.get("worker_process_start_method") == "spawn" and protocol.get("extractor_source", {}).get("sha256") == EXPECTED_EXTRACTOR, "track_protocol_extractor_binding_mismatch")
    require(protocol.get("producer", {}).get("plan_raw_sha256") == EXPECTED_PLAN_RAW and protocol.get("producer", {}).get("plan_fingerprint_sha256") == EXPECTED_PLAN_FINGERPRINT, "track_protocol_plan_binding_mismatch")
    consumer_receipt = protocol.get("producer", {}).get("consumer", {})
    require(protocol.get("producer", {}).get("all_files_sha256") == EXPECTED_RUN_FILES, "track_protocol_producer_inventory_mismatch")
    require(consumer_receipt.get("seal_sha256") == "5f4f6677b5740a010e89120670f751a16b5e929899545b9be2e7856340290350" and consumer_receipt.get("inventory_sha256") == "097736a5a6cfb74882ca2f005001f34223370f6f86d84ff4ad9536de6ce003ce" and consumer_receipt.get("audit_sha256") == "fa06babb3a106bf1ed3acc337a6dc3bddc2c7df35dbd176f6c545c80cf599e30" and consumer_receipt.get("inventory_count") == 3, "track_protocol_consumer_receipt_mismatch")
    require(protocol.get("grid_inventory", {}).get("sha256") == EXPECTED_GRID and protocol.get("latency_s") == EXPECTED_LATENCY, "track_protocol_grid_or_latency_mismatch")
    require(protocol.get("intended") == {"contexts": 160, "videos": 1600, "video_refs_per_context": 10, "workers": 4}, "track_protocol_intended_mismatch")
    require(protocol.get("runtime") == {"mediapipe_version": "0.10.21", "opencv_version": "4.10.0", "numpy_version": "1.26.4"}, "track_runtime_versions_mismatch")
    require(protocol.get("latency_s") == EXPECTED_LATENCY and protocol.get("extractor_source", {}).get("sha256") == EXPECTED_EXTRACTOR, "track_protocol_latency_or_extractor_mismatch")
    require(protocol.get("feature_source", {}).get("sha256") == "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4", "track_protocol_feature_helper_mismatch")
    require(protocol.get("native_visual_lineage", {}).get("tone_adapter_sha256") == "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98", "track_protocol_tone_adapter_mismatch")

    files = all_doc.get("files")
    require(isinstance(files, dict) and len(files) == 3202 and seal.get("file_count") == 3202, "track_all_files_count_mismatch")
    require(all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in files.items()), "invalid_all_files_entry")
    result_identity_fields = {"row_id", "ordered_eval_index", "sample_id", "speaker_id", "split", "direction", "visual_id", "video_path", "video_sha256", "visual_npz", "visual_npz_sha256", "status"}
    results = read_identity_jsonl(RUN / "evaluation_results.jsonl", result_identity_fields)
    results_by_id = {row.get("row_id"): row for row in results}
    require(len(results_by_id) == len(results) == 12160, "duplicate_or_missing_producer_row_id")
    require(len(results) == 12160 and all(row.get("status") == "scored" for row in results), "producer_result_coverage_mismatch")
    expected_videos: dict[tuple[str, str, str], dict[str, Any]] = {}
    contexts: dict[tuple[str, str], int] = defaultdict(int)
    speakers: dict[str, str] = {}
    plan_rows = plan.get("rows")
    require(isinstance(plan_rows, list) and len(plan_rows) == 12160, "producer_plan_row_coverage_mismatch")
    for planned in plan_rows:
        row = results_by_id.get(planned.get("row_id"))
        require(isinstance(row, dict), "producer_plan_result_join_missing")
        for field in ("ordered_eval_index", "sample_id", "speaker_id", "split", "direction", "visual_id"):
            require(planned.get(field) == row.get(field), f"producer_plan_result_identity_mismatch:{field}")
        key = (str(row["sample_id"]), str(row["direction"]), str(row["visual_id"]))
        item = {k: row[k] for k in ("row_id", "sample_id", "speaker_id", "split", "direction", "visual_id", "video_path", "video_sha256", "visual_npz", "visual_npz_sha256")}
        if key in expected_videos:
            prior = expected_videos[key]
            require((prior["video_path"], prior["video_sha256"], prior["visual_npz"], prior["visual_npz_sha256"], prior["speaker_id"], prior["split"]) == (item["video_path"], item["video_sha256"], item["visual_npz"], item["visual_npz_sha256"], item["speaker_id"], item["split"]), "producer_visual_alias_identity_mismatch")
        else:
            expected_videos[key] = item
        contexts[(key[0], key[1])] += 1
        speakers[key[0]] = str(row["speaker_id"])
    require(len(contexts) == 160 and set(contexts.values()) == {76}, "producer_context_coverage_mismatch")
    require(len(expected_videos) == 1600 and {key[2] for key in expected_videos} == set(VISUAL_IDS), "producer_visual_coverage_mismatch")
    require(len(speakers) == 80 and len(set(speakers.values())) == 40 and Counter(speakers.values()).most_common()[0][1] == 2, "producer_speaker_coverage_mismatch")

    expected_paths = set(TOP_FILES)
    for sample, direction, visual_id in expected_videos:
        stem = f"{sample}__{direction}__{visual_id}"
        expected_paths.add(f"tracks/{stem}.npz")
        expected_paths.add(f"tracks/{stem}.json")
    require(expected_paths == set(files), "track_inventory_exact_path_coverage_mismatch")
    for relative, expected_hash in files.items():
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in relative, f"unsafe_track_inventory_path:{relative}")
        member = TRACK / rel
        require(member.is_file() and sha(member) == expected_hash, f"track_inventory_member_hash_mismatch:{relative}")

    progress_rows = read_jsonl(progress_path)
    require(len(progress_rows) == 1600, "progress_count_mismatch")
    progress: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in progress_rows:
        key = (str(row.get("sample_id")), str(row.get("direction")), str(row.get("visual_id")))
        require(key not in progress and row.get("status") == "complete", "progress_duplicate_or_incomplete")
        require(re.fullmatch(r"[0-9a-f]{64}", str(row.get("feature_sha256", ""))) is not None, "progress_feature_sha_invalid")
        require(re.fullmatch(r"[0-9a-f]{64}", str(row.get("feature_metadata_sha256", ""))) is not None, "progress_metadata_sha_invalid")
        progress[key] = row
    require(set(progress) == set(expected_videos), "progress_video_key_coverage_mismatch")

    grid_doc = read_json(GRID)
    freeze_doc, freeze_seal = map(read_json, (FREEZE, FREEZE_SEAL))
    require(freeze_doc.get("status") == "FROZEN_BEFORE_GEOMETRY_READ" and freeze_doc.get("protocol_id") == "world_visual_20261001_two_class_recipient_clock_v1", "availability_freeze_status_or_protocol_mismatch")
    require(freeze_doc.get("selection_uses_mouth_aperture_or_landmark_values") is False and freeze_doc.get("selection_uses_only_sealed_identity_grid_labels_timestamps_and_valid_masks") is True, "availability_freeze_selection_policy_mismatch")
    require(freeze_seal.get("status") == "FROZEN_BEFORE_GEOMETRY_READ" and freeze_seal.get("freeze_sha256") == EXPECTED_FREEZE and freeze_seal.get("selection_uses_mouth_aperture_or_landmarks") is False, "availability_freeze_seal_binding_mismatch")
    freeze_contexts = {(str(row["sample_id"]), str(row["direction"])): row for row in freeze_doc["context_rows"]}
    require(len(freeze_contexts) == 160 and len(freeze_doc.get("speaker_rows", [])) == 80, "availability_freeze_context_coverage_mismatch")
    effective = {(str(row["sample_id"]), str(row["direction"])): row for row in grid_doc["effective_recipient_grids"]}
    required_meta = {"status", "row_id", "sample_id", "speaker_id", "direction", "visual_id", "video_path", "video_sha256", "video_bytes", "feature_path", "feature_sha256", "feature_metadata_path", "decoded_frame_count", "declared_frame_count", "fps", "width", "height", "valid_frames", "invalid_frames", "recipient_grid_sha256", "recipient_phone_intervals_count", "arrays"}
    metadata_rows = []
    sidecar_keys = Counter()
    array_schemas: dict[str, Counter[str]] = {name: Counter() for name in ARRAY_NAMES}
    overlay_contexts = set()
    included_sources = [protocol_path, status_path, all_path, seal_path, progress_path]
    for key, expected in sorted(expected_videos.items()):
        sample, direction, visual_id = key
        stem = f"{sample}__{direction}__{visual_id}"
        feature_rel, sidecar_rel = f"tracks/{stem}.npz", f"tracks/{stem}.json"
        feature_path, sidecar_path = TRACK / feature_rel, TRACK / sidecar_rel
        require(sha(sidecar_path) == progress[key]["feature_metadata_sha256"] == files[sidecar_rel], f"sidecar_progress_or_inventory_sha_mismatch:{stem}")
        require(progress[key]["feature_sha256"] == files[feature_rel], f"feature_progress_or_inventory_sha_mismatch:{stem}")
        meta = read_json(sidecar_path)
        require(set(meta) == required_meta, f"sidecar_schema_mismatch:{stem}")
        for field in ("row_id", "sample_id", "speaker_id", "direction", "visual_id", "video_path", "video_sha256"):
            require(meta.get(field) == expected.get(field), f"sidecar_producer_identity_mismatch:{field}:{stem}")
        require(meta.get("status") == "complete" and meta.get("feature_sha256") == files[feature_rel], f"sidecar_feature_binding_mismatch:{stem}")
        require(Path(meta["feature_path"]).resolve() == feature_path.resolve(), f"sidecar_feature_path_mismatch:{stem}")
        require(Path(meta["feature_metadata_path"]).resolve() == sidecar_path.resolve(), f"sidecar_metadata_path_mismatch:{stem}")
        require(meta.get("recipient_grid_sha256") == effective[(sample, direction)].get("sha256"), f"sidecar_grid_hash_binding_mismatch:{stem}")
        n = meta.get("decoded_frame_count")
        require(isinstance(n, int) and n > 0 and isinstance(meta.get("declared_frame_count"), int), f"sidecar_frame_count_invalid:{stem}")
        require(meta.get("valid_frames", -1) + meta.get("invalid_frames", -1) == n and progress[key].get("decoded_frame_count") == n, f"sidecar_valid_frame_count_mismatch:{stem}")
        arrays = meta.get("arrays")
        require(isinstance(arrays, dict) and set(arrays) == ARRAY_NAMES, f"sidecar_array_name_mismatch:{stem}")
        for name, descriptor in arrays.items():
            shape = descriptor.get("shape")
            require(isinstance(shape, list) and shape and shape[0] == n, f"sidecar_array_clock_mismatch:{name}:{stem}")
            require(isinstance(descriptor.get("dtype"), str) and re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256", ""))), f"sidecar_array_descriptor_invalid:{name}:{stem}")
            array_schemas[name][json.dumps({"shape_tail": shape[1:], "dtype": descriptor["dtype"]}, sort_keys=True, separators=(",", ":"))] += 1
        gkey = (sample, direction)
        grid = effective[gkey]
        expected_condition = "natural" if direction == "tts_to_natural" else "tts"
        require(grid.get("condition") == expected_condition, f"effective_grid_condition_mismatch:{stem}")
        frozen_context = freeze_contexts.get(gkey)
        require(isinstance(frozen_context, dict), f"availability_freeze_context_missing:{stem}")
        frozen_grid = frozen_context.get("effective_grid", {})
        require((frozen_grid.get("path"), frozen_grid.get("sha256"), frozen_grid.get("overlay_used")) == (grid.get("path"), grid.get("sha256"), grid.get("overlay_used")), f"availability_freeze_effective_grid_mismatch:{stem}")
        phone_intervals = frozen_context.get("phone_intervals")
        require(isinstance(phone_intervals, list) and len(phone_intervals) == meta.get("recipient_phone_intervals_count") and len(phone_intervals) > 0, f"frozen_phone_interval_count_mismatch:{stem}")
        require(all(set(phone) == {"phone_index", "label", "start_s", "end_s"} for phone in phone_intervals), f"frozen_phone_interval_schema_mismatch:{stem}")
        frozen_ref = frozen_context.get("feature_refs", {}).get(visual_id)
        require(isinstance(frozen_ref, dict), f"availability_freeze_feature_ref_missing:{stem}")
        require((frozen_ref.get("row_id"), frozen_ref.get("video_path"), frozen_ref.get("video_sha256"), frozen_ref.get("feature_path"), frozen_ref.get("feature_sha256"), frozen_ref.get("feature_metadata_sha256")) == (expected.get("row_id"), expected.get("video_path"), expected.get("video_sha256"), feature_rel, files[feature_rel], files[sidecar_rel]), f"availability_freeze_feature_identity_mismatch:{stem}")
        require((frozen_ref.get("frame_count"), frozen_ref.get("valid_frames"), frozen_ref.get("timestamp_sha256"), frozen_ref.get("valid_mask_sha256")) == (meta.get("decoded_frame_count"), meta.get("valid_frames"), arrays.get("timestamps_s", {}).get("sha256"), arrays.get("valid", {}).get("sha256")), f"availability_freeze_feature_array_binding_mismatch:{stem}")
        context_refs = [x for x in grid_doc["all_grid_refs"] if x.get("sample_id") == sample and x.get("condition") == expected_condition]
        roles = {str(x.get("role")): x for x in context_refs}
        require(len(context_refs) == 3 and set(roles) == {"original", "producer_selected", "effective_recipient_visual_clock"}, f"grid_ref_role_coverage_mismatch:{stem}")
        require((roles["effective_recipient_visual_clock"].get("path"), roles["effective_recipient_visual_clock"].get("sha256")) == (grid.get("path"), grid.get("sha256")), f"effective_grid_ref_mismatch:{stem}")
        require((roles["producer_selected"].get("path"), roles["producer_selected"].get("sha256")) == (grid.get("path"), grid.get("sha256")), f"selected_grid_ref_mismatch:{stem}")
        overlay_used = bool(grid.get("overlay_used"))
        original_changed = (roles["original"].get("path"), roles["original"].get("sha256")) != (grid.get("path"), grid.get("sha256"))
        require(overlay_used == original_changed, f"overlay_grid_lineage_mismatch:{stem}")
        if overlay_used:
            require(sample == "S0064_BAC009S0064W0121", f"unexpected_overlay_sample:{stem}")
            overlay_contexts.add(gkey)
        sidecar_keys.update(meta.keys())
        metadata_rows.append({"sample_id": sample, "speaker_id": expected["speaker_id"], "direction": direction, "visual_id": visual_id, "row_id": expected["row_id"], "video_sha256": expected["video_sha256"], "feature_sha256": files[feature_rel], "feature_bytes": feature_path.stat().st_size, "feature_metadata_sha256": files[sidecar_rel], "feature_metadata_bytes": sidecar_path.stat().st_size, "decoded_frame_count": n, "valid_frames": meta["valid_frames"], "invalid_frames": meta["invalid_frames"], "recipient_grid_path": grid.get("path"), "recipient_grid_sha256": meta["recipient_grid_sha256"], "recipient_grid_overlay_used": overlay_used, "phone_interval_count": meta["recipient_phone_intervals_count"], "freeze_phone_intervals_sha256": hashlib.sha256(canonical(phone_intervals)).hexdigest(), "frozen_common_clock": frozen_context.get("common_clock"), "frozen_phone_class_counts": frozen_context.get("phone_class_counts"), "array_schemas": {k: {"shape": v["shape"], "dtype": v["dtype"], "sha256": v["sha256"]} for k, v in arrays.items()}})
        included_sources.append(sidecar_path)
    require(overlay_contexts == {("S0064_BAC009S0064W0121", "natural_to_tts"), ("S0064_BAC009S0064W0121", "tts_to_natural")}, "grid_overlay_context_coverage_mismatch")

    EXPORT.mkdir(mode=0o700, parents=True, exist_ok=False)
    source_members = {}
    for path in included_sources:
        relative = path.relative_to(TRACK).as_posix()
        source_members[relative] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "metadata, clocks, identities, sidecar and seal readback only; no NPZ values, mouth metrics, scores, or effects read",
        "track_root": str(TRACK),
        "run_root": str(RUN),
        "producer_bindings": {"execution_plan_raw_sha256": EXPECTED_PLAN_RAW, "execution_plan_fingerprint_sha256": EXPECTED_PLAN_FINGERPRINT, "producer_all_files_sha256": EXPECTED_RUN_FILES, "extractor_source_sha256": EXPECTED_EXTRACTOR, "grid_inventory_sha256": EXPECTED_GRID, "visual_availability_freeze_sha256": EXPECTED_FREEZE, "visual_availability_freeze_seal_sha256": EXPECTED_FREEZE_SEAL, "visual_consumer_source_sha256": EXPECTED_CONSUMER},
        "external_input_paths": {"execution_plan": str(PLAN), "grid_inventory": str(GRID), "visual_availability_freeze": str(FREEZE), "visual_availability_freeze_seal": str(FREEZE_SEAL), "runtime_flags": str(RUNTIME_FLAGS), "producer_run": str(RUN), "consumer_output": str(CONSUMER)},
        "producer_file_seals": {"run_status_sha256": EXPECTED_RUN_STATUS, "protocol_sha256": EXPECTED_RUN_PROTOCOL, "rows_sha256": EXPECTED_RUN_ROWS, "results_sha256": EXPECTED_RUN_RESULTS, "all_files_sha256": EXPECTED_RUN_FILES, "runtime_flags_path": str(RUNTIME_FLAGS), "runtime_flags_sha256": EXPECTED_RUNTIME_FLAGS, "consumer_receipt": consumer_receipt},
        "producer_file_seals": {"run_status_sha256": EXPECTED_RUN_STATUS, "protocol_sha256": EXPECTED_RUN_PROTOCOL, "rows_sha256": EXPECTED_RUN_ROWS, "results_sha256": EXPECTED_RUN_RESULTS, "all_files_sha256": EXPECTED_RUN_FILES, "runtime_flags_path": str(RUNTIME_FLAGS), "runtime_flags_sha256": EXPECTED_RUNTIME_FLAGS, "consumer_receipt": consumer_receipt},
        "track_seals": {"protocol_sha256": sha(protocol_path), "status_sha256": sha(status_path), "all_files_sha256": sha(all_path), "seal_sha256": sha(seal_path), "progress_sha256": sha(progress_path)},
        "visual_availability_freeze": {"path": str(FREEZE), "sha256": EXPECTED_FREEZE, "seal_path": str(FREEZE_SEAL), "seal_sha256": EXPECTED_FREEZE_SEAL, "status": freeze_doc["status"], "contexts": len(freeze_contexts), "eligible_contexts": sum(bool(row.get("availability_only_endpoint", {}).get("primary_visual_endpoint_eligible")) for row in freeze_contexts.values())},
        "coverage": {"contexts": 160, "samples": 80, "speakers": 40, "videos": 1600, "complete_progress_rows": len(progress_rows), "failed_rows": 0, "all_files_entries": len(files), "sidecars_verified": len(metadata_rows), "sidecars_with_external_hash_match": len(metadata_rows), "physical_overlay_contexts": len(overlay_contexts), "overlay_samples": sorted({x[0] for x in overlay_contexts})},
        "member_hash_verification": {"verified_all_files_entries": len(files), "npz_payloads_opened": False, "npz_files_sha256_streamed": sum(1 for name in files if name.endswith(".npz")), "note": "NPZ members were hashed as opaque byte streams; no arrays or geometry values were decoded."},
        "runtime": protocol["runtime"],
        "track_status": {k: status[k] for k in ("status", "planned_videos", "completed_videos", "failed_videos", "progress_sha256", "finished_utc")},
        "track_seal": {k: seal[k] for k in ("status", "protocol_sha256", "status_sha256", "all_files_sha256", "file_count", "completed_videos", "failed_videos")},
        "sidecar_required_fields": sorted(required_meta),
        "sidecar_observed_fields": sorted(sidecar_keys),
        "array_schema_histograms": {name: dict(counts) for name, counts in array_schemas.items()},
        "video_metadata_rows": metadata_rows,
        "source_member_manifest": source_members,
    }
    report_path = EXPORT / "track_metadata_inventory.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    source_members["track_metadata_inventory.json"] = {"bytes": report_path.stat().st_size, "sha256": sha(report_path)}
    tar_path = EXPORT / "track_metadata_bundle.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as archive:
        for path in included_sources:
            archive.add(path, arcname=path.relative_to(TRACK).as_posix(), recursive=False)
        archive.add(report_path, arcname="track_metadata_inventory.json", recursive=False)
    archive_sha = sha(tar_path)
    archive_bytes = tar_path.stat().st_size
    parts = []
    with tar_path.open("rb") as source:
        i = 0
        while block := source.read(CHUNK_BYTES):
            part = EXPORT / f"track_metadata_bundle.tar.gz.part{i:03d}"
            with part.open("xb") as output:
                output.write(block)
                output.flush()
                os.fsync(output.fileno())
            parts.append({"path": str(part), "name": part.name, "bytes": part.stat().st_size, "sha256": sha(part)})
            i += 1
    manifest = {"schema_version": 1, "status": "SEALED", "archive_path": str(tar_path), "archive_sha256": archive_sha, "archive_bytes": archive_bytes, "chunk_bytes": CHUNK_BYTES, "parts": parts, "member_count": len(source_members), "source_member_manifest_sha256": hashlib.sha256(canonical(source_members)).hexdigest(), "audit_report_path": str(report_path), "audit_report_sha256": sha(report_path), "track_all_files_sha256": sha(all_path), "track_seal_sha256": sha(seal_path), "track_progress_sha256": sha(progress_path), "created_utc_epoch": int(time.time())}
    manifest_path = EXPORT / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "METADATA_READBACK_PASS", "report_path": str(report_path), "report_sha256": sha(report_path), "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path), "archive_bytes": archive_bytes, "archive_sha256": archive_sha, "chunks": len(parts), "members": len(source_members), "contexts": 160, "videos": 1600, "all_files_entries": len(files), "npz_values_read": False}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
