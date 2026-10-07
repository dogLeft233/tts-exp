#!/usr/bin/env python3
"""Create a fail-closed fit40 V3 launch binding from completed native RMS artifacts.

This tool validates already-produced fit-only artifacts and writes an argv/env
binding. It does not import a model, initialize CUDA, render, score, or launch
the optimizer. All producer paths and raw hashes are resolved at invocation
time from the sealed static configuration and the terminal producer receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


CONFIG_SCHEMA = "cross_tfg_fit40_transition_head_v3_late_binding_attempt10_config_v1"
BUNDLE_SCHEMA = "cross_tfg_fit40_transition_head_v3_source_bundle_v1"
OUTPUT_SCHEMA = "cross_tfg_fit40_transition_head_v3_late_binding_attempt10_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_SOURCE_MANIFEST_BYTES = 61469
EXPECTED_HEAD_CONFIG_SHA256 = "0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5"
EXPECTED_HEAD_CONFIG_BYTES = 10721
EXPECTED_PRODUCER_SHA256 = "3438935a4cdfcfc4487f25308ce53e102d247dd77540185b14ed2b70cae26600"
EXPECTED_PRODUCER_BYTES = 63227
EXPECTED_RUNNER_SHA256 = "25872da29e8ef6dd914ec046ee479f00eeb2e491a602fb6e8b1bcf6804484c49"
EXPECTED_RUNNER_BYTES = 84458
EXPECTED_TARGET_RMS = 0.0376838172675746
EXPECTED_ROW_IDS = [
    f"S{speaker:04d}_j{slot:02d}_" for speaker in range(52, 72) for slot in (0, 15)
]


class BindingError(RuntimeError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))


def fail(message: str) -> None:
    raise BindingError(message)


def read_json(path: Path, *, expected_bytes: int | None = None, expected_sha256: str | None = None) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        fail(f"refusing symlink JSON input: {path}")
    resolved = path.resolve(strict=True)
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        fail(f"expected regular JSON input: {resolved}")
    raw = resolved.read_bytes()
    if expected_bytes is not None and len(raw) != int(expected_bytes):
        fail(f"JSON byte count differs from pinned value: {resolved}")
    if expected_sha256 is not None and sha_bytes(raw) != expected_sha256.lower():
        fail(f"JSON raw SHA-256 differs from pinned value: {resolved}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid UTF-8 JSON at {resolved}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected JSON object: {resolved}")
    return value, raw


def pin_regular(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    if path.is_symlink():
        fail(f"refusing symlink input: {path}")
    resolved = path.resolve(strict=True)
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        fail(f"expected regular file: {resolved}")
    digest = sha_file(resolved)
    result = {"path": str(resolved), "bytes": st.st_size, "sha256": digest}
    if expected is not None:
        if st.st_size != int(expected.get("bytes", -1)) or digest != str(expected.get("sha256", "")).lower():
            fail(f"file differs from pinned artifact record: {resolved}")
    return result


def is_within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def safe_member(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts or "\\" in relative:
        fail(f"unsafe relative path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    current = resolved_root
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            fail(f"symlink in input path: {current}")
    resolved = current.resolve(strict=True)
    if not is_within(resolved, resolved_root) or not resolved.is_file():
        fail(f"relative input escaped root or is not a file: {resolved}")
    return resolved


def expected_source_rows(source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if source_manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1":
        fail("source manifest schema is not the frozen V5 fit40 source transfer manifest")
    if source_manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        fail("source manifest is not a prepared, fit-only source roster")
    if source_manifest.get("split") != "historical_fit_only":
        fail("source manifest top-level split is not historical_fit_only")
    scope = source_manifest.get("scope")
    if not isinstance(scope, dict) or scope.get("dev_audio_read") is not False or scope.get("confirm_audio_read") is not False:
        fail("source manifest scope opened development or confirmation audio")
    rows = source_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        fail("source manifest must contain exactly 40 fit rows")
    expected_pairs = [(f"S{speaker:04d}", slot) for speaker in range(52, 72) for slot in (0, 15)]
    pairs = [(str(row.get("speaker_id")), int(row.get("canonical_j", -1))) for row in rows]
    if pairs != expected_pairs:
        fail("source manifest is not the frozen 20-speaker canonical j=0/j=15 roster in order")
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if len(set(row_ids)) != 40 or any(not row_id.startswith(prefix) for row_id, prefix in zip(row_ids, EXPECTED_ROW_IDS)):
        fail("source manifest row IDs are duplicate or differ from the canonical speaker/endpoint keys")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(sample_ids) or len(set(sample_ids)) != 40:
        fail("source manifest sample IDs are empty or duplicate")
    for row in rows:
        if row.get("source_encoding") != "PCM_16":
            fail(f"non-PCM16 source row rejected: {row.get('row_id')}")
    return rows


def verify_source_audio(source_manifest: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    resolved_root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not resolved_root.is_dir():
        fail("fit40 source root must be a real directory")
    pins = []
    for row in source_manifest["rows"]:
        relative = str(row.get("transfer_relative_path", ""))
        path = safe_member(resolved_root, relative)
        record = pin_regular(path)
        if record["bytes"] != int(row.get("transfer_file_bytes", -1)):
            fail(f"source WAV byte count differs from the sealed source row: {row['row_id']}")
        if record["sha256"] != str(row.get("transfer_file_sha256", "")).lower():
            fail(f"source WAV SHA differs from the sealed source row: {row['row_id']}")
        if int(row.get("sample_rate_hz", -1)) != 16000 or int(row.get("channels", -1)) != 1:
            fail(f"source WAV metadata is not mono 16 kHz: {row['row_id']}")
        if int(row.get("sample_count", -1)) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_pcm_payload_sha256", ""))):
            fail(f"source PCM16 payload identity is incomplete: {row['row_id']}")
        pins.append({"row_id": str(row["row_id"]), **record, "source_pcm_payload_sha256": str(row["source_pcm_payload_sha256"]).lower()})
    return pins


def verify_bundle(config: dict[str, Any], code_root: Path, source_root: Path) -> dict[str, Any]:
    stage_ref = config["source_stage_receipt"]
    stage_path = Path(stage_ref["path"])
    stage, stage_raw = read_json(stage_path, expected_sha256=stage_ref["expected_raw_sha256"])
    if stage.get("schema") != "cross_tfg_fit40_native_transition_head_source_stage_receipt_v1" or stage.get("status") != "PASS_SOURCE_ONLY_FIT40_V3_PAYLOAD_EXTRACTED":
        fail("fit120 source stage receipt is not a successful, fit-only source extraction")
    bundle = config["bundle"]
    if stage.get("destination") != str(code_root.resolve(strict=True)):
        fail("source stage receipt extraction destination differs from the configured code root")
    if stage.get("archive", {}).get("bytes") != int(bundle["archive_bytes"]) or stage.get("archive", {}).get("sha256") != bundle["archive_sha256"]:
        fail("source stage receipt archive binding differs from the frozen source ZIP")
    if stage.get("bundle_manifest", {}).get("bytes") != int(bundle["manifest_bytes"]) or stage.get("bundle_manifest", {}).get("sha256") != bundle["manifest_sha256"]:
        fail("source stage receipt manifest binding differs from the frozen source manifest")
    manifest_path = code_root / "payload_manifest.json"
    bundle_manifest, bundle_manifest_raw = read_json(
        manifest_path,
        expected_bytes=int(bundle["manifest_bytes"]),
        expected_sha256=str(bundle["manifest_sha256"]),
    )
    if bundle_manifest.get("schema") != BUNDLE_SCHEMA:
        fail("unsupported fit120 source bundle manifest schema")
    members = bundle_manifest.get("members")
    extracted = stage.get("members")
    if not isinstance(members, list) or not isinstance(extracted, list):
        fail("bundle member records are absent")
    expected_by_archive: dict[str, dict[str, Any]] = {}
    for member in members:
        name = str(member.get("archive_path", ""))
        if not name.startswith("payload/") or name in expected_by_archive:
            fail(f"invalid or duplicate bundle member: {name}")
        expected_by_archive[name] = member
    actual_by_archive = {str(member.get("archive_path")): member for member in extracted}
    if len(actual_by_archive) != len(extracted) or set(actual_by_archive) != set(expected_by_archive):
        fail("extracted source member set differs from the sealed bundle manifest")
    actual_source_files = []
    for name, member in sorted(expected_by_archive.items()):
        staged = actual_by_archive[name]
        relative = str(PurePosixPath(name).relative_to("payload"))
        path = safe_member(code_root, relative)
        pin = pin_regular(path, {"bytes": int(member["bytes"]), "sha256": str(member["sha256"]).lower()})
        if str(Path(staged.get("path", "")).resolve(strict=True)) != pin["path"]:
            fail(f"stage receipt path differs from extracted source member: {name}")
        if int(staged.get("bytes", -1)) != pin["bytes"] or str(staged.get("sha256", "")).lower() != pin["sha256"]:
            fail(f"stage receipt member pin differs from source bundle: {name}")
        actual_source_files.append({"archive_path": name, **pin})
    return {
        "stage_receipt": {"path": str(stage_path.resolve(strict=True)), "bytes": len(stage_raw), "sha256": sha_bytes(stage_raw)},
        "bundle_manifest": {"path": str(manifest_path.resolve(strict=True)), "bytes": len(bundle_manifest_raw), "sha256": sha_bytes(bundle_manifest_raw)},
        "members": actual_source_files,
    }


def validate_producer(config: dict[str, Any], source_rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    paths = config["producer"]
    receipt_path = Path(paths["execution_receipt_path"])
    receipt, receipt_raw = read_json(receipt_path)
    if receipt.get("schema") != "cross_tfg_fit40_native_rms_geometry_execution_receipt_v1":
        fail("fit40 geometry execution receipt schema mismatch")
    if receipt.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" or receipt.get("child_return_code") != 0 or receipt.get("child_killed_reason") is not None:
        fail("fit40 geometry producer was not a successful terminal run")
    if int(receipt.get("completed_rows", -1)) != 40 or int(receipt.get("expected_rows", -1)) != 40:
        fail("fit40 geometry producer did not complete its full 40-row denominator")
    if receipt.get("model_score_computed") is not False or receipt.get("training_performed") is not False or receipt.get("development_audio_opened") is not False or receipt.get("confirmation_audio_opened") is not False:
        fail("producer scope is not the frozen fit-only, no-score/no-training scope")
    ledger = receipt.get("row_ledger")
    expected_ids = [str(row["row_id"]) for row in source_rows]
    if not isinstance(ledger, list) or [str(row.get("row_id")) for row in ledger] != expected_ids or any(row.get("status") != "PASS_ROW_GEOMETRY" for row in ledger):
        fail("producer terminal row ledger is not all 40 expected rows PASS_ROW_GEOMETRY")
    if not isinstance(receipt.get("gpu_admission_samples"), list) or len(receipt["gpu_admission_samples"]) != 2 or any(row.get("status") != "PASS" for row in receipt["gpu_admission_samples"]):
        fail("producer receipt lacks two successful GPU admission samples")
    final_state = receipt.get("final_producer_state")
    if not isinstance(final_state, dict) or final_state.get("schema") != "cross_tfg_fit40_rms_geometry_worker_state_v2" or final_state.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
        fail("producer receipt does not contain a successful final worker state")
    state_path = Path(paths["worker_state_path"])
    state, state_raw = read_json(state_path)
    if state != final_state:
        fail("on-disk terminal producer state differs from the supervisor’s final state")
    geometry_pin = receipt.get("geometry_manifest")
    cache_pin = receipt.get("natural_audio_cache_manifest")
    if not isinstance(geometry_pin, dict) or not isinstance(cache_pin, dict):
        fail("producer receipt lacks geometry/cache manifest pins")
    if final_state.get("geometry_manifest") != geometry_pin or final_state.get("audio_cache_manifest") != cache_pin:
        fail("worker state and supervisor receipt manifest pins differ")
    geometry_path = Path(paths["geometry_manifest_path"])
    geometry, geometry_raw = read_json(
        geometry_path,
        expected_bytes=int(geometry_pin.get("bytes", -1)),
        expected_sha256=str(geometry_pin.get("sha256", "")),
    )
    if str(geometry_path.resolve(strict=True)) != str(Path(str(geometry_pin.get("path", ""))).resolve(strict=True)):
        fail("geometry manifest path differs from producer receipt")
    if geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1" or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
        fail("geometry manifest is not a completed fit-only artifact")
    geometry_payload = {key: value for key, value in geometry.items() if key != "payload_sha256"}
    if geometry.get("payload_sha256") != canonical_sha(geometry_payload):
        fail("geometry manifest canonical payload digest mismatch")
    if int(geometry.get("fit_rows", -1)) != 40 or geometry.get("method_id") != "rms_fixed":
        fail("geometry manifest denominator or baseline method differs")
    if (int(geometry.get("source_clock_fps_num", -1)), int(geometry.get("source_clock_fps_den", -1))) != (25, 1):
        fail("geometry manifest source clock differs from 25/1")
    if geometry.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries":
        fail("geometry manifest changes the original PCM/MFCC query clock")
    baseline = geometry.get("baseline_control", {})
    if float(baseline.get("target_rms", -1.0)) != EXPECTED_TARGET_RMS or baseline.get("peak_ceiling") is not None or baseline.get("peak_limited") is not False:
        fail("geometry artifact is not exact unbounded fixed-RMS baseline")
    cache_path = Path(paths["cache_manifest_path"])
    cache, cache_raw = read_json(
        cache_path,
        expected_bytes=int(cache_pin.get("bytes", -1)),
        expected_sha256=str(cache_pin.get("sha256", "")),
    )
    if str(cache_path.resolve(strict=True)) != str(Path(str(cache_pin.get("path", ""))).resolve(strict=True)):
        fail("audio cache manifest path differs from producer receipt")
    cache_payload = {key: value for key, value in cache.items() if key != "manifest_sha256"}
    if cache.get("schema") != "cross_tfg_syncnet_full_audio_embedding_cache_manifest_v1" or int(cache.get("row_count", -1)) != 40 or cache.get("manifest_sha256") != canonical_sha(cache_payload):
        fail("natural-audio cache manifest schema/count/canonical digest mismatch")
    geometry_rows = geometry.get("rows")
    cache_rows = cache.get("rows")
    if not isinstance(geometry_rows, list) or not isinstance(cache_rows, list) or len(geometry_rows) != 40 or len(cache_rows) != 40:
        fail("geometry or audio-cache row denominator is not forty")
    ids = [str(row.get("row_id")) for row in geometry_rows]
    cache_ids = [str(row.get("row_id")) for row in cache_rows]
    if ids != expected_ids or cache_ids != expected_ids or len(set(ids)) != 40:
        fail("geometry/cache row IDs or order differ from the canonical fit40 roster")
    if geometry.get("source_pins", {}).get("source_manifest", {}).get("sha256") != EXPECTED_SOURCE_MANIFEST_SHA256:
        fail("geometry source pin differs from the frozen fit40 transfer manifest")
    parent_plan_pin = config.get("source_pins", {}).get("parent_plan", {})
    if geometry.get("source_pins", {}).get("v2_runtime_plan", {}).get("sha256") != parent_plan_pin.get("sha256"):
        fail("geometry source pin differs from the fresh context-bound V2 parent plan")
    producer_pin = geometry.get("source_pins", {}).get("geometry_producer", {})
    if int(producer_pin.get("bytes", -1)) != EXPECTED_PRODUCER_BYTES or str(producer_pin.get("sha256", "")).lower() != EXPECTED_PRODUCER_SHA256:
        fail("geometry manifest does not bind the frozen fit40 geometry producer source")
    identity = geometry.get("runtime_gpu_identity")
    if not isinstance(identity, dict) or identity.get("schema") != "cross_tfg_runtime_gpu_identity_v1":
        fail("geometry manifest lacks actual producer runtime GPU identity")
    index = int(identity.get("device_index", -1))
    uuid = str(identity.get("uuid", "")).strip().lower()
    device_literal = str(identity.get("device_literal", ""))
    if index < 0 or not re.fullmatch(r"[a-zA-Z0-9-]{8,64}", uuid) or device_literal not in ("cuda", f"cuda:{index}"):
        fail("producer GPU UUID/device binding is malformed")
    profile = identity.get("runtime_profile")
    if not isinstance(profile, dict):
        fail("producer runtime GPU identity has no exact runtime profile")
    if int(profile.get("seed", -1)) != int(plan.get("full_context_fit_v2", {}).get("fit_seed", -2)) or int(profile.get("seed", -1)) != 20261003:
        fail("producer and frozen fit seed differ")
    if profile.get("frame_rate") != {"numerator": 25, "denominator": 1}:
        fail("producer runtime profile does not preserve 25/1 frame clock")
    for row in geometry_rows:
        if row.get("runtime_gpu_identity") != identity:
            fail(f"row GPU identity differs from producer identity: {row.get('row_id')}")
        admission = row.get("runtime_gpu_row_admission")
        if not isinstance(admission, dict) or admission.get("status") != "PASS" or admission.get("selected_gpu", {}).get("uuid") != identity.get("uuid") or admission.get("other_compute_apps"):
            fail(f"row does not carry a successful same-GPU admission record: {row.get('row_id')}")
        mapping = row.get("prediction_map_evidence")
        if not isinstance(mapping, dict) or not isinstance(mapping.get("entries"), list) or len(mapping["entries"]) != 32:
            fail(f"row lacks 32 measured prediction-frame mappings: {row.get('row_id')}")
        if len(row.get("output_source_clock_frames", [])) != 32 or len(row.get("track_local_frame_indices", [])) != 32 or len(row.get("prediction_source_clock_frames", [])) != 32:
            fail(f"row geometry does not contain all 32 source/track/prediction indices: {row.get('row_id')}")
        if int(row.get("query_start_frame", -1)) < 15:
            fail(f"row query does not leave full 31-lag support: {row.get('row_id')}")
        # Detailed PCM decode, cache byte verification and renderer parity are repeated by the fit runner before optimizer updates.
    for cache_row in cache_rows:
        root = cache_path.parent.resolve(strict=True)
        for path_key, size_key, sha_key in (("npz_path", "npz_bytes", "npz_sha256"), ("metadata_path", None, "metadata_sha256")):
            path = Path(str(cache_row.get(path_key, "")))
            if path.is_symlink():
                fail(f"cache member is a symlink: {path}")
            resolved = path.resolve(strict=True)
            if not is_within(resolved, root) or not resolved.is_file():
                fail(f"cache member escapes its cache directory: {resolved}")
            pin = pin_regular(resolved)
            expected_bytes = cache_row.get(size_key) if size_key is not None else None
            if (expected_bytes is not None and pin["bytes"] != int(expected_bytes)) or pin["sha256"] != str(cache_row.get(sha_key, "")).lower():
                fail(f"audio cache member does not match manifest pin: {resolved}")
    return {
        "producer_receipt": {"path": str(receipt_path.resolve(strict=True)), "bytes": len(receipt_raw), "sha256": sha_bytes(receipt_raw)},
        "producer_state": {"path": str(state_path.resolve(strict=True)), "bytes": len(state_raw), "sha256": sha_bytes(state_raw)},
        "geometry_manifest": {"path": str(geometry_path.resolve(strict=True)), "bytes": len(geometry_raw), "sha256": sha_bytes(geometry_raw), "payload_sha256": geometry["payload_sha256"]},
        "audio_cache_manifest": {"path": str(cache_path.resolve(strict=True)), "bytes": len(cache_raw), "sha256": sha_bytes(cache_raw), "manifest_sha256": cache["manifest_sha256"]},
        "runtime_gpu_identity": identity,
        "geometry_row_count": len(geometry_rows),
        "audio_cache_row_count": len(cache_rows),
    }


def make_binding(config: dict[str, Any], *, config_path: Path, config_raw: bytes, stage_receipt_sha: str) -> dict[str, Any]:
    code_root = Path(config["remote_paths"]["code_root"])
    repo_root = code_root / "src"
    control_root = Path(config["remote_paths"]["control_root"])
    source_root = Path(config["remote_paths"]["source_root"])
    paths = config["remote_paths"]
    plan_path = Path(paths["parent_plan_path"])
    if not is_within(plan_path.resolve(strict=True), repo_root.resolve(strict=True)):
        fail("context-bound V2 parent plan must be staged inside the fresh source-code root")
    source_manifest_path = repo_root / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json"
    head_config_path = repo_root / "runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json"
    runner_path = repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt10.py"
    producer_path = repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt05.py"
    late_binder_path = repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/late_bind_fit40_native_transition_head_v3_attempt10.py"
    parent_plan_pin = config["source_pins"]["parent_plan"]
    plan, plan_raw = read_json(plan_path, expected_bytes=int(parent_plan_pin["bytes"]), expected_sha256=str(parent_plan_pin["sha256"]))
    parent_plan_payload_sha256 = str(parent_plan_pin["payload_sha256"]).lower()
    plan_payload = {key: value for key, value in plan.items() if key != "plan_payload_sha256"}
    if plan.get("plan_payload_sha256") != parent_plan_payload_sha256 or canonical_sha(plan_payload) != parent_plan_payload_sha256:
        fail("fresh context-bound V2 parent plan canonical payload digest mismatch")
    source_manifest, source_raw = read_json(source_manifest_path, expected_bytes=EXPECTED_SOURCE_MANIFEST_BYTES, expected_sha256=EXPECTED_SOURCE_MANIFEST_SHA256)
    if source_manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1" or source_manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        fail("fit40 source manifest is not a materialized fit-only source roster")
    scope = source_manifest.get("scope", {})
    for key in ("candidate_transform_executed", "confirm_audio_read", "dev_audio_read", "model_forward", "render", "score"):
        if scope.get(key) is not False:
            fail(f"fit40 source manifest scope includes prohibited prior effect access: {key}")
    rows = expected_source_rows(source_manifest)
    source_audio_pins = verify_source_audio(source_manifest, source_root)
    head_config, head_config_raw = read_json(head_config_path, expected_bytes=EXPECTED_HEAD_CONFIG_BYTES, expected_sha256=EXPECTED_HEAD_CONFIG_SHA256)
    constructor = head_config.get("head_constructor", {})
    if head_config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1" or constructor.get("target_rms") != EXPECTED_TARGET_RMS or constructor.get("peak_ceiling") is not None:
        fail("V3 constructor config is not the exact qualified unbounded-RMS constructor")
    source_stage_receipt_path = Path(config["source_stage_receipt"]["path"])
    config["source_stage_receipt"]["expected_raw_sha256"] = stage_receipt_sha
    bundle_qc = verify_bundle(config, code_root, source_root)
    for source_path, expected_bytes, expected_sha in (
        (runner_path, EXPECTED_RUNNER_BYTES, EXPECTED_RUNNER_SHA256),
        (producer_path, EXPECTED_PRODUCER_BYTES, EXPECTED_PRODUCER_SHA256),
        (late_binder_path, config["source_pins"]["late_binder"]["bytes"], config["source_pins"]["late_binder"]["sha256"]),
    ):
        actual = pin_regular(source_path)
        if actual["bytes"] != int(expected_bytes) or actual["sha256"] != str(expected_sha).lower():
            fail(f"training/producer/late-binder source differs from exact frozen pin: {source_path}")
    producer = validate_producer(config, rows, plan)
    geometry = Path(producer["geometry_manifest"]["path"])
    output = Path(paths["training_output_dir"])
    if output.exists() or output.is_symlink():
        fail("fit120 output directory already exists; refusing reuse or resume")
    if not output.parent.is_dir() or output.parent.is_symlink():
        fail("fit120 output parent must already exist as a real directory")
    device = str(producer["runtime_gpu_identity"]["device_literal"])
    profile = producer["runtime_gpu_identity"]["runtime_profile"]
    if profile.get("cuda_visible_devices") in (None, ""):
        fail("producer runtime identity has no CUDA_VISIBLE_DEVICES value")
    strict_profile = {
        "seed": 20261003, "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "torch_use_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    }
    for key, expected in strict_profile.items():
        if profile.get(key) != expected:
            fail(f"producer runtime does not match the strict QC06 deterministic policy: {key}")
    environment = {
        "CROSS_TFG_PSF_ROOT": str(plan["runtime"]["execution"]["python_speech_features_root"]),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "20261003",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(profile["cuda_visible_devices"]),
    }
    argv = [
        str(plan["runtime"]["execution"]["python_executable_literal"]), "-B", str(runner_path),
        "--parent-plan", str(plan_path), "--expected-parent-plan-sha256", str(parent_plan_pin["sha256"]),
        "--expected-parent-plan-payload-sha256", parent_plan_payload_sha256,
        "--source-manifest", str(source_manifest_path), "--expected-source-manifest-sha256", EXPECTED_SOURCE_MANIFEST_SHA256,
        "--source-root", str(source_root),
        "--geometry-manifest", str(geometry), "--expected-geometry-manifest-sha256", producer["geometry_manifest"]["sha256"],
        "--head-config", str(head_config_path), "--expected-head-config-sha256", EXPECTED_HEAD_CONFIG_SHA256,
        "--output-dir", str(output), "--device", device,
    ]
    qualification_only = config.get("qualification_only") is True
    if qualification_only:
        argv.append("--qualification-only")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "READY_FIT_ONLY_STRICT_BACKWARD_QUALIFICATION_BINDING" if qualification_only else "READY_FIT_ONLY_TRAINING_LAUNCH_BINDING",
        "scope": {
            "fit_rows": 40, "fit_speakers": 20, "canonical_endpoints": [0, 15],
            "optimizer_steps": 120, "pass_count": 3, "seed": 20261003,
            "qualification_only": qualification_only,
            "development_audio_opened": False, "confirmation_audio_opened": False,
            "model_selection": False, "peak_ceiling": None,
            "determinism_policy": {
                "torch_use_deterministic_algorithms": True,
                "torch_num_threads": 1, "torch_num_interop_threads": 1,
                "cublas_workspace_config": ":4096:8",
                "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                "cudnn_benchmark": False, "cudnn_deterministic": True,
                "pythonhashseed": "20261003",
                "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
            },
        },
        "config": {"path": str(config_path.resolve(strict=True)), "bytes": len(config_raw), "sha256": sha_bytes(config_raw)},
        "source_stage": bundle_qc,
        "parent_plan": {"path": str(plan_path.resolve(strict=True)), "bytes": len(plan_raw), "sha256": sha_bytes(plan_raw), "payload_sha256": parent_plan_payload_sha256},
        "fit_source_manifest": {"path": str(source_manifest_path.resolve(strict=True)), "bytes": len(source_raw), "sha256": sha_bytes(source_raw), "rows": source_audio_pins},
        "head_config": {"path": str(head_config_path.resolve(strict=True)), "bytes": len(head_config_raw), "sha256": sha_bytes(head_config_raw), "constructor": constructor},
        "fit40_geometry_and_cache": producer,
        "source_bundle_member_count": len(bundle_qc["members"]),
        "runner": {"path": str(runner_path.resolve(strict=True)), "bytes": EXPECTED_RUNNER_BYTES, "sha256": EXPECTED_RUNNER_SHA256},
        "producer_source": {"path": str(producer_path.resolve(strict=True)), "bytes": EXPECTED_PRODUCER_BYTES, "sha256": EXPECTED_PRODUCER_SHA256},
        "device": {"literal": device, "uuid": producer["runtime_gpu_identity"]["uuid"], "index": producer["runtime_gpu_identity"]["device_index"]},
        "resource_gate_required_before_launch": {
            "stable_samples": 2, "sample_interval_seconds": 5,
            "minimum_free_gpu_mib": 18432, "unknown_compute_pids": "wait; never terminate other processes",
            "minimum_shm_free_bytes": 4294967296,
            "same_gpu_uuid_as_geometry_producer": producer["runtime_gpu_identity"]["uuid"],
        },
        "working_directory": str(repo_root.resolve(strict=True)),
        "environment": environment,
        "argv": argv,
        "training_output_dir": str(output),
        "binding_written_at": str(Path(paths["binding_output_path"]).resolve(strict=False)),
        "execution_started": False,
        "optimizer_created_or_updated": False,
        "note": "This binding does not execute the runner. Qualification-only runs complete the same all-40 same-GPU zero gate and full-objective strict backward check but create no optimizer; training runs preserve the full 40-row/120-update protocol and repeat the checks before their first optimizer update.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-stage-receipt-sha256", required=True)
    parser.add_argument("--write-binding", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_config_sha256.lower()):
            fail("config SHA must be a 64-character lowercase hexadecimal digest")
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_stage_receipt_sha256.lower()):
            fail("stage receipt SHA must be a 64-character lowercase hexadecimal digest")
        config, config_raw = read_json(args.config, expected_sha256=args.expected_config_sha256)
        if config.get("schema") != CONFIG_SCHEMA:
            fail("unsupported static late-binding config schema")
        binding = make_binding(config, config_path=args.config, config_raw=config_raw, stage_receipt_sha=args.expected_source_stage_receipt_sha256.lower())
        binding_path = Path(config["remote_paths"]["binding_output_path"])
        if not args.write_binding:
            print(json.dumps(binding, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        if binding_path.exists() or binding_path.is_symlink():
            fail(f"refusing to overwrite an existing launch binding: {binding_path}")
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with binding_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"schema": OUTPUT_SCHEMA, "status": binding["status"], "binding": {"path": str(binding_path), "bytes": len(raw), "sha256": sha_bytes(raw)}, "execution_started": False}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"schema": OUTPUT_SCHEMA, "status": "NOT_READY", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
