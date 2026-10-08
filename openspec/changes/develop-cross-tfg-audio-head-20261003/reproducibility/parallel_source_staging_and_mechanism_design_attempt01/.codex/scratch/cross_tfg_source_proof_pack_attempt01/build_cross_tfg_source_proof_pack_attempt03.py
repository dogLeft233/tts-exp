#!/usr/bin/env python3
"""Create a sealed, metadata-first inventory for the cross-TFG handoff.

Default operation is --inventory: read and hash only the explicitly listed
small source/evidence files, then create one self-hashed JSON index with O_EXCL.
Optional --copy-to is a separate create-only operation that copies only entries
marked copy=true in an existing sealed index. Dataset/media archives and large
ledgers are references only and are never copied by this utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "cross_tfg_source_proof_pack_index_attempt03.json"
TRANSCRIPT_METADATA_PATH = HERE / "aishell_transcript_hub_metadata_attempt01.json"
MAX_COPY_BYTES = 2_000_000
COPY_SUFFIXES = {".json", ".py", ".md", ".diff"}
COPY_BASENAME_EXCEPTIONS = {"execution_record", "result_manifest", "run_status"}
SCHEMA = "cross_tfg_source_proof_pack_index_v1"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = dict(value)
    clone.pop(field, None)
    return sha256_bytes(canonical_bytes(clone))


def write_exclusive(path: Path, raw: bytes) -> None:
    assert_workspace_path(path, allow_missing_leaf=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_workspace_path(path, allow_missing_leaf=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def relative_parts(value: str) -> tuple[str, ...]:
    """Accept only normalized, workspace-relative paths with no traversal."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"invalid relative path: {value!r}")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise RuntimeError(f"absolute or drive-qualified path rejected: {value!r}")
    normalized = value.replace("\\", "/")
    parts = tuple(normalized.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"empty, dot, or traversal path component rejected: {value!r}")
    return parts


def assert_workspace_path(path: Path, *, allow_missing_leaf: bool = False) -> Path:
    """Require lexical and resolved containment under ROOT and reject symlink components."""
    root = ROOT.resolve(strict=True)
    candidate = path if path.is_absolute() else ROOT / path
    try:
        lexical = candidate.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"path is outside workspace: {path}") from exc
    if any(component in (".", "..") for component in lexical.parts):
        raise RuntimeError(f"dot or traversal component rejected: {path}")
    current = ROOT
    components = lexical.parts
    for index, component in enumerate(components):
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"symlink path component rejected: {current}")
        if not current.exists() and not (allow_missing_leaf and index == len(components) - 1):
            # Intermediate missing folders are allowed only for an output path;
            # the path is still checked lexically and again after creation.
            if allow_missing_leaf:
                continue
            raise FileNotFoundError(current)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"resolved path escapes workspace: {path}") from exc
    return candidate


def workspace_relative_file(rel: str) -> Path:
    parts = relative_parts(rel)
    path = ROOT.joinpath(*parts)
    assert_workspace_path(path)
    return path


TRANSCRIPT_METADATA = {
    "schema": "aishell_transcript_metadata_only_v1",
    "source_role": "public third-party Hugging Face AISHELL-1 mirror metadata",
    "repository": "shenyunhang/AISHELL-1",
    "pinned_commit": "2724409d538167445e43ebf846990319f12a1cbf",
    "path": "data_aishell/transcript/aishell_transcript_v0.8.txt",
    "bytes": 10091431,
    "git_blob_oid_sha1": "05217b79e3df3f9ea07f667bf028e7b1a4e39d2a",
    "sha256": None,
    "sha256_unavailable_from_metadata": True,
    "metadata_observation": {
        "tree_endpoint": "https://huggingface.co/api/datasets/shenyunhang/AISHELL-1/tree/2724409d538167445e43ebf846990319f12a1cbf?recursive=true",
        "immutable_file_head_status": 200,
        "immutable_file_head_content_length": 10091431,
        "immutable_file_head_etag": "05217b79e3df3f9ea07f667bf028e7b1a4e39d2a",
        "immutable_file_head_repo_commit": "2724409d538167445e43ebf846990319f12a1cbf",
        "x_linked_etag": None,
    },
    "scope": {
        "transcript_body_downloaded_or_opened": False,
        "transcript_rows_or_text_read": False,
        "confirm_audio_or_text_accessed": False,
        "asr_called": False,
    },
    "integrity_note": "The SHA-1 is the Hub Git blob object id from metadata; it is not a SHA-256 claim and this mirror metadata does not authenticate the upstream corpus publisher.",
}


def ensure_transcript_metadata() -> Path:
    raw = (json.dumps(TRANSCRIPT_METADATA, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TRANSCRIPT_METADATA_PATH.exists():
        if TRANSCRIPT_METADATA_PATH.read_bytes() != raw:
            raise RuntimeError("existing transcript metadata descriptor differs; preserving it")
    else:
        write_exclusive(TRANSCRIPT_METADATA_PATH, raw)
    return TRANSCRIPT_METADATA_PATH


class Inventory:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.external_refs: list[dict[str, Any]] = []

    def add_file(self, rel: str, group: str, *, copy: bool,
                 role: str, note: str = "") -> None:
        parts = relative_parts(rel)
        key = PurePosixPath(*parts).as_posix()
        path = workspace_relative_file(key)
        if not path.is_file():
            raise FileNotFoundError(f"required regular file is missing or symlinked: {rel}")
        if key in self.entries:
            prior = self.entries[key]
            if prior["copy"] != copy:
                raise RuntimeError(f"conflicting copy policy for {key}")
            return
        size = path.stat().st_size
        if copy:
            if (path.suffix.lower() not in COPY_SUFFIXES and path.name not in COPY_BASENAME_EXCEPTIONS) or size > MAX_COPY_BYTES:
                raise RuntimeError(f"copy policy rejected non-small text/evidence file: {rel} ({size} bytes)")
        self.entries[key] = {
            "relative_path": key,
            "absolute_source_path": str(path.resolve()),
            "bytes": size,
            "sha256": sha256_file(path),
            "copy": copy,
            "group": group,
            "role": role,
            "note": note,
        }

    def add_tree(self, rel_root: str, group: str, *, name_prefix: str = "",
                 copy: bool = True, suffixes: set[str] | None = None) -> None:
        base = workspace_relative_file(rel_root)
        if not base.is_dir():
            raise FileNotFoundError(f"required directory is missing: {rel_root}")
        wanted_suffixes = suffixes or COPY_SUFFIXES
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix.lower() not in wanted_suffixes:
                continue
            if name_prefix and not path.name.startswith(name_prefix):
                continue
            rel = path.relative_to(ROOT).as_posix()
            self.add_file(rel, group, copy=copy, role="request_or_response_evidence")

    def add_external(self, *, identifier: str, group: str, path: str,
                     bytes_count: int | None, sha256: str | None,
                     provenance: str, note: str) -> None:
        self.external_refs.append({
            "identifier": identifier,
            "group": group,
            "path": path,
            "bytes": bytes_count,
            "sha256": sha256,
            "provenance": provenance,
            "copy": False,
            "note": note,
        })


def latest_observer_snapshot(inv: Inventory) -> dict[str, Any]:
    base_rel = ".codex/scratch/dev360_candidate180_status_observer_attempt10"
    polls = ROOT / Path(base_rel) / "polls"
    candidates = []
    for obs_path in polls.glob("*/observation.json"):
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        stamp_text = str(obs.get("observed_at_utc", ""))
        if not stamp_text:
            continue
        stamp = datetime.fromisoformat(stamp_text.replace("Z", "+00:00"))
        candidates.append((stamp, obs_path, obs))
    if not candidates:
        raise RuntimeError("observer10 has no timestamped poll observations")
    stamp, obs_path, obs = max(candidates, key=lambda item: (item[0], item[2].get("poll_id", "")))
    poll_dir = obs_path.parent
    req_path = poll_dir / "request.json"
    resp_path = poll_dir / "response.json"
    req_raw = req_path.read_bytes()
    resp_raw = resp_path.read_bytes()
    response = json.loads(resp_raw.decode("utf-8"))
    if len(req_raw) != obs.get("request_bytes") or sha256_bytes(req_raw) != obs.get("request_raw_sha256"):
        raise RuntimeError("latest observer request does not match observation pins")
    if len(resp_raw) != obs.get("response_bytes") or sha256_bytes(resp_raw) != obs.get("response_raw_sha256"):
        raise RuntimeError("latest observer response does not match observation pins")
    if (response.get("ok") is not True or response.get("code") != 0
            or response.get("incomplete_stream") is not False
            or response.get("stream_error") is not None
            or response.get("timed_out") is not False
            or response.get("truncated") != {"stdout": False, "stderr": False}):
        raise RuntimeError("latest observer response is not a complete, clean readback")
    detail = obs.get("detail") or {}
    run_status = detail.get("run_status") or {}
    # The observed detached launcher PID/ticks are distinct from the child
    # runner PID in run_status; bind the former to the actual launch receipt
    # later and only require the observer's own start-tick echo here.
    if obs.get("candidate_start_ticks") != detail.get("start_ticks"):
        raise RuntimeError("latest observer process start ticks differ from its detail snapshot")
    if detail.get("planned_rows") != 180 or run_status.get("planned_rows") != 180:
        raise RuntimeError("latest observer no longer binds the frozen 180-row candidate")
    for rel in (obs_path.relative_to(ROOT).as_posix(), req_path.relative_to(ROOT).as_posix(), resp_path.relative_to(ROOT).as_posix()):
        inv.add_file(rel, "candidate180_latest_observer_snapshot", copy=True,
                     role="latest_timestamp_selected_observer_readback")
    return {
        "observation_path": obs_path.relative_to(ROOT).as_posix(),
        "request_path": req_path.relative_to(ROOT).as_posix(),
        "response_path": resp_path.relative_to(ROOT).as_posix(),
        "observed_at_utc": stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "poll_id": obs.get("poll_id"),
        "status": obs.get("status"),
        "candidate_run_id": obs.get("candidate_run_id"),
        "pid": obs.get("candidate_pid"),
        "start_ticks": obs.get("candidate_start_ticks"),
        "completed_rows": run_status.get("completed_rows"),
        "planned_rows": run_status.get("planned_rows"),
        "run_status": run_status.get("status"),
        "status_counts": detail.get("status_counts"),
        "automatic_experiment_pass": obs.get("automatic_experiment_pass"),
        "candidate_admission_boolean": detail.get("candidate_admission_boolean"),
        "response_code": response.get("code"),
        "response_complete": True,
    }


def build_inventory() -> dict[str, Any]:
    inv = Inventory()
    transcript_meta = ensure_transcript_metadata()

    # Analysis contract and metadata-only support audit.
    for rel, group in [
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/build_cross_tfg_source_proof_pack_attempt01.py", "proof_pack_builder"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/build_cross_tfg_source_proof_pack_attempt02.py", "proof_pack_builder"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/build_cross_tfg_source_proof_pack_attempt03.py", "proof_pack_builder"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/cross_tfg_source_proof_pack_index_attempt01.json", "prior_proof_pack_inventory"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/audit_guard20_geometry_attempt01.py", "future_dev20_geometry"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/guard20_geometry_audit_attempt01.json", "future_dev20_geometry"),
        (".codex/scratch/cross_tfg_source_proof_pack_attempt01/aishell_transcript_hub_metadata_attempt01.json", "transcript_metadata_only"),
        (".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json", "future_dev20_protocol"),
        (".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.md", "future_dev20_protocol"),
        (".codex/scratch/dev360_candidate180_joint540_readiness_attempt05/readiness_staging_index_attempt05.json", "candidate180_readiness_staging_only"),
        (".codex/scratch/dev360_candidate180_joint540_readiness_attempt05/source_manifest_stage_attempt05.json", "candidate180_readiness_staging_only"),
    ]:
        inv.add_file(rel, group, copy=True, role="frozen_contract_or_audit")

    # Preserve the three actually completed readiness staging transactions.
    # There is intentionally no q03 request or candidate-terminal claim here.
    readiness_base = ".codex/scratch/dev360_candidate180_joint540_readiness_attempt05/requests_stage_attempt01"
    readiness_responses = {
        "q00_create_namespace.json.response.json": (380, "b8da2249591e9e105e28dbac3894111b894bb54397e4868c63938674b0f3a55b"),
        "q01_upload_worker.json.response.json": (450, "3e61fb5a209e8e3bb946df896333157987d7e9d96586168319635238242634a8"),
        "q02_upload_contract.json.response.json": (456, "5cd9b13e23fcc864540d2edb059c80d8844184390ac137cb1580f08f253887fc"),
    }
    readiness_transactions = []
    for request_name, response_name in [
        ("q00_create_namespace.json", "q00_create_namespace.json.response.json"),
        ("q01_upload_worker.json", "q01_upload_worker.json.response.json"),
        ("q02_upload_contract.json", "q02_upload_contract.json.response.json"),
    ]:
        inv.add_file(f"{readiness_base}/{request_name}", "candidate180_readiness_staging_only",
                     copy=True, role="actual_readiness_stage_request")
        response_rel = f"{readiness_base}/{response_name}"
        response_path = workspace_relative_file(response_rel)
        expected_bytes, expected_sha = readiness_responses[response_name]
        response_raw = response_path.read_bytes()
        actual_sha = sha256_bytes(response_raw)
        if len(response_raw) != expected_bytes or actual_sha != expected_sha:
            raise RuntimeError(f"readiness staging response does not match its actual pin: {response_rel}")
        inv.add_file(response_rel, "candidate180_readiness_staging_only",
                     copy=True, role="actual_readiness_stage_response")
        readiness_transactions.append({
            "request_path": f"{readiness_base}/{request_name}",
            "response_path": response_rel,
            "response_bytes": len(response_raw),
            "response_raw_sha256": actual_sha,
        })

    # Future Dev20 cohort source pins and actual source-stage evidence.
    future_base = ".codex/scratch/dev20_future_cohort_metadata_attempt06"
    for filename, role in [
        ("future_dev20_portable_source_manifest_attempt02.json", "20_source_rows_and_original_waveform_pins"),
        ("aishell_future_dev20_exact_source_download_plan_attempt01.json", "20_source_download_plan"),
        ("aishell_future_dev20_download_readback_attempt01.json", "20_source_download_readback"),
        ("future_dev20_source_bundle_build_receipt_attempt02.json", "source_archive_build_receipt"),
    ]:
        inv.add_file(f"{future_base}/{filename}", "future_dev20_source", copy=True, role=role)
    inv.add_file(f"{future_base}/future_dev20_source_bundle_attempt02.zip", "future_dev20_source", copy=False,
                 role="source_archive_reference_only", note="Contains source audio; intentionally excluded from proof-pack copy.")
    source_manifest = json.loads((ROOT / future_base / "future_dev20_portable_source_manifest_attempt02.json").read_text(encoding="utf-8"))
    future_stage_qc = json.loads((ROOT / ".codex/scratch/dev20_future_cohort_source_stage_readback_attempt02/downloads_attempt02/source_stage_qc_attempt01.json").read_text(encoding="utf-8"))
    future_root_stage_qc = json.loads((ROOT / ".codex/scratch/root_future_dev20_actual_stage_receipt_qc_attempt01.json").read_text(encoding="utf-8"))
    future_root_pcm_qc = json.loads((ROOT / ".codex/scratch/root_future_dev20_full_pcm_qc_attempt01.json").read_text(encoding="utf-8"))
    future_independent_pcm_qc = json.loads((ROOT / ".codex/scratch/dev20_future_cohort_independent_qc_attempt01/dev20_full_independent_pcm_f32_qc_attempt01.json").read_text(encoding="utf-8"))
    future_independent_stage_qc = json.loads((ROOT / ".codex/scratch/dev20_future_cohort_source_stage_readback_qc_attempt03/source_stage_receipt_independent_qc_attempt03.json").read_text(encoding="utf-8"))
    if (future_stage_qc.get("status") != "ALL_20_SOURCE_AND_RAW_F32_VERIFIED_CREATE_ONLY"
            or future_root_stage_qc.get("status") != "PASS_ACTUAL_RECEIPT_ALL20_ROWS_AND48_OUTPUT_PINS"
            or future_root_pcm_qc.get("status") != "PASS_ALL20_FULL_SOURCE_AND_INDEPENDENT_RAW_F32"
            or future_independent_pcm_qc.get("status") != "PASS_20_OF_20_MANUAL_PCM16_TO_F32_EQUALS_PINNED_CANONICAL_READER"
            or future_independent_stage_qc.get("status") != "PASS_ACTUAL_REMOTE_RECEIPT_SELF_HASH_AND_ALL_48_OUTPUT_PINS"):
        raise RuntimeError("future Dev20 full-byte/raw-F32 stage evidence no longer has its reviewed pass statuses")
    for row in source_manifest["rows"]:
        inv.add_external(
            identifier=f"future_dev20_source:{row['sample_id']}", group="future_dev20_source_media",
            path=row.get("source_path") or row.get("path") or row.get("archive_member") or "manifest row",
            bytes_count=row.get("bytes"), sha256=row.get("sha256"),
            provenance="future_dev20_portable_source_manifest_attempt02.json",
            note="Pinned natural WAV is reference-only; no WAV body is copied into this pack.",
        )
    packet04 = ".codex/scratch/dev20_future_cohort_upload_packet_attempt04"
    for filename in [
        "build_dev20_upload_packet_attempt04.py", "extract_and_qc_dev20_source_stage_attempt02.py",
        "request_index_attempt04.json", "00_create_remote_source_directory.json",
        "00_create_remote_source_directory.json.response.json", "01_upload_source_bundle.json",
        "01_upload_source_bundle.json.response.json", "02_upload_portable_manifest.json",
        "02_upload_portable_manifest.json.response.json", "03_upload_stage_qc_worker.json",
        "03_upload_stage_qc_worker.json.response.json", "04_extract_and_full_raw_f32_qc.json",
        "04_extract_and_full_raw_f32_qc.json.response.json",
    ]:
        inv.add_file(f"{packet04}/{filename}", "future_dev20_source_stage", copy=True, role="actual_stage_request_or_response")
    stage_readback = ".codex/scratch/dev20_future_cohort_source_stage_readback_attempt02"
    for rel in [
        f"{stage_readback}/build_source_stage_readback_packet_attempt02.py",
        f"{stage_readback}/request_index_attempt02.json",
        f"{stage_readback}/requests_attempt02/00_download_source_stage_qc_receipt.json",
        f"{stage_readback}/requests_attempt02/00_download_source_stage_qc_receipt.json.response.json",
        f"{stage_readback}/downloads_attempt02/source_stage_qc_attempt01.json",
        ".codex/scratch/root_future_dev20_actual_stage_receipt_qc_attempt01.json",
        ".codex/scratch/root_future_dev20_actual_stage_receipt_qc_attempt01.py",
        ".codex/scratch/root_future_dev20_full_pcm_qc_attempt01.json",
        ".codex/scratch/root_future_dev20_full_pcm_qc_attempt01.py",
        ".codex/scratch/dev20_future_cohort_independent_qc_attempt01/dev20_full_independent_pcm_f32_qc_attempt01.json",
        ".codex/scratch/dev20_future_cohort_source_stage_readback_qc_attempt03/source_stage_receipt_independent_qc_attempt03.json",
    ]:
        inv.add_file(rel, "future_dev20_source_stage", copy=True, role="actual_stage_or_pcm_readback_qc")

    # Fit40 DirectD0 source-stage, geometry-only recovery, and launcher/status03 evidence.
    d0 = ".codex/scratch/fit40_d0_gatedc_training_packet_attempt02"
    d0_bundle_manifest_rel = f"{d0}/source_bundle_manifest_attempt02.json"
    d0_bundle_manifest = json.loads(workspace_relative_file(d0_bundle_manifest_rel).read_text(encoding="utf-8"))
    if (d0_bundle_manifest.get("schema") != "fit40_d0_gatedc_source_bundle_attempt02_v1"
            or d0_bundle_manifest.get("member_count") != 23
            or len(d0_bundle_manifest.get("members", [])) != 23
            or d0_bundle_manifest.get("self_sha256") != self_hash(d0_bundle_manifest, "self_sha256")):
        raise RuntimeError("Fit40 DirectD0 source bundle manifest schema/count/self-hash failed")
    seen_d0_members: set[str] = set()
    for member in d0_bundle_manifest["members"]:
        rel = member.get("relative_path")
        if rel in seen_d0_members:
            raise RuntimeError(f"duplicate Fit40 DirectD0 bundle source member: {rel}")
        seen_d0_members.add(rel)
        inv.add_file(rel, "fit40_d0_source_closure", copy=True,
                     role="individual_pinned_fit40_d0_source_member")
        recorded = inv.entries[PurePosixPath(*relative_parts(rel)).as_posix()]
        if (recorded["bytes"] != member.get("bytes")
                or recorded["sha256"] != member.get("sha256")):
            raise RuntimeError(f"Fit40 DirectD0 source member differs from frozen manifest: {rel}")
    if len(seen_d0_members) != 23:
        raise RuntimeError("Fit40 DirectD0 source closure did not contain exactly 23 unique members")
    inv.add_file(d0_bundle_manifest_rel, "fit40_d0_source_closure", copy=True,
                 role="pinned_23_member_source_bundle_manifest")
    d0_contract_dir = ".codex/scratch/fit40_d0_direction_gate_dc_source_contract_attempt02"
    d0_contract_manifest_rel = f"{d0_contract_dir}/source_contract_manifest_attempt02.json"
    d0_contract_manifest = json.loads(workspace_relative_file(d0_contract_manifest_rel).read_text(encoding="utf-8"))
    if d0_contract_manifest.get("status") != "SOURCE_ONLY_STATIC_QC_PASS_NO_EXECUTION":
        raise RuntimeError("Fit40 DirectD0 source-contract manifest status changed")
    if not all(d0_contract_manifest.get("checks", {}).values()):
        raise RuntimeError("Fit40 DirectD0 source-contract static checks no longer all pass")
    contract_pin = d0_contract_manifest.get("contract", {})
    d0_contract_rel = contract_pin.get("path")
    if (not d0_contract_rel or not isinstance(contract_pin.get("bytes"), int)
            or not isinstance(contract_pin.get("sha256"), str)):
        raise RuntimeError("Fit40 DirectD0 source-contract pin is incomplete")
    d0_contract_path = workspace_relative_file(d0_contract_rel)
    if (d0_contract_path.stat().st_size != contract_pin["bytes"]
            or sha256_file(d0_contract_path) != contract_pin["sha256"]):
        raise RuntimeError("Fit40 DirectD0 source-contract markdown differs from its manifest pin")
    inv.add_file(d0_contract_rel, "fit40_d0_source_closure", copy=True,
                 role="fit40_d0_training_objective_and_scope_contract")
    inv.add_file(d0_contract_manifest_rel, "fit40_d0_source_closure", copy=True,
                 role="source_contract_static_review_manifest")
    d0_stage_qc = json.loads((ROOT / f"{d0}/source_receipt_readback_attempt01/downloads_attempt01/source_stage_receipt_attempt02.json").read_text(encoding="utf-8"))
    d0_root_qc = json.loads((ROOT / ".codex/scratch/root_d0_actual_source_stage_receipt_qc_attempt01.json").read_text(encoding="utf-8"))
    if d0_stage_qc.get("status") != "PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU" or d0_root_qc.get("status") != "PASS_ACTUAL_STAGE_RECEIPT_AND_ALL23_SOURCE_BYTES":
        raise RuntimeError("Fit40 DirectD0 source-stage evidence no longer has its reviewed pass status")
    inv.add_file(f"{d0}/fit40_d0_gatedc_source_attempt02.zip", "fit40_d0_stage", copy=False,
                 role="source_archive_reference_only", note="The source archive is referenced by its manifest/receipt and is not copied.")
    for rel in [
        f"{d0}/source_receipt_readback_attempt01/actual_stage_readback_preparation_qc_attempt01.json",
        f"{d0}/source_receipt_readback_attempt01/downloads_attempt01/source_stage_receipt_attempt02.json",
        f"{d0}/source_receipt_readback_attempt01/request_index_attempt01.json",
        f"{d0}/source_receipt_readback_attempt01/requests_attempt01/00_download_actual_source_stage_receipt_attempt02.json",
        f"{d0}/source_receipt_readback_attempt01/requests_attempt01/00_download_actual_source_stage_receipt_attempt02.json.response.json",
        ".codex/scratch/root_d0_actual_source_stage_receipt_qc_attempt01.json",
        ".codex/scratch/root_d0_actual_source_stage_receipt_qc_attempt01.py",
        f"{d0}/geometry_recovery_independent_qc_attempt01.json",
        f"{d0}/geometry_recovery_independent_qc_attempt02.json",
        f"{d0}/request_index_geometry_recovery_attempt02.json",
        f"{d0}/requests_attempt05/10_stage_exact_geometry_manifest_create_or_identical_attempt02.json",
        f"{d0}/requests_attempt05/10_stage_exact_geometry_manifest_create_or_identical_attempt02.json.response.json",
        f"{d0}/requests_attempt04/11_extract_and_verify_pinned_source_closure.json",
        f"{d0}/requests_attempt04/11_extract_and_verify_pinned_source_closure.json.response.json",
    ]:
        inv.add_file(rel, "fit40_d0_stage", copy=True, role="source_stage_or_geometry_recovery_evidence")
    q10_response = json.loads((ROOT / f"{d0}/requests_attempt05/10_stage_exact_geometry_manifest_create_or_identical_attempt02.json.response.json").read_text(encoding="utf-8"))
    q10_stdout = json.loads(q10_response["stdout"])
    inv.add_external(
        identifier="Fit40 staged geometry manifest attempt05",
        group="fit40_d0_geometry_reference",
        path=q10_stdout["destination"],
        bytes_count=q10_stdout["bytes"],
        sha256=q10_stdout["sha256"],
        provenance=f"{d0}/requests_attempt05/10_stage_exact_geometry_manifest_create_or_identical_attempt02.json.response.json",
        note="Remote source-clock geometry manifest is pinned by its actual create-or-identical response; no local copy or audio/media body is included.",
    )
    fix03 = f"{d0}/launcher_status_receipt_fix_attempt03"
    for filename, role in [
        ("launch_fit40_d0_gatedc_detached_attempt03.py", "launcher03"),
        ("read_fit40_d0_gatedc_status_attempt03.py", "status_reader03"),
        ("source_review_manifest_attempt03.json", "source_review_manifest03"),
        ("launcher_attempt02_to_attempt03.diff", "launcher_source_delta"),
        ("status_attempt02_to_attempt03.diff", "status_source_delta"),
        ("watch_attempt01_to_attempt02.diff", "watch_source_delta"),
        ("actual_source_stage_upload_binding_attempt01.json", "source_stage_upload_binding"),
        ("dispatch_packet_attempt01/request_index_source_upload_attempt01.json", "actual_source_upload_index"),
        ("dispatch_packet_attempt01/request_index_launch_only_attempt09.json", "actual_launch_only_index09"),
        ("dispatch_packet_attempt01/00_upload_launcher03.json", "actual_launcher03_upload_request"),
        ("dispatch_packet_attempt01/00_upload_launcher03.json.response.json", "actual_launcher03_upload_response"),
        ("dispatch_packet_attempt01/01_upload_status_reader03.json", "actual_status03_upload_request"),
        ("dispatch_packet_attempt01/01_upload_status_reader03.json.response.json", "actual_status03_upload_response"),
        ("dispatch_packet_attempt01/12_launch_fit40_d0_gatedc_once_attempt03.json", "launch_request_present_but_not_dispatched"),
    ]:
        inv.add_file(f"{fix03}/{filename}", "fit40_d0_launcher_status03", copy=True, role=role)
    for dir_rel, prefixes in [
        (f"{d0}/requests_attempt04", ("00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "11_")),
        (f"{d0}/requests_attempt05", ("10_stage_exact_geometry_manifest_create_or_identical_attempt02",)),
    ]:
        base = ROOT / dir_rel
        for path in sorted(base.glob("*.json")):
            if any(path.name.startswith(prefix) for prefix in prefixes):
                inv.add_file(path.relative_to(ROOT).as_posix(), "fit40_d0_source_upload", copy=True,
                             role="actual_source_or_geometry_request_or_response")

    # Candidate180 actual launch readback, exact plan/rows bindings, and observer10 bootstrap/latest snapshot.
    plan_dir = ".codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03"
    for filename in ["candidate_plan_attempt14.json", "candidate_planned_rows_attempt14.jsonl"]:
        inv.add_file(f"{plan_dir}/{filename}", "candidate180_plan", copy=False,
                     role="exact_launched_plan_or_rows_reference_only")
    for filename in ["candidate_plan_manifest_attempt15.json", "candidate_runtime_repair_binding_attempt10.json"]:
        inv.add_file(f"{plan_dir}/{filename}", "candidate180_plan", copy=True,
                     role="exact_candidate_plan_manifest_or_runtime_repair_binding")
    inv.add_file(".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/runtime_binding_attempt09.json",
                 "candidate180_runtime_binding", copy=False,
                 role="samehost_runtime_binding_reference_only")
    status13 = ".codex/scratch/dev360_candidate180_status13_readback_attempt01"
    for rel in [
        f"{status13}/request_index_attempt01.json",
        f"{status13}/downloads_attempt01/candidate180_codex_candidate180_cli03_attempt03_launch01.launch_receipt.json",
        f"{status13}/requests_attempt01/00_download_actual_candidate_launch_receipt.json",
        f"{status13}/requests_attempt01/00_download_actual_candidate_launch_receipt.json.response.json",
        f"{status13}/requests_attempt01/01_read_candidate_status13_once.json",
        f"{status13}/requests_attempt01/01_read_candidate_status13_once.json.response.json",
    ]:
        inv.add_file(rel, "candidate180_launch_readback", copy=True, role="actual_detached_launch_or_status13_evidence")
    inv.add_file("scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py",
                 "candidate180_source", copy=True, role="pinned_candidate_dispatcher_source")
    inv.add_file("scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt15.py",
                 "candidate180_source", copy=True, role="pinned_step120_candidate_core_source")
    inv.add_file("scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py",
                 "stagea_clock_source", copy=True, role="pinned_stagea_scoring_clock_source")
    inv.add_file("scripts/experiments/cross_tfg_audio_head_20261003/musetalk_five_arg_shim_attempt07.py",
                 "native_frontend_clock_sources", copy=True, role="pinned_musetalk_adapter_source")
    inv.add_file("scripts/experiments/cross_tfg_audio_head_20261003/ditto_five_arg_shim_attempt06.py",
                 "native_frontend_clock_sources", copy=True, role="pinned_ditto_adapter_source")
    for rel in [
        ".codex/scratch/dev20_w2l_native_frame_binding_attempt03/probe_dev20_w2l_native_frame_binding_attempt03.py",
        ".codex/scratch/dev20_w2l_native_frame_binding_attempt03/probe_config_attempt03.json",
        ".codex/scratch/dev20_w2l_native_frame_binding_attempt03_readback_attempt01/downloads/dev20_native_frame_binding_attempt03.json",
        ".codex/scratch/dev20_w2l_native_frame_binding_attempt03_readback_attempt01/downloads/preflight_attempt03.json",
    ]:
        inv.add_file(rel, "native_frontend_clock_sources", copy=True, role="pinned_w2l_frame_count_probe_or_receipt")

    observer = ".codex/scratch/dev360_candidate180_status_observer_attempt10"
    for rel in [
        f"{observer}/read_candidate180_status_observer_attempt10.py",
        f"{observer}/source_manifest_attempt10.json",
        f"{observer}/attempt09_to_attempt10.diff",
        f"{observer}/root_actual_observer_launch_receipt_attempt01.json",
    ]:
        inv.add_file(rel, "candidate180_observer10", copy=True, role="observer10_source_or_launch_evidence")
    inv.add_tree(f"{observer}/bootstrap", "candidate180_observer_bootstrap", copy=True,
                 suffixes={".json"})
    observer_snapshot = latest_observer_snapshot(inv)

    # Frozen control terminal and evidence summaries. The full ledger remains a reference only.
    for rel, copy, role in [
        (".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/terminal_content_qc_attempt01.json", True, "actual_controls_terminal_content_qc"),
        (".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/transfer_reconstruction_receipt_attempt01.json", True, "actual_controls_terminal_reconstruction_receipt"),
        (".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/controls_rms_identity_descriptive_summary_attempt01.json", True, "controls360_metadata_summary"),
        (".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/results.jsonl", False, "controls360_full_ledger_reference_only"),
        (".codex/scratch/root_failed_candidate180_full_qc_attempt01.json", True, "failed_candidate180_full_terminal_qc"),
    ]:
        inv.add_file(rel, "controls360_terminal", copy=copy, role=role)

    # Include the actual six-file terminal readback metadata and small payloads.
    # The full results ledger and strict attestation remain pinned references.
    transfer_root = ".codex/scratch/dev360_terminal_transfer_attempt06/after_plan_download_attempt01"
    reconstruction_path = ".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/transfer_reconstruction_receipt_attempt01.json"
    reconstruction = json.loads((ROOT / reconstruction_path).read_text(encoding="utf-8"))
    if reconstruction.get("status") != "PASS_TRANSFER_INTEGRITY_AND_LEDGER_RECONSTRUCTION":
        raise RuntimeError("actual controls360 terminal reconstruction receipt no longer has its reviewed status")
    for rel in [
        ".codex/scratch/dev360_terminal_transfer_attempt04/inventory_download_attempt03/downloads_attempt03/terminal_inventory_attempt02.json",
        ".codex/scratch/dev360_terminal_transfer_attempt05/after_inventory_download_attempt01/download_plan_attempt02.json",
        f"{transfer_root}/terminal_file_download_request_index_attempt01.json",
        reconstruction_path,
        ".codex/scratch/dev360_terminal_transfer_attempt09/independent_terminal_qc_attempt01.json",
        ".codex/scratch/dev360_terminal_transfer_attempt09/independent_terminal_qc_attempt02.json",
        ".codex/scratch/dev360_terminal_transfer_attempt09/independent_terminal_qc_attempt03.json",
        ".codex/scratch/dev360_terminal_transfer_attempt09/verify_terminal_transfers_attempt09.py",
    ]:
        inv.add_file(rel, "controls360_actual_six_file_readback", copy=True,
                     role="actual_controls360_inventory_plan_or_independent_qc")
    inv.add_tree(f"{transfer_root}/requests_attempt01", "controls360_actual_six_file_readback",
                 copy=True, suffixes={".json"})
    artifact_policy = {
        "execution_record": (True, "actual_execution_record"),
        "result_manifest": (True, "actual_result_manifest"),
        "run_status": (True, "actual_run_status"),
        "results_ledger": (False, "full_results_ledger_reference_only"),
        "generation_repeat_qc_ledger": (False, "empty_repeat_qc_ledger_reference_only"),
        "strict_determinism_attestation": (False, "large_strict_attestation_reference_only"),
    }
    for artifact in reconstruction.get("actual_six_artifacts", []):
        policy = artifact_policy.get(artifact.get("name"))
        if policy is None:
            raise RuntimeError(f"unexpected actual controls360 artifact: {artifact.get('name')}")
        copy, role = policy
        local_path = Path(artifact["local_path"])
        try:
            relative = local_path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"actual terminal file escaped the workspace inventory: {local_path}") from exc
        path = ROOT / Path(relative)
        if (not path.is_file() or path.stat().st_size != artifact["bytes"]
                or sha256_file(path) != artifact["raw_sha256"]):
            raise RuntimeError(f"actual controls360 terminal file differs from reconstruction pin: {artifact['name']}")
        # results.jsonl was already indexed from the canonical reconstruction folder.
        if relative != ".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/results.jsonl":
            inv.add_file(relative, "controls360_actual_six_file_readback", copy=copy, role=role)

    # Small contextual protocol pointer. The ADR path is deliberately a pointer, not a copied note.
    inv.add_external(
        identifier="BM ADR020 post-projection spectral timing controls",
        group="basic_memory_decision_pointer",
        path="basic-memory/docs/adr/020-post-projection-spectral-timing-controls.md",
        bytes_count=None, sha256=None,
        provenance="Root-maintained Basic Memory note",
        note="Pointer only; this builder does not read or rewrite the memory note.",
    )
    inv.add_external(
        identifier="Future Dev20 protocol06",
        group="future_dev20_protocol_pointer",
        path=".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json",
        bytes_count=(ROOT / ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json").stat().st_size,
        sha256=sha256_file(ROOT / ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json"),
        provenance="Local frozen protocol file",
        note="JSON and Markdown protocol are included among pack entries.",
    )

    # Verify candidate launch pins match the exact local metadata outputs and latest observer identity.
    launch_receipt_path = ROOT / status13 / "downloads_attempt01/candidate180_codex_candidate180_cli03_attempt03_launch01.launch_receipt.json"
    launch = json.loads(launch_receipt_path.read_text(encoding="utf-8"))
    plan_rel = f"{plan_dir}/candidate_plan_attempt14.json"
    rows_rel = f"{plan_dir}/candidate_planned_rows_attempt14.jsonl"
    manifest_rel = f"{plan_dir}/candidate_plan_manifest_attempt15.json"
    binding_rel = f"{plan_dir}/candidate_runtime_repair_binding_attempt10.json"
    for rel, actual in [(plan_rel, launch["candidate_plan"]), (rows_rel, launch["candidate_rows"]),
                        (manifest_rel, launch["candidate_plan_manifest"]), (binding_rel, launch["candidate_runtime_repair_binding"])]:
        p = ROOT / rel
        if p.stat().st_size != actual["bytes"] or sha256_file(p) != actual["raw_sha256"]:
            raise RuntimeError(f"candidate launch receipt does not match local source pin for {rel}")
    candidate_plan = json.loads((ROOT / plan_rel).read_text(encoding="utf-8"))
    if candidate_plan.get("status") != "FROZEN" or len(candidate_plan.get("rows", [])) != 180:
        raise RuntimeError("local candidate plan is not the frozen 180-row plan from the launch receipt")
    if observer_snapshot["pid"] != launch.get("pid") or observer_snapshot["start_ticks"] != launch.get("start_ticks"):
        raise RuntimeError("latest observer PID/start ticks differ from the actual launch receipt")
    if observer_snapshot.get("candidate_run_id") != launch.get("run_id"):
        raise RuntimeError("latest observer run id differs from the actual launch receipt")

    # The future audio references are copied as manifest metadata only.
    cohort_rows = source_manifest["rows"]
    speaker_counts: dict[str, int] = {}
    for row in cohort_rows:
        speaker_counts[str(row["speaker_id"])] = speaker_counts.get(str(row["speaker_id"]), 0) + 1
    if len(cohort_rows) != 20 or sorted(speaker_counts.values()) != [2] * 10:
        raise RuntimeError("future Dev20 source reference no longer has the frozen 10x2 roster")

    entries = sorted(inv.entries.values(), key=lambda row: row["relative_path"])
    copied = [row for row in entries if row["copy"]]
    referenced = [row for row in entries if not row["copy"]]
    transcript_pin = {
        "metadata_descriptor_path": TRANSCRIPT_METADATA_PATH.relative_to(ROOT).as_posix(),
        "metadata_descriptor_bytes": TRANSCRIPT_METADATA_PATH.stat().st_size,
        "metadata_descriptor_sha256": sha256_file(TRANSCRIPT_METADATA_PATH),
        "repository": TRANSCRIPT_METADATA["repository"],
        "pinned_commit": TRANSCRIPT_METADATA["pinned_commit"],
        "path": TRANSCRIPT_METADATA["path"],
        "bytes": TRANSCRIPT_METADATA["bytes"],
        "git_blob_oid_sha1": TRANSCRIPT_METADATA["git_blob_oid_sha1"],
        "sha256": None,
        "body_read": False,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "attempt": 3,
        "status": "PASS_SEALED_METADATA_INVENTORY_COPY_NOT_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "copy_performed": False,
            "archive_created": False,
            "git_operation_performed": False,
            "remote_operation_performed": False,
            "model_or_gpu_started": False,
            "asr_or_api_called": False,
            "future_dev20_scores_read": False,
            "confirm_audio_or_text_opened": False,
            "media_or_weight_bodies_in_copy_set": False,
        },
        "copy_contract": {
            "copy_mode_requires_separate_explicit_destination": True,
            "destination_must_not_exist": True,
            "copy_entries_are_small_allowlisted_text_or_json_evidence_only": True,
            "max_copy_entry_bytes": MAX_COPY_BYTES,
            "copy_suffix_allowlist": sorted(COPY_SUFFIXES),
            "copy_basename_exceptions": sorted(COPY_BASENAME_EXCEPTIONS),
            "reference_only_entries_are_never_copied": True,
            "portable_archives_and_wav_bodies_excluded": True,
        },
        "future_dev20": {
            "source_roster_rows": len(cohort_rows),
            "source_speaker_counts": dict(sorted(speaker_counts.items())),
            "source_wavs_copied": False,
            "actual_source_stage_status": future_stage_qc["status"],
            "stage_receipt_path": ".codex/scratch/dev20_future_cohort_source_stage_readback_attempt02/downloads_attempt02/source_stage_qc_attempt01.json",
            "root_and_independent_pcm_qc_included": True,
            "native_video_or_future_score_support_established": False,
            "protocol06_pointer": ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json",
        },
        "fit40_d0": {
            "source_member_count": 23,
            "source_member_paths": sorted(seen_d0_members),
            "source_bundle_manifest_path": d0_bundle_manifest_rel,
            "source_members_included_individually": True,
            "source_contract_markdown_path": d0_contract_rel,
            "source_contract_manifest_path": d0_contract_manifest_rel,
            "source_bundle_manifest_included": True,
            "training_contract_markdown_and_manifest_included": True,
            "root_source_receipt_qc_code_included": True,
            "source_archive_copied": False,
            "actual_source_stage_qc_included": True,
            "geometry_recovery_requests_included": True,
            "launcher_status03_source_and_upload_evidence_included": True,
            "launch_only_request_attempt09_included": True,
            "launch_request_was_dispatched_by_this_builder": False,
        },
        "candidate180_readiness_staging": {
            "staging_only": True,
            "actual_transactions_included": readiness_transactions,
            "q03_present_or_dispatched": False,
            "candidate_terminal_or_admission_claimed": False,
        },
        "candidate180": {
            "launch_receipt_path": f"{status13}/downloads_attempt01/candidate180_codex_candidate180_cli03_attempt03_launch01.launch_receipt.json",
            "run_id": launch.get("run_id"),
            "pid": launch.get("pid"),
            "start_ticks": launch.get("start_ticks"),
            "plan_raw_sha256": launch["candidate_plan"]["raw_sha256"],
            "planned_rows_raw_sha256": launch["candidate_rows"]["raw_sha256"],
            "plan_manifest_raw_sha256": launch["candidate_plan_manifest"]["raw_sha256"],
            "latest_observer_snapshot": observer_snapshot,
            "terminal": False,
            "admitted": False,
        },
        "transcript_metadata_only": transcript_pin,
        "basic_memory_pointers": {
            "adr020": "basic-memory/docs/adr/020-post-projection-spectral-timing-controls.md",
            "future_dev20_protocol06": ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json",
        },
        "entries": entries,
        "external_references": sorted(inv.external_refs, key=lambda row: (row["group"], row["identifier"])),
        "copy_entry_count": len(copied),
        "reference_entry_count": len(referenced),
        "copy_total_bytes": sum(row["bytes"] for row in copied),
        "reference_total_local_bytes": sum(row["bytes"] for row in referenced),
    }
    payload["self_sha256"] = self_hash(payload, "self_sha256")
    return payload


def write_index() -> None:
    if INDEX_PATH.exists():
        raise FileExistsError(f"preserving existing sealed proof-pack index: {INDEX_PATH}")
    payload = build_inventory()
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    write_exclusive(INDEX_PATH, raw)
    print(json.dumps({
        "status": payload["status"],
        "index_path": str(INDEX_PATH.resolve()),
        "index_bytes": len(raw),
        "index_sha256": sha256_bytes(raw),
        "self_sha256": payload["self_sha256"],
        "copy_entries": payload["copy_entry_count"],
        "copy_total_bytes": payload["copy_total_bytes"],
        "reference_entries": payload["reference_entry_count"],
        "latest_observer_at": payload["candidate180"]["latest_observer_snapshot"]["observed_at_utc"],
        "latest_observer_progress": payload["candidate180"]["latest_observer_snapshot"]["completed_rows"],
    }, sort_keys=True))


def copy_from_index(index_path: Path, destination: Path) -> None:
    index_path = assert_workspace_path(index_path)
    destination = assert_workspace_path(destination, allow_missing_leaf=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to copy into an existing destination: {destination}")
    raw = index_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if (payload.get("schema") != SCHEMA or payload.get("self_sha256") != self_hash(payload, "self_sha256")):
        raise RuntimeError("proof-pack index schema or self-hash invalid")
    if payload.get("status") != "PASS_SEALED_METADATA_INVENTORY_COPY_NOT_RUN":
        raise RuntimeError("proof-pack index status is not the create-only inventory state")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("proof-pack entries must be a list")
    keys: set[str] = set()
    copy_entries = []
    reference_bytes = 0
    copy_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("proof-pack entry is not an object")
        rel = entry.get("relative_path")
        parts = relative_parts(rel)
        key = PurePosixPath(*parts).as_posix()
        if key in keys:
            raise RuntimeError(f"duplicate proof-pack entry path: {key}")
        keys.add(key)
        if type(entry.get("copy")) is not bool:
            raise RuntimeError(f"entry copy flag must be boolean: {key}")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise RuntimeError(f"entry byte count or SHA-256 is invalid: {key}")
        source = workspace_relative_file(key)
        if not source.is_file() or source.stat().st_size != size:
            raise RuntimeError(f"indexed source is missing or its byte count changed: {key}")
        if entry["copy"]:
            copy_entries.append(entry)
            copy_bytes += size
        else:
            reference_bytes += size
    reference_entries = len(entries) - len(copy_entries)
    if (len(copy_entries) != payload.get("copy_entry_count")
            or reference_entries != payload.get("reference_entry_count")
            or copy_bytes != payload.get("copy_total_bytes")
            or reference_bytes != payload.get("reference_total_local_bytes")):
        raise RuntimeError("proof-pack entry counts or byte totals do not match the sealed summary")
    destination_rel = None
    try:
        destination_rel = destination.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError("copy destination must remain inside the workspace") from exc
    if not destination_rel.parts:
        raise RuntimeError("workspace root itself cannot be the copy destination")
    # Validate every source and prospective destination path before creating any output.
    for entry in copy_entries:
        rel = PurePosixPath(*relative_parts(entry["relative_path"]))
        source = workspace_relative_file(rel.as_posix())
        target = destination.joinpath(*rel.parts)
        assert_workspace_path(target, allow_missing_leaf=True)
        if (source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]
                or (source.suffix.lower() not in COPY_SUFFIXES and source.name not in COPY_BASENAME_EXCEPTIONS)
                or source.stat().st_size > MAX_COPY_BYTES):
            raise RuntimeError(f"source changed after index seal or violates copy policy: {entry['relative_path']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert_workspace_path(destination, allow_missing_leaf=True)
    destination.mkdir(parents=False, exist_ok=False)
    assert_workspace_path(destination)
    try:
        for entry in copy_entries:
            rel = PurePosixPath(*relative_parts(entry["relative_path"]))
            source = workspace_relative_file(rel.as_posix())
            target = destination.joinpath(*rel.parts)
            assert_workspace_path(target, allow_missing_leaf=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            assert_workspace_path(target, allow_missing_leaf=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing to overwrite copy target: {target}")
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
            if target.stat().st_size != entry["bytes"] or sha256_file(target) != entry["sha256"]:
                raise RuntimeError(f"copied file failed raw pin check: {entry['relative_path']}")
        index_target = destination / index_path.name
        assert_workspace_path(index_target, allow_missing_leaf=True)
        with index_target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Leave an interrupted create-only destination intact for inspection.
        raise
    print(json.dumps({"status": "PASS_COPY_CREATE_ONLY", "destination": str(destination.resolve()),
                      "copied_entries": len(copy_entries), "index_sha256": sha256_bytes(raw)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true", help="create a new sealed pin inventory (default operation)")
    parser.add_argument("--copy-to", type=Path, help="copy an existing sealed index to a fresh destination; not used by this task")
    parser.add_argument("--index", type=Path, default=INDEX_PATH, help="sealed index for --copy-to")
    args = parser.parse_args()
    if args.copy_to is not None:
        copy_from_index(args.index, args.copy_to)
    else:
        write_index()


if __name__ == "__main__":
    main()
