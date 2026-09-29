"""Continuation-run import and preflight gates.

The parent run is treated as an immutable evidence store.  This module makes
the new run explicit, validates the parent transitively, and exposes only
read-only symlinked A artifacts to the continuation.  B artifacts always live
under the new run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DependencyBlockedError,
    InputInvalidError,
    file_sha256,
    read_json,
    read_self_hashed_json,
    resource_gate,
    resource_plan,
    write_self_hashed_json,
)
from .leaptalk_adapter import discover, runtime_snapshot


def _resolve(value: str | Path, *, root: Path = config.REPO) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = root / target
    return target.resolve()


def _binding(path: Path, expected_hash: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise InputInvalidError(f"missing bound file: {path}")
    actual = file_sha256(path)
    if expected_hash is not None and actual != expected_hash:
        raise InputInvalidError(f"hash mismatch: {path}: {actual} != {expected_hash}")
    size = int(path.stat().st_size)
    if expected_bytes is not None and size != expected_bytes:
        raise InputInvalidError(f"byte count mismatch: {path}: {size} != {expected_bytes}")
    return {"path": str(path), "bytes": size, "sha256": actual}


def _verify_evidence_bindings(spec_root: Path) -> dict[str, Any]:
    evidence_path = spec_root / "evidence-bindings.json"
    evidence = read_json(evidence_path)
    if not isinstance(evidence, Mapping) or evidence.get("change_id") != "complete-tts-native-gain-attribution":
        raise InputInvalidError("completion evidence-bindings.json has the wrong change_id")
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, raw in enumerate(evidence.get("bindings", [])):
        try:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
                raise InputInvalidError("binding row is malformed")
            target = _resolve(str(raw["path"]))
            expected_hash = str(raw.get("sha256"))
            expected_bytes = int(raw["bytes"])
            try:
                binding = _binding(target, expected_hash, expected_bytes)
                checked.append({"index": index, "kind": "current", **binding})
            except InputInvalidError:
                relative = target.relative_to(config.REPO)
                if not (str(relative).startswith("scripts/") or str(relative).startswith("tests/")):
                    raise
                snapshot = config.REPO / "pre_repair_snapshot" / relative
                binding = _binding(snapshot, expected_hash, expected_bytes)
                checked.append({"index": index, "kind": "pre_repair_snapshot", "source": str(target), **binding})
        except (OSError, InputInvalidError, TypeError, ValueError) as exc:
            failures.append({"index": index, "path": raw.get("path") if isinstance(raw, Mapping) else None, "reason": str(exc)})
    if failures:
        raise InputInvalidError(f"completion evidence bindings failed: {failures[:8]}")
    return {"path": str(evidence_path), "sha256": file_sha256(evidence_path), "binding_count": len(checked), "checked": checked, "failures": []}


def _verify_pre_repair_snapshot(spec_root: Path, evidence_path: Path) -> dict[str, Any]:
    evidence = read_json(evidence_path)
    snapshot_root = config.REPO / "pre_repair_snapshot"
    rows: list[dict[str, Any]] = []
    for raw in evidence.get("bindings", []) if isinstance(evidence, Mapping) else []:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            continue
        relative = Path(str(raw["path"]))
        if not (str(relative).startswith("scripts/") or str(relative).startswith("tests/")):
            continue
        target = snapshot_root / relative
        checked = _binding(target, str(raw.get("sha256")), int(raw["bytes"]))
        rows.append({"source": str(relative), "snapshot": checked})
    if not rows:
        raise InputInvalidError("pre_repair_snapshot contains no bound source/test files")
    return {"root": str(snapshot_root), "file_count": len(rows), "files": rows, "purpose": "immutable pre-repair source and test bytes"}


def _verify_manifest_file(path: Path, label: str) -> dict[str, Any]:
    value = read_self_hashed_json(path)
    return {"label": label, "path": str(path), "sha256": file_sha256(path), "status": value.get("status"), "payload": value}


def _verify_row_file(row: Mapping[str, Any], path_key: str, hash_key: str, label: str) -> dict[str, Any]:
    value = row.get(path_key)
    if isinstance(value, Mapping):
        path_value = value.get("path")
        expected = value.get("sha256")
    else:
        path_value = value
        expected = row.get(hash_key)
    if not isinstance(path_value, str):
        raise InputInvalidError(f"{label} has no path")
    return _binding(_resolve(path_value), str(expected) if expected else None)


def _verify_parent_transitive(parent: Path) -> dict[str, Any]:
    required = {
        "protocol": parent / "protocol.json",
        "inputs": parent / "inputs.json",
        "claims": parent / "claim_registry.json",
        "provenance": parent / "model_provenance.json",
        "assets": parent / "00_audit/assets.json",
        "audio": parent / "01_audio/manifest.json",
        "fixed": parent / "02_fixed_video/manifest.json",
        "a": parent / "02_fixed_video/a_manifest.json",
        "analysis": parent / "05_analysis/summary.json",
        "perception": parent / "06_perception/package.json",
        "perception_analysis": parent / "06_perception/analysis.json",
        "final": parent / "final.json",
        "validation": parent / "validation.json",
    }
    loaded = {name: _verify_manifest_file(path, name) for name, path in required.items()}
    assets = loaded["assets"]["payload"]
    audio = loaded["audio"]["payload"]
    fixed = loaded["fixed"]["payload"]
    a_manifest = loaded["a"]["payload"]
    analysis = loaded["analysis"]["payload"]
    final = loaded["final"]["payload"]
    validation = loaded["validation"]["payload"]
    if assets.get("status") != "COMPLETE" or int(assets.get("record_count", -1)) != len(config.SAMPLE_IDS):
        raise InputInvalidError("parent audit is not a complete 12-record cohort")
    if audio.get("status") != "COMPLETE" or int(audio.get("record_count", -1)) != 180:
        raise InputInvalidError("parent audio does not contain 180 complete records")
    if fixed.get("status") != "COMPLETE" or int(fixed.get("video_count", -1)) != 36:
        raise InputInvalidError("parent fixed-video manifest is incomplete")
    if a_manifest.get("status") != "COMPLETE" or int(a_manifest.get("science_cell_count", -1)) != 180 or int(a_manifest.get("control_cell_count", -1)) != 18:
        raise InputInvalidError("parent A manifest is not 180+18 complete")
    if validation.get("status") != "valid":
        raise InputInvalidError("parent independent validation is not valid")
    if final.get("automatic_complete") is not False and final.get("engineering_status") not in {"PARTIAL", "AUTOMATIC_COMPLETE"}:
        raise InputInvalidError("parent final state is malformed")

    file_count = 0
    source_file_count = 0
    asset_groups: set[str] = set()
    for asset in assets.get("records", []):
        if not isinstance(asset, Mapping):
            raise InputInvalidError("parent asset row is malformed")
        sample_id = int(asset.get("sample_id", -1))
        group = str(asset.get("source_group", ""))
        if not group or group in asset_groups:
            raise InputInvalidError(f"parent source_group is missing or duplicated: {group}")
        asset_groups.add(group)
        _verify_row_file(asset, "real_video", "real_video_sha256", f"parent real video/{sample_id}")
        portrait = asset.get("portrait")
        if not isinstance(portrait, Mapping):
            raise InputInvalidError(f"parent portrait binding is missing: {sample_id}")
        portrait_path = portrait.get("path")
        portrait_hash = portrait.get("file_sha256", portrait.get("sha256"))
        _binding(_resolve(str(portrait_path)), str(portrait_hash) if portrait_hash else None)
        for source in config.SOURCES:
            source_row = asset.get("sources", {}).get(source)
            if not isinstance(source_row, Mapping):
                raise InputInvalidError(f"parent source asset is missing: {sample_id}/{source}")
            required_keys = ("media", "audio_media", "real_video") if source == "R" else ("media", "audio_media", "crop", "matrix", "crop_selection", "worker_result")
            for key in required_keys:
                if key not in source_row:
                    raise InputInvalidError(f"parent source binding is missing: {sample_id}/{source}/{key}")
                binding = source_row[key]
                if isinstance(binding, Mapping):
                    _binding(_resolve(str(binding.get("path", ""))), str(binding.get("sha256")) if binding.get("sha256") else None)
                else:
                    _binding(_resolve(str(binding)))
                source_file_count += 1
            for probe_key, binding_key in (("media_probe", "media"), ("video_probe", "crop")):
                probe = source_row.get(probe_key)
                bound = source_row.get(binding_key)
                if isinstance(probe, Mapping) and isinstance(bound, Mapping) and probe.get("file_sha256") != bound.get("sha256"):
                    raise InputInvalidError(f"parent probe hash differs: {sample_id}/{source}/{probe_key}")
    for row in audio.get("records", []):
        _verify_row_file(row, "path", "file_sha256", f"parent audio/{row.get('sample_id')}/{row.get('source')}/{row.get('condition')}")
        file_count += 1
    for row in fixed.get("videos", []):
        _verify_row_file(row, "path", "file_sha256", f"parent fixed video/{row.get('sample_id')}/{row.get('video_type')}")
        selection = row.get("selection")
        if not isinstance(selection, str):
            raise InputInvalidError(f"parent fixed video lacks selection: {row.get('sample_id')}/{row.get('video_type')}")
        _binding(_resolve(selection), str(row.get("selection_sha256")))
        file_count += 2
    for row in list(a_manifest.get("cells", [])) + list(a_manifest.get("controls", [])):
        _verify_row_file(row, "matrix_path", "matrix_hash", f"parent A matrix/{row.get('id')}/{row.get('video_type')}/{row.get('eval_condition')}")
        matrix = np.load(_resolve(str(row["matrix_path"])), allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or not np.isfinite(matrix).all():
            raise InputInvalidError(f"parent A matrix is invalid: {row.get('id')}")
        file_count += 1
    content = analysis.get("content_retrieval", {})
    if content.get("status") != "COMPLETE" or int(content.get("pair_count", -1)) != config.EXPECTED_WRONG_CONTENT:
        raise InputInvalidError("parent wrong-content diagnostic is incomplete")
    unit = analysis.get("unit_geometry", {})
    if unit.get("status") not in {"COMPLETE", "PARTIAL"} or len(unit.get("rows", [])) != 180:
        raise InputInvalidError("parent unit-geometry diagnostic is incomplete")
    audio_checks = sum(
        int(row.get("condition") == "GAIN" and abs(float(row.get("operation", {}).get("measured_gain_db", 999.0)) + 6.0) <= 0.01)
        + int(row.get("condition") == "NOISE" and abs(float(row.get("operation", {}).get("measured_snr_db", 999.0)) - 20.0) <= 0.1)
        for row in audio.get("records", [])
    )
    if audio_checks != 72:
        raise InputInvalidError(f"parent manipulation checks are incomplete: {audio_checks}/72")
    replay = a_manifest.get("v15_replay", {})
    if replay.get("status") != "PASS" or int(replay.get("comparison_count", -1)) != 24:
        raise InputInvalidError("parent v15 replay is incomplete")
    return {
        "parent": str(parent),
        "manifest_hashes": {name: item["sha256"] for name, item in loaded.items()},
        "verified_a_science": 180,
        "verified_a_controls": 18,
        "verified_audio_records": 180,
        "verified_audio_manipulation_checks": audio_checks,
        "verified_v15_replay": 24,
        "verified_wrong_content_pairs": int(content["pair_count"]),
        "verified_unit_rows": len(unit.get("rows", [])),
        "verified_source_asset_files": source_file_count,
        "verified_source_group_count": len(asset_groups),
        "parent_engineering_status": final.get("engineering_status"),
        "parent_validation_status": validation.get("status"),
        "transitive_file_count": file_count,
    }


def _symlink_read_only(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise InputInvalidError(f"reusable parent directory is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise InputInvalidError(f"continuation target already points elsewhere: {target}")
    else:
        os.symlink(source, target, target_is_directory=True)
    return {"path": str(target), "target": str(source), "mode": "read_only_symlink", "source_resolved": str(target.resolve())}


def import_parent_stage(
    paths: config.RunPaths,
    *,
    parent_run: Path,
    completion_spec: Path,
    run_id: str,
) -> dict[str, Any]:
    parent = _resolve(parent_run)
    spec_root = _resolve(completion_spec)
    if not parent.is_dir():
        raise InputInvalidError(f"parent run is missing: {parent}")
    if not spec_root.is_dir():
        raise InputInvalidError(f"completion spec is missing: {spec_root}")
    evidence_path = spec_root / "evidence-bindings.json"
    evidence = _verify_evidence_bindings(spec_root)
    snapshot = _verify_pre_repair_snapshot(spec_root, evidence_path)
    transitive = _verify_parent_transitive(parent)
    if Path(transitive["parent"]).resolve() != parent:
        raise InputInvalidError("parent path changed while importing")

    parent_assets_path = parent / "00_audit/assets.json"
    parent_assets = read_self_hashed_json(parent_assets_path)
    current_assets = {
        **parent_assets,
        "run_id": run_id,
        "continuation": {
            "parent_run": str(parent),
            "parent_assets_sha256": file_sha256(parent_assets_path),
            "completion_spec": str(spec_root),
            "completion_spec_sha256": file_sha256(spec_root / "specs/tts-native-gain-completion/spec.md"),
            "parent_is_read_only": True,
        },
        "parent_evidence": transitive,
        "status": "COMPLETE",
    }
    paths.audit.mkdir(parents=True, exist_ok=True)
    write_self_hashed_json(paths.audit / "assets.json", current_assets)
    reuse_rows = [
        _symlink_read_only(parent / "01_audio", paths.audio),
        _symlink_read_only(parent / "02_fixed_video", paths.fixed_video),
    ]
    reuse = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "run_id": run_id,
        "parent_run": str(parent),
        "read_only": True,
        "entries": reuse_rows,
        "a_manifest_sha256": file_sha256(parent / "02_fixed_video/a_manifest.json"),
        "audio_manifest_sha256": file_sha256(parent / "01_audio/manifest.json"),
        "no_writable_hard_links": True,
    }
    write_self_hashed_json(paths.reuse_manifest, reuse)
    write_self_hashed_json(paths.parent_evidence, {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "run_id": run_id,
        "completion_spec": evidence,
        "pre_repair_snapshot": snapshot,
        "parent_transitive": transitive,
        "parent_assets_sha256": file_sha256(parent_assets_path),
        "parent_assets_copy_sha256": file_sha256(paths.audit / "assets.json"),
        "parent_immutable": True,
    })
    parent_protocol = read_self_hashed_json(parent / "protocol.json")
    current_code = {str(item.relative_to(config.REPO)): file_sha256(item) for item in config.package_files() if item.is_file()}
    current_docs = {
        str(item.relative_to(config.REPO)): file_sha256(item)
        for item in (config.PROPOSAL, config.DESIGN, config.PROTOCOL, config.SPEC, config.TASKS, spec_root / "README.md", spec_root / "design.md", spec_root / "tasks.md", spec_root / "evidence-bindings.json")
        if item.is_file()
    }
    write_self_hashed_json(paths.protocol, {
        **parent_protocol,
        "run_id": run_id,
        "status": "COMPLETE",
        "continuation_spec": str(spec_root),
        "continuation_spec_sha256": file_sha256(spec_root / "specs/tts-native-gain-completion/spec.md"),
        "parent_protocol_sha256": file_sha256(parent / "protocol.json"),
        "parent_evidence_sha256": file_sha256(paths.parent_evidence),
        "reuse_manifest_sha256": file_sha256(paths.reuse_manifest),
        "frozen_code": current_code,
        "frozen_documents": current_docs,
        "code_snapshot_refrozen_before_scoring": False,
        "parent_a_producer_code_is_distinct": True,
    })
    write_self_hashed_json(paths.inputs, {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "COMPLETE",
        "assets": str(paths.audit / "assets.json"),
        "assets_sha256": file_sha256(paths.audit / "assets.json"),
        "parent_evidence": str(paths.parent_evidence),
        "parent_evidence_sha256": file_sha256(paths.parent_evidence),
        "reuse_manifest": str(paths.reuse_manifest),
        "reuse_manifest_sha256": file_sha256(paths.reuse_manifest),
        "parent_run": str(parent),
    })
    write_self_hashed_json(paths.claims, {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "COMPLETE",
        "claims": [
            {"id": "H1", "status": "UNTESTED", "boundary": "fixed-video evaluation path"},
            {"id": "H2", "status": "UNTESTED", "boundary": "B generation evidence and valid controls"},
            {"id": "H3", "status": "DESCRIPTIVE_ONLY", "boundary": "prespecified audio rhythm summaries"},
            {"id": "H4", "status": "NOT_CAUSALLY_IDENTIFIED", "boundary": "training distribution/checkpoint selection"},
            {"id": "PERCEPTION", "status": "NOT_ASSESSED", "boundary": "human ratings are never inferred"},
        ],
        "historical_vs_fresh": "parent A is reused; B must be freshly generated by the bound LeapTalk configuration",
    })
    write_self_hashed_json(paths.audit / "resource_plan.json", resource_plan() | {
        "stage": "import-parent",
        "parent_run": str(parent),
        "disk_policy": "no deletion of parent or unrelated data",
        "gpu_peak_budget_bytes": config.GPU_PEAK_BUDGET_BYTES,
    })
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "import-parent",
        "status": "COMPLETE",
        "parent_run": str(parent),
        "parent_evidence": str(paths.parent_evidence),
        "reuse_manifest": str(paths.reuse_manifest),
        "verified": transitive,
        "read_only_reuse": reuse_rows,
    }


def preflight_stage(paths: config.RunPaths, *, model_config: Path) -> dict[str, Any]:
    if not paths.parent_evidence.is_file() or not paths.reuse_manifest.is_file():
        raise InputInvalidError("preflight requires import-parent to complete first")
    provenance: dict[str, Any]
    try:
        provenance = discover(_resolve(model_config))
    except DependencyBlockedError as exc:
        payload = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "stage": "preflight",
            "status": "DEPENDENCY_BLOCKED",
            "engineering_status": "DEPENDENCY_BLOCKED",
            "reason": str(exc),
            "model_config": str(_resolve(model_config)),
            "runtime": runtime_snapshot(),
            "resource": resource_plan(),
        }
        write_self_hashed_json(paths.preflight, payload)
        write_self_hashed_json(paths.provenance, {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "family": "LeapTalk",
            "status": "DEPENDENCY_BLOCKED",
            "reason": str(exc),
            "no_wav2lip_substitution": True,
        })
        return payload
    gate = resource_gate(
        gpu_peak_bytes=config.GPU_PEAK_BUDGET_BYTES,
        disk_temp_bytes=config.CELL_TEMP_BUDGET_BYTES,
        disk_persistent_bytes=0,
        require_gpu=True,
        allowed_compute_pids={os.getpid()},
    )
    status = "COMPLETE" if gate.get("gate") == "PASS" else "RESOURCE_WAIT"
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "preflight",
        "status": status,
        "engineering_status": status,
        "provenance": provenance,
        "runtime": runtime_snapshot(),
        "resource": gate,
        "model_config": str(_resolve(model_config)),
        "no_wav2lip_substitution": True,
    }
    write_self_hashed_json(paths.preflight, payload)
    write_self_hashed_json(paths.provenance, {**provenance, "schema_version": 1, "protocol_id": config.PROTOCOL_ID, "preflight_sha256": file_sha256(paths.preflight)})
    return payload
