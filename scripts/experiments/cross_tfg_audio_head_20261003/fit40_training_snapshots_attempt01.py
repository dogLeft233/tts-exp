#!/usr/bin/env python3
"""Verify and extract explicitly unselected snapshots from a completed Fit40 run.

The tool accepts only the two pinned Fit40 training families below. It reads
the completed training ledger, checks the full optimizer/source chain, and
writes the actual persisted ``theta_after_update`` values at steps 30, 60, 90,
and 120. It does not score clips, pick a theta, or create a native scoring
plan. The three-backend Stage-A endpoint remains a separate, late-bound input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_STEPS = (30, 60, 90, 120)
FIT_SPEAKERS = tuple(f"S{n:04d}" for n in range(52, 72))
FIT_PAIRS = tuple((speaker, j) for speaker in FIT_SPEAKERS for j in (0, 15))
REQUIRED_MODEL_IDS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
LAG_OFFSETS = tuple(range(-15, 16))


class SnapshotError(RuntimeError):
    """A pinned input does not satisfy the frozen snapshot contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise SnapshotError(f"refusing symlink input: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SnapshotError(f"expected a regular file: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved)}


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    pin = _pin(path)
    raw = Path(pin["path"]).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid JSON input: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"expected JSON object: {path}")
    return value, raw


def _expect_ref(ref: Any, actual: Mapping[str, Any], label: str) -> None:
    if not isinstance(ref, Mapping):
        raise SnapshotError(f"{label} pin is missing")
    for key in ("bytes", "sha256"):
        if ref.get(key) != actual.get(key):
            raise SnapshotError(f"{label} {key} differs from the producer pin")
    raw_path = ref.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise SnapshotError(f"{label} producer path is missing")


def _source_pin(relative: str, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    path = REPO_ROOT / Path(relative)
    actual = _pin(path)
    if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha256:
        raise SnapshotError(f"frozen source pin changed: {relative}")
    return actual


PROFILES: dict[str, dict[str, Any]] = {
    "fit40_native_attempt11": {
        "family_id": "fit40_native_transition_head_v3_attempt11",
        "method_prefix": "fit40_native_transition_head_v3_attempt11",
        "completion_schema": "cross_tfg_fit40_native_transition_head_v3_training_runner_attempt11",
        "completion_status": "COMPLETE_FIT40_V3_120_STEPS",
        "runtime_schema": "cross_tfg_fit40_native_transition_runtime_pins_v1",
        "training_record_schema": "cross_tfg_v3_candidate_training_record_v1",
        "checkpoint_schema": "cross_tfg_v3_parameter_checkpoint_v1",
        "head_schema": "cross_tfg_shared_recipient_transition_head_v3",
        "runner": {
            "relative": "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt11.py",
            "bytes": 84494,
            "sha256": "3da9c6a1800c4c023a04b5885632efa2ca58d09ab5a8033939fdaa0f4b391219",
        },
        "head_module": {
            "relative": "scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
            "bytes": 32233,
            "sha256": "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87",
        },
        "v3_v4_adapter": {
            "relative": "scripts/audio_enhancement/v3_v4_native_runtime_adapter_v7_attempt01.py",
            "bytes": 35785,
            "sha256": "c307d0b7f363fb5a0805ee7466a470f19a416960793452afb704ec9d95bdb51c",
        },
        "qualification_schema": "cross_tfg_fit40_strict_backward_operator_qualification_v1",
        "qualification_status": "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER",
        "objective_weights": {
            "relative_Cbest_improvement": 1.0,
            "far_lag_contrast_improvement": 0.25,
            "D0_nonworsening_guard": 2.0,
            "Canchor_nonworsening_guard": 1.0,
            "visual_embedding_log_norm_ratio_squared": 0.1,
            "waveform_relative_mse_to_rms_reference": 0.02,
            "mask_total_variation": 0.002,
        },
    },
    "fit40_d0_direction_gatedc_attempt02": {
        "family_id": "fit40_d0_direction_gatedc_attempt02",
        "method_prefix": "fit40_d0_direction_gatedc_attempt02",
        "completion_schema": "cross_tfg_fit40_native_transition_head_v3_d0_direction_gatedc_runner_attempt02",
        "completion_status": "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS",
        "runtime_schema": "cross_tfg_fit40_native_transition_runtime_pins_v2_d0_direction_gatedc_attempt01",
        "training_record_schema": "cross_tfg_v3_d0_direction_gatedc_training_record_v1",
        "checkpoint_schema": "cross_tfg_v3_d0_direction_gatedc_parameter_checkpoint_v1",
        "head_schema": "cross_tfg_shared_recipient_transition_head_v3_d0_gate_dc_attempt01",
        "runner": {
            "relative": "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_d0_direction_gatedc_attempt02.py",
            "bytes": 99177,
            "sha256": "a13c652da3ef3072deeccdd364a4a450e27247e42f0ce9ec94f4245a969cd1f7",
        },
        "head_module": {
            "relative": "scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py",
            "bytes": 33497,
            "sha256": "1835213f4e05e03f6921b4f822535c3ea177433e5512e01ce8959eca272f22ee",
        },
        "v3_v4_adapter": {
            "relative": "scripts/audio_enhancement/v3_v4_native_runtime_adapter_v9_attempt01.py",
            "bytes": None,
            "sha256": "a9596ac2dd53b446123f9057540d025922a369c69d80d3ec7a094833800173cf",
        },
        "qualification_schema": "cross_tfg_fit40_strict_backward_operator_qualification_v1",
        "qualification_status": "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER",
        "objective_weights": {
            "relative_D0_improvement": 1.0,
            "relative_zero_lag_cosine_improvement": 1.0,
            "Cbest_nonworsening_guard": 1.0,
            "waveform_relative_mse_to_rms_reference": 0.02,
            "mask_total_variation": 0.002,
        },
    },
}


STATIC_INPUTS = {
    "source_manifest": {
        "relative": "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json",
        "bytes": 61469,
        "sha256": "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020",
    },
    "geometry_manifest": {
        "relative": "runs/cross_tfg_audio_head_20261003/fit40_native_rms_geometry_attempt05/fit40_native_rms_geometry_manifest.json",
        "bytes": 1421429,
        "sha256": "efa5b0bfba926e6769bcc2d1b2fae2300a7c7e50fdb7dcb5ded56e58552c9208",
        "payload_sha256": "d37d7d5fb50b18168a0f79e07583264bac5e9cc77ae6a471cebe924476d1b34f",
    },
    "head_config": {
        "relative": "runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json",
        "bytes": 10721,
        "sha256": "0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5",
        "constructor_sha256": "eaf51303012125c2de40dc102fc04a983c130b6d2aac31cd96b6e5648c3e27b8",
    },
    "parent_plan": {
        "relative": "runs/cross_tfg_audio_head_20261003/fit_v2_fullcontext_attempt05_newgpu_qc06/plan.json",
        "bytes": 63319,
        "sha256": "ee4e43290ecd2abde869bbd6781bb769fbccfbebce50983e2760ae9a17d6404c",
        "payload_sha256": "ff6cdedba33870a9bb3dead44ca5751b382d2cdb6f2f3b44923f2822c0614a0c",
    },
    "audio_io": {
        "relative": "scripts/audio_enhancement/audio_io.py",
        "bytes": 14294,
        "sha256": "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004",
    },
    "baselines": {
        "relative": "scripts/audio_enhancement/baselines.py",
        "bytes": 15123,
        "sha256": "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2",
    },
    "fit40_geometry_producer": {
        "relative": "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt05.py",
        "bytes": 63227,
        "sha256": "3438935a4cdfcfc4487f25308ce53e102d247dd77540185b14ed2b70cae26600",
    },
    "v2_runtime": {
        "relative": "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2_attempt02.py",
        "bytes": 45820,
        "sha256": "d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b",
    },
    "v4_runtime": {
        "relative": "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py",
        "bytes": 59835,
        "sha256": "dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458",
    },
    "enhance_v3_cli": {
        "relative": "scripts/audio_enhancement/enhance_v3_attempt02.py",
        "bytes": 42653,
        "sha256": "7a560332ce86a82ffede4254b015cabf5f10176e6f601bfc172a331f7bb885e6",
    },
    "feature_probe": {
        "relative": "scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_attempt02.py",
        "bytes": None,
        "sha256": "4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c",
    },
    "feature_contract": {
        "relative": "scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_contract_v2.md",
        "bytes": None,
        "sha256": "9329b5fd60a7a8047a89a4bd1ff3855812d211a5e427ce51ecb81984aaa51205",
    },
}


def _safe_local_ref(root: Path, producer_ref: Any, role: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(producer_ref, Mapping):
        raise SnapshotError(f"{role} producer reference is missing")
    source_path = producer_ref.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise SnapshotError(f"{role} producer reference lacks a path")
    basename = PurePosixPath(source_path.replace("\\", "/")).name
    if basename in ("", ".", "..") or Path(basename).name != basename:
        raise SnapshotError(f"unsafe {role} filename from producer reference")
    candidate = root / basename
    if candidate.is_symlink():
        raise SnapshotError(f"refusing symlinked {role}: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root.resolve(strict=True) or not resolved.is_file():
        raise SnapshotError(f"{role} escaped the artifact directory")
    actual = _pin(resolved)
    _expect_ref(producer_ref, actual, role)
    return resolved, actual


def _localize_runtime_path(path_text: Any, label: str) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise SnapshotError(f"{label} source path is missing")
    normalized = path_text.replace("\\", "/")
    marker = "/code/src/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif normalized.startswith("scripts/") or normalized.startswith("runs/"):
        relative = normalized
    else:
        raise SnapshotError(f"{label} source path has no portable repository-relative suffix")
    path = REPO_ROOT / Path(relative)
    if path.is_symlink():
        raise SnapshotError(f"refusing symlinked frozen source {label}: {path}")
    return path


def _source_from_ref(ref: Any, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise SnapshotError(f"runtime pin for {label} is missing")
    if not isinstance(ref.get("path"), str) or not ref.get("path"):
        raise SnapshotError(f"runtime pin for {label} has no producer path")
    relative = expected.get("relative")
    path = REPO_ROOT / Path(relative) if isinstance(relative, str) else _localize_runtime_path(ref.get("path"), label)
    actual = _pin(path)
    expected_size = expected.get("bytes")
    if expected_size is not None and actual["bytes"] != expected_size:
        raise SnapshotError(f"frozen source size changed for {label}")
    if actual["sha256"] != expected.get("sha256") or actual["bytes"] != ref.get("bytes"):
        raise SnapshotError(f"runtime source pin differs for {label}")
    if ref.get("sha256") != actual["sha256"]:
        raise SnapshotError(f"runtime source SHA differs for {label}")
    return actual


def _profile_sources(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for name, spec in STATIC_INPUTS.items():
        if name in ("feature_probe", "feature_contract") and not (
            profile["method_prefix"].startswith("fit40_d0_")
        ):
            continue
        if name == "head_config":
            pin = _source_pin(spec["relative"], spec["bytes"], spec["sha256"])
        elif spec.get("bytes") is not None:
            pin = _source_pin(spec["relative"], spec["bytes"], spec["sha256"])
        else:
            pin = _pin(REPO_ROOT / Path(spec["relative"]))
            if pin["sha256"] != spec["sha256"]:
                raise SnapshotError(f"frozen source pin changed: {spec['relative']}")
        pins[name] = pin
    for name in ("runner", "head_module", "v3_v4_adapter"):
        spec = profile[name]
        pin = _pin(REPO_ROOT / Path(spec["relative"]))
        if pin["sha256"] != spec["sha256"]:
            raise SnapshotError(f"frozen source pin changed: {spec['relative']}")
        if spec.get("bytes") is not None and pin["bytes"] != spec["bytes"]:
            raise SnapshotError(f"frozen source size changed: {spec['relative']}")
        pins[name] = pin
    return pins


def _literal_assignments(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in names:
            try:
                found[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    return found


def _verify_runner_constants(profile: Mapping[str, Any], source_pin: Mapping[str, Any]) -> dict[str, Any]:
    values = _literal_assignments(Path(source_pin["path"]))
    required = {
        "EXPECTED_SCHEMA": profile["completion_schema"],
        "EXPECTED_SOURCE_MANIFEST_SHA256": STATIC_INPUTS["source_manifest"]["sha256"],
        "EXPECTED_HEAD_CONFIG_SHA256": STATIC_INPUTS["head_config"]["sha256"],
        "EXPECTED_V3_HEAD_SHA256": profile["head_module"]["sha256"],
        "OPTIMIZER_STEPS": 120,
        "FIT_SEED": 20261003,
        "OBJECTIVE_WEIGHTS": profile["objective_weights"],
    }
    for name, expected in required.items():
        if values.get(name) != expected:
            raise SnapshotError(f"pinned training runner constant changed: {name}")
    # The exact runner raw hash pins the frozen comprehension that creates the
    # speaker/pair constants. Roster order is independently checked against
    # both the full source manifest and every optimizer row below.
    return {name: values.get(name) for name in required}


def _normalize_theta(value: Any, label: str) -> list[list[float]]:
    if isinstance(value, Mapping):
        if (value.get("name") != "band_contrasts_db" or value.get("shape") != [3, 3]
                or value.get("dtype") != "float32"
                or value.get("temporal_components") != ["static", "positive_flux", "negative_flux"]
                or value.get("non_reference_band_order") != [0, 1, 2]):
            raise SnapshotError(f"{label} theta descriptor differs from the frozen nine-parameter head")
        values = value.get("values")
    else:
        values = value
    if (not isinstance(values, list) or len(values) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in values)):
        raise SnapshotError(f"{label} theta is not a 3x3 matrix")
    result: list[list[float]] = []
    for row in values:
        normalized_row: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise SnapshotError(f"{label} theta contains a nonnumeric parameter")
            scalar = float(item)
            if not math.isfinite(scalar):
                raise SnapshotError(f"{label} theta contains a nonfinite parameter")
            normalized_row.append(scalar)
        result.append(normalized_row)
    return result


def _theta_bytes(values: list[list[float]]) -> bytes:
    flat = [scalar for row in values for scalar in row]
    return struct.pack("<9f", *flat)


def _same_theta(left: Any, right: Any, label: str) -> None:
    a = _normalize_theta(left, f"{label}.left")
    b = _normalize_theta(right, f"{label}.right")
    if a != b:
        raise SnapshotError(f"{label} theta values do not exactly match")


def _verify_fit_roster(source: Mapping[str, Any], geometry: Mapping[str, Any], runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = source.get("rows")
    fit_rows = runtime.get("fit_rows")
    geo_rows = geometry.get("rows")
    geo_loader = runtime.get("geometry_loader_records")
    if not all(isinstance(item, list) for item in (rows, fit_rows, geo_rows, geo_loader)):
        raise SnapshotError("source, geometry, or runtime Fit roster is missing")
    if not all(len(item) == 40 for item in (rows, fit_rows, geo_rows, geo_loader)):
        raise SnapshotError("Fit source-stage roster must contain exactly 40 rows")
    if source.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1" or source.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise SnapshotError("source-stage manifest is not the frozen Fit-only source transfer receipt")
    if source.get("split") != "historical_fit_only" or source.get("fit_speakers") != list(FIT_SPEAKERS):
        raise SnapshotError("source-stage manifest split or speaker roster differs")
    source_scope = source.get("scope", {})
    if (source_scope.get("dev_audio_read") is not False or source_scope.get("confirm_audio_read") is not False
            or source_scope.get("model_forward") is not False or source_scope.get("render") is not False
            or source_scope.get("score") is not False):
        raise SnapshotError("source-stage manifest scope is not treatment-free Fit-only")
    if (geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1"
            or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY"):
        raise SnapshotError("native geometry source-stage receipt is not complete Fit-only RMS geometry")
    if geometry.get("fit_speakers") != list(FIT_SPEAKERS):
        raise SnapshotError("geometry Fit speaker roster differs")
    geo_payload = {key: value for key, value in geometry.items() if key != "payload_sha256"}
    if geometry.get("payload_sha256") != sha256_bytes(canonical_bytes(geo_payload)):
        raise SnapshotError("geometry manifest canonical payload hash is invalid")
    if geometry.get("source_pins", {}).get("source_manifest", {}).get("sha256") != STATIC_INPUTS["source_manifest"]["sha256"]:
        raise SnapshotError("geometry manifest does not bind the exact Fit source manifest")

    expected_rows: list[dict[str, Any]] = []
    row_ids: list[str] = []
    for index, ((speaker, canonical_j), src, trained, geom, loaded) in enumerate(
            zip(FIT_PAIRS, rows, fit_rows, geo_rows, geo_loader)):
        # Sample IDs are fixed by the sealed source manifest; verify row-key shape below.
        if src.get("speaker_id") != speaker or src.get("canonical_j") != canonical_j:
            raise SnapshotError(f"source manifest Fit row order changed at row {index}")
        if not isinstance(src.get("sample_id"), str) or not src["sample_id"].startswith(f"BAC009{speaker}W"):
            raise SnapshotError(f"source manifest Fit sample identity changed at row {index}")
        expected_id = str(src.get("row_id"))
        if (trained.get("row_id") != expected_id or trained.get("sample_id") != src.get("sample_id")
                or trained.get("speaker_id") != speaker or trained.get("canonical_j") != canonical_j):
            raise SnapshotError(f"runtime Fit input identity differs from source manifest at row {index}")
        source_pcm = trained.get("natural_pcm", {})
        if (source_pcm.get("channels") != 1 or source_pcm.get("sample_rate_hz") != 16000
                or source_pcm.get("sample_count") != src.get("sample_count")
                or source_pcm.get("pcm_sha256") != src.get("canonical_float32_pcm_sha256")
                or source_pcm.get("file_sha256") != src.get("transfer_file_sha256")
                or source_pcm.get("direct_source_pcm16_sha256") != src.get("source_pcm_payload_sha256")):
            raise SnapshotError(f"runtime natural PCM lineage differs from source manifest at row {index}")
        if (geom.get("row_id") != expected_id or geom.get("sample_id") != src.get("sample_id")
                or geom.get("speaker_id") != speaker or geom.get("split") != "fit"
                or geom.get("method_id") != "rms_fixed"):
            raise SnapshotError(f"native geometry row identity/method differs at row {index}")
        geometry_source = geom.get("source_input", {})
        context = geom.get("full_context_fit", {})
        natural_cache = geom.get("natural_audio_embedding_cache", {})
        if (geometry_source.get("file_sha256") != src.get("transfer_file_sha256")
                or geometry_source.get("canonical_float32_pcm_sha256") != src.get("canonical_float32_pcm_sha256")
                or geometry_source.get("sample_count") != src.get("sample_count")
                or geometry_source.get("source_pcm16_sha256") != src.get("source_pcm_payload_sha256")
                or context.get("source_sample_count") != src.get("sample_count")
                or context.get("visual_query_count") != natural_cache.get("query_count")
                or context.get("visual_query_count", 0) <= 0):
            raise SnapshotError(f"geometry source clock/sample lineage differs at row {index}")
        baseline = geom.get("rms_fixed_baseline", {})
        if (baseline.get("waveform_sample_count_unchanged") is not True
                or baseline.get("method_id") != "rms_fixed"):
            raise SnapshotError(f"geometry baseline changes the source clock at row {index}")
        if loaded.get("row_id") != expected_id:
            raise SnapshotError(f"runtime geometry-loader receipt differs at row {index}")
        row_ids.append(expected_id)
        expected_rows.append({
            "row_id": expected_id,
            "sample_id": src["sample_id"],
            "speaker_id": speaker,
            "canonical_j": canonical_j,
            "natural_pcm": {
                "sample_rate_hz": 16000,
                "channels": 1,
                "sample_count": int(src["sample_count"]),
                "file_sha256": src["transfer_file_sha256"],
                "pcm_float32_sha256": src["canonical_float32_pcm_sha256"],
                "source_pcm16_payload_sha256": src["source_pcm_payload_sha256"],
            },
        })
    if len(set(row_ids)) != 40:
        raise SnapshotError("Fit source roster contains duplicate row IDs")
    if runtime.get("training_order") != "exact source-manifest order repeated in three complete passes; no shuffle, early stopping, dev, or confirmation":
        raise SnapshotError("runtime training order is not the frozen three-pass Fit-only order")
    return expected_rows


def _verify_step_log(raw: bytes, rows: list[dict[str, Any]], profile: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    lines = raw.splitlines()
    if len(lines) != 120 or any(not line for line in lines):
        raise SnapshotError("training_steps.jsonl must contain exactly 120 nonempty records")
    records: list[dict[str, Any]] = []
    by_step: dict[int, dict[str, Any]] = {}
    previous_theta: Any = [[0.0] * 3 for _ in range(3)]
    expected_optimizer = {"class": "Adam", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8}
    for step_index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"invalid optimizer record at line {step_index}") from exc
        if not isinstance(record, dict) or record.get("step") != step_index:
            raise SnapshotError(f"optimizer steps are not contiguous at step {step_index}")
        row = rows[(step_index - 1) % 40]
        if (record.get("pass_index_zero_based") != (step_index - 1) // 40
                or record.get("row_index_zero_based") != (step_index - 1) % 40
                or record.get("row_id") != row["row_id"]
                or record.get("sample_id") != row["sample_id"]
                or record.get("speaker_id") != row["speaker_id"]
                or record.get("canonical_j") != row["canonical_j"]
                or record.get("source_pcm_float32_sha256") != row["natural_pcm"]["pcm_float32_sha256"]):
            raise SnapshotError(f"optimizer source row/order binding differs at step {step_index}")
        if record.get("optimizer") != expected_optimizer:
            raise SnapshotError(f"optimizer identity/schedule differs at step {step_index}")
        gradient_count = record.get("gradient_nonzero_scalar_count")
        if type(gradient_count) is not int or not 1 <= gradient_count <= 9:
            raise SnapshotError(f"optimizer step {step_index} has no valid nonzero nine-parameter gradient")
        for field in ("gradient_l2", "gradient_max_abs", "loss"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SnapshotError(f"optimizer step {step_index} has an invalid {field}")
        before = _normalize_theta(record.get("theta_before_update"), f"step{step_index}.theta_before_update")
        after = _normalize_theta(record.get("theta_after_update"), f"step{step_index}.theta_after_update")
        _same_theta(before, previous_theta, f"step{step_index} continuity")
        previous_theta = after
        records.append(record)
        by_step[step_index] = record
    return records, by_step


def _validate_artifact_chain(artifact_dir: Path, family: str) -> dict[str, Any]:
    if family not in PROFILES:
        raise SnapshotError(f"unknown explicit training family: {family}")
    profile = PROFILES[family]
    root_arg = artifact_dir
    if root_arg.is_symlink():
        raise SnapshotError("artifact directory may not be a symlink")
    root = root_arg.resolve(strict=True)
    if not root.is_dir():
        raise SnapshotError("artifact directory is not a directory")

    static_pins = _profile_sources(profile)
    runner_constants = _verify_runner_constants(profile, static_pins["runner"])
    source_manifest_path = REPO_ROOT / Path(STATIC_INPUTS["source_manifest"]["relative"])
    geometry_path = REPO_ROOT / Path(STATIC_INPUTS["geometry_manifest"]["relative"])
    config_path = REPO_ROOT / Path(STATIC_INPUTS["head_config"]["relative"])
    parent_plan_path = REPO_ROOT / Path(STATIC_INPUTS["parent_plan"]["relative"])
    source_manifest, source_raw = _read_json(source_manifest_path)
    geometry, geometry_raw = _read_json(geometry_path)
    head_config, head_config_raw = _read_json(config_path)
    parent_plan, parent_plan_raw = _read_json(parent_plan_path)
    for name, raw in (("source_manifest", source_raw), ("geometry_manifest", geometry_raw),
                      ("head_config", head_config_raw), ("parent_plan", parent_plan_raw)):
        expected = STATIC_INPUTS[name]
        if len(raw) != expected["bytes"] or sha256_bytes(raw) != expected["sha256"]:
            raise SnapshotError(f"static Fit source-stage input changed: {name}")
    if geometry.get("payload_sha256") != STATIC_INPUTS["geometry_manifest"]["payload_sha256"]:
        raise SnapshotError("geometry manifest payload digest differs from its fixed pin")
    parent_payload = {key: value for key, value in parent_plan.items() if key != "plan_payload_sha256"}
    if (parent_plan.get("plan_payload_sha256") != STATIC_INPUTS["parent_plan"]["payload_sha256"]
            or sha256_bytes(canonical_bytes(parent_payload)) != STATIC_INPUTS["parent_plan"]["payload_sha256"]):
        raise SnapshotError("training parent plan canonical payload digest is invalid")
    constructor = head_config.get("head_constructor")
    if not isinstance(constructor, dict) or sha256_bytes(canonical_bytes(constructor)) != STATIC_INPUTS["head_config"]["constructor_sha256"]:
        raise SnapshotError("head constructor differs from its fixed canonical digest")
    if (constructor.get("sample_rate_hz") != 16000 or constructor.get("video_fps_num") != 25
            or constructor.get("video_fps_den") != 1 or constructor.get("max_abs_mask_db") != 3.0
            or constructor.get("target_rms") != 0.0376838172675746
            or constructor.get("peak_ceiling") is not None):
        raise SnapshotError("head constructor changed the fixed sample/video clock, RMS, or mask bound")

    completion_path = root / "completion.json"
    completion, completion_raw = _read_json(completion_path)
    status_path = root / "status.json"
    status, status_raw = _read_json(status_path)
    if (completion.get("schema") != profile["completion_schema"]
            or completion.get("status") != profile["completion_status"]
            or completion.get("completed_optimizer_steps") != 120
            or completion.get("fit_rows") != 40 or completion.get("fit_speakers") != 20
            or completion.get("zero_gate_rows_passed") != 40
            or completion.get("development_audio_opened") is not False
            or completion.get("confirmation_audio_opened") is not False
            or completion.get("selection_record_created") is not False):
        raise SnapshotError("completion receipt is not a complete, unselected 120-step Fit40 run")
    if (status.get("schema") != profile["completion_schema"]
            or status.get("status") != profile["completion_status"]
            or status.get("completed_optimizer_steps") != 120
            or status.get("stage") != "complete"):
        raise SnapshotError("status.json does not report the same completed 120-step run")

    declared = {}
    for role in ("checkpoint", "training_record", "runtime_source_pins", "training_steps", "strict_backward_qualification"):
        path, actual = _safe_local_ref(root, completion.get(role), role)
        declared[role] = {"path": str(path), **{k: actual[k] for k in ("bytes", "sha256")}}
        _expect_ref(status.get(role), actual, f"status.{role}")
    if completion.get("training_record_raw_sha256") != declared["training_record"]["sha256"]:
        raise SnapshotError("completion training-record digest differs from its exact raw bytes")
    checkpoint, checkpoint_raw = _read_json(Path(declared["checkpoint"]["path"]))
    training_record, training_raw = _read_json(Path(declared["training_record"]["path"]))
    runtime, runtime_raw = _read_json(Path(declared["runtime_source_pins"]["path"]))
    qualification, qualification_raw = _read_json(Path(declared["strict_backward_qualification"]["path"]))
    steps_path = Path(declared["training_steps"]["path"])
    steps_raw = steps_path.read_bytes()

    if (training_record.get("schema") != profile["training_record_schema"]
            or training_record.get("status") != "COMPLETE"
            or training_record.get("completed_optimizer_steps") != 120
            or training_record.get("planned_optimizer_steps") != 120
            or training_record.get("pass_count") != 3 or training_record.get("rows_per_pass") != 40
            or training_record.get("fit_row_count") != 40 or training_record.get("fit_speaker_count") != 20
            or training_record.get("fit_seed") != 20261003
            or training_record.get("development_audio_opened") is not False
            or training_record.get("confirmation_audio_opened") is not False
            or training_record.get("selection_record_created") is not False
            or training_record.get("selection_status") != "NOT_SELECTED"):
        raise SnapshotError("training record does not declare exactly the complete Fit-only 120-step scope")
    if training_record.get("objective", {}).get("weights") != profile["objective_weights"]:
        raise SnapshotError("training objective weights differ from the pinned runner source")
    if training_record.get("fit_source_manifest_sha256") != STATIC_INPUTS["source_manifest"]["sha256"]:
        raise SnapshotError("training record does not bind the exact Fit source manifest")
    if training_record.get("native_geometry_manifest_sha256") != STATIC_INPUTS["geometry_manifest"]["sha256"]:
        raise SnapshotError("training record does not bind the exact Fit geometry manifest")
    if training_record.get("constructor_sha256") != STATIC_INPUTS["head_config"]["constructor_sha256"]:
        raise SnapshotError("training record does not bind the exact head constructor")
    if training_record.get("head_implementation_sha256") != profile["head_module"]["sha256"]:
        raise SnapshotError("training record does not bind the exact head implementation")
    if (training_record.get("checkpoint_sha256") != declared["checkpoint"]["sha256"]
            or training_record.get("training_steps_jsonl_sha256") != declared["training_steps"]["sha256"]
            or training_record.get("runtime_source_pins_sha256") != declared["runtime_source_pins"]["sha256"]):
        raise SnapshotError("training record artifact digests do not match the exact local bytes")
    if runtime.get("schema") != profile["runtime_schema"]:
        raise SnapshotError("runtime source-pins schema differs from the explicit Fit family")
    if (runtime.get("head_constructor_sha256") != STATIC_INPUTS["head_config"]["constructor_sha256"]
            or runtime.get("optimizer") != {"class": "Adam", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8}):
        raise SnapshotError("runtime pins do not bind the expected constructor or Adam schedule")
    if qualification.get("schema") != profile["qualification_schema"] or qualification.get("status") != profile["qualification_status"]:
        raise SnapshotError("full-objective backward qualification is absent or not PASS")

    # The runtime pin bundle records the exact source files used by the Fit run.
    module_roles = {
        "audio_io": "audio_io", "baselines": "baselines",
        "enhance_v3_cli": "enhance_v3_cli", "fit40_geometry_producer": "fit40_geometry_producer",
        "v2_runtime": "v2_runtime", "v4_runtime": "v4_runtime",
        "v3_head": "head_module", "v3_v4_adapter": "v3_v4_adapter",
    }
    modules = runtime.get("training_modules")
    if not isinstance(modules, Mapping):
        raise SnapshotError("runtime source pins lack training_modules")
    checked_modules = {}
    for module_name, profile_name in module_roles.items():
        expected = profile[profile_name] if profile_name in ("head_module", "v3_v4_adapter") else STATIC_INPUTS[profile_name]
        checked_modules[module_name] = _source_from_ref(modules.get(module_name), expected, module_name)
    if profile["method_prefix"].startswith("fit40_d0_"):
        for module_name, profile_name in (("native_encoder_feature_probe", "feature_probe"),
                                          ("native_encoder_feature_contract", "feature_contract")):
            checked_modules[module_name] = _source_from_ref(modules.get(module_name), STATIC_INPUTS[profile_name], module_name)

    # Validate the runtime's own fixed references, including the parent plan.
    for name, expected in (("runner", profile["runner"]), ("head_config", STATIC_INPUTS["head_config"]),
                           ("fit_source_manifest", STATIC_INPUTS["source_manifest"]),
                           ("native_geometry_manifest", STATIC_INPUTS["geometry_manifest"])):
        _source_from_ref(runtime.get(name), expected, f"runtime.{name}")
    parent_ref = runtime.get("parent_plan")
    parent_source = _source_from_ref(parent_ref, STATIC_INPUTS["parent_plan"], "runtime.parent_plan")
    if parent_ref.get("payload_sha256") != STATIC_INPUTS["parent_plan"]["payload_sha256"]:
        raise SnapshotError("runtime parent-plan payload pin differs from the exact training plan")
    if runtime.get("strict_backward_qualification", {}).get("artifact", {}).get("sha256") != declared["strict_backward_qualification"]["sha256"]:
        raise SnapshotError("runtime strict qualification reference differs from local artifact bytes")
    if runtime.get("strict_backward_qualification", {}).get("artifact", {}).get("bytes") != declared["strict_backward_qualification"]["bytes"]:
        raise SnapshotError("runtime strict qualification size differs from local artifact bytes")
    if runtime.get("fit_source_manifest", {}).get("sha256") != STATIC_INPUTS["source_manifest"]["sha256"]:
        raise SnapshotError("runtime source-manifest reference differs from the exact input")
    if runtime.get("native_geometry_manifest", {}).get("payload_sha256") != STATIC_INPUTS["geometry_manifest"]["payload_sha256"]:
        raise SnapshotError("runtime geometry payload reference differs from the exact input")

    if (checkpoint.get("schema") != profile["checkpoint_schema"]
            or checkpoint.get("head_schema") != profile["head_schema"]
            or checkpoint.get("constructor_sha256") != STATIC_INPUTS["head_config"]["constructor_sha256"]
            or checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED"):
        raise SnapshotError("checkpoint schema, constructor, or unselected state differs")
    implementation_ref = checkpoint.get("implementation")
    if (not isinstance(implementation_ref, Mapping)
            or implementation_ref.get("sha256") != profile["head_module"]["sha256"]
            or implementation_ref.get("bytes") != profile["head_module"]["bytes"]):
        raise SnapshotError("checkpoint head-implementation pin differs")
    checkpoint_bindings = checkpoint.get("training_artifact_bindings", {})
    for name, expected in (
        ("training_steps_jsonl_sha256", declared["training_steps"]["sha256"]),
        ("runtime_source_pins_sha256", declared["runtime_source_pins"]["sha256"]),
        ("completed_optimizer_steps", 120),
        ("fit_source_manifest_sha256", STATIC_INPUTS["source_manifest"]["sha256"]),
        ("native_geometry_manifest_sha256", STATIC_INPUTS["geometry_manifest"]["sha256"]),
    ):
        if checkpoint_bindings.get(name) != expected:
            raise SnapshotError(f"checkpoint training binding differs at {name}")
    if checkpoint_bindings.get("training_score_arrays_sha256") != training_record.get("training_score_arrays_sha256"):
        raise SnapshotError("checkpoint and training record score-array bindings differ")
    if checkpoint_bindings.get("parent_plan_raw_sha256") != STATIC_INPUTS["parent_plan"]["sha256"]:
        raise SnapshotError("checkpoint parent-plan raw SHA differs")
    _same_theta(checkpoint.get("parameter"), checkpoint.get("parameter"), "checkpoint parameter descriptor")

    if profile["method_prefix"].startswith("fit40_d0_"):
        gate_pin, gate_actual = _safe_local_ref(root, completion.get("fit40_gate_dc_control"), "fit40_gate_dc_control")
        runtime_gate = runtime.get("fit40_gate_dc_control", {})
        _expect_ref(runtime_gate.get("artifact"), gate_actual, "runtime.fit40_gate_dc_control")
        gate_record = training_record.get("gate_dc_control", {})
        _expect_ref(gate_record.get("artifact"), gate_actual, "training_record.fit40_gate_dc_control")
        if (gate_record.get("status") != "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA"
                or gate_record.get("optimizer_updates") != 0
                or gate_record.get("branch_rows") != 80):
            raise SnapshotError("Direct-D0 GateDC receipt is not the complete same-theta score-only control")
        if checkpoint_bindings.get("fit40_gate_dc_control_sha256") != gate_actual["sha256"]:
            raise SnapshotError("checkpoint does not bind actual GateDC control bytes")
        declared["fit40_gate_dc_control"] = {"path": str(gate_pin), "bytes": gate_actual["bytes"], "sha256": gate_actual["sha256"]}
    else:
        if training_record.get("gate_dc_control") is not None:
            raise SnapshotError("legacy native family unexpectedly claims Direct-D0 GateDC")

    rows = _verify_fit_roster(source_manifest, geometry, runtime)
    _, steps_by_number = _verify_step_log(steps_raw, rows, profile)
    _same_theta(checkpoint.get("parameter"), steps_by_number[120].get("theta_after_update"), "terminal checkpoint")
    step_bytes_sha = sha256_bytes(_theta_bytes(_normalize_theta(steps_by_number[120]["theta_after_update"], "step120")))
    artifact_pins = {
        "completion": {"path": str(completion_path.resolve()), "bytes": len(completion_raw), "sha256": sha256_bytes(completion_raw)},
        "status": {"path": str(status_path.resolve()), "bytes": len(status_raw), "sha256": sha256_bytes(status_raw)},
        **declared,
        "source_manifest": static_pins["source_manifest"],
        "geometry_manifest": static_pins["geometry_manifest"],
        "head_config": static_pins["head_config"],
        "head_module": static_pins["head_module"],
        "runner": static_pins["runner"],
        "parent_plan": parent_source,
        "training_modules": checked_modules,
    }
    return {
        "profile": profile,
        "artifacts": artifact_pins,
        "completion": completion,
        "training_record": training_record,
        "checkpoint": checkpoint,
        "runtime_source_pins": runtime,
        "source_manifest": source_manifest,
        "geometry_manifest": geometry,
        "head_config": head_config,
        "constructor": constructor,
        "fit_rows": rows,
        "step_records": steps_by_number,
        "completion_raw_sha256": sha256_bytes(completion_raw),
        "training_record_raw_sha256": sha256_bytes(training_raw),
        "training_steps_raw_sha256": sha256_bytes(steps_raw),
        "runtime_source_pins_raw_sha256": sha256_bytes(runtime_raw),
        "checkpoint_raw_sha256": sha256_bytes(checkpoint_raw),
        "strict_backward_qualification_raw_sha256": sha256_bytes(qualification_raw),
        "terminal_theta_f32_sha256": step_bytes_sha,
        "runner_constants": runner_constants,
    }


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[field] = sha256_bytes(canonical_bytes({key: value for key, value in sealed.items() if key != field}))
    return sealed


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def extract_snapshots(artifact_dir: Path, family: str, output_dir: Path) -> dict[str, Any]:
    verified = _validate_artifact_chain(artifact_dir, family)
    if output_dir.is_symlink():
        raise SnapshotError("snapshot output directory may not be a symlink")
    output = output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"snapshot output must be absent: {output}")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    snapshot_ids = []
    for step in SNAPSHOT_STEPS:
        record = verified["step_records"][step]
        theta = _normalize_theta(record["theta_after_update"], f"step{step}.theta_after_update")
        theta_raw = _theta_bytes(theta)
        snapshot_id = f"{verified['profile']['family_id']}_source_step{step:03d}"
        method_id = f"{verified['profile']['method_prefix']}_step{step:03d}"
        if snapshot_id in snapshot_ids:
            raise SnapshotError("duplicate snapshot ID")
        snapshot_ids.append(snapshot_id)
        payload = {
            "schema": "cross_tfg_fit40_parameter_snapshot_v1",
            "snapshot_id": snapshot_id,
            "method_id": method_id,
            "training_family": verified["profile"]["family_id"],
            "source_optimizer_step": step,
            "optimizer_step_is_selected": False,
            "selection_status": "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN",
            "parameter": {
                "name": "band_contrasts_db",
                "dtype": "float32",
                "shape": [3, 3],
                "temporal_components": ["static", "positive_flux", "negative_flux"],
                "non_reference_band_order": [0, 1, 2],
                "values": theta,
                "little_endian_f32_sha256": sha256_bytes(theta_raw),
            },
            "head_constructor": verified["constructor"],
            "head_constructor_sha256": STATIC_INPUTS["head_config"]["constructor_sha256"],
            "head_schema": verified["profile"]["head_schema"],
            "head_implementation": verified["artifacts"]["head_module"],
            "source_training_record": verified["artifacts"]["training_record"],
            "source_training_steps": verified["artifacts"]["training_steps"],
            "source_checkpoint": verified["artifacts"]["checkpoint"],
            "source_completion": verified["artifacts"]["completion"],
            "source_runtime_pins": verified["artifacts"]["runtime_source_pins"],
            "source_fit_manifest": verified["artifacts"]["source_manifest"],
            "source_geometry_manifest": verified["artifacts"]["geometry_manifest"],
            "source_parent_plan": verified["artifacts"]["parent_plan"],
            "training_step_record_sha256": sha256_bytes(canonical_bytes(record)),
            "training_step_row_id": record["row_id"],
            "training_step_sample_id": record["sample_id"],
            "training_step_source_pcm_float32_sha256": record["source_pcm_float32_sha256"],
        }
        sealed = _seal(payload, "snapshot_sha256")
        filename = f"snapshot_step{step:03d}.json"
        raw = json.dumps(sealed, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        _write_exclusive(output / filename, raw)
        entries.append({
            "snapshot_id": snapshot_id,
            "method_id": method_id,
            "source_optimizer_step": step,
            "path": filename,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "snapshot_sha256": sealed["snapshot_sha256"],
        })

    endpoint_contract = {
        "schema": "cross_tfg_fit40_native_three_backend_syncnet_endpoint_v1",
        "status": "UNBOUND_NATIVE_FIT_SCORES_NOT_RUN",
        "required_model_ids": list(REQUIRED_MODEL_IDS),
        "primary_score_audio": "native_roi_original_pcm16",
        "required_fields": {
            "distance_by_query_lag": {"dtype": "float32", "shape": "[N,31]", "finite": True},
            "query_frame_index": {"dtype": "int32", "units": "source_clock_frame"},
            "query_timestamp_num": {"dtype": "int64", "units": "reduced_rational_numerator"},
            "query_timestamp_den": {"dtype": "int64", "units": "reduced_rational_denominator"},
            "track_frame_index": {"dtype": "int32", "units": "selected_track_crop_row"},
            "valid_query_mask": {"dtype": "bool", "shape": "[N]"},
            "lag_offsets_frames": {"dtype": "int32", "values": list(LAG_OFFSETS)},
        },
        "shared_support_rule": "derive actual common valid query keys from all six conditions per exact Fit source/model/seed; never assume a fixed query count",
        "qualification_refs": None,
        "runtime_binding_ref": None,
        "capacity_receipt_ref": None,
        "proxy_exclusion": "The W2L-only 320-row post-projection proxy is diagnostic evidence only and cannot satisfy this three-backend endpoint.",
    }
    index_payload = {
        "schema": "cross_tfg_fit40_snapshot_set_index_v1",
        "status": "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND",
        "training_family": verified["profile"]["family_id"],
        "training_artifact_pins": verified["artifacts"],
        "training_counts": {
            "completed_optimizer_steps": 120,
            "source_fit_rows": 40,
            "fit_speakers": 20,
            "passes": 3,
            "snapshot_steps": list(SNAPSHOT_STEPS),
        },
        "snapshots": entries,
        "native_endpoint": endpoint_contract,
        "fit_roster": verified["fit_rows"],
        "source_boundaries": {
            "dev_audio_opened": False,
            "confirmation_audio_opened": False,
            "scores_read_for_snapshot_extraction": False,
            "theta_selection_performed": False,
            "source_clock_changed": False,
            "training_geometry_role": "pinned Wav2Lip Fit40 training geometry only; not a three-backend SyncNet endpoint",
        },
    }
    index = _seal(index_payload, "index_sha256")
    index_raw = json.dumps(index, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _write_exclusive(output / "snapshot_set_index.json", index_raw)
    return {"output_dir": str(output), "index_path": str(output / "snapshot_set_index.json"),
            "index_bytes": len(index_raw), "index_raw_sha256": sha256_bytes(index_raw),
            "index_sha256": index["index_sha256"], "snapshots": entries}


def verify_snapshot_index(path: Path, expected_raw_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    index, raw = _read_json(path)
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_raw_sha256.lower():
        raise SnapshotError("snapshot index raw SHA differs from caller pin")
    if (index.get("schema") != "cross_tfg_fit40_snapshot_set_index_v1"
            or index.get("status") != "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND"
            or index.get("index_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in index.items() if k != "index_sha256"}))):
        raise SnapshotError("snapshot index schema/status/self-hash is invalid")
    if index.get("native_endpoint", {}).get("status") != "UNBOUND_NATIVE_FIT_SCORES_NOT_RUN":
        raise SnapshotError("snapshot set falsely claims a native scoring endpoint")
    if index.get("native_endpoint", {}).get("required_model_ids") != list(REQUIRED_MODEL_IDS):
        raise SnapshotError("snapshot set does not require all three native backends")
    if index.get("native_endpoint", {}).get("required_fields", {}).get("lag_offsets_frames", {}).get("values") != list(LAG_OFFSETS):
        raise SnapshotError("snapshot native endpoint lag schema changed")
    entries = index.get("snapshots")
    if not isinstance(entries, list) or len(entries) != 4:
        raise SnapshotError("snapshot index does not have exactly four records")
    base = path.resolve(strict=True).parent
    records: dict[str, Any] = {}
    for step, entry in zip(SNAPSHOT_STEPS, entries):
        if entry.get("source_optimizer_step") != step:
            raise SnapshotError("snapshot source steps are not the predeclared 30/60/90/120 order")
        filename = entry.get("path")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise SnapshotError("snapshot index record path is unsafe")
        record_path = base / filename
        record, record_raw = _read_json(record_path)
        if len(record_raw) != entry.get("bytes") or sha256_bytes(record_raw) != entry.get("sha256"):
            raise SnapshotError("snapshot raw bytes differ from the set index pin")
        if record.get("snapshot_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in record.items() if k != "snapshot_sha256"})):
            raise SnapshotError("snapshot record self-hash is invalid")
        if (record.get("source_optimizer_step") != step or record.get("selection_status") != "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN"
                or record.get("method_id") != entry.get("method_id")):
            raise SnapshotError("snapshot method/source-step identity is invalid")
        _normalize_theta(record.get("parameter"), f"snapshot step {step}")
        records[record["method_id"]] = record
    return index, records


class FitSnapshotTransform:
    """Apply one extracted head snapshot to canonical source PCM without rescoring."""

    def __init__(self, *, snapshot: Mapping[str, Any], head_module_path: Path,
                 head_module_sha256: str, device: str,
                 baselines_path: Path | None = None):
        import importlib.util
        import numpy as np
        import torch

        head_pin = _pin(head_module_path)
        if head_pin["sha256"] != head_module_sha256.lower():
            raise SnapshotError("head implementation raw SHA differs from the snapshot pin")
        if snapshot.get("head_implementation", {}).get("sha256") != head_pin["sha256"]:
            raise SnapshotError("snapshot and loaded head implementation pins differ")
        constructor = snapshot.get("head_constructor")
        if sha256_bytes(canonical_bytes(constructor)) != snapshot.get("head_constructor_sha256"):
            raise SnapshotError("snapshot head constructor canonical digest is invalid")
        spec = importlib.util.spec_from_file_location("fit40_snapshot_head", head_pin["path"])
        if spec is None or spec.loader is None:
            raise SnapshotError("cannot load the pinned head implementation")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        head_class = getattr(module, "SharedRecipientTransitionHeadV3", None)
        if head_class is None:
            raise SnapshotError("pinned head module lacks SharedRecipientTransitionHeadV3")
        self.torch = torch
        self.np = np
        if not isinstance(device, str) or not device:
            raise SnapshotError("snapshot transform requires an explicit runtime device")
        self.device = torch.device(device)
        self.head = head_class(**constructor).to(device=self.device)
        theta = np.asarray(snapshot["parameter"]["values"], dtype=np.float32)
        if theta.shape != (3, 3) or not np.isfinite(theta).all():
            raise SnapshotError("snapshot theta is not finite [3,3] float32")
        with torch.no_grad():
            self.head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device=self.device))
        self.head.eval()
        self.snapshot = dict(snapshot)
        if baselines_path is None:
            baselines_path = REPO_ROOT / "scripts/audio_enhancement/baselines.py"
        baselines_pin = _pin(baselines_path)
        if baselines_pin["sha256"] != STATIC_INPUTS["baselines"]["sha256"]:
            raise SnapshotError("fixed global RMS baseline source pin changed")
        import sys as _sys
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        from scripts.audio_enhancement.baselines import apply_global_rms
        self.apply_global_rms = apply_global_rms
        self.constructor = constructor

    def transform(self, samples: Any) -> tuple[Any, dict[str, Any]]:
        np = self.np
        torch = self.torch
        source = np.ascontiguousarray(samples, dtype=np.float32)
        if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
            raise SnapshotError("snapshot transform requires nonempty canonical mono float32 source PCM")
        baseline, baseline_info = self.apply_global_rms(
            source,
            target_rms=float(self.constructor["target_rms"]),
            peak_ceiling=self.constructor["peak_ceiling"],
            silence_epsilon=float(self.constructor["rms_epsilon"]),
        )
        with torch.inference_mode():
            result = self.head(
                torch.from_numpy(source).to(device=self.device),
                torch.from_numpy(np.ascontiguousarray(baseline)).to(device=self.device),
            )
        if result.get("schema") != self.snapshot.get("head_schema"):
            raise SnapshotError("snapshot head forward returned an unexpected schema")
        output_tensor = result.get("output_pcm_float32")
        if not isinstance(output_tensor, torch.Tensor) or output_tensor.dtype != torch.float32:
            raise SnapshotError("snapshot transform returned a non-float32 tensor")
        if output_tensor.ndim != 1 or output_tensor.numel() != source.size or not bool(torch.isfinite(output_tensor).all().item()):
            raise SnapshotError("snapshot transform changed source clock or returned nonfinite PCM")
        output = np.ascontiguousarray(output_tensor.detach().cpu().numpy(), dtype=np.float32)
        if output.shape != source.shape or not np.isfinite(output).all():
            raise SnapshotError("snapshot transform violated exact source sample-count contract")
        peak = float(np.max(np.abs(output.astype(np.float64))))
        return output, {
            "operation": "fit40_persisted_parameter_snapshot",
            "method_id": self.snapshot["method_id"],
            "training_family": self.snapshot["training_family"],
            "source_optimizer_step": self.snapshot["source_optimizer_step"],
            "training_checkpoint_sha256": self.snapshot["source_checkpoint"]["sha256"],
            "snapshot_sha256": self.snapshot["snapshot_sha256"],
            "input_pcm_float32_sha256": sha256_bytes(source.astype("<f4", copy=False).tobytes()),
            "output_pcm_float32_sha256": sha256_bytes(output.astype("<f4", copy=False).tobytes()),
            "sample_rate_hz": self.constructor["sample_rate_hz"],
            "source_sample_count": int(source.size),
            "output_sample_count": int(output.size),
            "source_clock_changed": False,
            "canonical_rms_baseline": baseline_info.__dict__,
            "peak_abs": peak,
            "peak_limiter_applied": False,
            "selection_status": "NOT_SELECTED",
            "native_three_backend_score_claim": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=tuple(PROFILES))
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="directory containing the actual completion/status/training artifacts")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, absent directory for the extracted snapshot set")
    args = parser.parse_args(argv)
    result = extract_snapshots(args.artifact_dir, args.family, args.output_dir)
    print(json.dumps({"status": "EXTRACTED_UNSELECTED", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
