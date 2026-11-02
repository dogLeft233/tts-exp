#!/usr/bin/env python3
"""Independently read back joined 540-row media and SyncNet evidence.

This CPU-only reader verifies the two sealed raw-media archives against the
joined media map, decodes the original PCM and restored mux audio, validates
the video frame clock, reuses the frozen mouth-motion feature implementation,
and recomputes the frozen SyncNet endpoints from the archived official score
matrices. A partial raw-media archive may produce only an INCOMPLETE readback;
it never upgrades missing/nonrenderer rows or strict receipts to PASS. It never
renders, rescores, opens Confirm data, or selects a method.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from collections import defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
JOINER_REL = BASE + "/join_v3_candidate_540_samehost_controls_attempt08.py"
JOINER_SHA256 = "7af975b83d8149b6b98c293500449f897aa8df380800324a7198cbc158ac797a"
STAGEA_CORE_REL = BASE + "/run_stagea_dev_attempt06.py"
STAGEA_CORE_SHA256 = "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"
RAW_MEDIA_COLLECTOR_REL = BASE + "/collect_v3_stagea_raw_media_evidence_attempt03.py"
RAW_MEDIA_COLLECTOR_SHA256 = "244c79a513f90f280e159087dbe211c8df0b8469cb2858d9cf5169428151f329"
CONTROLS_SCORE_COLLECTOR_REL = BASE + "/collect_v3_samehost_dev360_controls_evidence_attempt03.py"
CONTROLS_SCORE_COLLECTOR_SHA256 = "aef8399b2c9fac925941ed73e0a201faedcf494ba97766f16be20e3876fbf17a"
CANDIDATE_DISPATCHER_REL = BASE + "/run_v3_step120_candidate_samehost_attempt13.py"
CANDIDATE_DISPATCHER_SHA256 = "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"
CROSS_FRONTEND_ADAPTER_REL = BASE + "/run_cross_frontend_dev_attempt09.py"
CROSS_FRONTEND_ADAPTER_SHA256 = "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"
VISUAL_GUARD_REL = "scripts/audio_enhancement_eval/visual_motion_guard.py"
VISUAL_GUARD_SHA256 = "e154a46aa70e3c434ecc3ae9618e82e072d06fe36126eca19571404f6ca6968f"
FEATURE_SOURCE_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
EXPECTED_METHODS = ("identity", "rms_fixed", "shared_recipient_transition_head_v3_fit40_step120")
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18",
)
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
BLOCK = 1 << 20
MAX_JSON_BYTES = 128 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_json(path: Path, label: str, cap: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > cap:
        raise ValueError(f"{label} exceeds byte cap: {path}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, raw


def parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_pinned_module(path: Path, module_name: str, expected_sha: str) -> Any:
    raw_hash = sha_file(path)
    if raw_hash != expected_sha:
        raise ValueError(f"pinned source SHA mismatch: {path}: {raw_hash}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_source_closure() -> dict[str, Any]:
    pins = {
        JOINER_REL: JOINER_SHA256,
        STAGEA_CORE_REL: STAGEA_CORE_SHA256,
        RAW_MEDIA_COLLECTOR_REL: RAW_MEDIA_COLLECTOR_SHA256,
        CONTROLS_SCORE_COLLECTOR_REL: CONTROLS_SCORE_COLLECTOR_SHA256,
        CANDIDATE_DISPATCHER_REL: CANDIDATE_DISPATCHER_SHA256,
        CROSS_FRONTEND_ADAPTER_REL: CROSS_FRONTEND_ADAPTER_SHA256,
        VISUAL_GUARD_REL: VISUAL_GUARD_SHA256,
    }
    observed = {}
    for rel, expected in pins.items():
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"frozen readback source must be a regular file: {rel}")
        actual = sha_file(path)
        if actual != expected:
            raise ValueError(f"frozen readback source changed: {rel}: {actual}")
        observed[rel] = {"path": str(path.resolve(strict=True)), "bytes": path.stat().st_size,
                         "sha256": actual}
    return observed


def validate_runtime_binding(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    binding, raw = read_json(path, "same-host runtime binding", 16 * 1024 * 1024)
    if (binding.get("status") != "PASS"
            or binding.get("binding_sha256") != payload_sha(binding, "binding_sha256")):
        raise ValueError("runtime binding is not PASS or its canonical self-hash is invalid")
    execution = binding.get("execution_bindings")
    expected = binding.get("expected_file_sha256")
    if not isinstance(execution, dict) or not isinstance(expected, dict):
        raise ValueError("runtime binding lacks exact execution/tool pins")
    tools = {}
    for key in ("ffmpeg", "ffprobe"):
        text = execution.get(key)
        digest = expected.get(key)
        if not isinstance(text, str) or not Path(text).is_absolute() or not isinstance(digest, str):
            raise ValueError(f"runtime binding lacks absolute {key} path and SHA pin")
        tool = Path(text).resolve(strict=True)
        actual = sha_file(tool)
        if actual != digest:
            raise ValueError(f"runtime {key} differs from its actual host pin: {actual}")
        tools[key] = str(tool)
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw),
            "sha256": sha_bytes(raw), "binding_sha256": binding["binding_sha256"],
            "status": binding["status"]}, tools


def open_archive_metadata(path: Path, joiner: Any) -> tuple[dict[str, Any], bytes, dict[str, tarfile.TarInfo]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"raw-media archive must be a regular file: {path}")
    with tarfile.open(path, "r:gz") as archive:
        members = joiner.validate_archive_members(archive)
        raw = joiner.read_tar_member(archive, members, "raw_media_archive_manifest.json", 16 * 1024 * 1024)
    value = json.loads(raw.decode("utf-8"))
    return value, raw, members


def load_raw_archive(path: Path, role: str, expected_ref: Mapping[str, Any],
                     joiner: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} raw-media archive must be a regular file: {path}")
    archive_bytes = path.stat().st_size
    archive_sha = sha_file(path)
    if archive_bytes != expected_ref.get("bytes") or archive_sha != expected_ref.get("sha256"):
        raise ValueError(f"{role} raw-media archive does not match the joined manifest pin")
    media_manifest, media_raw, _ = open_archive_metadata(path, joiner)
    expected_rows = 360 if role == "samehost_controls" else 180
    if media_manifest.get("status") not in {"PASS", "INCOMPLETE"}:
        raise ValueError(f"{role} raw-media archive has an unknown terminal status")
    producer_names = {
        "plan": "producer/plan.json", "planned_rows": "producer/planned_rows.jsonl",
        "plan_manifest": "producer/plan_manifest.json", "result_manifest": "producer/result_manifest.json",
        "results": "producer/results.jsonl",
    }
    with tarfile.open(path, "r:gz") as archive:
        members = joiner.validate_archive_members(archive)
        raw_docs = {key: joiner.read_tar_member(archive, members, name, MAX_JSON_BYTES)
                    for key, name in producer_names.items()}
    plan = json.loads(raw_docs["plan"].decode("utf-8"))
    plan_manifest = json.loads(raw_docs["plan_manifest"].decode("utf-8"))
    result_manifest = json.loads(raw_docs["result_manifest"].decode("utf-8"))
    planned_rows = parse_jsonl(raw_docs["planned_rows"], f"{role} frozen planned rows")
    result_rows = parse_jsonl(raw_docs["results"], f"{role} producer results")
    verified = joiner.verify_raw_media_archive(
        path, archive_bytes, archive_sha, role=role,
        expected_plan_sha256=sha_bytes(raw_docs["plan"]),
        expected_result_manifest_sha256=sha_bytes(raw_docs["result_manifest"]),
        expected_results_sha256=sha_bytes(raw_docs["results"]),
        expected_rows=expected_rows,
    )
    if (media_manifest.get("source_plan_raw_sha256") != sha_bytes(raw_docs["plan"])
            or media_manifest.get("source_planned_rows_raw_sha256") != sha_bytes(raw_docs["planned_rows"])
            or media_manifest.get("source_plan_manifest_raw_sha256") != sha_bytes(raw_docs["plan_manifest"])
            or media_manifest.get("source_result_manifest_raw_sha256") != sha_bytes(raw_docs["result_manifest"])
            or media_manifest.get("source_results_raw_sha256") != sha_bytes(raw_docs["results"])
            or len(planned_rows) != expected_rows or len(result_rows) != expected_rows
            or planned_rows != plan.get("rows")
            or plan_manifest.get("manifest_sha256") != joiner.payload_sha(plan_manifest, "manifest_sha256")
            or result_manifest.get("manifest_sha256") != joiner.payload_sha(result_manifest, "manifest_sha256")):
        raise ValueError(f"{role} embedded producer inputs differ from the raw-media manifest")
    plan_by_id = {str(row["row_id"]): row for row in planned_rows}
    result_by_id = {str(row["row_id"]): row for row in result_rows}
    if len(plan_by_id) != expected_rows or set(plan_by_id) != set(result_by_id):
        raise ValueError(f"{role} producer denominator/order has duplicates or missing rows")
    native_candidate_rows = []
    if role == "step120_candidate":
        model_by_id = {item.get("model_id"): item for item in plan.get("models", [])}
        wav2lip_id = "wav2lip_gan_bac9a81e"
        for row_id, planned in plan_by_id.items():
            if planned.get("model_id") != wav2lip_id:
                continue
            result = result_by_id[row_id]
            if result.get("status") != "PASS":
                native_candidate_rows.append({"row_id": row_id, "status": result.get("status"),
                                              "native_output_binding": "NOT_AVAILABLE_FOR_NONPASS_ROW"})
                continue
            generator = model_by_id.get(wav2lip_id, {}).get("generator", {})
            render = result.get("render", {})
            generation = render.get("generation", {}) if isinstance(render, dict) else {}
            input_audio = generation.get("generator_input_audio", {})
            frame_binding = planned.get("wav2lip_native_frame_binding", {})
            expected_frames = frame_binding.get("expected_inference_frame_count")
            natural = planned.get("natural_pcm", {})
            mux_audio = result.get("mux_audio_readback", {})
            if (generation.get("native_entry_path") != generator.get("native_entry_path")
                    or generation.get("native_entry_sha256") != generator.get("native_entry_sha256")
                    or input_audio.get("path") != result.get("generator_pcm", {}).get("path")
                    or input_audio.get("pcm_float32_sha256") != result.get("generator_pcm_sha256")
                    or input_audio.get("sample_count") != natural.get("sample_count")
                    or input_audio.get("sample_rate_hz") != 16000 or input_audio.get("channels") != 1
                    or type(expected_frames) is not int or expected_frames <= 0
                    or generation.get("expected_frame_count") != expected_frames
                    or render.get("frame_count") != expected_frames
                    or (render.get("fps_num"), render.get("fps_den")) != (25, 1)
                    or mux_audio.get("status") != "PASS"
                    or mux_audio.get("codec") != "pcm_f32le"
                    or mux_audio.get("sample_rate_hz") != 16000 or mux_audio.get("channels") != 1
                    or mux_audio.get("sample_count") != natural.get("sample_count")
                    or mux_audio.get("decoded_float32_pcm_sha256") != natural.get("pcm_sha256")
                    or mux_audio.get("matches_natural_canonical_pcm") is not True):
                raise ValueError(f"candidate Wav2Lip row does not bind its native input/frame map/original PCM: {row_id}")
            native_candidate_rows.append({"row_id": row_id, "status": "PASS",
                                          "native_entry_path": generator.get("native_entry_path"),
                                          "native_entry_sha256": generator.get("native_entry_sha256"),
                                          "native_frame_count": expected_frames,
                                          "generator_pcm_path": input_audio.get("path"),
                                          "generator_pcm_float32_sha256": input_audio.get("pcm_float32_sha256"),
                                          "generator_pcm_sample_count": input_audio.get("sample_count"),
                                          "generator_pcm_bytes_in_raw_media_archive": False,
                                          "original_pcm_mux_readback_sha256": mux_audio.get("decoded_float32_pcm_sha256")})
    return {
        "role": role, "path": str(path.resolve(strict=True)), "bytes": archive_bytes,
        "sha256": archive_sha, "manifest": media_manifest,
        "manifest_raw_sha256": sha_bytes(media_raw), "plan": plan,
        "planned_rows": planned_rows, "plan_by_id": plan_by_id,
        "results": result_rows, "result_by_id": result_by_id,
        "row_media_by_id": verified["row_media_by_id"],
        "assets": media_manifest["assets"],
        "candidate_native_output_bindings": native_candidate_rows,
    }


def read_join(join_dir: Path, joiner: Any) -> dict[str, Any]:
    if join_dir.is_symlink() or not join_dir.is_dir():
        raise ValueError(f"joint output must be a real directory: {join_dir}")
    manifest_path = join_dir / "join_manifest.json"
    manifest, manifest_raw = read_json(manifest_path, "joint join manifest", 16 * 1024 * 1024)
    if (manifest.get("schema") != "cross_tfg_v3_three_method_540row_join_manifest_v1"
            or manifest.get("manifest_sha256") != joiner.payload_sha(manifest, "manifest_sha256")
            or manifest.get("planned_rows", {}).get("total") != 540
            or manifest.get("confirm_audio_read") is not False
            or manifest.get("confirm_effects_read") is not False
            or manifest.get("method_selected") is not False):
        raise ValueError("joint join manifest schema, 540-row scope, or read-only constraints differ")
    refs = {
        "joint_plan": ("joint_plan.json", manifest.get("joint_plan", {})),
        "joint_results": ("joint_results.jsonl", manifest.get("joint_results", {})),
        "joint_common_support": ("joint_common_support.jsonl", manifest.get("support_rows", {})),
        "joint_media_map": ("joint_media_map.json", manifest.get("joint_media_map", {})),
        "paired_summaries": ("paired_summaries.json", manifest.get("paired_summaries", {})),
    }
    raw_files: dict[str, bytes] = {}
    for key, (name, ref) in refs.items():
        path = join_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"joined {key} must be a regular file: {name}")
        raw = path.read_bytes()
        if (len(raw) != ref.get("bytes") or sha_bytes(raw) != ref.get("sha256")):
            raise ValueError(f"joined {key} file differs from manifest: {name}")
        raw_files[key] = raw
    plan = json.loads(raw_files["joint_plan"].decode("utf-8"))
    results = parse_jsonl(raw_files["joint_results"], "joint SyncNet result ledger")
    supports = parse_jsonl(raw_files["joint_common_support"], "joint common-support ledger")
    media_map = json.loads(raw_files["joint_media_map"].decode("utf-8"))
    summary = json.loads(raw_files["paired_summaries"].decode("utf-8"))
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != joiner.payload_sha(plan, "plan_sha256")
            or len(plan.get("rows", [])) != 540 or len(results) != 540 or len(supports) != 180
            or media_map.get("schema") != "cross_tfg_v3_samehost_540_raw_media_input_map_v1"
            or media_map.get("map_sha256") != joiner.payload_sha(media_map, "map_sha256")
            or media_map.get("independent_visual_clock_pcm_rechecks_performed") is not False):
        raise ValueError("joined plan/result/support/media-map schema or denominator differs")
    methods = defaultdict(set)
    base_contexts = set()
    sample_speakers: dict[str, set[str]] = defaultdict(set)
    plan_models: set[str] = set()
    plan_seeds: set[int] = set()
    for row in plan["rows"]:
        method = row.get("method_id")
        model = row.get("model_id")
        sample = row.get("sample_id")
        speaker = row.get("speaker_id")
        seed = row.get("seed")
        if (row.get("split") != "dev" or method not in EXPECTED_METHODS or model not in EXPECTED_MODELS
                or not isinstance(sample, str) or not isinstance(speaker, str)
                or type(seed) is not int):
            raise ValueError("joined plan row leaves the frozen Dev/method/model/seed roster")
        context = (sample, speaker, model, seed)
        if context in methods[method]:
            raise ValueError(f"joined plan contains a duplicate method context: {method}/{context}")
        methods[method].add(context)
        base_contexts.add((sample, speaker, seed))
        sample_speakers[sample].add(speaker)
        plan_models.add(model)
        plan_seeds.add(seed)
    if (plan_models != set(EXPECTED_MODELS) or plan_seeds != {20261003, 20261004, 20261005}
            or len(base_contexts) != 60 or len(sample_speakers) != 20
            or len({speaker for values in sample_speakers.values() for speaker in values}) != 10
            or any(len(values) != 1 for values in sample_speakers.values())):
        raise ValueError("joined plan no longer has the frozen 20-clip/10-speaker/3-seed cohort")
    expected_contexts = {(sample, speaker, model, seed)
                         for sample, speaker, seed in base_contexts for model in EXPECTED_MODELS}
    if any(methods.get(method) != expected_contexts for method in EXPECTED_METHODS):
        raise ValueError("joined plan does not contain the exact three-method 540-row cohort")
    manifest_media = manifest.get("raw_media_inputs", {})
    if (manifest_media.get("status") not in {"PASS", "INCOMPLETE"}
            or manifest_media.get("independent_visual_clock_pcm_rechecks_performed") is not False
            or media_map.get("status") not in {"PASS", "INCOMPLETE"}
            or media_map.get("status") != manifest_media.get("status")
            or media_map.get("row_count") != 540
            or len(media_map.get("rows", [])) != 540
            or len({row.get("row_id") for row in media_map.get("rows", [])}) != 540):
        raise ValueError("joint media input map is incomplete or has an unexpected prior-recheck claim")
    asset_refs = manifest.get("assets")
    if not isinstance(asset_refs, list) or len(asset_refs) != manifest.get("asset_count"):
        raise ValueError("joined raw SyncNet asset inventory is malformed")
    asset_by_path = {}
    for ref in asset_refs:
        rel = ref.get("path") if isinstance(ref, dict) else None
        path = PurePosixPath(rel) if isinstance(rel, str) else None
        if (path is None or path.is_absolute() or ".." in path.parts or "\\" in rel
                or rel in asset_by_path or not rel.startswith("assets/")):
            raise ValueError("joined raw SyncNet asset inventory contains an unsafe or duplicate path")
        asset_by_path[rel] = ref
    if {row.get("row_id") for row in plan["rows"]} != {row.get("row_id") for row in results}:
        raise ValueError("joint plan/result row-ID sets differ")
    if len({row.get("row_id") for row in plan["rows"]}) != 540:
        raise ValueError("joint plan contains duplicate row IDs")
    return {"manifest": manifest, "manifest_raw_sha256": sha_bytes(manifest_raw),
            "plan": plan, "results": results, "supports": supports,
            "media_map": media_map, "summary": summary, "asset_by_path": asset_by_path,
            "join_dir": join_dir.resolve(strict=True)}


def load_sync_readback(joined: Mapping[str, Any], joiner: Any, core: Any) -> dict[str, Any]:
    join_dir = joined["join_dir"]
    plan = joined["plan"]
    results = joined["results"]
    expected_rows = {row["row_id"]: row for row in plan["rows"]}
    result_by_id = {row["row_id"]: row for row in results}
    if set(expected_rows) != set(result_by_id):
        raise ValueError("joint Sync result rows do not cover their exact plan")
    asset_bytes: dict[str, bytes] = {}
    row_records: dict[str, dict[str, Any]] = {}
    for row_id, row in result_by_id.items():
        planned = expected_rows[row_id]
        for field in IDENTITY_FIELDS:
            if row.get(field) != planned.get(field):
                raise ValueError(f"joint Sync result identity mismatch at {field}: {row_id}")
        if row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"}:
            raise ValueError(f"joint Sync row status is invalid: {row_id}")
        if row.get("status") != "PASS":
            row_records[row_id] = {"row_id": row_id, "status": row["status"],
                                   "failure_reason": row.get("failure_reason"),
                                   "matrix_readback": "NOT_AVAILABLE"}
            continue
        score = row.get("score", {})
        desc = score.get("raw_matrix_npz", {})
        asset_ref = row.get("joint_raw_score_asset", {})
        digest = desc.get("file_sha256")
        rel = asset_ref.get("path")
        if (not isinstance(digest, str) or len(digest) != 64
                or asset_ref.get("sha256") != digest or rel != f"assets/{digest}.npz"
                or rel not in joined["asset_by_path"]):
            raise ValueError(f"joint Sync row lacks its exact content-addressed NPZ: {row_id}")
        manifest_ref = joined["asset_by_path"][rel]
        if (manifest_ref.get("sha256") != digest
                or type(asset_ref.get("bytes")) is not int
                or manifest_ref.get("bytes") != asset_ref.get("bytes")):
            raise ValueError(f"joint Sync NPZ file pin differs among row and manifest: {row_id}")
        raw = asset_bytes.get(digest)
        if raw is None:
            path = join_dir / rel
            if path.is_symlink() or not path.is_file() or path.stat().st_size != manifest_ref.get("bytes"):
                raise ValueError(f"joined SyncNet NPZ asset is missing/unsafe: {rel}")
            raw = path.read_bytes()
            if sha_bytes(raw) != digest:
                raise ValueError(f"joined SyncNet NPZ bytes changed: {rel}")
            asset_bytes[digest] = raw
        arrays = joiner.load_score_arrays(raw, desc, f"joint row {row_id}", digest)
        row["_internal_arrays"] = arrays
        row_records[row_id] = {
            "row_id": row_id, "status": "PASS", "matrix_readback": "PASS",
            "matrix_file_sha256": digest, "matrix_file_bytes": len(raw),
            "query_count": int(arrays["distance_by_query_lag"].shape[0]),
            "query_key_sha256": sha_bytes(canonical(joiner.query_keys(arrays))),
        }

    core._finish_supports(plan, results)
    plan_by_context: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in plan["rows"]:
        key = (row.get("split"), row.get("sample_id"), row.get("speaker_id"), row.get("model_id"),
               row.get("face_id"), row.get("seed"))
        plan_by_context[key].append(row)
    stored_supports = {
        (row.get("split"), row.get("sample_id"), row.get("speaker_id"), row.get("model_id"),
         row.get("face_id"), row.get("seed")): row for row in joined["supports"]
    }
    if len(stored_supports) != len(joined["supports"]) or len(stored_supports) != 180:
        raise ValueError("joint support ledger has duplicate/missing contexts")
    for key, group in plan_by_context.items():
        stored = stored_supports.get(key)
        if stored is None or len({row.get("method_id") for row in group}) != 3:
            raise ValueError(f"joint common support context or three-method roster is missing: {key}")
        group_results = [result_by_id[row["row_id"]] for row in group]
        method_status = {row["method_id"]: row.get("status") for row in group_results}
        if stored.get("method_status") != method_status:
            raise ValueError(f"joint common-support method status differs from the score rows: {key}")
        all_pass = all(row.get("status") == "PASS" for row in group_results)
        if all_pass:
            valid_sets = []
            for result in group_results:
                arrays = result["_internal_arrays"]
                keys = joiner.query_keys(arrays)
                valid = arrays["valid_query_mask"].astype(bool, copy=False)
                valid_sets.append({query for query, good in zip(keys, valid, strict=True) if bool(good)})
            common = set.intersection(*valid_sets)
            full_keys = sorted(common, key=lambda key: (key[0], Fraction(key[1], key[2]), key[3]))
        else:
            full_keys = []
        expected_keys_by_support = {
            "full": full_keys,
            "guard20": full_keys[20:-20] if len(full_keys) >= 41 else [],
        }
        for support_id, expected_keys in expected_keys_by_support.items():
            stored_support = stored.get("support", {}).get(support_id, {})
            if (stored_support.get("query_keys") != [list(key) for key in expected_keys]
                    or stored_support.get("query_count") != len(expected_keys)
                    or stored_support.get("support_mask_sha256") != core._support_sha(support_id, expected_keys)
                    or stored_support.get("eligible") != (len(expected_keys) >= (1 if support_id == "full" else 41))):
                raise ValueError(f"joint common support does not recompute from archived score matrices: {key}/{support_id}")
            for result in group_results:
                score_support = result.get("score", {}).get("support_by_support", {}).get(support_id, {})
                if (score_support.get("query_count") != len(expected_keys)
                        or score_support.get("support_mask_sha256") != core._support_sha(support_id, expected_keys)
                        or score_support.get("eligible") != (len(expected_keys) >= (1 if support_id == "full" else 41))):
                    raise ValueError(f"row-level endpoint support does not recompute from NPZ arrays: {result['row_id']}/{support_id}")

    rows_by_method: dict[str, dict[tuple[str, str, str, int], dict[str, Any]]] = {
        method: {} for method in EXPECTED_METHODS
    }
    for row in plan["rows"]:
        key = (row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))
        method = row["method_id"]
        if method not in rows_by_method or key in rows_by_method[method]:
            raise ValueError(f"joint Sync rows have an unexpected method or duplicate context: {method}/{key}")
        rows_by_method[method][key] = result_by_id[row["row_id"]]
    if any(len(rows_by_method[method]) != 180 for method in EXPECTED_METHODS):
        raise ValueError("joint Sync readback did not preserve 180 rows per method")
    planned_speakers: dict[str, set[str]] = defaultdict(set)
    for row in plan["rows"]:
        planned_speakers[str(row["speaker_id"])].add(str(row["sample_id"]))
    seeds = tuple(sorted({int(row["seed"]) for row in plan["rows"]}))
    recomputed_contrasts: dict[str, Any] = {}
    recomputed_guards: dict[str, Any] = {}
    for support_id in ("full", "guard20"):
        for contrast_id, method_a, method_b in joiner.CONTRASTS:
            key = f"{contrast_id}:{support_id}"
            recomputed_contrasts[key] = joiner.aggregate_contrast(
                rows_by_method, planned_speakers, seeds, tuple(EXPECTED_MODELS), support_id,
                method_a, method_b, "Cbest")
            for metric in ("D0", "Canchor"):
                value = joiner.aggregate_contrast(
                    rows_by_method, planned_speakers, seeds, tuple(EXPECTED_MODELS), support_id,
                    method_a, method_b, metric)
                value["metric"] = metric
                recomputed_guards[f"{key}:{metric}"] = value
    saved_summary = joined["summary"]
    if (canonical(recomputed_contrasts) != canonical(saved_summary.get("contrasts"))
            or canonical(recomputed_guards) != canonical(saved_summary.get("guard_contrasts"))):
        raise ValueError("paired SyncNet summaries do not recompute from the joined raw matrix assets")
    for method in EXPECTED_METHODS:
        expected_count = sum(
            row.get("score", {}).get("support_by_support", {}).get("full", {}).get("eligible") is True
            for row in rows_by_method[method].values())
        if expected_count != saved_summary.get("eligibility_counts", {}).get("full", {}).get(method):
            raise ValueError(f"joined SyncNet eligibility count differs from raw matrix readback: {method}")
    for row_id, row in result_by_id.items():
        if row.get("status") == "PASS":
            row_records[row_id]["metrics_by_support"] = row["score"]["metrics_by_support"]
            row_records[row_id]["support_by_support"] = row["score"]["support_by_support"]
    return {
        "status": "PASS" if sum(record.get("status") == "PASS" for record in row_records.values()) == 540 else "INCOMPLETE",
        "row_records": row_records,
        "matrix_asset_count": len(asset_bytes),
        "matrix_asset_bytes_hashed": sum(len(raw) for raw in asset_bytes.values()),
        "recomputed_contrast_count": len(recomputed_contrasts),
        "recomputed_guard_contrast_count": len(recomputed_guards),
        "recomputed_contrasts_sha256": sha_bytes(canonical(recomputed_contrasts)),
        "recomputed_guard_contrasts_sha256": sha_bytes(canonical(recomputed_guards)),
        "stored_summary_sha256": sha_bytes(canonical(saved_summary)),
    }


def stream_member_to_path(stream: Any, output: Path, expected_bytes: int, cap: int) -> tuple[int, str]:
    if expected_bytes < 0 or expected_bytes > cap:
        raise ValueError(f"archive member exceeds caller-supplied per-member byte cap: {expected_bytes}")
    digest = hashlib.sha256()
    total = 0
    with output.open("xb") as target:
        while True:
            block = stream.read(BLOCK)
            if not block:
                break
            total += len(block)
            if total > expected_bytes or total > cap:
                raise ValueError("archive member exceeded its declared/caller byte bound")
            digest.update(block)
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if total != expected_bytes:
        raise ValueError("archive member byte count differs from its sealed declaration")
    return total, digest.hexdigest()


def run_media_readback(archives: Mapping[str, Mapping[str, Any]], joined: Mapping[str, Any],
                       core: Any, guard: Any, tools: Mapping[str, str], scratch_root: Path,
                       out_dir: Path, max_member_bytes: int, reserve_free_bytes: int) -> dict[str, Any]:
    ffmpeg, ffprobe = tools["ffmpeg"], tools["ffprobe"]
    plan_by_id = {row["row_id"]: row for row in joined["plan"]["rows"]}
    media_map_rows = joined["media_map"].get("rows", [])
    if (len(media_map_rows) != 540 or len({row.get("row_id") for row in media_map_rows}) != 540
            or {row.get("row_id") for row in media_map_rows} != set(plan_by_id)):
        raise ValueError("joined media map does not preserve the exact 540-row plan identities")
    for media_row in media_map_rows:
        row_id = media_row.get("row_id")
        planned = plan_by_id[row_id]
        method_id = planned.get("method_id")
        expected_role = ("samehost_controls" if method_id in {"identity", "rms_fixed"}
                         else "step120_candidate" if method_id == "shared_recipient_transition_head_v3_fit40_step120"
                         else None)
        archive_role = media_row.get("archive_role")
        if expected_role is None or archive_role != expected_role:
            raise ValueError(f"joined media map row has the wrong producer archive role: {row_id}")
        producer_row = archives[archive_role]["row_media_by_id"].get(row_id)
        if not isinstance(producer_row, dict):
            raise ValueError(f"joined media map row is absent from its pinned raw-media archive: {row_id}")
        producer_plan_row = archives[archive_role]["plan_by_id"].get(row_id)
        if not isinstance(producer_plan_row, dict):
            raise ValueError(f"pinned raw-media archive plan lacks its mapped row: {row_id}")
        for field in IDENTITY_FIELDS:
            if producer_plan_row.get(field) != planned.get(field):
                raise ValueError(f"raw-media producer plan differs from the joined plan at {field}: {row_id}")
        for field in ("row_id", "method_id", "sample_id", "speaker_id", "model_id", "seed"):
            if media_row.get(field) != planned.get(field):
                raise ValueError(f"joined media map differs from its frozen plan at {field}: {row_id}")
        producer_fields = {
            "media_status": "status", "source_pcm_sha256": "source_pcm_sha256",
            "source_pcm_file_sha256": "source_pcm_file_sha256", "silent_video": "silent_video",
            "muxed_video": "muxed_video", "timestamps": "timestamps", "fps_num": "fps_num",
            "fps_den": "fps_den", "frame_count": "frame_count",
            "video_stream_sha256": "video_stream_sha256", "mux_audio_readback": "mux_audio_readback",
        }
        for map_field, producer_field in producer_fields.items():
            if media_row.get(map_field) != producer_row.get(producer_field):
                raise ValueError(f"joined media map differs from its source archive at {map_field}: {row_id}")
    feature_path = ROOT / guard.FEATURE_SOURCE_REL
    landmarker_path = ROOT / "checkpoints/mediapipe/face_landmarker.task"
    if (feature_path.is_symlink() or not feature_path.is_file()
            or landmarker_path.is_symlink() or not landmarker_path.is_file()
            or sha_file(feature_path) != FEATURE_SOURCE_SHA256
            or sha_file(landmarker_path) != LANDMARKER_SHA256):
        raise ValueError("pinned visual-motion feature source or landmarker asset SHA mismatch")
    scratch_root.mkdir(parents=True, exist_ok=True)
    if scratch_root.is_symlink() or not scratch_root.is_dir():
        raise ValueError("media readback scratch root must be a real directory")
    max_asset_bytes = max(int(item.get("bytes", 0)) for archive in archives.values() for item in archive["assets"])
    if shutil.disk_usage(scratch_root).free < max_asset_bytes + reserve_free_bytes:
        raise ValueError("scratch filesystem lacks one largest media member plus its required free-space reserve")
    out_trajectories = out_dir / "trajectories"
    out_trajectories.mkdir()
    feature_module = guard._load_feature_module(feature_path)
    mediapipe_module, landmarker = None, None
    try:
        mediapipe_module, landmarker = feature_module.create_landmarker(landmarker_path)
        media_asset_records: dict[str, dict[str, Any]] = {}
        row_media_checks: dict[tuple[str, str], dict[str, Any]] = {}
        for role, archive in archives.items():
            manifest = archive["manifest"]
            assets_by_path = {item["path"]: item for item in archive["assets"]}
            if len(assets_by_path) != len(archive["assets"]):
                raise ValueError(f"{role} raw-media archive has duplicate asset member paths")
            refs_by_member: dict[str, list[str]] = defaultdict(list)
            for row_id, row_media in archive["row_media_by_id"].items():
                if row_media.get("status") != "PASS":
                    row_media_checks[(role, row_id)] = {"status": "INCOMPLETE", "producer_status": row_media.get("status")}
                    continue
                source_assets = [asset for asset in archive["assets"]
                                 if asset.get("role") == "source_pcm"
                                 and asset.get("sha256") == row_media.get("source_pcm_file_sha256")]
                if len(source_assets) != 1:
                    raise ValueError(f"{role} source PCM asset reference is ambiguous: {row_id}")
                refs_by_member[source_assets[0]["path"]].append(row_id)
                for asset_role in ("silent_video", "muxed_video", "timestamps"):
                    ref = row_media.get(asset_role)
                    if not isinstance(ref, dict) or ref.get("path") not in assets_by_path:
                        raise ValueError(f"{role} row lacks a sealed {asset_role} member: {row_id}")
                    refs_by_member[ref["path"]].append(row_id)
                row_media_checks[(role, row_id)] = {
                    "status": "PENDING", "source_pcm_sha256": row_media.get("source_pcm_sha256"),
                    "source_pcm_file_sha256": row_media.get("source_pcm_file_sha256"),
                    "silent_video_sha256": row_media.get("silent_video", {}).get("sha256"),
                    "muxed_video_sha256": row_media.get("muxed_video", {}).get("sha256"),
                    "timestamps_sha256": row_media.get("timestamps", {}).get("sha256"),
                    "video_stream_sha256": row_media.get("video_stream_sha256"),
                    "fps_num": row_media.get("fps_num"), "fps_den": row_media.get("fps_den"),
                    "frame_count": row_media.get("frame_count"),
                }
            visual_asset_records: dict[str, dict[str, Any]] = {}
            video_clock_records: dict[str, dict[str, Any]] = {}
            timestamp_clock_records: dict[str, dict[str, Any]] = {}
            media_probe_records: dict[str, dict[str, Any]] = {}
            seen: set[str] = set()
            with tempfile.TemporaryDirectory(prefix=f"joint-readback-{role}-", dir=scratch_root) as tmp_text:
                tmp_root = Path(tmp_text)
                with Path(archive["path"]).open("rb") as raw_stream:
                    with tarfile.open(fileobj=raw_stream, mode="r|gz") as tar:
                        for info in tar:
                            asset = assets_by_path.get(info.name)
                            if asset is None:
                                continue
                            if not info.isfile() or info.size != asset.get("bytes"):
                                raise ValueError(f"{role} archive member type/size changed: {info.name}")
                            tar_member = tar.extractfile(info)
                            if tar_member is None:
                                raise ValueError(f"{role} archive member cannot be read: {info.name}")
                            temp_path = tmp_root / (sha_bytes(info.name.encode("utf-8")) + Path(info.name).suffix)
                            byte_count, digest = stream_member_to_path(tar_member, temp_path,
                                                                      int(asset["bytes"]), max_member_bytes)
                            if byte_count != asset.get("bytes") or digest != asset.get("sha256"):
                                raise ValueError(f"{role} media member does not match its sealed SHA: {info.name}")
                            seen.add(info.name)
                            refs = refs_by_member.get(info.name, [])
                            try:
                                asset_role = asset.get("role")
                                if asset_role == "source_pcm":
                                    probe = core._ffprobe(temp_path, ffprobe)
                                    audio = [item for item in probe.get("streams", []) if item.get("codec_type") == "audio"]
                                    decoded = core._decode_bytes(temp_path, ffmpeg, codec="f32le")
                                    row_results = []
                                    for row_id in refs:
                                        planned = archive["plan_by_id"][row_id]
                                        natural = planned.get("natural_pcm", {})
                                        expected_pcm = natural.get("pcm_sha256")
                                        count = natural.get("sample_count")
                                        good = (len(audio) == 1 and audio[0].get("codec_name") == "pcm_f32le"
                                                and int(audio[0].get("sample_rate", 0)) == 16000
                                                and int(audio[0].get("channels", 0)) == 1
                                                and type(count) is int and len(decoded) == count * 4
                                                and sha_bytes(decoded) == expected_pcm)
                                        good = good and archive["row_media_by_id"][row_id].get("source_pcm_sha256") == expected_pcm
                                        row_results.append(good)
                                    media_probe_records[info.name] = {
                                        "role": asset_role, "bytes": byte_count, "sha256": digest,
                                        "decoded_pcm_bytes": len(decoded), "decoded_pcm_sha256": sha_bytes(decoded),
                                        "status": "PASS" if all(row_results) and row_results else "FAILED",
                                        "referencing_row_count": len(refs),
                                        "stream": audio[0] if len(audio) == 1 else None,
                                    }
                                elif asset_role in {"silent_video", "muxed_video"}:
                                    frame_refs = {(int(archive["row_media_by_id"][row_id]["fps_num"]),
                                                   int(archive["row_media_by_id"][row_id]["fps_den"]),
                                                   int(archive["row_media_by_id"][row_id]["frame_count"]))
                                                  for row_id in refs}
                                    if len(frame_refs) != 1:
                                        raise ValueError(f"shared media file has conflicting frozen frame clocks: {info.name}")
                                    fps_num, fps_den, frame_count = next(iter(frame_refs))
                                    frames, video_stream = core._frame_clock(temp_path, ffprobe, fps_num, fps_den)
                                    if len(frames) != frame_count:
                                        raise ValueError(f"decoded frame count differs from frozen row metadata: {info.name}")
                                    es_sha = core._video_es_sha(temp_path, ffmpeg)
                                    if any(archive["row_media_by_id"][row_id].get("video_stream_sha256") != es_sha
                                           for row_id in refs):
                                        raise ValueError(f"decoded video stream hash differs from its frozen row pin: {info.name}")
                                    video_clock_records[info.name] = {
                                        "video_stream_sha256": es_sha,
                                        "clock_frames": frames,
                                        "frame_count": frame_count,
                                        "fps_num": fps_num,
                                        "fps_den": fps_den,
                                    }
                                    if asset_role == "silent_video":
                                        sequence = feature_module.extract_video_features(
                                            temp_path, landmarker_path,
                                            landmarker_handle=(mediapipe_module, landmarker))
                                        if (sequence.metadata.get("decoded_frame_count") != frame_count
                                                or len(sequence.valid) != frame_count):
                                            raise ValueError(f"visual decoder frame count differs from sealed media clock: {info.name}")
                                        exact_times = [Fraction(item["timestamp_num"], item["timestamp_den"])
                                                       for item in frames]
                                        if not np.allclose(sequence.timestamps_s,
                                                          np.asarray([float(item) for item in exact_times]),
                                                          rtol=0.0, atol=1e-5):
                                            raise ValueError(f"visual decoder clock differs from FFprobe source clock: {info.name}")
                                        metrics, arrays = guard._trajectory_summary(sequence, exact_times)
                                        member_key = sha_bytes((role + "\0" + info.name).encode("utf-8"))
                                        trajectory_path = out_trajectories / f"{member_key}.npz"
                                        if trajectory_path.exists():
                                            raise FileExistsError(f"duplicate trajectory output: {trajectory_path}")
                                        npz_ref = guard._write_npz(trajectory_path, arrays)
                                        npz_ref["path"] = trajectory_path.relative_to(out_dir).as_posix()
                                        visual_asset_records[info.name] = {
                                            "status": "PASS", "file_sha256": digest,
                                            "video_stream_sha256": es_sha, "frame_count": frame_count,
                                            "fps_num": fps_num, "fps_den": fps_den,
                                            "ffprobe_stream": video_stream, "clock_frames": frames,
                                            "clock_signature": sha_bytes(canonical({
                                                "fps_num": fps_num, "fps_den": fps_den,
                                                "frame_count": frame_count,
                                                "timestamps": [[item.numerator, item.denominator] for item in exact_times],
                                            })),
                                            "visual_npz": npz_ref, "metrics": metrics,
                                        }
                                        media_probe_records[info.name] = {
                                            "role": asset_role, "bytes": byte_count, "sha256": digest,
                                            "video_stream_sha256": es_sha, "status": "PASS",
                                            "referencing_row_count": len(refs), "frame_count": frame_count,
                                            "fps_num": fps_num, "fps_den": fps_den,
                                            "clock_frames": frames,
                                            "visual_npz": npz_ref, "motion_metrics": metrics,
                                        }
                                    else:
                                        probe = core._ffprobe(temp_path, ffprobe)
                                        streams = probe.get("streams", [])
                                        audios = [item for item in streams if item.get("codec_type") == "audio"]
                                        videos = [item for item in streams if item.get("codec_type") == "video"]
                                        decoded = core._decode_bytes(temp_path, ffmpeg, codec="f32le")
                                        good_rows = []
                                        for row_id in refs:
                                            planned = archive["plan_by_id"][row_id]
                                            natural = planned.get("natural_pcm", {})
                                            expected_count = natural.get("sample_count")
                                            pcm_sha = natural.get("pcm_sha256")
                                            media_row = archive["row_media_by_id"][row_id]
                                            expected_frames = []
                                            for index in range(frame_count):
                                                expected_time = Fraction(index * fps_den, fps_num)
                                                expected_frames.append({
                                                    "frame_index": index,
                                                    "timestamp_num": expected_time.numerator,
                                                    "timestamp_den": expected_time.denominator,
                                                })
                                            good = (len(videos) == 1 and len(audios) == 1
                                                    and audios[0].get("codec_name") == "pcm_f32le"
                                                    and int(audios[0].get("sample_rate", 0)) == 16000
                                                    and int(audios[0].get("channels", 0)) == 1
                                                    and type(expected_count) is int and len(decoded) == expected_count * 4
                                                    and sha_bytes(decoded) == pcm_sha
                                                    and frames == expected_frames
                                                    and media_row.get("video_stream_sha256") == es_sha)
                                            good_rows.append(good)
                                        media_probe_records[info.name] = {
                                            "role": asset_role, "bytes": byte_count, "sha256": digest,
                                            "decoded_pcm_bytes": len(decoded), "decoded_pcm_sha256": sha_bytes(decoded),
                                            "video_stream_sha256": es_sha,
                                            "status": "PASS" if all(good_rows) and good_rows else "FAILED",
                                            "referencing_row_count": len(refs),
                                            "audio_stream": audios[0] if len(audios) == 1 else None,
                                            "video_stream": videos[0] if len(videos) == 1 else None,
                                            "clock_frames": frames,
                                        }
                                elif asset_role == "timestamps":
                                    frame_refs = {(int(archive["row_media_by_id"][row_id]["fps_num"]),
                                                   int(archive["row_media_by_id"][row_id]["fps_den"]),
                                                   int(archive["row_media_by_id"][row_id]["frame_count"]))
                                                  for row_id in refs}
                                    if len(frame_refs) != 1:
                                        raise ValueError(f"shared timestamp file has conflicting frozen clocks: {info.name}")
                                    fps_num, fps_den, frame_count = next(iter(frame_refs))
                                    exact_times = guard._clock_sidecar(temp_path, digest, frame_count, fps_num, fps_den)
                                    expected_frames = [{"frame_index": index, "timestamp_num": value.numerator,
                                                        "timestamp_den": value.denominator}
                                                       for index, value in enumerate(exact_times)]
                                    timestamp_clock_records[info.name] = {
                                        "expected_frames": expected_frames,
                                        "frame_count": frame_count,
                                        "fps_num": fps_num,
                                        "fps_den": fps_den,
                                        "referencing_rows": refs,
                                    }
                                    media_probe_records[info.name] = {
                                        "role": asset_role, "bytes": byte_count, "sha256": digest,
                                        "status": "PENDING_VIDEO_CLOCKS", "frame_count": frame_count,
                                        "fps_num": fps_num, "fps_den": fps_den,
                                    }
                                else:
                                    raise ValueError(f"unknown raw-media asset role: {asset_role}")
                                if asset_role == "silent_video":
                                    for row_id in refs:
                                        row_media_checks[(role, row_id)]["silent_video_status"] = "PASS"
                                        row_media_checks[(role, row_id)]["visual_status"] = "PASS"
                                        row_media_checks[(role, row_id)]["visual_npz"] = visual_asset_records[info.name]["visual_npz"]
                                        row_media_checks[(role, row_id)]["motion_metrics"] = visual_asset_records[info.name]["metrics"]
                                        row_media_checks[(role, row_id)]["clock_signature"] = visual_asset_records[info.name]["clock_signature"]
                                elif asset_role in {"source_pcm", "muxed_video"}:
                                    media_probe_records[info.name]["referencing_rows"] = refs
                                    if media_probe_records[info.name]["status"] != "PASS":
                                        for row_id in refs:
                                            row_media_checks[(role, row_id)][f"{asset_role}_status"] = "FAILED"
                                    else:
                                        for row_id in refs:
                                            row_media_checks[(role, row_id)][f"{asset_role}_status"] = "PASS"
                            except Exception as exc:
                                media_probe_records[info.name] = {
                                    "role": asset.get("role"), "status": "FAILED",
                                    "error_type": type(exc).__name__, "error": str(exc),
                                    "bytes": byte_count, "sha256": digest,
                                }
                                for row_id in refs:
                                    row_media_checks[(role, row_id)][f"{asset.get('role')}_status"] = "FAILED"
                            finally:
                                temp_path.unlink(missing_ok=True)
            if seen != set(assets_by_path):
                missing = sorted(set(assets_by_path) - seen)
                raise ValueError(f"{role} archive stream omitted declared media members: {missing[:5]}")
            for timestamp_path, timestamp_record in timestamp_clock_records.items():
                refs = timestamp_record["referencing_rows"]
                expected_frames = timestamp_record["expected_frames"]
                matching_video_paths = {
                    archive["row_media_by_id"][row_id][video_role]["path"]
                    for row_id in refs for video_role in ("silent_video", "muxed_video")
                }
                video_clocks = [video_clock_records.get(video_path) for video_path in matching_video_paths]
                expected_streams = {
                    archive["row_media_by_id"][row_id].get("video_stream_sha256") for row_id in refs
                }
                clock_ok = (len(expected_streams) == 1 and bool(video_clocks) and all(
                    item is not None and item.get("clock_frames") == expected_frames
                    and item.get("video_stream_sha256") in expected_streams
                    for item in video_clocks
                ))
                visual_ok = all(
                    archive["row_media_by_id"][row_id]["silent_video"]["path"] in visual_asset_records
                    for row_id in refs
                )
                status = "PASS" if clock_ok and visual_ok else "FAILED"
                media_probe_records[timestamp_path]["status"] = status
                if status == "PASS":
                    for row_id in refs:
                        row_media_checks[(role, row_id)]["timestamp_clock_status"] = "PASS"
                        row_media_checks[(role, row_id)]["timestamps_status"] = "PASS"
                        silent_path = archive["row_media_by_id"][row_id]["silent_video"]["path"]
                        row_media_checks[(role, row_id)]["clock_signature"] = visual_asset_records[silent_path]["clock_signature"]
                else:
                    media_probe_records[timestamp_path]["error"] = (
                        "timestamp sidecar differs from decoded silent/muxed video clocks, "
                        "or silent visual readback is incomplete"
                    )
                    for row_id in refs:
                        row_media_checks[(role, row_id)]["timestamp_clock_status"] = "FAILED"
                        row_media_checks[(role, row_id)]["timestamps_status"] = "FAILED"
                        row_media_checks[(role, row_id)]["visual_status"] = "FAILED"
            for row_id, record in archive["row_media_by_id"].items():
                check = row_media_checks[(role, row_id)]
                if record.get("status") != "PASS":
                    continue
                check["status"] = "PASS" if all(check.get(key) == "PASS" for key in (
                    "source_pcm_status", "silent_video_status", "muxed_video_status",
                    "timestamps_status", "visual_status")) else "FAILED"
                check["producer_status"] = "PASS"
            for member, record in media_probe_records.items():
                media_asset_records[f"{role}:{member}"] = record
        row_items = []
        for joint_row in joined["media_map"]["rows"]:
            role = joint_row.get("archive_role")
            archive_role = "samehost_controls" if role == "samehost_controls" else "step120_candidate"
            row_id = joint_row.get("row_id")
            checks = row_media_checks.get((archive_role, row_id), {"status": "MISSING"})
            row_items.append({
                "row_id": row_id, "method_id": joint_row.get("method_id"),
                "sample_id": joint_row.get("sample_id"), "speaker_id": joint_row.get("speaker_id"),
                "model_id": joint_row.get("model_id"), "seed": joint_row.get("seed"),
                "archive_role": archive_role, "media_status": checks.get("status"),
                "producer_media_status": checks.get("producer_status"),
                "media_eligible": False,
                "source_pcm_sha256": checks.get("source_pcm_sha256"),
                "source_pcm_file_sha256": checks.get("source_pcm_file_sha256"),
                "decoded_source_pcm_sha256": next((record.get("decoded_pcm_sha256")
                    for key, record in media_asset_records.items()
                    if key.startswith(archive_role + ":") and record.get("role") == "source_pcm"
                    and record.get("sha256") == checks.get("source_pcm_file_sha256")), None),
                "silent_video_sha256": checks.get("silent_video_sha256"),
                "muxed_video_sha256": checks.get("muxed_video_sha256"),
                "decoded_mux_pcm_sha256": next((record.get("decoded_pcm_sha256")
                    for key, record in media_asset_records.items()
                    if key.startswith(archive_role + ":") and record.get("role") == "muxed_video"
                    and record.get("sha256") == checks.get("muxed_video_sha256")), None),
                "timestamps_sha256": checks.get("timestamps_sha256"),
                "clock_signature": checks.get("clock_signature"),
                "fps_num": checks.get("fps_num"), "fps_den": checks.get("fps_den"),
                "frame_count": checks.get("frame_count"),
                "visual_status": checks.get("visual_status"),
                "motion_metrics": checks.get("motion_metrics"), "visual_npz": checks.get("visual_npz"),
            })
        if len(row_items) != 540 or len({row["row_id"] for row in row_items}) != 540:
            raise ValueError("joint media readback did not preserve the exact 540-row denominator")
        pcm_by_context: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        pcm_consistency_failures: list[dict[str, Any]] = []
        for row in row_items:
            context = (str(row["sample_id"]), str(row["speaker_id"]), str(row["model_id"]), int(row["seed"]))
            decoded_pcm = row.get("decoded_source_pcm_sha256")
            if isinstance(decoded_pcm, str) and decoded_pcm == row.get("source_pcm_sha256"):
                pcm_by_context[context].append(row)
            elif row.get("media_status") == "PASS":
                row["media_status"] = "FAILED"
                row["original_pcm_status"] = "FAILED"
                pcm_consistency_failures.append({"context": list(context), "row_id": row["row_id"],
                                                 "reason": "decoded PCM differs from the row's source PCM pin"})
            else:
                pcm_consistency_failures.append({"context": list(context), "row_id": row["row_id"],
                                                 "reason": "decoded source PCM is unavailable"})
        for context, context_rows in pcm_by_context.items():
            methods = {row["method_id"] for row in context_rows}
            digests = {row["decoded_source_pcm_sha256"] for row in context_rows}
            if methods != set(EXPECTED_METHODS) or len(context_rows) != 3 or len(digests) != 1:
                for row in context_rows:
                    row["media_status"] = "FAILED"
                    row["original_pcm_status"] = "FAILED"
                pcm_consistency_failures.append({"context": list(context),
                                                 "row_ids": sorted(row["row_id"] for row in context_rows),
                                                 "reason": "three methods do not share one decoded original PCM"})
            else:
                for row in context_rows:
                    row["original_pcm_status"] = "PASS"
        pcm_consistency_status = ("PASS" if len(pcm_by_context) == 180 and not pcm_consistency_failures
                                  else "INCOMPLETE")
        # Eligibility reflects the final status after decoded PCM and three-method context checks.
        for row in row_items:
            row["media_eligible"] = row.get("media_status") == "PASS"
        pass_count = sum(row["media_status"] == "PASS" for row in row_items)
        return {
            "status": "PASS" if pass_count == 540 and pcm_consistency_status == "PASS" else "INCOMPLETE",
            "row_count": len(row_items), "row_pass_count": pass_count,
            "row_status_counts": {state: sum(row.get("media_status") == state for row in row_items)
                                  for state in ("PASS", "FAILED", "INCOMPLETE", "MISSING")},
            "producer_row_status_counts": {state: sum(row.get("producer_media_status") == state for row in row_items)
                                           for state in ("PASS", "FAILED", "INCOMPLETE", "MISSING")},
            "rows": row_items, "asset_count": len(media_asset_records),
            "original_pcm_context_count": len(pcm_by_context),
            "original_pcm_consistency_status": pcm_consistency_status,
            "original_pcm_consistency_failures": pcm_consistency_failures,
            "asset_status_counts": {state: sum(record.get("status") == state for record in media_asset_records.values())
                                    for state in ("PASS", "FAILED")},
            "assets": media_asset_records,
            "feature_source_sha256": FEATURE_SOURCE_SHA256,
            "landmarker_asset_sha256": LANDMARKER_SHA256,
            "visual_metrics_definition": "frozen eye-normalized mouth geometry on the sealed decoded video frame clock; descriptive only",
            "resampling_or_retiming_performed": False,
        }
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    source_closure = verify_source_closure()
    for path, label, expect_dir in (
        (args.runtime_binding, "runtime binding", False),
        (args.join_dir, "joint output", True),
        (args.baseline_archive, "baseline raw-media archive", False),
        (args.candidate_archive, "candidate raw-media archive", False),
    ):
        if path.is_symlink() or (not path.is_dir() if expect_dir else not path.is_file()):
            raise ValueError(f"{label} has an unsafe or unexpected path type: {path}")
    for path, label in ((args.out_dir, "output"), (args.scratch_root, "scratch")):
        if path.is_symlink():
            raise ValueError(f"{label} path cannot be a symlink: {path}")
    binding_meta, tools = validate_runtime_binding(args.runtime_binding.resolve(strict=True))
    joiner = load_pinned_module(ROOT / JOINER_REL, "frozen_540_joiner_for_readback", JOINER_SHA256)
    core = load_pinned_module(ROOT / STAGEA_CORE_REL, "frozen_stagea_core_for_readback", STAGEA_CORE_SHA256)
    guard = load_pinned_module(ROOT / VISUAL_GUARD_REL, "frozen_visual_motion_guard_for_readback", VISUAL_GUARD_SHA256)
    joined = read_join(args.join_dir.resolve(strict=True), joiner)
    base_ref = joined["manifest"]["raw_media_inputs"]["baseline_archive"]
    candidate_ref = joined["manifest"]["raw_media_inputs"]["candidate_archive"]
    baseline = load_raw_archive(args.baseline_archive.resolve(strict=True), "samehost_controls", base_ref, joiner)
    candidate = load_raw_archive(args.candidate_archive.resolve(strict=True), "step120_candidate", candidate_ref, joiner)
    archives = {"samehost_controls": baseline, "step120_candidate": candidate}
    sync = load_sync_readback(joined, joiner, core)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"readback output must be a fresh real directory: {out_dir}")
    out_dir.mkdir(parents=True)
    scratch = args.scratch_root.resolve()
    started = time.time()
    try:
        media = run_media_readback(archives, joined, core, guard, tools, scratch,
                                   out_dir, args.max_member_bytes, args.reserve_free_bytes)
        sync_lines = b"".join(canonical(sync["row_records"][row["row_id"]]) + b"\n"
                               for row in joined["plan"]["rows"])
        media_lines = b"".join(canonical(row) + b"\n" for row in media["rows"])
        write_exclusive(out_dir / "joint_syncnet_readback.jsonl", sync_lines)
        write_exclusive(out_dir / "joint_media_motion_readback.jsonl", media_lines)
        media_core = {key: value for key, value in media.items() if key not in {"rows", "assets"}}
        write_exclusive(out_dir / "media_asset_readback.json", pretty(media_core))
        write_exclusive(out_dir / "media_asset_readback_details.json", pretty(media["assets"]))
        status = "PASS" if sync["status"] == "PASS" and media["status"] == "PASS" else "INCOMPLETE"
        receipt: dict[str, Any] = {
            "schema": "cross_tfg_v3_joint_media_motion_sync_readback_v1",
            "status": status, "run_id": joined["manifest"].get("run_id"),
            "planned_rows": 540, "score_denominator_preserved": True,
            "syncnet_readback_status": sync["status"],
            "media_motion_readback_status": media["status"],
            "row_status_counts": media["row_status_counts"],
            "raw_media_archive_pins": {
                key: {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"],
                      "manifest_raw_sha256": item["manifest_raw_sha256"]}
                for key, item in archives.items()},
            "candidate_w2l_native_output_bindings": candidate["candidate_native_output_bindings"],
            "candidate_w2l_generator_pcm_bytes_in_raw_media_archive": False,
            "joined_manifest": {"path": str((joined["join_dir"] / "join_manifest.json").resolve()),
                                "sha256": joined["manifest_raw_sha256"],
                                "payload_sha256": joined["manifest"].get("manifest_sha256")},
            "runtime_binding": binding_meta, "tool_paths": tools,
            "source_closure": source_closure,
            "syncnet_readback": {key: value for key, value in sync.items() if key != "row_records"},
            "media_motion_readback": media_core,
            "output_files": [], "started_unix": started, "ended_unix": time.time(),
            "model_execution_performed": False, "syncnet_rescoring_performed": False,
            "audio_or_video_generation_performed": False, "confirm_data_opened": False,
            "method_selected": False,
        }
        output_files = []
        for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
            output_files.append({"path": path.relative_to(out_dir).as_posix(),
                                 "bytes": path.stat().st_size, "sha256": sha_file(path)})
        receipt["output_files"] = output_files
        receipt["receipt_sha256"] = payload_sha(receipt, "receipt_sha256")
        receipt_raw = pretty(receipt)
        write_exclusive(out_dir / "readback_receipt.json", receipt_raw)
        print(json.dumps({"status": status, "output_dir": str(out_dir),
                          "planned_rows": 540, "syncnet_readback_status": sync["status"],
                          "media_motion_readback_status": media["status"],
                          "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
        return 0 if status == "PASS" else 2
    except Exception as exc:
        failure = {
            "schema": "cross_tfg_v3_joint_media_motion_sync_readback_v1",
            "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "started_unix": started,
            "ended_unix": time.time(), "confirm_data_opened": False,
            "method_selected": False, "model_execution_performed": False,
            "syncnet_rescoring_performed": False, "audio_or_video_generation_performed": False,
        }
        failure["receipt_sha256"] = payload_sha(failure, "receipt_sha256")
        failure_path = out_dir / "readback_failure.json"
        if not failure_path.exists():
            write_exclusive(failure_path, pretty(failure))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-dir", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm/cross_tfg_joint_media_motion_sync_readback"))
    parser.add_argument("--max-member-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--reserve-free-bytes", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args(argv)
    if args.max_member_bytes <= 0 or args.reserve_free_bytes < 0:
        parser.error("byte caps must be positive and the reserve nonnegative")
    try:
        return run(args)
    except Exception as exc:
        print(f"joint_readback_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
