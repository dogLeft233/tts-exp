#!/usr/bin/env python3
"""Extract sealed WORLD v6 face tracks on recipient clocks, without scoring effects."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

GRID_INVENTORY_SHA256 = "7b3e879b9a34a6ea20159e770d6530fc77b19fd013381ce4017e87cbbc19203b"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
VIDEO_FEATURES_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
VISUAL_CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
NATIVE_EXTRACTOR_SHA256 = "a9d5fa9879acfc53cd2578d983bf343aaefedf4fbfa4bd36f38b00797e01817d"
NATIVE_CONSUMER_SHA256 = "7311aecb3ef0d6e622b40254f595003d60d7964dee17a4d207b7012d0403e4b2"
TONE_ADAPTER_SHA256 = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
CONSUMER_SHA256 = "9feaf39eb6cad01c9dc76d86535fae4ce95379a8ed775a2e70e0bc93286eb056"
RUNNER_SHA256 = "275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac"
WRAPPER_SHA256 = "708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa"
PLAN_RAW_SHA256 = "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d"
PLAN_FINGERPRINT_SHA256 = "1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc"
QA04_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
CONSUMER_RUN = Path("/dev/shm/tts-research-20261001/independent_validation/world_consumer04_attempt02/output")
EXPECTED_VISUAL_IDS = {"V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E"}
EXPECTED_CONTEXTS = 160
EXPECTED_VIDEOS = 1600
LATENCY_SECONDS = -0.085

_FEATURES = None
_LANDMARKER_HANDLE = None
_LANDMARKER_PATH = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {path.name}:{line_no}")
                result.append(value)
    return result


def parse_phone_tier(path: Path) -> list[dict[str, Any]]:
    """Retain the exact recipient `phones` tier intervals and labels."""
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    rows: list[dict[str, Any]] = []
    tier_active = False
    saw_phone_tier = False
    current: dict[str, Any] = {}
    in_interval = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"item\s*\[\d+\]:", stripped):
            tier_active = False
            in_interval = False
            current = {}
        if stripped.startswith('name = "'):
            name = stripped.split("=", 1)[1].strip().strip('"')
            tier_active = name.casefold() == "phones"
            saw_phone_tier |= tier_active
        if not tier_active:
            continue
        if re.match(r"intervals\s*\[\d+\]:", stripped):
            in_interval = True
            current = {}
        elif in_interval and stripped.startswith("xmin ="):
            current["start_s"] = float(stripped.split("=", 1)[1].strip())
        elif in_interval and stripped.startswith("xmax ="):
            current["end_s"] = float(stripped.split("=", 1)[1].strip())
        elif in_interval and stripped.startswith("text ="):
            raw = stripped.split("=", 1)[1].strip()
            if not (raw.startswith('"') and raw.endswith('"')) or set(current) != {"start_s", "end_s"}:
                raise ValueError(f"malformed phone interval in {path}")
            label = raw[1:-1].replace('""', '"')
            start, end = float(current["start_s"]), float(current["end_s"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError(f"invalid phone interval in {path}")
            rows.append({"phone_index": len(rows), "label": label, "start_s": start, "end_s": end})
            in_interval = False
            current = {}
    if not saw_phone_tier or not rows:
        raise ValueError(f"TextGrid has no nonempty phones tier: {path}")
    return rows


def verify_consumer_output(consumer_dir: Path, run_dir: Path) -> dict[str, Any]:
    audit_path = consumer_dir / "world_consumer_audit.json"
    seal_path = consumer_dir / "consumer_output_seal.json"
    inventory_path = consumer_dir / "consumer_all_files_sha256.json"
    if not all(path.is_file() for path in (audit_path, seal_path, inventory_path)):
        raise ValueError("consumer_output_seal_or_audit_missing")
    audit, seal, inventory = map(read_json, (audit_path, seal_path, inventory_path))
    if audit.get("status") != "PASS" or audit.get("error_count") != 0:
        raise ValueError("independent_consumer_audit_not_pass")
    if seal.get("status") != "SEALED" or seal.get("producer_run_root") != str(run_dir.resolve()):
        raise ValueError("consumer_output_seal_binding_mismatch")
    if seal.get("consumer_source_sha256") != CONSUMER_SHA256:
        raise ValueError("consumer_source_binding_mismatch")
    if seal.get("producer_plan_raw_file_sha256") != PLAN_RAW_SHA256 or seal.get("producer_plan_embedded_fingerprint_sha256") != PLAN_FINGERPRINT_SHA256:
        raise ValueError("consumer_plan_binding_mismatch")
    if seal.get("producer_qa04_sha256") != QA04_SHA256:
        raise ValueError("consumer_qa04_binding_mismatch")
    if sha256_file(inventory_path) != seal.get("inventory_sha256"):
        raise ValueError("consumer_output_inventory_hash_mismatch")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise ValueError("consumer_output_inventory_schema_mismatch")
    actual = {p.relative_to(consumer_dir).as_posix() for p in consumer_dir.rglob("*") if p.is_file() and p not in {inventory_path, seal_path}}
    if actual != set(expected):
        raise ValueError("consumer_output_inventory_coverage_mismatch")
    for relative, expected_hash in expected.items():
        path = consumer_dir / relative
        if sha256_file(path) != expected_hash:
            raise ValueError(f"consumer_output_member_hash_mismatch:{relative}")
    stats = read_json(consumer_dir / "world_consumer_statistics.json")
    if stats.get("status") != "COMPLETE" or stats.get("effects_computed") is not True:
        raise ValueError("consumer_statistics_not_complete")
    return {"audit_sha256": sha256_file(audit_path), "seal_sha256": sha256_file(seal_path), "inventory_sha256": sha256_file(inventory_path), "inventory_count": len(expected)}


def verify_producer(run_dir: Path, execution_plan_path: Path, consumer_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.resolve(strict=True)
    consumer_receipt = verify_consumer_output(consumer_dir.resolve(strict=True), run_dir)
    plan_path = execution_plan_path.resolve(strict=True)
    if sha256_file(plan_path) != PLAN_RAW_SHA256:
        raise ValueError("producer_execution_plan_raw_sha_mismatch")
    plan = read_json(plan_path)
    plan_fp = hashlib.sha256(canonical_json({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest()
    if plan.get("plan_sha256") != PLAN_FINGERPRINT_SHA256 or plan_fp != PLAN_FINGERPRINT_SHA256:
        raise ValueError("producer_execution_plan_fingerprint_mismatch")
    status = read_json(run_dir / "run_status.json")
    expected_status = {"status":"complete", "completed_contexts":160, "failed_contexts":0, "scored_rows":12160, "failed_rows":0, "unique_matrix_artifacts":5760, "unique_video_artifacts":1600, "execution_plan_sha256":PLAN_FINGERPRINT_SHA256}
    if any(status.get(k) != v for k,v in expected_status.items()):
        raise ValueError("producer_status_not_complete_or_counts_mismatch")
    protocol = read_json(run_dir / "evaluation_protocol.json")
    if protocol.get("execution_plan_sha256") != PLAN_FINGERPRINT_SHA256:
        raise ValueError("producer_protocol_plan_fingerprint_mismatch")
    results = read_jsonl(run_dir / "evaluation_results.jsonl")
    rows = read_jsonl(run_dir / "evaluation_rows.jsonl")
    plan_rows = {r.get("row_id"): r for r in plan.get("rows", [])}
    result_rows = {r.get("row_id"): r for r in results}
    plan_row_ids = {r.get("row_id") for r in rows}
    if len(plan_rows) != 12160 or len(result_rows) != 12160 or len(plan_row_ids) != 12160:
        raise ValueError("producer_row_cardinality_mismatch")
    if set(plan_rows) != set(result_rows) or set(plan_row_ids) != set(result_rows):
        raise ValueError("producer_row_id_join_mismatch")
    joined = []
    frozen_keys = ("ordered_eval_index", "sample_id", "speaker_id", "split", "direction", "visual_id")
    for row_id, planned in plan_rows.items():
        result = result_rows[row_id]
        if result.get("status") != "scored":
            raise ValueError(f"producer_result_not_scored:{row_id}")
        for key in frozen_keys:
            if planned.get(key) != result.get(key):
                raise ValueError(f"producer_plan_result_identity_mismatch:{key}")
        if result.get("visual_id") not in EXPECTED_VISUAL_IDS:
            raise ValueError("unexpected_visual_id")
        joined.append(result)
    context_rows: dict[tuple[str,str], list[dict[str,Any]]] = defaultdict(list)
    videos: dict[tuple[str,str,str], dict[str,Any]] = {}
    for row in joined:
        context_rows[(str(row["sample_id"]), str(row["direction"]))].append(row)
        key = (str(row["sample_id"]), str(row["direction"]), str(row["visual_id"]))
        item = {k: row.get(k) for k in ("sample_id","speaker_id","split","direction","visual_id","row_id","video_path","video_sha256","visual_npz","visual_npz_sha256")}
        if key in videos:
            prior = videos[key]
            for field in ("video_path","video_sha256","visual_npz","visual_npz_sha256"):
                if prior.get(field) != item.get(field):
                    raise ValueError(f"video_alias_binding_mismatch:{field}")
        else:
            videos[key] = item
    if len(context_rows) != EXPECTED_CONTEXTS or any(len(v) != 76 for v in context_rows.values()):
        raise ValueError("producer_context_alias_coverage_mismatch")
    if len(videos) != EXPECTED_VIDEOS or {k[2] for k in videos} != EXPECTED_VISUAL_IDS:
        raise ValueError("producer_visual_video_coverage_mismatch")
    return joined, list(videos.values()), {"run_status_sha256":sha256_file(run_dir/"run_status.json"), "protocol_sha256":sha256_file(run_dir/"evaluation_protocol.json"), "rows_sha256":sha256_file(run_dir/"evaluation_rows.jsonl"), "results_sha256":sha256_file(run_dir/"evaluation_results.jsonl"), "all_files_sha256":sha256_file(run_dir/"all_files_sha256.json"), "plan_raw_sha256":PLAN_RAW_SHA256, "plan_fingerprint_sha256":PLAN_FINGERPRINT_SHA256, "consumer":consumer_receipt, "run_root":str(run_dir)}


def verify_grid_inventory(grid_inventory_path: Path, joined_videos: list[dict[str, Any]]) -> dict[tuple[str,str], dict[str,Any]]:
    if sha256_file(grid_inventory_path) != GRID_INVENTORY_SHA256:
        raise ValueError("grid_inventory03_sha_mismatch")
    inventory = read_json(grid_inventory_path)
    if inventory.get("status") != "PASS":
        raise ValueError("grid_inventory03_not_pass")
    effective = inventory.get("effective_recipient_grids")
    if not isinstance(effective, list) or len(effective) != EXPECTED_CONTEXTS:
        raise ValueError("effective_recipient_grid_count_mismatch")
    by_context = {}
    for ref in effective:
        key = (str(ref.get("sample_id")), str(ref.get("direction")))
        if key in by_context:
            raise ValueError("duplicate_effective_recipient_grid")
        if ref.get("timeline_policy") != "recipient clock" or ref.get("role") != "effective_recipient_visual_clock":
            raise ValueError("effective_grid_role_or_timeline_mismatch")
        if ref.get("condition") != ("natural" if key[1] == "tts_to_natural" else "tts"):
            raise ValueError("recipient_condition_timeline_mismatch")
        path = Path(ref.get("path", ""))
        if not path.is_file() or sha256_file(path) != ref.get("sha256"):
            raise ValueError(f"recipient_grid_file_hash_mismatch:{key[0]}:{key[1]}")
        all_refs = [x for x in inventory.get("all_grid_refs", []) if x.get("sample_id") == key[0]]
        overlay = bool(ref.get("overlay_used"))
        if overlay and key[0] != "S0064_BAC009S0064W0121":
            raise ValueError("unexpected_physical_alignment_overlay")
        by_context[key] = {"effective":ref, "all_grid_refs":all_refs, "overlay_used":overlay}
    observed_contexts = {(str(x.get("sample_id")),str(x.get("direction"))) for x in joined_videos}
    if set(by_context) != observed_contexts:
        raise ValueError("grid_inventory_context_set_differs_from_video_set")
    return by_context


def check_resources(output_dir: Path, workers: int) -> dict[str, int]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    df = subprocess.run(["df", "-PB1", str(output_dir.parent)], capture_output=True, text=True, check=False)
    if df.returncode != 0:
        raise RuntimeError("tmpfs_df_unavailable")
    try:
        tmpfs_free = int(df.stdout.splitlines()[1].split()[3])
    except Exception as exc:
        raise RuntimeError("tmpfs_df_parse_failure") from exc
    meminfo = Path("/proc/meminfo").read_text()
    available_kb = int(re.search(r"^MemAvailable:\s+(\d+)", meminfo, re.M).group(1))
    cgroup_current_path = Path("/sys/fs/cgroup/memory.current")
    cgroup_max_path = Path("/sys/fs/cgroup/memory.max")
    if cgroup_current_path.exists() and cgroup_max_path.exists():
        current = int(cgroup_current_path.read_text().strip())
        max_text = cgroup_max_path.read_text().strip()
        cgroup_headroom = (2**63-1) if max_text == "max" else int(max_text)-current
    else:
        cgroup_headroom = 2**63-1
    if tmpfs_free < 2 * 1024**3 or available_kb * 1024 < 8 * 1024**3 or cgroup_headroom < 4 * 1024**3:
        raise RuntimeError("visual_resource_guard_failed")
    return {"tmpfs_free_bytes":tmpfs_free,"host_mem_available_bytes":available_kb*1024,"cgroup_headroom_bytes":cgroup_headroom,"workers":workers}


def initialize_worker(helper_path: str, target: str) -> None:
    global _FEATURES, _LANDMARKER_HANDLE, _LANDMARKER_PATH
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    sys.path.insert(0, str(Path(helper_path).resolve().parent))
    import video_features as features
    _FEATURES = features
    _LANDMARKER_PATH = Path(target).resolve()
    _LANDMARKER_HANDLE = features.create_landmarker(_LANDMARKER_PATH)


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def extract_one(task: dict[str, Any]) -> dict[str, Any]:
    video = Path(task["video_path"]).resolve(strict=True)
    if sha256_file(video) != task["video_sha256"]:
        raise ValueError("video_sha_mismatch")
    grid = Path(task["recipient_grid_path"]).resolve(strict=True)
    if sha256_file(grid) != task["recipient_grid_sha256"]:
        raise ValueError("recipient_grid_sha_mismatch")
    phone_intervals = parse_phone_tier(grid)
    sequence = _FEATURES.extract_video_features(video, _LANDMARKER_PATH, landmarker_handle=_LANDMARKER_HANDLE)
    metadata = dict(sequence.metadata)
    metadata.update({
        "visual_track_protocol": "world_visual_tracks_20261001_v1",
        "sample_id": task["sample_id"], "speaker_id": task["speaker_id"], "split": task["split"],
        "direction": task["direction"], "visual_id": task["visual_id"], "row_id": task["row_id"],
        "video_sha256": task["video_sha256"], "video_path_from_producer": task["video_path_from_producer"],
        "recipient_grid_path": str(grid), "recipient_grid_sha256": task["recipient_grid_sha256"],
        "recipient_grid_condition": task["recipient_grid_condition"], "recipient_grid_overlay_used": task["recipient_grid_overlay_used"],
        "grid_inventory_all_refs_for_context": task["grid_inventory_all_refs_for_context"],
        "timeline_policy": "recipient clock; no donor grid used",
        "recipient_phone_intervals": phone_intervals,
        "fixed_latency_s": LATENCY_SECONDS,
        "feature_helper_sha256": VIDEO_FEATURES_SHA256,
        "native_visual_protocol_id": "tts_mechanism_visual_20260930_v2",
        "native_visual_core_sha256": VISUAL_CORE_SHA256,
        "native_visual_extractor_sha256": NATIVE_EXTRACTOR_SHA256,
        "native_visual_consumer_v2_sha256": NATIVE_CONSUMER_SHA256,
        "native_visual_tone_adapter_sha256": TONE_ADAPTER_SHA256,
        "landmarker_asset_sha256": LANDMARKER_SHA256,
        "mediapipe_version": task["mediapipe_version"], "opencv_version": task["opencv_version"], "numpy_version": task["numpy_version"],
    })
    sequence = replace(sequence, metadata=metadata)
    target = Path(task["feature_path"])
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError("refusing_existing_visual_feature")
    target.parent.mkdir(parents=True, exist_ok=True)
    _FEATURES.save_visual_sequence(target, sequence)
    metadata_path = target.with_suffix(".json")
    arrays = {
        "landmarks":sequence.landmarks, "canonical_mouth":sequence.canonical_mouth, "valid":sequence.valid,
        "mouth_open":sequence.mouth_open,"mouth_width":sequence.mouth_width,"mouth_height":sequence.mouth_height,
        "mouth_area":sequence.mouth_area,"timestamps_s":sequence.timestamps_s,
    }
    result = {
        "status":"complete", "row_id":task["row_id"], "sample_id":task["sample_id"], "speaker_id":task["speaker_id"],
        "direction":task["direction"],"visual_id":task["visual_id"],"video_path":task["video_path_from_producer"],
        "video_sha256":task["video_sha256"],"video_bytes":video.stat().st_size,
        "feature_path":str(target),"feature_sha256":sha256_file(target),"feature_metadata_path":str(metadata_path),
        "feature_metadata_sha256":sha256_file(metadata_path),"decoded_frame_count":sequence.metadata["decoded_frame_count"],
        "declared_frame_count":sequence.metadata["declared_frame_count"],"fps":sequence.metadata["fps"],
        "width":sequence.metadata["width"],"height":sequence.metadata["height"],"valid_frames":int(sequence.valid.sum()),
        "invalid_frames":int((~sequence.valid).sum()),"recipient_grid_sha256":task["recipient_grid_sha256"],
        "recipient_phone_intervals_count":len(phone_intervals),
        "arrays":{name:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":array_sha(value)} for name,value in arrays.items()},
    }
    atomic_json(metadata_path, result)
    return result


def source_code_hash() -> str:
    return sha256_file(Path(__file__).resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--consumer-output", type=Path, default=CONSUMER_RUN)
    parser.add_argument("--grid-inventory", type=Path, required=True)
    parser.add_argument("--feature-helper", type=Path, required=True)
    parser.add_argument("--landmarker-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    run = args.run_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise SystemExit("refusing_existing_visual_output_namespace")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("workers_must_be_1_to_8")
    for name, path, expected in (
        ("landmarker_asset", args.landmarker_asset, LANDMARKER_SHA256),
        ("feature_helper", args.feature_helper, VIDEO_FEATURES_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"{name}_sha256_mismatch")
    resources = check_resources(output, args.workers)
    joined_rows, video_rows, producer_seals = verify_producer(run, args.execution_plan, args.consumer_output)
    grid_map = verify_grid_inventory(args.grid_inventory, video_rows)
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        raise SystemExit(f"isolated_mediapipe_runtime_unavailable:{type(exc).__name__}")
    if np.__version__ != "1.26.4" or cv2.__version__ != "4.10.0" or mp.__version__ != "0.10.21":
        raise SystemExit("visual_runtime_version_pin_mismatch")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    # Create the landmarker in the parent once as an API/resource smoke test,
    # then close it before worker processes create their own handles.
    import importlib.util
    sys.path.insert(0, str(args.feature_helper.resolve().parent))
    import video_features
    _mp, smoke_task = video_features.create_landmarker(args.landmarker_asset.resolve())
    smoke_task.close()
    if args.preflight_only:
        print(json.dumps({"status":"PREFLIGHT_PASS","contexts":160,"videos":1600,"grid_inventory_sha256":GRID_INVENTORY_SHA256,"feature_helper_sha256":VIDEO_FEATURES_SHA256,"landmarker_sha256":LANDMARKER_SHA256,"mediapipe_version":mp.__version__,"opencv_version":cv2.__version__,"numpy_version":np.__version__,"resources":resources,"producer_seals":producer_seals},sort_keys=True,separators=(",",":")))
        return 0

    output.mkdir(parents=True, exist_ok=False)
    tracks_dir = output / "tracks"
    progress_path = output / "progress.jsonl"
    tasks = []
    versions = {"mediapipe_version":mp.__version__,"opencv_version":cv2.__version__,"numpy_version":np.__version__}
    by_context = {(v["sample_id"],v["direction"]):v for v in video_rows}
    for row in video_rows:
        context_key=(str(row["sample_id"]),str(row["direction"]))
        grid=grid_map[context_key]["effective"]
        stem=f"{row['sample_id']}__{row['direction']}__{row['visual_id']}"
        tasks.append({
            **{k:row[k] for k in ("row_id","sample_id","speaker_id","split","direction","visual_id","video_sha256")},
            "video_path_from_producer":row["video_path"],
            "video_path":str((run / row["video_path"]).resolve()),
            "feature_path":str(tracks_dir/(stem+".npz")),
            "recipient_grid_path":grid["path"],"recipient_grid_sha256":grid["sha256"],
            "recipient_grid_condition":grid["condition"],"recipient_grid_overlay_used":grid["overlay_used"],
            "grid_inventory_all_refs_for_context":grid["all_grid_refs"],
            **versions,
        })
    if len(tasks)!=EXPECTED_VIDEOS:
        raise SystemExit("task_count_mismatch")
    manifest = {
        "schema_version":1,"status":"PLAN_FROZEN","classification":"sealed WORLD v6 recipient-clock video feature extraction; no score/effect analysis",
        "protocol_id":"world_visual_tracks_20261001_v1","producer":producer_seals,
        "grid_inventory":{"path":str(args.grid_inventory.resolve()),"sha256":GRID_INVENTORY_SHA256,"contexts":len(grid_map),"timeline":"effective recipient phone grid only; phone map not used as video clock"},
        "visual_ids":sorted(EXPECTED_VISUAL_IDS),"intended":{"contexts":160,"videos":1600,"video_refs_per_context":10,"workers":args.workers},
        "latency_s":LATENCY_SECONDS,"latency_use":"frozen downstream phone endpoint; this extraction stores full recipient clock and does not shift timestamps",
        "feature_source":{"path":str(args.feature_helper.resolve()),"sha256":VIDEO_FEATURES_SHA256,"schema":"VisualSequence NPZ: landmarks, canonical_mouth, valid, mouth_open, mouth_width, mouth_height, mouth_area, timestamps_s, metadata_json"},
        "native_visual_lineage":{"protocol_id":"tts_mechanism_visual_20260930_v2","analysis_core_sha256":VISUAL_CORE_SHA256,"native_extractor_sha256":NATIVE_EXTRACTOR_SHA256,"native_consumer_v2_sha256":NATIVE_CONSUMER_SHA256,"tone_adapter_sha256":TONE_ADAPTER_SHA256,"velocity_definition":"absolute adjacent-frame difference of mouth_open divided by timestamp delta, only when both frames are valid; derived by downstream frozen consumer"},
        "landmarker":{"path":str(args.landmarker_asset.resolve()),"sha256":LANDMARKER_SHA256},"runtime":versions,
        "runtime_resource_preflight":resources,"extractor_source":{"path":str(Path(__file__).resolve()),"sha256":source_code_hash()},
        "output_path":str(output),"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    }
    atomic_json(output/"visual_track_protocol.json",manifest)
    completed=[]; errors=[]
    with progress_path.open("x",encoding="utf-8") as journal:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(str(args.feature_helper.resolve()),str(args.landmarker_asset.resolve()))) as pool:
            future_map={pool.submit(extract_one,task):task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                task=future_map[future]
                try:
                    result=future.result()
                    completed.append(result)
                    record={"status":"complete","row_id":result["row_id"],"sample_id":result["sample_id"],"direction":result["direction"],"visual_id":result["visual_id"],"feature_sha256":result["feature_sha256"],"decoded_frame_count":result["decoded_frame_count"]}
                except Exception as exc:
                    result={"status":"failed","row_id":task["row_id"],"sample_id":task["sample_id"],"direction":task["direction"],"visual_id":task["visual_id"],"video_sha256":task["video_sha256"],"error_type":type(exc).__name__,"error":str(exc)[:1000]}
                    errors.append(result); record=result
                journal.write(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n")
                journal.flush(); os.fsync(journal.fileno())
                done=len(completed)+len(errors)
                if done%20==0 or done==len(tasks):
                    print(json.dumps({"done":done,"completed":len(completed),"failed":len(errors),"planned":len(tasks)},sort_keys=True,separators=(",",":")),flush=True)
    status_payload={
        "schema_version":1,"status":"COMPLETE" if len(completed)==EXPECTED_VIDEOS and not errors else "COMPLETE_WITH_FAILURES",
        "planned_videos":EXPECTED_VIDEOS,"completed_videos":len(completed),"failed_videos":len(errors),
        "progress_sha256":sha256_file(progress_path),"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "errors":errors,
    }
    status_path=output/"visual_track_status.json"
    atomic_json(status_path,status_payload)
    excluded={"all_files_sha256.json","visual_track_status.json","visual_track_seal.json"}
    files={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in excluded}
    all_path=output/"all_files_sha256.json"
    atomic_json(all_path,{"files":files})
    complete=len(completed)==EXPECTED_VIDEOS and not errors
    seal={"schema_version":1,"status":"SEALED" if complete else "SEALED_WITH_FAILURES","protocol_sha256":sha256_file(output/"visual_track_protocol.json"),"status_sha256":sha256_file(status_path),"all_files_sha256":sha256_file(all_path),"all_files_excluded":["all_files_sha256.json","visual_track_status.json","visual_track_seal.json"],"file_count":len(files),"completed_videos":len(completed),"failed_videos":len(errors)}
    atomic_json(output/"visual_track_seal.json",seal)
    print(json.dumps({"status":manifest["status"],"completed_videos":len(completed),"failed_videos":len(errors),"seal_sha256":sha256_file(output/"visual_track_seal.json"),"all_files_sha256":sha256_file(all_path)},sort_keys=True,separators=(",",":")))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
