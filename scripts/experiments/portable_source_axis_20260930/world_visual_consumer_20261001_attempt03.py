#!/usr/bin/env python3
"""Availability-first visual analysis for the sealed WORLD v6 recipient clocks.

`freeze-availability` reads only clocks, validity masks, TextGrid labels, and
sealed identities. `analyze` is permitted only after that freeze and then
loads mouth/landmark measurements. It reports the CAL-frozen bilabial versus
open-vowel endpoint, eight physical-arm contrasts against RT, the RT/FIXED
bridge, and the declared secondary geometry/AUC summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

EXPECTED = {
    "track_extractor": "84d2cd9470dda9cf39ceeba1e5183fe2ac10b0d4be7c0c384788a3f1d90377d8",
    "grid_inventory": "7b3e879b9a34a6ea20159e770d6530fc77b19fd013381ce4017e87cbbc19203b",
    "plan_raw": "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d",
    "plan_fingerprint": "1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc",
    "producer_all_files": "dfbbc35bea359d12ae7383671534fb6e1e08e7f62532de9faad49c37226ffe8c",
    "producer_protocol": "8c29c8f757c29a88642f69301197560e2356ed641a959bb0bd59fa0f109473ee",
    "producer_rows": "d0f81f8d35fa9761d746c7420259d8605fc9191b1665ab171b0813fce844881f",
    "producer_results": "e494f4dacaeb30520631f061a2b7f0e80ca05da716f7b609bcd7bdba64feaf9d",
    "producer_status": "361b4514f0bab18edcd5e114517cd5b80e3ec76fcb2bd56f873e237e7cda4b7f",
    "runtime_flags": "7455c2594579dca28ddd099aea25b7b9006694e5fab40baa4a064bae301581b1",
    "consumer_source": "8e40c3017ca7c53924302522678c18854c9b0702ce2223821b03d7fd05166a92",
    "consumer_seal": "5f4f6677b5740a010e89120670f751a16b5e929899545b9be2e7856340290350",
    "consumer_inventory": "097736a5a6cfb74882ca2f005001f34223370f6f86d84ff4ad9536de6ce003ce",
    "consumer_audit": "fa06babb3a106bf1ed3acc337a6dc3bddc2c7df35dbd176f6c545c80cf599e30",
    "consumer_protocol": "e2e61f18c221123ccd43433333a405de685e3b0d6e77d3e082b21b11aca7aae5",
    "qa04": "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d",
    "visual_core": "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b",
    "tone_adapter": "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98",
    "video_features": "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4",
    "landmarker": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "calibration": "0a1897a03248322a4868ceae5e194b848212ef33858b4a1cb002f5fe58bd21a4",
    "calibration_freeze": "45cc2ee35980177b126a05b79fb19caa06e43624c24c1c7f85c41575c677b287",
    "phone_mapping_raw": "ddb1f75f1d2a41a5059ad9aea09a408bd094867002d19c0ff4ad7c2c4519feae",
    "phone_mapping_fingerprint": "0c81568e931147f9bf2f5374372d55b97e833b4383a355290581fb2e22f8c9a7",
    "eligibility": "acc6cacfb8f5b35c13163157c6de70f90bc6e34ae54364036935923982a85ba5",
}
PLAN_FINGERPRINT_KEY = "plan_sha256"
LATENCY_S = -0.085
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
VISUAL_IDS = ("V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E")
TREATMENTS = ("V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
EXPECTED_CONTEXTS = 160
EXPECTED_SAMPLES = 80
EXPECTED_SPEAKERS = 40
EXPECTED_VIDEOS = 1600


class VisualConsumerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisualConsumerError(f"expected_object:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise VisualConsumerError(f"expected_json_object:{path.name}:{number}")
            rows.append(item)
    return rows


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def parse_phone_tier(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tier_active = False
    current: dict[str, Any] = {}
    in_interval = False
    saw_tier = False
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        text = line.strip()
        if re.match(r"item\s*\[\d+\]:", text):
            tier_active, in_interval, current = False, False, {}
        if text.startswith('name = "'):
            tier_active = text.split("=", 1)[1].strip().strip('"').casefold() == "phones"
            saw_tier |= tier_active
        if not tier_active:
            continue
        if re.match(r"intervals\s*\[\d+\]:", text):
            in_interval, current = True, {}
        elif in_interval and text.startswith("xmin ="):
            current["start_s"] = float(text.split("=", 1)[1].strip())
        elif in_interval and text.startswith("xmax ="):
            current["end_s"] = float(text.split("=", 1)[1].strip())
        elif in_interval and text.startswith("text ="):
            raw = text.split("=", 1)[1].strip()
            if not (raw.startswith('"') and raw.endswith('"')) or set(current) != {"start_s", "end_s"}:
                raise VisualConsumerError(f"malformed_phone_interval:{path}")
            start, end = float(current["start_s"]), float(current["end_s"])
            label = raw[1:-1].replace('""', '"')
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise VisualConsumerError(f"invalid_phone_interval:{path}")
            rows.append({"phone_index": len(rows), "label": label, "start_s": start, "end_s": end})
            in_interval, current = False, {}
    if not saw_tier or not rows:
        raise VisualConsumerError(f"nonempty_phones_tier_missing:{path}")
    return rows


def load_modules(core_path: Path, tone_path: Path) -> tuple[Any, Any]:
    if sha256_file(core_path) != EXPECTED["visual_core"] or sha256_file(tone_path) != EXPECTED["tone_adapter"]:
        raise VisualConsumerError("visual_core_or_tone_adapter_sha_mismatch")
    spec = importlib.util.spec_from_file_location("tts_mechanism_visual_tone_20261001", tone_path)
    if spec is None or spec.loader is None:
        raise VisualConsumerError("tone_adapter_import_spec_unavailable")
    tone = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tone
    spec.loader.exec_module(tone)
    tone.install_rules()
    core = tone._CORE
    if sha256_file(Path(core.__file__)) != EXPECTED["visual_core"]:
        raise VisualConsumerError("tone_adapter_loaded_unexpected_core")
    return core, tone


def _safe_file_inventory(root: Path) -> set[str]:
    found: set[str] = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            path = base / name
            if stat.S_ISLNK(path.lstat().st_mode):
                raise VisualConsumerError(f"unexpected_symlink:{path}")
        for name in files:
            path = base / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise VisualConsumerError(f"unexpected_nonregular:{path}")
            found.add(path.relative_to(root).as_posix())
    return found


def verify_consumer(consumer_dir: Path, run_dir: Path) -> dict[str, Any]:
    seal_path = consumer_dir / "consumer_output_seal.json"
    inventory_path = consumer_dir / "consumer_all_files_sha256.json"
    audit_path = consumer_dir / "world_consumer_audit.json"
    stats_path = consumer_dir / "world_consumer_statistics.json"
    protocol_path = consumer_dir / "consumer_protocol.json"
    if not all(p.is_file() for p in (seal_path, inventory_path, audit_path, stats_path, protocol_path)):
        raise VisualConsumerError("independent_world_consumer_outputs_missing")
    # Validate the statistics file by its sealed hash only. Availability is
    # frozen without parsing the producer's metric payload.
    seal, inventory, audit = map(read_json, (seal_path, inventory_path, audit_path))
    if sha256_file(seal_path) != "5f4f6677b5740a010e89120670f751a16b5e929899545b9be2e7856340290350":
        raise VisualConsumerError("consumer_seal_sha_mismatch")
    if seal.get("status") != "SEALED" or seal.get("producer_run_root") != str(run_dir.resolve()):
        raise VisualConsumerError("consumer_seal_run_binding_mismatch")
    if seal.get("consumer_source_sha256") != EXPECTED["consumer_source"]:
        raise VisualConsumerError("consumer_source_hash_mismatch")
    if seal.get("producer_plan_raw_file_sha256") != EXPECTED["plan_raw"] or seal.get("producer_plan_embedded_fingerprint_sha256") != EXPECTED["plan_fingerprint"] or seal.get("producer_qa04_sha256") != EXPECTED["qa04"]:
        raise VisualConsumerError("consumer_producer_input_binding_mismatch")
    if sha256_file(inventory_path) != EXPECTED["consumer_inventory"] or seal.get("inventory_sha256") != EXPECTED["consumer_inventory"]:
        raise VisualConsumerError("consumer_inventory_sha_mismatch")
    if sha256_file(audit_path) != EXPECTED["consumer_audit"] or audit.get("status") != "PASS" or audit.get("error_count") != 0:
        raise VisualConsumerError("consumer_audit_not_pass")
    if sha256_file(protocol_path) != EXPECTED["consumer_protocol"]:
        raise VisualConsumerError("consumer_protocol_sha_mismatch")
    expected = inventory.get("files")
    if not isinstance(expected, dict):
        raise VisualConsumerError("consumer_inventory_schema_mismatch")
    actual = _safe_file_inventory(consumer_dir) - {"consumer_all_files_sha256.json", "consumer_output_seal.json"}
    if actual != set(expected) or set(expected) != {"consumer_protocol.json", "world_consumer_audit.json", "world_consumer_statistics.json"}:
        raise VisualConsumerError("consumer_inventory_coverage_mismatch")
    for relative, expected_sha in expected.items():
        if sha256_file(consumer_dir / relative) != expected_sha:
            raise VisualConsumerError(f"consumer_member_sha_mismatch:{relative}")
    return {"seal_sha256": sha256_file(seal_path), "inventory_sha256": sha256_file(inventory_path), "audit_sha256": sha256_file(audit_path), "protocol_sha256": sha256_file(protocol_path), "statistics_sha256": sha256_file(stats_path), "inventory_members": len(expected)}


def verify_producer(run_dir: Path, plan_path: Path, consumer_dir: Path, runtime_flags_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    run_dir, plan_path, consumer_dir = run_dir.resolve(strict=True), plan_path.resolve(strict=True), consumer_dir.resolve(strict=True)
    if sha256_file(plan_path) != EXPECTED["plan_raw"]:
        raise VisualConsumerError("producer_plan_raw_sha_mismatch")
    plan = read_json(plan_path)
    fingerprint = canonical_hash({k: v for k, v in plan.items() if k != PLAN_FINGERPRINT_KEY})
    if plan.get(PLAN_FINGERPRINT_KEY) != EXPECTED["plan_fingerprint"] or fingerprint != EXPECTED["plan_fingerprint"]:
        raise VisualConsumerError("producer_plan_embedded_fingerprint_mismatch")
    file_pins = {
        "all_files_sha256.json": EXPECTED["producer_all_files"],
        "evaluation_protocol.json": EXPECTED["producer_protocol"],
        "evaluation_rows.jsonl": EXPECTED["producer_rows"],
        "evaluation_results.jsonl": EXPECTED["producer_results"],
        "run_status.json": EXPECTED["producer_status"],
    }
    for name, expected_sha in file_pins.items():
        if sha256_file(run_dir / name) != expected_sha:
            raise VisualConsumerError(f"producer_file_sha_mismatch:{name}")
    if sha256_file(runtime_flags_path) != EXPECTED["runtime_flags"]:
        raise VisualConsumerError("producer_runtime_flags_sha_mismatch")
    status = read_json(run_dir / "run_status.json")
    desired = {"status":"complete","completed_contexts":160,"failed_contexts":0,"scored_rows":12160,"failed_rows":0,"unique_matrix_artifacts":5760,"unique_video_artifacts":1600,"execution_plan_sha256":EXPECTED["plan_fingerprint"]}
    if any(status.get(k) != v for k,v in desired.items()):
        raise VisualConsumerError("producer_run_status_not_complete")
    protocol = read_json(run_dir / "evaluation_protocol.json")
    if protocol.get("execution_plan_sha256") != EXPECTED["plan_fingerprint"]:
        raise VisualConsumerError("producer_protocol_plan_binding_mismatch")
    consumer = verify_consumer(consumer_dir, run_dir)
    rows = read_jsonl(run_dir / "evaluation_rows.jsonl")
    results = read_jsonl(run_dir / "evaluation_results.jsonl")
    planned = {r.get("row_id"): r for r in plan.get("rows", [])}
    row_ids = {r.get("row_id") for r in rows}
    result_map = {r.get("row_id"): r for r in results}
    if len(planned) != 12160 or len(row_ids) != 12160 or len(result_map) != 12160 or set(planned) != row_ids or row_ids != set(result_map):
        raise VisualConsumerError("producer_row_cardinality_or_join_mismatch")
    joined: list[dict[str, Any]] = []
    expected_video_refs: dict[tuple[str, str, str], dict[str, Any]] = {}
    context_rows: dict[tuple[str, str], int] = Counter()
    for row_id, planned_row in planned.items():
        result = result_map[row_id]
        if result.get("status") != "scored":
            raise VisualConsumerError(f"unscored_producer_row:{row_id}")
        for key in ("ordered_eval_index", "sample_id", "speaker_id", "split", "direction", "visual_id"):
            if planned_row.get(key) != result.get(key):
                raise VisualConsumerError(f"producer_plan_result_identity_mismatch:{key}:{row_id}")
        context = (str(result["sample_id"]), str(result["direction"]))
        context_rows[context] += 1
        key = (*context, str(result["visual_id"]))
        item = {k: result.get(k) for k in ("sample_id","speaker_id","split","direction","visual_id","row_id","video_path","video_sha256","visual_npz","visual_npz_sha256")}
        if key in expected_video_refs:
            prior = expected_video_refs[key]
            for field in ("video_path","video_sha256","visual_npz","visual_npz_sha256","speaker_id","split"):
                if prior.get(field) != item.get(field):
                    raise VisualConsumerError(f"producer_visual_alias_mismatch:{field}:{key}")
        else:
            expected_video_refs[key] = item
        joined.append(result)
    if len(context_rows) != EXPECTED_CONTEXTS or any(value != 76 for value in context_rows.values()) or len(expected_video_refs) != EXPECTED_VIDEOS:
        raise VisualConsumerError("producer_context_visual_coverage_mismatch")
    if {key[2] for key in expected_video_refs} != set(VISUAL_IDS) or {key[1] for key in expected_video_refs} != set(DIRECTIONS):
        raise VisualConsumerError("producer_visual_ids_or_directions_mismatch")
    speakers_by_sample: dict[str, str] = {}
    for (sample, _direction, _visual), item in expected_video_refs.items():
        speaker = str(item["speaker_id"])
        if sample in speakers_by_sample and speakers_by_sample[sample] != speaker:
            raise VisualConsumerError("sample_speaker_alias_mismatch")
        speakers_by_sample[sample] = speaker
    if len(speakers_by_sample) != EXPECTED_SAMPLES or len(set(speakers_by_sample.values())) != EXPECTED_SPEAKERS or Counter(speakers_by_sample.values()).most_common()[0][1] != 2:
        raise VisualConsumerError("producer_cohort_80x40_mismatch")
    return plan, joined, expected_video_refs, {"run_status_sha256":sha256_file(run_dir/"run_status.json"),"protocol_sha256":sha256_file(run_dir/"evaluation_protocol.json"),"rows_sha256":sha256_file(run_dir/"evaluation_rows.jsonl"),"results_sha256":sha256_file(run_dir/"evaluation_results.jsonl"),"all_files_sha256":sha256_file(run_dir/"all_files_sha256.json"),"runtime_flags_sha256":sha256_file(runtime_flags_path),"plan_raw_sha256":EXPECTED["plan_raw"],"plan_fingerprint_sha256":fingerprint,"consumer":consumer,"run_root":str(run_dir)}


def verify_track_seal(track_dir: Path, expected_videos: Mapping[tuple[str,str,str], dict[str,Any]], extractor_path: Path, output_dir: Path) -> tuple[dict[tuple[str,str,str], dict[str,Any]], dict[str,Any]]:
    track_dir = track_dir.resolve(strict=True)
    if sha256_file(extractor_path) != EXPECTED["track_extractor"]:
        raise VisualConsumerError("track_extractor_source_hash_mismatch")
    protocol_path, status_path = track_dir/"visual_track_protocol.json", track_dir/"visual_track_status.json"
    all_path, seal_path, progress_path = track_dir/"all_files_sha256.json", track_dir/"visual_track_seal.json", track_dir/"progress.jsonl"
    if not all(p.is_file() for p in (protocol_path,status_path,all_path,seal_path,progress_path)):
        raise VisualConsumerError("track_output_seal_files_missing")
    protocol, status, inventory, seal = map(read_json, (protocol_path,status_path,all_path,seal_path))
    if protocol.get("status") != "PLAN_FROZEN" or protocol.get("extractor_source",{}).get("sha256") != EXPECTED["track_extractor"] or protocol.get("worker_process_start_method") != "spawn":
        raise VisualConsumerError("track_protocol_source_binding_mismatch")
    if status.get("status") != "COMPLETE" or status.get("planned_videos") != 1600 or status.get("completed_videos") != 1600 or status.get("failed_videos") != 0 or status.get("errors"):
        raise VisualConsumerError("track_status_not_complete")
    if seal.get("status") != "SEALED" or seal.get("completed_videos") != 1600 or seal.get("failed_videos") != 0:
        raise VisualConsumerError("track_output_seal_not_complete")
    if sha256_file(all_path) != seal.get("all_files_sha256") or seal.get("protocol_sha256") != sha256_file(protocol_path) or seal.get("status_sha256") != sha256_file(status_path):
        raise VisualConsumerError("track_output_seal_sha_mismatch")
    files = inventory.get("files")
    if not isinstance(files, dict) or len(files) != 3202 or len(files) != seal.get("file_count"):
        raise VisualConsumerError("track_file_inventory_schema_or_count_mismatch")
    actual = _safe_file_inventory(track_dir) - {"visual_track_status.json","visual_track_seal.json","all_files_sha256.json"}
    if actual != set(files):
        raise VisualConsumerError("track_file_inventory_coverage_mismatch")
    for relative, expected_sha in files.items():
        if sha256_file(track_dir/relative) != expected_sha:
            raise VisualConsumerError(f"track_member_sha_mismatch:{relative}")
    progress_rows = read_jsonl(progress_path)
    progress_by_key = {}
    for row in progress_rows:
        key = (str(row.get("sample_id")),str(row.get("direction")),str(row.get("visual_id")))
        if key in progress_by_key:
            raise VisualConsumerError(f"duplicate_track_progress_row:{key}")
        progress_by_key[key] = row
    if len(progress_by_key) != EXPECTED_VIDEOS or set(progress_by_key) != set(expected_videos) or any(r.get("status") != "complete" for r in progress_by_key.values()):
        raise VisualConsumerError("track_progress_coverage_mismatch")
    observed = {}
    for key, video_ref in expected_videos.items():
        sample, direction, visual_id = key
        relative = f"tracks/{sample}__{direction}__{visual_id}.npz"
        metadata_relative = f"tracks/{sample}__{direction}__{visual_id}.json"
        feature_path, metadata_path = track_dir/relative, track_dir/metadata_relative
        meta = read_json(metadata_path)
        progress = progress_by_key[key]
        if progress.get("row_id") != video_ref["row_id"] or progress.get("feature_sha256") != files.get(relative) or meta.get("feature_sha256") != files.get(relative):
            raise VisualConsumerError(f"track_to_producer_hash_join_mismatch:{key}")
        if meta.get("video_sha256") != video_ref.get("video_sha256") or meta.get("row_id") != video_ref.get("row_id") or meta.get("visual_id") != visual_id:
            raise VisualConsumerError(f"track_source_video_binding_mismatch:{key}")
        observed[key] = {"row":video_ref,"feature_path":feature_path,"feature_sha256":files[relative],"metadata_path":metadata_path,"metadata":meta,"decoded_frame_count":meta.get("decoded_frame_count"),"feature_metadata_sha256":sha256_file(metadata_path)}
    return observed, {"protocol_sha256":sha256_file(protocol_path),"status_sha256":sha256_file(status_path),"all_files_sha256":sha256_file(all_path),"seal_sha256":sha256_file(seal_path),"progress_sha256":sha256_file(progress_path),"inventory_count":len(files),"completed_videos":len(progress_rows)}


def verify_grid_inventory(path: Path, expected_contexts: set[tuple[str,str]]) -> dict[tuple[str,str], dict[str,Any]]:
    if sha256_file(path) != EXPECTED["grid_inventory"]:
        raise VisualConsumerError("grid_inventory_sha_mismatch")
    inventory = read_json(path)
    if inventory.get("status") != "PASS" or len(inventory.get("effective_recipient_grids", [])) != EXPECTED_CONTEXTS:
        raise VisualConsumerError("grid_inventory_not_pass_or_incomplete")
    result = {}
    for ref in inventory["effective_recipient_grids"]:
        key = (str(ref.get("sample_id")),str(ref.get("direction")))
        expected_condition = "natural" if key[1] == "tts_to_natural" else "tts"
        if key in result or ref.get("role") != "effective_recipient_visual_clock" or ref.get("timeline_policy") != "recipient clock" or ref.get("condition") != expected_condition:
            raise VisualConsumerError(f"invalid_effective_grid_ref:{key}")
        path_value = Path(str(ref.get("path", "")))
        if not path_value.is_file() or sha256_file(path_value) != ref.get("sha256"):
            raise VisualConsumerError(f"effective_grid_file_sha_mismatch:{key}")
        overlay = bool(ref.get("overlay_used"))
        if overlay and key[0] != "S0064_BAC009S0064W0121":
            raise VisualConsumerError("unexpected_overlay_sample")
        all_refs = [x for x in inventory.get("all_grid_refs", []) if x.get("sample_id") == key[0] and x.get("condition") == expected_condition]
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_ref in all_refs:
            by_role[str(source_ref.get("role"))].append(source_ref)
            source_path = Path(str(source_ref.get("path", "")))
            source_sha = str(source_ref.get("sha256", ""))
            if source_ref.get("exists") is not True or source_ref.get("actual_sha256") != source_sha or not source_path.is_file() or sha256_file(source_path) != source_sha:
                raise VisualConsumerError(f"grid_source_ref_hash_mismatch:{key}:{source_ref.get('role')}")
        if len(all_refs) != 3 or any(len(by_role[role]) != 1 for role in ("original", "producer_selected", "effective_recipient_visual_clock")):
            raise VisualConsumerError(f"grid_source_ref_role_coverage_mismatch:{key}")
        original = by_role["original"][0]
        selected = by_role["producer_selected"][0]
        effective_ref = by_role["effective_recipient_visual_clock"][0]
        if (selected.get("path"), selected.get("sha256")) != (ref.get("path"), ref.get("sha256")):
            raise VisualConsumerError(f"selected_grid_differs_from_effective:{key}")
        if overlay != (selected.get("path") != original.get("path") or selected.get("sha256") != original.get("sha256")):
            raise VisualConsumerError(f"overlay_flag_source_mismatch:{key}")
        if (effective_ref.get("path"), effective_ref.get("sha256")) != (ref.get("path"), ref.get("sha256")):
            raise VisualConsumerError(f"effective_grid_inventory_ref_mismatch:{key}")
        if not all_refs:
            raise VisualConsumerError(f"grid_inventory_missing_original_ref:{key}")
        result[key] = {"effective":ref,"all_refs":all_refs,"original_refs":[original],"producer_selected_refs":[selected],"overlay_refs":[selected] if overlay else [],"overlay_used":overlay}
    if set(result) != expected_contexts:
        raise VisualConsumerError("effective_grid_context_set_mismatch")
    return result


def load_clock(feature_path: Path, feature_sha: str, sidecar: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    if sha256_file(feature_path) != feature_sha:
        raise VisualConsumerError("feature_file_hash_changed")
    with np.load(feature_path, allow_pickle=False) as archive:
        required = {"timestamps_s","valid","metadata_json","mouth_open","mouth_width","landmarks"}
        if not required.issubset(set(archive.files)):
            raise VisualConsumerError("visual_npz_required_array_missing")
        timestamps = np.asarray(archive["timestamps_s"])
        valid = np.asarray(archive["valid"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    arrays = sidecar.get("arrays",{})
    for name,value,expected_dtype in (("timestamps_s",timestamps,"float64"),("valid",valid,"bool")):
        item=arrays.get(name,{})
        if list(value.shape)!=item.get("shape") or str(value.dtype)!=item.get("dtype") or str(value.dtype)!=expected_dtype or array_sha(value)!=item.get("sha256"):
            raise VisualConsumerError(f"availability_array_binding_mismatch:{name}")
    if timestamps.ndim!=1 or valid.dtype!=np.bool_ or valid.shape!=timestamps.shape or not np.isfinite(timestamps).all() or (len(timestamps)>1 and np.any(np.diff(timestamps)<=0)):
        raise VisualConsumerError("malformed_visual_clock_or_valid_mask")
    if len(timestamps)!=int(sidecar.get("decoded_frame_count",-1)) or len(timestamps)!=int(metadata.get("decoded_frame_count",-1)):
        raise VisualConsumerError("visual_clock_decoded_count_mismatch")
    return timestamps,valid,metadata


def common_clock(track_availability: Mapping[str,tuple[np.ndarray,np.ndarray,dict[str,Any]]]) -> tuple[np.ndarray,dict[str,np.ndarray],dict[str,Any]]:
    base_name="V_RT"
    if base_name not in track_availability:
        raise VisualConsumerError("RT_baseline_clock_missing")
    base_times=track_availability[base_name][0]
    base_valid=track_availability[base_name][1]
    tolerance=0.5*float(np.median(np.diff(base_times)))+1e-12 if len(base_times)>1 else 0.5/float(track_availability[base_name][2].get("fps",25.0))
    maps={}
    common=np.ones(len(base_times),dtype=bool)
    matched_frames={}
    for name,(times,valid,_meta) in track_availability.items():
        index_map=np.full(len(base_times),-1,dtype=np.int64)
        i=j=0
        while i<len(base_times) and j<len(times):
            delta=float(times[j]-base_times[i])
            if abs(delta)<=tolerance:
                index_map[i]=j; i+=1; j+=1
            elif delta < -tolerance:
                j+=1
            else:
                i+=1
        maps[name]=index_map
        matched=index_map>=0
        current=np.zeros(len(base_times),dtype=bool)
        current[matched]=valid[index_map[matched]]
        common &= current
        matched_frames[name]=int(matched.sum())
    support={"recipient_clock_visual_id":"V_RT","clock_frames":len(base_times),"valid_common_frames":int(common.sum()),"valid_common_mask_sha256":array_sha(common),"common_coverage":float(common.mean()) if len(common) else 0.0,"timestamp_tolerance_s":tolerance,"matched_frames_by_visual_id":matched_frames,"timeline_policy":"full recipient clock; monotonic timestamp match; all ten visual tracks and valid masks; no time warp or fixed-length crop"}
    return common,maps,support


def phone_counts(phones: Sequence[Mapping[str,Any]], query_times: np.ndarray, tone: Any, latency: float, frame_minimum: int) -> tuple[dict[str,int],dict[str,int],dict[str,list[dict[str,Any]]]]:
    classes=("bilabial","open_vowel")
    expected={k:0 for k in classes}; supported={k:0 for k in classes}; tokens={k:[] for k in classes}
    for phone in phones:
        category=tone.phone_class(phone["label"])
        if category is None: continue
        expected[category]+=1
        start=float(phone["start_s"])+latency; end=float(phone["end_s"])+latency
        n=int(np.count_nonzero((query_times>=start)&(query_times<end)))
        row={"phone_index":int(phone["phone_index"]),"raw_label":str(phone["label"]),"visual_class":category,"start_s":float(phone["start_s"]),"end_s":float(phone["end_s"]),"common_frames_at_latency":n}
        if n>=frame_minimum:
            supported[category]+=1; tokens[category].append(row)
    return expected,supported,tokens


def preflight_inputs(args: argparse.Namespace, core: Any, tone: Any) -> tuple[dict[str,Any],list[dict[str,Any]],dict[tuple[str,str,str],dict[str,Any]],dict[tuple[str,str],dict[str,Any]],dict[str,Any],dict[str,Any],dict[str,Any]]:
    run_dir=args.run_dir.resolve(strict=True)
    plan,joined,videos,producer=verify_producer(run_dir,args.execution_plan,args.consumer_output,args.runtime_flags)
    if sha256_file(args.grid_inventory)!=EXPECTED["grid_inventory"]:
        raise VisualConsumerError("grid_inventory_sha_mismatch")
    eligibility=read_json(args.eligibility_manifest)
    if sha256_file(args.eligibility_manifest)!=EXPECTED["eligibility"] or eligibility.get("records_total")!=80 or eligibility.get("records_pair_qualified")!=79 or eligibility.get("records_pair_unobserved")!=1:
        raise VisualConsumerError("original_mfa_eligibility_manifest_mismatch")
    eligibility_rows={str(r["id"]):r for r in eligibility.get("records",[])}
    if len(eligibility_rows)!=80 or sum(bool(r.get("pair_qualified")) for r in eligibility_rows.values())!=79:
        raise VisualConsumerError("original_mfa_eligibility_row_count_mismatch")
    cal=read_json(args.calibration)
    freeze=read_json(args.calibration_freeze)
    mapping=read_json(args.phone_mapping)
    if sha256_file(args.calibration)!=EXPECTED["calibration"] or sha256_file(args.calibration_freeze)!=EXPECTED["calibration_freeze"] or sha256_file(args.phone_mapping)!=EXPECTED["phone_mapping_raw"]:
        raise VisualConsumerError("calibration_input_raw_sha_mismatch")
    if canonical_hash({k:v for k,v in mapping.items() if k!="mapping_sha256"})!=EXPECTED["phone_mapping_fingerprint"] or mapping.get("mapping_sha256")!=EXPECTED["phone_mapping_fingerprint"]:
        raise VisualConsumerError("tone_mapping_embedded_fingerprint_mismatch")
    if cal.get("status")!="CALIBRATION_COMPLETE" or cal.get("protocol_id")!="tts_mechanism_visual_20261001_mandarin_tone_v2" or freeze.get("status")!="FROZEN_CAL_ONLY" or freeze.get("evaluation_textgrids_read") is not False or freeze.get("evaluation_effects_read") is not False:
        raise VisualConsumerError("calibration_freeze_not_cal_only")
    minimums={k:int(v) for k,v in cal.get("protocol",{}).get("min_phone_tokens_per_clip_by_class",{}).items()}
    if set(minimums)!={"bilabial","open_vowel"} or min(minimums.values())<1:
        raise VisualConsumerError("calibration_visual_class_minimums_missing")
    latency=float(cal.get("protocol",{}).get("latency",{}).get("latency_s"))
    frame_min=int(cal.get("protocol",{}).get("phone_min_common_frames",1))
    if not math.isclose(latency,LATENCY_S,abs_tol=1e-12) or frame_min<1:
        raise VisualConsumerError("calibration_frozen_latency_or_frame_minimum_mismatch")
    for label,path,expected in (("core",args.core,EXPECTED["visual_core"]),("tone",args.tone_adapter,EXPECTED["tone_adapter"]),("features",args.video_features,EXPECTED["video_features"]),("extractor",args.extractor,EXPECTED["track_extractor"])):
        if sha256_file(path)!=expected: raise VisualConsumerError(f"source_hash_mismatch:{label}")
    sample_ids={str(r.get("sample_id")) for r in joined}
    contexts={(sid,direction) for sid in sample_ids for direction in DIRECTIONS}
    if len(sample_ids)!=80 or len(contexts)!=160 or set(eligibility_rows)!=sample_ids:
        raise VisualConsumerError("cohort_or_eligibility_set_mismatch")
    grid_map=verify_grid_inventory(args.grid_inventory,contexts)
    track_rows,track_seals=verify_track_seal(args.track_dir,videos,args.extractor,args.output_dir)
    return plan,joined,videos,grid_map,{"producer":producer,"track":track_seals,"track_rows":track_rows,"eligibility":eligibility,"eligibility_rows":eligibility_rows,"calibration":cal,"calibration_freeze":freeze,"mapping":mapping,"minimums":minimums,"latency":latency,"frame_minimum":frame_min,"samples":sorted(sample_ids),"speakers":sorted(set(str(r["speaker_id"]) for r in joined))},producer,track_seals


def freeze_availability(args: argparse.Namespace) -> dict[str,Any]:
    core,tone=load_modules(args.core,args.tone_adapter)
    plan,joined,videos,grid_map,meta,producer,track_seals=preflight_inputs(args,core,tone)
    track_rows=meta["track_rows"]; eligibility_rows=meta["eligibility_rows"]
    rows=[]
    for sample in meta["samples"]:
        speaker=str(eligibility_rows[sample]["speaker"])
        for direction in DIRECTIONS:
            context=(sample,direction)
            grid_info=grid_map[context]["effective"]
            phones=parse_phone_tier(Path(grid_info["path"]))
            availability={}
            for visual_id in VISUAL_IDS:
                ref=track_rows[(sample,direction,visual_id)]
                clock,valid,feature_metadata=load_clock(ref["feature_path"],ref["feature_sha256"],ref["metadata"])
                if feature_metadata.get("sample_id")!=sample or feature_metadata.get("direction")!=direction or feature_metadata.get("visual_id")!=visual_id or feature_metadata.get("video_sha256")!=ref["row"].get("video_sha256"):
                    raise VisualConsumerError(f"feature_embedded_identity_mismatch:{context}:{visual_id}")
                if feature_metadata.get("recipient_grid_sha256")!=grid_info["sha256"] or feature_metadata.get("timeline_policy")!="recipient clock; no donor grid used":
                    raise VisualConsumerError(f"feature_recipient_grid_binding_mismatch:{context}:{visual_id}")
                availability[visual_id]=(clock,valid,feature_metadata)
            common,maps,support=common_clock(availability)
            times=availability["V_RT"][0][common]
            expected,supported,tokens=phone_counts(phones,times,tone,meta["latency"],meta["frame_minimum"])
            both=all(supported[k]>=meta["minimums"][k] for k in meta["minimums"])
            original=bool(eligibility_rows[sample].get("pair_qualified"))
            rows.append({
                "sample_id":sample,"speaker_id":speaker,"direction":direction,"recipient_condition":"natural" if direction=="tts_to_natural" else "tts",
                "original_mfa_pair_qualified":original,"original_mfa_eligibility_record_sha256":sha256_file(args.eligibility_manifest),
                "effective_grid":{"path":grid_info["path"],"sha256":grid_info["sha256"],"overlay_used":grid_map[context]["overlay_used"],"original_grid_refs":grid_map[context]["original_refs"],"overlay_grid_refs":grid_map[context]["overlay_refs"]},
                "phone_intervals":phones,"phone_class_counts":{"expected":expected,"supported_on_common_clock":supported,"minimum_tokens_by_class":meta["minimums"],"minimum_common_frames_per_phone":meta["frame_minimum"],"supported_tokens_at_minus_85ms":tokens},
                "common_clock":support,"feature_refs":{visual_id:{"row_id":track_rows[(sample,direction,visual_id)]["row"]["row_id"],"video_path":track_rows[(sample,direction,visual_id)]["row"]["video_path"],"video_sha256":track_rows[(sample,direction,visual_id)]["row"]["video_sha256"],"feature_path":track_rows[(sample,direction,visual_id)]["feature_path"].relative_to(args.track_dir.resolve()).as_posix(),"feature_sha256":track_rows[(sample,direction,visual_id)]["feature_sha256"],"feature_metadata_sha256":track_rows[(sample,direction,visual_id)]["feature_metadata_sha256"],"frame_count":len(availability[visual_id][0]),"valid_frames":int(availability[visual_id][1].sum()),"timestamp_sha256":array_sha(availability[visual_id][0]),"valid_mask_sha256":array_sha(availability[visual_id][1])} for visual_id in VISUAL_IDS},
                "availability_only_endpoint":{"both_classes_available":bool(both),"primary_visual_endpoint_eligible":bool(original and both and common.any()),"original_unqualified_not_restored_by_overlay":bool(grid_map[context]["overlay_used"] and not original)},
            })
    if len(rows)!=160 or len({r["sample_id"] for r in rows})!=80 or len({r["speaker_id"] for r in rows})!=40:
        raise VisualConsumerError("availability_rows_cohort_mismatch")
    speaker_rows=[]
    for speaker in meta["speakers"]:
        speaker_contexts=[r for r in rows if r["speaker_id"]==speaker]
        for direction in DIRECTIONS:
            clips=[r for r in speaker_contexts if r["direction"]==direction]
            speaker_rows.append({"speaker_id":speaker,"direction":direction,"clip_ids":sorted(r["sample_id"] for r in clips),"complete_two_clip_primary_availability":bool(len(clips)==2 and all(r["availability_only_endpoint"]["primary_visual_endpoint_eligible"] for r in clips))})
    eligible_by_direction={d:sum(bool(r["availability_only_endpoint"]["primary_visual_endpoint_eligible"]) for r in rows if r["direction"]==d) for d in DIRECTIONS}
    complete_speakers_by_direction={d:sum(bool(r["complete_two_clip_primary_availability"]) for r in speaker_rows if r["direction"]==d) for d in DIRECTIONS}
    out=args.output_dir.resolve()
    freeze_path=out/"world_visual_availability_freeze.json"
    freeze={"schema_version":1,"status":"FROZEN_BEFORE_GEOMETRY_READ","protocol_id":"world_visual_20261001_two_class_recipient_clock_v1","created_utc":datetime.now(timezone.utc).isoformat(),"selection_uses_mouth_aperture_or_landmark_values":False,"selection_uses_only_sealed_identity_grid_labels_timestamps_and_valid_masks":True,
        "intended_denominator":{"samples":80,"speakers":40,"clips_per_speaker":2,"directions":2,"contexts":160,"videos":1600},
        "frozen_visual_rule":{"class_labels":["bilabial","open_vowel"],"calibration_protocol_id":meta["calibration"].get("protocol_id"),"latency_s":meta["latency"],"minimum_phone_tokens_per_class":meta["minimums"],"minimum_common_frames_per_phone":meta["frame_minimum"],"common_clock":"recipient V_RT timestamps; require timestamp match and valid mask for all ten video IDs; no warp, crop, or donor clock","grid_rule":"effective recipient TextGrid phones tier; S0064 overlay is recorded, while original pair_qualified remains false and inference eligibility is not restored","tone_rule":"visual classifier removes only five frozen tone bars before applying the old visual normalizer; the old core also strips its preexisting marks","primary_outcome":"mean open-vowel aperture minus mean bilabial aperture at -0.085 s","secondary_outcomes":["zero-latency phone separation","per-clip phone-token Mann-Whitney AUC with ties=.5","overall aperture","absolute aperture velocity","mouth width and height/area","yaw and pitch proxies"],"contrasts":[* [f"{arm}_minus_V_RT" for arm in TREATMENTS],"V_RT_minus_V_FIXED"]},
        "input_seals":{"plan_raw_sha256":EXPECTED["plan_raw"],"plan_fingerprint_sha256":EXPECTED["plan_fingerprint"],"producer_all_files_sha256":producer["all_files_sha256"],"producer_protocol_sha256":producer["protocol_sha256"],"producer_rows_sha256":producer["rows_sha256"],"producer_results_sha256":producer["results_sha256"],"producer_status_sha256":producer["run_status_sha256"],"producer_runtime_flags_sha256":producer["runtime_flags_sha256"],"producer_consumer":producer["consumer"],"track_seals":track_seals,"grid_inventory_sha256":EXPECTED["grid_inventory"],"eligibility_manifest_sha256":EXPECTED["eligibility"],"calibration_sha256":EXPECTED["calibration"],"calibration_freeze_sha256":EXPECTED["calibration_freeze"],"phone_mapping_sha256":EXPECTED["phone_mapping_raw"],"phone_mapping_fingerprint":EXPECTED["phone_mapping_fingerprint"],"visual_core_sha256":EXPECTED["visual_core"],"tone_adapter_sha256":EXPECTED["tone_adapter"],"track_extractor_sha256":EXPECTED["track_extractor"],"video_features_sha256":EXPECTED["video_features"],"landmarker_sha256":EXPECTED["landmarker"]},
        "coverage":{"intended_samples":80,"intended_speakers":40,"intended_contexts":160,"intended_videos":1600,"original_mfa_pair_qualified_samples":sum(bool(r.get("pair_qualified")) for r in meta["eligibility_rows"].values()),"original_mfa_pair_unqualified_samples":sum(not bool(r.get("pair_qualified")) for r in meta["eligibility_rows"].values()),"primary_available_contexts_by_direction":eligible_by_direction,"complete_two_clip_speakers_by_direction":complete_speakers_by_direction,"effective_overlay_contexts":sum(bool(r["effective_grid"]["overlay_used"]) for r in rows),"overlay_contexts_not_eligible_due_original_mfa":sum(bool(r["effective_grid"]["overlay_used"] and not r["original_mfa_pair_qualified"]) for r in rows)},"context_rows":rows,"speaker_rows":speaker_rows}
    atomic_json(freeze_path,freeze)
    freeze_sha=sha256_file(freeze_path)
    freeze_seal={"status":"FROZEN_BEFORE_GEOMETRY_READ","freeze_path":str(freeze_path),"freeze_sha256":freeze_sha,"consumer_source_sha256":sha256_file(Path(__file__).resolve()),"selection_uses_mouth_aperture_or_landmarks":False}
    seal_path=out/"world_visual_availability_freeze.seal.json"
    atomic_json(seal_path,freeze_seal)
    return {"status":freeze["status"],"freeze_path":str(freeze_path),"freeze_sha256":freeze_sha,"seal_sha256":sha256_file(seal_path),"contexts":160,"videos":1600,"eligible_contexts_by_direction":eligible_by_direction,"complete_speakers_by_direction":complete_speakers_by_direction}


def load_full_track(ref: Mapping[str,Any], core: Any) -> Any:
    path=Path(ref["feature_path"]); sidecar=read_json(ref["metadata_path"])
    if sha256_file(path)!=ref["feature_sha256"]:
        raise VisualConsumerError("feature_changed_after_availability_freeze")
    with np.load(path,allow_pickle=False) as archive:
        required={"timestamps_s","valid","mouth_open","mouth_width","mouth_height","mouth_area","landmarks","canonical_mouth","metadata_json"}
        if not required.issubset(set(archive.files)):
            raise VisualConsumerError("analysis_feature_array_missing")
        arrays={name:np.asarray(archive[name]) for name in required if name!="metadata_json"}
        metadata=json.loads(str(archive["metadata_json"].item()))
    if metadata.get("video_sha256")!=ref["video_sha256"] or metadata.get("recipient_grid_sha256")!=ref["grid_sha256"]:
        raise VisualConsumerError("analysis_feature_embedded_source_mismatch")
    for name,value in arrays.items():
        item=sidecar.get("arrays",{}).get(name,{})
        if item.get("shape")!=list(value.shape) or item.get("dtype")!=str(value.dtype) or item.get("sha256")!=array_sha(value):
            raise VisualConsumerError(f"analysis_feature_array_sha_mismatch:{name}")
    t=np.asarray(arrays["timestamps_s"],dtype=np.float64); v=np.asarray(arrays["valid"],dtype=bool)
    aperture=np.asarray(arrays["mouth_open"],dtype=np.float64); width=np.asarray(arrays["mouth_width"],dtype=np.float64); landmarks=np.asarray(arrays["landmarks"],dtype=np.float64)
    track=core.MouthTrack(t,aperture,v,width,landmarks,metadata)
    return track,{"mouth_height":np.asarray(arrays["mouth_height"],dtype=np.float64),"mouth_area":np.asarray(arrays["mouth_area"],dtype=np.float64),"canonical_mouth_shape":list(arrays["canonical_mouth"].shape),"feature_arrays_sha256":{name:array_sha(value) for name,value in arrays.items()}}


def track_phone_summary(phones: Sequence[Mapping[str,Any]], track: Any, indices: np.ndarray, core: Any, tone: Any, latency: float, minimums: Mapping[str,int]) -> dict[str,Any]:
    tokens,_expected=core._phone_apertures(phones,track,indices,latency)
    values={"bilabial":[],"open_vowel":[]}
    for row in tokens:
        if row.get("observed"): values[row["category"]].append(float(row["aperture_mean"]))
    means={k:(float(np.mean(values[k],dtype=np.float64)) if len(values[k])>=int(minimums[k]) else None) for k in values}
    separation=float(means["open_vowel"]-means["bilabial"]) if None not in means.values() else None
    auc=tone._mann_whitney_auc(values["open_vowel"],values["bilabial"]) if all(len(values[k])>=int(minimums[k]) for k in values) else None
    return {"class_aperture_means":means,"phone_separation":separation,"phone_token_auc":float(auc) if auc is not None else None,"observed_tokens_by_class":{k:len(values[k]) for k in values},"phone_tokens":tokens}


def speaker_summary(clip_values: Mapping[str,float|None], speaker_by_sample: Mapping[str,str], intended_samples: Sequence[str], intended_speakers: Sequence[str], seed: int=BOOTSTRAP_SEED) -> dict[str,Any]:
    grouped:dict[str,list[float]]=defaultdict(list)
    for sample in intended_samples:
        value=clip_values.get(sample)
        if value is not None and math.isfinite(float(value)): grouped[speaker_by_sample[sample]].append(float(value))
    complete={speaker:float(np.mean(values,dtype=np.float64)) for speaker,values in grouped.items() if len(values)==2}
    names=sorted(complete)
    data=np.asarray([complete[s] for s in names],dtype=np.float64)
    if len(data):
        rng=np.random.Generator(np.random.PCG64(seed))
        indices=rng.integers(0,len(data),size=(BOOTSTRAP_DRAWS,len(data)),endpoint=False)
        means=data[indices].mean(axis=1)
        interval=[float(np.quantile(means,.005,method="linear")),float(np.quantile(means,.995,method="linear"))]
        index_sha=array_sha(indices.astype(np.int32))
        estimate=float(data.mean())
    else:
        interval=None;index_sha=None;estimate=None
    return {"equal_speaker_mean":estimate,"ci99":interval,"complete_two_clip_speakers":len(names),"intended_speakers":len(intended_speakers),"complete_clips":sum(len(v) for v in grouped.values()),"intended_clips":len(intended_samples),"missing_speakers":sorted(set(intended_speakers)-set(names)),"speaker_values":dict(sorted(complete.items())),"bootstrap_draws":BOOTSTRAP_DRAWS,"bootstrap_seed":seed,"bootstrap_index_sha256":index_sha,"unit":"speaker; mean of two clip-level outcomes before equal-speaker averaging"}


def analyze(args: argparse.Namespace) -> dict[str,Any]:
    core,tone=load_modules(args.core,args.tone_adapter)
    freeze_path=args.availability_freeze.resolve(strict=True)
    freeze=read_json(freeze_path)
    seal_path=freeze_path.with_name("world_visual_availability_freeze.seal.json")
    freeze_seal=read_json(seal_path)
    if freeze.get("status")!="FROZEN_BEFORE_GEOMETRY_READ" or freeze.get("selection_uses_mouth_aperture_or_landmark_values") is not False or freeze_seal.get("freeze_sha256")!=sha256_file(freeze_path) or freeze_seal.get("consumer_source_sha256")!=sha256_file(Path(__file__).resolve()):
        raise VisualConsumerError("availability_freeze_not_sealed_to_current_consumer")
    if sha256_file(args.track_dir/"visual_track_seal.json")!=freeze["input_seals"]["track_seals"]["seal_sha256"]:
        raise VisualConsumerError("track_seal_changed_after_availability_freeze")
    context_rows=freeze["context_rows"]
    analysis_rows=[]
    for frozen in context_rows:
        sample,direction=str(frozen["sample_id"]),str(frozen["direction"])
        context=(sample,direction)
        phones=frozen["phone_intervals"]
        feature_refs=frozen["feature_refs"]
        loaded={}; extras={}
        for visual_id in VISUAL_IDS:
            ref=dict(feature_refs[visual_id])
            ref["feature_path"]=args.track_dir/ref["feature_path"]
            ref["metadata_path"]=args.track_dir/"tracks"/(f"{sample}__{direction}__{visual_id}.json")
            ref["grid_sha256"]=frozen["effective_grid"]["sha256"]
            track,extra=load_full_track(ref,core)
            loaded[visual_id]=track;extras[visual_id]=extra
        core_tracks={"baseline":loaded["V_RT"],**{key:loaded[key] for key in VISUAL_IDS if key!="V_RT"}}
        common,maps,support=core._common_clock(core_tracks)
        if int(common.sum())!=int(frozen["common_clock"]["valid_common_frames"]) or array_sha(common)!=frozen["common_clock"]["valid_common_mask_sha256"] or len(common)!=int(frozen["common_clock"]["clock_frames"]):
            raise VisualConsumerError(f"common_clock_changed_after_freeze:{context}")
        common_base=np.flatnonzero(common)
        arm_metrics={}
        global_metrics={}
        for visual_id in VISUAL_IDS:
            map_key="baseline" if visual_id=="V_RT" else visual_id
            track=loaded[visual_id]
            indices=maps[map_key][common_base]
            arm_metrics[visual_id]={"minus_85ms":track_phone_summary(phones,track,indices,core,tone,LATENCY_S,freeze["frozen_visual_rule"]["minimum_phone_tokens_per_class"]),"zero_latency":track_phone_summary(phones,track,indices,core,tone,0.0,freeze["frozen_visual_rule"]["minimum_phone_tokens_per_class"])}
            base_metric=core._global_track_metrics(track,maps[map_key],common,loaded["V_RT"].timestamps_s)
            height=extras[visual_id]["mouth_height"][indices]
            area=extras[visual_id]["mouth_area"][indices]
            base_metric["mouth_height"]={"mean":float(height.mean()) if len(height) else None,"median":float(np.median(height)) if len(height) else None}
            base_metric["mouth_area"]={"mean":float(area.mean()) if len(area) else None,"median":float(np.median(area)) if len(area) else None}
            global_metrics[visual_id]=base_metric
        qualified=bool(frozen["original_mfa_pair_qualified"])
        eligible=bool(frozen["availability_only_endpoint"]["primary_visual_endpoint_eligible"])
        contrasts={}
        arm_contrasts={**{f"{arm}_minus_V_RT":(arm,"V_RT") for arm in TREATMENTS},"V_RT_minus_V_FIXED":("V_RT","V_FIXED")}
        for contrast,(left,right) in arm_contrasts.items():
            contrasts[contrast]={"minus_85ms":{},"zero_latency":{},"geometry":{}}
            for latency_key in ("minus_85ms","zero_latency"):
                left_metric,right_metric=arm_metrics[left][latency_key],arm_metrics[right][latency_key]
                for metric in ("phone_separation","phone_token_auc"):
                    lv,rv=left_metric.get(metric),right_metric.get(metric)
                    contrasts[contrast][latency_key][metric]=(float(lv)-float(rv)) if lv is not None and rv is not None else None
            for metric in ("aperture","mouth_width","absolute_aperture_velocity_per_s","yaw_proxy","pitch_proxy","mouth_height","mouth_area"):
                lv=global_metrics[left].get(metric,{}).get("mean");rv=global_metrics[right].get(metric,{}).get("mean")
                contrasts[contrast]["geometry"][metric]=(float(lv)-float(rv)) if lv is not None and rv is not None else None
        row={"sample_id":sample,"speaker_id":frozen["speaker_id"],"direction":direction,"recipient_condition":frozen["recipient_condition"],"original_mfa_pair_qualified":qualified,"effective_grid":frozen["effective_grid"],"pre_effect_availability":frozen["availability_only_endpoint"],"common_clock":support,"track_refs":feature_refs,"per_arm_phone_metrics":arm_metrics,"per_arm_geometry_metrics":global_metrics,"contrasts":contrasts}
        analysis_rows.append(row)
    samples=sorted({r["sample_id"] for r in context_rows})
    speakers=sorted({r["speaker_id"] for r in context_rows})
    summaries={}
    for direction in DIRECTIONS:
        dir_rows=[r for r in analysis_rows if r["direction"]==direction]
        by_sample={r["sample_id"]:r for r in dir_rows}
        speaker_by_sample={r["sample_id"]:r["speaker_id"] for r in dir_rows}
        outcomes={}
        for contrast in [* [f"{arm}_minus_V_RT" for arm in TREATMENTS],"V_RT_minus_V_FIXED"]:
            outcomes[contrast]={}
            for metric in ("phone_separation","phone_token_auc"):
                for latency_key in ("minus_85ms","zero_latency"):
                    values={}
                    for sample,row in by_sample.items():
                        is_phone_endpoint=bool(row["original_mfa_pair_qualified"] and row["pre_effect_availability"]["primary_visual_endpoint_eligible"])
                        values[sample]=row["contrasts"][contrast][latency_key][metric] if is_phone_endpoint else None
                    outcomes[contrast][f"{latency_key}_{metric}"]=speaker_summary(values,speaker_by_sample,samples,speakers)
            for metric in ("aperture","mouth_width","absolute_aperture_velocity_per_s","yaw_proxy","pitch_proxy","mouth_height","mouth_area"):
                values={sample:row["contrasts"][contrast]["geometry"][metric] for sample,row in by_sample.items()}
                outcomes[contrast][f"geometry_{metric}"]=speaker_summary(values,speaker_by_sample,samples,speakers)
        arm_levels={}
        for visual_id in VISUAL_IDS:
            arm_levels[visual_id]={}
            for metric in ("phone_separation","phone_token_auc"):
                for latency_key in ("minus_85ms","zero_latency"):
                    values={}
                    for sample,row in by_sample.items():
                        okay=bool(row["original_mfa_pair_qualified"] and row["pre_effect_availability"]["primary_visual_endpoint_eligible"])
                        values[sample]=row["per_arm_phone_metrics"][visual_id][latency_key].get(metric) if okay else None
                    arm_levels[visual_id][f"{latency_key}_{metric}"]=speaker_summary(values,speaker_by_sample,samples,speakers)
            for metric in ("aperture","mouth_width","absolute_aperture_velocity_per_s","yaw_proxy","pitch_proxy","mouth_height","mouth_area"):
                values={sample:row["per_arm_geometry_metrics"][visual_id][metric]["mean"] for sample,row in by_sample.items()}
                arm_levels[visual_id][f"geometry_{metric}"]=speaker_summary(values,speaker_by_sample,samples,speakers)
        summaries[direction]={"recipient_condition":"natural" if direction=="tts_to_natural" else "tts","intended_clips":80,"intended_speakers":40,"primary_contrasts":outcomes,"arm_levels":arm_levels,"original_mfa_pair_qualified_clips":sum(bool(r["original_mfa_pair_qualified"]) for r in dir_rows),"primary_available_clips":sum(bool(r["pre_effect_availability"]["primary_visual_endpoint_eligible"]) for r in dir_rows),"complete_two_clip_speakers_primary_available":sum(len([r for r in dir_rows if r["speaker_id"]==speaker and r["pre_effect_availability"]["primary_visual_endpoint_eligible"] and r["original_mfa_pair_qualified"]])==2 for speaker in speakers)}
    output=args.output_dir.resolve()
    analysis_path=output/"world_visual_analysis.json"
    result={"schema_version":1,"status":"COMPLETE","analysis_type":"world_physical_two_class_visual_endpoint_and_geometry","created_utc":datetime.now(timezone.utc).isoformat(),"interpretation_scope":"visual aperture/phone association and geometric description; no speech-quality or perception inference","method":{"primary":"per-context mean token open-vowel aperture minus bilabial aperture at CAL-frozen -0.085 s; arm contrasts vs V_RT and V_RT−V_FIXED bridge","directions":"tts_to_natural (natural recipient) and natural_to_tts (TTS recipient) reported separately; no donor phone clock","classification":"CAL-frozen bilabial/open_vowel visual classes after the tone adapter's five-codepoint removal and old normalizer","latency_s":LATENCY_S,"support":"all ten visual IDs on full V_RT recipient clock with common timestamp-match and valid masks","inference_denominator":"80 clips/40 speakers per direction retained; clip endpoint requires original MFA pair qualification plus CAL class token minima; incomplete speakers reported missing","overlay":"S0064 effective phone grid is the frozen overlay, but the original unqualified record remains in the intended denominator and is not restored for endpoint inference","bootstrap":{"draws":BOOTSTRAP_DRAWS,"seed":BOOTSTRAP_SEED,"ci_quantiles":[0.005,0.995],"unit":"equal speaker; two clips averaged first"},"secondary":["zero-latency class separation","per-clip token AUC, open vowel positive, ties=.5","overall aperture, mouth width/height/area, absolute aperture velocity, yaw/pitch"]},"input_seals":{"availability_freeze_path":str(args.availability_freeze.resolve()),"availability_freeze_sha256":sha256_file(args.availability_freeze),"availability_freeze_seal_sha256":sha256_file(args.availability_freeze.with_name("world_visual_availability_freeze.seal.json")),"track_output_root":str(args.track_dir.resolve()),"producer_plan_raw_sha256":EXPECTED["plan_raw"],"producer_plan_fingerprint_sha256":EXPECTED["plan_fingerprint"],"track_extractor_sha256":EXPECTED["track_extractor"],"grid_inventory_sha256":EXPECTED["grid_inventory"],"eligibility_manifest_sha256":EXPECTED["eligibility"],"calibration_sha256":EXPECTED["calibration"],"calibration_freeze_sha256":EXPECTED["calibration_freeze"],"tone_mapping_raw_sha256":EXPECTED["phone_mapping_raw"],"tone_mapping_fingerprint":EXPECTED["phone_mapping_fingerprint"],"visual_core_sha256":EXPECTED["visual_core"],"tone_adapter_sha256":EXPECTED["tone_adapter"]},"intended_denominator":{"samples":80,"speakers":40,"clips_per_speaker":2,"directions":2,"contexts":160,"videos":1600},"direction_summaries":summaries,"context_rows":analysis_rows}
    atomic_json(analysis_path,result)
    report={"status":"PASS","error_count":0,"analysis_sha256":sha256_file(analysis_path),"input_freeze_sha256":sha256_file(args.availability_freeze),"context_count":160,"video_count":1600,"score_or_effect_arrays_read":False}
    atomic_json(output/"world_visual_analysis_audit.json",report)
    files={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"world_visual_analysis_all_files_sha256.json","world_visual_analysis_seal.json"}}
    all_path=output/"world_visual_analysis_all_files_sha256.json"
    atomic_json(all_path,{"files":files})
    seal={"status":"SEALED","protocol_id":"world_visual_20261001_two_class_recipient_clock_v1","analysis_sha256":sha256_file(analysis_path),"audit_sha256":sha256_file(output/"world_visual_analysis_audit.json"),"availability_freeze_sha256":sha256_file(args.availability_freeze),"all_files_sha256":sha256_file(all_path),"file_count":len(files),"consumer_source_sha256":sha256_file(Path(__file__).resolve())}
    atomic_json(output/"world_visual_analysis_seal.json",seal)
    return {"status":"COMPLETE","analysis_path":str(analysis_path),"analysis_sha256":seal["analysis_sha256"],"seal_sha256":sha256_file(output/"world_visual_analysis_seal.json"),"files":len(files)}


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("freeze-availability","analyze"):
        child=sub.add_parser(name)
        child.add_argument("--run-dir",type=Path,required=True)
        child.add_argument("--execution-plan",type=Path,required=True)
        child.add_argument("--consumer-output",type=Path,required=True)
        child.add_argument("--runtime-flags",type=Path,required=True)
        child.add_argument("--track-dir",type=Path,required=True)
        child.add_argument("--grid-inventory",type=Path,required=True)
        child.add_argument("--eligibility-manifest",type=Path,required=True)
        child.add_argument("--calibration",type=Path,required=True)
        child.add_argument("--calibration-freeze",type=Path,required=True)
        child.add_argument("--phone-mapping",type=Path,required=True)
        child.add_argument("--core",type=Path,required=True)
        child.add_argument("--tone-adapter",type=Path,required=True)
        child.add_argument("--video-features",type=Path,required=True)
        child.add_argument("--extractor",type=Path,required=True)
        child.add_argument("--output-dir",type=Path,required=True)
    analyze_parser=sub.choices["analyze"]
    analyze_parser.add_argument("--availability-freeze",type=Path,required=True)
    return parser


def main(argv: Sequence[str]|None=None) -> int:
    args=build_parser().parse_args(argv)
    if args.command=="freeze-availability":
        result=freeze_availability(args)
    else:
        result=analyze(args)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(",",":")))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
