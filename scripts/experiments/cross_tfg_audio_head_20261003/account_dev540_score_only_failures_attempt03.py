#!/usr/bin/env python3
"""Produce an engineering-only 540-row score/status ledger, without media gates.

This binds terminal score ledgers and their matrix references. It preserves
failed and missing rows, but does not read NPZ contents or media, recompute
SyncNet, assert strict-renderer/visual/clock/PCM eligibility, or select a
method. Its output cannot replace the frozen joiner/readback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003/"
PINNED_SOURCES = {
    "joiner": ("join_v3_candidate_540_samehost_controls_attempt08.py", 84574, "7af975b83d8149b6b98c293500449f897aa8df380800324a7198cbc158ac797a"),
    "stagea": ("run_stagea_dev_attempt06.py", 79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
    "controls_collector": ("collect_v3_samehost_dev360_controls_evidence_attempt03.py", 24475, "aef8399b2c9fac925941ed73e0a201faedcf494ba97766f16be20e3876fbf17a"),
    "candidate_dispatcher": ("run_v3_step120_candidate_samehost_attempt13.py", 35309, "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"),
    "candidate_adapter": ("run_cross_frontend_dev_attempt09.py", 23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
    "baseline_adapter": ("build_v3_candidate_samehost_controls_adapter_attempt06.py", 47404, "0a61f1c33099f7cd36ce497495b10eb70edc43f49ab911eaaa6e568bb4b32933"),
}
MODELS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
PREFIX = dict(zip(MODELS, ("w2l", "muse", "ditto")))
IDENTITY, RMS, CANDIDATE = "identity", "rms_fixed", "shared_recipient_transition_head_v3_fit40_step120"
METHODS = (IDENTITY, RMS, CANDIDATE)
SEEDS = (20261003, 20261004, 20261005)
IDENTITY_FIELDS = ("row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
                   "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256")
MAX_FILE = 8 * 1024 * 1024 * 1024
MAX_MEMBER = 128 * 1024 * 1024
MAX_UNPACKED = 4 * 1024 * 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def self_sha(value: Mapping[str, Any], field: str) -> str:
    return digest(canonical({k: v for k, v in value.items() if k != field}))


def real_file(path: Path, label: str, cap: int) -> Path:
    p = Path(os.path.abspath(os.fspath(path)))
    cur = Path(p.anchor)
    for part in p.parts[1:]:
        cur /= part
        if cur.is_symlink() or getattr(cur, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses symlink/junction: {cur}")
    if not p.is_file() or p.stat().st_size > cap:
        raise ValueError(f"{label} absent/nonregular/over cap: {p}")
    return p.resolve(strict=True)


def hash_file(path: Path, label: str, cap: int, expected_bytes: int | None = None,
              expected_sha: str | None = None) -> tuple[Path, int, str]:
    p = real_file(path, label, cap)
    h, n = hashlib.sha256(), 0
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            n += len(block)
            if n > cap:
                raise ValueError(f"{label} exceeded cap while hashing")
            h.update(block)
    actual = h.hexdigest()
    if n != p.stat().st_size or (expected_bytes is not None and n != expected_bytes) or (
            expected_sha is not None and actual != expected_sha.lower()):
        raise ValueError(f"{label} byte/SHA pin mismatch")
    return p, n, actual


def read_json(path: Path, label: str, expected_sha: str, cap: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    p = real_file(path, label, cap)
    raw = p.read_bytes()
    if len(raw) != p.stat().st_size or digest(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, raw


def read_jsonl(path: Path, label: str, expected_sha: str) -> tuple[list[dict[str, Any]], bytes]:
    p = real_file(path, label, 256 * 1024 * 1024)
    raw = p.read_bytes()
    if len(raw) != p.stat().st_size or digest(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    rows = []
    for i, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            obj = json.loads(line.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError(f"{label} line {i} is not an object")
            rows.append(obj)
    return rows, raw


def source_closure() -> list[dict[str, Any]]:
    records = []
    for role, (name, size, sha256) in PINNED_SOURCES.items():
        path, count, actual = hash_file(ROOT / BASE / name, f"pinned {role}", 16 * 1024 * 1024, size, sha256)
        records.append({"role": role, "path": str(path), "bytes": count, "sha256": actual})
    path, count, actual = hash_file(Path(__file__), "this score-only source", 2 * 1024 * 1024)
    records.append({"role": "score_only_reader", "path": str(path), "bytes": count, "sha256": actual})
    return records


def open_archive(path: Path, nbytes: int, sha256: str):
    resolved, count, actual = hash_file(path, "controls score archive", MAX_FILE, nbytes, sha256)
    tar = tarfile.open(resolved, "r:gz")
    try:
        members: dict[str, tarfile.TarInfo] = {}
        unpacked = 0
        for item in tar.getmembers():
            name, posix = item.name, PurePosixPath(item.name)
            if (not name or "\\" in name or ":" in name or posix.is_absolute() or ".." in posix.parts
                    or name in members or not (item.isfile() or item.isdir()) or item.issym() or item.islnk()):
                raise ValueError(f"unsafe/duplicate tar member: {name!r}")
            if item.isfile():
                if item.size < 0 or item.size > MAX_MEMBER:
                    raise ValueError(f"tar member exceeds cap: {name}")
                unpacked += item.size
                if unpacked > MAX_UNPACKED:
                    raise ValueError("controls archive exceeds uncompressed cap")
            members[name] = item
        manifest_raw = read_member(tar, members, "evidence_archive_manifest.json", 4 * 1024 * 1024)
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
                or manifest.get("role") != "read_only_raw_score_and_samehost_identity_rms_control_evidence"
                or manifest.get("confirm_speakers_read") is not False
                or manifest.get("row_count") != 360 or manifest.get("planned_row_count") != 360
                or manifest.get("result_row_count") != 360 or manifest.get("models") != list(MODELS)
                or manifest.get("methods") != [IDENTITY, RMS]
                or manifest.get("status") not in {"PASS", "INCOMPLETE"}
                or manifest.get("join_eligible") is not (manifest.get("status") == "PASS")
                or manifest.get("manifest_sha256") != self_sha(manifest, "manifest_sha256")):
            raise ValueError("controls archive schema/status/denominator/self-hash mismatch")
        return tar, members, manifest, manifest_raw
    except Exception:
        tar.close()
        raise


def read_member(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], name: str, cap: int) -> bytes:
    info = members.get(name)
    if info is None or not info.isfile() or info.size > cap:
        raise ValueError(f"missing/unsafe/oversized tar metadata member: {name}")
    stream = tar.extractfile(info)
    if stream is None:
        raise ValueError(f"cannot read tar member: {name}")
    raw = stream.read(cap + 1)
    if len(raw) != info.size or len(raw) > cap:
        raise ValueError(f"tar member byte count mismatch: {name}")
    return raw


def member_json(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], name: str,
                cap: int) -> tuple[dict[str, Any], bytes]:
    raw = read_member(tar, members, name, cap)
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"tar member is not a JSON object: {name}")
    return obj, raw


def member_jsonl(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], name: str,
                 cap: int) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_member(tar, members, name, cap)
    rows = [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"tar ledger contains non-object rows: {name}")
    return rows, raw


def matrix_reference(row: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    if row.get("status") != "PASS":
        return {"status": "NOT_AVAILABLE"}
    desc = row.get("score", {}).get("raw_matrix_npz", {})
    member, matrix_sha = desc.get("path"), desc.get("file_sha256")
    if not isinstance(member, str) or not isinstance(matrix_sha, str) or len(matrix_sha) != 64:
        raise ValueError(f"PASS row has no pinned score matrix: {row.get('row_id')}")
    record = {"status": "PRODUCER_REFERENCE_ONLY", "path_or_member": member,
              "sha256": matrix_sha, "bytes": desc.get("bytes"),
              "arrays_metadata": desc.get("arrays"), "content_rehashed": False}
    if assets is not None:
        asset = assets.get(member)
        if (asset is None or member != f"assets/{matrix_sha}.npz" or asset.get("sha256") != matrix_sha
                or (type(desc.get("bytes")) is int and asset.get("bytes") != desc["bytes"])):
            raise ValueError(f"controls row matrix reference is not in its pinned archive: {row.get('row_id')}")
        record["bytes"] = asset.get("bytes")
    return record


def make_row(row: Mapping[str, Any] | None, planned: Mapping[str, Any], method: str,
             assets: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    status = row.get("status") if row is not None else "MISSING"
    if status not in {"PASS", "FAILED", "INCOMPLETE", "MISSING"}:
        raise ValueError(f"unknown row status: {planned.get('row_id')}")
    if row is not None:
        for field in IDENTITY_FIELDS:
            if row.get(field) != planned.get(field):
                raise ValueError(f"row identity differs from plan at {field}: {planned.get('row_id')}")
    score = row.get("score", {}) if row else {}
    evaluator = score.get("official_evaluator", {}).get("receipt", {})
    return {**{field: planned.get(field) for field in IDENTITY_FIELDS},
            "method_id": method, "status": status,
            "failure_reason": row.get("failure_reason") if row else "planned_row_missing_from_result_ledger",
            "official_evaluator_status": evaluator.get("status"),
            "score_model_sha256": score.get("score_model_sha256"),
            "producer_metrics_by_support": score.get("metrics_by_support") if status == "PASS" else None,
            "producer_support_by_support": score.get("support_by_support") if status == "PASS" else None,
            "raw_score_matrix": matrix_reference(row, assets) if row is not None else {"status": "NOT_AVAILABLE"},
            "media_eligibility": "NOT_EVALUATED"}


def controls_rows(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                  manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    asset_list = manifest.get("assets", [])
    if not isinstance(asset_list, list) or len(asset_list) != manifest.get("raw_score_asset_count"):
        raise ValueError("controls score asset inventory count mismatch")
    assets: dict[str, Mapping[str, Any]] = {}
    for item in asset_list:
        if (not isinstance(item, dict) or item.get("role") != "raw_score_arrays"
                or not isinstance(item.get("path"), str) or item["path"] in assets
                or len(item.get("sha256", "")) != 64 or item["path"] != f"assets/{item['sha256']}.npz"
                or type(item.get("bytes")) is not int or item["path"] not in members
                or not members[item["path"]].isfile() or members[item["path"]].size != item["bytes"]):
            raise ValueError("malformed controls NPZ inventory/header")
        assets[item["path"]] = item
    output: dict[str, dict[str, Any]] = {}
    total = Counter()
    for model in MODELS:
        prefix = PREFIX[model]
        plan, plan_raw = member_json(tar, members, f"producer/{prefix}/plan.json", 8 * 1024 * 1024)
        manifest_row, manifest_raw = member_json(tar, members, f"producer/{prefix}/result_manifest.json", 2 * 1024 * 1024)
        results, results_raw = member_jsonl(tar, members, f"producer/{prefix}/results.jsonl", 64 * 1024 * 1024)
        planned = plan.get("rows", [])
        planned_by_id = {row.get("row_id"): row for row in planned}
        rows_by_id = {row.get("row_id"): row for row in results}
        if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != self_sha(plan, "plan_sha256")
                or len(plan.get("models", [])) != 1
                or plan.get("models", [{}])[0].get("model_id") != model or len(planned) != 120
                or len(planned_by_id) != 120 or len(results) != 120 or len(rows_by_id) != 120
                or set(planned_by_id) != set(rows_by_id)
                or manifest_row.get("schema_version") != "cross-tfg-audio-eval-v1"
                or manifest_row.get("run_id") != plan.get("run_id")
                or manifest_row.get("plan_sha256") != plan.get("plan_sha256")
                or manifest_row.get("plan_raw_sha256") != digest(plan_raw)
                or manifest_row.get("results_jsonl_sha256") != digest(results_raw)
                or manifest_row.get("planned_rows") != 120 or manifest_row.get("result_rows") != 120
                or manifest_row.get("manifest_sha256") != self_sha(manifest_row, "manifest_sha256")
                or manifest_row.get("status") not in {"PASS", "INCOMPLETE"}):
            raise ValueError(f"controls {prefix} plan/result identity chain mismatch")
        counts = Counter(row.get("status") for row in results)
        if any(row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"} for row in results):
            raise ValueError(f"controls {prefix} has unknown row status")
        if {s: int(manifest_row.get("status_counts", {}).get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")} != {
                s: int(counts.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
            raise ValueError(f"controls {prefix} status counts mismatch")
        expected = "PASS" if not counts["FAILED"] and not counts["INCOMPLETE"] else "INCOMPLETE"
        if manifest_row.get("status") != expected:
            raise ValueError(f"controls {prefix} status contradicts row ledger")
        for row_id, source in planned_by_id.items():
            row = rows_by_id[row_id]
            method = source.get("method_id")
            if method not in (IDENTITY, RMS):
                raise ValueError(f"controls plan has an unexpected method: {row_id}")
            rec = make_row(row, source, method, assets)
            output[row_id] = {"planned": source, "result": row, "record": rec}
            total[rec["status"]] += 1
    if len(output) != 360 or {s: int(manifest.get("status_counts", {}).get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")} != {
            s: int(total.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
        raise ValueError("controls archives do not bind the full 360 row status denominator")
    expected = "PASS" if not total["FAILED"] and not total["INCOMPLETE"] and bool(asset_list) else "INCOMPLETE"
    if manifest.get("status") != expected:
        raise ValueError("controls outer status contradicts its 360-row score inventory")
    return output


def candidate_plan_rows(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != self_sha(plan, "plan_sha256")
            or tuple(model.get("model_id") for model in plan.get("models", [])) != MODELS
            or len(plan.get("methods", [])) != 1 or plan["methods"][0].get("method_id") != CANDIDATE
            or plan["methods"][0].get("config", {}).get("fixed_optimizer_steps") != 120):
        raise ValueError("candidate plan is not the frozen three-model step-120 Dev plan")
    rows = plan.get("rows", [])
    by_id = {row.get("row_id"): row for row in rows}
    if len(rows) != 180 or len(by_id) != 180:
        raise ValueError("candidate plan denominator is not 180 unique rows")
    per_model: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    contexts: set[tuple[str, str, int]] = set()
    sample_speaker = {}
    source_identity_by_context: dict[tuple[str, str, int], tuple[Any, Any, Any]] = {}
    for row in rows:
        model = row.get("model_id")
        if model not in MODELS or row.get("split") != "dev" or row.get("method_id") != CANDIDATE:
            raise ValueError("candidate plan has unexpected split/method/model")
        base = (row.get("sample_id"), row.get("speaker_id"), int(row.get("seed")))
        per_model[model].add(base)
        if model == MODELS[0]:
            contexts.add(base)
        source_identity = (row.get("natural_pcm"), row.get("source_face"), row.get("source_video"))
        prior_source_identity = source_identity_by_context.setdefault(base, source_identity)
        if prior_source_identity != source_identity:
            raise ValueError(f"candidate backends do not share exact natural PCM/face/video input: {base}")
        if set(row.get("baseline_reference_row_ids", {})) != {IDENTITY, RMS}:
            raise ValueError(f"candidate plan lacks baseline row IDs: {row.get('row_id')}")
        prior = sample_speaker.setdefault(base[0], base[1])
        if prior != base[1]:
            raise ValueError("candidate clips map to multiple speakers")
    if (len(contexts) != 60 or len({x[0] for x in contexts}) != 20 or len({x[1] for x in contexts}) != 10
            or {x[2] for x in contexts} != set(SEEDS)
            or any(per_model[model] != contexts for model in MODELS)
            or set(Counter(sample_speaker.values()).values()) != {2}):
        raise ValueError("candidate plan context denominator differs from 20 clips x 10 speakers x 3 seeds")
    return by_id


def build(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    source_refs = source_closure()
    plan, plan_raw = read_json(args.candidate_plan, "candidate plan", args.candidate_plan_sha256)
    plan_by_id = candidate_plan_rows(plan)
    planned_rows, planned_raw = read_jsonl(args.candidate_planned_rows, "candidate planned rows", args.candidate_planned_rows_sha256)
    if planned_rows != plan.get("rows") or b"".join(canonical(row) + b"\n" for row in planned_rows) != planned_raw:
        raise ValueError("candidate planned-row file differs from frozen plan")
    result_manifest, manifest_raw = read_json(args.candidate_result_manifest, "candidate terminal manifest", args.candidate_result_manifest_sha256, 4 * 1024 * 1024)
    results, results_raw = read_jsonl(args.candidate_results, "candidate terminal results", args.candidate_results_sha256)
    if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
            or result_manifest.get("run_id") != plan.get("run_id") or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
            or result_manifest.get("plan_raw_sha256") != digest(plan_raw)
            or result_manifest.get("results_jsonl_sha256") != digest(results_raw)
            or result_manifest.get("planned_rows") != 180 or result_manifest.get("result_rows") != len(results)
            or result_manifest.get("status") not in {"PASS", "INCOMPLETE"}
            or result_manifest.get("manifest_sha256") != self_sha(result_manifest, "manifest_sha256")):
        raise ValueError("candidate terminal manifest is not bound to the exact frozen plan/results")
    observed = Counter(row.get("status") for row in results)
    if any(row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"} for row in results):
        raise ValueError("candidate ledger has unknown row status")
    if {s: int(result_manifest.get("status_counts", {}).get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")} != {
            s: int(observed.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
        raise ValueError("candidate row status counts mismatch")
    result_by_id = {row.get("row_id"): row for row in results}
    if len(result_by_id) != len(results) or any(row_id not in plan_by_id for row_id in result_by_id):
        raise ValueError("candidate results contain duplicate or unplanned IDs")
    if result_manifest["status"] == "PASS" and (len(result_by_id) != 180 or observed["FAILED"] or observed["INCOMPLETE"]):
        raise ValueError("candidate manifest claims PASS despite incomplete denominator/statuses")
    if result_manifest["status"] == "INCOMPLETE" and not (len(result_by_id) < 180 or observed["FAILED"] or observed["INCOMPLETE"]):
        raise ValueError("candidate manifest claims INCOMPLETE with no failed/missing row")

    archive, archive_bytes, archive_sha = hash_file(args.controls_evidence_archive, "controls score archive",
                                                     MAX_FILE, args.controls_evidence_archive_bytes,
                                                     args.controls_evidence_archive_sha256)
    tar, members, control_manifest, control_manifest_raw = open_archive(archive, archive_bytes, archive_sha)
    try:
        controls = controls_rows(tar, members, control_manifest)
        assets = {item["path"]: item for item in control_manifest.get("assets", [])}
    finally:
        tar.close()
    baseline = plan.get("execution", {}).get("baseline_controls", {})
    if (baseline.get("run_id") != control_manifest.get("run_id")
            or baseline.get("plan_raw_sha256") != control_manifest.get("controls_plan_raw_sha256")):
        raise ValueError("candidate plan does not cite this exact controls run/plan")

    rows_out: list[dict[str, Any]] = []
    context_status: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for planned in plan.get("rows", []):
        row_id = planned["row_id"]
        key = (planned["sample_id"], planned["speaker_id"], planned["model_id"], int(planned["seed"]))
        ref_ids = planned["baseline_reference_row_ids"]
        method_status = {}
        for method in (IDENTITY, RMS):
            source = controls.get(ref_ids[method])
            if source is None:
                raise ValueError(f"candidate row references absent {method} row: {row_id}")
            cp, cr = source["planned"], source["result"]
            for field in ("sample_id", "speaker_id", "model_id", "audio_frontend", "face_id", "seed"):
                if cp.get(field) != planned.get(field) or cr.get(field) != cp.get(field):
                    raise ValueError(f"control/candidate mismatch at {field}: {row_id}/{method}")
            for field in ("natural_pcm", "source_face", "source_video", "score_audio"):
                if cp.get(field) != planned.get(field):
                    raise ValueError(f"control/candidate input mismatch at {field}: {row_id}/{method}")
            if cp.get("method_id") != method:
                raise ValueError(f"candidate baseline reference has the wrong method: {row_id}/{method}")
            record = make_row(cr, cp, method, assets)
            record["paired_candidate_row_id"] = row_id
            record["context_key"] = list(key)
            rows_out.append(record)
            method_status[method] = record["status"]
        candidate = result_by_id.get(row_id)
        if candidate is not None:
            for field in IDENTITY_FIELDS:
                if candidate.get(field) != planned.get(field):
                    raise ValueError(f"candidate result differs from plan at {field}: {row_id}")
        record = make_row(candidate, planned, CANDIDATE)
        record["paired_control_row_ids"] = dict(ref_ids)
        record["context_key"] = list(key)
        rows_out.append(record)
        method_status[CANDIDATE] = record["status"]
        context_status[key] = {"speaker_id": key[1], "model_id": key[2], "method_status": method_status,
                               "score_complete": all(method_status[m] == "PASS" for m in METHODS)}
    if len(rows_out) != 540 or len({row["row_id"] for row in rows_out}) != 540 or len(context_status) != 180:
        raise ValueError("score-only report did not preserve the unique 540-row denominator")

    by_model = {}
    for model in MODELS:
        rows = [value for key, value in context_status.items() if key[2] == model]
        by_model[model] = {
            "contexts_planned": 60,
            "score_complete_contexts": sum(row["score_complete"] for row in rows),
            "row_status_by_method": {method: {status: sum(row["method_status"][method] == status for row in rows)
                                               for status in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
                                     for method in METHODS},
            "speakers_planned": 10,
            "speakers_complete_all_2_clips_x_3_seeds": sum(
                1 for speaker in {row["speaker_id"] for row in rows}
                if sum(row["speaker_id"] == speaker for row in rows) == 6
                and all(row["score_complete"] for row in rows if row["speaker_id"] == speaker)),
            "media_eligibility": "NOT_EVALUATED"}
    counts = {status: sum(row["status"] == status for row in rows_out)
              for status in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows_out)
    source_path, source_bytes, source_sha = hash_file(Path(__file__), "score-only reader source", 4 * 1024 * 1024)
    evidence = {
        "schema": "cross_tfg_dev540_score_only_failure_accounting_v1",
        "status": "ENGINEERING_SCORE_ONLY_MEDIA_INCOMPLETE",
        "run_id": plan.get("run_id"), "planned_rows": 540, "row_status_counts": counts,
        "controls_rows": 360, "candidate_rows": 180, "contexts": 180, "contexts_per_model": 60,
        "models": list(MODELS), "methods": list(METHODS), "seed_family": list(SEEDS),
        "per_model_coverage": by_model,
        "controls_archive": {"path": str(archive), "bytes": archive_bytes, "sha256": archive_sha,
                              "manifest_raw_sha256": digest(control_manifest_raw),
                              "producer_status": control_manifest.get("status"),
                              "join_eligible": control_manifest.get("join_eligible")},
        "candidate_inputs": {
            "plan": {"path": str(args.candidate_plan.resolve(strict=True)), "bytes": len(plan_raw), "sha256": digest(plan_raw)},
            "planned_rows": {"path": str(args.candidate_planned_rows.resolve(strict=True)), "bytes": len(planned_raw), "sha256": digest(planned_raw)},
            "result_manifest": {"path": str(args.candidate_result_manifest.resolve(strict=True)), "bytes": len(manifest_raw), "sha256": digest(manifest_raw)},
            "results": {"path": str(args.candidate_results.resolve(strict=True)), "bytes": len(results_raw), "sha256": digest(results_raw)},
            "producer_status": result_manifest.get("status"), "observed_rows": len(results),
            "missing_planned_rows": 180 - len(result_by_id)},
        "rows": {"path": "score_only_rows.jsonl", "bytes": len(rows_raw), "sha256": digest(rows_raw), "row_count": 540},
        "matrix_contents_rehashed": False, "score_values_recomputed": False,
        "media_opened": False, "strict_media_qualification_claimed": False,
        "visual_clock_pcm_guards_run": False, "method_acceptance": False,
        "method_selected": False, "confirm_data_opened": False,
        "source_closure": [*source_refs, {"path": str(source_path), "bytes": source_bytes, "sha256": source_sha}],
        "created_unix": time.time()}
    evidence["receipt_sha256"] = self_sha(evidence, "receipt_sha256")
    return evidence, rows_raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-evidence-archive", type=Path, required=True)
    parser.add_argument("--controls-evidence-archive-bytes", type=int, required=True)
    parser.add_argument("--controls-evidence-archive-sha256", required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan-sha256", required=True)
    parser.add_argument("--candidate-planned-rows", type=Path, required=True)
    parser.add_argument("--candidate-planned-rows-sha256", required=True)
    parser.add_argument("--candidate-result-manifest", type=Path, required=True)
    parser.add_argument("--candidate-result-manifest-sha256", required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-results-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence, rows_raw = build(args)
        output = Path(os.path.abspath(os.fspath(args.output_dir)))
        cur = Path(output.anchor)
        for part in output.parts[1:]:
            cur /= part
            if cur.is_symlink() or getattr(cur, "is_junction", lambda: False)():
                raise ValueError(f"output path traverses symlink/junction: {cur}")
        output.mkdir(parents=True, exist_ok=False)
        for path, raw in ((output / "score_only_rows.jsonl", rows_raw),
                          (output / "score_only_failure_accounting.json", pretty(evidence))):
            with path.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status": evidence["status"], "planned_rows": 540,
                          "row_status_counts": evidence["row_status_counts"],
                          "receipt_sha256": evidence["receipt_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
