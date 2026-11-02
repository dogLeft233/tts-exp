#!/usr/bin/env python3
"""Write a non-authoritative score/status ledger for the fixed Dev540 rows.

The reader binds 360 same-host control rows and 180 fixed-step candidate rows,
including failures and missing results. It reports producer score summaries
and matrix references only: it does not read media or NPZ contents, recompute
SyncNet/supports, establish renderer/media eligibility, or select a method.
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
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
SOURCES = {
    BASE + "/join_v3_candidate_540_samehost_controls_attempt08.py": (84574, "7af975b83d8149b6b98c293500449f897aa8df380800324a7198cbc158ac797a"),
    BASE + "/run_stagea_dev_attempt06.py": (79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
    BASE + "/collect_v3_stagea_raw_media_evidence_attempt03.py": (43476, "244c79a513f90f280e159087dbe211c8df0b8469cb2858d9cf5169428151f329"),
    BASE + "/collect_v3_samehost_dev360_controls_evidence_attempt03.py": (24475, "aef8399b2c9fac925941ed73e0a201faedcf494ba97766f16be20e3876fbf17a"),
    BASE + "/run_v3_step120_candidate_samehost_attempt13.py": (35309, "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"),
    BASE + "/run_cross_frontend_dev_attempt09.py": (23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
    BASE + "/build_v3_candidate_samehost_controls_adapter_attempt06.py": (47404, "0a61f1c33099f7cd36ce497495b10eb70edc43f49ab911eaaa6e568bb4b32933"),
}
MODELS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
PREFIX = dict(zip(MODELS, ("w2l", "muse", "ditto")))
IDENTITY, RMS, CANDIDATE = "identity", "rms_fixed", "shared_recipient_transition_head_v3_fit40_step120"
METHODS = (IDENTITY, RMS, CANDIDATE)
SEEDS = (20261003, 20261004, 20261005)
IDENTITY_FIELDS = ("row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
                   "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256")
MAX_INPUT = 8 * 1024 * 1024 * 1024
MAX_MEMBER = 128 * 1024 * 1024
MAX_UNPACKED = 4 * 1024 * 1024 * 1024


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha(canonical({key: item for key, item in value.items() if key != field}))


def real_file(path: Path, label: str, cap: int) -> Path:
    p = Path(os.path.abspath(os.fspath(path)))
    cur = Path(p.anchor)
    for part in p.parts[1:]:
        cur /= part
        if cur.is_symlink() or getattr(cur, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cur}")
    if not p.is_file() or p.stat().st_size > cap:
        raise ValueError(f"{label} is absent, nonregular, or over cap: {p}")
    return p.resolve(strict=True)


def hash_file(path: Path, label: str, cap: int) -> tuple[Path, int, str]:
    p = real_file(path, label, cap)
    digest, count = hashlib.sha256(), 0
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            count += len(block)
            if count > cap:
                raise ValueError(f"{label} exceeded its byte cap while hashing")
            digest.update(block)
    if count != p.stat().st_size:
        raise ValueError(f"{label} changed while hashing")
    return p, count, digest.hexdigest()


def read_pinned(path: Path, label: str, expected: str, cap: int) -> tuple[dict[str, Any], bytes]:
    p = real_file(path, label, cap)
    raw = p.read_bytes()
    if len(raw) != p.stat().st_size or sha(raw) != expected.lower():
        raise ValueError(f"{label} raw bytes differ from their pin")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj, raw


def read_jsonl(path: Path, label: str, expected: str) -> tuple[list[dict[str, Any]], bytes]:
    p = real_file(path, label, 256 * 1024 * 1024)
    raw = p.read_bytes()
    if len(raw) != p.stat().st_size or sha(raw) != expected.lower():
        raise ValueError(f"{label} raw bytes differ from their pin")
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {number} is not an object")
        rows.append(row)
    return rows, raw


def source_pins() -> list[dict[str, Any]]:
    refs = []
    for relative, (nbytes, digest) in SOURCES.items():
        p = real_file(ROOT / relative, f"source {relative}", 16 * 1024 * 1024)
        raw = p.read_bytes()
        if len(raw) != nbytes or sha(raw) != digest:
            raise ValueError(f"pinned source changed: {relative}")
        refs.append({"path": str(p), "bytes": nbytes, "sha256": digest})
    p = real_file(Path(__file__), "score-only reader source", 2 * 1024 * 1024)
    raw = p.read_bytes()
    refs.append({"path": str(p), "bytes": len(raw), "sha256": sha(raw)})
    return refs


def open_score_archive(path: Path, nbytes: int, digest: str) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo], dict[str, Any], bytes]:
    p, size, actual = hash_file(path, "controls score archive", MAX_INPUT)
    if size != nbytes or actual != digest.lower():
        raise ValueError("controls score archive differs from exact byte/SHA pins")
    tar = tarfile.open(p, mode="r:gz")
    try:
        members: dict[str, tarfile.TarInfo] = {}
        unpacked = 0
        for info in tar.getmembers():
            name, posix = info.name, PurePosixPath(info.name)
            if (not name or "\\" in name or ":" in name or posix.is_absolute() or ".." in posix.parts
                    or name in members or not (info.isdir() or info.isfile()) or info.issym() or info.islnk()):
                raise ValueError(f"unsafe/duplicate score archive member {name!r}")
            if info.isfile():
                if info.size < 0 or info.size > MAX_MEMBER:
                    raise ValueError(f"score archive member exceeds cap: {name}")
                unpacked += info.size
                if unpacked > MAX_UNPACKED:
                    raise ValueError("score archive exceeds aggregate uncompressed cap")
            members[name] = info

        def member_json(member: str, cap: int) -> tuple[dict[str, Any], bytes]:
            info = members.get(member)
            if info is None or not info.isfile() or info.size > cap:
                raise ValueError(f"missing/unsafe score archive metadata member {member}")
            stream = tar.extractfile(info)
            if stream is None:
                raise ValueError(f"cannot read score archive member {member}")
            raw = stream.read(cap + 1)
            if len(raw) != info.size or len(raw) > cap:
                raise ValueError(f"score archive metadata member byte count mismatch {member}")
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError(f"score archive member is not an object: {member}")
            return obj, raw

        manifest, raw = member_json("evidence_archive_manifest.json", 4 * 1024 * 1024)
        if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
                or manifest.get("role") != "read_only_raw_score_and_samehost_identity_rms_control_evidence"
                or manifest.get("confirm_speakers_read") is not False
                or manifest.get("row_count") != 360 or manifest.get("planned_row_count") != 360
                or manifest.get("result_row_count") != 360
                or manifest.get("status") not in {"PASS", "INCOMPLETE"}
                or manifest.get("join_eligible") is not (manifest.get("status") == "PASS")
                or manifest.get("models") != list(MODELS) or manifest.get("methods") != [IDENTITY, RMS]
                or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
            raise ValueError("controls score archive scope/status/self-hash mismatch")
        return tar, members, manifest, raw
    except Exception:
        tar.close()
        raise


def archive_jsonl(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo], member: str,
                  cap: int) -> tuple[list[dict[str, Any]], bytes]:
    info = members.get(member)
    if info is None or not info.isfile() or info.size > cap:
        raise ValueError(f"missing/unsafe score archive ledger {member}")
    stream = tar.extractfile(info)
    if stream is None:
        raise ValueError(f"cannot read score archive ledger {member}")
    raw = stream.read(cap + 1)
    if len(raw) != info.size or len(raw) > cap:
        raise ValueError(f"score archive ledger byte count mismatch {member}")
    rows = [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"score archive ledger has a non-object row: {member}")
    return rows, raw


def score_ref(row: Mapping[str, Any], controls_assets: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if row.get("status") != "PASS":
        return {"status": "NOT_AVAILABLE"}
    score = row.get("score", {})
    desc = score.get("raw_matrix_npz", {})
    digest, member = desc.get("file_sha256"), desc.get("path")
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(member, str):
        raise ValueError(f"PASS score row lacks its producer matrix reference: {row.get('row_id')}")
    ref = {"status": "PRODUCER_REFERENCE_ONLY", "path": member, "sha256": digest,
           "bytes": desc.get("bytes"), "arrays_metadata": desc.get("arrays"),
           "content_rehashed_by_this_reader": False}
    if controls_assets is not None:
        asset = controls_assets.get(member)
        if (asset is None or asset.get("sha256") != digest or member != f"assets/{digest}.npz"
                or (type(desc.get("bytes")) is int and asset.get("bytes") != desc.get("bytes"))):
            raise ValueError(f"controls row matrix ref is absent from the hashed outer archive: {row.get('row_id')}")
        ref["bytes"] = asset.get("bytes")
    return ref


def ledger_row(row: Mapping[str, Any] | None, planned: Mapping[str, Any], method: str,
               controls_assets: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    status = row.get("status") if row is not None else "MISSING"
    if status not in {"PASS", "FAILED", "INCOMPLETE", "MISSING"}:
        raise ValueError(f"unknown row status for {planned.get('row_id')}")
    if status not in {"PASS", "FAILED", "INCOMPLETE", "MISSING"}:
        raise ValueError(f"unknown row status for {planned.get('row_id')}")
    if row is not None:
        for field in IDENTITY_FIELDS:
            if row.get(field) != planned.get(field):
                raise ValueError(f"row/plan identity mismatch at {field}: {planned.get('row_id')}")
    score = row.get("score", {}) if row else {}
    evaluator = score.get("official_evaluator", {}).get("receipt", {})
    return {**{field: planned.get(field) for field in IDENTITY_FIELDS},
            "method_id": method, "status": status,
            "failure_reason": row.get("failure_reason") if row else "missing_result_row",
            "score_model_sha256": score.get("score_model_sha256"),
            "official_evaluator_status": evaluator.get("status"),
            "reported_metrics_by_support": score.get("metrics_by_support") if status == "PASS" else None,
            "reported_support_by_support": score.get("support_by_support") if status == "PASS" else None,
            "raw_score_matrix": score_ref(row, controls_assets) if row else {"status": "NOT_AVAILABLE"},
            "media_eligibility": "NOT_EVALUATED"}


def parse_controls(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                   manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    assets_list = manifest.get("assets")
    if not isinstance(assets_list, list) or len(assets_list) != manifest.get("raw_score_asset_count"):
        raise ValueError("controls score archive asset count mismatch")
    assets: dict[str, Mapping[str, Any]] = {}
    for asset in assets_list:
        if (not isinstance(asset, dict) or asset.get("role") != "raw_score_arrays"
                or not isinstance(asset.get("path"), str) or asset["path"] in assets
                or len(asset.get("sha256", "")) != 64 or asset["path"] != f"assets/{asset['sha256']}.npz"
                or type(asset.get("bytes")) is not int):
            raise ValueError("controls asset descriptor is malformed")
        if asset["path"] not in members or not members[asset["path"]].isfile() or members[asset["path"]].size != asset["bytes"]:
            raise ValueError(f"controls NPZ member descriptor differs from archive headers: {asset['path']}")
        assets[asset["path"]] = asset
    by_id: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for model in MODELS:
        prefix = dict(zip(MODELS, ("w2l", "muse", "ditto")))[model]
        plan, plan_raw = read_archive_json(tar, members, f"producer/{prefix}/plan.json", 8 * 1024 * 1024)
        result_manifest, result_manifest_raw = read_archive_json(tar, members, f"producer/{prefix}/result_manifest.json", 2 * 1024 * 1024)
        results, result_raw = archive_jsonl(tar, members, f"producer/{prefix}/results.jsonl", 64 * 1024 * 1024)
        if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
                or plan.get("models", [{}])[0].get("model_id") != model or len(plan.get("rows", [])) != 120
                or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")
                or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
                or result_manifest.get("plan_raw_sha256") != sha(plan_raw)
                or result_manifest.get("results_jsonl_sha256") != sha(result_raw)
                or result_manifest.get("run_id") != plan.get("run_id")
                or result_manifest.get("planned_rows") != 120 or result_manifest.get("result_rows") != 120
                or result_manifest.get("status") not in {"PASS", "INCOMPLETE"} or len(results) != 120):
            raise ValueError(f"controls {prefix} plan/result chain invalid")
        planned_by_id = {row.get("row_id"): row for row in plan["rows"]}
        if len(planned_by_id) != 120:
            raise ValueError(f"controls {prefix} plan duplicates row IDs")
        rows_by_id = {row.get("row_id"): row for row in results}
        if len(rows_by_id) != 120 or set(rows_by_id) != set(planned_by_id):
            raise ValueError(f"controls {prefix} results fail the 120-row denominator")
        counts = Counter(row.get("status") for row in results)
        if any(row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"} for row in results):
            raise ValueError(f"controls {prefix} contains an unknown row status")
        if {name: int(result_manifest.get("status_counts", {}).get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")} != {
                name: int(counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")}:
            raise ValueError(f"controls {prefix} status counts mismatch")
        expected_status = "PASS" if not counts["FAILED"] and not counts["INCOMPLETE"] else "INCOMPLETE"
        if result_manifest.get("status") != expected_status:
            raise ValueError(f"controls {prefix} manifest status contradicts row counts")
        for planned in plan["rows"]:
            row = rows_by_id[planned["row_id"]]
            method = planned.get("method_id")
            if method not in (IDENTITY, RMS):
                raise ValueError(f"controls plan contains unexpected method {method}")
            rec = ledger_row(row, planned, method, assets)
            by_id[planned["row_id"]] = {"planned": planned, "row": row, "record": rec}
            records.append(rec)
    if len(records) != 360 or len(by_id) != 360:
        raise ValueError("controls score ledger does not preserve all 360 rows")
    totals = Counter(row["status"] for row in records)
    if {s: int(manifest.get("status_counts", {}).get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")} != {
            s: int(totals.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
        raise ValueError("controls outer manifest status counts differ from its three producer ledgers")
    expected_status = "PASS" if not totals["FAILED"] and not totals["INCOMPLETE"] and bool(manifest.get("assets")) else "INCOMPLETE"
    if manifest.get("status") != expected_status:
        raise ValueError("controls outer manifest status contradicts rows/score asset inventory")
    return by_id, records


def read_archive_json(tar: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                      member: str, cap: int) -> tuple[dict[str, Any], bytes]:
    info = members.get(member)
    if info is None or not info.isfile() or info.size > cap:
        raise ValueError(f"missing/unsafe archive metadata member {member}")
    stream = tar.extractfile(info)
    if stream is None:
        raise ValueError(f"cannot read archive metadata member {member}")
    raw = stream.read(cap + 1)
    if len(raw) != info.size or len(raw) > cap:
        raise ValueError(f"archive member byte count mismatch {member}")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"archive metadata member is not a JSON object: {member}")
    return obj, raw


def validate_candidate_plan(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or tuple(model.get("model_id") for model in plan.get("models", [])) != MODELS
            or len(plan.get("methods", [])) != 1 or plan["methods"][0].get("method_id") != CANDIDATE
            or plan["methods"][0].get("config", {}).get("fixed_optimizer_steps") != 120):
        raise ValueError("candidate plan is not frozen to the step-120 three-model method")
    rows = plan.get("rows", [])
    by_id = {row.get("row_id"): row for row in rows}
    if len(rows) != 180 or len(by_id) != 180:
        raise ValueError("candidate plan does not preserve exactly 180 unique rows")
    contexts, per_model = set(), defaultdict(set)
    sample_speaker = {}
    for row in rows:
        if row.get("method_id") != CANDIDATE or row.get("split") != "dev" or row.get("model_id") not in MODELS:
            raise ValueError("candidate plan contains an unexpected split/method/model")
        base = (row.get("sample_id"), row.get("speaker_id"), int(row.get("seed")))
        key = (*base[:2], row["model_id"], base[2])
        per_model[row["model_id"]].add(base)
        if row["model_id"] == MODELS[0]:
            contexts.add(base)
        if set(row.get("baseline_reference_row_ids", {})) != {IDENTITY, RMS}:
            raise ValueError(f"candidate row lacks both frozen baseline IDs: {row.get('row_id')}")
        old = sample_speaker.setdefault(base[0], base[1])
        if old != base[1]:
            raise ValueError("candidate sample maps to more than one speaker")
    if (len(contexts) != 60 or len({x[0] for x in contexts}) != 20 or len({x[1] for x in contexts}) != 10
            or {x[2] for x in contexts} != set(SEEDS)
            or any(per_model[model] != contexts for model in MODELS)):
        raise ValueError("candidate plan differs from the fixed 60-context/model denominator")
    if set(Counter(sample_speaker.values()).values()) != {2}:
        raise ValueError("candidate cohort does not contain two clips per speaker")
    return by_id


def build(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    pinned_sources = source_pins()
    plan, plan_raw = read_pinned(args.candidate_plan, "candidate plan", args.candidate_plan_sha256, 16 * 1024 * 1024)
    candidate_plan = validate_candidate_plan(plan)
    planned, planned_raw = read_jsonl(args.candidate_planned_rows, "candidate planned rows", args.candidate_planned_rows_sha256)
    if planned != plan.get("rows") or b"".join(canonical(row) + b"\n" for row in planned) != planned_raw:
        raise ValueError("candidate planned-row ledger differs from the sealed plan")
    result_manifest, result_manifest_raw = read_pinned(args.candidate_result_manifest, "candidate terminal manifest",
                                                       args.candidate_result_manifest_sha256, 4 * 1024 * 1024)
    result_rows, result_raw = read_jsonl(args.candidate_results, "candidate terminal results", args.candidate_results_sha256)
    if (result_manifest.get("run_id") != plan.get("run_id")
            or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
            or result_manifest.get("plan_raw_sha256") != sha(plan_raw)
            or result_manifest.get("results_jsonl_sha256") != sha(result_raw)
            or result_manifest.get("planned_rows") != 180 or result_manifest.get("result_rows") != len(result_rows)
            or result_manifest.get("status") not in {"PASS", "INCOMPLETE"}
            or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
        raise ValueError("candidate terminal manifest does not bind its 180-row plan/results")
    observed = Counter(row.get("status") for row in result_rows)
    if any(row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"} for row in result_rows):
        raise ValueError("candidate result ledger contains an unknown row status")
    expected = result_manifest.get("status_counts", {})
    if {s: int(expected.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")} != {
            s: int(observed.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
        raise ValueError("candidate row status counts differ from terminal manifest")
    candidate_by_id = {}
    for row in result_rows:
        row_id = row.get("row_id")
        if row_id not in candidate_plan or row_id in candidate_by_id:
            raise ValueError(f"candidate results contain unexpected/duplicate row {row_id}")
        candidate_by_id[row_id] = row
    if result_manifest["status"] == "PASS" and (len(candidate_by_id) != 180 or observed["FAILED"] or observed["INCOMPLETE"]):
        raise ValueError("candidate PASS manifest contradicts row statuses/denominator")
    if result_manifest["status"] == "INCOMPLETE" and not (len(candidate_by_id) < 180 or observed["FAILED"] or observed["INCOMPLETE"]):
        raise ValueError("candidate INCOMPLETE manifest has no failed/missing row")

    controls, controls_bytes, controls_sha = hash_file(args.controls_evidence_archive, "controls score archive", MAX_INPUT)
    if controls_bytes != args.controls_evidence_archive_bytes or controls_sha != args.controls_evidence_archive_sha256.lower():
        raise ValueError("controls score archive differs from its exact byte/SHA pin")
    tar, members, control_manifest, control_manifest_raw = open_score_archive(controls, controls_bytes, controls_sha)
    try:
        controls_by_id, _ = parse_controls(tar, members, control_manifest)
        asset_members = {item["path"]: item for item in control_manifest.get("assets", [])}
    finally:
        tar.close()
    baseline = plan.get("execution", {}).get("baseline_controls", {})
    if (baseline.get("run_id") != control_manifest.get("run_id")
            or baseline.get("plan_raw_sha256") != control_manifest.get("controls_plan_raw_sha256")):
        raise ValueError("candidate plan does not cite the exact controls run/plan in its score archive")
    candidate_output = Path(os.path.abspath(os.fspath(args.candidate_output_root)))
    cur = Path(candidate_output.anchor)
    for part in candidate_output.parts[1:]:
        cur /= part
        if cur.is_symlink() or getattr(cur, "is_junction", lambda: False)():
            raise ValueError(f"candidate output root traverses a symlink/junction: {cur}")
    if not candidate_output.is_dir():
        raise ValueError("candidate output root must be an existing directory")

    rows_out, contexts = [], {}
    for planned_row in plan["rows"]:
        row_id = planned_row["row_id"]
        key = (planned_row["sample_id"], planned_row["speaker_id"], planned_row["model_id"], int(planned_row["seed"]))
        refs = planned_row["baseline_reference_row_ids"]
        statuses = {}
        for method in (IDENTITY, RMS):
            ref = controls_by_id.get(refs[method])
            if ref is None:
                raise ValueError(f"candidate row references absent {method} score row: {row_id}")
            cp, cr = ref["planned"], ref["row"]
            for field in ("sample_id", "speaker_id", "model_id", "audio_frontend", "face_id", "seed"):
                if cp.get(field) != planned_row.get(field) or cr.get(field) != cp.get(field):
                    raise ValueError(f"baseline/candidate context mismatch at {field}: {row_id}/{method}")
            for field in ("natural_pcm", "source_face", "source_video", "score_audio"):
                if cp.get(field) != planned_row.get(field):
                    raise ValueError(f"baseline/candidate original source mismatch at {field}: {row_id}/{method}")
            if cp.get("method_id") != method:
                raise ValueError(f"candidate baseline ID resolves to wrong method: {row_id}/{method}")
            record = ledger_row(cr, cp, method, asset_members)
            record.update({"paired_candidate_row_id": row_id, "context_key": list(key)})
            rows_out.append(record)
            statuses[method] = record["status"]
        candidate = candidate_by_id.get(row_id)
        if candidate is not None:
            for field in IDENTITY_FIELDS:
                if candidate.get(field) != planned_row.get(field):
                    raise ValueError(f"candidate result/plan mismatch at {field}: {row_id}")
        record = ledger_row(candidate, planned_row, CANDIDATE)
        record.update({"paired_control_row_ids": dict(refs), "context_key": list(key)})
        rows_out.append(record)
        statuses[CANDIDATE] = record["status"]
        contexts[key] = {"sample_id": key[0], "speaker_id": key[1], "model_id": key[2], "seed": key[3],
                         "method_status": statuses,
                         "score_complete": all(statuses[m] == "PASS" for m in METHODS)}
    if len(rows_out) != 540 or len({row["row_id"] for row in rows_out}) != 540 or len(contexts) != 180:
        raise ValueError("score-only output does not preserve the exact unique 540 rows/180 contexts")
    per_model = {}
    for model in MODELS:
        group = [value for key, value in contexts.items() if key[2] == model]
        method_counts = {method: {s: sum(row["method_status"][method] == s for row in group)
                                  for s in ("PASS", "FAILED", "INCOMPLETE", "MISSING")} for method in METHODS}
        speakers = sorted({row["speaker_id"] for row in group})
        complete_speakers = [speaker for speaker in speakers
            if sum(row["speaker_id"] == speaker for row in group) == 6
            and all(row["score_complete"] for row in group if row["speaker_id"] == speaker)]
        per_model[model] = {"planned_contexts": 60,
                            "score_complete_contexts": sum(row["score_complete"] for row in group),
                            "method_row_status_counts": method_counts,
                            "planned_speakers": 10,
                            "speakers_complete_across_2_clips_x_3_seeds": len(complete_speakers),
                            "complete_speaker_ids": complete_speakers,
                            "media_eligibility": "NOT_EVALUATED"}
    counts = {s: sum(row["status"] == s for row in rows_out) for s in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows_out)
    source_path, source_size, source_sha = hash_file(Path(__file__), "score-only reader", 2 * 1024 * 1024)
    evidence = {
        "schema": "cross_tfg_dev540_score_only_failure_accounting_v1",
        "status": "ENGINEERING_SCORE_ONLY_MEDIA_INCOMPLETE",
        "run_id": plan.get("run_id"), "planned_rows": 540, "row_status_counts": counts,
        "controls_rows": 360, "candidate_rows": 180, "contexts": 180, "contexts_per_model": 60,
        "method_order": list(METHODS), "model_order": list(MODELS), "seed_family": list(SEEDS),
        "per_model_status_and_speaker_coverage": per_model,
        "controls_archive": {"path": str(controls), "bytes": controls_bytes, "sha256": controls_sha,
                              "manifest_raw_sha256": sha(control_manifest_raw),
                              "producer_status": control_manifest.get("status"),
                              "join_eligible": control_manifest.get("join_eligible")},
        "candidate_inputs": {
            "plan": {"path": str(args.candidate_plan.resolve(strict=True)), "bytes": len(plan_raw), "sha256": sha(plan_raw)},
            "planned_rows": {"path": str(args.candidate_planned_rows.resolve(strict=True)), "bytes": len(planned_raw), "sha256": sha(planned_raw)},
            "result_manifest": {"path": str(args.candidate_result_manifest.resolve(strict=True)), "bytes": len(result_manifest_raw), "sha256": sha(result_manifest_raw)},
            "results": {"path": str(args.candidate_results.resolve(strict=True)), "bytes": len(result_raw), "sha256": sha(result_raw)},
            "candidate_output_root": str(candidate_output.resolve(strict=True)),
            "producer_status": result_manifest.get("status"), "observed_rows": len(result_rows),
            "missing_planned_rows": 180 - len(candidate_by_id)},
        "rows_file": {"path": "score_only_rows.jsonl", "bytes": len(rows_raw), "sha256": sha(rows_raw), "row_count": 540},
        "matrix_content_rehashed": False, "reported_metrics_recomputed": False,
        "media_opened": False, "strict_media_qualification_claimed": False,
        "visual_clock_pcm_guards_run": False, "method_acceptance": False,
        "method_selected": False, "confirm_data_opened": False,
        "source_closure": [*pinned_sources, {"path": str(source_path), "bytes": source_size, "sha256": source_sha}],
        "created_unix": time.time()}
    evidence["receipt_sha256"] = payload_sha(evidence, "receipt_sha256")
    return evidence, rows_raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controls-evidence-archive", type=Path, required=True)
    ap.add_argument("--controls-evidence-archive-bytes", type=int, required=True)
    ap.add_argument("--controls-evidence-archive-sha256", required=True)
    ap.add_argument("--candidate-plan", type=Path, required=True)
    ap.add_argument("--candidate-plan-sha256", required=True)
    ap.add_argument("--candidate-planned-rows", type=Path, required=True)
    ap.add_argument("--candidate-planned-rows-sha256", required=True)
    ap.add_argument("--candidate-result-manifest", type=Path, required=True)
    ap.add_argument("--candidate-result-manifest-sha256", required=True)
    ap.add_argument("--candidate-results", type=Path, required=True)
    ap.add_argument("--candidate-results-sha256", required=True)
    ap.add_argument("--candidate-output-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
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
