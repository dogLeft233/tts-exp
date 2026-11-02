#!/usr/bin/env python3
"""Create a non-authoritative score-only failure ledger for the frozen Dev540 cohort.

This reader preserves all 360 control rows and 180 candidate rows, including
missing/non-renderer failures. It binds producer-reported score summaries and
raw NPZ byte pins, but does not open media, recompute SyncNet, assert media or
strict-renderer eligibility, or select a method. Its output is always marked
engineering-only and cannot replace the official same-host join/readback.
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
EXPECTED_SOURCES = {
    BASE + "/join_v3_candidate_540_samehost_controls_attempt08.py": (84574, "7af975b83d8149b6b98c293500449f897aa8df380800324a7198cbc158ac797a"),
    BASE + "/run_stagea_dev_attempt06.py": (79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
    BASE + "/collect_v3_stagea_raw_media_evidence_attempt03.py": (43476, "244c79a513f90f280e159087dbe211c8df0b8469cb2858d9cf5169428151f329"),
    BASE + "/collect_v3_samehost_dev360_controls_evidence_attempt03.py": (24475, "aef8399b2c9fac925941ed73e0a201faedcf494ba97766f16be20e3876fbf17a"),
    BASE + "/run_v3_step120_candidate_samehost_attempt13.py": (35309, "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"),
    BASE + "/run_cross_frontend_dev_attempt09.py": (23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
    BASE + "/build_v3_candidate_samehost_controls_adapter_attempt06.py": (47404, "0a61f1c33099f7cd36ce497495b10eb70edc43f49ab911eaaa6e568bb4b32933"),
}
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
MODEL_PREFIX = {
    "wav2lip_gan_bac9a81e": "w2l",
    "musetalk_fixed_portrait_attempt16": "muse",
    "ditto_hubert_fixed_portrait_attempt18": "ditto",
}
METHOD_IDENTITY = "identity"
METHOD_RMS = "rms_fixed"
METHOD_V3 = "shared_recipient_transition_head_v3_fit40_step120"
METHODS = (METHOD_IDENTITY, METHOD_RMS, METHOD_V3)
SEEDS = (20261003, 20261004, 20261005)
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
MAX_ARCHIVE_UNPACKED_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def safe_regular(path: Path, label: str, cap: int) -> tuple[Path, bytes]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    if not lexical.is_file() or lexical.stat().st_size > cap:
        raise ValueError(f"{label} is absent, nonregular, or exceeds its byte cap: {lexical}")
    raw = lexical.read_bytes()
    if len(raw) != lexical.stat().st_size:
        raise ValueError(f"{label} changed while being read: {lexical}")
    return lexical.resolve(strict=True), raw


def safe_path_only(path: Path, label: str, cap: int) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    if not lexical.is_file() or lexical.stat().st_size > cap:
        raise ValueError(f"{label} is absent, nonregular, or exceeds its byte cap: {lexical}")
    return lexical.resolve(strict=True)


def hash_regular(path: Path, label: str, cap: int) -> tuple[Path, int, str]:
    resolved = safe_path_only(path, label, cap)
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            if size > cap:
                raise ValueError(f"{label} grew beyond its byte cap while hashing")
            digest.update(block)
    if size != resolved.stat().st_size:
        raise ValueError(f"{label} changed while being hashed")
    return resolved, size, digest.hexdigest()


def read_json(path: Path, label: str, expected_sha: str, cap: int = MAX_JSON_BYTES) -> tuple[Any, bytes]:
    _, raw = safe_regular(path, label, cap)
    if sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA-256 mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, raw


def read_jsonl(path: Path, label: str, expected_sha: str) -> tuple[list[dict[str, Any]], bytes]:
    _, raw = safe_regular(path, label, MAX_JSONL_BYTES)
    if sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line.decode("utf-8"))
        if not isinstance(item, dict):
            raise ValueError(f"{label} line {line_no} is not an object")
        rows.append(item)
    return rows, raw


def validate_source_closure() -> list[dict[str, Any]]:
    observed = []
    for relative, (expected_bytes, expected_sha) in EXPECTED_SOURCES.items():
        path = ROOT / relative
        resolved, raw = safe_regular(path, f"pinned source {relative}", 16 * 1024 * 1024)
        actual = sha_bytes(raw)
        if len(raw) != expected_bytes or actual != expected_sha:
            raise ValueError(f"pinned source changed: {relative}")
        observed.append({"path": str(resolved), "bytes": len(raw), "sha256": actual})
    return observed


def safe_tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    unpacked = 0
    for info in archive.getmembers():
        name = info.name
        posix = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or posix.is_absolute() or ".." in posix.parts
                or name in members or not info.isfile() or info.issym() or info.islnk()):
            if info.isdir() and name not in members and not posix.is_absolute() and ".." not in posix.parts and "\\" not in name:
                members[name] = info
                continue
            raise ValueError(f"controls archive has an unsafe/duplicate/nonregular member: {name!r}")
        if info.size < 0 or info.size > MAX_MEMBER_BYTES:
            raise ValueError(f"controls archive member exceeds its cap: {name}")
        unpacked += info.size
        if unpacked > MAX_ARCHIVE_UNPACKED_BYTES:
            raise ValueError("controls archive exceeds the score-only aggregate uncompressed cap")
        members[name] = info
    return members


def tar_member_bytes(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                     name: str, cap: int) -> bytes:
    info = members.get(name)
    if info is None or not info.isfile() or info.size > cap:
        raise ValueError(f"controls archive member is absent/nonregular/over cap: {name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"could not read controls archive member: {name}")
    raw = stream.read(cap + 1)
    if len(raw) != info.size or len(raw) > cap:
        raise ValueError(f"controls archive member size changed/exceeds cap: {name}")
    return raw


def tar_member_hash(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                    name: str, cap: int) -> tuple[int, str]:
    info = members.get(name)
    if info is None or not info.isfile() or info.size > cap:
        raise ValueError(f"controls score matrix member is absent/nonregular/over cap: {name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"could not read controls score matrix member: {name}")
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1 << 20), b""):
        size += len(block)
        if size > cap:
            raise ValueError(f"controls score matrix member exceeds cap: {name}")
        digest.update(block)
    if size != info.size:
        raise ValueError(f"controls score matrix member byte count mismatch: {name}")
    return size, digest.hexdigest()


def open_controls_archive(path: Path, expected_bytes: int, expected_sha: str) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo], dict[str, Any], bytes]:
    resolved, archive_bytes, archive_sha = hash_regular(path, "controls score archive", MAX_ARCHIVE_BYTES)
    if archive_bytes != expected_bytes or archive_sha != expected_sha.lower():
        raise ValueError("controls score archive differs from exact raw byte/SHA pins")
    archive = tarfile.open(resolved, mode="r:gz")
    try:
        members = safe_tar_members(archive)
        manifest_raw = tar_member_bytes(archive, members, "evidence_archive_manifest.json", 4 * 1024 * 1024)
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
                or manifest.get("role") != "read_only_raw_score_and_samehost_identity_rms_control_evidence"
                or manifest.get("confirm_speakers_read") is not False
                or manifest.get("row_count") != 360 or manifest.get("planned_row_count") != 360
                or manifest.get("result_row_count") != 360
                or manifest.get("status") not in {"PASS", "INCOMPLETE"}
                or manifest.get("join_eligible") is not (manifest.get("status") == "PASS")
                or manifest.get("methods") != [METHOD_IDENTITY, METHOD_RMS]
                or set(manifest.get("models", [])) != set(EXPECTED_MODELS)
                or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
            raise ValueError("controls score archive has an invalid schema/status/denominator/self-hash")
        return archive, members, manifest, manifest_raw
    except Exception:
        archive.close()
        raise


def verify_controls_scores(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                           archive_manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    asset_refs = archive_manifest.get("assets")
    if not isinstance(asset_refs, list) or len(asset_refs) != archive_manifest.get("raw_score_asset_count"):
        raise ValueError("controls archive asset inventory count mismatch")
    asset_by_path: dict[str, dict[str, Any]] = {}
    for asset in asset_refs:
        member = asset.get("path") if isinstance(asset, dict) else None
        if (not isinstance(member, str) or not member.startswith("assets/") or member in asset_by_path
                or asset.get("role") != "raw_score_arrays" or len(asset.get("sha256", "")) != 64
                or member != f"assets/{asset['sha256']}.npz" or type(asset.get("bytes")) is not int):
            raise ValueError("controls archive has a malformed score-matrix asset descriptor")
        asset_by_path[member] = asset
    if sum(int(item["bytes"]) for item in asset_refs) != archive_manifest.get("raw_score_asset_bytes_uncompressed"):
        raise ValueError("controls archive score asset byte total differs from its manifest")

    controls_by_id: dict[str, dict[str, Any]] = {}
    rows_out: list[dict[str, Any]] = []
    model_statuses: dict[str, Any] = {}
    for model_id in EXPECTED_MODELS:
        prefix = MODEL_PREFIX[model_id]
        plan_raw = tar_member_bytes(archive, members, f"producer/{prefix}/plan.json", 8 * 1024 * 1024)
        result_manifest_raw = tar_member_bytes(archive, members, f"producer/{prefix}/result_manifest.json", 2 * 1024 * 1024)
        results_raw = tar_member_bytes(archive, members, f"producer/{prefix}/results.jsonl", 64 * 1024 * 1024)
        model_plan = json.loads(plan_raw.decode("utf-8"))
        result_manifest = json.loads(result_manifest_raw.decode("utf-8"))
        rows = [json.loads(line.decode("utf-8")) for line in results_raw.splitlines() if line.strip()]
        if (model_plan.get("status") != "FROZEN"
                or model_plan.get("plan_sha256") != payload_sha(model_plan, "plan_sha256")
                or model_plan.get("run_id") != result_manifest.get("run_id")
                or model_plan.get("models", [{}])[0].get("model_id") != model_id
                or len(model_plan.get("rows", [])) != 120
                or len({row.get("row_id") for row in model_plan["rows"]}) != 120
                or result_manifest.get("planned_rows") != 120 or result_manifest.get("result_rows") != 120
                or result_manifest.get("plan_sha256") != model_plan.get("plan_sha256")
                or result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
                or result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw)
                or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")
                or result_manifest.get("status") not in {"PASS", "INCOMPLETE"}
                or len(rows) != 120):
            raise ValueError(f"controls {prefix} score package schema/plan/result binding mismatch")
        planned = {row.get("row_id"): row for row in model_plan["rows"]}
        if len(planned) != 120:
            raise ValueError(f"controls {prefix} plan has duplicate row IDs")
        result_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = row.get("row_id")
            if row_id not in planned or row_id in result_by_id or row.get("model_id") != model_id:
                raise ValueError(f"controls {prefix} result has an unexpected/duplicate row")
            source = planned[row_id]
            for field in ("row_id", "sample_id", "speaker_id", "model_id", "audio_frontend", "face_id", "seed", "method_id"):
                if row.get(field) != source.get(field):
                    raise ValueError(f"controls {prefix} result-plan mismatch for {field}: {row_id}")
            if row.get("method_id") not in {METHOD_IDENTITY, METHOD_RMS}:
                raise ValueError(f"controls {prefix} contains an unplanned method row")
            if row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"}:
                raise ValueError(f"controls {prefix} contains an unknown row status")
            matrix = {"status": "NOT_AVAILABLE"}
            if row.get("status") == "PASS":
                desc = row.get("score", {}).get("raw_matrix_npz", {})
                digest = desc.get("file_sha256")
                member = desc.get("path")
                asset = asset_by_path.get(member)
                if (not isinstance(digest, str) or len(digest) != 64 or asset is None
                        or asset.get("sha256") != digest
                        or (type(desc.get("bytes")) is int and asset.get("bytes") != desc.get("bytes"))):
                    raise ValueError(f"PASS controls row lacks a matching raw-score asset descriptor: {row_id}")
                nbytes, actual_sha = tar_member_hash(archive, members, member, MAX_MEMBER_BYTES)
                if nbytes != asset["bytes"] or actual_sha != digest:
                    raise ValueError(f"PASS controls row score bytes differ from their content hash: {row_id}")
                matrix = {"status": "HASHED", "archive_member": member, "bytes": nbytes, "sha256": actual_sha,
                          "arrays_metadata": desc.get("arrays")}
            record = score_row_record(row, source, matrix)
            result_by_id[row_id] = record
            controls_by_id[row_id] = {"planned": source, "result": row, "record": record}
        counts = Counter(row.get("status") for row in rows)
        expected_counts = result_manifest.get("status_counts", {})
        if {name: int(expected_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")} != {
            name: int(counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")
        }:
            raise ValueError(f"controls {prefix} row status counts differ from its manifest")
        expected_model_status = "PASS" if not counts["FAILED"] and not counts["INCOMPLETE"] else "INCOMPLETE"
        if result_manifest.get("status") != expected_model_status:
            raise ValueError(f"controls {prefix} manifest status contradicts its row counts")
        for row in model_plan["rows"]:
            record = result_by_id[row["row_id"]]
            rows_out.append(record)
        model_statuses[model_id] = {
            "planned_rows": 120,
            "status_counts": {name: int(counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")},
            "pair_contexts": 60,
            "identity_rms_pairs_complete": sum(
                1 for identity in rows if identity.get("method_id") == METHOD_IDENTITY
                and result_by_id.get(identity.get("row_id"), {}).get("status") == "PASS"
                and any(r.get("method_id") == METHOD_RMS
                        and (r.get("sample_id"), r.get("speaker_id"), r.get("seed")) ==
                            (identity.get("sample_id"), identity.get("speaker_id"), identity.get("seed"))
                        and r.get("status") == "PASS" for r in rows)),
        }
    total_counts = Counter(record["status"] for record in rows_out)
    if sum(total_counts.values()) != 360:
        raise ValueError("controls score ledger does not preserve all 360 rows")
    manifest_counts = archive_manifest.get("status_counts", {})
    if {name: int(manifest_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")} != {
        name: int(total_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")
    }:
        raise ValueError("controls archive aggregate row-status counts disagree with producer packages")
    expected_archive_status = "PASS" if not total_counts["FAILED"] and not total_counts["INCOMPLETE"] and bool(asset_refs) else "INCOMPLETE"
    if archive_manifest.get("status") != expected_archive_status:
        raise ValueError("controls archive status contradicts its row counts/score asset inventory")
    return controls_by_id, rows_out, {"row_count": 360,
        "run_id": archive_manifest.get("run_id"),
        "controls_plan_raw_sha256": archive_manifest.get("controls_plan_raw_sha256"),
        "status_counts": {name: int(total_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")},
        "models": model_statuses}


def score_row_record(row: Mapping[str, Any], planned: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    score = row.get("score", {})
    return {
        **{field: row.get(field) for field in IDENTITY_FIELDS},
        "status": row.get("status"),
        "failure_reason": row.get("failure_reason"),
        "natural_pcm_sha256": row.get("natural_pcm_sha256", planned.get("natural_pcm", {}).get("pcm_sha256")),
        "source_face_sha256": row.get("source_face_sha256", planned.get("source_face", {}).get("file_sha256")),
        "source_video_sha256": row.get("source_video_sha256", planned.get("source_video", {}).get("file_sha256")),
        "official_score_model_sha256": score.get("score_model_sha256"),
        "official_evaluator_status": score.get("official_evaluator", {}).get("receipt", {}).get("status"),
        "producer_reported_metrics_by_support": score.get("metrics_by_support") if row.get("status") == "PASS" else None,
        "producer_reported_support_by_support": score.get("support_by_support") if row.get("status") == "PASS" else None,
        "raw_score_matrix": dict(matrix),
        "score_values_recomputed": False,
    }


def validate_model_contexts(plan: Mapping[str, Any]) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    methods = plan.get("methods", [])
    models = plan.get("models", [])
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or tuple(model.get("model_id") for model in models) != EXPECTED_MODELS
            or len(methods) != 1 or methods[0].get("method_id") != METHOD_V3
            or methods[0].get("config", {}).get("fixed_optimizer_steps") != 120):
        raise ValueError("candidate plan is not the exact frozen three-model step-120 Dev plan")
    rows = plan.get("rows", [])
    if len(rows) != 180 or len({row.get("row_id") for row in rows}) != 180:
        raise ValueError("candidate plan does not preserve 180 unique planned rows")
    contexts: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("row_id")
        if (not isinstance(row_id, str) or row_id in rows_by_id
                or row.get("method_id") != METHOD_V3 or row.get("split") != "dev"
                or row.get("model_id") not in EXPECTED_MODELS):
            raise ValueError("candidate planned rows contain an invalid model/method/ID")
        key = (row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))
        if key in contexts or row.get("face_id") != "primary_face_static_portrait_v1":
            raise ValueError(f"candidate row context is duplicate or outside the fixed Dev face: {key}")
        refs = row.get("baseline_reference_row_ids", {})
        if set(refs) != {METHOD_IDENTITY, METHOD_RMS} or any(not isinstance(value, str) for value in refs.values()):
            raise ValueError(f"candidate row has incomplete frozen control references: {row_id}")
        contexts[key] = row
        rows_by_id[row_id] = row
    if (len(contexts) != 180 or len({(k[0], k[1], k[3]) for k in contexts}) != 60
            or len({key[0] for key in contexts}) != 20 or len({key[1] for key in contexts}) != 10
            or {key[3] for key in contexts} != set(SEEDS)):
        raise ValueError("candidate rows are not the frozen 20 clip x 10 speaker x 3 seed contexts")
    context_base: dict[tuple[str, str, int], dict[str, Any]] = {}
    per_model: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for key, row in contexts.items():
        base = (key[0], key[1], key[3])
        if base in context_base:
            other = context_base[base]
            for field in ("natural_pcm", "source_face", "source_video", "score_audio"):
                if row.get(field) != other.get(field):
                    raise ValueError(f"candidate frontends disagree on the same original context input: {base}/{field}")
        else:
            context_base[base] = row
        per_model[key[2]].add(base)
    if any(len(per_model[model]) != 60 for model in EXPECTED_MODELS):
        raise ValueError("candidate plan lacks exactly 60 contexts for one or more models")
    all_contexts = set(context_base)
    if any(per_model[model] != all_contexts for model in EXPECTED_MODELS):
        raise ValueError("candidate models do not cover the same exact 60 planned contexts")
    sample_speaker: dict[str, str] = {}
    for sample, speaker, seed in context_base:
        if sample in sample_speaker and sample_speaker[sample] != speaker:
            raise ValueError(f"sample is assigned to multiple speakers: {sample}")
        sample_speaker[sample] = speaker
    if any({seed for sample2, speaker2, seed in context_base if sample2 == sample} != set(SEEDS)
           for sample in sample_speaker):
        raise ValueError("a clip does not carry the exact three-seed family")
    if set(Counter(sample_speaker.values()).values()) != {2}:
        raise ValueError("the frozen 20-clip cohort is not balanced at two clips per speaker")
    return contexts, rows_by_id


def candidate_row_record(row: Mapping[str, Any] | None, planned: Mapping[str, Any],
                         candidate_root: Path) -> dict[str, Any]:
    status = row.get("status") if row is not None else "MISSING"
    if status not in {"PASS", "FAILED", "INCOMPLETE", "MISSING"}:
        raise ValueError(f"candidate result has an unknown row status: {planned.get('row_id')}")
    if row is None:
        reason = "planned_row_absent_from_terminal_candidate_ledger"
        matrix = {"status": "NOT_AVAILABLE"}
    else:
        for field in IDENTITY_FIELDS:
            if row.get(field) != planned.get(field):
                raise ValueError(f"candidate result differs from its plan at {field}: {planned.get('row_id')}")
        reason = row.get("failure_reason")
        matrix = {"status": "NOT_AVAILABLE"}
        if status == "PASS":
            score = row.get("score", {})
            desc = score.get("raw_matrix_npz", {})
            digest = desc.get("file_sha256")
            raw_path = desc.get("path")
            if not isinstance(digest, str) or len(digest) != 64 or not isinstance(raw_path, str):
                raise ValueError(f"PASS candidate row lacks its raw NPZ path/hash: {planned.get('row_id')}")
            path = Path(raw_path)
            if not path.is_absolute():
                path = candidate_root / path
            resolved, matrix_bytes, matrix_sha = hash_regular(path, "candidate raw score matrix", MAX_MEMBER_BYTES)
            try:
                resolved.relative_to(candidate_root.resolve(strict=True))
            except ValueError as exc:
                raise ValueError("candidate raw score matrix escapes the candidate output root") from exc
            if matrix_sha != digest or (type(desc.get("bytes")) is int and matrix_bytes != desc["bytes"]):
                raise ValueError(f"candidate raw score matrix hash/bytes mismatch: {planned.get('row_id')}")
            matrix = {"status": "HASHED", "path": str(resolved), "bytes": matrix_bytes, "sha256": digest,
                      "arrays_metadata": desc.get("arrays")}
    return score_row_record(row or {**{key: planned.get(key) for key in IDENTITY_FIELDS},
                                    "status": status, "failure_reason": reason}, planned, matrix)


def build_accounting(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    sources = validate_source_closure()
    reader_path, reader_raw = safe_regular(Path(__file__), "score-only reader source", 16 * 1024 * 1024)
    sources.append({"path": str(reader_path), "bytes": len(reader_raw), "sha256": sha_bytes(reader_raw)})
    candidate_plan, candidate_plan_raw = read_json(args.candidate_plan, "candidate plan", args.candidate_plan_sha256)
    if candidate_plan.get("plan_sha256") != payload_sha(candidate_plan, "plan_sha256"):
        raise ValueError("candidate plan self-hash mismatch")
    contexts, plan_by_id = validate_model_contexts(candidate_plan)
    planned_rows, planned_raw = read_jsonl(args.candidate_planned_rows, "candidate planned rows", args.candidate_planned_rows_sha256)
    if planned_rows != candidate_plan.get("rows") or b"".join(canonical(row) + b"\n" for row in planned_rows) != planned_raw:
        raise ValueError("candidate planned-row ledger differs from the exact frozen candidate plan")
    result_manifest, manifest_raw = read_json(args.candidate_result_manifest,
                                               "terminal candidate result manifest",
                                               args.candidate_result_manifest_sha256)
    candidate_rows, results_raw = read_jsonl(args.candidate_results, "terminal candidate results", args.candidate_results_sha256)
    if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
            or result_manifest.get("run_id") != candidate_plan.get("run_id")
            or result_manifest.get("plan_sha256") != candidate_plan.get("plan_sha256")
            or result_manifest.get("plan_raw_sha256") != sha_bytes(candidate_plan_raw)
            or result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw)
            or result_manifest.get("planned_rows") != 180
            or result_manifest.get("result_rows") != len(candidate_rows)
            or result_manifest.get("status") not in {"PASS", "INCOMPLETE"}
            or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
        raise ValueError("candidate terminal result manifest does not bind the exact plan and result ledger")
    observed_counts = Counter(str(row.get("status")) for row in candidate_rows)
    expected_counts = result_manifest.get("status_counts", {})
    if {name: int(expected_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")} != {
        name: int(observed_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")
    }:
        raise ValueError("candidate result status counts differ from its manifest")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        row_id = row.get("row_id")
        if row_id not in plan_by_id or row_id in candidate_by_id:
            raise ValueError(f"candidate results have an unplanned or duplicate ID: {row_id}")
        if row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"}:
            raise ValueError(f"candidate result has an unknown status: {row_id}")
        candidate_by_id[row_id] = row
    if result_manifest.get("status") == "PASS" and (observed_counts["FAILED"] or observed_counts["INCOMPLETE"]
            or len(candidate_by_id) != 180):
        raise ValueError("candidate manifest claims PASS despite failures/missing planned rows")
    if result_manifest.get("status") == "INCOMPLETE" and not (
            observed_counts["FAILED"] or observed_counts["INCOMPLETE"] or len(candidate_by_id) < 180):
        raise ValueError("candidate manifest claims INCOMPLETE without any incomplete/missing row")
    candidate_root, _ = safe_regular_marker(args.candidate_output_root)

    controls_archive, control_members, control_manifest, control_manifest_raw = open_controls_archive(
        args.controls_evidence_archive, args.controls_evidence_archive_bytes,
        args.controls_evidence_archive_sha256)
    try:
        controls_by_id, control_records, control_summary = verify_controls_scores(
            controls_archive, control_members, control_manifest)
    finally:
        controls_archive.close()
    baseline_binding = candidate_plan.get("execution", {}).get("baseline_controls", {})
    if (baseline_binding.get("run_id") != control_manifest.get("run_id")
            or baseline_binding.get("plan_raw_sha256") != control_manifest.get("controls_plan_raw_sha256")):
        raise ValueError("candidate plan does not cite the exact control run/plan represented by its score archive")

    # Pair every planned candidate row with its exact two frozen control IDs.
    joint_rows: list[dict[str, Any]] = []
    candidate_records_by_id: dict[str, dict[str, Any]] = {}
    context_records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in candidate_plan.get("rows", []):
        row_id = row["row_id"]
        key = (row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))
        refs = row["baseline_reference_row_ids"]
        context_status: dict[str, str] = {}
        for method in (METHOD_IDENTITY, METHOD_RMS):
            control_id = refs[method]
            source = controls_by_id.get(control_id)
            if source is None:
                raise ValueError(f"candidate plan references an absent control row: {control_id}")
            control_plan = source["planned"]
            control_result = source["result"]
            for field in ("sample_id", "speaker_id", "model_id", "audio_frontend", "face_id", "seed"):
                if control_plan.get(field) != row.get(field) or control_result.get(field) != control_plan.get(field):
                    raise ValueError(f"candidate/control source context mismatch at {field}: {row_id}/{control_id}")
            for field in ("natural_pcm", "source_face", "source_video", "score_audio"):
                if control_plan.get(field) != row.get(field):
                    raise ValueError(f"candidate/control original input binding mismatch at {field}: {row_id}/{control_id}")
            if control_plan.get("method_id") != method or control_result.get("method_id") != method:
                raise ValueError(f"candidate reference points to the wrong control method: {row_id}/{method}")
            record = source["record"]
            joint_rows.append({**record, "paired_candidate_row_id": row_id,
                               "candidate_context_key": [*key[:3], key[3]]})
            context_status[method] = record["status"]
        candidate_record = candidate_row_record(candidate_by_id.get(row_id), row, candidate_root)
        candidate_records_by_id[row_id] = candidate_record
        joint_rows.append({**candidate_record, "paired_control_row_ids": dict(refs),
                           "candidate_context_key": [*key[:3], key[3]]})
        context_status[METHOD_V3] = candidate_record["status"]
        context_records[key] = {
            "sample_id": row["sample_id"], "speaker_id": row["speaker_id"],
            "model_id": row["model_id"], "seed": int(row["seed"]),
            "method_status": context_status,
            "score_complete": all(context_status.get(method) == "PASS" for method in METHODS),
            "media_eligible": False,
        }
    if len(joint_rows) != 540 or len({row.get("row_id") for row in joint_rows}) != 540:
        raise ValueError("score-only ledger failed to preserve exactly 540 unique method rows")

    # Status-only coverage is reported per paired 60-context model cohort;
    # no support masks, metrics, inferential statistics, or media claims are recomputed.
    by_model: dict[str, Any] = {}
    all_speakers = sorted({key[1] for key in context_records})
    for model_id in EXPECTED_MODELS:
        model_contexts = [item for key, item in context_records.items() if key[2] == model_id]
        per_method = {}
        for method in METHODS:
            per_method[method] = {status: sum(item["method_status"][method] == status for item in model_contexts)
                                  for status in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
        complete_speakers = []
        for speaker in all_speakers:
            speaker_contexts = [item for item in model_contexts if item["speaker_id"] == speaker]
            if len(speaker_contexts) == 6 and all(item["score_complete"] for item in speaker_contexts):
                complete_speakers.append(speaker)
        by_model[model_id] = {
            "contexts_planned": len(model_contexts), "contexts_score_complete": sum(item["score_complete"] for item in model_contexts),
            "method_row_status_counts": per_method,
            "speakers_planned": 10, "speakers_score_complete_all_6_clip_seed_contexts": len(complete_speakers),
            "complete_speaker_ids": complete_speakers,
            "media_eligibility": "NOT_EVALUATED",
        }
    context_counts = {"planned": len(context_records),
                      "score_complete": sum(item["score_complete"] for item in context_records.values()),
                      "incomplete": sum(not item["score_complete"] for item in context_records.values())}
    status_counts = {status: sum(row.get("status") == status for row in joint_rows)
                     for status in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
    payload = {
        "schema": "cross_tfg_dev540_score_only_failure_accounting_v1",
        "status": "ENGINEERING_SCORE_ONLY_MEDIA_INCOMPLETE",
        "scope": "samehost_dev_360_controls_plus_fit40_step120_candidate_180",
        "planned_rows": 540,
        "row_status_counts": status_counts,
        "context_counts": context_counts,
        "contexts_per_model": 60,
        "model_order": list(EXPECTED_MODELS),
        "method_order": list(METHODS),
        "seed_family": list(SEEDS),
        "per_model_status_and_speaker_coverage": by_model,
        "controls_archive": {"path": str(args.controls_evidence_archive.resolve(strict=True)),
                              "bytes": args.controls_evidence_archive_bytes,
                              "sha256": args.controls_evidence_archive_sha256.lower(),
                              "manifest_raw_sha256": sha_bytes(control_manifest_raw),
                              "producer_status": control_manifest.get("status"),
                              "join_eligible": control_manifest.get("join_eligible")},
        "candidate_inputs": {
            "plan": {"path": str(args.candidate_plan.resolve(strict=True)), "bytes": len(candidate_plan_raw), "sha256": sha_bytes(candidate_plan_raw)},
            "planned_rows": {"path": str(args.candidate_planned_rows.resolve(strict=True)), "bytes": len(planned_raw), "sha256": sha_bytes(planned_raw)},
            "result_manifest": {"path": str(args.candidate_result_manifest.resolve(strict=True)), "bytes": len(manifest_raw), "sha256": sha_bytes(manifest_raw)},
            "results": {"path": str(args.candidate_results.resolve(strict=True)), "bytes": len(results_raw), "sha256": sha_bytes(results_raw)},
            "candidate_output_root": str(candidate_root),
            "producer_status": result_manifest.get("status"),
            "observed_rows": len(candidate_rows),
            "synthesized_missing_candidate_rows": 180 - len(candidate_by_id),
        },
        "score_matrix_bytes_hashed": True,
        "score_matrix_values_recomputed": False,
        "producer_reported_metric_summaries_included": True,
        "media_opened": False,
        "visual_clock_pcm_guards_run": False,
        "native_strict_media_qualification_claimed": False,
        "method_acceptance": False,
        "method_selected": False,
        "confirm_data_opened": False,
        "source_closure": sources,
        "control_status_summary": control_summary,
        "candidate_status_counts": {name: int(observed_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")},
        "created_unix": time.time(),
    }
    rows_raw = b"".join(canonical(row) + b"\n" for row in joint_rows)
    payload["rows"] = {"path": "score_only_rows.jsonl", "bytes": len(rows_raw), "sha256": sha_bytes(rows_raw), "row_count": len(joint_rows)}
    payload["receipt_sha256"] = payload_sha(payload, "receipt_sha256")
    return payload, rows_raw


def safe_regular_marker(path: Path) -> tuple[Path, None]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"candidate output root traverses a symlink/junction: {cursor}")
    if not lexical.is_dir():
        raise ValueError("candidate output root must be an existing real directory")
    return lexical.resolve(strict=True), None


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
    parser.add_argument("--candidate-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload, rows_raw = build_accounting(args)
        out_dir = Path(os.path.abspath(os.fspath(args.output_dir)))
        if out_dir.exists() or out_dir.is_symlink() or getattr(out_dir, "is_junction", lambda: False)():
            raise FileExistsError(f"score-only output namespace must be new: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=False)
        for path, raw in ((out_dir / "score_only_rows.jsonl", rows_raw),
                          (out_dir / "score_only_failure_accounting.json", pretty(payload))):
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps({"status": payload["status"], "output_dir": str(out_dir),
                          "planned_rows": payload["planned_rows"],
                          "row_status_counts": payload["row_status_counts"],
                          "receipt_sha256": payload["receipt_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
