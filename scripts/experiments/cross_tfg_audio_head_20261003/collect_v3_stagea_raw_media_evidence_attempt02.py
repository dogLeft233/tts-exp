#!/usr/bin/env python3
"""Seal raw Stage-A render/audio inputs for independent visual and PCM guards.

The collector reads only the frozen Dev plan, Stage-A ledgers, and the exact
files named by PASS rows. It does not run a renderer/scorer or open Confirm
data. It packages silent and muxed video, rational timestamp sidecars, and
deduplicated canonical source PCM WAVs in a separate bounded archive. The
manifest maps every planned row to content-addressed members so a later
consumer can reconstruct and independently run the frozen visual/clock/PCM
checks; ledger hashes alone are not treated as those checks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
SEEDS = (20261003, 20261004, 20261005)
CONTROLS_METHODS = ("identity", "rms_fixed")
CANDIDATE_METHOD = "shared_recipient_transition_head_v3_fit40_step120"
ARCHIVE_SCHEMA = "cross_tfg_v3_stagea_raw_media_evidence_archive_v2"
ARCHIVE_ROLE = "raw_dev_render_clock_and_pcm_inputs_for_independent_guards"
BLOCK = 1 << 20


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
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > cap:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size):
        raise ValueError(f"{label} changed while being read")
    return lexical.resolve(strict=True), raw


def read_json(path: Path, label: str, expected: str | None, cap: int) -> tuple[dict[str, Any], bytes]:
    _, raw = safe_regular(path, label, cap)
    if expected is not None and sha_bytes(raw) != expected.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj, raw


def hash_file(path: Path, expected_sha: str, cap: int, label: str) -> tuple[int, str]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > cap:
        raise ValueError(f"{label} is not a bounded, nonempty regular file")
    digest = hashlib.sha256()
    with lexical.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    after = lexical.stat(follow_symlinks=False)
    actual = digest.hexdigest()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or actual != expected_sha.lower()):
        raise ValueError(f"{label} changed or differs from its frozen SHA-256")
    return before.st_size, actual



def collect_strict_determinism_evidence(output_root: Path, plan: dict[str, Any],
                                        plan_raw: bytes, planned_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rehash the runner's strict bootstrap attestation and every process receipt."""
    attestation_path = output_root / "runtime_determinism_attestation.json"
    attestation_obj, attestation_raw = read_json(attestation_path, "strict determinism attestation", None,
                                                16 * 1024 * 1024)
    execution = plan.get("execution", {})
    runtime_ref = execution.get("samehost_runtime_binding", {})
    strict_ref = execution.get("strict_renderer_determinism", {})
    if (not isinstance(runtime_ref, dict) or not isinstance(runtime_ref.get("raw_sha256"), str)
            or len(runtime_ref["raw_sha256"]) != 64):
        raise ValueError("frozen plan does not bind its same-host runtime-binding raw SHA")
    if (attestation_obj.get("schema") != "cross_tfg_v3_samehost_renderer_determinism_attestation_v1"
            or attestation_obj.get("status") != "PASS"
            or attestation_obj.get("attestation_sha256") != payload_sha(attestation_obj, "attestation_sha256")
            or attestation_obj.get("gpu_uuid") != runtime_ref.get("gpu_uuid")
            or attestation_obj.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or attestation_obj.get("runtime_binding_raw_sha256") != runtime_ref.get("raw_sha256")
            or attestation_obj.get("strict_bootstrap_source_sha256") != strict_ref.get("bootstrap_source", {}).get("raw_sha256")
            or attestation_obj.get("policy_values") != strict_ref.get("policy_values")
            or attestation_obj.get("row_count") != len(planned_rows)
            or attestation_obj.get("native_renderer_row_receipt_count") != len(planned_rows)
            or attestation_obj.get("confirm_audio_read") is not False
            or attestation_obj.get("confirm_data_opened") is not False):
        raise ValueError("strict renderer attestation is invalid or bound to other plan/runtime inputs")
    planned_by_id = {row["row_id"]: row for row in planned_rows}
    native_by_row = attestation_obj.get("native_renderer_receipts_by_row")
    descriptors = attestation_obj.get("process_receipts")
    if (not isinstance(native_by_row, dict) or set(native_by_row) != set(planned_by_id)
            or not isinstance(descriptors, list) or len(descriptors) != attestation_obj.get("process_receipt_count")
            or len(descriptors) < len(planned_rows)):
        raise ValueError("strict renderer attestation does not preserve one native renderer receipt per plan row")
    by_path: dict[str, dict[str, Any]] = {}
    archived: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("strict renderer process receipt descriptor is malformed")
        path_text = descriptor.get("path")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute() or path_text in by_path:
            raise ValueError("strict renderer process receipt path is absent, relative, or duplicated")
        path = Path(path_text)
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError("strict renderer process receipt escapes the Stage-A output root") from exc
        _, raw = safe_regular(path, "strict renderer process receipt", 256 * 1024)
        record = json.loads(raw.decode("utf-8"))
        row_id = descriptor.get("row_id")
        planned = planned_by_id.get(row_id)
        if (planned is None or sha_bytes(raw) != descriptor.get("sha256")
                or len(raw) != descriptor.get("bytes")
                or record.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
                or record.get("status") != "PASS"
                or record.get("receipt_sha256") != payload_sha(record, "receipt_sha256")
                or record.get("receipt_sha256") != descriptor.get("receipt_sha256", record.get("receipt_sha256"))
                or record.get("bootstrap_source_sha256") != attestation_obj.get("strict_bootstrap_source_sha256")
                or record.get("policy") != attestation_obj.get("policy_values")
                or record.get("row_id") != row_id
                or record.get("model_id") != planned.get("model_id")
                or record.get("audio_frontend") != planned.get("audio_frontend")
                or record.get("seed") != planned.get("seed")
                or record.get("pid") != descriptor.get("pid")
                or record.get("entry_matches_backend") is not descriptor.get("entry_matches_backend")):
            raise ValueError(f"strict renderer process receipt changed or differs from row policy: {path.name}")
        descriptor_copy = {**descriptor, "archive_member": f"producer/strict_determinism/receipts/{path.name}"}
        by_path[path_text] = {"descriptor": descriptor_copy, "raw": raw, "record": record}
        archived.append(descriptor_copy)
    for row_id, native in native_by_row.items():
        if not isinstance(native, dict):
            raise ValueError(f"native renderer row receipt is malformed: {row_id}")
        matches = [item for item in by_path.values()
                   if item["descriptor"]["path"] == native.get("path")
                   and item["descriptor"]["sha256"] == native.get("sha256")
                   and item["descriptor"]["pid"] == native.get("pid")
                   and item["descriptor"].get("entry_matches_backend") is True
                   and item["descriptor"]["row_id"] == row_id]
        if len(matches) != 1 or matches[0]["record"].get("entry_path") != native.get("entry_path"):
            raise ValueError(f"strict attestation lacks one exact native renderer receipt: {row_id}")
    return {
        "attestation_raw": attestation_raw,
        "attestation": attestation_obj,
        "receipts": by_path,
        "archive_descriptor": {
            "attestation_member": "producer/strict_determinism/attestation.json",
            "attestation_bytes": len(attestation_raw),
            "attestation_raw_sha256": sha_bytes(attestation_raw),
            "attestation_sha256": attestation_obj["attestation_sha256"],
            "receipt_count": len(archived),
            "native_renderer_row_receipt_count": len(planned_rows),
            "receipts": archived,
        },
    }

def safe_tar_path(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)):
        raise ValueError(f"unsafe archive member path: {name!r}")


class BoundedWriter:
    """Reject a compressed archive as soon as its explicit byte cap is crossed."""
    def __init__(self, raw: Any, cap: int):
        self.raw = raw
        self.cap = cap
        self.written = 0

    def write(self, data: bytes) -> int:
        if self.written + len(data) > self.cap:
            raise ValueError("compressed media archive exceeded its caller-supplied byte cap")
        count = self.raw.write(data)
        self.written += count
        return count

    def tell(self) -> int:
        return self.written

    def flush(self) -> None:
        self.raw.flush()


def add_bytes(archive: tarfile.TarFile, name: str, raw: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(raw), 0o600, 0
    archive.addfile(info, io.BytesIO(raw))


def add_file(archive: tarfile.TarFile, name: str, path: Path,
             expected_sha: str, expected_bytes: int, member_cap: int) -> None:
    size, digest = hash_file(path, expected_sha, member_cap, f"media member {name}")
    if size != expected_bytes:
        raise ValueError(f"media member byte count changed: {name}")
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o600, 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)
    if hash_file(path, digest, member_cap, f"media member after packaging {name}")[0] != size:
        raise ValueError(f"media member changed while being packaged: {name}")


def _source_rows(role: str, plan: dict[str, Any], planned_rows: list[dict[str, Any]]) -> None:
    expected_methods = CONTROLS_METHODS if role == "samehost_controls" else (CANDIDATE_METHOD,)
    expected_count = 360 if role == "samehost_controls" else 180
    if len(planned_rows) != expected_count or len({row.get("row_id") for row in planned_rows}) != expected_count:
        raise ValueError(f"{role} plan does not preserve its exact {expected_count}-row denominator")
    if tuple(model.get("model_id") for model in plan.get("models", [])) != MODELS:
        raise ValueError("raw-media plan uses a different qualified backend roster")
    if tuple(method.get("method_id") for method in plan.get("methods", [])) != expected_methods:
        raise ValueError(f"raw-media plan method roster differs for {role}")
    contexts: set[tuple[str, str, int]] = set()
    counts: dict[tuple[str, str], int] = {}
    contexts_by_cell: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    for row in planned_rows:
        if row.get("split") != "dev" or row.get("model_id") not in MODELS or row.get("method_id") not in expected_methods:
            raise ValueError(f"raw-media plan row is outside the frozen {role} scope")
        context = (str(row.get("sample_id")), str(row.get("speaker_id")), int(row.get("seed")))
        contexts.add(context)
        cell = (str(row["model_id"]), str(row["method_id"]))
        counts[cell] = counts.get(cell, 0) + 1
        contexts_by_cell.setdefault(cell, set()).add(context)
    expected_cells = {(model, method) for model in MODELS for method in expected_methods}
    samples = {sample for sample, _, _ in contexts}
    speakers = {speaker for _, speaker, _ in contexts}
    sample_seed_sets = {sample: {seed for candidate, _, seed in contexts if candidate == sample} for sample in samples}
    if (len(contexts) != 60 or len({speaker for _, speaker, _ in contexts}) != 10
            or {seed for _, _, seed in contexts} != set(SEEDS)
            or len(samples) != 20 or len(speakers) != 10
            or any(len({speaker for candidate, speaker, _ in contexts if candidate == sample}) != 1 for sample in samples)
            or any(sample_seed_sets[sample] != set(SEEDS) for sample in samples)
            or set(counts) != expected_cells or any(counts[cell] != 60 for cell in expected_cells)
            or any(contexts_by_cell[cell] != contexts for cell in expected_cells)):
        raise ValueError("raw-media plan is not the exact 20 clips × 3 seeds context roster")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    plan, plan_raw = read_json(args.plan, "frozen plan", args.plan_raw_sha256, 32 * 1024 * 1024)
    plan_manifest, plan_manifest_raw = read_json(args.plan_manifest, "plan manifest",
                                                  args.plan_manifest_raw_sha256, 4 * 1024 * 1024)
    _, planned_raw = safe_regular(args.planned_rows, "planned rows", 32 * 1024 * 1024)
    if (plan.get("status") != "FROZEN" or plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan_manifest.get("manifest_sha256") != payload_sha(plan_manifest, "manifest_sha256")
            or plan_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or plan_manifest.get("planned_rows_sha256") != sha_bytes(planned_raw)):
        raise ValueError("frozen plan/manifest self and raw bindings mismatch")
    expected_manifest_schema = ("cross_tfg_v3_samehost_dev360_controls_plan_v1" if args.role == "samehost_controls"
                                else "cross_tfg_v3_candidate_only_dev_plan_v1")
    if plan_manifest.get("schema") != expected_manifest_schema:
        raise ValueError("plan manifest schema does not match the selected raw-media collection role")
    planned_rows = [json.loads(line.decode("utf-8")) for line in planned_raw.splitlines() if line.strip()]
    if planned_rows != plan.get("rows") or b"".join(canonical(row) + b"\n" for row in planned_rows) != planned_raw:
        raise ValueError("planned-row ledger differs from canonical frozen plan rows")
    _source_rows(args.role, plan, planned_rows)
    strict_evidence = collect_strict_determinism_evidence(output_root=Path(os.path.abspath(os.fspath(args.stagea_output_root))),
                                                          plan=plan, plan_raw=plan_raw, planned_rows=planned_rows)
    expected = {row["row_id"]: row for row in planned_rows}

    output_root = Path(os.path.abspath(os.fspath(args.stagea_output_root)))
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("Stage-A output root must be an existing nonsymlink directory")
    final_path, partial_path = output_root / "results.jsonl", output_root / "results.partial.jsonl"
    result_path = final_path if final_path.is_file() else partial_path
    results: list[dict[str, Any]] = []
    results_raw = b""
    if result_path.is_file():
        _, results_raw = safe_regular(result_path, "Stage-A result ledger", 128 * 1024 * 1024)
        results = [json.loads(line.decode("utf-8")) for line in results_raw.splitlines() if line.strip()]
    result_manifest_path = output_root / "result_manifest.json"
    result_manifest: dict[str, Any] | None = None
    result_manifest_raw = b""
    if result_manifest_path.is_file():
        result_manifest, result_manifest_raw = read_json(result_manifest_path, "Stage-A result manifest",
                                                          None, 4 * 1024 * 1024)
        if (result_manifest.get("run_id") != plan.get("run_id")
                or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
                or result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
                or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")
                or (results_raw and result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw))):
            raise ValueError("Stage-A result manifest does not bind the exact frozen producer inputs")
        if result_path == final_path and (result_manifest.get("planned_rows") != len(planned_rows)
                or result_manifest.get("result_rows") != len(results)):
            raise ValueError("final Stage-A ledger counts differ from its bound result manifest")
    elif result_path == final_path:
        raise ValueError("final Stage-A ledger has no result manifest")

    observed: dict[str, dict[str, Any]] = {}
    identity_fields = ("split", "sample_id", "speaker_id", "model_id", "audio_frontend",
                       "model_role", "face_id", "seed", "method_id")
    for index, row in enumerate(results, 1):
        row_id = row.get("row_id")
        if row_id not in expected or row_id in observed:
            raise ValueError(f"Stage-A ledger has an unexpected/duplicate row at line {index}")
        planned = expected[row_id]
        for field in identity_fields:
            if row.get(field) != planned.get(field):
                raise ValueError(f"Stage-A row identity differs from its plan at {field}: {row_id}")
        pcm = planned.get("natural_pcm", {})
        if (row.get("natural_pcm_sha256") not in (None, pcm.get("pcm_sha256"))
                or row.get("score_audio_sha256") not in (None, pcm.get("pcm_sha256"))
                or row.get("source_face_sha256") not in (None, planned.get("source_face", {}).get("file_sha256"))
                or row.get("source_video_sha256") not in (None, planned.get("source_video", {}).get("file_sha256"))):
            raise ValueError(f"Stage-A row PCM/source identity differs from plan: {row_id}")
        if row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"}:
            raise ValueError(f"Stage-A result has an unknown status: {row_id}")
        observed[row_id] = row

    max_member = args.max_unpacked_bytes
    asset_map: dict[tuple[str, str], dict[str, Any]] = {}
    asset_paths: dict[tuple[str, str], Path] = {}
    row_media: list[dict[str, Any]] = []
    total_unpacked = 0

    def register(row_id: str, role: str, path_value: Any, expected_sha: Any,
                 expected_bytes: Any = None, *, must_be_under_output: bool) -> dict[str, Any]:
        nonlocal total_unpacked
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"PASS row {row_id} lacks {role} path")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"PASS row {row_id} lacks {role} SHA-256")
        path = Path(path_value)
        if not path.is_absolute():
            path = output_root / path
        path = Path(os.path.abspath(os.fspath(path)))
        if must_be_under_output:
            try:
                path.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"PASS row {role} escapes Stage-A output: {row_id}") from exc
        size, digest = hash_file(path, expected_sha, max_member, f"{role} for {row_id}")
        if expected_bytes is not None and (isinstance(expected_bytes, bool) or expected_bytes != size):
            raise ValueError(f"PASS row {role} byte count differs from its frozen reference: {row_id}")
        key = (role, digest)
        asset = asset_map.get(key)
        if asset is None:
            suffix = path.suffix.lower()
            if suffix not in {".mkv", ".mp4", ".wav", ".json"}:
                suffix = ".bin"
            member = f"media/{role}/{digest}{suffix}"
            safe_tar_path(member)
            asset = {"role": role, "path": member, "bytes": size, "sha256": digest}
            asset_map[key] = asset
            asset_paths[key] = path
            total_unpacked += size
            if total_unpacked > args.max_unpacked_bytes:
                raise ValueError("unique raw-media members exceed the caller-supplied uncompressed byte cap")
        elif asset["bytes"] != size:
            raise ValueError(f"identical {role} digest has conflicting sizes")
        return {"path": asset["path"], "bytes": size, "sha256": digest}

    for planned in planned_rows:
        row_id = planned["row_id"]
        result = observed.get(row_id)
        if result is None or result.get("status") != "PASS":
            row_media.append({"row_id": row_id, "status": result.get("status") if result else "MISSING",
                              "assets": None})
            continue
        render = result.get("render")
        if not isinstance(render, dict) or render.get("status") != "PASS":
            raise ValueError(f"PASS Stage-A row lacks a PASS render record: {row_id}")
        pcm = planned.get("natural_pcm", {})
        audio_file = register(row_id, "source_pcm", pcm.get("path"), pcm.get("file_sha256"),
                              must_be_under_output=False)
        silent = register(row_id, "silent_video", render.get("silent_video_path"),
                          render.get("silent_video_sha256"), must_be_under_output=True)
        muxed = register(row_id, "muxed_video", render.get("muxed_video_path"),
                         render.get("muxed_video_sha256"), must_be_under_output=True)
        timestamps = register(row_id, "timestamps", render.get("timestamps_path"),
                              render.get("timestamps_sha256"), must_be_under_output=True)
        row_media.append({
            "row_id": row_id, "status": "PASS",
            "source_pcm_sha256": pcm.get("pcm_sha256"), "source_pcm_file_sha256": audio_file["sha256"],
            "silent_video": silent, "muxed_video": muxed, "timestamps": timestamps,
            "fps_num": render.get("fps_num"), "fps_den": render.get("fps_den"),
            "frame_count": render.get("frame_count"),
            "video_stream_sha256": render.get("video_stream_sha256"),
            "mux_audio_readback": result.get("mux_audio_readback", {}),
        })

    if len(row_media) != len(planned_rows):
        raise ValueError("raw-media row map does not preserve the frozen denominator")
    status_counts = {status: sum(item["status"] == status for item in row_media)
                     for status in ("PASS", "FAILED", "INCOMPLETE", "MISSING")}
    media_pass = status_counts["PASS"] == len(planned_rows)
    manifest: dict[str, Any] = {
        "schema": ARCHIVE_SCHEMA, "role": ARCHIVE_ROLE,
        "status": "PASS" if media_pass else "INCOMPLETE",
        "join_eligible_for_raw_visual_and_pcm_inputs": media_pass,
        "run_role": args.role, "run_id": plan.get("run_id"),
        "source_plan_raw_sha256": sha_bytes(plan_raw),
        "source_plan_payload_sha256": plan.get("plan_sha256"),
        "source_plan_manifest_raw_sha256": sha_bytes(plan_manifest_raw),
        "source_planned_rows_raw_sha256": sha_bytes(planned_raw),
        "source_result_manifest_raw_sha256": sha_bytes(result_manifest_raw) if result_manifest_raw else None,
        "source_results_raw_sha256": sha_bytes(results_raw) if results_raw else None,
        "source_results_member": result_path.name if results_raw else None,
        "planned_rows": len(planned_rows), "observed_result_rows": len(results),
        "row_status_counts": status_counts, "row_media": row_media,
        "assets": [asset_map[key] for key in sorted(asset_map)],
        "strict_determinism": strict_evidence["archive_descriptor"],
        "asset_count": len(asset_map), "asset_bytes_uncompressed": total_unpacked,
        "media_roles": ["silent_video", "muxed_video", "timestamps", "source_pcm"],
        "bounds": {"max_unpacked_bytes": args.max_unpacked_bytes,
                   "max_archive_bytes": args.max_archive_bytes,
                   "reserve_free_bytes": args.reserve_free_bytes},
        "collector": {"source_path": str(Path(__file__).resolve()),
                      "source_sha256": sha_bytes(Path(__file__).read_bytes()),
                      "created_unix": time.time(),
                      "raw_media_bytes_hashed": True,
                      "video_decoding_or_visual_metrics_computed": False,
                      "audio_decoding_or_pcm_conversion_performed": False,
                      "confirm_data_opened": False},
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_raw = pretty(manifest)

    strict_extra_bytes = len(strict_evidence["attestation_raw"]) + sum(
        len(item["raw"]) for item in strict_evidence["receipts"].values())
    metadata_bytes = (len(manifest_raw) + len(plan_raw) + len(planned_raw) + len(plan_manifest_raw)
                      + len(result_manifest_raw) + len(results_raw) + strict_extra_bytes)
    if total_unpacked + metadata_bytes > args.max_unpacked_bytes:
        raise ValueError("raw media plus producer metadata exceed the caller-supplied uncompressed byte cap")

    output_archive = Path(os.path.abspath(os.fspath(args.output_archive)))
    if output_archive.exists() or output_archive.is_symlink():
        raise FileExistsError(f"refusing existing raw-media archive: {output_archive}")
    cursor = Path(output_archive.anchor)
    for part in output_archive.parts[1:-1]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"media archive target traverses a symlink/junction: {cursor}")
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    if output_archive.parent.is_symlink() or getattr(output_archive.parent, "is_junction", lambda: False)():
        raise ValueError("media archive target parent must be a real directory")
    partial = output_archive.with_name(output_archive.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"refusing existing partial raw-media archive: {partial}")
    free_before = shutil.disk_usage(output_archive.parent).free
    if free_before < args.max_archive_bytes + args.reserve_free_bytes:
        raise ValueError("raw-media archive target lacks its caller-supplied archive cap plus reserve")
    try:
        with partial.open("xb") as raw_stream:
            bounded = BoundedWriter(raw_stream, args.max_archive_bytes)
            with tarfile.open(fileobj=bounded, mode="w:gz", compresslevel=6) as archive:
                add_bytes(archive, "raw_media_archive_manifest.json", manifest_raw)
                add_bytes(archive, "producer/plan.json", plan_raw)
                add_bytes(archive, "producer/planned_rows.jsonl", planned_raw)
                add_bytes(archive, "producer/plan_manifest.json", plan_manifest_raw)
                add_bytes(archive, "producer/strict_determinism/attestation.json", strict_evidence["attestation_raw"])
                for item in strict_evidence["receipts"].values():
                    add_bytes(archive, item["descriptor"]["archive_member"], item["raw"])
                if result_manifest_raw:
                    add_bytes(archive, "producer/result_manifest.json", result_manifest_raw)
                if results_raw:
                    add_bytes(archive, f"producer/{result_path.name}", results_raw)
                for key in sorted(asset_paths):
                    asset = asset_map[key]
                    add_file(archive, asset["path"], asset_paths[key], asset["sha256"],
                             int(asset["bytes"]), max_member)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        archive_size = partial.stat().st_size
        if archive_size > args.max_archive_bytes:
            raise ValueError("raw-media archive exceeded its caller-supplied compressed byte cap")
        os.replace(partial, output_archive)
    except BaseException:
        if partial.exists() and not partial.is_symlink():
            partial.unlink()
        raise

    archive_sha = hashlib.sha256()
    with output_archive.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            archive_sha.update(block)
    receipt: dict[str, Any] = {
        "schema": "cross_tfg_v3_stagea_raw_media_archive_receipt_v2",
        "status": manifest["status"],
        "join_eligible_for_raw_visual_and_pcm_inputs": manifest["join_eligible_for_raw_visual_and_pcm_inputs"],
        "archive_path": str(output_archive), "archive_bytes": output_archive.stat().st_size,
        "archive_raw_sha256": archive_sha.hexdigest(),
        "manifest_raw_sha256": sha_bytes(manifest_raw), "manifest_sha256": manifest["manifest_sha256"],
        "planned_rows": len(planned_rows), "row_status_counts": status_counts,
        "asset_count": len(asset_map), "asset_bytes_uncompressed": total_unpacked,
        "target_free_bytes_before": free_before,
        "target_free_bytes_after": shutil.disk_usage(output_archive.parent).free,
    }
    receipt["receipt_sha256"] = payload_sha(receipt, "receipt_sha256")
    receipt_path = output_archive.with_suffix(output_archive.suffix + ".receipt.json")
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt)); stream.flush(); os.fsync(stream.fileno())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("samehost_controls", "step120_candidate"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--stagea-output-root", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--max-unpacked-bytes", type=int, required=True)
    parser.add_argument("--max-archive-bytes", type=int, required=True)
    parser.add_argument("--reserve-free-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if any(isinstance(value, bool) or value <= 0 for value in
           (args.max_unpacked_bytes, args.max_archive_bytes)) or args.reserve_free_bytes < 0:
        raise ValueError("raw-media byte caps must be positive and reserve must be nonnegative")
    receipt = collect(args)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
