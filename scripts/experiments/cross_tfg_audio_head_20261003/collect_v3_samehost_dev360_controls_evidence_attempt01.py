#!/usr/bin/env python3
"""Collect a fresh same-host Dev360 controls ledger and raw-score archive.

This is a metadata/score-array collector. It never opens generated media,
natural audio, Confirm data, model weights, or runs a model. Missing Stage-A
rows are represented as failed denominator rows. The archive retains only the
three sliced plan/ledger packages, their manifests, and content-addressed raw
SyncNet score-array NPZ members; MP4, WAV, JPEG, crop, and tracking artifacts
are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
EXPECTED_METHODS = ("identity", "rms_fixed")
EXPECTED_SEEDS = (20261003, 20261004, 20261005)
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_NPZ_BYTES = 64 * 1024 * 1024
MAX_TOTAL_NPZ_BYTES = 2 * 1024 * 1024 * 1024
ARCHIVE_SCHEMA = "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
ARCHIVE_ROLE = "read_only_raw_score_and_samehost_identity_rms_control_evidence"


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


def safe_regular(path: Path, label: str, limit: int) -> tuple[Path, bytes]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return lexical.resolve(strict=True), raw


def load_json(path: Path, label: str, expected_sha: str | None = None,
              cap: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    _, raw = safe_regular(path, label, cap)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def load_jsonl(path: Path, label: str, cap: int = MAX_JSON_BYTES) -> tuple[list[dict[str, Any]], bytes]:
    _, raw = safe_regular(path, label, cap)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_no} is not a JSON object")
        rows.append(value)
    return rows, raw


def write_tar_bytes(archive: tarfile.TarFile, name: str, raw: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    info.mode = 0o600
    info.mtime = 0
    import io
    archive.addfile(info, io.BytesIO(raw))


def file_sha_and_size(path: Path, expected_sha: str, label: str) -> tuple[int, str]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > MAX_NPZ_BYTES:
        raise ValueError(f"{label} is not a bounded nonempty regular file")
    digest = hashlib.sha256()
    with lexical.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = lexical.stat(follow_symlinks=False)
    actual = digest.hexdigest()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or actual != expected_sha.lower()):
        raise ValueError(f"{label} changed during collection or differs from its ledger SHA")
    return before.st_size, actual


def validate_frozen_plan(plan: dict[str, Any], plan_raw: bytes, rows_raw: bytes,
                         plan_manifest: dict[str, Any], manifest_raw: bytes) -> list[dict[str, Any]]:
    if (plan.get("status") != "FROZEN" or plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")):
        raise ValueError("controls plan status/schema/self-seal mismatch")
    if (plan_manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_plan_v1"
            or plan_manifest.get("status") != "FROZEN"
            or plan_manifest.get("manifest_sha256") != payload_sha(plan_manifest, "manifest_sha256")
            or plan_manifest.get("plan_sha256") != plan.get("plan_sha256")
            or plan_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or plan_manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or plan_manifest.get("planned_rows") != 360):
        raise ValueError("controls plan manifest does not bind the exact frozen plan and 360-row roster")
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    if rows != plan.get("rows") or len(rows) != 360:
        raise ValueError("planned_rows.jsonl is not the exact ordered 360-row plan")
    if b"".join(canonical(row) + b"\n" for row in rows) != rows_raw:
        raise ValueError("planned_rows.jsonl is not canonical JSONL")
    if tuple(method.get("method_id") for method in plan.get("methods", [])) != EXPECTED_METHODS:
        raise ValueError("control plan is not exactly identity and fixed RMS")
    if tuple(model.get("model_id") for model in plan.get("models", [])) != EXPECTED_MODELS:
        raise ValueError("control plan frontend roster differs from the qualified three-backend roster")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise ValueError("control plan does not explicitly keep Confirm closed")
    if len({row.get("row_id") for row in rows}) != 360:
        raise ValueError("control plan has duplicate row IDs")
    counts = {(model, method): 0 for model in EXPECTED_MODELS for method in EXPECTED_METHODS}
    contexts: set[tuple[str, str, int]] = set()
    for row in rows:
        if row.get("split") != "dev" or row.get("model_id") not in EXPECTED_MODELS or row.get("method_id") not in EXPECTED_METHODS:
            raise ValueError("control row is outside the frozen Dev/model/method scope")
        counts[(row["model_id"], row["method_id"])] += 1
        contexts.add((row["sample_id"], row["speaker_id"], int(row["seed"])))
    if any(value != 60 for value in counts.values()) or len(contexts) != 60:
        raise ValueError("control roster is not 20 clips × 3 seeds × 3 models × 2 methods")
    if {seed for _, _, seed in contexts} != set(EXPECTED_SEEDS):
        raise ValueError("control seed family changed")
    return rows


def incomplete_row(planned: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        **{key: planned.get(key) for key in (
            "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
            "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
        )},
        "schema_version": "cross_tfg_audio_head_stagea_v1",
        "status": "FAILED",
        "failure_reason": reason,
        "natural_pcm_sha256": planned.get("natural_pcm", {}).get("pcm_sha256"),
        "score_audio_sha256": planned.get("natural_pcm", {}).get("pcm_sha256"),
        "source_face_sha256": planned.get("source_face", {}).get("file_sha256"),
        "source_video_sha256": planned.get("source_video", {}).get("file_sha256"),
        "render": {"status": "NOT_STARTED"},
        "mux_audio_readback": {"status": "NOT_STARTED"},
        "score": {"status": "NOT_RUN"},
        "collector_synthesized_missing_result": True,
    }


def tar_add_path(archive: tarfile.TarFile, member: str, path: Path, expected_sha: str,
                 expected_bytes: int) -> None:
    size, actual = file_sha_and_size(path, expected_sha, f"raw score asset {path}")
    if size != expected_bytes:
        raise ValueError(f"raw score asset byte count changed: {path}")
    info = tarfile.TarInfo(member)
    info.size = size
    info.mode = 0o600
    info.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)
    if file_sha_and_size(path, actual, f"raw score asset after packaging {path}")[0] != size:
        raise ValueError(f"raw score asset changed while being archived: {path}")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    plan, plan_raw = load_json(args.plan, "controls plan", args.plan_raw_sha256, 32 * 1024 * 1024)
    plan_manifest, plan_manifest_raw = load_json(args.plan_manifest, "controls plan manifest",
                                                 args.plan_manifest_raw_sha256, 4 * 1024 * 1024)
    _, planned_rows_raw = safe_regular(args.planned_rows, "planned rows", 32 * 1024 * 1024)
    planned = validate_frozen_plan(plan, plan_raw, planned_rows_raw, plan_manifest, plan_manifest_raw)
    expected_by_id = {row["row_id"]: row for row in planned}

    output = Path(os.path.abspath(os.fspath(args.stagea_output_root)))
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Stage-A output root must be an existing nonsymlink directory")
    final_results_path = output / "results.jsonl"
    partial_results_path = output / "results.partial.jsonl"
    results_path = final_results_path if final_results_path.is_file() else partial_results_path
    result_rows: list[dict[str, Any]] = []
    stagea_results_raw = b""
    stagea_results_sha: str | None = None
    if results_path.is_file():
        result_rows, stagea_results_raw = load_jsonl(results_path, "Stage-A result ledger")
        stagea_results_sha = sha_bytes(stagea_results_raw)

    result_manifest: dict[str, Any] | None = None
    result_manifest_raw = b""
    original_manifest_path = output / "result_manifest.json"
    if original_manifest_path.is_file():
        result_manifest, result_manifest_raw = load_json(original_manifest_path, "Stage-A result manifest", cap=4 * 1024 * 1024)
        if (result_manifest.get("run_id") != plan.get("run_id")
                or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
                or result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
                or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
            raise ValueError("Stage-A result manifest plan/self binding mismatch")
        if stagea_results_sha is not None and result_manifest.get("results_jsonl_sha256") != stagea_results_sha:
            raise ValueError("Stage-A result manifest results digest mismatch")
        if results_path == final_results_path and (
                result_manifest.get("planned_rows") != 360
                or result_manifest.get("result_rows") != len(result_rows)):
            raise ValueError("final Stage-A ledger row counts differ from its manifest or fixed 360-row plan")
    elif results_path == final_results_path:
        raise ValueError("final Stage-A ledger has no result manifest")

    observed: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(result_rows, 1):
        row_id = row.get("row_id")
        if row_id not in expected_by_id or row_id in observed:
            raise ValueError(f"Stage-A result ledger has unexpected or duplicate row at line {line_no}")
        expected = expected_by_id[row_id]
        for field in ("split", "sample_id", "speaker_id", "model_id", "audio_frontend", "face_id", "seed", "method_id"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"Stage-A row/plan identity mismatch at {field}: {row_id}")
        source_pcm = expected.get("natural_pcm", {}).get("pcm_sha256")
        if (row.get("natural_pcm_sha256") not in (None, source_pcm)
                or row.get("score_audio_sha256") not in (None, source_pcm)
                or row.get("source_face_sha256") not in (None, expected.get("source_face", {}).get("file_sha256"))
                or row.get("source_video_sha256") not in (None, expected.get("source_video", {}).get("file_sha256"))):
            raise ValueError(f"Stage-A row source/audio identity differs from the frozen original context: {row_id}")
        observed[row_id] = row

    complete_rows: list[dict[str, Any]] = []
    for planned_row in planned:
        row_id = planned_row["row_id"]
        result = observed.get(row_id)
        if result is None:
            result = incomplete_row(planned_row, "collector_synthesized_missing_stagea_row")
        else:
            result = dict(result)
        if result.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise ValueError(f"Stage-A result has an unrecognized status: {row_id}")
        complete_rows.append(result)
    if len(complete_rows) != 360 or {row["row_id"] for row in complete_rows} != set(expected_by_id):
        raise ValueError("collector failed to preserve the exact planned denominator")

    assets: dict[str, dict[str, Any]] = {}
    asset_paths: dict[str, Path] = {}
    total_asset_bytes = 0
    for row in complete_rows:
        if row.get("status") != "PASS":
            continue
        ref = row.get("score", {}).get("raw_matrix_npz", {})
        digest = ref.get("file_sha256")
        raw_path = ref.get("path")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"PASS row lacks a raw score matrix file reference: {row['row_id']}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = output / path
        path = Path(os.path.abspath(os.fspath(path)))
        try:
            path.relative_to(output)
        except ValueError as exc:
            raise ValueError(f"raw score matrix path escapes the Stage-A output root: {row['row_id']}") from exc
        size, actual = file_sha_and_size(path, digest, f"raw score matrix {row['row_id']}")
        asset = assets.get(actual)
        if asset is None:
            assets[actual] = {"role": "raw_score_arrays", "path": f"assets/{actual}.npz",
                              "bytes": size, "sha256": actual}
            asset_paths[actual] = path
            total_asset_bytes += size
            if total_asset_bytes > MAX_TOTAL_NPZ_BYTES:
                raise ValueError("unique raw score NPZ aggregate exceeds the explicit 2 GiB archive cap")
        elif asset["bytes"] != size:
            raise ValueError("same score-array digest has conflicting byte lengths")
        # The archived descriptor names only its archive member. The original
        # host path is omitted; it is not needed to re-read the actual arrays.
        row["score"]["raw_matrix_npz"] = {**ref, "path": f"assets/{actual}.npz"}

    producer_packages: dict[str, dict[str, bytes]] = {}
    model_to_key = dict(zip(EXPECTED_MODELS, ("wav2lip", "muse", "ditto")))
    status_totals = {status: sum(row.get("status") == status for row in complete_rows)
                     for status in ("PASS", "FAILED", "INCOMPLETE")}
    for model_id in EXPECTED_MODELS:
        prefix = model_to_key[model_id]
        model_plan = dict(plan)
        model_plan["run_id"] = f"{plan['run_id']}_{prefix}"
        model_plan["models"] = [model for model in plan["models"] if model.get("model_id") == model_id]
        model_plan["rows"] = [row for row in plan["rows"] if row.get("model_id") == model_id]
        model_plan["generation_repeat_qc"] = []
        model_plan["execution"] = {**plan.get("execution", {}), "planned_rows": 120,
                                    "source_controls_run_id": plan["run_id"],
                                    "source_controls_plan_raw_sha256": sha_bytes(plan_raw),
                                    "frontend_subset": prefix}
        model_plan.pop("plan_sha256", None)
        model_plan["plan_sha256"] = payload_sha(model_plan, "plan_sha256")
        model_plan_raw = pretty(model_plan)

        rows_for_model = [row for row in complete_rows if row.get("model_id") == model_id]
        if len(rows_for_model) != 120:
            raise ValueError(f"producer {prefix} does not preserve its 120-row planned denominator")
        results_raw = b"".join(canonical(row) + b"\n" for row in rows_for_model)
        counts = {status: sum(row.get("status") == status for row in rows_for_model)
                  for status in ("PASS", "FAILED", "INCOMPLETE")}
        producer_manifest = {
            "schema_version": "cross-tfg-audio-eval-v1",
            "run_id": model_plan["run_id"],
            "status": "PASS" if counts["FAILED"] == 0 and counts["INCOMPLETE"] == 0 else "INCOMPLETE",
            "plan_sha256": model_plan["plan_sha256"],
            "plan_raw_sha256": sha_bytes(model_plan_raw),
            "results_jsonl_sha256": sha_bytes(results_raw),
            "planned_rows": 120, "result_rows": 120,
            "status_counts": counts,
            "collector_source_stagea_manifest_raw_sha256": sha_bytes(result_manifest_raw) if result_manifest_raw else None,
            "collector_stagea_ledger_sha256": stagea_results_sha,
            "collector_synthesized_missing_rows": sum(bool(row.get("collector_synthesized_missing_result")) for row in rows_for_model),
        }
        producer_manifest["manifest_sha256"] = payload_sha(producer_manifest, "manifest_sha256")
        producer_packages[prefix] = {
            "plan.json": model_plan_raw,
            "result_manifest.json": pretty(producer_manifest),
            "results.jsonl": results_raw,
        }

    manifest: dict[str, Any] = {
        "schema": ARCHIVE_SCHEMA,
        "role": ARCHIVE_ROLE,
        "status": "PASS" if status_totals["FAILED"] == 0 and status_totals["INCOMPLETE"] == 0 and len(assets) > 0 else "INCOMPLETE",
        "join_eligible": status_totals["FAILED"] == 0 and status_totals["INCOMPLETE"] == 0 and len(assets) > 0,
        "run_id": plan["run_id"],
        "controls_plan_sha256": plan["plan_sha256"],
        "controls_plan_raw_sha256": sha_bytes(plan_raw),
        "controls_plan_manifest_raw_sha256": sha_bytes(plan_manifest_raw),
        "controls_planned_rows_raw_sha256": sha_bytes(planned_rows_raw),
        "stagea_output_root": str(output),
        "stagea_result_manifest_raw_sha256": sha_bytes(result_manifest_raw) if result_manifest_raw else None,
        "stagea_result_ledger_sha256": stagea_results_sha,
        "stagea_result_source": results_path.name if stagea_results_sha else None,
        "row_count": 360,
        "planned_row_count": 360,
        "result_row_count": len(result_rows),
        "status_counts": status_totals,
        "models": list(EXPECTED_MODELS),
        "methods": list(EXPECTED_METHODS),
        "seeds": list(EXPECTED_SEEDS),
        "confirm_speakers_read": False,
        "confirm_audio_opened": False,
        "development_audio_opened_by_collector": False,
        "generated_media_opened_by_collector": False,
        "generated_media_included": False,
        "audio_included": False,
        "raw_score_asset_count": len(assets),
        "raw_score_asset_bytes_uncompressed": total_asset_bytes,
        "assets": [assets[key] for key in sorted(assets)],
        "collector": {"source_path": str(Path(__file__).resolve()),
                       "source_sha256": sha_bytes(Path(__file__).read_bytes()),
                       "collected_unix": time.time(),
                       "missing_rows_preserved_as_failures": True,
                       "media_artifacts_excluded": ["MP4", "WAV", "JPEG", "crop images", "tracking videos"]},
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_raw = pretty(manifest)

    archive_path = Path(os.path.abspath(os.fspath(args.output_archive)))
    if archive_path.exists() or archive_path.is_symlink():
        raise FileExistsError(f"refusing existing evidence archive: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(archive_path.name + ".partial")
    if temp_path.exists() or temp_path.is_symlink():
        raise FileExistsError(f"refusing existing partial archive: {temp_path}")
    with temp_path.open("xb") as raw_stream:
        with tarfile.open(fileobj=raw_stream, mode="w:gz", compresslevel=6) as archive:
            write_tar_bytes(archive, "evidence_archive_manifest.json", manifest_raw)
            for prefix in ("wav2lip", "muse", "ditto"):
                for name in ("plan.json", "result_manifest.json", "results.jsonl"):
                    write_tar_bytes(archive, f"producer/{prefix}/{name}", producer_packages[prefix][name])
            for digest in sorted(asset_paths):
                asset = assets[digest]
                tar_add_path(archive, asset["path"], asset_paths[digest], digest, int(asset["bytes"]))
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    os.replace(temp_path, archive_path)
    archive_sha = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            archive_sha.update(block)
    receipt = {
        "schema": "cross_tfg_v3_samehost_dev360_controls_archive_receipt_v1",
        "status": manifest["status"], "join_eligible": manifest["join_eligible"],
        "archive_path": str(archive_path), "archive_bytes": archive_path.stat().st_size,
        "archive_raw_sha256": archive_sha.hexdigest(),
        "archive_manifest_raw_sha256": sha_bytes(manifest_raw),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": 360, "status_counts": status_totals,
        "raw_score_asset_count": len(assets), "raw_score_asset_bytes_uncompressed": total_asset_bytes,
        "media_included": False,
    }
    receipt["receipt_sha256"] = payload_sha(receipt, "receipt_sha256")
    receipt_path = archive_path.with_suffix(archive_path.suffix + ".receipt.json")
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt)); stream.flush(); os.fsync(stream.fileno())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--stagea-output-root", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = collect(args)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
