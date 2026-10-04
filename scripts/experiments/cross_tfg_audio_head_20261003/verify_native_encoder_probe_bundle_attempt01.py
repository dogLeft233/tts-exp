#!/usr/bin/env python3
"""Safely reassemble, verify, and optionally extract the probe readback TAR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, BinaryIO


FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
PROBE_PLAN_SHA256 = "f205263a89cc135dad52a7db5b11efca927aab457ee640c694bd050863af811f"
RUNNER_SHA256 = "4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c"
FRONTENDS = {
    "wav2lip": "wav2lip_gan_bac9a81e",
    "muse": "musetalk_fixed_portrait_attempt16",
    "ditto": "ditto_hubert_fixed_portrait_attempt18",
}
CONDITIONS = (
    "natural",
    "global_minus_6db",
    "global_plus_6db",
    "fixed_rms_0p0376838173",
    "local_r090_accent015_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent_0.30_tilt0",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        raise RuntimeError(f"unsafe archive path: {value!r}")
    return path.as_posix()


def stream_hash(stream: BinaryIO) -> tuple[int, str]:
    h = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        count += len(block)
        h.update(block)
    return count, h.hexdigest()


def read_json_member(tf: tarfile.TarFile, name: str) -> Any:
    member = tf.getmember(name)
    stream = tf.extractfile(member)
    if stream is None:
        raise RuntimeError(f"cannot read TAR JSON member {name}")
    with stream:
        return json.load(stream)


def verify_summary_rosters(
    tf: tarfile.TarFile,
    manifest: dict[str, Any],
    fit_plan: dict[str, Any],
    probe_plan: dict[str, Any],
) -> None:
    sources = {r["sample_id"]: r for r in probe_plan["input_rows"]}
    expected_pairs = {(sid, cond) for sid in sources for cond in CONDITIONS}
    members = {m["path"]: m for m in manifest["members"]}
    for frontend, model_id in FRONTENDS.items():
        summary_path = f"{frontend}/summary.json"
        summary = read_json_member(tf, summary_path)
        if summary.get("status") != "COMPLETE" or summary.get("frontend") != frontend or summary.get("model_id") != model_id:
            raise RuntimeError(f"bad summary identity/status for {frontend}")
        if summary.get("plan", {}).get("sha256") != FIT_PLAN_SHA256:
            raise RuntimeError(f"summary plan binding differs for {frontend}")
        if summary.get("row_count") != 56 or len(summary.get("rows", [])) != 56:
            raise RuntimeError(f"wrong row count for {frontend}")
        rows_seen: set[tuple[str, str]] = set()
        for row in summary["rows"]:
            key = (row.get("sample_id"), row.get("condition"))
            if row.get("status") != "PASS" or key not in expected_pairs or key in rows_seen:
                raise RuntimeError(f"duplicate, missing, or non-PASS summary row for {frontend}: {key}")
            rows_seen.add(key)
            expected_source = sources[key[0]]
            source_pcm = row.get("source_pcm", {})
            natural = expected_source["natural_pcm"]
            if (
                row.get("speaker_id") != expected_source["speaker_id"]
                or source_pcm.get("wav_sha256") != natural.get("file_sha256")
                or source_pcm.get("pcm_float32_sha256") != natural.get("pcm_sha256")
                or source_pcm.get("sample_count") != natural.get("sample_count")
                or source_pcm.get("sample_rate_hz") != natural.get("sample_rate_hz")
                or source_pcm.get("encoding") != "PCM_16"
            ):
                raise RuntimeError(f"natural source binding differs for {frontend}/{key}")
            array = row.get("feature_array", {})
            array_name = f"{frontend}/features/{key[0]}__{key[1]}.npz"
            member = members.get(array_name)
            if member is None or array.get("sha256") != member.get("sha256") or array.get("bytes") != member.get("bytes"):
                raise RuntimeError(f"summary feature array seal differs for {array_name}")
        if rows_seen != expected_pairs:
            raise RuntimeError(f"incomplete condition roster for {frontend}")
        scope = summary.get("scope", {})
        if (
            scope.get("split") != "historical_fit_only"
            or scope.get("source_clip_count") != 8
            or scope.get("completed_source_condition_units") != 56
            or scope.get("development_audio_opened") is not False
            or scope.get("confirmation_audio_opened") is not False
            or scope.get("generator_model_forward") is not False
            or scope.get("video_created_or_read") is not False
            or scope.get("syncnet_forward_or_scores_read") is not False
        ):
            raise RuntimeError(f"scope violation or incomplete scope for {frontend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", required=True, help="folder with bundle_receipt.json and bundle.partNNN")
    parser.add_argument("--fit-plan", required=True)
    parser.add_argument("--probe-plan", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--tar-out", required=True, help="fresh local TAR destination")
    parser.add_argument("--extract-to", required=True, help="fresh local extraction directory")
    args = parser.parse_args()
    chunk_dir = Path(args.chunk_dir)
    receipt_path = chunk_dir / "bundle_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("missing/nonregular bundle receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "cross_tfg_native_audio_encoder_probe_archive_receipt_v1" or receipt.get("status") != "COMPLETE":
        raise RuntimeError("bundle receipt is not COMPLETE")
    for path_str, expected in ((args.fit_plan, FIT_PLAN_SHA256), (args.probe_plan, PROBE_PLAN_SHA256), (args.runner, RUNNER_SHA256)):
        path = Path(path_str)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"local provenance source mismatch: {path}")
    fit_plan = json.loads(Path(args.fit_plan).read_text(encoding="utf-8"))
    probe_plan = json.loads(Path(args.probe_plan).read_text(encoding="utf-8"))
    if fit_plan.get("run_id") != "cross_tfg_shared_frequency_profiles_fit_20261003_attempt03" or len(fit_plan.get("rows", [])) != 360:
        raise RuntimeError("local frozen fit plan identity mismatch")
    if probe_plan.get("schema") != "cross_tfg_native_audio_encoder_probe_plan_v1" or len(probe_plan.get("input_rows", [])) != 8:
        raise RuntimeError("local probe plan identity mismatch")
    parts = receipt.get("chunks")
    if not isinstance(parts, list) or len(parts) != receipt.get("tar_chunk_count") or not parts:
        raise RuntimeError("chunk receipt count mismatch")
    expected_part_names = [f"bundle.part{i:03d}" for i in range(len(parts))]
    observed_part_names = sorted(p.name for p in chunk_dir.glob("bundle.part*"))
    if observed_part_names != expected_part_names:
        raise RuntimeError("downloaded chunk filenames do not exactly match receipt")
    tar_path = Path(args.tar_out)
    extract_root = Path(args.extract_to)
    if tar_path.exists() or extract_root.exists() or not tar_path.parent.is_dir() or not extract_root.parent.is_dir():
        raise RuntimeError("TAR/extraction destination must be fresh with existing parents")
    whole = hashlib.sha256()
    total = 0
    with tar_path.open("xb") as out:
        for i, entry in enumerate(parts):
            expected_name = expected_part_names[i]
            if entry.get("name") != expected_name or int(entry.get("bytes", -1)) < 0:
                raise RuntimeError(f"invalid chunk entry {i}")
            chunk = chunk_dir / expected_name
            if chunk.is_symlink() or not chunk.is_file() or chunk.stat().st_size != entry["bytes"] or sha256_file(chunk) != entry["sha256"]:
                raise RuntimeError(f"chunk seal mismatch: {expected_name}")
            with chunk.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    out.write(block)
                    whole.update(block)
                    total += len(block)
        out.flush()
    if total != receipt.get("tar_bytes") or whole.hexdigest() != receipt.get("tar_sha256"):
        raise RuntimeError("reassembled TAR size/SHA mismatch")

    extract_root.mkdir(parents=False, exist_ok=False)
    with tarfile.open(tar_path, "r:") as tf:
        members = tf.getmembers()
        names: list[str] = []
        for member in members:
            name = safe_name(member.name)
            if name != member.name or not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"nonregular or unsafe TAR member: {member.name}")
            names.append(name)
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate TAR member names")
        if "archive_manifest.json" not in names:
            raise RuntimeError("archive manifest missing")
        manifest_bytes = tf.extractfile(tf.getmember("archive_manifest.json")).read()
        if hashlib.sha256(manifest_bytes).hexdigest() != receipt.get("archive_manifest_sha256"):
            raise RuntimeError("archive manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "cross_tfg_native_audio_encoder_probe_archive_members_v1" or manifest.get("status") != "COMPLETE":
            raise RuntimeError("archive member manifest invalid")
        bindings = manifest.get("bindings", {})
        if bindings.get("fit_plan", {}).get("sha256") != FIT_PLAN_SHA256 or bindings.get("probe_plan", {}).get("sha256") != PROBE_PLAN_SHA256:
            raise RuntimeError("archive manifest provenance mismatch")
        if bindings.get("runner", {}).get("sha256") != RUNNER_SHA256:
            raise RuntimeError("archive manifest runner pin mismatch")
        expected_members = {entry["path"]: entry for entry in manifest.get("members", [])}
        if len(expected_members) != 174 or set(names) != set(expected_members) | {"archive_manifest.json"}:
            raise RuntimeError("archive member roster/count mismatch")
        for name, entry in expected_members.items():
            member = tf.getmember(name)
            stream = tf.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read TAR member: {name}")
            with stream:
                nbytes, digest = stream_hash(stream)
            if nbytes != entry.get("bytes") or digest != entry.get("sha256"):
                raise RuntimeError(f"TAR member hash mismatch: {name}")
        verify_summary_rosters(tf, manifest, fit_plan, probe_plan)
        for member in members:
            if member.name == "archive_manifest.json":
                continue
            name = safe_name(member.name)
            destination = extract_root.joinpath(*PurePosixPath(name).parts)
            try:
                destination.resolve().relative_to(extract_root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"extraction path escaped root: {name}") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot extract TAR member: {name}")
            with stream, destination.open("xb") as out:
                shutil_copyfileobj(stream, out)
        (extract_root / "archive_manifest.json").write_bytes(manifest_bytes)
    print(json.dumps({
        "status": "VERIFIED_AND_EXTRACTED",
        "tar_bytes": total,
        "tar_sha256": whole.hexdigest(),
        "archive_member_count": 175,
        "feature_npz_count": 168,
        "frontend_rows": 56,
        "extract_root": str(extract_root),
    }, sort_keys=True))


def shutil_copyfileobj(source: BinaryIO, target: BinaryIO) -> None:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        target.write(block)


if __name__ == "__main__":
    main()
