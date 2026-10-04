#!/usr/bin/env python3
"""Verify and stream-pack the three native encoder probe outputs.

This is a post-run, read-only collector.  It does not invoke any model,
modify source output, or inspect development/confirmation audio.  NPZs are
checked one at a time so resident memory remains bounded.  The final TAR is
written directly as fixed-size chunks; no second full-size archive copy is
created in tmpfs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import sys
import tarfile
import time
from typing import Any, BinaryIO
import zipfile


FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
PROBE_PLAN_SHA256 = "f205263a89cc135dad52a7db5b11efca927aab457ee640c694bd050863af811f"
RUNNER_SHA256 = "4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c"
FRONTENDS = {
    "wav2lip": "wav2lip_gan_bac9a81e",
    "muse": "musetalk_fixed_portrait_attempt16",
    "ditto": "ditto_hubert_fixed_portrait_attempt18",
}
CHUNK_BYTES = 6_500_000
RESERVE_BYTES = 1_073_741_824
CONDITION_IDS = (
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def check_bound_file(path: Path, expected_sha: str, expected_size: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular source file: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"source byte count mismatch: {path}")
    if digest != expected_sha:
        raise RuntimeError(f"source SHA256 mismatch: {path}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def safe_member_name(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return p.as_posix()


def npy_header(data: BinaryIO) -> tuple[tuple[int, ...], str, bool]:
    if data.read(6) != b"\x93NUMPY":
        raise RuntimeError("bad NPY magic")
    version = data.read(2)
    if len(version) != 2:
        raise RuntimeError("truncated NPY version")
    if version[0] == 1:
        raw = data.read(2)
        if len(raw) != 2:
            raise RuntimeError("truncated NPY v1 header length")
        header_len = struct.unpack("<H", raw)[0]
    elif version[0] in (2, 3):
        raw = data.read(4)
        if len(raw) != 4:
            raise RuntimeError("truncated NPY v2/v3 header length")
        header_len = struct.unpack("<I", raw)[0]
    else:
        raise RuntimeError(f"unsupported NPY version: {version!r}")
    if header_len > 1_000_000:
        raise RuntimeError("unreasonable NPY header size")
    header = data.read(header_len)
    if len(header) != header_len:
        raise RuntimeError("truncated NPY header")
    import ast
    fields = ast.literal_eval(header.decode("latin1").strip())
    shape = tuple(int(x) for x in fields["shape"])
    dtype = str(fields["descr"])
    fortran = bool(fields["fortran_order"])
    if not shape or any(x < 0 for x in shape) or "O" in dtype:
        raise RuntimeError("invalid/object NPY array")
    return shape, dtype, fortran


def validate_npz(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    expected = {
        "features.npy": (tuple(int(x) for x in row["feature_array"]["shape"]), row["feature_array"]["dtype"]),
        "query_index.npy": ((int(row["feature_array"]["shape"][0]),), "int32"),
        "query_timestamp_num.npy": ((int(row["feature_array"]["shape"][0]),), "int32"),
        "query_timestamp_den.npy": ((int(row["feature_array"]["shape"][0]),), "int32"),
    }
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        if names != set(expected):
            raise RuntimeError(f"unexpected NPZ member set for {path.name}: {sorted(names)}")
        for name, (shape_expected, dtype_expected) in expected.items():
            with zf.open(name, "r") as member:
                shape, dtype, fortran = npy_header(member)
                if shape != shape_expected:
                    raise RuntimeError(f"NPY shape mismatch for {path.name}/{name}")
                if np.dtype(dtype).name != dtype_expected:
                    raise RuntimeError(f"NPY dtype mismatch for {path.name}/{name}: {dtype}")
                if fortran:
                    raise RuntimeError(f"unexpected Fortran array for {path.name}/{name}")
        bad_crc = zf.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"NPZ CRC failure in {path.name}/{bad_crc}")
    with np.load(path, allow_pickle=False) as arrays:
        if not np.isfinite(arrays["features"]).all():
            raise RuntimeError(f"non-finite encoder feature values in {path.name}")
        n = int(row["feature_array"]["shape"][0])
        if not np.array_equal(arrays["query_index"], np.arange(n, dtype=np.int32)):
            raise RuntimeError(f"query index mismatch in {path.name}")
        if not np.array_equal(arrays["query_timestamp_num"], np.arange(n, dtype=np.int32)):
            raise RuntimeError(f"query timestamp numerator mismatch in {path.name}")
        if not np.array_equal(arrays["query_timestamp_den"], np.full(n, 25, dtype=np.int32)):
            raise RuntimeError(f"query timestamp denominator mismatch in {path.name}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_expected(fit_plan_path: Path, probe_plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    fit_binding = check_bound_file(fit_plan_path, FIT_PLAN_SHA256)
    probe_binding = check_bound_file(probe_plan_path, PROBE_PLAN_SHA256)
    fit = read_json(fit_plan_path)
    probe = read_json(probe_plan_path)
    if fit.get("run_id") != "cross_tfg_shared_frequency_profiles_fit_20261003_attempt03" or len(fit.get("rows", [])) != 360:
        raise RuntimeError("frozen fit plan identity/row count mismatch")
    if probe.get("schema") != "cross_tfg_native_audio_encoder_probe_plan_v1":
        raise RuntimeError("unexpected probe protocol schema")
    if probe.get("scope", {}).get("units") != 168 or probe.get("scope", {}).get("historical_fit_samples") != 8:
        raise RuntimeError("probe protocol scope mismatch")
    if probe.get("scope", {}).get("development_audio_opened") is not False or probe.get("scope", {}).get("confirmation_audio_opened") is not False:
        raise RuntimeError("probe protocol permits forbidden split access")
    conditions = tuple(x["condition_id"] for x in probe.get("conditions", []))
    if conditions != CONDITION_IDS:
        raise RuntimeError("probe protocol condition roster mismatch")
    rows = probe.get("input_rows", [])
    if len(rows) != 8 or any(r.get("split") != "fit" for r in rows):
        raise RuntimeError("probe protocol must bind exactly eight fit inputs")
    if len({r.get("sample_id") for r in rows}) != 8 or len({r.get("speaker_id") for r in rows}) != 4:
        raise RuntimeError("probe sample/speaker roster mismatch")
    plan_sources: dict[str, dict[str, Any]] = {}
    identity_rows = [
        r for r in fit["rows"]
        if r.get("split") == "fit" and r.get("method_id") == "identity" and r.get("model_id") == FRONTENDS["wav2lip"]
    ]
    if len(identity_rows) != 8:
        raise RuntimeError("expected exactly eight frozen W2L identity rows in fit plan")
    for r in identity_rows:
        sid = r["sample_id"]
        plan_sources[sid] = {"speaker_id": r["speaker_id"], "natural_pcm": r["natural_pcm"]}
    probe_sources: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = r["sample_id"]
        pcm = r["natural_pcm"]
        probe_sources[sid] = {"speaker_id": r["speaker_id"], "natural_pcm": pcm}
    if set(plan_sources) != set(probe_sources):
        raise RuntimeError("probe and fit plan sample IDs differ")
    for sid in plan_sources:
        left, right = plan_sources[sid], probe_sources[sid]
        if left["speaker_id"] != right["speaker_id"] or left["natural_pcm"] != right["natural_pcm"]:
            raise RuntimeError(f"fit-plan PCM/source binding differs for {sid}")
    bindings = {
        "fit_plan": fit_binding,
        "probe_plan": probe_binding,
        "runner": {"sha256": RUNNER_SHA256},
    }
    return fit, probe, probe_sources


def verify_frontend(frontend: str, out_root: Path, expected_model: str, probe_sources: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if out_root.is_symlink() or not out_root.is_dir():
        raise RuntimeError(f"missing/non-directory output root: {out_root}")
    summary_path = out_root / "summary.json"
    receipt_path = out_root / "receipt.json"
    if summary_path.is_symlink() or receipt_path.is_symlink() or not summary_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"frontend output is not terminal: {frontend}")
    summary = read_json(summary_path)
    receipt = read_json(receipt_path)
    if summary.get("schema") != "cross_tfg_native_audio_encoder_probe_v1" or summary.get("status") != "COMPLETE":
        raise RuntimeError(f"summary is not COMPLETE for {frontend}")
    if receipt.get("schema") != "cross_tfg_native_audio_encoder_probe_receipt_v1" or receipt.get("status") != "COMPLETE":
        raise RuntimeError(f"receipt is not COMPLETE for {frontend}")
    if summary.get("frontend") != frontend or summary.get("model_id") != expected_model:
        raise RuntimeError(f"frontend/model identity mismatch for {frontend}")
    if summary.get("plan", {}).get("sha256") != FIT_PLAN_SHA256:
        raise RuntimeError(f"summary does not bind the frozen 360-row plan for {frontend}")
    runtime = summary.get("runtime", {})
    if runtime.get("frontend") != frontend or runtime.get("model_id") != expected_model:
        raise RuntimeError(f"runtime frontend/model identity mismatch for {frontend}")
    if runtime.get("numpy") != "1.26.4" or "3.10.12" not in str(runtime.get("python", "")):
        raise RuntimeError(f"pinned Python/NumPy runtime mismatch for {frontend}")
    scope = summary.get("scope", {})
    required_scope = {
        "split": "historical_fit_only",
        "source_clip_count": 8,
        "planned_source_condition_units": 56,
        "completed_source_condition_units": 56,
        "audio_encoder_forward": True,
        "generator_model_forward": False,
        "video_created_or_read": False,
        "syncnet_forward_or_scores_read": False,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "persistent_transformed_waveforms": False,
        "temporary_float32_waveforms_created_and_deleted": True,
        "audio_annotations_or_phoneme_alignments_loaded": False,
        "onnx_runtime_loaded": frontend == "ditto",
    }
    for key, expected in required_scope.items():
        if scope.get(key) != expected:
            raise RuntimeError(f"scope mismatch for {frontend}/{key}: {scope.get(key)!r}")
    if summary.get("condition_order") != list(CONDITION_IDS):
        raise RuntimeError(f"condition order mismatch for {frontend}")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 56 or summary.get("row_count") != 56:
        raise RuntimeError(f"row count mismatch for {frontend}")
    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("sample_id"), row.get("condition"))
        if key in row_map or row.get("status") != "PASS":
            raise RuntimeError(f"duplicate/non-PASS row in {frontend}: {key}")
        if key[0] not in probe_sources or key[1] not in CONDITION_IDS:
            raise RuntimeError(f"unexpected row in {frontend}: {key}")
        source = probe_sources[key[0]]
        if row.get("speaker_id") != source["speaker_id"]:
            raise RuntimeError(f"speaker mismatch for {frontend}/{key}")
        src = row.get("source_pcm", {})
        plan_pcm = source["natural_pcm"]
        if (
            src.get("wav_sha256") != plan_pcm.get("file_sha256")
            or src.get("pcm_float32_sha256") != plan_pcm.get("pcm_sha256")
            or src.get("sample_count") != plan_pcm.get("sample_count")
            or src.get("sample_rate_hz") != plan_pcm.get("sample_rate_hz")
            or src.get("encoding") != "PCM_16"
        ):
            raise RuntimeError(f"natural PCM source mismatch for {frontend}/{key}")
        fref = row.get("feature_array", {})
        path = Path(fref.get("path", ""))
        try:
            path = path.resolve(strict=True)
            path.relative_to(out_root.resolve(strict=True))
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(f"feature array escaped output root: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"missing/nonregular feature array: {path}")
        if path.stat().st_size != fref.get("bytes") or sha256_file(path) != fref.get("sha256"):
            raise RuntimeError(f"feature array seal mismatch: {path}")
        validate_npz(path, row)
        row_map[key] = row
    expected_pairs = {(sid, condition) for sid in probe_sources for condition in CONDITION_IDS}
    if set(row_map) != expected_pairs:
        raise RuntimeError(f"incomplete 8x7 roster for {frontend}")
    for sid in probe_sources:
        maps = [json.dumps(row_map[(sid, cond)]["clock_map"], sort_keys=True, separators=(",", ":")) for cond in CONDITION_IDS]
        if len(set(maps)) != 1:
            raise RuntimeError(f"source clock map differs across conditions: {frontend}/{sid}")

    expected_output_paths = {summary_path.resolve(strict=True), *(Path(r["feature_array"]["path"]).resolve(strict=True) for r in rows)}
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 57 or receipt.get("output_file_count") != 57:
        raise RuntimeError(f"receipt output count mismatch for {frontend}")
    receipt_paths: set[Path] = set()
    member_entries: list[dict[str, Any]] = []
    for entry in outputs:
        p = Path(entry.get("path", ""))
        try:
            p = p.resolve(strict=True)
            p.relative_to(out_root.resolve(strict=True))
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(f"receipt output is missing/outside frontend root: {entry.get('path')}") from exc
        if p in receipt_paths or p not in expected_output_paths:
            raise RuntimeError(f"unexpected/duplicate output in receipt: {p}")
        receipt_paths.add(p)
        if p.stat().st_size != entry.get("bytes") or sha256_file(p) != entry.get("sha256"):
            raise RuntimeError(f"receipt file seal mismatch: {p}")
        relative = p.relative_to(out_root.resolve()).as_posix()
        member_entries.append({"path": f"{frontend}/{relative}", "bytes": p.stat().st_size, "sha256": entry["sha256"]})
    if receipt_paths != expected_output_paths:
        raise RuntimeError(f"receipt file roster incomplete for {frontend}")
    actual_files = {p.resolve(strict=True) for p in out_root.rglob("*") if p.is_file()}
    expected_files_set = expected_output_paths | {receipt_path.resolve(strict=True)}
    if actual_files != expected_files_set:
        extra = sorted(str(p) for p in actual_files - expected_files_set)
        missing = sorted(str(p) for p in expected_files_set - actual_files)
        raise RuntimeError(f"unexpected frontend output file set for {frontend}: extra={extra[:3]} missing={missing[:3]}")
    receipt_self_sha = sha256_file(receipt_path)
    member_entries.append({"path": f"{frontend}/receipt.json", "bytes": receipt_path.stat().st_size, "sha256": receipt_self_sha})
    rows_digest = hashlib.sha256("\n".join(f"{sid}\t{cond}" for sid in sorted(probe_sources) for cond in CONDITION_IDS).encode()).hexdigest()
    return member_entries, [{"frontend": frontend, "model_id": expected_model, "summary_sha256": sha256_file(summary_path), "receipt_sha256": receipt_self_sha, "row_count": len(rows), "roster_sha256": rows_digest}]


class ChunkWriter:
    def __init__(self, out_dir: Path, chunk_size: int):
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.index = 0
        self.current: BinaryIO | None = None
        self.current_hash: Any = None
        self.current_size = 0
        self.total_hash = hashlib.sha256()
        self.total_size = 0
        self.chunks: list[dict[str, Any]] = []
        self._open_next()

    def _open_next(self) -> None:
        if self.current is not None:
            self.current.flush()
            os.fsync(self.current.fileno())
            self.current.close()
            self.chunks.append({"name": self.current.name, "bytes": self.current_size, "sha256": self.current_hash.hexdigest()})
        path = self.out_dir / f"bundle.part{self.index:03d}"
        self.current = path.open("xb")
        self.current_hash = hashlib.sha256()
        self.current_size = 0
        self.index += 1

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        written = 0
        while written < len(view):
            if self.current_size >= self.chunk_size:
                self._open_next()
            take = min(len(view) - written, self.chunk_size - self.current_size)
            piece = view[written:written + take]
            assert self.current is not None
            n = self.current.write(piece)
            if n != take:
                raise OSError("short chunk write")
            self.current_hash.update(piece)
            self.total_hash.update(piece)
            self.current_size += take
            self.total_size += take
            written += take
        return written

    def tell(self) -> int:
        return self.total_size

    def flush(self) -> None:
        if self.current is not None:
            self.current.flush()

    def close(self) -> None:
        if self.current is not None:
            self.current.flush()
            os.fsync(self.current.fileno())
            self.current.close()
            self.chunks.append({"name": Path(self.current.name).name, "bytes": self.current_size, "sha256": self.current_hash.hexdigest()})
            self.current = None


def tar_add_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(safe_member_name(name))
    info.size = len(payload)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    import io
    tf.addfile(info, io.BytesIO(payload))


def tar_add_file(tf: tarfile.TarFile, source: Path, name: str) -> None:
    info = tarfile.TarInfo(safe_member_name(name))
    info.size = source.stat().st_size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with source.open("rb") as f:
        tf.addfile(info, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-plan", required=True)
    parser.add_argument("--probe-plan", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    parser.add_argument("--reserve-bytes", type=int, default=RESERVE_BYTES)
    args = parser.parse_args()
    started = time.time()
    fit_plan_path = Path(args.fit_plan)
    probe_plan_path = Path(args.probe_plan)
    runner_path = Path(args.runner)
    fit_plan, probe_plan, probe_sources = load_expected(fit_plan_path, probe_plan_path)
    runner_binding = check_bound_file(runner_path, RUNNER_SHA256)
    output_root = Path(args.output_root)
    archive_dir = Path(args.archive_dir)
    if archive_dir.exists():
        raise FileExistsError(f"fresh archive directory already exists: {archive_dir}")
    if not archive_dir.parent.is_dir():
        raise FileNotFoundError(f"archive parent must already exist: {archive_dir.parent}")

    members: list[dict[str, Any]] = []
    frontend_reports: list[dict[str, Any]] = []
    for frontend, model_id in FRONTENDS.items():
        added, report = verify_frontend(frontend, output_root / frontend, model_id, probe_sources)
        members.extend(added)
        frontend_reports.extend(report)
    expected_files = 3 * (56 + 2)
    if len(members) != expected_files:
        raise RuntimeError(f"expected {expected_files} archive members, found {len(members)}")
    names = [m["path"] for m in members]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate archive member names")
    total_source_bytes = sum(int(m["bytes"]) for m in members)
    estimate = total_source_bytes + len(members) * 2048 + 10240
    free = shutil.disk_usage(archive_dir.parent).free
    if free < estimate + args.reserve_bytes:
        raise RuntimeError(f"insufficient free space: have {free}, need {estimate + args.reserve_bytes}")
    if args.chunk_bytes < 1_048_576 or args.chunk_bytes > 6_500_000:
        raise RuntimeError("chunk size must be between 1 MiB and 6,500,000 bytes")
    archive_dir.mkdir(parents=False, exist_ok=False)

    member_manifest = {
        "schema": "cross_tfg_native_audio_encoder_probe_archive_members_v1",
        "status": "COMPLETE",
        "scope": {
            "historical_fit_samples": 8,
            "conditions_per_sample": 7,
            "frontends": list(FRONTENDS),
            "source_condition_encoder_units": 168,
            "archive_member_count_including_manifest": expected_files + 1,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "generator_forward": False,
            "video_or_syncnet": False,
        },
        "bindings": {
            "fit_plan": check_bound_file(fit_plan_path, FIT_PLAN_SHA256),
            "probe_plan": check_bound_file(probe_plan_path, PROBE_PLAN_SHA256),
            "runner": runner_binding,
        },
        "frontend_reports": frontend_reports,
        "members": sorted(members, key=lambda x: x["path"]),
        "member_payload_bytes": total_source_bytes,
        "preflight_free_bytes": free,
        "estimated_stream_tar_bytes_upper_bound": estimate,
    }
    member_manifest_bytes = canonical_json(member_manifest)
    tar_pathname = "native_encoder_probe_outputs.tar"
    chunk_writer = ChunkWriter(archive_dir, args.chunk_bytes)
    try:
        with tarfile.open(fileobj=chunk_writer, mode="w|") as tf:
            for entry in sorted(members, key=lambda x: x["path"]):
                frontend, relative = entry["path"].split("/", 1)
                tar_add_file(tf, output_root / frontend / Path(*PurePosixPath(relative).parts), entry["path"])
            tar_add_bytes(tf, "archive_manifest.json", member_manifest_bytes)
    finally:
        chunk_writer.close()
    if not chunk_writer.chunks or any(c["bytes"] > args.chunk_bytes for c in chunk_writer.chunks):
        raise RuntimeError("chunk writer produced invalid chunk roster")
    receipt = {
        "schema": "cross_tfg_native_audio_encoder_probe_archive_receipt_v1",
        "status": "COMPLETE",
        "tar_name": tar_pathname,
        "tar_bytes": chunk_writer.total_size,
        "tar_sha256": chunk_writer.total_hash.hexdigest(),
        "tar_chunk_count": len(chunk_writer.chunks),
        "chunk_bytes_limit": args.chunk_bytes,
        "chunks": [{**c, "name": Path(c["name"]).name} for c in chunk_writer.chunks],
        "archive_manifest_sha256": hashlib.sha256(member_manifest_bytes).hexdigest(),
        "archive_manifest_bytes": len(member_manifest_bytes),
        "source_payload_bytes": total_source_bytes,
        "output_member_count": expected_files,
        "frontends": frontend_reports,
        "bindings": member_manifest["bindings"],
        "scope": member_manifest["scope"],
        "elapsed_seconds": time.time() - started,
    }
    receipt_path = archive_dir / "bundle_receipt.json"
    with receipt_path.open("xb") as f:
        f.write(canonical_json(receipt))
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": receipt["status"], "tar_sha256": receipt["tar_sha256"], "tar_bytes": receipt["tar_bytes"], "chunks": len(receipt["chunks"]), "receipt": str(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
