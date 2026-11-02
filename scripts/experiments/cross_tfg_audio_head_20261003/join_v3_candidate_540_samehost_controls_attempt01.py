#!/usr/bin/env python3
"""Join 180 fixed-step V3 candidate rows with 360 exact identity/RMS controls.

This is a read-only evidence join plus deterministic scoring/summary step. It
does not render audio or video, open Confirm data, or select a method. Every
planned V3 row remains in the denominator, including FAILED, INCOMPLETE, and
missing rows. Supports are recomputed from the exact intersection of valid
source-clock query keys across identity, rms_fixed, and the V3 candidate.
Cbest remains primary; D0 and Canchor are emitted as fixed guard endpoints.
The paired candidate-builder source is a local adapter for this joiner only;
it must not be run as a standalone historical-archive builder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import tarfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
BASELINE_ARCHIVE_REL = None  # Same-host evidence path is late-bound.
BASELINE_ARCHIVE_BYTES = None  # Same-host archive byte count is late-bound.
BASELINE_ARCHIVE_SHA256 = None  # Same-host archive SHA-256 is late-bound.
CANDIDATE_BUILDER_REL = BASE + "/build_v3_candidate_samehost_controls_adapter_attempt01.py"
CANDIDATE_BUILDER_BYTES = 47404
CANDIDATE_BUILDER_SHA256 = "085253885c2ed8cb88268dd07aa604be923b00a64f054d01e7dddbdfdb3aaacd"
STAGEA_CORE_REL = BASE + "/run_stagea_dev_attempt03.py"
STAGEA_CORE_BYTES = 67_141
STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHOD_IDENTITY = "identity"
METHOD_RMS = "rms_fixed"
METHOD_V3 = "shared_recipient_transition_head_v3_fit40_step120"
METHODS = (METHOD_IDENTITY, METHOD_RMS, METHOD_V3)
BASELINE_FRONTENDS = {
    "wav2lip_gan_bac9a81e": "w2l",
    "musetalk_fixed_portrait_attempt16": "muse",
    "ditto_hubert_fixed_portrait_attempt18": "ditto",
}
ARRAY_DTYPES = {
    "query_frame_index": "<i4",
    "query_timestamp_num": "<i8",
    "query_timestamp_den": "<i8",
    "track_frame_index": "<i4",
    "valid_query_mask": "|b1",
    "distance_by_query_lag": "<f4",
    "lag_offsets_frames": "<i4",
}
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
CONTRASTS = (
    ("v3_minus_identity", METHOD_V3, METHOD_IDENTITY),
    ("v3_minus_rms_fixed", METHOD_V3, METHOD_RMS),
)
SUPPORT_RULES = {"full": 1, "guard20": 41}
BOOTSTRAP_SEED = 2026100301
BOOTSTRAP_DRAWS = 10_000


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


def safe_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} has a symlink or junction component: {cursor}")
        if cursor.exists() and cursor != lexical and not cursor.is_dir():
            raise ValueError(f"{label} has a non-directory parent component: {cursor}")
    if must_exist and not lexical.exists():
        raise FileNotFoundError(f"{label} does not exist: {lexical}")
    return lexical.resolve(strict=must_exist)


def read_regular(path: Path, label: str, limit: int = 128 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} has a symlink or junction component: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    with lexical.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = lexical.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return raw


def read_json(path: Path, label: str, limit: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, limit)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def read_jsonl(path: Path, label: str, limit: int = 128 * 1024 * 1024) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_regular(path, label, limit)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_no} is not a JSON object")
        rows.append(value)
    return rows, raw


def exact_repo_source(project_root: Path, rel: str, byte_count: int, expected_sha: str,
                      label: str) -> tuple[Path, bytes]:
    path = safe_path(project_root / Path(rel), label)
    raw = read_regular(path, label, 16 * 1024 * 1024)
    if len(raw) != byte_count or sha_bytes(raw) != expected_sha:
        raise ValueError(f"{label} differs from its frozen byte/SHA pin")
    return path, raw


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned source module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for info in archive.getmembers():
        name = info.name
        rel = PurePosixPath(name)
        if not name or "\\" in name or rel.is_absolute() or ".." in rel.parts or name in members:
            raise ValueError(f"unsafe or duplicate archive path: {name!r}")
        if not (info.isdir() or info.isfile()) or info.issym() or info.islnk() or info.isdev():
            raise ValueError(f"archive member is not a regular file/directory: {name}")
        members[name] = info
    return members


def read_tar_member(archive: tarfile.TarFile, members: Mapping[str, tarfile.TarInfo],
                    name: str, cap: int) -> bytes:
    info = members.get(name)
    if info is None or not info.isfile() or info.size < 0 or info.size > cap:
        raise ValueError(f"missing, nonregular, or oversized archive member: {name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"cannot read archive member: {name}")
    raw = stream.read(cap + 1)
    if len(raw) != info.size:
        raise ValueError(f"archive member size mismatch: {name}")
    return raw


def load_score_arrays(raw: bytes, descriptor: Mapping[str, Any], label: str,
                      expected_file_sha: str) -> dict[str, np.ndarray]:
    if sha_bytes(raw) != expected_file_sha:
        raise ValueError(f"{label} NPZ file SHA mismatch")
    meta = descriptor.get("arrays")
    if not isinstance(meta, dict) or set(meta) != set(ARRAY_DTYPES):
        raise ValueError(f"{label} NPZ array descriptor keys differ from Stage-A schema")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if set(data.files) != set(ARRAY_DTYPES):
                raise ValueError(f"{label} NPZ array keys differ from Stage-A schema")
            for name, dtype_str in ARRAY_DTYPES.items():
                arr = np.asarray(data[name])
                if arr.dtype.str != dtype_str:
                    raise ValueError(f"{label} array {name} dtype {arr.dtype.str} != {dtype_str}")
                info = meta.get(name)
                if not isinstance(info, dict):
                    raise ValueError(f"{label} array {name} has no metadata")
                raw_array = np.ascontiguousarray(arr).tobytes(order="C")
                if list(arr.shape) != info.get("shape") or dtype_str != info.get("dtype") or sha_bytes(raw_array) != info.get("array_sha256"):
                    raise ValueError(f"{label} array {name} shape/dtype/payload SHA mismatch")
                arrays[name] = np.ascontiguousarray(arr)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"{label} raw score NPZ is invalid: {exc}") from exc
    matrix = arrays["distance_by_query_lag"]
    n = int(matrix.shape[0]) if matrix.ndim == 2 else -1
    if matrix.ndim != 2 or matrix.shape[1] != 31 or n <= 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{label} distance matrix must be finite N×31 with N>0")
    for name in ("query_frame_index", "query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if arrays[name].shape != (n,):
            raise ValueError(f"{label} array {name} length differs from the distance matrix")
    if (arrays["lag_offsets_frames"].shape != (31,)
            or not np.array_equal(arrays["lag_offsets_frames"], np.arange(-15, 16, dtype="<i4"))):
        raise ValueError(f"{label} lag axis differs from the frozen -15..+15 offsets")
    if np.any(arrays["query_timestamp_den"] <= 0):
        raise ValueError(f"{label} contains nonpositive timestamp denominators")
    return arrays


def query_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int, int, int]]:
    keys = []
    for frame, num, den, track in zip(arrays["query_frame_index"], arrays["query_timestamp_num"],
                                      arrays["query_timestamp_den"], arrays["track_frame_index"]):
        frac = Fraction(int(num), int(den))
        keys.append((int(frame), frac.numerator, frac.denominator, int(track)))
    if len(set(keys)) != len(keys):
        raise ValueError("a raw score matrix contains duplicate source-clock query keys")
    return keys


def row_context(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (str(row["sample_id"]), str(row["speaker_id"]), str(row["model_id"]),
            str(row["face_id"]), int(row["seed"]))


def safe_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_control_arrays(archive_path: Path, builder: Any,
                        controls: Mapping[str, Mapping[tuple[str, str, str, int], dict[str, Any]]]
                        ) -> tuple[dict[str, Any], dict[str, bytes]]:
    result_rows: dict[str, Any] = {}
    raw_assets: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = validate_archive_members(archive)
        manifest_raw = read_tar_member(archive, members, "evidence_archive_manifest.json", 2 * 1024 * 1024)
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_evidence_archive_v1"
                or manifest.get("role") != "read_only_raw_score_and_samehost_identity_rms_control_evidence"
                or manifest.get("row_count") != 360
                or manifest.get("join_eligible") is not True
                or manifest.get("manifest_sha256") != builder.payload_sha(manifest, "manifest_sha256")
                or manifest.get("confirm_speakers_read") is not False
                or manifest.get("manifest_sha256") != builder.payload_sha(manifest, "manifest_sha256")):
            raise ValueError("baseline archive manifest schema/scope/self-seal mismatch")
        assets = manifest.get("assets")
        if (not isinstance(assets, list) or len(assets) != manifest.get("raw_score_asset_count")
                or len(assets) > 360):
            raise ValueError("same-host archive asset roster must match its sealed content-addressed score-array count")
        by_file_sha: dict[str, dict[str, Any]] = {}
        for asset in assets:
            if not isinstance(asset, dict) or asset.get("role") != "raw_score_arrays":
                continue
            digest = asset.get("sha256")
            rel = asset.get("path")
            if (not isinstance(digest, str) or len(digest) != 64 or not isinstance(rel, str)
                    or PurePosixPath(rel).is_absolute() or "\\" in rel or ".." in PurePosixPath(rel).parts
                    or rel != f"assets/{digest}.npz" or digest in by_file_sha):
                raise ValueError("baseline raw-score asset manifest has an unsafe or duplicate entry")
            by_file_sha[digest] = asset
        for method_id in (METHOD_IDENTITY, METHOD_RMS):
            for key, row in controls[method_id].items():
                file_ref = row.get("score", {}).get("raw_matrix_npz", {})
                digest = file_ref.get("file_sha256")
                asset = by_file_sha.get(digest)
                if not isinstance(asset, dict):
                    raise ValueError(f"baseline raw-score asset is absent from the archive manifest: {digest}")
                member = asset["path"]
                if member not in raw_assets:
                    raw = read_tar_member(archive, members, member, 64 * 1024 * 1024)
                    if len(raw) != asset.get("bytes") or sha_bytes(raw) != digest or digest != asset.get("sha256"):
                        raise ValueError(f"baseline raw-score asset bytes/SHA mismatch: {member}")
                    raw_assets[digest] = raw
                arrays = load_score_arrays(raw_assets[digest], file_ref,
                                           f"baseline {method_id} {key}", digest)
                row["_internal_arrays"] = arrays
                result_rows[row["row_id"]] = row
    return {"archive_manifest": manifest, "archive_manifest_raw_sha256": sha_bytes(manifest_raw)}, raw_assets


def load_candidate_arrays(rows_by_id: dict[str, dict[str, Any]], plan_by_id: Mapping[str, dict[str, Any]],
                          assets_root: Path) -> dict[str, bytes]:
    copied: dict[str, bytes] = {}
    for row_id, row in rows_by_id.items():
        planned = plan_by_id[row_id]
        if row.get("status") != "PASS":
            continue
        desc = row.get("score", {}).get("raw_matrix_npz", {})
        digest = desc.get("file_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"candidate PASS row has no raw score matrix hash: {row_id}")
        path = safe_path(assets_root / "assets" / f"{digest}.npz", f"candidate matrix {row_id}")
        raw = read_regular(path, f"candidate matrix {row_id}", 64 * 1024 * 1024)
        arrays = load_score_arrays(raw, desc, f"candidate {row_id}", digest)
        row["_internal_arrays"] = arrays
        copied[digest] = raw
        for name, value in (("sample_id", planned.get("sample_id")), ("speaker_id", planned.get("speaker_id")),
                            ("model_id", planned.get("model_id")), ("face_id", planned.get("face_id")),
                            ("seed", planned.get("seed")), ("method_id", METHOD_V3),
                            ("audio_frontend", planned.get("audio_frontend")), ("row_id", row_id)):
            if row.get(name) != value:
                raise ValueError(f"candidate result/plan mismatch at {name}: {row_id}")
        pcm = planned.get("natural_pcm", {}).get("pcm_sha256")
        if (row.get("natural_pcm_sha256") != pcm or row.get("score_audio_sha256") != pcm
                or row.get("source_face_sha256") != planned.get("source_face", {}).get("file_sha256")
                or row.get("source_video_sha256") != planned.get("source_video", {}).get("file_sha256")):
            raise ValueError(f"candidate result does not bind exact natural PCM/face/video: {row_id}")
        score = row.get("score", {})
        model_by_id = {m["model_id"]: m for m in plan_by_id[row_id].get("_models", [])}
        model = model_by_id.get(row["model_id"])
        if model is None:
            raise ValueError("internal candidate model lookup failed")
        receipt = score.get("official_evaluator", {}).get("receipt", {})
        render = row.get("render", {})
        if (score.get("status") != "PASS" or score.get("score_model_sha256") != model.get("score_model_sha256")
                or score.get("preprocessing_sha256") != model.get("preprocessing_sha256")
                or receipt.get("status") != "PASS" or receipt.get("checkpoint_sha256") != model.get("score_model_sha256")
                or (render.get("fps_num"), render.get("fps_den")) != (model.get("fps_num"), model.get("fps_den"))):
            raise ValueError(f"candidate row does not satisfy the frozen evaluator/model/FPS contract: {row_id}")
    return copied


def validate_output_dir(path: Path, project_root: Path) -> Path:
    requested = path if path.is_absolute() else project_root / path
    out = Path(os.path.abspath(os.fspath(requested)))
    try:
        rel = out.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("output directory must be lexically inside project root") from exc
    if not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("output directory cannot be the project root or contain traversal")
    cursor = project_root
    for part in rel.parts[:-1]:
        cursor /= part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"output parent contains a symlink/junction: {cursor}")
        if cursor.exists() and not cursor.is_dir():
            raise ValueError(f"output parent is not a directory: {cursor}")
    target = cursor / rel.parts[-1]
    if target.exists() or target.is_symlink() or getattr(target, "is_junction", lambda: False)():
        raise FileExistsError(f"output directory already exists: {target}")
    cursor = project_root
    for part in rel.parts[:-1]:
        cursor /= part
        if not cursor.exists():
            cursor.mkdir()
    return target


def speaker_summary(context_deltas: Mapping[tuple[str, str, int], float],
                    planned_speakers: Mapping[str, set[str]], seeds: tuple[int, ...]) -> dict[str, Any]:
    by_speaker: dict[str, dict[str, Any]] = {}
    complete_values: dict[str, float] = {}
    for speaker, samples in sorted(planned_speakers.items()):
        sample_means: dict[str, float | None] = {}
        eligible_counts: dict[str, int] = {}
        for sample in sorted(samples):
            vals = [context_deltas.get((sample, speaker, seed)) for seed in seeds]
            eligible_counts[sample] = sum(v is not None for v in vals)
            sample_means[sample] = float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None
        complete = all(eligible_counts[sample] == len(seeds) for sample in samples)
        partial = [v for v in sample_means.values() if v is not None]
        value = float(np.mean(partial)) if partial else None
        by_speaker[speaker] = {
            "complete_speaker": complete,
            "planned_samples": len(samples),
            "eligible_seed_rows": int(sum(eligible_counts.values())),
            "eligible_seed_rows_by_sample": eligible_counts,
            "sample_means": sample_means,
            "partial_speaker_mean": value,
            "speaker_mean": value if complete else None,
        }
        if complete and value is not None:
            complete_values[speaker] = value
    all_ids = sorted(planned_speakers)
    complete_ids = sorted(complete_values)
    full_mean = float(np.mean([complete_values[s] for s in all_ids])) if len(complete_ids) == len(all_ids) else None
    subset_mean = float(np.mean([complete_values[s] for s in complete_ids])) if complete_ids else None

    def boot(ids: list[str], value_map: Mapping[str, float]) -> dict[str, Any]:
        if not ids:
            return {"status": "INCOMPLETE_NO_COMPLETE_SPEAKERS", "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED, "lower": None, "upper": None}
        values = np.asarray([value_map[s] for s in ids], dtype=np.float64)
        indexes = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
            0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)), dtype=np.int64)
        draws = values[indexes].mean(axis=1)
        return {"status": "DESCRIPTIVE_SPEAKER_BOOTSTRAP", "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED, "bit_generator": "PCG64", "confidence_level": 0.95,
                "lower": float(np.quantile(draws, 0.025)), "upper": float(np.quantile(draws, 0.975))}

    result = {
        "planned_speaker_count": len(all_ids),
        "available_speaker_count": len(complete_ids),
        "complete_speaker_ids": complete_ids,
        "incomplete_speaker_ids": sorted(set(all_ids) - set(complete_ids)),
        "speaker_equal_full_cohort_mean": full_mean,
        "complete_speaker_subset_mean_descriptive": subset_mean,
        "full_cohort_bootstrap": boot(all_ids, complete_values) if full_mean is not None else {
            "status": "INCOMPLETE_FULL_COHORT", "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED, "lower": None, "upper": None},
        "complete_speaker_subset_bootstrap_descriptive": boot(complete_ids, complete_values),
        "by_speaker": by_speaker,
    }
    return result


def aggregate_contrast(rows_by_method: Mapping[str, Mapping[tuple[str, str, str, int], dict[str, Any]]],
                       planned_speakers: Mapping[str, set[str]], seeds: tuple[int, ...],
                       model_ids: tuple[str, ...], support_id: str,
                       method_a: str, method_b: str, metric_name: str) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    per_model: dict[str, Any] = {}
    speaker_values_by_model: dict[str, dict[str, float]] = {}
    context_tables: dict[str, dict[tuple[str, str, int], float]] = {}
    planned_context_count = len(seeds) * sum(len(v) for v in planned_speakers.values())
    for model_id in model_ids:
        deltas: dict[tuple[str, str, int], float] = {}
        row_a = rows_by_method[method_a]
        row_b = rows_by_method[method_b]
        statuses: Counter[str] = Counter()
        for (sample, speaker, model, seed), a in row_a.items():
            if model != model_id:
                continue
            b = row_b.get((sample, speaker, model, seed))
            if b is None:
                statuses["MISSING_COMPARATOR"] += 1
                continue
            ma = a.get("score", {}).get("metrics_by_support", {}).get(support_id, {}).get(metric_name)
            mb = b.get("score", {}).get("metrics_by_support", {}).get(support_id, {}).get(metric_name)
            eligible_a = a.get("score", {}).get("support_by_support", {}).get(support_id, {}).get("eligible") is True
            eligible_b = b.get("score", {}).get("support_by_support", {}).get(support_id, {}).get("eligible") is True
            if (a.get("status") != "PASS" or b.get("status") != "PASS" or not eligible_a or not eligible_b
                    or not isinstance(ma, (int, float)) or not isinstance(mb, (int, float))
                    or not math.isfinite(float(ma)) or not math.isfinite(float(mb))):
                statuses["INELIGIBLE_PRESERVED"] += 1
                continue
            deltas[(sample, speaker, int(seed))] = float(ma) - float(mb)
            statuses["ELIGIBLE_PAIRED"] += 1
        summary = speaker_summary(deltas, planned_speakers, seeds)
        summary["planned_context_rows"] = planned_context_count
        summary["eligible_paired_context_rows"] = len(deltas)
        summary["row_status_counts"] = dict(statuses)
        per_model[model_id] = summary
        context_tables[model_id] = deltas
        speaker_values_by_model[model_id] = {
            speaker: data["speaker_mean"] for speaker, data in summary["by_speaker"].items()
            if data["speaker_mean"] is not None
        }

    all_speakers = sorted(planned_speakers)
    common_complete = [speaker for speaker in all_speakers
                       if all(speaker in speaker_values_by_model[m] for m in model_ids)]
    full_all = len(common_complete) == len(all_speakers)
    model_equal_speakers = {
        speaker: float(np.mean([speaker_values_by_model[m][speaker] for m in model_ids]))
        for speaker in common_complete
    }
    overall_full = float(np.mean([model_equal_speakers[s] for s in all_speakers])) if full_all else None
    overall_subset = float(np.mean(list(model_equal_speakers.values()))) if model_equal_speakers else None
    if common_complete:
        shared_rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indexes = shared_rng.integers(0, len(common_complete), size=(BOOTSTRAP_DRAWS, len(common_complete)), dtype=np.int64)
        values = np.asarray([model_equal_speakers[s] for s in common_complete], dtype=np.float64)
        distribution = values[indexes].mean(axis=1)
        subset_ci = {"status": "DESCRIPTIVE_COMMON_SPEAKER_BOOTSTRAP", "draws": BOOTSTRAP_DRAWS,
                     "seed": BOOTSTRAP_SEED, "bit_generator": "PCG64", "confidence_level": 0.95,
                     "lower": float(np.quantile(distribution, 0.025)),
                     "upper": float(np.quantile(distribution, 0.975))}
    else:
        subset_ci = {"status": "INCOMPLETE_NO_COMMON_COMPLETE_SPEAKERS", "draws": BOOTSTRAP_DRAWS,
                     "seed": BOOTSTRAP_SEED, "lower": None, "upper": None}
    overall = {
        "aggregation": "equal mean across three model-specific speaker means, with speakers equal-weighted within each model",
        "planned_speaker_count": len(all_speakers),
        "complete_speaker_count_common_to_all_models": len(common_complete),
        "complete_speaker_ids_common_to_all_models": common_complete,
        "full_cohort_model_equal_mean": overall_full,
        "common_complete_speaker_subset_mean_descriptive": overall_subset,
        "full_cohort_bootstrap": {"status": "INCOMPLETE_FULL_COHORT", "draws": BOOTSTRAP_DRAWS,
                                  "seed": BOOTSTRAP_SEED, "lower": None, "upper": None},
        "common_complete_speaker_subset_bootstrap_descriptive": subset_ci,
        "by_speaker_model_equal": model_equal_speakers,
    }
    if full_all:
        all_rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        indexes = all_rng.integers(0, len(all_speakers), size=(BOOTSTRAP_DRAWS, len(all_speakers)), dtype=np.int64)
        values = np.asarray([model_equal_speakers[s] for s in all_speakers], dtype=np.float64)
        distribution = values[indexes].mean(axis=1)
        overall["full_cohort_bootstrap"] = {
            "status": "SPEAKER_EQUAL_BOOTSTRAP", "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED, "bit_generator": "PCG64", "confidence_level": 0.95,
            "lower": float(np.quantile(distribution, 0.025)),
            "upper": float(np.quantile(distribution, 0.975)),
        }
    return {"per_model": per_model, "model_equal_overall": overall}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan-raw-sha256", required=True)
    parser.add_argument("--candidate-planned-rows", type=Path, required=True)
    parser.add_argument("--candidate-planned-rows-sha256", required=True)
    parser.add_argument("--candidate-result-manifest", type=Path, required=True)
    parser.add_argument("--candidate-result-manifest-raw-sha256", required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-results-sha256", required=True)
    parser.add_argument("--candidate-cli-config", type=Path, required=True)
    parser.add_argument("--candidate-cli-config-raw-sha256", required=True)
    parser.add_argument("--candidate-assets-root", type=Path, required=True,
                        help="readback root containing assets/<raw_score_npz_sha256>.npz")
    parser.add_argument("--baseline-evidence-archive", type=Path, required=True)
    parser.add_argument("--baseline-evidence-archive-bytes", type=int, required=True)
    parser.add_argument("--baseline-evidence-archive-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = safe_path(args.project_root, "project root")
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from the materialized repository root")
    builder_path, builder_raw = exact_repo_source(project_root, CANDIDATE_BUILDER_REL,
                                                  CANDIDATE_BUILDER_BYTES, CANDIDATE_BUILDER_SHA256,
                                                  "V3 candidate builder")
    core_path, core_raw = exact_repo_source(project_root, STAGEA_CORE_REL,
                                            STAGEA_CORE_BYTES, STAGEA_CORE_SHA256,
                                            "Stage-A core")
    builder = load_module(builder_path, "dynamic_v3_candidate_builder_samehost_controls")
    core = load_module(core_path, "pinned_stagea_core_attempt03")

    archive_path = safe_path(args.baseline_evidence_archive, "baseline evidence archive")
    archive_raw_hash = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            archive_raw_hash.update(block)
    if (archive_path.stat().st_size != args.baseline_evidence_archive_bytes
            or archive_raw_hash.hexdigest() != args.baseline_evidence_archive_sha256.lower()):
        raise ValueError("same-host raw-score archive differs from its supplied exact byte/SHA pin")
    if args.baseline_evidence_archive_bytes <= 0 or len(args.baseline_evidence_archive_sha256) != 64:
        raise ValueError("same-host archive dynamic byte/SHA pin is invalid")
    builder.BASELINE_EVIDENCE_ARCHIVE_BYTES = args.baseline_evidence_archive_bytes
    builder.BASELINE_EVIDENCE_ARCHIVE_SHA256 = args.baseline_evidence_archive_sha256.lower()

    plan_path = safe_path(args.candidate_plan, "candidate plan")
    plan, plan_raw = read_json(plan_path, "candidate plan", 16 * 1024 * 1024)
    if (sha_bytes(plan_raw) != args.candidate_plan_raw_sha256.lower()
            or plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != builder.payload_sha(plan, "plan_sha256")
            or len(plan.get("rows", [])) != 180):
        raise ValueError("candidate plan raw pin/schema/payload or 180-row roster mismatch")
    if tuple(model.get("model_id") for model in plan.get("models", [])) != EXPECTED_MODELS:
        raise ValueError("candidate plan does not contain the exact three frozen frontends")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise ValueError("candidate plan does not certify closed Confirm data")
    candidate_config_path = safe_path(args.candidate_cli_config, "candidate CLI config")
    candidate_bundle = builder._candidate_bundle(project_root, candidate_config_path,
                                                args.candidate_cli_config_raw_sha256.lower())
    if candidate_bundle.get("optimizer_steps") != 120:
        raise ValueError("candidate provenance is not bound to fixed terminal step 120")
    methods_plan = plan.get("methods", [])
    if len(methods_plan) != 1 or methods_plan[0].get("method_id") != METHOD_V3:
        raise ValueError("candidate plan method is not the fixed step-120 V3 head")
    method_config = methods_plan[0].get("config", {})
    if (method_config.get("v3_cli_config_sha256") != sha_bytes(read_regular(candidate_config_path, "candidate CLI config"))
            or method_config.get("candidate_checkpoint_sha256") != candidate_bundle["checkpoint_sha256"]
            or method_config.get("candidate_training_record_sha256") != candidate_bundle["training_record_sha256"]
            or method_config.get("candidate_fixed_step_selection_record_sha256") != candidate_bundle["selection_record_sha256"]
            or method_config.get("fixed_optimizer_steps") != 120):
        raise ValueError("candidate plan method configuration differs from the frozen fixed-step candidate")
    plan_by_id: dict[str, dict[str, Any]] = {}
    for row in plan["rows"]:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id in plan_by_id or row.get("method_id") != METHOD_V3:
            raise ValueError("candidate plan has missing/duplicate IDs or a non-V3 method row")
        if row.get("split") != "dev" or row.get("face_id") != "primary_face_static_portrait_v1":
            raise ValueError("candidate row split/face differs from the frozen Dev context")
        plan_by_id[row_id] = row

    context_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in plan["rows"]:
        key = (row["sample_id"], row["speaker_id"], int(row["seed"]))
        previous = context_by_key.get(key)
        if previous is None:
            context_by_key[key] = row
        elif (previous.get("natural_pcm") != row.get("natural_pcm")
              or previous.get("source_face") != row.get("source_face")
              or previous.get("source_video") != row.get("source_video")
              or previous.get("score_audio") != row.get("score_audio")):
            raise ValueError(f"candidate frontends do not share the same original source context: {key}")
    if len(context_by_key) != 60 or len({k[0] for k in context_by_key}) != 20 or len({k[1] for k in context_by_key}) != 10:
        raise ValueError("candidate context denominator differs from 20 clips × 10 speakers × 3 seeds")
    seeds = tuple(sorted({key[2] for key in context_by_key}))
    if len(seeds) != 3 or seeds != (20261003, 20261004, 20261005):
        raise ValueError("candidate seed family differs from the frozen three Dev seeds")
    speaker_by_sample: dict[str, str] = {}
    for sample, speaker, seed in context_by_key:
        if sample in speaker_by_sample and speaker_by_sample[sample] != speaker:
            raise ValueError(f"one sample maps to multiple speakers: {sample}")
        speaker_by_sample[sample] = speaker
    for sample in speaker_by_sample:
        if {seed for candidate_sample, _, seed in context_by_key if candidate_sample == sample} != set(seeds):
            raise ValueError(f"sample lacks one of the three frozen seeds: {sample}")
    per_model_contexts = {
        model_id: {(r["sample_id"], r["speaker_id"], int(r["seed"])) for r in plan["rows"] if r["model_id"] == model_id}
        for model_id in EXPECTED_MODELS
    }
    if any(len(per_model_contexts[m]) != 60 or per_model_contexts[m] != set(context_by_key) for m in EXPECTED_MODELS):
        raise ValueError("candidate plan does not cover the same exact 60 contexts once for each model")
    contexts_for_baseline = list(context_by_key.values())
    controls, baseline_provenance = builder._baseline_rows_from_evidence_archive(
        archive_path, contexts_for_baseline, plan["models"]
    )
    baseline_asset_meta, baseline_assets = load_control_arrays(archive_path, builder, controls)

    planned_rows_path = safe_path(args.candidate_planned_rows, "candidate planned rows")
    planned_rows, planned_rows_raw = read_jsonl(planned_rows_path, "candidate planned rows", 16 * 1024 * 1024)
    if sha_bytes(planned_rows_raw) != args.candidate_planned_rows_sha256.lower() or planned_rows != plan["rows"]:
        raise ValueError("candidate planned rows file does not byte-bind the frozen plan row roster")
    if b"".join(canonical(row) + b"\n" for row in plan["rows"]) != planned_rows_raw:
        raise ValueError("candidate planned rows file is not the Stage-A canonical JSONL serialization")

    manifest_path = safe_path(args.candidate_result_manifest, "candidate result manifest")
    result_manifest, result_manifest_raw = read_json(manifest_path, "candidate result manifest", 2 * 1024 * 1024)
    if (sha_bytes(result_manifest_raw) != args.candidate_result_manifest_raw_sha256.lower()
            or result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
            or result_manifest.get("run_id") != plan.get("run_id")
            or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
            or result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
        raise ValueError("candidate result manifest raw/self/plan binding mismatch")
    result_path = safe_path(args.candidate_results, "candidate results")
    candidate_rows_list, result_raw = read_jsonl(result_path, "candidate results", 128 * 1024 * 1024)
    if (sha_bytes(result_raw) != args.candidate_results_sha256.lower()
            or result_manifest.get("results_jsonl_sha256") != sha_bytes(result_raw)
            or result_manifest.get("planned_rows") != 180
            or result_manifest.get("result_rows") != len(candidate_rows_list)):
        raise ValueError("candidate results bytes/manifest denominator mismatch")
    observed_counts = Counter(str(row.get("status")) for row in candidate_rows_list)
    expected_counts = result_manifest.get("status_counts", {})
    if {name: int(expected_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")} != {
        name: int(observed_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")
    }:
        raise ValueError("candidate result manifest status counts differ from actual ledger")
    if result_manifest.get("status") not in {"PASS", "INCOMPLETE"}:
        raise ValueError("candidate result manifest has an unrecognized terminal status")
    if result_manifest["status"] == "PASS" and (observed_counts["FAILED"] or observed_counts["INCOMPLETE"]):
        raise ValueError("candidate result manifest claims PASS despite failed/incomplete rows")
    if result_manifest["status"] == "INCOMPLETE" and not (observed_counts["FAILED"] or observed_counts["INCOMPLETE"]):
        raise ValueError("candidate result manifest claims INCOMPLETE without a failed row")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidate_rows_list:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id not in plan_by_id or row_id in candidate_by_id:
            raise ValueError(f"candidate results contain missing, unplanned, or duplicate row ID: {row_id}")
        if row.get("status") not in {"PASS", "FAILED", "INCOMPLETE"}:
            raise ValueError(f"candidate row has unknown status: {row_id}")
        candidate_by_id[row_id] = row
        for field in IDENTITY_FIELDS:
            if row.get(field) != plan_by_id[row_id].get(field):
                raise ValueError(f"candidate result/plan differs at {field}: {row_id}")
        if row.get("status") != "PASS" and not row.get("failure_reason"):
            raise ValueError(f"candidate failure row lacks a reason: {row_id}")
    missing_ids = sorted(set(plan_by_id) - set(candidate_by_id))
    for row_id in missing_ids:
        planned = plan_by_id[row_id]
        candidate_by_id[row_id] = {
            **{field: planned.get(field) for field in IDENTITY_FIELDS},
            "schema_version": "cross_tfg_audio_head_stagea_v1",
            "status": "INCOMPLETE",
            "failure_reason": "result_row_missing_from_terminal_candidate_ledger",
            "natural_pcm_sha256": planned.get("natural_pcm", {}).get("pcm_sha256"),
            "score_audio_sha256": planned.get("natural_pcm", {}).get("pcm_sha256"),
            "source_face_sha256": planned.get("source_face", {}).get("file_sha256"),
            "source_video_sha256": planned.get("source_video", {}).get("file_sha256"),
            "render": {"status": "NOT_RECORDED"},
            "mux_audio_readback": {"status": "NOT_RECORDED"},
            "score": {"status": "NOT_RUN"},
        }
    if len(candidate_by_id) != 180:
        raise ValueError("candidate rows were dropped instead of preserving the planned 180-row denominator")

    models_by_id = {m["model_id"]: m for m in plan["models"]}
    for row_id, row in candidate_by_id.items():
        if row["status"] != "PASS":
            continue
        planned = plan_by_id[row_id]
        key = (planned["sample_id"], planned["speaker_id"], planned["model_id"], int(planned["seed"]))
        for method_id in (METHOD_IDENTITY, METHOD_RMS):
            baseline = controls[method_id][key]
            if planned.get("baseline_reference_row_ids", {}).get(method_id) != baseline.get("row_id"):
                raise ValueError(f"candidate plan baseline reference ID mismatch for {row_id} / {method_id}")
            if (baseline.get("natural_pcm_sha256") != planned.get("natural_pcm", {}).get("pcm_sha256")
                    or baseline.get("score_audio_sha256") != planned.get("natural_pcm", {}).get("pcm_sha256")
                    or baseline.get("source_face_sha256") != planned.get("source_face", {}).get("file_sha256")
                    or baseline.get("source_video_sha256") != planned.get("source_video", {}).get("file_sha256")
                    or baseline.get("audio_frontend") != planned.get("audio_frontend")
                    or baseline.get("face_id") != planned.get("face_id")):
                raise ValueError(f"candidate row does not pair to exact PCM/face/video/frontend control: {row_id}")
        if row.get("score", {}).get("official_evaluator", {}).get("receipt", {}).get("expected_query_count") != row.get("score", {}).get("raw_matrix_npz", {}).get("arrays", {}).get("distance_by_query_lag", {}).get("shape", [None])[0]:
            raise ValueError(f"candidate evaluator query count differs from raw matrix shape: {row_id}")
        if row.get("score", {}).get("score_model_sha256") != models_by_id[row["model_id"]].get("score_model_sha256"):
            raise ValueError(f"candidate score model differs from frozen frontend: {row_id}")

    candidate_assets_root = safe_path(args.candidate_assets_root, "candidate asset root")
    candidate_asset_bytes = load_candidate_arrays(candidate_by_id, plan_by_id_with_models(plan_by_id, models_by_id), candidate_assets_root)

    joint_rows_plan: list[dict[str, Any]] = []
    joint_results: list[dict[str, Any]] = []
    rows_by_method: dict[str, dict[tuple[str, str, str, int], dict[str, Any]]] = {m: {} for m in METHODS}
    for source in plan["rows"]:
        key = (source["sample_id"], source["speaker_id"], source["model_id"], int(source["seed"]))
        refs = source.get("baseline_reference_row_ids", {})
        if set(refs) != {METHOD_IDENTITY, METHOD_RMS}:
            raise ValueError(f"candidate plan lacks exact identity/RMS reference IDs: {source['row_id']}")
        for method_id in (METHOD_IDENTITY, METHOD_RMS):
            row_id = refs[method_id]
            row = controls[method_id][key]
            if row.get("row_id") != row_id or row.get("method_id") != method_id:
                raise ValueError(f"baseline reference row binding mismatch: {source['row_id']} / {method_id}")
            if (key in rows_by_method[method_id]) or row_id in {r.get("row_id") for r in joint_rows_plan}:
                raise ValueError("baseline control row reused across candidate contexts")
            planned_control = {field: source.get(field) for field in IDENTITY_FIELDS}
            planned_control.update({"row_id": row_id, "method_id": method_id,
                                    "method_config_sha256": row.get("method_config_sha256"),
                                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                                    "joint_source_row_id": row_id})
            joint_rows_plan.append(planned_control)
            rows_by_method[method_id][key] = row
            joint_results.append(row)
        candidate = candidate_by_id[source["row_id"]]
        if key in rows_by_method[METHOD_V3]:
            raise ValueError("duplicate candidate context in joint row map")
        joint_candidate_plan = {field: source.get(field) for field in IDENTITY_FIELDS}
        joint_candidate_plan.update({"joint_source_row_id": source["row_id"],
                                     "baseline_reference_row_ids": refs})
        joint_rows_plan.append(joint_candidate_plan)
        rows_by_method[METHOD_V3][key] = candidate
        joint_results.append(candidate)
    if len(joint_rows_plan) != 540 or any(len(rows_by_method[m]) != 180 for m in METHODS):
        raise ValueError("three-method join must retain exactly 540 planned rows, 180 per method")

    joint_plan = {
        "schema_version": "cross-tfg-audio-eval-v1",
        "status": "FROZEN",
        "run_id": "cross_tfg_audio_head_20261003_v3_fit40_dev540_join_attempt02",
        "split": "dev",
        "methods": [{"method_id": METHOD_IDENTITY, "kind": "frozen_control"},
                    {"method_id": METHOD_RMS, "kind": "frozen_control"},
                    {"method_id": METHOD_V3, "kind": "fit_only_candidate"}],
        "models": plan["models"],
        "protocol": plan.get("protocol"),
        "rows": joint_rows_plan,
        "analysis": {
            "role": "development_only_descriptive_candidate_join",
            "methods": list(METHODS),
            "planned_rows": {METHOD_IDENTITY: 180, METHOD_RMS: 180, METHOD_V3: 180, "total": 540},
            "contrasts": [{"contrast_id": name, "method_a": a, "method_b": b, "metric": "Cbest", "role": "primary"}
                          for name, a, b in CONTRASTS],
            "support": {"source_clock_query_key": ["query_frame_index", "query_timestamp_num/den reduced", "track_frame_index"],
                        "full_minimum_common_queries": SUPPORT_RULES["full"],
                        "guard20": "trim first/last 20 keys from the three-method common source-clock order",
                        "guard20_minimum_common_queries": SUPPORT_RULES["guard20"]},
            "aggregation": {"seed_within_sample": "equal_weight", "samples_within_speaker": "equal_weight",
                            "speakers_within_model": "equal_weight", "models": "equal_weight",
                            "full_cohort_requires_every_speaker_clip_and_seed_eligible": True,
                            "complete_speaker_subset_is_descriptive_only": True},
            "bootstrap": {"bit_generator": "PCG64", "seed": BOOTSTRAP_SEED, "draw_count": BOOTSTRAP_DRAWS,
                          "confidence_level": 0.95, "unit": "speaker_equal_weight", "p_values": False},
            "excluded_method": {"method_id": "rms_localcompress", "reason": "excluded from frozen primary V3 comparisons; exploratory reference only"},
            "development_weights_selected": False,
            "confirm_audio_read": False,
            "confirm_effects_read": False,
        },
    }
    joint_plan["plan_sha256"] = payload_sha(joint_plan, "plan_sha256")
    for result in joint_results:
        result.setdefault("score", {})
        if result["status"] == "PASS":
            arrays = result.get("_internal_arrays")
            if not isinstance(arrays, dict):
                raise ValueError(f"PASS row lacks loaded raw score arrays: {result.get('row_id')}")
            digest = result.get("score", {}).get("raw_matrix_npz", {}).get("file_sha256")
            raw_asset = (baseline_assets if result.get("method_id") in (METHOD_IDENTITY, METHOD_RMS)
                         else candidate_asset_bytes).get(digest)
            if raw_asset is None:
                # Content-addressed files can be shared across methods; resolve
                # from the combined verified asset set in that case.
                raw_asset = baseline_assets.get(digest) or candidate_asset_bytes.get(digest)
            if raw_asset is None:
                raise ValueError(f"raw matrix copy is absent from the verified asset set: {result.get('row_id')}")
            result["joint_raw_score_asset"] = {
                "path": f"assets/{digest}.npz", "bytes": len(raw_asset), "sha256": digest,
            }
            result["score"].pop("lag_curve_by_support", None)
            result["score"].pop("support_by_support", None)
            result["score"].pop("metrics_by_support", None)
    core._finish_supports(joint_plan, joint_results)

    common_support_rows: list[dict[str, Any]] = []
    contexts: dict[tuple[str, str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joint_rows_plan:
        contexts[(row["split"], row["sample_id"], row["speaker_id"], row["model_id"], int(row["seed"]))][row["method_id"]] = row
    for key, method_plans in sorted(contexts.items()):
        sample, speaker, model, face, seed = key[1], key[2], key[3], None, key[4]
        results_for_context = {m: rows_by_method[m][(sample, speaker, model, seed)] for m in METHODS}
        all_pass = all(row.get("status") == "PASS" for row in results_for_context.values())
        if all_pass:
            valid_sets = {}
            key_lists = {}
            for method_id, result in results_for_context.items():
                arrays = result["_internal_arrays"]
                keys = query_keys(arrays)
                valid = arrays["valid_query_mask"].astype(bool, copy=False)
                valid_sets[method_id] = {k for k, good in zip(keys, valid) if bool(good)}
                key_lists[method_id] = keys
            common = set.intersection(*(valid_sets[m] for m in METHODS))
            ordered = sorted(common, key=lambda k: (k[0], Fraction(k[1], k[2]), k[3]))
        else:
            ordered = []
        full_keys = ordered
        guard_keys = full_keys[20:-20] if len(full_keys) >= 41 else []
        example = plan_by_id[next(row["row_id"] for row in plan["rows"]
                                  if row["sample_id"] == sample and row["speaker_id"] == speaker
                                  and row["model_id"] == model and int(row["seed"]) == seed)]
        face = example["face_id"]
        common_support_rows.append({
            "schema": "cross_tfg_v3_three_method_common_support_v1",
            "split": "dev", "sample_id": sample, "speaker_id": speaker,
            "model_id": model, "face_id": face, "seed": seed,
            "method_status": {m: results_for_context[m].get("status") for m in METHODS},
            "status": "PASS" if all_pass else "INCOMPLETE_CONTEXT_PRESERVED",
            "support": {
                "full": {"query_count": len(full_keys), "eligible": len(full_keys) >= SUPPORT_RULES["full"],
                         "support_mask_sha256": core._support_sha("full", full_keys),
                         "query_keys": [list(k) for k in full_keys]},
                "guard20": {"query_count": len(guard_keys), "eligible": len(guard_keys) >= SUPPORT_RULES["guard20"],
                            "support_mask_sha256": core._support_sha("guard20", guard_keys),
                            "query_keys": [list(k) for k in guard_keys]},
            },
        })

    planned_speakers: dict[str, set[str]] = defaultdict(set)
    for row in plan["rows"]:
        planned_speakers[row["speaker_id"]].add(row["sample_id"])
    model_ids = tuple(EXPECTED_MODELS)
    summary: dict[str, Any] = {
        "schema": "cross_tfg_v3_540row_paired_descriptive_summary_v1",
        "status": "COMPLETE_WITH_CANDIDATE_FAILURES" if result_manifest["status"] != "PASS" or missing_ids else "COMPLETE_DESCRIPTIVE_JOIN",
        "planned_rows": 540,
        "observed_baseline_rows": 360,
        "observed_candidate_rows": len(candidate_rows_list),
        "missing_candidate_rows_preserved_as_incomplete": len(missing_ids),
        "candidate_status_counts_observed": {name: int(observed_counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE")},
        "candidate_result_manifest_status": result_manifest["status"],
        "method_order": list(METHODS),
        "model_order": list(model_ids),
        "confirm_audio_read": False,
        "method_selection_performed": False,
        "localcompress_in_primary_join": False,
        "support_rules": {"full_minimum_common_queries": 1, "guard20_minimum_common_queries": 41,
                          "guard20_trim": "first/last 20 from exact three-method valid source-clock intersection"},
        "primary_metric": "Cbest",
        "guard_endpoint_metrics": ["D0", "Canchor"],
        "guard_endpoint_policy": "paired descriptive deltas on the identical frozen support; no re-selection or threshold retuning",
        "contrasts": {},
        "guard_contrasts": {},
        "eligibility_counts": {},
    }
    for support_id in ("full", "guard20"):
        eligible_by_method = {}
        for method_id in METHODS:
            eligible_by_method[method_id] = sum(
                row.get("score", {}).get("support_by_support", {}).get(support_id, {}).get("eligible") is True
                for row in rows_by_method[method_id].values()
            )
        summary["eligibility_counts"][support_id] = eligible_by_method
        for contrast_id, method_a, method_b in CONTRASTS:
            contrast_key = f"{contrast_id}:{support_id}"
            summary["contrasts"][contrast_key] = aggregate_contrast(
                rows_by_method, planned_speakers, seeds, model_ids, support_id, method_a, method_b, "Cbest"
            )
            for guard_metric in ("D0", "Canchor"):
                guard_summary = aggregate_contrast(
                    rows_by_method, planned_speakers, seeds, model_ids, support_id, method_a, method_b, guard_metric
                )
                guard_summary["metric"] = guard_metric
                summary["guard_contrasts"][f"{contrast_key}:{guard_metric}"] = guard_summary

    out_dir = validate_output_dir(args.output_dir, project_root)
    out_dir.mkdir(parents=False, exist_ok=False)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir()
    all_assets = {**baseline_assets, **candidate_asset_bytes}
    asset_records = []
    for digest, raw in sorted(all_assets.items()):
        rel = f"assets/{digest}.npz"
        target = out_dir / rel
        safe_write_new(target, raw)
        asset_records.append({"path": rel, "bytes": len(raw), "sha256": sha_bytes(raw)})
    plan_raw_out = pretty(joint_plan)
    results_raw_out = b"".join(canonical({k: v for k, v in row.items() if not k.startswith("_internal_")}) + b"\n"
                                 for row in joint_results)
    supports_raw_out = b"".join(canonical(row) + b"\n" for row in common_support_rows)
    summary_raw_out = pretty(summary)
    safe_write_new(out_dir / "joint_plan.json", plan_raw_out)
    safe_write_new(out_dir / "joint_results.jsonl", results_raw_out)
    safe_write_new(out_dir / "joint_common_support.jsonl", supports_raw_out)
    safe_write_new(out_dir / "paired_summaries.json", summary_raw_out)
    manifest = {
        "schema": "cross_tfg_v3_three_method_540row_join_manifest_v1",
        "status": summary["status"],
        "run_id": joint_plan["run_id"],
        "method_order": list(METHODS), "model_order": list(model_ids),
        "planned_rows": {METHOD_IDENTITY: 180, METHOD_RMS: 180, METHOD_V3: 180, "total": 540},
        "candidate_missing_rows": len(missing_ids),
        "candidate_result_manifest": {"path": str(manifest_path), "bytes": len(result_manifest_raw),
                                       "sha256": sha_bytes(result_manifest_raw), "status": result_manifest["status"]},
        "baseline_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                             "sha256": args.baseline_evidence_archive_sha256.lower(),
                             "manifest_raw_sha256": baseline_asset_meta["archive_manifest_raw_sha256"],
                             "role": "read_only_raw_score_and_samehost_identity_rms_control_evidence", "confirm_speakers_read": False},
        "candidate_plan": {"path": str(plan_path), "bytes": len(plan_raw), "sha256": sha_bytes(plan_raw)},
        "candidate_planned_rows": {"path": str(planned_rows_path), "bytes": len(planned_rows_raw), "sha256": sha_bytes(planned_rows_raw)},
        "candidate_results": {"path": str(result_path), "bytes": len(result_raw), "sha256": sha_bytes(result_raw)},
        "candidate_cli_config": {"path": str(candidate_config_path), "bytes": len(read_regular(candidate_config_path, "candidate config")),
                                 "sha256": args.candidate_cli_config_raw_sha256.lower()},
        "source_closure": {
            "candidate_builder": {"path": CANDIDATE_BUILDER_REL, "bytes": len(builder_raw), "sha256": sha_bytes(builder_raw)},
            "stagea_core": {"path": STAGEA_CORE_REL, "bytes": len(core_raw), "sha256": sha_bytes(core_raw)},
            "baseline_archive_provenance": baseline_provenance,
            "candidate_optimizer_steps": 120,
            "candidate_checkpoint_sha256": candidate_bundle["checkpoint_sha256"],
            "candidate_training_record_sha256": candidate_bundle["training_record_sha256"],
            "candidate_selection_record_sha256": candidate_bundle["selection_record_sha256"],
        },
        "support_rows": {"path": "joint_common_support.jsonl", "bytes": len(supports_raw_out), "sha256": sha_bytes(supports_raw_out), "contexts": len(common_support_rows)},
        "joint_plan": {"path": "joint_plan.json", "bytes": len(plan_raw_out), "sha256": sha_bytes(plan_raw_out)},
        "joint_results": {"path": "joint_results.jsonl", "bytes": len(results_raw_out), "sha256": sha_bytes(results_raw_out)},
        "paired_summaries": {"path": "paired_summaries.json", "bytes": len(summary_raw_out), "sha256": sha_bytes(summary_raw_out)},
        "assets": asset_records,
        "asset_count": len(asset_records),
        "asset_bytes": sum(item["bytes"] for item in asset_records),
        "support_rules": joint_plan["analysis"]["support"],
        "aggregation_and_bootstrap": joint_plan["analysis"]["aggregation"],
        "primary_contrasts": joint_plan["analysis"]["contrasts"],
        "method_selected": False,
        "development_weights_selected": False,
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "model_execution_performed": False,
        "p_values_computed": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_raw = pretty(manifest)
    safe_write_new(out_dir / "join_manifest.json", manifest_raw)
    print(json.dumps({"status": manifest["status"], "output_dir": str(out_dir), "planned_rows": 540,
                      "method_order": list(METHODS), "model_order": list(model_ids),
                      "candidate_missing_rows": len(missing_ids), "confirm_audio_read": False,
                      "method_selected": False, "manifest_sha256": sha_bytes(manifest_raw)},
                     sort_keys=True, ensure_ascii=False))
    return 0


def plan_by_id_with_models(plan_by_id: Mapping[str, dict[str, Any]], models: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    # Small view adapter used by the raw-matrix verifier; avoids mutating source plan rows.
    return {row_id: {**row, "_models": list(models.values())} for row_id, row in plan_by_id.items()}


if __name__ == "__main__":
    raise SystemExit(main())
