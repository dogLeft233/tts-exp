#!/usr/bin/env python3
"""Join a candidate-only visual-motion guard to a matching identity guard.

The join is descriptive: it never uses SyncNet scores and never changes either
source ledger.  Pairing requires the same dev sample, speaker, Wav2Lip model,
face and generation seed.  Frame trajectories are compared only at exact
common rational timestamps where both trackers are valid; no interpolation is
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "cross-tfg-paired-visual-motion-join-v1"
PAIR_FIELDS = (
    "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256",
    "checkpoint_sha256",
)
JOIN_FIELDS = ("split", "sample_id", "speaker_id", "model_id", "audio_frontend", "model_role", "face_id", "seed")
TRAJECTORY_ARRAYS = (
    "frame_index_int32",
    "timestamp_num_int64",
    "timestamp_den_int64",
    "tracking_valid_bool",
    "mouth_aperture_eye_normalized_float64",
    "mouth_width_eye_normalized_float64",
    "mouth_height_eye_normalized_float64",
    "mouth_area_eye_normalized_squared_float64",
)


class JoinError(RuntimeError):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _payload_sha(value: Mapping[str, Any], self_key: str) -> str:
    return _sha_bytes(_canonical({key: item for key, item in value.items() if key != self_key}))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JoinError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise JoinError(f"expected JSON object at {path}:{number}")
                rows.append(value)
    return rows


def _verify_raw(path: Path, expected: str | None, label: str) -> str:
    actual = _sha_file(path)
    if expected and actual != expected:
        raise JoinError(f"{label} SHA mismatch: got {actual}, expected {expected}")
    return actual


def _verify_visual_dir(
    root: Path,
    expected_manifest_sha: str | None,
    label: str,
    producer_plan: Mapping[str, Any],
    producer_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    manifest_path = root / "visual_motion_manifest.json"
    results_path = root / "visual_motion_results.jsonl"
    manifest_raw_sha = _verify_raw(manifest_path, expected_manifest_sha, f"{label} visual manifest")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != "cross-tfg-visual-motion-guard-v1" or manifest.get("status") not in {"PASS", "INCOMPLETE"}:
        raise JoinError(f"{label} visual manifest has an invalid terminal status")
    if manifest.get("run_id") != producer_plan.get("run_id"):
        raise JoinError(f"{label} visual manifest run_id does not bind the supplied producer plan")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise JoinError(f"{label} visual manifest payload seal mismatch")
    results_raw_sha = _sha_file(results_path)
    listed = {item.get("path"): item for item in manifest.get("files", []) if isinstance(item, dict)}
    if "visual_motion_results.jsonl" not in listed:
        raise JoinError(f"{label} visual manifest omits its result ledger")
    result_entry = listed["visual_motion_results.jsonl"]
    if result_entry.get("sha256") != results_raw_sha or result_entry.get("bytes") != results_path.stat().st_size:
        raise JoinError(f"{label} visual result-ledger seal mismatch")
    for item in manifest.get("files", []):
        rel = item.get("path")
        if not isinstance(rel, str):
            raise JoinError(f"{label} visual manifest has an invalid path")
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            raise JoinError(f"{label} visual manifest path escapes its artifact root")
        if not path.is_file() or path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise JoinError(f"{label} visual artifact seal mismatch: {rel}")
    rows = _jsonl(results_path)
    planned_rows = producer_plan.get("rows")
    if not isinstance(planned_rows, list):
        raise JoinError(f"{label} producer plan has no row ledger")
    planned_by_id = {str(row.get("row_id")): row for row in planned_rows}
    produced_by_id = {str(row.get("row_id")): row for row in producer_rows}
    if len(planned_by_id) != len(planned_rows) or len(produced_by_id) != len(producer_rows):
        raise JoinError(f"{label} producer row ids are not unique")
    for visual in rows:
        row_id = str(visual.get("row_id"))
        planned = planned_by_id.get(row_id)
        producer = produced_by_id.get(row_id)
        if planned is None or producer is None:
            raise JoinError(f"{label} visual row is not present in the exact producer plan/results: {row_id}")
        for field in PAIR_FIELDS:
            if visual.get(field) != planned.get(field) or visual.get(field) != producer.get(field):
                raise JoinError(f"{label} visual row {field} differs from its producer row: {row_id}")
        if visual.get("producer_status") != producer.get("status"):
            raise JoinError(f"{label} visual row producer_status differs from producer result: {row_id}")
        planned_video = planned.get("source_video", {})
        if planned_video.get("file_sha256") != producer.get("source_video_sha256"):
            raise JoinError(f"{label} producer source video differs from its frozen plan: {row_id}")
        if visual.get("source_face_sha256") != producer.get("source_face_sha256"):
            raise JoinError(f"{label} visual source face differs from producer result: {row_id}")
        if producer.get("status") == "PASS":
            render = producer.get("render", {})
            clock = visual.get("render_clock", {})
            video = visual.get("video", {})
            if video.get("file_sha256") != render.get("silent_video_sha256"):
                raise JoinError(f"{label} visual video SHA differs from producer render: {row_id}")
            if any(clock.get(key) != render.get(key) for key in ("fps_num", "fps_den", "frame_count", "timestamps_sha256")):
                raise JoinError(f"{label} visual frame clock differs from producer render: {row_id}")
        visual["producer_source_video_sha256"] = producer.get("source_video_sha256")
    return manifest, rows, manifest_raw_sha, results_raw_sha


def _load_trajectory(root: Path, row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    ref = row.get("visual_npz")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise JoinError(f"visual row has no trajectory ref: {row.get('row_id')}")
    path = (root / ref["path"]).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise JoinError(f"trajectory path missing or escapes root: {path}")
    raw_sha = _sha_file(path)
    if raw_sha != ref.get("file_sha256"):
        raise JoinError(f"trajectory file SHA mismatch: {row.get('row_id')}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if not set(TRAJECTORY_ARRAYS).issubset(archive.files):
            raise JoinError(f"trajectory is missing required arrays: {row.get('row_id')}")
        for key in archive.files:
            value = np.asarray(archive[key])
            descriptor = ref.get("arrays", {}).get(key)
            if not isinstance(descriptor, dict):
                raise JoinError(f"trajectory array is unsealed: {row.get('row_id')}:{key}")
            array_sha = _sha_bytes(np.ascontiguousarray(value).tobytes(order="C"))
            if descriptor.get("dtype") != value.dtype.str or descriptor.get("shape") != list(value.shape) or descriptor.get("array_sha256") != array_sha:
                raise JoinError(f"trajectory array seal mismatch: {row.get('row_id')}:{key}")
            arrays[key] = value
    lengths = {arrays[key].shape for key in TRAJECTORY_ARRAYS}
    if len(lengths) != 1 or len(next(iter(lengths))) != 1:
        raise JoinError(f"trajectory arrays have inconsistent shapes: {row.get('row_id')}")
    n = next(iter(lengths))[0]
    if n != row.get("render_clock", {}).get("frame_count"):
        raise JoinError(f"trajectory frame count disagrees with its row clock: {row.get('row_id')}")
    if not np.array_equal(arrays["frame_index_int32"], np.arange(n, dtype=np.int32)):
        raise JoinError(f"trajectory frame indices are not contiguous: {row.get('row_id')}")
    den = arrays["timestamp_den_int64"]
    if np.any(den <= 0):
        raise JoinError(f"trajectory timestamp denominator is invalid: {row.get('row_id')}")
    times = [Fraction(int(num), int(d)) for num, d in zip(arrays["timestamp_num_int64"], den)]
    if len(set(times)) != len(times):
        raise JoinError(f"trajectory timestamps are not unique: {row.get('row_id')}")
    clock = row.get("render_clock", {})
    fps = Fraction(int(clock.get("fps_num", 0)), int(clock.get("fps_den", 0)))
    if fps <= 0 or any(stamp != Fraction(index, 1) / fps for index, stamp in enumerate(times)):
        raise JoinError(f"trajectory rational timestamps disagree with declared frame clock: {row.get('row_id')}")
    arrays["_timestamp_num"] = np.asarray([t.numerator for t in times], dtype=np.int64)
    arrays["_timestamp_den"] = np.asarray([t.denominator for t in times], dtype=np.int64)
    return arrays, raw_sha


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in JOIN_FIELDS)


def _describe(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {f"{prefix}_count": 0, f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_p05": None, f"{prefix}_p95": None, f"{prefix}_rms": None}
    return {
        f"{prefix}_count": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p05": float(np.quantile(finite, 0.05)),
        f"{prefix}_p95": float(np.quantile(finite, 0.95)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(finite)))),
    }


def _pair(candidate: Mapping[str, Any], identity: Mapping[str, Any], cand_arrays: Mapping[str, np.ndarray], ident_arrays: Mapping[str, np.ndarray], cand_npz_sha: str, ident_npz_sha: str) -> dict[str, Any]:
    if _key(candidate) != _key(identity):
        raise JoinError("attempted to pair rows with different source identity keys")
    ca = cand_arrays
    ia = ident_arrays
    ctimes = [Fraction(int(n), int(d)) for n, d in zip(ca["_timestamp_num"], ca["_timestamp_den"])]
    itimes = [Fraction(int(n), int(d)) for n, d in zip(ia["_timestamp_num"], ia["_timestamp_den"])]
    c_valid = np.asarray(ca["tracking_valid_bool"], dtype=bool)
    i_valid = np.asarray(ia["tracking_valid_bool"], dtype=bool)
    c_index = {stamp: idx for idx, stamp in enumerate(ctimes) if c_valid[idx]}
    i_index = {stamp: idx for idx, stamp in enumerate(itimes) if i_valid[idx]}
    common = sorted(set(c_index).intersection(i_index))
    c_ap = np.asarray(ca["mouth_aperture_eye_normalized_float64"], dtype=np.float64)
    i_ap = np.asarray(ia["mouth_aperture_eye_normalized_float64"], dtype=np.float64)
    c_area = np.asarray(ca["mouth_area_eye_normalized_squared_float64"], dtype=np.float64)
    i_area = np.asarray(ia["mouth_area_eye_normalized_squared_float64"], dtype=np.float64)
    c_width = np.asarray(ca["mouth_width_eye_normalized_float64"], dtype=np.float64)
    i_width = np.asarray(ia["mouth_width_eye_normalized_float64"], dtype=np.float64)
    ci = np.asarray([c_index[t] for t in common], dtype=np.int64)
    ii = np.asarray([i_index[t] for t in common], dtype=np.int64)
    aperture_delta = c_ap[ci] - i_ap[ii]
    area_delta = c_area[ci] - i_area[ii]
    width_delta = c_width[ci] - i_width[ii]
    result: dict[str, Any] = {
        "status": "PASS" if common else "INCOMPLETE_NO_COMMON_VALID_TIMESTAMPS",
        **{key: candidate.get(key) for key in PAIR_FIELDS},
        "candidate_method_id": candidate.get("method_id"),
        "identity_method_id": identity.get("method_id"),
        "candidate_row_id": candidate.get("row_id"),
        "identity_row_id": identity.get("row_id"),
        "candidate_visual_npz_sha256": cand_npz_sha,
        "identity_visual_npz_sha256": ident_npz_sha,
        "candidate_video_sha256": candidate.get("video", {}).get("file_sha256"),
        "identity_video_sha256": identity.get("video", {}).get("file_sha256"),
        "candidate_source_video_sha256": candidate.get("producer_source_video_sha256"),
        "identity_source_video_sha256": identity.get("producer_source_video_sha256"),
        "candidate_clock_signature": candidate.get("render_clock", {}).get("clock_signature"),
        "identity_clock_signature": identity.get("render_clock", {}).get("clock_signature"),
        "candidate_valid_frames": int(c_valid.sum()),
        "identity_valid_frames": int(i_valid.sum()),
        "common_exact_timestamp_valid_frames": int(len(common)),
        "candidate_method_valid_fraction": float(c_valid.mean()) if c_valid.size else 0.0,
        "identity_method_valid_fraction": float(i_valid.mean()) if i_valid.size else 0.0,
        "common_support_fraction_of_smaller_valid_set": (float(len(common) / min(int(c_valid.sum()), int(i_valid.sum()))) if min(int(c_valid.sum()), int(i_valid.sum())) else 0.0),
        "common_timestamp_first": ({"num": common[0].numerator, "den": common[0].denominator} if common else None),
        "common_timestamp_last": ({"num": common[-1].numerator, "den": common[-1].denominator} if common else None),
        "interpretation": "descriptive paired geometry; no interpolation; mouth motion magnitude is not a quality score",
    }
    result.update(_describe(aperture_delta, "aperture_delta_ipd"))
    result.update(_describe(area_delta, "area_delta_ipd2"))
    result.update(_describe(width_delta, "width_delta_ipd"))

    # Velocity contrasts are restricted to consecutive timestamps valid in both
    # rows.  This avoids comparing derivatives over different tracking gaps.
    transition_delta: list[float] = []
    for left, right in zip(common, common[1:]):
        if right - left != Fraction(1, 1) / Fraction(int(candidate["render_clock"]["fps_num"]), int(candidate["render_clock"]["fps_den"])):
            continue
        c0, c1 = c_index[left], c_index[right]
        i0, i1 = i_index[left], i_index[right]
        dt = float(right - left)
        cv = (c_ap[c1] - c_ap[c0]) / dt
        iv = (i_ap[i1] - i_ap[i0]) / dt
        if math.isfinite(cv) and math.isfinite(iv):
            transition_delta.append(cv - iv)
    result.update(_describe(np.asarray(transition_delta, dtype=np.float64), "aperture_velocity_delta_ipd_per_s"))
    return result


def _read_and_check_rows(plan_path: Path, results_path: Path, manifest_path: Path, expected_plan_sha: str | None, expected_results_sha: str | None, expected_manifest_sha: str | None, role: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    plan_sha = _verify_raw(plan_path, expected_plan_sha, f"{role} plan")
    results_sha = _verify_raw(results_path, expected_results_sha, f"{role} results")
    manifest_sha = _verify_raw(manifest_path, expected_manifest_sha, f"{role} result manifest")
    plan = _json(plan_path)
    manifest = _json(manifest_path)
    if plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise JoinError(f"{role} plan payload seal mismatch")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise JoinError(f"{role} result manifest payload seal mismatch")
    manifest_plan_sha = manifest.get("plan_raw_sha256")
    manifest_results_sha = manifest.get("results_raw_sha256", manifest.get("results_jsonl_sha256"))
    if manifest_plan_sha != plan_sha or manifest_results_sha != results_sha:
        raise JoinError(f"{role} result manifest does not bind plan/results bytes")
    planned = plan.get("rows")
    rows = _jsonl(results_path)
    if not isinstance(planned, list) or len(planned) != len(rows):
        raise JoinError(f"{role} result ledger count does not match its plan")
    expected_ids = [r.get("row_id") for r in planned]
    actual_ids = [r.get("row_id") for r in rows]
    if expected_ids != actual_ids:
        raise JoinError(f"{role} results are not in frozen plan order")
    status_counts = manifest.get("status_counts", {})
    if manifest.get("result_rows") != len(rows) or sum(int(status_counts.get(k, 0)) for k in ("PASS", "FAILED", "INCOMPLETE")) != len(rows):
        raise JoinError(f"{role} result manifest row/status totals mismatch")
    plan_by_id = {r.get("row_id"): r for r in planned}
    for row in rows:
        frozen = plan_by_id.get(row.get("row_id"))
        if frozen is None or any(row.get(field) != frozen.get(field) for field in PAIR_FIELDS):
            raise JoinError(f"{role} result identity differs from plan: {row.get('row_id')}")
    seals = {"plan_raw_sha256": plan_sha, "results_raw_sha256": results_sha, "result_manifest_raw_sha256": manifest_sha}
    return plan, rows, manifest, seals


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("candidate", "identity"):
        parser.add_argument(f"--{prefix}-plan", type=Path, required=True)
        parser.add_argument(f"--{prefix}-results", type=Path, required=True)
        parser.add_argument(f"--{prefix}-result-manifest", type=Path, required=True)
        parser.add_argument(f"--{prefix}-visual-dir", type=Path, required=True)
        parser.add_argument(f"--expected-{prefix}-plan-sha256")
        parser.add_argument(f"--expected-{prefix}-results-sha256")
        parser.add_argument(f"--expected-{prefix}-result-manifest-sha256")
        parser.add_argument(f"--expected-{prefix}-visual-manifest-sha256")
    parser.add_argument("--candidate-method", default="spectral_head_fit_v3_zero_lag_unitcos")
    parser.add_argument("--identity-method", default="identity")
    parser.add_argument("--expected-pairs", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise JoinError(f"output already exists: {args.out}")
    cand_plan, cand_results, cand_manifest, cand_seals = _read_and_check_rows(
        args.candidate_plan, args.candidate_results, args.candidate_result_manifest,
        args.expected_candidate_plan_sha256, args.expected_candidate_results_sha256,
        args.expected_candidate_result_manifest_sha256, "candidate",
    )
    id_plan, id_results, id_manifest, id_seals = _read_and_check_rows(
        args.identity_plan, args.identity_results, args.identity_result_manifest,
        args.expected_identity_plan_sha256, args.expected_identity_results_sha256,
        args.expected_identity_result_manifest_sha256, "identity",
    )
    cand_vis_manifest, cand_vis_rows, cand_vis_manifest_sha, cand_vis_results_sha = _verify_visual_dir(
        args.candidate_visual_dir, args.expected_candidate_visual_manifest_sha256, "candidate", cand_plan, cand_results
    )
    id_vis_manifest, id_vis_rows, id_vis_manifest_sha, id_vis_results_sha = _verify_visual_dir(
        args.identity_visual_dir, args.expected_identity_visual_manifest_sha256, "identity", id_plan, id_results
    )

    candidate_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in cand_vis_rows:
        if row.get("method_id") != args.candidate_method:
            continue
        key = _key(row)
        if key in candidate_by_key:
            raise JoinError(f"duplicate candidate visual pair key: {key}")
        candidate_by_key[key] = row
    identity_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in id_vis_rows:
        if row.get("method_id") != args.identity_method or row.get("model_id") != "wav2lip_gan_bac9a81e":
            continue
        key = _key(row)
        if key in identity_by_key:
            raise JoinError(f"duplicate identity visual pair key: {key}")
        identity_by_key[key] = row
    if len(candidate_by_key) != args.expected_pairs:
        raise JoinError(f"candidate visual row count mismatch: {len(candidate_by_key)} != {args.expected_pairs}")
    if set(candidate_by_key) != set(identity_by_key):
        missing_identity = len(set(candidate_by_key) - set(identity_by_key))
        extra_identity = len(set(identity_by_key) - set(candidate_by_key))
        raise JoinError(f"candidate/identity exact-pair keys differ: missing_identity={missing_identity}, extra_identity={extra_identity}")

    # Only the selected candidate and identity rows are required to be PASS in
    # their frozen ledgers; all other planned rows retain their original status.
    candidate_plan_ids = {r.get("row_id") for r in cand_plan.get("rows", [])}
    identity_plan_ids = {r.get("row_id") for r in id_plan.get("rows", [])}
    if any(r.get("row_id") not in candidate_plan_ids for r in candidate_by_key.values()):
        raise JoinError("candidate visual row is absent from the frozen candidate plan")
    if any(r.get("row_id") not in identity_plan_ids for r in identity_by_key.values()):
        raise JoinError("identity visual row is absent from the frozen identity plan")

    pairs: list[dict[str, Any]] = []
    for key in sorted(candidate_by_key, key=lambda item: (item[1], item[-1])):
        c_row = candidate_by_key[key]
        i_row = identity_by_key[key]
        if c_row.get("status") != "PASS" or c_row.get("producer_status") != "PASS" or i_row.get("status") != "PASS" or i_row.get("producer_status") != "PASS":
            pairs.append({
                **{field: c_row.get(field) for field in PAIR_FIELDS},
                "status": "UNPAIRED_VISUAL_ROW_NOT_PASS",
                "candidate_method_id": c_row.get("method_id"),
                "identity_method_id": i_row.get("method_id"),
                "candidate_row_id": c_row.get("row_id"),
                "identity_row_id": i_row.get("row_id"),
                "candidate_status": c_row.get("status"),
                "candidate_producer_status": c_row.get("producer_status"),
                "candidate_failure_reason": c_row.get("failure_reason"),
                "identity_status": i_row.get("status"),
                "identity_producer_status": i_row.get("producer_status"),
                "identity_failure_reason": i_row.get("failure_reason"),
                "candidate_visual_npz_sha256": None,
                "identity_visual_npz_sha256": None,
                "common_exact_timestamp_valid_frames": 0,
                "aperture_delta_ipd_mean": None,
                "area_delta_ipd2_mean": None,
                "width_delta_ipd_mean": None,
                "aperture_velocity_delta_ipd_per_s_mean": None,
                "interpretation": "planned pair retained in denominator; visual trajectory unavailable because one source guard row did not PASS",
            })
            continue
        c_arrays, c_npz_sha = _load_trajectory(args.candidate_visual_dir, c_row)
        i_arrays, i_npz_sha = _load_trajectory(args.identity_visual_dir, i_row)
        pairs.append(_pair(c_row, i_row, c_arrays, i_arrays, c_npz_sha, i_npz_sha))

    args.out.mkdir(parents=True)
    pair_path = args.out / "paired_motion_deltas.jsonl"
    with pair_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in pairs:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    speaker_groups: dict[str, list[dict[str, Any]]] = {}
    sample_groups: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        speaker_groups.setdefault(str(row["speaker_id"]), []).append(row)
        sample_groups.setdefault(str(row["sample_id"]), []).append(row)
    metric_prefixes = (
        "aperture_delta_ipd_mean", "aperture_delta_ipd_median", "area_delta_ipd2_mean",
        "width_delta_ipd_mean", "aperture_velocity_delta_ipd_per_s_mean",
    )
    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        estimable = [
            row for row in group
            if row.get("status") == "PASS" and int(row.get("common_exact_timestamp_valid_frames", 0)) > 0
        ]
        planned_sample_seeds: dict[str, set[int]] = {}
        estimable_sample_seeds: dict[str, set[int]] = {}
        planned_speaker_cells: dict[str, set[tuple[str, int]]] = {}
        estimable_speaker_cells: dict[str, set[tuple[str, int]]] = {}
        for row in group:
            sample = str(row["sample_id"])
            speaker = str(row["speaker_id"])
            seed = int(row["seed"])
            planned_sample_seeds.setdefault(sample, set()).add(seed)
            planned_speaker_cells.setdefault(speaker, set()).add((sample, seed))
        for row in estimable:
            sample = str(row["sample_id"])
            speaker = str(row["speaker_id"])
            seed = int(row["seed"])
            estimable_sample_seeds.setdefault(sample, set()).add(seed)
            estimable_speaker_cells.setdefault(speaker, set()).add((sample, seed))
        output: dict[str, Any] = {
            "paired_rows": len(group),
            "speaker_count": len({r["speaker_id"] for r in group}),
            "planned_pair_count": len(group),
            "estimable_pair_count": len(estimable),
            "planned_clip_count": len(planned_sample_seeds),
            "clips_with_any_estimable_seed": sum(bool(estimable_sample_seeds.get(sample)) for sample in planned_sample_seeds),
            "clips_fully_estimable_all_planned_seeds": sum(
                estimable_sample_seeds.get(sample, set()) == seeds
                for sample, seeds in planned_sample_seeds.items()
            ),
            "planned_speaker_count": len(planned_speaker_cells),
            "speakers_with_any_estimable_pair": sum(bool(estimable_speaker_cells.get(speaker)) for speaker in planned_speaker_cells),
            "speakers_fully_estimable_all_planned_clip_seed_cells": sum(
                estimable_speaker_cells.get(speaker, set()) == cells
                for speaker, cells in planned_speaker_cells.items()
            ),
            "planned_seed_count": len({int(r["seed"]) for r in group}),
            "estimable_seed_count": len({int(r["seed"]) for r in estimable}),
            "planned_seed_replicate_count": len(group),
            "estimable_seed_replicate_count": len(estimable),
        }
        for key in metric_prefixes:
            vals = [float(r[key]) for r in estimable if r.get(key) is not None]
            output[key] = _mean(vals)
        output["mean_common_frames"] = _mean([float(r["common_exact_timestamp_valid_frames"]) for r in estimable])
        return output

    summary = {
        "schema_version": SCHEMA,
        "status": "PASS" if len(pairs) == args.expected_pairs and all(row.get("status") == "PASS" for row in pairs) else "INCOMPLETE",
        "pair_count": len(pairs),
        "paired_pass_count": sum(1 for row in pairs if row.get("status") == "PASS"),
        "zero_common_support_pairs": sum(1 for row in pairs if row["common_exact_timestamp_valid_frames"] == 0),
        "planned_clip_count": len({row["sample_id"] for row in pairs}),
        "estimable_clip_count": len({row["sample_id"] for row in pairs if row.get("status") == "PASS"}),
        "planned_speaker_count": len({row["speaker_id"] for row in pairs}),
        "estimable_speaker_count": len({row["speaker_id"] for row in pairs if row.get("status") == "PASS"}),
        "planned_seed_count": len({int(row["seed"]) for row in pairs}),
        "estimable_seed_count": len({int(row["seed"]) for row in pairs if row.get("status") == "PASS"}),
        "planned_seed_replicate_count": len(pairs),
        "estimable_seed_replicate_count": sum(1 for row in pairs if row.get("status") == "PASS"),
        "candidate_method_id": args.candidate_method,
        "identity_method_id": args.identity_method,
        "candidate_run_id": cand_plan.get("run_id"),
        "identity_run_id": id_plan.get("run_id"),
        "interpretation": "descriptive cross-run visual-motion pairing only; SyncNet endpoints are not used; no interpolation is performed",
        "candidate_inputs": {**cand_seals, "visual_manifest_run_id": cand_vis_manifest.get("run_id"), "visual_manifest_raw_sha256": cand_vis_manifest_sha, "visual_results_raw_sha256": cand_vis_results_sha, "visual_manifest_payload_sha256": cand_vis_manifest.get("manifest_sha256")},
        "identity_inputs": {**id_seals, "visual_manifest_run_id": id_vis_manifest.get("run_id"), "visual_manifest_raw_sha256": id_vis_manifest_sha, "visual_results_raw_sha256": id_vis_results_sha, "visual_manifest_payload_sha256": id_vis_manifest.get("manifest_sha256")},
        "aggregate_all_pairs": aggregate(pairs),
        "aggregate_by_speaker": {key: aggregate(value) for key, value in sorted(speaker_groups.items())},
        "aggregate_by_sample": {key: aggregate(value) for key, value in sorted(sample_groups.items())},
        "paired_rows_path": pair_path.name,
        "paired_rows_sha256": _sha_file(pair_path),
        "paired_rows_bytes": pair_path.stat().st_size,
    }
    summary["manifest_sha256"] = _payload_sha(summary, "manifest_sha256")
    manifest_path = args.out / "motion_join_manifest.json"
    manifest_path.write_bytes(_canonical(summary) + b"\n")
    print(json.dumps({"status": summary["status"], "pair_count": len(pairs), "manifest": str(manifest_path), "manifest_sha256": _sha_file(manifest_path), "pairs_sha256": summary["paired_rows_sha256"]}, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JoinError as exc:
        raise SystemExit(f"visual-motion join failed: {exc}") from exc
