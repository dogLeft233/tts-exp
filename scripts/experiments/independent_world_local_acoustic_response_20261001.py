#!/usr/bin/env python3
"""Independent local response/dose analysis for the sealed WORLD run.

This module imports only NumPy and the Python standard library. It does not
import the WORLD producer or SyncNet scorer. It refuses to read score arrays
until the run, physical waveform QA, and independent WORLD score audit all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PLAN_SHA256 = "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d"
EXPECTED_QA04_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
EXPECTED_INVENTORY_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"
EXPECTED_CAL_MANIFEST_SHA256 = "297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc"
EXPECTED_CAL_QC_SHA256 = "5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5"
EXPECTED_RECIPIENTS = 80
EXPECTED_SPEAKERS = 40
EXPECTED_CONTEXTS = 160
EXPECTED_ALIAS_ROWS = 12_160
EXPECTED_ANCHOR = {"k": 3, "column": 18, "official_offset": -3}
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
NULL_SEED_BASE = 20260930
NULL_DRAWS = 100

PHONE_LABELS = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)
OUTCOMES = ("whole", "early", "middle", "late")
SUPPORTS = ("full", "valid", "guard20", "guard15")
GEOMETRIES = ("raw", "unit")
ARMS = ("EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
PHYSICAL_DOSE_ARMS = ("EX", "SP", "BOTH", "F0", "AP")
DOSE_BANDS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


class AnalysisError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checked_file(path: Path, expected: str, label: str) -> str:
    path = path.resolve(strict=True)
    actual = sha256_file(path)
    if actual != expected:
        raise AnalysisError(f"{label} SHA mismatch: {path}: {actual} != {expected}")
    return actual


def npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def expected_mel_starts(mel_frames: int) -> np.ndarray:
    """Official Wav2Lip 80-Hz / 25-fps, 16-column chunk loop and tail fallback."""
    if mel_frames < 16:
        raise AnalysisError(f"mel has fewer than 16 columns: {mel_frames}")
    starts: list[int] = []
    index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > mel_frames:
            starts.append(mel_frames - 16)
            break
        starts.append(start)
        index += 1
    return np.asarray(starts, dtype=np.int32)


def expected_anchor(row: dict[str, Any]) -> bool:
    return (
        row.get("fixed_anchor_k") == EXPECTED_ANCHOR["k"]
        and row.get("fixed_anchor_column") == EXPECTED_ANCHOR["column"]
        and row.get("fixed_anchor_official_offset") == EXPECTED_ANCHOR["official_offset"]
    )


def inventory_sha(obj: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({key: value for key, value in obj.items() if key != "inventory_sha256"})).hexdigest()


def _find_named_record(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if isinstance(value, dict):
        return value
    raise AnalysisError(f"expected protocol input seal {key!r}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Read seals and row metadata only; score matrices are not opened here."""
    run = args.run_dir.resolve(strict=True)
    out = args.output_dir.resolve()
    if out == run or run in out.parents:
        raise AnalysisError("analysis output must be outside the sealed WORLD run")
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"refusing nonempty analysis output: {out}")

    protocol_path = run / "evaluation_protocol.json"
    rows_path = run / "evaluation_rows.jsonl"
    results_path = run / "evaluation_results.jsonl"
    status_path = run / "run_status.json"
    all_files_path = run / "all_files_sha256.json"
    code_seal_path = run / "code_seal.json"
    for path in (protocol_path, rows_path, results_path, status_path, all_files_path, code_seal_path):
        if not path.is_file():
            raise AnalysisError(f"missing required sealed run file: {path}")

    protocol = read_json(protocol_path)
    protocol_digest = hashlib.sha256(canonical_json({k: v for k, v in protocol.items() if k != "protocol_sha256"})).hexdigest()
    if protocol.get("protocol_sha256") != protocol_digest:
        raise AnalysisError("protocol canonical self-hash mismatch")
    if protocol.get("execution_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise AnalysisError("execution plan SHA differs from the frozen WORLD plan")
    plan_path = args.execution_plan.resolve(strict=True)
    plan_sha = checked_file(plan_path, EXPECTED_PLAN_SHA256, "execution plan")

    run_status = read_json(status_path)
    if run_status.get("status") != "complete" or run_status.get("completed_contexts") != EXPECTED_CONTEXTS or run_status.get("failed_contexts") != 0:
        raise AnalysisError(f"WORLD run is not fully complete: {run_status.get('status')}/{run_status.get('completed_contexts')}/{run_status.get('failed_contexts')}")
    if run_status.get("planned_rows") != EXPECTED_ALIAS_ROWS or run_status.get("scored_rows") != EXPECTED_ALIAS_ROWS or run_status.get("failed_rows") != 0:
        raise AnalysisError("WORLD run does not contain the complete 12,160-row intended grid")

    qa_path = args.physical_qa.resolve(strict=True)
    qa_sha = checked_file(qa_path, EXPECTED_QA04_SHA256, "independent physical QA")
    qa = read_json(qa_path)
    if qa.get("status") != "PASS" or qa.get("failures") != []:
        raise AnalysisError("physical waveform QA is not PASS")
    cov = qa.get("coverage", {})
    if cov.get("completed_recipient_contexts") != 160 or cov.get("expected_physical_wavs") != 1920 or cov.get("unique_physical_wavs") != 1920 or cov.get("expected_npz_artifacts") != 1920 or cov.get("unique_npz_artifacts") != 1920:
        raise AnalysisError("physical QA coverage differs from the frozen 160-context plan")

    consumer_audit_path = args.consumer_audit.resolve(strict=True)
    consumer_stats_path = args.consumer_statistics.resolve(strict=True)
    consumer_audit = read_json(consumer_audit_path)
    consumer_stats = read_json(consumer_stats_path)
    if consumer_audit.get("status") != "PASS" or consumer_stats.get("status") != "COMPLETE":
        raise AnalysisError("independent WORLD score consumer has not passed completely")
    if consumer_stats.get("source_audit_sha256") != sha256_file(consumer_audit_path):
        raise AnalysisError("independent score statistics do not bind the supplied audit report")
    if consumer_audit.get("run_root") != str(run) or consumer_audit.get("run_seals", {}).get("raw_protocol_sha256") != sha256_file(protocol_path):
        raise AnalysisError("independent score consumer audited a different protocol")
    if not consumer_stats.get("scientific_eligibility", {}).get("full_cohort_complete"):
        raise AnalysisError("independent score consumer reports incomplete intended cohort")

    inventory_path = args.phone_inventory.resolve(strict=True)
    inventory_sha_raw = checked_file(inventory_path, EXPECTED_INVENTORY_SHA256, "CAL phone inventory")
    inventory = read_json(inventory_path)
    if inventory.get("status") != "FROZEN_CAL_ONLY_PHONE_INVENTORY" or inventory.get("selection_split") != "calibration" or inventory.get("selected_phone_labels") != list(PHONE_LABELS):
        raise AnalysisError("frozen CAL phone inventory identity mismatch")
    if inventory_sha(inventory) != inventory.get("inventory_sha256"):
        raise AnalysisError("CAL inventory embedded self-hash mismatch")
    src = inventory.get("source", {})
    if src.get("calibration_manifest_file_sha256") != EXPECTED_CAL_MANIFEST_SHA256 or src.get("calibration_qc_sha256") != EXPECTED_CAL_QC_SHA256:
        raise AnalysisError("CAL-only label-selection inputs do not match the frozen CAL manifests")

    physical_manifest_path = args.physical_manifest.resolve(strict=True)
    physical = read_json(physical_manifest_path)
    physical_sha = sha256_file(physical_manifest_path)
    input_seals = protocol.get("input_seals", {})
    physical_seal = _find_named_record(input_seals, "physical_manifest")
    if physical_seal.get("sha256") != physical_sha:
        raise AnalysisError("physical manifest does not match sealed WORLD input SHA")
    if physical.get("status") != "EVALUATION_WAVEFORMS_READY" or physical.get("split") != "evaluation":
        raise AnalysisError("physical EVAL manifest is not ready")
    if len([row for row in physical.get("records", []) if row.get("split") == "evaluation"]) != EXPECTED_RECIPIENTS:
        raise AnalysisError("physical EVAL manifest does not contain the intended 80 recipients")

    raw_rows = rows_path.read_bytes()
    row_sha = hashlib.sha256(raw_rows).hexdigest()
    if row_sha != protocol.get("row_manifest_sha256"):
        raise AnalysisError("run row manifest is not bound by its protocol")
    plan_rows = read_jsonl(rows_path)
    results = read_jsonl(results_path)
    if len(plan_rows) != EXPECTED_ALIAS_ROWS or len(results) != EXPECTED_ALIAS_ROWS:
        raise AnalysisError("planned or scored alias row count differs from 12,160")
    if any(row.get("status") != "scored" for row in results) or any(not expected_anchor(row) for row in results):
        raise AnalysisError("a planned score row failed or its fixed anchor differs from CAL k=3/column18/offset-3")
    row_ids = [row.get("row_id") for row in plan_rows]
    result_ids = [row.get("row_id") for row in results]
    if len(set(row_ids)) != EXPECTED_ALIAS_ROWS or row_ids != result_ids:
        raise AnalysisError("planned/result row identities differ or are duplicated")
    context_keys = {(row.get("sample_id"), row.get("direction")) for row in plan_rows}
    if len(context_keys) != EXPECTED_CONTEXTS:
        raise AnalysisError("planned alias rows do not span 80 recipients × 2 recipient clocks")

    all_files = read_json(all_files_path)
    if not isinstance(all_files, dict) or len(all_files) < 1:
        raise AnalysisError("producer all-files inventory is malformed")
    all_files_sha = sha256_file(all_files_path)
    files_map = all_files.get("files", all_files)
    for relative, path, digest in (
        ("evaluation_protocol.json", protocol_path, sha256_file(protocol_path)),
        ("evaluation_rows.jsonl", rows_path, row_sha),
        ("evaluation_results.jsonl", results_path, sha256_file(results_path)),
        ("run_status.json", status_path, sha256_file(status_path)),
        ("code_seal.json", code_seal_path, sha256_file(code_seal_path)),
    ):
        if files_map.get(relative) != digest:
            raise AnalysisError(f"producer all-files inventory does not bind {relative}")
    code_seal = read_json(code_seal_path)
    if code_seal.get("protocol_sha256") != sha256_file(protocol_path) or code_seal.get("row_manifest_sha256") != row_sha:
        raise AnalysisError("producer code seal does not bind protocol and rows")

    return {
        "run_dir": run,
        "output_dir": out,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "plan_path": plan_path,
        "plan_sha256": plan_sha,
        "rows_path": rows_path,
        "row_manifest_sha256": row_sha,
        "results_path": results_path,
        "results_sha256": sha256_file(results_path),
        "run_status": run_status,
        "all_files_path": all_files_path,
        "all_files_sha256": all_files_sha,
        "code_seal_sha256": sha256_file(code_seal_path),
        "physical_manifest_path": physical_manifest_path,
        "physical_manifest_sha256": physical_sha,
        "physical_qa_path": qa_path,
        "physical_qa_sha256": qa_sha,
        "consumer_audit_path": consumer_audit_path,
        "consumer_audit_sha256": sha256_file(consumer_audit_path),
        "consumer_statistics_path": consumer_stats_path,
        "consumer_statistics_sha256": sha256_file(consumer_stats_path),
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_sha_raw,
        "planned_row_count": len(plan_rows),
        "result_row_count": len(results),
        "context_count": len(context_keys),
        "rows": plan_rows,
        "results": results,
        "physical_manifest": physical,
        "physical_records": {row["id"]: row for row in physical["records"] if row.get("split") == "evaluation"},
    }


def nearest_indices(grid: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    query = np.asarray(targets, dtype=np.float64)
    right = np.searchsorted(values, query, side="left")
    right = np.clip(right, 0, len(values) - 1)
    left = np.clip(right - 1, 0, len(values) - 1)
    choose_left = np.abs(query - values[left]) <= np.abs(values[right] - query)
    return np.where(choose_left, left, right).astype(np.int64)


def _phone_intervals(phone_map: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {label: [] for label in PHONE_LABELS}
    for item in phone_map.get("matched_phones", []):
        label = item.get("recipient_normalized_label")
        if label not in result:
            continue
        interval = item.get("recipient_interval_s")
        if not isinstance(interval, list) or len(interval) != 2:
            raise AnalysisError("phone map has a malformed exact recipient interval")
        start, end = map(float, interval)
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise AnalysisError("phone map has a nonpositive or nonfinite exact interval")
        result[label].append({"phone_index": int(item["recipient_phone_index"]), "start": start, "end": end, "word_index": int(item["word_index"])})
    for label, tokens in result.items():
        tokens.sort(key=lambda row: (row["start"], row["end"], row["phone_index"]))
        for left, right in zip(tokens, tokens[1:]):
            if right["start"] < left["end"] - 1e-9:
                raise AnalysisError(f"overlapping matched recipient phone intervals for {label}")
    return result


def window_unions(starts: np.ndarray, query_count: int, mel_frames: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    if query_count > len(starts) - 5:
        raise AnalysisError("SyncNet V query count exceeds the five-frame Wav2Lip clock")
    unions: list[np.ndarray] = []
    indptr = [0]
    flat: list[int] = []
    frame_cols = np.arange(16, dtype=np.int64)
    for query in range(query_count):
        cols = np.unique(np.concatenate([starts[frame] + frame_cols for frame in range(query, query + 5)]))
        if cols.size == 0 or cols.min() < 0 or cols.max() >= mel_frames:
            raise AnalysisError("visual receptive-field set union escapes the saved mel array")
        unions.append(cols)
        flat.extend(int(value) for value in cols)
        indptr.append(len(flat))
    return unions, np.asarray(indptr, dtype=np.int32), np.asarray(flat, dtype=np.int32)


def phone_weights(
    unions: list[np.ndarray], mel_frames: int, intervals: dict[str, list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, list[list[list[int]]], np.ndarray]:
    centers = np.arange(mel_frames, dtype=np.float64) / 80.0
    values = np.zeros((len(PHONE_LABELS), len(OUTCOMES), len(unions)), dtype=np.float32)
    stft = np.zeros((len(PHONE_LABELS), len(unions)), dtype=np.float32)
    token_windows: list[list[list[int]]] = [[[] for _ in OUTCOMES] for _ in PHONE_LABELS]
    slots = np.zeros(len(unions), dtype=np.int16)
    for row, cols in enumerate(unions):
        slots[row] = sum(16 for _ in range(5))
        center_times = centers[cols]
        for label_index, label in enumerate(PHONE_LABELS):
            whole_mask = np.zeros(len(cols), dtype=bool)
            phase_masks = [np.zeros(len(cols), dtype=bool) for _ in range(3)]
            stft_mass = np.zeros(len(cols), dtype=np.float64)
            for token in intervals[label]:
                start, end = token["start"], token["end"]
                member = (center_times >= start) & (center_times < end)
                if np.any(whole_mask & member):
                    raise AnalysisError(f"duplicate token coverage for {label}")
                whole_mask |= member
                normalized = (center_times - start) / (end - start)
                phase_masks[0] |= member & (normalized < 1.0 / 3.0)
                phase_masks[1] |= member & (normalized >= 1.0 / 3.0) & (normalized < 2.0 / 3.0)
                phase_masks[2] |= member & (normalized >= 2.0 / 3.0) & (normalized < 1.0)
                # Sensitivity: fraction of the centered 50-ms STFT support in
                # this exact phone. Out-of-recording padding remains non-phone.
                left = center_times - 0.025
                right = center_times + 0.025
                stft_mass += np.maximum(0.0, np.minimum(right, end) - np.maximum(left, start)) / 0.05
            if np.any(stft_mass > 1.0 + 1e-6):
                raise AnalysisError(f"STFT phone coverage exceeds one for {label}")
            masks = [whole_mask, *phase_masks]
            for outcome_index, mask in enumerate(masks):
                values[label_index, outcome_index, row] = np.count_nonzero(mask) / float(len(cols))
                if np.any(mask):
                    token_windows[label_index][outcome_index].append(row)
            stft[label_index, row] = float(np.mean(stft_mass, dtype=np.float64))
    if not np.allclose(values[:, 1:, :].sum(axis=1), values[:, 0, :], rtol=0, atol=1e-7):
        raise AnalysisError("early/middle/late phone weights do not partition whole-token weight")
    return values, stft, token_windows, slots


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return None
    order = np.argsort(x[good], kind="stable")
    sx, sw = x[good][order], w[good][order]
    return float(sx[np.searchsorted(np.cumsum(sw), 0.5 * sw.sum(), side="left")])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    total = float(w[good].sum(dtype=np.float64))
    return float(np.dot(x[good], w[good]) / total) if total > 0 else None


def load_dose_inputs(physical_context: dict[str, Any], direction: str, expected_map_sha: str) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    direction_record = physical_context["directions"][direction]
    map_meta = direction_record["phone_map"]
    map_path = Path(map_meta["path"]).resolve(strict=True)
    checked_file(map_path, str(map_meta["file_sha256"]), "EVAL phone map")
    if map_meta["file_sha256"] != expected_map_sha:
        raise AnalysisError("physical manifest phone-map SHA differs from the sealed WORLD context")
    phone_map = read_json(map_path)
    if phone_map.get("pair_qualified") is not True:
        raise AnalysisError("EVAL physical phone map is not pair-qualified")
    audit_meta = direction_record["parameter_transfer_audit"]
    audit_path = Path(audit_meta["path"]).resolve(strict=True)
    checked_file(audit_path, str(audit_meta["file_sha256"]), "parameter-transfer audit")
    audit = npz_arrays(audit_path)
    features: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("RT", *PHYSICAL_DOSE_ARMS):
        meta = direction_record["arms"][arm]["parameters"]
        path = Path(meta["path"]).resolve(strict=True)
        checked_file(path, str(meta["file_sha256"]), f"{arm} WORLD parameter file")
        features[arm] = npz_arrays(path)
    return phone_map, features, audit


def _band_indices(freq: np.ndarray, lo: int, hi: int, last: bool) -> np.ndarray:
    if last:
        return (freq >= lo) & (freq <= hi)
    return (freq >= lo) & (freq < hi)


def acoustic_dose_records(
    *, sample_id: str, speaker_id: str, direction: str, intervals: dict[str, list[dict[str, Any]]],
    features: dict[str, dict[str, np.ndarray]], audit: dict[str, np.ndarray],
    centers: np.ndarray, support_cols: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Describe actual arm dose on the same exact-phone RF column exposure."""
    rt = features["RT"]
    times = np.asarray(rt["time"], dtype=np.float64)
    f0_rt = np.asarray(rt["f0"], dtype=np.float64)
    sp_rt = np.asarray(rt["sp"], dtype=np.float64)
    ap_rt = np.asarray(rt["ap"], dtype=np.float64)
    freq = np.asarray(rt["frequency_hz"], dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise AnalysisError("WORLD recipient time grid is not finite and strictly increasing")
    frame_idx = nearest_indices(times, centers)
    mapped = np.asarray(audit["mapped_frames"], dtype=bool)
    joint = np.asarray(audit["joint_voiced"], dtype=bool)
    if mapped.shape != times.shape or joint.shape != times.shape:
        raise AnalysisError("transfer audit frame masks do not match recipient WORLD clock")
    normalized_rt = sp_rt / np.maximum(sp_rt.sum(axis=1, keepdims=True), 1e-30)
    rt_log_shape = np.log(np.maximum(normalized_rt, 1e-12))
    rt_tv = np.zeros(len(times), dtype=np.float64)
    rt_tv[1:] = np.mean(np.abs(np.diff(rt_log_shape, axis=0)), axis=1, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for support, col_weights in support_cols.items():
        for label in PHONE_LABELS:
            col_mask = np.zeros(len(centers), dtype=bool)
            for token in intervals[label]:
                col_mask |= (centers >= token["start"]) & (centers < token["end"])
            exposed = col_mask & (col_weights > 0)
            cols = np.flatnonzero(exposed)
            weights = col_weights[cols].astype(np.float64)
            frame_ids = frame_idx[cols]
            exact_column_weight = float(weights.sum(dtype=np.float64))
            map_ok = mapped[frame_ids] if len(frame_ids) else np.zeros(0, dtype=bool)
            joint_ok = joint[frame_ids] if len(frame_ids) else np.zeros(0, dtype=bool)
            for arm in PHYSICAL_DOSE_ARMS:
                target = features[arm]
                f0_arm = np.asarray(target["f0"], dtype=np.float64)
                sp_arm = np.asarray(target["sp"], dtype=np.float64)
                ap_arm = np.asarray(target["ap"], dtype=np.float64)
                if not (f0_arm.shape == f0_rt.shape and sp_arm.shape == sp_rt.shape and ap_arm.shape == ap_rt.shape):
                    raise AnalysisError(f"{arm} WORLD parameter shape differs from RT")
                norm_arm = sp_arm / np.maximum(sp_arm.sum(axis=1, keepdims=True), 1e-30)
                hellinger = np.sqrt(0.5 * np.sum((np.sqrt(np.maximum(norm_arm, 0.0)) - np.sqrt(np.maximum(normalized_rt, 0.0))) ** 2, axis=1, dtype=np.float64))
                both_voiced = (f0_rt > 0) & (f0_arm > 0)
                pair_voice_cols = map_ok & joint_ok & both_voiced[frame_ids] if len(frame_ids) else np.zeros(0, dtype=bool)
                cents = np.full(len(frame_ids), np.nan, dtype=np.float64)
                if len(frame_ids):
                    valid_voice = both_voiced[frame_ids]
                    cents[valid_voice] = 1200.0 * np.log2(f0_arm[frame_ids[valid_voice]] / f0_rt[frame_ids[valid_voice]])
                ap_band_all: dict[str, Any] = {}
                ap_band_joint: dict[str, Any] = {}
                for band_index, (lo, hi) in enumerate(DOSE_BANDS):
                    bins = _band_indices(freq, lo, hi, band_index == len(DOSE_BANDS) - 1)
                    delta = np.mean(np.abs(ap_arm[:, bins] - ap_rt[:, bins]), axis=1, dtype=np.float64) if np.any(bins) else np.full(len(times), np.nan)
                    ap_band_all[f"{lo}_{hi}"] = _weighted_mean(delta[frame_ids[map_ok]], weights[map_ok]) if np.any(map_ok) else None
                    ap_band_joint[f"{lo}_{hi}"] = _weighted_mean(delta[frame_ids[pair_voice_cols]], weights[pair_voice_cols]) if np.any(pair_voice_cols) else None
                # Adjacent WORLD-frame spectral dynamics are included only when
                # both centers fall in the same exact recipient phone and map.
                pair_weights = np.zeros(max(0, len(times) - 1), dtype=np.float64)
                if len(frame_ids):
                    frame_weight = np.bincount(frame_ids[map_ok], weights=weights[map_ok], minlength=len(times))
                    for token in intervals[label]:
                        in_phone = (times >= token["start"]) & (times < token["end"])
                        adjacent = in_phone[1:] & in_phone[:-1] & mapped[1:] & mapped[:-1]
                        pair_weights[adjacent] = np.minimum(frame_weight[:-1][adjacent], frame_weight[1:][adjacent])
                tv_arm = np.zeros(len(times), dtype=np.float64)
                log_arm = np.log(np.maximum(norm_arm, 1e-12))
                tv_arm[1:] = np.mean(np.abs(np.diff(log_arm, axis=0)), axis=1, dtype=np.float64)
                tv_weight = pair_weights
                tv_arm_values = tv_arm[1:]
                tv_rt_values = rt_tv[1:]
                rows.append({
                    "sample_id": sample_id, "speaker_id": speaker_id, "direction": direction,
                    "arm": arm, "support": support, "phone": label,
                    "matched_phone_token_count": sum(1 for token in intervals[label] if np.any((centers >= token["start"]) & (centers < token["end"]) & (col_weights > 0))),
                    "exposed_mel_column_window_weight": exact_column_weight,
                    "mapped_column_window_weight": float(weights[map_ok].sum(dtype=np.float64)) if len(weights) else 0.0,
                    "joint_voiced_audit_column_window_weight": float(weights[joint_ok].sum(dtype=np.float64)) if len(weights) else 0.0,
                    "mapped_column_fraction": float(weights[map_ok].sum() / exact_column_weight) if exact_column_weight else None,
                    "joint_voiced_audit_column_fraction": float(weights[joint_ok].sum() / exact_column_weight) if exact_column_weight else None,
                    "sp_shape_hellinger_mean": _weighted_mean(hellinger[frame_ids[map_ok]], weights[map_ok]) if np.any(map_ok) else None,
                    "sp_adjacent_logshape_tv_rt_mean": _weighted_mean(tv_rt_values, tv_weight),
                    "sp_adjacent_logshape_tv_arm_mean": _weighted_mean(tv_arm_values, tv_weight),
                    "sp_adjacent_logshape_tv_arm_minus_rt": (
                        _weighted_mean(tv_arm_values - tv_rt_values, tv_weight) if np.any(tv_weight > 0) else None
                    ),
                    "sp_adjacent_pair_count": int(np.count_nonzero(tv_weight)),
                    "f0_cents_signed_mean_pairvoiced": _weighted_mean(cents[pair_voice_cols], weights[pair_voice_cols]) if np.any(pair_voice_cols) else None,
                    "f0_cents_absolute_median_pairvoiced": _weighted_median(np.abs(cents[pair_voice_cols]), weights[pair_voice_cols]) if np.any(pair_voice_cols) else None,
                    "f0_pairvoiced_column_window_weight": float(weights[pair_voice_cols].sum(dtype=np.float64)) if len(weights) else 0.0,
                    "voicing_mismatch_fraction_mapped": (
                        float(np.average((f0_rt[frame_ids[map_ok]] > 0) != (f0_arm[frame_ids[map_ok]] > 0), weights=weights[map_ok]))
                        if np.any(map_ok) else None
                    ),
                    "ap_absolute_difference_by_band_all_mapped": ap_band_all,
                    "ap_absolute_difference_by_band_jointvoiced": ap_band_joint,
                    "dose_is_descriptive_not_a_randomized_phone_effect": True,
                })
    return rows


def load_context_and_local_matrices(
    *, run_dir: Path, sample_id: str, direction: str, context_rows: dict[tuple[str, str, str, str], dict[str, Any]],
    result_rows: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    baseline_row = result_rows[(sample_id, direction, "EX", "q00")]
    context_path = run_dir / baseline_row["context_npz"]
    if sha256_file(context_path) != baseline_row["context_npz_sha256"]:
        raise AnalysisError(f"context NPZ hash mismatch: {sample_id}/{direction}")
    context_meta_path = context_path.with_suffix(".json")
    context_meta = read_json(context_meta_path)
    if context_meta.get("context_npz", {}).get("sha256") != baseline_row["context_npz_sha256"]:
        raise AnalysisError(f"context sidecar does not bind context NPZ: {sample_id}/{direction}")
    arrays = npz_arrays(context_path)
    if arrays["raw_audio_embeddings_float32"].dtype != np.float32:
        raise AnalysisError("RAW SyncNet audio embeddings must be float32")

    scores: dict[str, dict[str, np.ndarray]] = {}
    saved_supports: dict[str, dict[str, np.ndarray]] = {}
    visuals: dict[str, dict[str, Any]] = {}
    base_path: Path | None = None
    for arm in ARMS:
        q00 = result_rows[(sample_id, direction, arm, "q00")]
        q10 = result_rows[(sample_id, direction, arm, "q10")]
        if q00["matrix_id"] != "V_RT__A_RAW" or q10["matrix_id"] != f"V_{arm}__A_RAW":
            raise AnalysisError(f"{arm} raw four-cell aliases do not preserve a shared RAW-A anchor")
        if q00["context_npz_sha256"] != q10["context_npz_sha256"] or q00["context_npz"] != q10["context_npz"]:
            raise AnalysisError(f"{arm} q00/q10 use different recipient contexts")
        base_score_path = run_dir / q00["score_npz"]
        arm_score_path = run_dir / q10["score_npz"]
        if sha256_file(base_score_path) != q00["score_npz_sha256"] or sha256_file(arm_score_path) != q10["score_npz_sha256"]:
            raise AnalysisError(f"score NPZ hash mismatch: {sample_id}/{direction}/{arm}")
        if base_path is None:
            base_path = base_score_path
        elif base_path != base_score_path:
            raise AnalysisError("q00 baseline visual/audio matrix is not shared across WORLD arms")
        base_arrays = npz_arrays(base_score_path)
        arm_arrays = npz_arrays(arm_score_path)
        saved_supports[arm] = {}
        for support in SUPPORTS:
            score_key = f"common_support_{support}_int32"
            if score_key not in arm_arrays:
                raise AnalysisError(f"score matrix lacks the frozen {support} support: {sample_id}/{direction}/{arm}")
            saved_supports[arm][support] = np.asarray(arm_arrays[score_key], dtype=np.int32)
        for domain in GEOMETRIES:
            key = f"cross_distance_{domain}_float32"
            zero_key = f"cross_distance_to_zero_{domain}_float32"
            if key not in arm_arrays or zero_key not in arm_arrays or key not in base_arrays or zero_key not in base_arrays:
                raise AnalysisError(f"missing full cross/zero arrays in cached {domain} matrix")
            if arm_arrays[key].dtype != np.float32 or base_arrays[key].dtype != np.float32:
                raise AnalysisError("cached distance matrices must remain float32")
            if arm_arrays[key].shape[1] != len(arrays["raw_audio_embeddings_float32"]):
                raise AnalysisError("full A_RAW distance axis differs from the stored raw audio embedding sequence")
            if arm_arrays[key].shape[0] != len(arrays[f"arm_{arm}_lip_embeddings"]):
                raise AnalysisError(f"full visual distance axis differs from stored {arm} lip embeddings")
            if base_arrays[key].shape[0] != len(arrays["arm_RT_lip_embeddings"]):
                raise AnalysisError("baseline visual distance axis differs from stored RT lip embeddings")
            scores.setdefault(arm, {})[domain] = {
                "baseline_cross": base_arrays[key], "baseline_zero": base_arrays[zero_key],
                "treated_cross": arm_arrays[key], "treated_zero": arm_arrays[zero_key],
            }
        starts = np.asarray(arrays[f"arm_{arm}_mel_starts"], dtype=np.int32)
        z = arrays[f"arm_{arm}_z"]
        mel = arrays[f"arm_{arm}_mel"]
        if not np.array_equal(starts, expected_mel_starts(mel.shape[1])):
            raise AnalysisError(f"stored {arm} mel starts differ from the exact official Wav2Lip chunk loop")
        if len(starts) != len(z) or len(arrays[f"arm_{arm}_lip_embeddings"]) != len(z) - 5:
            raise AnalysisError(f"{arm} latent-to-video-to-SyncNet clock is inconsistent")
        if arm != "EX" and not np.array_equal(starts, arrays["arm_RT_mel_starts"]):
            raise AnalysisError("WORLD arms do not share the recipient-fixed Wav2Lip mel clock")
        for row in (q00, q10):
            if row.get("common_window_count") != len(arrays["common_support_full_int32"]):
                raise AnalysisError("row common count differs from saved common query support")
            if not expected_anchor(row):
                raise AnalysisError("fixed anchor changed from CAL k=3 / column18 / official offset-3")
        for visual_id, source_arm, row in (("V_RT", "RT", q00), (f"V_{arm}", arm, q10)):
            visual_path = run_dir / row["visual_npz"]
            if sha256_file(visual_path) != row["visual_npz_sha256"]:
                raise AnalysisError(f"visual NPZ hash mismatch: {sample_id}/{direction}/{visual_id}")
            visual_data = npz_arrays(visual_path)
            if not np.array_equal(visual_data["latent_z_float32"], arrays[f"arm_{source_arm}_z"]):
                raise AnalysisError(f"visual latent differs from saved {source_arm} producer latent")
            if not np.array_equal(visual_data["lip_embeddings_float32"], arrays[f"arm_{source_arm}_lip_embeddings"]):
                raise AnalysisError(f"visual embedding differs from saved {source_arm} producer V array")
            visuals[visual_id] = visual_data
    supports: dict[str, np.ndarray] = {}
    for support in SUPPORTS:
        key = f"common_support_{support}_int32"
        supports[support] = np.asarray(arrays[key], dtype=np.int32)
        for arm in ARMS:
            if not np.array_equal(saved_supports[arm][support], supports[support]):
                raise AnalysisError(f"{support} support arrays differ across the stored common recipient clock")
    return arrays, context_meta, scores, {"visuals": visuals, "supports": supports, "saved_supports": saved_supports}


def _anchor_values(cross: np.ndarray, zero: np.ndarray, queries: np.ndarray, audio_n: int) -> np.ndarray:
    q = np.asarray(queries, dtype=np.int64)
    source = q + EXPECTED_ANCHOR["k"]
    valid = source < audio_n
    values = np.asarray(zero[q], dtype=np.float64).copy()
    values[valid] = cross[q[valid], source[valid]].astype(np.float64)
    return values


def _null_anchor_values(cross: np.ndarray, zero: np.ndarray, queries: np.ndarray, permutations: np.ndarray, audio_n: int) -> np.ndarray:
    q = np.asarray(queries, dtype=np.int64)
    source = q + EXPECTED_ANCHOR["k"]
    valid = source < audio_n
    values = np.broadcast_to(np.asarray(zero[q], dtype=np.float64)[None, :], (len(permutations), len(q))).copy()
    if np.any(valid):
        audio_indices = permutations[:, source[valid]]
        query_indices = np.broadcast_to(q[valid][None, :], audio_indices.shape)
        values[:, valid] = cross[query_indices, audio_indices].astype(np.float64)
    return values


def _bootstrap_family(endpoint_records: list[dict[str, Any]], draws: int = BOOTSTRAP_DRAWS) -> tuple[float | None, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, EXPECTED_SPEAKERS, size=(draws, EXPECTED_SPEAKERS), dtype=np.int16)
    counts = np.zeros((draws, EXPECTED_SPEAKERS), dtype=np.uint8)
    for row in range(draws):
        counts[row] = np.bincount(indices[row], minlength=EXPECTED_SPEAKERS).astype(np.uint8)
    counts64 = counts.astype(np.float64)
    max_t = np.zeros(draws, dtype=np.float64)
    std_errors: list[float] = []
    estimates: list[float] = []
    complete_records: list[dict[str, Any]] = []
    for record in endpoint_records:
        speaker_values = np.asarray(record["speaker_values"], dtype=np.float64)
        if speaker_values.shape != (EXPECTED_SPEAKERS,) or not np.isfinite(speaker_values).all():
            record["status"] = "INCOMPLETE_SUPPORT"
            continue
        estimate = float(np.mean(speaker_values, dtype=np.float64))
        boot = counts64 @ speaker_values / EXPECTED_SPEAKERS
        se = float(np.std(boot, ddof=1, dtype=np.float64))
        if se > 0:
            max_t = np.maximum(max_t, np.abs((boot - estimate) / se))
        elif abs(estimate) > 1e-15:
            # A constant nonzero endpoint has a degenerate bootstrap interval.
            pass
        record["estimate_speaker_equal_mean"] = estimate
        record["bootstrap_se"] = se
        std_errors.append(se)
        estimates.append(estimate)
        complete_records.append(record)
    if not complete_records:
        return None, max_t
    critical = float(np.quantile(max_t, 0.99, method="linear"))
    for record in complete_records:
        estimate = record["estimate_speaker_equal_mean"]
        se = record["bootstrap_se"]
        record["ci99_simultaneous_max_t"] = [float(estimate - critical * se), float(estimate + critical * se)]
        record["simultaneous_family_critical_99"] = critical
        record.pop("speaker_values", None)
    return critical, max_t


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    sealed = preflight(args)
    out: Path = sealed["output_dir"]
    out.mkdir(parents=True, exist_ok=False)
    cache_dir = out / "local_windows"
    cache_dir.mkdir()
    rows_by_context: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    results_by_context: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in sealed["rows"]:
        if row.get("family") == "raw" and row.get("cell") in {"q00", "q10"}:
            rows_by_context[(row["sample_id"], row["direction"], row["condition"], row["cell"])] = row
    for row in sealed["results"]:
        if row.get("family") == "raw" and row.get("cell") in {"q00", "q10"}:
            results_by_context[(row["sample_id"], row["direction"], row["condition"], row["cell"])] = row

    endpoint_clips: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    intended_missing: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    local_dose_path = out / "local_acoustic_dose.jsonl"
    cache_inventory: dict[str, str] = {}
    context_records: list[dict[str, Any]] = []
    with local_dose_path.open("w", encoding="utf-8", newline="\n") as dose_out:
        for sample_id, direction in sorted({(row["sample_id"], row["direction"]) for row in sealed["rows"]}):
            rec_rows = {(s, d, arm, cell): row for (s, d, arm, cell), row in results_by_context.items() if s == sample_id and d == direction}
            arrays, context_meta, scores, extras = load_context_and_local_matrices(
                run_dir=sealed["run_dir"], sample_id=sample_id, direction=direction,
                context_rows=rows_by_context, result_rows=rec_rows,
            )
            example = results_by_context[(sample_id, direction, "EX", "q00")]
            speaker_id = str(example["speaker_id"])
            physical_context = sealed["physical_records"][sample_id]
            physical_direction = physical_context["directions"][direction]
            if physical_context.get("speaker") != speaker_id:
                raise AnalysisError(f"physical and score row speaker differ: {sample_id}")
            if context_meta.get("mapping_sha256") != physical_direction.get("phone_map", {}).get("file_sha256"):
                raise AnalysisError(f"physical phone-map hash differs from the sealed WORLD context: {sample_id}/{direction}")
            phone_map, parameters, transfer_audit = load_dose_inputs(
                physical_context, direction, context_meta["mapping_sha256"],
            )
            intervals = _phone_intervals(phone_map)
            starts = np.asarray(arrays["arm_RT_mel_starts"], dtype=np.int32)
            mel = np.asarray(arrays["arm_RT_mel"], dtype=np.float32)
            n_query = len(arrays["common_support_full_int32"])
            unions, indptr, flat_cols = window_unions(starts, n_query, mel.shape[1])
            weights, stft_weights, token_windows, slots = phone_weights(unions, mel.shape[1], intervals)
            supports = extras["supports"]
            support_cols: dict[str, np.ndarray] = {}
            for support, query_ids in supports.items():
                mult = np.zeros(mel.shape[1], dtype=np.int32)
                for index in query_ids:
                    np.add.at(mult, unions[int(index)], 1)
                support_cols[support] = mult

            n_audio = len(arrays["raw_audio_embeddings_float32"])
            rng = np.random.Generator(np.random.PCG64(NULL_SEED_BASE + int(example["ordered_eval_index"])))
            permutations = np.stack([rng.permutation(n_audio) for _ in range(NULL_DRAWS)], axis=0).astype(np.int64, copy=False)
            g_arrays: dict[str, np.ndarray] = {}
            null_sha: dict[str, str] = {}
            for arm in ARMS:
                for domain in GEOMETRIES:
                    matrix = scores[arm][domain]
                    base = _anchor_values(matrix["baseline_cross"], matrix["baseline_zero"], supports["full"], n_audio)
                    treated = _anchor_values(matrix["treated_cross"], matrix["treated_zero"], supports["full"], n_audio)
                    g = (base - treated).astype(np.float32)
                    g_arrays[f"g_{arm}_{domain}"] = g
                    null_sha[f"{arm}_{domain}"] = array_sha256(permutations)
                    for support, query_ids in supports.items():
                        q = np.asarray(query_ids, dtype=np.int64)
                        if q.size == 0:
                            for label_index, label in enumerate(PHONE_LABELS):
                                for outcome_index, outcome in enumerate(OUTCOMES):
                                    key = (direction, arm, domain, support, label, outcome)
                                    intended_missing[key].append({"sample_id": sample_id, "speaker_id": speaker_id, "reason": "empty_frozen_support"})
                            continue
                        base_null = _null_anchor_values(matrix["baseline_cross"], matrix["baseline_zero"], q, permutations, n_audio)
                        treated_null = _null_anchor_values(matrix["treated_cross"], matrix["treated_zero"], q, permutations, n_audio)
                        null_g = (base_null - treated_null).astype(np.float64)
                        obs_g = g[q].astype(np.float64)
                        for label_index, label in enumerate(PHONE_LABELS):
                            for outcome_index, outcome in enumerate(OUTCOMES):
                                w = weights[label_index, outcome_index, q].astype(np.float64)
                                denom = float(len(q))
                                obs_value = float(np.dot(w, obs_g) / denom)
                                null_draws = np.sum(null_g * w[None, :], axis=1, dtype=np.float64) / denom
                                exposure = float(w.sum(dtype=np.float64) / denom)
                                exposed_windows = int(np.count_nonzero(w > 0))
                                conditional = float(np.dot(w, obs_g) / w.sum(dtype=np.float64)) if w.sum(dtype=np.float64) > 0 else None
                                # The distinct-token support is computed by exact
                                # recipient interval and the saved RF columns.
                                cols_for_support = support_cols[support] > 0
                                token_count = 0
                                for token in intervals[label]:
                                    member = (np.arange(mel.shape[1], dtype=np.float64) / 80.0 >= token["start"]) & (np.arange(mel.shape[1], dtype=np.float64) / 80.0 < token["end"])
                                    if np.any(member & cols_for_support):
                                        token_count += 1
                                endpoint_key = (direction, arm, domain, support, label, outcome)
                                endpoint_clips[endpoint_key][sample_id] = {
                                    "speaker_id": speaker_id,
                                    "observed": obs_value,
                                    "null_draws": null_draws,
                                    "null_mean": float(np.mean(null_draws, dtype=np.float64)),
                                    "exposure_fraction_over_support_windows": exposure,
                                    "exposed_support_windows": exposed_windows,
                                    "matched_phone_tokens_with_rf_exposure": token_count,
                                    "conditional_token_mean": conditional,
                                    "intended_window_count": int(len(q)),
                                }
            for dose_row in acoustic_dose_records(
                sample_id=sample_id, speaker_id=speaker_id, direction=direction,
                intervals=intervals, features=parameters, audit=transfer_audit,
                centers=np.arange(mel.shape[1], dtype=np.float64) / 80.0,
                support_cols=support_cols,
            ):
                dose_out.write(json.dumps(dose_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

            cache_path = cache_dir / f"{sample_id}__{direction}.npz"
            cache_arrays = {
                "phone_labels_unicode": np.asarray(PHONE_LABELS, dtype="U8"),
                "outcome_names_unicode": np.asarray(OUTCOMES, dtype="U8"),
                "supports_full_int32": supports["full"],
                "supports_valid_int32": supports["valid"],
                "supports_guard20_int32": supports["guard20"],
                "supports_guard15_int32": supports["guard15"],
                "phone_rf_weights_float32": weights,
                "phone_stft_sensitivity_weights_float32": stft_weights,
                "window_union_indptr_int32": indptr,
                "window_union_columns_int32": flat_cols,
                "syncnet_pre_dedup_mel_slot_count_int16": slots,
                "null_permutations_int64": permutations,
                **g_arrays,
            }
            np.savez_compressed(cache_path, **cache_arrays)
            cache_inventory[str(cache_path.relative_to(out)).replace("\\", "/")] = sha256_file(cache_path)
            context_records.append({
                "sample_id": sample_id, "speaker_id": speaker_id, "direction": direction,
                "ordered_eval_index": int(example["ordered_eval_index"]),
                "query_count": int(n_query), "raw_audio_domain_count": int(n_audio),
                "mel_frames": int(mel.shape[1]), "video_window_count": int(len(unions)),
                "union_column_count_total": int(len(flat_cols)), "pre_dedup_mel_slots_per_window": 80,
                "permutation_seed": NULL_SEED_BASE + int(example["ordered_eval_index"]),
                "permutation_sha256": array_sha256(permutations),
                "phone_map_sha256": physical_direction["phone_map"]["file_sha256"],
                "context_npz_sha256": example["context_npz_sha256"],
                "cache_npz": str(cache_path.relative_to(out)).replace("\\", "/"),
                "cache_sha256": cache_inventory[str(cache_path.relative_to(out)).replace("\\", "/")],
                "support_counts": {key: int(value.size) for key, value in supports.items()},
            })
    if len(context_records) != EXPECTED_CONTEXTS:
        raise AnalysisError(f"local response contexts are {len(context_records)}, expected 160")

    endpoints: list[dict[str, Any]] = []
    intended_endpoint_count = len(ARMS) * 2 * len(GEOMETRIES) * len(SUPPORTS) * len(PHONE_LABELS) * len(OUTCOMES) * 2
    for key in sorted(set(endpoint_clips) | set(intended_missing), key=lambda value: tuple(str(x) for x in value)):
        direction, arm, geometry, support, label, outcome = key
        by_sample = endpoint_clips.get(key, {})
        missing = list(intended_missing.get(key, []))
        expected_samples = {sample_id for sample_id, d in {(row["sample_id"], row["direction"]) for row in sealed["rows"]} if d == direction}
        missing.extend({"sample_id": sid, "reason": "missing_local_window_record"} for sid in sorted(expected_samples - set(by_sample)))
        if missing or len(by_sample) != EXPECTED_RECIPIENTS:
            endpoints.extend([
                {"direction": direction, "arm": arm, "geometry": geometry, "support": support, "phone": label, "outcome": outcome, "estimand": est, "status": "INCOMPLETE_SUPPORT", "complete_clip_count": len(by_sample), "expected_clip_count": EXPECTED_RECIPIENTS, "missing_examples": missing[:20]}
                for est in ("observed", "time_excess")
            ])
            continue
        by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in by_sample.values():
            by_speaker[value["speaker_id"]].append(value)
        if len(by_speaker) != EXPECTED_SPEAKERS or any(len(items) != 2 for items in by_speaker.values()):
            endpoints.extend([
                {"direction": direction, "arm": arm, "geometry": geometry, "support": support, "phone": label, "outcome": outcome, "estimand": est, "status": "INCOMPLETE_SPEAKER_BLOCK", "complete_speaker_count": sum(len(x) == 2 for x in by_speaker.values()), "expected_speaker_count": EXPECTED_SPEAKERS}
                for est in ("observed", "time_excess")
            ])
            continue
        speakers = sorted(by_speaker)
        obs_speaker = np.asarray([np.mean([item["observed"] for item in by_speaker[speaker]], dtype=np.float64) for speaker in speakers], dtype=np.float64)
        null_by_speaker = np.stack([
            np.mean(np.stack([item["null_draws"] for item in by_speaker[speaker]], axis=0), axis=0, dtype=np.float64)
            for speaker in speakers
        ], axis=0)
        excess_speaker = obs_speaker - null_by_speaker.mean(axis=1, dtype=np.float64)
        null_mean = float(null_by_speaker.mean(dtype=np.float64))
        summaries = {
            "observed": {
                "speaker_values": obs_speaker,
                "null_mean_over_100_full_A_permutations": null_mean,
                "exposure_fraction_mean": float(np.mean([v["exposure_fraction_over_support_windows"] for v in by_sample.values()])),
                "exposed_clip_count": int(sum(any(item["exposed_support_windows"] > 0 for item in by_speaker[speaker]) for speaker in speakers)),
                "exposed_speaker_count": int(sum(any(item["exposed_support_windows"] > 0 for item in by_speaker[speaker]) for speaker in speakers)),
                "conditional_token_mean_exposed_clip_mean": float(np.mean([v["conditional_token_mean"] for v in by_sample.values() if v["conditional_token_mean"] is not None])) if any(v["conditional_token_mean"] is not None for v in by_sample.values()) else None,
            },
            "time_excess": {
                "speaker_values": excess_speaker,
                "null_mean_over_100_full_A_permutations": null_mean,
                "exposure_fraction_mean": float(np.mean([v["exposure_fraction_over_support_windows"] for v in by_sample.values()])),
                "exposed_clip_count": int(sum(any(item["exposed_support_windows"] > 0 for item in by_speaker[speaker]) for speaker in speakers)),
                "exposed_speaker_count": int(sum(any(item["exposed_support_windows"] > 0 for item in by_speaker[speaker]) for speaker in speakers)),
            },
        }
        for estimand, values in summaries.items():
            endpoints.append({
                "direction": direction, "arm": arm, "arm_stratum": "primary" if arm in {"EX", "SP", "BOTH", "F0", "AP"} else "control",
                "geometry": geometry, "support": support, "phone": label, "outcome": outcome,
                "estimand": estimand, "status": "pending_simultaneous_interval", **values,
            })

    inferential = [row for row in endpoints if row.get("status") == "pending_simultaneous_interval"]
    critical, _ = _bootstrap_family(inferential)
    for row in inferential:
        row["status"] = "ok"
        row["speaker_count"] = EXPECTED_SPEAKERS
        row["clip_count"] = EXPECTED_RECIPIENTS
        row["bootstrap_seed"] = BOOTSTRAP_SEED
        row["bootstrap_draws"] = BOOTSTRAP_DRAWS
        row["family_size_planned"] = intended_endpoint_count
        row["family_size_complete"] = len(inferential)
        row["family_critical_99_max_t"] = critical
        row.pop("speaker_values", None)
    local_meta = {
        "schema_version": 1,
        "status": "COMPLETE" if len(inferential) == intended_endpoint_count else "INCOMPLETE_SUPPORT",
        "estimand": "fixed-anchor per-window visual distance response G_i = D(V_RT_i,A_RAW[i+3]) - D(V_X_i,A_RAW[i+3]); positive means X visual is closer to unchanged recipient RAW audio",
        "score_anchor": EXPECTED_ANCHOR,
        "score_clock_seconds": "WORLD/SyncNet local response uses exact score row i and RAW-A i+3; CAL MediaPipe -85ms latency is not applied",
        "cal_phone_inventory_sha256": sealed["inventory_sha256"],
        "phone_labels": list(PHONE_LABELS),
        "mel_receptive_field": "five 25-fps video frames; set-union of their five stored 16-column Wav2Lip mel chunks; exact column center m/80 seconds; final Wav2Lip fallback start is verified",
        "stft": {"n_fft": 800, "win_length": 800, "hop_length": 200, "window": "Hann", "center": True, "pad_mode": "constant", "sensitivity": "50-ms centered waveform support overlap; descriptive only"},
        "primary_local_contribution": "sum_i(w_i*G_i)/frozen_support_n; absent phone is zero-weight in the fixed denominator, not a null gate",
        "time_null": "100 full-domain permutations of the complete RAW-A embedding sequence; PCG64(20260930+ordered_eval_index); one permutation shared by RT/X and all local labels within context; out-of-A i+3 uses the scorer's saved zero-distance row",
        "inference": "equal two clips within speaker; equal weight 40 speakers; one shared 20,000x40 speaker bootstrap; two-sided 99% studentized max-|t| interval over all arms, recipient directions, geometries, supports, 16 phones, whole/three phases, observed and time-excess",
        "dose": "co-located descriptive WORLD parameter summaries; not a randomized phone-specific effect or mediation estimate",
        "input_seals": {key: value for key, value in sealed.items() if key.endswith("sha256") or key in {"planned_row_count", "result_row_count", "context_count"}},
        "context_count": len(context_records),
        "local_window_cache_count": len(cache_inventory),
        "phone_dose_rows": sum(1 for _ in local_dose_path.open("r", encoding="utf-8")),
        "simultaneous_family_endpoint_count": len(inferential),
        "planned_family_endpoint_count": intended_endpoint_count,
        "simultaneous_max_t_critical_99": critical,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutation_seed_rule": "PCG64(20260930 + ordered_eval_index), 100 successive permutation(n_RAW_A) calls",
        "independent_analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "no_producer_scoring_imports": True,
    }
    (out / "local_context_inventory.json").write_text(json.dumps(context_records, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "local_response_summary.json").write_text(json.dumps({"metadata": local_meta, "endpoints": endpoints}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_files_sha256.json":
            output_files[str(path.relative_to(out)).replace("\\", "/")] = sha256_file(path)
    (out / "output_files_sha256.json").write_text(json.dumps({"files": output_files}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": local_meta["status"], "output_dir": str(out), "summary_sha256": sha256_file(out / "local_response_summary.json"), "summary_path": str(out / "local_response_summary.json"), "endpoint_count": len(endpoints), "complete_endpoint_count": len(inferential), "context_count": len(context_records)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--physical-manifest", required=True, type=Path)
    parser.add_argument("--physical-qa", required=True, type=Path)
    parser.add_argument("--consumer-audit", required=True, type=Path)
    parser.add_argument("--consumer-statistics", required=True, type=Path)
    parser.add_argument("--phone-inventory", type=Path, default=Path(__file__).resolve().parents[2] / "openspec/changes/diagnose-world-local-acoustic-response-20261001/cal_phone_inventory.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="validate seals and intended row coverage without loading score arrays")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sealed = preflight(args)
        if args.preflight_only:
            print(json.dumps({"status": "PREFLIGHT_PASS", "run_dir": str(sealed["run_dir"]), "protocol_sha256": sealed["protocol_sha256"], "row_manifest_sha256": sealed["row_manifest_sha256"], "context_count": sealed["context_count"], "score_rows": sealed["result_row_count"], "output_dir": str(sealed["output_dir"])}, sort_keys=True))
            return 0
        result = analyze(args)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
