#!/usr/bin/env python3
"""Repair only the P1 full-support SyncNet distances from saved features.

The official distance worker truncates both modalities to ``min(A, V)`` before
padding audio.  This utility recomputes the full-support rows from the complete
saved audio and visual embedding arrays, using the same float32 Torch
pairwise-distance operation and epsilon.  It never loads a SyncNet model or
changes the primary guard20 fields in the source score document.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[3]
LAG_RADIUS = 15
LAG_COUNT = 31
EXPECTED_CONDITIONS = (
    "RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2",
    "DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6",
)
FULL_FIELDS = (
    "distance_path_full", "distance_sha256_full", "distance_shape_full", "q_support_full",
    "curve31_full", "C_full", "B_full", "Dmin_full", "D0_full", "bestlag_full",
    "search_gain_full", "anchor_full", "full_context_overlay",
)


class OverlayError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)


def _load_npy(
    path_value: Any,
    expected_sha: Any,
    label: str,
    expected_dtype: Any = np.float32,
) -> np.ndarray:
    if not isinstance(path_value, str) or not path_value:
        raise OverlayError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_file():
        raise OverlayError(f"{label} file is missing: {path}")
    if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
        raise OverlayError(f"{label} file hash mismatch: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise OverlayError(f"{label} NPY cannot be loaded: {path}: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise OverlayError(f"{label} must be a nonempty rank-2 array: {path}")
    expected_dtype = np.dtype(expected_dtype)
    if value.dtype != expected_dtype or not np.isfinite(value).all():
        raise OverlayError(f"{label} must be finite {expected_dtype}: {path}")
    return value


def full_context_distances(audio_features: np.ndarray, visual_features: np.ndarray, q_indices: Iterable[int], lag_radius: int = LAG_RADIUS) -> np.ndarray:
    """Compute exact full-context SyncNet pairwise distances for selected q rows."""

    audio = np.asarray(audio_features)
    visual = np.asarray(visual_features)
    queries = [int(q) for q in q_indices]
    if audio.ndim != 2 or visual.ndim != 2 or audio.dtype != np.float32 or visual.dtype != np.float32:
        raise OverlayError("audio/visual features must be rank-2 float32 arrays")
    if audio.shape[1] != visual.shape[1] or audio.shape[1] < 1 or not np.isfinite(audio).all() or not np.isfinite(visual).all():
        raise OverlayError("audio/visual embedding dimensions or values are invalid")
    if not queries or len(set(queries)) != len(queries):
        raise OverlayError("full q support must be a nonempty unique list")
    if lag_radius < 0:
        raise OverlayError("lag radius must be nonnegative")
    lags = np.arange(-lag_radius, lag_radius + 1, dtype=np.int64)
    if any(q < 0 or q >= visual.shape[0] or q + int(lags[0]) < 0 or q + int(lags[-1]) >= audio.shape[0] for q in queries):
        raise OverlayError("a requested q row lacks real visual or audio context")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on supported runtime
        raise OverlayError("Torch is required for the frozen float32 pairwise-distance operation") from exc

    torch.set_num_threads(1)
    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio))
    visual_tensor = torch.from_numpy(np.ascontiguousarray(visual))
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for q in queries:
            indices = torch.arange(q - lag_radius, q + lag_radius + 1, dtype=torch.long)
            left = visual_tensor[q : q + 1].expand(2 * lag_radius + 1, -1)
            right = audio_tensor.index_select(0, indices)
            row = torch.nn.functional.pairwise_distance(left, right, eps=1e-6)
            rows.append(row.numpy().astype(np.float64))
    matrix = np.stack(rows, axis=0)
    if matrix.shape != (len(queries), 2 * lag_radius + 1) or not np.isfinite(matrix).all():
        raise OverlayError(f"full-context distance matrix is invalid: {matrix.shape}")
    return matrix


def curve_metrics(curve: np.ndarray, anchor_index: int) -> dict[str, Any]:
    value = np.asarray(curve, dtype=np.float64)
    if value.shape != (LAG_COUNT,) or not np.isfinite(value).all() or not 0 <= int(anchor_index) < LAG_COUNT:
        raise OverlayError("full-context curve or fixed anchor index is invalid")
    best = int(np.argmin(value))
    median = float(np.median(value))
    minimum = float(np.min(value))
    zero = float(value[LAG_RADIUS])
    anchor = float(value[int(anchor_index)])
    return {
        "B": median,
        "Dmin": minimum,
        "D0": zero,
        "C": median - minimum,
        "bestlag": best - LAG_RADIUS,
        "argmin_index": best,
        "offset_at_lag_boundary": best in (0, LAG_COUNT - 1),
        "search_gain": zero - minimum,
        "anchor_D": anchor,
        "anchor_C": median - anchor,
        "anchor_lag_index": int(anchor_index),
        "anchor_lag_frames": int(anchor_index - LAG_RADIUS),
    }


def _condition_map(speaker: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = speaker.get("conditions")
    if not isinstance(rows, list):
        raise OverlayError(f"{speaker.get('speaker_id')}: conditions is not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("condition_id"), str):
            raise OverlayError(f"{speaker.get('speaker_id')}: invalid condition row")
        condition_id = row["condition_id"]
        if condition_id in result:
            raise OverlayError(f"{speaker.get('speaker_id')}: duplicate condition {condition_id}")
        result[condition_id] = row
    return result


def _validate_support(speaker: dict[str, Any], conditions: dict[str, dict[str, Any]]) -> tuple[list[int], list[int]]:
    support = speaker.get("support")
    if not isinstance(support, dict) or support.get("status") != "PASS":
        raise OverlayError(f"{speaker.get('speaker_id')}: common support is not PASS")
    primary = support.get("primary_guard20_q_indices")
    full = support.get("full_q_indices")
    if not isinstance(primary, list) or not primary or not isinstance(full, list) or not full:
        raise OverlayError(f"{speaker.get('speaker_id')}: primary/full support is empty")
    primary = [int(q) for q in primary]
    full = [int(q) for q in full]
    if primary != sorted(set(primary)) or full != sorted(set(full)) or not set(primary).issubset(full):
        raise OverlayError(f"{speaker.get('speaker_id')}: support indices are not ordered unique nested sets")
    audio_count = support.get("audio_window_count")
    counts = support.get("visual_window_counts")
    if not isinstance(audio_count, int) or not isinstance(counts, dict) or set(counts) != set(EXPECTED_CONDITIONS):
        raise OverlayError(f"{speaker.get('speaker_id')}: common-support length metadata is incomplete")
    if any(not isinstance(count, int) or count < 1 for count in counts.values()):
        raise OverlayError(f"{speaker.get('speaker_id')}: visual window counts are invalid")
    limit = min(audio_count, *counts.values())
    expected_primary = list(range(20, limit - 20))
    expected_full = list(range(LAG_RADIUS, min(min(counts.values()), audio_count - LAG_RADIUS)))
    if primary != expected_primary or full != expected_full:
        raise OverlayError(f"{speaker.get('speaker_id')}: q support differs from the frozen guard20/full formulas")
    for condition_id, row in conditions.items():
        if condition_id not in EXPECTED_CONDITIONS:
            raise OverlayError(f"unexpected condition {condition_id}")
        if row.get("q_support") != primary or row.get("q_support_full") != full:
            raise OverlayError(f"{speaker.get('speaker_id')}/{condition_id}: row support differs from speaker support")
        if row.get("lags") != list(range(-LAG_RADIUS, LAG_RADIUS + 1)):
            raise OverlayError(f"{speaker.get('speaker_id')}/{condition_id}: lag order differs from the frozen 31-column contract")
        visual_count = counts.get(condition_id)
        if not isinstance(visual_count, int) or not full or full[-1] >= visual_count or full[0] - LAG_RADIUS < 0 or full[-1] + LAG_RADIUS >= audio_count:
            raise OverlayError(f"{speaker.get('speaker_id')}/{condition_id}: full support includes non-real context")
    return primary, full


def _save_npy_new(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)
    return sha256_file(path)


def _validate_primary_unchanged(original: dict[str, Any], derived: dict[str, Any]) -> None:
    old_speakers = {row["speaker_id"]: row for row in original["speakers"]}
    new_speakers = {row["speaker_id"]: row for row in derived["speakers"]}
    if old_speakers.keys() != new_speakers.keys():
        raise OverlayError("derived score document changed the planned speaker records")
    for sid, old_speaker in old_speakers.items():
        old_conditions = _condition_map(old_speaker)
        new_conditions = _condition_map(new_speakers[sid])
        if old_conditions.keys() != new_conditions.keys():
            raise OverlayError(f"derived score document changed condition rows for {sid}")
        for condition_id, old_row in old_conditions.items():
            new_row = new_conditions[condition_id]
            if any(field not in new_row or old_row[field] != new_row[field] for field in old_row if field not in FULL_FIELDS):
                raise OverlayError(f"non-full score fields changed for {sid}/{condition_id}")
            if any(field not in new_row for field in old_row):
                raise OverlayError(f"score fields were removed for {sid}/{condition_id}")


def build_overlay(score_path: Path, output_root: Path, selected_speakers: set[str] | None = None) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only overlay output already exists: {output_root}")
    score_raw = score_path.read_bytes()
    source_sha = hashlib.sha256(score_raw).hexdigest()
    try:
        source = json.loads(score_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverlayError(f"score_results input is not valid UTF-8 JSON: {score_path}") from exc
    if source.get("schema_version") != 1 or not isinstance(source.get("protocol_sha256"), str) or not isinstance(source.get("planned_speakers"), list):
        raise OverlayError("source score document does not match the frozen P1 score schema")
    if len(set(source["planned_speakers"])) != len(source["planned_speakers"]):
        raise OverlayError("source planned speaker list contains duplicates")
    if not isinstance(source.get("planned_conditions"), list) or tuple(row.get("condition_id") for row in source["planned_conditions"]) != EXPECTED_CONDITIONS:
        raise OverlayError("source planned conditions do not match the frozen eight-cell order")
    speakers = source.get("speakers")
    if not isinstance(speakers, list) or len(speakers) != len(source["planned_speakers"]):
        raise OverlayError("source score rows do not preserve the planned speaker denominator")
    by_sid = {row.get("speaker_id"): row for row in speakers if isinstance(row, dict)}
    if set(by_sid) != set(source["planned_speakers"]):
        raise OverlayError("source score speaker rows do not match the planned roster exactly")
    if selected_speakers is not None and not selected_speakers.issubset(set(by_sid)):
        raise OverlayError(f"requested speakers are absent: {sorted(selected_speakers - set(by_sid))}")

    output_root.mkdir(parents=True, exist_ok=False)
    corrected = copy.deepcopy(source)
    derived_by_sid = {row["speaker_id"]: row for row in corrected["speakers"]}
    artifacts: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    targeted = set(selected_speakers) if selected_speakers is not None else set(by_sid)

    for sid in source["planned_speakers"]:
        if sid not in targeted:
            continue
        source_speaker = by_sid[sid]
        source_conditions = _condition_map(source_speaker)
        if tuple(source_conditions) != EXPECTED_CONDITIONS:
            raise OverlayError(f"{sid}: condition order differs from frozen plan")
        derived_conditions = _condition_map(derived_by_sid[sid])
        support = source_speaker.get("support", {})
        if support.get("status") != "PASS":
            skipped.append({"speaker_id": sid, "reason": f"support_status={support.get('status')}"})
            continue
        primary_q, full_q = _validate_support(source_speaker, source_conditions)
        for condition_id in EXPECTED_CONDITIONS:
            row = source_conditions[condition_id]
            if row.get("status") != "SCORED":
                skipped.append({"speaker_id": sid, "condition_id": condition_id, "reason": f"score_status={row.get('status')}"})
                continue
            audio_features = _load_npy(row.get("audio_feature_path"), row.get("audio_feature_sha256"), f"{sid}/{condition_id} audio features")
            visual_features = _load_npy(row.get("feature_path"), row.get("feature_sha256"), f"{sid}/{condition_id} visual features")
            if audio_features.shape[0] != support["audio_window_count"] or visual_features.shape[0] != support["visual_window_counts"][condition_id]:
                raise OverlayError(f"{sid}/{condition_id}: saved feature lengths differ from frozen support metadata")
            old_full = _load_npy(
                row.get("distance_path_full"),
                row.get("distance_sha256_full"),
                f"{sid}/{condition_id} original full distance",
                expected_dtype=np.float64,
            )
            if old_full.shape != (len(full_q), LAG_COUNT) or row.get("distance_shape_full") != list(old_full.shape):
                raise OverlayError(f"{sid}/{condition_id}: original full distance shape/support mismatch")
            old_primary = _load_npy(
                row.get("distance_path"),
                row.get("distance_sha256"),
                f"{sid}/{condition_id} primary distance",
                expected_dtype=np.float64,
            )
            if old_primary.shape != (len(primary_q), LAG_COUNT) or row.get("distance_shape") != list(old_primary.shape):
                raise OverlayError(f"{sid}/{condition_id}: primary distance shape/support mismatch")

            full = full_context_distances(audio_features, visual_features, full_q)
            if full.shape != old_full.shape:
                raise OverlayError(f"{sid}/{condition_id}: corrected full distance shape changed")
            q_to_full = {q: index for index, q in enumerate(full_q)}
            primary_overlap = np.asarray([full[q_to_full[q]] - old_primary[index] for index, q in enumerate(primary_q) if q in q_to_full])
            if primary_overlap.size and float(np.max(np.abs(primary_overlap))) > 1e-4:
                raise OverlayError(f"{sid}/{condition_id}: recomputed features disagree with original primary distances beyond 1e-4")

            min_context = min(audio_features.shape[0], visual_features.shape[0])
            unaffected_deltas: list[float] = []
            tail_positions: list[list[int]] = []
            for row_index, q in enumerate(full_q):
                for col_index, lag in enumerate(range(-LAG_RADIUS, LAG_RADIUS + 1)):
                    audio_index = q + lag
                    if audio_index >= min_context:
                        tail_positions.append([row_index, col_index])
                    else:
                        unaffected_deltas.append(float(abs(full[row_index, col_index] - old_full[row_index, col_index])))
            max_unaffected_delta = max(unaffected_deltas, default=0.0)
            if max_unaffected_delta > 1e-4:
                raise OverlayError(f"{sid}/{condition_id}: overlay changes non-truncated contexts (max={max_unaffected_delta:.9g})")
            diff = np.abs(full - old_full)
            curve = full.mean(axis=0, dtype=np.float64)
            raw_curve = np.asarray(source_conditions["RAW_R1"].get("curve31"), dtype=np.float64)
            anchor_obj = source_conditions["RAW_R1"].get("anchor")
            if raw_curve.shape != (LAG_COUNT,) or not isinstance(anchor_obj, dict):
                raise OverlayError(f"{sid}: original RAW_R1 primary anchor is missing")
            anchor_index = int(anchor_obj.get("fixed_anchor_lag_index", -1))
            if anchor_index != int(np.argmin(raw_curve)):
                raise OverlayError(f"{sid}: frozen RAW_R1 guard20 anchor does not match its argmin")
            metrics = curve_metrics(curve, anchor_index)
            out_path = output_root / "distances" / sid / f"{condition_id}_full_context.npy"
            out_sha = _save_npy_new(out_path, full)
            derived_row = derived_conditions[condition_id]
            derived_row.update({
                "distance_path_full": str(out_path.resolve()),
                "distance_sha256_full": out_sha,
                "distance_shape_full": list(full.shape),
                "q_support_full": list(full_q),
                "curve31_full": curve.tolist(),
                "C_full": metrics["C"], "B_full": metrics["B"], "Dmin_full": metrics["Dmin"],
                "D0_full": metrics["D0"], "bestlag_full": metrics["bestlag"],
                "search_gain_full": metrics["search_gain"],
                "anchor_full": {
                    "D": metrics["anchor_D"], "C": metrics["anchor_C"],
                    "bestlag": metrics["bestlag"], "search_gain": metrics["search_gain"],
                    "fixed_anchor_lag_index": anchor_index,
                    "fixed_anchor_lag_frames": anchor_index - LAG_RADIUS,
                    "reference_condition": "RAW_R1",
                },
                "full_context_overlay": {
                    "schema": "p1_full_context_distance_overlay_v1",
                    "source_distance_path_full": row["distance_path_full"],
                    "source_distance_sha256_full": row["distance_sha256_full"],
                    "corrected_distance_sha256_full": out_sha,
                    "audio_feature_sha256": row["audio_feature_sha256"],
                    "visual_feature_sha256": row["feature_sha256"],
                    "max_abs_delta_in_unaffected_contexts": max_unaffected_delta,
                    "max_abs_delta_all_cells": float(np.max(diff)) if diff.size else 0.0,
                    "changed_cell_count_gt_1e-5": int(np.count_nonzero(diff > 1e-5)),
                    "expected_audio_tail_cell_count": len(tail_positions),
                },
            })
            artifacts.append({
                "speaker_id": sid, "condition_id": condition_id,
                "audio_feature_path": row["audio_feature_path"], "audio_feature_sha256": row["audio_feature_sha256"],
                "audio_feature_shape": list(audio_features.shape),
                "visual_feature_path": row["feature_path"], "visual_feature_sha256": row["feature_sha256"],
                "visual_feature_shape": list(visual_features.shape),
                "source_distance_path_full": row["distance_path_full"],
                "source_distance_sha256_full": row["distance_sha256_full"],
                "source_distance_shape_full": list(old_full.shape),
                "overlay_distance_path_full": str(out_path.resolve()), "overlay_distance_sha256_full": out_sha,
                "overlay_distance_shape_full": list(full.shape), "q_support_full": list(full_q),
                "primary_overlap_max_abs_error": float(np.max(np.abs(primary_overlap))) if primary_overlap.size else None,
                "max_abs_delta_in_unaffected_contexts": max_unaffected_delta,
                "max_abs_delta_all_cells": float(np.max(diff)) if diff.size else 0.0,
                "changed_cell_count_gt_1e-5": int(np.count_nonzero(diff > 1e-5)),
                "expected_audio_tail_cell_count": len(tail_positions),
                "tail_cell_positions_row_col": tail_positions,
                "metrics_full": {
                    "B": metrics["B"], "Dmin": metrics["Dmin"], "D0": metrics["D0"],
                    "C": metrics["C"], "bestlag": metrics["bestlag"],
                    "search_gain": metrics["search_gain"],
                    "anchor_D": metrics["anchor_D"], "anchor_C": metrics["anchor_C"],
                    "fixed_anchor_lag_index": anchor_index,
                },
            })

    overlay_core = {
        "schema": "p1_full_context_distance_overlay_v1",
        "source_score_results_sha256": source_sha,
        "protocol_sha256": source["protocol_sha256"],
        "syncnet_model_sha256": source.get("syncnet_model_sha256"),
        "syncnet_code_hash": source.get("syncnet_code_hash"),
        "distance_contract": "Torch float32 pairwise_distance(eps=1e-6); visual row q versus all real audio rows q-15..q+15; no truncation before selecting contexts",
        "lag_frames": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
        "selected_speakers": [sid for sid in source["planned_speakers"] if sid in targeted],
        "artifacts": artifacts,
        "skipped_rows": skipped,
    }
    overlay_sha = canonical_sha256(overlay_core)
    corrected["derived_full_context_overlay"] = {
        "schema": "p1_full_context_distance_overlay_v1",
        "source_score_results_path": str(score_path.resolve()),
        "source_score_results_sha256": source_sha,
        "overlay_sha256": overlay_sha,
        "overlay_manifest_path": str((output_root / "full_context_overlay_manifest.json").resolve()),
        "primary_score_fields_preserved": True,
        "full_sensitivity_fields_corrected_only": True,
    }
    _validate_primary_unchanged(source, corrected)
    corrected_path = output_root / "score_results_full_context_overlay.json"
    write_json_new(corrected_path, corrected)
    manifest = {
        **overlay_core,
        "overlay_sha256": overlay_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_score_results_path": str(score_path.resolve()),
        "source_score_results_sha256": source_sha,
        "source_score_results_bytes": len(score_raw),
        "overlay_source_code_path": str(Path(__file__).resolve()),
        "overlay_source_code_sha256": sha256_file(Path(__file__).resolve()),
        "derived_score_results_path": str(corrected_path.resolve()),
        "derived_score_results_sha256": sha256_file(corrected_path),
        "derived_score_results_bytes": corrected_path.stat().st_size,
        "primary_score_fields_preserved": True,
        "original_score_results_modified": False,
        "model_forward_performed": False,
    }
    write_json_new(output_root / "full_context_overlay_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--speaker", action="append", help="limit this create-only overlay to one or more named speakers")
    args = parser.parse_args()
    manifest = build_overlay(args.score_results, args.output_root, set(args.speaker) if args.speaker else None)
    print(json.dumps({
        "status": "COMPLETE",
        "overlay_sha256": manifest["overlay_sha256"],
        "derived_score_results_sha256": manifest["derived_score_results_sha256"],
        "artifact_count": len(manifest["artifacts"]),
        "skipped_count": len(manifest["skipped_rows"]),
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
