#!/usr/bin/env python3
"""Metadata-only GUARD20 geometry audit for already-frozen audio and controls.

This reads the existing controls ledger and the future-Dev20 source manifest.
It never opens audio/video bodies, reads score values, or inspects future effects.
The per-row query counts from controls are geometry diagnostics; they are not a
three-condition common-support qualification for the future cohort.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / ".codex/scratch/dev360_terminal_transfer_attempt09/verify_output_attempt01/results.jsonl"
SOURCE_MANIFEST = ROOT / ".codex/scratch/dev20_future_cohort_metadata_attempt06/future_dev20_portable_source_manifest_attempt02.json"
SCORER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"
STAGEA = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py"
PROTOCOL = ROOT / ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json"
OUT = Path(__file__).with_name("guard20_geometry_audit_attempt01.json")

SAMPLES_PER_25FPS_FRAME = 640
SYNCNET_QUERY_TAIL = 5
GUARD_TRIM_EACH_END = 20
GUARD_MIN_REMAINING = 41
REQUIRED_COMMON_QUERY_COUNT = GUARD_TRIM_EACH_END * 2 + GUARD_MIN_REMAINING


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw)}


def self_hash(payload: dict[str, Any], field: str) -> str:
    material = dict(payload)
    material.pop(field, None)
    raw = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(raw)


def read_controls_geometry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    statuses: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    models: Counter[str] = Counter()
    seeds: Counter[int] = Counter()
    with LEDGER.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows_seen += 1
            status = row.get("status")
            statuses[str(status)] += 1
            methods[str(row.get("method_id"))] += 1
            models[str(row.get("model_id"))] += 1
            seeds[int(row["seed"])] += 1

            score = row.get("score") or {}
            support = score.get("support_by_support") or {}
            full = (support.get("full") or {}).get("query_count")
            guard = (support.get("guard20") or {}).get("query_count")
            pcm = score.get("primary_pcm16_window") or {}
            score_samples = pcm.get("sample_count")
            evaluator = score.get("official_evaluator") or {}
            receipt = evaluator.get("receipt") or {}
            video_frames = receipt.get("video_frame_count")
            evaluator_queries = receipt.get("expected_query_count")
            source_samples = (row.get("generator_pcm") or {}).get("sample_count")
            model_id = row.get("model_id")
            sample_id = row.get("sample_id")
            speaker_id = row.get("speaker_id")
            if not all(type(value) is int for value in (full, guard, score_samples, video_frames, evaluator_queries, source_samples)):
                raise ValueError(f"controls geometry fields are missing or non-integer at ledger line {line_number}")
            expected_queries = max(0, min(video_frames, score_samples // SAMPLES_PER_25FPS_FRAME) - SYNCNET_QUERY_TAIL)
            if evaluator_queries != expected_queries or full != expected_queries:
                raise ValueError(f"Stage-A query formula does not reproduce the frozen controls row at line {line_number}")
            expected_guard = max(0, full - 2 * GUARD_TRIM_EACH_END)
            if guard != expected_guard:
                raise ValueError(f"GUARD20 trimming does not reproduce the frozen controls row at line {line_number}")
            samples[str(sample_id)].append({
                "speaker_id": str(speaker_id),
                "sample_id": str(sample_id),
                "model_id": str(model_id),
                "method_id": str(row.get("method_id")),
                "seed": int(row["seed"]),
                "status": str(status),
                "natural_source_samples": source_samples,
                "score_pcm16_window_samples": score_samples,
                "score_pcm16_window_frames": score_samples // SAMPLES_PER_25FPS_FRAME,
                "evaluator_video_frames": video_frames,
                "full_query_count_single_condition": full,
                "guard20_count_single_condition": guard,
            })

    sample_summaries = []
    for sample_id, entries in sorted(samples.items()):
        lengths = {entry["natural_source_samples"] for entry in entries}
        if len(lengths) != 1:
            raise ValueError(f"frozen source length differs across controls rows: {sample_id}")
        sample_summaries.append({
            "speaker_id": entries[0]["speaker_id"],
            "sample_id": sample_id,
            "planned_rows": len(entries),
            "natural_source_samples": entries[0]["natural_source_samples"],
            "source_25fps_frame_floor": entries[0]["natural_source_samples"] // SAMPLES_PER_25FPS_FRAME,
            "min_evaluator_video_frames": min(entry["evaluator_video_frames"] for entry in entries),
            "min_score_window_frames": min(entry["score_pcm16_window_frames"] for entry in entries),
            "min_full_query_count_single_condition": min(entry["full_query_count_single_condition"] for entry in entries),
            "max_full_query_count_single_condition": max(entry["full_query_count_single_condition"] for entry in entries),
            "min_guard20_count_single_condition": min(entry["guard20_count_single_condition"] for entry in entries),
            "max_guard20_count_single_condition": max(entry["guard20_count_single_condition"] for entry in entries),
            "all_rows_pass": all(entry["status"] == "PASS" for entry in entries),
            "scope": "two-condition per-row geometry only; no three-condition common-support claim",
        })

    control_summary = {
        "row_count": rows_seen,
        "status_counts": dict(sorted(statuses.items())),
        "method_counts": dict(sorted(methods.items())),
        "model_counts": dict(sorted(models.items())),
        "seed_counts": {str(seed): count for seed, count in sorted(seeds.items())},
        "sample_count": len(samples),
        "speaker_count": len({row["speaker_id"] for row in sample_summaries}),
        "sample_geometry": sample_summaries,
    }
    return sample_summaries, control_summary


def read_future_source_geometry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != "aishell_future_dev20_portable_source_manifest_attempt02_v1":
        raise ValueError("future Dev20 manifest schema differs from the pinned cohort")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("future Dev20 source manifest must contain exactly twenty preselected WAV rows")
    counts = Counter(str(row["speaker_id"]) for row in rows)
    if len(counts) != 10 or set(counts.values()) != {2}:
        raise ValueError("future Dev20 source roster is not ten speakers with exactly two frozen clips each")
    if manifest.get("selection", {}).get("confirm_audio_or_scores_accessed") is not False:
        raise ValueError("future Dev20 source manifest does not attest the no-Confirm selection scope")
    result = []
    for row in rows:
        n = row.get("riff", {}).get("sample_count")
        rate = row.get("riff", {}).get("sample_rate_hz")
        if type(n) is not int or rate != 16000:
            raise ValueError(f"future Dev20 audio-clock metadata invalid for {row.get('sample_id')}")
        result.append({
            "speaker_id": row["speaker_id"],
            "sample_id": row["sample_id"],
            "source_wav_bytes": row["bytes"],
            "source_wav_sha256": row["sha256"],
            "sample_count": n,
            "sample_rate_hz": rate,
            "source_25fps_frame_floor": n // SAMPLES_PER_25FPS_FRAME,
            "audio_side_full_query_upper_bound_if_full_track": max(0, n // SAMPLES_PER_25FPS_FRAME - SYNCNET_QUERY_TAIL),
            "necessary_audio_length_bound_for_81_queries": n >= (REQUIRED_COMMON_QUERY_COUNT + SYNCNET_QUERY_TAIL) * SAMPLES_PER_25FPS_FRAME and n // SAMPLES_PER_25FPS_FRAME >= REQUIRED_COMMON_QUERY_COUNT + SYNCNET_QUERY_TAIL,
            "source_wav_body_copied_or_opened_by_this_audit": False,
        })
    summary = {
        "status": manifest.get("status"),
        "speaker_counts": dict(sorted(counts.items())),
        "row_count": len(result),
        "min_source_sample_count": min(row["sample_count"] for row in result),
        "max_source_sample_count": max(row["sample_count"] for row in result),
        "min_source_frame_floor": min(row["source_25fps_frame_floor"] for row in result),
        "all_rows_meet_necessary_audio_length_bound": all(row["necessary_audio_length_bound_for_81_queries"] for row in result),
        "current_three_condition_common_support_known": False,
        "current_native_video_or_crop_tracks_scored": False,
        "rows": result,
    }
    return result, summary


def main() -> None:
    controls_rows, controls = read_controls_geometry()
    future_rows, future = read_future_source_geometry()
    payload: dict[str, Any] = {
        "schema": "cross_tfg_guard20_geometry_metadata_audit_attempt01_v1",
        "status": "PASS_GEOMETRY_FORMULA_AND_METADATA_BOUND_ONLY",
        "scope": {
            "future_audio_or_video_opened": False,
            "future_effects_or_scores_read": False,
            "confirm_audio_or_text_opened": False,
            "asr_or_api_called": False,
            "gpu_or_model_started": False,
            "control_score_values_read_or_emitted": False,
            "control_ledger_use": "geometry metadata only: source length, PCM16 window length, evaluator video frame count, and declared per-row support counts",
        },
        "source_pins": {
            "controls_results_jsonl": file_pin(LEDGER),
            "future_dev20_source_manifest": file_pin(SOURCE_MANIFEST),
            "stagea_syncnet_evaluator": file_pin(SCORER),
            "stagea_run_stagea_attempt06": file_pin(STAGEA),
            "future_dev20_protocol06_json": file_pin(PROTOCOL),
        },
        "frozen_rules": {
            "sample_rate_hz": 16000,
            "source_clock_fps": "25/1",
            "samples_per_source_frame": SAMPLES_PER_25FPS_FRAME,
            "stagea_query_formula": "max(0, min(decoded_video_frame_count, floor(score_pcm16_window_samples / 640)) - 5)",
            "guard20_rule": "common valid query-key intersection across identity, rms_fixed, and DirectD0; remove 20 queries at each end; at least 41 remain",
            "minimum_full_common_query_count": REQUIRED_COMMON_QUERY_COUNT,
            "per-condition_necessary_frame_floor": REQUIRED_COMMON_QUERY_COUNT + SYNCNET_QUERY_TAIL,
            "per-condition_necessary_pcm16_window_samples": (REQUIRED_COMMON_QUERY_COUNT + SYNCNET_QUERY_TAIL) * SAMPLES_PER_25FPS_FRAME,
            "source_length_bound_is_sufficient": False,
            "no_postscore_replacement_or_subset_rescue": True,
        },
        "controls_geometry": controls,
        "future_dev20_source_length_screen": future,
        "interpretation": {
            "current_cohort_source_screen": "all twenty staged clips meet the necessary audio-side length bound; this is not proof that native video/crop tracks or the three-condition common query intersection reach 81",
            "known_short_control_example": "the 51,376-sample controls clip has floor 80 source frames, below the 86-frame necessary bound and its observed per-row support is below GUARD20 eligibility; the 55,729-sample next-shortest case has observed per-row query counts 81–82, but still is only a two-condition historical geometry reference",
            "future_admission": "keep all 540 planned rows; after actual outputs, compute exact common query keys for each frozen sample/model/face/seed and record every short-support failure without replacement",
        },
    }
    payload["self_sha256"] = self_hash(payload, "self_sha256")
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = __import__("os").open(OUT, __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_EXCL, 0o600)
    with __import__("os").fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        __import__("os").fsync(stream.fileno())
    print(json.dumps({"status": payload["status"], "receipt_path": str(OUT.resolve()), "receipt_bytes": len(raw), "receipt_sha256": sha256(raw), "self_sha256": payload["self_sha256"], "controls_rows": controls["row_count"], "future_dev20_rows": future["row_count"], "min_future_sample_count": future["min_source_sample_count"], "required_audio_sample_count": payload["frozen_rules"]["per-condition_necessary_pcm16_window_samples"]}, sort_keys=True))


if __name__ == "__main__":
    main()
