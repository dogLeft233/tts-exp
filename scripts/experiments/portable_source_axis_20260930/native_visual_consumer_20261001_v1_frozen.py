#!/usr/bin/env python3
"""Freeze native visual availability, then analyze the sealed 4-cell tracks.

The availability command reads only timestamps/valid masks and original MFA
phone labels. The analyze command requires that immutable availability freeze
and only then reads mouth aperture, width, and pose values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NATIVE_PACKAGE_ROOT = Path(r"C:\Users\dogle\.codex\tts-exp-artifacts\20261001\native_geometry_visual_20261001_attempt01")
EXPECTED_PACKAGE_SHA = "73c2f0d9993b9af5f070259ffb869efdf8105030e06b9be4f81e8098330ad0c8"
EXPECTED_EMBEDDED_PACKAGE_SHA = "1d274f37013793f722473590bd23f424bc0e66bfec17be65f909865984d65143"
EXPECTED_NATIVE_SPEC_SHA = "b11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd"
EXPECTED_FREEZE_SHA = "45cc2ee35980177b126a05b79fb19caa06e43624c24c1c7f85c41575c677b287"
EXPECTED_CALIBRATION_SHA = "0a1897a03248322a4868ceae5e194b848212ef33858b4a1cb002f5fe58bd21a4"
EXPECTED_VISUAL_CORE_SHA = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
EXPECTED_TONE_ADAPTER_SHA = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
EXPECTED_VIDEO_FEATURES_SHA = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
EXPECTED_MFA_PRODUCER_SHA = "929b9e1d0aef5effc246f117ff33147ec59ca0e683aedc735dabd64ed6f39857"
EXPECTED_ALIGNMENT_PROTOCOL_SHA = "aeb753c937a8c1dfd8d6d837f4ee79d6b653a1aa6cfdb6479b2a86e154dc6f18"
EXPECTED_MFA_QUALIFICATION_SHA = "dbebb40ce39f1a8f4a71493c0ae19871e97948d186908d2f1545e0015a301650"
EXPECTED_EVAL_ELIGIBILITY_SHA = "acc6cacfb8f5b35c13163157c6de70f90bc6e34ae54364036935923982a85ba5"
EXPECTED_LANDMARKER_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
EXPECTED_CAL_BUNDLE_SHA = "8245326a145296504b0186977cf68e080d1a91f12bbf2f197b403a1db8cd81a8"
EXPECTED_CAL_QUALIFICATION_SHA = "dbebb40ce39f1a8f4a71493c0ae19871e97948d186908d2f1545e0015a301650"
EXPECTED_ALIGNMENT_FREEZE_SHA = "aeb753c937a8c1dfd8d6d837f4ee79d6b653a1aa6cfdb6479b2a86e154dc6f18"
CALIBRATION_PATH = REPO / "runs/recovered_20260930/independent_validation/calibration_visual_20260930/calibration_visual_tone_v2_20261001.json"
CALIBRATION_FREEZE_PATH = REPO / "runs/recovered_20260930/independent_validation/calibration_visual_20260930/visual_tone_v2_freeze_20261001.json"
CELLS_BY_SOURCE = {"natural": {"raw": "natural_raw", "fixed": "natural_fixed"},
                   "tts": {"raw": "tts_raw", "fixed": "tts_fixed"}}
CONTRASTS = {
    "native_t_raw_minus_n_raw": ("T_RAW", "N_RAW"),
    "natural_fixed_minus_natural_raw": ("N_FIXED", "N_RAW"),
    "tts_fixed_minus_tts_raw": ("T_FIXED", "T_RAW"),
    "change_in_t_minus_n_gap_after_fixed": "difference_in_differences",
}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930


class NativeVisualError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NativeVisualError(f"expected JSON object: {path}")
    return value


def parse_phone_tier(path: Path) -> list[dict[str, Any]]:
    """Read the exact `phones` IntervalTier labels/times from a long TextGrid."""
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    tier_active = False
    saw_phone_tier = False
    current: dict[str, Any] = {}
    in_interval = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"item\s*\[\d+\]:", stripped):
            tier_active = False
            in_interval = False
            current = {}
        if stripped.startswith('name = "'):
            name = stripped.split("=", 1)[1].strip().strip('"')
            tier_active = name.casefold() == "phones"
            saw_phone_tier |= tier_active
        if not tier_active:
            continue
        if re.match(r"intervals\s*\[\d+\]:", stripped):
            in_interval = True
            current = {}
        elif in_interval and stripped.startswith("xmin ="):
            current["start_s"] = float(stripped.split("=", 1)[1].strip())
        elif in_interval and stripped.startswith("xmax ="):
            current["end_s"] = float(stripped.split("=", 1)[1].strip())
        elif in_interval and stripped.startswith("text ="):
            raw = stripped.split("=", 1)[1].strip()
            if not (raw.startswith('"') and raw.endswith('"')):
                raise NativeVisualError(f"malformed phone label in TextGrid: {path}")
            label = raw[1:-1].replace('""', '"')
            if set(current) != {"start_s", "end_s"}:
                raise NativeVisualError(f"incomplete phone interval in TextGrid: {path}")
            start, end = float(current["start_s"]), float(current["end_s"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise NativeVisualError(f"invalid phone interval in TextGrid: {path}")
            rows.append({"label": label, "start_s": start, "end_s": end})
            in_interval = False
            current = {}
    if not saw_phone_tier or not rows:
        raise NativeVisualError(f"TextGrid has no nonempty phones tier: {path}")
    return rows


def bind_inputs(extraction_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    extraction_manifest_path = extraction_dir / "extraction_manifest.json"
    extraction = read_json(extraction_manifest_path)
    if extraction.get("status") != "COMPLETE" or extraction.get("cohort", {}).get("completed_videos") != 320:
        raise NativeVisualError("native feature extraction is not a complete 320-video grid")
    if extraction.get("input_seals", {}).get("package_manifest_raw_sha256") != EXPECTED_PACKAGE_SHA or extraction.get("input_seals", {}).get("package_manifest_embedded_sha256") != EXPECTED_EMBEDDED_PACKAGE_SHA:
        raise NativeVisualError("extracted tracks are not bound to the frozen native video package")
    package_manifest_path = NATIVE_PACKAGE_ROOT / "native_package_manifest.json"
    if sha256_file(package_manifest_path) != EXPECTED_PACKAGE_SHA:
        raise NativeVisualError("native package manifest changed after extraction")
    package = read_json(package_manifest_path)
    if package.get("spec_sha256") != EXPECTED_NATIVE_SPEC_SHA or package.get("original_mfa_evidence", {}).get("physical_overlay_applied") is not False:
        raise NativeVisualError("native visual diagnostic must use original physical-free MFA inputs")

    mfa_root = extraction_dir / "source/mfa/mfa"
    producer_path = mfa_root / "mfa_producer_manifest.json"
    alignment_path = mfa_root / "alignment_protocol_frozen.json"
    qualification_path = mfa_root / "calibration_qualification_frozen.json"
    eligibility_path = mfa_root / "evaluation_endpoint_eligibility.json"
    for path, expected, name in (
        (producer_path, EXPECTED_MFA_PRODUCER_SHA, "MFA producer"),
        (alignment_path, EXPECTED_ALIGNMENT_PROTOCOL_SHA, "MFA alignment protocol"),
        (qualification_path, EXPECTED_MFA_QUALIFICATION_SHA, "MFA qualification"),
        (eligibility_path, EXPECTED_EVAL_ELIGIBILITY_SHA, "EVAL phone eligibility"),
    ):
        if sha256_file(path) != expected:
            raise NativeVisualError(f"original {name} hash mismatch")
    eligibility = read_json(eligibility_path)
    if eligibility.get("records_total") != 80 or eligibility.get("records_pair_qualified") != 79 or eligibility.get("records_pair_unobserved") != 1:
        raise NativeVisualError("original EVAL qualification no longer matches the frozen 79/80 record")
    if len(eligibility.get("records", [])) != 80:
        raise NativeVisualError("original EVAL qualification is missing intended records")
    mfa_evidence = package.get("original_mfa_evidence", {})
    textgrid_rows = mfa_evidence.get("textgrids", [])
    if mfa_evidence.get("textgrid_count") != 160 or len(textgrid_rows) != 160:
        raise NativeVisualError("native package does not bind all 160 original MFA TextGrids")
    textgrid_root = (extraction_dir / "source/mfa/mfa/textgrids/evaluation").resolve(strict=True)
    for entry in textgrid_rows:
        relative = Path(str(entry["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise NativeVisualError("MFA TextGrid manifest contains an unsafe relative path")
        grid = (textgrid_root / relative).resolve(strict=True)
        if textgrid_root not in grid.parents or sha256_file(grid) != str(entry["sha256"]):
            raise NativeVisualError(f"original MFA TextGrid differs from the sealed package: {relative}")

    calibration_path = CALIBRATION_PATH.resolve(strict=True)
    freeze_path = CALIBRATION_FREEZE_PATH.resolve(strict=True)
    if sha256_file(calibration_path) != EXPECTED_CALIBRATION_SHA or sha256_file(freeze_path) != EXPECTED_FREEZE_SHA:
        raise NativeVisualError("CAL-only visual phone freeze/checkpoint hash mismatch")
    calibration = read_json(calibration_path)
    freeze = read_json(freeze_path)
    if calibration.get("status") != "CALIBRATION_COMPLETE" or calibration.get("protocol_id") != "tts_mechanism_visual_20261001_mandarin_tone_v2":
        raise NativeVisualError("CAL tone-aware visual calibration status/protocol mismatch")
    if freeze.get("status") != "FROZEN_CAL_ONLY" or freeze.get("evaluation_textgrids_read") is not False or freeze.get("evaluation_effects_read") is not False:
        raise NativeVisualError("visual phone rule freeze is not CAL-only")
    core_path = REPO / "scripts/experiments/tts_mechanism_visual_20260930.py"
    tone_path = REPO / "scripts/experiments/tts_mechanism_visual_tone_20261001.py"
    video_features_path = REPO / "scripts/experiments/lrs3_tts_visual_advantage/video_features.py"
    for path, expected, label in (
        (core_path, EXPECTED_VISUAL_CORE_SHA, "visual core"),
        (tone_path, EXPECTED_TONE_ADAPTER_SHA, "tone adapter"),
        (video_features_path, EXPECTED_VIDEO_FEATURES_SHA, "video feature extractor"),
    ):
        if sha256_file(path) != expected:
            raise NativeVisualError(f"frozen {label} source hash mismatch")
    if freeze.get("source_hashes", {}).get("analysis_core_sha256") != EXPECTED_VISUAL_CORE_SHA or freeze.get("source_hashes", {}).get("tone_adapter_sha256") != EXPECTED_TONE_ADAPTER_SHA:
        raise NativeVisualError("CAL freeze does not bind the current v2 phone rules")
    protocol = calibration.get("protocol", {})
    minimums = protocol.get("min_phone_tokens_per_clip_by_class")
    latency_s = protocol.get("latency", {}).get("latency_s")
    if not isinstance(minimums, dict) or set(minimums) != {"bilabial", "open_vowel"} or latency_s is None:
        raise NativeVisualError("CAL freeze lacks class minima or the frozen latency")
    if not math.isclose(float(latency_s), -0.085, abs_tol=1e-12) or any(int(minimums[k]) < 1 for k in minimums):
        raise NativeVisualError("CAL frozen latency/count threshold differs from the visual protocol")
    import scripts.experiments.tts_mechanism_visual_tone_20261001 as tone
    tone.install_rules()
    return package, eligibility, calibration, tone


def load_clock_availability(feature_path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if sha256_file(feature_path) != expected_sha:
        raise NativeVisualError(f"feature SHA mismatch: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as archive:
        if not {"timestamps_s", "valid", "metadata_json"}.issubset(archive.files):
            raise NativeVisualError(f"feature lacks required clock/valid arrays: {feature_path}")
        timestamps = np.asarray(archive["timestamps_s"])
        valid = np.asarray(archive["valid"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if timestamps.dtype != np.float64 or timestamps.ndim != 1 or valid.dtype != np.bool_ or valid.shape != timestamps.shape:
        raise NativeVisualError(f"malformed clock/valid arrays: {feature_path}")
    if not np.isfinite(timestamps).all() or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0)):
        raise NativeVisualError(f"invalid recipient clock: {feature_path}")
    return timestamps, valid, metadata


def _paired_source_support(raw_clock: np.ndarray, raw_valid: np.ndarray, raw_meta: dict[str, Any],
                           fixed_clock: np.ndarray, fixed_valid: np.ndarray, fixed_meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    from scripts.experiments.tts_mechanism_visual_20260930 import _common_clock
    tracks = {
        "baseline": SimpleNamespace(timestamps_s=raw_clock, valid=raw_valid,
                                     metadata={"fps": float(raw_meta.get("fps", 25.0))}),
        "fixed": SimpleNamespace(timestamps_s=fixed_clock, valid=fixed_valid,
                                  metadata={"fps": float(fixed_meta.get("fps", 25.0))}),
    }
    return _common_clock(tracks)


def phone_class_counts(phones: Sequence[Mapping[str, Any]], base_times: np.ndarray, common: np.ndarray,
                       tone: Any, latency_s: float, frame_minimum: int) -> tuple[dict[str, int], dict[str, int], dict[str, list[dict[str, Any]]]]:
    expected = {"bilabial": 0, "open_vowel": 0}
    supported = {"bilabial": 0, "open_vowel": 0}
    supported_tokens: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
    query_times = base_times[common]
    for phone in phones:
        category = tone.phone_class(phone["label"])
        if category is None:
            continue
        expected[category] += 1
        lo, hi = float(phone["start_s"]) + latency_s, float(phone["end_s"]) + latency_s
        frame_count = int(np.count_nonzero((query_times >= lo) & (query_times < hi)))
        if frame_count >= frame_minimum:
            supported[category] += 1
            supported_tokens[category].append({
                "label": str(phone["label"]), "start_s": float(phone["start_s"]),
                "end_s": float(phone["end_s"]), "common_frames_at_frozen_latency": frame_count,
            })
    return expected, supported, supported_tokens


def _mfa_phone_rows(record: Mapping[str, Any], source: str, textgrid_root: Path,
                    expected_textgrid_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info = record[source]
    if not isinstance(info, Mapping):
        raise NativeVisualError(f"MFA endpoint record missing {source}: {record.get('id')}")
    speaker, sample = str(record["speaker"]), str(record["id"])
    suffix = "natural" if source == "natural" else "tts"
    grid = textgrid_root / speaker / f"{sample}__{suffix}.TextGrid"
    actual_sha = sha256_file(grid)
    if actual_sha != expected_textgrid_sha or info.get("textgrid_sha256") != actual_sha:
        raise NativeVisualError(f"original {source} TextGrid hash mismatch: {sample}")
    parsed = parse_phone_tier(grid)
    tokens = info.get("tokens")
    if not isinstance(tokens, list) or len(parsed) != len(tokens):
        raise NativeVisualError(f"original MFA token count differs from its TextGrid: {sample}/{source}")
    phones: list[dict[str, Any]] = []
    for index, (grid_phone, token) in enumerate(zip(parsed, tokens, strict=True)):
        expected_label = str(token.get("raw_token", token.get("token", "")))
        if grid_phone["label"] != expected_label or not math.isclose(grid_phone["start_s"], float(token["start_s"]), rel_tol=0.0, abs_tol=1e-10) or not math.isclose(grid_phone["end_s"], float(token["end_s"]), rel_tol=0.0, abs_tol=1e-10):
            raise NativeVisualError(f"MFA token row {index} does not reproduce the bound TextGrid: {sample}/{source}")
        phones.append(grid_phone)
    return phones, {"qualified": bool(info.get("qualified")), "errors": list(info.get("errors", [])),
                    "speech_phone_count": int(info.get("speech_phone_count", 0)),
                    "unknown_oov_phone_count": int(info.get("unknown_oov_phone_count", 0)),
                    "textgrid_sha256": actual_sha}


def freeze_availability(extraction_dir: Path, outdir: Path) -> dict[str, Any]:
    extraction_dir = extraction_dir.resolve(strict=True)
    outdir = outdir.resolve()
    output_path = outdir / "native_phone_availability_freeze.json"
    if output_path.exists():
        raise FileExistsError(f"availability freeze already exists: {output_path}")
    package, eligibility, calibration, tone = bind_inputs(extraction_dir)
    extraction_manifest_path = extraction_dir / "extraction_manifest.json"
    extraction = read_json(extraction_manifest_path)
    eligibility_by_id = {str(row["id"]): row for row in eligibility["records"]}
    video_info = {(str(row["sample_id"]), str(row["cell"])): row for row in package["videos"]}
    feature_info = {(str(row["sample_id"]), str(row["cell"])): row for row in extraction["features"] if row.get("status") == "extracted"}
    if len(eligibility_by_id) != 80 or len(video_info) != 320 or len(feature_info) != 320:
        raise NativeVisualError("phone availability requires 80 MFA rows and 320 successfully extracted tracks")
    calibration_protocol = calibration["protocol"]
    minima = {key: int(value) for key, value in calibration_protocol["min_phone_tokens_per_clip_by_class"].items()}
    latency = float(calibration_protocol["latency"]["latency_s"])
    frame_minimum = int(calibration_protocol.get("phone_min_common_frames", 1))
    textgrid_root = extraction_dir / "source/mfa/mfa/textgrids/evaluation"
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(eligibility_by_id):
        alignment = eligibility_by_id[sample_id]
        speaker = str(alignment["speaker"])
        phone_rows: dict[str, list[dict[str, Any]]] = {}
        endpoint: dict[str, dict[str, Any]] = {}
        source_support: dict[str, dict[str, Any]] = {}
        feature_refs: dict[str, dict[str, Any]] = {}
        for source in ("natural", "tts"):
            phones, endpoint[source] = _mfa_phone_rows(
                alignment, source, textgrid_root, str(alignment[source]["textgrid_sha256"]),
            )
            phone_rows[source] = phones
            raw_cell, fixed_cell = CELLS_BY_SOURCE[source]["raw"], CELLS_BY_SOURCE[source]["fixed"]
            raw_ref, fixed_ref = feature_info[(sample_id, raw_cell)], feature_info[(sample_id, fixed_cell)]
            raw_path, fixed_path = Path(raw_ref["feature_path"]), Path(fixed_ref["feature_path"])
            raw_clock, raw_valid, raw_meta = load_clock_availability(raw_path, str(raw_ref["feature_sha256"]))
            fixed_clock, fixed_valid, fixed_meta = load_clock_availability(fixed_path, str(fixed_ref["feature_sha256"]))
            for cell, clock, valid, ref in (
                (raw_cell, raw_clock, raw_valid, raw_ref),
                (fixed_cell, fixed_clock, fixed_valid, fixed_ref),
            ):
                if (array_sha256(clock, "<f8") != str(ref["timestamp_sha256"])
                        or array_sha256(valid, "?") != str(ref["valid_mask_sha256"])):
                    raise NativeVisualError(f"clock/valid-array seal mismatch: {sample_id}/{cell}")
            if len(raw_clock) != int(video_info[(sample_id, raw_cell)]["video"]["decoded_frames"]) or len(fixed_clock) != int(video_info[(sample_id, fixed_cell)]["video"]["decoded_frames"]):
                raise NativeVisualError(f"feature clock length differs from sealed video probe: {sample_id}/{source}")
            common, maps, support = _paired_source_support(raw_clock, raw_valid, raw_meta, fixed_clock, fixed_valid, fixed_meta)
            expected, supported, token_support = phone_class_counts(phones, raw_clock, common, tone, latency, frame_minimum)
            both_classes = all(supported[category] >= minima[category] for category in minima)
            endpoint_source = bool(endpoint[source]["qualified"] and both_classes and common.any())
            source_support[source] = {
                "endpoint_qualified_by_original_mfa": endpoint[source]["qualified"],
                "endpoint_errors": endpoint[source]["errors"],
                "unknown_oov_phone_count": endpoint[source]["unknown_oov_phone_count"],
                "expected_phone_tokens_by_class": expected,
                "common_clock_supported_phone_tokens_by_class": supported,
                "minimum_tokens_by_class_from_cal": minima,
                "joint_two_class_available": bool(both_classes),
                "endpoint_available_for_visual_contrast": bool(endpoint_source),
                "clock_frames": len(raw_clock),
                "valid_frames_raw": int(raw_valid.sum()),
                "valid_frames_fixed": int(fixed_valid.sum()),
                "common_valid_frames_raw_fixed": int(common.sum()),
                "common_coverage_raw_fixed": float(common.mean()) if len(common) else 0.0,
                "timestamp_tolerance_s": support["timestamp_tolerance_s"],
                "matched_frames_by_cell": support["matched_frames_by_arm"],
                "max_abs_clock_delta_s_by_cell": support["max_abs_clock_delta_s_by_arm"],
                "supported_phone_intervals": token_support,
            }
            feature_refs[source] = {
                raw_cell: {"path": str(raw_path.resolve()), "sha256": raw_ref["feature_sha256"],
                           "video_sha256": raw_ref["video_sha256"], "frame_count": int(raw_ref["frame_count"]),
                           "valid_mask_sha256": raw_ref["valid_mask_sha256"], "timestamp_sha256": raw_ref["timestamp_sha256"]},
                fixed_cell: {"path": str(fixed_path.resolve()), "sha256": fixed_ref["feature_sha256"],
                             "video_sha256": fixed_ref["video_sha256"], "frame_count": int(fixed_ref["frame_count"]),
                             "valid_mask_sha256": fixed_ref["valid_mask_sha256"], "timestamp_sha256": fixed_ref["timestamp_sha256"]},
            }
        original_pair = bool(alignment.get("pair_qualified")) and all(endpoint[s]["qualified"] for s in ("natural", "tts"))
        sources_full = all(source_support[s]["endpoint_available_for_visual_contrast"] for s in ("natural", "tts"))
        all_four_videos = len(feature_refs["natural"]) == 2 and len(feature_refs["tts"]) == 2
        contrast_eligible = {
            "native_t_raw_minus_n_raw": bool(original_pair and sources_full and all_four_videos),
            "natural_fixed_minus_natural_raw": bool(source_support["natural"]["endpoint_available_for_visual_contrast"] and len(feature_refs["natural"]) == 2),
            "tts_fixed_minus_tts_raw": bool(source_support["tts"]["endpoint_available_for_visual_contrast"] and len(feature_refs["tts"]) == 2),
            "change_in_t_minus_n_gap_after_fixed": bool(original_pair and sources_full and all_four_videos),
        }
        rows.append({
            "sample_id": sample_id, "speaker_id": speaker, "split": str(alignment.get("split", "evaluation")),
            "original_mfa_pair_qualified": bool(alignment.get("pair_qualified")),
            "original_mfa_natural_qualified": bool(endpoint["natural"]["qualified"]),
            "original_mfa_tts_qualified": bool(endpoint["tts"]["qualified"]),
            "native_four_video_coverage": bool(all_four_videos),
            "source_phone_availability": source_support,
            "contrast_eligibility_from_pre_effect_availability": contrast_eligible,
            "feature_refs": feature_refs,
            "phone_intervals_by_source": phone_rows,
        })
    speaker_ids = sorted({row["speaker_id"] for row in rows})
    if len(rows) != 80 or len(speaker_ids) != 40 or any(sum(r["speaker_id"] == s for r in rows) != 2 for s in speaker_ids):
        raise NativeVisualError("availability freeze changed the intended 80 clips / 40 speakers / two clips each")
    speaker_rows = []
    for speaker in speaker_ids:
        clip_rows = [row for row in rows if row["speaker_id"] == speaker]
        speaker_rows.append({
            "speaker_id": speaker,
            "clip_ids": [row["sample_id"] for row in clip_rows],
            "complete_contrasts": {
                contrast: bool(len(clip_rows) == 2 and all(row["contrast_eligibility_from_pre_effect_availability"][contrast] for row in clip_rows))
                for contrast in CONTRASTS
            },
            "joint_four_video_available": bool(len(clip_rows) == 2 and all(row["native_four_video_coverage"] for row in clip_rows)),
        })
    clip_coverage = {
        contrast: sum(row["contrast_eligibility_from_pre_effect_availability"][contrast] for row in rows)
        for contrast in CONTRASTS
    }
    speaker_coverage = {
        contrast: sum(row["complete_contrasts"][contrast] for row in speaker_rows)
        for contrast in CONTRASTS
    }
    result = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_APERTURE_EFFECT_READ",
        "analysis_type": "native_4cell_visual_availability_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_uses_aperture_values": False,
        "aperture_landmarks_width_pose_arrays_loaded": False,
        "intended_denominator": {"clips": 80, "speakers": 40, "clips_per_speaker": 2},
        "frozen_rule": {"latency_s": latency, "minimum_phone_tokens_per_clip_by_class": minima,
                        "minimum_valid_common_frames_per_phone": frame_minimum,
                        "phone_classifier_source": "tone adapter removes five frozen tone-bar characters then applies old visual normalizer/classes",
                        "class_eligibility_uses_only_labels_grid_intervals_timestamps_valid_masks": True,
                        "physical_phone_identity_rule_applied": False},
        "input_seals": {
            "native_spec_sha256": EXPECTED_NATIVE_SPEC_SHA,
            "native_package_manifest_path": str(NATIVE_PACKAGE_ROOT / "native_package_manifest.json"),
            "native_package_manifest_raw_sha256": sha256_file(NATIVE_PACKAGE_ROOT / "native_package_manifest.json"),
            "native_package_manifest_embedded_sha256": EXPECTED_EMBEDDED_PACKAGE_SHA,
            "source_protocol_sha256": package["source_protocol"]["sha256"],
            "source_rows_sha256": package["source_rows"]["sha256"],
            "extraction_manifest_path": str(extraction_manifest_path),
            "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
            "feature_inventory_sha256": sha256_file(extraction_dir / "feature_inventory_sha256.json"),
            "mfa_producer_manifest_sha256": EXPECTED_MFA_PRODUCER_SHA,
            "mfa_alignment_protocol_sha256": EXPECTED_ALIGNMENT_PROTOCOL_SHA,
            "mfa_calibration_qualification_sha256": EXPECTED_MFA_QUALIFICATION_SHA,
            "mfa_eval_endpoint_eligibility_sha256": EXPECTED_EVAL_ELIGIBILITY_SHA,
            "mfa_grid_count": 160,
            "physical_overlay_applied": False,
            "calibration_freeze_path": str(CALIBRATION_FREEZE_PATH.resolve()),
            "calibration_freeze_sha256": sha256_file(CALIBRATION_FREEZE_PATH),
            "calibration_analysis_path": str(CALIBRATION_PATH.resolve()),
            "calibration_analysis_sha256": sha256_file(CALIBRATION_PATH),
            "landmarker_asset_sha256": EXPECTED_LANDMARKER_SHA,
        },
        "source_code": {
            "consumer_path": str(Path(__file__).resolve()),
            "consumer_sha256": sha256_file(Path(__file__).resolve()),
            "visual_core_sha256": EXPECTED_VISUAL_CORE_SHA,
            "tone_adapter_sha256": EXPECTED_TONE_ADAPTER_SHA,
        },
        "coverage": {
            "mfa_pair_qualified_clips": sum(bool(row["original_mfa_pair_qualified"]) for row in rows),
            "mfa_pair_unobserved_clips": sum(not bool(row["original_mfa_pair_qualified"]) for row in rows),
            "native_four_video_clips": sum(bool(row["native_four_video_coverage"]) for row in rows),
            "all_intended_clips": len(rows),
            "all_intended_speakers": len(speaker_rows),
            "clips_eligible_by_contrast": clip_coverage,
            "complete_two_clip_speakers_by_contrast": speaker_coverage,
            "complete_two_clip_speakers_joint_four_video": sum(bool(row["joint_four_video_available"]) for row in speaker_rows),
            "original_mfa_expected_pair_qualified": 79,
            "original_mfa_expected_pair_unobserved": 1,
        },
        "clip_rows": rows,
        "speaker_rows": speaker_rows,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, result)
    return result


def load_track(feature_path: Path, expected_sha: str) -> Any:
    if sha256_file(feature_path) != expected_sha:
        raise NativeVisualError(f"feature file changed after availability freeze: {feature_path}")
    from scripts.experiments.tts_mechanism_visual_20260930 import MouthTrack
    with np.load(feature_path, allow_pickle=False) as archive:
        required = {"timestamps_s", "valid", "mouth_open", "mouth_width", "landmarks", "metadata_json"}
        if not required.issubset(archive.files):
            raise NativeVisualError(f"feature archive lacks analysis fields: {feature_path}")
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        valid = np.asarray(archive["valid"], dtype=bool)
        aperture = np.asarray(archive["mouth_open"], dtype=np.float64)
        width = np.asarray(archive["mouth_width"], dtype=np.float64)
        landmarks = np.asarray(archive["landmarks"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    return MouthTrack(timestamps, aperture, valid, width, landmarks, metadata)


def summarize_tokens(phones: Sequence[Mapping[str, Any]], track: Any, frame_indices: np.ndarray,
                     tone: Any, latency_s: float, minimums: Mapping[str, int]) -> dict[str, Any]:
    core = tone._CORE
    token_rows, _expected = core._phone_apertures(phones, track, frame_indices, latency_s)
    by_class = {"bilabial": [], "open_vowel": []}
    for token in token_rows:
        if token["observed"]:
            by_class[str(token["category"])].append(float(token["aperture_mean"]))
    means = {
        category: float(np.mean(values, dtype=np.float64)) if len(values) >= int(minimums[category]) else None
        for category, values in by_class.items()
    }
    separation = float(means["open_vowel"] - means["bilabial"]) if all(value is not None for value in means.values()) else None
    auc = tone._mann_whitney_auc(by_class["open_vowel"], by_class["bilabial"])
    return {
        "bilabial_aperture": means["bilabial"], "open_vowel_aperture": means["open_vowel"],
        "phone_separation": separation, "phone_token_auc": float(auc) if auc is not None else None,
        "observed_phone_tokens_by_class": {key: len(values) for key, values in by_class.items()},
        "phone_tokens": token_rows,
    }


def speaker_weighted_summary(sample_values: Mapping[str, float | None], speakers: Mapping[str, str],
                             intended_samples: Sequence[str], intended_speakers: Sequence[str],
                             *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for sample in intended_samples:
        value = sample_values.get(sample)
        if value is not None and math.isfinite(float(value)):
            by_speaker[speakers[sample]].append(float(value))
    complete = {speaker: float(np.mean(values, dtype=np.float64)) for speaker, values in by_speaker.items() if len(values) == 2}
    names = sorted(complete)
    data = np.asarray([complete[name] for name in names], dtype=np.float64)
    if len(data):
        rng = np.random.Generator(np.random.PCG64(seed))
        sample_indices = rng.integers(0, len(data), size=(BOOTSTRAP_DRAWS, len(data)), endpoint=False)
        draw_means = data[sample_indices].mean(axis=1)
        ci = [float(np.quantile(draw_means, 0.005, method="linear")), float(np.quantile(draw_means, 0.995, method="linear"))]
        estimate = float(data.mean())
    else:
        ci, estimate = None, None
    return {
        "equal_speaker_mean": estimate, "ci99": ci,
        "complete_two_clip_speakers": len(names), "intended_speakers": len(intended_speakers),
        "complete_clips": sum(len(v) for v in by_speaker.values()), "intended_clips": len(intended_samples),
        "missing_speakers": sorted(set(intended_speakers) - set(names)),
        "speaker_values": complete, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed, "unit": "speaker; each speaker mean of two clip-level contrasts",
    }


def summarize_contrast(rows: list[dict[str, Any]], speakers: Mapping[str, str], intended_samples: Sequence[str],
                       intended_speakers: Sequence[str], value_getter: Any, eligibility_key: str) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    clip_values: dict[str, float | None] = {}
    reasons: Counter[str] = Counter()
    for sample in intended_samples:
        row = by_id[sample]
        if not row["contrast_eligibility_from_pre_effect_availability"][eligibility_key]:
            clip_values[sample] = None
            reasons["pre_effect_availability_or_mfa_qualification"] += 1
            continue
        value = value_getter(row)
        clip_values[sample] = float(value) if value is not None and math.isfinite(float(value)) else None
        if clip_values[sample] is None:
            reasons["metric_unobserved_on_cal_latency_support"] += 1
    summary = speaker_weighted_summary(clip_values, speakers, intended_samples, intended_speakers)
    summary["contrast_eligible_clips_from_availability"] = sum(by_id[s]["contrast_eligibility_from_pre_effect_availability"][eligibility_key] for s in intended_samples)
    summary["metric_observed_clips"] = sum(value is not None for value in clip_values.values())
    summary["missing_clip_reasons"] = dict(reasons)
    summary["clip_values"] = clip_values
    return summary


def analyze_tracks(extraction_dir: Path, availability_path: Path, outdir: Path) -> dict[str, Any]:
    extraction_dir = extraction_dir.resolve(strict=True)
    availability_path = availability_path.resolve(strict=True)
    outdir = outdir.resolve()
    report_path = outdir / "native_4cell_visual_analysis.json"
    if report_path.exists():
        raise FileExistsError(f"native visual analysis already exists: {report_path}")
    availability = read_json(availability_path)
    if availability.get("status") != "FROZEN_BEFORE_APERTURE_EFFECT_READ" or availability.get("selection_uses_aperture_values") is not False or availability.get("aperture_landmarks_width_pose_arrays_loaded") is not False:
        raise NativeVisualError("native availability freeze is not a pre-effect availability-only record")
    consumer_sha = sha256_file(Path(__file__).resolve())
    if availability.get("source_code", {}).get("consumer_sha256") != consumer_sha:
        raise NativeVisualError("consumer source changed after pre-effect availability freeze")
    seal_path = availability_path.with_suffix(".seal.json")
    if not seal_path.is_file():
        raise NativeVisualError(f"availability freeze sidecar is missing: {seal_path}")
    seal = read_json(seal_path)
    if (seal.get("availability_freeze_path") != str(availability_path)
            or seal.get("availability_freeze_sha256") != sha256_file(availability_path)
            or seal.get("consumer_sha256") != consumer_sha):
        raise NativeVisualError("availability freeze SHA does not match its analysis binding")
    package, _eligibility, calibration, tone = bind_inputs(extraction_dir)
    extraction_manifest_path = extraction_dir / "extraction_manifest.json"
    extraction = read_json(extraction_manifest_path)
    if availability.get("input_seals", {}).get("extraction_manifest_sha256") != sha256_file(extraction_manifest_path):
        raise NativeVisualError("availability freeze is bound to a different feature extraction manifest")
    rows = availability["clip_rows"]
    sample_ids = [str(row["sample_id"]) for row in rows]
    speaker_by_sample = {str(row["sample_id"]): str(row["speaker_id"]) for row in rows}
    speakers = sorted(set(speaker_by_sample.values()))
    minimums = {key: int(value) for key, value in calibration["protocol"]["min_phone_tokens_per_clip_by_class"].items()}
    latency = float(calibration["protocol"]["latency"]["latency_s"])
    per_clip: dict[str, Any] = {}
    global_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for availability_row in rows:
        sample = str(availability_row["sample_id"])
        phones_by_source = availability_row["phone_intervals_by_source"]
        refs = availability_row["feature_refs"]
        cell_tracks: dict[str, Any] = {}
        pair_states: dict[str, dict[str, Any]] = {}
        row_metrics: dict[str, Any] = {}
        for source in ("natural", "tts"):
            raw_cell, fixed_cell = CELLS_BY_SOURCE[source]["raw"], CELLS_BY_SOURCE[source]["fixed"]
            raw_ref = refs[source][raw_cell]
            fixed_ref = refs[source][fixed_cell]
            raw_track = load_track(Path(raw_ref["path"]), str(raw_ref["sha256"]))
            fixed_track = load_track(Path(fixed_ref["path"]), str(fixed_ref["sha256"]))
            tracks = {"baseline": raw_track, "fixed": fixed_track}
            common, maps, support = tone._CORE._common_clock(tracks)
            frozen_common_count = int(availability_row["source_phone_availability"][source]["common_valid_frames_raw_fixed"])
            if int(common.sum()) != frozen_common_count:
                raise NativeVisualError(f"common valid clock changed after availability freeze: {sample}/{source}")
            common_indices = np.flatnonzero(common)
            base_times = raw_track.timestamps_s
            phone_data = {}
            zero_phone_data = {}
            global_data = {}
            for cell_name, track in (("raw", raw_track), ("fixed", fixed_track)):
                frame_indices = maps[cell_name][common_indices]
                phone_data[cell_name] = summarize_tokens(phones_by_source[source], track, frame_indices, tone, latency, minimums)
                zero_phone_data[cell_name] = summarize_tokens(phones_by_source[source], track, frame_indices, tone, 0.0, minimums)
                global_data[cell_name] = tone._CORE._global_track_metrics(track, maps[cell_name], common, base_times)
                global_metrics[sample][CELLS_BY_SOURCE[source][cell_name]] = global_data[cell_name]
                cell_tracks[CELLS_BY_SOURCE[source][cell_name]] = track
            pair_states[source] = {
                "common_clock": support,
                "metrics_at_cal_latency": phone_data,
                "zero_latency_sensitivity": zero_phone_data,
                "overall_track_metrics": global_data,
            }
        row_metrics = {
            "sample_id": sample, "speaker_id": speaker_by_sample[sample],
            "original_mfa_pair_qualified": availability_row["original_mfa_pair_qualified"],
            "phone_class_availability": availability_row["source_phone_availability"],
            "contrast_eligibility": availability_row["contrast_eligibility_from_pre_effect_availability"],
            "feature_refs": refs,
            "source_results": pair_states,
        }
        per_clip[sample] = row_metrics

    def cell_value(sample: str, source: str, cell: str, field: str, latency_kind: str = "metrics_at_cal_latency") -> float | None:
        value = per_clip[sample]["source_results"][source][latency_kind][cell][field]
        return None if value is None else float(value)

    contrast_fields: dict[str, dict[str, Any]] = {}
    for contrast, operands in CONTRASTS.items():
        contrast_fields[contrast] = {}
        for metric in ("phone_separation", "phone_token_auc"):
            def get_value(row: dict[str, Any], metric: str = metric, contrast: str = contrast) -> float | None:
                sid = str(row["sample_id"])
                if contrast == "native_t_raw_minus_n_raw":
                    left, right = cell_value(sid, "tts", "raw", metric), cell_value(sid, "natural", "raw", metric)
                    return left - right if left is not None and right is not None else None
                if contrast == "natural_fixed_minus_natural_raw":
                    left, right = cell_value(sid, "natural", "fixed", metric), cell_value(sid, "natural", "raw", metric)
                    return left - right if left is not None and right is not None else None
                if contrast == "tts_fixed_minus_tts_raw":
                    left, right = cell_value(sid, "tts", "fixed", metric), cell_value(sid, "tts", "raw", metric)
                    return left - right if left is not None and right is not None else None
                tf, nf = cell_value(sid, "tts", "fixed", metric), cell_value(sid, "natural", "fixed", metric)
                tr, nr = cell_value(sid, "tts", "raw", metric), cell_value(sid, "natural", "raw", metric)
                return (tf - nf) - (tr - nr) if None not in (tf, nf, tr, nr) else None
            contrast_fields[contrast][metric] = summarize_contrast(rows, speaker_by_sample, sample_ids, speakers, get_value, contrast)
            zero_metric = f"zero_latency_{metric}"
            def get_zero(row: dict[str, Any], metric: str = metric, contrast: str = contrast) -> float | None:
                sid = str(row["sample_id"])
                if contrast == "native_t_raw_minus_n_raw":
                    left, right = cell_value(sid, "tts", "raw", metric, "zero_latency_sensitivity"), cell_value(sid, "natural", "raw", metric, "zero_latency_sensitivity")
                    return left - right if left is not None and right is not None else None
                if contrast == "natural_fixed_minus_natural_raw":
                    left, right = cell_value(sid, "natural", "fixed", metric, "zero_latency_sensitivity"), cell_value(sid, "natural", "raw", metric, "zero_latency_sensitivity")
                    return left - right if left is not None and right is not None else None
                if contrast == "tts_fixed_minus_tts_raw":
                    left, right = cell_value(sid, "tts", "fixed", metric, "zero_latency_sensitivity"), cell_value(sid, "tts", "raw", metric, "zero_latency_sensitivity")
                    return left - right if left is not None and right is not None else None
                tf, nf = cell_value(sid, "tts", "fixed", metric, "zero_latency_sensitivity"), cell_value(sid, "natural", "fixed", metric, "zero_latency_sensitivity")
                tr, nr = cell_value(sid, "tts", "raw", metric, "zero_latency_sensitivity"), cell_value(sid, "natural", "raw", metric, "zero_latency_sensitivity")
                return (tf - nf) - (tr - nr) if None not in (tf, nf, tr, nr) else None
            contrast_fields[contrast][zero_metric] = summarize_contrast(rows, speaker_by_sample, sample_ids, speakers, get_zero, contrast)

    overall_summary: dict[str, Any] = {}
    for cell in ("natural_raw", "natural_fixed", "tts_raw", "tts_fixed"):
        overall_summary[cell] = {}
        for metric in ("aperture", "mouth_width", "absolute_aperture_velocity_per_s", "yaw_proxy", "pitch_proxy"):
            for statistic in ("mean", "median"):
                sample_values: dict[str, float | None] = {}
                for sample in sample_ids:
                    summary = global_metrics[sample].get(cell, {}).get(metric, {})
                    value = summary.get(statistic)
                    sample_values[sample] = float(value) if value is not None else None
                overall_summary[cell][f"{metric}_{statistic}"] = speaker_weighted_summary(sample_values, speaker_by_sample, sample_ids, speakers)

    group_summaries = {
        "cells": {},
        "speaker_weighting": "each clip is summarized first; a speaker contributes the mean of their two clips; speakers are equal-weighted",
    }
    for cell in ("natural_raw", "natural_fixed", "tts_raw", "tts_fixed"):
        group_summaries["cells"][cell] = {}
        for metric in ("phone_separation", "phone_token_auc"):
            sample_values: dict[str, float | None] = {}
            source = "natural" if cell.startswith("natural") else "tts"
            subtype = "raw" if cell.endswith("raw") else "fixed"
            for sample in sample_ids:
                sample_values[sample] = cell_value(sample, source, subtype, metric)
            group_summaries["cells"][cell][metric] = speaker_weighted_summary(sample_values, speaker_by_sample, sample_ids, speakers)

    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "analysis_type": "native_4cell_visual_phone_and_geometry_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_scope": "independent visual geometry and phone-class association; no perception or event-timing claim",
        "input_seals": {
            "availability_freeze_path": str(availability_path),
            "availability_freeze_sha256": sha256_file(availability_path),
            "availability_consumer_sha256": consumer_sha,
            "native_package_manifest_sha256": sha256_file(NATIVE_PACKAGE_ROOT / "native_package_manifest.json"),
            "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
            "calibration_freeze_sha256": sha256_file(CALIBRATION_FREEZE_PATH),
            "calibration_analysis_sha256": sha256_file(CALIBRATION_PATH),
            "mfa_eval_endpoint_eligibility_sha256": EXPECTED_EVAL_ELIGIBILITY_SHA,
            "physical_overlay_applied": False,
        },
        "method": {
            "primary_phone_endpoint": "mean token-level open-vowel aperture minus mean token-level bilabial aperture at the CAL-frozen latency",
            "latency_s": latency,
            "zero_latency": "secondary sensitivity using the same video tracks and MFA phone intervals at zero lag",
            "auc": "per-clip Mann-Whitney AUC over phone-token mean apertures; open vowel positive; ties 0.5",
            "clock": "within each source, RAW/FIXED pair shares full RAW timestamp clock and common valid support; N and T are not warped/intersected",
            "phone_class_normalization": "visual tone adapter removes only the five frozen tone-bar characters before the preexisting normalizer, which also removes length/aspiration/palatalization/labialization/dental marks",
            "mfa": "original separate N/T `phones` tiers, bound by original grid hashes and qualification; physical overlay not applied",
            "statistics": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "ci": [0.005, 0.995], "unit": "equal 40 speakers; each contributes mean of two clips", "missingness": "intended 80 clips/40 speakers retained; each contrast reports matched complete-speaker support"},
            "global_metrics": "overall aperture, mouth width, absolute aperture velocity per second, yaw proxy and pitch proxy on the source-pair common valid clock",
            "event_timing": "not analyzed here; CAL rematching remains exploratory and uncalibrated",
        },
        "intended_denominator": {"clips": 80, "speakers": 40, "clips_per_speaker": 2},
        "pre_effect_availability": availability["coverage"],
        "cell_summaries": group_summaries,
        "contrast_summaries": contrast_fields,
        "overall_geometry_summaries": overall_summary,
        "per_clip": per_clip,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    availability_parser = sub.add_parser("freeze-availability")
    availability_parser.add_argument("--extraction-dir", type=Path, required=True)
    availability_parser.add_argument("--outdir", type=Path, required=True)
    analysis_parser = sub.add_parser("analyze")
    analysis_parser.add_argument("--extraction-dir", type=Path, required=True)
    analysis_parser.add_argument("--availability-freeze", type=Path, required=True)
    analysis_parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-availability":
        result = freeze_availability(args.extraction_dir, args.outdir)
        path = args.outdir.resolve() / "native_phone_availability_freeze.json"
        seal_path = path.with_suffix(".seal.json")
        atomic_json(seal_path, {"availability_freeze_path": str(path.resolve()), "availability_freeze_sha256": sha256_file(path),
                                "consumer_sha256": sha256_file(Path(__file__).resolve())})
        print(json.dumps({"status": result["status"], "coverage": result["coverage"], "path": str(path.resolve()), "sha256": sha256_file(path),
                          "seal_path": str(seal_path.resolve()), "seal_sha256": sha256_file(seal_path)}, ensure_ascii=False, separators=(",", ":")))
        return 0
    result = analyze_tracks(args.extraction_dir, args.availability_freeze, args.outdir)
    path = args.outdir.resolve() / "native_4cell_visual_analysis.json"
    print(json.dumps({"status": result["status"], "path": str(path), "sha256": sha256_file(path)}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
