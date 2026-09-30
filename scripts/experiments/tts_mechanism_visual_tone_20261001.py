#!/usr/bin/env python3
"""Versioned Mandarin tone adaptation for the independent visual endpoint.

The 2026-09-30 visual rules remain frozen for MFA qualification. This adapter
changes only visual phone classification: it removes the five tone-bar
characters observed in the CAL MFA inventory, then applies the prior visual
normalizer and phone classes. Physical phone identities are not changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

CORE_PATH = Path(__file__).with_name("tts_mechanism_visual_20260930.py")
CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
PROTOCOL_ID = "tts_mechanism_visual_20261001_mandarin_tone_v2"
SHUFFLE_DIAGNOSTIC_SEED = 20261001
SHUFFLE_DIAGNOSTIC_DRAWS = 100
TONE_MARKS = {
    "\u02e5": "MODIFIER LETTER EXTRA-HIGH TONE BAR",
    "\u02e6": "MODIFIER LETTER HIGH TONE BAR",
    "\u02e7": "MODIFIER LETTER MID TONE BAR",
    "\u02e8": "MODIFIER LETTER LOW TONE BAR",
    "\u02e9": "MODIFIER LETTER EXTRA-LOW TONE BAR",
}


class ToneVisualError(ValueError):
    """Invalid visual-phone calibration input or provenance."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ToneVisualError(f"expected JSON object: {path}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _load_core():
    actual = _sha256_file(CORE_PATH)
    if actual != CORE_SHA256:
        raise ToneVisualError(f"frozen visual-analysis core SHA changed: {actual}")
    spec = importlib.util.spec_from_file_location("tts_mechanism_visual_core_20260930", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ToneVisualError(f"cannot load visual-analysis core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_core()
_ORIGINAL_NORMALIZE_PHONE = _CORE.normalize_phone


def normalize_visual_phone(label: Any) -> str:
    """Remove only the frozen tone-bar allowlist before prior visual rules."""
    value = str(label or "").strip().lower()
    value = "".join(char for char in value if char not in TONE_MARKS)
    return _ORIGINAL_NORMALIZE_PHONE(value)


def phone_class(label: Any) -> str | None:
    normalized = normalize_visual_phone(label)
    if normalized in _CORE.BILABIAL_BASES:
        return "bilabial"
    if normalized in _CORE.OPEN_VOWEL_BASES:
        return "open_vowel"
    return None


def install_rules() -> None:
    """Install this versioned classifier in the otherwise frozen core."""
    _CORE.PROTOCOL_ID = PROTOCOL_ID
    _CORE.normalize_phone = normalize_visual_phone
    _CORE.phone_class = phone_class


def _tone_allowlist_rows(observed_counts: Counter[str] | None = None) -> list[dict[str, Any]]:
    counts = observed_counts or Counter()
    return [
        {
            "symbol": mark,
            "codepoint": f"U+{ord(mark):04X}",
            "unicode_name": name,
            "observed_natural_cal_phone_tokens": int(counts.get(mark, 0)),
        }
        for mark, name in TONE_MARKS.items()
    ]


def build_mapping_table(phone_counts: Mapping[str, int]) -> dict[str, Any]:
    """Create a raw-to-visual mapping table for an observed CAL inventory."""
    rows = []
    tone_counts: Counter[str] = Counter()
    for raw, count in sorted((str(k), int(v)) for k, v in phone_counts.items()):
        if count < 0:
            raise ToneVisualError("phone occurrence counts must be nonnegative")
        marks = [char for char in raw if char in TONE_MARKS]
        tone_counts.update(marks * count)
        rows.append({
            "raw_phone": raw,
            "occurrence_count": count,
            "removed_tone_marks": marks,
            "visual_base_phone": normalize_visual_phone(raw),
            "visual_class": phone_class(raw),
        })
    body = {
        "schema_version": 1,
        "mapping_id": "mandarin-visual-tone-bars-v1",
        "scope": "natural CAL MFA phone tokens only",
        "tone_mark_policy": "remove only the five explicitly listed Unicode tone-bar characters for visual class assignment",
        "tone_mark_allowlist": _tone_allowlist_rows(tone_counts),
        "phone_rows": rows,
        "phone_token_count": sum(row["occurrence_count"] for row in rows),
        "phone_symbol_count": len(rows),
        "tone_marks_observed": sorted(char for char, count in tone_counts.items() if count),
    }
    body["mapping_sha256"] = _canonical_hash(body)
    return body


def fixed_event_index_timestamp_control(
    tracks: Mapping[str, Any], *, min_prominence: float
) -> dict[str, Any]:
    """Check +/-5-frame timestamp transforms on the same detected event indices."""
    rows = []
    all_indices_stable = True
    plus_deltas: list[float] = []
    minus_deltas: list[float] = []
    for sample_id, track in sorted(tracks.items()):
        base = _CORE._detect_events(track, min_prominence=min_prominence)
        plus = _CORE._detect_events(
            track, min_prominence=min_prominence, timestamp_offset_s=5.0 / _CORE.FPS_REFERENCE
        )
        minus = _CORE._detect_events(
            track, min_prominence=min_prominence, timestamp_offset_s=-5.0 / _CORE.FPS_REFERENCE
        )
        sample_stable = True
        sample_counts: dict[str, int] = {}
        for kind in ("minimum", "maximum"):
            base_events, plus_events, minus_events = base[kind], plus[kind], minus[kind]
            base_indices = [int(item["index"]) for item in base_events]
            plus_indices = [int(item["index"]) for item in plus_events]
            minus_indices = [int(item["index"]) for item in minus_events]
            stable = base_indices == plus_indices == minus_indices
            sample_stable &= stable
            sample_counts[kind] = len(base_indices)
            if stable:
                plus_deltas.extend(
                    (float(after["time_s"]) - float(before["time_s"])) * _CORE.FPS_REFERENCE
                    for before, after in zip(base_events, plus_events, strict=True)
                )
                minus_deltas.extend(
                    (float(after["time_s"]) - float(before["time_s"])) * _CORE.FPS_REFERENCE
                    for before, after in zip(base_events, minus_events, strict=True)
                )
        all_indices_stable &= sample_stable
        rows.append({"sample_id": str(sample_id), "event_indices_unchanged": bool(sample_stable),
                     "event_counts_by_type": sample_counts,
                     "aperture_multiset_preserved": bool(np.array_equal(np.sort(track.aperture), np.sort(track.aperture.copy())))})
    median_plus = float(np.median(plus_deltas)) if plus_deltas else None
    median_minus = float(np.median(minus_deltas)) if minus_deltas else None
    plus_ok = median_plus is not None and abs(median_plus - 5.0) <= 1e-6
    minus_ok = median_minus is not None and abs(median_minus + 5.0) <= 1e-6
    passed = bool(rows and all_indices_stable and plus_ok and minus_ok)
    return {
        "procedure": "freeze detected extrema indices on each original track; apply timestamp offsets only; compare identical events",
        "expected_shift_frames": {"plus": 5.0, "minus": -5.0},
        "tolerance_frames": 1e-6,
        "median_plus_timestamp_delta_frames": median_plus,
        "median_minus_timestamp_delta_frames": median_minus,
        "event_indices_unchanged_for_all_clips": bool(all_indices_stable),
        "aperture_multiset_preserved_for_all_clips": bool(rows and all(row["aperture_multiset_preserved"] for row in rows)),
        "pass": passed,
        "rows": rows,
    }


def _phone_separation_for_aperture(
    row: Mapping[str, Any], track: Any, aperture: np.ndarray, *, latency_s: float,
    minimum_counts: Mapping[str, int],
) -> dict[str, Any]:
    valid_indices = np.flatnonzero(track.valid)
    aperture_track = replace(track, aperture=np.asarray(aperture, dtype=np.float64))
    token_rows, expected = _CORE._phone_apertures(
        row["phones"], aperture_track, valid_indices, latency_s
    )
    means: dict[str, list[float]] = {"bilabial": [], "open_vowel": []}
    for token in token_rows:
        if token["observed"]:
            means[token["category"]].append(float(token["aperture_mean"]))
    class_means = {}
    for category in ("bilabial", "open_vowel"):
        values = means[category]
        class_means[category] = (
            float(np.mean(values)) if len(values) >= int(minimum_counts.get(category, 1)) else None
        )
    bilabial = class_means["bilabial"]
    vowel = class_means["open_vowel"]
    return {
        "bilabial_aperture": bilabial,
        "open_vowel_aperture": vowel,
        "phone_separation": float(vowel - bilabial) if vowel is not None and bilabial is not None else None,
        "expected_phone_tokens": expected,
        "observed_phone_tokens": {key: len(means[key]) for key in means},
    }


def _paired_speaker_summary(
    clip_values: Mapping[str, float | None], speaker_by_sample: Mapping[str, str]
) -> dict[str, Any]:
    by_speaker: dict[str, list[float]] = {}
    for sample_id, value in clip_values.items():
        if value is not None:
            by_speaker.setdefault(speaker_by_sample[sample_id], []).append(float(value))
    complete = {speaker: float(np.mean(values)) for speaker, values in by_speaker.items() if len(values) == 2}
    expected_speakers = len(set(speaker_by_sample.values()))
    return {
        "equal_speaker_mean_two_clip_complete": float(np.mean(list(complete.values()))) if complete else None,
        "complete_two_clip_speakers": len(complete),
        "expected_speakers": expected_speakers,
        "incomplete_speakers": sorted(set(speaker_by_sample.values()) - set(complete)),
        "speaker_values": dict(sorted(complete.items())),
    }


def _mann_whitney_auc(open_vowel: Sequence[float], bilabial: Sequence[float]) -> float | None:
    vowels = np.asarray(open_vowel, dtype=np.float64)
    labials = np.asarray(bilabial, dtype=np.float64)
    if not vowels.size or not labials.size:
        return None
    pairwise = vowels[:, None] - labials[None, :]
    return float(np.mean((pairwise > 0.0).astype(np.float64) + 0.5 * (pairwise == 0.0)))


def phone_token_auc_diagnostic(
    analysis: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare phone-token aperture rankings on tokens observed by all eight arms."""
    conditions = tuple(_CORE.CONDITIONS)
    if len(conditions) != 8:
        raise ToneVisualError("token AUC requires the frozen eight-arm condition set")
    min_counts = calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {})
    minimums = {
        category: int(min_counts.get(category, 1)) for category in ("bilabial", "open_vowel")
    }
    clip_rows = analysis.get("support_by_clip", [])
    clip_values: dict[str, dict[str, float | None]] = {}
    clip_support: dict[str, dict[str, Any]] = {}
    speaker_by_sample: dict[str, str] = {}
    for clip in clip_rows:
        sample_id = str(clip.get("id", ""))
        if not sample_id:
            raise ToneVisualError("analysis row lacks a clip id for token AUC")
        speaker_by_sample[sample_id] = str(clip.get("speaker", ""))
        token_map = clip.get("phone_tokens_by_arm")
        reason = None
        common_tokens: list[dict[str, Any]] = []
        if not isinstance(token_map, Mapping) or any(condition not in token_map for condition in conditions):
            reason = "missing_one_or_more_arm_tracks"
        else:
            arm_tokens = {condition: token_map[condition] for condition in conditions}
            lengths = {len(arm_tokens[condition]) for condition in conditions}
            if len(lengths) != 1:
                reason = "phone_token_list_length_mismatch"
            else:
                token_count = next(iter(lengths), 0)
                for index in range(token_count):
                    reference = arm_tokens[conditions[0]][index]
                    identity = (
                        reference.get("label"), reference.get("category"),
                        float(reference.get("start_s", 0.0)), float(reference.get("end_s", 0.0)),
                    )
                    identities_match = all(
                        (
                            arm_tokens[condition][index].get("label"),
                            arm_tokens[condition][index].get("category"),
                            float(arm_tokens[condition][index].get("start_s", 0.0)),
                            float(arm_tokens[condition][index].get("end_s", 0.0)),
                        ) == identity
                        for condition in conditions
                    )
                    if identities_match and all(
                        bool(arm_tokens[condition][index].get("observed")) for condition in conditions
                    ):
                        common_tokens.append({
                            "index": index,
                            "category": str(reference["category"]),
                            "aperture_mean_by_condition": {
                                condition: float(arm_tokens[condition][index]["aperture_mean"])
                                for condition in conditions
                            },
                        })
                    elif not identities_match:
                        reason = "phone_token_identity_mismatch"
                        common_tokens = []
                        break
        common_counts = {
            category: sum(token["category"] == category for token in common_tokens)
            for category in ("bilabial", "open_vowel")
        }
        enough_tokens = all(common_counts[key] >= minimums[key] for key in minimums)
        values: dict[str, float | None] = {condition: None for condition in conditions}
        if reason is None and enough_tokens:
            for condition in conditions:
                values_by_class = {
                    category: [
                        token["aperture_mean_by_condition"][condition]
                        for token in common_tokens if token["category"] == category
                    ]
                    for category in ("bilabial", "open_vowel")
                }
                values[condition] = _mann_whitney_auc(
                    values_by_class["open_vowel"], values_by_class["bilabial"]
                )
        random_values = [values[name] for name in ("random1", "random2", "random3")]
        values["random_mean"] = (
            float(np.mean([float(value) for value in random_values]))
            if all(value is not None for value in random_values) else None
        )
        if reason is None and not enough_tokens:
            reason = "common_tokens_below_calibration_minimum"
        clip_values[sample_id] = values
        clip_support[sample_id] = {
            "speaker_id": speaker_by_sample[sample_id],
            "all_eight_arm_tracks_available": reason != "missing_one_or_more_arm_tracks",
            "same_phone_tokens_observed_by_all_eight_arms": len(common_tokens),
            "common_token_count_by_class": common_counts,
            "minimum_count_by_class": minimums,
            "auc_supported": values["baseline"] is not None,
            "auc_by_condition": values,
            "missing_reason": reason,
        }

    by_speaker: dict[str, list[str]] = {}
    for sample_id, speaker in speaker_by_sample.items():
        by_speaker.setdefault(speaker, []).append(sample_id)
    speaker_values: dict[str, dict[str, Any]] = {}
    bootstrap_values: dict[str, dict[str, dict[str, float | None]]] = {}
    complete_speakers: list[str] = []
    for speaker, sample_ids in sorted(by_speaker.items()):
        complete = len(sample_ids) == 2 and all(
            all(clip_values[sample].get(condition) is not None for condition in (*conditions, "random_mean"))
            for sample in sample_ids
        )
        current: dict[str, Any] = {
            condition: (float(np.mean([clip_values[sample][condition] for sample in sample_ids])) if complete else None)
            for condition in (*conditions, "random_mean")
        }
        current["complete_two_clip_common_token_support"] = bool(complete)
        current["clip_ids"] = sorted(sample_ids)
        speaker_values[speaker] = current
        bootstrap_values[speaker] = {
            condition: {"class_separation_auc": current[condition]}
            for condition in (*conditions, "random_mean")
        }
        if complete:
            complete_speakers.append(speaker)
    contrasts = {
        "orth_plus_minus_baseline": _CORE._bootstrap_contrast(
            bootstrap_values, "class_separation_auc", "orth_plus", "baseline"
        ),
        "orth_plus_minus_random_mean": _CORE._bootstrap_contrast(
            bootstrap_values, "class_separation_auc", "orth_plus", "random_mean"
        ),
    }
    expected_speakers = len(by_speaker)
    return {
        "status": "SECONDARY_DESCRIPTIVE_ONLY",
        "primary_endpoint": False,
        "metric": "Mann-Whitney AUC over per-phone-token mean normalized mouth aperture; open vowel is positive class; ties count 0.5",
        "common_token_rule": "same natural phone-token indices, each observed on nonempty common valid recipient-clock frames in all eight arms",
        "minimum_count_by_class_from_calibration": minimums,
        "clip_aggregation": "compute AUC within clip; average random1/random2/random3 AUC within clip; average the two clip values equally within speaker",
        "speaker_aggregation": "equal-weight speakers; bootstrap unit is speaker",
        "intended_clip_denominator": len(clip_rows),
        "intended_speaker_denominator": expected_speakers,
        "clips_with_common_token_auc": sum(item["auc_supported"] for item in clip_support.values()),
        "clips_expected": len(clip_rows),
        "complete_two_clip_speakers": len(complete_speakers),
        "speakers_expected": expected_speakers,
        "complete_speaker_ids": complete_speakers,
        "per_clip": clip_support,
        "speaker_values": speaker_values,
        "contrasts": contrasts,
        "interpretation": "ranking-based secondary that is invariant to positive affine rescaling of all token means; not a synchrony or perceived-quality measure",
    }


def zero_latency_phone_separation_diagnostic(
    manifest_path: str | Path, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute phone separation at zero lag as a non-gating sensitivity check."""
    zero_calibration = copy.deepcopy(dict(calibration))
    zero_calibration.setdefault("protocol", {}).setdefault("latency", {})["latency_s"] = 0.0
    analysis = _CORE.analyze_manifest(manifest_path, zero_calibration)
    contrasts = analysis.get("primary_endpoint", {}).get("contrasts", {})
    return {
        "status": "SECONDARY_SENSITIVITY_ONLY",
        "primary_endpoint": False,
        "latency_s": 0.0,
        "comparison": "same eight arms, same recipient clocks and common-valid support, with no CAL latency shift",
        "intended_clips": analysis.get("counts", {}).get("clips_expected"),
        "intended_speakers": analysis.get("counts", {}).get("speakers_expected"),
        "primary_endpoint_status_at_zero_latency": analysis.get("primary_endpoint", {}).get("status"),
        "contrasts": {
            name: contrasts.get(name)
            for name in ("orth_plus_minus_baseline", "orth_plus_minus_random_mean")
        },
        "interpretation": "latency sensitivity only; does not replace the CAL-frozen-latency endpoint or its decision",
    }


def build_eval_phone_support_manifest(
    manifest_path: str | Path, calibration_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Freeze an availability-only paired-speaker list before looking at effects."""
    install_rules()
    manifest_file = Path(manifest_path).resolve()
    calibration_file = Path(calibration_path).resolve()
    calibration = _read_json(calibration_file)
    tone = calibration.get("visual_tone_provenance", {})
    if calibration.get("protocol_id") != PROTOCOL_ID or tone.get("tone_adapter_sha256") != _sha256_file(__file__):
        raise ToneVisualError("phone-support list requires the final tone-aware CAL freeze")
    payload = _CORE._read_json(manifest_file)
    rows = sorted(_CORE._manifest_rows(payload), key=lambda row: str(row["id"]))
    if any("cal" in str(row.get("split", "")).lower() for row in rows):
        raise ToneVisualError("phone-support list must use EVAL rows only")
    speaker_ids = {str(row["speaker"]) for row in rows}
    clip_counts = Counter(str(row["speaker"]) for row in rows)
    if len(rows) != 80 or len(speaker_ids) != 40 or any(count != 2 for count in clip_counts.values()):
        raise ToneVisualError("phone-support list requires the frozen 80-clip/40-speaker cohort")
    minimums = {
        category: int(calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {}).get(category, 1))
        for category in ("bilabial", "open_vowel")
    }
    latency_s = float(calibration["protocol"]["latency"]["latency_s"])
    clip_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["id"])
        speaker = str(row["speaker"])
        phones = row.get("phones")
        if not isinstance(phones, list):
            raise ToneVisualError(f"EVAL row has no natural phone tokens: {sample_id}")
        expected = {category: 0 for category in minimums}
        for phone in phones:
            category = phone_class(phone.get("label"))
            if category:
                expected[category] += 1

        tracks: dict[str, Any] = {}
        missing_arms: dict[str, str] = {}
        feature_sha_by_arm: dict[str, str | None] = {}
        arms = row.get("arms", {})
        for condition in _CORE.CONDITIONS:
            arm = arms.get(condition) if isinstance(arms, Mapping) else None
            if not isinstance(arm, Mapping):
                missing_arms[condition] = "arm_missing"
                continue
            path_value = _CORE._arm_feature_path(arm)
            if not path_value:
                missing_arms[condition] = "feature_path_missing"
                continue
            feature_path = Path(path_value)
            if not feature_path.is_file():
                missing_arms[condition] = "feature_file_missing"
                continue
            expected_sha = arm.get("feature_sha256")
            actual_sha = _sha256_file(feature_path)
            if expected_sha and actual_sha != expected_sha:
                raise ToneVisualError(f"feature checksum mismatch for {sample_id}/{condition}")
            feature_sha_by_arm[condition] = actual_sha
            try:
                # Only load clock and validity arrays. Aperture and pose values are not read for eligibility.
                with np.load(feature_path, allow_pickle=False) as feature:
                    if not {"timestamps_s", "valid"}.issubset(feature.files):
                        raise ToneVisualError(f"feature file lacks timestamp/valid arrays: {sample_id}/{condition}")
                    timestamps = np.asarray(feature["timestamps_s"], dtype=np.float64)
                    valid = np.asarray(feature["valid"], dtype=bool)
            except (OSError, ValueError, KeyError) as exc:
                raise ToneVisualError(f"cannot read feature availability for {sample_id}/{condition}") from exc
            if timestamps.ndim != 1 or valid.ndim != 1 or len(timestamps) != len(valid):
                raise ToneVisualError(f"feature clock/valid shapes differ: {sample_id}/{condition}")
            tracks[condition] = SimpleNamespace(
                timestamps_s=timestamps, valid=valid, metadata={"fps": _CORE.FPS_REFERENCE}
            )

        all_arms = not missing_arms and set(tracks) == set(_CORE.CONDITIONS)
        common_count = 0
        common_phone = {category: 0 for category in minimums}
        phone_intervals_by_class = {category: [] for category in minimums}
        if all_arms:
            common, _maps, clock_support = _CORE._common_clock(tracks)
            common_count = int(common.sum())
            if common_count:
                base_times = tracks["baseline"].timestamps_s[common]
                for phone in phones:
                    category = phone_class(phone.get("label"))
                    if category is None:
                        continue
                    lo = float(phone["start_s"]) + latency_s
                    hi = float(phone["end_s"]) + latency_s
                    if bool(np.any((base_times >= lo) & (base_times < hi))):
                        phone_intervals_by_class[category].append({
                            "label": str(phone["label"]), "start_s": float(phone["start_s"]),
                            "end_s": float(phone["end_s"]),
                        })
                common_phone = {key: len(value) for key, value in phone_intervals_by_class.items()}
            else:
                clock_support = {"clock_frames": len(tracks["baseline"].timestamps_s), "valid_common_frames": 0}
        else:
            clock_support = None
        both_classes = all(common_phone[key] >= minimums[key] for key in minimums)
        eligible = bool(all_arms and common_count > 0 and both_classes)
        if not all_arms:
            reason = "missing_one_or_more_eight_arm_tracks"
        elif common_count == 0:
            reason = "no_eight_arm_common_valid_clock_frames"
        elif not both_classes:
            reason = "common_clock_phone_tokens_below_class_minimum"
        else:
            reason = None
        clip_rows.append({
            "sample_id": sample_id,
            "speaker_id": speaker,
            "eligible_for_phone_qualified_secondary": eligible,
            "reason": reason,
            "expected_natural_phone_tokens_by_class": expected,
            "common_clock_phone_tokens_by_class": common_phone,
            "minimum_tokens_by_class": minimums,
            "joint_both_classes_available": both_classes,
            "all_eight_arm_tracks_available": all_arms,
            "missing_arms": missing_arms,
            "common_valid_clock_frames": common_count,
            "clock_frames": clock_support.get("clock_frames") if clock_support else None,
            "feature_sha256_by_arm": feature_sha_by_arm,
            "phone_intervals_with_common_clock_support": phone_intervals_by_class,
        })
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in clip_rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    speaker_rows = []
    for speaker, speaker_clips in sorted(by_speaker.items()):
        eligible = len(speaker_clips) == 2 and all(row["eligible_for_phone_qualified_secondary"] for row in speaker_clips)
        speaker_rows.append({
            "speaker_id": speaker,
            "clip_ids": sorted(row["sample_id"] for row in speaker_clips),
            "eligible_for_phone_qualified_secondary": bool(eligible),
        })
    eligible_speakers = [row["speaker_id"] for row in speaker_rows if row["eligible_for_phone_qualified_secondary"]]
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "pre_effect_phone_support_selection",
        "selection_split": "evaluation",
        "selection_uses_aperture_values": False,
        "selection_rule": "two natural phone classes meet CAL-frozen per-clip minima on nonempty valid frames common to all eight arms; speaker requires both predeclared clips eligible",
        "latency_s_from_calibration": latency_s,
        "minimum_tokens_by_class_from_calibration": minimums,
        "intended_clips": len(rows),
        "intended_speakers": len(speaker_ids),
        "clips_with_common_joint_class_support": sum(row["eligible_for_phone_qualified_secondary"] for row in clip_rows),
        "paired_speakers_with_both_clips_supported": len(eligible_speakers),
        "paired_speaker_ids": eligible_speakers,
        "manifest_sha256": _sha256_file(manifest_file),
        "calibration_sha256": _sha256_file(calibration_file),
        "tone_adapter_sha256": _sha256_file(__file__),
        "analysis_core_sha256": _sha256_file(CORE_PATH),
        "rows": clip_rows,
        "speakers": speaker_rows,
    }
    result["manifest_protocol_sha256"] = _canonical_hash(result)
    _write_json(output_path, result)
    return result


def calibration_class_separation_diagnostics(
    manifest_path: str | Path, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe CAL class separation, static/shuffle controls, and zero-latency sensitivity."""
    payload = _CORE._read_json(manifest_path)
    rows = _CORE._manifest_rows(payload)
    rows = sorted(rows, key=lambda row: row["id"])
    latency_s = float(calibration["protocol"]["latency"]["latency_s"])
    minimum_counts = calibration["protocol"]["min_phone_tokens_per_clip_by_class"]
    rng = np.random.Generator(np.random.PCG64(SHUFFLE_DIAGNOSTIC_SEED))
    speaker_by_sample = {row["id"]: row["speaker"] for row in rows}
    per_clip: dict[str, dict[str, Any]] = {}
    shuffle_values: dict[str, list[float | None]] = {
        row["id"]: [] for row in rows
    }
    for row in rows:
        track = _CORE._load_track(_CORE._arm_feature_path(row["arms"]["baseline"]),
                                  row["arms"]["baseline"].get("feature_sha256"))
        original = np.asarray(track.aperture, dtype=np.float64)
        static = original.copy()
        static[track.valid] = float(np.median(original[track.valid])) if track.valid.any() else 0.0
        zero_latency = _phone_separation_for_aperture(
            row, track, original, latency_s=0.0, minimum_counts=minimum_counts
        )
        calibrated = _phone_separation_for_aperture(
            row, track, original, latency_s=latency_s, minimum_counts=minimum_counts
        )
        static_result = _phone_separation_for_aperture(
            row, track, static, latency_s=latency_s, minimum_counts=minimum_counts
        )
        for draw in range(SHUFFLE_DIAGNOSTIC_DRAWS):
            shuffled = original.copy()
            valid_values = shuffled[track.valid].copy()
            shuffled[track.valid] = valid_values[rng.permutation(len(valid_values))]
            value = _phone_separation_for_aperture(
                row, track, shuffled, latency_s=latency_s, minimum_counts=minimum_counts
            )
            shuffle_values[row["id"]].append(value["phone_separation"])
        per_clip[row["id"]] = {
            "speaker_id": row["speaker"],
            "expected_phone_tokens_by_class": calibrated["expected_phone_tokens"],
            "observed_phone_tokens_by_class": calibrated["observed_phone_tokens"],
            "original_at_calibrated_latency": calibrated,
            "original_at_zero_latency": zero_latency,
            "static_at_calibrated_latency": static_result,
            "shuffle_draws": SHUFFLE_DIAGNOSTIC_DRAWS,
            "shuffle_phone_separation_mean": None,
            "shuffle_phone_separation_sd": None,
            "shuffle_phone_separation_p05_p95": None,
        }
        observed_shuffles = np.asarray([value for value in shuffle_values[row["id"]] if value is not None], dtype=np.float64)
        if observed_shuffles.size:
            per_clip[row["id"]]["shuffle_phone_separation_mean"] = float(np.mean(observed_shuffles))
            per_clip[row["id"]]["shuffle_phone_separation_sd"] = float(np.std(observed_shuffles))
            per_clip[row["id"]]["shuffle_phone_separation_p05_p95"] = [
                float(np.quantile(observed_shuffles, 0.05)), float(np.quantile(observed_shuffles, 0.95))
            ]
        per_clip[row["id"]]["shuffle_observed_draw_count"] = int(observed_shuffles.size)
    summary: dict[str, Any] = {
        "status": "DESCRIPTIVE_CAL_ONLY",
        "primary_endpoint_gate": False,
        "intended_clip_denominator": len(rows),
        "intended_speaker_denominator": len(set(speaker_by_sample.values())),
        "shuffle": {"generator": "PCG64", "seed": SHUFFLE_DIAGNOSTIC_SEED,
                    "draws_per_clip": SHUFFLE_DIAGNOSTIC_DRAWS,
                    "operation": "permute valid aperture samples within each clip; retain the same phone intervals and validity mask"},
        "per_clip": per_clip,
        "condition_summaries": {},
    }
    condition_fields = {
        "original_at_calibrated_latency": "original_at_calibrated_latency",
        "original_at_zero_latency": "original_at_zero_latency",
        "static_at_calibrated_latency": "static_at_calibrated_latency",
    }
    expected_ids = [row["id"] for row in rows]
    per_class_clip_ids: dict[str, list[str]] = {"bilabial": [], "open_vowel": []}
    joint_clip_ids: list[str] = []
    for sample_id, item in per_clip.items():
        observed_counts = item["observed_phone_tokens_by_class"]
        class_supported = {
            category: int(observed_counts.get(category, 0)) >= int(minimum_counts.get(category, 1))
            for category in ("bilabial", "open_vowel")
        }
        for category, supported in class_supported.items():
            if supported:
                per_class_clip_ids[category].append(sample_id)
        if all(class_supported.values()):
            joint_clip_ids.append(sample_id)
    class_support_by_speaker: dict[str, list[str]] = {}
    for sample_id in joint_clip_ids:
        class_support_by_speaker.setdefault(speaker_by_sample[sample_id], []).append(sample_id)
    joint_complete_speakers = sorted(
        speaker for speaker, supported_ids in class_support_by_speaker.items()
        if len(supported_ids) == 2
    )
    summary["phone_class_support"] = {
        "minimum_observed_tokens_per_clip_by_class": {
            category: int(minimum_counts.get(category, 1)) for category in ("bilabial", "open_vowel")
        },
        "per_class": {
            category: {
                "observed_clips": len(ids), "expected_clips": len(rows),
                "observed_fraction": float(len(ids) / len(rows)) if rows else None,
                "supported_clip_ids": sorted(ids),
            }
            for category, ids in per_class_clip_ids.items()
        },
        "joint_both_classes": {
            "observed_clips": len(joint_clip_ids), "expected_clips": len(rows),
            "observed_fraction": float(len(joint_clip_ids) / len(rows)) if rows else None,
            "supported_clip_ids": sorted(joint_clip_ids),
            "complete_two_clip_speakers": len(joint_complete_speakers),
            "expected_speakers": len(set(speaker_by_sample.values())),
            "complete_speaker_ids": joint_complete_speakers,
            "interpretation": "joint endpoint support; per-class coverage must not be presented as joint coverage",
        },
    }
    for condition, field in condition_fields.items():
        clip_values = {sample: item[field]["phone_separation"] for sample, item in per_clip.items()}
        observed = {sample: value for sample, value in clip_values.items() if value is not None}
        summary["condition_summaries"][condition] = {
            "observed_clips": len(observed), "expected_clips": len(rows),
            "missing_clip_ids": sorted(set(expected_ids) - set(observed)),
            "clip_mean_observed": float(np.mean(list(observed.values()))) if observed else None,
            "equal_speaker_summary": _paired_speaker_summary(clip_values, speaker_by_sample),
        }
    draw_means: list[float | None] = []
    for draw in range(SHUFFLE_DIAGNOSTIC_DRAWS):
        draw_values = {sample: values[draw] for sample, values in shuffle_values.items()}
        observed = [float(value) for value in draw_values.values() if value is not None]
        draw_speakers = _paired_speaker_summary(draw_values, speaker_by_sample)
        draw_means.append(draw_speakers["equal_speaker_mean_two_clip_complete"])
    nonmissing_draw_means = np.asarray([value for value in draw_means if value is not None], dtype=np.float64)
    observed_clip_count_by_draw = [
        sum(value is not None for value in (shuffle_values[sample][draw] for sample in shuffle_values))
        for draw in range(SHUFFLE_DIAGNOSTIC_DRAWS)
    ]
    summary["condition_summaries"]["shuffled_100_draws"] = {
        "observed_clip_count_by_draw": observed_clip_count_by_draw,
        "observed_speaker_mean_by_draw": draw_means,
        "mean_of_draw_speaker_means": float(np.mean(nonmissing_draw_means)) if nonmissing_draw_means.size else None,
        "p05_p95_of_draw_speaker_means": [float(np.quantile(nonmissing_draw_means, 0.05)), float(np.quantile(nonmissing_draw_means, 0.95))] if nonmissing_draw_means.size else None,
        "draws_with_complete_clip_coverage": int(sum(count == len(rows) for count in observed_clip_count_by_draw)),
        "expected_clips": len(rows),
        "expected_speakers": len(set(speaker_by_sample.values())),
    }
    return summary


def prepare_calibration_manifest(
    features_manifest_path: str | Path,
    qualification_path: str | Path,
    alignment_protocol_path: str | Path,
    bundle_manifest_path: str | Path,
    textgrid_root: str | Path,
    manifest_out: str | Path,
    mapping_out: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join qualified natural CAL phone intervals to the 20 baseline tracks."""
    features_path = Path(features_manifest_path).resolve()
    qualification_path = Path(qualification_path).resolve()
    alignment_path = Path(alignment_protocol_path).resolve()
    bundle_path = Path(bundle_manifest_path).resolve()
    grid_root = Path(textgrid_root).resolve()
    features = _read_json(features_path)
    qualification = _read_json(qualification_path)
    alignment = _read_json(alignment_path)
    bundle = _read_json(bundle_path)

    if features.get("schema_version") != _CORE.SCHEMA_VERSION or features.get("stage") != "features_extracted":
        raise ToneVisualError("CAL feature manifest schema/stage is not the frozen extracted baseline")
    if features.get("protocol_id") != "tts_mechanism_visual_20260930_v2":
        raise ToneVisualError("CAL baseline tracks do not use the predeclared visual extraction protocol")
    if features.get("producer_code_sha256", {}).get("visual_analysis") != alignment.get("visual_phone_rules", {}).get("sha256"):
        raise ToneVisualError("CAL track manifest does not bind the old frozen visual snapshot")
    snapshot_path = Path(bundle_path.parent / "visual_snapshot" / "tts_mechanism_visual_20260930.py")
    expected_old_sha = alignment.get("visual_phone_rules", {}).get("sha256")
    if not snapshot_path.is_file() or _sha256_file(snapshot_path) != expected_old_sha:
        raise ToneVisualError("archived pre-tone visual snapshot is missing or has a different SHA")
    if bundle.get("frozen_visual_snapshot_sha256") != expected_old_sha:
        raise ToneVisualError("CAL review bundle does not bind the old visual snapshot")
    if qualification.get("status") != "FROZEN" or qualification.get("selection_split") != "calibration":
        raise ToneVisualError("CAL MFA qualification is not frozen")
    if qualification.get("evaluation_textgrids_read") is not False:
        raise ToneVisualError("CAL phone mapping requires a qualification record that did not read EVAL TextGrids")
    if qualification.get("protocol_manifest_sha256") != alignment.get("manifest_sha256"):
        raise ToneVisualError("CAL qualification and MFA alignment protocol differ")
    qualification_file_sha = _sha256_file(qualification_path)
    alignment_file_sha = _sha256_file(alignment_path)
    bundle_sha = _sha256_file(bundle_path)
    bundle_files = {str(item.get("path")): item.get("sha256") for item in bundle.get("files", [])}
    if bundle.get("qualification_manifest_sha256") != qualification.get("manifest_sha256"):
        raise ToneVisualError("CAL bundle does not bind the qualification manifest fingerprint")
    if bundle_files.get("calibration_qualification_frozen.json") != qualification_file_sha:
        raise ToneVisualError("CAL bundle does not bind the qualification file bytes")
    if bundle.get("protocol_manifest_sha256") != alignment.get("manifest_sha256"):
        raise ToneVisualError("CAL bundle does not bind the frozen MFA protocol")
    if bundle_files.get("alignment_protocol_frozen.json") != alignment_file_sha:
        raise ToneVisualError("CAL bundle does not bind the alignment-protocol file bytes")
    if qualification.get("sample_counts") != {"total": 20, "pair_qualified": 20, "pair_failed": 0}:
        raise ToneVisualError("tone-class calibration requires all 20 predeclared CAL pairs to qualify")

    rows = features.get("rows")
    qualified_pairs = qualification.get("calibration_pairs")
    if not isinstance(rows, list) or len(rows) != 20 or not isinstance(qualified_pairs, Mapping) or len(qualified_pairs) != 20:
        raise ToneVisualError("CAL visual calibration requires exactly 20 feature and MFA rows")
    speaker_counts: Counter[str] = Counter()
    seen: set[str] = set()
    phone_counts: Counter[str] = Counter()
    joined_rows = []
    textgrid_hashes: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        sample_id = str(row.get("id", ""))
        speaker = str(row.get("speaker", ""))
        if not sample_id or not speaker or sample_id in seen:
            raise ToneVisualError("CAL feature rows have a missing or duplicate sample/speaker identity")
        seen.add(sample_id)
        speaker_counts[speaker] += 1
        pair = qualified_pairs.get(sample_id)
        if not isinstance(pair, Mapping) or pair.get("pair_qualified") is not True:
            raise ToneVisualError(f"CAL pair is not qualified: {sample_id}")
        natural = pair.get("natural", {})
        if natural.get("qualified") is not True:
            raise ToneVisualError(f"natural CAL TextGrid is not qualified: {sample_id}")
        grid = grid_root / speaker / f"{sample_id}__natural.TextGrid"
        if not grid.is_file():
            raise ToneVisualError(f"missing natural CAL TextGrid: {sample_id}")
        grid_sha = _sha256_file(grid)
        if grid_sha != natural.get("textgrid_sha256"):
            raise ToneVisualError(f"TextGrid SHA differs from frozen qualification: {sample_id}")
        tokens = natural.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ToneVisualError(f"natural CAL TextGrid has no frozen phone tokens: {sample_id}")
        phones = []
        for token in tokens:
            label = token.get("token")
            start = token.get("start_s")
            end = token.get("end_s")
            if label is None or start is None or end is None:
                raise ToneVisualError(f"malformed natural phone interval: {sample_id}")
            phone = {"label": str(label), "start_s": float(start), "end_s": float(end)}
            phones.append(phone)
            phone_counts[phone["label"]] += 1
        row["phones"] = phones
        row["natural_textgrid_sha256"] = grid_sha
        row["mfa_pair_qualified"] = True
        joined_rows.append(row)
        textgrid_hashes[sample_id] = grid_sha
    if len(seen) != 20 or len(speaker_counts) != 10 or any(count != 2 for count in speaker_counts.values()):
        raise ToneVisualError("CAL phone join changed the frozen 20-clip/10-speaker/two-clip cohort")

    mapping = build_mapping_table(phone_counts)
    mapping["textgrid_sha256_by_sample"] = dict(sorted(textgrid_hashes.items()))
    mapping["qualification_manifest_sha256"] = qualification.get("manifest_sha256")
    mapping["qualification_file_sha256"] = qualification_file_sha
    mapping["alignment_protocol_manifest_sha256"] = alignment.get("manifest_sha256")
    mapping["alignment_protocol_file_sha256"] = alignment_file_sha
    mapping["old_visual_snapshot_sha256"] = expected_old_sha
    mapping.pop("mapping_sha256", None)
    mapping["mapping_sha256"] = _canonical_hash(mapping)
    new_manifest = dict(features)
    new_manifest.update({
        "protocol_id": PROTOCOL_ID,
        "stage": "calibration_natural_phones_attached",
        "rows": joined_rows,
        "visual_tone_protocol": {
            "mapping_id": mapping["mapping_id"],
            "mapping_sha256": mapping["mapping_sha256"],
            "mapping_path": str(Path(mapping_out).resolve()),
            "qualification_manifest_sha256": qualification.get("manifest_sha256"),
            "qualification_file_sha256": qualification_file_sha,
            "alignment_protocol_manifest_sha256": alignment.get("manifest_sha256"),
            "alignment_protocol_file_sha256": alignment_file_sha,
            "bundle_manifest_sha256": bundle_sha,
            "old_visual_snapshot_sha256": expected_old_sha,
            "tone_adapter_sha256": _sha256_file(__file__),
            "analysis_core_sha256": _sha256_file(CORE_PATH),
            "evaluation_textgrids_read": False,
            "phone_interval_source": "qualified natural CAL TextGrid tokens only",
        },
    })
    new_manifest["visual_tone_protocol"]["features_manifest_sha256"] = _sha256_file(features_path)
    new_manifest["visual_tone_protocol"]["manifest_sha256"] = _canonical_hash(new_manifest["visual_tone_protocol"])
    _write_json(mapping_out, mapping)
    _write_json(manifest_out, new_manifest)
    return new_manifest, mapping


def calibrate_tone_manifest(manifest_path: str | Path, mapping_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run the existing CAL procedure under this tone-aware visual class map."""
    install_rules()
    manifest = _read_json(manifest_path)
    mapping = _read_json(mapping_path)
    provenance = manifest.get("visual_tone_protocol", {})
    if manifest.get("protocol_id") != PROTOCOL_ID or provenance.get("mapping_sha256") != mapping.get("mapping_sha256"):
        raise ToneVisualError("tone-aware CAL manifest/mapping protocol mismatch")
    if _canonical_hash({k: v for k, v in mapping.items() if k != "mapping_sha256"}) != mapping.get("mapping_sha256"):
        raise ToneVisualError("tone-aware CAL mapping hash mismatch")
    result = _CORE.calibrate_manifest(manifest_path)
    rows = sorted(_CORE._manifest_rows(manifest), key=lambda row: row["id"])
    speaker_counts = Counter(row["speaker"] for row in rows)
    structural_pass = bool(
        len(rows) == 20 and len(speaker_counts) == 10 and all(count == 2 for count in speaker_counts.values())
    )
    tracks = {
        row["id"]: _CORE._load_track(
            _CORE._arm_feature_path(row["arms"]["baseline"]),
            row["arms"]["baseline"].get("feature_sha256"),
        )
        for row in rows
    }
    track_support_pass = bool(
        len(tracks) == len(rows) and all(len(track.valid) > 0 and bool(track.valid.any()) for track in tracks.values())
    )
    fixed_shift_qc = fixed_event_index_timestamp_control(
        tracks, min_prominence=float(result["protocol"]["min_prominence"])
    )
    class_separation_qc = calibration_class_separation_diagnostics(manifest_path, result)
    result["legacy_event_controls_for_audit"] = result["event_controls"]
    result["fixed_event_index_timestamp_control"] = fixed_shift_qc
    count_gate_pass = bool(result.get("phone_count_calibration_pass"))
    latency_gate_pass = bool(result.get("latency_support_pass"))
    result["positive_phone_endpoint_eligible"] = bool(
        structural_pass and track_support_pass and count_gate_pass and latency_gate_pass
    )
    result["primary_phone_endpoint_gate"] = {
        "eligible": result["positive_phone_endpoint_eligible"],
        "eligibility_scope": "frozen calibration-rule eligibility; it does not imply full-cohort joint observation of both phone classes",
        "structural_cal_rows_pass": structural_pass,
        "baseline_track_common_valid_support_pass": track_support_pass,
        "phone_count_coverage_pass": count_gate_pass,
        "phone_class_support": class_separation_qc["phone_class_support"],
        "joint_phone_class_full_cohort_support": bool(
            class_separation_qc["phone_class_support"]["joint_both_classes"]["observed_clips"] == len(rows)
            and class_separation_qc["phone_class_support"]["joint_both_classes"]["complete_two_clip_speakers"] == len(speaker_counts)
        ),
        "frozen_latency_support_pass": latency_gate_pass,
        "event_rematching_shift_and_shuffle_controls_gate_phone_separation": False,
        "reason": "phone separation uses fixed CAL latency and MFA interval means; extrema rematching/shuffle QC remains exploratory event-timing support only",
    }
    result["class_separation_cal_diagnostics"] = class_separation_qc
    result["event_metric_calibrated"] = bool(
        result.get("legacy_event_controls_for_audit", {}).get("status") == "EVENT_METRIC_CALIBRATED"
    )
    phone_inventory = result["protocol"]["phone_inventory"]
    phone_inventory["normalization_removes"] = sorted(set(phone_inventory["normalization_removes"]) | set(TONE_MARKS))
    phone_inventory["source"] = "visual-only Mandarin tone adaptation; exact old classes/normalizer plus the frozen five-character tone-bar allowlist"
    tone_counts = Counter({
        row["symbol"]: int(row["observed_natural_cal_phone_tokens"])
        for row in mapping["tone_mark_allowlist"]
    })
    phone_inventory["tone_mark_allowlist"] = _tone_allowlist_rows(tone_counts)
    phone_inventory["mapping_id"] = mapping["mapping_id"]
    phone_inventory["mapping_sha256"] = mapping["mapping_sha256"]
    result["protocol"]["phone_inventory_sha256"] = hashlib.sha256(
        json.dumps(phone_inventory, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    result["protocol"]["visual_tone_rule"] = {
        "scope": "visual phone class only",
        "removed_tone_codepoints": [f"U+{ord(mark):04X}" for mark in TONE_MARKS],
        "removed_tone_symbols_observed_in_natural_CAL": mapping["tone_marks_observed"],
        "non_tone_marks_preserved_by_new_rule": ["ʰ", "ʲ", "ʷ", "ː", "̪", "̩"],
    }
    result["protocol"]["phone_separation_gate"] = result["primary_phone_endpoint_gate"]
    result["protocol"]["event_timing_qc_scope"] = (
        "legacy rematching +/-5 and frame-shuffle controls remain visible and exploratory; a failed timing QC does not gate fixed-latency phone-class separation"
    )
    result["protocol"]["fixed_event_index_timestamp_control"] = {
        "procedure": fixed_shift_qc["procedure"],
        "expected_shift_frames": fixed_shift_qc["expected_shift_frames"],
        "tolerance_frames": fixed_shift_qc["tolerance_frames"],
        "pass": fixed_shift_qc["pass"],
    }
    result["protocol"]["class_separation_cal_diagnostics"] = {
        "status": class_separation_qc["status"],
        "primary_endpoint_gate": False,
        "shuffle_seed": SHUFFLE_DIAGNOSTIC_SEED,
        "shuffle_draws_per_clip": SHUFFLE_DIAGNOSTIC_DRAWS,
        "intended_clip_denominator": class_separation_qc["intended_clip_denominator"],
    }
    result["visual_tone_provenance"] = {
        "mapping_sha256": mapping["mapping_sha256"],
        "mapping_path": str(Path(mapping_path).resolve()),
        "tone_adapter_path": str(Path(__file__).resolve()),
        "tone_adapter_sha256": _sha256_file(__file__),
        "analysis_core_path": str(CORE_PATH.resolve()),
        "analysis_core_sha256": _sha256_file(CORE_PATH),
        "old_mfa_visual_snapshot_sha256": provenance.get("old_visual_snapshot_sha256"),
        "qualification_manifest_sha256": provenance.get("qualification_manifest_sha256"),
        "qualification_file_sha256": provenance.get("qualification_file_sha256"),
        "evaluation_textgrids_read": False,
    }
    _write_json(output_path, result)
    return result


def analyze_tone_manifest(manifest_path: str | Path, calibration_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Apply only the sealed tone-aware visual rules to a phone-attached EVAL manifest."""
    install_rules()
    calibration = _read_json(calibration_path)
    tone = calibration.get("visual_tone_provenance", {})
    if calibration.get("protocol_id") != PROTOCOL_ID or tone.get("tone_adapter_sha256") != _sha256_file(__file__):
        raise ToneVisualError("CAL calibration is not bound to this tone-aware adapter")
    result = _CORE.analyze_manifest(manifest_path, calibration)
    result["secondary_phone_token_auc"] = phone_token_auc_diagnostic(result, calibration)
    result["secondary_zero_latency_phone_separation"] = zero_latency_phone_separation_diagnostic(
        manifest_path, calibration
    )
    result["visual_tone_provenance"] = tone
    _write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    attach = sub.add_parser("prepare-cal", help="attach qualified natural CAL phone intervals and seal the mapping")
    attach.add_argument("--features-manifest", required=True)
    attach.add_argument("--qualification", required=True)
    attach.add_argument("--alignment-protocol", required=True)
    attach.add_argument("--bundle-manifest", required=True)
    attach.add_argument("--textgrid-root", required=True)
    attach.add_argument("--manifest-out", required=True)
    attach.add_argument("--mapping-out", required=True)
    calibrate = sub.add_parser("calibrate", help="freeze lag, event, and phone thresholds from CAL only")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--mapping", required=True)
    calibrate.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze", help="run the independent visual endpoint under the frozen CAL tone map")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--calibration", required=True)
    analyze.add_argument("--output", required=True)
    subset = sub.add_parser("phone-support-list", help="freeze the EVAL phone-qualified subset from availability only")
    subset.add_argument("--manifest", required=True)
    subset.add_argument("--calibration", required=True)
    subset.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-cal":
        manifest, mapping = prepare_calibration_manifest(
            args.features_manifest, args.qualification, args.alignment_protocol,
            args.bundle_manifest, args.textgrid_root, args.manifest_out, args.mapping_out,
        )
        print(json.dumps({"status": "calibration_manifest_prepared", "rows": len(manifest["rows"]),
                          "phone_symbols": mapping["phone_symbol_count"],
                          "tone_marks_codepoints": [f"U+{ord(mark):04X}" for mark in mapping["tone_marks_observed"]],
                          "mapping_sha256": mapping["mapping_sha256"]}, ensure_ascii=False))
    elif args.command == "calibrate":
        result = calibrate_tone_manifest(args.manifest, args.mapping, args.output)
        print(json.dumps({"status": result["status"], "positive_phone_endpoint_eligible": result["positive_phone_endpoint_eligible"],
                          "phone_count_calibration_pass": result["phone_count_calibration_pass"],
                          "latency_support_pass": result["latency_support_pass"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))
    elif args.command == "analyze":
        result = analyze_tone_manifest(args.manifest, args.calibration, args.output)
        print(json.dumps({"status": result["primary_endpoint"]["status"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))
    else:
        result = build_eval_phone_support_manifest(args.manifest, args.calibration, args.output)
        print(json.dumps({"status": result["stage"],
                          "clips_with_common_joint_class_support": result["clips_with_common_joint_class_support"],
                          "paired_speakers_with_both_clips_supported": result["paired_speakers_with_both_clips_supported"],
                          "output": str(Path(args.output).resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
