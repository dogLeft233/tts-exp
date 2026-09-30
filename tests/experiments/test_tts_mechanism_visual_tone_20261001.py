from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from scripts.experiments import tts_mechanism_visual_tone_20261001 as visual


@pytest.mark.parametrize("label", ["a˥", "a˥˩", "a˧˥", "a˨˩˦"])
def test_mandarin_tone_bars_are_removed_for_visual_open_vowel_class_only(label: str) -> None:
    assert visual.phone_class(label) == "open_vowel"


def test_explicit_tone_allowlist_matches_cal_inventory_codepoints() -> None:
    assert [(f"U+{ord(mark):04X}", name) for mark, name in visual.TONE_MARKS.items()] == [
        ("U+02E5", "MODIFIER LETTER EXTRA-HIGH TONE BAR"),
        ("U+02E6", "MODIFIER LETTER HIGH TONE BAR"),
        ("U+02E7", "MODIFIER LETTER MID TONE BAR"),
        ("U+02E8", "MODIFIER LETTER LOW TONE BAR"),
        ("U+02E9", "MODIFIER LETTER EXTRA-LOW TONE BAR"),
    ]


def test_visual_mapping_preserves_non_tone_phone_distinctions() -> None:
    assert visual.normalize_visual_phone("aj˥") == "aj"
    assert visual.phone_class("aj˥") is None
    assert visual.phone_class("e˥") is None
    assert visual.phone_class("à") is None  # U+0300 is not in the tone-bar allowlist.
    assert visual.phone_class("p˥") == "bilabial"
    assert visual.phone_class("pʰ") == "bilabial"  # Existing visual aspiration rule is unchanged.


def test_mapping_table_records_each_raw_label_and_observed_mark() -> None:
    table = visual.build_mapping_table(Counter({"a˥": 2, "a˨˩˦": 1, "aj˥": 1, "p": 3}))
    rows = {row["raw_phone"]: row for row in table["phone_rows"]}
    assert rows["a˥"]["visual_base_phone"] == "a"
    assert rows["a˥"]["visual_class"] == "open_vowel"
    assert rows["aj˥"]["visual_base_phone"] == "aj"
    assert rows["aj˥"]["visual_class"] is None
    assert rows["p"]["visual_class"] == "bilabial"
    assert table["tone_marks_observed"] == ["˥", "˦", "˨", "˩"]
    assert table["phone_token_count"] == 7
    unsigned = {key: value for key, value in table.items() if key != "mapping_sha256"}
    expected = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert table["mapping_sha256"] == expected


def test_rules_bind_the_frozen_analysis_core_and_new_protocol_id() -> None:
    visual.install_rules()
    assert visual._CORE.PROTOCOL_ID == visual.PROTOCOL_ID
    assert visual._CORE.phone_class is visual.phone_class


def test_fixed_event_index_timestamp_control_recovers_both_known_shifts() -> None:
    count = 101
    timestamps = np.arange(count, dtype=float) / 25.0
    aperture = 0.5 + 0.25 * np.sin(2.0 * np.pi * timestamps * 3.0)
    valid = np.ones(count, dtype=bool)
    landmarks = np.zeros((count, 478, 2), dtype=float)
    track = visual._CORE.MouthTrack(
        timestamps_s=timestamps,
        aperture=aperture,
        valid=valid,
        mouth_width=np.ones(count, dtype=float),
        landmarks=landmarks,
        metadata={"fps": 25.0},
    )
    result = visual.fixed_event_index_timestamp_control({"sample": track}, min_prominence=0.01)
    assert result["pass"] is True
    assert result["event_indices_unchanged_for_all_clips"] is True
    assert result["aperture_multiset_preserved_for_all_clips"] is True
    assert result["median_plus_timestamp_delta_frames"] == pytest.approx(5.0, abs=1e-6)
    assert result["median_minus_timestamp_delta_frames"] == pytest.approx(-5.0, abs=1e-6)


def test_mann_whitney_auc_counts_ties_half_and_ignores_positive_affine_rescaling() -> None:
    first = visual._mann_whitney_auc([0.5, 1.0], [0.5, 0.0])
    rescaled = visual._mann_whitney_auc([7.0, 12.0], [7.0, 2.0])
    assert first == pytest.approx(0.875)
    assert rescaled == pytest.approx(first)
    assert visual._mann_whitney_auc([1.0], [1.0]) == pytest.approx(0.5)
    assert visual._mann_whitney_auc([], [1.0]) is None


def test_phone_token_auc_uses_same_tokens_across_all_arms_and_equal_clip_weighting() -> None:
    conditions = visual._CORE.CONDITIONS
    clips = []
    for clip_index in range(2):
        token_rows = {
            "baseline": [
                {"label": "p", "category": "bilabial", "start_s": 0.0, "end_s": 0.1,
                 "observed": True, "aperture_mean": 0.1 + 0.1 * clip_index},
                {"label": "a˥", "category": "open_vowel", "start_s": 0.1, "end_s": 0.2,
                 "observed": True, "aperture_mean": 0.8 + 0.1 * clip_index},
            ]
        }
        for arm_index, condition in enumerate(conditions):
            if condition != "baseline":
                token_rows[condition] = [dict(token) for token in token_rows["baseline"]]
                token_rows[condition][0]["aperture_mean"] += 0.01 * arm_index
                token_rows[condition][1]["aperture_mean"] += 0.01 * arm_index
        clips.append({"id": f"clip{clip_index}", "speaker": "speaker1", "phone_tokens_by_arm": token_rows})
    diagnostics = visual.phone_token_auc_diagnostic(
        {"support_by_clip": clips},
        {"protocol": {"min_phone_tokens_per_clip_by_class": {"bilabial": 1, "open_vowel": 1}}},
    )
    assert diagnostics["clips_with_common_token_auc"] == 2
    assert diagnostics["complete_two_clip_speakers"] == 1
    assert diagnostics["per_clip"]["clip0"]["same_phone_tokens_observed_by_all_eight_arms"] == 2
    assert diagnostics["speaker_values"]["speaker1"]["baseline"] == pytest.approx(1.0)
    assert diagnostics["contrasts"]["orth_plus_minus_baseline"]["observed_speakers"] == 1


def test_phone_token_auc_preserves_missing_class_support_in_denominator() -> None:
    conditions = visual._CORE.CONDITIONS
    token_rows = {}
    for condition in conditions:
        token_rows[condition] = [
            {"label": "p", "category": "bilabial", "start_s": 0.0, "end_s": 0.1,
             "observed": True, "aperture_mean": 0.1},
            {"label": "a˥", "category": "open_vowel", "start_s": 0.1, "end_s": 0.2,
             "observed": condition != "random3", "aperture_mean": 0.9},
        ]
    result = visual.phone_token_auc_diagnostic(
        {"support_by_clip": [{"id": "clip", "speaker": "speaker", "phone_tokens_by_arm": token_rows}]},
        {"protocol": {"min_phone_tokens_per_clip_by_class": {"bilabial": 1, "open_vowel": 1}}},
    )
    assert result["intended_clip_denominator"] == 1
    assert result["clips_with_common_token_auc"] == 0
    assert result["per_clip"]["clip"]["common_token_count_by_class"]["open_vowel"] == 0
    assert result["per_clip"]["clip"]["missing_reason"] == "common_tokens_below_calibration_minimum"


def test_zero_latency_secondary_keeps_frozen_calibration_unmodified(monkeypatch) -> None:
    observed_latencies = []

    def fake_analyze(_manifest_path, calibration):
        observed_latencies.append(calibration["protocol"]["latency"]["latency_s"])
        return {
            "counts": {"clips_expected": 80, "speakers_expected": 40},
            "primary_endpoint": {
                "status": "INCOMPLETE_PAIRED_SPEAKER_SUPPORT",
                "contrasts": {
                    "orth_plus_minus_baseline": {"mean": 0.1},
                    "orth_plus_minus_random_mean": {"mean": 0.2},
                },
            },
        }

    monkeypatch.setattr(visual._CORE, "analyze_manifest", fake_analyze)
    calibration = {"protocol": {"latency": {"latency_s": -0.085}}}
    result = visual.zero_latency_phone_separation_diagnostic("unused.json", calibration)
    assert observed_latencies == [0.0]
    assert calibration["protocol"]["latency"]["latency_s"] == -0.085
    assert result["status"] == "SECONDARY_SENSITIVITY_ONLY"
    assert result["contrasts"]["orth_plus_minus_baseline"]["mean"] == 0.1


def test_eval_phone_support_list_uses_only_valid_clock_arrays(tmp_path) -> None:
    conditions = visual._CORE.CONDITIONS
    timestamps = np.arange(50, dtype=np.float64) / 25.0
    feature_paths = {}
    for condition in conditions:
        path = tmp_path / f"{condition}.npz"
        np.savez(path, timestamps_s=timestamps, valid=np.ones(len(timestamps), dtype=bool))
        feature_paths[condition] = path
    invalid_path = tmp_path / "one_invalid_clock.npz"
    np.savez(invalid_path, timestamps_s=timestamps, valid=np.zeros(len(timestamps), dtype=bool))
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "protocol_id": visual.PROTOCOL_ID,
        "visual_tone_provenance": {"tone_adapter_sha256": visual._sha256_file(visual.__file__)},
        "protocol": {
            "min_phone_tokens_per_clip_by_class": {"bilabial": 1, "open_vowel": 1},
            "latency": {"latency_s": 0.0},
        },
    }), encoding="utf-8")
    rows = []
    for speaker_index in range(40):
        for clip_index in range(2):
            sample_id = f"S{speaker_index:03d}_C{clip_index}"
            arms = {
                condition: {
                    "feature_path": str(feature_paths[condition]),
                    "feature_sha256": visual._sha256_file(feature_paths[condition]),
                }
                for condition in conditions
            }
            if speaker_index == 0 and clip_index == 1:
                arms["random3"] = {
                    "feature_path": str(invalid_path),
                    "feature_sha256": visual._sha256_file(invalid_path),
                }
            rows.append({
                "id": sample_id,
                "speaker": f"S{speaker_index:03d}",
                "split": "evaluation",
                "phones": [
                    {"label": "p", "start_s": 0.20, "end_s": 0.40},
                    {"label": "a˥", "start_s": 0.60, "end_s": 0.80},
                ],
                "arms": arms,
            })
    manifest_path = tmp_path / "eval_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": visual._CORE.SCHEMA_VERSION, "rows": rows}), encoding="utf-8")
    output_path = tmp_path / "phone_support.json"
    result = visual.build_eval_phone_support_manifest(manifest_path, calibration_path, output_path)
    assert result["selection_uses_aperture_values"] is False
    assert result["intended_clips"] == 80
    assert result["clips_with_common_joint_class_support"] == 79
    assert result["paired_speakers_with_both_clips_supported"] == 39
    invalid_row = next(row for row in result["rows"] if row["sample_id"] == "S000_C1")
    assert invalid_row["common_valid_clock_frames"] == 0
    assert "aperture" not in invalid_row
    assert output_path.is_file()
