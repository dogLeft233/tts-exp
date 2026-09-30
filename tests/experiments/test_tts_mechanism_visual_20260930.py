from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.experiments import tts_mechanism_visual_20260930 as visual  # noqa: E402


def _phones(cycles: int = 6) -> list[dict[str, object]]:
    phones: list[dict[str, object]] = []
    for cycle in range(cycles):
        vowel = 0.5 + 2.0 * cycle
        labial = 1.5 + 2.0 * cycle
        phones.append({"label": "a", "start_s": vowel - 0.08, "end_s": vowel + 0.08})
        phones.append({"label": "p", "start_s": labial - 0.08, "end_s": labial + 0.08})
    return phones


def _track(amplitude: float = 0.04, *, phase: float = 0.0, n: int = 300) -> visual.MouthTrack:
    times = np.arange(n, dtype=np.float64) / 25.0
    aperture = 0.14 + amplitude * np.sin(np.pi * (times - phase))
    valid = np.ones(n, dtype=bool)
    landmarks = np.zeros((n, 478, 2), dtype=np.float64)
    landmarks[:, 33] = [0.2, 0.25]
    landmarks[:, 263] = [0.8, 0.25]
    landmarks[:, 1] = [0.5, 0.45]
    landmarks[:, 13] = [0.5, 0.55]
    landmarks[:, 14] = [0.5, 0.56]
    width = np.full(n, 0.42, dtype=np.float64)
    return visual.MouthTrack(times, aperture, valid, width, landmarks, {"fps": 25.0})


def _write_track(path: Path, track: visual.MouthTrack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps_s=track.timestamps_s,
        mouth_open=track.aperture,
        valid=track.valid,
        mouth_width=track.mouth_width,
        landmarks=track.landmarks,
        metadata_json=np.asarray(json.dumps(track.metadata), dtype=np.str_),
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "rows": rows}), encoding="utf-8")


def _make_rows(
    tmp_path: Path,
    *,
    evaluation: bool,
    cycles_by_clip: tuple[int, int] = (6, 6),
    orth_plus_by_clip: tuple[float, float] = (0.075, 0.075),
) -> tuple[Path, dict[str, list[Path]]]:
    rows: list[dict[str, object]] = []
    tracks: dict[str, list[Path]] = {}
    speakers = [f"spk{index:02d}" for index in range(4 if not evaluation else 3)]
    for speaker_index, speaker in enumerate(speakers):
        for clip_index in range(2):
            sample_id = f"{speaker}_utt{clip_index}"
            cycles = cycles_by_clip[clip_index]
            phone_rows = _phones(cycles)
            duration_frames = max(300, int((phone_rows[-1]["end_s"] + 0.2) * 25) + 1)
            amplitudes = {
                "baseline": 0.04,
                "full_plus": 0.06,
                "full_minus": 0.03,
                "orth_plus": orth_plus_by_clip[clip_index],
                "orth_minus": 0.025,
                "random1": 0.035,
                "random2": 0.04,
                "random3": 0.045,
            }
            arms: dict[str, dict[str, str]] = {}
            feature_paths = []
            conditions = visual.CONDITIONS if evaluation else ("baseline",)
            for condition in conditions:
                path = tmp_path / ("eval" if evaluation else "cal") / sample_id / f"{condition}.npz"
                _write_track(path, _track(amplitudes[condition], phase=speaker_index * 0.0001, n=duration_frames))
                feature_paths.append(path)
                arms[condition] = {"feature_path": str(path)}
            tracks[sample_id] = feature_paths
            rows.append({
                "id": sample_id,
                "speaker": speaker,
                "phones": phone_rows,
                "arms": arms,
            })
    manifest = tmp_path / ("eval.json" if evaluation else "cal.json")
    _write_manifest(manifest, rows)
    return manifest, tracks


def test_phone_classes_use_the_frozen_mfa_mapping() -> None:
    assert visual.phone_class("pʰ") == "bilabial"
    assert visual.phone_class("mʲ") == "bilabial"
    assert visual.phone_class("a") == "open_vowel"
    assert visual.phone_class("ɑː") == "open_vowel"
    assert visual.phone_class("n") is None


def test_ordered_event_match_is_one_to_one_and_penalizes_misses() -> None:
    result = visual._ordered_match([1.0, 1.1, 3.0], [{"time_s": 1.05}, {"time_s": 3.2}], tolerance_s=0.25)
    assert result["expected_count"] == 3
    assert result["matched_count"] == 2
    assert result["missed_count"] == 1
    assert result["unmatched_detected_count"] == 0
    assert result["penalized_abs_error_s"] == pytest.approx((0.05 + 0.2 + 0.25) / 3)


def test_calibration_freezes_latency_and_validates_temporal_controls(tmp_path: Path) -> None:
    manifest, _ = _make_rows(tmp_path, evaluation=False)
    result = visual.calibrate_manifest(manifest)
    controls = result["event_controls"]
    assert result["calibration_speakers"] == 4
    assert result["protocol"]["latency"]["latency_s"] is not None
    assert controls["known_shift_controls"]["plus5_sign_and_size_pass"] is True
    assert controls["known_shift_controls"]["minus5_sign_and_size_pass"] is True
    assert controls["static_repeat_control"]["pass"] is True
    assert controls["frame_shuffle_control"]["aperture_distribution_exactly_preserved"] is True
    assert controls["frame_shuffle_control"]["median_penalized_error_increased"] is True
    assert result["protocol"]["min_phone_tokens_per_clip_by_class"] == {"bilabial": 6, "open_vowel": 6}


def test_common_clock_retains_full_recipient_duration_and_tracks_missing_frames() -> None:
    baseline = _track(n=300)
    changed_valid = baseline.valid.copy()
    changed_valid[120] = False
    changed_aperture = baseline.aperture.copy()
    changed_aperture[120] = 0.0
    changed_landmarks = baseline.landmarks.copy()
    changed_landmarks[120] = 0.0
    changed = visual.MouthTrack(
        baseline.timestamps_s, changed_aperture, changed_valid,
        baseline.mouth_width.copy(), changed_landmarks, baseline.metadata,
    )
    support, maps, details = visual._common_clock({"baseline": baseline, "orth_plus": changed})
    assert len(support) == 300
    assert support.sum() == 299
    assert maps["orth_plus"][120] == 120
    assert details["clock_frames"] == 300
    assert details["common_coverage"] == pytest.approx(299 / 300)


def test_fixed_audio_phone_separation_requires_both_primary_contrasts(tmp_path: Path) -> None:
    calibration_manifest, _ = _make_rows(tmp_path, evaluation=False)
    calibration = visual.calibrate_manifest(calibration_manifest)
    evaluation_manifest, _ = _make_rows(tmp_path, evaluation=True)
    result = visual.analyze_manifest(evaluation_manifest, calibration)
    primary = result["primary_endpoint"]
    assert result["counts"]["clips_expected"] == 6
    assert result["counts"]["speakers_expected"] == 3
    assert primary["positive_confirmation"] is True
    assert primary["status"] == "INDEPENDENT_PHONE_SEPARATION_CONFIRMED"
    assert primary["contrasts"]["orth_plus_minus_baseline"]["ci99"][0] > 0
    assert primary["contrasts"]["orth_plus_minus_random_mean"]["ci99"][0] > 0
    assert result["event_timing_exploratory"]["primary_endpoint"] is False
    for clip in result["support_by_clip"]:
        arm_values = clip["phone_class_means_and_separation_by_arm"]
        expected_random = np.mean([arm_values[name]["phone_separation"] for name in ("random1", "random2", "random3")])
        assert clip["random_phone_separation_mean"] == pytest.approx(expected_random)
        assert clip["support"]["all_8_arm_common_valid_frames"] == clip["support"]["common_valid_frames"]


def test_analysis_rejects_calibration_from_another_protocol_version(tmp_path: Path) -> None:
    calibration_manifest, _ = _make_rows(tmp_path, evaluation=False)
    calibration = visual.calibrate_manifest(calibration_manifest)
    evaluation_manifest, _ = _make_rows(tmp_path, evaluation=True)
    calibration["protocol_id"] = "tts_mechanism_visual_20260930_v1"
    with pytest.raises(visual.VisualMechanismError, match="protocol_id"):
        visual.analyze_manifest(evaluation_manifest, calibration)


def test_missing_arm_stays_in_clip_and_speaker_denominators(tmp_path: Path) -> None:
    calibration_manifest, _ = _make_rows(tmp_path, evaluation=False)
    calibration = visual.calibrate_manifest(calibration_manifest)
    evaluation_manifest, _ = _make_rows(tmp_path, evaluation=True)
    payload = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    payload["rows"][0]["arms"].pop("random2")
    _write_manifest(evaluation_manifest, payload["rows"])

    result = visual.analyze_manifest(evaluation_manifest, calibration)
    primary = result["primary_endpoint"]
    assert len(result["support_by_clip"]) == 6
    assert result["support_by_clip"][0]["observed"] is False
    assert result["support_by_clip"][0]["missing_arms"] == {"random2": "arm_missing"}
    contrast = primary["contrasts"]["orth_plus_minus_random_mean"]
    assert contrast["expected_speakers"] == 3
    assert contrast["observed_speakers"] == 2
    assert contrast["missing_speakers"] == ["spk00"]
    assert primary["positive_confirmation"] is False


def test_primary_estimand_weights_clips_equally_before_speakers(tmp_path: Path) -> None:
    calibration_manifest, _ = _make_rows(tmp_path, evaluation=False)
    calibration = visual.calibrate_manifest(calibration_manifest)
    evaluation_manifest, _ = _make_rows(
        tmp_path, evaluation=True,
        cycles_by_clip=(6, 12), orth_plus_by_clip=(0.075, 0.04),
    )
    result = visual.analyze_manifest(evaluation_manifest, calibration)
    speaker = result["speaker_values"]["spk00"]
    # Clip 2 contains twice as many tokens, so this verifies that the speaker
    # contrast averages the two clip contrasts instead of pooling tokens.
    clips = [row for row in result["support_by_clip"] if row["speaker"] == "spk00"]
    clip_effects = [
        row["phone_class_means_and_separation_by_arm"]["orth_plus"]["phone_separation"]
        - row["phone_class_means_and_separation_by_arm"]["baseline"]["phone_separation"]
        for row in clips
    ]
    observed = speaker["orth_plus"]["phone_separation"] - speaker["baseline"]["phone_separation"]
    assert observed == pytest.approx(float(np.mean(clip_effects)))
    assert observed != pytest.approx((clip_effects[0] + 2 * clip_effects[1]) / 3)
