from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_DIR = REPO / "scripts/experiments/lrs3_tts_visual_advantage"
sys.path.insert(0, str(MODULE_DIR))

import video_features as video_features_module  # noqa: E402
from video_features import MOUTH_INDICES, VisualSequence, mouth_features, save_visual_sequence, load_visual_sequence  # noqa: E402
from visual_metrics import exact_time_distance, visual_dtw_distance  # noqa: E402


def make_sequence(frame_count: int = 8, transform: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> VisualSequence:
    scale, shift_x, shift_y = transform
    landmarks = np.zeros((frame_count, 478, 2), dtype=np.float64)
    for index in range(frame_count):
        openness = 0.02 + 0.01 * index
        landmarks[index, 33] = [0.0, 0.0]
        landmarks[index, 263] = [2.0, 0.0]
        landmarks[index, 1] = [1.0, 0.5]
        landmarks[index, MOUTH_INDICES] = np.array([
            [0.35, -0.25], [0.45, -0.24], [0.55, -0.23], [0.65, -0.22],
            [0.75, -0.21], [1.00, -0.20], [1.25, -0.21], [1.35, -0.22],
            [1.45, -0.23], [1.55, -0.24], [1.65, -0.25], [1.70, -0.26],
            [1.60, 0.28], [1.45, 0.29], [1.30, 0.30], [1.15, 0.31],
            [1.00, openness], [0.85, 0.31], [0.70, 0.30], [0.55, 0.29],
            [0.40, 0.28], [0.30, 0.27], [0.25, 0.10], [0.35, 0.05],
            [0.55, 0.02], [1.00, 0.01], [1.45, 0.02], [1.65, 0.05],
            [1.75, 0.10], [1.70, -0.10], [1.65, -0.20],
        ])
        landmarks[index] = landmarks[index] * scale + [shift_x, shift_y]
    features = mouth_features(landmarks)
    return VisualSequence(
        landmarks=landmarks,
        canonical_mouth=features["canonical_mouth"],
        valid=features["valid"],
        mouth_open=features["mouth_open"],
        mouth_width=features["mouth_width"],
        mouth_height=features["mouth_height"],
        mouth_area=features["mouth_area"],
        timestamps_s=np.arange(frame_count, dtype=np.float64) / 25.0,
        metadata={"synthetic": True, "schema_version": 1},
    )


def test_canonicalization_removes_translation_and_scale():
    reference = make_sequence()
    transformed = make_sequence(transform=(3.0, 10.0, -4.0))
    result = exact_time_distance(reference, transformed)
    assert result.distance < 1e-12
    assert result.coverage == 1.0


def test_exact_time_detects_shape_and_temporal_changes():
    reference = make_sequence()
    changed = make_sequence()
    changed.landmarks[:, MOUTH_INDICES, 1] += 0.4
    changed_features = mouth_features(changed.landmarks)
    changed = VisualSequence(
        landmarks=changed.landmarks,
        canonical_mouth=changed_features["canonical_mouth"],
        valid=changed_features["valid"],
        mouth_open=changed_features["mouth_open"],
        mouth_width=changed_features["mouth_width"],
        mouth_height=changed_features["mouth_height"],
        mouth_area=changed_features["mouth_area"],
        timestamps_s=changed.timestamps_s,
        metadata=changed.metadata,
    )
    assert exact_time_distance(reference, changed).distance > 0.1


def test_invalid_landmark_frame_is_excluded():
    landmarks = make_sequence().landmarks.copy()
    landmarks[2, 33] = np.nan
    features = mouth_features(landmarks)
    assert not features["valid"][2]
    assert np.all(features["canonical_mouth"][2] == 0.0)


def test_visual_dtw_accepts_monotonic_time_warp_but_not_reverse():
    reference = make_sequence(frame_count=20)
    warped_landmarks = reference.landmarks[
        [0, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ]
    warped_features = mouth_features(warped_landmarks)
    warped = VisualSequence(
        landmarks=warped_landmarks,
        canonical_mouth=warped_features["canonical_mouth"],
        valid=warped_features["valid"],
        mouth_open=warped_features["mouth_open"],
        mouth_width=warped_features["mouth_width"],
        mouth_height=warped_features["mouth_height"],
        mouth_area=warped_features["mouth_area"],
        timestamps_s=np.arange(20, dtype=np.float64) / 25.0,
        metadata={"synthetic": True},
    )
    exact = exact_time_distance(reference, warped)
    dtw = visual_dtw_distance(reference, warped, band_fraction=0.25, max_run=2)
    assert np.isfinite(exact.distance)
    assert np.isfinite(dtw.distance)
    assert dtw.distance < 1e-3

    reverse_landmarks = reference.landmarks[::-1]
    reverse_features = mouth_features(reverse_landmarks)
    reverse = VisualSequence(
        landmarks=reverse_landmarks,
        canonical_mouth=reverse_features["canonical_mouth"],
        valid=reverse_features["valid"],
        mouth_open=reverse_features["mouth_open"],
        mouth_width=reverse_features["mouth_width"],
        mouth_height=reverse_features["mouth_height"],
        mouth_area=reverse_features["mouth_area"],
        timestamps_s=np.arange(20, dtype=np.float64) / 25.0,
        metadata={"synthetic": True},
    )
    assert visual_dtw_distance(reference, reverse).distance > 0.0


def test_exact_time_pairing_is_monotonic_one_to_one_and_counts_tails():
    reference = make_sequence(frame_count=2)
    candidate = make_sequence(frame_count=2)
    candidate = VisualSequence(
        landmarks=candidate.landmarks,
        canonical_mouth=candidate.canonical_mouth,
        valid=candidate.valid,
        mouth_open=candidate.mouth_open,
        mouth_width=candidate.mouth_width,
        mouth_height=candidate.mouth_height,
        mouth_area=candidate.mouth_area,
        timestamps_s=np.asarray([0.01, 0.07], dtype=np.float64),
        metadata=candidate.metadata,
    )
    result = exact_time_distance(reference, candidate)
    assert result.valid_frames == 1
    assert result.coverage == pytest.approx(0.5)

    ambiguous_reference = replace(reference, timestamps_s=np.asarray([0.01, 0.02], dtype=np.float64))
    ambiguous_candidate = replace(candidate, timestamps_s=np.asarray([0.0, 0.019], dtype=np.float64))
    ambiguous = exact_time_distance(ambiguous_reference, ambiguous_candidate, tolerance_s=0.02)
    assert ambiguous.valid_frames == 2
    assert ambiguous.coverage == 1.0


def test_dtw_allows_configured_repeated_direction_runs():
    short = make_sequence(frame_count=1)
    long = make_sequence(frame_count=3)
    result = visual_dtw_distance(short, long, band_fraction=1.0, max_run=2)
    assert np.isfinite(result.distance)
    assert result.path_length == 3
    assert np.isnan(visual_dtw_distance(short, long, band_fraction=1.0, max_run=1).distance)


def test_dtw_band_preserves_original_positions_across_invalid_gap():
    reference = make_sequence(frame_count=5)
    candidate = make_sequence(frame_count=5)
    valid = candidate.valid.copy()
    valid[2] = False
    canonical = candidate.canonical_mouth.copy()
    canonical[2] = 0.0
    scalar_values = {
        "mouth_open": candidate.mouth_open.copy(),
        "mouth_width": candidate.mouth_width.copy(),
        "mouth_height": candidate.mouth_height.copy(),
        "mouth_area": candidate.mouth_area.copy(),
    }
    for value in scalar_values.values():
        value[2] = 0.0
    candidate = VisualSequence(
        landmarks=candidate.landmarks,
        canonical_mouth=canonical,
        valid=valid,
        mouth_open=scalar_values["mouth_open"],
        mouth_width=scalar_values["mouth_width"],
        mouth_height=scalar_values["mouth_height"],
        mouth_area=scalar_values["mouth_area"],
        timestamps_s=candidate.timestamps_s,
        metadata=candidate.metadata,
    )
    result = visual_dtw_distance(reference, candidate, band_fraction=0.1, max_run=2)
    assert np.isnan(result.distance)


def test_unrelated_invalid_landmark_does_not_invalidate_mouth_features():
    landmarks = make_sequence().landmarks.copy()
    landmarks[2, 100] = np.nan
    features = mouth_features(landmarks)
    assert features["valid"][2]


def test_invalid_derived_features_must_be_zero():
    source = make_sequence()
    canonical = source.canonical_mouth.copy()
    canonical[0, 0, 0] = 1.0
    valid = source.valid.copy()
    valid[0] = False
    with pytest.raises(ValueError):
        VisualSequence(
            landmarks=source.landmarks,
            canonical_mouth=canonical,
            valid=valid,
            mouth_open=source.mouth_open,
            mouth_width=source.mouth_width,
            mouth_height=source.mouth_height,
            mouth_area=source.mouth_area,
            timestamps_s=source.timestamps_s,
            metadata=source.metadata,
        )


def test_visual_sequence_round_trip(tmp_path: Path):
    source = make_sequence()
    path = tmp_path / "features.npz"
    save_visual_sequence(path, source)
    restored = load_visual_sequence(path)
    np.testing.assert_array_equal(restored.landmarks, source.landmarks)
    np.testing.assert_array_equal(restored.canonical_mouth, source.canonical_mouth)
    np.testing.assert_array_equal(restored.valid, source.valid)
    np.testing.assert_array_equal(restored.mouth_open, source.mouth_open)
    np.testing.assert_array_equal(restored.mouth_width, source.mouth_width)
    np.testing.assert_array_equal(restored.mouth_height, source.mouth_height)
    np.testing.assert_array_equal(restored.mouth_area, source.mouth_area)
    np.testing.assert_array_equal(restored.timestamps_s, source.timestamps_s)
    assert restored.metadata == source.metadata


def test_extractor_fails_closed_on_malformed_decoded_frame(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "malformed.mp4"
    video_path.write_bytes(b"placeholder")

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            return {
                1: 25.0,
                2: 2.0,
                3: 2.0,
                4: 2.0,
                5: 0.0,
            }[property_id]

        def read(self):
            return True, np.zeros((2, 2), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _: capture,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FOURCC=5,
    )
    fake_landmarker = SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(video_features_module, "_landmarker", lambda _: (SimpleNamespace(), fake_landmarker))

    with pytest.raises(ValueError, match="malformed decoded frame"):
        video_features_module.extract_video_features(video_path, tmp_path / "asset.task")
    assert capture.released


def test_visual_sequence_loader_rejects_schema_coercion(tmp_path: Path):
    source = make_sequence()
    arrays = {
        "landmarks": source.landmarks,
        "canonical_mouth": source.canonical_mouth,
        "mouth_open": source.mouth_open,
        "mouth_width": source.mouth_width,
        "mouth_height": source.mouth_height,
        "mouth_area": source.mouth_area,
        "timestamps_s": source.timestamps_s,
    }
    invalid_archives = (
        (source.valid.astype(np.uint8), {"schema_version": 1}),
        (source.valid, {"schema_version": True}),
    )
    for index, (valid, metadata) in enumerate(invalid_archives):
        path = tmp_path / f"invalid_{index}.npz"
        np.savez_compressed(
            path,
            **arrays,
            valid=valid,
            metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
        )
        with pytest.raises(ValueError):
            load_visual_sequence(path)


def test_invalid_dtw_configuration_is_rejected():
    sequence = make_sequence()
    with pytest.raises(ValueError):
        visual_dtw_distance(sequence, sequence, band_fraction=1.1)
    with pytest.raises(ValueError):
        visual_dtw_distance(sequence, sequence, max_run=0)
