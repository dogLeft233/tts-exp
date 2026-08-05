"""Tests for frame quality selection (scripts/frame_quality.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_frame_quality",
    _SCRIPTS / "frame_quality.py",
)
assert _spec is not None
fq = importlib.util.module_from_spec(_spec)
sys.modules["_frame_quality"] = fq
assert _spec.loader is not None
_spec.loader.exec_module(fq)


def test_compute_blur_score_ranks_sharp_above_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    blurred = cv2_median_blur(sharp, ksize=9)
    assert fq.compute_blur_score(sharp) > fq.compute_blur_score(blurred)


def cv2_median_blur(img, ksize):
    import cv2

    return cv2.medianBlur(img, ksize)


def metrics(yaw=0.0, area=100000.0, blur=100.0):
    return fq.FaceMetrics(yaw_deg=yaw, face_area_px=area, blur_score=blur)


def test_quality_score_prefers_frontal_over_profile():
    assert fq.quality_score(metrics(yaw=0.0)) > fq.quality_score(metrics(yaw=25.0))


def test_quality_score_prefers_larger_face():
    assert fq.quality_score(metrics(area=200000.0)) > fq.quality_score(metrics(area=50000.0))


def test_quality_score_prefers_sharper_frame():
    assert fq.quality_score(metrics(blur=200.0)) > fq.quality_score(metrics(blur=50.0))


def test_is_acceptable_rejects_profile_beyond_yaw_limit():
    assert fq.is_acceptable(metrics(yaw=20.0))
    assert not fq.is_acceptable(metrics(yaw=35.0))


def test_is_acceptable_rejects_tiny_face():
    assert fq.is_acceptable(metrics(area=25000.0))
    assert not fq.is_acceptable(metrics(area=3000.0))


def test_select_best_frame_picks_highest_scoring_acceptable_frame():
    table = {
        0: metrics(yaw=10.0, area=50000.0, blur=50.0),
        1: metrics(yaw=5.0, area=80000.0, blur=120.0),
        2: metrics(yaw=15.0, area=60000.0, blur=90.0),
    }
    idx, chosen = fq.select_best_frame([0, 1, 2], lambda frame: table[frame])
    assert idx == 1
    assert chosen == table[1]


def test_select_best_frame_skips_frames_without_detected_face():
    table = {
        0: None,
        1: metrics(yaw=0.0, area=100000.0, blur=100.0),
    }
    idx, _ = fq.select_best_frame([0, 1], lambda frame: table[frame])
    assert idx == 1


def test_select_best_frame_returns_none_when_none_acceptable():
    table = {
        0: metrics(yaw=45.0, area=3000.0, blur=1.0),
        1: metrics(yaw=40.0, area=2000.0, blur=1.0),
    }
    result = fq.select_best_frame([0, 1], lambda frame: table[frame])
    assert result is None
