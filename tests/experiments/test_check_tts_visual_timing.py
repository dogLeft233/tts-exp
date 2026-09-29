from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/experiments"))

import check_tts_visual_timing as checker  # noqa: E402


def test_checker_recomputes_empty_flat_sequence_as_low_motion():
    landmarks = np.zeros((30, 478, 2), dtype=np.float64)
    landmarks[:, 33] = [0.2, 0.2]
    landmarks[:, 263] = [0.8, 0.2]
    landmarks[:, 13] = [0.5, 0.5]
    landmarks[:, 14] = [0.5, 0.5]
    landmarks[:, 61] = [0.3, 0.5]
    landmarks[:, 291] = [0.7, 0.5]
    result = checker._independent_events(landmarks, np.ones(30, dtype=bool), np.arange(30) / 25.0, 100, 100)
    assert result["status"] == "LOW_MOTION"
    assert result["events"] == []


def test_checker_event_distance_keeps_reference_denominator():
    result = checker._distance([{"time_s": 1.0}, {"time_s": 2.0}], [])
    assert result["distance_ms"] == 240.0
    assert result["missing_reference_count"] == 2
