"""Tests for landmark geometry in scripts/frame_detectors.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_frame_detectors", _SCRIPTS / "frame_detectors.py"
)
assert _spec is not None
fd = importlib.util.module_from_spec(_spec)
sys.modules["_frame_detectors"] = fd
assert _spec.loader is not None
_spec.loader.exec_module(fd)


def make_frontal_landmarks(w=640, h=640, eye_dist=100.0):
    """Synthetic 478-point landmarks: frontal face centered at (cx, cy)."""
    cx, cy = w / 2, h / 2
    pts = np.tile(np.array([cx / w, cy / h], dtype=np.float64), (478, 1))
    pts[33] = [(cx - eye_dist / 2) / w, cy / h]
    pts[263] = [(cx + eye_dist / 2) / w, cy / h]
    pts[1] = [cx / w, (cy + 30) / h]
    pts[13] = [cx / w, (cy + 90) / h]
    pts[14] = [cx / w, (cy + 92) / h]
    pts[159] = [(cx - eye_dist / 4) / w, (cy - 5) / h]
    pts[145] = [(cx - eye_dist / 4) / w, (cy + 5) / h]
    pts[386] = [(cx + eye_dist / 4) / w, (cy - 5) / h]
    pts[374] = [(cx + eye_dist / 4) / w, (cy + 5) / h]
    return pts


def test_frontal_face_yaw_near_zero():
    m = fd._metrics_from_landmarks(make_frontal_landmarks(), 640, 640)
    assert m is not None
    assert abs(m.yaw_deg) < 5.0
    assert m.face_area_px > 0
    assert m.mouth_open_ratio is not None


def test_profile_face_increases_yaw():
    frontal = make_frontal_landmarks()
    profile = frontal.copy()
    profile[1] = [0.5 + 0.12, 0.55]  # nose offset right -> yaw grows
    yaw_frontal = fd._metrics_from_landmarks(frontal, 640, 640).yaw_deg
    yaw_profile = fd._metrics_from_landmarks(profile, 640, 640).yaw_deg
    assert yaw_profile > yaw_frontal + 10.0


def test_mouth_open_increases_ratio():
    closed = make_frontal_landmarks()
    opened = closed.copy()
    opened[14] = [0.5, (0.5 + 90 + 25) / 640]
    r_closed = fd._metrics_from_landmarks(closed, 640, 640).mouth_open_ratio
    r_opened = fd._metrics_from_landmarks(opened, 640, 640).mouth_open_ratio
    assert r_opened > r_closed


def test_eye_open_ear_computed():
    m = fd._metrics_from_landmarks(make_frontal_landmarks(), 640, 640)
    assert m.eye_open is not None and m.eye_open > 0.0
