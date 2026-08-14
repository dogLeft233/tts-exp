from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_aishell1_n25_audio_artifacts.py"
spec = importlib.util.spec_from_file_location("audio_artifact_audit", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_robust_threshold_is_deterministic_and_resistant_to_outlier() -> None:
    values = np.array([1.0, 1.1, 0.9, 1.05, 1.02, 30.0])
    first = module.robust_threshold(values)
    second = module.robust_threshold(values)
    assert first == second
    assert first < 30.0


def test_impulse_detector_finds_localized_discontinuity() -> None:
    time = np.arange(int(1.2 * module.SAMPLE_RATE)) / module.SAMPLE_RATE
    clean = 0.05 * np.sin(2 * np.pi * 180 * time)
    values = clean.copy()
    values[int(0.55 * module.SAMPLE_RATE)] += 1.0
    _, control_local = module.extract_audio_features(clean.astype(np.float32))
    _, local = module.extract_audio_features(values.astype(np.float32))
    thresholds = module.calibrate_thresholds([control_local])
    events = module.detect_local_events(local, thresholds, values.size)
    impulse = [event for event in events if event["kind"] == "impulsive_discontinuity"]
    assert impulse
    assert any(0.45 < event["start_s"] < 0.7 for event in impulse)


def test_repetition_detector_finds_nontrivial_repeated_chunk() -> None:
    rng = np.random.default_rng(4)
    chunk = rng.normal(0, 0.2, int(0.24 * module.SAMPLE_RATE)).astype(np.float32)
    values = np.concatenate([rng.normal(0, 0.02, int(0.35 * module.SAMPLE_RATE)), chunk, rng.normal(0, 0.02, int(0.3 * module.SAMPLE_RATE)), chunk])
    _, local = module.extract_audio_features(values)
    first = module.detect_repetitions(local)
    assert first
    assert any(event["lag_s"] > 0.25 and event["duration_s"] >= 0.10 for event in first)


def test_repetition_detector_does_not_flag_short_periodicity() -> None:
    values = 0.2 * np.sin(2 * np.pi * 180 * np.arange(int(2.0 * module.SAMPLE_RATE)) / module.SAMPLE_RATE)
    _, local = module.extract_audio_features(values.astype(np.float32))
    events = module.detect_repetitions(local)
    assert not events


def test_json_safe_replaces_nonfinite_values() -> None:
    payload = module.json_safe({"nan": float("nan"), "inf": float("inf"), "ok": np.float32(2.5)})
    assert payload == {"nan": None, "inf": None, "ok": 2.5}
