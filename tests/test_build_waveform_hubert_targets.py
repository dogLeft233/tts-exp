#!/usr/bin/env python3
"""Tests for strict weak waveform target construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_waveform_hubert_targets import build_targets, validate_pair  # noqa: E402


def _write_pair_manifest(tmp_path: Path, *, repeated: bool = False) -> Path:
    sr = 16_000
    natural = np.sin(np.linspace(0, 4 * np.pi, sr, dtype=np.float32))
    tts = np.cos(np.linspace(0, 4 * np.pi, sr, dtype=np.float32))
    natural_path = tmp_path / "natural.wav"
    tts_path = tmp_path / "tts.wav"
    sf.write(natural_path, natural, sr)
    sf.write(tts_path, tts, sr)
    labels = ["a", "a"] if repeated else ["a", "b"]
    tokens = [
        {"label": label, "start_s": index * 0.5, "end_s": (index + 1) * 0.5, "confidence": 1.0}
        for index, label in enumerate(labels)
    ]
    records = [
        {
            "paired_key": "p1",
            "condition": "natural",
            "audio_path": str(natural_path),
            "transcript": "a b",
            "split": "train",
            "speaker_group": "spk1",
            "alignment_source": "mfa",
            "tokens": tokens,
        },
        {
            "paired_key": "p1",
            "condition": "tts",
            "audio_path": str(tts_path),
            "transcript": "a b",
            "split": "train",
            "speaker_group": "tts_provider",
            "alignment_source": "mfa",
            "tokens": tokens,
        },
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def _write_24k_pair_manifest(tmp_path: Path) -> tuple[Path, float]:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_sample_rate = 24_000
    source_sample_count = 24_001
    source_duration_s = source_sample_count / source_sample_rate
    tts_path = tmp_path / "tts_24k.wav"
    tts = np.cos(np.linspace(0, 4 * np.pi, source_sample_count, dtype=np.float32))
    sf.write(tts_path, tts, source_sample_rate)
    payload["records"][1]["audio_path"] = str(tts_path)
    payload["records"][1]["tokens"] = [
        {"label": "a", "start_s": 0.0, "end_s": source_duration_s / 2, "confidence": 1.0},
        {"label": "b", "start_s": source_duration_s / 2, "end_s": source_duration_s, "confidence": 1.0},
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, source_duration_s


def test_build_targets_creates_strict_weak_target(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    result = build_targets(manifest, tmp_path / "targets")
    assert len(result["accepted"]) == 1
    assert not result["excluded"]
    row = result["accepted"][0]
    assert row["target_definition"] == "weak_phone_local_tts_target"
    assert row["target_sha256"]
    assert row["target_sample_count"] == row["natural_sample_count"]
    assert Path(row["target_path"]).exists()


def test_24k_tts_spans_use_source_duration_and_record_conversion(tmp_path: Path) -> None:
    manifest, source_duration_s = _write_24k_pair_manifest(tmp_path)
    result = build_targets(manifest, tmp_path / "targets")
    assert len(result["accepted"]) == 1
    row = result["accepted"][0]
    assert row["tts_source_sample_rate"] == 24_000
    assert row["tts_source_sample_count"] == 24_001
    assert row["tts_source_duration_s"] == pytest.approx(source_duration_s)
    assert row["tts_sample_rate"] == 16_000
    assert row["tts_sample_count"] == 16_001
    assert row["tts_resampling"] == {
        "applied": True,
        "method": "scipy.signal.resample_poly",
        "source_sample_rate": 24_000,
        "target_sample_rate": 16_000,
        "source_duration_validation": "tts_alignment_spans_validated_against_source_duration_before_resampling",
        "up": 2,
        "down": 3,
        "converted_sample_count": 16_001,
        "converted_duration_s": pytest.approx(16_001 / 16_000),
    }


def test_24k_tts_span_past_source_duration_is_rejected(tmp_path: Path) -> None:
    manifest, source_duration_s = _write_24k_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][1]["tokens"][-1]["end_s"] = source_duration_s + 1e-4
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "outside audio duration" in result["excluded"][0]["reason"]


def test_unsupported_tts_sample_rate_is_rejected(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    tts_path = tmp_path / "tts_22k.wav"
    sf.write(tts_path, np.zeros(22_050, dtype=np.float32), 22_050)
    payload["records"][1]["audio_path"] = str(tts_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "16 kHz or explicit 24 kHz" in result["excluded"][0]["reason"]


def test_low_coverage_remains_rejected_after_24k_validation_fix(tmp_path: Path) -> None:
    manifest, _ = _write_24k_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][1]["tokens"][1]["label"] = "different"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "coverage" in result["excluded"][0]["reason"]


def test_repeated_phone_labels_remain_distinct_matched_spans(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path, repeated=True)
    result = build_targets(manifest, tmp_path / "targets")
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["matched_token_count"] == 2


def test_uniform_fallback_alignment_is_excluded(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][0]["alignment_source"] = "uniform_fallback"
    manifest.write_text(json.dumps(payload))
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "alignment_source" in result["excluded"][0]["reason"]


def test_transcript_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][1]["transcript"] = "different"
    manifest.write_text(json.dumps(payload))
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "transcripts" in result["excluded"][0]["reason"]


def test_overlapping_alignment_is_rejected(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][0]["tokens"][1]["start_s"] = 0.25
    manifest.write_text(json.dumps(payload))
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "ordered and non-overlapping" in result["excluded"][0]["reason"]


def test_missing_arm_is_rejected(tmp_path: Path) -> None:
    manifest = _write_pair_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"] = payload["records"][:1]
    manifest.write_text(json.dumps(payload))
    result = build_targets(manifest, tmp_path / "targets")
    assert not result["accepted"]
    assert "exactly natural and tts" in result["excluded"][0]["reason"]
