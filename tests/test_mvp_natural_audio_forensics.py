"""Tests for scripts/26_mvp_natural_audio_forensics.py."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "26_mvp_natural_audio_forensics.py"
spec = importlib.util.spec_from_file_location("mvp_forensics", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_parse_ids_and_collect_exact_set(tmp_path):
    ids = module.parse_ids("51-53,55")
    assert ids == (51, 52, 53, 55)
    audio = tmp_path / "audio"
    audio.mkdir()
    for sid in ids:
        sf.write(audio / f"{sid}.wav", np.zeros(1600, dtype=np.float32), 16000)
    assert set(module.collect_audio_files(audio, ids)) == set(ids)


def test_collect_audio_files_rejects_missing_and_duplicate(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    sf.write(audio / "51.wav", np.zeros(100, dtype=np.float32), 16000)
    sf.write(audio / "0051.wav", np.zeros(100, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="duplicates"):
        module.collect_audio_files(audio, (51,))
    with pytest.raises(ValueError, match="missing"):
        module.collect_audio_files(audio, (51, 52))


def _write_scores(path: Path, ids=(51, 52), mismatch=False):
    rows = []
    for sid in ids:
        for condition in ("natural", "mvp"):
            rows.append({
                "sample_id": sid, "condition": condition,
                "sync_c": 5.04526 if condition == "natural" else 4.16242,
                "sync_d": 7.90412 if condition == "natural" else 7.93172,
                "av_offset": 0.7 if condition == "natural" else 0.08,
                "run_id": "test", "video_path": f"{condition}/{sid}.mp4",
            })
    if mismatch:
        rows[0]["sync_c"] = 10
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_score_ledger_cohort_and_delta_direction(tmp_path):
    path = tmp_path / "scores.json"
    _write_scores(path, mismatch=True)
    rows = module.load_score_ledger(path, (51, 52))
    pairs = module.score_pairs(rows, (51, 52))
    assert pairs[0]["delta_sync_c"] < 0
    assert pairs[0]["delta_sync_d"] > 0
    with pytest.raises(ValueError, match="does not reconcile"):
        module.reconcile_scores(pairs, expected={"natural_sync_c": 5.0})


def test_score_reconciliation_expected_full_summary():
    rows = []
    for sid in range(51, 101):
        rows.extend([
            {"sample_id": sid, "condition": "natural", "sync_c": 5.04526, "sync_d": 7.90412, "av_offset": 0.7, "run_id": "n"},
            {"sample_id": sid, "condition": "mvp", "sync_c": 4.16242, "sync_d": 7.93172, "av_offset": 0.08, "run_id": "m"},
        ])
    pairs = module.score_pairs(rows, range(51, 101))
    summary = module.reconcile_scores(pairs)
    assert summary["delta_sync_c"] == pytest.approx(-0.88284)
    assert summary["delta_av_offset"] == pytest.approx(-0.62)


def test_wav_metadata_hash_and_clipping(tmp_path):
    path = tmp_path / "x.wav"
    y = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
    sf.write(path, y, 16000, subtype="PCM_16")
    meta = module.wav_metadata(path)
    assert meta["sample_rate_native"] == 16000
    assert meta["channels_native"] == 1
    assert len(meta["sha256"]) == 64
    assert meta["clipped_samples"] == 2


def test_textgrid_pause_merge_and_metrics(tmp_path):
    path = tmp_path / "51.TextGrid"
    path.write_text('''File type = "ooTextFile"\nitem [1]:\n    intervals [1]:\n        xmin = 0\n        xmax = 0.5\n        text = "a"\n    intervals [2]:\n        xmin = 0.5\n        xmax = 0.6\n        text = "sil"\n    intervals [3]:\n        xmin = 0.6\n        xmax = 0.7\n        text = "sp"\n    intervals [4]:\n        xmin = 0.7\n        xmax = 1.0\n        text = "b"\n''', encoding="utf-8")
    result = module.textgrid_features(path)
    assert result["alignment_phone_count"] == 2
    assert result["internal_pause_count"] == 1
    assert result["internal_pause_total"] == pytest.approx(0.2)


def test_deterministic_permutation_bootstrap_and_constant_correlation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert module.paired_permutation(values, n=100, seed=42) == module.paired_permutation(values, n=100, seed=42)
    assert module.bootstrap_mean(values, n=100, seed=42) == module.bootstrap_mean(values, n=100, seed=42)
    pairs = [{"sample_id": 51, "delta_sync_c": 1, "delta_sync_d": 1}, {"sample_id": 52, "delta_sync_c": 2, "delta_sync_d": 2}, {"sample_id": 53, "delta_sync_c": 3, "delta_sync_d": 3}]
    rows = [{"sample_id": sid, "delta_constant": 1.0} for sid in (51, 52, 53)]
    result = module.feature_correlations(pairs, rows)
    assert all(row["p_value"] is None for row in result)


def test_feature_extraction_smoke(tmp_path):
    path = tmp_path / "51.wav"
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(path, 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    rows = module.extract_audio_features({51: path}, "natural")
    assert len(rows) == 1
    assert rows[0]["sample_id"] == 51
    assert rows[0]["sample_rate"] == 16000
    assert "lufs" in rows[0]


def test_mvp_sidecar_rejects_supplied_natural_provenance(tmp_path):
    natural = tmp_path / "natural"
    mvp = tmp_path / "mvp"
    natural.mkdir(); mvp.mkdir()
    sf.write(natural / "51.wav", np.zeros(1000, dtype=np.float32), 16000)
    sf.write(mvp / "51.wav", np.zeros(1000, dtype=np.float32), 16000)
    (mvp / "51.json").write_text(json.dumps({
        "sample_id": 51, "sample_rate": 16000, "sample_count": 1000,
        "finite": True, "clipped_sample_count": 0,
        "natural_sample_count": 2000, "exact_natural_length": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="exact-length metadata mismatch"):
        module.load_mvp_sidecars({51: mvp / "51.wav"}, {51: natural / "51.wav"})


    summary = {"n_pairs": 0, "scores": {}}
    report = module.build_report(summary, [], alignment_available=False)
    assert "score ledger was unavailable" in report


def test_strict_named_tier_parser_excludes_word_intervals(tmp_path):
    path = tmp_path / "51.TextGrid"
    path.write_text('''File type = "ooTextFile"\nitem [1]:\n    class = "IntervalTier"\n    name = "words"\n    intervals: size = 1\n    intervals [1]:\n        xmin = 0\n        xmax = 1\n        text = "甲"\nitem [2]:\n    class = "IntervalTier"\n    name = "phones"\n    intervals: size = 2\n    intervals [1]:\n        xmin = 0\n        xmax = 0.4\n        text = "a˥"\n    intervals [2]:\n        xmin = 0.4\n        xmax = 1\n        text = ""\n''', encoding="utf-8")
    tiers = module.parse_textgrid_tiers(path)
    assert len(tiers["words"]) == 1
    assert len(tiers["phones"]) == 2
    assert module._strip_phone_tone("a˥") == "a"


def test_stage_mfa_corpus_isolated_and_probe_reports_blocked(tmp_path):
    audio = tmp_path / "audio"; transcripts = tmp_path / "transcripts"
    audio.mkdir(); transcripts.mkdir()
    sf.write(audio / "51.wav", np.zeros(1600, dtype=np.float32), 16000)
    (transcripts / "51.txt").write_text("测试", encoding="utf-8")
    staged = module.stage_mfa_corpus({51: audio / "51.wav"}, {51: transcripts / "51.txt"}, tmp_path / "stage", "mvp")
    assert len(staged["records"]) == 1
    assert (tmp_path / "stage" / "mvp" / "audio" / "0051.wav").exists()
    probe = module.probe_mfa("/definitely/missing/mfa", "mandarin_mfa", "mandarin_mfa")
    assert probe["status"] == "unavailable"


def test_phone_pairing_refuses_sequence_mismatch():
    natural = [{"base_symbol": "a", "symbol": "a", "duration_s": 0.2, "start_s": 0.0, "end_s": 0.2}]
    mvp = [{"base_symbol": "i", "symbol": "i", "duration_s": 0.2, "start_s": 0.0, "end_s": 0.2}]
    rows, audit = module.pair_phone_rows(natural, mvp, 51)
    assert rows == []
    assert audit["phone_pairing_status"] == "mismatch"


def test_sync_c_decline_is_added_to_score_pairs():
    rows = [
        {"sample_id": 51, "condition": "natural", "sync_c": 5.0, "sync_d": 8.0, "av_offset": 0.7},
        {"sample_id": 51, "condition": "mvp", "sync_c": 4.0, "sync_d": 8.1, "av_offset": 0.1},
    ]
    pair = module.score_pairs(rows, (51,))[0]
    assert pair["delta_sync_c"] == -1.0
    assert pair["sync_c_decline"] == 1.0


def test_local_prosody_smoke_has_null_safe_fields(tmp_path):
    path = tmp_path / "51.wav"
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(path, 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    phones = [{"sample_id": 51, "condition": "mvp", "phone_index": 0, "symbol": "a", "base_symbol": "a", "start_s": 0.1, "end_s": 0.5, "duration_s": 0.4, "center_s": 0.3, "start_norm": 0.1, "end_norm": 0.5, "duration_norm": 0.4, "audio_sha256": "x", "textgrid_sha256": "y", "alignment_status": "strict"}]
    rows = module.extract_local_prosody(path, phones)
    assert len(rows) == 1
    assert "f0_mean" in rows[0]
    assert rows[0]["frame_count"] > 0
