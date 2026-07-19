"""Tests for English pair preparation and shared configuration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_prepare_english_pairs", _SCRIPTS / "26_prepare_english_pairs.py"
)
_prepare = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prepare)

from utils import deep_merge, detect_sample_ids, load_config


def test_deep_merge_preserves_unrelated_defaults():
    base = {"paths": {"audio_dir": "old", "image_dir": "images"}, "seed": 42}
    override = {"paths": {"audio_dir": "english"}}
    merged = deep_merge(base, override)
    assert merged == {
        "paths": {"audio_dir": "english", "image_dir": "images"},
        "seed": 42,
    }
    assert base["paths"]["audio_dir"] == "old"


def test_detect_sample_ids_prefers_explicit_config(tmp_path):
    cfg = {"samples": {"ids": [13, 1, 9], "smoke_id": 9}}
    assert detect_sample_ids(tmp_path, cfg=cfg) == [1, 9, 13]
    assert detect_sample_ids(tmp_path, smoke=True, cfg=cfg) == [9]


def test_prepare_pairs_canonicalizes_and_matches_text(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "config.yaml").write_text("{}\n")
    natural_dir = tmp_path / "natural"
    tts_dir = tmp_path / "tts"
    image_dir = tmp_path / "images"
    for directory in (natural_dir, tts_dir, image_dir):
        directory.mkdir()

    duration = 0.5
    natural = 0.1 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * duration)) / 16000)
    tts = 0.1 * np.sin(2 * np.pi * 220 * np.arange(int(24000 * duration)) / 24000)
    sf.write(natural_dir / "1.wav", natural, 16000, subtype="PCM_16")
    sf.write(tts_dir / "1.wav", tts, 24000, subtype="PCM_16")
    (image_dir / "1.png").write_bytes(b"test-image")

    natural_manifest = [{
        "sample_id": 1,
        "librispeech_id": "1-2-3",
        "speaker_id": 1,
        "chapter_id": 2,
        "text": "SAME TEXT",
        "local_path": "natural/1.wav",
    }]
    tts_manifest = {
        "results": [{
            "sample_id": 1,
            "text": "SAME TEXT",
            "generated_audio": "tts/1.wav",
            "backend": "dashscope_vc",
            "model": "qwen3-tts-vc-2026-01-22",
        }]
    }
    (natural_dir / "manifest.json").write_text(json.dumps(natural_manifest))
    (tts_dir / "manifest.json").write_text(json.dumps(tts_manifest))

    override = {
        "paths": {
            "audio_dir": "natural",
            "tts_audio_dir": "tts",
            "image_dir": "images",
            "natural_manifest": "natural/manifest.json",
            "tts_manifest": "tts/manifest.json",
        },
        "samples": {"ids": [1], "smoke_id": 1},
        "experiment": {
            "canonical_sample_rate": 16000,
            "canonical_subtype": "PCM_16",
        },
    }
    config_path = scripts / "english.yaml"
    config_path.write_text(yaml.safe_dump(override))
    cfg = load_config(tmp_path, config_path)

    result = _prepare.prepare_pairs(tmp_path, "test_run", cfg)
    assert result["complete"] is True
    assert result["samples_ok"] == 1
    record = result["records"][0]
    assert record["text"] == "SAME TEXT"
    assert record["natural"]["canonical_qc"]["sample_rate_hz"] == 16000
    assert record["tts"]["canonical_qc"]["sample_rate_hz"] == 16000
    assert record["tts"]["canonical_qc"]["channels"] == 1
    assert "voice_id" not in json.dumps(result) or record["tts"]["voice_id"] is None
