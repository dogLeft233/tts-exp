"""Tests for deterministic MDC multilingual pair preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "_prepare_mdc_pairs", _SCRIPTS / "prepare_mdc_pairs.py"
)
assert _spec is not None
prepare = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(prepare)

_report_spec = importlib.util.spec_from_file_location(
    "_report_stage", _SCRIPTS / "05_report.py"
)
assert _report_spec is not None
report = importlib.util.module_from_spec(_report_spec)
assert _report_spec.loader is not None
_report_spec.loader.exec_module(report)


def test_select_samples_uses_longest_records_and_stable_manifest_order():
    records = [
        {"sample_id": "en_001", "language_code": "en", "duration_s": 4.0},
        {"sample_id": "en_002", "language_code": "en", "duration_s": 8.0},
        {"sample_id": "en_003", "language_code": "en", "duration_s": 8.0},
        {"sample_id": "de_001", "language_code": "de", "duration_s": 99.0},
    ]
    selected = prepare.select_samples(records, "en", count=3)
    assert [row["sample_id"] for row in selected] == [
        "en_002",
        "en_003",
        "en_001",
    ]


def test_build_config_uses_numeric_ids_and_language_settings():
    override = prepare.build_config_override("de", "mdc_de_run", sample_count=13)
    assert override["samples"]["ids"] == list(range(1, 14))
    assert override["tts"]["language"] == "German"
    assert override["ditto"]["conditions"] == ["natural_raw", "tts_raw"]
    assert override["eval"]["require_complete_pairs"] is True
    assert override["paths"]["audio_dir"] == "runs/mdc_de_run/00_pairs/natural"


def test_legacy_languages_have_qwen_language_settings():
    expected = {
        "pt": "Portuguese",
        "ja": "Japanese",
        "fr": "French",
        "ru": "Russian",
    }
    for language_code, qwen_language in expected.items():
        override = prepare.build_config_override(language_code, "legacy_run", sample_count=13)
        assert override["tts"]["language"] == qwen_language


def test_select_samples_can_exclude_used_source_records():
    records = [
        {"sample_id": "en_001", "language_code": "en", "duration_s": 9.0},
        {"sample_id": "en_002", "language_code": "en", "duration_s": 8.0},
        {"sample_id": "en_003", "language_code": "en", "duration_s": 7.0},
        {"sample_id": "en_004", "language_code": "en", "duration_s": 6.0},
    ]
    selected = prepare.select_samples(
        records,
        "en",
        count=2,
        exclude_sample_ids={"en_001", "en_002"},
    )
    assert [row["sample_id"] for row in selected] == ["en_003", "en_004"]


def test_select_samples_filters_sparse_text_records():
    records = [
        {"sample_id": "pt_short", "language_code": "pt", "duration_s": 10.0, "text": "oi"},
        {
            "sample_id": "pt_good",
            "language_code": "pt",
            "duration_s": 10.0,
            "text": "esta frase tem conteúdo suficiente",
        },
        {
            "sample_id": "pt_long",
            "language_code": "pt",
            "duration_s": 12.0,
            "text": "esta é uma frase mais longa com conteúdo suficiente",
        },
    ]
    selected = prepare.select_samples(
        records,
        "pt",
        count=2,
        min_text_chars_per_s=1.5,
    )
    assert [row["sample_id"] for row in selected] == ["pt_long", "pt_good"]


def test_build_config_preserves_explicit_target_sample_ids():
    override = prepare.build_config_override(
        "en",
        "replacement_run",
        sample_count=3,
        sample_ids=[5, 8, 9],
        max_new_tokens=256,
    )
    assert override["samples"]["ids"] == [5, 8, 9]
    assert override["samples"]["count"] == 3
    assert override["tts"]["faster_qwen3"]["max_new_tokens"] == 256


def test_prepare_language_writes_explicit_target_ids_to_config(tmp_path):
    import numpy as np
    import soundfile as sf

    manifest_path = tmp_path / "manifest.json"
    source = tmp_path / "source.wav"
    sf.write(source, np.ones(24000, dtype=np.float32) * 0.1, 24000)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "5.png").write_bytes(b"placeholder")
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "en_backup",
                        "language_code": "en",
                        "duration_s": 1.0,
                        "file_path": "source.wav",
                        "text": "backup text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prepare.prepare_language(
        repo=tmp_path,
        run_id="replacement",
        language_code="en",
        manifest_path=manifest_path,
        image_dir=Path("images"),
        sample_count=1,
        sample_ids=[5],
        ffmpeg_bin="ffmpeg",
    )
    config = (tmp_path / "runs/replacement/config_override.yaml").read_text()
    assert "ids:\n  - 5" in config


def test_report_includes_run_metadata(tmp_path):
    rows = [
        {
            "sample_id": 1,
            "natural_raw_c": 1.0,
            "natural_raw_d": 2.0,
            "tts_raw_c": 1.5,
            "tts_raw_d": 1.5,
            "natural_resamp_c": None,
            "natural_resamp_d": None,
            "tts_resamp_c": None,
            "tts_resamp_d": None,
        }
    ]
    report.generate_report(
        rows,
        ["natural_raw", "tts_raw"],
        tmp_path,
        run_metadata={
            "design": "self_clone",
            "language": "German",
            "sample_description": "MDC German, 13 paired samples",
        },
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "**Design**: self_clone" in text
    assert "**Language**: German" in text
    assert "**Samples**: MDC German, 13 paired samples" in text


def test_report_preserves_legacy_sample_description(tmp_path):
    rows = [
        {
            "sample_id": 1,
            "natural_raw_c": 1.0,
            "natural_raw_d": 2.0,
            "tts_raw_c": 1.5,
            "tts_raw_d": 1.5,
            "natural_resamp_c": None,
            "natural_resamp_d": None,
            "tts_resamp_c": None,
            "tts_resamp_d": None,
        }
    ]
    report.generate_report(rows, ["natural_raw", "tts_raw"], tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "**Samples**: 10 paired audio-image pairs from AISHELL-1 + HDTF" in text
    assert "**Design**:" not in text
