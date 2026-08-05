"""Tests for deterministic HDTF paired-source preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_download_spec = importlib.util.spec_from_file_location(
    "_download_hdtf_paired_videos",
    _SCRIPTS / "download_hdtf_paired_videos.py",
)
assert _download_spec is not None
hdtf = importlib.util.module_from_spec(_download_spec)
assert _download_spec.loader is not None
_download_spec.loader.exec_module(hdtf)

_prepare_spec = importlib.util.spec_from_file_location(
    "_prepare_hdtf_pairs",
    _SCRIPTS / "prepare_hdtf_pairs.py",
)
assert _prepare_spec is not None
prepare = importlib.util.module_from_spec(_prepare_spec)
assert _prepare_spec.loader is not None
_prepare_spec.loader.exec_module(prepare)


def test_select_pairs_filters_media_and_deduplicates_speakers():
    audio = [
        {"local_path": "data/WRA_Alice0.m4a", "duration_s": 20.0},
        {"local_path": "data/WRA_Alice1.m4a", "duration_s": 18.0},
        {"local_path": "data/WDA_Bob.m4a", "duration_s": 25.0},
        {"local_path": "data/WRA_Short.m4a", "duration_s": 5.0},
        {"local_path": "data/RD_Radio1.m4a", "duration_s": 10.0},
    ]
    video_stems = {
        "WRA_Alice0",
        "WRA_Alice1",
        "WDA_Bob",
        "WRA_Short",
        "RD_Radio1",
    }
    selected = hdtf.select_pairs(audio, video_stems, count=2)
    assert [item["stem"] for item in selected] == ["WRA_Alice1", "WDA_Bob"]


def test_prepare_config_uses_hdtf_paths_and_english_protocol():
    config = prepare.build_config_override("hdtf_run", sample_count=13)
    assert config["paths"]["audio_dir"] == "runs/hdtf_run/00_pairs/natural"
    assert config["paths"]["image_dir"] == "runs/hdtf_run/00_pairs/images"
    assert config["tts"]["language"] == "English"
    assert config["ditto"]["conditions"] == ["natural_raw", "tts_raw"]
    assert config["eval"]["require_complete_pairs"] is True


def test_prepare_pairs_replaces_failed_candidate(monkeypatch, tmp_path):
    audio = tmp_path / "good.wav"
    video = tmp_path / "good.mp4"
    audio2 = tmp_path / "good2.wav"
    video2 = tmp_path / "good2.mp4"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    audio2.write_bytes(b"audio2")
    video2.write_bytes(b"video2")
    manifest = tmp_path / "video_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "stem": "WRA_Bad",
                        "speaker_key": "WRA_Bad",
                        "audio_path": str(tmp_path / "missing.wav"),
                        "video_local_path": str(tmp_path / "missing.mp4"),
                        "duration_s": 8.0,
                        "remote_path": "/bad.mp4",
                        "video_sha256": "bad",
                    },
                    {
                        "stem": "WRA_Good",
                        "speaker_key": "WRA_Good",
                        "audio_path": str(audio),
                        "video_local_path": str(video),
                        "duration_s": 8.0,
                        "remote_path": "/good.mp4",
                        "video_sha256": "good",
                    },
                    {
                        "stem": "WRA_Bad2",
                        "speaker_key": "WRA_Bad2",
                        "audio_path": str(tmp_path / "missing2.wav"),
                        "video_local_path": str(tmp_path / "missing2.mp4"),
                        "duration_s": 8.0,
                        "remote_path": "/bad2.mp4",
                        "video_sha256": "bad2",
                    },
                    {
                        "stem": "WRA_Good2",
                        "speaker_key": "WRA_Good2",
                        "audio_path": str(audio2),
                        "video_local_path": str(video2),
                        "duration_s": 8.0,
                        "remote_path": "/good2.mp4",
                        "video_sha256": "good2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(prepare, "probe_duration", lambda *_args: 8.0)
    def write_audio(_source, destination, *_args, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"crop")

    def write_frame(_source, destination, *_args, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"frame")

    monkeypatch.setattr(prepare, "crop_audio", write_audio)
    monkeypatch.setattr(prepare, "extract_frame", write_frame)
    monkeypatch.setattr(
        prepare,
        "validate_audio",
        lambda _path: {"sample_rate_hz": 24000},
    )
    monkeypatch.setattr(
        prepare,
        "validate_image",
        lambda _path: {"width": 512, "height": 288},
    )

    metadata = prepare.prepare_pairs(
        repo=tmp_path,
        run_id="replacement_test",
        video_manifest_path=manifest,
        count=2,
    )

    assert metadata["records"][0]["stem"] == "WRA_Good"
    assert metadata["records"][0]["replacement_for"]["stem"] == "WRA_Bad"
    assert metadata["records"][0]["rejected_candidates"] == [
        metadata["rejected_candidates"][0]
    ]
    assert metadata["records"][1]["stem"] == "WRA_Good2"
    assert metadata["records"][1]["replacement_for"]["stem"] == "WRA_Bad2"
    assert metadata["records"][1]["rejected_candidates"] == [
        metadata["rejected_candidates"][1]
    ]
