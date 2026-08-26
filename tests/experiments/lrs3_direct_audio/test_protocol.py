from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.lrs3_direct_audio.run_wavlm_hifigan_lrs3 import (
    cluster_bootstrap,
    exact_length,
    load_records,
    parse_syncnet,
    summarize_scores,
)


def test_exact_length_crop_and_pad() -> None:
    cropped, crop_meta = exact_length(np.arange(5, dtype=np.float32), 3)
    padded, pad_meta = exact_length(np.arange(2, dtype=np.float32), 4)
    assert cropped.tolist() == [0.0, 1.0, 2.0]
    assert crop_meta["action"] == "right_crop"
    assert padded.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert pad_meta["action"] == "right_zero_pad"


def test_load_records_resolves_relative_assets(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "lrs3",
                "records": [
                    {
                        "sample_id": "clip-1",
                        "source_group": "group-1",
                        "clip_id": "00001",
                        "video_local_path": video.name,
                        "natural_audio_path": audio.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records = load_records(manifest, tmp_path, 0, 1)
    assert records[0]["video"] == str(video.resolve())
    assert records[0]["natural_audio"] == str(audio.resolve())
    assert records[0]["video_sha256"]
    assert records[0]["natural_audio_sha256"]


def test_parse_syncnet(tmp_path: Path) -> None:
    log = tmp_path / "score.log"
    log.write_text("Confidence: 5.125\nMin dist: 7.250\nAV offset: -2\n", encoding="utf-8")
    assert parse_syncnet(log) == {"sync_c": 5.125, "sync_d": 7.25, "av_offset": -2}


def test_summary_uses_replacement_contrast() -> None:
    records = [
        {"sample_id": "a", "source_group": "g1"},
        {"sample_id": "b", "source_group": "g2"},
    ]
    scores = [
        {"sample_id": "a", "cell": "natural_video_natural_audio", "sync_c": 5.0, "sync_d": 8.0},
        {"sample_id": "a", "cell": "direct_video_direct_audio", "sync_c": 5.5, "sync_d": 7.5},
        {"sample_id": "a", "cell": "direct_video_natural_audio", "sync_c": 5.2, "sync_d": 7.8},
        {"sample_id": "b", "cell": "natural_video_natural_audio", "sync_c": 6.0, "sync_d": 7.0},
        {"sample_id": "b", "cell": "direct_video_direct_audio", "sync_c": 5.5, "sync_d": 7.5},
        {"sample_id": "b", "cell": "direct_video_natural_audio", "sync_c": 6.1, "sync_d": 6.8},
    ]
    summary = summarize_scores(records, scores)
    replacement_c = summary["replacement_direct_video_minus_natural_video_sync_c"]
    replacement_d = summary["replacement_direct_video_minus_natural_video_sync_d"]
    assert replacement_c["mean"] == pytest.approx(0.15)
    assert replacement_d["mean"] == pytest.approx(0.2)
    assert summary["wins"]["replacement_direct_video_minus_natural_video"] == {"c": 2, "d": 2, "joint": 2, "n": 2}


def test_cluster_bootstrap_is_reproducible() -> None:
    first = cluster_bootstrap([1.0, 2.0, 3.0], ["a", "b", "b"], 7, draws=100)
    second = cluster_bootstrap([1.0, 2.0, 3.0], ["a", "b", "b"], 7, draws=100)
    assert first == second
