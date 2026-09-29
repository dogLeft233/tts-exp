from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.experiments.lrs3_bridge_tts_quality import config
from scripts.experiments.lrs3_bridge_tts_quality import render as render_module
from scripts.experiments.lrs3_bridge_tts_quality.audio import read_pcm16, write_pcm16
from scripts.experiments.lrs3_bridge_tts_quality.common import file_sha256
from scripts.experiments.lrs3_bridge_tts_quality.render import (
    _prepare_driver_audio,
    _render_one,
)


def test_driver_audio_adds_fixed_right_context_without_changing_source(tmp_path) -> None:
    source = tmp_path / "bridge.wav"
    values = np.arange(config.MIN_AUDIO_SAMPLES, dtype=np.int16)
    write_pcm16(source, values)

    row = _prepare_driver_audio(source, tmp_path / "05_videos", "B_LOCAL", "sample")

    driver, driver_meta = read_pcm16(row["driver_audio"])
    np.testing.assert_array_equal(driver[: values.size], values)
    np.testing.assert_array_equal(driver[values.size :], np.zeros(config.WAV2LIP_TAIL_PAD_SAMPLES, dtype=np.int16))
    assert row["source_audio"] == str(source.resolve())
    assert row["source_audio_sha256"] == file_sha256(source)
    assert row["driver_tail_pad_samples"] == config.WAV2LIP_TAIL_PAD_SAMPLES
    assert driver_meta["sample_count"] == values.size + config.WAV2LIP_TAIL_PAD_SAMPLES


def test_render_creates_wav2lip_temp_directory(tmp_path, monkeypatch) -> None:
    face = tmp_path / "face.mp4"
    audio = tmp_path / "audio.wav"
    face.write_bytes(b"face")
    audio.write_bytes(b"audio")
    output_stage = tmp_path / "05_videos"

    def fake_run(command, *, cwd, env, stdout, stderr, check):
        work_dir = Path(cwd)
        assert (work_dir / "temp").is_dir()
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_bytes(b"rendered")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(render_module.subprocess, "run", fake_run)
    row = _render_one(
        {"sample_id": "sample", "source_group": "group", "face_video": {"path": str(face)}},
        {"arms": [{"arm": arm, "output": str(audio)} for arm in ("N", "B0", "B_LOCAL", "B_CLOUD")]},
        "N",
        0,
        20260913,
        {"box": [0, 224, 0, 224], "mode": "full_frame"},
        output_stage,
        {"checkpoint_sha256": "checkpoint"},
    )

    assert row["output"] == str((output_stage / "videos/sample/repeat_0/N/video.mp4").resolve())
