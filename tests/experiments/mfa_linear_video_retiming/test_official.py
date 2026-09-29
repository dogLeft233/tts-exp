from __future__ import annotations

import pickle
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError
from scripts.experiments.mfa_linear_video_retiming.official import (
    _matrix_from_activesd,
    _pipeline_crop_files,
    _verify_mux_video,
    parse_official_log,
    recompute_official_curve,
)
from scripts.experiments.mfa_linear_video_retiming.check import (
    _load_official_matrix_independently,
    _official_log_metrics,
)
from scripts.experiments.phone_gain_static_tfg_mfa.official_score import strict_mux


def _binaries() -> tuple[Path, Path] | None:
    ffmpeg = Path("/home/wjj/miniconda3/bin/ffmpeg")
    ffprobe = Path("/home/wjj/miniconda3/bin/ffprobe")
    if not ffmpeg.is_file():
        found = shutil.which("ffmpeg")
        ffmpeg = Path(found) if found else ffmpeg
    if not ffprobe.is_file():
        found = shutil.which("ffprobe")
        ffprobe = Path(found) if found else ffprobe
    return (ffmpeg, ffprobe) if ffmpeg.is_file() and ffprobe.is_file() else None


def _write_video(path: Path, frames: np.ndarray, ffmpeg: Path) -> None:
    from scripts.experiments.static_image_bridge.render_worker import encode_ffv1_stream

    encode_ffv1_stream(frames, width=frames.shape[2], height=frames.shape[1], output=path, ffmpeg=ffmpeg)


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def test_official_log_parser_requires_one_score_and_rejects_boundary_peak(tmp_path: Path) -> None:
    log = tmp_path / "syncnet.log"
    log.write_text("Confidence: 1.25\nMin dist: 4.5\nAV offset: -2\n", encoding="utf-8")
    assert parse_official_log(log) == {"sync_c": 1.25, "sync_d": 4.5, "offset": -2}

    log.write_text("Confidence: 1.25\nConfidence: 1.3\nMin dist: 4.5\nAV offset: 0\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="expected one"):
        parse_official_log(log)
    log.write_text("Confidence: 1.25\nMin dist: 4.5\nAV offset: 15\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="BOUNDARY_PEAK"):
        parse_official_log(log)


def test_official_distance_curve_uses_fixed_31_lag_axis() -> None:
    matrix = np.full((31, 5), 3.0, dtype=np.float32)
    matrix[14] = 0.5

    result = recompute_official_curve(matrix)

    assert result["matrix_shape"] == [31, 5]
    assert result["sync_d"] == pytest.approx(0.5)
    assert result["sync_c"] == pytest.approx(2.5)
    assert result["d0"] == pytest.approx(3.0)
    assert result["offset"] == 1
    assert result["best_lag"] == -1


def test_pipeline_crop_discovery_uses_pinned_syncnet_pycrop_layout(tmp_path: Path) -> None:
    reference = "sample_ref"
    crop = tmp_path / "pycrop" / reference / "00000.avi"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"fixture")

    assert _pipeline_crop_files(tmp_path, reference) == [crop]
    assert _pipeline_crop_files(tmp_path, "missing") == []


def test_producer_and_checker_both_read_the_raw_activesd_pickle(tmp_path: Path) -> None:
    path = tmp_path / "activesd.pckl"
    matrix = np.arange(31 * 7, dtype=np.float32).reshape(31, 7) / 17.0
    with path.open("wb") as handle:
        pickle.dump({"distances": matrix}, handle)

    assert np.array_equal(_matrix_from_activesd(path), matrix)
    assert np.array_equal(_load_official_matrix_independently(path), matrix)


def test_independent_log_recompute_finds_the_syncnet_log(tmp_path: Path) -> None:
    log = tmp_path / "syncnet.log"
    log.write_text("Confidence: 1.25\nMin dist: 4.5\nAV offset: -2\n", encoding="utf-8")
    result = {"commands": [{"argv": ["python", "/sync/run_syncnet.py"], "log": str(log)}]}

    assert _official_log_metrics(result) == {"sync_c": 1.25, "sync_d": 4.5, "offset": -2}


@pytest.mark.skipif(_binaries() is None, reason="ffmpeg/ffprobe are required for media contract test")
def test_strict_mux_preserves_exact_video_pixels_pts_and_pcm(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _binaries()  # type: ignore[misc]
    video = tmp_path / "video.mkv"
    audio = tmp_path / "audio.wav"
    muxed = tmp_path / "muxed.mkv"
    frames = np.zeros((8, 16, 18, 3), dtype=np.uint8)
    for index in range(frames.shape[0]):
        frames[index] = index * 17
    _write_video(video, frames, ffmpeg)
    _write_wav(audio, (np.arange(6400, dtype=np.int32) % 1000 - 500).astype(np.int16))

    receipt = strict_mux(video=video, audio=audio, output=muxed, ffmpeg=ffmpeg, ffprobe=ffprobe,
                         log_path=tmp_path / "mux.log")
    audit = _verify_mux_video(video, muxed, ffprobe)

    assert receipt["audio_pcm_exact"] is True
    assert receipt["video_stream_copy_verified_by_metadata"] is True
    assert audit["pixels_identical"] is True
    assert audit["pts_identical"] is True
