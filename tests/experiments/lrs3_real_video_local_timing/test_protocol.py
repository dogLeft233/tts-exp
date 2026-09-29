from pathlib import Path

import pytest

from scripts.experiments.lrs3_real_video_local_timing import config
from scripts.experiments.lrs3_real_video_local_timing.protocol import (
    audio_video_duration_within_bounded_tail,
    audio_video_duration_within_one_frame,
    audio_video_tail_contract,
    audit_cohort,
    video_timeline,
)


def test_duration_contract_allows_exactly_one_frame() -> None:
    assert audio_video_duration_within_one_frame(151 * 640, 150)


def test_duration_contract_rejects_more_than_one_frame() -> None:
    assert not audio_video_duration_within_one_frame(152 * 640, 150)


@pytest.mark.parametrize(
    ("delta", "accepted", "tail_class"),
    [
        (-641, False, "invalid_short_tail"),
        (-640, True, "within_original_bound"),
        (0, True, "within_original_bound"),
        (640, True, "within_original_bound"),
        (768, True, "extended_audio_tail"),
        (896, True, "extended_audio_tail"),
        (1280, True, "extended_audio_tail"),
        (1281, False, "invalid_long_tail"),
    ],
)
def test_bounded_tail_contract_uses_integer_boundaries(delta: int, accepted: bool, tail_class: str) -> None:
    samples = 150 * config.SAMPLES_PER_FRAME + delta
    assert audio_video_duration_within_bounded_tail(samples, 150) is accepted
    result = audio_video_tail_contract(samples, 150)
    assert result["delta_samples"] == delta
    assert result["accepted"] is accepted
    assert result["tail_class"] == tail_class


def test_input_audit_keeps_all_records_when_multiple_inputs_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.experiments.lrs3_real_video_local_timing import protocol
    from scripts.experiments.lrs3_real_video_local_timing.common import file_sha256

    sources = []
    for index in range(config.EXPECTED_RECORD_COUNT):
        video = tmp_path / f"video-{index}.mp4"
        audio = tmp_path / f"audio-{index}.wav"
        video.write_bytes(f"video-{index}".encode())
        audio.write_bytes(f"audio-{index}".encode())
        sources.append(
            {
                "sample_id": f"sample-{index}",
                "source_group": f"group-{index}",
                "transcript": f"text-{index}",
                "face_video": {"path": str(video), "sha256": file_sha256(video)},
                "natural_audio": {"path": str(audio), "sha256": file_sha256(audio)},
            }
        )

    def fake_timeline(path: Path) -> dict[str, object]:
        if path.name == "video-3.mp4":
            raise protocol.DiagnosticError("non-continuous PTS")
        return {"frame_count": 150, "first_pts_time": 0.0}

    def fake_audio(path: Path) -> dict[str, object]:
        count = 150 * config.SAMPLES_PER_FRAME + (1281 if path.name == "audio-4.wav" else 0)
        return {"sample_count": count, "sample_rate": 16000, "channels": 1, "sample_width": 2, "container_sha256": file_sha256(path), "pcm_sha256": "pcm"}

    def fake_pairing(source, video, audio, timeline, audio_meta):
        if source["sample_id"] == "sample-5":
            raise protocol.DiagnosticError("source evidence missing")
        return {"start_difference_ms": 0.0}

    monkeypatch.setattr(protocol, "video_timeline", fake_timeline)
    monkeypatch.setattr(protocol, "pcm_wav_metadata", fake_audio)
    monkeypatch.setattr(protocol, "source_pairing_evidence", fake_pairing)
    sources[6]["natural_audio"] = {"path": str(tmp_path / "missing.wav"), "sha256": "missing"}

    audit, valid_sources = audit_cohort({"records": sources})
    assert audit["status"] == "blocked"
    assert audit["record_count"] == config.EXPECTED_RECORD_COUNT
    assert len(audit["records"]) == config.EXPECTED_RECORD_COUNT
    assert audit["blocked_count"] == 4
    assert {row["sample_id"] for row in audit["records"] if not row["passed"]} == {"sample-3", "sample-4", "sample-5", "sample-6"}
    assert len(valid_sources) == config.EXPECTED_RECORD_COUNT - 4


def _probe_payload(*, start_time: str = "0.000000", discontinuous: bool = False) -> dict[str, object]:
    points = [index * 512 for index in range(10)]
    if discontinuous:
        points[5] += 256
    return {
        "streams": [
            {
                "codec_type": "video",
                "r_frame_rate": "25/1",
                "nb_read_frames": "10",
                "start_time": start_time,
                "duration": "0.400000",
                "width": 224,
                "height": 224,
                "time_base": "1/12800",
                "codec_name": "mpeg4",
            }
        ],
        "frames": [
            {"best_effort_timestamp": point, "pts": point, "pts_time": f"{index / 25:.6f}"}
            for index, point in enumerate(points)
        ],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_time": "0.001001"}, "does not start at zero"),
        ({"discontinuous": True}, "not a 25 fps timeline"),
    ],
)
def test_video_timeline_blocks_invalid_start_or_pts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], message: str) -> None:
    from scripts.experiments.lrs3_real_video_local_timing import protocol

    path = tmp_path / "video.mp4"
    path.write_bytes(b"probe")
    monkeypatch.setattr(protocol, "ffprobe_json", lambda *_args, **_kwargs: _probe_payload(**kwargs))
    with pytest.raises(protocol.DiagnosticError, match=message):
        video_timeline(path)
