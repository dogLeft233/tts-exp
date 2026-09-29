from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.experiments.phoneme_tfg_association import generation
from scripts.experiments.phoneme_tfg_association.protocol import ProtocolError


def test_first_frame_loader_rejects_lrs3_visual_path(tmp_path):
    path = tmp_path / "lrs3" / "frame.png"
    path.parent.mkdir()
    assert cv2.imwrite(str(path), np.zeros((224, 224, 3), dtype=np.uint8))
    with pytest.raises(ProtocolError, match="LRS3 visual input is forbidden"):
        generation._load_first_frame(path)


def test_renderer_repeats_one_external_frame_and_never_loads_source_video(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference.png"
    reference = np.full((8, 8, 3), 17, dtype=np.uint8)
    assert cv2.imwrite(str(reference_path), reference)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"canonical-audio")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    observed: dict[str, object] = {}

    class FakeWorker:
        def load_frames(self, _path):  # pragma: no cover - proves this path is forbidden by the test
            raise AssertionError("renderer must not load a source-video timeline")

        def mel_chunks(self, _path):
            return [np.zeros((80, 16), dtype=np.float32) for _ in range(3)], {"mel_chunk_count": 3}

        def render_arm(self, _model, frames, boxes, chunks, _batch_size, _device):
            observed["frames"] = frames
            observed["boxes"] = boxes
            observed["chunks"] = chunks
            return [frame.copy() for frame in frames]

        def encode_video(self, frames, output, _ffmpeg):
            observed["encoded_frames"] = frames
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"video-only")

    monkeypatch.setattr(generation, "_wav2lip_worker", lambda: FakeWorker())

    def fake_mux(_video_only: Path, _audio: Path, output: Path, _ffmpeg: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"muxed")
        return {"command": ["fake-mux"]}

    monkeypatch.setattr(generation, "_mux_pcm", fake_mux)
    cell = {
        "cell_key": "g::s::natural",
        "protocol_id": "phoneme_tfg_association_v1",
        "protocol_hash": "hash",
        "block_id": "block",
        "sample_id": "s",
        "source_group": "g",
        "tfg": "wav2lip",
        "arm": "natural",
        "seed": 1,
        "feature": {"arms": {"natural": {"audio": str(audio_path), "pcm_sha256": "pcm"}}},
        "reference": {
            "visual_source": str(tmp_path / "external.mp4"),
            "visual_source_sha256": generation.file_sha256(reference_path),
            "reference": str(reference_path),
            "reference_sha256": generation.file_sha256(reference_path),
            "crop_sha256": "crop",
            "box_top_bottom_left_right": [0, 8, 0, 8],
        },
    }
    # The visual source is only provenance here; the frozen reference frame is
    # the actual renderer input and must be external to any LRS3 path.
    external_source = tmp_path / "external.mp4"
    external_source.write_bytes(b"external-visual")
    cell["reference"]["visual_source_sha256"] = generation.file_sha256(external_source)

    receipt = generation.render_wav2lip(cell, tmp_path, {"wav2lip_checkpoint": str(checkpoint), "ffmpeg": "ffmpeg", "wav2lip_batch_size": 4}, model=object(), device="cpu")
    frames = observed["frames"]
    assert isinstance(frames, list) and len(frames) == 3
    assert all(np.array_equal(frame, reference) for frame in frames)
    assert receipt["fixed_reference_frame_repeated"] is True
    assert receipt["visual_source_used_for_generation"] == "one_frame_repeated_for_each_mel_chunk"
