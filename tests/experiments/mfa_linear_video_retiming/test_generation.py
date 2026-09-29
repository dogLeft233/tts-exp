from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError, file_sha256
from scripts.experiments.mfa_linear_video_retiming.generation import (
    build_render_argv,
    canonicalize_tail,
    image_rgb_sha256,
    _verify_worker_receipt,
)


def _write_video(path: Path, frames: np.ndarray, ffmpeg: Path) -> None:
    from scripts.experiments.static_image_bridge.render_worker import encode_ffv1_stream

    encode_ffv1_stream(frames, width=frames.shape[2], height=frames.shape[1], output=path, ffmpeg=ffmpeg)


def test_render_argv_binds_correct_worker_model_audio_and_xyxy(tmp_path: Path) -> None:
    config = {
        "paths": {
            "wav2lip_python": "/env/wav2lip/bin/python",
            "render_worker": "/repo/render_worker.py",
            "wav2lip_checkpoint": "/repo/wav2lip_gan.pth",
            "ffmpeg": "/tools/ffmpeg",
        },
        "models": {"wav2lip_batch_size": 4, "seed": 20260923, "device": "cuda"},
    }
    argv = build_render_argv(
        config,
        image=tmp_path / "face.png",
        image_sha256="a" * 64,
        audio=tmp_path / "natural.wav",
        box_xyxy=[138, 90, 357, 387],
        output=tmp_path / "N.mkv",
        receipt=tmp_path / "N.json",
    )

    assert argv[:2] == ["/env/wav2lip/bin/python", "/repo/render_worker.py"]
    assert argv[argv.index("--audio") + 1].endswith("natural.wav")
    assert argv[argv.index("--box") + 1:argv.index("--box") + 5] == ["138", "90", "357", "387"]
    assert argv[argv.index("--checkpoint") + 1] == "/repo/wav2lip_gan.pth"
    assert "--face" not in argv and "-shortest" not in argv


def test_render_argv_keeps_the_venv_interpreter_symlink(tmp_path: Path) -> None:
    python_link = tmp_path / "venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to("/usr/bin/python3.12")
    config = {
        "paths": {
            "wav2lip_python": str(python_link),
            "render_worker": "/repo/render_worker.py",
            "wav2lip_checkpoint": "/repo/wav2lip_gan.pth",
            "ffmpeg": "/tools/ffmpeg",
        },
        "models": {"wav2lip_batch_size": 4, "seed": 20260923, "device": "cuda"},
    }

    argv = build_render_argv(
        config,
        image=tmp_path / "face.png",
        image_sha256="a" * 64,
        audio=tmp_path / "natural.wav",
        box_xyxy=[138, 90, 357, 387],
        output=tmp_path / "N.mkv",
        receipt=tmp_path / "N.json",
    )

    assert argv[0] == str(python_link.absolute())
    assert Path(argv[0]) != python_link.resolve()


def test_worker_receipt_requires_real_hashes_static_png_and_frozen_paths(tmp_path: Path) -> None:
    root = tmp_path / "Wav2Lip"
    (root / "models").mkdir(parents=True)
    audio_module = root / "audio.py"
    models_module = root / "models" / "__init__.py"
    class_source = root / "models" / "wav2lip.py"
    for path in (audio_module, models_module, class_source):
        path.write_text("source\n")
    image = tmp_path / "face.png"
    image.write_bytes(b"png bytes")
    audio = tmp_path / "natural.wav"
    audio.write_bytes(b"n pcm")
    checkpoint = tmp_path / "wav2lip_gan.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "video.mkv"
    output.write_bytes(b"ffv1")
    image_sha = "1" * 64
    audio_sha = file_sha256(audio)
    config = {"paths": {"wav2lip_checkpoint": str(checkpoint), "wav2lip_python": "/env/bin/python", "wav2lip_root": str(root)}}
    receipt = {
        "status": "complete",
        "input_mode": "one_png_only",
        "image": str(image),
        "image_rgb_sha256": image_sha,
        "audio": str(audio),
        "audio_sha256": audio_sha,
        "box_xyxy": [138, 90, 357, 387],
        "source_frame_indices": [0, 0],
        "frames_rendered": 2,
        "fps": 25.0,
        "output_codec": "ffv1",
        "output": str(output),
        "output_sha256": file_sha256(output),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "python_executable": "/env/bin/python",
        "python_realpath": "/env/bin/python",
        "python_prefix": "/env",
        "loaded_parameter_sha256": hashlib.sha256(b"model").hexdigest(),
    }
    for key, path in (("audio_module", audio_module), ("models_module", models_module), ("wav2lip_class_source", class_source)):
        receipt[key] = str(path)
        receipt[f"{key}_sha256"] = file_sha256(path)

    _verify_worker_receipt(
        receipt, config=config, image=image, image_sha=image_sha,
        audio=audio, audio_sha=audio_sha, box=[138, 90, 357, 387], output=output,
    )
    receipt["audio_sha256"] = "0" * 64
    with pytest.raises(ProtocolError, match="audio role"):
        _verify_worker_receipt(
            receipt, config=config, image=image, image_sha=image_sha,
            audio=audio, audio_sha=audio_sha, box=[138, 90, 357, 387], output=output,
        )
    receipt["audio_sha256"] = audio_sha
    receipt["python_executable"] = "/usr/bin/python3.12"
    with pytest.raises(ProtocolError, match="frozen Python environment"):
        _verify_worker_receipt(
            receipt, config=config, image=image, image_sha=image_sha,
            audio=audio, audio_sha=audio_sha, box=[138, 90, 357, 387], output=output,
        )


def test_static_image_rgb_hash_uses_decoded_rgb_pixels(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    pixels = np.zeros((4, 5, 3), dtype=np.uint8)
    pixels[1, 2] = [17, 32, 248]
    assert cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    assert image_rgb_sha256(path) == hashlib.sha256(pixels.tobytes()).hexdigest()


@pytest.mark.parametrize("difference,expected_action", [(2, "repeat_last_frame"), (-2, "right_crop"), (0, "identity")])
def test_tail_repair_only_changes_the_video_boundary(
    tmp_path: Path,
    difference: int,
    expected_action: str,
) -> None:
    ffmpeg = Path("/home/wjj/miniconda3/bin/ffmpeg")
    raw = tmp_path / "raw.mkv"
    source = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    for index in range(source.shape[0]):
        source[index] = index * 10
    _write_video(raw, source, ffmpeg)
    target_count = source.shape[0] + difference
    output = tmp_path / "canonical.mkv"

    receipt = canonicalize_tail(raw, output, target_frame_count=target_count, ffmpeg=ffmpeg)

    capture = cv2.VideoCapture(str(output))
    result = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result.append(frame)
    capture.release()
    decoded = np.stack(result)
    assert decoded.shape[0] == target_count
    assert receipt["tail_action"] == expected_action
    if difference > 0:
        assert np.array_equal(decoded[:6], source)
        assert np.array_equal(decoded[-1], source[-1])
    elif difference < 0:
        assert np.array_equal(decoded, source[:-2])
    else:
        assert np.array_equal(decoded, source)


def test_tail_repair_rejects_more_than_five_frames(tmp_path: Path) -> None:
    ffmpeg = Path("/home/wjj/miniconda3/bin/ffmpeg")
    raw = tmp_path / "raw.mkv"
    _write_video(raw, np.zeros((6, 8, 8, 3), dtype=np.uint8), ffmpeg)

    with pytest.raises(ProtocolError, match="MEDIA_LENGTH_MISMATCH"):
        canonicalize_tail(raw, tmp_path / "canonical.mkv", target_frame_count=12, ffmpeg=ffmpeg)
