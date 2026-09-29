import pytest

from scripts.experiments.static_image_bridge import config, generate
from scripts.experiments.static_image_bridge.common import (
    ProtocolError,
    file_sha256,
    write_self_hashed_json,
)


def test_successful_video_cell_is_resumed_without_worker_rerun(tmp_path, monkeypatch) -> None:
    paths = config.RunPaths(tmp_path)
    output = paths.video_dir / "N" / "sample.mkv"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"complete-video")
    sidecar = output.with_suffix(".json")
    write_self_hashed_json(
        sidecar,
        {
            "sample_id": "sample",
            "video_arm": "N",
            "output_sha256": file_sha256(output),
            "source_frame_indices": [0] * 25,
            "frame_count": 25,
        },
    )
    monkeypatch.setattr(generate, "run_logged", lambda *args, **kwargs: pytest.fail("successful cell was rerun"))
    result = generate.render_one(paths, {"sample_id": "sample", "static_reference": {}}, {}, "N")
    assert result["output_sha256"] == file_sha256(output)


def test_partial_video_cell_is_rejected_instead_of_silently_retried(tmp_path) -> None:
    paths = config.RunPaths(tmp_path)
    output = paths.video_dir / "N" / "sample.mkv"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"partial-video")
    with pytest.raises(ProtocolError, match="partial video cell"):
        generate.render_one(paths, {"sample_id": "sample", "static_reference": {}}, {}, "N")
