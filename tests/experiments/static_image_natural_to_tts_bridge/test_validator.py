import numpy as np
import pytest

from scripts.experiments.static_image_bridge import config
from scripts.experiments.static_image_bridge.common import (
    ProtocolError,
    file_sha256,
    write_pcm16,
    write_self_hashed_json,
)
from scripts.experiments.static_image_bridge.validate import _validate_scores


def _fixture(tmp_path):
    inputs = {"records": []}
    audio_rows = []
    video_rows = {}
    score_rows = []
    for index in range(config.EXPECTED_RECORD_COUNT):
        sid = f"sample_{index:02d}"
        score_box = {"order": "fixture", "center": [1.0, 1.0], "side": 2, "box": [0, 0, 2, 2], "scale_from_detection": 1.5}
        inputs["records"].append({"sample_id": sid, "source_group": sid, "static_reference": {"score_box": score_box}})
        n_path = tmp_path / f"{sid}_N.wav"
        b_path = tmp_path / f"{sid}_B.wav"
        n_meta = write_pcm16(n_path, np.zeros(1024, dtype=np.int16))
        b_meta = write_pcm16(b_path, np.ones(1024, dtype=np.int16))
        audio_rows.append({"sample_id": sid, "arms": {"N": n_meta, "B": b_meta}, "controls": {"ND": n_meta}})
        video_path = tmp_path / f"{sid}_N.mkv"
        video_path.write_bytes(b"fixture-video")
        video_rows[(sid, "N")] = {"output": str(video_path), "output_sha256": file_sha256(video_path)}
        matrix_path = tmp_path / f"{sid}_matrix.npy"
        np.save(matrix_path, np.ones((20, 31), dtype=np.float64), allow_pickle=False)
        score_rows.append(
            {
                "protocol_id": config.PROTOCOL_ID,
                "sample_id": sid,
                "video_arm": "N",
                "audio_arm": "N",
                "video_path": str(video_path),
                "video_sha256": file_sha256(video_path),
                "audio_path": str(n_path),
                "audio_sha256": file_sha256(n_path),
                "audio_pcm_sha256": n_meta["pcm_sha256"],
                "fixed_score_box": score_box,
                "matrix_path": str(matrix_path),
                "matrix_sha256": file_sha256(matrix_path),
                "matrix_shape": [20, 31],
                "device": "cuda",
                "model_sha256": config.SYNCNET_MODEL_SHA256,
            }
        )
    paths = config.RunPaths(tmp_path)
    write_self_hashed_json(paths.scores_manifest, {"status": "complete", "rows": score_rows})
    return paths, inputs, {"rows": audio_rows}, video_rows, score_rows


def test_validator_accepts_bound_fixture_and_rejects_matrix_audio_and_missing_cell(tmp_path) -> None:
    paths, inputs, audio, videos, rows = _fixture(tmp_path)
    expected = (("N", "N"),)
    assert len(_validate_scores(paths, inputs, expected, audio, videos)) == config.EXPECTED_RECORD_COUNT

    matrix_path = tmp_path / "sample_00_matrix.npy"
    np.save(matrix_path, np.zeros((20, 31), dtype=np.float64), allow_pickle=False)
    with pytest.raises(ProtocolError, match="matrix hash changed"):
        _validate_scores(paths, inputs, expected, audio, videos)

    paths, inputs, audio, videos, rows = _fixture(tmp_path / "audio_binding")
    rows[0]["audio_path"] = audio["rows"][0]["arms"]["B"]["path"]
    rows[0]["audio_sha256"] = audio["rows"][0]["arms"]["B"]["container_sha256"]
    write_self_hashed_json(paths.scores_manifest, {"status": "complete", "rows": rows})
    with pytest.raises(ProtocolError, match="audio binding changed"):
        _validate_scores(paths, inputs, expected, audio, videos)

    paths, inputs, audio, videos, rows = _fixture(tmp_path / "missing_cell")
    write_self_hashed_json(paths.scores_manifest, {"status": "complete", "rows": rows[1:]})
    with pytest.raises(ProtocolError, match="score cell set mismatch"):
        _validate_scores(paths, inputs, expected, audio, videos)
