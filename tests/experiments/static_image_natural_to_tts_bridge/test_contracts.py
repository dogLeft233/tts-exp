from scripts.experiments.static_image_bridge import config
from scripts.experiments.static_image_bridge.common import (
    ProtocolError,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def test_frozen_cell_counts_and_stage_partition() -> None:
    assert config.expected_video_count() == 110
    assert config.expected_cell_count() == 198
    assert config.expected_cell_count("A") == 66
    assert config.expected_cell_count("B") == 132
    assert set(config.STAGE_A_CELLS).isdisjoint(config.STAGE_B_CELLS)


def test_self_hashed_json_detects_tampering(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_self_hashed_json(path, {"status": "complete", "value": 1})
    assert verify_self_hashed_json(path)["value"] == 1
    path.write_text(path.read_text(encoding="utf-8").replace('"value": 1', '"value": 2'), encoding="utf-8")
    try:
        verify_self_hashed_json(path)
    except ProtocolError as exc:
        assert "self-hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered self-hashed JSON was accepted")
