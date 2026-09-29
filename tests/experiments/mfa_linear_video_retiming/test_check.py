from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.check import (
    _candidate_for_map_sha,
    _generation_frame_counts,
    _expected_official_cells,
    _render_pixels_independently,
    np_metrics,
    rebuild_q_independently,
)
from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError
from scripts.experiments.mfa_linear_video_retiming.retime import build_map


def test_checker_finds_selected_candidate_by_map_hash_not_candidate_id() -> None:
    mapping = {"frame_count": 4, "q": [0.0, 1.0, 2.0, 3.0]}
    state = {"candidates": {
        "identity": {"candidate_id": "identity", "map_sha256": "map-hash", "map": mapping},
        "alias": {"candidate_id": "alias", "map_sha256": "map-hash", "map": dict(mapping)},
    }}

    candidate = _candidate_for_map_sha(state, "map-hash", sample_id="1")

    assert candidate["candidate_id"] == "alias"
    with pytest.raises(ProtocolError, match="SELECTED_MAP_CANDIDATE_MISSING"):
        _candidate_for_map_sha(state, "missing", sample_id="1")


def test_checker_rejects_map_hash_collision_with_different_maps() -> None:
    state = {"candidates": {
        "a": {"candidate_id": "a", "map_sha256": "same", "map": {"q": [0, 1]}},
        "b": {"candidate_id": "b", "map_sha256": "same", "map": {"q": [0, 2]}},
    }}

    with pytest.raises(ProtocolError, match="SELECTED_MAP_HASH_COLLISION"):
        _candidate_for_map_sha(state, "same", sample_id="1")


def test_checker_rebuilds_map_from_knots_and_rejects_tampered_values() -> None:
    mapping = build_map(64, 62, [0.5, 1.0, 0.0], positions=[5, 29, 55])

    assert np.array_equal(rebuild_q_independently(mapping), np.asarray(mapping["q"]))
    mapping["q"][20] += 0.25
    with pytest.raises(ProtocolError, match="RECONSTRUCTION_MISMATCH"):
        rebuild_q_independently(mapping)


def test_checker_pixel_renderer_uses_float64_blend_and_round_half_up() -> None:
    source = np.asarray([0, 1, 2, 3], dtype=np.uint8).reshape(4, 1, 1, 1)
    source = np.repeat(source, 3, axis=3)
    q = np.asarray([0.0, 1.5, 2.0, 3.0], dtype=np.float64)

    rendered = _render_pixels_independently(source, q)

    assert rendered[:, 0, 0, 0].tolist() == [0, 2, 2, 3]


def test_generation_frame_support_uses_paired_audio_sample_count() -> None:
    audio = {
        "sample_count": 83_937,
        "natural": {"path": "/audio/N.wav", "sha256": "n"},
        "mfa_linear": {"path": "/audio/M.wav", "sha256": "m"},
    }

    assert _generation_frame_counts(audio, 132) == (132, 132)
    assert _generation_frame_counts(audio, 127) == (132, 127)


def test_independent_metric_recompute_uses_lag_curve_and_local_blocks() -> None:
    matrix = np.full((50, 31), 2.0, dtype=np.float32)
    matrix[:, 16] = 1.0
    matrix[:25, 14] = 0.5
    metrics = np_metrics(matrix, np.arange(50))

    assert metrics["sync_d"] == pytest.approx(1.0)
    assert metrics["offset"] == -1
    assert len([row for row in metrics if row == "local_median_d0"]) == 1
    assert metrics["local_abs_offset_q90"] >= 0.0


def _official_fixture(sample_ids: list[str], *, smoke: bool):
    frozen_records = []
    sealed_records = []
    transfer_records = []
    for sid in sample_ids:
        frozen_records.append({
            "sample_id": sid, "paired_key": f"key-{sid}", "speaker_id": f"speaker-{sid}",
            "audio": {"natural": {"path": f"/audio/N-{sid}.wav"},
                      "mfa_linear": {"path": f"/audio/M-{sid}.wav"}},
        })
        videos = {arm: {"path": f"/video/p3-{arm}-{sid}.mkv"}
                  for arm in ("N", "M", "R", "GLOBAL", "NEAREST", "MIRROR")}
        sealed_records.append({"sample_id": sid, "videos": videos})
        portraits = {}
        for portrait in ("6", "9"):
            portraits[portrait] = {
                "N": {"canonical_video": f"/video/p{portrait}-N-{sid}.mkv"},
                "M": {"canonical_video": f"/video/p{portrait}-M-{sid}.mkv"},
                "R": {"path": f"/video/p{portrait}-R-{sid}.mkv"},
            }
        transfer_records.append({"sample_id": sid, "portraits": portraits})
    return ({"records": frozen_records}, {"records": sealed_records}, {"records": transfer_records})


def test_checker_reconstructs_exact_8_cell_smoke_and_42_cell_formal_denominators() -> None:
    frozen, sealed, transfer = _official_fixture(["1"], smoke=True)
    smoke = _expected_official_cells(frozen, sealed, transfer, engineering_only=True)
    assert len(smoke) == 8
    assert [row["cell_key"] for row in smoke[:6]] == [
        "p3_s1_vN_aN", "p3_s1_vM_aN", "p3_s1_vR_aN", "p3_s1_vGLOBAL_aN",
        "p3_s1_vNEAREST_aN", "p3_s1_vMIRROR_aN",
    ]
    assert [row["audio_role"] for row in smoke[-2:]] == ["M", "M"]

    frozen, sealed, transfer = _official_fixture(["1", "101", "201"], smoke=False)
    formal = _expected_official_cells(frozen, sealed, transfer, engineering_only=False)
    assert len(formal) == 42
    assert sum(row["portrait_id"] == "3" for row in formal) == 24
    assert sum(row["portrait_id"] in {"6", "9"} for row in formal) == 18
    assert formal[6]["audio"] == "/audio/M-1.wav"
    assert formal[8]["portrait_id"] == "6"
