from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_face_roi_replacement import config
from scripts.experiments.wav2lip_face_roi_replacement.analysis import _gate_ci
from scripts.experiments.wav2lip_face_roi_replacement.audio import (
    phase_preserving_blend,
    reconstruct_warped_pcm,
)
from scripts.experiments.wav2lip_face_roi_replacement.common import (
    file_sha256,
    write_self_hashed_json,
)
from scripts.experiments.wav2lip_face_roi_replacement.render import _audio_paths
from scripts.experiments.wav2lip_face_roi_replacement.roi import _review_row
from scripts.experiments.wav2lip_face_roi_replacement.scoring import reconstruct_global
from scripts.experiments.wav2lip_face_roi_replacement.validate import (
    bootstrap_ci,
    validate_run,
)


def test_registered_counts_and_run_root_are_frozen() -> None:
    assert config.EXPECTED_CONTROL_VIDEO_COUNT == 66
    assert config.EXPECTED_CONTROL_SCORE_COUNT == 198
    assert config.EXPECTED_BRIDGE_VIDEO_COUNT == 22
    assert config.EXPECTED_BRIDGE_CELL_COUNT == 44
    assert config.run_root_for("smoke").name == "wav2lip_face_roi_replacement_smoke"


def test_video_arms_resolve_to_registered_audio_arms(tmp_path) -> None:
    paths = {}
    arms = (config.AUDIO_N, config.AUDIO_N_REPEAT, config.AUDIO_W)
    manifest_arms = []
    for arm in arms:
        path = tmp_path / f"{arm}.wav"
        path.write_bytes(arm.encode("ascii"))
        paths[arm] = str(path)
        manifest_arms.append({"arm": arm, "output": str(path), "output_sha256": file_sha256(path)})
    record = {"sample_id": "sample", "audio": {"manifest_row": {"arms": manifest_arms}}}

    resolved = _audio_paths(record, (config.VIDEO_GN, config.VIDEO_GW, config.VIDEO_GNR, config.VIDEO_GB))

    assert resolved == {
        config.VIDEO_GN: str(tmp_path / "N.wav"),
        config.VIDEO_GW: str(tmp_path / "W.wav"),
        config.VIDEO_GNR: str(tmp_path / "N_REPEAT.wav"),
        config.VIDEO_GB: str(tmp_path / "N.wav"),
    }


def test_warp_reconstruction_preserves_length_and_endpoints() -> None:
    samples = np.rint(7000.0 * np.sin(np.linspace(0.0, 40.0, 16_000, endpoint=False))).astype("<i2")
    warped, mapping = reconstruct_warped_pcm(samples.tobytes())
    values = np.frombuffer(warped, dtype="<i2")
    assert values.size == samples.size
    assert mapping[0] == 0.0
    assert mapping[-1] == samples.size - 1
    assert np.all(np.diff(mapping) > 0.0)


def test_bridge_reconstruction_is_finite_and_fixed_length() -> None:
    natural = np.rint(5000.0 * np.sin(np.linspace(0.0, 80.0, 16_000, endpoint=False))).astype("<i2")
    mfa = np.rint(5000.0 * np.sin(np.linspace(0.0, 84.0, 16_000, endpoint=False))).astype("<i2")
    waveform, metadata = phase_preserving_blend(natural, mfa)
    assert waveform.size == natural.size
    assert np.isfinite(waveform).all()
    assert metadata["alpha"] == 0.75
    assert metadata["phase_policy"] == "natural_phase"


def test_roi_review_rejects_full_frame_and_accepts_local_box(tmp_path) -> None:
    overlays = {}
    for label in ("head", "middle", "tail"):
        path = tmp_path / f"{label}.png"
        path.write_bytes(b"overlay")
        overlays[label] = str(path)
    base = {"sample_id": "x", "source_group": "g", "frame_count": 3, "width": 224, "height": 224, "raw_boxes": [[10, 10, 100, 120]] * 3, "overlays": overlays}
    accepted = _review_row({**base, "boxes": [[10, 120, 10, 100]] * 3})
    rejected = _review_row({**base, "boxes": [[0, 224, 0, 224]] * 3})
    assert accepted["decision"] == "PASS"
    assert rejected["decision"] == "BLOCKED"
    assert any("full-frame" in error for error in rejected["errors"])


def test_official_global_reconstruction_uses_vshift_columns() -> None:
    matrix = np.ones((12, 31), dtype=np.float64)
    matrix[:, 18] = 0.25
    result = reconstruct_global(matrix)
    assert result["offset"] == -3
    assert result["sync_d"] == 0.25
    assert result["sync_c"] == 0.75


def test_validator_bootstrap_is_sorted_group_bootstrap() -> None:
    values = [1.0, 3.0, 10.0]
    groups = ["b", "a", "c"]
    result = bootstrap_ci(values, groups, draws=100)
    assert len(result) == 2
    assert result[0] <= np.mean(values) <= result[1]


def test_strict_gate_does_not_accept_boundary_ci() -> None:
    assert not _gate_ci({"ci95": [-0.10, 0.09]}, lower=-0.10, strict_lower=True)
    assert not _gate_ci({"ci95": [-0.01, 0.10]}, upper=0.10, strict_upper=True)
    assert _gate_ci({"ci95": [-0.09, 0.09]}, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True)


def test_validator_accepts_only_a_self_hashed_engineering_block(tmp_path) -> None:
    result_path = tmp_path / "result.md"
    result_path.write_text("BLOCKED\n", encoding="utf-8")
    write_self_hashed_json(
        tmp_path / "final.json",
        {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": None,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
        },
    )
    validation = validate_run(tmp_path)
    assert validation["status"] == "valid"
    assert validation["mode"] == "engineering_blocked"
