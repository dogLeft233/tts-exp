from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments import wav2lip_probe_runtime as rt
from scripts.experiments.wav2lip_phone_core_shrinkage import config
from scripts.experiments.wav2lip_phone_core_shrinkage.construct import (
    construct_candidates,
    exposure_diagnostics,
)
from scripts.experiments.wav2lip_phone_core_shrinkage.runner import (
    _audio_binding_equal,
    _candidates,
    _identity_hashes,
    _parent_n_score,
    _prepare,
    _replay_plan,
    group_exposure,
)
from scripts.experiments.wav2lip_phone_core_shrinkage.validate import (
    _validate_exposure_contract,
    _validate_old_driver_replay,
)
from scripts.experiments.wav2lip_phone_core_shrinkage.validator_math import (
    matrix_from_embeddings,
)
from scripts.experiments.wav2lip_probe_runtime import ProtocolError


def test_length_five_weights_and_boundary_identity():
    mel = np.zeros((80, 308), dtype=np.float32)
    mel[:, 105:110] = np.arange(5, dtype=np.float32)
    result = construct_candidates(
        mel, [{"mask_sha256": "x", "start": 105, "end": 110, "label": "a"}]
    )
    assert result["metadata"]["cores"][0]["weights"] == [0.0, 0.5, 1.0, 0.5, 0.0]
    assert np.array_equal(result["PHONE_CORE"][:, :105], mel[:, :105])
    assert np.array_equal(result["PHONE_CORE"][:, 110:], mel[:, 110:])


def test_frequency_dimension_is_not_time_dimension():
    mel = np.zeros((80, 308), dtype=np.float32)
    mel[3, 105] = 2.0
    result = construct_candidates(
        mel, [{"mask_sha256": "x", "start": 105, "end": 110, "label": "a"}]
    )
    assert result["PHONE_CORE"].shape == (80, 308)
    assert result["PHONE_CORE"][3, 105] == pytest.approx(2.0)


def test_short_core_remains_natural_when_no_weight_is_written():
    mel = np.random.default_rng(1).normal(size=(80, 308)).astype(np.float32)
    result = construct_candidates(mel, [])
    assert np.array_equal(result["N"], mel)


def test_record_without_u_exposure_is_reported_without_rejecting_candidate():
    mel = (0.1 * np.random.default_rng(2).normal(size=(80, 308))).astype(np.float32)
    cores = [{"mask_sha256": "outside", "start": 0, "end": 5, "label": "a"}]
    result = construct_candidates(mel, cores)

    diagnostic = exposure_diagnostics(mel, result, cores)
    assert diagnostic["record_exposed"] is False
    assert diagnostic["exposed_rows"] == []
    assert diagnostic["changed_columns"] == {
        "PHONE_CORE": [1, 2, 3],
        "GENERIC_CORE": [1, 2, 3],
    }
    assert result["metadata"]["record_exposed"] is False


def test_group_exposure_keeps_pair_when_one_record_has_no_exposure():
    rows = [
        {"source_group": "g1", "record_exposed": False},
        {"source_group": "g1", "record_exposed": True},
        {"source_group": "g2", "record_exposed": False},
        {"source_group": "g2", "record_exposed": False},
    ]
    support = group_exposure(rows)
    assert support == {"g1": True, "g2": False}


def test_replay_plan_matches_shared_gpu_generation_schema(tmp_path):
    paths = config.RunPaths(tmp_path)
    parent = {
        "rows": [
            {
                "sample_id": "a",
                "video_arm": "N",
                "audio_arm": "N",
                "score": {"source_audio": "/audio-a.wav"},
            },
            {
                "sample_id": "b",
                "video_arm": "N",
                "audio_arm": "N",
                "score": {"source_audio": "/audio-b.wav"},
            },
        ]
    }
    drivers = {
        "rows": [
            {
                "sample_id": "a",
                "source_group": "g1",
                "static_face": {"path": "/face-a.npy"},
                "arms": {"N": {"path": "/mel-a.npy"}},
            },
            {
                "sample_id": "b",
                "source_group": "g2",
                "static_face": {"path": "/face-b.npy"},
                "arms": {"N": {"path": "/mel-b.npy"}},
            },
        ]
    }

    plan = _replay_plan(paths, parent, drivers)

    assert len(plan["rows"]) == 2
    for row in plan["rows"]:
        assert set(row) == {
            "sample_id",
            "static_face",
            "source_audio",
            "arms",
            "outputs",
        }
        assert set(row["arms"]) == {"N_REPLAY"}
        assert set(row["outputs"]) == {"N_REPLAY"}


def test_audio_binding_accepts_fresh_and_parent_pcm_field_names():
    fresh = {"source_audio_sha256": "audio", "frozen_pcm_sha256": "pcm"}
    parent = {"source_audio_sha256": "audio", "source_pcm_sha256": "pcm"}
    assert _audio_binding_equal(fresh, parent)
    assert not _audio_binding_equal({**fresh, "source_audio_sha256": "changed"}, parent)
    assert not _audio_binding_equal({**fresh, "frozen_pcm_sha256": "changed"}, parent)


def test_replay_uses_parent_natural_audio_cell():
    parent = {
        "rows": [
            {
                "sample_id": "a",
                "video_arm": "N",
                "audio_arm": "A_DELAY",
                "score": {"source_audio": "delay"},
            },
            {
                "sample_id": "a",
                "video_arm": "N",
                "audio_arm": "N",
                "score": {"source_audio": "natural"},
            },
        ]
    }
    assert _parent_n_score(parent, "a")["source_audio"] == "natural"


def test_validator_rebuilds_long_parity_matrix():
    visual = np.zeros((158, 1024), dtype=np.float32)
    audio = np.zeros((158, 1024), dtype=np.float32)
    assert matrix_from_embeddings(visual, audio).shape == (158, 31)


@pytest.mark.parametrize(
    "validation_payload", [None, {"status": "FAIL"}, {"status": "PASS"}]
)
def test_candidates_refuse_missing_or_failed_control_validation(
    tmp_path, validation_payload
):
    paths = config.RunPaths(tmp_path)
    rt.write_json(paths.controls, {"status": "PASS"})
    if validation_payload is not None:
        rt.write_json(paths.control_validation, validation_payload)
    with pytest.raises(ProtocolError, match="independent control valid"):
        _candidates(paths, {})


def test_prepare_resume_rejects_changed_code_spec_or_helper_identity(
    tmp_path, monkeypatch
):
    paths = config.RunPaths(tmp_path)
    rt.write_json(
        paths.protocol, {"code_sha256": {}, "helper_sha256": {}, "spec_sha256": {}}
    )
    monkeypatch.setattr(
        "scripts.experiments.wav2lip_phone_core_shrinkage.runner._fixed",
        lambda: ({}, {}, {}),
    )

    with pytest.raises(ProtocolError, match="resume identity changed"):
        _prepare(paths, resume=True)


def test_identity_hashes_include_local_code_and_external_helpers():
    identity = _identity_hashes()
    assert (
        "scripts/experiments/wav2lip_phone_core_shrinkage/runner.py"
        in identity["code_sha256"]
    )
    assert "scripts/experiments/wav2lip_probe_runtime.py" in identity["helper_sha256"]
    assert (
        "openspec/changes/complete-wav2lip-phone-core-group-support/design.md"
        in identity["spec_sha256"]
    )


def test_resigned_exposure_audit_is_rejected_when_record_exposure_changes():
    saved_rows = [
        {"sample_id": "a", "source_group": "g"},
        {"sample_id": "b", "source_group": "g"},
    ]
    for index in range(7):
        saved_rows.extend(
            [
                {"sample_id": f"s{index}a", "source_group": f"g{index}"},
                {"sample_id": f"s{index}b", "source_group": f"g{index}"},
            ]
        )
    saved = {"rows": saved_rows}
    expected = {
        str(row["sample_id"]): (str(row["source_group"]), [30]) for row in saved_rows
    }
    expected["b"] = ("g", [])
    protocol = {
        "exposure_gate": {
            "unit": "source_group",
            "group_count": 8,
            "unsupported_groups": [],
        }
    }
    audit = {
        "rows": [
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "record_exposed": bool(expected[row["sample_id"]][1]),
                "group_exposed": True,
                "exposed_rows": expected[row["sample_id"]][1],
            }
            for row in saved_rows
        ]
    }
    _validate_exposure_contract(protocol, audit, saved, expected)
    audit["rows"][0]["record_exposed"] = False
    with pytest.raises(ProtocolError, match="exposure audit mismatch"):
        _validate_exposure_contract(protocol, audit, saved, expected)


def test_old_successful_driver_rows_are_replayed_exactly():
    drivers = rt.load_self(config.RunPaths(config.OLD_RUN).drivers)
    assert _validate_old_driver_replay(drivers) == 14
