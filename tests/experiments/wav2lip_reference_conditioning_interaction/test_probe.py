from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_reference_conditioning_interaction import (
    config as runner_config,
)
from scripts.experiments.wav2lip_reference_conditioning_interaction.common import (
    ProtocolError,
    file_sha256,
    group_bootstrap,
    write_json,
)
from scripts.experiments.wav2lip_reference_conditioning_interaction.runner import (
    _gpu,
    _identity_hashes,
    _parity_source_group,
    _prepare,
)
from scripts.experiments.wav2lip_reference_conditioning_interaction.score import (
    matrix_from_embeddings,
    metrics,
    worker_arrays,
)
from scripts.experiments.wav2lip_reference_conditioning_interaction.validate import (
    _domain,
    _fixed_input_errors,
    _full_matrix,
    _full_metrics,
    _score_metrics_from_raw,
)


def test_group_bootstrap_is_eight_source_groups() -> None:
    values = [float(i) for i in range(16)]
    groups = [f"g{i // 2}" for i in range(16)]
    result = group_bootstrap(values, groups, draws=200)
    assert result["draws"] == 200
    assert len(result["group_labels"]) == 8


def test_missing_reference_is_not_silently_substituted(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        np.load(tmp_path / "missing.npy", allow_pickle=False)


def test_matched_delay_uses_q_r_plus_j_minus_ten() -> None:
    visual = np.zeros((88, 1024), dtype=np.float32)
    audio = np.zeros((88, 1024), dtype=np.float32)
    audio[35, 0] = 7.0
    visual[30, 0] = 7.0
    natural = matrix_from_embeddings(visual, audio, lag_start=-15, rows=[30])
    matched = matrix_from_embeddings(visual, audio, lag_start=-10, rows=[30])
    assert int(np.argmin(natural[0])) == 20
    assert int(np.argmin(matched[0])) == 15
    assert metrics(natural, lag_start=-15)["offset"] == -5
    assert metrics(matched, lag_start=-10)["offset"] == -5


def test_matched_domain_rejects_rows_without_real_audio_support() -> None:
    visual = np.zeros((88, 1024), dtype=np.float32)
    audio = np.zeros((88, 1024), dtype=np.float32)
    with pytest.raises(ProtocolError):
        matrix_from_embeddings(visual, audio, lag_start=-10, rows=[0])


def test_validator_domain_has_expected_endpoint_and_physical_offset() -> None:
    visual = np.zeros((88, 1024), dtype=np.float32)
    audio = np.zeros((88, 1024), dtype=np.float32)
    audio[35, 0] = 7.0
    visual[30, 0] = 7.0
    natural = _domain(visual, audio, -15)
    matched = _domain(visual, audio, -10)
    assert natural["min_index"] == 20 and natural["offset"] == -5
    assert matched["min_index"] == 15 and matched["offset"] == -5
    assert len(natural["curve"]) == 31 and len(matched["curve"]) == 31


def test_validator_rejects_changed_frozen_binding(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen")
    config = __import__(
        "scripts.experiments.wav2lip_reference_conditioning_interaction.validate",
        fromlist=["config"],
    ).config
    monkeypatch.setattr(
        config,
        "FIXED_HASHES",
        {
            name: file_sha256(source)
            for name in (
                "drivers",
                "control_scores",
                "q_candidates",
                "old_control_scores",
                "old_protocol",
                "old_drivers",
                "old_reference",
                "checkpoint",
                "syncnet",
            )
        },
    )
    for name in (
        "DRIVERS",
        "CONTROL_SCORES",
        "Q_CANDIDATES",
        "OLD_CONTROL_SCORES",
        "OLD_PROTOCOL",
        "OLD_DRIVERS",
        "OLD_REFERENCE",
        "WAV2LIP_CHECKPOINT",
        "SYNCNET_MODEL",
    ):
        monkeypatch.setattr(config, name, source)
    refs = {
        "rows": [
            {"sample_id": "x", "source_video": {"path": str(source), "sha256": "wrong"}}
        ]
    }
    errors = _fixed_input_errors({"rows": []}, refs)
    assert "binding_changed:x/source_video" in errors


def test_worker_arrays_accepts_long_parity_embedding_cache(tmp_path) -> None:
    visual_path, audio_path = tmp_path / "visual.npy", tmp_path / "audio.npy"
    visual = np.zeros((158, 1024), dtype=np.float32)
    audio = np.zeros_like(visual)
    np.save(visual_path, visual, allow_pickle=False)
    np.save(audio_path, audio, allow_pickle=False)
    worker_path = tmp_path / "worker.json"
    write_json(
        worker_path,
        {
            "visual": str(visual_path),
            "visual_sha256": file_sha256(visual_path),
            "audio_embedding": str(audio_path),
            "audio_embedding_sha256": file_sha256(audio_path),
        },
    )
    loaded_visual, loaded_audio = worker_arrays(
        {"worker": str(worker_path)}, expected_rows=None
    )
    assert loaded_visual.shape == loaded_audio.shape == (158, 1024)


def test_gpu_plan_resume_ignores_only_its_self_hash(tmp_path) -> None:
    paths = runner_config.RunPaths(tmp_path)
    plan_rows = [{"sample_id": "x", "arm": "N"}]
    plan_path = tmp_path / "plans" / "probe.json"
    from scripts.experiments.wav2lip_reference_conditioning_interaction.common import (
        write_json as write_common_json,
    )

    write_common_json(
        plan_path,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_reference_conditioning_interaction",
            "label": "probe",
            "rows": plan_rows,
        },
    )
    result_path = tmp_path / "generation" / "probe.json"
    write_common_json(
        result_path,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_reference_conditioning_interaction",
            "status": "complete",
            "rows": [],
        },
    )
    assert _gpu(paths, "probe", plan_rows, resume=True)["status"] == "complete"


def test_prepare_resume_rejects_changed_code_spec_or_helper_identity(
    tmp_path, monkeypatch
):
    paths = runner_config.RunPaths(tmp_path)
    write_json(
        paths.protocol, {"code_sha256": {}, "helper_sha256": {}, "spec_sha256": {}}
    )
    rows = [{"sample_id": f"s{i}", "source_group": f"g{i // 2}"} for i in range(16)]
    old_rows = [{"sample_id": row["sample_id"]} for row in rows]
    monkeypatch.setattr(
        "scripts.experiments.wav2lip_reference_conditioning_interaction.runner._fixed",
        lambda: ({"rows": rows}, {}, {}, {}),
    )
    original_load = __import__(
        "scripts.experiments.wav2lip_reference_conditioning_interaction.runner",
        fromlist=["load_self_hashed"],
    ).load_self_hashed

    def load(path):
        if path == runner_config.OLD_REFERENCE:
            return {"rows": old_rows}
        return original_load(path)

    monkeypatch.setattr(
        "scripts.experiments.wav2lip_reference_conditioning_interaction.runner.load_self_hashed",
        load,
    )
    with pytest.raises(ProtocolError, match="resume identity changed"):
        _prepare(paths, resume=True)


def test_identity_hashes_include_local_code_and_external_helpers():
    identity = _identity_hashes()
    assert (
        "scripts/experiments/wav2lip_reference_conditioning_interaction/runner.py"
        in identity["code_sha256"]
    )
    assert (
        "scripts/experiments/lrs3_real_video_local_timing/media.py"
        in identity["helper_sha256"]
    )
    assert (
        "openspec/changes/complete-wav2lip-reference-matched-control/design.md"
        in identity["spec_sha256"]
    )


def test_fixed_parity_fixture_outside_cohort_gets_explicit_group() -> None:
    assert (
        _parity_source_group(
            {"rows": [{"sample_id": "cohort", "source_group": "g"}]}, "fixed"
        )
        == "parity"
    )


def test_validator_rebuilds_score_matrix_from_worker_embeddings(tmp_path) -> None:
    visual_path, audio_path = tmp_path / "visual.npy", tmp_path / "audio.npy"
    visual = np.zeros((88, 1024), dtype=np.float32)
    audio = np.zeros_like(visual)
    np.save(visual_path, visual, allow_pickle=False)
    np.save(audio_path, audio, allow_pickle=False)
    worker_path = tmp_path / "worker.json"
    write_json(
        worker_path,
        {
            "visual": str(visual_path),
            "visual_sha256": file_sha256(visual_path),
            "audio_embedding": str(audio_path),
            "audio_embedding_sha256": file_sha256(audio_path),
        },
    )
    matrix_path = tmp_path / "matrix.npy"
    matrix = _full_matrix(visual, audio)
    np.save(matrix_path, matrix, allow_pickle=False)
    u = _full_metrics(matrix[np.asarray(runner_config.U_ROWS)])
    score = {
        "worker": str(worker_path),
        "matrix": str(matrix_path),
        "matrix_sha256": file_sha256(matrix_path),
        "U": u,
    }
    assert _score_metrics_from_raw(score, "fixture")["U"]["min_index"] == u["min_index"]
    np.save(matrix_path, matrix + 1.0, allow_pickle=False)
    with pytest.raises(ProtocolError, match="matrix hash mismatch"):
        _score_metrics_from_raw(score, "fixture")
