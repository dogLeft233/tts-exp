from __future__ import annotations

from pathlib import Path

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    CANONICAL_FACE_PREFLIGHT_PATH,
    CANONICAL_STAGE00_ID,
    CANONICAL_STAGE00_PATH,
    CANONICAL_SYNCNET_PREFLIGHT_PATH,
    MFA_ROOT_DIR,
    build_exploratory_pool,
    build_mfa3_command,
    dictionary_missing_tokens,
    load_json,
    run_mfa3_alignment,
    validate_stage00,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_strict_replacement import (
    CELL_SPECS,
    PRIMARY_CELLS,
    _asset,
    _failure,
    build_protocol_manifest,
)


def test_dictionary_prefilter_uses_mfa3_normalization() -> None:
    assert dictionary_missing_tokens("3RD FISH", {"THIRD", "FISH"}) == []
    assert dictionary_missing_tokens("3RD FISH", {"FISH"}) == ["THIRD"]


def test_exploratory_pool_is_disjoint_and_large_enough() -> None:
    pool = build_exploratory_pool()
    assert pool["counts"]["fit_pool_records"] == 271
    assert pool["counts"]["eligible_records"] >= 24
    ids = [row["sample_id"] for row in pool["records"]]
    assert len(ids) == len(set(ids))
    assert pool["selection"]["score_based_selection"] is False
    assert pool["selection"]["alignment_result_based_selection"] is False


def test_candidate_stage00_binding_is_canonical(tmp_path: Path) -> None:
    stage00 = load_json(CANONICAL_STAGE00_PATH)
    validate_stage00(
        CANONICAL_STAGE00_PATH,
        stage00,
        expected_stage_id=CANONICAL_STAGE00_ID,
        expected_path=CANONICAL_STAGE00_PATH,
    )
    try:
        validate_stage00(
            tmp_path / "manifest.json",
            stage00,
            expected_stage_id=CANONICAL_STAGE00_ID,
            expected_path=CANONICAL_STAGE00_PATH,
        )
    except ValueError as exc:
        assert "canonical candidate input" in str(exc)
    else:
        raise AssertionError("non-canonical Stage00 path was accepted")


def test_exploratory_mfa_command_binds_dictionary_and_model(tmp_path: Path) -> None:
    command = build_mfa3_command(tmp_path / "input", tmp_path / "output")
    assert command[1:3] == ["align", "--clean"]
    assert command[-3:-1] == ["english_mfa", "english_mfa"]


def test_strict_replacement_protocol_binds_four_cells() -> None:
    manifest = build_protocol_manifest(CANONICAL_STAGE00_PATH, CANONICAL_STAGE00_PATH.parent.parent / "02_candidate_audio_retry4/candidate_manifest.json")
    assert manifest["cohort"]["record_count"] == 146
    assert set(manifest["cells"]) == {"G_N_E_N", "G_M_E_N", "G_N_E_M", "G_M_E_M"}
    assert tuple(manifest["primary_cells"]) == PRIMARY_CELLS
    assert manifest["cells"]["G_M_E_N"]["role"] == "authoritative_target"
    assert CELL_SPECS["G_N_E_N"]["role"] == "authoritative_baseline"


def test_face_ready_strict_protocol_uses_frozen_engineering_subset() -> None:
    manifest = build_protocol_manifest(
        CANONICAL_STAGE00_PATH,
        CANONICAL_STAGE00_PATH.parent.parent / "02_candidate_audio_retry4/candidate_manifest.json",
        CANONICAL_FACE_PREFLIGHT_PATH,
    )
    assert manifest["cohort"]["record_count"] == 135
    assert manifest["selection"]["face_preflight_used"] is True
    assert manifest["selection"]["score_based_selection"] is False
    assert manifest["parents"]["face_preflight"]["path"] == str(CANONICAL_FACE_PREFLIGHT_PATH.resolve())


def test_syncnet_preflight_binds_structural_subset() -> None:
    manifest = build_protocol_manifest(
        CANONICAL_STAGE00_PATH,
        CANONICAL_STAGE00_PATH.parent.parent / "02_candidate_audio_retry4/candidate_manifest.json",
        CANONICAL_FACE_PREFLIGHT_PATH,
        CANONICAL_SYNCNET_PREFLIGHT_PATH,
    )
    assert manifest["cohort"]["record_count"] == 133
    assert manifest["selection"]["syncnet_preflight_used"] is True
    assert manifest["selection"]["score_based_selection"] is False
    assert manifest["parents"]["syncnet_preflight"]["path"] == str(CANONICAL_SYNCNET_PREFLIGHT_PATH.resolve())


def test_exploratory_alignment_binds_root_and_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    seen = {}

    def runner(command, *, check, capture_output, text, env):
        seen.update({"command": command, "env": env, "check": check})
        output_dir.mkdir(exist_ok=True)
        (output_dir / "sample.TextGrid").write_text("", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    result = run_mfa3_alignment(
        input_dir,
        output_dir,
        expected_sample_ids=["sample"],
        runner=runner,
    )
    assert result["textgrid_count"] == 1
    assert seen["check"] is False
    assert seen["env"]["MFA_ROOT_DIR"] == str(MFA_ROOT_DIR.resolve())
    assert str(Path(seen["command"][0]).parent) in seen["env"]["PATH"].split(":"), seen


def test_strict_failure_writes_artifacts_before_protocol_manifest(tmp_path: Path) -> None:
    summary = _failure(tmp_path, "render failed", manifest={"status": "locked"})
    assert summary["engineering_decision"] == "BLOCKED"
    assert (tmp_path / "failure.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "decision.json").is_file()


def test_asset_binding_preserves_symlink_invocation_path(tmp_path: Path) -> None:
    link = tmp_path / "python"
    link.symlink_to(Path(__file__))
    binding = _asset(link)
    assert binding["path"] == str(link.absolute())
