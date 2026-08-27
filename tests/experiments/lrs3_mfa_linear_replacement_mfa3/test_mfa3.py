from __future__ import annotations

from pathlib import Path

from scripts.experiments.lrs3_mfa_linear_replacement.protocol import file_sha256
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.protocol import (
    _contains_exact_model,
    _contains_exact_version,
)
import scripts.experiments.lrs3_mfa_linear_replacement_mfa3.run_stage01_mfa3 as mfa3_stage01
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.mfa3_alignment import (
    MFA_ACOUSTIC_MODEL,
    MFA_DICTIONARY,
    NORMALIZATION_POLICY_ID,
    build_mfa3_command,
    normalize_mfa3_transcript,
    normalization_events,
    run_mfa3_alignment,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.run_stage01_mfa3 import (
    validate_candidate_result_ids,
    validate_mfa_contract,
    validate_stage00_binding,
)


def test_mfa3_normalization_expands_numbers_and_removes_registered_event() -> None:
    text = "3 QUESTIONS THE 1ST QUESTION IN 2011 {LG}"
    normalized = normalize_mfa3_transcript(text)
    assert normalized == "THREE QUESTIONS THE FIRST QUESTION IN TWO THOUSAND ELEVEN"
    events = normalization_events(text)
    assert [event["kind"] for event in events] == ["number", "ordinal", "number", "nonlexical_event_removed"]
    assert NORMALIZATION_POLICY_ID.endswith("v1")


def test_mfa3_normalization_rejects_invalid_ordinal() -> None:
    for token in ("0TH", "11ST", "12ND", "13RD"):
        try:
            normalize_mfa3_transcript(token)
        except ValueError as exc:
            assert "ordinal" in str(exc)
        else:
            raise AssertionError(f"invalid ordinal accepted: {token}")


    assert normalize_mfa3_transcript("WE'RE WHAT'S ACR'S") == "WE'RE WHAT'S ACR'S"
    try:
        normalize_mfa3_transcript("HELLO {UNKNOWN}")
    except ValueError as exc:
        assert "unsupported nonlexical" in str(exc)
    else:
        raise AssertionError("unsupported annotation was accepted")


def test_mfa3_command_binds_english_models() -> None:
    assert build_mfa3_command("input", "output", mfa_executable="mfa") == [
        "mfa", "align", "--clean", "--overwrite", "input",
        MFA_DICTIONARY, MFA_ACOUSTIC_MODEL, "output",
    ]


def test_mfa3_metadata_matching_is_exact() -> None:
    assert _contains_exact_version("3.4.1", "3.4.1")
    assert not _contains_exact_version("3.4.10", "3.4.1")
    assert _contains_exact_model("['english_mfa']", "english_mfa")
    assert not _contains_exact_model("['english_mfa_old']", "english_mfa")


def test_mfa3_alignment_binds_root_environment(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    mfa_executable = tmp_path / "mfa3/bin/mfa"
    mfa_executable.parent.mkdir(parents=True)
    mfa_executable.write_text("mfa", encoding="utf-8")
    sample_ids = [f"sample_{index:02d}" for index in range(24)]
    seen = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_runner(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        for sample_id in sample_ids:
            (output_dir / f"{sample_id}.TextGrid").write_text("ok", encoding="utf-8")
        return Result()

    result = run_mfa3_alignment(
        "input",
        output_dir,
        mfa_executable=str(mfa_executable),
        mfa_root_dir=tmp_path / "mfa-root",
        expected_sample_ids=sample_ids,
        runner=fake_runner,
    )
    assert result["returncode"] == 0
    assert seen["command"][5:7] == [MFA_DICTIONARY, MFA_ACOUSTIC_MODEL]
    assert seen["env"]["MFA_ROOT_DIR"] == str((tmp_path / "mfa-root").resolve())
    assert seen["env"]["PATH"].split(":", 1)[0] == str(mfa_executable.parent.resolve())


def test_mfa3_contract_binds_files_and_ids(tmp_path) -> None:
    executable = tmp_path / "mfa"
    root = tmp_path / "root"
    dictionary = root / "pretrained_models/dictionary/english_mfa.dict"
    acoustic = root / "pretrained_models/acoustic/english_mfa.zip"
    config = root / "global_config.yaml"
    for path in (executable, dictionary, acoustic, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))
    contract = {
        "version": "3.4.1",
        "root_dir": str(root.resolve()),
        "root_config": {"path": str(config), "sha256": file_sha256(config)},
        "executable": {"path": str(executable), "sha256": file_sha256(executable)},
        "dictionary": {"name": MFA_DICTIONARY, "path": str(dictionary), "sha256": file_sha256(dictionary)},
        "acoustic_model": {"name": MFA_ACOUSTIC_MODEL, "path": str(acoustic), "sha256": file_sha256(acoustic)},
    }
    validate_mfa_contract({"mfa_contract": contract}, mfa_executable=str(executable), mfa_root_dir=root)
    validate_candidate_result_ids([{"sample_id": "a"}, {"sample_id": "b"}], ["a", "b"])
    try:
        validate_candidate_result_ids([{"sample_id": "b"}, {"sample_id": "a"}], ["a", "b"])
    except ValueError as exc:
        assert "ordered frozen cohort" in str(exc)
    else:
        raise AssertionError("wrong candidate order was accepted")


    original_repo = mfa3_stage01.REPO
    mfa3_stage01.REPO = tmp_path
    try:
        stage00_dir = tmp_path / "runs/lrs3_mfa_linear_replacement_mfa3_20260825/00_protocol_lock_mfa3_retry2"
        stage00_dir.mkdir(parents=True)
        manifest = stage00_dir / "manifest.json"
        manifest.write_text('{"stage_id":"00_protocol_lock_mfa3_retry2"}\n', encoding="utf-8")
        (stage00_dir / "manifest.sha256").write_text(file_sha256(manifest) + "\n", encoding="utf-8")
        validate_stage00_binding(manifest, "01_candidate_audio_mfa3_retry2", {"stage_id": "00_protocol_lock_mfa3_retry2"})
        manifest.write_text('{"stage_id":"00_protocol_lock_mfa3_retry2","changed":true}\n', encoding="utf-8")
        try:
            validate_stage00_binding(manifest, "01_candidate_audio_mfa3_retry2", {"stage_id": "00_protocol_lock_mfa3_retry2"})
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("modified Stage00 manifest was accepted")
    finally:
        mfa3_stage01.REPO = original_repo
