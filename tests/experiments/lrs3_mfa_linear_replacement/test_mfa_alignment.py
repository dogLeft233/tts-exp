from __future__ import annotations

import hashlib
import wave

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    AlignmentError,
    build_frame_mapping,
    build_mfa_command,
    extend_final_token_for_feature_tail,
    frame_owners,
    parse_textgrid,
    parsed_token_hash,
    prepare_mfa_corpus,
    run_mfa_alignment,
)


def _tokens(labels):
    rows = []
    start = 0.0
    for label, duration in labels:
        rows.append({"token": label, "label": label, "start_s": start, "end_s": start + duration})
        start += duration
    return rows


def test_parse_textgrid_preserves_silence_and_rejects_unknown(tmp_path) -> None:
    path = tmp_path / "sample.TextGrid"
    path.write_text(
        'name = "phones"\n'
        'intervals [1]: xmin = 0 xmax = 0.02 text = ""\n'
        'intervals [2]: xmin = 0.02 xmax = 0.06 text = "AA"\n'
        'intervals [3]: xmin = 0.06 xmax = 0.08 text = "sil"\n',
        encoding="utf-8",
    )
    tokens = parse_textgrid(path)
    assert [row["silence"] for row in tokens] == [True, False, True]
    assert parsed_token_hash(tokens)
    path.write_text(
        'name = "phones"\nintervals [1]: xmin = 0 xmax = 0.08 text = "spn"\n',
        encoding="utf-8",
    )
    with pytest.raises(AlignmentError, match="unknown/OOV"):
        parse_textgrid(path)


def test_frame_owners_requires_complete_natural_clock() -> None:
    tokens = _tokens([("sil", 0.02), ("AA", 0.04), ("sil", 0.02)])
    owners = frame_owners(4, tokens)
    assert [owner.label for owner in owners] == ["sil", "aa", "aa", "sil"]
    with pytest.raises(AlignmentError, match="no phone owner"):
        frame_owners(5, tokens)


def test_unmatched_silence_uses_only_tts_silence() -> None:
    natural = _tokens([("sp", 0.02), ("AA", 0.06)])
    tts = _tokens([("sil", 0.02), ("AA", 0.06)])
    mapping, stats = build_frame_mapping(4, 4, natural, tts)
    assert mapping[0].mapping_type == "silence_fallback"
    assert stats["speech_fallback_frames"] == 0
    assert stats["silence_fallback_frames"] >= 1


def test_ordered_matching_prefers_speech_over_silence_insertions() -> None:
    natural = _tokens([("f", 0.02), ("sil", 0.02), ("a", 0.02), ("sil", 0.02), ("aj", 0.02)])
    tts = _tokens([("f", 0.02), ("a", 0.02), ("aj", 0.02), ("sil", 0.02)])
    mapping, stats = build_frame_mapping(5, 4, natural, tts)
    assert stats["speech_fallback_frames"] == 0
    assert mapping[2].mapping_type == "matched_phone"


    natural = _tokens([("AA", 0.04), ("B", 0.04)])
    tts = _tokens([("AA", 0.08)])
    with pytest.raises(AlignmentError, match="unmatched natural speech"):
        build_frame_mapping(4, 4, natural, tts)


def test_tail_extension_is_bounded_and_does_not_change_internal_spans() -> None:
    tokens = _tokens([("AA", 0.04), ("sil", 0.04)])
    extended, amount = extend_final_token_for_feature_tail(tokens, 5)
    assert amount == pytest.approx(0.01)
    assert extended[0]["end_s"] == tokens[0]["end_s"]
    with pytest.raises(AlignmentError, match="exceeds bound"):
        extend_final_token_for_feature_tail(tokens, 20)


def test_mfa_command_is_argument_vector_and_corpus_is_side_specific(tmp_path) -> None:
    command = build_mfa_command("natural", "natural_textgrids", mfa_executable="mfa")
    assert command == ["mfa", "align", "--clean", "--overwrite", "natural", "english_us_mfa", "english_mfa", "natural_textgrids"]
    wav = tmp_path / "input.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    records = [
        {"sample_id": f"sample_{index:02d}", "natural_audio": str(wav), "natural_audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "transcript": "hello   world"}
        for index in range(24)
    ]
    entries = prepare_mfa_corpus(records, tmp_path / "corpus", audio_key="natural_audio", expected_records=records, audio_hash_key="natural_audio_sha256")
    assert entries[0]["transcript"] == "HELLO WORLD"
    assert (tmp_path / "corpus" / "sample_00.lab").read_text() == "HELLO WORLD\n"

    output_dir = tmp_path / "mfa-output"
    output_dir.mkdir()

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_runner(*args, **kwargs):
        for index in range(24):
            (output_dir / f"sample_{index:02d}.TextGrid").write_text("", encoding="utf-8")
        return Result()

    result = run_mfa_alignment("in", output_dir, expected_sample_ids=[f"sample_{index:02d}" for index in range(24)], runner=fake_runner)
    assert result["returncode"] == 0
