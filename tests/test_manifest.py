from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from manifest import check_speaker_leakage, load_manifest, paired_keys, validate_manifest, write_manifest


def _record(**overrides):
    record = {
        "dataset": "demo",
        "utterance_id": "utt-1",
        "speaker_id": "spk-1",
        "paired_key": "pair-1",
        "condition": "natural",
        "transcript": "你好",
        "audio_path": "audio.wav",
        "split": "train",
        "alignment_source": "mfa",
        "license": "Apache-2.0",
    }
    record.update(overrides)
    return record


def test_validate_manifest_rejects_duplicate_condition_key():
    with pytest.raises(ValueError, match="duplicate key"):
        validate_manifest([_record(), _record(utterance_id="utt-1")])


def test_validate_manifest_rejects_unknown_alignment_source():
    with pytest.raises(ValueError, match="alignment_source"):
        validate_manifest([_record(alignment_source="guess")])


def test_write_and_load_versioned_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, [_record()])
    payload = path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in payload
    assert load_manifest(path)[0]["utterance_id"] == "utt-1"


def test_pairing_and_speaker_leakage_checks():
    records = [
        _record(),
        _record(
            utterance_id="utt-1-tts",
            condition="tts",
            audio_path="tts.wav",
        ),
        _record(
            utterance_id="utt-2",
            paired_key="pair-2",
            speaker_id="spk-1",
            split="test",
        ),
    ]
    assert paired_keys(records) == {"pair-1"}
    assert check_speaker_leakage(records) == ["spk-1"]


def test_load_rejects_wrong_schema(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 99, "records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(path)
