from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location(
    "prepare_mdc_english_alignment", SCRIPTS / "prepare_mdc_english_alignment.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_phone_parser_maps_visemes_and_silence(tmp_path: Path):
    tg = tmp_path / "sample.TextGrid"
    tg.write_text(
        '''File type = "ooTextFile"\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 0.5\n        intervals: size = 3\n        intervals [1]:\n            xmin = 0\n            xmax = 0.1\n            text = "sil"\n        intervals [2]:\n            xmin = 0.1\n            xmax = 0.3\n            text = "p"\n        intervals [3]:\n            xmin = 0.3\n            xmax = 0.5\n            text = "iy"\n''',
        encoding="utf-8",
    )
    tokens = module.parse_textgrid_phones(tg)
    assert tokens[0]["is_silence"] is True
    assert tokens[1]["phoneme"] == "p"
    assert tokens[1]["viseme"] == "pbmv"
    assert tokens[2]["viseme"] == "other"


def test_phone_parser_normalizes_noise_unknown_and_mixed_case(tmp_path: Path):
    tg = tmp_path / "labels.TextGrid"
    tg.write_text(
        '''item [1]:
 class = "IntervalTier"
 name = "phones"
 xmin = 0
 xmax = 0.4
 intervals: size = 4
 intervals [1]:
 xmin = 0
 xmax = 0.1
 text = "H#"
 intervals [2]:
 xmin = 0.1
 xmax = 0.2
 text = "<SIL>"
 intervals [3]:
 xmin = 0.2
 xmax = 0.3
 text = "<UNK>"
 intervals [4]:
 xmin = 0.3
 xmax = 0.4
 text = "p"
''',
        encoding="utf-8",
    )
    tokens = module.parse_textgrid_phones(tg)
    assert [token["viseme"] for token in tokens] == ["sil", "sil", "sil", "pbmv"]
    assert tokens[0]["is_silence"] is True
    assert tokens[2]["is_unknown"] is True


def test_validate_tokens_rejects_gaps_and_zero_duration_speech(tmp_path: Path):
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    for tokens, message in (
        ([{"token": "p", "start_s": 0.1, "end_s": 1.0, "is_non_speech": False}], "start at zero"),
        ([{"token": "p", "start_s": 0.0, "end_s": 0.0, "is_non_speech": False}], "zero-duration"),
    ):
        try:
            module.validate_tokens(tokens, wav)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid alignment should fail")


def test_validate_tokens_rejects_out_of_duration(tmp_path: Path):
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    tokens = [{
        "token": "p", "start_s": 0.0, "end_s": 2.0,
        "is_silence": False, "viseme": "pbmv",
    }]
    try:
        module.validate_tokens(tokens, wav)
    except ValueError as exc:
        assert "exceeds audio duration" in str(exc)
    else:
        raise AssertionError("out-of-duration alignment should fail")


def test_validate_tokens_rejects_trailing_uncovered_audio(tmp_path: Path):
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    tokens = [{
        "token": "p", "start_s": 0.0, "end_s": 0.9,
        "is_silence": False, "viseme": "pbmv",
    }]
    try:
        module.validate_tokens(tokens, wav)
    except ValueError as exc:
        assert "do not cover audio duration" in str(exc)
    else:
        raise AssertionError("trailing uncovered audio should fail")


def test_source_sample_id_must_match_pair_key(tmp_path: Path):
    wav = tmp_path / "natural.wav"
    sf.write(wav, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    audio_hash = module.sha256_file(wav)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"samples": [{
        "sample_id": "en_001", "language_code": "en",
        "file_path": "natural.wav", "sha256": audio_hash, "text": "test",
    }]}), encoding="utf-8")
    source_hash = module.sha256_file(source)
    pair = {
        "source_manifest": "source.json",
        "source_manifest_sha256": source_hash,
        "records": [{
            "sample_key": "en_002", "source_sample_id": "en_001",
            "source_audio_path": "natural.wav", "source_audio_sha256": audio_hash,
            "text": "test",
        }],
    }
    try:
        module._validate_source_provenance(tmp_path, pair)
    except ValueError as exc:
        assert "source_sample_id mismatch" in str(exc)
    else:
        raise AssertionError("source and pair IDs must agree")


def test_tts_provenance_requires_exact_pair_key_set(tmp_path: Path):
    pair_path = tmp_path / "pair.json"
    pair_payload = {
        "dataset": "mdc_tts",
        "records": [
            {"sample_key": "en_001", "paired_key": "en_001"},
            {"sample_key": "en_002", "paired_key": "en_002"},
        ],
    }
    pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")
    tts_payload = {
        "dataset": "mdc_tts", "language_code": "en", "provider": "mock_tts",
        "source_pair_manifest_sha256": module.sha256_file(pair_path),
    }
    try:
        module._validate_tts_provenance(
            tmp_path, pair_path, pair_payload, tts_payload,
            {"en_001": {"provider": "mock_tts", "paired_key": "en_001"}},
        )
    except ValueError as exc:
        assert "TTS/pair sample-key mismatch" in str(exc)
    else:
        raise AssertionError("TTS and pair key sets must agree")


def test_alignment_manifest_never_reuses_natural_tokens(tmp_path: Path):
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    natural = pair_dir / "natural.wav"
    tts = pair_dir / "tts.wav"
    sf.write(natural, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    sf.write(tts, np.full(36000, 0.1, dtype=np.float32), 24000, subtype="PCM_16")
    natural_sha256 = module.sha256_file(natural)
    tts_sha256 = module.sha256_file(tts)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({
            "samples": [{
                "sample_id": "en_001",
                "language_code": "en",
                "file_path": "pairs/natural.wav",
                "sha256": natural_sha256,
                "text": "test",
            }],
        }),
        encoding="utf-8",
    )
    source_manifest_sha256 = module.sha256_file(source_manifest)
    pair_payload = {
        "dataset": "mdc_tts",
        "language_code": "en",
        "license": "CC0-1.0",
        "source_manifest": "source.json",
        "source_manifest_sha256": source_manifest_sha256,
        "records": [{
            "sample_key": "en_001",
            "source_sample_id": "en_001",
            "source_audio_path": "pairs/natural.wav",
            "source_audio_sha256": natural_sha256,
            "paired_key": "en_001",
            "condition": "natural",
            "text": "test",
            "audio_path": "pairs/natural.wav",
            "speaker_id": "author:a",
            "canonical_qc": {"sha256": natural_sha256},
        }],
    }
    pair_manifest = tmp_path / "pair.json"
    pair_manifest.write_text(json.dumps(pair_payload), encoding="utf-8")
    tts_payload = {
        "dataset": "mdc_tts",
        "language_code": "en",
        "provider": "mock_tts",
        "source_pair_manifest": str(pair_manifest.relative_to(tmp_path)),
        "source_pair_manifest_sha256": module.sha256_file(pair_manifest),
        "results": {"en_001": {
            "sample_key": "en_001",
            "paired_key": "en_001",
            "condition": "tts",
            "status": "ok",
            "text": "test",
            "utterance_id": "mdc_tts/en_001/tts",
            "reference_audio": "pairs/natural.wav",
            "reference_sha256": natural_sha256,
            "generated_audio": "pairs/tts.wav",
            "provider": "mock_tts",
            "canonical_qc": {"sha256": tts_sha256},
        }},
    }
    tts_meta = tmp_path / "tts.json"
    tts_meta.write_text(json.dumps(tts_payload), encoding="utf-8")
    natural_tg = tmp_path / "nat"
    tts_tg = tmp_path / "tts"
    natural_tg.mkdir(); tts_tg.mkdir()
    for directory, end, phone in ((natural_tg, "1.0", "p"), (tts_tg, "1.5", "b")):
        (directory / "en_001.TextGrid").write_text(
            f'''item [1]:\n class = "IntervalTier"\n name = "phones"\n xmin = 0\n xmax = {end}\n intervals: size = 1\n intervals [1]:\n xmin = 0\n xmax = {end}\n text = "{phone}"\n''',
            encoding="utf-8",
        )
    result = module.build_alignment_manifest(
        tmp_path, pair_manifest, tts_meta, natural_tg, tts_tg,
        tmp_path / "alignment.json", ["en_001"],
    )
    assert result["complete"] is True
    rows = {row["condition"]: row for row in result["records"]}
    assert rows["natural"]["tokens"][0]["token"] == "p"
    assert rows["tts"]["tokens"][0]["token"] == "b"
    assert rows["natural"]["tokens"] is not rows["tts"]["tokens"]
    assert rows["natural"]["sample_id"] == "en_001"
    assert rows["natural"]["tts_provider"] is None
    assert rows["tts"]["sample_id"] == "en_001"
    assert rows["tts"]["tts_provider"] == "mock_tts"


def test_sample_ids_require_exact_mdc_inventory():
    assert module._validate_sample_ids(["en_001", "en_050"]) == ["en_001", "en_050"]
    for sample_ids in (["en_000"], ["en_051"], ["en_bad"], ["en_001", "en_001"]):
        try:
            module._validate_sample_ids(sample_ids)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MDC sample IDs should fail")


def test_tts_result_mapping_rejects_outer_key_mismatch(tmp_path: Path):
    path = tmp_path / "tts.json"
    path.write_text(
        json.dumps({"results": {"en_001": {"sample_key": "en_002", "paired_key": "en_002"}}}),
        encoding="utf-8",
    )
    try:
        module._load_tts_results(path)
    except ValueError as exc:
        assert "key mismatch" in str(exc)
    else:
        raise AssertionError("outer and internal TTS keys must agree")


def test_mfa_staging_arm_must_not_be_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    arm = tmp_path / "arm"
    arm.symlink_to(target, target_is_directory=True)
    try:
        module._clear_mfa_staging(arm)
    except ValueError as exc:
        assert "must not be a symlink" in str(exc)
    else:
        raise AssertionError("symlinked MFA staging arms must fail")
