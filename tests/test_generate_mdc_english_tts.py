from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location(
    "generate_mdc_english_tts", SCRIPTS / "generate_mdc_english_tts.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MockProvider:
    name = "mock_tts"

    def generate_voice_clone(self, *, text, ref_audio_path, ref_text, language):
        assert text == ref_text
        assert language == "English"
        assert ref_audio_path.exists()
        return type(
            "Result",
            (),
            {
                "audio": np.full(16000, 0.1, dtype=np.float32),
                "sample_rate": 16000,
                "backend_meta": {"backend": self.name, "model_id": "mock-1"},
            },
        )()


def _pair_manifest(tmp_path: Path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    pair_dir = tmp_path / "pairs"
    natural_dir = pair_dir / "natural"
    natural_dir.mkdir(parents=True)
    natural = natural_dir / "en_001.wav"
    sf.write(natural, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = pair_dir / "pair_manifest.json"
    manifest.write_text(
        json.dumps({
            "records": [{
                "sample_key": "en_001",
                "paired_key": "en_001",
                "text": "A short test sentence.",
                "audio_path": str(natural.relative_to(tmp_path)),
            }],
        }),
        encoding="utf-8",
    )
    return manifest


def test_generate_tts_writes_stable_pair_record(tmp_path: Path):
    pair_manifest = _pair_manifest(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  language: English\n  seed: 42\n", encoding="utf-8")
    result = module.generate_tts(
        repo=tmp_path,
        pair_manifest_path=pair_manifest,
        run_id="tts_test",
        config_path=config,
        provider=MockProvider(),
        provider_cfg={"language": "English", "seed": 42, "retry": 0},
        sample_ids=["en_001"],
    )
    assert result["complete"] is True
    record = result["results"]["en_001"]
    assert record["paired_key"] == "en_001"
    assert record["condition"] == "tts"
    assert record["utterance_id"] == "mdc_tts/en_001/tts"
    assert record["provider"] == "mock_tts"
    assert record["reference_role"] == "paired_natural_audio"
    assert record["reference_sample_key"] == "en_001"
    assert record["reference_is_natural_arm"] is True
    assert record["voice_identity_relation"] == "reference_conditioned_not_independently_verified"
    assert result["source_pair_manifest"] == str(pair_manifest.relative_to(tmp_path))
    assert result["source_pair_manifest_sha256"] == module.sha256_file(pair_manifest)
    assert record["canonical_qc"]["sample_rate_hz"] == 16000
    assert Path(tmp_path / record["generated_audio"]).is_file()


def test_generate_tts_requires_complete_default_set(tmp_path: Path):
    pair_manifest = _pair_manifest(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  language: English\n", encoding="utf-8")
    try:
        module.generate_tts(
            repo=tmp_path,
            pair_manifest_path=pair_manifest,
            run_id="tts_test",
            config_path=config,
            provider=MockProvider(),
            provider_cfg={"language": "English"},
        )
    except ValueError as exc:
        assert "pair records missing" in str(exc)
    else:
        raise AssertionError("default incomplete run should fail")
