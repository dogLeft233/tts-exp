from pathlib import Path

import pytest

from scripts.experiments.asr_sync_error_correlation.config import frozen_config, validate_decoder_options, validate_frozen_config
from scripts.experiments.asr_sync_error_correlation.io import canonical_hash, file_sha256
from scripts.experiments.asr_sync_error_correlation.preflight import validate_model_lock
from scripts.experiments.asr_sync_error_correlation.protocol import validate_manifest_rows


def test_default_config_is_frozen(tmp_path: Path):
    config = frozen_config(tmp_path)
    validate_frozen_config(config, tmp_path)
    assert config["arms"] == ["natural", "tts"]
    assert config["analysis"]["bootstrap_draws"] == 10000


@pytest.mark.parametrize("key,value", [("beam_search", True), ("language_model", "lm"), ("lexicon", "lexicon.txt"), ("spell_correction", True), ("reference_conditioned", True)])
def test_decoder_correction_is_rejected(key, value):
    with pytest.raises(ValueError):
        validate_decoder_options({key: value})


def test_manifest_validator_requires_exact_paired_cohort():
    rows = []
    for index in range(24):
        common = {"sample_id": f"s{index}", "source_group": f"g{index}", "split": "fit", "video_sha256": f"v{index}"}
        for arm in ("natural", "tts"):
            rows.append({**common, "arm": arm, "arm_key": f"s{index}:{arm}"})
    validate_manifest_rows(rows)
    with pytest.raises(ValueError):
        validate_manifest_rows(rows[:-1])


def test_model_lock_rejects_mutation_and_main_revision(tmp_path: Path):
    blob = tmp_path / "weights.bin"
    blob.write_bytes(b"weights")
    files = [{"path": "weights.bin", "size": blob.stat().st_size, "sha256": file_sha256(blob)}]
    lock = {
        "model_id": "facebook/wav2vec2-large-960h-lv60-self",
        "revision": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119",
        "snapshot_path": str(tmp_path),
        "files": files,
        "snapshot_hash": canonical_hash(files),
        "tokenizer_vocabulary_hash": "vocab-hash",
    }
    assert validate_model_lock(lock)
    blob.write_bytes(b"changed")
    assert not validate_model_lock(lock)
    lock["revision"] = "main"
    assert not validate_model_lock(lock)


def test_scientific_override_is_rejected(tmp_path: Path):
    config = frozen_config(tmp_path)
    config["analysis"]["low_score_k"] = 2.0
    with pytest.raises(ValueError):
        validate_frozen_config(config, tmp_path)
