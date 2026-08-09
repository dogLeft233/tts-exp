from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_multiset_pair_manifest.py"
spec = importlib.util.spec_from_file_location("prepare_multiset_pair_manifest", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _record(key: str) -> dict:
    return {
        "paired_key": key,
        "text": "hello",
        "natural_audio_path": "natural.wav",
        "tts_audio_path": "tts.wav",
        "source_video_path": "source.png",
        "natural": {"sha256": "a" * 64},
        "tts": {"sha256": "b" * 64},
    }


def test_build_heldout_preflight_excludes_training_pairs() -> None:
    result = module.build_heldout_preflight(
        [_record("train"), _record("held")],
        training_pair_ids={"train"},
        adapter_hash="c" * 64,
        ditto_dependency="ditto@abc",
        syncnet_dependency="syncnet@def",
    )
    assert result["split_classification"] == "heldout_not_executed"
    assert result["heldout_pair_ids"] == ["held"]
    assert result["evaluation_protocol"]["conditions"] == ["ordinary", "identity", "adapted", "random", "raw_tts"]


def test_validate_heldout_manifest_rejects_overlap_and_missing_provenance() -> None:
    manifest = {
        "records": [_record("held")],
        "evaluation_protocol": {"conditions": ["ordinary", "identity", "adapted", "random", "raw_tts"]},
    }
    with pytest.raises(ValueError, match="overlap"):
        module.validate_heldout_manifest(manifest, training_pair_ids={"held"}, heldout_pair_ids={"held"})
    bad = {**manifest, "records": [{**_record("held"), "split": "heldout", "natural": {}}]}
    with pytest.raises(ValueError, match="provenance"):
        module.validate_heldout_manifest(bad, training_pair_ids=set(), heldout_pair_ids={"held"})
