from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("aishell3_adapter", SCRIPTS / "39_prepare_aishell3_manifest.py")
adapter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(adapter)

from manifest import load_manifest, validate_manifest, write_manifest
from tfg_feature_common import embedding_file_stem


def _wav(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * sample_rate)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "aishell3"
    for split, speakers in {"train": ("S0001", "S0002"), "dev": ("S0003",)}.items():
        lines = []
        for speaker in speakers:
            for index in (1, 2):
                utt = f"BAC009{speaker}W{index:04d}"
                _wav(root / split / "wav" / speaker / f"{utt}.wav")
                lines.append(f"{utt} 中文测试 {index}")
        (root / split / "content.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / split / "content.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_build_manifest_is_deterministic_and_speaker_stratified(tmp_path):
    root = _dataset(tmp_path)
    first = adapter.build_manifest(root, split="train", speaker_limit=2, max_utterances=3, seed=7)
    second = adapter.build_manifest(root, split="train", speaker_limit=2, max_utterances=3, seed=7)
    assert [item["utterance_id"] for item in first] == [item["utterance_id"] for item in second]
    assert len({item["speaker_id"] for item in first}) == 2
    assert all(item["representation_only"] for item in first)
    assert all(item["paired_key"] is None for item in first)
    assert all(item["alignment_source"] == "missing" for item in first)


def test_manifest_writes_and_loads_versioned_records(tmp_path):
    root = _dataset(tmp_path)
    records = adapter.build_manifest(root, split="train", max_utterances=1)
    path = tmp_path / "manifest.json"
    write_manifest(path, records)
    loaded = load_manifest(path)
    assert loaded[0]["dataset"] == "aishell3"
    assert loaded[0]["sample_id"].startswith("train/")
    assert embedding_file_stem(loaded[0], "hubert").startswith("aishell3_train_")


def test_uniform_fallback_is_explicit(tmp_path):
    root = _dataset(tmp_path)
    records = adapter.build_manifest(root, split="train", max_utterances=1, uniform_fallback=True)
    assert records[0]["alignment_source"] == "uniform_fallback"
    assert records[0]["alignment_confidence"] == 0.0
    assert records[0]["tokens"]


def test_missing_audio_fails_by_default(tmp_path):
    root = _dataset(tmp_path)
    missing = next((root / "train" / "wav").rglob("*.wav"))
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="audio not found"):
        adapter.build_manifest(root, split="train")


def test_duplicate_labels_fail(tmp_path):
    root = _dataset(tmp_path)
    labels = root / "train" / "content.txt"
    labels.write_text(labels.read_text(encoding="utf-8") + labels.read_text(encoding="utf-8").splitlines()[0] + " duplicate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate utterance label"):
        adapter.build_manifest(root, split="train")


def test_representation_only_none_pair_is_validated():
    record = {
        "dataset": "aishell3",
        "utterance_id": "train/utt-1",
        "speaker_id": "S1",
        "paired_key": None,
        "condition": "natural",
        "transcript": "你好",
        "audio_path": "audio.wav",
        "split": "train",
        "alignment_source": "missing",
        "license": "Apache-2.0",
        "representation_only": True,
    }
    assert validate_manifest([record])[0]["paired_key"] is None
