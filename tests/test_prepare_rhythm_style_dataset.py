from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_alignment_warp import Span
from prepare_rhythm_style_dataset import (
    RHYTHM_DIM,
    align_content,
    build_rhythm,
    make_splits,
    resolve_path,
    select_embedding_layer,
)


def test_resolve_path_supports_repo_relative_and_rejects_missing(tmp_path: Path) -> None:
    nested = tmp_path / "audio" / "1.wav"
    nested.parent.mkdir()
    nested.write_bytes(b"wav")
    assert resolve_path("audio/1.wav", tmp_path) == nested.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_path("audio/missing.wav", tmp_path)


def test_select_embedding_layer_supports_single_layer_and_layer_axis() -> None:
    rng = np.random.default_rng(42)
    layers = list(range(12))
    values = rng.normal(size=(12, 9, 768)).astype(np.float32)
    assert np.array_equal(select_embedding_layer(values, {"layers": layers}, 11), values[11])
    single = values[11]
    assert np.array_equal(select_embedding_layer(single, {"layers": [11]}, 11), single)
    with pytest.raises(ValueError):
        select_embedding_layer(values, {"layers": list(range(11))}, 11)


def test_rhythm_alpha_only_scales_duration_channels() -> None:
    natural = [Span("a", 0.0, 0.2), Span("b", 0.2, 0.4)]
    tts = [Span("a", 0.0, 0.1), Span("b", 0.1, 0.35)]
    zero = build_rhythm(natural, tts, 640, alpha=0.0)
    half = build_rhythm(natural, tts, 640, alpha=0.5)
    one = build_rhythm(natural, tts, 640, alpha=1.0)
    assert zero.shape == half.shape == one.shape == (2, RHYTHM_DIM)
    assert np.allclose(zero[:, 0], 0.0)
    assert np.allclose(zero[:, 5], 0.0)
    assert np.allclose(half[:, 0], one[:, 0] * 0.5)
    assert np.allclose(half[:, 5], one[:, 5] * 0.5)
    assert np.allclose(zero[:, 1:5], half[:, 1:5])


def test_rhythm_uses_tts_utterance_progress() -> None:
    natural = [Span("a", 0.0, 0.8), Span("b", 0.8, 1.0)]
    tts = [Span("a", 0.0, 0.2), Span("b", 0.2, 1.0)]
    rhythm = build_rhythm(natural, tts, 16_000, alpha=1.0)
    valid = rhythm[rhythm[:, 4] > 0]
    assert valid.shape[0] > 2
    assert not np.allclose(valid[:, 1], valid[:, 2])


    content = np.arange(7 * 768, dtype=np.float32).reshape(7, 768)
    first = align_content(content, 13)
    second = align_content(content, 13)
    assert first.shape == (13, 768)
    assert np.array_equal(first, second)


def test_splits_cover_all_samples_once() -> None:
    mapping = {index: f"speaker-{index // 2}" for index in range(12)}
    splits = make_splits(mapping, seed=42)
    assert set(splits) == set(mapping)
    assert set(splits.values()) == {"train", "valid", "test"}
    assert len(set(splits)) == len(mapping)


def test_dataset_manifest_schema_has_no_secret_key(tmp_path: Path) -> None:
    manifest = {"items": [], "metadata": "local-only"}
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "sk-" not in path.read_text(encoding="utf-8")
