from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.experiments import check_wav2lip_noise_cross as checker
from scripts.experiments.wav2lip_noise_cross_metrics import canonical_hash


def test_independent_matrix_rebuild_matches_known_float32_contract() -> None:
    visual = np.zeros((32, 1024), dtype=np.float32)
    audio = np.zeros((32, 1024), dtype=np.float32)
    audio[:, 0] = np.arange(32, dtype=np.float32)
    value = checker._matrix_from_embeddings(visual, audio)
    assert value.shape == (32, 31)
    assert np.isfinite(value).all()
    assert value[15, 14] < value[15, 15]


def test_self_hash_reader_rejects_tampered_artifact(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    body = {"schema_version": 1, "value": 3}
    path.write_text(json.dumps({**body, "artifact_sha256": canonical_hash(body)}), encoding="utf-8")
    assert checker._read_self(path)["value"] == 3
    path.write_text(json.dumps({"schema_version": 1, "value": 4, "artifact_sha256": canonical_hash(body)}), encoding="utf-8")
    with pytest.raises(Exception):
        checker._read_self(path)


def test_close_json_checks_nested_float_values() -> None:
    assert checker._close_json({"x": [1.0, 2.0]}, {"x": [1.0, 2.0 + 1e-9]})
    assert not checker._close_json({"x": [1.0]}, {"x": [1.1]})
