#!/usr/bin/env python3
"""Tests for strict natural-only two-stage waveform export."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import export_feature_targeted_waveform as exporter  # noqa: E402
from prepare_two_stage_hubert_manifest import file_sha256  # noqa: E402


def _export_list(path: Path, item: dict[str, object]) -> Path:
    source = path / "export_list.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "two_stage_natural_audio_export",
                "items": [item],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return source


def test_export_list_rejects_tts_fields_and_duplicate_keys(tmp_path: Path) -> None:
    audio = tmp_path / "natural.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000)
    item = {
        "paired_key": "item",
        "natural_path": str(audio),
        "natural_sha256": file_sha256(audio),
        "tts_path": "forbidden.wav",
    }
    with pytest.raises(ValueError, match="only paired_key"):
        exporter.load_export_items(_export_list(tmp_path, item))

    duplicate = _export_list(
        tmp_path,
        {
            "paired_key": "item",
            "natural_path": str(audio),
            "natural_sha256": file_sha256(audio),
        },
    )
    payload = json.loads(duplicate.read_text(encoding="utf-8"))
    payload["items"].append(dict(payload["items"][0]))
    duplicate.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        exporter.load_export_items(duplicate)


def test_export_writes_only_verified_natural_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "natural.wav"
    values = 0.3 * np.sin(np.linspace(0.0, 2.0, 1600, dtype=np.float32))
    sf.write(audio, values, 16_000)
    export_list = _export_list(
        tmp_path,
        {
            "paired_key": "item",
            "natural_path": str(audio),
            "natural_sha256": file_sha256(audio),
        },
    )
    stage1 = tmp_path / "best.pt"
    stage2 = tmp_path / "stage2.pt"
    stage1.write_bytes(b"stage1")
    stage2.write_bytes(b"stage2")

    class FakeRenderer:
        interface = {"hidden_size": 768, "sample_rate": 16_000}
        waveform_model = SimpleNamespace(residual_scale=0.05)

        def eval(self):
            return self

        def __call__(self, natural: torch.Tensor, *, compute_enhanced_features: bool):
            assert compute_enhanced_features is False
            return {"waveform": natural + 0.001}

    monkeypatch.setattr(
        exporter,
        "_load_renderer",
        lambda *args, **kwargs: (FakeRenderer(), {"selection": {"metric": "valid_realize_combined"}}),
    )
    output = tmp_path / "exported"
    payload = exporter.export_two_stage_natural_audio(export_list, stage1, stage2, output, device="cpu")

    assert payload["count"] == 1
    assert payload["scope"] == "natural_audio_only_no_tts_or_heldout_input"
    assert set(payload["items"][0]) >= {"natural_sha256", "enhanced_sha256", "residual"}
    assert (output / "item.wav").is_file()
    assert (output / "enhanced_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="fresh"):
        exporter.export_two_stage_natural_audio(export_list, stage1, stage2, output, device="cpu")
