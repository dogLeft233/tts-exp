#!/usr/bin/env python3
"""Tests for strict direct raw-TTS Stage-1 feature-puller training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_waveform_hubert_targets import build_targets, file_sha256
from prepare_two_stage_hubert_manifest import build_two_stage_manifest
from train_tts_aligned_feature_puller import (
    CHECKPOINT_SELECTION_METRIC,
    TrainConfig,
    extract_hubert_layers,
    fit_stage1_statistics,
    load_checkpoint,
    load_stage1_records,
    prepare_record,
    train_stage1,
)
from waveform_hubert_enhancer import EncoderInterface

DIM = 768


class FakeHuBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool) -> SimpleNamespace:
        assert output_hidden_states is True
        self.calls += 1
        frames = max(1, waveform.shape[-1] // 320)
        signal = waveform[:, : frames * 320].reshape(waveform.shape[0], frames, 320).mean(dim=-1)
        base = signal.unsqueeze(-1).expand(-1, -1, DIM) * self.scale
        hidden = tuple(base + float(index) for index in range(7))
        return SimpleNamespace(hidden_states=hidden)


def _tokens() -> list[dict[str, object]]:
    return [
        {"label": "a", "start_s": 0.0, "end_s": 0.5, "confidence": 1.0},
        {"label": "b", "start_s": 0.5, "end_s": 1.0, "confidence": 1.0},
    ]


def _source_record(
    *,
    key: str,
    split: str,
    condition: str,
    path: Path,
) -> dict[str, object]:
    return {
        "paired_key": key,
        "condition": condition,
        "variant": "raw",
        "tts_provider": "faster_qwen3" if condition == "tts" else None,
        "audio_path": str(path),
        "filepath": str(path),
        "source_sha256": file_sha256(path),
        "transcript": f"transcript {key}",
        "split": split,
        "speaker_group": f"speaker_{split}",
        "alignment_source": "mfa",
        "tokens": _tokens(),
    }


def _manifest(tmp_path: Path) -> Path:
    records: list[dict[str, object]] = []
    for index, (key, split) in enumerate((("train_pair", "train"), ("valid_pair", "valid"))):
        natural_path = tmp_path / f"{key}_natural.wav"
        tts_path = tmp_path / f"{key}_tts.wav"
        time = np.linspace(0.0, 1.0, 16_000, dtype=np.float32)
        sf.write(natural_path, np.sin((index + 1) * np.pi * time), 16_000)
        sf.write(tts_path, np.cos((index + 2) * np.pi * time), 16_000)
        records.extend(
            [
                _source_record(key=key, split=split, condition="natural", path=natural_path),
                _source_record(key=key, split=split, condition="tts", path=tts_path),
            ]
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": records}, allow_nan=False), encoding="utf-8")
    parent_dir = tmp_path / "parent"
    build_targets(source, parent_dir)
    manifest = tmp_path / "stage.json"
    build_two_stage_manifest(parent_dir / "target_manifest.json", manifest)
    return manifest


def _interface() -> dict[str, object]:
    return EncoderInterface().as_dict()


def test_multilayer_extraction_uses_one_frozen_encoder_call() -> None:
    encoder = FakeHuBERT()
    waveform = torch.ones(1, 1, 1280)

    layers = extract_hubert_layers(
        encoder,
        waveform,
        layers=(2, 6),
        interface=_interface(),
    )

    assert encoder.calls == 1
    assert set(layers) == {2, 6}
    assert layers[2].shape == (1, 4, DIM)
    assert layers[6].shape == (1, 4, DIM)
    assert not layers[2].requires_grad
    assert encoder.scale.requires_grad


def test_prepare_record_uses_only_raw_audio_and_direct_alignment(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    groups, _ = load_stage1_records(manifest)
    encoder = FakeHuBERT()

    prepared = prepare_record(
        groups["train"][0],
        encoder=encoder,
        interface=_interface(),
        device=torch.device("cpu"),
    )

    assert encoder.calls == 2
    assert prepared["encoder_calls"] == {"natural_l2_l6": 1, "raw_tts_l6": 1}
    assert prepared["target_mask"].dtype is torch.bool
    assert prepared["target_mask"].any()
    assert "target_path" not in prepared
    assert "waveform" not in prepared
    assert tuple(prepared["natural_layer6"].shape) == tuple(prepared["aligned_tts_layer6"].shape)


def test_stage1_loader_rejects_heldout_before_any_feature_work(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["split"] = "heldout"
    manifest.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot load split"):
        load_stage1_records(manifest)



def test_train_only_statistics_accept_variable_feature_lengths() -> None:
    generator = torch.Generator().manual_seed(3)
    records = []
    for frames in (3, 5):
        natural_l2 = torch.randn(1, frames, DIM, generator=generator)
        natural_l6 = torch.randn(1, frames, DIM, generator=generator)
        target = natural_l6 + 0.1
        records.append(
            {
                "split": "train",
                "natural_layer2": natural_l2,
                "natural_layer6": natural_l6,
                "aligned_tts_layer6": target,
                "padding_mask": torch.ones(1, frames, dtype=torch.bool),
                "target_mask": torch.ones(1, frames, dtype=torch.bool),
            }
        )

    normalizer, bound = fit_stage1_statistics(records, bound_quantile=1.0)

    assert normalizer.source_frame_count == 8
    assert bound.source_frame_count == 8
    assert bound.value > 0


def test_deterministic_cpu_smoke_writes_valid_last_and_best_checkpoints(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "run"
    encoder = FakeHuBERT()

    result = train_stage1(
        manifest,
        output,
        config=TrainConfig(
            epochs=2,
            learning_rate=0.02,
            bound_quantile=1.0,
            seed=13,
            device="cpu",
        ),
        encoder=encoder,
        interface=_interface(),
    )

    assert result["epochs"] == 2
    assert np.isfinite(result["best_metric"])
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()
    last = load_checkpoint(output / "last.pt")
    best = load_checkpoint(output / "best.pt")
    assert set(last["split_identity"]) == {"train", "valid"}
    assert "heldout" not in json.dumps(last["split_identity"])
    assert last["selection"]["metric"] == CHECKPOINT_SELECTION_METRIC
    assert last["valid_raw_tts_proximity"]["item_count"] == 1
    assert 0 <= last["valid_raw_tts_proximity"]["goal_closer_item_count"] <= 1
    assert np.isfinite(last["valid_raw_tts_proximity"]["relative_improvement"])
    assert best["history"][-1]["valid"][CHECKPOINT_SELECTION_METRIC] >= 0.0
    assert encoder.calls == 4
    for entry in last["history"]:
        for row in entry["train_diagnostics"] + entry["valid_diagnostics"]:
            assert row["encoder_calls"] == {"natural_l2_l6": 1, "raw_tts_l6": 1}
            assert row["delta"]["max"] <= last["model_config"]["hard_bound"]["normalized_l6_l2_cap"] + 1e-5



def test_stage1_training_refuses_nonfresh_output_directory(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fresh"):
        train_stage1(
            manifest,
            output,
            config=TrainConfig(epochs=1),
            encoder=FakeHuBERT(),
            interface=_interface(),
        )
    manifest = _manifest(tmp_path)
    output = tmp_path / "run"
    train_stage1(
        manifest,
        output,
        config=TrainConfig(epochs=1, bound_quantile=1.0, seed=2),
        encoder=FakeHuBERT(),
        interface=_interface(),
    )
    payload = torch.load(output / "best.pt", weights_only=False)
    state = dict(payload["state_dict"])
    name = next(iter(state))
    state[name] = state[name][..., :1]
    payload["state_dict"] = state
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)

    with pytest.raises(ValueError, match="state_dict"):
        load_checkpoint(invalid)
