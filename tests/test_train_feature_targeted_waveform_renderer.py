#!/usr/bin/env python3
"""Synthetic CPU tests for frozen-target Stage-2 training."""

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

from build_waveform_hubert_targets import build_targets, file_sha256  # noqa: E402
from feature_targeted_waveform_renderer import FeatureTargetedWaveformRenderer, load_frozen_stage1  # noqa: E402
from prepare_two_stage_hubert_manifest import build_two_stage_manifest  # noqa: E402
from run_two_stage_diagnostics import _round_robin_train_subset, _split_boundary  # noqa: E402
from train_feature_targeted_waveform_renderer import (  # noqa: E402
    CHECKPOINT_SELECTION_METRIC,
    TARGET_MODES,
    Stage2LossWeights,
    TargetFirstSchedule,
    TrainConfig,
    compute_stage2_loss,
    load_checkpoint,
    train_stage2,
)
from train_tts_aligned_feature_puller import TrainConfig as Stage1TrainConfig, train_stage1  # noqa: E402
from waveform_hubert_enhancer import EncoderInterface  # noqa: E402

DIM = 768


class FakeHuBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool) -> SimpleNamespace:
        assert output_hidden_states is True
        self.calls += 1
        frames = waveform.shape[-1] // 320
        signal = waveform[:, : frames * 320].reshape(waveform.shape[0], frames, 320).mean(dim=-1)
        base = signal.unsqueeze(-1).expand(-1, -1, DIM) * self.scale
        hidden = tuple(base + float(index) for index in range(7))
        return SimpleNamespace(hidden_states=hidden)


def _tokens() -> list[dict[str, object]]:
    return [
        {"label": "a", "start_s": 0.0, "end_s": 0.5, "confidence": 1.0},
        {"label": "b", "start_s": 0.5, "end_s": 1.0, "confidence": 1.0},
    ]


def _source_record(*, key: str, split: str, condition: str, path: Path) -> dict[str, object]:
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
        natural = tmp_path / f"{key}_natural.wav"
        tts = tmp_path / f"{key}_tts.wav"
        time = np.linspace(0.0, 1.0, 16_000, dtype=np.float32)
        sf.write(natural, 0.3 * np.sin((index + 1) * np.pi * time), 16_000)
        sf.write(tts, 0.3 * np.cos((index + 2) * np.pi * time), 16_000)
        records.extend((
            _source_record(key=key, split=split, condition="natural", path=natural),
            _source_record(key=key, split=split, condition="tts", path=tts),
        ))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": records}, allow_nan=False), encoding="utf-8")
    parent = tmp_path / "parent"
    build_targets(source, parent)
    manifest = tmp_path / "stage.json"
    build_two_stage_manifest(parent / "target_manifest.json", manifest)
    return manifest


def _stage1_best(tmp_path: Path, manifest: Path) -> Path:
    output = tmp_path / "stage1"
    train_stage1(
        manifest,
        output,
        config=Stage1TrainConfig(epochs=1, learning_rate=0.02, bound_quantile=1.0, seed=3),
        encoder=FakeHuBERT(),
        interface=EncoderInterface().as_dict(),
    )
    return output / "best.pt"


def test_target_first_schedule_delays_only_preservation_terms() -> None:
    schedule = TargetFirstSchedule(warmup_epochs=2)
    weights = Stage2LossWeights()

    early = schedule.active_weights(weights, 0)
    late = schedule.active_weights(weights, 2)

    assert early["realize_cosine"] == weights.realize_cosine
    assert early["realize_smooth_l1"] == weights.realize_smooth_l1
    assert early["content_cosine"] == 0.0
    assert early["content_smooth_l1"] == 0.0
    assert early["energy"] == 0.0
    assert late["content_cosine"] == weights.content_cosine
    assert late["energy"] == weights.energy


def test_stage2_loss_has_exact_feature_targeted_whitelist() -> None:
    natural = torch.zeros(1, 1, 640)
    features = torch.zeros(1, 2, DIM, requires_grad=True)
    output = {
        "waveform": natural,
        "residual": natural,
        "natural_layer2": features.detach(),
        "natural_layer6": features.detach(),
        "stage1_goal_layer6": torch.ones_like(features),
        "enhanced_layer2": features,
        "enhanced_layer6": features,
        "padding_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    weights = Stage2LossWeights()
    loss, terms, diagnostics = compute_stage2_loss(
        output,
        natural,
        active_weights=TargetFirstSchedule().active_weights(weights, 1),
        smooth_l1_beta=weights.smooth_l1_beta,
    )
    loss.backward()

    assert set(terms) == {
        "realize_cosine",
        "realize_smooth_l1",
        "content_cosine",
        "content_smooth_l1",
        "energy",
        "residual_l1",
        "residual_smoothness",
        "anti_clipping",
        "total",
    }
    assert diagnostics == {"real_frame_count": 2, "stage1_target_frame_count": 2}
    assert features.grad is not None and torch.isfinite(features.grad).all()




def test_stage2_oracle_loss_uses_only_target_mask() -> None:
    natural = torch.zeros(1, 1, 640)
    features = torch.zeros(1, 2, DIM, requires_grad=True)
    output = {
        "waveform": natural,
        "residual": natural,
        "natural_layer2": features.detach(),
        "natural_layer6": features.detach(),
        "stage1_goal_layer6": torch.ones_like(features),
        "enhanced_layer2": features,
        "enhanced_layer6": features,
        "padding_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    target = torch.ones_like(features)
    mask = torch.tensor([[True, False]])
    weights = Stage2LossWeights()
    _, _, diagnostics = compute_stage2_loss(
        output,
        natural,
        active_weights=TargetFirstSchedule().active_weights(weights, 1),
        smooth_l1_beta=weights.smooth_l1_beta,
        realize_target=target,
        realize_mask=mask,
        target_mode="aligned_raw_tts_oracle",
    )
    assert diagnostics == {
        "real_frame_count": 2,
        "target_mode": "aligned_raw_tts_oracle",
        "realize_target_frame_count": 1,
    }


def test_stage2_cpu_smoke_uses_verified_stage1_and_excludes_weak_targets(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stage1_best = _stage1_best(tmp_path, manifest)
    output = tmp_path / "stage2"
    encoder = FakeHuBERT()

    result = train_stage2(
        manifest,
        stage1_best,
        output,
        config=TrainConfig(epochs=2, learning_rate=0.02, seed=5),
        schedule=TargetFirstSchedule(warmup_epochs=1),
        encoder=encoder,
        interface=EncoderInterface().as_dict(),
        conditioning_dim=8,
        channels=8,
        dilations=(1,),
        residual_scale=0.05,
    )

    assert result["epochs"] == 2
    assert np.isfinite(result["best_metric"])
    assert (output / "last.pt").is_file()
    assert (output / "best.pt").is_file()
    stage1, artifact = load_frozen_stage1(stage1_best)
    renderer = FeatureTargetedWaveformRenderer(
        FakeHuBERT(),
        stage1,
        artifact,
        interface=EncoderInterface(),
        conditioning_dim=8,
        channels=8,
        dilations=(1,),
        residual_scale=0.05,
    )
    last = load_checkpoint(output / "last.pt", renderer)
    assert last["selection"]["metric"] == CHECKPOINT_SELECTION_METRIC
    assert set(last["split_identity"]) == {"train", "valid"}
    assert last["stage1_checkpoint"]["required_filename"] == "best.pt"
    assert last["encoder_interface"]["hidden_size"] == DIM
    assert set(last["loss_weights"]) == {
        "realize_cosine",
        "realize_smooth_l1",
        "content_cosine",
        "content_smooth_l1",
        "energy",
        "residual_l1",
        "residual_smoothness",
        "anti_clipping",
        "smooth_l1_beta",
    }
    assert "raw_tts" not in last["loss_weights"]
    assert "weak_target" not in last["loss_weights"]
    assert encoder.calls == 8
    for entry in last["history"]:
        assert "calibration_gradient_alignment" in entry
        assert entry["valid"]["eligible_fraction"] == 1.0
        for row in entry["train_diagnostics"] + entry["valid_diagnostics"]:
            assert row["encoder_calls"]["raw_tts"] == 0
            assert row["prepared_encoder_calls"]["raw_tts"] == 0
            assert row["enhanced"]["clipped_sample_count"] == 0


def test_stage2_refuses_checkpoint_from_different_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stage1_best = _stage1_best(tmp_path, manifest)
    tampered = tmp_path / "different_stage.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["excluded_provenance"] = dict(
        payload["excluded_provenance"],
        parent_rejected_pair_count=payload["excluded_provenance"]["parent_rejected_pair_count"] + 1,
    )
    tampered.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")



def test_diagnostic_subset_selection_is_nested_and_speaker_disjoint() -> None:
    records = [
        {"paired_key": "b2", "speaker_group": "b", "split": "train"},
        {"paired_key": "a2", "speaker_group": "a", "split": "train"},
        {"paired_key": "b1", "speaker_group": "b", "split": "train"},
        {"paired_key": "a1", "speaker_group": "a", "split": "train"},
        {"paired_key": "v1", "speaker_group": "v", "split": "valid"},
    ]
    boundary = _split_boundary(
        records,
        {"heldout": [{"speaker_group": "h"}]},
    )
    assert boundary["counts"] == {"train": 4, "valid": 1}
    k1 = _round_robin_train_subset(records[:4], 1)
    k4 = _round_robin_train_subset(records[:4], 4)
    assert [row["paired_key"] for row in k1] == ["a1"]
    assert [row["paired_key"] for row in k4] == ["a1", "b1", "a2", "b2"]
    assert [row["paired_key"] for row in k1] == [row["paired_key"] for row in k4[:1]]
    assert set(row["speaker_group"] for row in k4) == {"a", "b"}


def test_stage2_oracle_target_mode_masks_unmatched_frames_and_records_provenance(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stage1_best = _stage1_best(tmp_path, manifest)
    output = tmp_path / "stage2_oracle"
    result = train_stage2(
        manifest,
        stage1_best,
        output,
        config=TrainConfig(epochs=1, learning_rate=0.02, seed=5),
        schedule=TargetFirstSchedule(warmup_epochs=1),
        target_mode="aligned_raw_tts_oracle",
        encoder=FakeHuBERT(),
        interface=EncoderInterface().as_dict(),
        conditioning_dim=8,
        channels=8,
        dilations=(1,),
        residual_scale=0.05,
    )

    assert result["epochs"] == 1
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert payload["target_mode"] == "aligned_raw_tts_oracle"
    assert payload["target_contract"] == {
        "layer": 6,
        "alignment": "matched_phone_spans_only",
        "mask_policy": "unmatched_frames_excluded",
    }
    for entry in payload["history"]:
        for row in entry["train_diagnostics"] + entry["valid_diagnostics"]:
            assert row["target_mode"] == "aligned_raw_tts_oracle"
            assert row["prepared_encoder_calls"]["raw_tts"] == 1
            assert row["loss_diagnostics"]["target_mode"] == "aligned_raw_tts_oracle"
            assert row["loss_diagnostics"]["realize_target_frame_count"] <= row["loss_diagnostics"]["real_frame_count"]
            assert row["realize_target_proximity"]["matched_frame_count"] == row["loss_diagnostics"]["realize_target_frame_count"]
    assert set(TARGET_MODES) == {"stage1_goal", "aligned_raw_tts_oracle"}
