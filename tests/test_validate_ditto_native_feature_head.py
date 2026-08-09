"""Pure tests for the Ditto-native validator's safety boundaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_ditto_native_feature_head.py"
spec = importlib.util.spec_from_file_location("validate_ditto_native_feature_head", SCRIPT)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def _tokens(*labels: str, end: float = 1.0) -> list[dict[str, object]]:
    width = end / len(labels)
    return [
        {"phoneme": label, "start_s": index * width, "end_s": (index + 1) * width}
        for index, label in enumerate(labels)
    ]


def _pair(tmp_path: Path) -> dict[str, object]:
    for name in ("natural.wav", "tts.wav"):
        (tmp_path / name).write_bytes(b"audio")
    return {
        "pair_key": "en_001",
        "natural_audio": "natural.wav",
        "tts_audio": "tts.wav",
        "natural_tokens": _tokens("sil", "AA"),
        "tts_tokens": _tokens("AA", "B", "noise"),
        "audio_durations_s": {"natural": 1.0, "tts": 1.0},
        "split": "train",
        "speaker_id": "author:a",
    }


def test_explicit_pairs_accept_independent_token_lists_and_normalize_labels(tmp_path: Path) -> None:
    pair = validator._normalise_pairs([_pair(tmp_path)], tmp_path)
    assert pair[0]["pair_key"] == "en_001"
    assert validator.token_label(0, pair[0]["natural_tokens"]) is None
    assert validator.token_label(20, pair[0]["natural_tokens"]) == "aa"
    assert validator.token_label(0, pair[0]["tts_tokens"]) == "aa"


def test_conflicting_identity_and_duplicate_pairs_fail_closed(tmp_path: Path) -> None:
    first = _pair(tmp_path)
    with pytest.raises(ValueError, match="conflicting pair identity"):
        validator._normalise_pairs([{**first, "paired_key": "other"}], tmp_path)
    with pytest.raises(ValueError, match="duplicate pair key"):
        validator._normalise_pairs([first, {**first}], tmp_path)


def test_manifest_duplicate_arm_and_text_mismatch_fail(tmp_path: Path) -> None:
    for name in ("natural.wav", "tts.wav"):
        (tmp_path / name).write_bytes(b"audio")
    base = {
        "pair_key": "en_001", "split": "train", "text": "hello",
        "audio_path": "natural.wav", "tokens": _tokens("AA"),
    }
    records = [
        {**base, "condition": "natural"},
        {**base, "condition": "natural", "utterance_id": "mdc/en_001/natural"},
        {**base, "condition": "tts", "audio_path": "tts.wav", "text": "bye"},
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate natural arm"):
        validator._manifest_to_pairs(path, repo_root=tmp_path)


def test_manifest_analysis_split_is_selectable(tmp_path: Path) -> None:
    for name in ("natural.wav", "tts.wav"):
        (tmp_path / name).write_bytes(b"audio")
    rows = []
    for condition, name in (("natural", "natural.wav"), ("tts", "tts.wav")):
        rows.append({
            "condition": condition, "sample_key": "en_001", "split": "analysis",
            "text": "hello", "audio_path": name, "tokens": _tokens("AA"),
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    pairs, meta = validator._manifest_to_pairs(path, repo_root=tmp_path, training_split="analysis")
    assert len(pairs) == 1
    assert meta["training_split"] == "analysis"


def test_alignment_gaps_and_duration_overrun_fail(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    pair["natural_tokens"] = [
        {"phoneme": "AA", "start_s": 0.1, "end_s": 1.0},
    ]
    with pytest.raises(ValueError, match="gap between token intervals|starts after zero"):
        validator._validate_pair(pair)
    pair = _pair(tmp_path)
    pair["audio_durations_s"] = {"natural": 0.5, "tts": 1.0}
    with pytest.raises(ValueError, match="exceeds audio duration"):
        validator._validate_pair(pair)


def test_residual_target_diagnostics_identifies_unreachable_components() -> None:
    result = validator.residual_target_diagnostics({"aa": np.array([0.1, 0.3])}, 0.1, 0.5)
    assert result["component_limit_on_delta"] == pytest.approx(0.2)
    assert result["unreachable_labels"] == ["aa"]
    assert result["unreachable_component_count"] == 1


def _valid_checkpoint(path: Path) -> None:
    model = validator.ResidualFeatureAdapter(
        input_dim=validator.DITTO_DIM, hidden_channels=2, dilations=(1,), residual_scale=0.1,
    )
    payload = {
        "schema_version": 1,
        "interface": {
            "input_dim": 1024, "frame_rate": 25.0,
            "frontend": "Ditto Wav2FeatHubert / hubert_streaming_fix_kv.onnx",
            "aggregation": "mean of two 20ms HuBERT outputs -> 25 fps",
        },
        "model_config": {"input_dim": 1024, "hidden_channels": 2, "dilations": [1], "residual_scale": 0.1},
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def test_checkpoint_rejects_nonfinite_tensor_and_interface_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "adapter.pt"
    _valid_checkpoint(path)
    assert validator.load_adapter(path, torch.device("cpu")).training is False
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["input_norm.weight"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="non-finite"):
        validator.load_adapter(path, torch.device("cpu"))
    _valid_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["interface"]["input_dim"] = 768
    torch.save(payload, path)
    with pytest.raises(ValueError, match="interface mismatch"):
        validator.load_adapter(path, torch.device("cpu"))


def test_hook_hides_streaming_when_adapter_installed() -> None:
    class Frontend:
        feat_dim = 1024
        support_streaming = True

        def wav2feat(self, audio, **kwargs):
            return np.zeros((2, 1024), dtype=np.float32)

    adapter = validator.ResidualFeatureAdapter(input_dim=1024, hidden_channels=2, dilations=(1,), residual_scale=0.1)
    hook = validator.DittoFeatureHook(Frontend(), adapter, torch.device("cpu"))
    assert hook.support_streaming is False
    with pytest.raises(RuntimeError, match="online Ditto injection"):
        hook(np.zeros(16000, dtype=np.float32))
