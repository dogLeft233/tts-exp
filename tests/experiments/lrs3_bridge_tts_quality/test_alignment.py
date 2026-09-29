from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.experiments.lrs3_bridge_tts_quality.targets import (
    _extract_features_covering_audio,
    _mfa_command,
    _run_mfa,
)
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    AlignmentError,
    build_frame_mapping,
    extend_final_token_for_feature_tail,
    validate_tokens,
)


def _tokens(end: float = 0.12) -> list[dict]:
    return [
        {"label": "aa", "start_s": 0.0, "end_s": 0.06},
        {"label": "sil", "start_s": 0.06, "end_s": end},
    ]


def test_mapping_keeps_natural_grid_and_uses_only_silence_fallback() -> None:
    mapping, stats = build_frame_mapping(5, 7, _tokens(), _tokens(0.17))
    assert [row.natural_frame_index for row in mapping] == list(range(5))
    assert stats["speech_fallback_frames"] == 0
    assert stats["silence_fallback_frames"] >= 0


def test_unknown_speech_and_invalid_phone_spans_fail() -> None:
    with pytest.raises(AlignmentError):
        validate_tokens([{"label": "spn", "start_s": 0.0, "end_s": 0.1}])
    with pytest.raises(AlignmentError):
        validate_tokens([{"label": "aa", "start_s": 0.1, "end_s": 0.2}])


def test_feature_tail_extension_is_bounded() -> None:
    extended, amount = extend_final_token_for_feature_tail(_tokens(0.10), 5, frame_stride_samples=320, sample_rate=16000, max_extension_s=0.02)
    assert amount >= 0.0
    assert extended[-1]["end_s"] >= 0.10
    with pytest.raises(AlignmentError):
        extend_final_token_for_feature_tail(_tokens(0.01), 20, frame_stride_samples=320, sample_rate=16000, max_extension_s=0.02)


def test_mfa_batch_command_uses_single_speaker_mode() -> None:
    command = _mfa_command(Path("/tmp/mfa-input"), Path("/tmp/mfa-output"))
    assert "--single_speaker" in command


def test_mfa_recovers_missing_textgrid_with_single_sample_retry(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_path = tmp_path / "mfa.log"
    input_dir.mkdir()
    for sample_id in ("a", "b"):
        (input_dir / f"{sample_id}.wav").write_bytes(b"wav")
        (input_dir / f"{sample_id}.lab").write_text("AA\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        corpus = Path(command[5])
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(path.stem for path in corpus.glob("*.wav"))
        if len(sample_ids) > 1:
            sample_ids = sample_ids[:1]
        for sample_id in sample_ids:
            (destination / f"{sample_id}.TextGrid").write_text("TextGrid\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="stdout", stderr="stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _run_mfa(input_dir, output_dir, log_path)

    assert sorted(path.stem for path in output_dir.glob("*.TextGrid")) == ["a", "b"]
    assert len(calls) == 2
    assert all("--single_speaker" in command for command in calls)
    assert result["missing_after_batch"] == ["b"]
    assert result["single_sample_retries"][0]["sample_id"] == "b"


def test_feature_extraction_adds_deterministic_right_context() -> None:
    class FakeAdapter:
        def extract(self, waveform):
            padded_samples = int(waveform.shape[1])
            frame_count = max(1, int(np.ceil(padded_samples / 320.0)) - 1)
            return torch.zeros((frame_count, 1024), dtype=torch.float32)

    waveform = np.zeros(3 * 320 + 64, dtype=np.float32)
    features, metadata = _extract_features_covering_audio(FakeAdapter(), waveform)

    assert features.shape == (4, 1024)
    assert metadata["original_samples"] == waveform.size
    assert metadata["right_pad_samples"] == 336
    assert metadata["feature_stride_samples"] == 320
    assert metadata["feature_frames"] == 4
