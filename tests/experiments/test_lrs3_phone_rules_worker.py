from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.experiments.lrs3_phone_rules_worker import (
    build_mfa_command,
    extract_features,
    parse_textgrid,
    read_pcm16,
    token_signature,
    write_pcm16,
)

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.4
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.4
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "a"
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "sil"
        intervals [3]:
            xmin = 0.2
            xmax = 0.4
            text = "b"
'''


def test_textgrid_parser_preserves_speech_and_silence(tmp_path) -> None:
    path = tmp_path / "sample.TextGrid"
    path.write_text(TEXTGRID, encoding="utf-8")
    tokens = parse_textgrid(path)
    assert [token["label"] for token in tokens] == ["a", "sil", "b"]
    assert [token["speech"] for token in tokens] == [True, False, True]
    assert token_signature(tokens) == token_signature(tokens)


def test_pcm16_io_is_exact_and_atomic(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    values = np.asarray([0, 1, -2, 32767, -32768], dtype=np.int16)
    metadata = write_pcm16(path, values)
    loaded, read_meta = read_pcm16(path)
    np.testing.assert_array_equal(loaded, values)
    assert metadata["pcm_sha256"] == read_meta["pcm_sha256"]


def test_mfa_command_is_an_argument_vector() -> None:
    command = build_mfa_command("corpus", "dict", "model", "out")
    assert command[:2] == ["mfa", "align"]
    assert "--clean" in command and "--overwrite" in command
    assert all(isinstance(part, str) for part in command)


class _Processor:
    do_normalize = True
    return_attention_mask = False

    def __call__(self, audio, *, sampling_rate, return_tensors, padding):
        assert sampling_rate == 16_000
        return {"input_values": torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)}

    def to_dict(self):
        return {"do_normalize": True}


class _Model:
    config = SimpleNamespace(
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_dilation=[1, 1, 1, 1, 1, 1, 1],
        num_hidden_layers=2,
        hidden_size=3,
    )

    def __call__(self, input_values, output_hidden_states=True):
        del input_values, output_hidden_states
        t = 4
        base = torch.tensor([[[1.0, 0.0, 0.0]] * t], dtype=torch.float32)
        middle = torch.tensor([[[0.0, 1.0, 0.0]] * t], dtype=torch.float32)
        final = torch.tensor([[[0.0, 0.0, 1.0]] * t], dtype=torch.float32)
        return SimpleNamespace(hidden_states=(base, middle, final))


def test_extract_features_records_explicit_frontend_centers() -> None:
    selected, times, metadata = extract_features(
        _Model(), _Processor(), np.zeros(16000, dtype=np.float32), 16000, [1, 2]
    )
    assert sorted(selected) == [1, 2]
    assert times[0] == (199.5 / 16000)
    assert times[1] - times[0] == pytest.approx(320 / 16000)
    assert metadata["frame_time_convention"] == "frontend_receptive_field_centers"
