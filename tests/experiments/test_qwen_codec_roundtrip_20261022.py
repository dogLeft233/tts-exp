"""No model forward: meaningful raw-time/precision/input contract controls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SPEC = importlib.util.spec_from_file_location("qwen_codec_stage", Path(__file__).parents[2] / "scripts" / "experiments" / "qwen_codec_roundtrip_20261022" / "codec_stage.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_rejects_16k_instead_of_silently_changing_clock():
    with pytest.raises(ValueError, match="24000"):
        MODULE.validate_waveform(np.ones(16000, np.float32), 16000)


@pytest.mark.parametrize("x", [np.array([np.nan]), np.array([np.inf]), np.zeros((100, 2)), np.array([])])
def test_invalid_waveform_is_not_silently_repaired(x):
    with pytest.raises(ValueError):
        MODULE.validate_waveform(x, 24000)


def test_float_output_preserves_longer_tail_and_over_fullscale_samples(tmp_path):
    # A decoder tail must remain observable; FLOAT must not saturate values > 1.
    x = np.linspace(-1.2, 1.2, 24123, dtype=np.float32)
    result = MODULE.preserve_float_wav(tmp_path / "raw.wav", x, 24000)
    y, sr = sf.read(tmp_path / "raw.wav", dtype="float32")
    assert sr == 24000 and np.array_equal(y, x)
    assert result["samples"] == 24123 and result["peak"] > 1
    assert result["duration_s"] == 24123 / 24000


def test_all_codebooks_must_be_retained():
    with pytest.raises(ValueError, match="T,16"):
        MODULE.validate_codes(np.zeros((20, 1), dtype=np.int64))
    valid = np.zeros((20, 16), dtype=np.int64)
    valid[:, 0] = 4095
    valid[:, 1:] = 2047
    assert MODULE.validate_codes(valid) is valid
    valid[0, 1] = 2048
    with pytest.raises(ValueError, match="vocabulary"):
        MODULE.validate_codes(valid)


def test_prequalified_input_sha_and_sham_samples_are_exact(tmp_path):
    p = tmp_path / "qualified.wav"
    x = np.sin(np.arange(24001) * .02).astype(np.float32)
    sf.write(p, x, 24000, subtype="FLOAT")
    values, sr, path = MODULE.load_input({"input24_path": str(p), "input24_sha256": MODULE.sha256(p)})
    assert sr == 24000 and path == p and np.array_equal(values, x)
    with pytest.raises(ValueError, match="SHA"):
        MODULE.load_input({"input24_path": str(p), "input24_sha256": "0" * 64})


def test_native_replication_padding_preserves_values_and_expected_clock():
    import torch
    x = torch.tensor([[[2., 3., 5., 7.]]], dtype=torch.float32)
    result = torch.nn.functional.pad(x, (2, 3), mode="replicate")
    assert torch.equal(result, torch.tensor([[[2., 2., 2., 3., 5., 7., 7., 7., 7.]]]))
    assert result.shape[-1] == x.shape[-1] + 5
