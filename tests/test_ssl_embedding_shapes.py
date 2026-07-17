"""Unit tests for scripts/15_extract_ssl_embeddings.py.

Tests helper functions without requiring torch or transformers downloads.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "15_extract_ssl_embeddings.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ssl_embeddings", str(_SCRIPT_PATH)
)
_ssl_embeddings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ssl_embeddings)

_frame_to_token_pooling = _ssl_embeddings._frame_to_token_pooling
_make_output_stem = _ssl_embeddings._make_output_stem
_validate_layers = _ssl_embeddings._validate_layers
_compute_frame_stride = _ssl_embeddings._compute_frame_stride

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

TEST_DIM = 768


def _make_fake_embeddings(n_frames: int, dim: int = TEST_DIM) -> np.ndarray:
    """Return deterministic fake embeddings for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_frames, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Token-pooling tests
# ---------------------------------------------------------------------------


class TestFrameToTokenPooling:
    """Tests for _frame_to_token_pooling."""

    def test_pooling_basic(self):
        """Pool frames 0, 1, 2 (times 0.0, 0.02, 0.04) for token [0.0, 0.059]."""
        frame_times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32)
        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        tokens = [{"token": "x", "start_s": 0.0, "end_s": 0.059}]
        result = _frame_to_token_pooling(emb, frame_times, tokens)

        assert len(result) == 1
        pooled = np.array(result[0]["embedding"], dtype=np.float32)
        expected = np.mean(emb[:3], axis=0)
        np.testing.assert_allclose(pooled, expected, rtol=1e-6)

    def test_pooling_middle_frames(self):
        """Token [0.01, 0.05] should pool frames at times 0.02 and 0.04."""
        frame_times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32)
        emb = _make_fake_embeddings(4)
        tokens = [{"token": "x", "start_s": 0.01, "end_s": 0.05}]
        result = _frame_to_token_pooling(emb, frame_times, tokens)

        assert len(result) == 1
        pooled = np.array(result[0]["embedding"], dtype=np.float32)
        expected = np.mean(emb[1:3], axis=0)
        np.testing.assert_allclose(pooled, expected, rtol=1e-6)

    def test_pooling_empty_interval(self):
        """Token interval with no matching frames should yield None."""
        frame_times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32)
        emb = _make_fake_embeddings(4)
        tokens = [{"token": "x", "start_s": 0.10, "end_s": 0.15}]
        result = _frame_to_token_pooling(emb, frame_times, tokens)

        assert len(result) == 1
        assert result[0]["embedding"] is None

    def test_pooling_empty_token_list(self):
        """Empty token list should produce empty result."""
        emb = _make_fake_embeddings(4)
        frame_times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32)
        result = _frame_to_token_pooling(emb, frame_times, [])
        assert result == []

    def test_pooling_edge_boundaries(self):
        """Token exactly at frame boundaries should include boundary frames."""
        frame_times = np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32)
        emb = np.ones((4, TEST_DIM), dtype=np.float32)
        tokens = [{"token": "x", "start_s": 0.02, "end_s": 0.04}]
        result = _frame_to_token_pooling(emb, frame_times, tokens)

        assert len(result) == 1
        pooled = np.array(result[0]["embedding"], dtype=np.float32)
        np.testing.assert_allclose(pooled, np.ones(TEST_DIM), rtol=1e-6)

    def test_pooling_single_frame(self):
        """Token covering exactly one frame."""
        frame_times = np.array([0.0, 0.02, 0.04], dtype=np.float32)
        rng = np.random.default_rng(7)
        emb = rng.standard_normal((3, TEST_DIM)).astype(np.float32)
        tokens = [{"token": "x", "start_s": 0.019, "end_s": 0.021}]
        result = _frame_to_token_pooling(emb, frame_times, tokens)

        assert len(result) == 1
        pooled = np.array(result[0]["embedding"], dtype=np.float32)
        np.testing.assert_allclose(pooled, emb[1], rtol=1e-6)


# ---------------------------------------------------------------------------
# Layer validation tests
# ---------------------------------------------------------------------------


class TestLayerValidation:
    """Tests for _validate_layers."""

    def test_valid_layers(self):
        result = _validate_layers([0, 6, 11, 12], num_model_layers=12, model_key="test")
        assert result == [0, 6, 11, 12]

    def test_negative_layer_raises(self):
        with pytest.raises(ValueError, match="Layer -1 out of range"):
            _validate_layers([-1, 6], num_model_layers=12, model_key="test")

    def test_exceeds_max_layer_raises(self):
        with pytest.raises(ValueError, match="Layer 13 out of range"):
            _validate_layers([0, 13], num_model_layers=12, model_key="test")

    def test_last_layer_valid(self):
        result = _validate_layers([12], num_model_layers=12, model_key="test")
        assert result == [12]

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="At least one layer"):
            _validate_layers([], num_model_layers=12, model_key="test")

    def test_deduplicates_and_sorts(self):
        result = _validate_layers([12, 0, 6, 12, 0], num_model_layers=12, model_key="test")
        assert result == [0, 6, 12]


# ---------------------------------------------------------------------------
# Output path tests
# ---------------------------------------------------------------------------


class TestOutputPaths:
    """Tests for _make_output_stem."""

    def test_filename_format(self):
        stem = _make_output_stem(1, "natural", "raw", "hubert")
        assert stem == "1_natural_raw_hubert"

    def test_tts_condition(self):
        stem = _make_output_stem(5, "faster_qwen3", "gain_matched", "xlsr")
        assert stem == "5_faster_qwen3_gain_matched_xlsr"

    def test_parts_are_underscore_separated(self):
        stem = _make_output_stem(10, "tts", "raw", "hubert")
        parts = stem.split("_")
        assert parts[0] == "10"
        assert len(parts) == 4


# ---------------------------------------------------------------------------
# Frame stride computation (mock)
# ---------------------------------------------------------------------------


class TestFrameStride:
    """Tests for _compute_frame_stride."""

    class _FakeConfig:
        conv_stride = [5, 2, 2, 2, 2, 2, 2]

    class _FakeConfigNoStride:
        """Simulate a config missing conv_stride (fallback path)."""
        pass

    def test_hubert_stride_is_320(self):
        stride = _compute_frame_stride(self._FakeConfig)
        assert stride == 320

    def test_fallback_stride(self):
        stride = _compute_frame_stride(self._FakeConfigNoStride)
        assert stride == 320


# ---------------------------------------------------------------------------
# Sine wave input test
# ---------------------------------------------------------------------------


class TestSineWaveInput:
    """Tests for audio input validation without model loading."""

    def test_sine_wave_creates_valid_input(self):
        """1-second 440 Hz sine wave produces correct shape and properties."""
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
        y = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

        assert y.shape == (sr,)
        assert y.dtype == np.float32
        assert len(y) == 16000
        assert -1.0 < y.min() < 1.0
        assert -1.0 < y.max() < 1.0
        assert abs(y.max() - 0.5) < 0.02

        rms = np.sqrt(np.mean(y ** 2))
        assert 0.3 < rms < 0.4

    def test_write_and_read_sine_wave(self):
        """Write sine wave to WAV and verify round-trip properties."""
        sr = 16000
        t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        y = 0.3 * np.sin(2.0 * np.pi * 880.0 * t)

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "test_sine.wav"

            try:
                import soundfile as sf
                sf.write(str(wav_path), y, sr)
            except ImportError:
                import scipy.io.wavfile as wavfile
                y_int16 = (y * 32767).astype(np.int16)
                wavfile.write(str(wav_path), sr, y_int16)

            import librosa
            y2, sr2 = librosa.load(str(wav_path), sr=None, mono=True)
            assert sr2 == sr
            assert abs(len(y2) - len(y)) <= sr // 100
            assert -1.0 < y2.max() <= 1.0
