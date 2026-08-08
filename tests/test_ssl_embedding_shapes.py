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
assert _spec is not None and _spec.loader is not None
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


    def test_mdc_string_id_writes_flat_embedding_outputs(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "natural.wav"
        audio_path.touch()
        output_dir = tmp_path / "run" / "embeddings"

        monkeypatch.setattr(
            _ssl_embeddings,
            "load_model",
            lambda model_name, device="cpu": (object(), 3, 320, 1),
        )
        monkeypatch.setattr(
            _ssl_embeddings,
            "_load_audio_mono",
            lambda filepath: (np.zeros(1600, dtype=np.float32), 16000),
        )
        monkeypatch.setattr(
            _ssl_embeddings,
            "extract_frame_embeddings",
            lambda model, audio, sample_rate, layers, device="cpu": (
                np.ones((1, 4, 3), dtype=np.float32),
                np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float32),
            ),
        )

        manifest = [{
            "sample_id": "en_001",
            "utterance_id": "mdc_tts/en_001/natural",
            "condition": "natural",
            "variant": "raw",
            "audio_path": str(audio_path),
            "dataset": "mdc_tts",
        }]
        counts = _ssl_embeddings.process_all(
            manifest, ["hubert"], [0], output_dir, device="cpu"
        )

        stem = "en_001_natural_raw_hubert"
        assert counts == {"processed": 1, "skipped": 0, "failed": 0}
        assert (output_dir / f"{stem}.npy").is_file()
        assert (output_dir / f"{stem}.json").is_file()
        assert not (output_dir / "mdc_tts").exists()

        second_counts = _ssl_embeddings.process_all(
            manifest, ["hubert"], [0], output_dir, device="cpu"
        )
        assert second_counts == {"processed": 0, "skipped": 1, "failed": 0}


class TestExtractionLayerMapping:
    """Tests for hidden-state index semantics."""

    def test_hidden_state_indices_are_inclusive_and_unshifted(self, monkeypatch):
        class FakeConfig:
            num_hidden_layers = 2
            conv_stride = [1]
            hidden_size = 2

        class FakeHiddenState:
            def __init__(self, value):
                self._array = np.full((1, 3, 2), value, dtype=np.float32)
                self.shape = self._array.shape

            def squeeze(self, axis):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._array.squeeze(0)

        hidden_states = tuple(FakeHiddenState(value) for value in range(3))

        class FakeModel:
            config = FakeConfig()

            def __call__(self, waveform, output_hidden_states):
                return type("Outputs", (), {"hidden_states": hidden_states})()

        class FakeTorch:
            class _NoGrad:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            @staticmethod
            def from_numpy(audio):
                return FakeTensor()

            @staticmethod
            def no_grad():
                return FakeTorch._NoGrad()

        class FakeTensor:
            def copy(self):
                return self

            def float(self):
                return self

            def unsqueeze(self, axis):
                return self

            def to(self, device):
                return self

        monkeypatch.setattr(_ssl_embeddings, "torch", FakeTorch)
        embeddings, _ = _ssl_embeddings.extract_frame_embeddings(
            FakeModel(), np.zeros(3, dtype=np.float32), 16000, [0, 1, 2]
        )
        assert embeddings.shape == (3, 3, 2)
        np.testing.assert_array_equal(embeddings[:, 0, 0], [0.0, 1.0, 2.0])




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
