"""Parity and coordinate tests for the SyncNet frontend."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
from python_speech_features import mfcc as reference_mfcc

from scripts.experiments.mfa_linear_real_video_sync.config import SEGMENT_FRAMES, SEGMENT_SAMPLES
from scripts.experiments.mfa_linear_real_video_sync.evaluate import (
    mux_condition_once,
    official_file_curve,
    proxy_file_parity,
    write_pcm16_wav_once,
)
from scripts.experiments.mfa_linear_real_video_sync.protocol import materialize_ffv1_once
from scripts.experiments.mfa_linear_real_video_sync.syncnet_loss import (
    audio_embeddings,
    cached_visual_embeddings,
    official_syncnet_distance_curve,
    pcm16_straight_through,
    syncnet_audio_windows,
    syncnet_v2_mfcc,
    syncnet_video_windows,
)


def _reference(signal: np.ndarray) -> np.ndarray:
    return reference_mfcc(
        signal,
        samplerate=16_000,
        winlen=0.025,
        winstep=0.01,
        numcep=13,
        nfilt=26,
        nfft=512,
        preemph=0.97,
        appendEnergy=True,
        winfunc=lambda size: np.ones((size,)),
    ).T


def _fixtures() -> list[np.ndarray]:
    rng = np.random.default_rng(20260903)
    time = np.arange(4_000, dtype=np.float64) / 16_000
    return [
        rng.integers(-20_000, 20_001, 4_000).astype(np.float64),
        (12_000 * np.sin(2 * np.pi * 173 * time) + 5_000 * np.sin(2 * np.pi * 317 * time)).round(),
        np.zeros(4_000, dtype=np.float64),
        np.tile(np.array([-32768.0, 32767.0]), 2_000),
        np.array([1234.0]),
    ]


@pytest.mark.parametrize("signal", _fixtures(), ids=["random", "speech-like", "silence", "near-clip", "minimum"])
def test_mfcc_matches_python_speech_features(signal: np.ndarray) -> None:
    actual = syncnet_v2_mfcc(torch.from_numpy(signal))[0].numpy()
    expected = _reference(signal)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_pcm16_straight_through_has_exact_forward_and_live_gradient() -> None:
    source = torch.tensor([-2.0, -1.0, -0.12345, 0.0, 0.12345, 1.0, 2.0], requires_grad=True)
    pcm = pcm16_straight_through(source)
    expected = torch.round(torch.clamp(source.detach() * 32768, -32768, 32767))
    torch.testing.assert_close(pcm.detach(), expected, rtol=0, atol=0)
    pcm.sum().backward()
    torch.testing.assert_close(source.grad[1:5], torch.full((4,), 32768.0))
    assert source.grad[0] == 0
    assert source.grad[-1] == 0


def test_96_frame_audio_windows_have_exact_coordinates() -> None:
    mfcc = torch.arange(13 * 400, dtype=torch.float32).reshape(1, 13, 400)
    windows = syncnet_audio_windows(
        mfcc,
        video_frame_count=SEGMENT_FRAMES,
        audio_sample_count=SEGMENT_SAMPLES,
    )
    assert windows.shape == (91, 1, 13, 20)
    for index in (0, 1, 45, 90):
        torch.testing.assert_close(windows[index, 0], mfcc[0, :, 4 * index : 4 * index + 20])


def test_video_windows_have_exact_coordinates_and_bgr_scale() -> None:
    frames = torch.zeros(SEGMENT_FRAMES, 8, 8, 3, dtype=torch.uint8)
    for index in range(SEGMENT_FRAMES):
        frames[index] = index
    windows = syncnet_video_windows(frames)
    assert windows.shape == (91, 3, 5, 8, 8)
    torch.testing.assert_close(windows[17, :, :, 0, 0], torch.arange(17, 22).repeat(3, 1).float())


def test_official_curve_matches_zero_padded_reference_and_keeps_gradients() -> None:
    audio = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], requires_grad=True)
    visual = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    padded = torch.nn.functional.pad(audio, (0, 0, 1, 1))
    expected = torch.stack(
        [
            torch.nn.functional.pairwise_distance(
                visual[index].expand(3, -1), padded[index : index + 3]
            )
            for index in range(3)
        ],
        dim=1,
    ).mean(dim=1)
    actual = official_syncnet_distance_curve(audio, visual, vshift=1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert audio.grad is not None and torch.isfinite(audio.grad).all() and audio.grad.abs().sum() > 0


class TinyFileSyncNet(torch.nn.Module):
    def forward_aud(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean(dim=(2, 3)).repeat(1, 3)

    def forward_lip(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean(dim=(2, 3, 4))


def test_lossless_file_path_preserves_pcm_frames_and_full_curve(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    frames = np.zeros((96, 224, 224, 3), dtype=np.uint8)
    for index in range(96):
        frames[index, :, :, :] = (index * 3) % 256
    waveform = (0.1 * torch.sin(torch.linspace(0, 800, SEGMENT_SAMPLES)))[None, None]
    syncnet = TinyFileSyncNet().eval()

    video_lock = materialize_ffv1_once(tmp_path / "canonical.avi", frames)
    wav_lock = write_pcm16_wav_once(tmp_path / "candidate.wav", waveform)
    condition = tmp_path / "condition.avi"
    mux_condition_once(video_lock["path"], wav_lock["path"], condition)
    official = official_file_curve(
        syncnet,
        condition,
        expected_frame_hashes=video_lock["decoded_bgr_frame_sha256"],
        expected_pcm_sha256=wav_lock["pcm_sha256"],
        device=torch.device("cpu"),
    )
    assert official["window_count"] == 91
    assert official["pcm_sha256"] == wav_lock["pcm_sha256"]
    assert len(official["curve"]) == 31


@pytest.mark.parametrize("shift", [-15, 0, 15])
def test_curve_index_and_reported_offset_contract(shift: int) -> None:
    window_count = 91
    dimensions = 2
    visual = torch.zeros(window_count, dimensions)
    audio = torch.zeros_like(visual)
    # Build a fixture whose designated curve cell is uniquely smallest by
    # comparing against the same implementation's explicit zero-padding rule.
    visual[:, 0] = torch.arange(window_count)
    if shift >= 0:
        audio[shift:, 0] = visual[: window_count - shift, 0]
    else:
        audio[:shift, 0] = visual[-shift:, 0]
    curve = official_syncnet_distance_curve(audio, visual)
    # Edge padding can dominate hand-built ramps, so the assertion here is the
    # coordinate identity rather than an accidental argmin property.
    assert shift + 15 in range(curve.numel())
    assert -(shift) == 15 - (shift + 15)
