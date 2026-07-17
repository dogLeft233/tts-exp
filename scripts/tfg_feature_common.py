"""Shared data structures and utilities for TFG audio feature analysis.

Provides dataclasses (AudioItem, TokenSpan, EmbeddingRecord), constants,
and statistical functions used across the Wav2Sem-style analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa
import pyloudnorm as pyln


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class AudioItem:
    """A single audio sample in the analysis dataset.

    Attributes
    ----------
    sample_id : int
        AISHELL-1 sample identifier (1-13, excluding 9).
    condition : str
        Source condition: "natural" or "tts".
    variant : str
        Processing variant: "raw" or "gain_matched".
    filepath : Path
        Absolute or relative path to the WAV file.
    duration_s : float
        Audio duration in seconds.
    sample_rate : int
        Audio sample rate in Hz.
    """

    sample_id: int
    condition: str
    variant: str
    filepath: Path
    duration_s: float
    sample_rate: int


@dataclass
class TokenSpan:
    """A single token (syllable) with alignment and viseme information.

    Attributes
    ----------
    token : str
        The token string (e.g. "ba").
    initial : str
        Pinyin initial consonant (e.g. "b").
    final : str
        Pinyin final vowel/diphthong (e.g. "a").
    tone : int
        Tone number (1-5; 5 = neutral/light tone).
    start_s : float
        Start time in seconds.
    end_s : float
        End time in seconds.
    confidence : float
        ASR/alignment confidence score in [0, 1].
    viseme : str
        Viseme label (e.g. "aa", "pp").
    """

    token: str
    initial: str
    final: str
    tone: int
    start_s: float
    end_s: float
    confidence: float
    viseme: str


@dataclass
class EmbeddingRecord:
    """Frame-level embeddings from a self-supervised speech model.

    Attributes
    ----------
    sample_id : int
        AISHELL-1 sample identifier.
    condition : str
        Source condition: "natural" or "tts".
    layer : int
        Transformer layer index from which embeddings were extracted.
    model : str
        Model name: "hubert" or "xlsr".
    frame_times : np.ndarray
        Time in seconds for each frame centre, shape (n_frames,).
    embeddings : np.ndarray
        Embedding vectors, shape (n_frames, dim).
    """

    sample_id: int
    condition: str
    layer: int
    model: str
    frame_times: np.ndarray
    embeddings: np.ndarray


# =============================================================================
# Constants
# =============================================================================

STUDY_SAMPLES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
"""All sample IDs in the study set (12 total; sample 9 excluded after SyncNet failure)."""

PILOT_SAMPLES: list[int] = [3, 6, 8, 10, 12]
"""Subset of 5 samples reserved for pilot experiments."""

PRIMARY_TTS: str = "faster_qwen3"
"""Primary TTS engine for analysis."""

VALIDATION_TTS: str = "f5_tts"
"""Validation/secondary TTS engine for cross-model checks."""

OUTPUT_BASE: Path = Path(__file__).resolve().parent.parent / "data" / "wav2sem_analysis"
"""Root output directory for all analysis artefacts."""

TARGET_SR: int = 16000
"""Target sample rate for all audio processing."""

TARGET_LUFS: float = -23.0
"""EBU R128 target integrated loudness level."""


# =============================================================================
# Audio utilities
# =============================================================================


def apply_linear_lufs_gain(
    y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS
) -> tuple[np.ndarray, float]:
    """Apply a pure linear gain to the waveform so it reaches *target_lufs*.

    No dynamic compression or peak limiting is performed — only a constant
    scalar multiplication of the entire signal.

    Parameters
    ----------
    y : np.ndarray
        Mono audio waveform, shape (n_samples,).
    sr : int
        Sample rate in Hz.
    target_lufs : float
        Desired integrated loudness in LUFS (default: -23.0).

    Returns
    -------
    y_scaled : np.ndarray
        Gain-adjusted waveform with same shape and dtype as *y*.
    applied_gain_db : float
        The gain applied in dB (i.e. ``target_lufs - current_lufs``).
    """
    meter = pyln.Meter(sr)
    current_lufs = float(meter.integrated_loudness(y))
    gain_db = target_lufs - current_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)
    y_scaled = (y * gain_linear).astype(y.dtype)
    return y_scaled, float(gain_db)


def load_audio_mono(filepath: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono at the requested sample rate.

    Parameters
    ----------
    filepath : str or Path
        Path to audio file.
    target_sr : int
        Desired sample rate in Hz (default: 16000).

    Returns
    -------
    y : np.ndarray
        Mono audio waveform, shape (n_samples,).
    sr : int
        Actual sample rate used (= *target_sr*).
    """
    y, sr_val = librosa.load(str(filepath), sr=target_sr, mono=True)
    return y, int(sr_val)


# =============================================================================
# Statistical utilities
# =============================================================================


def paired_permutation_test(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 10000, random_seed: int = 42
) -> tuple[float, float, np.ndarray]:
    """Paired permutation test for the null hypothesis *mean(a) == mean(b)*.

    The test statistic is the mean of the paired differences ``a - b``.
    Labels are randomly flipped for each permutation to build the null
    distribution.

    Parameters
    ----------
    a : np.ndarray
        First paired sample, shape (n,).
    b : np.ndarray
        Second paired sample, shape (n,).
    n_permutations : int
        Number of random permutations (default: 10000).
    random_seed : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    p_value : float
        Two-tailed permutation p-value.
    observed_statistic : float
        Mean difference ``mean(a - b)`` observed in the data.
    null_distribution : np.ndarray
        Array of *n_permutations* test-statistic values under the null,
        shape (n_permutations,).
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired_permutation_test expects equal-length arrays, "
            f"got len(a)={len(a)}, len(b)={len(b)}"
        )
    if len(a) == 0:
        raise ValueError("paired_permutation_test expects non-empty arrays")

    rng = np.random.default_rng(random_seed)
    diff = a - b
    observed = float(np.mean(diff))

    null_dist = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        null_dist[i] = float(np.mean(signs * diff))

    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))
    return p_value, observed, null_dist


def bootstrap_paired_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean of paired differences.

    Uses the percentile method: the ``(1 - ci_level) / 2`` and
    ``1 - (1 - ci_level) / 2`` quantiles of the bootstrap distribution.

    The percentile method may have below-nominal coverage for small samples
    (n < 20).  For such cases the bias-corrected-and-accelerated (BCa)
    bootstrap is preferred; see Efron & Tibshirani (1993, Ch. 14).

    Parameters
    ----------
    a : np.ndarray
        First paired sample, shape (n,).
    b : np.ndarray
        Second paired sample, shape (n,).
    n_bootstrap : int
        Number of bootstrap resamples (default: 10000).
    ci_level : float
        Confidence level in (0, 1) (default: 0.95).
    random_seed : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    mean_diff : float
        Point estimate (= mean of ``a - b``).
    """
    if len(a) != len(b):
        raise ValueError(
            f"bootstrap_paired_ci expects equal-length arrays, "
            f"got len(a)={len(a)}, len(b)={len(b)}"
        )
    if len(a) == 0:
        raise ValueError("bootstrap_paired_ci expects non-empty arrays")

    rng = np.random.default_rng(random_seed)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))

    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[i] = float(np.mean(diff[idx]))

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    return ci_lower, ci_upper, mean_diff


def fdr_bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for multiple hypothesis testing.

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw p-values, shape (n_tests,).

    Returns
    -------
    corrected : np.ndarray
        FDR-adjusted p-values (q-values), same shape as *p_values*.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    corrected_sorted = np.minimum(1.0, sorted_p * n / ranks)
    corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
    corrected = np.empty(n, dtype=np.float64)
    corrected[order] = corrected_sorted
    return corrected


# =============================================================================
# Output setup
# =============================================================================

_SUBDIRS: tuple[str, ...] = (
    "manifest",
    "alignments",
    "embeddings",
    "metrics",
    "figures",
)


def ensure_output_dirs() -> dict[str, Path]:
    """Create all Wav2Sem analysis output directories.

    Returns
    -------
    dirs : dict[str, Path]
        Mapping from subdirectory name to resolved ``Path``, with
        ``OUTPUT_BASE`` stored under key ``"base"``.
    """
    base = OUTPUT_BASE.resolve()
    dirs: dict[str, Path] = {"base": base}
    for sub in _SUBDIRS:
        p = base / sub
        p.mkdir(parents=True, exist_ok=True)
        dirs[sub] = p
    return dirs
