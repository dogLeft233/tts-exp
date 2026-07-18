#!/usr/bin/env python3
"""20_causal_feature_interventions.py — Minimal causal feature interventions based on Phase 1 candidates.

Applies targeted audio transformations (LUFS matching, spectral-tilt matching,
dynamic range adjustment) to isolate causal effects of specific features.
Each intervention is verified to actually change the target metric.

Pilot-first: --pilot processes PILOT_SAMPLES and reports whether the effect is
consistent enough to warrant expanding to the full STUDY_SAMPLES set.

Usage:
    python scripts/20_causal_feature_interventions.py --pilot --verify-only --smoke
    python scripts/20_causal_feature_interventions.py --pilot
    python scripts/20_causal_feature_interventions.py --samples STUDY
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import (
    PILOT_SAMPLES,
    STUDY_SAMPLES,
    TARGET_LUFS,
    TARGET_SR,
    apply_linear_lufs_gain,
    load_audio_mono,
    ensure_output_dirs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sample-to-audio-path mapping (natural audio from the AISHELL-1 evaluation run)
_RUN_BASE = _REPO / "runs" / "aishell1_ldn_20260707T070936Z"
_NAT_AUDIO_DIR = _REPO / "data" / "data" / "audio"
_TTS_AUDIO_DIR = _REPO / "runs" / "r2_faster_qwen3_20260707T145233Z" / "02_tts"

# Fallback audio dirs if the primary path doesn't exist
_FALLBACK_NAT = _REPO / "runs" / "r5_f5_tts" / "02_tts"
_FALLBACK_TTS = _FALLBACK_NAT


def _resolve_audio_dirs(input_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve real natural/TTS audio directories.

    ``--input-dir`` follows the documented ``natural/`` and ``tts/`` layout.
    Without an override, use the repository's AISHELL natural audio and the
    matching Faster-Qwen3 run output.
    """
    if input_dir is not None:
        candidates = [
            (input_dir / "natural", input_dir / "tts"),
            (input_dir / "natural_raw", input_dir / "tts_raw"),
        ]
    else:
        candidates = [
            (_NAT_AUDIO_DIR, _TTS_AUDIO_DIR),
            (_FALLBACK_NAT, _FALLBACK_TTS),
        ]

    for nat_dir, tts_dir in candidates:
        if nat_dir.is_dir() and tts_dir.is_dir():
            if any(nat_dir.glob("*.wav")) and any(tts_dir.glob("*.wav")):
                return nat_dir, tts_dir

    # Preserve the existing synthetic-smoke fallback, but return the first
    # candidate so the caller's warning identifies the attempted paths.
    return candidates[0]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InterventionResult:
    """Results from a single intervention run."""

    name: str
    target_metric: str
    expected_direction: str  # "increase" or "decrease"
    verified_count: int = 0
    total: int = 0
    pre_mean: float = float("nan")
    post_mean: float = float("nan")
    mean_delta: float = float("nan")
    pilot_result: str = "UNKNOWN"
    per_sample_deltas: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio transformations
# ---------------------------------------------------------------------------


def compute_lufs(y: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS."""
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))


def apply_lufs_match(
    y_src: np.ndarray, sr: int, target_lufs: float
) -> np.ndarray:
    """Apply linear gain so *y_src* reaches *target_lufs*."""
    y_scaled, _gain = apply_linear_lufs_gain(y_src, sr, target_lufs=target_lufs)
    return y_scaled


def compute_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz."""
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    if np.sum(spec) == 0:
        return 0.0
    return float(np.sum(freqs * spec) / np.sum(spec))


def compute_spectral_tilt(y: np.ndarray, sr: int) -> float:
    """Compute spectral tilt as the slope of log-mag vs log-freq (linear regression).

    Returns the slope (dB/octave equivalent).  Negative values indicate
    high-frequency roll-off (warmer/darker sound).  Zero = flat (white).
    Positive values indicate high-frequency emphasis (brighter).
    """
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    mask = (freqs > 80.0) & (freqs < sr / 2 - 1) & (spec > 1e-12)
    if np.sum(mask) < 4:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_mag = np.log10(spec[mask])
    slope = float(np.polyfit(log_f, log_mag, 1)[0])
    return slope


def apply_spectral_tilt_match(
    y_src: np.ndarray, sr: int, target_slope: float
) -> np.ndarray:
    """Match the spectral tilt of *y_src* to *target_slope*.

    Applies an inverse filter in the frequency domain to change the long-term
    average spectrum slope without changing the overall energy.
    """
    current_slope = compute_spectral_tilt(y_src, sr)
    delta_slope = target_slope - current_slope

    n = len(y_src)
    spec = np.fft.rfft(y_src)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # The measured tilt is d(log10(magnitude)) / d(log10(frequency)).
    # Convert that amplitude-ratio slope to dB before applying the filter.
    log_f = np.log10(np.maximum(freqs, 1.0))
    ref_log = np.log10(1000.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_db = 20.0 * delta_slope * (log_f - ref_log)
    gain_db = np.where(np.isfinite(gain_db), gain_db, 0.0)
    gain_db[0] = 0.0  # DC unchanged
    gain_linear = 10.0 ** (gain_db / 20.0)

    spec_filtered = spec * gain_linear
    y_out = np.fft.irfft(spec_filtered, n=n).astype(y_src.dtype)

    # Preserve original RMS energy
    rms_src = np.sqrt(np.mean(y_src ** 2))
    rms_out = np.sqrt(np.mean(y_out ** 2))
    if rms_out > 0:
        y_out = y_out * (rms_src / rms_out)

    return y_out


def compute_energy_envelope(y: np.ndarray, sr: int, window_ms: float = 50.0) -> np.ndarray:
    """Compute RMS energy envelope with window."""
    win_len = int(window_ms * sr / 1000.0)
    win_len = max(win_len, 1)
    env_sq = np.convolve(y ** 2, np.ones(win_len) / win_len, mode="same")
    return np.sqrt(np.maximum(env_sq, 0.0))


def compute_energy_env_std(y: np.ndarray, sr: int) -> float:
    """Compute standard deviation of the energy envelope."""
    env = compute_energy_envelope(y, sr)
    return float(np.std(env))


def apply_dynamic_transform(
    y: np.ndarray, sr: int, exponent: float
) -> np.ndarray:
    """Apply an exponent transform to the RMS envelope.

    exponent < 1.0 compresses dynamic range (smooths amplitude variations).
    exponent > 1.0 expands dynamic range (exaggerates variations).
    exponent = 1.0 is identity.

    Resynthesis: y_out = sign(y) * (envelope_normalized ^ exponent) * envelope, then
    re-scale to preserve overall RMS energy.
    """
    env = compute_energy_envelope(y, sr)
    env_mean = np.mean(env) + 1e-12
    env_norm = env / env_mean

    # Apply exponent to normalized envelope
    env_transformed = env_norm ** exponent

    # Resynthesise: scale the original signal by transformed envelope
    # Avoid division by zero
    safe_env = np.maximum(env, 1e-12)
    y_out = y * (env_transformed * env_mean / safe_env)

    # Clamp to prevent clipping
    peak = np.max(np.abs(y_out))
    if peak > 0.99:
        y_out = y_out / peak * 0.99

    return y_out.astype(y.dtype)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def measure(y: np.ndarray, metric: str, sr: int) -> float:
    """Dispatch wrapper for metric computation."""
    if metric == "LUFS":
        return compute_lufs(y, sr)
    elif metric == "spectral_tilt":
        return compute_spectral_tilt(y, sr)
    elif metric == "spectral_centroid":
        return compute_spectral_centroid(y, sr)
    elif metric == "energy_env_std":
        return compute_energy_env_std(y, sr)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def verify_intervention(
    original_audio: np.ndarray,
    transformed_audio: np.ndarray,
    target_metric: str,
    expected_direction: str,
    sr: int,
) -> tuple[bool, float, float, float]:
    """Verify that the intervention changed the target feature in the expected direction.

    Parameters
    ----------
    original_audio : np.ndarray
        Pre-intervention waveform.
    transformed_audio : np.ndarray
        Post-intervention waveform.
    target_metric : str
        Metric name from {"LUFS", "spectral_tilt", "spectral_centroid", "energy_env_std"}.
    expected_direction : str
        "increase" or "decrease".
    sr : int
        Sample rate.

    Returns
    -------
    passed : bool
        True if the delta matches expected_direction.
    delta : float
        after - before difference.
    before : float
        Pre-intervention metric value.
    after : float
        Post-intervention metric value.
    """
    before = measure(original_audio, target_metric, sr)
    after = measure(transformed_audio, target_metric, sr)
    delta = after - before
    if expected_direction == "increase" and delta > 0:
        return True, delta, before, after
    elif expected_direction == "decrease" and delta < 0:
        return True, delta, before, after
    else:
        return False, delta, before, after


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def load_sample_pairs(
    sample_ids: list[int],
    nat_dir: Path,
    tts_dir: Path,
    sr_target: int = TARGET_SR,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[int]]:
    """Load natural and TTS audio for the given sample IDs.

    Returns (natural_audio, tts_audio, loaded_ids) — only samples where both
    files exist are returned.
    """
    natural: dict[int, np.ndarray] = {}
    tts: dict[int, np.ndarray] = {}
    loaded: list[int] = []

    for sid in sample_ids:
        nat_path = nat_dir / f"{sid}.wav"
        tts_path = tts_dir / f"{sid}.wav"
        if nat_path.exists() and tts_path.exists():
            y_nat, sr = load_audio_mono(nat_path, target_sr=sr_target)
            y_tts, sr = load_audio_mono(tts_path, target_sr=sr_target)
            natural[sid] = y_nat
            tts[sid] = y_tts
            loaded.append(sid)

    if not loaded:
        print("WARNING: No sample audio files found. Generating synthetic test tones.")
        rng = np.random.default_rng(42)
        dur = int(3.0 * sr_target)
        t_line = np.linspace(0, 3.0, dur, endpoint=False)
        for sid in sample_ids:
            natural[sid] = (
                0.3 * np.sin(2 * np.pi * 440 * t_line)
                + 0.2 * np.sin(2 * np.pi * 880 * t_line)
            ).astype(np.float32)
            natural[sid] *= 0.3 / np.max(np.abs(natural[sid]))
            natural[sid] += 0.003 * rng.standard_normal(dur).astype(np.float32)
            tts[sid] = (
                0.35 * np.sin(2 * np.pi * 440 * t_line)
                + 0.25 * np.sin(2 * np.pi * 880 * t_line)
            ).astype(np.float32)
            tts[sid] *= 0.5 / np.max(np.abs(tts[sid]))
            tts[sid] += 0.002 * rng.standard_normal(dur).astype(np.float32)
            loaded.append(sid)

    return natural, tts, loaded


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def run_lufs_interventions(
    natural: dict[int, np.ndarray],
    tts: dict[int, np.ndarray],
    sample_ids: list[int],
    sr: int,
) -> list[InterventionResult]:
    """LUFS-matching: bidirectional matching between natural and TTS."""
    results: list[InterventionResult] = []

    # Direction 1: TTS -> natural LUFS (reduce TTS loudness)
    res = InterventionResult(
        name="LUFS_match_tts_to_nat",
        target_metric="LUFS",
        expected_direction="decrease" if _mean_lufs(tts, sample_ids, sr) > _mean_lufs(natural, sample_ids, sr) else "increase",
    )
    for sid in sample_ids:
        tgt_lufs = compute_lufs(natural[sid], sr)
        transformed = apply_lufs_match(tts[sid], sr, tgt_lufs)
        passed, delta, pre, post = verify_intervention(
            tts[sid], transformed, "LUFS", res.expected_direction, sr
        )
        if passed:
            res.verified_count += 1
        res.total += 1
        res.per_sample_deltas[sid] = delta
    res.pre_mean = float(np.mean([compute_lufs(tts[s], sr) for s in sample_ids]))
    res.post_mean = float(np.mean([compute_lufs(
        apply_lufs_match(tts[s], sr, compute_lufs(natural[s], sr)), sr
    ) for s in sample_ids]))
    res.mean_delta = float(np.mean(list(res.per_sample_deltas.values())))
    results.append(res)

    # Direction 2: natural -> TTS LUFS (boost natural)
    res2 = InterventionResult(
        name="LUFS_match_nat_to_tts",
        target_metric="LUFS",
        expected_direction="increase" if _mean_lufs(tts, sample_ids, sr) > _mean_lufs(natural, sample_ids, sr) else "decrease",
    )
    for sid in sample_ids:
        tgt_lufs = compute_lufs(tts[sid], sr)
        transformed = apply_lufs_match(natural[sid], sr, tgt_lufs)
        passed, delta, pre, post = verify_intervention(
            natural[sid], transformed, "LUFS", res2.expected_direction, sr
        )
        if passed:
            res2.verified_count += 1
        res2.total += 1
        res2.per_sample_deltas[sid] = delta
    res2.pre_mean = float(np.mean([compute_lufs(natural[s], sr) for s in sample_ids]))
    res2.post_mean = float(np.mean([compute_lufs(
        apply_lufs_match(natural[s], sr, compute_lufs(tts[s], sr)), sr
    ) for s in sample_ids]))
    res2.mean_delta = float(np.mean(list(res2.per_sample_deltas.values())))
    results.append(res2)

    return results


def _mean_lufs(audio_dict: dict[int, np.ndarray], sample_ids: list[int], sr: int) -> float:
    return float(np.mean([compute_lufs(audio_dict[s], sr) for s in sample_ids]))


def run_spectral_interventions(
    natural: dict[int, np.ndarray],
    tts: dict[int, np.ndarray],
    sample_ids: list[int],
    sr: int,
) -> list[InterventionResult]:
    """Spectral-tilt matching: bidirectional."""
    results: list[InterventionResult] = []

    # Direction 1: TTS -> natural tilt (make TTS brighter/darker to match natural)
    res = InterventionResult(
        name="spectral_tilt_match_tts_to_nat",
        target_metric="spectral_tilt",
        expected_direction="unknown",
    )
    for sid in sample_ids:
        nat_tilt = compute_spectral_tilt(natural[sid], sr)
        transformed = apply_spectral_tilt_match(tts[sid], sr, nat_tilt)
        tts_tilt_before = compute_spectral_tilt(tts[sid], sr)
        res.expected_direction = "increase" if nat_tilt > tts_tilt_before else "decrease"
        passed, delta, pre, post = verify_intervention(
            tts[sid], transformed, "spectral_tilt", res.expected_direction, sr
        )
        if passed:
            res.verified_count += 1
        res.total += 1
        res.per_sample_deltas[sid] = delta
    if res.total > 0:
        res.mean_delta = float(np.mean(list(res.per_sample_deltas.values())))
        res.pre_mean = float(np.mean([compute_spectral_tilt(tts[s], sr) for s in sample_ids]))
        res.post_mean = float(np.mean([compute_spectral_tilt(
            apply_spectral_tilt_match(tts[s], sr, compute_spectral_tilt(natural[s], sr)), sr
        ) for s in sample_ids]))
    results.append(res)

    # Direction 2: natural -> TTS tilt
    res2 = InterventionResult(
        name="spectral_tilt_match_nat_to_tts",
        target_metric="spectral_tilt",
        expected_direction="unknown",
    )
    for sid in sample_ids:
        tts_tilt = compute_spectral_tilt(tts[sid], sr)
        transformed = apply_spectral_tilt_match(natural[sid], sr, tts_tilt)
        nat_tilt_before = compute_spectral_tilt(natural[sid], sr)
        res2.expected_direction = "increase" if tts_tilt > nat_tilt_before else "decrease"
        passed, delta, pre, post = verify_intervention(
            natural[sid], transformed, "spectral_tilt", res2.expected_direction, sr
        )
        if passed:
            res2.verified_count += 1
        res2.total += 1
        res2.per_sample_deltas[sid] = delta
    if res2.total > 0:
        res2.mean_delta = float(np.mean(list(res2.per_sample_deltas.values())))
        res2.pre_mean = float(np.mean([compute_spectral_tilt(natural[s], sr) for s in sample_ids]))
        res2.post_mean = float(np.mean([compute_spectral_tilt(
            apply_spectral_tilt_match(natural[s], sr, compute_spectral_tilt(tts[s], sr)), sr
        ) for s in sample_ids]))
    results.append(res2)

    return results


def run_dynamic_interventions(
    natural: dict[int, np.ndarray],
    tts: dict[int, np.ndarray],
    sample_ids: list[int],
    sr: int,
) -> list[InterventionResult]:
    """Dynamic range adjustment: compression and expansion."""
    results: list[InterventionResult] = []

    COMPRESS_EXPONENT = 0.7
    EXPAND_EXPONENT = 1.3

    # Compression: reduce dynamic range -> energy_env_std should decrease
    for source_dict, cond_name, exponent, name_suffix, expected_dir in [
        (tts, "tts", COMPRESS_EXPONENT, "_tts_compress", "decrease"),
        (natural, "natural", COMPRESS_EXPONENT, "_nat_compress", "decrease"),
    ]:
        res = InterventionResult(
            name=f"dynamic_compression{name_suffix}",
            target_metric="energy_env_std",
            expected_direction=expected_dir,
        )
        for sid in sample_ids:
            transformed = apply_dynamic_transform(source_dict[sid], sr, exponent)
            passed, delta, pre, post = verify_intervention(
                source_dict[sid], transformed, "energy_env_std", expected_dir, sr
            )
            if passed:
                res.verified_count += 1
            res.total += 1
            res.per_sample_deltas[sid] = delta
        if res.total > 0:
            res.mean_delta = float(np.mean(list(res.per_sample_deltas.values())))
            pre_vals = [compute_energy_env_std(source_dict[s], sr) for s in sample_ids]
            post_vals = [compute_energy_env_std(
                apply_dynamic_transform(source_dict[s], sr, exponent), sr
            ) for s in sample_ids]
            res.pre_mean = float(np.mean(pre_vals))
            res.post_mean = float(np.mean(post_vals))
        results.append(res)

    for source_dict, cond_name, exponent, name_suffix, expected_dir in [
        (tts, "tts", EXPAND_EXPONENT, "_tts_expand", "increase"),
        (natural, "natural", EXPAND_EXPONENT, "_nat_expand", "increase"),
    ]:
        res = InterventionResult(
            name=f"dynamic_expansion{name_suffix}",
            target_metric="energy_env_std",
            expected_direction=expected_dir,
        )
        for sid in sample_ids:
            transformed = apply_dynamic_transform(source_dict[sid], sr, exponent)
            passed, delta, pre, post = verify_intervention(
                source_dict[sid], transformed, "energy_env_std", expected_dir, sr
            )
            if passed:
                res.verified_count += 1
            res.total += 1
            res.per_sample_deltas[sid] = delta
        if res.total > 0:
            res.mean_delta = float(np.mean(list(res.per_sample_deltas.values())))
            pre_vals = [compute_energy_env_std(source_dict[s], sr) for s in sample_ids]
            post_vals = [compute_energy_env_std(
                apply_dynamic_transform(source_dict[s], sr, exponent), sr
            ) for s in sample_ids]
            res.pre_mean = float(np.mean(pre_vals))
            res.post_mean = float(np.mean(post_vals))
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Pilot evaluation
# ---------------------------------------------------------------------------


def evaluate_pilot(results: list[InterventionResult]) -> str:
    """Evaluate whether pilot results are conclusive enough to expand.

    Returns "PASS", "INCONCLUSIVE", or "FAIL".
    A PASS requires at least one intervention with >= 3/5 verified.
    INCONCLUSIVE means some interventions had partial success (1-2/5).
    FAIL means all interventions had zero verified deltas.
    """
    pass_count = 0
    inconclusive_count = 0
    fail_count = 0
    for res in results:
        if res.total == 0:
            continue
        if res.verified_count >= 3:  # majority of 5 pilot samples
            res.pilot_result = "PASS"
            pass_count += 1
        elif res.verified_count == 0:
            res.pilot_result = "FAIL"
            fail_count += 1
        else:
            res.pilot_result = "INCONCLUSIVE"
            inconclusive_count += 1

    if pass_count >= 1:
        return "PILOT PASSED — ready to expand to full 12 samples"
    elif inconclusive_count >= 1:
        return "PILOT INCONCLUSIVE — effect too small or inconsistent"
    else:
        return "PILOT INCONCLUSIVE — no intervention shows consistent effect"


# ---------------------------------------------------------------------------
# Smoke test mode
# ---------------------------------------------------------------------------


def run_smoke(sample_ids: list[int]) -> list[InterventionResult]:
    """Run all interventions on synthetic test tones (no file I/O needed)."""
    sr = 16000
    natural, tts, loaded = load_sample_pairs(sample_ids, Path("/nonexistent"), Path("/nonexistent"))

    all_results: list[InterventionResult] = []
    all_results.extend(run_lufs_interventions(natural, tts, loaded, sr))
    all_results.extend(run_spectral_interventions(natural, tts, loaded, sr))
    all_results.extend(run_dynamic_interventions(natural, tts, loaded, sr))
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal causal feature interventions (Phase 2)"
    )
    parser.add_argument(
        "--input-dir", type=Path,
        help="Base directory containing natural/ and tts/ subdirs"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory for intervention results"
    )
    parser.add_argument(
        "--samples", choices=["PILOT", "STUDY"], default="PILOT",
        help="Sample set to use (default: PILOT)"
    )
    parser.add_argument(
        "--interventions", type=str, default="LUFS,SPECTRAL,DYNAMIC",
        help="Comma-separated intervention types (default: LUFS,SPECTRAL,DYNAMIC)"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot mode: process only PILOT_SAMPLES and report pass/inconclusive"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing transformed audio, don't regenerate"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run with synthetic test tones (no real audio files needed)"
    )
    args = parser.parse_args()

    # Resolve sample list
    if args.pilot:
        sample_ids = list(PILOT_SAMPLES)
    elif args.samples == "PILOT":
        sample_ids = list(PILOT_SAMPLES)
    else:
        sample_ids = list(STUDY_SAMPLES)

    # Parse intervention types
    intervention_types = [t.strip().upper() for t in args.interventions.split(",")]

    sr = TARGET_SR

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = ensure_output_dirs()["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[InterventionResult] = []

    if args.smoke:
        print(f"SMOKE MODE: using synthetic test tones for {len(sample_ids)} samples")
        all_results = run_smoke(sample_ids)
    else:
        nat_dir, tts_dir = _resolve_audio_dirs(args.input_dir)
        print(f"Audio inputs: natural={nat_dir}, tts={tts_dir}")
        natural, tts, loaded = load_sample_pairs(sample_ids, nat_dir, tts_dir)

        if "LUFS" in intervention_types:
            print(f"--- LUFS matching ({len(loaded)} samples) ---")
            all_results.extend(run_lufs_interventions(natural, tts, loaded, sr))

        if "SPECTRAL" in intervention_types:
            print(f"--- Spectral-tilt matching ({len(loaded)} samples) ---")
            all_results.extend(run_spectral_interventions(natural, tts, loaded, sr))

        if "DYNAMIC" in intervention_types:
            print(f"--- Dynamic range adjustment ({len(loaded)} samples) ---")
            all_results.extend(run_dynamic_interventions(natural, tts, loaded, sr))

    # Print per-intervention summary
    for res in all_results:
        print(f"\n[{res.name}]")
        print(f"  target: {res.target_metric}, expected: {res.expected_direction}")
        print(f"  verified: {res.verified_count}/{res.total}")
        print(f"  pre_mean: {res.pre_mean:.4f}, post_mean: {res.post_mean:.4f}")
        print(f"  mean_delta: {res.mean_delta:.4f}")
        print(f"  pilot_result: {res.pilot_result}")

    # Pilot evaluation
    if args.pilot:
        pilot_status = evaluate_pilot(all_results)
        print(f"\n{'=' * 60}")
        print(pilot_status)
        print(f"{'=' * 60}")

    # Write JSON output
    output_path = out_dir / "intervention_results.json"
    phase = "pilot" if args.pilot or args.samples == "PILOT" else "study"
    output = {
        "phase": phase,
        "samples": sample_ids,
        "interventions": [
            {
                "name": r.name,
                "target_metric": r.target_metric,
                "expected_direction": r.expected_direction,
                "verified_count": r.verified_count,
                "total": r.total,
                "pre_mean": r.pre_mean,
                "post_mean": r.post_mean,
                "mean_delta": r.mean_delta,
                "pilot_result": r.pilot_result,
                "per_sample_deltas": r.per_sample_deltas,
            }
            for r in all_results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
