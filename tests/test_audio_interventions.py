"""Unit tests for scripts/20_causal_feature_interventions.py.

All tests use synthetic audio (generated test tones and noise) — no real
audio files are needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import TARGET_SR

_SCRIPT_PATH = _REPO / "scripts" / "20_causal_feature_interventions.py"
_MODULE_NAME = "_causal_interventions"
_spec = importlib.util.spec_from_file_location(
    _MODULE_NAME, str(_SCRIPT_PATH)
)
_ci = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _ci
_spec.loader.exec_module(_ci)

compute_lufs = _ci.compute_lufs
compute_spectral_tilt = _ci.compute_spectral_tilt
compute_spectral_centroid = _ci.compute_spectral_centroid
compute_energy_env_std = _ci.compute_energy_env_std
apply_lufs_match = _ci.apply_lufs_match
apply_spectral_tilt_match = _ci.apply_spectral_tilt_match
apply_dynamic_transform = _ci.apply_dynamic_transform
verify_intervention = _ci.verify_intervention
measure = _ci.measure
evaluate_pilot = _ci.evaluate_pilot
InterventionResult = _ci.InterventionResult
resolve_audio_dirs = _ci._resolve_audio_dirs


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SR = 16000


def _make_sine(duration_s: float = 3.0, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_white_noise(duration_s: float = 3.0, amplitude: float = 0.1) -> np.ndarray:
    """Generate white noise."""
    return (amplitude * RNG.standard_normal(int(duration_s * SR))).astype(np.float32)


def _make_pinkish_noise(duration_s: float = 3.0, amplitude: float = 0.1) -> np.ndarray:
    """Generate noise with a -6 dB/oct slope (roughly pink-ish)."""
    n = int(duration_s * SR)
    white = RNG.standard_normal(n)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec = np.fft.rfft(white)
    gain = 1.0 / np.maximum(np.sqrt(np.maximum(freqs, 1.0)), 1.0)
    spec *= gain
    y = np.fft.irfft(spec, n=n)
    y = amplitude * y / np.max(np.abs(y))
    return y.astype(np.float32)


def _make_loud_tone(duration_s: float = 3.0) -> np.ndarray:
    """Generate a loud sine tone (~ -12 dB LUFS)."""
    return _make_sine(duration_s, amplitude=0.95)


def _make_quiet_tone(duration_s: float = 3.0) -> np.ndarray:
    """Generate a quiet sine tone (~ -30 dB LUFS)."""
    return _make_sine(duration_s, amplitude=0.1)


# ---------------------------------------------------------------------------
# LUFS matching
# ---------------------------------------------------------------------------


class TestLufsMatching:
    def test_matching_reduces_loudness(self):
        """Apply LUFS matching from loud to quiet, verify post-LUFS is closer to target."""
        loud = _make_loud_tone()
        target_lufs = -30.0
        transformed = apply_lufs_match(loud, SR, target_lufs)

        orig_lufs = compute_lufs(loud, SR)
        post_lufs = compute_lufs(transformed, SR)

        assert orig_lufs > post_lufs, f"Expected original LUFS ({orig_lufs:.1f}) > post ({post_lufs:.1f})"
        assert abs(post_lufs - target_lufs) < 3.0, (
            f"Post-LUFS ({post_lufs:.1f}) should be within 3 dB of target ({target_lufs})"
        )
        assert orig_lufs - post_lufs > 2.0

    def test_matching_toward_target(self):
        """LUFS matching moves audio toward target LUFS."""
        y = _make_sine(amplitude=0.5)
        target = -23.0
        transformed = apply_lufs_match(y, SR, target)
        post_lufs = compute_lufs(transformed, SR)
        assert abs(post_lufs - target) < 3.0


# ---------------------------------------------------------------------------
# Spectral tilt
# ---------------------------------------------------------------------------


class TestSpectralTiltComputation:
    def test_white_noise_near_zero_tilt(self):
        """White noise has near-zero spectral tilt (flat spectrum)."""
        noise = _make_white_noise(duration_s=2.0)
        tilt = compute_spectral_tilt(noise, SR)
        assert abs(tilt) < 3.0, f"White noise tilt ({tilt:.3f}) should be near zero"

    def test_lowpass_noise_has_negative_tilt(self):
        """Low-pass filtered noise has negative tilt (high-frequency roll-off)."""
        white = _make_white_noise(duration_s=2.0)
        # Apply strong low-pass by averaging in time domain
        window = np.ones(40) / 40.0
        lowpass = np.convolve(white, window, mode="same")
        tilt_lp = compute_spectral_tilt(lowpass.astype(np.float32), SR)
        assert tilt_lp < 0.0, f"Lowpassed noise tilt ({tilt_lp:.3f}) should be negative"

    def test_pinkish_noise_negative_tilt(self):
        """Pink-ish noise (high-frequency roll-off) should have negative tilt."""
        pink = _make_pinkish_noise(duration_s=2.0)
        tilt = compute_spectral_tilt(pink, SR)
        assert tilt < 0.0, f"Pink noise tilt ({tilt:.3f}) should be negative"


class TestSpectralTiltMatching:
    def test_direction_moves_toward_target(self):
        """Apply tilt correction, verify slope moves toward target."""
        source = _make_pinkish_noise(duration_s=2.0)
        target_tilt = 0.0  # flat

        before = compute_spectral_tilt(source, SR)
        transformed = apply_spectral_tilt_match(source, SR, target_tilt)
        after = compute_spectral_tilt(transformed, SR)

        assert abs(after - target_tilt) < abs(before - target_tilt), (
            f"After tilt ({after:.3f}) should be closer to target ({target_tilt}) than before ({before:.3f})"
        )

    def test_matches_known_targets(self):
        """Two different signals with different tilts — matching works for both."""
        signal_a = _make_white_noise(duration_s=2.0)
        tilt_a = compute_spectral_tilt(signal_a, SR)
        tilt_target = tilt_a + 2.0

        transformed = apply_spectral_tilt_match(signal_a, SR, tilt_target)
        tilt_after = compute_spectral_tilt(transformed, SR)

        dist_before = abs(tilt_a - tilt_target)
        dist_after = abs(tilt_after - tilt_target)
        assert dist_after < dist_before, (
            f"dist_before={dist_before:.3f} dist_after={dist_after:.3f}"
        )

    def test_reaches_target_tilt_magnitude(self):
        """Matching must apply the requested tilt, not merely move directionally."""
        source = _make_pinkish_noise(duration_s=2.0)
        before = compute_spectral_tilt(source, SR)
        target = before + 0.5

        transformed = apply_spectral_tilt_match(source, SR, target)
        after = compute_spectral_tilt(transformed, SR)

        assert abs(after - target) < 0.05, (
            f"after tilt ({after:.3f}) should reach target ({target:.3f})"
        )


# ---------------------------------------------------------------------------
# Dynamic range
# ---------------------------------------------------------------------------


class TestDynamicRange:
    def test_compression_reduces_variance(self):
        """Compress dynamic range, verify energy_env_std decreases."""
        # Signal with varying amplitude
        t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        y = (env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        before = compute_energy_env_std(y, SR)
        compressed = apply_dynamic_transform(y, SR, exponent=0.7)
        after = compute_energy_env_std(compressed, SR)

        assert after < before, (
            f"Compression should reduce energy_env_std: before={before:.6f} after={after:.6f}"
        )

    def test_expansion_increases_variance(self):
        """Expand dynamic range, verify energy_env_std increases."""
        t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
        env = 0.5 + 0.3 * np.sin(2 * np.pi * 1.0 * t)
        y = (env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        before = compute_energy_env_std(y, SR)
        expanded = apply_dynamic_transform(y, SR, exponent=1.3)
        after = compute_energy_env_std(expanded, SR)

        assert after > before, (
            f"Expansion should increase energy_env_std: before={before:.6f} after={after:.6f}"
        )

    def test_identity_exponent_unchanged(self):
        """Exponent 1.0 should leave signal essentially unchanged."""
        t = np.linspace(0, 1.0, int(1.0 * SR), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        y += 0.01 * RNG.standard_normal(len(y)).astype(np.float32)

        transformed = apply_dynamic_transform(y, SR, exponent=1.0)
        np.testing.assert_allclose(y, transformed, atol=0.1)


# ---------------------------------------------------------------------------
# Verify intervention
# ---------------------------------------------------------------------------


class TestVerifyIntervention:
    def test_passes_correct_direction(self):
        """Increase test case with 'increase' expected -> passes."""
        y_quiet = _make_quiet_tone()
        y_loud = apply_lufs_match(y_quiet, SR, -12.0)

        passed, delta, before, after = verify_intervention(
            y_quiet, y_loud, "LUFS", "increase", SR
        )
        assert passed, f"Expected pass for LUFS increase, got delta={delta}"
        assert delta > 0
        assert after > before

    def test_fails_wrong_direction(self):
        """Decrease test case with 'increase' expected -> fails."""
        y_loud = _make_loud_tone()
        y_quiet = apply_lufs_match(y_loud, SR, -30.0)

        passed, delta, before, after = verify_intervention(
            y_loud, y_quiet, "LUFS", "increase", SR
        )
        assert not passed, f"Expected fail for LUFS decrease with 'increase' direction, got delta={delta}"
        assert delta < 0
        assert after < before

    def test_passes_decrease_direction(self):
        """Decrease test case with 'decrease' expected -> passes."""
        y_loud = _make_loud_tone()
        y_quiet = apply_lufs_match(y_loud, SR, -30.0)

        passed, delta, before, after = verify_intervention(
            y_loud, y_quiet, "LUFS", "decrease", SR
        )
        assert passed, f"Expected pass for LUFS decrease, got delta={delta}"

    def test_energy_env_std_verification(self):
        """Verify works with energy_env_std metric."""
        t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        y = (env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        compressed = apply_dynamic_transform(y, SR, exponent=0.7)
        passed, delta, before, after = verify_intervention(
            y, compressed, "energy_env_std", "decrease", SR
        )
        assert passed


# ---------------------------------------------------------------------------
# Pilot threshold detection
# ---------------------------------------------------------------------------


class TestPilotThreshold:
    def test_pilot_passes_with_4_of_5_verified(self):
        """4/5 verified = pilot passes."""
        results = [
            InterventionResult(
                name="test_intervention",
                target_metric="LUFS",
                expected_direction="decrease",
                verified_count=4,
                total=5,
                pre_mean=-18.0,
                post_mean=-23.0,
            ),
        ]
        status = evaluate_pilot(results)
        assert "PASSED" in status

    def test_pilot_inconclusive_with_2_of_5_verified(self):
        """2/5 verified = pilot inconclusive."""
        results = [
            InterventionResult(
                name="test_intervention",
                target_metric="LUFS",
                expected_direction="decrease",
                verified_count=2,
                total=5,
                pre_mean=-18.0,
                post_mean=-22.0,
            ),
        ]
        status = evaluate_pilot(results)
        assert "INCONCLUSIVE" in status

    def test_multiple_interventions_one_pass_sufficient(self):
        """At least one passing intervention = pilot passes overall."""
        results = [
            InterventionResult(
                name="fail_intervention",
                target_metric="LUFS",
                expected_direction="decrease",
                verified_count=0,
                total=5,
            ),
            InterventionResult(
                name="pass_intervention",
                target_metric="energy_env_std",
                expected_direction="decrease",
                verified_count=3,
                total=5,
            ),
            InterventionResult(
                name="inconclusive_intervention",
                target_metric="spectral_tilt",
                expected_direction="increase",
                verified_count=2,
                total=5,
            ),
        ]
        status = evaluate_pilot(results)
        assert "PASSED" in status


# ---------------------------------------------------------------------------
# Bidirectional intervention inverse
# ---------------------------------------------------------------------------


class TestBidirectionalInverse:
    def test_lufs_bidirectional_opposite_deltas(self):
        """TTS->NAT and NAT->TTS LUFS adjustments should have opposite deltas."""
        t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
        y_nat = _make_sine(amplitude=0.2)  # quieter (natural)
        y_tts = _make_sine(amplitude=0.8)  # louder (TTS)

        # TTS -> natural LUFS: make TTS quieter
        nat_lufs = compute_lufs(y_nat, SR)
        tts_to_nat = apply_lufs_match(y_tts, SR, nat_lufs)
        _, delta_tts_to_nat, _, _ = verify_intervention(
            y_tts, tts_to_nat, "LUFS", "decrease", SR
        )

        # natural -> TTS LUFS: make natural louder
        tts_lufs = compute_lufs(y_tts, SR)
        nat_to_tts = apply_lufs_match(y_nat, SR, tts_lufs)
        _, delta_nat_to_tts, _, _ = verify_intervention(
            y_nat, nat_to_tts, "LUFS", "increase", SR
        )

        # One delta is positive, the other negative
        assert delta_tts_to_nat < 0, f"TTS->nat should decrease LUFS, got {delta_tts_to_nat}"
        assert delta_nat_to_tts > 0, f"Nat->TTS should increase LUFS, got {delta_nat_to_tts}"
        assert delta_tts_to_nat * delta_nat_to_tts < 0, (
            f"Deltas should have opposite signs: {delta_tts_to_nat} vs {delta_nat_to_tts}"
        )


# ---------------------------------------------------------------------------
# measure() function
# ---------------------------------------------------------------------------


class TestMeasure:
    def test_measure_lufs(self):
        y = _make_loud_tone()
        val = measure(y, "LUFS", SR)
        assert isinstance(val, float)
        assert not np.isnan(val)

    def test_measure_spectral_tilt(self):
        y = _make_white_noise(duration_s=2.0)
        val = measure(y, "spectral_tilt", SR)
        assert isinstance(val, float)
        assert not np.isnan(val)

    def test_measure_spectral_centroid(self):
        y = _make_sine(freq=440.0)
        val = measure(y, "spectral_centroid", SR)
        assert isinstance(val, float)
        assert val > 0

    def test_measure_energy_env_std(self):
        y = _make_sine()
        val = measure(y, "energy_env_std", SR)
        assert isinstance(val, float)
        assert val > 0

    def test_measure_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            measure(_make_sine(), "nonexistent", SR)


# ---------------------------------------------------------------------------
# Smoke test: run_smoke
# ---------------------------------------------------------------------------


class TestRunSmoke:
    def test_run_smoke_produces_results(self):
        """Smoke test produces non-empty intervention results."""
        results = _ci.run_smoke([1, 2, 3, 4, 5])
        assert len(results) > 0
        for r in results:
            assert r.total == 5
            assert not np.isnan(r.mean_delta)
            assert 0 <= r.verified_count <= r.total


# ---------------------------------------------------------------------------
# Audio directory resolution
# ---------------------------------------------------------------------------


class TestAudioDirectoryResolution:
    def test_default_tts_dir_is_phase1_faster_qwen3_run(self):
        """Default interventions must use the TTS source used by Phase 1."""
        assert _ci._TTS_AUDIO_DIR == (
            _REPO / "runs" / "r2_faster_qwen3_20260707T145233Z" / "02_tts"
        )

    def test_input_dir_resolves_natural_and_tts_subdirectories(self, tmp_path):
        """The CLI input root must select real natural/tts audio directories."""
        natural_dir = tmp_path / "natural"
        tts_dir = tmp_path / "tts"
        natural_dir.mkdir()
        tts_dir.mkdir()
        (natural_dir / "1.wav").touch()
        (tts_dir / "1.wav").touch()

        assert resolve_audio_dirs(tmp_path) == (natural_dir, tts_dir)


# ---------------------------------------------------------------------------
# __main__ smoke via CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help(self):
        import subprocess
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0
        assert "--smoke" in proc.stdout

    def test_smoke_pilot_runs(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), "--pilot", "--verify-only", "--smoke",
                 "--output-dir", td],
                capture_output=True, text=True, timeout=120,
            )
            assert proc.returncode == 0
            assert "intervention_results.json" in proc.stdout

    def test_smoke_output_file(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir()
            proc = subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), "--pilot", "--verify-only", "--smoke",
                 "--output-dir", str(out_dir)],
                capture_output=True, text=True, timeout=120,
            )
            assert proc.returncode == 0
            result_path = out_dir / "intervention_results.json"
            assert result_path.exists()
            data = json.loads(result_path.read_text())
            assert data["phase"] == "pilot"
            assert len(data["interventions"]) > 0
