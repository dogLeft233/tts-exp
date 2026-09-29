from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_phone_rules_audio import (
    build_edit_mask,
    make_gain_control,
    render_rule_arm,
    spectral_emphasis,
    validate_waveform,
)


def _pcm(length: int = 16_000) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / 16_000.0
    return np.rint(8_000 * np.sin(2 * np.pi * 220 * time) + 2_000 * np.sin(2 * np.pi * 2_200 * time)).astype(np.int16)


def _tokens() -> list[dict]:
    return [
        {"label": "a", "start_s": 0.10, "end_s": 0.45, "speech": True},
        {"label": "sil", "start_s": 0.45, "end_s": 0.60, "silence": True},
        {"label": "b", "start_s": 0.60, "end_s": 0.95, "speech": True},
    ]


def test_mask_protects_pause_and_phone_edges() -> None:
    mask = build_edit_mask(16_000, _tokens())
    assert np.all(mask[: int(0.11 * 16_000)] == 0.0)
    assert np.all(mask[int(0.45 * 16_000): int(0.60 * 16_000)] == 0.0)
    assert np.all(mask[int(0.95 * 16_000):] == 0.0)
    assert np.any(mask > 0.0)
    assert np.all((mask >= 0.0) & (mask <= 1.0))


def test_identity_and_rule_arms_keep_length_and_pause_pcm_exact() -> None:
    source = _pcm()
    identity, identity_meta = render_rule_arm(source, _tokens(), "identity")
    np.testing.assert_array_equal(identity, source)
    assert identity_meta["effective_edit"] is False
    mask = build_edit_mask(source.size, _tokens())
    rendered, meta = render_rule_arm(source, _tokens(), "spectral_drc")
    assert rendered.dtype == np.int16
    assert rendered.size == source.size
    np.testing.assert_array_equal(rendered[mask <= 0.0], source[mask <= 0.0])
    assert "alpha" in meta
    assert validate_waveform(source, rendered, mask)["pass"] is True


def test_rule_render_is_deterministic_and_ablation_names_are_fixed() -> None:
    source = _pcm()
    first, first_meta = render_rule_arm(source, _tokens(), "spectral_only")
    second, second_meta = render_rule_arm(source, _tokens(), "spectral_only")
    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    drc, _ = render_rule_arm(source, _tokens(), "drc_only")
    assert drc.shape == source.shape


def test_gain_control_matches_target_without_hidden_limiter() -> None:
    source = np.rint(8_000 * np.sin(2 * np.pi * np.arange(16_000) / 16_000.0)).astype(np.int16)
    mask = np.ones(source.size, dtype=np.float64)
    target = float(np.sum((source.astype(np.float64) / 32768.0 * 1.1) ** 2))
    output, metadata = make_gain_control(source, mask, target)
    assert metadata["valid"] is True
    assert metadata["g"] == pytest.approx(1.1, abs=2e-4)
    assert metadata["energy_error_db"] == pytest.approx(0.0, abs=0.05)
    assert np.max(np.abs(output)) <= 32767


def test_spectral_short_input_and_nonfinite_input_are_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        spectral_emphasis(np.zeros(256, dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        spectral_emphasis(np.asarray([0.0, np.nan] * 300, dtype=np.float64))


def test_full_rule_handles_near_peak_by_residual_alpha() -> None:
    source = np.full(16_000, 32_000, dtype=np.int16)
    rendered, metadata = render_rule_arm(source, _tokens(), "spectral_drc")
    assert rendered.dtype == np.int16
    assert np.max(rendered) <= 32767
    assert 0.0 <= metadata["alpha"] <= 1.0
