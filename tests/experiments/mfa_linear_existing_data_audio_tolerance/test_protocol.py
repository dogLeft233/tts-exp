from __future__ import annotations

import pytest

from scripts.experiments.mfa_linear_existing_data_audio_tolerance.config import DiagnosticConfig, validate_serialized_config
from scripts.experiments.mfa_linear_existing_data_audio_tolerance.real_video import decide_diagnostic


def test_diagnostic_configuration_exposes_post_hoc_tolerance() -> None:
    payload = DiagnosticConfig().to_dict()
    assert payload["original_mel_limit"] == 0.10
    assert payload["diagnostic_mel_limit"] == 0.11
    validate_serialized_config(payload)
    payload["diagnostic_mel_limit"] = 0.20
    with pytest.raises(ValueError, match="frozen"):
        validate_serialized_config(payload)


def test_diagnostic_never_emits_scientific_claim() -> None:
    rows = [
        {
            "sample_id": f"row-{index}",
            "engineering_valid": True,
            "scientific_success_observed": index < 6,
            "d_gain": 0.004,
            "c_gain": 0.004,
        }
        for index in range(8)
    ]
    result = decide_diagnostic(rows)
    assert result["status"] == "DIAGNOSTIC_REAL_VIDEO_COMPLETE"
    assert result["scientific_claim_available"] is False
    assert result["unchanged_real_video_gate_pass"] is True
