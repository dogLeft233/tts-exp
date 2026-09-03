"""Machine-recomputable stage decision tests."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.experiments.mfa_linear_real_video_sync.config import PrototypeConfig
from scripts.experiments.mfa_linear_real_video_sync.evaluate import decide_p1, decide_p2
from scripts.experiments.mfa_linear_real_video_sync.run import build_parser
from scripts.experiments.mfa_linear_real_video_sync.validate import validate_run


def curve(target: int, *, distance: float, confidence: float) -> list[float]:
    values = np.full(31, distance + confidence + 0.2, dtype=np.float64)
    values[target + 15] = distance
    # Keep the median exactly distance + confidence.
    values[:15] = distance + confidence
    if target + 15 < 15:
        values[30] = distance + confidence + 0.2
    return values.tolist()


def passing_evidence() -> dict[str, object]:
    return {
        "input_lock": True,
        "input_isolation": True,
        "visual_coordinates": True,
        "target_unambiguous": True,
        "offset_sign": True,
        "mfcc_parity": True,
        "identity": True,
        "frozen_state": True,
        "candidate_gradient": True,
        "proxy_step0_loss": 1.0,
        "proxy_final_loss": 0.5,
        "official_step0_curve": curve(0, distance=1.0, confidence=0.5),
        "official_final_curve": curve(0, distance=0.99, confidence=0.51),
        "proxy_final_curve": curve(0, distance=0.99, confidence=0.51),
        "target_offset": 0,
        "mel_distance": 0.05,
        "saturation_fraction": 0.0,
        "final_qc": {"finite": True, "exact_shape": True, "residual_bound_pass": True, "pcm_saturation_pass": True},
    }


def test_failed_p0_artifact_graph_is_valid_and_has_no_training(tmp_path) -> None:
    (tmp_path / "01_p0_seam").mkdir()
    (tmp_path / "01_p0_seam/test_results.json").write_text(
        '{"status":"failed","returncode":2}', encoding="utf-8"
    )
    (tmp_path / "decision.json").write_text(
        '{"status":"ENGINEERING_NO_GO","pass":false,"failed_stage":"P0_SEAM","error":"synthetic failure"}',
        encoding="utf-8",
    )
    result = validate_run(tmp_path)
    assert result["artifact_graph_valid"]
    assert not result["scientific_pass"]


def test_scientific_config_and_cli_do_not_expose_search_or_retry() -> None:
    config = PrototypeConfig()
    serialized = config.to_dict()
    assert serialized["optimizer"] == "AdamW"
    assert serialized["learning_rate"] == 1e-4
    assert serialized["betas"] == [0.9, 0.999]
    assert serialized["epsilon"] == 1e-8
    assert serialized["weight_decay"] == 0.01
    assert serialized["gradient_clip_norm"] == 1.0
    assert serialized["p1_steps"] == 20
    assert serialized["p2_steps"] == 100
    with pytest.raises(ValueError, match="frozen"):
        PrototypeConfig(learning_rate=2e-4).validate()
    options = {action.dest for action in build_parser()._actions}
    assert {"repo", "output", "run_p2", "resume"}.issubset(options)
    assert not options.intersection({"seed", "steps", "learning_rate", "retry", "warm_start", "checkpoint"})


def test_p1_pass_requires_every_gate() -> None:
    decision = decide_p1(passing_evidence())
    assert decision["pass"]
    assert decision["status"] == "FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED"
    assert "generalization" in decision["disallowed_claims"]


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("input_lock", False, "INPUT_LOCK_FAILURE"),
        ("input_isolation", False, "NATURAL_INPUT_LEAKAGE"),
        ("visual_coordinates", False, "VISUAL_COORDINATE_MISMATCH"),
        ("target_unambiguous", False, "TARGET_OFFSET_AMBIGUOUS"),
        ("offset_sign", False, "OFFSET_SIGN_MISMATCH"),
        ("mfcc_parity", False, "MFCC_PARITY_FAILURE"),
        ("identity", False, "IDENTITY_FAILURE"),
        ("frozen_state", False, "FROZEN_STATE_MUTATION"),
        ("candidate_gradient", False, "NO_CANDIDATE_GRADIENT"),
        ("mel_distance", 0.10001, "TRUST_REGION_FAILURE"),
        ("saturation_fraction", 0.00011, "TRUST_REGION_FAILURE"),
        ("proxy_final_loss", 1.0, "NO_PROXY_LOSS_DESCENT"),
    ],
)
def test_p1_failure_precedence(field: str, value: object, status: str) -> None:
    evidence = passing_evidence()
    evidence[field] = value
    assert decide_p1(evidence)["status"] == status


def test_proxy_only_improvement_never_passes() -> None:
    evidence = passing_evidence()
    evidence["official_final_curve"] = evidence["official_step0_curve"]
    evidence["proxy_final_curve"] = evidence["official_step0_curve"]
    decision = decide_p1(evidence)
    assert not decision["pass"]
    assert decision["status"] == "OFFICIAL_D_MARGIN_FAILURE"


def test_proxy_official_parity_is_mandatory() -> None:
    evidence = passing_evidence()
    proxy = list(evidence["proxy_final_curve"])
    proxy[0] += 0.0011
    evidence["proxy_final_curve"] = proxy
    assert decide_p1(evidence)["status"] == "FILE_PROXY_PARITY_FAILURE"


def test_official_target_and_gap_are_mandatory() -> None:
    evidence = passing_evidence()
    final = list(evidence["official_final_curve"])
    final[14] = final[15] - 0.01
    evidence["official_final_curve"] = final
    evidence["proxy_final_curve"] = final
    assert decide_p1(evidence)["status"] == "OFFICIAL_OFFSET_FAILURE"


def test_p2_requires_three_joint_wins_and_positive_medians() -> None:
    records = [copy.deepcopy(passing_evidence()) for _ in range(4)]
    decision = decide_p2(records)
    assert decision["pass"]
    assert decision["joint_wins"] == 4
    records[0]["official_final_curve"] = records[0]["official_step0_curve"]
    records[0]["proxy_final_curve"] = records[0]["official_step0_curve"]
    assert decide_p2(records)["pass"]
    records[1]["official_final_curve"] = records[1]["official_step0_curve"]
    records[1]["proxy_final_curve"] = records[1]["official_step0_curve"]
    assert not decide_p2(records)["pass"]
