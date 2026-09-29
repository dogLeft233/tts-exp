from __future__ import annotations

from scripts.experiments.lrs3_wav2lip_timing_transfer import config
from scripts.experiments.lrs3_wav2lip_timing_transfer.analysis import (
    decision_from_counts,
)


def _all(value: int) -> dict[str, int]:
    return {name: value for name in ("A", "B", "C", "O")}


def test_decision_tree_keeps_17_18_and_19_20_boundaries() -> None:
    assert decision_from_counts(21, 22, 22, _all(22)) == "REPEATABILITY_FAILED"
    assert decision_from_counts(22, 19, 22, _all(22)) == "BASELINE_INCONCLUSIVE"
    assert decision_from_counts(22, 20, 20, {**_all(22), "A": 17}) == "AUDIO_CONTROL_UNRESOLVED"
    assert decision_from_counts(22, 20, 20, {**_all(22), "A": 18, "B": 17}) == "GENERATED_ENDPOINT_UNRESOLVED"
    assert decision_from_counts(22, 20, 20, {**_all(22), "A": 18, "B": 18, "C": 17}) == "GENERATED_RESPONSE_UNRESOLVED"
    assert decision_from_counts(22, 20, 20, _all(18)) == "LOCAL_RESPONSE_ESTABLISHED"
    assert config.MIN_SUCCESS_RECORDS == 18
    assert config.MIN_BASELINE_RECORDS == 20
