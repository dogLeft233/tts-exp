from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_delay_search_support import config
from scripts.experiments.wav2lip_delay_search_support.analysis import decision_flags, legacy_distance_matrix, matched_distance_matrix, metrics_from_matrix
from scripts.experiments.wav2lip_delay_search_support.common import ProtocolError


def _shifted_embeddings(lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    visual = rng.normal(size=(config.EMBEDDING_ROWS, config.EMBEDDING_DIM)).astype(np.float32)
    natural = rng.normal(size=visual.shape).astype(np.float32)
    for row in range(config.EMBEDDING_ROWS):
        q = row + lag
        if 0 <= q < config.EMBEDDING_ROWS:
            natural[q] = visual[row]
    delayed = np.zeros_like(natural)
    delayed[config.DELAY_FRAMES :] = natural[: -config.DELAY_FRAMES]
    return visual, natural, delayed


def test_expected_column_32_is_outside_legacy_domain_but_matched_domain_recovers_minus_five():
    visual, natural, delayed = _shifted_embeddings(12)
    natural_matrix = matched_distance_matrix(visual, natural, config.NATURAL_LAG_START)
    delay_matrix = matched_distance_matrix(visual, delayed, config.DELAY_LAG_START)
    natural_metrics = metrics_from_matrix(natural_matrix, offset_base=15)
    delay_metrics = metrics_from_matrix(delay_matrix, offset_base=10)
    assert natural_metrics["min_index"] == 27
    assert natural_metrics["min_index"] + config.DELAY_FRAMES == 32
    assert delay_metrics["min_index"] == 27
    assert delay_metrics["offset"] - natural_metrics["offset"] == -5


def test_internal_peak_is_preserved_by_legacy_and_matched_domains():
    visual, natural, delayed = _shifted_embeddings(0)
    old = legacy_distance_matrix(visual, natural)
    matched = matched_distance_matrix(visual, natural, config.NATURAL_LAG_START)
    assert metrics_from_matrix(old[np.asarray(config.U_ROWS)], offset_base=15)["min_index"] == 15
    assert metrics_from_matrix(matched, offset_base=15)["min_index"] == 15
    assert np.isfinite(matched).all()
    assert delayed.shape == natural.shape


def test_matched_domain_rejects_unavailable_support_instead_of_padding():
    visual, natural, _ = _shifted_embeddings(0)
    with pytest.raises(ProtocolError):
        matched_distance_matrix(visual, natural, -31)


def test_terminal_flags_separate_fourteen_pass_from_complete_boundary_explanation():
    anomaly = set(config.EXPECTED_ANOMALIES)
    partial = decision_flags(anchor_pass=True, matched_pass_count=14, old_failure_ids=anomaly, expected_failure_ids=anomaly, recovered_ids={next(iter(anomaly))})
    assert partial["corrected_control_pass"] is True
    assert partial["boundary_explanation_complete"] is False
    assert partial["terminal_decision"] == "CONTROL_UNRESOLVED"
    complete = decision_flags(anchor_pass=True, matched_pass_count=16, old_failure_ids=anomaly, expected_failure_ids=anomaly, recovered_ids=anomaly)
    assert complete["corrected_control_pass"] is True
    assert complete["boundary_explanation_complete"] is True
    assert complete["content_probe_revision_eligible"] is True
