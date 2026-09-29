"""Contracts that let compact visual evidence retain the original metric."""
import numpy as np
from scipy.special import logsumexp

from scripts.experiments.check_tts_independent_visual import independent_ctc
from scripts.experiments.tts_level_visual_calibration_20260927 import cluster
from scripts.experiments.vsr_tts_metrics import ctc_nll


def test_selected_original_logp_preserves_repeated_token_ctc():
    logits = np.random.default_rng(21).normal(size=(19, 41))
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    columns = np.array([0, 7, 12, 19, 31])
    mapping = {int(token): i for i, token in enumerate(columns)}
    target = [7, 12, 12, 31, 19]
    selected_target = [mapping[token] for token in target]
    expected = ctc_nll(logp, target) / len(target)
    assert abs(ctc_nll(logp[:, columns], selected_target) / len(target) - expected) < 1e-10
    assert abs(independent_ctc(logp[:, columns], selected_target) - expected) < 1e-10
    # Retained probability mass is below one: normalization would change the endpoint.
    normalized = logp[:, columns] - logsumexp(logp[:, columns], axis=1, keepdims=True)
    assert abs(ctc_nll(normalized, selected_target) / len(target) - expected) > 1


def test_speaker_weighting_is_not_utterance_weighting():
    result = cluster([2, 2, 2, 10], ['many', 'many', 'many', 'single'])
    assert result['speaker_mean'] == 6
    assert result['per_speaker'] == {'many': 2, 'single': 10}
    assert result['speakers'] == 2


def test_bootstrap_is_invariant_to_utterance_order():
    x = [1., 3., -2., 4., 6.]
    names = ['b', 'b', 'a', 'c', 'c']
    assert cluster(x, names) == cluster(x[::-1], names[::-1])
