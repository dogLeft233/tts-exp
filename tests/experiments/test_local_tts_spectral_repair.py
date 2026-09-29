import numpy as np
import pytest

from scripts.experiments.local_tts_spectral_repair import (
    cluster_interval,
    local_mask,
    matching_token_pairs,
    natural_phase_blend,
    render_audio,
)


def test_matching_tokens_does_not_bridge_an_unmatched_phone():
    natural = [
        {"token": "a", "is_silence": False, "is_unknown_speech": False},
        {"token": "x", "is_silence": False, "is_unknown_speech": False},
        {"token": "b", "is_silence": False, "is_unknown_speech": False},
    ]
    tts = [
        {"token": "a", "is_silence": False, "is_unknown_speech": False},
        {"token": "b", "is_silence": False, "is_unknown_speech": False},
    ]
    assert matching_token_pairs(natural, tts) == {0: 0, 2: 1}


def test_local_edit_preserves_every_sample_outside_the_window():
    samples = np.arange(16000, dtype=np.int32)
    natural = np.rint(1000 * np.sin(2 * np.pi * samples / 40)).astype(np.int16)
    donor = np.rint(1100 * np.sin(2 * np.pi * samples / 50)).astype(np.int16)
    blended = natural_phase_blend(natural, donor)
    region = {"start_s": 0.3, "end_s": 0.7}
    output, detail = render_audio(natural, blended, region)
    outside = local_mask(len(natural), region) == 0
    assert np.array_equal(output[outside], natural[outside])
    assert len(output) == len(natural)
    assert detail["edited_samples"] > 0
    assert np.max(np.abs(output.astype(np.int32))) < 32768


def test_no_selected_window_is_exact_identity():
    natural = np.array([0, 100, -100, 1], dtype=np.int16)
    output, detail = render_audio(natural, np.zeros(4), None)
    assert np.array_equal(output, natural)
    assert detail["edited_samples"] == 0


def test_phase_blend_rejects_different_sample_clocks():
    with pytest.raises(ValueError, match="sample clocks"):
        natural_phase_blend(np.zeros(100, dtype=np.int16), np.zeros(101, dtype=np.int16))


def test_cluster_interval_distinguishes_speaker_and_utterance_weighting():
    rows = [{"speaker": speaker, "cells": {"N": {"sync_c": 0}, "target": {"sync_c": delta}}}
            for speaker, delta in [("A", 1), ("A", 1), ("B", -1)]]
    equal = cluster_interval(rows, "sync_c")
    primary = cluster_interval(rows, "sync_c", weighting="utterance")
    assert equal["mean"] == 0
    assert primary["mean"] == pytest.approx(1 / 3)
    assert primary["weighting"] == "utterance"
    assert primary["speaker_count"] == 2
    assert primary == cluster_interval(rows, "sync_c", weighting="utterance")
    with pytest.raises(ValueError, match="unknown weighting"):
        cluster_interval(rows, "sync_c", weighting="invalid")
