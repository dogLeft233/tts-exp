from scripts.analyze_aishell1_phone_duration_gap import _correlation, _matched_pairs, _rank


def token(label, start, end, silence=False):
    return {
        "token": label,
        "start_s": start,
        "end_s": end,
        "is_silence": silence,
    }


def test_matched_pairs_excludes_silence_from_phone_error():
    natural = [token("sil", 0.0, 0.1, True), token("a", 0.1, 0.3)]
    tts = [token("sil", 0.0, 0.05, True), token("a", 0.05, 0.35)]

    pairs, stats = _matched_pairs(natural, tts)

    assert len(pairs) == 1
    assert pairs[0]["delta_s"] == 0.1
    assert stats["matched_silence_count"] == 1
    assert stats["matched_speech_phone_count"] == 1
    assert stats["speech_match_rate"] == 1.0


def test_matched_pairs_reports_label_mismatch_without_fabricating_duration_pair():
    natural = [token("a", 0.0, 0.2), token("b", 0.2, 0.4)]
    tts = [token("a", 0.0, 0.3), token("x", 0.3, 0.6)]

    pairs, stats = _matched_pairs(natural, tts)

    assert len(pairs) == 1
    assert pairs[0]["phone"] == "a"
    assert stats["unmatched_natural_token_count"] == 1
    assert stats["unmatched_tts_token_count"] == 1
    assert stats["speech_match_rate"] == 0.5


def test_rank_and_correlation_are_deterministic():
    assert _rank([2.0, 1.0, 3.0]) == [2.0, 1.0, 3.0]
    assert _correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
