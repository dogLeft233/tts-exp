import pytest

from scripts.experiments.asr_sync_error_correlation.word_errors import (
    align_words,
    error_alignment,
    normalize_text,
    parse_lrs3_txt,
    validate_official_timing,
)


def spans(words):
    return [{"start_s": i * 0.5, "end_s": i * 0.5 + 0.4} for i, _ in enumerate(words)]


def test_normalizer_and_fixture_parser():
    assert normalize_text("  don't—stop  ") == "DON'T STOP"
    parsed = parse_lrs3_txt("Text:  DON'T STOP\nConf: 4\nWORD START END ASDSCORE\nDON'T 0.0 0.4 4.5\nSTOP 0.5 0.9 5.0\n")
    validate_official_timing(parsed, "DON'T STOP")
    assert parsed["words"][1]["end_s"] == 0.9


def test_lrs3_nonlexical_and_numeric_rows_are_not_words():
    text = "Text:  QUESTIONS ST\nWORD START END\n3 0.0 0.2\nQUESTIONS 0.2 0.5\n{LG} 0.5 0.7\n1ST 0.7 0.9\n"
    parsed = parse_lrs3_txt(text)
    validate_official_timing(parsed, "3 QUESTIONS {LG} 1ST")
    assert [row["word"] for row in parsed["words"]] == ["QUESTIONS", "ST"]
    assert [row["raw_word"] for row in parsed["ignored_words"]] == ["3", "{LG}"]
    assert normalize_text("3 QUESTIONS {LG} 1ST") == "QUESTIONS ST"


def test_edit_tie_order_and_substitution_span():
    ops = align_words(["A", "B"], ["A", "C"])
    assert [op["operation"] for op in ops] == ["equal", "substitution"]
    result = error_alignment(["A", "B"], ["A", "C"], spans(["A", "B"]), spans(["A", "C"]))
    assert result["counts"] == {"substitutions": 1, "deletions": 0, "insertions": 0}
    assert result["error_spans"][0]["start_s"] == 0.5


def test_insert_delete_and_equal_blocks():
    result = error_alignment(["A", "B", "C"], ["A", "X", "C"], spans(["A", "B", "C"]), spans(["A", "X", "C"]))
    assert len(result["error_spans"]) == 1
    exact = error_alignment(["A"], ["A"], spans(["A"]), spans(["A"]))
    assert exact["error_spans"] == [] and exact["word_error_rate"] == 0
    with pytest.raises(ValueError):
        error_alignment(["A"], ["X"], [None], spans(["X"]))
