from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/experiments/source_axis_mfa_20260930.py"
SPEC = importlib.util.spec_from_file_location("source_axis_mfa_20260930_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mfa = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mfa
SPEC.loader.exec_module(mfa)


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens

    def parse_phones(self, _path):
        return self.tokens


def _token(label, start, end, silence=False):
    return {"token": label, "start_s": start, "end_s": end, "duration_s": end - start, "is_silence": silence}


def test_phone_classes_match_visual_protocol_and_strip_frozen_marks():
    visual, _ = mfa.load_visual_phone_tools()
    assert mfa.classify_phone("pʰ", visual) == "bilabial"
    assert mfa.classify_phone("ɑː", visual) == "open_vowel"
    assert mfa.classify_phone("i", visual) is None


def test_textgrid_quality_uses_audio_clock_and_exact_visual_phone_classes(tmp_path):
    visual, _ = mfa.load_visual_phone_tools()
    grid = tmp_path / "x.TextGrid"
    grid.write_text('File type = "ooTextFile"\nxmax = 1.0\n', encoding="utf-8")
    tokens = [
        _token("sil", 0.0, 0.1, True),
        _token("ɑː", 0.1, 0.5),
        _token("pʰ", 0.5, 0.9),
        _token("sil", 0.9, 1.0, True),
    ]
    result = mfa.check_condition_grid(grid, 1.0, Parser(tokens), visual)
    assert result["qualified"] is True
    assert result["open_vowel_phone_count"] == 1
    assert result["bilabial_phone_count"] == 1


def test_spn_is_rejected_even_when_parser_marks_it_as_silence(tmp_path):
    visual, _ = mfa.load_visual_phone_tools()
    grid = tmp_path / "x.TextGrid"
    grid.write_text('File type = "ooTextFile"\nxmax = 1.0\n', encoding="utf-8")
    tokens = [_token("spn", 0.0, 1.0, True)]
    result = mfa.check_condition_grid(grid, 1.0, Parser(tokens), visual)
    assert result["qualified"] is False
    assert "unknown_or_oov_phone" in result["errors"]


def test_textgrid_must_cover_the_audio_clock(tmp_path):
    visual, _ = mfa.load_visual_phone_tools()
    grid = tmp_path / "x.TextGrid"
    grid.write_text('File type = "ooTextFile"\nxmax = 0.8\n', encoding="utf-8")
    result = mfa.check_condition_grid(grid, 1.0, Parser([_token("p", 0.1, 0.8)]), visual)
    assert result["qualified"] is False
    assert "textgrid_clock_mismatch" in result["errors"]
    assert "audio_start_not_covered" in result["errors"]
    assert "audio_end_not_covered" in result["errors"]
