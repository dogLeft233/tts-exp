"""Unit tests for scripts/14_prepare_mandarin_alignment.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tfg_feature_common import TokenSpan

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "14_prepare_mandarin_alignment.py"
_spec = importlib.util.spec_from_file_location("_prep_alignment", str(_SCRIPT_PATH))
_prep_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep_alignment)

_split_pinyin_syllable = _prep_alignment._split_pinyin_syllable
_clean_text = _prep_alignment._clean_text
text_to_pinyin_tokens = _prep_alignment.text_to_pinyin_tokens
load_viseme_map = _prep_alignment.load_viseme_map
pinyin_to_viseme = _prep_alignment.pinyin_to_viseme
parse_ctm_file = _prep_alignment.parse_ctm_file
_mfa_available = _prep_alignment._mfa_available
_discover_audio = _prep_alignment._discover_audio
build_manifest = _prep_alignment.build_manifest
VISEME_MAP_PATH = _prep_alignment.VISEME_MAP_PATH
PROJECT_ROOT = _prep_alignment.PROJECT_ROOT

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None
    Style = None


# ============================================================================
# Pinyin conversion tests
# ============================================================================


class TestPinyinConversion:
    """Tests for Chinese text -> pinyin conversion."""

    @pytest.mark.skipif(pinyin is None, reason="pypinyin not installed")
    def test_pinyin_conversion(self):
        tokens = text_to_pinyin_tokens("你好世界")
        assert len(tokens) == 4

        char_0, initial_0, final_0, tone_0 = tokens[0]
        assert char_0 == "你"
        assert initial_0 == "n"
        assert final_0 == "i"
        assert tone_0 == 3

        char_1, initial_1, final_1, tone_1 = tokens[1]
        assert char_1 == "好"
        assert initial_1 == "h"
        assert final_1 == "ao"
        assert tone_1 == 3

    @pytest.mark.skipif(pinyin is None, reason="pypinyin not installed")
    def test_pinyin_with_punctuation(self):
        tokens = text_to_pinyin_tokens("你好，世界！")
        assert len(tokens) == 4

    @pytest.mark.skipif(pinyin is None, reason="pypinyin not installed")
    def test_empty_string(self):
        tokens = text_to_pinyin_tokens("")
        assert tokens == []

    def test_split_apical_vowel(self):
        init, final, tone = _split_pinyin_syllable("shi4")
        assert init == "sh"
        assert final == "-i"
        assert tone == 4

    def test_split_zero_initial(self):
        init, final, tone = _split_pinyin_syllable("a1")
        assert init == "∅"
        assert final == "a"
        assert tone == 1

    def test_split_neutral_tone(self):
        init, final, tone = _split_pinyin_syllable("ma")
        assert init == "m"
        assert final == "a"
        assert tone == 5


class TestCleanText:
    """Tests for Chinese text cleaning."""

    def test_strips_punctuation(self):
        assert _clean_text("你好，世界！") == "你好世界"

    def test_strips_whitespace(self):
        assert _clean_text("  你好 世界  ") == "你好世界"

    def test_keeps_only_chinese(self):
        assert _clean_text("abc你好123") == "你好"

    def test_empty_returns_empty(self):
        assert _clean_text("") == ""


# ============================================================================
# Viseme map tests
# ============================================================================


class TestVisemeMap:
    """Tests for mandarin_viseme_map.yaml loading and classification."""

    def test_viseme_map_separates_bilabial_and_nonlabial(self):
        vm = load_viseme_map()
        init_map = vm["initial_to_viseme"]
        assert init_map["b"] == "bilabial"
        assert init_map["t"] == "nonlabial_consonant"

    def test_viseme_map_rounds_vowel(self):
        vm = load_viseme_map()
        final_map = vm["final_to_viseme"]
        assert final_map["u"] == "rounded_vowel"
        assert final_map["a"] == "open_vowel"

    def test_empty_initial_maps_correctly(self):
        vm = load_viseme_map()
        init_map = vm["initial_to_viseme"]
        assert "∅" in init_map
        assert init_map["∅"] in (
            "bilabial",
            "labiodental",
            "rounded_vowel",
            "open_vowel",
            "front_vowel",
            "nonlabial_consonant",
            "silence",
        )

    def test_viseme_map_loads_all_initials(self):
        vm = load_viseme_map()
        init_map = vm["initial_to_viseme"]
        expected = {
            "b", "p", "m", "f",
            "d", "t", "n", "l",
            "g", "k", "h",
            "j", "q", "x",
            "zh", "ch", "sh", "r",
            "z", "c", "s",
            "y", "w",
            "∅",
        }
        assert set(init_map.keys()) == expected
        assert len(init_map) == 24

    def test_viseme_map_loads_all_finals(self):
        vm = load_viseme_map()
        final_map = vm["final_to_viseme"]
        assert len(final_map) >= 36
        expected = {
            "a", "o", "e", "i", "u", "ü",
            "ai", "ei", "ao", "ou",
            "an", "en", "ang", "eng", "ong", "er",
            "ia", "ie", "iao", "iu",
            "ian", "in", "iang", "ing", "iong",
            "ua", "uo", "uai", "ui",
            "uan", "un", "uang", "ueng",
            "üe", "üan", "ün",
            "-i", "-e",
        }
        assert set(final_map.keys()) == expected
        assert len(final_map) == 38

    def test_all_viseme_labels_are_valid(self):
        vm = load_viseme_map()
        valid = set(vm["viseme_categories"])
        for cat in vm["initial_to_viseme"].values():
            assert cat in valid
        for cat in vm["final_to_viseme"].values():
            assert cat in valid

    def test_pinyin_to_viseme_prefers_initial(self):
        assert pinyin_to_viseme("b", "u") == "bilabial"

    def test_pinyin_to_viseme_zero_initial(self):
        assert pinyin_to_viseme("∅", "a") == "open_vowel"

    def test_pinyin_to_viseme_nonlabial_initial(self):
        assert pinyin_to_viseme("t", "a") == "nonlabial_consonant"


# ============================================================================
# CTM parsing tests
# ============================================================================


class TestCtmParsing:
    """Tests for MFA CTM output parsing."""

    def _write_ctm(self, tmpdir: Path, lines: list[str]) -> Path:
        ctm = tmpdir / "test.ctm"
        ctm.write_text("\n".join(lines), encoding="utf-8")
        return ctm

    def test_ctm_parse_skips_short_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctm = self._write_ctm(tmp, [
                "utt_1 1 0.0 0.01 short",
                "utt_1 1 0.02 0.10 good",
                "utt_1 1 0.13 0.039 barely_short",
                "utt_1 1 0.20 0.04 ok",
            ])
            tokens, stats = parse_ctm_file(ctm, min_duration_s=0.04)
            assert stats["total"] == 4
            assert stats["short_dropped"] == 2
            assert stats["kept"] == 2

    def test_ctm_parse_skips_low_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctm = self._write_ctm(tmp, [
                "utt_1 1 0.0 0.10 token_conf_0.5",
                "utt_1 1 0.2 0.10 token_conf_0.95",
                "utt_1 1 0.4 0.10 token_conf_0.79",
                "utt_1 1 0.6 0.10 token_conf_0.8",
            ])
            tokens, stats = parse_ctm_file(ctm, min_confidence=0.8, min_duration_s=0.0)
            assert stats["confidence_dropped"] == 2
            assert stats["kept"] >= 2

    def test_ctm_parse_empty_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctm = self._write_ctm(tmp, [])
            tokens, stats = parse_ctm_file(ctm)
            assert tokens == []
            assert stats["total"] == 0

    def test_ctm_parse_nonexistent_file(self):
        tokens, stats = parse_ctm_file(Path("/nonexistent/path.ctm"))
        assert tokens == []
        assert stats["total"] == 0

    def test_ctm_parse_missing_confidence_field(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctm = self._write_ctm(tmp, [
                "utt_1 1 0.0 0.10 token_no_conf",
            ])
            tokens, stats = parse_ctm_file(ctm, min_confidence=0.8, min_duration_s=0.0)
            assert stats["kept"] == 1
            assert tokens[0]["confidence"] == 1.0


# ============================================================================
# Integration / smoke tests
# ============================================================================


class TestBuildManifestCheckOnly:
    """Smoke test for build_manifest in check-only mode."""

    def test_check_only_returns_empty(self):
        result = build_manifest(
            sample_ids=[1],
            audio_dir=Path("data/data/audio"),
            transcript_dir=Path("data/data/transcript"),
            output_dir=Path("/tmp/test_output"),
            check_only=True,
        )
        assert result == []


class TestLoadVisemeMap:
    """Verify the YAML file loads and has correct structure."""

    def test_viseme_map_file_exists(self):
        assert VISEME_MAP_PATH.exists()

    def test_viseme_map_has_required_keys(self):
        vm = load_viseme_map()
        assert "viseme_categories" in vm
        assert "initial_to_viseme" in vm
        assert "final_to_viseme" in vm

    def test_viseme_categories_count(self):
        vm = load_viseme_map()
        assert len(vm["viseme_categories"]) == 7


# ============================================================================
# Source audio discovery tests
# ============================================================================


class TestDiscoverAudio:
    """Tests for _discover_audio."""

    def test_natural_audio_found(self):
        audio_dir = Path(__file__).resolve().parent.parent / "data" / "data" / "audio"
        if (audio_dir / "1.wav").exists():
            path = _discover_audio(1, "natural", audio_dir)
            assert path is not None
            assert path.name == "1.wav"

    def test_natural_audio_not_found(self):
        path = _discover_audio(999, "natural", Path("/tmp/nonexistent"))
        assert path is None

    def test_unknown_condition(self):
        path = _discover_audio(1, "nonexistent_tts", Path("/tmp"))
        assert path is None


# ============================================================================
# TokenSpan dataclass integration
# ============================================================================


class TestTokenSpanIntegration:
    """Verify TokenSpan from tfg_feature_common is compatible."""

    def test_token_span_roundtrip(self):
        span = TokenSpan(
            token="组",
            initial="z",
            final="u",
            tone=3,
            start_s=0.12,
            end_s=0.34,
            confidence=0.95,
            viseme="nonlabial_consonant",
        )
        d = {
            "token": span.token,
            "initial": span.initial,
            "final": span.final,
            "tone": span.tone,
            "start_s": span.start_s,
            "end_s": span.end_s,
            "confidence": span.confidence,
            "viseme": span.viseme,
        }
        assert d["token"] == "组"
        assert d["initial"] == "z"
        assert d["final"] == "u"
        assert d["tone"] == 3
        assert d["viseme"] == "nonlabial_consonant"


# ============================================================================
# MFA availability
# ============================================================================


class TestMfaAvailability:
    """Tests for MFA binary presence check."""

    def test_mfa_available_returns_bool(self):
        result = _mfa_available()
        assert isinstance(result, bool)
